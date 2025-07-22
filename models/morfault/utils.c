#include "morfault.h"
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

// ---------------------------------------
// SetInitialConditions
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "SetInitialConditions"
PetscErrorCode SetInitialConditions(FDPDE fdPV, FDPDE fdT, FDPDE fdphi, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  DM             dmTcoeff, dmphicoeff;
  Vec            xPV, xguess, xTprev, xTguess, xTcoeff, xTcoeffprev, xphiprev, xphiguess, xphicoeff, xphicoeffprev;
  PetscFunctionBeginUser;

  // initialize T: half-space cooling model
  PetscCall(HalfSpaceCooling_MOR(usr));

  // initialize constant solid porosity field
  PetscCall(VecSet(usr->xphi,1.0-usr->par->phi0)); 
  if (usr->par->model_setup_phi==0) {
    PetscCall(SetInitialPorosityField(usr));
  }
  
  // set swarm initial size and coordinates
  PetscInt ppcell[] = {usr->par->ppcell,usr->par->ppcell};
  PetscCall(MPointCoordLayout_DomainVolumeWithCellList(usr->dmswarm,0,NULL,0.5,ppcell,COOR_INITIALIZE));

  // swarm initial condition - output only id field
  PetscCall(SetSwarmInitialCondition(usr->dmswarm,usr));

  // Update marker phase fractions on the dmstag 
  PetscCall(UpdateMarkerPhaseFractions(usr->dmswarm,usr->dmMPhase,usr->xMPhase,usr));

  // Update lithostatic pressure
  PetscCall(UpdateLithostaticPressure(usr->dmPlith,usr->xPlith,usr));

  // Create initial guess for PV - viscous solution
  usr->plasticity = PETSC_FALSE; 
  PetscPrintf(PETSC_COMM_WORLD,"# (PV) Rheology: VISCO-ELASTIC \n");
  usr->nd->dt = usr->nd->dtmax;
  PetscCall(FDPDESolve(fdPV,NULL));
  PetscCall(FDPDEGetSolution(fdPV,&xPV));
  PetscCall(VecCopy(xPV,usr->xPV));
  PetscCall(FDPDEGetSolutionGuess(fdPV,&xguess));  
  PetscCall(VecCopy(usr->xPV,xguess));
  PetscCall(VecDestroy(&xguess));
  PetscCall(VecDestroy(&xPV));
  usr->plasticity = PETSC_TRUE; 

  // Update fluid velocity to zero and v=vs
  PetscCall(ComputeFluidAndBulkVelocity(usr->dmPV,usr->xPV,usr->dmPlith,usr->xPlith,usr->dmphi,usr->xphi,usr->dmVel,usr->xVel,usr));

  // Initialize guess and previous solution in fdT
  PetscCall(FDPDEAdvDiffGetPrevSolution(fdT,&xTprev));
  PetscCall(VecCopy(usr->xT,xTprev));
  PetscCall(FDPDEGetSolutionGuess(fdT,&xTguess));
  PetscCall(VecCopy(xTprev,xTguess));
  PetscCall(VecDestroy(&xTprev));
  PetscCall(VecDestroy(&xTguess));

  PetscCall(FDPDEGetCoefficient(fdT,&dmTcoeff,&xTcoeff));
  PetscCall(FDPDEAdvDiffGetPrevCoefficient(fdT,&xTcoeffprev));
  PetscCall(FormCoefficient_T(fdT,usr->dmT,usr->xT,dmTcoeff,xTcoeffprev,usr));
  PetscCall(VecCopy(xTcoeffprev,xTcoeff));
  PetscCall(VecDestroy(&xTcoeffprev));

  // Initialize guess and previous solution in fdphi
  PetscCall(FDPDEAdvDiffGetPrevSolution(fdphi,&xphiprev));
  PetscCall(VecCopy(usr->xphi,xphiprev));
  PetscCall(FDPDEGetSolutionGuess(fdphi,&xphiguess));
  PetscCall(VecCopy(xphiprev,xphiguess));
  PetscCall(VecDestroy(&xphiprev));
  PetscCall(VecDestroy(&xphiguess));

  PetscCall(FDPDEGetCoefficient(fdphi,&dmphicoeff,&xphicoeff));
  PetscCall(FDPDEAdvDiffGetPrevCoefficient(fdphi,&xphicoeffprev));
  PetscCall(FormCoefficient_phi(fdphi,usr->dmphi,usr->xphi,dmphicoeff,xphicoeffprev,usr));
  PetscCall(VecCopy(xphicoeffprev,xphicoeff));
  PetscCall(VecDestroy(&xphicoeffprev));

  // Output initial conditions
  PetscCall(DoOutput(fdPV,fdT,fdphi,usr));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// Half-Space Cooling model
// ---------------------------------------
PetscErrorCode HalfSpaceCooling_MOR(void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz, iT, icenter;
  PetscScalar    **coordx,**coordz, ***xx, Tm, Ts, xmor, Hs, Tiso, ziso;
  Vec            x, xlocal;
  DM             dm;
  PetscFunctionBeginUser;

  dm  = usr->dmT;
  x   = usr->xT;
  Ts   = usr->par->Ttop;
  Tm   = usr->par->Tbot;
  xmor = usr->nd->xmin+usr->nd->L/2.0;
  Hs   = usr->par->Hs;
  Tiso = 1200+T_KELVIN; // deg C

  PetscCall(DMStagGetGlobalSizes(dm, &Nx, &Nz,NULL));
  PetscCall(DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 
  PetscCall(DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter)); 
  PetscCall(DMStagGetLocationSlot(dm, ELEMENT, 0, &iT)); 

  PetscCall(DMCreateLocalVector(dm, &xlocal)); 
  PetscCall(DMStagVecGetArray(dm, xlocal, &xx)); 
  
  ziso = usr->nd->zmin;
  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      PetscScalar age, T, nd_T;
      
      if (usr->par->model_setup==0) age = usr->par->age*1.0e6*SEC_YEAR; // constant age in Myr
      else age  = usr->par->age*1.0e6*SEC_YEAR + dim_param(fabs(coordx[i][icenter]),usr->scal->x)/dim_param(usr->nd->uT,usr->scal->v); // age varying with distance from axis + initial age

      // half-space cooling temperature - take into account free surface
      T = HalfSpaceCoolingTemp(Tm,Ts,-Hs-dim_param(coordz[j][icenter],usr->scal->x),usr->scal->kappa,age,usr->par->hs_factor); 
      if (T-Ts<0.0) T = Ts;

      // save depth to isotherm
      if ((i==(int)nx*0.5) & (T>=Tiso)) ziso = PetscMax(ziso,coordz[j][icenter]);

      // nd_T = (T - usr->par->T0)/usr->par->DT;
      nd_T = nd_paramT(T,Ts,usr->scal->DT);

      // temperature (phi=0)
      xx[j][i][iT] = nd_T;
    }
  }
  if (usr->nd->z_bc==0.0) {
    usr->nd->z_bc = ziso;
  }

  // Restore arrays
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));
  PetscCall(DMStagVecRestoreArray(dm,xlocal,&xx)); 
  PetscCall(DMLocalToGlobalBegin(dm,xlocal,INSERT_VALUES,x)); 
  PetscCall(DMLocalToGlobalEnd  (dm,xlocal,INSERT_VALUES,x)); 
  PetscCall(VecDestroy(&xlocal)); 

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// SetInitialPorosityField
// ---------------------------------------
PetscErrorCode SetInitialPorosityField(void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz, iE, icenter;
  PetscScalar    **coordx,**coordz, ***xx;
  PetscScalar    xc, zc, phi_max,sigma,sigma_v;
  Vec            x, xlocal;
  DM             dm;
  PetscFunctionBeginUser;

  dm  = usr->dmphi;
  x   = usr->xphi;

  phi_max = usr->par->phi_max_bc; // 1e-3;
  sigma   = usr->par->sigma_bc;   // 0.1 - 0.001;
  sigma_v = usr->par->sigma_bc_h; 

  xc = 0.0;
  // zc = usr->nd->zmin+usr->nd->H*0.2; // default
  zc = usr->nd->z_bc;

  PetscCall(DMStagGetGlobalSizes(dm, &Nx, &Nz,NULL));
  PetscCall(DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 
  PetscCall(DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter)); 
  PetscCall(DMStagGetLocationSlot(dm, ELEMENT, 0, &iE)); 

  PetscCall(DMCreateLocalVector(dm, &xlocal)); 
  PetscCall(DMStagVecGetArray(dm, xlocal, &xx)); 
  
  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      PetscScalar phi, xp, zp;
      xp = coordx[i][icenter] - xc;
      zp = coordz[j][icenter] - zc;
      phi = usr->par->phi0 + phi_max*PetscExpScalar(-xp*xp/sigma - zp*zp/sigma_v);

      xx[j][i][iE] = 1.0-phi; 

    }
  }

  // Restore arrays
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));
  PetscCall(DMStagVecRestoreArray(dm,xlocal,&xx)); 
  PetscCall(DMLocalToGlobalBegin(dm,xlocal,INSERT_VALUES,x)); 
  PetscCall(DMLocalToGlobalEnd  (dm,xlocal,INSERT_VALUES,x)); 
  PetscCall(VecDestroy(&xlocal)); 

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// SetSwarmInitialCondition - can set different lithologies; 
// Default: sticky-air and a rock layer
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "SetSwarmInitialCondition"
PetscErrorCode SetSwarmInitialCondition(DM dmswarm, void *ctx)
{
  UsrData   *usr = (UsrData*)ctx;
  PetscScalar *pcoor,*pfield,*pfield0, *pfield1, *pfield2,*pfield3, *pfield4, *pfield5, ztop;
  PetscInt  npoints,p;
  PetscFunctionBeginUser;

  ztop = usr->nd->zmin+usr->nd->H;
  PetscCall(DMSwarmGetLocalSize(dmswarm,&npoints));
  PetscCall(DMSwarmGetField(dmswarm,DMSwarmPICField_coor,NULL,NULL,(void**)&pcoor));
    
  PetscCall(DMSwarmGetField(dmswarm,"id",NULL,NULL,(void**)&pfield));
  PetscCall(DMSwarmGetField(dmswarm,"id0",NULL,NULL,(void**)&pfield0));
  PetscCall(DMSwarmGetField(dmswarm,"id1",NULL,NULL,(void**)&pfield1));
  PetscCall(DMSwarmGetField(dmswarm,"id2",NULL,NULL,(void**)&pfield2));
  PetscCall(DMSwarmGetField(dmswarm,"id3",NULL,NULL,(void**)&pfield3));
  PetscCall(DMSwarmGetField(dmswarm,"id4",NULL,NULL,(void**)&pfield4));
  PetscCall(DMSwarmGetField(dmswarm,"id5",NULL,NULL,(void**)&pfield5));

  for (p=0; p<npoints; p++) {
    PetscScalar xcoor,zcoor, h, dh;
    
    xcoor = pcoor[2*p+0];
    zcoor = pcoor[2*p+1];

    // dummy fields used for projection
    pfield0[p] = 0;
    pfield1[p] = 0;
    pfield2[p] = 0;
    pfield3[p] = 0;
    pfield4[p] = 0;
    pfield5[p] = 0;

    // default mantle
    pfield[p] = usr->par->mat5_id; 

    // sticky-air
    if (zcoor>=ztop-usr->nd->Hs) pfield[p] = usr->par->mat0_id; 

    // add some layering in the mantle
    // if ((zcoor>=ztop-3.0*usr->nd->Hs) && (zcoor<ztop-2.0*usr->nd->Hs)) pfield[p] = usr->par->mat1_id; 
    // if ((zcoor>=ztop-5.0*usr->nd->Hs) && (zcoor<ztop-4.0*usr->nd->Hs)) pfield[p] = usr->par->mat1_id; 
    // if ((zcoor>=ztop-7.0*usr->nd->Hs) && (zcoor<ztop-6.0*usr->nd->Hs)) pfield[p] = usr->par->mat1_id; 

    // sediments 1km
    h  = usr->nd->Hs;
    dh = nd_param(1e3,usr->scal->x);
    // if ((zcoor>=ztop-h-dh) && (zcoor<ztop-h)) pfield[p] = usr->par->mat1_id; 

    // basalt 2 km
    // h  += dh;
    dh = nd_param(3e3,usr->scal->x);
    if ((zcoor>=ztop-h-dh) && (zcoor<ztop-h)) pfield[p] = usr->par->mat2_id; 

    // gabbro 5 km
    h  += dh;
    dh = nd_param(5e3,usr->scal->x);
    if ((zcoor>=ztop-h-dh) && (zcoor<ztop-h)) pfield[p] = usr->par->mat3_id; 

    // mantle layering of 5 km
    h  += 2.0*dh;
    dh = nd_param(5e3,usr->scal->x);
    if ((zcoor>=ztop-h-dh) && (zcoor<ztop-h)) pfield[p] = usr->par->mat4_id; 

    h  += 2.0*dh;
    dh = nd_param(5e3,usr->scal->x);
    if ((zcoor>=ztop-h-dh) && (zcoor<ztop-h)) pfield[p] = usr->par->mat4_id; 

    h  += 2.0*dh;
    dh = nd_param(5e3,usr->scal->x);
    if ((zcoor>=ztop-h-dh) && (zcoor<ztop-h)) pfield[p] = usr->par->mat4_id; 

    // update binary representation
    if (pfield[p]==0) pfield0[p] = 1;
    if (pfield[p]==1) pfield1[p] = 1;
    if (pfield[p]==2) pfield2[p] = 1;
    if (pfield[p]==3) pfield3[p] = 1;
    if (pfield[p]==4) pfield4[p] = 1;
    if (pfield[p]==5) pfield5[p] = 1;
  }
  PetscCall(DMSwarmRestoreField(dmswarm,"id",NULL,NULL,(void**)&pfield));
  PetscCall(DMSwarmRestoreField(dmswarm,"id0",NULL,NULL,(void**)&pfield0));
  PetscCall(DMSwarmRestoreField(dmswarm,"id1",NULL,NULL,(void**)&pfield1));
  PetscCall(DMSwarmRestoreField(dmswarm,"id2",NULL,NULL,(void**)&pfield2));
  PetscCall(DMSwarmRestoreField(dmswarm,"id3",NULL,NULL,(void**)&pfield3));
  PetscCall(DMSwarmRestoreField(dmswarm,"id4",NULL,NULL,(void**)&pfield4));
  PetscCall(DMSwarmRestoreField(dmswarm,"id5",NULL,NULL,(void**)&pfield5));
  PetscCall(DMSwarmRestoreField(dmswarm,DMSwarmPICField_coor,NULL,NULL,(void**)&pcoor));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// AddMarkerInflux
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "AddMarkerInflux"
PetscErrorCode AddMarkerInflux(DM dmswarm, void *ctx)
{
  UsrData   *usr = (UsrData*)ctx;
  PetscScalar *pcoor,*pfield,*pfield0, *pfield1, *pfield2,*pfield3, *pfield4, *pfield5;
  PetscInt  npoints,p,nmark_in,mx,mz,cnt;
  PetscScalar  dxcell, dzcell, dzin, value;
  PetscRandom    rnd;
  PetscFunctionBeginUser;

  dxcell = usr->nd->L/usr->par->nx/usr->par->ppcell;
  dzcell = usr->nd->H/usr->par->nz/usr->par->ppcell;

  // influx
  usr->nd->dzin += usr->nd->Vin_rock*usr->nd->dt;
  mx = (int)(usr->nd->L/dxcell);
  mz = (int)(usr->nd->dzin/dzcell);
  nmark_in = mx*mz;

  if (nmark_in==0) { PetscFunctionReturn(PETSC_SUCCESS); }
  
  PetscCall(DMSwarmAddNPoints(dmswarm,nmark_in));
  PetscCall(DMSwarmGetLocalSize(dmswarm,&npoints));
  PetscCall(DMSwarmGetField(dmswarm,DMSwarmPICField_coor,NULL,NULL,(void**)&pcoor));
  PetscCall(DMSwarmGetField(dmswarm,"id",NULL,NULL,(void**)&pfield));
  PetscCall(DMSwarmGetField(dmswarm,"id0",NULL,NULL,(void**)&pfield0));
  PetscCall(DMSwarmGetField(dmswarm,"id1",NULL,NULL,(void**)&pfield1));
  PetscCall(DMSwarmGetField(dmswarm,"id2",NULL,NULL,(void**)&pfield2));
  PetscCall(DMSwarmGetField(dmswarm,"id3",NULL,NULL,(void**)&pfield3));
  PetscCall(DMSwarmGetField(dmswarm,"id4",NULL,NULL,(void**)&pfield4));
  PetscCall(DMSwarmGetField(dmswarm,"id5",NULL,NULL,(void**)&pfield5));

  PetscCall(PetscRandomCreate(PETSC_COMM_SELF,&rnd));
  PetscCall(PetscRandomSetInterval(rnd,0.0,dxcell*0.5));
  PetscCall(PetscRandomSetFromOptions(rnd));

  PetscScalar dx0, dz0;
  dx0 = usr->nd->xmin+dxcell*0.5;
  dz0 = usr->nd->zmin+dzcell*0.5;

  for (p=0; p<npoints; p++) {
    PetscScalar xcoor,zcoor;
    
    xcoor = pcoor[2*p+0];
    zcoor = pcoor[2*p+1];
    if ((xcoor==0.0) && (zcoor==0.0)) {
      PetscCall(PetscRandomGetValue(rnd,&value));
      pcoor[2*p+0] = dx0+value;
      PetscCall(PetscRandomGetValue(rnd,&value));
      pcoor[2*p+1] = dz0+value;

      dx0 += dxcell;
      if (dx0>usr->nd->xmin+usr->nd->L) {
        dx0 = usr->nd->xmin+dxcell*0.5;
        dz0 += dzcell;
      }

      pfield[p] = usr->par->matid_default; 
      pfield0[p] = 0;
      pfield1[p] = 0;
      pfield2[p] = 0;
      pfield3[p] = 0;
      pfield4[p] = 0;
      pfield5[p] = 1;
    }
  }

  PetscCall(PetscRandomDestroy(&rnd));
  PetscCall(DMSwarmRestoreField(dmswarm,"id",NULL,NULL,(void**)&pfield));
  PetscCall(DMSwarmRestoreField(dmswarm,"id0",NULL,NULL,(void**)&pfield0));
  PetscCall(DMSwarmRestoreField(dmswarm,"id1",NULL,NULL,(void**)&pfield1));
  PetscCall(DMSwarmRestoreField(dmswarm,"id2",NULL,NULL,(void**)&pfield2));
  PetscCall(DMSwarmRestoreField(dmswarm,"id3",NULL,NULL,(void**)&pfield3));
  PetscCall(DMSwarmRestoreField(dmswarm,"id4",NULL,NULL,(void**)&pfield4));
  PetscCall(DMSwarmRestoreField(dmswarm,"id5",NULL,NULL,(void**)&pfield5));
  PetscCall(DMSwarmRestoreField(dmswarm,DMSwarmPICField_coor,NULL,NULL,(void**)&pcoor));
  PetscCall(DMSwarmMigrate(dmswarm,PETSC_TRUE));

  // reset
  usr->nd->dzin = 0.0;

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// AddMarkerInflux_FreeSurface
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "AddMarkerInflux_FreeSurface"
PetscErrorCode AddMarkerInflux_FreeSurface(DM dmswarm, void *ctx)
{
  UsrData   *usr = (UsrData*)ctx;
  PetscScalar *pcoor,*pfield,*pfield0, *pfield1, *pfield2,*pfield3, *pfield4, *pfield5;
  PetscInt  npoints,p,nmark_in,mx,mz,cnt;
  PetscScalar  dxcell, dzcell, dzin, value;
  PetscRandom    rnd;
  PetscFunctionBeginUser;

  dxcell = usr->nd->L/usr->par->nx/(usr->par->ppcell+1);
  dzcell = usr->nd->H/usr->par->nz/(usr->par->ppcell+1);

  // influx
  usr->nd->dzin_fs += usr->nd->Vin_free*usr->nd->dt;
  mx = (int)(usr->nd->L/dxcell);
  mz = (int)(usr->nd->dzin_fs/dzcell);
  nmark_in = mx*mz;

  if (nmark_in==0) { PetscFunctionReturn(PETSC_SUCCESS); }
  
  PetscCall(DMSwarmAddNPoints(dmswarm,nmark_in)); // inserted at (0,0)
  PetscCall(DMSwarmGetLocalSize(dmswarm,&npoints));
  PetscCall(DMSwarmGetField(dmswarm,DMSwarmPICField_coor,NULL,NULL,(void**)&pcoor));
  PetscCall(DMSwarmGetField(dmswarm,"id",NULL,NULL,(void**)&pfield));
  PetscCall(DMSwarmGetField(dmswarm,"id0",NULL,NULL,(void**)&pfield0));
  PetscCall(DMSwarmGetField(dmswarm,"id1",NULL,NULL,(void**)&pfield1));
  PetscCall(DMSwarmGetField(dmswarm,"id2",NULL,NULL,(void**)&pfield2));
  PetscCall(DMSwarmGetField(dmswarm,"id3",NULL,NULL,(void**)&pfield3));
  PetscCall(DMSwarmGetField(dmswarm,"id4",NULL,NULL,(void**)&pfield4));
  PetscCall(DMSwarmGetField(dmswarm,"id5",NULL,NULL,(void**)&pfield5));

  PetscCall(PetscRandomCreate(PETSC_COMM_SELF,&rnd));
  PetscCall(PetscRandomSetInterval(rnd,0.0,dxcell*0.5));
  PetscCall(PetscRandomSetFromOptions(rnd));

  PetscScalar dx0, dz0;
  dx0 = usr->nd->xmin+dxcell*0.5;
  dz0 = usr->nd->H+usr->nd->zmin-dzcell*0.5;

  for (p=0; p<npoints; p++) {
    PetscScalar xcoor,zcoor;
    
    xcoor = pcoor[2*p+0];
    zcoor = pcoor[2*p+1];
    if ((xcoor==0.0) && (zcoor==0.0)) {
      PetscCall(PetscRandomGetValue(rnd,&value));
      pcoor[2*p+0] = dx0+value;
      PetscCall(PetscRandomGetValue(rnd,&value));
      pcoor[2*p+1] = dz0+value;

      dx0 += dxcell;
      if (dx0>usr->nd->xmin+usr->nd->L) {
        dx0  = usr->nd->xmin+dxcell*0.5;
        dz0 -= dzcell;
      }

      pfield[p] = usr->par->mat0_id; // sticky-water id=0
      pfield0[p] = 1;
      pfield1[p] = 0;
      pfield2[p] = 0;
      pfield3[p] = 0;
      pfield4[p] = 0;
      pfield5[p] = 0;
    }
  }

  PetscCall(PetscRandomDestroy(&rnd));
  PetscCall(DMSwarmRestoreField(dmswarm,"id",NULL,NULL,(void**)&pfield));
  PetscCall(DMSwarmRestoreField(dmswarm,"id0",NULL,NULL,(void**)&pfield0));
  PetscCall(DMSwarmRestoreField(dmswarm,"id1",NULL,NULL,(void**)&pfield1));
  PetscCall(DMSwarmRestoreField(dmswarm,"id2",NULL,NULL,(void**)&pfield2));
  PetscCall(DMSwarmRestoreField(dmswarm,"id3",NULL,NULL,(void**)&pfield3));
  PetscCall(DMSwarmRestoreField(dmswarm,"id4",NULL,NULL,(void**)&pfield4));
  PetscCall(DMSwarmRestoreField(dmswarm,"id5",NULL,NULL,(void**)&pfield5));
  PetscCall(DMSwarmRestoreField(dmswarm,DMSwarmPICField_coor,NULL,NULL,(void**)&pcoor));
  PetscCall(DMSwarmMigrate(dmswarm,PETSC_TRUE));

  // reset
  usr->nd->dzin_fs = 0.0;

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// MarkerControl - not parallel
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "MarkerControl"
PetscErrorCode MarkerControl(DM dmswarm, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       i,j,iE, icenter, sx, sz, nx, nz, p, npoints;
  PetscInt       *pcellid;
  PetscScalar    **coordx, **coordz, ***cnt_sw, ***cnt_nw, ***cnt_se, ***cnt_ne;
  PetscScalar    *pcoor,*pfield,*pfield0, *pfield1, *pfield2,*pfield3, *pfield4, *pfield5;
  DM             dmcell;
  Vec            cnt_sw_local, cnt_nw_local, cnt_se_local, cnt_ne_local;
  const char     *cellid;
  DMSwarmCellDM  celldm;
  PetscFunctionBeginUser;

  // Count nmarker/quarter of cell 
  dmcell = usr->dmphi;
  PetscCall(DMStagGetCorners(dmcell, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 
  PetscCall(DMStagGetProductCoordinateArraysRead(dmcell,&coordx,&coordz,NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmcell,DMSTAG_ELEMENT,&icenter));
  PetscCall(DMStagGetLocationSlot(dmcell,DMSTAG_ELEMENT,0,&iE));

  PetscCall(DMSwarmGetLocalSize(dmswarm,&npoints));
  PetscCall(DMSwarmGetField(dmswarm,DMSwarmPICField_coor,NULL,NULL,(void**)&pcoor));

  PetscCall(DMSwarmGetCellDMActive(dmswarm, &celldm));
  PetscCall(DMSwarmCellDMGetCellID(celldm, &cellid));
  PetscCall(DMSwarmGetField(dmswarm,cellid,NULL,NULL,(void**)&pcellid));

  // local vectors
  PetscCall(DMCreateLocalVector(dmcell,&cnt_sw_local));
  PetscCall(DMCreateLocalVector(dmcell,&cnt_nw_local));
  PetscCall(DMCreateLocalVector(dmcell,&cnt_se_local));
  PetscCall(DMCreateLocalVector(dmcell,&cnt_ne_local));

  PetscCall(DMStagVecGetArray(dmcell,cnt_sw_local,&cnt_sw));
  PetscCall(DMStagVecGetArray(dmcell,cnt_nw_local,&cnt_nw));
  PetscCall(DMStagVecGetArray(dmcell,cnt_se_local,&cnt_se));
  PetscCall(DMStagVecGetArray(dmcell,cnt_ne_local,&cnt_ne));

  // count markers/quarter of cell
  for (p=0; p<npoints; p++) {
    PetscInt cellid = -1;
    PetscInt geid[]={0,0,0};
    PetscScalar xcoor, zcoor, xc, zc;
    
    cellid = pcellid[p];
    PetscCall(DMStagGetLocalElementGlobalIndices(dmcell,cellid,geid));

    xcoor = pcoor[2*p+0];
    zcoor = pcoor[2*p+1];

    xc = coordx[geid[0]][icenter];
    zc = coordz[geid[1]][icenter];

    if ((xcoor<=xc) & (zcoor<=zc)) cnt_sw[geid[1]][geid[0]][iE] += 1.0;
    if ((xcoor<=xc) & (zcoor> zc)) cnt_nw[geid[1]][geid[0]][iE] += 1.0;
    if ((xcoor> xc) & (zcoor<=zc)) cnt_se[geid[1]][geid[0]][iE] += 1.0;
    if ((xcoor> xc) & (zcoor> zc)) cnt_ne[geid[1]][geid[0]][iE] += 1.0;
  }

  PetscCall(DMSwarmRestoreField(dmswarm,cellid,NULL,NULL,(void**)&pcellid));
  PetscCall(DMSwarmRestoreField(dmswarm,DMSwarmPICField_coor,NULL,NULL,(void**)&pcoor));

  // count number markers to insert
  PetscInt nmark_in;
  nmark_in = 0;
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      if (cnt_sw[j][i][iE]==0.0) nmark_in+=1;
      if (cnt_nw[j][i][iE]==0.0) nmark_in+=1; 
      if (cnt_se[j][i][iE]==0.0) nmark_in+=1;
      if (cnt_ne[j][i][iE]==0.0) nmark_in+=1;
    }
  }

  PetscPrintf(PETSC_COMM_WORLD,"# (DMSWARM) Marker control: insert %d markers \n",nmark_in);

  // return if needed
  if (nmark_in==0) { 
    PetscCall(DMStagRestoreProductCoordinateArraysRead(dmcell,&coordx,&coordz,NULL));
    PetscCall(DMStagVecRestoreArray(dmcell,cnt_sw_local,&cnt_sw));
    PetscCall(DMStagVecRestoreArray(dmcell,cnt_nw_local,&cnt_nw));
    PetscCall(DMStagVecRestoreArray(dmcell,cnt_se_local,&cnt_se));
    PetscCall(DMStagVecRestoreArray(dmcell,cnt_ne_local,&cnt_ne));
    
    PetscCall(VecDestroy(&cnt_sw_local));
    PetscCall(VecDestroy(&cnt_nw_local));
    PetscCall(VecDestroy(&cnt_se_local));
    PetscCall(VecDestroy(&cnt_ne_local));

    PetscFunctionReturn(PETSC_SUCCESS); 
  }

  PetscScalar dxcell;
  dxcell = usr->nd->L/usr->par->nx/4;

  // add new markers
  PetscCall(DMSwarmAddNPoints(dmswarm,nmark_in)); // inserted at (0,0)
  PetscCall(DMSwarmGetLocalSize(dmswarm,&npoints));
  PetscCall(DMSwarmGetField(dmswarm,DMSwarmPICField_coor,NULL,NULL,(void**)&pcoor));
  PetscCall(DMSwarmGetField(dmswarm,"id",NULL,NULL,(void**)&pfield));
  PetscCall(DMSwarmGetField(dmswarm,"id0",NULL,NULL,(void**)&pfield0));
  PetscCall(DMSwarmGetField(dmswarm,"id1",NULL,NULL,(void**)&pfield1));
  PetscCall(DMSwarmGetField(dmswarm,"id2",NULL,NULL,(void**)&pfield2));
  PetscCall(DMSwarmGetField(dmswarm,"id3",NULL,NULL,(void**)&pfield3));
  PetscCall(DMSwarmGetField(dmswarm,"id4",NULL,NULL,(void**)&pfield4));
  PetscCall(DMSwarmGetField(dmswarm,"id5",NULL,NULL,(void**)&pfield5));

  // allocate memory arrays for coords
  PetscScalar *pcoorx_new, *pcoorz_new, *pid_new;
  PetscCall(PetscCalloc1(nmark_in,&pcoorx_new));
  PetscCall(PetscCalloc1(nmark_in,&pcoorz_new));
  PetscCall(PetscCalloc1(nmark_in,&pid_new));

  PetscRandom rnd;
  PetscCall(PetscRandomCreate(PETSC_COMM_SELF,&rnd));
  PetscCall(PetscRandomSetInterval(rnd,0.0,dxcell*0.5));
  PetscCall(PetscRandomSetFromOptions(rnd));

  PetscInt ip = 0;

  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      PetscScalar value, dist;

      if (cnt_sw[j][i][iE]==0.0) {
        // generate new coord
        PetscCall(PetscRandomGetValue(rnd,&value));
        pcoorx_new[ip] = coordx[i][icenter]-dxcell+value;
        PetscCall(PetscRandomGetValue(rnd,&value));
        pcoorz_new[ip] = coordz[j][icenter]-dxcell+value;

        // determine phase_id based on closest marker
        dist = usr->nd->L;
        for (p=0; p<npoints; p++) {
          PetscScalar pcoorx,pcoorz, xd, zd,dist_old;
    
          pcoorx = pcoor[2*p+0];
          pcoorz = pcoor[2*p+1];

          xd = pcoorx-pcoorx_new[ip];
          zd = pcoorz-pcoorz_new[ip];

          dist_old = dist;
          dist = PetscMin(dist,PetscPowScalar((xd*xd+zd*zd),0.5));

          if (dist<dist_old) {
            pid_new[ip] = pfield[p];
          }
        }
        ip += 1;
      }

      if (cnt_nw[j][i][iE]==0.0) {
        // generate new coord
        PetscCall(PetscRandomGetValue(rnd,&value));
        pcoorx_new[ip] = coordx[i][icenter]-dxcell+value;
        PetscCall(PetscRandomGetValue(rnd,&value));
        pcoorz_new[ip] = coordz[j][icenter]+dxcell+value;

        // determine phase_id based on closest marker
        dist = usr->nd->L;
        for (p=0; p<npoints; p++) {
          PetscScalar pcoorx,pcoorz, xd, zd,dist_old;
    
          pcoorx = pcoor[2*p+0];
          pcoorz = pcoor[2*p+1];

          xd = pcoorx-pcoorx_new[ip];
          zd = pcoorz-pcoorz_new[ip];

          dist_old = dist;
          dist = PetscMin(dist,PetscPowScalar((xd*xd+zd*zd),0.5));

          if (dist<dist_old) {
            pid_new[ip] = pfield[p];
          }
        }
        ip += 1;
      }

      if (cnt_se[j][i][iE]==0.0) {
        // generate new coord
        PetscCall(PetscRandomGetValue(rnd,&value));
        pcoorx_new[ip] = coordx[i][icenter]+dxcell+value;
        PetscCall(PetscRandomGetValue(rnd,&value));
        pcoorz_new[ip] = coordz[j][icenter]-dxcell+value;

        // determine phase_id based on closest marker
        dist = usr->nd->L;
        for (p=0; p<npoints; p++) {
          PetscScalar pcoorx,pcoorz, xd, zd,dist_old;
    
          pcoorx = pcoor[2*p+0];
          pcoorz = pcoor[2*p+1];

          xd = pcoorx-pcoorx_new[ip];
          zd = pcoorz-pcoorz_new[ip];

          dist_old = dist;
          dist = PetscMin(dist,PetscPowScalar((xd*xd+zd*zd),0.5));

          if (dist<dist_old) {
            pid_new[ip] = pfield[p];
          }
        }
        ip += 1;
      }

      if (cnt_ne[j][i][iE]==0.0) {
        // generate new coord
        PetscCall(PetscRandomGetValue(rnd,&value));
        pcoorx_new[ip] = coordx[i][icenter]+dxcell+value;
        PetscCall(PetscRandomGetValue(rnd,&value));
        pcoorz_new[ip] = coordz[j][icenter]+dxcell+value;

        // determine phase_id based on closest marker
        dist = usr->nd->L;
        for (p=0; p<npoints; p++) {
          PetscScalar pcoorx,pcoorz, xd, zd,dist_old;
    
          pcoorx = pcoor[2*p+0];
          pcoorz = pcoor[2*p+1];

          xd = pcoorx-pcoorx_new[ip];
          zd = pcoorz-pcoorz_new[ip];

          dist_old = dist;
          dist = PetscMin(dist,PetscPowScalar((xd*xd+zd*zd),0.5));

          if (dist<dist_old) {
            pid_new[ip] = pfield[p];
          }
        }
        ip += 1;
      }
    }
  }

  // assign new particles
  ip = 0;
  for (p=0; p<npoints; p++) {
    PetscScalar pcoorx,pcoorz;

    pcoorx = pcoor[2*p+0];
    pcoorz = pcoor[2*p+1];

    if ((pcoorx==0.0) && (pcoorz==0.0)) {
      pcoor[2*p+0] = pcoorx_new[ip];
      pcoor[2*p+1] = pcoorz_new[ip];

      pfield[p] = pid_new[ip];

      pfield0[p] = 0;
      pfield1[p] = 0;
      pfield2[p] = 0;
      pfield3[p] = 0;
      pfield4[p] = 0;
      pfield5[p] = 0;

      if (pid_new[ip] == 0) pfield0[p] = 1;
      if (pid_new[ip] == 1) pfield1[p] = 1;
      if (pid_new[ip] == 2) pfield2[p] = 1;
      if (pid_new[ip] == 3) pfield3[p] = 1;
      if (pid_new[ip] == 4) pfield4[p] = 1;
      if (pid_new[ip] == 5) pfield5[p] = 1;

      ip += 1;
    }
  }

  PetscCall(PetscFree(pcoorx_new));
  PetscCall(PetscFree(pcoorz_new));
  PetscCall(PetscFree(pid_new));

  PetscCall(PetscRandomDestroy(&rnd));
  PetscCall(DMSwarmRestoreField(dmswarm,"id",NULL,NULL,(void**)&pfield));
  PetscCall(DMSwarmRestoreField(dmswarm,"id0",NULL,NULL,(void**)&pfield0));
  PetscCall(DMSwarmRestoreField(dmswarm,"id1",NULL,NULL,(void**)&pfield1));
  PetscCall(DMSwarmRestoreField(dmswarm,"id2",NULL,NULL,(void**)&pfield2));
  PetscCall(DMSwarmRestoreField(dmswarm,"id3",NULL,NULL,(void**)&pfield3));
  PetscCall(DMSwarmRestoreField(dmswarm,"id4",NULL,NULL,(void**)&pfield4));
  PetscCall(DMSwarmRestoreField(dmswarm,"id5",NULL,NULL,(void**)&pfield5));
  PetscCall(DMSwarmRestoreField(dmswarm,DMSwarmPICField_coor,NULL,NULL,(void**)&pcoor));
  PetscCall(DMSwarmMigrate(dmswarm,PETSC_TRUE));

  // clean 
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dmcell,&coordx,&coordz,NULL));
  PetscCall(DMStagVecRestoreArray(dmcell,cnt_sw_local,&cnt_sw));
  PetscCall(DMStagVecRestoreArray(dmcell,cnt_nw_local,&cnt_nw));
  PetscCall(DMStagVecRestoreArray(dmcell,cnt_se_local,&cnt_se));
  PetscCall(DMStagVecRestoreArray(dmcell,cnt_ne_local,&cnt_ne));
  
  PetscCall(VecDestroy(&cnt_sw_local));
  PetscCall(VecDestroy(&cnt_nw_local));
  PetscCall(VecDestroy(&cnt_se_local));
  PetscCall(VecDestroy(&cnt_ne_local));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// Update marker phase fractions
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "UpdateMarkerPhaseFractions"
PetscErrorCode UpdateMarkerPhaseFractions(DM dmswarm, DM dmMPhase, Vec xMPhase, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  DM             dm;
  PetscInt       id;
  PetscFunctionBeginUser;

  dm = usr->dmPV;
  // Project swarm into coefficient
  id = 0;
  PetscCall(MPoint_ProjectQ1_arith_general_AP(dmswarm,"id0",dm,dmMPhase,0,id,xMPhase));//vertex
  PetscCall(MPoint_ProjectQ1_arith_general_AP(dmswarm,"id0",dm,dmMPhase,1,id,xMPhase));//face
  PetscCall(MPoint_ProjectQ1_arith_general_AP(dmswarm,"id0",dm,dmMPhase,2,id,xMPhase));//cell

  id = 1;
  PetscCall(MPoint_ProjectQ1_arith_general_AP(dmswarm,"id1",dm,dmMPhase,0,id,xMPhase));//vertex
  PetscCall(MPoint_ProjectQ1_arith_general_AP(dmswarm,"id1",dm,dmMPhase,1,id,xMPhase));//face
  PetscCall(MPoint_ProjectQ1_arith_general_AP(dmswarm,"id1",dm,dmMPhase,2,id,xMPhase));//cell

  id = 2;
  PetscCall(MPoint_ProjectQ1_arith_general_AP(dmswarm,"id2",dm,dmMPhase,0,id,xMPhase));//vertex
  PetscCall(MPoint_ProjectQ1_arith_general_AP(dmswarm,"id2",dm,dmMPhase,1,id,xMPhase));//face
  PetscCall(MPoint_ProjectQ1_arith_general_AP(dmswarm,"id2",dm,dmMPhase,2,id,xMPhase));//cell

  id = 3;
  PetscCall(MPoint_ProjectQ1_arith_general_AP(dmswarm,"id3",dm,dmMPhase,0,id,xMPhase));//vertex
  PetscCall(MPoint_ProjectQ1_arith_general_AP(dmswarm,"id3",dm,dmMPhase,1,id,xMPhase));//face
  PetscCall(MPoint_ProjectQ1_arith_general_AP(dmswarm,"id3",dm,dmMPhase,2,id,xMPhase));//cell

  id = 4;
  PetscCall(MPoint_ProjectQ1_arith_general_AP(dmswarm,"id4",dm,dmMPhase,0,id,xMPhase));//vertex
  PetscCall(MPoint_ProjectQ1_arith_general_AP(dmswarm,"id4",dm,dmMPhase,1,id,xMPhase));//face
  PetscCall(MPoint_ProjectQ1_arith_general_AP(dmswarm,"id4",dm,dmMPhase,2,id,xMPhase));//cell

  id = 5;
  PetscCall(MPoint_ProjectQ1_arith_general_AP(dmswarm,"id5",dm,dmMPhase,0,id,xMPhase));//vertex
  PetscCall(MPoint_ProjectQ1_arith_general_AP(dmswarm,"id5",dm,dmMPhase,1,id,xMPhase));//face
  PetscCall(MPoint_ProjectQ1_arith_general_AP(dmswarm,"id5",dm,dmMPhase,2,id,xMPhase));//cell

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// GetMarkerDensityPerCell
// Note: can be extended to retrieve count and array of cell ids with nmark < nmark_threshold 
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "GetMarkerDensityPerCell"
PetscErrorCode GetMarkerDensityPerCell(DM dmswarm, DM dm, PetscInt nmark[2])
{
  PetscInt       p, npoints, slot, *pcellid;
  PetscScalar    ***cnt;
  Vec            x, xlocal;
  DM             dmcom;
  const char     *cellid;
  DMSwarmCellDM  celldm;
  PetscFunctionBeginUser;

  PetscCall(DMSwarmGetLocalSize(dmswarm,&npoints));
  PetscCall(DMSwarmGetCellDMActive(dmswarm, &celldm));
  PetscCall(DMSwarmCellDMGetCellID(celldm, &cellid));
  PetscCall(DMSwarmGetField(dmswarm,cellid,NULL,NULL,(void**)&pcellid));

  // create compatible dm with 1 center dof
  PetscCall(DMStagCreateCompatibleDMStag(dm,0,0,1,0,&dmcom));
  PetscCall(DMCreateGlobalVector(dmcom,&x));
  PetscCall(DMCreateLocalVector(dmcom,&xlocal));
  PetscCall(DMStagVecGetArray(dmcom,xlocal,&cnt));
  PetscCall(DMStagGetLocationSlot(dmcom,DMSTAG_ELEMENT,0,&slot));

  // count markers/cell
  for (p=0; p<npoints; p++) {
    PetscInt cellid = -1;
    PetscInt geid[]={0,0,0};

    cellid = pcellid[p];
    PetscCall(DMStagGetLocalElementGlobalIndices(dmcom,cellid,geid));
    cnt[ geid[1] ][ geid[0] ][ slot ] += 1.0;
  }

  PetscCall(DMSwarmRestoreField(dmswarm,cellid,NULL,NULL,(void**)&pcellid));

  PetscCall(DMStagVecRestoreArray(dmcom,xlocal,&cnt));
  PetscCall(DMLocalToGlobal(dmcom,xlocal,ADD_VALUES,x));

  // get min/max 
  PetscScalar gmin, gmax;
  PetscCall(VecMin(x, NULL, &gmin)); 
  PetscCall(VecMax(x, NULL, &gmax)); 

  PetscCall(VecDestroy(&xlocal));
  PetscCall(VecDestroy(&x));
  PetscCall(DMDestroy(&dmcom));

  nmark[0] = (int)gmin;
  nmark[1] = (int)gmax;

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// Update Lithostatic pressure
// ---------------------------------------
PetscErrorCode UpdateLithostaticPressure(DM dm, Vec x, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz, idx, icenter, iwtc[MAX_MAT_PHASE];
  PetscScalar    **coordx,**coordz, ***xx, ***xwt, rho0[MAX_MAT_PHASE], wt[MAX_MAT_PHASE];
  Vec            xlocal,xMPhaselocal;
  PetscFunctionBeginUser;

  PetscCall(DMStagGetGlobalSizes(dm,&Nx,&Nz,NULL));
  PetscCall(DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 
  PetscCall(DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter)); 
  PetscCall(DMStagGetLocationSlot(dm, ELEMENT, 0, &idx)); 

  PetscCall(DMCreateLocalVector(dm, &xlocal)); 
  PetscCall(DMStagVecGetArray(dm, xlocal, &xx)); 

  // get material phase fractions
  PetscCall(DMCreateLocalVector(usr->dmMPhase, &xMPhaselocal)); 
  PetscCall(DMGlobalToLocalBegin(usr->dmMPhase,usr->xMPhase,INSERT_VALUES,xMPhaselocal)); 
  PetscCall(DMGlobalToLocalEnd  (usr->dmMPhase,usr->xMPhase,INSERT_VALUES,xMPhaselocal)); 
  PetscCall(DMStagVecGetArray(usr->dmMPhase, xMPhaselocal, &xwt)); 

  PetscCall(DMStagGetLocationSlot(usr->dmMPhase, ELEMENT, 0, &iwtc[0])); 
  PetscCall(DMStagGetLocationSlot(usr->dmMPhase, ELEMENT, 1, &iwtc[1])); 
  PetscCall(DMStagGetLocationSlot(usr->dmMPhase, ELEMENT, 2, &iwtc[2])); 
  PetscCall(DMStagGetLocationSlot(usr->dmMPhase, ELEMENT, 3, &iwtc[3])); 
  PetscCall(DMStagGetLocationSlot(usr->dmMPhase, ELEMENT, 4, &iwtc[4])); 
  PetscCall(DMStagGetLocationSlot(usr->dmMPhase, ELEMENT, 5, &iwtc[5])); 

  // Loop over local domain
  for (i = sx; i <sx+nx; i++) {
    for (j = sz+nz-1; j > sz-1; j--) { // start from top column - not parallel!
      PetscInt    iph;
      PetscScalar rho, dz;
      
      // solid material density
      for (iph = 0; iph < usr->nph; iph++) { rho0[iph] = Density(usr->mat_nd[iph].rho0,usr->mat[iph].rho_func); }
      PetscCall(GetMatPhaseFraction(i,j,xwt,iwtc,usr->nph,wt)); 
      rho = WeightAverageValue(rho0,wt,usr->nph); 

      if (j==sz+nz-1) {
        dz  = 0.5*(coordz[j][icenter]-coordz[j+1][icenter]);
        xx[j][i][idx] = LithostaticPressure(rho,dz);
      } else {
        dz  = coordz[j][icenter]-coordz[j+1][icenter];
        xx[j][i][idx] = xx[j+1][i][idx] + LithostaticPressure(rho,dz); 
      }
    }
  }

  // Restore arrays
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));
  PetscCall(DMStagVecRestoreArray(dm,xlocal,&xx)); 
  PetscCall(DMLocalToGlobalBegin(dm,xlocal,INSERT_VALUES,x)); 
  PetscCall(DMLocalToGlobalEnd  (dm,xlocal,INSERT_VALUES,x)); 
  PetscCall(VecDestroy(&xlocal)); 

  PetscCall(DMStagVecRestoreArray(usr->dmMPhase,xMPhaselocal,&xwt));
  PetscCall(VecDestroy(&xMPhaselocal)); 

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// DoOutput
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "DoOutput"
PetscErrorCode DoOutput(FDPDE fdPV, FDPDE fdT, FDPDE fdphi,void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  char           fout[FNAME_LENGTH];
  DM             dmPVcoeff, dmTcoeff, dmphicoeff;
  Vec            xPVcoeff, xTcoeff, xphicoeff;
  PetscFunctionBeginUser;

  if ((usr->par->restart) && (usr->nd->istep==usr->par->restart)) {
    PetscCall(PetscSNPrintf(usr->par->fdir_out,sizeof(usr->par->fdir_out),"Timestep%d_r",usr->nd->istep));
  } else {
    PetscCall(PetscSNPrintf(usr->par->fdir_out,sizeof(usr->par->fdir_out),"Timestep%d",usr->nd->istep));
  }
  PetscCall(CreateDirectory(usr->par->fdir_out));

  // Output bag and parameters
  PetscCall(OutputParameters(usr)); 

  // Output solution vectors
  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_xPV_ts%d",usr->par->fdir_out,usr->nd->istep));
  PetscCall(DMStagViewBinaryPython(usr->dmPV,usr->xPV,fout));

  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_xT_ts%d",usr->par->fdir_out,usr->nd->istep));
  PetscCall(DMStagViewBinaryPython(usr->dmT,usr->xT,fout));

  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_xphi_ts%d",usr->par->fdir_out,usr->nd->istep));
  PetscCall(DMStagViewBinaryPython(usr->dmphi,usr->xphi,fout));

  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_xVel_ts%d",usr->par->fdir_out,usr->nd->istep));
  PetscCall(DMStagViewBinaryPython(usr->dmVel,usr->xVel,fout));

  // DMSwarm - only 'id' is required
  const char     *fieldname[] = {"id"};
  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_pic_ts%d.xmf",usr->par->fdir_out,usr->nd->istep));
  PetscCall(DMSwarmViewFieldsXDMF(usr->dmswarm,fout,1,fieldname)); 
  // PetscCall(DMSwarmViewXDMF(usr->dmswarm,fout)); // output all swarm (to be avoided)

  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_xMPhase_ts%d",usr->par->fdir_out,usr->nd->istep));
  PetscCall(DMStagViewBinaryPython(usr->dmMPhase,usr->xMPhase,fout));

  // coefficients
  PetscCall(FDPDEGetCoefficient(fdT,&dmTcoeff,&xTcoeff));
  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_xTcoeff_ts%d",usr->par->fdir_out,usr->nd->istep));
  PetscCall(DMStagViewBinaryPython(dmTcoeff,xTcoeff,fout));

  PetscCall(FDPDEGetCoefficient(fdphi,&dmphicoeff,&xphicoeff));
  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_xphicoeff_ts%d",usr->par->fdir_out,usr->nd->istep));
  PetscCall(DMStagViewBinaryPython(dmphicoeff,xphicoeff,fout));

  PetscCall(FDPDEGetCoefficient(fdPV,&dmPVcoeff,&xPVcoeff));
  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_xPVcoeff_ts%d",usr->par->fdir_out,usr->nd->istep));
  PetscCall(DMStagViewBinaryPython(dmPVcoeff,xPVcoeff,fout));

  // previous sol and coeff - for debug
  // Vec  xphiprev, xphiprevcoeff;
  // PetscCall(FDPDEAdvDiffGetPrevSolution(fdphi,&xphiprev));
  // PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_xphiprev_ts%d",usr->par->fdir_out,usr->nd->istep));
  // PetscCall(DMStagViewBinaryPython(usr->dmphi,xphiprev,fout));
  // PetscCall(VecDestroy(&xphiprev));

  // PetscCall(FDPDEAdvDiffGetPrevCoefficient(fdphi,&xphiprevcoeff));
  // PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_xphiprevcoeff_ts%d",usr->par->fdir_out,usr->nd->istep));
  // PetscCall(DMStagViewBinaryPython(dmphicoeff,xphiprevcoeff,fout));
  // PetscCall(VecDestroy(&xphiprevcoeff));

  Vec  xphiguess;
  PetscCall(FDPDEGetSolutionGuess(fdphi,&xphiguess));
  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_xphiguess_ts%d",usr->par->fdir_out,usr->nd->istep));
  PetscCall(DMStagViewBinaryPython(usr->dmphi,xphiguess,fout));
  PetscCall(VecDestroy(&xphiguess));

  Vec  xTguess;
  PetscCall(FDPDEGetSolutionGuess(fdT,&xTguess));
  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_xTguess_ts%d",usr->par->fdir_out,usr->nd->istep));
  PetscCall(DMStagViewBinaryPython(usr->dmT,xTguess,fout));
  PetscCall(VecDestroy(&xTguess));

  // material properties
  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_matProp_ts%d",usr->par->fdir_out,usr->nd->istep));
  PetscCall(DMStagViewBinaryPython(usr->dmmatProp,usr->xmatProp,fout));

  // residuals
  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_resPV_ts%d",usr->par->fdir_out,usr->nd->istep));
  PetscCall(DMStagViewBinaryPython(usr->dmPV,fdPV->r,fout));

  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_resT_ts%d",usr->par->fdir_out,usr->nd->istep));
  PetscCall(DMStagViewBinaryPython(usr->dmT,fdT->r,fout));

  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_resphi_ts%d",usr->par->fdir_out,usr->nd->istep));
  PetscCall(DMStagViewBinaryPython(usr->dmphi,fdphi->r,fout));

  // pressures
  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_xPlith_ts%d",usr->par->fdir_out,usr->nd->istep));
  PetscCall(DMStagViewBinaryPython(usr->dmPlith,usr->xPlith,fout));

  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_xDP_ts%d",usr->par->fdir_out,usr->nd->istep));
  PetscCall(DMStagViewBinaryPython(usr->dmPlith,usr->xDP,fout));

  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_xDPold_ts%d",usr->par->fdir_out,usr->nd->istep));
  PetscCall(DMStagViewBinaryPython(usr->dmPlith,usr->xDP_old,fout));

  // strain rates, stresses, dotlam
  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_xeps_ts%d",usr->par->fdir_out,usr->nd->istep));
  PetscCall(DMStagViewBinaryPython(usr->dmeps,usr->xeps,fout));

  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_xtau_ts%d",usr->par->fdir_out,usr->nd->istep));
  PetscCall(DMStagViewBinaryPython(usr->dmeps,usr->xtau,fout));

  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_xtauold_ts%d",usr->par->fdir_out,usr->nd->istep));
  PetscCall(DMStagViewBinaryPython(usr->dmeps,usr->xtau_old,fout));

  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_xplast_ts%d",usr->par->fdir_out,usr->nd->istep));
  PetscCall(DMStagViewBinaryPython(usr->dmPlith,usr->xplast,fout));

  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_xstrain_ts%d",usr->par->fdir_out,usr->nd->istep));
  PetscCall(DMStagViewBinaryPython(usr->dmPlith,usr->xstrain,fout));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// Compute Fluid and Bulk velocity from PV and porosity solutions
// ---------------------------------------
PetscErrorCode ComputeFluidAndBulkVelocity(DM dmPV, Vec xPV, DM dmPlith, Vec xPlith, DM dmphi, Vec xphi, DM dmVel, Vec xVel, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  PetscInt       i, j, ii, sx, sz, nx, nz, Nx, Nz, iprev,inext,icenter;
  PetscScalar    ***xx, dx, dz, k_hat[4], i_hat[4], A;
  PetscScalar    **coordx,**coordz;
  Vec            xVellocal,xPVlocal,xphilocal,xPlithlocal;
  PetscScalar    ***_xPVlocal,***_xphilocal, ***_xPlithlocal;
  PetscInt       pv_slot[6],phi_slot,v_slot[8],iL,iR,iU,iD,iP,iPlith;
  PetscLogDouble tlog[2];
  PetscFunctionBeginUser;

  PetscTime(&tlog[0]);
  k_hat[0] = 0.0;
  k_hat[1] = 0.0;
  k_hat[2] = usr->par->k_hat;
  k_hat[3] = usr->par->k_hat;

  i_hat[0] = 1.0;
  i_hat[1] = 1.0;
  i_hat[2] = 0.0;
  i_hat[3] = 0.0;
  A = usr->nd->R*usr->nd->R;

  PetscCall(DMStagGetGlobalSizes(dmVel,&Nx,&Nz,NULL));
  PetscCall(DMStagGetCorners(dmVel,&sx,&sz,NULL,&nx,&nz,NULL,NULL,NULL,NULL)); 

  // get coordinates of dmPV for center and edges
  PetscCall(DMStagGetProductCoordinateArraysRead(dmPV,&coordx,&coordz,NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmPV,LEFT,&iprev));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmPV,RIGHT,&inext)); 
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmPV,ELEMENT,&icenter)); 
  
  PetscCall(DMCreateLocalVector(dmVel,&xVellocal));
  PetscCall(DMStagVecGetArray(dmVel, xVellocal, &xx)); 

  PetscCall(DMGetLocalVector(dmPV, &xPVlocal)); 
  PetscCall(DMGlobalToLocal (dmPV, xPV, INSERT_VALUES, xPVlocal)); 
  PetscCall(DMStagVecGetArrayRead(dmPV,xPVlocal,&_xPVlocal));

  PetscCall(DMGetLocalVector(dmPlith, &xPlithlocal)); 
  PetscCall(DMGlobalToLocal (dmPlith, xPlith, INSERT_VALUES, xPlithlocal)); 
  PetscCall(DMStagVecGetArrayRead(dmPlith,xPlithlocal,&_xPlithlocal));

  PetscCall(DMGetLocalVector(dmphi, &xphilocal)); 
  PetscCall(DMGlobalToLocal (dmphi, xphi, INSERT_VALUES, xphilocal)); 
  PetscCall(DMStagVecGetArrayRead(dmphi,xphilocal,&_xphilocal));

  // get slots
  iP = 0; iPlith = 1;
  iL = 2; iR  = 3;
  iD = 4; iU  = 5;
  PetscCall(DMStagGetLocationSlot(dmPV,ELEMENT,PV_ELEMENT_P, &pv_slot[iP])); 
  PetscCall(DMStagGetLocationSlot(dmPlith,ELEMENT,0, &pv_slot[iPlith])); 
  PetscCall(DMStagGetLocationSlot(dmPV,LEFT,   PV_FACE_VS,   &pv_slot[iL]));
  PetscCall(DMStagGetLocationSlot(dmPV,RIGHT,  PV_FACE_VS,   &pv_slot[iR]));
  PetscCall(DMStagGetLocationSlot(dmPV,DOWN,   PV_FACE_VS,   &pv_slot[iD]));
  PetscCall(DMStagGetLocationSlot(dmPV,UP,     PV_FACE_VS,   &pv_slot[iU]));
  PetscCall(DMStagGetLocationSlot(dmphi,ELEMENT,0,&phi_slot));

  PetscCall(DMStagGetLocationSlot(dmVel,LEFT,  VEL_FACE_VF, &v_slot[0]));
  PetscCall(DMStagGetLocationSlot(dmVel,RIGHT, VEL_FACE_VF, &v_slot[1]));
  PetscCall(DMStagGetLocationSlot(dmVel,DOWN,  VEL_FACE_VF, &v_slot[2]));
  PetscCall(DMStagGetLocationSlot(dmVel,UP,    VEL_FACE_VF, &v_slot[3]));
  PetscCall(DMStagGetLocationSlot(dmVel,LEFT,  VEL_FACE_V,  &v_slot[4]));
  PetscCall(DMStagGetLocationSlot(dmVel,RIGHT, VEL_FACE_V,  &v_slot[5]));
  PetscCall(DMStagGetLocationSlot(dmVel,DOWN,  VEL_FACE_V,  &v_slot[6]));
  PetscCall(DMStagGetLocationSlot(dmVel,UP,    VEL_FACE_V,  &v_slot[7]));

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      PetscInt    im, jm;
      PetscScalar p[5], Q[5]; 
      PetscScalar vf, Kphi, Bf, vs[4], gradP[4], gradPlith[4], phi[4];

      // grid spacing - assume constant
      dx = coordx[i][inext]-coordx[i][iprev];
      dz = coordz[j][inext]-coordz[j][iprev];

      // get PV data
      vs[0] = _xPVlocal[j][i][pv_slot[iL]]; 
      vs[1] = _xPVlocal[j][i][pv_slot[iR]];
      vs[2] = _xPVlocal[j][i][pv_slot[iD]];
      vs[3] = _xPVlocal[j][i][pv_slot[iU]];

      p[0] = _xPVlocal[j][i][pv_slot[iP]];
      if (i == 0   ) im = i; else im = i-1; p[1] = _xPVlocal[j][im][pv_slot[iP]];
      if (i == Nx-1) im = i; else im = i+1; p[2] = _xPVlocal[j][im][pv_slot[iP]];
      if (j == 0   ) jm = j; else jm = j-1; p[3] = _xPVlocal[jm][i][pv_slot[iP]];
      if (j == Nz-1) jm = j; else jm = j+1; p[4] = _xPVlocal[jm][i][pv_slot[iP]];

      gradP[0] = (p[0]-p[1])/dx;
      gradP[1] = (p[2]-p[0])/dx;
      gradP[2] = (p[0]-p[3])/dz;
      gradP[3] = (p[4]-p[0])/dz;

      p[0] = _xPlithlocal[j][i][pv_slot[iPlith]];
      if (i == 0   ) im = i; else im = i-1; p[1] = _xPlithlocal[j][im][pv_slot[iPlith]];
      if (i == Nx-1) im = i; else im = i+1; p[2] = _xPlithlocal[j][im][pv_slot[iPlith]];
      if (j == 0   ) jm = j; else jm = j-1; p[3] = _xPlithlocal[jm][i][pv_slot[iPlith]];
      if (j == Nz-1) jm = j; else jm = j+1; p[4] = _xPlithlocal[jm][i][pv_slot[iPlith]];

      gradPlith[0] = (p[0]-p[1])/dx;
      gradPlith[1] = (p[2]-p[0])/dx;
      gradPlith[2] = (p[0]-p[3])/dz;
      gradPlith[3] = (p[4]-p[0])/dz;

      // get porosity - from solid porosity
      Q[0] = 1.0 - _xphilocal[j][i][phi_slot];
      if (i == 0   ) im = i; else im = i-1; Q[1] = 1.0 - _xphilocal[j][im][phi_slot];
      if (i == Nx-1) im = i; else im = i+1; Q[2] = 1.0 - _xphilocal[j][im][phi_slot];
      if (j == 0   ) jm = j; else jm = j-1; Q[3] = 1.0 - _xphilocal[jm][i][phi_slot];
      if (j == Nz-1) jm = j; else jm = j+1; Q[4] = 1.0 - _xphilocal[jm][i][phi_slot];

      // porosity on edges
      phi[0] = (Q[1]+Q[0])*0.5; 
      phi[1] = (Q[2]+Q[0])*0.5; 
      phi[2] = (Q[3]+Q[0])*0.5; 
      phi[3] = (Q[4]+Q[0])*0.5; 
      
      for (ii = 0; ii < 4; ii++) {
        // correct for negative porosity
        if (phi[ii]<0.0) phi[ii] = 0.0;

        // fluid velocity A = R^2
        vf = LiquidVelocity(A,vs[ii],phi[ii],usr->par->n,gradP[ii],gradPlith[ii],usr->nd->rhof*k_hat[ii]);
        xx[j][i][v_slot[ii]] = vf;

        // bulk velocity
        xx[j][i][v_slot[4+ii]] = Mixture(vs[ii],vf,phi[ii]); 
      }
    }
  }

  // Restore arrays
  PetscCall(DMStagVecRestoreArray(dmVel,xVellocal,&xx)); 
  PetscCall(DMLocalToGlobalBegin(dmVel,xVellocal,INSERT_VALUES,xVel)); 
  PetscCall(DMLocalToGlobalEnd  (dmVel,xVellocal,INSERT_VALUES,xVel)); 
  PetscCall(VecDestroy(&xVellocal)); 

  PetscCall(DMStagRestoreProductCoordinateArraysRead(dmPV,&coordx,&coordz,NULL));
  PetscCall(DMStagVecRestoreArrayRead(dmPV,xPVlocal,&_xPVlocal));
  PetscCall(DMRestoreLocalVector(dmPV, &xPVlocal)); 
  PetscCall(DMStagVecRestoreArrayRead(dmPlith,xPlithlocal,&_xPlithlocal));
  PetscCall(DMRestoreLocalVector(dmPlith, &xPlithlocal)); 
  PetscCall(DMStagVecRestoreArrayRead(dmphi,xphilocal,&_xphilocal));
  PetscCall(DMRestoreLocalVector(dmphi, &xphilocal)); 

  PetscTime(&tlog[1]);
  if (usr->par->log_info) {
    printf("  ComputeFluidAndBulkVelocity: total                 %1.2e\n",tlog[1]-tlog[0]);
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// Compute timestep based on Liquid Velocity - assumed to be used with dmVel, xVel
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "LiquidVelocityExplicitTimestep"
PetscErrorCode LiquidVelocityExplicitTimestep(DM dm, Vec x, PetscScalar *dt)
{
  PetscScalar    domain_dt, global_dt, eps, dx, dz, cell_dt, cell_dt_x, cell_dt_z;
  PetscInt       iprev=-1, inext=-1;
  PetscInt       i, j, sx, sz, nx, nz, v_slot[4];
  PetscScalar    **coordx, **coordz, ***_xlocal;
  Vec            xlocal;
  PetscFunctionBeginUser;

  PetscCall(DMGetLocalVector(dm, &xlocal)); 
  PetscCall(DMGlobalToLocal (dm, x, INSERT_VALUES, xlocal)); 
  PetscCall(DMStagVecGetArrayRead(dm,xlocal,&_xlocal));

  domain_dt = 1.0e32;
  eps = 1.0e-32; /* small shift to avoid dividing by zero */

  PetscCall(DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 
  PetscCall(DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm,LEFT,&iprev)); 
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm,RIGHT,&inext)); 

  // get slots
  PetscCall(DMStagGetLocationSlot(dm,LEFT,  VEL_FACE_VF, &v_slot[0]));
  PetscCall(DMStagGetLocationSlot(dm,RIGHT, VEL_FACE_VF, &v_slot[1]));
  PetscCall(DMStagGetLocationSlot(dm,DOWN,  VEL_FACE_VF, &v_slot[2]));
  PetscCall(DMStagGetLocationSlot(dm,UP,    VEL_FACE_VF, &v_slot[3]));

  // Loop over elements - velocity is located on edge and c=1
  for (j = sz; j<sz+nz; j++) {
    for (i = sx; i<sx+nx; i++) {
      PetscScalar   xx[4];
      
      xx[0] = _xlocal[j][i][v_slot[0]];
      xx[1] = _xlocal[j][i][v_slot[1]];
      xx[2] = _xlocal[j][i][v_slot[2]];
      xx[3] = _xlocal[j][i][v_slot[3]];

      dx = coordx[0][inext]-coordx[0][iprev];
      dz = coordz[0][inext]-coordz[0][iprev];

      /* compute dx, dy for this cell */
      cell_dt_x = dx / PetscMax(PetscMax(PetscAbsScalar(xx[0]), PetscAbsScalar(xx[1])), eps);
      cell_dt_z = dz / PetscMax(PetscMax(PetscAbsScalar(xx[2]), PetscAbsScalar(xx[3])), eps);
      cell_dt   = PetscMin(cell_dt_x,cell_dt_z);
      domain_dt = PetscMin(domain_dt,cell_dt);
    }
  }

  // MPI exchange global min/max
  PetscCall(MPI_Allreduce(&domain_dt,&global_dt,1,MPI_DOUBLE,MPI_MIN,PetscObjectComm((PetscObject)dm)));

  // Return vectors and arrays
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));
  PetscCall(DMStagVecRestoreArrayRead(dm,xlocal,&_xlocal));
  PetscCall(DMRestoreLocalVector(dm,&xlocal)); 

  // Return value
  *dt = global_dt;

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// CreateDirectory
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "CreateDirectory"
PetscErrorCode CreateDirectory(const char *name)
{
  PetscMPIInt rank;
  int         status;
  PetscFunctionBeginUser;

  // create a new directory if it doesn't exist on rank zero
  PetscCall(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  if(rank==0) {
    status = mkdir(name,0777);
    if(!status) PetscPrintf(PETSC_COMM_WORLD,"# New directory created: %s \n",name);
    else        PetscPrintf(PETSC_COMM_WORLD,"# Did not create new directory: %s \n",name);
  }
  PetscCall(MPI_Barrier(PETSC_COMM_WORLD)); 

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// Output Parameters
// ---------------------------------------
PetscErrorCode OutputParameters(void *ctx) 
{
  UsrData        *usr = (UsrData*)ctx;
  char           prefix[FNAME_LENGTH],fout[FNAME_LENGTH],string[FNAME_LENGTH];
  FILE           *fp = NULL;
  PetscViewer    viewer;
  PetscFunctionBeginUser;

  // Output bag and parameters
  PetscCall(PetscSNPrintf(prefix,sizeof(prefix),"%s/%s",usr->par->fdir_out,"parameters"));
  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s.pbin",prefix));
  PetscCall(PetscViewerBinaryOpen(usr->comm,fout,FILE_MODE_WRITE,&viewer));

  // create an additional file for loading in python
  PetscCall(PetscSNPrintf(string,sizeof(string),"%s.py",prefix));
  fp = fopen(string,"w");
  if (!fp) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Cannot open file %s",string);
  fprintf(fp,"import PetscBinaryIO as pio\n");
  fprintf(fp,"import numpy as np\n\n");
  fprintf(fp,"def _PETScBinaryFilePrefix():\n");
  fprintf(fp,"  return \"%s\"\n",prefix);
  fprintf(fp,"\n");
  fprintf(fp,"def _PETScBinaryLoad():\n");
  fprintf(fp,"  io = pio.PetscBinaryIO()\n");
  fprintf(fp,"  filename = \"%s\"\n",fout);
  fprintf(fp,"  data = dict()\n");
  fprintf(fp,"  with open(filename) as fp:\n");

  // parameters - scal
  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->scal->x,1,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->scal->eta,1,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->scal->rho,1,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->scal->v,1,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->scal->t,1,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->scal->DT,1,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->scal->tau,1,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->scal->kappa,1,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->scal->kT,1,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->scal->kphi,1,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->scal->Gamma,1,PETSC_DOUBLE));

  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['scalx'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['scaleta'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['scalrho'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['scalv'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['scalt'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['scalDT'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['scaltau'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['scalkappa'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['scalkT'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['scalkphi'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['scalGamma'] = v\n");

  // parameters - nd
  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->nd->L,1,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->nd->H,1,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->nd->xmin,1,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->nd->zmin,1,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->nd->Hs,1,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->nd->Vext,1,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->nd->Vin,1,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->nd->Tbot,1,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->nd->Ttop,1,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->nd->eta_min,1,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->nd->eta_max,1,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->nd->eta_K,1,PETSC_DOUBLE));

  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->nd->Vin_free,1,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->nd->Vin_rock,1,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->nd->dzin,1,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->nd->dzin_fs,1,PETSC_DOUBLE));

  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['L'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['H'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['xmin'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['zmin'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['Hs'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['Vext'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['Vin'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['Tbot'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['Ttop'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['eta_min'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['eta_max'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['eta_K'] = v\n");

  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['Vin_free'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['Vin_rock'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['dzin'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['dzin_fs'] = v\n");

  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->nd->istep,1,PETSC_INT));
  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->nd->t,1,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->nd->dt,1,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->nd->tmax,1,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->nd->dtmax,1,PETSC_DOUBLE));

  fprintf(fp,"    v = io.readInteger(fp)\n"); fprintf(fp,"    data['istep'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['t'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['dt'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['tmax'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['dtmax'] = v\n");

  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->nd->delta,1,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->nd->R,1,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->nd->Ra,1,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->nd->Gamma,1,PETSC_DOUBLE));

  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['delta'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['R'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['Ra'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['Gamma'] = v\n");

  // // material properties
  // PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->nph,1,PETSC_INT));
  // fprintf(fp,"    v = io.readInteger(fp)\n"); fprintf(fp,"    data['nph'] = v\n");
  // PetscScalar iph;
  // char        fout[FNAME_LENGTH];
  // for (iph = 0; iph < usr->nph; iph++) {
  //   PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->mat_nd[iph].rho0,1,PETSC_DOUBLE));
  //   PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->mat_nd[iph].cp,1,PETSC_DOUBLE));
  //   PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->mat_nd[iph].kT,1,PETSC_DOUBLE));
  //   PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->mat_nd[iph].kappa,1,PETSC_DOUBLE));
  //   PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->mat_nd[iph].eta0,1,PETSC_DOUBLE));
  //   PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->mat_nd[iph].zeta0,1,PETSC_DOUBLE));
  //   PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->mat_nd[iph].G,1,PETSC_DOUBLE));
  //   PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->mat_nd[iph].Z0,1,PETSC_DOUBLE));
  //   PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->mat_nd[iph].C,1,PETSC_DOUBLE));
  //   PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->mat_nd[iph].sigmat,1,PETSC_DOUBLE));
  //   PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->mat_nd[iph].theta,1,PETSC_DOUBLE));
  //   PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->mat_nd[iph].alpha,1,PETSC_DOUBLE));

  //   fprintf(fp,"    v = io.readReal(fp)\n"); PetscCall(PetscSNPrintf(fout,sizeof(fout),"    data['mat_nd[%d].rho0'] = v\n",iph); fprintf(fp,fout));
  //   fprintf(fp,"    v = io.readReal(fp)\n"); PetscCall(PetscSNPrintf(fout,sizeof(fout),"    data['mat_nd[%d].cp'] = v\n",iph); fprintf(fp,fout));
  //   fprintf(fp,"    v = io.readReal(fp)\n"); PetscCall(PetscSNPrintf(fout,sizeof(fout),"    data['mat_nd[%d].kT'] = v\n",iph); fprintf(fp,fout));
  //   fprintf(fp,"    v = io.readReal(fp)\n"); PetscCall(PetscSNPrintf(fout,sizeof(fout),"    data['mat_nd[%d].kappa'] = v\n",iph); fprintf(fp,fout));
  //   fprintf(fp,"    v = io.readReal(fp)\n"); PetscCall(PetscSNPrintf(fout,sizeof(fout),"    data['mat_nd[%d].eta0'] = v\n",iph); fprintf(fp,fout));
  //   fprintf(fp,"    v = io.readReal(fp)\n"); PetscCall(PetscSNPrintf(fout,sizeof(fout),"    data['mat_nd[%d].zeta0'] = v\n",iph); fprintf(fp,fout));
  //   fprintf(fp,"    v = io.readReal(fp)\n"); PetscCall(PetscSNPrintf(fout,sizeof(fout),"    data['mat_nd[%d].G'] = v\n",iph); fprintf(fp,fout));
  //   fprintf(fp,"    v = io.readReal(fp)\n"); PetscCall(PetscSNPrintf(fout,sizeof(fout),"    data['mat_nd[%d].Z0'] = v\n",iph); fprintf(fp,fout));
  //   fprintf(fp,"    v = io.readReal(fp)\n"); PetscCall(PetscSNPrintf(fout,sizeof(fout),"    data['mat_nd[%d].C'] = v\n",iph); fprintf(fp,fout));
  //   fprintf(fp,"    v = io.readReal(fp)\n"); PetscCall(PetscSNPrintf(fout,sizeof(fout),"    data['mat_nd[%d].sigmat'] = v\n",iph); fprintf(fp,fout));
  //   fprintf(fp,"    v = io.readReal(fp)\n"); PetscCall(PetscSNPrintf(fout,sizeof(fout),"    data['mat_nd[%d].theta'] = v\n",iph); fprintf(fp,fout));
  //   fprintf(fp,"    v = io.readReal(fp)\n"); PetscCall(PetscSNPrintf(fout,sizeof(fout),"    data['mat_nd[%d].alpha'] = v\n",iph); fprintf(fp,fout));
  // }

  // Note: readBag() in PetscBinaryIO.py is not yet implemented, so will close the python file without reading bag
  PetscCall(PetscBagView(usr->bag,viewer));

  fprintf(fp,"    return data\n\n");
  fclose(fp);

  PetscCall(PetscViewerDestroy(&viewer));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// Load Parameters
// ---------------------------------------
PetscErrorCode LoadParametersFromFile(void *ctx) 
{
  UsrData        *usr = (UsrData*)ctx;
  char           fout[FNAME_LENGTH];
  PetscViewer    viewer;
  PetscFunctionBeginUser;

  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/parameters.pbin",usr->par->fdir_out));
  PetscCall(PetscViewerBinaryOpen(usr->comm,fout,FILE_MODE_READ,&viewer));

  // parameters - scal
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->scal->x,1,NULL,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->scal->eta,1,NULL,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->scal->rho,1,NULL,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->scal->v,1,NULL,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->scal->t,1,NULL,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->scal->DT,1,NULL,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->scal->tau,1,NULL,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->scal->kappa,1,NULL,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->scal->kT,1,NULL,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->scal->kphi,1,NULL,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->scal->Gamma,1,NULL,PETSC_DOUBLE));

  // parameters - nd
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->nd->L,1,NULL,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->nd->H,1,NULL,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->nd->xmin,1,NULL,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->nd->zmin,1,NULL,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->nd->Hs,1,NULL,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->nd->Vext,1,NULL,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->nd->Vin,1,NULL,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->nd->Tbot,1,NULL,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->nd->Ttop,1,NULL,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->nd->eta_min,1,NULL,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->nd->eta_max,1,NULL,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->nd->eta_K,1,NULL,PETSC_DOUBLE));

  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->nd->Vin_free,1,NULL,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->nd->Vin_rock,1,NULL,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->nd->dzin,1,NULL,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->nd->dzin_fs,1,NULL,PETSC_DOUBLE));
  
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->nd->istep,1,NULL,PETSC_INT));
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->nd->t,1,NULL,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->nd->dt,1,NULL,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->nd->tmax,1,NULL,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->nd->dtmax,1,NULL,PETSC_DOUBLE));

  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->nd->delta,1,NULL,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->nd->R,1,NULL,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->nd->Ra,1,NULL,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->nd->Gamma,1,NULL,PETSC_DOUBLE));

  // material properties
  
  // save timestep
  PetscInt    tstep, tout;
  PetscScalar tmax, dtmax;

  tstep = usr->par->tstep;
  tout  = usr->par->tout;
  tmax  = usr->par->tmax;
  dtmax = usr->par->dtmax;

  // read bag
  PetscCall(PetscBagLoad(viewer,usr->bag));
  PetscCall(PetscViewerDestroy(&viewer));

  usr->par->tstep = tstep;
  usr->par->tout  = tout;
  usr->par->tmax = tmax;
  usr->par->dtmax= dtmax;

  // non-dimensionalize necessary params
  usr->nd->tmax  = nd_param(usr->par->tmax*SEC_YEAR,usr->scal->t);
  usr->nd->dtmax = nd_param(usr->par->dtmax*SEC_YEAR,usr->scal->t);

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// LoadRestartFromFile
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "LoadRestartFromFile"
PetscErrorCode LoadRestartFromFile(FDPDE fdPV, FDPDE fdT, FDPDE fdphi, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  DM             dm;
  Vec            x, xTprev, xTcoeff, xTcoeffprev, xTguess, xphiprev, xphicoeff, xphicoeffprev, xphiguess;
  char           fout[FNAME_LENGTH];
  PetscViewer    viewer;
  PetscFunctionBeginUser;

  usr->plasticity = PETSC_TRUE; 
  usr->nd->istep = usr->par->restart;
  PetscCall(PetscSNPrintf(usr->par->fdir_out,sizeof(usr->par->fdir_out),"Timestep%d",usr->nd->istep));

  // load time data
  PetscCall(LoadParametersFromFile(usr));

  // correct restart variable from bag
  usr->par->restart = usr->nd->istep;
  PetscCall(InputPrintData(usr));

  // load vector data
  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_xPV_ts%d",usr->par->fdir_out,usr->nd->istep));
  PetscCall(DMStagReadBinaryPython(&dm,&x,fout));
  PetscCall(VecCopy(x,usr->xPV));
  PetscCall(VecCopy(x,fdPV->xguess));
  PetscCall(VecDestroy(&x)); 
  PetscCall(DMDestroy(&dm)); 

  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_xT_ts%d",usr->par->fdir_out,usr->nd->istep));
  PetscCall(DMStagReadBinaryPython(&dm,&x,fout));
  PetscCall(VecCopy(x,usr->xT));
  PetscCall(VecCopy(x,fdT->xguess));
  PetscCall(VecDestroy(&x)); 
  PetscCall(DMDestroy(&dm)); 

  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_xphi_ts%d",usr->par->fdir_out,usr->nd->istep));
  PetscCall(DMStagReadBinaryPython(&dm,&x,fout));
  PetscCall(VecCopy(x,usr->xphi));
  PetscCall(VecDestroy(&x)); 
  PetscCall(DMDestroy(&dm)); 

  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_resPV_ts%d",usr->par->fdir_out,usr->nd->istep));
  PetscCall(DMStagReadBinaryPython(&dm,&x,fout));
  PetscCall(VecCopy(x,fdPV->r));
  PetscCall(VecDestroy(&x)); 
  PetscCall(DMDestroy(&dm)); 

  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_resT_ts%d",usr->par->fdir_out,usr->nd->istep));
  PetscCall(DMStagReadBinaryPython(&dm,&x,fout));
  PetscCall(VecCopy(x,fdT->r));
  PetscCall(VecDestroy(&x)); 
  PetscCall(DMDestroy(&dm)); 

  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_resphi_ts%d",usr->par->fdir_out,usr->nd->istep));
  PetscCall(DMStagReadBinaryPython(&dm,&x,fout));
  PetscCall(VecCopy(x,fdphi->r));
  PetscCall(VecDestroy(&x)); 
  PetscCall(DMDestroy(&dm)); 

  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_xDP_ts%d",usr->par->fdir_out,usr->nd->istep));
  PetscCall(DMStagReadBinaryPython(&dm,&x,fout));
  PetscCall(VecCopy(x,usr->xDP_old));
  PetscCall(VecDestroy(&x)); 
  PetscCall(DMDestroy(&dm)); 

  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_xtau_ts%d",usr->par->fdir_out,usr->nd->istep));
  PetscCall(DMStagReadBinaryPython(&dm,&x,fout));
  PetscCall(VecCopy(x,usr->xtau_old));
  PetscCall(VecDestroy(&x)); 
  PetscCall(DMDestroy(&dm)); 

  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_xstrain_ts%d",usr->par->fdir_out,usr->nd->istep));
  PetscCall(DMStagReadBinaryPython(&dm,&x,fout));
  PetscCall(VecCopy(x,usr->xstrain));
  PetscCall(VecDestroy(&x)); 
  PetscCall(DMDestroy(&dm)); 

  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_xVel_ts%d",usr->par->fdir_out,usr->nd->istep));
  PetscCall(DMStagReadBinaryPython(&dm,&x,fout));
  PetscCall(VecCopy(x,usr->xVel));
  PetscCall(VecDestroy(&x)); 
  PetscCall(DMDestroy(&dm)); 
  
  // markers - read XDMF file
  const char     *fieldname[] = {"id"};
  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_pic_ts%d",usr->par->fdir_out,usr->nd->istep));
  PetscCall(DMSwarmReadBinaryXDMF_Seq(usr->dmswarm,fout,1,fieldname)); 
  PetscCall(UpdateMarkerPhaseFractions(usr->dmswarm,usr->dmMPhase,usr->xMPhase,usr));

  // initialize guess and previous solution in fdT
  PetscCall(FDPDEAdvDiffGetPrevSolution(fdT,&xTprev));
  PetscCall(VecCopy(usr->xT,xTprev));

  PetscCall(FDPDEGetSolutionGuess(fdT,&xTguess));
  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_xTguess_ts%d",usr->par->fdir_out,usr->nd->istep));
  PetscCall(DMStagReadBinaryPython(&dm,&x,fout));
  PetscCall(VecCopy(x,xTguess));
  PetscCall(VecDestroy(&x)); 
  PetscCall(DMDestroy(&dm)); 
  PetscCall(VecDestroy(&xTprev));
  PetscCall(VecDestroy(&xTguess));

  // PetscCall(FDPDEGetSolutionGuess(fdT,&xTguess));
  // PetscCall(VecCopy(xTprev,xTguess));
  // PetscCall(VecDestroy(&xTprev));
  // PetscCall(VecDestroy(&xTguess));

  PetscCall(FDPDEAdvDiffGetPrevCoefficient(fdT,&xTcoeffprev));
  PetscCall(FDPDEGetCoefficient(fdT,NULL,&xTcoeff));
  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_xTcoeff_ts%d",usr->par->fdir_out,usr->nd->istep));
  PetscCall(DMStagReadBinaryPython(&dm,&x,fout));
  PetscCall(VecCopy(x,xTcoeffprev));
  PetscCall(VecCopy(x,xTcoeff));
  PetscCall(VecDestroy(&xTcoeffprev));
  PetscCall(VecDestroy(&x)); 
  PetscCall(DMDestroy(&dm)); 

  // initialize guess and previous solution in fdphi
  PetscCall(FDPDEAdvDiffGetPrevSolution(fdphi,&xphiprev));
  PetscCall(VecCopy(usr->xphi,xphiprev));
  
  PetscCall(FDPDEGetSolutionGuess(fdphi,&xphiguess));
  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_xphiguess_ts%d",usr->par->fdir_out,usr->nd->istep));
  PetscCall(DMStagReadBinaryPython(&dm,&x,fout));
  PetscCall(VecCopy(x,xphiguess));
  PetscCall(VecDestroy(&x)); 
  PetscCall(DMDestroy(&dm)); 
  PetscCall(VecDestroy(&xphiprev));
  PetscCall(VecDestroy(&xphiguess));

  PetscCall(FDPDEAdvDiffGetPrevCoefficient(fdphi,&xphicoeffprev));
  PetscCall(FDPDEGetCoefficient(fdphi,NULL,&xphicoeff));
  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_xphicoeff_ts%d",usr->par->fdir_out,usr->nd->istep));
  PetscCall(DMStagReadBinaryPython(&dm,&x,fout));
  PetscCall(VecCopy(x,xphicoeffprev));
  PetscCall(VecCopy(x,xphicoeff));
  PetscCall(VecDestroy(&xphicoeffprev));
  PetscCall(VecDestroy(&x)); 
  PetscCall(DMDestroy(&dm)); 

  // Output load conditions
  PetscCall(DoOutput(fdPV,fdT,fdphi,usr));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// DMSwarmReadBinaryXDMF_Seq - Sequential read of marker XDMF files
// WARNING: should be generalised to read nfields!
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "DMSwarmReadBinaryXDMF_Seq"
PetscErrorCode DMSwarmReadBinaryXDMF_Seq(DM dmswarm, const char *fout, PetscInt nfield, const char *fieldname[1])
{
  FILE           *fp;
  char           fname[FNAME_LENGTH],str[FNAME_LENGTH],*ptr;
  PetscInt       i,j,nm;
  PetscFunctionBeginUser;

  // Read XDMF file and get array of data
  PetscCall(PetscSNPrintf(fname,sizeof(fname),"%s.xmf",fout));
  fp = fopen(fname, "r" );

  ptr = fgets(str,FNAME_LENGTH,fp); if (!ptr) { str[0] = 0;}
  ptr = fgets(str,FNAME_LENGTH,fp); if (!ptr) { str[0] = 0;}
  ptr = fgets(str,FNAME_LENGTH,fp); if (!ptr) { str[0] = 0;}
  ptr = fgets(str,FNAME_LENGTH,fp); if (!ptr) { str[0] = 0;}
  ptr = fgets(str,FNAME_LENGTH,fp); if (!ptr) { str[0] = 0;} // line containing nmark

  char tmp[256], *p;
  for (i=0; i<strlen(str); i++) {
    j=0;
    while(str[i]>='0' && str[i]<='9'){
     tmp[j]=str[i];
     i++; j++;
    }
  }
  nm = (int)strtol(tmp, &p, 10);

  ptr = fgets(str,FNAME_LENGTH,fp); if (!ptr) { str[0] = 0;}
  ptr = fgets(str,FNAME_LENGTH,fp); if (!ptr) { str[0] = 0;}
  fclose(fp);

  // allocate memory to arrays
  PetscScalar  *x, *z, *id;
	size_t      sz;
  PetscViewer v;

  sz = (size_t)nm*sizeof(PetscScalar);
	PetscCall(PetscMalloc(sz, &x)); 
  PetscCall(PetscMalloc(sz, &z)); 
  PetscCall(PetscMalloc(sz, &id));

  // get binary data from .pbin
  PetscCall(PetscSNPrintf(fname,sizeof(fname),"%s_swarm_fields.pbin",fout));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,fname,FILE_MODE_READ,&v));

  // topology
  for (i=0; i<nm; i++) {
    PetscInt pvertex[3]; // pvertex[0] = 1; pvertex[1] = 1; pvertex[2] = i;
    PetscCall(PetscViewerBinaryRead(v,pvertex,3,NULL,PETSC_INT));
  }

  // coordinates
  for (i=0; i<nm; i++) {
    PetscScalar xp[2];
    PetscCall(PetscViewerBinaryRead(v,xp,2,NULL,PETSC_DOUBLE));
    x[i] = xp[0]; z[i] = xp[1];
  }

  // id
  for (i=0; i<nm; i++) {
    PetscCall(PetscViewerBinaryRead(v,&id[i],1,NULL,PETSC_DOUBLE));
  }

  PetscCall(PetscViewerDestroy(&v));

  // set swarm size
  PetscCall(DMSwarmSetLocalSizes(dmswarm,nm,0)); 
  
  // Populate dmswarm 'id' with new data
  PetscScalar *pcoor,*pfield,*pfield0, *pfield1, *pfield2,*pfield3, *pfield4, *pfield5;
  PetscCall(DMSwarmGetField(dmswarm,DMSwarmPICField_coor,NULL,NULL,(void**)&pcoor));
  PetscCall(DMSwarmGetField(dmswarm,"id",NULL,NULL,(void**)&pfield));
  PetscCall(DMSwarmGetField(dmswarm,"id0",NULL,NULL,(void**)&pfield0));
  PetscCall(DMSwarmGetField(dmswarm,"id1",NULL,NULL,(void**)&pfield1));
  PetscCall(DMSwarmGetField(dmswarm,"id2",NULL,NULL,(void**)&pfield2));
  PetscCall(DMSwarmGetField(dmswarm,"id3",NULL,NULL,(void**)&pfield3));
  PetscCall(DMSwarmGetField(dmswarm,"id4",NULL,NULL,(void**)&pfield4));
  PetscCall(DMSwarmGetField(dmswarm,"id5",NULL,NULL,(void**)&pfield5));

  for (i=0; i<nm; i++) {
    pcoor[2*i+0] = x[i];
    pcoor[2*i+1] = z[i];

    // dummy fields used for projection
    pfield[i]  = (int) id[i];
    pfield0[i] = 0;
    pfield1[i] = 0;
    pfield2[i] = 0;
    pfield3[i] = 0;
    pfield4[i] = 0;
    pfield5[i] = 0;

    // update binary representation
    if (pfield[i]==0) pfield0[i] = 1;
    if (pfield[i]==1) pfield1[i] = 1;
    if (pfield[i]==2) pfield2[i] = 1;
    if (pfield[i]==3) pfield3[i] = 1;
    if (pfield[i]==4) pfield4[i] = 1;
    if (pfield[i]==5) pfield5[i] = 1;
  }
  
  PetscCall(DMSwarmRestoreField(dmswarm,"id",NULL,NULL,(void**)&pfield));
  PetscCall(DMSwarmRestoreField(dmswarm,"id0",NULL,NULL,(void**)&pfield0));
  PetscCall(DMSwarmRestoreField(dmswarm,"id1",NULL,NULL,(void**)&pfield1));
  PetscCall(DMSwarmRestoreField(dmswarm,"id2",NULL,NULL,(void**)&pfield2));
  PetscCall(DMSwarmRestoreField(dmswarm,"id3",NULL,NULL,(void**)&pfield3));
  PetscCall(DMSwarmRestoreField(dmswarm,"id4",NULL,NULL,(void**)&pfield4));
  PetscCall(DMSwarmRestoreField(dmswarm,"id5",NULL,NULL,(void**)&pfield5));
  PetscCall(DMSwarmRestoreField(dmswarm,DMSwarmPICField_coor,NULL,NULL,(void**)&pcoor));

   // Migrate swarm - assign cells and sub-domanin to points
  PetscCall(DMSwarmMigrate(dmswarm,PETSC_TRUE));

  // free arrays
  PetscCall(PetscFree(x)); 
  PetscCall(PetscFree(z)); 
  PetscCall(PetscFree(id)); 

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// UpdateStrainRates
// ---------------------------------------
PetscErrorCode UpdateStrainRates_Array(DM dm, Vec x, void *ctx)
/* dm - dmstag, x - solution PV vector */
{
  UsrData        *usr = (UsrData*) ctx;
  DM             dmeps;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz, nslots, ii;
  Vec            xeps, xepslocal, xlocal;
  PetscScalar    ***xxeps, ***xx;
  PetscScalar    **coordx,**coordz;
  PetscFunctionBeginUser;

  dmeps = usr->dmeps;
  xeps  = usr->xeps;
  nslots= 4;

  // Local vectors
  PetscCall(DMCreateLocalVector (dmeps,&xepslocal)); 
  PetscCall(DMStagVecGetArray(dmeps,xepslocal,&xxeps)); 

  PetscCall(DMGetLocalVector(dm,&xlocal)); 
  PetscCall(DMGlobalToLocal (dm,x,INSERT_VALUES,xlocal)); 
  PetscCall(DMStagVecGetArray(dm,xlocal,&xx)); 
  PetscCall(DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));

  // Get domain corners
  PetscCall(DMStagGetGlobalSizes(dmeps, &Nx, &Nz,NULL));
  PetscCall(DMStagGetCorners(dmeps, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 

  // Get location slots
  PetscInt ise[4], isld[4], isrd[4], islu[4], isru[4]; 
  for (ii = 0; ii < nslots; ii++) { 
    PetscCall(DMStagGetLocationSlot(dmeps,ELEMENT   ,ii,&ise[ii] )); 
    PetscCall(DMStagGetLocationSlot(dmeps,DOWN_LEFT ,ii,&isld[ii])); 
    PetscCall(DMStagGetLocationSlot(dmeps,DOWN_RIGHT,ii,&isrd[ii])); 
    PetscCall(DMStagGetLocationSlot(dmeps,UP_LEFT   ,ii,&islu[ii])); 
    PetscCall(DMStagGetLocationSlot(dmeps,UP_RIGHT  ,ii,&isru[ii])); 
  }

  // Loop over local domain and get strain rates
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      PetscInt     idx[4];
      PetscScalar  epsII[5], exx[5], ezz[5], exz[5];

      // strain rates in center and corner
      ii = 0; PetscCall(DMStagGetArrayPointStrainRates(dm,xx,coordx,coordz,i,j,DMSTAG_ELEMENT   ,&epsII[ii],&exx[ii],&ezz[ii],&exz[ii])); 
      ii = 1; PetscCall(DMStagGetArrayPointStrainRates(dm,xx,coordx,coordz,i,j,DMSTAG_DOWN_LEFT ,&epsII[ii],&exx[ii],&ezz[ii],&exz[ii])); 
      ii = 2; PetscCall(DMStagGetArrayPointStrainRates(dm,xx,coordx,coordz,i,j,DMSTAG_DOWN_RIGHT,&epsII[ii],&exx[ii],&ezz[ii],&exz[ii])); 
      ii = 3; PetscCall(DMStagGetArrayPointStrainRates(dm,xx,coordx,coordz,i,j,DMSTAG_UP_LEFT   ,&epsII[ii],&exx[ii],&ezz[ii],&exz[ii])); 
      ii = 4; PetscCall(DMStagGetArrayPointStrainRates(dm,xx,coordx,coordz,i,j,DMSTAG_UP_RIGHT  ,&epsII[ii],&exx[ii],&ezz[ii],&exz[ii])); 

      // boundaries
      if (i==0) { // down left
        ezz[1] = ezz[0];
        exx[1] = exx[0];
      }

      if (i==Nx-1) { // down right
        ezz[2] = ezz[0];
        exx[2] = exx[0];
      }

      if (j==0) { // down left
        exx[1] = exx[0];
        ezz[1] = ezz[0];
      }

      if (j==Nz-1) { // up left
        exx[3] = exx[0];
        ezz[3] = ezz[0];
      }

      if ((i==Nx-1) && (j==Nz-1)) { // up right
        exx[4] = exx[0];
        ezz[4] = ezz[0];
      }

      if ((i==0) || (i==Nx-1) || (j==0) || (j==Nz-1)) { // boundaries
        for (ii = 1; ii < 5; ii++) {
          epsII[ii] = PetscPowScalar(0.5*(exx[ii]*exx[ii] + ezz[ii]*ezz[ii] + 2.0*exz[ii]*exz[ii]),0.5);
        }
      }

      // save strain-rates
      for (ii = 0; ii < 5; ii++) { 
        if (ii==0) { idx[0] = ise[0];  idx[1] = ise[1];  idx[2] = ise[2];  idx[3] = ise[3];  }
        if (ii==1) { idx[0] = isld[0]; idx[1] = isld[1]; idx[2] = isld[2]; idx[3] = isld[3]; }
        if (ii==2) { idx[0] = isrd[0]; idx[1] = isrd[1]; idx[2] = isrd[2]; idx[3] = isrd[3]; }
        if (ii==3) { idx[0] = islu[0]; idx[1] = islu[1]; idx[2] = islu[2]; idx[3] = islu[3]; }
        if (ii==4) { idx[0] = isru[0]; idx[1] = isru[1]; idx[2] = isru[2]; idx[3] = isru[3]; }

        xxeps[j][i][idx[0]] = exx[ii];
        xxeps[j][i][idx[1]] = ezz[ii];
        xxeps[j][i][idx[2]] = exz[ii];
        xxeps[j][i][idx[3]] = epsII[ii];
      }
    }
  }

  // Restore and map local to global
  PetscCall(DMStagVecRestoreArray(dmeps,xepslocal,&xxeps)); 
  PetscCall(DMLocalToGlobal(dmeps,xepslocal,INSERT_VALUES,xeps)); 
  PetscCall(VecDestroy(&xepslocal)); 

  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));
  PetscCall(DMStagVecRestoreArray(dm,xlocal,&xx)); 
  PetscCall(DMRestoreLocalVector(dm,&xlocal)); 

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// IntegratePlasticStrain
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "IntegratePlasticStrain"
PetscErrorCode IntegratePlasticStrain(DM dm, Vec lam, Vec dotlam, void *ctx)
{
  UsrData        *usr = (UsrData*) ctx;
  Vec            lamlocal, dotlamlocal;
  PetscScalar    dt;
  PetscInt       i,j, sx, sz, nx, nz, iP;
  PetscScalar    ***_lam, ***_dotlam;
  PetscFunctionBeginUser;

  dt = usr->nd->dt; 
  PetscCall(DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 
  // Create local vector
  PetscCall(DMCreateLocalVector(dm, &lamlocal)); 
  PetscCall(DMGlobalToLocalBegin(dm,lam,INSERT_VALUES,lamlocal));
  PetscCall(DMGlobalToLocalEnd(dm,lam,INSERT_VALUES,lamlocal));
  PetscCall(DMStagVecGetArray(dm, lamlocal, &_lam)); 

  PetscCall(DMCreateLocalVector(dm, &dotlamlocal)); 
  PetscCall(DMGlobalToLocal (dm, dotlam, INSERT_VALUES, dotlamlocal)); 
  PetscCall(DMStagVecGetArray(dm, dotlamlocal, &_dotlam)); 
  PetscCall(DMStagGetLocationSlot(dm,DMSTAG_ELEMENT,0,&iP));

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      _lam[j][i][iP] += _dotlam[j][i][iP] * dt;
    }
  }

  // Restore arrays, local vectors
  PetscCall(DMStagVecRestoreArray(dm,lamlocal,&_lam));
  PetscCall(DMLocalToGlobalBegin(dm,lamlocal,INSERT_VALUES,lam)); 
  PetscCall(DMLocalToGlobalEnd  (dm,lamlocal,INSERT_VALUES,lam)); 
  PetscCall(VecDestroy(&lamlocal)); 

  PetscCall(DMStagVecRestoreArray(dm,dotlamlocal,&_dotlam));
  PetscCall(DMRestoreLocalVector(dm, &dotlamlocal )); 

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// PhaseDiagram_1Component
// ---------------------------------------
PetscErrorCode PhaseDiagram_1Component(DM dmT, Vec xT, DM dm, Vec x, DM dmphase, Vec xphase, void *ctx)
{
  UsrData        *usr = (UsrData*) ctx;
  PetscInt       i, j, sx, sz, nx, nz, iE, isurf, icenter, iwtc[MAX_MAT_PHASE];
  PetscScalar    ***xx, ***xwt, ***xxT;
  Vec            xlocal, xphaselocal, xTlocal;
  PetscScalar    **coordx,**coordz, T0, DT, L;
  PetscFunctionBeginUser;

  DT = usr->scal->DT;
  T0 = usr->par->Ttop;
  L  = usr->par->La;

  PetscCall(DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 
  PetscCall(DMStagGetLocationSlot(dm, ELEMENT, 0, &iE)); 
  PetscCall(DMStagGetLocationSlot(dmphase, ELEMENT, 0, &isurf)); 

  PetscCall(DMCreateLocalVector(dm, &xlocal)); 
  PetscCall(DMGlobalToLocal (dm, x, INSERT_VALUES, xlocal)); 
  PetscCall(DMStagVecGetArray(dm, xlocal, &xx)); 
  PetscCall(DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter)); 

  PetscCall(DMCreateLocalVector(dmT, &xTlocal)); 
  PetscCall(DMGlobalToLocal (dmT, xT, INSERT_VALUES, xTlocal)); 
  PetscCall(DMStagVecGetArray(dmT, xTlocal, &xxT)); 

  // get material phase fractions
  PetscCall(DMCreateLocalVector(dmphase, &xphaselocal)); 
  PetscCall(DMGlobalToLocal (dmphase, xphase, INSERT_VALUES, xphaselocal)); 
  PetscCall(DMStagVecGetArray(dmphase, xphaselocal, &xwt)); 

  PetscCall(DMStagGetLocationSlot(usr->dmMPhase, ELEMENT, 0, &iwtc[0])); 
  PetscCall(DMStagGetLocationSlot(usr->dmMPhase, ELEMENT, 1, &iwtc[1])); 
  PetscCall(DMStagGetLocationSlot(usr->dmMPhase, ELEMENT, 2, &iwtc[2])); 
  PetscCall(DMStagGetLocationSlot(usr->dmMPhase, ELEMENT, 3, &iwtc[3])); 
  PetscCall(DMStagGetLocationSlot(usr->dmMPhase, ELEMENT, 4, &iwtc[4])); 
  PetscCall(DMStagGetLocationSlot(usr->dmMPhase, ELEMENT, 5, &iwtc[5])); 
  
  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      PetscInt iph;
      PetscScalar  Tsol, T, phi, phinew, z, nd_T;
      PetscScalar cp, cp0[MAX_MAT_PHASE], wt[MAX_MAT_PHASE];

      // solid material heat capacity
      for (iph = 0; iph < usr->nph; iph++) { cp0[iph] = usr->mat[iph].cp; }
      PetscCall(GetMatPhaseFraction(i,j,xwt,iwtc,usr->nph,wt)); 
      cp = WeightAverageValue(cp0,wt,usr->nph); 

      z = (coordz[j][icenter]+usr->nd->Hs)*usr->scal->x;
      Tsol = Tsolidus_1Component(usr->par->Tsol0,z);
      T = xxT[j][i][iE]*DT+T0;
      phi = 1.0 - xx[j][i][iE];
      phinew = phi;

      // PetscPrintf(PETSC_COMM_WORLD,"# i=%d j=%d cp = %1.12e z = %1.12e Tsol = %1.12e T = %1.12e phi= %1.12e\n",i,j,cp,z,Tsol,T,phi);

      if (T>Tsol) {
        phinew = phi + PetscMin(1.0-phi,cp*(T-Tsol)/L);
      }
      if (T<Tsol) {
        phinew = phi - PetscMin(phi,cp*(Tsol-T)/L);
      }
      T = T-(phinew-phi)*L/cp;
      nd_T = nd_paramT(T,T0,DT);

      // return values
      xxT[j][i][iE] = nd_T;
      xx[j][i][iE] = 1.0 - phinew;
    }
  }

  // Restore arrays
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));
  PetscCall(DMStagVecRestoreArray(dm,xlocal,&xx)); 
  PetscCall(DMLocalToGlobalBegin(dm,xlocal,INSERT_VALUES,x)); 
  PetscCall(DMLocalToGlobalEnd  (dm,xlocal,INSERT_VALUES,x)); 
  PetscCall(VecDestroy(&xlocal)); 

  PetscCall(DMStagVecRestoreArray(dmT,xTlocal,&xxT)); 
  PetscCall(DMLocalToGlobalBegin(dmT,xTlocal,INSERT_VALUES,xT)); 
  PetscCall(DMLocalToGlobalEnd  (dmT,xTlocal,INSERT_VALUES,xT)); 
  PetscCall(VecDestroy(&xTlocal)); 

  PetscCall(DMStagVecRestoreArray(dmphase,xphaselocal,&xwt)); 
  PetscCall(VecDestroy(&xphaselocal)); 

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// CorrectNegativePorosity
// ---------------------------------------
PetscErrorCode CorrectNegativePorosity(DM dm, Vec x)
{
  PetscInt       i, j, sx, sz, nx, nz, iE;
  PetscScalar    ***xx;
  Vec            xlocal;
  PetscFunctionBeginUser;

  PetscCall(DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 
  PetscCall(DMStagGetLocationSlot(dm, ELEMENT, 0, &iE)); 

  PetscCall(DMCreateLocalVector(dm, &xlocal)); 
  PetscCall(DMGlobalToLocal (dm, x, INSERT_VALUES, xlocal)); 
  PetscCall(DMStagVecGetArray(dm, xlocal, &xx)); 
  
  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      if (xx[j][i][iE]>1.0) xx[j][i][iE] = 1.0;
    }
  }

  // Restore arrays
  PetscCall(DMStagVecRestoreArray(dm,xlocal,&xx)); 
  PetscCall(DMLocalToGlobalBegin(dm,xlocal,INSERT_VALUES,x)); 
  PetscCall(DMLocalToGlobalEnd  (dm,xlocal,INSERT_VALUES,x)); 
  PetscCall(VecDestroy(&xlocal)); 

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// CheckNegativePorosity
// ---------------------------------------
PetscErrorCode CheckNegativePorosity(DM dm, Vec x, PetscBool *masscons)
{
  PetscInt       i, j, sx, sz, nx, nz, iE;
  PetscScalar    ***xx;
  Vec            xlocal;
  PetscBool      flag;
  PetscFunctionBeginUser;

  PetscCall(DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 
  PetscCall(DMStagGetLocationSlot(dm, ELEMENT, 0, &iE)); 

  PetscCall(DMCreateLocalVector(dm, &xlocal)); 
  PetscCall(DMGlobalToLocal (dm, x, INSERT_VALUES, xlocal)); 
  PetscCall(DMStagVecGetArray(dm, xlocal, &xx)); 

  flag = PETSC_TRUE;
  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      if (xx[j][i][iE]>1.0) {
        flag = PETSC_FALSE;
        break;
      }
    }
  }

  // Restore arrays
  PetscCall(DMStagVecRestoreArray(dm,xlocal,&xx)); 
  PetscCall(DMLocalToGlobalBegin(dm,xlocal,INSERT_VALUES,x)); 
  PetscCall(DMLocalToGlobalEnd  (dm,xlocal,INSERT_VALUES,x)); 
  PetscCall(VecDestroy(&xlocal)); 

  *masscons = flag;

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// CorrectPorosityFreeSurface
// ---------------------------------------
PetscErrorCode CorrectPorosityFreeSurface(DM dm, Vec x, DM dmphase, Vec xphase)
{
  PetscInt       i, j, sx, sz, nx, nz, iE, isurf;
  PetscScalar    ***xx, ***xxphase;
  Vec            xlocal, xphaselocal;
  PetscFunctionBeginUser;

  PetscCall(DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 
  PetscCall(DMStagGetLocationSlot(dm, ELEMENT, 0, &iE)); 
  PetscCall(DMStagGetLocationSlot(dmphase, ELEMENT, 0, &isurf)); 

  PetscCall(DMCreateLocalVector(dm, &xlocal)); 
  PetscCall(DMGlobalToLocal (dm, x, INSERT_VALUES, xlocal)); 
  PetscCall(DMStagVecGetArray(dm, xlocal, &xx)); 

  PetscCall(DMCreateLocalVector(dmphase, &xphaselocal)); 
  PetscCall(DMGlobalToLocal (dmphase, xphase, INSERT_VALUES, xphaselocal)); 
  PetscCall(DMStagVecGetArray(dmphase, xphaselocal, &xxphase)); 
  
  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      if (xxphase[j][i][isurf]>0.0) xx[j][i][iE] = 1.0;
    }
  }

  // Restore arrays
  PetscCall(DMStagVecRestoreArray(dm,xlocal,&xx)); 
  PetscCall(DMLocalToGlobalBegin(dm,xlocal,INSERT_VALUES,x)); 
  PetscCall(DMLocalToGlobalEnd  (dm,xlocal,INSERT_VALUES,x)); 
  PetscCall(VecDestroy(&xlocal)); 

  PetscCall(DMStagVecRestoreArray(dmphase,xphaselocal,&xxphase)); 
  PetscCall(VecDestroy(&xphaselocal)); 

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// GetMatPhaseFraction
// ---------------------------------------
PetscErrorCode GetMatPhaseFraction(PetscInt i, PetscInt j, PetscScalar ***xwt, PetscInt *iwtc, PetscInt n, PetscScalar *wt)
{ 
  PetscInt ii;
  PetscFunctionBeginUser;
  for (ii = 0; ii <n; ii++) {
    wt[ii] = xwt[j][i][iwtc[ii]];
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
PetscErrorCode GetCornerAvgFromCenter(PetscScalar *Ac, PetscScalar *Acorner)
{
  PetscFunctionBeginUser;
  Acorner[0] = (Ac[0]+Ac[1]+Ac[3]+Ac[5])*0.25;
  Acorner[1] = (Ac[0]+Ac[2]+Ac[3]+Ac[6])*0.25;
  Acorner[2] = (Ac[0]+Ac[1]+Ac[4]+Ac[7])*0.25;
  Acorner[3] = (Ac[0]+Ac[2]+Ac[4]+Ac[8])*0.25;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
PetscErrorCode Get9PointCenterValues(PetscInt i, PetscInt j, PetscInt idx, PetscInt Nx, PetscInt Nz, PetscScalar ***xx, PetscScalar *x)
{
  PetscInt  im, jm, ip, jp;
  PetscFunctionBeginUser;
    // get property in center
    if (i == 0   ) im = i; else im = i-1;
    if (i == Nx-1) ip = i; else ip = i+1;
    if (j == 0   ) jm = j; else jm = j-1;
    if (j == Nz-1) jp = j; else jp = j+1;

    x[0] = xx[j ][i ][idx]; // i  ,j   - C
    x[1] = xx[j ][im][idx]; // i-1,j   - L
    x[2] = xx[j ][ip][idx]; // i+1,j   - R
    x[3] = xx[jm][i ][idx]; // i  ,j-1 - D
    x[4] = xx[jp][i ][idx]; // i  ,j+1 - U
    x[5] = xx[jm][im][idx]; // i-1,j-1 - LD
    x[6] = xx[jm][ip][idx]; // i+1,j-1 - RD
    x[7] = xx[jp][im][idx]; // i-1,j+1 - LU
    x[8] = xx[jp][ip][idx]; // i+1,j+1 - RU
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
PetscErrorCode GetTensorPointValues(PetscInt i, PetscInt j, PetscInt *idx, PetscScalar ***xx, PetscScalar *x)
{
  PetscFunctionBeginUser;
    x[0] = xx[j][i][idx[0]]; // xx
    x[1] = xx[j][i][idx[1]]; // zz
    x[2] = xx[j][i][idx[2]]; // xz
    x[3] = xx[j][i][idx[3]]; // II
  PetscFunctionReturn(PETSC_SUCCESS);
}