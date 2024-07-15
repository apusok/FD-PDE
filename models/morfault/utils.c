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
  PetscErrorCode ierr;
  
  PetscFunctionBeginUser;

  // initialize T: half-space cooling model
  ierr = HalfSpaceCooling_MOR(usr);CHKERRQ(ierr);

  // initialize constant solid porosity field
  ierr = VecSet(usr->xphi,1.0-usr->par->phi0); CHKERRQ(ierr);
  // ierr = SetInitialPorosityField(usr);CHKERRQ(ierr);

  // set swarm initial size and coordinates
  PetscInt ppcell[] = {usr->par->ppcell,usr->par->ppcell};
  ierr = MPointCoordLayout_DomainVolumeWithCellList(usr->dmswarm,0,NULL,0.5,ppcell,COOR_INITIALIZE);CHKERRQ(ierr);

  // swarm initial condition - output only id field
  ierr = SetSwarmInitialCondition(usr->dmswarm,usr);CHKERRQ(ierr);

  // Update marker phase fractions on the dmstag 
  ierr = UpdateMarkerPhaseFractions(usr->dmswarm,usr->dmMPhase,usr->xMPhase,usr);CHKERRQ(ierr);

  // Update lithostatic pressure
  ierr = UpdateLithostaticPressure(usr->dmPlith,usr->xPlith,usr);CHKERRQ(ierr);

  // Create initial guess for PV - viscous solution
  usr->plasticity = PETSC_FALSE; 
  PetscPrintf(PETSC_COMM_WORLD,"# (PV) Rheology: VISCO-ELASTIC \n");
  usr->nd->dt = usr->nd->dtmax;
  ierr = FDPDESolve(fdPV,NULL);CHKERRQ(ierr);
  ierr = FDPDEGetSolution(fdPV,&xPV);CHKERRQ(ierr);
  ierr = VecCopy(xPV,usr->xPV);CHKERRQ(ierr);
  ierr = FDPDEGetSolutionGuess(fdPV,&xguess); CHKERRQ(ierr); 
  ierr = VecCopy(usr->xPV,xguess);CHKERRQ(ierr);
  ierr = VecDestroy(&xguess);CHKERRQ(ierr);
  ierr = VecDestroy(&xPV);CHKERRQ(ierr);
  usr->plasticity = PETSC_TRUE; 

  // Update fluid velocity to zero and v=vs
  ierr = ComputeFluidAndBulkVelocity(usr->dmPV,usr->xPV,usr->dmPlith,usr->xPlith,usr->dmphi,usr->xphi,usr->dmVel,usr->xVel,usr);CHKERRQ(ierr);

  // Initialize guess and previous solution in fdT
  ierr = FDPDEAdvDiffGetPrevSolution(fdT,&xTprev);CHKERRQ(ierr);
  ierr = VecCopy(usr->xT,xTprev);CHKERRQ(ierr);
  ierr = FDPDEGetSolutionGuess(fdT,&xTguess);CHKERRQ(ierr);
  ierr = VecCopy(xTprev,xTguess);CHKERRQ(ierr);
  ierr = VecDestroy(&xTprev);CHKERRQ(ierr);
  ierr = VecDestroy(&xTguess);CHKERRQ(ierr);

  ierr = FDPDEGetCoefficient(fdT,&dmTcoeff,&xTcoeff);CHKERRQ(ierr);
  ierr = FDPDEAdvDiffGetPrevCoefficient(fdT,&xTcoeffprev);CHKERRQ(ierr);
  ierr = FormCoefficient_T(fdT,usr->dmT,usr->xT,dmTcoeff,xTcoeffprev,usr);CHKERRQ(ierr);
  ierr = VecCopy(xTcoeffprev,xTcoeff);CHKERRQ(ierr);
  ierr = VecDestroy(&xTcoeffprev);CHKERRQ(ierr);

  // Initialize guess and previous solution in fdphi
  ierr = FDPDEAdvDiffGetPrevSolution(fdphi,&xphiprev);CHKERRQ(ierr);
  ierr = VecCopy(usr->xphi,xphiprev);CHKERRQ(ierr);
  ierr = FDPDEGetSolutionGuess(fdphi,&xphiguess);CHKERRQ(ierr);
  ierr = VecCopy(xphiprev,xphiguess);CHKERRQ(ierr);
  ierr = VecDestroy(&xphiprev);CHKERRQ(ierr);
  ierr = VecDestroy(&xphiguess);CHKERRQ(ierr);

  ierr = FDPDEGetCoefficient(fdphi,&dmphicoeff,&xphicoeff);CHKERRQ(ierr);
  ierr = FDPDEAdvDiffGetPrevCoefficient(fdphi,&xphicoeffprev);CHKERRQ(ierr);
  ierr = FormCoefficient_phi(fdphi,usr->dmphi,usr->xphi,dmphicoeff,xphicoeffprev,usr);CHKERRQ(ierr);
  ierr = VecCopy(xphicoeffprev,xphicoeff);CHKERRQ(ierr);
  ierr = VecDestroy(&xphicoeffprev);CHKERRQ(ierr);

  // Output initial conditions
  ierr = DoOutput(fdPV,fdT,fdphi,usr);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// Half-Space Cooling model
// ---------------------------------------
PetscErrorCode HalfSpaceCooling_MOR(void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz, iT, icenter;
  PetscScalar    **coordx,**coordz, ***xx, Tm, Ts, xmor, Hs;
  Vec            x, xlocal;
  DM             dm;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  dm  = usr->dmT;
  x   = usr->xT;
  Ts   = usr->par->Ttop;
  Tm   = usr->par->Tbot;
  xmor = usr->nd->xmin+usr->nd->L/2.0;
  Hs   = usr->par->Hs;

  ierr = DMStagGetGlobalSizes(dm, &Nx, &Nz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter);CHKERRQ(ierr); 
  ierr = DMStagGetLocationSlot(dm, ELEMENT, 0, &iT); CHKERRQ(ierr);

  ierr = DMCreateLocalVector(dm, &xlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm, xlocal, &xx); CHKERRQ(ierr);
  
  PetscScalar xs, zs, r;
  xs = usr->par->incl_x;
  zs = usr->par->incl_z;
  r = usr->par->incl_r;

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      PetscScalar age, T, nd_T, xp, zp, rp;
      
      if (usr->par->model_setup<=1) age = usr->par->age*1.0e6*SEC_YEAR; // constant age in Myr
      else age  = usr->par->age*1.0e6*SEC_YEAR + dim_param(fabs(coordx[i][icenter]),usr->scal->x)/dim_param(usr->nd->Vext,usr->scal->v); // age varying with distance from axis + initial age

      // half-space cooling temperature - take into account free surface
      T = HalfSpaceCoolingTemp(Tm,Ts,-Hs-dim_param(coordz[j][icenter],usr->scal->x),usr->scal->kappa,age,usr->par->hs_factor); 
      if (T-Ts<0.0) T = Ts;

      // add initial T perturbation
      if (usr->par->model_setup==1) {
        xp = dim_param(coordx[i][icenter],usr->scal->x)-xs;
        zp = dim_param(coordz[j][icenter],usr->scal->x)-zs;
        rp = PetscSqrtScalar(xp*xp+zp*zp);
        if (rp<=r) {T += (r-rp)/r*usr->par->incl_dT;}
      }

      // nd_T = (T - usr->par->T0)/usr->par->DT;
      nd_T = nd_paramT(T,Ts,usr->scal->DT);

      // temperature (phi=0)
      xx[j][i][iT] = nd_T;
    }
  }

  // Restore arrays
  ierr = DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArray(dm,xlocal,&xx); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm,xlocal,INSERT_VALUES,x); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,xlocal,INSERT_VALUES,x); CHKERRQ(ierr);
  ierr = VecDestroy(&xlocal); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// SetInitialPorosityField
// ---------------------------------------
PetscErrorCode SetInitialPorosityField(void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz, iE, icenter;
  PetscScalar    **coordx,**coordz, ***xx, Hs;
  Vec            x, xlocal;
  DM             dm;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  dm  = usr->dmphi;
  x   = usr->xphi;
  Hs  = usr->nd->Hs;

  ierr = DMStagGetGlobalSizes(dm, &Nx, &Nz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter);CHKERRQ(ierr); 
  ierr = DMStagGetLocationSlot(dm, ELEMENT, 0, &iE); CHKERRQ(ierr);

  ierr = DMCreateLocalVector(dm, &xlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm, xlocal, &xx); CHKERRQ(ierr);
  
  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      if (coordz[j][icenter]<=-Hs){ xx[j][i][iE] = 1.0-usr->par->phi0; } 
      else                        { xx[j][i][iE] = 1.0-usr->par->phi0;}
    }
  }

  // Restore arrays
  ierr = DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArray(dm,xlocal,&xx); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm,xlocal,INSERT_VALUES,x); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,xlocal,INSERT_VALUES,x); CHKERRQ(ierr);
  ierr = VecDestroy(&xlocal); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// SetSwarmInitialCondition
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "SetSwarmInitialCondition"
PetscErrorCode SetSwarmInitialCondition(DM dmswarm, void *ctx)
{
  UsrData   *usr = (UsrData*)ctx;
  PetscScalar *pcoor,*pfield,*pfield0, *pfield1, *pfield2,*pfield3, *pfield4, *pfield5, ztop;
  PetscInt  npoints,p;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ztop = usr->nd->zmin+usr->nd->H;
  ierr = DMSwarmGetLocalSize(dmswarm,&npoints);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dmswarm,DMSwarmPICField_coor,NULL,NULL,(void**)&pcoor);CHKERRQ(ierr);
    
  ierr = DMSwarmGetField(dmswarm,"id",NULL,NULL,(void**)&pfield);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dmswarm,"id0",NULL,NULL,(void**)&pfield0);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dmswarm,"id1",NULL,NULL,(void**)&pfield1);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dmswarm,"id2",NULL,NULL,(void**)&pfield2);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dmswarm,"id3",NULL,NULL,(void**)&pfield3);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dmswarm,"id4",NULL,NULL,(void**)&pfield4);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dmswarm,"id5",NULL,NULL,(void**)&pfield5);CHKERRQ(ierr);

  PetscScalar xs, zs, r, xp, zp, rp;
  xs = usr->par->incl_x;
  zs = usr->par->incl_z;
  r = usr->par->incl_r;

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

    // weak seed
    if (usr->par->model_setup==0) {
      xp = dim_param(xcoor,usr->scal->x)-xs;
      zp = dim_param(zcoor,usr->scal->x)-zs;
      rp = PetscSqrtScalar(xp*xp+zp*zp);
      if (rp<=r) { pfield[p] = usr->par->mat1_id; }
    }

    // update binary representation
    if (pfield[p]==0) pfield0[p] = 1;
    if (pfield[p]==1) pfield1[p] = 1;
    if (pfield[p]==2) pfield2[p] = 1;
    if (pfield[p]==3) pfield3[p] = 1;
    if (pfield[p]==4) pfield4[p] = 1;
    if (pfield[p]==5) pfield5[p] = 1;
  }
  ierr = DMSwarmRestoreField(dmswarm,"id",NULL,NULL,(void**)&pfield);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(dmswarm,"id0",NULL,NULL,(void**)&pfield0);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(dmswarm,"id1",NULL,NULL,(void**)&pfield1);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(dmswarm,"id2",NULL,NULL,(void**)&pfield2);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(dmswarm,"id3",NULL,NULL,(void**)&pfield3);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(dmswarm,"id4",NULL,NULL,(void**)&pfield4);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(dmswarm,"id5",NULL,NULL,(void**)&pfield5);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(dmswarm,DMSwarmPICField_coor,NULL,NULL,(void**)&pcoor);CHKERRQ(ierr);

  PetscFunctionReturn(0);
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
  PetscErrorCode ierr;
  PetscFunctionBegin;

  dxcell = usr->nd->L/usr->par->nx/usr->par->ppcell;
  dzcell = usr->nd->H/usr->par->nz/usr->par->ppcell;

  // influx
  usr->nd->dzin += usr->nd->Vin*usr->nd->dt;
  mx = (int)(usr->nd->L/dxcell);
  mz = (int)(usr->nd->dzin/dzcell);
  nmark_in = mx*mz;

  if (nmark_in==0) { PetscFunctionReturn(0); }
  
  ierr = DMSwarmAddNPoints(dmswarm,nmark_in);
  ierr = DMSwarmGetLocalSize(dmswarm,&npoints);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dmswarm,DMSwarmPICField_coor,NULL,NULL,(void**)&pcoor);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dmswarm,"id",NULL,NULL,(void**)&pfield);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dmswarm,"id0",NULL,NULL,(void**)&pfield0);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dmswarm,"id1",NULL,NULL,(void**)&pfield1);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dmswarm,"id2",NULL,NULL,(void**)&pfield2);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dmswarm,"id3",NULL,NULL,(void**)&pfield3);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dmswarm,"id4",NULL,NULL,(void**)&pfield4);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dmswarm,"id5",NULL,NULL,(void**)&pfield5);CHKERRQ(ierr);

  ierr = PetscRandomCreate(PETSC_COMM_SELF,&rnd);CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(rnd,0.0,dxcell*0.5);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rnd);CHKERRQ(ierr);

  PetscScalar dx0, dz0;
  dx0 = usr->nd->xmin+dxcell*0.5;
  dz0 = usr->nd->zmin+dzcell*0.5;

  for (p=0; p<npoints; p++) {
    PetscScalar xcoor,zcoor;
    
    xcoor = pcoor[2*p+0];
    zcoor = pcoor[2*p+1];
    if ((xcoor==0.0) && (zcoor==0.0)) {
      ierr = PetscRandomGetValue(rnd,&value);CHKERRQ(ierr);
      pcoor[2*p+0] = dx0+value;
      ierr = PetscRandomGetValue(rnd,&value);CHKERRQ(ierr);
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

  ierr = PetscRandomDestroy(&rnd);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(dmswarm,"id",NULL,NULL,(void**)&pfield);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(dmswarm,"id0",NULL,NULL,(void**)&pfield0);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(dmswarm,"id1",NULL,NULL,(void**)&pfield1);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(dmswarm,"id2",NULL,NULL,(void**)&pfield2);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(dmswarm,"id3",NULL,NULL,(void**)&pfield3);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(dmswarm,"id4",NULL,NULL,(void**)&pfield4);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(dmswarm,"id5",NULL,NULL,(void**)&pfield5);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(dmswarm,DMSwarmPICField_coor,NULL,NULL,(void**)&pcoor);CHKERRQ(ierr);
  ierr = DMSwarmMigrate(dmswarm,PETSC_TRUE);CHKERRQ(ierr);

  // reset
  usr->nd->dzin = 0.0;

  PetscFunctionReturn(0);
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
  PetscErrorCode ierr;
  PetscFunctionBegin;

  dm = usr->dmPV;
  // Project swarm into coefficient
  id = 0;
  ierr = MPoint_ProjectQ1_arith_general(dmswarm,"id0",dm,dmMPhase,0,id,xMPhase);CHKERRQ(ierr);//vertex
  ierr = MPoint_ProjectQ1_arith_general(dmswarm,"id0",dm,dmMPhase,1,id,xMPhase);CHKERRQ(ierr);//face
  ierr = MPoint_ProjectQ1_arith_general(dmswarm,"id0",dm,dmMPhase,2,id,xMPhase);CHKERRQ(ierr);//cell

  id = 1;
  ierr = MPoint_ProjectQ1_arith_general(dmswarm,"id1",dm,dmMPhase,0,id,xMPhase);CHKERRQ(ierr);//vertex
  ierr = MPoint_ProjectQ1_arith_general(dmswarm,"id1",dm,dmMPhase,1,id,xMPhase);CHKERRQ(ierr);//face
  ierr = MPoint_ProjectQ1_arith_general(dmswarm,"id1",dm,dmMPhase,2,id,xMPhase);CHKERRQ(ierr);//cell

  id = 2;
  ierr = MPoint_ProjectQ1_arith_general(dmswarm,"id2",dm,dmMPhase,0,id,xMPhase);CHKERRQ(ierr);//vertex
  ierr = MPoint_ProjectQ1_arith_general(dmswarm,"id2",dm,dmMPhase,1,id,xMPhase);CHKERRQ(ierr);//face
  ierr = MPoint_ProjectQ1_arith_general(dmswarm,"id2",dm,dmMPhase,2,id,xMPhase);CHKERRQ(ierr);//cell

  id = 3;
  ierr = MPoint_ProjectQ1_arith_general(dmswarm,"id3",dm,dmMPhase,0,id,xMPhase);CHKERRQ(ierr);//vertex
  ierr = MPoint_ProjectQ1_arith_general(dmswarm,"id3",dm,dmMPhase,1,id,xMPhase);CHKERRQ(ierr);//face
  ierr = MPoint_ProjectQ1_arith_general(dmswarm,"id3",dm,dmMPhase,2,id,xMPhase);CHKERRQ(ierr);//cell

  id = 4;
  ierr = MPoint_ProjectQ1_arith_general(dmswarm,"id4",dm,dmMPhase,0,id,xMPhase);CHKERRQ(ierr);//vertex
  ierr = MPoint_ProjectQ1_arith_general(dmswarm,"id4",dm,dmMPhase,1,id,xMPhase);CHKERRQ(ierr);//face
  ierr = MPoint_ProjectQ1_arith_general(dmswarm,"id4",dm,dmMPhase,2,id,xMPhase);CHKERRQ(ierr);//cell

  id = 5;
  ierr = MPoint_ProjectQ1_arith_general(dmswarm,"id5",dm,dmMPhase,0,id,xMPhase);CHKERRQ(ierr);//vertex
  ierr = MPoint_ProjectQ1_arith_general(dmswarm,"id5",dm,dmMPhase,1,id,xMPhase);CHKERRQ(ierr);//face
  ierr = MPoint_ProjectQ1_arith_general(dmswarm,"id5",dm,dmMPhase,2,id,xMPhase);CHKERRQ(ierr);//cell

  PetscFunctionReturn(0);
}

// ---------------------------------------
// GetMarkerDensityPerCell
// Note: can be extended to retrieve count and array of cell ids with nmark < nmark_threshold 
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "GetMarkerDensityPerCell"
PetscErrorCode GetMarkerDensityPerCell(DM dmswarm, DM dm, PetscInt nmark[2])
{
  PetscInt       p, npoints, ncell, *pcellid, *cnt, Nx, Nz;
  Vec            xlocal;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = DMSwarmGetSize(dmswarm,&npoints);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dmswarm,DMSwarmPICField_cellid,NULL,NULL,(void**)&pcellid);CHKERRQ(ierr);
  ierr = DMStagGetGlobalSizes(dm, &Nx, &Nz,NULL);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &xlocal); CHKERRQ(ierr);
  ierr = VecGetSize(xlocal,&ncell);CHKERRQ(ierr);
  ierr = PetscCalloc1(ncell,&cnt);CHKERRQ(ierr);

  // count markers/cell
  for (p=0; p<npoints; p++) {
    PetscInt cellid = -1;
    cellid = pcellid[p];
    cnt[cellid] += 1.0;
  }

  // get min/max 
  PetscInt lmin, lmax, gmin = 0, gmax = 0;
  lmin = 1000;
  lmax = 0;
  for (p=0; p<ncell; p++) {
    PetscBool in_global_space;
    ierr = DMStagLocalElementIndexInGlobalSpace_2d(dm,p,&in_global_space);CHKERRQ(ierr);
    if (in_global_space) {
      if (cnt[p]<lmin) lmin = cnt[p];
      if (cnt[p]>lmax) lmax = cnt[p];
    }
  }

  // MPI exchange global min/max
  ierr = MPI_Allreduce(&lmin,&gmin,1,MPI_INT,MPI_MIN,PetscObjectComm((PetscObject)dm));CHKERRQ(ierr);
  ierr = MPI_Allreduce(&lmax,&gmax,1,MPI_INT,MPI_MAX,PetscObjectComm((PetscObject)dm));CHKERRQ(ierr);

  ierr = DMSwarmRestoreField(dmswarm,DMSwarmPICField_cellid,NULL,NULL,(void**)&pcellid);CHKERRQ(ierr);
  ierr = PetscFree(cnt);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &xlocal); CHKERRQ(ierr);

  nmark[0] = gmin;
  nmark[1] = gmax;

  PetscFunctionReturn(0);
}

// ---------------------------------------
// Update Lithostatic pressure
// ---------------------------------------
PetscErrorCode UpdateLithostaticPressure(DM dm, Vec x, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  PetscInt       i, j, sx, sz, nx, nz, idx, icenter, iwtc[MAX_MAT_PHASE];
  PetscScalar    **coordx,**coordz, ***xx, ***xwt, rho0[MAX_MAT_PHASE], wt[MAX_MAT_PHASE];
  Vec            xlocal,xMPhaselocal;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter);CHKERRQ(ierr); 
  ierr = DMStagGetLocationSlot(dm, ELEMENT, 0, &idx); CHKERRQ(ierr);

  ierr = DMCreateLocalVector(dm, &xlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm, xlocal, &xx); CHKERRQ(ierr);

  // get material phase fractions
  ierr = DMCreateLocalVector(usr->dmMPhase, &xMPhaselocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(usr->dmMPhase,usr->xMPhase,INSERT_VALUES,xMPhaselocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (usr->dmMPhase,usr->xMPhase,INSERT_VALUES,xMPhaselocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(usr->dmMPhase, xMPhaselocal, &xwt); CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(usr->dmMPhase, ELEMENT, 0, &iwtc[0]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmMPhase, ELEMENT, 1, &iwtc[1]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmMPhase, ELEMENT, 2, &iwtc[2]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmMPhase, ELEMENT, 3, &iwtc[3]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmMPhase, ELEMENT, 4, &iwtc[4]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmMPhase, ELEMENT, 5, &iwtc[5]); CHKERRQ(ierr);

  // Loop over local domain
  for (i = sx; i <sx+nx; i++) {
    for (j = sz+nz-1; j > -1; j--) { // start from top column - not parallel!
      PetscInt    iph;
      PetscScalar rho;
      // solid material density
      for (iph = 0; iph < usr->nph; iph++) { rho0[iph] = Density(usr->mat[iph].rho0,usr->mat[iph].rho_func); }

      // get material phase fraction
      ierr = GetMatPhaseFraction(i,j,xwt,iwtc,usr->nph,wt); CHKERRQ(ierr);
      rho = WeightAverageValue(rho0,wt,usr->nph); 
      xx[j][i][idx] = LithostaticPressure(rho,usr->scal->rho,coordz[j][icenter]); 
    }
  }

  // Restore arrays
  ierr = DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArray(dm,xlocal,&xx); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm,xlocal,INSERT_VALUES,x); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,xlocal,INSERT_VALUES,x); CHKERRQ(ierr);
  ierr = VecDestroy(&xlocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArray(usr->dmMPhase,xMPhaselocal,&xwt);CHKERRQ(ierr);
  ierr = VecDestroy(&xMPhaselocal); CHKERRQ(ierr);

  PetscFunctionReturn(0);
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
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  if ((usr->par->restart) && (usr->nd->istep==usr->par->restart)) {
    ierr = PetscSNPrintf(usr->par->fdir_out,sizeof(usr->par->fdir_out),"Timestep%d_r",usr->nd->istep);
  } else {
    ierr = PetscSNPrintf(usr->par->fdir_out,sizeof(usr->par->fdir_out),"Timestep%d",usr->nd->istep);
  }
  ierr = CreateDirectory(usr->par->fdir_out);CHKERRQ(ierr);

  // Output bag and parameters
  ierr = OutputParameters(usr);CHKERRQ(ierr); 

  // Output solution vectors
  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_xPV_ts%d",usr->par->fdir_out,usr->nd->istep);
  ierr = DMStagViewBinaryPython(usr->dmPV,usr->xPV,fout);CHKERRQ(ierr);

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_xT_ts%d",usr->par->fdir_out,usr->nd->istep);
  ierr = DMStagViewBinaryPython(usr->dmT,usr->xT,fout);CHKERRQ(ierr);

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_xphi_ts%d",usr->par->fdir_out,usr->nd->istep);
  ierr = DMStagViewBinaryPython(usr->dmphi,usr->xphi,fout);CHKERRQ(ierr);

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_xVel_ts%d",usr->par->fdir_out,usr->nd->istep);
  ierr = DMStagViewBinaryPython(usr->dmVel,usr->xVel,fout);CHKERRQ(ierr);

  // DMSwarm - only 'id' is required
  const char     *fieldname[] = {"id"};
  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_pic_ts%d.xmf",usr->par->fdir_out,usr->nd->istep);
  ierr = DMSwarmViewFieldsXDMF(usr->dmswarm,fout,1,fieldname); CHKERRQ(ierr);
  // ierr = DMSwarmViewXDMF(usr->dmswarm,fout);CHKERRQ(ierr);

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_xMPhase_ts%d",usr->par->fdir_out,usr->nd->istep);
  ierr = DMStagViewBinaryPython(usr->dmMPhase,usr->xMPhase,fout);CHKERRQ(ierr);

  // coefficients
  ierr = FDPDEGetCoefficient(fdT,&dmTcoeff,&xTcoeff);CHKERRQ(ierr);
  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_xTcoeff_ts%d",usr->par->fdir_out,usr->nd->istep);
  ierr = DMStagViewBinaryPython(dmTcoeff,xTcoeff,fout);CHKERRQ(ierr);

  ierr = FDPDEGetCoefficient(fdphi,&dmphicoeff,&xphicoeff);CHKERRQ(ierr);
  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_xphicoeff_ts%d",usr->par->fdir_out,usr->nd->istep);
  ierr = DMStagViewBinaryPython(dmphicoeff,xphicoeff,fout);CHKERRQ(ierr);

  ierr = FDPDEGetCoefficient(fdPV,&dmPVcoeff,&xPVcoeff);CHKERRQ(ierr);
  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_xPVcoeff_ts%d",usr->par->fdir_out,usr->nd->istep);
  ierr = DMStagViewBinaryPython(dmPVcoeff,xPVcoeff,fout);CHKERRQ(ierr);

  // material properties - eta, zeta, G, Z, permeability, density
  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_matProp_ts%d",usr->par->fdir_out,usr->nd->istep);
  ierr = DMStagViewBinaryPython(usr->dmmatProp,usr->xmatProp,fout);CHKERRQ(ierr);

  // residuals
  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_resPV_ts%d",usr->par->fdir_out,usr->nd->istep);
  ierr = DMStagViewBinaryPython(usr->dmPV,fdPV->r,fout);CHKERRQ(ierr);

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_resT_ts%d",usr->par->fdir_out,usr->nd->istep);
  ierr = DMStagViewBinaryPython(usr->dmT,fdT->r,fout);CHKERRQ(ierr);

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_resphi_ts%d",usr->par->fdir_out,usr->nd->istep);
  ierr = DMStagViewBinaryPython(usr->dmphi,fdphi->r,fout);CHKERRQ(ierr);

  // pressures
  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_xPlith_ts%d",usr->par->fdir_out,usr->nd->istep);
  ierr = DMStagViewBinaryPython(usr->dmPlith,usr->xPlith,fout);CHKERRQ(ierr);

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_xDP_ts%d",usr->par->fdir_out,usr->nd->istep);
  ierr = DMStagViewBinaryPython(usr->dmPlith,usr->xDP,fout);CHKERRQ(ierr);

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_xDPold_ts%d",usr->par->fdir_out,usr->nd->istep);
  ierr = DMStagViewBinaryPython(usr->dmPlith,usr->xDP_old,fout);CHKERRQ(ierr);

  // strain rates, stresses, dotlam
  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_xeps_ts%d",usr->par->fdir_out,usr->nd->istep);
  ierr = DMStagViewBinaryPython(usr->dmeps,usr->xeps,fout);CHKERRQ(ierr);

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_xtau_ts%d",usr->par->fdir_out,usr->nd->istep);
  ierr = DMStagViewBinaryPython(usr->dmeps,usr->xtau,fout);CHKERRQ(ierr);

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_xtauold_ts%d",usr->par->fdir_out,usr->nd->istep);
  ierr = DMStagViewBinaryPython(usr->dmeps,usr->xtau_old,fout);CHKERRQ(ierr);

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_xplast_ts%d",usr->par->fdir_out,usr->nd->istep);
  ierr = DMStagViewBinaryPython(usr->dmPlith,usr->xplast,fout);CHKERRQ(ierr);

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_xstrain_ts%d",usr->par->fdir_out,usr->nd->istep);
  ierr = DMStagViewBinaryPython(usr->dmPlith,usr->xstrain,fout);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// Compute Fluid and Bulk velocity from PV and porosity solutions
// ---------------------------------------
PetscErrorCode ComputeFluidAndBulkVelocity(DM dmPV, Vec xPV, DM dmPlith, Vec xPlith, DM dmphi, Vec xphi, DM dmVel, Vec xVel, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  PetscInt       i, j, ii, sx, sz, nx, nz, Nx, Nz, iprev,inext,icenter;
  PetscScalar    ***xx, dx, dz, k_hat[4], A;
  PetscScalar    **coordx,**coordz;
  Vec            xVellocal,xPVlocal,xphilocal,xPlithlocal;
  PetscScalar    ***_xPVlocal,***_xphilocal, ***_xPlithlocal;
  PetscInt       pv_slot[6],phi_slot,v_slot[8],iL,iR,iU,iD,iP,iPlith;
  PetscLogDouble tlog[2];
  PetscErrorCode ierr;
  PetscFunctionBegin;

  PetscTime(&tlog[0]);
  k_hat[0] = 0.0;
  k_hat[1] = 0.0;
  k_hat[2] = usr->par->k_hat;
  k_hat[3] = usr->par->k_hat;
  A = usr->scal->kphi*usr->scal->eta/(usr->scal->x*usr->scal->x*usr->par->mu);

  ierr = DMStagGetGlobalSizes(dmVel,&Nx,&Nz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dmVel,&sx,&sz,NULL,&nx,&nz,NULL,NULL,NULL,NULL); CHKERRQ(ierr);

  // get coordinates of dmPV for center and edges
  ierr = DMStagGetProductCoordinateArraysRead(dmPV,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmPV,LEFT,&iprev);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmPV,RIGHT,&inext);CHKERRQ(ierr); 
  ierr = DMStagGetProductCoordinateLocationSlot(dmPV,ELEMENT,&icenter);CHKERRQ(ierr); 
  
  ierr = DMCreateLocalVector(dmVel,&xVellocal);CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dmVel, xVellocal, &xx); CHKERRQ(ierr);

  ierr = DMGetLocalVector(dmPV, &xPVlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dmPV, xPV, INSERT_VALUES, xPVlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(dmPV,xPVlocal,&_xPVlocal);CHKERRQ(ierr);

  ierr = DMGetLocalVector(dmPlith, &xPlithlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dmPlith, xPlith, INSERT_VALUES, xPlithlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(dmPlith,xPlithlocal,&_xPlithlocal);CHKERRQ(ierr);

  ierr = DMGetLocalVector(dmphi, &xphilocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dmphi, xphi, INSERT_VALUES, xphilocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(dmphi,xphilocal,&_xphilocal);CHKERRQ(ierr);

  // get slots
  iP = 0; iPlith = 1;
  iL = 2; iR  = 3;
  iD = 4; iU  = 5;
  ierr = DMStagGetLocationSlot(dmPV,DMSTAG_ELEMENT,PV_ELEMENT_P, &pv_slot[iP]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmPlith,DMSTAG_ELEMENT,0, &pv_slot[iPlith]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmPV,DMSTAG_LEFT,   PV_FACE_VS,   &pv_slot[iL]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmPV,DMSTAG_RIGHT,  PV_FACE_VS,   &pv_slot[iR]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmPV,DMSTAG_DOWN,   PV_FACE_VS,   &pv_slot[iD]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmPV,DMSTAG_UP,     PV_FACE_VS,   &pv_slot[iU]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmphi,DMSTAG_ELEMENT,0,&phi_slot);CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(dmVel,DMSTAG_LEFT,  VEL_FACE_VF, &v_slot[0]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmVel,DMSTAG_RIGHT, VEL_FACE_VF, &v_slot[1]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmVel,DMSTAG_DOWN,  VEL_FACE_VF, &v_slot[2]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmVel,DMSTAG_UP,    VEL_FACE_VF, &v_slot[3]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmVel,DMSTAG_LEFT,  VEL_FACE_V,  &v_slot[4]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmVel,DMSTAG_RIGHT, VEL_FACE_V,  &v_slot[5]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmVel,DMSTAG_DOWN,  VEL_FACE_V,  &v_slot[6]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmVel,DMSTAG_UP,    VEL_FACE_V,  &v_slot[7]);CHKERRQ(ierr);

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
      // gradPlith[2] = (p[0]-p[3])/dz;
      // gradPlith[3] = (p[4]-p[0])/dz;
      gradPlith[2] = 0.0;
      gradPlith[3] = 0.0;

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
        // permeability
        Kphi = Permeability(phi[ii],usr->par->n);

        // fluid buoyancy - revise for more complex density model
        // Bf = usr->par->rhof/usr->scal->rho*k_hat[ii]; // orig
        Bf = k_hat[ii]; 

        // fluid velocity
        vf = FluidVelocity(A,Kphi,vs[ii],phi[ii],gradP[ii],gradPlith[ii],Bf);
        xx[j][i][v_slot[ii]] = vf;

        // bulk velocity
        xx[j][i][v_slot[4+ii]] = Mixture(vs[ii],vf,phi[ii]); 
      }
    }
  }

  // Restore arrays
  ierr = DMStagVecRestoreArray(dmVel,xVellocal,&xx); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dmVel,xVellocal,INSERT_VALUES,xVel); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dmVel,xVellocal,INSERT_VALUES,xVel); CHKERRQ(ierr);
  ierr = VecDestroy(&xVellocal); CHKERRQ(ierr);

  ierr = DMStagRestoreProductCoordinateArraysRead(dmPV,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArrayRead(dmPV,xPVlocal,&_xPVlocal);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmPV, &xPVlocal); CHKERRQ(ierr);
  ierr = DMStagVecRestoreArrayRead(dmPlith,xPlithlocal,&_xPlithlocal);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmPlith, &xPlithlocal); CHKERRQ(ierr);
  ierr = DMStagVecRestoreArrayRead(dmphi,xphilocal,&_xphilocal);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmphi, &xphilocal); CHKERRQ(ierr);

  PetscTime(&tlog[1]);
  if (usr->par->log_info) {
    printf("  ComputeFluidAndBulkVelocity: total                 %1.2e\n",tlog[1]-tlog[0]);
  }

  PetscFunctionReturn(0);
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
  PetscErrorCode ierr;
  PetscFunctionBegin;

  // create a new directory if it doesn't exist on rank zero
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  if(rank==0) {
    status = mkdir(name,0777);
    if(!status) PetscPrintf(PETSC_COMM_WORLD,"# New directory created: %s \n",name);
    else        PetscPrintf(PETSC_COMM_WORLD,"# Did not create new directory: %s \n",name);
  }
  ierr = MPI_Barrier(PETSC_COMM_WORLD); CHKERRQ(ierr);

  PetscFunctionReturn(0);
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
  PetscErrorCode ierr;
  PetscFunctionBegin;

  // Output bag and parameters
  ierr = PetscSNPrintf(prefix,sizeof(prefix),"%s/%s",usr->par->fdir_out,"parameters");
  ierr = PetscSNPrintf(fout,sizeof(fout),"%s.pbin",prefix);
  ierr = PetscViewerBinaryOpen(usr->comm,fout,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);

  // create an additional file for loading in python
  ierr = PetscSNPrintf(string,sizeof(string),"%s.py",prefix);
  fp = fopen(string,"w");
  if (!fp) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Cannot open file %s",string);
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
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->scal->x,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->scal->eta,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->scal->rho,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->scal->v,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->scal->t,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->scal->DT,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->scal->tau,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->scal->kappa,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->scal->kT,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->scal->kphi,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->scal->Gamma,1,PETSC_DOUBLE);CHKERRQ(ierr);

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
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->nd->L,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->nd->H,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->nd->xmin,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->nd->zmin,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->nd->Hs,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->nd->Vext,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->nd->Vin,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->nd->Tbot,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->nd->Ttop,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->nd->eta_min,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->nd->eta_max,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->nd->eta_K,1,PETSC_DOUBLE);CHKERRQ(ierr);

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

  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->nd->istep,1,PETSC_INT);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->nd->t,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->nd->dt,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->nd->tmax,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->nd->dtmax,1,PETSC_DOUBLE);CHKERRQ(ierr);

  fprintf(fp,"    v = io.readInteger(fp)\n"); fprintf(fp,"    data['istep'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['t'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['dt'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['tmax'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['dtmax'] = v\n");

  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->nd->delta,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->nd->R,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->nd->Ra,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->nd->Gamma,1,PETSC_DOUBLE);CHKERRQ(ierr);

  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['delta'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['R'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['Ra'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['Gamma'] = v\n");

  // // material properties
  // ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->nph,1,PETSC_INT);CHKERRQ(ierr);
  // fprintf(fp,"    v = io.readInteger(fp)\n"); fprintf(fp,"    data['nph'] = v\n");
  // PetscScalar iph;
  // char        fout[FNAME_LENGTH];
  // for (iph = 0; iph < usr->nph; iph++) {
  //   ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->mat_nd[iph].rho0,1,PETSC_DOUBLE);CHKERRQ(ierr);
  //   ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->mat_nd[iph].cp,1,PETSC_DOUBLE);CHKERRQ(ierr);
  //   ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->mat_nd[iph].kT,1,PETSC_DOUBLE);CHKERRQ(ierr);
  //   ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->mat_nd[iph].kappa,1,PETSC_DOUBLE);CHKERRQ(ierr);
  //   ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->mat_nd[iph].eta0,1,PETSC_DOUBLE);CHKERRQ(ierr);
  //   ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->mat_nd[iph].zeta0,1,PETSC_DOUBLE);CHKERRQ(ierr);
  //   ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->mat_nd[iph].G,1,PETSC_DOUBLE);CHKERRQ(ierr);
  //   ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->mat_nd[iph].Z0,1,PETSC_DOUBLE);CHKERRQ(ierr);
  //   ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->mat_nd[iph].C,1,PETSC_DOUBLE);CHKERRQ(ierr);
  //   ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->mat_nd[iph].sigmat,1,PETSC_DOUBLE);CHKERRQ(ierr);
  //   ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->mat_nd[iph].theta,1,PETSC_DOUBLE);CHKERRQ(ierr);
  //   ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->mat_nd[iph].alpha,1,PETSC_DOUBLE);CHKERRQ(ierr);

  //   fprintf(fp,"    v = io.readReal(fp)\n"); ierr = PetscSNPrintf(fout,sizeof(fout),"    data['mat_nd[%d].rho0'] = v\n",iph); fprintf(fp,fout);
  //   fprintf(fp,"    v = io.readReal(fp)\n"); ierr = PetscSNPrintf(fout,sizeof(fout),"    data['mat_nd[%d].cp'] = v\n",iph); fprintf(fp,fout);
  //   fprintf(fp,"    v = io.readReal(fp)\n"); ierr = PetscSNPrintf(fout,sizeof(fout),"    data['mat_nd[%d].kT'] = v\n",iph); fprintf(fp,fout);
  //   fprintf(fp,"    v = io.readReal(fp)\n"); ierr = PetscSNPrintf(fout,sizeof(fout),"    data['mat_nd[%d].kappa'] = v\n",iph); fprintf(fp,fout);
  //   fprintf(fp,"    v = io.readReal(fp)\n"); ierr = PetscSNPrintf(fout,sizeof(fout),"    data['mat_nd[%d].eta0'] = v\n",iph); fprintf(fp,fout);
  //   fprintf(fp,"    v = io.readReal(fp)\n"); ierr = PetscSNPrintf(fout,sizeof(fout),"    data['mat_nd[%d].zeta0'] = v\n",iph); fprintf(fp,fout);
  //   fprintf(fp,"    v = io.readReal(fp)\n"); ierr = PetscSNPrintf(fout,sizeof(fout),"    data['mat_nd[%d].G'] = v\n",iph); fprintf(fp,fout);
  //   fprintf(fp,"    v = io.readReal(fp)\n"); ierr = PetscSNPrintf(fout,sizeof(fout),"    data['mat_nd[%d].Z0'] = v\n",iph); fprintf(fp,fout);
  //   fprintf(fp,"    v = io.readReal(fp)\n"); ierr = PetscSNPrintf(fout,sizeof(fout),"    data['mat_nd[%d].C'] = v\n",iph); fprintf(fp,fout);
  //   fprintf(fp,"    v = io.readReal(fp)\n"); ierr = PetscSNPrintf(fout,sizeof(fout),"    data['mat_nd[%d].sigmat'] = v\n",iph); fprintf(fp,fout);
  //   fprintf(fp,"    v = io.readReal(fp)\n"); ierr = PetscSNPrintf(fout,sizeof(fout),"    data['mat_nd[%d].theta'] = v\n",iph); fprintf(fp,fout);
  //   fprintf(fp,"    v = io.readReal(fp)\n"); ierr = PetscSNPrintf(fout,sizeof(fout),"    data['mat_nd[%d].alpha'] = v\n",iph); fprintf(fp,fout);
  // }

  // Note: readBag() in PetscBinaryIO.py is not yet implemented, so will close the python file without reading bag
  // output bag
  ierr = PetscBagView(usr->bag,viewer);CHKERRQ(ierr);

  fprintf(fp,"    return data\n\n");
  fclose(fp);

  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// Load Parameters
// ---------------------------------------
PetscErrorCode LoadParametersFromFile(void *ctx) 
{
  UsrData        *usr = (UsrData*)ctx;
  char           fout[FNAME_LENGTH];
  PetscViewer    viewer;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/parameters.pbin",usr->par->fdir_out);
  ierr = PetscViewerBinaryOpen(usr->comm,fout,FILE_MODE_READ,&viewer);CHKERRQ(ierr);

  // parameters - scal
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->scal->x,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->scal->eta,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->scal->rho,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->scal->v,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->scal->t,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->scal->DT,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->scal->tau,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->scal->kappa,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->scal->kT,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->scal->kphi,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->scal->Gamma,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);

  // parameters - nd
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->L,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->H,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->xmin,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->zmin,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->Hs,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->Vext,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->Vin,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->Tbot,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->Ttop,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->eta_min,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->eta_max,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->eta_K,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->istep,1,NULL,PETSC_INT);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->t,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->dt,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->tmax,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->dtmax,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);

  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->delta,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->R,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->Ra,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->Gamma,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);

  // material properties ?
  
  // save timestep
  PetscInt    tstep, tout;
  PetscScalar tmax, dtmax;

  tstep = usr->par->tstep;
  tout  = usr->par->tout;
  tmax  = usr->par->tmax;
  dtmax = usr->par->dtmax;

  // read bag
  ierr = PetscBagLoad(viewer,usr->bag);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  usr->par->tstep = tstep;
  usr->par->tout  = tout;
  usr->par->tmax = tmax;
  usr->par->dtmax= dtmax;

  // non-dimensionalize necessary params
  usr->nd->tmax  = nd_param(usr->par->tmax*SEC_YEAR,usr->scal->t);
  usr->nd->dtmax = nd_param(usr->par->dtmax*SEC_YEAR,usr->scal->t);

  PetscFunctionReturn(0);
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
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  usr->plasticity = PETSC_TRUE; 
  usr->nd->istep = usr->par->restart;
  ierr = PetscSNPrintf(usr->par->fdir_out,sizeof(usr->par->fdir_out),"Timestep%d",usr->nd->istep);

  // load time data
  ierr = LoadParametersFromFile(usr);CHKERRQ(ierr);

  // correct restart variable from bag
  usr->par->restart = usr->nd->istep;
  ierr = InputPrintData(usr);CHKERRQ(ierr);

  // load vector data
  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_xPV_ts%d",usr->par->fdir_out,usr->nd->istep);
  ierr = DMStagReadBinaryPython(&dm,&x,fout);CHKERRQ(ierr);
  ierr = VecCopy(x,usr->xPV);CHKERRQ(ierr);
  ierr = VecCopy(x,fdPV->xguess);CHKERRQ(ierr);
  ierr = VecDestroy(&x); CHKERRQ(ierr);
  ierr = DMDestroy(&dm); CHKERRQ(ierr);

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_xT_ts%d",usr->par->fdir_out,usr->nd->istep);
  ierr = DMStagReadBinaryPython(&dm,&x,fout);CHKERRQ(ierr);
  ierr = VecCopy(x,usr->xT);CHKERRQ(ierr);
  ierr = VecCopy(x,fdT->xguess);CHKERRQ(ierr);
  ierr = VecDestroy(&x); CHKERRQ(ierr);
  ierr = DMDestroy(&dm); CHKERRQ(ierr);

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_xphi_ts%d",usr->par->fdir_out,usr->nd->istep);
  ierr = DMStagReadBinaryPython(&dm,&x,fout);CHKERRQ(ierr);
  ierr = VecCopy(x,usr->xphi);CHKERRQ(ierr);
  ierr = VecDestroy(&x); CHKERRQ(ierr);
  ierr = DMDestroy(&dm); CHKERRQ(ierr);

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_resPV_ts%d",usr->par->fdir_out,usr->nd->istep);
  ierr = DMStagReadBinaryPython(&dm,&x,fout);CHKERRQ(ierr);
  ierr = VecCopy(x,fdPV->r);CHKERRQ(ierr);
  ierr = VecDestroy(&x); CHKERRQ(ierr);
  ierr = DMDestroy(&dm); CHKERRQ(ierr);

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_resT_ts%d",usr->par->fdir_out,usr->nd->istep);
  ierr = DMStagReadBinaryPython(&dm,&x,fout);CHKERRQ(ierr);
  ierr = VecCopy(x,fdT->r);CHKERRQ(ierr);
  ierr = VecDestroy(&x); CHKERRQ(ierr);
  ierr = DMDestroy(&dm); CHKERRQ(ierr);

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_resphi_ts%d",usr->par->fdir_out,usr->nd->istep);
  ierr = DMStagReadBinaryPython(&dm,&x,fout);CHKERRQ(ierr);
  ierr = VecCopy(x,fdphi->r);CHKERRQ(ierr);
  ierr = VecDestroy(&x); CHKERRQ(ierr);
  ierr = DMDestroy(&dm); CHKERRQ(ierr);

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_xDP_ts%d",usr->par->fdir_out,usr->nd->istep);
  ierr = DMStagReadBinaryPython(&dm,&x,fout);CHKERRQ(ierr);
  ierr = VecCopy(x,usr->xDP_old);CHKERRQ(ierr);
  ierr = VecDestroy(&x); CHKERRQ(ierr);
  ierr = DMDestroy(&dm); CHKERRQ(ierr);

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_xtau_ts%d",usr->par->fdir_out,usr->nd->istep);
  ierr = DMStagReadBinaryPython(&dm,&x,fout);CHKERRQ(ierr);
  ierr = VecCopy(x,usr->xtau_old);CHKERRQ(ierr);
  ierr = VecDestroy(&x); CHKERRQ(ierr);
  ierr = DMDestroy(&dm); CHKERRQ(ierr);

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_xstrain_ts%d",usr->par->fdir_out,usr->nd->istep);
  ierr = DMStagReadBinaryPython(&dm,&x,fout);CHKERRQ(ierr);
  ierr = VecCopy(x,usr->xstrain);CHKERRQ(ierr);
  ierr = VecDestroy(&x); CHKERRQ(ierr);
  ierr = DMDestroy(&dm); CHKERRQ(ierr);
  
  // markers - read XDMF file
  const char     *fieldname[] = {"id"};
  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_pic_ts%d",usr->par->fdir_out,usr->nd->istep);
  ierr = DMSwarmReadBinaryXDMF_Seq(usr->dmswarm,fout,1,fieldname); CHKERRQ(ierr);
  ierr = UpdateMarkerPhaseFractions(usr->dmswarm,usr->dmMPhase,usr->xMPhase,usr);CHKERRQ(ierr);

  // initialize guess and previous solution in fdT
  ierr = FDPDEAdvDiffGetPrevSolution(fdT,&xTprev);CHKERRQ(ierr);
  ierr = VecCopy(usr->xT,xTprev);CHKERRQ(ierr);
  ierr = FDPDEGetSolutionGuess(fdT,&xTguess);CHKERRQ(ierr);
  ierr = VecCopy(xTprev,xTguess);CHKERRQ(ierr);
  ierr = VecDestroy(&xTprev);CHKERRQ(ierr);
  ierr = VecDestroy(&xTguess);CHKERRQ(ierr);

  ierr = FDPDEAdvDiffGetPrevCoefficient(fdT,&xTcoeffprev);CHKERRQ(ierr);
  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_xTcoeff_ts%d",usr->par->fdir_out,usr->nd->istep);
  ierr = DMStagReadBinaryPython(&dm,&x,fout);CHKERRQ(ierr);
  ierr = VecCopy(x,xTcoeffprev);CHKERRQ(ierr);
  ierr = VecDestroy(&xTcoeffprev);CHKERRQ(ierr);
  ierr = VecDestroy(&x); CHKERRQ(ierr);
  ierr = DMDestroy(&dm); CHKERRQ(ierr);

  // initialize guess and previous solution in fdphi
  ierr = FDPDEAdvDiffGetPrevSolution(fdphi,&xphiprev);CHKERRQ(ierr);
  ierr = VecCopy(usr->xphi,xphiprev);CHKERRQ(ierr);
  ierr = FDPDEGetSolutionGuess(fdphi,&xphiguess);CHKERRQ(ierr);
  ierr = VecCopy(xphiprev,xphiguess);CHKERRQ(ierr);
  ierr = VecDestroy(&xphiprev);CHKERRQ(ierr);
  ierr = VecDestroy(&xphiguess);CHKERRQ(ierr);

  ierr = FDPDEAdvDiffGetPrevCoefficient(fdphi,&xphicoeffprev);CHKERRQ(ierr);
  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_xphicoeff_ts%d",usr->par->fdir_out,usr->nd->istep);
  ierr = DMStagReadBinaryPython(&dm,&x,fout);CHKERRQ(ierr);
  ierr = VecCopy(x,xphicoeffprev);CHKERRQ(ierr);
  ierr = VecDestroy(&xphicoeffprev);CHKERRQ(ierr);
  ierr = VecDestroy(&x); CHKERRQ(ierr);
  ierr = DMDestroy(&dm); CHKERRQ(ierr);

  // Output load conditions
  ierr = DoOutput(fdPV,fdT,fdphi,usr);CHKERRQ(ierr);

  PetscFunctionReturn(0);
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
  char           fname[FNAME_LENGTH],str[FNAME_LENGTH];
  PetscInt       i,j,nm;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  // Read XDMF file and get array of data
  ierr = PetscSNPrintf(fname,sizeof(fname),"%s.xmf",fout);
  fp = fopen(fname, "r" );
  fgets(str,FNAME_LENGTH,fp);
  fgets(str,FNAME_LENGTH,fp);
  fgets(str,FNAME_LENGTH,fp);
  fgets(str,FNAME_LENGTH,fp);
  fgets(str,FNAME_LENGTH,fp); // line containing nmark

  char tmp[256], *p;
  for (i=0; i<strlen(str); i++) {
    j=0;
    while(str[i]>='0' && str[i]<='9'){
     tmp[j]=str[i];
     i++; j++;
    }
  }
  nm = (int)strtol(tmp, &p, 10);

  fgets(str,FNAME_LENGTH,fp);
  fgets(str,FNAME_LENGTH,fp);
  fclose(fp);

  // allocate memory to arrays
  PetscScalar  *x, *z, *id;
	size_t      sz;
  PetscViewer v;

  sz = (size_t)nm*sizeof(PetscScalar);
	ierr = PetscMalloc(sz, &x); CHKERRQ(ierr);
  ierr = PetscMalloc(sz, &z); CHKERRQ(ierr);
  ierr = PetscMalloc(sz, &id);CHKERRQ(ierr);

  // get binary data from .pbin
  ierr = PetscSNPrintf(fname,sizeof(fname),"%s._swarm_fields.pbin",fout);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,fname,FILE_MODE_READ,&v);CHKERRQ(ierr);

  // topology
  for (i=0; i<nm; i++) {
    PetscInt pvertex[3]; // pvertex[0] = 1; pvertex[1] = 1; pvertex[2] = i;
    ierr = PetscViewerBinaryRead(v,pvertex,3,NULL,PETSC_INT);CHKERRQ(ierr);
  }

  // coordinates
  for (i=0; i<nm; i++) {
    PetscScalar xp[2];
    ierr = PetscViewerBinaryRead(v,xp,2,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
    x[i] = xp[0]; z[i] = xp[1];
  }

  // id
  for (i=0; i<nm; i++) {
    ierr = PetscViewerBinaryRead(v,&id[i],1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  }

  ierr = PetscViewerDestroy(&v);CHKERRQ(ierr);

  // set swarm size
  ierr = DMSwarmSetLocalSizes(dmswarm,nm,0); CHKERRQ(ierr);
  
  // Populate dmswarm 'id' with new data
  PetscScalar *pcoor,*pfield,*pfield0, *pfield1, *pfield2,*pfield3, *pfield4, *pfield5;
  ierr = DMSwarmGetField(dmswarm,DMSwarmPICField_coor,NULL,NULL,(void**)&pcoor);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dmswarm,"id",NULL,NULL,(void**)&pfield);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dmswarm,"id0",NULL,NULL,(void**)&pfield0);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dmswarm,"id1",NULL,NULL,(void**)&pfield1);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dmswarm,"id2",NULL,NULL,(void**)&pfield2);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dmswarm,"id3",NULL,NULL,(void**)&pfield3);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dmswarm,"id4",NULL,NULL,(void**)&pfield4);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dmswarm,"id5",NULL,NULL,(void**)&pfield5);CHKERRQ(ierr);

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
  
  ierr = DMSwarmRestoreField(dmswarm,"id",NULL,NULL,(void**)&pfield);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(dmswarm,"id0",NULL,NULL,(void**)&pfield0);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(dmswarm,"id1",NULL,NULL,(void**)&pfield1);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(dmswarm,"id2",NULL,NULL,(void**)&pfield2);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(dmswarm,"id3",NULL,NULL,(void**)&pfield3);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(dmswarm,"id4",NULL,NULL,(void**)&pfield4);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(dmswarm,"id5",NULL,NULL,(void**)&pfield5);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(dmswarm,DMSwarmPICField_coor,NULL,NULL,(void**)&pcoor);CHKERRQ(ierr);

   // Migrate swarm - assign cells and sub-domanin to points
  ierr = DMSwarmMigrate(dmswarm,PETSC_TRUE);CHKERRQ(ierr);

  // free arrays
  ierr = PetscFree(x); CHKERRQ(ierr);
  ierr = PetscFree(z); CHKERRQ(ierr);
  ierr = PetscFree(id); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// UpdateStrainRates
// ---------------------------------------
PetscErrorCode UpdateStrainRates(DM dm, Vec x, void *ctx)
{
  UsrData        *usr = (UsrData*) ctx;
  DM             dmeps;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz, idx, ii;
  PetscInt       iprev, inext, icenter;
  PetscScalar    ***xx;
  PetscScalar    **coordx,**coordz;
  Vec            xeps, xepslocal,xlocal;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  dmeps = usr->dmeps;
  xeps  = usr->xeps;

  // Local vectors
  ierr = DMCreateLocalVector (dmeps,&xepslocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dmeps,xepslocal,&xx); CHKERRQ(ierr);

  ierr = DMGetLocalVector(dm,&xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm,x,INSERT_VALUES,xlocal); CHKERRQ(ierr);

  // Get domain corners
  ierr = DMStagGetGlobalSizes(dmeps, &Nx, &Nz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dmeps, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  // Get dm coordinates array
  ierr = DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,LEFT,&iprev);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,RIGHT,&inext);CHKERRQ(ierr);

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      DMStagStencil  pointC, pointN[4];
      PetscScalar    epsIIc, exxc, ezzc, exzc, epsIIn[4], exxn[4], ezzn[4], exzn[4];

      // Strain rates: center
      pointC.i = i; pointC.j = j; pointC.loc = ELEMENT; pointC.c = 0;
      ierr = DMStagGetPointStrainRates(dm,xlocal,1,&pointC,&epsIIc,&exxc,&ezzc,&exzc); CHKERRQ(ierr);

      ierr = DMStagGetLocationSlot(dmeps,ELEMENT,0,&idx); CHKERRQ(ierr); xx[j][i][idx] = exxc;
      ierr = DMStagGetLocationSlot(dmeps,ELEMENT,1,&idx); CHKERRQ(ierr); xx[j][i][idx] = ezzc;
      ierr = DMStagGetLocationSlot(dmeps,ELEMENT,2,&idx); CHKERRQ(ierr); xx[j][i][idx] = exzc;
      ierr = DMStagGetLocationSlot(dmeps,ELEMENT,3,&idx); CHKERRQ(ierr); xx[j][i][idx] = epsIIc;

      // Strain rates: corner
      pointN[0].i = i; pointN[0].j = j; pointN[0].loc = DOWN_LEFT;  pointN[0].c = 0;
      pointN[1].i = i; pointN[1].j = j; pointN[1].loc = DOWN_RIGHT; pointN[1].c = 0;
      pointN[2].i = i; pointN[2].j = j; pointN[2].loc = UP_LEFT;    pointN[2].c = 0;
      pointN[3].i = i; pointN[3].j = j; pointN[3].loc = UP_RIGHT;   pointN[3].c = 0;
      ierr = DMStagGetPointStrainRates(dm,xlocal,4,pointN,epsIIn,exxn,ezzn,exzn); CHKERRQ(ierr);

      if (i==0) { // boundaries
        pointC.i = i; pointC.j = j; pointC.loc = ELEMENT; pointC.c = 0;
        ierr = DMStagGetPointStrainRates(dm,xlocal,1,&pointC,&epsIIc,&exxc,&ezzc,&exzc); CHKERRQ(ierr);
        ezzn[0] = ezzc;
        exxn[0] = exxc;
      }

      if (i==Nx-1) { // boundaries
        pointC.i = i; pointC.j = j; pointC.loc = ELEMENT; pointC.c = 0;
        ierr = DMStagGetPointStrainRates(dm,xlocal,1,&pointC,&epsIIc,&exxc,&ezzc,&exzc); CHKERRQ(ierr);
        ezzn[1] = ezzc;
        exxn[1] = exxc;
      }

      if (j==0) { // boundaries
        pointC.i = i; pointC.j = j; pointC.loc = ELEMENT; pointC.c = 0;
        ierr = DMStagGetPointStrainRates(dm,xlocal,1,&pointC,&epsIIc,&exxc,&ezzc,&exzc); CHKERRQ(ierr);
        exxn[0] = exxc;
        ezzn[0] = ezzc;
      }

      if (j==Nz-1) { // boundaries
        pointC.i = i; pointC.j = j; pointC.loc = ELEMENT; pointC.c = 0;
        ierr = DMStagGetPointStrainRates(dm,xlocal,1,&pointC,&epsIIc,&exxc,&ezzc,&exzc); CHKERRQ(ierr);
        exxn[2] = exxc;
        ezzn[2] = ezzc;
      }

      if ((i==Nx-1) && (j==Nz-1)) { // boundaries
        pointC.i = i; pointC.j = j; pointC.loc = ELEMENT; pointC.c = 0;
        ierr = DMStagGetPointStrainRates(dm,xlocal,1,&pointC,&epsIIc,&exxc,&ezzc,&exzc); CHKERRQ(ierr);
        exxn[3] = exxc;
        ezzn[3] = ezzc;
      }

      if ((i==0) || (i==Nx-1) || (j==0) || (j==Nz-1)) { // boundaries
        for (ii = 0; ii < 4; ii++) {
          epsIIn[ii] = PetscPowScalar(0.5*(exxn[ii]*exxn[ii] + ezzn[ii]*ezzn[ii] + 2.0*exzn[ii]*exzn[ii]),0.5);
        }
      }

      ierr = DMStagGetLocationSlot(dmeps,DOWN_LEFT,0,&idx); CHKERRQ(ierr); xx[j][i][idx] = exxn[0];
      ierr = DMStagGetLocationSlot(dmeps,DOWN_LEFT,1,&idx); CHKERRQ(ierr); xx[j][i][idx] = ezzn[0];
      ierr = DMStagGetLocationSlot(dmeps,DOWN_LEFT,2,&idx); CHKERRQ(ierr); xx[j][i][idx] = exzn[0];
      ierr = DMStagGetLocationSlot(dmeps,DOWN_LEFT,3,&idx); CHKERRQ(ierr); xx[j][i][idx] = epsIIn[0];

      ierr = DMStagGetLocationSlot(dmeps,DOWN_RIGHT,0,&idx); CHKERRQ(ierr); xx[j][i][idx] = exxn[1];
      ierr = DMStagGetLocationSlot(dmeps,DOWN_RIGHT,1,&idx); CHKERRQ(ierr); xx[j][i][idx] = ezzn[1];
      ierr = DMStagGetLocationSlot(dmeps,DOWN_RIGHT,2,&idx); CHKERRQ(ierr); xx[j][i][idx] = exzn[1];
      ierr = DMStagGetLocationSlot(dmeps,DOWN_RIGHT,3,&idx); CHKERRQ(ierr); xx[j][i][idx] = epsIIn[1];

      ierr = DMStagGetLocationSlot(dmeps,UP_LEFT,0,&idx); CHKERRQ(ierr); xx[j][i][idx] = exxn[2];
      ierr = DMStagGetLocationSlot(dmeps,UP_LEFT,1,&idx); CHKERRQ(ierr); xx[j][i][idx] = ezzn[2];
      ierr = DMStagGetLocationSlot(dmeps,UP_LEFT,2,&idx); CHKERRQ(ierr); xx[j][i][idx] = exzn[2];
      ierr = DMStagGetLocationSlot(dmeps,UP_LEFT,3,&idx); CHKERRQ(ierr); xx[j][i][idx] = epsIIn[2];

      ierr = DMStagGetLocationSlot(dmeps,UP_RIGHT,0,&idx); CHKERRQ(ierr); xx[j][i][idx] = exxn[3];
      ierr = DMStagGetLocationSlot(dmeps,UP_RIGHT,1,&idx); CHKERRQ(ierr); xx[j][i][idx] = ezzn[3];
      ierr = DMStagGetLocationSlot(dmeps,UP_RIGHT,2,&idx); CHKERRQ(ierr); xx[j][i][idx] = exzn[3];
      ierr = DMStagGetLocationSlot(dmeps,UP_RIGHT,3,&idx); CHKERRQ(ierr); xx[j][i][idx] = epsIIn[3];
    }
  }

  // Restore arrays
  ierr = DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);

  // Restore and map local to global
  ierr = DMStagVecRestoreArray(dmeps,xepslocal,&xx); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dmeps,xepslocal,INSERT_VALUES,xeps); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dmeps,xepslocal,INSERT_VALUES,xeps); CHKERRQ(ierr);
  ierr = VecDestroy(&xepslocal); CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(dm, &xlocal ); CHKERRQ(ierr);

  PetscFunctionReturn(0);
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
  PetscErrorCode ierr;

  dt = usr->nd->dt; // dim or nd?
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  // Create local vector
  ierr = DMCreateLocalVector(dm, &lamlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,lam,INSERT_VALUES,lamlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm,lam,INSERT_VALUES,lamlocal);CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm, lamlocal, &_lam); CHKERRQ(ierr);

  ierr = DMCreateLocalVector(dm, &dotlamlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm, dotlam, INSERT_VALUES, dotlamlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm, dotlamlocal, &_dotlam); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm,DMSTAG_ELEMENT,0,&iP);CHKERRQ(ierr);

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      _lam[j][i][iP] += _dotlam[j][i][iP] * dt;
    }
  }

  // Restore arrays, local vectors
  ierr = DMStagVecRestoreArray(dm,lamlocal,&_lam);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm,lamlocal,INSERT_VALUES,lam); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,lamlocal,INSERT_VALUES,lam); CHKERRQ(ierr);
  ierr = VecDestroy(&lamlocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArray(dm,dotlamlocal,&_dotlam);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &dotlamlocal ); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// GetMatPhaseFraction
// ---------------------------------------
PetscErrorCode GetMatPhaseFraction(PetscInt i, PetscInt j, PetscScalar ***xwt, PetscInt *iwtc, PetscInt n, PetscScalar *wt)
{ 
  PetscInt ii;
  PetscFunctionBegin;
  for (ii = 0; ii <n; ii++) {
    wt[ii] = xwt[j][i][iwtc[ii]];
  }
  PetscFunctionReturn(0);
}

// ---------------------------------------
PetscErrorCode GetCornerAvgFromCenter(PetscScalar *Ac, PetscScalar *Acorner)
{
  PetscFunctionBegin;
  Acorner[0] = (Ac[0]+Ac[1]+Ac[3]+Ac[5])*0.25;
  Acorner[1] = (Ac[0]+Ac[2]+Ac[3]+Ac[6])*0.25;
  Acorner[2] = (Ac[0]+Ac[1]+Ac[4]+Ac[7])*0.25;
  Acorner[3] = (Ac[0]+Ac[2]+Ac[4]+Ac[8])*0.25;
  PetscFunctionReturn(0);
}

// ---------------------------------------
PetscErrorCode Get9PointCenterValues(PetscInt i, PetscInt j, PetscInt idx, PetscInt Nx, PetscInt Nz, PetscScalar ***xx, PetscScalar *x)
{
  PetscInt  im, jm, ip, jp;
  PetscFunctionBegin;
    // get porosity, p, Plith, T - center
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
  PetscFunctionReturn(0);
}

// ---------------------------------------
PetscErrorCode GetTensorPointValues(PetscInt i, PetscInt j, PetscInt *idx, PetscScalar ***xx, PetscScalar *x)
{
  PetscFunctionBegin;
    x[0] = xx[j][i][idx[0]]; // xx
    x[1] = xx[j][i][idx[1]]; // zz
    x[2] = xx[j][i][idx[2]]; // xz
    x[3] = xx[j][i][idx[3]]; // II
  PetscFunctionReturn(0);
}