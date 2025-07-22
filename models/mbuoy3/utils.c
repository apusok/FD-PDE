#include "mbuoy3.h"
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

// ---------------------------------------
// SetInitialConditions
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "SetInitialConditions"
PetscErrorCode SetInitialConditions(FDPDE fdPV, FDPDE fdHC, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  DM             dmP, dmHCcoeff;
  Vec            xP, xPprev, xHCprev, xHCguess, xHCcoeff, xHCcoeffprev, xEnth;
  char           fout[FNAME_LENGTH];
  PetscFunctionBeginUser;

  // corner flow model for PV
  PetscCall(CornerFlow_MOR(usr));

  // half-space cooling model - initialize H, C
  PetscCall(HalfSpaceCooling_MOR(usr));

  PetscCall(PetscSNPrintf(usr->par->fdir_out,sizeof(usr->par->fdir_out),"Timestep%d",usr->nd->istep));
  PetscCall(CreateDirectory(usr->par->fdir_out));
  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_xHC_HS_ts%d",usr->par->fdir_out,usr->nd->istep));
  PetscCall(DMStagViewBinaryPython(usr->dmHC,usr->xHC,fout));

  // Update lithostatic pressure
  PetscCall(FDPDEEnthalpyGetPressure(fdHC,&dmP,&xP));
  PetscCall(UpdateLithostaticPressure(dmP,xP,usr));
  PetscCall(FDPDEEnthalpyGetPrevPressure(fdHC,&xPprev));
  PetscCall(VecCopy(xP,xPprev));
  PetscCall(VecDestroy(&xP));
  PetscCall(VecDestroy(&xPprev));
  PetscCall(DMDestroy(&dmP));

  // Update Enthalpy diagnostics
  PetscCall(FDPDEEnthalpyUpdateDiagnostics(fdHC,usr->dmHC,usr->xHC,&usr->dmEnth,&usr->xEnth)); 
  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_xEnth_HS_ts%d",usr->par->fdir_out,usr->nd->istep));
  PetscCall(DMStagViewBinaryPython(usr->dmEnth,usr->xEnth,fout));

  // Correct H-S*phi and C=Cs to ensure phi=phi*phi_init
  PetscCall(CorrectInitialHCZeroPorosity(usr->dmEnth,usr->xEnth,usr));

  if (usr->par->initial_bulk_comp) { // initial bulk composition
    PetscCall(CorrectInitialHCBulkComposition(usr));
  }

  // Update Enthalpy again for visualization and to initialize xEnthold
  PetscCall(FDPDEEnthalpyUpdateDiagnostics(fdHC,usr->dmHC,usr->xHC,NULL,&xEnth)); 
  PetscCall(VecCopy(xEnth,usr->xEnth));
  PetscCall(VecDestroy(&xEnth));

  PetscCall(VecDuplicate(usr->xEnth,&usr->xEnthold));
  PetscCall(VecCopy(usr->xEnth,usr->xEnthold));

  // Update fluid velocity to zero and v=vs
  PetscCall(ComputeFluidAndBulkVelocity(usr->dmPV,usr->xPV,usr->dmEnth,usr->xEnth,usr->dmVel,usr->xVel,usr));

  // Initialize guess and previous solution in fdHC
  PetscCall(FDPDEEnthalpyGetPrevSolution(fdHC,&xHCprev));
  PetscCall(VecCopy(usr->xHC,xHCprev));
  PetscCall(FDPDEGetSolutionGuess(fdHC,&xHCguess));
  PetscCall(VecCopy(xHCprev,xHCguess));
  PetscCall(VecDestroy(&xHCprev));
  PetscCall(VecDestroy(&xHCguess));

  // Set initial coefficient structure
  PetscCall(FDPDEGetCoefficient(fdHC,&dmHCcoeff,&xHCcoeff));
  PetscCall(FDPDEEnthalpyGetPrevCoefficient(fdHC,&xHCcoeffprev));
  PetscCall(FormCoefficient_HC(fdHC,usr->dmHC,usr->xHC,dmHCcoeff,xHCcoeffprev,usr));
  PetscCall(VecCopy(xHCcoeffprev,xHCcoeff));

  // Output prev coefficient
  PetscCall(PetscSNPrintf(usr->par->fdir_out,sizeof(usr->par->fdir_out),"Timestep%d",usr->nd->istep));
  PetscCall(CreateDirectory(usr->par->fdir_out));
  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_xHCcoeff_ts%d",usr->par->fdir_out,usr->nd->istep));
  PetscCall(DMStagViewBinaryPython(dmHCcoeff,xHCcoeffprev,fout));
  PetscCall(VecDestroy(&xHCcoeffprev));

  // Output initial conditions
  PetscCall(DoOutput(fdPV,fdHC,usr));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// MOR Corner flow model for PV
// ---------------------------------------
PetscErrorCode CornerFlow_MOR(void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz, idx;
  PetscInt       iprev, inext, icenter;
  PetscScalar    rangle, radalpha, sina, eta0, C1, C4, u0;
  PetscScalar    ***xx;
  PetscScalar    **coordx,**coordz;
  Vec            x, xlocal;
  DM             dm;
  PetscFunctionBeginUser;

  dm = usr->dmPV;
  x  = usr->xPV;

  // calculate ridge parameters
  rangle = 0.0;
  radalpha = rangle*PETSC_PI/180;
  sina = PetscSinScalar(radalpha);
  C1   =  2.0*sina*sina/(PETSC_PI-2.0*radalpha-PetscSinScalar(2.0*radalpha));
  C4   = -2.0/(PETSC_PI-2.0*radalpha-PetscSinScalar(2.0*radalpha));
  u0   = usr->nd->U0;
  eta0 = nd_param(usr->par->eta0,usr->scal->eta);

  PetscCall(DMStagGetGlobalSizes(dm, &Nx, &Nz,NULL));
  PetscCall(DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 
  PetscCall(DMCreateLocalVector(dm, &xlocal)); 
  PetscCall(DMStagVecGetArray(dm, xlocal, &xx)); 

// Get dm coordinates array
  PetscCall(DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter)); 
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm,LEFT,&iprev));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm,RIGHT,&inext)); 

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      PetscScalar xp, zp, v[2], p;

      // Vx
      xp = coordx[i][iprev ]; 
      zp = coordz[j][icenter];
      evaluate_CornerFlow_MOR(C1, C4, u0, eta0, xp, zp, v, &p);
      PetscCall(DMStagGetLocationSlot(dm, LEFT, 0, &idx)); 
      xx[j][i][idx] = v[0];

      if (i == Nx-1) {
        xp = coordx[i][inext  ];
        zp = coordz[j][icenter];
        evaluate_CornerFlow_MOR(C1, C4, u0, eta0, xp, zp, v, &p);
        PetscCall(DMStagGetLocationSlot(dm, RIGHT, 0, &idx)); 
        xx[j][i][idx] = v[0];
      }

      // Vz
      xp = coordx[i][icenter];
      zp = coordz[j][iprev  ];
      evaluate_CornerFlow_MOR(C1, C4, u0, eta0, xp, zp, v, &p);
      PetscCall(DMStagGetLocationSlot(dm, DOWN, 0, &idx)); 
      xx[j][i][idx] = v[1];

      if (j == Nz-1) {
        xp = coordx[i][icenter];
        zp = coordz[j][inext  ];
        evaluate_CornerFlow_MOR(C1, C4, u0, eta0, xp, zp, v, &p);
        PetscCall(DMStagGetLocationSlot(dm, UP, 0, &idx)); 
        xx[j][i][idx] = v[1];
      }
    
      // P
      xp = coordx[i][icenter];
      zp = coordz[j][icenter];
      evaluate_CornerFlow_MOR(C1, C4, u0, eta0, xp, zp, v, &p);
      PetscCall(DMStagGetLocationSlot(dm, ELEMENT, 0, &idx)); 
      xx[j][i][idx] = p;

      // no need to initialize pc here
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
// Half-Space Cooling model for MOR
// ---------------------------------------
PetscErrorCode HalfSpaceCooling_MOR(void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz, iH, iC, icenter;
  PetscScalar    **coordx,**coordz, ***xx, Cs0, Tm, xmor;
  Vec            x, xlocal;
  DM             dm;
  PetscFunctionBeginUser;

  dm  = usr->dmHC;
  x   = usr->xHC;
  Cs0 = usr->par->C0;
  Tm  = (usr->par->Tp-T_KELVIN)*exp(-usr->nd->A*usr->nd->zmin)+T_KELVIN;
  xmor = usr->nd->xmor;

  PetscCall(DMStagGetGlobalSizes(dm, &Nx, &Nz,NULL));
  PetscCall(DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 
  PetscCall(DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter)); 
  PetscCall(DMStagGetLocationSlot(dm, ELEMENT, 0, &iH)); 
  PetscCall(DMStagGetLocationSlot(dm, ELEMENT, 1, &iC)); 

  PetscCall(DMCreateLocalVector(dm, &xlocal)); 
  PetscCall(DMStagVecGetArray(dm, xlocal, &xx)); 

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      PetscScalar T, nd_T, age, Ta;

      // half-space cooling temperature
      age  = dim_param(coordx[i][icenter],usr->scal->x)/dim_param(usr->nd->U0,usr->scal->v);

      // shift T from the axis if required with xmor
      age  = dim_param(fabs(coordx[i][icenter])-xmor,usr->scal->x)/dim_param(usr->nd->U0,usr->scal->v);
      if (age <= 0.0) T = Tm; 
      else T = HalfSpaceCoolingTemp(Tm,usr->par->Ts,-dim_param(coordz[j][icenter],usr->scal->x),usr->par->kappa,age,usr->par->hs_factor); 

      // check adiabat in the mantle
      Ta  = (usr->par->Tp-T_KELVIN)*exp(-usr->nd->A*coordz[j][icenter])+T_KELVIN;
      //if (T>Ta) T = (3.0*Ta+T)*0.25;
      nd_T = (T - usr->par->T0)/usr->par->DT;

      // enthalpy H = S*phi+T (phi=0)
      xx[j][i][iH] = nd_T;

      // initial bulk composition C0 = Cs0 (phi=0)
      xx[j][i][iC] = (Cs0-usr->par->C0)/usr->par->DC;
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
// Update Lithostatic pressure
// ---------------------------------------
PetscErrorCode UpdateLithostaticPressure(DM dm, Vec x, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz, idx, icenter;
  PetscScalar    **coordx,**coordz, ***xx;
  Vec            xlocal;
  PetscFunctionBeginUser;

  PetscCall(DMStagGetGlobalSizes(dm, &Nx, &Nz,NULL));
  PetscCall(DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 
  PetscCall(DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter)); 
  PetscCall(DMStagGetLocationSlot(dm, ELEMENT, 0, &idx)); 

  PetscCall(DMCreateLocalVector(dm, &xlocal)); 
  PetscCall(DMStagVecGetArray(dm, xlocal, &xx)); 

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      PetscScalar rho;
      rho  = usr->par->rho0; 
      xx[j][i][idx] = LithostaticPressure(rho,usr->par->drho,coordz[j][icenter]);
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
// Correct initial enthalpy and bulk composition for zero porosity
// ---------------------------------------
PetscErrorCode CorrectInitialHCZeroPorosity(DM dmEnth, Vec xEnth, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  PetscInt       i, j, sx, sz, nx, nz, iH, iC, dm_slot[3];
  PetscScalar    ***xx, ***enth;
  Vec            x, xlocal, xnewlocal;
  DM             dm;
  PetscLogDouble tlog[2];
  PetscFunctionBeginUser;

  PetscTime(&tlog[0]);
  dm  = usr->dmHC;
  x   = usr->xHC;

  PetscCall(DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 
  PetscCall(DMStagGetLocationSlot(dm, ELEMENT, 0, &iH)); 
  PetscCall(DMStagGetLocationSlot(dm, ELEMENT, 1, &iC)); 

  PetscCall(DMGetLocalVector(dm, &xlocal)); 
  PetscCall(DMGlobalToLocal (dm, x, INSERT_VALUES, xlocal)); 
  PetscCall(DMStagVecGetArray(dm, xlocal, &xx)); 

  PetscCall(DMGetLocalVector(dmEnth, &xnewlocal)); 
  PetscCall(DMGlobalToLocal (dmEnth, xEnth, INSERT_VALUES, xnewlocal)); 
  PetscCall(DMStagVecGetArrayRead(dmEnth,xnewlocal,&enth));

  PetscCall(DMStagGetLocationSlot(dmEnth,DMSTAG_ELEMENT,ENTH_ELEMENT_PHI,&dm_slot[0])); 
  PetscCall(DMStagGetLocationSlot(dmEnth,DMSTAG_ELEMENT,ENTH_ELEMENT_CS ,&dm_slot[1])); 
  PetscCall(DMStagGetLocationSlot(dmEnth,DMSTAG_ELEMENT,ENTH_ELEMENT_CF ,&dm_slot[2])); 

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      PetscScalar   xs[3],phi;

      xs[0] = enth[j][i][dm_slot[0]];
      xs[1] = enth[j][i][dm_slot[1]];
      xs[2] = enth[j][i][dm_slot[2]];

      phi = xs[0]*usr->par->phi_init;

      // H + S*(phi-phi0)
      xx[j][i][iH] += usr->nd->S*(phi - xs[0]); 

      // bulk composition
      xx[j][i][iC] = (1.0-phi)*xs[1] + phi*xs[2]; 
    }
  }

  // Restore arrays
  PetscCall(DMStagVecRestoreArray(dm,xlocal,&xx)); 
  PetscCall(DMLocalToGlobalBegin(dm,xlocal,INSERT_VALUES,x)); 
  PetscCall(DMLocalToGlobalEnd  (dm,xlocal,INSERT_VALUES,x)); 
  PetscCall(DMRestoreLocalVector(dm, &xlocal)); 

  PetscCall(DMStagVecRestoreArrayRead(dmEnth,xnewlocal,&enth));
  PetscCall(DMRestoreLocalVector(dmEnth, &xnewlocal)); 

  PetscTime(&tlog[1]);
  if (usr->par->log_info) {
    printf("  CorrectInitialHCZeroPorosity: total                %1.2e\n",tlog[1]-tlog[0]);
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// Correct initial bulk composition in entire domain to be the same as beneath the axis
// ---------------------------------------
PetscErrorCode CorrectInitialHCBulkComposition(void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  PetscInt       i, j, sx, sz, nx, nz, iH, iC;
  PetscScalar    ***xx;
  Vec            x, xlocal;
  DM             dm;
  PetscMPIInt    size;
  PetscFunctionBeginUser;

  dm  = usr->dmHC;
  x   = usr->xHC;

  PetscCall(MPI_Comm_size(usr->comm,&size));
  PetscCall(DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 
  PetscCall(DMStagGetLocationSlot(dm, ELEMENT, 1, &iC)); 

  PetscCall(DMGetLocalVector(dm, &xlocal)); 
  PetscCall(DMGlobalToLocal (dm, x, INSERT_VALUES, xlocal)); 
  PetscCall(DMStagVecGetArray(dm, xlocal, &xx)); 

  PetscScalar *xmor;
  PetscInt *_send, *_recv, irank, s_rank[2], s_neigh[2];
  
  // create data 
  PetscCall(PetscCalloc1(nz,&xmor)); 
  PetscCall(PetscCalloc1(size,&_send)); 
  PetscCall(PetscCalloc1(size,&_recv)); 
  
  for (irank = 0; irank < size; irank++) {
    _send[irank] = -1;
    _recv[irank] = -1;
  }

  s_rank[0] = sx; s_rank[1] = sz;

  // first all procs send/recv start coord to each other
  for (irank = 0; irank < size; irank++) {
    if (irank!=usr->rank) {
      PetscCall(MPI_Send(&s_rank,2,MPI_INT,irank,0,usr->comm));
      PetscCall(MPI_Recv(&s_neigh,2,MPI_INT,irank,0,usr->comm,MPI_STATUS_IGNORE));

      if (s_rank[1]==s_neigh[1]) {
        if (s_rank[0] ==0) _send[irank] = 1;
        if (s_neigh[0]==0) _recv[irank] = 1;
      }
    }
  }

  // save xmor data
  if (sx==0) {
    for (j = sz; j < sz+nz; j++) {
      xmor[j-sz] = xx[j][0][iC]; 
    }
  }

  // send/receiv xmor data
  for (irank = 0; irank < size; irank++) {
    if (irank!=usr->rank) {
      if (_send[irank] == 1) {
        PetscCall(MPI_Send(xmor,nz,MPI_DOUBLE,irank,0,usr->comm));
      }
      if (_recv[irank] == 1) {
        PetscCall(MPI_Recv(xmor,nz,MPI_DOUBLE,irank,0,usr->comm,MPI_STATUS_IGNORE));
      }
    }
  }

  // correct bulk composition as beneath MOR
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      xx[j][i][iC] = xmor[j-sz]; 
    }
  }

  // free data 
  PetscCall(PetscFree(xmor));
  PetscCall(PetscFree(_send));
  PetscCall(PetscFree(_recv));

  // Restore arrays
  PetscCall(DMStagVecRestoreArray(dm,xlocal,&xx)); 
  PetscCall(DMLocalToGlobalBegin(dm,xlocal,INSERT_VALUES,x)); 
  PetscCall(DMLocalToGlobalEnd  (dm,xlocal,INSERT_VALUES,x)); 
  PetscCall(DMRestoreLocalVector(dm, &xlocal)); 

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// Compute Fluid and Bulk velocity from PV and porosity solutions
// ---------------------------------------
PetscErrorCode ComputeFluidAndBulkVelocity(DM dmPV, Vec xPV, DM dmEnth, Vec xEnth, DM dmVel, Vec xVel, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  PetscInt       i, j, ii, sx, sz, nx, nz, Nx, Nz, iprev,inext,icenter;
  PetscScalar    ***xx, dx, dz, k_hat[4];
  PetscScalar    **coordx,**coordz;
  Vec            xVellocal,xPVlocal,xEnthlocal;
  PetscScalar    ***_xPVlocal,***_xEnthlocal;
  PetscInt       pv_slot[6],phi_slot,v_slot[8],iL,iR,iU,iD,iP,iPc;
  PetscLogDouble tlog[2];
  PetscFunctionBeginUser;

  PetscTime(&tlog[0]);
  k_hat[0] = 0.0;
  k_hat[1] = 0.0;
  k_hat[2] = usr->par->k_hat;
  k_hat[3] = usr->par->k_hat;

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

  PetscCall(DMGetLocalVector(dmEnth, &xEnthlocal)); 
  PetscCall(DMGlobalToLocal (dmEnth, xEnth, INSERT_VALUES, xEnthlocal)); 
  PetscCall(DMStagVecGetArrayRead(dmEnth,xEnthlocal,&_xEnthlocal));

  // get slots
  iP = 0; iPc = 1;
  iL = 2; iR  = 3;
  iD = 4; iU  = 5;
  PetscCall(DMStagGetLocationSlot(dmPV,DMSTAG_ELEMENT,PV_ELEMENT_P, &pv_slot[iP])); 
  PetscCall(DMStagGetLocationSlot(dmPV,DMSTAG_ELEMENT,PV_ELEMENT_PC,&pv_slot[iPc])); 
  PetscCall(DMStagGetLocationSlot(dmPV,DMSTAG_LEFT,   PV_FACE_VS,   &pv_slot[iL]));
  PetscCall(DMStagGetLocationSlot(dmPV,DMSTAG_RIGHT,  PV_FACE_VS,   &pv_slot[iR]));
  PetscCall(DMStagGetLocationSlot(dmPV,DMSTAG_DOWN,   PV_FACE_VS,   &pv_slot[iD]));
  PetscCall(DMStagGetLocationSlot(dmPV,DMSTAG_UP,     PV_FACE_VS,   &pv_slot[iU]));
  PetscCall(DMStagGetLocationSlot(dmEnth,DMSTAG_ELEMENT,ENTH_ELEMENT_PHI,&phi_slot));

  PetscCall(DMStagGetLocationSlot(dmVel,DMSTAG_LEFT,  VEL_FACE_VF, &v_slot[0]));
  PetscCall(DMStagGetLocationSlot(dmVel,DMSTAG_RIGHT, VEL_FACE_VF, &v_slot[1]));
  PetscCall(DMStagGetLocationSlot(dmVel,DMSTAG_DOWN,  VEL_FACE_VF, &v_slot[2]));
  PetscCall(DMStagGetLocationSlot(dmVel,DMSTAG_UP,    VEL_FACE_VF, &v_slot[3]));
  PetscCall(DMStagGetLocationSlot(dmVel,DMSTAG_LEFT,  VEL_FACE_V,  &v_slot[4]));
  PetscCall(DMStagGetLocationSlot(dmVel,DMSTAG_RIGHT, VEL_FACE_V,  &v_slot[5]));
  PetscCall(DMStagGetLocationSlot(dmVel,DMSTAG_DOWN,  VEL_FACE_V,  &v_slot[6]));
  PetscCall(DMStagGetLocationSlot(dmVel,DMSTAG_UP,    VEL_FACE_V,  &v_slot[7]));

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      PetscInt    im, jm;
      PetscScalar p[5], Q[5]; 
      PetscScalar vf, K, Bf, vs[4], gradP[4], gradPc[4], phi[4];

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

      p[0] = _xPVlocal[j][i][pv_slot[iPc]];
      if (i == 0   ) im = i; else im = i-1; p[1] = _xPVlocal[j][im][pv_slot[iPc]];
      if (i == Nx-1) im = i; else im = i+1; p[2] = _xPVlocal[j][im][pv_slot[iPc]];
      if (j == 0   ) jm = j; else jm = j-1; p[3] = _xPVlocal[jm][i][pv_slot[iPc]];
      if (j == Nz-1) jm = j; else jm = j+1; p[4] = _xPVlocal[jm][i][pv_slot[iPc]];

      gradPc[0] = (p[0]-p[1])/dx;
      gradPc[1] = (p[2]-p[0])/dx;
      gradPc[2] = (p[0]-p[3])/dz;
      gradPc[3] = (p[4]-p[0])/dz;

      // porosity
      Q[0] = _xEnthlocal[j][i][phi_slot];
      if (i == 0   ) im = i; else im = i-1; Q[1] = _xEnthlocal[j][im][phi_slot];
      if (i == Nx-1) im = i; else im = i+1; Q[2] = _xEnthlocal[j][im][phi_slot];
      if (j == 0   ) jm = j; else jm = j-1; Q[3] = _xEnthlocal[jm][i][phi_slot];
      if (j == Nz-1) jm = j; else jm = j+1; Q[4] = _xEnthlocal[jm][i][phi_slot];

      // porosity on edges
      phi[0] = (Q[1]+Q[0])*0.5; 
      phi[1] = (Q[2]+Q[0])*0.5; 
      phi[2] = (Q[3]+Q[0])*0.5; 
      phi[3] = (Q[4]+Q[0])*0.5; 

      if ((fabs(coordx[i][icenter])<=usr->nd->xmor) && (j==nz+sz-1)) { // dphi/dz=0 just beneath the axis
        phi[3] = usr->par->fextract*phi[2]; // phi[3];
      }
      
      for (ii = 0; ii < 4; ii++) {
        // permeability
        K = Permeability(phi[ii],usr->par->n);

        // fluid buoyancy
        Bf = FluidBuoyancy(0.0,0.0,usr->nd->alpha_s,usr->nd->beta_s);

        // fluid velocity
        vf = FluidVelocity(vs[ii],phi[ii],gradP[ii],gradPc[ii],Bf,K,k_hat[ii]);
        if ((ii==3) && (fabs(coordx[i][icenter])<=usr->nd->xmor) && (j==nz+sz-1)) {
          vf = usr->par->fextract*vf;
        }
        xx[j][i][v_slot[ii]] = vf;

        // bulk velocity
        xx[j][i][v_slot[4+ii]] = BulkVelocity(vs[ii],vf,phi[ii]);
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
  PetscCall(DMStagVecRestoreArrayRead(dmEnth,xEnthlocal,&_xEnthlocal));
  PetscCall(DMRestoreLocalVector(dmEnth, &xEnthlocal)); 

  PetscTime(&tlog[1]);
  if (usr->par->log_info) {
    printf("  ComputeFluidAndBulkVelocity: total                 %1.2e\n",tlog[1]-tlog[0]);
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// Update eta, zeta, permeability, rho, rho_f, rho_s for output
// ---------------------------------------
PetscErrorCode UpdateMaterialProperties(DM dmEnth, Vec xEnth, DM dmmatProp, Vec xmatProp, void *ctx)
{
  UsrData        *usr = (UsrData*) ctx;
  NdParams       *nd;
  Params         *par;
  ScalParams     *scal;
  PetscInt       i, j, sx, sz, nx, nz;
  PetscInt       ii,iphi,iT,iCS,iCF,idx[6];
  PetscScalar    ***xx, ***_xEnthlocal;
  Vec            xmatProplocal,xEnthlocal;
  PetscLogDouble tlog[2];
  PetscFunctionBeginUser;

  PetscTime(&tlog[0]);
  nd  = usr->nd;
  par = usr->par;
  scal= usr->scal;

  PetscCall(DMStagGetCorners(dmmatProp,&sx,&sz,NULL,&nx,&nz,NULL,NULL,NULL,NULL));   
  PetscCall(DMCreateLocalVector(dmmatProp,&xmatProplocal));
  PetscCall(DMStagVecGetArray(dmmatProp, xmatProplocal, &xx)); 

  PetscCall(DMGetLocalVector(dmEnth, &xEnthlocal)); 
  PetscCall(DMGlobalToLocal (dmEnth, xEnth, INSERT_VALUES, xEnthlocal)); 
  PetscCall(DMStagVecGetArrayRead(dmEnth,xEnthlocal,&_xEnthlocal));

  PetscCall(DMStagGetLocationSlot(dmEnth,DMSTAG_ELEMENT,ENTH_ELEMENT_PHI,&iphi));
  PetscCall(DMStagGetLocationSlot(dmEnth,DMSTAG_ELEMENT,ENTH_ELEMENT_T,  &iT));
  PetscCall(DMStagGetLocationSlot(dmEnth,DMSTAG_ELEMENT,ENTH_ELEMENT_CS, &iCS));
  PetscCall(DMStagGetLocationSlot(dmEnth,DMSTAG_ELEMENT,ENTH_ELEMENT_CF, &iCF));

  for (ii = 0; ii < 6; ii++) {
    PetscCall(DMStagGetLocationSlot(dmmatProp,DMSTAG_ELEMENT,ii,&idx[ii])); 
  }

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      PetscScalar   d[6], eta, zeta, K, rho, rhof, rhos, CF, CS, T, phi;

      phi = _xEnthlocal[j][i][iphi];
      T   = _xEnthlocal[j][i][iT  ];
      CS  = _xEnthlocal[j][i][iCS ];
      CF  = _xEnthlocal[j][i][iCF ];

      eta  = ShearViscosity(T*par->DT+par->T0,phi,par->EoR,par->Teta0,par->lambda,nd->eta_min,nd->eta_max,par->visc_shear);
      zeta = BulkViscosity(nd->visc_ratio,T*par->DT+par->T0,phi,par->EoR,par->Teta0,par->phi_min,par->zetaExp,nd->eta_min,nd->eta_max,par->visc_bulk); 
      K    = Permeability(phi,usr->par->n);
      rhos = SolidDensity(par->rho0,par->drho,T,CS,nd->alpha_s,nd->beta_s,par->buoyancy);
      rhof = FluidDensity(par->rho0,par->drho,T,CF,nd->alpha_s,nd->beta_s,par->buoyancy);
      rho  = BulkDensity(rhos,rhof,phi);

      d[0] = eta;
      d[1] = zeta; 
      d[2] = K;
      d[3] = rho;
      d[4] = rhof;
      d[5] = rhos;

      for (ii = 0; ii < 6; ii++) xx[j][i][idx[ii]] = d[ii];
    }
  }

  // Restore arrays
  PetscCall(DMStagVecRestoreArray(dmmatProp,xmatProplocal,&xx)); 
  PetscCall(DMLocalToGlobalBegin(dmmatProp,xmatProplocal,INSERT_VALUES,xmatProp)); 
  PetscCall(DMLocalToGlobalEnd  (dmmatProp,xmatProplocal,INSERT_VALUES,xmatProp)); 
  PetscCall(VecDestroy(&xmatProplocal)); 

  PetscCall(DMStagVecRestoreArrayRead(dmEnth,xEnthlocal,&_xEnthlocal));
  PetscCall(DMRestoreLocalVector(dmEnth,&xEnthlocal)); 

  PetscTime(&tlog[1]);
  if (usr->par->log_info) {
    printf("  UpdateMaterialProperties: total                    %1.2e\n",tlog[1]-tlog[0]);
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// LoadRestartFromFile
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "LoadRestartFromFile"
PetscErrorCode LoadRestartFromFile(FDPDE fdPV, FDPDE fdHC, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  DM             dm, dmEnth, dmP, dmHCcoeff;
  Vec            x, xP, xPprev, xHCprev, xHCguess,xHCcoeff, xHCcoeffprev;
  char           fout[FNAME_LENGTH];
  PetscViewer    viewer;
  PetscFunctionBeginUser;

  usr->nd->istep = usr->par->restart;
  PetscCall(PetscSNPrintf(usr->par->fdir_out,sizeof(usr->par->fdir_out),"Timestep%d",usr->nd->istep));

  // load time data
  PetscCall(LoadParametersFromFile(usr));

  // correct restart variable from bag
  usr->par->restart = usr->nd->istep;
  PetscCall(InputPrintData(usr));

  // load PV data
  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_xPV_ts%d",usr->par->fdir_out,usr->nd->istep));
  PetscCall(DMStagReadBinaryPython(&dm,&x,fout));
  PetscCall(VecCopy(x,usr->xPV));
  PetscCall(VecCopy(x,fdPV->xguess));
  PetscCall(VecDestroy(&x)); 
  PetscCall(DMDestroy(&dm)); 
  
  // load HC data
  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_xHC_ts%d",usr->par->fdir_out,usr->nd->istep));
  PetscCall(DMStagReadBinaryPython(&dm,&x,fout));
  PetscCall(VecCopy(x,usr->xHC));
  PetscCall(VecCopy(x,fdHC->xguess));
  PetscCall(VecDestroy(&x)); 
  PetscCall(DMDestroy(&dm)); 

  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_resPV_ts%d",usr->par->fdir_out,usr->nd->istep));
  PetscCall(DMStagReadBinaryPython(&dm,&x,fout));
  PetscCall(VecCopy(x,fdPV->r));
  PetscCall(VecDestroy(&x)); 
  PetscCall(DMDestroy(&dm)); 

  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_resHC_ts%d",usr->par->fdir_out,usr->nd->istep));
  PetscCall(DMStagReadBinaryPython(&dm,&x,fout));
  PetscCall(VecCopy(x,fdHC->r));
  PetscCall(VecDestroy(&x)); 
  PetscCall(DMDestroy(&dm)); 
  
  // load lithostatic pressure
  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_xPressure_ts%d",usr->par->fdir_out,usr->nd->istep));
  PetscCall(DMStagReadBinaryPython(&dm,&x,fout));
  PetscCall(FDPDEEnthalpyGetPressure(fdHC,&dmP,&xP));
  PetscCall(VecCopy(x,xP));
  PetscCall(VecDestroy(&xP));
  PetscCall(VecDestroy(&x)); 
  PetscCall(DMDestroy(&dm)); 

  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_xPressurePrev_ts%d",usr->par->fdir_out,usr->nd->istep));
  PetscCall(DMStagReadBinaryPython(&dm,&x,fout));
  PetscCall(FDPDEEnthalpyGetPrevPressure(fdHC,&xPprev));
  PetscCall(VecCopy(x,xPprev));
  PetscCall(VecDestroy(&xPprev));
  PetscCall(VecDestroy(&x)); 
  PetscCall(DMDestroy(&dm)); 
  PetscCall(DMDestroy(&dmP));

  // load Enthalpy diagnostics
  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_xEnth_ts%d",usr->par->fdir_out,usr->nd->istep));
  PetscCall(DMStagReadBinaryPython(&dmEnth,&x,fout));
  usr->dmEnth = dmEnth;
  PetscCall(VecDuplicate(x,&usr->xEnth));
  PetscCall(VecDuplicate(x,&usr->xEnthold));
  PetscCall(VecCopy(x,usr->xEnth));
  PetscCall(VecCopy(x,usr->xEnthold)); // this should be loaded from file too
  PetscCall(VecDestroy(&x)); 

  // load velocity
  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_xVel_ts%d",usr->par->fdir_out,usr->nd->istep));
  PetscCall(DMStagReadBinaryPython(&dm,&x,fout));
  PetscCall(VecCopy(x,usr->xVel));
  PetscCall(VecDestroy(&x)); 
  PetscCall(DMDestroy(&dm)); 

  // initialize guess and previous solution in fdHC
  PetscCall(FDPDEEnthalpyGetPrevSolution(fdHC,&xHCprev));
  PetscCall(VecCopy(usr->xHC,xHCprev));
  PetscCall(VecDestroy(&xHCprev));

  // load coefficient structure
  PetscCall(FDPDEGetCoefficient(fdHC,&dmHCcoeff,&xHCcoeff));
  PetscCall(FDPDEEnthalpyGetPrevCoefficient(fdHC,&xHCcoeffprev));

  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_xHCcoeff_ts%d",usr->par->fdir_out,usr->nd->istep));
  PetscCall(DMStagReadBinaryPython(&dm,&x,fout));
  PetscCall(VecCopy(x,xHCcoeffprev));
  PetscCall(VecCopy(x,xHCcoeff));
  PetscCall(VecDestroy(&x)); 
  PetscCall(DMDestroy(&dm)); 
  PetscCall(VecDestroy(&xHCcoeffprev));

  // Output load conditions
  PetscCall(DoOutput(fdPV,fdHC,usr));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// DoOutput
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "DoOutput"
PetscErrorCode DoOutput(FDPDE fdPV, FDPDE fdHC, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  char           fout[FNAME_LENGTH];
  DM             dmPVcoeff, dmHCcoeff,dmP;
  Vec            xPVcoeff, xHCcoeff, xP, xPprev;
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

  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_xHC_ts%d",usr->par->fdir_out,usr->nd->istep));
  PetscCall(DMStagViewBinaryPython(usr->dmHC,usr->xHC,fout));

  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_xEnth_ts%d",usr->par->fdir_out,usr->nd->istep));
  PetscCall(DMStagViewBinaryPython(usr->dmEnth,usr->xEnth,fout));

  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_xVel_ts%d",usr->par->fdir_out,usr->nd->istep));
  PetscCall(DMStagViewBinaryPython(usr->dmVel,usr->xVel,fout));

  if (usr->nd->istep > 0) {
    // coefficients
    PetscCall(FDPDEGetCoefficient(fdHC,&dmHCcoeff,&xHCcoeff));
    PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_xHCcoeff_ts%d",usr->par->fdir_out,usr->nd->istep));
    PetscCall(DMStagViewBinaryPython(dmHCcoeff,xHCcoeff,fout));

    PetscCall(FDPDEGetCoefficient(fdPV,&dmPVcoeff,&xPVcoeff));
    PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_xPVcoeff_ts%d",usr->par->fdir_out,usr->nd->istep));
    PetscCall(DMStagViewBinaryPython(dmPVcoeff,xPVcoeff,fout));

    // material properties eta, permeability, density
    PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_matProp_ts%d",usr->par->fdir_out,usr->nd->istep));
    PetscCall(DMStagViewBinaryPython(usr->dmmatProp,usr->xmatProp,fout));
  }

  // residuals
  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_resPV_ts%d",usr->par->fdir_out,usr->nd->istep));
  PetscCall(DMStagViewBinaryPython(usr->dmPV,fdPV->r,fout));

  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_resHC_ts%d",usr->par->fdir_out,usr->nd->istep));
  PetscCall(DMStagViewBinaryPython(usr->dmHC,fdHC->r,fout));

  // lithostatic pressure
  PetscCall(FDPDEEnthalpyGetPressure(fdHC,&dmP,&xP));
  PetscCall(FDPDEEnthalpyGetPrevPressure(fdHC,&xPprev));

  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_xPressure_ts%d",usr->par->fdir_out,usr->nd->istep));
  PetscCall(DMStagViewBinaryPython(dmP,xP,fout));

  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_xPressurePrev_ts%d",usr->par->fdir_out,usr->nd->istep));
  PetscCall(DMStagViewBinaryPython(dmP,xPprev,fout));

  PetscCall(VecDestroy(&xP));
  PetscCall(VecDestroy(&xPprev));
  PetscCall(DMDestroy(&dmP));

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
// Compute crustal thickness and fluxes out for x<=xMOR
// ---------------------------------------
PetscErrorCode ComputeMeltExtractOutflux(void *ctx) 
{
  UsrData       *usr = (UsrData*) ctx;
  PetscInt       i, j, imor_s, imor_e, sx, sz, nx, nz, Nx, Nz, iprev,inext;
  PetscInt       iphi,iCF,v_slot[2];
  PetscScalar    **coordx,**coordz;
  PetscScalar    out_F, out_C, gout_F, gout_C, phi_max, vfz_max, gphi_max, gvfz_max, full_ridge;
  DM             dmHC, dmVel, dmEnth;
  Vec            xVellocal, xEnthlocal;
  PetscScalar    ***_xEnthlocal,***_xVellocal;
  PetscLogDouble tlog[2];
  PetscFunctionBeginUser;

  PetscTime(&tlog[0]);
  dmHC = usr->dmHC;
  dmVel= usr->dmVel;
  dmEnth=usr->dmEnth;

  if (usr->par->full_ridge) full_ridge = 2.0;
  else full_ridge = 1.0;

  // get coordinates of dmVel for edges
  PetscCall(DMStagGetGlobalSizes(dmVel, &Nx, &Nz,NULL));
  PetscCall(DMStagGetCorners(dmVel,&sx,&sz,NULL,&nx,&nz,NULL,NULL,NULL,NULL)); 
  PetscCall(DMStagGetProductCoordinateArraysRead(dmVel,&coordx,&coordz,NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmVel,LEFT,&iprev));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmVel,RIGHT,&inext)); 

  PetscCall(DMGetLocalVector(dmVel, &xVellocal)); 
  PetscCall(DMGlobalToLocal (dmVel, usr->xVel, INSERT_VALUES, xVellocal)); 
  PetscCall(DMStagVecGetArrayRead(dmVel,xVellocal,&_xVellocal));

  PetscCall(DMGetLocalVector(dmEnth, &xEnthlocal)); 
  PetscCall(DMGlobalToLocal (dmEnth, usr->xEnth, INSERT_VALUES, xEnthlocal)); 
  PetscCall(DMStagVecGetArrayRead(dmEnth,xEnthlocal,&_xEnthlocal));

  PetscCall(DMStagGetLocationSlot(dmEnth,DMSTAG_ELEMENT,ENTH_ELEMENT_PHI,&iphi));
  PetscCall(DMStagGetLocationSlot(dmEnth,DMSTAG_ELEMENT,ENTH_ELEMENT_CF ,&iCF ));
  PetscCall(DMStagGetLocationSlot(dmVel,DMSTAG_DOWN,VEL_FACE_VF, &v_slot[0]));
  PetscCall(DMStagGetLocationSlot(dmVel,DMSTAG_UP,  VEL_FACE_VF, &v_slot[1]));

  // check indices for xMOR
  imor_s = Nx;
  imor_e = 0;
  for (i = sx; i <sx+nx; i++) {
    PetscScalar xc;
    xc = (coordx[i][inext]+coordx[i][iprev])*0.5;
    if (fabs(xc) <= usr->nd->xmor) {
      imor_s = PetscMin(i,imor_s);
      imor_e = PetscMax(i,imor_e);
    }
  }

  gout_F   = 0.0;
  gout_C   = 0.0;
  gvfz_max = 0.0;
  gphi_max = 0.0;

  out_F = 0.0;
  out_C = 0.0;
  phi_max = 0.0;
  vfz_max = 0.0;

  // Loop over local domain for xMOR
  if ((imor_s>=sx) && (imor_e<sx+nx) && (sz+nz==Nz)) { // check if MOR axis is on processor
    for (i = imor_s; i < imor_e+1; i++) {
      PetscScalar vf, dz, phi, Cf, flux_ij;
      j = Nz-1;

      // get fluid velocity (z)
      vf = (_xVellocal[j][i][v_slot[0]]+_xVellocal[j][i][v_slot[1]])*0.5;

      // get porosity and CF
      phi = _xEnthlocal[j][i][iphi];
      Cf  = _xEnthlocal[j][i][iCF ];

      vfz_max = PetscMax(vfz_max,vf);
      phi_max = PetscMax(phi_max,phi);

      dz = coordz[j][inext]-coordz[j][iprev];
      flux_ij = phi*vf*dz;
      out_F += flux_ij;
      out_C += flux_ij*Cf;
    }
  }

  // Parallel
  PetscCall(MPI_Allreduce(&out_F,&gout_F,1,MPI_DOUBLE,MPI_SUM,usr->comm));
  PetscCall(MPI_Allreduce(&out_C,&gout_C,1,MPI_DOUBLE,MPI_SUM,usr->comm));

  // Parallel phi_max, vfz_max
  PetscCall(MPI_Allreduce(&vfz_max,&gvfz_max,1,MPI_DOUBLE,MPI_MAX,usr->comm));
  PetscCall(MPI_Allreduce(&phi_max,&gphi_max,1,MPI_DOUBLE,MPI_MAX,usr->comm));

  PetscScalar C, F, h_crust, t;
  t = (usr->nd->t+usr->nd->dt)*usr->scal->t/SEC_YEAR; 
  if (gout_F==0.0) C = usr->par->C0;
  else C = gout_C/gout_F*usr->par->DC+usr->par->C0;
  F = gout_F*usr->par->rho0*usr->scal->v*usr->scal->x*SEC_YEAR/full_ridge; // kg/m/year
  h_crust = F/(usr->par->rho0*usr->par->U0)*1.0e2; // m

  // Output
  PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");
  PetscPrintf(PETSC_COMM_WORLD,"# xMOR FLUXES: t = %1.12e [yr] C = %1.12e [wt. frac.] F = %1.12e [kg/m/yr] h_crust = %1.12e [m]\n",t,C,F,h_crust);
  PetscPrintf(PETSC_COMM_WORLD,"#              phi_max = %1.12e vfz_max = %1.12e [cm/yr]\n",gphi_max,gvfz_max*usr->scal->v/1.0e-2*SEC_YEAR);

  // Restore arrays
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dmVel,&coordx,&coordz,NULL));
  PetscCall(DMStagVecRestoreArrayRead(dmVel,xVellocal,&_xVellocal));
  PetscCall(DMRestoreLocalVector(dmVel, &xVellocal)); 
  PetscCall(DMStagVecRestoreArrayRead(dmEnth,xEnthlocal,&_xEnthlocal));
  PetscCall(DMRestoreLocalVector(dmEnth,&xEnthlocal)); 

  PetscTime(&tlog[1]);
  if (usr->par->log_info) {
    printf("  ComputeMeltExtractOutflux: total                   %1.2e\n",tlog[1]-tlog[0]);
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// Calculate asymmetry in full ridge models
// ---------------------------------------
PetscErrorCode ComputeAsymmetryFullRidge(void *ctx) 
{
  UsrData       *usr = (UsrData*) ctx;
  PetscInt       i, j, sx, sz, nx, nz, icenter, iphi;
  PetscScalar    **coordx,**coordz,***_xEnthlocal;
  PetscScalar    A = 0.0, A_left, A_right, gA_left, gA_right, dx, dz, xmor;
  DM             dmEnth;
  Vec            xEnthlocal;
  PetscLogDouble tlog[2];
  PetscFunctionBeginUser;
  
  if (!usr->par->full_ridge) PetscFunctionReturn(PETSC_SUCCESS);

  PetscTime(&tlog[0]);
  dmEnth=usr->dmEnth;

  // get coordinates
  PetscCall(DMStagGetCorners(dmEnth,&sx,&sz,NULL,&nx,&nz,NULL,NULL,NULL,NULL)); 
  PetscCall(DMStagGetProductCoordinateArraysRead(dmEnth,&coordx,&coordz,NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmEnth,ELEMENT,&icenter));

  PetscCall(DMGetLocalVector(dmEnth, &xEnthlocal)); 
  PetscCall(DMGlobalToLocal (dmEnth, usr->xEnth, INSERT_VALUES, xEnthlocal)); 
  PetscCall(DMStagVecGetArrayRead(dmEnth,xEnthlocal,&_xEnthlocal));
  PetscCall(DMStagGetLocationSlot(dmEnth,DMSTAG_ELEMENT,ENTH_ELEMENT_PHI,&iphi));

  xmor    = 0.0;
  A_left  = 0.0;
  A_right = 0.0;
  gA_left = 0.0;
  gA_right= 0.0;
  
  dx = coordx[sx+1][icenter]-coordx[sx][icenter];
  dz = coordz[sz+1][icenter]-coordz[sz][icenter];
  
  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      PetscScalar   phi, xc, F = 0.0;
      // get porosity
      phi = _xEnthlocal[j][i][iphi];
      if (phi>0.0) F = 1.0;
      if (coordx[i][icenter]<= xmor) A_left  +=F*dx*dz;
      if (coordx[i][icenter]>= xmor) A_right +=F*dx*dz;
    }
  }

  // Parallel
  PetscCall(MPI_Allreduce(&A_left ,&gA_left ,1,MPI_DOUBLE,MPI_SUM,usr->comm));
  PetscCall(MPI_Allreduce(&A_right,&gA_right,1,MPI_DOUBLE,MPI_SUM,usr->comm));

  if (gA_left+gA_right>0.0) A = 2.0*gA_right/(gA_left+gA_right)-1.0;

  // Output
  PetscPrintf(PETSC_COMM_WORLD,"# Asymmetry (full ridge): A = %1.12e A_left = %1.12e A_right = %1.12e \n",A,gA_left,gA_right);
  PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");

  // Restore arrays
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dmEnth,&coordx,&coordz,NULL));
  PetscCall(DMStagVecRestoreArrayRead(dmEnth,xEnthlocal,&_xEnthlocal));
  PetscCall(DMRestoreLocalVector(dmEnth,&xEnthlocal)); 

  PetscTime(&tlog[1]);
  if (usr->par->log_info) {
    printf("  ComputeAsymmetryFullRidge: total                   %1.2e\n",tlog[1]-tlog[0]);
  }

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
  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->scal->v,1,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->scal->t,1,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->scal->K,1,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->scal->P,1,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->scal->eta,1,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->scal->rho,1,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->scal->H,1,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->scal->Gamma,1,PETSC_DOUBLE));

  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['scalx'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['scalv'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['scalt'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['scalK'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['scalP'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['scaleta'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['scalrho'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['scalH'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['scalGamma'] = v\n");

  // parameters - nd
  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->nd->L,1,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->nd->H,1,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->nd->xmin,1,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->nd->zmin,1,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->nd->xmor,1,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->nd->U0,1,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->nd->visc_ratio,1,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->nd->eta_min,1,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->nd->eta_max,1,PETSC_DOUBLE));

  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['L'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['H'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['xmin'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['zmin'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['xsill'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['U0'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['visc_ratio'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['eta_min'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['eta_max'] = v\n");

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
  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->nd->alpha_s,1,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->nd->beta_s,1,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->nd->A,1,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->nd->S,1,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->nd->PeT,1,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->nd->PeC,1,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->nd->thetaS,1,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->nd->G,1,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->nd->RM,1,PETSC_DOUBLE));

  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['delta'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['alpha_s'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['beta_s'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['A'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['S'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['PeT'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['PeC'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['thetaS'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['G'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['RM'] = v\n");

  // Note: readBag() in PetscBinaryIO.py is not yet implemented, so will close the python file without reading bag
  // Also some bag parameters needed for scaling
  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->par->C0,1,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->par->DC,1,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->par->T0,1,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryWrite(viewer,(void*)&usr->par->DT,1,PETSC_DOUBLE));

  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['C0'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['DC'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['T0'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['DT'] = v\n");

  // output bag
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
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->scal->v,1,NULL,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->scal->t,1,NULL,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->scal->K,1,NULL,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->scal->P,1,NULL,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->scal->eta,1,NULL,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->scal->rho,1,NULL,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->scal->H,1,NULL,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->scal->Gamma,1,NULL,PETSC_DOUBLE));

  // parameters - nd
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->nd->L,1,NULL,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->nd->H,1,NULL,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->nd->xmin,1,NULL,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->nd->zmin,1,NULL,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->nd->xmor,1,NULL,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->nd->U0,1,NULL,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->nd->visc_ratio,1,NULL,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->nd->eta_min,1,NULL,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->nd->eta_max,1,NULL,PETSC_DOUBLE));

  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->nd->istep,1,NULL,PETSC_INT));
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->nd->t,1,NULL,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->nd->dt,1,NULL,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->nd->tmax,1,NULL,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->nd->dtmax,1,NULL,PETSC_DOUBLE));

  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->nd->delta,1,NULL,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->nd->alpha_s,1,NULL,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->nd->beta_s,1,NULL,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->nd->A,1,NULL,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->nd->S,1,NULL,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->nd->PeT,1,NULL,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->nd->PeC,1,NULL,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->nd->thetaS,1,NULL,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->nd->G,1,NULL,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->nd->RM,1,NULL,PETSC_DOUBLE));

  // these bag parameters are needed for scaling in python
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->par->C0,1,NULL,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->par->DC,1,NULL,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->par->T0,1,NULL,PETSC_DOUBLE));
  PetscCall(PetscViewerBinaryRead(viewer,(void*)&usr->par->DT,1,NULL,PETSC_DOUBLE));

  // save timestep
  PetscInt    tstep, tout, hc_cycles;
  PetscScalar tmax, dtmax;

  tstep = usr->par->tstep;
  tout  = usr->par->tout;
  hc_cycles = usr->par->hc_cycles;
  tmax  = usr->par->tmax;
  dtmax = usr->par->dtmax;

  // save buoyancy
  PetscInt    buoy_phi, buoy_C, buoy_T;
  PetscScalar beta;

  buoy_phi = usr->par->buoy_phi;
  buoy_C   = usr->par->buoy_C;
  buoy_T   = usr->par->buoy_T;
  beta   = usr->par->beta;

  // save forcing
  PetscInt    forcing;
  PetscScalar dTdx_bottom,dCdx_bottom;

  forcing = usr->par->forcing;
  dTdx_bottom = usr->par->dTdx_bottom;
  dCdx_bottom = usr->par->dCdx_bottom;

  // read bag
  PetscCall(PetscBagLoad(viewer,usr->bag));
  PetscCall(PetscViewerDestroy(&viewer));

  usr->par->tstep = tstep;
  usr->par->tout  = tout;
  usr->par->hc_cycles = hc_cycles;
  usr->par->tmax = tmax;
  usr->par->dtmax= dtmax;

  // non-dimensionalize necessary params
  usr->nd->tmax  = nd_param(usr->par->tmax*SEC_YEAR,usr->scal->t);
  usr->nd->dtmax = nd_param(usr->par->dtmax*SEC_YEAR,usr->scal->t);

  // buoyancy 
  usr->par->buoy_phi = buoy_phi;
  usr->par->buoy_C   = buoy_C; 
  usr->par->buoy_T   = buoy_T; 
  usr->par->beta     = beta; 

  if ((usr->par->buoy_phi==0) && (usr->par->buoy_C==0) && (usr->par->buoy_T==0)) usr->par->buoyancy = 0;
  if ((usr->par->buoy_phi==1) && (usr->par->buoy_C==0) && (usr->par->buoy_T==0)) usr->par->buoyancy = 1;
  if ((usr->par->buoy_phi==0) && (usr->par->buoy_C>=1) && (usr->par->buoy_T==0)) usr->par->buoyancy = 2;
  if ((usr->par->buoy_phi==1) && (usr->par->buoy_C>=1) && (usr->par->buoy_T==0)) usr->par->buoyancy = 3;
  if ((usr->par->buoy_phi==0) && (usr->par->buoy_C==0) && (usr->par->buoy_T>=1)) usr->par->buoyancy = 4;
  if ((usr->par->buoy_phi==1) && (usr->par->buoy_C==0) && (usr->par->buoy_T>=1)) usr->par->buoyancy = 5;
  if ((usr->par->buoy_phi==0) && (usr->par->buoy_C>=1) && (usr->par->buoy_T>=1)) usr->par->buoyancy = 6;
  if ((usr->par->buoy_phi==1) && (usr->par->buoy_C>=1) && (usr->par->buoy_T>=1)) usr->par->buoyancy = 7;

  usr->nd->beta_s  = usr->par->beta*usr->par->rho0*usr->par->DC/usr->par->drho;
  usr->nd->beta_ls = usr->par->beta*usr->par->DC;

  // forcing
  usr->par->forcing = forcing;
  usr->par->dTdx_bottom = dTdx_bottom;
  usr->par->dCdx_bottom = dCdx_bottom;

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// Compute Melting Rate (Gamma)
// ---------------------------------------
PetscErrorCode ComputeGamma(DM dmmatProp, Vec xmatProp, DM dmPV, Vec xPV, DM dmEnth, Vec xEnth, Vec xEnthold, void *ctx) 
{
  UsrData       *usr = (UsrData*) ctx;
  PetscInt       i, j, ii, sx, sz, nx, nz, Nx, Nz,idx,iprev,inext,icenter;
  PetscScalar    **coordx,**coordz,***xx;
  Vec            xmatProplocal,xPVlocal,xEnthlocal,xEntholdlocal;
  PetscScalar    ***_xPVlocal,***_xEnthlocal,***_xEntholdlocal;
  PetscInt       pv_slot[4],iphi;
  PetscLogDouble tlog[2];
  PetscFunctionBeginUser;

  PetscTime(&tlog[0]);
  PetscCall(DMStagGetGlobalSizes(dmPV,&Nx,&Nz,NULL));
  PetscCall(DMStagGetCorners(dmPV,&sx,&sz,NULL,&nx,&nz,NULL,NULL,NULL,NULL)); 

  // get coordinates of dmPV for center and edges
  PetscCall(DMStagGetProductCoordinateArraysRead(dmPV,&coordx,&coordz,NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmPV,LEFT,&iprev));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmPV,RIGHT,&inext)); 
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmPV,ELEMENT,&icenter)); 
  
  PetscCall(DMGetLocalVector(dmmatProp,&xmatProplocal));
  PetscCall(DMGlobalToLocal (dmmatProp, xmatProp, INSERT_VALUES, xmatProplocal)); 
  PetscCall(DMStagVecGetArray(dmmatProp, xmatProplocal, &xx)); 

  PetscCall(DMGetLocalVector(dmPV, &xPVlocal)); 
  PetscCall(DMGlobalToLocal (dmPV, xPV, INSERT_VALUES, xPVlocal)); 
  PetscCall(DMStagVecGetArrayRead(dmPV,xPVlocal,&_xPVlocal));

  PetscCall(DMGetLocalVector(dmEnth, &xEnthlocal)); 
  PetscCall(DMGlobalToLocal (dmEnth, xEnth, INSERT_VALUES, xEnthlocal)); 
  PetscCall(DMStagVecGetArrayRead(dmEnth,xEnthlocal,&_xEnthlocal));

  PetscCall(DMGetLocalVector(dmEnth, &xEntholdlocal)); 
  PetscCall(DMGlobalToLocal (dmEnth, xEnthold, INSERT_VALUES, xEntholdlocal)); 
  PetscCall(DMStagVecGetArrayRead(dmEnth,xEntholdlocal,&_xEntholdlocal));

  // get location slots
  PetscCall(DMStagGetLocationSlot(dmPV,DMSTAG_LEFT,   PV_FACE_VS,   &pv_slot[0]));
  PetscCall(DMStagGetLocationSlot(dmPV,DMSTAG_RIGHT,  PV_FACE_VS,   &pv_slot[1]));
  PetscCall(DMStagGetLocationSlot(dmPV,DMSTAG_DOWN,   PV_FACE_VS,   &pv_slot[2]));
  PetscCall(DMStagGetLocationSlot(dmPV,DMSTAG_UP,     PV_FACE_VS,   &pv_slot[3]));
  PetscCall(DMStagGetLocationSlot(dmEnth,DMSTAG_ELEMENT,ENTH_ELEMENT_PHI,&iphi));
  PetscCall(DMStagGetLocationSlot(dmmatProp,ELEMENT,6,&idx)); 

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      PetscInt    im, jm;
      PetscScalar v[5], phi[9], phis[9], phiold[9], adv, dx[3], dz[3];

      // get solid velocity
      v[0] = 0.0;
      v[1] = _xPVlocal[j][i][pv_slot[0]];
      v[2] = _xPVlocal[j][i][pv_slot[1]];
      v[3] = _xPVlocal[j][i][pv_slot[2]];
      v[4] = _xPVlocal[j][i][pv_slot[3]];

      // get phi data
      phi[0] = _xEnthlocal[j][i][iphi]; // Qi,j -C
      if (i == 0   ) { im = i; jm = j; } else { im = i-1; jm = j  ; } phi[1] = _xEnthlocal[jm][im][iphi];// Qi-1,j -W
      if (i == Nx-1) { im = i; jm = j; } else { im = i+1; jm = j  ; } phi[2] = _xEnthlocal[jm][im][iphi];// Qi+1,j -E
      if (j == 0   ) { im = i; jm = j; } else { im = i  ; jm = j-1; } phi[3] = _xEnthlocal[jm][im][iphi];// Qi,j-1 -S
      if (j == Nz-1) { im = i; jm = j; } else { im = i  ; jm = j+1; } phi[4] = _xEnthlocal[jm][im][iphi];// Qi,j+1 -N

      if (i <= 1   ) { phi[5] = phi[2]; } else { im = i-2; jm = j  ; } phi[5] = _xEnthlocal[jm][im][iphi];// Qi-2,j -WW
      if (i >= Nx-2) { phi[6] = phi[1]; } else { im = i+2; jm = j  ; } phi[6] = _xEnthlocal[jm][im][iphi];// Qi+2,j -EE
      if (j <= 1   ) { phi[7] = phi[4]; } else { im = i  ; jm = j-2; } phi[7] = _xEnthlocal[jm][im][iphi];// Qi,j-2 -SS
      if (j >= Nz-2) { phi[8] = phi[3]; } else { im = i  ; jm = j+2; } phi[8] = _xEnthlocal[jm][im][iphi];// Qi,j+2 -NN

      phiold[0] = _xEntholdlocal[j][i][iphi]; // Qi,j -C
      if (i == 0   ) { im = i; jm = j; } else { im = i-1; jm = j  ; } phiold[1] = _xEntholdlocal[jm][im][iphi];// Qi-1,j -W
      if (i == Nx-1) { im = i; jm = j; } else { im = i+1; jm = j  ; } phiold[2] = _xEntholdlocal[jm][im][iphi];// Qi+1,j -E
      if (j == 0   ) { im = i; jm = j; } else { im = i  ; jm = j-1; } phiold[3] = _xEntholdlocal[jm][im][iphi];// Qi,j-1 -S
      if (j == Nz-1) { im = i; jm = j; } else { im = i  ; jm = j+1; } phiold[4] = _xEntholdlocal[jm][im][iphi];// Qi,j+1 -N
      
      if (i <= 1   ) { phiold[5] = phiold[2]; } else { im = i-2; jm = j  ; } phiold[5] = _xEntholdlocal[jm][im][iphi];// Qi-2,j -WW
      if (i >= Nx-2) { phiold[6] = phiold[1]; } else { im = i+2; jm = j  ; } phiold[6] = _xEntholdlocal[jm][im][iphi];// Qi+2,j -EE
      if (j <= 1   ) { phiold[7] = phiold[4]; } else { im = i  ; jm = j-2; } phiold[7] = _xEntholdlocal[jm][im][iphi];// Qi,j-2 -SS
      if (j >= Nz-2) { phiold[8] = phiold[3]; } else { im = i  ; jm = j+2; } phiold[8] = _xEntholdlocal[jm][im][iphi];// Qi,j+2 -NN
      
      for (ii = 0; ii <9; ii++) { phis[ii] = 1.0 - phi[ii]; }

      // Grid spacings
      if (i == Nx-1) dx[0] = coordx[i  ][icenter]-coordx[i-1][icenter];
      else           dx[0] = coordx[i+1][icenter]-coordx[i  ][icenter];

      if (i == 0) dx[1] = coordx[i+1][icenter]-coordx[i  ][icenter];
      else        dx[1] = coordx[i  ][icenter]-coordx[i-1][icenter];
      dx[2]  = (dx[0]+dx[1])*0.5;

      if (j == Nz-1) dz[0] = coordz[j  ][icenter]-coordz[j-1][icenter];
      else           dz[0] = coordz[j+1][icenter]-coordz[j  ][icenter];

      if (j == 0) dz[1] = coordz[j+1][icenter]-coordz[j  ][icenter];
      else        dz[1] = coordz[j  ][icenter]-coordz[j-1][icenter];
      dz[2] = (dz[0]+dz[1])*0.5;

      // advection term div(phis*vs) - assume backward Euler+Fromm scheme: gamma = dphi/dt - adv^new
      PetscCall(AdvectionResidual(v,phis,dx,dz,usr->nd->dt,ADV_FROMM,&adv)); 

      // update gamma
      xx[j][i][idx] = (phi[0]-phiold[0])/usr->nd->dt - adv;
    }
  }

  // Restore arrays
  PetscCall(DMStagVecRestoreArray(dmmatProp,xmatProplocal,&xx)); 
  PetscCall(DMLocalToGlobalBegin(dmmatProp,xmatProplocal,INSERT_VALUES,xmatProp)); 
  PetscCall(DMLocalToGlobalEnd  (dmmatProp,xmatProplocal,INSERT_VALUES,xmatProp)); 
  PetscCall(DMRestoreLocalVector(dmmatProp,&xmatProplocal)); 

  PetscCall(DMStagRestoreProductCoordinateArraysRead(dmPV,&coordx,&coordz,NULL));
  PetscCall(DMStagVecRestoreArrayRead(dmPV,xPVlocal,&_xPVlocal));
  PetscCall(DMRestoreLocalVector(dmPV, &xPVlocal)); 

  PetscCall(DMStagVecRestoreArrayRead(dmEnth,xEnthlocal,&_xEnthlocal));
  PetscCall(DMRestoreLocalVector(dmEnth, &xEnthlocal)); 

  PetscCall(DMStagVecRestoreArrayRead(dmEnth,xEntholdlocal,&_xEntholdlocal));
  PetscCall(DMRestoreLocalVector(dmEnth, &xEntholdlocal)); 

  PetscTime(&tlog[1]);
  if (usr->par->log_info) {
    printf("  ComputeGamma: total                                %1.2e\n",tlog[1]-tlog[0]);
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}
