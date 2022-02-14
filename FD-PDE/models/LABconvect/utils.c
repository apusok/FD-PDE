#include "LABconvect.h"
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
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  // zero PV initial guess (default)
  // half-space cooling model - initialize H, C
  ierr = HalfSpaceCooling(usr);CHKERRQ(ierr);

  ierr = PetscSNPrintf(usr->par->fdir_out,sizeof(usr->par->fdir_out),"Timestep%d",usr->nd->istep);
  ierr = CreateDirectory(usr->par->fdir_out);CHKERRQ(ierr);
  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_xHC_HS_ts%d",usr->par->fdir_out,usr->nd->istep);
  ierr = DMStagViewBinaryPython(usr->dmHC,usr->xHC,fout);CHKERRQ(ierr);

  // Update lithostatic pressure
  ierr = FDPDEEnthalpyGetPressure(fdHC,&dmP,&xP);CHKERRQ(ierr);
  ierr = UpdateLithostaticPressure(dmP,xP,usr);CHKERRQ(ierr);
  ierr = FDPDEEnthalpyGetPrevPressure(fdHC,&xPprev);CHKERRQ(ierr);
  ierr = VecCopy(xP,xPprev);CHKERRQ(ierr);
  ierr = VecDestroy(&xP);CHKERRQ(ierr);
  ierr = VecDestroy(&xPprev);CHKERRQ(ierr);
  ierr = DMDestroy(&dmP);CHKERRQ(ierr);

  // Update Enthalpy diagnostics
  ierr = FDPDEEnthalpyUpdateDiagnostics(fdHC,usr->dmHC,usr->xHC,&usr->dmEnth,&usr->xEnth); CHKERRQ(ierr);
  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_xEnth_HS_ts%d",usr->par->fdir_out,usr->nd->istep);
  ierr = DMStagViewBinaryPython(usr->dmEnth,usr->xEnth,fout);CHKERRQ(ierr);

  // Correct H-S*phi and C=Cs to ensure phi=phi*phi_init
  ierr = CorrectInitialHCZeroPorosity(usr->dmEnth,usr->xEnth,usr);CHKERRQ(ierr);

  // Update Enthalpy again for visualization and to initialize xEnthold
  ierr = FDPDEEnthalpyUpdateDiagnostics(fdHC,usr->dmHC,usr->xHC,NULL,&xEnth); CHKERRQ(ierr);
  ierr = VecCopy(xEnth,usr->xEnth);CHKERRQ(ierr);
  ierr = VecDestroy(&xEnth);CHKERRQ(ierr);

  ierr = VecDuplicate(usr->xEnth,&usr->xEnthold);CHKERRQ(ierr);
  ierr = VecCopy(usr->xEnth,usr->xEnthold);CHKERRQ(ierr);

  // Update fluid velocity to zero and v=vs
  ierr = ComputeFluidAndBulkVelocity(usr->dmPV,usr->xPV,usr->dmEnth,usr->xEnth,usr->dmVel,usr->xVel,usr);CHKERRQ(ierr);

  // Initialize guess and previous solution in fdHC
  ierr = FDPDEEnthalpyGetPrevSolution(fdHC,&xHCprev);CHKERRQ(ierr);
  ierr = VecCopy(usr->xHC,xHCprev);CHKERRQ(ierr);
  ierr = FDPDEGetSolutionGuess(fdHC,&xHCguess);CHKERRQ(ierr);
  ierr = VecCopy(xHCprev,xHCguess);CHKERRQ(ierr);
  ierr = VecDestroy(&xHCprev);CHKERRQ(ierr);
  ierr = VecDestroy(&xHCguess);CHKERRQ(ierr);

  // Set initial coefficient structure
  ierr = FDPDEGetCoefficient(fdHC,&dmHCcoeff,&xHCcoeff);CHKERRQ(ierr);
  ierr = FDPDEEnthalpyGetPrevCoefficient(fdHC,&xHCcoeffprev);CHKERRQ(ierr);
  ierr = FormCoefficient_HC(fdHC,usr->dmHC,usr->xHC,dmHCcoeff,xHCcoeffprev,usr);CHKERRQ(ierr);
  ierr = VecCopy(xHCcoeffprev,xHCcoeff);CHKERRQ(ierr);

  // Output prev coefficient
  ierr = PetscSNPrintf(usr->par->fdir_out,sizeof(usr->par->fdir_out),"Timestep%d",usr->nd->istep);
  ierr = CreateDirectory(usr->par->fdir_out);CHKERRQ(ierr);
  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_xHCcoeff_ts%d",usr->par->fdir_out,usr->nd->istep);
  ierr = DMStagViewBinaryPython(dmHCcoeff,xHCcoeffprev,fout);CHKERRQ(ierr);
  ierr = VecDestroy(&xHCcoeffprev);CHKERRQ(ierr);

  // Output initial conditions
  ierr = DoOutput(fdPV,fdHC,usr);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// Half-Space Cooling model
// ---------------------------------------
PetscErrorCode HalfSpaceCooling(void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz, iH, iC, icenter;
  PetscScalar    **coordx,**coordz, ***xx, Cs0, Tm, Cdepl, zdepl;
  Vec            x, xlocal;
  DM             dm;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  dm  = usr->dmHC;
  x   = usr->xHC;
  Cs0 = usr->par->C0;
  Tm  = (usr->par->Tp-T_KELVIN)*exp(-usr->nd->A*usr->nd->zmin)+T_KELVIN;
  Cdepl = usr->par->C0+usr->par->DC*usr->par->depletion;
  zdepl = usr->nd->zdepl;

  ierr = DMStagGetGlobalSizes(dm, &Nx, &Nz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter);CHKERRQ(ierr); 
  ierr = DMStagGetLocationSlot(dm, ELEMENT, 0, &iH); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm, ELEMENT, 1, &iC); CHKERRQ(ierr);

  ierr = DMCreateLocalVector(dm, &xlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm, xlocal, &xx); CHKERRQ(ierr);

  // Loop over local domain
  for (i = sx; i <sx+nx; i++) {
    for (j = sz; j < sz+nz; j++) {
      PetscScalar T, nd_T, age, C=0.0, zdepl0;

      // half-space cooling temperature
      age = usr->nd->age*usr->scal->t;
      T = HalfSpaceCoolingTemp(Tm,usr->par->Ts,-dim_param(coordz[j][icenter],usr->scal->x),usr->par->kappa,age); 
      nd_T = (T - usr->par->T0)/usr->par->DT;

      // enthalpy H = S*phi+T (phi=0)
      xx[j][i][iH] = nd_T;

      // add perturbation
      zdepl0 = zdepl*(1.0+ usr->par->Adepl*PetscCosScalar(usr->par->ndepl*PETSC_PI*coordx[i][icenter]));

      // initial bulk composition C0 = Cs0 (phi=0) - impose gradient in lithosphere
      if (coordz[j][icenter]>=zdepl0) {
        C = Cs0+(coordz[j][icenter]-zdepl)*(Cdepl-Cs0)/PetscAbsScalar(zdepl);
      } else {
        C = Cs0;
      }
      xx[j][i][iC] = (C-usr->par->C0)/usr->par->DC;
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
PetscErrorCode HalfSpaceCooling_Tdepletion(void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz, iH, iC, icenter;
  PetscScalar    **coordx,**coordz, ***xx, Cs0, Tm, Tdepl, Cdepl;
  Vec            x, xlocal;
  DM             dm;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  dm  = usr->dmHC;
  x   = usr->xHC;
  Cs0 = usr->par->C0;
  Tm  = (usr->par->Tp-T_KELVIN)*exp(-usr->nd->A*usr->nd->zmin)+T_KELVIN;
  Tdepl = (usr->par->Tdepletion - usr->par->T0)/usr->par->DT;
  Cdepl = usr->par->C0+usr->par->DC*usr->par->depletion;

  ierr = DMStagGetGlobalSizes(dm, &Nx, &Nz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter);CHKERRQ(ierr); 
  ierr = DMStagGetLocationSlot(dm, ELEMENT, 0, &iH); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm, ELEMENT, 1, &iC); CHKERRQ(ierr);

  ierr = DMCreateLocalVector(dm, &xlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm, xlocal, &xx); CHKERRQ(ierr);

  // Loop over local domain
  for (i = sx; i <sx+nx; i++) {
    PetscScalar zdepl = 0.0; 
    for (j = sz; j < sz+nz; j++) {
      PetscScalar T, nd_T, age, Tdepl0, C=0.0;

      // half-space cooling temperature
      age = usr->nd->age*usr->scal->t;
      T = HalfSpaceCoolingTemp(Tm,usr->par->Ts,-dim_param(coordz[j][icenter],usr->scal->x),usr->par->kappa,age); 
      nd_T = (T - usr->par->T0)/usr->par->DT;

      // enthalpy H = S*phi+T (phi=0)
      xx[j][i][iH] = nd_T;

      // add perturbation to Tdepl
      Tdepl0 = Tdepl + PetscCosScalar(2.0*PETSC_PI*coordx[i][icenter]);

      // initial bulk composition C0 = Cs0 (phi=0) - impose gradient in lithosphere
      if (nd_T<=Tdepl0) {
        zdepl = PetscMin(zdepl,coordz[j][icenter]);
        C = Cs0+(coordz[j][icenter]-zdepl)*(Cdepl-Cs0)/PetscAbsScalar(zdepl);
      } else {
        C = Cs0;
      }
      xx[j][i][iC] = (C-usr->par->C0)/usr->par->DC;
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
// Update Lithostatic pressure
// ---------------------------------------
PetscErrorCode UpdateLithostaticPressure(DM dm, Vec x, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz, idx, icenter;
  PetscScalar    **coordx,**coordz, ***xx;
  Vec            xlocal;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = DMStagGetGlobalSizes(dm, &Nx, &Nz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter);CHKERRQ(ierr); 
  ierr = DMStagGetLocationSlot(dm, ELEMENT, 0, &idx); CHKERRQ(ierr);

  ierr = DMCreateLocalVector(dm, &xlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm, xlocal, &xx); CHKERRQ(ierr);

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      PetscScalar rho;
      rho  = usr->par->rho0; 
      xx[j][i][idx] = LithostaticPressure(rho,usr->par->drho,coordz[j][icenter]);
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
// Correct initial enthalpy and bulk composition for zero porosity
// ---------------------------------------
PetscErrorCode CorrectInitialHCZeroPorosity(DM dmEnth, Vec xEnth, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  PetscInt       i, j, sx, sz, nx, nz, iH, iC, dm_slot[3];
  PetscScalar    ***xx, ***enth, phi_init;
  Vec            x, xlocal, xnewlocal;
  DM             dm;
  PetscLogDouble tlog[2];
  PetscErrorCode ierr;
  PetscFunctionBegin;

  PetscTime(&tlog[0]);
  dm  = usr->dmHC;
  x   = usr->xHC;
  phi_init = 0.0; //1.0e-4;

  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm, ELEMENT, 0, &iH); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm, ELEMENT, 1, &iC); CHKERRQ(ierr);

  ierr = DMGetLocalVector(dm, &xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm, x, INSERT_VALUES, xlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm, xlocal, &xx); CHKERRQ(ierr);

  ierr = DMGetLocalVector(dmEnth, &xnewlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dmEnth, xEnth, INSERT_VALUES, xnewlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(dmEnth,xnewlocal,&enth);CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(dmEnth,DMSTAG_ELEMENT,ENTH_ELEMENT_PHI,&dm_slot[0]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmEnth,DMSTAG_ELEMENT,ENTH_ELEMENT_CS ,&dm_slot[1]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmEnth,DMSTAG_ELEMENT,ENTH_ELEMENT_CF ,&dm_slot[2]); CHKERRQ(ierr);

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      PetscScalar   xs[3],phi;

      xs[0] = enth[j][i][dm_slot[0]];
      xs[1] = enth[j][i][dm_slot[1]];
      xs[2] = enth[j][i][dm_slot[2]];

      phi = xs[0]*phi_init;

      // H + S*(phi-phi0)
      xx[j][i][iH] += usr->nd->S*(phi - xs[0]); 

      // bulk composition
      xx[j][i][iC] = (1.0-phi)*xs[1] + phi*xs[2]; 
    }
  }

  // Restore arrays
  ierr = DMStagVecRestoreArray(dm,xlocal,&xx); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm,xlocal,INSERT_VALUES,x); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,xlocal,INSERT_VALUES,x); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &xlocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArrayRead(dmEnth,xnewlocal,&enth);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmEnth, &xnewlocal); CHKERRQ(ierr);

  PetscTime(&tlog[1]);
  if (usr->par->log_info) {
    printf("  CorrectInitialHCZeroPorosity: total                %1.2e\n",tlog[1]-tlog[0]);
  }

  PetscFunctionReturn(0);
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
  PetscErrorCode ierr;
  PetscFunctionBegin;

  PetscTime(&tlog[0]);
  k_hat[0] = 0.0;
  k_hat[1] = 0.0;
  k_hat[2] = usr->par->k_hat;
  k_hat[3] = usr->par->k_hat;

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

  ierr = DMGetLocalVector(dmEnth, &xEnthlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dmEnth, xEnth, INSERT_VALUES, xEnthlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(dmEnth,xEnthlocal,&_xEnthlocal);CHKERRQ(ierr);

  // get slots
  iP = 0; iPc = 1;
  iL = 2; iR  = 3;
  iD = 4; iU  = 5;
  ierr = DMStagGetLocationSlot(dmPV,DMSTAG_ELEMENT,PV_ELEMENT_P, &pv_slot[iP]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmPV,DMSTAG_ELEMENT,PV_ELEMENT_PC,&pv_slot[iPc]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmPV,DMSTAG_LEFT,   PV_FACE_VS,   &pv_slot[iL]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmPV,DMSTAG_RIGHT,  PV_FACE_VS,   &pv_slot[iR]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmPV,DMSTAG_DOWN,   PV_FACE_VS,   &pv_slot[iD]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmPV,DMSTAG_UP,     PV_FACE_VS,   &pv_slot[iU]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmEnth,DMSTAG_ELEMENT,ENTH_ELEMENT_PHI,&phi_slot);CHKERRQ(ierr);

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
      
      if (usr->dtype0!=DM_BOUNDARY_PERIODIC) {
        if (i == 0   ) im = i; else im = i-1; p[1] = _xPVlocal[j][im][pv_slot[iP]];
        if (i == Nx-1) im = i; else im = i+1; p[2] = _xPVlocal[j][im][pv_slot[iP]];
      } else {
        im = i-1; p[1] = _xPVlocal[j][im][pv_slot[iP]];
        im = i+1; p[2] = _xPVlocal[j][im][pv_slot[iP]];
      }
      if (j == 0   ) jm = j; else jm = j-1; p[3] = _xPVlocal[jm][i][pv_slot[iP]];
      if (j == Nz-1) jm = j; else jm = j+1; p[4] = _xPVlocal[jm][i][pv_slot[iP]];

      gradP[0] = (p[0]-p[1])/dx;
      gradP[1] = (p[2]-p[0])/dx;
      gradP[2] = (p[0]-p[3])/dz;
      gradP[3] = (p[4]-p[0])/dz;

      p[0] = _xPVlocal[j][i][pv_slot[iPc]];

      if (usr->dtype0!=DM_BOUNDARY_PERIODIC) {
        if (i == 0   ) im = i; else im = i-1; p[1] = _xPVlocal[j][im][pv_slot[iPc]];
        if (i == Nx-1) im = i; else im = i+1; p[2] = _xPVlocal[j][im][pv_slot[iPc]];
      } else {
        im = i-1; p[1] = _xPVlocal[j][im][pv_slot[iPc]];
        im = i+1; p[2] = _xPVlocal[j][im][pv_slot[iPc]];
      }
      if (j == 0   ) jm = j; else jm = j-1; p[3] = _xPVlocal[jm][i][pv_slot[iPc]];
      if (j == Nz-1) jm = j; else jm = j+1; p[4] = _xPVlocal[jm][i][pv_slot[iPc]];

      gradPc[0] = (p[0]-p[1])/dx;
      gradPc[1] = (p[2]-p[0])/dx;
      gradPc[2] = (p[0]-p[3])/dz;
      gradPc[3] = (p[4]-p[0])/dz;

      // porosity
      Q[0] = _xEnthlocal[j][i][phi_slot];

      if (usr->dtype0!=DM_BOUNDARY_PERIODIC) {
        if (i == 0   ) im = i; else im = i-1; Q[1] = _xEnthlocal[j][im][phi_slot];
        if (i == Nx-1) im = i; else im = i+1; Q[2] = _xEnthlocal[j][im][phi_slot];
      } else {
        im = i-1; Q[1] = _xEnthlocal[j][im][phi_slot];
        im = i+1; Q[2] = _xEnthlocal[j][im][phi_slot];
      }
      if (j == 0   ) jm = j; else jm = j-1; Q[3] = _xEnthlocal[jm][i][phi_slot];
      if (j == Nz-1) jm = j; else jm = j+1; Q[4] = _xEnthlocal[jm][i][phi_slot];

      // porosity on edges
      phi[0] = (Q[1]+Q[0])*0.5; 
      phi[1] = (Q[2]+Q[0])*0.5; 
      phi[2] = (Q[3]+Q[0])*0.5; 
      phi[3] = (Q[4]+Q[0])*0.5; 
      
      for (ii = 0; ii < 4; ii++) {
        // permeability
        K = Permeability(phi[ii],usr->par->phi_max,usr->par->n);

        // fluid buoyancy
        Bf = FluidBuoyancy(0.0,0.0,usr->nd->alpha_s,usr->nd->beta_s);

        // fluid velocity
        vf = FluidVelocity(vs[ii],phi[ii],gradP[ii],gradPc[ii],Bf,K,k_hat[ii]);
        xx[j][i][v_slot[ii]] = vf;

        // bulk velocity
        xx[j][i][v_slot[4+ii]] = BulkVelocity(vs[ii],vf,phi[ii]);
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
  ierr = DMStagVecRestoreArrayRead(dmEnth,xEnthlocal,&_xEnthlocal);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmEnth, &xEnthlocal); CHKERRQ(ierr);

  PetscTime(&tlog[1]);
  if (usr->par->log_info) {
    printf("  ComputeFluidAndBulkVelocity: total                 %1.2e\n",tlog[1]-tlog[0]);
  }

  PetscFunctionReturn(0);
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
  PetscErrorCode ierr;
  PetscFunctionBegin;

  PetscTime(&tlog[0]);
  nd  = usr->nd;
  par = usr->par;
  scal= usr->scal;

  ierr = DMStagGetCorners(dmmatProp,&sx,&sz,NULL,&nx,&nz,NULL,NULL,NULL,NULL); CHKERRQ(ierr);  
  ierr = DMCreateLocalVector(dmmatProp,&xmatProplocal);CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dmmatProp, xmatProplocal, &xx); CHKERRQ(ierr);

  ierr = DMGetLocalVector(dmEnth, &xEnthlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dmEnth, xEnth, INSERT_VALUES, xEnthlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(dmEnth,xEnthlocal,&_xEnthlocal);CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(dmEnth,DMSTAG_ELEMENT,ENTH_ELEMENT_PHI,&iphi);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmEnth,DMSTAG_ELEMENT,ENTH_ELEMENT_T,  &iT);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmEnth,DMSTAG_ELEMENT,ENTH_ELEMENT_CS, &iCS);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmEnth,DMSTAG_ELEMENT,ENTH_ELEMENT_CF, &iCF);CHKERRQ(ierr);

  for (ii = 0; ii < 6; ii++) {
    ierr = DMStagGetLocationSlot(dmmatProp,DMSTAG_ELEMENT,ii,&idx[ii]); CHKERRQ(ierr);
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
      K    = Permeability(phi,usr->par->phi_max,usr->par->n);
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
  ierr = DMStagVecRestoreArray(dmmatProp,xmatProplocal,&xx); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dmmatProp,xmatProplocal,INSERT_VALUES,xmatProp); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dmmatProp,xmatProplocal,INSERT_VALUES,xmatProp); CHKERRQ(ierr);
  ierr = VecDestroy(&xmatProplocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArrayRead(dmEnth,xEnthlocal,&_xEnthlocal);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmEnth,&xEnthlocal); CHKERRQ(ierr);

  PetscTime(&tlog[1]);
  if (usr->par->log_info) {
    printf("  UpdateMaterialProperties: total                    %1.2e\n",tlog[1]-tlog[0]);
  }

  PetscFunctionReturn(0);
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
  PetscErrorCode ierr;
  PetscFunctionBegin;

  PetscTime(&tlog[0]);
  ierr = DMStagGetGlobalSizes(dmPV,&Nx,&Nz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dmPV,&sx,&sz,NULL,&nx,&nz,NULL,NULL,NULL,NULL); CHKERRQ(ierr);

  // get coordinates of dmPV for center and edges
  ierr = DMStagGetProductCoordinateArraysRead(dmPV,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmPV,LEFT,&iprev);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmPV,RIGHT,&inext);CHKERRQ(ierr); 
  ierr = DMStagGetProductCoordinateLocationSlot(dmPV,ELEMENT,&icenter);CHKERRQ(ierr); 
  
  ierr = DMGetLocalVector(dmmatProp,&xmatProplocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dmmatProp, xmatProp, INSERT_VALUES, xmatProplocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dmmatProp, xmatProplocal, &xx); CHKERRQ(ierr);

  ierr = DMGetLocalVector(dmPV, &xPVlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dmPV, xPV, INSERT_VALUES, xPVlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(dmPV,xPVlocal,&_xPVlocal);CHKERRQ(ierr);

  ierr = DMGetLocalVector(dmEnth, &xEnthlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dmEnth, xEnth, INSERT_VALUES, xEnthlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(dmEnth,xEnthlocal,&_xEnthlocal);CHKERRQ(ierr);

  ierr = DMGetLocalVector(dmEnth, &xEntholdlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dmEnth, xEnthold, INSERT_VALUES, xEntholdlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(dmEnth,xEntholdlocal,&_xEntholdlocal);CHKERRQ(ierr);

  // get location slots
  ierr = DMStagGetLocationSlot(dmPV,DMSTAG_LEFT,   PV_FACE_VS,   &pv_slot[0]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmPV,DMSTAG_RIGHT,  PV_FACE_VS,   &pv_slot[1]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmPV,DMSTAG_DOWN,   PV_FACE_VS,   &pv_slot[2]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmPV,DMSTAG_UP,     PV_FACE_VS,   &pv_slot[3]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmEnth,DMSTAG_ELEMENT,ENTH_ELEMENT_PHI,&iphi);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmmatProp,ELEMENT,6,&idx); CHKERRQ(ierr);

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

      if (usr->dtype0!=DM_BOUNDARY_PERIODIC) {
        if (i == 0   ) { im = i; jm = j; } else { im = i-1; jm = j  ; } phi[1] = _xEnthlocal[jm][im][iphi];// Qi-1,j -W
        if (i == Nx-1) { im = i; jm = j; } else { im = i+1; jm = j  ; } phi[2] = _xEnthlocal[jm][im][iphi];// Qi+1,j -E
      } else {
        im = i-1; jm = j  ; phi[1] = _xEnthlocal[jm][im][iphi];// Qi-1,j -W
        im = i+1; jm = j  ; phi[2] = _xEnthlocal[jm][im][iphi];// Qi+1,j -E
      }
      if (j == 0   ) { im = i; jm = j; } else { im = i  ; jm = j-1; } phi[3] = _xEnthlocal[jm][im][iphi];// Qi,j-1 -S
      if (j == Nz-1) { im = i; jm = j; } else { im = i  ; jm = j+1; } phi[4] = _xEnthlocal[jm][im][iphi];// Qi,j+1 -N

      if (usr->dtype0!=DM_BOUNDARY_PERIODIC) {
        if (i <= 1   ) { phi[5] = phi[2]; } else { im = i-2; jm = j  ; } phi[5] = _xEnthlocal[jm][im][iphi];// Qi-2,j -WW
        if (i >= Nx-2) { phi[6] = phi[1]; } else { im = i+2; jm = j  ; } phi[6] = _xEnthlocal[jm][im][iphi];// Qi+2,j -EE
      } else {
        im = i-2; jm = j  ; phi[5] = _xEnthlocal[jm][im][iphi];// Qi-2,j -WW
        im = i+2; jm = j  ; phi[6] = _xEnthlocal[jm][im][iphi];// Qi+2,j -EE
      }
      if (j <= 1   ) { phi[7] = phi[4]; } else { im = i  ; jm = j-2; } phi[7] = _xEnthlocal[jm][im][iphi];// Qi,j-2 -SS
      if (j >= Nz-2) { phi[8] = phi[3]; } else { im = i  ; jm = j+2; } phi[8] = _xEnthlocal[jm][im][iphi];// Qi,j+2 -NN

      phiold[0] = _xEntholdlocal[j][i][iphi]; // Qi,j -C

      if (usr->dtype0!=DM_BOUNDARY_PERIODIC) {
        if (i == 0   ) { im = i; jm = j; } else { im = i-1; jm = j  ; } phiold[1] = _xEntholdlocal[jm][im][iphi];// Qi-1,j -W
        if (i == Nx-1) { im = i; jm = j; } else { im = i+1; jm = j  ; } phiold[2] = _xEntholdlocal[jm][im][iphi];// Qi+1,j -E
      } else {
        im = i-1; jm = j  ; phiold[1] = _xEntholdlocal[jm][im][iphi];// Qi-1,j -W
        im = i+1; jm = j  ; phiold[2] = _xEntholdlocal[jm][im][iphi];// Qi+1,j -E
      }
      if (j == 0   ) { im = i; jm = j; } else { im = i  ; jm = j-1; } phiold[3] = _xEntholdlocal[jm][im][iphi];// Qi,j-1 -S
      if (j == Nz-1) { im = i; jm = j; } else { im = i  ; jm = j+1; } phiold[4] = _xEntholdlocal[jm][im][iphi];// Qi,j+1 -N
      
      if (usr->dtype0!=DM_BOUNDARY_PERIODIC) {
        if (i <= 1   ) { phiold[5] = phiold[2]; } else { im = i-2; jm = j  ; } phiold[5] = _xEntholdlocal[jm][im][iphi];// Qi-2,j -WW
        if (i >= Nx-2) { phiold[6] = phiold[1]; } else { im = i+2; jm = j  ; } phiold[6] = _xEntholdlocal[jm][im][iphi];// Qi+2,j -EE
      } else {
        im = i-2; jm = j  ; phiold[5] = _xEntholdlocal[jm][im][iphi];// Qi-2,j -WW
        im = i+2; jm = j  ; phiold[6] = _xEntholdlocal[jm][im][iphi];// Qi+2,j -EE
      }
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
      ierr = AdvectionResidual(v,phis,dx,dz,ADV_FROMM,&adv); CHKERRQ(ierr);

      // update gamma
      xx[j][i][idx] = (phi[0]-phiold[0])/usr->nd->dt - adv;
    }
  }

  // Restore arrays
  ierr = DMStagVecRestoreArray(dmmatProp,xmatProplocal,&xx); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dmmatProp,xmatProplocal,INSERT_VALUES,xmatProp); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dmmatProp,xmatProplocal,INSERT_VALUES,xmatProp); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmmatProp,&xmatProplocal); CHKERRQ(ierr);

  ierr = DMStagRestoreProductCoordinateArraysRead(dmPV,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArrayRead(dmPV,xPVlocal,&_xPVlocal);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmPV, &xPVlocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArrayRead(dmEnth,xEnthlocal,&_xEnthlocal);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmEnth, &xEnthlocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArrayRead(dmEnth,xEntholdlocal,&_xEntholdlocal);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmEnth, &xEntholdlocal); CHKERRQ(ierr);

  PetscTime(&tlog[1]);
  if (usr->par->log_info) {
    printf("  ComputeGamma: total                                %1.2e\n",tlog[1]-tlog[0]);
  }

  PetscFunctionReturn(0);
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

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_xHC_ts%d",usr->par->fdir_out,usr->nd->istep);
  ierr = DMStagViewBinaryPython(usr->dmHC,usr->xHC,fout);CHKERRQ(ierr);

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_xEnth_ts%d",usr->par->fdir_out,usr->nd->istep);
  ierr = DMStagViewBinaryPython(usr->dmEnth,usr->xEnth,fout);CHKERRQ(ierr);

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_xVel_ts%d",usr->par->fdir_out,usr->nd->istep);
  ierr = DMStagViewBinaryPython(usr->dmVel,usr->xVel,fout);CHKERRQ(ierr);

  if (usr->nd->istep > 0) {
    // coefficients
    ierr = FDPDEGetCoefficient(fdHC,&dmHCcoeff,&xHCcoeff);CHKERRQ(ierr);
    ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_xHCcoeff_ts%d",usr->par->fdir_out,usr->nd->istep);
    ierr = DMStagViewBinaryPython(dmHCcoeff,xHCcoeff,fout);CHKERRQ(ierr);

    ierr = FDPDEGetCoefficient(fdPV,&dmPVcoeff,&xPVcoeff);CHKERRQ(ierr);
    ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_xPVcoeff_ts%d",usr->par->fdir_out,usr->nd->istep);
    ierr = DMStagViewBinaryPython(dmPVcoeff,xPVcoeff,fout);CHKERRQ(ierr);

    // material properties eta, permeability, density
    ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_matProp_ts%d",usr->par->fdir_out,usr->nd->istep);
    ierr = DMStagViewBinaryPython(usr->dmmatProp,usr->xmatProp,fout);CHKERRQ(ierr);
  }

  // residuals
  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_resPV_ts%d",usr->par->fdir_out,usr->nd->istep);
  ierr = DMStagViewBinaryPython(usr->dmPV,fdPV->r,fout);CHKERRQ(ierr);

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_resHC_ts%d",usr->par->fdir_out,usr->nd->istep);
  ierr = DMStagViewBinaryPython(usr->dmHC,fdHC->r,fout);CHKERRQ(ierr);

  // lithostatic pressure
  ierr = FDPDEEnthalpyGetPressure(fdHC,&dmP,&xP);CHKERRQ(ierr);
  ierr = FDPDEEnthalpyGetPrevPressure(fdHC,&xPprev);CHKERRQ(ierr);

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_xPressure_ts%d",usr->par->fdir_out,usr->nd->istep);
  ierr = DMStagViewBinaryPython(dmP,xP,fout);CHKERRQ(ierr);

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_xPressurePrev_ts%d",usr->par->fdir_out,usr->nd->istep);
  ierr = DMStagViewBinaryPython(dmP,xPprev,fout);CHKERRQ(ierr);

  ierr = VecDestroy(&xP);CHKERRQ(ierr);
  ierr = VecDestroy(&xPprev);CHKERRQ(ierr);
  ierr = DMDestroy(&dmP);CHKERRQ(ierr);

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
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->scal->v,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->scal->t,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->scal->K,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->scal->P,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->scal->eta,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->scal->rho,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->scal->H,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->scal->Gamma,1,PETSC_DOUBLE);CHKERRQ(ierr);

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
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->nd->L,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->nd->H,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->nd->xmin,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->nd->zmin,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->nd->age,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->nd->visc_ratio,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->nd->eta_min,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->nd->eta_max,1,PETSC_DOUBLE);CHKERRQ(ierr);

  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['L'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['H'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['xmin'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['zmin'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['age'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['visc_ratio'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['eta_min'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['eta_max'] = v\n");

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
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->nd->alpha_s,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->nd->beta_s,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->nd->A,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->nd->S,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->nd->PeT,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->nd->PeC,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->nd->thetaS,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->nd->G,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->nd->RM,1,PETSC_DOUBLE);CHKERRQ(ierr);

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
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->par->C0,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->par->DC,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->par->T0,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,(void*)&usr->par->DT,1,PETSC_DOUBLE);CHKERRQ(ierr);

  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['C0'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['DC'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['T0'] = v\n");
  fprintf(fp,"    v = io.readReal(fp)\n"); fprintf(fp,"    data['DT'] = v\n");

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
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->scal->v,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->scal->t,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->scal->K,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->scal->P,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->scal->eta,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->scal->rho,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->scal->H,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->scal->Gamma,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);

  // parameters - nd
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->L,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->H,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->xmin,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->zmin,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->age,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->visc_ratio,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->eta_min,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->eta_max,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);

  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->istep,1,NULL,PETSC_INT);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->t,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->dt,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->tmax,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->dtmax,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);

  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->delta,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->alpha_s,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->beta_s,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->A,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->S,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->PeT,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->PeC,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->thetaS,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->G,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->nd->RM,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);

  // these bag parameters are needed for scaling in python
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->par->C0,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->par->DC,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->par->T0,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,(void*)&usr->par->DT,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);

  // save timestep
  PetscInt    tstep, tout;
  PetscScalar tmax, dtmax;

  tstep = usr->par->tstep;
  tout  = usr->par->tout;
  tmax  = usr->par->tmax;
  dtmax = usr->par->dtmax;

  // save buoyancy
  PetscInt    buoy_phi, buoy_C, buoy_T;
  PetscScalar beta;

  buoy_phi = usr->par->buoy_phi;
  buoy_C   = usr->par->buoy_C;
  buoy_T   = usr->par->buoy_T;
  beta   = usr->par->beta;

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

  PetscFunctionReturn(0);
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
  PetscErrorCode ierr;
  
  PetscFunctionBeginUser;

  usr->nd->istep = usr->par->restart;
  ierr = PetscSNPrintf(usr->par->fdir_out,sizeof(usr->par->fdir_out),"Timestep%d",usr->nd->istep);

  // load time data
  ierr = LoadParametersFromFile(usr);CHKERRQ(ierr);

  // correct restart variable from bag
  usr->par->restart = usr->nd->istep;
  ierr = InputPrintData(usr);CHKERRQ(ierr);

  // load PV data
  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_xPV_ts%d",usr->par->fdir_out,usr->nd->istep);
  ierr = DMStagReadBinaryPython(&dm,&x,fout);CHKERRQ(ierr);
  ierr = VecCopy(x,usr->xPV);CHKERRQ(ierr);
  ierr = VecCopy(x,fdPV->xguess);CHKERRQ(ierr);
  ierr = VecDestroy(&x); CHKERRQ(ierr);
  ierr = DMDestroy(&dm); CHKERRQ(ierr);
  
  // load HC data
  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_xHC_ts%d",usr->par->fdir_out,usr->nd->istep);
  ierr = DMStagReadBinaryPython(&dm,&x,fout);CHKERRQ(ierr);
  ierr = VecCopy(x,usr->xHC);CHKERRQ(ierr);
  ierr = VecCopy(x,fdHC->xguess);CHKERRQ(ierr);
  ierr = VecDestroy(&x); CHKERRQ(ierr);
  ierr = DMDestroy(&dm); CHKERRQ(ierr);

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_resPV_ts%d",usr->par->fdir_out,usr->nd->istep);
  ierr = DMStagReadBinaryPython(&dm,&x,fout);CHKERRQ(ierr);
  ierr = VecCopy(x,fdPV->r);CHKERRQ(ierr);
  ierr = VecDestroy(&x); CHKERRQ(ierr);
  ierr = DMDestroy(&dm); CHKERRQ(ierr);

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_resHC_ts%d",usr->par->fdir_out,usr->nd->istep);
  ierr = DMStagReadBinaryPython(&dm,&x,fout);CHKERRQ(ierr);
  ierr = VecCopy(x,fdHC->r);CHKERRQ(ierr);
  ierr = VecDestroy(&x); CHKERRQ(ierr);
  ierr = DMDestroy(&dm); CHKERRQ(ierr);
  
  // load lithostatic pressure
  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_xPressure_ts%d",usr->par->fdir_out,usr->nd->istep);
  ierr = DMStagReadBinaryPython(&dm,&x,fout);CHKERRQ(ierr);
  ierr = FDPDEEnthalpyGetPressure(fdHC,&dmP,&xP);CHKERRQ(ierr);
  ierr = VecCopy(x,xP);CHKERRQ(ierr);
  ierr = VecDestroy(&xP);CHKERRQ(ierr);
  ierr = VecDestroy(&x); CHKERRQ(ierr);
  ierr = DMDestroy(&dm); CHKERRQ(ierr);

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_xPressurePrev_ts%d",usr->par->fdir_out,usr->nd->istep);
  ierr = DMStagReadBinaryPython(&dm,&x,fout);CHKERRQ(ierr);
  ierr = FDPDEEnthalpyGetPrevPressure(fdHC,&xPprev);CHKERRQ(ierr);
  ierr = VecCopy(x,xPprev);CHKERRQ(ierr);
  ierr = VecDestroy(&xPprev);CHKERRQ(ierr);
  ierr = VecDestroy(&x); CHKERRQ(ierr);
  ierr = DMDestroy(&dm); CHKERRQ(ierr);
  ierr = DMDestroy(&dmP);CHKERRQ(ierr);

  // load Enthalpy diagnostics
  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_xEnth_ts%d",usr->par->fdir_out,usr->nd->istep);
  ierr = DMStagReadBinaryPython(&dmEnth,&x,fout);CHKERRQ(ierr);
  usr->dmEnth = dmEnth;
  ierr = VecDuplicate(x,&usr->xEnth);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&usr->xEnthold);CHKERRQ(ierr);
  ierr = VecCopy(x,usr->xEnth);CHKERRQ(ierr);
  ierr = VecCopy(x,usr->xEnthold);CHKERRQ(ierr); // this should be loaded from file too
  ierr = VecDestroy(&x); CHKERRQ(ierr);

  // load velocity
  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_xVel_ts%d",usr->par->fdir_out,usr->nd->istep);
  ierr = DMStagReadBinaryPython(&dm,&x,fout);CHKERRQ(ierr);
  ierr = VecCopy(x,usr->xVel);CHKERRQ(ierr);
  ierr = VecDestroy(&x); CHKERRQ(ierr);
  ierr = DMDestroy(&dm); CHKERRQ(ierr);

  // initialize guess and previous solution in fdHC
  ierr = FDPDEEnthalpyGetPrevSolution(fdHC,&xHCprev);CHKERRQ(ierr);
  ierr = VecCopy(usr->xHC,xHCprev);CHKERRQ(ierr);
  ierr = VecDestroy(&xHCprev);CHKERRQ(ierr);

  // load coefficient structure
  ierr = FDPDEGetCoefficient(fdHC,&dmHCcoeff,&xHCcoeff);CHKERRQ(ierr);
  ierr = FDPDEEnthalpyGetPrevCoefficient(fdHC,&xHCcoeffprev);CHKERRQ(ierr);

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_xHCcoeff_ts%d",usr->par->fdir_out,usr->nd->istep);
  ierr = DMStagReadBinaryPython(&dm,&x,fout);CHKERRQ(ierr);
  ierr = VecCopy(x,xHCcoeffprev);CHKERRQ(ierr);
  ierr = VecCopy(x,xHCcoeff);CHKERRQ(ierr);
  ierr = VecDestroy(&x); CHKERRQ(ierr);
  ierr = DMDestroy(&dm); CHKERRQ(ierr);
  ierr = VecDestroy(&xHCcoeffprev);CHKERRQ(ierr);

  // Output load conditions
  ierr = DoOutput(fdPV,fdHC,usr);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}