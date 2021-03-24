#include "MORbuoyancy.h"

// ---------------------------------------
// SetInitialConditions
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "SetInitialConditions"
PetscErrorCode SetInitialConditions(FDPDE fdPV, FDPDE fdHC, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  DM             dmP, dmHCcoeff, dmEnth;
  Vec            xP, xPprev, xHCprev, xHCguess, xHCcoeffprev, xEnth;
  char           fout[FNAME_LENGTH];
  PetscErrorCode ierr;
  
  PetscFunctionBeginUser;

  // corner flow model for PV
  ierr = CornerFlow_MOR(usr);CHKERRQ(ierr);

  // half-space cooling model - initialize H, C
  ierr = HalfSpaceCooling_MOR(usr);CHKERRQ(ierr);

  // ierr = PetscSNPrintf(usr->par->fdir_out,sizeof(usr->par->fdir_out),"Timestep%d",usr->par->istep);
  // ierr = CreateDirectory(usr->par->fdir_out);CHKERRQ(ierr);
  // ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_xHC_HS_ts%d",usr->par->fdir_out,usr->par->istep);
  // ierr = DMStagViewBinaryPython(usr->dmHC,usr->xHC,fout);CHKERRQ(ierr);

  // Update lithostatic pressure
  ierr = FDPDEEnthalpyGetPressure(fdHC,&dmP,&xP);CHKERRQ(ierr);
  ierr = UpdateLithostaticPressure(dmP,xP,usr);CHKERRQ(ierr);
  ierr = FDPDEEnthalpyGetPrevPressure(fdHC,&xPprev);CHKERRQ(ierr);
  ierr = VecCopy(xP,xPprev);CHKERRQ(ierr);
  ierr = VecDestroy(&xP);CHKERRQ(ierr);
  ierr = VecDestroy(&xPprev);CHKERRQ(ierr);
  ierr = DMDestroy(&dmP);CHKERRQ(ierr);

  // Update Enthalpy diagnostics
  ierr = FDPDEEnthalpyUpdateDiagnostics(fdHC,usr->dmHC,usr->xHC,&dmEnth,&xEnth); CHKERRQ(ierr);
  usr->dmEnth = dmEnth;
  ierr = VecDuplicate(xEnth,&usr->xEnth);CHKERRQ(ierr);
  ierr = VecCopy(xEnth,usr->xEnth);CHKERRQ(ierr);
  ierr = VecDestroy(&xEnth);CHKERRQ(ierr);

  // ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_xEnth0_ts%d",usr->par->fdir_out,usr->par->istep);
  // ierr = DMStagViewBinaryPython(usr->dmEnth,usr->xEnth,fout);CHKERRQ(ierr);

  // Correct H-S*phi and C=Cs to ensure phi=0
  ierr = CorrectInitialHCZeroPorosity(usr->dmEnth,usr->xEnth,usr);CHKERRQ(ierr);

  // ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_xHC_HS2_ts%d",usr->par->fdir_out,usr->par->istep);
  // ierr = DMStagViewBinaryPython(usr->dmHC,usr->xHC,fout);CHKERRQ(ierr);

  // Update Enthalpy again
  ierr = FDPDEEnthalpyUpdateDiagnostics(fdHC,usr->dmHC,usr->xHC,NULL,&xEnth); CHKERRQ(ierr);
  ierr = VecCopy(xEnth,usr->xEnth);CHKERRQ(ierr);
  ierr = VecDestroy(&xEnth);CHKERRQ(ierr);

  // Extract porosity and temperature and set phi=0.0
  ierr = ExtractTemperaturePorosity(usr->dmEnth,usr->xEnth,usr,PETSC_TRUE);CHKERRQ(ierr);
  ierr = VecCopy(usr->xphiT,usr->xphiTold);CHKERRQ(ierr);

  // Update fluid velocity to zero and v=vs
  ierr = ComputeFluidAndBulkVelocity(usr->dmPV,usr->xPV,usr->dmHC,usr->xphiT,usr->dmVel,usr->xVel,usr);CHKERRQ(ierr);

  // Initialize guess and previous solution in fdHC
  ierr = FDPDEEnthalpyGetPrevSolution(fdHC,&xHCprev);CHKERRQ(ierr);
  ierr = VecCopy(usr->xHC,xHCprev);CHKERRQ(ierr);
  ierr = FDPDEGetSolutionGuess(fdHC,&xHCguess);CHKERRQ(ierr);
  ierr = VecCopy(xHCprev,xHCguess);CHKERRQ(ierr);
  ierr = VecDestroy(&xHCprev);CHKERRQ(ierr);
  ierr = VecDestroy(&xHCguess);CHKERRQ(ierr);

  // Set initial coefficient structure
  ierr = FDPDEGetCoefficient(fdHC,&dmHCcoeff,NULL);CHKERRQ(ierr);
  ierr = FDPDEEnthalpyGetPrevCoefficient(fdHC,&xHCcoeffprev);CHKERRQ(ierr);
  ierr = FormCoefficient_HC(fdHC,usr->dmHC,usr->xHC,dmHCcoeff,xHCcoeffprev,usr);CHKERRQ(ierr);

  // Output prev coefficient
  ierr = PetscSNPrintf(usr->par->fdir_out,sizeof(usr->par->fdir_out),"Timestep%d",usr->par->istep);
  ierr = CreateDirectory(usr->par->fdir_out);CHKERRQ(ierr);
  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_xHCcoeff_ts%d",usr->par->fdir_out,usr->par->istep);
  ierr = DMStagViewBinaryPython(dmHCcoeff,xHCcoeffprev,fout);CHKERRQ(ierr);
  ierr = VecDestroy(&xHCcoeffprev);CHKERRQ(ierr);

  // // Initialize guess fdPV
  // ierr = FDPDEGetSolutionGuess(fdPV,&xPV);CHKERRQ(ierr);
  // ierr = VecCopy(usr->xPV,xPV);CHKERRQ(ierr);
  // ierr = VecDestroy(&xPV);CHKERRQ(ierr);

  // Output initial conditions
  ierr = DoOutput(fdPV,fdHC,usr);CHKERRQ(ierr);

  PetscFunctionReturn(0);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;

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

  ierr = DMStagGetGlobalSizes(dm, &Nx, &Nz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dm, &xlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm, xlocal, &xx); CHKERRQ(ierr);

// Get dm coordinates array
  ierr = DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter);CHKERRQ(ierr); 
  ierr = DMStagGetProductCoordinateLocationSlot(dm,LEFT,&iprev);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,RIGHT,&inext);CHKERRQ(ierr); 

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      PetscScalar xp, zp, v[2], p;

      // Vx
      xp = coordx[i][iprev ]; 
      zp = coordz[j][icenter];
      evaluate_CornerFlow_MOR(C1, C4, u0, eta0, xp, zp, v, &p);
      ierr = DMStagGetLocationSlot(dm, LEFT, 0, &idx); CHKERRQ(ierr);
      xx[j][i][idx] = v[0];

      if (i == Nx-1) {
        xp = coordx[i][inext  ];
        zp = coordz[j][icenter];
        evaluate_CornerFlow_MOR(C1, C4, u0, eta0, xp, zp, v, &p);
        ierr = DMStagGetLocationSlot(dm, RIGHT, 0, &idx); CHKERRQ(ierr);
        xx[j][i][idx] = v[0];
      }

      // Vz
      xp = coordx[i][icenter];
      zp = coordz[j][iprev  ];
      evaluate_CornerFlow_MOR(C1, C4, u0, eta0, xp, zp, v, &p);
      ierr = DMStagGetLocationSlot(dm, DOWN, 0, &idx); CHKERRQ(ierr);
      xx[j][i][idx] = v[1];

      if (j == Nz-1) {
        xp = coordx[i][icenter];
        zp = coordz[j][inext  ];
        evaluate_CornerFlow_MOR(C1, C4, u0, eta0, xp, zp, v, &p);
        ierr = DMStagGetLocationSlot(dm, UP, 0, &idx); CHKERRQ(ierr);
        xx[j][i][idx] = v[1];
      }
    
      // P
      xp = coordx[i][icenter];
      zp = coordz[j][icenter];
      evaluate_CornerFlow_MOR(C1, C4, u0, eta0, xp, zp, v, &p);
      ierr = DMStagGetLocationSlot(dm, ELEMENT, 0, &idx); CHKERRQ(ierr);
      xx[j][i][idx] = p;
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
// Half-Space Cooling model for MOR
// ---------------------------------------
PetscErrorCode HalfSpaceCooling_MOR(void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz, iH, iC, icenter;
  PetscScalar    **coordx,**coordz, ***xx, Cs0, Tm, xsill;
  Vec            x, xlocal;
  DM             dm;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  dm  = usr->dmHC;
  x   = usr->xHC;
  Cs0 = usr->par->C0;
  Tm  = (usr->par->Tp-T_KELVIN)*exp(-usr->nd->A*usr->nd->zmin)+T_KELVIN;
  xsill = usr->nd->xsill/usr->par->xsill_extract;

  ierr = DMStagGetGlobalSizes(dm, &Nx, &Nz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter);CHKERRQ(ierr); 
  ierr = DMStagGetLocationSlot(dm, ELEMENT, 0, &iH); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm, ELEMENT, 1, &iC); CHKERRQ(ierr);

  ierr = DMCreateLocalVector(dm, &xlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm, xlocal, &xx); CHKERRQ(ierr);

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      PetscScalar T, nd_T, age, Ta;

      // half-space cooling temperature
      if (usr->par->extract_mech==0) {
        age  = dim_param(coordx[i][icenter],usr->scal->x)/dim_param(usr->nd->U0,usr->scal->v);
      } else if ((usr->par->extract_mech==1) || (usr->par->extract_mech==2)) {
        age  = dim_param(coordx[i][icenter]-xsill,usr->scal->x)/dim_param(usr->nd->U0,usr->scal->v);
        if (age <= 0.0) age = dim_param(coordx[0][icenter],usr->scal->x)/dim_param(usr->nd->U0,usr->scal->v);
      }
      T = HalfSpaceCoolingTemp(Tm,usr->par->Ts,-dim_param(coordz[j][icenter],usr->scal->x),usr->par->kappa,age); 

      // check adiabat in the mantle
      {
        Ta  = (usr->par->Tp-T_KELVIN)*exp(-usr->nd->A*coordz[j][icenter])+T_KELVIN;
        if (T>Ta) T = Ta;
      }
      nd_T = (T - usr->par->T0)/usr->par->DT;

      // enthalpy H = S*phi+T (phi=0)
      xx[j][i][iH] = nd_T;

      // initial bulk composition C0 = Cs0 (phi=0)
      xx[j][i][iC] = (Cs0-usr->par->C0)/usr->par->DC;
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
      rho  = usr->par->rho0; // this should be bulk density?
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
  PetscInt       i, j, sx, sz, nx, nz, iH, iC;
  PetscScalar    ***xx;
  Vec            x, xlocal, xnewlocal;
  DM             dm;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  dm  = usr->dmHC;
  x   = usr->xHC;

  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm, ELEMENT, 0, &iH); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm, ELEMENT, 1, &iC); CHKERRQ(ierr);

  ierr = DMGetLocalVector(dm, &xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm, x, INSERT_VALUES, xlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm, xlocal, &xx); CHKERRQ(ierr);

  ierr = DMGetLocalVector(dmEnth, &xnewlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dmEnth, xEnth, INSERT_VALUES, xnewlocal); CHKERRQ(ierr);

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      DMStagStencil point[3];
      PetscScalar   xs[3],phi;
      point[0].i = i; point[0].j = j; point[0].loc = DMSTAG_ELEMENT; point[0].c = 3; // phi // add labels for Enthalpy dofs
      point[1].i = i; point[1].j = j; point[1].loc = DMSTAG_ELEMENT; point[1].c = 7; // CS
      point[2].i = i; point[2].j = j; point[2].loc = DMSTAG_ELEMENT; point[2].c = 9; // CF
      ierr = DMStagVecGetValuesStencil(dmEnth,xnewlocal,3,point,xs); CHKERRQ(ierr);
      phi = xs[0]*usr->par->phi_init;

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
  ierr = DMRestoreLocalVector(dmEnth, &xnewlocal); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// Extract T and porosity from Enthalpy update; PETSC_FALSE - zero porosity
// ---------------------------------------
PetscErrorCode ExtractTemperaturePorosity(DM dmEnth, Vec xEnth, void *ctx, PetscBool flag)
{
  UsrData       *usr = (UsrData*) ctx;
  PetscInt       i, j, sx, sz, nx, nz, iphi, iT;
  PetscScalar    ***xx;
  Vec            x, xlocal, xnewlocal;
  DM             dm;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  dm  = usr->dmHC;
  x   = usr->xphiT;

  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm, ELEMENT, 0, &iphi); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm, ELEMENT, 1, &iT); CHKERRQ(ierr);

  ierr = DMCreateLocalVector(dm,&xlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm, x, INSERT_VALUES, xlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm, xlocal, &xx); CHKERRQ(ierr);

  ierr = DMGetLocalVector(dmEnth, &xnewlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dmEnth, xEnth, INSERT_VALUES, xnewlocal); CHKERRQ(ierr);

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      DMStagStencil point;
      PetscScalar   phi = 0.0, T = 0.0;

      point.i = i; point.j = j; point.loc = DMSTAG_ELEMENT; point.c = 1; 
      ierr = DMStagVecGetValuesStencil(dmEnth,xnewlocal,1,&point,&T); CHKERRQ(ierr);
      point.c = 3; ierr = DMStagVecGetValuesStencil(dmEnth,xnewlocal,1,&point,&phi); CHKERRQ(ierr);
      if (flag) phi *= usr->par->phi_init;

      xx[j][i][iphi] = phi;
      xx[j][i][iT]   = T;
    }
  }

  // Restore arrays
  ierr = DMStagVecRestoreArray(dm,xlocal,&xx); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm,xlocal,INSERT_VALUES,x); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,xlocal,INSERT_VALUES,x); CHKERRQ(ierr);
  ierr = VecDestroy(&xlocal);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmEnth, &xnewlocal); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// Compute Fluid and Bulk velocity from PV and porosity solutions
// ---------------------------------------
PetscErrorCode ComputeFluidAndBulkVelocity(DM dmPV, Vec xPV, DM dmHC, Vec xphiT, DM dmVel, Vec xVel, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  PetscInt       i, j, ii, sx, sz, nx, nz, Nx, Nz, idx,iprev,inext,icenter;
  PetscScalar    ***xx, dx, dz, k_hat[4];
  PetscScalar    **coordx,**coordz;
  Vec            xVellocal,xPVlocal,xphiTlocal;
  PetscErrorCode ierr;
  PetscFunctionBegin;

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

  ierr = DMGetLocalVector(dmHC, &xphiTlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dmHC, xphiT, INSERT_VALUES, xphiTlocal); CHKERRQ(ierr);

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      DMStagStencil point[9],pointQ[5];
      PetscScalar pv[9], xp[3],zp[3], Q[5];
      PetscScalar vf, K, Bf, vs[4], gradP[4], phi[4];

      // get PV data
      point[0].i = i; point[0].j = j; point[0].loc = LEFT;    point[0].c = 0;
      point[1].i = i; point[1].j = j; point[1].loc = RIGHT;   point[1].c = 0;
      point[2].i = i; point[2].j = j; point[2].loc = DOWN;    point[2].c = 0;
      point[3].i = i; point[3].j = j; point[3].loc = UP;      point[3].c = 0;
      point[4].i = i; point[4].j = j; point[4].loc = ELEMENT; point[4].c = 0;

      point[5].i = i-1; point[5].j = j  ; point[5].loc = ELEMENT; point[5].c = 0;
      point[6].i = i+1; point[6].j = j  ; point[6].loc = ELEMENT; point[6].c = 0;
      point[7].i = i  ; point[7].j = j-1; point[7].loc = ELEMENT; point[7].c = 0;
      point[8].i = i  ; point[8].j = j+1; point[8].loc = ELEMENT; point[8].c = 0;

      // correct for domain edges 
      if (i == 0   ) point[5] = point[0];
      if (i == Nx-1) point[6] = point[0];
      if (j == 0   ) point[7] = point[0];
      if (j == Nz-1) point[8] = point[0];

      ierr = DMStagVecGetValuesStencil(dmPV,xPVlocal,9,point,pv); CHKERRQ(ierr);

      // grid spacing - assume constant
      dx = coordx[i][inext]-coordx[i][iprev];
      dz = coordz[j][inext]-coordz[j][iprev];

      vs[0] = pv[0]; 
      vs[1] = pv[1]; 
      vs[2] = pv[2]; 
      vs[3] = pv[3]; 

      gradP[0] = (pv[4]-pv[5])/dx;
      gradP[1] = (pv[6]-pv[4])/dx;
      gradP[2] = (pv[4]-pv[7])/dz;
      gradP[3] = (pv[8]-pv[4])/dz;

      // porosity
      pointQ[0].i = i  ; pointQ[0].j = j  ; pointQ[0].loc = ELEMENT; pointQ[0].c = 0;
      pointQ[1].i = i-1; pointQ[1].j = j  ; pointQ[1].loc = ELEMENT; pointQ[1].c = 0;
      pointQ[2].i = i+1; pointQ[2].j = j  ; pointQ[2].loc = ELEMENT; pointQ[2].c = 0;
      pointQ[3].i = i  ; pointQ[3].j = j-1; pointQ[3].loc = ELEMENT; pointQ[3].c = 0;
      pointQ[4].i = i  ; pointQ[4].j = j+1; pointQ[4].loc = ELEMENT; pointQ[4].c = 0;

      if (i == 0   ) pointQ[1] = pointQ[0];
      if (i == Nx-1) pointQ[2] = pointQ[0];
      if (j == 0   ) pointQ[3] = pointQ[0];
      if (j == Nz-1) pointQ[4] = pointQ[0];

      if (i == 0   ) { xp[0] = coordx[i][icenter];} else { xp[0] = coordx[i-1][icenter];}
      if (i == Nx-1) { xp[2] = coordx[i][icenter];} else { xp[2] = coordx[i+1][icenter];}
      if (j == 0   ) { zp[0] = coordz[j][icenter];} else { zp[0] = coordz[j-1][icenter];}
      if (j == Nz-1) { zp[2] = coordz[j][icenter];} else { zp[2] = coordz[j+1][icenter];}
      xp[1] = coordx[i][icenter];
      zp[1] = coordz[j][icenter];

      ierr = DMStagVecGetValuesStencil(dmHC,xphiTlocal,5,pointQ,Q); CHKERRQ(ierr);

      // porosity on edges
      // phi[0] = interp1DLin_3Points(coordx[i][iprev],xp[0],Q[1],xp[1],Q[0],xp[2],Q[2]); 
      // phi[1] = interp1DLin_3Points(coordx[i][inext],xp[0],Q[1],xp[1],Q[0],xp[2],Q[2]); 
      // phi[2] = interp1DLin_3Points(coordz[j][iprev],zp[0],Q[3],zp[1],Q[0],zp[2],Q[4]); 
      // phi[3] = interp1DLin_3Points(coordz[j][inext],zp[0],Q[3],zp[1],Q[0],zp[2],Q[4]); 

      phi[0] = (Q[1]+Q[0])*0.5; 
      phi[1] = (Q[2]+Q[0])*0.5; 
      phi[2] = (Q[3]+Q[0])*0.5; 
      phi[3] = (Q[4]+Q[0])*0.5; 
      
      for (ii = 0; ii < 4; ii++) {
        // permeability
        K = Permeability(phi[ii],usr->par->phi0,usr->par->phi_max,usr->par->n,usr->par->phi_cutoff);

        // fluid buoyancy
        Bf = 0.0; //FluidBuoyancy(T,CF,usr->nd->alpha_s,usr->nd->beta_s);

        // fluid velocity
        vf = FluidVelocity(vs[ii],phi[ii],gradP[ii],Bf,K,k_hat[ii],usr->par->phi_cutoff);
        ierr = DMStagGetLocationSlot(dmVel, point[ii].loc,0, &idx); CHKERRQ(ierr);

        if ((i==0) && (point[ii].loc == LEFT)) vf = 0.0; // vfx on left boundary
        xx[j][i][idx] = vf;

        // bulk velocity
        ierr = DMStagGetLocationSlot(dmVel, point[ii].loc,1, &idx); CHKERRQ(ierr);
        xx[j][i][idx] = BulkVelocity(vs[ii],vf,phi[ii]);
      }
    }
  }

  // Restore arrays
  ierr = DMStagVecRestoreArray(dmVel,xVellocal,&xx); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dmVel,xVellocal,INSERT_VALUES,xVel); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dmVel,xVellocal,INSERT_VALUES,xVel); CHKERRQ(ierr);
  ierr = VecDestroy(&xVellocal); CHKERRQ(ierr);

  ierr = DMStagRestoreProductCoordinateArraysRead(dmPV,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmPV, &xPVlocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmHC, &xphiTlocal); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// Update eta, zeta, permeability, rho, rho_f, rho_s for output
// ---------------------------------------
PetscErrorCode UpdateMaterialProperties(DM dmHC, Vec xHC, Vec xphiT, DM dmEnth, Vec xEnth, DM dmmatProp, Vec xmatProp, void *ctx)
{
  UsrData        *usr = (UsrData*) ctx;
  NdParams       *nd;
  Params         *par;
  ScalParams     *scal;
  PetscInt       i, j, sx, sz, nx, nz, idx;
  PetscScalar    ***xx;
  Vec            xmatProplocal,xEnthlocal,xphiTlocal,xHClocal;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  nd  = usr->nd;
  par = usr->par;
  scal= usr->scal;

  ierr = DMStagGetCorners(dmmatProp,&sx,&sz,NULL,&nx,&nz,NULL,NULL,NULL,NULL); CHKERRQ(ierr);  
  ierr = DMCreateLocalVector(dmmatProp,&xmatProplocal);CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dmmatProp, xmatProplocal, &xx); CHKERRQ(ierr);

  ierr = DMGetLocalVector(dmEnth, &xEnthlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dmEnth, xEnth, INSERT_VALUES, xEnthlocal); CHKERRQ(ierr);

  ierr = DMGetLocalVector(dmHC, &xphiTlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dmHC, xphiT, INSERT_VALUES, xphiTlocal); CHKERRQ(ierr);

  ierr = DMGetLocalVector(dmHC, &xHClocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dmHC, xHC, INSERT_VALUES, xHClocal); CHKERRQ(ierr);

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      DMStagStencil point;
      PetscScalar   eta, zeta, K, rho, rhof, rhos, CF, CS, T, phi;

      point.i = i; point.j = j; point.loc = ELEMENT;
      point.c = 0; ierr = DMStagVecGetValuesStencil(dmHC,xphiTlocal,1,&point,&phi); CHKERRQ(ierr);
      point.c = 1; ierr = DMStagVecGetValuesStencil(dmHC,xphiTlocal,1,&point,&T); CHKERRQ(ierr);

      point.c = 7; ierr = DMStagVecGetValuesStencil(dmEnth,xEnthlocal,1,&point,&CS); CHKERRQ(ierr);
      point.c = 9; ierr = DMStagVecGetValuesStencil(dmEnth,xEnthlocal,1,&point,&CF); CHKERRQ(ierr);
      
      eta  = ShearViscosity(T*par->DT+par->T0,phi,par->EoR,par->Teta0,par->lambda,scal->eta,nd->eta_min,nd->eta_max,par->visc_shear);
      zeta = BulkViscosity(T*par->DT+par->T0,phi,par->EoR,par->Teta0,scal->eta,nd->visc_ratio,par->zetaExp,nd->eta_min,nd->eta_max,usr->par->phi_cutoff,par->visc_bulk);
      K    = Permeability(phi,usr->par->phi0,usr->par->phi_max,usr->par->n,usr->par->phi_cutoff);
       
      rhos = SolidDensity(par->rho0,par->drho,T,CS,nd->alpha_s,nd->beta_s,par->buoyancy);
      rhof = FluidDensity(par->rho0,par->drho,T,CF,nd->alpha_s,nd->beta_s,par->buoyancy);
      rho  = BulkDensity(rhos,rhof,phi,par->buoyancy);
      
      ierr = DMStagGetLocationSlot(dmmatProp, point.loc, 0, &idx); CHKERRQ(ierr);
      xx[j][i][idx] = eta;

      ierr = DMStagGetLocationSlot(dmmatProp, point.loc, 1, &idx); CHKERRQ(ierr);
      xx[j][i][idx] = zeta;

      ierr = DMStagGetLocationSlot(dmmatProp, point.loc, 2, &idx); CHKERRQ(ierr);
      xx[j][i][idx] = K;

      ierr = DMStagGetLocationSlot(dmmatProp, point.loc, 3, &idx); CHKERRQ(ierr);
      xx[j][i][idx] = rho;

      ierr = DMStagGetLocationSlot(dmmatProp, point.loc, 4, &idx); CHKERRQ(ierr);
      xx[j][i][idx] = rhof;

      ierr = DMStagGetLocationSlot(dmmatProp, point.loc, 5, &idx); CHKERRQ(ierr);
      xx[j][i][idx] = rhos;
    }
  }

  // Restore arrays
  ierr = DMStagVecRestoreArray(dmmatProp,xmatProplocal,&xx); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dmmatProp,xmatProplocal,INSERT_VALUES,xmatProp); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dmmatProp,xmatProplocal,INSERT_VALUES,xmatProp); CHKERRQ(ierr);
  ierr = VecDestroy(&xmatProplocal); CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(dmHC, &xphiTlocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmHC, &xHClocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmEnth,&xEnthlocal); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}