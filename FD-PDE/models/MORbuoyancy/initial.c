#include "MORbuoyancy.h"

// ---------------------------------------
// SetInitialConditions_HS (using half-space cooling model)
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "SetInitialConditions_HS"
PetscErrorCode SetInitialConditions_HS(FDPDE fdPV, FDPDE fdH, FDPDE fdC, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  char           fout[FNAME_LENGTH];
  PetscErrorCode ierr;
  
  PetscFunctionBeginUser;

  // corner flow model for PV
  ierr = CornerFlow_MOR(usr);CHKERRQ(ierr);

  // half-space cooling model
  ierr = HalfSpaceCooling_MOR(usr);CHKERRQ(ierr);

  // set initial porosity zero
  ierr = VecSet(usr->xphi,0.0);CHKERRQ(ierr);

  // set initial composition
  PetscScalar Cf0, Cs0;
  Cf0 = usr->par->C0-usr->par->DC;
  Cs0 = usr->par->C0;
  ierr = VecSet(usr->xCf,(Cf0-usr->par->C0)/usr->par->DC);CHKERRQ(ierr);
  ierr = VecSet(usr->xCs,(Cs0-usr->par->C0)/usr->par->DC);CHKERRQ(ierr);
  ierr = UpdateComposition(usr);CHKERRQ(ierr);

  // correct for solidus
  ierr = CorrectTemperatureForSolidus(usr);CHKERRQ(ierr);

  // transform xT for xTheta
  ierr = UpdateThetaFromTemp(usr);CHKERRQ(ierr);

  // calculate enthalpy
  ierr = UpdateEnthalpy(usr);CHKERRQ(ierr);

  ierr = DoOutput(usr);CHKERRQ(ierr);

  // set initial porosity to initphi = 1e-4
  // ierr = VecSet(usr->xphi,usr->par->initphi);CHKERRQ(ierr);

  // copy variables into fd-pde objects
  // ierr = FDPDEAdvDiffGetPrevSolution(fdH,&xHprev);CHKERRQ(ierr);
  // ierr = SetInitialH(dmH,xHprev,usr);CHKERRQ(ierr);

  // ierr = FDPDEGetCoefficient(fdphi,&dmphicoeff,NULL);CHKERRQ(ierr);
  // ierr = FDPDEAdvDiffGetPrevCoefficient(fdphi,&phicoeffprev);CHKERRQ(ierr);
  // ierr = SetInitialPorosityCoefficient(dmphicoeff,phicoeffprev,usr);CHKERRQ(ierr);

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
  eta0 = usr->nd->eta0;

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
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz, idx, icenter;
  PetscScalar    ***xx;
  PetscScalar    **coordx,**coordz;
  Vec            x, xlocal;
  DM             dm;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  dm = usr->dmHC;
  x  = usr->xT;

  ierr = DMStagGetGlobalSizes(dm, &Nx, &Nz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dm, &xlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm, xlocal, &xx); CHKERRQ(ierr);

// Get dm coordinates array
  ierr = DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter);CHKERRQ(ierr); 

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      PetscScalar T, age;
      age = dim_param(coordx[i][icenter],usr->scal->x)/dim_param(usr->nd->U0,usr->scal->v);
      T   = T_KELVIN + (usr->par->Tp-T_KELVIN)*erf(-dim_param(coordz[j][icenter],usr->scal->x)/(2.0*sqrt(usr->par->kappa*age)));
      ierr = DMStagGetLocationSlot(dm, ELEMENT, 0, &idx); CHKERRQ(ierr);
      xx[j][i][idx] = (T - usr->par->T0)/usr->par->DT;
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
// CorrectTemperatureForSolidus
// ---------------------------------------
PetscErrorCode CorrectTemperatureForSolidus(void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz, idx, icenter;
  PetscScalar    ***xx, ***xxCs;
  PetscScalar    **coordx,**coordz;
  Vec            x, xlocal, xCslocal;
  DM             dm;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  dm = usr->dmHC;
  x  = usr->xTsol;

  ierr = DMStagGetGlobalSizes(dm, &Nx, &Nz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  ierr = DMGetLocalVector(dm, &xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm, x, INSERT_VALUES, xlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm, xlocal, &xx); CHKERRQ(ierr);

  ierr = DMGetLocalVector(dm, &xCslocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm, usr->xCs, INSERT_VALUES, xCslocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm, xCslocal, &xxCs); CHKERRQ(ierr);

// Get dm coordinates array
  ierr = DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter);CHKERRQ(ierr); 

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      PetscScalar Tsol, Plith, rho;
      ierr = DMStagGetLocationSlot(dm, ELEMENT, 0, &idx); CHKERRQ(ierr);
      rho  = usr->par->rho0; // this should be bulk density
      Plith= LithostaticPressure(rho,usr->par->drho,coordz[j][icenter]);
      Tsol = Solidus(xxCs[j][i][idx],Plith,usr->nd->G,PETSC_FALSE);
      // correct for solidus
      xx[j][i][idx] = Tsol;
      // PetscPrintf(PETSC_COMM_WORLD,"# Temp HS = %1.12e Tsol = %1.12e Plith = %1.12e G = %1.12e  \n",xx[j][i][idx],Tsol,Plith,usr->nd->G);
    }
  }

  // Restore arrays
  ierr = DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArray(dm,xlocal,&xx); CHKERRQ(ierr);
  ierr = DMStagVecRestoreArray(dm,xCslocal,&xxCs); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&xlocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&xCslocal); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}