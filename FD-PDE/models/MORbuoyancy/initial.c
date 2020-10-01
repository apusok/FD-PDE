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

  // half-space cooling model - initialize T, C, H, phi
  ierr = HalfSpaceCooling_MOR(usr);CHKERRQ(ierr);

  // output variables
  ierr = DoOutput(usr);CHKERRQ(ierr);

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
  PetscScalar    ***xxT, ***xxphi, ***xxC, ***xxCf, ***xxCs, ***xxTsol, ***xxtheta, ***xxH;
  PetscScalar    **coordx,**coordz;
  PetscScalar    Cf0, Cs0;
  Vec            xTlocal, xphilocal, xClocal, xCflocal, xCslocal, xTsollocal, xthetalocal, xHlocal;
  DM             dm;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  dm  = usr->dmHC;
  Cf0 = usr->par->C0-usr->par->DC;
  Cs0 = usr->par->C0;

  ierr = DMStagGetGlobalSizes(dm, &Nx, &Nz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter);CHKERRQ(ierr); 
  ierr = DMStagGetLocationSlot(dm, ELEMENT, 0, &idx); CHKERRQ(ierr);

  ierr = DMCreateLocalVector(dm,&xTlocal); CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dm,&xphilocal); CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dm,&xClocal); CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dm,&xCflocal); CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dm,&xCslocal); CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dm,&xTsollocal); CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dm,&xthetalocal); CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dm,&xHlocal); CHKERRQ(ierr);

  ierr = DMStagVecGetArray(dm,xTlocal,  &xxT); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm,xphilocal,&xxphi); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm,xClocal,  &xxC); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm,xCflocal, &xxCf); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm,xCslocal, &xxCs); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm,xTsollocal,&xxTsol); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm,xthetalocal,&xxtheta); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm,xHlocal,  &xxH); CHKERRQ(ierr);

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      PetscScalar T, age, Plith, rho, Az;

      // half-space cooling temperature
      age = dim_param(coordx[i][icenter],usr->scal->x)/dim_param(usr->nd->U0,usr->scal->v);
      T   = T_KELVIN + (usr->par->Tp-T_KELVIN)*erf(-dim_param(coordz[j][icenter],usr->scal->x)/(2.0*sqrt(usr->par->kappa*age)));
      xxT[j][i][idx] = (T - usr->par->T0)/usr->par->DT;

      // initial porosity
      xxphi[j][i][idx] = 0.0;

      // initial phase composition
      xxCf[j][i][idx] = (Cf0-usr->par->C0)/usr->par->DC;
      xxCs[j][i][idx] = (Cs0-usr->par->C0)/usr->par->DC;

      // bulk composition C = phi*Cf+(1-phi)*Cs
      xxC[j][i][idx] = BulkComposition(xxCf[j][i][idx],xxCs[j][i][idx],xxphi[j][i][idx]);

      // solidus temperature
      rho  = usr->par->rho0; // this should be bulk density
      Plith= LithostaticPressure(rho,usr->par->drho,coordz[j][icenter]);
      xxTsol[j][i][idx] = Solidus(xxCs[j][i][idx],Plith,usr->nd->G,PETSC_FALSE);

      // potential temperature
      Az = -usr->nd->A*coordz[j][icenter]; 
      xxtheta[j][i][idx] = Temp2Theta(xxT[j][i][idx],Az);
      
      // enthalpy
      xxH[j][i][idx] = TotalEnthalpy(xxphi[j][i][idx],xxtheta[j][i][idx],Az,usr->nd->S,usr->nd->thetaS);

    }
  }

  // Restore arrays
  ierr = DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArray(dm,xTlocal,&xxT); CHKERRQ(ierr);
  ierr = DMStagVecRestoreArray(dm,xphilocal,&xxphi); CHKERRQ(ierr);
  ierr = DMStagVecRestoreArray(dm,xClocal,&xxC); CHKERRQ(ierr);
  ierr = DMStagVecRestoreArray(dm,xCflocal,&xxCf); CHKERRQ(ierr);
  ierr = DMStagVecRestoreArray(dm,xCslocal,&xxCs); CHKERRQ(ierr);
  ierr = DMStagVecRestoreArray(dm,xTsollocal,&xxTsol); CHKERRQ(ierr);
  ierr = DMStagVecRestoreArray(dm,xthetalocal,&xxtheta); CHKERRQ(ierr);
  ierr = DMStagVecRestoreArray(dm,xHlocal,&xxH); CHKERRQ(ierr);

  ierr = DMLocalToGlobalBegin(dm,xTlocal,INSERT_VALUES,usr->xT); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,xTlocal,INSERT_VALUES,usr->xT); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm,xphilocal,INSERT_VALUES,usr->xphi); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,xphilocal,INSERT_VALUES,usr->xphi); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm,xClocal,INSERT_VALUES,usr->xC); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,xClocal,INSERT_VALUES,usr->xC); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm,xCflocal,INSERT_VALUES,usr->xCf); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,xCflocal,INSERT_VALUES,usr->xCf); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm,xCslocal,INSERT_VALUES,usr->xCs); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,xCslocal,INSERT_VALUES,usr->xCs); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm,xTsollocal,INSERT_VALUES,usr->xTsol); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,xTsollocal,INSERT_VALUES,usr->xTsol); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm,xthetalocal,INSERT_VALUES,usr->xTheta); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,xthetalocal,INSERT_VALUES,usr->xTheta); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm,xHlocal,INSERT_VALUES,usr->xH); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,xHlocal,INSERT_VALUES,usr->xH); CHKERRQ(ierr);

  ierr = VecDestroy(&xTlocal); CHKERRQ(ierr);
  ierr = VecDestroy(&xphilocal); CHKERRQ(ierr);
  ierr = VecDestroy(&xClocal); CHKERRQ(ierr);
  ierr = VecDestroy(&xCflocal); CHKERRQ(ierr);
  ierr = VecDestroy(&xCslocal); CHKERRQ(ierr);
  ierr = VecDestroy(&xTsollocal); CHKERRQ(ierr);
  ierr = VecDestroy(&xthetalocal); CHKERRQ(ierr);
  ierr = VecDestroy(&xHlocal); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
