#include "stagridge.h"

// ---------------------------------------
// FormFunctionPV
// ---------------------------------------
PetscErrorCode FormFunctionPV(SNES snes, Vec x, Vec f, void *ctx)
{
  SolverCtx      *sol = (SolverCtx*) ctx;
  PetscInt       i, j, Nx, Nz, sx, sz, nx, nz;
  Vec            xlocal, coefflocal, flocal;
  PetscInt       nEntries, idx;
  PetscScalar    ***ff;
  PetscScalar    xx[4], fval, dx, dz;
  DMStagStencil  col[4];
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Assign pointers and other variables
  Nx = sol->grd->nx;
  Nz = sol->grd->nz;
  dx = sol->grd->dx;
  dz = sol->grd->dz;

  // Get local domain
  ierr = DMStagGetCorners(sol->dmPV, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  // Map global vectors to local domain
  ierr = DMGetLocalVector(sol->dmPV, &xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (sol->dmPV, x, INSERT_VALUES, xlocal); CHKERRQ(ierr);

  // Map coefficient data to local domain
  ierr = DMGetLocalVector(sol->dmCoeff, &coefflocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (sol->dmCoeff, sol->coeff, INSERT_VALUES, coefflocal); CHKERRQ(ierr);

  // Map global vectors to local domain
  ierr = DMCreateLocalVector(sol->dmPV, &flocal); CHKERRQ(ierr);

  // Get residual array
  ierr = DMStagVecGetArrayDOF(sol->dmPV, flocal, &ff); CHKERRQ(ierr);

  // ---------------------------------------
  // Interior domain
  // ---------------------------------------
  // Loop over elements
  for (j = sz; j<sz+nz; ++j) {
    for (i = sx; i<sx+nx; ++i) {
      // 1) Continuity equation
      // Get stencil values
      nEntries = 4;
      col[0].i = i; col[0].j = j; col[0].loc = LEFT;    col[0].c = 0;
      col[1].i = i; col[1].j = j; col[1].loc = RIGHT;   col[1].c = 0;
      col[2].i = i; col[2].j = j; col[2].loc = DOWN;    col[2].c = 0;
      col[3].i = i; col[3].j = j; col[3].loc = UP;      col[3].c = 0;
      ierr = DMStagVecGetValuesStencil(sol->dmPV, xlocal, nEntries, col, xx); CHKERRQ(ierr);
      
      // Calculate residual
      fval = (xx[1]-xx[0])/dx + (xx[3]-xx[2])/dz;

      // Set residual in array
      ierr = DMStagGetLocationSlot(sol->dmPV, ELEMENT, 0, &idx); CHKERRQ(ierr);
      ff[j][i][idx] = fval;

      // 2) X-Momentum equation
      if ((i > 0) && (j > 0) && (j < Nz-1)) {
        // Get residual
        ierr = XMomentumResidual(sol, xlocal, i, j, NONE, &fval); CHKERRQ(ierr);

        // Set residual in array
        ierr = DMStagGetLocationSlot(sol->dmPV, LEFT, 0, &idx); CHKERRQ(ierr);
        ff[j][i][idx] = fval;
      }

      // 3) Z-Momentum equation
      if ((j > 0) && (i > 0) && (i < Nx-1)) {
        // Get residual
        ierr = ZMomentumResidual(sol, xlocal, coefflocal, i, j, NONE, &fval); CHKERRQ(ierr);

        // Set residual in array
        ierr = DMStagGetLocationSlot(sol->dmPV, DOWN, 0, &idx); CHKERRQ(ierr);
        ff[j][i][idx] = fval;
      }
    }
  }

  // ---------------------------------------
  // Boundary conditions
  // ---------------------------------------
  if (sol->grd->mtype == MOR){
    // MOR Analytic
    ierr = BoundaryConditions_MORAnalytic(sol, xlocal, ff); CHKERRQ(ierr);
  } else {
    // General (SolCx, Free  slip, etc)
    ierr = BoundaryConditions_General(sol, xlocal, coefflocal, ff); CHKERRQ(ierr);
  }

  // Restore arrays, local vectors
  ierr = DMStagVecRestoreArrayDOF(sol->dmPV,flocal,&ff); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(sol->dmCoeff,&coefflocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(sol->dmPV,   &xlocal    ); CHKERRQ(ierr);

  // ---------------------------------------
  // Return and clean up
  // ---------------------------------------
  // Map local to global
  ierr = DMLocalToGlobalBegin(sol->dmPV,flocal,INSERT_VALUES,f); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (sol->dmPV,flocal,INSERT_VALUES,f); CHKERRQ(ierr);

  ierr = VecDestroy(&flocal); CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

// ---------------------------------------
// FormFunctionHT
// ---------------------------------------
PetscErrorCode FormFunctionHT(SNES snes, Vec x, Vec f, void *ctx)
{
  SolverCtx      *sol = (SolverCtx*) ctx;
  PetscInt       i, j, Nx, Nz, sx, sz, nx, nz, idx;
  Vec            xlocal, flocal, coefflocal, pvlocal;
  PetscScalar    ***ff, fval;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Assign pointers and other variables
  Nx = sol->grd->nx;
  Nz = sol->grd->nz;

  // Get local domain
  ierr = DMStagGetCorners(sol->dmHT, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  // Map global vectors to local domain
  ierr = DMGetLocalVector(sol->dmHT, &xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (sol->dmHT, x, INSERT_VALUES, xlocal); CHKERRQ(ierr);

  // Map global vectors to local domain
  ierr = DMCreateLocalVector(sol->dmHT, &flocal); CHKERRQ(ierr);

  // Map coefficient + stokes data to local domain
  ierr = DMGetLocalVector(sol->dmCoeff, &coefflocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (sol->dmCoeff, sol->coeff, INSERT_VALUES, coefflocal); CHKERRQ(ierr);

  ierr = DMGetLocalVector(sol->dmPV, &pvlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (sol->dmPV, sol->x, INSERT_VALUES, pvlocal); CHKERRQ(ierr);

  // Get residual array
  ierr = DMStagVecGetArrayDOF(sol->dmHT, flocal, &ff); CHKERRQ(ierr);

  // ---------------------------------------
  // Interior domain
  // ---------------------------------------
  // Loop over elements
  for (j = sz; j<sz+nz-1; ++j) {
    for (i = sx; i<sx+nx-1; ++i) {
      if ((j > 0) || (j < Nz-1) || (i > 0) || (i < Nx-1)) {
        // Get residual
        ierr = EnergyResidual(sol, xlocal, pvlocal, coefflocal, i, j, &fval); CHKERRQ(ierr);

        // Set residual in array
        ierr = DMStagGetLocationSlot(sol->dmHT, ELEMENT, 0, &idx); CHKERRQ(ierr);
        ff[j][i][idx] = fval;
      }
    }
  }

  // ---------------------------------------
  // Boundary conditions
  // ---------------------------------------
  ierr = BoundaryConditionsTemp(sol, xlocal, ff); CHKERRQ(ierr);

  // Restore arrays, local vectors
  ierr = DMStagVecRestoreArrayDOF(sol->dmHT,flocal,&ff); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(sol->dmHT,   &xlocal    ); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(sol->dmCoeff,&coefflocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(sol->dmPV,   &pvlocal   ); CHKERRQ(ierr);

  // ---------------------------------------
  // Return and clean up
  // ---------------------------------------
  // Map local to global
  ierr = DMLocalToGlobalBegin(sol->dmHT,flocal,INSERT_VALUES,f); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (sol->dmHT,flocal,INSERT_VALUES,f); CHKERRQ(ierr);

  ierr = VecDestroy(&flocal); CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}