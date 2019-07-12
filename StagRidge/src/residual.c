#include "stagridge.h"

// ---------------------------------------
// FormFunctionPV
// ---------------------------------------
PetscErrorCode FormFunctionPV(SNES snes, Vec x, Vec f, void *ctx)
{
  SolverCtx      *sol = (SolverCtx*) ctx;
  PetscInt       i, j, Nx, Nz, sx, sz, nx, nz;
  Vec            xlocal, coefflocal,flocal;
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

  // Loop over elements - Interior domain
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
  // LEFT
  i = sx;
  if (i == 0) {
    // Free slip
    if (sol->grd->bcleft == FREE_SLIP){
      for (j = sz; j<sz+nz; ++j) {
        // Vx - Dirichlet
        // Get stencil values
        col[0].i = i  ; col[0].j  = j  ; col[0].loc  = LEFT;    col[0].c   = 0;
        ierr = DMStagVecGetValuesStencil(sol->dmPV, xlocal, 1, col, xx); CHKERRQ(ierr);

        // Calculate residual
        fval = xx[0] - sol->grd->Vleft;
        
        // Set residual in array
        ierr = DMStagGetLocationSlot(sol->dmPV, LEFT, 0, &idx); CHKERRQ(ierr);
        ff[j][i][idx] = fval;
        
        // Vz - zero horizontal flux (dVz/dx = 0)
        if (j > 0) {
          // Get residual
          ierr = ZMomentumResidual(sol, xlocal, coefflocal, i, j, BCLEFT, &fval); CHKERRQ(ierr);

          // Set residual in array
          ierr = DMStagGetLocationSlot(sol->dmPV, DOWN, 0, &idx); CHKERRQ(ierr);
          ff[j][i][idx] = fval;
        }
      }
    } // add other BC
  }

  // RIGHT
  i = sx+nx-1;
  if (i == Nx-1) {
    // Free slip
    if (sol->grd->bcright == FREE_SLIP){
      for (j = sz; j<sz+nz; ++j) {
        // Vx - Dirichlet
        // Get stencil values
        col[0].i = i ; col[0].j  = j  ; col[0].loc  = RIGHT;    col[0].c   = 0;
        ierr = DMStagVecGetValuesStencil(sol->dmPV, xlocal, 1, col, xx); CHKERRQ(ierr);

        // Calculate residual
        fval = xx[0] - sol->grd->Vright;
        
        // Set residual in array
        ierr = DMStagGetLocationSlot(sol->dmPV, RIGHT, 0, &idx); CHKERRQ(ierr);
        ff[j][i][idx] = fval;

        // Vz - zero horizontal flux (dVz/dx = 0)
        if (j > 0) {
          // Get residual
          ierr = ZMomentumResidual(sol, xlocal, coefflocal, i, j, BCRIGHT, &fval); CHKERRQ(ierr);

          // Set residual in array
          ierr = DMStagGetLocationSlot(sol->dmPV, DOWN, 0, &idx); CHKERRQ(ierr);
          ff[j][i][idx] = fval;
        }
      }
    } // add other BC
  }

// DOWN
  j = sz;
  if (j == 0) {
    // Free slip
    if (sol->grd->bcdown == FREE_SLIP){
      for (i = sx; i<sx+nx; ++i) {
        // Vz - Dirichlet
        // Get stencil values
        col[0].i = i  ; col[0].j  = j ; col[0].loc  = DOWN;    col[0].c   = 0;
        ierr = DMStagVecGetValuesStencil(sol->dmPV, xlocal, 1, col, xx); CHKERRQ(ierr);

        // Calculate residual
        fval = xx[0] - sol->grd->Vdown;
        
        // Set residual in array
        ierr = DMStagGetLocationSlot(sol->dmPV, DOWN, 0, &idx); CHKERRQ(ierr);
        ff[j][i][idx] = fval;

        // Vx - zero vertical flux (dVx/dz = 0)
        if (i > 0) {
          // Get residual
          ierr = XMomentumResidual(sol, xlocal, i, j, BCDOWN, &fval); CHKERRQ(ierr);

          // Set residual in array
          ierr = DMStagGetLocationSlot(sol->dmPV, LEFT, 0, &idx); CHKERRQ(ierr);
          ff[j][i][idx] = fval;
        }
      }
    } // add other BC
  }

  // UP
  j = sz+nz-1;
  if (j == Nz-1) {
    // Free slip
    if (sol->grd->bcup == FREE_SLIP){
      for (i = sx; i<sx+nx; ++i) {
        // Vz - Dirichlet
        // Get stencil values
        col[0].i = i; col[0].j = j; col[0].loc = UP; col[0].c = 0;
        ierr = DMStagVecGetValuesStencil(sol->dmPV, xlocal, 1, col, xx); CHKERRQ(ierr);

        // Calculate residual
        fval = xx[0] - sol->grd->Vup;
        
        // Set residual in array
        ierr = DMStagGetLocationSlot(sol->dmPV, UP, 0, &idx); CHKERRQ(ierr);
        ff[j][i][idx] = fval;

        // Vx - zero vertical flux (dVx/dz = 0)
        if (i > 0) {
          // Get residual
          ierr = XMomentumResidual(sol, xlocal, i, j, BCUP, &fval); CHKERRQ(ierr);

          // Set residual in array
          ierr = DMStagGetLocationSlot(sol->dmPV, LEFT, 0, &idx); CHKERRQ(ierr);
          ff[j][i][idx] = fval;
        }
      }
    } // add other BC
  }

  // Map local to global
  ierr = DMLocalToGlobalBegin(sol->dmPV,flocal,INSERT_VALUES,f); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (sol->dmPV,flocal,INSERT_VALUES,f); CHKERRQ(ierr);

  // Restore arrays, local vectors
  ierr = DMStagVecRestoreArrayDOF(sol->dmPV,flocal,&ff); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(sol->dmCoeff,&coefflocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(sol->dmPV,   &xlocal    ); CHKERRQ(ierr);

  ierr = VecDestroy(&flocal); CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}
