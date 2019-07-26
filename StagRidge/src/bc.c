#include "stagridge.h"

// ---------------------------------------
// Boundary Conditions - General
// ---------------------------------------
PetscErrorCode BoundaryConditions_General(SolverCtx *sol, Vec xlocal, Vec coefflocal, PetscScalar ***ff)
{
  PetscInt       i, j, idx;
  PetscInt       Nx, Nz, nx, nz, sx, sz;
  PetscScalar    fval, xx, fzero = 0.0;
  DMStagStencil  point;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  // Assign pointers and other variables
  Nx = sol->grd->nx;
  Nz = sol->grd->nz;

  // Get local domain
  ierr = DMStagGetCorners(sol->dmPV, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  // LEFT
  i = sx;
  if (i == 0) {
    // Free slip
    if (sol->grd->bcleft == FREE_SLIP){
      for (j = sz; j<sz+nz; ++j) {
        // Vx - Dirichlet
        // Get stencil values
        point.i = i  ; point.j  = j  ; point.loc  = LEFT;    point.c   = 0;
        ierr = DMStagVecGetValuesStencil(sol->dmPV, xlocal, 1, &point, &xx); CHKERRQ(ierr);

        // Calculate residual
        fval = xx - fzero;
        
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
        point.i = i ; point.j  = j  ; point.loc  = RIGHT;    point.c   = 0;
        ierr = DMStagVecGetValuesStencil(sol->dmPV, xlocal, 1, &point, &xx); CHKERRQ(ierr);

        // Calculate residual
        fval = xx - fzero;
        
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
        point.i = i  ; point.j  = j ; point.loc  = DOWN;    point.c   = 0;
        ierr = DMStagVecGetValuesStencil(sol->dmPV, xlocal, 1, &point, &xx); CHKERRQ(ierr);

        // Calculate residual
        fval = xx - fzero;
        
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
        point.i = i; point.j = j; point.loc = UP; point.c = 0;
        ierr = DMStagVecGetValuesStencil(sol->dmPV, xlocal, 1, &point, &xx); CHKERRQ(ierr);

        // Calculate residual
        fval = xx - fzero;
        
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

  PetscFunctionReturn(0);
}

// ---------------------------------------
// Boundary Conditions - MORAnalytic: constrain values in the lithosphere (P = 0, Vx = u0, Vz = 0) and boundaries
// ---------------------------------------
PetscErrorCode BoundaryConditions_MORAnalytic(SolverCtx *sol, Vec xlocal, PetscScalar ***ff)
{
  PetscInt       i, j, idx;
  PetscInt       Nx, Nz, nx, nz, sx, sz;
  PetscScalar    xx[3], xp[3], zp[3], r[3];
  PetscScalar    sina, fval;
  DMStagStencil  point[3];
  Vec            coordLocal;
  DM             dmCoord;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  // Assign pointers and other variables
  Nx = sol->grd->nx;
  Nz = sol->grd->nz;
  sina = sol->usr->mor_sina;

  // Get local domain
  ierr = DMStagGetCorners(sol->dmPV, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  // Get coordinates of dmPV
  ierr = DMGetCoordinatesLocal(sol->dmPV, &coordLocal); CHKERRQ(ierr);
  ierr = DMGetCoordinateDM    (sol->dmPV, &dmCoord   ); CHKERRQ(ierr);

  // Loop over elements and assign constraints
  for (j = sz; j<sz+nz; ++j) {
    for (i = sx; i<sx+nx; ++i) {
      // Get stencil points - INTERIOR (lid) - LEFT, DOWN, ELEMENT
      point[0].i = i; point[0].j = j; point[0].loc = ELEMENT; point[0].c = 0; // P
      point[1].i = i; point[1].j = j; point[1].loc = LEFT;    point[1].c = 0; // Vx
      point[2].i = i; point[2].j = j; point[2].loc = DOWN;    point[2].c = 0; // Vz

      // Get coordinates and stencil values
      ierr = GetCoordinatesStencil(dmCoord, coordLocal, 3, point, xp, zp); CHKERRQ(ierr);
      ierr = DMStagVecGetValuesStencil(sol->dmPV, xlocal, 3, point, xx);   CHKERRQ(ierr);

      // Calculate positions relative to the lid 
      r[0] = PetscPowScalar(xp[0]*xp[0]+zp[0]*zp[0],0.5);
      r[1] = PetscPowScalar(xp[1]*xp[1]+zp[1]*zp[1],0.5);
      r[2] = PetscPowScalar(xp[2]*xp[2]+zp[2]*zp[2],0.5);

      // 1) Constrain P - ELEMENT
      if (PetscAbsScalar(zp[0])<=r[0]*sina){ 
        fval = 0.0; // Lid
      } else { 
        // LEFT, BOTTOM, RIGHT, TOP Boundaries (in case alpha=0)
        if ((i == 0) || (j == 0) || (i == Nx-1) || (j == Nz-1)){
          ierr = MORAnalytic_P(sol, xp[0], zp[0], &fval); CHKERRQ(ierr);
        }
      }
      // Set residual
      ierr = DMStagGetLocationSlot(sol->dmPV, ELEMENT, 0, &idx); CHKERRQ(ierr);
      ff[j][i][idx] = xx[0] - fval;

      // 2) Constrain Vx - LEFT
      if (PetscAbsScalar(zp[1])<=r[1]*sina){ 
        fval = sol->scal->u0; // Lid
      } else { 
        // LEFT, BOTTOM, TOP Boundaries
        if ((i == 0) || (j == 0) || (j == Nz-1)){
          ierr = MORAnalytic_Vx(sol, xp[1], zp[1], &fval); CHKERRQ(ierr);
        }
      }

      // Set residual
      ierr = DMStagGetLocationSlot(sol->dmPV, LEFT, 0, &idx); CHKERRQ(ierr);
      ff[j][i][idx] = xx[1] - fval;

      // 3) Constrain Vz - DOWN
      if (PetscAbsScalar(zp[2])<=r[2]*sina){ 
        fval = 0.0; // Lid
      } else { 
        // LEFT, BOTTOM, RIGHT Boundaries
        if ((i == 0) || (j == 0) || (i == Nx-1)){
          ierr = MORAnalytic_Vz(sol, xp[2], zp[2], &fval); CHKERRQ(ierr);
        }
      }

      // Set residual
      ierr = DMStagGetLocationSlot(sol->dmPV, DOWN, 0, &idx); CHKERRQ(ierr);
      ff[j][i][idx] = xx[2] - fval;

      // Vx - RIGHT 
      if (i == Nx-1) {
        // Get stencil point
        point[0].i = i; point[0].j = j; point[0].loc = RIGHT; point[0].c = 0;

        // Get coordinates and stencil values
        ierr = GetCoordinatesStencil(dmCoord, coordLocal, 1, point, xp, zp); CHKERRQ(ierr);
        ierr = DMStagVecGetValuesStencil(sol->dmPV, xlocal, 1, point, xx);   CHKERRQ(ierr);

        // Calculate positions relative to the lid 
        r[0] = PetscPowScalar(xp[0]*xp[0]+zp[0]*zp[0],0.5);

        // Constrain Vx - RIGHT
        if (PetscAbsScalar(zp[0])<=r[0]*sina){
          fval = sol->scal->u0;
        } else {
          ierr = MORAnalytic_Vx(sol, xp[0], zp[0], &fval); CHKERRQ(ierr);
        }
      }

      // Set residual
      ierr = DMStagGetLocationSlot(sol->dmPV, RIGHT, 0, &idx); CHKERRQ(ierr);
      ff[j][i][idx] = xx[0] - fval;

      // Vz - UP
      if (j == Nz-1) {
        point[0].i = i; point[0].j = j; point[0].loc = UP; point[0].c = 0;

        // Get coordinates and stencil values
        ierr = GetCoordinatesStencil(dmCoord, coordLocal, 1, point, xp, zp); CHKERRQ(ierr);
        ierr = DMStagVecGetValuesStencil(sol->dmPV, xlocal, 1, point, xx);   CHKERRQ(ierr);

        // Calculate positions relative to the lid 
        r[0] = PetscPowScalar(xp[0]*xp[0]+zp[0]*zp[0],0.5);

        // Constrain Vz
        if (PetscAbsScalar(zp[0])<=r[0]*sina){
          fval = 0.0;
        } else {
          ierr = MORAnalytic_Vz(sol, xp[0], zp[0], &fval); CHKERRQ(ierr);
        }
      }
      // Set residual
      ierr = DMStagGetLocationSlot(sol->dmPV, UP, 0, &idx); CHKERRQ(ierr);
      ff[j][i][idx] = xx[0] - fval;
    }
  }

  PetscFunctionReturn(0);
}
