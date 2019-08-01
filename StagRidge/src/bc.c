#include "stagridge.h"
#include "../tests/cornerflow.h"

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
  PetscScalar    xx, xp, zp, r;
  PetscScalar    sina, v[2], p;
  DMStagStencil  point;
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

      // 1) Constrain P - ELEMENT
      // Get coordinates and stencil values
      point.i = i; point.j = j; point.loc = ELEMENT; point.c = 0; // P
      ierr = GetCoordinatesStencil(dmCoord, coordLocal, 1, &point,&xp,&zp); CHKERRQ(ierr);
      ierr = DMStagVecGetValuesStencil(sol->dmPV,xlocal,1, &point,&xx    ); CHKERRQ(ierr);

      // Calculate positions relative to the lid 
      r = PetscPowScalar(xp*xp+zp*zp,0.5);

      if (zp>=-r*sina){ 
        // Set residual
        ierr = DMStagGetLocationSlot(sol->dmPV, point.loc, point.c, &idx); CHKERRQ(ierr);
        ff[j][i][idx] = xx - 0.0; // Lid
        //PetscPrintf(PETSC_COMM_SELF,"# P lid ff[%d][%d][%d]=%f\n",j,i,idx,ff[j][i][idx]);

      } else { 
        // LEFT, BOTTOM, RIGHT, TOP Boundaries (in case alpha=0)
        if ((i == 0) || (j == 0) || (i == Nx-1) || (j == Nz-1)){
          evaluate_CornerFlow_MOR(sol->usr->mor_A, sol->usr->mor_B, xp, zp,v,&p);

          // Set residual
          ierr = DMStagGetLocationSlot(sol->dmPV, point.loc, point.c, &idx); CHKERRQ(ierr);
          ff[j][i][idx] = xx - p;
          //PetscPrintf(PETSC_COMM_SELF,"# P border ff[%d][%d][%d]=%f p=%f x=%f z=%f\n",j,i,idx,ff[j][i][idx],p,xp,zp);
        }
      }

      // 2) Constrain Vx - LEFT
      // Get coordinates and stencil values
      point.i = i; point.j = j; point.loc = LEFT; point.c = 0; // Vx
      ierr = GetCoordinatesStencil(dmCoord, coordLocal, 1, &point,&xp,&zp); CHKERRQ(ierr);
      ierr = DMStagVecGetValuesStencil(sol->dmPV,xlocal,1, &point,&xx    ); CHKERRQ(ierr);

      // Calculate positions relative to the lid 
      r = PetscPowScalar(xp*xp+zp*zp,0.5);

      if (zp>=-r*sina){ 
        // Set residual
        ierr = DMStagGetLocationSlot(sol->dmPV, point.loc, point.c, &idx); CHKERRQ(ierr);
        ff[j][i][idx] = xx - sol->scal->u0; // Lid
        // PetscPrintf(PETSC_COMM_SELF,"# Vx lid ff[%d][%d][%d]=%f\n",j,i,idx,ff[j][i][idx]);

      } else { 
        // LEFT, BOTTOM, TOP Boundaries
        if ((i == 0) || (j == 0) || (j == Nz-1)){
          evaluate_CornerFlow_MOR(sol->usr->mor_A, sol->usr->mor_B, xp, zp,v,&p);

          // Set residual
          ierr = DMStagGetLocationSlot(sol->dmPV, point.loc, point.c, &idx); CHKERRQ(ierr);
          ff[j][i][idx] = xx - v[0];
          // PetscPrintf(PETSC_COMM_SELF,"# Vx border ff[%d][%d][%d]=%f\n",j,i,idx,ff[j][i][idx]);
        }
      }

      // 3) Constrain Vz - DOWN
      // Get coordinates and stencil values
      point.i = i; point.j = j; point.loc = DOWN; point.c = 0; // Vz
      ierr = GetCoordinatesStencil(dmCoord, coordLocal, 1, &point,&xp,&zp); CHKERRQ(ierr);
      ierr = DMStagVecGetValuesStencil(sol->dmPV,xlocal,1, &point,&xx    ); CHKERRQ(ierr);

      // Calculate positions relative to the lid 
      r = PetscPowScalar(xp*xp+zp*zp,0.5);

      if (zp>=-r*sina){ 
        // Set residual
        ierr = DMStagGetLocationSlot(sol->dmPV, point.loc, point.c, &idx); CHKERRQ(ierr);
        ff[j][i][idx] = xx - 0.0; // Lid
        // PetscPrintf(PETSC_COMM_SELF,"# Vz lid ff[%d][%d][%d]=%f\n",j,i,idx,ff[j][i][idx]);

      } else { 
        // LEFT, BOTTOM, RIGHT Boundaries
        if ((i == 0) || (j == 0) || (i == Nx-1)){
          evaluate_CornerFlow_MOR(sol->usr->mor_A, sol->usr->mor_B, xp, zp,v,&p);
          // Set residual
          ierr = DMStagGetLocationSlot(sol->dmPV, point.loc, point.c, &idx); CHKERRQ(ierr);
          ff[j][i][idx] = xx - v[1];
          // PetscPrintf(PETSC_COMM_SELF,"# Vz border ff[%d][%d][%d]=%f\n",j,i,idx,ff[j][i][idx]);
        }
      }

      // 4) Vx - RIGHT 
      if (i == Nx-1) {
        // Get coordinates and stencil values
        point.i = i; point.j = j; point.loc = RIGHT; point.c = 0; // Vx
        ierr = GetCoordinatesStencil(dmCoord, coordLocal, 1, &point,&xp,&zp); CHKERRQ(ierr);
        ierr = DMStagVecGetValuesStencil(sol->dmPV,xlocal,1, &point,&xx    ); CHKERRQ(ierr);

        // Calculate positions relative to the lid 
        r = PetscPowScalar(xp*xp+zp*zp,0.5);

        // Constrain Vx - RIGHT
        if (zp>=-r*sina){
          // Set residual
          ierr = DMStagGetLocationSlot(sol->dmPV, point.loc, point.c, &idx); CHKERRQ(ierr);
          ff[j][i][idx] = xx - sol->scal->u0; // lid
          // PetscPrintf(PETSC_COMM_SELF,"# Vx lid right ff[%d][%d][%d]=%f\n",j,i,idx,ff[j][i][idx]);

        } else {
          evaluate_CornerFlow_MOR(sol->usr->mor_A, sol->usr->mor_B, xp, zp,v,&p);
          // Set residual
          ierr = DMStagGetLocationSlot(sol->dmPV, point.loc, point.c, &idx); CHKERRQ(ierr);
          ff[j][i][idx] = xx - v[0];
          // PetscPrintf(PETSC_COMM_SELF,"# Vx border right ff[%d][%d][%d]=%f\n",j,i,idx,ff[j][i][idx]);
        }
      }

      // 5) Vz - UP
      if (j == Nz-1) {
        // Get coordinates and stencil values
        point.i = i; point.j = j; point.loc = UP; point.c = 0; // Vz
        ierr = GetCoordinatesStencil(dmCoord, coordLocal, 1, &point,&xp,&zp); CHKERRQ(ierr);
        ierr = DMStagVecGetValuesStencil(sol->dmPV,xlocal,1, &point,&xx    ); CHKERRQ(ierr);

        // Calculate positions relative to the lid 
        r = PetscPowScalar(xp*xp+zp*zp,0.5);

        // Constrain Vz
        if (zp>=-r*sina){
          // Set residual
          ierr = DMStagGetLocationSlot(sol->dmPV, point.loc, point.c, &idx); CHKERRQ(ierr);
          ff[j][i][idx] = xx - 0.0; // lid
          // PetscPrintf(PETSC_COMM_SELF,"# Vz lid up ff[%d][%d][%d]=%f\n",j,i,idx,ff[j][i][idx]);

        } else {
          evaluate_CornerFlow_MOR(sol->usr->mor_A, sol->usr->mor_B, xp, zp,v,&p);
          // Set residual
          ierr = DMStagGetLocationSlot(sol->dmPV, point.loc, point.c, &idx); CHKERRQ(ierr);
          ff[j][i][idx] = xx - v[1];
          // PetscPrintf(PETSC_COMM_SELF,"# Vz border up ff[%d][%d][%d]=%f\n",j,i,idx,ff[j][i][idx]);
        }
      }
    }
  }

  PetscFunctionReturn(0);
}
