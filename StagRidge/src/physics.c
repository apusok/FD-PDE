#include "stagridge.h"

// ---------------------------------------
// XMomentumResidual
// ---------------------------------------
PetscErrorCode XMomentumResidual(SolverCtx *sol, Vec xlocal, PetscInt i, PetscInt j, enum LocationType loctype, PetscScalar *ff)
{
  PetscScalar    dVxdz, dVzdx, dPdx, dVxdx, ffi;
  PetscInt       nEntries = 11;
  PetscScalar    xx[11], dx, dz;
  PetscScalar    etaLeft, etaRight, etaUp, etaDown;
  DMStagStencil  col[11];
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Get variables
  dx = sol->grd->dx;
  dz = sol->grd->dz;

  // Get stencil values
  col[0].i  = i  ; col[0].j  = j  ; col[0].loc  = LEFT;    col[0].c   = 0; // Vx(i  ,j  )
  col[1].i  = i  ; col[1].j  = j-1; col[1].loc  = LEFT;    col[1].c   = 0; // Vx(i  ,j-1)
  col[2].i  = i  ; col[2].j  = j+1; col[2].loc  = LEFT;    col[2].c   = 0; // Vx(i  ,j+1)
  col[3].i  = i-1; col[3].j  = j  ; col[3].loc  = LEFT;    col[3].c   = 0; // Vx(i-1,j  )
  col[4].i  = i  ; col[4].j  = j  ; col[4].loc  = RIGHT;   col[4].c   = 0; // Vx(i+1,j  )
  col[5].i  = i-1; col[5].j  = j  ; col[5].loc  = DOWN;    col[5].c   = 0; // Vz(i-1,j-1)
  col[6].i  = i  ; col[6].j  = j  ; col[6].loc  = DOWN;    col[6].c   = 0; // Vz(i  ,j-1)
  col[7].i  = i-1; col[7].j  = j  ; col[7].loc  = UP;      col[7].c   = 0; // Vz(i-1,j  )
  col[8].i  = i  ; col[8].j  = j  ; col[8].loc  = UP;      col[8].c   = 0; // Vz(i  ,j  )
  col[9].i  = i-1; col[9].j  = j  ; col[9].loc  = ELEMENT; col[9].c   = 0; // P (i-1,j  )
  col[10].i = i  ; col[10].j = j  ; col[10].loc = ELEMENT; col[10].c  = 0; // P (i  ,j  )

  // For boundaries copy the missing stencil entry with the main DOF
  if (loctype == BCUP ) {
    // Free slip
    if (sol->grd->bcup   == FREE_SLIP) col[2] = col[0];
  } else if (loctype == BCDOWN) {
    // Free slip
    if (sol->grd->bcdown == FREE_SLIP) col[1] = col[0];
  }

  // Get values
  ierr = DMStagVecGetValuesStencil(sol->dmPV, xlocal, nEntries, col, xx); CHKERRQ(ierr);

  // Calculate second invariant and effective viscosity
  ierr = CalcEffViscosity(sol, xlocal, i-1,j  ,CENTER, &etaLeft ); CHKERRQ(ierr);
  ierr = CalcEffViscosity(sol, xlocal, i  ,j  ,CENTER, &etaRight); CHKERRQ(ierr);
  ierr = CalcEffViscosity(sol, xlocal, i  ,j+1,CORNER, &etaUp   ); CHKERRQ(ierr);
  ierr = CalcEffViscosity(sol, xlocal, i  ,j  ,CORNER, &etaDown ); CHKERRQ(ierr);

  //etaLeft  = sol->usr->eta0;
  //etaRight = sol->usr->eta0;
  //etaUp    = sol->usr->eta0;
  //etaDown  = sol->usr->eta0;

  // Calculate residual
  dPdx  = (xx[10]-xx[9])/dx;
  dVxdx = etaRight*(xx[4]-xx[0])/dx - etaLeft*(xx[0]-xx[3])/dx;
  dVxdz = etaUp   *(xx[2]-xx[0])/dz - etaDown*(xx[0]-xx[1])/dz;
  dVzdx = etaUp   *(xx[8]-xx[7])/dx - etaDown*(xx[6]-xx[5])/dx;
  ffi   = -dPdx + 2.0*dVxdx/dx + dVxdz/dz + dVzdx/dz;

  *ff = ffi;
  PetscFunctionReturn(0);
}

// ---------------------------------------
// ZMomentumResidual
// ---------------------------------------
PetscErrorCode ZMomentumResidual(SolverCtx *sol, Vec xlocal, Vec coefflocal, PetscInt i, PetscInt j, enum LocationType loctype, PetscScalar *ff)
{
  PetscScalar    dVxdz, dVzdx, dPdz, dVzdz, rhog, ffi;
  PetscInt       nEntries = 11;
  PetscScalar    xx[11], dx, dz;
  PetscScalar    etaLeft, etaRight, etaUp, etaDown, rho[2];
  DMStagStencil  col[11];
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Get variables
  dx = sol->grd->dx;
  dz = sol->grd->dz;

  // Get density values
  col[0].i = i; col[0].j = j  ; col[0].loc = ELEMENT; col[0].c = 0;
  col[1].i = i; col[1].j = j-1; col[1].loc = ELEMENT; col[1].c = 0;

  ierr = DMStagVecGetValuesStencil(sol->dmCoeff, coefflocal, 2, col, rho); CHKERRQ(ierr);
  rhog = -sol->usr->g * 0.5 * (rho[0] + rho[1]);

  // Get stencil values
  col[0].i  = i  ; col[0].j  = j  ; col[0].loc  = DOWN;    col[0].c   = 0; // Vz(i  ,j  )
  col[1].i  = i  ; col[1].j  = j  ; col[1].loc  = UP;      col[1].c   = 0; // Vz(i  ,j+1)
  col[2].i  = i  ; col[2].j  = j-1; col[2].loc  = DOWN;    col[2].c   = 0; // Vz(i  ,j-1)
  col[3].i  = i-1; col[3].j  = j  ; col[3].loc  = DOWN;    col[3].c   = 0; // Vz(i-1,j  )
  col[4].i  = i+1; col[4].j  = j  ; col[4].loc  = DOWN;    col[4].c   = 0; // Vz(i+1,j  )
  col[5].i  = i  ; col[5].j  = j  ; col[5].loc  = LEFT;    col[5].c   = 0; // Vx(i-1,j  )
  col[6].i  = i  ; col[6].j  = j  ; col[6].loc  = RIGHT;   col[6].c   = 0; // Vx(i  ,j  )
  col[7].i  = i  ; col[7].j  = j-1; col[7].loc  = LEFT;    col[7].c   = 0; // Vx(i-1,j-1)
  col[8].i  = i  ; col[8].j  = j-1; col[8].loc  = RIGHT;   col[8].c   = 0; // Vx(i  ,j-1)
  col[9].i  = i  ; col[9].j  = j  ; col[9].loc  = ELEMENT; col[9].c   = 0; // P (i  ,j  )
  col[10].i = i  ; col[10].j = j-1; col[10].loc = ELEMENT; col[10].c  = 0; // P (i  ,j-1)

  // For boundaries copy the missing stencil entry with the main DOF
  if (loctype == BCLEFT ) {
    // Free slip
    if (sol->grd->bcleft  == FREE_SLIP) col[3] = col[0];
  } else if (loctype == BCRIGHT) {
    // Free slip
    if (sol->grd->bcright == FREE_SLIP) col[4] = col[0]; 
  }

  // Get values
  ierr = DMStagVecGetValuesStencil(sol->dmPV, xlocal, nEntries, col, xx); CHKERRQ(ierr);

  // Calculate second invariant and effective viscosity
  ierr = CalcEffViscosity(sol, xlocal, i  ,j  ,CORNER, &etaLeft ); CHKERRQ(ierr);
  ierr = CalcEffViscosity(sol, xlocal, i+1,j  ,CORNER, &etaRight); CHKERRQ(ierr);
  ierr = CalcEffViscosity(sol, xlocal, i  ,j  ,CENTER, &etaUp   ); CHKERRQ(ierr);
  ierr = CalcEffViscosity(sol, xlocal, i  ,j-1,CENTER, &etaDown ); CHKERRQ(ierr);

  //etaLeft  = sol->usr->eta0;
  //etaRight = sol->usr->eta0;
  //etaUp    = sol->usr->eta0;
  //etaDown  = sol->usr->eta0;

  // Calculate residual
  dPdz  = (xx[9]-xx[10])/dz;
  dVzdz = etaUp   *(xx[1]-xx[0])/dz - etaDown *(xx[0]-xx[2])/dz;
  dVzdx = etaRight*(xx[4]-xx[0])/dx - etaLeft *(xx[0]-xx[3])/dx;
  dVxdz = etaLeft *(xx[5]-xx[7])/dz - etaRight*(xx[6]-xx[8])/dz;
  ffi   = -dPdz + 2.0*dVzdz/dz + dVzdx/dx + dVxdz/dx - rhog;

  *ff = ffi;

  PetscFunctionReturn(0);
}

// ---------------------------------------
// XMomentumStencil
// ---------------------------------------
PetscErrorCode XMomentumStencil(PetscInt i,PetscInt j, PetscInt Nx, PetscInt Nz, DMStagStencil *col)
{
  PetscFunctionBegin;

  // Original stencil entries
  col[0].i  = i  ; col[0].j  = j  ; col[0].loc  = LEFT;    col[0].c   = 0;
  col[1].i  = i  ; col[1].j  = j-1; col[1].loc  = LEFT;    col[1].c   = 0;
  col[2].i  = i  ; col[2].j  = j+1; col[2].loc  = LEFT;    col[2].c   = 0;
  col[3].i  = i-1; col[3].j  = j  ; col[3].loc  = LEFT;    col[3].c   = 0;
  col[4].i  = i  ; col[4].j  = j  ; col[4].loc  = RIGHT;   col[4].c   = 0;
  col[5].i  = i-1; col[5].j  = j  ; col[5].loc  = DOWN;    col[5].c   = 0;
  col[6].i  = i  ; col[6].j  = j  ; col[6].loc  = DOWN;    col[6].c   = 0;
  col[7].i  = i-1; col[7].j  = j  ; col[7].loc  = UP;      col[7].c   = 0;
  col[8].i  = i  ; col[8].j  = j  ; col[8].loc  = UP;      col[8].c   = 0;
  col[9].i  = i-1; col[9].j  = j  ; col[9].loc  = ELEMENT; col[9].c   = 0;
  col[10].i = i  ; col[10].j = j  ; col[10].loc = ELEMENT; col[10].c  = 0;

  // Nonlinear stencil extension (strain rates)
  col[11].i = i-1; col[11].j = j+1; col[11].loc = UP;      col[11].c  = 0;
  col[12].i = i  ; col[12].j = j+1; col[12].loc = UP;      col[12].c  = 0;
  col[13].i = i-2; col[13].j = j  ; col[13].loc = UP;      col[13].c  = 0;
  col[14].i = i-2; col[14].j = j  ; col[14].loc = DOWN;    col[14].c  = 0;
  col[15].i = i+1; col[15].j = j  ; col[15].loc = UP;      col[15].c  = 0;
  col[16].i = i+1; col[16].j = j  ; col[16].loc = DOWN;    col[16].c  = 0;
  col[17].i = i-1; col[17].j = j-1; col[17].loc = DOWN;    col[17].c  = 0;
  col[18].i = i  ; col[18].j = j-1; col[18].loc = DOWN;    col[18].c  = 0;

  col[19].i = i-1; col[19].j = j+1; col[19].loc = LEFT;    col[19].c  = 0;
  col[20].i = i  ; col[20].j = j+1; col[20].loc = RIGHT;   col[20].c  = 0;
  col[21].i = i-1; col[21].j = j-1; col[21].loc = LEFT;    col[21].c  = 0;
  col[22].i = i  ; col[22].j = j-1; col[22].loc = RIGHT;   col[22].c  = 0;

  // Boundaries - correct for missing values
  if (i == 1) {
    col[13] = col[7]; col[14] = col[5];
  } else if (i == Nx-1) {
    col[15] = col[8]; col[16] = col[6];
  }

  if (j == 0) {
    col[1] = col[0]; col[17] = col[5]; col[18] = col[6]; col[21] = col[3]; col[22] = col[4];
  } else if (j == Nz-1) {
    col[2] = col[0]; col[11] = col[7]; col[12] = col[8]; col[19] = col[3]; col[20] = col[4];
  }

  PetscFunctionReturn(0);
}
// ---------------------------------------
// ZMomentumStencil
// ---------------------------------------
PetscErrorCode ZMomentumStencil(PetscInt i,PetscInt j, PetscInt Nx, PetscInt Nz, DMStagStencil *col)
{
  PetscFunctionBegin;

  // Original stencil entries
  col[0].i = i  ; col[0].j  = j  ; col[0].loc  = DOWN;     col[0].c  = 0;
  col[1].i = i  ; col[1].j  = j-1; col[1].loc  = DOWN;     col[1].c  = 0;
  col[2].i = i  ; col[2].j  = j+1; col[2].loc  = DOWN;     col[2].c  = 0;
  col[3].i = i-1; col[3].j  = j  ; col[3].loc  = DOWN;     col[3].c  = 0;
  col[4].i = i+1; col[4].j  = j  ; col[4].loc  = DOWN;     col[4].c  = 0;
  col[5].i = i  ; col[5].j  = j-1; col[5].loc  = LEFT;     col[5].c  = 0;
  col[6].i = i  ; col[6].j  = j-1; col[6].loc  = RIGHT;    col[6].c  = 0;
  col[7].i = i  ; col[7].j  = j  ; col[7].loc  = LEFT;     col[7].c  = 0;
  col[8].i = i  ; col[8].j  = j  ; col[8].loc  = RIGHT;    col[8].c  = 0;
  col[9].i = i  ; col[9].j  = j-1; col[9].loc  = ELEMENT;  col[9].c  = 0;
  col[10].i = i ; col[10].j = j  ; col[10].loc = ELEMENT;  col[10].c = 0;

  // Nonlinear stencil extension (strain rates)
  col[11].i = i  ; col[11].j = j+1; col[11].loc = LEFT;    col[11].c  = 0;
  col[12].i = i  ; col[12].j = j+1; col[12].loc = RIGHT;   col[12].c  = 0;
  col[13].i = i-1; col[13].j = j  ; col[13].loc = LEFT;    col[13].c  = 0;
  col[14].i = i+1; col[14].j = j  ; col[14].loc = RIGHT;   col[14].c  = 0;
  col[15].i = i-1; col[15].j = j-1; col[15].loc = LEFT;    col[15].c  = 0;
  col[16].i = i+1; col[16].j = j-1; col[16].loc = RIGHT;   col[16].c  = 0;
  col[17].i = i  ; col[17].j = j-2; col[17].loc = LEFT;    col[17].c  = 0;
  col[18].i = i  ; col[18].j = j-2; col[18].loc = RIGHT;   col[18].c  = 0;

  col[19].i = i-1; col[19].j = j  ; col[19].loc = UP;      col[19].c  = 0;
  col[20].i = i+1; col[20].j = j  ; col[20].loc = UP;      col[20].c  = 0;
  col[21].i = i-1; col[21].j = j-1; col[21].loc = DOWN;    col[21].c  = 0;
  col[22].i = i+1; col[22].j = j-1; col[22].loc = DOWN;    col[22].c  = 0;

  // Boundaries - correct for missing values
  if (j == 1) {
    col[17] = col[5]; col[18] = col[6];
  } else if (j == Nz-1) {
    col[11] = col[7]; col[12] = col[8];
  }

  if (i == 0) {
    col[3] = col[0]; col[13] = col[7]; col[15] = col[5]; col[19] = col[2]; col[21] = col[1];
  } else if (i == Nx-1) {
    col[4] = col[0]; col[14] = col[8]; col[16] = col[6]; col[20] = col[2]; col[22] = col[1];
  }

  PetscFunctionReturn(0);
}