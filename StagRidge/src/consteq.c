#include "stagridge.h"

// ---------------------------------------
// CalcEffViscosity
// ---------------------------------------
PetscErrorCode CalcEffViscosity(SolverCtx *sol, Vec xlocal, PetscInt i, PetscInt j, enum LocationType loctype, PetscScalar *etaeff)
{
  PetscScalar    epsII = 0.0;
  PetscScalar    inv_eta_diff, inv_eta_disl, eta;
  PetscInt       nEntries, Nx, Nz;
  PetscScalar    xx[12], dx, dz;
  DMStagStencil  col[12];
  PetscScalar    eps_xx, eps_zz, eps_xz, epsIIs2;
  PetscScalar    eps_xzi[4], eps_xxi[4], eps_zzi[4];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  
  /* Strain rate is defined in: 
      Corner - eps_xz 
      Center - eps_xx, eps_zz 
   */

  // Assign pointers and other variables
  Nx = sol->grd->nx;
  Nz = sol->grd->nz;
  dx = sol->grd->dx;
  dz = sol->grd->dz;

  /* Second invariant of strain rate in CENTER
      epsII^2 = 1/2*(eps_xx^2 + eps_zz^2 + 2*eps_xz_center^2);
      eps_xz_center - interpolated from eps_xz corners
  */ 
  if (loctype == CENTER) {
    // get all entries needed
    nEntries = 12;
    col[0].i  = i  ; col[0].j  = j+1; col[0].loc  = LEFT ; col[0].c   = 0; // Vx(i  ,j+1) LEFT
    col[1].i  = i  ; col[1].j  = j  ; col[1].loc  = LEFT ; col[1].c   = 0; // Vx(i  ,j  ) LEFT
    col[2].i  = i  ; col[2].j  = j-1; col[2].loc  = LEFT ; col[2].c   = 0; // Vx(i  ,j-1) LEFT
    col[3].i  = i  ; col[3].j  = j+1; col[3].loc  = RIGHT; col[3].c   = 0; // Vx(i  ,j+1) RIGHT
    col[4].i  = i  ; col[4].j  = j  ; col[4].loc  = RIGHT; col[4].c   = 0; // Vx(i  ,j  ) RIGHT
    col[5].i  = i  ; col[5].j  = j-1; col[5].loc  = RIGHT; col[5].c   = 0; // Vx(i  ,j-1) RIGHT
    col[6].i  = i-1; col[6].j  = j  ; col[6].loc  = UP;    col[6].c   = 0; // Vz(i-1,j  ) UP
    col[7].i  = i  ; col[7].j  = j  ; col[7].loc  = UP;    col[7].c   = 0; // Vz(i  ,j  ) UP
    col[8].i  = i+1; col[8].j  = j  ; col[8].loc  = UP;    col[8].c   = 0; // Vz(i+1,j  ) UP
    col[9].i  = i-1; col[9].j  = j  ; col[9].loc  = DOWN;  col[9].c   = 0; // Vz(i-1,j  ) DOWN
    col[10].i = i  ; col[10].j = j  ; col[10].loc = DOWN;  col[10].c  = 0; // Vz(i  ,j  ) DOWN 
    col[11].i = i+1; col[11].j = j  ; col[11].loc = DOWN;  col[11].c  = 0; // Vz(i+1,j  ) DOWN

    // Boundaries - LEFT
    if (i == 0){
      // zero flux
      if (sol->grd->bcleft == FREE_SLIP) {
        col[6] = col[7]; col[9] = col[10];
        
        // include corners
        if (j == 0){
          col[2] = col[1]; col[5] = col[4];
        } else if (j == Nz-1){
          col[0] = col[1]; col[3] = col[4];
        }
      }
    }

    // Boundaries - RIGHT
    if (i == Nx-1){
      // zero flux
      if (sol->grd->bcright == FREE_SLIP) {
        col[8] = col[7]; col[11] = col[10];

        // include corners
        if (j == 0){
          col[2] = col[1]; col[5] = col[4];
        } else if (j == Nz-1){
          col[0] = col[1]; col[3] = col[4];
        }
      }
    }

    // Boundaries - DOWN
    if ((j == 0) && (i > 0) && (i < Nx-1)){
      // zero flux
      if (sol->grd->bcdown == FREE_SLIP) {
        col[2] = col[1]; col[5] = col[4];
      }
    }

    // Boundaries - UP
    if ((j == Nz-1) && (i > 0) && (i < Nx-1)){
      // zero flux
      if (sol->grd->bcup == FREE_SLIP) {
        col[0] = col[1]; col[3] = col[4];
      }
    }
    
    ierr = DMStagVecGetValuesStencil(sol->dmPV, xlocal, nEntries, col, xx); CHKERRQ(ierr);

    // Calculate strain rates
    eps_xx = (xx[4]-xx[1] )/dx;
    eps_zz = (xx[7]-xx[10])/dz;

    eps_xzi[0] = 0.5*((xx[7] -xx[6] )/dx + (xx[0]-xx[1])/dz); 
    eps_xzi[1] = 0.5*((xx[8] -xx[7] )/dx + (xx[3]-xx[4])/dz);
    eps_xzi[2] = 0.5*((xx[10]-xx[9] )/dx + (xx[1]-xx[2])/dz);
    eps_xzi[3] = 0.5*((xx[11]-xx[10])/dx + (xx[4]-xx[5])/dz);

    // Interpolate eps_xz
    eps_xz = 0.25*(eps_xzi[0] + eps_xzi[1] + eps_xzi[2] + eps_xzi[3]);
  }

  /* Second invariant of strain rate in CORNER
      epsII^2 = 1/2*(eps_xx_corner^2 + eps_zz_corner^2 + 2*eps_xz^2);
      eps_xx_corner, eps_zz_corner - interpolated from centers
  */ 
  if (loctype == CORNER) {
    // get all entries needed
    nEntries = 12;
    col[0].i  = i-1; col[0].j  = j  ; col[0].loc  = LEFT ; col[0].c   = 0;
    col[1].i  = i  ; col[1].j  = j  ; col[1].loc  = LEFT ; col[1].c   = 0;
    col[2].i  = i  ; col[2].j  = j  ; col[2].loc  = RIGHT; col[2].c   = 0;
    col[3].i  = i-1; col[3].j  = j-1; col[3].loc  = LEFT ; col[3].c   = 0;
    col[4].i  = i  ; col[4].j  = j-1; col[4].loc  = LEFT ; col[4].c   = 0;
    col[5].i  = i  ; col[5].j  = j-1; col[5].loc  = RIGHT; col[5].c   = 0;
    col[6].i  = i-1; col[6].j  = j  ; col[6].loc  = UP;    col[6].c   = 0;
    col[7].i  = i-1; col[7].j  = j  ; col[7].loc  = DOWN;  col[7].c   = 0;
    col[8].i  = i-1; col[8].j  = j-1; col[8].loc  = DOWN;  col[8].c   = 0;
    col[9].i  = i  ; col[9].j  = j  ; col[9].loc  = UP;    col[9].c   = 0;
    col[10].i = i  ; col[10].j = j  ; col[10].loc = DOWN;  col[10].c  = 0;
    col[11].i = i  ; col[11].j = j-1; col[11].loc = DOWN;  col[11].c  = 0;

    // Boundaries - UP
    if (j == Nz){
      // zero flux
      if (sol->grd->bcup == FREE_SLIP) {
        col[0] = col[3]; col[1] = col[4]; col[2] = col[5]; col[6] = col[7]; col[9] = col[10];
      
      // include corners
        if (i == 0){
          col[0] = col[4]; col[6] = col[10]; col[3] = col[4]; col[7] = col[10]; col[8] = col[11];
        } else if (i == Nx){
          col[2] = col[4]; col[9] = col[7]; col[5] = col[4]; col[10] = col[7]; col[11] = col[8];
        }
      }
    }

    // Boundaries - DOWN
    if (j == 0){
      // zero flux
      if (sol->grd->bcdown == FREE_SLIP) {
        col[3] = col[0]; col[4] = col[1]; col[5] = col[2]; col[8] = col[7]; col[11] = col[10];

        // include corners
        if (i == 0){
          col[3] = col[1]; col[8] = col[10]; col[0] = col[1]; col[6] = col[9]; col[7] = col[11];
        } else if (i == Nx){
          col[5] = col[1]; col[11] = col[7]; col[2] = col[1]; col[9] = col[6]; col[10] = col[7];
        }
      }
    }

    // Boundaries - LEFT
    if ((i == 0) && (j > 0) && (j < Nz)) {
      // zero flux
      if (sol->grd->bcleft == FREE_SLIP) {
        col[0] = col[1]; col[3] = col[4]; col[6] = col[9]; col[7] = col[10]; col[8] = col[11];
      }
    }

    // Boundaries - RIGHT
    if ((i == 0) && (j > 0) && (j < Nz)) {
      // zero flux
      if (sol->grd->bcleft == FREE_SLIP) {
        col[2] = col[1]; col[5] = col[4]; col[9] = col[6]; col[10] = col[7]; col[11] = col[8];
      }
    }

    ierr = DMStagVecGetValuesStencil(sol->dmPV, xlocal, nEntries, col, xx); CHKERRQ(ierr);

    // Calculate strain rates
    eps_xz = 0.5*((xx[1] -xx[4] )/dx + (xx[10]-xx[7])/dz);

    eps_xxi[0] = (xx[1] -xx[0] )/dx; 
    eps_xxi[1] = (xx[2] -xx[1] )/dx; 
    eps_xxi[2] = (xx[4] -xx[3] )/dx; 
    eps_xxi[3] = (xx[5] -xx[4] )/dx;
    
    eps_zzi[0] = (xx[6] -xx[7] )/dz; 
    eps_zzi[1] = (xx[7] -xx[8] )/dz; 
    eps_zzi[2] = (xx[9] -xx[10])/dz; 
    eps_zzi[3] = (xx[10]-xx[11])/dz;

    // Interpolate eps_xx, eps_zz to corner
    eps_xx = 0.25*(eps_xxi[0] + eps_xxi[1] + eps_xxi[2] + eps_xxi[3]);
    eps_zz = 0.25*(eps_zzi[0] + eps_zzi[1] + eps_zzi[2] + eps_zzi[3]);
  }

  // EPSII squared
  epsIIs2 = 0.5*(eps_xx*eps_xx + eps_zz*eps_zz + 2.0*eps_xz*eps_xz);

  // Second invariant of strain rate
  epsII = PetscPowScalar(epsIIs2,0.5);

  // Rheology parameters
  sol->Pdiff = 1/(2.0*sol->usr->eta0);
  sol->Pdisl = 1/(2.0*sol->usr->eta0);

  // Calculate effective viscosity - should include component ratios
  inv_eta_diff = 2.0 * sol->Pdiff;

  if (sol->usr->ndisl==0) { // linear viscosity
    eta = 1/inv_eta_diff; 
  } else {
    inv_eta_disl = 2.0 * PetscPowScalar(sol->Pdisl,1/sol->usr->ndisl) * PetscPowScalar(epsII,1-1/sol->usr->ndisl);
    eta = 1/(inv_eta_diff+inv_eta_disl);
  }

  *etaeff = eta;

  PetscFunctionReturn(0);
}