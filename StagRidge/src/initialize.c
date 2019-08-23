#include "stagridge.h"

// ---------------------------------------
// InitializeModel
// ---------------------------------------
PetscErrorCode InitializeModel(SolverCtx *sol)
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  
  // Different model setups
  if ((sol->grd->mtype == SOLCX) || (sol->grd->mtype == SOLCX_EFF)){
    ierr = InitializeModel_SolCx(sol); CHKERRQ(ierr);
  } else if (sol->grd->mtype == MOR  ){
    ierr = InitializeModel_MOR(sol); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

// ---------------------------------------
// InitializeModel_SolCx - density defined on Vz-edges
// ---------------------------------------
PetscErrorCode InitializeModel_SolCx(SolverCtx *sol)
{
  PetscInt       i, j, ii, sx, sz, nx, nz;
  Vec            coordLocal;
  DM             dmCoord;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  
  // Access vector with density
  ierr = DMCreateGlobalVector(sol->dmCoeff, &sol->coeff); CHKERRQ(ierr);
  
  // Get domain corners
  ierr = DMStagGetCorners(sol->dmCoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  
  // Get coordinates
  ierr = DMGetCoordinatesLocal(sol->dmCoeff, &coordLocal); CHKERRQ(ierr);
  ierr = DMGetCoordinateDM    (sol->dmCoeff, &dmCoord   ); CHKERRQ(ierr);
  
  // Loop over local domain
  for (j = sz; j < sz+nz; ++j) {
    for (i = sx; i <sx+nx; ++i) {

      DMStagStencil point[5];
      PetscScalar   xp[5], zp[5], rho[5];
        
      // Set stencil values
      point[0].i = i; point[0].j = j; point[0].loc = ELEMENT; point[0].c = 0;
      point[1].i = i; point[1].j = j; point[1].loc = LEFT;    point[1].c = 0;
      point[2].i = i; point[2].j = j; point[2].loc = RIGHT;   point[2].c = 0;
      point[3].i = i; point[3].j = j; point[3].loc = DOWN;    point[3].c = 0;
      point[4].i = i; point[4].j = j; point[4].loc = UP;      point[4].c = 0;

      // Get coordinates
      ierr = GetCoordinatesStencil(dmCoord, coordLocal, 5, point, xp, zp); CHKERRQ(ierr);

      // Set densities
      for (ii = 0; ii < 5; ++ii) {
        rho[ii]  = PetscSinScalar(PETSC_PI*zp[ii]) * PetscCosScalar(PETSC_PI*xp[ii]); //sin(pi*z)*cos(pi*x); 
      }

      // Put values back
      ierr = DMStagVecSetValuesStencil(sol->dmCoeff,sol->coeff,5,point,rho,INSERT_VALUES); CHKERRQ(ierr);
    }
  }
  
  // Vector Assembly and Restore local vector
  ierr = VecAssemblyBegin(sol->coeff); CHKERRQ(ierr);
  ierr = VecAssemblyEnd  (sol->coeff); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// InitializeModel_MOR
// ---------------------------------------
PetscErrorCode InitializeModel_MOR(SolverCtx *sol)
{
  PetscInt       i, j, sx, sz, nx, nz;
  Vec            coordLocal;
  DM             dmCoord;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  
  // Access vector with density
  ierr = DMCreateGlobalVector(sol->dmCoeff, &sol->coeff); CHKERRQ(ierr);
  
  // Get domain corners
  ierr = DMStagGetCorners(sol->dmCoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  
  // Get coordinates
  ierr = DMGetCoordinatesLocal(sol->dmCoeff, &coordLocal); CHKERRQ(ierr);
  ierr = DMGetCoordinateDM    (sol->dmCoeff, &dmCoord   ); CHKERRQ(ierr);
  
  // Loop over local domain
  for (j = sz; j < sz+nz; ++j) {
    for (i = sx; i <sx+nx; ++i) {

      DMStagStencil point[5];
      PetscScalar   rho[5];
      
      // Constant density (zero)
      rho[0] = sol->scal->rho0; rho[1] = rho[0]; rho[2] = rho[0]; rho[3] = rho[0]; rho[4] = rho[0];

      // Set density value - ELEMENT
      point[0].i = i; point[0].j = j; point[0].loc = ELEMENT; point[0].c = 0;
      point[1] = point[0]; point[1].loc = DOWN;
      point[2] = point[0]; point[2].loc = UP;
      point[3] = point[0]; point[3].loc = LEFT;
      point[4] = point[0]; point[4].loc = RIGHT;
      ierr = DMStagVecSetValuesStencil(sol->dmCoeff,sol->coeff,5,point,rho,INSERT_VALUES); CHKERRQ(ierr);
    }
  }
  
  // Vector Assembly and Restore local vector
  ierr = VecAssemblyBegin(sol->coeff); CHKERRQ(ierr);
  ierr = VecAssemblyEnd  (sol->coeff); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// InitializeModelTemp
// ---------------------------------------
PetscErrorCode InitializeModelTemp(SolverCtx *sol)
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  
  // Velocity vector sol->x
  ierr = InitializeModelTemp_Vel(sol); CHKERRQ(ierr);

  // Density
  ierr = InitializeModelTemp_Rho(sol); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// InitializeModelTemp_Rho
// ---------------------------------------
PetscErrorCode InitializeModelTemp_Rho(SolverCtx *sol)
{
  PetscInt       i, j, sx, sz, nx, nz;
  Vec            coordLocal;
  DM             dmCoord;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  
  // Access vector with density
  ierr = DMCreateGlobalVector(sol->dmCoeff, &sol->coeff); CHKERRQ(ierr);
  
  // Get domain corners
  ierr = DMStagGetCorners(sol->dmCoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  
  // Get coordinates
  ierr = DMGetCoordinatesLocal(sol->dmCoeff, &coordLocal); CHKERRQ(ierr);
  ierr = DMGetCoordinateDM    (sol->dmCoeff, &dmCoord   ); CHKERRQ(ierr);
  
  // Loop over local domain
  for (j = sz; j < sz+nz; ++j) {
    for (i = sx; i <sx+nx; ++i) {

      DMStagStencil point[5];
      PetscScalar   rho[5];
      
      // Constant density (zero)
      rho[0] = sol->scal->rho0; rho[1] = rho[0]; rho[2] = rho[0]; rho[3] = rho[0]; rho[4] = rho[0];

      // Set density value - ELEMENT
      point[0].i = i; point[0].j = j; point[0].loc = ELEMENT; point[0].c = 0;
      point[1] = point[0]; point[1].loc = DOWN;
      point[2] = point[0]; point[2].loc = UP;
      point[3] = point[0]; point[3].loc = LEFT;
      point[4] = point[0]; point[4].loc = RIGHT;
      ierr = DMStagVecSetValuesStencil(sol->dmCoeff,sol->coeff,5,point,rho,INSERT_VALUES); CHKERRQ(ierr);
    }
  }
  
  // Vector Assembly and Restore local vector
  ierr = VecAssemblyBegin(sol->coeff); CHKERRQ(ierr);
  ierr = VecAssemblyEnd  (sol->coeff); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// InitializeModelTemp_Vel - initialize Stokes solution (velocity and pressure)
// ---------------------------------------
PetscErrorCode InitializeModelTemp_Vel(SolverCtx *sol)
{
  PetscInt       i, j, sx, sz, nx, nz;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  
  // Get domain corners
  ierr = DMStagGetCorners(sol->dmPV, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
    
  // Loop over local domain
  for (j = sz; j < sz+nz; ++j) {
    for (i = sx; i <sx+nx; ++i) {

      DMStagStencil point[5];
      PetscScalar   val[5];
      
      // Constant velocity (zero) - modify if necessary
      val[0] = 0.0; // P

      // for advdiff_analytic vx = 0, vz = 1
      if (sol->grd->mtype==ADVDIFF_ANALYTIC) {
        val[1] = 1.0; // vz
        val[2] = 1.0; // vz
      } else {
        val[1] = 0.0; // vz
        val[2] = 0.0; // vz
      }
      val[3] = 0.0; // vx
      val[4] = 0.0; // vx

      // Set density value - ELEMENT
      point[0].i = i; point[0].j = j; point[0].loc = ELEMENT; point[0].c = 0;
      point[1] = point[0]; point[1].loc = DOWN;
      point[2] = point[0]; point[2].loc = UP;
      point[3] = point[0]; point[3].loc = LEFT;
      point[4] = point[0]; point[4].loc = RIGHT;
      ierr = DMStagVecSetValuesStencil(sol->dmPV,sol->x,5,point,val,INSERT_VALUES); CHKERRQ(ierr);
    }
  }
  
  // Vector Assembly and Restore local vector
  ierr = VecAssemblyBegin(sol->x); CHKERRQ(ierr);
  ierr = VecAssemblyEnd  (sol->x); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}