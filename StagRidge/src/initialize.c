#include "stagridge.h"

// ---------------------------------------
// InitializeModel
// ---------------------------------------
PetscErrorCode InitializeModel(SolverCtx *sol)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  
  // Different model setups
  if        (sol->grd->mtype == SOLCX){
    ierr = InitializeModel_SolCx(sol); CHKERRQ(ierr);
  } else if (sol->grd->mtype == MOR  ){
    //ierr = InitializeModel_MOR(sol); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}
// ---------------------------------------
// InitializeModel_SolCx - density defined on Vz-edges
// ---------------------------------------
PetscErrorCode InitializeModel_SolCx(SolverCtx *sol)
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

      DMStagStencil point;
      PetscScalar   x[2], rho;
        
      // Set density value - ELEMENT
      point.i = i; point.j = j; point.loc = ELEMENT; point.c = 0;
      ierr = GetCoordinateStencilPoint(dmCoord, coordLocal, point, x); CHKERRQ(ierr);
      rho  = PetscSinScalar(PETSC_PI*x[1]) * PetscCosScalar(PETSC_PI*x[0]); //sin(pi*z)*cos(pi*x); 
      //rho  = 1.0;
      ierr = DMStagVecSetValuesStencil(sol->dmCoeff,sol->coeff,1,&point,&rho,INSERT_VALUES); CHKERRQ(ierr);

      // Set density value - DOWN
      point.loc = DOWN;
      ierr = GetCoordinateStencilPoint(dmCoord, coordLocal, point, x); CHKERRQ(ierr);
      rho  = PetscSinScalar(PETSC_PI*x[1]) * PetscCosScalar(PETSC_PI*x[0]); //sin(pi*z)*cos(pi*x); 
      //rho  = 1.0;
      ierr = DMStagVecSetValuesStencil(sol->dmCoeff,sol->coeff,1,&point,&rho,INSERT_VALUES); CHKERRQ(ierr);

      // Set density value - UP
      point.loc = UP;
      ierr = GetCoordinateStencilPoint(dmCoord, coordLocal, point, x); CHKERRQ(ierr);
      rho  = PetscSinScalar(PETSC_PI*x[1]) * PetscCosScalar(PETSC_PI*x[0]); //sin(pi*z)*cos(pi*x); 
      //rho  = 1.0;
      ierr = DMStagVecSetValuesStencil(sol->dmCoeff,sol->coeff,1,&point,&rho,INSERT_VALUES); CHKERRQ(ierr);

      // Set density value - LEFT
      point.loc = LEFT;
      ierr = GetCoordinateStencilPoint(dmCoord, coordLocal, point, x); CHKERRQ(ierr);
      rho  = PetscSinScalar(PETSC_PI*x[1]) * PetscCosScalar(PETSC_PI*x[0]); //sin(pi*z)*cos(pi*x); 
      //rho  = 1.0;
      ierr = DMStagVecSetValuesStencil(sol->dmCoeff,sol->coeff,1,&point,&rho,INSERT_VALUES); CHKERRQ(ierr);

      // Set density value - RIGHT
      point.loc = RIGHT;
      ierr = GetCoordinateStencilPoint(dmCoord, coordLocal, point, x); CHKERRQ(ierr);
      rho  = PetscSinScalar(PETSC_PI*x[1]) * PetscCosScalar(PETSC_PI*x[0]); //sin(pi*z)*cos(pi*x); 
      //rho  = 1.0;
      ierr = DMStagVecSetValuesStencil(sol->dmCoeff,sol->coeff,1,&point,&rho,INSERT_VALUES); CHKERRQ(ierr);
    }
  }
  
  // Vector Assembly and Restore local vector
  ierr = VecAssemblyBegin(sol->coeff); CHKERRQ(ierr);
  ierr = VecAssemblyEnd  (sol->coeff); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}