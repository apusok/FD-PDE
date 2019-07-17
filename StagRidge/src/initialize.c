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
// InitializeModel_SolCx
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

      DMStagStencil point, pointCoordx, pointCoordz;
      PetscScalar   x, z, rho;
        
      // Get coordinate of rho point
      point.i = i; point.j = j; point.loc = ELEMENT; point.c = 0;

      pointCoordx = point; pointCoordx.c = 0;
      pointCoordz = point; pointCoordz.c = 1;

      ierr = DMStagVecGetValuesStencil(dmCoord,coordLocal,1,&pointCoordx,&x); CHKERRQ(ierr);
      ierr = DMStagVecGetValuesStencil(dmCoord,coordLocal,1,&pointCoordz,&z); CHKERRQ(ierr);
        
      // Set density value
      rho  = PetscSinScalar(PETSC_PI*z) * PetscCosScalar(PETSC_PI*x); //sin(pi*z)*cos(pi*x); 
      ierr = DMStagVecSetValuesStencil(sol->dmCoeff,sol->coeff,1,&point,&rho,INSERT_VALUES); CHKERRQ(ierr);
    }
  }
  
  // Vector Assembly and Restore local vector
  ierr = VecAssemblyBegin(sol->coeff); CHKERRQ(ierr);
  ierr = VecAssemblyEnd  (sol->coeff); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}