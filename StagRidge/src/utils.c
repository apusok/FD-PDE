#include "stagridge.h"

// Concatenate two strings
PetscErrorCode StrCreateConcatenate(const char s1[], const char s2[], char **_result)
{
  size_t l1, l2;
  char *result;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  ierr = PetscStrlen(s1,&l1); CHKERRQ(ierr); 
  ierr = PetscStrlen(s2,&l2); CHKERRQ(ierr); 
  ierr = PetscMalloc1(l1+l2+1, &result); CHKERRQ(ierr); // +1 for the null-terminator
  ierr = PetscStrcpy(result, s1); CHKERRQ(ierr); 
  ierr = PetscStrcat(result, s2); CHKERRQ(ierr);
  *_result = result;

  PetscFunctionReturn(0);
}

// Get coordinate of stencil point
PetscErrorCode GetCoordinatesStencil(DM dm, Vec vec, PetscInt n, DMStagStencil point[], PetscScalar x[], PetscScalar z[])
{
  PetscScalar    xx[2];
  PetscInt       i;
  DMStagStencil  pointx[2];
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  for (i = 0; i<n; ++i) {
    // Get coordinates x,z of point
    pointx[0] = point[i]; pointx[0].c = 0;
    pointx[1] = point[i]; pointx[1].c = 1;

    ierr = DMStagVecGetValuesStencil(dm,vec,2,pointx,xx); CHKERRQ(ierr);
    x[i] = xx[0];
    z[i] = xx[1];
  }

  PetscFunctionReturn(0);
}