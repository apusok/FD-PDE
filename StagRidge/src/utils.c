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
PetscErrorCode GetCoordinateStencilPoint(DM dm, Vec vec, DMStagStencil point, PetscScalar x[])
{
    DMStagStencil  pointx[2];
    PetscErrorCode ierr;
    PetscFunctionBeginUser;
    
    // Get coordinates x,z of point
    pointx[0] = point; pointx[0].c = 0;
    pointx[1] = point; pointx[1].c = 1;
    ierr = DMStagVecGetValuesStencil(dm,vec,2,pointx,x); CHKERRQ(ierr);

    PetscFunctionReturn(0);
}