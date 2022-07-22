
#ifndef __composite_prealloc_utils_h__
#define __composite_prealloc_utils_h__

PetscErrorCode FDPDECoupledCreateMatrix(PetscInt ndm,DM dm[],MatType mtype,Mat *A);
PetscErrorCode FDPDECoupledCreateMatrix2(PetscInt ndm,DM dm[],PetscBool mask[],MatType mtype,Mat *A);

#endif
