
#if !defined(__PREALLOC_HELPER_H)
#define __PREALLOC_HELPER_H

#include <petscis.h>
#include <petscmat.h>
#include <petscdm.h>

PetscErrorCode MatGetPreallocator(Mat A,Mat *preallocator);
PetscErrorCode MatPreallocatePhaseBegin(Mat A,Mat *preallocator);
PetscErrorCode MatPreallocatePhaseEnd(Mat A);

#endif
