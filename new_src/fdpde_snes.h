#ifndef __fdpde_snes_h__
#define __fdpde_snes_h__

#include <petscvec.h>
#include <petscsnes.h>

#define SNESPICARDLS "picardls"

// ---------------------------------------
// Function definitions
// ---------------------------------------

// PetscErrorCode SNESPicardComputeFunctionDefault(SNES snes, Vec x, Vec f, void *ctx);
// PetscErrorCode SNESPicardLSGetAuxillarySolution(SNES snes,Vec *x);
// PetscErrorCode SNESPicardLSSetSplitFunction(SNES snes,Vec F,PetscErrorCode (*f)(SNES,Vec,Vec,Vec,void*));
// PetscErrorCode SNESCreate_PicardLS(SNES snes);

PetscErrorCode MatGetPreallocator(Mat A,Mat *preallocator);
PetscErrorCode MatPreallocatePhaseBegin(Mat A,Mat *preallocator);
PetscErrorCode MatPreallocatePhaseEnd(Mat A);

#endif