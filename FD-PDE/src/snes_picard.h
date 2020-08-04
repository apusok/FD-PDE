
#ifndef __snes_picard_h__
#define __snes_picard_h__

#include <petscvec.h>
#include <petscsnes.h>

#define SNESNPICARDLS "picardls"

PetscErrorCode SNESPicardComputeFunctionDefault(SNES snes, Vec x, Vec f, void *ctx);
PetscErrorCode SNESPicardLSGetAuxillarySolution(SNES snes,Vec *x);
PetscErrorCode SNESPicardLSSetSplitFunction(SNES snes,Vec F,
                                            PetscErrorCode (*f)(SNES,Vec,Vec,Vec,void*));
PetscErrorCode SNESCreate_PicardLS(SNES snes);

#endif
