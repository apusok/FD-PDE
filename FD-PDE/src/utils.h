
#ifndef __UTILS_H__
#define __UTILS_H__

#include "petsc.h"
#include "petscvec.h"
#include "petscdm.h"
#include "petscdmstag.h"

PetscErrorCode DMStagExtract1DComponent(DM,Vec,DMStagStencilLocation,PetscInt,PetscScalar,PetscScalar*);
PetscErrorCode GetCoordinatesStencil(DM,Vec,PetscInt,DMStagStencil*,PetscScalar*,PetscScalar*);

#endif
