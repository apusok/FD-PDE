/* Contains tools for constitutive equations  */

#ifndef CONSTEQ_H
#define CONSTEQ_H

#include "petsc.h"

// ---------------------------------------
// Function definitions
// ---------------------------------------
PetscErrorCode DMStagGetPointStrainRates(DM,Vec,PetscInt,DMStagStencil*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*);

PetscErrorCode get_exx_center(DM,Vec,PetscScalar**,PetscScalar**,PetscInt,PetscInt,PetscInt[],PetscScalar*);
PetscErrorCode get_ezz_center(DM,Vec,PetscScalar**,PetscScalar**,PetscInt,PetscInt,PetscInt[],PetscScalar*);
PetscErrorCode get_exz_cornerSW(DM,Vec,PetscScalar**,PetscScalar**,PetscInt,PetscInt,PetscInt[],PetscScalar*);
PetscErrorCode get_exz_cornerNW(DM,Vec,PetscScalar**,PetscScalar**,PetscInt,PetscInt,PetscInt[],PetscScalar*);
PetscErrorCode get_exz_cornerSE(DM,Vec,PetscScalar**,PetscScalar**,PetscInt,PetscInt,PetscInt[],PetscScalar*);
PetscErrorCode get_exz_cornerNE(DM,Vec,PetscScalar**,PetscScalar**,PetscInt,PetscInt,PetscInt[],PetscScalar*);

#endif