/* Finite Differences-PDE (FD-PDE) object for [ADVDIFF] */

#ifndef FDPDE_ADVDIFF_H
#define FDPDE_ADVDIFF_H

#include "petsc.h"
#include "fdpde.h"

// ---------------------------------------
// Function definitions
// ---------------------------------------
PetscErrorCode FDPDECreate_AdvDiff(FDPDE);
PetscErrorCode CreateCoefficient_AdvDiff(FDPDE);
PetscErrorCode JacobianCreate_AdvDiff(FDPDE,Mat*);
PetscErrorCode JacobianPreallocator_AdvDiff(FDPDE,Mat);

// PREALLOCATOR
PetscErrorCode EnergyStencil(PetscInt,PetscInt,PetscInt,PetscInt,DMStagStencil*);

// RESIDUAL
PetscErrorCode FormFunction_AdvDiff(SNES,Vec,Vec,void*);
PetscErrorCode EnergyResidual(DM,Vec,DM,Vec,PetscScalar**,PetscScalar**,PetscInt,PetscInt,PetscInt[],PetscScalar*);
PetscErrorCode DMStagBCListApply_AdvDiff(DM,Vec,DM,Vec,DMStagBC*,PetscInt,PetscScalar**,PetscScalar**,PetscInt[],PetscScalar***);

#endif