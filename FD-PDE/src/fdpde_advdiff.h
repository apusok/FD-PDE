/* Finite Differences-PDE (FD-PDE) object for [ADVDIFF] */

#ifndef FDPDE_ADVDIFF_H
#define FDPDE_ADVDIFF_H

#include "petsc.h"
#include "fdpde.h"

// ---------------------------------------
// Function definitions
// ---------------------------------------
PetscErrorCode FDPDECreate_AdvDiff(FDPDE);
PetscErrorCode JacobianCreate_AdvDiff(FDPDE,Mat*);
PetscErrorCode JacobianPreallocator_AdvDiff(FDPDE,Mat);

// PREALLOCATOR
PetscErrorCode EnergyStencil(PetscInt,PetscInt,PetscInt,PetscInt,DMStagStencil*);

// RESIDUAL
PetscErrorCode FormFunction_AdvDiff(SNES,Vec,Vec,void*);
PetscErrorCode EnergyResidual(DM,Vec,DM,Vec,PetscScalar**,PetscScalar**,PetscInt,PetscInt,AdvectType,PetscScalar*);
PetscErrorCode DMStagBCListApply_AdvDiff(DM,Vec,DM,Vec,DMStagBC*,PetscInt,PetscScalar**,PetscScalar**,PetscScalar***);

// ADVECTION
PetscErrorCode AdvectionResidual(PetscScalar[],PetscScalar[],PetscScalar[],PetscScalar[],AdvectType,PetscScalar*);
PetscScalar UpwindAdvection(PetscScalar[], PetscScalar[], PetscScalar[], PetscScalar[]);
PetscScalar FrommAdvection(PetscScalar[], PetscScalar[], PetscScalar[], PetscScalar[]);

#endif