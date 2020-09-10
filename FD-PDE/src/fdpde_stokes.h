/* Finite Differences-PDE (FD-PDE) object for [STOKES] */

#ifndef FDPDE_STOKES_H
#define FDPDE_STOKES_H

#include "petsc.h"
#include "fdpde.h"

#define STENCIL_STOKES_MOMENTUM_LIN    11
#define STENCIL_STOKES_MOMENTUM_NONLIN 27

// ---------------------------------------
// Function definitions
// ---------------------------------------
PetscErrorCode FDPDECreate_Stokes(FDPDE);
PetscErrorCode JacobianPreallocator_Stokes(FDPDE,Mat);
PetscErrorCode JacobianCreate_Stokes(FDPDE,Mat*);

// PREALLOCATOR STENCIL
PetscErrorCode ContinuityStencil(PetscInt,PetscInt,DMStagStencil*);
PetscErrorCode XMomentumStencil(PetscInt,PetscInt,PetscInt,PetscInt,DMStagStencil*,PetscInt);
PetscErrorCode ZMomentumStencil(PetscInt,PetscInt,PetscInt,PetscInt,DMStagStencil*,PetscInt);

// RESIDUAL STENCILS
PetscErrorCode ContinuityResidual(DM,Vec,DM,Vec,PetscScalar**,PetscScalar**,PetscInt,PetscInt,PetscInt[],PetscScalar*);
PetscErrorCode XMomentumResidual(DM,Vec,DM,Vec,PetscScalar**,PetscScalar**,PetscInt,PetscInt,PetscInt[],PetscScalar*);
PetscErrorCode ZMomentumResidual(DM,Vec,DM,Vec,PetscScalar**,PetscScalar**,PetscInt,PetscInt,PetscInt[],PetscScalar*);

// RESIDUAL
PetscErrorCode FormFunction_Stokes(SNES, Vec, Vec, void*);
PetscErrorCode FormFunctionSplit_Stokes(SNES snes, Vec x, Vec x2, Vec f, void *ctx);
PetscErrorCode DMStagBCListApplyFace_Stokes(DM, Vec,DM, Vec, DMStagBC*, PetscInt, PetscScalar**, PetscScalar**,PetscInt[], PetscScalar***);
PetscErrorCode DMStagBCListApplyElement_Stokes(DM, Vec,DM, Vec, DMStagBC*, PetscInt, PetscScalar**, PetscScalar**,PetscInt[], PetscScalar***);

#endif
