/* Finite Differences (FD) PDE object [STOKES] */

#ifndef FDSTOKES_H
#define FDSTOKES_H

#include "petsc.h"
#include "fd.h"

// ---------------------------------------
// Function definitions
// ---------------------------------------
PetscErrorCode FDCreate_Stokes(FD);
PetscErrorCode FDCreateCoefficient_Stokes(FD);
PetscErrorCode FDJacobianPreallocator_Stokes(FD);

// FD STOKES PREALLOCATOR STENCIL
PetscErrorCode ContinuityStencil(PetscInt,PetscInt,DMStagStencil*);
PetscErrorCode XMomentumStencil(PetscInt,PetscInt,PetscInt,DMStagStencil*);
PetscErrorCode ZMomentumStencil(PetscInt,PetscInt,PetscInt,DMStagStencil*);

// FD STOKES PHYSICS
PetscErrorCode ContinuityResidual(DM,Vec,DM,Vec,PetscScalar**,PetscScalar**,PetscInt,PetscInt,PetscInt[],PetscScalar*);
PetscErrorCode XMomentumResidual(DM,Vec,DM,Vec,PetscScalar**,PetscScalar**,PetscInt,PetscInt,PetscInt[],PetscScalar*);
PetscErrorCode ZMomentumResidual(DM,Vec,DM,Vec,PetscScalar**,PetscScalar**,PetscInt,PetscInt,PetscInt[],PetscScalar*);

// RESIDUAL
PetscErrorCode FormFunction_Stokes(SNES, Vec, Vec, void*);
PetscErrorCode FDBCApplyStokes(DM, Vec,DM, Vec, DMStagBC*, PetscInt, PetscScalar**, PetscScalar**,PetscInt[], PetscScalar***);


#endif