/* Finite differences staggered grid context for STOKES-DARCY-2FIELD equations */

#ifndef FDPDE_STOKESDARCY2FIELD_H
#define FDPDE_STOKESDARCY2FIELD_H

#include "petsc.h"
#include "fdpde.h"

// ---------------------------------------
// Function definitions
// ---------------------------------------
PetscErrorCode FDPDECreate_StokesDarcy2Field(FDPDE);

// Function pointers
PetscErrorCode JacobianPreallocator_StokesDarcy2Field(FDPDE,Mat);
PetscErrorCode JacobianCreate_StokesDarcy2Field(FDPDE,Mat*);

// RESIDUAL STENCILS
PetscErrorCode ContinuityStencil_StokesDarcy2Field(PetscInt,PetscInt,PetscInt,PetscInt,DMStagStencil*);
PetscErrorCode ContinuityResidual_Darcy2Field(DM,Vec,DM,Vec,PetscScalar**,PetscScalar**,PetscInt,PetscInt,PetscInt[],PetscScalar*);
PetscErrorCode XMomentumResidual_Darcy2Field(DM,Vec,DM,Vec,PetscScalar**,PetscScalar**,PetscInt,PetscInt,PetscInt[],PetscScalar*);
PetscErrorCode ZMomentumResidual_Darcy2Field(DM,Vec,DM,Vec,PetscScalar**,PetscScalar**,PetscInt,PetscInt,PetscInt[],PetscScalar*);

// RESIDUAL
PetscErrorCode FormFunction_StokesDarcy2Field(SNES, Vec, Vec, void*);
PetscErrorCode DMStagBCListApplyFace_StokesDarcy2Field(DM, Vec,DM, Vec, DMStagBC*, PetscInt, PetscScalar**, PetscScalar**,PetscInt[], PetscScalar***);
PetscErrorCode DMStagBCListApplyElement_StokesDarcy2Field(DM, Vec,DM, Vec, DMStagBC*, PetscInt, PetscScalar**, PetscScalar**,PetscInt[], PetscScalar***);

#endif