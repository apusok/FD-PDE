/* Finite differences staggered grid context for STOKES-DARCY-2FIELD equations */

#ifndef FDPDE_STOKESDARCY2FIELD_H
#define FDPDE_STOKESDARCY2FIELD_H

#include "petsc.h"
#include "fdpde.h"

// dof ids
#define SD2_DOF_P   0
#define SD2_DOF_V   0

#define SD2_COEFF_VERTEX_A    0 

#define SD2_COEFF_ELEMENT_C   0
#define SD2_COEFF_ELEMENT_A   1 
#define SD2_COEFF_ELEMENT_D1  2 

#define SD2_COEFF_FACE_B   0 
#define SD2_COEFF_FACE_D2  1 
#define SD2_COEFF_FACE_D3  2 

// ---------------------------------------
// Function definitions
// ---------------------------------------
PetscErrorCode FDPDECreate_StokesDarcy2Field(FDPDE);

// Function pointers
PetscErrorCode JacobianPreallocator_StokesDarcy2Field(FDPDE,Mat);
PetscErrorCode JacobianCreate_StokesDarcy2Field(FDPDE,Mat*);

// RESIDUAL STENCILS
PetscErrorCode GetLocationSlots_Darcy2Field(DM,PetscInt*,PetscInt*,PetscInt*);
PetscErrorCode ContinuityStencil_StokesDarcy2Field(PetscInt,PetscInt,PetscInt,PetscInt,DMStagStencil*);
PetscErrorCode ContinuityResidual_Darcy2Field(PetscInt,PetscInt,PetscScalar***,PetscScalar***,PetscScalar**,PetscScalar**,PetscInt[],PetscInt[],PetscInt[],PetscInt[],PetscScalar*);
PetscErrorCode XMomentumResidual_Darcy2Field(PetscInt,PetscInt,PetscScalar***,PetscScalar***,PetscScalar**,PetscScalar**,PetscInt[],PetscInt[],PetscInt[],PetscScalar*);
PetscErrorCode ZMomentumResidual_Darcy2Field(PetscInt,PetscInt,PetscScalar***,PetscScalar***,PetscScalar**,PetscScalar**,PetscInt[],PetscInt[],PetscInt[],PetscScalar*);

// RESIDUAL
PetscErrorCode FormFunction_StokesDarcy2Field(SNES, Vec, Vec, void*);
PetscErrorCode DMStagBCListApplyFace_StokesDarcy2Field(PetscScalar***,PetscScalar***,DMStagBC*,PetscInt,PetscScalar**,PetscScalar**,PetscInt[],PetscInt[],PetscInt[],PetscScalar***);
PetscErrorCode DMStagBCListApplyElement_StokesDarcy2Field(PetscScalar***,PetscScalar***,DMStagBC*,PetscInt,PetscScalar**,PetscScalar**,PetscInt[],PetscInt[],PetscInt[],PetscScalar***);

#endif