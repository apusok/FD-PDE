/* Finite differences staggered grid context for STOKES-DARCY-3FIELD equations */

#ifndef FDPDE_STOKESDARCY3FIELD_H
#define FDPDE_STOKESDARCY3FIELD_H

#include "petsc.h"
#include "fdpde.h"

// dof ids
#define SD3_DOF_P   0
#define SD3_DOF_PC  1
#define SD3_DOF_V   0

#define SD3_COEFF_VERTEX_A    0 

#define SD3_COEFF_ELEMENT_C   0
#define SD3_COEFF_ELEMENT_A   1 
#define SD3_COEFF_ELEMENT_D1  2 
#define SD3_COEFF_ELEMENT_DC  3 

#define SD3_COEFF_FACE_B   0 
#define SD3_COEFF_FACE_D2  1 
#define SD3_COEFF_FACE_D3  2 
#define SD3_COEFF_FACE_D4  3 

// ---------------------------------------
// Function definitions
// ---------------------------------------
PetscErrorCode FDPDECreate_StokesDarcy2Field(FDPDE);

// Jacobian preallocator
PetscErrorCode JacobianPreallocator_StokesDarcy3Field(FDPDE,Mat);
PetscErrorCode JacobianCreate_StokesDarcy3Field(FDPDE,Mat*);

// RESIDUAL STENCILS
PetscErrorCode GetLocationSlots_Darcy3Field(DM,DM,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*);
PetscErrorCode ContinuityStencil_StokesDarcy3Field(PetscInt,PetscInt,PetscInt,PetscInt,DMBoundaryType,DMBoundaryType,DMStagStencil*);
PetscErrorCode CompactionStencil_StokesDarcy3Field(PetscInt,PetscInt,PetscInt,PetscInt,DMStagStencil*);
PetscErrorCode ContinuityResidual_Darcy3Field(PetscInt,PetscInt,PetscScalar***,PetscScalar***,PetscScalar**,PetscScalar**,PetscInt[],PetscInt[],PetscInt[],PetscInt[],PetscInt[],DMBoundaryType,DMBoundaryType,PetscScalar*);
PetscErrorCode CompactionResidual(PetscInt,PetscInt,PetscScalar***,PetscScalar***,PetscScalar**,PetscScalar**,PetscInt[],PetscInt[],PetscInt[],PetscScalar*);

// RESIDUAL
PetscErrorCode FormFunction_StokesDarcy3Field(SNES, Vec, Vec, void*);
PetscErrorCode DMStagBCListApplyFace_StokesDarcy3Field(PetscScalar***,PetscScalar***,DMStagBC*,PetscInt,PetscScalar**,PetscScalar**,PetscInt[],PetscInt[],PetscInt[],PetscInt[],PetscInt[],DMBoundaryType,DMBoundaryType,PetscScalar***);
PetscErrorCode DMStagBCListApplyElement_StokesDarcy3Field(PetscScalar***,PetscScalar***,DMStagBC*,PetscInt,PetscScalar**,PetscScalar**,PetscInt[],PetscInt[],PetscInt[],PetscInt[],PetscInt[],PetscInt[],DMBoundaryType,DMBoundaryType,PetscScalar***);

#endif