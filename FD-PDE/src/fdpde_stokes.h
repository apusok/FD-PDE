/* Finite Differences-PDE (FD-PDE) object for [STOKES] */

#ifndef FDPDE_STOKES_H
#define FDPDE_STOKES_H

#include "petsc.h"
#include "fdpde.h"

#define STENCIL_STOKES_MOMENTUM_LIN    11
#define STENCIL_STOKES_MOMENTUM_NONLIN 27

// dof ids
#define S_DOF_P   0
#define S_DOF_V   0

#define S_COEFF_VERTEX_A    0 
#define S_COEFF_ELEMENT_C   0
#define S_COEFF_ELEMENT_A   1 
#define S_COEFF_FACE_B      0 

// ---------------------------------------
// Function definitions
// ---------------------------------------
PetscErrorCode FDPDECreate_Stokes(FDPDE);
PetscErrorCode JacobianPreallocator_Stokes(FDPDE,Mat);
PetscErrorCode JacobianCreate_Stokes(FDPDE,Mat*);

// PREALLOCATOR STENCIL
PetscErrorCode ContinuityStencil(PetscInt,PetscInt,DMStagStencil*);
PetscErrorCode XMomentumStencil(PetscInt,PetscInt,PetscInt,PetscInt,DMStagStencil*,DMBoundaryType,DMBoundaryType,PetscInt);
PetscErrorCode ZMomentumStencil(PetscInt,PetscInt,PetscInt,PetscInt,DMStagStencil*,DMBoundaryType,DMBoundaryType,PetscInt);

// RESIDUAL STENCILS
PetscErrorCode GetLocationSlots(DM,DM,PetscInt*,PetscInt*,PetscInt*,PetscInt*);
PetscErrorCode ContinuityResidual(PetscInt,PetscInt,PetscScalar***,PetscScalar***,PetscScalar**,PetscScalar**,PetscInt[],PetscInt[],PetscInt[],PetscScalar*);
PetscErrorCode XMomentumResidual(PetscInt,PetscInt,PetscScalar***,PetscScalar***,PetscScalar**,PetscScalar**,PetscInt[],PetscInt[],PetscInt[],PetscInt[],PetscInt[],DMBoundaryType,PetscScalar*);
PetscErrorCode ZMomentumResidual(PetscInt,PetscInt,PetscScalar***,PetscScalar***,PetscScalar**,PetscScalar**,PetscInt[],PetscInt[],PetscInt[],PetscInt[],PetscInt[],DMBoundaryType,PetscScalar*);

// RESIDUAL
PetscErrorCode FormFunction_Stokes(SNES, Vec, Vec, void*);
PetscErrorCode FormFunctionSplit_Stokes(SNES snes, Vec x, Vec x2, Vec f, void *ctx);
PetscErrorCode DMStagBCListApplyFace_Stokes(PetscScalar***,PetscScalar***,DMStagBC*,PetscInt,PetscScalar**,PetscScalar**,PetscInt[],PetscInt[],PetscInt[],PetscInt[],PetscInt[],DMBoundaryType,DMBoundaryType,PetscScalar***);
PetscErrorCode DMStagBCListApplyElement_Stokes(PetscScalar***,PetscScalar***,DMStagBC*,PetscInt,PetscScalar**,PetscScalar**,PetscInt[],PetscInt[],PetscInt[],PetscInt[],PetscInt[],PetscScalar***);

#endif
