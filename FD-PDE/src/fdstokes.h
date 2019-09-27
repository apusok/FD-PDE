/* Finite Differences (FD) PDE object [STOKES] */

#ifndef FDSTOKES_H
#define FDSTOKES_H

#include "petsc.h"
#include "prealloc_helper.h"
#include "bc.h"
#include "fd.h"
// #include "coefficient.h"

// typedef struct {
//   Coefficient eta_c, eta_n;
//   Coefficient fux, fuz, fp;
// } CoeffStokes;

// ---------------------------------------
// Function definitions
// ---------------------------------------
PetscErrorCode FDCreate_Stokes(FD);
PetscErrorCode FDDestroy_Stokes(FD);
PetscErrorCode FDView_Stokes(FD,PetscViewer);
PetscErrorCode FDCreateCoefficient_Stokes(FD);
// PetscErrorCode FDStokesGetCoefficients(FD,Coefficient*, Coefficient*,Coefficient*,Coefficient*,Coefficient*);
// PetscErrorCode FDStokesSetData(FD, DM, DM, BCList*, PetscInt);
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
PetscErrorCode FDBCApplyStokes(DM, Vec,DM, Vec, BCList*, PetscInt, PetscScalar**, PetscScalar**,PetscInt[], PetscScalar***);


#endif