/* Finite Differences-PDE (FD-PDE) object for [ADVDIFF] */

#ifndef FDPDE_ADVDIFF_H
#define FDPDE_ADVDIFF_H

#include "petsc.h"
#include "fdpde.h"

// ---------------------------------------
// Struct definitions
// ---------------------------------------
// Advection type
typedef enum { ADV_UNINIT = 0, ADV_NONE, ADV_UPWIND, ADV_FROMM } AdvectSchemeType;

// Time-stepping type
typedef enum { TS_UNINIT = 0, TS_NONE, TS_FORWARD_EULER, TS_BACKWARD_EULER, TS_CRANK_NICHOLSON } TimeStepSchemeType;

// user defined and model-dependent variables
typedef struct {
  AdvectSchemeType   advtype;
  TimeStepSchemeType timesteptype;
  Vec                xprev,coeffprev;
  PetscScalar        dt,dt_user,CFL,theta;
} AdvDiffData;

// ---------------------------------------
// Function definitions
// ---------------------------------------
PetscErrorCode FDPDECreate_AdvDiff(FDPDE);
// PetscErrorCode FDPDESetUp_AdvDiff(FDPDE);
PetscErrorCode FDPDEView_AdvDiff(FDPDE);
PetscErrorCode FDPDEDestroy_AdvDiff(FDPDE);
PetscErrorCode JacobianCreate_AdvDiff(FDPDE,Mat*);
PetscErrorCode JacobianPreallocator_AdvDiff(FDPDE,Mat);

// PREALLOCATOR
PetscErrorCode EnergyStencil(PetscInt,PetscInt,PetscInt,PetscInt,DMStagStencil*);

// RESIDUAL
PetscErrorCode FormFunction_AdvDiff(SNES,Vec,Vec,void*);
PetscErrorCode EnergyResidual(DM,Vec,DM,Vec,PetscScalar**,PetscScalar**,PetscInt,PetscInt,AdvectSchemeType,PetscScalar*,PetscScalar*);
PetscErrorCode DMStagBCListApply_AdvDiff(DM,Vec,DM,Vec,DMStagBC*,PetscInt,PetscScalar**,PetscScalar**,PetscScalar***);
PetscErrorCode UpdateTimeStep_AdvDiff(AdvDiffData*,DM,Vec);

// ADVECTION
PetscErrorCode AdvectionResidual(PetscScalar[],PetscScalar[],PetscScalar[],PetscScalar[],AdvectSchemeType,PetscScalar*);
PetscScalar UpwindAdvection(PetscScalar[], PetscScalar[], PetscScalar[], PetscScalar[]);
PetscScalar FrommAdvection(PetscScalar[], PetscScalar[], PetscScalar[], PetscScalar[]);

// Set Functions
PetscErrorCode FDPDEAdvDiffSetAdvectSchemeType(FDPDE, AdvectSchemeType);
PetscErrorCode FDPDEAdvDiffSetTimeStepSchemeType(FDPDE, TimeStepSchemeType);
PetscErrorCode FDPDEAdvDiffGetPrevSolution(FDPDE,Vec*);
PetscErrorCode FDPDEAdvDiffSetTimestep(FDPDE,PetscScalar,PetscScalar);
PetscErrorCode FDPDEAdvDiffGetTimestep(FDPDE, PetscScalar*);

#endif