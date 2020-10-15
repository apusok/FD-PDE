/* Finite Differences-PDE (FD-PDE) object for [ENTHALPY] */

#ifndef FDPDE_ENTHALPY_H
#define FDPDE_ENTHALPY_H

#include "petsc.h"
#include "fdpde.h"
#include "fdpde_advdiff.h"

// DOF color for coefficients
#define COEFF_A1  0 // element
#define COEFF_B1  1
#define COEFF_D1  2
#define COEFF_A2  3
#define COEFF_B2  4
#define COEFF_D2  5
#define COEFF_M1  6
#define COEFF_N1  7
#define COEFF_O1  8
#define COEFF_P1  9
#define COEFF_Q1  10
#define COEFF_M2  11
#define COEFF_N2  12
#define COEFF_O2  13
#define COEFF_P2  14
#define COEFF_Q2  15

#define COEFF_C1  0 // edges
#define COEFF_C2  1
#define COEFF_v   2
#define COEFF_vf  3
#define COEFF_vs  4

// ---------------------------------------
// Struct definitions
// ---------------------------------------
// user defined and model-dependent variables
typedef struct {
  AdvectSchemeType   advtype;
  TimeStepSchemeType timesteptype;
  Vec                xprev,coeffprev;
  PetscScalar        dt,theta;
  PetscErrorCode    (*form_CS)(FDPDE,DM,Vec,DM,Vec,DM,Vec,void*);
  PetscErrorCode    (*form_CF)(FDPDE,DM,Vec,DM,Vec,DM,Vec,void*);
  void               *user_context;
  DM                 dmphiT, dmcomp;
  Vec                xphiT,xCF,xCS;
  PetscInt           ncomponents, energy_variable;
} EnthalpyData;

// ---------------------------------------
// Function definitions
// ---------------------------------------
PetscErrorCode FDPDECreate_Enthalpy(FDPDE);
PetscErrorCode FDPDEView_Enthalpy(FDPDE);
PetscErrorCode FDPDESetUp_Enthalpy(FDPDE);
PetscErrorCode FDPDEDestroy_Enthalpy(FDPDE);
PetscErrorCode JacobianCreate_Enthalpy(FDPDE,Mat*);
PetscErrorCode JacobianPreallocator_Enthalpy(FDPDE,Mat);

// PREALLOCATOR
PetscErrorCode EnthalpyNonzeroStencil(PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,DMStagStencil*);

// RESIDUAL
PetscErrorCode FormFunction_Enthalpy(SNES,Vec,Vec,void*);
PetscErrorCode EnthalpyResidual_H(DM,Vec,DM,Vec,Vec,Vec,DM,Vec,PetscScalar**,PetscScalar**,EnthalpyData*,PetscInt,PetscInt,PetscScalar*);
PetscErrorCode EnthalpySteadyStateOperator_H(DM,Vec,DM,Vec,DM,Vec,PetscScalar**,PetscScalar**,PetscInt,PetscInt,AdvectSchemeType,PetscScalar*);
PetscErrorCode EnthalpyResidual_TP(DM,Vec,DM,Vec,Vec,Vec,DM,Vec,PetscScalar**,PetscScalar**,EnthalpyData*,PetscInt,PetscInt,PetscScalar*);
PetscErrorCode EnthalpySteadyStateOperator_TP(DM,Vec,DM,Vec,DM,Vec,PetscScalar**,PetscScalar**,PetscInt,PetscInt,AdvectSchemeType,PetscScalar*);
PetscErrorCode BulkCompositionResidual(DM,Vec,DM,Vec,Vec,Vec,DM,Vec,DM,Vec,Vec,PetscScalar**,PetscScalar**,EnthalpyData*,PetscInt,PetscInt,PetscInt,PetscScalar*);
PetscErrorCode BulkCompositionSteadyStateOperator(DM,Vec,DM,Vec,DM,Vec,DM,Vec,Vec,PetscScalar**,PetscScalar**,PetscInt,PetscInt,PetscInt,AdvectSchemeType,PetscScalar*);
PetscErrorCode DMStagBCListApply_Enthalpy(DM,Vec,DMStagBC*,PetscInt,PetscScalar***);

// // Set Functions
PetscErrorCode FDPDEEnthalpySetAdvectSchemeType(FDPDE, AdvectSchemeType);
PetscErrorCode FDPDEEnthalpySetTimeStepSchemeType(FDPDE, TimeStepSchemeType);

PetscErrorCode FDPDEEnthalpyGetPrevSolution(FDPDE,Vec*);
PetscErrorCode FDPDEEnthalpyGetPrevCoefficient(FDPDE,Vec*);
PetscErrorCode FDPDEEnthalpyGetPorosityTemperature(FDPDE,DM*,Vec*);
PetscErrorCode FDPDEEnthalpyGetPhaseComposition(FDPDE,DM*,Vec*,Vec*);

PetscErrorCode FDPDEEnthalpySetTimestep(FDPDE,PetscScalar);
PetscErrorCode FDPDEEnthalpyGetTimestep(FDPDE, PetscScalar*);
// PetscErrorCode FDPDEEnthalpyComputeExplicitTimestep(FDPDE, PetscScalar*);
PetscErrorCode FDPDEEnthalpySetFunctionsPhaseDiagram(FDPDE,PetscErrorCode(*form_CF)(FDPDE fd,DM,Vec,DM,Vec,DM,Vec,void*), PetscErrorCode (*form_CS)(FDPDE fd,DM,Vec,DM,Vec,DM,Vec,void*), void *data);
PetscErrorCode FDPDEEnthalpySetNumberComponentsPhaseDiagram(FDPDE,PetscInt);
PetscErrorCode FDPDEEnthalpySetEnergyPrimaryVariable(FDPDE,const char);

// FDPDEEnthalpyUpdateDiagnostics(fd)
// FDPDEEnthalpySolvePhiT(fd)
// FDPDEEnthalpyUpdateCompositions(fd)

#endif