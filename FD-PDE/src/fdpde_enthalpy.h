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

#define COEFF_C1  0 // edges
#define COEFF_C2  1
#define COEFF_v   2
#define COEFF_vf  3
#define COEFF_vs  4

#define MAX_COMPONENTS 20
#define STENCIL_ENTHALPY_NONZERO_PREALLOC 9

// ---------------------------------------
// Struct definitions
// ---------------------------------------
typedef struct {
  PetscScalar  H,T,TP,P,phi;
  PetscScalar  C[MAX_COMPONENTS],CF[MAX_COMPONENTS],CS[MAX_COMPONENTS];
} ThermoState;

typedef struct {
  PetscScalar  A1,B1,D1,A2,B2,D2;
  PetscScalar  C1[4],C2[4],v[4],vs[4],vf[4];
} CoeffState;

// user defined and model-dependent variables
typedef struct {
  AdvectSchemeType   advtype;
  TimeStepSchemeType timesteptype;
  DM                 dmP;
  Vec                xprev,coeffprev,xP,xPprev;
  PetscScalar        dt,theta;
  PetscErrorCode    (*form_user_bc)(DM,Vec,PetscScalar***,void*); // PRELIM
  PetscErrorCode    (*form_enthalpy_method)(FDPDE,PetscInt,PetscInt,PetscScalar,PetscScalar[],PetscScalar,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscInt,void*);
  void               *user_context;
  void               *user_context_bc; // PRELIM
  PetscInt           ncomponents, energy_variable;
  char               *description_enthalpy;
} EnthalpyData;

//enthalpy_method(fd,i,j,H,C,P,&TP,&T,&phi,CF,CS,ncomp,user);
//enthalpy_method(fd,i,j,TP,C,P,&H,&T,&phi,CF,CS,ncomp,user);

// ---------------------------------------
// Function definitions
// ---------------------------------------
PetscErrorCode FDPDECreate_Enthalpy(FDPDE);
PetscErrorCode FDPDEView_Enthalpy(FDPDE);
PetscErrorCode FDPDEDestroy_Enthalpy(FDPDE);
PetscErrorCode FDPDESetup_Enthalpy(FDPDE);
PetscErrorCode JacobianCreate_Enthalpy(FDPDE,Mat*);
PetscErrorCode JacobianPreallocator_Enthalpy(FDPDE,Mat);

// PREALLOCATOR
PetscErrorCode EnthalpyNonzeroStencil(PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,DMStagStencil*);

// RESIDUAL
PetscErrorCode FormFunction_Enthalpy(SNES,Vec,Vec,void*);
// PetscErrorCode DMStagBCListApply_Enthalpy(DM,Vec,DMStagBC*,PetscInt,PetscScalar***);

PetscErrorCode ApplyEnthalpyMethod(FDPDE,DM,Vec,DM,Vec,DM,Vec,EnthalpyData*,ThermoState*,CoeffState*);
PetscErrorCode CoeffCellData(DM,Vec,PetscInt,PetscInt,CoeffState*);
PetscErrorCode SolutionCellData(DM,Vec,PetscInt,PetscInt,PetscScalar*,PetscScalar*);
PetscErrorCode EnthalpyResidual(DM,ThermoState*,CoeffState*,ThermoState*,CoeffState*,PetscScalar**,PetscScalar**,EnthalpyData*,PetscInt,PetscInt, PetscScalar*);
PetscErrorCode EnthalpySteadyStateOperator(DM,ThermoState*,CoeffState*,PetscScalar**,PetscScalar**,PetscInt,PetscInt,AdvectSchemeType,PetscScalar*);
PetscErrorCode BulkCompositionResidual(DM dm,ThermoState*,CoeffState*,ThermoState*,CoeffState*,PetscScalar**,PetscScalar**,EnthalpyData*,PetscInt,PetscInt,PetscInt,PetscScalar*);
PetscErrorCode BulkCompositionSteadyStateOperator(DM,ThermoState*,CoeffState*,PetscScalar**,PetscScalar**,PetscInt,PetscInt,PetscInt,AdvectSchemeType,PetscScalar*);

// Set/Get Functions
PetscErrorCode FDPDEEnthalpySetAdvectSchemeType(FDPDE, AdvectSchemeType);
PetscErrorCode FDPDEEnthalpySetTimeStepSchemeType(FDPDE, TimeStepSchemeType);

PetscErrorCode FDPDEEnthalpySetTimestep(FDPDE,PetscScalar);
PetscErrorCode FDPDEEnthalpyGetTimestep(FDPDE, PetscScalar*);
PetscErrorCode FDPDEEnthalpyComputeExplicitTimestep(FDPDE, PetscScalar*);

PetscErrorCode FDPDEEnthalpyGetPrevSolution(FDPDE,Vec*);
PetscErrorCode FDPDEEnthalpyGetPrevCoefficient(FDPDE,Vec*);
PetscErrorCode FDPDEEnthalpyGetPressure(FDPDE,DM*,Vec*);
PetscErrorCode FDPDEEnthalpyGetPrevPressure(FDPDE,Vec*);

PetscErrorCode FDPDEEnthalpySetEnergyPrimaryVariable(FDPDE,const char);
PetscErrorCode FDPDEEnthalpySetNumberComponentsPhaseDiagram(FDPDE,PetscInt);
PetscErrorCode FDPDEEnthalpySetEnthalpyMethod(FDPDE fd, PetscErrorCode(*form_enthalpy_method)(FDPDE,PetscInt,PetscInt,PetscScalar,PetscScalar[],PetscScalar,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscInt,void*),const char description[],void*);
PetscErrorCode FDPDEEnthalpySetUserBC(FDPDE,PetscErrorCode(*form_user_bc)(DM,Vec,PetscScalar***,void*),void*); // PRELIM
PetscErrorCode FDPDEEnthalpyUpdateDiagnostics(FDPDE,DM,Vec,DM*,Vec*);

#endif