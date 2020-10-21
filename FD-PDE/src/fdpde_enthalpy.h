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
#define COEFF_a   6
#define COEFF_b   7
#define COEFF_c   8
#define COEFF_d   9
#define COEFF_e  10
#define COEFF_P  11

#define COEFF_C1  0 // edges
#define COEFF_C2  1
#define COEFF_v   2
#define COEFF_vf  3
#define COEFF_vs  4

#define MAX_COMPONENTS 20

// ---------------------------------------
// Struct definitions
// ---------------------------------------
typedef struct {
  PetscScalar  H,T,TP,P,phi;
  PetscScalar  C[MAX_COMPONENTS],CF[MAX_COMPONENTS],CS[MAX_COMPONENTS];
} ThermoState;

typedef struct {
  PetscScalar  A1,B1,D1,A2,B2,D2,a,b,c,d,e;
  PetscScalar  C1[4],C2[4],v[4],vs[4],vf[4];
} CoeffState;

// user defined and model-dependent variables
typedef struct {
  AdvectSchemeType   advtype;
  TimeStepSchemeType timesteptype;
  Vec                xprev,coeffprev;
  PetscScalar        dt,theta;
  PetscErrorCode    (*form_Tsol_Tliq)(PetscScalar[],PetscScalar,PetscInt,void*,PetscScalar*,PetscScalar*);
  PetscErrorCode    (*form_Cs_Cf)(PetscScalar,PetscScalar[],PetscScalar,PetscInt,void*,PetscScalar*,PetscScalar*);
  PetscErrorCode    (*form_phi)(PetscScalar,PetscScalar[],PetscScalar,void*,PetscScalar*);
  PetscErrorCode    (*form_user_bc)(DM,Vec,PetscScalar***,void*);
  void               *user_context;
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
PetscErrorCode DMStagBCListApply_Enthalpy(DM,Vec,DMStagBC*,PetscInt,PetscScalar***);

PetscErrorCode Enthalpy_TP(DM,Vec,DM,Vec,EnthalpyData*,ThermoState*,CoeffState*);
PetscErrorCode Enthalpy_H(DM,Vec,DM,Vec,EnthalpyData*,ThermoState*,CoeffState*);
PetscErrorCode CoeffCellData(DM,Vec,PetscInt,PetscInt,CoeffState*, PetscScalar*);
PetscErrorCode SolutionCellData(DM,Vec,PetscInt,PetscInt,PetscScalar*,PetscScalar*);
PetscErrorCode EnthalpyCellData_TPC(PetscScalar,PetscScalar*,PetscScalar,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,EnthalpyData*,CoeffState);
PetscErrorCode EnthalpyCellData_HC(PetscScalar,PetscScalar*,PetscScalar,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,EnthalpyData*,CoeffState);
PetscErrorCode EnthalpyResidual(DM,ThermoState*,CoeffState*,ThermoState*,CoeffState*,PetscScalar**,PetscScalar**,EnthalpyData*,PetscInt,PetscInt, PetscScalar*);
PetscErrorCode EnthalpySteadyStateOperator(DM,ThermoState*,CoeffState*,PetscScalar**,PetscScalar**,PetscInt,PetscInt,AdvectSchemeType,PetscScalar*);
PetscErrorCode BulkCompositionResidual(DM dm,ThermoState*,CoeffState*,ThermoState*,CoeffState*,PetscScalar**,PetscScalar**,EnthalpyData*,PetscInt,PetscInt,PetscInt,PetscScalar*);
PetscErrorCode BulkCompositionSteadyStateOperator(DM,ThermoState*,CoeffState*,PetscScalar**,PetscScalar**,PetscInt,PetscInt,PetscInt,AdvectSchemeType,PetscScalar*);

// // Set Functions
PetscErrorCode FDPDEEnthalpySetAdvectSchemeType(FDPDE, AdvectSchemeType);
PetscErrorCode FDPDEEnthalpySetTimeStepSchemeType(FDPDE, TimeStepSchemeType);

PetscErrorCode FDPDEEnthalpyGetPrevSolution(FDPDE,Vec*);
PetscErrorCode FDPDEEnthalpyGetPrevCoefficient(FDPDE,Vec*);

PetscErrorCode FDPDEEnthalpySetTimestep(FDPDE,PetscScalar);
PetscErrorCode FDPDEEnthalpyGetTimestep(FDPDE, PetscScalar*);
// PetscErrorCode FDPDEEnthalpyComputeExplicitTimestep(FDPDE, PetscScalar*);
PetscErrorCode FDPDEEnthalpySetNumberComponentsPhaseDiagram(FDPDE,PetscInt);
PetscErrorCode FDPDEEnthalpySetFunctionsPhaseDiagram(FDPDE, PetscErrorCode(*form_Tsol_Tliq)(PetscScalar[],PetscScalar,PetscInt,void*,PetscScalar*,PetscScalar*), 
                                                            PetscErrorCode(*form_Cs_Cf)(PetscScalar,PetscScalar[],PetscScalar,PetscInt,void*,PetscScalar*,PetscScalar*), 
                                                            PetscErrorCode(*form_phi)(PetscScalar,PetscScalar[],PetscScalar,void*,PetscScalar*),
                                                            void *data);
PetscErrorCode FDPDEEnthalpySetEnergyPrimaryVariable(FDPDE,const char);
PetscErrorCode FDPDEEnthalpySetUserBC(FDPDE,PetscErrorCode(*form_user_bc)(DM,Vec,PetscScalar***,void*));

#endif