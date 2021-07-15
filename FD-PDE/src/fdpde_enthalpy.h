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
// Error checking
// ---------------------------------------
typedef enum {
  STATE_VALID                 =  0,
  PHI_STATE_INVALID           = -1,
  ERR_PHI_DIVIDE_BY_ZERO      = -2,
  ERR_SOLID_PHI_DIVIDE_BY_ZERO= -3,
  ERR_DIVIDE_BY_ZERO          = -4,
  ERR_INF_NAN_VALUE           = -5,
  DIM_T_KELVIN_STATE_INVALID  = -6,
  DIM_T_CELSIUS_STATE_INVALID = -7,
  DIM_STATE_INVALID           = -8,
  DIM_C_STATE_INVALID         = -9,
  DIM_CF_STATE_INVALID        = -10,
  DIM_CS_STATE_INVALID        = -11,
  STATE_INVALID_IERR          = -12,
  STATE_INVALID               = -100
} EnthEvalErrorCode;

#define ENTH_CHECK_PHI(phi) \
  if (phi < 0.0) return(PHI_STATE_INVALID); \
  if (phi > 1.0) return(PHI_STATE_INVALID); \

#define ENTH_CHECK_PHI_DIVIDE_BY_ZERO(phi) \
  if (phi < 1.0e-12) return(ERR_PHI_DIVIDE_BY_ZERO); \

#define ENTH_CHECK_SOLID_PHI_DIVIDE_BY_ZERO(phi) \
  if (1.0 - phi < 1.0e-12) return(ERR_SOLID_PHI_DIVIDE_BY_ZERO); \

#define ENTH_CHECK_DIVIDE_BY_ZERO(Q) \
  if (Q < 1.0e-12) return(ERR_DIVIDE_BY_ZERO); \

#define ENTH_CHECK_INF_NAN(Q) \
  if (PetscIsInfOrNan(Q)) return(ERR_INF_NAN_VALUE); \

#define ENTH_CHECK_DIM_T_KELVIN(T,T0,DT) \
  if ((T*DT+T0)<0) return(DIM_T_KELVIN_STATE_INVALID); \

#define ENTH_CHECK_DIM_T_CELSIUS(T,T0,DT) \
  if ((T*DT+T0)+ 273.15 <0) return(DIM_T_CELSIUS_STATE_INVALID); \

#define ENTH_CHECK_DIM_VALUE(Q,scal) \
  if ((Q*scal)<0) return(DIM_STATE_INVALID); \

#define ENTH_CHECK_DIM_C_VALUE(C,C0,DC,N) \
  { int i; \
    for (i=0; i<N; i++) { \
      if (C[i]*DC+C0 < 0) return(DIM_C_STATE_INVALID); \
    } \
  } \

#define ENTH_CHECK_DIM_CF_VALUE(CF,C0,DC,N) \
  { int i; \
    for (i=0; i<N; i++) { \
      if (CF[i]*DC+C0 < 0) return(DIM_CF_STATE_INVALID); \
    } \
  } \

#define ENTH_CHECK_DIM_CS_VALUE(CS,C0,DC,N) \
  { int i; \
    for (i=0; i<N; i++) { \
      if (CS[i]*DC+C0 < 0) return(DIM_CS_STATE_INVALID); \
    } \
  } \

#define ENTH_CHECK_IERR(ierr) if ((ierr) < 0) return(STATE_INVALID_IERR);

// ---------------------------------------
// Struct definitions
// ---------------------------------------
typedef struct {
  PetscScalar  H,T,TP,P,phi;
  PetscScalar  C[MAX_COMPONENTS],CF[MAX_COMPONENTS],CS[MAX_COMPONENTS];
  EnthEvalErrorCode err;
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
  PetscErrorCode    (*form_user_bc)(DM,Vec,PetscScalar***,void*); // internal BC
  EnthEvalErrorCode (*form_enthalpy_method)(PetscScalar,PetscScalar[],PetscScalar,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscInt,void*);
  PetscErrorCode    (*form_TP)(PetscScalar,PetscScalar,PetscScalar*,void*);
  void               *user_context;
  void               *user_context_bc; // internal BC
  void               *user_context_tp;
  PetscInt           ncomponents, nreports;
  char               *description_enthalpy;
} EnthalpyData;

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
PetscErrorCode DMStagBCListApply_Enthalpy(DM,Vec,DMStagBC*,PetscInt,PetscScalar***);

PetscErrorCode ApplyEnthalpyMethod(FDPDE,DM,Vec,DM,Vec,EnthalpyData*,ThermoState*,const char[]);
PetscErrorCode UpdateCoeffStructure(FDPDE,DM,Vec,CoeffState*);
PetscErrorCode ApplyEnthalpyReport_Failure(FDPDE,PetscViewer,EnthalpyData*,ThermoState*);
// PetscErrorCode CoeffCellData(DM,Vec,PetscInt,PetscInt,CoeffState*);
// PetscErrorCode SolutionCellData(DM,Vec,PetscInt,PetscInt,PetscScalar*,PetscScalar*);
// PetscErrorCode EnthalpyResidual(DM,ThermoState*,CoeffState*,ThermoState*,CoeffState*,PetscScalar**,PetscScalar**,EnthalpyData*,PetscInt,PetscInt, PetscScalar*);
// PetscErrorCode EnthalpySteadyStateOperator(DM,ThermoState*,CoeffState*,PetscScalar**,PetscScalar**,PetscInt,PetscInt,AdvectSchemeType,PetscScalar*);
// PetscErrorCode BulkCompositionResidual(DM dm,ThermoState*,CoeffState*,ThermoState*,CoeffState*,PetscScalar**,PetscScalar**,EnthalpyData*,PetscInt,PetscInt,PetscInt,PetscScalar*);
// PetscErrorCode BulkCompositionSteadyStateOperator(DM,ThermoState*,CoeffState*,PetscScalar**,PetscScalar**,PetscInt,PetscInt,PetscInt,AdvectSchemeType,PetscScalar*);

// Set/Get Functions
PetscErrorCode FDPDEEnthalpySetAdvectSchemeType(FDPDE,AdvectSchemeType);
PetscErrorCode FDPDEEnthalpySetTimeStepSchemeType(FDPDE,TimeStepSchemeType);

PetscErrorCode FDPDEEnthalpySetTimestep(FDPDE,PetscScalar);
PetscErrorCode FDPDEEnthalpyGetTimestep(FDPDE, PetscScalar*);
PetscErrorCode FDPDEEnthalpyComputeExplicitTimestep(FDPDE, PetscScalar*);

PetscErrorCode FDPDEEnthalpyGetPrevSolution(FDPDE,Vec*);
PetscErrorCode FDPDEEnthalpyGetPrevCoefficient(FDPDE,Vec*);
PetscErrorCode FDPDEEnthalpyGetPressure(FDPDE,DM*,Vec*);
PetscErrorCode FDPDEEnthalpyGetPrevPressure(FDPDE,Vec*);

PetscErrorCode FDPDEEnthalpySetNumberComponentsPhaseDiagram(FDPDE,PetscInt);
PetscErrorCode FDPDEEnthalpySetEnthalpyMethod(FDPDE,EnthEvalErrorCode(*form_enthalpy_method)(PetscScalar,PetscScalar[],PetscScalar,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscInt,void*),const char[],void*);
PetscErrorCode FDPDEEnthalpySetPotentialTemp(FDPDE,PetscErrorCode(*form_TP)(PetscScalar,PetscScalar,PetscScalar*,void*),void*);
PetscErrorCode FDPDEEnthalpySetUserBC(FDPDE,PetscErrorCode(*form_user_bc)(DM,Vec,PetscScalar***,void*),void*); // internal BC
PetscErrorCode FDPDEEnthalpyUpdateDiagnostics(FDPDE,DM,Vec,DM*,Vec*);

#endif