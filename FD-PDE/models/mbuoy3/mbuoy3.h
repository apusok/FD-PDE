#ifndef MBUOY3_H
#define MBUOY3_H

#include "petsc.h"
#include "../../src/fdpde_stokesdarcy3field.h"
#include "../../src/fdpde_enthalpy.h"
#include "../../src/fdpde_advdiff.h"
#include "../../src/dmstagoutput.h"
#include "../../src/benchmark_cornerflow.h"

// ---------------------------------------
// Data structures
// ---------------------------------------
// general
#define FNAME_LENGTH  200
#define SEC_YEAR      31536000 //3600.00*24.00*365.00
#define T_KELVIN      273.15
#define PHI_CUTOFF    1e-12

// define convenient names for DMStagStencilLocation
#define DOWN_LEFT  DMSTAG_DOWN_LEFT
#define DOWN       DMSTAG_DOWN
#define DOWN_RIGHT DMSTAG_DOWN_RIGHT
#define LEFT       DMSTAG_LEFT
#define ELEMENT    DMSTAG_ELEMENT
#define RIGHT      DMSTAG_RIGHT
#define UP_LEFT    DMSTAG_UP_LEFT
#define UP         DMSTAG_UP
#define UP_RIGHT   DMSTAG_UP_RIGHT

#define HC_ELEMENT_H  0
#define HC_ELEMENT_C  1

#define PV_ELEMENT_P   0
#define PV_ELEMENT_PC  1
#define PV_FACE_VS     0

#define PVCOEFF_VERTEX_A    0 
#define PVCOEFF_ELEMENT_C   0
#define PVCOEFF_ELEMENT_A   1 
#define PVCOEFF_ELEMENT_D1  2 
#define PVCOEFF_ELEMENT_DC  3 
#define PVCOEFF_FACE_B   0 
#define PVCOEFF_FACE_D2  1 
#define PVCOEFF_FACE_D3  2 
#define PVCOEFF_FACE_D4  3 

#define HCCOEFF_ELEMENT_A1  0
#define HCCOEFF_ELEMENT_B1  1
#define HCCOEFF_ELEMENT_D1  2
#define HCCOEFF_ELEMENT_A2  3
#define HCCOEFF_ELEMENT_B2  4
#define HCCOEFF_ELEMENT_D2  5
#define HCCOEFF_FACE_C1  0
#define HCCOEFF_FACE_C2  1
#define HCCOEFF_FACE_V   2
#define HCCOEFF_FACE_VF  3
#define HCCOEFF_FACE_VS  4

#define ENTH_ELEMENT_H   0
#define ENTH_ELEMENT_T   1
#define ENTH_ELEMENT_TP  2
#define ENTH_ELEMENT_PHI 3
#define ENTH_ELEMENT_P   4
#define ENTH_ELEMENT_C   5
#define ENTH_ELEMENT_CS  7
#define ENTH_ELEMENT_CF  9

#define VEL_FACE_VF     0
#define VEL_FACE_V      1

// ---------------------------------------
// Application Context
// ---------------------------------------
typedef struct {
  PetscInt       nx, nz;
  PetscScalar    L, H, xmin, zmin, xmor;
  PetscScalar    k_hat, g, U0;
  PetscScalar    Tp, Ts, cp, La, rho0, drho, alpha, beta, kappa, D;
  PetscScalar    n, K0, phi_max, eta0, zeta0, mu, eta_min, eta_max, lambda, EoR, Teta0, zetaExp; 
  PetscScalar    C0, DC, T0, Ms, Mf, gamma_inv, DT, phi_init, phi_min, fextract;
  PetscInt       ts_scheme, adv_scheme, tout, tstep, restart, full_ridge;
  PetscScalar    tmax, dtmax, dt_out;
  PetscInt       visc_shear, visc_bulk, buoyancy, buoy_phi, buoy_C, buoy_T, extract_mech, initial_bulk_comp, hc_cycles, vf_nonlinear;
  char           fname_in[FNAME_LENGTH], fname_out[FNAME_LENGTH], fdir_out[FNAME_LENGTH]; 
  PetscBool      start_run, log_info;
} Params;

typedef struct {
  PetscScalar    x, v, t, K, P, eta, rho, H, Gamma;
} ScalParams;

typedef struct {
  PetscScalar    L, H, xmin, zmin, xmor, U0, visc_ratio, eta_min, eta_max;
  PetscScalar    tmax, dtmax, t, dt, dt_out;
  PetscInt       istep;
  PetscScalar    delta, alpha_s, beta_s, alpha_ls, beta_ls, A, S, PeT, PeC, thetaS, G, RM;
} NdParams;

// user defined and model-dependent variables
typedef struct {
  Params        *par;
  NdParams      *nd;
  ScalParams    *scal;
  PetscBag      bag;
  MPI_Comm      comm;
  PetscMPIInt   rank;
  DM            dmPV, dmHC, dmVel, dmEnth, dmmatProp;
  Vec           xPV, xHC, xVel, xEnth, xEnthold, xmatProp;
} UsrData;

// ---------------------------------------
// Function Definitions
// ---------------------------------------
// input
PetscErrorCode UserParamsCreate(UsrData**,int,char**);
PetscErrorCode UserParamsDestroy(UsrData*);
PetscErrorCode InputParameters(UsrData**);
PetscErrorCode InputPrintData(UsrData*);
PetscErrorCode DefineScalingParameters(UsrData*);
PetscErrorCode NondimensionalizeParameters(UsrData*);

// physics
PetscErrorCode Numerical_solution(void*);
PetscErrorCode FormCoefficient_PV(FDPDE, DM, Vec, DM, Vec, void*);
PetscErrorCode FormCoefficient_HC(FDPDE, DM, Vec, DM, Vec, void*);
PetscErrorCode FormCoefficient_HC_VF_nonlinear(FDPDE, DM, Vec, DM, Vec, void*);
EnthEvalErrorCode Form_Enthalpy(PetscScalar,PetscScalar[],PetscScalar,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscInt,void*); 
PetscErrorCode Form_PotentialTemperature(PetscScalar,PetscScalar,PetscScalar*,void*); 

// boundary conditions
PetscErrorCode FormBCList_PV(DM, Vec, DMStagBCList, void*);
PetscErrorCode FormBCList_HC(DM, Vec, DMStagBCList, void*);
PetscErrorCode FormBCList_PV_FullRidge(DM, Vec, DMStagBCList, void*);
PetscErrorCode FormBCList_HC_FullRidge(DM, Vec, DMStagBCList, void*);

// constitutive equations
PetscErrorCode Porosity(PetscScalar,PetscScalar,PetscScalar,PetscScalar*,PetscScalar,PetscScalar,PetscScalar);
PetscScalar Solidus(PetscScalar,PetscScalar,PetscScalar,PetscBool);
PetscScalar Liquidus(PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscBool);
PetscScalar LithostaticPressure(PetscScalar,PetscScalar,PetscScalar);
PetscScalar TotalEnthalpy(PetscScalar,PetscScalar,PetscScalar);
PetscScalar PhiRes(PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscScalar);
PetscScalar FluidVelocity(PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscScalar); 
PetscScalar BulkVelocity(PetscScalar,PetscScalar,PetscScalar);
PetscScalar Permeability(PetscScalar,PetscScalar,PetscScalar);
PetscScalar FluidBuoyancy(PetscScalar,PetscScalar,PetscScalar,PetscScalar);
PetscScalar HalfSpaceCoolingTemp(PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscScalar);
PetscScalar SolidDensity(PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscInt);
PetscScalar FluidDensity(PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscInt); 
PetscScalar BulkDensity(PetscScalar,PetscScalar,PetscScalar); 

PetscScalar Buoyancy_phi(PetscScalar,PetscInt);
PetscScalar Buoyancy_Composition(PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscInt);
PetscScalar Buoyancy_Temperature(PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscInt);   

PetscScalar ShearViscosity(PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscInt);  
PetscScalar BulkViscosity(PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscInt);
PetscScalar ArrheniusTerm_Viscosity(PetscScalar,PetscScalar,PetscScalar); 

// utils
PetscErrorCode SetInitialConditions(FDPDE, FDPDE, void*);
PetscErrorCode CornerFlow_MOR(void*);
PetscErrorCode HalfSpaceCooling_MOR(void*);
PetscErrorCode UpdateLithostaticPressure(DM,Vec,void*);
PetscErrorCode CorrectInitialHCZeroPorosity(DM,Vec,void*);
PetscErrorCode CorrectInitialHCBulkComposition(void*);
PetscErrorCode ComputeFluidAndBulkVelocity(DM,Vec,DM,Vec,DM,Vec,void*);
PetscErrorCode UpdateMaterialProperties(DM,Vec,DM,Vec,void*);

PetscErrorCode DoOutput(FDPDE,FDPDE,void*);
PetscErrorCode LoadRestartFromFile(FDPDE, FDPDE, void*);
PetscErrorCode CreateDirectory(const char*);
PetscErrorCode OutputParameters(void*); 
PetscErrorCode LoadParametersFromFile(void*);
PetscErrorCode ComputeMeltExtractOutflux(void*); 
PetscErrorCode ComputeAsymmetryFullRidge(void*); 
PetscErrorCode ComputeGamma(DM,Vec,DM,Vec,DM,Vec,Vec,void*); 

// ---------------------------------------
// Useful functions
// ---------------------------------------
static PetscScalar nd_param (PetscScalar x, PetscScalar scal) { return(x/scal);}
static PetscScalar dim_param(PetscScalar x, PetscScalar scal) { return(x*scal);}

#endif