#ifndef MORFAULT_H
#define MORFAULT_H

#include "petsc.h"
#include "../../src/fdpde_stokesdarcy2field.h"
#include "../../src/fdpde_advdiff.h"
#include "../../src/consteq.h"
#include "../../src/dmstagoutput.h"
#include "../../src/material_point.h"

// ---------------------------------------
// Data structures
// ---------------------------------------
// general
#define FNAME_LENGTH  200
#define SEC_YEAR      31536000 //3600.00*24.00*365.00
#define T_KELVIN      273.15
#define PHI_CUTOFF    1e-12
#define MAX_MAT_PHASE 6

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

#define PV_ELEMENT_P   0
#define PV_FACE_VS     0
#define T_ELEMENT      0

#define PVCOEFF_VERTEX_A    0 
#define PVCOEFF_ELEMENT_C   0
#define PVCOEFF_ELEMENT_A   1 
#define PVCOEFF_ELEMENT_D1  2 
#define PVCOEFF_FACE_B      0 
#define PVCOEFF_FACE_D2     1 
#define PVCOEFF_FACE_D3     2 

// TCOEFF and PHICOEFF have same structure
#define TCOEFF_ELEMENT_A   0
#define TCOEFF_ELEMENT_C   1 
#define TCOEFF_FACE_B      0 
#define TCOEFF_FACE_u      1 

#define VEL_FACE_VF        0
#define VEL_FACE_V         1

#define MATPROP_ELEMENT_ETA     0
#define MATPROP_ELEMENT_ETA_V   1
#define MATPROP_ELEMENT_ETA_E   2
#define MATPROP_ELEMENT_ETA_P   3
#define MATPROP_ELEMENT_ZETA    4
#define MATPROP_ELEMENT_ZETA_V  5
#define MATPROP_ELEMENT_ZETA_E  6
#define MATPROP_ELEMENT_ZETA_P  7
#define MATPROP_ELEMENT_Z       8
#define MATPROP_ELEMENT_G       9
#define MATPROP_ELEMENT_C       10
#define MATPROP_ELEMENT_SIGMAT  11
#define MATPROP_ELEMENT_THETA   12
#define MATPROP_ELEMENT_RHO     13
#define MATPROP_ELEMENT_KPHI    14
#define MATPROP_ELEMENT_CHIP    15
#define MATPROP_ELEMENT_CHIS    16
#define MATPROP_NPROP           17

// ---------------------------------------
// Application Context
// ---------------------------------------
typedef struct {
  PetscInt       nx, nz;
  PetscScalar    L, H, Hs, xmin, zmin;
  PetscScalar    k_hat, g, Ttop, Tbot, R, Vext, uT, rhof, q, age, Gamma;
  PetscScalar    hs_factor, drho, kphi0, n, mu, eta_min, eta_max, phi_min, phi0, eta_K, Zmax, beta, EoR, Teta0, zetaExp;
  PetscInt       ts_scheme, adv_scheme, tout, tstep, ppcell, Nmax, rheology, two_phase, model_setup, model_setup_phi, restart, inflow_bc;
  PetscScalar    dt_out, tmax, dtmax, tf_tol, strain_max, hcc, phi_max_bc, sigma_bc, sigma_bc_h, z_bc;
  PetscInt       mat0_id, mat1_id, mat2_id, mat3_id, mat4_id, mat5_id, marker_phases, matid_default;
  PetscScalar    mat0_rho0, mat0_alpha, mat0_cp, mat0_kT, mat0_kappa; 
  PetscScalar    mat1_rho0, mat1_alpha, mat1_cp, mat1_kT, mat1_kappa; 
  PetscScalar    mat2_rho0, mat2_alpha, mat2_cp, mat2_kT, mat2_kappa; 
  PetscScalar    mat3_rho0, mat3_alpha, mat3_cp, mat3_kT, mat3_kappa; 
  PetscScalar    mat4_rho0, mat4_alpha, mat4_cp, mat4_kT, mat4_kappa; 
  PetscScalar    mat5_rho0, mat5_alpha, mat5_cp, mat5_kT, mat5_kappa; 
  PetscInt       mat0_rho_function, mat1_rho_function, mat2_rho_function, mat3_rho_function, mat4_rho_function, mat5_rho_function;
  PetscInt       mat0_eta_function, mat1_eta_function, mat2_eta_function, mat3_eta_function, mat4_eta_function, mat5_eta_function;
  PetscInt       mat0_zeta_function, mat1_zeta_function, mat2_zeta_function, mat3_zeta_function, mat4_zeta_function, mat5_zeta_function;
  PetscScalar    mat0_eta0, mat1_eta0, mat2_eta0, mat3_eta0, mat4_eta0, mat5_eta0;
  PetscScalar    mat0_zeta0, mat1_zeta0, mat2_zeta0, mat3_zeta0, mat4_zeta0, mat5_zeta0;
  PetscScalar    mat0_G, mat1_G, mat2_G, mat3_G, mat4_G, mat5_G, mat0_Z0, mat1_Z0, mat2_Z0, mat3_Z0, mat4_Z0, mat5_Z0; 
  PetscScalar    mat0_C, mat1_C, mat2_C, mat3_C, mat4_C, mat5_C;
  PetscScalar    mat0_sigmat, mat1_sigmat, mat2_sigmat, mat3_sigmat, mat4_sigmat, mat5_sigmat;
  PetscScalar    mat0_theta, mat1_theta, mat2_theta, mat3_theta, mat4_theta, mat5_theta; 
  char           mat0_name[FNAME_LENGTH],mat1_name[FNAME_LENGTH],mat2_name[FNAME_LENGTH], mat3_name[FNAME_LENGTH],mat4_name[FNAME_LENGTH],mat5_name[FNAME_LENGTH];
  char           fname_in[FNAME_LENGTH], fname_out[FNAME_LENGTH], fdir_out[FNAME_LENGTH]; 
  PetscBool      log_info, start_run;
} Params;

typedef struct {
  PetscScalar    x, v, t, tau, rho, eta, kappa, kT, kphi, DT, Gamma;
} ScalParams;

typedef struct {
  PetscScalar    L, H, Hs, xmin, zmin, Vext, Vin, uT, R, delta, eta_min, eta_max, eta_K, Zmax;
  PetscScalar    Tbot, Ttop, Ra, Gamma, rhof;
  PetscScalar    tmax, dtmax, t, dt, dt_out, dzin, dzin_fs, z_bc;
  PetscScalar    Vin_free, Vin_rock;
  PetscInt       istep;
} NdParams;

typedef struct { 
  PetscScalar   rho0, alpha, cp, kT, kappa;
  PetscScalar   eta0, zeta0, G, Z0, C, sigmat, theta;
  PetscInt      rho_func, eta_func, zeta_func;
} MaterialProp;

// user defined and model-dependent variables
typedef struct {
  Params        *par;
  NdParams      *nd;
  ScalParams    *scal;
  PetscInt      nph;
  MaterialProp  mat[MAX_MAT_PHASE],mat_nd[MAX_MAT_PHASE];
  PetscBag      bag;
  MPI_Comm      comm;
  PetscMPIInt   rank;
  PetscBool     plasticity;
  DM            dmphi, dmPV, dmT, dmswarm, dmVel, dmMPhase, dmPlith, dmeps, dmmatProp;
  Vec           xPV, xT, xphi, xVel, xMPhase, xPlith, xeps, xtau, xtau_old;
  Vec           xDP, xDP_old, xplast, xmatProp, xstrain;
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
PetscErrorCode FormCoefficient_PV_DPL(FDPDE, DM, Vec, DM, Vec, void*);
PetscErrorCode FormCoefficient_PV_Stokes_DPL(FDPDE, DM, Vec, DM, Vec, void*);
PetscErrorCode FormCoefficient_T(FDPDE, DM, Vec, DM, Vec, void*);
PetscErrorCode FormCoefficient_phi(FDPDE, DM, Vec, DM, Vec, void*);
PetscErrorCode RheologyPointwise(PetscInt,PetscInt,PetscScalar***,PetscInt*,PetscScalar,PetscScalar,PetscScalar*,PetscScalar*,PetscScalar*,PetscInt*,PetscScalar*,void*);
PetscErrorCode RheologyPointwise_VEP(PetscInt,PetscInt,PetscScalar***,PetscInt*,PetscScalar,PetscScalar,PetscScalar*,PetscScalar*,PetscScalar*,PetscInt*,PetscScalar*,void*);
PetscErrorCode RheologyPointwise_V(PetscInt,PetscInt,PetscScalar***,PetscInt*,PetscScalar,PetscScalar,PetscScalar*,PetscScalar*,PetscScalar*,PetscInt*,PetscScalar*,void*);
PetscErrorCode RheologyPointwise_VE(PetscInt,PetscInt,PetscScalar***,PetscInt*,PetscScalar,PetscScalar,PetscScalar*,PetscScalar*,PetscScalar*,PetscInt*,PetscScalar*,void*);
PetscErrorCode RheologyPointwise_DPL(PetscInt,PetscInt,PetscScalar***,PetscInt*,PetscScalar,PetscScalar,PetscScalar*,PetscScalar*,PetscScalar*,PetscInt*,PetscScalar*,void*);
PetscErrorCode DecompactRheologyVars_DPL(PetscInt,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,
                                     PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,
                                     PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*);

// boundary conditions
PetscErrorCode FormBCList_PV(DM, Vec, DMStagBCList, void*);
PetscErrorCode FormBCList_PV_Stokes(DM, Vec, DMStagBCList, void*);
PetscErrorCode FormBCList_T(DM, Vec, DMStagBCList, void*);
PetscErrorCode FormBCList_phi(DM, Vec, DMStagBCList, void*);

// constitutive equations
PetscScalar HalfSpaceCoolingTemp(PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscScalar);
PetscScalar LithostaticPressure(PetscScalar,PetscScalar);
PetscScalar Density(PetscScalar,PetscInt);
PetscScalar Permeability(PetscScalar,PetscScalar);
PetscScalar ShearViscosity(PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscInt); 
PetscScalar CompactionViscosity(PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscInt);
PetscScalar ArrheniusTerm_Viscosity(PetscScalar,PetscScalar,PetscScalar); 
PetscScalar LiquidVelocity(PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscScalar); 
PetscScalar Mixture(PetscScalar,PetscScalar,PetscScalar);
PetscScalar TensileStrength(PetscScalar,PetscScalar,PetscScalar,PetscInt); 
PetscScalar ElasticShearModulus(PetscScalar,PetscScalar); 
PetscScalar PoroElasticModulus(PetscScalar,PetscScalar,PetscScalar);
PetscScalar TensorSecondInvariant(PetscScalar,PetscScalar,PetscScalar);
PetscScalar ViscosityHarmonicAvg(PetscScalar,PetscScalar,PetscScalar); 
PetscErrorCode Plastic_LocalSolver(PetscScalar*,PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscScalar,void*,PetscScalar[]);

// utils
PetscErrorCode SetSwarmInitialCondition(DM,void*);
PetscErrorCode AddMarkerInflux(DM,void*);
PetscErrorCode AddMarkerInflux_FreeSurface(DM,void*);
PetscErrorCode SetInitialConditions(FDPDE, FDPDE,FDPDE,void*);
PetscErrorCode HalfSpaceCooling_MOR(void*);
PetscErrorCode SetInitialPorosityField(void*);
PetscErrorCode UpdateMarkerPhaseFractions(DM,DM,Vec,void*);
PetscErrorCode UpdateLithostaticPressure(DM,Vec,void*);
PetscErrorCode DoOutput(FDPDE,FDPDE,FDPDE,void*);
PetscErrorCode UpdateStrainRates_Array(DM,Vec,void*);
PetscErrorCode IntegratePlasticStrain(DM,Vec,Vec,void*);
PetscErrorCode ComputeFluidAndBulkVelocity(DM,Vec,DM,Vec,DM,Vec,DM,Vec,void*);
PetscErrorCode LiquidVelocityExplicitTimestep(DM,Vec,PetscScalar*);
PetscErrorCode CreateDirectory(const char*);
PetscErrorCode LoadRestartFromFile(FDPDE,FDPDE,FDPDE,void*);
PetscErrorCode OutputParameters(void*); 
PetscErrorCode LoadParametersFromFile(void*);
PetscErrorCode DMSwarmReadBinaryXDMF_Seq(DM,const char*,PetscInt,const char*[1]);
PetscErrorCode GetMarkerDensityPerCell(DM,DM,PetscInt[2]);
PetscErrorCode CorrectPorosityFreeSurface(DM,Vec,DM,Vec);
PetscErrorCode CorrectNegativePorosity(DM,Vec);
PetscErrorCode CheckNegativePorosity(DM,Vec,PetscBool*);

PetscErrorCode GetMatPhaseFraction(PetscInt,PetscInt,PetscScalar***,PetscInt*,PetscInt,PetscScalar*);
PetscErrorCode GetCornerAvgFromCenter(PetscScalar*,PetscScalar*);
PetscErrorCode Get9PointCenterValues(PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscScalar***,PetscScalar*);
PetscErrorCode GetTensorPointValues(PetscInt,PetscInt,PetscInt*,PetscScalar***,PetscScalar*);

// // ---------------------------------------
// // Useful functions
// // ---------------------------------------
static PetscScalar nd_param (PetscScalar x, PetscScalar scal) { return(x/scal);}
static PetscScalar dim_param(PetscScalar x, PetscScalar scal) { return(x*scal);}
static PetscScalar nd_paramT (PetscScalar x, PetscScalar x0, PetscScalar scal) { return((x-x0)/scal);}
static PetscScalar dim_paramT(PetscScalar x, PetscScalar x0, PetscScalar scal) { return(x*scal+x0);}
static PetscScalar WeightAverageValue(PetscScalar *a, PetscScalar *wt, PetscInt n) {
  PetscInt    i;
  PetscScalar awt = 0.0;
  for (i = 0; i <n; i++) { awt += a[i]*wt[i]; }
  return awt;
}

#endif