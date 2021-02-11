#include "petsc.h"
#include "../../src/fdpde_stokesdarcy2field.h"
#include "../../src/fdpde_enthalpy.h"
#include "../../src/dmstagoutput.h"
#include "../../src/benchmark_cornerflow.h"

// ---------------------------------------
// Data structures
// ---------------------------------------
// general
#define FNAME_LENGTH  200
#define SEC_YEAR      31536000 //3600.00*24.00*365.00
#define T_KELVIN      273.15

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

// ---------------------------------------
// Application Context
// ---------------------------------------
typedef struct {
  PetscInt       nx, nz;
  PetscScalar    L, H, xmin, zmin, xMOR;
  PetscScalar    k_hat, g, U0;
  PetscScalar    Tp, Ts, cp, La, rho0, drho, alpha, beta, kappa, D;
  PetscScalar    phi0, n, K0, phi_max, eta0, zeta0, mu, eta_min, eta_max, lambda, EoR, Teta0, zetaExp; 
  PetscScalar    C0, DC, T0, Ms, Mf, gamma_inv, DT;
  PetscInt       ts_scheme, adv_scheme, tout, tstep, istep;
  PetscScalar    tmax, dtmax;
  PetscInt       out_count, buoyancy;
  PetscBool      dim_out;
  char           fname_in[FNAME_LENGTH], fname_out[FNAME_LENGTH], fdir_out[FNAME_LENGTH]; 
} Params;

typedef struct {
  PetscScalar    x, v, t, K, P, eta, rho, H, Gamma;
} ScalParams;

typedef struct {
  PetscScalar    L, H, xmin, zmin, xMOR, U0, visc_ratio;
  PetscScalar    tmax, dtmax, t, dt;
  PetscScalar    delta, alpha_s, beta_s, A, S, PeT, PeC, thetaS, G, RM;
} NdParams;

// user defined and model-dependent variables
typedef struct {
  Params        *par;
  NdParams      *nd;
  ScalParams    *scal;
  PetscBag       bag;
  MPI_Comm       comm;
  PetscMPIInt    rank;
  DM             dmPV, dmHC, dmVel, dmEnth;
  Vec            xPV, xHC, xVel, xphiT, xEnth;
} UsrData;

// ---------------------------------------
// Function Definitions
// ---------------------------------------
// input parameters
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
EnthEvalErrorCode Form_Enthalpy(PetscScalar,PetscScalar[],PetscScalar,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscInt,void*); 
PetscErrorCode Form_PotentialTemperature(PetscScalar,PetscScalar,PetscScalar*,void*); 

// boundary conditions
PetscErrorCode FormBCList_PV(DM, Vec, DMStagBCList, void*);
PetscErrorCode FormBCList_HC(DM, Vec, DMStagBCList, void*);

// initial conditions
PetscErrorCode SetInitialConditions(FDPDE, FDPDE, void*);
PetscErrorCode CornerFlow_MOR(void*);
PetscErrorCode HalfSpaceCooling_MOR(void*);
PetscErrorCode UpdateLithostaticPressure(DM,Vec,void*);
PetscErrorCode CorrectInitialHCZeroPorosity(DM,Vec,void*);
PetscErrorCode ExtractTemperaturePorosity(DM,Vec,void*,PetscBool);
PetscErrorCode ComputeFluidAndBulkVelocity(DM,Vec,DM,Vec,DM,Vec,void*);

// constitutive equations
PetscErrorCode Porosity(PetscScalar,PetscScalar,PetscScalar,PetscScalar*,PetscScalar,PetscScalar,PetscScalar);
PetscScalar Solidus(PetscScalar,PetscScalar,PetscScalar,PetscBool);
PetscScalar Liquidus(PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscBool);
PetscScalar LithostaticPressure(PetscScalar,PetscScalar,PetscScalar);
PetscScalar TotalEnthalpy(PetscScalar,PetscScalar,PetscScalar);
PetscScalar PhiRes(PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscScalar);
PetscScalar FluidVelocity(PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscScalar); 
PetscScalar BulkVelocity(PetscScalar,PetscScalar,PetscScalar);
PetscScalar Permeability(PetscScalar,PetscScalar,PetscScalar,PetscScalar);
PetscScalar FluidBuoyancy(PetscScalar,PetscScalar,PetscScalar,PetscScalar);
PetscScalar HalfSpaceCoolingTemp(PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscScalar);
PetscScalar ShearViscosity(PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscScalar);  
PetscScalar BulkViscosity(PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscScalar); 
PetscScalar Buoyancy(PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscInt); 

// utils
PetscErrorCode DoOutput(FDPDE,FDPDE,void*);
// PetscErrorCode ScaleSolutionPV(DM,Vec,Vec*,void*);
// PetscErrorCode ScaleVectorUniform(DM,Vec,Vec*,PetscScalar);
// PetscErrorCode ScaleTemperatureComposition(DM,Vec,Vec*,void*,PetscInt);
PetscErrorCode CreateDirectory(const char*);

// ---------------------------------------
// Useful functions
// ---------------------------------------
static PetscScalar nd_param (PetscScalar x, PetscScalar scal) { return(x/scal);}
static PetscScalar dim_param(PetscScalar x, PetscScalar scal) { return(x*scal);}

// interpolate/extrapolate 1D linear (need to provide 3 points x0,x1,x2 with values Q0, Q1, Q2)
static PetscScalar interp1DLin_3Points(PetscScalar xi, PetscScalar x0, PetscScalar Q0, PetscScalar x1, PetscScalar Q1, PetscScalar x2, PetscScalar Q2)
  { PetscScalar a,b,result, tol = 1e-10;

  if (xi < x1) { // first half
    if ((x1-x0)*(x1-x0) < tol ) { a = (Q2-Q1)/(x2-x1); b = Q1 - a*x1; } // extrp
    else                        { a = (Q1-Q0)/(x1-x0); b = Q0 - a*x0; } // intrp
  }
  if (xi >= x1) { // second half
    if ((x2-x1)*(x2-x1) < tol ) { a = (Q1-Q0)/(x1-x0); b = Q0 - a*x0; }
    else                        { a = (Q2-Q1)/(x2-x1); b = Q1 - a*x1; }
  }
  result = a*xi+b;
  return(result);
}
