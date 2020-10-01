#include "petsc.h"
#include "../../src/fdpde_stokesdarcy2field.h"
#include "../../src/fdpde_advdiff.h"
#include "../../src/fdpde_composite.h"
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
  PetscScalar    Tp, cp, La, rho0, drho, alpha, beta, kappa, D;
  PetscScalar    phi0, n, K0, phi_max, eta0, zeta0, mu, eta_min, eta_max, lambda, EoR, Teta0; 
  PetscScalar    C0, DC, T0, Ms, Mf, gamma_inv, DT;
  PetscInt       ts_scheme, adv_scheme, tout, tstep;
  PetscScalar    tmax, dtmax;
  PetscInt       out_count;
  PetscBool      dim_out;
  char           fname_in[FNAME_LENGTH], fname_out[FNAME_LENGTH]; 
} Params;

typedef struct {
  PetscScalar    x, v, t, K, P, eta, rho, H, Gamma;
} ScalParams;

typedef struct {
  PetscScalar    L, H, xmin, zmin, xMOR, U0, eta0, zeta0;
  PetscScalar    tmax, dtmax, t, dt, tprev;
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
  DM             dmPV,dmHC;
  Vec            xPV,xT,xTheta,xphi,xC,xCf,xCs,xH,xTsol; // non-dimensional vectors
  Vec            xHprev, xCprev; 
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
PetscErrorCode FormCoefficient_H(FDPDE, DM, Vec, DM, Vec, void*);
PetscErrorCode FormCoefficient_C(FDPDE, DM, Vec, DM, Vec, void*);

// boundary conditions
PetscErrorCode FormBCList_PV(DM, Vec, DMStagBCList, void*);
PetscErrorCode FormBCList_H(DM, Vec, DMStagBCList, void*);
PetscErrorCode FormBCList_C(DM, Vec, DMStagBCList, void*);

// initial conditions
PetscErrorCode SetInitialConditions_HS(FDPDE, FDPDE, FDPDE, void*);
PetscErrorCode CornerFlow_MOR(void*);
PetscErrorCode HalfSpaceCooling_MOR(void*);
PetscErrorCode CorrectTemperatureForSolidus(void*);

// constitutive equations
PetscScalar Temp2Theta(PetscScalar,PetscScalar);
PetscScalar Theta2Temp(PetscScalar,PetscScalar);
PetscScalar BulkComposition(PetscScalar,PetscScalar,PetscScalar);  
PetscScalar Solidus(PetscScalar,PetscScalar,PetscScalar,PetscBool);
PetscScalar Liquidus(PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscBool);
PetscScalar LithostaticPressure(PetscScalar,PetscScalar,PetscScalar);
PetscScalar TotalEnthalpy(PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscScalar);

// utils
PetscErrorCode DoOutput(void*);
PetscErrorCode ScaleSolutionPV(DM,Vec,Vec*,void*);
PetscErrorCode ScaleVectorUniform(DM,Vec,Vec*,PetscScalar);
PetscErrorCode ScaleTemperatureComposition(DM,Vec,Vec*,void*,PetscInt);

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
