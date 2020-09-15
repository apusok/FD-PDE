#include "petsc.h"
#include "../../src/fdpde_stokesdarcy2field.h"
#include "../../src/fdpde_advdiff.h"
#include "../../src/dmstagoutput.h"

// ---------------------------------------
// Data structures
// ---------------------------------------
// general
#define FNAME_LENGTH  200
#define SEC_YEAR      31536000 //3600.00*24.00*365.00

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
  PetscScalar    L, H, xmin, zmin;
  PetscScalar    k_hat, g, U0;
  PetscScalar    Tp, cp, La, rho0, drho, alpha, beta, kappa, D;
  PetscScalar    phi0, n, K0, phi_max, eta0, zeta0, mu, eta_min, eta_max, lambda, EoR, Teta0; 
  PetscScalar    C0, DC, T0, Ms, Mf, gamma_inv, DT;
  char           fname_in[FNAME_LENGTH], fname_out[FNAME_LENGTH]; 
} Params;

typedef struct {
  PetscScalar    x, v, t, K, P, eta, rho, H, Gamma;
} ScalParams;

typedef struct {
  PetscScalar    L, H, xmin, zmin, U0, eta0, zeta0;
  PetscScalar    delta, alpha_s, beta_s, A, S, PeT, PeC, theta_s, G, RM;
} NdParams;

// user defined and model-dependent variables
typedef struct {
  Params        *par;
  NdParams      *nd;
  ScalParams    *scal;
  PetscBag       bag;
  MPI_Comm       comm;
  PetscMPIInt    rank;
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

// ---------------------------------------
// Useful functions
// ---------------------------------------
static PetscScalar nd_param (PetscScalar x, PetscScalar scal) { return(x/scal);}
static PetscScalar dim_param(PetscScalar x, PetscScalar scal) { return(x*scal);}