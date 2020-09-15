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
  PetscScalar    C0, DC, T0, Ms, Mf, gamma_inv;
  char           fname_in[FNAME_LENGTH], fname_out[FNAME_LENGTH]; 
} Params;

// typedef struct {
//   PetscInt       nx,nz,n,ts_scheme,adv_scheme,tout,tstep;
//   PetscScalar    L,H,xmin,zmin,k_hat,u0,phi0,eta,zeta,xMOR,Gamma,xi,tmax,dtmax;
//   PetscScalar    t,dt,tprev;
// } NDParams;

// typedef struct {
//   PetscScalar   h, v, t, K, P, eta, Gamma, rho, delta;
// } ScalParams;

// user defined and model-dependent variables
typedef struct {
  Params        *par;
  // NDParams      *nd;
  // ScalParams    *scal;
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