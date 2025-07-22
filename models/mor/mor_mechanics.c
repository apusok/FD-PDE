// ---------------------------------------
// Mid-ocean ridge model solving for mechanics (conservation of mass and momentum)
// Solves for coupled (P, v) and Q=(1-phi) evolution, where P-dynamic pressure, v-solid velocity, phi-porosity.
// run: ./mor_mechanics -pc_type lu -pc_factor_mat_solver_type umfpack -pc_factor_mat_ordering_type external -nx 20 -nz 20 -snes_monitor 
// python output: ./python/mor_mechanics_plot_X.py
// ---------------------------------------
static char help[] = "Mid-ocean ridge model solving for mechanics\n\n";

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

#include <petsc.h>
#include "../../src/fdpde_stokesdarcy2field.h"
#include "../../src/fdpde_advdiff.h"

// ---------------------------------------
// Application Context
// ---------------------------------------
#define FNAME_LENGTH 200
#define SEC_YEAR     31536000 //3600.00*24.00*365.00

// parameters (bag)
typedef struct {
  PetscInt       nx, nz;
  PetscScalar    L, H;
  PetscScalar    xmin, zmin;
  PetscScalar    k_hat, g, phi0, K0, u0, drho, rho0, n, mu, eta, zeta, xMOR;
  PetscInt       ts_scheme, adv_scheme, tout, tstep;
  PetscScalar    tmax, dtmax;
  char           fname_out[FNAME_LENGTH]; 
} Params;

typedef struct {
  PetscInt       nx,nz,n,ts_scheme,adv_scheme,tout,tstep;
  PetscScalar    L,H,xmin,zmin,k_hat,u0,phi0,eta,zeta,xMOR,Gamma,xi,tmax,dtmax;
  PetscScalar    t,dt,tprev;
} NDParams;

typedef struct {
  PetscScalar   h, v, t, K, P, eta, Gamma, rho, delta;
} ScalParams;

// user defined and model-dependent variables
typedef struct {
  Params        *par;
  NDParams      *nd;
  ScalParams    *scal;
  PetscBag       bag;
  MPI_Comm       comm;
  PetscMPIInt    rank;
  Vec            xphiprev,xPV,xscal;
  DM             dmphi,dmPV,dmscal;
} UsrData;

// ---------------------------------------
// Function definitions
// ---------------------------------------
PetscErrorCode Numerical_solution(void*);
PetscErrorCode InputParameters(UsrData**);
PetscErrorCode InputPrintData(UsrData*);
PetscErrorCode DefineScalingParameters(UsrData*);
PetscErrorCode NondimensionalizeParameters(UsrData*);
PetscErrorCode FormCoefficient_PV(FDPDE, DM, Vec, DM, Vec, void*);
PetscErrorCode FormCoefficient_phi(FDPDE, DM, Vec, DM, Vec, void*);
PetscErrorCode FormBCList_PV(DM, Vec, DMStagBCList, void*);
PetscErrorCode FormBCList_phi(DM, Vec, DMStagBCList, void*);
PetscErrorCode SetInitialPorosityProfile(DM,Vec,void*);
PetscErrorCode SetInitialPorosityCoefficient(DM,Vec,void*);
PetscErrorCode ComputeFluidVelocity(DM,Vec,DM,Vec,Vec*,void*);
PetscErrorCode ScaleParametersAndOutput(DM,Vec,DM,Vec,Vec,void*);

const char coeff_description_stokesdarcy[] =
"  << Stokes-Darcy Coefficients >> \n"
"  A = delta^2*eta  \n"
"  B = -phi*k_hat\n" 
"  C = 0 \n"
"  D1 = delta^2*xi, xi=zeta-2/3eta \n"
"  D2 = -Kphi \n"
"  D3 = Kphi*k_hat, Kphi = (phi/phi0)^n \n";

const char bc_description_stokesdarcy[] =
"  << Stokes-Darcy BCs >> \n"
"  LEFT, RIGHT, DOWN, UP: \n";

const char coeff_description_phi[] =
"  << Porosity Coefficients (dimensionless) >> \n"
"  A = 1.0 \n"
"  B = 0 \n"
"  C = Gamma \n"
"  u = [ux, uz] - StokesDarcy solid velocity \n";

const char bc_description_phi[] =
"  << Porosity BCs >> \n"
"  LEFT, RIGHT, DOWN, UP: \n";

// ---------------------------------------
// Useful functions
// ---------------------------------------
static PetscScalar nd_param (PetscScalar x, PetscScalar scal) { return(x/scal);}
static PetscScalar dim_param(PetscScalar x, PetscScalar scal) { return(x*scal);}

// fluid velocity
static PetscScalar fluid_velocity(PetscScalar vsolid, PetscScalar phi, PetscScalar phi0, PetscScalar n, PetscScalar gradP, PetscScalar i_hat)
  { PetscScalar result;
  result = vsolid - PetscPowScalar(phi/phi0,n)/phi*(gradP - i_hat);
  return(result);
}

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

// ---------------------------------------
// Application functions
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "Numerical_solution"
PetscErrorCode Numerical_solution(void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  NDParams      *nd;
  FDPDE          fdPV,fdphi;
  DM             dmPV, dmphi, dmphicoeff;
  Vec            xPV,xphi,xmms_PV,xmms_phi,xphiprev,phicoeffprev,phicoeff,xF;
  PetscInt       nx, nz, istep = 0;
  PetscScalar    xmin, zmin, xmax, zmax;//, dt;
  char           fout[FNAME_LENGTH];
  PetscFunctionBeginUser;

  nd = usr->nd;

  // Element count
  nx = nd->nx;
  nz = nd->nz;

  // Domain coords
  xmin = nd->xmin;
  zmin = nd->zmin;
  xmax = nd->xmin+nd->L;
  zmax = nd->zmin+nd->H;

  // Set up Stokes-Darcy system
  PetscPrintf(PETSC_COMM_WORLD,"# Set FD-PDE StokesDarcy2Field\n");
  PetscCall(FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_STOKESDARCY2FIELD,&fdPV));
  PetscCall(FDPDESetUp(fdPV));
  PetscCall(FDPDESetFunctionBCList(fdPV,FormBCList_PV,bc_description_stokesdarcy,usr)); 
  PetscCall(FDPDESetFunctionCoefficient(fdPV,FormCoefficient_PV,coeff_description_stokesdarcy,usr)); 
  PetscCall(SNESSetFromOptions(fdPV->snes)); 
  PetscCall(SNESSetOptionsPrefix(fdPV->snes,"pv_")); 

  // Set up Porosity (Advection-diffusion) system
  PetscPrintf(PETSC_COMM_WORLD,"# Set FD-PDE AdvDiff (porosity)\n");
  PetscCall(FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_ADVDIFF,&fdphi));
  PetscCall(FDPDESetUp(fdphi));

  if (nd->adv_scheme==0) { PetscCall(FDPDEAdvDiffSetAdvectSchemeType(fdphi,ADV_UPWIND)); }
  if (nd->adv_scheme==1) { PetscCall(FDPDEAdvDiffSetAdvectSchemeType(fdphi,ADV_FROMM)); }
  
  if (nd->ts_scheme ==  0) { PetscCall(FDPDEAdvDiffSetTimeStepSchemeType(fdphi,TS_FORWARD_EULER)); }
  if (nd->ts_scheme ==  1) { PetscCall(FDPDEAdvDiffSetTimeStepSchemeType(fdphi,TS_BACKWARD_EULER)); }
  if (nd->ts_scheme ==  2) { PetscCall(FDPDEAdvDiffSetTimeStepSchemeType(fdphi,TS_CRANK_NICHOLSON ));}

  PetscCall(FDPDESetFunctionBCList(fdphi,FormBCList_phi,bc_description_phi,usr)); 
  PetscCall(FDPDESetFunctionCoefficient(fdphi,FormCoefficient_phi,coeff_description_phi,usr)); 
  PetscCall(SNESSetFromOptions(fdphi->snes)); 
  PetscCall(SNESSetOptionsPrefix(fdphi->snes,"phi_")); 

  // Prepare usr data for coupling
  PetscCall(FDPDEGetDM(fdPV,&dmPV)); 
  PetscCall(FDPDEGetDM(fdphi,&dmphi)); 
  usr->dmPV  = dmPV;
  usr->dmphi = dmphi;

  PetscCall(FDPDEGetSolution(fdPV,&xPV));
  PetscCall(FDPDEGetSolution(fdphi,&xphi));
  PetscCall(VecDuplicate(xphi,&usr->xphiprev));
  PetscCall(VecDuplicate(xPV,&usr->xPV));
  PetscCall(VecDestroy(&xphi));
  PetscCall(VecDestroy(&xPV));

  // Create a new DM and vector for scaled variable fields
  PetscCall(DMStagCreateCompatibleDMStag(usr->dmPV,0,2,3,0,&usr->dmscal)); 
  PetscCall(DMSetUp(usr->dmscal)); 
  PetscCall(DMStagSetUniformCoordinatesProduct(usr->dmscal,dim_param(xmin,usr->scal->h),dim_param(xmax,usr->scal->h),dim_param(zmin,usr->scal->h),dim_param(zmax,usr->scal->h),0.0,0.0));
  PetscCall(DMCreateGlobalVector(usr->dmscal,&usr->xscal));

  // Set initial porosity profile (t=0)
  PetscPrintf(PETSC_COMM_WORLD,"# Set initial porosity profile\n");
  PetscCall(FDPDEAdvDiffGetPrevSolution(fdphi,&xphiprev));
  PetscCall(SetInitialPorosityProfile(dmphi,xphiprev,usr));

  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s_phi_initial",usr->par->fname_out));
  PetscCall(DMStagViewBinaryPython(dmphi,xphiprev,fout));

  PetscCall(VecCopy(xphiprev,usr->xphiprev));
  PetscCall(VecDestroy(&xphiprev));

  // Solve StokesDarcy - to calculate velocities
  PetscPrintf(PETSC_COMM_WORLD,"# Set initial porosity coefficient (1) - Stokes Solve \n");
  PetscCall(FDPDESolve(fdPV,NULL));
  PetscCall(FDPDEGetSolution(fdPV,&xPV));

  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s_PV_initial",usr->par->fname_out));
  PetscCall(DMStagViewBinaryPython(dmPV,xPV,fout));

  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s_PV_initial_residual",usr->par->fname_out));
  PetscCall(DMStagViewBinaryPython(dmPV,fdPV->r,fout));

  PetscCall(VecCopy(xPV,usr->xPV));
  PetscCall(VecDestroy(&xPV));

  // Set initial coefficient structure
  PetscPrintf(PETSC_COMM_WORLD,"# Set initial porosity coefficient (2) \n");
  PetscCall(FDPDEGetCoefficient(fdphi,&dmphicoeff,NULL));
  PetscCall(FDPDEAdvDiffGetPrevCoefficient(fdphi,&phicoeffprev));
  PetscCall(SetInitialPorosityCoefficient(dmphicoeff,phicoeffprev,usr));
  // PetscCall(DMStagViewBinaryPython(dmphicoeff,phicoeffprev,"out_phicoeffprev"));
  PetscCall(VecDestroy(&phicoeffprev));

  // Time loop
  while ((nd->t <= nd->tmax) && (istep<nd->tstep)) {
    PetscPrintf(PETSC_COMM_WORLD,"# TIMESTEP %d: \n",istep);

    // Set dt for porosity evolution 
    // PetscCall(FDPDEAdvDiffComputeExplicitTimestep(fdphi,&dt));
    // nd->dt = PetscMin(dt,nd->dtmax);
    nd->dt = nd->dtmax;
    PetscCall(FDPDEAdvDiffSetTimestep(fdphi,nd->dt));

    // Update time
    nd->tprev = nd->t;
    nd->t    += nd->dt;

    // Porosity Solver - solve for phi_new, t
    PetscPrintf(PETSC_COMM_WORLD,"# Porosity Solver \n");
    PetscCall(FDPDESolve(fdphi,NULL));

    // converged = PETSC_FALSE;
    // while (!converged) {
    //   PetscCall(FDPDESolve(fdphi,&converged));
    //   if (!converged) { // Reduce dt if not converged
    //     nd->dt *= dt_damp;
    //     PetscCall(FDPDEAdvDiffSetTimestep(fdphi,nd->dt));
    //   }
    // }
    PetscCall(FDPDEGetSolution(fdphi,&xphi));

    // Porosity: copy solution and coefficient to old
    PetscCall(FDPDEAdvDiffGetPrevSolution(fdphi,&xphiprev));
    PetscCall(VecCopy(xphi,xphiprev));
    PetscCall(VecCopy(xphiprev,usr->xphiprev));
    PetscCall(VecDestroy(&xphiprev));

    PetscCall(FDPDEGetCoefficient(fdphi,&dmphicoeff,&phicoeff));
    PetscCall(FDPDEAdvDiffGetPrevCoefficient(fdphi,&phicoeffprev));
    PetscCall(VecCopy(phicoeff,phicoeffprev));
    PetscCall(VecDestroy(&phicoeffprev));

    // StokesDarcy Solver - using phi_old, tprev
    PetscPrintf(PETSC_COMM_WORLD,"# StokesDarcy Solver \n");
    PetscCall(FDPDESolve(fdPV,NULL));
    PetscCall(FDPDEGetSolution(fdPV,&xPV));
    PetscCall(VecCopy(xPV,usr->xPV));

    // Output solution and calculate fluid velocity
    if (istep % nd->tout == 0 ) {
      PetscPrintf(PETSC_COMM_WORLD,"# Write data to file \n");
      PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s_PV_nd_ts%1.3d",usr->par->fname_out,istep));
      PetscCall(DMStagViewBinaryPython(dmPV,xPV,fout));

      PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s_phi_nd_ts%1.3d",usr->par->fname_out,istep));
      PetscCall(DMStagViewBinaryPython(dmphi,xphi,fout));

      PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s_PV_residual_ts%1.3d",usr->par->fname_out,istep));
      PetscCall(DMStagViewBinaryPython(dmPV,fdPV->r,fout));

      PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s_phi_residual_ts%1.3d",usr->par->fname_out,istep));
      PetscCall(DMStagViewBinaryPython(dmphi,fdphi->r,fout));

      PetscCall(ComputeFluidVelocity(dmPV,xPV,dmphi,xphi,&xF,usr));
      PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s_xF_nd_ts%1.3d",usr->par->fname_out,istep));
      PetscCall(DMStagViewBinaryPython(dmPV,xF,fout));

      PetscCall(ScaleParametersAndOutput(dmPV,xPV,dmphi,xphi,xF,usr));
      PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s_dim_sol_ts%1.3d",usr->par->fname_out,istep));
      PetscCall(DMStagViewBinaryPython(usr->dmscal,usr->xscal,fout));

      PetscCall(VecDestroy(&xF));
    }

    // Clean up
    PetscCall(VecDestroy(&xPV));
    PetscCall(VecDestroy(&xphi));

    // increment timestep
    istep++;

    PetscPrintf(PETSC_COMM_WORLD,"# TIME: time = %1.12e [yr] dt = %1.12e [yr] \n",nd->t*usr->scal->t/SEC_YEAR,nd->dt*usr->scal->t/SEC_YEAR);
    PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");
  }

  // Destroy objects
  PetscCall(VecDestroy(&xmms_PV));
  PetscCall(VecDestroy(&xmms_phi));
  PetscCall(DMDestroy(&dmPV));
  PetscCall(DMDestroy(&dmphi));
  PetscCall(DMDestroy(&usr->dmscal));
  PetscCall(FDPDEDestroy(&fdPV));
  PetscCall(FDPDEDestroy(&fdphi));
  PetscCall(VecDestroy(&usr->xPV));
  PetscCall(VecDestroy(&usr->xphiprev));
  PetscCall(VecDestroy(&usr->xscal));

  PetscFunctionReturn(PETSC_SUCCESS);
}
// ---------------------------------------
// InputParameters
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "InputParameters"
PetscErrorCode InputParameters(UsrData **_usr)
{
  UsrData       *usr;
  Params        *par;
  PetscBag       bag;
  PetscFunctionBeginUser;

  // Allocate memory to application context
  PetscCall(PetscMalloc1(1, &usr)); 

  // Get time, comm and rank
  usr->comm = PETSC_COMM_WORLD;
  PetscCall(MPI_Comm_rank(PETSC_COMM_WORLD, &usr->rank)); 

  // Create bag
  PetscCall(PetscBagCreate (usr->comm,sizeof(Params),&usr->bag)); 
  PetscCall(PetscBagGetData(usr->bag,(void **)&usr->par)); 
  PetscCall(PetscBagSetName(usr->bag,"UserParamBag","- User defined parameters -")); 

  // Define some pointers for easy access
  bag = usr->bag;
  par = usr->par;

  // Initialize domain variables
  PetscCall(PetscBagRegisterInt(bag, &par->nx, 10, "nx", "Element count in the x-dir [-]")); 
  PetscCall(PetscBagRegisterInt(bag, &par->nz, 10, "nz", "Element count in the z-dir [-]")); 

  PetscCall(PetscBagRegisterScalar(bag, &par->xmin, 0.0, "xmin", "Start coordinate of domain in x-dir [m]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->zmin, -100.0e3, "zmin", "Start coordinate of domain in z-dir [m]")); 

  PetscCall(PetscBagRegisterScalar(bag, &par->L, 200.0e3, "L", "Length of domain in x-dir [m]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->H, 100.0e3, "H", "Height of domain in z-dir [m]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->xMOR, 200.0e3, "xMOR", "Distance from mid-ocean ridge axis for melt extraction [m]")); 

  // Physical and material parameters
  PetscCall(PetscBagRegisterScalar(bag, &par->k_hat, 0.0, "k_hat", "Direction of unit vertical vector [-]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->g, 10.0, "g", "Gravitational acceleration [m^2/s]")); 

  PetscCall(PetscBagRegisterScalar(bag, &par->phi0, 0.01, "phi0", "Reference porosity [-]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->K0, 1.0e-7, "K0", "Reference permeability [m^2]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->u0, 2.0, "u0", "Half-spreading rate [cm/yr]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->drho, 500.0, "drho", "Density difference [kg/m^3]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->rho0, 3000.0, "rho0", "Reference density [kg/m^3]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->n, 3.0, "n", "Porosity exponent")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->mu, 1.0, "mu", "Fluid viscosity [Pa.s]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->eta, 1.0e18, "eta", "Shear viscosity [Pa.s]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->zeta, 1.0e20, "zeta", "Bulk viscosity [Pa.s]")); 

  // Time stepping and advection parameters
  PetscCall(PetscBagRegisterInt(bag, &par->ts_scheme,2, "ts_scheme", "Time stepping scheme 0-forward euler, 1-backward euler, 2-crank-nicholson")); 
  PetscCall(PetscBagRegisterInt(bag, &par->adv_scheme,1, "adv_scheme", "Advection scheme 0-upwind, 1-fromm")); 

  PetscCall(PetscBagRegisterInt(bag, &par->tout,1, "tout", "Output every tout time step")); 
  PetscCall(PetscBagRegisterInt(bag, &par->tstep,1, "tstep", "Maximum no of time steps")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->tmax, 1.0e6, "tmax", "Maximum time [yr]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->dtmax, 1.0e3, "dtmax", "Maximum time step size [yr]")); 

  // Input/output 
  PetscCall(PetscBagRegisterString(bag,&par->fname_out,FNAME_LENGTH,"out_solution","output_file","Name for output file, set with: -output_file <filename>")); 

  // return pointer
  *_usr = usr;

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// InputPrintData
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "InputPrintData"
PetscErrorCode InputPrintData(UsrData *usr)
{
  char           date[30], *opts;
  PetscFunctionBeginUser;

  // Get date
  PetscCall(PetscGetDate(date,30)); 
  PetscCall(PetscOptionsGetAll(NULL, &opts)); 

  // Print header and petsc options
  PetscPrintf(usr->comm,"# --------------------------------------- #\n");
  PetscPrintf(usr->comm,"# MID-OCEAN RIDGE Model (MECHANICS): %s \n",&(date[0]));
  PetscPrintf(usr->comm,"# --------------------------------------- #\n");
  PetscPrintf(usr->comm,"# PETSc options: %s \n",opts);
  PetscPrintf(usr->comm,"# --------------------------------------- #\n");

  // Print usr bag
  PetscCall(PetscBagView(usr->bag,PETSC_VIEWER_STDOUT_WORLD)); 
  PetscPrintf(usr->comm,"# --------------------------------------- #\n");

  // Free memory
  PetscCall(PetscFree(opts)); 

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// DefineScalingParameters - define scales
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "DefineScalingParameters"
PetscErrorCode DefineScalingParameters(UsrData *usr)
{
  ScalParams     *scal;
  PetscFunctionBeginUser;

  // Allocate memory
  PetscCall(PetscMalloc1(1, &scal)); 

  scal->h = usr->par->H;
  scal->v = usr->par->K0*usr->par->drho*usr->par->g/usr->par->mu;
  scal->t = scal->h/scal->v;
  scal->K = usr->par->K0;
  scal->P = usr->par->drho*usr->par->g*scal->h;
  scal->eta = usr->par->eta;
  scal->rho = usr->par->rho0;
  scal->Gamma = scal->v*scal->rho/scal->h;
  scal->delta = PetscSqrtScalar(scal->eta*scal->K/usr->par->mu)/scal->h;
 
  usr->scal = scal;

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// NondimensionalizeParameters
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "NondimensionalizeParameters"
PetscErrorCode NondimensionalizeParameters(UsrData *usr)
{
  NDParams       *nd;
  ScalParams     *scal;
  Params         *par;
  PetscFunctionBeginUser;

  // Allocate memory
  PetscCall(PetscMalloc1(1, &nd)); 

  scal = usr->scal;
  par  = usr->par;

  // Transform to SI units necessary params
  par->u0    = par->u0*1.0e-2/SEC_YEAR; //[cm/yr] to [m/s]
  par->tmax  = par->tmax*SEC_YEAR;      //[yr] to [s]
  par->dtmax = par->dtmax*SEC_YEAR;     //[yr] to [s]

  nd->nx = par->nx;
  nd->nz = par->nz;
  nd->k_hat = par->k_hat;
  nd->n = par->n;
  nd->phi0 = par->phi0;

  nd->ts_scheme = par->ts_scheme;
  nd->adv_scheme = par->adv_scheme;
  nd->tout = par->tout;
  nd->tstep = par->tstep;

  nd->t  = 0.0;
  nd->dt = 0.0;

  // non-dimensionalize
  nd->xmin = nd_param(par->xmin,scal->h);
  nd->zmin = nd_param(par->zmin,scal->h);
  nd->H = nd_param(par->H,scal->h);
  nd->L = nd_param(par->L,scal->h);
  nd->xMOR = nd_param(par->xMOR,scal->h);
  nd->u0 = nd_param(par->u0,scal->v);
  nd->eta = nd_param(par->eta,scal->eta);
  nd->zeta = nd_param(par->zeta,scal->eta);
  nd->tmax = nd_param(par->tmax,scal->t);
  nd->dtmax = nd_param(par->dtmax,scal->t);

  nd->xi = nd->zeta - 2.0/3.0*nd->eta;
  nd->Gamma = 0.0;

  usr->nd = nd;

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// FormCoefficient_PV
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormCoefficient_PV"
PetscErrorCode FormCoefficient_PV(FDPDE fd, DM dm, Vec x, DM dmcoeff, Vec coeff, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  NDParams       *nd;
  ScalParams     *scal;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz;
  DM             dmphi;
  Vec            xphi, xphilocal, coefflocal;
  PetscScalar    **coordx,**coordz;
  PetscInt       iprev, inext, icenter;
  PetscScalar    ***c;
  PetscFunctionBeginUser;

  // Get dm and solution vector for porosity
  dmphi = usr->dmphi;
  xphi  = usr->xphiprev;

  nd   = usr->nd;
  scal = usr->scal;

  PetscCall(DMCreateLocalVector(dmphi,&xphilocal));
  PetscCall(DMGlobalToLocalBegin(dmphi,xphi,INSERT_VALUES,xphilocal));
  PetscCall(DMGlobalToLocalEnd(dmphi,xphi,INSERT_VALUES,xphilocal));

  // Get dmcoeff
  PetscCall(DMStagGetGlobalSizes(dmcoeff,&Nx,&Nz,NULL));
  PetscCall(DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 
  PetscCall(DMStagGetProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmcoeff,LEFT,&iprev));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmcoeff,RIGHT,&inext)); 
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmcoeff,ELEMENT,&icenter));

  // Create coefficient local vector
  PetscCall(DMCreateLocalVector(dmcoeff, &coefflocal)); 
  PetscCall(DMStagVecGetArray(dmcoeff, coefflocal, &c)); 
  
  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      PetscInt idx;

      { // A = delta^2*eta (center, c=1)
        DMStagStencil point;
        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 1;
        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
        c[j][i][idx] = scal->delta*scal->delta*nd->eta;
      }

      { // A = delta^2*eta (corner, c=0)
        DMStagStencil point[4];
        PetscInt      ii;

        point[0].i = i; point[0].j = j; point[0].loc = DOWN_LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = DOWN_RIGHT; point[1].c = 0;
        point[2].i = i; point[2].j = j; point[2].loc = UP_LEFT;    point[2].c = 0;
        point[3].i = i; point[3].j = j; point[3].loc = UP_RIGHT;   point[3].c = 0;

        for (ii = 0; ii < 4; ii++) {
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] = scal->delta*scal->delta*nd->eta;
        }
      }

      { // B = -phi*k_hat (edges, c=0)
        DMStagStencil point[4], pointQ[3];
        PetscScalar   Q[3], Qinterp;
        PetscScalar   zp[4],rhs[4],zQ[3];
        PetscInt      ii;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 0;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = 0;
        point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = 0;

        zp[0] = coordz[j][icenter];
        zp[1] = coordz[j][icenter];
        zp[2] = coordz[j][iprev  ];
        zp[3] = coordz[j][inext  ];

        // Bx = 0
        rhs[0] = 0.0;
        rhs[1] = 0.0;

        // Bz = -phi*k_hat; get porosity - take into account domain borders
        pointQ[0].i = i; pointQ[0].j = j-1; pointQ[0].loc = ELEMENT; pointQ[0].c = 0;
        pointQ[1].i = i; pointQ[1].j = j  ; pointQ[1].loc = ELEMENT; pointQ[1].c = 0;
        pointQ[2].i = i; pointQ[2].j = j+1; pointQ[2].loc = ELEMENT; pointQ[2].c = 0;

        zQ[1] = coordz[j][icenter];
        if (j == 0   ) { pointQ[0] = pointQ[1]; zQ[0] = zQ[1];}
        else           {zQ[0] = coordz[j-1][icenter];}
        if (j == Nz-1) { pointQ[2] = pointQ[1]; zQ[2] = zQ[1];}
        else           {zQ[2] = coordz[j+1][icenter];}

        PetscCall(DMStagVecGetValuesStencil(dmphi,xphilocal,3,pointQ,Q)); 
        Qinterp = interp1DLin_3Points(zp[2],zQ[0],Q[0],zQ[1],Q[1],zQ[2],Q[2]); // Q = 1-phi
        rhs[2]  = nd->k_hat*(1.0-Qinterp);

        Qinterp = interp1DLin_3Points(zp[3],zQ[0],Q[0],zQ[1],Q[1],zQ[2],Q[2]);
        rhs[3]  = nd->k_hat*(1.0-Qinterp);

        for (ii = 0; ii < 4; ii++) {
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] = rhs[ii];
        }
      }

      { // C = 0.0 (center, c=0)
        DMStagStencil point;
        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 0;
        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
        c[j][i][idx] = 0.0;
      }

      { // D1 = delta^2*xi (center, c=2)
        DMStagStencil point;
        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 2;
        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
        c[j][i][idx] = scal->delta*scal->delta*nd->xi;
      }

      { // D2 = -Kphi (edges, c=1) - Kphi = (phi/phi0)^n
        DMStagStencil point[4], pointQ[5];
        PetscScalar   xp[4],zp[4], xQ[3], zQ[3], Q[5], Qinterp, rhs[4];
        PetscInt      ii;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 1;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 1;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = 1;
        point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = 1;

        xp[0] = coordx[i][iprev  ]; zp[0] = coordz[j][icenter];
        xp[1] = coordx[i][inext  ]; zp[1] = coordz[j][icenter];
        xp[2] = coordx[i][icenter]; zp[2] = coordz[j][iprev  ];
        xp[3] = coordx[i][icenter]; zp[3] = coordz[j][inext  ];

        // get porosity - take into account domain borders
        pointQ[0].i = i-1; pointQ[0].j = j  ; pointQ[0].loc = ELEMENT; pointQ[0].c = 0;
        pointQ[1].i = i  ; pointQ[1].j = j  ; pointQ[1].loc = ELEMENT; pointQ[1].c = 0;
        pointQ[2].i = i+1; pointQ[2].j = j  ; pointQ[2].loc = ELEMENT; pointQ[2].c = 0;
        pointQ[3].i = i  ; pointQ[3].j = j-1; pointQ[3].loc = ELEMENT; pointQ[3].c = 0;
        pointQ[4].i = i  ; pointQ[4].j = j+1; pointQ[4].loc = ELEMENT; pointQ[4].c = 0;

        xQ[1] = coordx[i][icenter];
        zQ[1] = coordz[j][icenter];
        if (i == 0   ) { pointQ[0] = pointQ[1]; xQ[0] = xQ[1];}
        else           { xQ[0] = coordx[i-1][icenter];}
        if (i == Nx-1) { pointQ[2] = pointQ[1]; xQ[2] = xQ[1];}
        else           { xQ[2] = coordx[i+1][icenter];}
        if (j == 0   ) { pointQ[3] = pointQ[1]; zQ[0] = zQ[1];}
        else           { zQ[0] = coordz[j-1][icenter];}
        if (j == Nz-1) { pointQ[4] = pointQ[1]; zQ[2] = zQ[1];}
        else           { zQ[2] = coordz[j+1][icenter];}

        PetscCall(DMStagVecGetValuesStencil(dmphi,xphilocal,5,pointQ,Q)); 

        Qinterp = interp1DLin_3Points(xp[0],xQ[0],Q[0],xQ[1],Q[1],xQ[2],Q[2]); // Q = 1-phi
        rhs[0] = -PetscPowScalar((1.0-Qinterp)/nd->phi0,nd->n); //left 

        Qinterp = interp1DLin_3Points(xp[1],xQ[0],Q[0],xQ[1],Q[1],xQ[2],Q[2]);
        rhs[1] = -PetscPowScalar((1.0-Qinterp)/nd->phi0,nd->n); //right

        Qinterp = interp1DLin_3Points(zp[2],zQ[0],Q[3],zQ[1],Q[1],zQ[2],Q[4]);
        rhs[2] = -PetscPowScalar((1.0-Qinterp)/nd->phi0,nd->n); // down

        Qinterp = interp1DLin_3Points(zp[3],zQ[0],Q[3],zQ[1],Q[1],zQ[2],Q[4]);
        rhs[3] = -PetscPowScalar((1.0-Qinterp)/nd->phi0,nd->n); // up

        for (ii = 0; ii < 4; ii++) {
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] = rhs[ii];
        }
      }

      { // D3 = Kphi*k_hat (edges, c=2) - Kphi = (phi/phi0)^n
        DMStagStencil point[4], pointQ[3];
        PetscScalar   xp[4],zp[4],Qinterp, Q[3], rhs[4],zQ[3];
        PetscInt      ii;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 2;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 2;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = 2;
        point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = 2;

        xp[0] = coordx[i][iprev  ]; zp[0] = coordz[j][icenter];
        xp[1] = coordx[i][inext  ]; zp[1] = coordz[j][icenter];
        xp[2] = coordx[i][icenter]; zp[2] = coordz[j][iprev  ];
        xp[3] = coordx[i][icenter]; zp[3] = coordz[j][inext  ];

        rhs[0] = 0.0; // dir of gravity only
        rhs[1] = 0.0; 

        // get porosity - take into account domain borders
        pointQ[0].i = i; pointQ[0].j = j-1; pointQ[0].loc = ELEMENT; pointQ[0].c = 0;
        pointQ[1].i = i; pointQ[1].j = j  ; pointQ[1].loc = ELEMENT; pointQ[1].c = 0;
        pointQ[2].i = i; pointQ[2].j = j+1; pointQ[2].loc = ELEMENT; pointQ[2].c = 0;

        zQ[1] = coordz[j][icenter];
        if (j == 0   ) { pointQ[0] = pointQ[1]; zQ[0] = zQ[1];}
        else           {zQ[0] = coordz[j-1][icenter];}
        if (j == Nz-1) { pointQ[2] = pointQ[1]; zQ[2] = zQ[1];}
        else           {zQ[2] = coordz[j+1][icenter];}

        PetscCall(DMStagVecGetValuesStencil(dmphi,xphilocal,3,pointQ,Q)); 
        Qinterp = interp1DLin_3Points(zp[2],zQ[0],Q[0],zQ[1],Q[1],zQ[2],Q[2]); // Q = 1-phi
        rhs[2]  = nd->k_hat*PetscPowScalar((1.0-Qinterp)/nd->phi0,nd->n);

        Qinterp = interp1DLin_3Points(zp[3],zQ[0],Q[0],zQ[1],Q[1],zQ[2],Q[2]);
        rhs[3]  = nd->k_hat*PetscPowScalar((1.0-Qinterp)/nd->phi0,nd->n); 

        for (ii = 0; ii < 4; ii++) {
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] = rhs[ii];
        }
      }
    }
  }

  // Restore arrays, local vectors
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL));

  PetscCall(DMStagVecRestoreArray(dmcoeff,coefflocal,&c));
  PetscCall(DMLocalToGlobalBegin(dmcoeff,coefflocal,INSERT_VALUES,coeff)); 
  PetscCall(DMLocalToGlobalEnd  (dmcoeff,coefflocal,INSERT_VALUES,coeff)); 
  
  PetscCall(VecDestroy(&coefflocal)); 
  PetscCall(VecDestroy(&xphilocal));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// FormBCList_PV - manufactured Dirichlet BC
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormBCList_PV"
PetscErrorCode FormBCList_PV(DM dm, Vec x, DMStagBCList bclist, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       k,n_bc,*idx_bc;
  PetscScalar    *value_bc,*x_bc;
  BCType         *type_bc;
  PetscFunctionBeginUser;

  // Reduced form: zero normal fluxes are implicitly integrated (i.e. gradP=0, dVxi/dxj=0).
  // No need to explicitly specify them, unless want to overwrite.

  // LEFT Boundary: vx = 0, partial_x(vz,P) = 0 (reduced form)
  // LEFT Vx = 0 
  PetscCall(DMStagBCListGetValues(bclist,'w','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // TOP Boundary: vx = u0, vz = 0, partial_z(P) = 0 (reduced form)
  // UP Vx = u0
  PetscCall(DMStagBCListGetValues(bclist,'n','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
      value_bc[k] = usr->nd->u0;
      type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // UP Vz = 0
  PetscCall(DMStagBCListGetValues(bclist,'n','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // RIGHT Boundary: partial_x(vx,vz) = 0, P = 0
  // RIGHT dVx/dx = 0
  PetscCall(DMStagBCListGetValues(bclist,'e','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN_T;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // RIGHT vz=0
  PetscCall(DMStagBCListGetValues(bclist,'e','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // RIGHT P = 0
  PetscCall(DMStagBCListGetValues(bclist,'e','o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // Bottom Boundary: partial_z(vx,vz) = 0, P = 0
  // DOWN vx = 0 
  PetscCall(DMStagBCListGetValues(bclist,'s','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // DOWN dVz/dz = 0
  PetscCall(DMStagBCListGetValues(bclist,'s','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN_T;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // DOWN P = 0
  PetscCall(DMStagBCListGetValues(bclist,'s','o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// FormCoefficient_phi
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormCoefficient_phi"
PetscErrorCode FormCoefficient_phi(FDPDE fd, DM dm, Vec x, DM dmcoeff, Vec coeff, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz,icenter;
  DM             dmPV = NULL;
  Vec            coefflocal;
  PetscScalar    ***c, **coordx, **coordz;
  Vec            xPV = NULL, xPVlocal;
  PetscFunctionBeginUser;

  // Get dm and solution vector for Stokes velocity
  dmPV = usr->dmPV;
  xPV  = usr->xPV;
  
  PetscCall(DMCreateLocalVector(dmPV,&xPVlocal));
  PetscCall(DMGlobalToLocalBegin(dmPV,xPV,INSERT_VALUES,xPVlocal));
  PetscCall(DMGlobalToLocalEnd(dmPV,xPV,INSERT_VALUES,xPVlocal));
  
  // Get domain corners and coordinates
  PetscCall(DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 
  PetscCall(DMStagGetGlobalSizes(dmcoeff,&Nx,&Nz,NULL));
  PetscCall(DMStagGetProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmcoeff,ELEMENT,&icenter));

  // Create coefficient local vector
  PetscCall(DMCreateLocalVector(dmcoeff, &coefflocal)); 
  PetscCall(DMStagVecGetArray(dmcoeff, coefflocal, &c)); 
  
  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {

      { // A = 1.0
        DMStagStencil point;
        PetscInt      idx;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 0;
        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
        c[j][i][idx] = 1.0;
      }

      { // C = Gamma
        DMStagStencil point;
        PetscInt      idx;
        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 1;
        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
        c[j][i][idx] = usr->nd->Gamma;
      }

      { // B = 0.0 (edge)
        DMStagStencil point[4];
        PetscInt      ii, idx;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 0;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = 0;
        point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = 0;

        for (ii = 0; ii < 4; ii++) {
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] = 0.0;
        }
      }

      { // u = velocity (edge) - StokesDarcy solid velocity
        PetscScalar   v[4];
        DMStagStencil point[4];
        PetscInt      ii, idx;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 0;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = 0;
        point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = 0;
        
        PetscCall(DMStagVecGetValuesStencil(dmPV,xPVlocal,4,point,v)); 

        for (ii = 0; ii < 4; ii++) {
          point[ii].c = 1;
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] = v[ii];
        }
      }
    }
  }

  // Restore arrays, local vectors
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL));
  PetscCall(DMStagVecRestoreArray(dmcoeff,coefflocal,&c));
  PetscCall(DMLocalToGlobalBegin(dmcoeff,coefflocal,INSERT_VALUES,coeff)); 
  PetscCall(DMLocalToGlobalEnd  (dmcoeff,coefflocal,INSERT_VALUES,coeff)); 
  PetscCall(VecDestroy(&coefflocal)); 
  
  PetscCall(VecDestroy(&xPVlocal));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// FormBCList_phi
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormBCList_phi"
PetscErrorCode FormBCList_phi(DM dm, Vec x, DMStagBCList bclist, void *ctx)
{
  UsrData     *usr = (UsrData*)ctx;
  PetscInt    k,n_bc,*idx_bc;
  PetscScalar *value_bc,*x_bc;
  BCType      *type_bc;
  PetscFunctionBeginUser;
  
  // Left = dphi/dx = d(1-phi)/dx = 0
  PetscCall(DMStagBCListGetValues(bclist,'w','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));

  // RIGHT:
  PetscCall(DMStagBCListGetValues(bclist,'e','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));

  // DOWN: phi = phi0
  PetscCall(DMStagBCListGetValues(bclist,'s','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 1.0 - usr->nd->phi0;
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));

  // UP: dphi/dz = 0 if x<=xMOR, phi=0 otherwise
  PetscCall(DMStagBCListGetValues(bclist,'n','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    if (x_bc[2*k]<=usr->nd->xMOR) {
      value_bc[k] = 0.0;
      type_bc[k] = BC_NEUMANN;
    } else {
      value_bc[k] = 1.0;
      type_bc[k] = BC_DIRICHLET_STAG;
    }
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
 
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// SetInitialPorosityProfile
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "SetInitialPorosityProfile"
PetscErrorCode SetInitialPorosityProfile(DM dm, Vec x, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  Vec           xlocal;
  PetscInt      i,j, sx, sz, nx, nz, icenter;
  PetscScalar   ***xx, **coordx, **coordz;
  PetscFunctionBeginUser;

  // Get domain corners
  PetscCall(DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 

  // Get dm coordinates array
  PetscCall(DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter));

  // Create local vector
  PetscCall(DMCreateLocalVector(dm, &xlocal)); 
  PetscCall(DMStagVecGetArray(dm, xlocal, &xx)); 

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      DMStagStencil point;
      PetscInt      idx;
      PetscScalar   phi;

      point.i = i; point.j = j; point.loc = ELEMENT; point.c = 0;
      PetscCall(DMStagGetLocationSlot(dm, point.loc, point.c, &idx)); 
      phi = usr->nd->phi0;// + usr->nd->phi0*usr->nd->phi0*PetscCosScalar(2.0*PETSC_PI*coordx[i][icenter])*PetscCosScalar(2.0*PETSC_PI*coordz[j][icenter]);
      xx[j][i][idx] = 1.0-phi; //(1.0-usr->nd->phi0);
    }
  }

  // Restore arrays, local vectors
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));

  PetscCall(DMStagVecRestoreArray(dm,xlocal,&xx));
  PetscCall(DMLocalToGlobalBegin(dm,xlocal,INSERT_VALUES,x)); 
  PetscCall(DMLocalToGlobalEnd  (dm,xlocal,INSERT_VALUES,x)); 
  
  PetscCall(VecDestroy(&xlocal)); 

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// SetInitialPorosityCoefficient
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "SetInitialPorosityCoefficient"
PetscErrorCode SetInitialPorosityCoefficient(DM dmcoeff, Vec coeff, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  DM             dmPV;
  PetscInt       i, j, sx, sz, nx, nz,iprev,inext,icenter;
  Vec            coefflocal,xPV,xPVlocal;
  PetscScalar    ***c, **coordx, **coordz;
  PetscFunctionBeginUser;

  // Get dm and solution vector for Stokes velocity
  dmPV = usr->dmPV;
  xPV  = usr->xPV;
  
  PetscCall(DMCreateLocalVector(dmPV,&xPVlocal));
  PetscCall(DMGlobalToLocalBegin(dmPV,xPV,INSERT_VALUES,xPVlocal));
  PetscCall(DMGlobalToLocalEnd(dmPV,xPV,INSERT_VALUES,xPVlocal));

  // Get domain corners
  PetscCall(DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 

  PetscCall(DMStagGetProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmcoeff,LEFT,&iprev));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmcoeff,RIGHT,&inext)); 
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmcoeff,ELEMENT,&icenter));

  // Create coefficient local vector
  PetscCall(DMCreateLocalVector(dmcoeff, &coefflocal)); 
  PetscCall(DMStagVecGetArray(dmcoeff, coefflocal, &c)); 
  
  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {

      { // A = 1.0
        DMStagStencil point;
        PetscInt      idx;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 0;
        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
        c[j][i][idx] = 1.0;
      }

      { // C = Gamma
        DMStagStencil point;
        PetscInt      idx;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 1;
        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
        c[j][i][idx] = usr->nd->Gamma;
      }

      { // B = 0.0 (edge)
        DMStagStencil point[4];
        PetscInt      ii, idx;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 0;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = 0;
        point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = 0;

        for (ii = 0; ii < 4; ii++) {
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, 0, &idx)); 
          c[j][i][idx] = 0.0;
        }
      }

      { // u = velocity (edge) - StokesDarcy solid velocity
        PetscScalar   v[4];
        DMStagStencil point[4];
        PetscInt      ii, idx;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 0;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = 0;
        point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = 0;
        
        PetscCall(DMStagVecGetValuesStencil(dmPV,xPVlocal,4,point,v)); 

        for (ii = 0; ii < 4; ii++) {
          point[ii].c = 1;
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] = v[ii];
        }
      }
    }
  }

  // Restore arrays, local vectors
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL));
  PetscCall(DMStagVecRestoreArray(dmcoeff,coefflocal,&c));
  PetscCall(DMLocalToGlobalBegin(dmcoeff,coefflocal,INSERT_VALUES,coeff)); 
  PetscCall(DMLocalToGlobalEnd  (dmcoeff,coefflocal,INSERT_VALUES,coeff)); 
  PetscCall(VecDestroy(&coefflocal)); 
  PetscCall(VecDestroy(&xPVlocal));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// ComputeFluidVelocity
// ---------------------------------------
PetscErrorCode ComputeFluidVelocity(DM dmPV,Vec xPV,DM dmphi,Vec xphi,Vec *_xF,void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz, idx;
  PetscInt       iprev, inext, icenter;
  PetscScalar    ***xxF;
  PetscScalar    **coordx,**coordz;
  Vec            xF,xFlocal,xphilocal,xPVlocal;
  PetscFunctionBeginUser;

  // Create local and global vector for fluid velocity
  PetscCall(DMCreateGlobalVector(dmPV,&xF     )); 
  PetscCall(DMCreateLocalVector (dmPV,&xFlocal)); 
  PetscCall(DMStagVecGetArray(dmPV,xFlocal,&xxF)); 

  PetscCall(DMCreateLocalVector(dmphi,&xphilocal));
  PetscCall(DMGlobalToLocalBegin(dmphi,xphi,INSERT_VALUES,xphilocal));
  PetscCall(DMGlobalToLocalEnd(dmphi,xphi,INSERT_VALUES,xphilocal));

  PetscCall(DMCreateLocalVector(dmPV,&xPVlocal));
  PetscCall(DMGlobalToLocalBegin(dmPV,xPV,INSERT_VALUES,xPVlocal));
  PetscCall(DMGlobalToLocalEnd(dmPV,xPV,INSERT_VALUES,xPVlocal));

  // Get domain corners and coordinates
  PetscCall(DMStagGetGlobalSizes(dmPV, &Nx, &Nz,NULL));
  PetscCall(DMStagGetCorners(dmPV, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 
  PetscCall(DMStagGetProductCoordinateArraysRead(dmPV,&coordx,&coordz,NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmPV,ELEMENT,&icenter)); 
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmPV,LEFT,&iprev));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmPV,RIGHT,&inext)); 

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      DMStagStencil point[9],pointQ[5];
      PetscScalar xx[9],Q[5],Qinterp[4],xp[3],zp[3],dx,dz,phi,gradP;

      // center points - dummy
      PetscCall(DMStagGetLocationSlot(dmPV,ELEMENT,0,&idx)); 
      xxF[j][i][idx] = 0.0;

      // edges - fluid velocity
      point[0].i = i; point[0].j = j; point[0].loc = LEFT;    point[0].c = 0;
      point[1].i = i; point[1].j = j; point[1].loc = RIGHT;   point[1].c = 0;
      point[2].i = i; point[2].j = j; point[2].loc = DOWN;    point[2].c = 0;
      point[3].i = i; point[3].j = j; point[3].loc = UP;      point[3].c = 0;
      point[4].i = i; point[4].j = j; point[4].loc = ELEMENT; point[4].c = 0;

      point[5].i = i-1; point[5].j = j  ; point[5].loc = ELEMENT; point[5].c = 0;
      point[6].i = i+1; point[6].j = j  ; point[6].loc = ELEMENT; point[6].c = 0;
      point[7].i = i  ; point[7].j = j-1; point[7].loc = ELEMENT; point[7].c = 0;
      point[8].i = i  ; point[8].j = j+1; point[8].loc = ELEMENT; point[8].c = 0;

      // correct for edges for all forms of BC: P = 0, dP/dx=0, dP/dz = 0
      if (i == 0   ) point[5] = point[0];
      if (i == Nx-1) point[6] = point[0];
      if (j == 0   ) point[7] = point[0];
      if (j == Nz-1) point[8] = point[0];

      PetscCall(DMStagVecGetValuesStencil(dmPV,xPVlocal,9,point,xx)); 
      
      // grid spacing - assume constant
      dx = coordx[i][inext]-coordx[i][iprev];
      dz = coordz[j][inext]-coordz[j][iprev];

      // porosity
      pointQ[0].i = i  ; pointQ[0].j = j  ; pointQ[0].loc = ELEMENT; pointQ[0].c = 0;
      pointQ[1].i = i-1; pointQ[1].j = j  ; pointQ[1].loc = ELEMENT; pointQ[1].c = 0;
      pointQ[2].i = i+1; pointQ[2].j = j  ; pointQ[2].loc = ELEMENT; pointQ[2].c = 0;
      pointQ[3].i = i  ; pointQ[3].j = j-1; pointQ[3].loc = ELEMENT; pointQ[3].c = 0;
      pointQ[4].i = i  ; pointQ[4].j = j+1; pointQ[4].loc = ELEMENT; pointQ[4].c = 0;

      if (i == 0   ) pointQ[1] = point[0];
      if (i == Nx-1) pointQ[2] = point[0];
      if (j == 0   ) pointQ[3] = point[0];
      if (j == Nz-1) pointQ[4] = point[0];

      if (i == 0   ) { xp[0] = coordx[i][icenter];} else { xp[0] = coordx[i-1][icenter];}
      if (i == Nx-1) { xp[2] = coordx[i][icenter];} else { xp[2] = coordx[i+1][icenter];}
      if (j == 0   ) { zp[0] = coordz[j][icenter];} else { zp[0] = coordz[j-1][icenter];}
      if (j == Nz-1) { zp[2] = coordz[j][icenter];} else { zp[2] = coordz[j+1][icenter];}
      xp[1] = coordx[i][icenter];
      zp[1] = coordz[j][icenter];

      PetscCall(DMStagVecGetValuesStencil(dmphi,xphilocal,5,pointQ,Q)); 

      // porosity on edges
      Qinterp[0] = interp1DLin_3Points(coordx[i][iprev],xp[0],Q[1],xp[1],Q[0],xp[2],Q[2]); 
      Qinterp[1] = interp1DLin_3Points(coordx[i][inext],xp[0],Q[1],xp[1],Q[0],xp[2],Q[2]); 
      Qinterp[2] = interp1DLin_3Points(coordz[j][iprev],zp[0],Q[3],zp[1],Q[0],zp[2],Q[4]); 
      Qinterp[3] = interp1DLin_3Points(coordz[j][inext],zp[0],Q[3],zp[1],Q[0],zp[2],Q[4]); 

      // left
      phi   = 1.0 - Qinterp[0];
      gradP = (xx[4]-xx[5])/dx;
      PetscCall(DMStagGetLocationSlot(dmPV,LEFT,0,&idx)); 
      xxF[j][i][idx] = fluid_velocity(xx[0],phi,usr->nd->phi0,usr->nd->n,gradP,0.0);

      // right
      phi   = 1.0 - Qinterp[1];
      gradP = (xx[6]-xx[4])/dx;
      PetscCall(DMStagGetLocationSlot(dmPV,RIGHT,0,&idx)); 
      xxF[j][i][idx] = fluid_velocity(xx[1],phi,usr->nd->phi0,usr->nd->n,gradP,0.0);

      // down
      phi   = 1.0 - Qinterp[2];
      gradP = (xx[4]-xx[7])/dz;
      PetscCall(DMStagGetLocationSlot(dmPV,DOWN,0,&idx)); 
      xxF[j][i][idx] = fluid_velocity(xx[2],phi,usr->nd->phi0,usr->nd->n,gradP,usr->nd->k_hat);

      // up
      phi   = 1.0 - Qinterp[3];
      gradP = (xx[8]-xx[4])/dz;
      PetscCall(DMStagGetLocationSlot(dmPV,UP,0,&idx)); 
      xxF[j][i][idx] = fluid_velocity(xx[3],phi,usr->nd->phi0,usr->nd->n,gradP,usr->nd->k_hat);
    }
  }

  // Restore arrays
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dmPV,&coordx,&coordz,NULL));

  // Restore and map local to global
  PetscCall(DMStagVecRestoreArray(dmPV,xFlocal,&xxF)); 
  PetscCall(DMLocalToGlobalBegin(dmPV,xFlocal,INSERT_VALUES,xF)); 
  PetscCall(DMLocalToGlobalEnd  (dmPV,xFlocal,INSERT_VALUES,xF)); 
  PetscCall(VecDestroy(&xFlocal)); 
  PetscCall(VecDestroy(&xPVlocal)); 
  PetscCall(VecDestroy(&xphilocal));

  // Assign pointers
  *_xF  = xF;
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// Scale-up parameters and output
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "ScaleParametersAndOutput"
PetscErrorCode ScaleParametersAndOutput(DM dmPV, Vec xPV, DM dmphi, Vec xphi, Vec xF, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  DM            dm;
  Vec           xlocal, xPVlocal, xFlocal, xphilocal;
  PetscInt      i,j, sx, sz, nx, nz, icenter;
  PetscScalar   ***xx, **coordx, **coordz;
  PetscFunctionBeginUser;

  // Get domain corners
  dm = usr->dmscal;
  PetscCall(DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 

  // Get dm coordinates array
  PetscCall(DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter));

  // Create local vector
  PetscCall(DMCreateLocalVector(dm, &xlocal)); 
  PetscCall(DMStagVecGetArray(dm, xlocal, &xx)); 

  // Get solution vectors
  PetscCall(DMCreateLocalVector(dmPV,&xPVlocal));
  PetscCall(DMGlobalToLocalBegin(dmPV,xPV,INSERT_VALUES,xPVlocal));
  PetscCall(DMGlobalToLocalEnd(dmPV,xPV,INSERT_VALUES,xPVlocal));

  PetscCall(DMCreateLocalVector(dmPV,&xFlocal));
  PetscCall(DMGlobalToLocalBegin(dmPV,xF,INSERT_VALUES,xFlocal));
  PetscCall(DMGlobalToLocalEnd(dmPV,xF,INSERT_VALUES,xFlocal));

  PetscCall(DMCreateLocalVector(dmphi,&xphilocal));
  PetscCall(DMGlobalToLocalBegin(dmphi,xphi,INSERT_VALUES,xphilocal));
  PetscCall(DMGlobalToLocalEnd(dmphi,xphi,INSERT_VALUES,xphilocal));

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      DMStagStencil point[5];
      PetscScalar   xval[5];
      PetscInt      ii, idx;

      // PV
      point[0].i = i; point[0].j = j; point[0].loc = ELEMENT; point[0].c = 0;
      point[1].i = i; point[1].j = j; point[1].loc = LEFT;    point[1].c = 0;
      point[2].i = i; point[2].j = j; point[2].loc = RIGHT;   point[2].c = 0;
      point[3].i = i; point[3].j = j; point[3].loc = DOWN;    point[3].c = 0;
      point[4].i = i; point[4].j = j; point[4].loc = UP;      point[4].c = 0;

      PetscCall(DMStagVecGetValuesStencil(dmPV,xPVlocal,5,point,xval)); 
      PetscCall(DMStagGetLocationSlot(dm, point[0].loc, 0, &idx)); 
      xx[j][i][idx] = dim_param(xval[0],usr->scal->P);

      for (ii = 1; ii < 5; ii++) {
        PetscCall(DMStagGetLocationSlot(dm, point[ii].loc, 0, &idx)); 
        xx[j][i][idx] = dim_param(xval[ii],usr->scal->v);
      }

      // xF
      point[0].i = i; point[0].j = j; point[0].loc = LEFT;    point[0].c = 0;
      point[1].i = i; point[1].j = j; point[1].loc = RIGHT;   point[1].c = 0;
      point[2].i = i; point[2].j = j; point[2].loc = DOWN;    point[2].c = 0;
      point[3].i = i; point[3].j = j; point[3].loc = UP;      point[3].c = 0;

      PetscCall(DMStagVecGetValuesStencil(dmPV,xFlocal,4,point,xval)); 
      for (ii = 0; ii < 4; ii++) {
        PetscCall(DMStagGetLocationSlot(dm, point[ii].loc, 1, &idx)); 
        xx[j][i][idx] = dim_param(xval[ii],usr->scal->v);
      }

      // phi
      point[0].i = i; point[0].j = j; point[0].loc = ELEMENT; point[0].c = 0;
      PetscCall(DMStagVecGetValuesStencil(dmphi,xphilocal,1,point,xval)); 
      PetscCall(DMStagGetLocationSlot(dm, point[0].loc, 1, &idx)); 
      xx[j][i][idx] = dim_param(xval[0],usr->scal->P);

      // Plith
      point[0].i = i; point[0].j = j; point[0].loc = ELEMENT; point[0].c = 2;
      PetscCall(DMStagGetLocationSlot(dm, point[0].loc, 2, &idx)); 
      xx[j][i][idx] = -usr->scal->rho*usr->par->g*coordz[j][icenter];
    }
  }

  // Restore arrays, local vectors
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));

  PetscCall(DMStagVecRestoreArray(dm,xlocal,&xx));
  PetscCall(DMLocalToGlobalBegin(dm,xlocal,INSERT_VALUES,usr->xscal)); 
  PetscCall(DMLocalToGlobalEnd  (dm,xlocal,INSERT_VALUES,usr->xscal)); 

  PetscCall(VecDestroy(&xlocal)); 
  PetscCall(VecDestroy(&xphilocal));
  PetscCall(VecDestroy(&xPVlocal));
  PetscCall(VecDestroy(&xFlocal));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// MAIN
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "main"
int main (int argc,char **argv)
{
  UsrData         *usr;
  PetscLogDouble  start_time, end_time;
    
  // Initialize application
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
 
  // Load command line or input file if required
  PetscCall(PetscOptionsInsert(PETSC_NULLPTR,&argc,&argv,NULL)); 

  // Input user parameters and print
  PetscCall(InputParameters(&usr)); 
  PetscCall(InputPrintData(usr)); 

  // Define scaling parameters and non-dimensionalize
  PetscCall(DefineScalingParameters(usr)); 
  PetscCall(NondimensionalizeParameters(usr));

  // Numerical solution using the FD pde object
  PetscCall(PetscTime(&start_time)); 
  PetscCall(Numerical_solution(usr)); 
  PetscCall(PetscTime(&end_time)); 
  PetscPrintf(PETSC_COMM_WORLD,"# Runtime: %g (sec) \n", end_time - start_time);
  PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");

  // Destroy objects
  PetscCall(PetscFree(usr->nd)); 
  PetscCall(PetscFree(usr->scal)); 
  PetscCall(PetscBagDestroy(&usr->bag)); 
  PetscCall(PetscFree(usr)); 

  // Finalize main
  PetscCall(PetscFinalize());
  return 0;
}
