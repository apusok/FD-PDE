// ---------------------------------------
// Mid-ocean ridge model solving for mechanics (conservation of mass and momentum)
// Solves for coupled (P, v) and Q=(1-phi) evolution, where P-dynamic pressure, v-solid velocity, phi-porosity.
// run: ./mor_mechanics.app -pc_type lu -pc_factor_mat_solver_type umfpack -nx 20 -nz 20 -snes_monitor 
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

#include "petsc.h"
#include "../../src/fdpde_stokesdarcy2field.h"
#include "../../src/fdpde_advdiff.h"
#include "../../src/dmstagoutput.h"

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
  PetscErrorCode ierr;

  PetscFunctionBegin;

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
  ierr = FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_STOKESDARCY2FIELD,&fdPV);CHKERRQ(ierr);
  ierr = FDPDESetUp(fdPV);CHKERRQ(ierr);
  ierr = FDPDESetFunctionBCList(fdPV,FormBCList_PV,bc_description_stokesdarcy,usr); CHKERRQ(ierr);
  ierr = FDPDESetFunctionCoefficient(fdPV,FormCoefficient_PV,coeff_description_stokesdarcy,usr); CHKERRQ(ierr);
  ierr = SNESSetFromOptions(fdPV->snes); CHKERRQ(ierr);
  ierr = SNESSetOptionsPrefix(fdPV->snes,"pv_"); CHKERRQ(ierr);

  // Set up Porosity (Advection-diffusion) system
  PetscPrintf(PETSC_COMM_WORLD,"# Set FD-PDE AdvDiff (porosity)\n");
  ierr = FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_ADVDIFF,&fdphi);CHKERRQ(ierr);
  ierr = FDPDESetUp(fdphi);CHKERRQ(ierr);

  if (nd->adv_scheme==0) { ierr = FDPDEAdvDiffSetAdvectSchemeType(fdphi,ADV_UPWIND);CHKERRQ(ierr); }
  if (nd->adv_scheme==1) { ierr = FDPDEAdvDiffSetAdvectSchemeType(fdphi,ADV_FROMM);CHKERRQ(ierr); }
  
  if (nd->ts_scheme ==  0) { ierr = FDPDEAdvDiffSetTimeStepSchemeType(fdphi,TS_FORWARD_EULER);CHKERRQ(ierr); }
  if (nd->ts_scheme ==  1) { ierr = FDPDEAdvDiffSetTimeStepSchemeType(fdphi,TS_BACKWARD_EULER);CHKERRQ(ierr); }
  if (nd->ts_scheme ==  2) { ierr = FDPDEAdvDiffSetTimeStepSchemeType(fdphi,TS_CRANK_NICHOLSON );CHKERRQ(ierr);}

  ierr = FDPDESetFunctionBCList(fdphi,FormBCList_phi,bc_description_phi,usr); CHKERRQ(ierr);
  ierr = FDPDESetFunctionCoefficient(fdphi,FormCoefficient_phi,coeff_description_phi,usr); CHKERRQ(ierr);
  ierr = SNESSetFromOptions(fdphi->snes); CHKERRQ(ierr);
  ierr = SNESSetOptionsPrefix(fdphi->snes,"phi_"); CHKERRQ(ierr);

  // Prepare usr data for coupling
  ierr = FDPDEGetDM(fdPV,&dmPV); CHKERRQ(ierr);
  ierr = FDPDEGetDM(fdphi,&dmphi); CHKERRQ(ierr);
  usr->dmPV  = dmPV;
  usr->dmphi = dmphi;

  ierr = FDPDEGetSolution(fdPV,&xPV);CHKERRQ(ierr);
  ierr = FDPDEGetSolution(fdphi,&xphi);CHKERRQ(ierr);
  ierr = VecDuplicate(xphi,&usr->xphiprev);CHKERRQ(ierr);
  ierr = VecDuplicate(xPV,&usr->xPV);CHKERRQ(ierr);
  ierr = VecDestroy(&xphi);CHKERRQ(ierr);
  ierr = VecDestroy(&xPV);CHKERRQ(ierr);

  // Create a new DM and vector for scaled variable fields
  ierr = DMStagCreateCompatibleDMStag(usr->dmPV,0,2,3,0,&usr->dmscal); CHKERRQ(ierr);
  ierr = DMSetUp(usr->dmscal); CHKERRQ(ierr);
  ierr = DMStagSetUniformCoordinatesProduct(usr->dmscal,dim_param(xmin,usr->scal->h),dim_param(xmax,usr->scal->h),dim_param(zmin,usr->scal->h),dim_param(zmax,usr->scal->h),0.0,0.0);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(usr->dmscal,&usr->xscal);CHKERRQ(ierr);

  // Set initial porosity profile (t=0)
  PetscPrintf(PETSC_COMM_WORLD,"# Set initial porosity profile\n");
  ierr = FDPDEAdvDiffGetPrevSolution(fdphi,&xphiprev);CHKERRQ(ierr);
  ierr = SetInitialPorosityProfile(dmphi,xphiprev,usr);CHKERRQ(ierr);

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s_phi_initial",usr->par->fname_out,istep);
  ierr = DMStagViewBinaryPython(dmphi,xphiprev,fout);CHKERRQ(ierr);

  ierr = VecCopy(xphiprev,usr->xphiprev);CHKERRQ(ierr);
  ierr = VecDestroy(&xphiprev);CHKERRQ(ierr);

  // Solve StokesDarcy - to calculate velocities
  PetscPrintf(PETSC_COMM_WORLD,"# Set initial porosity coefficient (1) - Stokes Solve \n");
  ierr = FDPDESolve(fdPV,NULL);CHKERRQ(ierr);
  ierr = FDPDEGetSolution(fdPV,&xPV);CHKERRQ(ierr);

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s_PV_initial",usr->par->fname_out,istep);
  ierr = DMStagViewBinaryPython(dmPV,xPV,fout);CHKERRQ(ierr);

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s_PV_initial_residual",usr->par->fname_out,istep);
  ierr = DMStagViewBinaryPython(dmPV,fdPV->r,fout);CHKERRQ(ierr);

  ierr = VecCopy(xPV,usr->xPV);CHKERRQ(ierr);
  ierr = VecDestroy(&xPV);CHKERRQ(ierr);

  // Set initial coefficient structure
  PetscPrintf(PETSC_COMM_WORLD,"# Set initial porosity coefficient (2) \n");
  ierr = FDPDEGetCoefficient(fdphi,&dmphicoeff,NULL);CHKERRQ(ierr);
  ierr = FDPDEAdvDiffGetPrevCoefficient(fdphi,&phicoeffprev);CHKERRQ(ierr);
  ierr = SetInitialPorosityCoefficient(dmphicoeff,phicoeffprev,usr);CHKERRQ(ierr);
  // ierr = DMStagViewBinaryPython(dmphicoeff,phicoeffprev,"out_phicoeffprev");CHKERRQ(ierr);
  ierr = VecDestroy(&phicoeffprev);CHKERRQ(ierr);

  // Time loop
  while ((nd->t <= nd->tmax) && (istep<nd->tstep)) {
    PetscPrintf(PETSC_COMM_WORLD,"# TIMESTEP %d: \n",istep);

    // Set dt for porosity evolution 
    // ierr = FDPDEAdvDiffComputeExplicitTimestep(fdphi,&dt);CHKERRQ(ierr);
    // nd->dt = PetscMin(dt,nd->dtmax);
    nd->dt = nd->dtmax;
    ierr = FDPDEAdvDiffSetTimestep(fdphi,nd->dt);CHKERRQ(ierr);

    // Update time
    nd->tprev = nd->t;
    nd->t    += nd->dt;

    // Porosity Solver - solve for phi_new, t
    PetscPrintf(PETSC_COMM_WORLD,"# Porosity Solver \n");
    ierr = FDPDESolve(fdphi,NULL);CHKERRQ(ierr);

    // converged = PETSC_FALSE;
    // while (!converged) {
    //   ierr = FDPDESolve(fdphi,&converged);CHKERRQ(ierr);
    //   if (!converged) { // Reduce dt if not converged
    //     nd->dt *= dt_damp;
    //     ierr = FDPDEAdvDiffSetTimestep(fdphi,nd->dt);CHKERRQ(ierr);
    //   }
    // }
    ierr = FDPDEGetSolution(fdphi,&xphi);CHKERRQ(ierr);

    // Porosity: copy solution and coefficient to old
    ierr = FDPDEAdvDiffGetPrevSolution(fdphi,&xphiprev);CHKERRQ(ierr);
    ierr = VecCopy(xphi,xphiprev);CHKERRQ(ierr);
    ierr = VecCopy(xphiprev,usr->xphiprev);CHKERRQ(ierr);
    ierr = VecDestroy(&xphiprev);CHKERRQ(ierr);

    ierr = FDPDEGetCoefficient(fdphi,&dmphicoeff,&phicoeff);CHKERRQ(ierr);
    ierr = FDPDEAdvDiffGetPrevCoefficient(fdphi,&phicoeffprev);CHKERRQ(ierr);
    ierr = VecCopy(phicoeff,phicoeffprev);CHKERRQ(ierr);
    ierr = VecDestroy(&phicoeffprev);CHKERRQ(ierr);

    // StokesDarcy Solver - using phi_old, tprev
    PetscPrintf(PETSC_COMM_WORLD,"# StokesDarcy Solver \n");
    ierr = FDPDESolve(fdPV,NULL);CHKERRQ(ierr);
    ierr = FDPDEGetSolution(fdPV,&xPV);CHKERRQ(ierr);
    ierr = VecCopy(xPV,usr->xPV);CHKERRQ(ierr);

    // Output solution and calculate fluid velocity
    if (istep % nd->tout == 0 ) {
      PetscPrintf(PETSC_COMM_WORLD,"# Write data to file \n");
      ierr = PetscSNPrintf(fout,sizeof(fout),"%s_PV_nd_ts%1.3d",usr->par->fname_out,istep);
      ierr = DMStagViewBinaryPython(dmPV,xPV,fout);CHKERRQ(ierr);

      ierr = PetscSNPrintf(fout,sizeof(fout),"%s_phi_nd_ts%1.3d",usr->par->fname_out,istep);
      ierr = DMStagViewBinaryPython(dmphi,xphi,fout);CHKERRQ(ierr);

      ierr = PetscSNPrintf(fout,sizeof(fout),"%s_PV_residual_ts%1.3d",usr->par->fname_out,istep);
      ierr = DMStagViewBinaryPython(dmPV,fdPV->r,fout);CHKERRQ(ierr);

      ierr = PetscSNPrintf(fout,sizeof(fout),"%s_phi_residual_ts%1.3d",usr->par->fname_out,istep);
      ierr = DMStagViewBinaryPython(dmphi,fdphi->r,fout);CHKERRQ(ierr);

      ierr = ComputeFluidVelocity(dmPV,xPV,dmphi,xphi,&xF,usr);CHKERRQ(ierr);
      ierr = PetscSNPrintf(fout,sizeof(fout),"%s_xF_nd_ts%1.3d",usr->par->fname_out,istep);
      ierr = DMStagViewBinaryPython(dmPV,xF,fout);CHKERRQ(ierr);

      ierr = ScaleParametersAndOutput(dmPV,xPV,dmphi,xphi,xF,usr);CHKERRQ(ierr);
      ierr = PetscSNPrintf(fout,sizeof(fout),"%s_dim_sol_ts%1.3d",usr->par->fname_out,istep);
      ierr = DMStagViewBinaryPython(usr->dmscal,usr->xscal,fout);CHKERRQ(ierr);

      ierr = VecDestroy(&xF);CHKERRQ(ierr);
    }

    // Clean up
    ierr = VecDestroy(&xPV);CHKERRQ(ierr);
    ierr = VecDestroy(&xphi);CHKERRQ(ierr);

    // increment timestep
    istep++;

    PetscPrintf(PETSC_COMM_WORLD,"# TIME: time = %1.12e [yr] dt = %1.12e [yr] \n",nd->t*usr->scal->t/SEC_YEAR,nd->dt*usr->scal->t/SEC_YEAR);
    PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");
  }

  // Destroy objects
  ierr = VecDestroy(&xmms_PV);CHKERRQ(ierr);
  ierr = VecDestroy(&xmms_phi);CHKERRQ(ierr);
  ierr = DMDestroy(&dmPV);CHKERRQ(ierr);
  ierr = DMDestroy(&dmphi);CHKERRQ(ierr);
  ierr = DMDestroy(&usr->dmscal);CHKERRQ(ierr);
  ierr = FDPDEDestroy(&fdPV);CHKERRQ(ierr);
  ierr = FDPDEDestroy(&fdphi);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xPV);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xphiprev);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xscal);CHKERRQ(ierr);

  PetscFunctionReturn(0);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Allocate memory to application context
  ierr = PetscMalloc1(1, &usr); CHKERRQ(ierr);

  // Get time, comm and rank
  usr->comm = PETSC_COMM_WORLD;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &usr->rank); CHKERRQ(ierr);

  // Create bag
  ierr = PetscBagCreate (usr->comm,sizeof(Params),&usr->bag); CHKERRQ(ierr);
  ierr = PetscBagGetData(usr->bag,(void **)&usr->par); CHKERRQ(ierr);
  ierr = PetscBagSetName(usr->bag,"UserParamBag","- User defined parameters -"); CHKERRQ(ierr);

  // Define some pointers for easy access
  bag = usr->bag;
  par = usr->par;

  // Initialize domain variables
  ierr = PetscBagRegisterInt(bag, &par->nx, 10, "nx", "Element count in the x-dir [-]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->nz, 10, "nz", "Element count in the z-dir [-]"); CHKERRQ(ierr);

  ierr = PetscBagRegisterScalar(bag, &par->xmin, 0.0, "xmin", "Start coordinate of domain in x-dir [m]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->zmin, -100.0e3, "zmin", "Start coordinate of domain in z-dir [m]"); CHKERRQ(ierr);

  ierr = PetscBagRegisterScalar(bag, &par->L, 200.0e3, "L", "Length of domain in x-dir [m]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->H, 100.0e3, "H", "Height of domain in z-dir [m]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->xMOR, 200.0e3, "xMOR", "Distance from mid-ocean ridge axis for melt extraction [m]"); CHKERRQ(ierr);

  // Physical and material parameters
  ierr = PetscBagRegisterScalar(bag, &par->k_hat, 0.0, "k_hat", "Direction of unit vertical vector [-]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->g, 10.0, "g", "Gravitational acceleration [m^2/s]"); CHKERRQ(ierr);

  ierr = PetscBagRegisterScalar(bag, &par->phi0, 0.01, "phi0", "Reference porosity [-]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->K0, 1.0e-7, "K0", "Reference permeability [m^2]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->u0, 2.0, "u0", "Half-spreading rate [cm/yr]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->drho, 500.0, "drho", "Density difference [kg/m^3]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->rho0, 3000.0, "rho0", "Reference density [kg/m^3]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->n, 3.0, "n", "Porosity exponent"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mu, 1.0, "mu", "Fluid viscosity [Pa.s]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->eta, 1.0e18, "eta", "Shear viscosity [Pa.s]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->zeta, 1.0e20, "zeta", "Bulk viscosity [Pa.s]"); CHKERRQ(ierr);

  // Time stepping and advection parameters
  ierr = PetscBagRegisterInt(bag, &par->ts_scheme,2, "ts_scheme", "Time stepping scheme 0-forward euler, 1-backward euler, 2-crank-nicholson"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->adv_scheme,1, "adv_scheme", "Advection scheme 0-upwind, 1-fromm"); CHKERRQ(ierr);

  ierr = PetscBagRegisterInt(bag, &par->tout,1, "tout", "Output every tout time step"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->tstep,1, "tstep", "Maximum no of time steps"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->tmax, 1.0e6, "tmax", "Maximum time [yr]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->dtmax, 1.0e3, "dtmax", "Maximum time step size [yr]"); CHKERRQ(ierr);

  // Input/output 
  ierr = PetscBagRegisterString(bag,&par->fname_out,FNAME_LENGTH,"out_solution","output_file","Name for output file, set with: -output_file <filename>"); CHKERRQ(ierr);

  // return pointer
  *_usr = usr;

  PetscFunctionReturn(0);
}

// ---------------------------------------
// InputPrintData
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "InputPrintData"
PetscErrorCode InputPrintData(UsrData *usr)
{
  char           date[30], *opts;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Get date
  ierr = PetscGetDate(date,30); CHKERRQ(ierr);
  ierr = PetscOptionsGetAll(NULL, &opts); CHKERRQ(ierr);

  // Print header and petsc options
  PetscPrintf(usr->comm,"# --------------------------------------- #\n");
  PetscPrintf(usr->comm,"# MID-OCEAN RIDGE Model (MECHANICS): %s \n",&(date[0]));
  PetscPrintf(usr->comm,"# --------------------------------------- #\n");
  PetscPrintf(usr->comm,"# PETSc options: %s \n",opts);
  PetscPrintf(usr->comm,"# --------------------------------------- #\n");

  // Print usr bag
  ierr = PetscBagView(usr->bag,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
  PetscPrintf(usr->comm,"# --------------------------------------- #\n");

  // Free memory
  ierr = PetscFree(opts); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// DefineScalingParameters - define scales
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "DefineScalingParameters"
PetscErrorCode DefineScalingParameters(UsrData *usr)
{
  ScalParams     *scal;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Allocate memory
  ierr = PetscMalloc1(1, &scal); CHKERRQ(ierr);

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

  PetscFunctionReturn(0);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Allocate memory
  ierr = PetscMalloc1(1, &nd); CHKERRQ(ierr);

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

  PetscFunctionReturn(0);
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
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  // Get dm and solution vector for porosity
  dmphi = usr->dmphi;
  xphi  = usr->xphiprev;

  nd   = usr->nd;
  scal = usr->scal;

  ierr = DMCreateLocalVector(dmphi,&xphilocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dmphi,xphi,INSERT_VALUES,xphilocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dmphi,xphi,INSERT_VALUES,xphilocal);CHKERRQ(ierr);

  // Get dmcoeff
  ierr = DMStagGetGlobalSizes(dmcoeff,&Nx,&Nz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmcoeff,LEFT,&iprev);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmcoeff,RIGHT,&inext);CHKERRQ(ierr); 
  ierr = DMStagGetProductCoordinateLocationSlot(dmcoeff,ELEMENT,&icenter);CHKERRQ(ierr);

  // Create coefficient local vector
  ierr = DMCreateLocalVector(dmcoeff, &coefflocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dmcoeff, coefflocal, &c); CHKERRQ(ierr);
  
  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      PetscInt idx;

      { // A = delta^2*eta (center, c=1)
        DMStagStencil point;
        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 1;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
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
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = scal->delta*scal->delta*nd->eta;
        }
      }

      { // B = -phi*k_hat (edges, c=0)
        DMStagStencil point[4], pointQ[3];
        PetscScalar   Q[3], Qinterp;
        PetscScalar   xp[4],zp[4],rhs[4],zQ[3];
        PetscInt      ii;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 0;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = 0;
        point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = 0;

        xp[0] = coordx[i][iprev  ]; zp[0] = coordz[j][icenter];
        xp[1] = coordx[i][inext  ]; zp[1] = coordz[j][icenter];
        xp[2] = coordx[i][icenter]; zp[2] = coordz[j][iprev  ];
        xp[3] = coordx[i][icenter]; zp[3] = coordz[j][inext  ];

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

        ierr = DMStagVecGetValuesStencil(dmphi,xphilocal,3,pointQ,Q); CHKERRQ(ierr);
        Qinterp = interp1DLin_3Points(zp[2],zQ[0],Q[0],zQ[1],Q[1],zQ[2],Q[2]); // Q = 1-phi
        rhs[2]  = nd->k_hat*(1.0-Qinterp);

        Qinterp = interp1DLin_3Points(zp[3],zQ[0],Q[0],zQ[1],Q[1],zQ[2],Q[2]);
        rhs[3]  = nd->k_hat*(1.0-Qinterp);

        for (ii = 0; ii < 4; ii++) {
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = rhs[ii];
        }
      }

      { // C = 0.0 (center, c=0)
        DMStagStencil point;
        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 0;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = 0.0;
      }

      { // D1 = delta^2*xi (center, c=2)
        DMStagStencil point;
        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 2;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
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

        ierr = DMStagVecGetValuesStencil(dmphi,xphilocal,5,pointQ,Q); CHKERRQ(ierr);

        Qinterp = interp1DLin_3Points(xp[0],xQ[0],Q[0],xQ[1],Q[1],xQ[2],Q[2]); // Q = 1-phi
        rhs[0] = -PetscPowScalar((1.0-Qinterp)/nd->phi0,nd->n); //left 

        Qinterp = interp1DLin_3Points(xp[1],xQ[0],Q[0],xQ[1],Q[1],xQ[2],Q[2]);
        rhs[1] = -PetscPowScalar((1.0-Qinterp)/nd->phi0,nd->n); //right

        Qinterp = interp1DLin_3Points(zp[2],zQ[0],Q[3],zQ[1],Q[1],zQ[2],Q[4]);
        rhs[2] = -PetscPowScalar((1.0-Qinterp)/nd->phi0,nd->n); // down

        Qinterp = interp1DLin_3Points(zp[3],zQ[0],Q[3],zQ[1],Q[1],zQ[2],Q[4]);
        rhs[3] = -PetscPowScalar((1.0-Qinterp)/nd->phi0,nd->n); // up

        for (ii = 0; ii < 4; ii++) {
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
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

        ierr = DMStagVecGetValuesStencil(dmphi,xphilocal,3,pointQ,Q); CHKERRQ(ierr);
        Qinterp = interp1DLin_3Points(zp[2],zQ[0],Q[0],zQ[1],Q[1],zQ[2],Q[2]); // Q = 1-phi
        rhs[2]  = nd->k_hat*PetscPowScalar((1.0-Qinterp)/nd->phi0,nd->n);

        Qinterp = interp1DLin_3Points(zp[3],zQ[0],Q[0],zQ[1],Q[1],zQ[2],Q[2]);
        rhs[3]  = nd->k_hat*PetscPowScalar((1.0-Qinterp)/nd->phi0,nd->n); 

        for (ii = 0; ii < 4; ii++) {
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = rhs[ii];
        }
      }
    }
  }

  // Restore arrays, local vectors
  ierr = DMStagRestoreProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL);CHKERRQ(ierr);

  ierr = DMStagVecRestoreArray(dmcoeff,coefflocal,&c);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  
  ierr = VecDestroy(&coefflocal); CHKERRQ(ierr);
  ierr = VecDestroy(&xphilocal);CHKERRQ(ierr);

  PetscFunctionReturn(0);
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
  PetscErrorCode ierr;
  
  PetscFunctionBegin;

  // Reduced form: zero normal fluxes are implicitly integrated (i.e. gradP=0, dVxi/dxj=0).
  // No need to explicitly specify them, unless want to overwrite.

  // LEFT Boundary: vx = 0, partial_x(vz,P) = 0 (reduced form)
  // LEFT Vx = 0 
  ierr = DMStagBCListGetValues(bclist,'w','-',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // TOP Boundary: vx = u0, vz = 0, partial_z(P) = 0 (reduced form)
  // UP Vx = u0
  ierr = DMStagBCListGetValues(bclist,'n','-',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
      value_bc[k] = usr->nd->u0;
      type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP Vz = 0
  ierr = DMStagBCListGetValues(bclist,'n','|',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // RIGHT Boundary: partial_x(vx,vz) = 0, P = 0
  // RIGHT dVx/dx = 0
  ierr = DMStagBCListGetValues(bclist,'e','-',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN_T;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // RIGHT vz=0
  ierr = DMStagBCListGetValues(bclist,'e','|',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // RIGHT P = 0
  ierr = DMStagBCListGetValues(bclist,'e','o',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // Bottom Boundary: partial_z(vx,vz) = 0, P = 0
  // DOWN vx = 0 
  ierr = DMStagBCListGetValues(bclist,'s','-',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN dVz/dz = 0
  ierr = DMStagBCListGetValues(bclist,'s','|',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN_T;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN P = 0
  ierr = DMStagBCListGetValues(bclist,'s','o',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  PetscFunctionReturn(0);
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
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  // Get dm and solution vector for Stokes velocity
  dmPV = usr->dmPV;
  xPV  = usr->xPV;
  
  ierr = DMCreateLocalVector(dmPV,&xPVlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dmPV,xPV,INSERT_VALUES,xPVlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dmPV,xPV,INSERT_VALUES,xPVlocal);CHKERRQ(ierr);
  
  // Get domain corners and coordinates
  ierr = DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = DMStagGetGlobalSizes(dmcoeff,&Nx,&Nz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmcoeff,ELEMENT,&icenter);CHKERRQ(ierr);

  // Create coefficient local vector
  ierr = DMCreateLocalVector(dmcoeff, &coefflocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dmcoeff, coefflocal, &c); CHKERRQ(ierr);
  
  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {

      { // A = 1.0
        DMStagStencil point;
        PetscInt      idx;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 0;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = 1.0;
      }

      { // C = Gamma
        DMStagStencil point;
        PetscInt      idx;
        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 1;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
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
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
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
        
        ierr = DMStagVecGetValuesStencil(dmPV,xPVlocal,4,point,v); CHKERRQ(ierr);

        for (ii = 0; ii < 4; ii++) {
          point[ii].c = 1;
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = v[ii];
        }
      }
    }
  }

  // Restore arrays, local vectors
  ierr = DMStagRestoreProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArray(dmcoeff,coefflocal,&c);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  ierr = VecDestroy(&coefflocal); CHKERRQ(ierr);
  
  ierr = VecDestroy(&xPVlocal);CHKERRQ(ierr);

  PetscFunctionReturn(0);
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
  PetscErrorCode ierr;
  PetscFunctionBegin;
  
  // Left = dphi/dx = d(1-phi)/dx = 0
  ierr = DMStagBCListGetValues(bclist,'w','o',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // RIGHT:
  ierr = DMStagBCListGetValues(bclist,'e','o',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN: phi = phi0
  ierr = DMStagBCListGetValues(bclist,'s','o',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 1.0 - usr->nd->phi0;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP: dphi/dz = 0 if x<=xMOR, phi=0 otherwise
  ierr = DMStagBCListGetValues(bclist,'n','o',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    if (x_bc[2*k]<=usr->nd->xMOR) {
      value_bc[k] = 0.0;
      type_bc[k] = BC_NEUMANN;
    } else {
      value_bc[k] = 1.0;
      type_bc[k] = BC_DIRICHLET;
    }
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
 
  PetscFunctionReturn(0);
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
  PetscErrorCode ierr;
  PetscFunctionBegin;

  // Get domain corners
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  // Get dm coordinates array
  ierr = DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter);CHKERRQ(ierr);

  // Create local vector
  ierr = DMCreateLocalVector(dm, &xlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm, xlocal, &xx); CHKERRQ(ierr);

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      DMStagStencil point;
      PetscInt      idx;
      PetscScalar   phi;

      point.i = i; point.j = j; point.loc = ELEMENT; point.c = 0;
      ierr = DMStagGetLocationSlot(dm, point.loc, point.c, &idx); CHKERRQ(ierr);
      phi = usr->nd->phi0;// + usr->nd->phi0*usr->nd->phi0*PetscCosScalar(2.0*PETSC_PI*coordx[i][icenter])*PetscCosScalar(2.0*PETSC_PI*coordz[j][icenter]);
      xx[j][i][idx] = 1.0-phi; //(1.0-usr->nd->phi0);
    }
  }

  // Restore arrays, local vectors
  ierr = DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);

  ierr = DMStagVecRestoreArray(dm,xlocal,&xx);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm,xlocal,INSERT_VALUES,x); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,xlocal,INSERT_VALUES,x); CHKERRQ(ierr);
  
  ierr = VecDestroy(&xlocal); CHKERRQ(ierr);

  PetscFunctionReturn(0);
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
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  // Get dm and solution vector for Stokes velocity
  dmPV = usr->dmPV;
  xPV  = usr->xPV;
  
  ierr = DMCreateLocalVector(dmPV,&xPVlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dmPV,xPV,INSERT_VALUES,xPVlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dmPV,xPV,INSERT_VALUES,xPVlocal);CHKERRQ(ierr);

  // Get domain corners
  ierr = DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  ierr = DMStagGetProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmcoeff,LEFT,&iprev);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmcoeff,RIGHT,&inext);CHKERRQ(ierr); 
  ierr = DMStagGetProductCoordinateLocationSlot(dmcoeff,ELEMENT,&icenter);CHKERRQ(ierr);

  // Create coefficient local vector
  ierr = DMCreateLocalVector(dmcoeff, &coefflocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dmcoeff, coefflocal, &c); CHKERRQ(ierr);
  
  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {

      { // A = 1.0
        DMStagStencil point;
        PetscInt      idx;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 0;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = 1.0;
      }

      { // C = Gamma
        DMStagStencil point;
        PetscInt      idx;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 1;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
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
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, 0, &idx); CHKERRQ(ierr);
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
        
        ierr = DMStagVecGetValuesStencil(dmPV,xPVlocal,4,point,v); CHKERRQ(ierr);

        for (ii = 0; ii < 4; ii++) {
          point[ii].c = 1;
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = v[ii];
        }
      }
    }
  }

  // Restore arrays, local vectors
  ierr = DMStagRestoreProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArray(dmcoeff,coefflocal,&c);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  ierr = VecDestroy(&coefflocal); CHKERRQ(ierr);
  ierr = VecDestroy(&xPVlocal);CHKERRQ(ierr);

  PetscFunctionReturn(0);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Create local and global vector for fluid velocity
  ierr = DMCreateGlobalVector(dmPV,&xF     ); CHKERRQ(ierr);
  ierr = DMCreateLocalVector (dmPV,&xFlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dmPV,xFlocal,&xxF); CHKERRQ(ierr);

  ierr = DMCreateLocalVector(dmphi,&xphilocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dmphi,xphi,INSERT_VALUES,xphilocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dmphi,xphi,INSERT_VALUES,xphilocal);CHKERRQ(ierr);

  ierr = DMCreateLocalVector(dmPV,&xPVlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dmPV,xPV,INSERT_VALUES,xPVlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dmPV,xPV,INSERT_VALUES,xPVlocal);CHKERRQ(ierr);

  // Get domain corners and coordinates
  ierr = DMStagGetGlobalSizes(dmPV, &Nx, &Nz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dmPV, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateArraysRead(dmPV,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmPV,ELEMENT,&icenter);CHKERRQ(ierr); 
  ierr = DMStagGetProductCoordinateLocationSlot(dmPV,LEFT,&iprev);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmPV,RIGHT,&inext);CHKERRQ(ierr); 

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      DMStagStencil point[9],pointQ[5];
      PetscScalar xx[9],Q[5],Qinterp[4],xp[3],zp[3],dx,dz,phi,gradP;

      // center points - dummy
      ierr = DMStagGetLocationSlot(dmPV,ELEMENT,0,&idx); CHKERRQ(ierr);
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

      ierr = DMStagVecGetValuesStencil(dmPV,xPVlocal,9,point,xx); CHKERRQ(ierr);
      
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

      ierr = DMStagVecGetValuesStencil(dmphi,xphilocal,5,pointQ,Q); CHKERRQ(ierr);

      // porosity on edges
      Qinterp[0] = interp1DLin_3Points(coordx[i][iprev],xp[0],Q[1],xp[1],Q[0],xp[2],Q[2]); 
      Qinterp[1] = interp1DLin_3Points(coordx[i][inext],xp[0],Q[1],xp[1],Q[0],xp[2],Q[2]); 
      Qinterp[2] = interp1DLin_3Points(coordz[j][iprev],zp[0],Q[3],zp[1],Q[0],zp[2],Q[4]); 
      Qinterp[3] = interp1DLin_3Points(coordz[j][inext],zp[0],Q[3],zp[1],Q[0],zp[2],Q[4]); 

      // left
      phi   = 1.0 - Qinterp[0];
      gradP = (xx[4]-xx[5])/dx;
      ierr  = DMStagGetLocationSlot(dmPV,LEFT,0,&idx); CHKERRQ(ierr);
      xxF[j][i][idx] = fluid_velocity(xx[0],phi,usr->nd->phi0,usr->nd->n,gradP,0.0);

      // right
      phi   = 1.0 - Qinterp[1];
      gradP = (xx[6]-xx[4])/dx;
      ierr  = DMStagGetLocationSlot(dmPV,RIGHT,0,&idx); CHKERRQ(ierr);
      xxF[j][i][idx] = fluid_velocity(xx[1],phi,usr->nd->phi0,usr->nd->n,gradP,0.0);

      // down
      phi   = 1.0 - Qinterp[2];
      gradP = (xx[4]-xx[7])/dz;
      ierr  = DMStagGetLocationSlot(dmPV,DOWN,0,&idx); CHKERRQ(ierr);
      xxF[j][i][idx] = fluid_velocity(xx[2],phi,usr->nd->phi0,usr->nd->n,gradP,usr->nd->k_hat);

      // up
      phi   = 1.0 - Qinterp[3];
      gradP = (xx[8]-xx[4])/dz;
      ierr  = DMStagGetLocationSlot(dmPV,UP,0,&idx); CHKERRQ(ierr);
      xxF[j][i][idx] = fluid_velocity(xx[3],phi,usr->nd->phi0,usr->nd->n,gradP,usr->nd->k_hat);
    }
  }

  // Restore arrays
  ierr = DMStagRestoreProductCoordinateArraysRead(dmPV,&coordx,&coordz,NULL);CHKERRQ(ierr);

  // Restore and map local to global
  ierr = DMStagVecRestoreArray(dmPV,xFlocal,&xxF); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dmPV,xFlocal,INSERT_VALUES,xF); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dmPV,xFlocal,INSERT_VALUES,xF); CHKERRQ(ierr);
  ierr = VecDestroy(&xFlocal); CHKERRQ(ierr);
  ierr = VecDestroy(&xPVlocal); CHKERRQ(ierr);
  ierr = VecDestroy(&xphilocal);CHKERRQ(ierr);

  // Assign pointers
  *_xF  = xF;
  
  PetscFunctionReturn(0);
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
  PetscErrorCode ierr;
  PetscFunctionBegin;

  // Get domain corners
  dm = usr->dmscal;
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  // Get dm coordinates array
  ierr = DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter);CHKERRQ(ierr);

  // Create local vector
  ierr = DMCreateLocalVector(dm, &xlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm, xlocal, &xx); CHKERRQ(ierr);

  // Get solution vectors
  ierr = DMCreateLocalVector(dmPV,&xPVlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dmPV,xPV,INSERT_VALUES,xPVlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dmPV,xPV,INSERT_VALUES,xPVlocal);CHKERRQ(ierr);

  ierr = DMCreateLocalVector(dmPV,&xFlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dmPV,xF,INSERT_VALUES,xFlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dmPV,xF,INSERT_VALUES,xFlocal);CHKERRQ(ierr);

  ierr = DMCreateLocalVector(dmphi,&xphilocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dmphi,xphi,INSERT_VALUES,xphilocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dmphi,xphi,INSERT_VALUES,xphilocal);CHKERRQ(ierr);

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

      ierr = DMStagVecGetValuesStencil(dmPV,xPVlocal,5,point,xval); CHKERRQ(ierr);
      ierr = DMStagGetLocationSlot(dm, point[0].loc, 0, &idx); CHKERRQ(ierr);
      xx[j][i][idx] = dim_param(xval[0],usr->scal->P);

      for (ii = 1; ii < 5; ii++) {
        ierr = DMStagGetLocationSlot(dm, point[ii].loc, 0, &idx); CHKERRQ(ierr);
        xx[j][i][idx] = dim_param(xval[ii],usr->scal->v);
      }

      // xF
      point[0].i = i; point[0].j = j; point[0].loc = LEFT;    point[0].c = 0;
      point[1].i = i; point[1].j = j; point[1].loc = RIGHT;   point[1].c = 0;
      point[2].i = i; point[2].j = j; point[2].loc = DOWN;    point[2].c = 0;
      point[3].i = i; point[3].j = j; point[3].loc = UP;      point[3].c = 0;

      ierr = DMStagVecGetValuesStencil(dmPV,xFlocal,4,point,xval); CHKERRQ(ierr);
      for (ii = 0; ii < 4; ii++) {
        ierr = DMStagGetLocationSlot(dm, point[ii].loc, 1, &idx); CHKERRQ(ierr);
        xx[j][i][idx] = dim_param(xval[ii],usr->scal->v);
      }

      // phi
      point[0].i = i; point[0].j = j; point[0].loc = ELEMENT; point[0].c = 0;
      ierr = DMStagVecGetValuesStencil(dmphi,xphilocal,1,point,xval); CHKERRQ(ierr);
      ierr = DMStagGetLocationSlot(dm, point[0].loc, 1, &idx); CHKERRQ(ierr);
      xx[j][i][idx] = dim_param(xval[0],usr->scal->P);

      // Plith
      point[0].i = i; point[0].j = j; point[0].loc = ELEMENT; point[0].c = 2;
      ierr = DMStagGetLocationSlot(dm, point[0].loc, 2, &idx); CHKERRQ(ierr);
      xx[j][i][idx] = -usr->scal->rho*usr->par->g*coordz[j][icenter];
    }
  }

  // Restore arrays, local vectors
  ierr = DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);

  ierr = DMStagVecRestoreArray(dm,xlocal,&xx);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm,xlocal,INSERT_VALUES,usr->xscal); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,xlocal,INSERT_VALUES,usr->xscal); CHKERRQ(ierr);

  ierr = VecDestroy(&xlocal); CHKERRQ(ierr);
  ierr = VecDestroy(&xphilocal);CHKERRQ(ierr);
  ierr = VecDestroy(&xPVlocal);CHKERRQ(ierr);
  ierr = VecDestroy(&xFlocal);CHKERRQ(ierr);

  PetscFunctionReturn(0);
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
  PetscErrorCode  ierr;
    
  // Initialize application
  ierr = PetscInitialize(&argc,&argv,(char*)0,help); if (ierr) return ierr;
 
  // Load command line or input file if required
  ierr = PetscOptionsInsert(PETSC_NULL,&argc,&argv,NULL); CHKERRQ(ierr);

  // Input user parameters and print
  ierr = InputParameters(&usr); CHKERRQ(ierr);
  ierr = InputPrintData(usr); CHKERRQ(ierr);

  // Define scaling parameters and non-dimensionalize
  ierr = DefineScalingParameters(usr); CHKERRQ(ierr);
  ierr = NondimensionalizeParameters(usr);CHKERRQ(ierr);

  // Numerical solution using the FD pde object
  ierr = PetscTime(&start_time); CHKERRQ(ierr);
  ierr = Numerical_solution(usr); CHKERRQ(ierr);
  ierr = PetscTime(&end_time); CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"# Runtime: %g (sec) \n", end_time - start_time);
  PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");

  // Destroy objects
  ierr = PetscFree(usr->nd); CHKERRQ(ierr);
  ierr = PetscFree(usr->scal); CHKERRQ(ierr);
  ierr = PetscBagDestroy(&usr->bag); CHKERRQ(ierr);
  ierr = PetscFree(usr); CHKERRQ(ierr);

  // Finalize main
  ierr = PetscFinalize();
  return ierr;
}