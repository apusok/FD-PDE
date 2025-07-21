// ---------------------------------------
// MMS test for porosity evolution - verify coupled system for two-phase flow 
// Solves for coupled (P, v) and Q=(1-phi) evolution, where P-dynamic pressure, v-solid velocity, phi-porosity.
// run: ./test_stokesdarcy2field_mms_porosity.sh -pc_type lu -pc_factor_mat_solver_type umfpack -pc_factor_mat_ordering_type external -nx 20 -nz 20 -snes_monitor 
// python test: ./python/test_stokesdarcy2field_mms_porosity.py
// sympy: ./mms/mms_porosity_evolution.py
// ---------------------------------------
static char help[] = "Application to verify the Stokes-Darcy and porosity evolution using MMS\n\n";

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

#include "../src/fdpde_stokesdarcy2field.h"
#include "../src/fdpde_advdiff.h"

// ---------------------------------------
// Application Context
// ---------------------------------------
#define FNAME_LENGTH 200

// parameters (bag)
typedef struct {
  PetscInt       nx, nz;
  PetscScalar    L, H;
  PetscScalar    xmin, zmin;
  PetscScalar    phi_0,p_s,m,n,e3,eta,zeta,xi;
  PetscInt       ts_scheme, adv_scheme, tout, tstep;
  PetscScalar    t, dt, tmax, dtmax, tprev;
  char           fname_out[FNAME_LENGTH]; 
  char           fdir_out[FNAME_LENGTH];
} Params;

// user defined and model-dependent variables
typedef struct {
  Params        *par;
  PetscBag       bag;
  MPI_Comm       comm;
  PetscMPIInt    rank;
  Vec            xphiprev,xPV;
  DM             dmphi,dmPV;
} UsrData;

// ---------------------------------------
// Function definitions
// ---------------------------------------
PetscErrorCode Numerical_solution(void*);
PetscErrorCode ComputeManufacturedSolutionTimestep(DM,Vec*,DM,Vec*,void*);
PetscErrorCode ComputeErrorNorms(DM,Vec,Vec,DM,Vec,Vec);
PetscErrorCode InputParameters(UsrData**);
PetscErrorCode InputPrintData(UsrData*);
PetscErrorCode FormCoefficient_PV(FDPDE, DM, Vec, DM, Vec, void*);
PetscErrorCode FormCoefficient_phi(FDPDE, DM, Vec, DM, Vec, void*);
PetscErrorCode FormBCList_PV(DM, Vec, DMStagBCList, void*);
PetscErrorCode FormBCList_phi(DM, Vec, DMStagBCList, void*);
PetscErrorCode SetInitialPorosityProfile(DM,Vec,void*);
PetscErrorCode SetInitialPorosityCoefficient(DM,Vec,void*);

const char coeff_description_stokesdarcy[] =
"  << Stokes-Darcy Coefficients >> \n"
"  A = eta, eta = 1 (constant viscosity)  \n"
"  B = -phi*e3 + fu (fux,fuz - manufactured)\n" 
"  C = fp (manufactured) \n"
"  D1 = xi, xi = zeta-2/3eta \n"
"  D2 = -Kphi \n"
"  D3 = Kphi*e3, Kphi = (phi/phi0)^n \n";

const char bc_description_stokesdarcy[] =
"  << Stokes-Darcy BCs >> \n"
"  LEFT, RIGHT, DOWN, UP: DIRICHLET (manufactured) \n";

const char coeff_description_phi[] =
"  << Porosity Coefficients (dimensionless) >> \n"
"  A = 1.0 \n"
"  B = 0 \n"
"  C = 0-fphi (manufactured)\n"
"  u = [ux, uz] - StokesDarcy solid velocity \n";

const char bc_description_phi[] =
"  << Porosity BCs >> \n"
"  LEFT, RIGHT, DOWN, UP: DIRICHLET (manufactured) \n";

// ---------------------------------------
// Useful functions
// ---------------------------------------
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
// Manufactured solutions
// ---------------------------------------
static PetscScalar get_p(PetscScalar x, PetscScalar z, PetscScalar t, PetscScalar eta, PetscScalar zeta, PetscScalar phi_0, PetscScalar p_s, PetscScalar m, PetscScalar n, PetscScalar e3)
{ PetscScalar result;
  result = p_s*cos(M_PI*m*x)*cos(M_PI*m*z);
  return(result);
}
static PetscScalar get_ux(PetscScalar x, PetscScalar z, PetscScalar t, PetscScalar eta, PetscScalar zeta, PetscScalar phi_0, PetscScalar p_s, PetscScalar m, PetscScalar n, PetscScalar e3)
{ PetscScalar result;
  result = M_PI*m*(1.0 - 1.0*cos(M_PI*m*x))*sin(M_PI*m*z) + 1.0*M_PI*m*sin(M_PI*m*x)*cos(M_PI*m*z);
  return(result);
}
static PetscScalar get_uz(PetscScalar x, PetscScalar z, PetscScalar t, PetscScalar eta, PetscScalar zeta, PetscScalar phi_0, PetscScalar p_s, PetscScalar m, PetscScalar n, PetscScalar e3)
{ PetscScalar result;
  result = -1.0*M_PI*m*(1.0 - cos(M_PI*m*z))*sin(M_PI*m*x) + 1.0*M_PI*m*sin(M_PI*m*z)*cos(M_PI*m*x);
  return(result);
}
static PetscScalar get_phi(PetscScalar x, PetscScalar z, PetscScalar t, PetscScalar eta, PetscScalar zeta, PetscScalar phi_0, PetscScalar p_s, PetscScalar m, PetscScalar n, PetscScalar e3)
{ PetscScalar result;
  result = -pow(t, 3)*(pow(x, 2) + pow(z, 2)) + 1.0;
  return(result);
}
static PetscScalar get_fux(PetscScalar x, PetscScalar z, PetscScalar t, PetscScalar eta, PetscScalar zeta, PetscScalar phi_0, PetscScalar p_s, PetscScalar m, PetscScalar n, PetscScalar e3)
{ PetscScalar result;
  result = eta*(-pow(M_PI, 3)*pow(m, 3)*(1.0 - 1.0*cos(M_PI*m*x))*sin(M_PI*m*z) - 4.0*pow(M_PI, 3)*pow(m, 3)*sin(M_PI*m*x)*cos(M_PI*m*z) + 1.0*pow(M_PI, 3)*pow(m, 3)*sin(M_PI*m*z)*cos(M_PI*m*x)) - 2.0*pow(M_PI, 3)*pow(m, 3)*(-0.66666666666666663*eta + zeta)*sin(M_PI*m*x)*cos(M_PI*m*z) + M_PI*m*p_s*sin(M_PI*m*x)*cos(M_PI*m*z);
  return(result);
}
static PetscScalar get_fuz(PetscScalar x, PetscScalar z, PetscScalar t, PetscScalar eta, PetscScalar zeta, PetscScalar phi_0, PetscScalar p_s, PetscScalar m, PetscScalar n, PetscScalar e3)
{ PetscScalar result;
  result = e3*(-pow(t, 3)*(pow(x, 2) + pow(z, 2)) + 1.0) + eta*(1.0*pow(M_PI, 3)*pow(m, 3)*(1.0 - cos(M_PI*m*z))*sin(M_PI*m*x) - 1.0*pow(M_PI, 3)*pow(m, 3)*sin(M_PI*m*x)*cos(M_PI*m*z) - 4.0*pow(M_PI, 3)*pow(m, 3)*sin(M_PI*m*z)*cos(M_PI*m*x)) - 2.0*pow(M_PI, 3)*pow(m, 3)*(-0.66666666666666663*eta + zeta)*sin(M_PI*m*z)*cos(M_PI*m*x) + M_PI*m*p_s*sin(M_PI*m*z)*cos(M_PI*m*x);
  return(result);
}
static PetscScalar get_fp(PetscScalar x, PetscScalar z, PetscScalar t, PetscScalar eta, PetscScalar zeta, PetscScalar phi_0, PetscScalar p_s, PetscScalar m, PetscScalar n, PetscScalar e3)
{ PetscScalar result;
  result = 2*pow(M_PI, 2)*pow(m, 2)*p_s*pow((-pow(t, 3)*(pow(x, 2) + pow(z, 2)) + 1.0)/phi_0, n)*cos(M_PI*m*x)*cos(M_PI*m*z) + 2.0*pow(M_PI, 2)*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) - 2*M_PI*m*n*p_s*pow(t, 3)*x*pow((-pow(t, 3)*(pow(x, 2) + pow(z, 2)) + 1.0)/phi_0, n)*sin(M_PI*m*x)*cos(M_PI*m*z)/(-pow(t, 3)*(pow(x, 2) + pow(z, 2)) + 1.0) + 2*n*pow(t, 3)*z*pow((-pow(t, 3)*(pow(x, 2) + pow(z, 2)) + 1.0)/phi_0, n)*(-e3 - M_PI*m*p_s*sin(M_PI*m*z)*cos(M_PI*m*x))/(-pow(t, 3)*(pow(x, 2) + pow(z, 2)) + 1.0);
  return(result);
}
static PetscScalar get_fphi(PetscScalar x, PetscScalar z, PetscScalar t, PetscScalar eta, PetscScalar zeta, PetscScalar phi_0, PetscScalar p_s, PetscScalar m, PetscScalar n, PetscScalar e3)
{ PetscScalar result;
  result = 2*pow(t, 3)*x*(M_PI*m*(1.0 - 1.0*cos(M_PI*m*x))*sin(M_PI*m*z) + 1.0*M_PI*m*sin(M_PI*m*x)*cos(M_PI*m*z)) + 2*pow(t, 3)*z*(-1.0*M_PI*m*(1.0 - cos(M_PI*m*z))*sin(M_PI*m*x) + 1.0*M_PI*m*sin(M_PI*m*z)*cos(M_PI*m*x)) + pow(t, 3)*(pow(x, 2) + pow(z, 2))*(-1.0*pow(M_PI, 2)*pow(m, 2)*sin(M_PI*m*x)*sin(M_PI*m*z) + 1.0*pow(M_PI, 2)*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z)) + pow(t, 3)*(pow(x, 2) + pow(z, 2))*(1.0*pow(M_PI, 2)*pow(m, 2)*sin(M_PI*m*x)*sin(M_PI*m*z) + 1.0*pow(M_PI, 2)*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z)) + 3*pow(t, 2)*(pow(x, 2) + pow(z, 2));
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
  FDPDE          fdPV,fdphi;
  DM             dmPV, dmphi, dmphicoeff;
  Vec            xPV,xphi,xmms_PV,xmms_phi,xphiprev,phicoeffprev,phicoeff;
  PetscInt       nx, nz, istep = 0;
  PetscScalar    xmin, zmin, xmax, zmax;
  char           fout[FNAME_LENGTH];
  PetscFunctionBeginUser;

  // Element count
  nx = usr->par->nx;
  nz = usr->par->nz;

  // Domain coords
  xmin = usr->par->xmin;
  zmin = usr->par->zmin;
  xmax = usr->par->xmin+usr->par->L;
  zmax = usr->par->zmin+usr->par->H;

  // 1. Stokes-Darcy: Create the FD-pde object
  PetscCall(FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_STOKESDARCY2FIELD,&fdPV));
  PetscCall(FDPDESetUp(fdPV));
  PetscCall(FDPDESetFunctionBCList(fdPV,FormBCList_PV,bc_description_stokesdarcy,usr)); 
  PetscCall(FDPDESetFunctionCoefficient(fdPV,FormCoefficient_PV,coeff_description_stokesdarcy,usr)); 
  PetscCall(SNESSetFromOptions(fdPV->snes)); 

  // 2. Porosity (Advection-diffusion)
  PetscCall(FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_ADVDIFF,&fdphi));
  PetscCall(FDPDESetUp(fdphi));

  if (usr->par->adv_scheme==0) { PetscCall(FDPDEAdvDiffSetAdvectSchemeType(fdphi,ADV_UPWIND)); }
  if (usr->par->adv_scheme==1) { PetscCall(FDPDEAdvDiffSetAdvectSchemeType(fdphi,ADV_FROMM)); }
  
  if (usr->par->ts_scheme ==  0) { PetscCall(FDPDEAdvDiffSetTimeStepSchemeType(fdphi,TS_FORWARD_EULER)); }
  if (usr->par->ts_scheme ==  1) { PetscCall(FDPDEAdvDiffSetTimeStepSchemeType(fdphi,TS_BACKWARD_EULER)); }
  if (usr->par->ts_scheme ==  2) { PetscCall(FDPDEAdvDiffSetTimeStepSchemeType(fdphi,TS_CRANK_NICHOLSON ));}

  PetscCall(FDPDESetFunctionBCList(fdphi,FormBCList_phi,bc_description_phi,usr)); 
  PetscCall(FDPDESetFunctionCoefficient(fdphi,FormCoefficient_phi,coeff_description_phi,usr)); 
  PetscCall(SNESSetFromOptions(fdphi->snes)); 

  // 3. Prepare usr data - for coupling
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

  // Set initial porosity profile (t=0)
  PetscCall(FDPDEAdvDiffGetPrevSolution(fdphi,&xphiprev));
  PetscCall(SetInitialPorosityProfile(dmphi,xphiprev,usr));
  PetscCall(VecCopy(xphiprev,usr->xphiprev));
  PetscCall(VecDestroy(&xphiprev));

  PetscCall(FDPDEGetCoefficient(fdphi,&dmphicoeff,NULL));
  PetscCall(FDPDEAdvDiffGetPrevCoefficient(fdphi,&phicoeffprev));
  PetscCall(SetInitialPorosityCoefficient(dmphicoeff,phicoeffprev,usr));
  PetscCall(VecDestroy(&phicoeffprev));

  // Time loop
  while ((usr->par->t <= usr->par->tmax) && (istep<=usr->par->tstep)) {
    PetscPrintf(PETSC_COMM_WORLD,"# TIMESTEP %d: \n",istep);

    // Set dt for porosity evolution 
    usr->par->dt = usr->par->dtmax;
    PetscCall(FDPDEAdvDiffSetTimestep(fdphi,usr->par->dt));

    // Update time
    usr->par->tprev = usr->par->t;
    usr->par->t    += usr->par->dt;

    // StokesDarcy Solver - using phi_old, tprev
    PetscCall(FDPDESolve(fdPV,NULL));
    PetscCall(FDPDEGetSolution(fdPV,&xPV));
    PetscCall(VecCopy(xPV,usr->xPV));

    // Porosity Solver - solve for phi_new, t - no iterations for MMS test (dt has to stay constant)
    PetscCall(FDPDESolve(fdphi,NULL));
    PetscCall(FDPDEGetSolution(fdphi,&xphi));

    // Compute manufactured solution and error norms per time step
    PetscCall(ComputeManufacturedSolutionTimestep(dmPV,&xmms_PV,dmphi,&xmms_phi,usr)); 
    PetscCall(ComputeErrorNorms(dmPV,xPV,xmms_PV,dmphi,xphi,xmms_phi));

    // Porosity: copy new solution and coefficient to old
    PetscCall(FDPDEAdvDiffGetPrevSolution(fdphi,&xphiprev));
    PetscCall(VecCopy(xphi,xphiprev));
    PetscCall(VecCopy(xphiprev,usr->xphiprev));
    PetscCall(VecDestroy(&xphiprev));

    PetscCall(FDPDEGetCoefficient(fdphi,&dmphicoeff,&phicoeff));
    PetscCall(FDPDEAdvDiffGetPrevCoefficient(fdphi,&phicoeffprev));
    PetscCall(VecCopy(phicoeff,phicoeffprev));
    PetscCall(VecDestroy(&phicoeffprev));

    // Output solution
    if (istep % usr->par->tout == 0 ) {
      PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s_PV_ts%1.3d",usr->par->fdir_out,usr->par->fname_out,istep));
      PetscCall(DMStagViewBinaryPython(dmPV,xPV,fout));

      PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s_phi_ts%1.3d",usr->par->fdir_out,usr->par->fname_out,istep));
      PetscCall(DMStagViewBinaryPython(dmphi,xphi,fout));

      PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s_mms_PV_ts%1.3d",usr->par->fdir_out,usr->par->fname_out,istep));
      PetscCall(DMStagViewBinaryPython(dmPV,xmms_PV,fout));

      PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s_mms_phi_ts%1.3d",usr->par->fdir_out,usr->par->fname_out,istep));
      PetscCall(DMStagViewBinaryPython(dmphi,xmms_phi,fout));
    }

    // Clean up
    PetscCall(VecDestroy(&xPV));
    PetscCall(VecDestroy(&xphi));
    PetscCall(VecDestroy(&xmms_PV));
    PetscCall(VecDestroy(&xmms_phi));

    // increment timestep
    istep++;

    PetscPrintf(PETSC_COMM_WORLD,"# TIME: time = %1.12e dt = %1.12e \n",usr->par->tprev,usr->par->dt);
    PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");
  }

  // Destroy objects
  PetscCall(DMDestroy(&dmPV));
  PetscCall(DMDestroy(&dmphi));
  PetscCall(FDPDEDestroy(&fdPV));
  PetscCall(FDPDEDestroy(&fdphi));
  PetscCall(VecDestroy(&usr->xPV));
  PetscCall(VecDestroy(&usr->xphiprev));

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
  PetscCall(PetscBagRegisterInt(bag, &par->nx, 4, "nx", "Element count in the x-dir")); 
  PetscCall(PetscBagRegisterInt(bag, &par->nz, 5, "nz", "Element count in the z-dir")); 

  PetscCall(PetscBagRegisterScalar(bag, &par->xmin, 0.0, "xmin", "Start coordinate of domain in x-dir")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->zmin, 0.0, "zmin", "Start coordinate of domain in z-dir")); 

  PetscCall(PetscBagRegisterScalar(bag, &par->L, 1.0, "L", "Length of domain in x-dir")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->H, 1.0, "H", "Height of domain in z-dir")); 

  // Physical and material parameters
  PetscCall(PetscBagRegisterScalar(bag, &par->e3, -1.0, "e3", "Direction of unit vertical vector")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->phi_0, 0.1, "phi_0", "Reference porosity")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->p_s, 1.0, "p_s", "Pressure amplitude")); 

  PetscCall(PetscBagRegisterScalar(bag, &par->m, 4.0, "m", "Trigonometric coefficient")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->n, 3.0, "n", "Porosity exponent")); 

  PetscCall(PetscBagRegisterScalar(bag, &par->eta, 1.0, "eta", "Scaled shear viscosity eta = eta_dim/eta0")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->zeta, 1.0, "zeta", "Scaled bulk viscosity zeta = zeta_dim/zeta0")); 

  par->xi = par->zeta - 2.0/3.0*par->eta;

  // Time stepping and advection
  PetscCall(PetscBagRegisterInt(bag, &par->ts_scheme,2, "ts_scheme", "Time stepping scheme 0-forward euler, 1-backward euler, 2-crank-nicholson")); 
  PetscCall(PetscBagRegisterInt(bag, &par->adv_scheme,1, "adv_scheme", "Advection scheme 0-upwind, 1-fromm")); 

  PetscCall(PetscBagRegisterInt(bag, &par->tout,1, "tout", "Output every tout time step")); 
  PetscCall(PetscBagRegisterInt(bag, &par->tstep,1, "tstep", "Maximum no of time steps")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->tmax, 1.0e2, "tmax", "Maximum time [-]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->dtmax, 1.0e-4, "dtmax", "Maximum time step size [-]")); 

  par->t  = 0.0;
  par->dt = 0.0;

  // Input/output 
  PetscCall(PetscBagRegisterString(bag,&par->fname_out,FNAME_LENGTH,"out_solution","output_file","Name for output file, set with: -output_file <filename>")); 
  PetscCall(PetscBagRegisterString(bag,&par->fdir_out,FNAME_LENGTH,"./","output_dir","Name for output directory, set with: -output_dir <dirname>")); 

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

  // Get petsc command options
  PetscCall(PetscOptionsGetAll(NULL, &opts)); 

  // Print header and petsc options
  PetscPrintf(usr->comm,"# --------------------------------------- #\n");
  PetscPrintf(usr->comm,"# Test_stokesdarcy2field_porosity_evolution: %s \n",&(date[0]));
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
// FormCoefficient_PV
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormCoefficient_PV"
PetscErrorCode FormCoefficient_PV(FDPDE fd, DM dm, Vec x, DM dmcoeff, Vec coeff, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz;
  DM             dmphi;
  Vec            xphi, xphilocal, coefflocal;
  PetscScalar    **coordx,**coordz;
  PetscInt       iprev, inext, icenter;
  PetscScalar    t,eta,zeta,xi,phi_0,p_s,m,n,e3;
  PetscScalar    ***c;
  PetscFunctionBeginUser;

  phi_0 = usr->par->phi_0;
  p_s = usr->par->p_s;
  m = usr->par->m;
  n = usr->par->n;
  e3 = usr->par->e3;
  eta = usr->par->eta;
  zeta = usr->par->zeta;
  xi = usr->par->xi;
  t  = usr->par->tprev;

  // Get dm and solution vector for porosity
  dmphi = usr->dmphi;
  xphi  = usr->xphiprev;

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

      { // A = eta (center, c=1)
        DMStagStencil point;
        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 1;
        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
        c[j][i][idx] = eta;
      }

      { // A = eta (corner, c=0)
        DMStagStencil point[4];
        PetscInt      ii;

        point[0].i = i; point[0].j = j; point[0].loc = DOWN_LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = DOWN_RIGHT; point[1].c = 0;
        point[2].i = i; point[2].j = j; point[2].loc = UP_LEFT;    point[2].c = 0;
        point[3].i = i; point[3].j = j; point[3].loc = UP_RIGHT;   point[3].c = 0;

        for (ii = 0; ii < 4; ii++) {
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] = eta;
        }
      }

      { // B = -phi*e3 + fu (fux,fuz - manufactured) (edges, c=0)
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

        // Bx = fux 
        rhs[0] = get_fux(xp[0],zp[0],t,eta,zeta,phi_0,p_s,m,n,e3);
        rhs[1] = get_fux(xp[1],zp[1],t,eta,zeta,phi_0,p_s,m,n,e3);

        // Bz = -phi*e3 + fuz
        rhs[2] = get_fuz(xp[2],zp[2],t,eta,zeta,phi_0,p_s,m,n,e3);
        rhs[3] = get_fuz(xp[3],zp[3],t,eta,zeta,phi_0,p_s,m,n,e3);

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
        rhs[2] -= e3*(1.0-Qinterp);

        Qinterp = interp1DLin_3Points(zp[3],zQ[0],Q[0],zQ[1],Q[1],zQ[2],Q[2]);
        rhs[3] -= e3*(1.0-Qinterp);

        for (ii = 0; ii < 4; ii++) {
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] = rhs[ii];
        }
      }

      { // C = (manufactured) (center, c=0)
        DMStagStencil point;
        PetscScalar   xp, zp;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 0;
        xp = coordx[i][icenter]; zp = coordz[j][icenter];
        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
        c[j][i][idx] = get_fp(xp,zp,t,eta,zeta,phi_0,p_s,m,n,e3);
      }

      { // D1 = xi (center, c=2)
        DMStagStencil point;
        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 2;
        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
        c[j][i][idx] = xi;
      }

      { // D2 = -Kphi (edges, c=1) - numerical, Kphi = (phi/phi0)^n
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
        rhs[0] = -PetscPowScalar((1.0-Qinterp)/phi_0,n); //left 

        Qinterp = interp1DLin_3Points(xp[1],xQ[0],Q[0],xQ[1],Q[1],xQ[2],Q[2]);
        rhs[1] = -PetscPowScalar((1.0-Qinterp)/phi_0,n); //right

        Qinterp = interp1DLin_3Points(zp[2],zQ[0],Q[3],zQ[1],Q[1],zQ[2],Q[4]);
        rhs[2] = -PetscPowScalar((1.0-Qinterp)/phi_0,n); // down

        Qinterp = interp1DLin_3Points(zp[3],zQ[0],Q[3],zQ[1],Q[1],zQ[2],Q[4]);
        rhs[3] = -PetscPowScalar((1.0-Qinterp)/phi_0,n); // up

        for (ii = 0; ii < 4; ii++) {
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] = rhs[ii];
        }
      }

      { // D3 = Kphi*e3 (edges, c=2) - numerical, Kphi = (phi/phi0)^n
        DMStagStencil point[4], pointQ[3];
        PetscScalar   zp[4],Qinterp, Q[3], rhs[4],zQ[3];
        PetscInt      ii;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 2;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 2;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = 2;
        point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = 2;

        zp[0] = coordz[j][icenter];
        zp[1] = coordz[j][icenter];
        zp[2] = coordz[j][iprev  ];
        zp[3] = coordz[j][inext  ];

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
        rhs[2]  = e3*PetscPowScalar((1.0-Qinterp)/phi_0,n);

        Qinterp = interp1DLin_3Points(zp[3],zQ[0],Q[0],zQ[1],Q[1],zQ[2],Q[2]);
        rhs[3]  = e3*PetscPowScalar((1.0-Qinterp)/phi_0,n); 

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
  PetscScalar    t,eta,zeta,phi_0,p_s,m,n,e3;
  BCType         *type_bc;
  PetscFunctionBeginUser;

  phi_0 = usr->par->phi_0;
  p_s = usr->par->p_s;
  m = usr->par->m;
  n = usr->par->n;
  e3 = usr->par->e3;
  eta = usr->par->eta;
  zeta = usr->par->zeta;
  t  = usr->par->tprev;

  // LEFT Boundary - Vx
  PetscCall(DMStagBCListGetValues(bclist,'w','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_ux(x_bc[2*k],x_bc[2*k+1],t,eta,zeta,phi_0,p_s,m,n,e3);
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // LEFT Boundary - Vz
  PetscCall(DMStagBCListGetValues(bclist,'w','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_uz(x_bc[2*k],x_bc[2*k+1],t,eta,zeta,phi_0,p_s,m,n,e3);
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // LEFT Boundary - P
  PetscCall(DMStagBCListGetValues(bclist,'w','o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_p(x_bc[2*k],x_bc[2*k+1],t,eta,zeta,phi_0,p_s,m,n,e3);
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  
  // RIGHT Boundary - Vx
  PetscCall(DMStagBCListGetValues(bclist,'e','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_ux(x_bc[2*k],x_bc[2*k+1],t,eta,zeta,phi_0,p_s,m,n,e3);
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // RIGHT Boundary - Vz
  PetscCall(DMStagBCListGetValues(bclist,'e','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_uz(x_bc[2*k],x_bc[2*k+1],t,eta,zeta,phi_0,p_s,m,n,e3);
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // RIGHT Boundary - P
  PetscCall(DMStagBCListGetValues(bclist,'e','o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_p(x_bc[2*k],x_bc[2*k+1],t,eta,zeta,phi_0,p_s,m,n,e3);
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // DOWN Boundary - Vx
  PetscCall(DMStagBCListGetValues(bclist,'s','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_ux(x_bc[2*k],x_bc[2*k+1],t,eta,zeta,phi_0,p_s,m,n,e3);
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // DOWN Boundary - Vz
  PetscCall(DMStagBCListGetValues(bclist,'s','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_uz(x_bc[2*k],x_bc[2*k+1],t,eta,zeta,phi_0,p_s,m,n,e3);
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // DOWN Boundary - P
  PetscCall(DMStagBCListGetValues(bclist,'s','o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_p(x_bc[2*k],x_bc[2*k+1],t,eta,zeta,phi_0,p_s,m,n,e3);
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // UP Boundary - Vx
  PetscCall(DMStagBCListGetValues(bclist,'n','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_ux(x_bc[2*k],x_bc[2*k+1],t,eta,zeta,phi_0,p_s,m,n,e3);
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // UP Boundary - Vz
  PetscCall(DMStagBCListGetValues(bclist,'n','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_uz(x_bc[2*k],x_bc[2*k+1],t,eta,zeta,phi_0,p_s,m,n,e3);
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // UP Boundary - P
  PetscCall(DMStagBCListGetValues(bclist,'n','o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_p(x_bc[2*k],x_bc[2*k+1],t,eta,zeta,phi_0,p_s,m,n,e3);
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
  PetscScalar    t,eta,zeta,phi_0,p_s,m,n,e3;
  Vec            xPV = NULL, xPVlocal;
  PetscFunctionBeginUser;

  phi_0 = usr->par->phi_0;
  p_s = usr->par->p_s;
  m = usr->par->m;
  n = usr->par->n;
  e3 = usr->par->e3;
  eta = usr->par->eta;
  zeta = usr->par->zeta;
  t  = usr->par->t;

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

      { // C = 0.0 - fphi, fphi manufactured solution
        DMStagStencil point;
        PetscInt      idx;
        PetscScalar   xp,zp;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 1;
        xp = coordx[i][icenter]; zp = coordz[j][icenter];
        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
        c[j][i][idx] = -get_fphi(xp,zp,t,eta,zeta,phi_0,p_s,m,n,e3);
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
// FormBCList_phi - manufactured Dirichlet BC 
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormBCList_phi"
PetscErrorCode FormBCList_phi(DM dm, Vec x, DMStagBCList bclist, void *ctx)
{
  UsrData     *usr = (UsrData*)ctx;
  PetscInt    k,n_bc,*idx_bc;
  PetscScalar *value_bc,*x_bc;
  PetscScalar t,eta,zeta,phi_0,p_s,m,n,e3;
  BCType      *type_bc;
  PetscFunctionBeginUser;

  phi_0 = usr->par->phi_0;
  p_s = usr->par->p_s;
  m = usr->par->m;
  n = usr->par->n;
  e3 = usr->par->e3;
  eta = usr->par->eta;
  zeta = usr->par->zeta;
  t  = usr->par->t;
  
  // Left = 1-phi
  PetscCall(DMStagBCListGetValues(bclist,'w','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 1.0-get_phi(x_bc[2*k],x_bc[2*k+1],t,eta,zeta,phi_0,p_s,m,n,e3);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));

  // RIGHT:
  PetscCall(DMStagBCListGetValues(bclist,'e','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 1.0-get_phi(x_bc[2*k],x_bc[2*k+1],t,eta,zeta,phi_0,p_s,m,n,e3);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));

  // DOWN:
  PetscCall(DMStagBCListGetValues(bclist,'s','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 1.0-get_phi(x_bc[2*k],x_bc[2*k+1],t,eta,zeta,phi_0,p_s,m,n,e3);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));

  // UP:
  PetscCall(DMStagBCListGetValues(bclist,'n','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 1.0-get_phi(x_bc[2*k],x_bc[2*k+1],t,eta,zeta,phi_0,p_s,m,n,e3);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
 
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// SetInitialPorosityProfile - Manufactured solution for t=0 - Gaussian shaped pulse (1-phi)
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "SetInitialPorosityProfile"
PetscErrorCode SetInitialPorosityProfile(DM dm, Vec x, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  Vec           xlocal;
  PetscInt      i,j, sx, sz, nx, nz, icenter;
  PetscScalar   eta,zeta,phi_0,p_s,m,n,e3;
  PetscScalar   ***xx, **coordx, **coordz;
  PetscFunctionBeginUser;

  // Parameters
  phi_0 = usr->par->phi_0;
  p_s = usr->par->p_s;
  m = usr->par->m;
  n = usr->par->n;
  e3 = usr->par->e3;
  eta = usr->par->eta;
  zeta = usr->par->zeta;

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
      PetscScalar   xp,zp;
      PetscInt      idx;

      point.i = i; point.j = j; point.loc = ELEMENT; point.c = 0;
      xp = coordx[i][icenter];
      zp = coordz[j][icenter];

      PetscCall(DMStagGetLocationSlot(dm, point.loc, point.c, &idx)); 
      xx[j][i][idx] = 1.0-get_phi(xp,zp,0.0,eta,zeta,phi_0,p_s,m,n,e3);
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
// SetInitialPorosityCoefficient - Manufactured solution for velocity at t=0 
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "SetInitialPorosityCoefficient"
PetscErrorCode SetInitialPorosityCoefficient(DM dmcoeff, Vec coeff, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       i, j, sx, sz, nx, nz,iprev,inext,icenter;
  PetscScalar    t,eta,zeta,phi_0,p_s,m,n,e3;
  Vec            coefflocal;
  PetscScalar    ***c, **coordx, **coordz;
  PetscFunctionBeginUser;

  phi_0 = usr->par->phi_0;
  p_s = usr->par->p_s;
  m = usr->par->m;
  n = usr->par->n;
  e3 = usr->par->e3;
  eta = usr->par->eta;
  zeta = usr->par->zeta;
  t  = usr->par->t;

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

      { // C = 0.0 - fphi
        DMStagStencil point;
        PetscInt      idx;
        PetscScalar   xp,zp;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 1;
        xp = coordx[i][icenter]; zp = coordz[j][icenter];
        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
        c[j][i][idx] = -get_fphi(xp,zp,t,eta,zeta,phi_0,p_s,m,n,e3);
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

      { // u = velocity (edge) = manufactured solution
        DMStagStencil point[4];
        PetscScalar   xp[4],zp[4],val[4];
        PetscInt      ii, idx;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 1;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 1;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = 1;
        point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = 1;

        xp[0] = coordx[i][iprev  ]; zp[0] = coordz[j][icenter];
        xp[1] = coordx[i][inext  ]; zp[1] = coordz[j][icenter];
        xp[2] = coordx[i][icenter]; zp[2] = coordz[j][iprev  ];
        xp[3] = coordx[i][icenter]; zp[3] = coordz[j][inext  ];

        val[0] = get_ux(xp[0],zp[0],0.0,eta,zeta,phi_0,p_s,m,n,e3);
        val[1] = get_ux(xp[1],zp[1],0.0,eta,zeta,phi_0,p_s,m,n,e3);
        val[2] = get_uz(xp[2],zp[2],0.0,eta,zeta,phi_0,p_s,m,n,e3);
        val[3] = get_uz(xp[3],zp[3],0.0,eta,zeta,phi_0,p_s,m,n,e3);

        for (ii = 0; ii < 4; ii++) {
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, 1, &idx)); 
          c[j][i][idx] = val[ii];
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

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// Compute manufactured solution
// ---------------------------------------
PetscErrorCode ComputeManufacturedSolutionTimestep(DM dmPV,Vec *_xmms_PV, DM dmphi, Vec *_xmms_phi, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz, idx;
  PetscInt       iprev, inext, icenter;
  PetscScalar    ***xxPV, ***xxphi;
  PetscScalar    **coordx,**coordz;
  Vec            xmms_PV,xmms_PVlocal,xmms_phi,xmms_philocal;
  PetscScalar    eta,zeta,phi_0,p_s,m,n,e3;
  PetscScalar    t, tprev;
  PetscFunctionBeginUser;

  phi_0 = usr->par->phi_0;
  p_s = usr->par->p_s;
  m = usr->par->m;
  n = usr->par->n;
  e3 = usr->par->e3;
  eta = usr->par->eta;
  zeta = usr->par->zeta;
  tprev = usr->par->tprev;
  t     = usr->par->t;

  // Create local and global vectors for MMS solutions
  PetscCall(DMCreateGlobalVector(dmPV,&xmms_PV     )); 
  PetscCall(DMCreateLocalVector (dmPV,&xmms_PVlocal)); 
  PetscCall(DMStagVecGetArray(dmPV,xmms_PVlocal,&xxPV)); 

  PetscCall(DMCreateGlobalVector(dmphi,&xmms_phi     )); 
  PetscCall(DMCreateLocalVector (dmphi,&xmms_philocal)); 
  PetscCall(DMStagVecGetArray(dmphi,xmms_philocal,&xxphi)); 

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

      // pressure
      PetscCall(DMStagGetLocationSlot(dmPV,ELEMENT,0,&idx)); 
      xxPV[j][i][idx] = get_p(coordx[i][icenter],coordz[j][icenter],tprev,eta,zeta,phi_0,p_s,m,n,e3);

      // ux
      PetscCall(DMStagGetLocationSlot(dmPV,LEFT,0,&idx)); 
      xxPV[j][i][idx] = get_ux(coordx[i][iprev],coordz[j][icenter],tprev,eta,zeta,phi_0,p_s,m,n,e3);

      if (i == Nx-1) {
        PetscCall(DMStagGetLocationSlot(dmPV,RIGHT,0,&idx)); 
        xxPV[j][i][idx] = get_ux(coordx[i][inext],coordz[j][icenter],tprev,eta,zeta,phi_0,p_s,m,n,e3);
      }
      
      // uz
      PetscCall(DMStagGetLocationSlot(dmPV,DOWN,0,&idx)); 
      xxPV[j][i][idx] = get_uz(coordx[i][icenter],coordz[j][iprev],tprev,eta,zeta,phi_0,p_s,m,n,e3);

      if (j == Nz-1) {
        PetscCall(DMStagGetLocationSlot(dmPV,UP,0,&idx)); 
        xxPV[j][i][idx] = get_uz(coordx[i][icenter],coordz[j][inext],tprev,eta,zeta,phi_0,p_s,m,n,e3);
      }

      // phi
      PetscCall(DMStagGetLocationSlot(dmphi,ELEMENT,0,&idx)); 
      xxphi[j][i][idx] = get_phi(coordx[i][icenter],coordz[j][icenter],t,eta,zeta,phi_0,p_s,m,n,e3);
    }
  }

  // Restore arrays
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dmPV,&coordx,&coordz,NULL));

  // Restore and map local to global
  PetscCall(DMStagVecRestoreArray(dmPV,xmms_PVlocal,&xxPV)); 
  PetscCall(DMLocalToGlobalBegin(dmPV,xmms_PVlocal,INSERT_VALUES,xmms_PV)); 
  PetscCall(DMLocalToGlobalEnd  (dmPV,xmms_PVlocal,INSERT_VALUES,xmms_PV)); 
  PetscCall(VecDestroy(&xmms_PVlocal)); 

  PetscCall(DMStagVecRestoreArray(dmphi,xmms_philocal,&xxphi)); 
  PetscCall(DMLocalToGlobalBegin(dmphi,xmms_philocal,INSERT_VALUES,xmms_phi)); 
  PetscCall(DMLocalToGlobalEnd  (dmphi,xmms_philocal,INSERT_VALUES,xmms_phi)); 
  PetscCall(VecDestroy(&xmms_philocal)); 

  // Assign pointers
  *_xmms_PV  = xmms_PV;
  *_xmms_phi = xmms_phi;
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// ComputeErrorNorms
// ---------------------------------------
PetscErrorCode ComputeErrorNorms(DM dmPV,Vec xPV,Vec xmms_PV,DM dmphi,Vec xphi,Vec xmms_phi)
{
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz;
  PetscScalar    xx[5], xa[5], dx, dz, dv;
  PetscScalar    nrm2phi, nrm2p, nrm2v, nrm2vx, nrm2vz, sum_err[4], gsum_err[4], sum_mms[4], gsum_mms[4];
  Vec            xPVlocal, xaPVlocal, xphilocal, xaphilocal;
  MPI_Comm       comm;
  PetscFunctionBeginUser;

  comm = PETSC_COMM_WORLD;

  // Get domain corners
  PetscCall(DMStagGetGlobalSizes(dmPV, &Nx, &Nz,NULL));
  PetscCall(DMStagGetCorners(dmPV, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 

  // Map global vectors to local domain
  PetscCall(DMGetLocalVector(dmPV, &xPVlocal)); 
  PetscCall(DMGlobalToLocal (dmPV, xPV, INSERT_VALUES, xPVlocal)); 
  PetscCall(DMGetLocalVector(dmPV, &xaPVlocal)); 
  PetscCall(DMGlobalToLocal (dmPV, xmms_PV, INSERT_VALUES, xaPVlocal)); 

  PetscCall(DMGetLocalVector(dmphi, &xphilocal)); 
  PetscCall(DMGlobalToLocal (dmphi, xphi, INSERT_VALUES, xphilocal)); 
  PetscCall(DMGetLocalVector(dmphi, &xaphilocal)); 
  PetscCall(DMGlobalToLocal (dmphi, xmms_phi, INSERT_VALUES, xaphilocal)); 

  // Initialize norms
  sum_err[0] = 0.0; sum_err[1] = 0.0; sum_err[2] = 0.0; sum_err[3] = 0.0;
  sum_mms[0] = 0.0; sum_mms[1] = 0.0; sum_mms[2] = 0.0; sum_mms[3] = 0.0;

  dx = 1.0/Nx;
  dz = 1.0/Nz;
  dv = dx*dz;

  // Loop over local domain to calculate ELEMENT errors
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      PetscScalar    ve[4], pe, v[4], p, phi, phie, Qx, Qa;
      DMStagStencil  point[5], pointQ;
      
      // Get stencil values
      point[0].i = i; point[0].j = j; point[0].loc = LEFT;    point[0].c = 0; // Vx
      point[1].i = i; point[1].j = j; point[1].loc = RIGHT;   point[1].c = 0; // Vx
      point[2].i = i; point[2].j = j; point[2].loc = DOWN;    point[2].c = 0; // Vz
      point[3].i = i; point[3].j = j; point[3].loc = UP;      point[3].c = 0; // Vz
      point[4].i = i; point[4].j = j; point[4].loc = ELEMENT; point[4].c = 0; // P

      pointQ.i = i; pointQ.j = j; pointQ.loc = ELEMENT; pointQ.c = 0; // Q = 1-phi

      // Get analytical and numerical solutions
      PetscCall(DMStagVecGetValuesStencil(dmPV, xPVlocal, 5,point,xx)); 
      PetscCall(DMStagVecGetValuesStencil(dmPV, xaPVlocal,5,point,xa)); 
      PetscCall(DMStagVecGetValuesStencil(dmphi, xphilocal, 1,&pointQ,&Qx)); 
      PetscCall(DMStagVecGetValuesStencil(dmphi, xaphilocal,1,&pointQ,&Qa));  // this is porosity

      // Error vectors - squared
      ve[0] = (xx[0]-xa[0])*(xx[0]-xa[0]); // Left
      ve[1] = (xx[1]-xa[1])*(xx[1]-xa[1]); // Right
      ve[2] = (xx[2]-xa[2])*(xx[2]-xa[2]); // Down
      ve[3] = (xx[3]-xa[3])*(xx[3]-xa[3]); // Up
      pe    = (xx[4]-xa[4])*(xx[4]-xa[4]); // elem
      phie  = (1-Qx-Qa)*(1-Qx-Qa);

      // MMS values - squared
      v[0] = xa[0]*xa[0]; // Left
      v[1] = xa[1]*xa[1]; // Right
      v[2] = xa[2]*xa[2]; // Down
      v[3] = xa[3]*xa[3]; // Up
      p    = xa[4]*xa[4]; // elem
      phi  = Qa*Qa;

      // Calculate sums for L2 norms - and normalize by magnitude of MMS solution
      if      (i == 0   ) { sum_err[0] += ve[0]*dv*0.5; sum_err[0] += ve[1]*dv; }
      else if (i == Nx-1) sum_err[0] += ve[1]*dv*0.5;
      else                sum_err[0] += ve[1]*dv;

      if      (j == 0   ) { sum_err[1] += ve[2]*dv*0.5; sum_err[1] += ve[3]*dv; }
      else if (j == Nz-1) sum_err[1] += ve[3]*dv*0.5;
      else                sum_err[1] += ve[3]*dv;
      sum_err[2] += pe*dv;
      sum_err[3] += phie*dv;

      if      (i == 0   ) { sum_mms[0] += v[0]*dv*0.5; sum_mms[0] += v[1]*dv; }
      else if (i == Nx-1) sum_mms[0] += v[1]*dv*0.5;
      else                sum_mms[0] += v[1]*dv;

      if      (j == 0   ) { sum_mms[1] += v[2]*dv*0.5; sum_mms[1] += v[3]*dv; }
      else if (j == Nz-1) sum_mms[1] += v[3]*dv*0.5;
      else                sum_mms[1] += v[3]*dv;
      sum_mms[2] += p*dv;
      sum_mms[3] += phi*dv;
    }
  }

  // Collect data 
  PetscCall(MPI_Allreduce(&sum_err, &gsum_err, 4, MPI_DOUBLE, MPI_SUM, comm)); 
  PetscCall(MPI_Allreduce(&sum_mms, &gsum_mms, 4, MPI_DOUBLE, MPI_SUM, comm)); 

  // L2 error norm = sqrt(gsum_err/gsum_mms)
  nrm2vx = PetscSqrtScalar(gsum_err[0]/gsum_mms[0]);
  nrm2vz = PetscSqrtScalar(gsum_err[1]/gsum_mms[1]);
  nrm2p  = PetscSqrtScalar(gsum_err[2]/gsum_mms[2]);
  nrm2phi= PetscSqrtScalar(gsum_err[3]/gsum_mms[3]);
  nrm2v  = PetscSqrtScalar((gsum_err[0]+gsum_err[1])/(gsum_mms[0]+gsum_mms[1]));

  // Restore arrays and vectors
  PetscCall(DMRestoreLocalVector(dmPV, &xPVlocal )); 
  PetscCall(DMRestoreLocalVector(dmPV, &xaPVlocal )); 
  PetscCall(DMRestoreLocalVector(dmphi, &xphilocal )); 
  PetscCall(DMRestoreLocalVector(dmphi, &xaphilocal )); 

  // Print information
  // PetscPrintf(comm,"# --------------------------------------- #\n");
  PetscPrintf(comm,"# NORMS: \n");
  PetscPrintf(comm,"# Velocity: norm2 = %1.12e norm2x = %1.12e norm2z = %1.12e \n",nrm2v,nrm2vx,nrm2vz);
  PetscPrintf(comm,"# Pressure: norm2 = %1.12e\n",nrm2p);
  PetscPrintf(comm,"# Porosity: norm2 = %1.12e\n",nrm2phi);
  PetscPrintf(comm,"# Porosity err-squared: num = %1.12e mms = %1.12e\n",gsum_err[3],gsum_mms[3]);
  PetscPrintf(comm,"# Grid info: hx = %1.12e hz = %1.12e\n",dx,dz);

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

  // Print user parameters
  PetscCall(InputPrintData(usr)); 

  // Numerical solution using the FD pde object
  PetscCall(PetscTime(&start_time)); 
  PetscCall(Numerical_solution(usr)); 
  PetscCall(PetscTime(&end_time)); 
  PetscPrintf(PETSC_COMM_WORLD,"# Runtime: %g (sec) \n", end_time - start_time);
  PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");

  // Destroy objects
  PetscCall(PetscBagDestroy(&usr->bag)); 
  PetscCall(PetscFree(usr));

  // Finalize main
  PetscCall(PetscFinalize());
  return 0;
}
