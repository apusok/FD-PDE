// ---------------------------------------
// MMS test for porosity evolution - verify coupled system for two-phase flow 
// Solves for coupled (P, v) and Q=(1-phi) evolution, where P-dynamic pressure, v-solid velocity, phi-porosity.
// run: ./tests/test_stokesdarcy2field_mms_porosity.app -pc_type lu -pc_factor_mat_solver_type umfpack -nx 20 -nz 20 -snes_monitor 
// python test: ./tests/python/test_stokesdarcy2field_mms_porosity.py
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

#include "petsc.h"
#include "../fdpde_stokesdarcy2field.h"
#include "../fdpde_advdiff.h"
#include "../dmstagoutput.h"

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
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Element count
  nx = usr->par->nx;
  nz = usr->par->nz;

  // Domain coords
  xmin = usr->par->xmin;
  zmin = usr->par->zmin;
  xmax = usr->par->xmin+usr->par->L;
  zmax = usr->par->zmin+usr->par->H;

  // 1. Stokes-Darcy: Create the FD-pde object
  ierr = FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_STOKESDARCY2FIELD,&fdPV);CHKERRQ(ierr);
  ierr = FDPDESetUp(fdPV);CHKERRQ(ierr);
  ierr = FDPDESetFunctionBCList(fdPV,FormBCList_PV,bc_description_stokesdarcy,usr); CHKERRQ(ierr);
  ierr = FDPDESetFunctionCoefficient(fdPV,FormCoefficient_PV,coeff_description_stokesdarcy,usr); CHKERRQ(ierr);
  ierr = SNESSetFromOptions(fdPV->snes); CHKERRQ(ierr);

  // 2. Porosity (Advection-diffusion)
  ierr = FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_ADVDIFF,&fdphi);CHKERRQ(ierr);
  ierr = FDPDESetUp(fdphi);CHKERRQ(ierr);

  if (usr->par->adv_scheme==0) { ierr = FDPDEAdvDiffSetAdvectSchemeType(fdphi,ADV_UPWIND);CHKERRQ(ierr); }
  if (usr->par->adv_scheme==1) { ierr = FDPDEAdvDiffSetAdvectSchemeType(fdphi,ADV_FROMM);CHKERRQ(ierr); }
  
  if (usr->par->ts_scheme ==  0) { ierr = FDPDEAdvDiffSetTimeStepSchemeType(fdphi,TS_FORWARD_EULER);CHKERRQ(ierr); }
  if (usr->par->ts_scheme ==  1) { ierr = FDPDEAdvDiffSetTimeStepSchemeType(fdphi,TS_BACKWARD_EULER);CHKERRQ(ierr); }
  if (usr->par->ts_scheme ==  2) { ierr = FDPDEAdvDiffSetTimeStepSchemeType(fdphi,TS_CRANK_NICHOLSON );CHKERRQ(ierr);}

  ierr = FDPDESetFunctionBCList(fdphi,FormBCList_phi,bc_description_phi,usr); CHKERRQ(ierr);
  ierr = FDPDESetFunctionCoefficient(fdphi,FormCoefficient_phi,coeff_description_phi,usr); CHKERRQ(ierr);
  ierr = SNESSetFromOptions(fdphi->snes); CHKERRQ(ierr);

  // 3. Prepare usr data - for coupling
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

  // Set initial porosity profile (t=0)
  ierr = FDPDEAdvDiffGetPrevSolution(fdphi,&xphiprev);CHKERRQ(ierr);
  ierr = SetInitialPorosityProfile(dmphi,xphiprev,usr);CHKERRQ(ierr);
  ierr = VecCopy(xphiprev,usr->xphiprev);CHKERRQ(ierr);
  ierr = VecDestroy(&xphiprev);CHKERRQ(ierr);

  ierr = FDPDEGetCoefficient(fdphi,&dmphicoeff,NULL);CHKERRQ(ierr);
  ierr = FDPDEAdvDiffGetPrevCoefficient(fdphi,&phicoeffprev);CHKERRQ(ierr);
  ierr = SetInitialPorosityCoefficient(dmphicoeff,phicoeffprev,usr);CHKERRQ(ierr);
  ierr = VecDestroy(&phicoeffprev);CHKERRQ(ierr);

  // Time loop
  while ((usr->par->t <= usr->par->tmax) && (istep<=usr->par->tstep)) {
    PetscPrintf(PETSC_COMM_WORLD,"# TIMESTEP %d: \n",istep);

    // Set dt for porosity evolution 
    usr->par->dt = usr->par->dtmax;
    ierr = FDPDEAdvDiffSetTimestep(fdphi,usr->par->dt);CHKERRQ(ierr);

    // Update time
    usr->par->tprev = usr->par->t;
    usr->par->t    += usr->par->dt;

    // StokesDarcy Solver - using phi_old, tprev
    ierr = FDPDESolve(fdPV,NULL);CHKERRQ(ierr);
    ierr = FDPDEGetSolution(fdPV,&xPV);CHKERRQ(ierr);
    ierr = VecCopy(xPV,usr->xPV);CHKERRQ(ierr);

    // Porosity Solver - solve for phi_new, t - no iterations for MMS test (dt has to stay constant)
    ierr = FDPDESolve(fdphi,NULL);CHKERRQ(ierr);
    ierr = FDPDEGetSolution(fdphi,&xphi);CHKERRQ(ierr);

    // Compute manufactured solution and error norms per time step
    ierr = ComputeManufacturedSolutionTimestep(dmPV,&xmms_PV,dmphi,&xmms_phi,usr); CHKERRQ(ierr);
    ierr = ComputeErrorNorms(dmPV,xPV,xmms_PV,dmphi,xphi,xmms_phi);CHKERRQ(ierr);

    // Porosity: copy new solution and coefficient to old
    ierr = FDPDEAdvDiffGetPrevSolution(fdphi,&xphiprev);CHKERRQ(ierr);
    ierr = VecCopy(xphi,xphiprev);CHKERRQ(ierr);
    ierr = VecCopy(xphiprev,usr->xphiprev);CHKERRQ(ierr);
    ierr = VecDestroy(&xphiprev);CHKERRQ(ierr);

    ierr = FDPDEGetCoefficient(fdphi,&dmphicoeff,&phicoeff);CHKERRQ(ierr);
    ierr = FDPDEAdvDiffGetPrevCoefficient(fdphi,&phicoeffprev);CHKERRQ(ierr);
    ierr = VecCopy(phicoeff,phicoeffprev);CHKERRQ(ierr);
    ierr = VecDestroy(&phicoeffprev);CHKERRQ(ierr);

    // Output solution
    if (istep % usr->par->tout == 0 ) {
      ierr = PetscSNPrintf(fout,sizeof(fout),"%s_PV_ts%1.3d",usr->par->fname_out,istep);
      ierr = DMStagViewBinaryPython(dmPV,xPV,fout);CHKERRQ(ierr);

      ierr = PetscSNPrintf(fout,sizeof(fout),"%s_phi_ts%1.3d",usr->par->fname_out,istep);
      ierr = DMStagViewBinaryPython(dmphi,xphi,fout);CHKERRQ(ierr);

      ierr = PetscSNPrintf(fout,sizeof(fout),"%s_mms_PV_ts%1.3d",usr->par->fname_out,istep);
      ierr = DMStagViewBinaryPython(dmPV,xmms_PV,fout);CHKERRQ(ierr);

      ierr = PetscSNPrintf(fout,sizeof(fout),"%s_mms_phi_ts%1.3d",usr->par->fname_out,istep);
      ierr = DMStagViewBinaryPython(dmphi,xmms_phi,fout);CHKERRQ(ierr);
    }

    // Clean up
    ierr = VecDestroy(&xPV);CHKERRQ(ierr);
    ierr = VecDestroy(&xphi);CHKERRQ(ierr);

    // increment timestep
    istep++;

    PetscPrintf(PETSC_COMM_WORLD,"# TIME: time = %1.12e dt = %1.12e \n",usr->par->tprev,usr->par->dt);
    PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");
  }

  // Destroy objects
  ierr = VecDestroy(&xmms_PV);CHKERRQ(ierr);
  ierr = VecDestroy(&xmms_phi);CHKERRQ(ierr);
  ierr = DMDestroy(&dmPV);CHKERRQ(ierr);
  ierr = DMDestroy(&dmphi);CHKERRQ(ierr);
  ierr = FDPDEDestroy(&fdPV);CHKERRQ(ierr);
  ierr = FDPDEDestroy(&fdphi);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xPV);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xphiprev);CHKERRQ(ierr);

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
  ierr = PetscBagRegisterInt(bag, &par->nx, 4, "nx", "Element count in the x-dir"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->nz, 5, "nz", "Element count in the z-dir"); CHKERRQ(ierr);

  ierr = PetscBagRegisterScalar(bag, &par->xmin, 0.0, "xmin", "Start coordinate of domain in x-dir"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->zmin, 0.0, "zmin", "Start coordinate of domain in z-dir"); CHKERRQ(ierr);

  ierr = PetscBagRegisterScalar(bag, &par->L, 1.0, "L", "Length of domain in x-dir"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->H, 1.0, "H", "Height of domain in z-dir"); CHKERRQ(ierr);

  // Physical and material parameters
  ierr = PetscBagRegisterScalar(bag, &par->e3, -1.0, "e3", "Direction of unit vertical vector"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->phi_0, 0.1, "phi_0", "Reference porosity"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->p_s, 1.0, "p_s", "Pressure amplitude"); CHKERRQ(ierr);

  ierr = PetscBagRegisterScalar(bag, &par->m, 4.0, "m", "Trigonometric coefficient"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->n, 3.0, "n", "Porosity exponent"); CHKERRQ(ierr);

  ierr = PetscBagRegisterScalar(bag, &par->eta, 1.0, "eta", "Scaled shear viscosity eta = eta_dim/eta0"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->zeta, 1.0, "zeta", "Scaled bulk viscosity zeta = zeta_dim/zeta0"); CHKERRQ(ierr);

  par->xi = par->zeta - 2.0/3.0*par->eta;

  // Time stepping and advection
  ierr = PetscBagRegisterInt(bag, &par->ts_scheme,2, "ts_scheme", "Time stepping scheme 0-forward euler, 1-backward euler, 2-crank-nicholson"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->adv_scheme,1, "adv_scheme", "Advection scheme 0-upwind, 1-fromm"); CHKERRQ(ierr);

  ierr = PetscBagRegisterInt(bag, &par->tout,1, "tout", "Output every tout time step"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->tstep,1, "tstep", "Maximum no of time steps"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->tmax, 1.0e2, "tmax", "Maximum time [-]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->dtmax, 1.0e-4, "dtmax", "Maximum time step size [-]"); CHKERRQ(ierr);

  par->t  = 0.0;
  par->dt = 0.0;

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

  // Get petsc command options
  ierr = PetscOptionsGetAll(NULL, &opts); CHKERRQ(ierr);

  // Print header and petsc options
  PetscPrintf(usr->comm,"# --------------------------------------- #\n");
  PetscPrintf(usr->comm,"# Test_stokesdarcy2field_porosity_evolution: %s \n",&(date[0]));
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
  PetscErrorCode ierr;

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

      { // A = eta (center, c=1)
        DMStagStencil point;
        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 1;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
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
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
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

        ierr = DMStagVecGetValuesStencil(dmphi,xphilocal,3,pointQ,Q); CHKERRQ(ierr);
        Qinterp = interp1DLin_3Points(zp[2],zQ[0],Q[0],zQ[1],Q[1],zQ[2],Q[2]); // Q = 1-phi
        rhs[2] -= e3*(1.0-Qinterp);

        Qinterp = interp1DLin_3Points(zp[3],zQ[0],Q[0],zQ[1],Q[1],zQ[2],Q[2]);
        rhs[3] -= e3*(1.0-Qinterp);

        for (ii = 0; ii < 4; ii++) {
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = rhs[ii];
        }
      }

      { // C = (manufactured) (center, c=0)
        DMStagStencil point;
        PetscScalar   xp, zp;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 0;
        xp = coordx[i][icenter]; zp = coordz[j][icenter];
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = get_fp(xp,zp,t,eta,zeta,phi_0,p_s,m,n,e3);
      }

      { // D1 = xi (center, c=2)
        DMStagStencil point;
        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 2;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
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

        ierr = DMStagVecGetValuesStencil(dmphi,xphilocal,5,pointQ,Q); CHKERRQ(ierr);

        Qinterp = interp1DLin_3Points(xp[0],xQ[0],Q[0],xQ[1],Q[1],xQ[2],Q[2]); // Q = 1-phi
        rhs[0] = -PetscPowScalar((1.0-Qinterp)/phi_0,n); //left 

        Qinterp = interp1DLin_3Points(xp[1],xQ[0],Q[0],xQ[1],Q[1],xQ[2],Q[2]);
        rhs[1] = -PetscPowScalar((1.0-Qinterp)/phi_0,n); //right

        Qinterp = interp1DLin_3Points(zp[2],zQ[0],Q[3],zQ[1],Q[1],zQ[2],Q[4]);
        rhs[2] = -PetscPowScalar((1.0-Qinterp)/phi_0,n); // down

        Qinterp = interp1DLin_3Points(zp[3],zQ[0],Q[3],zQ[1],Q[1],zQ[2],Q[4]);
        rhs[3] = -PetscPowScalar((1.0-Qinterp)/phi_0,n); // up

        for (ii = 0; ii < 4; ii++) {
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
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

        ierr = DMStagVecGetValuesStencil(dmphi,xphilocal,3,pointQ,Q); CHKERRQ(ierr);
        Qinterp = interp1DLin_3Points(zp[2],zQ[0],Q[0],zQ[1],Q[1],zQ[2],Q[2]); // Q = 1-phi
        rhs[2]  = e3*PetscPowScalar((1.0-Qinterp)/phi_0,n);

        Qinterp = interp1DLin_3Points(zp[3],zQ[0],Q[0],zQ[1],Q[1],zQ[2],Q[2]);
        rhs[3]  = e3*PetscPowScalar((1.0-Qinterp)/phi_0,n); 

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
  PetscScalar    t,eta,zeta,phi_0,p_s,m,n,e3;
  BCType         *type_bc;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;

  phi_0 = usr->par->phi_0;
  p_s = usr->par->p_s;
  m = usr->par->m;
  n = usr->par->n;
  e3 = usr->par->e3;
  eta = usr->par->eta;
  zeta = usr->par->zeta;
  t  = usr->par->tprev;

  // LEFT Boundary - Vx
  ierr = DMStagBCListGetValues(bclist,'w','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_ux(x_bc[2*k],x_bc[2*k+1],t,eta,zeta,phi_0,p_s,m,n,e3);
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // LEFT Boundary - Vz
  ierr = DMStagBCListGetValues(bclist,'w','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=1; k<n_bc-1; k++) {
    value_bc[k] = get_uz(x_bc[2*k],x_bc[2*k+1],t,eta,zeta,phi_0,p_s,m,n,e3);
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // LEFT Boundary - P
  ierr = DMStagBCListGetValues(bclist,'w','o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_p(x_bc[2*k],x_bc[2*k+1],t,eta,zeta,phi_0,p_s,m,n,e3);
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  
  // RIGHT Boundary - Vx
  ierr = DMStagBCListGetValues(bclist,'e','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_ux(x_bc[2*k],x_bc[2*k+1],t,eta,zeta,phi_0,p_s,m,n,e3);
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // RIGHT Boundary - Vz
  ierr = DMStagBCListGetValues(bclist,'e','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_uz(x_bc[2*k],x_bc[2*k+1],t,eta,zeta,phi_0,p_s,m,n,e3);
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // RIGHT Boundary - P
  ierr = DMStagBCListGetValues(bclist,'e','o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_p(x_bc[2*k],x_bc[2*k+1],t,eta,zeta,phi_0,p_s,m,n,e3);
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN Boundary - Vx
  ierr = DMStagBCListGetValues(bclist,'s','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_ux(x_bc[2*k],x_bc[2*k+1],t,eta,zeta,phi_0,p_s,m,n,e3);
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN Boundary - Vz
  ierr = DMStagBCListGetValues(bclist,'s','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_uz(x_bc[2*k],x_bc[2*k+1],t,eta,zeta,phi_0,p_s,m,n,e3);
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN Boundary - P
  ierr = DMStagBCListGetValues(bclist,'s','o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_p(x_bc[2*k],x_bc[2*k+1],t,eta,zeta,phi_0,p_s,m,n,e3);
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP Boundary - Vx
  ierr = DMStagBCListGetValues(bclist,'n','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_ux(x_bc[2*k],x_bc[2*k+1],t,eta,zeta,phi_0,p_s,m,n,e3);
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP Boundary - Vz
  ierr = DMStagBCListGetValues(bclist,'n','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_uz(x_bc[2*k],x_bc[2*k+1],t,eta,zeta,phi_0,p_s,m,n,e3);
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP Boundary - P
  ierr = DMStagBCListGetValues(bclist,'n','o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_p(x_bc[2*k],x_bc[2*k+1],t,eta,zeta,phi_0,p_s,m,n,e3);
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

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
  PetscScalar    t,eta,zeta,phi_0,p_s,m,n,e3;
  Vec            xPV = NULL, xPVlocal;
  PetscErrorCode ierr;

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

      { // C = 0.0 - fphi, fphi manufactured solution
        DMStagStencil point;
        PetscInt      idx;
        PetscScalar   xp,zp;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 1;
        xp = coordx[i][icenter]; zp = coordz[j][icenter];
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  PetscFunctionBegin;

  phi_0 = usr->par->phi_0;
  p_s = usr->par->p_s;
  m = usr->par->m;
  n = usr->par->n;
  e3 = usr->par->e3;
  eta = usr->par->eta;
  zeta = usr->par->zeta;
  t  = usr->par->t;
  
  // Left = 1-phi
  ierr = DMStagBCListGetValues(bclist,'w','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 1.0-get_phi(x_bc[2*k],x_bc[2*k+1],t,eta,zeta,phi_0,p_s,m,n,e3);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // RIGHT:
  ierr = DMStagBCListGetValues(bclist,'e','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 1.0-get_phi(x_bc[2*k],x_bc[2*k+1],t,eta,zeta,phi_0,p_s,m,n,e3);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN:
  ierr = DMStagBCListGetValues(bclist,'s','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 1.0-get_phi(x_bc[2*k],x_bc[2*k+1],t,eta,zeta,phi_0,p_s,m,n,e3);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP:
  ierr = DMStagBCListGetValues(bclist,'n','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 1.0-get_phi(x_bc[2*k],x_bc[2*k+1],t,eta,zeta,phi_0,p_s,m,n,e3);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
 
  PetscFunctionReturn(0);
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
  PetscErrorCode ierr;
  PetscFunctionBegin;

  // Parameters
  phi_0 = usr->par->phi_0;
  p_s = usr->par->p_s;
  m = usr->par->m;
  n = usr->par->n;
  e3 = usr->par->e3;
  eta = usr->par->eta;
  zeta = usr->par->zeta;

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
      PetscScalar   xp,zp;
      PetscInt      idx;

      point.i = i; point.j = j; point.loc = ELEMENT; point.c = 0;
      xp = coordx[i][icenter];
      zp = coordz[j][icenter];

      ierr = DMStagGetLocationSlot(dm, point.loc, point.c, &idx); CHKERRQ(ierr);
      xx[j][i][idx] = 1.0-get_phi(xp,zp,0.0,eta,zeta,phi_0,p_s,m,n,e3);
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
  PetscErrorCode ierr;

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

      { // C = 0.0 - fphi
        DMStagStencil point;
        PetscInt      idx;
        PetscScalar   xp,zp;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 1;
        xp = coordx[i][icenter]; zp = coordz[j][icenter];
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
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
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, 0, &idx); CHKERRQ(ierr);
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
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, 1, &idx); CHKERRQ(ierr);
          c[j][i][idx] = val[ii];
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

  PetscFunctionReturn(0);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;

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
  ierr = DMCreateGlobalVector(dmPV,&xmms_PV     ); CHKERRQ(ierr);
  ierr = DMCreateLocalVector (dmPV,&xmms_PVlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dmPV,xmms_PVlocal,&xxPV); CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(dmphi,&xmms_phi     ); CHKERRQ(ierr);
  ierr = DMCreateLocalVector (dmphi,&xmms_philocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dmphi,xmms_philocal,&xxphi); CHKERRQ(ierr);

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

      // pressure
      ierr = DMStagGetLocationSlot(dmPV,ELEMENT,0,&idx); CHKERRQ(ierr);
      xxPV[j][i][idx] = get_p(coordx[i][icenter],coordz[j][icenter],tprev,eta,zeta,phi_0,p_s,m,n,e3);

      // ux
      ierr = DMStagGetLocationSlot(dmPV,LEFT,0,&idx); CHKERRQ(ierr);
      xxPV[j][i][idx] = get_ux(coordx[i][iprev],coordz[j][icenter],tprev,eta,zeta,phi_0,p_s,m,n,e3);

      if (i == Nx-1) {
        ierr = DMStagGetLocationSlot(dmPV,RIGHT,0,&idx); CHKERRQ(ierr);
        xxPV[j][i][idx] = get_ux(coordx[i][inext],coordz[j][icenter],tprev,eta,zeta,phi_0,p_s,m,n,e3);
      }
      
      // uz
      ierr = DMStagGetLocationSlot(dmPV,DOWN,0,&idx); CHKERRQ(ierr);
      xxPV[j][i][idx] = get_uz(coordx[i][icenter],coordz[j][iprev],tprev,eta,zeta,phi_0,p_s,m,n,e3);

      if (j == Nz-1) {
        ierr = DMStagGetLocationSlot(dmPV,UP,0,&idx); CHKERRQ(ierr);
        xxPV[j][i][idx] = get_uz(coordx[i][icenter],coordz[j][inext],tprev,eta,zeta,phi_0,p_s,m,n,e3);
      }

      // phi
      ierr = DMStagGetLocationSlot(dmphi,ELEMENT,0,&idx); CHKERRQ(ierr);
      xxphi[j][i][idx] = get_phi(coordx[i][icenter],coordz[j][icenter],t,eta,zeta,phi_0,p_s,m,n,e3);
    }
  }

  // Restore arrays
  ierr = DMStagRestoreProductCoordinateArraysRead(dmPV,&coordx,&coordz,NULL);CHKERRQ(ierr);

  // Restore and map local to global
  ierr = DMStagVecRestoreArray(dmPV,xmms_PVlocal,&xxPV); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dmPV,xmms_PVlocal,INSERT_VALUES,xmms_PV); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dmPV,xmms_PVlocal,INSERT_VALUES,xmms_PV); CHKERRQ(ierr);
  ierr = VecDestroy(&xmms_PVlocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArray(dmphi,xmms_philocal,&xxphi); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dmphi,xmms_philocal,INSERT_VALUES,xmms_phi); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dmphi,xmms_philocal,INSERT_VALUES,xmms_phi); CHKERRQ(ierr);
  ierr = VecDestroy(&xmms_philocal); CHKERRQ(ierr);

  // Assign pointers
  *_xmms_PV  = xmms_PV;
  *_xmms_phi = xmms_phi;
  
  PetscFunctionReturn(0);
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

  PetscErrorCode ierr;
  PetscFunctionBegin;

  comm = PETSC_COMM_WORLD;

  // Get domain corners
  ierr = DMStagGetGlobalSizes(dmPV, &Nx, &Nz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dmPV, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  // Map global vectors to local domain
  ierr = DMGetLocalVector(dmPV, &xPVlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dmPV, xPV, INSERT_VALUES, xPVlocal); CHKERRQ(ierr);
  ierr = DMGetLocalVector(dmPV, &xaPVlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dmPV, xmms_PV, INSERT_VALUES, xaPVlocal); CHKERRQ(ierr);

  ierr = DMGetLocalVector(dmphi, &xphilocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dmphi, xphi, INSERT_VALUES, xphilocal); CHKERRQ(ierr);
  ierr = DMGetLocalVector(dmphi, &xaphilocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dmphi, xmms_phi, INSERT_VALUES, xaphilocal); CHKERRQ(ierr);

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
      ierr = DMStagVecGetValuesStencil(dmPV, xPVlocal, 5,point,xx); CHKERRQ(ierr);
      ierr = DMStagVecGetValuesStencil(dmPV, xaPVlocal,5,point,xa); CHKERRQ(ierr);
      ierr = DMStagVecGetValuesStencil(dmphi, xphilocal, 1,&pointQ,&Qx); CHKERRQ(ierr);
      ierr = DMStagVecGetValuesStencil(dmphi, xaphilocal,1,&pointQ,&Qa); CHKERRQ(ierr); // this is porosity

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
  ierr = MPI_Allreduce(&sum_err, &gsum_err, 4, MPI_DOUBLE, MPI_SUM, comm); CHKERRQ(ierr);
  ierr = MPI_Allreduce(&sum_mms, &gsum_mms, 4, MPI_DOUBLE, MPI_SUM, comm); CHKERRQ(ierr);

  // L2 error norm = sqrt(gsum_err/gsum_mms)
  nrm2vx = PetscSqrtScalar(gsum_err[0]/gsum_mms[0]);
  nrm2vz = PetscSqrtScalar(gsum_err[1]/gsum_mms[1]);
  nrm2p  = PetscSqrtScalar(gsum_err[2]/gsum_mms[2]);
  nrm2phi= PetscSqrtScalar(gsum_err[3]/gsum_mms[3]);
  nrm2v  = PetscSqrtScalar((gsum_err[0]+gsum_err[1])/(gsum_mms[0]+gsum_mms[1]));

  // Restore arrays and vectors
  ierr = DMRestoreLocalVector(dmPV, &xPVlocal ); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmPV, &xaPVlocal ); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmphi, &xphilocal ); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmphi, &xaphilocal ); CHKERRQ(ierr);

  // Print information
  // PetscPrintf(comm,"# --------------------------------------- #\n");
  PetscPrintf(comm,"# NORMS: \n");
  PetscPrintf(comm,"# Velocity: norm2 = %1.12e norm2x = %1.12e norm2z = %1.12e \n",nrm2v,nrm2vx,nrm2vz);
  PetscPrintf(comm,"# Pressure: norm2 = %1.12e\n",nrm2p);
  PetscPrintf(comm,"# Porosity: norm2 = %1.12e\n",nrm2phi);
  PetscPrintf(comm,"# Porosity err-squared: num = %1.12e mms = %1.12e\n",gsum_err[3],gsum_mms[3]);
  PetscPrintf(comm,"# Grid info: hx = %1.12e hz = %1.12e\n",dx,dz);

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

  // Print user parameters
  ierr = InputPrintData(usr); CHKERRQ(ierr);

  // Numerical solution using the FD pde object
  ierr = PetscTime(&start_time); CHKERRQ(ierr);
  ierr = Numerical_solution(usr); CHKERRQ(ierr);
  ierr = PetscTime(&end_time); CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"# Runtime: %g (sec) \n", end_time - start_time);
  PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");

  // Destroy objects
  ierr = PetscBagDestroy(&usr->bag); CHKERRQ(ierr);
  ierr = PetscFree(usr);             CHKERRQ(ierr);

  // Finalize main
  ierr = PetscFinalize();
  return ierr;
}
