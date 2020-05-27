// ---------------------------------------
// MMS test to verify a power-law effective viscosity approach, where eta=eta0*(epsII/eps0)^(1/np-1), zeta = eta/phi
// run: ./tests/test_effvisc_mms.app -pc_type lu -pc_factor_mat_solver_type umfpack -snes_monitor -ksp_monitor -nx 20 -nz 20
// python test: ./tests/python/test_effvisc_mms.py
// python sympy: ./mms/mms_effvisc_powerlaw.py
// ---------------------------------------
static char help[] = "Application to verify a power-law effective viscosity for Stokes and StokesDarcy using MMS \n\n";

// System of equations (nondimensional with u0 = drho*g*h^2/eta0) - also ignore body forces (k_hat=0)
// Stokes:
// -grad(P)+div(eta symgrad(v)) + rho*k_hat = 0
// div(v) = 0 
// StokesDarcy:
// -grad(P)+div(eta symgrad(v)) + grad((zeta-2/3eta)div(v))+phi*k_hat = 0
// div(v) - R^2 div(Kphi(grad(P)-k_hat)) = 0, where R=delta/h, delta = sqrt(K0*eta0/mu) 

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
#include "../fdpde_stokes.h"
#include "../consteq.h"
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
  PetscScalar    eta0, eps0, np, R;
  PetscScalar    phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat;
  char           fname_out[FNAME_LENGTH]; 
} Params;

// user defined and model-dependent variables
typedef struct {
  Params        *par;
  PetscBag       bag;
  MPI_Comm       comm;
  PetscMPIInt    rank;
  DM             dm;
} UsrData;

// ---------------------------------------
// Function definitions
// ---------------------------------------
PetscErrorCode Numerical_solution(void*,PetscInt);
PetscErrorCode ComputeManufacturedSolution(DM,Vec*,void*);
PetscErrorCode ComputeErrorNorms(DM,Vec,Vec,PetscInt test);
PetscErrorCode InputParameters(UsrData**);
PetscErrorCode InputPrintData(UsrData*);
PetscErrorCode FormCoefficient_Stokes(FDPDE, DM, Vec, DM, Vec, void*);
PetscErrorCode FormCoefficient_StokesDarcy(FDPDE, DM, Vec, DM, Vec, void*);
PetscErrorCode FormBCList(DM, Vec, DMStagBCList, void*);

// Manufactured solutions
static PetscScalar get_p(PetscScalar x, PetscScalar z, PetscScalar eta0, PetscScalar eps0, PetscScalar np, PetscScalar phi_0, PetscScalar phi_s, PetscScalar p_s, PetscScalar psi_s, PetscScalar U_s, PetscScalar m, PetscScalar n, PetscScalar k_hat, PetscScalar R)
{ PetscScalar result;
  result = p_s*cos(M_PI*m*x)*cos(M_PI*m*z);
  return(result);
}
static PetscScalar get_ux(PetscScalar x, PetscScalar z, PetscScalar eta0, PetscScalar eps0, PetscScalar np, PetscScalar phi_0, PetscScalar phi_s, PetscScalar p_s, PetscScalar psi_s, PetscScalar U_s, PetscScalar m, PetscScalar n, PetscScalar k_hat, PetscScalar R)
{ PetscScalar result;
  result = sin(M_PI*x)*sin(2.0*M_PI*z) + 2.0;
  return(result);
}
static PetscScalar get_uz(PetscScalar x, PetscScalar z, PetscScalar eta0, PetscScalar eps0, PetscScalar np, PetscScalar phi_0, PetscScalar phi_s, PetscScalar p_s, PetscScalar psi_s, PetscScalar U_s, PetscScalar m, PetscScalar n, PetscScalar k_hat, PetscScalar R)
{ PetscScalar result;
  result = 0.5*cos(M_PI*x)*cos(2.0*M_PI*z) + 2.0;
  return(result);
}
static PetscScalar get_phi(PetscScalar x, PetscScalar z, PetscScalar eta0, PetscScalar eps0, PetscScalar np, PetscScalar phi_0, PetscScalar phi_s, PetscScalar p_s, PetscScalar psi_s, PetscScalar U_s, PetscScalar m, PetscScalar n, PetscScalar k_hat, PetscScalar R)
{ PetscScalar result;
  result = phi_0*(phi_s*cos(M_PI*m*x)*cos(M_PI*m*z) + 1.0);
  return(result);
}
static PetscScalar get_Kphi(PetscScalar x, PetscScalar z, PetscScalar eta0, PetscScalar eps0, PetscScalar np, PetscScalar phi_0, PetscScalar phi_s, PetscScalar p_s, PetscScalar psi_s, PetscScalar U_s, PetscScalar m, PetscScalar n, PetscScalar k_hat, PetscScalar R)
{ PetscScalar result;
  result = pow(phi_s*cos(M_PI*m*x)*cos(M_PI*m*z) + 1.0, n);
  return(result);
}
static PetscScalar get_fux_stokes(PetscScalar x, PetscScalar z, PetscScalar eta0, PetscScalar eps0, PetscScalar np, PetscScalar phi_0, PetscScalar phi_s, PetscScalar p_s, PetscScalar psi_s, PetscScalar U_s, PetscScalar m, PetscScalar n, PetscScalar k_hat, PetscScalar R)
{ PetscScalar result;
  result = 2.0*M_PI*eta0*pow(sqrt(0.5625*pow(M_PI, 2)*pow(sin(M_PI*x), 2)*pow(cos(2.0*M_PI*z), 2) + 1.0*pow(M_PI, 2)*pow(sin(2.0*M_PI*z), 2)*pow(cos(M_PI*x), 2))/eps0, -1 + 1.0/np)*(-1 + 1.0/np)*1.0/(0.5625*pow(M_PI, 2)*pow(sin(M_PI*x), 2)*pow(cos(2.0*M_PI*z), 2) + 1.0*pow(M_PI, 2)*pow(sin(2.0*M_PI*z), 2)*pow(cos(M_PI*x), 2))*(-1.0*pow(M_PI, 3)*sin(M_PI*x)*pow(sin(2.0*M_PI*z), 2)*cos(M_PI*x) + 0.5625*pow(M_PI, 3)*sin(M_PI*x)*cos(M_PI*x)*pow(cos(2.0*M_PI*z), 2))*sin(2.0*M_PI*z)*cos(M_PI*x) + 1.5*M_PI*eta0*pow(sqrt(0.5625*pow(M_PI, 2)*pow(sin(M_PI*x), 2)*pow(cos(2.0*M_PI*z), 2) + 1.0*pow(M_PI, 2)*pow(sin(2.0*M_PI*z), 2)*pow(cos(M_PI*x), 2))/eps0, -1 + 1.0/np)*(-1 + 1.0/np)*1.0/(0.5625*pow(M_PI, 2)*pow(sin(M_PI*x), 2)*pow(cos(2.0*M_PI*z), 2) + 1.0*pow(M_PI, 2)*pow(sin(2.0*M_PI*z), 2)*pow(cos(M_PI*x), 2))*(-1.125*pow(M_PI, 3)*pow(sin(M_PI*x), 2)*sin(2.0*M_PI*z)*cos(2.0*M_PI*z) + 2.0*pow(M_PI, 3)*sin(2.0*M_PI*z)*pow(cos(M_PI*x), 2)*cos(2.0*M_PI*z))*sin(M_PI*x)*cos(2.0*M_PI*z) - 5.0*pow(M_PI, 2)*eta0*pow(sqrt(0.5625*pow(M_PI, 2)*pow(sin(M_PI*x), 2)*pow(cos(2.0*M_PI*z), 2) + 1.0*pow(M_PI, 2)*pow(sin(2.0*M_PI*z), 2)*pow(cos(M_PI*x), 2))/eps0, -1 + 1.0/np)*sin(M_PI*x)*sin(2.0*M_PI*z) + M_PI*m*p_s*sin(M_PI*m*x)*cos(M_PI*m*z);
  return(result);
}
static PetscScalar get_fuz_stokes(PetscScalar x, PetscScalar z, PetscScalar eta0, PetscScalar eps0, PetscScalar np, PetscScalar phi_0, PetscScalar phi_s, PetscScalar p_s, PetscScalar psi_s, PetscScalar U_s, PetscScalar m, PetscScalar n, PetscScalar k_hat, PetscScalar R)
{ PetscScalar result;
  result = 1.5*M_PI*eta0*pow(sqrt(0.5625*pow(M_PI, 2)*pow(sin(M_PI*x), 2)*pow(cos(2.0*M_PI*z), 2) + 1.0*pow(M_PI, 2)*pow(sin(2.0*M_PI*z), 2)*pow(cos(M_PI*x), 2))/eps0, -1 + 1.0/np)*(-1 + 1.0/np)*1.0/(0.5625*pow(M_PI, 2)*pow(sin(M_PI*x), 2)*pow(cos(2.0*M_PI*z), 2) + 1.0*pow(M_PI, 2)*pow(sin(2.0*M_PI*z), 2)*pow(cos(M_PI*x), 2))*(-1.0*pow(M_PI, 3)*sin(M_PI*x)*pow(sin(2.0*M_PI*z), 2)*cos(M_PI*x) + 0.5625*pow(M_PI, 3)*sin(M_PI*x)*cos(M_PI*x)*pow(cos(2.0*M_PI*z), 2))*sin(M_PI*x)*cos(2.0*M_PI*z) - 2.0*M_PI*eta0*pow(sqrt(0.5625*pow(M_PI, 2)*pow(sin(M_PI*x), 2)*pow(cos(2.0*M_PI*z), 2) + 1.0*pow(M_PI, 2)*pow(sin(2.0*M_PI*z), 2)*pow(cos(M_PI*x), 2))/eps0, -1 + 1.0/np)*(-1 + 1.0/np)*1.0/(0.5625*pow(M_PI, 2)*pow(sin(M_PI*x), 2)*pow(cos(2.0*M_PI*z), 2) + 1.0*pow(M_PI, 2)*pow(sin(2.0*M_PI*z), 2)*pow(cos(M_PI*x), 2))*(-1.125*pow(M_PI, 3)*pow(sin(M_PI*x), 2)*sin(2.0*M_PI*z)*cos(2.0*M_PI*z) + 2.0*pow(M_PI, 3)*sin(2.0*M_PI*z)*pow(cos(M_PI*x), 2)*cos(2.0*M_PI*z))*sin(2.0*M_PI*z)*cos(M_PI*x) - 2.5*pow(M_PI, 2)*eta0*pow(sqrt(0.5625*pow(M_PI, 2)*pow(sin(M_PI*x), 2)*pow(cos(2.0*M_PI*z), 2) + 1.0*pow(M_PI, 2)*pow(sin(2.0*M_PI*z), 2)*pow(cos(M_PI*x), 2))/eps0, -1 + 1.0/np)*cos(M_PI*x)*cos(2.0*M_PI*z) + M_PI*m*p_s*sin(M_PI*m*z)*cos(M_PI*m*x);
  return(result);
}
static PetscScalar get_fp_stokes(PetscScalar x, PetscScalar z, PetscScalar eta0, PetscScalar eps0, PetscScalar np, PetscScalar phi_0, PetscScalar phi_s, PetscScalar p_s, PetscScalar psi_s, PetscScalar U_s, PetscScalar m, PetscScalar n, PetscScalar k_hat, PetscScalar R)
{ PetscScalar result;
  result = 0;
  return(result);
}
static PetscScalar get_fux_stokesdarcy(PetscScalar x, PetscScalar z, PetscScalar eta0, PetscScalar eps0, PetscScalar np, PetscScalar phi_0, PetscScalar phi_s, PetscScalar p_s, PetscScalar psi_s, PetscScalar U_s, PetscScalar m, PetscScalar n, PetscScalar k_hat, PetscScalar R)
{ PetscScalar result;
  result = 2.0*M_PI*eta0*pow(sqrt(0.5625*pow(M_PI, 2)*pow(sin(M_PI*x), 2)*pow(cos(2.0*M_PI*z), 2) + 1.0*pow(M_PI, 2)*pow(sin(2.0*M_PI*z), 2)*pow(cos(M_PI*x), 2))/eps0, -1 + 1.0/np)*(-1 + 1.0/np)*1.0/(0.5625*pow(M_PI, 2)*pow(sin(M_PI*x), 2)*pow(cos(2.0*M_PI*z), 2) + 1.0*pow(M_PI, 2)*pow(sin(2.0*M_PI*z), 2)*pow(cos(M_PI*x), 2))*(-1.0*pow(M_PI, 3)*sin(M_PI*x)*pow(sin(2.0*M_PI*z), 2)*cos(M_PI*x) + 0.5625*pow(M_PI, 3)*sin(M_PI*x)*cos(M_PI*x)*pow(cos(2.0*M_PI*z), 2))*sin(2.0*M_PI*z)*cos(M_PI*x) + 1.5*M_PI*eta0*pow(sqrt(0.5625*pow(M_PI, 2)*pow(sin(M_PI*x), 2)*pow(cos(2.0*M_PI*z), 2) + 1.0*pow(M_PI, 2)*pow(sin(2.0*M_PI*z), 2)*pow(cos(M_PI*x), 2))/eps0, -1 + 1.0/np)*(-1 + 1.0/np)*1.0/(0.5625*pow(M_PI, 2)*pow(sin(M_PI*x), 2)*pow(cos(2.0*M_PI*z), 2) + 1.0*pow(M_PI, 2)*pow(sin(2.0*M_PI*z), 2)*pow(cos(M_PI*x), 2))*(-1.125*pow(M_PI, 3)*pow(sin(M_PI*x), 2)*sin(2.0*M_PI*z)*cos(2.0*M_PI*z) + 2.0*pow(M_PI, 3)*sin(2.0*M_PI*z)*pow(cos(M_PI*x), 2)*cos(2.0*M_PI*z))*sin(M_PI*x)*cos(2.0*M_PI*z) - 5.0*pow(M_PI, 2)*eta0*pow(sqrt(0.5625*pow(M_PI, 2)*pow(sin(M_PI*x), 2)*pow(cos(2.0*M_PI*z), 2) + 1.0*pow(M_PI, 2)*pow(sin(2.0*M_PI*z), 2)*pow(cos(M_PI*x), 2))/eps0, -1 + 1.0/np)*sin(M_PI*x)*sin(2.0*M_PI*z) + M_PI*m*p_s*sin(M_PI*m*x)*cos(M_PI*m*z);
  return(result);
}
static PetscScalar get_fuz_stokesdarcy(PetscScalar x, PetscScalar z, PetscScalar eta0, PetscScalar eps0, PetscScalar np, PetscScalar phi_0, PetscScalar phi_s, PetscScalar p_s, PetscScalar psi_s, PetscScalar U_s, PetscScalar m, PetscScalar n, PetscScalar k_hat, PetscScalar R)
{ PetscScalar result;
  result = 1.5*M_PI*eta0*pow(sqrt(0.5625*pow(M_PI, 2)*pow(sin(M_PI*x), 2)*pow(cos(2.0*M_PI*z), 2) + 1.0*pow(M_PI, 2)*pow(sin(2.0*M_PI*z), 2)*pow(cos(M_PI*x), 2))/eps0, -1 + 1.0/np)*(-1 + 1.0/np)*1.0/(0.5625*pow(M_PI, 2)*pow(sin(M_PI*x), 2)*pow(cos(2.0*M_PI*z), 2) + 1.0*pow(M_PI, 2)*pow(sin(2.0*M_PI*z), 2)*pow(cos(M_PI*x), 2))*(-1.0*pow(M_PI, 3)*sin(M_PI*x)*pow(sin(2.0*M_PI*z), 2)*cos(M_PI*x) + 0.5625*pow(M_PI, 3)*sin(M_PI*x)*cos(M_PI*x)*pow(cos(2.0*M_PI*z), 2))*sin(M_PI*x)*cos(2.0*M_PI*z) - 2.0*M_PI*eta0*pow(sqrt(0.5625*pow(M_PI, 2)*pow(sin(M_PI*x), 2)*pow(cos(2.0*M_PI*z), 2) + 1.0*pow(M_PI, 2)*pow(sin(2.0*M_PI*z), 2)*pow(cos(M_PI*x), 2))/eps0, -1 + 1.0/np)*(-1 + 1.0/np)*1.0/(0.5625*pow(M_PI, 2)*pow(sin(M_PI*x), 2)*pow(cos(2.0*M_PI*z), 2) + 1.0*pow(M_PI, 2)*pow(sin(2.0*M_PI*z), 2)*pow(cos(M_PI*x), 2))*(-1.125*pow(M_PI, 3)*pow(sin(M_PI*x), 2)*sin(2.0*M_PI*z)*cos(2.0*M_PI*z) + 2.0*pow(M_PI, 3)*sin(2.0*M_PI*z)*pow(cos(M_PI*x), 2)*cos(2.0*M_PI*z))*sin(2.0*M_PI*z)*cos(M_PI*x) - 2.5*pow(M_PI, 2)*eta0*pow(sqrt(0.5625*pow(M_PI, 2)*pow(sin(M_PI*x), 2)*pow(cos(2.0*M_PI*z), 2) + 1.0*pow(M_PI, 2)*pow(sin(2.0*M_PI*z), 2)*pow(cos(M_PI*x), 2))/eps0, -1 + 1.0/np)*cos(M_PI*x)*cos(2.0*M_PI*z) + k_hat*phi_0*(phi_s*cos(M_PI*m*x)*cos(M_PI*m*z) + 1.0) + M_PI*m*p_s*sin(M_PI*m*z)*cos(M_PI*m*x);
  return(result);
}
static PetscScalar get_fp_stokesdarcy(PetscScalar x, PetscScalar z, PetscScalar eta0, PetscScalar eps0, PetscScalar np, PetscScalar phi_0, PetscScalar phi_s, PetscScalar p_s, PetscScalar psi_s, PetscScalar U_s, PetscScalar m, PetscScalar n, PetscScalar k_hat, PetscScalar R)
{ PetscScalar result;
  result = pow(R, 2)*(-pow(M_PI, 2)*pow(m, 2)*n*p_s*phi_s*pow(phi_s*cos(M_PI*m*x)*cos(M_PI*m*z) + 1.0, n)*pow(sin(M_PI*m*x), 2)*pow(cos(M_PI*m*z), 2)/(phi_s*cos(M_PI*m*x)*cos(M_PI*m*z) + 1.0) + 2*pow(M_PI, 2)*pow(m, 2)*p_s*pow(phi_s*cos(M_PI*m*x)*cos(M_PI*m*z) + 1.0, n)*cos(M_PI*m*x)*cos(M_PI*m*z) + M_PI*m*n*phi_s*(-k_hat - M_PI*m*p_s*sin(M_PI*m*z)*cos(M_PI*m*x))*pow(phi_s*cos(M_PI*m*x)*cos(M_PI*m*z) + 1.0, n)*sin(M_PI*m*z)*cos(M_PI*m*x)/(phi_s*cos(M_PI*m*x)*cos(M_PI*m*z) + 1.0));
  return(result);
}

// ---------------------------------------
// Application functions
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "Numerical_solution"
PetscErrorCode Numerical_solution(void *ctx, PetscInt test)
{
  UsrData       *usr = (UsrData*) ctx;
  FDPDE          fd;
  DM             dm;
  Vec            x, xMMS;
  PetscInt       nx, nz;
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

  // Create the FD-pde object
  if (test==1) { ierr = FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_STOKES,&fd);CHKERRQ(ierr); }
  if (test==2) { ierr = FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_STOKESDARCY2FIELD,&fd);CHKERRQ(ierr); }

  ierr = FDPDESetUp(fd);CHKERRQ(ierr);
  ierr = FDPDEGetDM(fd,&dm); CHKERRQ(ierr);

  // Create manufactured solution
  ierr = ComputeManufacturedSolution(dm,&xMMS,usr); CHKERRQ(ierr);

  // Set BC evaluation function
  ierr = FDPDESetFunctionBCList(fd,FormBCList,"n/a",usr); CHKERRQ(ierr);

  // Set coefficients evaluation function
  if (test==1) { ierr = FDPDESetFunctionCoefficient(fd,FormCoefficient_Stokes,"n/a",usr); CHKERRQ(ierr); }
  if (test==2) { ierr = FDPDESetFunctionCoefficient(fd,FormCoefficient_StokesDarcy,"n/a",usr); CHKERRQ(ierr); }
  ierr = FDPDEView(fd); CHKERRQ(ierr);

  // Create initial guess with a linear viscous (np=1.0)
  PetscScalar np;
  Vec         xguess;
  np = usr->par->np;
  usr->par->np = 1.0;
  ierr = FDPDESolve(fd,NULL);CHKERRQ(ierr);
  ierr = FDPDEGetSolution(fd,&x);CHKERRQ(ierr); 
  ierr = FDPDEGetSolutionGuess(fd,&xguess); CHKERRQ(ierr); 
  ierr = VecCopy(x,xguess);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&xguess);CHKERRQ(ierr);
  usr->par->np = np;

  // FD SNES Solver
  ierr = FDPDESolve(fd,NULL);CHKERRQ(ierr);
  ierr = FDPDEGetSolution(fd,&x);CHKERRQ(ierr); 

  // Output solution to file
  if (test==1) { ierr = PetscSNPrintf(fout,sizeof(fout),"%s_stokes",usr->par->fname_out); }
  if (test==2) { ierr = PetscSNPrintf(fout,sizeof(fout),"%s_stokesdarcy",usr->par->fname_out); }
  ierr = DMStagViewBinaryPython(dm,x,fout);CHKERRQ(ierr);

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s_residual",usr->par->fname_out);
  ierr = DMStagViewBinaryPython(dm,fd->r,fout);CHKERRQ(ierr);

  // Compute norms
  ierr = ComputeErrorNorms(dm,x,xMMS,test);CHKERRQ(ierr);

  // Destroy objects
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&xMMS);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = FDPDEDestroy(&fd);CHKERRQ(ierr);

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
  ierr = PetscBagRegisterScalar(bag, &par->eta0, 1.0, "eta0", "Reference shear viscosity"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->eps0, 0.1, "eps0", "Reference background strainrate"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->np, 1.0, "nexp", "Power-law exponent"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->phi_0, 0.1, "phi_0", "Reference porosity"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->phi_s, 0.1, "phi_s", "Porosity amplitude"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->p_s, 1.0, "p_s", "Pressure amplitude"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->psi_s, 1.0, "psi_s", "Vector potential function amplitude"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->U_s, 1.0, "U_s", "Scalar potential function amplitude"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->m, 2.0, "m", "Trigonometric coefficient"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->n, 2.0, "n", "Porosity exponent"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->k_hat, 0.0, "k_hat", "Direction of unit vertical vector +/-1.0"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->R, 1.0, "R", "Compaction factor R = delta/h"); CHKERRQ(ierr);

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
  PetscPrintf(usr->comm,"# Test_effective_viscosity: %s \n",&(date[0]));
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
// FormCoefficient_Stokes
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormCoefficient_Stokes"
PetscErrorCode FormCoefficient_Stokes(FDPDE fd, DM dm, Vec x, DM dmcoeff, Vec coeff, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz;
  Vec            coefflocal, xlocal;
  PetscScalar    **coordx,**coordz;
  PetscInt       iprev, inext, icenter;
  PetscScalar    eta0,eps0,np,R,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat;
  PetscScalar    ***c;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  eta0 = usr->par->eta0;
  eps0 = usr->par->eps0;
  np   = usr->par->np;
  R    = usr->par->R;
  phi_0= usr->par->phi_0;
  phi_s= usr->par->phi_s;
  p_s  = usr->par->p_s;
  psi_s= usr->par->psi_s;
  U_s  = usr->par->U_s;
  m    = usr->par->m;
  n    = usr->par->n;
  k_hat= usr->par->k_hat;

  // Get domain corners
  ierr = DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = DMStagGetGlobalSizes(dmcoeff,&Nx,&Nz,NULL);CHKERRQ(ierr);

  // Get dm coordinates array
  ierr = DMStagGet1dCoordinateArraysDOFRead(dmcoeff,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGet1dCoordinateLocationSlot(dmcoeff,LEFT,&iprev);CHKERRQ(ierr);
  ierr = DMStagGet1dCoordinateLocationSlot(dmcoeff,RIGHT,&inext);CHKERRQ(ierr); 
  ierr = DMStagGet1dCoordinateLocationSlot(dmcoeff,ELEMENT,&icenter);CHKERRQ(ierr);

  // Create coefficient local vector
  ierr = DMCreateLocalVector(dmcoeff, &coefflocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayDOF(dmcoeff, coefflocal, &c); CHKERRQ(ierr);

  ierr = DMGetLocalVector(dm, &xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm, x, INSERT_VALUES, xlocal); CHKERRQ(ierr);
  
  // Loop over local domain - set initial density and viscosity
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      PetscInt idx;

      { // A = eta (center, c=1)
        DMStagStencil point;
        PetscScalar   eta,epsII,exx,ezz,exz;
        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 1;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);

        // get effective viscosity
        ierr = DMStagGetPointStrainRates(dm,xlocal,1,&point,&epsII,&exx,&ezz,&exz); CHKERRQ(ierr);
        eta = eta0*PetscPowScalar(epsII/eps0,1.0/np-1.0);
        c[j][i][idx] = eta;
      }

      { // A = eta (corner, c=0)
        DMStagStencil point[4];
        PetscScalar   eta,epsII[4],exx[4],ezz[4],exz[4];
        PetscInt      ii;

        point[0].i = i; point[0].j = j; point[0].loc = DOWN_LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = DOWN_RIGHT; point[1].c = 0;
        point[2].i = i; point[2].j = j; point[2].loc = UP_LEFT;    point[2].c = 0;
        point[3].i = i; point[3].j = j; point[3].loc = UP_RIGHT;   point[3].c = 0;

        // get effective viscosity
        ierr = DMStagGetPointStrainRates(dm,xlocal,4,point,epsII,exx,ezz,exz); CHKERRQ(ierr);

        for (ii = 0; ii < 4; ii++) {
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          // if (PetscAbsScalar(epsII[ii])>0) eta = eta0*PetscPowScalar(epsII[ii]/eps0,1.0/np-1.0);
          // else eta = eta0;
          eta = eta0*PetscPowScalar(epsII[ii]/eps0,1.0/np-1.0);
          c[j][i][idx] = eta;
        }
      }

      { // B = fu (fux,fuz - manufactured) (edges, c=0)
        DMStagStencil point[4];
        PetscScalar   xp[4],zp[4],rhs[4];
        PetscInt      ii;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 0;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = 0;
        point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = 0;

        xp[0] = coordx[i][iprev  ]; zp[0] = coordz[j][icenter];
        xp[1] = coordx[i][inext  ]; zp[1] = coordz[j][icenter];
        xp[2] = coordx[i][icenter]; zp[2] = coordz[j][iprev  ];
        xp[3] = coordx[i][icenter]; zp[3] = coordz[j][inext  ];

        rhs[0] = get_fux_stokes(xp[0],zp[0],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
        rhs[1] = get_fux_stokes(xp[1],zp[1],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
        rhs[2] = get_fuz_stokes(xp[2],zp[2],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
        rhs[3] = get_fuz_stokes(xp[3],zp[3],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);

        for (ii = 0; ii < 4; ii++) {
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = rhs[ii];
        }
      }

      { // C = fp (manufactured) (center, c=0)
        DMStagStencil point;
        PetscScalar   xp, zp;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 0;
        xp = coordx[i][icenter]; zp = coordz[j][icenter];
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = get_fp_stokes(xp,zp,eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
      }
    }
  }

  // Restore arrays, local vectors
  ierr = DMStagRestore1dCoordinateArraysDOFRead(dmcoeff,&coordx,&coordz,NULL);CHKERRQ(ierr);

  ierr = DMStagVecRestoreArrayDOF(dmcoeff,coefflocal,&c);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&xlocal); CHKERRQ(ierr);
  
  ierr = VecDestroy(&coefflocal); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// FormCoefficient_StokesDarcy
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormCoefficient_StokesDarcy"
PetscErrorCode FormCoefficient_StokesDarcy(FDPDE fd, DM dm, Vec x, DM dmcoeff, Vec coeff, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       i, j, sx, sz, nx, nz;
  Vec            coefflocal, xlocal;
  PetscScalar    **coordx,**coordz;
  PetscInt       iprev, inext, icenter;
  PetscScalar    eta0,eps0,np,R,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat;
  PetscScalar    ***c;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  eta0 = usr->par->eta0;
  eps0 = usr->par->eps0;
  np   = usr->par->np;
  R    = usr->par->R;
  phi_0= usr->par->phi_0;
  phi_s= usr->par->phi_s;
  p_s  = usr->par->p_s;
  psi_s= usr->par->psi_s;
  U_s  = usr->par->U_s;
  m    = usr->par->m;
  n    = usr->par->n;
  k_hat= usr->par->k_hat;

  // Get domain corners
  ierr = DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  // Get dm coordinates array
  ierr = DMStagGet1dCoordinateArraysDOFRead(dmcoeff,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGet1dCoordinateLocationSlot(dmcoeff,LEFT,&iprev);CHKERRQ(ierr);
  ierr = DMStagGet1dCoordinateLocationSlot(dmcoeff,RIGHT,&inext);CHKERRQ(ierr); 
  ierr = DMStagGet1dCoordinateLocationSlot(dmcoeff,ELEMENT,&icenter);CHKERRQ(ierr);

  // Create coefficient local vector
  ierr = DMCreateLocalVector(dmcoeff, &coefflocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayDOF(dmcoeff, coefflocal, &c); CHKERRQ(ierr);

  ierr = DMGetLocalVector(dm, &xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm, x, INSERT_VALUES, xlocal); CHKERRQ(ierr);
  
  // Loop over local domain - set initial density and viscosity
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      PetscInt idx;

      { // A = eta (center, c=1)
        DMStagStencil point;
        PetscScalar   eta,epsII,exx,ezz,exz;
        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 1;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);

        // get effective viscosity
        ierr = DMStagGetPointStrainRates(dm,xlocal,1,&point,&epsII,&exx,&ezz,&exz); CHKERRQ(ierr);
        eta = eta0*PetscPowScalar(epsII/eps0,1.0/np-1.0);
        c[j][i][idx] = eta;
      }

      { // A = eta (corner, c=0)
        DMStagStencil point[4];
        PetscScalar   eta,epsII[4],exx[4],ezz[4],exz[4];
        PetscInt      ii;

        point[0].i = i; point[0].j = j; point[0].loc = DOWN_LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = DOWN_RIGHT; point[1].c = 0;
        point[2].i = i; point[2].j = j; point[2].loc = UP_LEFT;    point[2].c = 0;
        point[3].i = i; point[3].j = j; point[3].loc = UP_RIGHT;   point[3].c = 0;

        // get effective viscosity
        ierr = DMStagGetPointStrainRates(dm,xlocal,4,point,epsII,exx,ezz,exz); CHKERRQ(ierr);

        for (ii = 0; ii < 4; ii++) {
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          // if (PetscAbsScalar(epsII[ii])>0) eta = eta0*PetscPowScalar(epsII[ii]/eps0,1.0/np-1.0);
          // else eta = eta0;
          eta = eta0*PetscPowScalar(epsII[ii]/eps0,1.0/np-1.0);
          c[j][i][idx] = eta;
        }
      }

      { // B = -phi*k_hat + fu (fux,fuz - manufactured) (edges, c=0)
        DMStagStencil point[4];
        PetscScalar   xp[4],zp[4],rhs[4];
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
        rhs[0] = get_fux_stokesdarcy(xp[0],zp[0],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
        rhs[1] = get_fux_stokesdarcy(xp[1],zp[1],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);

        // Bz = -phi*k_hat + fuz
        rhs[2] = get_fuz_stokesdarcy(xp[2],zp[2],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
        rhs[3] = get_fuz_stokesdarcy(xp[3],zp[3],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);

        rhs[2] -= k_hat*get_phi(xp[2],zp[2],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
        rhs[3] -= k_hat*get_phi(xp[3],zp[3],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);

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
        c[j][i][idx] = get_fp_stokesdarcy(xp,zp,eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
      }

      { // D1 = xi=zeta-2/3eta (center, c=2)
        DMStagStencil point;
        PetscScalar   phi,xi,zeta,eta,epsII,exx,ezz,exz;
        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 2;

        // get effective viscosity
        ierr = DMStagGetPointStrainRates(dm,xlocal,1,&point,&epsII,&exx,&ezz,&exz); CHKERRQ(ierr);
        eta  = eta0*PetscPowScalar(epsII/eps0,1.0/np-1.0);

        phi  = get_phi(coordx[i][icenter],coordz[j][icenter],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
        zeta = eta/phi;
        xi   = zeta-2.0/3.0*eta;

        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = xi;
      }

      { // D2 = -R^2*Kphi (edges, c=1)
        DMStagStencil point[4];
        PetscScalar   xp[4],zp[4];
        PetscInt      ii;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 1;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 1;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = 1;
        point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = 1;

        xp[0] = coordx[i][iprev  ]; zp[0] = coordz[j][icenter];
        xp[1] = coordx[i][inext  ]; zp[1] = coordz[j][icenter];
        xp[2] = coordx[i][icenter]; zp[2] = coordz[j][iprev  ];
        xp[3] = coordx[i][icenter]; zp[3] = coordz[j][inext  ];

        for (ii = 0; ii < 4; ii++) {
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = -R*R*get_Kphi(xp[ii],zp[ii],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
        }
      }

      { // D3 = R^2*Kphi*k_hat (edges, c=2)
        DMStagStencil point[4];
        PetscScalar   xp[4],zp[4];
        PetscInt      ii;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 2;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 2;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = 2;
        point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = 2;

        xp[0] = coordx[i][iprev  ]; zp[0] = coordz[j][icenter];
        xp[1] = coordx[i][inext  ]; zp[1] = coordz[j][icenter];
        xp[2] = coordx[i][icenter]; zp[2] = coordz[j][iprev  ];
        xp[3] = coordx[i][icenter]; zp[3] = coordz[j][inext  ];

        for (ii = 0; ii < 2; ii++) {
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = 0.0;
        }

        for (ii = 2; ii < 4; ii++) {
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = R*R*k_hat*get_Kphi(xp[ii],zp[ii],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
        }
      }
    }
  }

  // Restore arrays, local vectors
  ierr = DMStagRestore1dCoordinateArraysDOFRead(dmcoeff,&coordx,&coordz,NULL);CHKERRQ(ierr);

  ierr = DMStagVecRestoreArrayDOF(dmcoeff,coefflocal,&c);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&xlocal); CHKERRQ(ierr);
  
  ierr = VecDestroy(&coefflocal); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// FormBCList
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormBCList"
PetscErrorCode FormBCList(DM dm, Vec x, DMStagBCList bclist, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       k,n_bc,*idx_bc;
  PetscScalar    *value_bc,*x_bc;
  PetscScalar    eta0,eps0,np,R,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat;
  BCType         *type_bc;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;

  eta0 = usr->par->eta0;
  eps0 = usr->par->eps0;
  np   = usr->par->np;
  R    = usr->par->R;
  phi_0= usr->par->phi_0;
  phi_s= usr->par->phi_s;
  p_s  = usr->par->p_s;
  psi_s= usr->par->psi_s;
  U_s  = usr->par->U_s;
  m    = usr->par->m;
  n    = usr->par->n;
  k_hat= usr->par->k_hat;

  // LEFT Boundary - Vx
  ierr = DMStagBCListGetValues(bclist,'w','-',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_ux(x_bc[2*k],x_bc[2*k+1],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // LEFT Boundary - Vz
  ierr = DMStagBCListGetValues(bclist,'w','|',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_uz(x_bc[2*k],x_bc[2*k+1],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // LEFT Boundary - P
  ierr = DMStagBCListGetValues(bclist,'w','o',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_p(x_bc[2*k],x_bc[2*k+1],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  
  // RIGHT Boundary - Vx
  ierr = DMStagBCListGetValues(bclist,'e','-',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_ux(x_bc[2*k],x_bc[2*k+1],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // RIGHT Boundary - Vz
  ierr = DMStagBCListGetValues(bclist,'e','|',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_uz(x_bc[2*k],x_bc[2*k+1],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // RIGHT Boundary - P
  ierr = DMStagBCListGetValues(bclist,'e','o',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_p(x_bc[2*k],x_bc[2*k+1],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN Boundary - Vx
  ierr = DMStagBCListGetValues(bclist,'s','-',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_ux(x_bc[2*k],x_bc[2*k+1],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN Boundary - Vz
  ierr = DMStagBCListGetValues(bclist,'s','|',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_uz(x_bc[2*k],x_bc[2*k+1],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN Boundary - P
  ierr = DMStagBCListGetValues(bclist,'s','o',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_p(x_bc[2*k],x_bc[2*k+1],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP Boundary - Vx
  ierr = DMStagBCListGetValues(bclist,'n','-',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_ux(x_bc[2*k],x_bc[2*k+1],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP Boundary - Vz
  ierr = DMStagBCListGetValues(bclist,'n','|',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_uz(x_bc[2*k],x_bc[2*k+1],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP Boundary - P
  ierr = DMStagBCListGetValues(bclist,'n','o',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_p(x_bc[2*k],x_bc[2*k+1],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// Compute manufactured solution
// ---------------------------------------
PetscErrorCode ComputeManufacturedSolution(DM dm,Vec *_xMMS, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz, idx;
  PetscInt       iprev, inext, icenter;
  PetscScalar    ***xxMMS;
  PetscScalar    **coordx,**coordz;
  Vec            xMMS, xMMSlocal;
  PetscScalar    eta0,eps0,np,R,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  eta0 = usr->par->eta0;
  eps0 = usr->par->eps0;
  np   = usr->par->np;
  R    = usr->par->R;
  phi_0= usr->par->phi_0;
  phi_s= usr->par->phi_s;
  p_s  = usr->par->p_s;
  psi_s= usr->par->psi_s;
  U_s  = usr->par->U_s;
  m    = usr->par->m;
  n    = usr->par->n;
  k_hat= usr->par->k_hat;

  // Create local and global vectors for MMS solution
  ierr = DMCreateGlobalVector(dm,&xMMS     ); CHKERRQ(ierr);
  ierr = DMCreateLocalVector (dm,&xMMSlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayDOF(dm,xMMSlocal,&xxMMS); CHKERRQ(ierr);

  // Get domain corners
  ierr = DMStagGetGlobalSizes(dm, &Nx, &Nz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  
// Get dm coordinates array
  ierr = DMStagGet1dCoordinateArraysDOFRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGet1dCoordinateLocationSlot(dm,ELEMENT,&icenter);CHKERRQ(ierr); 
  ierr = DMStagGet1dCoordinateLocationSlot(dm,LEFT,&iprev);CHKERRQ(ierr);
  ierr = DMStagGet1dCoordinateLocationSlot(dm,RIGHT,&inext);CHKERRQ(ierr); 

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {

      // pressure
      ierr = DMStagGetLocationSlot(dm,ELEMENT,0,&idx); CHKERRQ(ierr);
      xxMMS[j][i][idx] = get_p(coordx[i][icenter],coordz[j][icenter],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);

      // ux
      ierr = DMStagGetLocationSlot(dm,LEFT,0,&idx); CHKERRQ(ierr);
      xxMMS[j][i][idx] = get_ux(coordx[i][iprev],coordz[j][icenter],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);

      if (i == Nx-1) {
        ierr = DMStagGetLocationSlot(dm,RIGHT,0,&idx); CHKERRQ(ierr);
        xxMMS[j][i][idx] = get_ux(coordx[i][inext],coordz[j][icenter],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
      }
      
      // uz
      ierr = DMStagGetLocationSlot(dm,DOWN,0,&idx); CHKERRQ(ierr);
      xxMMS[j][i][idx] = get_uz(coordx[i][icenter],coordz[j][iprev],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);

      if (j == Nz-1) {
        ierr = DMStagGetLocationSlot(dm,UP,0,&idx); CHKERRQ(ierr);
        xxMMS[j][i][idx] = get_uz(coordx[i][icenter],coordz[j][inext],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
      }
    }
  }

  // Restore arrays
  ierr = DMStagRestore1dCoordinateArraysDOFRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);

  // Restore and map local to global
  ierr = DMStagVecRestoreArrayDOF(dm,xMMSlocal,&xxMMS); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm,xMMSlocal,INSERT_VALUES,xMMS); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,xMMSlocal,INSERT_VALUES,xMMS); CHKERRQ(ierr);
  ierr = VecDestroy(&xMMSlocal); CHKERRQ(ierr);
  ierr = DMStagViewBinaryPython(dm,xMMS,"out_mms_solution");CHKERRQ(ierr);

  // Assign pointers
  *_xMMS  = xMMS;
  
  PetscFunctionReturn(0);
}

// ---------------------------------------
// ComputeErrorNorms
// ---------------------------------------
PetscErrorCode ComputeErrorNorms(DM dm,Vec x,Vec xMMS,PetscInt test)
{
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz;
  PetscScalar    xx[5], xa[5], dx, dz, dv;
  PetscScalar    nrm2p, nrm2v, nrm2vx, nrm2vz, sum_err[3], gsum_err[3], sum_mms[3], gsum_mms[3];
  Vec            xlocal, xalocal;
  MPI_Comm       comm;

  PetscErrorCode ierr;
  PetscFunctionBegin;

  comm = PETSC_COMM_WORLD;

  // Get domain corners
  ierr = DMStagGetGlobalSizes(dm, &Nx, &Nz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  // Map global vectors to local domain
  ierr = DMGetLocalVector(dm, &xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm, x, INSERT_VALUES, xlocal); CHKERRQ(ierr);

  ierr = DMGetLocalVector(dm, &xalocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm, xMMS, INSERT_VALUES, xalocal); CHKERRQ(ierr);

  // Initialize norms
  sum_err[0] = 0.0; sum_err[1] = 0.0; sum_err[2] = 0.0;
  sum_mms[0] = 0.0; sum_mms[1] = 0.0; sum_mms[2] = 0.0;

  dx = 1.0/Nx;
  dz = 1.0/Nz;
  dv = dx*dz;

  // Loop over local domain to calculate ELEMENT errors
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      
      PetscScalar    ve[4], pe, v[4], p;
      DMStagStencil  point[5];
      
      // Get stencil values
      point[0].i = i; point[0].j = j; point[0].loc = LEFT;    point[0].c = 0; // Vx
      point[1].i = i; point[1].j = j; point[1].loc = RIGHT;   point[1].c = 0; // Vx
      point[2].i = i; point[2].j = j; point[2].loc = DOWN;    point[2].c = 0; // Vz
      point[3].i = i; point[3].j = j; point[3].loc = UP;      point[3].c = 0; // Vz
      point[4].i = i; point[4].j = j; point[4].loc = ELEMENT; point[4].c = 0; // P

      // Get numerical solution
      ierr = DMStagVecGetValuesStencil(dm, xlocal, 5, point, xx); CHKERRQ(ierr);

      // Get analytical solution
      ierr = DMStagVecGetValuesStencil(dm, xalocal, 5, point, xa); CHKERRQ(ierr);

      // Error vectors - squared
      ve[0] = (xx[0]-xa[0])*(xx[0]-xa[0]); // Left
      ve[1] = (xx[1]-xa[1])*(xx[1]-xa[1]); // Right
      ve[2] = (xx[2]-xa[2])*(xx[2]-xa[2]); // Down
      ve[3] = (xx[3]-xa[3])*(xx[3]-xa[3]); // Up
      pe    = (xx[4]-xa[4])*(xx[4]-xa[4]); // elem

      // MMS values - squared
      v[0] = xa[0]*xa[0]; // Left
      v[1] = xa[1]*xa[1]; // Right
      v[2] = xa[2]*xa[2]; // Down
      v[3] = xa[3]*xa[3]; // Up
      p    = xa[4]*xa[4]; // elem

      // Calculate sums for L2 norms as in Katz, ch 13 - but taking into account the staggered grid
      if      (i == 0   ) { sum_err[0] += ve[0]*dv*0.5; sum_err[0] += ve[1]*dv; }
      else if (i == Nx-1) sum_err[0] += ve[1]*dv*0.5;
      else                sum_err[0] += ve[1]*dv;

      if      (j == 0   ) { sum_err[1] += ve[2]*dv*0.5; sum_err[1] += ve[3]*dv; }
      else if (j == Nz-1) sum_err[1] += ve[3]*dv*0.5;
      else                sum_err[1] += ve[3]*dv;
      sum_err[2] += pe*dv;

      if      (i == 0   ) { sum_mms[0] += v[0]*dv*0.5; sum_mms[0] += v[1]*dv; }
      else if (i == Nx-1) sum_mms[0] += v[1]*dv*0.5;
      else                sum_mms[0] += v[1]*dv;

      if      (j == 0   ) { sum_mms[1] += v[2]*dv*0.5; sum_mms[1] += v[3]*dv; }
      else if (j == Nz-1) sum_mms[1] += v[3]*dv*0.5;
      else                sum_mms[1] += v[3]*dv;
      sum_mms[2] += p*dv;
    }
  }

  // Collect data 
  ierr = MPI_Allreduce(&sum_err, &gsum_err, 3, MPI_DOUBLE, MPI_SUM, comm); CHKERRQ(ierr);
  ierr = MPI_Allreduce(&sum_mms, &gsum_mms, 3, MPI_DOUBLE, MPI_SUM, comm); CHKERRQ(ierr);

  // L2 error norm = sqrt(gsum_err/gsum_mms)
  nrm2vx = PetscSqrtScalar(gsum_err[0]/gsum_mms[0]);
  nrm2vz = PetscSqrtScalar(gsum_err[1]/gsum_mms[1]);
  nrm2p  = PetscSqrtScalar(gsum_err[2]/gsum_mms[2]);
  nrm2v  = PetscSqrtScalar((gsum_err[0]+gsum_err[1])/(gsum_mms[0]+gsum_mms[1]));

  // Restore arrays and vectors
  ierr = DMRestoreLocalVector(dm, &xlocal ); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &xalocal ); CHKERRQ(ierr);

  // Print information
  PetscPrintf(comm,"# --------------------------------------- #\n");
  PetscPrintf(comm,"# NORMS: \n");
  PetscPrintf(comm,"# Velocity test%d: norm2 = %1.12e norm2x = %1.12e norm2z = %1.12e \n",test,nrm2v,nrm2vx,nrm2vz);
  PetscPrintf(comm,"# Pressure test%d: norm2 = %1.12e\n",test,nrm2p);
  PetscPrintf(comm,"# Grid info test%d: hx = %1.12e hz = %1.12e\n",test,dx,dz);

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

  // Numerical solution using the FD pde object - Stokes
  ierr = PetscTime(&start_time); CHKERRQ(ierr);
  ierr = Numerical_solution(usr,1); CHKERRQ(ierr);
  ierr = PetscTime(&end_time); CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"# Runtime Stokes: %g (sec) \n", end_time - start_time);
  PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");

  // Numerical solution using the FD pde object - StokesDarcy
  ierr = PetscTime(&start_time); CHKERRQ(ierr);
  ierr = Numerical_solution(usr,2); CHKERRQ(ierr);
  ierr = PetscTime(&end_time); CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"# Runtime StokesDarcy: %g (sec) \n", end_time - start_time);
  PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");

  // Destroy objects
  ierr = PetscBagDestroy(&usr->bag); CHKERRQ(ierr);
  ierr = PetscFree(usr);             CHKERRQ(ierr);

  // Finalize main
  ierr = PetscFinalize();
  return ierr;
}
