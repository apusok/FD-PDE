// ---------------------------------------
// MMS test for Katz, 2019, Ch 13.
// run: ./tests/test_stokesdarcy2field_mms_katz_ch13.app -pc_type lu -pc_factor_mat_solver_type umfpack -nx 10 -nz 10
// python test: ./tests/python/test_stokesdarcy2field_mms_katz_ch13.py
// python sympy: ./mms/mms_katz_ch13_stokes_darcy.py
// ---------------------------------------
static char help[] = "Application to verify two-phase flow implementation with an MMS (Katz, Ch 13) \n\n";

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
  PetscScalar    delta,phi_0,phi_s,p_s,psi_s,U_s,m,n,e3;
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
PetscErrorCode StokesDarcy_Numerical(void*);
PetscErrorCode ComputeManufacturedSolution(DM,Vec*,void*);
PetscErrorCode ComputeExtraParameters(DM,void*);
PetscErrorCode ComputeErrorNorms(DM,Vec,Vec);
PetscErrorCode InputParameters(UsrData**);
PetscErrorCode InputPrintData(UsrData*);
PetscErrorCode FormCoefficient(FDPDE, DM, Vec, DM, Vec, void*);
PetscErrorCode FormBCList(DM, Vec, DMStagBCList, void*);

// Manufactured solutions
// Note: get_fp/fux/fuz_potential are calculated in potential form (eq 13.37) - but they are identical to get_fp, get_fux, f_uz
static PetscScalar get_p(PetscScalar x, PetscScalar z, PetscScalar delta, PetscScalar phi_0, PetscScalar phi_s, PetscScalar p_s,PetscScalar psi_s, PetscScalar U_s, PetscScalar m, PetscScalar n, PetscScalar e3)
{ PetscScalar result;
  result = p_s*cos(M_PI*m*x)*cos(M_PI*m*z);
  return(result);
}
static PetscScalar get_ux(PetscScalar x, PetscScalar z, PetscScalar delta, PetscScalar phi_0, PetscScalar phi_s, PetscScalar p_s,PetscScalar psi_s, PetscScalar U_s, PetscScalar m, PetscScalar n, PetscScalar e3)
{ PetscScalar result;
  result = M_PI*U_s*m*sin(M_PI*m*x)*cos(M_PI*m*z) + M_PI*m*psi_s*(1.0 - cos(M_PI*m*x))*sin(M_PI*m*z);
  return(result);
}
static PetscScalar get_uz(PetscScalar x, PetscScalar z, PetscScalar delta, PetscScalar phi_0, PetscScalar phi_s, PetscScalar p_s,PetscScalar psi_s, PetscScalar U_s, PetscScalar m, PetscScalar n, PetscScalar e3)
{ PetscScalar result;
  result = M_PI*U_s*m*sin(M_PI*m*z)*cos(M_PI*m*x) - M_PI*m*psi_s*(1.0 - cos(M_PI*m*z))*sin(M_PI*m*x);
  return(result);
}
static PetscScalar get_fux(PetscScalar x, PetscScalar z, PetscScalar delta, PetscScalar phi_0, PetscScalar phi_s, PetscScalar p_s,PetscScalar psi_s, PetscScalar U_s, PetscScalar m, PetscScalar n, PetscScalar e3)
{ PetscScalar result;
  result = -2*pow(M_PI, 3)*U_s*pow(delta, 2)*pow(m, 3)*sin(M_PI*m*x)*cos(M_PI*m*z) + pow(delta, 2)*(-4.0*pow(M_PI, 3)*U_s*pow(m, 3)*sin(M_PI*m*x)*cos(M_PI*m*z) - pow(M_PI, 3)*pow(m, 3)*psi_s*(1.0 - cos(M_PI*m*x))*sin(M_PI*m*z) + 1.0*pow(M_PI, 3)*pow(m, 3)*psi_s*sin(M_PI*m*z)*cos(M_PI*m*x)) + M_PI*m*p_s*sin(M_PI*m*x)*cos(M_PI*m*z);
  return(result);
}
static PetscScalar get_fuz(PetscScalar x, PetscScalar z, PetscScalar delta, PetscScalar phi_0, PetscScalar phi_s, PetscScalar p_s,PetscScalar psi_s, PetscScalar U_s, PetscScalar m, PetscScalar n, PetscScalar e3)
{ PetscScalar result;
  result = -2*pow(M_PI, 3)*U_s*pow(delta, 2)*pow(m, 3)*sin(M_PI*m*z)*cos(M_PI*m*x) + pow(delta, 2)*(-4.0*pow(M_PI, 3)*U_s*pow(m, 3)*sin(M_PI*m*z)*cos(M_PI*m*x) + pow(M_PI, 3)*pow(m, 3)*psi_s*(1.0 - cos(M_PI*m*z))*sin(M_PI*m*x) - 1.0*pow(M_PI, 3)*pow(m, 3)*psi_s*sin(M_PI*m*x)*cos(M_PI*m*z)) - e3*phi_0*(phi_s*cos(M_PI*m*x)*cos(M_PI*m*z) + 1.0) + M_PI*m*p_s*sin(M_PI*m*z)*cos(M_PI*m*x);
  return(result);
}
static PetscScalar get_fp(PetscScalar x, PetscScalar z, PetscScalar delta, PetscScalar phi_0, PetscScalar phi_s, PetscScalar p_s,PetscScalar psi_s, PetscScalar U_s, PetscScalar m, PetscScalar n, PetscScalar e3)
{ PetscScalar result;
  result = 2*pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) - pow(M_PI, 2)*pow(m, 2)*n*p_s*phi_s*pow(phi_s*cos(M_PI*m*x)*cos(M_PI*m*z) + 1.0, n)*pow(sin(M_PI*m*x), 2)*pow(cos(M_PI*m*z), 2)/(phi_s*cos(M_PI*m*x)*cos(M_PI*m*z) + 1.0) + 2*pow(M_PI, 2)*pow(m, 2)*p_s*pow(phi_s*cos(M_PI*m*x)*cos(M_PI*m*z) + 1.0, n)*cos(M_PI*m*x)*cos(M_PI*m*z) + M_PI*m*n*phi_s*(e3 - M_PI*m*p_s*sin(M_PI*m*z)*cos(M_PI*m*x))*pow(phi_s*cos(M_PI*m*x)*cos(M_PI*m*z) + 1.0, n)*sin(M_PI*m*z)*cos(M_PI*m*x)/(phi_s*cos(M_PI*m*x)*cos(M_PI*m*z) + 1.0);
  return(result);
}
// static PetscScalar get_fux_potential(PetscScalar x, PetscScalar z, PetscScalar delta, PetscScalar phi_0, PetscScalar phi_s, PetscScalar p_s,PetscScalar psi_s, PetscScalar U_s, PetscScalar m, PetscScalar n, PetscScalar e3)
// { PetscScalar result;
//   result = -2*pow(M_PI, 3)*U_s*pow(delta, 2)*pow(m, 3)*sin(M_PI*m*x)*cos(M_PI*m*z) + pow(delta, 2)*(-4.0*pow(M_PI, 3)*U_s*pow(m, 3)*sin(M_PI*m*x)*cos(M_PI*m*z) - pow(M_PI, 3)*pow(m, 3)*psi_s*(1.0 - cos(M_PI*m*x))*sin(M_PI*m*z) + 1.0*pow(M_PI, 3)*pow(m, 3)*psi_s*sin(M_PI*m*z)*cos(M_PI*m*x)) + M_PI*m*p_s*sin(M_PI*m*x)*cos(M_PI*m*z);
//   return(result);
// }
// static PetscScalar get_fuz_potential(PetscScalar x, PetscScalar z, PetscScalar delta, PetscScalar phi_0, PetscScalar phi_s, PetscScalar p_s,PetscScalar psi_s, PetscScalar U_s, PetscScalar m, PetscScalar n, PetscScalar e3)
// { PetscScalar result;
//   result = -2*pow(M_PI, 3)*U_s*pow(delta, 2)*pow(m, 3)*sin(M_PI*m*z)*cos(M_PI*m*x) + pow(delta, 2)*(-4.0*pow(M_PI, 3)*U_s*pow(m, 3)*sin(M_PI*m*z)*cos(M_PI*m*x) + pow(M_PI, 3)*pow(m, 3)*psi_s*(1.0 - cos(M_PI*m*z))*sin(M_PI*m*x) - 1.0*pow(M_PI, 3)*pow(m, 3)*psi_s*sin(M_PI*m*x)*cos(M_PI*m*z)) - e3*phi_0*(phi_s*cos(M_PI*m*x)*cos(M_PI*m*z) + 1.0) + M_PI*m*p_s*sin(M_PI*m*z)*cos(M_PI*m*x);
//   return(result);
// }
// static PetscScalar get_fp_potential(PetscScalar x, PetscScalar z, PetscScalar delta, PetscScalar phi_0, PetscScalar phi_s, PetscScalar p_s,PetscScalar psi_s, PetscScalar U_s, PetscScalar m, PetscScalar n, PetscScalar e3)
// { PetscScalar result;
//   result = 2*pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) - pow(M_PI, 2)*pow(m, 2)*n*p_s*phi_s*pow(phi_s*cos(M_PI*m*x)*cos(M_PI*m*z) + 1.0, n)*pow(sin(M_PI*m*x), 2)*pow(cos(M_PI*m*z), 2)/(phi_s*cos(M_PI*m*x)*cos(M_PI*m*z) + 1.0) + 2*pow(M_PI, 2)*pow(m, 2)*p_s*pow(phi_s*cos(M_PI*m*x)*cos(M_PI*m*z) + 1.0, n)*cos(M_PI*m*x)*cos(M_PI*m*z) + M_PI*m*n*phi_s*(e3 - M_PI*m*p_s*sin(M_PI*m*z)*cos(M_PI*m*x))*pow(phi_s*cos(M_PI*m*x)*cos(M_PI*m*z) + 1.0, n)*sin(M_PI*m*z)*cos(M_PI*m*x)/(phi_s*cos(M_PI*m*x)*cos(M_PI*m*z) + 1.0);
//   return(result);
// }
static PetscScalar get_Kphi(PetscScalar x, PetscScalar z, PetscScalar delta, PetscScalar phi_0, PetscScalar phi_s, PetscScalar p_s,PetscScalar psi_s, PetscScalar U_s, PetscScalar m, PetscScalar n, PetscScalar e3)
{ PetscScalar result;
  result = pow(phi_s*cos(M_PI*m*x)*cos(M_PI*m*z) + 1.0, n);
  return(result);
}
static PetscScalar get_phi(PetscScalar x, PetscScalar z, PetscScalar delta, PetscScalar phi_0, PetscScalar phi_s, PetscScalar p_s,PetscScalar psi_s, PetscScalar U_s, PetscScalar m, PetscScalar n, PetscScalar e3)
{ PetscScalar result;
  result = phi_0*(phi_s*cos(M_PI*m*x)*cos(M_PI*m*z) + 1.0);
  return(result);
}
static PetscScalar get_psi(PetscScalar x, PetscScalar z, PetscScalar delta, PetscScalar phi_0, PetscScalar phi_s, PetscScalar p_s,PetscScalar psi_s, PetscScalar U_s, PetscScalar m, PetscScalar n, PetscScalar e3)
{ PetscScalar result;
  result = psi_s*(1.0 - cos(M_PI*m*x))*(1.0 - cos(M_PI*m*z));
  return(result);
}
static PetscScalar get_U(PetscScalar x, PetscScalar z, PetscScalar delta, PetscScalar phi_0, PetscScalar phi_s, PetscScalar p_s,PetscScalar psi_s, PetscScalar U_s, PetscScalar m, PetscScalar n, PetscScalar e3)
{ PetscScalar result;
  result = -U_s*cos(M_PI*m*x)*cos(M_PI*m*z);
  return(result);
}
static PetscScalar get_gradUx(PetscScalar x, PetscScalar z, PetscScalar delta, PetscScalar phi_0, PetscScalar phi_s, PetscScalar p_s,PetscScalar psi_s, PetscScalar U_s, PetscScalar m, PetscScalar n, PetscScalar e3)
{ PetscScalar result;
  result = M_PI*U_s*m*sin(M_PI*m*x)*cos(M_PI*m*z);
  return(result);
}
static PetscScalar get_gradUz(PetscScalar x, PetscScalar z, PetscScalar delta, PetscScalar phi_0, PetscScalar phi_s, PetscScalar p_s,PetscScalar psi_s, PetscScalar U_s, PetscScalar m, PetscScalar n, PetscScalar e3)
{ PetscScalar result;
  result = M_PI*U_s*m*sin(M_PI*m*z)*cos(M_PI*m*x);
  return(result);
}
static PetscScalar get_curl_psix(PetscScalar x, PetscScalar z, PetscScalar delta, PetscScalar phi_0, PetscScalar phi_s, PetscScalar p_s,PetscScalar psi_s, PetscScalar U_s, PetscScalar m, PetscScalar n, PetscScalar e3)
{ PetscScalar result;
  result = M_PI*m*psi_s*(1.0 - cos(M_PI*m*x))*sin(M_PI*m*z);
  return(result);
}
static PetscScalar get_curl_psiz(PetscScalar x, PetscScalar z, PetscScalar delta, PetscScalar phi_0, PetscScalar phi_s, PetscScalar p_s,PetscScalar psi_s, PetscScalar U_s, PetscScalar m, PetscScalar n, PetscScalar e3)
{ PetscScalar result;
  result = -M_PI*m*psi_s*(1.0 - cos(M_PI*m*z))*sin(M_PI*m*x);
  return(result);
}

const char coeff_description[] =
"  << Stokes-Darcy Coefficients >> \n"
"  A = delta^2 = eta0*K0/H^2 \n"
"  B = phi*e3 + fu (fux,fuz - manufactured)\n" 
"  C = fp (manufactured) \n"
"  D1 = delta^2 \n"
"  D2 = -Kphi \n"
"  D3 = -Kphi*e3 \n";

const char bc_description[] =
"  << Stokes-Darcy BCs >> \n"
"  LEFT, RIGHT, DOWN, UP: DIRICHLET (manufactured) \n";

// ---------------------------------------
// Application functions
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "StokesDarcy_Numerical"
PetscErrorCode StokesDarcy_Numerical(void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  FDPDE          fd;
  DM             dm, dmcoeff;
  Vec            x, xMMS, coeff;
  PetscInt       nx, nz;
  PetscScalar    xmin, zmin, xmax, zmax;
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
  ierr = FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_STOKESDARCY2FIELD,&fd);CHKERRQ(ierr);
  ierr = FDPDESetUp(fd);CHKERRQ(ierr);
  ierr = FDPDEGetDM(fd,&dm); CHKERRQ(ierr);

  // Create manufactured solution
  ierr = ComputeManufacturedSolution(dm,&xMMS,usr); CHKERRQ(ierr);

  // Calculate and output extra parameters: psi, U, curl_psi, gradU, phi
  ierr = ComputeExtraParameters(dm,usr); CHKERRQ(ierr);

  // Set BC evaluation function
  ierr = FDPDESetFunctionBCList(fd,FormBCList,bc_description,usr); CHKERRQ(ierr);

  // Set coefficients evaluation function
  ierr = FDPDESetFunctionCoefficient(fd,FormCoefficient,coeff_description,usr); CHKERRQ(ierr);
  ierr = FDPDEView(fd); CHKERRQ(ierr);

  // FD SNES Solver
  ierr = FDPDESolve(fd,NULL);CHKERRQ(ierr);
  ierr = FDPDEGetSolution(fd,&x);CHKERRQ(ierr); 

  // Output solution to file
  ierr = DMStagViewBinaryPython(dm,x,usr->par->fname_out);CHKERRQ(ierr);
  ierr = DMStagViewBinaryPython(dm,fd->r,"out_residual");CHKERRQ(ierr);

  // Output coefficient to file
  ierr = FDPDEGetCoefficient(fd,&dmcoeff,&coeff);CHKERRQ(ierr);
  ierr = DMStagViewBinaryPython(dmcoeff,coeff,"out_coefficients");CHKERRQ(ierr);

  // Compute norms
  ierr = ComputeErrorNorms(dm,x,xMMS);CHKERRQ(ierr);

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
  ierr = PetscBagRegisterScalar(bag, &par->e3, 0.0, "e3", "Direction of unit vertical vector"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->delta, 1.0, "delta", "Viscosity factor sigma^2 = eta0*K0/H^2"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->phi_0, 0.1, "phi_0", "Reference porosity"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->phi_s, 1.0, "phi_s", "Porosity amplitude"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->p_s, 1.0, "p_s", "Pressure amplitude"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->psi_s, 1.0, "psi_s", "Vector potential function amplitude"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->U_s, 1.0, "U_s", "Scalar potential function amplitude"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->m, 2.0, "m", "Trigonometric coefficient"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->n, 2.0, "n", "Porosity exponent"); CHKERRQ(ierr);

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
  PetscPrintf(usr->comm,"# Test_stokesdarcy2field_mms_katz_ch13: %s \n",&(date[0]));
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
// FormCoefficient
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormCoefficient"
PetscErrorCode FormCoefficient(FDPDE fd, DM dm, Vec x, DM dmcoeff, Vec coeff, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       i, j, sx, sz, nx, nz;
  Vec            coefflocal;
  PetscScalar    **coordx,**coordz;
  PetscInt       iprev, inext, icenter;
  PetscScalar    delta,phi_0,phi_s,p_s,psi_s,U_s,m,n,e3;
  PetscScalar    ***c;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  delta = usr->par->delta;
  phi_0 = usr->par->phi_0;
  phi_s = usr->par->phi_s;
  p_s = usr->par->p_s;
  psi_s = usr->par->psi_s;
  U_s = usr->par->U_s;
  m = usr->par->m;
  n = usr->par->n;
  e3 = usr->par->e3;

  // Get domain corners
  ierr = DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  // Get dm coordinates array
  ierr = DMStagGetProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmcoeff,LEFT,&iprev);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmcoeff,RIGHT,&inext);CHKERRQ(ierr); 
  ierr = DMStagGetProductCoordinateLocationSlot(dmcoeff,ELEMENT,&icenter);CHKERRQ(ierr);

  // Create coefficient local vector
  ierr = DMCreateLocalVector(dmcoeff, &coefflocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dmcoeff, coefflocal, &c); CHKERRQ(ierr);
  
  // Loop over local domain - set initial density and viscosity
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      PetscInt idx;

      { // A = delta^2 (center, c=1)
        DMStagStencil point;
        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 1;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = delta*delta;
      }

      { // A = delta^2 (corner, c=0)
        DMStagStencil point[4];
        PetscInt      ii;

        point[0].i = i; point[0].j = j; point[0].loc = DOWN_LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = DOWN_RIGHT; point[1].c = 0;
        point[2].i = i; point[2].j = j; point[2].loc = UP_LEFT;    point[2].c = 0;
        point[3].i = i; point[3].j = j; point[3].loc = UP_RIGHT;   point[3].c = 0;

        for (ii = 0; ii < 4; ii++) {
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = delta*delta;
        }
      }

      { // B = phi*e3 + fu (fux,fuz - manufactured) (edges, c=0)
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
        rhs[0] = get_fux(xp[0],zp[0],delta,phi_0,phi_s,p_s,psi_s,U_s,m,n,e3);
        rhs[1] = get_fux(xp[1],zp[1],delta,phi_0,phi_s,p_s,psi_s,U_s,m,n,e3);

        // Bz = phi*e3 + fuz
        rhs[2] = get_fuz(xp[2],zp[2],delta,phi_0,phi_s,p_s,psi_s,U_s,m,n,e3);
        rhs[3] = get_fuz(xp[3],zp[3],delta,phi_0,phi_s,p_s,psi_s,U_s,m,n,e3);

        rhs[2] += e3*get_phi(xp[2],zp[2],delta,phi_0,phi_s,p_s,psi_s,U_s,m,n,e3);
        rhs[3] += e3*get_phi(xp[3],zp[3],delta,phi_0,phi_s,p_s,psi_s,U_s,m,n,e3);

        for (ii = 0; ii < 4; ii++) {
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = rhs[ii];
        }
      }

      { // C = (manufactured) (center, c=0)
        DMStagStencil point;
        PetscScalar   xp, zp, rhs;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 0;
        xp = coordx[i][icenter]; zp = coordz[j][icenter];
        rhs = get_fp(xp,zp,delta,phi_0,phi_s,p_s,psi_s,U_s,m,n,e3);

        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = rhs;
      }

      { // D1 = delta^2 (center, c=2)
        DMStagStencil point;
        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 2;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = delta*delta;
      }

      { // D2 = -Kphi (edges, c=1)
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
          c[j][i][idx] = -get_Kphi(xp[ii],zp[ii],delta,phi_0,phi_s,p_s,psi_s,U_s,m,n,e3);
        }
      }

      { // D3 = -Kphi*e3 (edges, c=2)
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
          c[j][i][idx] = -e3*get_Kphi(xp[ii],zp[ii],delta,phi_0,phi_s,p_s,psi_s,U_s,m,n,e3);
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
// FormBCList
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormBCList"
PetscErrorCode FormBCList(DM dm, Vec x, DMStagBCList bclist, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       k,n_bc,*idx_bc;
  PetscScalar    *value_bc,*x_bc;
  PetscScalar    delta,phi_0,phi_s,p_s,psi_s,U_s,m,n,e3;
  BCType         *type_bc;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;

  delta = usr->par->delta;
  phi_0 = usr->par->phi_0;
  phi_s = usr->par->phi_s;
  p_s = usr->par->p_s;
  psi_s = usr->par->psi_s;
  U_s = usr->par->U_s;
  m = usr->par->m;
  n = usr->par->n;
  e3 = usr->par->e3;

  // LEFT Boundary - Vx
  ierr = DMStagBCListGetValues(bclist,'w','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_ux(x_bc[2*k],x_bc[2*k+1],delta,phi_0,phi_s,p_s,psi_s,U_s,m,n,e3);
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // LEFT Boundary - Vz
  ierr = DMStagBCListGetValues(bclist,'w','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=1; k<n_bc-1; k++) {
    value_bc[k] = get_uz(x_bc[2*k],x_bc[2*k+1],delta,phi_0,phi_s,p_s,psi_s,U_s,m,n,e3);
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // LEFT Boundary - P
  ierr = DMStagBCListGetValues(bclist,'w','o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_p(x_bc[2*k],x_bc[2*k+1],delta,phi_0,phi_s,p_s,psi_s,U_s,m,n,e3);
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  
  // RIGHT Boundary - Vx
  ierr = DMStagBCListGetValues(bclist,'e','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_ux(x_bc[2*k],x_bc[2*k+1],delta,phi_0,phi_s,p_s,psi_s,U_s,m,n,e3);
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // RIGHT Boundary - Vz
  ierr = DMStagBCListGetValues(bclist,'e','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=1; k<n_bc-1; k++) {
    value_bc[k] = get_uz(x_bc[2*k],x_bc[2*k+1],delta,phi_0,phi_s,p_s,psi_s,U_s,m,n,e3);
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // RIGHT Boundary - P
  ierr = DMStagBCListGetValues(bclist,'e','o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_p(x_bc[2*k],x_bc[2*k+1],delta,phi_0,phi_s,p_s,psi_s,U_s,m,n,e3);
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN Boundary - Vx
  ierr = DMStagBCListGetValues(bclist,'s','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=1; k<n_bc-1; k++) {
    value_bc[k] = get_ux(x_bc[2*k],x_bc[2*k+1],delta,phi_0,phi_s,p_s,psi_s,U_s,m,n,e3);
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN Boundary - Vz
  ierr = DMStagBCListGetValues(bclist,'s','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_uz(x_bc[2*k],x_bc[2*k+1],delta,phi_0,phi_s,p_s,psi_s,U_s,m,n,e3);
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN Boundary - P
  ierr = DMStagBCListGetValues(bclist,'s','o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_p(x_bc[2*k],x_bc[2*k+1],delta,phi_0,phi_s,p_s,psi_s,U_s,m,n,e3);
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP Boundary - Vx
  ierr = DMStagBCListGetValues(bclist,'n','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=1; k<n_bc-1; k++) {
    value_bc[k] = get_ux(x_bc[2*k],x_bc[2*k+1],delta,phi_0,phi_s,p_s,psi_s,U_s,m,n,e3);
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP Boundary - Vz
  ierr = DMStagBCListGetValues(bclist,'n','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_uz(x_bc[2*k],x_bc[2*k+1],delta,phi_0,phi_s,p_s,psi_s,U_s,m,n,e3);
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP Boundary - P
  ierr = DMStagBCListGetValues(bclist,'n','o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_p(x_bc[2*k],x_bc[2*k+1],delta,phi_0,phi_s,p_s,psi_s,U_s,m,n,e3);
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

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
  PetscScalar    delta,phi_0,phi_s,p_s,psi_s,U_s,m,n,e3;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  delta = usr->par->delta;
  phi_0 = usr->par->phi_0;
  phi_s = usr->par->phi_s;
  p_s = usr->par->p_s;
  psi_s = usr->par->psi_s;
  U_s = usr->par->U_s;
  m = usr->par->m;
  n = usr->par->n;
  e3 = usr->par->e3;

  // Create local and global vectors for MMS solution
  ierr = DMCreateGlobalVector(dm,&xMMS     ); CHKERRQ(ierr);
  ierr = DMCreateLocalVector (dm,&xMMSlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm,xMMSlocal,&xxMMS); CHKERRQ(ierr);

  // Get domain corners
  ierr = DMStagGetGlobalSizes(dm, &Nx, &Nz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  
// Get dm coordinates array
  ierr = DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter);CHKERRQ(ierr); 
  ierr = DMStagGetProductCoordinateLocationSlot(dm,LEFT,&iprev);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,RIGHT,&inext);CHKERRQ(ierr); 

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {

      // pressure
      ierr = DMStagGetLocationSlot(dm,ELEMENT,0,&idx); CHKERRQ(ierr);
      xxMMS[j][i][idx] = get_p(coordx[i][icenter],coordz[j][icenter],delta,phi_0,phi_s,p_s,psi_s,U_s,m,n,e3);

      // ux
      ierr = DMStagGetLocationSlot(dm,LEFT,0,&idx); CHKERRQ(ierr);
      xxMMS[j][i][idx] = get_ux(coordx[i][iprev],coordz[j][icenter],delta,phi_0,phi_s,p_s,psi_s,U_s,m,n,e3);

      if (i == Nx-1) {
        ierr = DMStagGetLocationSlot(dm,RIGHT,0,&idx); CHKERRQ(ierr);
        xxMMS[j][i][idx] = get_ux(coordx[i][inext],coordz[j][icenter],delta,phi_0,phi_s,p_s,psi_s,U_s,m,n,e3);
      }
      
      // uz
      ierr = DMStagGetLocationSlot(dm,DOWN,0,&idx); CHKERRQ(ierr);
      xxMMS[j][i][idx] = get_uz(coordx[i][icenter],coordz[j][iprev],delta,phi_0,phi_s,p_s,psi_s,U_s,m,n,e3);

      if (j == Nz-1) {
        ierr = DMStagGetLocationSlot(dm,UP,0,&idx); CHKERRQ(ierr);
        xxMMS[j][i][idx] = get_uz(coordx[i][icenter],coordz[j][inext],delta,phi_0,phi_s,p_s,psi_s,U_s,m,n,e3);
      }
    }
  }

  // Restore arrays
  ierr = DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);

  // Restore and map local to global
  ierr = DMStagVecRestoreArray(dm,xMMSlocal,&xxMMS); CHKERRQ(ierr);
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
PetscErrorCode ComputeErrorNorms(DM dm,Vec x,Vec xMMS)
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
  PetscPrintf(comm,"# Velocity: norm2 = %1.12e norm2x = %1.12e norm2z = %1.12e \n",nrm2v,nrm2vx,nrm2vz);
  PetscPrintf(comm,"# Pressure: norm2 = %1.12e\n",nrm2p);
  PetscPrintf(comm,"# Grid info: hx = %1.12e hz = %1.12e\n",dx,dz);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// Calculate extra parameters: psi, U, curl_psi, gradU, phi
// ---------------------------------------
PetscErrorCode ComputeExtraParameters(DM dm, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz, idx;
  PetscInt       iprev, inext, icenter;
  PetscScalar    ***xx;
  PetscScalar    **coordx,**coordz;
  DM             dmextra;
  Vec            x, xlocal;
  PetscScalar    delta,phi_0,phi_s,p_s,psi_s,U_s,m,n,e3;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  delta = usr->par->delta;
  phi_0 = usr->par->phi_0;
  phi_s = usr->par->phi_s;
  p_s = usr->par->p_s;
  psi_s = usr->par->psi_s;
  U_s = usr->par->U_s;
  m = usr->par->m;
  n = usr->par->n;
  e3 = usr->par->e3;

  // // Create new dmextra (3 element, 2 edge dofs)
  // ierr = DMStagCreateCompatibleDMStag(dm,0,2,3,0,&dmextra); CHKERRQ(ierr);

  // Create new dmextra (7 element, with 3 scalar and 2 velocity fiels)
  ierr = DMStagCreateCompatibleDMStag(dm,0,0,7,0,&dmextra); CHKERRQ(ierr);
  ierr = DMSetUp(dmextra); CHKERRQ(ierr);
  ierr = DMStagSetUniformCoordinatesProduct(dmextra,usr->par->xmin,usr->par->xmin+usr->par->L,usr->par->zmin,usr->par->zmin+usr->par->H,0.0,0.0);CHKERRQ(ierr);

  // Create local and global vectors
  ierr = DMCreateGlobalVector(dmextra,&x); CHKERRQ(ierr);
  ierr = DMCreateLocalVector (dmextra,&xlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dmextra,xlocal,&xx); CHKERRQ(ierr);

  // Get domain corners
  ierr = DMStagGetGlobalSizes(dmextra, &Nx, &Nz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dmextra, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  
// Get dm coordinates array
  ierr = DMStagGetProductCoordinateArraysRead(dmextra,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmextra,ELEMENT,&icenter);CHKERRQ(ierr); 
  ierr = DMStagGetProductCoordinateLocationSlot(dmextra,LEFT,&iprev);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmextra,RIGHT,&inext);CHKERRQ(ierr); 

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {

      // psi - element 0
      ierr = DMStagGetLocationSlot(dmextra,ELEMENT,0,&idx); CHKERRQ(ierr);
      xx[j][i][idx] = get_psi(coordx[i][icenter],coordz[j][icenter],delta,phi_0,phi_s,p_s,psi_s,U_s,m,n,e3);

      // U - element 1
      ierr = DMStagGetLocationSlot(dmextra,ELEMENT,1,&idx); CHKERRQ(ierr);
      xx[j][i][idx] = get_U(coordx[i][icenter],coordz[j][icenter],delta,phi_0,phi_s,p_s,psi_s,U_s,m,n,e3);

      // phi - element 2
      ierr = DMStagGetLocationSlot(dmextra,ELEMENT,2,&idx); CHKERRQ(ierr);
      xx[j][i][idx] = get_phi(coordx[i][icenter],coordz[j][icenter],delta,phi_0,phi_s,p_s,psi_s,U_s,m,n,e3);

      // curl_psix - element 3
      ierr = DMStagGetLocationSlot(dmextra,ELEMENT,3,&idx); CHKERRQ(ierr);
      xx[j][i][idx] = get_curl_psix(coordx[i][icenter],coordz[j][icenter],delta,phi_0,phi_s,p_s,psi_s,U_s,m,n,e3);

      // curl_psiz - element 4
      ierr = DMStagGetLocationSlot(dmextra,ELEMENT,4,&idx); CHKERRQ(ierr);
      xx[j][i][idx] = get_curl_psiz(coordx[i][icenter],coordz[j][icenter],delta,phi_0,phi_s,p_s,psi_s,U_s,m,n,e3);

      // gradUx - element 3
      ierr = DMStagGetLocationSlot(dmextra,ELEMENT,5,&idx); CHKERRQ(ierr);
      xx[j][i][idx] = get_gradUx(coordx[i][icenter],coordz[j][icenter],delta,phi_0,phi_s,p_s,psi_s,U_s,m,n,e3);

      // curl_gradUz - element 4
      ierr = DMStagGetLocationSlot(dmextra,ELEMENT,6,&idx); CHKERRQ(ierr);
      xx[j][i][idx] = get_gradUz(coordx[i][icenter],coordz[j][icenter],delta,phi_0,phi_s,p_s,psi_s,U_s,m,n,e3);

      // // curl_psix - edge 0, left/right
      // ierr = DMStagGetLocationSlot(dmextra,LEFT,0,&idx); CHKERRQ(ierr);
      // xx[j][i][idx] = get_curl_psix(coordx[i][iprev],coordz[j][icenter],delta,phi_0,phi_s,p_s,psi_s,U_s,m,n,e3);

      // if (i == Nx-1) {
      //   ierr = DMStagGetLocationSlot(dmextra,RIGHT,0,&idx); CHKERRQ(ierr);
      //   xx[j][i][idx] = get_curl_psix(coordx[i][inext],coordz[j][icenter],delta,phi_0,phi_s,p_s,psi_s,U_s,m,n,e3);
      // }
      
      // // curl_psiz - edge 0, down/up
      // ierr = DMStagGetLocationSlot(dmextra,DOWN,0,&idx); CHKERRQ(ierr);
      // xx[j][i][idx] = get_curl_psiz(coordx[i][icenter],coordz[j][iprev],delta,phi_0,phi_s,p_s,psi_s,U_s,m,n,e3);

      // if (j == Nz-1) {
      //   ierr = DMStagGetLocationSlot(dmextra,UP,0,&idx); CHKERRQ(ierr);
      //   xx[j][i][idx] = get_curl_psiz(coordx[i][icenter],coordz[j][inext],delta,phi_0,phi_s,p_s,psi_s,U_s,m,n,e3);
      // }

      // // gradUx - edge 1, left/right
      // ierr = DMStagGetLocationSlot(dmextra,LEFT,1,&idx); CHKERRQ(ierr);
      // xx[j][i][idx] = get_gradUx(coordx[i][iprev],coordz[j][icenter],delta,phi_0,phi_s,p_s,psi_s,U_s,m,n,e3);

      // if (i == Nx-1) {
      //   ierr = DMStagGetLocationSlot(dmextra,RIGHT,1,&idx); CHKERRQ(ierr);
      //   xx[j][i][idx] = get_gradUx(coordx[i][inext],coordz[j][icenter],delta,phi_0,phi_s,p_s,psi_s,U_s,m,n,e3);
      // }
      
      // // gradUz - edge 1, down/up
      // ierr = DMStagGetLocationSlot(dmextra,DOWN,1,&idx); CHKERRQ(ierr);
      // xx[j][i][idx] = get_gradUz(coordx[i][icenter],coordz[j][iprev],delta,phi_0,phi_s,p_s,psi_s,U_s,m,n,e3);

      // if (j == Nz-1) {
      //   ierr = DMStagGetLocationSlot(dmextra,UP,1,&idx); CHKERRQ(ierr);
      //   xx[j][i][idx] = get_gradUz(coordx[i][icenter],coordz[j][inext],delta,phi_0,phi_s,p_s,psi_s,U_s,m,n,e3);
      // }
    }
  }

  // Restore arrays
  ierr = DMStagRestoreProductCoordinateArraysRead(dmextra,&coordx,&coordz,NULL);CHKERRQ(ierr);

  // Restore and map local to global
  ierr = DMStagVecRestoreArray(dmextra,xlocal,&xx); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dmextra,xlocal,INSERT_VALUES,x); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dmextra,xlocal,INSERT_VALUES,x); CHKERRQ(ierr);
  ierr = DMStagViewBinaryPython(dmextra,x,"out_extra_parameters");CHKERRQ(ierr);

  // clean 
  ierr = VecDestroy(&xlocal); CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = DMDestroy(&dmextra);CHKERRQ(ierr);
  
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

  // Start time
  ierr = PetscTime(&start_time); CHKERRQ(ierr);
 
  // Load command line or input file if required
  ierr = PetscOptionsInsert(PETSC_NULL,&argc,&argv,NULL); CHKERRQ(ierr);

  // Input user parameters and print
  ierr = InputParameters(&usr); CHKERRQ(ierr);

  // Print user parameters
  ierr = InputPrintData(usr); CHKERRQ(ierr);

  // Numerical solution using the FD pde object
  ierr = StokesDarcy_Numerical(usr); CHKERRQ(ierr);

  // Destroy objects
  ierr = PetscBagDestroy(&usr->bag); CHKERRQ(ierr);
  ierr = PetscFree(usr);             CHKERRQ(ierr);

  // End time
  ierr = PetscTime(&end_time); CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"# Total runtime: %g (sec) \n", end_time - start_time);
  PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");
  
  // Finalize main
  ierr = PetscFinalize();
  return ierr;
}
