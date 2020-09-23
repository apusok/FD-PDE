// ---------------------------------------
// Rhebergen et al. 2014, SIAM - Ex. 6.1
// run: ./tests/test_stokesdarcy2field_rhebergen-siam-2014.app -pc_type lu -pc_factor_mat_solver_type umfpack -nx 10 -nz 10
// python test: ./tests/python/test_stokesdarcy2field_mms_rhebergen_siam_2014.py
// python sympy: ./mms/mms_rhebergen_2014_siam.py
// ---------------------------------------
static char help[] = "Application to solve the rhebergen-siam-2014 benchmark with FD-PDE \n\n";

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
  PetscScalar    alpha, k_ls, k_us, e3, A;
  char           fname_out[FNAME_LENGTH]; 
  char           fname_in [FNAME_LENGTH];  
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
PetscErrorCode ComputeErrorNorms(DM,Vec,Vec);
PetscErrorCode InputParameters(UsrData**);
PetscErrorCode InputPrintData(UsrData*);
PetscErrorCode FormCoefficient(FDPDE, DM, Vec, DM, Vec, void*);
PetscErrorCode FormBCList(DM, Vec, DMStagBCList, void*);

// Solution for k = 1.0
// static PetscScalar get_k(PetscScalar x, PetscScalar z, PetscScalar k_ls, PetscScalar k_us, PetscScalar alpha, PetscScalar A)
// { PetscScalar result;
//   result = 1.0;
//   return(result);
// }
// static PetscScalar get_p(PetscScalar x, PetscScalar z, PetscScalar k_ls, PetscScalar k_us, PetscScalar alpha, PetscScalar A)
// { PetscScalar result;
//   result = -cos(4.0*M_PI*x)*cos(2.0*M_PI*z);
//   return(result);
// }
// static PetscScalar get_ux(PetscScalar x, PetscScalar z, PetscScalar k_ls, PetscScalar k_us, PetscScalar alpha, PetscScalar A)
// { PetscScalar result;
//   result = sin(M_PI*x)*sin(2.0*M_PI*z) + 4.0*M_PI*sin(4.0*M_PI*x)*cos(2.0*M_PI*z) + 2.0;
//   return(result);
// }
// static PetscScalar get_uz(PetscScalar x, PetscScalar z, PetscScalar k_ls, PetscScalar k_us, PetscScalar alpha, PetscScalar A)
// { PetscScalar result;
//   result = 2.0*M_PI*sin(2.0*M_PI*z)*cos(4.0*M_PI*x) + 0.5*cos(M_PI*x)*cos(2.0*M_PI*z) + 2.0;
//   return(result);
// }
// static PetscScalar get_fux(PetscScalar x, PetscScalar z, PetscScalar k_ls, PetscScalar k_us, PetscScalar alpha, PetscScalar A)
// { PetscScalar result;
//   result = A*(-5.0*pow(M_PI, 2)*sin(M_PI*x)*sin(2.0*M_PI*z) - 160.0*pow(M_PI, 3)*sin(4.0*M_PI*x)*cos(2.0*M_PI*z)) - 80.0*pow(M_PI, 3)*alpha*sin(4.0*M_PI*x)*cos(2.0*M_PI*z) - 4.0*M_PI*sin(4.0*M_PI*x)*cos(2.0*M_PI*z);
//   return(result);
// }
// static PetscScalar get_fuz(PetscScalar x, PetscScalar z, PetscScalar k_ls, PetscScalar k_us, PetscScalar alpha, PetscScalar A)
// { PetscScalar result;
//   result = A*(-80.0*pow(M_PI, 3)*sin(2.0*M_PI*z)*cos(4.0*M_PI*x) - 2.5*pow(M_PI, 2)*cos(M_PI*x)*cos(2.0*M_PI*z)) - 40.0*pow(M_PI, 3)*alpha*sin(2.0*M_PI*z)*cos(4.0*M_PI*x) - 2.0*M_PI*sin(2.0*M_PI*z)*cos(4.0*M_PI*x);
//   return(result);
// }
// static PetscScalar get_fp(PetscScalar x, PetscScalar z, PetscScalar k_ls, PetscScalar k_us, PetscScalar alpha, PetscScalar A)
// { PetscScalar result;
//   result = 0;
//   return(result);
// }

static PetscScalar get_k(PetscScalar x, PetscScalar z, PetscScalar k_ls, PetscScalar k_us, PetscScalar alpha, PetscScalar A)
{ PetscScalar result;
  result = (-0.25002270099550483*k_ls + 0.25002270099550483*k_us)*((-3.99981840852519*k_ls + 0.00018159147480978355*k_us)/(k_ls - k_us) + tanh(10.0*x - 5.0) + tanh(10.0*z - 5.0) + 2.0);
  return(result);
}
static PetscScalar get_p(PetscScalar x, PetscScalar z, PetscScalar k_ls, PetscScalar k_us, PetscScalar alpha, PetscScalar A)
{ PetscScalar result;
  result = -cos(4.0*M_PI*x)*cos(2.0*M_PI*z);
  return(result);
}
static PetscScalar get_ux(PetscScalar x, PetscScalar z, PetscScalar k_ls, PetscScalar k_us, PetscScalar alpha, PetscScalar A)
{ PetscScalar result;
  result = 4.0*M_PI*(-0.25002270099550483*k_ls + 0.25002270099550483*k_us)*((-3.99981840852519*k_ls + 0.00018159147480978355*k_us)/(k_ls - k_us) + tanh(10.0*x - 5.0) + tanh(10.0*z - 5.0) + 2.0)*sin(4.0*M_PI*x)*cos(2.0*M_PI*z) + sin(M_PI*x)*sin(2.0*M_PI*z) + 2.0;
  return(result);
}
static PetscScalar get_uz(PetscScalar x, PetscScalar z, PetscScalar k_ls, PetscScalar k_us, PetscScalar alpha, PetscScalar A)
{ PetscScalar result;
  result = 2.0*M_PI*(-0.25002270099550483*k_ls + 0.25002270099550483*k_us)*((-3.99981840852519*k_ls + 0.00018159147480978355*k_us)/(k_ls - k_us) + tanh(10.0*x - 5.0) + tanh(10.0*z - 5.0) + 2.0)*sin(2.0*M_PI*z)*cos(4.0*M_PI*x) + 0.5*cos(M_PI*x)*cos(2.0*M_PI*z) + 2.0;
  return(result);
}
static PetscScalar get_fux(PetscScalar x, PetscScalar z, PetscScalar k_ls, PetscScalar k_us, PetscScalar alpha, PetscScalar A)
{ PetscScalar result;
  result = A*(68.0*pow(M_PI, 2)*(10.0 - 10.0*pow(tanh(10.0*x - 5.0), 2))*(-0.25002270099550483*k_ls + 0.25002270099550483*k_us)*cos(4.0*M_PI*x)*cos(2.0*M_PI*z) - 24.0*pow(M_PI, 2)*(10.0 - 10.0*pow(tanh(10.0*z - 5.0), 2))*(-0.25002270099550483*k_ls + 0.25002270099550483*k_us)*sin(4.0*M_PI*x)*sin(2.0*M_PI*z) - 80.0*M_PI*(20.0 - 20.0*pow(tanh(10.0*x - 5.0), 2))*(-0.25002270099550483*k_ls + 0.25002270099550483*k_us)*sin(4.0*M_PI*x)*cos(2.0*M_PI*z)*tanh(10.0*x - 5.0) - 40.0*M_PI*(20.0 - 20.0*pow(tanh(10.0*z - 5.0), 2))*(-0.25002270099550483*k_ls + 0.25002270099550483*k_us)*sin(4.0*M_PI*x)*cos(2.0*M_PI*z)*tanh(10.0*z - 5.0) - 160.0*pow(M_PI, 3)*(-0.25002270099550483*k_ls + 0.25002270099550483*k_us)*((-3.99981840852519*k_ls + 0.00018159147480978355*k_us)/(k_ls - k_us) + tanh(10.0*x - 5.0) + tanh(10.0*z - 5.0) + 2.0)*sin(4.0*M_PI*x)*cos(2.0*M_PI*z) - 5.0*pow(M_PI, 2)*sin(M_PI*x)*sin(2.0*M_PI*z)) + alpha*(36.0*pow(M_PI, 2)*(10.0 - 10.0*pow(tanh(10.0*x - 5.0), 2))*(-0.25002270099550483*k_ls + 0.25002270099550483*k_us)*cos(4.0*M_PI*x)*cos(2.0*M_PI*z) - 8.0*pow(M_PI, 2)*(10.0 - 10.0*pow(tanh(10.0*z - 5.0), 2))*(-0.25002270099550483*k_ls + 0.25002270099550483*k_us)*sin(4.0*M_PI*x)*sin(2.0*M_PI*z) - 40.0*M_PI*(20.0 - 20.0*pow(tanh(10.0*x - 5.0), 2))*(-0.25002270099550483*k_ls + 0.25002270099550483*k_us)*sin(4.0*M_PI*x)*cos(2.0*M_PI*z)*tanh(10.0*x - 5.0) - 80.0*pow(M_PI, 3)*(-0.25002270099550483*k_ls + 0.25002270099550483*k_us)*((-3.99981840852519*k_ls + 0.00018159147480978355*k_us)/(k_ls - k_us) + tanh(10.0*x - 5.0) + tanh(10.0*z - 5.0) + 2.0)*sin(4.0*M_PI*x)*cos(2.0*M_PI*z)) - 4.0*M_PI*sin(4.0*M_PI*x)*cos(2.0*M_PI*z);
  return(result);
}
static PetscScalar get_fuz(PetscScalar x, PetscScalar z, PetscScalar k_ls, PetscScalar k_us, PetscScalar alpha, PetscScalar A)
{ PetscScalar result;
  result = A*(-24.0*pow(M_PI, 2)*(10.0 - 10.0*pow(tanh(10.0*x - 5.0), 2))*(-0.25002270099550483*k_ls + 0.25002270099550483*k_us)*sin(4.0*M_PI*x)*sin(2.0*M_PI*z) + 32.0*pow(M_PI, 2)*(10.0 - 10.0*pow(tanh(10.0*z - 5.0), 2))*(-0.25002270099550483*k_ls + 0.25002270099550483*k_us)*cos(4.0*M_PI*x)*cos(2.0*M_PI*z) - 20.0*M_PI*(20.0 - 20.0*pow(tanh(10.0*x - 5.0), 2))*(-0.25002270099550483*k_ls + 0.25002270099550483*k_us)*sin(2.0*M_PI*z)*cos(4.0*M_PI*x)*tanh(10.0*x - 5.0) - 40.0*M_PI*(20.0 - 20.0*pow(tanh(10.0*z - 5.0), 2))*(-0.25002270099550483*k_ls + 0.25002270099550483*k_us)*sin(2.0*M_PI*z)*cos(4.0*M_PI*x)*tanh(10.0*z - 5.0) - 80.0*pow(M_PI, 3)*(-0.25002270099550483*k_ls + 0.25002270099550483*k_us)*((-3.99981840852519*k_ls + 0.00018159147480978355*k_us)/(k_ls - k_us) + tanh(10.0*x - 5.0) + tanh(10.0*z - 5.0) + 2.0)*sin(2.0*M_PI*z)*cos(4.0*M_PI*x) - 2.5*pow(M_PI, 2)*cos(M_PI*x)*cos(2.0*M_PI*z)) + alpha*(-8.0*pow(M_PI, 2)*(10.0 - 10.0*pow(tanh(10.0*x - 5.0), 2))*(-0.25002270099550483*k_ls + 0.25002270099550483*k_us)*sin(4.0*M_PI*x)*sin(2.0*M_PI*z) + 24.0*pow(M_PI, 2)*(10.0 - 10.0*pow(tanh(10.0*z - 5.0), 2))*(-0.25002270099550483*k_ls + 0.25002270099550483*k_us)*cos(4.0*M_PI*x)*cos(2.0*M_PI*z) - 20.0*M_PI*(20.0 - 20.0*pow(tanh(10.0*z - 5.0), 2))*(-0.25002270099550483*k_ls + 0.25002270099550483*k_us)*sin(2.0*M_PI*z)*cos(4.0*M_PI*x)*tanh(10.0*z - 5.0) - 40.0*pow(M_PI, 3)*(-0.25002270099550483*k_ls + 0.25002270099550483*k_us)*((-3.99981840852519*k_ls + 0.00018159147480978355*k_us)/(k_ls - k_us) + tanh(10.0*x - 5.0) + tanh(10.0*z - 5.0) + 2.0)*sin(2.0*M_PI*z)*cos(4.0*M_PI*x)) - 2.0*M_PI*sin(2.0*M_PI*z)*cos(4.0*M_PI*x);
  return(result);
}
static PetscScalar get_fp(PetscScalar x, PetscScalar z, PetscScalar k_ls, PetscScalar k_us, PetscScalar alpha, PetscScalar A)
{ PetscScalar result;
  result = 0;
  return(result);
}

// ---------------------------------------
// Some descriptions - for the nondimensional equations 2.17
// Eq. 2.17: A = 0.5, B = manufactured , C = manufactured, D1 = alpha, D2 = -k, D3 = 0.0
// alpha and k are explained in the paper
// ---------------------------------------
const char coeff_description[] =
"  << Stokes-Darcy Coefficients >> \n"
"  A = 0.5 \n"
"  B = (manufactured)\n" 
"  C = (manufactured) \n"
"  D1 = alpha \n"
"  D2 = -k \n"
"  D3 = k*e3 = 0.0 \n";

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
  Vec            x, xMMS, coeff;//, xguess;
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

  // Set BC evaluation function
  ierr = FDPDESetFunctionBCList(fd,FormBCList,bc_description,usr); CHKERRQ(ierr);

  // Set coefficients evaluation function
  ierr = FDPDESetFunctionCoefficient(fd,FormCoefficient,coeff_description,usr); CHKERRQ(ierr);
  ierr = FDPDEView(fd); CHKERRQ(ierr);

  // Initialize guess
  // ierr = FDPDEGetSolutionGuess(fd,&xguess);CHKERRQ(ierr);
  // ierr = VecCopy(xMMS,xguess);CHKERRQ(ierr);
  // ierr = VecSet(xguess,1.0);CHKERRQ(ierr);
  // ierr = VecDestroy(&xguess);CHKERRQ(ierr);

  // FD SNES Solver
  ierr = FDPDESolve(fd,NULL);CHKERRQ(ierr);
  ierr = FDPDEGetSolution(fd,&x);CHKERRQ(ierr); 

  // ierr = FDPDEGetSolution(fd,&x);CHKERRQ(ierr); 
  // ierr = VecCopy(xMMS,x);CHKERRQ(ierr);
  // ierr = SNESComputeFunction(fd->snes,x,fd->r);CHKERRQ(ierr); 

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
  ierr = PetscBagRegisterScalar(bag, &par->alpha, 1000.0, "alpha", "Shear/compaction viscosity term"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->k_ls, 0.5, "k_ls", "k lower bound star"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->k_us, 1.5, "k_us", "k upper bound star"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->A, 0.5, "A", "viscosity coefficient"); CHKERRQ(ierr);

  // Input/output 
  ierr = PetscBagRegisterString(bag,&par->fname_out,FNAME_LENGTH,"out_solution","output_file","Name for output file, set with: -output_file <filename>"); CHKERRQ(ierr);

  // Other variables
  par->fname_in[0] = '\0';

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
  PetscPrintf(usr->comm,"# Test_stokesdarcy2field_rhebergen-siam2014: %s \n",&(date[0]));
  PetscPrintf(usr->comm,"# --------------------------------------- #\n");
  PetscPrintf(usr->comm,"# PETSc options: %s \n",opts);
  PetscPrintf(usr->comm,"# --------------------------------------- #\n");

  // Input file info
  if (usr->par->fname_in[0] == '\0') { // string is empty
    PetscPrintf(usr->comm,"# Input options file: NONE \n");
  }
  else {
    PetscPrintf(usr->comm,"# Input options file: %s \n",usr->par->fname_in);
  }
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
  PetscScalar    ***c;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

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

      { // A = 0.5 (center, c=1)
        DMStagStencil point;
        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 1;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = usr->par->A;
      }

      { // A = 0.5 (corner, c=0)
        DMStagStencil point[4];
        PetscInt      ii;

        point[0].i = i; point[0].j = j; point[0].loc = DOWN_LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = DOWN_RIGHT; point[1].c = 0;
        point[2].i = i; point[2].j = j; point[2].loc = UP_LEFT;    point[2].c = 0;
        point[3].i = i; point[3].j = j; point[3].loc = UP_RIGHT;   point[3].c = 0;

        for (ii = 0; ii < 4; ii++) {
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = usr->par->A;
        }
      }

      { // B = (manufactured) (edges, c=0)
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

        rhs[0] = get_fux(xp[0],zp[0],usr->par->k_ls,usr->par->k_us,usr->par->alpha,usr->par->A);
        rhs[1] = get_fux(xp[1],zp[1],usr->par->k_ls,usr->par->k_us,usr->par->alpha,usr->par->A);
        rhs[2] = get_fuz(xp[2],zp[2],usr->par->k_ls,usr->par->k_us,usr->par->alpha,usr->par->A);
        rhs[3] = get_fuz(xp[3],zp[3],usr->par->k_ls,usr->par->k_us,usr->par->alpha,usr->par->A);

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
        rhs = get_fp(xp,zp,usr->par->k_ls,usr->par->k_us,usr->par->alpha,usr->par->A);

        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = rhs;
      }

      { // D1 = alpha (center, c=2)
        DMStagStencil point;
        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 2;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = usr->par->alpha;
      }

      { // D2 = -k (edges, c=1)
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
          c[j][i][idx] = -get_k(xp[ii],zp[ii],usr->par->k_ls,usr->par->k_us,usr->par->alpha,usr->par->A);
        }
      }

      { // D3 = k*e3 (edges, c=2)
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
          c[j][i][idx] = get_k(xp[ii],zp[ii],usr->par->k_ls,usr->par->k_us,usr->par->alpha,usr->par->A)*usr->par->e3;
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
  BCType         *type_bc;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;

  // LEFT Boundary - Vx
  ierr = DMStagBCListGetValues(bclist,'w','-',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_ux(x_bc[2*k],x_bc[2*k+1],usr->par->k_ls,usr->par->k_us,usr->par->alpha,usr->par->A);
    type_bc[k] = BC_DIRICHLET_TRUE;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // LEFT Boundary - Vz
  ierr = DMStagBCListGetValues(bclist,'w','|',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_uz(x_bc[2*k],x_bc[2*k+1],usr->par->k_ls,usr->par->k_us,usr->par->alpha,usr->par->A);
    type_bc[k] = BC_DIRICHLET_TRUE;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // LEFT Boundary - P
  ierr = DMStagBCListGetValues(bclist,'w','o',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_p(x_bc[2*k],x_bc[2*k+1],usr->par->k_ls,usr->par->k_us,usr->par->alpha,usr->par->A);
    type_bc[k] = BC_DIRICHLET_TRUE;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  
  // RIGHT Boundary - Vx
  ierr = DMStagBCListGetValues(bclist,'e','-',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_ux(x_bc[2*k],x_bc[2*k+1],usr->par->k_ls,usr->par->k_us,usr->par->alpha,usr->par->A);
    type_bc[k] = BC_DIRICHLET_TRUE;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // RIGHT Boundary - Vz
  ierr = DMStagBCListGetValues(bclist,'e','|',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_uz(x_bc[2*k],x_bc[2*k+1],usr->par->k_ls,usr->par->k_us,usr->par->alpha,usr->par->A);
    type_bc[k] = BC_DIRICHLET_TRUE;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // RIGHT Boundary - P
  ierr = DMStagBCListGetValues(bclist,'e','o',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_p(x_bc[2*k],x_bc[2*k+1],usr->par->k_ls,usr->par->k_us,usr->par->alpha,usr->par->A);
    type_bc[k] = BC_DIRICHLET_TRUE;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN Boundary - Vx
  ierr = DMStagBCListGetValues(bclist,'s','-',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_ux(x_bc[2*k],x_bc[2*k+1],usr->par->k_ls,usr->par->k_us,usr->par->alpha,usr->par->A);
    type_bc[k] = BC_DIRICHLET_TRUE;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN Boundary - Vz
  ierr = DMStagBCListGetValues(bclist,'s','|',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_uz(x_bc[2*k],x_bc[2*k+1],usr->par->k_ls,usr->par->k_us,usr->par->alpha,usr->par->A);
    type_bc[k] = BC_DIRICHLET_TRUE;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN Boundary - P
  ierr = DMStagBCListGetValues(bclist,'s','o',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_p(x_bc[2*k],x_bc[2*k+1],usr->par->k_ls,usr->par->k_us,usr->par->alpha,usr->par->A);
    type_bc[k] = BC_DIRICHLET_TRUE;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP Boundary - Vx
  ierr = DMStagBCListGetValues(bclist,'n','-',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_ux(x_bc[2*k],x_bc[2*k+1],usr->par->k_ls,usr->par->k_us,usr->par->alpha,usr->par->A);
    type_bc[k] = BC_DIRICHLET_TRUE;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP Boundary - Vz
  ierr = DMStagBCListGetValues(bclist,'n','|',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_uz(x_bc[2*k],x_bc[2*k+1],usr->par->k_ls,usr->par->k_us,usr->par->alpha,usr->par->A);
    type_bc[k] = BC_DIRICHLET_TRUE;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP Boundary - P
  ierr = DMStagBCListGetValues(bclist,'n','o',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_p(x_bc[2*k],x_bc[2*k+1],usr->par->k_ls,usr->par->k_us,usr->par->alpha,usr->par->A);
    type_bc[k] = BC_DIRICHLET_TRUE;
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
  PetscScalar    k_ls, k_us, A, alpha;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  k_ls = usr->par->k_ls;
  k_us = usr->par->k_us;
  alpha = usr->par->alpha;
  A     = usr->par->A;

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
      xxMMS[j][i][idx] = get_p(coordx[i][icenter],coordz[j][icenter],k_ls,k_us,alpha,A);

      // ux
      ierr = DMStagGetLocationSlot(dm,LEFT,0,&idx); CHKERRQ(ierr);
      xxMMS[j][i][idx] = get_ux(coordx[i][iprev],coordz[j][icenter],k_ls,k_us,alpha,A);

      if (i == Nx-1) {
        ierr = DMStagGetLocationSlot(dm,RIGHT,0,&idx); CHKERRQ(ierr);
        xxMMS[j][i][idx] = get_ux(coordx[i][inext],coordz[j][icenter],k_ls,k_us,alpha,A);
      }
      
      // uz
      ierr = DMStagGetLocationSlot(dm,DOWN,0,&idx); CHKERRQ(ierr);
      xxMMS[j][i][idx] = get_uz(coordx[i][icenter],coordz[j][iprev],k_ls,k_us,alpha,A);

      if (j == Nz-1) {
        ierr = DMStagGetLocationSlot(dm,UP,0,&idx); CHKERRQ(ierr);
        xxMMS[j][i][idx] = get_uz(coordx[i][icenter],coordz[j][inext],k_ls,k_us,alpha,A);
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
  PetscScalar    nrm[3], gnrm[3];
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
  nrm[0] = 0.0; nrm[1] = 0.0; nrm[2] = 0.0;
  dx = 1.0/Nx;
  dz = 1.0/Nz;
  dv = dx*dz;

  // Loop over local domain to calculate ELEMENT errors
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      
      PetscScalar    ve[4], pe;
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

      // Calculate errors
      ve[0] = PetscAbsScalar(xx[0]-xa[0]); // Left
      ve[1] = PetscAbsScalar(xx[1]-xa[1]); // Right
      ve[2] = PetscAbsScalar(xx[2]-xa[2]); // Down
      ve[3] = PetscAbsScalar(xx[3]-xa[3]); // Up
      pe    = PetscAbsScalar(xx[4]-xa[4]); // elem

      // Calculate norms
      if      (i == 0   ) { nrm[0] += ve[0]*dv*0.5; nrm[0] += ve[1]*dv; }
      else if (i == Nx-1) nrm[0] += ve[1]*dv*0.5;
      else                nrm[0] += ve[1]*dv;

      if      (j == 0   ) { nrm[1] += ve[2]*dv*0.5; nrm[1] += ve[3]*dv; }
      else if (j == Nz-1) nrm[1] += ve[3]*dv*0.5;
      else                nrm[1] += ve[3]*dv;

      nrm[2] += pe*dv;
    }
  }

  // Collect data 
  ierr = MPI_Allreduce(&nrm, &gnrm, 3, MPI_DOUBLE, MPI_SUM, comm); CHKERRQ(ierr);

  // Restore arrays and vectors
  ierr = DMRestoreLocalVector(dm, &xlocal ); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &xalocal ); CHKERRQ(ierr);

  // Print information
  PetscPrintf(comm,"# --------------------------------------- #\n");
  PetscPrintf(comm,"# NORMS: \n");
  PetscPrintf(comm,"# Velocity: norm1 = %1.12e norm1x = %1.12e norm1z = %1.12e \n",gnrm[0]+gnrm[1],gnrm[0],gnrm[1]);
  PetscPrintf(comm,"# Pressure: norm1 = %1.12e\n",gnrm[2]);
  PetscPrintf(comm,"# Grid info: hx = %1.12e hz = %1.12e \n",dx,dz);

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

  // Save input options filename
  for (int i = 1; i < argc; i++) {
    PetscBool flg;
    
    ierr = PetscStrcmp(argv[i],"-options_file",&flg); CHKERRQ(ierr);
    if (flg) { ierr = PetscStrcpy(usr->par->fname_in, argv[i+1]); CHKERRQ(ierr); }
  }

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
