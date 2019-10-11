/* Application to solve the Corner flow (mid-ocean ridges) benchmark with FD-PDE */
static char help[] = "Application to solve the 2D corner flow (mid-ocean ridges) benchmark with FD-PDE \n\n";
// run: ./tests/test_stokes_mor.app -pc_type lu -pc_factor_mat_solver_type umfpack -nx 10 -nz 10

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
#include "../fdpde_stokes.h"
#include "../benchmark_cornerflow.h"
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
  PetscScalar    eta0, u0, rangle, rho0, g;
  PetscScalar    C1, C4, sina, radalpha;
  char           fname_out[FNAME_LENGTH]; 
  char           fname_in [FNAME_LENGTH];  
} Params;

// user defined and model-dependent variables
typedef struct {
  Params        *par;
  PetscBag       bag;
  PetscInt       dofV, dofP;
  MPI_Comm       comm;
  PetscMPIInt    rank;
} UsrData;

// ---------------------------------------
// Function definitions
// ---------------------------------------
PetscErrorCode SNESStokes_MOR(DM*,Vec*,void*);
PetscErrorCode Analytic_MOR(DM,Vec*,void*);
PetscErrorCode InputParameters(UsrData**);
PetscErrorCode InputPrintData(UsrData*);
PetscErrorCode FormCoefficient(DM, Vec, DM, Vec, void*);
PetscErrorCode FormBCList_MOR(DM, Vec, DMStagBCList, void*);
PetscErrorCode DoOutput_Stokes(DM,Vec,const char[]);
PetscErrorCode DoOutput(DM,Vec,const char[]);

// ---------------------------------------
// Some descriptions
// ---------------------------------------
const char coeff_description[] =
"  << Stokes MOR Coefficients >> \n"
"  eta_n/eta_c = eta0\n"
"  fux = 0 \n" 
"  fuz = rho*g = 0 \n" 
"  fp = 0 (incompressible)\n";

const char bc_description[] =
"  << Stokes MOR BCs >> \n"
"  ALL: Vx, Vz, P = analytical \n"
"  LID (rangle>0): Vx = u0, Vz = 0, P = 0 - not implemented\n";

// ---------------------------------------
// Application functions
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "SNESStokes_MOR"
PetscErrorCode SNESStokes_MOR(DM *_dm, Vec *_x, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  FDPDE          fd;
  DM             dmPV;
  Vec            x;
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
  ierr = FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_STOKES,&fd);CHKERRQ(ierr);
  ierr = FDPDESetUp(fd);CHKERRQ(ierr);

  // Set BC evaluation function
  ierr = FDPDESetFunctionBCList(fd,FormBCList_MOR,bc_description,usr); CHKERRQ(ierr);

  // Set coefficients evaluation function
  ierr = FDPDESetFunctionCoefficient(fd,FormCoefficient,coeff_description,usr); CHKERRQ(ierr);
  ierr = FDPDEView(fd); CHKERRQ(ierr);

  ierr = PetscOptionsSetValue(NULL, "-snes_monitor",         ""); CHKERRQ(ierr);
  ierr = PetscOptionsSetValue(NULL, "-ksp_monitor",          ""); CHKERRQ(ierr);
  ierr = PetscOptionsSetValue(NULL, "-snes_converged_reason",""); CHKERRQ(ierr);
  ierr = PetscOptionsSetValue(NULL, "-ksp_converged_reason", ""); CHKERRQ(ierr);

  // FD SNES Solver
  ierr = FDPDESolve(fd,NULL);CHKERRQ(ierr);

  // Get solution vector
  ierr = FDPDEGetSolution(fd,&x);CHKERRQ(ierr); 
  ierr = FDPDEGetDM(fd, &dmPV); CHKERRQ(ierr);

  // Output solution to file
  // ierr = DoOutput(dmPV,x,"numerical_solution_mor.vtr");CHKERRQ(ierr);
  ierr = DoOutput_Stokes(dmPV,x,"numerical_solution_mor.vtr");CHKERRQ(ierr);

  // Destroy FD-PDE object
  ierr = FDPDEDestroy(&fd);CHKERRQ(ierr);

  *_x  = x;
  *_dm = dmPV;

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
  ierr = PetscBagRegisterScalar(bag, &par->zmin,-1.0, "zmin", "Start coordinate of domain in z-dir"); CHKERRQ(ierr);

  ierr = PetscBagRegisterScalar(bag, &par->L, 1.0, "L", "Length of domain in x-dir"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->H, 1.0, "H", "Height of domain in z-dir"); CHKERRQ(ierr);

  // Physical and material parameters
  ierr = PetscBagRegisterScalar(bag, &par->g, 1.0, "g", "Gravitational acceleration"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->eta0, 1.0, "eta0", "Reference viscosity"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->rho0, 0.0, "rho0", "Reference density"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->u0, 1.0, "u0", "Half-spreading rate [cm/yr]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->rangle, 0.0, "rangle", "Ridge angle [deg]"); CHKERRQ(ierr);

  if (par->rangle>0.0) SETERRQ(usr->comm,PETSC_ERR_SUP,"Internal boundary conditions not implemented for rangle>0!");

  // Input/output 
  ierr = PetscBagRegisterString(bag,&par->fname_out,FNAME_LENGTH,"output","output_file","Name for output file, set with: -output_file <filename>"); CHKERRQ(ierr);

  // Other variables
  par->fname_in[0] = '\0';

  par->radalpha = par->rangle*PETSC_PI/180;
  par->sina = PetscSinScalar(par->radalpha);
  par->C1 = 2*par->sina*par->sina/(PETSC_PI-2*par->radalpha-PetscSinScalar(2*par->radalpha));
  par->C4 = -2/(PETSC_PI-2*par->radalpha-PetscSinScalar(2*par->radalpha));

  // dofs
  usr->dofV = (par->nx+1)*par->nz + par->nx*(par->nz+1);
  usr->dofP = par->nx*par->nz;

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
  PetscPrintf(usr->comm,"# Test_stokes_mor (Corner flow): %s \n",&(date[0]));
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
PetscErrorCode FormCoefficient(DM dm, Vec x, DM dmcoeff, Vec coeff, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       i, j, sx, sz, nx, nz;
  Vec            coefflocal;
  PetscScalar    ***c, g, rho;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  // User parameters
  g   = -usr->par->g;
  rho = usr->par->rho0;

  // Get domain corners
  ierr = DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  // Create coefficient local vector
  ierr = DMCreateLocalVector(dmcoeff, &coefflocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayDOF(dmcoeff, coefflocal, &c); CHKERRQ(ierr);
  
  // Loop over local domain - set initial density and viscosity
  for (j = sz; j < sz+nz; ++j) {
    for (i = sx; i <sx+nx; ++i) {

      { // fux = 0.0
        DMStagStencil point[2];
        PetscInt      ii, idx;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 0;

        for (ii = 0; ii < 2; ++ii) {
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = 0.0;
        }
      }

      { // fuz = rho*g = 0 
        DMStagStencil point[2];
        PetscScalar   fval = 0.0;
        PetscInt      ii, idx;

        point[0].i = i; point[0].j = j; point[0].loc = DOWN; point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = UP;   point[1].c = 0;

        for (ii = 0; ii < 2; ++ii) {
          fval = g*rho; 
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = fval;
        }
      }

      { // fp = 0.0
        DMStagStencil point;
        PetscInt      idx;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 0;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = 0.0;
      }

      { // eta_c = eta0
        DMStagStencil point;
        PetscScalar   fval = 0.0;
        PetscInt      idx;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 1;

        fval = usr->par->eta0;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = fval;
      }

      { // eta_n = eta0
        DMStagStencil point[4];
        PetscScalar   fval = 0.0;
        PetscInt      ii, idx;

        point[0].i = i; point[0].j = j; point[0].loc = DOWN_LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = DOWN_RIGHT; point[1].c = 0;
        point[2].i = i; point[2].j = j; point[2].loc = UP_LEFT;    point[2].c = 0;
        point[3].i = i; point[3].j = j; point[3].loc = UP_RIGHT;   point[3].c = 0;

        for (ii = 0; ii < 4; ++ii) {
          fval = usr->par->eta0;
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = fval;
        }
      }
    }
  }

  // Restore arrays, local vectors
  ierr = DMStagVecRestoreArrayDOF(dmcoeff,coefflocal,&c);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  
  ierr = VecDestroy(&coefflocal); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// FormBCList_MOR
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormBCList_MOR"
PetscErrorCode FormBCList_MOR(DM dm, Vec x, DMStagBCList bclist, void *ctx)
{
  UsrData     *usr = (UsrData*)ctx;
  PetscInt     k,n_bc,*idx_bc;
  PetscScalar *value_bc,*x_bc;
  BCType      *type_bc;
  PetscScalar  p, v[2], C1, C4, u0, eta0;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;

  C1 = usr->par->C1;
  C4 = usr->par->C4;
  u0 = usr->par->u0;
  eta0 = usr->par->eta0;

  // LEFT Boundary - Vx
  ierr = DMStagBCListGetValues(bclist,'w','-',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    evaluate_CornerFlow_MOR(C1, C4, u0, eta0, x_bc[2*k], x_bc[2*k+1], v, &p);
    value_bc[k] = v[0];
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // LEFT Boundary - Vz
  ierr = DMStagBCListGetValues(bclist,'w','|',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    evaluate_CornerFlow_MOR(C1, C4, u0, eta0, x_bc[2*k], x_bc[2*k+1], v, &p);
    value_bc[k] = v[1];
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // LEFT Boundary - P
  ierr = DMStagBCListGetValues(bclist,'w','o',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    evaluate_CornerFlow_MOR(C1, C4, u0, eta0, x_bc[2*k], x_bc[2*k+1], v, &p);
    value_bc[k] = p;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  
  // RIGHT Boundary - Vx
  ierr = DMStagBCListGetValues(bclist,'e','-',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    evaluate_CornerFlow_MOR(C1, C4, u0, eta0, x_bc[2*k], x_bc[2*k+1], v, &p);
    value_bc[k] = v[0];
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // RIGHT Boundary - Vz
  ierr = DMStagBCListGetValues(bclist,'e','|',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    evaluate_CornerFlow_MOR(C1, C4, u0, eta0, x_bc[2*k], x_bc[2*k+1], v, &p);
    value_bc[k] = v[1];
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // RIGHT Boundary - P
  ierr = DMStagBCListGetValues(bclist,'e','o',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    evaluate_CornerFlow_MOR(C1, C4, u0, eta0, x_bc[2*k], x_bc[2*k+1], v, &p);
    value_bc[k] = p;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN Boundary - Vx
  ierr = DMStagBCListGetValues(bclist,'s','-',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    evaluate_CornerFlow_MOR(C1, C4, u0, eta0, x_bc[2*k], x_bc[2*k+1], v, &p);
    value_bc[k] = v[0];
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN Boundary - Vz
  ierr = DMStagBCListGetValues(bclist,'s','|',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    evaluate_CornerFlow_MOR(C1, C4, u0, eta0, x_bc[2*k], x_bc[2*k+1], v, &p);
    value_bc[k] = v[1];
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN Boundary - P
  ierr = DMStagBCListGetValues(bclist,'s','o',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    evaluate_CornerFlow_MOR(C1, C4, u0, eta0, x_bc[2*k], x_bc[2*k+1], v, &p);
    value_bc[k] = p;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP Boundary - Vx
  ierr = DMStagBCListGetValues(bclist,'n','-',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    evaluate_CornerFlow_MOR(C1, C4, u0, eta0, x_bc[2*k], x_bc[2*k+1], v, &p);
    value_bc[k] = v[0];
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP Boundary - Vz
  ierr = DMStagBCListGetValues(bclist,'n','|',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    evaluate_CornerFlow_MOR(C1, C4, u0, eta0, x_bc[2*k], x_bc[2*k+1], v, &p);
    value_bc[k] = v[1];
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP Boundary - P
  ierr = DMStagBCListGetValues(bclist,'n','o',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    evaluate_CornerFlow_MOR(C1, C4, u0, eta0, x_bc[2*k], x_bc[2*k+1], v, &p);
    value_bc[k] = p;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// Create corner flow (MOR) analytical solution
// ---------------------------------------
PetscErrorCode Analytic_MOR(DM dm,Vec *_x, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz, idx;
  PetscInt       iprev, inext, icenter;
  PetscScalar    eta0, C1, C4, u0;
  PetscScalar    ***xx;
  PetscScalar    **coordx,**coordz;
  Vec            x, xlocal;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Get parameters
  C1 = usr->par->C1;
  C4 = usr->par->C4;
  u0 = usr->par->u0;
  eta0 = usr->par->eta0;

  // Create local and global vector associated with DM
  ierr = DMCreateGlobalVector(dm, &x     ); CHKERRQ(ierr);
  ierr = DMCreateLocalVector (dm, &xlocal); CHKERRQ(ierr);

  // Get array associated with vector
  ierr = DMStagVecGetArrayDOF(dm,xlocal,&xx); CHKERRQ(ierr);

  // Get domain corners
  ierr = DMStagGetGlobalSizes(dm, &Nx, &Nz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  
// Get dm coordinates array
  ierr = DMStagGet1dCoordinateArraysDOFRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGet1dCoordinateLocationSlot(dm,ELEMENT,&icenter);CHKERRQ(ierr); 
  ierr = DMStagGet1dCoordinateLocationSlot(dm,LEFT,&iprev);CHKERRQ(ierr);
  ierr = DMStagGet1dCoordinateLocationSlot(dm,RIGHT,&inext);CHKERRQ(ierr); 

  // Loop over local domain to calculate the SolCx analytical solution
  for (j = sz; j < sz+nz; ++j) {
    for (i = sx; i <sx+nx; ++i) {
      PetscScalar xp, zp, v[2], p;

      // Vx
      xp = coordx[i][iprev ]; 
      zp = coordz[j][icenter];
      evaluate_CornerFlow_MOR(C1, C4, u0, eta0, xp, zp, v, &p);
      ierr = DMStagGetLocationSlot(dm, LEFT, 0, &idx); CHKERRQ(ierr);
      xx[j][i][idx] = v[0];

      if (i == Nx-1) {
        xp = coordx[i][inext  ];
        zp = coordz[j][icenter];
        evaluate_CornerFlow_MOR(C1, C4, u0, eta0, xp, zp, v, &p);
        ierr = DMStagGetLocationSlot(dm, RIGHT, 0, &idx); CHKERRQ(ierr);
        xx[j][i][idx] = v[0];
      }

      // Vz
      xp = coordx[i][icenter];
      zp = coordz[j][iprev  ];
      evaluate_CornerFlow_MOR(C1, C4, u0, eta0, xp, zp, v, &p);
      ierr = DMStagGetLocationSlot(dm, DOWN, 0, &idx); CHKERRQ(ierr);
      xx[j][i][idx] = v[1];

      if (j == Nz-1) {
        xp = coordx[i][icenter];
        zp = coordz[j][inext  ];
        evaluate_CornerFlow_MOR(C1, C4, u0, eta0, xp, zp, v, &p);
        ierr = DMStagGetLocationSlot(dm, UP, 0, &idx); CHKERRQ(ierr);
        xx[j][i][idx] = v[1];
      }
    
      // P
      xp = coordx[i][icenter];
      zp = coordz[j][icenter];
      evaluate_CornerFlow_MOR(C1, C4, u0, eta0, xp, zp, v, &p);
      ierr = DMStagGetLocationSlot(dm, ELEMENT, 0, &idx); CHKERRQ(ierr);
      xx[j][i][idx] = p;
    }
  }

  // Restore arrays
  ierr = DMStagRestore1dCoordinateArraysDOFRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArrayDOF(dm,xlocal,&xx); CHKERRQ(ierr);

  // Map local to global
  ierr = DMLocalToGlobalBegin(dm,xlocal,INSERT_VALUES,x); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,xlocal,INSERT_VALUES,x); CHKERRQ(ierr);

  ierr = VecDestroy(&xlocal); CHKERRQ(ierr);

  ierr = DoOutput(dm,x,"analytic_solution_mor.vtr");CHKERRQ(ierr);

  // Assign pointers
  *_x  = x;
  
  PetscFunctionReturn(0);
}

// ---------------------------------------
// DoOutput_Stokes
// ---------------------------------------
PetscErrorCode DoOutput_Stokes(DM dm,Vec x,const char fname[])
{
  DMStagOutputLabel *labels;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  // get labels list - reflects the structure of DMStag (dofs)
  ierr = DMStagOutputGetLabels(dm,&labels); CHKERRQ(ierr);

  // add labels to output
  ierr = DMStagOutputAddLabel(dm,labels,"Velocity []",0,LEFT   ); // faces (vector)
  ierr = DMStagOutputAddLabel(dm,labels,"Pressure []",0,ELEMENT); // element (scalar)

  // output - may choose different types
  // ierr = DMStagOutputVTKBinary(dm,x,labels,VTK_CENTER,fname);CHKERRQ(ierr);
  ierr = DMStagOutputVTKBinary(dm,x,labels,VTK_CORNER,fname);CHKERRQ(ierr);

  // Free labels
  ierr = PetscFree(labels);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// DoOutput
// ---------------------------------------
PetscErrorCode DoOutput(DM dm,Vec x,const char fname[])
{
  DM             dmVel,  daVel, daP;
  Vec            vecVel, vaVel, vecP;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  // Create a new DM and Vec for velocity
  ierr = DMStagCreateCompatibleDMStag(dm,0,0,2,0,&dmVel); CHKERRQ(ierr);
  ierr = DMSetUp(dmVel); CHKERRQ(ierr);
  ierr = DMStagSetUniformCoordinatesExplicit(dmVel,0.0,1.0,-1.0,0.0,0.0,0.0); CHKERRQ(ierr);

  // Create global vectors
  ierr = DMCreateGlobalVector(dmVel,&vecVel); CHKERRQ(ierr);
  
  // Loop over elements
  {
    PetscInt     i, j, sx, sz, nx, nz;
    Vec          xlocal;
    
    // Access local vector
    ierr = DMGetLocalVector(dm,&xlocal); CHKERRQ(ierr);
    ierr = DMGlobalToLocal (dm,x,INSERT_VALUES,xlocal); CHKERRQ(ierr);
    
    // Get corners
    ierr = DMStagGetCorners(dmVel,&sx,&sz,NULL,&nx,&nz,NULL,NULL,NULL,NULL); CHKERRQ(ierr);
    
    // Loop
    for (j = sz; j < sz+nz; ++j) {
      for (i = sx; i < sx+nx; ++i) {
        DMStagStencil from[4], to[2];
        PetscScalar   valFrom[4], valTo[2];
        
        from[0].i = i; from[0].j = j; from[0].loc = UP;    from[0].c = 0;
        from[1].i = i; from[1].j = j; from[1].loc = DOWN;  from[1].c = 0;
        from[2].i = i; from[2].j = j; from[2].loc = LEFT;  from[2].c = 0;
        from[3].i = i; from[3].j = j; from[3].loc = RIGHT; from[3].c = 0;
        
        // Get values from stencil locations
        ierr = DMStagVecGetValuesStencil(dm,xlocal,4,from,valFrom); CHKERRQ(ierr);
        
        // Average edge values to obtain ELEMENT values
        to[0].i = i; to[0].j = j; to[0].loc = ELEMENT; to[0].c = 0; valTo[0] = 0.5 * (valFrom[2] + valFrom[3]);
        to[1].i = i; to[1].j = j; to[1].loc = ELEMENT; to[1].c = 1; valTo[1] = 0.5 * (valFrom[0] + valFrom[1]);
        
        // Return values in new dm - averaged velocities
        ierr = DMStagVecSetValuesStencil(dmVel,vecVel,2,to,valTo,INSERT_VALUES); CHKERRQ(ierr);
      }
    }
    
    // Vector assembly
    ierr = VecAssemblyBegin(vecVel); CHKERRQ(ierr);
    ierr = VecAssemblyEnd  (vecVel); CHKERRQ(ierr);
    
    // Restore vector
    ierr = DMRestoreLocalVector(dm, &xlocal); CHKERRQ(ierr);
  }

  // Create individual DMDAs for sub-grids of our DMStag objects
  ierr = DMStagVecSplitToDMDA(dm,x,ELEMENT,0,&daP,&vecP); CHKERRQ(ierr);
  ierr = PetscObjectSetName  ((PetscObject)vecP,"Pressure");         CHKERRQ(ierr);
  
  ierr = DMStagVecSplitToDMDA(dmVel,vecVel,ELEMENT,-3,&daVel,&vaVel); CHKERRQ(ierr); // note -3 : output 2 DOFs
  ierr = PetscObjectSetName  ((PetscObject)vaVel,"Velocity");          CHKERRQ(ierr);

  // Dump element-based fields to a .vtr file
  {
    PetscViewer viewer;

    // Warning: is being output as Point Data instead of Cell Data - the grid is shifted to be in the center points.
    ierr = PetscViewerVTKOpen(PetscObjectComm((PetscObject)daVel),fname,FILE_MODE_WRITE,&viewer); CHKERRQ(ierr);
    ierr = VecView(vaVel,    viewer); CHKERRQ(ierr);
    ierr = VecView(vecP,     viewer); CHKERRQ(ierr);
    
    // Free memory
    ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
  }

  // Destroy DMDAs and Vecs
  ierr = VecDestroy(&vecVel); CHKERRQ(ierr);
  ierr = VecDestroy(&vaVel ); CHKERRQ(ierr);
  ierr = VecDestroy(&vecP  ); CHKERRQ(ierr);
  
  ierr = DMDestroy(&dmVel  ); CHKERRQ(ierr);
  ierr = DMDestroy(&daVel  ); CHKERRQ(ierr);
  ierr = DMDestroy(&daP    ); CHKERRQ(ierr);

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
  DM              dmStokes;
  Vec             xStokes,xAnalytic;
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

  //Numerical solution using the FD pde object
  ierr = SNESStokes_MOR(&dmStokes, &xStokes, usr); CHKERRQ(ierr);

  // Analytical solution
  ierr = Analytic_MOR(dmStokes, &xAnalytic, usr); CHKERRQ(ierr);

  // Destroy objects
  ierr = DMDestroy(&dmStokes); CHKERRQ(ierr);
  ierr = VecDestroy(&xStokes); CHKERRQ(ierr);
  ierr = VecDestroy(&xAnalytic); CHKERRQ(ierr);

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
