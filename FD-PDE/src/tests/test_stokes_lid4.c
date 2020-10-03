// ---------------------------------------
// Modified from ./tests/test_stokes_solcx.c
// Four-Lid-driven rotational flow benchmark - constant grid spacing
// Direction of four lid movement: Up->rightward, Right->downward, Bottom->leftward, Left->upward 
// run: ./tests/test_stokes_lid4.app -pc_type lu -pc_factor_mat_solver_type umfpack -nx 10 -nz 10
// python test: ./tests/python/test_stokes_lid4.py
// ---------------------------------------
static char help[] = "Application to solve the four-lid-driven rotational flow benchmark with FD-PDE \n\n";

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
  PetscScalar    eta0, eta1, g;
  char           fname_out[FNAME_LENGTH]; 
  char           fname_in [FNAME_LENGTH];  
} Params;

// user defined and model-dependent variables
typedef struct {
  Params        *par;
  PetscBag       bag;
  MPI_Comm       comm;
  PetscMPIInt    rank;
} UsrData;

// ---------------------------------------
// Function definitions
// ---------------------------------------
PetscErrorCode SNESStokes_lid(DM*,Vec*,void*);
//PetscErrorCode Analytic_Solcx(DM,Vec*,void*);
PetscErrorCode InputParameters(UsrData**);
PetscErrorCode InputPrintData(UsrData*);
PetscErrorCode FormCoefficient(FDPDE, DM, Vec, DM, Vec, void*);
PetscErrorCode FormBCList(DM, Vec, DMStagBCList, void*);
//PetscErrorCode ComputeErrorNorms(DM,Vec,Vec,void*);

// ---------------------------------------
// Some descriptions
// ---------------------------------------
const char coeff_description[] =
"  << Stokes Coefficients >> \n"
"  eta_n/eta_c = eta0 if x<0.5, eta1 if x>0.5"
"  fux = 0 \n" 
"  fuz = rho*g \n"
"  g = 0 \n"
"  fp = 0 (incompressible)\n";

const char bc_description[] =
"  << Stokes BCs >> \n"
"  LEFT: Vx = 0, Vz = 1\n"
"  RIGHT: Vx = 0, Vz = -1\n" 
"  DOWN: Vz = 0, Vx = -1\n" 
"  UP: Vz = 0, Vx = 1 \n";

// ---------------------------------------
// Application functions
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "SNESStokes_lid"
PetscErrorCode SNESStokes_lid(DM *_dm, Vec *_x, void *ctx)
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
  ierr = FDPDESetLinearPreallocatorStencil(fd,PETSC_TRUE);CHKERRQ(ierr);
  ierr = FDPDESetUp(fd);CHKERRQ(ierr);
  // User can modify the dm coordinates anywhere between FDPDESetUp() and FDPDESolve()

  // Set BC evaluation function
  ierr = FDPDESetFunctionBCList(fd,FormBCList,bc_description,NULL); CHKERRQ(ierr);
  
  // Set coefficients evaluation function
  ierr = FDPDESetFunctionCoefficient(fd,FormCoefficient,coeff_description,usr); CHKERRQ(ierr);
  ierr = FDPDEView(fd); CHKERRQ(ierr);

  // Some SNES options
  PetscInt  maxit, maxf, its;
  PetscReal atol, rtol, stol;
  SNES      snes;

  ierr = FDPDEGetSNES(fd,&snes);CHKERRQ(ierr);

  // SNES Options - default info on convergence
  ierr = PetscOptionsSetValue(NULL, "-snes_monitor",         ""); CHKERRQ(ierr);
  ierr = PetscOptionsSetValue(NULL, "-ksp_monitor",          ""); CHKERRQ(ierr);
  ierr = PetscOptionsSetValue(NULL, "-snes_converged_reason",""); CHKERRQ(ierr);
  ierr = PetscOptionsSetValue(NULL, "-ksp_converged_reason", ""); CHKERRQ(ierr);
  //ierr = SNESSetOptionsPrefix(snes,"stk_");CHKERRQ(ierr);

    // FD SNES Solver
  ierr = FDPDESolve(fd,NULL);CHKERRQ(ierr);

  ierr = SNESGetIterationNumber(fd->snes,&its);    CHKERRQ(ierr);
  ierr = SNESGetTolerances(fd->snes, &atol, &rtol, &stol, &maxit, &maxf); CHKERRQ(ierr);
  
  // Print some SNES diagnostics
  ierr = PetscPrintf(fd->comm,"Number of SNES iterations = %d\n",its);
  ierr = PetscPrintf(fd->comm,"SNES: atol = %g, rtol = %g, stol = %g, maxit = %D, maxf = %D\n",(double)atol,(double)rtol,(double)stol,maxit,maxf); CHKERRQ(ierr);

  // Get solution vector
  ierr = FDPDEGetSolution(fd,&x);CHKERRQ(ierr); 
  ierr = FDPDEGetDM(fd, &dmPV); CHKERRQ(ierr);

  // Output solution to file
  ierr = DMStagViewBinaryPython(dmPV,x,usr->par->fname_out);CHKERRQ(ierr);
  {
    DM dmcoeff;
    Vec coeff;
    ierr = FDPDEGetCoefficient(fd,&dmcoeff,&coeff);CHKERRQ(ierr);
    ierr = DMStagViewBinaryPython(dmcoeff,coeff,"out_coefficients");CHKERRQ(ierr);
  }

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
  ierr = PetscBagRegisterScalar(bag, &par->zmin, 0.0, "zmin", "Start coordinate of domain in z-dir"); CHKERRQ(ierr);

  ierr = PetscBagRegisterScalar(bag, &par->L, 1.0, "L", "Length of domain in x-dir"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->H, 1.0, "H", "Height of domain in z-dir"); CHKERRQ(ierr);

  // Physical and material parameters
  ierr = PetscBagRegisterScalar(bag, &par->g, 0.0, "g", "Gravitational acceleration"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->eta0, 1.0, "eta0", "Viscosity eta0"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->eta1, 1.0, "eta1", "Viscosity eta1"); CHKERRQ(ierr);

  // Input/output 
  ierr = PetscBagRegisterString(bag,&par->fname_out,FNAME_LENGTH,"out_num_solution","output_file","Name for output file, set with: -output_file <filename>"); CHKERRQ(ierr);

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
  PetscPrintf(usr->comm,"# Test_stokes_lid: %s \n",&(date[0]));
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
  Vec            coefflocal, xlocal;
  PetscScalar    **coordx,**coordz;
  PetscInt       iprev, inext, icenter;
  PetscScalar    ***c;
  PetscScalar    g, L2 = 0.5;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  // Density is defined (edges): sin(pi*z)*cos(pi*x); 
  // Viscosity is defined (corner and center): solcx_eta0, for 0<=x<=0.5, solcx_eta1, for 0.5<x<=1
  // DM dm, Vec x - used for non-linear coefficients

  // User parameters
  g = -usr->par->g;

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

  ierr = DMGetLocalVector(dm, &xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm, x, INSERT_VALUES, xlocal); CHKERRQ(ierr);
  
  // Loop over local domain - set initial density and viscosity
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {

      { // fux = 0.0
        DMStagStencil point[2];
        PetscInt      ii, idx;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 0;

        for (ii = 0; ii < 2; ii++) {
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = 0.0;
        }
      }

      { // fuz = rho*g
        DMStagStencil point[2];
        PetscScalar   xp[2], zp[2], fval = 0.0;
        PetscInt      ii, idx;

        point[0].i = i; point[0].j = j; point[0].loc = DOWN; point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = UP;   point[1].c = 0;

        xp[0] = coordx[i][icenter]; zp[0] = coordz[j][iprev  ];
        xp[1] = coordx[i][icenter]; zp[1] = coordz[j][inext  ];

        for (ii = 0; ii < 2; ii++) {
          fval = g*PetscSinScalar(PETSC_PI*zp[ii]) * PetscCosScalar(PETSC_PI*xp[ii]); 
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = fval;
        }
      }

      { // fp = 0.0
        DMStagStencil point;
        PetscInt      idx;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 0;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);

        c[j][i][idx] = 0.0; //p/1.e4;
      }

      { // eta_c = eta0:eta1
        DMStagStencil point;
        PetscScalar   xp, fval = 0.0;
        PetscInt      idx;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 1;
        xp = coordx[i][icenter];

        if (xp <= L2) fval = usr->par->eta0;
        else          fval = usr->par->eta1;

        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = fval;
      }

      { // eta_n = eta0:eta1
        DMStagStencil point[4];
        PetscScalar   xp[4], fval = 0.0;
        PetscInt      ii, idx;

        point[0].i = i; point[0].j = j; point[0].loc = DOWN_LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = DOWN_RIGHT; point[1].c = 0;
        point[2].i = i; point[2].j = j; point[2].loc = UP_LEFT;    point[2].c = 0;
        point[3].i = i; point[3].j = j; point[3].loc = UP_RIGHT;   point[3].c = 0;

        xp[0] = coordx[i][iprev]; 
        xp[1] = coordx[i][inext];
        xp[2] = coordx[i][iprev];
        xp[3] = coordx[i][inext];

        for (ii = 0; ii < 4; ii++) {
          if (xp[ii] <= L2) fval = usr->par->eta0;
          else              fval = usr->par->eta1;

          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = fval;
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

  ierr = DMRestoreLocalVector(dm, &xlocal ); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// FormBCList
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormBCList"
PetscErrorCode FormBCList(DM dm, Vec x, DMStagBCList bclist, void *ctx)
{
  PetscInt    k,n_bc,*idx_bc;
  PetscScalar *value_bc;
  BCType      *type_bc;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;

  
  // Vz=0 on left boundary (w)
  ierr = DMStagBCListGetValues(bclist,'w','|',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 1.0;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  
  // Vz = 0 on right boundary (e)
  ierr = DMStagBCListGetValues(bclist,'e','|',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = -1.0;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  
  
    // Vx=1 on top boundary (n)
  ierr = DMStagBCListGetValues(bclist,'n','-',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 1.0;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  
  // Vx=0 on bottom boundary (s)
  ierr = DMStagBCListGetValues(bclist,'s','-',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = -1.0;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  
  
  // Vx=0 on left boundary (w)
  ierr = DMStagBCListGetValues(bclist,'w','-',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  
  // Vx=0 on right boundary (e)
  ierr = DMStagBCListGetValues(bclist,'e','-',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  
  // Vz=0 on top boundary (n)
  ierr = DMStagBCListGetValues(bclist,'n','|',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  
  // Vz=0 on bottom boundary (s)
  ierr = DMStagBCListGetValues(bclist,'s','|',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  

  
  // Pin pressure p=0 at the bottom left corner
  ierr = DMStagBCListGetValues(bclist,'s','o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  value_bc[0] = 0.0;
  type_bc[0] = BC_DIRICHLET_STAG;

  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  
  //  DMStagBCListView(bclist);

  
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
  Vec             xStokes; //,xAnalytic;
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
  ierr = SNESStokes_lid(&dmStokes, &xStokes, usr); CHKERRQ(ierr);

  /*
  // Analytical solution
  ierr = Analytic_Solcx(dmStokes, &xAnalytic, usr); CHKERRQ(ierr);

  // Compute norms
  ierr = ComputeErrorNorms(dmStokes, xStokes, xAnalytic, usr); CHKERRQ(ierr);
  */
  // Destroy objects
  ierr = DMDestroy(&dmStokes); CHKERRQ(ierr);
  ierr = VecDestroy(&xStokes); CHKERRQ(ierr);
  //ierr = VecDestroy(&xAnalytic); CHKERRQ(ierr);

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
