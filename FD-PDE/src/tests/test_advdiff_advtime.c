// ---------------------------------------
// (ADVDIFF) Pure advection and time-stepping test
// run: ./tests/test_advdiff_advtime.app -pc_type lu -pc_factor_mat_solver_type umfpack -nx 10 -nz 10
// ---------------------------------------
static char help[] = "Application to solve advection of a Gaussian pulse in time (ADVDIFF) with FD-PDE \n\n";

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
#include "../fdpde_advdiff.h"
#include "../dmstagoutput.h"

// ---------------------------------------
// Application Context
// ---------------------------------------
#define FNAME_LENGTH 200

// parameters (bag)
typedef struct {
  PetscInt       nx, nz, tstep;
  PetscScalar    L, H;
  PetscScalar    xmin, zmin;
  PetscScalar    k, rho, cp, ux, uz;
  PetscScalar    t, dt;
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
PetscErrorCode Numerical_solution(void*,PetscInt);
PetscErrorCode InputParameters(UsrData**);
PetscErrorCode InputPrintData(UsrData*);
PetscErrorCode FormCoefficient(DM,Vec,DM,Vec,void*);
PetscErrorCode FormBCList(DM,Vec,DMStagBCList,void*);
PetscErrorCode SetGaussianInitialGuess(DM,Vec);

// ---------------------------------------
// Some descriptions
// ---------------------------------------
const char coeff_description[] =
"  << ADVDIFF - Advection Time Coefficients >> \n"
"  A = 1 (element)\n"
"  B = 0 (edge)\n"
"  C = 0 (element)\n"
"  u = [ux, uz] (edge)\n";

const char bc_description[] =
"  << Advection Time BCs >> \n"
"  LEFT: periodic \n"
"  RIGHT: periodic \n" 
"  DOWN: periodic \n" 
"  UP: periodic \n";

// ---------------------------------------
// Application functions
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "Numerical_solution"
PetscErrorCode Numerical_solution(void *ctx,PetscInt ts_scheme)
{
  UsrData       *usr = (UsrData*) ctx;
  DM             dm;
  Vec            x, xprev;
  FDPDE          fd;
  PetscInt       nx, nz, istep, tstep;
  PetscScalar    dx, dz,xmin, zmin, xmax, zmax, dt;
  char           fout[FNAME_LENGTH];
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Element count
  nx = usr->par->nx;
  nz = usr->par->nz;

  istep = 0;
  tstep = usr->par->tstep;

  // Domain coords
  dx = usr->par->L/(2*nx-2);
  dz = usr->par->H/(2*nz-2);

  // modify coord of dm such that unknowns are located on the boundary limits (0,1)
  xmin = usr->par->xmin-dx;
  zmin = usr->par->zmin-dz;
  xmax = usr->par->xmin+usr->par->L+dx;
  zmax = usr->par->zmin+usr->par->H+dz;

  // Create the FD-pde object
  ierr = FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_ADVDIFF,&fd);CHKERRQ(ierr);
  ierr = FDPDESetUp(fd);CHKERRQ(ierr);
  ierr = FDPDEAdvDiffSetAdvectSchemeType(fd,ADV_UPWIND);CHKERRQ(ierr);

  if (ts_scheme == 0) { ierr = FDPDEAdvDiffSetTimeStepSchemeType(fd,TS_FORWARD_EULER);CHKERRQ(ierr); }
  if (ts_scheme == 1) { ierr = FDPDEAdvDiffSetTimeStepSchemeType(fd,TS_BACKWARD_EULER);CHKERRQ(ierr); }
  if (ts_scheme == 2) { ierr = FDPDEAdvDiffSetTimeStepSchemeType(fd,TS_CRANK_NICHOLSON );CHKERRQ(ierr); }

  // Set BC evaluation function
  ierr = FDPDESetFunctionBCList(fd,FormBCList,bc_description,NULL); CHKERRQ(ierr);

  // Set coefficients evaluation function
  ierr = FDPDESetFunctionCoefficient(fd,FormCoefficient,coeff_description,usr); CHKERRQ(ierr);
  ierr = FDPDEView(fd); CHKERRQ(ierr);

  // Set timestep and CFL
  ierr = FDPDEAdvDiffSetTimestep(fd,usr->par->dt,0.0);CHKERRQ(ierr);

  // Set initial distribution - xguess
  ierr = FDPDEGetDM(fd, &dm); CHKERRQ(ierr);
  // ierr = FDPDEGetSolutionGuess(fd, &xguess);CHKERRQ(ierr);
  // ierr = VecSet(xguess,0.0);CHKERRQ(ierr);

  ierr = FDPDEAdvDiffGetPrevSolution(fd,&xprev);CHKERRQ(ierr);
  ierr = SetGaussianInitialGuess(dm,xprev);CHKERRQ(ierr);
  
  // ierr = VecDestroy(&xguess);CHKERRQ(ierr);
  ierr = VecDestroy(&xprev);CHKERRQ(ierr);

  // Time loop
  while (istep < tstep) {
    PetscPrintf(PETSC_COMM_WORLD,"# Timestep %d out of %d: time %1.3f\n\n",istep,tstep,usr->par->t);

    // FD SNES Solver
    ierr = FDPDESolve(fd,NULL);CHKERRQ(ierr);
    ierr = FDPDEGetSolution(fd,&x);CHKERRQ(ierr); 

    // Output solution
    if (istep % 5 == 0 ) {
      ierr = PetscSNPrintf(fout,sizeof(fout),"%s_ts%1.3d",usr->par->fname_out,istep);
      ierr = DMStagViewBinaryPython(dm,x,fout);CHKERRQ(ierr);
    }

    // Copy old solution to new
    ierr = FDPDEAdvDiffGetPrevSolution(fd,&xprev);CHKERRQ(ierr);
    ierr = VecCopy(x,xprev);CHKERRQ(ierr);

    // Destroy objects
    ierr = VecDestroy(&x);CHKERRQ(ierr);
    ierr = VecDestroy(&xprev);CHKERRQ(ierr);

    // increment time and timestep
    ierr = FDPDEAdvDiffGetTimestep(fd,&dt);CHKERRQ(ierr);
    usr->par->t += dt;
    istep++;
  }

  // Destroy FD-PDE object
  ierr = DMDestroy(&dm); CHKERRQ(ierr);
  ierr = FDPDEDestroy(&fd);CHKERRQ(ierr);

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

  // Numerical solution
  ierr = Numerical_solution(usr,0); CHKERRQ(ierr); // forward euler
  // ierr = Numerical_solution(usr,1); CHKERRQ(ierr); // backward euler
  // ierr = Numerical_solution(usr,2); CHKERRQ(ierr); // crank-nicholson

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
  ierr = PetscBagRegisterInt(bag, &par->tstep, 51, "tstep", "Number of time steps"); CHKERRQ(ierr);

  ierr = PetscBagRegisterScalar(bag, &par->xmin, 0.0, "xmin", "Start coordinate of domain in x-dir"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->zmin, 0.0, "zmin", "Start coordinate of domain in z-dir"); CHKERRQ(ierr);

  ierr = PetscBagRegisterScalar(bag, &par->L, 10.0, "L", "Length of domain in x-dir"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->H, 10.0, "H", "Height of domain in z-dir"); CHKERRQ(ierr);

  // Physical and material parameters
  ierr = PetscBagRegisterScalar(bag, &par->k, 0.0, "k", "Thermal conductivity"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->rho, 1.0, "rho", "Density"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->cp, 1.0, "cp", "Heat capacity"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->ux, 2.0, "ux", "Horizontal velocity"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->uz, 2.0, "uz", "Vertical velocity"); CHKERRQ(ierr);

  ierr = PetscBagRegisterScalar(bag, &par->dt, 1.0e-2, "dt", "Time step size"); CHKERRQ(ierr);

  // Input/output 
  ierr = PetscBagRegisterString(bag,&par->fname_out,FNAME_LENGTH,"out_num_advtime","output_file","Name for output file, set with: -output_file <filename>"); CHKERRQ(ierr);

  // Other variables
  par->fname_in[0] = '\0';
  par->t = 0.0;

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
  PetscPrintf(usr->comm,"# Test_advdiff_advtime: %s \n",&(date[0]));
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
// SetGaussianInitialGuess
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "SetGaussianInitialGuess"
PetscErrorCode SetGaussianInitialGuess(DM dm, Vec xguess)
{
  Vec            xglocal;
  PetscInt       i,j, sx, sz, nx, nz, icenter;
  PetscScalar    A, x0, z0, taox, taoz;
  PetscScalar    ***xg, **coordx, **coordz;

  PetscErrorCode ierr;
  PetscFunctionBegin;

  // Gaussian function parameters
  A    = 10.0;
  x0   = 5;
  z0   = 5;
  taox = 1.0;
  taoz = 1.0;

  // Get domain corners
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  // Get dm coordinates array
  ierr = DMStagGet1dCoordinateArraysDOFRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGet1dCoordinateLocationSlot(dm,ELEMENT,&icenter);CHKERRQ(ierr);

  // Create coefficient local vector
  ierr = DMCreateLocalVector(dm, &xglocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayDOF(dm, xglocal, &xg); CHKERRQ(ierr);

  // Loop over local domain - set initial density and viscosity
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      DMStagStencil point;
      PetscScalar   xp, zp, fval = 0.0;
      PetscInt      idx;

      point.i = i; point.j = j; point.loc = ELEMENT; point.c = 0;

      xp = coordx[i][icenter]; 
      zp = coordz[j][icenter];

      fval = A*PetscExpReal(-( (xp-x0)*(xp-x0)/taox/taox + (zp-z0)*(zp-z0)/taoz/taoz )); 
      ierr = DMStagGetLocationSlot(dm, point.loc, point.c, &idx); CHKERRQ(ierr);
      xg[j][i][idx] = fval;
    }
  }

  // Restore arrays, local vectors
  ierr = DMStagRestore1dCoordinateArraysDOFRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);

  ierr = DMStagVecRestoreArrayDOF(dm,xglocal,&xg);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm,xglocal,INSERT_VALUES,xguess); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,xglocal,INSERT_VALUES,xguess); CHKERRQ(ierr);
  
  ierr = VecDestroy(&xglocal); CHKERRQ(ierr);

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
  PetscScalar    rho, k, cp, v[2];
  PetscScalar    ***c;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  // Element: A = rho*cp (dof 0), C = heat production/sink (dof 1)
  // Edges: k (dof 0), velocity (dof 1)

  // User parameters
  rho = usr->par->rho;
  k   = usr->par->k;
  cp  = usr->par->cp;
  v[0]= usr->par->ux;
  v[1]= usr->par->uz;

  // Get domain corners
  ierr = DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  // Create coefficient local vector
  ierr = DMCreateLocalVector(dmcoeff, &coefflocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayDOF(dmcoeff, coefflocal, &c); CHKERRQ(ierr);
  
  // Loop over local domain - set initial density and viscosity
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {

      { // A = rho*cp
        DMStagStencil point;
        PetscInt      idx;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 0;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = rho*cp;
      }

      { // C = 0.0
        DMStagStencil point;
        PetscInt      idx;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 1;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = 0.0;
      }

      { // B = k = 0.0 (edge)
        DMStagStencil point[4];
        PetscInt      ii, idx;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 0;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = 0;
        point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = 0;

        for (ii = 0; ii < 4; ii++) {
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = k;
        }
      }

      { // u = velocity (edge)
        DMStagStencil point[4];
        PetscInt      ii, idx;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 1;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 1;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = 1;
        point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = 1;

        for (ii = 0; ii < 2; ii++) {
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = v[0];
        }
        for (ii = 2; ii < 4; ii++) {
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = v[1];
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
// FormBCList - periodic BC on Left/Down
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormBCList"
PetscErrorCode FormBCList(DM dm, Vec x, DMStagBCList bclist, void *ctx)
{
  Vec            xlocal;
  PetscInt       Nx, Nz, sx, sz, nx, nz;
  PetscInt       i,j, k,n_bc,*idx_bc, icenter;
  PetscScalar   *value_bc,*x_bc, xx;
  PetscScalar   **coordx,**coordz;
  BCType        *type_bc;
  DMStagStencil  point;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  
  // Get the dm dimensions 
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = DMStagGetGlobalSizes(dm,&Nx,&Nz,NULL);CHKERRQ(ierr);
  ierr = DMStagGet1dCoordinateArraysDOFRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGet1dCoordinateLocationSlot(dm,ELEMENT,&icenter);CHKERRQ(ierr);

  // Map global vectors to local domain
  ierr = DMGetLocalVector(dm, &xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm, x, INSERT_VALUES, xlocal); CHKERRQ(ierr);

  // Left: T(0,z) = T(Nx-1,z)
  ierr = DMStagBCListGetValues(bclist,'w','o',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    for (j = sz; j < sz+nz; j++) {
      if (x_bc[2*k+1]==coordz[j][icenter]) {
        point.i = Nx-1; point.j = j; point.loc = ELEMENT; point.c = 0;
        ierr = DMStagVecGetValuesStencil(dm,xlocal,1,&point, &xx); CHKERRQ(ierr);
        value_bc[k] = xx;
        type_bc[k] = BC_DIRICHLET;
      }
    }
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // Right: T(Nx-1,z) = T(0,z)
  ierr = DMStagBCListGetValues(bclist,'e','o',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    for (j = sz; j < sz+nz; j++) {
      if (x_bc[2*k+1]==coordz[j][icenter]) {
        point.i = 0; point.j = j; point.loc = ELEMENT; point.c = 0;
        ierr = DMStagVecGetValuesStencil(dm,xlocal,1,&point, &xx); CHKERRQ(ierr);
        value_bc[k] = xx;
        type_bc[k] = BC_DIRICHLET;
      }
    }
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  
  // DOWN: T(x,0) = T(x,Nz-1)
  ierr = DMStagBCListGetValues(bclist,'s','o',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    for (i = sx; i < sx+nx; i++) {
      if (x_bc[2*k]==coordx[i][icenter]) {
        point.i = i; point.j = Nz-1; point.loc = ELEMENT; point.c = 0;
        ierr = DMStagVecGetValuesStencil(dm,xlocal,1,&point, &xx); CHKERRQ(ierr);
        value_bc[k] = xx;
        type_bc[k] = BC_DIRICHLET;
      }
    }
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP: T(x,Nz-1) = T(x,0)
  ierr = DMStagBCListGetValues(bclist,'n','o',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    for (i = sx; i < sx+nx; i++) {
      if (x_bc[2*k]==coordx[i][icenter]) {
        point.i = i; point.j = 0; point.loc = ELEMENT; point.c = 0;
        ierr = DMStagVecGetValuesStencil(dm,xlocal,1,&point, &xx); CHKERRQ(ierr);
        value_bc[k] = xx;
        type_bc[k] = BC_DIRICHLET;
      }
    }
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  ierr = DMStagRestore1dCoordinateArraysDOFRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&xlocal); CHKERRQ(ierr);
 
  PetscFunctionReturn(0);
}