// ---------------------------------------
// (ADVDIFF) Pure advection and time-stepping test - PERIODIC BCs
// run: ./tests/test_advdiff_periodic.app -pc_type lu -pc_factor_mat_solver_type umfpack -pc_factor_mat_ordering_type external -nx 10 -nz 10
// python test: ./tests/python/test_advdiff_periodic.py
// ---------------------------------------
static char help[] = "Application to solve advection of a Gaussian pulse in time (ADVDIFF) with FD-PDE and PERIODIC BCs\n\n";

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
#include "../src/fdpde_advdiff.h"
#include "../src/dmstagoutput.h"

// ---------------------------------------
// Application Context
// ---------------------------------------
#define FNAME_LENGTH 200

// parameters (bag)
typedef struct {
  PetscInt       nx, nz, tstep;
  PetscInt       tout;
  PetscScalar    L, H;
  PetscScalar    xmin, zmin;
  PetscScalar    k, rho, cp, ux, uz;
  PetscScalar    t, dt;
  PetscScalar    A, x0, z0, taox, taoz;
  char           fname_out[FNAME_LENGTH]; 
  char           fname_in [FNAME_LENGTH];  
  char           fdir_out[FNAME_LENGTH]; 
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
PetscErrorCode Numerical_solution(void*);
PetscErrorCode InputParameters(UsrData**);
PetscErrorCode InputPrintData(UsrData*);
PetscErrorCode FormCoefficient(FDPDE,DM,Vec,DM,Vec,void*);
PetscErrorCode FormBCList(DM,Vec,DMStagBCList,void*);
PetscErrorCode SetGaussianInitialGuess(DM,Vec,void*);
PetscErrorCode Analytic_AdvTime(DM,void*,PetscInt);

// ---------------------------------------
// Some descriptions
// ---------------------------------------
const char coeff_description[] =
"  << ADVDIFF - Coefficients >> \n"
"  A = 1 (element)\n"
"  B = 0 (edge)\n"
"  C = 0 (element)\n"
"  u = [ux, uz] (edge)\n";

const char bc_description[] =
"  << ADVDIFF - BCs: PERIODIC >> \n";

// ---------------------------------------
// Application functions
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "Numerical_solution"
PetscErrorCode Numerical_solution(void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  DM             dm, dmcoeff;
  Vec            x, xprev, xguess;
  Vec            coeff, coeffprev;
  FDPDE          fd;
  PetscInt       nx, nz, istep, tstep;
  PetscScalar    dx, dz,xmin, zmin, xmax, zmax, dt;
  char           fout[FNAME_LENGTH];
  PetscFunctionBeginUser;

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
  PetscCall(FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_ADVDIFF,&fd));
  PetscCall(FDPDESetDMBoundaryType(fd,DM_BOUNDARY_PERIODIC,DM_BOUNDARY_PERIODIC));
  PetscCall(FDPDESetUp(fd));
  PetscCall(FDPDEAdvDiffSetAdvectSchemeType(fd,ADV_FROMM));
  PetscCall(FDPDEAdvDiffSetTimeStepSchemeType(fd,TS_CRANK_NICHOLSON ));
  PetscCall(FDPDESetFunctionBCList(fd,FormBCList,bc_description,NULL)); 
  PetscCall(FDPDESetFunctionCoefficient(fd,FormCoefficient,coeff_description,usr)); 

  // Set timestep
  PetscCall(FDPDEAdvDiffSetTimestep(fd,usr->par->dt));
  PetscCall(FDPDEView(fd)); 

  // Set initial distribution - xguess
  PetscCall(FDPDEGetDM(fd, &dm)); 
  PetscCall(FDPDEGetSolutionGuess(fd, &xguess));
  PetscCall(SetGaussianInitialGuess(dm,xguess,usr));

  PetscCall(FDPDEAdvDiffGetPrevSolution(fd,&xprev));
  PetscCall(VecCopy(xguess,xprev));
  
  // Set initial coefficient structure
  PetscCall(FDPDEGetCoefficient(fd,&dmcoeff,NULL));
  PetscCall(FDPDEAdvDiffGetPrevCoefficient(fd,&coeffprev));
  PetscCall(FormCoefficient(fd,dm,xprev,dmcoeff,coeffprev,usr));
  PetscCall(VecDestroy(&coeffprev));
  PetscCall(VecDestroy(&xguess));
  PetscCall(VecDestroy(&xprev));

  // Time loop
  while (istep < tstep) {
    PetscPrintf(PETSC_COMM_WORLD,"# Timestep %d out of %d: time %1.3f\n",istep,tstep,usr->par->t);

    // FD SNES Solver
    PetscCall(FDPDESolve(fd,NULL));
    PetscCall(FDPDEGetSolution(fd,&x)); 

    // increment time
    PetscCall(FDPDEAdvDiffGetTimestep(fd,&dt));
    usr->par->t += dt;

    // Copy old solution to new
    PetscCall(FDPDEAdvDiffGetPrevSolution(fd,&xprev));
    PetscCall(VecCopy(x,xprev));
    PetscCall(VecDestroy(&xprev));

    // Copy old coefficient to new
    PetscCall(FDPDEGetCoefficient(fd,NULL,&coeff));
    PetscCall(FDPDEAdvDiffGetPrevCoefficient(fd,&coeffprev));
    PetscCall(VecCopy(coeff,coeffprev));
    PetscCall(VecDestroy(&coeffprev));

    // Output solution
    if (istep % usr->par->tout == 0 ) {
      PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s_tstep%1.3d",usr->par->fdir_out,usr->par->fname_out,istep));
      PetscCall(DMStagViewBinaryPython(dm,x,fout));

      // Calculate analytical solution and output
      PetscCall(Analytic_AdvTime(dm,usr,istep));  
    }

    // Destroy objects
    PetscCall(VecDestroy(&x));

    // increment timestep
    istep++;
  }

  // Destroy FD-PDE object
  PetscCall(DMDestroy(&dm)); 
  PetscCall(FDPDEDestroy(&fd));

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
  PetscCall(PetscTime(&start_time)); 
  PetscCall(PetscOptionsInsert(PETSC_NULL,&argc,&argv,NULL)); 
  PetscCall(InputParameters(&usr)); 

  // Save input options filename
  for (int i = 1; i < argc; i++) {
    PetscBool flg;
    
    PetscCall(PetscStrcmp(argv[i],"-options_file",&flg)); 
    if (flg) { PetscCall(PetscStrcpy(usr->par->fname_in, argv[i+1]));  }
  }

  // Print user parameters
  PetscCall(InputPrintData(usr)); 

  // Numerical solution
  PetscCall(Numerical_solution(usr)); 

  PetscCall(PetscBagDestroy(&usr->bag)); 
  PetscCall(PetscFree(usr));             

  // End time
  PetscCall(PetscTime(&end_time)); 
  PetscPrintf(PETSC_COMM_WORLD,"# Total runtime: %g (sec) \n", end_time - start_time);
  PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");
  
  // Finalize main
  PetscCall(PetscFinalize());
  return 0;
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
  PetscCall(PetscBagRegisterInt(bag, &par->tstep, 1, "tstep", "Number of time steps")); 
  PetscCall(PetscBagRegisterInt(bag, &par->tout,5,"tout", "Output every <tout> time steps")); 

  PetscCall(PetscBagRegisterScalar(bag, &par->xmin, 0.0, "xmin", "Start coordinate of domain in x-dir")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->zmin, 0.0, "zmin", "Start coordinate of domain in z-dir")); 

  PetscCall(PetscBagRegisterScalar(bag, &par->L, 10.0, "L", "Length of domain in x-dir")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->H, 10.0, "H", "Height of domain in z-dir")); 

  // Physical and material parameters
  PetscCall(PetscBagRegisterScalar(bag, &par->k, 0.0, "k", "Thermal conductivity")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->rho, 1.0, "rho", "Density")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->cp, 1.0, "cp", "Heat capacity")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->ux, 2.0, "ux", "Horizontal velocity")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->uz, 2.0, "uz", "Vertical velocity")); 

  PetscCall(PetscBagRegisterScalar(bag, &par->dt, 1.0e-2, "dt", "Time step size")); 

  // Gaussian-shape initial guess
  PetscCall(PetscBagRegisterScalar(bag, &par->A, 10.0, "A", "Amplitude gaussian")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->x0,5.0, "x0", "Offset x-dir")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->z0,5.0, "z0", "Offset z-dir")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->taox,1.0, "taox", "tao parameter x-dir")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->taoz,1.0, "taoz", "tao parameter z-dir")); 

  // Input/output 
  PetscCall(PetscBagRegisterString(bag,&par->fname_out,FNAME_LENGTH,"out_num_adv_periodic","output_file","Name for output file, set with: -output_file <filename>")); 
  PetscCall(PetscBagRegisterString(bag,&par->fdir_out,FNAME_LENGTH,"./","output_dir","Name for output directory, set with: -output_dir <dirname>")); 

  // Other variables
  par->fname_in[0] = '\0';
  par->t = 0.0;

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
  PetscPrintf(usr->comm,"# Test_advdiff_periodic: %s \n",&(date[0]));
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
  PetscCall(PetscBagView(usr->bag,PETSC_VIEWER_STDOUT_WORLD)); 
  PetscPrintf(usr->comm,"# --------------------------------------- #\n");

  // Free memory
  PetscCall(PetscFree(opts)); 

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// SetGaussianInitialGuess
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "SetGaussianInitialGuess"
PetscErrorCode SetGaussianInitialGuess(DM dm, Vec xguess, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  Vec            xglocal;
  PetscInt       i,j, sx, sz, nx, nz, icenter;
  PetscScalar    A, x0, taox,z0, taoz;
  PetscScalar    ***xg, **coordx, **coordz;
  PetscFunctionBeginUser;

  // Gaussian function parameters
  A    = usr->par->A;
  x0   = usr->par->x0;
  taox = usr->par->taox;
  z0   = usr->par->z0;
  taoz = usr->par->taoz;

  // Get domain corners
  PetscCall(DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 

  // Get dm coordinates array
  PetscCall(DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter));

  // Create coefficient local vector
  PetscCall(DMCreateLocalVector(dm, &xglocal)); 
  PetscCall(DMStagVecGetArray(dm, xglocal, &xg)); 

  // Loop over local domain - set initial density and viscosity
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      DMStagStencil point;
      PetscScalar   xp,zp, fval = 0.0;
      PetscInt      idx;

      point.i = i; point.j = j; point.loc = ELEMENT; point.c = 0;
      xp = coordx[i][icenter]; 
      zp = coordz[j][icenter]; 
      fval = A*PetscExpReal(-( (xp-x0)*(xp-x0)/taox/taox + (zp-z0)*(zp-z0)/taoz/taoz )); 
      PetscCall(DMStagGetLocationSlot(dm, point.loc, point.c, &idx)); 
      xg[j][i][idx] = fval;
    }
  }

  // Restore arrays, local vectors
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));

  PetscCall(DMStagVecRestoreArray(dm,xglocal,&xg));
  PetscCall(DMLocalToGlobalBegin(dm,xglocal,INSERT_VALUES,xguess)); 
  PetscCall(DMLocalToGlobalEnd  (dm,xglocal,INSERT_VALUES,xguess)); 
  
  PetscCall(VecDestroy(&xglocal)); 

  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscScalar    rho, k, cp, v[2];
  PetscScalar    ***c;
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
  PetscCall(DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 

  // Create coefficient local vector
  PetscCall(DMCreateLocalVector(dmcoeff, &coefflocal)); 
  PetscCall(DMStagVecGetArray(dmcoeff, coefflocal, &c)); 
  
  // Loop over local domain - set initial density and viscosity
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {

      { // A = rho*cp
        DMStagStencil point;
        PetscInt      idx;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 0;
        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
        c[j][i][idx] = rho*cp;
      }

      { // C = 0.0
        DMStagStencil point;
        PetscInt      idx;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 1;
        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
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
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
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
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] = v[0];
        }
        for (ii = 2; ii < 4; ii++) {
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] = v[1];
        }
      }
    }
  }

  // Restore arrays, local vectors
  PetscCall(DMStagVecRestoreArray(dmcoeff,coefflocal,&c));
  PetscCall(DMLocalToGlobalBegin(dmcoeff,coefflocal,INSERT_VALUES,coeff)); 
  PetscCall(DMLocalToGlobalEnd  (dmcoeff,coefflocal,INSERT_VALUES,coeff)); 
  
  PetscCall(VecDestroy(&coefflocal)); 

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// FormBCList - PERIODIC
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormBCList"
PetscErrorCode FormBCList(DM dm, Vec x, DMStagBCList bclist, void *ctx)
{
  PetscInt       k,n_bc,*idx_bc;
  PetscScalar   *value_bc,*x_bc;
  BCType        *type_bc;
  PetscFunctionBeginUser;
  
  // Left
  PetscCall(DMStagBCListGetValues(bclist,'w','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    type_bc[k] = BC_PERIODIC;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));

  // Right
  PetscCall(DMStagBCListGetValues(bclist,'e','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    type_bc[k] = BC_PERIODIC;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
  
  // DOWN
  PetscCall(DMStagBCListGetValues(bclist,'s','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    type_bc[k] = BC_PERIODIC;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));

  // UP
  PetscCall(DMStagBCListGetValues(bclist,'n','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    type_bc[k] = BC_PERIODIC;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// Create analytical solution
// ---------------------------------------
PetscErrorCode Analytic_AdvTime(DM dm,void *ctx, PetscInt istep)
{
  UsrData       *usr = (UsrData*) ctx;
  PetscInt       i, j, sx, sz, nx, nz, idx,icenter;
  PetscScalar    A, x0, taox, t, ux, z0,taoz,uz;
  PetscScalar    ***xx;
  PetscScalar    **coordx,**coordz;
  char           fout[FNAME_LENGTH];
  Vec            x, xlocal;
  PetscFunctionBeginUser;

  // Gaussian function parameters
  A    = usr->par->A;
  x0   = usr->par->x0;
  taox = usr->par->taox;
  t    = usr->par->t;
  ux   = usr->par->ux;
  z0   = usr->par->z0;
  taoz = usr->par->taoz;
  uz   = usr->par->uz;

  // Create local and global vector associated with DM
  PetscCall(DMCreateGlobalVector(dm, &x     )); 
  PetscCall(DMCreateLocalVector (dm, &xlocal)); 

  // Get array associated with vector
  PetscCall(DMStagVecGetArray(dm,xlocal,&xx)); 

  // Get domain corners
  PetscCall(DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 
  
// Get dm coordinates array
  PetscCall(DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter)); 

  // Loop over local domain to calculate the SolCx analytical solution
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      PetscScalar  xp,zp, fval=0.0;
      xp = coordx[i][icenter]; 
      zp = coordz[j][icenter];
      if (xp>=usr->par->L) xp = xp-usr->par->L;
      if (zp>=usr->par->H) zp = zp-usr->par->H;
      fval = A*PetscExpReal(-((xp-x0-ux*t)*(xp-x0-ux*t)/taox/taox  + (zp-z0-uz*t)*(zp-z0-uz*t)/taoz/taoz )); 
      PetscCall(DMStagGetLocationSlot(dm, ELEMENT, 0, &idx)); 
      xx[j][i][idx] = fval;
    }
  }

  // Restore arrays
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));
  PetscCall(DMStagVecRestoreArray(dm,xlocal,&xx)); 

  // Map local to global
  PetscCall(DMLocalToGlobalBegin(dm,xlocal,INSERT_VALUES,x)); 
  PetscCall(DMLocalToGlobalEnd  (dm,xlocal,INSERT_VALUES,x)); 

  PetscCall(VecDestroy(&xlocal)); 

  // output
  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s_tstep%1.3d",usr->par->fdir_out,"out_analytic_solution",istep));
  PetscCall(DMStagViewBinaryPython(dm,x,fout));

  PetscCall(VecDestroy(&x)); 
  
  PetscFunctionReturn(PETSC_SUCCESS);
}
