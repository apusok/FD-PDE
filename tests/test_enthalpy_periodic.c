// ---------------------------------------
// (ENTHALPY) Pure advection and time-stepping test - PERIODIC BCs
// run: ./tests/test_enthlpy_periodic.app -pc_type lu -pc_factor_mat_solver_type umfpack -pc_factor_mat_ordering_type external -nx 10 -nz 10
// python test: ./tests/python/test_enthalpy_periodic.py
// ---------------------------------------
static char help[] = "Application to solve advection of a Gaussian pulse in time (ENTHALPY) with FD-PDE and PERIODIC BCs\n\n";

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
#include "../src/fdpde_enthalpy.h"
#include "../src/dmstagoutput.h"

// ---------------------------------------
// Application Context
// ---------------------------------------
#define FNAME_LENGTH 200

// parameters (bag)
typedef struct {
  PetscInt       nx, nz, tstep;
  PetscInt       tout, ncomp;
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
PetscErrorCode InputParameters(UsrData**);
PetscErrorCode InputPrintData(UsrData*);
PetscErrorCode Numerical_solution(void*);
PetscErrorCode FormCoefficient(FDPDE, DM, Vec, DM, Vec, void*);
PetscErrorCode FormBCList(DM, Vec, DMStagBCList, void*);
PetscErrorCode Form_PotentialTemperature(PetscScalar,PetscScalar,PetscScalar*,void*); 
EnthEvalErrorCode Form_Enthalpy(PetscScalar,PetscScalar[],PetscScalar,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscInt,void*); 
PetscErrorCode SetGaussianInitialGuess(DM,Vec,void*);

const char coeff_description[] =
"  << ENTHALPY Coefficients >> \n"
"  A1 = 1, B1 = 0, C1 = 0, D1 = 0  \n"
"  A2 = 0, B2 = 1, C2 = 0, D2 = 0  \n"
"  v = [ux,uz], vs = [ux,uz], vf = [ux,uz] \n";

const char bc_description[] =
"  << ENTHALPY BCs - PERIODIC >> \n";

const char enthalpy_method_description[] =
"  << ENTHALPY METHOD >> \n"
"  Input: H, C, P \n"
"  Output: T  = H, \n"
"          Cf = Cs = C, \n"
"          phi = 1.0 \n";

// ---------------------------------------
// Application functions
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "Numerical_solution"
PetscErrorCode Numerical_solution(void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  Params        *par;
  FDPDE          fd;
  DM             dm, dmcoeff, dmP;
  Vec            x, xprev, xguess, xcoeff, xcoeffprev, xP, xPprev;
  PetscInt       nx, nz, istep, tstep;
  PetscScalar    xmin, zmin, xmax, zmax, dx, dz, dt;
  char           fout[FNAME_LENGTH];
  PetscErrorCode ierr;
  PetscFunctionBegin;

  par = usr->par;
  // Element count
  nx = par->nx;
  nz = par->nz;

  istep = 0;
  tstep = usr->par->tstep;

  // Domain coords
  dx = par->L/(2*nx-2);
  dz = par->H/(2*nz-2);
  xmin = par->xmin-dx;
  zmin = par->zmin-dz;
  xmax = par->xmin+par->L+dx;
  zmax = par->zmin+par->H+dz;

  // Set up Enthalpy system
  ierr = FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_ENTHALPY,&fd);CHKERRQ(ierr);
  ierr = FDPDEEnthalpySetNumberComponentsPhaseDiagram(fd,par->ncomp);CHKERRQ(ierr);
  ierr = FDPDESetDMBoundaryType(fd,DM_BOUNDARY_PERIODIC,DM_BOUNDARY_PERIODIC);CHKERRQ(ierr);
  ierr = FDPDESetUp(fd);CHKERRQ(ierr);

  ierr = FDPDESetFunctionBCList(fd,FormBCList,bc_description,usr); CHKERRQ(ierr);
  ierr = FDPDESetFunctionCoefficient(fd,FormCoefficient,coeff_description,usr); CHKERRQ(ierr);
  ierr = FDPDEEnthalpySetAdvectSchemeType(fd,ADV_FROMM);CHKERRQ(ierr);
  ierr = FDPDEEnthalpySetTimeStepSchemeType(fd,TS_CRANK_NICHOLSON );CHKERRQ(ierr);

  ierr = FDPDEEnthalpySetTimestep(fd,par->dt); CHKERRQ(ierr);
  ierr = FDPDEEnthalpySetEnthalpyMethod(fd,Form_Enthalpy,enthalpy_method_description,usr);CHKERRQ(ierr);
  ierr = FDPDEEnthalpySetPotentialTemp(fd,Form_PotentialTemperature,usr);CHKERRQ(ierr);
  ierr = FDPDEView(fd);CHKERRQ(ierr);

  // Set initial condition 
  ierr = FDPDEGetDM(fd,&dm);CHKERRQ(ierr);
  ierr = FDPDEEnthalpyGetPrevSolution(fd,&xprev);CHKERRQ(ierr);
  ierr = SetGaussianInitialGuess(dm,xprev,usr);CHKERRQ(ierr);
  ierr = FDPDEGetSolutionGuess(fd,&xguess);CHKERRQ(ierr);
  ierr = VecCopy(xprev,xguess);CHKERRQ(ierr);
  ierr = VecDestroy(&xguess);CHKERRQ(ierr);

  // Set initial coefficient structure
  ierr = FDPDEGetCoefficient(fd,&dmcoeff,NULL);CHKERRQ(ierr);
  ierr = FDPDEEnthalpyGetPrevCoefficient(fd,&xcoeffprev);CHKERRQ(ierr);
  ierr = FormCoefficient(fd,dm,xprev,dmcoeff,xcoeffprev,usr);CHKERRQ(ierr);
  ierr = VecDestroy(&xcoeffprev);CHKERRQ(ierr);
  ierr = VecDestroy(&xprev);CHKERRQ(ierr);

  // Set initial pressure - not needed so initialize with constant
  ierr = FDPDEEnthalpyGetPressure(fd,&dmP,NULL);CHKERRQ(ierr);
  ierr = FDPDEEnthalpyGetPrevPressure(fd,&xPprev);CHKERRQ(ierr);
  ierr = VecSet(xPprev,-1.0);CHKERRQ(ierr);
  ierr = VecDestroy(&xPprev);CHKERRQ(ierr);

  // Time loop
  while (istep<tstep) {
    PetscPrintf(PETSC_COMM_WORLD,"# Timestep %d out of %d: time %1.3f\n",istep,tstep,usr->par->t);

    // update pressure
    ierr = FDPDEEnthalpyGetPressure(fd,NULL,&xP);CHKERRQ(ierr);
    ierr = VecSet(xP,istep);CHKERRQ(ierr);

    // Enthalpy Solver
    ierr = FDPDESolve(fd,NULL);CHKERRQ(ierr);
    ierr = FDPDEGetSolution(fd,&x);CHKERRQ(ierr);

    // increment time
    ierr = FDPDEEnthalpyGetTimestep(fd,&dt);CHKERRQ(ierr);
    usr->par->t += dt;

    // Copy solution and coefficient to old
    ierr = FDPDEEnthalpyGetPrevSolution(fd,&xprev);CHKERRQ(ierr);
    ierr = VecCopy(x,xprev);CHKERRQ(ierr);
    ierr = VecDestroy(&xprev);CHKERRQ(ierr);

    ierr = FDPDEGetCoefficient(fd,&dmcoeff,&xcoeff);CHKERRQ(ierr);
    ierr = FDPDEEnthalpyGetPrevCoefficient(fd,&xcoeffprev);CHKERRQ(ierr);
    ierr = VecCopy(xcoeff,xcoeffprev);CHKERRQ(ierr);
    ierr = VecDestroy(&xcoeffprev);CHKERRQ(ierr);

    ierr = FDPDEEnthalpyGetPrevPressure(fd,&xPprev);CHKERRQ(ierr);
    ierr = VecCopy(xP,xPprev);CHKERRQ(ierr);
    ierr = VecDestroy(&xP);CHKERRQ(ierr);
    ierr = VecDestroy(&xPprev);CHKERRQ(ierr);

    // Output solution
    if (istep % par->tout == 0 ) {
      ierr = PetscSNPrintf(fout,sizeof(fout),"%s/%s_ts%1.3d",par->fdir_out,par->fname_out,istep);
      ierr = DMStagViewBinaryPython(dm,x,fout);CHKERRQ(ierr);
    }
    ierr = VecDestroy(&x);CHKERRQ(ierr);

    // increment timestep
    istep++;
  }

  // Destroy objects
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = DMDestroy(&dmP);CHKERRQ(ierr);
  ierr = FDPDEDestroy(&fd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// ---------------------------------------
// Phase Diagram
// ---------------------------------------
EnthEvalErrorCode Form_Enthalpy(PetscScalar H,PetscScalar C[],PetscScalar P,PetscScalar *_T,PetscScalar *_phi,PetscScalar *CF,PetscScalar *CS,PetscInt ncomp, void *ctx) 
{
  PetscInt     ii;
  PetscScalar  T, phi;

  T   = H;
  phi = 1.0;
  for (ii = 0; ii<ncomp; ii++) { 
    CS[ii] = C[ii];
    CF[ii] = C[ii];
  }

  // assign pointers
  *_T = T;
  *_phi = phi;
  return(STATE_VALID);
}

PetscErrorCode Form_PotentialTemperature(PetscScalar T,PetscScalar P,PetscScalar *_TP, void *ctx) 
{
  PetscScalar  TP;
  PetscFunctionBegin;

  TP = T*1.0;
  *_TP = TP;

  PetscFunctionReturn(0);
}

// ---------------------------------------
// FormBCList
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormBCList"
PetscErrorCode FormBCList(DM dm, Vec x, DMStagBCList bclist, void *ctx)
{
  UsrData     *usr = (UsrData*)ctx;
  PetscInt    k,n_bc,*idx_bc, ii;
  PetscScalar *value_bc,*x_bc;
  BCType      *type_bc;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  
  for (ii=0; ii<usr->par->ncomp; ii++) {
    // Down:
    ierr = DMStagBCListGetValues(bclist,'s','o',ii,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
    for (k=0; k<n_bc; k++) type_bc[k] = BC_PERIODIC;
    ierr = DMStagBCListInsertValues(bclist,'o',ii,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

    // Top:
    ierr = DMStagBCListGetValues(bclist,'n','o',ii,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
    for (k=0; k<n_bc; k++) type_bc[k] = BC_PERIODIC;
    ierr = DMStagBCListInsertValues(bclist,'o',ii,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

    // Left:
    ierr = DMStagBCListGetValues(bclist,'w','o',ii,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
    for (k=0; k<n_bc; k++) type_bc[k] = BC_PERIODIC;
    ierr = DMStagBCListInsertValues(bclist,'o',ii,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

    // Right:
    ierr = DMStagBCListGetValues(bclist,'e','o',ii,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
    for (k=0; k<n_bc; k++) type_bc[k] = BC_PERIODIC;
    ierr = DMStagBCListInsertValues(bclist,'o',ii,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  }

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
  PetscScalar    ***c;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  // Get domain corners
  ierr = DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dmcoeff, &coefflocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dmcoeff, coefflocal, &c); CHKERRQ(ierr);

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {

      { // ELEMENT
        DMStagStencil point;
        PetscInt      idx;

        point.i = i; point.j = j; point.loc = ELEMENT;  
        point.c = COEFF_A1; ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = 1.0;

        point.c = COEFF_B1; ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = 0.0;

        point.c = COEFF_D1; ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = 0.0;

        point.c = COEFF_A2; ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = 0.0;

        point.c = COEFF_B2; ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = 1.0;

        point.c = COEFF_D2; ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = 0.0;
      }

      { // FACES
        PetscInt      ii, idx, v[4];
        DMStagStencil point[4];
        v[0] = usr->par->ux;
        v[1] = usr->par->ux;
        v[2] = usr->par->uz;
        v[3] = usr->par->uz;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT; 
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN; 
        point[3].i = i; point[3].j = j; point[3].loc = UP; 

        for (ii = 0; ii < 4; ii++) {
          point[ii].c = COEFF_C1; ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = 0.0;

          point[ii].c = COEFF_C2; ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = 0.0; 

          point[ii].c = COEFF_v; ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = v[ii];

          point[ii].c = COEFF_vf; ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = v[ii];

          point[ii].c = COEFF_vs; ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = v[ii];
        }
      }
    }
  }

  // Restore arrays, local vectors
  ierr = DMStagVecRestoreArray(dmcoeff,coefflocal,&c);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  
  ierr = VecDestroy(&coefflocal); CHKERRQ(ierr);

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
  ierr = PetscBagRegisterInt(bag, &par->tstep, 1, "tstep", "Number of time steps"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->tout,5,"tout", "Output every <tout> time steps"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->ncomp,2,"ncomp", "Number of petrological components"); CHKERRQ(ierr);

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

  // Gaussian-shape initial guess
  ierr = PetscBagRegisterScalar(bag, &par->A, 10.0, "A", "Amplitude gaussian"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->x0,5.0, "x0", "Offset x-dir"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->z0,5.0, "z0", "Offset z-dir"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->taox,1.0, "taox", "tao parameter x-dir"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->taoz,1.0, "taoz", "tao parameter z-dir"); CHKERRQ(ierr);

  // Input/output 
  ierr = PetscBagRegisterString(bag,&par->fname_out,FNAME_LENGTH,"out_num_enth_periodic","output_file","Name for output file, set with: -output_file <filename>"); CHKERRQ(ierr);
  ierr = PetscBagRegisterString(bag,&par->fdir_out,FNAME_LENGTH,"./","output_dir","Name for output directory, set with: -output_dir <dirname>"); CHKERRQ(ierr);

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
  ierr = PetscOptionsGetAll(NULL, &opts); CHKERRQ(ierr);

  // Print header and petsc options
  PetscPrintf(usr->comm,"# --------------------------------------- #\n");
  PetscPrintf(usr->comm,"# 2-D Diffusion (Enthalpy): %s \n",&(date[0]));
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

  // Numerical solution using the FD pde object
  ierr = PetscTime(&start_time); CHKERRQ(ierr);
  ierr = Numerical_solution(usr); CHKERRQ(ierr);
  ierr = PetscTime(&end_time); CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"# Runtime: %g (sec) \n", end_time - start_time);
  PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");

  // Destroy objects
  ierr = PetscBagDestroy(&usr->bag); CHKERRQ(ierr);
  ierr = PetscFree(usr); CHKERRQ(ierr);

  // Finalize main
  ierr = PetscFinalize();
  return ierr;
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
  PetscInt       i,j,ii, sx, sz, nx, nz, icenter;
  PetscScalar    A, x0, taox,z0, taoz;
  PetscScalar    ***xg, **coordx, **coordz;

  PetscErrorCode ierr;
  PetscFunctionBegin;

  // Gaussian function parameters
  A    = usr->par->A;
  x0   = usr->par->x0;
  taox = usr->par->taox;
  z0   = usr->par->z0;
  taoz = usr->par->taoz;

  // Get domain corners
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  // Get dm coordinates array
  ierr = DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter);CHKERRQ(ierr);

  // Create coefficient local vector
  ierr = DMCreateLocalVector(dm, &xglocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm, xglocal, &xg); CHKERRQ(ierr);

  // Loop over local domain - set initial density and viscosity
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      DMStagStencil point;
      PetscScalar   xp,zp, fval = 0.0;
      PetscInt      idx;

      xp = coordx[i][icenter]; 
      zp = coordz[j][icenter]; 
      fval = A*PetscExpReal(-( (xp-x0)*(xp-x0)/taox/taox + (zp-z0)*(zp-z0)/taoz/taoz )); 

      for (ii = 0; ii <usr->par->ncomp; ii++) {
        point.i = i; point.j = j; point.loc = ELEMENT; point.c = ii;
        ierr = DMStagGetLocationSlot(dm, point.loc, point.c, &idx); CHKERRQ(ierr);
        xg[j][i][idx] = fval;
      }
    }
  }

  // Restore arrays, local vectors
  ierr = DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);

  ierr = DMStagVecRestoreArray(dm,xglocal,&xg);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm,xglocal,INSERT_VALUES,xguess); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,xglocal,INSERT_VALUES,xguess); CHKERRQ(ierr);
  
  ierr = VecDestroy(&xglocal); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}