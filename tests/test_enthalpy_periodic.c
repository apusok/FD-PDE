// ---------------------------------------
// (ENTHALPY) Pure advection and time-stepping test - PERIODIC BCs
// run: ./test_enthalpy_periodic_ -pc_type lu -pc_factor_mat_solver_type umfpack -pc_factor_mat_ordering_type external -nx 10 -nz 10 -log_view
// python test: ./python/test_enthalpy_periodic.py
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

#include "../src/fdpde_enthalpy.h"

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
  PetscFunctionBeginUser;

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
  PetscCall(FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_ENTHALPY,&fd));
  PetscCall(FDPDEEnthalpySetNumberComponentsPhaseDiagram(fd,par->ncomp));
  PetscCall(FDPDESetDMBoundaryType(fd,DM_BOUNDARY_PERIODIC,DM_BOUNDARY_PERIODIC));
  PetscCall(FDPDESetUp(fd));

  PetscCall(FDPDESetFunctionBCList(fd,FormBCList,bc_description,usr)); 
  PetscCall(FDPDESetFunctionCoefficient(fd,FormCoefficient,coeff_description,usr)); 
  PetscCall(FDPDEEnthalpySetAdvectSchemeType(fd,ADV_FROMM));
  PetscCall(FDPDEEnthalpySetTimeStepSchemeType(fd,TS_CRANK_NICHOLSON ));

  PetscCall(FDPDEEnthalpySetTimestep(fd,par->dt)); 
  PetscCall(FDPDEEnthalpySetEnthalpyMethod(fd,Form_Enthalpy,enthalpy_method_description,usr));
  PetscCall(FDPDEEnthalpySetPotentialTemp(fd,Form_PotentialTemperature,usr));
  PetscCall(FDPDEView(fd));

  // Set initial condition 
  PetscCall(FDPDEGetDM(fd,&dm));
  PetscCall(FDPDEEnthalpyGetPrevSolution(fd,&xprev));
  PetscCall(SetGaussianInitialGuess(dm,xprev,usr));
  PetscCall(FDPDEGetSolutionGuess(fd,&xguess));
  PetscCall(VecCopy(xprev,xguess));
  PetscCall(VecDestroy(&xguess));

  // Set initial coefficient structure
  PetscCall(FDPDEGetCoefficient(fd,&dmcoeff,NULL));
  PetscCall(FDPDEEnthalpyGetPrevCoefficient(fd,&xcoeffprev));
  PetscCall(FormCoefficient(fd,dm,xprev,dmcoeff,xcoeffprev,usr));
  PetscCall(VecDestroy(&xcoeffprev));
  PetscCall(VecDestroy(&xprev));

  // Set initial pressure - not needed so initialize with constant
  PetscCall(FDPDEEnthalpyGetPressure(fd,&dmP,NULL));
  PetscCall(FDPDEEnthalpyGetPrevPressure(fd,&xPprev));
  PetscCall(VecSet(xPprev,-1.0));
  PetscCall(VecDestroy(&xPprev));

  // Time loop
  while (istep<tstep) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"# Timestep %d out of %d: time %1.3f\n",istep,tstep,usr->par->t));

    // update pressure
    PetscCall(FDPDEEnthalpyGetPressure(fd,NULL,&xP));
    PetscCall(VecSet(xP,istep));

    // Enthalpy Solver
    PetscCall(FDPDESolve(fd,NULL));
    PetscCall(FDPDEGetSolution(fd,&x));

    // increment time
    PetscCall(FDPDEEnthalpyGetTimestep(fd,&dt));
    usr->par->t += dt;

    // Copy solution and coefficient to old
    PetscCall(FDPDEEnthalpyGetPrevSolution(fd,&xprev));
    PetscCall(VecCopy(x,xprev));
    PetscCall(VecDestroy(&xprev));

    PetscCall(FDPDEGetCoefficient(fd,&dmcoeff,&xcoeff));
    PetscCall(FDPDEEnthalpyGetPrevCoefficient(fd,&xcoeffprev));
    PetscCall(VecCopy(xcoeff,xcoeffprev));
    PetscCall(VecDestroy(&xcoeffprev));

    PetscCall(FDPDEEnthalpyGetPrevPressure(fd,&xPprev));
    PetscCall(VecCopy(xP,xPprev));
    PetscCall(VecDestroy(&xP));
    PetscCall(VecDestroy(&xPprev));

    // Output solution
    if (istep % par->tout == 0 ) {
      PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s_ts%1.3d",par->fdir_out,par->fname_out,istep));
      PetscCall(DMStagViewBinaryPython(dm,x,fout));
    }
    PetscCall(VecDestroy(&x));

    // increment timestep
    istep++;
  }

  // Destroy objects
  PetscCall(DMDestroy(&dm));
  PetscCall(DMDestroy(&dmP));
  PetscCall(FDPDEDestroy(&fd));
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscFunctionBeginUser;

  TP = T*1.0;
  *_TP = TP;

  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscFunctionBeginUser;
  
  for (ii=0; ii<usr->par->ncomp; ii++) {
    // Down:
    PetscCall(DMStagBCListGetValues(bclist,'s','o',ii,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
    for (k=0; k<n_bc; k++) type_bc[k] = BC_PERIODIC;
    PetscCall(DMStagBCListInsertValues(bclist,'o',ii,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

    // Top:
    PetscCall(DMStagBCListGetValues(bclist,'n','o',ii,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
    for (k=0; k<n_bc; k++) type_bc[k] = BC_PERIODIC;
    PetscCall(DMStagBCListInsertValues(bclist,'o',ii,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

    // Left:
    PetscCall(DMStagBCListGetValues(bclist,'w','o',ii,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
    for (k=0; k<n_bc; k++) type_bc[k] = BC_PERIODIC;
    PetscCall(DMStagBCListInsertValues(bclist,'o',ii,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

    // Right:
    PetscCall(DMStagBCListGetValues(bclist,'e','o',ii,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
    for (k=0; k<n_bc; k++) type_bc[k] = BC_PERIODIC;
    PetscCall(DMStagBCListInsertValues(bclist,'o',ii,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  }

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
  PetscScalar    ***c;
  PetscFunctionBeginUser;

  // Get domain corners
  PetscCall(DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 
  PetscCall(DMCreateLocalVector(dmcoeff, &coefflocal)); 
  PetscCall(DMStagVecGetArray(dmcoeff, coefflocal, &c)); 

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {

      { // ELEMENT
        DMStagStencil point;
        PetscInt      idx;

        point.i = i; point.j = j; point.loc = ELEMENT;  
        point.c = COEFF_A1; PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
        c[j][i][idx] = 1.0;

        point.c = COEFF_B1; PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
        c[j][i][idx] = 0.0;

        point.c = COEFF_D1; PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
        c[j][i][idx] = 0.0;

        point.c = COEFF_A2; PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
        c[j][i][idx] = 0.0;

        point.c = COEFF_B2; PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
        c[j][i][idx] = 1.0;

        point.c = COEFF_D2; PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
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
          point[ii].c = COEFF_C1; PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] = 0.0;

          point[ii].c = COEFF_C2; PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] = 0.0; 

          point[ii].c = COEFF_v; PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] = v[ii];

          point[ii].c = COEFF_vf; PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] = v[ii];

          point[ii].c = COEFF_vs; PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] = v[ii];
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
  PetscCall(PetscBagRegisterInt(bag, &par->ncomp,2,"ncomp", "Number of petrological components")); 

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
  PetscCall(PetscBagRegisterString(bag,&par->fname_out,FNAME_LENGTH,"out_num_enth_periodic","output_file","Name for output file, set with: -output_file <filename>")); 
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
  PetscCall(PetscOptionsGetAll(NULL, &opts)); 

  // Print header and petsc options
  PetscCall(PetscPrintf(usr->comm,"# --------------------------------------- #\n"));
  PetscCall(PetscPrintf(usr->comm,"# 2-D Diffusion (Enthalpy): %s \n",&(date[0])));
  PetscCall(PetscPrintf(usr->comm,"# --------------------------------------- #\n"));
  PetscCall(PetscPrintf(usr->comm,"# PETSc options: %s \n",opts));
  PetscCall(PetscPrintf(usr->comm,"# --------------------------------------- #\n"));

  // Print usr bag
  PetscCall(PetscBagView(usr->bag,PETSC_VIEWER_STDOUT_WORLD)); 
  PetscCall(PetscPrintf(usr->comm,"# --------------------------------------- #\n"));

  // Free memory
  PetscCall(PetscFree(opts)); 

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
  PetscCall(InputPrintData(usr)); 

  // Numerical solution using the FD pde object
  PetscCall(PetscTime(&start_time)); 
  PetscCall(Numerical_solution(usr)); 
  PetscCall(PetscTime(&end_time)); 
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"# Runtime: %g (sec) \n", end_time - start_time));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n"));

  // Destroy objects
  PetscCall(PetscBagDestroy(&usr->bag)); 
  PetscCall(PetscFree(usr)); 

  // Finalize main
  PetscCall(PetscFinalize());
  return 0;
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

      xp = coordx[i][icenter]; 
      zp = coordz[j][icenter]; 
      fval = A*PetscExpReal(-( (xp-x0)*(xp-x0)/taox/taox + (zp-z0)*(zp-z0)/taoz/taoz )); 

      for (ii = 0; ii <usr->par->ncomp; ii++) {
        point.i = i; point.j = j; point.loc = ELEMENT; point.c = ii;
        PetscCall(DMStagGetLocationSlot(dm, point.loc, point.c, &idx)); 
        xg[j][i][idx] = fval;
      }
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