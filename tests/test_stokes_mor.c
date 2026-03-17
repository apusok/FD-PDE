// ---------------------------------------
// Corner flow (mid-ocean ridges) benchmark
// run: ./test_stokes_mor_ -pc_type lu -pc_factor_mat_solver_type umfpack -pc_factor_mat_ordering_type external -nx 10 -nz 10
// python test: ./python/test_stokes_mor.py
// ---------------------------------------
static char help[] = "Application to solve the 2D corner flow (mid-ocean ridges) benchmark with FD-PDE \n\n";

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

#include "../src/benchmark_cornerflow.h"
#include "../src/fdpde_stokes.h"

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
PetscErrorCode SNESStokes_MOR(DM*,Vec*,void*);
PetscErrorCode Analytic_MOR(DM,Vec*,void*);
PetscErrorCode InputParameters(UsrData**);
PetscErrorCode InputPrintData(UsrData*);
PetscErrorCode FormCoefficient(FDPDE, DM, Vec, DM, Vec, void*);
PetscErrorCode FormBCList_MOR(DM, Vec, DMStagBCList, void*);

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
  char           fout[FNAME_LENGTH];
  PetscFunctionBeginUser;

  // Element count
  nx = usr->par->nx;
  nz = usr->par->nz;

  // Domain coords
  xmin = usr->par->xmin;
  zmin = usr->par->zmin;
  xmax = usr->par->xmin+usr->par->L;
  zmax = usr->par->zmin+usr->par->H;

  // Create the FD-pde object
  PetscCall(FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_STOKES,&fd));
  PetscCall(FDPDESetUp(fd));

  // Set BC evaluation function
  PetscCall(FDPDESetFunctionBCList(fd,FormBCList_MOR,bc_description,usr)); 

  // Set coefficients evaluation function
  PetscCall(FDPDESetFunctionCoefficient(fd,FormCoefficient,coeff_description,usr)); 
  PetscCall(FDPDEView(fd)); 

  PetscCall(PetscOptionsSetValue(NULL, "-snes_monitor",         "")); 
  PetscCall(PetscOptionsSetValue(NULL, "-ksp_monitor",          "")); 
  PetscCall(PetscOptionsSetValue(NULL, "-snes_converged_reason","")); 
  PetscCall(PetscOptionsSetValue(NULL, "-ksp_converged_reason", "")); 

  // FD SNES Solver
  PetscCall(FDPDESolve(fd,NULL));

  // Get solution vector
  PetscCall(FDPDEGetSolution(fd,&x)); 
  PetscCall(FDPDEGetDM(fd, &dmPV)); 

  // Output solution to file
  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s",usr->par->fdir_out,usr->par->fname_out));
  PetscCall(DMStagViewBinaryPython(dmPV,x,fout));

  // Destroy FD-PDE object
  PetscCall(FDPDEDestroy(&fd));

  *_x  = x;
  *_dm = dmPV;

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

  PetscCall(PetscBagRegisterScalar(bag, &par->xmin, 0.0, "xmin", "Start coordinate of domain in x-dir")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->zmin,-1.0, "zmin", "Start coordinate of domain in z-dir")); 

  PetscCall(PetscBagRegisterScalar(bag, &par->L, 1.0, "L", "Length of domain in x-dir")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->H, 1.0, "H", "Height of domain in z-dir")); 

  // Physical and material parameters
  PetscCall(PetscBagRegisterScalar(bag, &par->g, 1.0, "g", "Gravitational acceleration")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->eta0, 1.0, "eta0", "Reference viscosity")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->rho0, 0.0, "rho0", "Reference density")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->u0, 1.0, "u0", "Half-spreading rate [cm/yr]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->rangle, 0.0, "rangle", "Ridge angle [deg]")); 

  if (par->rangle>0.0) SETERRQ(usr->comm,PETSC_ERR_SUP,"Internal boundary conditions not implemented for rangle>0!");

  // Input/output 
  PetscCall(PetscBagRegisterString(bag,&par->fname_out,FNAME_LENGTH,"out_num_solution_mor","output_file","Name for output file, set with: -output_file <filename>")); 
  PetscCall(PetscBagRegisterString(bag,&par->fdir_out,FNAME_LENGTH,"./","output_dir","Name for output directory, set with: -output_dir <dirname>")); 

  // Other variables
  par->fname_in[0] = '\0';

  par->radalpha = par->rangle*PETSC_PI/180;
  par->sina = PetscSinScalar(par->radalpha);
  par->C1 = 2*par->sina*par->sina/(PETSC_PI-2*par->radalpha-PetscSinScalar(2*par->radalpha));
  par->C4 = -2/(PETSC_PI-2*par->radalpha-PetscSinScalar(2*par->radalpha));

    
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
  PetscCall(PetscPrintf(usr->comm,"# --------------------------------------- #\n"));
  PetscCall(PetscPrintf(usr->comm,"# Test_stokes_mor (Corner flow): %s \n",&(date[0])));
  PetscCall(PetscPrintf(usr->comm,"# --------------------------------------- #\n"));
  PetscCall(PetscPrintf(usr->comm,"# PETSc options: %s \n",opts));
  PetscCall(PetscPrintf(usr->comm,"# --------------------------------------- #\n"));

  // Input file info
  if (usr->par->fname_in[0] == '\0') { // string is empty
    PetscCall(PetscPrintf(usr->comm,"# Input options file: NONE \n"));
  }
  else {
    PetscCall(PetscPrintf(usr->comm,"# Input options file: %s \n",usr->par->fname_in));
  }
  PetscCall(PetscPrintf(usr->comm,"# --------------------------------------- #\n"));

  // Print usr bag
  PetscCall(PetscBagView(usr->bag,PETSC_VIEWER_STDOUT_WORLD)); 
  PetscCall(PetscPrintf(usr->comm,"# --------------------------------------- #\n"));

  // Free memory
  PetscCall(PetscFree(opts)); 

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
  PetscScalar    ***c, g, rho;
  PetscFunctionBeginUser;

  // User parameters
  g   = -usr->par->g;
  rho = usr->par->rho0;

  // Get domain corners
  PetscCall(DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 

  // Create coefficient local vector
  PetscCall(DMCreateLocalVector(dmcoeff, &coefflocal)); 
  PetscCall(DMStagVecGetArray(dmcoeff, coefflocal, &c)); 
  
  // Loop over local domain - set initial density and viscosity
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {

      { // fux = 0.0
        DMStagStencil point[2];
        PetscInt      ii, idx;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 0;

        for (ii = 0; ii < 2; ii++) {
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] = 0.0;
        }
      }

      { // fuz = rho*g = 0 
        DMStagStencil point[2];
        PetscScalar   fval = 0.0;
        PetscInt      ii, idx;

        point[0].i = i; point[0].j = j; point[0].loc = DOWN; point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = UP;   point[1].c = 0;

        for (ii = 0; ii < 2; ii++) {
          fval = g*rho; 
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] = fval;
        }
      }

      { // fp = 0.0
        DMStagStencil point;
        PetscInt      idx;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 0;
        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
        c[j][i][idx] = 0.0;
      }

      { // eta_c = eta0
        DMStagStencil point;
        PetscScalar   fval = 0.0;
        PetscInt      idx;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 1;

        fval = usr->par->eta0;
        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
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

        for (ii = 0; ii < 4; ii++) {
          fval = usr->par->eta0;
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] = fval;
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
  PetscFunctionBeginUser;

  C1 = usr->par->C1;
  C4 = usr->par->C4;
  u0 = usr->par->u0;
  eta0 = usr->par->eta0;

  // NOTES: the first and last points for each interior boundary are actually resting ON the two adjacent boundaries, so loop from 1 to n_bc-2 instead of from 0 to n_bc-1 for interior boundaries.
  
  // LEFT Boundary - Vx
  PetscCall(DMStagBCListGetValues(bclist,'w','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    evaluate_CornerFlow_MOR(C1, C4, u0, eta0, x_bc[2*k], x_bc[2*k+1], v, &p);
    value_bc[k] = v[0];
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // LEFT Boundary - Vz
  PetscCall(DMStagBCListGetValues(bclist,'w','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=1; k<n_bc-1; k++) {
    evaluate_CornerFlow_MOR(C1, C4, u0, eta0, x_bc[2*k], x_bc[2*k+1], v, &p);
    value_bc[k] = v[1];
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // RIGHT Boundary - Vx
  PetscCall(DMStagBCListGetValues(bclist,'e','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    evaluate_CornerFlow_MOR(C1, C4, u0, eta0, x_bc[2*k], x_bc[2*k+1], v, &p);
    value_bc[k] = v[0];
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // RIGHT Boundary - Vz
  PetscCall(DMStagBCListGetValues(bclist,'e','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=1; k<n_bc-1; k++) {
    evaluate_CornerFlow_MOR(C1, C4, u0, eta0, x_bc[2*k], x_bc[2*k+1], v, &p);
    value_bc[k] = v[1];
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // DOWN Boundary - Vx
  PetscCall(DMStagBCListGetValues(bclist,'s','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=1; k<n_bc-1; k++) {
    evaluate_CornerFlow_MOR(C1, C4, u0, eta0, x_bc[2*k], x_bc[2*k+1], v, &p);
    value_bc[k] = v[0];
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // DOWN Boundary - Vz
  PetscCall(DMStagBCListGetValues(bclist,'s','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    evaluate_CornerFlow_MOR(C1, C4, u0, eta0, x_bc[2*k], x_bc[2*k+1], v, &p);
    value_bc[k] = v[1];
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // UP Boundary - Vx
  PetscCall(DMStagBCListGetValues(bclist,'n','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=1; k<n_bc-1; k++) {
    evaluate_CornerFlow_MOR(C1, C4, u0, eta0, x_bc[2*k], x_bc[2*k+1], v, &p);
    value_bc[k] = v[0];
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // UP Boundary - Vz
  PetscCall(DMStagBCListGetValues(bclist,'n','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    evaluate_CornerFlow_MOR(C1, C4, u0, eta0, x_bc[2*k], x_bc[2*k+1], v, &p);
    value_bc[k] = v[1];
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
    
  // Pin reference values for pressure along the entire bottom boundary
  PetscCall(DMStagBCListGetValues(bclist,'s','o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    evaluate_CornerFlow_MOR(C1, C4, u0, eta0, x_bc[2*k], x_bc[2*k+1], v, &p);
    value_bc[k] = p;
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  
  PetscFunctionReturn(PETSC_SUCCESS);
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
  char           fout[FNAME_LENGTH];
  PetscFunctionBeginUser;

  // Get parameters
  C1 = usr->par->C1;
  C4 = usr->par->C4;
  u0 = usr->par->u0;
  eta0 = usr->par->eta0;

  // Create local and global vector associated with DM
  PetscCall(DMCreateGlobalVector(dm, &x     )); 
  PetscCall(DMCreateLocalVector (dm, &xlocal)); 

  // Get array associated with vector
  PetscCall(DMStagVecGetArray(dm,xlocal,&xx)); 

  // Get domain corners
  PetscCall(DMStagGetGlobalSizes(dm, &Nx, &Nz,NULL));
  PetscCall(DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 
  
// Get dm coordinates array
  PetscCall(DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter)); 
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm,LEFT,&iprev));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm,RIGHT,&inext)); 

  // Loop over local domain to calculate the SolCx analytical solution
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      PetscScalar xp, zp, v[2], p;

      // Vx
      xp = coordx[i][iprev ]; 
      zp = coordz[j][icenter];
      evaluate_CornerFlow_MOR(C1, C4, u0, eta0, xp, zp, v, &p);
      PetscCall(DMStagGetLocationSlot(dm, LEFT, 0, &idx)); 
      xx[j][i][idx] = v[0];

      if (i == Nx-1) {
        xp = coordx[i][inext  ];
        zp = coordz[j][icenter];
        evaluate_CornerFlow_MOR(C1, C4, u0, eta0, xp, zp, v, &p);
        PetscCall(DMStagGetLocationSlot(dm, RIGHT, 0, &idx)); 
        xx[j][i][idx] = v[0];
      }

      // Vz
      xp = coordx[i][icenter];
      zp = coordz[j][iprev  ];
      evaluate_CornerFlow_MOR(C1, C4, u0, eta0, xp, zp, v, &p);
      PetscCall(DMStagGetLocationSlot(dm, DOWN, 0, &idx)); 
      xx[j][i][idx] = v[1];

      if (j == Nz-1) {
        xp = coordx[i][icenter];
        zp = coordz[j][inext  ];
        evaluate_CornerFlow_MOR(C1, C4, u0, eta0, xp, zp, v, &p);
        PetscCall(DMStagGetLocationSlot(dm, UP, 0, &idx)); 
        xx[j][i][idx] = v[1];
      }
    
      // P
      xp = coordx[i][icenter];
      zp = coordz[j][icenter];
      evaluate_CornerFlow_MOR(C1, C4, u0, eta0, xp, zp, v, &p);
      PetscCall(DMStagGetLocationSlot(dm, ELEMENT, 0, &idx)); 
      xx[j][i][idx] = p;
    }
  }

  // Restore arrays
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));
  PetscCall(DMStagVecRestoreArray(dm,xlocal,&xx)); 

  // Map local to global
  PetscCall(DMLocalToGlobalBegin(dm,xlocal,INSERT_VALUES,x)); 
  PetscCall(DMLocalToGlobalEnd  (dm,xlocal,INSERT_VALUES,x)); 

  PetscCall(VecDestroy(&xlocal)); 

  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s",usr->par->fdir_out,"out_analytic_solution_mor"));
  PetscCall(DMStagViewBinaryPython(dm,x,fout));

  // Assign pointers
  *_x  = x;
  
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
  DM              dmStokes;
  Vec             xStokes,xAnalytic;
  PetscLogDouble  start_time, end_time;
    
  // Initialize application
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));

  // Start time
  PetscCall(PetscTime(&start_time)); 
 
  // Load command line or input file if required
  PetscCall(PetscOptionsInsert(PETSC_NULLPTR,&argc,&argv,NULL)); 

  // Input user parameters and print
  PetscCall(InputParameters(&usr)); 

  // Save input options filename
  for (int i = 1; i < argc; i++) {
    PetscBool flg;
    
    PetscCall(PetscStrcmp(argv[i],"-options_file",&flg)); 
    if (flg) { PetscCall(PetscStrcpy(usr->par->fname_in, argv[i+1]));  }
  }

  // Print user parameters
  PetscCall(InputPrintData(usr)); 

  //Numerical solution using the FD pde object
  PetscCall(SNESStokes_MOR(&dmStokes, &xStokes, usr)); 

  // Analytical solution
  PetscCall(Analytic_MOR(dmStokes, &xAnalytic, usr)); 

  // Destroy objects
  PetscCall(DMDestroy(&dmStokes)); 
  PetscCall(VecDestroy(&xStokes)); 
  PetscCall(VecDestroy(&xAnalytic)); 

  PetscCall(PetscBagDestroy(&usr->bag)); 
  PetscCall(PetscFree(usr));

  // End time
  PetscCall(PetscTime(&end_time)); 
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"# Total runtime: %g (sec) \n", end_time - start_time));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n"));
  
  // Finalize main
  PetscCall(PetscFinalize());
  return 0;
}
