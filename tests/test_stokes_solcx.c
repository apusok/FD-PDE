// ---------------------------------------
// SOLCX benchmark - constant grid spacing
// run: ./test_stokes_solcx_ -pc_type lu -pc_factor_mat_solver_type umfpack -pc_factor_mat_ordering_type external -nx 10 -nz 10
// python test: ./python/test_stokes_solcx.py
// ---------------------------------------
static char help[] = "Application to solve the SolCx benchmark with FD-PDE \n\n";

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

#include "../src/benchmark_solcx.h"
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
  PetscScalar    eta0, eta1, g;
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
PetscErrorCode SNESStokes_Solcx(DM*,Vec*,void*);
PetscErrorCode Analytic_Solcx(DM,Vec*,void*);
PetscErrorCode InputParameters(UsrData**);
PetscErrorCode InputPrintData(UsrData*);
PetscErrorCode FormCoefficient(FDPDE, DM, Vec, DM, Vec, void*);
PetscErrorCode FormBCList(DM, Vec, DMStagBCList, void*);
PetscErrorCode ComputeErrorNorms(DM,Vec,Vec,void*);

// ---------------------------------------
// Some descriptions
// ---------------------------------------
const char coeff_description[] =
"  << Stokes Coefficients >> \n"
"  eta_n/eta_c = eta0 if x<0.5, eta1 if x>=0.5\n"
"  fux = 0 \n" 
"  fuz = rho*g \n" 
"  fp = 0 (incompressible)\n";

const char bc_description[] =
"  << Stokes BCs >> \n"
"  LEFT: Vx = 0, dVz/dx = 0\n"
"  RIGHT: Vx = 0, dVz/dx = 0\n" 
"  DOWN: Vz = 0, dVx/dz = 0\n" 
"  UP: Vz = 0, dVx/dz = 0 \n";

// ---------------------------------------
// Application functions
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "SNESStokes_SolCx"
PetscErrorCode SNESStokes_Solcx(DM *_dm, Vec *_x, void *ctx)
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
  PetscCall(FDPDESetLinearPreallocatorStencil(fd,PETSC_TRUE));
  PetscCall(FDPDESetUp(fd));
  // User can modify the dm coordinates anywhere between FDPDESetUp() and FDPDESolve()

  // Set BC evaluation function
  PetscCall(FDPDESetFunctionBCList(fd,FormBCList,bc_description,NULL)); 

  // Set coefficients evaluation function
  PetscCall(FDPDESetFunctionCoefficient(fd,FormCoefficient,coeff_description,usr)); 
  PetscCall(FDPDEView(fd)); 

  // Some SNES options
  PetscInt  maxit, maxf, its;
  PetscReal atol, rtol, stol;
  SNES      snes;

  PetscCall(FDPDEGetSNES(fd,&snes));

  // SNES Options - default info on convergence
  PetscCall(PetscOptionsSetValue(NULL, "-snes_monitor",         "")); 
  PetscCall(PetscOptionsSetValue(NULL, "-ksp_monitor",          "")); 
  PetscCall(PetscOptionsSetValue(NULL, "-snes_converged_reason","")); 
  PetscCall(PetscOptionsSetValue(NULL, "-ksp_converged_reason", "")); 
  //PetscCall(SNESSetOptionsPrefix(snes,"stk_"));

  // FD SNES Solver
  PetscCall(FDPDESolve(fd,NULL));

  // MatView(fd->J,PETSC_VIEWER_STDOUT_WORLD);

  PetscCall(SNESGetIterationNumber(fd->snes,&its));    
  PetscCall(SNESGetTolerances(fd->snes, &atol, &rtol, &stol, &maxit, &maxf)); 
  
  // Print some SNES diagnostics
  PetscCall(PetscPrintf(fd->comm,"Number of SNES iterations = %d\n",its));
  PetscCall(PetscPrintf(fd->comm,"SNES: atol = %g, rtol = %g, stol = %g, maxit = %D, maxf = %D\n",(double)atol,(double)rtol,(double)stol,maxit,maxf)); 

  // Get solution vector
  PetscCall(FDPDEGetSolution(fd,&x)); 
  PetscCall(FDPDEGetDM(fd, &dmPV)); 

  // Output solution to file
  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s",usr->par->fdir_out,usr->par->fname_out));
  PetscCall(DMStagViewBinaryPython(dmPV,x,fout));
  {
    DM dmcoeff;
    Vec coeff;
    PetscCall(FDPDEGetCoefficient(fd,&dmcoeff,&coeff));
    PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s",usr->par->fdir_out,"out_coefficients"));
    PetscCall(DMStagViewBinaryPython(dmcoeff,coeff,fout));
  }

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
  PetscCall(PetscBagRegisterScalar(bag, &par->zmin, 0.0, "zmin", "Start coordinate of domain in z-dir")); 

  PetscCall(PetscBagRegisterScalar(bag, &par->L, 1.0, "L", "Length of domain in x-dir")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->H, 1.0, "H", "Height of domain in z-dir")); 

  // Physical and material parameters
  PetscCall(PetscBagRegisterScalar(bag, &par->g, 1.0, "g", "Gravitational acceleration")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->eta0, 1.0, "eta0", "Viscosity eta0")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->eta1, 1.0, "eta1", "Viscosity eta1")); 

  // Input/output 
  PetscCall(PetscBagRegisterString(bag,&par->fname_out,FNAME_LENGTH,"out_num_solution","output_file","Name for output file, set with: -output_file <filename>")); 
  PetscCall(PetscBagRegisterString(bag,&par->fdir_out,FNAME_LENGTH,"./","output_dir","Name for output directory, set with: -output_dir <dirname>")); 

  // Other variables
  par->fname_in[0] = '\0';

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
  PetscPrintf(usr->comm,"# Test_stokes_solcx: %s \n",&(date[0]));
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
  PetscScalar    g, L2 = 0.5;
  PetscFunctionBeginUser;

  // Density is defined (edges): sin(pi*z)*cos(pi*x); 
  // Viscosity is defined (corner and center): solcx_eta0, for 0<=x<=0.5, solcx_eta1, for 0.5<x<=1
  // DM dm, Vec x - used for non-linear coefficients

  // User parameters
  g = -usr->par->g;

  // Get domain corners
  PetscCall(DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 

  // Get dm coordinates array
  PetscCall(DMStagGetProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmcoeff,LEFT,&iprev));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmcoeff,RIGHT,&inext)); 
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmcoeff,ELEMENT,&icenter));

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

      { // eta_c = eta0:eta1
        DMStagStencil point;
        PetscScalar   xp, fval = 0.0;
        PetscInt      idx;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 1;
        xp = coordx[i][icenter];

        if (xp <= L2) fval = usr->par->eta0;
        else          fval = usr->par->eta1;

        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
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

          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] = fval;
        }
      }
    }
  }

  // Restore arrays, local vectors
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL));

  PetscCall(DMStagVecRestoreArray(dmcoeff,coefflocal,&c));
  PetscCall(DMLocalToGlobalBegin(dmcoeff,coefflocal,INSERT_VALUES,coeff)); 
  PetscCall(DMLocalToGlobalEnd  (dmcoeff,coefflocal,INSERT_VALUES,coeff)); 
  
  PetscCall(VecDestroy(&coefflocal)); 

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// FormBCList
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormBCList"
PetscErrorCode FormBCList(DM dm, Vec x, DMStagBCList bclist, void *ctx)
{
  PetscInt    sx, sz, nx, nz;
  PetscInt    k,n_bc,*idx_bc;
  PetscScalar *value_bc,*x_bc;
  BCType      *type_bc;
  PetscFunctionBeginUser;

  PetscCall(DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 

  // dVz/dx=0 on left boundary (w)
  PetscCall(DMStagBCListGetValues(bclist,'w','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // dVz/dx=0 on right boundary (e)
  PetscCall(DMStagBCListGetValues(bclist,'e','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // dVx/dz=0 on top boundary (n)
  PetscCall(DMStagBCListGetValues(bclist,'n','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // dVx/dz=0 on bottom boundary (s)
  PetscCall(DMStagBCListGetValues(bclist,'s','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // Vx=0 on left boundary (w)
  PetscCall(DMStagBCListGetValues(bclist,'w','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  
  // Vx=0 on right boundary (e)
  PetscCall(DMStagBCListGetValues(bclist,'e','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  
  // Vz=0 on top boundary (n)
  PetscCall(DMStagBCListGetValues(bclist,'n','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  
  // Vz=0 on bottom boundary (s)
  PetscCall(DMStagBCListGetValues(bclist,'s','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // Pin a reference value for pressure in the cell at the bottom left corner
  // Warning: need to ensure valid boundary on processor
  if ((sx==0) && (sz==0)) {
    PetscCall(DMStagBCListGetValues(bclist,'s','o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
    value_bc[0] = 0.0;
    type_bc[0] = BC_DIRICHLET_STAG;
    PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// CreateSolCx analytical solution
// ---------------------------------------
PetscErrorCode Analytic_Solcx(DM dm,Vec *_x, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz, idx;
  PetscInt       iprev, inext, icenter;
  PetscScalar    eta0, eta1, xc;
  PetscScalar    ***xx;
  PetscScalar    **coordx,**coordz;
  Vec            x, xlocal;
  char           fout[FNAME_LENGTH];
  PetscFunctionBeginUser;

  // Get parameters
  eta0 = usr->par->eta0;
  eta1 = usr->par->eta1;
  xc   = 0.5;

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
      PetscScalar    xp[2];
      PetscReal      pressure, vel[2], total_stress[3], strain_rate[3];
      
      // 1) Vx - Calculate SolCx
      xp[0] = coordx[i][iprev ];
      xp[1] = coordz[j][icenter];
      evaluate_solCx(xp,eta0,eta1,xc,1,vel,&pressure,total_stress,strain_rate);
      PetscCall(DMStagGetLocationSlot(dm, LEFT, 0, &idx)); 
      xx[j][i][idx] = vel[0];

      if (i == Nx-1) {
        xp[0] = coordx[i][inext  ];
        xp[1] = coordz[j][icenter];
        evaluate_solCx(xp,eta0,eta1,xc,1,vel,&pressure,total_stress,strain_rate);
        PetscCall(DMStagGetLocationSlot(dm, RIGHT, 0, &idx)); 
        xx[j][i][idx] = vel[0];
      }
      
      // 2) Vz
      xp[0] = coordx[i][icenter];
      xp[1] = coordz[j][iprev  ];
      evaluate_solCx(xp,eta0,eta1,xc,1,vel,&pressure,total_stress,strain_rate);
      PetscCall(DMStagGetLocationSlot(dm, DOWN, 0, &idx)); 
      xx[j][i][idx] = vel[1];

      if (j == Nz-1) {
        xp[0] = coordx[i][icenter];
        xp[1] = coordz[j][inext  ];
        evaluate_solCx(xp,eta0,eta1,xc,1,vel,&pressure,total_stress,strain_rate);
        PetscCall(DMStagGetLocationSlot(dm, UP, 0, &idx)); 
        xx[j][i][idx] = vel[1];
      }
    
      // 3) Pressure
      xp[0] = coordx[i][icenter];
      xp[1] = coordz[j][icenter];
      evaluate_solCx(xp,eta0,eta1,xc,1,vel,&pressure,total_stress,strain_rate);
      PetscCall(DMStagGetLocationSlot(dm, ELEMENT, 0, &idx)); 
      xx[j][i][idx] = pressure;
    }
  }

  // Restore arrays
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));
  PetscCall(DMStagVecRestoreArray(dm,xlocal,&xx)); 

  // Map local to global
  PetscCall(DMLocalToGlobalBegin(dm,xlocal,INSERT_VALUES,x)); 
  PetscCall(DMLocalToGlobalEnd  (dm,xlocal,INSERT_VALUES,x)); 

  PetscCall(VecDestroy(&xlocal)); 
  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s",usr->par->fdir_out,"out_analytic_solution"));
  PetscCall(DMStagViewBinaryPython(dm,x,fout));

  // Assign pointers
  *_x  = x;
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// ComputeErrorNorms
// ---------------------------------------
PetscErrorCode ComputeErrorNorms(DM dm,Vec x,Vec xanalytic, void *ctx)
{
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz;
  PetscScalar    xx[5], xa[5], dx, dz, dv;
  PetscScalar    nrm[3], gnrm[3], totp, avgp, gavgp;
  Vec            xlocal, xalocal;
  MPI_Comm       comm;
  PetscFunctionBeginUser;

  comm = PETSC_COMM_WORLD;

  // Get domain corners
  PetscCall(DMStagGetGlobalSizes(dm, &Nx, &Nz,NULL));
  PetscCall(DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 

  // Map global vectors to local domain
  PetscCall(DMGetLocalVector(dm, &xlocal)); 
  PetscCall(DMGlobalToLocal (dm, x, INSERT_VALUES, xlocal)); 

  PetscCall(DMCreateLocalVector (dm, &xalocal)); 
  PetscCall(DMGlobalToLocal (dm, xanalytic, INSERT_VALUES, xalocal)); 

  // Loop over local domain to calculate average pressure
  totp = 0.0; avgp = 0.0;
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      PetscScalar    p;
      DMStagStencil  point;
      
      // Get stencil values
      point.i = i; point.j = j; point.loc = ELEMENT; point.c = 0;
      PetscCall(DMStagVecGetValuesStencil(dm, xlocal, 1, &point, &p)); 

      // Average pressure
      totp += p;
    }
  }
  // Collect data 
  PetscCall(MPI_Allreduce(&totp, &gavgp, 1, MPI_DOUBLE, MPI_SUM, comm)); 
  avgp = gavgp/Nx/Nz;

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
      PetscCall(DMStagVecGetValuesStencil(dm, xlocal, 5, point, xx)); 

      // Get analytical solution
      PetscCall(DMStagVecGetValuesStencil(dm, xalocal, 5, point, xa)); 

      // Calculate errors
      ve[0] = PetscAbsScalar(xx[0]-xa[0]); // Left
      ve[1] = PetscAbsScalar(xx[1]-xa[1]); // Right
      ve[2] = PetscAbsScalar(xx[2]-xa[2]); // Down
      ve[3] = PetscAbsScalar(xx[3]-xa[3]); // Up
      pe    = PetscAbsScalar(xx[4]-avgp-xa[4]); // normalized pressure 

      // Calculate norms as in Duretz et al. 2011
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
  PetscCall(MPI_Allreduce(&nrm, &gnrm, 3, MPI_DOUBLE, MPI_SUM, comm)); 

  // Restore arrays and vectors
  PetscCall(DMRestoreLocalVector(dm, &xlocal )); 
  PetscCall(VecDestroy(&xalocal)); 

  // Print information
  PetscPrintf(comm,"# --------------------------------------- #\n");
  PetscPrintf(comm,"# NORMS: \n");
  PetscPrintf(comm,"# Velocity: norm1 = %1.12e norm1x = %1.12e norm1z = %1.12e \n",gnrm[0]+gnrm[1],gnrm[0],gnrm[1]);
  PetscPrintf(comm,"# Pressure: norm1 = %1.12e\n",gnrm[2]);
  PetscPrintf(comm,"# Grid info: hx = %1.12e hz = %1.12e \n",dx,dz);

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
  PetscCall(PetscInitialize(&argc, &argv,(char*)0, help));

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

  // Numerical solution using the FD pde object
  PetscCall(SNESStokes_Solcx(&dmStokes, &xStokes, usr)); 

  // Analytical solution
  PetscCall(Analytic_Solcx(dmStokes, &xAnalytic, usr)); 

  // Compute norms
  PetscCall(ComputeErrorNorms(dmStokes, xStokes, xAnalytic, usr)); 

  // Destroy objects
  PetscCall(DMDestroy(&dmStokes)); 
  PetscCall(VecDestroy(&xStokes)); 
  PetscCall(VecDestroy(&xAnalytic)); 

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
