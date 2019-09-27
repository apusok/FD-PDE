/* Application to solve the SolCx benchmark with FD-PDE */
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

#include "petsc.h"
#include "../fdstokes.h"
#include "../ex43-solcx.h"

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
  PetscInt       bcleft, bcright, bcup, bcdown;
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
PetscErrorCode SNESStokes_Solcx(DM*,Vec*,void*);
PetscErrorCode Analytic_Solcx(DM,Vec*,void*);
PetscErrorCode InputParameters(UsrData**);
PetscErrorCode InputPrintData(UsrData*);
PetscErrorCode InitializeDensity_SolCx(UsrData*, DM, Vec);
PetscErrorCode InitializeViscosityCenter_SolCx(UsrData*, DM, Vec);
PetscErrorCode InitializeViscosityCorner_SolCx(UsrData*, DM, Vec);
PetscErrorCode eval_fux(Coefficient, void*);
PetscErrorCode eval_fuz(Coefficient, void*);
PetscErrorCode eval_fp(Coefficient, void*);
PetscErrorCode eval_eta_n(Coefficient, void*);
PetscErrorCode eval_eta_c(Coefficient, void*);
PetscErrorCode StokesBCListPopulate(DM, DMStagBC*, PetscInt);
PetscErrorCode StokesBCListPopulateList(DM dm, DMStagBCList bclist);
PetscErrorCode ComputeErrorNorms(DM,Vec,Vec,void*);
PetscErrorCode DoOutput(DM,Vec,const char[]);

// ---------------------------------------
// Application functions
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "SNESStokes_SolCx"
PetscErrorCode SNESStokes_Solcx(DM *_dm, Vec *_x, void *ctx)
{
  UsrData     *usr = (UsrData*) ctx;
  FD           fd;
  DMStagBCList bclist;
  DM           dmPV, dmCoeff;
  Vec          x, coeff;
  PetscInt     nx, nz, dofPV0, dofPV1, dofPV2;
  PetscInt     dofCf0, dofCf1, dofCf2, stencilWidth;
  PetscScalar  xmin, zmin, xmax, zmax;
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

  // stencil dofs - need to be more explicit what are the location and c
  dofPV0 = 0; dofPV1 = 1; dofPV2 = 1; // dmPV: Vx, Vz (edges), P (element)
  dofCf0 = 1; dofCf1 = 1; dofCf2 = 1; // dmCoeff: rho (edges), eta (corner, elements)
  stencilWidth = 1;

  // Create DMStag object for Stokes unknowns: dmPV (P-element, v-vertex)
  ierr = DMStagCreate2d(usr->comm, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, nx, nz, 
            PETSC_DECIDE, PETSC_DECIDE, dofPV0, dofPV1, dofPV2, 
            DMSTAG_STENCIL_BOX, stencilWidth, NULL,NULL, &dmPV); CHKERRQ(ierr);
  ierr = DMSetFromOptions(dmPV); CHKERRQ(ierr);
  ierr = DMSetUp         (dmPV); CHKERRQ(ierr);
  //ierr = DMStagSetUniformCoordinatesExplicit(dmPV, xmin, xmax, zmin, zmax, 0.0, 0.0); CHKERRQ(ierr);
  ierr = DMStagSetUniformCoordinatesProduct(dmPV, xmin, xmax, zmin, zmax, 0.0, 0.0);CHKERRQ(ierr);

  // Create DMStag object for Stokes coefficients: dmCoeff 
  ierr = DMStagCreateCompatibleDMStag(dmPV, dofCf0, dofCf1, dofCf2, 0, &dmCoeff); CHKERRQ(ierr);
  ierr = DMSetUp(dmCoeff); CHKERRQ(ierr);
  //ierr = DMStagSetUniformCoordinatesExplicit(dmCoeff, xmin, xmax, zmin, zmax, 0.0, 0.0); CHKERRQ(ierr);
  ierr = DMStagSetUniformCoordinatesProduct(dmCoeff, xmin, xmax, zmin, zmax, 0.0, 0.0);CHKERRQ(ierr);

  // Create boundary conditions list
  //ierr = DMStagBCCreateDefault(dmPV,&bclist,&nbc);CHKERRQ(ierr);
  //ierr = StokesBCListPopulate(dmPV,bclist,nbc);CHKERRQ(ierr);

  ierr = DMStagBCListCreate(dmPV,&bclist);CHKERRQ(ierr);
  ierr = StokesBCListPopulateList(dmPV,bclist);CHKERRQ(ierr);
  
  
  // Create the FD-pde object
  ierr = FDCreate(PETSC_COMM_WORLD,&fd); CHKERRQ(ierr);
  ierr = FDSetType(fd,STOKES); CHKERRQ(ierr);
  ierr = FDStokesSetData(fd,dmPV,dmCoeff,bclist);CHKERRQ(ierr);
  //ierr = FDSetOptionsPrefix(fd,"stk_"); CHKERRQ(ierr);

  // Set coefficients evaluation functions for eta, f
  {
    Coefficient eta_c, eta_n, fux, fuz, fp;
    ierr = FDStokesGetCoefficients(fd,&eta_c,&eta_n,&fux,&fuz,&fp); CHKERRQ(ierr);
    ierr = CoefficientSetEvaluate(eta_c,eval_eta_c,(void*)usr); CHKERRQ(ierr);
    ierr = CoefficientSetEvaluate(eta_n,eval_eta_n,(void*)usr); CHKERRQ(ierr);
    ierr = CoefficientSetEvaluate(fux,eval_fux,(void*)usr); CHKERRQ(ierr);
    ierr = CoefficientSetEvaluate(fuz,eval_fuz,(void*)usr); CHKERRQ(ierr);
    ierr = CoefficientSetEvaluate(fp ,eval_fp ,(void*)usr); CHKERRQ(ierr);
  }

  // FD SNES Solver
  ierr = FDSetSolveSNES(fd);CHKERRQ(ierr);
  ierr = FDGetSolution(fd,&x,&coeff);CHKERRQ(ierr); // output both the solution and coefficients (NULL if not needed)

  //ierr = FDView();CHKERRQ(ierr); //standard ASCII view on screen

  // Output solution and coefficient (CENTER or CORNER) - dm and associated vector
  ierr = DoOutput(dmPV,x,"numerical_solution.vtr");CHKERRQ(ierr);
  //ierr = DMStagVTKDump(dmPV,x);CHKERRQ(ierr);
  //ierr = DMStagVTKDump(dmCoeff,coeff);CHKERRQ(ierr);

  // Destroy FD-PDE object
  ierr = DMStagBCListDestroy(&bclist);CHKERRQ(ierr);
  ierr = FDDestroy(&fd);CHKERRQ(ierr);
  ierr = DMDestroy(&dmCoeff);CHKERRQ(ierr);
  ierr = VecDestroy(&coeff);CHKERRQ(ierr);

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
  ierr = PetscBagRegisterScalar(bag, &par->g, 1.0, "g", "Gravitational acceleration"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->eta0, 1.0, "eta0", "Viscosity eta0"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->eta1, 1.0, "eta1", "Viscosity eta1"); CHKERRQ(ierr);

  // Boundary conditions

  // Input/output 
  ierr = PetscBagRegisterString(bag,&par->fname_out,FNAME_LENGTH,"output","output_file","Name for output file, set with: -output_file <filename>"); CHKERRQ(ierr);

  // Other variables
  par->fname_in[0] = '\0';

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
  ierr = PetscBagView(usr->bag,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
  PetscPrintf(usr->comm,"# --------------------------------------- #\n");

  // Free memory
  ierr = PetscFree(opts); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
// ---------------------------------------
// InitializeDensity_SolCx - array
// Density is defined (edges): sin(pi*z)*cos(pi*x); 
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "InitializeDensity_SolCx"
PetscErrorCode InitializeDensity_SolCx(UsrData *usr, DM dm, Vec coeff)
{
  PetscInt       i, j, sx, sz, nx, nz;
  Vec            coefflocal;
  PetscScalar    **coordx,**coordz;
  PetscInt       iprev, inext, icenter;
  PetscScalar    ***c;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  // Get domain corners
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  // Get dm coordinates array
  ierr = DMStagGet1dCoordinateArraysDOFRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGet1dCoordinateLocationSlot(dm,LEFT,&iprev);CHKERRQ(ierr);
  ierr = DMStagGet1dCoordinateLocationSlot(dm,RIGHT,&inext);CHKERRQ(ierr); 
  ierr = DMStagGet1dCoordinateLocationSlot(dm,ELEMENT,&icenter);CHKERRQ(ierr);

  // Map global vectors to local domain
  ierr = DMGetLocalVector(dm, &coefflocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm, coeff, INSERT_VALUES, coefflocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayDOF(dm, coefflocal, &c); CHKERRQ(ierr);
  
  // Loop over local domain - set initial density and viscosity
  for (j = sz; j < sz+nz; ++j) {
    for (i = sx; i <sx+nx; ++i) {

      DMStagStencil point[4];
      PetscScalar   xp[4], zp[4], fval=0.0;
      PetscInt      ii, idx;
        
      // Set stencil values
      point[0].i = i; point[0].j = j; point[0].loc = LEFT;    point[0].c = 0;
      point[1].i = i; point[1].j = j; point[1].loc = RIGHT;   point[1].c = 0;
      point[2].i = i; point[2].j = j; point[2].loc = DOWN;    point[2].c = 0;
      point[3].i = i; point[3].j = j; point[3].loc = UP;      point[3].c = 0;

      // Get coordinates
      xp[0] = coordx[i][iprev  ]; zp[0] = coordz[j][icenter];
      xp[1] = coordx[i][inext  ]; zp[1] = coordz[j][icenter];
      xp[2] = coordx[i][icenter]; zp[2] = coordz[j][iprev  ];
      xp[3] = coordx[i][icenter]; zp[3] = coordz[j][inext  ];

      // Set densities
      for (ii = 0; ii < 4; ++ii) {
        fval = PetscSinScalar(PETSC_PI*zp[ii]) * PetscCosScalar(PETSC_PI*xp[ii]); 
        ierr = DMStagGetLocationSlot(dm, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = fval;
      }
    }
  }

  // Restore arrays, local vectors
  ierr = DMStagVecRestoreArrayDOF(dm,coefflocal,&c);CHKERRQ(ierr);
  ierr = DMLocalToGlobal(dm,coefflocal,INSERT_VALUES,coeff);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&coefflocal);CHKERRQ(ierr);
  ierr = DMStagRestore1dCoordinateArraysDOFRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
// ---------------------------------------
// InitializeViscosityCenter_SolCx
// Viscosity is defined (corner and center): solcx_eta0, for 0<=x<=0.5, solcx_eta1, for 0.5<x<=1
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "InitializeViscosityCenter_SolCx"
PetscErrorCode InitializeViscosityCenter_SolCx(UsrData *usr, DM dm, Vec coeff)
{
  PetscInt       i, j, sx, sz, nx, nz;
  Vec            coefflocal;
  PetscScalar    **coordx,**coordz;
  PetscInt       icenter;
  PetscScalar    L2 = 0.5;
  PetscScalar    ***c;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  // Get domain corners
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  // Get dm coordinates array
  ierr = DMStagGet1dCoordinateArraysDOFRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGet1dCoordinateLocationSlot(dm,ELEMENT,&icenter);CHKERRQ(ierr);

  // Map global vectors to local domain
  ierr = DMGetLocalVector(dm, &coefflocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm, coeff, INSERT_VALUES, coefflocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayDOF(dm, coefflocal, &c); CHKERRQ(ierr);
  
  // Loop over local domain - set initial density and viscosity
  for (j = sz; j < sz+nz; ++j) {
    for (i = sx; i <sx+nx; ++i) {
      DMStagStencil point;
      PetscScalar   xp, fval=0.0;
      PetscInt      idx;
        
      // Set stencil values
      point.i = i; point.j = j; point.loc = ELEMENT; point.c = 0;

      // Get coordinates
      xp = coordx[i][icenter];

      // Set eta
      if (xp <= L2) fval = usr->par->eta0;
      else          fval = usr->par->eta1;

      ierr = DMStagGetLocationSlot(dm, point.loc, point.c, &idx); CHKERRQ(ierr);
      c[j][i][idx] = fval;
    }
  }

  // Restore arrays, local vectors
  ierr = DMStagVecRestoreArrayDOF(dm,coefflocal,&c); CHKERRQ(ierr);
  ierr = DMLocalToGlobal(dm,coefflocal,INSERT_VALUES,coeff);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&coefflocal);       CHKERRQ(ierr);
  ierr = DMStagRestore1dCoordinateArraysDOFRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
// ---------------------------------------
// InitializeViscosityCorner_SolCx
// Viscosity is defined (corner and center): solcx_eta0, for 0<=x<=0.5, solcx_eta1, for 0.5<x<=1
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "InitializeViscosityCorner_SolCx"
PetscErrorCode InitializeViscosityCorner_SolCx(UsrData *usr, DM dm, Vec coeff)
{
  PetscInt       i, j, sx, sz, nx, nz;
  Vec            coefflocal;
  PetscScalar    **coordx,**coordz;
  PetscInt       iprev, inext;
  PetscScalar    L2 = 0.5;
  PetscScalar    ***c;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  // Get domain corners
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  // Get dm coordinates array
  ierr = DMStagGet1dCoordinateArraysDOFRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGet1dCoordinateLocationSlot(dm,LEFT,&iprev);CHKERRQ(ierr);
  ierr = DMStagGet1dCoordinateLocationSlot(dm,RIGHT,&inext);CHKERRQ(ierr); 

  // Map global vectors to local domain
  ierr = DMGetLocalVector(dm, &coefflocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm, coeff, INSERT_VALUES, coefflocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayDOF(dm, coefflocal, &c); CHKERRQ(ierr);
  
  // Loop over local domain - set initial density and viscosity
  for (j = sz; j < sz+nz; ++j) {
    for (i = sx; i <sx+nx; ++i) {
      DMStagStencil point[4];
      PetscScalar   xp[4], fval=0.0;
      PetscInt      ii, idx;
        
      // Set stencil values
      point[0].i = i; point[0].j = j; point[0].loc = DOWN_LEFT;  point[0].c = 0;
      point[1].i = i; point[1].j = j; point[1].loc = DOWN_RIGHT; point[1].c = 0;
      point[2].i = i; point[2].j = j; point[2].loc = UP_LEFT;    point[2].c = 0;
      point[3].i = i; point[3].j = j; point[3].loc = UP_RIGHT;   point[3].c = 0;

      // Get coordinates
      xp[0] = coordx[i][iprev];
      xp[1] = coordx[i][inext];
      xp[2] = coordx[i][iprev];
      xp[3] = coordx[i][inext];

      // Set eta
      for (ii = 0; ii < 4; ++ii) {
        if (xp[ii] <= L2) fval = usr->par->eta0;
        else              fval = usr->par->eta1;

        ierr = DMStagGetLocationSlot(dm, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = fval;
      }
    }
  }

  // Restore arrays, local vectors
  ierr = DMStagVecRestoreArrayDOF(dm,coefflocal,&c); CHKERRQ(ierr);
  ierr = DMLocalToGlobal(dm,coefflocal,INSERT_VALUES,coeff);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&coefflocal);       CHKERRQ(ierr);
  ierr = DMStagRestore1dCoordinateArraysDOFRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// eval_fux = 0
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "eval_fux"
PetscErrorCode eval_fux(Coefficient c, void *data)
{
  FD fd;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  fd = c->fd;
  // Extract part from dmcoeff [Vx] and multiply by prefactor a=0.0
  ierr = DMStagExtract1DComponent(fd->dmcoeff, fd->coeff, RIGHT, 0, 0.0, c->coeff);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// eval_fuz = rho*g
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "eval_fuz"
PetscErrorCode eval_fuz(Coefficient c, void *data)
{
  FD fd;
  UsrData *usr = (UsrData*)data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  fd = c->fd;

  // Initialize/update density
  ierr = InitializeDensity_SolCx(usr,fd->dmcoeff,fd->coeff);CHKERRQ(ierr);

  // Extract Rho part from dmcoeff [Vz] and multiply by prefactor a=-g
  ierr = DMStagExtract1DComponent(fd->dmcoeff, fd->coeff, DOWN, 0, -usr->par->g, c->coeff);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// eval_fp = 0
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "eval_fp"
PetscErrorCode eval_fp(Coefficient c, void *data)
{
  FD fd;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  fd = c->fd;
  // Extract part from dmcoeff [eta_c/dummy location] and multiply by prefactor a=0.0
  ierr = DMStagExtract1DComponent(fd->dmcoeff, fd->coeff, ELEMENT, 0, 0.0, c->coeff);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// eval_eta_n = eta0_n
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "eval_eta_n"
PetscErrorCode eval_eta_n(Coefficient c, void *data)
{
  FD fd;
  UsrData *usr = (UsrData*)data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  fd = c->fd;

  // Initialize/Update viscosity in corners
  ierr = InitializeViscosityCorner_SolCx(usr,fd->dmcoeff,fd->coeff);CHKERRQ(ierr);

  // Extract part from dmCoeff [eta_n corners]
  ierr = DMStagExtract1DComponent(fd->dmcoeff, fd->coeff, DOWN_RIGHT, 0, 1.0, c->coeff);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// eval_eta_c = eta0_c
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "eval_eta_c"
PetscErrorCode eval_eta_c(Coefficient c, void *data)
{
  FD fd;
  UsrData *usr = (UsrData*)data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  fd = c->fd;

  // Initialize/Update viscosity in center
  ierr = InitializeViscosityCenter_SolCx(usr,fd->dmcoeff,fd->coeff);CHKERRQ(ierr);

  // Extract part from dmCoeff [eta_c center]
  ierr = DMStagExtract1DComponent(fd->dmcoeff, fd->coeff, ELEMENT, 0, 1.0, c->coeff);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// StokesListPopulate(dmPV,&bclist,nbc)
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "StokesBCListPopulate"
PetscErrorCode StokesBCListPopulate(DM dm, DMStagBC *bclist, PetscInt ndof)
{
  DMStagBC         *list;
  PetscInt       i, j, idx, ibc, Nx, Nz;
  PetscInt       iright, ileft, idown, iup;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;

  list = bclist;

  // Warning: right[i] and left[i+1] do not have the same loc, idx
  ierr = DMStagGetGlobalSizes(dm, &Nx, &Nz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm,LEFT,0,&ileft); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm,RIGHT,0,&iright); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm,DOWN,0,&idown); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm,UP,0,&iup); CHKERRQ(ierr);

  // Loop on boundary dofs
  for (ibc = 0; ibc<ndof; ibc++) {
    i   = list[ibc].point.i;
    j   = list[ibc].point.j;
    idx = list[ibc].idx;

    // Dirichlet BC Vx=0 on LEFT
    if ((i==0) && (idx==ileft)) { //5
      list[ibc].type = BC_DIRICHLET;
      list[ibc].val  = 0.0;
    }

    // Dirichlet BC Vx=0 on RIGHT
    if ((i==Nx-1) && (idx==iright)) {//5
      list[ibc].type = BC_DIRICHLET;
      list[ibc].val  = 0.0;
    }

    // Dirichlet BC Vz=0 on DOWN
    if ((j==0) && (idx==idown)) {
      list[ibc].type = BC_DIRICHLET;
      list[ibc].val  = 0.0;
    }

    // Dirichlet BC Vz=0 on UP
    if ((j==Nz-1) && (idx==iup)) {
      list[ibc].type = BC_DIRICHLET;
      list[ibc].val  = 0.0;
    }

    // Neumann BC dVz/dx = 0 on LEFT/RIGHT
    if (((i==0) || (i==Nx-1)) && (j>0) && (idx==idown)) {
      list[ibc].type = BC_NEUMANN;
      list[ibc].val  = 0.0;
    }

    // Neumann BC dVx/dz = 0 on DOWN/UP - one missing defined as RIGHT (below)
    if (((j==0) || (j==Nz-1)) && (i>0) && (idx==ileft)) {
      list[ibc].type = BC_NEUMANN;
      list[ibc].val  = 0.0;
    }

    if (((j==0) || (j==Nz-1)) && (i==Nx-2) && (idx==iright)) {
      list[ibc].type = BC_NEUMANN;
      list[ibc].val  = 0.0;
    }
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StokesBCListPopulateList"
PetscErrorCode StokesBCListPopulateList(DM dm, DMStagBCList bclist)
{
  DMStagBC       *list;
  PetscInt    k,n_bc,*idx_bc;
  PetscScalar *x_bc,*value_bc;
  BCType      *type_bc;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  
  list = bclist->bc_f;
  
  ierr = DMStagBCListGetValues(bclist,'w','-',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  
  ierr = DMStagBCListGetValues(bclist,'e','-',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  
  ierr = DMStagBCListGetValues(bclist,'n','|',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  

  ierr = DMStagBCListGetValues(bclist,'e','|',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  ////
  
  ierr = DMStagBCListGetValues(bclist,'w','|',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  
  
  ierr = DMStagBCListGetValues(bclist,'e','|',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  
  
  ierr = DMStagBCListGetValues(bclist,'n','-',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  
  
  ierr = DMStagBCListGetValues(bclist,'e','-',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  
  
  
  
  PetscFunctionReturn(0);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Get parameters
  eta0 = usr->par->eta0;
  eta1 = usr->par->eta1;
  xc   = 0.5;

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
      PetscScalar    xp[2];
      PetscReal      pressure, vel[2], total_stress[3], strain_rate[3];
      
      // 1) Vx - Calculate SolCx
      xp[0] = coordx[i][iprev ];
      xp[1] = coordz[j][icenter];
      evaluate_solCx(xp,eta0,eta1,xc,1,vel,&pressure,total_stress,strain_rate);
      ierr = DMStagGetLocationSlot(dm, LEFT, 0, &idx); CHKERRQ(ierr);
      xx[j][i][idx] = vel[0];

      if (i == Nx-1) {
        xp[0] = coordx[i][inext  ];
        xp[1] = coordz[j][icenter];
        evaluate_solCx(xp,eta0,eta1,xc,1,vel,&pressure,total_stress,strain_rate);
        ierr = DMStagGetLocationSlot(dm, RIGHT, 0, &idx); CHKERRQ(ierr);
        xx[j][i][idx] = vel[0];
      }
      
      // 2) Vz
      xp[0] = coordx[i][icenter];
      xp[1] = coordz[j][iprev  ];
      evaluate_solCx(xp,eta0,eta1,xc,1,vel,&pressure,total_stress,strain_rate);
      ierr = DMStagGetLocationSlot(dm, DOWN, 0, &idx); CHKERRQ(ierr);
      xx[j][i][idx] = vel[1];

      if (j == Nz-1) {
        xp[0] = coordx[i][icenter];
        xp[1] = coordz[j][inext  ];
        evaluate_solCx(xp,eta0,eta1,xc,1,vel,&pressure,total_stress,strain_rate);
        ierr = DMStagGetLocationSlot(dm, UP, 0, &idx); CHKERRQ(ierr);
        xx[j][i][idx] = vel[1];
      }
    
      // 3) Pressure
      xp[0] = coordx[i][icenter];
      xp[1] = coordz[j][icenter];
      evaluate_solCx(xp,eta0,eta1,xc,1,vel,&pressure,total_stress,strain_rate);
      ierr = DMStagGetLocationSlot(dm, ELEMENT, 0, &idx); CHKERRQ(ierr);
      xx[j][i][idx] = pressure;
    }
  }

  // Restore arrays
  ierr = DMStagRestore1dCoordinateArraysDOFRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArrayDOF(dm,xlocal,&xx); CHKERRQ(ierr);

  // Map local to global
  ierr = DMLocalToGlobalBegin(dm,xlocal,INSERT_VALUES,x); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,xlocal,INSERT_VALUES,x); CHKERRQ(ierr);

  ierr = VecDestroy(&xlocal); CHKERRQ(ierr);

  ierr = DoOutput(dm,x,"analytic_solution.vtr");CHKERRQ(ierr);

  // Assign pointers
  *_x  = x;
  
  PetscFunctionReturn(0);
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

  PetscErrorCode ierr;
  PetscFunctionBegin;

  comm = PETSC_COMM_WORLD;

  // Get domain corners
  ierr = DMStagGetGlobalSizes(dm, &Nx, &Nz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  // Map global vectors to local domain
  ierr = DMGetLocalVector(dm, &xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm, x, INSERT_VALUES, xlocal); CHKERRQ(ierr);

  ierr = DMCreateLocalVector (dm, &xalocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm, xanalytic, INSERT_VALUES, xalocal); CHKERRQ(ierr);

  // Loop over local domain to calculate average pressure
  totp = 0.0; avgp = 0.0;
  for (j = sz; j < sz+nz; ++j) {
    for (i = sx; i <sx+nx; ++i) {
      PetscScalar    p;
      DMStagStencil  point;
      
      // Get stencil values
      point.i = i; point.j = j; point.loc = ELEMENT; point.c = 0; // P
      ierr = DMStagVecGetValuesStencil(dm, xlocal, 1, &point, &p); CHKERRQ(ierr);

      // Average pressure
      totp += p;
    }
  }
  // Collect data 
  ierr = MPI_Allreduce(&totp, &gavgp, 1, MPI_DOUBLE, MPI_SUM, comm); CHKERRQ(ierr);
  avgp = gavgp/Nx/Nz;

  // Initialize norms
  nrm[0] = 0.0; nrm[1] = 0.0; nrm[2] = 0.0;
  dx = 1.0/Nx;
  dz = 1.0/Nz;
  dv = dx*dz;

  // Loop over local domain to calculate ELEMENT errors
  for (j = sz; j < sz+nz; ++j) {
    for (i = sx; i <sx+nx; ++i) {
      
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
  ierr = MPI_Allreduce(&nrm, &gnrm, 3, MPI_DOUBLE, MPI_SUM, comm); CHKERRQ(ierr);

  // Restore arrays and vectors
  ierr = DMRestoreLocalVector(dm, &xlocal ); CHKERRQ(ierr);
  ierr = VecDestroy(&xalocal); CHKERRQ(ierr);

  // Print information
  PetscPrintf(comm,"# --------------------------------------- #\n");
  PetscPrintf(comm,"# NORMS: \n");
  PetscPrintf(comm,"# Velocity: norm1 = %1.12e norm1x = %1.12e norm1z = %1.12e \n",gnrm[0]+gnrm[1],gnrm[0],gnrm[1]);
  PetscPrintf(comm,"# Pressure: norm1 = %1.12e\n",gnrm[2]);
  PetscPrintf(comm,"# Grid info: hx = %1.12e hz = %1.12e \n",dx,dz);

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
  ierr = DMStagSetUniformCoordinatesExplicit(dmVel,0.0,1.0,0.0,1.0,0.0,0.0); CHKERRQ(ierr);

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

  // Numerical solution using the FD pde object
  ierr = SNESStokes_Solcx(&dmStokes, &xStokes, usr); CHKERRQ(ierr);

  // Analytical solution
  ierr = Analytic_Solcx(dmStokes, &xAnalytic, usr); CHKERRQ(ierr);

  // Compute norms
  ierr = ComputeErrorNorms(dmStokes, xStokes, xAnalytic, usr); CHKERRQ(ierr);

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
