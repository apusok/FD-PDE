static char help[] = "Solves isoviscous 2D Stokes equations with SNES\n";
// 2-D Single-phase Stokes equations: x(i), z(j) directions.

// Run program:
// mpiexec -n 1 ./stokes_snes -snes_mf
// mpiexec -n 1 ./stokes_snes -pc_type jacobi -snes_fd
// mpiexec -n 1 ./stokes_snes -pc_type jacobi

#include <petscdm.h>
#include <petscksp.h>
#include <petscdmstag.h>
#include <petscdmda.h>
#include <petscsnes.h>
#include "prealloc_helper.h"

// ---------------------------------------
// Data structures
// ---------------------------------------
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

enum LocationType
{
	CENTER,  // center (for effective viscosity)
	CORNER,  // corner (for effective viscosity)
  BCLEFT,  // left (for BC)
  BCRIGHT, // right (for BC)
  BCUP,    // up (for BC)
  BCDOWN,  // down (for BC)
  NONE
};

enum BCType
{
	FREE_SLIP, 
	NO_SLIP
};

enum ModelType
{
	SOLCX, 
	MOR
};

// ---------------------------------------
// Application Context
// ---------------------------------------
// user defined and model-dependent variables
typedef struct {
  MPI_Comm       comm;
  PetscInt       nx, nz;
  PetscScalar    L, H;
  PetscScalar    xmin, zmin;
  PetscScalar    rho1, rho2, eta0, ndisl;
  PetscScalar    g;
  enum ModelType mtype;
} UsrData;

// grid variables
typedef struct {
  // stencils: dof0 per vertex, dof1 per edge, dof1 per face/element
  PetscInt dofPV0, dofPV1, dofPV2;
  PetscInt dofCf0, dofCf1, dofCf2;
  PetscInt stencilWidth;

  // domain parameters
  PetscInt     nx, nz;
  PetscScalar  dx, dz;
  PetscScalar  xmin, zmin, xmax, zmax;
  enum BCType  bcleft, bcright, bcup, bcdown;
  PetscScalar  Vleft, Vright, Vup, Vdown;
} GridData;

// solver variables
typedef struct {
  MPI_Comm     comm;
  PetscMPIInt  rank;
  UsrData      *usr;
  GridData     *grd;
  DM           dmPV, dmCoeff;
  Vec          coeff;
  Vec          r, x, xguess;
  Mat          J;
  PetscScalar  Pdiff, Pdisl;
} SolverCtx;

// ---------------------------------------
// Function Definitions
// ---------------------------------------
PetscErrorCode CreateSystem(SolverCtx*);
PetscErrorCode JacobianMatrixPreallocation(SolverCtx*);
PetscErrorCode XMomentumStencil(PetscInt, PetscInt, PetscInt, PetscInt, DMStagStencil*);
PetscErrorCode ZMomentumStencil(PetscInt, PetscInt, PetscInt, PetscInt, DMStagStencil*);
PetscErrorCode InitializeModel(SolverCtx*);
PetscErrorCode InitializeModel_SolCx(SolverCtx*);
PetscErrorCode FormFunctionPV(SNES, Vec, Vec, void*); // global to local
PetscErrorCode FormFunctionPV1(SNES, Vec, Vec, void*); // insert in global vector
PetscErrorCode XMomentumResidual(SolverCtx*, Vec, PetscInt, PetscInt, enum LocationType, PetscScalar*);
PetscErrorCode ZMomentumResidual(SolverCtx*, Vec, Vec, PetscInt, PetscInt, enum LocationType, PetscScalar*);
PetscErrorCode CalcEffViscosity(SolverCtx*, Vec, PetscInt, PetscInt, enum LocationType, PetscScalar*);
PetscErrorCode FormInitialGuess(SolverCtx*);
PetscErrorCode SolveSystem(SNES, SolverCtx*);
PetscErrorCode DoOutput(SolverCtx*);

// ---------------------------------------
// Main function
// ---------------------------------------
int main(int argc,char **argv)
{
  PetscErrorCode  ierr;
  SolverCtx       *sol;
  UsrData         usr;
  GridData        grd;
  SNES            snes;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help); if (ierr) return ierr;

  // allocate memory to application context
  ierr = PetscMalloc1(1, &sol); CHKERRQ(ierr);
  //ierr = PetscMalloc1(1, &usr); CHKERRQ(ierr);
  //ierr = PetscMalloc1(1, &grd); CHKERRQ(ierr);
  
  // ---------------------------------------
  /* Set Parameters
     1. Set default params
     2. Read params from file
     3. Read params from command line
     4. Print params
     5. Scaling of params (if necessary) */
  // ---------------------------------------

  // ---------------------------------------
  // Initialize default (user) variables
  // ---------------------------------------
  usr.comm = PETSC_COMM_WORLD;
  usr.nx   = 20;   // element count x-dir
  usr.nz   = 20;   // element count z-dir
  usr.xmin = 0.0;  // x-min coordinate
  usr.zmin = 0.0;  // z-min coordinate
  usr.L    = 1.0;  // length of domain
  usr.H    = 1.0;  // height of domain
  usr.rho1 = 2;    // density 1
  usr.rho2 = 1;    // density 2
  usr.eta0 = 1;    // reference viscosity
  usr.ndisl= 0;    // power coefficient (dislocation)
  usr.g    = 1.0;  // gravitational acceleration (non-dimensional)

  // Model type
  usr.mtype = SOLCX; 
  
  // ---------------------------------------
  // Read variables from command line (or from file)
  // ---------------------------------------
  ierr = PetscOptionsGetInt(NULL, NULL, "-nx", &usr.nx, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, NULL, "-nz", &usr.nz, NULL); CHKERRQ(ierr);

  ierr = PetscOptionsGetScalar(NULL, NULL, "-eta0" , &usr.eta0 , NULL); CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(NULL, NULL, "-ndisl", &usr.ndisl, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(NULL, NULL, "-rho1" , &usr.rho1 , NULL); CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(NULL, NULL, "-rho2" , &usr.rho2 , NULL); CHKERRQ(ierr);

  // ---------------------------------------
  // Initialize other context variables
  // ---------------------------------------
  // Grid context - contains scaled grid defined by user options
  // scale variables
  grd.nx   = usr.nx;
  grd.nz   = usr.nz;
  grd.xmin = usr.xmin;
  grd.zmin = usr.zmin;
  grd.xmax = usr.xmin + usr.L;
  grd.zmax = usr.zmin + usr.H;
  grd.dx   = (grd.xmax - grd.xmin)/(grd.nx);
  grd.dz   = (grd.zmax - grd.zmin)/(grd.nz);

  // stencil 
  grd.dofPV0 = 0; grd.dofPV1 = 1; grd.dofPV2 = 1; // Vx, Vz, P
  grd.dofCf0 = 0; grd.dofCf1 = 0; grd.dofCf2 = 1; // rho_element
  grd.stencilWidth = 1;

  // boundary conditions
  grd.bcleft  = FREE_SLIP;
  grd.bcright = FREE_SLIP;
  grd.bcup    = FREE_SLIP;
  grd.bcdown  = FREE_SLIP;

  grd.Vleft  = 0.0;
  grd.Vright = 0.0;
  grd.Vup    = 0.0;
  grd.Vdown  = 0.0;

  // Solver context
  sol->comm = usr.comm;
  sol->usr  = &usr;
  sol->grd  = &grd;

  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &sol->rank); CHKERRQ(ierr);
  // print variables

  // ---------------------------------------
  // Create DM data structures
  // ---------------------------------------
  // create DMStag object: dmStokes(P-element,v-vertex)
  ierr = DMStagCreate2d(sol->comm, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, grd.nx, grd.nz, 
            PETSC_DECIDE, PETSC_DECIDE, grd.dofPV0, grd.dofPV1, grd.dofPV2, DMSTAG_STENCIL_BOX, grd.stencilWidth, NULL,NULL, &sol->dmPV); CHKERRQ(ierr);

  // set dm options
  ierr = DMSetFromOptions(sol->dmPV); CHKERRQ(ierr);
  ierr = DMSetUp         (sol->dmPV); CHKERRQ(ierr);
  
  // set coordinates dmPV
  ierr = DMStagSetUniformCoordinatesExplicit(sol->dmPV, grd.xmin, grd.xmax, grd.zmin, grd.zmax, 0.0, 0.0); CHKERRQ(ierr);
  
  // create dmCoeff(rho-element c=0,eta-element c=1)
  ierr = DMStagCreateCompatibleDMStag(sol->dmPV, grd.dofCf0, grd.dofCf1, grd.dofCf2, 0, &sol->dmCoeff); CHKERRQ(ierr);
  ierr = DMSetUp(sol->dmCoeff); CHKERRQ(ierr);
  
  // set coordinates dmCoeff
  ierr = DMStagSetUniformCoordinatesExplicit(sol->dmCoeff, grd.xmin, grd.xmax, grd.zmin, grd.zmax, 0.0, 0.0); CHKERRQ(ierr);

  // ---------------------------------------
  // Create global and matrices vectors
  // ---------------------------------------
  ierr = CreateSystem(sol); CHKERRQ(ierr);

  // Create model setup
  ierr = InitializeModel(sol); CHKERRQ(ierr);

  // ---------------------------------------
  // Create nonlinear solver context
  // ---------------------------------------
  ierr = SNESCreate(sol->comm,&snes); CHKERRQ(ierr);

  // set dm to snes
  ierr = SNESSetDM(snes, sol->dmPV); CHKERRQ(ierr);

  // set solution - need to do this for FD colouring to function correctly
  ierr = SNESSetSolution(snes, sol->x); CHKERRQ(ierr);

  // set function evaluation routine
  ierr = SNESSetFunction(snes, sol->r, FormFunctionPV, sol); CHKERRQ(ierr);

  // set Jacobian
  ierr = SNESSetJacobian(snes, sol->J, sol->J, SNESComputeJacobianDefaultColor, NULL);CHKERRQ(ierr);

  // ---------------------------------------
  // SNES Options
  // ---------------------------------------
  // customize DEFAULT SNES options
  //ierr = SNESGetKSP(snes, &ksp); CHKERRQ(ierr);
  //ierr = KSPGetPC  (ksp,  &pc ); CHKERRQ(ierr);
  //ierr = PCSetType (pc, PCNONE); CHKERRQ(ierr);

  // Get default info on convergence
  ierr = PetscOptionsSetValue(NULL, "-snes_monitor",          ""); CHKERRQ(ierr);
  ierr = PetscOptionsSetValue(NULL, "-ksp_monitor",           ""); CHKERRQ(ierr);
  ierr = PetscOptionsSetValue(NULL, "-snes_converged_reason", ""); CHKERRQ(ierr);
  ierr = PetscOptionsSetValue(NULL, "-ksp_converged_reason",  ""); CHKERRQ(ierr);
  
  // overwrite default options from command line
  ierr = SNESSetFromOptions(snes); CHKERRQ(ierr);
  
  // ---------------------------------------
  // Initialize and solve application
  // ---------------------------------------
  // evaluate initial guess
  ierr = FormInitialGuess(sol); CHKERRQ(ierr);
  
  // solve non-linear system
  ierr = SolveSystem(snes, sol); CHKERRQ(ierr);
  
  // ---------------------------------------
  // OUTPUT solution to file
  // ---------------------------------------
  ierr = DoOutput(sol); CHKERRQ(ierr);

  // ---------------------------------------
  // Clean Up
  // ---------------------------------------
  // destroy PETSc objects
  ierr = VecDestroy(&sol->coeff  ); CHKERRQ(ierr);
  ierr = VecDestroy(&sol->xguess ); CHKERRQ(ierr);
  ierr = VecDestroy(&sol->x      ); CHKERRQ(ierr);
  ierr = VecDestroy(&sol->r      ); CHKERRQ(ierr);

  ierr = MatDestroy(&sol->J      ); CHKERRQ(ierr);
  
  ierr = DMDestroy(&sol->dmPV   ); CHKERRQ(ierr);
  ierr = DMDestroy(&sol->dmCoeff); CHKERRQ(ierr);
  
  ierr = SNESDestroy(&snes); CHKERRQ(ierr);
  
  //ierr = PetscFree(grd); CHKERRQ(ierr);
  //ierr = PetscFree(usr); CHKERRQ(ierr);
  ierr = PetscFree(sol); CHKERRQ(ierr);
  
  // Finalize main
  ierr = PetscFinalize();
  return ierr;
}

// ---------------------------------------
// ---------------------------------------
// OTHER FUNCTIONS
// ---------------------------------------
// ---------------------------------------

// ---------------------------------------
// CreateSystem
// ---------------------------------------
PetscErrorCode CreateSystem(SolverCtx *sol)
{
  PetscInt       sz;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  
  // Create global vectors
  ierr = DMCreateGlobalVector(sol->dmPV,&sol->x); CHKERRQ(ierr); // solution
  ierr = VecDuplicate(sol->x, &sol->r     );      CHKERRQ(ierr); // residual
  ierr = VecDuplicate(sol->x, &sol->xguess);      CHKERRQ(ierr); // initial guess for solver
  
  // Get global vector size
  ierr = VecGetSize(sol->x, &sz); CHKERRQ(ierr);

  // Create Jacobian
  ierr = DMCreateMatrix(sol->dmPV, &sol->J); CHKERRQ(ierr);

  // Matrix preallocation
  ierr = JacobianMatrixPreallocation(sol); CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

// ---------------------------------------
// JacobianMatrixPreallocation
// ---------------------------------------
PetscErrorCode JacobianMatrixPreallocation(SolverCtx *sol)
{
  PetscInt       Nx, Nz;               // global variables
  PetscInt       i, j, sx, sz, nx, nz; // local variables
  Mat            preallocator = NULL;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  // Assign pointers and other variables
  Nx = sol->grd->nx;
  Nz = sol->grd->nz;

  // MatPreallocate begin
  ierr = MatPreallocatePhaseBegin(sol->J, &preallocator); CHKERRQ(ierr);
  
  // Get local domain
  ierr = DMStagGetCorners(sol->dmPV, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  // Get non-zero pattern for preallocator - Loop over all local elements 
  PetscInt      nEntries;
  PetscScalar   vv[23];
  DMStagStencil row, col[23];
  
  // Zero entries
  ierr = PetscMemzero(vv,sizeof(PetscScalar)*23); CHKERRQ(ierr);

  for (j = sz; j<sz+nz; ++j) {
    for (i = sx; i<sx+nx; ++i) {
    
      // Top boundary velocity Dirichlet
      if (j == Nz-1) {
        nEntries = 1;
        row.i    = i; row.j = j; row.loc = UP; row.c = 0;
        col[0]   = row;
        
        ierr = DMStagMatSetValuesStencil(sol->dmPV,preallocator,1,&row,nEntries,col,vv,INSERT_VALUES); CHKERRQ(ierr);
      }
      
      // Bottom boundary velocity Dirichlet
      if (j == 0) {
        nEntries = 1;
        row.i    = i; row.j = j; row.loc = DOWN; row.c = 0;
        col[0]   = row;
        
        ierr = DMStagMatSetValuesStencil(sol->dmPV,preallocator,1,&row,nEntries,col,vv,INSERT_VALUES); CHKERRQ(ierr);
      } 

      // Right Boundary velocity Dirichlet
      if (i == Nx-1) {
        nEntries = 1;
        row.i    = i; row.j = j; row.loc = RIGHT; row.c = 0;
        col[0]   = row;
        ierr = DMStagMatSetValuesStencil(sol->dmPV,preallocator,1,&row,nEntries,col,vv,INSERT_VALUES);CHKERRQ(ierr);
      }
      
      // Left velocity Dirichlet
      if (i == 0) {
        nEntries = 1;
        row.i = i; row.j = j; row.loc = LEFT; row.c = 0;
        col[0]   = row;
        ierr = DMStagMatSetValuesStencil(sol->dmPV,preallocator,1,&row,nEntries,col,vv,INSERT_VALUES);CHKERRQ(ierr);
      } 

      // Continuity equation (P) : V_x + V_z = 0
      nEntries = 5;
      row.i = i; row.j = j; row.loc = ELEMENT; row.c = 0;
      col[0].i = i; col[0].j = j; col[0].loc = LEFT;    col[0].c = 0;
      col[1].i = i; col[1].j = j; col[1].loc = RIGHT;   col[1].c = 0;
      col[2].i = i; col[2].j = j; col[2].loc = DOWN;    col[2].c = 0;
      col[3].i = i; col[3].j = j; col[3].loc = UP;      col[3].c = 0;
      col[4] = row;
      ierr = DMStagMatSetValuesStencil(sol->dmPV,preallocator,1,&row,nEntries,col,vv,INSERT_VALUES); CHKERRQ(ierr);

      // X-momentum equation : (u_xx + u_zz) - p_x = rhog^x (rhog_x=0)
      if (i > 0) {
          nEntries  = 23;
          row.i = i; row.j = j; row.loc = LEFT; row.c = 0;

          // Get stencil entries
          ierr = XMomentumStencil(i,j,Nx,Nz,col); CHKERRQ(ierr);

        // Insert X-momentum entries
        ierr = DMStagMatSetValuesStencil(sol->dmPV,preallocator,1,&row,nEntries,col,vv,INSERT_VALUES); CHKERRQ(ierr);
      }

      // Z-momentum equation : (u_xx + u_zz) - p_z = rhog^z
      if (j > 0) {
        nEntries = 23;
        row.i    = i  ; row.j     = j  ; row.loc     = DOWN;     row.c     = 0;
        
        // Get stencil entries
        ierr = ZMomentumStencil(i,j,Nx,Nz,col); CHKERRQ(ierr);

        // Insert Z-momentum entries
        ierr = DMStagMatSetValuesStencil(sol->dmPV,preallocator,1,&row,nEntries,col,vv,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  
  // Push the non-zero pattern defined within preallocator into the Jacobian
  ierr = MatPreallocatePhaseEnd(sol->J); CHKERRQ(ierr);
  
  // View preallocated struct of the Jacobian
  //ierr = MatView(sol->J,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

  // Matrix assembly
  ierr = MatAssemblyBegin(sol->J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (sol->J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
// ---------------------------------------
// XMomentumStencil
// ---------------------------------------
PetscErrorCode XMomentumStencil(PetscInt i,PetscInt j, PetscInt Nx, PetscInt Nz, DMStagStencil *col)
{
  PetscFunctionBegin;

  // Original stencil entries
  col[0].i  = i  ; col[0].j  = j  ; col[0].loc  = LEFT;    col[0].c   = 0;
  col[1].i  = i  ; col[1].j  = j-1; col[1].loc  = LEFT;    col[1].c   = 0;
  col[2].i  = i  ; col[2].j  = j+1; col[2].loc  = LEFT;    col[2].c   = 0;
  col[3].i  = i-1; col[3].j  = j  ; col[3].loc  = LEFT;    col[3].c   = 0;
  col[4].i  = i  ; col[4].j  = j  ; col[4].loc  = RIGHT;   col[4].c   = 0;
  col[5].i  = i-1; col[5].j  = j  ; col[5].loc  = DOWN;    col[5].c   = 0;
  col[6].i  = i  ; col[6].j  = j  ; col[6].loc  = DOWN;    col[6].c   = 0;
  col[7].i  = i-1; col[7].j  = j  ; col[7].loc  = UP;      col[7].c   = 0;
  col[8].i  = i  ; col[8].j  = j  ; col[8].loc  = UP;      col[8].c   = 0;
  col[9].i  = i-1; col[9].j  = j  ; col[9].loc  = ELEMENT; col[9].c   = 0;
  col[10].i = i  ; col[10].j = j  ; col[10].loc = ELEMENT; col[10].c  = 0;

  // Nonlinear stencil extension (strain rates)
  col[11].i = i-1; col[11].j = j+1; col[11].loc = UP;      col[11].c  = 0;
  col[12].i = i  ; col[12].j = j+1; col[12].loc = UP;      col[12].c  = 0;
  col[13].i = i-2; col[13].j = j  ; col[13].loc = UP;      col[13].c  = 0;
  col[14].i = i-2; col[14].j = j  ; col[14].loc = DOWN;    col[14].c  = 0;
  col[15].i = i+1; col[15].j = j  ; col[15].loc = UP;      col[15].c  = 0;
  col[16].i = i+1; col[16].j = j  ; col[16].loc = DOWN;    col[16].c  = 0;
  col[17].i = i-1; col[17].j = j-1; col[17].loc = DOWN;    col[17].c  = 0;
  col[18].i = i  ; col[18].j = j-1; col[18].loc = DOWN;    col[18].c  = 0;

  col[19].i = i-1; col[19].j = j+1; col[19].loc = LEFT;    col[19].c  = 0;
  col[20].i = i  ; col[20].j = j+1; col[20].loc = RIGHT;   col[20].c  = 0;
  col[21].i = i-1; col[21].j = j-1; col[21].loc = LEFT;    col[21].c  = 0;
  col[22].i = i  ; col[22].j = j-1; col[22].loc = RIGHT;   col[22].c  = 0;

  // Boundaries - correct for missing values
  if (i == 1) {
    col[13] = col[7]; col[14] = col[5];
  } else if (i == Nx-1) {
    col[15] = col[8]; col[16] = col[6];
  }

  if (j == 0) {
    col[1] = col[0]; col[17] = col[5]; col[18] = col[6]; col[21] = col[3]; col[22] = col[4];
  } else if (j == Nz-1) {
    col[2] = col[0]; col[11] = col[7]; col[12] = col[8]; col[19] = col[3]; col[20] = col[4];
  }

  PetscFunctionReturn(0);
}
// ---------------------------------------
// ZMomentumStencil
// ---------------------------------------
PetscErrorCode ZMomentumStencil(PetscInt i,PetscInt j, PetscInt Nx, PetscInt Nz, DMStagStencil *col)
{
  PetscFunctionBegin;

  // Original stencil entries
  col[0].i = i  ; col[0].j  = j  ; col[0].loc  = DOWN;     col[0].c  = 0;
  col[1].i = i  ; col[1].j  = j-1; col[1].loc  = DOWN;     col[1].c  = 0;
  col[2].i = i  ; col[2].j  = j+1; col[2].loc  = DOWN;     col[2].c  = 0;
  col[3].i = i-1; col[3].j  = j  ; col[3].loc  = DOWN;     col[3].c  = 0;
  col[4].i = i+1; col[4].j  = j  ; col[4].loc  = DOWN;     col[4].c  = 0;
  col[5].i = i  ; col[5].j  = j-1; col[5].loc  = LEFT;     col[5].c  = 0;
  col[6].i = i  ; col[6].j  = j-1; col[6].loc  = RIGHT;    col[6].c  = 0;
  col[7].i = i  ; col[7].j  = j  ; col[7].loc  = LEFT;     col[7].c  = 0;
  col[8].i = i  ; col[8].j  = j  ; col[8].loc  = RIGHT;    col[8].c  = 0;
  col[9].i = i  ; col[9].j  = j-1; col[9].loc  = ELEMENT;  col[9].c  = 0;
  col[10].i = i ; col[10].j = j  ; col[10].loc = ELEMENT;  col[10].c = 0;

  // Nonlinear stencil extension (strain rates)
  col[11].i = i  ; col[11].j = j+1; col[11].loc = LEFT;    col[11].c  = 0;
  col[12].i = i  ; col[12].j = j+1; col[12].loc = RIGHT;   col[12].c  = 0;
  col[13].i = i-1; col[13].j = j  ; col[13].loc = LEFT;    col[13].c  = 0;
  col[14].i = i+1; col[14].j = j  ; col[14].loc = RIGHT;   col[14].c  = 0;
  col[15].i = i-1; col[15].j = j-1; col[15].loc = LEFT;    col[15].c  = 0;
  col[16].i = i+1; col[16].j = j-1; col[16].loc = RIGHT;   col[16].c  = 0;
  col[17].i = i  ; col[17].j = j-2; col[17].loc = LEFT;    col[17].c  = 0;
  col[18].i = i  ; col[18].j = j-2; col[18].loc = RIGHT;   col[18].c  = 0;

  col[19].i = i-1; col[19].j = j  ; col[19].loc = UP;      col[19].c  = 0;
  col[20].i = i+1; col[20].j = j  ; col[20].loc = UP;      col[20].c  = 0;
  col[21].i = i-1; col[21].j = j-1; col[21].loc = DOWN;    col[21].c  = 0;
  col[22].i = i+1; col[22].j = j-1; col[22].loc = DOWN;    col[22].c  = 0;

  // Boundaries - correct for missing values
  if (j == 1) {
    col[17] = col[5]; col[18] = col[6];
  } else if (j == Nz-1) {
    col[11] = col[7]; col[12] = col[8];
  }

  if (i == 0) {
    col[3] = col[0]; col[13] = col[7]; col[15] = col[5]; col[19] = col[2]; col[21] = col[1];
  } else if (i == Nx-1) {
    col[4] = col[0]; col[14] = col[8]; col[16] = col[6]; col[20] = col[2]; col[22] = col[1];
  }

  PetscFunctionReturn(0);
}

// ---------------------------------------
// InitializeModel
// ---------------------------------------
PetscErrorCode InitializeModel(SolverCtx *sol)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  
  // Different model setups
  if        (sol->usr->mtype == SOLCX){
    ierr = InitializeModel_SolCx(sol); CHKERRQ(ierr);
  } else if (sol->usr->mtype == MOR  ){
    //ierr = InitializeModel_MOR(sol); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}
// ---------------------------------------
// InitializeModel_SolCx
// ---------------------------------------
PetscErrorCode InitializeModel_SolCx(SolverCtx *sol)
{
  PetscInt       i, j, sx, sz, nx, nz;
  Vec            coordLocal;
  DM             dmCoord;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  
  // Access vector with density
  ierr = DMCreateGlobalVector(sol->dmCoeff, &sol->coeff); CHKERRQ(ierr);
  
  // Get domain corners
  ierr = DMStagGetCorners(sol->dmCoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  
  // Get coordinates
  ierr = DMGetCoordinatesLocal(sol->dmCoeff, &coordLocal); CHKERRQ(ierr);
  ierr = DMGetCoordinateDM    (sol->dmCoeff, &dmCoord   ); CHKERRQ(ierr);
  
  // Loop over local domain
  for (j = sz; j < sz+nz; ++j) {
    for (i = sx; i <sx+nx; ++i) {

      DMStagStencil point, pointCoordx, pointCoordz;
      PetscScalar   x, z, rho;
        
      // Get coordinate of rho point
      point.i = i; point.j = j; point.loc = ELEMENT; point.c = 0;

      pointCoordx = point; pointCoordx.c = 0;
      pointCoordz = point; pointCoordz.c = 1;

      ierr = DMStagVecGetValuesStencil(dmCoord,coordLocal,1,&pointCoordx,&x); CHKERRQ(ierr);
      ierr = DMStagVecGetValuesStencil(dmCoord,coordLocal,1,&pointCoordz,&z); CHKERRQ(ierr);
        
      // Set density value
      rho  = PetscSinScalar(PETSC_PI*z) * PetscCosScalar(PETSC_PI*x); //sin(pi*z)*cos(pi*x); 
      ierr = DMStagVecSetValuesStencil(sol->dmCoeff,sol->coeff,1,&point,&rho,INSERT_VALUES); CHKERRQ(ierr);
    }
  }
  
  // Vector Assembly and Restore local vector
  ierr = VecAssemblyBegin(sol->coeff); CHKERRQ(ierr);
  ierr = VecAssemblyEnd  (sol->coeff); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
// ---------------------------------------
// FormFunctionPV
// ---------------------------------------
PetscErrorCode FormFunctionPV(SNES snes, Vec x, Vec f, void *ctx)
{
  SolverCtx      *sol = (SolverCtx*) ctx;
  PetscInt       i, j, Nx, Nz, sx, sz, nx, nz;
  Vec            xlocal, coefflocal,flocal;
  PetscInt       nEntries, idx;
  PetscScalar    ***ff;
  PetscScalar    xx[4], fval, dx, dz;
  DMStagStencil  col[4];
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Assign pointers and other variables
  Nx = sol->grd->nx;
  Nz = sol->grd->nz;
  dx = sol->grd->dx;
  dz = sol->grd->dz;

  // Get local domain
  ierr = DMStagGetCorners(sol->dmPV, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  // Map global vectors to local domain
  ierr = DMGetLocalVector(sol->dmPV, &xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (sol->dmPV, x, INSERT_VALUES, xlocal); CHKERRQ(ierr);

  // Map coefficient data to local domain
  ierr = DMGetLocalVector(sol->dmCoeff, &coefflocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (sol->dmCoeff, sol->coeff, INSERT_VALUES, coefflocal); CHKERRQ(ierr);

  // Map global vectors to local domain
  ierr = DMCreateLocalVector(sol->dmPV, &flocal); CHKERRQ(ierr);

  // Get residual array
  ierr = DMStagVecGetArrayDOF(sol->dmPV, flocal, &ff); CHKERRQ(ierr);

  // Loop over elements - Interior domain
  for (j = sz; j<sz+nz; ++j) {
    for (i = sx; i<sx+nx; ++i) {
      // 1) Continuity equation
      // Get stencil values
      nEntries = 4;
      col[0].i = i; col[0].j = j; col[0].loc = LEFT;    col[0].c = 0;
      col[1].i = i; col[1].j = j; col[1].loc = RIGHT;   col[1].c = 0;
      col[2].i = i; col[2].j = j; col[2].loc = DOWN;    col[2].c = 0;
      col[3].i = i; col[3].j = j; col[3].loc = UP;      col[3].c = 0;
      ierr = DMStagVecGetValuesStencil(sol->dmPV, xlocal, nEntries, col, xx); CHKERRQ(ierr);
      
      // Calculate residual
      fval = (xx[1]-xx[0])/dx + (xx[3]-xx[2])/dz;

      // Set residual in array
      ierr = DMStagGetLocationSlot(sol->dmPV, ELEMENT, 0, &idx); CHKERRQ(ierr);
      ff[j][i][idx] = fval;

      // 2) X-Momentum equation
      if ((i > 0) && (j > 0) && (j < Nz-1)) {
        // Get residual
        ierr = XMomentumResidual(sol, xlocal, i, j, NONE, &fval); CHKERRQ(ierr);

        // Set residual in array
        ierr = DMStagGetLocationSlot(sol->dmPV, LEFT, 0, &idx); CHKERRQ(ierr);
        ff[j][i][idx] = fval;
      }

      // 3) Z-Momentum equation
      if ((j > 0) && (i > 0) && (i < Nx-1)) {
        // Get residual
        ierr = ZMomentumResidual(sol, xlocal, coefflocal, i, j, NONE, &fval); CHKERRQ(ierr);

        // Set residual in array
        ierr = DMStagGetLocationSlot(sol->dmPV, DOWN, 0, &idx); CHKERRQ(ierr);
        ff[j][i][idx] = fval;
      }
    }
  }
  // ---------------------------------------
  // Boundary conditions
  // ---------------------------------------
  // LEFT
  i = sx;
  if (i == 0) {
    // Free slip
    if (sol->grd->bcleft == FREE_SLIP){
      for (j = sz; j<sz+nz; ++j) {
        // Vx - Dirichlet
        // Get stencil values
        col[0].i = i  ; col[0].j  = j  ; col[0].loc  = LEFT;    col[0].c   = 0;
        ierr = DMStagVecGetValuesStencil(sol->dmPV, xlocal, 1, col, xx); CHKERRQ(ierr);

        // Calculate residual
        fval = xx[0] - sol->grd->Vleft;
        
        // Set residual in array
        ierr = DMStagGetLocationSlot(sol->dmPV, LEFT, 0, &idx); CHKERRQ(ierr);
        ff[j][i][idx] = fval;
        
        // Vz - zero horizontal flux (dVz/dx = 0)
        if (j > 0) {
          // Get residual
          ierr = ZMomentumResidual(sol, xlocal, coefflocal, i, j, BCLEFT, &fval); CHKERRQ(ierr);

          // Set residual in array
          ierr = DMStagGetLocationSlot(sol->dmPV, DOWN, 0, &idx); CHKERRQ(ierr);
          ff[j][i][idx] = fval;
        }
      }
    } // add other BC
  }

  // RIGHT
  i = sx+nx-1;
  if (i == Nx-1) {
    // Free slip
    if (sol->grd->bcright == FREE_SLIP){
      for (j = sz; j<sz+nz; ++j) {
        // Vx - Dirichlet
        // Get stencil values
        col[0].i = i ; col[0].j  = j  ; col[0].loc  = RIGHT;    col[0].c   = 0;
        ierr = DMStagVecGetValuesStencil(sol->dmPV, xlocal, 1, col, xx); CHKERRQ(ierr);

        // Calculate residual
        fval = xx[0] - sol->grd->Vright;
        
        // Set residual in array
        ierr = DMStagGetLocationSlot(sol->dmPV, RIGHT, 0, &idx); CHKERRQ(ierr);
        ff[j][i][idx] = fval;

        // Vz - zero horizontal flux (dVz/dx = 0)
        if (j > 0) {
          // Get residual
          ierr = ZMomentumResidual(sol, xlocal, coefflocal, i, j, BCRIGHT, &fval); CHKERRQ(ierr);

          // Set residual in array
          ierr = DMStagGetLocationSlot(sol->dmPV, DOWN, 0, &idx); CHKERRQ(ierr);
          ff[j][i][idx] = fval;
        }
      }
    } // add other BC
  }

// DOWN
  j = sz;
  if (j == 0) {
    // Free slip
    if (sol->grd->bcdown == FREE_SLIP){
      for (i = sx; i<sx+nx; ++i) {
        // Vz - Dirichlet
        // Get stencil values
        col[0].i = i  ; col[0].j  = j ; col[0].loc  = DOWN;    col[0].c   = 0;
        ierr = DMStagVecGetValuesStencil(sol->dmPV, xlocal, 1, col, xx); CHKERRQ(ierr);

        // Calculate residual
        fval = xx[0] - sol->grd->Vdown;
        
        // Set residual in array
        ierr = DMStagGetLocationSlot(sol->dmPV, DOWN, 0, &idx); CHKERRQ(ierr);
        ff[j][i][idx] = fval;

        // Vx - zero vertical flux (dVx/dz = 0)
        if (i > 0) {
          // Get residual
          ierr = XMomentumResidual(sol, xlocal, i, j, BCDOWN, &fval); CHKERRQ(ierr);

          // Set residual in array
          ierr = DMStagGetLocationSlot(sol->dmPV, LEFT, 0, &idx); CHKERRQ(ierr);
          ff[j][i][idx] = fval;
        }
      }
    } // add other BC
  }

  // UP
  j = sz+nz-1;
  if (j == Nz-1) {
    // Free slip
    if (sol->grd->bcup == FREE_SLIP){
      for (i = sx; i<sx+nx; ++i) {
        // Vz - Dirichlet
        // Get stencil values
        col[0].i = i; col[0].j = j; col[0].loc = UP; col[0].c = 0;
        ierr = DMStagVecGetValuesStencil(sol->dmPV, xlocal, 1, col, xx); CHKERRQ(ierr);

        // Calculate residual
        fval = xx[0] - sol->grd->Vup;
        
        // Set residual in array
        ierr = DMStagGetLocationSlot(sol->dmPV, UP, 0, &idx); CHKERRQ(ierr);
        ff[j][i][idx] = fval;

        // Vx - zero vertical flux (dVx/dz = 0)
        if (i > 0) {
          // Get residual
          ierr = XMomentumResidual(sol, xlocal, i, j, BCUP, &fval); CHKERRQ(ierr);

          // Set residual in array
          ierr = DMStagGetLocationSlot(sol->dmPV, LEFT, 0, &idx); CHKERRQ(ierr);
          ff[j][i][idx] = fval;
        }
      }
    } // add other BC
  }

  // Map local to global
  ierr = DMLocalToGlobalBegin(sol->dmPV,flocal,INSERT_VALUES,f); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (sol->dmPV,flocal,INSERT_VALUES,f); CHKERRQ(ierr);

  // Restore arrays, local vectors
  ierr = DMStagVecRestoreArrayDOF(sol->dmPV,flocal,&ff); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(sol->dmCoeff,&coefflocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(sol->dmPV,   &xlocal    ); CHKERRQ(ierr);

  ierr = VecDestroy(&flocal); CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}
// ---------------------------------------
// FormFunctionPV1
// ---------------------------------------
PetscErrorCode FormFunctionPV1(SNES snes, Vec x, Vec f, void *ctx)
{
  SolverCtx      *sol = (SolverCtx*) ctx;
  PetscInt       i, j, Nx, Nz, sx, sz, nx, nz;
  Vec            xlocal, coefflocal;
  PetscInt       nEntries;
  PetscScalar    xx[11], ff, dx, dz;
  DMStagStencil  row, col[11];
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Assign pointers and other variables
  Nx = sol->grd->nx;
  Nz = sol->grd->nz;
  dx = sol->grd->dx;
  dz = sol->grd->dz;

  // Get local domain
  ierr = DMStagGetCorners(sol->dmPV, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  // Map global vectors to local domain
  ierr = DMGetLocalVector(sol->dmPV, &xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (sol->dmPV, sol->x, INSERT_VALUES, xlocal); CHKERRQ(ierr);

  // Map coefficient data to local domain
  ierr = DMGetLocalVector(sol->dmCoeff, &coefflocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (sol->dmCoeff, sol->coeff, INSERT_VALUES, coefflocal); CHKERRQ(ierr);

  // Loop over elements - Interior domain
  for (j = sz; j<sz+nz; ++j) {
    for (i = sx; i<sx+nx; ++i) {
      // 1) Continuity equation
      // Get stencil values
      nEntries = 4;
      col[0].i = i; col[0].j = j; col[0].loc = LEFT;    col[0].c = 0;
      col[1].i = i; col[1].j = j; col[1].loc = RIGHT;   col[1].c = 0;
      col[2].i = i; col[2].j = j; col[2].loc = DOWN;    col[2].c = 0;
      col[3].i = i; col[3].j = j; col[3].loc = UP;      col[3].c = 0;
      ierr = DMStagVecGetValuesStencil(sol->dmPV, xlocal, nEntries, col, xx); CHKERRQ(ierr);
      
      // Calculate residual
      ff = (xx[1]-xx[0])/dx + (xx[3]-xx[2])/dz;

      // Set function in vector
      row.i = i; row.j = j; row.loc = ELEMENT; row.c = 0;
      ierr = DMStagVecSetValuesStencil(sol->dmPV, f, 1, &row, &ff, INSERT_VALUES); CHKERRQ(ierr);

      // 2) X-Momentum equation
      if ((i > 0) && (j > 0) && (j < Nz-1)) {
        // Get residual
        ierr = XMomentumResidual(sol, xlocal, i, j, NONE, &ff); CHKERRQ(ierr);

        // Set function in vector
        row.i = i; row.j = j; row.loc = LEFT; row.c = 0;
        ierr = DMStagVecSetValuesStencil(sol->dmPV, f, 1, &row, &ff, INSERT_VALUES); CHKERRQ(ierr);
      }

      // 3) Z-Momentum equation
      if ((j > 0) && (i > 0) && (i < Nx-1)) {
        // Get residual
        ierr = ZMomentumResidual(sol, xlocal, coefflocal, i, j, NONE, &ff); CHKERRQ(ierr);

        // Set function in vector
        row.i = i; row.j = j; row.loc = DOWN; row.c = 0;
        ierr = DMStagVecSetValuesStencil(sol->dmPV, f, 1, &row, &ff, INSERT_VALUES); CHKERRQ(ierr);
      }
    }
  }
  // ---------------------------------------
  // Boundary conditions
  // ---------------------------------------
  // LEFT
  i = sx;
  if (i == 0) {
    // Free slip
    if (sol->grd->bcleft == FREE_SLIP){
      for (j = sz; j<sz+nz; ++j) {
        // Vx - Dirichlet
        // Get stencil values
        col[0].i = i  ; col[0].j  = j  ; col[0].loc  = LEFT;    col[0].c   = 0;
        ierr = DMStagVecGetValuesStencil(sol->dmPV, xlocal, 1, col, xx); CHKERRQ(ierr);

        // Calculate residual
        ff = xx[0] - sol->grd->Vleft;
        
        // Set function in vector
        row  = col[0];
        ierr = DMStagVecSetValuesStencil(sol->dmPV, f, 1, &row, &ff, INSERT_VALUES); CHKERRQ(ierr);
        
        // Vz - zero horizontal flux (dVz/dx = 0)
        if (j > 0) {
          // Get residual
          ierr = ZMomentumResidual(sol, xlocal, coefflocal, i, j, BCLEFT, &ff); CHKERRQ(ierr);

          // Set function in vector
          row.i = i; row.j = j; row.loc = DOWN; row.c = 0;
          ierr = DMStagVecSetValuesStencil(sol->dmPV, f, 1, &row, &ff, INSERT_VALUES); CHKERRQ(ierr);
        }
      }
    } // add other BC
  }

  // RIGHT
  i = sx+nx-1;
  if (i == Nx-1) {
    // Free slip
    if (sol->grd->bcright == FREE_SLIP){
      for (j = sz; j<sz+nz; ++j) {
        // Vx - Dirichlet
        // Get stencil values
        col[0].i = i ; col[0].j  = j  ; col[0].loc  = RIGHT;    col[0].c   = 0;
        ierr = DMStagVecGetValuesStencil(sol->dmPV, xlocal, 1, col, xx); CHKERRQ(ierr);

        // Calculate residual
        ff = xx[0] - sol->grd->Vright;
        
        // Set function in vector
        row  = col[0];
        ierr = DMStagVecSetValuesStencil(sol->dmPV, f, 1, &row, &ff, INSERT_VALUES); CHKERRQ(ierr);

        // Vz - zero horizontal flux (dVz/dx = 0)
        if (j > 0) {
          // Get residual
          ierr = ZMomentumResidual(sol, xlocal, coefflocal, i, j, BCRIGHT, &ff); CHKERRQ(ierr);

          // Set function in vector
          row.i = i; row.j = j; row.loc = DOWN; row.c = 0;
          ierr = DMStagVecSetValuesStencil(sol->dmPV, f, 1, &row, &ff, INSERT_VALUES); CHKERRQ(ierr);
        }
      }
    } // add other BC
  }

// DOWN
  j = sz;
  if (j == 0) {
    // Free slip
    if (sol->grd->bcdown == FREE_SLIP){
      for (i = sx; i<sx+nx; ++i) {
        // Vz - Dirichlet
        // Get stencil values
        col[0].i = i  ; col[0].j  = j ; col[0].loc  = DOWN;    col[0].c   = 0;
        ierr = DMStagVecGetValuesStencil(sol->dmPV, xlocal, 1, col, xx); CHKERRQ(ierr);

        // Calculate residual
        ff = xx[0] - sol->grd->Vdown;
        
        // Set function in vector
        row  = col[0];
        ierr = DMStagVecSetValuesStencil(sol->dmPV, f, 1, &row, &ff, INSERT_VALUES); CHKERRQ(ierr);

        // Vx - zero vertical flux (dVx/dz = 0)
        if (i > 0) {
          // Get residual
          ierr = XMomentumResidual(sol, xlocal, i, j, BCDOWN, &ff); CHKERRQ(ierr);

          // Set function in vector
          row.i = i; row.j = j; row.loc = LEFT; row.c = 0;
          ierr = DMStagVecSetValuesStencil(sol->dmPV, f, 1, &row, &ff, INSERT_VALUES); CHKERRQ(ierr);
        }
      }
    } // add other BC
  }

  // UP
  j = sz+nz-1;
  if (j == Nz-1) {
    // Free slip
    if (sol->grd->bcup == FREE_SLIP){
      for (i = sx; i<sx+nx; ++i) {
        // Vz - Dirichlet
        // Get stencil values
        col[0].i = i; col[0].j = j; col[0].loc = UP; col[0].c = 0;
        ierr = DMStagVecGetValuesStencil(sol->dmPV, xlocal, 1, col, xx); CHKERRQ(ierr);

        // Calculate residual
        ff = xx[0] - sol->grd->Vup;
        
        // Set function in vector
        row  = col[0];
        ierr = DMStagVecSetValuesStencil(sol->dmPV, f, 1, &row, &ff, INSERT_VALUES); CHKERRQ(ierr);

        // Vx - zero vertical flux (dVx/dz = 0)
        if (i > 0) {
          // Get residual
          ierr = XMomentumResidual(sol, xlocal, i, j, BCUP, &ff); CHKERRQ(ierr);

          // Set function in vector
          row.i = i; row.j = j; row.loc = LEFT; row.c = 0;
          ierr = DMStagVecSetValuesStencil(sol->dmPV, f, 1, &row, &ff, INSERT_VALUES); CHKERRQ(ierr);
        }
      }
    } // add other BC
  }

  // Restore arrays, local vectors
  ierr = DMRestoreLocalVector(sol->dmCoeff,&coefflocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(sol->dmPV,   &xlocal    ); CHKERRQ(ierr);
  
  // Assembly vector
  ierr = VecAssemblyBegin(f); CHKERRQ(ierr);
  ierr = VecAssemblyEnd  (f); CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

// ---------------------------------------
// XMomentumResidual
// ---------------------------------------
PetscErrorCode XMomentumResidual(SolverCtx *sol, Vec xlocal, PetscInt i, PetscInt j, enum LocationType loctype, PetscScalar *ff)
{
  PetscScalar    dVxdz, dVzdx, dPdx, dVxdx, ffi;
  PetscInt       nEntries = 11;
  PetscScalar    xx[11], dx, dz;
  PetscScalar    etaLeft, etaRight, etaUp, etaDown;
  DMStagStencil  col[11];
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Get variables
  dx = sol->grd->dx;
  dz = sol->grd->dz;

  // Get stencil values
  col[0].i  = i  ; col[0].j  = j  ; col[0].loc  = LEFT;    col[0].c   = 0; // Vx(i  ,j  )
  col[1].i  = i  ; col[1].j  = j-1; col[1].loc  = LEFT;    col[1].c   = 0; // Vx(i  ,j-1)
  col[2].i  = i  ; col[2].j  = j+1; col[2].loc  = LEFT;    col[2].c   = 0; // Vx(i  ,j+1)
  col[3].i  = i-1; col[3].j  = j  ; col[3].loc  = LEFT;    col[3].c   = 0; // Vx(i-1,j  )
  col[4].i  = i  ; col[4].j  = j  ; col[4].loc  = RIGHT;   col[4].c   = 0; // Vx(i+1,j  )
  col[5].i  = i-1; col[5].j  = j  ; col[5].loc  = DOWN;    col[5].c   = 0; // Vz(i-1,j-1)
  col[6].i  = i  ; col[6].j  = j  ; col[6].loc  = DOWN;    col[6].c   = 0; // Vz(i  ,j-1)
  col[7].i  = i-1; col[7].j  = j  ; col[7].loc  = UP;      col[7].c   = 0; // Vz(i-1,j  )
  col[8].i  = i  ; col[8].j  = j  ; col[8].loc  = UP;      col[8].c   = 0; // Vz(i  ,j  )
  col[9].i  = i-1; col[9].j  = j  ; col[9].loc  = ELEMENT; col[9].c   = 0; // P (i-1,j  )
  col[10].i = i  ; col[10].j = j  ; col[10].loc = ELEMENT; col[10].c  = 0; // P (i  ,j  )

  // For boundaries copy the missing stencil entry with the main DOF
  if (loctype == BCUP ) {
    // Free slip
    if (sol->grd->bcup   == FREE_SLIP) col[2] = col[0];
  } else if (loctype == BCDOWN) {
    // Free slip
    if (sol->grd->bcdown == FREE_SLIP) col[1] = col[0];
  }

  // Get values
  ierr = DMStagVecGetValuesStencil(sol->dmPV, xlocal, nEntries, col, xx); CHKERRQ(ierr);

  // Calculate second invariant and effective viscosity
  //ierr = CalcEffViscosity(sol, xlocal, i-1,j  ,CENTER, &etaLeft ); CHKERRQ(ierr);
  //ierr = CalcEffViscosity(sol, xlocal, i  ,j  ,CENTER, &etaRight); CHKERRQ(ierr);
  //ierr = CalcEffViscosity(sol, xlocal, i  ,j+1,CORNER, &etaUp   ); CHKERRQ(ierr);
  //ierr = CalcEffViscosity(sol, xlocal, i  ,j  ,CORNER, &etaDown ); CHKERRQ(ierr);

  etaLeft  = sol->usr->eta0;
  etaRight = sol->usr->eta0;
  etaUp    = sol->usr->eta0;
  etaDown  = sol->usr->eta0;

  // Calculate residual
  dPdx  = (xx[10]-xx[9])/dx;
  dVxdx = etaRight*(xx[4]-xx[0])/dx - etaLeft*(xx[0]-xx[3])/dx;
  dVxdz = etaUp   *(xx[2]-xx[0])/dz - etaDown*(xx[0]-xx[1])/dz;
  dVzdx = etaUp   *(xx[8]-xx[7])/dx - etaDown*(xx[6]-xx[5])/dx;
  ffi   = -dPdx + 2.0*dVxdx/dx + dVxdz/dz + dVzdx/dz;

  *ff = ffi;
  PetscFunctionReturn(0);
}

// ---------------------------------------
// ZMomentumResidual
// ---------------------------------------
PetscErrorCode ZMomentumResidual(SolverCtx *sol, Vec xlocal, Vec coefflocal, PetscInt i, PetscInt j, enum LocationType loctype, PetscScalar *ff)
{
  PetscScalar    dVxdz, dVzdx, dPdz, dVzdz, rhog, ffi;
  PetscInt       nEntries = 11;
  PetscScalar    xx[11], dx, dz;
  PetscScalar    etaLeft, etaRight, etaUp, etaDown, rho[2];
  DMStagStencil  col[11];
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Get variables
  dx = sol->grd->dx;
  dz = sol->grd->dz;

  // Get density values
  col[0].i = i; col[0].j = j  ; col[0].loc = ELEMENT; col[0].c = 0;
  col[1].i = i; col[1].j = j-1; col[1].loc = ELEMENT; col[1].c = 0;

  ierr = DMStagVecGetValuesStencil(sol->dmCoeff, coefflocal, 2, col, rho); CHKERRQ(ierr);
  rhog = -sol->usr->g * 0.5 * (rho[0] + rho[1]);

  // Get stencil values
  col[0].i  = i  ; col[0].j  = j  ; col[0].loc  = DOWN;    col[0].c   = 0; // Vz(i  ,j  )
  col[1].i  = i  ; col[1].j  = j  ; col[1].loc  = UP;      col[1].c   = 0; // Vz(i  ,j+1)
  col[2].i  = i  ; col[2].j  = j-1; col[2].loc  = DOWN;    col[2].c   = 0; // Vz(i  ,j-1)
  col[3].i  = i-1; col[3].j  = j  ; col[3].loc  = DOWN;    col[3].c   = 0; // Vz(i-1,j  )
  col[4].i  = i+1; col[4].j  = j  ; col[4].loc  = DOWN;    col[4].c   = 0; // Vz(i+1,j  )
  col[5].i  = i  ; col[5].j  = j  ; col[5].loc  = LEFT;    col[5].c   = 0; // Vx(i-1,j  )
  col[6].i  = i  ; col[6].j  = j  ; col[6].loc  = RIGHT;   col[6].c   = 0; // Vx(i  ,j  )
  col[7].i  = i  ; col[7].j  = j-1; col[7].loc  = LEFT;    col[7].c   = 0; // Vx(i-1,j-1)
  col[8].i  = i  ; col[8].j  = j-1; col[8].loc  = RIGHT;   col[8].c   = 0; // Vx(i  ,j-1)
  col[9].i  = i  ; col[9].j  = j  ; col[9].loc  = ELEMENT; col[9].c   = 0; // P (i  ,j  )
  col[10].i = i  ; col[10].j = j-1; col[10].loc = ELEMENT; col[10].c  = 0; // P (i  ,j-1)

  // For boundaries copy the missing stencil entry with the main DOF
  if (loctype == BCLEFT ) {
    // Free slip
    if (sol->grd->bcleft  == FREE_SLIP) col[3] = col[0];
  } else if (loctype == BCRIGHT) {
    // Free slip
    if (sol->grd->bcright == FREE_SLIP) col[4] = col[0]; 
  }

  // Get values
  ierr = DMStagVecGetValuesStencil(sol->dmPV, xlocal, nEntries, col, xx); CHKERRQ(ierr);

  // Calculate second invariant and effective viscosity
  //ierr = CalcEffViscosity(sol, xlocal, i  ,j  ,CORNER, &etaLeft ); CHKERRQ(ierr);
  //ierr = CalcEffViscosity(sol, xlocal, i+1,j  ,CORNER, &etaRight); CHKERRQ(ierr);
  //ierr = CalcEffViscosity(sol, xlocal, i  ,j  ,CENTER, &etaUp   ); CHKERRQ(ierr);
  //ierr = CalcEffViscosity(sol, xlocal, i  ,j-1,CENTER, &etaDown ); CHKERRQ(ierr);

  etaLeft  = sol->usr->eta0;
  etaRight = sol->usr->eta0;
  etaUp    = sol->usr->eta0;
  etaDown  = sol->usr->eta0;

  // Calculate residual
  dPdz  = (xx[9]-xx[10])/dz;
  dVzdz = etaUp   *(xx[1]-xx[0])/dz - etaDown *(xx[0]-xx[2])/dz;
  dVzdx = etaRight*(xx[4]-xx[0])/dx - etaLeft *(xx[0]-xx[3])/dx;
  dVxdz = etaLeft *(xx[5]-xx[7])/dz - etaRight*(xx[6]-xx[8])/dz;
  ffi   = -dPdz + 2.0*dVzdz/dz + dVzdx/dx + dVxdz/dx - rhog;

  *ff = ffi;

  PetscFunctionReturn(0);
}
// ---------------------------------------
// CalcEffViscosity
// ---------------------------------------
PetscErrorCode CalcEffViscosity(SolverCtx *sol, Vec xlocal, PetscInt i, PetscInt j, enum LocationType loctype, PetscScalar *etaeff)
{
  PetscScalar    epsII = 0.0;
  PetscScalar    inv_eta_diff, inv_eta_disl, eta;
  PetscInt       nEntries, Nx, Nz;
  PetscScalar    xx[12], dx, dz;
  DMStagStencil  col[12];
  PetscScalar    eps_xx, eps_zz, eps_xz, epsIIs2;
  PetscScalar    eps_xzi[4], eps_xxi[4], eps_zzi[4];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  
  /* Strain rate is defined in: 
      Corner - eps_xz 
      Center - eps_xx, eps_zz 
   */

  // Assign pointers and other variables
  Nx = sol->grd->nx;
  Nz = sol->grd->nz;
  dx = sol->grd->dx;
  dz = sol->grd->dz;

  /* Second invariant of strain rate in CENTER
      epsII^2 = 1/2*(eps_xx^2 + eps_zz^2 + 2*eps_xz_center^2);
      eps_xz_center - interpolated from eps_xz corners
  */ 
  if (loctype == CENTER) {
    // get all entries needed
    nEntries = 12;
    col[0].i  = i  ; col[0].j  = j+1; col[0].loc  = LEFT ; col[0].c   = 0; // Vx(i  ,j+1) LEFT
    col[1].i  = i  ; col[1].j  = j  ; col[1].loc  = LEFT ; col[1].c   = 0; // Vx(i  ,j  ) LEFT
    col[2].i  = i  ; col[2].j  = j-1; col[2].loc  = LEFT ; col[2].c   = 0; // Vx(i  ,j-1) LEFT
    col[3].i  = i  ; col[3].j  = j+1; col[3].loc  = RIGHT; col[3].c   = 0; // Vx(i  ,j+1) RIGHT
    col[4].i  = i  ; col[4].j  = j  ; col[4].loc  = RIGHT; col[4].c   = 0; // Vx(i  ,j  ) RIGHT
    col[5].i  = i  ; col[5].j  = j-1; col[5].loc  = RIGHT; col[5].c   = 0; // Vx(i  ,j-1) RIGHT
    col[6].i  = i-1; col[6].j  = j  ; col[6].loc  = UP;    col[6].c   = 0; // Vz(i-1,j  ) UP
    col[7].i  = i  ; col[7].j  = j  ; col[7].loc  = UP;    col[7].c   = 0; // Vz(i  ,j  ) UP
    col[8].i  = i+1; col[8].j  = j  ; col[8].loc  = UP;    col[8].c   = 0; // Vz(i+1,j  ) UP
    col[9].i  = i-1; col[9].j  = j  ; col[9].loc  = DOWN;  col[9].c   = 0; // Vz(i-1,j  ) DOWN
    col[10].i = i  ; col[10].j = j  ; col[10].loc = DOWN;  col[10].c  = 0; // Vz(i  ,j  ) DOWN 
    col[11].i = i+1; col[11].j = j  ; col[11].loc = DOWN;  col[11].c  = 0; // Vz(i+1,j  ) DOWN

    // Boundaries - LEFT
    if (i == 0){
      // zero flux
      if (sol->grd->bcleft == FREE_SLIP) {
        col[6] = col[7]; col[9] = col[10];
        
        // include corners
        if (j == 0){
          col[2] = col[1]; col[5] = col[4];
        } else if (j == Nz-1){
          col[0] = col[1]; col[3] = col[4];
        }
      }
    }

    // Boundaries - RIGHT
    if (i == Nx-1){
      // zero flux
      if (sol->grd->bcright == FREE_SLIP) {
        col[8] = col[7]; col[11] = col[10];

        // include corners
        if (j == 0){
          col[2] = col[1]; col[5] = col[4];
        } else if (j == Nz-1){
          col[0] = col[1]; col[3] = col[4];
        }
      }
    }

    // Boundaries - DOWN
    if ((j == 0) && (i > 0) && (i < Nx-1)){
      // zero flux
      if (sol->grd->bcdown == FREE_SLIP) {
        col[2] = col[1]; col[5] = col[4];
      }
    }

    // Boundaries - UP
    if ((j == Nz-1) && (i > 0) && (i < Nx-1)){
      // zero flux
      if (sol->grd->bcup == FREE_SLIP) {
        col[0] = col[1]; col[3] = col[4];
      }
    }
    
    ierr = DMStagVecGetValuesStencil(sol->dmPV, xlocal, nEntries, col, xx); CHKERRQ(ierr);

    // Calculate strain rates
    eps_xx = (xx[4]-xx[1] )/dx;
    eps_zz = (xx[7]-xx[10])/dz;

    eps_xzi[0] = 0.5*((xx[7] -xx[6] )/dx + (xx[0]-xx[1])/dz); 
    eps_xzi[1] = 0.5*((xx[8] -xx[7] )/dx + (xx[3]-xx[4])/dz);
    eps_xzi[2] = 0.5*((xx[10]-xx[9] )/dx + (xx[1]-xx[2])/dz);
    eps_xzi[3] = 0.5*((xx[11]-xx[10])/dx + (xx[4]-xx[5])/dz);

    // Interpolate eps_xz
    eps_xz = 0.25*(eps_xzi[0] + eps_xzi[1] + eps_xzi[2] + eps_xzi[3]);
  }

  /* Second invariant of strain rate in CORNER
      epsII^2 = 1/2*(eps_xx_corner^2 + eps_zz_corner^2 + 2*eps_xz^2);
      eps_xx_corner, eps_zz_corner - interpolated from centers
  */ 
  if (loctype == CORNER) {
    // get all entries needed
    nEntries = 12;
    col[0].i  = i-1; col[0].j  = j  ; col[0].loc  = LEFT ; col[0].c   = 0;
    col[1].i  = i  ; col[1].j  = j  ; col[1].loc  = LEFT ; col[1].c   = 0;
    col[2].i  = i  ; col[2].j  = j  ; col[2].loc  = RIGHT; col[2].c   = 0;
    col[3].i  = i-1; col[3].j  = j-1; col[3].loc  = LEFT ; col[3].c   = 0;
    col[4].i  = i  ; col[4].j  = j-1; col[4].loc  = LEFT ; col[4].c   = 0;
    col[5].i  = i  ; col[5].j  = j-1; col[5].loc  = RIGHT; col[5].c   = 0;
    col[6].i  = i-1; col[6].j  = j  ; col[6].loc  = UP;    col[6].c   = 0;
    col[7].i  = i-1; col[7].j  = j  ; col[7].loc  = DOWN;  col[7].c   = 0;
    col[8].i  = i-1; col[8].j  = j-1; col[8].loc  = DOWN;  col[8].c   = 0;
    col[9].i  = i  ; col[9].j  = j  ; col[9].loc  = UP;    col[9].c   = 0;
    col[10].i = i  ; col[10].j = j  ; col[10].loc = DOWN;  col[10].c  = 0;
    col[11].i = i  ; col[11].j = j-1; col[11].loc = DOWN;  col[11].c  = 0;

    // Boundaries - UP
    if (j == Nz){
      // zero flux
      if (sol->grd->bcup == FREE_SLIP) {
        col[0] = col[3]; col[1] = col[4]; col[2] = col[5]; col[6] = col[7]; col[9] = col[10];
      
      // include corners
        if (i == 0){
          col[0] = col[4]; col[6] = col[10]; col[3] = col[4]; col[7] = col[10]; col[8] = col[11];
        } else if (i == Nx){
          col[2] = col[4]; col[9] = col[7]; col[5] = col[4]; col[10] = col[7]; col[11] = col[8];
        }
      }
    }

    // Boundaries - DOWN
    if (j == 0){
      // zero flux
      if (sol->grd->bcdown == FREE_SLIP) {
        col[3] = col[0]; col[4] = col[1]; col[5] = col[2]; col[8] = col[7]; col[11] = col[10];

        // include corners
        if (i == 0){
          col[3] = col[1]; col[8] = col[10]; col[0] = col[1]; col[6] = col[9]; col[7] = col[11];
        } else if (i == Nx){
          col[5] = col[1]; col[11] = col[7]; col[2] = col[1]; col[9] = col[6]; col[10] = col[7];
        }
      }
    }

    // Boundaries - LEFT
    if ((i == 0) && (j > 0) && (j < Nz)) {
      // zero flux
      if (sol->grd->bcleft == FREE_SLIP) {
        col[0] = col[1]; col[3] = col[4]; col[6] = col[9]; col[7] = col[10]; col[8] = col[11];
      }
    }

    // Boundaries - RIGHT
    if ((i == 0) && (j > 0) && (j < Nz)) {
      // zero flux
      if (sol->grd->bcleft == FREE_SLIP) {
        col[2] = col[1]; col[5] = col[4]; col[9] = col[6]; col[10] = col[7]; col[11] = col[8];
      }
    }

    ierr = DMStagVecGetValuesStencil(sol->dmPV, xlocal, nEntries, col, xx); CHKERRQ(ierr);

    // Calculate strain rates
    eps_xz = 0.5*((xx[1] -xx[4] )/dx + (xx[10]-xx[7])/dz);

    eps_xxi[0] = (xx[1] -xx[0] )/dx; 
    eps_xxi[1] = (xx[2] -xx[1] )/dx; 
    eps_xxi[2] = (xx[4] -xx[3] )/dx; 
    eps_xxi[3] = (xx[5] -xx[4] )/dx;
    
    eps_zzi[0] = (xx[6] -xx[7] )/dz; 
    eps_zzi[1] = (xx[7] -xx[8] )/dz; 
    eps_zzi[2] = (xx[9] -xx[10])/dz; 
    eps_zzi[3] = (xx[10]-xx[11])/dz;

    // Interpolate eps_xx, eps_zz to corner
    eps_xx = 0.25*(eps_xxi[0] + eps_xxi[1] + eps_xxi[2] + eps_xxi[3]);
    eps_zz = 0.25*(eps_zzi[0] + eps_zzi[1] + eps_zzi[2] + eps_zzi[3]);
  }

  // EPSII squared
  epsIIs2 = 0.5*(eps_xx*eps_xx + eps_zz*eps_zz + 2.0*eps_xz*eps_xz);

  // Second invariant of strain rate
  epsII = PetscPowScalar(epsIIs2,0.5);

  // Rheology parameters
  sol->Pdiff = 1/(2.0*sol->usr->eta0);
  sol->Pdisl = 1/(2.0*sol->usr->eta0);

  // Calculate effective viscosity - should include component ratios
  inv_eta_diff = 2.0 * sol->Pdiff;

  if (sol->usr->ndisl==0) {
    eta = 1/inv_eta_diff; // linear viscosity
  } else {
    inv_eta_disl = 2.0 * PetscPowScalar(sol->Pdisl,1/sol->usr->ndisl) * PetscPowScalar(epsII,1-1/sol->usr->ndisl);
    eta = 1/(inv_eta_diff+inv_eta_disl);
  }

  etaeff = &eta;

  PetscFunctionReturn(0);
}

// ---------------------------------------
// FormInitialGuess
// ---------------------------------------
PetscErrorCode FormInitialGuess(SolverCtx *sol)
{
  PetscScalar    pval = -0.00001;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Initial guess - for first timestep
  ierr = VecSet(sol->xguess, pval); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// SolveSystem
// ---------------------------------------
PetscErrorCode SolveSystem(SNES snes, SolverCtx *sol)
{
  SNESConvergedReason reason;
  PetscInt       maxit, maxf, its;
  PetscReal      atol, rtol, stol;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Copy initial guess to solution
  ierr = VecCopy(sol->xguess,sol->x);          CHKERRQ(ierr);

  // Solve the non-linear system
  ierr = SNESSolve(snes,0,sol->x);             CHKERRQ(ierr);
  ierr = SNESGetConvergedReason(snes,&reason); CHKERRQ(ierr);
  ierr = SNESGetIterationNumber(snes,&its);    CHKERRQ(ierr);
  ierr = SNESGetTolerances(snes, &atol, &rtol, &stol, &maxit, &maxf); CHKERRQ(ierr);
  
  // Print some diagnostics
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of SNES iterations = %d\n",its);
  ierr = PetscPrintf(sol->comm,"SNES: atol = %g, rtol = %g, stol = %g, maxit = %D, maxf = %D\n",(double)atol,(double)rtol,(double)stol,maxit,maxf); CHKERRQ(ierr);

  // Analyze convergence
  if (reason<0) {
    // NOT converged
    if (reason < 0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_CONV_FAILED,"Nonlinear solve failed!"); CHKERRQ(ierr);
  } else {
    // converged - copy initial guess for next timestep
    ierr = VecCopy(sol->x,sol->xguess); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

// ---------------------------------------
// DoOutput
// ---------------------------------------
PetscErrorCode DoOutput(SolverCtx *sol)
{
  DM             dmVel,  daVel, dmEta,  daEta, daP,  daRho;
  Vec            vecVel, vaVel, vecEta, vaEta, vecP, vecRho;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  // Create a new DM and Vec for velocity
  ierr = DMStagCreateCompatibleDMStag(sol->dmPV,0,0,2,0,&dmVel); CHKERRQ(ierr);
  ierr = DMSetUp(dmVel); CHKERRQ(ierr);

  // Create a new DM and Vec for viscosity
  ierr = DMStagCreateCompatibleDMStag(sol->dmPV,0,0,1,0,&dmEta); CHKERRQ(ierr);
  ierr = DMSetUp(dmEta); CHKERRQ(ierr);
  
  // Set Coordinates
  ierr = DMStagSetUniformCoordinatesExplicit(dmVel,sol->grd->xmin,sol->grd->xmax,sol->grd->zmin,sol->grd->zmax,0.0,0.0); CHKERRQ(ierr);
  ierr = DMStagSetUniformCoordinatesExplicit(dmEta,sol->grd->xmin,sol->grd->xmax,sol->grd->zmin,sol->grd->zmax,0.0,0.0); CHKERRQ(ierr);
  
  // Create global vectors
  ierr = DMCreateGlobalVector(dmVel,&vecVel); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dmEta,&vecEta); CHKERRQ(ierr);
  
  // Loop over elements
  {
    PetscInt     i, j, sx, sz, nx, nz;
    PetscScalar  eta;
    Vec          xlocal;
    
    // Access local vector
    ierr = DMGetLocalVector(sol->dmPV,&xlocal); CHKERRQ(ierr);
    ierr = DMGlobalToLocal (sol->dmPV,sol->x,INSERT_VALUES,xlocal); CHKERRQ(ierr);
    
    // Get corners
    ierr = DMStagGetCorners(dmVel,&sx,&sz,NULL,&nx,&nz,NULL,NULL,NULL,NULL); CHKERRQ(ierr);
    
    // Loop
    for (j = sz; j < sz+nz; ++j) {
      for (i = sx; i < sx+nx; ++i) {
        DMStagStencil from[4], to[2], row;
        PetscScalar   valFrom[4], valTo[2];
        
        from[0].i = i; from[0].j = j; from[0].loc = UP;    from[0].c = 0;
        from[1].i = i; from[1].j = j; from[1].loc = DOWN;  from[1].c = 0;
        from[2].i = i; from[2].j = j; from[2].loc = LEFT;  from[2].c = 0;
        from[3].i = i; from[3].j = j; from[3].loc = RIGHT; from[3].c = 0;
        
        // Get values from stencil locations
        ierr = DMStagVecGetValuesStencil(sol->dmPV,xlocal,4,from,valFrom); CHKERRQ(ierr);
        
        // Average edge values to obtain ELEMENT values
        to[0].i = i; to[0].j = j; to[0].loc = ELEMENT; to[0].c = 0; valTo[0] = 0.5 * (valFrom[2] + valFrom[3]);
        to[1].i = i; to[1].j = j; to[1].loc = ELEMENT; to[1].c = 1; valTo[1] = 0.5 * (valFrom[0] + valFrom[1]);
        
        // Return values in new dm - averaged velocities
        ierr = DMStagVecSetValuesStencil(dmVel,vecVel,2,to,valTo,INSERT_VALUES); CHKERRQ(ierr);

        // Calculate element viscosity
        ierr = CalcEffViscosity(sol, xlocal, i, j, CENTER, &eta); CHKERRQ(ierr);

        // Return values in new dm - viscosity
        row.i = i; row.j = j; row.loc = ELEMENT; row.c = 0;
        ierr = DMStagVecSetValuesStencil(dmEta,vecEta,1,&row,&eta,INSERT_VALUES); CHKERRQ(ierr);
      }
    }
    
    // Vector assembly
    ierr = VecAssemblyBegin(vecVel); CHKERRQ(ierr);
    ierr = VecAssemblyEnd  (vecVel); CHKERRQ(ierr);

    ierr = VecAssemblyBegin(vecEta); CHKERRQ(ierr);
    ierr = VecAssemblyEnd  (vecEta); CHKERRQ(ierr);
    
    // Restore vector
      // Restore arrays, local vectors
    ierr = DMRestoreLocalVector(sol->dmPV,   &xlocal    ); CHKERRQ(ierr);
  }

  // Create individual DMDAs for sub-grids of our DMStag objects
  ierr = DMStagVecSplitToDMDA(sol->dmPV,sol->x,ELEMENT, 0,&daP,&vecP); CHKERRQ(ierr);
  ierr = PetscObjectSetName  ((PetscObject)vecP,"Pressure");         CHKERRQ(ierr);
  
  ierr = DMStagVecSplitToDMDA(dmVel, vecVel,ELEMENT,-3,&daVel,&vaVel); CHKERRQ(ierr); // note -3 : output 2 DOFs
  ierr = PetscObjectSetName  ((PetscObject)vaVel,"Velocity");          CHKERRQ(ierr);

  ierr = DMStagVecSplitToDMDA(sol->dmCoeff,sol->coeff,ELEMENT,0, &daRho, &vecRho); CHKERRQ(ierr);
  ierr = PetscObjectSetName  ((PetscObject)vecRho,"Density");                      CHKERRQ(ierr);

  ierr = DMStagVecSplitToDMDA(dmEta, vecEta,ELEMENT,0,&daEta,&vaEta); CHKERRQ(ierr);
  ierr = PetscObjectSetName  ((PetscObject)vaEta,"Eta");              CHKERRQ(ierr);

  // Dump element-based fields to a .vtr file
  {
    PetscViewer viewer;
    ierr = PetscViewerVTKOpen(PetscObjectComm((PetscObject)daVel),"stagridge_element.vtr",FILE_MODE_WRITE,&viewer); CHKERRQ(ierr);
    
    ierr = VecView(vecRho,   viewer); CHKERRQ(ierr);
    ierr = VecView(vaEta,    viewer); CHKERRQ(ierr);
    ierr = VecView(vaVel,    viewer); CHKERRQ(ierr);
    ierr = VecView(vecP,     viewer); CHKERRQ(ierr);
    
    ierr = PetscViewerDestroy  (&viewer); CHKERRQ(ierr);
  }

  // Destroy DMDAs and Vecs
  ierr = VecDestroy(&vecVel); CHKERRQ(ierr);
  ierr = VecDestroy(&vaVel ); CHKERRQ(ierr);
  ierr = VecDestroy(&vecP  ); CHKERRQ(ierr);
  ierr = VecDestroy(&vecRho); CHKERRQ(ierr);
  ierr = VecDestroy(&vecEta); CHKERRQ(ierr);
  ierr = VecDestroy(&vaEta ); CHKERRQ(ierr);
  
  ierr = DMDestroy(&dmVel  ); CHKERRQ(ierr);
  ierr = DMDestroy(&daVel  ); CHKERRQ(ierr);
  ierr = DMDestroy(&daP    ); CHKERRQ(ierr);
  ierr = DMDestroy(&daRho  ); CHKERRQ(ierr);
  ierr = DMDestroy(&dmEta  ); CHKERRQ(ierr);
  ierr = DMDestroy(&daEta  ); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}