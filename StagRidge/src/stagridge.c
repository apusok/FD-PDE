static char help[] = "Solves nonlinear 2D Stokes equations and temperature \n\n";
// 2-D Single-phase Stokes model: x(i), z(j) directions.

// Run program:
// mpiexec -n 1 ./stagridge -snes_mf
// mpiexec -n 1 ./stagridge -pc_type jacobi -snes_fd
// mpiexec -n 1 ./stagridge -pc_type jacobi -nx 21 -nz 41

#include "stagridge.h"

// ---------------------------------------
// Main function
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "main"
int main (int argc,char **argv)
{
  PetscErrorCode  ierr;
  SolverCtx       *sol;
  UsrData         usr;
  GridData        grd;
  SNES            snes;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help); if (ierr) return ierr;

  // allocate memory to application context
  ierr = PetscMalloc1(1, &sol); CHKERRQ(ierr);
  
  // ---------------------------------------
  /* Input - Set Parameters
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
  
  ierr = PetscFree(sol); CHKERRQ(ierr);
  
  // Finalize main
  ierr = PetscFinalize();
  return ierr;
}