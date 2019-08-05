static char help[] = "Solves nonlinear 2D Stokes equations and temperature \n\n";
// 2-D Single-phase Stokes model: x(i), z(j) directions.

// Run program:
// mpiexec -n 1 ./stagridge -snes_mf
// mpiexec -n 1 ./stagridge -pc_type jacobi -snes_fd
// mpiexec -n 1 ./stagridge -pc_type jacobi -nx 21 -nz 41
// mpiexec -n 1 ./stagridge -options_file <FNAME>

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
  SNES            snes;
  PetscLogDouble  start_time, end_time;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help); if (ierr) return ierr;

  // ---------------------------------------
  // Start code
  // ---------------------------------------
  ierr = PetscTime(&start_time); CHKERRQ(ierr);
 
  // ---------------------------------------
  // Load command line or input file if required
  // ---------------------------------------
  ierr = PetscOptionsInsert(PETSC_NULL,&argc,&argv,NULL); CHKERRQ(ierr);

  // ---------------------------------------
  // Input parameters
  // ---------------------------------------
  ierr = InputParameters(&sol); CHKERRQ(ierr);

  // Save input options filename
  for (int i = 1; i < argc; i++) {
    PetscBool flg;
    
    ierr = PetscStrcmp(argv[i],"-options_file",&flg); CHKERRQ(ierr);
    if (flg) { ierr = PetscStrcpy(sol->usr->fname_in, argv[i+1]); CHKERRQ(ierr); }
  }

  // Print parameters
  ierr = InputPrintData(sol); CHKERRQ(ierr);

  // ---------------------------------------
  // Create DM data structures
  // ---------------------------------------
  // create DMStag object: dmStokes(P-element,v-vertex)
  ierr = DMStagCreate2d(sol->comm, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, sol->grd->nx, sol->grd->nz, 
            PETSC_DECIDE, PETSC_DECIDE, sol->grd->dofPV0, sol->grd->dofPV1, sol->grd->dofPV2, 
            DMSTAG_STENCIL_BOX, sol->grd->stencilWidth, NULL,NULL, &sol->dmPV); CHKERRQ(ierr);

  // set dm options
  ierr = DMSetFromOptions(sol->dmPV); CHKERRQ(ierr);
  ierr = DMSetUp         (sol->dmPV); CHKERRQ(ierr);
  
  // set coordinates dmPV
  ierr = DMStagSetUniformCoordinatesExplicit(sol->dmPV, sol->grd->xmin, sol->grd->xmax, sol->grd->zmin, sol->grd->zmax, 0.0, 0.0); CHKERRQ(ierr);
  
  // create dmCoeff (for material properties like density) 
  ierr = DMStagCreateCompatibleDMStag(sol->dmPV, sol->grd->dofCf0, sol->grd->dofCf1, sol->grd->dofCf2, 0, &sol->dmCoeff); CHKERRQ(ierr);
  ierr = DMSetUp(sol->dmCoeff); CHKERRQ(ierr);
  
  // set coordinates dmCoeff
  ierr = DMStagSetUniformCoordinatesExplicit(sol->dmCoeff, sol->grd->xmin, sol->grd->xmax, sol->grd->zmin, sol->grd->zmax, 0.0, 0.0); CHKERRQ(ierr);

  // ---------------------------------------
  // Create global vectors and matrices
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
  // Benchmark Solution
  // ---------------------------------------
  if (sol->usr->tests) ierr = DoBenchmarks(sol); CHKERRQ(ierr);
  
  // ---------------------------------------
  // OUTPUT solution to file
  // ---------------------------------------
  ierr = DoOutput(sol); CHKERRQ(ierr);

  // ---------------------------------------
  // Clean Up - destroy PETSc objects
  // ---------------------------------------
  // vectors
  ierr = VecDestroy(&sol->coeff  ); CHKERRQ(ierr);
  ierr = VecDestroy(&sol->xguess ); CHKERRQ(ierr);
  ierr = VecDestroy(&sol->x      ); CHKERRQ(ierr);
  ierr = VecDestroy(&sol->r      ); CHKERRQ(ierr);

  // matrices
  ierr = MatDestroy(&sol->J      ); CHKERRQ(ierr);
  
  // DMs
  ierr = DMDestroy(&sol->dmPV   ); CHKERRQ(ierr);
  ierr = DMDestroy(&sol->dmCoeff); CHKERRQ(ierr);
  
  // snes
  ierr = SNESDestroy(&snes); CHKERRQ(ierr);

  // petscbag
  ierr = PetscBagDestroy(&sol->bag); CHKERRQ(ierr);

  // solver context
  ierr = PetscFree(sol->scal); CHKERRQ(ierr);
  ierr = PetscFree(sol->grd ); CHKERRQ(ierr);
  ierr = PetscFree(sol);       CHKERRQ(ierr);

  // ---------------------------------------
  // End code
  // ---------------------------------------
  ierr = PetscTime(&end_time); CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"# Total runtime: %g (sec) \n", end_time - start_time);
  PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");
  
  // Finalize main
  ierr = PetscFinalize();
  return ierr;
}
