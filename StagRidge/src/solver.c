#include "stagridge.h"

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
// CreateSystemTemp
// ---------------------------------------
PetscErrorCode CreateSystemTemp(SolverCtx *sol)
{
  PetscInt       sz;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  
  // Create global vectors
  ierr = DMCreateGlobalVector(sol->dmHT,&sol->T); CHKERRQ(ierr); // solution
  ierr = VecDuplicate(sol->T, &sol->Tr     );     CHKERRQ(ierr); // residual
  ierr = VecDuplicate(sol->T, &sol->Tguess );     CHKERRQ(ierr); // initial guess for solver
  
  // Get global vector size
  ierr = VecGetSize(sol->T, &sz); CHKERRQ(ierr);

  // Create Jacobian
  ierr = DMCreateMatrix(sol->dmHT, &sol->JT); CHKERRQ(ierr);

  // Matrix preallocation
  ierr = JacobianMatrixPreallocationTemp(sol); CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

// ---------------------------------------
// JacobianMatrixPreallocationTemp
// ---------------------------------------
PetscErrorCode JacobianMatrixPreallocationTemp(SolverCtx *sol)
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
  ierr = MatPreallocatePhaseBegin(sol->JT, &preallocator); CHKERRQ(ierr);
  
  // Get local domain
  ierr = DMStagGetCorners(sol->dmHT, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  // Get non-zero pattern for preallocator - Loop over all local elements 
  PetscInt      nEntries;
  PetscScalar   vv[5];
  DMStagStencil row, col[5];
  
  // Zero entries
  ierr = PetscMemzero(vv,sizeof(PetscScalar)*5); CHKERRQ(ierr);

  for (j = sz; j<sz+nz; ++j) {
    for (i = sx; i<sx+nx; ++i) {
    
      // Boundary velocity Dirichlet
      if ((j == 0) || (j == Nz-1) || (i == 0) || (i == Nx-1)) {
        nEntries = 1;
        row.i    = i; row.j = j; row.loc = ELEMENT; row.c = 0;
        col[0]   = row;
        ierr = DMStagMatSetValuesStencil(sol->dmHT,preallocator,1,&row,nEntries,col,vv,INSERT_VALUES); CHKERRQ(ierr);
      } else {
        // Energy equation
        nEntries = 5;
        row.i = i; row.j = j; row.loc = ELEMENT; row.c = 0;
        col[0].i = i-1; col[0].j = j;   col[0].loc = ELEMENT; col[0].c = 0;
        col[1].i = i+1; col[1].j = j;   col[1].loc = ELEMENT; col[1].c = 0;
        col[2].i = i;   col[2].j = j-1; col[2].loc = ELEMENT; col[2].c = 0;
        col[3].i = i;   col[3].j = j+1; col[3].loc = ELEMENT; col[3].c = 0;
        col[4] = row;
        ierr = DMStagMatSetValuesStencil(sol->dmHT,preallocator,1,&row,nEntries,col,vv,INSERT_VALUES); CHKERRQ(ierr);
      }
    }
  }
  
  // Push the non-zero pattern defined within preallocator into the Jacobian
  ierr = MatPreallocatePhaseEnd(sol->JT); CHKERRQ(ierr);
  
  // View preallocated struct of the Jacobian
  //ierr = MatView(sol->JT,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

  // Matrix assembly
  ierr = MatAssemblyBegin(sol->JT,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (sol->JT,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

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
          nEntries  = 11;
          row.i = i; row.j = j; row.loc = LEFT; row.c = 0;

          // Get stencil entries
          ierr = XMomentumStencil(i,j,Nx,Nz,col); CHKERRQ(ierr);

        // Insert X-momentum entries
        ierr = DMStagMatSetValuesStencil(sol->dmPV,preallocator,1,&row,nEntries,col,vv,INSERT_VALUES); CHKERRQ(ierr);
      }

      // Z-momentum equation : (u_xx + u_zz) - p_z = rhog^z
      if (j > 0) {
        nEntries = 11;
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
// FormInitialGuessTemp
// ---------------------------------------
PetscErrorCode FormInitialGuessTemp(SolverCtx *sol)
{
  PetscScalar    pval = 1.0;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Initial guess - for first timestep
  ierr = VecSet(sol->Tguess, pval); CHKERRQ(ierr);

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
// SolveSystemTemp
// ---------------------------------------
PetscErrorCode SolveSystemTemp(SNES snes, SolverCtx *sol)
{
  SNESConvergedReason reason;
  PetscInt       maxit, maxf, its;
  PetscReal      atol, rtol, stol;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Copy initial guess to solution
  ierr = VecCopy(sol->Tguess,sol->T);          CHKERRQ(ierr);

  // Solve the non-linear system
  ierr = SNESSolve(snes,0,sol->T);             CHKERRQ(ierr);
  ierr = SNESGetConvergedReason(snes,&reason); CHKERRQ(ierr);
  ierr = SNESGetIterationNumber(snes,&its);    CHKERRQ(ierr);
  ierr = SNESGetTolerances(snes, &atol, &rtol, &stol, &maxit, &maxf); CHKERRQ(ierr);
  
  // Print some diagnostics
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of SNES Temp iterations = %d\n",its);
  ierr = PetscPrintf(sol->comm,"SNES Temp: atol = %g, rtol = %g, stol = %g, maxit = %D, maxf = %D\n",(double)atol,(double)rtol,(double)stol,maxit,maxf); CHKERRQ(ierr);

  // Analyze convergence
  if (reason<0) {
    // NOT converged
    if (reason < 0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_CONV_FAILED,"Nonlinear Temp solve failed!"); CHKERRQ(ierr);
  } else {
    // converged - copy initial guess for next timestep
    ierr = VecCopy(sol->T,sol->Tguess); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}