#include "stagridge.h"
#include "../tests/ex43-solcx.h"

// ---------------------------------------
// Main benchmark routine
// ---------------------------------------
PetscErrorCode DoBenchmarks(SolverCtx *sol)
{
  DM           dmAnalytic;
  Vec          xAnalytic;

  PetscErrorCode ierr;

  PetscFunctionBegin;

  // SolCx
  if (sol->grd->mtype == SOLCX){
      ierr = CreateSolCx(sol,&dmAnalytic,&xAnalytic); CHKERRQ(ierr);
  } else {
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"No benchmark mtype specified!"); CHKERRQ(ierr);
  }

  // Calculate norms
  ierr = CalculateErrorNorms(sol, dmAnalytic, xAnalytic); CHKERRQ(ierr);

  // Free memory
  ierr = DMDestroy(&dmAnalytic); CHKERRQ(ierr);
  ierr = VecDestroy(&xAnalytic); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// CreateSolCx analytical solution
// ---------------------------------------
PetscErrorCode CreateSolCx(SolverCtx *sol,DM *_da,Vec *_x)
{
  PetscInt       i, j, sx, sz, nx, nz, idx, Nx, Nz;
  PetscScalar    eta0, eta1, xc;
  PetscScalar    ***xx;
  DM             da, cda;
  Vec            x, xlocal;
  Vec            coords;

  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Get parameters
  eta0 = sol->usr->solcx_eta0;
  eta1 = sol->usr->solcx_eta1;
  xc   = 0.5;

  Nx   = sol->grd->nx;
  Nz   = sol->grd->nz;
  
  // Create identical DM as sol->dmPV for analytical solution
  ierr = DMStagCreateCompatibleDMStag(sol->dmPV, 0, 0, 3, 0, &da); CHKERRQ(ierr);
  ierr = DMSetUp(da); CHKERRQ(ierr);
  
  // Set coordinates for new DM 
  ierr = DMStagSetUniformCoordinatesExplicit(da, sol->grd->xmin, sol->grd->xmax, sol->grd->zmin, sol->grd->zmax, 0.0, 0.0); CHKERRQ(ierr);
  
  // Create local and global vector associated with DM
  ierr = DMCreateGlobalVector(da, &x     ); CHKERRQ(ierr);
  ierr = DMCreateLocalVector (da, &xlocal); CHKERRQ(ierr);

  // Get array associated with vector
  ierr = DMStagVecGetArrayDOF(da,xlocal,&xx); CHKERRQ(ierr);

  // Get domain corners
  ierr = DMStagGetCorners(da, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  
  // Get coordinates
  ierr = DMGetCoordinatesLocal(da, &coords); CHKERRQ(ierr);
  ierr = DMGetCoordinateDM    (da, &cda   ); CHKERRQ(ierr);

  // Loop over local domain to calculate the SolCx analytical solution
  for (j = sz; j < sz+nz; ++j) {
    for (i = sx; i <sx+nx; ++i) {
      
      DMStagStencil  pointx, pointz;
      PetscScalar    pos[2];
      PetscReal      pressure, vel[2], total_stress[3], strain_rate[3]; // Real vs Scalar?
      
      // Get coordinate of center point
      pointx.i = i; pointx.j = j; pointx.loc = ELEMENT; pointx.c = 0;
      pointz   = pointx; pointz.c = 1;

      ierr = DMStagVecGetValuesStencil(cda,coords,1,&pointx,&pos[0]); CHKERRQ(ierr);
      ierr = DMStagVecGetValuesStencil(cda,coords,1,&pointz,&pos[1]); CHKERRQ(ierr);

      // Calculate SolCx
      evaluate_solCx(pos,eta0,eta1,xc,1,vel,&pressure,total_stress,strain_rate);

      // Set value in xx array - Vx
      ierr = DMStagGetLocationSlot(da, ELEMENT, 0, &idx); CHKERRQ(ierr);
      xx[j][i][idx] = vel[0];

      // Set value in xx array - Vz
      ierr = DMStagGetLocationSlot(da, ELEMENT, 1, &idx); CHKERRQ(ierr);
      xx[j][i][idx] = vel[1];

      // Set value in xx array
      ierr = DMStagGetLocationSlot(da, ELEMENT, 2, &idx); CHKERRQ(ierr);
      xx[j][i][idx] = pressure;
    }
  }

  // Restore array
  ierr = DMStagVecRestoreArrayDOF(da,xlocal,&xx); CHKERRQ(ierr);

  // Map local to global
  ierr = DMLocalToGlobalBegin(da,xlocal,INSERT_VALUES,x); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (da,xlocal,INSERT_VALUES,x); CHKERRQ(ierr);

  ierr = VecDestroy(&xlocal); CHKERRQ(ierr);

  // Assign pointers
  *_da = da;
  *_x  = x;
  
  PetscFunctionReturn(0);
}

// ---------------------------------------
// CalculateErrorNorms - calculate norms for values in ELEMENT
// ---------------------------------------
PetscErrorCode CalculateErrorNorms(SolverCtx *sol,DM da,Vec x)
{
  PetscInt       i, j, sx, sz, nx, nz;
  PetscInt       N, Nx, Nz;
  PetscInt       indp, indv;
  PetscScalar    xx[5], xa[3], dx, dz;
  PetscScalar    nrm1[2], nrm2[2], nrminf[2];
  PetscScalar    *verror, *perror;
  Vec            xlocal, xalocal;
  Vec            v_err, p_err;

  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Assign pointers and other variables
  Nx = sol->grd->nx;
  Nz = sol->grd->nz;
  dx = sol->grd->dx;
  dz = sol->grd->dz;
  N  = Nx*Nz;

  // Get domain corners
  ierr = DMStagGetCorners(da, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  // Map global vectors to local domain
  ierr = DMGetLocalVector(sol->dmPV, &xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (sol->dmPV, sol->x, INSERT_VALUES, xlocal); CHKERRQ(ierr);

  ierr = DMGetLocalVector(da, &xalocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (da, x, INSERT_VALUES, xalocal); CHKERRQ(ierr);

  // Create error vectors
  ierr = VecCreate(sol->comm, &v_err); CHKERRQ(ierr);
  ierr = VecCreate(sol->comm, &p_err); CHKERRQ(ierr);

  ierr = VecSetSizes(v_err, 2*nx*nz, 2*sol->grd->nx*sol->grd->nz); CHKERRQ(ierr);
  ierr = VecSetSizes(p_err,   nx*nz,   sol->grd->nx*sol->grd->nz); CHKERRQ(ierr);

  ierr = VecSetFromOptions(v_err); CHKERRQ(ierr);
  ierr = VecSetFromOptions(p_err); CHKERRQ(ierr);
  
  ierr = VecGetArray(v_err, &verror); CHKERRQ(ierr);
  ierr = VecGetArray(p_err, &perror); CHKERRQ(ierr);
  
  // Loop over local domain to calculate ELEMENT errors
  indv = 0; indp = 0;
  for (j = sz; j < sz+nz; ++j) {
    for (i = sx; i <sx+nx; ++i) {
      
      DMStagStencil  col[5], cal[3];
      
      // Get stencil values
      col[0].i  = i  ; col[0].j  = j  ; col[0].loc  = LEFT;    col[0].c   = 0; // Vx
      col[1].i  = i  ; col[1].j  = j  ; col[1].loc  = RIGHT;   col[1].c   = 0; // Vx
      col[2].i  = i  ; col[2].j  = j  ; col[2].loc  = DOWN;    col[2].c   = 0; // Vz
      col[3].i  = i  ; col[3].j  = j  ; col[3].loc  = UP;      col[3].c   = 0; // Vz
      col[4].i  = i  ; col[4].j  = j  ; col[4].loc  = ELEMENT; col[4].c   = 0; // P

      cal[0].i  = i  ; cal[0].j  = j  ; cal[0].loc  = ELEMENT; cal[0].c   = 0; // Vx
      cal[1].i  = i  ; cal[1].j  = j  ; cal[1].loc  = ELEMENT; cal[1].c   = 1; // Vz
      cal[2].i  = i  ; cal[2].j  = j  ; cal[2].loc  = ELEMENT; cal[2].c   = 2; // P

      // Get numerical solution
      ierr = DMStagVecGetValuesStencil(sol->dmPV, xlocal, 5, col, xx); CHKERRQ(ierr);

      // Get analytical solution
      ierr = DMStagVecGetValuesStencil(da, xalocal, 3, cal, xa); CHKERRQ(ierr);

      // Calculate errors
      verror[indv] = (xx[1]+xx[0])*0.5 - xa[0]; indv++; // element Vx
      verror[indv] = (xx[3]+xx[2])*0.5 - xa[1]; indv++; // element Vz
      perror[indp] = xx[4] - xa[2]; indp++;
    }
  }
  // Restore arrays and vectors
  ierr = DMRestoreLocalVector(sol->dmPV, &xlocal ); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,        &xalocal); CHKERRQ(ierr);

  ierr = VecRestoreArray(v_err, &verror); CHKERRQ(ierr);
  ierr = VecRestoreArray(p_err, &perror); CHKERRQ(ierr);

  // Assemble vectors
  ierr = VecAssemblyBegin(v_err); CHKERRQ(ierr);
  ierr = VecAssemblyEnd  (v_err); CHKERRQ(ierr);

  ierr = VecAssemblyBegin(p_err); CHKERRQ(ierr);
  ierr = VecAssemblyEnd  (p_err); CHKERRQ(ierr);

  // Calculate norms
  ierr = VecNorm(v_err, NORM_1, &nrm1[0]); CHKERRQ(ierr);
  ierr = VecNorm(p_err, NORM_1, &nrm1[1]); CHKERRQ(ierr);

  ierr = VecNorm(v_err, NORM_2, &nrm2[0]); CHKERRQ(ierr);
  ierr = VecNorm(p_err, NORM_2, &nrm2[1]); CHKERRQ(ierr);

  ierr = VecNorm(v_err, NORM_INFINITY, &nrminf[0]); CHKERRQ(ierr);
  ierr = VecNorm(p_err, NORM_INFINITY, &nrminf[1]); CHKERRQ(ierr);

  // Print information
  PetscPrintf(sol->comm,"# --------------------------------------- #\n");
  PetscPrintf(sol->comm,"# NORMS: \n");
  PetscPrintf(sol->comm,"# Velocity: norm1 = %1.12e norm2 = %1.12e norm_inf = %1.12e\n",nrm1[0]/N,nrm2[0]/PetscSqrtScalar(N),nrminf[0]);
  PetscPrintf(sol->comm,"# Pressure: norm1 = %1.12e norm2 = %1.12e norm_inf = %1.12e\n",nrm1[1]/N,nrm2[1]/PetscSqrtScalar(N),nrminf[1]);
  PetscPrintf(sol->comm,"# Grid info: hx = %1.12e hz = %1.12e \n",dx,dz);

  // Destroy objects
  ierr = VecDestroy(&v_err); CHKERRQ(ierr);
  ierr = VecDestroy(&p_err); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}