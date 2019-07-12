#include "stagridge.h"
#include "ex43-solcx.h"

// ---------------------------------------
// Main benchmark routine
// ---------------------------------------
PetscErrorCode DoBenchmarks(SolverCtx *sol)
{
  DM           dmSolCx;
  Vec          xSolCx;

  PetscErrorCode ierr;

  PetscFunctionBegin;

  // SolCx
  if (sol->grd->mtype == SOLCX){
      ierr = CreateSolCx(sol,*dmSolCx,*xSolCx); CHKERRQ(ierr);
  } else {
      SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_USER,"No benchmark mtype specified!"); CHKERRQ(ierr);
  }

  // Calculate norms

  // Free memory

  PetscFunctionReturn(0);
}

// ---------------------------------------
// CreateSolCx analytical solution
// ---------------------------------------
PetscErrorCode CreateSolCx(SolverCtx *sol,DM *_da,Vec *_x)
{
  PetscInt       i, j, sx, sz, nx, nz, idx;
  PetscScalar    eta0, eta1, xc;
  PetscScalar    **xx;
  DM             da, cda;
  Vec            x;
  Vec            coords;

  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Get parameters
  eta0 = sol->usr->solcx_eta0;
  eta1 = sol->usr->solcx_eta1;
  xc   = 0.5;
  
  // Create identical DM as sol->dmPV for analytical solution
  ierr = DMStagCreateCompatibleDMStag(sol->dmPV, sol->grd->dofPV0, sol->grd->dofPV1, sol->grd->dofPV2, 0, &da); CHKERRQ(ierr);
  ierr = DMSetUp(da); CHKERRQ(ierr);
  
  // Set coordinates for new DM 
  ierr = DMStagSetUniformCoordinatesExplicit(da, sol->grd->xmin, sol->grd->xmax, sol->grd->zmin, sol->grd->zmax, 0.0, 0.0); CHKERRQ(ierr);
  
  // Create global vector associated with DM
  ierr = DMCreateGlobalVector(da, &x); CHKERRQ(ierr);

  // Get array associated with vector
  ierr = DMStagVecGetArrayDOF(da,x,&xx); CHKERRQ(ierr);

  // Get domain corners
  ierr = DMStagGetCorners(da, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  
  // Get coordinates
  ierr = DMGetCoordinatesLocal(da, &coords); CHKERRQ(ierr);
  ierr = DMGetCoordinateDM    (da, &cda   ); CHKERRQ(ierr);

  // Loop over local domain to calculate the SolCx analytical solution
  for (j = sz; j < sz+nz; ++j) {
    for (i = sx; i <sx+nx; ++i) {
      
      DMStagStencil  point, pointCoordx, pointCoordz;
      PetscScalar    pos[2];
      PetscReal      pressure, vel[2], total_stress[3], strain_rate[3]; // Real vs Scalar?
      
      // 1) Vx
      // Get coordinate of Vx point
      point.i = i; point.j = j; point.loc = LEFT; point.c = 0;

      pointCoordx = point; pointCoordx.c = 0;
      pointCoordz = point; pointCoordz.c = 1;

      ierr = DMStagVecGetValuesStencil(cda,coords,1,&pointCoordx,&pos[0]); CHKERRQ(ierr);
      ierr = DMStagVecGetValuesStencil(cda,coords,1,&pointCoordz,&pos[1]); CHKERRQ(ierr);

      // Calculate SolCx
      ierr = evaluate_solCx(pos,eta0,eta1,xc,1,vel,&pressure,total_stress,strain_rate); CHKERRQ(ierr);

      // Set value in xx array
      ierr = DMStagGetLocationSlot(da, LEFT, 0, &idx); CHKERRQ(ierr);
      xx[j][i][idx] = vel[0];

      if (i == Nx-1) {
        point.i = i; point.j = j; point.loc = RIGHT; point.c = 0;

        pointCoordx = point; pointCoordx.c = 0;
        pointCoordz = point; pointCoordz.c = 1;

        ierr = DMStagVecGetValuesStencil(cda,coords,1,&pointCoordx,&pos[0]); CHKERRQ(ierr);
        ierr = DMStagVecGetValuesStencil(cda,coords,1,&pointCoordz,&pos[1]); CHKERRQ(ierr);

        // Calculate SolCx
        ierr = evaluate_solCx(pos,eta0,eta1,xc,1,vel,&pressure,total_stress,strain_rate); CHKERRQ(ierr);

        // Set value in xx array
        ierr = DMStagGetLocationSlot(da, RIGHT, 0, &idx); CHKERRQ(ierr);
        xx[j][i][idx] = vel[0];
      }
      
      // 2) Vz
      // Get coordinate of Vz point
      point.i = i; point.j = j; point.loc = DOWN; point.c = 0;

      pointCoordx = point; pointCoordx.c = 0;
      pointCoordz = point; pointCoordz.c = 1;

      ierr = DMStagVecGetValuesStencil(cda,coords,1,&pointCoordx,&pos[0]); CHKERRQ(ierr);
      ierr = DMStagVecGetValuesStencil(cda,coords,1,&pointCoordz,&pos[1]); CHKERRQ(ierr);

      // Calculate SolCx
      ierr = evaluate_solCx(pos,eta0,eta1,xc,1,vel,&pressure,total_stress,strain_rate); CHKERRQ(ierr);

      // Set value in xx array
      ierr = DMStagGetLocationSlot(da, DOWN, 0, &idx); CHKERRQ(ierr);
      xx[j][i][idx] = vel[1];

      if (i == Nz-1) {
        point.i = i; point.j = j; point.loc = UP; point.c = 0;

        pointCoordx = point; pointCoordx.c = 0;
        pointCoordz = point; pointCoordz.c = 1;

        ierr = DMStagVecGetValuesStencil(cda,coords,1,&pointCoordx,&pos[0]); CHKERRQ(ierr);
        ierr = DMStagVecGetValuesStencil(cda,coords,1,&pointCoordz,&pos[1]); CHKERRQ(ierr);

        // Calculate SolCx
        ierr = evaluate_solCx(pos,eta0,eta1,xc,1,vel,&pressure,total_stress,strain_rate); CHKERRQ(ierr);

        // Set value in xx array
        ierr = DMStagGetLocationSlot(da, UP, 0, &idx); CHKERRQ(ierr);
        xx[j][i][idx] = vel[1];
      }
    
      // 3) Pressure
      // Get coordinate of Vz point
      point.i = i; point.j = j; point.loc = ELEMENT; point.c = 0;

      pointCoordx = point; pointCoordx.c = 0;
      pointCoordz = point; pointCoordz.c = 1;

      ierr = DMStagVecGetValuesStencil(cda,coords,1,&pointCoordx,&pos[0]); CHKERRQ(ierr);
      ierr = DMStagVecGetValuesStencil(cda,coords,1,&pointCoordz,&pos[1]); CHKERRQ(ierr);

      // Calculate SolCx
      ierr = evaluate_solCx(pos,eta0,eta1,xc,1,vel,&pressure,total_stress,strain_rate); CHKERRQ(ierr);

      // Set value in xx array
      ierr = DMStagGetLocationSlot(da, ELEMENT, 0, &idx); CHKERRQ(ierr);
      xx[j][i][idx] = pressure;
    }
  }

  // Restore array
  ierr = DMStagVecRestoreArrayDOF(da,x,&xx); CHKERRQ(ierr);

  // Assign pointers
  *_da = da;
  *_x  = x;
  
  PetscFunctionReturn(0);
}