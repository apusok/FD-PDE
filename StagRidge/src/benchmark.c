#include "stagridge.h"
#include "../tests/ex43-solcx.h"

// ---------------------------------------
// Main benchmark routine
// ---------------------------------------
PetscErrorCode DoBenchmarks(SolverCtx *sol)
{
  DM           dmAnalytic;
  Vec          xAnalytic;
  PetscLogDouble  start_time, end_time;

  PetscErrorCode ierr;

  PetscFunctionBegin;

  // SolCx
  if (sol->grd->mtype == SOLCX){
    PetscPrintf(sol->comm,"# --------------------------------------- #\n");
    PetscPrintf(sol->comm,"# SolCx Benchmark \n");
    ierr = CreateSolCx(sol,&dmAnalytic,&xAnalytic); CHKERRQ(ierr);
    ierr = DoOutput_SolCx(dmAnalytic,xAnalytic); CHKERRQ(ierr);
  } else {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"No benchmark mtype specified!"); CHKERRQ(ierr);
  }

  // Calculate norms
  ierr = PetscTime(&start_time); CHKERRQ(ierr);
  ierr = CalculateErrorNorms(sol, dmAnalytic, xAnalytic); CHKERRQ(ierr);
  ierr = PetscTime(&end_time); CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"# Time: %g (sec) \n", end_time - start_time);

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
  ierr = DMStagCreateCompatibleDMStag(sol->dmPV, sol->grd->dofPV0, sol->grd->dofPV1, sol->grd->dofPV2, 0, &da); CHKERRQ(ierr);
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
      
      DMStagStencil  point;
      PetscScalar    pos[2];
      PetscReal      pressure, vel[2], total_stress[3], strain_rate[3]; // Real vs Scalar?
      
      // 1) Vx
      // Get coordinate of Vx point
      point.i = i; point.j = j; point.loc = LEFT; point.c = 0;
      ierr = GetCoordinateStencilPoint(cda,coords, point, pos); CHKERRQ(ierr);

      // Calculate SolCx
      evaluate_solCx(pos,eta0,eta1,xc,1,vel,&pressure,total_stress,strain_rate);

      // Set value in xx array
      ierr = DMStagGetLocationSlot(da, LEFT, 0, &idx); CHKERRQ(ierr);
      xx[j][i][idx] = vel[0];

      if (i == Nx-1) {
        point.i = i; point.j = j; point.loc = RIGHT; point.c = 0;
        ierr = GetCoordinateStencilPoint(cda,coords, point, pos); CHKERRQ(ierr);

        // Calculate SolCx
        evaluate_solCx(pos,eta0,eta1,xc,1,vel,&pressure,total_stress,strain_rate);

        // Set value in xx array
        ierr = DMStagGetLocationSlot(da, RIGHT, 0, &idx); CHKERRQ(ierr);
        xx[j][i][idx] = vel[0];
      }
      
      // 2) Vz
      // Get coordinate of Vz point
      point.i = i; point.j = j; point.loc = DOWN; point.c = 0;
      ierr = GetCoordinateStencilPoint(cda,coords, point, pos); CHKERRQ(ierr);

      // Calculate SolCx
      evaluate_solCx(pos,eta0,eta1,xc,1,vel,&pressure,total_stress,strain_rate);

      // Set value in xx array
      ierr = DMStagGetLocationSlot(da, DOWN, 0, &idx); CHKERRQ(ierr);
      xx[j][i][idx] = vel[1];

      if (i == Nz-1) {
        point.i = i; point.j = j; point.loc = UP; point.c = 0;
        ierr = GetCoordinateStencilPoint(cda,coords, point, pos); CHKERRQ(ierr);

        // Calculate SolCx
        evaluate_solCx(pos,eta0,eta1,xc,1,vel,&pressure,total_stress,strain_rate);

        // Set value in xx array
        ierr = DMStagGetLocationSlot(da, UP, 0, &idx); CHKERRQ(ierr);
        xx[j][i][idx] = vel[1];
      }
    
      // 3) Pressure
      point.i = i; point.j = j; point.loc = ELEMENT; point.c = 0;
      ierr = GetCoordinateStencilPoint(cda,coords, point, pos); CHKERRQ(ierr);

      // Calculate SolCx
      evaluate_solCx(pos,eta0,eta1,xc,1,vel,&pressure,total_stress,strain_rate);

      // Set value in xx array
      ierr = DMStagGetLocationSlot(da, ELEMENT, 0, &idx); CHKERRQ(ierr);
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
// CalculateErrorNorms
// ---------------------------------------
PetscErrorCode CalculateErrorNorms(SolverCtx *sol,DM da,Vec x)
{
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz;
  PetscScalar    xx[5], xa[5], dx, dz, dv;
  PetscScalar    nrm[3], gnrm[3], totp, avgp, gavgp;
  Vec            xlocal, xalocal;

  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Assign pointers and other variables
  Nx = sol->grd->nx;
  Nz = sol->grd->nz;
  dx = sol->grd->dx;
  dz = sol->grd->dz;

  // Get domain corners
  ierr = DMStagGetCorners(da, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  // Map global vectors to local domain
  ierr = DMGetLocalVector(sol->dmPV, &xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (sol->dmPV, sol->x, INSERT_VALUES, xlocal); CHKERRQ(ierr);

  ierr = DMGetLocalVector(da, &xalocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (da, x, INSERT_VALUES, xalocal); CHKERRQ(ierr);

  totp = 0.0; avgp = 0.0;

  // Loop over local domain to calculate average pressure
  for (j = sz; j < sz+nz; ++j) {
    for (i = sx; i <sx+nx; ++i) {
      PetscScalar    p;
      DMStagStencil  col;
      
      // Get stencil values
      col.i  = i  ; col.j  = j  ; col.loc  = ELEMENT; col.c   = 0; // P

      // Get numerical solution
      ierr = DMStagVecGetValuesStencil(sol->dmPV, xlocal, 1, &col, &p); CHKERRQ(ierr);

      // Average pressure
      totp += p;
    }
  }
  // Collect data 
  ierr = MPI_Allreduce(&totp, &gavgp, 1, MPI_DOUBLE, MPI_SUM, sol->comm); CHKERRQ(ierr);
  avgp = gavgp/Nx/Nz;
  
  // Initialize norms
  nrm[0] = 0.0; nrm[1] = 0.0; nrm[2] = 0.0;
  dv = dx*dz;

  // Loop over local domain to calculate ELEMENT errors
  for (j = sz; j < sz+nz; ++j) {
    for (i = sx; i <sx+nx; ++i) {
      
      PetscScalar    ve[4], pe;
      DMStagStencil  col[5];
      
      // Get stencil values
      col[0].i  = i  ; col[0].j  = j  ; col[0].loc  = LEFT;    col[0].c   = 0; // Vx
      col[1].i  = i  ; col[1].j  = j  ; col[1].loc  = RIGHT;   col[1].c   = 0; // Vx
      col[2].i  = i  ; col[2].j  = j  ; col[2].loc  = DOWN;    col[2].c   = 0; // Vz
      col[3].i  = i  ; col[3].j  = j  ; col[3].loc  = UP;      col[3].c   = 0; // Vz
      col[4].i  = i  ; col[4].j  = j  ; col[4].loc  = ELEMENT; col[4].c   = 0; // P

      // Get numerical solution
      ierr = DMStagVecGetValuesStencil(sol->dmPV, xlocal, 5, col, xx); CHKERRQ(ierr);

      // Get analytical solution
      ierr = DMStagVecGetValuesStencil(da, xalocal, 5, col, xa); CHKERRQ(ierr);

      // Calculate errors
      ve[0] = PetscAbsScalar(xx[0]-xa[0]); // Left
      ve[1] = PetscAbsScalar(xx[1]-xa[1]); // Right
      ve[2] = PetscAbsScalar(xx[2]-xa[2]); // Down
      ve[3] = PetscAbsScalar(xx[3]-xa[3]); // Up
      pe    = PetscAbsScalar(xx[4]-avgp-xa[4]); // normalized pressure 

      //PetscPrintf(sol->comm,"# [%d,%d]Vx: ve_left = %1.12e ve_right = %1.12e\n",i,j,ve[0],ve[1]);
      //PetscPrintf(sol->comm,"# [%d,%d]Vz: ve_down = %1.12e ve_up = %1.12e \n",i,j,ve[2],ve[3]);
      //PetscPrintf(sol->comm,"# [%d,%d]P: pe = %1.12e \n",i,j,pe);

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
  ierr = MPI_Allreduce(&nrm, &gnrm, 3, MPI_DOUBLE, MPI_SUM, sol->comm); CHKERRQ(ierr);

  // Restore arrays and vectors
  ierr = DMRestoreLocalVector(sol->dmPV, &xlocal ); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,        &xalocal); CHKERRQ(ierr);

  // Print information
  PetscPrintf(sol->comm,"# --------------------------------------- #\n");
  PetscPrintf(sol->comm,"# NORMS: \n");
  PetscPrintf(sol->comm,"# Velocity: norm1 = %1.12e norm1x = %1.12e norm1z = %1.12e \n",gnrm[0]+gnrm[1],gnrm[0],gnrm[1]);
  PetscPrintf(sol->comm,"# Pressure: norm1 = %1.12e\n",gnrm[2]);
  PetscPrintf(sol->comm,"# Grid info: hx = %1.12e hz = %1.12e \n",dx,dz);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// DoOutput_SolCx
// ---------------------------------------
PetscErrorCode DoOutput_SolCx(DM da,Vec x)
{
  DM             dmVel,  daVel, daP;
  Vec            vecVel, vaVel, vecP;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  // Create a new DM and Vec for velocity
  ierr = DMStagCreateCompatibleDMStag(da,0,0,2,0,&dmVel); CHKERRQ(ierr);
  ierr = DMSetUp(dmVel); CHKERRQ(ierr);

  // Set Coordinates
  ierr = DMStagSetUniformCoordinatesExplicit(dmVel,0.0,1.0,0.0,1.0,0.0,0.0); CHKERRQ(ierr);

  // Create global vectors
  ierr = DMCreateGlobalVector(dmVel,&vecVel); CHKERRQ(ierr);
  
  // Loop over elements
  {
    PetscInt     i, j, sx, sz, nx, nz;
    Vec          xlocal;
    
    // Access local vector
    ierr = DMGetLocalVector(da,&xlocal); CHKERRQ(ierr);
    ierr = DMGlobalToLocal (da,x,INSERT_VALUES,xlocal); CHKERRQ(ierr);
    
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
        ierr = DMStagVecGetValuesStencil(da,xlocal,4,from,valFrom); CHKERRQ(ierr);
        
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
    ierr = DMRestoreLocalVector(da, &xlocal); CHKERRQ(ierr);
  }

  // Create individual DMDAs for sub-grids of our DMStag objects
  ierr = DMStagVecSplitToDMDA(da,x,ELEMENT, 0,&daP,&vecP); CHKERRQ(ierr);
  ierr = PetscObjectSetName  ((PetscObject)vecP,"Pressure");         CHKERRQ(ierr);
  
  ierr = DMStagVecSplitToDMDA(dmVel, vecVel,ELEMENT,-3,&daVel,&vaVel); CHKERRQ(ierr); // note -3 : output 2 DOFs
  ierr = PetscObjectSetName  ((PetscObject)vaVel,"Velocity");          CHKERRQ(ierr);

  // Dump element-based fields to a .vtr file
  {
    PetscViewer viewer;

    // Warning: is being output as Point Data instead of Cell Data - the grid is shifted to be in the center points.
    ierr = PetscViewerVTKOpen(PetscObjectComm((PetscObject)daVel),"solcx_analytic.vtr",FILE_MODE_WRITE,&viewer); CHKERRQ(ierr);
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