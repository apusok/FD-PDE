#include "stagridge.h"
#include "../tests/ex43-solcx.h"
#include "../tests/cornerflow.h"


// ---------------------------------------
// Main benchmark routine
// ---------------------------------------
PetscErrorCode DoBenchmarks(SolverCtx *sol)
{
  DM           dmAnalytic;
  Vec          xAnalytic;

  PetscErrorCode ierr;
  PetscFunctionBegin;

  // Benchmarks - should be modified
  if ((sol->grd->mtype == SOLCX) || (sol->grd->mtype == SOLCX_EFF)){
    PetscPrintf(sol->comm,"# --------------------------------------- #\n");
    PetscPrintf(sol->comm,"# SolCx Benchmark \n");
    ierr = CreateSolCx(sol,&dmAnalytic,&xAnalytic); CHKERRQ(ierr);
    ierr = DoOutput_Analytic(sol,dmAnalytic,xAnalytic); CHKERRQ(ierr);
    ierr = CalculateErrorNorms(sol, dmAnalytic, xAnalytic); CHKERRQ(ierr);

  } else if (sol->grd->mtype == MOR){
    PetscPrintf(sol->comm,"# --------------------------------------- #\n");
    PetscPrintf(sol->comm,"# Corner flow (MOR) Benchmark \n");
    ierr = CreateMORAnalytic(sol,&dmAnalytic,&xAnalytic); CHKERRQ(ierr);
    ierr = DoOutput_Analytic(sol,dmAnalytic,xAnalytic); CHKERRQ(ierr);
    ierr = CalculateErrorNorms(sol, dmAnalytic, xAnalytic); CHKERRQ(ierr);

  } else if (sol->grd->mtype == LAPLACE){
    PetscPrintf(sol->comm,"# --------------------------------------- #\n");
    PetscPrintf(sol->comm,"# LAPLACE (diffusion) Benchmark \n");
    ierr = CreateLaplaceAnalytic(sol,&dmAnalytic,&xAnalytic); CHKERRQ(ierr);
    ierr = DoOutputTemp_Analytic(sol,dmAnalytic,xAnalytic); CHKERRQ(ierr);
    ierr = CalculateErrorNormsTemp(sol, dmAnalytic, xAnalytic); CHKERRQ(ierr);

  } else {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"No benchmark mtype specified!"); CHKERRQ(ierr);
  }

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
      ierr = GetCoordinatesStencil(cda, coords, 1, &point, &pos[0], &pos[1]); CHKERRQ(ierr);

      // Calculate SolCx
      evaluate_solCx(pos,eta0,eta1,xc,1,vel,&pressure,total_stress,strain_rate);

      // Set value in xx array
      ierr = DMStagGetLocationSlot(da, LEFT, 0, &idx); CHKERRQ(ierr);
      xx[j][i][idx] = vel[0];

      if (i == Nx-1) {
        point.i = i; point.j = j; point.loc = RIGHT; point.c = 0;
        ierr = GetCoordinatesStencil(cda, coords, 1, &point, &pos[0], &pos[1]); CHKERRQ(ierr);

        // Calculate SolCx
        evaluate_solCx(pos,eta0,eta1,xc,1,vel,&pressure,total_stress,strain_rate);

        // Set value in xx array
        ierr = DMStagGetLocationSlot(da, RIGHT, 0, &idx); CHKERRQ(ierr);
        xx[j][i][idx] = vel[0];
      }
      
      // 2) Vz
      // Get coordinate of Vz point
      point.i = i; point.j = j; point.loc = DOWN; point.c = 0;
      ierr = GetCoordinatesStencil(cda, coords, 1, &point, &pos[0], &pos[1]); CHKERRQ(ierr);

      // Calculate SolCx
      evaluate_solCx(pos,eta0,eta1,xc,1,vel,&pressure,total_stress,strain_rate);

      // Set value in xx array
      ierr = DMStagGetLocationSlot(da, DOWN, 0, &idx); CHKERRQ(ierr);
      xx[j][i][idx] = vel[1];

      if (j == Nz-1) {
        point.i = i; point.j = j; point.loc = UP; point.c = 0;
        ierr = GetCoordinatesStencil(cda, coords, 1, &point, &pos[0], &pos[1]); CHKERRQ(ierr);

        // Calculate SolCx
        evaluate_solCx(pos,eta0,eta1,xc,1,vel,&pressure,total_stress,strain_rate);

        // Set value in xx array
        ierr = DMStagGetLocationSlot(da, UP, 0, &idx); CHKERRQ(ierr);
        xx[j][i][idx] = vel[1];
      }
    
      // 3) Pressure
      point.i = i; point.j = j; point.loc = ELEMENT; point.c = 0;
      ierr = GetCoordinatesStencil(cda, coords, 1, &point, &pos[0], &pos[1]); CHKERRQ(ierr);

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

  // Loop over local domain to calculate average pressure
  totp = 0.0; avgp = 0.0;
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

  if (sol->grd->mtype==MOR){
    avgp = 0.0;
  }

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
// DoOutput_Analytic
// ---------------------------------------
PetscErrorCode DoOutput_Analytic(SolverCtx *sol, DM da,Vec x)
{
  DM             dmVel,  daVel, daP;
  Vec            vecVel, vaVel, vecP;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  // Create a new DM and Vec for velocity
  ierr = DMStagCreateCompatibleDMStag(da,0,0,2,0,&dmVel); CHKERRQ(ierr);
  ierr = DMSetUp(dmVel); CHKERRQ(ierr);

  // Set Coordinates
  ierr = DMStagSetUniformCoordinatesExplicit(dmVel,sol->grd->xmin, sol->grd->xmax, sol->grd->zmin, sol->grd->zmax, 0.0, 0.0); CHKERRQ(ierr);

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
    ierr = PetscViewerVTKOpen(PetscObjectComm((PetscObject)daVel),"analytic_solution.vtr",FILE_MODE_WRITE,&viewer); CHKERRQ(ierr);
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
// Create MOR analytical solution (corner flow)
// ---------------------------------------
PetscErrorCode CreateMORAnalytic(SolverCtx *sol,DM *_da,Vec *_x)
{
  PetscInt       i, j, sx, sz, nx, nz, idx, Nx, Nz;
  PetscScalar    ***xx;
  DMStagStencil  point;
  PetscScalar    xp, zp, r, sina;
  PetscScalar    v[2], p, C1, C4, u0, eta0;
  DM             da, cda;
  Vec            x, xlocal;
  Vec            coords;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Get parameters
  Nx   = sol->grd->nx;
  Nz   = sol->grd->nz;
  sina = sol->usr->mor_sina;
  u0   = sol->scal->u0;
  eta0 = sol->scal->eta0;
  C1   = sol->usr->mor_C1;
  C4   = sol->usr->mor_C4;
  
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

  // Loop over elements and assign constraints
  for (j = sz; j<sz+nz; ++j) {
    for (i = sx; i<sx+nx; ++i) {

      // 1) Constrain P - ELEMENT
      // Get coordinates and stencil values
      point.i = i; point.j = j; point.loc = ELEMENT; point.c = 0; // P
      ierr = GetCoordinatesStencil(cda, coords, 1, &point, &xp, &zp); CHKERRQ(ierr);
      ierr = DMStagGetLocationSlot(da, point.loc, point.c, &idx);     CHKERRQ(ierr);

      // Calculate positions relative to the lid 
      r = PetscPowScalar(xp*xp+zp*zp,0.5);

      // Set value
      if (zp>=-r*sina){ 
        xx[j][i][idx] = 0.0; // Lid
      } else { 
        evaluate_CornerFlow_MOR(C1, C4, u0, eta0, xp, zp, v, &p);
        xx[j][i][idx] = p;
      }

      // 2) Constrain Vx - LEFT
      // Get coordinates and stencil values
      point.i = i; point.j = j; point.loc = LEFT; point.c = 0; // Vx
      ierr = GetCoordinatesStencil(cda, coords, 1, &point, &xp, &zp); CHKERRQ(ierr);
      ierr = DMStagGetLocationSlot(da, point.loc, point.c, &idx);     CHKERRQ(ierr);

      // Calculate positions relative to the lid 
      r = PetscPowScalar(xp*xp+zp*zp,0.5);

      // Set value
      if (zp>=-r*sina){ 
        xx[j][i][idx] = sol->scal->u0; // Lid
      } else { 
        evaluate_CornerFlow_MOR(C1, C4, u0, eta0, xp, zp, v, &p);
        xx[j][i][idx] = v[0];
      }

      // 3) Constrain Vz - DOWN
      // Get coordinates and stencil values
      point.i = i; point.j = j; point.loc = DOWN; point.c = 0; // Vz
      ierr = GetCoordinatesStencil(cda, coords, 1, &point, &xp, &zp); CHKERRQ(ierr);
      ierr = DMStagGetLocationSlot(da, point.loc, point.c, &idx);     CHKERRQ(ierr);

      // Calculate positions relative to the lid 
      r = PetscPowScalar(xp*xp+zp*zp,0.5);

      // Set value
      if (zp>=-r*sina){ 
        xx[j][i][idx] = 0.0; // Lid
      } else { 
        evaluate_CornerFlow_MOR(C1, C4, u0, eta0, xp, zp, v, &p);
        xx[j][i][idx] = v[1];
      }

      // 4) Vx - RIGHT 
      if (i == Nx-1) {
        // Get coordinates and stencil values
        point.i = i; point.j = j; point.loc = RIGHT; point.c = 0; // Vx right
        ierr = GetCoordinatesStencil(cda, coords, 1, &point, &xp, &zp); CHKERRQ(ierr);
        ierr = DMStagGetLocationSlot(da, point.loc, point.c, &idx);     CHKERRQ(ierr);

        // Calculate positions relative to the lid 
        r = PetscPowScalar(xp*xp+zp*zp,0.5);

        // Constrain Vx - RIGHT
        if (zp>=-r*sina){
          xx[j][i][idx] = sol->scal->u0;
        } else {
          evaluate_CornerFlow_MOR(C1, C4, u0, eta0, xp, zp, v, &p);
          xx[j][i][idx] = v[0];
        }
      }

      // 5) Vz - UP
      if (j == Nz-1) {
        // Get coordinates and stencil values
        point.i = i; point.j = j; point.loc = UP; point.c = 0; // Vz up
        ierr = GetCoordinatesStencil(cda, coords, 1, &point, &xp, &zp); CHKERRQ(ierr);
        ierr = DMStagGetLocationSlot(da, point.loc, point.c, &idx);     CHKERRQ(ierr);

        // Calculate positions relative to the lid 
        r = PetscPowScalar(xp*xp+zp*zp,0.5);

        // Constrain Vz
        if (zp>=-r*sina){
          xx[j][i][idx] = 0.0;
        } else {
          evaluate_CornerFlow_MOR(C1, C4, u0, eta0, xp, zp, v, &p);
          xx[j][i][idx] = v[1];
        }
      }
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
// Create LAPLACE (diffusion) analytical solution - TEMP
// ---------------------------------------
PetscErrorCode CreateLaplaceAnalytic(SolverCtx *sol,DM *_da,Vec *_x)
{
  PetscInt       i, j, sx, sz, nx, nz, idx;
  PetscScalar    ***xx;
  DMStagStencil  point;
  PetscScalar    xp, zp, A;
  DM             da, cda;
  Vec            x, xlocal;
  Vec            coords;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Get parameters
  A = 1.0/PetscSinhReal(PETSC_PI);
  
  // Create identical DM as sol->dmHT for analytical solution
  ierr = DMStagCreateCompatibleDMStag(sol->dmHT, sol->grd->dofHT0, sol->grd->dofHT1, sol->grd->dofHT2, 0, &da); CHKERRQ(ierr);
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

  // Loop over elements and assign constraints
  for (j = sz; j<sz+nz; ++j) {
    for (i = sx; i<sx+nx; ++i) {

      // Constrain ELEMENT values for the Laplace equation (nabla^2 u = 0)
      // Get coordinates and stencil values
      point.i = i; point.j = j; point.loc = ELEMENT; point.c = 0; 
      ierr = GetCoordinatesStencil(cda, coords, 1, &point, &xp, &zp); CHKERRQ(ierr);
      ierr = DMStagGetLocationSlot(da, point.loc, point.c, &idx);     CHKERRQ(ierr);

      // Analytical solution
      xx[j][i][idx] = A*PetscSinScalar(PETSC_PI*xp)*PetscSinhReal(PETSC_PI*zp);
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
// DoOutputTemp_Analytic
// ---------------------------------------
PetscErrorCode DoOutputTemp_Analytic(SolverCtx *sol, DM da,Vec x)
{
  DM             daT;
  Vec            vecT;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  // Create individual DMDAs for sub-grids of our DMStag objects
  ierr = DMStagVecSplitToDMDA(da,x,ELEMENT, 0,&daT,&vecT); CHKERRQ(ierr);
  ierr = PetscObjectSetName  ((PetscObject)vecT,"Temperature");         CHKERRQ(ierr);

  // Dump element-based fields to a .vtr file
  {
    PetscViewer viewer;

    // Warning: is being output as Point Data instead of Cell Data - the grid is shifted to be in the center points.
    ierr = PetscViewerVTKOpen(PetscObjectComm((PetscObject)daT),"analytic_solution_temp.vtr",FILE_MODE_WRITE,&viewer); CHKERRQ(ierr);
    ierr = VecView(vecT,     viewer); CHKERRQ(ierr);
    
    // Free memory
    ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
  }

  // Destroy DMDAs and Vecs
  ierr = VecDestroy(&vecT  ); CHKERRQ(ierr);
  ierr = DMDestroy(&daT    ); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// CalculateErrorNormsTemp
// ---------------------------------------
PetscErrorCode CalculateErrorNormsTemp(SolverCtx *sol,DM da,Vec x)
{
  PetscInt       i, j, sx, sz, nx, nz;
  PetscScalar    xx, xa, dx, dz, dv;
  PetscScalar    nrmT, gnrmT;
  Vec            xlocal, xalocal;

  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Assign pointers and other variables
  dx = sol->grd->dx;
  dz = sol->grd->dz;

  // Get domain corners
  ierr = DMStagGetCorners(da, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  // Map global vectors to local domain
  ierr = DMGetLocalVector(sol->dmHT, &xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (sol->dmHT, sol->T, INSERT_VALUES, xlocal); CHKERRQ(ierr);

  ierr = DMGetLocalVector(da, &xalocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (da, x, INSERT_VALUES, xalocal); CHKERRQ(ierr);

  // Initialize norm
  nrmT = 0.0;
  dv = dx*dz;

  // Loop over local domain to calculate ELEMENT errors
  for (j = sz; j < sz+nz; ++j) {
    for (i = sx; i <sx+nx; ++i) {
      PetscScalar    Te;
      DMStagStencil  col;
      
      // Get stencil values
      col.i = i; col.j = j; col.loc = ELEMENT; col.c = 0; // T

      // Get numerical solution
      ierr = DMStagVecGetValuesStencil(sol->dmHT, xlocal, 1, &col, &xx); CHKERRQ(ierr);

      // Get analytical solution
      ierr = DMStagVecGetValuesStencil(da, xalocal, 1, &col, &xa); CHKERRQ(ierr);

      // Calculate errors per element
      Te = PetscAbsScalar(xx-xa);

      // Calculate norms as in Duretz et al. 2011
      nrmT += Te*dv;
    }
  }

  // Collect data 
  ierr = MPI_Allreduce(&nrmT, &gnrmT, 1, MPI_DOUBLE, MPI_SUM, sol->comm); CHKERRQ(ierr);

  // Restore arrays and vectors
  ierr = DMRestoreLocalVector(sol->dmHT, &xlocal ); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,        &xalocal); CHKERRQ(ierr);

  // Print information
  PetscPrintf(sol->comm,"# --------------------------------------- #\n");
  PetscPrintf(sol->comm,"# NORMS: \n");
  PetscPrintf(sol->comm,"# Temperature: norm1 = %1.12e\n",gnrmT);
  PetscPrintf(sol->comm,"# Grid info: hx = %1.12e hz = %1.12e \n",dx,dz);

  PetscFunctionReturn(0);
}