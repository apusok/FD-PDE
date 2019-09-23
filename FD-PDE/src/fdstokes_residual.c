#include "fdstokes.h"

// ---------------------------------------
// FormFunction_Stokes
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormFunction_Stokes"
PetscErrorCode FormFunction_Stokes(SNES snes, Vec x, Vec f, void *ctx)
{
  FD             fd = (FD)ctx;
  CoeffStokes    *cdata;
  DM             dmPV;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz;
  Vec            xlocal, flocal;
  PetscInt       idx, n[9], nbc;
  PetscInt       iprev, inext, icenter;
  PetscScalar    ***ff;
  PetscScalar    **coordx,**coordz;
  BCList         *bclist;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Assign pointers and other variables
  cdata   = (CoeffStokes*)fd->coeff_context;
  dmPV    = fd->dmstag;
  Nx = fd->Nx;
  Nz = fd->Nz;

  // Get BC list
  bclist = fd->bc_list;
  nbc    = fd->nbc;

  // Update coefficients
  ierr = CoefficientEvaluate(cdata->eta_n);CHKERRQ(ierr);
  ierr = CoefficientEvaluate(cdata->eta_c);CHKERRQ(ierr);
  ierr = CoefficientEvaluate(cdata->fux);CHKERRQ(ierr);
  ierr = CoefficientEvaluate(cdata->fuz);CHKERRQ(ierr);
  ierr = CoefficientEvaluate(cdata->fp );CHKERRQ(ierr);

  // Get local domain
  ierr = DMStagGetCorners(dmPV, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = DMStagGet1dCoordinateLocationSlot(dmPV,DMSTAG_ELEMENT,&icenter);CHKERRQ(ierr); 
  ierr = DMStagGet1dCoordinateLocationSlot(dmPV,DMSTAG_LEFT,&iprev);CHKERRQ(ierr);
  ierr = DMStagGet1dCoordinateLocationSlot(dmPV,DMSTAG_RIGHT,&inext);CHKERRQ(ierr); 

  // Save useful variables for residual calculations
  n[0] = sx; n[1] = sz; n[2] = nx; n[3] = nz;
  n[4] = Nx; n[5] = Nz;
  n[6] = icenter; n[7] = iprev; n[8] = inext;

  // Map global vectors to local domain
  ierr = DMGetLocalVector(dmPV, &xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dmPV, x, INSERT_VALUES, xlocal); CHKERRQ(ierr);

  // Get dm coordinates array
  ierr = DMStagGet1dCoordinateArraysDOFRead(dmPV,&coordx,&coordz,NULL);CHKERRQ(ierr);

  // Create residual local vector
  ierr = DMCreateLocalVector(dmPV, &flocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayDOF(dmPV, flocal, &ff); CHKERRQ(ierr);

  // Loop over elements
  for (j = sz; j<sz+nz; ++j) {
    for (i = sx; i<sx+nx; ++i) {
      PetscScalar fval;

      // 1) Continuity equation
      ierr = ContinuityResidual(dmPV,xlocal,coordx,coordz,cdata->fp->coeff,i,j,n,&fval);CHKERRQ(ierr);
      ierr = DMStagGetLocationSlot(dmPV, DMSTAG_ELEMENT, 0, &idx); CHKERRQ(ierr);
      ff[j][i][idx] = fval;

      // 2) X-Momentum equation
      if (i > 0) {
        ierr = XMomentumResidual(dmPV,xlocal,coordx,coordz,cdata->eta_n->coeff,cdata->eta_c->coeff,cdata->fux->coeff,i,j,n,&fval);CHKERRQ(ierr);
        ierr = DMStagGetLocationSlot(dmPV, DMSTAG_LEFT, 0, &idx); CHKERRQ(ierr);
        ff[j][i][idx] = fval;
      }

      // 3) Z-Momentum equation
      if (j > 0) {
        ierr = ZMomentumResidual(dmPV,xlocal,coordx,coordz,cdata->eta_n->coeff,cdata->eta_c->coeff,cdata->fuz->coeff,i,j,n,&fval);CHKERRQ(ierr);
        ierr = DMStagGetLocationSlot(dmPV, DMSTAG_DOWN, 0, &idx); CHKERRQ(ierr);
        ff[j][i][idx] = fval;
      }
    }
  }

  // Boundary conditions
  ierr = FDBCApplyStokes(dmPV,xlocal,bclist,nbc,coordx,coordz,cdata->eta_n->coeff,cdata->eta_c->coeff,n,ff);CHKERRQ(ierr);

  // Restore arrays, local vectors
  ierr = DMStagRestore1dCoordinateArraysDOFRead(dmPV,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArrayDOF(dmPV,flocal,&ff); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmPV,&xlocal); CHKERRQ(ierr);

  // Map local to global
  ierr = DMLocalToGlobalBegin(dmPV,flocal,INSERT_VALUES,f); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dmPV,flocal,INSERT_VALUES,f); CHKERRQ(ierr);

  ierr = VecDestroy(&flocal); CHKERRQ(ierr);

  // // View vectors
  // ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
  // ierr = VecView(f,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

// ---------------------------------------
// FDBCApplyStokes
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDBCApplyStokes"
PetscErrorCode FDBCApplyStokes(DM dm, Vec xlocal, BCList *bclist, PetscInt nbc, PetscScalar **coordx, PetscScalar **coordz, PetscScalar *eta_n, PetscScalar *eta_c,PetscInt n[], PetscScalar ***ff)
{
  PetscScalar    xx, dx, dz;
  PetscScalar    etaLeft, etaRight, etaUp, etaDown;
  PetscInt       i, j, ibc, idx, iprev, inext;
  PetscInt       sx, sz, nz, Nx, Nz;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  
  // DM domain info
  sx = n[0]; sz = n[1];
  nz = n[3];
  Nx = n[4]; Nz = n[5];
  iprev = n[7]; inext = n[8];
  
  // Loop over all boundaries
  for (ibc = 0; ibc<nbc; ibc++) {
    if (bclist[ibc].type == BC_DIRICHLET) {
      i   = bclist[ibc].point.i;
      j   = bclist[ibc].point.j;
      idx = bclist[ibc].idx;
      
      // Get residual value
      ierr = DMStagVecGetValuesStencil(dm, xlocal, 1, &bclist[ibc].point, &xx); CHKERRQ(ierr);
      ff[j][i][idx] = xx - bclist[ibc].val;
    }
    
    if (bclist[ibc].type == BC_NEUMANN) {
      i   = bclist[ibc].point.i;
      j   = bclist[ibc].point.j;
      idx = bclist[ibc].idx;
      
      // Stokes flow - add flux terms
      if ((j == 0) && (bclist[ibc].point.loc == DMSTAG_LEFT)) { // Vx down
        etaDown = eta_n[i-sx+(j-sz)*nz];
        dz = coordz[j][inext]-coordz[j][iprev];
        ff[j][i][idx] += -2.0*etaDown*bclist[ibc].val/dz;
      }
      
      if ((j == 0) && (bclist[ibc].point.loc == DMSTAG_RIGHT)) { // Vx down
        etaDown = eta_n[i+1-sx+(j-sz)*nz];
        dz = coordz[j][inext]-coordz[j][iprev];
        ff[j][i][idx] += -2.0*etaDown*bclist[ibc].val/dz;
      }
      
      if ((j == Nz-1) && (bclist[ibc].point.loc == DMSTAG_LEFT)) { // Vx up
        etaUp = eta_n[i-sx+(j+1-sz)*nz];
        dz = coordz[j][inext]-coordz[j][iprev];
        ff[j][i][idx] += 2.0*etaUp*bclist[ibc].val/dz;
      }
      
      if ((j == Nz-1) && (bclist[ibc].point.loc == DMSTAG_RIGHT)) { // Vx up
        etaUp = eta_n[i+1-sx+(j+1-sz)*nz];
        dz = coordz[j][inext]-coordz[j][iprev];
        ff[j][i][idx] += 2.0*etaUp*bclist[ibc].val/dz;
      }
      
      if ((i == 0) && (bclist[ibc].point.loc == DMSTAG_DOWN)) { // Vz left
        etaLeft = eta_n[i-sx+(j-sz)*nz];
        dx = coordx[i][inext]-coordx[i][iprev];
        ff[j][i][idx] += -2.0*etaLeft*bclist[ibc].val/dx;
      }
      
      if ((i == Nx-1) && (bclist[ibc].point.loc == DMSTAG_DOWN)) { // Vz right
        etaRight = eta_n[i+1-sx+(j-sz)*nz];
        dx = coordx[i][inext]-coordx[i][iprev];
        ff[j][i][idx] += 2.0*etaRight*bclist[ibc].val/dx;
      }
    }
  }
  PetscFunctionReturn(0);
}
