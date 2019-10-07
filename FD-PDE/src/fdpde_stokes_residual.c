#include "fdpde_stokes.h"

// ---------------------------------------
/*@
FormFunction_Stokes - (STOKES) Residual evaluation function

Use: internal
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormFunction_Stokes"
PetscErrorCode FormFunction_Stokes(SNES snes, Vec x, Vec f, void *ctx)
{
  FDPDE          fd = (FDPDE)ctx;
  DM             dmPV, dmCoeff;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz;
  Vec            xlocal, flocal, coefflocal;
  PetscInt       idx, n[5];
  PetscInt       iprev, inext, icenter;
  PetscScalar    ***ff;
  PetscScalar    **coordx,**coordz;
  DMStagBCList   bclist;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Assign pointers and other variables
  dmPV    = fd->dmstag;
  dmCoeff = fd->dmcoeff;
  Nx = fd->Nx;
  Nz = fd->Nz;

  // Update BC list
  ierr = fd->bclist->evaluate(dmPV,x,fd->bclist,fd->bclist->data);CHKERRQ(ierr);
  bclist = fd->bclist;

  // Update coefficients
  ierr = fd->ops->form_coefficient(dmPV,x,dmCoeff,fd->coeff,fd->user_context);CHKERRQ(ierr);

  // Get local domain
  ierr = DMStagGetCorners(dmPV, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = DMStagGet1dCoordinateLocationSlot(dmPV,DMSTAG_ELEMENT,&icenter);CHKERRQ(ierr); 
  ierr = DMStagGet1dCoordinateLocationSlot(dmPV,DMSTAG_LEFT,&iprev);CHKERRQ(ierr);
  ierr = DMStagGet1dCoordinateLocationSlot(dmPV,DMSTAG_RIGHT,&inext);CHKERRQ(ierr); 

  // Save useful variables for residual calculations
  n[0] = Nx; n[1] = Nz; n[2] = icenter; n[3] = iprev; n[4] = inext;

  // Map global vectors to local domain
  ierr = DMGetLocalVector(dmPV, &xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dmPV, x, INSERT_VALUES, xlocal); CHKERRQ(ierr);

  ierr = DMGetLocalVector(dmCoeff, &coefflocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dmCoeff, fd->coeff, INSERT_VALUES, coefflocal); CHKERRQ(ierr);

  // Get dm coordinates array
  ierr = DMStagGet1dCoordinateArraysDOFRead(dmPV,&coordx,&coordz,NULL);CHKERRQ(ierr);

  // Create residual local vector
  ierr = DMCreateLocalVector(dmPV, &flocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayDOF(dmPV, flocal, &ff); CHKERRQ(ierr);

  // Loop over elements
  for (j = sz; j<sz+nz; j++) {
    for (i = sx; i<sx+nx; i++) {
      PetscScalar fval;

      // 1) Continuity equation
      ierr = ContinuityResidual(dmPV,xlocal,dmCoeff,coefflocal,coordx,coordz,i,j,n,&fval);CHKERRQ(ierr);
      ierr = DMStagGetLocationSlot(dmPV, DMSTAG_ELEMENT, 0, &idx); CHKERRQ(ierr);
      ff[j][i][idx] = fval;

      // 2) X-Momentum equation
      if (i > 0) {
        ierr = XMomentumResidual(dmPV,xlocal,dmCoeff,coefflocal,coordx,coordz,i,j,n,&fval);CHKERRQ(ierr);
        ierr = DMStagGetLocationSlot(dmPV, DMSTAG_LEFT, 0, &idx); CHKERRQ(ierr);
        ff[j][i][idx] = fval;
      }

      // 3) Z-Momentum equation
      if (j > 0) {
        ierr = ZMomentumResidual(dmPV,xlocal,dmCoeff,coefflocal,coordx,coordz,i,j,n,&fval);CHKERRQ(ierr);
        ierr = DMStagGetLocationSlot(dmPV, DMSTAG_DOWN, 0, &idx); CHKERRQ(ierr);
        ff[j][i][idx] = fval;
      }
    }
  }

  // Boundary conditions
  ierr = DMStagBCListApply_Stokes(dmPV,xlocal,dmCoeff,coefflocal,bclist->bc_f,bclist->nbc_face,coordx,coordz,n,ff);CHKERRQ(ierr);

  // Restore arrays, local vectors
  ierr = DMStagRestore1dCoordinateArraysDOFRead(dmPV,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArrayDOF(dmPV,flocal,&ff); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmPV,&xlocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmCoeff,&coefflocal); CHKERRQ(ierr);

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
/*@
DMStagBCListApply_Stokes - (STOKES) function to apply boundary conditions for Stokes equations

Use: internal
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "DMStagBCListApply_Stokes"
PetscErrorCode DMStagBCListApply_Stokes(DM dm, Vec xlocal,DM dmcoeff, Vec coefflocal, DMStagBC *bclist, PetscInt nbc, PetscScalar **coordx, PetscScalar **coordz,PetscInt n[], PetscScalar ***ff)
{
  PetscScalar    xx, dx, dz;
  PetscScalar    etaLeft, etaRight, etaUp, etaDown;
  PetscInt       i, j, ibc, idx, iprev, inext, Nx, Nz;
  DMStagStencil  point;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  // dm domain info
  Nx = n[0]; Nz = n[1]; iprev = n[3]; inext = n[4];

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
        point.i = i; point.j = j; point.loc = DMSTAG_DOWN_LEFT; point.c = 0;
        ierr = DMStagVecGetValuesStencil(dmcoeff, coefflocal, 1, &point, &etaDown); CHKERRQ(ierr);
        dz = coordz[j][inext]-coordz[j][iprev];
        ff[j][i][idx] += -2.0*etaDown*bclist[ibc].val/dz;
      }

      if ((j == 0) && (bclist[ibc].point.loc == DMSTAG_RIGHT)) { // Vx down-special case
        point.i = i; point.j = j; point.loc = DMSTAG_DOWN_RIGHT; point.c = 0;
        ierr = DMStagVecGetValuesStencil(dmcoeff, coefflocal, 1, &point, &etaDown); CHKERRQ(ierr);
        dz = coordz[j][inext]-coordz[j][iprev];
        ff[j][i][idx] += -2.0*etaDown*bclist[ibc].val/dz;
      }

      if ((j == Nz-1) && (bclist[ibc].point.loc == DMSTAG_LEFT)) { // Vx up
        point.i = i; point.j = j; point.loc = DMSTAG_UP_LEFT; point.c = 0;
        ierr = DMStagVecGetValuesStencil(dmcoeff, coefflocal, 1, &point, &etaUp); CHKERRQ(ierr);
        dz = coordz[j][inext]-coordz[j][iprev];
        ff[j][i][idx] += 2.0*etaUp*bclist[ibc].val/dz;
      }

      if ((j == Nz-1) && (bclist[ibc].point.loc == DMSTAG_RIGHT)) { // Vx up - special case
        point.i = i; point.j = j; point.loc = DMSTAG_UP_RIGHT; point.c = 0;
        ierr = DMStagVecGetValuesStencil(dmcoeff, coefflocal, 1, &point, &etaUp); CHKERRQ(ierr);
        dz = coordz[j][inext]-coordz[j][iprev];
        ff[j][i][idx] += 2.0*etaUp*bclist[ibc].val/dz;
      }

      if ((i == 0) && (bclist[ibc].point.loc == DMSTAG_DOWN)) { // Vz left
        point.i = i; point.j = j; point.loc = DMSTAG_DOWN_LEFT; point.c = 0;
        ierr = DMStagVecGetValuesStencil(dmcoeff, coefflocal, 1, &point, &etaLeft); CHKERRQ(ierr);
        dx = coordx[i][inext]-coordx[i][iprev];
        ff[j][i][idx] += -2.0*etaLeft*bclist[ibc].val/dx;
      }

      if ((i == Nx-1) && (bclist[ibc].point.loc == DMSTAG_DOWN)) { // Vz right
        point.i = i; point.j = j; point.loc = DMSTAG_DOWN_RIGHT; point.c = 0;
        ierr = DMStagVecGetValuesStencil(dmcoeff, coefflocal, 1, &point, &etaRight); CHKERRQ(ierr);
        dx = coordx[i][inext]-coordx[i][iprev];
        ff[j][i][idx] += 2.0*etaRight*bclist[ibc].val/dx;
      }
    }
  }
  PetscFunctionReturn(0);
}