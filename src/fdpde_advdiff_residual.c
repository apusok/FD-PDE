#include "fdpde_advdiff.h"

// ---------------------------------------
/*@
FormFunction_AdvDiff - (ADVDIFF) Residual evaluation function

Use: internal
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormFunction_AdvDiff"
PetscErrorCode FormFunction_AdvDiff(SNES snes, Vec x, Vec f, void *ctx)
{
  FDPDE          fd = (FDPDE)ctx;
  AdvDiffData    *ad;
  DM             dm, dmcoeff;
  Vec            xlocal, coefflocal, flocal;
  Vec            xprevlocal, coeffprevlocal;
  PetscInt       Nx, Nz, sx, sz, nx, nz;
  PetscInt       i,j, idx,icenter;
  PetscScalar    fval;
  DMStagBCList   bclist;
  PetscScalar    **coordx,**coordz;
  PetscScalar    ***ff;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  if (!fd->ops->form_coefficient) SETERRQ(fd->comm,PETSC_ERR_ARG_NULL,"Form coefficient function pointer is NULL. Must call FDPDESetFunctionCoefficient() and provide a non-NULL function pointer.");
  
  // Assign pointers and other variables
  dm    = fd->dmstag;
  dmcoeff = fd->dmcoeff;
  ad = fd->data;
  
  xprevlocal     = NULL;
  coeffprevlocal = NULL;

  Nx = fd->Nx;
  Nz = fd->Nz;

  // Update BC list
  bclist = fd->bclist;
  if (fd->bclist->evaluate) {
    ierr = fd->bclist->evaluate(dm,x,bclist,bclist->data);CHKERRQ(ierr);
  }

  // Update coefficients
  ierr = fd->ops->form_coefficient(fd,dm,x,dmcoeff,fd->coeff,fd->user_context);CHKERRQ(ierr);

  // Get local domain
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  // Map global vectors to local domain
  ierr = DMGetLocalVector(dm, &xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm, x, INSERT_VALUES, xlocal); CHKERRQ(ierr);

  ierr = DMGetLocalVector(dmcoeff, &coefflocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dmcoeff, fd->coeff, INSERT_VALUES, coefflocal); CHKERRQ(ierr);

  // Map the previous time step vectors
  if (ad->timesteptype != TS_NONE) {
    ierr = DMGetLocalVector(dm, &xprevlocal); CHKERRQ(ierr);
    ierr = DMGlobalToLocal (dm, ad->xprev, INSERT_VALUES, xprevlocal); CHKERRQ(ierr);

    ierr = DMGetLocalVector(dmcoeff, &coeffprevlocal); CHKERRQ(ierr);
    ierr = DMGlobalToLocal (dmcoeff, ad->coeffprev, INSERT_VALUES, coeffprevlocal); CHKERRQ(ierr);

    // Check time step
    if (!ad->dt) {
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"A valid time step size for FD-PDE ADVDIFF was not set! Set with FDPDEAdvDiffSetTimestep()");
    }
  }

  // Get dm coordinates array
  ierr = DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);

  // Create residual local vector
  ierr = DMCreateLocalVector(dm, &flocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm, flocal, &ff); CHKERRQ(ierr);

  // Loop over elements
  for (j = sz; j<sz+nz; j++) {
    for (i = sx; i<sx+nx; i++) {
      ierr = AdvDiffResidual(dm,xlocal,dmcoeff,coefflocal,xprevlocal,coeffprevlocal,coordx,coordz,ad,i,j,-1,0.0,fd->dm_btype0,fd->dm_btype1,&fval); CHKERRQ(ierr);
      ierr = DMStagGetLocationSlot(dm, DMSTAG_ELEMENT, 0, &idx); CHKERRQ(ierr);
      ff[j][i][idx] = fval;
    }
  }

  // Boundary conditions - only element dofs
  ierr = DMStagBCListApply_AdvDiff(dm,xlocal,dmcoeff,coefflocal,xprevlocal,coeffprevlocal,bclist->bc_e,bclist->nbc_element,coordx,coordz,ad,Nx,Nz,fd->dm_btype0,fd->dm_btype1,ff);CHKERRQ(ierr);

  // Restore arrays, local vectors
  ierr = DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArray(dm,flocal,&ff); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&xlocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmcoeff,&coefflocal); CHKERRQ(ierr);

  if (ad->timesteptype != TS_NONE) {
    ierr = DMRestoreLocalVector(dm, &xprevlocal); CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dmcoeff, &coeffprevlocal); CHKERRQ(ierr);
  }

  // Map local to global
  ierr = DMLocalToGlobalBegin(dm,flocal,INSERT_VALUES,f); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,flocal,INSERT_VALUES,f); CHKERRQ(ierr);

  ierr = VecDestroy(&flocal); CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
AdvDiffResidual - (ADVDIFF) calculates the steady state advdiff residual per dof

Use: internal
@*/
// ---------------------------------------
PetscErrorCode AdvDiffResidual(DM dm, Vec xlocal, DM dmcoeff,Vec coefflocal, Vec xprevlocal, Vec coeffprevlocal, PetscScalar **coordx, PetscScalar **coordz, AdvDiffData *ad, PetscInt i, PetscInt j, PetscInt bc_type, PetscScalar bc_val, DMBoundaryType dm_btype0, DMBoundaryType dm_btype1,PetscScalar *_fval)
{
  PetscScalar   xx, xxprev;
  PetscScalar   fval=0.0, fval0=0.0, fval1=0.0;
  PetscScalar   A, A0, A1;
  DMStagStencil point;

  PetscErrorCode ierr;

  PetscFunctionBegin;

  if (ad->timesteptype == TS_NONE) {
    // steady-state operator
    ierr = AdvDiffSteadyStateOperator(dm,xlocal,dmcoeff,coefflocal,coordx,coordz,i,j,ad->advtype,bc_type,bc_val,dm_btype0,dm_btype1,&fval,&A); CHKERRQ(ierr);
  } else { 
    // time-dependent solution
    ierr = AdvDiffSteadyStateOperator(dm,xprevlocal,dmcoeff,coeffprevlocal,coordx,coordz,i,j,ad->advtype,bc_type,bc_val,dm_btype0,dm_btype1,&fval0,&A0); CHKERRQ(ierr);
    ierr = AdvDiffSteadyStateOperator(dm,xlocal,dmcoeff,coefflocal,coordx,coordz,i,j,ad->advtype,bc_type,bc_val,dm_btype0,dm_btype1,&fval1,&A1); CHKERRQ(ierr);

    point.i = i; point.j = j; point.loc = DMSTAG_ELEMENT; point.c = 0;
    ierr = DMStagVecGetValuesStencil(dm,xlocal,1,&point,&xx); CHKERRQ(ierr);
    ierr = DMStagVecGetValuesStencil(dm,xprevlocal,1,&point,&xxprev); CHKERRQ(ierr);

    fval = xx - xxprev + ad->dt*(ad->theta*fval1/A1 + (1-ad->theta)*fval0/A0 );
  }

  *_fval = fval;

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
AdvDiffSteadyStateOperator - (ADVDIFF) calculates the steady state advdiff residual per dof

Use: internal
@*/
// ---------------------------------------
PetscErrorCode AdvDiffSteadyStateOperator(DM dm, Vec xlocal, DM dmcoeff,Vec coefflocal, PetscScalar **coordx, PetscScalar **coordz, PetscInt i, PetscInt j, AdvectSchemeType advtype,PetscInt bc_type, PetscScalar bc_val, DMBoundaryType dm_btype0, DMBoundaryType dm_btype1, PetscScalar *ff,PetscScalar *_A)
{
  PetscScalar    ffi;
  PetscInt       Nx, Nz, icenter;
  PetscScalar    xx[9], cx[10], u[5];
  PetscScalar    dx[3], dz[3];
  PetscScalar    A, B_Left, B_Right, B_Up, B_Down, C;
  PetscScalar    dQ2dx, dQ2dz, diff, adv;
  DMStagStencil  point[10];
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Get variables
  ierr = DMStagGetGlobalSizes(dm,&Nx,&Nz,NULL);CHKERRQ(ierr); 
  ierr = DMStagGetProductCoordinateLocationSlot(dm,DMSTAG_ELEMENT,&icenter);CHKERRQ(ierr); 

  // Coefficients
  point[0].i = i; point[0].j = j; point[0].loc = DMSTAG_ELEMENT; point[0].c = 0; // A
  point[1].i = i; point[1].j = j; point[1].loc = DMSTAG_ELEMENT; point[1].c = 1; // C
  point[2].i = i; point[2].j = j; point[2].loc = DMSTAG_LEFT;    point[2].c = 0; // B_left
  point[3].i = i; point[3].j = j; point[3].loc = DMSTAG_RIGHT;   point[3].c = 0; // B_right
  point[4].i = i; point[4].j = j; point[4].loc = DMSTAG_DOWN;    point[4].c = 0; // B_down
  point[5].i = i; point[5].j = j; point[5].loc = DMSTAG_UP;      point[5].c = 0; // B_up
  point[6].i = i; point[6].j = j; point[6].loc = DMSTAG_LEFT;    point[6].c = 1; // u_left
  point[7].i = i; point[7].j = j; point[7].loc = DMSTAG_RIGHT;   point[7].c = 1; // u_right
  point[8].i = i; point[8].j = j; point[8].loc = DMSTAG_DOWN;    point[8].c = 1; // u_down
  point[9].i = i; point[9].j = j; point[9].loc = DMSTAG_UP;      point[9].c = 1; // u_up

  ierr = DMStagVecGetValuesStencil(dmcoeff,coefflocal,10,point,cx); CHKERRQ(ierr);

  // Assign variables
  A = cx[0];
  C = cx[1];

  B_Left  = cx[2];
  B_Right = cx[3];
  B_Down  = cx[4];
  B_Up    = cx[5];

  u[0] = 0.0; //
  u[1] = cx[6]; // u_left
  u[2] = cx[7]; // u_right
  u[3] = cx[8]; // u_down
  u[4] = cx[9]; // u_up

  // Grid spacings
  if (i == Nx-1) dx[0] = coordx[i  ][icenter]-coordx[i-1][icenter];
  else           dx[0] = coordx[i+1][icenter]-coordx[i  ][icenter];

  if (i == 0) dx[1] = coordx[i+1][icenter]-coordx[i  ][icenter];
  else        dx[1] = coordx[i  ][icenter]-coordx[i-1][icenter];
  dx[2]  = (dx[0]+dx[1])*0.5;

  if (j == Nz-1) dz[0] = coordz[j  ][icenter]-coordz[j-1][icenter];
  else           dz[0] = coordz[j+1][icenter]-coordz[j  ][icenter];

  if (j == 0) dz[1] = coordz[j+1][icenter]-coordz[j  ][icenter];
  else        dz[1] = coordz[j  ][icenter]-coordz[j-1][icenter];
  dz[2] = (dz[0]+dz[1])*0.5;

  // Get stencil values - diffusion
  point[0].i = i  ; point[0].j = j  ; point[0].loc = DMSTAG_ELEMENT; point[0].c = 0; // Qi,j -C
  point[1].i = i-1; point[1].j = j  ; point[1].loc = DMSTAG_ELEMENT; point[1].c = 0; // Qi-1,j -W
  point[2].i = i+1; point[2].j = j  ; point[2].loc = DMSTAG_ELEMENT; point[2].c = 0; // Qi+1,j -E
  point[3].i = i  ; point[3].j = j-1; point[3].loc = DMSTAG_ELEMENT; point[3].c = 0; // Qi,j-1 -S
  point[4].i = i  ; point[4].j = j+1; point[4].loc = DMSTAG_ELEMENT; point[4].c = 0; // Qi,j+1 -N

  // Get stencil values - advection (need to take into account outside boundaries)
  point[5].i = i-2; point[5].j = j  ; point[5].loc = DMSTAG_ELEMENT; point[5].c = 0; // Qi-2,j -WW
  point[6].i = i+2; point[6].j = j  ; point[6].loc = DMSTAG_ELEMENT; point[6].c = 0; // Qi+2,j -EE
  point[7].i = i  ; point[7].j = j-2; point[7].loc = DMSTAG_ELEMENT; point[7].c = 0; // Qi,j-2 -SS
  point[8].i = i  ; point[8].j = j+2; point[8].loc = DMSTAG_ELEMENT; point[8].c = 0; // Qi,j+2 -NN

  if (dm_btype0!=DM_BOUNDARY_PERIODIC) {
    if (i == 1   ) point[5] = point[2];
    if (i == Nx-2) point[6] = point[1];
    if (i == 0   ) { point[1] = point[0]; point[5] = point[2]; }
    if (i == Nx-1) { point[2] = point[0]; point[6] = point[1]; }
  }

  if (dm_btype1!=DM_BOUNDARY_PERIODIC) {
    if (j == 1   ) point[7] = point[4];
    if (j == Nz-2) point[8] = point[3];
    if (j == 0   ) { point[3] = point[0]; point[7] = point[4]; }
    if (j == Nz-1) { point[4] = point[0]; point[8] = point[3]; }
  }

  ierr = DMStagVecGetValuesStencil(dm,xlocal,9,point,xx); CHKERRQ(ierr);

  // add Neumann BC
  if (bc_type == 0) { xx[1] = xx[0] - bc_val*dx[1]; xx[5] = xx[2] - 3.0*bc_val*dx[1]; } // left 
  if (bc_type == 1) { xx[2] = xx[0] + bc_val*dx[0]; xx[6] = xx[1] + 3.0*bc_val*dx[0]; } // right
  if (bc_type == 2) { xx[3] = xx[0] - bc_val*dz[1]; xx[7] = xx[4] - 3.0*bc_val*dz[1]; } // down
  if (bc_type == 3) { xx[4] = xx[0] + bc_val*dz[0]; xx[8] = xx[3] + 3.0*bc_val*dz[0]; } // up
  if (bc_type == 4) { xx[5] = xx[2] - 3.0*bc_val*dx[1]; } // left+1 
  if (bc_type == 5) { xx[6] = xx[1] + 3.0*bc_val*dx[0]; } // right-1
  if (bc_type == 6) { xx[7] = xx[4] - 3.0*bc_val*dz[1]; } // down+1
  if (bc_type == 7) { xx[8] = xx[3] + 3.0*bc_val*dz[0]; } // up-1

  // Calculate diff residual
  dQ2dx = B_Right*(xx[2]-xx[0])/dx[0] - B_Left*(xx[0]-xx[1])/dx[1];
  dQ2dz = B_Up   *(xx[4]-xx[0])/dz[0] - B_Down*(xx[0]-xx[3])/dz[1];
  diff = dQ2dx/dx[2] + dQ2dz/dz[2];

  // Calculate diffadv residual
  ierr = AdvectionResidual(u,xx,dx,dz,advtype,&adv); CHKERRQ(ierr);
  ffi  = A*adv - diff + C;

  *ff = ffi;
  *_A = A;

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
DMStagBCListApply_AdvDiff - function to apply boundary conditions for ADVDIFF equations

Use: internal
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "DMStagBCListApply_AdvDiff"
PetscErrorCode DMStagBCListApply_AdvDiff(DM dm, Vec xlocal,DM dmcoeff, Vec coefflocal, Vec xprevlocal, Vec coeffprevlocal, DMStagBC *bclist, PetscInt nbc, PetscScalar **coordx, PetscScalar **coordz, AdvDiffData *ad, PetscInt Nx, PetscInt Nz, DMBoundaryType dm_btype0, DMBoundaryType dm_btype1,PetscScalar ***ff)
{
  PetscScalar    xx, fval;
  PetscInt       i, j, ibc, idx;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  // Loop over all boundaries
  for (ibc = 0; ibc<nbc; ibc++) {

    if (bclist[ibc].type == BC_PERIODIC) { // normal stencil for i,j - should come before other BCs are set
      i   = bclist[ibc].point.i;
      j   = bclist[ibc].point.j;
      idx = bclist[ibc].idx;
      ierr = AdvDiffResidual(dm,xlocal,dmcoeff,coefflocal,xprevlocal,coeffprevlocal,coordx,coordz,ad,i,j,-1,0.0,dm_btype0,dm_btype1,&fval); CHKERRQ(ierr);
      ff[j][i][idx] = fval;
    }

    if (bclist[ibc].type == BC_DIRICHLET) {
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"BC_DIRICHLET type on the true boundary for FDPDE_ADVDIFF [ELEMENT] is not yet implemented. Use BC_DIRICHLET_STAG type instead!");
    }
    
    if (bclist[ibc].type == BC_DIRICHLET_STAG) {
      i   = bclist[ibc].point.i;
      j   = bclist[ibc].point.j;
      idx = bclist[ibc].idx;

      // Get residual value
      ierr = DMStagVecGetValuesStencil(dm, xlocal, 1, &bclist[ibc].point, &xx); CHKERRQ(ierr);
      ff[j][i][idx] = xx - bclist[ibc].val;
    }

    if (bclist[ibc].type == BC_NEUMANN) {
      // Add flux terms - for first and second points (needed for second order advection schemes)
      i   = bclist[ibc].point.i;
      j   = bclist[ibc].point.j;
      idx = bclist[ibc].idx;

      if (i == 0) { // left
        ierr = AdvDiffResidual(dm,xlocal,dmcoeff,coefflocal,xprevlocal,coeffprevlocal,coordx,coordz,ad,i  ,j,0,bclist[ibc].val,dm_btype0,dm_btype1,&fval); CHKERRQ(ierr);
        ff[j][i  ][idx] = fval;
        ierr = AdvDiffResidual(dm,xlocal,dmcoeff,coefflocal,xprevlocal,coeffprevlocal,coordx,coordz,ad,i+1,j,4,bclist[ibc].val,dm_btype0,dm_btype1,&fval); CHKERRQ(ierr);
        ff[j][i+1][idx] = fval;
      }
      if (i == Nx-1) { // right
        ierr = AdvDiffResidual(dm,xlocal,dmcoeff,coefflocal,xprevlocal,coeffprevlocal,coordx,coordz,ad,i  ,j,1,bclist[ibc].val,dm_btype0,dm_btype1,&fval); CHKERRQ(ierr);
        ff[j][i  ][idx] = fval;
        ierr = AdvDiffResidual(dm,xlocal,dmcoeff,coefflocal,xprevlocal,coeffprevlocal,coordx,coordz,ad,i-1,j,5,bclist[ibc].val,dm_btype0,dm_btype1,&fval); CHKERRQ(ierr);
        ff[j][i-1][idx] = fval;
      }
      if (j == 0) { // down
        ierr = AdvDiffResidual(dm,xlocal,dmcoeff,coefflocal,xprevlocal,coeffprevlocal,coordx,coordz,ad,i,j  ,2,bclist[ibc].val,dm_btype0,dm_btype1,&fval); CHKERRQ(ierr);
        ff[j  ][i][idx] = fval;
        ierr = AdvDiffResidual(dm,xlocal,dmcoeff,coefflocal,xprevlocal,coeffprevlocal,coordx,coordz,ad,i,j+1,6,bclist[ibc].val,dm_btype0,dm_btype1,&fval); CHKERRQ(ierr);
        ff[j+1][i][idx] = fval;
      }
      if (j == Nz-1) { // up
        ierr = AdvDiffResidual(dm,xlocal,dmcoeff,coefflocal,xprevlocal,coeffprevlocal,coordx,coordz,ad,i,j  ,3,bclist[ibc].val,dm_btype0,dm_btype1,&fval); CHKERRQ(ierr);
        ff[j  ][i][idx] = fval;
        ierr = AdvDiffResidual(dm,xlocal,dmcoeff,coefflocal,xprevlocal,coeffprevlocal,coordx,coordz,ad,i,j-1,7,bclist[ibc].val,dm_btype0,dm_btype1,&fval); CHKERRQ(ierr);
        ff[j-1][i][idx] = fval;
      }
    }
  }

  PetscFunctionReturn(0);
}

