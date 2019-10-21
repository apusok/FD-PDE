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
  if (!fd->ops->form_coefficient) SETERRQ(fd->comm,PETSC_ERR_ARG_NULL,"Form coefficient function pointer is NULL. Must call FDPDESetFunctionCoefficient() and provide a non-NULL function pointer.");
  
  // Assign pointers and other variables
  dmPV    = fd->dmstag;
  dmCoeff = fd->dmcoeff;
  Nx = fd->Nx;
  Nz = fd->Nz;

  // Update BC list
  bclist = fd->bclist;
  if (fd->bclist->evaluate) {
    ierr = fd->bclist->evaluate(dmPV,x,bclist,bclist->data);CHKERRQ(ierr);
  }

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

  // Boundary conditions - edges and element
  ierr = DMStagBCListApplyFace_Stokes(dmPV,xlocal,dmCoeff,coefflocal,bclist->bc_f,bclist->nbc_face,coordx,coordz,n,ff);CHKERRQ(ierr);
  ierr = DMStagBCListApplyElement_Stokes(dmPV,xlocal,dmCoeff,coefflocal,bclist->bc_e,bclist->nbc_element,coordx,coordz,n,ff);CHKERRQ(ierr);

  // Restore arrays, local vectors
  ierr = DMStagRestore1dCoordinateArraysDOFRead(dmPV,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArrayDOF(dmPV,flocal,&ff); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmPV,&xlocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmCoeff,&coefflocal); CHKERRQ(ierr);

  // Map local to global
  ierr = DMLocalToGlobalBegin(dmPV,flocal,INSERT_VALUES,f); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dmPV,flocal,INSERT_VALUES,f); CHKERRQ(ierr);

  ierr = VecDestroy(&flocal); CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
ContinuityResidual - (STOKES) calculates the continuity residual per dof

Use: internal
@*/
// ---------------------------------------
PetscErrorCode ContinuityResidual(DM dm, Vec xlocal, DM dmcoeff,Vec coefflocal, PetscScalar **coordx, PetscScalar **coordz, PetscInt i, PetscInt j, PetscInt n[],PetscScalar *ff)
{
  PetscScalar    ffi, xx[4], rhs, dx, dz;
  PetscInt       iprev, inext, nEntries = 4;
  DMStagStencil  point[4];
  PetscErrorCode ierr;
  PetscFunctionBegin;

  iprev = n[3]; 
  inext = n[4];

  // Get stencil values
  point[0].i = i; point[0].j = j; point[0].loc = DMSTAG_LEFT;  point[0].c = 0;
  point[1].i = i; point[1].j = j; point[1].loc = DMSTAG_RIGHT; point[1].c = 0;
  point[2].i = i; point[2].j = j; point[2].loc = DMSTAG_DOWN;  point[2].c = 0;
  point[3].i = i; point[3].j = j; point[3].loc = DMSTAG_UP;    point[3].c = 0;
  ierr = DMStagVecGetValuesStencil(dm, xlocal, nEntries, point, xx); CHKERRQ(ierr);
  
  // Coefficients
  point[0].i = i; point[0].j = j; point[0].loc = DMSTAG_ELEMENT;  point[0].c = 0;
  ierr = DMStagVecGetValuesStencil(dmcoeff, coefflocal, 1, point, &rhs); CHKERRQ(ierr);

  // Calculate residual
  dx = coordx[i][inext]-coordx[i][iprev];
  dz = coordz[j][inext]-coordz[j][iprev];
  ffi = (xx[1]-xx[0])/dx + (xx[3]-xx[2])/dz - rhs;

  *ff = ffi;
  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
XMomentumResidual - (STOKES) calculates the Vx momentum residual per dof

Use: internal
@*/
// ---------------------------------------
PetscErrorCode XMomentumResidual(DM dm, Vec xlocal, DM dmcoeff,Vec coefflocal, PetscScalar **coordx,PetscScalar **coordz,PetscInt i, PetscInt j,PetscInt n[],PetscScalar *ff)
{
  PetscScalar    dVxdz, dVzdx, dPdx, dVxdx, rhs, ffi;
  PetscInt       nEntries = 11, Nz, iprev, inext, icenter;
  PetscScalar    xx[11], dx, dx1, dx2, dz, dz1, dz2;
  PetscScalar    etaLeft, etaRight, etaUp, etaDown;
  DMStagStencil  point[11];
  PetscErrorCode ierr;

  PetscFunctionBegin;

  Nz = n[1]; icenter = n[2]; iprev = n[3]; inext = n[4];

  // Get stencil values
  point[0].i  = i  ; point[0].j  = j  ; point[0].loc  = DMSTAG_LEFT;    point[0].c   = 0; // Vx(i  ,j  )
  point[1].i  = i  ; point[1].j  = j-1; point[1].loc  = DMSTAG_LEFT;    point[1].c   = 0; // Vx(i  ,j-1)
  point[2].i  = i  ; point[2].j  = j+1; point[2].loc  = DMSTAG_LEFT;    point[2].c   = 0; // Vx(i  ,j+1)
  point[3].i  = i-1; point[3].j  = j  ; point[3].loc  = DMSTAG_LEFT;    point[3].c   = 0; // Vx(i-1,j  )
  point[4].i  = i  ; point[4].j  = j  ; point[4].loc  = DMSTAG_RIGHT;   point[4].c   = 0; // Vx(i+1,j  )
  point[5].i  = i-1; point[5].j  = j  ; point[5].loc  = DMSTAG_DOWN;    point[5].c   = 0; // Vz(i-1,j-1)
  point[6].i  = i  ; point[6].j  = j  ; point[6].loc  = DMSTAG_DOWN;    point[6].c   = 0; // Vz(i  ,j-1)
  point[7].i  = i-1; point[7].j  = j  ; point[7].loc  = DMSTAG_UP;      point[7].c   = 0; // Vz(i-1,j  )
  point[8].i  = i  ; point[8].j  = j  ; point[8].loc  = DMSTAG_UP;      point[8].c   = 0; // Vz(i  ,j  )
  point[9].i  = i-1; point[9].j  = j  ; point[9].loc  = DMSTAG_ELEMENT; point[9].c   = 0; // P (i-1,j  )
  point[10].i = i  ; point[10].j = j  ; point[10].loc = DMSTAG_ELEMENT; point[10].c  = 0; // P (i  ,j  )

  // For boundaries remove the flux term
  if (j == Nz-1) point[2] = point[0];
  if (j == 0   ) point[1] = point[0];

  // Residual values
  ierr = DMStagVecGetValuesStencil(dm, xlocal, nEntries, point, xx); CHKERRQ(ierr);

  // Coefficients
  PetscScalar  cx[5];
  point[0].i = i  ; point[0].j = j; point[0].loc = DMSTAG_LEFT;      point[0].c = 0; // rhs
  point[1].i = i-1; point[1].j = j; point[1].loc = DMSTAG_ELEMENT;   point[1].c = 1; // etc_c - left
  point[2].i = i  ; point[2].j = j; point[2].loc = DMSTAG_ELEMENT;   point[2].c = 1; // etc_c - right
  point[3].i = i  ; point[3].j = j; point[3].loc = DMSTAG_UP_LEFT;   point[3].c = 0; // etc_n - up
  point[4].i = i  ; point[4].j = j; point[4].loc = DMSTAG_DOWN_LEFT; point[4].c = 0; // etc_n - down

  ierr = DMStagVecGetValuesStencil(dmcoeff, coefflocal, 5, point, cx); CHKERRQ(ierr);

  rhs      = cx[0];
  etaLeft  = cx[1];
  etaRight = cx[2];
  etaUp    = cx[3];
  etaDown  = cx[4];

  // Grid spacings - need to correct for missing values
  dx  = coordx[i  ][icenter]-coordx[i-1][icenter];
  dx1 = coordx[i-1][inext  ]-coordx[i-1][iprev  ];
  dx2 = coordx[i  ][inext  ]-coordx[i  ][iprev  ];
  dz  = coordz[j  ][inext  ]-coordz[j  ][iprev  ];

  // Correct for boundaries
  if (j == 0) {
    dz1 = 2.0*(coordz[j][icenter]-coordz[j][iprev]);
  } else {
    dz1 = coordz[j  ][icenter]-coordz[j-1][icenter];
  }
  if (j == Nz-1) {
    dz2 = 2.0*(coordz[j][inext]-coordz[j][icenter]);
  } else {
    dz2 = coordz[j+1][icenter]-coordz[j  ][icenter];
  }

  // Calculate new residual
  dPdx  = (xx[10]-xx[9])/dx;
  dVxdx = etaRight*(xx[4]-xx[0])/dx2 - etaLeft*(xx[0]-xx[3])/dx1;
  dVxdz = etaUp   *(xx[2]-xx[0])/dz2 - etaDown*(xx[0]-xx[1])/dz1;
  dVzdx = etaUp   *(xx[8]-xx[7])/dx  - etaDown*(xx[6]-xx[5])/dx;

  ffi   = -dPdx + 2.0*dVxdx/dx + dVxdz/dz + dVzdx/dz - rhs;

  *ff = ffi;
  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
ZMomentumResidual - (STOKES) calculates the Vz momentum residual per dof

Use: internal
@*/
// ---------------------------------------
PetscErrorCode ZMomentumResidual(DM dm, Vec xlocal, DM dmcoeff, Vec coefflocal,PetscScalar **coordx,PetscScalar **coordz,PetscInt i, PetscInt j,PetscInt n[],PetscScalar *ff)
{
  PetscScalar    dVxdz, dVzdx, dPdz, dVzdz, rhs, ffi;
  PetscInt       nEntries = 11, Nx, iprev, inext, icenter;
  PetscScalar    xx[11], dx, dz, dx1, dx2, dz1, dz2;
  PetscScalar    etaLeft, etaRight, etaUp, etaDown;
  DMStagStencil  point[11];
  PetscErrorCode ierr;

  PetscFunctionBegin;

  Nx = n[0]; icenter = n[2]; iprev = n[3]; inext = n[4];

  // Get stencil values
  point[0].i  = i  ; point[0].j  = j  ; point[0].loc  = DMSTAG_DOWN;    point[0].c   = 0; // Vz(i  ,j  )
  point[1].i  = i  ; point[1].j  = j  ; point[1].loc  = DMSTAG_UP;      point[1].c   = 0; // Vz(i  ,j+1)
  point[2].i  = i  ; point[2].j  = j-1; point[2].loc  = DMSTAG_DOWN;    point[2].c   = 0; // Vz(i  ,j-1)
  point[3].i  = i-1; point[3].j  = j  ; point[3].loc  = DMSTAG_DOWN;    point[3].c   = 0; // Vz(i-1,j  )
  point[4].i  = i+1; point[4].j  = j  ; point[4].loc  = DMSTAG_DOWN;    point[4].c   = 0; // Vz(i+1,j  )
  point[5].i  = i  ; point[5].j  = j  ; point[5].loc  = DMSTAG_LEFT;    point[5].c   = 0; // Vx(i-1,j  )
  point[6].i  = i  ; point[6].j  = j  ; point[6].loc  = DMSTAG_RIGHT;   point[6].c   = 0; // Vx(i  ,j  )
  point[7].i  = i  ; point[7].j  = j-1; point[7].loc  = DMSTAG_LEFT;    point[7].c   = 0; // Vx(i-1,j-1)
  point[8].i  = i  ; point[8].j  = j-1; point[8].loc  = DMSTAG_RIGHT;   point[8].c   = 0; // Vx(i  ,j-1)
  point[9].i  = i  ; point[9].j  = j-1; point[9].loc  = DMSTAG_ELEMENT; point[9].c   = 0; // P (i  ,j-1)
  point[10].i = i  ; point[10].j = j  ; point[10].loc = DMSTAG_ELEMENT; point[10].c  = 0; // P (i  ,j  )
  
  // For boundaries remove the flux term
  if (i == 0   ) point[3] = point[0];
  if (i == Nx-1) point[4] = point[0];

  // Get values
  ierr = DMStagVecGetValuesStencil(dm, xlocal, nEntries, point, xx); CHKERRQ(ierr);

  // Coefficients
  PetscScalar  cx[5];
  point[0].i = i  ; point[0].j = j  ; point[0].loc = DMSTAG_DOWN;       point[0].c = 0; // rhs
  point[1].i = i  ; point[1].j = j  ; point[1].loc = DMSTAG_DOWN_LEFT;  point[1].c = 0; // etc_n - left
  point[2].i = i  ; point[2].j = j  ; point[2].loc = DMSTAG_DOWN_RIGHT; point[2].c = 0; // etc_n - right
  point[3].i = i  ; point[3].j = j  ; point[3].loc = DMSTAG_ELEMENT;    point[3].c = 1; // etc_c - up
  point[4].i = i  ; point[4].j = j-1; point[4].loc = DMSTAG_ELEMENT;    point[4].c = 1; // etc_c - down

  ierr = DMStagVecGetValuesStencil(dmcoeff, coefflocal, 5, point, cx); CHKERRQ(ierr);

  rhs      = cx[0];
  etaLeft  = cx[1];
  etaRight = cx[2];
  etaUp    = cx[3];
  etaDown  = cx[4];

  // Grid spacings
  dx  = coordx[i  ][inext  ]-coordx[i  ][iprev  ];
  dz  = coordz[j  ][icenter]-coordz[j-1][icenter];
  dz1 = coordz[j-1][inext  ]-coordz[j-1][iprev  ];
  dz2 = coordz[j  ][inext  ]-coordz[j  ][iprev  ];

  // Correct for boundaries
  if (i == 0) {
    dx1 = 2.0*(coordx[i][icenter]-coordx[i][iprev]);
  } else {
    dx1 = coordx[i  ][icenter]-coordx[i-1][icenter];
  }
  if (i == Nx-1) {
    dx2 = 2.0*(coordx[i][inext]-coordx[i][icenter]);
  } else {
    dx2 = coordx[i+1][icenter]-coordx[i  ][icenter];
  }

  // Calculate residual
  dPdz  = (xx[10]-xx[9])/dz;
  dVzdz = etaUp   *(xx[1]-xx[0])/dz2 - etaDown *(xx[0]-xx[2])/dz1;
  dVzdx = etaRight*(xx[4]-xx[0])/dx2 - etaLeft *(xx[0]-xx[3])/dx1;
  dVxdz = etaRight*(xx[6]-xx[8])/dz - etaLeft *(xx[5]-xx[7])/dz;

  ffi   = -dPdz + 2.0*dVzdz/dz + dVzdx/dx + dVxdz/dx - rhs;

  *ff = ffi;

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
DMStagBCListApplyFace_Stokes - (STOKES) function to apply boundary conditions for Stokes equations [flux terms to boundary conditions]

Use: internal
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "DMStagBCListApplyFace_Stokes"
PetscErrorCode DMStagBCListApplyFace_Stokes(DM dm, Vec xlocal,DM dmcoeff, Vec coefflocal, DMStagBC *bclist, PetscInt nbc, PetscScalar **coordx, PetscScalar **coordz,PetscInt n[], PetscScalar ***ff)
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

// ---------------------------------------
/*@
DMStagBCListApplyElement_Stokes - (STOKES) function to apply boundary conditions for Stokes equations [flux terms to boundary conditions]

Use: internal
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "DMStagBCListApplyElement_Stokes"
PetscErrorCode DMStagBCListApplyElement_Stokes(DM dm, Vec xlocal,DM dmcoeff, Vec coefflocal, DMStagBC *bclist, PetscInt nbc, PetscScalar **coordx, PetscScalar **coordz,PetscInt n[], PetscScalar ***ff)
{
  PetscScalar    xx;
  PetscInt       i, j, ibc, idx;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

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
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"BC type NEUMANN for FDPDE_STOKES [ELEMENT] is not yet implemented.");
    }
  }
  PetscFunctionReturn(0);
}