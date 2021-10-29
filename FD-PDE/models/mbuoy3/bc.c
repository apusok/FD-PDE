#include "mbuoy3.h"

// ---------------------------------------
// FormBCList_PV
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormBCList_PV"
PetscErrorCode FormBCList_PV(DM dm, Vec x, DMStagBCList bclist, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       k,n_bc,*idx_bc;
  PetscInt       i, sx, sz, nx, nz, Nx, Nz, iprev, icenter, dm_slot;
  PetscScalar    *value_bc,*x_bc,*x_bc_stag, xx[2], dx;
  BCType         *type_bc;
  PetscScalar    **coordx,**coordz, ***_xlocal;
  Vec            xlocal;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;

  // Get solution dm/vector
  ierr = DMStagGetGlobalSizes(dm,&Nx,&Nz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,LEFT,&iprev);CHKERRQ(ierr);

  // Map global vectors to local domain
  ierr = DMGetLocalVector(dm, &xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm, x, INSERT_VALUES, xlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(dm,xlocal,&_xlocal);CHKERRQ(ierr);

  // RIGHT dVx/dx = 0 (point on true boundary)
  ierr = DMStagBCListGetValues(bclist,'e','-',PV_FACE_VS,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN_T;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',PV_FACE_VS,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN tau_xz = 0, dVx/dz=-dVz/dx (point not on true boundary)
  ierr = DMStagBCListGetValues(bclist,'s','-',PV_FACE_VS,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm,DMSTAG_DOWN,PV_FACE_VS,&dm_slot); CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    for (i = sx; i < sx+nx; i++) {
      if (x_bc[2*k]==coordx[i][iprev]) {
        xx[0] =  _xlocal[0][i-1][dm_slot]; // this is allowed without a parallel check because n_bc can be zero
        xx[1] =  _xlocal[0][i  ][dm_slot];
        dx = coordx[i][icenter] - coordx[i-1][icenter];
        value_bc[k] = -(xx[1]-xx[0])/dx;
        type_bc[k] = BC_NEUMANN;
      }
    }
  }
  ierr = DMStagBCListInsertValues(bclist,'-',PV_FACE_VS,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);

  // LEFT Vx = 0 (point on true boundary)
  ierr = DMStagBCListGetValues(bclist,'w','-',PV_FACE_VS,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET; // BC_DIRICHLET_STAG;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',PV_FACE_VS,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP Vx = u0 (point not on true boundary)
  ierr = DMStagBCListGetValues(bclist,'n','-',PV_FACE_VS,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = usr->nd->U0;
    // value_bc[k] = usr->nd->U0*erf(x_bc_stag[2*k]/usr->nd->xmor);
    type_bc[k] = BC_DIRICHLET; // BC_DIRICHLET_STAG;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',PV_FACE_VS,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);

  // RIGHT dVz/dx = 0 (point not on true boundary)
  ierr = DMStagBCListGetValues(bclist,'e','|',PV_FACE_VS,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',PV_FACE_VS,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN dVz/dz = 0 (point on true boundary)
  ierr = DMStagBCListGetValues(bclist,'s','|',PV_FACE_VS,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN_T;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',PV_FACE_VS,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);

  // LEFT dVz/dx = 0 (point not on true boundary)
  ierr = DMStagBCListGetValues(bclist,'w','|',PV_FACE_VS,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',PV_FACE_VS,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP Vz = 0 (point on true boundary)
  ierr = DMStagBCListGetValues(bclist,'n','|',PV_FACE_VS,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET; // BC_DIRICHLET_STAG;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',PV_FACE_VS,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);

  // LEFT dP/dx = 0
  ierr = DMStagBCListGetValues(bclist,'w','o',PV_ELEMENT_P,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',PV_ELEMENT_P,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP dP/dz = 0
  ierr = DMStagBCListGetValues(bclist,'n','o',PV_ELEMENT_P,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',PV_ELEMENT_P,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);

 // RIGHT P = 0
  ierr = DMStagBCListGetValues(bclist,'e','o',PV_ELEMENT_P,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',PV_ELEMENT_P,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN P = 0
  ierr = DMStagBCListGetValues(bclist,'s','o',PV_ELEMENT_P,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',PV_ELEMENT_P,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);

  // Compaction pressure
  // Left dPc/dx = 0 (or Pc = 0)
  ierr = DMStagBCListGetValues(bclist,'w','o',PV_ELEMENT_PC,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',PV_ELEMENT_PC,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);
  
  // RIGHT Pc = 0
  ierr = DMStagBCListGetValues(bclist,'e','o',PV_ELEMENT_PC,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',PV_ELEMENT_PC,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN Pc = 0
  ierr = DMStagBCListGetValues(bclist,'s','o',PV_ELEMENT_PC,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',PV_ELEMENT_PC,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP dPc/dz = 0 (or Pc = 0)
  ierr = DMStagBCListGetValues(bclist,'n','o',PV_ELEMENT_PC,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',PV_ELEMENT_PC,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);

  // restore
  ierr = DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArrayRead(dm,xlocal,&_xlocal);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&xlocal); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// FormBCList_PV_FullRidge
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormBCList_PV_FullRidge"
PetscErrorCode FormBCList_PV_FullRidge(DM dm, Vec x, DMStagBCList bclist, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       k,n_bc,*idx_bc;
  PetscInt       i, sx, sz, nx, nz, Nx, Nz, iprev, icenter,dm_slot;
  PetscScalar    *value_bc,*x_bc,*x_bc_stag, xx[2], dx;
  BCType         *type_bc;
  PetscScalar    **coordx,**coordz, ***_xlocal;
  Vec            xlocal;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;

  // Get solution dm/vector
  ierr = DMStagGetGlobalSizes(dm,&Nx,&Nz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,LEFT,&iprev);CHKERRQ(ierr);

  // Map global vectors to local domain
  ierr = DMGetLocalVector(dm, &xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm, x, INSERT_VALUES, xlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(dm,xlocal,&_xlocal);CHKERRQ(ierr);

  // RIGHT dVx/dx = 0 (point on true boundary)
  ierr = DMStagBCListGetValues(bclist,'e','-',PV_FACE_VS,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN_T;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',PV_FACE_VS,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN tau_xz = 0, dVx/dz=-dVz/dx (point not on true boundary)
  ierr = DMStagBCListGetValues(bclist,'s','-',PV_FACE_VS,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm,DMSTAG_DOWN,PV_FACE_VS,&dm_slot); CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    for (i = sx; i < sx+nx; i++) {
      if (x_bc[2*k]==coordx[i][iprev]) {
        xx[0] =  _xlocal[0][i-1][dm_slot]; // this is allowed without a parallel check because n_bc can be zero
        xx[1] =  _xlocal[0][i  ][dm_slot];
        dx = coordx[i][icenter] - coordx[i-1][icenter];
        value_bc[k] = -(xx[1]-xx[0])/dx;
        type_bc[k] = BC_NEUMANN;
      }
    }
  }
  ierr = DMStagBCListInsertValues(bclist,'-',PV_FACE_VS,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);

  // LEFT dVx/dx = 0
  ierr = DMStagBCListGetValues(bclist,'w','-',PV_FACE_VS,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN_T;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',PV_FACE_VS,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP Vx = u0 (point not on true boundary)
  ierr = DMStagBCListGetValues(bclist,'n','-',PV_FACE_VS,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    if (x_bc[2*k]< 0.0) value_bc[k] = -usr->nd->U0;
    else                value_bc[k] =  usr->nd->U0;
    type_bc[k] = BC_DIRICHLET; // BC_DIRICHLET_STAG;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',PV_FACE_VS,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);

  // RIGHT dVz/dx = 0 (point not on true boundary)
  ierr = DMStagBCListGetValues(bclist,'e','|',PV_FACE_VS,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',PV_FACE_VS,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN dVz/dz = 0 (point on true boundary)
  ierr = DMStagBCListGetValues(bclist,'s','|',PV_FACE_VS,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN_T;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',PV_FACE_VS,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);

  // LEFT dVz/dx = 0 (point not on true boundary)
  ierr = DMStagBCListGetValues(bclist,'w','|',PV_FACE_VS,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',PV_FACE_VS,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP Vz = 0 (point on true boundary)
  ierr = DMStagBCListGetValues(bclist,'n','|',PV_FACE_VS,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET; // BC_DIRICHLET_STAG;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',PV_FACE_VS,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);

  // LEFT P = 0
  ierr = DMStagBCListGetValues(bclist,'w','o',PV_ELEMENT_P,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',PV_ELEMENT_P,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP dP/dz = 0
  ierr = DMStagBCListGetValues(bclist,'n','o',PV_ELEMENT_P,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',PV_ELEMENT_P,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);

 // RIGHT P = 0
  ierr = DMStagBCListGetValues(bclist,'e','o',PV_ELEMENT_P,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',PV_ELEMENT_P,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN P = 0
  ierr = DMStagBCListGetValues(bclist,'s','o',PV_ELEMENT_P,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',PV_ELEMENT_P,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);

  // Compaction pressure
  // Left Pc = 0
  ierr = DMStagBCListGetValues(bclist,'w','o',PV_ELEMENT_PC,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',PV_ELEMENT_PC,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);
  
  // RIGHT Pc = 0
  ierr = DMStagBCListGetValues(bclist,'e','o',PV_ELEMENT_PC,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',PV_ELEMENT_PC,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN Pc = 0
  ierr = DMStagBCListGetValues(bclist,'s','o',PV_ELEMENT_PC,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',PV_ELEMENT_PC,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP dPc/dz = 0 (or Pc = 0)
  ierr = DMStagBCListGetValues(bclist,'n','o',PV_ELEMENT_PC,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',PV_ELEMENT_PC,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);

  // restore
  ierr = DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArrayRead(dm,xlocal,&_xlocal);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&xlocal); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// FormBCList_HC
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormBCList_HC"
PetscErrorCode FormBCList_HC(DM dm, Vec x, DMStagBCList bclist, void *ctx)
{
  UsrData     *usr = (UsrData*)ctx;
  PetscInt    k,n_bc,*idx_bc;
  PetscScalar *value_bc,*x_bc;
  PetscScalar Hp, Hc, age, T, Tp, Ts, Tm, scalx, scalv, u0, kappa, xmor;
  BCType      *type_bc;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  scalx = usr->scal->x;
  scalv = usr->scal->v;
  u0 = usr->nd->U0;
  kappa = usr->par->kappa;
  Tp = usr->par->Tp;
  Ts = usr->par->Ts;
  Tm  = (usr->par->Tp-T_KELVIN)*exp(-usr->nd->A*usr->nd->zmin)+T_KELVIN;
  xmor = usr->nd->xmor;

  // ENTHALPY
  // LEFT: dH/dx = 0
  ierr = DMStagBCListGetValues(bclist,'w','o',HC_ELEMENT_H,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',HC_ELEMENT_H,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // RIGHT: dH/dx = 0 
  ierr = DMStagBCListGetValues(bclist,'e','o',HC_ELEMENT_H,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',HC_ELEMENT_H,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN: H = Hp, enthalpy corresponding to the potential temperature at zero porosity (HP)
  ierr = DMStagBCListGetValues(bclist,'s','o',HC_ELEMENT_H,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    age = dim_param(x_bc[2*k]-xmor,scalx)/dim_param(u0,scalv);
    if (age <= 0.0) age = dim_param(x_bc[2*0],scalx)/dim_param(u0,scalv);
    T = HalfSpaceCoolingTemp(Tm,Ts,-dim_param(x_bc[2*k+1],scalx),kappa,age,usr->par->hs_factor); 
    Hp = (T - usr->par->T0)/usr->par->DT;
    value_bc[k] = Hp; 
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',HC_ELEMENT_H,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP: H = Hc, Hc is enthalpy corresponding to 0 deg C
  ierr = DMStagBCListGetValues(bclist,'n','o',HC_ELEMENT_H,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    age = dim_param(x_bc[2*k]-xmor,scalx)/dim_param(u0,scalv);
    if (age <= 0.0) age = dim_param(x_bc[2*0],scalx)/dim_param(u0,scalv);
    T = HalfSpaceCoolingTemp(Tm,Ts,-dim_param(x_bc[2*k+1],scalx),kappa,age,usr->par->hs_factor); 
    Hc = (T - usr->par->T0)/usr->par->DT;
    value_bc[k] = Hc;
    type_bc[k] = BC_DIRICHLET;
  }
  
  // x<=xmor: dH/dz = 0
  for (k=0; k<n_bc; k++) {
    if (x_bc[2*k]<=xmor) {
      value_bc[k] = 0.0;
      type_bc[k] = BC_NEUMANN;
    } 
  }
  ierr = DMStagBCListInsertValues(bclist,'o',HC_ELEMENT_H,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // COMPOSITION
  // LEFT: dC/dx = 0
  ierr = DMStagBCListGetValues(bclist,'w','o',HC_ELEMENT_C,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',HC_ELEMENT_C,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // RIGHT: dC/dx = 0
  ierr = DMStagBCListGetValues(bclist,'e','o',HC_ELEMENT_C,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',HC_ELEMENT_C,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN: C = C0 (dim) C = 0 (non-dim)
  ierr = DMStagBCListGetValues(bclist,'s','o',HC_ELEMENT_C,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',HC_ELEMENT_C,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP: dC/dz = 0
  ierr = DMStagBCListGetValues(bclist,'n','o',HC_ELEMENT_C,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',HC_ELEMENT_C,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// FormBCList_HC_FullRidge
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormBCList_HC_FullRidge"
PetscErrorCode FormBCList_HC_FullRidge(DM dm, Vec x, DMStagBCList bclist, void *ctx)
{
  UsrData     *usr = (UsrData*)ctx;
  PetscInt    k,n_bc,*idx_bc;
  PetscScalar *value_bc,*x_bc;
  PetscScalar Hp, Hc, age, T, Tp, Ts, Tm, scalx, scalv, u0, kappa, xmor;
  BCType      *type_bc;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  scalx = usr->scal->x;
  scalv = usr->scal->v;
  u0 = usr->nd->U0;
  kappa = usr->par->kappa;
  Tp = usr->par->Tp;
  Ts = usr->par->Ts;
  Tm  = (usr->par->Tp-T_KELVIN)*exp(-usr->nd->A*usr->nd->zmin)+T_KELVIN;
  xmor = usr->nd->xmor;

  // ENTHALPY
  // LEFT: dH/dx = 0
  ierr = DMStagBCListGetValues(bclist,'w','o',HC_ELEMENT_H,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',HC_ELEMENT_H,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // RIGHT: dH/dx = 0 
  ierr = DMStagBCListGetValues(bclist,'e','o',HC_ELEMENT_H,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',HC_ELEMENT_H,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN: H = Hp, enthalpy corresponding to the potential temperature at zero porosity (HP)
  ierr = DMStagBCListGetValues(bclist,'s','o',HC_ELEMENT_H,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    age = dim_param(fabs(x_bc[2*k])-xmor,scalx)/dim_param(u0,scalv);
    if (age <= 0.0) T = Tm;
    else T = HalfSpaceCoolingTemp(Tm,Ts,-dim_param(x_bc[2*k+1],scalx),kappa,age,usr->par->hs_factor); 
    Hp = (T - usr->par->T0)/usr->par->DT;
    value_bc[k] = Hp; 
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',HC_ELEMENT_H,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP: H = Hc, Hc is enthalpy corresponding to 0 deg C
  ierr = DMStagBCListGetValues(bclist,'n','o',HC_ELEMENT_H,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    age = dim_param(fabs(x_bc[2*k])-xmor,scalx)/dim_param(u0,scalv);
    if (age <= 0.0) T = Tm;
    else T = HalfSpaceCoolingTemp(Tm,Ts,-dim_param(x_bc[2*k+1],scalx),kappa,age,usr->par->hs_factor); 
    Hc = (T - usr->par->T0)/usr->par->DT;
    value_bc[k] = Hc;
    type_bc[k] = BC_DIRICHLET;
  }
  
  // fabs(x)<=xmor: dH/dz = 0
  for (k=0; k<n_bc; k++) {
    if (fabs(x_bc[2*k])<=xmor) {
      value_bc[k] = 0.0;
      type_bc[k] = BC_NEUMANN;
    } 
  }
  ierr = DMStagBCListInsertValues(bclist,'o',HC_ELEMENT_H,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // COMPOSITION
  // LEFT: dC/dx = 0
  ierr = DMStagBCListGetValues(bclist,'w','o',HC_ELEMENT_C,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',HC_ELEMENT_C,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // RIGHT: dC/dx = 0
  ierr = DMStagBCListGetValues(bclist,'e','o',HC_ELEMENT_C,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',HC_ELEMENT_C,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN: C = C0 (dim) C = 0 (non-dim)
  ierr = DMStagBCListGetValues(bclist,'s','o',HC_ELEMENT_C,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',HC_ELEMENT_C,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP: dC/dz = 0
  ierr = DMStagBCListGetValues(bclist,'n','o',HC_ELEMENT_C,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',HC_ELEMENT_C,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}