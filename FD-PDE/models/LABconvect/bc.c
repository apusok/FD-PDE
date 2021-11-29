#include "LABconvect.h"

// ---------------------------------------
// FormBCList_PV
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormBCList_PV"
PetscErrorCode FormBCList_PV(DM dm, Vec x, DMStagBCList bclist, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       k,n_bc,*idx_bc;
  PetscInt       i, sx, sz, nx, nz, Nx, Nz, iprev, icenter;
  PetscScalar    *value_bc,*x_bc,*x_bc_stag;
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

  // RIGHT Vx = 0 (point on true boundary)
  ierr = DMStagBCListGetValues(bclist,'e','-',PV_FACE_VS,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',PV_FACE_VS,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN (free slip) dVx/dz = 0, (point not on true boundary)
  ierr = DMStagBCListGetValues(bclist,'s','-',PV_FACE_VS,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',PV_FACE_VS,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);

  // LEFT Vx = 0 (point on true boundary)
  ierr = DMStagBCListGetValues(bclist,'w','-',PV_FACE_VS,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET; 
  }
  ierr = DMStagBCListInsertValues(bclist,'-',PV_FACE_VS,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP dVx/dz = (point not on true boundary)
  ierr = DMStagBCListGetValues(bclist,'n','-',PV_FACE_VS,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN; 
  }
  ierr = DMStagBCListInsertValues(bclist,'-',PV_FACE_VS,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);

  // RIGHT dVz/dx = 0 (point not on true boundary)
  ierr = DMStagBCListGetValues(bclist,'e','|',PV_FACE_VS,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',PV_FACE_VS,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN (free slip) Vz = 0 (point on true boundary)
  ierr = DMStagBCListGetValues(bclist,'s','|',PV_FACE_VS,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;
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

  // UP P = 0
  ierr = DMStagBCListGetValues(bclist,'n','o',PV_ELEMENT_P,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',PV_ELEMENT_P,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);

 // RIGHT dP/dx = 0
  ierr = DMStagBCListGetValues(bclist,'e','o',PV_ELEMENT_P,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',PV_ELEMENT_P,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN dP/dz = 0
  ierr = DMStagBCListGetValues(bclist,'s','o',PV_ELEMENT_P,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',PV_ELEMENT_P,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);

  // Compaction pressure
  // Left dPc/dx = 0
  ierr = DMStagBCListGetValues(bclist,'w','o',PV_ELEMENT_PC,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',PV_ELEMENT_PC,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);
  
  // RIGHT dPc/dx = 0
  ierr = DMStagBCListGetValues(bclist,'e','o',PV_ELEMENT_PC,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',PV_ELEMENT_PC,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN dPc/dz = 0
  ierr = DMStagBCListGetValues(bclist,'s','o',PV_ELEMENT_PC,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',PV_ELEMENT_PC,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP Pc = 0
  ierr = DMStagBCListGetValues(bclist,'n','o',PV_ELEMENT_PC,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET_STAG;
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
  PetscScalar Hp, Hc, age, T, Tp, Ts, Tm, scalx, kappa;
  BCType      *type_bc;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  scalx = usr->scal->x;
  kappa = usr->par->kappa;
  Tp = usr->par->Tp;
  Ts = usr->par->Ts;
  Tm  = (usr->par->Tp-T_KELVIN)*exp(-usr->nd->A*usr->nd->zmin)+T_KELVIN;

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

  // DOWN: H = Hbottom = Hp+DT_bottom
  ierr = DMStagBCListGetValues(bclist,'s','o',HC_ELEMENT_H,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    age = usr->nd->age*usr->scal->t;
    T = HalfSpaceCoolingTemp(Tm,Ts,-dim_param(x_bc[2*k+1],scalx),kappa,age); 
    Hp = (T + usr->par->DT_bottom - usr->par->T0)/usr->par->DT;
    value_bc[k] = Hp; 
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',HC_ELEMENT_H,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP: H = Hc, Hc is enthalpy corresponding to 0 deg C
  ierr = DMStagBCListGetValues(bclist,'n','o',HC_ELEMENT_H,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    age = usr->nd->age*usr->scal->t;
    T = HalfSpaceCoolingTemp(Tm,Ts,-dim_param(x_bc[2*k+1],scalx),kappa,age); 
    Hc = (T - usr->par->T0)/usr->par->DT;
    value_bc[k] = Hc;
    type_bc[k] = BC_DIRICHLET;
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

  // UP: C = Cdepl (or dC/dz = 0)
  ierr = DMStagBCListGetValues(bclist,'n','o',HC_ELEMENT_C,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = (usr->par->C0+usr->par->DC*usr->par->depletion-usr->par->C0)/usr->par->DC;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',HC_ELEMENT_C,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// FormBCList_PV_InflowOutflowBottom
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormBCList_PV_InflowOutflowBottom"
PetscErrorCode FormBCList_PV_InflowOutflowBottom(DM dm, Vec x, DMStagBCList bclist, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       k,n_bc,*idx_bc;
  PetscInt       i, sx, sz, nx, nz, Nx, Nz, iprev, icenter;
  PetscScalar    *value_bc,*x_bc,*x_bc_stag;
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

  // RIGHT Vx = 0 (point on true boundary)
  ierr = DMStagBCListGetValues(bclist,'e','-',PV_FACE_VS,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',PV_FACE_VS,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN inflow/outflow dVx/dz = 0, (point not on true boundary)
  ierr = DMStagBCListGetValues(bclist,'s','-',PV_FACE_VS,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',PV_FACE_VS,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);

  // LEFT Vx = 0 (point on true boundary)
  ierr = DMStagBCListGetValues(bclist,'w','-',PV_FACE_VS,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET; 
  }
  ierr = DMStagBCListInsertValues(bclist,'-',PV_FACE_VS,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP dVx/dz = (point not on true boundary)
  ierr = DMStagBCListGetValues(bclist,'n','-',PV_FACE_VS,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN; 
  }
  ierr = DMStagBCListInsertValues(bclist,'-',PV_FACE_VS,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);

  // RIGHT dVz/dx = 0 (point not on true boundary)
  ierr = DMStagBCListGetValues(bclist,'e','|',PV_FACE_VS,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',PV_FACE_VS,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN inflow/outflow dVz/dz = 0 (point on true boundary)
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

  // UP P = 0
  ierr = DMStagBCListGetValues(bclist,'n','o',PV_ELEMENT_P,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',PV_ELEMENT_P,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);

 // RIGHT dP/dx = 0
  ierr = DMStagBCListGetValues(bclist,'e','o',PV_ELEMENT_P,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',PV_ELEMENT_P,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN P=0 
  ierr = DMStagBCListGetValues(bclist,'s','o',PV_ELEMENT_P,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',PV_ELEMENT_P,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);

  // Compaction pressure
  // Left dPc/dx = 0
  ierr = DMStagBCListGetValues(bclist,'w','o',PV_ELEMENT_PC,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',PV_ELEMENT_PC,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);
  
  // RIGHT dPc/dx = 0
  ierr = DMStagBCListGetValues(bclist,'e','o',PV_ELEMENT_PC,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',PV_ELEMENT_PC,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN Pc=0 
  ierr = DMStagBCListGetValues(bclist,'s','o',PV_ELEMENT_PC,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',PV_ELEMENT_PC,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP Pc = 0
  ierr = DMStagBCListGetValues(bclist,'n','o',PV_ELEMENT_PC,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',PV_ELEMENT_PC,&n_bc,&idx_bc,&x_bc,&x_bc_stag,&value_bc,&type_bc);CHKERRQ(ierr);

  // restore
  ierr = DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArrayRead(dm,xlocal,&_xlocal);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&xlocal); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// FormBCList_HC_InflowOutflowBottom
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormBCList_HC_InflowOutflowBottom"
PetscErrorCode FormBCList_HC_InflowOutflowBottom(DM dm, Vec x, DMStagBCList bclist, void *ctx)
{
  UsrData     *usr = (UsrData*)ctx;
  DM          dmPV;
  Vec         xPV, xPVlocal;
  PetscInt    i, Nx, Nz, sx, sz, nx, nz, icenter, dm_slot;
  PetscInt    k,n_bc,*idx_bc;
  PetscScalar *value_bc,*x_bc, **coordx, **coordz, ***_xPVlocal;
  PetscScalar Hp, Hc, age, T, Tp, Ts, Tm, scalx, kappa, vsz;
  BCType      *type_bc;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  scalx = usr->scal->x;
  kappa = usr->par->kappa;
  Tp = usr->par->Tp;
  Ts = usr->par->Ts;
  Tm  = (usr->par->Tp-T_KELVIN)*exp(-usr->nd->A*usr->nd->zmin)+T_KELVIN;

  dmPV = usr->dmPV;
  xPV  = usr->xPV;

  // Get PV solution dm/vector
  ierr = DMStagGetGlobalSizes(dmPV,&Nx,&Nz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dmPV, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateArraysRead(dmPV,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmPV,ELEMENT,&icenter);CHKERRQ(ierr);

  // Map global vectors to local domain
  ierr = DMGetLocalVector(dmPV, &xPVlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dmPV, xPV, INSERT_VALUES, xPVlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(dmPV,xPVlocal,&_xPVlocal);CHKERRQ(ierr);

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

  // DOWN: H = Hbottom = Hp+DT_bottom if vz>0, dH/dz=0 if vz<0
  ierr = DMStagGetLocationSlot(dmPV,DMSTAG_DOWN,PV_FACE_VS,&dm_slot); CHKERRQ(ierr);
  ierr = DMStagBCListGetValues(bclist,'s','o',HC_ELEMENT_H,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    for (i = sx; i < sx+nx; i++) {
      if (x_bc[2*k]==coordx[i][icenter]) {
        vsz = _xPVlocal[0][i][dm_slot];
        if (vsz>=0.0) { //H = Hbottom
          age = usr->nd->age*usr->scal->t;
          T = HalfSpaceCoolingTemp(Tm,Ts,-dim_param(x_bc[2*k+1],scalx),kappa,age); 
          Hp = (T + usr->par->DT_bottom - usr->par->T0)/usr->par->DT;
          value_bc[k] = Hp; 
          type_bc[k] = BC_DIRICHLET;
        } else { // dH/dz = 0
          value_bc[k] = 0.0; 
          type_bc[k] = BC_NEUMANN;
        }
      }
    }
  }
  ierr = DMStagBCListInsertValues(bclist,'o',HC_ELEMENT_H,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP: H = Hc, Hc is enthalpy corresponding to 0 deg C
  ierr = DMStagBCListGetValues(bclist,'n','o',HC_ELEMENT_H,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    age = usr->nd->age*usr->scal->t;
    T = HalfSpaceCoolingTemp(Tm,Ts,-dim_param(x_bc[2*k+1],scalx),kappa,age); 
    Hc = (T - usr->par->T0)/usr->par->DT;
    value_bc[k] = Hc;
    type_bc[k] = BC_DIRICHLET;
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

  // DOWN: C = C0 (dim) C = 0 (non-dim) if vz>0, dC/dz=0 if vz<0
  ierr = DMStagBCListGetValues(bclist,'s','o',HC_ELEMENT_C,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    for (i = sx; i < sx+nx; i++) {
      if (x_bc[2*k]==coordx[i][icenter]) {
        vsz = _xPVlocal[0][i][dm_slot];
        if (vsz>=0.0) { // C = C0
          value_bc[k] = 0.0;
          type_bc[k] = BC_DIRICHLET;
        } else { // dC/dz = 0
          value_bc[k] = 0.0; 
          type_bc[k] = BC_NEUMANN;
        }
      }
    }
  }
  ierr = DMStagBCListInsertValues(bclist,'o',HC_ELEMENT_C,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP: C = Cdepl (or dC/dz = 0)
  ierr = DMStagBCListGetValues(bclist,'n','o',HC_ELEMENT_C,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = (usr->par->C0+usr->par->DC*usr->par->depletion-usr->par->C0)/usr->par->DC;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',HC_ELEMENT_C,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // restore
  ierr = DMStagRestoreProductCoordinateArraysRead(dmPV,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArrayRead(dmPV,xPVlocal,&_xPVlocal);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmPV,&xPVlocal); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}