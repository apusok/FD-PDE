#include "morfault.h"

// ---------------------------------------
// FormBCList_PV
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormBCList_PV"
PetscErrorCode FormBCList_PV(DM dm, Vec x, DMStagBCList bclist, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       k, n_bc, *idx_bc, sx, sz, nx, nz, Nx, Nz, j, icenter;
  PetscScalar    *value_bc,*x_bc,vext,***xwt,**coordx,**coordz;
  PetscScalar    zl, zr, Hl, Hr;
  BCType         *type_bc;
  Vec            xMPhaselocal;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  vext = usr->nd->Vext;

    if (usr->par->inflow_bc>0) { // top and bottom inflow
    ierr = DMStagGetGlobalSizes(dm,&Nx,&Nz,NULL);CHKERRQ(ierr);
    ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
    ierr = DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
    ierr = DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter);CHKERRQ(ierr);

    // get material phase fractions
    ierr = DMCreateLocalVector(usr->dmMPhase, &xMPhaselocal); CHKERRQ(ierr);
    ierr = DMGlobalToLocal(usr->dmMPhase,usr->xMPhase,INSERT_VALUES,xMPhaselocal); CHKERRQ(ierr);
    ierr = DMStagVecGetArray(usr->dmMPhase, xMPhaselocal, &xwt); CHKERRQ(ierr);

    PetscInt iwtl,iwtr;
    ierr = DMStagGetLocationSlot(usr->dmMPhase, LEFT, 0, &iwtl); CHKERRQ(ierr);
    ierr = DMStagGetLocationSlot(usr->dmMPhase, RIGHT, 0, &iwtr); CHKERRQ(ierr);

    zl = usr->nd->H + usr->nd->zmin;
    zr = usr->nd->H + usr->nd->zmin;
    for (j=sz; j<sz+nz; j++) {
      if (xwt[j][0   ][iwtl]>0.0) zl = PetscMin(zl,coordz[j][icenter]);
      if (xwt[j][Nx-1][iwtr]>0.0) zr = PetscMin(zr,coordz[j][icenter]);
    }

    Hl = usr->nd->H + usr->nd->zmin - zl;
    Hr = usr->nd->H + usr->nd->zmin - zr;

    usr->nd->Vin_free = vext*(Hl+Hr)/usr->nd->L;
    usr->nd->Vin_rock = usr->nd->Vin - usr->nd->Vin_free;

    ierr = DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
    ierr = DMStagVecRestoreArray(usr->dmMPhase,xMPhaselocal,&xwt);CHKERRQ(ierr);
    ierr = VecDestroy(&xMPhaselocal); CHKERRQ(ierr);
  }

  // Vx  
  // DOWN Boundary: dVx/dz = 0
  ierr = DMStagBCListGetValues(bclist,'s','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP Boundary: dVx/dz = 0
  ierr = DMStagBCListGetValues(bclist,'n','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // LEFT Boundary: Vx = -Vext
  ierr = DMStagBCListGetValues(bclist,'w','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = -vext;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // RIGHT Boundary: Vx = Vext
  ierr = DMStagBCListGetValues(bclist,'e','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = vext;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // Vz
  // LEFT Boundary: dVz/dx = 0
  ierr = DMStagBCListGetValues(bclist,'w','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // RIGHT Boundary: dVz/dx = 0
  ierr = DMStagBCListGetValues(bclist,'e','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN Boundary: Vz = Vin
  ierr = DMStagBCListGetValues(bclist,'s','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    // value_bc[k] = usr->nd->Vin;
    value_bc[k] = usr->nd->Vin_rock;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP Boundary: Vz = 0
  ierr = DMStagBCListGetValues(bclist,'n','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    // value_bc[k] = 0.0;
    value_bc[k] = -usr->nd->Vin_free;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // p
  // LEFT Boundary: dp/dz = 0
  ierr = DMStagBCListGetValues(bclist,'w','o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  
  // RIGHT Boundary: dp/dz = 0
  ierr = DMStagBCListGetValues(bclist,'e','o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN Boundary: p = 0
  ierr = DMStagBCListGetValues(bclist,'s','o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP Boundary: dp/dz = 0
  ierr = DMStagBCListGetValues(bclist,'n','o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET_STAG; // p = 0
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

// ---------------------------------------
PetscErrorCode FormBCList_PV_Stokes(DM dm, Vec x, DMStagBCList bclist, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       k, n_bc, *idx_bc, sx, sz, nx, nz, Nx, Nz, j, icenter;
  PetscScalar    *value_bc,*x_bc,vext,***xwt,**coordx,**coordz;
  PetscScalar    zl, zr, Hl, Hr;
  BCType         *type_bc;
  Vec            xMPhaselocal;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  vext = usr->nd->Vext;

  if (usr->par->inflow_bc>0) { // top and bottom inflow
    ierr = DMStagGetGlobalSizes(dm,&Nx,&Nz,NULL);CHKERRQ(ierr);
    ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
    ierr = DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
    ierr = DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter);CHKERRQ(ierr);

    // get material phase fractions
    ierr = DMCreateLocalVector(usr->dmMPhase, &xMPhaselocal); CHKERRQ(ierr);
    ierr = DMGlobalToLocal(usr->dmMPhase,usr->xMPhase,INSERT_VALUES,xMPhaselocal); CHKERRQ(ierr);
    ierr = DMStagVecGetArray(usr->dmMPhase, xMPhaselocal, &xwt); CHKERRQ(ierr);

    PetscInt iwtl,iwtr;
    ierr = DMStagGetLocationSlot(usr->dmMPhase, LEFT, 0, &iwtl); CHKERRQ(ierr);
    ierr = DMStagGetLocationSlot(usr->dmMPhase, RIGHT, 0, &iwtr); CHKERRQ(ierr);

    zl = usr->nd->H + usr->nd->zmin;
    zr = usr->nd->H + usr->nd->zmin;
    for (j=sz; j<sz+nz; j++) {
      if (xwt[j][0   ][iwtl]>0.0) zl = PetscMin(zl,coordz[j][icenter]);
      if (xwt[j][Nx-1][iwtr]>0.0) zr = PetscMin(zr,coordz[j][icenter]);
    }

    Hl = usr->nd->H + usr->nd->zmin - zl;
    Hr = usr->nd->H + usr->nd->zmin - zr;

    usr->nd->Vin_free = vext*(Hl+Hr)/usr->nd->L;
    usr->nd->Vin_rock = usr->nd->Vin - usr->nd->Vin_free;

    ierr = DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
    ierr = DMStagVecRestoreArray(usr->dmMPhase,xMPhaselocal,&xwt);CHKERRQ(ierr);
    ierr = VecDestroy(&xMPhaselocal); CHKERRQ(ierr);
  }

  // Vx  
  // DOWN Boundary: dVx/dz = 0
  ierr = DMStagBCListGetValues(bclist,'s','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP Boundary: dVx/dz = 0
  ierr = DMStagBCListGetValues(bclist,'n','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // LEFT Boundary: Vx = -Vext
  ierr = DMStagBCListGetValues(bclist,'w','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = -vext;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // RIGHT Boundary: Vx = Vext
  ierr = DMStagBCListGetValues(bclist,'e','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = vext;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // Vz
  // LEFT Boundary: dVz/dx = 0
  ierr = DMStagBCListGetValues(bclist,'w','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // RIGHT Boundary: dVz/dx = 0
  ierr = DMStagBCListGetValues(bclist,'e','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN Boundary: Vz = Vin
  ierr = DMStagBCListGetValues(bclist,'s','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    // value_bc[k] = usr->nd->Vin;
    value_bc[k] = usr->nd->Vin_rock;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP Boundary: Vz = -Vin_free
  ierr = DMStagBCListGetValues(bclist,'n','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    // value_bc[k] = 0.0;
    value_bc[k] = -usr->nd->Vin_free;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP Boundary: pin P = 0
  ierr = DMStagBCListGetValues(bclist,'n','o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<1; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// FormBCList_T
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormBCList_T"
PetscErrorCode FormBCList_T(DM dm, Vec x, DMStagBCList bclist, void *ctx)
{
  UsrData     *usr = (UsrData*)ctx;
  PetscInt    k,n_bc,*idx_bc;
  PetscScalar *value_bc, *x_bc, Tbot, nd_T, age;
  BCType      *type_bc;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  // Left: dT/dx=0
  ierr = DMStagBCListGetValues(bclist,'w','o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  
  // RIGHT: dT/dx=0
  ierr = DMStagBCListGetValues(bclist,'e','o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN: T = Tbot
  ierr = DMStagBCListGetValues(bclist,'s','o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    // value_bc[k] = usr->nd->Tbot;
    
    if ((usr->par->model_setup<=1) || (usr->par->model_setup==3) || (usr->par->model_setup==5) || (usr->par->model_setup==10)) age = usr->par->age*1.0e6*SEC_YEAR; // constant age
    else age  = usr->par->age*1.0e6*SEC_YEAR + dim_param(fabs(x_bc[2*k]),usr->scal->x)/dim_param(usr->nd->uT,usr->scal->v); // variable age

    Tbot = HalfSpaceCoolingTemp(usr->par->Tbot,usr->par->Ttop,usr->par->H-usr->par->Hs,usr->scal->kappa,age,usr->par->hs_factor); 
    
    // constant initial T
    if (usr->par->model_setup==4) { Tbot = usr->par->Tinit;}

    nd_T = nd_paramT(Tbot,usr->par->Ttop,usr->scal->DT);
    value_bc[k] = nd_T;
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP: T = Ttop
  ierr = DMStagBCListGetValues(bclist,'n','o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = usr->nd->Ttop;
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
 
  PetscFunctionReturn(0);
}

// ---------------------------------------
// FormBCList_phi
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormBCList_phi"
PetscErrorCode FormBCList_phi(DM dm, Vec x, DMStagBCList bclist, void *ctx)
{
  UsrData     *usr = (UsrData*)ctx;
  PetscInt    k,n_bc,*idx_bc;
  PetscScalar *value_bc,*x_bc, phi, phi_max,sigma;
  BCType      *type_bc;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  phi_max = usr->par->phi_max_bc; // 1e-3;
  sigma   = usr->par->sigma_bc;   // 0.1 - 0.001;
  
  // Left: dphis/dx = 0
  ierr = DMStagBCListGetValues(bclist,'w','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // RIGHT: dphis/dx = 0
  ierr = DMStagBCListGetValues(bclist,'e','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN: phis = 1.0 - func(phi0)
  ierr = DMStagBCListGetValues(bclist,'s','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    // phi = usr->par->phi0;
    if (usr->par->model_setup<5) phi = usr->par->phi0 + phi_max*PetscExpScalar(-x_bc[2*k]*x_bc[2*k]/sigma);
    else phi = 0.0;
    value_bc[k] = 1.0 - phi;
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP: dphis/dz = 0 
  ierr = DMStagBCListGetValues(bclist,'n','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  // for (k=0; k<n_bc; k++) {
  //   value_bc[k] = 0.0;
  //   type_bc[k] = BC_NEUMANN;
  // }
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 1.0;
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
 
  PetscFunctionReturn(0);
}