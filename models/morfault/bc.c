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
  PetscFunctionBeginUser;

  vext = usr->nd->Vext;

  if (usr->par->inflow_bc>0) { // top and bottom inflow
    PetscCall(DMStagGetGlobalSizes(dm,&Nx,&Nz,NULL));
    PetscCall(DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 
    PetscCall(DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));
    PetscCall(DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter));

    // get material phase fractions
    PetscCall(DMCreateLocalVector(usr->dmMPhase, &xMPhaselocal)); 
    PetscCall(DMGlobalToLocal(usr->dmMPhase,usr->xMPhase,INSERT_VALUES,xMPhaselocal)); 
    PetscCall(DMStagVecGetArray(usr->dmMPhase, xMPhaselocal, &xwt)); 

    PetscInt iwtl,iwtr;
    PetscCall(DMStagGetLocationSlot(usr->dmMPhase, LEFT, 0, &iwtl)); 
    PetscCall(DMStagGetLocationSlot(usr->dmMPhase, RIGHT, 0, &iwtr)); 

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

    PetscCall(DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));
    PetscCall(DMStagVecRestoreArray(usr->dmMPhase,xMPhaselocal,&xwt));
    PetscCall(VecDestroy(&xMPhaselocal)); 
  }

  // Vx  
  // DOWN Boundary: dVx/dz = 0
  PetscCall(DMStagBCListGetValues(bclist,'s','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // UP Boundary: dVx/dz = 0
  PetscCall(DMStagBCListGetValues(bclist,'n','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // LEFT Boundary: Vx = -Vext
  PetscCall(DMStagBCListGetValues(bclist,'w','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = -vext;
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // RIGHT Boundary: Vx = Vext
  PetscCall(DMStagBCListGetValues(bclist,'e','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = vext;
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // Vz
  // LEFT Boundary: dVz/dx = 0
  PetscCall(DMStagBCListGetValues(bclist,'w','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // RIGHT Boundary: dVz/dx = 0
  PetscCall(DMStagBCListGetValues(bclist,'e','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // DOWN Boundary: Vz = Vin_rock
  PetscCall(DMStagBCListGetValues(bclist,'s','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = usr->nd->Vin_rock;
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // UP Boundary: Vz = Vin_free_surface
  PetscCall(DMStagBCListGetValues(bclist,'n','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = -usr->nd->Vin_free;
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // p
  // LEFT Boundary: dp/dz = 0
  PetscCall(DMStagBCListGetValues(bclist,'w','o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  
  // RIGHT Boundary: dp/dz = 0
  PetscCall(DMStagBCListGetValues(bclist,'e','o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // DOWN Boundary: p = 0
  PetscCall(DMStagBCListGetValues(bclist,'s','o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // UP Boundary: dp/dz = 0
  PetscCall(DMStagBCListGetValues(bclist,'n','o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET_STAG; // p = 0
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscFunctionBeginUser;

  vext = usr->nd->Vext;

  if (usr->par->inflow_bc>0) { // top and bottom inflow
    PetscCall(DMStagGetGlobalSizes(dm,&Nx,&Nz,NULL));
    PetscCall(DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 
    PetscCall(DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));
    PetscCall(DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter));

    // get material phase fractions
    PetscCall(DMCreateLocalVector(usr->dmMPhase, &xMPhaselocal)); 
    PetscCall(DMGlobalToLocal(usr->dmMPhase,usr->xMPhase,INSERT_VALUES,xMPhaselocal)); 
    PetscCall(DMStagVecGetArray(usr->dmMPhase, xMPhaselocal, &xwt)); 

    PetscInt iwtl,iwtr;
    PetscCall(DMStagGetLocationSlot(usr->dmMPhase, LEFT, 0, &iwtl)); 
    PetscCall(DMStagGetLocationSlot(usr->dmMPhase, RIGHT, 0, &iwtr)); 

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

    PetscCall(DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));
    PetscCall(DMStagVecRestoreArray(usr->dmMPhase,xMPhaselocal,&xwt));
    PetscCall(VecDestroy(&xMPhaselocal)); 
  }

  // Vx  
  // DOWN Boundary: dVx/dz = 0
  PetscCall(DMStagBCListGetValues(bclist,'s','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // UP Boundary: dVx/dz = 0
  PetscCall(DMStagBCListGetValues(bclist,'n','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // LEFT Boundary: Vx = -Vext
  PetscCall(DMStagBCListGetValues(bclist,'w','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = -vext;
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // RIGHT Boundary: Vx = Vext
  PetscCall(DMStagBCListGetValues(bclist,'e','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = vext;
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // Vz
  // LEFT Boundary: dVz/dx = 0
  PetscCall(DMStagBCListGetValues(bclist,'w','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // RIGHT Boundary: dVz/dx = 0
  PetscCall(DMStagBCListGetValues(bclist,'e','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // DOWN Boundary: Vz = Vin_rock
  PetscCall(DMStagBCListGetValues(bclist,'s','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = usr->nd->Vin_rock;
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // UP Boundary: Vz = -Vin_free
  PetscCall(DMStagBCListGetValues(bclist,'n','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = -usr->nd->Vin_free;
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // UP Boundary: pin P = 0
  PetscCall(DMStagBCListGetValues(bclist,'n','o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<1; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscFunctionBeginUser;

  // Left: dT/dx=0
  PetscCall(DMStagBCListGetValues(bclist,'w','o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  
  // RIGHT: dT/dx=0
  PetscCall(DMStagBCListGetValues(bclist,'e','o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));

  // DOWN: T = Tbot
  PetscCall(DMStagBCListGetValues(bclist,'s','o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {    
    if (usr->par->model_setup==0) age = usr->par->age*1.0e6*SEC_YEAR; // constant age
    else age  = usr->par->age*1.0e6*SEC_YEAR + dim_param(fabs(x_bc[2*k]),usr->scal->x)/dim_param(usr->nd->uT,usr->scal->v); // variable age

    Tbot = HalfSpaceCoolingTemp(usr->par->Tbot,usr->par->Ttop,usr->par->H-usr->par->Hs,usr->scal->kappa,age,usr->par->hs_factor); 
    nd_T = nd_paramT(Tbot,usr->par->Ttop,usr->scal->DT);

    value_bc[k] = nd_T;
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // UP: T = Ttop
  PetscCall(DMStagBCListGetValues(bclist,'n','o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = usr->nd->Ttop;
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
 
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscFunctionBeginUser;

  phi_max = usr->par->phi_max_bc; // 1e-3;
  sigma   = usr->par->sigma_bc;   // 0.1 - 0.001;
  
  // Left: dphis/dx = 0
  PetscCall(DMStagBCListGetValues(bclist,'w','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));

  // RIGHT: dphis/dx = 0
  PetscCall(DMStagBCListGetValues(bclist,'e','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));

  // DOWN: phis = 1.0 - func(phi0)
  PetscCall(DMStagBCListGetValues(bclist,'s','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    if (usr->par->model_setup_phi==1) phi = usr->par->phi0 + phi_max*PetscExpScalar(-x_bc[2*k]*x_bc[2*k]/sigma);
    else phi = usr->par->phi0;
    value_bc[k] = 1.0 - phi;
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));

  // UP: phi = 0 
  PetscCall(DMStagBCListGetValues(bclist,'n','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 1.0 - usr->par->phi0;
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
 
  PetscFunctionReturn(PETSC_SUCCESS);
}