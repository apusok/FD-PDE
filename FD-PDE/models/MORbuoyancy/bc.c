#include "MORbuoyancy.h"

// ---------------------------------------
// FormBCList_PV
// Warning: when retrieving bc coordinates the following order is important only for Velocity - Dirichlet (&x_bc_boundary,&x_bc_stag)
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormBCList_PV"
PetscErrorCode FormBCList_PV(DM dm, Vec x, DMStagBCList bclist, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       k,n_bc,*idx_bc;
  PetscInt       i, sx, sz, nx, nz, Nx, Nz, iprev, icenter;
  PetscScalar    *value_bc,*x_bc,*x_bc_true, xx[2], dx;
  BCType         *type_bc;
  DMStagStencil  point[2];
  PetscScalar    **coordx,**coordz;
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

  // RIGHT dVx/dx = 0 (on true boundary)
  ierr = DMStagBCListGetValues(bclist,'e','-',0,&n_bc,&idx_bc,&x_bc_true,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN_T;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc_true,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN tau_xz = 0, dVx/dz=-dVz/dx (not on true boundary)
  ierr = DMStagBCListGetValues(bclist,'s','-',0,&n_bc,&idx_bc,&x_bc_true,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    for (i = sx; i < sx+nx; i++) {
      if (x_bc[2*k]==coordx[i][iprev]) {
        point[0].i = i-1; point[0].j = 0; point[0].loc = DOWN; point[0].c = 0;
        point[1].i = i  ; point[1].j = 0; point[1].loc = DOWN; point[1].c = 0;
        ierr = DMStagVecGetValuesStencil(dm,xlocal,2,point,xx); CHKERRQ(ierr);
        dx = coordx[i][icenter] - coordx[i-1][icenter];
        value_bc[k] = -(xx[1]-xx[0])/dx;
        type_bc[k] = BC_NEUMANN;
      }
    }
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc_true,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // LEFT Vx = 0 (on true boundary)
  ierr = DMStagBCListGetValues(bclist,'w','-',0,&n_bc,&idx_bc,&x_bc_true,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET; // BC_DIRICHLET_STAG; 
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc_true,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP Vx = u0 (not on true boundary)
  ierr = DMStagBCListGetValues(bclist,'n','-',0,&n_bc,&idx_bc,&x_bc,&x_bc_true,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    // PetscPrintf(PETSC_COMM_WORLD,"# UP VX %d [x = %f z = %f] [x_true = %f z_true = %f] \n",k,x_bc[2*k],x_bc[2*k+1],x_bc_true[2*k],x_bc_true[2*k+1]);
    value_bc[k] = usr->nd->U0;
    type_bc[k] = BC_DIRICHLET; // BC_DIRICHLET_STAG; 
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,&x_bc_true,&value_bc,&type_bc);CHKERRQ(ierr);

  // RIGHT dVz/dx = 0 (not on true boundary)
  ierr = DMStagBCListGetValues(bclist,'e','|',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN dVz/dz = 0 (on true boundary)
  ierr = DMStagBCListGetValues(bclist,'s','|',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN_T;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // LEFT dVz/dx = 0 (not on true boundary)
  ierr = DMStagBCListGetValues(bclist,'w','|',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP Vz = 0 (on true boundary)
  ierr = DMStagBCListGetValues(bclist,'n','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET; // BC_DIRICHLET_STAG;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // LEFT dP/dz = 0
  ierr = DMStagBCListGetValues(bclist,'w','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP dP/dz = 0
  ierr = DMStagBCListGetValues(bclist,'n','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

 // RIGHT P = 0
  ierr = DMStagBCListGetValues(bclist,'e','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET_STAG; //BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN P = 0
  ierr = DMStagBCListGetValues(bclist,'s','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET_STAG; //BC_DIRICHLET; 
  }

  // // Pinpoint pressure in lower right corner
  // if ((sz == 0) && (sx+nx == Nx)) {
  //   value_bc[n_bc-1] = 0.0;
  //   type_bc[n_bc-1] = BC_DIRICHLET_STAG; //BC_DIRICHLET; 
  // }

  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // restore
  ierr = DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&xlocal); CHKERRQ(ierr);

  // ierr = DMStagBCListView(bclist);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// FormBCList_HC
// Warning: use the real (staggered) coords for the center points in setting boundary conditions!
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormBCList_HC"
PetscErrorCode FormBCList_HC(DM dm, Vec x, DMStagBCList bclist, void *ctx)
{
  UsrData     *usr = (UsrData*)ctx;
  PetscInt    k,n_bc,*idx_bc;
  PetscScalar *value_bc,*x_bc;
  PetscScalar Hp, Hc, age, T, Tp, Ts, scalx, scalv, u0, kappa;
  BCType      *type_bc;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  scalx = usr->scal->x;
  scalv = usr->scal->v;
  u0 = usr->nd->U0;
  kappa = usr->par->kappa;
  Tp = usr->par->Tp;
  Ts = usr->par->Ts;

  // ENTHALPY
  // LEFT: dH/dx = 0
  ierr = DMStagBCListGetValues(bclist,'w','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // RIGHT: dH/dx = 0 
  ierr = DMStagBCListGetValues(bclist,'e','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN: H = Hp, enthalpy corresponding to the potential temperature at zero porosity (HP)
  ierr = DMStagBCListGetValues(bclist,'s','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    age = dim_param(x_bc[2*k],scalx)/dim_param(u0,scalv);
    T = HalfSpaceCoolingTemp(Tp,Ts,-dim_param(x_bc[2*k+1],scalx),kappa,age); 
    Hp = (T - usr->par->T0)/usr->par->DT;
    value_bc[k] = Hp; 
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP: H = Hc, MOR: dH/dz = 0, Hc is enthalpy corresponding to 0 deg C
  ierr = DMStagBCListGetValues(bclist,'n','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    if (x_bc[2*k]<=usr->nd->xMOR) {
      value_bc[k] = 0.0;
      type_bc[k] = BC_NEUMANN;
    } else {
      age = dim_param(x_bc[2*k],scalx)/dim_param(u0,scalv);
      T = HalfSpaceCoolingTemp(Tp,Ts,-dim_param(x_bc[2*k+1],scalx),kappa,age); 
      Hc = (T - usr->par->T0)/usr->par->DT;
      value_bc[k] = Hc;
      type_bc[k] = BC_DIRICHLET;
    }
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // COMPOSITION
  // LEFT: dC/dx = 0
  ierr = DMStagBCListGetValues(bclist,'w','o',1,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',1,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // RIGHT: dC/dx = 0
  ierr = DMStagBCListGetValues(bclist,'e','o',1,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',1,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN: C = C0 (dim) C = 0 (non-dim)
  ierr = DMStagBCListGetValues(bclist,'s','o',1,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;  // (usr->par->C0-usr->par->C0)/usr->par->DC
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',1,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP: dC/dz = 0
  ierr = DMStagBCListGetValues(bclist,'n','o',1,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',1,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
 
  PetscFunctionReturn(0);
}
