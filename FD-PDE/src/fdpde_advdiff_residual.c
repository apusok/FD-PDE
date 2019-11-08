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

  // Update timestep - after form coefficient
  ierr = UpdateTimeStep_AdvDiff(ad,dmcoeff,coefflocal);CHKERRQ(ierr);

  // Map the previous time step vectors
  if (ad->timesteptype != TS_NONE) {
    ierr = DMGetLocalVector(dm, &xprevlocal); CHKERRQ(ierr);
    ierr = DMGlobalToLocal (dm, ad->xprev, INSERT_VALUES, xprevlocal); CHKERRQ(ierr);

    if (!ad->coeffcalled) {
      ierr = fd->ops->form_coefficient(fd,dm,ad->xprev,dmcoeff,ad->coeffprev,fd->user_context);CHKERRQ(ierr);
      ad->coeffcalled = PETSC_TRUE;
    }
    
    ierr = DMGetLocalVector(dmcoeff, &coeffprevlocal); CHKERRQ(ierr);
    ierr = DMGlobalToLocal (dmcoeff, ad->coeffprev, INSERT_VALUES, coeffprevlocal); CHKERRQ(ierr);
  }

  // Get dm coordinates array
  ierr = DMStagGet1dCoordinateArraysDOFRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);

  // Create residual local vector
  ierr = DMCreateLocalVector(dm, &flocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayDOF(dm, flocal, &ff); CHKERRQ(ierr);

  // Loop over elements
  for (j = sz; j<sz+nz; j++) {
    for (i = sx; i<sx+nx; i++) {
      PetscScalar   xx, xxprev;
      PetscScalar   fval=0.0, fval0=0.0, fval1=0.0;
      PetscScalar   A, A0, A1;
      DMStagStencil point;

      if ((i > 0) && (i < Nx-1) && (j > 0) && (j < Nz-1)) {
        if (ad->timesteptype == TS_NONE) {
          // steady-state solution
          ierr = EnergyResidual(dm,xlocal,dmcoeff,coefflocal,coordx,coordz,i,j,ad->advtype,&fval,&A); CHKERRQ(ierr);
        } else { 
          // time-dependent solution - provided the valid flags pass the error check above
          ierr = EnergyResidual(dm,xprevlocal,dmcoeff,coeffprevlocal,coordx,coordz,i,j,ad->advtype,&fval0,&A0); CHKERRQ(ierr);
          ierr = EnergyResidual(dm,xlocal,dmcoeff,coefflocal,coordx,coordz,i,j,ad->advtype,&fval1,&A1); CHKERRQ(ierr);

          point.i = i; point.j = j; point.loc = DMSTAG_ELEMENT; point.c = 0;
          ierr = DMStagVecGetValuesStencil(dm,xlocal,1,&point,&xx); CHKERRQ(ierr);
          ierr = DMStagVecGetValuesStencil(dm,xprevlocal,1,&point,&xxprev); CHKERRQ(ierr);

          fval = xx - xxprev + ad->dt*(ad->theta*fval1/A1 + (1-ad->theta)*fval0/A0 );
        }

        ierr = DMStagGetLocationSlot(dm, DMSTAG_ELEMENT, 0, &idx); CHKERRQ(ierr);
        ff[j][i][idx] = fval;
      }
    }
  }

  // Boundary conditions - only element dofs
  ierr = DMStagBCListApply_AdvDiff(dm,xlocal,dmcoeff,coefflocal,bclist->bc_e,bclist->nbc_element,coordx,coordz,ff);CHKERRQ(ierr);

  // Restore arrays, local vectors
  ierr = DMStagRestore1dCoordinateArraysDOFRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArrayDOF(dm,flocal,&ff); CHKERRQ(ierr);
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
EnergyResidual - (ADVDIFF) calculates the steady state advdiff residual per dof

Use: internal
@*/
// ---------------------------------------
PetscErrorCode EnergyResidual(DM dm, Vec xlocal, DM dmcoeff,Vec coefflocal, PetscScalar **coordx, PetscScalar **coordz, PetscInt i, PetscInt j, AdvectSchemeType advtype,PetscScalar *ff,PetscScalar *_A)
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
  ierr = DMStagGet1dCoordinateLocationSlot(dm,DMSTAG_ELEMENT,&icenter);CHKERRQ(ierr); 

  // Coefficients
  point[0].i = i; point[0].j = j; point[0].loc = DMSTAG_ELEMENT; point[0].c = 0; // A = rho*cp
  point[1].i = i; point[1].j = j; point[1].loc = DMSTAG_ELEMENT; point[1].c = 1; // C = heat production/sink
  point[2].i = i; point[2].j = j; point[2].loc = DMSTAG_LEFT;    point[2].c = 0; // B_left = k
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
  dx[0] = coordx[i+1][icenter]-coordx[i  ][icenter];
  dx[1] = coordx[i  ][icenter]-coordx[i-1][icenter];
  dx[2]  = (dx[0]+dx[1])*0.5;

  dz[0] = coordz[j+1][icenter]-coordz[j  ][icenter];
  dz[1] = coordz[j  ][icenter]-coordz[j-1][icenter];
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

  if (i == 1) point[5] = point[1];
  if (j == 1) point[7] = point[3];
  if (i == Nx-1) point[6] = point[2];
  if (j == Nz-1) point[8] = point[4];

  ierr = DMStagVecGetValuesStencil(dm,xlocal,9,point,xx); CHKERRQ(ierr);

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
PetscErrorCode DMStagBCListApply_AdvDiff(DM dm, Vec xlocal,DM dmcoeff, Vec coefflocal, DMStagBC *bclist, PetscInt nbc, PetscScalar **coordx, PetscScalar **coordz, PetscScalar ***ff)
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
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"BC type NEUMANN for FDPDE_ADVDIFF is not yet implemented.");
    }
  }

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
UpdateTimeStep_AdvDiff - function to update time step size for ADVDIFF equations

Use: internal
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "UpdateTimeStep_AdvDiff"
PetscErrorCode UpdateTimeStep_AdvDiff(AdvDiffData *ad, DM dmcoeff, Vec coefflocal)
{
  PetscScalar    domain_dt, global_dt, eps, dx, dz, cell_dt, cell_dt_x, cell_dt_z;
  PetscInt       iprev=-1, inext=-1;
  PetscInt       i, j, sx, sz, nx, nz;
  PetscScalar    **coordx, **coordz;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  // check time-step scheme
  if (ad->timesteptype == TS_UNINIT) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Time stepping scheme for the FD-PDE ADVDIFF was not set! Set with FDPDEAdvDiffSetTimeStepSchemeType()");
  }

  if ((!ad->dt_user) && (!ad->dtflg)) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"No time step size or method was set! Set with FDPDEAdvDiffSetTimestep()");
  }

  // return if not required
  if (ad->timesteptype == TS_NONE) PetscFunctionReturn(0);

  if ((ad->dt_user) && (!ad->dtflg)) {
    ad->dt = ad->dt_user;
    PetscFunctionReturn(0);
  }

  domain_dt = 1.0e32;
  eps = 1.0e-32; /* small shift to avoid dividing by zero */

  ierr = DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = DMStagGet1dCoordinateArraysDOFRead(dmcoeff,&coordx,&coordz,NULL);CHKERRQ(ierr);

  ierr = DMStagGet1dCoordinateLocationSlot(dmcoeff,DMSTAG_LEFT,&iprev);CHKERRQ(ierr); 
  ierr = DMStagGet1dCoordinateLocationSlot(dmcoeff,DMSTAG_RIGHT,&inext);CHKERRQ(ierr); 

  // Loop over elements - velocity is located on edge and c=1
  for (j = sz; j<sz+nz; j++) {
    for (i = sx; i<sx+nx; i++) {
      DMStagStencil point[4];
      PetscScalar   xx[4];

      point[0].i = i; point[0].j = j; point[0].loc = DMSTAG_LEFT;  point[0].c = 1;
      point[1].i = i; point[1].j = j; point[1].loc = DMSTAG_RIGHT; point[1].c = 1;
      point[2].i = i; point[2].j = j; point[2].loc = DMSTAG_DOWN;  point[2].c = 1;
      point[3].i = i; point[3].j = j; point[3].loc = DMSTAG_UP;    point[3].c = 1;

      ierr = DMStagVecGetValuesStencil(dmcoeff,coefflocal,4,point,xx); CHKERRQ(ierr);

      dx = coordx[0][inext]-coordx[0][iprev];
      dz = coordz[0][inext]-coordz[0][iprev];

      /* compute dx, dy for this cell */
      cell_dt_x = dx / PetscMax(PetscMax(PetscAbsScalar(xx[0]), PetscAbsScalar(xx[1])), eps);
      cell_dt_z = dz / PetscMax(PetscMax(PetscAbsScalar(xx[2]), PetscAbsScalar(xx[3])), eps);
      cell_dt   = PetscMin(cell_dt_x,cell_dt_z);
      domain_dt = PetscMin(domain_dt,cell_dt);
    }
  }

  // MPI exchange global min/max
  ierr = MPI_Allreduce(&domain_dt,&global_dt,1,MPI_DOUBLE,MPI_MIN,PetscObjectComm((PetscObject)dmcoeff));CHKERRQ(ierr);

  ierr = DMStagRestore1dCoordinateArraysDOFRead(dmcoeff,&coordx,&coordz,NULL);CHKERRQ(ierr);

  if (ad->dt_user) {
    ad->dt = PetscMin(ad->dt_user,global_dt);
  } else {
    ad->dt = global_dt;
  }

  PetscFunctionReturn(0);
}
