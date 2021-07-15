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
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz, n[5];
  Vec            xlocal, flocal, coefflocal;
  PetscInt       iprev, inext, icenter;
  PetscScalar    ***ff,***_xlocal,***_coefflocal;
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
  ierr = fd->ops->form_coefficient(fd,dmPV,x,dmCoeff,fd->coeff,fd->user_context);CHKERRQ(ierr);

  // Get local domain
  ierr = DMStagGetCorners(dmPV, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmPV,DMSTAG_ELEMENT,&icenter);CHKERRQ(ierr); 
  ierr = DMStagGetProductCoordinateLocationSlot(dmPV,DMSTAG_LEFT,&iprev);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmPV,DMSTAG_RIGHT,&inext);CHKERRQ(ierr); 

  // Save useful variables for residual calculations
  n[0] = Nx; n[1] = Nz; n[2] = icenter; n[3] = iprev; n[4] = inext;

  // Map global vectors to local domain
  ierr = DMGetLocalVector(dmPV, &xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dmPV, x, INSERT_VALUES, xlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(dmPV,xlocal,&_xlocal);CHKERRQ(ierr);

  ierr = DMGetLocalVector(dmCoeff, &coefflocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dmCoeff, fd->coeff, INSERT_VALUES, coefflocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(dmCoeff,coefflocal,&_coefflocal);CHKERRQ(ierr);

  // Get dm coordinates array
  ierr = DMStagGetProductCoordinateArraysRead(dmPV,&coordx,&coordz,NULL);CHKERRQ(ierr);

  // Create residual local vector
  ierr = DMCreateLocalVector(dmPV, &flocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dmPV, flocal, &ff); CHKERRQ(ierr);

  // Get location slots
  PetscInt pv_slot[5],coeff_e[2],coeff_f[4],coeff_v[4];
  ierr = GetLocationSlots(dmPV,dmCoeff,pv_slot,coeff_e,coeff_v,coeff_f); CHKERRQ(ierr);

  // Loop over elements
  for (j = sz; j<sz+nz; j++) {
    for (i = sx; i<sx+nx; i++) {
      PetscScalar fval;

      // 1) Continuity equation
      ierr = ContinuityResidual(i,j,_xlocal,_coefflocal,coordx,coordz,n,pv_slot,coeff_e,&fval);CHKERRQ(ierr);
      ff[j][i][pv_slot[4]] = fval;

      // 2) X-Momentum equation
      if (i > 0) {
        ierr = XMomentumResidual(i,j,_xlocal,_coefflocal,coordx,coordz,n,pv_slot,coeff_e,coeff_f,coeff_v,&fval);CHKERRQ(ierr);
        ff[j][i][pv_slot[0]] = fval; // LEFT
      }

      // 3) Z-Momentum equation
      if (j > 0) {
        ierr = ZMomentumResidual(i,j,_xlocal,_coefflocal,coordx,coordz,n,pv_slot,coeff_e,coeff_f,coeff_v,&fval);CHKERRQ(ierr);
        ff[j][i][pv_slot[2]] = fval; // DOWN
      }
    }
  }

  // Boundary conditions - edges and element
  ierr = DMStagBCListApplyFace_Stokes(_xlocal,_coefflocal,bclist->bc_f,bclist->nbc_face,coordx,coordz,n,pv_slot,coeff_e,coeff_f,coeff_v,ff);CHKERRQ(ierr);
  ierr = DMStagBCListApplyElement_Stokes(_xlocal,_coefflocal,bclist->bc_e,bclist->nbc_element,coordx,coordz,n,pv_slot,coeff_e,coeff_f,coeff_v,ff);CHKERRQ(ierr);

  // Restore arrays, local vectors
  ierr = DMStagRestoreProductCoordinateArraysRead(dmPV,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArray(dmPV,flocal,&ff); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dmPV,flocal,INSERT_VALUES,f); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dmPV,flocal,INSERT_VALUES,f); CHKERRQ(ierr);
  ierr = VecDestroy(&flocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArrayRead(dmPV,xlocal,&_xlocal);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmPV,&xlocal); CHKERRQ(ierr);
  ierr = DMStagVecRestoreArrayRead(dmCoeff,coefflocal,&_coefflocal);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmCoeff,&coefflocal); CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode FormFunctionSplit_Stokes(SNES snes, Vec x, Vec x2, Vec f, void *ctx)
{
  FDPDE          fd = (FDPDE)ctx;
  DM             dmPV, dmCoeff;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz, n[5];
  Vec            xlocal, flocal, coefflocal;
  PetscInt       iprev, inext, icenter;
  PetscScalar    ***ff, ***_xlocal, ***_coefflocal;
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
  ierr = fd->ops->form_coefficient_split(fd,dmPV,x,x2,dmCoeff,fd->coeff,fd->user_context);CHKERRQ(ierr);
  
  // Get local domain
  ierr = DMStagGetCorners(dmPV, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmPV,DMSTAG_ELEMENT,&icenter);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmPV,DMSTAG_LEFT,&iprev);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmPV,DMSTAG_RIGHT,&inext);CHKERRQ(ierr);
  
  // Save useful variables for residual calculations
  n[0] = Nx; n[1] = Nz; n[2] = icenter; n[3] = iprev; n[4] = inext;
  
  // Map global vectors to local domain
  ierr = DMGetLocalVector(dmPV, &xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dmPV, x, INSERT_VALUES, xlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(dmPV,xlocal,&_xlocal);CHKERRQ(ierr);
  
  ierr = DMGetLocalVector(dmCoeff, &coefflocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dmCoeff, fd->coeff, INSERT_VALUES, coefflocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(dmCoeff,coefflocal,&_coefflocal);CHKERRQ(ierr);
  
  // Get dm coordinates array
  ierr = DMStagGetProductCoordinateArraysRead(dmPV,&coordx,&coordz,NULL);CHKERRQ(ierr);
  
  // Create residual local vector
  ierr = DMCreateLocalVector(dmPV, &flocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dmPV, flocal, &ff); CHKERRQ(ierr);

  // Get location slots
  PetscInt pv_slot[5],coeff_e[2],coeff_f[4],coeff_v[4];
  ierr = GetLocationSlots(dmPV,dmCoeff,pv_slot,coeff_e,coeff_v,coeff_f); CHKERRQ(ierr);

  // Loop over elements
  for (j = sz; j<sz+nz; j++) {
    for (i = sx; i<sx+nx; i++) {
      PetscScalar fval;
      
      // 1) Continuity equation
      ierr = ContinuityResidual(i,j,_xlocal,_coefflocal,coordx,coordz,n,pv_slot,coeff_e,&fval);CHKERRQ(ierr);
      ff[j][i][pv_slot[4]] = fval;
      
      // 2) X-Momentum equation
      if (i > 0) {
        ierr = XMomentumResidual(i,j,_xlocal,_coefflocal,coordx,coordz,n,pv_slot,coeff_e,coeff_f,coeff_v,&fval);CHKERRQ(ierr);
        ff[j][i][pv_slot[0]] = fval; // LEFT
      }
      
      // 3) Z-Momentum equation
      if (j > 0) {
        ierr = ZMomentumResidual(i,j,_xlocal,_coefflocal,coordx,coordz,n,pv_slot,coeff_e,coeff_f,coeff_v,&fval);CHKERRQ(ierr);
        ff[j][i][pv_slot[2]] = fval; // DOWN
      }
    }
  }
  
  // Boundary conditions - edges and element
  ierr = DMStagBCListApplyFace_Stokes(_xlocal,_coefflocal,bclist->bc_f,bclist->nbc_face,coordx,coordz,n,pv_slot,coeff_e,coeff_f,coeff_v,ff);CHKERRQ(ierr);
  ierr = DMStagBCListApplyElement_Stokes(_xlocal,_coefflocal,bclist->bc_e,bclist->nbc_element,coordx,coordz,n,pv_slot,coeff_e,coeff_f,coeff_v,ff);CHKERRQ(ierr);
  
  // Restore arrays, local vectors
  ierr = DMStagRestoreProductCoordinateArraysRead(dmPV,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArray(dmPV,flocal,&ff); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dmPV,flocal,INSERT_VALUES,f); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dmPV,flocal,INSERT_VALUES,f); CHKERRQ(ierr);
  ierr = VecDestroy(&flocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArrayRead(dmPV,xlocal,&_xlocal);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmPV,&xlocal); CHKERRQ(ierr);
  ierr = DMStagVecRestoreArrayRead(dmCoeff,coefflocal,&_coefflocal);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmCoeff,&coefflocal); CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
GetLocationSlots - (STOKES) calculates the continuity residual per dof
Use: internal
@*/
// ---------------------------------------
PetscErrorCode GetLocationSlots(DM dmPV, DM dmCoeff, PetscInt *pv_slot, PetscInt *coeff_e, PetscInt *coeff_v, PetscInt *coeff_f)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = DMStagGetLocationSlot(dmPV,DMSTAG_LEFT,   S_DOF_V, &pv_slot[0]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmPV,DMSTAG_RIGHT,  S_DOF_V, &pv_slot[1]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmPV,DMSTAG_DOWN,   S_DOF_V, &pv_slot[2]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmPV,DMSTAG_UP,     S_DOF_V, &pv_slot[3]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmPV,DMSTAG_ELEMENT,S_DOF_P, &pv_slot[4]); CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(dmCoeff,DMSTAG_ELEMENT,S_COEFF_ELEMENT_C,&coeff_e[S_COEFF_ELEMENT_C]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmCoeff,DMSTAG_ELEMENT,S_COEFF_ELEMENT_A,&coeff_e[S_COEFF_ELEMENT_A]); CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(dmCoeff,DMSTAG_DOWN_LEFT, S_COEFF_VERTEX_A,&coeff_v[0]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmCoeff,DMSTAG_DOWN_RIGHT,S_COEFF_VERTEX_A,&coeff_v[1]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmCoeff,DMSTAG_UP_LEFT,   S_COEFF_VERTEX_A,&coeff_v[2]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmCoeff,DMSTAG_UP_RIGHT,  S_COEFF_VERTEX_A,&coeff_v[3]);CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(dmCoeff,DMSTAG_LEFT,  S_COEFF_FACE_B, &coeff_f[0]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmCoeff,DMSTAG_RIGHT, S_COEFF_FACE_B, &coeff_f[1]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmCoeff,DMSTAG_DOWN,  S_COEFF_FACE_B, &coeff_f[2]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmCoeff,DMSTAG_UP,    S_COEFF_FACE_B, &coeff_f[3]);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
ContinuityResidual - (STOKES) calculates the continuity residual per dof

Use: internal
@*/
// ---------------------------------------
PetscErrorCode ContinuityResidual(PetscInt i, PetscInt j,PetscScalar ***_xlocal,PetscScalar ***_coefflocal, PetscScalar **coordx, PetscScalar **coordz, PetscInt n[], PetscInt pv_slot[], PetscInt coeff_e[], PetscScalar *ff)
{
  PetscScalar    ffi, xx[4], C, dx, dz;
  PetscInt       iprev, inext;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  iprev = n[3]; 
  inext = n[4];

  // Get stencil values
  xx[0] = _xlocal[j][i][pv_slot[0]]; // v_left
  xx[1] = _xlocal[j][i][pv_slot[1]]; // v_right
  xx[2] = _xlocal[j][i][pv_slot[2]]; // v_down
  xx[3] = _xlocal[j][i][pv_slot[3]]; // v_up

  // Coefficients
  C = _coefflocal[j][i][coeff_e[S_COEFF_ELEMENT_C]];

  // Calculate residual
  dx = coordx[i][inext]-coordx[i][iprev];
  dz = coordz[j][inext]-coordz[j][iprev];
  ffi = (xx[1]-xx[0])/dx + (xx[3]-xx[2])/dz - C;

  *ff = ffi;
  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
XMomentumResidual - (STOKES) calculates the Vx momentum residual per dof

Use: internal
@*/
// ---------------------------------------
PetscErrorCode XMomentumResidual(PetscInt i, PetscInt j,PetscScalar ***_xlocal,PetscScalar ***_coefflocal, PetscScalar **coordx,PetscScalar **coordz,PetscInt n[], PetscInt pv_slot[], PetscInt coeff_e[],PetscInt coeff_f[],PetscInt coeff_v[],PetscScalar *ff)
{
  PetscScalar    dVx2dz, dVz2dx, dPdx, dVx2dx, ffi;
  PetscInt       Nz, iprev, inext, icenter,iL,iR,iU,iD,iP, jm, jp;
  PetscScalar    xx[11], dx, dx1, dx2, dz, dz1, dz2;
  PetscScalar    A_Left, A_Right, A_Up, A_Down, Bx;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  Nz = n[1]; icenter = n[2]; iprev = n[3]; inext = n[4];

  iL = 0; iR  = 1; iD = 2; iU  = 3; iP = 4;
  if (j == 0   ) jm = j; else jm = j-1;
  if (j == Nz-1) jp = j; else jp = j+1;

  // Get stencil values
  xx[0] = _xlocal[j ][i  ][pv_slot[iL]]; // Vx(i  ,j  )
  xx[1] = _xlocal[jm][i  ][pv_slot[iL]]; // Vx(i  ,j-1)
  xx[2] = _xlocal[jp][i  ][pv_slot[iL]]; // Vx(i  ,j+1)
  xx[3] = _xlocal[j ][i-1][pv_slot[iL]]; // Vx(i-1,j  )
  xx[4] = _xlocal[j ][i  ][pv_slot[iR]]; // Vx(i+1,j  )

  xx[5] = _xlocal[j ][i-1][pv_slot[iD]]; // Vz(i-1,j-1)
  xx[6] = _xlocal[j ][i  ][pv_slot[iD]]; // Vz(i  ,j-1)
  xx[7] = _xlocal[j ][i-1][pv_slot[iU]]; // Vz(i-1,j  )
  xx[8] = _xlocal[j ][i  ][pv_slot[iU]]; // Vz(i  ,j  )
  xx[9] = _xlocal[j ][i-1][pv_slot[iP]]; // P (i-1,j  )
  xx[10]= _xlocal[j ][i  ][pv_slot[iP]]; // P (i  ,j  )

  // Coefficients
  Bx      = _coefflocal[j ][i  ][coeff_f[iL]];
  A_Left  = _coefflocal[j ][i-1][coeff_e[S_COEFF_ELEMENT_A]];
  A_Right = _coefflocal[j ][i  ][coeff_e[S_COEFF_ELEMENT_A]];
  A_Up    = _coefflocal[j ][i  ][coeff_v[2]];
  A_Down  = _coefflocal[j ][i  ][coeff_v[0]];

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
  dVx2dx = A_Right*(xx[4]-xx[0])/dx2 - A_Left*(xx[0]-xx[3])/dx1;
  dVx2dz = A_Up   *(xx[2]-xx[0])/dz2 - A_Down*(xx[0]-xx[1])/dz1;
  dVz2dx = A_Up   *(xx[8]-xx[7])/dx  - A_Down*(xx[6]-xx[5])/dx;

  ffi   = -dPdx + 2.0*dVx2dx/dx + dVx2dz/dz + dVz2dx/dz - Bx;

  *ff = ffi;
  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
ZMomentumResidual - (STOKES) calculates the Vz momentum residual per dof

Use: internal
@*/
// ---------------------------------------
PetscErrorCode ZMomentumResidual(PetscInt i, PetscInt j,PetscScalar ***_xlocal,PetscScalar ***_coefflocal, PetscScalar **coordx,PetscScalar **coordz,PetscInt n[], PetscInt pv_slot[], PetscInt coeff_e[],PetscInt coeff_f[],PetscInt coeff_v[],PetscScalar *ff)
{
  PetscScalar    dVx2dz, dVz2dx, dPdz, dVz2dz, ffi;
  PetscInt       Nx, iprev, inext, icenter,iL,iR,iD,iU,iP,is,ie;
  PetscScalar    xx[11], dx, dz, dx1, dx2, dz1, dz2;
  PetscScalar    A_Left, A_Right, A_Up, A_Down, Bz;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  Nx = n[0]; icenter = n[2]; iprev = n[3]; inext = n[4];

  iL = 0; iR  = 1; iD = 2; iU  = 3; iP = 4;
  if (i == 0   ) is = i; else is = i-1;
  if (i == Nx-1) ie = i; else ie = i+1;

  // Get stencil values
  xx[0] = _xlocal[j  ][i ][pv_slot[iD]]; // Vz(i  ,j  )
  xx[1] = _xlocal[j  ][i ][pv_slot[iU]]; // Vz(i  ,j+1)
  xx[2] = _xlocal[j-1][i ][pv_slot[iD]]; // Vz(i  ,j-1)
  xx[3] = _xlocal[j  ][is][pv_slot[iD]]; // Vz(i-1,j  )
  xx[4] = _xlocal[j  ][ie][pv_slot[iD]]; // Vz(i+1,j  )

  xx[5] = _xlocal[j  ][i ][pv_slot[iL]]; // Vx(i-1,j  )
  xx[6] = _xlocal[j  ][i ][pv_slot[iR]]; // Vx(i  ,j  )
  xx[7] = _xlocal[j-1][i ][pv_slot[iL]]; // Vx(i-1,j-1)
  xx[8] = _xlocal[j-1][i ][pv_slot[iR]]; // Vx(i-1,j-1)
  xx[9] = _xlocal[j-1][i ][pv_slot[iP]]; // P (i  ,j-1)
  xx[10]= _xlocal[j  ][i ][pv_slot[iP]]; // P (i  ,j  )

  // Coefficients
  Bz      = _coefflocal[j  ][i][coeff_f[iD]];
  A_Left  = _coefflocal[j  ][i][coeff_v[0]];
  A_Right = _coefflocal[j  ][i][coeff_v[1]];
  A_Up    = _coefflocal[j  ][i][coeff_e[S_COEFF_ELEMENT_A]];
  A_Down  = _coefflocal[j-1][i][coeff_e[S_COEFF_ELEMENT_A]];

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
  dVz2dz = A_Up   *(xx[1]-xx[0])/dz2 - A_Down *(xx[0]-xx[2])/dz1;
  dVz2dx = A_Right*(xx[4]-xx[0])/dx2 - A_Left *(xx[0]-xx[3])/dx1;
  dVx2dz = A_Right*(xx[6]-xx[8])/dz  - A_Left *(xx[5]-xx[7])/dz;

  ffi   = -dPdz + 2.0*dVz2dz/dz + dVz2dx/dx + dVx2dz/dx - Bz;

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
PetscErrorCode DMStagBCListApplyFace_Stokes(PetscScalar ***_xlocal,PetscScalar ***_coefflocal, DMStagBC *bclist, PetscInt nbc, PetscScalar **coordx, PetscScalar **coordz,PetscInt n[], PetscInt pv_slot[], PetscInt coeff_e[],PetscInt coeff_f[],PetscInt coeff_v[], PetscScalar ***ff)
{
  PetscScalar    xx, xxT[2], dx, dz;
  PetscScalar    A_Left, A_Right, A_Up, A_Down;
  PetscInt       i, j, ibc, idx, iprev, inext, Nx, Nz,iL,iR,iD,iU;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  // dm domain info
  Nx = n[0]; Nz = n[1]; iprev = n[3]; inext = n[4];
  iL = 0; iR  = 1; iD = 2; iU  = 3;

  // Loop over all boundaries
  for (ibc = 0; ibc<nbc; ibc++) {

    if (bclist[ibc].type == BC_DIRICHLET_STAG) {
      i   = bclist[ibc].point.i;
      j   = bclist[ibc].point.j;
      idx = bclist[ibc].idx;

      // access the value on this point
      xx = _xlocal[j][i][idx];
      ff[j][i][idx] = xx - bclist[ibc].val;
    }
    
    if (bclist[ibc].type == BC_DIRICHLET) {
      i   = bclist[ibc].point.i;
      j   = bclist[ibc].point.j;
      idx = bclist[ibc].idx;

      // access the value on this point
      xx = _xlocal[j][i][idx];

      // Stokes flow - add flux terms
      if ((j == 0) && (i > 0) && (bclist[ibc].point.loc == DMSTAG_LEFT)) { // Vx down - only interior points
        A_Down = _coefflocal[j][i][coeff_v[0]];
        dz = coordz[j][inext]-coordz[j][iprev];
        ff[j][i][idx] += -2.0 * A_Down*( xx - bclist[ibc].val)/dz/dz;
      }

      else if ((j == 0) && (i < Nx-1) && (bclist[ibc].point.loc == DMSTAG_RIGHT)) { // Vx down-special case
        A_Down = _coefflocal[j][i][coeff_v[1]];
        dz = coordz[j][inext]-coordz[j][iprev];
        ff[j][i][idx] += -2.0 * A_Down*( xx - bclist[ibc].val)/dz/dz;
      }

      else if ((j == Nz-1) && (i > 0) && (bclist[ibc].point.loc == DMSTAG_LEFT)) { // Vx up - only interior points
        A_Up = _coefflocal[j][i][coeff_v[2]];
        dz = coordz[j][inext]-coordz[j][iprev];
        ff[j][i][idx] += 2.0 * A_Up*( bclist[ibc].val - xx)/dz/dz;
      }

      else if ((j == Nz-1) && (i < Nx-1) && (bclist[ibc].point.loc == DMSTAG_RIGHT)) { // Vx up - special case
        A_Up = _coefflocal[j][i][coeff_v[3]];
        dz = coordz[j][inext]-coordz[j][iprev];
        ff[j][i][idx] += 2.0 * A_Up*( bclist[ibc].val - xx)/dz/dz;
      }

      else if ((i == 0) && (j > 0) && (bclist[ibc].point.loc == DMSTAG_DOWN)) { // Vz left
        A_Left = _coefflocal[j][i][coeff_v[0]];
        dx = coordx[i][inext]-coordx[i][iprev];
        ff[j][i][idx] += -2.0 * A_Left*( xx - bclist[ibc].val)/dx/dx;
      }

      else if ((i == Nx-1) && (j > 0) && (bclist[ibc].point.loc == DMSTAG_DOWN)) { // Vz right
        A_Right = _coefflocal[j][i][coeff_v[1]];
        dx = coordx[i][inext]-coordx[i][iprev];
        ff[j][i][idx] += 2.0 * A_Right*( bclist[ibc].val - xx)/dx/dx;
      }

      else {
        ff[j][i][idx] = xx - bclist[ibc].val;
      }
      
    }

    if (bclist[ibc].type == BC_NEUMANN) {
      i   = bclist[ibc].point.i;
      j   = bclist[ibc].point.j;
      idx = bclist[ibc].idx;

      // Stokes flow - add flux terms
      if ((j == 0) && (bclist[ibc].point.loc == DMSTAG_LEFT)) { // Vx down
        A_Down = _coefflocal[j][i][coeff_v[0]];
        dz = coordz[j][inext]-coordz[j][iprev];
        ff[j][i][idx] += -A_Down*bclist[ibc].val/dz;
      }

      if ((j == 0) && (bclist[ibc].point.loc == DMSTAG_RIGHT)) { // Vx down-special case
        A_Down = _coefflocal[j][i][coeff_v[1]];
        dz = coordz[j][inext]-coordz[j][iprev];
        ff[j][i][idx] += -A_Down*bclist[ibc].val/dz;
      }

      if ((j == Nz-1) && (bclist[ibc].point.loc == DMSTAG_LEFT)) { // Vx up
        A_Up = _coefflocal[j][i][coeff_v[2]];
        dz = coordz[j][inext]-coordz[j][iprev];
        ff[j][i][idx] += A_Up*bclist[ibc].val/dz;
      }

      if ((j == Nz-1) && (bclist[ibc].point.loc == DMSTAG_RIGHT)) { // Vx up - special case
        A_Up = _coefflocal[j][i][coeff_v[3]];
        dz = coordz[j][inext]-coordz[j][iprev];
        ff[j][i][idx] += A_Up*bclist[ibc].val/dz;
      }

      if ((i == 0) && (bclist[ibc].point.loc == DMSTAG_DOWN)) { // Vz left
        A_Left = _coefflocal[j][i][coeff_v[0]];
        dx = coordx[i][inext]-coordx[i][iprev];
        ff[j][i][idx] += -A_Left*bclist[ibc].val/dx;
      }

      if ((i == Nx-1) && (bclist[ibc].point.loc == DMSTAG_DOWN)) { // Vz right
        A_Right = _coefflocal[j][i][coeff_v[1]];
        dx = coordx[i][inext]-coordx[i][iprev];
        ff[j][i][idx] += A_Right*bclist[ibc].val/dx;
      }
    }

    if (bclist[ibc].type == BC_NEUMANN_T) { // of the form dvi/dxi
      i   = bclist[ibc].point.i;
      j   = bclist[ibc].point.j;
      idx = bclist[ibc].idx;

      if ((i == 0) && (bclist[ibc].point.loc == DMSTAG_LEFT)) { // left dVx/dx = a
        xxT[0] = _xlocal[j][i  ][pv_slot[iL]];
        xxT[1] = _xlocal[j][i+1][pv_slot[iL]];
        dx = coordx[i][inext]-coordx[i][iprev];
        ff[j][i][idx] = xxT[1]-xxT[0]-bclist[ibc].val*dx;
      }
      if ((i == Nx-1) && (bclist[ibc].point.loc == DMSTAG_RIGHT)) { // right dVx/dx = a
        xxT[0] = _xlocal[j][i  ][pv_slot[iL]];
        xxT[1] = _xlocal[j][i  ][pv_slot[iR]];
        dx = coordx[i][inext]-coordx[i][iprev];
        ff[j][i][idx] = xxT[1]-xxT[0]-bclist[ibc].val*dx;
      }
      if ((j == 0) && (bclist[ibc].point.loc == DMSTAG_DOWN)) { // down dVz/dz = a
        xxT[0] = _xlocal[j  ][i][pv_slot[iD]];
        xxT[1] = _xlocal[j+1][i][pv_slot[iD]];
        dz = coordz[j][inext]-coordz[j][iprev];
        ff[j][i][idx] = xxT[1]-xxT[0]-bclist[ibc].val*dz;
      }
      if ((j == Nz-1) && (bclist[ibc].point.loc == DMSTAG_UP)) { // up dVz/dz = a
        xxT[0] = _xlocal[j-1][i][pv_slot[iU]];
        xxT[1] = _xlocal[j  ][i][pv_slot[iU]];
        dz = coordz[j][inext]-coordz[j][iprev];
        ff[j][i][idx] = xxT[1]-xxT[0]-bclist[ibc].val*dz;
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
PetscErrorCode DMStagBCListApplyElement_Stokes(PetscScalar ***_xlocal,PetscScalar ***_coefflocal, DMStagBC *bclist, PetscInt nbc, PetscScalar **coordx, PetscScalar **coordz,PetscInt n[], PetscInt pv_slot[], PetscInt coeff_e[],PetscInt coeff_f[],PetscInt coeff_v[], PetscScalar ***ff)
{
  PetscScalar    xx;
  PetscInt       i, j, ibc, idx;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  // Loop over all boundaries
  for (ibc = 0; ibc<nbc; ibc++) {
    if (bclist[ibc].type == BC_DIRICHLET_STAG) {
      i   = bclist[ibc].point.i;
      j   = bclist[ibc].point.j;
      idx = bclist[ibc].idx;

      // Get residual value
      xx = _xlocal[j][i][idx];
      ff[j][i][idx] = xx - bclist[ibc].val;
    }

    if (bclist[ibc].type == BC_DIRICHLET) {
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"BC_DIRICHLET type on the true boundary for FDPDE_STOKES [ELEMENT] is not yet implemented. Use BC_DIRICHLET_STAG type instead!");
    }
    
    if (bclist[ibc].type == BC_NEUMANN) {
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"BC type NEUMANN for FDPDE_STOKES [ELEMENT] is not yet implemented.");
    }
  }

  PetscFunctionReturn(0);
}
