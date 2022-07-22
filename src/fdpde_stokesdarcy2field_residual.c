#include "fdpde_stokes.h"
#include "fdpde_stokesdarcy2field.h"

// ---------------------------------------
/*@
FormFunction_StokesDarcy2Field - Residual evaluation function

Use: internal
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormFunction_StokesDarcy2Field"
PetscErrorCode FormFunction_StokesDarcy2Field(SNES snes, Vec x, Vec f, void *ctx)
{
  FDPDE          fd = (FDPDE)ctx;
  DM             dmPV, dmCoeff;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz, n[5];
  Vec            xlocal, flocal, coefflocal;
  PetscInt       iprev, inext, icenter;
  PetscScalar    ***ff,***_xlocal,***_coefflocal;;
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
  PetscInt pv_slot[5],coeff_e[3],coeff_v[4],coeff_B[4],coeff_D2[4],coeff_D3[4];
  ierr = GetLocationSlots(dmPV,dmCoeff,pv_slot,coeff_e,coeff_v,coeff_B); CHKERRQ(ierr);
  ierr = GetLocationSlots_Darcy2Field(dmCoeff,coeff_e,coeff_D2,coeff_D3); CHKERRQ(ierr);

  // Loop over elements
  for (j = sz; j<sz+nz; j++) {
    for (i = sx; i<sx+nx; i++) {
      PetscScalar fval, fval1;

      // 1) Stokes Continuity equation + div(v_D)
      ierr = ContinuityResidual(i,j,_xlocal,_coefflocal,coordx,coordz,n,pv_slot,coeff_e,&fval);CHKERRQ(ierr);
      ierr = ContinuityResidual_Darcy2Field(i,j,_xlocal,_coefflocal,coordx,coordz,n,pv_slot,coeff_D2,coeff_D3,fd->dm_btype0,fd->dm_btype1,&fval1);CHKERRQ(ierr);
      ff[j][i][pv_slot[4]] = fval + fval1;

      // 2) Stokes X-Momentum equation + grad (P_D)
      if (i > 0) {
        ierr = XMomentumResidual(i,j,_xlocal,_coefflocal,coordx,coordz,n,pv_slot,coeff_e,coeff_B,coeff_v,fd->dm_btype1,&fval);CHKERRQ(ierr);
        ierr = XMomentumResidual_Darcy2Field(i,j,_xlocal,_coefflocal,coordx,coordz,n,pv_slot,coeff_e,&fval1);CHKERRQ(ierr);
        ff[j][i][pv_slot[0]] = fval + fval1; // LEFT
      }

      // 3) Stokes Z-Momentum equation + grad (P_D)
      if (j > 0) {
        ierr = ZMomentumResidual(i,j,_xlocal,_coefflocal,coordx,coordz,n,pv_slot,coeff_e,coeff_B,coeff_v,fd->dm_btype0,&fval);CHKERRQ(ierr);
        ierr = ZMomentumResidual_Darcy2Field(i,j,_xlocal,_coefflocal,coordx,coordz,n,pv_slot,coeff_e,&fval1);CHKERRQ(ierr);
        ff[j][i][pv_slot[2]] = fval + fval1; // DOWN
      }
    }
  }

  // Boundary conditions - edges and element
  ierr = DMStagBCListApplyFace_StokesDarcy2Field(_xlocal,_coefflocal,bclist->bc_f,bclist->nbc_face,coordx,coordz,n,pv_slot,coeff_e,coeff_B,coeff_v,fd->dm_btype0,fd->dm_btype1,ff);CHKERRQ(ierr);
  ierr = DMStagBCListApplyElement_StokesDarcy2Field(_xlocal,_coefflocal,bclist->bc_e,bclist->nbc_element,coordx,coordz,n,pv_slot,coeff_e,coeff_D2,coeff_D3,fd->dm_btype0,fd->dm_btype1,ff);CHKERRQ(ierr);

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
GetLocationSlots_Darcy2Field - (STOKESDARCY2FIELD) get dmstag location slots
Use: internal
@*/
// ---------------------------------------
PetscErrorCode GetLocationSlots_Darcy2Field(DM dmCoeff, PetscInt *coeff_e, PetscInt *coeff_D2, PetscInt *coeff_D3)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = DMStagGetLocationSlot(dmCoeff,DMSTAG_ELEMENT,SD2_COEFF_ELEMENT_C, &coeff_e[SD2_COEFF_ELEMENT_C]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmCoeff,DMSTAG_ELEMENT,SD2_COEFF_ELEMENT_A, &coeff_e[SD2_COEFF_ELEMENT_A]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmCoeff,DMSTAG_ELEMENT,SD2_COEFF_ELEMENT_D1,&coeff_e[SD2_COEFF_ELEMENT_D1]);CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(dmCoeff,DMSTAG_LEFT,  SD2_COEFF_FACE_D2, &coeff_D2[0]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmCoeff,DMSTAG_RIGHT, SD2_COEFF_FACE_D2, &coeff_D2[1]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmCoeff,DMSTAG_DOWN,  SD2_COEFF_FACE_D2, &coeff_D2[2]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmCoeff,DMSTAG_UP,    SD2_COEFF_FACE_D2, &coeff_D2[3]);CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(dmCoeff,DMSTAG_LEFT,  SD2_COEFF_FACE_D3, &coeff_D3[0]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmCoeff,DMSTAG_RIGHT, SD2_COEFF_FACE_D3, &coeff_D3[1]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmCoeff,DMSTAG_DOWN,  SD2_COEFF_FACE_D3, &coeff_D3[2]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmCoeff,DMSTAG_UP,    SD2_COEFF_FACE_D3, &coeff_D3[3]);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
ContinuityResidual_Darcy2Field - (STOKESDARCY2FIELD) calculates the div(Darcy flux) per cell

Use: internal
@*/
// ---------------------------------------
PetscErrorCode ContinuityResidual_Darcy2Field(PetscInt i, PetscInt j,PetscScalar ***_xlocal,PetscScalar ***_coefflocal, PetscScalar **coordx, PetscScalar **coordz, PetscInt n[],PetscInt pv_slot[], PetscInt coeff_D2[],PetscInt coeff_D3[],DMBoundaryType dm_btype0,DMBoundaryType dm_btype1,PetscScalar *ff)
{
  PetscScalar    ffi, xx[5], D2[4], D3[4], dPdx[4],qD[4],dx, dx1, dx2, dz, dz1, dz2;
  PetscInt       ii, iprev, inext,icenter, Nx, Nz, im, jm, ip, jp, P_slot;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  Nx = n[0]; Nz = n[1]; icenter = n[2]; iprev = n[3]; inext = n[4];

  P_slot = 4;
  if (dm_btype0!=DM_BOUNDARY_PERIODIC) {
    if (i == 0   ) im = i; else im = i-1;
    if (i == Nx-1) ip = i; else ip = i+1;
  } else {
    im = i-1;
    ip = i+1;
  }
  if (dm_btype1!=DM_BOUNDARY_PERIODIC) {
    if (j == 0   ) jm = j; else jm = j-1;
    if (j == Nz-1) jp = j; else jp = j+1;
  } else {
    jm = j-1;
    jp = j+1;
  }

  // Get stencil values
  xx[0] = _xlocal[j ][i ][pv_slot[P_slot]];
  xx[1] = _xlocal[j ][ip][pv_slot[P_slot]];
  xx[2] = _xlocal[j ][im][pv_slot[P_slot]];
  xx[3] = _xlocal[jp][i ][pv_slot[P_slot]];
  xx[4] = _xlocal[jm][i ][pv_slot[P_slot]];
  
  // Coefficients - D2, D3
  for (ii = 0; ii < 4; ii++) {
    D2[ii] = _coefflocal[j][i][coeff_D2[ii]];
    D3[ii] = _coefflocal[j][i][coeff_D3[ii]];
  }

  // Grid spacings - Correct for boundaries
  dx  = coordx[i  ][inext  ]-coordx[i  ][iprev  ];
  dz  = coordz[j  ][inext  ]-coordz[j  ][iprev  ];

  if (i == 0)    { dx1 = 2.0*(coordx[i][icenter]-coordx[i][iprev]); } 
  else           { dx1 = coordx[i  ][icenter]-coordx[i-1][icenter]; }
  if (i == Nx-1) { dx2 = 2.0*(coordx[i][inext]-coordx[i][icenter]); } 
  else           { dx2 = coordx[i+1][icenter]-coordx[i  ][icenter]; }

  if (j == 0)    { dz1 = 2.0*(coordz[j][icenter]-coordz[j][iprev]); } 
  else           { dz1 = coordz[j  ][icenter]-coordz[j-1][icenter]; }
  if (j == Nz-1) { dz2 = 2.0*(coordz[j][inext]-coordz[j][icenter]); } 
  else           { dz2 = coordz[j+1][icenter]-coordz[j  ][icenter]; }

  // Calculate residual div (D2*grad(p)+D3)
  dPdx[0] = (xx[0]-xx[2])/dx1; // dP/dx_left
  dPdx[1] = (xx[1]-xx[0])/dx2; // dP/dx_right
  dPdx[2] = (xx[0]-xx[4])/dz1; // dP/dz_down
  dPdx[3] = (xx[3]-xx[0])/dz2; // dP/dz_up

  // Darcy flux = D2*grad(p)+D3
  qD[0] = D2[0]*dPdx[0]+D3[0];
  qD[1] = D2[1]*dPdx[1]+D3[1];
  qD[2] = D2[2]*dPdx[2]+D3[2];
  qD[3] = D2[3]*dPdx[3]+D3[3];

  // div(Darcy flux)
  ffi = (qD[1]-qD[0])/dx + (qD[3]-qD[2])/dz; 

  *ff = ffi;
  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
XMomentumResidual_Darcy2Field - (STOKESDARCY2FIELD) calculates the term grad(D1 div(u))

Use: internal
@*/
// ---------------------------------------
PetscErrorCode XMomentumResidual_Darcy2Field(PetscInt i, PetscInt j,PetscScalar ***_xlocal,PetscScalar ***_coefflocal, PetscScalar **coordx,PetscScalar **coordz,PetscInt n[],PetscInt pv_slot[], PetscInt coeff_e[],PetscScalar *ff)
{
  PetscInt       iprev, inext, icenter,iL,iR,iU,iD;
  PetscScalar    xx[7], dx, dx1, dx2, dz, divu1, divu2, ffi, D1[2];
  PetscErrorCode ierr;
  PetscFunctionBegin;

  icenter = n[2]; iprev = n[3]; inext = n[4];
  iL = 0; iR  = 1; iD = 2; iU  = 3;

  // Get stencil values
  xx[0] = _xlocal[j ][i  ][pv_slot[iL]]; // Vx(i  ,j  )
  xx[1] = _xlocal[j ][i  ][pv_slot[iR]]; // Vx(i+1,j  )
  xx[2] = _xlocal[j ][i  ][pv_slot[iD]]; // Vz(i  ,j  )
  xx[3] = _xlocal[j ][i  ][pv_slot[iU]]; // Vz(i  ,j+1)
  xx[4] = _xlocal[j ][i-1][pv_slot[iL]]; // Vx(i-1,j  )
  xx[5] = _xlocal[j ][i-1][pv_slot[iD]]; // Vz(i-1,j  )
  xx[6] = _xlocal[j ][i-1][pv_slot[iU]]; // Vz(i-1,j+1)

  // Coefficients
  D1[0] = _coefflocal[j ][i-1][coeff_e[SD2_COEFF_ELEMENT_D1]];
  D1[1] = _coefflocal[j ][i  ][coeff_e[SD2_COEFF_ELEMENT_D1]];

  // Grid spacings - need to correct for missing values
  dx  = coordx[i  ][icenter]-coordx[i-1][icenter];
  dx1 = coordx[i-1][inext  ]-coordx[i-1][iprev  ];
  dx2 = coordx[i  ][inext  ]-coordx[i  ][iprev  ];
  dz  = coordz[j  ][inext  ]-coordz[j  ][iprev  ];

  // Calculate new residual
  divu1 = (xx[0]-xx[4])/dx1 + (xx[6]-xx[5])/dz;
  divu2 = (xx[1]-xx[0])/dx2 + (xx[3]-xx[2])/dz;
  ffi   = (D1[1]*divu2 - D1[0]*divu1)/dx;

  *ff = ffi;
  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
ZMomentumResidual_Darcy2Field - (STOKESDARCY2FIELD) calculates the term grad(D1 div(u))

Use: internal
@*/
// ---------------------------------------
PetscErrorCode ZMomentumResidual_Darcy2Field(PetscInt i, PetscInt j,PetscScalar ***_xlocal,PetscScalar ***_coefflocal, PetscScalar **coordx,PetscScalar **coordz,PetscInt n[],PetscInt pv_slot[], PetscInt coeff_e[],PetscScalar *ff)
{
  PetscInt       iprev, inext, icenter,iL,iR,iU,iD;
  PetscScalar    xx[7], dx, dz1, dz2, dz, divu1, divu2, ffi, D1[2];
  PetscErrorCode ierr;
  PetscFunctionBegin;

  icenter = n[2]; iprev = n[3]; inext = n[4];
  iL = 0; iR  = 1; iD = 2; iU  = 3;

  // Get stencil values
  xx[0] = _xlocal[j  ][i][pv_slot[iL]]; // Vx(i  ,j  )
  xx[1] = _xlocal[j  ][i][pv_slot[iR]]; // Vx(i+1,j  )
  xx[2] = _xlocal[j  ][i][pv_slot[iD]]; // Vz(i  ,j  )
  xx[3] = _xlocal[j  ][i][pv_slot[iU]]; // Vz(i  ,j+1)
  xx[4] = _xlocal[j-1][i][pv_slot[iL]]; // Vx(i  ,j-1)
  xx[5] = _xlocal[j-1][i][pv_slot[iR]]; // Vx(i+1,j-1)
  xx[6] = _xlocal[j-1][i][pv_slot[iD]]; // Vz(i  ,j-1)

  // Coefficients
  D1[0] = _coefflocal[j-1][i][coeff_e[SD2_COEFF_ELEMENT_D1]];
  D1[1] = _coefflocal[j  ][i][coeff_e[SD2_COEFF_ELEMENT_D1]];

  // Grid spacings - need to correct for missing values
  dx  = coordx[i  ][inext  ]-coordx[i  ][iprev  ];
  dz  = coordz[j  ][icenter]-coordz[j-1][icenter];
  dz1 = coordz[j-1][inext  ]-coordz[j-1][iprev  ];
  dz2 = coordz[j  ][inext  ]-coordz[j  ][iprev  ];

  // Calculate new residual
  divu1 = (xx[5]-xx[4])/dx + (xx[2]-xx[6])/dz1;
  divu2 = (xx[1]-xx[0])/dx + (xx[3]-xx[2])/dz2;
  ffi   = (D1[1]*divu2 - D1[0]*divu1)/dz;

  *ff = ffi;

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
DMStagBCListApplyFace_StokesDarcy2Field - function to apply boundary conditions for StokesDarcy equations [flux terms to boundary conditions]

Use: internal
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "DMStagBCListApplyFace_StokesDarcy2Field"
PetscErrorCode DMStagBCListApplyFace_StokesDarcy2Field(PetscScalar ***_xlocal,PetscScalar ***_coefflocal, DMStagBC *bclist, PetscInt nbc, PetscScalar **coordx, PetscScalar **coordz,PetscInt n[], PetscInt pv_slot[], PetscInt coeff_e[], PetscInt coeff_B[], PetscInt coeff_v[], DMBoundaryType dm_btype0, DMBoundaryType dm_btype1, PetscScalar ***ff)
{
  PetscScalar    xx, xxT[2],dx, dz;
  PetscScalar    A_Left, A_Right, A_Up, A_Down;
  PetscInt       i, j, ibc, idx, iprev, inext, Nx, Nz,iL,iR,iD,iU;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  // dm domain info
  Nx = n[0]; Nz = n[1]; iprev = n[3]; inext = n[4];
  iL = 0; iR  = 1; iD = 2; iU  = 3;

  // Loop over all boundaries
  for (ibc = 0; ibc<nbc; ibc++) {

    if (bclist[ibc].type == BC_PERIODIC) { // normal stencil for i,j - should come before other BCs are set
      PetscScalar fval = 0.0, fval1 = 0.0;
      i   = bclist[ibc].point.i;
      j   = bclist[ibc].point.j;
      idx = bclist[ibc].idx;
      if ((bclist[ibc].point.loc == DMSTAG_LEFT) || (bclist[ibc].point.loc == DMSTAG_RIGHT)) {
        ierr = XMomentumResidual(i,j,_xlocal,_coefflocal,coordx,coordz,n,pv_slot,coeff_e,coeff_B,coeff_v,dm_btype1,&fval);CHKERRQ(ierr);
        ierr = XMomentumResidual_Darcy2Field(i,j,_xlocal,_coefflocal,coordx,coordz,n,pv_slot,coeff_e,&fval1);CHKERRQ(ierr);
      }
      if ((bclist[ibc].point.loc == DMSTAG_DOWN) || (bclist[ibc].point.loc == DMSTAG_UP)) {
        ierr = ZMomentumResidual(i,j,_xlocal,_coefflocal,coordx,coordz,n,pv_slot,coeff_e,coeff_B,coeff_v,dm_btype0,&fval);CHKERRQ(ierr);
        ierr = ZMomentumResidual_Darcy2Field(i,j,_xlocal,_coefflocal,coordx,coordz,n,pv_slot,coeff_e,&fval1);CHKERRQ(ierr);
      }
      ff[j][i][idx] = fval+fval1;
    }

    if (bclist[ibc].type == BC_DIRICHLET_STAG) {
      i   = bclist[ibc].point.i;
      j   = bclist[ibc].point.j;
      idx = bclist[ibc].idx;

      // Get residual value
      xx = _xlocal[j][i][idx];
      ff[j][i][idx] = xx - bclist[ibc].val;
    }
    
    if (bclist[ibc].type == BC_DIRICHLET) {
      i   = bclist[ibc].point.i;
      j   = bclist[ibc].point.j;
      idx = bclist[ibc].idx;

      // Get residual value
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
DMStagBCListApplyElement_StokesDarcy2Field - function to apply boundary conditions for StokesDarcy equations [flux terms to boundary conditions]

Use: internal
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "DMStagBCListApplyElement_StokesDarcy2Field"
PetscErrorCode DMStagBCListApplyElement_StokesDarcy2Field(PetscScalar ***_xlocal,PetscScalar ***_coefflocal, DMStagBC *bclist, PetscInt nbc, PetscScalar **coordx, PetscScalar **coordz,PetscInt n[], PetscInt pv_slot[], PetscInt coeff_e[], PetscInt coeff_D2[], PetscInt coeff_D3[], DMBoundaryType dm_btype0, DMBoundaryType dm_btype1, PetscScalar ***ff)
{
  PetscScalar    xx, dx, dz;
  PetscScalar    D2_Left, D2_Right, D2_Up, D2_Down;
  PetscInt       i, j, ibc, idx, iprev, inext, Nx, Nz,iL,iR,iD,iU;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  // dm domain info
  Nx = n[0]; Nz = n[1]; iprev = n[3]; inext = n[4];
  iL = 0; iR  = 1; iD = 2; iU  = 3;

  // Loop over all boundaries
  for (ibc = 0; ibc<nbc; ibc++) {

    if (bclist[ibc].type == BC_PERIODIC) { // normal stencil for i,j - should come before other BCs are set
      PetscScalar fval = 0.0, fval1 = 0.0;
      i   = bclist[ibc].point.i;
      j   = bclist[ibc].point.j;
      idx = bclist[ibc].idx;
      ierr = ContinuityResidual(i,j,_xlocal,_coefflocal,coordx,coordz,n,pv_slot,coeff_e,&fval);CHKERRQ(ierr);
      ierr = ContinuityResidual_Darcy2Field(i,j,_xlocal,_coefflocal,coordx,coordz,n,pv_slot,coeff_D2,coeff_D3,dm_btype0,dm_btype1,&fval1);CHKERRQ(ierr);
      ff[j][i][idx] = fval + fval1;
    }

    if (bclist[ibc].type == BC_DIRICHLET_STAG) {
      i   = bclist[ibc].point.i;
      j   = bclist[ibc].point.j;
      idx = bclist[ibc].idx;

      // Get residual value
      xx = _xlocal[j][i][idx];
      ff[j][i][idx] = xx - bclist[ibc].val;
    }

    if (bclist[ibc].type == BC_DIRICHLET) {
      i   = bclist[ibc].point.i;
      j   = bclist[ibc].point.j;
      idx = bclist[ibc].idx;

      // Get residual value
      xx = _xlocal[j][i][idx];

      // Add darcy flux terms
      if ((i == 0) && (bclist[ibc].point.loc == DMSTAG_ELEMENT)) { 
        D2_Left = _coefflocal[j][i][coeff_D2[iL]];
        dx = coordx[i][inext]-coordx[i][iprev];
        ff[j][i][idx] += -2.0 * D2_Left* (xx -bclist[ibc].val)/dx/dx;
      }

      if ((i == Nx-1) && (bclist[ibc].point.loc == DMSTAG_ELEMENT)) { 
        D2_Right = _coefflocal[j][i][coeff_D2[iR]];
        dx = coordx[i][inext]-coordx[i][iprev];
        ff[j][i][idx] += 2.0 * D2_Right* (bclist[ibc].val - xx)/dx/dx;
      }

      if ((j == 0) && (bclist[ibc].point.loc == DMSTAG_ELEMENT)) { 
        D2_Down = _coefflocal[j][i][coeff_D2[iD]];
        dz = coordz[j][inext]-coordz[j][iprev];
        ff[j][i][idx] += -2.0 * D2_Down* (xx - bclist[ibc].val)/dz/dz;
      }

      if ((j == Nz-1) && (bclist[ibc].point.loc == DMSTAG_ELEMENT)) { 
        D2_Up = _coefflocal[j][i][coeff_D2[iU]];
        dz = coordz[j][inext]-coordz[j][iprev];
        ff[j][i][idx] += 2.0 * D2_Up* (bclist[ibc].val - xx)/dz/dz;
      }
      
    }

    if (bclist[ibc].type == BC_NEUMANN) {
      i   = bclist[ibc].point.i;
      j   = bclist[ibc].point.j;
      idx = bclist[ibc].idx;

      // Add darcy flux terms
      if ((i == 0) && (bclist[ibc].point.loc == DMSTAG_ELEMENT)) { 
        D2_Left = _coefflocal[j][i][coeff_D2[iL]];
        dx = coordx[i][inext]-coordx[i][iprev];
        ff[j][i][idx] += -D2_Left*bclist[ibc].val/dx;
      }

      if ((i == Nx-1) && (bclist[ibc].point.loc == DMSTAG_ELEMENT)) { 
        D2_Right = _coefflocal[j][i][coeff_D2[iR]];
        dx = coordx[i][inext]-coordx[i][iprev];
        ff[j][i][idx] += D2_Right*bclist[ibc].val/dx;
      }

      if ((j == 0) && (bclist[ibc].point.loc == DMSTAG_ELEMENT)) { 
        D2_Down = _coefflocal[j][i][coeff_D2[iD]];
        dz = coordz[j][inext]-coordz[j][iprev];
        ff[j][i][idx] += -D2_Down*bclist[ibc].val/dz;
      }

      if ((j == Nz-1) && (bclist[ibc].point.loc == DMSTAG_ELEMENT)) { 
        D2_Up = _coefflocal[j][i][coeff_D2[iU]];
        dz = coordz[j][inext]-coordz[j][iprev];
        ff[j][i][idx] += D2_Up*bclist[ibc].val/dz;
      }
    }
  }
  PetscFunctionReturn(0);
}
