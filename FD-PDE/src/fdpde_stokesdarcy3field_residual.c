#include "fdpde_stokes.h"
#include "fdpde_stokesdarcy3field.h"

// ---------------------------------------
/*@
FormFunction_StokesDarcy3Field - Residual evaluation function

Use: internal
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormFunction_StokesDarcy3Field"
PetscErrorCode FormFunction_StokesDarcy3Field(SNES snes, Vec x, Vec f, void *ctx)
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
  PetscLogDouble tlog[10];
  
  PetscFunctionBegin;
  PetscTime(&tlog[0]);
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
  PetscTime(&tlog[1]);

  // Update coefficients
  ierr = fd->ops->form_coefficient(fd,dmPV,x,dmCoeff,fd->coeff,fd->user_context);CHKERRQ(ierr);
  PetscTime(&tlog[2]);

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
  PetscTime(&tlog[3]);

  // Get location slots
  PetscInt pv_slot[6],coeff_e[4],coeff_v[4],coeff_B[4],coeff_D2[4],coeff_D3[4],coeff_D4[4];
  ierr = GetLocationSlots(dmPV,dmCoeff,pv_slot,coeff_e,coeff_v,coeff_B); CHKERRQ(ierr);
  ierr = GetLocationSlots_Darcy3Field(dmPV,dmCoeff,pv_slot,coeff_e,coeff_D2,coeff_D3,coeff_D4);CHKERRQ(ierr);

  // Loop over elements
  for (j = sz; j<sz+nz; j++) {
    for (i = sx; i<sx+nx; i++) {
      PetscScalar fval, fval1;

      // Continuity equation
      ierr = ContinuityResidual(i,j,_xlocal,_coefflocal,coordx,coordz,n,pv_slot,coeff_e,&fval);CHKERRQ(ierr);
      ierr = ContinuityResidual_Darcy3Field(i,j,_xlocal,_coefflocal,coordx,coordz,n,pv_slot,coeff_D2,coeff_D3,coeff_D4,&fval1);CHKERRQ(ierr);
      ff[j][i][pv_slot[4]] = fval + fval1; // ELEMENT 0

      // Compaction pressure equation
      ierr = CompactionResidual(i,j,_xlocal,_coefflocal,coordx,coordz,n,pv_slot,coeff_e,&fval);CHKERRQ(ierr);
      ff[j][i][pv_slot[5]] = fval; // ELEMENT 1

      // X-Momentum equation - same as for Stokes
      if (i > 0) {
        ierr = XMomentumResidual(i,j,_xlocal,_coefflocal,coordx,coordz,n,pv_slot,coeff_e,coeff_B,coeff_v,&fval);CHKERRQ(ierr);
        ff[j][i][pv_slot[0]] = fval; // LEFT
      }

      // Z-Momentum equation - same as for Stokes
      if (j > 0) {
        ierr = ZMomentumResidual(i,j,_xlocal,_coefflocal,coordx,coordz,n,pv_slot,coeff_e,coeff_B,coeff_v,&fval);CHKERRQ(ierr);
        ff[j][i][pv_slot[2]] = fval; // DOWN
      }
    }
  }
  PetscTime(&tlog[4]);

  // Boundary conditions - edges and element
  ierr = DMStagBCListApplyFace_StokesDarcy3Field(_xlocal,_coefflocal,bclist->bc_f,bclist->nbc_face,coordx,coordz,n,pv_slot,coeff_v,ff);CHKERRQ(ierr);
  ierr = DMStagBCListApplyElement_StokesDarcy3Field(_xlocal,_coefflocal,bclist->bc_e,bclist->nbc_element,coordx,coordz,n,pv_slot,coeff_D2,ff);CHKERRQ(ierr);
  PetscTime(&tlog[5]);

  // Restore arrays, local vectors
  ierr = DMStagRestoreProductCoordinateArraysRead(dmPV,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArray(dmPV,flocal,&ff); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmPV,&xlocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmCoeff,&coefflocal); CHKERRQ(ierr);

  // Map local to global
  ierr = DMLocalToGlobalBegin(dmPV,flocal,INSERT_VALUES,f); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dmPV,flocal,INSERT_VALUES,f); CHKERRQ(ierr);

  ierr = VecDestroy(&flocal); CHKERRQ(ierr);
  PetscTime(&tlog[6]);
  
  if (fd->log_info) {
    printf("  FormFunction_StokesDarcy3Field: total            %1.2e\n",tlog[6]-tlog[0]);
    printf("  FormFunction_StokesDarcy3Field: bclist->evaluate %1.2e\n",tlog[1]-tlog[0]);
    printf("  FormFunction_StokesDarcy3Field: form_coefficient %1.2e\n",tlog[2]-tlog[1]);
    printf("  FormFunction_StokesDarcy3Field: g2l(input)       %1.2e\n",tlog[3]-tlog[2]);
    printf("  FormFunction_StokesDarcy3Field: cell-loop        %1.2e\n",tlog[4]-tlog[3]);
    printf("  FormFunction_StokesDarcy3Field: bclist_set       %1.2e\n",tlog[5]-tlog[4]);
    printf("  FormFunction_StokesDarcy3Field: g2l(output)      %1.2e\n",tlog[6]-tlog[5]);
    printf("----------------------------------------------------------------------\n");
  }
  
  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
GetLocationSlots_Darcy3Field - (STOKESDARCY3FIELD) get dmstag location slots
Use: internal
@*/
// ---------------------------------------
PetscErrorCode GetLocationSlots_Darcy3Field(DM dmPV, DM dmCoeff, PetscInt *pv_slot, PetscInt *coeff_e, PetscInt *coeff_D2, PetscInt *coeff_D3, PetscInt *coeff_D4)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = DMStagGetLocationSlot(dmPV,DMSTAG_LEFT,   SD3_DOF_V, &pv_slot[0]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmPV,DMSTAG_RIGHT,  SD3_DOF_V, &pv_slot[1]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmPV,DMSTAG_DOWN,   SD3_DOF_V, &pv_slot[2]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmPV,DMSTAG_UP,     SD3_DOF_V, &pv_slot[3]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmPV,DMSTAG_ELEMENT,SD3_DOF_P, &pv_slot[4]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmPV,DMSTAG_ELEMENT,SD3_DOF_PC,&pv_slot[5]); CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(dmCoeff,DMSTAG_ELEMENT,SD3_COEFF_ELEMENT_C, &coeff_e[SD3_COEFF_ELEMENT_C]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmCoeff,DMSTAG_ELEMENT,SD3_COEFF_ELEMENT_A, &coeff_e[SD3_COEFF_ELEMENT_A]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmCoeff,DMSTAG_ELEMENT,SD3_COEFF_ELEMENT_D1,&coeff_e[SD3_COEFF_ELEMENT_D1]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmCoeff,DMSTAG_ELEMENT,SD3_COEFF_ELEMENT_DC,&coeff_e[SD3_COEFF_ELEMENT_DC]);CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(dmCoeff,DMSTAG_LEFT,  SD3_COEFF_FACE_D2, &coeff_D2[0]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmCoeff,DMSTAG_RIGHT, SD3_COEFF_FACE_D2, &coeff_D2[1]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmCoeff,DMSTAG_DOWN,  SD3_COEFF_FACE_D2, &coeff_D2[2]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmCoeff,DMSTAG_UP,    SD3_COEFF_FACE_D2, &coeff_D2[3]);CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(dmCoeff,DMSTAG_LEFT,  SD3_COEFF_FACE_D3, &coeff_D3[0]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmCoeff,DMSTAG_RIGHT, SD3_COEFF_FACE_D3, &coeff_D3[1]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmCoeff,DMSTAG_DOWN,  SD3_COEFF_FACE_D3, &coeff_D3[2]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmCoeff,DMSTAG_UP,    SD3_COEFF_FACE_D3, &coeff_D3[3]);CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(dmCoeff,DMSTAG_LEFT,  SD3_COEFF_FACE_D4, &coeff_D4[0]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmCoeff,DMSTAG_RIGHT, SD3_COEFF_FACE_D4, &coeff_D4[1]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmCoeff,DMSTAG_DOWN,  SD3_COEFF_FACE_D4, &coeff_D4[2]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmCoeff,DMSTAG_UP,    SD3_COEFF_FACE_D4, &coeff_D4[3]);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
ContinuityResidual_Darcy3Field - (STOKESDARCY3FIELD) calculates the div(Darcy flux) per cell

Use: internal
@*/
// ---------------------------------------
PetscErrorCode ContinuityResidual_Darcy3Field(PetscInt i, PetscInt j,PetscScalar ***_xlocal,PetscScalar ***_coefflocal,PetscScalar **coordx, PetscScalar **coordz, PetscInt n[],PetscInt pv_slot[], PetscInt coeff_D2[],PetscInt coeff_D3[],PetscInt coeff_D4[],PetscScalar *ff)
{
  PetscScalar    ffi, xx[10], D2[4], D3[4], D4[4], dPdx[4], dPCdx[4], qD[4], dx, dx1, dx2, dz, dz1, dz2;
  PetscInt       ii, iprev, inext, icenter, Nx, Nz, im, jm, ip, jp, P_slot,Pc_slot;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  Nx = n[0]; Nz = n[1]; icenter = n[2]; iprev = n[3]; inext = n[4];

  P_slot = 4; Pc_slot = 5;
  if (i == 0   ) im = i; else im = i-1;
  if (i == Nx-1) ip = i; else ip = i+1;
  if (j == 0   ) jm = j; else jm = j-1;
  if (j == Nz-1) jp = j; else jp = j+1;

  // Get stencil values
  xx[0] = _xlocal[j ][i ][pv_slot[P_slot]];
  xx[1] = _xlocal[j ][ip][pv_slot[P_slot]];
  xx[2] = _xlocal[j ][im][pv_slot[P_slot]];
  xx[3] = _xlocal[jp][i ][pv_slot[P_slot]];
  xx[4] = _xlocal[jm][i ][pv_slot[P_slot]];

  xx[5] = _xlocal[j ][i ][pv_slot[Pc_slot]];
  xx[6] = _xlocal[j ][ip][pv_slot[Pc_slot]];
  xx[7] = _xlocal[j ][im][pv_slot[Pc_slot]];
  xx[8] = _xlocal[jp][i ][pv_slot[Pc_slot]];
  xx[9] = _xlocal[jm][i ][pv_slot[Pc_slot]];
  
  // Coefficients - D2, D3, D4
  for (ii = 0; ii < 4; ii++) {
    D2[ii] = _coefflocal[j][i][coeff_D2[ii]];
    D3[ii] = _coefflocal[j][i][coeff_D3[ii]];
    D4[ii] = _coefflocal[j][i][coeff_D4[ii]];
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

  // Calculate residual div (D2*grad(p)+D3 + D4*grad(Pc))
  dPdx[0] = (xx[0]-xx[2])/dx1; // dP/dx_left
  dPdx[1] = (xx[1]-xx[0])/dx2; // dP/dx_right
  dPdx[2] = (xx[0]-xx[4])/dz1; // dP/dz_down
  dPdx[3] = (xx[3]-xx[0])/dz2; // dP/dz_up

  dPCdx[0] = (xx[5]-xx[7])/dx1;
  dPCdx[1] = (xx[6]-xx[5])/dx2;
  dPCdx[2] = (xx[5]-xx[9])/dz1;
  dPCdx[3] = (xx[8]-xx[5])/dz2;

  // Darcy flux = D2*grad(p)+D3
  for (ii = 0; ii < 4; ii++) { qD[ii] = D2[ii]*dPdx[ii] + D3[ii] + D4[ii]*dPCdx[ii]; }

  // div(Darcy flux)
  ffi = (qD[1]-qD[0])/dx + (qD[3]-qD[2])/dz; 

  *ff = ffi;
  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
CompactionResidual - calculates the compaction pressure residual per dof

Use: internal
@*/
// ---------------------------------------
PetscErrorCode CompactionResidual(PetscInt i, PetscInt j,PetscScalar ***_xlocal,PetscScalar ***_coefflocal,PetscScalar **coordx, PetscScalar **coordz, PetscInt n[],PetscInt pv_slot[], PetscInt coeff_e[],PetscScalar *ff)
{
  PetscScalar    ffi, xx[5], DC, D1, dx, dz;
  PetscInt       iprev, inext,iL,iR,iU,iD,Pc_slot;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  iprev = n[3]; 
  inext = n[4];
  iL = 0; iR  = 1; iD = 2; iU  = 3; Pc_slot = 5;

  // Get stencil values
  xx[0] = _xlocal[j][i][pv_slot[Pc_slot]];
  xx[1] = _xlocal[j][i][pv_slot[iL]];
  xx[2] = _xlocal[j][i][pv_slot[iR]];
  xx[3] = _xlocal[j][i][pv_slot[iD]];
  xx[4] = _xlocal[j][i][pv_slot[iU]];
  
  // Coefficients
  D1 = _coefflocal[j][i][coeff_e[SD3_COEFF_ELEMENT_D1]];
  DC = _coefflocal[j][i][coeff_e[SD3_COEFF_ELEMENT_DC]];

  // Calculate residual
  dx = coordx[i][inext]-coordx[i][iprev];
  dz = coordz[j][inext]-coordz[j][iprev];
  ffi = (xx[2]-xx[1])/dx + (xx[4]-xx[3])/dz + D1*xx[0] - DC;

  *ff = ffi;
  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
DMStagBCListApplyFace_StokesDarcy3Field - function to apply boundary conditions for StokesDarcy 3 Field (face)

Use: internal
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "DMStagBCListApplyFace_StokesDarcy3Field"
PetscErrorCode DMStagBCListApplyFace_StokesDarcy3Field(PetscScalar ***_xlocal,PetscScalar ***_coefflocal, DMStagBC *bclist, PetscInt nbc, PetscScalar **coordx, PetscScalar **coordz,PetscInt n[], PetscInt pv_slot[], PetscInt coeff_v[], PetscScalar ***ff)
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
        xxT[0] = _xlocal[j][i][pv_slot[iL]];
        xxT[1] = _xlocal[j][i][pv_slot[iR]];
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
DMStagBCListApplyElement_StokesDarcy3Field - function to apply boundary conditions for StokesDarcy equations 3 field

Use: internal
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "DMStagBCListApplyElement_StokesDarcy3Field"
PetscErrorCode DMStagBCListApplyElement_StokesDarcy3Field(PetscScalar ***_xlocal,PetscScalar ***_coefflocal, DMStagBC *bclist, PetscInt nbc, PetscScalar **coordx, PetscScalar **coordz,PetscInt n[], PetscInt pv_slot[], PetscInt coeff_D2[], PetscScalar ***ff)
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

      // Pressure
      if ((i == 0) && (bclist[ibc].point.c == SD3_DOF_P)) { 
        D2_Left = _coefflocal[j][i][coeff_D2[iL]];
        dx = coordx[i][inext]-coordx[i][iprev];
        ff[j][i][idx] += -2.0 * D2_Left* (xx -bclist[ibc].val)/dx/dx;
      }

      if ((i == Nx-1) && (bclist[ibc].point.c == SD3_DOF_P)) { 
        D2_Right = _coefflocal[j][i][coeff_D2[iR]];
        dx = coordx[i][inext]-coordx[i][iprev];
        ff[j][i][idx] += 2.0 * D2_Right* (bclist[ibc].val - xx)/dx/dx;
      }

      if ((j == 0) && (bclist[ibc].point.c == SD3_DOF_P)) { 
        D2_Down = _coefflocal[j][i][coeff_D2[iD]];
        dz = coordz[j][inext]-coordz[j][iprev];
        ff[j][i][idx] += -2.0 * D2_Down* (xx - bclist[ibc].val)/dz/dz;
      }

      if ((j == Nz-1) && (bclist[ibc].point.c == SD3_DOF_P)) { 
        D2_Up = _coefflocal[j][i][coeff_D2[iU]];
        dz = coordz[j][inext]-coordz[j][iprev];
        ff[j][i][idx] += 2.0 * D2_Up* (bclist[ibc].val - xx)/dz/dz;
      }

      if (bclist[ibc].point.c == SD3_DOF_PC) {
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"BC_DIRICHLET type on the true boundary for FDPDE_STOKESDARCY3FIELD [PC-ELEMENT] is not yet implemented. Use BC_DIRICHLET_STAG type instead!");
      } 
    }

    if (bclist[ibc].type == BC_NEUMANN) {
      i   = bclist[ibc].point.i;
      j   = bclist[ibc].point.j;
      idx = bclist[ibc].idx;

      // Pressure
      if ((i == 0) && (bclist[ibc].point.c == SD3_DOF_P)) { 
        D2_Left = _coefflocal[j][i][coeff_D2[iL]];
        dx = coordx[i][inext]-coordx[i][iprev];
        ff[j][i][idx] += -D2_Left*bclist[ibc].val/dx;
      }

      if ((i == Nx-1) && (bclist[ibc].point.c == SD3_DOF_P)) { 
        D2_Right = _coefflocal[j][i][coeff_D2[iR]];
        dx = coordx[i][inext]-coordx[i][iprev];
        ff[j][i][idx] += D2_Right*bclist[ibc].val/dx;
      }

      if ((j == 0) && (bclist[ibc].point.c == SD3_DOF_P)) { 
        D2_Down = _coefflocal[j][i][coeff_D2[iD]];
        dz = coordz[j][inext]-coordz[j][iprev];
        ff[j][i][idx] += -D2_Down*bclist[ibc].val/dz;
      }

      if ((j == Nz-1) && (bclist[ibc].point.c == SD3_DOF_P)) { 
        D2_Up = _coefflocal[j][i][coeff_D2[iU]];
        dz = coordz[j][inext]-coordz[j][iprev];
        ff[j][i][idx] += D2_Up*bclist[ibc].val/dz;
      }

      if (bclist[ibc].point.c == SD3_DOF_PC) {
        if (bclist[ibc].val != 0.0) {
          SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Non-zero BC type NEUMANN for FDPDE_STOKESDARCY3FIELD [PC-ELEMENT] is not yet implemented.");
        }
      } 
    }
  }
  PetscFunctionReturn(0);
}
