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

  ierr = DMGetLocalVector(dmCoeff, &coefflocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dmCoeff, fd->coeff, INSERT_VALUES, coefflocal); CHKERRQ(ierr);

  // Get dm coordinates array
  ierr = DMStagGetProductCoordinateArraysRead(dmPV,&coordx,&coordz,NULL);CHKERRQ(ierr);

  // Create residual local vector
  ierr = DMCreateLocalVector(dmPV, &flocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dmPV, flocal, &ff); CHKERRQ(ierr);

  // Loop over elements
  for (j = sz; j<sz+nz; j++) {
    for (i = sx; i<sx+nx; i++) {
      PetscScalar fval, fval1;

      // Continuity equation
      ierr = ContinuityResidual(dmPV,xlocal,dmCoeff,coefflocal,coordx,coordz,i,j,n,&fval);CHKERRQ(ierr);
      ierr = ContinuityResidual_Darcy3Field(dmPV,xlocal,dmCoeff,coefflocal,coordx,coordz,i,j,n,&fval1);CHKERRQ(ierr);
      ierr = DMStagGetLocationSlot(dmPV, DMSTAG_ELEMENT, SD3_DOF_P, &idx); CHKERRQ(ierr);
      ff[j][i][idx] = fval + fval1;

      // Compaction pressure equation
      ierr = CompactionResidual(dmPV,xlocal,dmCoeff,coefflocal,coordx,coordz,i,j,n,&fval);CHKERRQ(ierr);
      ierr = DMStagGetLocationSlot(dmPV, DMSTAG_ELEMENT, SD3_DOF_PC, &idx); CHKERRQ(ierr);
      ff[j][i][idx] = fval;

      // X-Momentum equation - same as for Stokes
      if (i > 0) {
        ierr = XMomentumResidual(dmPV,xlocal,dmCoeff,coefflocal,coordx,coordz,i,j,n,&fval);CHKERRQ(ierr);
        ierr = DMStagGetLocationSlot(dmPV, DMSTAG_LEFT, SD3_DOF_V, &idx); CHKERRQ(ierr);
        ff[j][i][idx] = fval;
      }

      // Z-Momentum equation - same as for Stokes
      if (j > 0) {
        ierr = ZMomentumResidual(dmPV,xlocal,dmCoeff,coefflocal,coordx,coordz,i,j,n,&fval);CHKERRQ(ierr);
        ierr = DMStagGetLocationSlot(dmPV, DMSTAG_DOWN, SD3_DOF_V, &idx); CHKERRQ(ierr);
        ff[j][i][idx] = fval;
      }
    }
  }

  // Boundary conditions - edges and element
  ierr = DMStagBCListApplyFace_StokesDarcy3Field(dmPV,xlocal,dmCoeff,coefflocal,bclist->bc_f,bclist->nbc_face,coordx,coordz,n,ff);CHKERRQ(ierr);
  ierr = DMStagBCListApplyElement_StokesDarcy3Field(dmPV,xlocal,dmCoeff,coefflocal,bclist->bc_e,bclist->nbc_element,coordx,coordz,n,ff);CHKERRQ(ierr);

  // Restore arrays, local vectors
  ierr = DMStagRestoreProductCoordinateArraysRead(dmPV,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArray(dmPV,flocal,&ff); CHKERRQ(ierr);
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
ContinuityResidual_Darcy3Field - (STOKESDARCY3FIELD) calculates the div(Darcy flux) per cell

Use: internal
@*/
// ---------------------------------------
PetscErrorCode ContinuityResidual_Darcy3Field(DM dm, Vec xlocal, DM dmcoeff,Vec coefflocal, PetscScalar **coordx, PetscScalar **coordz, PetscInt i, PetscInt j, PetscInt n[],PetscScalar *ff)
{
  PetscScalar    ffi, xx[10], D2[4], D3[4], D4[4], dPdx[4], dPCdx[4], qD[4], dx, dx1, dx2, dz, dz1, dz2;
  PetscInt       ii, iprev, inext, icenter, Nx, Nz;
  DMStagStencil  point[10];
  PetscErrorCode ierr;
  PetscFunctionBegin;

  Nx = n[0]; Nz = n[1]; icenter = n[2]; iprev = n[3]; inext = n[4];

  // Get stencil values
  point[0].i = i  ; point[0].j = j  ; point[0].loc = DMSTAG_ELEMENT; point[0].c = SD3_DOF_P;
  point[1].i = i+1; point[1].j = j  ; point[1].loc = DMSTAG_ELEMENT; point[1].c = SD3_DOF_P;
  point[2].i = i-1; point[2].j = j  ; point[2].loc = DMSTAG_ELEMENT; point[2].c = SD3_DOF_P;
  point[3].i = i  ; point[3].j = j+1; point[3].loc = DMSTAG_ELEMENT; point[3].c = SD3_DOF_P;
  point[4].i = i  ; point[4].j = j-1; point[4].loc = DMSTAG_ELEMENT; point[4].c = SD3_DOF_P;

  point[5].i = i  ; point[5].j = j  ; point[5].loc = DMSTAG_ELEMENT; point[5].c = SD3_DOF_PC;
  point[6].i = i+1; point[6].j = j  ; point[6].loc = DMSTAG_ELEMENT; point[6].c = SD3_DOF_PC;
  point[7].i = i-1; point[7].j = j  ; point[7].loc = DMSTAG_ELEMENT; point[7].c = SD3_DOF_PC;
  point[8].i = i  ; point[8].j = j+1; point[8].loc = DMSTAG_ELEMENT; point[8].c = SD3_DOF_PC;
  point[9].i = i  ; point[9].j = j-1; point[9].loc = DMSTAG_ELEMENT; point[9].c = SD3_DOF_PC;

  // For boundaries remove the flux terms
  if (i == Nx-1) { point[1] = point[0]; point[6] = point[5]; }
  if (i == 0   ) { point[2] = point[0]; point[7] = point[5]; }
  if (j == Nz-1) { point[3] = point[0]; point[8] = point[5]; }
  if (j == 0   ) { point[4] = point[0]; point[9] = point[5]; }
  ierr = DMStagVecGetValuesStencil(dm, xlocal, 10, point, xx); CHKERRQ(ierr);
  
  // Coefficients - D2, D3, D4
  point[0].i = i; point[0].j = j; point[0].loc = DMSTAG_LEFT;  point[0].c = SD3_COEFF_FACE_D2;
  point[1].i = i; point[1].j = j; point[1].loc = DMSTAG_RIGHT; point[1].c = SD3_COEFF_FACE_D2;
  point[2].i = i; point[2].j = j; point[2].loc = DMSTAG_DOWN;  point[2].c = SD3_COEFF_FACE_D2;
  point[3].i = i; point[3].j = j; point[3].loc = DMSTAG_UP;    point[3].c = SD3_COEFF_FACE_D2;
  ierr = DMStagVecGetValuesStencil(dmcoeff, coefflocal, 4, point, D2); CHKERRQ(ierr);

  for (ii = 0; ii < 4; ii++) { point[ii].c = SD3_COEFF_FACE_D3; }
  ierr = DMStagVecGetValuesStencil(dmcoeff, coefflocal, 4, point, D3); CHKERRQ(ierr);

  for (ii = 0; ii < 4; ii++) { point[ii].c = SD3_COEFF_FACE_D4; }
  ierr = DMStagVecGetValuesStencil(dmcoeff, coefflocal, 4, point, D4); CHKERRQ(ierr);

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
PetscErrorCode CompactionResidual(DM dm, Vec xlocal, DM dmcoeff,Vec coefflocal, PetscScalar **coordx, PetscScalar **coordz, PetscInt i, PetscInt j, PetscInt n[],PetscScalar *ff)
{
  PetscScalar    ffi, xx[5], DC, D1, dx, dz;
  PetscInt       iprev, inext, nEntries = 5;
  DMStagStencil  point[5];
  PetscErrorCode ierr;
  PetscFunctionBegin;

  iprev = n[3]; 
  inext = n[4];

  // Get stencil values
  point[0].i = i; point[0].j = j; point[0].loc = DMSTAG_ELEMENT; point[0].c = SD3_DOF_PC;
  point[1].i = i; point[1].j = j; point[1].loc = DMSTAG_LEFT;    point[1].c = SD3_DOF_V;
  point[2].i = i; point[2].j = j; point[2].loc = DMSTAG_RIGHT;   point[2].c = SD3_DOF_V;
  point[3].i = i; point[3].j = j; point[3].loc = DMSTAG_DOWN;    point[3].c = SD3_DOF_V;
  point[4].i = i; point[4].j = j; point[4].loc = DMSTAG_UP;      point[4].c = SD3_DOF_V;
  ierr = DMStagVecGetValuesStencil(dm, xlocal, nEntries, point, xx); CHKERRQ(ierr);
  
  // Coefficients
  point[0].i = i; point[0].j = j; point[0].loc = DMSTAG_ELEMENT;  point[0].c = SD3_COEFF_ELEMENT_D1;
  ierr = DMStagVecGetValuesStencil(dmcoeff, coefflocal, 1, point, &D1); CHKERRQ(ierr);

  point[0].i = i; point[0].j = j; point[0].loc = DMSTAG_ELEMENT;  point[0].c = SD3_COEFF_ELEMENT_DC;
  ierr = DMStagVecGetValuesStencil(dmcoeff, coefflocal, 1, point, &DC); CHKERRQ(ierr);

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
PetscErrorCode DMStagBCListApplyFace_StokesDarcy3Field(DM dm, Vec xlocal,DM dmcoeff, Vec coefflocal, DMStagBC *bclist, PetscInt nbc, PetscScalar **coordx, PetscScalar **coordz,PetscInt n[], PetscScalar ***ff)
{
  PetscScalar    xx, xxT[2],dx, dz;
  PetscScalar    A_Left, A_Right, A_Up, A_Down;
  PetscInt       i, j, ibc, idx, iprev, inext, Nx, Nz;
  DMStagStencil  point, pointT[2];
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  // dm domain info
  Nx = n[0]; Nz = n[1]; iprev = n[3]; inext = n[4];

  // Loop over all boundaries
  for (ibc = 0; ibc<nbc; ibc++) {

    if (bclist[ibc].type == BC_DIRICHLET_STAG) {
      i   = bclist[ibc].point.i;
      j   = bclist[ibc].point.j;
      idx = bclist[ibc].idx;

      // Get residual value
      ierr = DMStagVecGetValuesStencil(dm, xlocal, 1, &bclist[ibc].point, &xx); CHKERRQ(ierr);
      ff[j][i][idx] = xx - bclist[ibc].val;
    }
    
    if (bclist[ibc].type == BC_DIRICHLET) {
      i   = bclist[ibc].point.i;
      j   = bclist[ibc].point.j;
      idx = bclist[ibc].idx;

      // Get residual value
      ierr = DMStagVecGetValuesStencil(dm, xlocal, 1, &bclist[ibc].point, &xx); CHKERRQ(ierr);

      // Stokes flow - add flux terms
      if ((j == 0) && (i > 0) && (bclist[ibc].point.loc == DMSTAG_LEFT)) { // Vx down - only interior points
        point.i = i; point.j = j; point.loc = DMSTAG_DOWN_LEFT; point.c = 0;
        ierr = DMStagVecGetValuesStencil(dmcoeff, coefflocal, 1, &point, &A_Down); CHKERRQ(ierr);
        dz = coordz[j][inext]-coordz[j][iprev];
        ff[j][i][idx] += -2.0 * A_Down*( xx - bclist[ibc].val)/dz/dz;
      }

      else if ((j == 0) && (i < Nx-1) && (bclist[ibc].point.loc == DMSTAG_RIGHT)) { // Vx down-special case
        point.i = i; point.j = j; point.loc = DMSTAG_DOWN_RIGHT; point.c = 0;
        ierr = DMStagVecGetValuesStencil(dmcoeff, coefflocal, 1, &point, &A_Down); CHKERRQ(ierr);
        dz = coordz[j][inext]-coordz[j][iprev];
        ff[j][i][idx] += -2.0 * A_Down*( xx - bclist[ibc].val)/dz/dz;
      }

      else if ((j == Nz-1) && (i > 0) && (bclist[ibc].point.loc == DMSTAG_LEFT)) { // Vx up - only interior points
        point.i = i; point.j = j; point.loc = DMSTAG_UP_LEFT; point.c = 0;
        ierr = DMStagVecGetValuesStencil(dmcoeff, coefflocal, 1, &point, &A_Up); CHKERRQ(ierr);
        dz = coordz[j][inext]-coordz[j][iprev];
        ff[j][i][idx] += 2.0 * A_Up*( bclist[ibc].val - xx)/dz/dz;
      }

      else if ((j == Nz-1) && (i < Nx-1) && (bclist[ibc].point.loc == DMSTAG_RIGHT)) { // Vx up - special case
        point.i = i; point.j = j; point.loc = DMSTAG_UP_RIGHT; point.c = 0;
        ierr = DMStagVecGetValuesStencil(dmcoeff, coefflocal, 1, &point, &A_Up); CHKERRQ(ierr);
        dz = coordz[j][inext]-coordz[j][iprev];
        ff[j][i][idx] += 2.0 * A_Up*( bclist[ibc].val - xx)/dz/dz;
      }

      else if ((i == 0) && (j > 0) && (bclist[ibc].point.loc == DMSTAG_DOWN)) { // Vz left
        point.i = i; point.j = j; point.loc = DMSTAG_DOWN_LEFT; point.c = 0;
        ierr = DMStagVecGetValuesStencil(dmcoeff, coefflocal, 1, &point, &A_Left); CHKERRQ(ierr);
        dx = coordx[i][inext]-coordx[i][iprev];
        ff[j][i][idx] += -2.0 * A_Left*( xx - bclist[ibc].val)/dx/dx;
      }

      else if ((i == Nx-1) && (j > 0) && (bclist[ibc].point.loc == DMSTAG_DOWN)) { // Vz right
        point.i = i; point.j = j; point.loc = DMSTAG_DOWN_RIGHT; point.c = 0;
        ierr = DMStagVecGetValuesStencil(dmcoeff, coefflocal, 1, &point, &A_Right); CHKERRQ(ierr);
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
        point.i = i; point.j = j; point.loc = DMSTAG_DOWN_LEFT; point.c = 0;
        ierr = DMStagVecGetValuesStencil(dmcoeff, coefflocal, 1, &point, &A_Down); CHKERRQ(ierr);
        dz = coordz[j][inext]-coordz[j][iprev];
        ff[j][i][idx] += -A_Down*bclist[ibc].val/dz;
      }

      if ((j == 0) && (bclist[ibc].point.loc == DMSTAG_RIGHT)) { // Vx down-special case
        point.i = i; point.j = j; point.loc = DMSTAG_DOWN_RIGHT; point.c = 0;
        ierr = DMStagVecGetValuesStencil(dmcoeff, coefflocal, 1, &point, &A_Down); CHKERRQ(ierr);
        dz = coordz[j][inext]-coordz[j][iprev];
        ff[j][i][idx] += -A_Down*bclist[ibc].val/dz;
      }

      if ((j == Nz-1) && (bclist[ibc].point.loc == DMSTAG_LEFT)) { // Vx up
        point.i = i; point.j = j; point.loc = DMSTAG_UP_LEFT; point.c = 0;
        ierr = DMStagVecGetValuesStencil(dmcoeff, coefflocal, 1, &point, &A_Up); CHKERRQ(ierr);
        dz = coordz[j][inext]-coordz[j][iprev];
        ff[j][i][idx] += A_Up*bclist[ibc].val/dz;
      }

      if ((j == Nz-1) && (bclist[ibc].point.loc == DMSTAG_RIGHT)) { // Vx up - special case
        point.i = i; point.j = j; point.loc = DMSTAG_UP_RIGHT; point.c = 0;
        ierr = DMStagVecGetValuesStencil(dmcoeff, coefflocal, 1, &point, &A_Up); CHKERRQ(ierr);
        dz = coordz[j][inext]-coordz[j][iprev];
        ff[j][i][idx] += A_Up*bclist[ibc].val/dz;
      }

      if ((i == 0) && (bclist[ibc].point.loc == DMSTAG_DOWN)) { // Vz left
        point.i = i; point.j = j; point.loc = DMSTAG_DOWN_LEFT; point.c = 0;
        ierr = DMStagVecGetValuesStencil(dmcoeff, coefflocal, 1, &point, &A_Left); CHKERRQ(ierr);
        dx = coordx[i][inext]-coordx[i][iprev];
        ff[j][i][idx] += -A_Left*bclist[ibc].val/dx;
      }

      if ((i == Nx-1) && (bclist[ibc].point.loc == DMSTAG_DOWN)) { // Vz right
        point.i = i; point.j = j; point.loc = DMSTAG_DOWN_RIGHT; point.c = 0;
        ierr = DMStagVecGetValuesStencil(dmcoeff, coefflocal, 1, &point, &A_Right); CHKERRQ(ierr);
        dx = coordx[i][inext]-coordx[i][iprev];
        ff[j][i][idx] += A_Right*bclist[ibc].val/dx;
      }
      
    }

    if (bclist[ibc].type == BC_NEUMANN_T) { // of the form dvi/dxi
      i   = bclist[ibc].point.i;
      j   = bclist[ibc].point.j;
      idx = bclist[ibc].idx;

      if ((i == 0) && (bclist[ibc].point.loc == DMSTAG_LEFT)) { // left dVx/dx = a
        pointT[0].i = i  ; pointT[0].j = j; pointT[0].loc = DMSTAG_LEFT; pointT[0].c = 0;
        pointT[1].i = i+1; pointT[1].j = j; pointT[1].loc = DMSTAG_LEFT; pointT[1].c = 0;
        ierr = DMStagVecGetValuesStencil(dm,xlocal,2,pointT,xxT); CHKERRQ(ierr);
        dx = coordx[i][inext]-coordx[i][iprev];
        ff[j][i][idx] = xxT[1]-xxT[0]-bclist[ibc].val*dx;
      }
      if ((i == Nx-1) && (bclist[ibc].point.loc == DMSTAG_RIGHT)) { // right dVx/dx = a
        pointT[0].i = i  ; pointT[0].j = j; pointT[0].loc = DMSTAG_LEFT ; pointT[0].c = 0;
        pointT[1].i = i  ; pointT[1].j = j; pointT[1].loc = DMSTAG_RIGHT; pointT[1].c = 0;
        ierr = DMStagVecGetValuesStencil(dm,xlocal,2,pointT,xxT); CHKERRQ(ierr);
        dx = coordx[i][inext]-coordx[i][iprev];
        ff[j][i][idx] = xxT[1]-xxT[0]-bclist[ibc].val*dx;
      }
      if ((j == 0) && (bclist[ibc].point.loc == DMSTAG_DOWN)) { // down dVz/dz = a
        pointT[0].i = i; pointT[0].j = j  ; pointT[0].loc = DMSTAG_DOWN; pointT[0].c = 0;
        pointT[1].i = i; pointT[1].j = j+1; pointT[1].loc = DMSTAG_DOWN; pointT[1].c = 0;
        ierr = DMStagVecGetValuesStencil(dm,xlocal,2,pointT,xxT); CHKERRQ(ierr);
        dz = coordz[j][inext]-coordz[j][iprev];
        ff[j][i][idx] = xxT[1]-xxT[0]-bclist[ibc].val*dz;
      }
      if ((j == Nz-1) && (bclist[ibc].point.loc == DMSTAG_UP)) { // up dVz/dz = a
        pointT[0].i = i; pointT[0].j = j-1; pointT[0].loc = DMSTAG_UP; pointT[0].c = 0;
        pointT[1].i = i; pointT[1].j = j  ; pointT[1].loc = DMSTAG_UP; pointT[1].c = 0;
        ierr = DMStagVecGetValuesStencil(dm,xlocal,2,pointT,xxT); CHKERRQ(ierr);
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
PetscErrorCode DMStagBCListApplyElement_StokesDarcy3Field(DM dm, Vec xlocal,DM dmcoeff, Vec coefflocal, DMStagBC *bclist, PetscInt nbc, PetscScalar **coordx, PetscScalar **coordz,PetscInt n[], PetscScalar ***ff)
{
  PetscScalar    xx, dx, dz;
  PetscScalar    D2_Left, D2_Right, D2_Up, D2_Down;
  PetscInt       i, j, ibc, idx, iprev, inext, Nx, Nz;
  DMStagStencil  point;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  // dm domain info
  Nx = n[0]; Nz = n[1]; iprev = n[3]; inext = n[4];

  // Loop over all boundaries
  for (ibc = 0; ibc<nbc; ibc++) {

    if (bclist[ibc].type == BC_DIRICHLET_STAG) {
      i   = bclist[ibc].point.i;
      j   = bclist[ibc].point.j;
      idx = bclist[ibc].idx;

      // Get residual value
      ierr = DMStagVecGetValuesStencil(dm, xlocal, 1, &bclist[ibc].point, &xx); CHKERRQ(ierr);
      ff[j][i][idx] = xx - bclist[ibc].val;
    }

    if (bclist[ibc].type == BC_DIRICHLET) {
      i   = bclist[ibc].point.i;
      j   = bclist[ibc].point.j;
      idx = bclist[ibc].idx;

      // Get residual value
      ierr = DMStagVecGetValuesStencil(dm, xlocal, 1, &bclist[ibc].point, &xx); CHKERRQ(ierr);

      // Pressure
      if ((i == 0) && (bclist[ibc].point.c == SD3_DOF_P)) { 
        point.i = i; point.j = j; point.loc = DMSTAG_LEFT; point.c = 1;
        ierr = DMStagVecGetValuesStencil(dmcoeff, coefflocal, 1, &point, &D2_Left); CHKERRQ(ierr);
        dx = coordx[i][inext]-coordx[i][iprev];
        ff[j][i][idx] += -2.0 * D2_Left* (xx -bclist[ibc].val)/dx/dx;
      }

      if ((i == Nx-1) && (bclist[ibc].point.c == SD3_DOF_P)) { 
        point.i = i; point.j = j; point.loc = DMSTAG_RIGHT; point.c = 1;
        ierr = DMStagVecGetValuesStencil(dmcoeff, coefflocal, 1, &point, &D2_Right); CHKERRQ(ierr);
        dx = coordx[i][inext]-coordx[i][iprev];
        ff[j][i][idx] += 2.0 * D2_Right* (bclist[ibc].val - xx)/dx/dx;
      }

      if ((j == 0) && (bclist[ibc].point.c == SD3_DOF_P)) { 
        point.i = i; point.j = j; point.loc = DMSTAG_DOWN; point.c = 1;
        ierr = DMStagVecGetValuesStencil(dmcoeff, coefflocal, 1, &point, &D2_Down); CHKERRQ(ierr);
        dz = coordz[j][inext]-coordz[j][iprev];
        ff[j][i][idx] += -2.0 * D2_Down* (xx - bclist[ibc].val)/dz/dz;
      }

      if ((j == Nz-1) && (bclist[ibc].point.c == SD3_DOF_P)) { 
        point.i = i; point.j = j; point.loc = DMSTAG_UP; point.c = 1;
        ierr = DMStagVecGetValuesStencil(dmcoeff, coefflocal, 1, &point, &D2_Up); CHKERRQ(ierr);
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
        point.i = i; point.j = j; point.loc = DMSTAG_LEFT; point.c = 1;
        ierr = DMStagVecGetValuesStencil(dmcoeff, coefflocal, 1, &point, &D2_Left); CHKERRQ(ierr);
        dx = coordx[i][inext]-coordx[i][iprev];
        ff[j][i][idx] += -D2_Left*bclist[ibc].val/dx;
      }

      if ((i == Nx-1) && (bclist[ibc].point.c == SD3_DOF_P)) { 
        point.i = i; point.j = j; point.loc = DMSTAG_RIGHT; point.c = 1;
        ierr = DMStagVecGetValuesStencil(dmcoeff, coefflocal, 1, &point, &D2_Right); CHKERRQ(ierr);
        dx = coordx[i][inext]-coordx[i][iprev];
        ff[j][i][idx] += D2_Right*bclist[ibc].val/dx;
      }

      if ((j == 0) && (bclist[ibc].point.c == SD3_DOF_P)) { 
        point.i = i; point.j = j; point.loc = DMSTAG_DOWN; point.c = 1;
        ierr = DMStagVecGetValuesStencil(dmcoeff, coefflocal, 1, &point, &D2_Down); CHKERRQ(ierr);
        dz = coordz[j][inext]-coordz[j][iprev];
        ff[j][i][idx] += -D2_Down*bclist[ibc].val/dz;
      }

      if ((j == Nz-1) && (bclist[ibc].point.c == SD3_DOF_P)) { 
        point.i = i; point.j = j; point.loc = DMSTAG_UP; point.c = 1;
        ierr = DMStagVecGetValuesStencil(dmcoeff, coefflocal, 1, &point, &D2_Up); CHKERRQ(ierr);
        dz = coordz[j][inext]-coordz[j][iprev];
        ff[j][i][idx] += D2_Up*bclist[ibc].val/dz;
      }

      if (bclist[ibc].point.c == SD3_DOF_PC) {
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"BC type NEUMANN for FDPDE_STOKESDARCY3FIELD [PC-ELEMENT] is not yet implemented.");
      } 
    }
  }
  PetscFunctionReturn(0);
}