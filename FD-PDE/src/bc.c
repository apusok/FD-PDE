#include "bc.h"

// ---------------------------------------
// FDBCListCreate
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDBCListCreate"
PetscErrorCode FDBCListCreate(DM dm, BCList **_list, PetscInt *_ndof)
{
  PetscInt Nx, Nz, sx, sz, nx, nz;
  PetscInt i, j, ii, idof, ndof, dof0, dof1, dof2;
  DMStagStencilLocation loc, loc1;
  BCList *list = NULL;
  PetscScalar    **coordx,**coordz;

  PetscErrorCode ierr;
  PetscFunctionBegin;

  // Count local boundary dofs
  ierr = DMStagGetGlobalSizes(dm, &Nx, &Nz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = DMStagGetDOF(dm,&dof0,&dof1,&dof2,NULL);CHKERRQ(ierr);

  ndof = 0; 
  // Count boundary dofs
  for (j = sz; j<sz+nz; j++) {
    for (i = sx; i<sx+nx; i++) {
      if ((i==0) || (i==Nx-1)) {
        if (dof0) { ndof+=  dof0; }// corner
        if (dof1) { ndof+=2*dof1; }// left/right, down/up
        if (dof2) { ndof+=  dof2; }// element
        if (j==Nz-1) {
          if (dof0) { ndof+=dof0; }// extra corner
          if (dof1) { ndof+=dof1; }// left/right
        }
      } else {
        if ((j==0) || (j==Nz-1)) {
          if (dof0) { ndof+=  dof0; }// corner
          if (dof1) { ndof+=2*dof1; }// left/right, down/up
          if (dof2) { ndof+=  dof2; }// element
          if (i==Nx-2) {
            if (dof0) { ndof+=dof0; }// down-right
            if (dof1) { ndof+=dof1; }// left/right
          }
        } 
      }
    }
  }

  // Return empty list
  if (!ndof) {
    *_list = list;
    *_ndof = ndof;
    PetscFunctionReturn(0);
  }

  // Allocate memory to list
  ierr = PetscMalloc((size_t)ndof*sizeof(BCList),&list);CHKERRQ(ierr);
  ierr = PetscMemzero(list,(size_t)ndof*sizeof(BCList));CHKERRQ(ierr);

  // Get dm coordinates array
  ierr = DMStagGet1dCoordinateArraysDOFRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);

  // Save grid info for BC dofs
  idof = 0;
  for (j = sz; j<sz+nz; j++) {
    for (i = sx; i<sx+nx; i++) {
      if ((i==0) || (i==Nx-1)) {
        if (dof0) {
          if (i==0) { loc = DMSTAG_DOWN_LEFT; }
          else      { loc = DMSTAG_DOWN_RIGHT;}
          for (ii = 0; ii<dof0; ii++) {
            ierr = FDBCGetEntry(dm,coordx,coordz,loc,ii,i,j,&list[idof]);CHKERRQ(ierr);
            idof++;
          }
        }
        if (dof1) { // 2 DOFs - LEFT, DOWN
          if (i==0) { loc = DMSTAG_LEFT; }
          else      { loc = DMSTAG_RIGHT;}
          loc1 = DMSTAG_DOWN;
          for (ii = 0; ii<dof1; ii++) {
            ierr = FDBCGetEntry(dm,coordx,coordz,loc,ii,i,j,&list[idof]);CHKERRQ(ierr); // Vx
            idof++;
            ierr = FDBCGetEntry(dm,coordx,coordz,loc1,ii,i,j,&list[idof]);CHKERRQ(ierr); // Vz
            idof++;
          }
        }
        if (dof2) {
          for (ii = 0; ii<dof2; ii++) {
            ierr = FDBCGetEntry(dm,coordx,coordz,DMSTAG_ELEMENT,ii,i,j,&list[idof]);CHKERRQ(ierr);
            idof++;
          }
        }
        if (j == Nz-1) {
          if (dof0) {
            if (i==0) { loc = DMSTAG_UP_LEFT; }
            else      { loc = DMSTAG_UP_RIGHT;}
            for (ii = 0; ii<dof0; ii++) {
              ierr = FDBCGetEntry(dm,coordx,coordz,loc,ii,i,j,&list[idof]);CHKERRQ(ierr);
              idof++;
            } 
          }
          if (dof1) {
            for (ii = 0; ii<dof1; ii++) {
              ierr = FDBCGetEntry(dm,coordx,coordz,DMSTAG_UP,ii,i,j,&list[idof]);CHKERRQ(ierr);
              idof++;
            } 
          }
        }
      } else {
        if ((j==0) || (j==Nz-1)) {
          if (dof0) {
            if (j==0) { loc = DMSTAG_DOWN_LEFT;}
            else      { loc = DMSTAG_UP_LEFT;  }
            for (ii = 0; ii<dof0; ii++) {
              ierr = FDBCGetEntry(dm,coordx,coordz,loc,ii,i,j,&list[idof]);CHKERRQ(ierr);
              idof++;
            }
          }
          if (dof1) { // 2 DOFs - LEFT, DOWN
            if (j==0) { loc = DMSTAG_DOWN;}
            else      { loc = DMSTAG_UP;  }
            loc1 = DMSTAG_LEFT;
            for (ii = 0; ii<dof1; ii++) {
              ierr = FDBCGetEntry(dm,coordx,coordz,loc,ii,i,j,&list[idof]);CHKERRQ(ierr); // Vz
              idof++;
              ierr = FDBCGetEntry(dm,coordx,coordz,loc1,ii,i,j,&list[idof]);CHKERRQ(ierr); // Vx
              idof++;
            }
          }
          if (dof2) {
            for (ii = 0; ii<dof2; ii++) {
              ierr = FDBCGetEntry(dm,coordx,coordz,DMSTAG_ELEMENT,ii,i,j,&list[idof]);CHKERRQ(ierr);
              idof++;
            }
          }
          if (i == Nx-2) {
            if (dof0) {
              if (j==0) { loc = DMSTAG_DOWN_RIGHT;}
              else      { loc = DMSTAG_UP_RIGHT;  }
              for (ii = 0; ii<dof0; ii++) {
                ierr = FDBCGetEntry(dm,coordx,coordz,loc,ii,i,j,&list[idof]);CHKERRQ(ierr);
                idof++;
              }
            }
            if (dof1) {
              for (ii = 0; ii<dof1; ii++) {
                ierr = FDBCGetEntry(dm,coordx,coordz,DMSTAG_RIGHT,ii,i,j,&list[idof]);CHKERRQ(ierr);
                idof++;
              } 
            }
          }
        } 
      }
    }
  }

  // Restore arrays, local vectors
  ierr = DMStagRestore1dCoordinateArraysDOFRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);

  // Initialize list type
  for (ii = 0; ii<ndof; ii++) {
    list[ii].type = BC_UNINIT;
    list[ii].val  = 0.0;
  }

  *_list = list;
  *_ndof = ndof;
  PetscFunctionReturn(0);
}

// // ---------------------------------------
// // FDBCListDestroy
// // ---------------------------------------
// #undef __FUNCT__
// #define __FUNCT__ "FDBCListDestroy"
// PetscErrorCode FDBCListDestroy(BCList *list)
// {
//   PetscErrorCode ierr;
//   PetscFunctionBegin;

//   ierr = PetscFree(list);CHKERRQ(ierr);
//   PetscFunctionReturn(0);
// }

// ---------------------------------------
// Get BC Entry details
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDBCGetEntry"
PetscErrorCode FDBCGetEntry(DM dm,PetscScalar **cx,PetscScalar **cz, DMStagStencilLocation loc, PetscInt c, PetscInt i, PetscInt j, BCList *list)
{
  PetscInt       ii = 0, jj = 0, iprev, inext, icenter;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  ierr = DMStagGet1dCoordinateLocationSlot(dm,DMSTAG_ELEMENT,&icenter);CHKERRQ(ierr); 
  ierr = DMStagGet1dCoordinateLocationSlot(dm,DMSTAG_LEFT,&iprev);CHKERRQ(ierr);
  ierr = DMStagGet1dCoordinateLocationSlot(dm,DMSTAG_RIGHT,&inext);CHKERRQ(ierr); 

  list->point.i   = i;
  list->point.j   = j;
  list->point.c   = c;
  list->point.loc = loc;

  ierr = DMStagGetLocationSlot(dm,loc,c,&list->idx); CHKERRQ(ierr);

  if ((loc == DMSTAG_DOWN_LEFT)  || (loc == DMSTAG_UP_LEFT)  || (loc == DMSTAG_LEFT))  ii = iprev;
  if ((loc == DMSTAG_ELEMENT)    || (loc == DMSTAG_DOWN)     || (loc == DMSTAG_UP))    ii = icenter;
  if ((loc == DMSTAG_DOWN_RIGHT) || (loc == DMSTAG_UP_RIGHT) || (loc == DMSTAG_RIGHT)) ii = inext;

  if ((loc == DMSTAG_DOWN_LEFT) || (loc == DMSTAG_DOWN) || (loc == DMSTAG_DOWN_RIGHT)) jj = iprev;
  if ((loc == DMSTAG_ELEMENT)   || (loc == DMSTAG_LEFT) || (loc == DMSTAG_RIGHT))      jj = icenter;
  if ((loc == DMSTAG_UP_LEFT)   || (loc == DMSTAG_UP)   || (loc == DMSTAG_UP_RIGHT))   jj = inext;

  list->coord[0] = cx[i][ii];
  list->coord[1] = cz[j][jj];

  PetscFunctionReturn(0);
}

// ---------------------------------------
// FDBCApplyStokes
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDBCApplyStokes"
PetscErrorCode FDBCApplyStokes(DM dm, Vec xlocal, BCList *bclist, PetscInt nbc, PetscScalar **coordx, PetscScalar **coordz, PetscScalar *eta_n, PetscScalar *eta_c,PetscInt n[], PetscScalar ***ff)
{
  PetscScalar    xx, dx, dz;
  PetscScalar    etaLeft, etaRight, etaUp, etaDown;
  PetscInt       i, j, ibc, idx, iprev, inext;
  PetscInt       sx, sz, nz, Nx, Nz;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  // DM domain info
  sx = n[0]; sz = n[1]; 
  nz = n[3];
  Nx = n[4]; Nz = n[5];
  iprev = n[7]; inext = n[8];

  // Loop over all boundaries
  for (ibc = 0; ibc<nbc; ibc++) {
    if (bclist[ibc].type == DIRICHLET) {
      i   = bclist[ibc].point.i;
      j   = bclist[ibc].point.j;
      idx = bclist[ibc].idx;

      // Get residual value
      ierr = DMStagVecGetValuesStencil(dm, xlocal, 1, &bclist[ibc].point, &xx); CHKERRQ(ierr);
      ff[j][i][idx] = xx - bclist[ibc].val;
    }

    if (bclist[ibc].type == NEUMANN) {
      i   = bclist[ibc].point.i;
      j   = bclist[ibc].point.j;
      idx = bclist[ibc].idx;

      // Stokes flow - add flux terms
      if ((j == 0) && (bclist[ibc].point.loc == DMSTAG_LEFT)) { // Vx down
        etaDown = eta_n[i-sx+(j-sz)*nz];
        dz = coordz[j][inext]-coordz[j][iprev];
        ff[j][i][idx] += -2.0*etaDown*bclist[ibc].val/dz;
      }

      if ((j == 0) && (bclist[ibc].point.loc == DMSTAG_RIGHT)) { // Vx down
        etaDown = eta_n[i+1-sx+(j-sz)*nz];
        dz = coordz[j][inext]-coordz[j][iprev];
        ff[j][i][idx] += -2.0*etaDown*bclist[ibc].val/dz;
      }

      if ((j == Nz-1) && (bclist[ibc].point.loc == DMSTAG_LEFT)) { // Vx up
        etaUp = eta_n[i-sx+(j+1-sz)*nz];
        dz = coordz[j][inext]-coordz[j][iprev];
        ff[j][i][idx] += 2.0*etaUp*bclist[ibc].val/dz;
      }

      if ((j == Nz-1) && (bclist[ibc].point.loc == DMSTAG_RIGHT)) { // Vx up
        etaUp = eta_n[i+1-sx+(j+1-sz)*nz];
        dz = coordz[j][inext]-coordz[j][iprev];
        ff[j][i][idx] += 2.0*etaUp*bclist[ibc].val/dz;
      }

      if ((i == 0) && (bclist[ibc].point.loc == DMSTAG_DOWN)) { // Vz left
        etaLeft = eta_n[i-sx+(j-sz)*nz];
        dx = coordx[i][inext]-coordx[i][iprev];
        ff[j][i][idx] += -2.0*etaLeft*bclist[ibc].val/dx;
      }

      if ((i == Nx-1) && (bclist[ibc].point.loc == DMSTAG_DOWN)) { // Vz right
        etaRight = eta_n[i+1-sx+(j-sz)*nz];
        dx = coordx[i][inext]-coordx[i][iprev];
        ff[j][i][idx] += 2.0*etaRight*bclist[ibc].val/dx;
      }
    }
  }
  PetscFunctionReturn(0);
}