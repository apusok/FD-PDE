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

// ---------------------------------------
// FDBCListDestroy
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDBCListDestroy"
PetscErrorCode FDBCListDestroy(BCList **_list)
{
  BCList *list;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (!_list) PetscFunctionReturn(0);
  list = *_list;
  ierr = PetscFree(list);CHKERRQ(ierr);
  *_list = NULL;
  PetscFunctionReturn(0);
}

// ---------------------------------------
// Get BC Entry details
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDBCGetEntry"
PetscErrorCode FDBCGetEntry(DM dm,PetscScalar **cx,PetscScalar **cz, DMStagStencilLocation loc, PetscInt c, PetscInt i, PetscInt j, BCList *list)
{
  PetscInt       ii = 0, jj = 0, iprev=-1, inext=-1, icenter=-1;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  if (loc == DMSTAG_ELEMENT){ ierr = DMStagGet1dCoordinateLocationSlot(dm,DMSTAG_ELEMENT,&icenter);CHKERRQ(ierr); }
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
