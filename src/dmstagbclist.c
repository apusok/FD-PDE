#include "dmstagbclist.h"

// ---------------------------------------
/*@
_DMStagBCFillIndices

Use: internal
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "_DMStagBCFillIndices"
static PetscErrorCode _DMStagBCFillIndices(DM dm,DMStagStencilLocation loc,PetscInt c,PetscInt i,PetscInt j,DMStagBC *bc)
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  bc->point.i = i;
  bc->point.j = j;
  bc->point.c = c;
  bc->point.loc = loc;
  ierr = DMStagGetLocationSlot(dm,loc,c,&bc->idx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
_DMStagBCFillCoords

Use: internal
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "_DMStagBCFillCoords"
static PetscErrorCode _DMStagBCFillCoords(PetscInt stratum_id,DM dm,PetscScalar **cx,PetscScalar **cz,DMStagStencilLocation loc,PetscInt i, PetscInt j,DMStagBC *bc)
{
  PetscInt       ii=-1,jj=-1,iprev=-1,inext=-1,icenter=-1;
  PetscInt       dof0,dof1,dof2;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  
  ierr = DMStagGetDOF(dm,&dof0,&dof1,&dof2,NULL);CHKERRQ(ierr);

  if (dof2) {ierr = DMStagGetProductCoordinateLocationSlot(dm,DMSTAG_ELEMENT,&icenter);CHKERRQ(ierr);} 
  if (dof0 || dof1) { 
    ierr = DMStagGetProductCoordinateLocationSlot(dm,DMSTAG_LEFT,&iprev);CHKERRQ(ierr);
    ierr = DMStagGetProductCoordinateLocationSlot(dm,DMSTAG_RIGHT,&inext);CHKERRQ(ierr);
  } 

  // ierr = DMStagGetProductCoordinateLocationSlot(dm,DMSTAG_ELEMENT,&icenter);CHKERRQ(ierr);
  // ierr = DMStagGetProductCoordinateLocationSlot(dm,DMSTAG_LEFT,&iprev);CHKERRQ(ierr);
  // ierr = DMStagGetProductCoordinateLocationSlot(dm,DMSTAG_RIGHT,&inext);CHKERRQ(ierr);

  if ((loc == DMSTAG_DOWN_LEFT)  || (loc == DMSTAG_UP_LEFT)  || (loc == DMSTAG_LEFT))  ii = iprev;
  if ((loc == DMSTAG_ELEMENT)    || (loc == DMSTAG_DOWN)     || (loc == DMSTAG_UP))    ii = icenter;
  if ((loc == DMSTAG_DOWN_RIGHT) || (loc == DMSTAG_UP_RIGHT) || (loc == DMSTAG_RIGHT)) ii = inext;
  
  if ((loc == DMSTAG_DOWN_LEFT) || (loc == DMSTAG_DOWN) || (loc == DMSTAG_DOWN_RIGHT)) jj = iprev;
  if ((loc == DMSTAG_ELEMENT)   || (loc == DMSTAG_LEFT) || (loc == DMSTAG_RIGHT))      jj = icenter;
  if ((loc == DMSTAG_UP_LEFT)   || (loc == DMSTAG_UP)   || (loc == DMSTAG_UP_RIGHT))   jj = inext;
  
  bc->coord[0] = cx[i][ii];
  bc->coord[1] = cz[j][jj];

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
_DMStagBCListSetupIndices

Use: internal
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "_DMStagBCListSetupIndices"
static PetscErrorCode _DMStagBCListSetupIndices(DMStagBCList list)
{
  PetscInt              Nx,Nz,sx,sz,nx,nz;
  PetscInt              i,j,ii,ndof0,ndof1,ndof2,dof0,dof1,dof2;
  DMStagStencilLocation loc,loc1;
  DM                    dm;
  PetscErrorCode        ierr;
  
  PetscFunctionBegin;
  dm = list->dm;
  ierr = DMStagGetGlobalSizes(dm,&Nx,&Nz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dm,&sx,&sz,NULL,&nx,&nz,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = DMStagGetDOF(dm,&dof0,&dof1,&dof2,NULL);CHKERRQ(ierr);
  
  // Save grid info for BC dofs
  ndof0 = ndof1 = ndof2 = 0;
  for (j = sz; j<sz+nz; j++) {
    for (i = sx; i<sx+nx; i++) {
      if ((i == 0) || (i == Nx-1)) {
          if (i == 0) { loc = DMSTAG_DOWN_LEFT; }
          else        { loc = DMSTAG_DOWN_RIGHT; }
          loc = (i == 0) ? DMSTAG_DOWN_LEFT : DMSTAG_DOWN_RIGHT;
          for (ii = 0; ii<dof0; ii++) {
            ierr = _DMStagBCFillIndices(dm,loc,ii,i,j,&list->bc_v[ndof0++]);CHKERRQ(ierr);
          }
          if (i == 0) { loc = DMSTAG_LEFT; }
          else        { loc = DMSTAG_RIGHT; }
          loc = (i == 0) ? DMSTAG_LEFT : DMSTAG_RIGHT;
          loc1 = DMSTAG_DOWN;
          for (ii = 0; ii<dof1; ii++) {
            ierr = _DMStagBCFillIndices(dm,loc, ii,i,j,&list->bc_f[ndof1++]);CHKERRQ(ierr);
            ierr = _DMStagBCFillIndices(dm,loc1,ii,i,j,&list->bc_f[ndof1++]);CHKERRQ(ierr);
          }
          for (ii = 0; ii<dof2; ii++) {
            ierr = _DMStagBCFillIndices(dm,DMSTAG_ELEMENT,ii,i,j,&list->bc_e[ndof2++]);CHKERRQ(ierr);
          }
        if (j == Nz-1) {
            if (i == 0) { loc = DMSTAG_UP_LEFT; }
            else        { loc = DMSTAG_UP_RIGHT; }
            loc = (i == 0) ? DMSTAG_UP_LEFT : DMSTAG_UP_RIGHT;
            for (ii = 0; ii<dof0; ii++) {
              ierr = _DMStagBCFillIndices(dm,loc,ii,i,j,&list->bc_v[ndof0++]);CHKERRQ(ierr);
            }
            for (ii = 0; ii<dof1; ii++) {
              ierr = _DMStagBCFillIndices(dm,DMSTAG_UP,ii,i,j,&list->bc_f[ndof1++]);CHKERRQ(ierr);
            }
        }
      } else {
        if ((j == 0) || (j == Nz-1)) {
            if (j == 0) { loc = DMSTAG_DOWN_LEFT; }
            else        { loc = DMSTAG_UP_LEFT; }
            loc = (j == 0) ? DMSTAG_DOWN_LEFT : DMSTAG_UP_LEFT;
            for (ii = 0; ii<dof0; ii++) {
              ierr = _DMStagBCFillIndices(dm,loc,ii,i,j,&list->bc_v[ndof0++]);CHKERRQ(ierr);
            }
            if (j == 0) { loc = DMSTAG_DOWN; }
            else        { loc = DMSTAG_UP; }
            loc = (j == 0) ? DMSTAG_DOWN : DMSTAG_UP;
            loc1 = DMSTAG_LEFT;
            for (ii = 0; ii<dof1; ii++) {
              ierr = _DMStagBCFillIndices(dm,loc, ii,i,j,&list->bc_f[ndof1++]);CHKERRQ(ierr);
              ierr = _DMStagBCFillIndices(dm,loc1,ii,i,j,&list->bc_f[ndof1++]);CHKERRQ(ierr);
            }
            for (ii = 0; ii<dof2; ii++) {
              ierr = _DMStagBCFillIndices(dm,DMSTAG_ELEMENT,ii,i,j,&list->bc_e[ndof2++]);CHKERRQ(ierr);
            }
          if (i == Nx-2) {
              if (j == 0) { loc = DMSTAG_DOWN_RIGHT; }
              else        { loc = DMSTAG_UP_RIGHT; }
              loc = (j == 0) ? DMSTAG_DOWN_RIGHT : DMSTAG_UP_RIGHT;
              for (ii = 0; ii<dof0; ii++) {
                ierr = _DMStagBCFillIndices(dm,loc,ii,i,j,&list->bc_v[ndof0++]);CHKERRQ(ierr);
              }
              for (ii = 0; ii<dof1; ii++) {
                ierr = _DMStagBCFillIndices(dm,DMSTAG_RIGHT,ii,i,j,&list->bc_f[ndof1++]);CHKERRQ(ierr);
              }
          }
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
_DMStagBCListSetupCoordinates

Use: internal
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "_DMStagBCListSetupCoordinates"
static PetscErrorCode _DMStagBCListSetupCoordinates(DMStagBCList list)
{
  PetscInt       ii,k;
  DM             dm,dmCoord;
  PetscBool      isProduct;
  DMType         dmType;
  PetscScalar    **coordx,**coordz;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  dm = list->dm;
  ierr = DMGetCoordinateDM(dm,&dmCoord);CHKERRQ(ierr);
  ierr = DMGetType(dmCoord,&dmType);CHKERRQ(ierr);
  ierr = PetscStrcmp(DMPRODUCT,dmType,&isProduct);CHKERRQ(ierr);
  if (!isProduct) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Coordinate DM must be of type DMPRODUCT");
  ierr = DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  {
    DMStagBC *bcs[] = { list->bc_v , list->bc_f, list->bc_e };
    PetscInt size[] = { list->nbc_vertex , list->nbc_face , list->nbc_element };
    
    for (k=0; k<3; k++) {
      for (ii = 0; ii<size[k]; ii++) {
        DMStagBC *b = &bcs[k][ii];
        ierr = _DMStagBCFillCoords(k,dm,coordx,coordz,b->point.loc,b->point.i,b->point.j,b);CHKERRQ(ierr);
      }
    }
  }
  ierr = DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
DMStagBCListSetupCoordinates - set coordinates for a DMStagBCList object

Input Parameter:
list - the DMStagBCList object

Use: user/internal
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "DMStagBCListSetupCoordinates"
PetscErrorCode DMStagBCListSetupCoordinates(DMStagBCList list)
{
  DM             dmCoord = NULL;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = DMGetCoordinateDM(list->dm,&dmCoord);
  if (!dmCoord) SETERRQ(PetscObjectComm((PetscObject)list->dm),PETSC_ERR_SUP,"DMStag must have coordinates defined. Hint - coordinates must be defined via DMPRODUCT)");
  ierr = _DMStagBCListSetupCoordinates(list);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
DMStagBCListGetVertexBCs - return a 1-D DMStagBC array containing the vertex (corner) boundary dofs

Input Parameter:
list - the DMStagBCList object

Output Parameters:
nbc - count of vertex dofs
l - 1-D DMStagBC array

Use: user/internal
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "DMStagBCListGetVertexBCs"
PetscErrorCode DMStagBCListGetVertexBCs(DMStagBCList list,PetscInt *nbc,DMStagBC *l[])
{
  PetscFunctionBegin;
  if (nbc) { *nbc = list->nbc_vertex; }
  if (l) { *l = list->bc_v; }
  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
DMStagBCListGetFaceBCs - return a 1-D DMStagBC array containing the face (edge) boundary dofs

Input Parameter:
list - the DMStagBCList object

Output Parameters:
nbc - count of edge dofs
l - 1-D DMStagBC array

Use: user/internal
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "DMStagBCListGetFaceBCs"
PetscErrorCode DMStagBCListGetFaceBCs(DMStagBCList list,PetscInt *nbc,DMStagBC *l[])
{
  PetscFunctionBegin;
  if (nbc) { *nbc = list->nbc_face; }
  if (l) { *l = list->bc_f; }
  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
DMStagBCListGetElementBCs - return a 1-D DMStagBC array containing the element (center) boundary dofs

Input Parameter:
list - the DMStagBCList object

Output Parameters:
nbc - count of element dofs
l - 1-D DMStagBC array

Use: user/internal
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "DMStagBCListGetElementBCs"
PetscErrorCode DMStagBCListGetElementBCs(DMStagBCList list,PetscInt *nbc,DMStagBC *l[])
{
  PetscFunctionBegin;
  if (nbc) { *nbc = list->nbc_element; }
  if (l) { *l = list->bc_e; }
  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
DMStagBCListCreate - creates the boundary conditions (BC) data structure for a DM

Input Parameter:
dm - a DM object

Output Parameter:
list - the DMStagBCList object

Use: internal
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "DMStagBCListCreate"
PetscErrorCode DMStagBCListCreate(DM dm,DMStagBCList *list)
{
  PetscErrorCode ierr;
  PetscInt       dim,Nx,Nz;
  PetscBool      isstag;
  DMStagBCList   l;
  
  PetscFunctionBegin;
  /* check assumptions */
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  if (dim != 2) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Only valid for 2d DM");
  ierr = PetscObjectTypeCompare((PetscObject)dm,DMSTAG,&isstag);CHKERRQ(ierr);
  if (!isstag) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Only valid for DMStag");
  ierr = DMStagGetGlobalSizes(dm,&Nx,&Nz,NULL);CHKERRQ(ierr);
  if (Nx < 3) SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Implementation of BCList only valid for n-cells-i >= 3. Found nx=%D",Nx);
  if (Nz < 3) SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Implementation of BCList only valid for n-cells-j >= 3. Found ny=%D",Nz);
  { /* Check for existence of a coordinateDM - I deliberately don't call CHKERRQ() after DMGetCoordinateDM() so that I can throw the error message below which is more helpful than the error thrown from DMGetCoordinateDM() */
    DM dmCoord = NULL;
    ierr = DMGetCoordinateDM(dm,&dmCoord);
    if (!dmCoord) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"DMStag must have coordinates defined. Hint - coordinates must be defined via DMPRODUCT");
  }
  
  ierr = PetscCalloc1(1,&l);CHKERRQ(ierr);
  l->dm = dm;
  l->evaluate = NULL;
  l->data = NULL;
  
  {
    PetscInt sx,sz,nx,nz;
    PetscInt ii,i,j,ndof0,ndof1,ndof2,dof0,dof1,dof2;
    
    // Count local boundary dofs
    ierr = DMStagGetCorners(dm,&sx,&sz,NULL,&nx,&nz,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
    ierr = DMStagGetDOF(dm,&dof0,&dof1,&dof2,NULL);CHKERRQ(ierr);
    
    // Count boundary dofs
    ndof0 = 0;
    ndof1 = 0;
    ndof2 = 0;
    for (j = sz; j<sz+nz; j++) {
      for (i = sx; i<sx+nx; i++) {
        if ((i == 0) || (i == Nx-1)) {
          ndof0 +=   dof0; // corner
          ndof1 += 2*dof1; // left/right, down/up
          ndof2 +=   dof2; // element
          if (j == (Nz-1)) {
            ndof0 += dof0; // extra corner
            ndof1 += dof1; // left/right
          }
        } else {
          if ((j == 0) || (j == Nz-1)) {
            ndof0 +=   dof0; // corner
            ndof1 += 2*dof1; // left/right, down/up
            ndof2 +=   dof2; // element
            if (i == (Nx-2)) {
              ndof0 += dof0; // down-right
              ndof1 += dof1; // left/right
            }
          }
        }
      }
    }
    
    l->nbc         = ndof0 + ndof1 + ndof2;
    l->nbc_vertex  = ndof0;
    l->nbc_face    = ndof1;
    l->nbc_element = ndof2;
    
    /* if we care about allocating empty arrays (although its permitted by petsc), add one extra */
    ierr = PetscCalloc1(l->nbc_vertex+1, &l->bc_v);CHKERRQ(ierr);
    ierr = PetscCalloc1(l->nbc_face+1,   &l->bc_f);CHKERRQ(ierr);
    ierr = PetscCalloc1(l->nbc_element+1,&l->bc_e);CHKERRQ(ierr);
    
    ierr = _DMStagBCListSetupIndices(l);CHKERRQ(ierr);
    
    ierr = _DMStagBCListSetupCoordinates(l);CHKERRQ(ierr);
    
    // Initialize list type
    for (ii = 0; ii<l->nbc_vertex; ii++) {
      l->bc_v[ii].type = BC_NULL;
      l->bc_v[ii].val  = 0.0;
    }
    
    for (ii = 0; ii<l->nbc_face; ii++) {
      l->bc_f[ii].type = BC_NULL;
      l->bc_f[ii].val  = 0.0;
    }
    
    for (ii = 0; ii<l->nbc_element; ii++) {
      l->bc_e[ii].type = BC_NULL;
      l->bc_e[ii].val  = 0.0;
    }
  }
  
  *list = l;
  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
DMStagBCListDestroy - destroy the boundary conditions (BC) data structure for a DM

Input Parameter:
list - the DMStagBCList object

Use: internal
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "DMStagBCListDestroy"
PetscErrorCode DMStagBCListDestroy(DMStagBCList *list)
{
  PetscErrorCode ierr;
  DMStagBCList   l;
  PetscFunctionBegin;
  if (!list) PetscFunctionReturn(0);
  if (!*list) PetscFunctionReturn(0);
  l = *list;
  ierr = PetscFree(l->bc_v);CHKERRQ(ierr);
  ierr = PetscFree(l->bc_f);CHKERRQ(ierr);
  ierr = PetscFree(l->bc_e);CHKERRQ(ierr);
  ierr = PetscFree(l);CHKERRQ(ierr);
  *list = NULL;
  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
_DMStagBCListGetIndices_J_left_and_right

Use: internal
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "_DMStagBCListGetIndices_J_left_and_right"
PetscErrorCode _DMStagBCListGetIndices_J_left_and_right(DMStagBCList list,PetscInt dof,PetscInt J,PetscInt *n,PetscInt *_idx[])
{
  PetscInt k,Nx,Nz,count=0;
  PetscInt *idx;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = DMStagGetGlobalSizes(list->dm,&Nx,&Nz,NULL);CHKERRQ(ierr);
  ierr = PetscCalloc1(2*Nx,&idx);CHKERRQ(ierr); /* over allocate */
  for (k=0; k<list->nbc_face; k++) {
    if (list->bc_f[k].point.j == J) {
      DMStagStencilLocation loc = list->bc_f[k].point.loc;
      if ((loc == DMSTAG_LEFT && list->bc_f[k].point.i != 0 )|| (loc == DMSTAG_RIGHT && list->bc_f[k].point.i != Nx-1)) {
        if (list->bc_f[k].point.c == dof) {
          idx[count++] = k;
        }
      }
    }
  }
  *n = count;
  *_idx = idx;
  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
_DMStagBCListGetIndices_J_updown

Use: internal
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "_DMStagBCListGetIndices_J_updown"
PetscErrorCode _DMStagBCListGetIndices_J_updown(DMStagBCList list,PetscInt dof,PetscInt J,DMStagStencilLocation dir,PetscInt *n,PetscInt *_idx[])
{
  PetscInt k,Nx,Nz,count=0;
  PetscInt *idx;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  if (dir != DMSTAG_UP && dir != DMSTAG_DOWN) SETERRQ(PetscObjectComm((PetscObject)list->dm),PETSC_ERR_SUP,"Direction (arg 3) must be either DMSTAG_UP or DMSTAG_DOWN");
  
  ierr = DMStagGetGlobalSizes(list->dm,&Nx,&Nz,NULL);CHKERRQ(ierr);
  ierr = PetscCalloc1(2*Nx,&idx);CHKERRQ(ierr); /* over allocate */
  for (k=0; k<list->nbc_face; k++) {
    if (list->bc_f[k].point.j == J) {
      DMStagStencilLocation loc = list->bc_f[k].point.loc;
      if ((loc == dir) && (list->bc_f[k].point.c == dof)) {
        idx[count++] = k;
      }
    }
  }
  *n = count;
  *_idx = idx;
  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
_DMStagBCListGetIndices_I_up_and_down

Use: internal
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "_DMStagBCListGetIndices_I_up_and_down"
PetscErrorCode _DMStagBCListGetIndices_I_up_and_down(DMStagBCList list,PetscInt dof,PetscInt II,PetscInt *n,PetscInt *_idx[])
{
  PetscInt k,Nx,Nz,count=0;
  PetscInt *idx;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = DMStagGetGlobalSizes(list->dm,&Nx,&Nz,NULL);CHKERRQ(ierr);
  ierr = PetscCalloc1(2*Nz,&idx);CHKERRQ(ierr); /* over allocate */
  for (k=0; k<list->nbc_face; k++) {
    if (list->bc_f[k].point.i == II) {
      DMStagStencilLocation loc = list->bc_f[k].point.loc;
      if ((loc == DMSTAG_UP && list->bc_f[k].point.j != Nz-1) || (loc == DMSTAG_DOWN && list->bc_f[k].point.j != 0)) {
        if (list->bc_f[k].point.c == dof) {
          idx[count++] = k;
        }
      }
    }
  }
  *n = count;
  *_idx = idx;
  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
_DMStagBCListGetIndices_I_leftright

Use: internal
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "_DMStagBCListGetIndices_I_leftright"
PetscErrorCode _DMStagBCListGetIndices_I_leftright(DMStagBCList list,PetscInt dof,PetscInt II,DMStagStencilLocation dir,PetscInt *n,PetscInt *_idx[])
{
  PetscInt k,Nx,Nz,count=0;
  PetscInt *idx;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  if (dir != DMSTAG_LEFT && dir != DMSTAG_RIGHT) SETERRQ(PetscObjectComm((PetscObject)list->dm),PETSC_ERR_SUP,"Direction (arg 3) must be either DMSTAG_LEFT or DMSTAG_RIGHT");
  
  ierr = DMStagGetGlobalSizes(list->dm,&Nx,&Nz,NULL);CHKERRQ(ierr);
  ierr = PetscCalloc1(2*Nz,&idx);CHKERRQ(ierr); /* over allocate */
  for (k=0; k<list->nbc_face; k++) {
    if (list->bc_f[k].point.i == II) {
      DMStagStencilLocation loc = list->bc_f[k].point.loc;
      if ((loc == dir) && (list->bc_f[k].point.c == dof)) {
        idx[count++] = k;
      }
    }
  }
  *n = count;
  *_idx = idx;
  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
_DMStagBCListGetIndices_center

Use: internal
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "_DMStagBCListGetIndices_center"
PetscErrorCode _DMStagBCListGetIndices_center(DMStagBCList list,PetscInt dof,PetscInt II,PetscInt JJ,PetscInt *n,PetscInt *_idx[])
{
  PetscInt k,Nx,Nz,count=0;
  PetscInt *idx;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = DMStagGetGlobalSizes(list->dm,&Nx,&Nz,NULL);CHKERRQ(ierr);

  if (II == -1) {
    ierr = PetscCalloc1(Nx,&idx);CHKERRQ(ierr);
    for (k=0; k<list->nbc_element; k++) {
      if ((list->bc_e[k].point.j == JJ) && (list->bc_e[k].point.c == dof)) {
        DMStagStencilLocation loc = list->bc_e[k].point.loc;
        if (loc == DMSTAG_ELEMENT) {
          idx[count++] = k;
        }
      }
    }
  }
  if (JJ == -1) {
    ierr = PetscCalloc1(Nz,&idx);CHKERRQ(ierr); /* over allocate */
    for (k=0; k<list->nbc_element; k++) {
      if ((list->bc_e[k].point.i == II) && (list->bc_e[k].point.c == dof)) {
        DMStagStencilLocation loc = list->bc_e[k].point.loc;
        if (loc == DMSTAG_ELEMENT) {
          idx[count++] = k;
        }
      }
    }
  }
  *n = count;
  *_idx = idx;
  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
DMStagBCListGetValues - get BC entries associated with a boundary and a dof

Input Parameter:
list - the DMStagBCList object
domain_face - boundary label: 'w' west, 'e' east, 'n' north, 's' south
label - dof label: '.' vertex, '-' edge (horizontal), '|' edge (vertical), 'o' element
dof - component degree of freedom (DMStagStencil c) 

Output Parameters:
_n - count of boundary dof
_idx - 1D array containing the index
_xc - 1D array containing the coordinates (true boundary)
_xc_stag - 1D array containing the coordinates (dof)
_value - 1D array containing the value
_type - 1D array containing BCtype

Notes:
User should call DMStagBCListInsertValues() to update the entries in DMStagBCList list.

Use: user
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "DMStagBCListGetValues"
PetscErrorCode DMStagBCListGetValues(DMStagBCList list,
                  const char domain_face,const char label, /* vertex -> . : edge -> {-,|} : element -> o */
                  PetscInt dof,
                  PetscInt *_n,PetscInt *_idx[],PetscScalar *_xc[],PetscScalar *_xc_stag[],PetscScalar *_value[],BCType *_type[])
{
  PetscInt n=0,*idx,Nx,Nz,k;
  PetscScalar *v,*xc,*xc_stag;
  BCType *t;
  DMStagBC *bc = NULL;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  if (label == '.') SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support to get vertex bc DOFs. Only support for getting face bc values. ");
  
  if (!_n || !_idx || !_value || !_type) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must provide a valid (non-NULL) pointer for n (arg 4), idx (arg 5), value (arg 7), type (arg 8)");
  
  ierr = DMStagGetGlobalSizes(list->dm,&Nx,&Nz,NULL);CHKERRQ(ierr);
  
  switch (domain_face) {
    case 'n':
      if (label == '-') {
        ierr = _DMStagBCListGetIndices_J_left_and_right(list,dof,Nz-1,&n,&idx);CHKERRQ(ierr); // vx
      } else if (label == '|') {
        ierr = _DMStagBCListGetIndices_J_updown(list,dof,Nz-1,DMSTAG_UP,&n,&idx);CHKERRQ(ierr); // vy
      } else if (label == 'o') {
        _DMStagBCListGetIndices_center(list,dof,-1,Nz-1,&n,&idx);CHKERRQ(ierr);
      } else {
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Direction[North] No support to get vertex data. Unknown stratum label %s provided - must be one of {'-','|','o'}",label);
      }
      break;
      
    case 's':
      if (label == '-') {
        ierr = _DMStagBCListGetIndices_J_left_and_right(list,dof,0,&n,&idx);CHKERRQ(ierr);
      } else if (label == '|') {
        ierr = _DMStagBCListGetIndices_J_updown(list,dof,0,DMSTAG_DOWN,&n,&idx);CHKERRQ(ierr); // vy
      } else if (label == 'o') {
        _DMStagBCListGetIndices_center(list,dof,-1,0,&n,&idx);CHKERRQ(ierr);
      } else {
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Direction[South] No support to get vertex data. Unknown stratum label %s provided - must be one of {'-','|','o'}",label);
      }
      break;

    case 'e':
      if (label == '-') {
        ierr = _DMStagBCListGetIndices_I_leftright(list,dof,Nx-1,DMSTAG_RIGHT,&n,&idx);CHKERRQ(ierr); // vx
      } else if (label == '|') {
        ierr = _DMStagBCListGetIndices_I_up_and_down(list,dof,Nx-1,&n,&idx);CHKERRQ(ierr); // vy
      } else if (label == 'o') {
        _DMStagBCListGetIndices_center(list,dof,Nx-1,-1,&n,&idx);CHKERRQ(ierr);
      } else {
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Direction[East] No support to get vertex data. Unknown stratum label %s provided - must be one of {'-','|','o'}",label);
      }
      break;

    case 'w':
      if (label == '-') {
        ierr = _DMStagBCListGetIndices_I_leftright(list,dof,0,DMSTAG_LEFT,&n,&idx);CHKERRQ(ierr); // vx
      } else if (label == '|') {
        ierr = _DMStagBCListGetIndices_I_up_and_down(list,dof,0,&n,&idx);CHKERRQ(ierr); // vy
      } else if (label == 'o') {
        _DMStagBCListGetIndices_center(list,dof,0,-1,&n,&idx);CHKERRQ(ierr);
      } else {
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Direction[West] No support to get vertex data. Unknown stratum label %s provided - must be one of {'-','|','o'}",label);
      }
      break;
      
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Direction %s is not supported. Direction must be one of {'n','e','s','w'}");
      break;
  }

  ierr = PetscCalloc1(2*n,&xc);CHKERRQ(ierr);
  ierr = PetscCalloc1(2*n,&xc_stag);CHKERRQ(ierr);
  ierr = PetscCalloc1(n,&v);CHKERRQ(ierr);
  ierr = PetscCalloc1(n,&t);CHKERRQ(ierr);
  
  /* copy coords */
  switch (label) {
    case '.':
      bc = list->bc_v;
      break;
    case '-':
      bc = list->bc_f;
      break;
    case '|':
      bc = list->bc_f;
      break;
    case 'o':
      bc = list->bc_e;
      break;
    default:
      break;
  }

  // Load the size of half grids
  PetscScalar *dx, *dz;
  PetscInt start[2], nx_local, nz_local;
  
  ierr = DMStagCellSizeLocal_2d(list->dm,&nx_local,&nz_local,&dx,&dz); CHKERRQ(ierr);
  ierr = DMStagGetCorners(list->dm,&start[0],&start[1],NULL,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);

  for (k=0; k<n; k++) {
    xc_stag[2*k+0] = bc[ idx[k] ].coord[0];
    xc_stag[2*k+1] = bc[ idx[k] ].coord[1];
    v[k]      = bc[ idx[k] ].val;
    t[k]      = bc[ idx[k] ].type;

    xc[2*k+0] = xc_stag[2*k+0];
    xc[2*k+1] = xc_stag[2*k+1];

    //Correct coordinates of interior boundary points to the corresponding true boundaries
    //Notes: it corrects the output of xc for the convenience of prescribing the boundary conditions, but not change the data stored in BCList.
    if (domain_face == 'n' && (label == '-' || label == 'o')) {
      if (start[1]+nz_local == Nz) {xc[2*k+1] += 0.5*dz[nz_local-1];}
      else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"North Boundary: Wrong indices for cell sizes.");
    }

    if (domain_face == 's' && (label == '-' || label == 'o')) {
      if (start[1] == 0) {xc[2*k+1] -= 0.5*dz[start[1]];}
      else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"South Boundary: Wrong indices for cell sizes.");
    }

    if (domain_face == 'w' && (label == '|' || label == 'o')) {
      if (start[0] == 0) {xc[2*k+0] -= 0.5*dx[start[0]];}
      else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"West Boundary: Wrong indices for cell sizes.");
    }

    if (domain_face == 'e' && (label == '|' || label == 'o')) {
      if (start[0]+nx_local == Nx) {xc[2*k+0] += 0.5*dx[nx_local-1];}
      else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"East Boundary: Wrong indices for cell sizes.");
    }
  }

  ierr = PetscFree(dx);CHKERRQ(ierr);
  ierr = PetscFree(dz);CHKERRQ(ierr);

  *_n = n;  *_idx = idx;  *_value = v;  *_type = t;
  if (_xc) { *_xc = xc; }
  else {ierr = PetscFree(xc);CHKERRQ(ierr);}
  if (_xc_stag) { *_xc_stag = xc_stag; }
  else {ierr = PetscFree(xc_stag);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
convert_stencil_label_to_stratum_index_2d - returns a number associated with the DMStag dof
0 - vertex
1 - edge
2 - element

Use: internal
@*/
// ---------------------------------------
static PetscInt convert_stencil_label_to_stratum_index_2d(char l)
{
  if (l == '-' || l == '|') {
    return(1);
  }
  if (l == 'o') {
    return(2);
  }
  if (l == '.') {
    return(0);
  }
  return(-1);
}

// // ---------------------------------------
// /*@
// convert_stencil_location_to_stratum_index_2d - returns a number associated with the DMStagStencilLocation
// 0 - vertex (DMSTAG_DOWN_LEFT, DMSTAG_DOWN_RIGHT, DMSTAG_UP_LEFT, DMSTAG_UP_RIGHT)
// 1 - edge (DMSTAG_UP, DMSTAG_DOWN, DMSTAG_LEFT, DMSTAG_RIGHT)
// 2 - element (DMSTAG_ELEMENT)

// Use: internal
// @*/
// // ---------------------------------------
// static PetscInt convert_stencil_location_to_stratum_index_2d(DMStagStencilLocation loc)
// {
//   if (loc == DMSTAG_UP || loc == DMSTAG_DOWN || loc == DMSTAG_LEFT || loc == DMSTAG_RIGHT) {
//     return(1);
//   }
//   if (loc == DMSTAG_ELEMENT) {
//     return(2);
//   }
//   if (loc == DMSTAG_UP_LEFT || loc == DMSTAG_UP_RIGHT || loc == DMSTAG_DOWN_LEFT || loc == DMSTAG_DOWN_RIGHT) {
//     return(0);
//   }
//   return(-1);
// }

// ---------------------------------------
/*@
DMStagBCListInsertValues - return BC entries associated with a boundary and a dof

Input Parameter:
list - the DMStagBCList object
label - dof label: '.' vertex, '-' edge (horizontal), '|' edge (vertical), 'o' element
dof - component degree of freedom (DMStagStencil c)
_n - count of boundary dof
_idx - 1D array containing the index
_xc - 1D array containing the coordinates (true boundary)
_xc_stag - 1D array containing the coordinates (dof)
_value - 1D array containing the value
_type - 1D array containing BCtype

Notes:
User should call DMStagBCListGetValues() before.

Use: user
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "DMStagBCListInsertValues"
PetscErrorCode DMStagBCListInsertValues(DMStagBCList list,const char label,
                                        PetscInt dof,
                                        PetscInt *_n,PetscInt *_idx[],PetscScalar *_xc[],PetscScalar *_xc_stag[],PetscScalar *_value[],BCType *_type[])
{
  DMStagBC *bc = NULL;
  PetscInt si,k,n;
  PetscInt *idx;
  PetscScalar *value;
  BCType *type;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  if (!_n || !_idx || !_value || !_type) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must provide a valid (non-NULL) pointer for n (arg 4) idx (arg 5), value (arg 8), type (arg 9)");
  
  n = *_n;
  idx = *_idx;
  value = *_value;
  type = *_type;
  
  //si = convert_stencil_location_to_stratum_index_2d(loc);
  si = convert_stencil_label_to_stratum_index_2d(label);
  switch (si) {
    case 0:
      bc = list->bc_v;
      break;
    case 1:
      bc = list->bc_f;
      break;
    case 2:
      bc = list->bc_e;
      break;
    default:
      break;
  }
  for (k=0; k<n; k++) {
    bc[ idx[k] ].val = value[k];
    bc[ idx[k] ].type = type[k];
  }

  ierr = PetscFree(*_idx);CHKERRQ(ierr);
  ierr = PetscFree(*_value);CHKERRQ(ierr);
  ierr = PetscFree(*_type);CHKERRQ(ierr);
  
  if (_xc) { ierr = PetscFree(*_xc);CHKERRQ(ierr);}
  if (_xc_stag) { ierr = PetscFree(*_xc_stag);CHKERRQ(ierr);}
  
  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
DMStagBCListView - ASCII output on PETSC_COMM_WORLD of DMStagBCList object

Input Parameter:
list - the DMStagBCList object

Use: user/internal
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "DMStagBCListView"
PetscErrorCode DMStagBCListView(DMStagBCList list)
{
  PetscErrorCode ierr;
  PetscInt       i,dof[] = {0,0,0};
  DMStagBC       *bc;
  
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"DMStagBCListView\n");
  ierr = DMStagGetDOF(list->dm,&dof[0],&dof[1],&dof[2],NULL);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"  stratum size: %D (vertices) %D (faces) %D (elements)\n",dof[0],dof[1],dof[2]);
  
  PetscPrintf(PETSC_COMM_SELF,"  bc_vertices: size %D\n",list->nbc_vertex);
  bc = list->bc_v;
  for (i=0; i<list->nbc_vertex; i++) {
    PetscPrintf(PETSC_COMM_SELF,"    [%D] x,y (%+1.2e,%+1.2e) i,j,loc (%D %D %D) dof %D value %+1.2e type %D\n",i,bc[i].coord[0],bc[i].coord[1],bc[i].point.i,bc[i].point.j,bc[i].point.loc,bc[i].point.c,bc[i].val,(PetscInt)bc[i].type);
  }
  PetscPrintf(PETSC_COMM_SELF,"  bc_faces: size %D\n",list->nbc_face);
  bc = list->bc_f;
  for (i=0; i<list->nbc_face; i++) {
    PetscPrintf(PETSC_COMM_SELF,"    [%D] x,y (%+1.2e,%+1.2e) i,j,loc (%D %D %D) dof %D value %+1.2e type %D\n",i,bc[i].coord[0],bc[i].coord[1],bc[i].point.i,bc[i].point.j,bc[i].point.loc,bc[i].point.c,bc[i].val,(PetscInt)bc[i].type);
  }
  
  PetscPrintf(PETSC_COMM_SELF,"  bc_elements: size %D\n",list->nbc_element);
  bc = list->bc_e;
  for (i=0; i<list->nbc_element; i++) {
    PetscPrintf(PETSC_COMM_SELF,"    [%D] x,y (%+1.2e,%+1.2e) i,j,loc (%D %D %D) dof %D value %+1.2e type %D\n",i,bc[i].coord[0],bc[i].coord[1],bc[i].point.i,bc[i].point.j,bc[i].point.loc,bc[i].point.c,bc[i].val,(PetscInt)bc[i].type);
  }
  PetscFunctionReturn(0);
}

/*
 Performs a search over the relevent boundary list (element, face, vertex).
 The search is terminated when
   (i) Values for bc.i, bc.j matche the input `target_cell_i`, `target_cell_j`.
   (ii) The bc location matches the relevent stencil location associated with `label`.
        e.g. if location = '|' we check either DMSTAG_UP or DMSTAG_DOWN
   (iii) The bc dof (bc.c) matches the target value for `dof`
 All sub-domains conduct the search over their sub-domain. 
 By definition only a single pin-point BC must be defined over the entire domain.
 Hence after the search is completed a global reduction is performed to ensure that 
 one and only one sub-domain has identified a pin-point. An error is emitted if
 none or several sub-domains identified a pin-point.
 
 Developer note:
   - Currently a pin-point BC is defined as type BC_DIRICHLET_STAG.
   This is completely correct, however in future we may wish to distinguish pin-point BCs from
   normal Dirichlet constraints. For example: we may wish to apply a special scaling to pin-point
   BCs to improve the condition number of the matrix; we may wish to ignore / filter pin-point BCs,
   for and instead prescribe a constant null-space removal function.
*/
static PetscErrorCode _DMStagBCListPinValue(DMStagBCList list,
                                            PetscInt target_cell_i,PetscInt target_cell_j,
                                            const char label,PetscInt dof,PetscScalar val)
{
  PetscErrorCode ierr;
  PetscInt       k,d,ndof,stag_dof[] = {0,0,0},len = 0;
  MPI_Comm       comm;
  PetscMPIInt    rank;
  int            found = 0;
  DMStagBC       *bcpoint = NULL;
  DMStagBC       *bc = NULL;
  DMStagStencilLocation locset[] = {DMSTAG_NULL_LOCATION,DMSTAG_NULL_LOCATION,DMSTAG_NULL_LOCATION,DMSTAG_NULL_LOCATION};
  
  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)(list->dm),&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  
  ierr = DMStagGetDOF(list->dm,&stag_dof[0],&stag_dof[1],&stag_dof[2],NULL);CHKERRQ(ierr);

  if (label == 'o') {
    len = list->nbc_element;
    bc = list->bc_e;
    ndof = stag_dof[2];
    locset[0] = DMSTAG_ELEMENT;
  }
  
  if (label == '-') {
    len = list->nbc_face;
    bc = list->bc_f;
    ndof = stag_dof[1];
    locset[0] = DMSTAG_LEFT;
    locset[1] = DMSTAG_RIGHT;
  }
  
  if (label == '|') {
    len = list->nbc_face;
    bc = list->bc_f;
    ndof = stag_dof[1];
    locset[0] = DMSTAG_UP;
    locset[1] = DMSTAG_DOWN;
  }

  if (label == '.') {
    len = list->nbc_vertex;
    bc = list->bc_v;
    ndof = stag_dof[0];
    locset[0] = DMSTAG_UP_LEFT;
    locset[1] = DMSTAG_UP_RIGHT;
    locset[2] = DMSTAG_DOWN_RIGHT;
    locset[3] = DMSTAG_DOWN_LEFT;
  }

  found = 0;
  for (k=0; k<len; k++) {
    if (found == 1) break; /* exit loop if pin-point located */
    if ((bc[k].point.i == target_cell_i) && (bc[k].point.j == target_cell_j)) { /* locate cell */

      /* check location is consistent with label, else skip to next bc point */
      if (label == '.') { /* vertices require comparison with four potential locations */
        if (   (bc[k].point.loc != locset[0])
            && (bc[k].point.loc != locset[1])
            && (bc[k].point.loc != locset[2])
            && (bc[k].point.loc != locset[3])
            ) continue;
      } else if (label == '-' || label == '|') { /* faces require comparison with two potential locations */
        if ((bc[k].point.loc != locset[0]) && (bc[k].point.loc != locset[1])) continue;
      } else if (label == 'o') { /* cells require comparison with on potential location */
        if (bc[k].point.loc != locset[0]) continue;
      }
      
      /* check `dof` matches the dof index associated with bc point */
      for (d=0; d<ndof; d++) {
        if (bc[k].point.c == dof) {
          found = 1; /* flag successful identification of the pin-point */
          bc[k].val  = val;
          bc[k].type = BC_DIRICHLET_STAG;
          bcpoint = &bc[k]; /* get pointer to matching bc point for reporting */
          break;
        }
      }
    }
  }
  
  /* Check that one and only one sub-domain identified a pin-point */
  ierr = MPI_Allreduce(MPI_IN_PLACE,&found,1,MPI_INT,MPI_SUM,comm);CHKERRQ(ierr);
  if (found == 0) SETERRQ(comm,PETSC_ERR_SUP,"No pin-point was identified on any sub-domain");
  if (found > 1) SETERRQ(comm,PETSC_ERR_SUP,"A target pin-points was identified on more than one sub-domain");
  
  /*
  if (bcpoint) {
    printf("[Pin-point BC]\n");
    printf("  pin point: i,j   %d %d <rank %d>\n",bcpoint->point.i,bcpoint->point.j,(int)rank);
    printf("  pin point: c     %d\n",bcpoint->point.c);
    printf("  pin point: coor  %+1.4e %+1.4e\n",bcpoint->coord[0],bcpoint->coord[1]);
    printf("  pin point: val   %+1.4e\n",bcpoint->val);
  }
  */
  PetscFunctionReturn(0);
}

/*
 Default pin-point boundary condition specification.
 Always pin the dof associated with the cell located in the lower left corner.
 
 Input:
 list  - The DMStagBCList object.
 label - DOF location identifier. One of {'o', '.', '|', '-'}.
 dof   - The index of the degree-of-freedom associated with `label` to constrain.
 val   - The value assigned to the constraint.
*/
PetscErrorCode DMStagBCListPinValue(DMStagBCList list,const char label,PetscInt dof,PetscScalar val)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = _DMStagBCListPinValue(list,0,0,label,dof,val);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
 Custom pin-point boundary condition allowing constraint to be imposed at 
 canonical locations (e.g. independent of the mesh size, or parallel decomposition)
 within the domain. The canonical locations are the corners of the domain.
 
 Input:
 list   - The DMStagBCList object.
 corner - Identifier for the domain corner. One of {DMSTAG_DOWN_RIGHT,DMSTAG_DOWN_LEFT,DMSTAG_UP_RIGHT,DMSTAG_UP_LEFT}.
 label  - DOF location identifier. One of {'o', '.', '|', '-'}.
 dof    - The index of the degree-of-freedom associated with `label` to constrain.
 val    - The value assigned to the constraint.
*/
PetscErrorCode DMStagBCListPinCornerValue(DMStagBCList list,DMStagStencilLocation corner,const char label,PetscInt dof,PetscScalar val)
{
  PetscInt       M,N;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = DMStagGetGlobalSizes(list->dm,&M,&N,NULL);CHKERRQ(ierr);
  switch (corner) {
    case DMSTAG_DOWN_RIGHT:
      ierr = _DMStagBCListPinValue(list,0,N-1,label,dof,val);CHKERRQ(ierr);
      break;
    case DMSTAG_DOWN_LEFT:
      ierr = _DMStagBCListPinValue(list,0,0,label,dof,val);CHKERRQ(ierr);
      break;
    case DMSTAG_UP_RIGHT:
      ierr = _DMStagBCListPinValue(list,M-1,N-1,label,dof,val);CHKERRQ(ierr);
      break;
    case DMSTAG_UP_LEFT:
      ierr = _DMStagBCListPinValue(list,M-1,0,label,dof,val);CHKERRQ(ierr);
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)list->dm),PETSC_ERR_SUP,"Corners of domain are identified with {down-right,down-left,up-right,up-left}");
      break;
  }
  PetscFunctionReturn(0);
}
