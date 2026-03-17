#include "fdpde_dmstag.h"

void pythonemit(FILE *fp,const char str[])
{
  if (fp) {
    fprintf(fp,"%s",str);
  }
}

void pythonemitvec(FILE *fp,const char name[])
{
  char pline[PETSC_MAX_PATH_LEN];
  if (fp) {
    pythonemit(fp,"    objecttype = io.readObjectType(fp)\n");
    pythonemit(fp,"    v = io.readVec(fp)\n");
    PetscSNPrintf(pline,PETSC_MAX_PATH_LEN-1,"    data['%s'] = v\n",name);
    pythonemit(fp,pline);
  }
}

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
  PetscFunctionBegin;
  bc->point.i = i;
  bc->point.j = j;
  bc->point.c = c;
  bc->point.loc = loc;
  PetscCall(DMStagGetLocationSlot(dm,loc,c,&bc->idx));
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscFunctionBegin;
  
  PetscCall(DMStagGetDOF(dm,&dof0,&dof1,&dof2,NULL));

  if (dof2) {PetscCall(DMStagGetProductCoordinateLocationSlot(dm,DMSTAG_ELEMENT,&icenter));} 
  if (dof0 || dof1) { 
    PetscCall(DMStagGetProductCoordinateLocationSlot(dm,DMSTAG_LEFT,&iprev));
    PetscCall(DMStagGetProductCoordinateLocationSlot(dm,DMSTAG_RIGHT,&inext));
  } 

  if ((loc == DMSTAG_DOWN_LEFT)  || (loc == DMSTAG_UP_LEFT)  || (loc == DMSTAG_LEFT))  ii = iprev;
  if ((loc == DMSTAG_ELEMENT)    || (loc == DMSTAG_DOWN)     || (loc == DMSTAG_UP))    ii = icenter;
  if ((loc == DMSTAG_DOWN_RIGHT) || (loc == DMSTAG_UP_RIGHT) || (loc == DMSTAG_RIGHT)) ii = inext;
  
  if ((loc == DMSTAG_DOWN_LEFT) || (loc == DMSTAG_DOWN) || (loc == DMSTAG_DOWN_RIGHT)) jj = iprev;
  if ((loc == DMSTAG_ELEMENT)   || (loc == DMSTAG_LEFT) || (loc == DMSTAG_RIGHT))      jj = icenter;
  if ((loc == DMSTAG_UP_LEFT)   || (loc == DMSTAG_UP)   || (loc == DMSTAG_UP_RIGHT))   jj = inext;
  
  bc->coord[0] = cx[i][ii];
  bc->coord[1] = cz[j][jj];

  PetscFunctionReturn(PETSC_SUCCESS);
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
  
  PetscFunctionBegin;
  dm = list->dm;
  PetscCall(DMStagGetGlobalSizes(dm,&Nx,&Nz,NULL));
  PetscCall(DMStagGetCorners(dm,&sx,&sz,NULL,&nx,&nz,NULL,NULL,NULL,NULL));
  PetscCall(DMStagGetDOF(dm,&dof0,&dof1,&dof2,NULL));
  
  // Save grid info for BC dofs
  ndof0 = ndof1 = ndof2 = 0;
  for (j = sz; j<sz+nz; j++) {
    for (i = sx; i<sx+nx; i++) {
      if ((i == 0) || (i == Nx-1)) {
          if (i == 0) { loc = DMSTAG_DOWN_LEFT; }
          else        { loc = DMSTAG_DOWN_RIGHT; }
          loc = (i == 0) ? DMSTAG_DOWN_LEFT : DMSTAG_DOWN_RIGHT;
          for (ii = 0; ii<dof0; ii++) {
            PetscCall(_DMStagBCFillIndices(dm,loc,ii,i,j,&list->bc_v[ndof0++]));
          }
          if (i == 0) { loc = DMSTAG_LEFT; }
          else        { loc = DMSTAG_RIGHT; }
          loc = (i == 0) ? DMSTAG_LEFT : DMSTAG_RIGHT;
          loc1 = DMSTAG_DOWN;
          for (ii = 0; ii<dof1; ii++) {
            PetscCall(_DMStagBCFillIndices(dm,loc, ii,i,j,&list->bc_f[ndof1++]));
            PetscCall(_DMStagBCFillIndices(dm,loc1,ii,i,j,&list->bc_f[ndof1++]));
          }
          for (ii = 0; ii<dof2; ii++) {
            PetscCall(_DMStagBCFillIndices(dm,DMSTAG_ELEMENT,ii,i,j,&list->bc_e[ndof2++]));
          }
        if (j == Nz-1) {
            if (i == 0) { loc = DMSTAG_UP_LEFT; }
            else        { loc = DMSTAG_UP_RIGHT; }
            loc = (i == 0) ? DMSTAG_UP_LEFT : DMSTAG_UP_RIGHT;
            for (ii = 0; ii<dof0; ii++) {
              PetscCall(_DMStagBCFillIndices(dm,loc,ii,i,j,&list->bc_v[ndof0++]));
            }
            for (ii = 0; ii<dof1; ii++) {
              PetscCall(_DMStagBCFillIndices(dm,DMSTAG_UP,ii,i,j,&list->bc_f[ndof1++]));
            }
        }
      } else {
        if ((j == 0) || (j == Nz-1)) {
            if (j == 0) { loc = DMSTAG_DOWN_LEFT; }
            else        { loc = DMSTAG_UP_LEFT; }
            loc = (j == 0) ? DMSTAG_DOWN_LEFT : DMSTAG_UP_LEFT;
            for (ii = 0; ii<dof0; ii++) {
              PetscCall(_DMStagBCFillIndices(dm,loc,ii,i,j,&list->bc_v[ndof0++]));
            }
            if (j == 0) { loc = DMSTAG_DOWN; }
            else        { loc = DMSTAG_UP; }
            loc = (j == 0) ? DMSTAG_DOWN : DMSTAG_UP;
            loc1 = DMSTAG_LEFT;
            for (ii = 0; ii<dof1; ii++) {
              PetscCall(_DMStagBCFillIndices(dm,loc, ii,i,j,&list->bc_f[ndof1++]));
              PetscCall(_DMStagBCFillIndices(dm,loc1,ii,i,j,&list->bc_f[ndof1++]));
            }
            for (ii = 0; ii<dof2; ii++) {
              PetscCall(_DMStagBCFillIndices(dm,DMSTAG_ELEMENT,ii,i,j,&list->bc_e[ndof2++]));
            }
          if (i == Nx-2) {
              if (j == 0) { loc = DMSTAG_DOWN_RIGHT; }
              else        { loc = DMSTAG_UP_RIGHT; }
              loc = (j == 0) ? DMSTAG_DOWN_RIGHT : DMSTAG_UP_RIGHT;
              for (ii = 0; ii<dof0; ii++) {
                PetscCall(_DMStagBCFillIndices(dm,loc,ii,i,j,&list->bc_v[ndof0++]));
              }
              for (ii = 0; ii<dof1; ii++) {
                PetscCall(_DMStagBCFillIndices(dm,DMSTAG_RIGHT,ii,i,j,&list->bc_f[ndof1++]));
              }
          }
        }
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
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

  PetscFunctionBegin;
  dm = list->dm;
  PetscCall(DMGetCoordinateDM(dm,&dmCoord));
  PetscCall(DMGetType(dmCoord,&dmType));
  PetscCall(PetscStrcmp(DMPRODUCT,dmType,&isProduct));
  if (!isProduct) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Coordinate DM must be of type DMPRODUCT");
  PetscCall(DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));
  {
    DMStagBC *bcs[] = { list->bc_v , list->bc_f, list->bc_e };
    PetscInt size[] = { list->nbc_vertex , list->nbc_face , list->nbc_element };
    
    for (k=0; k<3; k++) {
      for (ii = 0; ii<size[k]; ii++) {
        DMStagBC *b = &bcs[k][ii];
        PetscCall(_DMStagBCFillCoords(k,dm,coordx,coordz,b->point.loc,b->point.i,b->point.j,b));
      }
    }
  }
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
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
  
  PetscFunctionBegin;
  PetscCall(DMGetCoordinateDM(list->dm,&dmCoord));
  if (!dmCoord) SETERRQ(PetscObjectComm((PetscObject)list->dm),PETSC_ERR_SUP,"DMStag must have coordinates defined. Hint - coordinates must be defined via DMPRODUCT)");
  PetscCall(_DMStagBCListSetupCoordinates(list));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// // ---------------------------------------
// /*@
// DMStagBCListGetVertexBCs - return a 1-D DMStagBC array containing the vertex (corner) boundary dofs

// Input Parameter:
// list - the DMStagBCList object

// Output Parameters:
// nbc - count of vertex dofs
// l - 1-D DMStagBC array

// Use: user/internal
// @*/
// // ---------------------------------------
// #undef __FUNCT__
// #define __FUNCT__ "DMStagBCListGetVertexBCs"
// PetscErrorCode DMStagBCListGetVertexBCs(DMStagBCList list,PetscInt *nbc,DMStagBC *l[])
// {
//   PetscFunctionBegin;
//   if (nbc) { *nbc = list->nbc_vertex; }
//   if (l) { *l = list->bc_v; }
//   PetscFunctionReturn(PETSC_SUCCESS);
// }

// // ---------------------------------------
// /*@
// DMStagBCListGetFaceBCs - return a 1-D DMStagBC array containing the face (edge) boundary dofs

// Input Parameter:
// list - the DMStagBCList object

// Output Parameters:
// nbc - count of edge dofs
// l - 1-D DMStagBC array

// Use: user/internal
// @*/
// // ---------------------------------------
// #undef __FUNCT__
// #define __FUNCT__ "DMStagBCListGetFaceBCs"
// PetscErrorCode DMStagBCListGetFaceBCs(DMStagBCList list,PetscInt *nbc,DMStagBC *l[])
// {
//   PetscFunctionBegin;
//   if (nbc) { *nbc = list->nbc_face; }
//   if (l) { *l = list->bc_f; }
//   PetscFunctionReturn(PETSC_SUCCESS);
// }

// // ---------------------------------------
// /*@
// DMStagBCListGetElementBCs - return a 1-D DMStagBC array containing the element (center) boundary dofs

// Input Parameter:
// list - the DMStagBCList object

// Output Parameters:
// nbc - count of element dofs
// l - 1-D DMStagBC array

// Use: user/internal
// @*/
// // ---------------------------------------
// #undef __FUNCT__
// #define __FUNCT__ "DMStagBCListGetElementBCs"
// PetscErrorCode DMStagBCListGetElementBCs(DMStagBCList list,PetscInt *nbc,DMStagBC *l[])
// {
//   PetscFunctionBegin;
//   if (nbc) { *nbc = list->nbc_element; }
//   if (l) { *l = list->bc_e; }
//   PetscFunctionReturn(PETSC_SUCCESS);
// }

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
  PetscInt       dim,Nx,Nz;
  PetscBool      isstag;
  DMStagBCList   l;
  
  PetscFunctionBegin;
  /* check assumptions */
  PetscCall(DMGetDimension(dm,&dim));
  if (dim != 2) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Only valid for 2d DM");
  PetscCall(PetscObjectTypeCompare((PetscObject)dm,DMSTAG,&isstag));
  if (!isstag) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Only valid for DMStag");
  PetscCall(DMStagGetGlobalSizes(dm,&Nx,&Nz,NULL));
  if (Nx < 3) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Implementation of BCList only valid for n-cells-i >= 3. Found nx=%D",Nx);
  if (Nz < 3) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Implementation of BCList only valid for n-cells-j >= 3. Found ny=%D",Nz);
  { /* Check for existence of a coordinateDM - I deliberately don't call CHKERRQ() after DMGetCoordinateDM() so that I can throw the error message below which is more helpful than the error thrown from DMGetCoordinateDM() */
    DM dmCoord = NULL;
    PetscCall(DMGetCoordinateDM(dm,&dmCoord));
    if (!dmCoord) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"DMStag must have coordinates defined. Hint - coordinates must be defined via DMPRODUCT");
  }
  
  PetscCall(PetscCalloc1(1,&l));
  l->dm = dm;
  l->evaluate = NULL;
  l->data = NULL;
  
  {
    PetscInt sx,sz,nx,nz;
    PetscInt ii,i,j,ndof0,ndof1,ndof2,dof0,dof1,dof2;
    
    // Count local boundary dofs
    PetscCall(DMStagGetCorners(dm,&sx,&sz,NULL,&nx,&nz,NULL,NULL,NULL,NULL));
    PetscCall(DMStagGetDOF(dm,&dof0,&dof1,&dof2,NULL));
    
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
    PetscCall(PetscCalloc1(l->nbc_vertex+1, &l->bc_v));
    PetscCall(PetscCalloc1(l->nbc_face+1,   &l->bc_f));
    PetscCall(PetscCalloc1(l->nbc_element+1,&l->bc_e));
    
    PetscCall(_DMStagBCListSetupIndices(l));
    
    PetscCall(_DMStagBCListSetupCoordinates(l));
    
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
  PetscFunctionReturn(PETSC_SUCCESS);
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
  DMStagBCList   l;
  PetscFunctionBegin;
  if (!list) PetscFunctionReturn(PETSC_SUCCESS);
  if (!*list) PetscFunctionReturn(PETSC_SUCCESS);
  l = *list;
  PetscCall(PetscFree(l->bc_v));
  PetscCall(PetscFree(l->bc_f));
  PetscCall(PetscFree(l->bc_e));
  PetscCall(PetscFree(l));
  *list = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
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
  
  PetscFunctionBegin;
  PetscCall(DMStagGetGlobalSizes(list->dm,&Nx,&Nz,NULL));
  PetscCall(PetscCalloc1(2*Nx,&idx)); /* over allocate */
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
  PetscFunctionReturn(PETSC_SUCCESS);
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
  
  PetscFunctionBegin;
  if (dir != DMSTAG_UP && dir != DMSTAG_DOWN) SETERRQ(PetscObjectComm((PetscObject)list->dm),PETSC_ERR_SUP,"Direction (arg 3) must be either DMSTAG_UP or DMSTAG_DOWN");
  
  PetscCall(DMStagGetGlobalSizes(list->dm,&Nx,&Nz,NULL));
  PetscCall(PetscCalloc1(2*Nx,&idx)); /* over allocate */
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
  PetscFunctionReturn(PETSC_SUCCESS);
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
  
  PetscFunctionBegin;
  PetscCall(DMStagGetGlobalSizes(list->dm,&Nx,&Nz,NULL));
  PetscCall(PetscCalloc1(2*Nz,&idx)); /* over allocate */
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
  PetscFunctionReturn(PETSC_SUCCESS);
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
  
  PetscFunctionBegin;
  if (dir != DMSTAG_LEFT && dir != DMSTAG_RIGHT) SETERRQ(PetscObjectComm((PetscObject)list->dm),PETSC_ERR_SUP,"Direction (arg 3) must be either DMSTAG_LEFT or DMSTAG_RIGHT");
  
  PetscCall(DMStagGetGlobalSizes(list->dm,&Nx,&Nz,NULL));
  PetscCall(PetscCalloc1(2*Nz,&idx)); /* over allocate */
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
  PetscFunctionReturn(PETSC_SUCCESS);
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
  
  PetscFunctionBegin;
  PetscCall(DMStagGetGlobalSizes(list->dm,&Nx,&Nz,NULL));

  if (II == -1) {
    PetscCall(PetscCalloc1(Nx,&idx));
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
    PetscCall(PetscCalloc1(Nz,&idx)); /* over allocate */
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
  PetscFunctionReturn(PETSC_SUCCESS);
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
  
  PetscFunctionBegin;
  if (label == '.') SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support to get vertex bc DOFs. Only support for getting face bc values. ");
  
  if (!_n || !_idx || !_value || !_type) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must provide a valid (non-NULL) pointer for n (arg 4), idx (arg 5), value (arg 7), type (arg 8)");
  
  PetscCall(DMStagGetGlobalSizes(list->dm,&Nx,&Nz,NULL));
  
  switch (domain_face) {
    case 'n':
      if (label == '-') {
        PetscCall(_DMStagBCListGetIndices_J_left_and_right(list,dof,Nz-1,&n,&idx)); // vx
      } else if (label == '|') {
        PetscCall(_DMStagBCListGetIndices_J_updown(list,dof,Nz-1,DMSTAG_UP,&n,&idx)); // vy
      } else if (label == 'o') {
        _DMStagBCListGetIndices_center(list,dof,-1,Nz-1,&n,&idx);
      } else {
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Direction[North] No support to get vertex data. Unknown stratum label %s provided - must be one of {'-','|','o'}",&label);
      }
      break;
      
    case 's':
      if (label == '-') {
        PetscCall(_DMStagBCListGetIndices_J_left_and_right(list,dof,0,&n,&idx));
      } else if (label == '|') {
        PetscCall(_DMStagBCListGetIndices_J_updown(list,dof,0,DMSTAG_DOWN,&n,&idx)); // vy
      } else if (label == 'o') {
        _DMStagBCListGetIndices_center(list,dof,-1,0,&n,&idx);
      } else {
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Direction[South] No support to get vertex data. Unknown stratum label %s provided - must be one of {'-','|','o'}",&label);
      }
      break;

    case 'e':
      if (label == '-') {
        PetscCall(_DMStagBCListGetIndices_I_leftright(list,dof,Nx-1,DMSTAG_RIGHT,&n,&idx)); // vx
      } else if (label == '|') {
        PetscCall(_DMStagBCListGetIndices_I_up_and_down(list,dof,Nx-1,&n,&idx)); // vy
      } else if (label == 'o') {
        _DMStagBCListGetIndices_center(list,dof,Nx-1,-1,&n,&idx);
      } else {
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Direction[East] No support to get vertex data. Unknown stratum label %s provided - must be one of {'-','|','o'}",&label);
      }
      break;

    case 'w':
      if (label == '-') {
        PetscCall(_DMStagBCListGetIndices_I_leftright(list,dof,0,DMSTAG_LEFT,&n,&idx)); // vx
      } else if (label == '|') {
        PetscCall(_DMStagBCListGetIndices_I_up_and_down(list,dof,0,&n,&idx)); // vy
      } else if (label == 'o') {
        _DMStagBCListGetIndices_center(list,dof,0,-1,&n,&idx);
      } else {
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Direction[West] No support to get vertex data. Unknown stratum label %s provided - must be one of {'-','|','o'}",&label);
      }
      break;
      
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Direction %s is not supported. Direction must be one of {'n','e','s','w'}",&label);
      break;
  }

  PetscCall(PetscCalloc1(2*n,&xc));
  PetscCall(PetscCalloc1(2*n,&xc_stag));
  PetscCall(PetscCalloc1(n,&v));
  PetscCall(PetscCalloc1(n,&t));
  
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
  
  PetscCall(DMStagCellSizeLocal_2d(list->dm,&nx_local,&nz_local,&dx,&dz)); 
  PetscCall(DMStagGetCorners(list->dm,&start[0],&start[1],NULL,NULL,NULL,NULL,NULL,NULL,NULL));

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

  PetscCall(PetscFree(dx));
  PetscCall(PetscFree(dz));

  *_n = n;  *_idx = idx;  *_value = v;  *_type = t;
  if (_xc) { *_xc = xc; }
  else {PetscCall(PetscFree(xc));}
  if (_xc_stag) { *_xc_stag = xc_stag; }
  else {PetscCall(PetscFree(xc_stag));}
  PetscFunctionReturn(PETSC_SUCCESS);
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

// // // ---------------------------------------
// // /*@
// // convert_stencil_location_to_stratum_index_2d - returns a number associated with the DMStagStencilLocation
// // 0 - vertex (DMSTAG_DOWN_LEFT, DMSTAG_DOWN_RIGHT, DMSTAG_UP_LEFT, DMSTAG_UP_RIGHT)
// // 1 - edge (DMSTAG_UP, DMSTAG_DOWN, DMSTAG_LEFT, DMSTAG_RIGHT)
// // 2 - element (DMSTAG_ELEMENT)

// // Use: internal
// // @*/
// // // ---------------------------------------
// // static PetscInt convert_stencil_location_to_stratum_index_2d(DMStagStencilLocation loc)
// // {
// //   if (loc == DMSTAG_UP || loc == DMSTAG_DOWN || loc == DMSTAG_LEFT || loc == DMSTAG_RIGHT) {
// //     return(1);
// //   }
// //   if (loc == DMSTAG_ELEMENT) {
// //     return(2);
// //   }
// //   if (loc == DMSTAG_UP_LEFT || loc == DMSTAG_UP_RIGHT || loc == DMSTAG_DOWN_LEFT || loc == DMSTAG_DOWN_RIGHT) {
// //     return(0);
// //   }
// //   return(-1);
// // }

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

  PetscCall(PetscFree(*_idx));
  PetscCall(PetscFree(*_value));
  PetscCall(PetscFree(*_type));
  
  if (_xc) { PetscCall(PetscFree(*_xc));}
  if (_xc_stag) { PetscCall(PetscFree(*_xc_stag));}
  
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscInt       i,dof[] = {0,0,0};
  DMStagBC       *bc;
  
  PetscFunctionBegin;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"DMStagBCListView\n"));
  PetscCall(DMStagGetDOF(list->dm,&dof[0],&dof[1],&dof[2],NULL));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"  stratum size: %D (vertices) %D (faces) %D (elements)\n",dof[0],dof[1],dof[2]));
  
  PetscCall(PetscPrintf(PETSC_COMM_SELF,"  bc_vertices: size %D\n",list->nbc_vertex));
  bc = list->bc_v;
  for (i=0; i<list->nbc_vertex; i++) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"    [%D] x,y (%+1.2e,%+1.2e) i,j,loc (%D %D %D) dof %D value %+1.2e type %D\n",i,bc[i].coord[0],bc[i].coord[1],bc[i].point.i,bc[i].point.j,bc[i].point.loc,bc[i].point.c,bc[i].val,(PetscInt)bc[i].type));
  }
  PetscCall(PetscPrintf(PETSC_COMM_SELF,"  bc_faces: size %D\n",list->nbc_face));
  bc = list->bc_f;
  for (i=0; i<list->nbc_face; i++) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"    [%D] x,y (%+1.2e,%+1.2e) i,j,loc (%D %D %D) dof %D value %+1.2e type %D\n",i,bc[i].coord[0],bc[i].coord[1],bc[i].point.i,bc[i].point.j,bc[i].point.loc,bc[i].point.c,bc[i].val,(PetscInt)bc[i].type));
  }
  
  PetscCall(PetscPrintf(PETSC_COMM_SELF,"  bc_elements: size %D\n",list->nbc_element));
  bc = list->bc_e;
  for (i=0; i<list->nbc_element; i++) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"    [%D] x,y (%+1.2e,%+1.2e) i,j,loc (%D %D %D) dof %D value %+1.2e type %D\n",i,bc[i].coord[0],bc[i].coord[1],bc[i].point.i,bc[i].point.j,bc[i].point.loc,bc[i].point.c,bc[i].val,(PetscInt)bc[i].type));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
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
  PetscInt       k,d,ndof,stag_dof[] = {0,0,0},len = 0;
  MPI_Comm       comm;
  PetscMPIInt    rank;
  int            found = 0;
  DMStagBC       *bcpoint = NULL;
  DMStagBC       *bc = NULL;
  DMStagStencilLocation locset[] = {DMSTAG_NULL_LOCATION,DMSTAG_NULL_LOCATION,DMSTAG_NULL_LOCATION,DMSTAG_NULL_LOCATION};
  
  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)(list->dm),&comm));
  PetscCall(MPI_Comm_rank(comm,&rank));
  
  PetscCall(DMStagGetDOF(list->dm,&stag_dof[0],&stag_dof[1],&stag_dof[2],NULL));

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
  PetscCall(MPI_Allreduce(MPI_IN_PLACE,&found,1,MPI_INT,MPI_SUM,comm));
  if (found == 0) SETERRQ(comm,PETSC_ERR_SUP,"No pin-point was identified on any sub-domain");
  if (found > 1) SETERRQ(comm,PETSC_ERR_SUP,"A target pin-points was identified on more than one sub-domain");
  
  if (bcpoint) {
    // PetscCall(PetscPrintf(PETSC_COMM_WORLD,"[Pin-point BC]\n"));
    // PetscCall(PetscPrintf(PETSC_COMM_WORLD,"  pin point: i,j   %d %d <rank %d>\n",bcpoint->point.i,bcpoint->point.j,(int)rank));
    // PetscCall(PetscPrintf(PETSC_COMM_WORLD,"  pin point: c     %d\n",bcpoint->point.c));
    // PetscCall(PetscPrintf(PETSC_COMM_WORLD,"  pin point: coor  %+1.4e %+1.4e\n",bcpoint->coord[0],bcpoint->coord[1]));
    // PetscCall(PetscPrintf(PETSC_COMM_WORLD,"  pin point: val   %+1.4e\n",bcpoint->val));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
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
  PetscFunctionBegin;
  PetscCall(_DMStagBCListPinValue(list,0,0,label,dof,val));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------s
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
  PetscFunctionBegin;
  PetscCall(DMStagGetGlobalSizes(list->dm,&M,&N,NULL));
  switch (corner) {
    case DMSTAG_DOWN_RIGHT:
      PetscCall(_DMStagBCListPinValue(list,0,N-1,label,dof,val));
      break;
    case DMSTAG_DOWN_LEFT:
      PetscCall(_DMStagBCListPinValue(list,0,0,label,dof,val));
      break;
    case DMSTAG_UP_RIGHT:
      PetscCall(_DMStagBCListPinValue(list,M-1,N-1,label,dof,val));
      break;
    case DMSTAG_UP_LEFT:
      PetscCall(_DMStagBCListPinValue(list,M-1,0,label,dof,val));
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)list->dm),PETSC_ERR_SUP,"Corners of domain are identified with {down-right,down-left,up-right,up-left}");
      break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
/* Note: This returns the numbers of cells and their sizes within a subdomain*/
PetscErrorCode DMStagCellSizeLocal_2d(DM dm, PetscInt *_nx, PetscInt *_ny, PetscScalar *_dx[],PetscScalar *_dy[])
{
  PetscInt          i,start[2],n[2];
  PetscScalar       *dx,*dy,**cArrX,**cArrY;
  PetscInt          iNext,iPrev;

  PetscFunctionBegin;
  PetscCall(DMStagGetCorners(dm,&start[0],&start[1],NULL,&n[0],&n[1],NULL,NULL,NULL,NULL));
  PetscCall(DMStagGetProductCoordinateArraysRead(dm,&cArrX,&cArrY,NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm,DMSTAG_LEFT,&iPrev));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm,DMSTAG_RIGHT,&iNext));
  PetscCall(PetscCalloc1(n[0],&dx));
  PetscCall(PetscCalloc1(n[1],&dy));
  for (i=start[0]; i<start[0]+n[0]; i++) {
    dx[i-start[0]] = PetscAbs(cArrX[i][iNext] - cArrX[i][iPrev]);
  }
  for (i=start[1]; i<start[1]+n[1]; i++) {
    dy[i-start[1]] = PetscAbs(cArrY[i][iNext] - cArrY[i][iPrev]);
  }
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm,&cArrX,&cArrY,NULL));
  *_nx = n[0];
  *_ny = n[1];
  *_dx = dx; *_dy= dy;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
/*@
DMStagViewBinaryPython_Seq - sequential (mpi size=1) output routine for DMStagViewBinaryPython() 

Writes a petsc binary file describing the DMStag object, and the data from the vector X.
The binary output pulls apart X and writes out seperate Vec objects for DOFs defined on the DMStag stratum.
Data living on an edge/face is decomposed into 2 (2D) or 3 (3D) face-wise Vec's.
The binary file created is named {prefix}.pbin.
 
The function also emits a python script named {prefix}.py which will load all binary data in the file.
The python script shoves all data written into a dict() to allow easy access / discovery of the data.
The named fields are:
   "x1d_vertex" - 1D array of x-coordinates associated with vertices
   "x1d_cell" - 1D array of x-coordinates associated with cells
   "y1d_vertex" - 1D array of y-coordinates associated with vertices
   "y1d_cell" - 1D array of y-coordinates associated with cells
   "X_vertex" - entries from X with correspond to DOFs on vertices
   "X_face_x" - entries from X with correspond to DOFs on faces with normals pointing in {+,-}x direction
   "X_face_y" - entries from X with correspond to DOFs on faces with normals pointing in {+,-}y direction
   "X_cell" - entries from X with correspond to DOFs on elements
 
Limitations:
   Supports sequential MPI jobs.
   Supports DMPRODUCT coordinates.

Use: internal
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "DMStagViewBinaryPython_SEQ"
PetscErrorCode DMStagViewBinaryPython_SEQ(DM dm,Vec X,const char prefix[])
{
  PetscViewer v;
  PetscInt M,N,P,dim;
  FILE *fp = NULL;
  char fname[PETSC_MAX_PATH_LEN],string[PETSC_MAX_PATH_LEN];
  MPI_Comm comm;
  PetscMPIInt size;
  PetscBool view_coords = PETSC_TRUE; /* ultimately this would be an input arg */
  PetscFunctionBegin;
  
  comm = PetscObjectComm((PetscObject)dm);
  PetscCall(MPI_Comm_size(comm,&size)); 
  if (size != 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Sequential only");

  // /* check for instances of "." in the file name so that the file can be imported */
  // {
  //   size_t k,len;
  //   PetscCall(PetscStrlen(prefix,&len));
  //   for (k=0; k<len; k++) if (prefix[k] == '.') PetscCall(PetscPrintf(comm,"[DMStagViewBinaryPython_SEQ] Warning: prefix %s contains the symbol '.'. Hence you will not be able to import the emitted python script. Consider change the prefix\n",prefix));
  // }
  
  PetscCall(PetscSNPrintf(fname,PETSC_MAX_PATH_LEN-1,"%s.pbin",prefix));
  PetscCall(PetscViewerBinaryOpen(comm,fname,FILE_MODE_WRITE,&v));
  
  PetscCall(PetscSNPrintf(string,PETSC_MAX_PATH_LEN-1,"%s.py",prefix));
  
  fp = fopen(string,"w");
  if (!fp) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Cannot open file %s",string);
  
  pythonemit(fp,"import PetscBinaryIO as pio\n");
  pythonemit(fp,"import numpy as np\n\n");

  pythonemit(fp,"def _PETScBinaryFilePrefix():\n");
  PetscSNPrintf(string,PETSC_MAX_PATH_LEN-1,"  return \"%s\"\n",prefix);
  pythonemit(fp,string);
  pythonemit(fp,"\n");

  pythonemit(fp,"def _PETScBinaryLoad():\n");
  pythonemit(fp,"  io = pio.PetscBinaryIO()\n");

  PetscSNPrintf(string,PETSC_MAX_PATH_LEN-1,"  filename = \"%s\"\n",fname);
  pythonemit(fp,string);
  pythonemit(fp,"  data = dict()\n");
  pythonemit(fp,"  with open(filename) as fp:\n");
  
  PetscCall(DMGetDimension(dm,&dim));
  PetscCall(DMStagGetGlobalSizes(dm,&M,&N,&P));
  
  PetscCall(PetscViewerBinaryWrite(v,(void*)&M,1,PETSC_INT));
  PetscCall(PetscViewerBinaryWrite(v,(void*)&N,1,PETSC_INT));
  PetscCall(PetscViewerBinaryWrite(v,(void*)&P,1,PETSC_INT));
  
  pythonemit(fp,"    v = io.readInteger(fp)\n"); pythonemit(fp,"    data['Nx'] = v\n");
  pythonemit(fp,"    v = io.readInteger(fp)\n"); pythonemit(fp,"    data['Ny'] = v\n");
  pythonemit(fp,"    v = io.readInteger(fp)\n"); pythonemit(fp,"    data['Nz'] = v\n");

  { // output  
    PetscInt dof[4],stencil_width;
    PetscCall(DMStagGetDOF(dm,&dof[0],&dof[1],&dof[2],&dof[3]));
    PetscCall(DMStagGetStencilWidth(dm,&stencil_width));

    PetscCall(PetscViewerBinaryWrite(v,(void*)&dim,1,PETSC_INT));
    PetscCall(PetscViewerBinaryWrite(v,(void*)&dof[0],1,PETSC_INT));
    PetscCall(PetscViewerBinaryWrite(v,(void*)&dof[1],1,PETSC_INT));
    PetscCall(PetscViewerBinaryWrite(v,(void*)&dof[2],1,PETSC_INT));
    PetscCall(PetscViewerBinaryWrite(v,(void*)&dof[3],1,PETSC_INT));
    PetscCall(PetscViewerBinaryWrite(v,(void*)&stencil_width,1,PETSC_INT));
    
    pythonemit(fp,"    v = io.readInteger(fp)\n"); pythonemit(fp,"    data['dim'] = v\n");
    pythonemit(fp,"    v = io.readInteger(fp)\n"); pythonemit(fp,"    data['dof0'] = v\n");
    pythonemit(fp,"    v = io.readInteger(fp)\n"); pythonemit(fp,"    data['dof1'] = v\n");
    pythonemit(fp,"    v = io.readInteger(fp)\n"); pythonemit(fp,"    data['dof2'] = v\n");
    pythonemit(fp,"    v = io.readInteger(fp)\n"); pythonemit(fp,"    data['dof3'] = v\n");
    pythonemit(fp,"    v = io.readInteger(fp)\n"); pythonemit(fp,"    data['stencil_width'] = v\n");
  }
  
  if (view_coords) {
    DM cdm,subDM;
    PetscBool isProduct;
    Vec coor;
    DM pda;
    Vec subX;
    PetscInt dof[4];
    
    PetscCall(DMGetCoordinateDM(dm,&cdm));
    PetscCall(PetscObjectTypeCompare((PetscObject)cdm,DMPRODUCT,&isProduct));
    if (isProduct) {
      if (dim >= 1) {
        PetscCall(DMProductGetDM(cdm,0,&subDM));
        PetscCall(DMGetCoordinates(subDM,&coor));
        PetscCall(VecView(coor,v));
        pythonemitvec(fp,"x1d");
        
        PetscCall(DMStagGetDOF(subDM,&dof[0],&dof[1],&dof[2],&dof[3]));

        // no need to check for dofs - output both vertex and center coordinates
        PetscCall(DMStagVecSplitToDMDA(subDM,coor,DMSTAG_LEFT,-dof[0],&pda,&subX));
        PetscCall(VecView(subX,v));
        pythonemitvec(fp,"x1d_vertex");
        PetscCall(VecDestroy(&subX));
        PetscCall(DMDestroy(&pda));

        PetscCall(DMStagVecSplitToDMDA(subDM,coor,DMSTAG_ELEMENT,-dof[1],&pda,&subX));
        PetscCall(VecView(subX,v));
        pythonemitvec(fp,"x1d_cell");
        PetscCall(VecDestroy(&subX));
        PetscCall(DMDestroy(&pda));
      }
      if (dim >= 2) {
        PetscCall(DMProductGetDM(cdm,1,&subDM));
        PetscCall(DMGetCoordinates(subDM,&coor));
        PetscCall(VecView(coor,v));
        pythonemitvec(fp,"y1d");
        
        PetscCall(DMStagGetDOF(subDM,&dof[0],&dof[1],&dof[2],&dof[3]));

        PetscCall(DMStagVecSplitToDMDA(subDM,coor,DMSTAG_LEFT,-dof[0],&pda,&subX));
        PetscCall(VecView(subX,v));
        pythonemitvec(fp,"y1d_vertex");
        PetscCall(VecDestroy(&subX));
        PetscCall(DMDestroy(&pda));

        PetscCall(DMStagVecSplitToDMDA(subDM,coor,DMSTAG_ELEMENT,-dof[1],&pda,&subX));
        PetscCall(VecView(subX,v));
        pythonemitvec(fp,"y1d_cell");
        PetscCall(VecDestroy(&subX));
        PetscCall(DMDestroy(&pda));
      }
      if (dim == 3) {
        PetscCall(DMProductGetDM(cdm,2,&subDM));
        PetscCall(DMGetCoordinates(subDM,&coor));
        PetscCall(VecView(coor,v));
        pythonemitvec(fp,"z1d");
        
        PetscCall(DMStagGetDOF(subDM,&dof[0],&dof[1],&dof[2],&dof[3]));

        PetscCall(DMStagVecSplitToDMDA(subDM,coor,DMSTAG_LEFT,-dof[0],&pda,&subX));
        PetscCall(VecView(subX,v));
        pythonemitvec(fp,"z1d_vertex");
        PetscCall(VecDestroy(&subX));
        PetscCall(DMDestroy(&pda));

        PetscCall(DMStagVecSplitToDMDA(subDM,coor,DMSTAG_ELEMENT,-dof[1],&pda,&subX));
        PetscCall(VecView(subX,v));
        pythonemitvec(fp,"z1d_cell");
        PetscCall(VecDestroy(&subX));
        PetscCall(DMDestroy(&pda));

      }
    } else SETERRQ(comm,PETSC_ERR_SUP,"Only supports coordinated defined via DMPRODUCT");
  }
  
  {
    DM pda;
    Vec subX;
    PetscInt dof[4];
    
    PetscCall(DMStagGetDOF(dm,&dof[0],&dof[1],&dof[2],&dof[3]));

    if (dim == 1) {
      if (dof[0] != 0) {
        PetscCall(DMStagVecSplitToDMDA(dm,X,DMSTAG_DOWN_LEFT,-dof[0],&pda,&subX));
        PetscCall(VecView(subX,v));
        pythonemitvec(fp,"X_vertex");
        PetscCall(VecDestroy(&subX));
        PetscCall(DMDestroy(&pda));
      }
      if (dof[1] != 0) {
        PetscCall(DMStagVecSplitToDMDA(dm,X,DMSTAG_ELEMENT,-dof[1],&pda,&subX));
        PetscCall(VecView(subX,v));
        pythonemitvec(fp,"X_cell");
        PetscCall(VecDestroy(&subX));
        PetscCall(DMDestroy(&pda));
      }
    } else if (dim == 2) {
      if (dof[0] != 0) {
        PetscCall(DMStagVecSplitToDMDA(dm,X,DMSTAG_DOWN_LEFT,-dof[0],&pda,&subX));
        PetscCall(VecView(subX,v));
        pythonemitvec(fp,"X_vertex");
        PetscCall(VecDestroy(&subX));
        PetscCall(DMDestroy(&pda));
      }
      if (dof[1] != 0) {
        PetscCall(DMStagVecSplitToDMDA(dm,X,DMSTAG_LEFT,-dof[1],&pda,&subX));
        PetscCall(VecView(subX,v));
        pythonemitvec(fp,"X_face_x");
        PetscCall(VecDestroy(&subX));
        PetscCall(DMDestroy(&pda));
        
        PetscCall(DMStagVecSplitToDMDA(dm,X,DMSTAG_DOWN,-dof[1],&pda,&subX));
        PetscCall(VecView(subX,v));
        pythonemitvec(fp,"X_face_y");
        PetscCall(VecDestroy(&subX));
        PetscCall(DMDestroy(&pda));
      }
      if (dof[2] != 0) {
        PetscCall(DMStagVecSplitToDMDA(dm,X,DMSTAG_ELEMENT,-dof[2],&pda,&subX));
        PetscCall(VecView(subX,v));
        pythonemitvec(fp,"X_cell");
        PetscCall(VecDestroy(&subX));
        PetscCall(DMDestroy(&pda));
      }
    } else if (dim == 3) {
      if (dof[0] != 0) {
        PetscCall(DMStagVecSplitToDMDA(dm,X,DMSTAG_DOWN_LEFT,-dof[0],&pda,&subX));
        PetscCall(VecView(subX,v));
        pythonemitvec(fp,"X_vertex");
        PetscCall(VecDestroy(&subX));
        PetscCall(DMDestroy(&pda));
      }
      if (dof[1] != 0) SETERRQ(comm,PETSC_ERR_SUP,"No support for edge data (3D)");
      if (dof[2] != 0) {
        PetscCall(DMStagVecSplitToDMDA(dm,X,DMSTAG_LEFT,-dof[2],&pda,&subX));
        PetscCall(VecView(subX,v));
        pythonemitvec(fp,"X_face_x");
        PetscCall(VecDestroy(&subX));
        PetscCall(DMDestroy(&pda));
        
        PetscCall(DMStagVecSplitToDMDA(dm,X,DMSTAG_DOWN,-dof[2],&pda,&subX));
        PetscCall(VecView(subX,v));
        pythonemitvec(fp,"X_face_y");
        PetscCall(VecDestroy(&subX));
        PetscCall(DMDestroy(&pda));
        
        PetscCall(DMStagVecSplitToDMDA(dm,X,DMSTAG_BACK,-dof[2],&pda,&subX));
        PetscCall(VecView(subX,v));
        pythonemitvec(fp,"X_face_z");
        PetscCall(VecDestroy(&subX));
        PetscCall(DMDestroy(&pda));
      }
      if (dof[3] != 0) {
        PetscCall(DMStagVecSplitToDMDA(dm,X,DMSTAG_ELEMENT,-dof[3],&pda,&subX));
        PetscCall(VecView(subX,v));
        pythonemitvec(fp,"X_cell");
        PetscCall(VecDestroy(&subX));
        PetscCall(DMDestroy(&pda));
      }
    }
  }
  
  pythonemit(fp,"    return data\n\n");
  
  pythonemit(fp,"def _PETScBinaryLoadReportNames(data):\n");
  PetscCall(PetscSNPrintf(string,PETSC_MAX_PATH_LEN-1,"  print('Filename: %s')\n",fname));
  pythonemit(fp,string);
  pythonemit(fp,"  print('Contents:')\n");
  pythonemit(fp,"  for key in data:\n");
  pythonemit(fp,"    print('  textual name registered:',key)\n\n");
  
  pythonemit(fp,"def demo_load_report():\n");
  pythonemit(fp," data = _PETScBinaryLoad()\n");
  pythonemit(fp," _PETScBinaryLoadReportNames(data)\n");
  
  PetscCall(PetscViewerDestroy(&v));
  fclose(fp);
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
/*@
DMStagViewBinaryPython_MPI - MPI output routine for DMStagViewBinaryPython()

Use: internal
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "DMStagViewBinaryPython_MPI"
PetscErrorCode DMStagViewBinaryPython_MPI(DM dm,Vec X,const char prefix[])
{
  PetscViewer v;
  PetscInt M,N,P,dim;
  FILE *fp = NULL;
  char fname[PETSC_MAX_PATH_LEN],string[PETSC_MAX_PATH_LEN];
  MPI_Comm comm;
  PetscMPIInt rank;
  PetscBool view_coords = PETSC_TRUE; /* ultimately this would be an input arg */
  PetscFunctionBegin;
  
  comm = PetscObjectComm((PetscObject)dm);
  PetscCall(MPI_Comm_rank(comm,&rank)); 
  
  /* check for instances of "." in the file name so that the file can be imported */
  {
    size_t k,len;
    PetscCall(PetscStrlen(prefix,&len));
    for (k=0; k<len; k++) if (prefix[k] == '.') PetscCall(PetscPrintf(comm,"[DMStagViewBinaryPython_SEQ] Warning: prefix %s contains the symbol '.'. Hence you will not be able to import the emiited python script. Consider change the prefix\n",prefix));
  }
  
  PetscCall(PetscSNPrintf(fname,PETSC_MAX_PATH_LEN-1,"%s.pbin",prefix));
  PetscCall(PetscViewerBinaryOpen(comm,fname,FILE_MODE_WRITE,&v));
  
  PetscCall(PetscSNPrintf(string,PETSC_MAX_PATH_LEN-1,"%s.py",prefix));
  
  if (rank == 0) {
    fp = fopen(string,"w");
    if (!fp) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Cannot open file %s",string);
  }
  pythonemit(fp,"import PetscBinaryIO as pio\n");
  pythonemit(fp,"import numpy as np\n\n");
  
  pythonemit(fp,"def _PETScBinaryFilePrefix():\n");
  PetscSNPrintf(string,PETSC_MAX_PATH_LEN-1,"  return \"%s\"\n",prefix);
  pythonemit(fp,string);
  pythonemit(fp,"\n");
  
  pythonemit(fp,"def _PETScBinaryLoad():\n");
  pythonemit(fp,"  io = pio.PetscBinaryIO()\n");
  
  PetscSNPrintf(string,PETSC_MAX_PATH_LEN-1,"  filename = \"%s\"\n",fname);
  pythonemit(fp,string);
  pythonemit(fp,"  data = dict()\n");
  pythonemit(fp,"  with open(filename) as fp:\n");
  
  PetscCall(DMGetDimension(dm,&dim));
  PetscCall(DMStagGetGlobalSizes(dm,&M,&N,&P));
  
  PetscCall(PetscViewerBinaryWrite(v,(void*)&M,1,PETSC_INT));
  PetscCall(PetscViewerBinaryWrite(v,(void*)&N,1,PETSC_INT));
  PetscCall(PetscViewerBinaryWrite(v,(void*)&P,1,PETSC_INT));
  
  pythonemit(fp,"    v = io.readInteger(fp)\n"); pythonemit(fp,"    data['Nx'] = v\n");
  pythonemit(fp,"    v = io.readInteger(fp)\n"); pythonemit(fp,"    data['Ny'] = v\n");
  pythonemit(fp,"    v = io.readInteger(fp)\n"); pythonemit(fp,"    data['Nz'] = v\n");

  { // output  
    PetscInt dof[4],stencil_width;
    PetscCall(DMStagGetDOF(dm,&dof[0],&dof[1],&dof[2],&dof[3]));
    PetscCall(DMStagGetStencilWidth(dm,&stencil_width));

    PetscCall(PetscViewerBinaryWrite(v,(void*)&dim,1,PETSC_INT));
    PetscCall(PetscViewerBinaryWrite(v,(void*)&dof[0],1,PETSC_INT));
    PetscCall(PetscViewerBinaryWrite(v,(void*)&dof[1],1,PETSC_INT));
    PetscCall(PetscViewerBinaryWrite(v,(void*)&dof[2],1,PETSC_INT));
    PetscCall(PetscViewerBinaryWrite(v,(void*)&dof[3],1,PETSC_INT));
    PetscCall(PetscViewerBinaryWrite(v,(void*)&stencil_width,1,PETSC_INT));
    
    pythonemit(fp,"    v = io.readInteger(fp)\n"); pythonemit(fp,"    data['dim'] = v\n");
    pythonemit(fp,"    v = io.readInteger(fp)\n"); pythonemit(fp,"    data['dof0'] = v\n");
    pythonemit(fp,"    v = io.readInteger(fp)\n"); pythonemit(fp,"    data['dof1'] = v\n");
    pythonemit(fp,"    v = io.readInteger(fp)\n"); pythonemit(fp,"    data['dof2'] = v\n");
    pythonemit(fp,"    v = io.readInteger(fp)\n"); pythonemit(fp,"    data['dof3'] = v\n");
    pythonemit(fp,"    v = io.readInteger(fp)\n"); pythonemit(fp,"    data['stencil_width'] = v\n");
  }
  
  if (view_coords) {
    DM cdm,subDM;
    PetscBool isProduct;
    Vec coor;
    DM pda;
    Vec subX;
    PetscInt dof[4],Mp,Np,Pp,ip,jp,kp;
    PetscMPIInt rank_1;
    PetscBool active;
    
    PetscCall(DMGetCoordinateDM(dm,&cdm));
    PetscCall(DMStagGetNumRanks(dm,&Mp,&Np,&Pp));
    PetscCall(PetscObjectTypeCompare((PetscObject)cdm,DMPRODUCT,&isProduct));
    if (isProduct) {
            
      if (dim >= 1) {
        PetscInt mlocal;
        Vec coorn;
        
        PetscCall(DMProductGetDM(cdm,0,&subDM));
        PetscCall(DMGetCoordinates(subDM,&coor));
        
        active = PETSC_FALSE;
        jp = 0;
        kp = 0;
        for (ip=0; ip<Mp; ip++) {
          rank_1 = ip + jp * Mp + kp * Mp * Np;
          if (rank_1 == rank) { active = PETSC_TRUE; break; }
        }
        
        mlocal = 0;
        if (active) {
          PetscCall(VecGetLocalSize(coor,&mlocal));
        }
        {
          const PetscScalar *LA_c = NULL;
          
          PetscCall(VecCreateMPIWithArray(comm,1,mlocal,PETSC_DECIDE,NULL,&coorn));
          if (active) {
            PetscCall(VecGetArrayRead(coor,&LA_c));
            PetscCall(VecPlaceArray(coorn,LA_c));
            PetscCall(VecRestoreArrayRead(coor,&LA_c));
          }
        }

        PetscCall(VecView(coorn,v));
        pythonemitvec(fp,"x1d");
        PetscCall(VecDestroy(&coorn));
        
        PetscCall(DMStagGetDOF(subDM,&dof[0],&dof[1],&dof[2],&dof[3]));

        // no need to check for dofs - output both vertex and center coordinates
        {
          PetscCall(DMStagVecSplitToDMDA(subDM,coor,DMSTAG_LEFT,-dof[0],&pda,&subX));
          
          mlocal = 0;
          if (active) {
            PetscCall(VecGetLocalSize(subX,&mlocal));
          }
          {
            const PetscScalar *LA_c = NULL;
            
            PetscCall(VecCreateMPIWithArray(comm,1,mlocal,PETSC_DECIDE,NULL,&coorn));
            if (active) {
              PetscCall(VecGetArrayRead(subX,&LA_c));
              PetscCall(VecPlaceArray(coorn,LA_c));
              PetscCall(VecRestoreArrayRead(subX,&LA_c));
            }
          }
          PetscCall(VecView(coorn,v));
          pythonemitvec(fp,"x1d_vertex");
          PetscCall(VecDestroy(&coorn));
          PetscCall(VecDestroy(&subX));
          PetscCall(DMDestroy(&pda));
        }
        
        {
          PetscCall(DMStagVecSplitToDMDA(subDM,coor,DMSTAG_ELEMENT,-dof[1],&pda,&subX));

          mlocal = 0;
          if (active) {
            PetscCall(VecGetLocalSize(subX,&mlocal));
          }
          {
            const PetscScalar *LA_c = NULL;
            
            PetscCall(VecCreateMPIWithArray(comm,1,mlocal,PETSC_DECIDE,NULL,&coorn));
            if (active) {
              PetscCall(VecGetArrayRead(subX,&LA_c));
              PetscCall(VecPlaceArray(coorn,LA_c));
              PetscCall(VecRestoreArrayRead(subX,&LA_c));
            }
          }
          PetscCall(VecView(coorn,v));
          pythonemitvec(fp,"x1d_cell");
          PetscCall(VecDestroy(&coorn));
          PetscCall(VecDestroy(&subX));
          PetscCall(DMDestroy(&pda));
        }
      }
      
      
      if (dim >= 2) {
        PetscInt mlocal;
        Vec coorn;

        PetscCall(DMProductGetDM(cdm,1,&subDM));
        PetscCall(DMGetCoordinates(subDM,&coor));

        active = PETSC_FALSE;
        ip = 0;
        kp = 0;
        for (jp=0; jp<Np; jp++) {
          rank_1 = ip + jp * Mp + kp * Mp * Np;
          if (rank_1 == rank) { active = PETSC_TRUE; break; }
        }
        
        mlocal = 0;
        if (active) {
          PetscCall(VecGetLocalSize(coor,&mlocal));
        }
        {
          const PetscScalar *LA_c = NULL;
          
          PetscCall(VecCreateMPIWithArray(comm,1,mlocal,PETSC_DECIDE,NULL,&coorn));
          if (active) {
            PetscCall(VecGetArrayRead(coor,&LA_c));
            PetscCall(VecPlaceArray(coorn,LA_c));
            PetscCall(VecRestoreArrayRead(coor,&LA_c));
          }
        }
        PetscCall(VecView(coorn,v));
        pythonemitvec(fp,"y1d");
        PetscCall(VecDestroy(&coorn));

        PetscCall(DMStagGetDOF(subDM,&dof[0],&dof[1],&dof[2],&dof[3]));
        {
          PetscCall(DMStagVecSplitToDMDA(subDM,coor,DMSTAG_LEFT,-dof[0],&pda,&subX));

          mlocal = 0;
          if (active) {
            PetscCall(VecGetLocalSize(subX,&mlocal));
          }
          {
            const PetscScalar *LA_c = NULL;
            
            PetscCall(VecCreateMPIWithArray(comm,1,mlocal,PETSC_DECIDE,NULL,&coorn));
            if (active) {
              PetscCall(VecGetArrayRead(subX,&LA_c));
              PetscCall(VecPlaceArray(coorn,LA_c));
              PetscCall(VecRestoreArrayRead(subX,&LA_c));
            }
          }
          PetscCall(VecView(coorn,v));

          pythonemitvec(fp,"y1d_vertex");
          PetscCall(VecDestroy(&coorn));
          PetscCall(VecDestroy(&subX));
          PetscCall(DMDestroy(&pda));
        }
        
        {
          PetscCall(DMStagVecSplitToDMDA(subDM,coor,DMSTAG_ELEMENT,-dof[1],&pda,&subX));
          
          mlocal = 0;
          if (active) {
            PetscCall(VecGetLocalSize(subX,&mlocal));
          }
          {
            const PetscScalar *LA_c = NULL;
            
            PetscCall(VecCreateMPIWithArray(comm,1,mlocal,PETSC_DECIDE,NULL,&coorn));
            if (active) {
              PetscCall(VecGetArrayRead(subX,&LA_c));
              PetscCall(VecPlaceArray(coorn,LA_c));
              PetscCall(VecRestoreArrayRead(subX,&LA_c));
            }
          }
          PetscCall(VecView(coorn,v));

          pythonemitvec(fp,"y1d_cell");
          PetscCall(VecDestroy(&coorn));
          PetscCall(VecDestroy(&subX));
          PetscCall(DMDestroy(&pda));
        }
      }
      
      
      if (dim == 3) {
        PetscInt mlocal;
        Vec coorn;

        PetscCall(DMProductGetDM(cdm,2,&subDM));
        PetscCall(DMGetCoordinates(subDM,&coor));
        
        active = PETSC_FALSE;
        ip = 0;
        jp = 0;
        for (kp=0; kp<Pp; kp++) {
          rank_1 = ip + jp * Mp + kp * Mp * Np;
          if (rank_1 == rank) { active = PETSC_TRUE; break; }
        }
        
        mlocal = 0;
        if (active) {
          PetscCall(VecGetLocalSize(coor,&mlocal));
        }
        {
          const PetscScalar *LA_c = NULL;
          
          PetscCall(VecCreateMPIWithArray(comm,1,mlocal,PETSC_DECIDE,NULL,&coorn));
          if (active) {
            PetscCall(VecGetArrayRead(coor,&LA_c));
            PetscCall(VecPlaceArray(coorn,LA_c));
            PetscCall(VecRestoreArrayRead(coor,&LA_c));
          }
        }
        PetscCall(VecView(coorn,v));
        pythonemitvec(fp,"z1d");
        PetscCall(VecDestroy(&coorn));
        
        PetscCall(DMStagGetDOF(subDM,&dof[0],&dof[1],&dof[2],&dof[3]));
        {
          PetscCall(DMStagVecSplitToDMDA(subDM,coor,DMSTAG_LEFT,-dof[0],&pda,&subX));

          mlocal = 0;
          if (active) {
            PetscCall(VecGetLocalSize(subX,&mlocal));
          }
          {
            const PetscScalar *LA_c = NULL;
            
            PetscCall(VecCreateMPIWithArray(comm,1,mlocal,PETSC_DECIDE,NULL,&coorn));
            if (active) {
              PetscCall(VecGetArrayRead(subX,&LA_c));
              PetscCall(VecPlaceArray(coorn,LA_c));
              PetscCall(VecRestoreArrayRead(subX,&LA_c));
            }
          }
          PetscCall(VecView(coorn,v));

          pythonemitvec(fp,"z1d_vertex");
          PetscCall(VecDestroy(&coorn));
          PetscCall(VecDestroy(&subX));
          PetscCall(DMDestroy(&pda));
        }
        
        {
          PetscCall(DMStagVecSplitToDMDA(subDM,coor,DMSTAG_ELEMENT,-dof[1],&pda,&subX));

          mlocal = 0;
          if (active) {
            PetscCall(VecGetLocalSize(subX,&mlocal));
          }
          {
            const PetscScalar *LA_c = NULL;
            
            PetscCall(VecCreateMPIWithArray(comm,1,mlocal,PETSC_DECIDE,NULL,&coorn));
            if (active) {
              PetscCall(VecGetArrayRead(subX,&LA_c));
              PetscCall(VecPlaceArray(coorn,LA_c));
              PetscCall(VecRestoreArrayRead(subX,&LA_c));
            }
          }
          PetscCall(VecView(coorn,v));

          pythonemitvec(fp,"z1d_cell");
          PetscCall(VecDestroy(&coorn));
          PetscCall(VecDestroy(&subX));
          PetscCall(DMDestroy(&pda));
        }

      }
    } else SETERRQ(comm,PETSC_ERR_SUP,"Only supports coordinated defined via DMPRODUCT");
  }
  
  {
    DM pda;
    Vec subX,subXn;
    PetscInt dof[4];
    
    PetscCall(DMStagGetDOF(dm,&dof[0],&dof[1],&dof[2],&dof[3]));
    
    if (dim == 1) {
      if (dof[0] != 0) {
        PetscCall(DMStagVecSplitToDMDA(dm,X,DMSTAG_DOWN_LEFT,-dof[0],&pda,&subX));
        PetscCall(DMDACreateNaturalVector(pda,&subXn));
        PetscCall(DMDAGlobalToNaturalBegin(pda,subX,INSERT_VALUES,subXn));
        PetscCall(DMDAGlobalToNaturalEnd(pda,subX,INSERT_VALUES,subXn));
        PetscCall(VecView(subXn,v));
        pythonemitvec(fp,"X_vertex");
        PetscCall(VecDestroy(&subXn));
        PetscCall(VecDestroy(&subX));
        PetscCall(DMDestroy(&pda));
      }
      if (dof[1] != 0) {
        PetscCall(DMStagVecSplitToDMDA(dm,X,DMSTAG_ELEMENT,-dof[1],&pda,&subX));
        PetscCall(DMDACreateNaturalVector(pda,&subXn));
        PetscCall(DMDAGlobalToNaturalBegin(pda,subX,INSERT_VALUES,subXn));
        PetscCall(DMDAGlobalToNaturalEnd(pda,subX,INSERT_VALUES,subXn));
        PetscCall(VecView(subXn,v));
        pythonemitvec(fp,"X_cell");
        PetscCall(VecDestroy(&subXn));
        PetscCall(VecDestroy(&subX));
        PetscCall(DMDestroy(&pda));
      }
    } else if (dim == 2) {
      if (dof[0] != 0) {
        PetscCall(DMStagVecSplitToDMDA(dm,X,DMSTAG_DOWN_LEFT,-dof[0],&pda,&subX));
        PetscCall(DMDACreateNaturalVector(pda,&subXn));
        PetscCall(DMDAGlobalToNaturalBegin(pda,subX,INSERT_VALUES,subXn));
        PetscCall(DMDAGlobalToNaturalEnd(pda,subX,INSERT_VALUES,subXn));
        PetscCall(VecView(subXn,v));
        pythonemitvec(fp,"X_vertex");
        PetscCall(VecDestroy(&subXn));
        PetscCall(VecDestroy(&subX));
        PetscCall(DMDestroy(&pda));
      }
      if (dof[1] != 0) {
        PetscCall(DMStagVecSplitToDMDA(dm,X,DMSTAG_LEFT,-dof[1],&pda,&subX));
        PetscCall(DMDACreateNaturalVector(pda,&subXn));
        PetscCall(DMDAGlobalToNaturalBegin(pda,subX,INSERT_VALUES,subXn));
        PetscCall(DMDAGlobalToNaturalEnd(pda,subX,INSERT_VALUES,subXn));
        PetscCall(VecView(subXn,v));
        pythonemitvec(fp,"X_face_x");
        PetscCall(VecDestroy(&subXn));
        PetscCall(VecDestroy(&subX));
        PetscCall(DMDestroy(&pda));
        
        PetscCall(DMStagVecSplitToDMDA(dm,X,DMSTAG_DOWN,-dof[1],&pda,&subX));
        PetscCall(DMDACreateNaturalVector(pda,&subXn));
        PetscCall(DMDAGlobalToNaturalBegin(pda,subX,INSERT_VALUES,subXn));
        PetscCall(DMDAGlobalToNaturalEnd(pda,subX,INSERT_VALUES,subXn));
        PetscCall(VecView(subXn,v));
        pythonemitvec(fp,"X_face_y");
        PetscCall(VecDestroy(&subXn));
        PetscCall(VecDestroy(&subX));
        PetscCall(DMDestroy(&pda));
      }
      if (dof[2] != 0) {
        PetscCall(DMStagVecSplitToDMDA(dm,X,DMSTAG_ELEMENT,-dof[2],&pda,&subX));
        PetscCall(DMDACreateNaturalVector(pda,&subXn));
        PetscCall(DMDAGlobalToNaturalBegin(pda,subX,INSERT_VALUES,subXn));
        PetscCall(DMDAGlobalToNaturalEnd(pda,subX,INSERT_VALUES,subXn));
        PetscCall(VecView(subXn,v));
        pythonemitvec(fp,"X_cell");
        PetscCall(VecDestroy(&subXn));
        PetscCall(VecDestroy(&subX));
        PetscCall(DMDestroy(&pda));
      }
    } else if (dim == 3) {
      if (dof[0] != 0) {
        PetscCall(DMStagVecSplitToDMDA(dm,X,DMSTAG_DOWN_LEFT,-dof[0],&pda,&subX));
        PetscCall(DMDACreateNaturalVector(pda,&subXn));
        PetscCall(DMDAGlobalToNaturalBegin(pda,subX,INSERT_VALUES,subXn));
        PetscCall(DMDAGlobalToNaturalEnd(pda,subX,INSERT_VALUES,subXn));
        PetscCall(VecView(subXn,v));
        pythonemitvec(fp,"X_vertex");
        PetscCall(VecDestroy(&subXn));
        PetscCall(VecDestroy(&subX));
        PetscCall(DMDestroy(&pda));
      }
      if (dof[1] != 0) SETERRQ(comm,PETSC_ERR_SUP,"No support for edge data (3D)");
      if (dof[2] != 0) {
        PetscCall(DMStagVecSplitToDMDA(dm,X,DMSTAG_LEFT,-dof[2],&pda,&subX));
        PetscCall(DMDACreateNaturalVector(pda,&subXn));
        PetscCall(DMDAGlobalToNaturalBegin(pda,subX,INSERT_VALUES,subXn));
        PetscCall(DMDAGlobalToNaturalEnd(pda,subX,INSERT_VALUES,subXn));
        PetscCall(VecView(subXn,v));
        pythonemitvec(fp,"X_face_x");
        PetscCall(VecDestroy(&subXn));
        PetscCall(VecDestroy(&subX));
        PetscCall(DMDestroy(&pda));
        
        PetscCall(DMStagVecSplitToDMDA(dm,X,DMSTAG_DOWN,-dof[2],&pda,&subX));
        PetscCall(DMDACreateNaturalVector(pda,&subXn));
        PetscCall(DMDAGlobalToNaturalBegin(pda,subX,INSERT_VALUES,subXn));
        PetscCall(DMDAGlobalToNaturalEnd(pda,subX,INSERT_VALUES,subXn));
        PetscCall(VecView(subXn,v));
        pythonemitvec(fp,"X_face_y");
        PetscCall(VecDestroy(&subXn));
        PetscCall(VecDestroy(&subX));
        PetscCall(DMDestroy(&pda));
        
        PetscCall(DMStagVecSplitToDMDA(dm,X,DMSTAG_BACK,-dof[2],&pda,&subX));
        PetscCall(DMDACreateNaturalVector(pda,&subXn));
        PetscCall(DMDAGlobalToNaturalBegin(pda,subX,INSERT_VALUES,subXn));
        PetscCall(DMDAGlobalToNaturalEnd(pda,subX,INSERT_VALUES,subXn));
        PetscCall(VecView(subXn,v));
        pythonemitvec(fp,"X_face_z");
        PetscCall(VecDestroy(&subXn));
        PetscCall(VecDestroy(&subX));
        PetscCall(DMDestroy(&pda));
      }
      if (dof[3] != 0) {
        PetscCall(DMStagVecSplitToDMDA(dm,X,DMSTAG_ELEMENT,-dof[3],&pda,&subX));
        PetscCall(DMDACreateNaturalVector(pda,&subXn));
        PetscCall(DMDAGlobalToNaturalBegin(pda,subX,INSERT_VALUES,subXn));
        PetscCall(DMDAGlobalToNaturalEnd(pda,subX,INSERT_VALUES,subXn));
        PetscCall(VecView(subXn,v));
        pythonemitvec(fp,"X_cell");
        PetscCall(VecDestroy(&subXn));
        PetscCall(VecDestroy(&subX));
        PetscCall(DMDestroy(&pda));
      }
    }
  }
  
  pythonemit(fp,"  return data\n\n");
  
  pythonemit(fp,"def _PETScBinaryLoadReportNames(data):\n");
  PetscCall(PetscSNPrintf(string,PETSC_MAX_PATH_LEN-1,"  print('Filename: %s')\n",fname));
  pythonemit(fp,string);
  pythonemit(fp,"  print('Contents:')\n");
  pythonemit(fp,"  for key in data:\n");
  pythonemit(fp,"    print('  textual name registered:',key)\n\n");
  
  pythonemit(fp,"def demo_load_report():\n");
  pythonemit(fp," data = _PETScBinaryLoad()\n");
  pythonemit(fp," _PETScBinaryLoadReportNames(data)\n");
  
  PetscCall(PetscViewerDestroy(&v));
  if (fp) fclose(fp);
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
/*@
DMStagViewBinaryPython - output a DMStag and associated vector as PetscBinary to be read in python

Input parameters:
dm - the DMStag object
X  - associated vector with dm
prefix - output name (no file extension)

Use: user
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "DMStagViewBinaryPython"
PetscErrorCode DMStagViewBinaryPython(DM dm,Vec X,const char prefix[])
{
  MPI_Comm comm;
  PetscMPIInt size;
  PetscFunctionBegin;
  
  comm = PetscObjectComm((PetscObject)dm);
  PetscCall(MPI_Comm_size(comm,&size)); 
  if (size == 1) {
    PetscCall(DMStagViewBinaryPython_SEQ(dm,X,prefix));
  } else {
    PetscCall(DMStagViewBinaryPython_MPI(dm,X,prefix));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "DMStagReadBinaryPython_SEQ"
PetscErrorCode DMStagReadBinaryPython_SEQ(DM *_dm,Vec *_x,const char prefix[])
{
  DM  dm;
  Vec x;
  PetscViewer v;
  char fname[PETSC_MAX_PATH_LEN];
  PetscInt M,N,P,dim,dof[4],stencil_width;
  MPI_Comm comm;
  PetscMPIInt size;
  PetscFunctionBegin;

  comm = PETSC_COMM_WORLD;
  PetscCall(MPI_Comm_size(comm,&size)); 
  if (size != 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Read is supported 'sequential' only");

  // this seems to be the fix for multiple dofs
  PetscCall(PetscOptionsSetValue(NULL, "-viewer_binary_skip_info", "")); 

  PetscCall(PetscSNPrintf(fname,PETSC_MAX_PATH_LEN-1,"%s.pbin",prefix));
  PetscCall(PetscViewerBinaryOpen(comm,fname,FILE_MODE_READ,&v));

  PetscCall(PetscViewerBinaryRead(v,(void*)&M,1,NULL,PETSC_INT));
  PetscCall(PetscViewerBinaryRead(v,(void*)&N,1,NULL,PETSC_INT));
  PetscCall(PetscViewerBinaryRead(v,(void*)&P,1,NULL,PETSC_INT));

  PetscCall(PetscViewerBinaryRead(v,(void*)&dim,1,NULL,PETSC_INT));
  PetscCall(PetscViewerBinaryRead(v,(void*)&dof[0],1,NULL,PETSC_INT));
  PetscCall(PetscViewerBinaryRead(v,(void*)&dof[1],1,NULL,PETSC_INT));
  PetscCall(PetscViewerBinaryRead(v,(void*)&dof[2],1,NULL,PETSC_INT));
  PetscCall(PetscViewerBinaryRead(v,(void*)&dof[3],1,NULL,PETSC_INT));
  PetscCall(PetscViewerBinaryRead(v,(void*)&stencil_width,1,NULL,PETSC_INT));

  // create dm and vector at this point
  if (dim==2) {
    PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,M,N,PETSC_DECIDE,PETSC_DECIDE, 
                        dof[0], dof[1], dof[2], DMSTAG_STENCIL_BOX,stencil_width, NULL,NULL, &dm));
    PetscCall(DMSetFromOptions(dm));
    PetscCall(DMSetUp(dm));
    PetscCall(DMStagSetUniformCoordinatesProduct(dm,0.0,1.0,0.0,1.0,0.0,0.0));
    PetscCall(DMCreateGlobalVector(dm,&x));
  } else {
    SETERRQ(comm,PETSC_ERR_SUP,"Only valid for 2d DM");
  }

  { // read coordinates
    DM cdm, subDM;
    Vec coor, subX;

    PetscCall(DMGetCoordinateDM(dm,&cdm));
    if (dim >= 1) {
      PetscCall(DMProductGetDM(cdm,0,&subDM));
      PetscCall(DMGetCoordinates(subDM,&coor));
      PetscCall(VecLoad(coor,v));  // x1d
      PetscCall(DMSetCoordinates(subDM,coor));

      // the following vectors are not needed
      PetscCall(VecCreate(comm,&subX));
      PetscCall(VecLoad(subX,v)); // x1d_vertex
      PetscCall(VecDestroy(&subX));

      PetscCall(VecCreate(comm,&subX));
      PetscCall(VecLoad(subX,v));  // x1d_cell
      PetscCall(VecDestroy(&subX));
    }
    if (dim >= 2) {
      PetscCall(DMProductGetDM(cdm,1,&subDM));
      PetscCall(DMGetCoordinates(subDM,&coor));
      PetscCall(VecLoad(coor,v));  // y1d
      PetscCall(DMSetCoordinates(subDM,coor));

      PetscCall(VecCreate(comm,&subX));
      PetscCall(VecLoad(subX,v)); // y1d_vertex
      PetscCall(VecDestroy(&subX)); 

      PetscCall(VecCreate(comm,&subX));
      PetscCall(VecLoad(subX,v)); // y1d_cell
      PetscCall(VecDestroy(&subX)); 
    }
    if (dim == 3) {
      PetscCall(DMProductGetDM(cdm,2,&subDM));
      PetscCall(DMGetCoordinates(subDM,&coor));
      PetscCall(VecLoad(coor,v));  // z1d
      PetscCall(DMSetCoordinates(subDM,coor));

      PetscCall(VecCreate(comm,&subX));
      PetscCall(VecLoad(subX,v)); // z1d_vertex
      PetscCall(VecDestroy(&subX)); 

      PetscCall(VecCreate(comm,&subX));
      PetscCall(VecLoad(subX,v)); // z1d_cell
      PetscCall(VecDestroy(&subX)); 
    }
  }

  { // read data
    DM pda;
    Vec subX;

    if (dim == 1) {
      SETERRQ(comm,PETSC_ERR_SUP,"Only valid for 2d DM");
    } else if (dim == 2) {

      PetscInt sx,sz,nx,nz,i,j,iloc,idof;
      Vec xlocal;
      PetscScalar ***xx;

      PetscCall(DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 
      
      if (dof[0] != 0) {
        PetscCall(DMStagVecSplitToDMDA(dm,x,DMSTAG_DOWN_LEFT,-dof[0],&pda,&subX));
        PetscCall(VecLoad(subX,v)); // X_vertex

        PetscCall(DMGetLocalVector(dm, &xlocal)); 
        PetscCall(DMGlobalToLocal (dm, x, INSERT_VALUES, xlocal)); 
        PetscCall(DMStagVecGetArray(dm, xlocal, &xx)); 

        // Loop over local domain
        for (idof = 0; idof < dof[0]; idof++) {
          PetscCall(DMStagGetLocationSlot(dm, DMSTAG_DOWN_LEFT, idof, &iloc)); 
          for (i = sx; i < sx+nx+1; i++) {
            for (j = sz; j <sz+nz+1; j++) {
              PetscScalar xval;
              PetscInt    ii;
              ii = (dof[0]*i+idof)+(dof[0]*j)*(M+1);
              PetscCall(VecGetValues(subX,1,&ii,&xval)); 
              xx[j][i][iloc] = xval; 
            }
          }
        }

        // Restore arrays
        PetscCall(DMStagVecRestoreArray(dm,xlocal,&xx)); 
        PetscCall(DMLocalToGlobalBegin(dm,xlocal,INSERT_VALUES,x)); 
        PetscCall(DMLocalToGlobalEnd  (dm,xlocal,INSERT_VALUES,x)); 
        PetscCall(DMRestoreLocalVector(dm, &xlocal)); 

        PetscCall(VecDestroy(&subX));
        PetscCall(DMDestroy(&pda));
      }
      if (dof[1] != 0) {
        PetscCall(DMStagVecSplitToDMDA(dm,x,DMSTAG_LEFT,-dof[1],&pda,&subX));
        PetscCall(VecLoad(subX,v)); // X_face_x

        PetscCall(DMGetLocalVector(dm, &xlocal)); 
        PetscCall(DMGlobalToLocal (dm, x, INSERT_VALUES, xlocal)); 
        PetscCall(DMStagVecGetArray(dm, xlocal, &xx)); 

        // Loop over local domain
        for (idof = 0; idof < dof[1]; idof++) {
          PetscCall(DMStagGetLocationSlot(dm, DMSTAG_LEFT, idof, &iloc)); 
          for (i = sx; i < sx+nx+1; i++) {
            for (j = sz; j <sz+nz; j++) {
              PetscScalar xval;
              PetscInt    ii;
              ii = (dof[1]*i+idof)+(dof[1]*j)*(M+1);
              PetscCall(VecGetValues(subX,1,&ii,&xval));  
              xx[j][i][iloc] = xval;
            }
          }
        }

        // Restore arrays
        PetscCall(DMStagVecRestoreArray(dm,xlocal,&xx)); 
        PetscCall(DMLocalToGlobalBegin(dm,xlocal,INSERT_VALUES,x)); 
        PetscCall(DMLocalToGlobalEnd  (dm,xlocal,INSERT_VALUES,x)); 
        PetscCall(DMRestoreLocalVector(dm, &xlocal)); 

        PetscCall(VecDestroy(&subX));
        PetscCall(DMDestroy(&pda));

        PetscCall(DMStagVecSplitToDMDA(dm,x,DMSTAG_DOWN,-dof[1],&pda,&subX));
        PetscCall(VecLoad(subX,v)); // X_face_y

        PetscCall(DMGetLocalVector(dm, &xlocal)); 
        PetscCall(DMGlobalToLocal (dm, x, INSERT_VALUES, xlocal)); 
        PetscCall(DMStagVecGetArray(dm, xlocal, &xx)); 

        // Loop over local domain
        for (idof = 0; idof < dof[1]; idof++) {
          PetscCall(DMStagGetLocationSlot(dm, DMSTAG_DOWN, idof, &iloc)); 
          for (j = sz; j < sz+nz+1; j++) {
            for (i = sx; i <sx+nx; i++) {
              PetscScalar xval;
              PetscInt    ii;
              ii = (dof[1]*i+idof)+(dof[1]*j)*M;
              PetscCall(VecGetValues(subX,1,&ii,&xval));  
              xx[j][i][iloc] = xval;
            }
          }
        }

        // Restore arrays
        PetscCall(DMStagVecRestoreArray(dm,xlocal,&xx)); 
        PetscCall(DMLocalToGlobalBegin(dm,xlocal,INSERT_VALUES,x)); 
        PetscCall(DMLocalToGlobalEnd  (dm,xlocal,INSERT_VALUES,x)); 
        PetscCall(DMRestoreLocalVector(dm, &xlocal)); 

        PetscCall(VecDestroy(&subX));
        PetscCall(DMDestroy(&pda));
      }
      if (dof[2] != 0) {
        PetscCall(DMStagVecSplitToDMDA(dm,x,DMSTAG_ELEMENT,-dof[2],&pda,&subX));
        PetscCall(VecLoad(subX,v)); // X_cell
        
        PetscCall(DMGetLocalVector(dm, &xlocal)); 
        PetscCall(DMGlobalToLocal (dm, x, INSERT_VALUES, xlocal)); 
        PetscCall(DMStagVecGetArray(dm, xlocal, &xx)); 

        // Loop over local domain
        for (idof = 0; idof < dof[2]; idof++) {
          PetscCall(DMStagGetLocationSlot(dm, DMSTAG_ELEMENT, idof, &iloc)); 
          for (j = sz; j <sz+nz; j++) {
            for (i = sx; i < sx+nx; i++) {
              PetscScalar xval;
              PetscInt    ii;
              ii = (dof[2]*i+idof)+(dof[2]*j)*M;
              PetscCall(VecGetValues(subX,1,&ii,&xval)); 
              xx[j][i][iloc] = xval; 
            }
          }
        }

        // Restore arrays
        PetscCall(DMStagVecRestoreArray(dm,xlocal,&xx)); 
        PetscCall(DMLocalToGlobalBegin(dm,xlocal,INSERT_VALUES,x)); 
        PetscCall(DMLocalToGlobalEnd  (dm,xlocal,INSERT_VALUES,x)); 
        PetscCall(DMRestoreLocalVector(dm, &xlocal)); 

        PetscCall(VecDestroy(&subX));
        PetscCall(DMDestroy(&pda));
      }
    } else if (dim == 3) {
      SETERRQ(comm,PETSC_ERR_SUP,"Only valid for 2d DM");
    }
  }

  PetscCall(PetscViewerDestroy(&v));

  *_dm = dm;
  *_x = x;

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
/*@
DMStagReadBinaryPython_MPI - MPI read routine for DMStagViewBinaryPython()

Use: internal
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "DMStagReadBinaryPython_MPI"
PetscErrorCode DMStagReadBinaryPython_MPI(DM *_dm,Vec *_x,const char prefix[])
{
  DM  dm;
  Vec x;
  PetscViewer v;
  char fname[PETSC_MAX_PATH_LEN];
  PetscInt M,N,P,dim,dof[4],stencil_width;
  MPI_Comm comm;
  PetscMPIInt rank;
  PetscMPIInt size;
  PetscFunctionBegin;

  comm = PETSC_COMM_WORLD;
  PetscCall(MPI_Comm_size(comm,&size)); 
  PetscCall(MPI_Comm_rank(comm,&rank)); 

  // this seems to be the fix for multiple dofs
  PetscCall(PetscOptionsSetValue(NULL, "-viewer_binary_skip_info", "")); 

  PetscCall(PetscSNPrintf(fname,PETSC_MAX_PATH_LEN-1,"%s.pbin",prefix));
  PetscCall(PetscViewerBinaryOpen(comm,fname,FILE_MODE_READ,&v));

  PetscCall(PetscViewerBinaryRead(v,(void*)&M,1,NULL,PETSC_INT));
  PetscCall(PetscViewerBinaryRead(v,(void*)&N,1,NULL,PETSC_INT));
  PetscCall(PetscViewerBinaryRead(v,(void*)&P,1,NULL,PETSC_INT));

  PetscCall(PetscViewerBinaryRead(v,(void*)&dim,1,NULL,PETSC_INT));
  PetscCall(PetscViewerBinaryRead(v,(void*)&dof[0],1,NULL,PETSC_INT));
  PetscCall(PetscViewerBinaryRead(v,(void*)&dof[1],1,NULL,PETSC_INT));
  PetscCall(PetscViewerBinaryRead(v,(void*)&dof[2],1,NULL,PETSC_INT));
  PetscCall(PetscViewerBinaryRead(v,(void*)&dof[3],1,NULL,PETSC_INT));
  PetscCall(PetscViewerBinaryRead(v,(void*)&stencil_width,1,NULL,PETSC_INT));

  // create dm and vector at this point
  if (dim==2) {
    PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,M,N,PETSC_DECIDE,PETSC_DECIDE, 
                        dof[0], dof[1], dof[2], DMSTAG_STENCIL_BOX,stencil_width, NULL,NULL, &dm));
    PetscCall(DMSetFromOptions(dm));
    PetscCall(DMSetUp(dm));
    PetscCall(DMStagSetUniformCoordinatesProduct(dm,0.0,1.0,0.0,1.0,0.0,0.0));
    PetscCall(DMCreateGlobalVector(dm,&x));
  } else {
    SETERRQ(comm,PETSC_ERR_SUP,"Only valid for 2d DM");
  }

  { // read coordinates
    DM cdm, subDM, pda;
    Vec coor, coorn,subX;
    PetscInt Mp,Np,Pp,mlocal,ip,jp,kp,sdof[4];
    PetscMPIInt rank_1;
    PetscBool active;

    PetscCall(DMGetCoordinateDM(dm,&cdm));
    PetscCall(DMStagGetNumRanks(dm,&Mp,&Np,&Pp));

    if (dim >= 1) {
      PetscCall(DMProductGetDM(cdm,0,&subDM));
      PetscCall(DMGetCoordinates(subDM,&coor));

      active = PETSC_FALSE;
      jp = 0;
      kp = 0;
      for (ip=0; ip<Mp; ip++) {
        rank_1 = ip + jp * Mp + kp * Mp * Np;
        if (rank_1 == rank) { active = PETSC_TRUE; break; }
      }

      // x1d
      mlocal = 0;
      if (active) { PetscCall(VecGetLocalSize(coor,&mlocal)); }
      {
        const PetscScalar *LA_c = NULL;
        PetscCall(VecCreateMPIWithArray(comm,1,mlocal,PETSC_DECIDE,NULL,&coorn));
        if (active) {
          PetscCall(VecGetArrayRead(coor,&LA_c));
          PetscCall(VecPlaceArray(coorn,LA_c));
          PetscCall(VecRestoreArrayRead(coor,&LA_c));
        }
      }
      PetscCall(VecLoad(coorn,v));  
      PetscCall(DMSetCoordinates(subDM,coor));
      PetscCall(VecDestroy(&coorn));

      // the following vectors only need to be read
      // x1d_vertex
      PetscCall(DMStagGetDOF(subDM,&sdof[0],&sdof[1],&sdof[2],&sdof[3]));
      PetscCall(DMStagVecSplitToDMDA(subDM,coor,DMSTAG_LEFT,-sdof[0],&pda,&subX));
      mlocal = 0;
      if (active) { PetscCall(VecGetLocalSize(subX,&mlocal)); }
      {
        const PetscScalar *LA_c = NULL;
        PetscCall(VecCreateMPIWithArray(comm,1,mlocal,PETSC_DECIDE,NULL,&coorn));
        if (active) {
          PetscCall(VecGetArrayRead(subX,&LA_c));
          PetscCall(VecPlaceArray(coorn,LA_c));
          PetscCall(VecRestoreArrayRead(subX,&LA_c));
        }
      }
      PetscCall(VecLoad(coorn,v)); 
      PetscCall(VecDestroy(&coorn));
      PetscCall(VecDestroy(&subX));
      PetscCall(DMDestroy(&pda));

      // x1d_cell
      PetscCall(DMStagVecSplitToDMDA(subDM,coor,DMSTAG_ELEMENT,-sdof[1],&pda,&subX));
      mlocal = 0;
      if (active) { PetscCall(VecGetLocalSize(subX,&mlocal)); }
      {
        const PetscScalar *LA_c = NULL;
        PetscCall(VecCreateMPIWithArray(comm,1,mlocal,PETSC_DECIDE,NULL,&coorn));
        if (active) {
          PetscCall(VecGetArrayRead(subX,&LA_c));
          PetscCall(VecPlaceArray(coorn,LA_c));
          PetscCall(VecRestoreArrayRead(subX,&LA_c));
        }
      }
      PetscCall(VecLoad(coorn,v));  
      PetscCall(VecDestroy(&coorn));
      PetscCall(VecDestroy(&subX));
      PetscCall(DMDestroy(&pda));
    }
    if (dim >= 2) {
      PetscCall(DMProductGetDM(cdm,1,&subDM));
      PetscCall(DMGetCoordinates(subDM,&coor));

      active = PETSC_FALSE;
      ip = 0;
      kp = 0;
      for (jp=0; jp<Np; jp++) {
        rank_1 = ip + jp * Mp + kp * Mp * Np;
        if (rank_1 == rank) { active = PETSC_TRUE; break; }
      }

      // y1d
      mlocal = 0;
      if (active) { PetscCall(VecGetLocalSize(coor,&mlocal)); }
      if (active) { PetscCall(VecGetLocalSize(coor,&mlocal)); }
      {
        const PetscScalar *LA_c = NULL;
        PetscCall(VecCreateMPIWithArray(comm,1,mlocal,PETSC_DECIDE,NULL,&coorn));
        if (active) {
          PetscCall(VecGetArrayRead(coor,&LA_c));
          PetscCall(VecPlaceArray(coorn,LA_c));
          PetscCall(VecRestoreArrayRead(coor,&LA_c));
        }
      }
      PetscCall(VecLoad(coorn,v));  
      PetscCall(DMSetCoordinates(subDM,coor));
      PetscCall(VecDestroy(&coorn));

      // y1d_vertex
      PetscCall(DMStagGetDOF(subDM,&sdof[0],&sdof[1],&sdof[2],&sdof[3]));
      PetscCall(DMStagVecSplitToDMDA(subDM,coor,DMSTAG_LEFT,-sdof[0],&pda,&subX));
      mlocal = 0;
      if (active) { PetscCall(VecGetLocalSize(subX,&mlocal)); }
      {
        const PetscScalar *LA_c = NULL;
        PetscCall(VecCreateMPIWithArray(comm,1,mlocal,PETSC_DECIDE,NULL,&coorn));
        if (active) {
          PetscCall(VecGetArrayRead(subX,&LA_c));
          PetscCall(VecPlaceArray(coorn,LA_c));
          PetscCall(VecRestoreArrayRead(subX,&LA_c));
        }
      }
      PetscCall(VecLoad(coorn,v));  
      PetscCall(VecDestroy(&coorn));
      PetscCall(VecDestroy(&subX));
      PetscCall(DMDestroy(&pda));

      // y1d_cell
      PetscCall(DMStagVecSplitToDMDA(subDM,coor,DMSTAG_ELEMENT,-sdof[1],&pda,&subX));
      mlocal = 0;
      if (active) { PetscCall(VecGetLocalSize(subX,&mlocal)); }
      {
        const PetscScalar *LA_c = NULL;
        PetscCall(VecCreateMPIWithArray(comm,1,mlocal,PETSC_DECIDE,NULL,&coorn));
        if (active) {
          PetscCall(VecGetArrayRead(subX,&LA_c));
          PetscCall(VecPlaceArray(coorn,LA_c));
          PetscCall(VecRestoreArrayRead(subX,&LA_c));
        }
      }
      PetscCall(VecLoad(coorn,v)); 
      PetscCall(VecDestroy(&coorn));
      PetscCall(VecDestroy(&subX));
      PetscCall(DMDestroy(&pda));
    }
    if (dim == 3) {
      PetscCall(DMProductGetDM(cdm,2,&subDM));
      PetscCall(DMGetCoordinates(subDM,&coor));

      active = PETSC_FALSE;
      ip = 0;
      jp = 0;
      for (kp=0; kp<Pp; kp++) {
        rank_1 = ip + jp * Mp + kp * Mp * Np;
        if (rank_1 == rank) { active = PETSC_TRUE; break; }
      }

      // z1d
      mlocal = 0;
      if (active) { PetscCall(VecGetLocalSize(coor,&mlocal)); }
      {
        const PetscScalar *LA_c = NULL;
        PetscCall(VecCreateMPIWithArray(comm,1,mlocal,PETSC_DECIDE,NULL,&coorn));
        if (active) {
          PetscCall(VecGetArrayRead(coor,&LA_c));
          PetscCall(VecPlaceArray(coorn,LA_c));
          PetscCall(VecRestoreArrayRead(coor,&LA_c));
        }
      }
      PetscCall(VecLoad(coorn,v));  
      PetscCall(DMSetCoordinates(subDM,coor));
      PetscCall(VecDestroy(&coorn));

      // z1d_vertex
      PetscCall(DMStagGetDOF(subDM,&sdof[0],&sdof[1],&sdof[2],&sdof[3]));
      PetscCall(DMStagVecSplitToDMDA(subDM,coor,DMSTAG_LEFT,-sdof[0],&pda,&subX));
      mlocal = 0;
      if (active) { PetscCall(VecGetLocalSize(subX,&mlocal)); }
      {
        const PetscScalar *LA_c = NULL;
        PetscCall(VecCreateMPIWithArray(comm,1,mlocal,PETSC_DECIDE,NULL,&coorn));
        if (active) {
          PetscCall(VecGetArrayRead(subX,&LA_c));
          PetscCall(VecPlaceArray(coorn,LA_c));
          PetscCall(VecRestoreArrayRead(subX,&LA_c));
        }
      }
      PetscCall(VecLoad(coorn,v)); 
      PetscCall(VecDestroy(&coorn));
      PetscCall(VecDestroy(&subX));
      PetscCall(DMDestroy(&pda));

      // z1d_cell
      PetscCall(DMStagVecSplitToDMDA(subDM,coor,DMSTAG_ELEMENT,-sdof[1],&pda,&subX));
      mlocal = 0;
      if (active) { PetscCall(VecGetLocalSize(subX,&mlocal)); }
      {
        const PetscScalar *LA_c = NULL;
        PetscCall(VecCreateMPIWithArray(comm,1,mlocal,PETSC_DECIDE,NULL,&coorn));
        if (active) {
          PetscCall(VecGetArrayRead(subX,&LA_c));
          PetscCall(VecPlaceArray(coorn,LA_c));
          PetscCall(VecRestoreArrayRead(subX,&LA_c));
        }
      }
      PetscCall(VecLoad(coorn,v)); 
      PetscCall(VecDestroy(&coorn));
      PetscCall(VecDestroy(&subX));
      PetscCall(DMDestroy(&pda));
    }
  }

  { // read data
    DM pda;
    Vec subX, subXn;

    if (dim == 1) {
      SETERRQ(comm,PETSC_ERR_SUP,"Only valid for 2d DM");
    } else if (dim == 2) {

      PetscInt sx,sz,nx,nz,i,j,iloc,idof;
      Vec xlocal,subxlocal;
      PetscScalar ***xx,**subxx;

      PetscCall(DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 
      
      if (dof[0] != 0) { // X_vertex
        PetscCall(DMStagVecSplitToDMDA(dm,x,DMSTAG_DOWN_LEFT,-dof[0],&pda,&subX));
        PetscCall(DMDACreateNaturalVector(pda,&subXn));
        PetscCall(VecLoad(subXn,v)); 
        PetscCall(DMDANaturalToGlobalBegin(pda,subXn,INSERT_VALUES,subX));
        PetscCall(DMDANaturalToGlobalEnd(pda,subXn,INSERT_VALUES,subX));

        // map the dmda and subX and run cellwise.
        PetscCall(DMGetLocalVector(dm, &xlocal)); 
        PetscCall(DMGlobalToLocal (dm, x, INSERT_VALUES, xlocal)); 
        PetscCall(DMStagVecGetArray(dm, xlocal, &xx)); 

        // Map global vectors to local domain
        PetscCall(DMGetLocalVector(pda, &subxlocal)); 
        PetscCall(DMGlobalToLocal (pda, subX, INSERT_VALUES, subxlocal)); 
        PetscCall(DMDAVecGetArray(pda, subxlocal,&subxx)); 

        // Loop over local domain
        for (idof = 0; idof < dof[0]; idof++) {
          PetscCall(DMStagGetLocationSlot(dm, DMSTAG_DOWN_LEFT, idof, &iloc)); 
          for (i = sx; i < sx+nx+1; i++) {
            for (j = sz; j <sz+nz+1; j++) {
              xx[j][i][iloc] = subxx[j][dof[0]*i+idof]; 
            }
          }
        }

        // Restore arrays
        PetscCall(DMDAVecRestoreArray(pda,subxlocal,&subxx)); 
        PetscCall(DMRestoreLocalVector(pda, &subxlocal)); 

        PetscCall(DMStagVecRestoreArray(dm,xlocal,&xx)); 
        PetscCall(DMLocalToGlobalBegin(dm,xlocal,INSERT_VALUES,x)); 
        PetscCall(DMLocalToGlobalEnd  (dm,xlocal,INSERT_VALUES,x)); 
        PetscCall(DMRestoreLocalVector(dm, &xlocal)); 

        PetscCall(VecDestroy(&subXn));
        PetscCall(VecDestroy(&subX));
        PetscCall(DMDestroy(&pda));
      }
      if (dof[1] != 0) {
        // X_face_x
        PetscCall(DMStagVecSplitToDMDA(dm,x,DMSTAG_LEFT,-dof[1],&pda,&subX)); 
        PetscCall(DMDACreateNaturalVector(pda,&subXn));
        PetscCall(VecLoad(subXn,v)); 
        PetscCall(DMDANaturalToGlobalBegin(pda,subXn,INSERT_VALUES,subX));
        PetscCall(DMDANaturalToGlobalEnd(pda,subXn,INSERT_VALUES,subX));

        PetscCall(DMGetLocalVector(dm, &xlocal)); 
        PetscCall(DMGlobalToLocal (dm, x, INSERT_VALUES, xlocal)); 
        PetscCall(DMStagVecGetArray(dm, xlocal, &xx)); 

        PetscCall(DMGetLocalVector(pda, &subxlocal)); 
        PetscCall(DMGlobalToLocal (pda, subX, INSERT_VALUES, subxlocal)); 
        PetscCall(DMDAVecGetArray(pda, subxlocal,&subxx)); 

        // Loop over local domain
        for (idof = 0; idof < dof[1]; idof++) {
          PetscCall(DMStagGetLocationSlot(dm, DMSTAG_LEFT, idof, &iloc)); 
          for (i = sx; i < sx+nx+1; i++) {
            for (j = sz; j <sz+nz; j++) {
              xx[j][i][iloc] = subxx[j][dof[1]*i+idof]; 
            }
          }
        }

        // Restore arrays
        PetscCall(DMDAVecRestoreArray(pda,subxlocal,&subxx)); 
        PetscCall(DMRestoreLocalVector(pda, &subxlocal)); 
        
        PetscCall(DMStagVecRestoreArray(dm,xlocal,&xx)); 
        PetscCall(DMLocalToGlobalBegin(dm,xlocal,INSERT_VALUES,x)); 
        PetscCall(DMLocalToGlobalEnd  (dm,xlocal,INSERT_VALUES,x)); 
        PetscCall(DMRestoreLocalVector(dm, &xlocal)); 

        PetscCall(VecDestroy(&subXn));
        PetscCall(VecDestroy(&subX));
        PetscCall(DMDestroy(&pda));

        // X_face_y
        PetscCall(DMStagVecSplitToDMDA(dm,x,DMSTAG_DOWN,-dof[1],&pda,&subX));
        PetscCall(DMDACreateNaturalVector(pda,&subXn));
        PetscCall(VecLoad(subXn,v)); 
        PetscCall(DMDANaturalToGlobalBegin(pda,subXn,INSERT_VALUES,subX));
        PetscCall(DMDANaturalToGlobalEnd(pda,subXn,INSERT_VALUES,subX));

        PetscCall(DMGetLocalVector(dm, &xlocal)); 
        PetscCall(DMGlobalToLocal (dm, x, INSERT_VALUES, xlocal)); 
        PetscCall(DMStagVecGetArray(dm, xlocal, &xx)); 

        PetscCall(DMGetLocalVector(pda, &subxlocal)); 
        PetscCall(DMGlobalToLocal (pda, subX, INSERT_VALUES, subxlocal)); 
        PetscCall(DMDAVecGetArray(pda, subxlocal,&subxx)); 

        // Loop over local domain
        for (idof = 0; idof < dof[1]; idof++) {
          PetscCall(DMStagGetLocationSlot(dm, DMSTAG_DOWN, idof, &iloc)); 
          for (j = sz; j < sz+nz+1; j++) {
            for (i = sx; i <sx+nx; i++) {
              xx[j][i][iloc] = subxx[j][dof[1]*i+idof]; 
            }
          }
        }

        // Restore arrays
        PetscCall(DMDAVecRestoreArray(pda,subxlocal,&subxx)); 
        PetscCall(DMRestoreLocalVector(pda, &subxlocal)); 
        
        PetscCall(DMStagVecRestoreArray(dm,xlocal,&xx)); 
        PetscCall(DMLocalToGlobalBegin(dm,xlocal,INSERT_VALUES,x)); 
        PetscCall(DMLocalToGlobalEnd  (dm,xlocal,INSERT_VALUES,x)); 
        PetscCall(DMRestoreLocalVector(dm, &xlocal)); 

        PetscCall(VecDestroy(&subXn));
        PetscCall(VecDestroy(&subX));
        PetscCall(DMDestroy(&pda));
      }
      if (dof[2] != 0) { // X_cell
        PetscCall(DMStagVecSplitToDMDA(dm,x,DMSTAG_ELEMENT,-dof[2],&pda,&subX));
        PetscCall(DMDACreateNaturalVector(pda,&subXn));
        PetscCall(VecLoad(subXn,v)); 
        PetscCall(DMDANaturalToGlobalBegin(pda,subXn,INSERT_VALUES,subX));
        PetscCall(DMDANaturalToGlobalEnd(pda,subXn,INSERT_VALUES,subX));
        
        PetscCall(DMGetLocalVector(dm, &xlocal)); 
        PetscCall(DMGlobalToLocal (dm, x, INSERT_VALUES, xlocal)); 
        PetscCall(DMStagVecGetArray(dm, xlocal, &xx)); 

        PetscCall(DMGetLocalVector(pda, &subxlocal)); 
        PetscCall(DMGlobalToLocal (pda, subX, INSERT_VALUES, subxlocal)); 
        PetscCall(DMDAVecGetArray(pda, subxlocal,&subxx)); 

        // Loop over local domain
        for (idof = 0; idof < dof[2]; idof++) {
          PetscCall(DMStagGetLocationSlot(dm, DMSTAG_ELEMENT, idof, &iloc)); 
          for (i = sx; i < sx+nx; i++) {
            for (j = sz; j <sz+nz; j++) {
              xx[j][i][iloc] = subxx[j][dof[2]*i+idof]; 
            }
          }
        }

        // Restore arrays
        PetscCall(DMDAVecRestoreArray(pda,subxlocal,&subxx)); 
        PetscCall(DMRestoreLocalVector(pda, &subxlocal)); 
        
        PetscCall(DMStagVecRestoreArray(dm,xlocal,&xx)); 
        PetscCall(DMLocalToGlobalBegin(dm,xlocal,INSERT_VALUES,x)); 
        PetscCall(DMLocalToGlobalEnd  (dm,xlocal,INSERT_VALUES,x)); 
        PetscCall(DMRestoreLocalVector(dm, &xlocal)); 

        PetscCall(VecDestroy(&subXn));
        PetscCall(VecDestroy(&subX));
        PetscCall(DMDestroy(&pda));
      }
    } else if (dim == 3) {
      SETERRQ(comm,PETSC_ERR_SUP,"Only valid for 2d DM");
    }
  }

  PetscCall(PetscViewerDestroy(&v));

  *_dm = dm;
  *_x = x;

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
/*@
DMStagReadBinaryPython - read file written with PetscBinary/python

Input parameters:
prefix - output name (no file extension)

Output parameters:
dm - the DMStag object
X  - associated vector with dm

Use: user
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "DMStagReadBinaryPython"
PetscErrorCode DMStagReadBinaryPython(DM *dm,Vec *x,const char prefix[])
{
  MPI_Comm comm;
  PetscMPIInt size;
  PetscFunctionBegin;
  
  comm = PETSC_COMM_WORLD;
  PetscCall(MPI_Comm_size(comm,&size)); 
  if (size == 1) {
    PetscCall(DMStagReadBinaryPython_SEQ(&(*dm),&(*x),prefix));
  } else {
    PetscCall(DMStagReadBinaryPython_MPI(&(*dm),&(*x),prefix));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// static PetscErrorCode DMStagGetProductCoordinateArrays_Private(DM dm,void* arrX,void* arrY,void* arrZ,PetscBool read)
// {
//   PetscInt       dim,d,dofCheck[DMSTAG_MAX_STRATA],s;
//   DM             dmCoord;
//   void*          arr[DMSTAG_MAX_DIM];
//   PetscBool      checkDof;
  
//   PetscFunctionBegin;
//   PetscValidHeaderSpecific(dm,DM_CLASSID,1);
//   PetscCall(DMGetDimension(dm,&dim));
//   if (dim > DMSTAG_MAX_DIM) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Not implemented for %" PetscInt_FMT " dimensions",dim);
//   arr[0] = arrX; arr[1] = arrY; arr[2] = arrZ;
//   PetscCall(DMGetCoordinateDM(dm,&dmCoord));
//   if (!dmCoord) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"DM does not have a coordinate DM");
//   {
//     PetscBool isProduct;
//     DMType    dmType;
//     PetscCall(DMGetType(dmCoord,&dmType));
//     PetscCall(PetscStrcmp(DMPRODUCT,dmType,&isProduct));
//     if (!isProduct) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Coordinate DM is not of type DMPRODUCT");
//   }
//   for (s=0; s<DMSTAG_MAX_STRATA; ++s) dofCheck[s] = 0;
//   checkDof = PETSC_FALSE;
//   for (d=0; d<dim; ++d) {
//     DM        subDM;
//     DMType    dmType;
//     PetscBool isStag;
//     PetscInt  dof[DMSTAG_MAX_STRATA],subDim;
//     Vec       coord1d_local;
    
//     /* Ignore unrequested arrays */
//     if (!arr[d]) continue;
    
//     PetscCall(DMProductGetDM(dmCoord,d,&subDM));
//     if (!subDM) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Coordinate DM is missing sub DM %" PetscInt_FMT,d);
//     PetscCall(DMGetDimension(subDM,&subDim));
//     if (subDim != 1) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Coordinate sub-DM is not of dimension 1");
//     PetscCall(DMGetType(subDM,&dmType));
//     PetscCall(PetscStrcmp(DMSTAG,dmType,&isStag));
//     if (!isStag) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Coordinate sub-DM is not of type DMSTAG");
//     PetscCall(DMStagGetDOF(subDM,&dof[0],&dof[1],&dof[2],&dof[3]));
//     if (!checkDof) {
//       for (s=0; s<DMSTAG_MAX_STRATA; ++s) dofCheck[s] = dof[s];
//       checkDof = PETSC_TRUE;
//     } else {
//       for (s=0; s<DMSTAG_MAX_STRATA; ++s) {
//         if (dofCheck[s] != dof[s]) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Coordinate sub-DMs have different dofs");
//       }
//     }
//     PetscCall(DMGetCoordinatesLocal(subDM,&coord1d_local));
//     if (read) {
//       PetscCall(DMStagVecGetArrayRead(subDM,coord1d_local,arr[d]));
//     } else {
//       PetscCall(DMStagVecGetArray(subDM,coord1d_local,arr[d]));
//     }
//   }
//   PetscFunctionReturn(PETSC_SUCCESS);
// }

// ---------------------------------------
// /*@C
//  DMStagGetProductCoordinateArraysRead - extract product coordinate arrays, read-only
 
//  Logically Collective
 
//  See the man page for DMStagGetProductCoordinateArrays() for more information.
 
//  Input Parameter:
//  . dm - the DMStag object
 
//  Output Parameters:
//  . arrX,arrY,arrZ - local 1D coordinate arrays
 
//  Level: intermediate
 
//  .seealso: DMSTAG, DMPRODUCT, DMStagGetProductCoordinateArrays(), DMStagSetUniformCoordinates(), DMStagSetUniformCoordinatesProduct(), DMStagGetProductCoordinateLocationSlot()
//  @*/
// PetscErrorCode _DMStagGetProductCoordinateArraysRead(DM dm,void* arrX,void* arrY,void* arrZ)
// {
//   PetscFunctionBegin;
//   PetscCall(DMStagGetProductCoordinateArrays_Private(dm,arrX,arrY,arrZ,PETSC_TRUE));
//   PetscFunctionReturn(PETSC_SUCCESS);
// }

// ---------------------------------------
// static PetscErrorCode DMStagRestoreProductCoordinateArrays_Private(DM dm,void *arrX,void *arrY,void *arrZ,PetscBool read)
// {
//   PetscInt        dim,d;
//   void*           arr[DMSTAG_MAX_DIM];
//   DM              dmCoord;
  
//   PetscFunctionBegin;
//   PetscValidHeaderSpecific(dm,DM_CLASSID,1);
//   PetscCall(DMGetDimension(dm,&dim));
//   if (dim > DMSTAG_MAX_DIM) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Not implemented for % dimensions" PetscInt_FMT,dim);
//   arr[0] = arrX; arr[1] = arrY; arr[2] = arrZ;
//   PetscCall(DMGetCoordinateDM(dm,&dmCoord));
//   for (d=0; d<dim; ++d) {
//     DM  subDM;
//     Vec coord1d_local;
    
//     /* Ignore unrequested arrays */
//     if (!arr[d]) continue;
    
//     PetscCall(DMProductGetDM(dmCoord,d,&subDM));
//     PetscCall(DMGetCoordinatesLocal(subDM,&coord1d_local));
//     if (read) {
//       PetscCall(DMStagVecRestoreArrayRead(subDM,coord1d_local,arr[d]));
//     } else {
//       PetscCall(DMStagVecRestoreArray(subDM,coord1d_local,arr[d]));
//     }
//   }
//   PetscFunctionReturn(PETSC_SUCCESS);
// }

// ---------------------------------------
// /*@C
//  DMStagRestoreProductCoordinateArraysRead - restore local product array access, read-only
 
//  Logically Collective
 
//  Input Parameter:
//  . dm - the DMStag object
 
//  Output Parameters:
//  . arrX,arrY,arrZ - local 1D coordinate arrays
 
//  Level: intermediate
 
//  .seealso: DMSTAG, DMStagGetProductCoordinateArrays(), DMStagGetProductCoordinateArraysRead()
//  @*/
// PetscErrorCode _DMStagRestoreProductCoordinateArraysRead(DM dm,void *arrX,void *arrY,void *arrZ)
// {
//   PetscFunctionBegin;
//   PetscCall(DMStagRestoreProductCoordinateArrays_Private(dm,arrX,arrY,arrZ,PETSC_TRUE));
//   PetscFunctionReturn(PETSC_SUCCESS);
// }

// ---------------------------------------
// /*@C
//  DMStagGetProductCoordinateLocationSlot - get slot for use with local product coordinate arrays
 
//  Not Collective
 
//  High-level helper function to get slot indices for 1D coordinate DMs,
//  for use with DMStagGetProductCoordinateArrays() and related functions.
 
//  Input Parameters:
//  + dm - the DMStag object
//  - loc - the grid location
 
//  Output Parameter:
//  . slot - the index to use in local arrays
 
//  Notes:
//  Checks that the coordinates are actually set up so that using the
//  slots from the first 1d coordinate sub-DM is valid for all the 1D coordinate sub-DMs.
 
//  Level: intermediate
 
//  .seealso: DMSTAG, DMPRODUCT, DMStagGetProductCoordinateArrays(), DMStagGetProductCoordinateArraysRead(), DMStagSetUniformCoordinates()
//  @*/
// PetscErrorCode _DMStagGetProductCoordinateLocationSlot(DM dm,DMStagStencilLocation loc,PetscInt *slot)
// {
//   DM             dmCoord;
//   PetscInt       dim,dofCheck[DMSTAG_MAX_STRATA],s,d;
  
//   PetscFunctionBegin;
//   PetscValidHeaderSpecific(dm,DM_CLASSID,1);
//   PetscCall(DMGetDimension(dm,&dim));
//   PetscCall(DMGetCoordinateDM(dm,&dmCoord));
//   if (!dmCoord) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"DM does not have a coordinate DM");
//   {
//     PetscBool isProduct;
//     DMType    dmType;
//     PetscCall(DMGetType(dmCoord,&dmType));
//     PetscCall(PetscStrcmp(DMPRODUCT,dmType,&isProduct));
//     if (!isProduct) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Coordinate DM is not of type DMPRODUCT");
//   }
//   for (s=0; s<DMSTAG_MAX_STRATA; ++s) dofCheck[s] = 0;
//   for (d=0; d<dim; ++d) {
//     DM        subDM;
//     DMType    dmType;
//     PetscBool isStag;
//     PetscInt  dof[DMSTAG_MAX_STRATA],subDim;
//     PetscCall(DMProductGetDM(dmCoord,d,&subDM));
//     if (!subDM) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Coordinate DM is missing sub DM %" PetscInt_FMT,d);
//     PetscCall(DMGetDimension(subDM,&subDim));
//     if (subDim != 1) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Coordinate sub-DM is not of dimension 1");
//     PetscCall(DMGetType(subDM,&dmType));
//     PetscCall(PetscStrcmp(DMSTAG,dmType,&isStag));
//     if (!isStag) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Coordinate sub-DM is not of type DMSTAG");
//     PetscCall(DMStagGetDOF(subDM,&dof[0],&dof[1],&dof[2],&dof[3]));
//     if (d == 0) {
//       const PetscInt component = 0;
//       for (s=0; s<DMSTAG_MAX_STRATA; ++s) dofCheck[s] = dof[s];
//       PetscCall(DMStagGetLocationSlot(subDM,loc,component,slot));
//     } else {
//       for (s=0; s<DMSTAG_MAX_STRATA; ++s) {
//         if (dofCheck[s] != dof[s]) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Coordinate sub-DMs have different dofs");
//       }
//     }
//   }
//   PetscFunctionReturn(PETSC_SUCCESS);
// }

// ---------------------------------------
// PetscErrorCode DMStagStencilLocationCanonicalize(DMStagStencilLocation loc,DMStagStencilLocation *locCanonical)
// {
//   PetscFunctionBegin;
//   switch (loc) {
//     case DMSTAG_ELEMENT:
//       *locCanonical = DMSTAG_ELEMENT;
//       break;
//     case DMSTAG_LEFT:
//     case DMSTAG_RIGHT:
//       *locCanonical = DMSTAG_LEFT;
//       break;
//     case DMSTAG_DOWN:
//     case DMSTAG_UP:
//       *locCanonical = DMSTAG_DOWN;
//       break;
//     case DMSTAG_BACK:
//     case DMSTAG_FRONT:
//       *locCanonical = DMSTAG_BACK;
//       break;
//     case DMSTAG_DOWN_LEFT :
//     case DMSTAG_DOWN_RIGHT :
//     case DMSTAG_UP_LEFT :
//     case DMSTAG_UP_RIGHT :
//       *locCanonical = DMSTAG_DOWN_LEFT;
//       break;
//     case DMSTAG_BACK_LEFT:
//     case DMSTAG_BACK_RIGHT:
//     case DMSTAG_FRONT_LEFT:
//     case DMSTAG_FRONT_RIGHT:
//       *locCanonical = DMSTAG_BACK_LEFT;
//       break;
//     case DMSTAG_BACK_DOWN:
//     case DMSTAG_BACK_UP:
//     case DMSTAG_FRONT_DOWN:
//     case DMSTAG_FRONT_UP:
//       *locCanonical = DMSTAG_BACK_DOWN;
//       break;
//     case DMSTAG_BACK_DOWN_LEFT:
//     case DMSTAG_BACK_DOWN_RIGHT:
//     case DMSTAG_BACK_UP_LEFT:
//     case DMSTAG_BACK_UP_RIGHT:
//     case DMSTAG_FRONT_DOWN_LEFT:
//     case DMSTAG_FRONT_DOWN_RIGHT:
//     case DMSTAG_FRONT_UP_LEFT:
//     case DMSTAG_FRONT_UP_RIGHT:
//       *locCanonical = DMSTAG_BACK_DOWN_LEFT;
//       break;
//     default :
//       *locCanonical = DMSTAG_NULL_LOCATION;
//       break;
//   }
//   PetscFunctionReturn(PETSC_SUCCESS);
// }

// ---------------------------------------
// /*@C
//  DMStagCreateISFromStencils - Create an IS, using global numberings, for a subset of DOF in a DMStag object
 
//  Collective
 
//  Input Parameters:
//  + dm - the DMStag object
//  . nStencil - the number of stencils provided
//  - stencils - an array of DMStagStencil objects (i,j, and k are ignored)
 
//  Output Parameter:
//  . is - the global IS
 
//  Note:
//  Redundant entries in s are ignored
 
//  Level: advanced
 
//  .seealso: DMSTAG, IS, DMStagStencil, DMCreateGlobalVector
//  @*/
// PetscErrorCode DMStagCreateISFromStencils(DM dm,PetscInt nStencil,DMStagStencil* stencils,IS *is)
// {
//   DMStagStencil          *ss;
//   PetscInt               *idx,*idxLocal;
//   const PetscInt         *ltogidx;
//   PetscInt               p,p2,pmax,i,j,k,d,dim,count,nidx;
//   ISLocalToGlobalMapping ltog;
//   PetscInt               start[DMSTAG_MAX_DIM],n[DMSTAG_MAX_DIM],extraPoint[DMSTAG_MAX_DIM];
  
//   PetscFunctionBegin;
//   PetscCall(DMGetDimension(dm,&dim));
//   if (dim<1 || dim>3) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Unsupported dimension %D",dim);
  
//   /* Only use non-redundant stencils */
//   PetscCall(PetscMalloc1(nStencil,&ss));
//   pmax = 0;
//   for (p=0; p<nStencil; ++p) {
//     PetscBool skip = PETSC_FALSE;
//     DMStagStencil stencilPotential = stencils[p];
//     PetscCall(DMStagStencilLocationCanonicalize(stencils[p].loc,&stencilPotential.loc));
//     for (p2=0; p2<pmax; ++p2) { /* Quadratic complexity algorithm in nStencil */
//       if (stencilPotential.loc == ss[p2].loc && stencilPotential.c == ss[p2].c) {
//         skip = PETSC_TRUE;
//         break;
//       }
//     }
//     if (!skip) {
//       ss[pmax] = stencilPotential;
//       ++pmax;
//     }
//   }
  
//   PetscCall(PetscMalloc1(pmax,&idxLocal));
//   PetscCall(DMGetLocalToGlobalMapping(dm,&ltog));
//   PetscCall(ISLocalToGlobalMappingGetIndices(ltog,&ltogidx));
//   PetscCall(DMStagGetCorners(dm,&start[0],&start[1],&start[2],&n[0],&n[1],&n[2],&extraPoint[0],&extraPoint[1],&extraPoint[2]));
//   for (d=dim; d<DMSTAG_MAX_DIM; ++d) {
//     start[d]      = 0;
//     n[d]          = 1; /* To allow for a single loop nest below */
//     extraPoint[d] = 0;
//   }
//   nidx = pmax; for (d=0; d<dim; ++d) nidx *= (n[d]+1); /* Overestimate (always assumes extraPoint) */
//   PetscCall(PetscMalloc1(nidx,&idx));
//   count = 0;
//   /* Note that unused loop variables are not accessed, for lower dimensions */
//   for (k=start[2]; k<start[2]+n[2]+extraPoint[2]; ++k) {
//     for (j=start[1]; j<start[1]+n[1]+extraPoint[1]; ++j) {
//       for (i=start[0]; i<start[0]+n[0]+extraPoint[0]; ++i) {
//         for(p=0; p<pmax; ++p) {
//           ss[p].i = i; ss[p].j = j; ss[p].k = k;
//         }
//         PetscCall(DMStagStencilToIndexLocal(dm,dim,pmax,ss,idxLocal));
//         for(p=0; p<pmax; ++p) {
//           const PetscInt gidx = ltogidx[idxLocal[p]];
//           if (gidx >= 0) {
//             idx[count] = gidx;
//             ++count;
//           }
//         }
//       }
//     }
//   }
//   PetscCall(ISLocalToGlobalMappingRestoreIndices(ltog,&ltogidx));
//   PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)dm),count,idx,PETSC_OWN_POINTER,is));
  
//   PetscCall(PetscFree(ss));
//   PetscCall(PetscFree(idxLocal));
//   PetscFunctionReturn(PETSC_SUCCESS);
// }

// ---------------------------------------
// /* Note: this does a linear search, which should be bisection, but an even
//  better strategy would be to use any information about which cell a particle
//  is already in, as in most applications it would be found there or in a
//  neighboring cell */
// static PetscErrorCode DMStagLocatePointsIS_2D_Product_Private(DM dm,Vec pos,IS *iscell)
// {
//   PetscInt          localSize,bs,p,npoints,start[2],n[2];
//   PetscInt          *cellidx;
//   const PetscScalar *_coor;
//   PetscScalar       **cArrX,**cArrY;
//   PetscInt          iNext,iPrev;
  
//   PetscFunctionBegin;
  
//   PetscCall(VecGetLocalSize(pos,&localSize));
//   PetscCall(VecGetBlockSize(pos,&bs));
//   npoints = localSize/bs;
  
//   PetscCall(PetscMalloc1(npoints,&cellidx));
//   PetscCall(VecGetArrayRead(pos,&_coor));
  
//   PetscCall(DMStagGetCorners(dm,&start[0],&start[1],NULL,&n[0],&n[1],NULL,NULL,NULL,NULL));
//   PetscCall(DMStagGetProductCoordinateArraysRead(dm,&cArrX,&cArrY,NULL));
//   PetscCall(DMStagGetProductCoordinateLocationSlot(dm,DMSTAG_LEFT,&iPrev));
//   PetscCall(DMStagGetProductCoordinateLocationSlot(dm,DMSTAG_RIGHT,&iNext));
  
//   for (p=0; p<npoints; p++) {
//     PetscReal coor_p[2];
    
//     coor_p[0] = PetscRealPart(_coor[2*p]);
//     coor_p[1] = PetscRealPart(_coor[2*p+1]);
    
//     if ((coor_p[0] >= cArrX[start[0]][iPrev]) && (coor_p[0] <= cArrX[start[0]+n[0]][iPrev]) && (coor_p[1] >= cArrY[start[1]][iPrev]) && (coor_p[1] <= cArrY[start[1]+n[1]][iPrev]))
//     {
//       PetscInt e,ind[2];
//       for (ind[0]=start[0]; ind[0]<start[0]+n[0]; ++ind[0]) {
//         if (coor_p[0] <= cArrX[ind[0]][iNext]) break;
//       }
//       for (ind[1]=start[1]; ind[1]<start[1]+n[1]; ++ind[1]) {
//         if (coor_p[1] <= cArrY[ind[1]][iNext]) break;
//       }
//       PetscCall(DMStagGetLocalElementIndex(dm,ind,&e));
//       cellidx[p] = e;
//     } else {
//       cellidx[p] = DMLOCATEPOINT_POINT_NOT_FOUND;
//     }
//   }
//   PetscCall(DMStagRestoreProductCoordinateArraysRead(dm,&cArrX,&cArrY,NULL));
//   PetscCall(VecRestoreArrayRead(pos,&_coor));
  
//   PetscCall(ISCreateGeneral(PETSC_COMM_SELF,npoints,cellidx,PETSC_OWN_POINTER,iscell));
//   PetscFunctionReturn(PETSC_SUCCESS);
// }

