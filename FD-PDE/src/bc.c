#include "bc.h"

static PetscErrorCode FDBCGetEntry(DM,PetscScalar**,PetscScalar**,DMStagStencilLocation,PetscInt,PetscInt,PetscInt,DMStagBC*);

// ---------------------------------------
// FDBCListCreate
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "DMStagBCCreateDefault"
PetscErrorCode DMStagBCCreateDefault(DM dm, DMStagBC **_list, PetscInt *_ndof)
{
  PetscInt Nx, Nz, sx, sz, nx, nz;
  PetscInt i, j, ii, idof, ndof, dof0, dof1, dof2;
  DMStagStencilLocation loc, loc1;
  DMStagBC *list = NULL;
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
  ierr = PetscMalloc((size_t)ndof*sizeof(DMStagBC),&list);CHKERRQ(ierr);
  ierr = PetscMemzero(list,(size_t)ndof*sizeof(DMStagBC));CHKERRQ(ierr);

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
    list[ii].type = BC_NULL;
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
#define __FUNCT__ "DMStagBCDestroy"
PetscErrorCode DMStagBCDestroy(DMStagBC **_list)
{
  DMStagBC *list;
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
static PetscErrorCode FDBCGetEntry(DM dm,PetscScalar **cx,PetscScalar **cz, DMStagStencilLocation loc, PetscInt c, PetscInt i, PetscInt j, DMStagBC *list)
{
  PetscInt       ii = -1, jj = -1, dof[]={0,0,0};
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  list->point.i   = i;
  list->point.j   = j;
  list->point.c   = c;
  list->point.loc = loc;
  ierr = DMStagGetLocationSlot(dm,loc,c,&list->idx); CHKERRQ(ierr);

  ierr = DMStagGetDOF(dm,&dof[0],&dof[1],&dof[2],NULL);CHKERRQ(ierr);

  if (dof[0]) { /* has vertices */
    PetscInt iprev=-1, inext=-1;
    
    ierr = DMStagGet1dCoordinateLocationSlot(dm,DMSTAG_LEFT, &iprev);CHKERRQ(ierr);
    ierr = DMStagGet1dCoordinateLocationSlot(dm,DMSTAG_RIGHT,&inext);CHKERRQ(ierr);

    if ((loc == DMSTAG_DOWN_LEFT)  || (loc == DMSTAG_UP_LEFT))  ii = iprev;
    if ((loc == DMSTAG_DOWN_RIGHT) || (loc == DMSTAG_UP_RIGHT)) ii = inext;
    
    if ((loc == DMSTAG_DOWN_LEFT) || (loc == DMSTAG_DOWN_RIGHT)) jj = iprev;
    if ((loc == DMSTAG_UP_LEFT)   || (loc == DMSTAG_UP_RIGHT))   jj = inext;
    
    list->coord[0] = cx[i][ii];
    list->coord[1] = cz[j][jj];
  }
  if (dof[1]) { /* has faces */
    PetscInt iprev=-1, inext=-1;
    
    ierr = DMStagGet1dCoordinateLocationSlot(dm,DMSTAG_LEFT, &iprev);CHKERRQ(ierr);
    ierr = DMStagGet1dCoordinateLocationSlot(dm,DMSTAG_RIGHT,&inext);CHKERRQ(ierr);

    ii = i;
    jj = j;
    if (loc == DMSTAG_LEFT)  ii = iprev;
    if (loc == DMSTAG_RIGHT) ii = inext;
    
    if (loc == DMSTAG_DOWN)  jj = iprev;
    if (loc == DMSTAG_UP)    jj = inext;
    
    if (loc == DMSTAG_LEFT || loc == DMSTAG_RIGHT) {
      list->coord[0] = cx[i][ii];
      list->coord[1] = 0.5 * (cz[0][jj] + cz[0][jj+1]);
    }
    
    if (loc == DMSTAG_DOWN || loc == DMSTAG_UP) {
      list->coord[0] = 0.5 * (cx[0][ii] + cx[0][ii+1]);
      list->coord[1] = cz[j][jj];
    }
  }
  if (dof[2]) { /* has elements */
    PetscInt icenter=-1;
    
    ierr = DMStagGet1dCoordinateLocationSlot(dm,DMSTAG_ELEMENT,&icenter);CHKERRQ(ierr);
    ii = icenter;
    jj = icenter;
    
    list->coord[0] = cx[i][ii];
    list->coord[1] = cz[j][jj];
  }

  PetscFunctionReturn(0);
}


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

#undef __FUNCT__
#define __FUNCT__ "_DMStagBCFillCoords"
static PetscErrorCode _DMStagBCFillCoords(PetscInt stratum_id,DM dm,PetscScalar **cx,PetscScalar **cz,DMStagStencilLocation loc,PetscInt i, PetscInt j,DMStagBC *bc)
{
  PetscInt       ii=-1,jj=-1,dof[]={0,0,0};
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  
  ierr = DMStagGetDOF(dm,&dof[0],&dof[1],&dof[2],NULL);CHKERRQ(ierr);

  switch (stratum_id) {
    case 0: /* vertices */
    {
      PetscInt iprev=-1, inext=-1;
      
      ierr = DMStagGet1dCoordinateLocationSlot(dm,DMSTAG_LEFT, &iprev);CHKERRQ(ierr);
      ierr = DMStagGet1dCoordinateLocationSlot(dm,DMSTAG_RIGHT,&inext);CHKERRQ(ierr);
      
      if ((loc == DMSTAG_DOWN_LEFT)  || (loc == DMSTAG_UP_LEFT))  ii = iprev;
      if ((loc == DMSTAG_DOWN_RIGHT) || (loc == DMSTAG_UP_RIGHT)) ii = inext;
      
      if ((loc == DMSTAG_DOWN_LEFT) || (loc == DMSTAG_DOWN_RIGHT)) jj = iprev;
      if ((loc == DMSTAG_UP_LEFT)   || (loc == DMSTAG_UP_RIGHT))   jj = inext;
      
      bc->coord[0] = cx[i][ii];
      bc->coord[1] = cz[j][jj];
    }
      break;
      
    case 1: /* faces */
    {
      PetscInt iprev=-1, inext=-1;
      
      ierr = DMStagGet1dCoordinateLocationSlot(dm,DMSTAG_LEFT, &iprev);CHKERRQ(ierr);
      ierr = DMStagGet1dCoordinateLocationSlot(dm,DMSTAG_RIGHT,&inext);CHKERRQ(ierr);
      
      ii = i;
      jj = j;
      if (loc == DMSTAG_LEFT)  ii = iprev;
      if (loc == DMSTAG_RIGHT) ii = inext;
      
      if (loc == DMSTAG_DOWN)  jj = iprev;
      if (loc == DMSTAG_UP)    jj = inext;
      
      if (loc == DMSTAG_LEFT || loc == DMSTAG_RIGHT) {
        bc->coord[0] = cx[i][ii];
        bc->coord[1] = 0.5 * (cz[0][jj] + cz[0][jj+1]);
      }
      
      if (loc == DMSTAG_DOWN || loc == DMSTAG_UP) {
        bc->coord[0] = 0.5 * (cx[0][ii] + cx[0][ii+1]);
        bc->coord[1] = cz[j][jj];
      }
    }
      break;
      
    case 2: /* elements */
    {
      PetscInt icenter=-1;
      
      ierr = DMStagGet1dCoordinateLocationSlot(dm,DMSTAG_ELEMENT,&icenter);CHKERRQ(ierr);
      ii = icenter;
      jj = icenter;
      
      bc->coord[0] = cx[i][ii];
      bc->coord[1] = cz[j][jj];
    }
      break;
      
    default:
      SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_USER,"Stratum index (arg 1) must {0,1,2}");
      break;
  }
  PetscFunctionReturn(0);
}


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
        //if (dof0) {
          if (i == 0) { loc = DMSTAG_DOWN_LEFT; }
          else        { loc = DMSTAG_DOWN_RIGHT; }
          loc = (i == 0) ? DMSTAG_DOWN_LEFT : DMSTAG_DOWN_RIGHT;
          for (ii = 0; ii<dof0; ii++) {
            ierr = _DMStagBCFillIndices(dm,loc,ii,i,j,&list->bc_v[ndof0++]);CHKERRQ(ierr);
          }
        //}
        //if (dof1) { // 2 DOFs - LEFT, DOWN
          if (i == 0) { loc = DMSTAG_LEFT; }
          else        { loc = DMSTAG_RIGHT; }
          loc = (i == 0) ? DMSTAG_LEFT : DMSTAG_RIGHT;
          loc1 = DMSTAG_DOWN;
          for (ii = 0; ii<dof1; ii++) {
            ierr = _DMStagBCFillIndices(dm,loc, ii,i,j,&list->bc_f[ndof1++]);CHKERRQ(ierr);
            ierr = _DMStagBCFillIndices(dm,loc1,ii,i,j,&list->bc_f[ndof1++]);CHKERRQ(ierr);
          }
        //}
        //if (dof2) {
          for (ii = 0; ii<dof2; ii++) {
            ierr = _DMStagBCFillIndices(dm,DMSTAG_ELEMENT,ii,i,j,&list->bc_e[ndof2++]);CHKERRQ(ierr);
          }
        //}
        if (j == Nz-1) {
          //if (dof0) {
            if (i == 0) { loc = DMSTAG_UP_LEFT; }
            else        { loc = DMSTAG_UP_RIGHT; }
            loc = (i == 0) ? DMSTAG_UP_LEFT : DMSTAG_UP_RIGHT;
            for (ii = 0; ii<dof0; ii++) {
              ierr = _DMStagBCFillIndices(dm,loc,ii,i,j,&list->bc_v[ndof0++]);CHKERRQ(ierr);
            }
          //}
          //if (dof1) {
            for (ii = 0; ii<dof1; ii++) {
              ierr = _DMStagBCFillIndices(dm,DMSTAG_UP,ii,i,j,&list->bc_f[ndof1++]);CHKERRQ(ierr);
            }
          //}
        }
      } else {
        if ((j == 0) || (j == Nz-1)) {
          //if (dof0) {
            if (j == 0) { loc = DMSTAG_DOWN_LEFT; }
            else        { loc = DMSTAG_UP_LEFT; }
            loc = (j == 0) ? DMSTAG_DOWN_LEFT : DMSTAG_UP_LEFT;
            for (ii = 0; ii<dof0; ii++) {
              ierr = _DMStagBCFillIndices(dm,loc,ii,i,j,&list->bc_v[ndof0++]);CHKERRQ(ierr);
            }
          //}
          //if (dof1) { // 2 DOFs - LEFT, DOWN
            if (j == 0) { loc = DMSTAG_DOWN; }
            else        { loc = DMSTAG_UP; }
            loc = (j == 0) ? DMSTAG_DOWN : DMSTAG_UP;
            loc1 = DMSTAG_LEFT;
            for (ii = 0; ii<dof1; ii++) {
              ierr = _DMStagBCFillIndices(dm,loc, ii,i,j,&list->bc_f[ndof1++]);CHKERRQ(ierr);
              ierr = _DMStagBCFillIndices(dm,loc1,ii,i,j,&list->bc_f[ndof1++]);CHKERRQ(ierr);
            }
          //}
          //if (dof2) {
            for (ii = 0; ii<dof2; ii++) {
              ierr = _DMStagBCFillIndices(dm,DMSTAG_ELEMENT,ii,i,j,&list->bc_e[ndof2++]);CHKERRQ(ierr);
            }
          //}
          if (i == Nx-2) {
            //if (dof0) {
              if (j == 0) { loc = DMSTAG_DOWN_RIGHT; }
              else        { loc = DMSTAG_UP_RIGHT; }
              loc = (j == 0) ? DMSTAG_DOWN_RIGHT : DMSTAG_UP_RIGHT;
              for (ii = 0; ii<dof0; ii++) {
                ierr = _DMStagBCFillIndices(dm,loc,ii,i,j,&list->bc_v[ndof0++]);CHKERRQ(ierr);
              }
            //}
            //if (dof1) {
              for (ii = 0; ii<dof1; ii++) {
                ierr = _DMStagBCFillIndices(dm,DMSTAG_RIGHT,ii,i,j,&list->bc_f[ndof1++]);CHKERRQ(ierr);
              }
            //}
          }
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

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
  if (!isProduct) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Coordinate DM is not of type DMPRODUCT");
  ierr = DMStagGet1dCoordinateArraysDOFRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
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
  ierr = DMStagRestore1dCoordinateArraysDOFRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMStagBCListSetupCoordinates"
PetscErrorCode DMStagBCListSetupCoordinates(DMStagBCList list)
{
  DM             dmCoord;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = DMGetCoordinateDM(list->dm,&dmCoord);CHKERRQ(ierr);
  if (!dmCoord) SETERRQ(PetscObjectComm((PetscObject)list->dm),PETSC_ERR_ARG_WRONGSTATE,"DM does not have a coordinate DM");
  ierr = _DMStagBCListSetupCoordinates(list);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMStagBCListGetVertexBCs"
PetscErrorCode DMStagBCListGetVertexBCs(DMStagBCList list,PetscInt *nbc,DMStagBC *l[])
{
  PetscFunctionBegin;
  if (nbc) { *nbc = list->nbc_vertex; }
  if (l) { *l = list->bc_v; }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMStagBCListGetFaceBCs"
PetscErrorCode DMStagBCListGetFaceBCs(DMStagBCList list,PetscInt *nbc,DMStagBC *l[])
{
  PetscFunctionBegin;
  if (nbc) { *nbc = list->nbc_face; }
  if (l) { *l = list->bc_f; }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMStagBCListGetElementBCs"
PetscErrorCode DMStagBCListGetElementBCs(DMStagBCList list,PetscInt *nbc,DMStagBC *l[])
{
  PetscFunctionBegin;
  if (nbc) { *nbc = list->nbc_element; }
  if (l) { *l = list->bc_e; }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMStagBCListCreate"
PetscErrorCode DMStagBCListCreate(DM dm,DMStagBCList *list)
{
  PetscErrorCode ierr;
  PetscInt       dim;
  PetscBool      isstag;
  DMStagBCList   l;
  
  PetscFunctionBegin;
  /* check assumptions */
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  if (dim != 2) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Only valid for 2d DM");
  ierr = PetscObjectTypeCompare((PetscObject)dm,DMSTAG,&isstag);CHKERRQ(ierr);
  if (!isstag) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Only valid for DMStag");
  
  ierr = PetscCalloc1(1,&l);CHKERRQ(ierr);
  l->dm = dm;
  l->evaluate = NULL;
  l->context = NULL;
  
  {
    PetscInt Nx,Nz,sx,sz,nx,nz;
    PetscInt ii,i,j,ndof0,ndof1,ndof2,dof0,dof1,dof2;
    
    // Count local boundary dofs
    ierr = DMStagGetGlobalSizes(dm,&Nx,&Nz,NULL);CHKERRQ(ierr);
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
    
    /* optionally setup the coordinates if the DM has coordinates defined */
    {
      DM dmCoord;
      ierr = DMGetCoordinateDM(dm,&dmCoord);CHKERRQ(ierr);
      if (dmCoord) {
        ierr = _DMStagBCListSetupCoordinates(l);CHKERRQ(ierr);
      }
    }
    
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

#undef __FUNCT__
#define __FUNCT__ "DMStagBCListDestroy"
PetscErrorCode DMStagBCListDestroy(DMStagBCList *list)
{
  PetscErrorCode ierr;
  DMStagBCList   l;
  PetscFunctionBegin;
  if (!list) PetscFunctionReturn(0);
  l = *list;
  ierr = PetscFree(l->bc_v);CHKERRQ(ierr);
  ierr = PetscFree(l->bc_f);CHKERRQ(ierr);
  ierr = PetscFree(l->bc_e);CHKERRQ(ierr);
  ierr = PetscFree(l);CHKERRQ(ierr);
  *list = NULL;
  PetscFunctionReturn(0);
}

#if 0
#undef __FUNCT__
#define __FUNCT__ "DMStagBCListTraverse"
PetscErrorCode DMStagBCListTraverse(DMStagBCList list,
                  PetscInt stratum_index,DMStagBCLocation domain_location,
                  PetscErrorCode (*f)(const DMStagStencil*,const PetscScalar*,PetscBool*,PetscScalar*,void*),void *context)
{
  PetscInt       k,n;
  DMStagBC       *bc;
  PetscScalar    val;
  PetscBool      constrained;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  switch (stratum_index) {
    case 0:
      ierr = DMStagBCListGetVertexBCs(list,&n,&bc);CHKERRQ(ierr);
      break;
    case 1:
      ierr = DMStagBCListGetFaceBCs(list,&n,&bc);CHKERRQ(ierr);
      break;
    case 2:
      ierr = DMStagBCListGetElementBCs(list,&n,&bc);CHKERRQ(ierr);
      break;
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Stratum index must be on of {0=vertex,1=face,2=element}");
      break;
  }
  
  for (k=0; k<n; k++) {
    if (bc[k].location != domain_location) continue;
    constrained = PETSC_FALSE;
    val = 0.0;
    ierr = (*f)((const DMStagStencil*)&bc[k].point,(const PetscScalar*)&bc[k].coord,&constrained,&val,context);CHKERRQ(ierr);
    if (constrained) {
      bc[k].val = val;
    }
  }
  PetscFunctionReturn(0);
}
#endif

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
    PetscPrintf(PETSC_COMM_SELF,"    [%D] x,y (%+1.2e,%+1.2e) i,j (%D %D %D)\n",i,bc[i].coord[0],bc[i].coord[1],bc[i].point.i,bc[i].point.j,bc[i].point.loc);
  }
  PetscPrintf(PETSC_COMM_SELF,"  bc_faces: size %D\n",list->nbc_face);
  bc = list->bc_f;
  for (i=0; i<list->nbc_face; i++) {
    PetscPrintf(PETSC_COMM_SELF,"    [%D] x,y (%+1.2e,%+1.2e) i,j (%D %D %D)\n",i,bc[i].coord[0],bc[i].coord[1],bc[i].point.i,bc[i].point.j,bc[i].point.loc);
  }
  
  PetscPrintf(PETSC_COMM_SELF,"  bc_elements: size %D\n",list->nbc_element);
  bc = list->bc_e;
  for (i=0; i<list->nbc_element; i++) {
    PetscPrintf(PETSC_COMM_SELF,"    [%D] x,y (%+1.2e,%+1.2e) i,j (%D %D %D)\n",i,bc[i].coord[0],bc[i].coord[1],bc[i].point.i,bc[i].point.j,bc[i].point.loc);
  }
  PetscFunctionReturn(0);
}
