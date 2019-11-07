
#include "petsc.h"

#include <petsc/private/dmstagimpl.h>

PetscErrorCode private_DMStagGetStencilType(DM dm,DMStagStencilType *stencilType)
{
  DM_Stag * const stag = (DM_Stag*)dm->data;
  *stencilType = stag->stencilType;
  return(0);
}

/* Convert an array of DMStagStencil objects to an array of indices into a local vector.
 The .c fields in pos must always be set (even if to 0).  */
static PetscErrorCode private_DMStagStencilToIndexLocal(DM dm,PetscInt n,const DMStagStencil *pos,PetscInt *ix)
{
  PetscErrorCode        ierr;
  const DM_Stag * const stag = (DM_Stag*)dm->data;
  PetscInt              idx,dim,startGhost[DMSTAG_MAX_DIM];
  const PetscInt        epe = stag->entriesPerElement;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMSTAG);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  
#if defined(PETSC_USE_DEBUG)
  ierr = DMStagGetGhostCorners(dm,&startGhost[0],&startGhost[1],&startGhost[2],NULL,NULL,NULL);CHKERRQ(ierr);
#else
  ierr = DMStagGetGhostCorners(dm,&startGhost[0],&startGhost[1],&startGhost[2],NULL,NULL,NULL);CHKERRQ(ierr);
#endif
  
  if (dim == 1) {
    for (idx=0; idx<n; ++idx) {
      const PetscInt eLocal = pos[idx].i - startGhost[0]; /* Local element number */
      ix[idx] = eLocal * epe + stag->locationOffsets[pos[idx].loc] + pos[idx].c;
    }
  } else if (dim == 2) {
    const PetscInt epr = stag->nGhost[0];
    ierr = DMStagGetGhostCorners(dm,&startGhost[0],&startGhost[1],NULL,NULL,NULL,NULL);CHKERRQ(ierr);
    for (idx=0; idx<n; ++idx) {
      const PetscInt eLocalx = pos[idx].i - startGhost[0];
      const PetscInt eLocaly = pos[idx].j - startGhost[1];
      const PetscInt eLocal = eLocalx + epr*eLocaly;
      ix[idx] = eLocal * epe + stag->locationOffsets[pos[idx].loc] + pos[idx].c;
    }
  } else if (dim == 3) {
    const PetscInt epr = stag->nGhost[0];
    const PetscInt epl = stag->nGhost[0]*stag->nGhost[1];
    ierr = DMStagGetGhostCorners(dm,&startGhost[0],&startGhost[1],&startGhost[2],NULL,NULL,NULL);CHKERRQ(ierr);
    for (idx=0; idx<n; ++idx) {
      const PetscInt eLocalx = pos[idx].i - startGhost[0];
      const PetscInt eLocaly = pos[idx].j - startGhost[1];
      const PetscInt eLocalz = pos[idx].k - startGhost[2];
      const PetscInt eLocal  = epl*eLocalz + epr*eLocaly + eLocalx;
      ix[idx] = eLocal * epe + stag->locationOffsets[pos[idx].loc] + pos[idx].c;
    }
  } else SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Unsupported dimension %d",dim);
  PetscFunctionReturn(0);
}

static PetscInt* convert_in_place(DM dm,PetscInt n,const DMStagStencil *pos)
{
  PetscInt *ix;
  PetscFunctionBegin;
  PetscMalloc1(n,&ix);
  private_DMStagStencilToIndexLocal(dm,n,pos,ix);
  //for (i=0; i<n; i++) printf("ix[%d] = %d\n",i,ix[i]);
  return(ix);
}

static PetscErrorCode FillStencilCentral_2D(DM dm,PetscInt i,PetscInt j,PetscInt Ni,PetscInt Nj,
                                 PetscInt *count,DMStagStencil point[])
{
  PetscInt d,v;
  PetscBool vertices=PETSC_FALSE,edges=PETSC_FALSE,cells=PETSC_FALSE;
  PetscInt dof[3];
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = DMStagGetDOF(dm,&dof[0],&dof[1],&dof[2],NULL);CHKERRQ(ierr);
  if (dof[0] > 0) vertices = PETSC_TRUE;
  if (dof[1] > 0) edges = PETSC_TRUE;
  if (dof[2] > 0) cells = PETSC_TRUE;
  if (vertices) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for vertices");
  
  v = 0;
  if (vertices) {
  }
  
  if (edges) {
    for (d=0; d<dof[1]; d++) {
      point[v].i = i;
      point[v].j = j;
      point[v].loc = DMSTAG_LEFT;
      point[v].c = d;
      v++;
    }
    
    for (d=0; d<dof[1]; d++) {
      point[v].i = i;
      point[v].j = j;
      point[v].loc = DMSTAG_RIGHT;
      point[v].c = d;
      v++;
    }
    
    for (d=0; d<dof[1]; d++) {
      point[v].i = i;
      point[v].j = j;
      point[v].loc = DMSTAG_UP;
      point[v].c = d;
      v++;
    }
    
    for (d=0; d<dof[1]; d++) {
      point[v].i = i;
      point[v].j = j;
      point[v].loc = DMSTAG_DOWN;
      point[v].c = d;
      v++;
    }
  }
  if (cells) {
    for (d=0; d<dof[2]; d++) {
      point[v].i = i;
      point[v].j = j;
      point[v].loc = DMSTAG_ELEMENT;
      point[v].c = d;
      v++;
    }
  }
  *count = v;
  PetscFunctionReturn(0);
}

static PetscErrorCode FillStencilBox_2D(DM dm,PetscInt i,PetscInt j,PetscInt Ni,PetscInt Nj,
                                 PetscInt *count,DMStagStencil point[])
{
  PetscInt d,ii,jj,si,sj,ei,ej,v,sw;
  PetscBool vertices=PETSC_FALSE,edges=PETSC_FALSE,cells=PETSC_FALSE;
  PetscInt dof[3];
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = DMStagGetStencilWidth(dm,&sw);CHKERRQ(ierr);
  ierr = DMStagGetDOF(dm,&dof[0],&dof[1],&dof[2],NULL);CHKERRQ(ierr);
  if (dof[0] > 0) vertices = PETSC_TRUE;
  if (dof[1] > 0) edges = PETSC_TRUE;
  if (dof[2] > 0) cells = PETSC_TRUE;
  if (vertices) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for vertices");
  
  si = PetscMax(0,i-sw);    sj = PetscMax(0,j-sw);
  ei = PetscMin(Ni-1,i+sw); ej = PetscMin(Nj-1,j+sw);
  
  v = 0;
  for (jj=sj; jj<=ej; jj++) {
    for (ii=si; ii<=ei; ii++) {
      if (vertices) {
      }
      
      if (edges) {
        for (d=0; d<dof[1]; d++) {
          point[v].i = ii;
          point[v].j = jj;
          point[v].loc = DMSTAG_LEFT;
          point[v].c = d;
          v++;
        }
        
        for (d=0; d<dof[1]; d++) {
          point[v].i = ii;
          point[v].j = jj;
          point[v].loc = DMSTAG_RIGHT;
          point[v].c = d;
          v++;
        }
        
        for (d=0; d<dof[1]; d++) {
          point[v].i = ii;
          point[v].j = jj;
          point[v].loc = DMSTAG_UP;
          point[v].c = d;
          v++;
        }
        
        for (d=0; d<dof[1]; d++) {
          point[v].i = ii;
          point[v].j = jj;
          point[v].loc = DMSTAG_DOWN;
          point[v].c = d;
          v++;
        }
      }
      if (cells) {
        for (d=0; d<dof[2]; d++) {
          point[v].i = ii;
          point[v].j = jj;
          point[v].loc = DMSTAG_ELEMENT;
          point[v].c = d;
          v++;
        }
      }
    }
  }
  *count = v;
  PetscFunctionReturn(0);
}

static PetscErrorCode FillStencilStar_2D(DM dm,PetscInt i,PetscInt j,PetscInt Ni,PetscInt Nj,
                                 PetscInt *count,DMStagStencil point[])
{
  PetscInt d,ii,v,sw,star_i[13],star_j[13],nvmax;
  PetscBool vertices=PETSC_FALSE,edges=PETSC_FALSE,cells=PETSC_FALSE;
  PetscInt dof[3];
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = DMStagGetStencilWidth(dm,&sw);CHKERRQ(ierr);
  if (sw > 2) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for anything other than star stencil width of 0, 1 or 2");
  ierr = DMStagGetDOF(dm,&dof[0],&dof[1],&dof[2],NULL);CHKERRQ(ierr);
  if (dof[0] > 0) vertices = PETSC_TRUE;
  if (dof[1] > 0) edges = PETSC_TRUE;
  if (dof[2] > 0) cells = PETSC_TRUE;
  if (vertices) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for vertices");
  
  /* sw = 0 */
  nvmax = 0;
  if (sw == 0) {
    star_i[nvmax] = i;   star_j[nvmax] = j;   nvmax++;
  }
  
  /* sw = 1 */
  /*
           [i,j+1]
   [i-1,j] [i,j]   [i+1,j]
           [i,j-1]
  */
  if (sw == 1) {
    star_i[nvmax] = i;   star_j[nvmax] = j+1; nvmax++;
    //
    star_i[nvmax] = i-1; star_j[nvmax] = j;   nvmax++;
    star_i[nvmax] = i;   star_j[nvmax] = j;   nvmax++;
    star_i[nvmax] = i+1; star_j[nvmax] = j;   nvmax++;
    //
    star_i[nvmax] = i;   star_j[nvmax] = j-1; nvmax++;
  }
  
  /* sw = 2 */
  /*
                       [i,j+2]
             [i-1,j+1] [i,j+1] [i+1,j+1]
     [i-2,j] [i-1,j]   [i,j]   [i+1,j]   [i+2,j]
             [i-1,j-1] [i,j-1] [i+1,j-1]
                       [i,j-2]
  */
  if (sw == 2) {
    star_i[nvmax] = i;   star_j[nvmax] = j+2; nvmax++;
    //
    star_i[nvmax] = i-1; star_j[nvmax] = j+1; nvmax++;
    star_i[nvmax] = i;   star_j[nvmax] = j+1; nvmax++;
    star_i[nvmax] = i+1; star_j[nvmax] = j+1; nvmax++;
    //
    star_i[nvmax] = i-2; star_j[nvmax] = j;   nvmax++;
    star_i[nvmax] = i-1; star_j[nvmax] = j;   nvmax++;
    star_i[nvmax] = i;   star_j[nvmax] = j;   nvmax++;
    star_i[nvmax] = i+1; star_j[nvmax] = j;   nvmax++;
    star_i[nvmax] = i+2; star_j[nvmax] = j;   nvmax++;
    //
    star_i[nvmax] = i-1; star_j[nvmax] = j-1; nvmax++;
    star_i[nvmax] = i;   star_j[nvmax] = j-1; nvmax++;
    star_i[nvmax] = i+1; star_j[nvmax] = j-1; nvmax++;
    //
    star_i[nvmax] = i;   star_j[nvmax] = j-2; nvmax++;
  }
  if (nvmax == 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for anything other than star stencil width of 0, 1 or 2");
  
  /*
  for (ii=0; ii<nvmax; ii++) {
    star_i[ii] = PetscMax(0,star_i[ii]);
    star_i[ii] = PetscMin(Ni-1,star_i[ii]);
    
    star_j[ii] = PetscMax(0,star_j[ii]);
    star_j[ii] = PetscMin(Nj-1,star_j[ii]);
  }
  */
  for (ii=0; ii<nvmax; ii++) {
    if (star_i[ii] < 0) { star_i[ii] = 0; }
    if (star_j[ii] < 0) { star_j[ii] = 0; }
    if (star_i[ii] > Ni-1) { star_i[ii] = Ni-1; }
    if (star_j[ii] > Nj-1) { star_j[ii] = Nj-1; }
  }
  
  v = 0;
  for (ii=0; ii<nvmax; ii++) {
    if (vertices) {
    }
    
    if (edges) {
      for (d=0; d<dof[1]; d++) {
        point[v].i = star_i[ii];
        point[v].j = star_j[ii];
        point[v].loc = DMSTAG_LEFT;
        point[v].c = d;
        v++;
      }
      
      for (d=0; d<dof[1]; d++) {
        point[v].i = star_i[ii];
        point[v].j = star_j[ii];
        point[v].loc = DMSTAG_RIGHT;
        point[v].c = d;
        v++;
      }
      
      for (d=0; d<dof[1]; d++) {
        point[v].i = star_i[ii];
        point[v].j = star_j[ii];
        point[v].loc = DMSTAG_UP;
        point[v].c = d;
        v++;
      }
      
      for (d=0; d<dof[1]; d++) {
        point[v].i = star_i[ii];
        point[v].j = star_j[ii];
        point[v].loc = DMSTAG_DOWN;
        point[v].c = d;
        v++;
      }
    }
    if (cells) {
      for (d=0; d<dof[2]; d++) {
        point[v].i = star_i[ii];
        point[v].j = star_j[ii];
        point[v].loc = DMSTAG_ELEMENT;
        point[v].c = d;
        v++;
      }
    }
  }
  *count = v;
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateStencilBuffer_2D(DM dm,PetscInt *_max_size,DMStagStencil *point[])
{
  PetscErrorCode ierr;
  PetscInt sw,size=0,cellsize,dof[3];
  PetscBool vertices=PETSC_FALSE,edges=PETSC_FALSE,cells=PETSC_FALSE;
  DMStagStencilType stencilType;
  
  PetscFunctionBegin;
  ierr = DMStagGetStencilWidth(dm,&sw);CHKERRQ(ierr);
  ierr = DMStagGetDOF(dm,&dof[0],&dof[1],&dof[2],NULL);CHKERRQ(ierr);
  if (dof[0] > 0) vertices = PETSC_TRUE;
  if (dof[1] > 0) edges = PETSC_TRUE;
  if (dof[2] > 0) cells = PETSC_TRUE;
  cellsize = 0;
  if (vertices) cellsize += 4 * dof[0]; /* count all vertices */
  if (edges)    cellsize += 4 * dof[1]; /* counter left/right/up/down edges */
  if (cells)    cellsize += dof[2];
  ierr = private_DMStagGetStencilType(dm,&stencilType);CHKERRQ(ierr);
  switch (stencilType) {
    case  DMSTAG_STENCIL_STAR:
    if (sw > 2) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for anything other than star stencil width of 1 or 2");
    if (sw == 0) {
      size = cellsize;
    } else if (sw == 1) {
      size = 5 * cellsize;
    } else if (sw == 2) {
      size = 13 * cellsize;
    } else {
      size = 0;
    }
    break;

    case  DMSTAG_STENCIL_BOX:
    size = (2*sw+1) * (2*sw+1) * cellsize;
    break;

    default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only stencil type DMSTAG_STENCIL_STAR and DMSTAG_STENCIL_BOX supported");
    break;
  }
  
  ierr = PetscCalloc1(size,point);CHKERRQ(ierr);
  *_max_size = size;
  
  PetscFunctionReturn(0);
}

static PetscErrorCode _preallocate_coupled(Mat p,PetscInt i,DM row_dm,PetscInt ncols,DM cols_dm[],PetscInt offset[],PetscBool col_mask[])
{
  PetscErrorCode ierr;
  PetscInt j;
  PetscInt *rowidx,*colidx;
  PetscInt ci,cj,sx,sz,nx,nz,Ni,Nj,ii,jj,d;
  PetscInt **indices;
  DMStagStencil *r_point_buffer,**c_point_buffer;
  PetscInt remap,r_max_size,*c_max_size;
  PetscErrorCode (*fill_stencil[50])(DM,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt*,DMStagStencil*);
  DMStagStencilType stencilType;
  PetscInt r_used,c_used;
  PetscMPIInt rank;
  
  PetscFunctionBegin;
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  for (j=0; j<ncols; j++) {
    ierr = private_DMStagGetStencilType(cols_dm[j],&stencilType);CHKERRQ(ierr);
    switch (stencilType) {
      case DMSTAG_STENCIL_STAR:
      fill_stencil[j] = FillStencilStar_2D;
      break;
      case DMSTAG_STENCIL_BOX:
      fill_stencil[j] = FillStencilBox_2D;
      break;
      default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only stencil type DMSTAG_STENCIL_STAR and DMSTAG_STENCIL_BOX supported");
      break;
    }
  }
  
  ierr = DMStagGetGlobalSizes(row_dm,&Ni,&Nj,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(row_dm,&sx,&sz,NULL,&nx,&nz,NULL,NULL,NULL,NULL); CHKERRQ(ierr);
  
  ierr = PetscCalloc1(ncols,&indices);CHKERRQ(ierr);
  for (d=0; d<ncols; d++) {
    ISLocalToGlobalMapping ltog;
    PetscInt ltog_size;
    
    ierr = DMGetLocalToGlobalMapping(cols_dm[d],&ltog);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingGetIndices(ltog,(const PetscInt**)&indices[d]);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingGetSize(ltog,&ltog_size);CHKERRQ(ierr);
    printf("[%d] ltog size %d\n",d,ltog_size);
  }
  
  /* insert */
  ierr = CreateStencilBuffer_2D(row_dm,&r_max_size,&r_point_buffer);CHKERRQ(ierr);
  printf("r_max_size[%d] = %d\n",i,r_max_size);
  
  ierr = PetscCalloc1(ncols,&c_point_buffer);CHKERRQ(ierr);
  ierr = PetscCalloc1(ncols,&c_max_size);CHKERRQ(ierr);
  for (j=0; j<ncols; j++) {
    ierr = CreateStencilBuffer_2D(cols_dm[j],&c_max_size[j],&c_point_buffer[j]);CHKERRQ(ierr);
    printf("[%d] c_max_size[%d] = %d\n",i,j,c_max_size[j]);
  }
  
  for (cj=sz; cj<sz+nz; cj++) {
    for (ci=sx; ci<sx+nx; ci++) {
      
      ierr = FillStencilCentral_2D(row_dm,ci,cj,Ni,Nj,&r_used,r_point_buffer);CHKERRQ(ierr);
      
      rowidx = convert_in_place(row_dm,r_used,r_point_buffer);CHKERRQ(ierr);
      for (ii=0; ii<r_used; ii++) {
        //printf("ii %d : rowidx[jj] %d\n",ii,rowidx[ii]);
        if (rowidx[ii] < 0) { continue; }
        remap = indices[i][ rowidx[ii] ];
        //printf("-->ii %d : remap %d\n",ii,remap);
        rowidx[ii] = remap + offset[i];
        //printf("-->-->ii %d : rowidx[jj] %d\n",ii,rowidx[ii]);
      }
      
      for (j=0; j<ncols; j++) {
        if (col_mask[j]) continue;
        
        ierr = fill_stencil[j](cols_dm[j],ci,cj,Ni,Nj,&c_used,c_point_buffer[j]);CHKERRQ(ierr);
        
        colidx = convert_in_place(cols_dm[j],c_used,c_point_buffer[j]);CHKERRQ(ierr);
        
        for (jj=0; jj<c_used; jj++) {
          //printf("jj %d : colidx[jj] %d\n",jj,colidx[jj]);
          if (colidx[jj] < 0) { continue; }
          remap = indices[j][ colidx[jj] ];
          colidx[jj] = remap + offset[j];
        }

        /*
        printf("i %d j %d (ci %d cj %d) ->\n",i,j,ci,cj);
        for (ii=0; ii<r_used; ii++)  printf(" %d ",rowidx[ii]); printf("\n");
        for (jj=0; jj<c_used; jj++)  printf(" %d ",colidx[jj]); printf("\n");
        */
        
        for (ii=0; ii<r_used; ii++) {
          for (jj=0; jj<c_used; jj++) {
            {ierr = MatSetValue(p,rowidx[ii],colidx[jj],1.0,INSERT_VALUES);CHKERRQ(ierr);}
          }
        }
        ierr = PetscFree(colidx);CHKERRQ(ierr);
      }
      ierr = PetscFree(rowidx);CHKERRQ(ierr);
    }
  }
  
  ierr = MatAssemblyBegin(p,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(p,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  
  for (j=0; j<ncols; j++) {
    ierr = PetscFree(c_point_buffer[j]);CHKERRQ(ierr);
  }
  ierr = PetscFree(c_point_buffer);CHKERRQ(ierr);
  ierr = PetscFree(c_max_size);CHKERRQ(ierr);
  ierr = PetscFree(r_point_buffer);CHKERRQ(ierr);
  
  ierr = PetscFree(indices);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode FDPDECoupledCreateMatrix(PetscInt ndm,DM dm[],MatType mtype,Mat *A)
{
  PetscErrorCode ierr;
  PetscInt i,d;
  PetscInt *offset,*m,*n,*M,*N,sizes[] = {0,0,0,0},Mo;
  PetscBool *col_mask;
  Mat preallocator;
  
  PetscFunctionBegin;
  /* check all DMs are DMSTAG */
  for (d=0; d<ndm; d++) {
    PetscBool isstag;
    ierr = PetscObjectTypeCompare((PetscObject)dm[d],DMSTAG,&isstag);CHKERRQ(ierr);
    if (!isstag) SETERRQ1(PetscObjectComm((PetscObject)dm[d]),PETSC_ERR_ARG_WRONG,"DM[%D] is not of type DMSTAG",d);
  }
  
  /* check sizes are consistent */
  {
    PetscInt Ni,Nj,jNi,jNj;
    
    ierr = DMStagGetGlobalSizes(dm[0],&Ni,&Nj,NULL);CHKERRQ(ierr);
    for (d=1; d<ndm; d++) {
      ierr = DMStagGetGlobalSizes(dm[d],&jNi,&jNj,NULL);CHKERRQ(ierr);
      if (Ni != jNi) SETERRQ2(PetscObjectComm((PetscObject)dm[0]),PETSC_ERR_ARG_WRONG,"DM (Ni=%D) does not match size of DM[0] (Ni=%D)",Ni,jNi);
      if (Nj != jNj) SETERRQ2(PetscObjectComm((PetscObject)dm[0]),PETSC_ERR_ARG_WRONG,"DM (Nj=%D) does not match size of DM[0] (Nj=%D)",Nj,jNj);
    }
  }
  
  /* determine global and local sizes */
  /* compute offsets for insertions */
  ierr = PetscCalloc1(ndm,&offset);CHKERRQ(ierr);
  ierr = PetscCalloc1(ndm,&col_mask);CHKERRQ(ierr);
  ierr = PetscCalloc1(ndm,&m);CHKERRQ(ierr);
  ierr = PetscCalloc1(ndm,&n);CHKERRQ(ierr);
  ierr = PetscCalloc1(ndm,&M);CHKERRQ(ierr);
  ierr = PetscCalloc1(ndm,&N);CHKERRQ(ierr);
  for (d=0; d<ndm; d++) {
    col_mask[d] = PETSC_FALSE;
  }
  for (d=0; d<ndm; d++) {
    Vec x;
    PetscInt size;
    DMCreateGlobalVector(dm[d],&x);CHKERRQ(ierr);
    VecGetSize(x,&size);
    M[d] = size;
    N[d] = size;
    VecGetLocalSize(x,&size);
    m[d] = size;
    n[d] = size;
    VecDestroy(&x);
  }
  for (d=1; d<ndm; d++) {
    offset[d] = offset[d-1] + m[d-1];
  }
  for (d=0; d<ndm; d++) {
    sizes[0] += m[d];
    sizes[1] += n[d];
    sizes[2] += M[d];
    sizes[3] += N[d];
  }
  Mo = 0;
  /*ierr = MPI_Scan(&sizes[0], &Mo, 1, MPI_INT, MPI_SUM, PetscObjectComm((PetscObject)dm[0]));CHKERRQ(ierr);*/
  /*Mo -= sizes[0];*/
  for (d=0; d<ndm; d++) {
    offset[d] += Mo;
  }
  
  for (d=0; d<ndm; d++) {
    printf("[%d] MxN %d %d mxn %d %d\n",d,M[d],N[d],m[d],n[d]);
    printf("offset[%d] %d\n",d,offset[d]);
  }
  printf("sizes MxN %d %d <global>\n",sizes[2],sizes[3]);
  printf("sizes mxn %d %d <local>\n",sizes[0],sizes[1]);
  
  ierr = MatCreate(PetscObjectComm((PetscObject)dm[0]),&preallocator);CHKERRQ(ierr);
  ierr = MatSetSizes(preallocator,sizes[0],sizes[1],sizes[2],sizes[3]);CHKERRQ(ierr);
  ierr = MatSetType(preallocator,MATPREALLOCATOR);CHKERRQ(ierr);
  ierr = MatSetUp(preallocator);CHKERRQ(ierr);
  
  /* preallocate */
  for (i=0; i<ndm; i++) {
    DM row_dm,*cols_dm;
    
    row_dm  = dm[i];
    cols_dm = dm;
    
    ierr = _preallocate_coupled(preallocator,i,row_dm,ndm,cols_dm,offset,col_mask);CHKERRQ(ierr);
  }
  
  ierr = MatCreate(PetscObjectComm((PetscObject)preallocator),A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,sizes[0],sizes[1],sizes[2],sizes[3]);CHKERRQ(ierr);
  ierr = MatSetType(*A,mtype);CHKERRQ(ierr);
  ierr = MatSetFromOptions(*A);CHKERRQ(ierr);
  ierr = MatSetUp(*A);CHKERRQ(ierr);
  ierr = MatPreallocatorPreallocate(preallocator,PETSC_TRUE,*A);CHKERRQ(ierr);
  
  ierr = MatDestroy(&preallocator);CHKERRQ(ierr);
  ierr = PetscFree(col_mask);CHKERRQ(ierr);
  ierr = PetscFree(M);CHKERRQ(ierr);
  ierr = PetscFree(N);CHKERRQ(ierr);
  ierr = PetscFree(m);CHKERRQ(ierr);
  ierr = PetscFree(n);CHKERRQ(ierr);
  ierr = PetscFree(offset);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode FDPDECoupledCreateMatrix2(PetscInt ndm,DM dm[],PetscBool mask[],MatType mtype,Mat *A)
{
  PetscErrorCode ierr;
  PetscInt i,j,d;
  PetscInt *offset,*m,*n,*M,*N,sizes[] = {0,0,0,0};
  PetscBool *col_mask;
  Mat preallocator;
  
  PetscFunctionBegin;
  /* check all DMs are DMSTAG */
  for (d=0; d<ndm; d++) {
    PetscBool isstag;
    ierr = PetscObjectTypeCompare((PetscObject)dm[d],DMSTAG,&isstag);CHKERRQ(ierr);
    if (!isstag) SETERRQ1(PetscObjectComm((PetscObject)dm[d]),PETSC_ERR_ARG_WRONG,"DM[%D] is not of type DMSTAG",d);
  }
  
  /* check sizes are consistent */
  {
    PetscInt Ni,Nj,jNi,jNj;
    
    ierr = DMStagGetGlobalSizes(dm[0],&Ni,&Nj,NULL);CHKERRQ(ierr);
    for (d=1; d<ndm; d++) {
      ierr = DMStagGetGlobalSizes(dm[d],&jNi,&jNj,NULL);CHKERRQ(ierr);
      if (Ni != jNi) SETERRQ2(PetscObjectComm((PetscObject)dm[0]),PETSC_ERR_ARG_WRONG,"DM (Ni=%D) does not match size of DM[0] (Ni=%D)",Ni,jNi);
      if (Nj != jNj) SETERRQ2(PetscObjectComm((PetscObject)dm[0]),PETSC_ERR_ARG_WRONG,"DM (Nj=%D) does not match size of DM[0] (Nj=%D)",Nj,jNj);
    }
  }
  
  /* determine global and local sizes */
  /* compute offsets for insertions */
  ierr = PetscCalloc1(ndm,&offset);CHKERRQ(ierr);
  ierr = PetscCalloc1(ndm,&col_mask);CHKERRQ(ierr);
  ierr = PetscCalloc1(ndm,&m);CHKERRQ(ierr);
  ierr = PetscCalloc1(ndm,&n);CHKERRQ(ierr);
  ierr = PetscCalloc1(ndm,&M);CHKERRQ(ierr);
  ierr = PetscCalloc1(ndm,&N);CHKERRQ(ierr);
  for (d=0; d<ndm; d++) {
    col_mask[d] = PETSC_FALSE;
  }
  for (d=0; d<ndm; d++) {
    Vec x;
    PetscInt size;
    DMCreateGlobalVector(dm[d],&x);CHKERRQ(ierr);
    VecGetSize(x,&size);
    M[d] = size;
    N[d] = size;
    VecGetLocalSize(x,&size);
    m[d] = size;
    n[d] = size;
    VecDestroy(&x);
  }
  for (d=1; d<ndm; d++) {
    offset[d] = offset[d-1] + m[d-1];
  }
  for (d=0; d<ndm; d++) {
    sizes[0] += m[d];
    sizes[1] += n[d];
    sizes[2] += M[d];
    sizes[3] += N[d];
  }
  for (d=0; d<ndm; d++) {
    printf("[%d] MxN %d %d mxn %d %d\n",d,M[d],N[d],m[d],n[d]);
    printf("offset[%d] %d\n",d,offset[d]);
  }
  printf("sizes MxN %d %d <global>\n",sizes[2],sizes[3]);
  printf("sizes mxn %d %d <local>\n",sizes[0],sizes[1]);
  
  ierr = MatCreate(PetscObjectComm((PetscObject)dm[0]),&preallocator);CHKERRQ(ierr);
  ierr = MatSetSizes(preallocator,sizes[0],sizes[1],sizes[2],sizes[3]);CHKERRQ(ierr);
  ierr = MatSetType(preallocator,MATPREALLOCATOR);CHKERRQ(ierr);
  ierr = MatSetUp(preallocator);CHKERRQ(ierr);
  
  /* preallocate */
  for (i=0; i<ndm; i++) {
    DM row_dm,*cols_dm;
    
    row_dm  = dm[i];
    cols_dm = dm;
    
    for (j=0; j<ndm; j++) {
      col_mask[j] = mask[i*ndm + j];
    }
    
    ierr = _preallocate_coupled(preallocator,i,row_dm,ndm,cols_dm,offset,col_mask);CHKERRQ(ierr);
  }
  
  ierr = MatCreate(PetscObjectComm((PetscObject)preallocator),A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,sizes[0],sizes[1],sizes[2],sizes[3]);CHKERRQ(ierr);
  ierr = MatSetType(*A,mtype);CHKERRQ(ierr);
  ierr = MatSetFromOptions(*A);CHKERRQ(ierr);
  ierr = MatSetUp(*A);CHKERRQ(ierr);
  ierr = MatPreallocatorPreallocate(preallocator,PETSC_TRUE,*A);CHKERRQ(ierr);
  
  ierr = MatDestroy(&preallocator);CHKERRQ(ierr);
  ierr = PetscFree(col_mask);CHKERRQ(ierr);
  ierr = PetscFree(M);CHKERRQ(ierr);
  ierr = PetscFree(N);CHKERRQ(ierr);
  ierr = PetscFree(m);CHKERRQ(ierr);
  ierr = PetscFree(n);CHKERRQ(ierr);
  ierr = PetscFree(offset);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}
