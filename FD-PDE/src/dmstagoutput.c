/* Output routines for DMStag */
#include "dmstagoutput.h"

// ---------------------------------------
/*@
DMStagOutputGetLabels - retrieve a structure for labels

Input parameter:
dm - DMStag object

Output parameters:
labels - type DMStagOutputLabel, length = total DMStag dofs

Notes:
If the user does not assign labels, then automatic labels will be created.

Use: user
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "DMStagOutputGetLabels"
PetscErrorCode DMStagOutputGetLabels(DM dm, DMStagOutputLabel **labels)
{
  DMStagOutputLabel *l;
  PetscInt       i, dof0, dof1, dof2;
  PetscInt       ndof = 0, idof;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  // Get dofs number
  ierr = DMStagGetDOF(dm,&dof0,&dof1,&dof2,NULL);CHKERRQ(ierr);
  ndof  = dof0 + dof1 + dof2;

  // Allocate memory for labels
  ierr = PetscMalloc((size_t)ndof*sizeof(DMStagOutputLabel),&l);CHKERRQ(ierr);
  ierr = PetscMemzero(l,(size_t)ndof*sizeof(DMStagOutputLabel));CHKERRQ(ierr);

  // Add default information - need error checks
  idof = 0;
  for (i = idof; i<dof0+idof; i++) {
    l[i].name[0] = '\0';
    l[i].c       = i-idof;
    l[i].type    = OUT_VERTEX;
    l[i].filled  = PETSC_FALSE;
  }
  idof += dof0;

  for (i = idof; i<dof1+idof; i++) {
    l[i].name[0] = '\0';
    l[i].c       = i-idof;
    l[i].type    = OUT_FACE;
    l[i].filled  = PETSC_FALSE;
  }
  idof += dof1;

  for (i = idof; i<dof2+idof; i++) {
    l[i].name[0] = '\0';
    l[i].c       = i-idof;
    l[i].type    = OUT_ELEMENT;
    l[i].filled  = PETSC_FALSE;
  }

  *labels = l;

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
DMStagOutputAddLabel

@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "DMStagOutputAddLabel"
PetscErrorCode DMStagOutputAddLabel(DM dm, DMStagOutputLabel *labels, const char name[], PetscInt c, DMStagStencilLocation loc)
{
  OutputType     ilabel;
  PetscInt       i, dof0, dof1, dof2, ndof, idof;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  // Get dofs number
  ierr = DMStagGetDOF(dm,&dof0,&dof1,&dof2,NULL);CHKERRQ(ierr);
  ndof  = dof0 + dof1 + dof2;

  if ((loc == DMSTAG_DOWN_LEFT) || (loc == DMSTAG_UP_LEFT) || (loc == DMSTAG_DOWN_RIGHT) || (loc == DMSTAG_UP_RIGHT)) {
    idof   = dof0;
    ilabel = OUT_VERTEX;
  }

  if ((loc == DMSTAG_DOWN) || (loc == DMSTAG_UP) || (loc == DMSTAG_RIGHT) || (loc == DMSTAG_LEFT)) {
    idof   = dof1;
    ilabel = OUT_FACE;
  }

  if (loc == DMSTAG_ELEMENT) {
    idof   = dof2;
    ilabel = OUT_ELEMENT;
  }

  if (c > idof) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Requested label for dof is invalid with the DMStag structure!");

  // add label
  for (i = 0; i<ndof; i++) {
    if ((labels[i].c == c) && (labels[i].type == ilabel)) {
      ierr = PetscStrcpy(labels[i].name,name); CHKERRQ(ierr);
      labels[i].filled  = PETSC_TRUE;
    }
  }

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
DMStagOutputVTKBinary - writes to a VTK file the DMStag and associated vector 

Input parameter:
dm - DMStag object
x - associated vector
fname - file name

Notes:
Assumptions are 1) plot center (element) values. 2) edge values are vector components (not always true).

Use: user
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "DMStagOutputVTKBinary"
PetscErrorCode DMStagOutputVTKBinary(DM dm, Vec x, DMStagOutputLabel *labels, const char fname[])
{
  Vec            xglobal, xlocal;
  PetscInt       dof = 0;
  PetscInt       dim, sx, sz, sz1, nx, nz, Nx, Nz;
  PetscInt       idof, ndof, dof0, dof1, dof2; 
  PetscMPIInt    size;
  PetscBool      isstag, flg;
  PetscScalar    **coordx,**coordz;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  // Check assumptions
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  if (dim != 2) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Only valid for 2D DMs");

  ierr = PetscObjectTypeCompare((PetscObject)dm,DMSTAG,&isstag);CHKERRQ(ierr);
  if (!isstag) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Only valid for DMStag");

  ierr = DMGetGlobalVector(dm,&xglobal);CHKERRQ(ierr);
  ierr = VecGetSize(x,&sz);
  ierr = VecGetSize(xglobal,&sz1);

  ierr = DMRestoreGlobalVector(dm,&xglobal);CHKERRQ(ierr);
  if(sz!=sz1) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_USER,"The provided vector does not match the global dimensions of the DM.");

  // Check serial/parallel
  ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size); CHKERRQ(ierr);
  if (size>1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"This output routine is not implemented in parallel!");

  // Prepare data for output - create buffers for vertex, face, element
  ierr = DMStagGetDOF(dm,&dof0,&dof1,&dof2,NULL);CHKERRQ(ierr);
  ierr = DMStagGetGlobalSizes(dm,&Nx,&Nz,NULL);CHKERRQ(ierr);

  // Get local domain
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  // Map global vectors to local domain
  ierr = DMGetLocalVector(dm, &xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm, x, INSERT_VALUES, xlocal); CHKERRQ(ierr);

  // Get dm coordinates array
  ierr = DMStagGet1dCoordinateArraysDOFRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);

  ndof = dof0+dof1+dof2;

  DMStagOutputBuffer *buff_v, *buff_f, *buff_e;

  buff_v = NULL;
  buff_f = NULL;
  buff_e = NULL;

  // Prepare vertex data
  if (dof0) {
    PetscInt i, j, ii, ilabel;

    // allocate memory
    ierr = PetscMalloc((size_t)dof0*sizeof(DMStagOutputBuffer),&buff_v);CHKERRQ(ierr);
    ierr = PetscMemzero(buff_v,(size_t)dof0*sizeof(DMStagOutputBuffer));CHKERRQ(ierr);

    for (ii = 0; ii<dof0; ii++) {
      // alocate memory for data
      ierr = PetscMalloc((size_t)(Nx*Nz)*sizeof(PetscScalar),&buff_v[ii].data);CHKERRQ(ierr);
      ierr = PetscMemzero(buff_v[ii].data,(size_t)(Nx*Nz)*sizeof(PetscScalar));CHKERRQ(ierr);

      // size
      buff_v[ii].size = (size_t)(Nx*Nz)*sizeof(PetscScalar);

      // label
      for (ilabel = 0; ilabel<ndof; ilabel++) {
        if ((labels[ilabel].c == ii) && (labels[ilabel].type == OUT_VERTEX)) {
          if (labels[ilabel].filled) {
            buff_v[ii].name = labels[ilabel].name;
          } else {
            char dname[OUTPUT_NAME_LENGTH];
            sprintf(dname,"Vertex variable %d",ilabel); 
            ierr = PetscStrallocpy(dname,&buff_v[ii].name);CHKERRQ(ierr);
          }
        }
      }

      // loop domain and calculate data - interp
      for (j = sz; j<sz+nz; j++) {
        for (i = sx; i<sx+nx; i++) {
          PetscScalar   fval = 0.0, xx[4];
          DMStagStencil point[4];

          point[0].i = i; point[0].j = j; point[0].loc = DMSTAG_DOWN_LEFT;  point[0].c = ii;
          point[1].i = i; point[1].j = j; point[1].loc = DMSTAG_DOWN_RIGHT; point[1].c = ii;
          point[2].i = i; point[2].j = j; point[2].loc = DMSTAG_UP_LEFT;    point[2].c = ii;
          point[3].i = i; point[3].j = j; point[3].loc = DMSTAG_UP_RIGHT;   point[3].c = ii;

          ierr = DMStagVecGetValuesStencil(dm,xlocal,4,point,xx); CHKERRQ(ierr);
          fval = (xx[0]+xx[1]+xx[2]+xx[3])*0.25;
          buff_v[ii].data[i+j*nz] = fval;
        }
      }
    }
  }

  // Prepare face data - vectors
  if (dof1) {
    PetscInt i, j, ii, ndof1, ilabel;

    // allocate memory
    ierr = PetscMalloc((size_t)dof1*sizeof(DMStagOutputBuffer),&buff_f);CHKERRQ(ierr);
    ierr = PetscMemzero(buff_f,(size_t)dof1*sizeof(DMStagOutputBuffer));CHKERRQ(ierr);

    for (ii = 0; ii<dof1; ii++) {
      // alocate memory for data *3
      ierr = PetscMalloc((size_t)(3*Nx*Nz)*sizeof(PetscScalar),&buff_f[ii].data);CHKERRQ(ierr);
      ierr = PetscMemzero(buff_f[ii].data,(size_t)(3*Nx*Nz)*sizeof(PetscScalar));CHKERRQ(ierr);

      // size
      buff_f[ii].size = (size_t)(3*Nx*Nz)*sizeof(PetscScalar);

      // label - only 3*ii
      for (ilabel = 0; ilabel<ndof; ilabel++) {
        if ((labels[ilabel].c == ii) && (labels[ilabel].type == OUT_FACE)) {
          if (labels[ilabel].filled) {
              buff_f[ii].name = labels[ilabel].name;
          } else {
            char dname[OUTPUT_NAME_LENGTH];
            sprintf(dname,"Face variable %d",ilabel); 
            ierr = PetscStrallocpy(dname,&buff_f[ii].name);CHKERRQ(ierr);
          }
        }
      }

      // loop domain and calculate data - interp
      for (j = sz; j<sz+nz; j++) {
        for (i = sx; i<sx+nx; i++) {
          PetscScalar   fval = 0.0, xx[4];
          DMStagStencil point[4];

          point[0].i = i; point[0].j = j; point[0].loc = DMSTAG_LEFT;  point[0].c = ii;
          point[1].i = i; point[1].j = j; point[1].loc = DMSTAG_RIGHT; point[1].c = ii;
          point[2].i = i; point[2].j = j; point[2].loc = DMSTAG_DOWN;  point[2].c = ii;
          point[3].i = i; point[3].j = j; point[3].loc = DMSTAG_UP;    point[3].c = ii;

          ierr = DMStagVecGetValuesStencil(dm,xlocal,4,point,xx); CHKERRQ(ierr);

          buff_f[ii].data[3*(i+j*nz)+0] = (xx[0]+xx[1])*0.5;
          buff_f[ii].data[3*(i+j*nz)+1] = (xx[2]+xx[3])*0.5;
          buff_f[ii].data[3*(i+j*nz)+2] = fval;
        }
      }
    }
  }

  // Prepare element data
  if (dof2) {
    PetscInt i, j, ii, ilabel;

    // allocate memory
    ierr = PetscMalloc((size_t)dof2*sizeof(DMStagOutputBuffer),&buff_e);CHKERRQ(ierr);
    ierr = PetscMemzero(buff_e,(size_t)dof2*sizeof(DMStagOutputBuffer));CHKERRQ(ierr);

    for (ii = 0; ii<dof2; ii++) {
      // alocate memory for data
      ierr = PetscMalloc((size_t)(Nx*Nz)*sizeof(PetscScalar),&buff_e[ii].data);CHKERRQ(ierr);
      ierr = PetscMemzero(buff_e[ii].data,(size_t)(Nx*Nz)*sizeof(PetscScalar));CHKERRQ(ierr);

      // size
      buff_e[ii].size = (size_t)(Nx*Nz)*sizeof(PetscScalar);

      // label
      for (ilabel = 0; ilabel<ndof; ilabel++) {
        if ((labels[ilabel].c == ii) && (labels[ilabel].type == OUT_ELEMENT)) {
          if (labels[ilabel].filled) {
            buff_e[ii].name = labels[ilabel].name;
          } else {
            char dname[OUTPUT_NAME_LENGTH];
            sprintf(dname,"Element variable %d",ilabel); 
            ierr = PetscStrallocpy(dname,&buff_e[ii].name);CHKERRQ(ierr);
          }
        }
      }

      // loop domain and calculate data
      for (j = sz; j<sz+nz; j++) {
        for (i = sx; i<sx+nx; i++) {
          PetscScalar   xx;
          DMStagStencil point;

          point.i = i; point.j = j; point.loc = DMSTAG_ELEMENT;  point.c = ii;
          ierr = DMStagVecGetValuesStencil(dm,xlocal,1,&point,&xx); CHKERRQ(ierr);
          buff_e[ii].data[i+j*nz] = xx;
        }
      }
    }
  }

  // prepare coordinates - corners only
  PetscScalar *cx, *cz;
  PetscScalar xprev, xnext, zprev, znext;
  PetscInt    i, j, iprev, inext, icenter;

  // alocate memory for data
  ierr = PetscMalloc((size_t)(Nx+1)*sizeof(PetscScalar),&cx);CHKERRQ(ierr);
  ierr = PetscMemzero(cx,(size_t)(Nx+1)*sizeof(PetscScalar));CHKERRQ(ierr);

  ierr = PetscMalloc((size_t)(Nz+1)*sizeof(PetscScalar),&cz);CHKERRQ(ierr);
  ierr = PetscMemzero(cz,(size_t)(Nz+1)*sizeof(PetscScalar));CHKERRQ(ierr);

  if (dof2) {ierr = DMStagGet1dCoordinateLocationSlot(dm,DMSTAG_ELEMENT,&icenter);CHKERRQ(ierr);} 
  if (dof0 || dof1) { 
    ierr = DMStagGet1dCoordinateLocationSlot(dm,DMSTAG_LEFT,&iprev);CHKERRQ(ierr);
    ierr = DMStagGet1dCoordinateLocationSlot(dm,DMSTAG_RIGHT,&inext);CHKERRQ(ierr);
  } 

  for (i = sx; i<sx+nx; i++) {
    if ((!dof0) && (!dof1) && (dof2)) { // only element coord
      xprev   = (coordx[i][icenter]+coordx[i-1][icenter])*0.5; 
      xnext   = (coordx[i][icenter]+coordx[i+1][icenter])*0.5;
    } else {
      xprev   = coordx[i][iprev]; 
      xnext   = coordx[i][inext];
    }
    cx[i] = xprev; cx[i+1] = xnext;
  }

  for (j = sz; j<sz+nz; j++) {
    if ((!dof0) && (!dof1) && (dof2)) { // only element coord
      zprev   = (coordz[j][icenter]+coordz[j-1][icenter])*0.5; 
      znext   = (coordz[j][icenter]+coordz[j+1][icenter])*0.5;
    } else {
      zprev   = coordz[j][iprev]; 
      znext   = coordz[j][inext];
    }
    cz[j] = zprev; cz[j+1] = znext;
  }

  // print data
  FILE     *fp;
  PetscInt  offset;
  offset = 0;

  fp = fopen(fname,"w");
  if(fp == NULL) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Cannot open file %s", fname);

  fprintf(fp,"<?xml version=\"1.0\"?>\n");
  fprintf(fp,"<VTKFile type=\"RectilinearGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n");

  fprintf(fp,"\t<RectilinearGrid WholeExtent=\"%d %d %d %d %d %d\">\n",0,Nx,0,Nz,0,0);
  fprintf(fp,"\t\t<Piece Extent=\"%d %d %d %d %d %d\">\n",0,Nx,0,Nz,0,0);

  fprintf(fp,"\t\t\t<Coordinates>\n");
  fprintf(fp,"\t\t\t\t<DataArray type=\"Float64\" Name=\"Xcoord\"  format=\"appended\"  offset=\"%d\" />\n",offset);
  offset += (size_t)1*sizeof(PetscInt)+(size_t)(Nx+1)*sizeof(PetscScalar);

  fprintf(fp,"\t\t\t\t<DataArray type=\"Float64\" Name=\"Ycoord\"  format=\"appended\"  offset=\"%d\" />\n",offset);
  offset += (size_t)1*sizeof(PetscInt)+(size_t)(Nz+1)*sizeof(PetscScalar);

  fprintf(fp,"\t\t\t\t<DataArray type=\"Float64\" Name=\"Zcoord\"  format=\"appended\"  offset=\"%d\" />\n",offset);
  offset += (size_t)1*sizeof(PetscInt)+(size_t)(1)*sizeof(PetscScalar);

  fprintf(fp,"\t\t\t</Coordinates>\n");

  fprintf(fp,"\t\t\t<PointData>\n");
  fprintf(fp,"\t\t\t</PointData>\n");

  fprintf(fp,"\t\t\t<CellData>\n");

  // edge (vectors)
  PetscInt ii;
  if (dof1) {
    for (ii = 0; ii<dof1; ii++) {
      fprintf(fp,"\t\t\t\t<DataArray type=\"Float64\" Name=\"%s\" NumberOfComponents=\"3\" format=\"appended\" offset=\"%d\" />\n",buff_f[ii].name,offset);
      offset += (size_t)1*sizeof(PetscInt)+(size_t)(3*Nx*Nz)*sizeof(PetscScalar);
    }
  }
  
  // vertex (scalars)
  if (dof0) {
    for (ii = 0; ii<dof0; ii++) {
      fprintf(fp,"\t\t\t\t<DataArray type=\"Float64\" Name=\"%s\" NumberOfComponents=\"1\" format=\"appended\" offset=\"%d\" />\n",buff_v[ii].name,offset);
      offset += (size_t)1*sizeof(PetscInt)+(size_t)(Nx*Nz)*sizeof(PetscScalar);
    }
  }
  
  // element (scalars)
  if (dof2) {
    for (ii = 0; ii<dof2; ii++) {
      fprintf(fp,"\t\t\t\t<DataArray type=\"Float64\" Name=\"%s\" NumberOfComponents=\"1\" format=\"appended\" offset=\"%d\" />\n",buff_e[ii].name,offset);
      offset += (size_t)1*sizeof(PetscInt)+(size_t)(Nx*Nz)*sizeof(PetscScalar);
    }
  }
  
  fprintf(fp,"\t\t\t</CellData>\n");

  fprintf(fp,"\t\t</Piece>\n");
  fprintf(fp,"\t</RectilinearGrid>\n");
  fprintf(fp,"\t<AppendedData encoding=\"raw\">\n");
  fprintf(fp,"_");

  PetscInt    nbytes;
  PetscScalar fval = 0.0;

  // coordinates
  nbytes = (size_t)(Nx+1)*sizeof(PetscScalar);
  fwrite(&nbytes,sizeof(PetscInt),1,fp);
  fwrite(cx,sizeof(PetscScalar),(size_t)Nx+1,fp);

  nbytes = (size_t)(Nz+1)*sizeof(PetscScalar);
  fwrite(&nbytes,sizeof(PetscInt),1,fp);
  fwrite(cz,sizeof(PetscScalar),(size_t)Nz+1,fp);

  nbytes = (size_t)1*sizeof(PetscScalar);
  fwrite(&nbytes,sizeof(PetscInt),1,fp);
  fwrite(&fval,sizeof(PetscScalar),1,fp);

  // edge - vectors
  if (dof1) {
    for (ii = 0; ii<dof1; ii++) {
      nbytes = (size_t)(3*Nx*Nz)*sizeof(PetscScalar);
      fwrite(&nbytes,sizeof(PetscInt),1,fp);
      fwrite(buff_f[ii].data,sizeof(PetscScalar),(size_t)(3*Nx*Nz),fp);
    }
  }

  // scalars - vertex
  if (dof0) {
    for (ii = 0; ii<dof0; ii++) {
      nbytes = (size_t)(Nx*Nz)*sizeof(PetscScalar);
      fwrite(&nbytes,sizeof(PetscInt),1,fp);
      fwrite(buff_v[ii].data,sizeof(PetscScalar),(size_t)(Nx*Nz),fp);
    }
  }

  // scalars - element
  if (dof2) {
    for (ii = 0; ii<dof2; ii++) {
      nbytes = (size_t)(Nx*Nz)*sizeof(PetscScalar);
      fwrite(&nbytes,sizeof(PetscInt),1,fp);
      fwrite(buff_e[ii].data,sizeof(PetscScalar),(size_t)(Nx*Nz),fp);
    }
  }
  fprintf(fp,"\n\t</AppendedData>\n");
  fprintf(fp,"</VTKFile>\n");

  fclose(fp);

  // restore
  ierr = DMStagRestore1dCoordinateArraysDOFRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&xlocal); CHKERRQ(ierr);

  // clear memory
  ierr = PetscFree(cx);CHKERRQ(ierr);
  ierr = PetscFree(cz);CHKERRQ(ierr);

  if (dof0) {
    for (ii = 0; ii<dof0; ii++) { ierr = PetscFree(buff_v[ii].data);CHKERRQ(ierr); }
    ierr = PetscFree(buff_v);CHKERRQ(ierr);
  }

  if (dof1) {
    for (ii = 0; ii<dof1; ii++) { ierr = PetscFree(buff_f[ii].data);CHKERRQ(ierr); }
    ierr = PetscFree(buff_f);CHKERRQ(ierr);
  }

  if (dof2) {
    for (ii = 0; ii<dof2; ii++) { ierr = PetscFree(buff_e[ii].data);CHKERRQ(ierr); }
    ierr = PetscFree(buff_e);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}