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

Use: user
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
_DMStagOutputCreateBuffer

Use: internal
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "_DMStagOutputCreateBuffer"
static PetscErrorCode _DMStagOutputCreateBuffer(DM dm, OutputVTKType type, DMStagOutputBuffer **bv, DMStagOutputBuffer **bf, DMStagOutputBuffer **be)
{
  PetscInt            ii, dof0, dof1, dof2, Nx, Nz, n;
  DMStagOutputBuffer *buff_v = NULL,*buff_f = NULL,*buff_e = NULL;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = DMStagGetDOF(dm,&dof0,&dof1,&dof2,NULL);CHKERRQ(ierr);
  ierr = DMStagGetGlobalSizes(dm,&Nx,&Nz,NULL);CHKERRQ(ierr);

  if (type == VTK_CENTER) n = Nx*Nz;
  if (type == VTK_CORNER) n = (Nx+1)*(Nz+1);

  // Prepare vertex data
  if (dof0) {
    ierr = PetscMalloc((size_t)dof0*sizeof(DMStagOutputBuffer),&buff_v);CHKERRQ(ierr);
    ierr = PetscMemzero(buff_v,(size_t)dof0*sizeof(DMStagOutputBuffer));CHKERRQ(ierr);

    for (ii = 0; ii<dof0; ii++) {
      ierr = PetscMalloc((size_t)n*sizeof(PetscScalar),&buff_v[ii].data);CHKERRQ(ierr);
      ierr = PetscMemzero(buff_v[ii].data,(size_t)n*sizeof(PetscScalar));CHKERRQ(ierr);

      buff_v[ii].size = (size_t)n*sizeof(PetscScalar);
    }
  }

    // Prepare face data - vectors
  if (dof1) {
    ierr = PetscMalloc((size_t)dof1*sizeof(DMStagOutputBuffer),&buff_f);CHKERRQ(ierr);
    ierr = PetscMemzero(buff_f,(size_t)dof1*sizeof(DMStagOutputBuffer));CHKERRQ(ierr);

    for (ii = 0; ii<dof1; ii++) {
      ierr = PetscMalloc((size_t)(3*n)*sizeof(PetscScalar),&buff_f[ii].data);CHKERRQ(ierr);
      ierr = PetscMemzero(buff_f[ii].data,(size_t)(3*n)*sizeof(PetscScalar));CHKERRQ(ierr);

      buff_f[ii].size = (size_t)(3*n)*sizeof(PetscScalar);
    }
  }

  // Prepare element data
  if (dof2) {
    ierr = PetscMalloc((size_t)dof2*sizeof(DMStagOutputBuffer),&buff_e);CHKERRQ(ierr);
    ierr = PetscMemzero(buff_e,(size_t)dof2*sizeof(DMStagOutputBuffer));CHKERRQ(ierr);

    for (ii = 0; ii<dof2; ii++) {
      ierr = PetscMalloc((size_t)n*sizeof(PetscScalar),&buff_e[ii].data);CHKERRQ(ierr);
      ierr = PetscMemzero(buff_e[ii].data,(size_t)n*sizeof(PetscScalar));CHKERRQ(ierr);

      buff_e[ii].size = (size_t)n*sizeof(PetscScalar);
    }
  }

  *bv = buff_v;
  *bf = buff_f;
  *be = buff_e;

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
_DMStagOutputAddLabelsBuffer

Use: internal
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "_DMStagOutputAddLabelsBuffer"
static PetscErrorCode _DMStagOutputAddLabelsBuffer(DM dm, DMStagOutputLabel *labels, DMStagOutputBuffer *buff_v, DMStagOutputBuffer *buff_f, DMStagOutputBuffer *buff_e)
{
  PetscInt       ii, ilabel, dof0, dof1, dof2, ndof;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = DMStagGetDOF(dm,&dof0,&dof1,&dof2,NULL);CHKERRQ(ierr);
  ndof = dof0+dof1+dof2;

  // Vertex
  if (dof0) {
    for (ii = 0; ii<dof0; ii++) {
      for (ilabel = 0; ilabel<ndof; ilabel++) {
        if ((labels[ilabel].c == ii) && (labels[ilabel].type == OUT_VERTEX)) {
          if (labels[ilabel].filled) {
            buff_v[ii].name = labels[ilabel].name;
          } else {
            char dname[OUTPUT_NAME_LENGTH];
            sprintf(dname,"Vertex scalar %d",ilabel); 
            ierr = PetscStrallocpy(dname,&buff_v[ii].name);CHKERRQ(ierr);
          }
        }
      }
    }
  }

    // Face data - vectors
  if (dof1) {
    for (ii = 0; ii<dof1; ii++) {
      for (ilabel = 0; ilabel<ndof; ilabel++) {
        if ((labels[ilabel].c == ii) && (labels[ilabel].type == OUT_FACE)) {
          if (labels[ilabel].filled) {
              buff_f[ii].name = labels[ilabel].name;
          } else {
            char dname[OUTPUT_NAME_LENGTH];
            sprintf(dname,"Face vector %d",ilabel); 
            ierr = PetscStrallocpy(dname,&buff_f[ii].name);CHKERRQ(ierr);
          }
        }
      }
    }
  }

  // Element
  if (dof2) {
    for (ii = 0; ii<dof2; ii++) {
      for (ilabel = 0; ilabel<ndof; ilabel++) {
        if ((labels[ilabel].c == ii) && (labels[ilabel].type == OUT_ELEMENT)) {
          if (labels[ilabel].filled) {
            buff_e[ii].name = labels[ilabel].name;
          } else {
            char dname[OUTPUT_NAME_LENGTH];
            sprintf(dname,"Element scalar %d",ilabel); 
            ierr = PetscStrallocpy(dname,&buff_e[ii].name);CHKERRQ(ierr);
          }
        }
      }
    }
  }

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
_DMStagOutputAddDataBuffer

Use: internal
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "_DMStagOutputAddDataBuffer"
static PetscErrorCode _DMStagOutputAddDataBuffer(DM dm, Vec xlocal, OutputVTKType type, DMStagOutputBuffer *buff_v, DMStagOutputBuffer *buff_f, DMStagOutputBuffer *buff_e)
{
  PetscInt       ii, dof0, dof1, dof2;
  PetscInt       i,j,nx,nz,sx,sz,Nx,Nz; 
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = DMStagGetDOF(dm,&dof0,&dof1,&dof2,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = DMStagGetGlobalSizes(dm,&Nx,&Nz,NULL);CHKERRQ(ierr);

  if (type == VTK_CENTER) {
    // Prepare vertex data
    if (dof0) {
      for (ii = 0; ii<dof0; ii++) {
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
            buff_v[ii].data[i+j*nx] = fval;
          }
        }
      }
    }

    // Prepare face data - vectors
    if (dof1) {
      for (ii = 0; ii<dof1; ii++) {
        for (j = sz; j<sz+nz; j++) {
          for (i = sx; i<sx+nx; i++) {
            PetscScalar   fval = 0.0, xx[4];
            DMStagStencil point[4];

            point[0].i = i; point[0].j = j; point[0].loc = DMSTAG_LEFT;  point[0].c = ii;
            point[1].i = i; point[1].j = j; point[1].loc = DMSTAG_RIGHT; point[1].c = ii;
            point[2].i = i; point[2].j = j; point[2].loc = DMSTAG_DOWN;  point[2].c = ii;
            point[3].i = i; point[3].j = j; point[3].loc = DMSTAG_UP;    point[3].c = ii;

            ierr = DMStagVecGetValuesStencil(dm,xlocal,4,point,xx); CHKERRQ(ierr);

            buff_f[ii].data[3*(i+j*nx)+0] = (xx[0]+xx[1])*0.5;
            buff_f[ii].data[3*(i+j*nx)+1] = (xx[2]+xx[3])*0.5;
            buff_f[ii].data[3*(i+j*nx)+2] = fval;
          }
        }
      }
    }

    // Prepare element data
    if (dof2) {
      for (ii = 0; ii<dof2; ii++) {
        for (j = sz; j<sz+nz; j++) {
          for (i = sx; i<sx+nx; i++) {
            PetscScalar   xx;
            DMStagStencil point;

            point.i = i; point.j = j; point.loc = DMSTAG_ELEMENT;  point.c = ii;
            ierr = DMStagVecGetValuesStencil(dm,xlocal,1,&point,&xx); CHKERRQ(ierr);
            buff_e[ii].data[i+j*nx] = xx;
          }
        }
      }
    }
  }

  // Warning: extrapolation!
  if (type == VTK_CORNER) {
    if (dof0) { // vertex data
      for (ii = 0; ii<dof0; ii++) {
        for (j = sz; j<sz+nz+1; j++) {
          for (i = sx; i<sx+nx+1; i++) {
            PetscScalar   xx;
            DMStagStencil point;

            point.i = i; point.j = j; point.loc = DMSTAG_DOWN_LEFT;  point.c = ii;
            ierr = DMStagVecGetValuesStencil(dm,xlocal,1,&point,&xx); CHKERRQ(ierr);
            buff_v[ii].data[i+j*(nx+1)] = xx;
          }
        }
      }
    }

    if (dof1) { // face data - vectors
      for (ii = 0; ii<dof1; ii++) {
        for (j = sz; j<sz+nz; j++) {
          for (i = sx; i<sx+nx; i++) {
            PetscScalar   fval = 0.0, xx[4];
            DMStagStencil point[4];

            if ((i > 0) && (j > 0)) {
              point[0].i = i  ; point[0].j = j-1; point[0].loc = DMSTAG_LEFT; point[0].c = ii;
              point[1].i = i  ; point[1].j = j  ; point[1].loc = DMSTAG_LEFT; point[1].c = ii;
              point[2].i = i-1; point[2].j = j  ; point[2].loc = DMSTAG_DOWN; point[2].c = ii;
              point[3].i = i  ; point[3].j = j  ; point[3].loc = DMSTAG_DOWN; point[3].c = ii;
              ierr = DMStagVecGetValuesStencil(dm,xlocal,4,point,xx); CHKERRQ(ierr);

              buff_f[ii].data[3*(i+j*(nx+1))+0] = (xx[0]+xx[1])*0.5;
              buff_f[ii].data[3*(i+j*(nx+1))+1] = (xx[2]+xx[3])*0.5;
              buff_f[ii].data[3*(i+j*(nx+1))+2] = fval;
            } 

            if ((i == 0) && (j > 0)) {
              point[0].i = i  ; point[0].j = j-1; point[0].loc = DMSTAG_LEFT; point[0].c = ii;
              point[1].i = i  ; point[1].j = j  ; point[1].loc = DMSTAG_LEFT; point[1].c = ii;
              point[2].i = i  ; point[2].j = j  ; point[2].loc = DMSTAG_DOWN; point[2].c = ii;
              point[3].i = i+1; point[3].j = j  ; point[3].loc = DMSTAG_DOWN; point[3].c = ii;
              ierr = DMStagVecGetValuesStencil(dm,xlocal,4,point,xx); CHKERRQ(ierr);

              buff_f[ii].data[3*(i+j*(nx+1))+0] = (xx[0]+xx[1])*0.5;
              buff_f[ii].data[3*(i+j*(nx+1))+1] = 2*xx[2]-xx[3]; // linear extrapolation
              buff_f[ii].data[3*(i+j*(nx+1))+2] = fval;
            }

            if ((i == Nx-1) && (j > 0)) {
              point[0].i = i  ; point[0].j = j-1; point[0].loc = DMSTAG_LEFT; point[0].c = ii;
              point[1].i = i  ; point[1].j = j  ; point[1].loc = DMSTAG_LEFT; point[1].c = ii;
              point[2].i = i  ; point[2].j = j  ; point[2].loc = DMSTAG_DOWN; point[2].c = ii;
              point[3].i = i-1; point[3].j = j  ; point[3].loc = DMSTAG_DOWN; point[3].c = ii;
              ierr = DMStagVecGetValuesStencil(dm,xlocal,4,point,xx); CHKERRQ(ierr);

              buff_f[ii].data[3*(i+1+j*(nx+1))+0] = (xx[0]+xx[1])*0.5;
              buff_f[ii].data[3*(i+1+j*(nx+1))+1] = 2*xx[2]-xx[3]; // linear extrapolation
              buff_f[ii].data[3*(i+1+j*(nx+1))+2] = fval;
            }

            if ((i > 0) && (j == 0)) {
              point[0].i = i  ; point[0].j = j  ; point[0].loc = DMSTAG_LEFT; point[0].c = ii;
              point[1].i = i  ; point[1].j = j+1; point[1].loc = DMSTAG_LEFT; point[1].c = ii;
              point[2].i = i-1; point[2].j = j  ; point[2].loc = DMSTAG_DOWN; point[2].c = ii;
              point[3].i = i  ; point[3].j = j  ; point[3].loc = DMSTAG_DOWN; point[3].c = ii;
              ierr = DMStagVecGetValuesStencil(dm,xlocal,4,point,xx); CHKERRQ(ierr);

              buff_f[ii].data[3*(i+j*(nx+1))+0] = 2*xx[0]-xx[1]; // linear extrapolation
              buff_f[ii].data[3*(i+j*(nx+1))+1] = (xx[2]+xx[3])*0.5; 
              buff_f[ii].data[3*(i+j*(nx+1))+2] = fval;
            }

            if ((i > 0) && (j == Nz-1)) {
              point[0].i = i  ; point[0].j = j  ; point[0].loc = DMSTAG_LEFT; point[0].c = ii;
              point[1].i = i  ; point[1].j = j-1; point[1].loc = DMSTAG_LEFT; point[1].c = ii;
              point[2].i = i-1; point[2].j = j  ; point[2].loc = DMSTAG_DOWN; point[2].c = ii;
              point[3].i = i  ; point[3].j = j  ; point[3].loc = DMSTAG_DOWN; point[3].c = ii;
              ierr = DMStagVecGetValuesStencil(dm,xlocal,4,point,xx); CHKERRQ(ierr);

              buff_f[ii].data[3*(i+(j+1)*(nx+1))+0] = 2*xx[0]-xx[1]; // linear extrapolation
              buff_f[ii].data[3*(i+(j+1)*(nx+1))+1] = (xx[2]+xx[3])*0.5; 
              buff_f[ii].data[3*(i+(j+1)*(nx+1))+2] = fval;
            }

            if ((i == 0) && (j == 0)) {
              point[0].i = i  ; point[0].j = j  ; point[0].loc = DMSTAG_LEFT; point[0].c = ii;
              point[1].i = i  ; point[1].j = j+1; point[1].loc = DMSTAG_LEFT; point[1].c = ii;
              point[2].i = i  ; point[2].j = j  ; point[2].loc = DMSTAG_DOWN; point[2].c = ii;
              point[3].i = i+1; point[3].j = j  ; point[3].loc = DMSTAG_DOWN; point[3].c = ii;
              ierr = DMStagVecGetValuesStencil(dm,xlocal,4,point,xx); CHKERRQ(ierr);

              buff_f[ii].data[3*(i+j*(nx+1))+0] = 2*xx[0]-xx[1]; // linear extrapolation
              buff_f[ii].data[3*(i+j*(nx+1))+1] = 2*xx[2]-xx[3]; // linear extrapolation
              buff_f[ii].data[3*(i+j*(nx+1))+2] = fval;
            }

            if ((i == 0) && (j == Nz-1)) {
              point[0].i = i  ; point[0].j = j  ; point[0].loc = DMSTAG_LEFT; point[0].c = ii;
              point[1].i = i  ; point[1].j = j-1; point[1].loc = DMSTAG_LEFT; point[1].c = ii;
              point[2].i = i  ; point[2].j = j  ; point[2].loc = DMSTAG_DOWN; point[2].c = ii;
              point[3].i = i+1; point[3].j = j  ; point[3].loc = DMSTAG_DOWN; point[3].c = ii;
              ierr = DMStagVecGetValuesStencil(dm,xlocal,4,point,xx); CHKERRQ(ierr);

              buff_f[ii].data[3*(i+(j+1)*(nx+1))+0] = 2*xx[0]-xx[1]; // linear extrapolation
              buff_f[ii].data[3*(i+(j+1)*(nx+1))+1] = 2*xx[2]-xx[3]; // linear extrapolation
              buff_f[ii].data[3*(i+(j+1)*(nx+1))+2] = fval;
            }
            
            if ((i == Nx-1) && (j == 0)) {
              point[0].i = i  ; point[0].j = j  ; point[0].loc = DMSTAG_LEFT; point[0].c = ii;
              point[1].i = i  ; point[1].j = j+1; point[1].loc = DMSTAG_LEFT; point[1].c = ii;
              point[2].i = i  ; point[2].j = j  ; point[2].loc = DMSTAG_DOWN; point[2].c = ii;
              point[3].i = i-1; point[3].j = j  ; point[3].loc = DMSTAG_DOWN; point[3].c = ii;
              ierr = DMStagVecGetValuesStencil(dm,xlocal,4,point,xx); CHKERRQ(ierr);

              buff_f[ii].data[3*(i+1+j*(nx+1))+0] = 2*xx[0]-xx[1]; // linear extrapolation
              buff_f[ii].data[3*(i+1+j*(nx+1))+1] = 2*xx[2]-xx[3]; // linear extrapolation
              buff_f[ii].data[3*(i+1+j*(nx+1))+2] = fval;
            }

            if ((i == Nx-1) && (j == Nz-1)) {
              point[0].i = i  ; point[0].j = j  ; point[0].loc = DMSTAG_LEFT; point[0].c = ii;
              point[1].i = i  ; point[1].j = j-1; point[1].loc = DMSTAG_LEFT; point[1].c = ii;
              point[2].i = i  ; point[2].j = j  ; point[2].loc = DMSTAG_DOWN; point[2].c = ii;
              point[3].i = i-1; point[3].j = j  ; point[3].loc = DMSTAG_DOWN; point[3].c = ii;
              ierr = DMStagVecGetValuesStencil(dm,xlocal,4,point,xx); CHKERRQ(ierr);

              buff_f[ii].data[3*(i+1+(j+1)*(nx+1))+0] = 2*xx[0]-xx[1]; // linear extrapolation
              buff_f[ii].data[3*(i+1+(j+1)*(nx+1))+1] = 2*xx[2]-xx[3]; // linear extrapolation
              buff_f[ii].data[3*(i+1+(j+1)*(nx+1))+2] = fval;
            }
          }
        }
      }
    }

    if (dof2) { // element data
      for (ii = 0; ii<dof2; ii++) {
        for (j = sz; j<sz+nz; j++) {
          for (i = sx; i<sx+nx; i++) {
            PetscScalar   xx[4];
            DMStagStencil point[4];

            if ((i > 0) && (j > 0)) {
              point[0].i = i-1; point[0].j = j-1; point[0].loc = DMSTAG_ELEMENT; point[0].c = ii;
              point[1].i = i-1; point[1].j = j  ; point[1].loc = DMSTAG_ELEMENT; point[1].c = ii;
              point[2].i = i  ; point[2].j = j-1; point[2].loc = DMSTAG_ELEMENT; point[2].c = ii;
              point[3].i = i  ; point[3].j = j  ; point[3].loc = DMSTAG_ELEMENT; point[3].c = ii;
              ierr = DMStagVecGetValuesStencil(dm,xlocal,4,point,xx); CHKERRQ(ierr);

              buff_e[ii].data[i+j*(nx+1)] = (xx[0]+xx[1]+xx[2]+xx[3])*0.25;
            }

            if ((i == 0) && (j > 0)) {
              point[0].i = i  ; point[0].j = j-1; point[0].loc = DMSTAG_ELEMENT; point[0].c = ii;
              point[1].i = i  ; point[1].j = j  ; point[1].loc = DMSTAG_ELEMENT; point[1].c = ii;
              point[2].i = i+1; point[2].j = j-1; point[2].loc = DMSTAG_ELEMENT; point[2].c = ii;
              point[3].i = i+1; point[3].j = j  ; point[3].loc = DMSTAG_ELEMENT; point[3].c = ii;
              ierr = DMStagVecGetValuesStencil(dm,xlocal,4,point,xx); CHKERRQ(ierr);

              buff_e[ii].data[i+j*(nx+1)] = 2.0*(xx[0]+xx[1])*0.5-(xx[2]+xx[3])*0.5;
            }

            if ((i == Nx-1) && (j > 0)) {
              point[0].i = i  ; point[0].j = j-1; point[0].loc = DMSTAG_ELEMENT; point[0].c = ii;
              point[1].i = i  ; point[1].j = j  ; point[1].loc = DMSTAG_ELEMENT; point[1].c = ii;
              point[2].i = i-1; point[2].j = j-1; point[2].loc = DMSTAG_ELEMENT; point[2].c = ii;
              point[3].i = i-1; point[3].j = j  ; point[3].loc = DMSTAG_ELEMENT; point[3].c = ii;
              ierr = DMStagVecGetValuesStencil(dm,xlocal,4,point,xx); CHKERRQ(ierr);

              buff_e[ii].data[i+1+j*(nx+1)] = 2.0*(xx[0]+xx[1])*0.5-(xx[2]+xx[3])*0.5;
            }

            if ((i > 0) && (j == 0)) {
              point[0].i = i-1; point[0].j = j  ; point[0].loc = DMSTAG_ELEMENT; point[0].c = ii;
              point[1].i = i  ; point[1].j = j  ; point[1].loc = DMSTAG_ELEMENT; point[1].c = ii;
              point[2].i = i-1; point[2].j = j+1; point[2].loc = DMSTAG_ELEMENT; point[2].c = ii;
              point[3].i = i  ; point[3].j = j+1; point[3].loc = DMSTAG_ELEMENT; point[3].c = ii;
              ierr = DMStagVecGetValuesStencil(dm,xlocal,4,point,xx); CHKERRQ(ierr);

              buff_e[ii].data[i+j*(nx+1)] = 2.0*(xx[0]+xx[1])*0.5-(xx[2]+xx[3])*0.5;
            }

            if ((i > 0) && (j == Nz-1)) {
              point[0].i = i-1; point[0].j = j  ; point[0].loc = DMSTAG_ELEMENT; point[0].c = ii;
              point[1].i = i  ; point[1].j = j  ; point[1].loc = DMSTAG_ELEMENT; point[1].c = ii;
              point[2].i = i-1; point[2].j = j-1; point[2].loc = DMSTAG_ELEMENT; point[2].c = ii;
              point[3].i = i  ; point[3].j = j-1; point[3].loc = DMSTAG_ELEMENT; point[3].c = ii;
              ierr = DMStagVecGetValuesStencil(dm,xlocal,4,point,xx); CHKERRQ(ierr);

              buff_e[ii].data[i+(j+1)*(nx+1)] = 2.0*(xx[0]+xx[1])*0.5-(xx[2]+xx[3])*0.5;
            }

            if ((i == 0) && (j == 0)) {
              point[0].i = i  ; point[0].j = j  ; point[0].loc = DMSTAG_ELEMENT; point[0].c = ii;
              point[1].i = i+1; point[1].j = j+1; point[1].loc = DMSTAG_ELEMENT; point[1].c = ii;
              ierr = DMStagVecGetValuesStencil(dm,xlocal,2,point,xx); CHKERRQ(ierr);

              buff_e[ii].data[i+j*(nx+1)] = 2.0*xx[0]-xx[1];
            }

            if ((i == Nx-1) && (j == 0)) {
              point[0].i = i  ; point[0].j = j  ; point[0].loc = DMSTAG_ELEMENT; point[0].c = ii;
              point[1].i = i-1; point[1].j = j+1; point[1].loc = DMSTAG_ELEMENT; point[1].c = ii;
              ierr = DMStagVecGetValuesStencil(dm,xlocal,2,point,xx); CHKERRQ(ierr);

              buff_e[ii].data[i+1+j*(nx+1)] = 2.0*xx[0]-xx[1];
            }

            if ((i == 0) && (j == Nz-1)) {
              point[0].i = i  ; point[0].j = j  ; point[0].loc = DMSTAG_ELEMENT; point[0].c = ii;
              point[1].i = i+1; point[1].j = j-1; point[1].loc = DMSTAG_ELEMENT; point[1].c = ii;
              ierr = DMStagVecGetValuesStencil(dm,xlocal,2,point,xx); CHKERRQ(ierr);

              buff_e[ii].data[i+(j+1)*(nx+1)] = 2.0*xx[0]-xx[1];
            }

            if ((i == Nx-1) && (j == Nz-1)) {
              point[0].i = i  ; point[0].j = j  ; point[0].loc = DMSTAG_ELEMENT; point[0].c = ii;
              point[1].i = i-1; point[1].j = j-1; point[1].loc = DMSTAG_ELEMENT; point[1].c = ii;
              ierr = DMStagVecGetValuesStencil(dm,xlocal,2,point,xx); CHKERRQ(ierr);

              buff_e[ii].data[i+1+(j+1)*(nx+1)] = 2.0*xx[0]-xx[1];
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
_DMStagOutputGetCoordinates2D

Use: internal
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "_DMStagOutputGetCoordinates2D"
static PetscErrorCode _DMStagOutputGetCoordinates2D(DM dm, PetscScalar **_cx, PetscScalar **_cz)
{
  PetscScalar **coordx,**coordz;
  PetscScalar *cx, *cz;
  PetscScalar xprev, xnext, zprev, znext;
  PetscInt    i, j, iprev, inext, icenter, Nx, Nz;
  PetscInt    sx, sz, nx, nz, dof0, dof1, dof2;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = DMStagGetDOF(dm,&dof0,&dof1,&dof2,NULL);CHKERRQ(ierr);
  ierr = DMStagGetGlobalSizes(dm,&Nx,&Nz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  // Get dm coordinates array
  ierr = DMStagGet1dCoordinateArraysDOFRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);

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

  // restore
  ierr = DMStagRestore1dCoordinateArraysDOFRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);

  *_cx = cx;
  *_cz = cz;

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
DMStagOutputVTKBinary - writes to a VTK file the DMStag and associated vector 

Input parameter:
dm - DMStag object
x - associated vector
labels - user-provided names (otherwise default names are given)
type - VTK_CENTER/VTK_CORNER
fname - file name

Notes:
It is assumed edge values are vector components (not always true).

Use: user
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "DMStagOutputVTKBinary"
PetscErrorCode DMStagOutputVTKBinary(DM dm, Vec x, DMStagOutputLabel *labels, OutputVTKType type, const char fname[])
{
  Vec            xglobal, xlocal;
  PetscInt       dim, sz, sz1;
  PetscMPIInt    size;
  PetscBool      isstag, flg;
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

  {
    FILE               *fp;
    PetscInt            offset = 0;
    DMStagOutputBuffer *buff_v, *buff_f, *buff_e;
    PetscScalar        *cx, *cz;
    PetscInt            ii, dof0, dof1, dof2, Nx, Nz, n; 
    PetscInt            nbytes;
    PetscScalar         fval = 0.0;

    // Prepare data for output - create buffers for vertex, face, element
    ierr = DMStagGetDOF(dm,&dof0,&dof1,&dof2,NULL);CHKERRQ(ierr);
    ierr = DMStagGetGlobalSizes(dm,&Nx,&Nz,NULL);CHKERRQ(ierr);

    // Map global vectors to local domain
    ierr = DMGetLocalVector(dm, &xlocal); CHKERRQ(ierr);
    ierr = DMGlobalToLocal (dm, x, INSERT_VALUES, xlocal); CHKERRQ(ierr);

    if (type == VTK_CENTER) n = Nx*Nz;
    if (type == VTK_CORNER) n = (Nx+1)*(Nz+1);

    // Create data buffers
    ierr = _DMStagOutputCreateBuffer(dm,type,&buff_v,&buff_f,&buff_e);CHKERRQ(ierr);
    ierr = _DMStagOutputAddLabelsBuffer(dm,labels,buff_v,buff_f,buff_e);CHKERRQ(ierr);
    ierr = _DMStagOutputAddDataBuffer(dm,xlocal,type,buff_v,buff_f,buff_e);CHKERRQ(ierr);

    // Get coordinates buffers (only corners)
    ierr = _DMStagOutputGetCoordinates2D(dm,&cx,&cz);CHKERRQ(ierr);

    // print data
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

    if (type == VTK_CORNER) {
      fprintf(fp,"\t\t\t<CellData>\n");
      fprintf(fp,"\t\t\t</CellData>\n");
      fprintf(fp,"\t\t\t<PointData>\n");
    }

    if (type == VTK_CENTER) {
      fprintf(fp,"\t\t\t<PointData>\n");
      fprintf(fp,"\t\t\t</PointData>\n");
      fprintf(fp,"\t\t\t<CellData>\n");
    }
    
    // edge (vectors)
    if (dof1) {
      for (ii = 0; ii<dof1; ii++) {
        fprintf(fp,"\t\t\t\t<DataArray type=\"Float64\" Name=\"%s\" NumberOfComponents=\"3\" format=\"appended\" offset=\"%d\" />\n",buff_f[ii].name,offset);
        offset += (size_t)1*sizeof(PetscInt)+(size_t)(3*n)*sizeof(PetscScalar);
      }
    }
    
    // vertex (scalars)
    if (dof0) {
      for (ii = 0; ii<dof0; ii++) {
        fprintf(fp,"\t\t\t\t<DataArray type=\"Float64\" Name=\"%s\" NumberOfComponents=\"1\" format=\"appended\" offset=\"%d\" />\n",buff_v[ii].name,offset);
        offset += (size_t)1*sizeof(PetscInt)+(size_t)n*sizeof(PetscScalar);
      }
    }
    
    // element (scalars)
    if (dof2) {
      for (ii = 0; ii<dof2; ii++) {
        fprintf(fp,"\t\t\t\t<DataArray type=\"Float64\" Name=\"%s\" NumberOfComponents=\"1\" format=\"appended\" offset=\"%d\" />\n",buff_e[ii].name,offset);
        offset += (size_t)1*sizeof(PetscInt)+(size_t)n*sizeof(PetscScalar);
      }
    }
    
    if (type == VTK_CORNER) fprintf(fp,"\t\t\t</PointData>\n");
    if (type == VTK_CENTER) fprintf(fp,"\t\t\t</CellData>\n");

    fprintf(fp,"\t\t</Piece>\n");
    fprintf(fp,"\t</RectilinearGrid>\n");
    fprintf(fp,"\t<AppendedData encoding=\"raw\">\n");
    fprintf(fp,"_");

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
        nbytes = (size_t)(3*n)*sizeof(PetscScalar);
        fwrite(&nbytes,sizeof(PetscInt),1,fp);
        fwrite(buff_f[ii].data,sizeof(PetscScalar),(size_t)(3*n),fp);
      }
    }

    // scalars - vertex
    if (dof0) {
      for (ii = 0; ii<dof0; ii++) {
        nbytes = (size_t)n*sizeof(PetscScalar);
        fwrite(&nbytes,sizeof(PetscInt),1,fp);
        fwrite(buff_v[ii].data,sizeof(PetscScalar),(size_t)n,fp);
      }
    }

    // scalars - element
    if (dof2) {
      for (ii = 0; ii<dof2; ii++) {
        nbytes = (size_t)n*sizeof(PetscScalar);
        fwrite(&nbytes,sizeof(PetscInt),1,fp);
        fwrite(buff_e[ii].data,sizeof(PetscScalar),(size_t)n,fp);
      }
    }
    fprintf(fp,"\n\t</AppendedData>\n");
    fprintf(fp,"</VTKFile>\n");

    fclose(fp);

    // restore
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
  }

  PetscFunctionReturn(0);
}


void pythonemit(FILE *fp,const char str[])
{
  fprintf(fp,"%s",str);
}

void pythonemitvec(FILE *fp,const char name[])
{
  char pline[PETSC_MAX_PATH_LEN];
  pythonemit(fp,"    objecttype = io.readObjectType(fp)\n");
  pythonemit(fp,"    v = io.readVec(fp)\n");
  PetscSNPrintf(pline,PETSC_MAX_PATH_LEN-1,"    data['%s'] = v\n",name);
  pythonemit(fp,pline);
}

/*
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
*/
#undef __FUNCT__
#define __FUNCT__ "DMStagViewBinaryPython_SEQ"
PetscErrorCode DMStagViewBinaryPython_SEQ(DM dm,Vec X,const char prefix[])
{
  PetscErrorCode ierr;
  PetscViewer v;
  PetscInt M,N,P,dim;
  FILE *fp = NULL;
  char fname[PETSC_MAX_PATH_LEN],string[PETSC_MAX_PATH_LEN];
  MPI_Comm comm;
  PetscMPIInt size;
  PetscBool view_coords = PETSC_TRUE; /* ultimately this would be an input arg */
  
  comm = PetscObjectComm((PetscObject)dm);
  ierr = MPI_Comm_size(comm,&size); CHKERRQ(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Sequential only");

  /* check for instances of "." in the file name so that the file can be imported */
  {
    size_t k,len;
    ierr = PetscStrlen(prefix,&len);CHKERRQ(ierr);
    for (k=0; k<len; k++) if (prefix[k] == '.') PetscPrintf(comm,"[DMStagViewBinaryPython_SEQ] Warning: prefix %s contains the symbol '.'. Hence you will not be able to import the emiited python script. Consider change the prefix\n",prefix);
  }
  
  ierr = PetscSNPrintf(fname,PETSC_MAX_PATH_LEN-1,"%s.pbin",prefix);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(comm,fname,FILE_MODE_WRITE,&v);CHKERRQ(ierr);
  
  ierr = PetscSNPrintf(string,PETSC_MAX_PATH_LEN-1,"%s.py",prefix);CHKERRQ(ierr);
  
  fp = fopen(string,"w");
  if (!fp) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Cannot open file %s",string);
  
  pythonemit(fp,"import PetscBinaryIO as pio\n");
  pythonemit(fp,"import numpy as np\n\n");

  pythonemit(fp,"def _PETScBinaryLoad():\n");
  pythonemit(fp,"  io = pio.PetscBinaryIO()\n");

  PetscSNPrintf(string,PETSC_MAX_PATH_LEN-1,"  filename = \"%s\"\n",fname);
  pythonemit(fp,string);
  pythonemit(fp,"  data = dict()\n");
  pythonemit(fp,"  with open(filename) as fp:\n");
  
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  ierr = DMStagGetGlobalSizes(dm,&M,&N,&P);CHKERRQ(ierr);
  
  ierr = PetscViewerBinaryWrite(v,(void*)&M,1,PETSC_INT,PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(v,(void*)&N,1,PETSC_INT,PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(v,(void*)&P,1,PETSC_INT,PETSC_FALSE);CHKERRQ(ierr);
  
  pythonemit(fp,"    v = io.readInteger(fp)\n"); pythonemit(fp,"    data['Nx'] = v\n");
  pythonemit(fp,"    v = io.readInteger(fp)\n"); pythonemit(fp,"    data['Ny'] = v\n");
  pythonemit(fp,"    v = io.readInteger(fp)\n"); pythonemit(fp,"    data['Nz'] = v\n");
  
  if (view_coords) {
    DM cdm,subDM;
    PetscBool isProduct;
    Vec coor;
    DM pda;
    Vec subX;
    PetscInt dof[4];
    
    ierr = DMGetCoordinateDM(dm,&cdm);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)cdm,DMPRODUCT,&isProduct);CHKERRQ(ierr);
    if (isProduct) {
      if (dim >= 1) {
        ierr = DMProductGetDM(cdm,0,&subDM);CHKERRQ(ierr);
        ierr = DMGetCoordinates(subDM,&coor);CHKERRQ(ierr);
        ierr = VecView(coor,v);CHKERRQ(ierr);
        pythonemitvec(fp,"x1d");
        
        ierr = DMStagGetDOF(subDM,&dof[0],&dof[1],&dof[2],&dof[3]);CHKERRQ(ierr);
        if (dof[0] != 0) {
          ierr = DMStagVecSplitToDMDA(subDM,coor,DMSTAG_LEFT,-dof[0],&pda,&subX);CHKERRQ(ierr);
          ierr = VecView(subX,v);CHKERRQ(ierr);
          pythonemitvec(fp,"x1d_vertex");
          ierr = VecDestroy(&subX);CHKERRQ(ierr);
          ierr = DMDestroy(&pda);CHKERRQ(ierr);
        }
        if (dof[1] != 0) {
          ierr = DMStagVecSplitToDMDA(subDM,coor,DMSTAG_ELEMENT,-dof[1],&pda,&subX);CHKERRQ(ierr);
          ierr = VecView(subX,v);CHKERRQ(ierr);
          pythonemitvec(fp,"x1d_cell");
          ierr = VecDestroy(&subX);CHKERRQ(ierr);
          ierr = DMDestroy(&pda);CHKERRQ(ierr);
        }
      }
      if (dim >= 2) {
        ierr = DMProductGetDM(cdm,1,&subDM);CHKERRQ(ierr);
        ierr = DMGetCoordinates(subDM,&coor);CHKERRQ(ierr);
        ierr = VecView(coor,v);CHKERRQ(ierr);
        pythonemitvec(fp,"y1d");
        
        ierr = DMStagGetDOF(subDM,&dof[0],&dof[1],&dof[2],&dof[3]);CHKERRQ(ierr);
        if (dof[0] != 0) {
          ierr = DMStagVecSplitToDMDA(subDM,coor,DMSTAG_LEFT,-dof[0],&pda,&subX);CHKERRQ(ierr);
          ierr = VecView(subX,v);CHKERRQ(ierr);
          pythonemitvec(fp,"y1d_vertex");
          ierr = VecDestroy(&subX);CHKERRQ(ierr);
          ierr = DMDestroy(&pda);CHKERRQ(ierr);
        }
        if (dof[1] != 0) {
          ierr = DMStagVecSplitToDMDA(subDM,coor,DMSTAG_ELEMENT,-dof[1],&pda,&subX);CHKERRQ(ierr);
          ierr = VecView(subX,v);CHKERRQ(ierr);
          pythonemitvec(fp,"y1d_cell");
          ierr = VecDestroy(&subX);CHKERRQ(ierr);
          ierr = DMDestroy(&pda);CHKERRQ(ierr);
        }
      }
      if (dim == 3) {
        ierr = DMProductGetDM(cdm,2,&subDM);CHKERRQ(ierr);
        ierr = DMGetCoordinates(subDM,&coor);CHKERRQ(ierr);
        ierr = VecView(coor,v);CHKERRQ(ierr);
        pythonemitvec(fp,"z1d");
        
        ierr = DMStagGetDOF(subDM,&dof[0],&dof[1],&dof[2],&dof[3]);CHKERRQ(ierr);
        if (dof[0] != 0) {
          ierr = DMStagVecSplitToDMDA(subDM,coor,DMSTAG_LEFT,-dof[0],&pda,&subX);CHKERRQ(ierr);
          ierr = VecView(subX,v);CHKERRQ(ierr);
          pythonemitvec(fp,"z1d_vertex");
          ierr = VecDestroy(&subX);CHKERRQ(ierr);
          ierr = DMDestroy(&pda);CHKERRQ(ierr);
        }
        if (dof[1] != 0) {
          ierr = DMStagVecSplitToDMDA(subDM,coor,DMSTAG_ELEMENT,-dof[1],&pda,&subX);CHKERRQ(ierr);
          ierr = VecView(subX,v);CHKERRQ(ierr);
          pythonemitvec(fp,"z1d_cell");
          ierr = VecDestroy(&subX);CHKERRQ(ierr);
          ierr = DMDestroy(&pda);CHKERRQ(ierr);
        }
        
      }
    } else SETERRQ(comm,PETSC_ERR_SUP,"Only supports coordinated defined via DMPRODUCT");
  }
  
  {
    DM pda;
    Vec subX;
    PetscInt dof[4];
    
    ierr = DMStagGetDOF(dm,&dof[0],&dof[1],&dof[2],&dof[3]);CHKERRQ(ierr);

    if (dim == 1) {
      if (dof[0] != 0) {
        ierr = DMStagVecSplitToDMDA(dm,X,DMSTAG_DOWN_LEFT,-dof[0],&pda,&subX);CHKERRQ(ierr);
        ierr = VecView(subX,v);CHKERRQ(ierr);
        pythonemitvec(fp,"X_vertex");
        ierr = VecDestroy(&subX);CHKERRQ(ierr);
        ierr = DMDestroy(&pda);CHKERRQ(ierr);
      }
      if (dof[1] != 0) {
        ierr = DMStagVecSplitToDMDA(dm,X,DMSTAG_ELEMENT,-dof[1],&pda,&subX);CHKERRQ(ierr);
        ierr = VecView(subX,v);CHKERRQ(ierr);
        pythonemitvec(fp,"X_cell");
        ierr = VecDestroy(&subX);CHKERRQ(ierr);
        ierr = DMDestroy(&pda);CHKERRQ(ierr);
      }
    } else if (dim == 2) {
      if (dof[0] != 0) {
        ierr = DMStagVecSplitToDMDA(dm,X,DMSTAG_DOWN_LEFT,-dof[0],&pda,&subX);CHKERRQ(ierr);
        ierr = VecView(subX,v);CHKERRQ(ierr);
        pythonemitvec(fp,"X_vertex");
        ierr = VecDestroy(&subX);CHKERRQ(ierr);
        ierr = DMDestroy(&pda);CHKERRQ(ierr);
      }
      if (dof[1] != 0) {
        ierr = DMStagVecSplitToDMDA(dm,X,DMSTAG_LEFT,-dof[1],&pda,&subX);CHKERRQ(ierr);
        ierr = VecView(subX,v);CHKERRQ(ierr);
        pythonemitvec(fp,"X_face_x");
        ierr = VecDestroy(&subX);CHKERRQ(ierr);
        ierr = DMDestroy(&pda);CHKERRQ(ierr);
        
        ierr = DMStagVecSplitToDMDA(dm,X,DMSTAG_DOWN,-dof[1],&pda,&subX);CHKERRQ(ierr);
        ierr = VecView(subX,v);CHKERRQ(ierr);
        pythonemitvec(fp,"X_face_y");
        ierr = VecDestroy(&subX);CHKERRQ(ierr);
        ierr = DMDestroy(&pda);CHKERRQ(ierr);
      }
      if (dof[2] != 0) {
        ierr = DMStagVecSplitToDMDA(dm,X,DMSTAG_ELEMENT,-dof[2],&pda,&subX);CHKERRQ(ierr);
        ierr = VecView(subX,v);CHKERRQ(ierr);
        pythonemitvec(fp,"X_cell");
        ierr = VecDestroy(&subX);CHKERRQ(ierr);
        ierr = DMDestroy(&pda);CHKERRQ(ierr);
      }
    } else if (dim == 3) {
      if (dof[0] != 0) {
        ierr = DMStagVecSplitToDMDA(dm,X,DMSTAG_DOWN_LEFT,-dof[0],&pda,&subX);CHKERRQ(ierr);
        ierr = VecView(subX,v);CHKERRQ(ierr);
        pythonemitvec(fp,"X_vertex");
        ierr = VecDestroy(&subX);CHKERRQ(ierr);
        ierr = DMDestroy(&pda);CHKERRQ(ierr);
      }
      if (dof[1] != 0) SETERRQ(comm,PETSC_ERR_SUP,"No support for edge data (3D)");
      if (dof[2] != 0) {
        ierr = DMStagVecSplitToDMDA(dm,X,DMSTAG_LEFT,-dof[2],&pda,&subX);CHKERRQ(ierr);
        ierr = VecView(subX,v);CHKERRQ(ierr);
        pythonemitvec(fp,"X_face_x");
        ierr = VecDestroy(&subX);CHKERRQ(ierr);
        ierr = DMDestroy(&pda);CHKERRQ(ierr);
        
        ierr = DMStagVecSplitToDMDA(dm,X,DMSTAG_DOWN,-dof[2],&pda,&subX);CHKERRQ(ierr);
        ierr = VecView(subX,v);CHKERRQ(ierr);
        pythonemitvec(fp,"X_face_y");
        ierr = VecDestroy(&subX);CHKERRQ(ierr);
        ierr = DMDestroy(&pda);CHKERRQ(ierr);
        
        ierr = DMStagVecSplitToDMDA(dm,X,DMSTAG_BACK,-dof[2],&pda,&subX);CHKERRQ(ierr);
        ierr = VecView(subX,v);CHKERRQ(ierr);
        pythonemitvec(fp,"X_face_z");
        ierr = VecDestroy(&subX);CHKERRQ(ierr);
        ierr = DMDestroy(&pda);CHKERRQ(ierr);
      }
      if (dof[3] != 0) {
        ierr = DMStagVecSplitToDMDA(dm,X,DMSTAG_ELEMENT,-dof[3],&pda,&subX);CHKERRQ(ierr);
        ierr = VecView(subX,v);CHKERRQ(ierr);
        pythonemitvec(fp,"X_cell");
        ierr = VecDestroy(&subX);CHKERRQ(ierr);
        ierr = DMDestroy(&pda);CHKERRQ(ierr);
      }
    }
  }
  
  pythonemit(fp,"  return data\n\n");
  
  pythonemit(fp,"def _PETScBinaryLoadReportNames(data):\n");
  ierr = PetscSNPrintf(string,PETSC_MAX_PATH_LEN-1,"  print('Filename: %s')\n",fname);CHKERRQ(ierr);
  pythonemit(fp,string);
  pythonemit(fp,"  print('Contents:')\n");
  pythonemit(fp,"  for key in data:\n");
  pythonemit(fp,"    print('  textual name registered:',key)\n\n");
  
  pythonemit(fp,"def demo_load_report():\n");
  pythonemit(fp," data = _PETScBinaryLoad()\n");
  pythonemit(fp," _PETScBinaryLoadReportNames(data)\n");
  
  ierr = PetscViewerDestroy(&v);CHKERRQ(ierr);
  fclose(fp);
  
  PetscFunctionReturn(0);
}

