/* Output routines for DMStag */
#include "dmstagoutput.h"

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
  
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  ierr = DMStagGetGlobalSizes(dm,&M,&N,&P);CHKERRQ(ierr);
  
  ierr = PetscViewerBinaryWrite(v,(void*)&M,1,PETSC_INT);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(v,(void*)&N,1,PETSC_INT);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(v,(void*)&P,1,PETSC_INT);CHKERRQ(ierr);
  
  pythonemit(fp,"    v = io.readInteger(fp)\n"); pythonemit(fp,"    data['Nx'] = v\n");
  pythonemit(fp,"    v = io.readInteger(fp)\n"); pythonemit(fp,"    data['Ny'] = v\n");
  pythonemit(fp,"    v = io.readInteger(fp)\n"); pythonemit(fp,"    data['Nz'] = v\n");

  { // output  
    PetscInt dof[4];
    ierr = DMStagGetDOF(dm,&dof[0],&dof[1],&dof[2],&dof[3]);CHKERRQ(ierr);

    ierr = PetscViewerBinaryWrite(v,(void*)&dim,1,PETSC_INT);CHKERRQ(ierr);
    ierr = PetscViewerBinaryWrite(v,(void*)&dof[0],1,PETSC_INT);CHKERRQ(ierr);
    ierr = PetscViewerBinaryWrite(v,(void*)&dof[1],1,PETSC_INT);CHKERRQ(ierr);
    ierr = PetscViewerBinaryWrite(v,(void*)&dof[2],1,PETSC_INT);CHKERRQ(ierr);
    ierr = PetscViewerBinaryWrite(v,(void*)&dof[3],1,PETSC_INT);CHKERRQ(ierr);
    
    pythonemit(fp,"    v = io.readInteger(fp)\n"); pythonemit(fp,"    data['dim'] = v\n");
    pythonemit(fp,"    v = io.readInteger(fp)\n"); pythonemit(fp,"    data['dof0'] = v\n");
    pythonemit(fp,"    v = io.readInteger(fp)\n"); pythonemit(fp,"    data['dof1'] = v\n");
    pythonemit(fp,"    v = io.readInteger(fp)\n"); pythonemit(fp,"    data['dof2'] = v\n");
    pythonemit(fp,"    v = io.readInteger(fp)\n"); pythonemit(fp,"    data['dof3'] = v\n");
  }
  
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

        // no need to check for dofs - output both vertex and center coordinates
        ierr = DMStagVecSplitToDMDA(subDM,coor,DMSTAG_LEFT,-dof[0],&pda,&subX);CHKERRQ(ierr);
        ierr = VecView(subX,v);CHKERRQ(ierr);
        pythonemitvec(fp,"x1d_vertex");
        ierr = VecDestroy(&subX);CHKERRQ(ierr);
        ierr = DMDestroy(&pda);CHKERRQ(ierr);

        ierr = DMStagVecSplitToDMDA(subDM,coor,DMSTAG_ELEMENT,-dof[1],&pda,&subX);CHKERRQ(ierr);
        ierr = VecView(subX,v);CHKERRQ(ierr);
        pythonemitvec(fp,"x1d_cell");
        ierr = VecDestroy(&subX);CHKERRQ(ierr);
        ierr = DMDestroy(&pda);CHKERRQ(ierr);
      }
      if (dim >= 2) {
        ierr = DMProductGetDM(cdm,1,&subDM);CHKERRQ(ierr);
        ierr = DMGetCoordinates(subDM,&coor);CHKERRQ(ierr);
        ierr = VecView(coor,v);CHKERRQ(ierr);
        pythonemitvec(fp,"y1d");
        
        ierr = DMStagGetDOF(subDM,&dof[0],&dof[1],&dof[2],&dof[3]);CHKERRQ(ierr);

        ierr = DMStagVecSplitToDMDA(subDM,coor,DMSTAG_LEFT,-dof[0],&pda,&subX);CHKERRQ(ierr);
        ierr = VecView(subX,v);CHKERRQ(ierr);
        pythonemitvec(fp,"y1d_vertex");
        ierr = VecDestroy(&subX);CHKERRQ(ierr);
        ierr = DMDestroy(&pda);CHKERRQ(ierr);

        ierr = DMStagVecSplitToDMDA(subDM,coor,DMSTAG_ELEMENT,-dof[1],&pda,&subX);CHKERRQ(ierr);
        ierr = VecView(subX,v);CHKERRQ(ierr);
        pythonemitvec(fp,"y1d_cell");
        ierr = VecDestroy(&subX);CHKERRQ(ierr);
        ierr = DMDestroy(&pda);CHKERRQ(ierr);
      }
      if (dim == 3) {
        ierr = DMProductGetDM(cdm,2,&subDM);CHKERRQ(ierr);
        ierr = DMGetCoordinates(subDM,&coor);CHKERRQ(ierr);
        ierr = VecView(coor,v);CHKERRQ(ierr);
        pythonemitvec(fp,"z1d");
        
        ierr = DMStagGetDOF(subDM,&dof[0],&dof[1],&dof[2],&dof[3]);CHKERRQ(ierr);

        ierr = DMStagVecSplitToDMDA(subDM,coor,DMSTAG_LEFT,-dof[0],&pda,&subX);CHKERRQ(ierr);
        ierr = VecView(subX,v);CHKERRQ(ierr);
        pythonemitvec(fp,"z1d_vertex");
        ierr = VecDestroy(&subX);CHKERRQ(ierr);
        ierr = DMDestroy(&pda);CHKERRQ(ierr);

        ierr = DMStagVecSplitToDMDA(subDM,coor,DMSTAG_ELEMENT,-dof[1],&pda,&subX);CHKERRQ(ierr);
        ierr = VecView(subX,v);CHKERRQ(ierr);
        pythonemitvec(fp,"z1d_cell");
        ierr = VecDestroy(&subX);CHKERRQ(ierr);
        ierr = DMDestroy(&pda);CHKERRQ(ierr);

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
  
  pythonemit(fp,"    return data\n\n");
  
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
  PetscErrorCode ierr;
  PetscViewer v;
  PetscInt M,N,P,dim;
  FILE *fp = NULL;
  char fname[PETSC_MAX_PATH_LEN],string[PETSC_MAX_PATH_LEN];
  MPI_Comm comm;
  PetscMPIInt rank;
  PetscBool view_coords = PETSC_TRUE; /* ultimately this would be an input arg */
  
  comm = PetscObjectComm((PetscObject)dm);
  ierr = MPI_Comm_rank(comm,&rank); CHKERRQ(ierr);
  
  /* check for instances of "." in the file name so that the file can be imported */
  {
    size_t k,len;
    ierr = PetscStrlen(prefix,&len);CHKERRQ(ierr);
    for (k=0; k<len; k++) if (prefix[k] == '.') PetscPrintf(comm,"[DMStagViewBinaryPython_SEQ] Warning: prefix %s contains the symbol '.'. Hence you will not be able to import the emiited python script. Consider change the prefix\n",prefix);
  }
  
  ierr = PetscSNPrintf(fname,PETSC_MAX_PATH_LEN-1,"%s.pbin",prefix);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(comm,fname,FILE_MODE_WRITE,&v);CHKERRQ(ierr);
  
  ierr = PetscSNPrintf(string,PETSC_MAX_PATH_LEN-1,"%s.py",prefix);CHKERRQ(ierr);
  
  if (rank == 0) {
    fp = fopen(string,"w");
    if (!fp) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Cannot open file %s",string);
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
  
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  ierr = DMStagGetGlobalSizes(dm,&M,&N,&P);CHKERRQ(ierr);
  
  ierr = PetscViewerBinaryWrite(v,(void*)&M,1,PETSC_INT);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(v,(void*)&N,1,PETSC_INT);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(v,(void*)&P,1,PETSC_INT);CHKERRQ(ierr);
  
  pythonemit(fp,"    v = io.readInteger(fp)\n"); pythonemit(fp,"    data['Nx'] = v\n");
  pythonemit(fp,"    v = io.readInteger(fp)\n"); pythonemit(fp,"    data['Ny'] = v\n");
  pythonemit(fp,"    v = io.readInteger(fp)\n"); pythonemit(fp,"    data['Nz'] = v\n");
  
  if (view_coords) {
    DM cdm,subDM;
    PetscBool isProduct;
    Vec coor;
    DM pda;
    Vec subX;
    PetscInt dof[4],Mp,Np,Pp,ip,jp,kp;
    PetscMPIInt rank_1;
    PetscBool active;
    
    ierr = DMGetCoordinateDM(dm,&cdm);CHKERRQ(ierr);
    ierr = DMStagGetNumRanks(dm,&Mp,&Np,&Pp);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)cdm,DMPRODUCT,&isProduct);CHKERRQ(ierr);
    if (isProduct) {
            
      if (dim >= 1) {
        PetscInt mlocal;
        Vec coorn;
        
        ierr = DMProductGetDM(cdm,0,&subDM);CHKERRQ(ierr);
        ierr = DMGetCoordinates(subDM,&coor);CHKERRQ(ierr);
        
        active = PETSC_FALSE;
        jp = 0;
        kp = 0;
        for (ip=0; ip<Mp; ip++) {
          rank_1 = ip + jp * Mp + kp * Mp * Np;
          if (rank_1 == rank) { active = PETSC_TRUE; break; }
        }
        
        mlocal = 0;
        if (active) {
          ierr = VecGetLocalSize(coor,&mlocal);CHKERRQ(ierr);
        }
        {
          const PetscScalar *LA_c = NULL;
          
          ierr = VecCreateMPIWithArray(comm,1,mlocal,PETSC_DECIDE,NULL,&coorn);CHKERRQ(ierr);
          if (active) {
            ierr = VecGetArrayRead(coor,&LA_c);CHKERRQ(ierr);
            ierr = VecPlaceArray(coorn,LA_c);CHKERRQ(ierr);
            ierr = VecRestoreArrayRead(coor,&LA_c);CHKERRQ(ierr);
          }
        }
        ierr = VecView(coorn,v);CHKERRQ(ierr);
        pythonemitvec(fp,"x1d");
        ierr = VecDestroy(&coorn);CHKERRQ(ierr);
        
        ierr = DMStagGetDOF(subDM,&dof[0],&dof[1],&dof[2],&dof[3]);CHKERRQ(ierr);
        if (dof[0] != 0) {
          ierr = DMStagVecSplitToDMDA(subDM,coor,DMSTAG_LEFT,-dof[0],&pda,&subX);CHKERRQ(ierr);
          
          mlocal = 0;
          if (active) {
            ierr = VecGetLocalSize(subX,&mlocal);CHKERRQ(ierr);
          }
          {
            const PetscScalar *LA_c = NULL;
            
            ierr = VecCreateMPIWithArray(comm,1,mlocal,PETSC_DECIDE,NULL,&coorn);CHKERRQ(ierr);
            if (active) {
              ierr = VecGetArrayRead(subX,&LA_c);CHKERRQ(ierr);
              ierr = VecPlaceArray(coorn,LA_c);CHKERRQ(ierr);
              ierr = VecRestoreArrayRead(subX,&LA_c);CHKERRQ(ierr);
            }
          }
          ierr = VecView(coorn,v);CHKERRQ(ierr);
          pythonemitvec(fp,"x1d_vertex");
          ierr = VecDestroy(&coorn);CHKERRQ(ierr);
          ierr = VecDestroy(&subX);CHKERRQ(ierr);
          ierr = DMDestroy(&pda);CHKERRQ(ierr);
        }
        
        if (dof[1] != 0) {
          ierr = DMStagVecSplitToDMDA(subDM,coor,DMSTAG_ELEMENT,-dof[1],&pda,&subX);CHKERRQ(ierr);

          mlocal = 0;
          if (active) {
            ierr = VecGetLocalSize(subX,&mlocal);CHKERRQ(ierr);
          }
          {
            const PetscScalar *LA_c = NULL;
            
            ierr = VecCreateMPIWithArray(comm,1,mlocal,PETSC_DECIDE,NULL,&coorn);CHKERRQ(ierr);
            if (active) {
              ierr = VecGetArrayRead(subX,&LA_c);CHKERRQ(ierr);
              ierr = VecPlaceArray(coorn,LA_c);CHKERRQ(ierr);
              ierr = VecRestoreArrayRead(subX,&LA_c);CHKERRQ(ierr);
            }
          }
          ierr = VecView(coorn,v);CHKERRQ(ierr);
          pythonemitvec(fp,"x1d_cell");
          ierr = VecDestroy(&coorn);CHKERRQ(ierr);
          ierr = VecDestroy(&subX);CHKERRQ(ierr);
          ierr = DMDestroy(&pda);CHKERRQ(ierr);
        }
      }
      
      
      if (dim >= 2) {
        PetscInt mlocal;
        Vec coorn;

        ierr = DMProductGetDM(cdm,1,&subDM);CHKERRQ(ierr);
        ierr = DMGetCoordinates(subDM,&coor);CHKERRQ(ierr);

        active = PETSC_FALSE;
        ip = 0;
        kp = 0;
        for (jp=0; jp<Np; jp++) {
          rank_1 = ip + jp * Mp + kp * Mp * Np;
          if (rank_1 == rank) { active = PETSC_TRUE; break; }
        }
        
        mlocal = 0;
        if (active) {
          ierr = VecGetLocalSize(coor,&mlocal);CHKERRQ(ierr);
        }
        {
          const PetscScalar *LA_c = NULL;
          
          ierr = VecCreateMPIWithArray(comm,1,mlocal,PETSC_DECIDE,NULL,&coorn);CHKERRQ(ierr);
          if (active) {
            ierr = VecGetArrayRead(coor,&LA_c);CHKERRQ(ierr);
            ierr = VecPlaceArray(coorn,LA_c);CHKERRQ(ierr);
            ierr = VecRestoreArrayRead(coor,&LA_c);CHKERRQ(ierr);
          }
        }
        ierr = VecView(coorn,v);CHKERRQ(ierr);
        pythonemitvec(fp,"y1d");
        ierr = VecDestroy(&coorn);CHKERRQ(ierr);

        ierr = DMStagGetDOF(subDM,&dof[0],&dof[1],&dof[2],&dof[3]);CHKERRQ(ierr);
        if (dof[0] != 0) {
          ierr = DMStagVecSplitToDMDA(subDM,coor,DMSTAG_LEFT,-dof[0],&pda,&subX);CHKERRQ(ierr);

          mlocal = 0;
          if (active) {
            ierr = VecGetLocalSize(subX,&mlocal);CHKERRQ(ierr);
          }
          {
            const PetscScalar *LA_c = NULL;
            
            ierr = VecCreateMPIWithArray(comm,1,mlocal,PETSC_DECIDE,NULL,&coorn);CHKERRQ(ierr);
            if (active) {
              ierr = VecGetArrayRead(subX,&LA_c);CHKERRQ(ierr);
              ierr = VecPlaceArray(coorn,LA_c);CHKERRQ(ierr);
              ierr = VecRestoreArrayRead(subX,&LA_c);CHKERRQ(ierr);
            }
          }
          ierr = VecView(coorn,v);CHKERRQ(ierr);

          pythonemitvec(fp,"y1d_vertex");
          ierr = VecDestroy(&coorn);CHKERRQ(ierr);
          ierr = VecDestroy(&subX);CHKERRQ(ierr);
          ierr = DMDestroy(&pda);CHKERRQ(ierr);
        }
        
        if (dof[1] != 0) {
          ierr = DMStagVecSplitToDMDA(subDM,coor,DMSTAG_ELEMENT,-dof[1],&pda,&subX);CHKERRQ(ierr);
          
          mlocal = 0;
          if (active) {
            ierr = VecGetLocalSize(subX,&mlocal);CHKERRQ(ierr);
          }
          {
            const PetscScalar *LA_c = NULL;
            
            ierr = VecCreateMPIWithArray(comm,1,mlocal,PETSC_DECIDE,NULL,&coorn);CHKERRQ(ierr);
            if (active) {
              ierr = VecGetArrayRead(subX,&LA_c);CHKERRQ(ierr);
              ierr = VecPlaceArray(coorn,LA_c);CHKERRQ(ierr);
              ierr = VecRestoreArrayRead(subX,&LA_c);CHKERRQ(ierr);
            }
          }
          ierr = VecView(coorn,v);CHKERRQ(ierr);

          pythonemitvec(fp,"y1d_cell");
          ierr = VecDestroy(&coorn);CHKERRQ(ierr);
          ierr = VecDestroy(&subX);CHKERRQ(ierr);
          ierr = DMDestroy(&pda);CHKERRQ(ierr);
        }
      }
      
      
      if (dim == 3) {
        PetscInt mlocal;
        Vec coorn;

        ierr = DMProductGetDM(cdm,2,&subDM);CHKERRQ(ierr);
        ierr = DMGetCoordinates(subDM,&coor);CHKERRQ(ierr);
        
        active = PETSC_FALSE;
        ip = 0;
        jp = 0;
        for (kp=0; kp<Pp; kp++) {
          rank_1 = ip + jp * Mp + kp * Mp * Np;
          if (rank_1 == rank) { active = PETSC_TRUE; break; }
        }
        
        mlocal = 0;
        if (active) {
          ierr = VecGetLocalSize(coor,&mlocal);CHKERRQ(ierr);
        }
        {
          const PetscScalar *LA_c = NULL;
          
          ierr = VecCreateMPIWithArray(comm,1,mlocal,PETSC_DECIDE,NULL,&coorn);CHKERRQ(ierr);
          if (active) {
            ierr = VecGetArrayRead(coor,&LA_c);CHKERRQ(ierr);
            ierr = VecPlaceArray(coorn,LA_c);CHKERRQ(ierr);
            ierr = VecRestoreArrayRead(coor,&LA_c);CHKERRQ(ierr);
          }
        }
        ierr = VecView(coorn,v);CHKERRQ(ierr);
        pythonemitvec(fp,"z1d");
        ierr = VecDestroy(&coorn);CHKERRQ(ierr);
        
        ierr = DMStagGetDOF(subDM,&dof[0],&dof[1],&dof[2],&dof[3]);CHKERRQ(ierr);
        if (dof[0] != 0) {
          ierr = DMStagVecSplitToDMDA(subDM,coor,DMSTAG_LEFT,-dof[0],&pda,&subX);CHKERRQ(ierr);

          mlocal = 0;
          if (active) {
            ierr = VecGetLocalSize(subX,&mlocal);CHKERRQ(ierr);
          }
          {
            const PetscScalar *LA_c = NULL;
            
            ierr = VecCreateMPIWithArray(comm,1,mlocal,PETSC_DECIDE,NULL,&coorn);CHKERRQ(ierr);
            if (active) {
              ierr = VecGetArrayRead(subX,&LA_c);CHKERRQ(ierr);
              ierr = VecPlaceArray(coorn,LA_c);CHKERRQ(ierr);
              ierr = VecRestoreArrayRead(subX,&LA_c);CHKERRQ(ierr);
            }
          }
          ierr = VecView(coorn,v);CHKERRQ(ierr);

          pythonemitvec(fp,"z1d_vertex");
          ierr = VecDestroy(&coorn);CHKERRQ(ierr);
          ierr = VecDestroy(&subX);CHKERRQ(ierr);
          ierr = DMDestroy(&pda);CHKERRQ(ierr);
        }
        
        if (dof[1] != 0) {
          ierr = DMStagVecSplitToDMDA(subDM,coor,DMSTAG_ELEMENT,-dof[1],&pda,&subX);CHKERRQ(ierr);

          mlocal = 0;
          if (active) {
            ierr = VecGetLocalSize(subX,&mlocal);CHKERRQ(ierr);
          }
          {
            const PetscScalar *LA_c = NULL;
            
            ierr = VecCreateMPIWithArray(comm,1,mlocal,PETSC_DECIDE,NULL,&coorn);CHKERRQ(ierr);
            if (active) {
              ierr = VecGetArrayRead(subX,&LA_c);CHKERRQ(ierr);
              ierr = VecPlaceArray(coorn,LA_c);CHKERRQ(ierr);
              ierr = VecRestoreArrayRead(subX,&LA_c);CHKERRQ(ierr);
            }
          }
          ierr = VecView(coorn,v);CHKERRQ(ierr);

          pythonemitvec(fp,"z1d_cell");
          ierr = VecDestroy(&coorn);CHKERRQ(ierr);
          ierr = VecDestroy(&subX);CHKERRQ(ierr);
          ierr = DMDestroy(&pda);CHKERRQ(ierr);
        }

      }
    } else SETERRQ(comm,PETSC_ERR_SUP,"Only supports coordinated defined via DMPRODUCT");
  }
  
  {
    DM pda;
    Vec subX,subXn;
    PetscInt dof[4];
    
    ierr = DMStagGetDOF(dm,&dof[0],&dof[1],&dof[2],&dof[3]);CHKERRQ(ierr);
    
    if (dim == 1) {
      if (dof[0] != 0) {
        ierr = DMStagVecSplitToDMDA(dm,X,DMSTAG_DOWN_LEFT,-dof[0],&pda,&subX);CHKERRQ(ierr);
        ierr = DMDACreateNaturalVector(pda,&subXn);CHKERRQ(ierr);
        ierr = DMDAGlobalToNaturalBegin(pda,subX,INSERT_VALUES,subXn);CHKERRQ(ierr);
        ierr = DMDAGlobalToNaturalEnd(pda,subX,INSERT_VALUES,subXn);CHKERRQ(ierr);
        ierr = VecView(subXn,v);CHKERRQ(ierr);
        pythonemitvec(fp,"X_vertex");
        ierr = VecDestroy(&subXn);CHKERRQ(ierr);
        ierr = VecDestroy(&subX);CHKERRQ(ierr);
        ierr = DMDestroy(&pda);CHKERRQ(ierr);
      }
      if (dof[1] != 0) {
        ierr = DMStagVecSplitToDMDA(dm,X,DMSTAG_ELEMENT,-dof[1],&pda,&subX);CHKERRQ(ierr);
        ierr = DMDACreateNaturalVector(pda,&subXn);CHKERRQ(ierr);
        ierr = DMDAGlobalToNaturalBegin(pda,subX,INSERT_VALUES,subXn);CHKERRQ(ierr);
        ierr = DMDAGlobalToNaturalEnd(pda,subX,INSERT_VALUES,subXn);CHKERRQ(ierr);
        ierr = VecView(subXn,v);CHKERRQ(ierr);
        pythonemitvec(fp,"X_cell");
        ierr = VecDestroy(&subXn);CHKERRQ(ierr);
        ierr = VecDestroy(&subX);CHKERRQ(ierr);
        ierr = DMDestroy(&pda);CHKERRQ(ierr);
      }
    } else if (dim == 2) {
      if (dof[0] != 0) {
        ierr = DMStagVecSplitToDMDA(dm,X,DMSTAG_DOWN_LEFT,-dof[0],&pda,&subX);CHKERRQ(ierr);
        ierr = DMDACreateNaturalVector(pda,&subXn);CHKERRQ(ierr);
        ierr = DMDAGlobalToNaturalBegin(pda,subX,INSERT_VALUES,subXn);CHKERRQ(ierr);
        ierr = DMDAGlobalToNaturalEnd(pda,subX,INSERT_VALUES,subXn);CHKERRQ(ierr);
        ierr = VecView(subXn,v);CHKERRQ(ierr);
        pythonemitvec(fp,"X_vertex");
        ierr = VecDestroy(&subXn);CHKERRQ(ierr);
        ierr = VecDestroy(&subX);CHKERRQ(ierr);
        ierr = DMDestroy(&pda);CHKERRQ(ierr);
      }
      if (dof[1] != 0) {
        ierr = DMStagVecSplitToDMDA(dm,X,DMSTAG_LEFT,-dof[1],&pda,&subX);CHKERRQ(ierr);
        ierr = DMDACreateNaturalVector(pda,&subXn);CHKERRQ(ierr);
        ierr = DMDAGlobalToNaturalBegin(pda,subX,INSERT_VALUES,subXn);CHKERRQ(ierr);
        ierr = DMDAGlobalToNaturalEnd(pda,subX,INSERT_VALUES,subXn);CHKERRQ(ierr);
        ierr = VecView(subXn,v);CHKERRQ(ierr);
        pythonemitvec(fp,"X_face_x");
        ierr = VecDestroy(&subXn);CHKERRQ(ierr);
        ierr = VecDestroy(&subX);CHKERRQ(ierr);
        ierr = DMDestroy(&pda);CHKERRQ(ierr);
        
        ierr = DMStagVecSplitToDMDA(dm,X,DMSTAG_DOWN,-dof[1],&pda,&subX);CHKERRQ(ierr);
        ierr = DMDACreateNaturalVector(pda,&subXn);CHKERRQ(ierr);
        ierr = DMDAGlobalToNaturalBegin(pda,subX,INSERT_VALUES,subXn);CHKERRQ(ierr);
        ierr = DMDAGlobalToNaturalEnd(pda,subX,INSERT_VALUES,subXn);CHKERRQ(ierr);
        ierr = VecView(subXn,v);CHKERRQ(ierr);
        pythonemitvec(fp,"X_face_y");
        ierr = VecDestroy(&subXn);CHKERRQ(ierr);
        ierr = VecDestroy(&subX);CHKERRQ(ierr);
        ierr = DMDestroy(&pda);CHKERRQ(ierr);
      }
      if (dof[2] != 0) {
        ierr = DMStagVecSplitToDMDA(dm,X,DMSTAG_ELEMENT,-dof[2],&pda,&subX);CHKERRQ(ierr);
        ierr = DMDACreateNaturalVector(pda,&subXn);CHKERRQ(ierr);
        ierr = DMDAGlobalToNaturalBegin(pda,subX,INSERT_VALUES,subXn);CHKERRQ(ierr);
        ierr = DMDAGlobalToNaturalEnd(pda,subX,INSERT_VALUES,subXn);CHKERRQ(ierr);
        ierr = VecView(subXn,v);CHKERRQ(ierr);
        pythonemitvec(fp,"X_cell");
        ierr = VecDestroy(&subXn);CHKERRQ(ierr);
        ierr = VecDestroy(&subX);CHKERRQ(ierr);
        ierr = DMDestroy(&pda);CHKERRQ(ierr);
      }
    } else if (dim == 3) {
      if (dof[0] != 0) {
        ierr = DMStagVecSplitToDMDA(dm,X,DMSTAG_DOWN_LEFT,-dof[0],&pda,&subX);CHKERRQ(ierr);
        ierr = DMDACreateNaturalVector(pda,&subXn);CHKERRQ(ierr);
        ierr = DMDAGlobalToNaturalBegin(pda,subX,INSERT_VALUES,subXn);CHKERRQ(ierr);
        ierr = DMDAGlobalToNaturalEnd(pda,subX,INSERT_VALUES,subXn);CHKERRQ(ierr);
        ierr = VecView(subXn,v);CHKERRQ(ierr);
        pythonemitvec(fp,"X_vertex");
        ierr = VecDestroy(&subXn);CHKERRQ(ierr);
        ierr = VecDestroy(&subX);CHKERRQ(ierr);
        ierr = DMDestroy(&pda);CHKERRQ(ierr);
      }
      if (dof[1] != 0) SETERRQ(comm,PETSC_ERR_SUP,"No support for edge data (3D)");
      if (dof[2] != 0) {
        ierr = DMStagVecSplitToDMDA(dm,X,DMSTAG_LEFT,-dof[2],&pda,&subX);CHKERRQ(ierr);
        ierr = DMDACreateNaturalVector(pda,&subXn);CHKERRQ(ierr);
        ierr = DMDAGlobalToNaturalBegin(pda,subX,INSERT_VALUES,subXn);CHKERRQ(ierr);
        ierr = DMDAGlobalToNaturalEnd(pda,subX,INSERT_VALUES,subXn);CHKERRQ(ierr);
        ierr = VecView(subXn,v);CHKERRQ(ierr);
        pythonemitvec(fp,"X_face_x");
        ierr = VecDestroy(&subXn);CHKERRQ(ierr);
        ierr = VecDestroy(&subX);CHKERRQ(ierr);
        ierr = DMDestroy(&pda);CHKERRQ(ierr);
        
        ierr = DMStagVecSplitToDMDA(dm,X,DMSTAG_DOWN,-dof[2],&pda,&subX);CHKERRQ(ierr);
        ierr = DMDACreateNaturalVector(pda,&subXn);CHKERRQ(ierr);
        ierr = DMDAGlobalToNaturalBegin(pda,subX,INSERT_VALUES,subXn);CHKERRQ(ierr);
        ierr = DMDAGlobalToNaturalEnd(pda,subX,INSERT_VALUES,subXn);CHKERRQ(ierr);
        ierr = VecView(subXn,v);CHKERRQ(ierr);
        pythonemitvec(fp,"X_face_y");
        ierr = VecDestroy(&subXn);CHKERRQ(ierr);
        ierr = VecDestroy(&subX);CHKERRQ(ierr);
        ierr = DMDestroy(&pda);CHKERRQ(ierr);
        
        ierr = DMStagVecSplitToDMDA(dm,X,DMSTAG_BACK,-dof[2],&pda,&subX);CHKERRQ(ierr);
        ierr = DMDACreateNaturalVector(pda,&subXn);CHKERRQ(ierr);
        ierr = DMDAGlobalToNaturalBegin(pda,subX,INSERT_VALUES,subXn);CHKERRQ(ierr);
        ierr = DMDAGlobalToNaturalEnd(pda,subX,INSERT_VALUES,subXn);CHKERRQ(ierr);
        ierr = VecView(subXn,v);CHKERRQ(ierr);
        pythonemitvec(fp,"X_face_z");
        ierr = VecDestroy(&subXn);CHKERRQ(ierr);
        ierr = VecDestroy(&subX);CHKERRQ(ierr);
        ierr = DMDestroy(&pda);CHKERRQ(ierr);
      }
      if (dof[3] != 0) {
        ierr = DMStagVecSplitToDMDA(dm,X,DMSTAG_ELEMENT,-dof[3],&pda,&subX);CHKERRQ(ierr);
        ierr = DMDACreateNaturalVector(pda,&subXn);CHKERRQ(ierr);
        ierr = DMDAGlobalToNaturalBegin(pda,subX,INSERT_VALUES,subXn);CHKERRQ(ierr);
        ierr = DMDAGlobalToNaturalEnd(pda,subX,INSERT_VALUES,subXn);CHKERRQ(ierr);
        ierr = VecView(subXn,v);CHKERRQ(ierr);
        pythonemitvec(fp,"X_cell");
        ierr = VecDestroy(&subXn);CHKERRQ(ierr);
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
  if (fp) fclose(fp);
  
  PetscFunctionReturn(0);
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
  PetscErrorCode ierr;
  MPI_Comm comm;
  PetscMPIInt size;
  PetscBool view_coords = PETSC_TRUE; /* ultimately this would be an input arg */
  
  comm = PetscObjectComm((PetscObject)dm);
  ierr = MPI_Comm_size(comm,&size); CHKERRQ(ierr);
  if (size == 1) {
    ierr = DMStagViewBinaryPython_SEQ(dm,X,prefix);CHKERRQ(ierr);
  } else {
    ierr = DMStagViewBinaryPython_MPI(dm,X,prefix);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
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
  PetscInt M,N,P,dim,dof[4];
  MPI_Comm comm;
  PetscMPIInt size;
  PetscErrorCode ierr;

  comm = PETSC_COMM_WORLD;
  ierr = MPI_Comm_size(comm,&size); CHKERRQ(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Read is supported 'sequential' only");

  // this seems to be the fix for multiple dofs
  ierr = PetscOptionsSetValue(NULL, "-viewer_binary_skip_info", ""); CHKERRQ(ierr);

  ierr = PetscSNPrintf(fname,PETSC_MAX_PATH_LEN-1,"%s.pbin",prefix);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(comm,fname,FILE_MODE_READ,&v);CHKERRQ(ierr);

  ierr = PetscViewerBinaryRead(v,(void*)&M,1,NULL,PETSC_INT);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(v,(void*)&N,1,NULL,PETSC_INT);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(v,(void*)&P,1,NULL,PETSC_INT);CHKERRQ(ierr);

  ierr = PetscViewerBinaryRead(v,(void*)&dim,1,NULL,PETSC_INT);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(v,(void*)&dof[0],1,NULL,PETSC_INT);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(v,(void*)&dof[1],1,NULL,PETSC_INT);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(v,(void*)&dof[2],1,NULL,PETSC_INT);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(v,(void*)&dof[3],1,NULL,PETSC_INT);CHKERRQ(ierr);

  // create dm and vector at this point
  if (dim==2) {
    ierr = DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,M,N,PETSC_DECIDE,PETSC_DECIDE, 
                        dof[0], dof[1], dof[2], DMSTAG_STENCIL_BOX,1, NULL,NULL, &dm);CHKERRQ(ierr);
    ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
    ierr = DMSetUp(dm);CHKERRQ(ierr);
    ierr = DMStagSetUniformCoordinatesProduct(dm,0.0,1.0,0.0,1.0,0.0,0.0);CHKERRQ(ierr);
    ierr = DMCreateGlobalVector(dm,&x);CHKERRQ(ierr);
  } else {
    SETERRQ(comm,PETSC_ERR_SUP,"Only valid for 2d DM");
  }

  { // read coordinates
    DM cdm, subDM;
    Vec coor, subX;

    ierr = DMGetCoordinateDM(dm,&cdm);CHKERRQ(ierr);
    if (dim >= 1) {
      ierr = DMProductGetDM(cdm,0,&subDM);CHKERRQ(ierr);
      ierr = DMGetCoordinates(subDM,&coor);CHKERRQ(ierr);
      ierr = VecLoad(coor,v); CHKERRQ(ierr); // x1d
      ierr = DMSetCoordinates(subDM,coor);CHKERRQ(ierr);

      // the following vectors are not needed
      ierr = VecCreate(comm,&subX);CHKERRQ(ierr);
      ierr = VecLoad(subX,v);CHKERRQ(ierr); // x1d_vertex
      ierr = VecDestroy(&subX);CHKERRQ(ierr);

      ierr = VecCreate(comm,&subX);CHKERRQ(ierr);
      ierr = VecLoad(subX,v);CHKERRQ(ierr);  // x1d_cell
      ierr = VecDestroy(&subX);CHKERRQ(ierr);
    }
    if (dim >= 2) {
      ierr = DMProductGetDM(cdm,1,&subDM);CHKERRQ(ierr);
      ierr = DMGetCoordinates(subDM,&coor);CHKERRQ(ierr);
      ierr = VecLoad(coor,v); CHKERRQ(ierr); // y1d
      ierr = DMSetCoordinates(subDM,coor);CHKERRQ(ierr);

      ierr = VecCreate(comm,&subX);CHKERRQ(ierr);
      ierr = VecLoad(subX,v);CHKERRQ(ierr); // y1d_vertex
      ierr = VecDestroy(&subX);CHKERRQ(ierr); 

      ierr = VecCreate(comm,&subX);CHKERRQ(ierr);
      ierr = VecLoad(subX,v);CHKERRQ(ierr); // y1d_cell
      ierr = VecDestroy(&subX);CHKERRQ(ierr); 
    }
    if (dim == 3) {
      ierr = DMProductGetDM(cdm,2,&subDM);CHKERRQ(ierr);
      ierr = DMGetCoordinates(subDM,&coor);CHKERRQ(ierr);
      ierr = VecLoad(coor,v); CHKERRQ(ierr); // z1d
      ierr = DMSetCoordinates(subDM,coor);CHKERRQ(ierr);

      ierr = VecCreate(comm,&subX);CHKERRQ(ierr);
      ierr = VecLoad(subX,v);CHKERRQ(ierr); // z1d_vertex
      ierr = VecDestroy(&subX);CHKERRQ(ierr); 

      ierr = VecCreate(comm,&subX);CHKERRQ(ierr);
      ierr = VecLoad(subX,v);CHKERRQ(ierr); // z1d_cell
      ierr = VecDestroy(&subX);CHKERRQ(ierr); 
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

      ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
      
      if (dof[0] != 0) {
        ierr = DMStagVecSplitToDMDA(dm,x,DMSTAG_DOWN_LEFT,-dof[0],&pda,&subX);CHKERRQ(ierr);
        ierr = VecLoad(subX,v);CHKERRQ(ierr); // X_vertex

        ierr = DMGetLocalVector(dm, &xlocal); CHKERRQ(ierr);
        ierr = DMGlobalToLocal (dm, x, INSERT_VALUES, xlocal); CHKERRQ(ierr);
        ierr = DMStagVecGetArray(dm, xlocal, &xx); CHKERRQ(ierr);

        // Loop over local domain
        for (idof = 0; idof < dof[0]; idof++) {
          ierr = DMStagGetLocationSlot(dm, DMSTAG_DOWN_LEFT, idof, &iloc); CHKERRQ(ierr);
          for (i = sx; i < sx+nx+1; i++) {
            for (j = sz; j <sz+nz+1; j++) {
              PetscScalar xval;
              PetscInt    ii;
              ii = (dof[0]*i+idof)+(dof[0]*j)*(M+1);
              ierr = VecGetValues(subX,1,&ii,&xval); 
              xx[j][i][iloc] = xval; 
            }
          }
        }

        // Restore arrays
        ierr = DMStagVecRestoreArray(dm,xlocal,&xx); CHKERRQ(ierr);
        ierr = DMLocalToGlobalBegin(dm,xlocal,INSERT_VALUES,x); CHKERRQ(ierr);
        ierr = DMLocalToGlobalEnd  (dm,xlocal,INSERT_VALUES,x); CHKERRQ(ierr);
        ierr = DMRestoreLocalVector(dm, &xlocal); CHKERRQ(ierr);

        ierr = VecDestroy(&subX);CHKERRQ(ierr);
        ierr = DMDestroy(&pda);CHKERRQ(ierr);
      }
      if (dof[1] != 0) {
        ierr = DMStagVecSplitToDMDA(dm,x,DMSTAG_LEFT,-dof[1],&pda,&subX);CHKERRQ(ierr);
        ierr = VecLoad(subX,v);CHKERRQ(ierr); // X_face_x

        ierr = DMGetLocalVector(dm, &xlocal); CHKERRQ(ierr);
        ierr = DMGlobalToLocal (dm, x, INSERT_VALUES, xlocal); CHKERRQ(ierr);
        ierr = DMStagVecGetArray(dm, xlocal, &xx); CHKERRQ(ierr);

        // Loop over local domain
        PetscInt icount = 0;
        for (idof = 0; idof < dof[1]; idof++) {
          ierr = DMStagGetLocationSlot(dm, DMSTAG_LEFT, idof, &iloc); CHKERRQ(ierr);
          for (i = sx; i < sx+nx+1; i++) {
            for (j = sz; j <sz+nz; j++) {
              PetscScalar xval;
              PetscInt    ii;
              ii = (dof[1]*i+idof)+(dof[1]*j)*(M+1);
              ierr = VecGetValues(subX,1,&ii,&xval);  
              xx[j][i][iloc] = xval;
            }
          }
        }

        // Restore arrays
        ierr = DMStagVecRestoreArray(dm,xlocal,&xx); CHKERRQ(ierr);
        ierr = DMLocalToGlobalBegin(dm,xlocal,INSERT_VALUES,x); CHKERRQ(ierr);
        ierr = DMLocalToGlobalEnd  (dm,xlocal,INSERT_VALUES,x); CHKERRQ(ierr);
        ierr = DMRestoreLocalVector(dm, &xlocal); CHKERRQ(ierr);

        ierr = VecDestroy(&subX);CHKERRQ(ierr);
        ierr = DMDestroy(&pda);CHKERRQ(ierr);

        ierr = DMStagVecSplitToDMDA(dm,x,DMSTAG_DOWN,-dof[1],&pda,&subX);CHKERRQ(ierr);
        ierr = VecLoad(subX,v);CHKERRQ(ierr); // X_face_y

        ierr = DMGetLocalVector(dm, &xlocal); CHKERRQ(ierr);
        ierr = DMGlobalToLocal (dm, x, INSERT_VALUES, xlocal); CHKERRQ(ierr);
        ierr = DMStagVecGetArray(dm, xlocal, &xx); CHKERRQ(ierr);

        // Loop over local domain
        for (idof = 0; idof < dof[1]; idof++) {
          ierr = DMStagGetLocationSlot(dm, DMSTAG_DOWN, idof, &iloc); CHKERRQ(ierr);
          for (j = sz; j < sz+nz+1; j++) {
            for (i = sx; i <sx+nx; i++) {
              PetscScalar xval;
              PetscInt    ii;
              ii = (dof[1]*i+idof)+(dof[1]*j)*M;
              ierr = VecGetValues(subX,1,&ii,&xval);  
              xx[j][i][iloc] = xval;
            }
          }
        }

        // Restore arrays
        ierr = DMStagVecRestoreArray(dm,xlocal,&xx); CHKERRQ(ierr);
        ierr = DMLocalToGlobalBegin(dm,xlocal,INSERT_VALUES,x); CHKERRQ(ierr);
        ierr = DMLocalToGlobalEnd  (dm,xlocal,INSERT_VALUES,x); CHKERRQ(ierr);
        ierr = DMRestoreLocalVector(dm, &xlocal); CHKERRQ(ierr);

        ierr = VecDestroy(&subX);CHKERRQ(ierr);
        ierr = DMDestroy(&pda);CHKERRQ(ierr);
      }
      if (dof[2] != 0) {
        ierr = DMStagVecSplitToDMDA(dm,x,DMSTAG_ELEMENT,-dof[2],&pda,&subX);CHKERRQ(ierr);
        ierr = VecLoad(subX,v);CHKERRQ(ierr); // X_cell
        
        ierr = DMGetLocalVector(dm, &xlocal); CHKERRQ(ierr);
        ierr = DMGlobalToLocal (dm, x, INSERT_VALUES, xlocal); CHKERRQ(ierr);
        ierr = DMStagVecGetArray(dm, xlocal, &xx); CHKERRQ(ierr);

        // Loop over local domain
        for (idof = 0; idof < dof[2]; idof++) {
          ierr = DMStagGetLocationSlot(dm, DMSTAG_ELEMENT, idof, &iloc); CHKERRQ(ierr);
          for (j = sz; j <sz+nz; j++) {
            for (i = sx; i < sx+nx; i++) {
              PetscScalar xval;
              PetscInt    ii;
              ii = (dof[2]*i+idof)+(dof[2]*j)*M;
              ierr = VecGetValues(subX,1,&ii,&xval); 
              xx[j][i][iloc] = xval; 
            }
          }
        }

        // Restore arrays
        ierr = DMStagVecRestoreArray(dm,xlocal,&xx); CHKERRQ(ierr);
        ierr = DMLocalToGlobalBegin(dm,xlocal,INSERT_VALUES,x); CHKERRQ(ierr);
        ierr = DMLocalToGlobalEnd  (dm,xlocal,INSERT_VALUES,x); CHKERRQ(ierr);
        ierr = DMRestoreLocalVector(dm, &xlocal); CHKERRQ(ierr);

        ierr = VecDestroy(&subX);CHKERRQ(ierr);
        ierr = DMDestroy(&pda);CHKERRQ(ierr);
      }
    } else if (dim == 3) {
      SETERRQ(comm,PETSC_ERR_SUP,"Only valid for 2d DM");
    }
  }

  ierr = PetscViewerDestroy(&v);CHKERRQ(ierr);

  *_dm = dm;
  *_x = x;

  PetscFunctionReturn(0);
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
  PetscErrorCode ierr;
  MPI_Comm comm;
  PetscMPIInt size;
  
  comm = PETSC_COMM_WORLD;
  ierr = MPI_Comm_size(comm,&size); CHKERRQ(ierr);
  if (size == 1) {
    ierr = DMStagReadBinaryPython_SEQ(&(*dm),&(*x),prefix);CHKERRQ(ierr);
  } else {
    SETERRQ(comm,PETSC_ERR_SUP,"Parallel read is not yet implemented.");
  }
  PetscFunctionReturn(0);
}