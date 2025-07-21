static char help[] = "DMStagCoupling test \n\n";
// run: ./test_coupling.sh -log_view

#include "../src/fdpde_snes.h"

PetscErrorCode test1(PetscInt mx,PetscInt my)
{
  DM              dms[20];
  PetscInt        dof0,dof1,dof2,stencilWidth,ndms,d;
  Mat             A;
  PetscFunctionBeginUser;
  
  dof0 = 0; dof1 = 0; dof2 = 1; /* (vertex) (face) (element) */
  stencilWidth = 1;
  ndms = 2;
  for (d=0; d<ndms; d++) {
    PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, mx, my,PETSC_DECIDE, PETSC_DECIDE, dof0, dof1, dof2,
                          DMSTAG_STENCIL_BOX, stencilWidth, NULL,NULL, &dms[d]));
    PetscCall(DMStagSetCoordinateDMType(dms[d],DMPRODUCT));
    PetscCall(DMSetFromOptions(dms[d]));
    PetscCall(DMSetUp(dms[d]));
    
    PetscCall(DMStagSetUniformCoordinatesProduct(dms[d],0.0,1.0,0.0,1.0,0.0,0.0));
  }
  
  PetscCall(FDPDECoupledCreateMatrix(ndms,dms,MATAIJ,&A));
  //PetscCall(FDPDECoupledCreateMatrix(1,&dms[0],MATAIJ,&A));
  PetscCall(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO));
  PetscCall(MatView(A,PETSC_VIEWER_STDOUT_WORLD));
  
  for (d=0; d<ndms; d++) {
    PetscCall(DMDestroy(&dms[d]));
  }
  PetscCall(MatDestroy(&A));
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode test2(PetscInt mx,PetscInt my)
{
  DM              dms[20];
  PetscInt        dof0,dof1,dof2,stencilWidth,ndms,d;
  Mat             A;
  PetscFunctionBeginUser;
  
  dof0 = 0; dof1 = 0; dof2 = 1; /* (vertex) (face) (element) */
  stencilWidth = 1;
  ndms = 2;
  for (d=0; d<ndms; d++) {
    PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, mx, my,PETSC_DECIDE, PETSC_DECIDE, dof0, dof1, dof2,
                          DMSTAG_STENCIL_STAR, stencilWidth, NULL,NULL, &dms[d]));
    PetscCall(DMStagSetCoordinateDMType(dms[d],DMPRODUCT));
    PetscCall(DMSetFromOptions(dms[d]));
    PetscCall(DMSetUp(dms[d]));
    
    PetscCall(DMStagSetUniformCoordinatesProduct(dms[d],0.0,1.0,0.0,1.0,0.0,0.0));
  }
  
  PetscCall(FDPDECoupledCreateMatrix(ndms,dms,MATAIJ,&A));
  //PetscCall(FDPDECoupledCreateMatrix(1,&dms[0],MATAIJ,&A));
  PetscCall(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO));
  PetscCall(MatView(A,PETSC_VIEWER_STDOUT_WORLD));
  
  for (d=0; d<ndms; d++) {
    PetscCall(DMDestroy(&dms[d]));
  }
  PetscCall(MatDestroy(&A));
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode test3(PetscInt mx,PetscInt my)
{
  DM              dms[20];
  PetscInt        dof0,dof1,dof2,stencilWidth,ndms,d;
  Mat             A;
  PetscFunctionBeginUser;
  
  stencilWidth = 1;
  ndms = 4;
  for (d=0; d<ndms; d++) {
    if (d == 0) {
      dof0 = 0; dof1 = 1; dof2 = 0; /* (vertex) (face) (element) */
    }
    else if (d == 1) {
      dof0 = 0; dof1 = 1; dof2 = 0; /* (vertex) (face) (element) */
    }
    else {
      dof0 = 0; dof1 = 0; dof2 = 1; /* (vertex) (face) (element) */
    }

    PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, mx, my,PETSC_DECIDE, PETSC_DECIDE, dof0, dof1, dof2,
                          DMSTAG_STENCIL_STAR, stencilWidth, NULL,NULL, &dms[d]));
    PetscCall(DMSetFromOptions(dms[d]));
    PetscCall(DMSetUp(dms[d]));
  }
  
  PetscCall(FDPDECoupledCreateMatrix(ndms,dms,MATAIJ,&A));
  PetscCall(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO));
  PetscCall(MatView(A,PETSC_VIEWER_STDOUT_WORLD));
  
  for (d=0; d<ndms; d++) {
    PetscCall(DMDestroy(&dms[d]));
  }
  PetscCall(MatDestroy(&A));
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode test4(PetscInt mx,PetscInt my)
{
  DM              dms[20];
  PetscInt        dof0,dof1,dof2,stencilWidth,ndms,d;
  Mat             A;
  PetscBool       mask[16];
  PetscFunctionBeginUser;
  
  mask[0]  = PETSC_FALSE;   mask[1]  = PETSC_TRUE;   mask[2]  = PETSC_FALSE;   mask[3]  = PETSC_TRUE;
  mask[4]  = PETSC_FALSE;   mask[5]  = PETSC_FALSE;   mask[6]  = PETSC_FALSE;   mask[7]  = PETSC_FALSE;
  mask[8]  = PETSC_FALSE;   mask[9]  = PETSC_FALSE;   mask[10] = PETSC_FALSE;   mask[11] = PETSC_TRUE;
  mask[12] = PETSC_TRUE;   mask[13] = PETSC_FALSE;   mask[14] = PETSC_FALSE;   mask[15] = PETSC_FALSE;
  
  stencilWidth = 1;
  ndms = 4;
  for (d=0; d<ndms; d++) {
    if (d == 0) {
      dof0 = 0; dof1 = 1; dof2 = 0; /* (vertex) (face) (element) */
    }
    else if (d == 1) {
      dof0 = 0; dof1 = 1; dof2 = 0; /* (vertex) (face) (element) */
    }
    else {
      dof0 = 0; dof1 = 0; dof2 = 1; /* (vertex) (face) (element) */
    }
    
    PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, mx, my,PETSC_DECIDE, PETSC_DECIDE, dof0, dof1, dof2,
                          DMSTAG_STENCIL_STAR, stencilWidth, NULL,NULL, &dms[d]));
    PetscCall(DMSetFromOptions(dms[d]));
    PetscCall(DMSetUp(dms[d]));
  }
  
  PetscCall(FDPDECoupledCreateMatrix2(ndms,dms,mask,MATAIJ,&A));
  PetscCall(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO));
  PetscCall(MatView(A,PETSC_VIEWER_STDOUT_WORLD));
  
  /* check an error is thrown */
  /*MatSetValue(A,65,0,1.0,INSERT_VALUES);*/
  
  for (d=0; d<ndms; d++) {
    PetscCall(DMDestroy(&dms[d]));
  }
  PetscCall(MatDestroy(&A));
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main (int argc,char **argv)
{
    
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  
  PetscCall(test1(6,6));
  //PetscCall(test2(3,3));
  //PetscCall(test3(3,3));
  //PetscCall(test4(3,3));

  PetscCall(PetscFinalize());
  return 0;
}
