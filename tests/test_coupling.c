static char help[] = "DMStagCoupling test \n\n";

#include "petsc.h"
#include "../src/composite_prealloc_utils.h"


PetscErrorCode test1(PetscInt mx,PetscInt my)
{
  DM              dms[20];
  PetscInt        dof0,dof1,dof2,stencilWidth,ndms,d;
  Mat             A;
  PetscErrorCode  ierr;
  
  dof0 = 0; dof1 = 0; dof2 = 1; /* (vertex) (face) (element) */
  stencilWidth = 1;
  ndms = 2;
  for (d=0; d<ndms; d++) {
    ierr = DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, mx, my,
                          PETSC_DECIDE, PETSC_DECIDE, dof0, dof1, dof2,
                          DMSTAG_STENCIL_BOX, stencilWidth, NULL,NULL, &dms[d]);CHKERRQ(ierr);
    ierr = DMStagSetCoordinateDMType(dms[d],DMPRODUCT);CHKERRQ(ierr);
    ierr = DMSetFromOptions(dms[d]);CHKERRQ(ierr);
    ierr = DMSetUp(dms[d]);CHKERRQ(ierr);
    
    ierr = DMStagSetUniformCoordinatesProduct(dms[d],0.0,1.0,0.0,1.0,0.0,0.0);CHKERRQ(ierr);
  }
  
  ierr = FDPDECoupledCreateMatrix(ndms,dms,MATAIJ,&A);CHKERRQ(ierr);
  //ierr = FDPDECoupledCreateMatrix(1,&dms[0],MATAIJ,&A);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr);
  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  
  for (d=0; d<ndms; d++) {
    ierr = DMDestroy(&dms[d]);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode test2(PetscInt mx,PetscInt my)
{
  DM              dms[20];
  PetscInt        dof0,dof1,dof2,stencilWidth,ndms,d;
  Mat             A;
  PetscErrorCode  ierr;
  
  dof0 = 0; dof1 = 0; dof2 = 1; /* (vertex) (face) (element) */
  stencilWidth = 1;
  ndms = 2;
  for (d=0; d<ndms; d++) {
    ierr = DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, mx, my,
                          PETSC_DECIDE, PETSC_DECIDE, dof0, dof1, dof2,
                          DMSTAG_STENCIL_STAR, stencilWidth, NULL,NULL, &dms[d]);CHKERRQ(ierr);
    ierr = DMStagSetCoordinateDMType(dms[d],DMPRODUCT);CHKERRQ(ierr);
    ierr = DMSetFromOptions(dms[d]);CHKERRQ(ierr);
    ierr = DMSetUp(dms[d]);CHKERRQ(ierr);
    
    ierr = DMStagSetUniformCoordinatesProduct(dms[d],0.0,1.0,0.0,1.0,0.0,0.0);CHKERRQ(ierr);
  }
  
  ierr = FDPDECoupledCreateMatrix(ndms,dms,MATAIJ,&A);CHKERRQ(ierr);
  //ierr = FDPDECoupledCreateMatrix(1,&dms[0],MATAIJ,&A);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr);
  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  
  for (d=0; d<ndms; d++) {
    ierr = DMDestroy(&dms[d]);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode test3(PetscInt mx,PetscInt my)
{
  DM              dms[20];
  PetscInt        dof0,dof1,dof2,stencilWidth,ndms,d;
  Mat             A;
  PetscErrorCode  ierr;
  
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

    ierr = DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, mx, my,
                          PETSC_DECIDE, PETSC_DECIDE, dof0, dof1, dof2,
                          DMSTAG_STENCIL_STAR, stencilWidth, NULL,NULL, &dms[d]);CHKERRQ(ierr);
    ierr = DMSetFromOptions(dms[d]);CHKERRQ(ierr);
    ierr = DMSetUp(dms[d]);CHKERRQ(ierr);
  }
  
  ierr = FDPDECoupledCreateMatrix(ndms,dms,MATAIJ,&A);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr);
  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  
  for (d=0; d<ndms; d++) {
    ierr = DMDestroy(&dms[d]);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode test4(PetscInt mx,PetscInt my)
{
  DM              dms[20];
  PetscInt        dof0,dof1,dof2,stencilWidth,ndms,d;
  Mat             A;
  PetscBool       mask[16];
  PetscErrorCode  ierr;
  
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
    
    ierr = DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, mx, my,
                          PETSC_DECIDE, PETSC_DECIDE, dof0, dof1, dof2,
                          DMSTAG_STENCIL_STAR, stencilWidth, NULL,NULL, &dms[d]);CHKERRQ(ierr);
    ierr = DMSetFromOptions(dms[d]);CHKERRQ(ierr);
    ierr = DMSetUp(dms[d]);CHKERRQ(ierr);
  }
  
  ierr = FDPDECoupledCreateMatrix2(ndms,dms,mask,MATAIJ,&A);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr);
  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  
  /* check an error is thrown */
  /*MatSetValue(A,65,0,1.0,INSERT_VALUES);*/
  
  for (d=0; d<ndms; d++) {
    ierr = DMDestroy(&dms[d]);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}



#undef __FUNCT__
#define __FUNCT__ "main"
int main (int argc,char **argv)
{
  PetscErrorCode  ierr;
    
  ierr = PetscInitialize(&argc,&argv,(char*)0,help); if (ierr) return(ierr);
  
  ierr = test1(6,6);CHKERRQ(ierr);
  //ierr = test2(3,3);CHKERRQ(ierr);
  //ierr = test3(3,3);CHKERRQ(ierr);
  //ierr = test4(3,3);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return(ierr);
}
