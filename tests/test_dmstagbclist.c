static char help[] = "DMStagBCList test \n\n";
// run: ./tests/test_dmstagbclist.app

#define DOWN_LEFT  DMSTAG_DOWN_LEFT
#define DOWN       DMSTAG_DOWN
#define DOWN_RIGHT DMSTAG_DOWN_RIGHT
#define LEFT       DMSTAG_LEFT
#define ELEMENT    DMSTAG_ELEMENT
#define RIGHT      DMSTAG_RIGHT
#define UP_LEFT    DMSTAG_UP_LEFT
#define UP         DMSTAG_UP
#define UP_RIGHT   DMSTAG_UP_RIGHT

#include "petsc.h"
#include "../src/dmstagbclist.h"

PetscErrorCode test1(PetscInt nx,PetscInt ny)
{
  DM              dm;
  PetscInt        dof0,dof1,dof2,stencilWidth;
  DMStagBCList    bclist;
  PetscErrorCode  ierr;
  
  dof0 = 0; dof1 = 1; dof2 = 1; /* (vertex) (face) (element) */
  stencilWidth = 1;
  
  ierr = DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, nx, ny,
                        PETSC_DECIDE, PETSC_DECIDE, dof0, dof1, dof2,
                        DMSTAG_STENCIL_BOX, stencilWidth, NULL,NULL, &dm);CHKERRQ(ierr);
  ierr = DMStagSetCoordinateDMType(dm,DMPRODUCT);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMSetUp(dm);CHKERRQ(ierr);
  
  ierr = DMStagSetUniformCoordinatesProduct(dm,0.0,1.0,0.0,1.0,0.0,0.0);CHKERRQ(ierr);

  ierr = DMStagBCListCreate(dm,&bclist);CHKERRQ(ierr);
  ierr = DMStagBCListView(bclist);CHKERRQ(ierr);
  ierr = DMStagBCListDestroy(&bclist);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode test2(PetscInt nx,PetscInt ny)
{
  DM              dm;
  PetscInt        dof0,dof1,dof2,stencilWidth;
  DMStagBCList    bclist;
  PetscErrorCode  ierr;
  
  dof0 = 0; dof1 = 1; dof2 = 1; /* (vertex) (face) (element) */
  stencilWidth = 1;
  
  ierr = DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, nx, ny,
                        PETSC_DECIDE, PETSC_DECIDE, dof0, dof1, dof2,
                        DMSTAG_STENCIL_BOX, stencilWidth, NULL,NULL, &dm);CHKERRQ(ierr);
  ierr = DMStagSetCoordinateDMType(dm,DMPRODUCT);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMSetUp(dm);CHKERRQ(ierr);
  
  ierr = DMStagSetUniformCoordinatesProduct(dm,0.0,1.0,0.0,1.0,0.0,0.0);CHKERRQ(ierr);
  
  ierr = DMStagBCListCreate(dm,&bclist);CHKERRQ(ierr);
  
  {
    PetscInt    k,n_bc,*idx_bc;
    PetscScalar *x_bc,*value_bc;
    BCType      *type_bc;

    /* -------------------------------------------- */
    /* request edge BC values (-) on a face (north) */
    ierr = DMStagBCListGetValues(bclist,'n','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
    
    for (k=0; k<n_bc; k++) {
      value_bc[k] = (PetscScalar)(k+1);
      type_bc[k] = BC_DIRICHLET;
    }
    
    /* Set edge BC values (-). No need to define the face, it is encoded in idx_bc[] */
    ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
    
    /* -------------------------------------------- */
    /* request edge BC values (|) on a face (north) */
    ierr = DMStagBCListGetValues(bclist,'n','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
    
    for (k=0; k<n_bc; k++) {
      value_bc[k] = (PetscScalar)(k+10);
      type_bc[k] = BC_NEUMANN;
    }
    
    /* Set edge BC values (-). No need to define the face, it is encoded in idx_bc[] */
    ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

    /* ---------------------------------------------- */
    /* request element BC values (o) on a face (east) */
    ierr = DMStagBCListGetValues(bclist,'e','o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
    
    for (k=0; k<n_bc; k++) {
      value_bc[k] = (PetscScalar)(k+1);
      type_bc[k] = BC_DIRICHLET;
    }
    
    /* Set edge BC values (-). No need to define the face, it is encoded in idx_bc[] */
    ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  }
  
  
  ierr = DMStagBCListView(bclist);CHKERRQ(ierr);
  ierr = DMStagBCListDestroy(&bclist);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode test3(PetscInt nx,PetscInt ny)
{
  DM              dm;
  PetscInt        dof0,dof1,dof2,stencilWidth;
  DMStagBCList    bclist;
  PetscErrorCode  ierr;
  
  dof0 = 0; dof1 = 1; dof2 = 1; /* (vertex) (face) (element) */
  stencilWidth = 1;
  
  ierr = DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, nx, ny,
                        PETSC_DECIDE, PETSC_DECIDE, dof0, dof1, dof2,
                        DMSTAG_STENCIL_BOX, stencilWidth, NULL,NULL, &dm);CHKERRQ(ierr);
  ierr = DMStagSetCoordinateDMType(dm,DMPRODUCT);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMSetUp(dm);CHKERRQ(ierr);
  
  ierr = DMStagSetUniformCoordinatesProduct(dm,0.0,1.0,0.0,1.0,0.0,0.0);CHKERRQ(ierr);
  
  ierr = DMStagBCListCreate(dm,&bclist);CHKERRQ(ierr);
  
  ierr = DMStagBCListPinValue(bclist,'|',0,33.0);CHKERRQ(ierr);
  fflush(stdout);
  ierr = MPI_Barrier(PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = DMStagBCListPinValue(bclist,'-',0,66.0);CHKERRQ(ierr);
  fflush(stdout);
  ierr = MPI_Barrier(PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = DMStagBCListPinValue(bclist,'o',0,99.0);CHKERRQ(ierr);
  fflush(stdout);
  ierr = MPI_Barrier(PETSC_COMM_WORLD);CHKERRQ(ierr);
 
  ierr = DMStagBCListView(bclist);CHKERRQ(ierr);
  ierr = DMStagBCListDestroy(&bclist);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode test4(PetscInt nx,PetscInt ny)
{
  DM              dm;
  PetscInt        dof0,dof1,dof2,stencilWidth;
  DMStagBCList    bclist;
  PetscErrorCode  ierr;
  
  dof0 = 0; dof1 = 1; dof2 = 1; /* (vertex) (face) (element) */
  stencilWidth = 1;
  
  ierr = DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, nx, ny,
                        PETSC_DECIDE, PETSC_DECIDE, dof0, dof1, dof2,
                        DMSTAG_STENCIL_BOX, stencilWidth, NULL,NULL, &dm);CHKERRQ(ierr);
  ierr = DMStagSetCoordinateDMType(dm,DMPRODUCT);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMSetUp(dm);CHKERRQ(ierr);
  
  ierr = DMStagSetUniformCoordinatesProduct(dm,0.0,1.0,0.0,1.0,0.0,0.0);CHKERRQ(ierr);
  
  ierr = DMStagBCListCreate(dm,&bclist);CHKERRQ(ierr);
  
  ierr = DMStagBCListPinCornerValue(bclist,DMSTAG_DOWN_LEFT,'o',0,33.0);CHKERRQ(ierr);
  fflush(stdout);
  ierr = MPI_Barrier(PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = DMStagBCListPinCornerValue(bclist,DMSTAG_DOWN_RIGHT,'o',0,66.0);CHKERRQ(ierr);
  fflush(stdout);
  ierr = MPI_Barrier(PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = DMStagBCListPinCornerValue(bclist,DMSTAG_UP_LEFT,'o',0,99.0);CHKERRQ(ierr);
  fflush(stdout);
  ierr = MPI_Barrier(PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = DMStagBCListPinCornerValue(bclist,DMSTAG_UP_RIGHT,'o',0,333.0);CHKERRQ(ierr);
  fflush(stdout);
  ierr = MPI_Barrier(PETSC_COMM_WORLD);CHKERRQ(ierr);
  
  
  ierr = DMStagBCListView(bclist);CHKERRQ(ierr);
  ierr = DMStagBCListDestroy(&bclist);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode test5(PetscInt nx,PetscInt ny)
{
  DM              dm;
  PetscInt        dof0,dof1,dof2,stencilWidth;
  DMStagBCList    bclist;
  PetscErrorCode  ierr;
  
  dof0 = 0; dof1 = 2; dof2 = 2; /* (vertex) (face) (element) */
  stencilWidth = 1;
  
  ierr = DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, nx, ny,
                        PETSC_DECIDE, PETSC_DECIDE, dof0, dof1, dof2,
                        DMSTAG_STENCIL_BOX, stencilWidth, NULL,NULL, &dm);CHKERRQ(ierr);
  ierr = DMStagSetCoordinateDMType(dm,DMPRODUCT);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMSetUp(dm);CHKERRQ(ierr);
  
  ierr = DMStagSetUniformCoordinatesProduct(dm,0.0,1.0,0.0,1.0,0.0,0.0);CHKERRQ(ierr);
  
  ierr = DMStagBCListCreate(dm,&bclist);CHKERRQ(ierr);
  
  {
    PetscInt    k,n_bc,*idx_bc;
    PetscScalar *x_bc,*value_bc,ct;
    BCType      *type_bc;

    ct = 0.0;
    
    /* request edge BC values (-) on a face (west) - for 2 DOFs */
    ierr = DMStagBCListGetValues(bclist,'w','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
    for (k=0; k<n_bc; k++) {
      ct += 1.0;
      value_bc[k] = ct;
      type_bc[k] = BC_DIRICHLET;
    }
    ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

    ierr = DMStagBCListGetValues(bclist,'w','-',1,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
    for (k=0; k<n_bc; k++) {
      ct += 1.0;
      value_bc[k] = ct;
      type_bc[k] = BC_NEUMANN;
    }
    ierr = DMStagBCListInsertValues(bclist,'-',1,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
    
    /* request edge BC values (|) on a face (n) - for 2 DOFs */
    ierr = DMStagBCListGetValues(bclist,'n','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
    for (k=0; k<n_bc; k++) {
      ct += 1.0;
      value_bc[k] = ct;
      type_bc[k] = BC_DIRICHLET_STAG;
    }
    ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

    ierr = DMStagBCListGetValues(bclist,'n','|',1,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
    for (k=0; k<n_bc; k++) {
      ct += 1.0;
      value_bc[k] = ct;
      type_bc[k] = BC_NEUMANN;
    }
    ierr = DMStagBCListInsertValues(bclist,'|',1,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

    /* request element BC values (o) on a face (south) - for 2 DOFs */
    ierr = DMStagBCListGetValues(bclist,'s','o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
    for (k=0; k<n_bc; k++) {
      ct += 1.0;
      value_bc[k] = ct;
      type_bc[k] = BC_DIRICHLET;
    }
    ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

    ierr = DMStagBCListGetValues(bclist,'s','o',1,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
    for (k=0; k<n_bc; k++) {
      ct += 1.0;
      value_bc[k] = ct;
      type_bc[k] = BC_NEUMANN;
    }
    ierr = DMStagBCListInsertValues(bclist,'o',1,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  }

  ierr = DMStagBCListView(bclist);CHKERRQ(ierr);
  ierr = DMStagBCListDestroy(&bclist);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main (int argc,char **argv)
{
  PetscErrorCode  ierr;
  PetscInt tid = 2;
  PetscInt m = 3,n = 3;
  
  ierr = PetscInitialize(&argc,&argv,(char*)0,help); if (ierr) return(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-tid",&tid,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  switch (tid) {
    case 1:
      ierr = test1(m,n);CHKERRQ(ierr);
      break;
    case 2:
      ierr = test2(m,n);CHKERRQ(ierr);
      break;
    case 3:
      ierr = test3(m,n);CHKERRQ(ierr); /* pin-point */
      break;
    case 4:
      ierr = test4(m,n);CHKERRQ(ierr); /* pin-point corner specification */
      break;
    case 5:
      ierr = test5(m,n);CHKERRQ(ierr); /* multiple dofs */
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Unknown test requested. Valid tests are -tid = {1,2,3,4,5}");
      break;
  }

  ierr = PetscFinalize();
  return(ierr);
}
