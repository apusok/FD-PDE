static char help[] = "DMStagBCList test \n\n";
// run: ./test_dmstagbclist.sh

#define DOWN_LEFT  DMSTAG_DOWN_LEFT
#define DOWN       DMSTAG_DOWN
#define DOWN_RIGHT DMSTAG_DOWN_RIGHT
#define LEFT       DMSTAG_LEFT
#define ELEMENT    DMSTAG_ELEMENT
#define RIGHT      DMSTAG_RIGHT
#define UP_LEFT    DMSTAG_UP_LEFT
#define UP         DMSTAG_UP
#define UP_RIGHT   DMSTAG_UP_RIGHT

#include "../src/fdpde_dmstag.h"

PetscErrorCode test1(PetscInt nx,PetscInt ny)
{
  DM              dm;
  PetscInt        dof0,dof1,dof2,stencilWidth;
  DMStagBCList    bclist;
  PetscFunctionBeginUser;
  
  dof0 = 0; dof1 = 1; dof2 = 1; /* (vertex) (face) (element) */
  stencilWidth = 1;
  
  PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, nx, ny,PETSC_DECIDE, PETSC_DECIDE, dof0, dof1, dof2,
                        DMSTAG_STENCIL_BOX, stencilWidth, NULL,NULL, &dm));
  PetscCall(DMStagSetCoordinateDMType(dm,DMPRODUCT));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm,0.0,1.0,0.0,1.0,0.0,0.0));

  PetscCall(DMStagBCListCreate(dm,&bclist));
  PetscCall(DMStagBCListView(bclist));
  PetscCall(DMStagBCListDestroy(&bclist));

  PetscCall(DMDestroy(&dm));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode test2(PetscInt nx,PetscInt ny)
{
  DM              dm;
  PetscInt        dof0,dof1,dof2,stencilWidth;
  DMStagBCList    bclist;
  PetscFunctionBeginUser;

  dof0 = 0; dof1 = 1; dof2 = 1; /* (vertex) (face) (element) */
  stencilWidth = 1;
  
  PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, nx, ny,PETSC_DECIDE, PETSC_DECIDE, dof0, dof1, dof2,
                        DMSTAG_STENCIL_BOX, stencilWidth, NULL,NULL, &dm));
  PetscCall(DMStagSetCoordinateDMType(dm,DMPRODUCT));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  
  PetscCall(DMStagSetUniformCoordinatesProduct(dm,0.0,1.0,0.0,1.0,0.0,0.0));
  
  PetscCall(DMStagBCListCreate(dm,&bclist));
  
  {
    PetscInt    k,n_bc,*idx_bc;
    PetscScalar *x_bc,*value_bc;
    BCType      *type_bc;

    /* -------------------------------------------- */
    /* request edge BC values (-) on a face (north) */
    PetscCall(DMStagBCListGetValues(bclist,'n','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
    
    for (k=0; k<n_bc; k++) {
      value_bc[k] = (PetscScalar)(k+1);
      type_bc[k] = BC_DIRICHLET;
    }
    
    /* Set edge BC values (-). No need to define the face, it is encoded in idx_bc[] */
    PetscCall(DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
    
    /* -------------------------------------------- */
    /* request edge BC values (|) on a face (north) */
    PetscCall(DMStagBCListGetValues(bclist,'n','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
    
    for (k=0; k<n_bc; k++) {
      value_bc[k] = (PetscScalar)(k+10);
      type_bc[k] = BC_NEUMANN;
    }
    
    /* Set edge BC values (-). No need to define the face, it is encoded in idx_bc[] */
    PetscCall(DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

    /* ---------------------------------------------- */
    /* request element BC values (o) on a face (east) */
    PetscCall(DMStagBCListGetValues(bclist,'e','o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
    
    for (k=0; k<n_bc; k++) {
      value_bc[k] = (PetscScalar)(k+1);
      type_bc[k] = BC_DIRICHLET;
    }
    
    /* Set edge BC values (-). No need to define the face, it is encoded in idx_bc[] */
    PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  }
  
  PetscCall(DMStagBCListView(bclist));
  PetscCall(DMStagBCListDestroy(&bclist));
  PetscCall(DMDestroy(&dm));
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode test3(PetscInt nx,PetscInt ny)
{
  DM              dm;
  PetscInt        dof0,dof1,dof2,stencilWidth;
  DMStagBCList    bclist;
  PetscFunctionBeginUser;

  dof0 = 0; dof1 = 1; dof2 = 1; /* (vertex) (face) (element) */
  stencilWidth = 1;
  
  PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, nx, ny,PETSC_DECIDE, PETSC_DECIDE, dof0, dof1, dof2,
                        DMSTAG_STENCIL_BOX, stencilWidth, NULL,NULL, &dm));
  PetscCall(DMStagSetCoordinateDMType(dm,DMPRODUCT));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  
  PetscCall(DMStagSetUniformCoordinatesProduct(dm,0.0,1.0,0.0,1.0,0.0,0.0));
  
  PetscCall(DMStagBCListCreate(dm,&bclist));
  
  PetscCall(DMStagBCListPinValue(bclist,'|',0,33.0));
  fflush(stdout);
  PetscCall(MPI_Barrier(PETSC_COMM_WORLD));
  PetscCall(DMStagBCListPinValue(bclist,'-',0,66.0));
  fflush(stdout);
  PetscCall(MPI_Barrier(PETSC_COMM_WORLD));
  PetscCall(DMStagBCListPinValue(bclist,'o',0,99.0));
  fflush(stdout);
  PetscCall(MPI_Barrier(PETSC_COMM_WORLD));
 
  PetscCall(DMStagBCListView(bclist));
  PetscCall(DMStagBCListDestroy(&bclist));
  PetscCall(DMDestroy(&dm));
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode test4(PetscInt nx,PetscInt ny)
{
  DM              dm;
  PetscInt        dof0,dof1,dof2,stencilWidth;
  DMStagBCList    bclist;
  PetscFunctionBeginUser;

  dof0 = 0; dof1 = 1; dof2 = 1; /* (vertex) (face) (element) */
  stencilWidth = 1;
  
  PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, nx, ny,PETSC_DECIDE, PETSC_DECIDE, dof0, dof1, dof2,
                        DMSTAG_STENCIL_BOX, stencilWidth, NULL,NULL, &dm));
  PetscCall(DMStagSetCoordinateDMType(dm,DMPRODUCT));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  
  PetscCall(DMStagSetUniformCoordinatesProduct(dm,0.0,1.0,0.0,1.0,0.0,0.0));
  
  PetscCall(DMStagBCListCreate(dm,&bclist));
  
  PetscCall(DMStagBCListPinCornerValue(bclist,DMSTAG_DOWN_LEFT,'o',0,33.0));
  fflush(stdout);
  PetscCall(MPI_Barrier(PETSC_COMM_WORLD));
  PetscCall(DMStagBCListPinCornerValue(bclist,DMSTAG_DOWN_RIGHT,'o',0,66.0));
  fflush(stdout);
  PetscCall(MPI_Barrier(PETSC_COMM_WORLD));
  PetscCall(DMStagBCListPinCornerValue(bclist,DMSTAG_UP_LEFT,'o',0,99.0));
  fflush(stdout);
  PetscCall(MPI_Barrier(PETSC_COMM_WORLD));
  PetscCall(DMStagBCListPinCornerValue(bclist,DMSTAG_UP_RIGHT,'o',0,333.0));
  fflush(stdout);
  PetscCall(MPI_Barrier(PETSC_COMM_WORLD));
  
  
  PetscCall(DMStagBCListView(bclist));
  PetscCall(DMStagBCListDestroy(&bclist));
  PetscCall(DMDestroy(&dm));
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode test5(PetscInt nx,PetscInt ny)
{
  DM              dm;
  PetscInt        dof0,dof1,dof2,stencilWidth;
  DMStagBCList    bclist;
  PetscFunctionBeginUser;

  dof0 = 0; dof1 = 2; dof2 = 2; /* (vertex) (face) (element) */
  stencilWidth = 1;
  
  PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, nx, ny,PETSC_DECIDE, PETSC_DECIDE, dof0, dof1, dof2,
                        DMSTAG_STENCIL_BOX, stencilWidth, NULL,NULL, &dm));
  PetscCall(DMStagSetCoordinateDMType(dm,DMPRODUCT));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  
  PetscCall(DMStagSetUniformCoordinatesProduct(dm,0.0,1.0,0.0,1.0,0.0,0.0));
  
  PetscCall(DMStagBCListCreate(dm,&bclist));
  
  {
    PetscInt    k,n_bc,*idx_bc;
    PetscScalar *x_bc,*value_bc,ct;
    BCType      *type_bc;

    ct = 0.0;
    
    /* request edge BC values (-) on a face (west) - for 2 DOFs */
    PetscCall(DMStagBCListGetValues(bclist,'w','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
    for (k=0; k<n_bc; k++) {
      ct += 1.0;
      value_bc[k] = ct;
      type_bc[k] = BC_DIRICHLET;
    }
    PetscCall(DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

    PetscCall(DMStagBCListGetValues(bclist,'w','-',1,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
    for (k=0; k<n_bc; k++) {
      ct += 1.0;
      value_bc[k] = ct;
      type_bc[k] = BC_NEUMANN;
    }
    PetscCall(DMStagBCListInsertValues(bclist,'-',1,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
    
    /* request edge BC values (|) on a face (n) - for 2 DOFs */
    PetscCall(DMStagBCListGetValues(bclist,'n','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
    for (k=0; k<n_bc; k++) {
      ct += 1.0;
      value_bc[k] = ct;
      type_bc[k] = BC_DIRICHLET_STAG;
    }
    PetscCall(DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

    PetscCall(DMStagBCListGetValues(bclist,'n','|',1,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
    for (k=0; k<n_bc; k++) {
      ct += 1.0;
      value_bc[k] = ct;
      type_bc[k] = BC_NEUMANN;
    }
    PetscCall(DMStagBCListInsertValues(bclist,'|',1,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

    /* request element BC values (o) on a face (south) - for 2 DOFs */
    PetscCall(DMStagBCListGetValues(bclist,'s','o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
    for (k=0; k<n_bc; k++) {
      ct += 1.0;
      value_bc[k] = ct;
      type_bc[k] = BC_DIRICHLET;
    }
    PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

    PetscCall(DMStagBCListGetValues(bclist,'s','o',1,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
    for (k=0; k<n_bc; k++) {
      ct += 1.0;
      value_bc[k] = ct;
      type_bc[k] = BC_NEUMANN;
    }
    PetscCall(DMStagBCListInsertValues(bclist,'o',1,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  }

  PetscCall(DMStagBCListView(bclist));
  PetscCall(DMStagBCListDestroy(&bclist));
  PetscCall(DMDestroy(&dm));
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main (int argc,char **argv)
{
  PetscInt tid = 2;
  PetscInt m = 3,n = 3;
  
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-tid",&tid,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  switch (tid) {
    case 1:
      PetscCall(test1(m,n));
      break;
    case 2:
      PetscCall(test2(m,n));
      break;
    case 3:
      PetscCall(test3(m,n)); /* pin-point */
      break;
    case 4:
      PetscCall(test4(m,n)); /* pin-point corner specification */
      break;
    case 5:
      PetscCall(test5(m,n)); /* multiple dofs */
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Unknown test requested. Valid tests are -tid = {1,2,3,4,5}");
      break;
  }

  PetscCall(PetscFinalize());
  return 0;
}
