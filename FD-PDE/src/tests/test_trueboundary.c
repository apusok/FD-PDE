static char help[] = "Test: DMStagCellSize_2d, Coordinates of true boundary returned from DMStagBCListGetValues. \n\n";
// run: ./tests/test_trueboundary.app

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
#include "../dmstagbclist.h"
#include "../dmstag_utils.h"

PetscErrorCode test1(PetscInt nx,PetscInt ny)
{
  DM              dm;
  PetscInt        dof0,dof1,dof2,stencilWidth;
  DMStagBCList    bclist;
  PetscInt        i,j;
  PetscScalar     *dx,*dy;
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

  PetscInt is, js, nx_local, ny_local;
  ierr = DMStagCellSizeLocal_2d(dm, &is, &js, &nx_local, &ny_local, &dx,&dy);CHKERRQ(ierr);

  for (j=0; j<ny; j++) {
    for (i=0; i<nx; i++) {
      PetscPrintf(PETSC_COMM_WORLD," i = %d, j = %d, dx = %g, dy = %g \n",i,j,dx[i],dy[j]);
    }
  }

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
    ierr = DMStagBCListGetValues(bclist,'n','-',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
    
    for (k=0; k<n_bc; k++) {
      value_bc[k] = (PetscScalar)(k+1);
      type_bc[k] = BC_DIRICHLET;
    }

    PetscScalarView(2*n_bc,x_bc,PETSC_VIEWER_STDOUT_WORLD);
    
    /* Set edge BC values (-). No need to define the face, it is encoded in idx_bc[] */
    ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
    
    /* -------------------------------------------- */
    /* request edge BC values (|) on a face (north) */
    ierr = DMStagBCListGetValues(bclist,'n','|',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
    
    for (k=0; k<n_bc; k++) {
      value_bc[k] = (PetscScalar)(k+10);
      type_bc[k] = BC_NEUMANN;
    }

    PetscScalarView(2*n_bc,x_bc,PETSC_VIEWER_STDOUT_WORLD);
    /* Set edge BC values (|). No need to define the face, it is encoded in idx_bc[] */
    ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

    /* -------------------------------------------- */
    /* request element BC values (o) on a face (north) */
    ierr = DMStagBCListGetValues(bclist,'n','o',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
    
    for (k=0; k<n_bc; k++) {
      value_bc[k] = (PetscScalar)(k+1.99);
      type_bc[k] = BC_DIRICHLET;
    }

    PetscScalarView(2*n_bc,x_bc,PETSC_VIEWER_STDOUT_WORLD);
    
    /* Set edge BC values (o). No need to define the face, it is encoded in idx_bc[] */
    ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
    
    /* -------------------------------------------- */
    /* request edge BC values (-) on a face (south) */
    ierr = DMStagBCListGetValues(bclist,'s','-',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
    
    for (k=0; k<n_bc; k++) {
      value_bc[k] = (PetscScalar)(k+1.5);
      type_bc[k] = BC_DIRICHLET;
    }

    PetscScalarView(2*n_bc,x_bc,PETSC_VIEWER_STDOUT_WORLD);
    /* Set edge BC values (-). No need to define the face, it is encoded in idx_bc[] */
    ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
    
    /* -------------------------------------------- */
    /* request edge BC values (|) on a face (south) */
    ierr = DMStagBCListGetValues(bclist,'s','|',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
    
    for (k=0; k<n_bc; k++) {
      value_bc[k] = (PetscScalar)(k+10.5);
      type_bc[k] = BC_NEUMANN;
    }

    PetscScalarView(2*n_bc,x_bc,PETSC_VIEWER_STDOUT_WORLD);
    /* Set edge BC values (|). No need to define the face, it is encoded in idx_bc[] */
    ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

    /* -------------------------------------------- */
    /* request element BC values (o) on a face (south) */
    ierr = DMStagBCListGetValues(bclist,'s','o',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
    
    for (k=0; k<n_bc; k++) {
      value_bc[k] = (PetscScalar)(k+1.79);
      type_bc[k] = BC_DIRICHLET;
    }

    PetscScalarView(2*n_bc,x_bc,PETSC_VIEWER_STDOUT_WORLD);
    
    /* Set edge BC values (o). No need to define the face, it is encoded in idx_bc[] */
    ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
    
    /* -------------------------------------------- */
    /* request edge BC values (-) on a face (west) */
    ierr = DMStagBCListGetValues(bclist,'w','-',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
    
    for (k=0; k<n_bc; k++) {
      value_bc[k] = (PetscScalar)(k+1.25);
      type_bc[k] = BC_DIRICHLET;
    }

    PetscScalarView(2*n_bc,x_bc,PETSC_VIEWER_STDOUT_WORLD);
    /* Set edge BC values (-). No need to define the face, it is encoded in idx_bc[] */
    ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
    
    /* -------------------------------------------- */
    /* request edge BC values (|) on a face (west) */
    ierr = DMStagBCListGetValues(bclist,'w','|',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
    
    for (k=0; k<n_bc; k++) {
      value_bc[k] = (PetscScalar)(k+10.25);
      type_bc[k] = BC_NEUMANN;
    }

    PetscScalarView(2*n_bc,x_bc,PETSC_VIEWER_STDOUT_WORLD);
    /* Set edge BC values (|). No need to define the face, it is encoded in idx_bc[] */
    ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

    /* -------------------------------------------- */
    /* request element BC values (o) on a face (west) */
    ierr = DMStagBCListGetValues(bclist,'w','o',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
    
    for (k=0; k<n_bc; k++) {
      value_bc[k] = (PetscScalar)(k+1.59);
      type_bc[k] = BC_DIRICHLET;
    }

    PetscScalarView(2*n_bc,x_bc,PETSC_VIEWER_STDOUT_WORLD);
    
    /* Set edge BC values (o). No need to define the face, it is encoded in idx_bc[] */
    ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
    
    /* -------------------------------------------- */
    /* request edge BC values (-) on a face (east) */
    ierr = DMStagBCListGetValues(bclist,'e','-',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
    
    for (k=0; k<n_bc; k++) {
      value_bc[k] = (PetscScalar)(k+1.75);
      type_bc[k] = BC_DIRICHLET;
    }

    PetscScalarView(2*n_bc,x_bc,PETSC_VIEWER_STDOUT_WORLD);
    /* Set edge BC values (-). No need to define the face, it is encoded in idx_bc[] */
    ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
    
    /* -------------------------------------------- */
    /* request edge BC values (|) on a face (east) */
    ierr = DMStagBCListGetValues(bclist,'e','|',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
    
    for (k=0; k<n_bc; k++) {
      value_bc[k] = (PetscScalar)(k+10.75);
      type_bc[k] = BC_NEUMANN;
    }

    PetscScalarView(2*n_bc,x_bc,PETSC_VIEWER_STDOUT_WORLD);
    /* Set edge BC values (-). No need to define the face, it is encoded in idx_bc[] */
    ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

    /* ---------------------------------------------- */
    /* request element BC values (o) on a face (east) */
    ierr = DMStagBCListGetValues(bclist,'e','o',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
    
    for (k=0; k<n_bc; k++) {
      value_bc[k] = (PetscScalar)(k+1.39);
      type_bc[k] = BC_DIRICHLET;
    }

    PetscScalarView(2*n_bc,x_bc,PETSC_VIEWER_STDOUT_WORLD);
    /* Set edge BC values (-). No need to define the face, it is encoded in idx_bc[] */
    ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
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
    
  ierr = PetscInitialize(&argc,&argv,(char*)0,help); if (ierr) return(ierr);
  
  ierr = test1(3,4);CHKERRQ(ierr);
  ierr = test2(3,4);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return(ierr);
}
