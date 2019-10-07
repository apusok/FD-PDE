static char help[] = "DMStagBCList test \n\n";
// run: ./tests/test_bclist

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
#include "../bc.h"

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
    ierr = DMStagBCListGetValues(bclist,'n','-',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
    
    for (k=0; k<n_bc; k++) {
      value_bc[k] = (PetscScalar)(k+1);
      type_bc[k] = BC_DIRICHLET;
    }
    
    /* Set edge BC values (-). No need to define the face, it is encoded in idx_bc[] */
    ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
    
    /* -------------------------------------------- */
    /* request edge BC values (|) on a face (north) */
    ierr = DMStagBCListGetValues(bclist,'n','|',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
    
    for (k=0; k<n_bc; k++) {
      value_bc[k] = (PetscScalar)(k+10);
      type_bc[k] = BC_NEUMANN;
    }
    
    /* Set edge BC values (-). No need to define the face, it is encoded in idx_bc[] */
    ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

    /* ---------------------------------------------- */
    /* request element BC values (o) on a face (east) */
    ierr = DMStagBCListGetValues(bclist,'e','o',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
    
    for (k=0; k<n_bc; k++) {
      value_bc[k] = (PetscScalar)(k+1);
      type_bc[k] = BC_DIRICHLET;
    }
    
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
  
  //ierr = test1(3,3);CHKERRQ(ierr);
  ierr = test2(3,3);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return(ierr);
}
