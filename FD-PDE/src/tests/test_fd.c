static char help[] = "FD-PDE test \n\n";

#include "petsc.h"
#include "../fd.h"

// test0 - create/destroy
PetscErrorCode test0(PetscInt nx,PetscInt nz)
{
  FD             fd;
  PetscErrorCode  ierr;
  
  ierr = FDCreate(PETSC_COMM_WORLD,nx,nz,0.0,1.0,0.0,1.0,STOKES,&fd);CHKERRQ(ierr);
  ierr = FDSetUp(fd);CHKERRQ(ierr);
  ierr = FDView(fd); CHKERRQ(ierr);
  ierr = FDDestroy(&fd);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

// test1 - 

#undef __FUNCT__
#define __FUNCT__ "main"
int main (int argc,char **argv)
{
  PetscErrorCode  ierr;
    
  ierr = PetscInitialize(&argc,&argv,(char*)0,help); if (ierr) return(ierr);
  ierr = test0(4,5);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return(ierr);
}
