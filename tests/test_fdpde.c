static char help[] = "FD-PDE test \n\n";
// run: ./tests/test_fd.app

#include "petsc.h"
#include "../src/fdpde.h"

// test0 - create/destroy
PetscErrorCode test0(PetscInt nx,PetscInt nz)
{
  FDPDE           fd;
  PetscErrorCode  ierr;
  
  ierr = FDPDECreate(PETSC_COMM_WORLD,nx,nz,0.0,1.0,0.0,1.0,FDPDE_STOKES,&fd);CHKERRQ(ierr);
  ierr = FDPDESetUp(fd);CHKERRQ(ierr);
  ierr = FDPDEView(fd); CHKERRQ(ierr);
  ierr = FDPDEDestroy(&fd);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

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
