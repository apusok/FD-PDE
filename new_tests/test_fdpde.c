static char help[] = "FD-PDE test \n\n";
// run: ./tests/test_fdpde -lov_view

#include "../new_src/fdpde_stokes.h"

// test0 - create/destroy
PetscErrorCode test0(PetscInt nx,PetscInt nz)
{
  FDPDE           fd;
  PetscFunctionBeginUser;

  PetscCall(FDPDECreate(PETSC_COMM_WORLD,nx,nz,0.0,1.0,0.0,1.0,FDPDE_STOKES,&fd));
  PetscCall(FDPDESetUp(fd));
  PetscCall(FDPDEView(fd)); 
  PetscCall(FDPDEDestroy(&fd));
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main (int argc,char **argv)
{    
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(test0(4,5));
  PetscCall(PetscFinalize());
  return 0;
}
