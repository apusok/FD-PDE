static char help[] = "FD-PDE ENTHALPY test \n\n";
// run: ./tests/test_enthalpy.app -log_view

#include "petsc.h"
#include "../fdpde_enthalpy.h"

// ---------------------------------------
// test0 - create/destroy
// ---------------------------------------
static PetscErrorCode FormCoefficient(FDPDE fd, DM dm, Vec x, DM dmcoeff, Vec coeff, void *ctx) { return 0; }
static PetscErrorCode FormBCList(DM dm, Vec x, DMStagBCList bclist, void *ctx) { return 0; }
static PetscErrorCode Liquidus(FDPDE fd, DM dm, Vec x, DM dmcoeff, Vec xcoeff, DM dmcomp, Vec xCF, void *ctx) { return 0; }
static PetscErrorCode Solidus(FDPDE fd, DM dm, Vec x, DM dmcoeff, Vec xcoeff, DM dmcomp, Vec xCS, void *ctx) { return 0; }

PetscErrorCode test0(PetscInt nx,PetscInt nz)
{
  FDPDE           fd;
  PetscErrorCode  ierr;
  
  ierr = FDPDECreate(PETSC_COMM_WORLD,nx,nz,0.0,1.0,0.0,1.0,FDPDE_ENTHALPY,&fd);CHKERRQ(ierr);
  ierr = FDPDEEnthalpySetNumberComponentsPhaseDiagram(fd,2);CHKERRQ(ierr);
  ierr = FDPDESetUp(fd);CHKERRQ(ierr);

  ierr = FDPDEEnthalpySetEnergyPrimaryVariable(fd,'H');CHKERRQ(ierr);
  ierr = FDPDEEnthalpySetEnergyPrimaryVariable(fd,'T');CHKERRQ(ierr);

  ierr = FDPDESetFunctionBCList(fd,FormBCList,NULL,NULL); CHKERRQ(ierr);
  ierr = FDPDESetFunctionCoefficient(fd,FormCoefficient,NULL,NULL); CHKERRQ(ierr);

  ierr = FDPDEEnthalpySetAdvectSchemeType(fd,ADV_FROMM);CHKERRQ(ierr);
  ierr = FDPDEEnthalpySetTimeStepSchemeType(fd,TS_CRANK_NICHOLSON);CHKERRQ(ierr);
  ierr = FDPDEEnthalpySetTimestep(fd,0.01); CHKERRQ(ierr);

  ierr = FDPDEEnthalpySetFunctionsPhaseDiagram(fd,Liquidus,Solidus,NULL);CHKERRQ(ierr);

  ierr = FDPDEView(fd); CHKERRQ(ierr);
  ierr = FDPDEDestroy(&fd);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

// ---------------------------------------
// MAIN
// ---------------------------------------
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