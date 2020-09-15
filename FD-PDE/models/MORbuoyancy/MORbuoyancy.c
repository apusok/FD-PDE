// ---------------------------------------
// Mid-ocean ridge model solving for conservation of mass, momentum, energy and composition using the Enthalpy Method
// Model after Katz 2008, 2010.
// run: ./MORbuoyancy.app -options_file model_test.opts
// python script: 
// ---------------------------------------
static char help[] = "Application to investigate the effect of magma-matrix buoyancy on flow beneath mid-ocean ridges \n\n";

#include "MORbuoyancy.h"

// ---------------------------------------
// MAIN
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "main"
int main (int argc,char **argv)
{
  UsrData         *usr;
  // PetscLogDouble  start_time, end_time;
  PetscErrorCode  ierr;

  // Initialize application
  ierr = PetscInitialize(&argc,&argv,(char*)0,help); if (ierr) return ierr;
  
  // Load command line or input file if required
  ierr = PetscOptionsInsert(PETSC_NULL,&argc,&argv,NULL); CHKERRQ(ierr);

  // Initialize parameters - set default, read input, characteristic scales and non-dimensional params
  ierr = UserParamsCreate(&usr,argc,argv); CHKERRQ(ierr);

  // Numerical solution

  // Destroy parameters
  ierr = UserParamsDestroy(usr); CHKERRQ(ierr);

  // Finalize main
  ierr = PetscFinalize();
  return ierr;
}