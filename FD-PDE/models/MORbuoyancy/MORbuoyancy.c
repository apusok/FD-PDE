// ---------------------------------------
// Mid-ocean ridge model solving for conservation of mass, momentum, energy and composition using the Enthalpy Method
// Model after Katz 2008, 2010.
// run: ./MORbuoyancy.app -options_file model_test.opts
// python script: 
// ---------------------------------------
static char help[] = "Application to investigate the effect of magma-matrix buoyancy on flow beneath mid-ocean ridges \n\n";

#include "MORbuoyancy.h"

// ---------------------------------------
// Descriptions
// ---------------------------------------
const char coeff_description_PV[] =
"  << Stokes-Darcy Coefficients >> \n"
"  A = delta^2*eta  \n"
"  B = (phi+B)*k_hat\n" 
"  C = 0 \n"
"  D1 = delta^2*xi, xi=zeta-2/3eta \n"
"  D2 = -K \n"
"  D3 = -K*(1+Bf)*k_hat, K = (phi/phi0)^n, Bf = 0 \n";

const char bc_description_PV[] =
"  << Stokes-Darcy BCs >> \n"
"  LEFT: Vx = 0, dVz/dx = 0, dP/dz = 0 \n"
"  RIGHT: dVx/dx = 0, dVz/dx = 0, P = 0 \n"
"  DOWN: tau_xz = 0, dVz/dz = 0, P = 0 \n"
"  UP: Vx = U0, Vz = 0, dP/dz = 0 \n";

const char coeff_description_H[] =
"  << Enthalpy (H) Coefficients >> \n"
"  A =  \n"
"  B =  \n" 
"  C =  \n"
"  u =  \n";

const char bc_description_H[] =
"  << Enthalpy (H) BCs >> \n"
"  LEFT: dH/dx = 0 \n"
"  RIGHT: dH/dx = 0 \n"
"  DOWN: H = Hp \n"
"  UP: H = Hc, MOR: dH/dz = 0 \n";

const char coeff_description_C[] =
"  << Composition (C) Coefficients >> \n"
"  A =  \n"
"  B =  \n" 
"  C =  \n"
"  u =  \n";

const char bc_description_C[] =
"  << Composition (C) BCs >> \n"
"  LEFT: dC/dx = 0 \n"
"  RIGHT: dC/dx = 0 \n"
"  DOWN: C = C0 \n"
"  UP: dC/dz = 0 \n";

// ---------------------------------------
// MAIN
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "main"
int main (int argc,char **argv)
{
  UsrData         *usr;
  PetscLogDouble  start_time, end_time;
  PetscErrorCode  ierr;

  // Initialize application
  ierr = PetscInitialize(&argc,&argv,(char*)0,help); if (ierr) return ierr;
  
  // Load command line or input file if required
  ierr = PetscOptionsInsert(PETSC_NULL,&argc,&argv,NULL); CHKERRQ(ierr);

  // Initialize parameters - set default, read input, characteristic scales and non-dimensional params
  ierr = UserParamsCreate(&usr,argc,argv); CHKERRQ(ierr);

  // Numerical solution
  ierr = PetscTime(&start_time); CHKERRQ(ierr);
  ierr = Numerical_solution(usr); CHKERRQ(ierr);
  ierr = PetscTime(&end_time); CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"# Runtime: %g (sec) \n", end_time - start_time);
  PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");

  // Destroy parameters
  ierr = UserParamsDestroy(usr); CHKERRQ(ierr);

  // Finalize main
  ierr = PetscFinalize();
  return ierr;
}

// ---------------------------------------
// Numerical solution
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "Numerical_solution"
PetscErrorCode Numerical_solution(void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  NdParams      *nd;
  Params        *par;
  PetscInt      nx, nz, istep = 0; 
  PetscScalar   xmin, xmax, zmin, zmax;
  FDPDE         fdPV, fdH, fdC, fdHC, fd[2];
  DM            dmPV, dmHC;
  Vec           xPV, xH;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  nd  = usr->nd;
  par = usr->par;

  // Element count
  nx = par->nx;
  nz = par->nz;

  // Domain coords
  xmin = nd->xmin;
  zmin = nd->zmin;
  xmax = nd->xmin+nd->L;
  zmax = nd->zmin+nd->H;

  // Set up PV - Stokes-Darcy system
  PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");
  PetscPrintf(PETSC_COMM_WORLD,"# Set-up PV system: FD-PDE StokesDarcy2Field\n");
  ierr = FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_STOKESDARCY2FIELD,&fdPV);CHKERRQ(ierr);
  ierr = FDPDESetUp(fdPV);CHKERRQ(ierr);
  ierr = FDPDEGetDM(fdPV,&dmPV); CHKERRQ(ierr);
  ierr = FDPDESetFunctionBCList(fdPV,FormBCList_PV,bc_description_PV,usr); CHKERRQ(ierr);
  ierr = FDPDESetFunctionCoefficient(fdPV,FormCoefficient_PV,coeff_description_PV,usr); CHKERRQ(ierr);
  ierr = SNESSetFromOptions(fdPV->snes); CHKERRQ(ierr);
  ierr = FDPDEView(fdPV); CHKERRQ(ierr);
  ierr = SNESSetOptionsPrefix(fdPV->snes,"pv_"); CHKERRQ(ierr);

  // Set up Enthalpy - ADVDIFF system
  PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");
  PetscPrintf(PETSC_COMM_WORLD,"# Set-up Enthalpy (H) system: FD-PDE AdvDiff\n");
  ierr = FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_ADVDIFF,&fdH);CHKERRQ(ierr);
  ierr = FDPDESetUp(fdH);CHKERRQ(ierr);
  ierr = FDPDEGetDM(fdH,&dmHC); CHKERRQ(ierr);
  ierr = FDPDESetFunctionBCList(fdH,FormBCList_H,bc_description_H,usr); CHKERRQ(ierr);
  ierr = FDPDESetFunctionCoefficient(fdH,FormCoefficient_H,coeff_description_H,usr); CHKERRQ(ierr);
  ierr = SNESSetFromOptions(fdH->snes); CHKERRQ(ierr);

  // Set up Composition - ADVDIFF system
  PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");
  PetscPrintf(PETSC_COMM_WORLD,"# Set-up Composition (C) system: FD-PDE AdvDiff\n");
  ierr = FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_ADVDIFF,&fdC);CHKERRQ(ierr);
  ierr = FDPDESetUp(fdC);CHKERRQ(ierr);
  ierr = FDPDESetFunctionBCList(fdC,FormBCList_C,bc_description_C,usr); CHKERRQ(ierr);
  ierr = FDPDESetFunctionCoefficient(fdC,FormCoefficient_C,coeff_description_C,usr); CHKERRQ(ierr);
  ierr = SNESSetFromOptions(fdC->snes); CHKERRQ(ierr);

  // Set up advection and time-stepping
  AdvectSchemeType advtype;
  if (par->adv_scheme==0) advtype = ADV_UPWIND;
  if (par->adv_scheme==1) advtype = ADV_UPWIND2;
  if (par->adv_scheme==2) advtype = ADV_FROMM;
  ierr = FDPDEAdvDiffSetAdvectSchemeType(fdH,advtype);CHKERRQ(ierr);
  ierr = FDPDEAdvDiffSetAdvectSchemeType(fdC,advtype);CHKERRQ(ierr);

  TimeStepSchemeType timesteptype;
  if (par->ts_scheme ==  0) timesteptype = TS_FORWARD_EULER;
  if (par->ts_scheme ==  1) timesteptype = TS_BACKWARD_EULER;
  if (par->ts_scheme ==  2) timesteptype = TS_CRANK_NICHOLSON;
  ierr = FDPDEAdvDiffSetTimeStepSchemeType(fdH,timesteptype);CHKERRQ(ierr);
  ierr = FDPDEAdvDiffSetTimeStepSchemeType(fdC,timesteptype);CHKERRQ(ierr);

  // Prepare data for coupling HC-PV
  usr->dmPV = dmPV;
  usr->dmHC = dmHC;

  ierr = FDPDEGetSolution(fdPV,&xPV);CHKERRQ(ierr);
  ierr = VecDuplicate(xPV,&usr->xPV);CHKERRQ(ierr);
  ierr = VecDestroy(&xPV);CHKERRQ(ierr);

  ierr = FDPDEGetSolution(fdH,&xH);CHKERRQ(ierr);
  ierr = VecDuplicate(xH,&usr->xH);CHKERRQ(ierr);
  ierr = VecDuplicate(xH,&usr->xT);CHKERRQ(ierr);
  ierr = VecDuplicate(xH,&usr->xTheta);CHKERRQ(ierr);
  ierr = VecDuplicate(xH,&usr->xphi);CHKERRQ(ierr);
  ierr = VecDuplicate(xH,&usr->xC);CHKERRQ(ierr);
  ierr = VecDuplicate(xH,&usr->xCf);CHKERRQ(ierr);
  ierr = VecDuplicate(xH,&usr->xCs);CHKERRQ(ierr);
  ierr = VecDestroy(&xH);CHKERRQ(ierr);

  // Initial conditions
  PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");
  PetscPrintf(PETSC_COMM_WORLD,"# Set initial conditions \n");
  ierr = SetInitialConditions_HS(fdPV,fdH,fdC,usr);CHKERRQ(ierr);

  // Set up HC composite system
  fd[0] = fdH;
  fd[1] = fdC;

  PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");
  PetscPrintf(PETSC_COMM_WORLD,"# Set-up Composite HC system\n");
  ierr = FDPDECreate2(usr->comm,&fdHC);CHKERRQ(ierr);
  ierr = FDPDESetType(fdHC,FDPDE_COMPOSITE);CHKERRQ(ierr);
  ierr = FDPDECompositeSetFDPDE(fdHC,2,fd);CHKERRQ(ierr);
  ierr = FDPDESetUp(fdHC);CHKERRQ(ierr);
  ierr = FDPDEView(fdHC); CHKERRQ(ierr);
  ierr = SNESSetFromOptions(fdHC->snes); CHKERRQ(ierr);
  ierr = FDPDEDestroy(&fd[0]);CHKERRQ(ierr);
  ierr = FDPDEDestroy(&fd[1]);CHKERRQ(ierr);

  // Time loop
  while ((par->t <= par->tmax) && (istep < par->tstep)) {
    PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");
    PetscPrintf(PETSC_COMM_WORLD,"# TIMESTEP %d: \n",istep);

    // Solve HC
    // Update fields 
    // Solve PV
    // Output solution

    // increment timestep
    istep++;

    PetscPrintf(PETSC_COMM_WORLD,"# TIME: time = %1.12e [yr] dt = %1.12e [yr] \n\n",par->t*usr->scal->t/SEC_YEAR,par->dt*usr->scal->t/SEC_YEAR);
  }

  // Destroy objects
  ierr = DMDestroy(&dmPV);CHKERRQ(ierr);
  ierr = DMDestroy(&dmHC);CHKERRQ(ierr);
  ierr = FDPDEDestroy(&fdPV);CHKERRQ(ierr);
  ierr = FDPDEDestroy(&fdHC);CHKERRQ(ierr);

  ierr = VecDestroy(&usr->xPV);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xdimPV);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xH);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xT);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xTheta);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xphi);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xC);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xCf);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xCs);CHKERRQ(ierr);
  ierr = DMDestroy(&usr->dmPV);CHKERRQ(ierr);
  ierr = DMDestroy(&usr->dmHC);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}