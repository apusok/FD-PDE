// ---------------------------------------
// Mid-ocean ridge model solving for conservation of mass, momentum, energy and composition using the Enthalpy Method
// Model after Katz 2008, 2010 - Using a 3-Field formulation (OPTIMIZED)
// run: ./mbuoy3.app -options_file model_test.opts
// ---------------------------------------
static char help[] = "Mid-ocean ridge with melt-matrix buoyancy model using a 3-Field formulation \n\n";

#include "mbuoy3.h"

// ---------------------------------------
// Descriptions
// ---------------------------------------
const char coeff_description_PV[] =
"  << Stokes-Darcy3Field Coefficients >> \n"
"  A = delta^2*eta \n"
"  B = (phi+B)*k_hat\n" 
"  C = 0 \n"
"  D1 = -1/(delta^2*xi), xi = zeta-2/3eta \n"
"  D2 = -K \n"
"  D3 = -K*(1+Bf)*k_hat, K = (phi/phi0)^n, Bf = 0 \n"
"  D4 = -K \n"
"  DC = 0 \n";

const char bc_description_PV[] =
"  << Stokes-Darcy BCs >> \n"
"  LEFT: Vx = 0, dVz/dx = 0, dP/dz = 0, Pc = 0 or dPc/dz = 0 \n"
"  RIGHT: dVx/dx = 0, dVz/dx = 0, P = 0, Pc = 0 \n"
"  DOWN: tau_xz = 0, dVz/dz = 0, P = 0, Pc = 0 \n"
"  UP: Vx = U0, Vz = 0, dP/dz = 0, Pc = 0 \n";

const char coeff_description_HC[] =
"  << Energy and Composition (Enthalpy-HC) Coefficients >> \n"
"  A1 = e^(-Az), B1 = -S, C1 = -1/PeT*e^(-Az), D1 = 0  \n"
"  A2 = 1, B2 = 1, C2 = -1/PeC, D2 = 0  \n"
"  v, vs, vf - Stokes-Darcy velocity \n";

const char bc_description_HC[] =
"  << Enthalpy (H) BCs >> \n"
"  LEFT: dH/dx = 0, dC/dx = 0 \n"
"  RIGHT: dH/dx = 0, dC/dx = 0  \n"
"  DOWN: H = Hp, C = C0 \n"
"  UP: H = Hc, dC/dz = 0, MOR: dH/dz = 0 \n";

const char enthalpy_method_description[] =
"  << ENTHALPY METHOD >> \n"
"  Input: H, C, P \n"
"  Output: T, phi, CF, CS \n"
"   > see Katz (2010) for full equations \n";

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
  UsrData        *usr = (UsrData*) ctx;
  NdParams       *nd;
  Params         *par;
  PetscInt       nx, nz; 
  PetscScalar    xmin, xmax, zmin, zmax, dt;
  FDPDE          fdPV, fdHC;
  DM             dmPV, dmHC, dmHCcoeff, dmP;
  Vec            xPV, xP, xPprev;
  Vec            xHC, xHCprev, xHCcoeff, xHCcoeffprev, xEnth;
  PetscBool      converged;
  char           fout[FNAME_LENGTH];
  PetscLogDouble start_time, end_time;
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

  // Set up mechanics - Stokes-Darcy system 3-Field (PV)
  PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");
  PetscPrintf(PETSC_COMM_WORLD,"# Set-up MECHANICS: FD-PDE StokesDarcy3Field (PV)\n");
  ierr = FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_STOKESDARCY3FIELD,&fdPV);CHKERRQ(ierr);
  ierr = FDPDESetUp(fdPV);CHKERRQ(ierr);
  if (usr->par->full_ridge) {
    ierr = FDPDESetFunctionBCList(fdPV,FormBCList_PV_FullRidge,bc_description_PV,usr); CHKERRQ(ierr);
  } else {
    ierr = FDPDESetFunctionBCList(fdPV,FormBCList_PV,bc_description_PV,usr); CHKERRQ(ierr);
  }
  ierr = FDPDESetFunctionCoefficient(fdPV,FormCoefficient_PV,coeff_description_PV,usr); CHKERRQ(ierr);
  ierr = SNESSetFromOptions(fdPV->snes); CHKERRQ(ierr);
  ierr = FDPDEView(fdPV); CHKERRQ(ierr);
  ierr = SNESSetOptionsPrefix(fdPV->snes,"pv_"); CHKERRQ(ierr);
  fdPV->output_solver_failure_report = PETSC_FALSE;

  // Set up Enthalpy system (HC)
  PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");
  PetscPrintf(PETSC_COMM_WORLD,"# Set-up ENERGY and COMPOSITION: FD-PDE Enthalpy (HC) \n");
  ierr = FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_ENTHALPY,&fdHC);CHKERRQ(ierr);
  ierr = FDPDESetUp(fdHC);CHKERRQ(ierr);
  if (usr->par->full_ridge) {
    ierr = FDPDESetFunctionBCList(fdHC,FormBCList_HC_FullRidge,bc_description_HC,usr); CHKERRQ(ierr);
  } else {
    ierr = FDPDESetFunctionBCList(fdHC,FormBCList_HC,bc_description_HC,usr); CHKERRQ(ierr);
  }

  if (usr->par->vf_nonlinear) { ierr = FDPDESetFunctionCoefficient(fdHC,FormCoefficient_HC_VF_nonlinear,coeff_description_HC,usr); CHKERRQ(ierr);} 
  else { ierr = FDPDESetFunctionCoefficient(fdHC,FormCoefficient_HC,coeff_description_HC,usr); CHKERRQ(ierr); }
  
  ierr = FDPDEEnthalpySetPotentialTemp(fdHC,Form_PotentialTemperature,usr);CHKERRQ(ierr);
  ierr = FDPDEEnthalpySetEnthalpyMethod(fdHC,Form_Enthalpy,enthalpy_method_description,usr);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(fdHC->snes); CHKERRQ(ierr);
  ierr = FDPDEView(fdHC); CHKERRQ(ierr);
  ierr = SNESSetOptionsPrefix(fdHC->snes,"hc_"); CHKERRQ(ierr);
  fdHC->output_solver_failure_report = PETSC_FALSE;

  // Set up advection and time-stepping
  AdvectSchemeType advtype;
  if (par->adv_scheme==0) advtype = ADV_UPWIND;
  if (par->adv_scheme==1) advtype = ADV_UPWIND2;
  if (par->adv_scheme==2) advtype = ADV_FROMM;
  ierr = FDPDEEnthalpySetAdvectSchemeType(fdHC,advtype);CHKERRQ(ierr);

  TimeStepSchemeType timesteptype;
  if (par->ts_scheme ==  0) timesteptype = TS_FORWARD_EULER;
  if (par->ts_scheme ==  1) timesteptype = TS_BACKWARD_EULER;
  if (par->ts_scheme ==  2) timesteptype = TS_CRANK_NICHOLSON;
  ierr = FDPDEEnthalpySetTimeStepSchemeType(fdHC,timesteptype);CHKERRQ(ierr);

  // Log info
  if (usr->par->log_info) {
    fdPV->log_info = PETSC_TRUE;
    fdHC->log_info = PETSC_TRUE;
  }
  
  // Prepare data for coupling HC-PV
  PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");
  PetscPrintf(PETSC_COMM_WORLD,"# Preparing data for PV-HC coupling \n");

  ierr = FDPDEGetDM(fdPV,&dmPV); CHKERRQ(ierr);
  usr->dmPV = dmPV;

  ierr = FDPDEGetDM(fdHC,&dmHC); CHKERRQ(ierr);
  usr->dmHC = dmHC;

  {
    PetscInt nRanks0, nRanks1, nRanks2, nRanks3, size;
    ierr = MPI_Comm_size(usr->comm,&size);CHKERRQ(ierr);
    ierr = DMStagGetNumRanks(dmPV,&nRanks0,&nRanks1,NULL); CHKERRQ(ierr);
    ierr = DMStagGetNumRanks(dmHC,&nRanks2,&nRanks3,NULL); CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"# Processor partitioning [%d cpus]: PV [%d,%d] HC [%d,%d]\n",size,nRanks0,nRanks1,nRanks2,nRanks3);
  }

  ierr = FDPDEGetSolution(fdPV,&xPV);CHKERRQ(ierr);
  ierr = VecDuplicate(xPV,&usr->xPV);CHKERRQ(ierr);
  ierr = VecDestroy(&xPV);CHKERRQ(ierr);

  ierr = FDPDEGetSolution(fdHC,&xHC);CHKERRQ(ierr);
  ierr = VecDuplicate(xHC,&usr->xHC);CHKERRQ(ierr);
  ierr = VecDestroy(&xHC);CHKERRQ(ierr);

  // Create dmVel for bulk and fluid velocities (dof=2 on faces)
  ierr = DMStagCreateCompatibleDMStag(usr->dmPV,0,2,0,0,&usr->dmVel); CHKERRQ(ierr);
  ierr = DMSetUp(usr->dmVel); CHKERRQ(ierr);
  ierr = DMStagSetUniformCoordinatesProduct(usr->dmVel,xmin,xmax,zmin,zmax,0.0,0.0);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(usr->dmVel,&usr->xVel);CHKERRQ(ierr);

  // Create dmmatProp for material properties (dof=7 in center)
  ierr = DMStagCreateCompatibleDMStag(usr->dmPV,0,0,7,0,&usr->dmmatProp); CHKERRQ(ierr);
  ierr = DMSetUp(usr->dmmatProp); CHKERRQ(ierr);
  ierr = DMStagSetUniformCoordinatesProduct(usr->dmmatProp,xmin,xmax,zmin,zmax,0.0,0.0);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(usr->dmmatProp,&usr->xmatProp);CHKERRQ(ierr);

  if (par->restart==0) {
    // Initial conditions - corner flow and half-space cooling model
    PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");
    PetscPrintf(PETSC_COMM_WORLD,"# Set initial conditions \n");
    ierr = SetInitialConditions(fdPV,fdHC,usr);CHKERRQ(ierr);
  } else { 
    // Restart from file
    PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");
    PetscPrintf(PETSC_COMM_WORLD,"# Restart from timestep %d \n",par->restart);
    ierr = LoadRestartFromFile(fdPV,fdHC,usr);CHKERRQ(ierr);
  } 

  nd->istep++;

  // Time loop
  while ((nd->t <= nd->tmax) && (nd->istep <= par->tstep)) {
    PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");
    PetscPrintf(PETSC_COMM_WORLD,"# TIMESTEP %d: \n",nd->istep);
    ierr = PetscTime(&start_time); CHKERRQ(ierr);
    
    // Solve energy and composition
    PetscPrintf(PETSC_COMM_WORLD,"# HC Solver \n");

    // Set time step size
    ierr   = FDPDEEnthalpyComputeExplicitTimestep(fdHC,&dt);CHKERRQ(ierr);
    nd->dt = PetscMin(dt,nd->dtmax);
    ierr   = FDPDEEnthalpySetTimestep(fdHC,nd->dt); CHKERRQ(ierr);

    converged = PETSC_FALSE;
    while (!converged) {
      PetscPrintf(PETSC_COMM_WORLD,"# Time-step (iteration): dt = %1.12e \n",nd->dt);
      ierr = FDPDESolve(fdHC,&converged);CHKERRQ(ierr);
      if (!converged) { // Reduce dt if not converged
        nd->dt *= 1e-1;
        ierr = FDPDEEnthalpySetTimestep(fdHC,nd->dt); CHKERRQ(ierr);
      }
    }
    PetscPrintf(PETSC_COMM_WORLD,"# Time-step (non-dimensional): dt = %1.12e dtmax = %1.12e dtmax_grid = %1.12e\n",nd->dt,nd->dtmax,dt);

    // Get solution
    ierr = FDPDEGetSolution(fdHC,&xHC);CHKERRQ(ierr);
    ierr = VecCopy(xHC,usr->xHC);CHKERRQ(ierr);
    ierr = VecDestroy(&xHC);CHKERRQ(ierr);

    // Update fields
    ierr = FDPDEEnthalpyUpdateDiagnostics(fdHC,usr->dmHC,usr->xHC,NULL,&xEnth); CHKERRQ(ierr);
    ierr = VecCopy(xEnth,usr->xEnth);CHKERRQ(ierr);
    ierr = VecDestroy(&xEnth);CHKERRQ(ierr);

    // Solve PV - default solves every timestep
    if ((nd->istep == 1 ) || (nd->istep % par->hc_cycles == 0 )) {
      PetscPrintf(PETSC_COMM_WORLD,"# PV Solver \n");
      ierr = FDPDESolve(fdPV,NULL);CHKERRQ(ierr);
      ierr = FDPDEGetSolution(fdPV,&xPV);CHKERRQ(ierr);
      ierr = VecCopy(xPV,usr->xPV);CHKERRQ(ierr);
      ierr = VecDestroy(&xPV);CHKERRQ(ierr);
    }

    // Update material properties for output
    ierr = UpdateMaterialProperties(usr->dmEnth,usr->xEnth,usr->dmmatProp,usr->xmatProp,usr);CHKERRQ(ierr);

    // Update fluid velocity
    ierr = ComputeFluidAndBulkVelocity(usr->dmPV,usr->xPV,usr->dmEnth,usr->xEnth,usr->dmVel,usr->xVel,usr);CHKERRQ(ierr);

    // Update melting rate and copy 
    ierr = ComputeGamma(usr->dmmatProp,usr->xmatProp,usr->dmPV,usr->xPV,usr->dmEnth,usr->xEnth,usr->xEnthold,usr); CHKERRQ(ierr);

    // Prepare data for next time-step
    ierr = VecCopy(usr->xEnth,usr->xEnthold);CHKERRQ(ierr);
    ierr = FDPDEEnthalpyGetPrevSolution(fdHC,&xHCprev);CHKERRQ(ierr);
    ierr = VecCopy(usr->xHC,xHCprev);CHKERRQ(ierr);
    ierr = VecDestroy(&xHCprev);CHKERRQ(ierr);
    ierr = FDPDEGetCoefficient(fdHC,&dmHCcoeff,&xHCcoeff);CHKERRQ(ierr);
    ierr = FDPDEEnthalpyGetPrevCoefficient(fdHC,&xHCcoeffprev);CHKERRQ(ierr);
    ierr = VecCopy(xHCcoeff,xHCcoeffprev);CHKERRQ(ierr);
    ierr = VecDestroy(&xHCcoeffprev);CHKERRQ(ierr);

    // Update lithostatic pressure 
    ierr = FDPDEEnthalpyGetPressure(fdHC,&dmP,&xP);CHKERRQ(ierr);
    ierr = FDPDEEnthalpyGetPrevPressure(fdHC,&xPprev);CHKERRQ(ierr);
    ierr = VecCopy(xP,xPprev);CHKERRQ(ierr);
    ierr = UpdateLithostaticPressure(dmP,xP,usr);CHKERRQ(ierr);
    ierr = VecDestroy(&xP);CHKERRQ(ierr);
    ierr = VecDestroy(&xPprev);CHKERRQ(ierr);
    ierr = DMDestroy(&dmP);CHKERRQ(ierr);

    // Compute fluxes out and crustal thickness
    ierr = ComputeMeltExtractOutflux(usr); CHKERRQ(ierr);
    ierr = ComputeAsymmetryFullRidge(usr); CHKERRQ(ierr);

    // Update time
    nd->t += nd->dt;

    // Output solution
    if ((nd->istep % par->tout == 0 ) || (fmod(nd->t,nd->dt_out) < nd->dt)) {
      ierr = DoOutput(fdPV,fdHC,usr);CHKERRQ(ierr);
    }

    nd->istep++;

    ierr = PetscTime(&end_time); CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"# TIME: time = %1.12e [yr] dt = %1.12e [yr] \n",nd->t*usr->scal->t/SEC_YEAR,nd->dt*usr->scal->t/SEC_YEAR);
    PetscPrintf(PETSC_COMM_WORLD,"# Timestep runtime: %g (sec) \n\n", end_time - start_time);
  }

  // Destroy objects
  ierr = FDPDEDestroy(&fdPV);CHKERRQ(ierr);
  ierr = FDPDEDestroy(&fdHC);CHKERRQ(ierr);

  ierr = VecDestroy(&usr->xPV);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xHC);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xVel);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xEnth);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xEnthold);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xmatProp);CHKERRQ(ierr);

  ierr = DMDestroy(&usr->dmPV);CHKERRQ(ierr);
  ierr = DMDestroy(&usr->dmHC);CHKERRQ(ierr);
  ierr = DMDestroy(&usr->dmVel);CHKERRQ(ierr);
  ierr = DMDestroy(&usr->dmEnth);CHKERRQ(ierr);
  ierr = DMDestroy(&usr->dmmatProp);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}