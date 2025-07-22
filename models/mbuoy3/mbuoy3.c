// ---------------------------------------
// Mid-ocean ridge model solving for conservation of mass, momentum, energy and composition using the Enthalpy Method
// Model after Katz 2008, 2010 - Using a 3-Field formulation (OPTIMIZED)
// run: ./mbuoy3 -options_file model_test.opts
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

  // Initialize application
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  
  // Load command line or input file if required
  PetscCall(PetscOptionsInsert(PETSC_NULLPTR,&argc,&argv,NULL)); 

  // Initialize parameters - set default, read input, characteristic scales and non-dimensional params
  PetscCall(UserParamsCreate(&usr,argc,argv)); 

  // Numerical solution
  PetscCall(PetscTime(&start_time)); 
  PetscCall(Numerical_solution(usr)); 
  PetscCall(PetscTime(&end_time)); 
  PetscPrintf(PETSC_COMM_WORLD,"# Runtime: %g (sec) \n", end_time - start_time);
  PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");

  // Destroy parameters
  PetscCall(UserParamsDestroy(usr)); 

  // Finalize main
  PetscCall(PetscFinalize());
  return 0;
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
  PetscFunctionBeginUser;

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
  PetscCall(FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_STOKESDARCY3FIELD,&fdPV));
  PetscCall(FDPDESetUp(fdPV));
  if (usr->par->full_ridge) {
    PetscCall(FDPDESetFunctionBCList(fdPV,FormBCList_PV_FullRidge,bc_description_PV,usr)); 
  } else {
    PetscCall(FDPDESetFunctionBCList(fdPV,FormBCList_PV,bc_description_PV,usr)); 
  }
  PetscCall(FDPDESetFunctionCoefficient(fdPV,FormCoefficient_PV,coeff_description_PV,usr)); 
  PetscCall(SNESSetFromOptions(fdPV->snes)); 
  PetscCall(FDPDEView(fdPV)); 
  PetscCall(SNESSetOptionsPrefix(fdPV->snes,"pv_")); 
  fdPV->output_solver_failure_report = PETSC_FALSE;

  // Set up Enthalpy system (HC)
  PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");
  PetscPrintf(PETSC_COMM_WORLD,"# Set-up ENERGY and COMPOSITION: FD-PDE Enthalpy (HC) \n");
  PetscCall(FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_ENTHALPY,&fdHC));
  PetscCall(FDPDESetUp(fdHC));
  if (usr->par->full_ridge) {
    PetscCall(FDPDESetFunctionBCList(fdHC,FormBCList_HC_FullRidge,bc_description_HC,usr)); 
  } else {
    PetscCall(FDPDESetFunctionBCList(fdHC,FormBCList_HC,bc_description_HC,usr)); 
  }

  if (usr->par->vf_nonlinear) { PetscCall(FDPDESetFunctionCoefficient(fdHC,FormCoefficient_HC_VF_nonlinear,coeff_description_HC,usr)); } 
  else { PetscCall(FDPDESetFunctionCoefficient(fdHC,FormCoefficient_HC,coeff_description_HC,usr));  }
  
  PetscCall(FDPDEEnthalpySetPotentialTemp(fdHC,Form_PotentialTemperature,usr));
  PetscCall(FDPDEEnthalpySetEnthalpyMethod(fdHC,Form_Enthalpy,enthalpy_method_description,usr));
  PetscCall(SNESSetFromOptions(fdHC->snes)); 
  PetscCall(FDPDEView(fdHC)); 
  PetscCall(SNESSetOptionsPrefix(fdHC->snes,"hc_")); 
  fdHC->output_solver_failure_report = PETSC_FALSE;

  // Set up advection and time-stepping
  AdvectSchemeType advtype;
  if (par->adv_scheme==0) advtype = ADV_UPWIND;
  if (par->adv_scheme==1) advtype = ADV_UPWIND2;
  if (par->adv_scheme==2) advtype = ADV_FROMM;
  PetscCall(FDPDEEnthalpySetAdvectSchemeType(fdHC,advtype));

  TimeStepSchemeType timesteptype;
  if (par->ts_scheme ==  0) timesteptype = TS_FORWARD_EULER;
  if (par->ts_scheme ==  1) timesteptype = TS_BACKWARD_EULER;
  if (par->ts_scheme ==  2) timesteptype = TS_CRANK_NICHOLSON;
  PetscCall(FDPDEEnthalpySetTimeStepSchemeType(fdHC,timesteptype));

  // Log info
  if (usr->par->log_info) {
    fdPV->log_info = PETSC_TRUE;
    fdHC->log_info = PETSC_TRUE;
  }
  
  // Prepare data for coupling HC-PV
  PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");
  PetscPrintf(PETSC_COMM_WORLD,"# Preparing data for PV-HC coupling \n");

  PetscCall(FDPDEGetDM(fdPV,&dmPV)); 
  usr->dmPV = dmPV;

  PetscCall(FDPDEGetDM(fdHC,&dmHC)); 
  usr->dmHC = dmHC;

  {
    PetscInt nRanks0, nRanks1, nRanks2, nRanks3, size;
    PetscCall(MPI_Comm_size(usr->comm,&size));
    PetscCall(DMStagGetNumRanks(dmPV,&nRanks0,&nRanks1,NULL)); 
    PetscCall(DMStagGetNumRanks(dmHC,&nRanks2,&nRanks3,NULL)); 
    PetscPrintf(PETSC_COMM_WORLD,"# Processor partitioning [%d cpus]: PV [%d,%d] HC [%d,%d]\n",size,nRanks0,nRanks1,nRanks2,nRanks3);
  }

  PetscCall(FDPDEGetSolution(fdPV,&xPV));
  PetscCall(VecDuplicate(xPV,&usr->xPV));
  PetscCall(VecDestroy(&xPV));

  PetscCall(FDPDEGetSolution(fdHC,&xHC));
  PetscCall(VecDuplicate(xHC,&usr->xHC));
  PetscCall(VecDestroy(&xHC));

  // Create dmVel for bulk and fluid velocities (dof=2 on faces)
  PetscCall(DMStagCreateCompatibleDMStag(usr->dmPV,0,2,0,0,&usr->dmVel)); 
  PetscCall(DMSetUp(usr->dmVel)); 
  PetscCall(DMStagSetUniformCoordinatesProduct(usr->dmVel,xmin,xmax,zmin,zmax,0.0,0.0));
  PetscCall(DMCreateGlobalVector(usr->dmVel,&usr->xVel));

  // Create dmmatProp for material properties (dof=7 in center)
  PetscCall(DMStagCreateCompatibleDMStag(usr->dmPV,0,0,7,0,&usr->dmmatProp)); 
  PetscCall(DMSetUp(usr->dmmatProp)); 
  PetscCall(DMStagSetUniformCoordinatesProduct(usr->dmmatProp,xmin,xmax,zmin,zmax,0.0,0.0));
  PetscCall(DMCreateGlobalVector(usr->dmmatProp,&usr->xmatProp));

  if (par->restart==0) {
    // Initial conditions - corner flow and half-space cooling model
    PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");
    PetscPrintf(PETSC_COMM_WORLD,"# Set initial conditions \n");
    PetscCall(SetInitialConditions(fdPV,fdHC,usr));
  } else { 
    // Restart from file
    PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");
    PetscPrintf(PETSC_COMM_WORLD,"# Restart from timestep %d \n",par->restart);
    PetscCall(LoadRestartFromFile(fdPV,fdHC,usr));
  } 

  nd->istep++;

  // Time loop
  while ((nd->t <= nd->tmax) && (nd->istep <= par->tstep)) {
    PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");
    PetscPrintf(PETSC_COMM_WORLD,"# TIMESTEP %d: \n",nd->istep);
    PetscCall(PetscTime(&start_time)); 
    
    // Solve energy and composition
    PetscPrintf(PETSC_COMM_WORLD,"# (HC) Energy-Composition Solver - Enthalpy Method \n");

    // Set time step size
    PetscCall(FDPDEEnthalpyComputeExplicitTimestep(fdHC,&dt));
    nd->dt = PetscMin(dt,nd->dtmax);
    PetscCall(FDPDEEnthalpySetTimestep(fdHC,nd->dt)); 

    converged = PETSC_FALSE;
    while (!converged) {
      PetscPrintf(PETSC_COMM_WORLD,"# Time-step (iteration): dt = %1.12e \n",nd->dt);
      PetscCall(FDPDESolve(fdHC,&converged));
      if (!converged) { // Reduce dt if not converged
        nd->dt *= 1e-1;
        PetscCall(FDPDEEnthalpySetTimestep(fdHC,nd->dt)); 
      }
    }
    PetscPrintf(PETSC_COMM_WORLD,"# Time-step (non-dimensional): dt = %1.12e dtmax = %1.12e dtmax_grid = %1.12e\n",nd->dt,nd->dtmax,dt);

    // Get solution
    PetscCall(FDPDEGetSolution(fdHC,&xHC));
    PetscCall(VecCopy(xHC,usr->xHC));
    PetscCall(VecDestroy(&xHC));

    // Update fields
    PetscCall(FDPDEEnthalpyUpdateDiagnostics(fdHC,usr->dmHC,usr->xHC,NULL,&xEnth)); 
    PetscCall(VecCopy(xEnth,usr->xEnth));
    PetscCall(VecDestroy(&xEnth));

    // Solve PV - default solves every timestep
    if ((nd->istep == 1 ) || (nd->istep % par->hc_cycles == 0 )) {
      PetscPrintf(PETSC_COMM_WORLD,"# (PV) Mechanics Solver - Stokes-Darcy3Field \n");
      // PetscCall(FDPDESolve(fdPV,NULL));

      SNESConvergedReason reason;
      converged = PETSC_FALSE;
      while (!converged) {
        PetscCall(FDPDESolve(fdPV,&converged));
        PetscCall(SNESGetConvergedReason(fdPV->snes,&reason)); 
        if (!converged) { // use pc_telescope if failed
          if (reason == -3) { // SNES_DIVERGED_LINEAR_SOLVE = -3 (error occurs in full ridge models with comp buoy)
            PetscPrintf(PETSC_COMM_WORLD,"# PV Solver FAILED - Switching to pc_telescope \n");
            PetscInt size;
            char     fsize[10];
            PetscCall(MPI_Comm_size(usr->comm,&size));
            PetscCall(PetscSNPrintf(fsize,sizeof(fsize),"%d",size));
            PetscCall(PetscOptionsClearValue(NULL,"-pv_pc_type"));
            PetscCall(PetscOptionsClearValue(NULL,"-pv_pc_factor_mat_solver_type"));

            PetscCall(PetscOptionsSetValue(NULL, "-pv_pc_type", "telescope")); 
            PetscCall(PetscOptionsSetValue(NULL, "-pv_pc_telescope_reduction_factor",fsize)); 
            PetscCall(PetscOptionsSetValue(NULL, "-pv_telescope_pc_type", "lu")); 
            PetscCall(PetscOptionsSetValue(NULL, "-pv_telescope_pc_factor_mat_solver_type", "mumps")); 
          } else {
            break; // terminate loop if different error
          }
        }
      }

      PetscCall(FDPDEGetSolution(fdPV,&xPV));
      PetscCall(VecCopy(xPV,usr->xPV));
      PetscCall(VecDestroy(&xPV));
    }

    // Update material properties for output
    PetscCall(UpdateMaterialProperties(usr->dmEnth,usr->xEnth,usr->dmmatProp,usr->xmatProp,usr));

    // Update fluid velocity
    PetscCall(ComputeFluidAndBulkVelocity(usr->dmPV,usr->xPV,usr->dmEnth,usr->xEnth,usr->dmVel,usr->xVel,usr));

    // Update melting rate and copy 
    PetscCall(ComputeGamma(usr->dmmatProp,usr->xmatProp,usr->dmPV,usr->xPV,usr->dmEnth,usr->xEnth,usr->xEnthold,usr)); 

    // Prepare data for next time-step
    PetscCall(VecCopy(usr->xEnth,usr->xEnthold));
    PetscCall(FDPDEEnthalpyGetPrevSolution(fdHC,&xHCprev));
    PetscCall(VecCopy(usr->xHC,xHCprev));
    PetscCall(VecDestroy(&xHCprev));
    PetscCall(FDPDEGetCoefficient(fdHC,&dmHCcoeff,&xHCcoeff));
    PetscCall(FDPDEEnthalpyGetPrevCoefficient(fdHC,&xHCcoeffprev));
    PetscCall(VecCopy(xHCcoeff,xHCcoeffprev));
    PetscCall(VecDestroy(&xHCcoeffprev));

    // Update lithostatic pressure 
    PetscCall(FDPDEEnthalpyGetPressure(fdHC,&dmP,&xP));
    PetscCall(FDPDEEnthalpyGetPrevPressure(fdHC,&xPprev));
    PetscCall(VecCopy(xP,xPprev));
    PetscCall(UpdateLithostaticPressure(dmP,xP,usr));
    PetscCall(VecDestroy(&xP));
    PetscCall(VecDestroy(&xPprev));
    PetscCall(DMDestroy(&dmP));

    // Compute fluxes out and crustal thickness
    PetscCall(ComputeMeltExtractOutflux(usr)); 
    PetscCall(ComputeAsymmetryFullRidge(usr)); 

    // Update time
    nd->t += nd->dt;

    // Output solution
    if ((nd->istep % par->tout == 0 ) || (fmod(nd->t,nd->dt_out) < nd->dt)) {
      PetscCall(DoOutput(fdPV,fdHC,usr));
    }

    nd->istep++;

    PetscCall(PetscTime(&end_time)); 
    PetscPrintf(PETSC_COMM_WORLD,"# TIME: time = %1.12e [yr] dt = %1.12e [yr] \n",nd->t*usr->scal->t/SEC_YEAR,nd->dt*usr->scal->t/SEC_YEAR);
    PetscPrintf(PETSC_COMM_WORLD,"# Timestep runtime: %g (sec) \n\n", end_time - start_time);
  }

  // Destroy objects
  PetscCall(FDPDEDestroy(&fdPV));
  PetscCall(FDPDEDestroy(&fdHC));

  PetscCall(VecDestroy(&usr->xPV));
  PetscCall(VecDestroy(&usr->xHC));
  PetscCall(VecDestroy(&usr->xVel));
  PetscCall(VecDestroy(&usr->xEnth));
  PetscCall(VecDestroy(&usr->xEnthold));
  PetscCall(VecDestroy(&usr->xmatProp));

  PetscCall(DMDestroy(&usr->dmPV));
  PetscCall(DMDestroy(&usr->dmHC));
  PetscCall(DMDestroy(&usr->dmVel));
  PetscCall(DMDestroy(&usr->dmEnth));
  PetscCall(DMDestroy(&usr->dmmatProp));

  PetscFunctionReturn(PETSC_SUCCESS);
}