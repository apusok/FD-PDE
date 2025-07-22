// ---------------------------------------
// Mid-ocean ridge model using theory of two-phase flow and visco-elasto-viscoplastic rheology
// run: ./morfault -log_view -options_file model_input.opts 
// ---------------------------------------
static char help[] = "Mid-ocean ridge model using theory of two-phase flow and visco-elasto-viscoplastic rheology \n\n";

#include "morfault.h"

// ---------------------------------------
// Descriptions
// ---------------------------------------
const char coeff_description_PV[] =
"  << Stokes-Darcy2Field Coefficients >> \n"
"  A = eta_eff \n"
"  B = body force+elastic\n" 
"  C = 0 \n"
"  D1 = zeta_eff-2/3 eta_eff \n"
"  D2 = -R^2*k_phi \n"
"  D3 = -R^2*k_phi(grad(Plith)-rho_ell*k_hat) \n";

const char coeff_description_T[] =
"  << Energy (AdvDiff) Coefficients >> \n"
"  A =  1 \n"
"  B =  1/Ra*kappa \n"
"  C =  0 \n"
"  v - Stokes-Darcy v_s velocity \n";

const char coeff_description_phi[] =
"  << Solid Porosity Coefficients (dimensionless) >> \n"
"  A = 1.0 \n"
"  B = 0 \n"
"  C = Gamma/rhos \n"
"  u = [ux, uz] - StokesDarcy solid velocity \n";

const char bc_description_PV[] =
"  << Stokes-Darcy BCs >> \n"
"  LEFT: Vx = -Vext, dVz/dx = 0, dp/dz = 0\n"
"  RIGHT: Vx = Vext, dVz/dx = 0, dp/dz = 0\n"
"  DOWN: dVx/dz = 0, Vz = Vin, p = 0 \n"
"  UP: dVx/dz = 0, Vz = 0, dp/dz = 0 \n";

const char bc_description_T[] =
"  << Energy (T) BCs >> \n"
"  LEFT: dT/dx = 0 \n"
"  RIGHT: dT/dx = 0 \n"
"  DOWN: T = Tbot \n"
"  UP: T = Ttop \n";

const char bc_description_phi[] =
"  << Porosity BCs >> \n"
"  LEFT: dphis/dx = 0 \n"
"  RIGHT: dphis/dx = 0 \n"
"  DOWN: phis = phis_bot \n"
"  UP: dphis/dz = 0 \n";

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
  PetscInt       nx, nz, iterPVphi; 
  PetscScalar    xmin, xmax, zmin, zmax, dt, dt_phi, dt_vf, dt_T;
  FDPDE          fdPV, fdT, fdphi;
  DM             dmPV, dmT, dmphi, dmswarm, dmTcoeff, dmphicoeff;
  Vec            xPV, xT, xphi, xTprev, xTcoeff, xTcoeffprev, xphiprev, xphicoeff, xphicoeffprev;
  PetscBool      converged, masscons;
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

  // Set up mechanics - Stokes-Darcy system (PV)
  PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");
  if (usr->par->two_phase == 1) {
    PetscPrintf(PETSC_COMM_WORLD,"# Set-up MECHANICS: FD-PDE StokesDarcy2Field (PV)\n");
    PetscCall(FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_STOKESDARCY2FIELD,&fdPV));
    PetscCall(FDPDESetUp(fdPV));
    PetscCall(FDPDESetFunctionBCList(fdPV,FormBCList_PV,bc_description_PV,usr)); 
    PetscCall(FDPDESetFunctionCoefficient(fdPV,FormCoefficient_PV_DPL,coeff_description_PV,usr)); 
  }

  if (usr->par->two_phase == 0) {
    PetscPrintf(PETSC_COMM_WORLD,"# Set-up MECHANICS: FD-PDE STOKES (PV)\n");
    PetscCall(FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_STOKES,&fdPV));
    PetscCall(FDPDESetUp(fdPV));
    PetscCall(FDPDESetFunctionBCList(fdPV,FormBCList_PV_Stokes,bc_description_PV,usr)); 
    PetscCall(FDPDESetFunctionCoefficient(fdPV,FormCoefficient_PV_Stokes_DPL,coeff_description_PV,usr)); 
    usr->par->phi0 = 0.0;
  }

  PetscCall(SNESSetFromOptions(fdPV->snes)); 
  PetscCall(SNESSetOptionsPrefix(fdPV->snes,"pv_")); 
  fdPV->output_solver_failure_report = PETSC_FALSE;
  PetscCall(FDPDEView(fdPV)); 

  // Set up advection and time-stepping
  AdvectSchemeType advtype;
  if (par->adv_scheme==0) advtype = ADV_UPWIND;
  if (par->adv_scheme==1) advtype = ADV_UPWIND2;
  if (par->adv_scheme==2) advtype = ADV_FROMM;
  if (par->adv_scheme==3) advtype = ADV_UPWIND_MINMOD;

  TimeStepSchemeType timesteptype;
  if (par->ts_scheme ==  0) timesteptype = TS_FORWARD_EULER;
  if (par->ts_scheme ==  1) timesteptype = TS_BACKWARD_EULER;
  if (par->ts_scheme ==  2) timesteptype = TS_CRANK_NICHOLSON;

  // Set up Temperature system (T)
  PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");
  PetscPrintf(PETSC_COMM_WORLD,"# Set-up ENERGY: FD-PDE AdvDiff (T) \n");
  PetscCall(FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_ADVDIFF,&fdT));
  PetscCall(FDPDESetUp(fdT));
  PetscCall(FDPDESetFunctionBCList(fdT,FormBCList_T,bc_description_T,usr)); 
  PetscCall(FDPDESetFunctionCoefficient(fdT,FormCoefficient_T,coeff_description_T,usr)); 
  PetscCall(SNESSetFromOptions(fdT->snes)); 
  PetscCall(SNESSetOptionsPrefix(fdT->snes,"t_")); 
  fdT->output_solver_failure_report = PETSC_FALSE;

  PetscCall(FDPDEAdvDiffSetAdvectSchemeType(fdT,advtype));
  PetscCall(FDPDEAdvDiffSetTimeStepSchemeType(fdT,timesteptype));
  PetscCall(FDPDEView(fdT)); 

  // Set up porosity evolution system
  PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");
  PetscPrintf(PETSC_COMM_WORLD,"# Set-up POROSITY (solid): FD-PDE AdvDiff (phi) \n");
  PetscCall(FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_ADVDIFF,&fdphi));
  PetscCall(FDPDESetUp(fdphi));
  PetscCall(FDPDESetFunctionBCList(fdphi,FormBCList_phi,bc_description_phi,usr)); 
  PetscCall(FDPDESetFunctionCoefficient(fdphi,FormCoefficient_phi,coeff_description_phi,usr)); 
  PetscCall(SNESSetFromOptions(fdphi->snes)); 
  PetscCall(SNESSetOptionsPrefix(fdphi->snes,"phi_")); 
  fdphi->output_solver_failure_report = PETSC_FALSE;

  PetscCall(FDPDEAdvDiffSetAdvectSchemeType(fdphi,advtype));
  PetscCall(FDPDEAdvDiffSetTimeStepSchemeType(fdphi,timesteptype));
  PetscCall(FDPDEView(fdphi)); 

  // Log info
  if (usr->par->log_info) {
    fdPV->log_info = PETSC_TRUE;
    fdT->log_info  = PETSC_TRUE;
    fdphi->log_info  = PETSC_TRUE;
  }
  
  // Prepare data for coupling T-PV
  PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");
  PetscPrintf(PETSC_COMM_WORLD,"# Preparing data for PV-T-phi coupling \n");

  PetscCall(FDPDEGetDM(fdPV,&dmPV)); 
  usr->dmPV = dmPV;

  PetscCall(FDPDEGetDM(fdT,&dmT)); 
  usr->dmT = dmT;

  PetscCall(FDPDEGetDM(fdphi,&dmphi)); 
  usr->dmphi = dmphi;

  { // Processor information
    PetscInt nRanks0, nRanks1, nRanks2, nRanks3, size;
    PetscCall(MPI_Comm_size(usr->comm,&size));
    PetscCall(DMStagGetNumRanks(dmPV,&nRanks0,&nRanks1,NULL)); 
    PetscCall(DMStagGetNumRanks(dmT,&nRanks2,&nRanks3,NULL)); 
    PetscPrintf(PETSC_COMM_WORLD,"# Processor partitioning [%d cpus]: PV [%d,%d] T [%d,%d]\n",size,nRanks0,nRanks1,nRanks2,nRanks3);
  }

  PetscCall(FDPDEGetSolution(fdPV,&xPV));
  PetscCall(VecDuplicate(xPV,&usr->xPV));
  PetscCall(VecDestroy(&xPV));

  PetscCall(FDPDEGetSolution(fdT,&xT));
  PetscCall(VecDuplicate(xT,&usr->xT));
  PetscCall(VecDestroy(&xT));

  PetscCall(FDPDEGetSolution(fdphi,&xphi));
  PetscCall(VecDuplicate(xphi,&usr->xphi));
  PetscCall(VecDestroy(&xphi));

  // Create dmeps/vec for strain rates - center and corner
  PetscCall(DMStagCreateCompatibleDMStag(dmPV,4,0,4,0,&usr->dmeps)); 
  PetscCall(DMSetUp(usr->dmeps)); 
  PetscCall(DMStagSetUniformCoordinatesProduct(usr->dmeps,xmin,xmax,zmin,zmax,0.0,0.0));
  PetscCall(DMCreateGlobalVector(usr->dmeps,&usr->xeps)); 
  PetscCall(DMCreateGlobalVector(usr->dmeps,&usr->xtau)); 
  PetscCall(DMCreateGlobalVector(usr->dmeps,&usr->xtau_old)); 
  PetscCall(VecSet(usr->xtau_old,0.0)); 

  // Create dmVel for bulk and fluid velocities (dof=7 on faces)
  PetscCall(DMStagCreateCompatibleDMStag(usr->dmPV,0,2,0,0,&usr->dmVel)); 
  PetscCall(DMSetUp(usr->dmVel)); 
  PetscCall(DMStagSetUniformCoordinatesProduct(usr->dmVel,xmin,xmax,zmin,zmax,0.0,0.0));
  PetscCall(DMCreateGlobalVector(usr->dmVel,&usr->xVel));

  // Create dmmatProp for material properties (dof=MATPROP_NPROP in center)
  PetscCall(DMStagCreateCompatibleDMStag(usr->dmPV,0,0,MATPROP_NPROP,0,&usr->dmmatProp)); 
  PetscCall(DMSetUp(usr->dmmatProp)); 
  PetscCall(DMStagSetUniformCoordinatesProduct(usr->dmmatProp,xmin,xmax,zmin,zmax,0.0,0.0));
  PetscCall(DMCreateGlobalVector(usr->dmmatProp,&usr->xmatProp));

  // Create dmPlith for lithostatic pressure
  PetscCall(DMStagCreateCompatibleDMStag(usr->dmPV,0,0,1,0,&usr->dmPlith)); 
  PetscCall(DMSetUp(usr->dmPlith)); 
  PetscCall(DMStagSetUniformCoordinatesProduct(usr->dmPlith,xmin,xmax,zmin,zmax,0.0,0.0));
  PetscCall(DMCreateGlobalVector(usr->dmPlith,&usr->xPlith));
  PetscCall(DMCreateGlobalVector(usr->dmPlith,&usr->xDP)); 
  PetscCall(DMCreateGlobalVector(usr->dmPlith,&usr->xDP_old)); 
  PetscCall(DMCreateGlobalVector(usr->dmPlith,&usr->xplast)); 
  PetscCall(VecSet(usr->xDP_old,0.0)); 

  // Create vec for plastic strain
  PetscCall(DMCreateGlobalVector(usr->dmPlith, &usr->xstrain)); 
  PetscCall(VecZeroEntries(usr->xstrain)); 

  // Create dmMPhase for marker phase fractions (lithology)
  PetscInt nm = usr->nph;
  PetscCall(DMStagCreateCompatibleDMStag(usr->dmPV,nm,nm,nm,0,&usr->dmMPhase)); 
  PetscCall(DMSetUp(usr->dmMPhase)); 
  PetscCall(DMStagSetUniformCoordinatesProduct(usr->dmMPhase,xmin,xmax,zmin,zmax,0.0,0.0));
  PetscCall(DMCreateGlobalVector(usr->dmMPhase,&usr->xMPhase));

  // Set up a swarm object and assign several fields
  PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");
  PetscPrintf(PETSC_COMM_WORLD,"# Set-up PARTICLES: DMSWARM \n");
  PetscCall(DMStagPICCreateDMSwarm(dmPV,&dmswarm));
  PetscCall(DMSwarmRegisterPetscDatatypeField(dmswarm,"id",1,PETSC_REAL)); // main field (user defined)
  PetscCall(DMSwarmRegisterPetscDatatypeField(dmswarm,"id0",1,PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(dmswarm,"id1",1,PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(dmswarm,"id2",1,PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(dmswarm,"id3",1,PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(dmswarm,"id4",1,PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(dmswarm,"id5",1,PETSC_REAL));
  PetscCall(DMStagPICFinalize(dmswarm));
  usr->dmswarm = dmswarm;
  
  if (par->restart==0) {
    // Initial condition and initial PV guess 
    PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");
    PetscPrintf(PETSC_COMM_WORLD,"# Set initial conditions and PV guess \n");
    PetscCall(SetInitialConditions(fdPV,fdT,fdphi,usr));
  } else { 
    // Restart from file
    PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");
    PetscPrintf(PETSC_COMM_WORLD,"# Restart from timestep %d \n",par->restart);
    PetscCall(LoadRestartFromFile(fdPV,fdT,fdphi,usr));
  } 

  nd->istep++;

  // Time loop
  while ((nd->t <= nd->tmax) && (nd->istep <= par->tstep)) {
    PetscPrintf(PETSC_COMM_WORLD,"\n# --------------------------------------- #\n");
    PetscPrintf(PETSC_COMM_WORLD,"# TIMESTEP %d: \n",nd->istep);
    PetscCall(PetscTime(&start_time)); 

    // Get a guess for timestep
    PetscCall(FDPDEAdvDiffComputeExplicitTimestep(fdT,&dt_T));
    PetscCall(FDPDEAdvDiffComputeExplicitTimestep(fdphi,&dt_phi));
    PetscCall(LiquidVelocityExplicitTimestep(usr->dmVel,usr->xVel,&dt_vf));

    dt     = PetscMin(dt_T,dt_phi);
    // dt     = PetscMin(dt,dt_vf);
    nd->dt = PetscMin(dt,nd->dtmax);

    PetscPrintf(PETSC_COMM_WORLD,"# Time-step (non-dimensional): dt_T = %1.12e dt_phi = %1.12e dt_vf = %1.12e dtmax = %1.12e \n",dt_T,dt_phi,dt_vf,nd->dtmax);

    // Update lithostatic pressure 
    PetscCall(UpdateLithostaticPressure(usr->dmPlith,usr->xPlith,usr));

    // Iterate PV, phis until phis requires no correction
    masscons = PETSC_FALSE;
    iterPVphi = 0;
    while (!masscons) {
      PetscPrintf(PETSC_COMM_WORLD,"\n# ITERATION PV-phi %d \n",iterPVphi);

      // Solve PV
      PetscPrintf(PETSC_COMM_WORLD,"# (PV) Mechanics Solver - Stokes-Darcy2Field \n");
      SNESConvergedReason reason;
      converged = PETSC_FALSE;
      while (!converged) {
        PetscPrintf(PETSC_COMM_WORLD,"# (PV) Time-step (iteration): dt = %1.12e \n",nd->dt);
        PetscCall(FDPDESolve(fdPV,&converged));
        PetscCall(SNESGetConvergedReason(fdPV->snes,&reason)); 
        if (!converged) { 
          // break; 
          nd->dt *= 0.5; // reduce timestep
        }
      }

      PetscCall(FDPDEGetSolution(fdPV,&xPV));
      PetscCall(VecCopy(xPV,usr->xPV));
      PetscCall(VecDestroy(&xPV));

      // Update fluid velocity
      PetscCall(ComputeFluidAndBulkVelocity(usr->dmPV,usr->xPV,usr->dmPlith,usr->xPlith,usr->dmphi,usr->xphi,usr->dmVel,usr->xVel,usr));

      // Porosity Solver 
      PetscCall(FDPDEAdvDiffSetTimestep(fdphi,nd->dt)); 
      PetscPrintf(PETSC_COMM_WORLD,"\n# (phi) Porosity Solver \n");
      converged = PETSC_FALSE;
      while (!converged) {
        // PetscPrintf(PETSC_COMM_WORLD,"# (phi) Time-step (iteration): dt = %1.12e \n",nd->dt);
        PetscCall(FDPDESolve(fdphi,&converged));
        PetscCall(SNESGetConvergedReason(fdphi->snes,&reason)); 
        if (!converged) { 
          break; 
        }
      }

      // Get solution
      PetscCall(FDPDEGetSolution(fdphi,&xphi));
      PetscCall(VecCopy(xphi,usr->xphi));
      PetscCall(VecDestroy(&xphi));

      // Check negative porosity
      // PetscCall(CheckNegativePorosity(usr->dmphi,usr->xphi,&masscons));
      
      if (!masscons) { 
        break; 
        // nd->dt *= 0.5; // reduce timestep
        // iterPVphi += 1;
      }
    }

    // Integrate the plastic strain
    PetscCall(IntegratePlasticStrain(usr->dmPlith,usr->xstrain,usr->xplast,usr)); 

    // Correct negative porosity
    // PetscCall(CorrectNegativePorosity(usr->dmphi,usr->xphi));

    // Correct porosity at the free surface
    if (usr->par->model_energy==0) {
      PetscCall(CorrectPorosityFreeSurface(usr->dmphi,usr->xphi,usr->dmMPhase,usr->xMPhase));
    }
    
    // Set timestep for T
    PetscCall(FDPDEAdvDiffSetTimestep(fdT,nd->dt)); 

    // Solve energy
    PetscPrintf(PETSC_COMM_WORLD,"\n# (T) Energy Solver \n");
    converged = PETSC_FALSE;
    while (!converged) {
      // PetscPrintf(PETSC_COMM_WORLD,"# Time-step (iteration): dt = %1.12e \n",nd->dt);
      PetscCall(FDPDESolve(fdT,&converged));
      if (!converged) { 
        break; 
        // nd->dt *= 1e-1;
        // PetscCall(FDPDEAdvDiffSetTimestep(fdT,nd->dt)); 
      }
    }
  
    // Get solution
    PetscCall(FDPDEGetSolution(fdT,&xT));
    PetscCall(VecCopy(xT,usr->xT));
    PetscCall(VecDestroy(&xT));

    // Melting and crystallisation - Boukare et al 2017
    if (usr->par->model_energy==1) {
      PetscCall(PhaseDiagram_1Component(usr->dmT,usr->xT,usr->dmphi,usr->xphi,usr->dmMPhase,usr->xMPhase,usr));
    }

    // Advect markers - RK1
    PetscPrintf(PETSC_COMM_WORLD,"\n# (DMSWARM) Advect and update lithological phase fractions \n");
    PetscInt nmark0, nmark1, nmark2, nmark[2];
    PetscCall(DMSwarmGetSize(usr->dmswarm,&nmark0));
    PetscCall(MPoint_AdvectRK1(usr->dmswarm,usr->dmPV,usr->xPV,nd->dt));
    PetscCall(DMSwarmGetSize(usr->dmswarm,&nmark1));
    // PetscCall(AddMarkerInflux(usr->dmswarm,usr)); 
    // PetscCall(AddMarkerInflux_FreeSurface(usr->dmswarm,usr)); 
    PetscCall(MarkerControl(usr->dmswarm,usr)); 
    PetscCall(DMSwarmGetSize(usr->dmswarm,&nmark2));
    PetscCall(UpdateMarkerPhaseFractions(usr->dmswarm,usr->dmMPhase,usr->xMPhase,usr));
    PetscPrintf(PETSC_COMM_WORLD,"# (DMSWARM) Marker number: Initial = %d After advection = %d After influx = %d \n",nmark0,nmark1,nmark2);
    PetscCall(GetMarkerDensityPerCell(usr->dmswarm,usr->dmMPhase,nmark));
    PetscPrintf(PETSC_COMM_WORLD,"# (DMSWARM) Marker density: min = %d max = %d per cell \n",nmark[0],nmark[1]);

    // Prepare data for next time-step
    PetscCall(FDPDEAdvDiffGetPrevSolution(fdT,&xTprev));
    PetscCall(VecCopy(usr->xT,xTprev));
    PetscCall(VecDestroy(&xTprev));
    PetscCall(FDPDEGetCoefficient(fdT,&dmTcoeff,&xTcoeff));
    PetscCall(FDPDEAdvDiffGetPrevCoefficient(fdT,&xTcoeffprev));
    PetscCall(VecCopy(xTcoeff,xTcoeffprev));
    PetscCall(VecDestroy(&xTcoeffprev));

    PetscCall(FDPDEAdvDiffGetPrevSolution(fdphi,&xphiprev));
    PetscCall(VecCopy(usr->xphi,xphiprev));
    PetscCall(VecDestroy(&xphiprev));
    PetscCall(FDPDEGetCoefficient(fdphi,&dmphicoeff,&xphicoeff));
    PetscCall(FDPDEAdvDiffGetPrevCoefficient(fdphi,&xphicoeffprev));
    PetscCall(VecCopy(xphicoeff,xphicoeffprev));
    PetscCall(VecDestroy(&xphicoeffprev));

    // Update time
    nd->t += nd->dt;

    // Output solution
    if ((nd->istep % par->tout == 0 ) || (fmod(nd->t,nd->dt_out) < nd->dt)) {
      PetscCall(DoOutput(fdPV,fdT,fdphi,usr));
    }

    // copy xtau, xDP to old
    PetscCall(VecCopy(usr->xtau, usr->xtau_old)); 
    PetscCall(VecCopy(usr->xDP, usr->xDP_old)); 

    nd->istep++;

    PetscCall(PetscTime(&end_time)); 
    PetscPrintf(PETSC_COMM_WORLD,"# TIME: time = %1.12e [yr] dt = %1.12e [yr] \n",nd->t*usr->scal->t/SEC_YEAR,nd->dt*usr->scal->t/SEC_YEAR);
    PetscPrintf(PETSC_COMM_WORLD,"# Timestep runtime: %g (sec) \n", end_time - start_time);
  }

  // Destroy objects
  PetscCall(FDPDEDestroy(&fdPV));
  PetscCall(FDPDEDestroy(&fdT));
  PetscCall(FDPDEDestroy(&fdphi));

  PetscCall(VecDestroy(&usr->xPV));
  PetscCall(VecDestroy(&usr->xT));
  PetscCall(VecDestroy(&usr->xphi));
  PetscCall(VecDestroy(&usr->xVel));
  PetscCall(VecDestroy(&usr->xMPhase));
  PetscCall(VecDestroy(&usr->xPlith));
  PetscCall(VecDestroy(&usr->xeps));
  PetscCall(VecDestroy(&usr->xtau));
  PetscCall(VecDestroy(&usr->xtau_old));
  PetscCall(VecDestroy(&usr->xDP));
  PetscCall(VecDestroy(&usr->xDP_old));
  PetscCall(VecDestroy(&usr->xplast));
  PetscCall(VecDestroy(&usr->xstrain));
  PetscCall(VecDestroy(&usr->xmatProp));

  PetscCall(DMDestroy(&usr->dmPV));
  PetscCall(DMDestroy(&usr->dmT));
  PetscCall(DMDestroy(&usr->dmphi));
  PetscCall(DMDestroy(&dmswarm));
  PetscCall(DMDestroy(&usr->dmVel));
  PetscCall(DMDestroy(&usr->dmMPhase));
  PetscCall(DMDestroy(&usr->dmPlith));
  PetscCall(DMDestroy(&usr->dmeps));
  PetscCall(DMDestroy(&usr->dmmatProp));

  PetscFunctionReturn(PETSC_SUCCESS);
}