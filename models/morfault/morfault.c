// ---------------------------------------
// Mid-ocean ridge model using theory of two-phase flow and visco-elasto-viscoplastic rheology
// run: ./morfault.app -log_view -options_file model_input.opts 
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
  PetscScalar    xmin, xmax, zmin, zmax, dt, dt_phi, dt_vf, dt_T;
  FDPDE          fdPV, fdT, fdphi;
  DM             dmPV, dmT, dmphi, dmswarm, dmTcoeff, dmphicoeff;
  Vec            xPV, xT, xphi, xTprev, xTcoeff, xTcoeffprev, xphiprev, xphicoeff, xphicoeffprev;
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

  // Set up mechanics - Stokes-Darcy system (PV)
  PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");
  if (usr->par->two_phase == 1) {
    PetscPrintf(PETSC_COMM_WORLD,"# Set-up MECHANICS: FD-PDE StokesDarcy2Field (PV)\n");
    ierr = FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_STOKESDARCY2FIELD,&fdPV);CHKERRQ(ierr);
    ierr = FDPDESetUp(fdPV);CHKERRQ(ierr);
    ierr = FDPDESetFunctionBCList(fdPV,FormBCList_PV,bc_description_PV,usr); CHKERRQ(ierr);
    ierr = FDPDESetFunctionCoefficient(fdPV,FormCoefficient_PV,coeff_description_PV,usr); CHKERRQ(ierr);
  }

  if (usr->par->two_phase == 0) {
    PetscPrintf(PETSC_COMM_WORLD,"# Set-up MECHANICS: FD-PDE STOKES (PV)\n");
    ierr = FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_STOKES,&fdPV);CHKERRQ(ierr);
    ierr = FDPDESetUp(fdPV);CHKERRQ(ierr);
    ierr = FDPDESetFunctionBCList(fdPV,FormBCList_PV_Stokes,bc_description_PV,usr); CHKERRQ(ierr);
    ierr = FDPDESetFunctionCoefficient(fdPV,FormCoefficient_PV_Stokes,coeff_description_PV,usr); CHKERRQ(ierr);
    usr->par->phi0 = 0.0;
  }

  ierr = SNESSetFromOptions(fdPV->snes); CHKERRQ(ierr);
  ierr = SNESSetOptionsPrefix(fdPV->snes,"pv_"); CHKERRQ(ierr);
  fdPV->output_solver_failure_report = PETSC_FALSE;
  ierr = FDPDEView(fdPV); CHKERRQ(ierr);

  // Set up advection and time-stepping
  AdvectSchemeType advtype;
  if (par->adv_scheme==0) advtype = ADV_UPWIND;
  if (par->adv_scheme==1) advtype = ADV_UPWIND2;
  if (par->adv_scheme==2) advtype = ADV_FROMM;

  TimeStepSchemeType timesteptype;
  if (par->ts_scheme ==  0) timesteptype = TS_FORWARD_EULER;
  if (par->ts_scheme ==  1) timesteptype = TS_BACKWARD_EULER;
  if (par->ts_scheme ==  2) timesteptype = TS_CRANK_NICHOLSON;

  // Set up Temperature system (T)
  PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");
  PetscPrintf(PETSC_COMM_WORLD,"# Set-up ENERGY: FD-PDE AdvDiff (T) \n");
  ierr = FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_ADVDIFF,&fdT);CHKERRQ(ierr);
  ierr = FDPDESetUp(fdT);CHKERRQ(ierr);
  ierr = FDPDESetFunctionBCList(fdT,FormBCList_T,bc_description_T,usr); CHKERRQ(ierr);
  ierr = FDPDESetFunctionCoefficient(fdT,FormCoefficient_T,coeff_description_T,usr); CHKERRQ(ierr);
  ierr = SNESSetFromOptions(fdT->snes); CHKERRQ(ierr);
  ierr = SNESSetOptionsPrefix(fdT->snes,"t_"); CHKERRQ(ierr);
  fdT->output_solver_failure_report = PETSC_FALSE;

  ierr = FDPDEAdvDiffSetAdvectSchemeType(fdT,advtype);CHKERRQ(ierr);
  ierr = FDPDEAdvDiffSetTimeStepSchemeType(fdT,timesteptype);CHKERRQ(ierr);
  ierr = FDPDEView(fdT); CHKERRQ(ierr);

  // Set up porosity evolution system
  PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");
  PetscPrintf(PETSC_COMM_WORLD,"# Set-up POROSITY (solid): FD-PDE AdvDiff (phi) \n");
  ierr = FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_ADVDIFF,&fdphi);CHKERRQ(ierr);
  ierr = FDPDESetUp(fdphi);CHKERRQ(ierr);
  ierr = FDPDESetFunctionBCList(fdphi,FormBCList_phi,bc_description_phi,usr); CHKERRQ(ierr);
  ierr = FDPDESetFunctionCoefficient(fdphi,FormCoefficient_phi,coeff_description_phi,usr); CHKERRQ(ierr);
  ierr = SNESSetFromOptions(fdphi->snes); CHKERRQ(ierr);
  ierr = SNESSetOptionsPrefix(fdphi->snes,"phi_"); CHKERRQ(ierr);
  fdphi->output_solver_failure_report = PETSC_FALSE;

  ierr = FDPDEAdvDiffSetAdvectSchemeType(fdphi,advtype);CHKERRQ(ierr);
  ierr = FDPDEAdvDiffSetTimeStepSchemeType(fdphi,timesteptype);CHKERRQ(ierr);
  ierr = FDPDEView(fdphi); CHKERRQ(ierr);

  // Log info
  if (usr->par->log_info) {
    fdPV->log_info = PETSC_TRUE;
    fdT->log_info  = PETSC_TRUE;
    fdphi->log_info  = PETSC_TRUE;
  }
  
  // Prepare data for coupling T-PV
  PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");
  PetscPrintf(PETSC_COMM_WORLD,"# Preparing data for PV-T-phi coupling \n");

  ierr = FDPDEGetDM(fdPV,&dmPV); CHKERRQ(ierr);
  usr->dmPV = dmPV;

  ierr = FDPDEGetDM(fdT,&dmT); CHKERRQ(ierr);
  usr->dmT = dmT;

  ierr = FDPDEGetDM(fdphi,&dmphi); CHKERRQ(ierr);
  usr->dmphi = dmphi;

  { // Processor information
    PetscInt nRanks0, nRanks1, nRanks2, nRanks3, size;
    ierr = MPI_Comm_size(usr->comm,&size);CHKERRQ(ierr);
    ierr = DMStagGetNumRanks(dmPV,&nRanks0,&nRanks1,NULL); CHKERRQ(ierr);
    ierr = DMStagGetNumRanks(dmT,&nRanks2,&nRanks3,NULL); CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"# Processor partitioning [%d cpus]: PV [%d,%d] T [%d,%d]\n",size,nRanks0,nRanks1,nRanks2,nRanks3);
  }

  ierr = FDPDEGetSolution(fdPV,&xPV);CHKERRQ(ierr);
  ierr = VecDuplicate(xPV,&usr->xPV);CHKERRQ(ierr);
  ierr = VecDestroy(&xPV);CHKERRQ(ierr);

  ierr = FDPDEGetSolution(fdT,&xT);CHKERRQ(ierr);
  ierr = VecDuplicate(xT,&usr->xT);CHKERRQ(ierr);
  ierr = VecDestroy(&xT);CHKERRQ(ierr);

  ierr = FDPDEGetSolution(fdphi,&xphi);CHKERRQ(ierr);
  ierr = VecDuplicate(xphi,&usr->xphi);CHKERRQ(ierr);
  ierr = VecDestroy(&xphi);CHKERRQ(ierr);

  // Create dmeps/vec for strain rates - center and corner
  ierr = DMStagCreateCompatibleDMStag(dmPV,4,0,4,0,&usr->dmeps); CHKERRQ(ierr);
  ierr = DMSetUp(usr->dmeps); CHKERRQ(ierr);
  ierr = DMStagSetUniformCoordinatesProduct(usr->dmeps,xmin,xmax,zmin,zmax,0.0,0.0);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(usr->dmeps,&usr->xeps); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(usr->dmeps,&usr->xtau); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(usr->dmeps,&usr->xtau_old); CHKERRQ(ierr);
  ierr = VecSet(usr->xtau_old,0.0); CHKERRQ(ierr);

  // Create dmVel for bulk and fluid velocities (dof=7 on faces)
  ierr = DMStagCreateCompatibleDMStag(usr->dmPV,0,2,0,0,&usr->dmVel); CHKERRQ(ierr);
  ierr = DMSetUp(usr->dmVel); CHKERRQ(ierr);
  ierr = DMStagSetUniformCoordinatesProduct(usr->dmVel,xmin,xmax,zmin,zmax,0.0,0.0);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(usr->dmVel,&usr->xVel);CHKERRQ(ierr);

  // Create dmmatProp for material properties (dof=MATPROP_NPROP in center)
  ierr = DMStagCreateCompatibleDMStag(usr->dmPV,0,0,MATPROP_NPROP,0,&usr->dmmatProp); CHKERRQ(ierr);
  ierr = DMSetUp(usr->dmmatProp); CHKERRQ(ierr);
  ierr = DMStagSetUniformCoordinatesProduct(usr->dmmatProp,xmin,xmax,zmin,zmax,0.0,0.0);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(usr->dmmatProp,&usr->xmatProp);CHKERRQ(ierr);

  // Create dmPlith for lithostatic pressure
  ierr = DMStagCreateCompatibleDMStag(usr->dmPV,0,0,1,0,&usr->dmPlith); CHKERRQ(ierr);
  ierr = DMSetUp(usr->dmPlith); CHKERRQ(ierr);
  ierr = DMStagSetUniformCoordinatesProduct(usr->dmPlith,xmin,xmax,zmin,zmax,0.0,0.0);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(usr->dmPlith,&usr->xPlith);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(usr->dmPlith,&usr->xDP); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(usr->dmPlith,&usr->xDP_old); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(usr->dmPlith,&usr->xplast); CHKERRQ(ierr);
  ierr = VecSet(usr->xDP_old,0.0); CHKERRQ(ierr);

  // Create vec for plastic strain
  ierr = DMCreateGlobalVector(usr->dmPlith, &usr->xstrain); CHKERRQ(ierr);
  ierr = VecZeroEntries(usr->xstrain); CHKERRQ(ierr);

  // Create dmMPhase for marker phase fractions (lithology)
  PetscInt nm = usr->nph;
  ierr = DMStagCreateCompatibleDMStag(usr->dmPV,nm,nm,nm,0,&usr->dmMPhase); CHKERRQ(ierr);
  ierr = DMSetUp(usr->dmMPhase); CHKERRQ(ierr);
  ierr = DMStagSetUniformCoordinatesProduct(usr->dmMPhase,xmin,xmax,zmin,zmax,0.0,0.0);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(usr->dmMPhase,&usr->xMPhase);CHKERRQ(ierr);

  // Set up a swarm object and assign several fields
  PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");
  PetscPrintf(PETSC_COMM_WORLD,"# Set-up PARTICLES: DMSWARM \n");
  ierr = DMStagPICCreateDMSwarm(dmPV,&dmswarm);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(dmswarm,"id",1,PETSC_REAL);CHKERRQ(ierr); // main field (user defined)
  ierr = DMSwarmRegisterPetscDatatypeField(dmswarm,"id0",1,PETSC_REAL);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(dmswarm,"id1",1,PETSC_REAL);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(dmswarm,"id2",1,PETSC_REAL);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(dmswarm,"id3",1,PETSC_REAL);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(dmswarm,"id4",1,PETSC_REAL);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(dmswarm,"id5",1,PETSC_REAL);CHKERRQ(ierr);
  ierr = DMStagPICFinalize(dmswarm);CHKERRQ(ierr);
  usr->dmswarm = dmswarm;
  
  if (par->restart==0) {
    // Initial condition and initial PV guess 
    PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");
    PetscPrintf(PETSC_COMM_WORLD,"# Set initial conditions and PV guess \n");
    ierr = SetInitialConditions(fdPV,fdT,fdphi,usr);CHKERRQ(ierr);
  } else { 
    // Restart from file
    PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");
    PetscPrintf(PETSC_COMM_WORLD,"# Restart from timestep %d \n",par->restart);
    ierr = LoadRestartFromFile(fdPV,fdT,fdphi,usr);CHKERRQ(ierr);
  } 

  nd->istep++;

  // Time loop
  while ((nd->t <= nd->tmax) && (nd->istep <= par->tstep)) {
    PetscPrintf(PETSC_COMM_WORLD,"\n# --------------------------------------- #\n");
    PetscPrintf(PETSC_COMM_WORLD,"# TIMESTEP %d: \n",nd->istep);
    ierr = PetscTime(&start_time); CHKERRQ(ierr);

    // Get a guess for timestep
    ierr   = FDPDEAdvDiffComputeExplicitTimestep(fdT,&dt_T);CHKERRQ(ierr);
    ierr   = FDPDEAdvDiffComputeExplicitTimestep(fdphi,&dt_phi);CHKERRQ(ierr);
    ierr   = LiquidVelocityExplicitTimestep(usr->dmVel,usr->xVel,&dt_vf);CHKERRQ(ierr);

    dt     = PetscMin(dt_T,dt_phi);
    dt     = PetscMin(dt,dt_vf);
    nd->dt = PetscMin(dt,nd->dtmax);
    ierr   = FDPDEAdvDiffSetTimestep(fdT,nd->dt); CHKERRQ(ierr);
    ierr   = FDPDEAdvDiffSetTimestep(fdphi,nd->dt); CHKERRQ(ierr);

    PetscPrintf(PETSC_COMM_WORLD,"# Time-step (non-dimensional): dt_T = %1.12e dt_phi = %1.12e dt_vf = %1.12e dtmax = %1.12e \n",dt_T,dt_phi,dt_vf,nd->dtmax);

    // Update lithostatic pressure 
    ierr = UpdateLithostaticPressure(usr->dmPlith,usr->xPlith,usr);CHKERRQ(ierr);

    // Solve PV
    PetscPrintf(PETSC_COMM_WORLD,"# (PV) Mechanics Solver - Stokes-Darcy2Field \n");
    SNESConvergedReason reason;
    converged = PETSC_FALSE;
    while (!converged) {
      // PetscPrintf(PETSC_COMM_WORLD,"# (PV) Time-step (iteration): dt = %1.12e \n",nd->dt);
      ierr = FDPDESolve(fdPV,&converged);CHKERRQ(ierr);
      ierr = SNESGetConvergedReason(fdPV->snes,&reason); CHKERRQ(ierr);
      if (!converged) { 
        break; 
      }
    }

    // Get solution
    ierr = FDPDEGetSolution(fdPV,&xPV);CHKERRQ(ierr);
    ierr = VecCopy(xPV,usr->xPV);CHKERRQ(ierr);
    ierr = VecDestroy(&xPV);CHKERRQ(ierr);

    // Integrate the plastic strain
    ierr = IntegratePlasticStrain(usr->dmPlith,usr->xstrain,usr->xplast,usr); CHKERRQ(ierr);

    // Update fluid velocity
    ierr = ComputeFluidAndBulkVelocity(usr->dmPV,usr->xPV,usr->dmPlith,usr->xPlith,usr->dmphi,usr->xphi,usr->dmVel,usr->xVel,usr);CHKERRQ(ierr);

    // Porosity Solver
    PetscPrintf(PETSC_COMM_WORLD,"\n# (phi) Porosity Solver \n");
    converged = PETSC_FALSE;
    while (!converged) {
      // PetscPrintf(PETSC_COMM_WORLD,"# (phi) Time-step (iteration): dt = %1.12e \n",nd->dt);
      ierr = FDPDESolve(fdphi,&converged);CHKERRQ(ierr);
      ierr = SNESGetConvergedReason(fdphi->snes,&reason); CHKERRQ(ierr);
      if (!converged) { 
        break; 
      }
    }

    // Get solution
    ierr = FDPDEGetSolution(fdphi,&xphi);CHKERRQ(ierr);
    ierr = VecCopy(xphi,usr->xphi);CHKERRQ(ierr);
    ierr = VecDestroy(&xphi);CHKERRQ(ierr);

    // Correct negative porosity
    ierr = CorrectNegativePorosity(usr->dmphi,usr->xphi);CHKERRQ(ierr);

    // Correct porosity at the free surface
    ierr = CorrectPorosityFreeSurface(usr->dmphi,usr->xphi,usr->dmMPhase,usr->xMPhase);CHKERRQ(ierr);

    // Solve energy
    PetscPrintf(PETSC_COMM_WORLD,"\n# (T) Energy Solver \n");
    converged = PETSC_FALSE;
    while (!converged) {
      // PetscPrintf(PETSC_COMM_WORLD,"# Time-step (iteration): dt = %1.12e \n",nd->dt);
      ierr = FDPDESolve(fdT,&converged);CHKERRQ(ierr);
      if (!converged) { 
        break; 
        // nd->dt *= 1e-1;
        // ierr = FDPDEAdvDiffSetTimestep(fdT,nd->dt); CHKERRQ(ierr);
      }
    }
    // PetscPrintf(PETSC_COMM_WORLD,"# Time-step (non-dimensional): dt = %1.12e dtmax = %1.12e dtmax_grid = %1.12e\n",nd->dt,nd->dtmax,dt);

    // Get solution
    ierr = FDPDEGetSolution(fdT,&xT);CHKERRQ(ierr);
    ierr = VecCopy(xT,usr->xT);CHKERRQ(ierr);
    ierr = VecDestroy(&xT);CHKERRQ(ierr);

    // Advect markers - RK1
    PetscPrintf(PETSC_COMM_WORLD,"\n# (DMSWARM) Advect and update lithological phase fractions \n");
    PetscInt nmark0, nmark1, nmark2, nmark[2];
    ierr = DMSwarmGetSize(usr->dmswarm,&nmark0);CHKERRQ(ierr);
    ierr = MPoint_AdvectRK1(usr->dmswarm,usr->dmPV,usr->xPV,nd->dt);CHKERRQ(ierr);
    ierr = DMSwarmGetSize(usr->dmswarm,&nmark1);CHKERRQ(ierr);
    ierr = AddMarkerInflux(usr->dmswarm,usr); CHKERRQ(ierr);
    ierr = DMSwarmGetSize(usr->dmswarm,&nmark2);CHKERRQ(ierr);
    ierr = UpdateMarkerPhaseFractions(usr->dmswarm,usr->dmMPhase,usr->xMPhase,usr);CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"# (DMSWARM) Marker number: Initial = %d After advection = %d After influx = %d \n",nmark0,nmark1,nmark2);
    ierr = GetMarkerDensityPerCell(usr->dmswarm,usr->dmMPhase,nmark);CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"# (DMSWARM) Marker density: min = %d max = %d per cell \n",nmark[0],nmark[1]);

    // Prepare data for next time-step
    ierr = FDPDEAdvDiffGetPrevSolution(fdT,&xTprev);CHKERRQ(ierr);
    ierr = VecCopy(usr->xT,xTprev);CHKERRQ(ierr);
    ierr = VecDestroy(&xTprev);CHKERRQ(ierr);
    ierr = FDPDEGetCoefficient(fdT,&dmTcoeff,&xTcoeff);CHKERRQ(ierr);
    ierr = FDPDEAdvDiffGetPrevCoefficient(fdT,&xTcoeffprev);CHKERRQ(ierr);
    ierr = VecCopy(xTcoeff,xTcoeffprev);CHKERRQ(ierr);
    ierr = VecDestroy(&xTcoeffprev);CHKERRQ(ierr);

    ierr = FDPDEAdvDiffGetPrevSolution(fdphi,&xphiprev);CHKERRQ(ierr);
    ierr = VecCopy(usr->xphi,xphiprev);CHKERRQ(ierr);
    ierr = VecDestroy(&xphiprev);CHKERRQ(ierr);
    ierr = FDPDEGetCoefficient(fdphi,&dmphicoeff,&xphicoeff);CHKERRQ(ierr);
    ierr = FDPDEAdvDiffGetPrevCoefficient(fdphi,&xphicoeffprev);CHKERRQ(ierr);
    ierr = VecCopy(xphicoeff,xphicoeffprev);CHKERRQ(ierr);
    ierr = VecDestroy(&xphicoeffprev);CHKERRQ(ierr);

    // Update time
    nd->t += nd->dt;

    // Output solution
    if ((nd->istep % par->tout == 0 ) || (fmod(nd->t,nd->dt_out) < nd->dt)) {
      ierr = DoOutput(fdPV,fdT,fdphi,usr);CHKERRQ(ierr);
    }

    // copy xtau, xDP to old
    ierr = VecCopy(usr->xtau, usr->xtau_old); CHKERRQ(ierr);
    ierr = VecCopy(usr->xDP, usr->xDP_old); CHKERRQ(ierr);

    nd->istep++;

    ierr = PetscTime(&end_time); CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"# TIME: time = %1.12e [yr] dt = %1.12e [yr] \n",nd->t*usr->scal->t/SEC_YEAR,nd->dt*usr->scal->t/SEC_YEAR);
    PetscPrintf(PETSC_COMM_WORLD,"# Timestep runtime: %g (sec) \n", end_time - start_time);
  }

  // Destroy objects
  ierr = FDPDEDestroy(&fdPV);CHKERRQ(ierr);
  ierr = FDPDEDestroy(&fdT);CHKERRQ(ierr);
  ierr = FDPDEDestroy(&fdphi);CHKERRQ(ierr);

  ierr = VecDestroy(&usr->xPV);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xT);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xphi);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xVel);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xMPhase);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xPlith);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xeps);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xtau);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xtau_old);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xDP);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xDP_old);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xplast);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xstrain);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xmatProp);CHKERRQ(ierr);

  ierr = DMDestroy(&usr->dmPV);CHKERRQ(ierr);
  ierr = DMDestroy(&usr->dmT);CHKERRQ(ierr);
  ierr = DMDestroy(&usr->dmphi);CHKERRQ(ierr);
  ierr = DMDestroy(&dmswarm);CHKERRQ(ierr);
  ierr = DMDestroy(&usr->dmVel);CHKERRQ(ierr);
  ierr = DMDestroy(&usr->dmMPhase);CHKERRQ(ierr);
  ierr = DMDestroy(&usr->dmPlith);CHKERRQ(ierr);
  ierr = DMDestroy(&usr->dmeps);CHKERRQ(ierr);
  ierr = DMDestroy(&usr->dmmatProp);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}