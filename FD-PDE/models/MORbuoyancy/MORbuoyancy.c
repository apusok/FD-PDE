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

const char coeff_description_HC[] =
"  << Energy and Composition (Enthalpy-HC) Coefficients >> \n"
"  A1 = e^Az, B1 = -S, C1 = -1/PeT*e^Az, D1 = 0  \n"
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
  UsrData       *usr = (UsrData*) ctx;
  NdParams      *nd;
  Params        *par;
  PetscInt      nx, nz; 
  PetscScalar   xmin, xmax, zmin, zmax;
  FDPDE         fdPV, fdHC;
  DM            dmPV, dmHC, dmHCcoeff, dmEnth, dmP;
  Vec           xPV, xP, xPprev;
  Vec           xHC, xHCprev, xHCcoeff, xHCcoeffprev,  xEnth;
  PetscBool     converged;
  char           fout[FNAME_LENGTH];
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
  PetscPrintf(PETSC_COMM_WORLD,"# Set-up MECHANICS: FD-PDE StokesDarcy2Field (PV)\n");
  ierr = FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_STOKESDARCY2FIELD,&fdPV);CHKERRQ(ierr);
  ierr = FDPDESetUp(fdPV);CHKERRQ(ierr);
  ierr = FDPDESetFunctionBCList(fdPV,FormBCList_PV,bc_description_PV,usr); CHKERRQ(ierr);
  ierr = FDPDESetFunctionCoefficient(fdPV,FormCoefficient_PV,coeff_description_PV,usr); CHKERRQ(ierr);
  ierr = SNESSetFromOptions(fdPV->snes); CHKERRQ(ierr);
  ierr = FDPDEView(fdPV); CHKERRQ(ierr);
  ierr = SNESSetOptionsPrefix(fdPV->snes,"pv_"); CHKERRQ(ierr);

  // Set up Enthalpy system (HC)
  PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");
  PetscPrintf(PETSC_COMM_WORLD,"# Set-up ENERGY and COMPOSITION: FD-PDE Enthalpy (HC) \n");
  ierr = FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_ENTHALPY,&fdHC);CHKERRQ(ierr);
  ierr = FDPDESetUp(fdHC);CHKERRQ(ierr);
  ierr = FDPDESetFunctionBCList(fdHC,FormBCList_HC,bc_description_HC,usr); CHKERRQ(ierr);
  ierr = FDPDESetFunctionCoefficient(fdHC,FormCoefficient_HC,coeff_description_HC,usr); CHKERRQ(ierr);
  ierr = FDPDEEnthalpySetEnthalpyMethod(fdHC,Form_Enthalpy,enthalpy_method_description,usr);CHKERRQ(ierr);
  ierr = FDPDEEnthalpySetPotentialTemp(fdHC,Form_PotentialTemperature,usr);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(fdHC->snes); CHKERRQ(ierr);
  ierr = FDPDEView(fdHC); CHKERRQ(ierr);
  ierr = SNESSetOptionsPrefix(fdHC->snes,"hc_"); CHKERRQ(ierr);

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

  // Prepare data for coupling HC-PV
  PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");
  PetscPrintf(PETSC_COMM_WORLD,"# Preparing data for PV-HC coupling \n");

  ierr = FDPDEGetDM(fdPV,&dmPV); CHKERRQ(ierr);
  usr->dmPV = dmPV;

  ierr = FDPDEGetDM(fdHC,&dmHC); CHKERRQ(ierr);
  usr->dmHC = dmHC;

  ierr = FDPDEGetSolution(fdPV,&xPV);CHKERRQ(ierr);
  ierr = VecDuplicate(xPV,&usr->xPV);CHKERRQ(ierr);
  ierr = VecDestroy(&xPV);CHKERRQ(ierr);

  ierr = FDPDEGetSolution(fdHC,&xHC);CHKERRQ(ierr);
  ierr = VecDuplicate(xHC,&usr->xHC);CHKERRQ(ierr);
  ierr = VecDuplicate(xHC,&usr->xphiT);CHKERRQ(ierr);
  ierr = VecDuplicate(xHC,&usr->xphiTold);CHKERRQ(ierr);
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

  // Initial conditions - corner flow and half-space cooling model
  PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");
  PetscPrintf(PETSC_COMM_WORLD,"# Set initial conditions \n");
  ierr = SetInitialConditions(fdPV,fdHC,usr);CHKERRQ(ierr);

  par->istep++;

  // Time loop
  while ((nd->t <= nd->tmax) && (par->istep <= par->tstep)) {
    PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");
    PetscPrintf(PETSC_COMM_WORLD,"# TIMESTEP %d: \n",par->istep);
    
    // Solve energy and composition
    PetscPrintf(PETSC_COMM_WORLD,"# HC Solver \n");

    // Set time step size
    nd->dt = nd->dtmax;
    ierr = FDPDEEnthalpySetTimestep(fdHC,nd->dt); CHKERRQ(ierr);

    converged = PETSC_FALSE;
    while (!converged) {
      ierr = FDPDESolve(fdHC,&converged);CHKERRQ(ierr);
      if (!converged) { // Reduce dt if not converged
        ierr = FDPDEEnthalpyComputeExplicitTimestep(fdHC,&nd->dt);CHKERRQ(ierr);
        ierr = FDPDEEnthalpySetTimestep(fdHC,nd->dt); CHKERRQ(ierr);
      }
    }
    PetscPrintf(PETSC_COMM_WORLD,"# Time-step (non-dimensional): dt = %1.12e dtmax = %1.12e \n",nd->dt,nd->dtmax);

    // Get solution
    ierr = FDPDEGetSolution(fdHC,&xHC);CHKERRQ(ierr);
    ierr = VecCopy(xHC,usr->xHC);CHKERRQ(ierr);
    ierr = VecDestroy(&xHC);CHKERRQ(ierr);

    // Update fields
    ierr = FDPDEEnthalpyUpdateDiagnostics(fdHC,usr->dmHC,usr->xHC,&dmEnth,&xEnth); CHKERRQ(ierr);
    ierr = VecCopy(xEnth,usr->xEnth);CHKERRQ(ierr);
    ierr = DMDestroy(&dmEnth);CHKERRQ(ierr);
    ierr = VecDestroy(&xEnth);CHKERRQ(ierr);
    
    // Extract porosity and temperature needed for PV
    ierr = ExtractTemperaturePorosity(usr->dmEnth,usr->xEnth,usr,PETSC_FALSE);CHKERRQ(ierr);
  
    // Solve PV
    PetscPrintf(PETSC_COMM_WORLD,"# PV Solver \n");
    ierr = FDPDESolve(fdPV,NULL);CHKERRQ(ierr);
    ierr = FDPDEGetSolution(fdPV,&xPV);CHKERRQ(ierr);
    ierr = VecCopy(xPV,usr->xPV);CHKERRQ(ierr);
    ierr = VecDestroy(&xPV);CHKERRQ(ierr);

    // PetscPrintf(PETSC_COMM_WORLD,"# JACOBIAN \n");
    // ierr = MatView(fdPV->J,PETSC_VIEWER_STDOUT_WORLD);

    // PetscPrintf(PETSC_COMM_WORLD,"# RESIDUAL \n");
    // ierr = VecView(fdPV->r,PETSC_VIEWER_STDOUT_WORLD);

    // Update material properties for output
    ierr = UpdateMaterialProperties(usr->dmHC,usr->xHC,usr->xphiT,usr->dmEnth,usr->xEnth,usr->dmmatProp,usr->xmatProp,usr);CHKERRQ(ierr);

    // Update fluid velocity
    ierr = ComputeFluidAndBulkVelocity(usr->dmPV,usr->xPV,usr->dmHC,usr->xphiT,usr->dmVel,usr->xVel,usr);CHKERRQ(ierr);

    // Update melting rate and copy 
    ierr = ComputeGamma(usr->dmmatProp,usr->xmatProp,usr->dmPV,usr->xPV,usr->dmHC,usr->xphiT,usr->xphiTold,usr); CHKERRQ(ierr);
    ierr = VecCopy(usr->xphiT,usr->xphiTold);CHKERRQ(ierr);

    // Prepare data for next time-step
    ierr = FDPDEEnthalpyGetPrevSolution(fdHC,&xHCprev);CHKERRQ(ierr);
    ierr = VecCopy(usr->xHC,xHCprev);CHKERRQ(ierr);
    ierr = VecDestroy(&xHCprev);CHKERRQ(ierr);
    ierr = FDPDEGetCoefficient(fdHC,&dmHCcoeff,&xHCcoeff);CHKERRQ(ierr);
    ierr = FDPDEEnthalpyGetPrevCoefficient(fdHC,&xHCcoeffprev);CHKERRQ(ierr);
    ierr = VecCopy(xHCcoeff,xHCcoeffprev);CHKERRQ(ierr);
    ierr = VecDestroy(&xHCcoeffprev);CHKERRQ(ierr);

    // Update lithostatic pressure 
    ierr = FDPDEEnthalpyGetPressure(fdHC,&dmP,&xP);CHKERRQ(ierr);
    ierr = UpdateLithostaticPressure(dmP,xP,usr);CHKERRQ(ierr);
    ierr = FDPDEEnthalpyGetPrevPressure(fdHC,&xPprev);CHKERRQ(ierr);
    ierr = VecCopy(xP,xPprev);CHKERRQ(ierr);
    ierr = VecDestroy(&xP);CHKERRQ(ierr);
    ierr = VecDestroy(&xPprev);CHKERRQ(ierr);
    ierr = DMDestroy(&dmP);CHKERRQ(ierr);

    // Output solution
    if (par->istep % par->tout == 0 ) {
      ierr = DoOutput(fdPV,fdHC,usr);CHKERRQ(ierr);
    }

    // Update time
    nd->t += nd->dt;
    par->istep++;

    PetscPrintf(PETSC_COMM_WORLD,"# TIME: time = %1.12e [Myr] dt = %1.12e [Myr] \n\n",nd->t*usr->scal->t/SEC_YEAR*1e-6,nd->dt*usr->scal->t/SEC_YEAR*1e-6);
  }

  // // Destroy objects
  ierr = FDPDEDestroy(&fdPV);CHKERRQ(ierr);
  ierr = FDPDEDestroy(&fdHC);CHKERRQ(ierr);

  ierr = VecDestroy(&usr->xPV);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xHC);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xphiT);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xphiTold);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xVel);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xEnth);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xmatProp);CHKERRQ(ierr);

  ierr = DMDestroy(&usr->dmPV);CHKERRQ(ierr);
  ierr = DMDestroy(&usr->dmHC);CHKERRQ(ierr);
  ierr = DMDestroy(&usr->dmVel);CHKERRQ(ierr);
  ierr = DMDestroy(&usr->dmEnth);CHKERRQ(ierr);
  ierr = DMDestroy(&usr->dmmatProp);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}