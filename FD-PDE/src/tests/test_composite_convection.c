// ---------------------------------------
// Mantle convection benchmark (Blankenbach et al. 1989)
// Steady-state models:
//    1A - eta0=1e23, b=0, c=0, Ra = 1e4
//    1B - eta0=1e22, b=0, c=0, Ra = 1e5
//    1C - eta0=1e21, b=0, c=0, Ra = 1e6
//    2A - eta0=1e23, b=ln(1000), c=0, Ra0 = 1e4
//    2B - eta0=1e23, b=ln(16384), c=ln(64), L=2500, Ra0 = 1e4
// Time-dependent models:
//    3A - eta0=1e23, b=0, c=0, L=1500, Ra = 216000
// run: ./tests/test_composite_convection.app -pc_type lu -pc_factor_mat_solver_type umfpack -nx 10 -nz 10
//
// Run benchmark tests:
// python: ./tests/python/test_composite_convection.py
//
// Problem cases: 
//    nd1 - composite P-v-T, nondimensional form as in Moresi and Zhong 1995 (Ra in momentum)
//    nd2 - composite P-v-T, nondimensional form as in Wilson et al. 2017 (Ra in energy)
//    dc  - decoupled P-v, T system, nondimensional form nd1
// ---------------------------------------
static char help[] = "Application to solve the mantle convection benchmark (Blankenbach et al. 1989) with FD-PDE \n\n";

// define convenient names for DMStagStencilLocation
#define DOWN_LEFT  DMSTAG_DOWN_LEFT
#define DOWN       DMSTAG_DOWN
#define DOWN_RIGHT DMSTAG_DOWN_RIGHT
#define LEFT       DMSTAG_LEFT
#define ELEMENT    DMSTAG_ELEMENT
#define RIGHT      DMSTAG_RIGHT
#define UP_LEFT    DMSTAG_UP_LEFT
#define UP         DMSTAG_UP
#define UP_RIGHT   DMSTAG_UP_RIGHT

#include "petsc.h"
#include "../fdpde_stokes.h"
#include "../fdpde_advdiff.h"
#include "../fdpde_composite.h"
#include "../dmstagoutput.h"

// ---------------------------------------
// Application Context
// ---------------------------------------
#define FNAME_LENGTH 200

// parameters (bag)
typedef struct {
  PetscInt       nx, nz;
  PetscScalar    L, H;
  PetscScalar    xmin, zmin, xmax, zmax;
  PetscScalar    Ttop, Tbot;
  PetscScalar    b, c, Ra, dT;
  PetscInt       ts_scheme, adv_scheme, test, tout, tstep;
  PetscScalar    t, dt, tmax, dtmax;
  char           fname_out[FNAME_LENGTH]; 
  char           fname_in [FNAME_LENGTH];  
} Params;

// user defined and model-dependent variables
typedef struct {
  Params        *par;
  PetscBag       bag;
  MPI_Comm       comm;
  PetscMPIInt    rank;
  Vec            xTprev;
} UsrData;

// ---------------------------------------
// Function definitions
// ---------------------------------------
PetscErrorCode InputParameters(UsrData**);
PetscErrorCode InputPrintData(UsrData*);

PetscErrorCode Test_composite_convection(void*,PetscInt);

PetscErrorCode FormCoefficient_Stokes_nd1(FDPDE, DM, Vec, DM, Vec, void*);
PetscErrorCode FormCoefficient_Temp_nd1(FDPDE, DM, Vec, DM, Vec, void*);
PetscErrorCode SetInitialTempCoefficient_nd1(DM,Vec,DM,Vec,void*);

PetscErrorCode FormCoefficient_Stokes_nd2(FDPDE, DM, Vec, DM, Vec, void*);
PetscErrorCode FormCoefficient_Temp_nd2(FDPDE, DM, Vec, DM, Vec, void*);
PetscErrorCode SetInitialTempCoefficient_nd2(DM,Vec,DM,Vec,void*);

PetscErrorCode FormBCList_Stokes(DM, Vec, DMStagBCList, void*);
PetscErrorCode FormBCList_Temp(DM, Vec, DMStagBCList, void*);
PetscErrorCode SetInitialTempProfile(DM,Vec,void*);
PetscErrorCode MantleConvectionDiagnostics(DM,Vec,DM,Vec,void*);

// ---------------------------------------
// Some descriptions
// ---------------------------------------
const char coeff_description_stokes_nd1[] =
"  << ND1 Stokes Coefficients >> \n"
"  eta_n/eta_c = eta_eff \n"
"  fux = 0 \n" 
"  fuz = -Ra*T \n"
"  fp = 0 (incompressible)\n"
"  where:\n"
"  eta_eff = eta0/eta0*PetscExpScalar( -b*(T-Ttop)/(Tbot-Ttop) + c*z/H ) \n";

const char coeff_description_temp_nd1[] =
"  << ND1 Temperature Coefficients >> \n"
"  A = 1.0 (element)\n"
"  B = 1.0 (edge)\n"
"  C = 0 (element)\n"
"  u = [ux, uz] (edge) - Stokes velocity \n";

const char coeff_description_stokes_nd2[] =
"  << ND2 Stokes Coefficients >> \n"
"  eta_n/eta_c = eta_eff \n"
"  fux = 0 \n" 
"  fuz = -T \n"
"  fp = 0 (incompressible)\n"
"  where:\n"
"  eta_eff = eta0/eta0*PetscExpScalar( -b*(T-Ttop)/(Tbot-Ttop) + c*z/H ) \n";

const char coeff_description_temp_nd2[] =
"  << ND2 Temperature Coefficients >> \n"
"  A = 1.0 (element)\n"
"  B = 1/Ra (edge)\n"
"  C = 0 (element)\n"
"  u = [ux, uz] (edge) - Stokes velocity \n";

const char bc_description_stokes[] =
"  << Stokes BCs >> \n"
"  LEFT: Vx = 0, dVz/dx = 0 (free slip) \n"
"  RIGHT: Vx = 0, dVz/dx = 0 (free slip) \n" 
"  DOWN: Vz = 0, dVx/dz = 0 (free slip) \n" 
"  UP: Vz = 0, dVx/dz = 0 (free slip) \n";

const char bc_description_temp[] =
"  << Temperature BCs >> \n"
"  LEFT: dT/dx = 0\n"
"  RIGHT: dT/dx = 0\n" 
"  DOWN: T = Tbot\n" 
"  UP: T = Ttop \n";

static PetscScalar EffectiveViscosity(PetscScalar T, PetscScalar Ttop, PetscScalar Tbot, PetscScalar b, PetscScalar c, PetscScalar H, PetscScalar z)
{ return PetscExpScalar( -b*(T-Ttop)/(Tbot-Ttop) + c*z/H ); }

// ---------------------------------------
// Application functions
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "Test_composite_convection"
PetscErrorCode Test_composite_convection(void *ctx, PetscInt nd)
{
  UsrData       *usr = (UsrData*) ctx;
  FDPDE          fd[2], fdmono, *pdes, fdtemp, fdstokes;
  DM             dmPV, dmT, dmTcoeff;
  Vec            x, xPV, xT, xTprev, xTguess, Tcoeff, Tcoeffprev;
  PetscInt       nx, nz, istep = 0, it;
  PetscScalar    xmin, zmin, xmax, zmax, dt_damp;
  char           fout[FNAME_LENGTH];
  PetscBool      converged;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;

  // Element count
  nx = usr->par->nx;
  nz = usr->par->nz;

  // Domain coords
  xmin = usr->par->xmin;
  zmin = usr->par->zmin;
  xmax = usr->par->xmax;
  zmax = usr->par->zmax;

  // Create the sub FD-pde objects
  // 1. Stokes
  ierr = FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_STOKES,&fdstokes);CHKERRQ(ierr);
  ierr = FDPDESetUp(fdstokes);CHKERRQ(ierr);
  ierr = FDPDESetFunctionBCList(fdstokes,FormBCList_Stokes,bc_description_stokes,NULL); CHKERRQ(ierr);

  if (nd==1) {
    ierr = FDPDESetFunctionCoefficient(fdstokes,FormCoefficient_Stokes_nd1,coeff_description_stokes_nd1,usr); CHKERRQ(ierr);
  } else {
    ierr = FDPDESetFunctionCoefficient(fdstokes,FormCoefficient_Stokes_nd2,coeff_description_stokes_nd2,usr); CHKERRQ(ierr);
  }

  ierr = SNESSetFromOptions(fdstokes->snes); CHKERRQ(ierr);
  ierr = FDPDEGetDM(fdstokes,&dmPV); CHKERRQ(ierr);

  // 2. Temperature (Advection-diffusion)
  ierr = FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_ADVDIFF,&fdtemp);CHKERRQ(ierr);
  ierr = FDPDESetUp(fdtemp);CHKERRQ(ierr);

  if (usr->par->adv_scheme==0) { ierr = FDPDEAdvDiffSetAdvectSchemeType(fdtemp,ADV_UPWIND);CHKERRQ(ierr); }
  if (usr->par->adv_scheme==1) { ierr = FDPDEAdvDiffSetAdvectSchemeType(fdtemp,ADV_FROMM);CHKERRQ(ierr); }
  
  // Set timestep and time-stepping scheme
  if (usr->par->ts_scheme ==  0) { ierr = FDPDEAdvDiffSetTimeStepSchemeType(fdtemp,TS_FORWARD_EULER);CHKERRQ(ierr); }
  if (usr->par->ts_scheme ==  1) { ierr = FDPDEAdvDiffSetTimeStepSchemeType(fdtemp,TS_BACKWARD_EULER);CHKERRQ(ierr); }
  if (usr->par->ts_scheme ==  2) { ierr = FDPDEAdvDiffSetTimeStepSchemeType(fdtemp,TS_CRANK_NICHOLSON );CHKERRQ(ierr);}

  ierr = FDPDESetFunctionBCList(fdtemp,FormBCList_Temp,bc_description_temp,usr); CHKERRQ(ierr);

  if (nd==1) {
    ierr = FDPDESetFunctionCoefficient(fdtemp,FormCoefficient_Temp_nd1,coeff_description_temp_nd1,usr); CHKERRQ(ierr);
  } else {
    ierr = FDPDESetFunctionCoefficient(fdtemp,FormCoefficient_Temp_nd2,coeff_description_temp_nd2,usr); CHKERRQ(ierr);
  }
  ierr = SNESSetFromOptions(fdtemp->snes); CHKERRQ(ierr);

  // Set initial temperature profile into xT, Tcoeff
  ierr = FDPDEGetDM(fdtemp,&dmT); CHKERRQ(ierr);
  ierr = FDPDEGetSolution(fdtemp,&xT);CHKERRQ(ierr);
  ierr = SetInitialTempProfile(dmT,xT,usr);CHKERRQ(ierr);

  ierr = FDPDEGetSolutionGuess(fdtemp,&xTguess);CHKERRQ(ierr);
  ierr = VecCopy(xT,xTguess);CHKERRQ(ierr);
  ierr = VecDestroy(&xTguess);CHKERRQ(ierr);

  ierr = FDPDEGetCoefficient(fdtemp,&dmTcoeff,&Tcoeff);CHKERRQ(ierr);

  if (nd==1) {
    ierr = SetInitialTempCoefficient_nd1(dmT,xT,dmTcoeff,Tcoeff,usr);CHKERRQ(ierr);
  } else {
    ierr = SetInitialTempCoefficient_nd2(dmT,xT,dmTcoeff,Tcoeff,usr);CHKERRQ(ierr);
  }

  ierr = VecDuplicate(xT,&usr->xTprev);CHKERRQ(ierr);
  ierr = VecDestroy(&xT);CHKERRQ(ierr);

  // Coupled system - save fd-pdes in array for composite form
  fd[0] = fdstokes;
  fd[1] = fdtemp;

  // Create the composite FD-PDE
  ierr = FDPDECreate2(usr->comm,&fdmono);CHKERRQ(ierr);
  ierr = FDPDESetType(fdmono,FDPDE_COMPOSITE);CHKERRQ(ierr);
  ierr = FDPDECompositeSetFDPDE(fdmono,2,fd);CHKERRQ(ierr);
  ierr = FDPDESetUp(fdmono);CHKERRQ(ierr);
  ierr = FDPDEView(fdmono); CHKERRQ(ierr);
  ierr = SNESSetFromOptions(fdmono->snes); CHKERRQ(ierr);

  // Destroy individual FD-PDE objects
  ierr = FDPDEDestroy(&fd[0]);CHKERRQ(ierr);
  ierr = FDPDEDestroy(&fd[1]);CHKERRQ(ierr);

  dt_damp = 1.0e-2;

  // Time loop
  while ((usr->par->t <= usr->par->tmax) && (istep<=usr->par->tstep)) {
    PetscPrintf(PETSC_COMM_WORLD,"# TIMESTEP %d: \n",istep);

    // Set dt for temperature advection 
    ierr = FDPDECompositeGetFDPDE(fdmono,NULL,&pdes);CHKERRQ(ierr);

    if (istep == 0) { // first timestep
      usr->par->dt = usr->par->dtmax*dt_damp*dt_damp;
    } else {
      PetscScalar dt;
      ierr = FDPDEAdvDiffComputeExplicitTimestep(pdes[1],&dt);CHKERRQ(ierr);
      usr->par->dt = PetscMin(dt,usr->par->dtmax);
      PetscPrintf(PETSC_COMM_WORLD,"# dt = %1.12e dtmax = %1.12e \n",dt,usr->par->dtmax);
    }
    ierr = FDPDEAdvDiffSetTimestep(pdes[1],usr->par->dt);CHKERRQ(ierr);

    // Temperature: copy new solution and coefficient to previous solution
    ierr = FDPDEGetSolution(pdes[1],&xT);CHKERRQ(ierr);
    ierr = FDPDEAdvDiffGetPrevSolution(pdes[1],&xTprev);CHKERRQ(ierr);
    ierr = VecCopy(xT,xTprev);CHKERRQ(ierr);
    ierr = VecCopy(xTprev,usr->xTprev);CHKERRQ(ierr);
    ierr = VecDestroy(&xTprev);CHKERRQ(ierr);
    ierr = VecDestroy(&xT);CHKERRQ(ierr);

    ierr = FDPDEGetCoefficient(pdes[1],&dmTcoeff,&Tcoeff);CHKERRQ(ierr);
    ierr = FDPDEAdvDiffGetPrevCoefficient(pdes[1],&Tcoeffprev);CHKERRQ(ierr);
    ierr = VecCopy(Tcoeff,Tcoeffprev);CHKERRQ(ierr);
    ierr = VecDestroy(&Tcoeffprev);CHKERRQ(ierr);

    // FD SNES Solver
    it = 0;
    converged = PETSC_FALSE;
    while (!converged) {
      // PetscPrintf(PETSC_COMM_WORLD,"# XMONO guess: \n");
      // ierr = VecView(fdmono->xguess,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
      PetscPrintf(PETSC_COMM_WORLD,"# Solver iteration %d \n",it);
      ierr = FDPDESolve(fdmono,&converged);CHKERRQ(ierr);
      // PetscPrintf(PETSC_COMM_WORLD,"# XMONO: %d \n",converged);
      // ierr = VecView(fdmono->x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

      // plot solution values
      /*{
        Vec  xlocal, xTlocal;
        PetscInt  i, j;

        // Get global mono solution
        ierr = FDPDEGetSolution(fdmono,&x);CHKERRQ(ierr);
        ierr = FDPDECompositeSynchronizeGlobalVectors(fdmono,x);CHKERRQ(ierr);

        // Get separate solutions
        ierr = FDPDEGetSolution(pdes[0],&xPV);CHKERRQ(ierr);
        ierr = FDPDEGetSolution(pdes[1],&xT);CHKERRQ(ierr);

        ierr = DMGetLocalVector(dmPV, &xlocal); CHKERRQ(ierr);
        ierr = DMGlobalToLocal (dmPV, xPV, INSERT_VALUES, xlocal); CHKERRQ(ierr);
        ierr = DMGetLocalVector(dmT, &xTlocal); CHKERRQ(ierr);
        ierr = DMGlobalToLocal (dmT, xT, INSERT_VALUES, xTlocal); CHKERRQ(ierr);

        for (j = 0; j<nz; j++) {
          for (i =0; i<nx; i++) {
            DMStagStencil  point;
            PetscScalar    x0,x1,x2,x3,x4,x5;

            point.i = i; point.j = j; point.loc = DMSTAG_LEFT; point.c = 0;
            ierr = DMStagVecGetValuesStencil(dmPV,xlocal,1,&point,&x0); CHKERRQ(ierr);
            point.i = i; point.j = j; point.loc = DMSTAG_RIGHT; point.c = 0;
            ierr = DMStagVecGetValuesStencil(dmPV,xlocal,1,&point,&x1); CHKERRQ(ierr);
            point.i = i; point.j = j; point.loc = DMSTAG_DOWN; point.c = 0;
            ierr = DMStagVecGetValuesStencil(dmPV,xlocal,1,&point,&x2); CHKERRQ(ierr);
            point.i = i; point.j = j; point.loc = DMSTAG_UP; point.c = 0;
            ierr = DMStagVecGetValuesStencil(dmPV,xlocal,1,&point,&x3); CHKERRQ(ierr);
            point.i = i; point.j = j; point.loc = DMSTAG_ELEMENT; point.c = 0;
            ierr = DMStagVecGetValuesStencil(dmPV,xlocal,1,&point,&x4); CHKERRQ(ierr);
            point.i = i; point.j = j; point.loc = DMSTAG_ELEMENT; point.c = 0;
            ierr = DMStagVecGetValuesStencil(dmT,xTlocal,1,&point,&x5); CHKERRQ(ierr);
            PetscPrintf(PETSC_COMM_WORLD,"# [i=%d, j=%d] Stokes: L=%1.12e R=%1.12e D=%1.12e U=%1.12e E=%1.12e Temp: E=%1.12e\n",i,j,x0,x1,x2,x3,x4,x5);

          }
        }

        ierr = DMRestoreLocalVector(dmPV,&xlocal); CHKERRQ(ierr);
        ierr = DMRestoreLocalVector(dmT,&xTlocal); CHKERRQ(ierr);

        ierr = VecDestroy(&xPV);CHKERRQ(ierr);
        ierr = VecDestroy(&xT);CHKERRQ(ierr);
      }*/

      /*{
        PetscPrintf(PETSC_COMM_WORLD,"# PV JACOBIAN:\n");
        ierr = MatView(pdes[0]->J,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
        PetscPrintf(PETSC_COMM_WORLD,"# TEMP JACOBIAN:\n");
        ierr = MatView(pdes[1]->J,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
        PetscPrintf(PETSC_COMM_WORLD,"# FULL JACOBIAN:\n");
        ierr = MatView(fdmono->J,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
     }*/

      if (!converged) { // Reduce dt if not converged
        usr->par->dt *= dt_damp;
        ierr = FDPDEAdvDiffSetTimestep(pdes[1],usr->par->dt);CHKERRQ(ierr);
        it++;
      }
    }

    // Update time
    usr->par->t += usr->par->dt;
    PetscPrintf(PETSC_COMM_WORLD,"# TIME: time = %1.12e dt = %1.12e \n",usr->par->t,usr->par->dt);

    // Get global mono solution
    ierr = FDPDEGetSolution(fdmono,&x);CHKERRQ(ierr);
    ierr = FDPDECompositeSynchronizeGlobalVectors(fdmono,x);CHKERRQ(ierr);

    // Get separate solutions
    ierr = FDPDEGetSolution(pdes[0],&xPV);CHKERRQ(ierr);
    ierr = FDPDEGetSolution(pdes[1],&xT);CHKERRQ(ierr);

    // Calculate diagnostics
    ierr = MantleConvectionDiagnostics(dmPV,xPV,dmT,xT,usr); CHKERRQ(ierr);

    // Output solution
    if (istep % usr->par->tout == 0 ) {
      ierr = PetscSNPrintf(fout,sizeof(fout),"%s_PV_m%d_ts%1.3d",usr->par->fname_out,usr->par->ts_scheme,istep);
      ierr = DMStagViewBinaryPython(dmPV,xPV,fout);CHKERRQ(ierr);
      ierr = PetscSNPrintf(fout,sizeof(fout),"%s_T_m%d_ts%1.3d",usr->par->fname_out,usr->par->ts_scheme,istep);
      ierr = DMStagViewBinaryPython(dmT,xT,fout);CHKERRQ(ierr);
    }

    // Clean up
    ierr = VecDestroy(&x);CHKERRQ(ierr);
    ierr = VecDestroy(&xPV);CHKERRQ(ierr);
    ierr = VecDestroy(&xT);CHKERRQ(ierr);

    // increment timestep
    istep++;
  }

  ierr = DMDestroy(&dmPV); CHKERRQ(ierr);
  ierr = DMDestroy(&dmT); CHKERRQ(ierr);
  ierr = FDPDEDestroy(&fdmono);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xTprev);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

// ---------------------------------------
// MantleConvectionDiagnostics
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "MantleConvectionDiagnostics"
PetscErrorCode MantleConvectionDiagnostics(DM dmPV, Vec xPV, DM dmT, Vec xT, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  Vec            xPVlocal, xTlocal;
  PetscInt       i,j, sx, sz, nx, nz, icenter, iprev, inext, Nx, Nz;
  PetscScalar    **coordx, **coordz;
  PetscScalar    Nu, lT[2], gT[2], dx, dz, L, H;
  PetscScalar    vrms, lvrms, gvrms, q;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  PetscPrintf(usr->comm,"# Mantle convection diagnostics: \n");
  PetscPrintf(usr->comm,"# Rayleigh number: Ra = %1.12e \n",usr->par->Ra);

  // Parameters
  L = usr->par->L;
  H = usr->par->H;
  lT[0] = 0.0;
  lT[1] = 0.0;

  // Get domain corners
  ierr = DMStagGetCorners(dmPV, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = DMStagGetGlobalSizes(dmPV,&Nx,&Nz,NULL);CHKERRQ(ierr);

  // Get dm coordinates array
  ierr = DMStagGet1dCoordinateArraysDOFRead(dmPV,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGet1dCoordinateLocationSlot(dmPV,ELEMENT,&icenter);CHKERRQ(ierr);
  ierr = DMStagGet1dCoordinateLocationSlot(dmPV,LEFT,&iprev);CHKERRQ(ierr);
  ierr = DMStagGet1dCoordinateLocationSlot(dmPV,RIGHT,&inext);CHKERRQ(ierr);

  // Parameters
  dx = coordx[0][inext]-coordx[0][iprev];
  dz = coordz[0][inext]-coordz[0][iprev];

  // Map all vectors to local domain - xPV, xT, xOut
  ierr = DMGetLocalVector(dmPV,&xPVlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dmPV,xPV,INSERT_VALUES,xPVlocal); CHKERRQ(ierr);

  ierr = DMGetLocalVector(dmT,&xTlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dmT,xT,INSERT_VALUES,xTlocal); CHKERRQ(ierr);

  // 1. Nusselt number 
  j = sz; 
  if (j==0) { // bottom mean temperature
    for (i = sx; i<sx+nx; i++) {
      DMStagStencil point;
      PetscScalar   T;

      point.i = i; point.j = j; point.loc = ELEMENT; point.c = 0;
      ierr = DMStagVecGetValuesStencil(dmT,xTlocal,1,&point,&T); CHKERRQ(ierr);
      lT[0] += T*dx;
    }
  }

  j = sz+nz-1; 
  if (j==Nz-1) { // surface temp gradient
    for (i = sx; i<sx+nx; i++) {
      DMStagStencil point[2];
      PetscScalar   T[2];

      point[0].i = i; point[0].j = j  ; point[0].loc = ELEMENT; point[0].c = 0;
      point[1].i = i; point[1].j = j-1; point[1].loc = ELEMENT; point[1].c = 0;
      ierr = DMStagVecGetValuesStencil(dmT,xTlocal,2,point,T); CHKERRQ(ierr);
      lT[1] += (T[0]-T[1])/dz*dx;
    }
  }

  ierr = MPI_Allreduce(&lT, &gT, 2, MPI_DOUBLE, MPI_SUM, usr->comm); CHKERRQ(ierr);

  // Nusselt number
  Nu = -H*gT[1]/gT[0];
  PetscPrintf(usr->comm,"# Nusselt number: Nu = %1.12e \n",Nu);

  // 2. Root-mean-square velocity (vrms)
  lvrms = 0.0;
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      DMStagStencil point[4];
      PetscScalar   v[4],vx,vz;

      point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 0;
      point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 0;
      point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = 0;
      point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = 0;
      ierr = DMStagVecGetValuesStencil(dmPV,xPVlocal,4,point,v); CHKERRQ(ierr);

      vx = (v[0]+v[1])*0.5;
      vz = (v[2]+v[3])*0.5;
      lvrms += (vx*vx+vz*vz)*dx*dz;
    }
  }
  ierr = MPI_Allreduce(&lvrms, &gvrms, 1, MPI_DOUBLE, MPI_SUM, usr->comm); CHKERRQ(ierr);

  // Vrms
  vrms = PetscSqrtReal(gvrms/H/L);
  PetscPrintf(usr->comm,"# Root-mean-squared velocity: vrms = %1.12e \n",vrms);

  // Non-dimensional temperature gradients at the corners of the cells
  i = sx;
  j = sz; 
  if ((i==0) && (j==0)) {
    DMStagStencil point[2];
    PetscScalar   T[2];

    point[0].i = i; point[0].j = j  ; point[0].loc = ELEMENT; point[0].c = 0;
    point[1].i = i; point[1].j = j+1; point[1].loc = ELEMENT; point[1].c = 0;
    ierr = DMStagVecGetValuesStencil(dmT,xTlocal,2,point,T); CHKERRQ(ierr);
    q = -H/usr->par->dT*(T[1]-T[0])/dz;
    PetscPrintf(usr->comm,"# Corner flux (down-left): q1 = %1.12e \n",q);
  }

  i = sx;
  j = sz+nz-1; 
  if ((i==0) && (j==Nz-1)) {
    DMStagStencil point[2];
    PetscScalar   T[2];

    point[0].i = i; point[0].j = j  ; point[0].loc = ELEMENT; point[0].c = 0;
    point[1].i = i; point[1].j = j-1; point[1].loc = ELEMENT; point[1].c = 0;
    ierr = DMStagVecGetValuesStencil(dmT,xTlocal,2,point,T); CHKERRQ(ierr);
    q = -H/usr->par->dT*(T[0]-T[1])/dz;
    PetscPrintf(usr->comm,"# Corner flux (up-left): q2 = %1.12e \n",q);
  } 

  i = sx+nx-1;
  j = sz; 
  if ((i==Nx-1) && (j==0)) {
    DMStagStencil point[2];
    PetscScalar   T[2];

    point[0].i = i; point[0].j = j  ; point[0].loc = ELEMENT; point[0].c = 0;
    point[1].i = i; point[1].j = j+1; point[1].loc = ELEMENT; point[1].c = 0;
    ierr = DMStagVecGetValuesStencil(dmT,xTlocal,2,point,T); CHKERRQ(ierr);
    q = -H/usr->par->dT*(T[1]-T[0])/dz;
    PetscPrintf(usr->comm,"# Corner flux (down-right): q3 = %1.12e \n",q);
  } 

  i = sx+nx-1;
  j = sz+nz-1; 
  if ((i==Nx-1) && (j==Nz-1)) {
    DMStagStencil point[2];
    PetscScalar   T[2];

    point[0].i = i; point[0].j = j  ; point[0].loc = ELEMENT; point[0].c = 0;
    point[1].i = i; point[1].j = j-1; point[1].loc = ELEMENT; point[1].c = 0;
    ierr = DMStagVecGetValuesStencil(dmT,xTlocal,2,point,T); CHKERRQ(ierr);
    q = -H/usr->par->dT*(T[0]-T[1])/dz;
    PetscPrintf(usr->comm,"# Corner flux (up-right): q4 = %1.12e \n",q);
  } 

  // Restore arrays, local vectors
  ierr = DMStagRestore1dCoordinateArraysDOFRead(dmPV,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmPV,&xPVlocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmT,&xTlocal); CHKERRQ(ierr);

  PetscPrintf(usr->comm,"\n");

  PetscFunctionReturn(0);
}

// ---------------------------------------
// SetInitialTempProfile - T(x,z) = 0.05*cos(pi*x)*sin(pi*z)
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "SetInitialTempProfile"
PetscErrorCode SetInitialTempProfile(DM dm, Vec x, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  Vec            xlocal;
  PetscInt       i,j, sx, sz, nx, nz, icenter;
  PetscScalar    ***xx, **coordx, **coordz;
  PetscScalar    Ttop, Tbot, a, p;

  PetscErrorCode ierr;
  PetscFunctionBegin;

  // Parameters
  Ttop = usr->par->Ttop;
  Tbot = usr->par->Tbot;
  a = (Ttop-Tbot)/(usr->par->zmax-usr->par->zmin);
  p = 0.05;

  // Get domain corners
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  // Get dm coordinates array
  ierr = DMStagGet1dCoordinateArraysDOFRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGet1dCoordinateLocationSlot(dm,ELEMENT,&icenter);CHKERRQ(ierr);

  // Create local vector
  ierr = DMCreateLocalVector(dm, &xlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayDOF(dm, xlocal, &xx); CHKERRQ(ierr);

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      DMStagStencil point;
      PetscScalar   xp,zp;
      PetscInt      idx;

      point.i = i; point.j = j; point.loc = ELEMENT; point.c = 0;
      xp = coordx[i][icenter];
      zp = coordz[j][icenter];

      ierr = DMStagGetLocationSlot(dm, point.loc, point.c, &idx); CHKERRQ(ierr);
      xx[j][i][idx] = Tbot+a*zp + p*PetscCosScalar(PETSC_PI*xp)*PetscSinScalar(PETSC_PI*zp);
    }
  }

  // Restore arrays, local vectors
  ierr = DMStagRestore1dCoordinateArraysDOFRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);

  ierr = DMStagVecRestoreArrayDOF(dm,xlocal,&xx);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm,xlocal,INSERT_VALUES,x); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,xlocal,INSERT_VALUES,x); CHKERRQ(ierr);
  
  ierr = VecDestroy(&xlocal); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// SetInitialTempCoefficient_nd1
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "SetInitialTempCoefficient_nd1"
PetscErrorCode SetInitialTempCoefficient_nd1(DM dm, Vec x, DM dmcoeff, Vec coeff, void *ctx)
{
  // UsrData        *usr = (UsrData*)ctx;
  PetscInt       i, j, sx, sz, nx, nz;
  Vec            coefflocal;
  PetscScalar    ***c;
  Vec            xlocal;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  // Get solution vector for temperature
  ierr = DMGetLocalVector(dm,&xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm,x,INSERT_VALUES,xlocal); CHKERRQ(ierr);

  // Get domain corners
  ierr = DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  // Create coefficient local vector
  ierr = DMCreateLocalVector(dmcoeff, &coefflocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayDOF(dmcoeff, coefflocal, &c); CHKERRQ(ierr);
  
  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {

      { // A = 1.0
        DMStagStencil point;
        PetscInt      idx;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 0;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = 1.0;
      }

      { // C = 0.0 - sources of heat/sink
        DMStagStencil point;
        PetscInt      idx;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 1;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = 0.0;
      }

      { // B = 1.0 (edge)
        DMStagStencil point[4];
        PetscInt      ii, idx;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 0;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = 0;
        point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = 0;

        for (ii = 0; ii < 4; ii++) {
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, 0, &idx); CHKERRQ(ierr);
          c[j][i][idx] = 1.0;
        }
      }

      { // u = velocity (edge) = 0.0 (initial)
        DMStagStencil point[4];
        PetscInt      ii, idx;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 1;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 1;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = 1;
        point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = 1;

        for (ii = 0; ii < 4; ii++) {
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, 1, &idx); CHKERRQ(ierr);
          c[j][i][idx] = 0.0;
        }
      }
    }
  }

  // Restore arrays, local vectors
  ierr = DMStagVecRestoreArrayDOF(dmcoeff,coefflocal,&c);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  ierr = VecDestroy(&coefflocal); CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(dm,&xlocal); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// SetInitialTempCoefficient_nd2
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "SetInitialTempCoefficient_nd2"
PetscErrorCode SetInitialTempCoefficient_nd2(DM dm, Vec x, DM dmcoeff, Vec coeff, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       i, j, sx, sz, nx, nz;
  Vec            coefflocal;
  PetscScalar    ***c;
  Vec            xlocal;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  // Get solution vector for temperature
  ierr = DMGetLocalVector(dm,&xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm,x,INSERT_VALUES,xlocal); CHKERRQ(ierr);

  // Get domain corners
  ierr = DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  // Create coefficient local vector
  ierr = DMCreateLocalVector(dmcoeff, &coefflocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayDOF(dmcoeff, coefflocal, &c); CHKERRQ(ierr);
  
  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {

      { // A = 1.0
        DMStagStencil point;
        PetscInt      idx;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 0;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = 1.0;
      }

      { // C = 0.0 - sources of heat/sink
        DMStagStencil point;
        PetscInt      idx;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 1;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = 0.0;
      }

      { // B = 1.0/Ra (edge)
        DMStagStencil point[4];
        PetscInt      ii, idx;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 0;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = 0;
        point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = 0;

        for (ii = 0; ii < 4; ii++) {
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, 0, &idx); CHKERRQ(ierr);
          c[j][i][idx] = 1.0/usr->par->Ra;
        }
      }

      { // u = velocity (edge) = 0.0 (initial)
        DMStagStencil point[4];
        PetscInt      ii, idx;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 1;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 1;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = 1;
        point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = 1;

        for (ii = 0; ii < 4; ii++) {
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, 1, &idx); CHKERRQ(ierr);
          c[j][i][idx] = 0.0;
        }
      }
    }
  }

  // Restore arrays, local vectors
  ierr = DMStagVecRestoreArrayDOF(dmcoeff,coefflocal,&c);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  ierr = VecDestroy(&coefflocal); CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(dm,&xlocal); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// FormCoefficient_Stokes_nd1
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormCoefficient_Stokes_nd1"
PetscErrorCode FormCoefficient_Stokes_nd1(FDPDE fd, DM dm, Vec x, DM dmcoeff, Vec coeff, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  DM             dmT = NULL;
  Vec            *aux_vecs, xT = NULL, xTlocal, xlocal;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz, naux;
  Vec            coefflocal;
  PetscScalar    **coordx,**coordz;
  PetscInt       iprev, inext, icenter;
  PetscScalar    ***cx;
  PetscScalar    eta, Ttop, Tbot,b,c, H, Ra;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  // User parameters
  Ttop  = usr->par->Ttop;
  Tbot  = usr->par->Tbot;
  b     = usr->par->b;
  c     = usr->par->c;
  H     = usr->par->H;
  Ra    = usr->par->Ra;
  
  // Get dm and solution vector for Temperature
  ierr = FDPDEGetAuxGlobalVectors(fd,&naux,&aux_vecs);CHKERRQ(ierr);  
  xT = aux_vecs[1];
  // xT  = usr->xTprev; // may pass previous solution, to remove non-linear coupling in Stokes
  ierr = VecGetDM(aux_vecs[1],&dmT); CHKERRQ(ierr);
  if (!dmT) SETERRQ(fd->comm,PETSC_ERR_USER,"Expected to obtain a non-NULL value for dmT");
  if (!xT) SETERRQ(fd->comm,PETSC_ERR_USER,"Expected to obtain a non-NULL value for xT");

  ierr = DMCreateLocalVector(dmT,&xTlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dmT,xT,INSERT_VALUES,xTlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dmT,xT,INSERT_VALUES,xTlocal);CHKERRQ(ierr);
  
  // Get solution vector for Stokes
  ierr = DMGetLocalVector(dm,&xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm,x,INSERT_VALUES,xlocal); CHKERRQ(ierr);

  // Get domain corners
  ierr = DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = DMStagGetGlobalSizes(dmcoeff,&Nx,&Nz,NULL);CHKERRQ(ierr);
  
  // Get dmcoeff coordinates array
  ierr = DMStagGet1dCoordinateArraysDOFRead(dmcoeff,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGet1dCoordinateLocationSlot(dmcoeff,LEFT,&iprev);CHKERRQ(ierr);
  ierr = DMStagGet1dCoordinateLocationSlot(dmcoeff,RIGHT,&inext);CHKERRQ(ierr);
  ierr = DMStagGet1dCoordinateLocationSlot(dmcoeff,ELEMENT,&icenter);CHKERRQ(ierr);

  // Create coefficient local vector
  ierr = DMCreateLocalVector(dmcoeff, &coefflocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayDOF(dmcoeff, coefflocal, &cx); CHKERRQ(ierr);
  
  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {

      { // fux = 0.0
        DMStagStencil point[2];
        PetscInt      ii, idx;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 0;

        for (ii = 0; ii < 2; ii++) {
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          cx[j][i][idx] = 0.0;
        }
      }

      { // fuz = -Ra*T - need to interpolate temp to Vz points
        PetscScalar   T[3], Tinterp;
        DMStagStencil point[2], pointT[3];
        PetscInt      ii, idx;

        point[0].i = i; point[0].j = j; point[0].loc = DOWN; point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = UP;   point[1].c = 0;
        
        pointT[0].i = i; pointT[0].j = j-1; pointT[0].loc = ELEMENT; pointT[0].c = 0;
        pointT[1].i = i; pointT[1].j = j  ; pointT[1].loc = ELEMENT; pointT[1].c = 0;
        pointT[2].i = i; pointT[2].j = j+1; pointT[2].loc = ELEMENT; pointT[2].c = 0;
        
        // take into account domain borders
        if (j == 0   ) pointT[0] = pointT[1];
        if (j == Nz-1) pointT[2] = pointT[1];
        
        ierr = DMStagVecGetValuesStencil(dmT,xTlocal,3,pointT,T); CHKERRQ(ierr);

        for (ii = 0; ii < 2; ii++) {
          Tinterp = (T[ii]+T[ii+1])*0.5; // assume constant grid spacing
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          cx[j][i][idx] = -Ra*Tinterp;
          // PetscPrintf(PETSC_COMM_WORLD,"# [%d,%d,%d] Tinterp = %1.12e cx = %1.12e\n",i,j,idx,Tinterp,cx[j][i][idx]);
        }
      }

      { // fp = 0.0
        DMStagStencil point;
        PetscInt      idx;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 0;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        cx[j][i][idx] = 0.0;
      }

      { // eta_c = eta_eff
        PetscScalar   zp, T;
        DMStagStencil point, pointT;
        PetscInt      idx;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 1;
        pointT = point; pointT.c = 0;
        ierr = DMStagVecGetValuesStencil(dmT,xTlocal,1,&pointT,&T); CHKERRQ(ierr);
        
        zp = coordz[j][icenter];
        eta = EffectiveViscosity(T,Ttop,Tbot,b,c,H,zp);

        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        cx[j][i][idx] = eta;
      }

      { // eta_n = eta_eff - need to interpolate Temp at corners
        DMStagStencil point[4], pointT[9];
        PetscScalar   zp[4], T[9], Tinterp[4];
        PetscInt      ii, idx;

        point[0].i = i; point[0].j = j; point[0].loc = DOWN_LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = DOWN_RIGHT; point[1].c = 0;
        point[2].i = i; point[2].j = j; point[2].loc = UP_LEFT;    point[2].c = 0;
        point[3].i = i; point[3].j = j; point[3].loc = UP_RIGHT;   point[3].c = 0;
        
        pointT[0].i = i-1; pointT[0].j = j-1; pointT[0].loc = ELEMENT; pointT[0].c = 0;
        pointT[1].i = i  ; pointT[1].j = j-1; pointT[1].loc = ELEMENT; pointT[1].c = 0;
        pointT[2].i = i+1; pointT[2].j = j-1; pointT[2].loc = ELEMENT; pointT[2].c = 0;
        pointT[3].i = i-1; pointT[3].j = j  ; pointT[3].loc = ELEMENT; pointT[3].c = 0;
        pointT[4].i = i  ; pointT[4].j = j  ; pointT[4].loc = ELEMENT; pointT[4].c = 0;
        pointT[5].i = i+1; pointT[5].j = j  ; pointT[5].loc = ELEMENT; pointT[5].c = 0;
        pointT[6].i = i-1; pointT[6].j = j+1; pointT[6].loc = ELEMENT; pointT[6].c = 0;
        pointT[7].i = i  ; pointT[7].j = j+1; pointT[7].loc = ELEMENT; pointT[7].c = 0;
        pointT[8].i = i+1; pointT[8].j = j+1; pointT[8].loc = ELEMENT; pointT[8].c = 0;
        
        // take into account domain borders
        if (i == 0   ) { pointT[0] = pointT[1]; pointT[3] = pointT[4]; pointT[6] = pointT[7]; }
        if (i == Nx-1) { pointT[2] = pointT[1]; pointT[5] = pointT[4]; pointT[8] = pointT[7]; }
        if (j == 0   ) { pointT[0] = pointT[3]; pointT[1] = pointT[4]; pointT[2] = pointT[5]; }
        if (j == Nz-1) { pointT[6] = pointT[3]; pointT[7] = pointT[4]; pointT[8] = pointT[5]; }
        
        if ((i == 0   ) && (j == 0   )) pointT[0] = pointT[4];
        if ((i == 0   ) && (j == Nz-1)) pointT[6] = pointT[4];
        if ((i == Nx-1) && (j == 0   )) pointT[2] = pointT[4];
        if ((i == Nx-1) && (j == Nz-1)) pointT[8] = pointT[4];
        
        ierr = DMStagVecGetValuesStencil(dmT,xTlocal,9,pointT,T); CHKERRQ(ierr);
        
        Tinterp[0] = (T[0]+T[1]+T[3]+T[4])*0.25; // assume constant grid spacing
        Tinterp[1] = (T[1]+T[2]+T[4]+T[5])*0.25;
        Tinterp[2] = (T[3]+T[4]+T[6]+T[7])*0.25;
        Tinterp[3] = (T[4]+T[5]+T[7]+T[8])*0.25;

        zp[0] = coordz[j][iprev];
        zp[1] = coordz[j][iprev];
        zp[2] = coordz[j][inext];
        zp[3] = coordz[j][inext];

        for (ii = 0; ii < 4; ii++) {
          eta = EffectiveViscosity(Tinterp[ii],Ttop,Tbot,b,c,H,zp[ii]);
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          cx[j][i][idx] = eta;
        }
      }
    }
  }

  // Restore arrays, local vectors
  ierr = DMStagRestore1dCoordinateArraysDOFRead(dmcoeff,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArrayDOF(dmcoeff,coefflocal,&cx);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  ierr = VecDestroy(&coefflocal); CHKERRQ(ierr);
  
  ierr = DMRestoreLocalVector(dm,&xlocal); CHKERRQ(ierr);
  
  ierr = VecDestroy(&xTlocal);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// FormCoefficient_Stokes_nd2
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormCoefficient_Stokes_nd2"
PetscErrorCode FormCoefficient_Stokes_nd2(FDPDE fd, DM dm, Vec x, DM dmcoeff, Vec coeff, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  DM             dmT = NULL;
  Vec            *aux_vecs, xT = NULL, xTlocal, xlocal;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz, naux;
  Vec            coefflocal;
  PetscScalar    **coordx,**coordz;
  PetscInt       iprev, inext, icenter;
  PetscScalar    ***cx;
  PetscScalar    eta, Ttop, Tbot,b,c, H;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  // User parameters
  Ttop  = usr->par->Ttop;
  Tbot  = usr->par->Tbot;
  b     = usr->par->b;
  c     = usr->par->c;
  H     = usr->par->H;
  
  // Get dm and solution vector for Temperature
  ierr = FDPDEGetAuxGlobalVectors(fd,&naux,&aux_vecs);CHKERRQ(ierr);
  xT = aux_vecs[1];
  ierr = VecGetDM(aux_vecs[1],&dmT); CHKERRQ(ierr);
  if (!dmT) SETERRQ(fd->comm,PETSC_ERR_USER,"Expected to obtain a non-NULL value for dmT");
  if (!xT) SETERRQ(fd->comm,PETSC_ERR_USER,"Expected to obtain a non-NULL value for xT");

  ierr = DMCreateLocalVector(dmT,&xTlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dmT,xT,INSERT_VALUES,xTlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dmT,xT,INSERT_VALUES,xTlocal);CHKERRQ(ierr);
  
  // Get solution vector for Stokes
  ierr = DMGetLocalVector(dm,&xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm,x,INSERT_VALUES,xlocal); CHKERRQ(ierr);

  // Get domain corners
  ierr = DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = DMStagGetGlobalSizes(dmcoeff,&Nx,&Nz,NULL);CHKERRQ(ierr);
  
  // Get dmcoeff coordinates array
  ierr = DMStagGet1dCoordinateArraysDOFRead(dmcoeff,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGet1dCoordinateLocationSlot(dmcoeff,LEFT,&iprev);CHKERRQ(ierr);
  ierr = DMStagGet1dCoordinateLocationSlot(dmcoeff,RIGHT,&inext);CHKERRQ(ierr);
  ierr = DMStagGet1dCoordinateLocationSlot(dmcoeff,ELEMENT,&icenter);CHKERRQ(ierr);

  // Create coefficient local vector
  ierr = DMCreateLocalVector(dmcoeff, &coefflocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayDOF(dmcoeff, coefflocal, &cx); CHKERRQ(ierr);
  
  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {

      { // fux = 0.0
        DMStagStencil point[2];
        PetscInt      ii, idx;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 0;

        for (ii = 0; ii < 2; ii++) {
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          cx[j][i][idx] = 0.0;
        }
      }

      { // fuz = -T - need to interpolate temp to Vz points
        PetscScalar   T[3], Tinterp;
        DMStagStencil point[2], pointT[3];
        PetscInt      ii, idx;

        point[0].i = i; point[0].j = j; point[0].loc = DOWN; point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = UP;   point[1].c = 0;
        
        pointT[0].i = i; pointT[0].j = j-1; pointT[0].loc = ELEMENT; pointT[0].c = 0;
        pointT[1].i = i; pointT[1].j = j  ; pointT[1].loc = ELEMENT; pointT[1].c = 0;
        pointT[2].i = i; pointT[2].j = j+1; pointT[2].loc = ELEMENT; pointT[2].c = 0;
        
        // take into account domain borders
        if (j == 0   ) pointT[0] = pointT[1];
        if (j == Nz-1) pointT[2] = pointT[1];
        
        ierr = DMStagVecGetValuesStencil(dmT,xTlocal,3,pointT,T); CHKERRQ(ierr);

        for (ii = 0; ii < 2; ii++) {
          Tinterp = (T[ii]+T[ii+1])*0.5; // assume constant grid spacing
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          cx[j][i][idx] = -Tinterp;
        }
      }

      { // fp = 0.0
        DMStagStencil point;
        PetscInt      idx;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 0;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        cx[j][i][idx] = 0.0;
      }

      { // eta_c = eta_eff
        PetscScalar   zp, T;
        DMStagStencil point, pointT;
        PetscInt      idx;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 1;
        pointT = point; pointT.c = 0;
        ierr = DMStagVecGetValuesStencil(dmT,xTlocal,1,&pointT,&T); CHKERRQ(ierr);
        
        zp = coordz[j][icenter];
        eta = EffectiveViscosity(T,Ttop,Tbot,b,c,H,zp);

        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        cx[j][i][idx] = eta;
      }

      { // eta_n = eta_eff - need to interpolate Temp at corners
        DMStagStencil point[4], pointT[9];
        PetscScalar   zp[4], T[9], Tinterp[4];
        PetscInt      ii, idx;

        point[0].i = i; point[0].j = j; point[0].loc = DOWN_LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = DOWN_RIGHT; point[1].c = 0;
        point[2].i = i; point[2].j = j; point[2].loc = UP_LEFT;    point[2].c = 0;
        point[3].i = i; point[3].j = j; point[3].loc = UP_RIGHT;   point[3].c = 0;
        
        pointT[0].i = i-1; pointT[0].j = j-1; pointT[0].loc = ELEMENT; pointT[0].c = 0;
        pointT[1].i = i  ; pointT[1].j = j-1; pointT[1].loc = ELEMENT; pointT[1].c = 0;
        pointT[2].i = i+1; pointT[2].j = j-1; pointT[2].loc = ELEMENT; pointT[2].c = 0;
        pointT[3].i = i-1; pointT[3].j = j  ; pointT[3].loc = ELEMENT; pointT[3].c = 0;
        pointT[4].i = i  ; pointT[4].j = j  ; pointT[4].loc = ELEMENT; pointT[4].c = 0;
        pointT[5].i = i+1; pointT[5].j = j  ; pointT[5].loc = ELEMENT; pointT[5].c = 0;
        pointT[6].i = i-1; pointT[6].j = j+1; pointT[6].loc = ELEMENT; pointT[6].c = 0;
        pointT[7].i = i  ; pointT[7].j = j+1; pointT[7].loc = ELEMENT; pointT[7].c = 0;
        pointT[8].i = i+1; pointT[8].j = j+1; pointT[8].loc = ELEMENT; pointT[8].c = 0;
        
        // take into account domain borders
        if (i == 0   ) { pointT[0] = pointT[1]; pointT[3] = pointT[4]; pointT[6] = pointT[7]; }
        if (i == Nx-1) { pointT[2] = pointT[1]; pointT[5] = pointT[4]; pointT[8] = pointT[7]; }
        if (j == 0   ) { pointT[0] = pointT[3]; pointT[1] = pointT[4]; pointT[2] = pointT[5]; }
        if (j == Nz-1) { pointT[6] = pointT[3]; pointT[7] = pointT[4]; pointT[8] = pointT[5]; }
        
        if ((i == 0   ) && (j == 0   )) pointT[0] = pointT[4];
        if ((i == 0   ) && (j == Nz-1)) pointT[6] = pointT[4];
        if ((i == Nx-1) && (j == 0   )) pointT[2] = pointT[4];
        if ((i == Nx-1) && (j == Nz-1)) pointT[8] = pointT[4];
        
        ierr = DMStagVecGetValuesStencil(dmT,xTlocal,9,pointT,T); CHKERRQ(ierr);
        
        Tinterp[0] = (T[0]+T[1]+T[3]+T[4])*0.25; // assume constant grid spacing
        Tinterp[1] = (T[1]+T[2]+T[4]+T[5])*0.25;
        Tinterp[2] = (T[3]+T[4]+T[6]+T[7])*0.25;
        Tinterp[3] = (T[4]+T[5]+T[7]+T[8])*0.25;

        zp[0] = coordz[j][iprev];
        zp[1] = coordz[j][iprev];
        zp[2] = coordz[j][inext];
        zp[3] = coordz[j][inext];

        for (ii = 0; ii < 4; ii++) {
          eta = EffectiveViscosity(Tinterp[ii],Ttop,Tbot,b,c,H,zp[ii]);
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          cx[j][i][idx] = eta;
        }
      }
    }
  }

  // Restore arrays, local vectors
  ierr = DMStagRestore1dCoordinateArraysDOFRead(dmcoeff,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArrayDOF(dmcoeff,coefflocal,&cx);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  ierr = VecDestroy(&coefflocal); CHKERRQ(ierr);
  
  ierr = DMRestoreLocalVector(dm,&xlocal); CHKERRQ(ierr);
  
  ierr = VecDestroy(&xTlocal);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// FormBCList_Stokes
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormBCList_Stokes"
PetscErrorCode FormBCList_Stokes(DM dm, Vec x, DMStagBCList bclist, void *ctx)
{
  PetscInt    k,n_bc,*idx_bc;
  PetscScalar *value_bc;
  BCType      *type_bc;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  
  // dVz/dx=0 on left boundary (w)
  ierr = DMStagBCListGetValues(bclist,'w','|',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  
  // dVz/dx=0 on right boundary (e)
  ierr = DMStagBCListGetValues(bclist,'e','|',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  
  // dVx/dz=0 on top boundary (n)
  ierr = DMStagBCListGetValues(bclist,'n','-',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  
  // dVx/dz=0 on bottom boundary (s)
  ierr = DMStagBCListGetValues(bclist,'s','-',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  
  // Vx=0 on left boundary (w)
  ierr = DMStagBCListGetValues(bclist,'w','-',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  
  // Vx=0 on right boundary (e)
  ierr = DMStagBCListGetValues(bclist,'e','-',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  
  // Vz=0 on top boundary (n)
  ierr = DMStagBCListGetValues(bclist,'n','|',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  
  // Vz=0 on bottom boundary (s)
  ierr = DMStagBCListGetValues(bclist,'s','|',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// FormCoefficient_Temp_nd1
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormCoefficient_Temp_nd1"
PetscErrorCode FormCoefficient_Temp_nd1(FDPDE fd, DM dm, Vec x, DM dmcoeff, Vec coeff, void *ctx)
{
  // UsrData        *usr = (UsrData*)ctx;
  PetscInt       i, j, sx, sz, nx, nz, naux, Nx, Nz;
  DM             dmPV = NULL;
  Vec            coefflocal;
  PetscScalar    ***c;
  Vec            *aux_vecs, xPV = NULL, xPVlocal, xlocal;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  // Get dm and solution vector for Stokes velocity
  ierr = FDPDEGetAuxGlobalVectors(fd,&naux,&aux_vecs);CHKERRQ(ierr);
  xPV = aux_vecs[0];
  ierr = VecGetDM(aux_vecs[0],&dmPV);CHKERRQ(ierr);
  if (!dmPV) SETERRQ(fd->comm,PETSC_ERR_USER,"Expected to obtain a non-NULL value for dmPV");
  if (!xPV) SETERRQ(fd->comm,PETSC_ERR_USER,"Expected to obtain a non-NULL value for xPV");
  
  ierr = DMCreateLocalVector(dmPV,&xPVlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dmPV,xPV,INSERT_VALUES,xPVlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dmPV,xPV,INSERT_VALUES,xPVlocal);CHKERRQ(ierr);
  
  // Get solution vector for temperature
  ierr = DMGetLocalVector(dm,&xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm,x,INSERT_VALUES,xlocal); CHKERRQ(ierr);

  // Get domain corners
  ierr = DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = DMStagGetGlobalSizes(dmcoeff,&Nx,&Nz,NULL);CHKERRQ(ierr);

  // Create coefficient local vector
  ierr = DMCreateLocalVector(dmcoeff, &coefflocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayDOF(dmcoeff, coefflocal, &c); CHKERRQ(ierr);
  
  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {

      { // A = 1.0
        DMStagStencil point;
        PetscInt      idx;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 0;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = 1.0;
      }

      { // C = 0.0 - sources of heat/sink
        DMStagStencil point;
        PetscInt      idx;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 1;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = 0.0;
      }

      { // B = 1.0 (edge)
        DMStagStencil point[4];
        PetscInt      ii, idx;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 0;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = 0;
        point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = 0;

        for (ii = 0; ii < 4; ii++) {
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = 1.0;
        }
      }

      { // u = velocity (edge) - Stokes velocity
        PetscScalar   v[4];
        DMStagStencil point[4];
        PetscInt      ii, idx;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 0;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = 0;
        point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = 0;
        
        ierr = DMStagVecGetValuesStencil(dmPV,xPVlocal,4,point,v); CHKERRQ(ierr);

        for (ii = 0; ii < 4; ii++) {
          point[ii].c = 1;
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = v[ii];
        }
      }
    }
  }

  // Restore arrays, local vectors
  ierr = DMStagVecRestoreArrayDOF(dmcoeff,coefflocal,&c);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  ierr = VecDestroy(&coefflocal); CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(dm,&xlocal); CHKERRQ(ierr);
  
  ierr = VecDestroy(&xPVlocal);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// FormCoefficient_Temp_nd2
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormCoefficient_Temp_nd2"
PetscErrorCode FormCoefficient_Temp_nd2(FDPDE fd, DM dm, Vec x, DM dmcoeff, Vec coeff, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       i, j, sx, sz, nx, nz, naux, Nx, Nz;
  DM             dmPV = NULL;
  Vec            coefflocal;
  PetscScalar    ***c;
  Vec            *aux_vecs, xPV = NULL, xPVlocal, xlocal;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  // Get dm and solution vector for Stokes velocity
  ierr = FDPDEGetAuxGlobalVectors(fd,&naux,&aux_vecs);CHKERRQ(ierr);
  xPV = aux_vecs[0];
  ierr = VecGetDM(aux_vecs[0],&dmPV);CHKERRQ(ierr);
  if (!dmPV) SETERRQ(fd->comm,PETSC_ERR_USER,"Expected to obtain a non-NULL value for dmPV");
  if (!xPV) SETERRQ(fd->comm,PETSC_ERR_USER,"Expected to obtain a non-NULL value for xPV");
  
  ierr = DMCreateLocalVector(dmPV,&xPVlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dmPV,xPV,INSERT_VALUES,xPVlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dmPV,xPV,INSERT_VALUES,xPVlocal);CHKERRQ(ierr);
  
  // Get solution vector for temperature
  ierr = DMGetLocalVector(dm,&xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm,x,INSERT_VALUES,xlocal); CHKERRQ(ierr);

  // Get domain corners
  ierr = DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = DMStagGetGlobalSizes(dmcoeff,&Nx,&Nz,NULL);CHKERRQ(ierr);

  // Create coefficient local vector
  ierr = DMCreateLocalVector(dmcoeff, &coefflocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayDOF(dmcoeff, coefflocal, &c); CHKERRQ(ierr);
  
  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {

      { // A = 1.0
        DMStagStencil point;
        PetscInt      idx;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 0;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = 1.0;
      }

      { // C = 0.0 - sources of heat/sink
        DMStagStencil point;
        PetscInt      idx;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 1;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = 0.0;
      }

      { // B = 1.0/Ra (edge)
        DMStagStencil point[4];
        PetscInt      ii, idx;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 0;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = 0;
        point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = 0;

        for (ii = 0; ii < 4; ii++) {
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = 1.0/usr->par->Ra;
        }
      }

      { // u = velocity (edge) - Stokes velocity
        PetscScalar   v[4];
        DMStagStencil point[4];
        PetscInt      ii, idx;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 0;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = 0;
        point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = 0;
        
        ierr = DMStagVecGetValuesStencil(dmPV,xPVlocal,4,point,v); CHKERRQ(ierr);

        for (ii = 0; ii < 4; ii++) {
          point[ii].c = 1;
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = v[ii];
        }
      }
    }
  }

  // Restore arrays, local vectors
  ierr = DMStagVecRestoreArrayDOF(dmcoeff,coefflocal,&c);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  ierr = VecDestroy(&coefflocal); CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(dm,&xlocal); CHKERRQ(ierr);
  
  ierr = VecDestroy(&xPVlocal);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// FormBCList_Temp
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormBCList_Temp"
PetscErrorCode FormBCList_Temp(DM dm, Vec x, DMStagBCList bclist, void *ctx)
{
  UsrData     *usr = (UsrData*)ctx;
  PetscInt    k,n_bc,*idx_bc;
  PetscScalar *value_bc;
  BCType      *type_bc;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  
  // Left: dT/dx=0
  ierr = DMStagBCListGetValues(bclist,'w','o',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  
  // RIGHT: dT/dx=0
  ierr = DMStagBCListGetValues(bclist,'e','o',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN: T = Tbot (Tij = 2/3*Tbot+1/3*Tij+1)
  ierr = DMStagBCListGetValues(bclist,'s','o',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = usr->par->Tbot;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP: T = Ttop (Tij = 2/3*Ttop+1/3*Tij-1)
  ierr = DMStagBCListGetValues(bclist,'n','o',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = usr->par->Ttop;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
 
  PetscFunctionReturn(0);
}

// ---------------------------------------
// InputParameters
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "InputParameters"
PetscErrorCode InputParameters(UsrData **_usr)
{
  UsrData       *usr;
  Params        *par;
  PetscBag       bag;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Allocate memory to application context
  ierr = PetscMalloc1(1, &usr); CHKERRQ(ierr);

  // Get time, comm and rank
  usr->comm = PETSC_COMM_WORLD;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &usr->rank); CHKERRQ(ierr);

  // Create bag
  ierr = PetscBagCreate (usr->comm,sizeof(Params),&usr->bag); CHKERRQ(ierr);
  ierr = PetscBagGetData(usr->bag,(void **)&usr->par); CHKERRQ(ierr);
  ierr = PetscBagSetName(usr->bag,"UserParamBag","- User defined parameters -"); CHKERRQ(ierr);

  // Define some pointers for easy access
  bag = usr->bag;
  par = usr->par;

  // Initialize domain variables
  ierr = PetscBagRegisterInt(bag, &par->nx, 4, "nx", "Element count in the x-dir"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->nz, 5, "nz", "Element count in the z-dir"); CHKERRQ(ierr);

  ierr = PetscBagRegisterScalar(bag, &par->xmin, 0.0, "xmin", "Start coordinate of domain in x-dir [m]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->zmin, 0.0, "zmin", "Start coordinate of domain in z-dir [m]"); CHKERRQ(ierr);

  ierr = PetscBagRegisterScalar(bag, &par->L, 1.0, "L", "Length of domain in x-dir [-]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->H, 1.0, "H", "Height of domain in z-dir [-]"); CHKERRQ(ierr);

  ierr = PetscBagRegisterInt(bag, &par->test,1, "test", "Test case 0-decoupled nd1, 1-composite nd1, 2-composite nd2"); CHKERRQ(ierr);

  // Time stepping and advection
  ierr = PetscBagRegisterInt(bag, &par->ts_scheme,0, "ts_scheme", "Time stepping scheme 0-forward euler, 1-backward euler, 2-crank-nicholson"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->adv_scheme,0, "adv_scheme", "Advection scheme 0-upwind, 1-fromm"); CHKERRQ(ierr);

  ierr = PetscBagRegisterInt(bag, &par->tout,1, "tout", "Output every tout time step"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->tstep,1, "tstep", "Maximum no of time steps"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->tmax, 2.5e-1, "tmax", "Maximum time [-]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->dtmax, 1.0e-3, "dtmax", "Maximum time step size [-]"); CHKERRQ(ierr);

  par->t  = 0.0;
  par->dt = 0.0;

  // Physical and material parameters
  ierr = PetscBagRegisterScalar(bag, &par->Ttop, 0.0, "Ttop", "Temperature top [-]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->Tbot, 1.0, "Tbot", "Temperature bottom [-]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->Ra, 1.0e4, "Ra", "Rayleigh number [-]"); CHKERRQ(ierr);
  
  ierr = PetscBagRegisterScalar(bag, &par->b, 0.0, "b", "Effective viscosity parameter b (T-dep) [-]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->c, 0.0, "c", "Effective viscosity parameter c (depth-dep) [-]"); CHKERRQ(ierr);

  // Input/output 
  ierr = PetscBagRegisterString(bag,&par->fname_out,FNAME_LENGTH,"out_convection","output_file","Name for output file, set with: -output_file <filename>"); CHKERRQ(ierr);

  // Other variables
  par->fname_in[0] = '\0';

  par->xmax = par->xmin+par->L;
  par->zmax = par->zmin+par->H;

  par->dT = par->Tbot-par->Ttop;

  // return pointer
  *_usr = usr;

  PetscFunctionReturn(0);
}

// ---------------------------------------
// InputPrintData
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "InputPrintData"
PetscErrorCode InputPrintData(UsrData *usr)
{
  char           date[30], *opts;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Get date
  ierr = PetscGetDate(date,30); CHKERRQ(ierr);

  // Get petsc command options
  ierr = PetscOptionsGetAll(NULL, &opts); CHKERRQ(ierr);

  // Print header and petsc options
  PetscPrintf(usr->comm,"# --------------------------------------- #\n");
  PetscPrintf(usr->comm,"# Test_mantle_convection: %s \n",&(date[0]));
  PetscPrintf(usr->comm,"# --------------------------------------- #\n");
  PetscPrintf(usr->comm,"# PETSc options: %s \n",opts);
  PetscPrintf(usr->comm,"# --------------------------------------- #\n");

  // Input file info
  if (usr->par->fname_in[0] == '\0') { // string is empty
    PetscPrintf(usr->comm,"# Input options file: NONE \n");
  }
  else {
    PetscPrintf(usr->comm,"# Input options file: %s \n",usr->par->fname_in);
  }
  PetscPrintf(usr->comm,"# --------------------------------------- #\n");

  // Print usr bag
  ierr = PetscBagView(usr->bag,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
  PetscPrintf(usr->comm,"# --------------------------------------- #\n");

  // Free memory
  ierr = PetscFree(opts); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

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

  // Start time
  ierr = PetscTime(&start_time); CHKERRQ(ierr);
 
  // Load command line or input file if required
  ierr = PetscOptionsInsert(PETSC_NULL,&argc,&argv,NULL); CHKERRQ(ierr);

  // Input user parameters and print
  ierr = InputParameters(&usr); CHKERRQ(ierr);

  // Save input options filename
  for (int i = 1; i < argc; i++) {
    PetscBool flg;
    
    ierr = PetscStrcmp(argv[i],"-options_file",&flg); CHKERRQ(ierr);
    if (flg) { ierr = PetscStrcpy(usr->par->fname_in, argv[i+1]); CHKERRQ(ierr); }
  }

  // Print user parameters
  ierr = InputPrintData(usr); CHKERRQ(ierr);

  // Numerical solution using the FD pde object
  ierr = Test_composite_convection(usr,usr->par->test); CHKERRQ(ierr);

  // Free memory
  ierr = PetscBagDestroy(&usr->bag); CHKERRQ(ierr);
  ierr = PetscFree(usr);             CHKERRQ(ierr);

  // End time
  ierr = PetscTime(&end_time); CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"# Total runtime: %g (sec) \n", end_time - start_time);
  PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");
  
  // Finalize main
  ierr = PetscFinalize();
  return ierr;
}