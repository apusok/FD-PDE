// ---------------------------------------
// Mantle convection benchmark (Blankenbach et al. 1989, Gerya, 2009)
// Steady-state models:
//    1A - eta0=1e23, b=0, c=0, Ra = 1e4
//    1B - eta0=1e22, b=0, c=0, Ra = 1e5
//    1C - eta0=1e21, b=0, c=0, Ra = 1e6
//    2A - eta0=1e23, b=ln(1000), c=0, Ra0 = 1e4
//    2B - eta0=1e23, b=ln(16384), c=ln(64), L=2500, Ra0 = 1e4
// Time-dependent models (not tested):
//    3A - eta0=1e23, b=0, c=0, L=1500, Ra = 216000
// run: ./test_decoupled_convection_ -pc_type lu -pc_factor_mat_solver_type umfpack -pc_factor_mat_ordering_type external -nx 10 -nz 10 -log_view
// python test: ./python/test_decoupled_convection.py
//
// NON-DIMENSIONLESS FORM as in Moresi and Solomatov (1995) - decoupled PV-T
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

#include "../src/fdpde_stokes.h"
#include "../src/fdpde_advdiff.h"

// ---------------------------------------
// Application Context
// ---------------------------------------
#define FNAME_LENGTH 200
#define SEC_YEAR     31536000 //3600.00*24.00*365.00

// parameters (bag)
typedef struct {
  PetscInt       nx, nz;
  PetscScalar    L, H, nd_L, nd_H;
  PetscScalar    scal_h, scal_v, scal_t; 
  PetscScalar    xmin, zmin, xmax, zmax;
  PetscScalar    g, Ttop, Tbot, k, cp, rho0, eta0, b, c, alpha, DT;
  PetscScalar    Ra, kappa, nd_Ttop, nd_Tbot;
  PetscInt       ts_scheme, adv_scheme, test, tout, tstep, boussinesq;
  PetscScalar    t, dt, tprev, tmax, dtmax, nd_tmax, nd_dtmax;
  char           fname_out[FNAME_LENGTH]; 
  char           fname_in [FNAME_LENGTH];  
  char           fdir_out[FNAME_LENGTH]; 
} Params;

// user defined and model-dependent variables
typedef struct {
  Params        *par;
  PetscBag       bag;
  MPI_Comm       comm;
  PetscMPIInt    rank;
  Vec            xTprev, xPV;
  DM             dmT, dmPV;
} UsrData;

// ---------------------------------------
// Function definitions
// ---------------------------------------
PetscErrorCode InputParameters(UsrData**);
PetscErrorCode InputPrintData(UsrData*);
PetscErrorCode Numerical_convection(void*);
PetscErrorCode FormCoefficient_Stokes(FDPDE, DM, Vec, DM, Vec, void*);
PetscErrorCode FormBCList_Stokes(DM, Vec, DMStagBCList, void*);
PetscErrorCode FormCoefficient_Temp(FDPDE, DM, Vec, DM, Vec, void*);
PetscErrorCode FormBCList_Temp(DM, Vec, DMStagBCList, void*);
PetscErrorCode SetInitialTempProfile(DM,Vec,void*);
PetscErrorCode MantleConvectionDiagnostics(DM,Vec,DM,Vec,void*);

// ---------------------------------------
// Some descriptions
// ---------------------------------------
const char coeff_description_stokes[] =
"  << Stokes Coefficients (dimensionless) >> \n"
"  A =  eta_eff, eta_eff = PetscExpScalar( -b*T + c*(1-z) ) \n"
"  B = -Ra*T*k, k-unit vector \n" 
"  C = 0 (incompressible)\n";

const char coeff_description_temp[] =
"  << Temperature Coefficients (dimensionless) >> \n"
"  A = rho_eff, rho_eff = 1-alpha*DT*T (element)\n"
"  B = 1.0 (edge)\n"
"  C = 0 (element)\n"
"  u = [ux, uz] (edge) - Stokes velocity \n";

const char bc_description_stokes[] =
"  << Stokes BCs >> \n"
"  LEFT: Vx = 0, dVz/dx = 0 (free slip) \n"
"  RIGHT: Vx = 0, dVz/dx = 0 (free slip) \n" 
"  DOWN: Vz = 0, dVx/dz = 0 (free slip) \n" 
"  UP: Vz = 0, dVx/dz = 0 (free slip) \n";

const char bc_description_temp[] =
"  << Temperature BCs (dimensionless)>> \n"
"  LEFT: dT/dx = 0\n"
"  RIGHT: dT/dx = 0\n" 
"  DOWN: T = 1 (Tbot)\n" 
"  UP: T = 0 (Ttop) \n";

static PetscScalar EffectiveViscosity(PetscScalar T, PetscScalar z, PetscScalar b, PetscScalar c)
{ return PetscExpScalar( -b*T + c*(1.0-z) ); }

static PetscScalar EffectiveDensity(PetscScalar T, PetscScalar alpha, PetscScalar DT)
{ return 1.0-alpha*DT*T; }

static PetscScalar EffectiveDensity_Boussinesq(PetscScalar T, PetscScalar alpha, PetscScalar DT)
{ return 1.0; }

static PetscScalar scal_dim(PetscScalar X, PetscScalar scal)
{ return X*scal; }

// ---------------------------------------
// Application functions
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "Numerical_convection"
PetscErrorCode Numerical_convection(void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  FDPDE          fdtemp, fdstokes;
  DM             dmPV, dmT, dmTcoeff;
  Vec            xPV, xT, xTprev, xTguess,Tcoeff, Tcoeffprev;
  PetscInt       nx, nz, istep = 0;
  PetscScalar    xmin, zmin, xmax, zmax, dt_damp;
  char           fout[FNAME_LENGTH];
  PetscBool      converged;
  PetscFunctionBeginUser;

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
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"# Set FD-PDE Stokes for pressure-velocity\n"));
  PetscCall(FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_STOKES,&fdstokes));
  PetscCall(FDPDESetUp(fdstokes));
  PetscCall(FDPDESetFunctionBCList(fdstokes,FormBCList_Stokes,bc_description_stokes,NULL)); 
  PetscCall(FDPDESetFunctionCoefficient(fdstokes,FormCoefficient_Stokes,coeff_description_stokes,usr)); 
  PetscCall(SNESSetFromOptions(fdstokes->snes)); 

  // 2. Temperature (Advection-diffusion)
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"# Set FD-PDE AdvDiff for temperature\n"));
  PetscCall(FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_ADVDIFF,&fdtemp));
  PetscCall(FDPDESetUp(fdtemp));

  if (usr->par->adv_scheme==0) { PetscCall(FDPDEAdvDiffSetAdvectSchemeType(fdtemp,ADV_UPWIND)); }
  if (usr->par->adv_scheme==1) { PetscCall(FDPDEAdvDiffSetAdvectSchemeType(fdtemp,ADV_FROMM)); }

  if (usr->par->ts_scheme ==  0) { PetscCall(FDPDEAdvDiffSetTimeStepSchemeType(fdtemp,TS_FORWARD_EULER)); }
  if (usr->par->ts_scheme ==  1) { PetscCall(FDPDEAdvDiffSetTimeStepSchemeType(fdtemp,TS_BACKWARD_EULER)); }
  if (usr->par->ts_scheme ==  2) { PetscCall(FDPDEAdvDiffSetTimeStepSchemeType(fdtemp,TS_CRANK_NICHOLSON ));}

  PetscCall(FDPDESetFunctionBCList(fdtemp,FormBCList_Temp,bc_description_temp,usr)); 
  PetscCall(FDPDESetFunctionCoefficient(fdtemp,FormCoefficient_Temp,coeff_description_temp,usr)); 
  PetscCall(SNESSetFromOptions(fdtemp->snes)); 

  // Prepare usr data - for coupling
  PetscCall(FDPDEGetDM(fdstokes,&dmPV)); 
  PetscCall(FDPDEGetDM(fdtemp,&dmT)); 
  usr->dmPV = dmPV;
  usr->dmT  = dmT;

  PetscCall(FDPDEGetSolution(fdstokes,&xPV));
  PetscCall(FDPDEGetSolution(fdtemp,&xT));
  PetscCall(VecDuplicate(xT,&usr->xTprev));
  PetscCall(VecDuplicate(xPV,&usr->xPV));
  PetscCall(VecDestroy(&xT));
  PetscCall(VecDestroy(&xPV));

  // Set initial temperature profile into xT, Tcoeff
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"# Set initial temperature profile\n"));
  PetscCall(FDPDEAdvDiffGetPrevSolution(fdtemp,&xTprev));
  PetscCall(SetInitialTempProfile(dmT,xTprev,usr));
  PetscCall(VecCopy(xTprev,usr->xTprev));

  // Initialize guess with previous solution 
  PetscCall(FDPDEGetSolutionGuess(fdtemp,&xTguess));
  PetscCall(VecCopy(xTprev,xTguess));
  PetscCall(VecDestroy(&xTguess));

  // Solve Stokes to calculate velocities
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"# Set initial PV profile\n"));
  PetscCall(FDPDESolve(fdstokes,NULL));
  PetscCall(FDPDEGetSolution(fdstokes,&xPV));
  PetscCall(VecCopy(xPV,usr->xPV));
  PetscCall(VecDestroy(&xPV));

  PetscCall(FDPDEGetCoefficient(fdtemp,&dmTcoeff,NULL));
  PetscCall(FDPDEAdvDiffGetPrevCoefficient(fdtemp,&Tcoeffprev));
  PetscCall(FormCoefficient_Temp(fdtemp,dmT,xTprev,dmTcoeff,Tcoeffprev,usr));
  PetscCall(VecDestroy(&Tcoeffprev));
  PetscCall(VecDestroy(&xTprev));

  dt_damp = 1.0e-2;

  // Time loop
  while ((usr->par->t <= usr->par->nd_tmax) && (istep<=usr->par->tstep)) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n"));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"# TIMESTEP %d: \n",istep));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"# Ra: %1.12e\n",usr->par->Ra));
    
    // Set dt for temperature advection 
    if (istep == 0) { // first timestep
      usr->par->dt = usr->par->nd_dtmax*dt_damp*dt_damp;
    } else {
      PetscScalar dt;
      PetscCall(FDPDEAdvDiffComputeExplicitTimestep(fdtemp,&dt));
      usr->par->dt = PetscMin(dt,usr->par->nd_dtmax);
    }
    // usr->par->dt = usr->par->dtmax;
    PetscCall(FDPDEAdvDiffSetTimestep(fdtemp,usr->par->dt));

    // Update time
    usr->par->tprev = usr->par->t;
    usr->par->t    += usr->par->dt;

    // Temperature Solver
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"# Temperature Solver: \n"));
    converged = PETSC_FALSE;
    while (!converged) {
      PetscCall(FDPDESolve(fdtemp,&converged));
      if (!converged) { // Reduce dt if not converged
        usr->par->dt *= dt_damp;
        PetscCall(FDPDEAdvDiffSetTimestep(fdtemp,usr->par->dt));
      }
    }

    PetscCall(FDPDEGetSolution(fdtemp,&xT));

    // Temperature: copy new solution and coefficient to old
    PetscCall(FDPDEAdvDiffGetPrevSolution(fdtemp,&xTprev));
    PetscCall(VecCopy(xT,xTprev));
    PetscCall(VecCopy(xTprev,usr->xTprev));
    PetscCall(VecDestroy(&xTprev));

    PetscCall(FDPDEGetCoefficient(fdtemp,&dmTcoeff,&Tcoeff));
    PetscCall(FDPDEAdvDiffGetPrevCoefficient(fdtemp,&Tcoeffprev));
    PetscCall(VecCopy(Tcoeff,Tcoeffprev));
    PetscCall(VecDestroy(&Tcoeffprev));

    // Stokes Solver - use Tprev
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"# Stokes Solver: \n"));
    PetscCall(FDPDESolve(fdstokes,NULL));
    PetscCall(FDPDEGetSolution(fdstokes,&xPV));
    PetscCall(VecCopy(xPV,usr->xPV));

    // Calculate diagnostics
    PetscCall(MantleConvectionDiagnostics(dmPV,xPV,dmT,xT,usr)); 

    // Output solution
    if (istep % usr->par->tout == 0 ) {
      PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s_PV_m%d_ts%1.3d",usr->par->fdir_out,usr->par->fname_out,usr->par->ts_scheme,istep));
      PetscCall(DMStagViewBinaryPython(dmPV,xPV,fout));
      PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s_T_m%d_ts%1.3d",usr->par->fdir_out,usr->par->fname_out,usr->par->ts_scheme,istep));
      PetscCall(DMStagViewBinaryPython(dmT,xT,fout));
    }

    // Clean up
    PetscCall(VecDestroy(&xPV));
    PetscCall(VecDestroy(&xT));

    // increment timestep
    istep++;

    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"# TIME: time = %1.12e dt = %1.12e tmax = %1.12e\n",usr->par->t,usr->par->dt,usr->par->nd_tmax));
  }

  PetscCall(DMDestroy(&dmPV)); 
  PetscCall(DMDestroy(&dmT)); 
  PetscCall(VecDestroy(&usr->xPV));
  PetscCall(VecDestroy(&usr->xTprev));
  PetscCall(FDPDEDestroy(&fdstokes));
  PetscCall(FDPDEDestroy(&fdtemp));
  
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscFunctionBeginUser;

  PetscCall(PetscPrintf(usr->comm,"# Mantle convection diagnostics: \n"));
  PetscCall(PetscPrintf(usr->comm,"# Rayleigh number: Ra = %1.12e \n",usr->par->Ra));

  // Parameters
  L = usr->par->L;
  H = usr->par->H;
  lT[0] = 0.0;
  lT[1] = 0.0;

  // Get domain corners
  PetscCall(DMStagGetCorners(dmPV, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 
  PetscCall(DMStagGetGlobalSizes(dmPV,&Nx,&Nz,NULL));

  // Get dm coordinates array
  PetscCall(DMStagGetProductCoordinateArraysRead(dmPV,&coordx,&coordz,NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmPV,ELEMENT,&icenter));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmPV,LEFT,&iprev));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmPV,RIGHT,&inext));

  // Parameters
  dx = coordx[0][inext]-coordx[0][iprev];
  dz = coordz[0][inext]-coordz[0][iprev];

  // Map all vectors to local domain - xPV, xT, xOut
  PetscCall(DMGetLocalVector(dmPV,&xPVlocal)); 
  PetscCall(DMGlobalToLocal (dmPV,xPV,INSERT_VALUES,xPVlocal)); 

  PetscCall(DMGetLocalVector(dmT,&xTlocal)); 
  PetscCall(DMGlobalToLocal (dmT,xT,INSERT_VALUES,xTlocal)); 

  // 1. Nusselt number = surface mean flux/mean bottom temp, using nondimensional parameters
  j = sz; 
  if (j==0) { // bottom mean temperature
    for (i = sx; i<sx+nx; i++) {
      DMStagStencil point;
      PetscScalar   T;

      point.i = i; point.j = j; point.loc = ELEMENT; point.c = 0;
      PetscCall(DMStagVecGetValuesStencil(dmT,xTlocal,1,&point,&T)); 
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
      PetscCall(DMStagVecGetValuesStencil(dmT,xTlocal,2,point,T)); 
      lT[1] += (T[0]-T[1])/dz*dx;
    }
  }

  PetscCall(MPI_Allreduce(&lT, &gT, 2, MPI_DOUBLE, MPI_SUM, usr->comm)); 

  // Nusselt number
  Nu = -gT[1]/gT[0];
  PetscCall(PetscPrintf(usr->comm,"# Nusselt number: Nu = %1.12e \n",Nu));

  // 2. Root-mean-square velocity (vrms) - using dimensional params
  lvrms = 0.0;
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      DMStagStencil point[4];
      PetscScalar   v[4],vx,vz;

      point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 0;
      point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 0;
      point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = 0;
      point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = 0;
      PetscCall(DMStagVecGetValuesStencil(dmPV,xPVlocal,4,point,v)); 

      vx = scal_dim((v[0]+v[1])*0.5,usr->par->scal_v);
      vz = scal_dim((v[2]+v[3])*0.5,usr->par->scal_v);
      lvrms += (vx*vx+vz*vz)*scal_dim(dx,usr->par->scal_h)*scal_dim(dz,usr->par->scal_h);
    }
  }
  PetscCall(MPI_Allreduce(&lvrms, &gvrms, 1, MPI_DOUBLE, MPI_SUM, usr->comm)); 

  // Vrms
  vrms = H/usr->par->kappa*PetscSqrtReal(gvrms/H/L);
  PetscCall(PetscPrintf(usr->comm,"# Root-mean-squared velocity: vrms = %1.12e \n",vrms));

  // Non-dimensional temperature gradients at the corners of the cells
  i = sx;
  j = sz; 
  if ((i==0) && (j==0)) {
    DMStagStencil point[2];
    PetscScalar   T[2];

    point[0].i = i; point[0].j = j  ; point[0].loc = ELEMENT; point[0].c = 0;
    point[1].i = i; point[1].j = j+1; point[1].loc = ELEMENT; point[1].c = 0;
    PetscCall(DMStagVecGetValuesStencil(dmT,xTlocal,2,point,T)); 
    q = -(T[1]-T[0])/dz;
    PetscCall(PetscPrintf(usr->comm,"# Corner flux (down-left): q1 = %1.12e \n",q));
  }

  i = sx;
  j = sz+nz-1; 
  if ((i==0) && (j==Nz-1)) {
    DMStagStencil point[2];
    PetscScalar   T[2];

    point[0].i = i; point[0].j = j  ; point[0].loc = ELEMENT; point[0].c = 0;
    point[1].i = i; point[1].j = j-1; point[1].loc = ELEMENT; point[1].c = 0;
    PetscCall(DMStagVecGetValuesStencil(dmT,xTlocal,2,point,T)); 
    q = -(T[0]-T[1])/dz;
    PetscCall(PetscPrintf(usr->comm,"# Corner flux (up-left): q2 = %1.12e \n",q));
  } 

  i = sx+nx-1;
  j = sz; 
  if ((i==Nx-1) && (j==0)) {
    DMStagStencil point[2];
    PetscScalar   T[2];

    point[0].i = i; point[0].j = j  ; point[0].loc = ELEMENT; point[0].c = 0;
    point[1].i = i; point[1].j = j+1; point[1].loc = ELEMENT; point[1].c = 0;
    PetscCall(DMStagVecGetValuesStencil(dmT,xTlocal,2,point,T)); 
    q = -(T[1]-T[0])/dz;
    PetscCall(PetscPrintf(usr->comm,"# Corner flux (down-right): q3 = %1.12e \n",q));
  } 

  i = sx+nx-1;
  j = sz+nz-1; 
  if ((i==Nx-1) && (j==Nz-1)) {
    DMStagStencil point[2];
    PetscScalar   T[2];

    point[0].i = i; point[0].j = j  ; point[0].loc = ELEMENT; point[0].c = 0;
    point[1].i = i; point[1].j = j-1; point[1].loc = ELEMENT; point[1].c = 0;
    PetscCall(DMStagVecGetValuesStencil(dmT,xTlocal,2,point,T)); 
    q = -(T[0]-T[1])/dz;
    PetscCall(PetscPrintf(usr->comm,"# Corner flux (up-right): q4 = %1.12e \n",q));
  } 

  // Restore arrays, local vectors
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dmPV,&coordx,&coordz,NULL));
  PetscCall(DMRestoreLocalVector(dmPV,&xPVlocal)); 
  PetscCall(DMRestoreLocalVector(dmT,&xTlocal)); 

  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscFunctionBeginUser;

  // Parameters
  Ttop = usr->par->nd_Ttop;
  Tbot = usr->par->nd_Tbot;
  a = (Ttop-Tbot)/(usr->par->zmax-usr->par->zmin);
  p = 0.05;

  // Get domain corners
  PetscCall(DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 

  // Get dm coordinates array
  PetscCall(DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter));

  // Create local vector
  PetscCall(DMCreateLocalVector(dm, &xlocal)); 
  PetscCall(DMStagVecGetArray(dm, xlocal, &xx)); 

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      DMStagStencil point;
      PetscScalar   xp,zp;
      PetscInt      idx;

      point.i = i; point.j = j; point.loc = ELEMENT; point.c = 0;
      xp = coordx[i][icenter];
      zp = coordz[j][icenter];

      PetscCall(DMStagGetLocationSlot(dm, point.loc, point.c, &idx)); 
      xx[j][i][idx] = Tbot+a*zp + p*PetscCosScalar(PETSC_PI*xp)*PetscSinScalar(PETSC_PI*zp);
    }
  }

  // Restore arrays, local vectors
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));

  PetscCall(DMStagVecRestoreArray(dm,xlocal,&xx));
  PetscCall(DMLocalToGlobalBegin(dm,xlocal,INSERT_VALUES,x)); 
  PetscCall(DMLocalToGlobalEnd  (dm,xlocal,INSERT_VALUES,x)); 
  
  PetscCall(VecDestroy(&xlocal)); 

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// FormCoefficient_Stokes
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormCoefficient_Stokes"
PetscErrorCode FormCoefficient_Stokes(FDPDE fd, DM dm, Vec x, DM dmcoeff, Vec coeff, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  DM             dmT = NULL;
  Vec            xT = NULL, xTlocal, xlocal;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz;
  Vec            coefflocal;
  PetscScalar    **coordx,**coordz;
  PetscInt       iprev, inext, icenter;
  PetscScalar    ***cx;
  PetscScalar    eta,b,c,Ra;
  PetscFunctionBeginUser;

  // User parameters
  b  = usr->par->b;
  c  = usr->par->c;
  Ra = usr->par->Ra;
  
  // Get dm and solution vector for Temperature
  dmT = usr->dmT;
  xT  = usr->xTprev;

  PetscCall(DMCreateLocalVector(dmT,&xTlocal));
  PetscCall(DMGlobalToLocalBegin(dmT,xT,INSERT_VALUES,xTlocal));
  PetscCall(DMGlobalToLocalEnd(dmT,xT,INSERT_VALUES,xTlocal));
  
  // Get solution vector for Stokes
  PetscCall(DMGetLocalVector(dm,&xlocal)); 
  PetscCall(DMGlobalToLocal (dm,x,INSERT_VALUES,xlocal)); 

  // Get domain corners
  PetscCall(DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 
  PetscCall(DMStagGetGlobalSizes(dmcoeff,&Nx,&Nz,NULL));
  
  // Get dmcoeff coordinates array
  PetscCall(DMStagGetProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmcoeff,LEFT,&iprev));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmcoeff,RIGHT,&inext));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmcoeff,ELEMENT,&icenter));

  // Create coefficient local vector
  PetscCall(DMCreateLocalVector(dmcoeff, &coefflocal)); 
  PetscCall(DMStagVecGetArray(dmcoeff, coefflocal, &cx)); 
  
  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {

      { // Bx = 0.0
        DMStagStencil point[2];
        PetscInt      ii, idx;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 0;

        for (ii = 0; ii < 2; ii++) {
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          cx[j][i][idx] = 0.0;
        }
      }

      { // Bz = -Ra*T - need to interpolate temp to Vz points
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
        
        PetscCall(DMStagVecGetValuesStencil(dmT,xTlocal,3,pointT,T)); 

        for (ii = 0; ii < 2; ii++) {
          Tinterp = (T[ii]+T[ii+1])*0.5; // assume constant grid spacing
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          cx[j][i][idx] = -Ra*Tinterp;
        }
      }

      { // C = 0.0
        DMStagStencil point;
        PetscInt      idx;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 0;
        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
        cx[j][i][idx] = 0.0;
      }

      { // A_center = eta_eff
        PetscScalar   zp, T;
        DMStagStencil point, pointT;
        PetscInt      idx;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 1;
        pointT = point; pointT.c = 0;
        PetscCall(DMStagVecGetValuesStencil(dmT,xTlocal,1,&pointT,&T)); 
        
        zp = coordz[j][icenter];
        eta = EffectiveViscosity(T,zp,b,c);

        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
        cx[j][i][idx] = eta;
      }

      { // A_corner = eta_eff - need to interpolate Temp at corners
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
        
        PetscCall(DMStagVecGetValuesStencil(dmT,xTlocal,9,pointT,T)); 
        
        Tinterp[0] = (T[0]+T[1]+T[3]+T[4])*0.25; // assume constant grid spacing
        Tinterp[1] = (T[1]+T[2]+T[4]+T[5])*0.25;
        Tinterp[2] = (T[3]+T[4]+T[6]+T[7])*0.25;
        Tinterp[3] = (T[4]+T[5]+T[7]+T[8])*0.25;

        zp[0] = coordz[j][iprev];
        zp[1] = coordz[j][iprev];
        zp[2] = coordz[j][inext];
        zp[3] = coordz[j][inext];

        for (ii = 0; ii < 4; ii++) {
          eta = EffectiveViscosity(Tinterp[ii],zp[ii],b,c);
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          cx[j][i][idx] = eta;
        }
      }
    }
  }

  // Restore arrays, local vectors
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL));
  PetscCall(DMStagVecRestoreArray(dmcoeff,coefflocal,&cx));
  PetscCall(DMLocalToGlobalBegin(dmcoeff,coefflocal,INSERT_VALUES,coeff)); 
  PetscCall(DMLocalToGlobalEnd  (dmcoeff,coefflocal,INSERT_VALUES,coeff)); 
  PetscCall(VecDestroy(&coefflocal)); 
  
  PetscCall(DMRestoreLocalVector(dm,&xlocal)); 
  
  PetscCall(VecDestroy(&xTlocal));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// FormBCList_Stokes
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormBCList_Stokes"
PetscErrorCode FormBCList_Stokes(DM dm, Vec x, DMStagBCList bclist, void *ctx)
{
  PetscInt    sx, sz, nx, nz;
  PetscInt    k,n_bc,*idx_bc;
  PetscScalar *value_bc;
  BCType      *type_bc;
  PetscFunctionBeginUser;

  PetscCall(DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 
  
  // dVz/dx=0 on left boundary (w)
  PetscCall(DMStagBCListGetValues(bclist,'w','|',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  
  // dVz/dx=0 on right boundary (e)
  PetscCall(DMStagBCListGetValues(bclist,'e','|',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  
  // dVx/dz=0 on top boundary (n)
  PetscCall(DMStagBCListGetValues(bclist,'n','-',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  
  // dVx/dz=0 on bottom boundary (s)
  PetscCall(DMStagBCListGetValues(bclist,'s','-',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  
  // Vx=0 on left boundary (w)
  PetscCall(DMStagBCListGetValues(bclist,'w','-',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  
  // Vx=0 on right boundary (e)
  PetscCall(DMStagBCListGetValues(bclist,'e','-',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  
  // Vz=0 on top boundary (n)
  PetscCall(DMStagBCListGetValues(bclist,'n','|',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  
  // Vz=0 on bottom boundary (s)
  PetscCall(DMStagBCListGetValues(bclist,'s','|',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));

  // pin pressure
  // Warning: need to ensure valid boundary on processor
  if ((sx==0) && (sz==0)) {
    PetscCall(DMStagBCListGetValues(bclist,'s','o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
    value_bc[0] = 0.0;
    type_bc[0] = BC_DIRICHLET_STAG;
    PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// FormCoefficient_Temp
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormCoefficient_Temp"
PetscErrorCode FormCoefficient_Temp(FDPDE fd, DM dm, Vec x, DM dmcoeff, Vec coeff, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz;
  DM             dmPV = NULL;
  Vec            coefflocal;
  PetscScalar    ***c;
  Vec            xPV = NULL, xPVlocal, xlocal;
  PetscFunctionBeginUser;

  // Get dm and solution vector for Stokes velocity
  dmPV = usr->dmPV;
  xPV  = usr->xPV;
  
  PetscCall(DMCreateLocalVector(dmPV,&xPVlocal));
  PetscCall(DMGlobalToLocalBegin(dmPV,xPV,INSERT_VALUES,xPVlocal));
  PetscCall(DMGlobalToLocalEnd(dmPV,xPV,INSERT_VALUES,xPVlocal));
  
  // Get solution vector for temperature
  PetscCall(DMGetLocalVector(dm,&xlocal)); 
  PetscCall(DMGlobalToLocal (dm,x,INSERT_VALUES,xlocal)); 

  // Get domain corners
  PetscCall(DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 
  PetscCall(DMStagGetGlobalSizes(dmcoeff,&Nx,&Nz,NULL));

  // Create coefficient local vector
  PetscCall(DMCreateLocalVector(dmcoeff, &coefflocal)); 
  PetscCall(DMStagVecGetArray(dmcoeff, coefflocal, &c)); 
  
  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {

      { // A = rho_eff
        DMStagStencil point;
        PetscInt      idx;
        PetscScalar   T;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 0;
        PetscCall(DMStagVecGetValuesStencil(dm,xlocal,1,&point,&T)); 
        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
        if (usr->par->boussinesq) {
          c[j][i][idx] = EffectiveDensity_Boussinesq(T,usr->par->alpha,usr->par->DT);
        } else {
          c[j][i][idx] = EffectiveDensity(T,usr->par->alpha,usr->par->DT);
        }
      }

      { // C = 0.0 - sources of heat/sink
        DMStagStencil point;
        PetscInt      idx;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 1;
        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
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
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
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
        
        PetscCall(DMStagVecGetValuesStencil(dmPV,xPVlocal,4,point,v)); 

        for (ii = 0; ii < 4; ii++) {
          point[ii].c = 1;
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] = v[ii];
        }
      }
    }
  }

  // Restore arrays, local vectors
  PetscCall(DMStagVecRestoreArray(dmcoeff,coefflocal,&c));
  PetscCall(DMLocalToGlobalBegin(dmcoeff,coefflocal,INSERT_VALUES,coeff)); 
  PetscCall(DMLocalToGlobalEnd  (dmcoeff,coefflocal,INSERT_VALUES,coeff)); 
  PetscCall(VecDestroy(&coefflocal)); 

  PetscCall(DMRestoreLocalVector(dm,&xlocal)); 
  
  PetscCall(VecDestroy(&xPVlocal));

  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscFunctionBeginUser;
  
  // Left: dT/dx=0
  PetscCall(DMStagBCListGetValues(bclist,'w','o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  
  // RIGHT: dT/dx=0
  PetscCall(DMStagBCListGetValues(bclist,'e','o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));

  // DOWN: T = Tbot (Tij = 2/3*Tbot+1/3*Tij+1)
  PetscCall(DMStagBCListGetValues(bclist,'s','o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = usr->par->nd_Tbot;
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));

  // UP: T = Ttop (Tij = 2/3*Ttop+1/3*Tij-1)
  PetscCall(DMStagBCListGetValues(bclist,'n','o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = usr->par->nd_Ttop;
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
 
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscFunctionBeginUser;

  // Allocate memory to application context
  PetscCall(PetscMalloc1(1, &usr)); 

  // Get time, comm and rank
  usr->comm = PETSC_COMM_WORLD;
  PetscCall(MPI_Comm_rank(PETSC_COMM_WORLD, &usr->rank)); 

  // Create bag
  PetscCall(PetscBagCreate (usr->comm,sizeof(Params),&usr->bag)); 
  PetscCall(PetscBagGetData(usr->bag,(void **)&usr->par)); 
  PetscCall(PetscBagSetName(usr->bag,"UserParamBag","- User defined parameters -")); 

  // Define some pointers for easy access
  bag = usr->bag;
  par = usr->par;

  // Initialize domain variables
  PetscCall(PetscBagRegisterInt(bag, &par->nx, 50, "nx", "Element count in the x-dir")); 
  PetscCall(PetscBagRegisterInt(bag, &par->nz, 50, "nz", "Element count in the z-dir")); 

  PetscCall(PetscBagRegisterScalar(bag, &par->xmin, 0.0, "xmin", "Start coordinate of domain in x-dir [m]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->zmin, 0.0, "zmin", "Start coordinate of domain in z-dir [m]")); 

  PetscCall(PetscBagRegisterScalar(bag, &par->L, 1.0e6, "L", "Length of domain in x-dir [m]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->H, 1.0e6, "H", "Height of domain in z-dir [m]")); 

  // PetscCall(PetscBagRegisterInt(bag, &par->test,2, "test", "Test case 1-nd1, 2-nd2")); 

  // Physical and material parameters
  PetscCall(PetscBagRegisterScalar(bag, &par->g, 10.0, "g", "Gravitational acceleration [m/s2]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->Ttop, 273.0, "Ttop", "Temperature top [K]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->Tbot, 1273.0, "Tbot", "Temperature bottom [k]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->k, 5.0, "k", "Thermal conductivity [W/m/K]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->cp, 1250.0, "cp", "Heat capacity [J/kg]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->rho0, 4000.0, "rho0", "Reference density [kg/m3]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->alpha, 2.5e-5, "alpha", "Thermal expansion coefficient [1/K]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->eta0, 1.0e23, "eta0", "Reference viscosity [Pa.s]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->b, 0.0, "b", "Effective viscosity parameter b (T-dep) [-]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->c, 0.0, "c", "Effective viscosity parameter c (depth-dep) [-]")); 

  par->DT = par->Tbot - par->Ttop;
  par->kappa = par->k/par->rho0/par->cp;
  par->Ra = par->rho0*par->alpha*par->DT*par->g*par->H*par->H*par->H/par->eta0/par->kappa;

  PetscCall(PetscBagRegisterInt(bag, &par->boussinesq,1, "boussinesq", "Boussinesq approximation for energy balance: 0-no, 1-yes")); 

  // Time stepping and advection
  PetscCall(PetscBagRegisterInt(bag, &par->ts_scheme,0, "ts_scheme", "Time stepping scheme 0-forward euler, 1-backward euler, 2-crank-nicholson")); 
  PetscCall(PetscBagRegisterInt(bag, &par->adv_scheme,0, "adv_scheme", "Advection scheme 0-upwind, 1-fromm")); 

  PetscCall(PetscBagRegisterInt(bag, &par->tout,1, "tout", "Output every tout time step")); 
  PetscCall(PetscBagRegisterInt(bag, &par->tstep,1, "tstep", "Maximum no of time steps")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->tmax, 5000.0, "tmax", "Maximum time [Myr]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->dtmax, 1.0, "dtmax", "Maximum time step size [Myr]")); 

  // Input/output 
  PetscCall(PetscBagRegisterString(bag,&par->fname_out,FNAME_LENGTH,"out_convection","output_file","Name for output file, set with: -output_file <filename>")); 
  PetscCall(PetscBagRegisterString(bag,&par->fdir_out,FNAME_LENGTH,"./","output_dir","Name for output directory, set with: -output_dir <dirname>")); 

  par->scal_h = par->H;
  par->scal_v = par->kappa/par->scal_h;
  par->scal_t = par->scal_h*par->scal_h/par->kappa;

  par->nd_H = par->H/par->scal_h;
  par->nd_L = par->L/par->scal_h;
  par->xmax = par->xmin+par->nd_L;
  par->zmax = par->zmin+par->nd_H;
  par->nd_Ttop = 0.0; // scaled as (Ttop-Ttop)/DT
  par->nd_Tbot = 1.0; 

  par->nd_tmax  = par->tmax*1.0e6*SEC_YEAR/par->scal_t;
  par->nd_dtmax = par->dtmax*1.0e6*SEC_YEAR/par->scal_t;
  par->t  = 0.0;
  par->tprev = 0.0;
  par->dt = 0.0;

  // Other variables
  par->fname_in[0] = '\0';

  // return pointer
  *_usr = usr;

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// InputPrintData
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "InputPrintData"
PetscErrorCode InputPrintData(UsrData *usr)
{
  char           date[30], *opts;
  PetscFunctionBeginUser;

  // Get date
  PetscCall(PetscGetDate(date,30)); 

  // Get petsc command options
  PetscCall(PetscOptionsGetAll(NULL, &opts)); 

  // Print header and petsc options
  PetscCall(PetscPrintf(usr->comm,"# --------------------------------------- #\n"));
  PetscCall(PetscPrintf(usr->comm,"# Test_convection_nd1: %s \n",&(date[0])));
  PetscCall(PetscPrintf(usr->comm,"# --------------------------------------- #\n"));
  PetscCall(PetscPrintf(usr->comm,"# PETSc options: %s \n",opts));
  PetscCall(PetscPrintf(usr->comm,"# --------------------------------------- #\n"));

  // Input file info
  if (usr->par->fname_in[0] == '\0') { // string is empty
    PetscCall(PetscPrintf(usr->comm,"# Input options file: NONE \n"));
  }
  else {
    PetscCall(PetscPrintf(usr->comm,"# Input options file: %s \n",usr->par->fname_in));
  }
  PetscCall(PetscPrintf(usr->comm,"# --------------------------------------- #\n"));

  // Print usr bag
  PetscCall(PetscBagView(usr->bag,PETSC_VIEWER_STDOUT_WORLD)); 
  PetscCall(PetscPrintf(usr->comm,"# --------------------------------------- #\n"));

  // Free memory
  PetscCall(PetscFree(opts)); 

  PetscFunctionReturn(PETSC_SUCCESS);
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
    
  // Initialize application
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));

  // Start time
  PetscCall(PetscTime(&start_time)); 
 
  // Load command line or input file if required
  PetscCall(PetscOptionsInsert(PETSC_NULLPTR,&argc,&argv,NULL)); 

  // Input user parameters and print
  PetscCall(InputParameters(&usr)); 

  // Save input options filename
  for (int i = 1; i < argc; i++) {
    PetscBool flg;
    
    PetscCall(PetscStrcmp(argv[i],"-options_file",&flg)); 
    if (flg) { PetscCall(PetscStrcpy(usr->par->fname_in, argv[i+1]));  }
  }

  // Print user parameters
  PetscCall(InputPrintData(usr)); 

  // Numerical solution using the FD pde object
  PetscCall(Numerical_convection(usr)); 

  // Free memory
  PetscCall(PetscBagDestroy(&usr->bag)); 
  PetscCall(PetscFree(usr));             

  // End time
  PetscCall(PetscTime(&end_time)); 
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"# Total runtime: %g (sec) \n", end_time - start_time));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n"));
  
  // Finalize main
  PetscCall(PetscFinalize());
  return 0;
}
