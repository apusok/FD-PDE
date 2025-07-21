// ---------------------------------------
// Convection test with side periodic boundary conditions
// run: ./test_convection_stokes_periodic.sh -pc_type lu -pc_factor_mat_solver_type umfpack -pc_factor_mat_ordering_type external -nx 10 -nz 10 -log_view
// python test: ./python/test_convection_stokes_periodic.py
// ---------------------------------------
static char help[] = "Convection in a box test with periodic boundary conditions \n\n";

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

// parameters (bag)
typedef struct {
  PetscInt       nx, nz;
  PetscScalar    L, H, xmin, zmin, Ttop, Tbot, Ra, n;
  PetscInt       tout, tstep;
  PetscScalar    t, dt, tprev, tmax, dtmax;
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

// ---------------------------------------
// Some descriptions
// ---------------------------------------
const char coeff_description_stokes[] =
"  << Stokes Coefficients (dimensionless) >> \n"
"  A =  1.0 \n"
"  B = -Ra*T*k, k-unit vector \n" 
"  C = 0 (incompressible)\n";

const char coeff_description_temp[] =
"  << Temperature Coefficients (dimensionless) >> \n"
"  A = 1.0 \n"
"  B = 1.0 (edge)\n"
"  C = 0 (element)\n"
"  u = [ux, uz] (edge) - Stokes velocity \n";

const char bc_description_stokes[] =
"  << Stokes BCs >> \n"
"  LEFT: PERIODIC \n"
"  RIGHT: PERIODIC \n" 
"  DOWN: Vz = 0, dVx/dz = 0 (free slip) \n" 
"  UP: Vz = 0, dVx/dz = 0 (free slip) \n";

const char bc_description_temp[] =
"  << Temperature BCs (dimensionless)>> \n"
"  LEFT: PERIODIC \n"
"  RIGHT: PERIODIC \n" 
"  DOWN: T = 1 (Tbot)\n" 
"  UP: T = 0 (Ttop) \n";

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
  xmax = usr->par->xmin+usr->par->L;
  zmax = usr->par->zmin+usr->par->H;

  // Create the sub FD-pde objects
  // 1. Stokes
  PetscPrintf(PETSC_COMM_WORLD,"# Set FD-PDE Stokes for pressure-velocity\n");
  PetscCall(FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_STOKES,&fdstokes));
  PetscCall(FDPDESetDMBoundaryType(fdstokes,DM_BOUNDARY_PERIODIC,DM_BOUNDARY_NONE));
  PetscCall(FDPDESetUp(fdstokes));
  PetscCall(FDPDESetFunctionBCList(fdstokes,FormBCList_Stokes,bc_description_stokes,NULL)); 
  PetscCall(FDPDESetFunctionCoefficient(fdstokes,FormCoefficient_Stokes,coeff_description_stokes,usr)); 
  PetscCall(SNESSetFromOptions(fdstokes->snes)); 

  // 2. Temperature (Advection-diffusion)
  PetscPrintf(PETSC_COMM_WORLD,"# Set FD-PDE AdvDiff for temperature\n");
  PetscCall(FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_ADVDIFF,&fdtemp));
  PetscCall(FDPDESetDMBoundaryType(fdtemp,DM_BOUNDARY_PERIODIC,DM_BOUNDARY_NONE));
  PetscCall(FDPDESetUp(fdtemp));

  PetscCall(FDPDEAdvDiffSetAdvectSchemeType(fdtemp,ADV_FROMM));
  PetscCall(FDPDEAdvDiffSetTimeStepSchemeType(fdtemp,TS_CRANK_NICHOLSON ));

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
  PetscPrintf(PETSC_COMM_WORLD,"# Set initial temperature profile\n");
  PetscCall(FDPDEAdvDiffGetPrevSolution(fdtemp,&xTprev));
  PetscCall(SetInitialTempProfile(dmT,xTprev,usr));
  PetscCall(VecCopy(xTprev,usr->xTprev));

  // Initialize guess with previous solution 
  PetscCall(FDPDEGetSolutionGuess(fdtemp,&xTguess));
  PetscCall(VecCopy(xTprev,xTguess));
  PetscCall(VecDestroy(&xTguess));

  // Solve Stokes to calculate velocities
  PetscPrintf(PETSC_COMM_WORLD,"# Set initial PV profile\n");
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
  while ((usr->par->t <= usr->par->tmax) && (istep<=usr->par->tstep)) {
    PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");
    PetscPrintf(PETSC_COMM_WORLD,"# TIMESTEP %d: \n",istep);
    PetscPrintf(PETSC_COMM_WORLD,"# Ra: %1.12e\n",usr->par->Ra);
    
    // Set dt for temperature advection 
    if (istep == 0) { // first timestep
      usr->par->dt = usr->par->dtmax*dt_damp*dt_damp;
    } else {
      PetscScalar dt;
      PetscCall(FDPDEAdvDiffComputeExplicitTimestep(fdtemp,&dt));
      usr->par->dt = PetscMin(dt,usr->par->dtmax);
    }
    // usr->par->dt = usr->par->dtmax;
    PetscCall(FDPDEAdvDiffSetTimestep(fdtemp,usr->par->dt));

    // Update time
    usr->par->tprev = usr->par->t;
    usr->par->t    += usr->par->dt;

    // Temperature Solver
    PetscPrintf(PETSC_COMM_WORLD,"# Temperature Solver: \n");
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
    PetscPrintf(PETSC_COMM_WORLD,"# Stokes Solver: \n");
    PetscCall(FDPDESolve(fdstokes,NULL));
    PetscCall(FDPDEGetSolution(fdstokes,&xPV));
    PetscCall(VecCopy(xPV,usr->xPV));

    // Output solution
    if (istep % usr->par->tout == 0 ) {
      PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s_PV_ts%1.3d",usr->par->fdir_out,usr->par->fname_out,istep));
      PetscCall(DMStagViewBinaryPython(dmPV,xPV,fout));
      PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s_T_ts%1.3d",usr->par->fdir_out,usr->par->fname_out,istep));
      PetscCall(DMStagViewBinaryPython(dmT,xT,fout));
    }

    // Clean up
    PetscCall(VecDestroy(&xPV));
    PetscCall(VecDestroy(&xT));

    // increment timestep
    istep++;

    PetscPrintf(PETSC_COMM_WORLD,"# TIME: time = %1.12e dt = %1.12e tmax = %1.12e\n",usr->par->t,usr->par->dt,usr->par->tmax);
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
// SetInitialTempProfile - T(x,z) = 0.05*cos(n*pi*x)*sin(n*pi*z)
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
  Ttop = usr->par->Ttop;
  Tbot = usr->par->Tbot;
  a = (Ttop-Tbot)/usr->par->H;
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
      xx[j][i][idx] = Tbot+a*zp + p*PetscCosScalar(usr->par->n*PETSC_PI*xp)*PetscSinScalar(usr->par->n*PETSC_PI*zp);
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
  PetscScalar    **coordx,**coordz, Ra;
  PetscInt       iprev, inext, icenter;
  PetscScalar    ***cx;
  PetscFunctionBeginUser;

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

      { // A_center = 1.0
        DMStagStencil point;
        PetscInt      idx;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 1;
        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
        cx[j][i][idx] = 1.0;
      }

      { // A_corner = 1.0
        DMStagStencil point[4];
        PetscInt      ii, idx;

        point[0].i = i; point[0].j = j; point[0].loc = DOWN_LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = DOWN_RIGHT; point[1].c = 0;
        point[2].i = i; point[2].j = j; point[2].loc = UP_LEFT;    point[2].c = 0;
        point[3].i = i; point[3].j = j; point[3].loc = UP_RIGHT;   point[3].c = 0;
        
        for (ii = 0; ii < 4; ii++) {
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          cx[j][i][idx] = 1.0;
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
  
  // PERIODIC left boundary (w)
  PetscCall(DMStagBCListGetValues(bclist,'w','|',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    type_bc[k] = BC_PERIODIC;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  
  // PERIODIC on right boundary (e)
  PetscCall(DMStagBCListGetValues(bclist,'e','|',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    type_bc[k] = BC_PERIODIC;
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
  
  // PERIODIC on left boundary (w)
  PetscCall(DMStagBCListGetValues(bclist,'w','-',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    type_bc[k] = BC_PERIODIC;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  
  // PERIODIC on right boundary (e)
  PetscCall(DMStagBCListGetValues(bclist,'e','-',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    type_bc[k] = BC_PERIODIC;
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

      { // A = 1.0
        DMStagStencil point;
        PetscInt      idx;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 0;
        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
        c[j][i][idx] = 1.0;
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
  
  // Left: PERIODIC
  PetscCall(DMStagBCListGetValues(bclist,'w','o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    type_bc[k] = BC_PERIODIC;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  
  // RIGHT: PERIODIC
  PetscCall(DMStagBCListGetValues(bclist,'e','o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    type_bc[k] = BC_PERIODIC;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));

  // DOWN: T = Tbot (Tij = 2/3*Tbot+1/3*Tij+1)
  PetscCall(DMStagBCListGetValues(bclist,'s','o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = usr->par->Tbot;
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));

  // UP: T = Ttop (Tij = 2/3*Ttop+1/3*Tij-1)
  PetscCall(DMStagBCListGetValues(bclist,'n','o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = usr->par->Ttop;
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

  PetscCall(PetscBagRegisterScalar(bag, &par->xmin, 0.0, "xmin", "Start coordinate of domain in x-dir [-]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->zmin, 0.0, "zmin", "Start coordinate of domain in z-dir [-]")); 

  PetscCall(PetscBagRegisterScalar(bag, &par->L, 1.0, "L", "Length of domain in x-dir [-]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->H, 1.0, "H", "Height of domain in z-dir [-]")); 

  // Physical and material parameters
  PetscCall(PetscBagRegisterScalar(bag, &par->Ttop, 0.0, "Ttop", "Temperature top [-]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->Tbot, 1.0, "Tbot", "Temperature bottom [-]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->Ra, 1.0e6, "Ra", "Rayleigh number [-]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->n, 2.0, "n", "Initial perturbation sin [-]")); 

  // Time stepping and advection
  PetscCall(PetscBagRegisterInt(bag, &par->tout,1, "tout", "Output every tout time step")); 
  PetscCall(PetscBagRegisterInt(bag, &par->tstep,1, "tstep", "Maximum no of time steps")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->tmax, 1e-3, "tmax", "Maximum time [-]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->dtmax, 1.0e-6, "dtmax", "Maximum time step size [-]")); 

  // Input/output 
  PetscCall(PetscBagRegisterString(bag,&par->fname_out,FNAME_LENGTH,"out_convection","output_file","Name for output file, set with: -output_file <filename>")); 
  PetscCall(PetscBagRegisterString(bag,&par->fdir_out,FNAME_LENGTH,"./","output_dir","Name for output directory, set with: -output_dir <dirname>")); 

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
  PetscPrintf(usr->comm,"# --------------------------------------- #\n");
  PetscPrintf(usr->comm,"# Test_convection_periodic: %s \n",&(date[0]));
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
  PetscCall(PetscBagView(usr->bag,PETSC_VIEWER_STDOUT_WORLD)); 
  PetscPrintf(usr->comm,"# --------------------------------------- #\n");

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
  PetscPrintf(PETSC_COMM_WORLD,"# Total runtime: %g (sec) \n", end_time - start_time);
  PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");
  
  // Finalize main
  PetscCall(PetscFinalize());
  return 0;
}
