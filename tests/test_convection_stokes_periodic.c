// ---------------------------------------
// Convection test with side periodic boundary conditions
// run: ./tests/test_convection_stokes_periodic.app -pc_type lu -pc_factor_mat_solver_type umfpack -pc_factor_mat_ordering_type external -nx 10 -nz 10
// python test: ./tests/python/test_convection_stokes_periodic.py
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

#include "petsc.h"
#include "../src/fdpde_stokes.h"
#include "../src/fdpde_advdiff.h"
#include "../src/dmstagoutput.h"

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
  PetscErrorCode ierr;
  
  PetscFunctionBegin;

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
  ierr = FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_STOKES,&fdstokes);CHKERRQ(ierr);
  ierr = FDPDESetDMBoundaryType(fdstokes,DM_BOUNDARY_PERIODIC,DM_BOUNDARY_NONE);CHKERRQ(ierr);
  ierr = FDPDESetUp(fdstokes);CHKERRQ(ierr);
  ierr = FDPDESetFunctionBCList(fdstokes,FormBCList_Stokes,bc_description_stokes,NULL); CHKERRQ(ierr);
  ierr = FDPDESetFunctionCoefficient(fdstokes,FormCoefficient_Stokes,coeff_description_stokes,usr); CHKERRQ(ierr);
  ierr = SNESSetFromOptions(fdstokes->snes); CHKERRQ(ierr);

  // 2. Temperature (Advection-diffusion)
  PetscPrintf(PETSC_COMM_WORLD,"# Set FD-PDE AdvDiff for temperature\n");
  ierr = FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_ADVDIFF,&fdtemp);CHKERRQ(ierr);
  ierr = FDPDESetDMBoundaryType(fdtemp,DM_BOUNDARY_PERIODIC,DM_BOUNDARY_NONE);CHKERRQ(ierr);
  ierr = FDPDESetUp(fdtemp);CHKERRQ(ierr);

  ierr = FDPDEAdvDiffSetAdvectSchemeType(fdtemp,ADV_FROMM);CHKERRQ(ierr);
  ierr = FDPDEAdvDiffSetTimeStepSchemeType(fdtemp,TS_CRANK_NICHOLSON );CHKERRQ(ierr);

  ierr = FDPDESetFunctionBCList(fdtemp,FormBCList_Temp,bc_description_temp,usr); CHKERRQ(ierr);
  ierr = FDPDESetFunctionCoefficient(fdtemp,FormCoefficient_Temp,coeff_description_temp,usr); CHKERRQ(ierr);
  ierr = SNESSetFromOptions(fdtemp->snes); CHKERRQ(ierr);

  // Prepare usr data - for coupling
  ierr = FDPDEGetDM(fdstokes,&dmPV); CHKERRQ(ierr);
  ierr = FDPDEGetDM(fdtemp,&dmT); CHKERRQ(ierr);
  usr->dmPV = dmPV;
  usr->dmT  = dmT;

  ierr = FDPDEGetSolution(fdstokes,&xPV);CHKERRQ(ierr);
  ierr = FDPDEGetSolution(fdtemp,&xT);CHKERRQ(ierr);
  ierr = VecDuplicate(xT,&usr->xTprev);CHKERRQ(ierr);
  ierr = VecDuplicate(xPV,&usr->xPV);CHKERRQ(ierr);
  ierr = VecDestroy(&xT);CHKERRQ(ierr);
  ierr = VecDestroy(&xPV);CHKERRQ(ierr);

  // Set initial temperature profile into xT, Tcoeff
  PetscPrintf(PETSC_COMM_WORLD,"# Set initial temperature profile\n");
  ierr = FDPDEAdvDiffGetPrevSolution(fdtemp,&xTprev);CHKERRQ(ierr);
  ierr = SetInitialTempProfile(dmT,xTprev,usr);CHKERRQ(ierr);
  ierr = VecCopy(xTprev,usr->xTprev);CHKERRQ(ierr);

  // Initialize guess with previous solution 
  ierr = FDPDEGetSolutionGuess(fdtemp,&xTguess);CHKERRQ(ierr);
  ierr = VecCopy(xTprev,xTguess);CHKERRQ(ierr);
  ierr = VecDestroy(&xTguess);CHKERRQ(ierr);

  // Solve Stokes to calculate velocities
  PetscPrintf(PETSC_COMM_WORLD,"# Set initial PV profile\n");
  ierr = FDPDESolve(fdstokes,NULL);CHKERRQ(ierr);
  ierr = FDPDEGetSolution(fdstokes,&xPV);CHKERRQ(ierr);
  ierr = VecCopy(xPV,usr->xPV);CHKERRQ(ierr);
  ierr = VecDestroy(&xPV);CHKERRQ(ierr);

  ierr = FDPDEGetCoefficient(fdtemp,&dmTcoeff,NULL);CHKERRQ(ierr);
  ierr = FDPDEAdvDiffGetPrevCoefficient(fdtemp,&Tcoeffprev);CHKERRQ(ierr);
  ierr = FormCoefficient_Temp(fdtemp,dmT,xTprev,dmTcoeff,Tcoeffprev,usr);CHKERRQ(ierr);
  ierr = VecDestroy(&Tcoeffprev);CHKERRQ(ierr);
  ierr = VecDestroy(&xTprev);CHKERRQ(ierr);

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
      ierr = FDPDEAdvDiffComputeExplicitTimestep(fdtemp,&dt);CHKERRQ(ierr);
      usr->par->dt = PetscMin(dt,usr->par->dtmax);
    }
    // usr->par->dt = usr->par->dtmax;
    ierr = FDPDEAdvDiffSetTimestep(fdtemp,usr->par->dt);CHKERRQ(ierr);

    // Update time
    usr->par->tprev = usr->par->t;
    usr->par->t    += usr->par->dt;

    // Temperature Solver
    PetscPrintf(PETSC_COMM_WORLD,"# Temperature Solver: \n");
    converged = PETSC_FALSE;
    while (!converged) {
      ierr = FDPDESolve(fdtemp,&converged);CHKERRQ(ierr);
      if (!converged) { // Reduce dt if not converged
        usr->par->dt *= dt_damp;
        ierr = FDPDEAdvDiffSetTimestep(fdtemp,usr->par->dt);CHKERRQ(ierr);
      }
    }

    ierr = FDPDEGetSolution(fdtemp,&xT);CHKERRQ(ierr);

    // Temperature: copy new solution and coefficient to old
    ierr = FDPDEAdvDiffGetPrevSolution(fdtemp,&xTprev);CHKERRQ(ierr);
    ierr = VecCopy(xT,xTprev);CHKERRQ(ierr);
    ierr = VecCopy(xTprev,usr->xTprev);CHKERRQ(ierr);
    ierr = VecDestroy(&xTprev);CHKERRQ(ierr);

    ierr = FDPDEGetCoefficient(fdtemp,&dmTcoeff,&Tcoeff);CHKERRQ(ierr);
    ierr = FDPDEAdvDiffGetPrevCoefficient(fdtemp,&Tcoeffprev);CHKERRQ(ierr);
    ierr = VecCopy(Tcoeff,Tcoeffprev);CHKERRQ(ierr);
    ierr = VecDestroy(&Tcoeffprev);CHKERRQ(ierr);

    // Stokes Solver - use Tprev
    PetscPrintf(PETSC_COMM_WORLD,"# Stokes Solver: \n");
    ierr = FDPDESolve(fdstokes,NULL);CHKERRQ(ierr);
    ierr = FDPDEGetSolution(fdstokes,&xPV);CHKERRQ(ierr);
    ierr = VecCopy(xPV,usr->xPV);CHKERRQ(ierr);

    // Output solution
    if (istep % usr->par->tout == 0 ) {
      ierr = PetscSNPrintf(fout,sizeof(fout),"%s/%s_PV_ts%1.3d",usr->par->fdir_out,usr->par->fname_out,istep);
      ierr = DMStagViewBinaryPython(dmPV,xPV,fout);CHKERRQ(ierr);
      ierr = PetscSNPrintf(fout,sizeof(fout),"%s/%s_T_ts%1.3d",usr->par->fdir_out,usr->par->fname_out,istep);
      ierr = DMStagViewBinaryPython(dmT,xT,fout);CHKERRQ(ierr);
    }

    // Clean up
    ierr = VecDestroy(&xPV);CHKERRQ(ierr);
    ierr = VecDestroy(&xT);CHKERRQ(ierr);

    // increment timestep
    istep++;

    PetscPrintf(PETSC_COMM_WORLD,"# TIME: time = %1.12e dt = %1.12e tmax = %1.12e\n",usr->par->t,usr->par->dt,usr->par->tmax);
  }

  ierr = DMDestroy(&dmPV); CHKERRQ(ierr);
  ierr = DMDestroy(&dmT); CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xPV);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xTprev);CHKERRQ(ierr);
  ierr = FDPDEDestroy(&fdstokes);CHKERRQ(ierr);
  ierr = FDPDEDestroy(&fdtemp);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
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

  PetscErrorCode ierr;
  PetscFunctionBegin;

  // Parameters
  Ttop = usr->par->Ttop;
  Tbot = usr->par->Tbot;
  a = (Ttop-Tbot)/usr->par->H;
  p = 0.05;

  // Get domain corners
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  // Get dm coordinates array
  ierr = DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter);CHKERRQ(ierr);

  // Create local vector
  ierr = DMCreateLocalVector(dm, &xlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm, xlocal, &xx); CHKERRQ(ierr);

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
      xx[j][i][idx] = Tbot+a*zp + p*PetscCosScalar(usr->par->n*PETSC_PI*xp)*PetscSinScalar(usr->par->n*PETSC_PI*zp);
    }
  }

  // Restore arrays, local vectors
  ierr = DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);

  ierr = DMStagVecRestoreArray(dm,xlocal,&xx);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm,xlocal,INSERT_VALUES,x); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,xlocal,INSERT_VALUES,x); CHKERRQ(ierr);
  
  ierr = VecDestroy(&xlocal); CHKERRQ(ierr);

  PetscFunctionReturn(0);
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
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  Ra = usr->par->Ra;
  
  // Get dm and solution vector for Temperature
  dmT = usr->dmT;
  xT  = usr->xTprev;

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
  ierr = DMStagGetProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmcoeff,LEFT,&iprev);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmcoeff,RIGHT,&inext);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmcoeff,ELEMENT,&icenter);CHKERRQ(ierr);

  // Create coefficient local vector
  ierr = DMCreateLocalVector(dmcoeff, &coefflocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dmcoeff, coefflocal, &cx); CHKERRQ(ierr);
  
  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {

      { // Bx = 0.0
        DMStagStencil point[2];
        PetscInt      ii, idx;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 0;

        for (ii = 0; ii < 2; ii++) {
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
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
        
        ierr = DMStagVecGetValuesStencil(dmT,xTlocal,3,pointT,T); CHKERRQ(ierr);

        for (ii = 0; ii < 2; ii++) {
          Tinterp = (T[ii]+T[ii+1])*0.5; // assume constant grid spacing
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          cx[j][i][idx] = -Ra*Tinterp;
        }
      }

      { // C = 0.0
        DMStagStencil point;
        PetscInt      idx;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 0;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        cx[j][i][idx] = 0.0;
      }

      { // A_center = 1.0
        DMStagStencil point;
        PetscInt      idx;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 1;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
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
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          cx[j][i][idx] = 1.0;
        }
      }
    }
  }

  // Restore arrays, local vectors
  ierr = DMStagRestoreProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArray(dmcoeff,coefflocal,&cx);CHKERRQ(ierr);
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
  PetscInt    sx, sz, nx, nz;
  PetscInt    k,n_bc,*idx_bc;
  PetscScalar *value_bc;
  BCType      *type_bc;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;

  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  
  // PERIODIC left boundary (w)
  ierr = DMStagBCListGetValues(bclist,'w','|',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    type_bc[k] = BC_PERIODIC;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  
  // PERIODIC on right boundary (e)
  ierr = DMStagBCListGetValues(bclist,'e','|',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    type_bc[k] = BC_PERIODIC;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  
  // dVx/dz=0 on top boundary (n)
  ierr = DMStagBCListGetValues(bclist,'n','-',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  
  // dVx/dz=0 on bottom boundary (s)
  ierr = DMStagBCListGetValues(bclist,'s','-',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  
  // PERIODIC on left boundary (w)
  ierr = DMStagBCListGetValues(bclist,'w','-',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    type_bc[k] = BC_PERIODIC;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  
  // PERIODIC on right boundary (e)
  ierr = DMStagBCListGetValues(bclist,'e','-',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    type_bc[k] = BC_PERIODIC;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  
  // Vz=0 on top boundary (n)
  ierr = DMStagBCListGetValues(bclist,'n','|',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  
  // Vz=0 on bottom boundary (s)
  ierr = DMStagBCListGetValues(bclist,'s','|',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // pin pressure
  // Warning: need to ensure valid boundary on processor
  if ((sx==0) && (sz==0)) {
    ierr = DMStagBCListGetValues(bclist,'s','o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
    value_bc[0] = 0.0;
    type_bc[0] = BC_DIRICHLET_STAG;
    ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
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
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  // Get dm and solution vector for Stokes velocity
  dmPV = usr->dmPV;
  xPV  = usr->xPV;
  
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
  ierr = DMStagVecGetArray(dmcoeff, coefflocal, &c); CHKERRQ(ierr);
  
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
  ierr = DMStagVecRestoreArray(dmcoeff,coefflocal,&c);CHKERRQ(ierr);
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
  
  // Left: PERIODIC
  ierr = DMStagBCListGetValues(bclist,'w','o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    type_bc[k] = BC_PERIODIC;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  
  // RIGHT: PERIODIC
  ierr = DMStagBCListGetValues(bclist,'e','o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    type_bc[k] = BC_PERIODIC;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN: T = Tbot (Tij = 2/3*Tbot+1/3*Tij+1)
  ierr = DMStagBCListGetValues(bclist,'s','o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = usr->par->Tbot;
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP: T = Ttop (Tij = 2/3*Ttop+1/3*Tij-1)
  ierr = DMStagBCListGetValues(bclist,'n','o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = usr->par->Ttop;
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
 
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
  ierr = PetscBagRegisterInt(bag, &par->nx, 50, "nx", "Element count in the x-dir"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->nz, 50, "nz", "Element count in the z-dir"); CHKERRQ(ierr);

  ierr = PetscBagRegisterScalar(bag, &par->xmin, 0.0, "xmin", "Start coordinate of domain in x-dir [-]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->zmin, 0.0, "zmin", "Start coordinate of domain in z-dir [-]"); CHKERRQ(ierr);

  ierr = PetscBagRegisterScalar(bag, &par->L, 1.0, "L", "Length of domain in x-dir [-]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->H, 1.0, "H", "Height of domain in z-dir [-]"); CHKERRQ(ierr);

  // Physical and material parameters
  ierr = PetscBagRegisterScalar(bag, &par->Ttop, 0.0, "Ttop", "Temperature top [-]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->Tbot, 1.0, "Tbot", "Temperature bottom [-]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->Ra, 1.0e6, "Ra", "Rayleigh number [-]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->n, 2.0, "n", "Initial perturbation sin [-]"); CHKERRQ(ierr);

  // Time stepping and advection
  ierr = PetscBagRegisterInt(bag, &par->tout,1, "tout", "Output every tout time step"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->tstep,1, "tstep", "Maximum no of time steps"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->tmax, 1e-3, "tmax", "Maximum time [-]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->dtmax, 1.0e-6, "dtmax", "Maximum time step size [-]"); CHKERRQ(ierr);

  // Input/output 
  ierr = PetscBagRegisterString(bag,&par->fname_out,FNAME_LENGTH,"out_convection","output_file","Name for output file, set with: -output_file <filename>"); CHKERRQ(ierr);
  ierr = PetscBagRegisterString(bag,&par->fdir_out,FNAME_LENGTH,"./","output_dir","Name for output directory, set with: -output_dir <dirname>"); CHKERRQ(ierr);

  par->t  = 0.0;
  par->tprev = 0.0;
  par->dt = 0.0;

  // Other variables
  par->fname_in[0] = '\0';

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
  ierr = Numerical_convection(usr); CHKERRQ(ierr);

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
