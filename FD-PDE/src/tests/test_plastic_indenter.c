// ---------------------------------------
// Exact solution for a rigid punch indenting a rigid plastic half space using the slip line field theory
// run: ./tests/test_plastic_indenter.app -pc_type lu -pc_factor_mat_solver_type umfpack -snes_monitor -ksp_monitor -nx 20 -nz 20
// python test: ./tests/python/test_plastic_indenter.py
// ---------------------------------------
static char help[] = "Application for a rigid punch indenting a rigid plastic half space (indenter test) \n\n";

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
#include "../consteq.h"
#include "../dmstagoutput.h"

// ---------------------------------------
// Application Context
// ---------------------------------------
#define FNAME_LENGTH 200

// parameters (bag)
typedef struct {
  PetscInt       nx, nz, smooth, harmonic;
  PetscScalar    L, H;
  PetscScalar    xmin, zmin;
  PetscScalar    eta0, rho0, g, p, vp, C, phi, etamax, etamin;
  PetscBool      plasticity;
  char           fname_out[FNAME_LENGTH]; 
} Params;

// user defined and model-dependent variables
typedef struct {
  Params        *par;
  PetscBag       bag;
  MPI_Comm       comm;
  PetscMPIInt    rank;
  DM             dm, dmeps;
  Vec            xeps;
} UsrData;

// ---------------------------------------
// Function definitions
// ---------------------------------------
PetscErrorCode Numerical_solution(void*);
PetscErrorCode InputParameters(UsrData**);
PetscErrorCode InputPrintData(UsrData*);
PetscErrorCode FormCoefficient(FDPDE, DM, Vec, DM, Vec, void*);
PetscErrorCode FormBCList(DM, Vec, DMStagBCList, void*);
PetscErrorCode UpdateStrainRates(DM,Vec,void*);

const char coeff_description[] =
"  << Stokes Coefficients >> \n"
"  eta_n/eta_c = eta_eff\n"
"  fux = 0 \n" 
"  fuz = rho*g \n" 
"  fp = 0 (incompressible)\n";

const char bc_description[] =
"  << Stokes BCs >> \n"
"  LEFT: (free slip) Vx = 0, dVz/dx = 0\n"
"  RIGHT: (free slip) Vx = 0, dVz/dx = 0\n" 
"  DOWN: (no slip) Vz = 0, Vx = 0\n" 
"  UP: (mixed) zero stress and Dirichlet \n";

// ---------------------------------------
// Application functions
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "Numerical_solution"
PetscErrorCode Numerical_solution(void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  FDPDE          fd;
  DM             dm, dmcoeff;
  Vec            x, xguess, xcoeff;
  PetscInt       nx, nz;
  PetscScalar    xmin, zmin, xmax, zmax;
  char           fout[FNAME_LENGTH];
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

  // Create the FD-pde object
  ierr = FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_STOKES,&fd);CHKERRQ(ierr);
  ierr = FDPDESetUp(fd);CHKERRQ(ierr);
  ierr = FDPDEGetDM(fd,&dm); CHKERRQ(ierr);

  // Create DM/vec for strain rates
  ierr = DMStagCreateCompatibleDMStag(dm,4,0,4,0,&usr->dmeps); CHKERRQ(ierr);
  ierr = DMSetUp(usr->dmeps); CHKERRQ(ierr);
  ierr = DMStagSetUniformCoordinatesProduct(usr->dmeps,xmin,xmax,zmin,zmax,0.0,0.0);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(usr->dmeps,&usr->xeps); CHKERRQ(ierr);

  // Set coefficients and BC evaluation function
  ierr = FDPDESetFunctionBCList(fd,FormBCList,bc_description,usr); CHKERRQ(ierr);
  ierr = FDPDESetFunctionCoefficient(fd,FormCoefficient,coeff_description,usr); CHKERRQ(ierr);
  ierr = FDPDEView(fd); CHKERRQ(ierr);

  // Create initial guess with a linear viscous
  PetscPrintf(PETSC_COMM_WORLD,"\n# INITIAL GUESS #\n");
  ierr = FDPDESolve(fd,NULL);CHKERRQ(ierr);
  ierr = FDPDEGetSolution(fd,&x);CHKERRQ(ierr); 

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s_solution_initial",usr->par->fname_out);
  ierr = DMStagViewBinaryPython(dm,x,fout);CHKERRQ(ierr);

  ierr = FDPDEGetCoefficient(fd,&dmcoeff,&xcoeff);CHKERRQ(ierr);
  ierr = PetscSNPrintf(fout,sizeof(fout),"%s_coefficient_initial",usr->par->fname_out);
  ierr = DMStagViewBinaryPython(dmcoeff,xcoeff,fout);CHKERRQ(ierr);

  ierr = FDPDEGetSolutionGuess(fd,&xguess); CHKERRQ(ierr); 
  ierr = VecCopy(x,xguess);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&xguess);CHKERRQ(ierr);
  usr->par->plasticity = PETSC_TRUE;

  // MatView(fd->J,PETSC_VIEWER_STDOUT_WORLD);

  // FD SNES Solver
  PetscPrintf(PETSC_COMM_WORLD,"\n# SNES SOLVE #\n");
  ierr = FDPDESolve(fd,NULL);CHKERRQ(ierr);
  ierr = FDPDEGetSolution(fd,&x);CHKERRQ(ierr); 

  // Output solution to file
  ierr = PetscSNPrintf(fout,sizeof(fout),"%s_solution",usr->par->fname_out);
  ierr = DMStagViewBinaryPython(dm,x,fout);CHKERRQ(ierr);

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s_strain",usr->par->fname_out);
  ierr = DMStagViewBinaryPython(usr->dmeps,usr->xeps,fout);CHKERRQ(ierr);

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s_residual",usr->par->fname_out);
  ierr = DMStagViewBinaryPython(dm,fd->r,fout);CHKERRQ(ierr);

  ierr = FDPDEGetCoefficient(fd,&dmcoeff,&xcoeff);CHKERRQ(ierr);
  ierr = PetscSNPrintf(fout,sizeof(fout),"%s_coefficient",usr->par->fname_out);
  ierr = DMStagViewBinaryPython(dmcoeff,xcoeff,fout);CHKERRQ(ierr);

  // Destroy objects
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xeps);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = DMDestroy(&usr->dmeps);CHKERRQ(ierr);
  ierr = FDPDEDestroy(&fd);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// FormCoefficient
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormCoefficient"
PetscErrorCode FormCoefficient(FDPDE fd, DM dm, Vec x, DM dmcoeff, Vec coeff, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz;
  Vec            coefflocal, xlocal, xepslocal;
  PetscScalar    **coordx,**coordz;
  PetscInt       iprev, inext, icenter;
  PetscScalar    ***c;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Strain rates
  ierr = UpdateStrainRates(dm,x,usr); CHKERRQ(ierr);
  ierr = DMGetLocalVector(usr->dmeps, &xepslocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmeps, usr->xeps, INSERT_VALUES, xepslocal); CHKERRQ(ierr);

  // Get domain corners
  ierr = DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = DMStagGetGlobalSizes(dmcoeff,&Nx,&Nz,NULL);CHKERRQ(ierr);

  // Get dm coordinates array
  ierr = DMStagGetProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmcoeff,LEFT,&iprev);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmcoeff,RIGHT,&inext);CHKERRQ(ierr); 
  ierr = DMStagGetProductCoordinateLocationSlot(dmcoeff,ELEMENT,&icenter);CHKERRQ(ierr);

  // Create coefficient local vector
  ierr = DMCreateLocalVector(dmcoeff, &coefflocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dmcoeff, coefflocal, &c); CHKERRQ(ierr);

  ierr = DMGetLocalVector(dm, &xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm, x, INSERT_VALUES, xlocal); CHKERRQ(ierr);

  // Loop over local domain - set initial density and viscosity
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      PetscInt idx;

      { // A = eta (center, c=1)
        DMStagStencil point, pointP;
        PetscScalar   Y,eta,eta_P,eta_VP,epsII,P,inv_eta, inv_eta_VP;

        // initialize non-plasticity viscosity
        eta = usr->par->eta0;

        if (usr->par->plasticity) {
          // second invariant of strain rate
          point.i = i; point.j = j; point.loc = ELEMENT; point.c = 3;
          ierr = DMStagVecGetValuesStencil(usr->dmeps,xepslocal,1,&point,&epsII); CHKERRQ(ierr);

          // pressure
          pointP.i = i; pointP.j = j; pointP.loc = ELEMENT; pointP.c = 0;
          ierr = DMStagVecGetValuesStencil(dm,xlocal,1,&pointP,&P); CHKERRQ(ierr);

          // plastic yield criterion
          Y = usr->par->C;
          // Y = usr->par->C + P*PetscSinScalar(PETSC_PI*usr->par->phi/180);
          eta_P = Y/(2.0*epsII);

          // Effective viscosity
          if (usr->par->harmonic) { // Harmonic averaging
            inv_eta_VP = 1.0/eta_P + 1.0/usr->par->eta0;
            inv_eta = 1.0/usr->par->etamax + inv_eta_VP;
            eta = usr->par->etamin + 1.0/inv_eta;
          } else {
            eta_VP = PetscMin(eta_P,usr->par->eta0);
            eta = PetscMin(PetscMax(eta_VP,usr->par->etamin),usr->par->etamax);
          }
        } 

        point.i = i; point.j = j; point.loc = ELEMENT; point.c = 1;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = eta;
      }

      { // A = eta (corner, c=0)
        DMStagStencil point[4],pointP[9];
        PetscScalar   eta,epsII[4],P[9], Pinterp[4], Y, eta_P, eta_VP, inv_eta, inv_eta_VP;
        PetscInt      ii;

        // second invariant of strain rate
        point[0].i = i; point[0].j = j; point[0].loc = DOWN_LEFT;  point[0].c = 3;
        point[1].i = i; point[1].j = j; point[1].loc = DOWN_RIGHT; point[1].c = 3;
        point[2].i = i; point[2].j = j; point[2].loc = UP_LEFT;    point[2].c = 3;
        point[3].i = i; point[3].j = j; point[3].loc = UP_RIGHT;   point[3].c = 3;
        ierr = DMStagVecGetValuesStencil(usr->dmeps,xepslocal,4,point,epsII); CHKERRQ(ierr);

        // pressure
        pointP[0].i = i-1; pointP[0].j = j-1; pointP[0].loc = ELEMENT; pointP[0].c = 0;
        pointP[1].i = i  ; pointP[1].j = j-1; pointP[1].loc = ELEMENT; pointP[1].c = 0;
        pointP[2].i = i+1; pointP[2].j = j-1; pointP[2].loc = ELEMENT; pointP[2].c = 0;
        pointP[3].i = i-1; pointP[3].j = j  ; pointP[3].loc = ELEMENT; pointP[3].c = 0;
        pointP[4].i = i  ; pointP[4].j = j  ; pointP[4].loc = ELEMENT; pointP[4].c = 0;
        pointP[5].i = i+1; pointP[5].j = j  ; pointP[5].loc = ELEMENT; pointP[5].c = 0;
        pointP[6].i = i-1; pointP[6].j = j+1; pointP[6].loc = ELEMENT; pointP[6].c = 0;
        pointP[7].i = i  ; pointP[7].j = j+1; pointP[7].loc = ELEMENT; pointP[7].c = 0;
        pointP[8].i = i+1; pointP[8].j = j+1; pointP[8].loc = ELEMENT; pointP[8].c = 0;

        // borders
        if (i==0   ) { pointP[0] = pointP[1]; pointP[3] = pointP[4]; pointP[6] = pointP[7]; } 
        if (i==Nx-1) { pointP[2] = pointP[1]; pointP[5] = pointP[4]; pointP[8] = pointP[7]; } 
        if (j==0   ) { pointP[0] = pointP[3]; pointP[1] = pointP[4]; pointP[2] = pointP[5]; } 
        if (j==Nz-1) { pointP[6] = pointP[3]; pointP[7] = pointP[4]; pointP[8] = pointP[5]; } 
        if ((i==0   ) && (j==0   )) {pointP[0] = pointP[4]; pointP[1] = pointP[4]; pointP[3] = pointP[4];}
        if ((i==Nx-1) && (j==0   )) {pointP[1] = pointP[4]; pointP[2] = pointP[4]; pointP[5] = pointP[4];}
        if ((i==0   ) && (j==Nz-1)) {pointP[3] = pointP[4]; pointP[6] = pointP[4]; pointP[7] = pointP[4];}
        if ((i==Nx-1) && (j==Nz-1)) {pointP[5] = pointP[4]; pointP[7] = pointP[4]; pointP[8] = pointP[4];}
        ierr = DMStagVecGetValuesStencil(dm,xlocal,9,pointP,P); CHKERRQ(ierr);

        // interpolate - assume constant grid spacing -> should be replaced with interpolation routines for every field
        Pinterp[0] = 0.25*(P[0]+P[1]+P[3]+P[4]);
        Pinterp[1] = 0.25*(P[1]+P[2]+P[4]+P[5]);
        Pinterp[2] = 0.25*(P[3]+P[4]+P[6]+P[7]);
        Pinterp[3] = 0.25*(P[4]+P[5]+P[7]+P[8]);

        for (ii = 0; ii < 4; ii++) {
          // linear viscous
          eta = usr->par->eta0;

          if (usr->par->plasticity) {
            // plastic yield criterion
            Y = usr->par->C;
            // Y = usr->par->C + Pinterp[ii]*PetscSinScalar(PETSC_PI*usr->par->phi/180);
            eta_P = Y/(2.0*epsII[ii]);

            // Effective viscosity
            if (usr->par->harmonic) { // Harmonic averaging
              inv_eta_VP = 1.0/eta_P + 1.0/usr->par->eta0;
              inv_eta = 1.0/usr->par->etamax + inv_eta_VP;
              eta = usr->par->etamin + 1.0/inv_eta;
            } else {
              eta_VP = PetscMin(eta_P,usr->par->eta0);
              eta = PetscMin(PetscMax(eta_VP,usr->par->etamin),usr->par->etamax);
            }
          }

          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, 0, &idx); CHKERRQ(ierr);
          c[j][i][idx] = eta;
        }
      }

      { // B = -rho*g (edges, c=0)
        DMStagStencil point[4];
        PetscScalar   rhs[4];
        PetscInt      ii;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 0;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = 0;
        point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = 0;

        rhs[0] = 0.0;
        rhs[1] = 0.0;
        rhs[2] = -usr->par->rho0*usr->par->g;
        rhs[3] = -usr->par->rho0*usr->par->g;

        for (ii = 0; ii < 4; ii++) {
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = rhs[ii];
        }
      }

      { // C = 0 (center, c=0)
        DMStagStencil point;
        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 0;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = 0.0;
      }
    }
  }

  // Restore arrays, local vectors
  ierr = DMStagRestoreProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL);CHKERRQ(ierr);

  ierr = DMStagVecRestoreArray(dmcoeff,coefflocal,&c);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  ierr = VecDestroy(&coefflocal); CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(dm,&xlocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmeps,&xepslocal); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// UpdateStrainRates
// ---------------------------------------
PetscErrorCode UpdateStrainRates(DM dm, Vec x, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  DM             dmeps;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz, idx;
  PetscInt       iprev, inext, icenter;
  PetscScalar    ***xx;
  PetscScalar    **coordx,**coordz;
  Vec            xeps, xepslocal,xlocal;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  dmeps = usr->dmeps;
  xeps  = usr->xeps;

  // Local vectors
  ierr = DMCreateLocalVector (dmeps,&xepslocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dmeps,xepslocal,&xx); CHKERRQ(ierr);

  ierr = DMGetLocalVector(dm,&xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm,x,INSERT_VALUES,xlocal); CHKERRQ(ierr);

  // Get domain corners
  ierr = DMStagGetGlobalSizes(dmeps, &Nx, &Nz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dmeps, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  
  // Get dm coordinates array
  ierr = DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter);CHKERRQ(ierr); 
  ierr = DMStagGetProductCoordinateLocationSlot(dm,LEFT,&iprev);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,RIGHT,&inext);CHKERRQ(ierr); 

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      DMStagStencil  pointC, pointN[4];
      PetscScalar    epsIIc, exxc, ezzc, exzc, epsIIn[4], exxn[4], ezzn[4], exzn[4];

      // Strain rates: center
      pointC.i = i; pointC.j = j; pointC.loc = ELEMENT; pointC.c = 0;
      ierr = DMStagGetPointStrainRates(dm,xlocal,1,&pointC,&epsIIc,&exxc,&ezzc,&exzc); CHKERRQ(ierr);

      // if ((i==0) || (i==Nx-1) || (j==0) || (j==Nz-1)) { // boundaries
      // }

      ierr = DMStagGetLocationSlot(dmeps,ELEMENT,0,&idx); CHKERRQ(ierr); xx[j][i][idx] = exxc;
      ierr = DMStagGetLocationSlot(dmeps,ELEMENT,1,&idx); CHKERRQ(ierr); xx[j][i][idx] = ezzc;
      ierr = DMStagGetLocationSlot(dmeps,ELEMENT,2,&idx); CHKERRQ(ierr); xx[j][i][idx] = exzc;
      ierr = DMStagGetLocationSlot(dmeps,ELEMENT,3,&idx); CHKERRQ(ierr); xx[j][i][idx] = epsIIc;

      // Strain rates: corner
      pointN[0].i = i; pointN[0].j = j; pointN[0].loc = DOWN_LEFT;  pointN[0].c = 0;
      pointN[1].i = i; pointN[1].j = j; pointN[1].loc = DOWN_RIGHT; pointN[1].c = 0;
      pointN[2].i = i; pointN[2].j = j; pointN[2].loc = UP_LEFT;    pointN[2].c = 0;
      pointN[3].i = i; pointN[3].j = j; pointN[3].loc = UP_RIGHT;   pointN[3].c = 0;
      ierr = DMStagGetPointStrainRates(dm,xlocal,4,pointN,epsIIn,exxn,ezzn,exzn); CHKERRQ(ierr);

      // if ((i==0) || (i==Nx-1) || (j==0) || (j==Nz-1)) { // boundaries
      // }

      ierr = DMStagGetLocationSlot(dmeps,DOWN_LEFT,0,&idx); CHKERRQ(ierr); xx[j][i][idx] = exxn[0];
      ierr = DMStagGetLocationSlot(dmeps,DOWN_LEFT,1,&idx); CHKERRQ(ierr); xx[j][i][idx] = ezzn[0];
      ierr = DMStagGetLocationSlot(dmeps,DOWN_LEFT,2,&idx); CHKERRQ(ierr); xx[j][i][idx] = exzn[0];
      ierr = DMStagGetLocationSlot(dmeps,DOWN_LEFT,3,&idx); CHKERRQ(ierr); xx[j][i][idx] = epsIIn[0];

      ierr = DMStagGetLocationSlot(dmeps,DOWN_RIGHT,0,&idx); CHKERRQ(ierr); xx[j][i][idx] = exxn[1];
      ierr = DMStagGetLocationSlot(dmeps,DOWN_RIGHT,1,&idx); CHKERRQ(ierr); xx[j][i][idx] = ezzn[1];
      ierr = DMStagGetLocationSlot(dmeps,DOWN_RIGHT,2,&idx); CHKERRQ(ierr); xx[j][i][idx] = exzn[1];
      ierr = DMStagGetLocationSlot(dmeps,DOWN_RIGHT,3,&idx); CHKERRQ(ierr); xx[j][i][idx] = epsIIn[1];

      ierr = DMStagGetLocationSlot(dmeps,UP_LEFT,0,&idx); CHKERRQ(ierr); xx[j][i][idx] = exxn[2];
      ierr = DMStagGetLocationSlot(dmeps,UP_LEFT,1,&idx); CHKERRQ(ierr); xx[j][i][idx] = ezzn[2];
      ierr = DMStagGetLocationSlot(dmeps,UP_LEFT,2,&idx); CHKERRQ(ierr); xx[j][i][idx] = exzn[2];
      ierr = DMStagGetLocationSlot(dmeps,UP_LEFT,3,&idx); CHKERRQ(ierr); xx[j][i][idx] = epsIIn[2];

      ierr = DMStagGetLocationSlot(dmeps,UP_RIGHT,0,&idx); CHKERRQ(ierr); xx[j][i][idx] = exxn[3];
      ierr = DMStagGetLocationSlot(dmeps,UP_RIGHT,1,&idx); CHKERRQ(ierr); xx[j][i][idx] = ezzn[3];
      ierr = DMStagGetLocationSlot(dmeps,UP_RIGHT,2,&idx); CHKERRQ(ierr); xx[j][i][idx] = exzn[3];
      ierr = DMStagGetLocationSlot(dmeps,UP_RIGHT,3,&idx); CHKERRQ(ierr); xx[j][i][idx] = epsIIn[3];
    }
  }

  // Restore arrays
  ierr = DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);

  // Restore and map local to global
  ierr = DMStagVecRestoreArray(dmeps,xepslocal,&xx); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dmeps,xepslocal,INSERT_VALUES,xeps); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dmeps,xepslocal,INSERT_VALUES,xeps); CHKERRQ(ierr);
  ierr = VecDestroy(&xepslocal); CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(dm, &xlocal ); CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

// ---------------------------------------
// FormBCList
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormBCList"
PetscErrorCode FormBCList(DM dm, Vec x, DMStagBCList bclist, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  PetscInt       sx, sz, nx, nz, Nx, Nz;
  PetscInt       i,k,n_bc,*idx_bc, iprev,icenter,inext;
  PetscScalar    *value_bc,*x_bc, xx[2], dx, ps, pe;
  PetscScalar    **coordx,**coordz;
  Vec            xlocal;
  BCType         *type_bc;
  DMStagStencil  point[2];
  PetscErrorCode ierr;
  
  PetscFunctionBegin;

  ps = (2*usr->par->xmin+usr->par->L)/2 - usr->par->p/2;
  pe = (2*usr->par->xmin+usr->par->L)/2 + usr->par->p/2;

  // Get solution dm/vector
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = DMStagGetGlobalSizes(dm,&Nx,&Nz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,LEFT,&iprev);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,RIGHT,&inext);CHKERRQ(ierr);

  // Map global vectors to local domain
  ierr = DMGetLocalVector(dm, &xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm, x, INSERT_VALUES, xlocal); CHKERRQ(ierr);
  
  // LEFT: free slip
  // Vx=0 on left boundary (w)
  ierr = DMStagBCListGetValues(bclist,'w','-',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // dVz/dx=0 on left boundary (w)
  ierr = DMStagBCListGetValues(bclist,'w','|',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  
  // RIGHT: free slip
  // Vx=0 on right boundary (e)
  ierr = DMStagBCListGetValues(bclist,'e','-',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // dVz/dx=0 on right boundary (e)
  ierr = DMStagBCListGetValues(bclist,'e','|',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN: no slip
  // Vx=0 on bottom boundary (s)
  ierr = DMStagBCListGetValues(bclist,'s','-',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // Vz=0 on bottom boundary (s)
  ierr = DMStagBCListGetValues(bclist,'s','|',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP: INDENTER and open boundary 
  // 1. Vx on top (n)
  ierr = DMStagBCListGetValues(bclist,'n','-',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // indenter Dirichlet BC
  for (k=0; k<n_bc; k++) {
    if ((x_bc[2*k]>=ps) && (x_bc[2*k]<=pe)) {
      if (!usr->par->smooth) { // rough or smooth
        value_bc[k] = 0.0;
        type_bc[k] = BC_DIRICHLET;
      }
    } 
  }

//  // Outside indenter: vx=0
//   for (k=0; k<n_bc; k++) {
//     if ((x_bc[2*k]<ps) || (x_bc[2*k]>pe)) {
//       value_bc[k] = 0;
//       type_bc[k] = BC_DIRICHLET;
//     }
//   }

  // Outside indenter: no shear stress dVx/dz=-dVz/dx
  for (k=1; k<n_bc; k++) {
    if ((x_bc[2*k]<ps) || (x_bc[2*k]>pe)) { 
      for (i = sx; i < sx+nx; i++) {
        if (x_bc[2*k]==coordx[i][iprev]) {
          point[0].i = i-1; point[0].j = Nz-1; point[0].loc = UP; point[0].c = 0;
          point[1].i = i  ; point[1].j = Nz-1; point[1].loc = UP; point[1].c = 0;
          if (i==0   ) point[0] = point[1];
          if (i==Nx-1) point[0] = point[1];
          ierr = DMStagVecGetValuesStencil(dm,xlocal,2,point,xx); CHKERRQ(ierr);
          if (i==0) dx = 2.0*(coordx[i][icenter] - coordx[i][iprev]);
          else      dx = coordx[i][icenter] - coordx[i-1][icenter];
          value_bc[k] = -(xx[1]-xx[0])/dx;
          type_bc[k] = BC_NEUMANN;
        }
      }
    }
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // 2. Vz on top boundary (n)
  ierr = DMStagBCListGetValues(bclist,'n','|',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // indenter Dirichlet BC
  for (k=0; k<n_bc; k++) {
    if ((x_bc[2*k]>=ps) && (x_bc[2*k]<=pe)) {
      value_bc[k] = -usr->par->vp;
      type_bc[k] = BC_DIRICHLET;
    } 
  }

  // Outside indenter: no normal stress dVz/dz=0 
  for (k=0; k<n_bc; k++) {
    if ((x_bc[2*k]<ps) || (x_bc[2*k]>pe)) { 
      value_bc[k] = 0.0;
      type_bc[k] = BC_NEUMANN_T;
    }
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // 3. P=0 on top boundary (n)
  ierr = DMStagBCListGetValues(bclist,'n','o',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    if ((x_bc[2*k]<ps) || (x_bc[2*k]>pe)) {
      value_bc[k] = 0.0;
      type_bc[k] = BC_DIRICHLET;
    }
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // restore
  ierr = DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&xlocal); CHKERRQ(ierr);

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
  ierr = PetscBagRegisterInt(bag, &par->nx, 5, "nx", "Element count in the x-dir"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->nz, 5, "nz", "Element count in the z-dir"); CHKERRQ(ierr);

  ierr = PetscBagRegisterScalar(bag, &par->xmin, -0.5, "xmin", "Start coordinate of domain in x-dir"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->zmin, -0.5, "zmin", "Start coordinate of domain in z-dir"); CHKERRQ(ierr);

  ierr = PetscBagRegisterScalar(bag, &par->L, 1.0, "L", "Length of domain in x-dir"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->H, 1.0, "H", "Height of domain in z-dir"); CHKERRQ(ierr);

  // Physical and material parameters
  ierr = PetscBagRegisterScalar(bag, &par->eta0, 1.0e3, "eta0", "Reference shear viscosity"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->p, 0.125, "p", "Indenter width"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->vp, 1.05, "vp", "Indenter velocity"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->rho0, 0.01, "rho0", "Reference density"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->g, 1.0, "g", "Gravitational acceleration"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->C, 1.0, "C", "Cohesion"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->phi, 0.0, "phi", "Angle of internal friction"); CHKERRQ(ierr);

  ierr = PetscBagRegisterInt(bag, &par->smooth, 0, "smooth", "0-rough punch, 1-smooth punch (Hill's)"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->harmonic, 0, "harmonic", "0-no 1-yes harmonic averaging"); CHKERRQ(ierr);
  
  par->etamax = par->eta0;
  par->etamin = 1.0e-6*par->eta0;
  par->plasticity = PETSC_FALSE;

  // Input/output 
  ierr = PetscBagRegisterString(bag,&par->fname_out,FNAME_LENGTH,"out_solution","output_file","Name for output file, set with: -output_file <filename>"); CHKERRQ(ierr);

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
  PetscPrintf(usr->comm,"# Test_plastic_indenter: %s \n",&(date[0]));
  PetscPrintf(usr->comm,"# --------------------------------------- #\n");
  PetscPrintf(usr->comm,"# PETSc options: %s \n",opts);
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
 
  // Load command line or input file if required
  ierr = PetscOptionsInsert(PETSC_NULL,&argc,&argv,NULL); CHKERRQ(ierr);

  // Input user parameters and print
  ierr = InputParameters(&usr); CHKERRQ(ierr);

  // Print user parameters
  ierr = InputPrintData(usr); CHKERRQ(ierr);

  // Numerical solution using the FD pde object
  ierr = PetscTime(&start_time); CHKERRQ(ierr);
  ierr = Numerical_solution(usr); CHKERRQ(ierr);
  ierr = PetscTime(&end_time); CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"# Runtime: %g (sec) \n", end_time - start_time);
  PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");

  // Destroy objects
  ierr = PetscBagDestroy(&usr->bag); CHKERRQ(ierr);
  ierr = PetscFree(usr);             CHKERRQ(ierr);

  // Finalize main
  ierr = PetscFinalize();
  return ierr;
}