// ---------------------------------------
// Shortening of a visco-(elasto)-plastic block in the absence of gravity
// Setup from T. Gerya, 2018, Ch. 13, ex. 13.2
// run: ./tests/test_vp_inclusion_gerya.app -pc_type lu -pc_factor_mat_solver_type umfpack -snes_monitor -ksp_monitor -nx 20 -nz 20
// python test: ./tests/python/test_vp_inclusion_gerya.py
// ---------------------------------------
static char help[] = "Application for shortening of a visco-(elasto)-plastic block in the absence of gravity \n\n";

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
  PetscScalar    eta0, eta_w, rho0, g, vi, C, phi, P;
  PetscScalar    stress, vel, length, visc; // scales
  PetscScalar    nd_eta0, nd_eta_w, nd_vi, nd_C, etamax, etamin, nd_P, C_factor; // non-dimensional
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
  Vec            xeps,xsigma;
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
"  LEFT/RIGHT: compression Vx=vi, free slip dvz/dx=0\n"
"  DOWN/UP: extension Vz=vi, free slip dvx/dz=0\n";

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
  ierr = DMCreateGlobalVector(usr->dmeps,&usr->xsigma); CHKERRQ(ierr);

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

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s_stress",usr->par->fname_out);
  ierr = DMStagViewBinaryPython(usr->dmeps,usr->xsigma,fout);CHKERRQ(ierr);

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s_residual",usr->par->fname_out);
  ierr = DMStagViewBinaryPython(dm,fd->r,fout);CHKERRQ(ierr);

  ierr = FDPDEGetCoefficient(fd,&dmcoeff,&xcoeff);CHKERRQ(ierr);
  ierr = PetscSNPrintf(fout,sizeof(fout),"%s_coefficient",usr->par->fname_out);
  ierr = DMStagViewBinaryPython(dmcoeff,xcoeff,fout);CHKERRQ(ierr);

  // Destroy objects
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xeps);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xsigma);CHKERRQ(ierr);
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
  Vec            coefflocal, xlocal, xepslocal, xsigmalocal;
  PetscScalar    **coordx,**coordz;
  PetscScalar    zblock_s, zblock_e, xis, xie, zis, zie;
  PetscInt       iprev, inext, icenter;
  PetscScalar    ***c, ***xxs;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // block and weak inclusion params
  zblock_s = 0.2*usr->par->H;
  zblock_e = 0.8*usr->par->H;
  xis = usr->par->xmin+usr->par->L/2-0.1*usr->par->L/2;
  xie = usr->par->xmin+usr->par->L/2+0.1*usr->par->L/2;
  zis = usr->par->zmin+usr->par->H/2-0.1*usr->par->H/2;
  zie = usr->par->zmin+usr->par->H/2+0.1*usr->par->H/2;

  // Strain rates
  ierr = UpdateStrainRates(dm,x,usr); CHKERRQ(ierr);
  ierr = DMGetLocalVector(usr->dmeps, &xepslocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmeps, usr->xeps, INSERT_VALUES, xepslocal); CHKERRQ(ierr);

  // Local vectors
  ierr = DMCreateLocalVector (usr->dmeps,&xsigmalocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(usr->dmeps,xsigmalocal,&xxs); CHKERRQ(ierr);

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

  // PetscPrintf(PETSC_COMM_WORLD,"\n# ITERATION #\n");
  // Loop over local domain - set initial density and viscosity
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      PetscInt idx;

      { // A = eta (center, c=1)
        DMStagStencil point, pointP;
        PetscScalar   Y,eta,eta_P,eta_VP,epsII,P,inv_eta, inv_eta_VP,exx,ezz,exz,sxx,szz,sxz,sigmaII;

        if ((coordz[j][icenter]<zblock_s) || (coordz[j][icenter]>zblock_e)) { 
          eta = usr->par->nd_eta_w; // top/bottom layer
          Y   = usr->par->nd_C*usr->par->C_factor; // plastic yield criterion
        } else if ((coordx[i][icenter]>=xis) && (coordx[i][icenter]<=xie) && (coordz[j][icenter]>=zis) && (coordz[j][icenter]<=zie)) {
          eta = usr->par->nd_eta_w; // inclusion
          Y   = usr->par->nd_C*usr->par->C_factor; // plastic yield criterion
        } else {
          // initialize non-plasticity viscosity
          eta = usr->par->nd_eta0;

          // plastic yield criterion
          Y = usr->par->nd_C;
          // Y = usr->par->nd_C + P*PetscSinScalar(PETSC_PI*usr->par->phi/180);
        }

        // second invariant of strain rate
        point.i = i; point.j = j; point.loc = ELEMENT; 
        point.c = 0; ierr = DMStagVecGetValuesStencil(usr->dmeps,xepslocal,1,&point,&exx); CHKERRQ(ierr);
        point.c = 1; ierr = DMStagVecGetValuesStencil(usr->dmeps,xepslocal,1,&point,&ezz); CHKERRQ(ierr);
        point.c = 2; ierr = DMStagVecGetValuesStencil(usr->dmeps,xepslocal,1,&point,&exz); CHKERRQ(ierr);
        point.c = 3; ierr = DMStagVecGetValuesStencil(usr->dmeps,xepslocal,1,&point,&epsII); CHKERRQ(ierr);

        // pressure
        pointP.i = i; pointP.j = j; pointP.loc = ELEMENT; pointP.c = 0;
        ierr = DMStagVecGetValuesStencil(dm,xlocal,1,&pointP,&P); CHKERRQ(ierr);

        // second invariant of stress
        sxx = -P+2.0*eta*exx;
        szz = -P+2.0*eta*ezz;
        sxz = 2.0*eta*exz;
        sigmaII = PetscPowScalar(0.5*(sxx*sxx + szz*szz + 2.0*sxz*sxz),0.5);

        if (usr->par->plasticity) {
          // eta_P = Y/(2.0*epsII);
          // if (eta_P>eta) {
          //   eta_P=eta;
          // }
          // if (eta_P<usr->par->etamin) eta_P = usr->par->etamin;
          // if (eta_P>usr->par->etamax) eta_P = usr->par->etamax;
          // eta=eta_P;

          if (sigmaII > Y) {
            eta_P = Y/(2.0*epsII);

            // Effective viscosity
            if (usr->par->harmonic) { // Harmonic averaging
              inv_eta_VP = 1.0/eta_P + 1.0/eta;
              inv_eta = 1.0/usr->par->etamax + inv_eta_VP;
              eta = usr->par->etamin + 1.0/inv_eta;
            } else {
              eta_VP = PetscMin(eta_P,eta);
              eta = PetscMin(PetscMax(eta_VP,usr->par->etamin),usr->par->etamax);
            }
          }
        }

        point.i = i; point.j = j; point.loc = ELEMENT; point.c = 1;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = eta;

        // save stresses for output
        ierr = DMStagGetLocationSlot(usr->dmeps,ELEMENT,0,&idx); CHKERRQ(ierr); xxs[j][i][idx] = sxx;
        ierr = DMStagGetLocationSlot(usr->dmeps,ELEMENT,1,&idx); CHKERRQ(ierr); xxs[j][i][idx] = szz;
        ierr = DMStagGetLocationSlot(usr->dmeps,ELEMENT,2,&idx); CHKERRQ(ierr); xxs[j][i][idx] = sxz;
        ierr = DMStagGetLocationSlot(usr->dmeps,ELEMENT,3,&idx); CHKERRQ(ierr); xxs[j][i][idx] = sigmaII;
      }

      { // A = eta (corner, c=0)
        DMStagStencil point[4],pointP[9];
        PetscScalar   eta,epsII[4],P[9], Pinterp[4], Y, eta_P, eta_VP, inv_eta, inv_eta_VP, xp[4],zp[4];
        PetscScalar   exx[4],ezz[4],exz[4],sxx[4],szz[4],sxz[4],sigmaII[4];
        PetscInt      ii;

        // second invariant of strain rate
        point[0].i = i; point[0].j = j; point[0].loc = DOWN_LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = DOWN_RIGHT; point[1].c = 0;
        point[2].i = i; point[2].j = j; point[2].loc = UP_LEFT;    point[2].c = 0;
        point[3].i = i; point[3].j = j; point[3].loc = UP_RIGHT;   point[3].c = 0;
        ierr = DMStagVecGetValuesStencil(usr->dmeps,xepslocal,4,point,exx); CHKERRQ(ierr);
        
        for (ii = 0; ii < 4; ii++) {point[ii].c = 1;}
        ierr = DMStagVecGetValuesStencil(usr->dmeps,xepslocal,4,point,ezz); CHKERRQ(ierr);

        for (ii = 0; ii < 4; ii++) {point[ii].c = 2;}
        ierr = DMStagVecGetValuesStencil(usr->dmeps,xepslocal,4,point,exz); CHKERRQ(ierr);

        // second invariant of strain rate
        for (ii = 0; ii < 4; ii++) {point[ii].c = 3;}
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

        // coordinates
        xp[0] = coordx[i][iprev]; zp[0] = coordz[j][iprev]; 
        xp[1] = coordx[i][inext]; zp[1] = coordz[j][iprev]; 
        xp[2] = coordx[i][iprev]; zp[2] = coordz[j][inext]; 
        xp[3] = coordx[i][inext]; zp[3] = coordz[j][inext]; 

        for (ii = 0; ii < 4; ii++) {
          if ((zp[ii]<zblock_s) || (zp[ii]>zblock_e)) { 
            eta = usr->par->nd_eta_w; // top/bottom layer
            Y   = usr->par->nd_C*usr->par->C_factor; // plastic yield criterion
          } else if ((xp[ii]>=xis) && (xp[ii]<=xie) && (zp[ii]>=zis) && (zp[ii]<=zie)) {
            eta = usr->par->nd_eta_w; // inclusion
            Y   = usr->par->nd_C*usr->par->C_factor; // plastic yield criterion
          } else {
            // linear viscous
            eta = usr->par->nd_eta0;
            // plastic yield criterion
            Y = usr->par->nd_C;
            // Y = usr->par->C + Pinterp[ii]*PetscSinScalar(PETSC_PI*usr->par->phi/180);
          }

          // second invariant of stress
          sxx[ii] = -Pinterp[ii]+2.0*eta*exx[ii];
          szz[ii] = -Pinterp[ii]+2.0*eta*ezz[ii];
          sxz[ii] = 2.0*eta*exz[ii];
          sigmaII[ii] = PetscPowScalar(0.5*(sxx[ii]*sxx[ii] + szz[ii]*szz[ii] + 2.0*sxz[ii]*sxz[ii]),0.5);

          if (usr->par->plasticity) {
            // eta_P = Y/(2.0*epsII[ii]);
            // if (eta_P>eta) {
            //   eta_P=eta;
            // }
            // if (eta_P<usr->par->etamin) eta_P = usr->par->etamin;
            // if (eta_P>usr->par->etamax) eta_P = usr->par->etamax;
            // eta=eta_P;

            if (sigmaII[ii] > Y) {
              eta_P = Y/(2.0*epsII[ii]);

              // Effective viscosity
              if (usr->par->harmonic) { // Harmonic averaging
                inv_eta_VP = 1.0/eta_P + 1.0/eta;
                inv_eta = 1.0/usr->par->etamax + inv_eta_VP;
                eta = usr->par->etamin + 1.0/inv_eta;
              } else {
                eta_VP = PetscMin(eta_P,eta);
                eta = PetscMin(PetscMax(eta_VP,usr->par->etamin),usr->par->etamax);
              }
            }
          }

          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, 0, &idx); CHKERRQ(ierr);
          c[j][i][idx] = eta;
        }

        // save stresses
        ierr = DMStagGetLocationSlot(usr->dmeps,DOWN_LEFT,0,&idx); CHKERRQ(ierr); xxs[j][i][idx] = sxx[0];
        ierr = DMStagGetLocationSlot(usr->dmeps,DOWN_LEFT,1,&idx); CHKERRQ(ierr); xxs[j][i][idx] = szz[0];
        ierr = DMStagGetLocationSlot(usr->dmeps,DOWN_LEFT,2,&idx); CHKERRQ(ierr); xxs[j][i][idx] = sxz[0];
        ierr = DMStagGetLocationSlot(usr->dmeps,DOWN_LEFT,3,&idx); CHKERRQ(ierr); xxs[j][i][idx] = sigmaII[0];

        ierr = DMStagGetLocationSlot(usr->dmeps,DOWN_RIGHT,0,&idx); CHKERRQ(ierr); xxs[j][i][idx] = sxx[1];
        ierr = DMStagGetLocationSlot(usr->dmeps,DOWN_RIGHT,1,&idx); CHKERRQ(ierr); xxs[j][i][idx] = szz[1];
        ierr = DMStagGetLocationSlot(usr->dmeps,DOWN_RIGHT,2,&idx); CHKERRQ(ierr); xxs[j][i][idx] = sxz[1];
        ierr = DMStagGetLocationSlot(usr->dmeps,DOWN_RIGHT,3,&idx); CHKERRQ(ierr); xxs[j][i][idx] = sigmaII[1];

        ierr = DMStagGetLocationSlot(usr->dmeps,UP_LEFT,0,&idx); CHKERRQ(ierr); xxs[j][i][idx] = sxx[2];
        ierr = DMStagGetLocationSlot(usr->dmeps,UP_LEFT,1,&idx); CHKERRQ(ierr); xxs[j][i][idx] = szz[2];
        ierr = DMStagGetLocationSlot(usr->dmeps,UP_LEFT,2,&idx); CHKERRQ(ierr); xxs[j][i][idx] = sxz[2];
        ierr = DMStagGetLocationSlot(usr->dmeps,UP_LEFT,3,&idx); CHKERRQ(ierr); xxs[j][i][idx] = sigmaII[2];

        ierr = DMStagGetLocationSlot(usr->dmeps,UP_RIGHT,0,&idx); CHKERRQ(ierr); xxs[j][i][idx] = sxx[3];
        ierr = DMStagGetLocationSlot(usr->dmeps,UP_RIGHT,1,&idx); CHKERRQ(ierr); xxs[j][i][idx] = szz[3];
        ierr = DMStagGetLocationSlot(usr->dmeps,UP_RIGHT,2,&idx); CHKERRQ(ierr); xxs[j][i][idx] = sxz[3];
        ierr = DMStagGetLocationSlot(usr->dmeps,UP_RIGHT,3,&idx); CHKERRQ(ierr); xxs[j][i][idx] = sigmaII[3];
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

  // Restore and map local to global
  ierr = DMStagVecRestoreArray(usr->dmeps,xsigmalocal,&xxs); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(usr->dmeps,xsigmalocal,INSERT_VALUES,usr->xsigma); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (usr->dmeps,xsigmalocal,INSERT_VALUES,usr->xsigma); CHKERRQ(ierr);
  ierr = VecDestroy(&xsigmalocal); CHKERRQ(ierr);

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
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz, idx, ii;
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
      //   exzc = 0.0;
      //   exxc = 0.0;
      //   ezzc = 0.0;
      //   epsIIc = PetscPowScalar(0.5*(exxc*exxc + ezzc*ezzc + 2.0*exzc*exzc),0.5);
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
      
      if (i==0) { // boundaries
        pointC.i = i; pointC.j = j; pointC.loc = ELEMENT; pointC.c = 0;
        ierr = DMStagGetPointStrainRates(dm,xlocal,1,&pointC,&epsIIc,&exxc,&ezzc,&exzc); CHKERRQ(ierr);
        ezzn[0] = ezzc;
        // PetscPrintf(PETSC_COMM_WORLD,"# L Strain [%d %d]: exx=%f ezz=%f exz=%f eII=%f #\n",i,j,exxn[0],ezzn[0],exzn[0],epsIIn[0]);
      }

      if (i==Nx-1) { // boundaries
        pointC.i = i; pointC.j = j; pointC.loc = ELEMENT; pointC.c = 0;
        ierr = DMStagGetPointStrainRates(dm,xlocal,1,&pointC,&epsIIc,&exxc,&ezzc,&exzc); CHKERRQ(ierr);
        ezzn[1] = ezzc;
        // PetscPrintf(PETSC_COMM_WORLD,"# R Strain [%d %d]: exx=%f ezz=%f exz=%f eII=%f #\n",i,j,exxn[1],ezzn[1],exzn[1],epsIIn[1]);
      }

      if (j==0) { // boundaries
        pointC.i = i; pointC.j = j; pointC.loc = ELEMENT; pointC.c = 0;
        ierr = DMStagGetPointStrainRates(dm,xlocal,1,&pointC,&epsIIc,&exxc,&ezzc,&exzc); CHKERRQ(ierr);
        exxn[0] = exxc;
        // PetscPrintf(PETSC_COMM_WORLD,"# D Strain [%d %d]: exx=%f ezz=%f exz=%f eII=%f #\n",i,j,exxn[0],ezzn[0],exzn[0],epsIIn[0]);
      }

      if (j==Nz-1) { // boundaries
        pointC.i = i; pointC.j = j; pointC.loc = ELEMENT; pointC.c = 0;
        ierr = DMStagGetPointStrainRates(dm,xlocal,1,&pointC,&epsIIc,&exxc,&ezzc,&exzc); CHKERRQ(ierr);
        exxn[2] = exxc;
        // PetscPrintf(PETSC_COMM_WORLD,"# U Strain [%d %d]: exx=%f ezz=%f exz=%f eII=%f #\n",i,j,exxn[2],ezzn[2],exzn[2],epsIIn[2]);
      }

      if ((i==Nx-1) && (j==Nz-1)) { // boundaries
        pointC.i = i; pointC.j = j; pointC.loc = ELEMENT; pointC.c = 0;
        ierr = DMStagGetPointStrainRates(dm,xlocal,1,&pointC,&epsIIc,&exxc,&ezzc,&exzc); CHKERRQ(ierr);
        exxn[3] = exxc;
        // PetscPrintf(PETSC_COMM_WORLD,"# U Strain [%d %d]: exx=%f ezz=%f exz=%f eII=%f #\n",i,j,exxn[2],ezzn[2],exzn[2],epsIIn[2]);
      }

      if ((i==0) || (i==Nx-1) || (j==0) || (j==Nz-1)) { // boundaries
        for (ii = 0; ii < 4; ii++) {
          epsIIn[ii] = PetscPowScalar(0.5*(exxn[ii]*exxn[ii] + ezzn[ii]*ezzn[ii] + 2.0*exzn[ii]*exzn[ii]),0.5);
        }
      }

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
  PetscInt       k,n_bc,*idx_bc;
  PetscScalar    *value_bc,*x_bc;
  BCType         *type_bc;
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
  
  // Vx=vi on left boundary (w)
  ierr = DMStagBCListGetValues(bclist,'w','-',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = usr->par->nd_vi;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  
  // Vx=-vi on right boundary (e)
  ierr = DMStagBCListGetValues(bclist,'e','-',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = -usr->par->nd_vi;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  
  // Vz=vi on top boundary (n)
  ierr = DMStagBCListGetValues(bclist,'n','|',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = usr->par->nd_vi;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  
  // Vz=-vi on bottom boundary (s)
  ierr = DMStagBCListGetValues(bclist,'s','|',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = -usr->par->nd_vi;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // // P=1 on left boundary (w)
  // ierr = DMStagBCListGetValues(bclist,'w','o',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  // for (k=0; k<n_bc; k++) {
  //   value_bc[k] = usr->par->nd_P;
  //   type_bc[k] = BC_DIRICHLET;
  // }
  // ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // // P=1 on right boundary (e)
  // ierr = DMStagBCListGetValues(bclist,'e','o',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  // for (k=0; k<n_bc; k++) {
  //   value_bc[k] = usr->par->nd_P;
  //   type_bc[k] = BC_DIRICHLET;
  // }
  // ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // P=1 on bottom boundary (s)
  ierr = DMStagBCListGetValues(bclist,'s','o',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = usr->par->nd_P;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // P=1 on top boundary (n)
  ierr = DMStagBCListGetValues(bclist,'n','o',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = usr->par->nd_P;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

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

  ierr = PetscBagRegisterScalar(bag, &par->xmin, 0.0, "xmin", "Start coordinate of domain in x-dir"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->zmin, 0.0, "zmin", "Start coordinate of domain in z-dir"); CHKERRQ(ierr);

  ierr = PetscBagRegisterScalar(bag, &par->L, 1.0e5, "L", "Length of domain in x-dir [m]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->H, 1.0e5, "H", "Height of domain in z-dir [m]"); CHKERRQ(ierr);

  // Physical and material parameters
  ierr = PetscBagRegisterScalar(bag, &par->eta0, 1.0e23, "eta0", "Block shear viscosity [Pa.s]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->eta_w, 1.0e17, "eta_w", "Weak zone shear viscosity [Pa.s]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->vi, 5.0e-9, "vi", "Extension/compression velocity [m/s]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->rho0, 3000, "rho0", "Reference density [kg/m3]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->g, 0.0, "g", "Gravitational acceleration [m/s2]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->C, 1.0e8, "C", "Cohesion [Pa]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->P, 1.0e8, "P", "Boundary pressure [Pa]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->phi, 37.0, "phi", "Angle of internal friction sin(phi) = 0.6 [deg]"); CHKERRQ(ierr);

  ierr = PetscBagRegisterInt(bag, &par->smooth, 0, "smooth", "0-rough punch, 1-smooth punch (Hill's)"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->harmonic, 0, "harmonic", "0-no 1-yes harmonic averaging"); CHKERRQ(ierr);
  
  // scales
  par->stress = par->C;
  par->length = par->H;
  par->visc   = 1e20;
  par->vel    = par->stress*par->length/par->visc; // stokes velocity

  // par->length = par->H;
  // par->visc   = 1e20;
  // par->vel    = par->vi; 
  // par->stress = par->vel*par->visc/par->length;

  // non-dimensionalize
  par->nd_eta0  = par->eta0/par->visc;
  par->nd_eta_w = par->eta_w/par->visc;
  par->nd_vi    = par->vi/par->vel;
  par->nd_C     = par->C/par->stress;
  par->nd_P     = par->P/par->stress;
  par->L        = par->L/par->length;
  par->H        = par->H/par->length;

  par->etamax = par->nd_eta0;
  par->etamin = par->nd_eta_w;
  par->C_factor = 1.0e8/par->C;

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
  PetscPrintf(usr->comm,"# Test_vp_inclusion_gerya: %s \n",&(date[0]));
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