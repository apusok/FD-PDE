// ---------------------------------------
// Shortening of a visco-plastic (von Mises criterion) block in the absence of gravity
// Setup from T. Gerya, 2018, Ch. 13, ex. 13.2
// run: ./tests/test_vp_inclusion_gerya.app -pc_type lu -pc_factor_mat_solver_type umfpack -snes_monitor -ksp_monitor -nx 20 -nz 20
// python test: ./tests/python/test_vp_inclusion_gerya.py
// ---------------------------------------
static char help[] = "Application for shortening of a visco-plastic block in the absence of gravity \n\n";

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
  PetscScalar    eta_b, eta_w, eta_i, vi, C_b, C_w, C_i, P;
  PetscScalar    stress, vel, length, visc; // scales
  PetscScalar    nd_eta_b, nd_eta_w, nd_eta_i, nd_vi, nd_C_b, nd_C_w, nd_C_i, etamax, etamin, nd_P; // non-dimensional
  PetscBool      plasticity;
  char           fname_out[FNAME_LENGTH]; 
} Params;

// user defined and model-dependent variables
typedef struct {
  Params        *par;
  PetscBag       bag;
  MPI_Comm       comm;
  PetscMPIInt    rank;
  DM             dmeps;
  Vec            xeps,xtau,xyield;
} UsrData;

// ---------------------------------------
// Function definitions
// ---------------------------------------
PetscErrorCode Numerical_solution(void*);
PetscErrorCode InputParameters(UsrData**);
PetscErrorCode InputPrintData(UsrData*);
PetscErrorCode FormCoefficient(FDPDE, DM, Vec, DM, Vec, void*);
PetscErrorCode FormCoefficientSplit(FDPDE, DM, Vec, Vec, DM, Vec, void*);
PetscErrorCode FormBCList(DM, Vec, DMStagBCList, void*);
PetscErrorCode UpdateStrainRates(DM,Vec,void*);
PetscErrorCode ScaleSolution(DM,Vec,Vec*,void*);
PetscErrorCode ScaleCoefficient(DM,Vec,Vec*,void*);
PetscErrorCode ScaleVectorUniform(DM,Vec,Vec*,PetscScalar);

const char coeff_description[] =
"  << Stokes Coefficients >> \n"
"  eta_n/eta_c = eta_eff\n"
"  fux = 0 \n" 
"  fuz = 0 (no body forces) \n" 
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
  Vec            x, xguess, xcoeff, xscaled;
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
  ierr = DMCreateGlobalVector(usr->dmeps,&usr->xtau); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(usr->dmeps,&usr->xyield); CHKERRQ(ierr);

  // Set coefficients and BC evaluation function
  ierr = FDPDESetFunctionBCList(fd,FormBCList,bc_description,usr); CHKERRQ(ierr);
  ierr = FDPDESetFunctionCoefficient(fd,FormCoefficient,coeff_description,usr); CHKERRQ(ierr);
  ierr = FDPDESetFunctionCoefficientSplit(fd,FormCoefficientSplit,coeff_description,usr); CHKERRQ(ierr);
  ierr = FDPDEView(fd); CHKERRQ(ierr);

  // Pin pressure
  ierr = FDPDEStokesPinPressure(fd,usr->par->nd_P,PETSC_TRUE); CHKERRQ(ierr);

  // Create initial guess with a linear viscous
  PetscPrintf(PETSC_COMM_WORLD,"\n# INITIAL GUESS #\n");
  ierr = FDPDESolve(fd,NULL);CHKERRQ(ierr);
  ierr = FDPDEGetSolution(fd,&x);CHKERRQ(ierr); 

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s_solution_initial",usr->par->fname_out);
  ierr = DMStagViewBinaryPython(dm,x,fout);CHKERRQ(ierr);

  ierr = ScaleSolution(dm,x,&xscaled,usr);CHKERRQ(ierr);
  ierr = PetscSNPrintf(fout,sizeof(fout),"%s_solution_initial_dim",usr->par->fname_out);
  ierr = DMStagViewBinaryPython(dm,xscaled,fout);CHKERRQ(ierr);
  ierr = VecDestroy(&xscaled);CHKERRQ(ierr);

  ierr = FDPDEGetCoefficient(fd,&dmcoeff,&xcoeff);CHKERRQ(ierr);
  ierr = PetscSNPrintf(fout,sizeof(fout),"%s_coefficient_initial",usr->par->fname_out);
  ierr = DMStagViewBinaryPython(dmcoeff,xcoeff,fout);CHKERRQ(ierr);

  ierr = ScaleCoefficient(dmcoeff,xcoeff,&xscaled,usr);CHKERRQ(ierr);
  ierr = PetscSNPrintf(fout,sizeof(fout),"%s_coefficient_initial_dim",usr->par->fname_out);
  ierr = DMStagViewBinaryPython(dmcoeff,xscaled,fout);CHKERRQ(ierr);
  ierr = VecDestroy(&xscaled);CHKERRQ(ierr);

  ierr = FDPDEGetSolutionGuess(fd,&xguess); CHKERRQ(ierr); 
  ierr = VecCopy(x,xguess);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&xguess);CHKERRQ(ierr);
  usr->par->plasticity = PETSC_TRUE;

  // MatView(fd->J,PETSC_VIEWER_STDOUT_WORLD);

#if 0
  {
    SNES snes_picard;
    Mat  J;
    DM   dmref,dm;
    
    PetscPrintf(PETSC_COMM_WORLD,"\n# PICARD SOLVE #\n");

    ierr = FDPDEGetDM(fd,&dmref);CHKERRQ(ierr);
    ierr = DMClone(dmref,&dm);CHKERRQ(ierr);

    ierr = fd->ops->create_jacobian(fd,&J);CHKERRQ(ierr);
    
    ierr = SNESCreate(fd->comm,&snes_picard);CHKERRQ(ierr);
    ierr = SNESSetOptionsPrefix(snes_picard,"p_");CHKERRQ(ierr);
    ierr = SNESSetDM(snes_picard,dm);CHKERRQ(ierr); /* attach a clone of the DM stag - see note on manpage for SNESSetDM() */
    ierr = SNESSetSolution(snes_picard,fd->x);CHKERRQ(ierr); // for FD colouring to function correctly
    
    ierr = SNESSetFunction(snes_picard,fd->r,SNESPicardComputeFunctionDefault,(void*)fd);CHKERRQ(ierr);
    
    ierr = SNESSetJacobian(snes_picard,J,J,SNESComputeJacobianDefaultColor,NULL);CHKERRQ(ierr);
    
    ierr = SNESSetType(snes_picard,SNESNPICARDLS);CHKERRQ(ierr);
    ierr = SNESSetFromOptions(snes_picard);CHKERRQ(ierr);
    ierr = SNESSetUp(snes_picard);CHKERRQ(ierr);

    ierr = SNESPicardLSSetSplitFunction(snes_picard,fd->r,fd->ops->form_function_split);CHKERRQ(ierr);

    {
      Vec xguess,x2;
      
      ierr = FDPDEGetSolutionGuess(fd,&xguess); CHKERRQ(ierr);
      ierr = SNESPicardLSGetAuxillarySolution(snes_picard,&x2);CHKERRQ(ierr);
      ierr = VecCopy(xguess,x2);CHKERRQ(ierr);
      ierr = VecDestroy(&xguess);CHKERRQ(ierr);
    }
    
    ierr = SNESSolve(snes_picard,NULL,fd->x);CHKERRQ(ierr);
    
    ierr = SNESDestroy(&snes_picard);CHKERRQ(ierr);
    ierr = DMDestroy(&dm);CHKERRQ(ierr);
    ierr = MatDestroy(&J);CHKERRQ(ierr);
  }
#endif
  
  PetscPrintf(PETSC_COMM_WORLD,"\n# PICARD SOLVE #\n");
  ierr = FDPDESolvePicard(fd,NULL);CHKERRQ(ierr);
  
  ierr = FDPDEGetSolution(fd,&x);CHKERRQ(ierr);
  ierr = FDPDEGetSolutionGuess(fd,&xguess); CHKERRQ(ierr);
  ierr = VecCopy(x,xguess);CHKERRQ(ierr);
  ierr = VecDestroy(&xguess);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);

  
  // FD SNES Solver
  PetscPrintf(PETSC_COMM_WORLD,"\n# SNES SOLVE #\n");
  ierr = FDPDESolve(fd,NULL);CHKERRQ(ierr);
  ierr = FDPDEGetSolution(fd,&x);CHKERRQ(ierr); 

  // Output solution to file
  ierr = PetscSNPrintf(fout,sizeof(fout),"%s_solution",usr->par->fname_out);
  ierr = DMStagViewBinaryPython(dm,x,fout);CHKERRQ(ierr);

  ierr = ScaleSolution(dm,x,&xscaled,usr);CHKERRQ(ierr);
  ierr = PetscSNPrintf(fout,sizeof(fout),"%s_solution_dim",usr->par->fname_out);
  ierr = DMStagViewBinaryPython(dm,xscaled,fout);CHKERRQ(ierr);
  ierr = VecDestroy(&xscaled);CHKERRQ(ierr);

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s_strain",usr->par->fname_out);
  ierr = DMStagViewBinaryPython(usr->dmeps,usr->xeps,fout);CHKERRQ(ierr);

  ierr = ScaleVectorUniform(usr->dmeps,usr->xeps,&xscaled,usr->par->stress/usr->par->visc);CHKERRQ(ierr);
  ierr = PetscSNPrintf(fout,sizeof(fout),"%s_strain_dim",usr->par->fname_out);
  ierr = DMStagViewBinaryPython(usr->dmeps,xscaled,fout);CHKERRQ(ierr);
  ierr = VecDestroy(&xscaled);CHKERRQ(ierr);

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s_stress",usr->par->fname_out);
  ierr = DMStagViewBinaryPython(usr->dmeps,usr->xtau,fout);CHKERRQ(ierr);

  ierr = ScaleVectorUniform(usr->dmeps,usr->xtau,&xscaled,usr->par->stress);CHKERRQ(ierr);
  ierr = PetscSNPrintf(fout,sizeof(fout),"%s_stress_dim",usr->par->fname_out);
  ierr = DMStagViewBinaryPython(usr->dmeps,xscaled,fout);CHKERRQ(ierr);
  ierr = VecDestroy(&xscaled);CHKERRQ(ierr);

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s_yield",usr->par->fname_out);
  ierr = DMStagViewBinaryPython(usr->dmeps,usr->xyield,fout);CHKERRQ(ierr);

  ierr = ScaleVectorUniform(usr->dmeps,usr->xyield,&xscaled,usr->par->stress);CHKERRQ(ierr);
  ierr = PetscSNPrintf(fout,sizeof(fout),"%s_yield_dim",usr->par->fname_out);
  ierr = DMStagViewBinaryPython(usr->dmeps,xscaled,fout);CHKERRQ(ierr);
  ierr = VecDestroy(&xscaled);CHKERRQ(ierr);

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s_residual",usr->par->fname_out);
  ierr = DMStagViewBinaryPython(dm,fd->r,fout);CHKERRQ(ierr);

  ierr = FDPDEGetCoefficient(fd,&dmcoeff,&xcoeff);CHKERRQ(ierr);
  ierr = PetscSNPrintf(fout,sizeof(fout),"%s_coefficient",usr->par->fname_out);
  ierr = DMStagViewBinaryPython(dmcoeff,xcoeff,fout);CHKERRQ(ierr);

  ierr = ScaleCoefficient(dmcoeff,xcoeff,&xscaled,usr);CHKERRQ(ierr);
  ierr = PetscSNPrintf(fout,sizeof(fout),"%s_coefficient_dim",usr->par->fname_out);
  ierr = DMStagViewBinaryPython(dmcoeff,xscaled,fout);CHKERRQ(ierr);
  ierr = VecDestroy(&xscaled);CHKERRQ(ierr);

  // Destroy objects
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xeps);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xtau);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xyield);CHKERRQ(ierr);
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
  Vec            coefflocal, xepslocal, xtaulocal, xyieldlocal;
  PetscScalar    **coordx,**coordz;
  PetscScalar    zblock_s, zblock_e, xis, xie, zis, zie;
  PetscInt       iprev, inext, icenter;
  PetscScalar    ***c, ***xxs, ***xxy;
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
  ierr = DMCreateLocalVector (usr->dmeps,&xtaulocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(usr->dmeps,xtaulocal,&xxs); CHKERRQ(ierr);

  ierr = DMCreateLocalVector (usr->dmeps,&xyieldlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(usr->dmeps,xyieldlocal,&xxy); CHKERRQ(ierr);

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

  // Loop over local domain - set initial density and viscosity
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      PetscInt idx;

      { // A = eta (center, c=1)
        DMStagStencil point;//, pointP;
        PetscScalar   Y,eta,eta_P,eta_VP,epsII,inv_eta, inv_eta_VP,exx,ezz,exz,txx,tzz,txz,tauII;

        if ((coordz[j][icenter]<zblock_s) || (coordz[j][icenter]>zblock_e)) { 
          eta = usr->par->nd_eta_w; // top/bottom layer
          Y   = usr->par->nd_C_w; // plastic yield criterion
        } else if ((coordx[i][icenter]>=xis) && (coordx[i][icenter]<=xie) && (coordz[j][icenter]>=zis) && (coordz[j][icenter]<=zie)) {
          eta = usr->par->nd_eta_i; // inclusion
          Y   = usr->par->nd_C_i; // plastic yield criterion
        } else {
          eta = usr->par->nd_eta_b; // block
          Y = usr->par->nd_C_b; // plastic yield criterion
        }

        // second invariant of strain rate
        point.i = i; point.j = j; point.loc = ELEMENT; 
        point.c = 0; ierr = DMStagVecGetValuesStencil(usr->dmeps,xepslocal,1,&point,&exx); CHKERRQ(ierr);
        point.c = 1; ierr = DMStagVecGetValuesStencil(usr->dmeps,xepslocal,1,&point,&ezz); CHKERRQ(ierr);
        point.c = 2; ierr = DMStagVecGetValuesStencil(usr->dmeps,xepslocal,1,&point,&exz); CHKERRQ(ierr);
        point.c = 3; ierr = DMStagVecGetValuesStencil(usr->dmeps,xepslocal,1,&point,&epsII); CHKERRQ(ierr);

        // second invariant of stress
        txx = 2.0*eta*exx;
        tzz = 2.0*eta*ezz;
        txz = 2.0*eta*exz;
        tauII = PetscPowScalar(0.5*(txx*txx + tzz*tzz + 2.0*txz*txz),0.5);

        if (usr->par->plasticity) {
          if (tauII > Y) {
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
        ierr = DMStagGetLocationSlot(usr->dmeps,ELEMENT,0,&idx); CHKERRQ(ierr); xxs[j][i][idx] = txx; xxy[j][i][idx] = Y;
        ierr = DMStagGetLocationSlot(usr->dmeps,ELEMENT,1,&idx); CHKERRQ(ierr); xxs[j][i][idx] = tzz;
        ierr = DMStagGetLocationSlot(usr->dmeps,ELEMENT,2,&idx); CHKERRQ(ierr); xxs[j][i][idx] = txz;
        ierr = DMStagGetLocationSlot(usr->dmeps,ELEMENT,3,&idx); CHKERRQ(ierr); xxs[j][i][idx] = tauII;
      }

      { // A = eta (corner, c=0)
        DMStagStencil point[4];
        PetscScalar   eta,epsII[4], Y[4], eta_P, eta_VP, inv_eta, inv_eta_VP, xp[4],zp[4];
        PetscScalar   exx[4],ezz[4],exz[4],txx[4],tzz[4],txz[4],tauII[4];
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

        // coordinates
        xp[0] = coordx[i][iprev]; zp[0] = coordz[j][iprev]; 
        xp[1] = coordx[i][inext]; zp[1] = coordz[j][iprev]; 
        xp[2] = coordx[i][iprev]; zp[2] = coordz[j][inext]; 
        xp[3] = coordx[i][inext]; zp[3] = coordz[j][inext]; 

        for (ii = 0; ii < 4; ii++) {
          if ((zp[ii]<zblock_s) || (zp[ii]>zblock_e)) { 
            eta = usr->par->nd_eta_w; // top/bottom layer
            Y[ii] = usr->par->nd_C_w; // plastic yield criterion
          } else if ((xp[ii]>=xis) && (xp[ii]<=xie) && (zp[ii]>=zis) && (zp[ii]<=zie)) {
            eta = usr->par->nd_eta_i; // inclusion
            Y[ii] = usr->par->nd_C_i; // plastic yield criterion
          } else {
            eta = usr->par->nd_eta_b; //block
            Y[ii] = usr->par->nd_C_b; // plastic yield criterion
          }

          // second invariant of stress
          txx[ii] = 2.0*eta*exx[ii];
          tzz[ii] = 2.0*eta*ezz[ii];
          txz[ii] = 2.0*eta*exz[ii];
          tauII[ii] = PetscPowScalar(0.5*(txx[ii]*txx[ii] + tzz[ii]*tzz[ii] + 2.0*txz[ii]*txz[ii]),0.5);

          if (usr->par->plasticity) {
            if (tauII[ii] > Y[ii]) {
              eta_P = Y[ii]/(2.0*epsII[ii]);

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
        ierr = DMStagGetLocationSlot(usr->dmeps,DOWN_LEFT,0,&idx); CHKERRQ(ierr); xxs[j][i][idx] = txx[0]; xxy[j][i][idx] = Y[0];
        ierr = DMStagGetLocationSlot(usr->dmeps,DOWN_LEFT,1,&idx); CHKERRQ(ierr); xxs[j][i][idx] = tzz[0];
        ierr = DMStagGetLocationSlot(usr->dmeps,DOWN_LEFT,2,&idx); CHKERRQ(ierr); xxs[j][i][idx] = txz[0];
        ierr = DMStagGetLocationSlot(usr->dmeps,DOWN_LEFT,3,&idx); CHKERRQ(ierr); xxs[j][i][idx] = tauII[0];

        ierr = DMStagGetLocationSlot(usr->dmeps,DOWN_RIGHT,0,&idx); CHKERRQ(ierr); xxs[j][i][idx] = txx[1]; xxy[j][i][idx] = Y[1];
        ierr = DMStagGetLocationSlot(usr->dmeps,DOWN_RIGHT,1,&idx); CHKERRQ(ierr); xxs[j][i][idx] = tzz[1];
        ierr = DMStagGetLocationSlot(usr->dmeps,DOWN_RIGHT,2,&idx); CHKERRQ(ierr); xxs[j][i][idx] = txz[1];
        ierr = DMStagGetLocationSlot(usr->dmeps,DOWN_RIGHT,3,&idx); CHKERRQ(ierr); xxs[j][i][idx] = tauII[1];

        ierr = DMStagGetLocationSlot(usr->dmeps,UP_LEFT,0,&idx); CHKERRQ(ierr); xxs[j][i][idx] = txx[2]; xxy[j][i][idx] = Y[2];
        ierr = DMStagGetLocationSlot(usr->dmeps,UP_LEFT,1,&idx); CHKERRQ(ierr); xxs[j][i][idx] = tzz[2];
        ierr = DMStagGetLocationSlot(usr->dmeps,UP_LEFT,2,&idx); CHKERRQ(ierr); xxs[j][i][idx] = txz[2];
        ierr = DMStagGetLocationSlot(usr->dmeps,UP_LEFT,3,&idx); CHKERRQ(ierr); xxs[j][i][idx] = tauII[2];

        ierr = DMStagGetLocationSlot(usr->dmeps,UP_RIGHT,0,&idx); CHKERRQ(ierr); xxs[j][i][idx] = txx[3]; xxy[j][i][idx] = Y[3];
        ierr = DMStagGetLocationSlot(usr->dmeps,UP_RIGHT,1,&idx); CHKERRQ(ierr); xxs[j][i][idx] = tzz[3];
        ierr = DMStagGetLocationSlot(usr->dmeps,UP_RIGHT,2,&idx); CHKERRQ(ierr); xxs[j][i][idx] = txz[3];
        ierr = DMStagGetLocationSlot(usr->dmeps,UP_RIGHT,3,&idx); CHKERRQ(ierr); xxs[j][i][idx] = tauII[3];
      }

      { // B = 0.0 (edges, c=0)
        DMStagStencil point[4];
        PetscScalar   rhs[4];
        PetscInt      ii;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 0;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = 0;
        point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = 0;

        rhs[0] = 0.0;
        rhs[1] = 0.0;
        rhs[2] = 0.0;
        rhs[3] = 0.0;

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
  ierr = DMStagVecRestoreArray(usr->dmeps,xtaulocal,&xxs); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(usr->dmeps,xtaulocal,INSERT_VALUES,usr->xtau); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (usr->dmeps,xtaulocal,INSERT_VALUES,usr->xtau); CHKERRQ(ierr);
  ierr = VecDestroy(&xtaulocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArray(usr->dmeps,xyieldlocal,&xxy); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(usr->dmeps,xyieldlocal,INSERT_VALUES,usr->xyield); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (usr->dmeps,xyieldlocal,INSERT_VALUES,usr->xyield); CHKERRQ(ierr);
  ierr = VecDestroy(&xyieldlocal); CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(usr->dmeps,&xepslocal); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode FormCoefficientSplit(FDPDE fd, DM dm, Vec x, Vec x2, DM dmcoeff, Vec coeff, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz;
  Vec            coefflocal, xepslocal, xtaulocal, xyieldlocal;
  PetscScalar    **coordx,**coordz;
  PetscScalar    zblock_s, zblock_e, xis, xie, zis, zie;
  PetscInt       iprev, inext, icenter;
  PetscScalar    ***c, ***xxs, ***xxy;
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
  ierr = UpdateStrainRates(dm,x2,usr); CHKERRQ(ierr);
  ierr = DMGetLocalVector(usr->dmeps, &xepslocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmeps, usr->xeps, INSERT_VALUES, xepslocal); CHKERRQ(ierr);
  
  // Local vectors
  ierr = DMCreateLocalVector (usr->dmeps,&xtaulocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(usr->dmeps,xtaulocal,&xxs); CHKERRQ(ierr);
  
  ierr = DMCreateLocalVector (usr->dmeps,&xyieldlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(usr->dmeps,xyieldlocal,&xxy); CHKERRQ(ierr);
  
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
  
  // Loop over local domain - set initial density and viscosity
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      PetscInt idx;
      
      { // A = eta (center, c=1)
        DMStagStencil point;//, pointP;
        PetscScalar   Y,eta,eta_P,eta_VP,epsII,inv_eta, inv_eta_VP,exx,ezz,exz,txx,tzz,txz,tauII;
        
        if ((coordz[j][icenter]<zblock_s) || (coordz[j][icenter]>zblock_e)) {
          eta = usr->par->nd_eta_w; // top/bottom layer
          Y   = usr->par->nd_C_w; // plastic yield criterion
        } else if ((coordx[i][icenter]>=xis) && (coordx[i][icenter]<=xie) && (coordz[j][icenter]>=zis) && (coordz[j][icenter]<=zie)) {
          eta = usr->par->nd_eta_i; // inclusion
          Y   = usr->par->nd_C_i; // plastic yield criterion
        } else {
          eta = usr->par->nd_eta_b; // block
          Y = usr->par->nd_C_b; // plastic yield criterion
        }
        
        // second invariant of strain rate
        point.i = i; point.j = j; point.loc = ELEMENT;
        point.c = 0; ierr = DMStagVecGetValuesStencil(usr->dmeps,xepslocal,1,&point,&exx); CHKERRQ(ierr);
        point.c = 1; ierr = DMStagVecGetValuesStencil(usr->dmeps,xepslocal,1,&point,&ezz); CHKERRQ(ierr);
        point.c = 2; ierr = DMStagVecGetValuesStencil(usr->dmeps,xepslocal,1,&point,&exz); CHKERRQ(ierr);
        point.c = 3; ierr = DMStagVecGetValuesStencil(usr->dmeps,xepslocal,1,&point,&epsII); CHKERRQ(ierr);
        
        // second invariant of stress
        txx = 2.0*eta*exx;
        tzz = 2.0*eta*ezz;
        txz = 2.0*eta*exz;
        tauII = PetscPowScalar(0.5*(txx*txx + tzz*tzz + 2.0*txz*txz),0.5);
        
        if (usr->par->plasticity) {
          if (tauII > Y) {
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
        ierr = DMStagGetLocationSlot(usr->dmeps,ELEMENT,0,&idx); CHKERRQ(ierr); xxs[j][i][idx] = txx; xxy[j][i][idx] = Y;
        ierr = DMStagGetLocationSlot(usr->dmeps,ELEMENT,1,&idx); CHKERRQ(ierr); xxs[j][i][idx] = tzz;
        ierr = DMStagGetLocationSlot(usr->dmeps,ELEMENT,2,&idx); CHKERRQ(ierr); xxs[j][i][idx] = txz;
        ierr = DMStagGetLocationSlot(usr->dmeps,ELEMENT,3,&idx); CHKERRQ(ierr); xxs[j][i][idx] = tauII;
      }
      
      { // A = eta (corner, c=0)
        DMStagStencil point[4];
        PetscScalar   eta,epsII[4], Y[4], eta_P, eta_VP, inv_eta, inv_eta_VP, xp[4],zp[4];
        PetscScalar   exx[4],ezz[4],exz[4],txx[4],tzz[4],txz[4],tauII[4];
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
        
        // coordinates
        xp[0] = coordx[i][iprev]; zp[0] = coordz[j][iprev];
        xp[1] = coordx[i][inext]; zp[1] = coordz[j][iprev];
        xp[2] = coordx[i][iprev]; zp[2] = coordz[j][inext];
        xp[3] = coordx[i][inext]; zp[3] = coordz[j][inext];
        
        for (ii = 0; ii < 4; ii++) {
          if ((zp[ii]<zblock_s) || (zp[ii]>zblock_e)) {
            eta = usr->par->nd_eta_w; // top/bottom layer
            Y[ii] = usr->par->nd_C_w; // plastic yield criterion
          } else if ((xp[ii]>=xis) && (xp[ii]<=xie) && (zp[ii]>=zis) && (zp[ii]<=zie)) {
            eta = usr->par->nd_eta_i; // inclusion
            Y[ii] = usr->par->nd_C_i; // plastic yield criterion
          } else {
            eta = usr->par->nd_eta_b; //block
            Y[ii] = usr->par->nd_C_b; // plastic yield criterion
          }
          
          // second invariant of stress
          txx[ii] = 2.0*eta*exx[ii];
          tzz[ii] = 2.0*eta*ezz[ii];
          txz[ii] = 2.0*eta*exz[ii];
          tauII[ii] = PetscPowScalar(0.5*(txx[ii]*txx[ii] + tzz[ii]*tzz[ii] + 2.0*txz[ii]*txz[ii]),0.5);
          
          if (usr->par->plasticity) {
            if (tauII[ii] > Y[ii]) {
              eta_P = Y[ii]/(2.0*epsII[ii]);
              
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
        ierr = DMStagGetLocationSlot(usr->dmeps,DOWN_LEFT,0,&idx); CHKERRQ(ierr); xxs[j][i][idx] = txx[0]; xxy[j][i][idx] = Y[0];
        ierr = DMStagGetLocationSlot(usr->dmeps,DOWN_LEFT,1,&idx); CHKERRQ(ierr); xxs[j][i][idx] = tzz[0];
        ierr = DMStagGetLocationSlot(usr->dmeps,DOWN_LEFT,2,&idx); CHKERRQ(ierr); xxs[j][i][idx] = txz[0];
        ierr = DMStagGetLocationSlot(usr->dmeps,DOWN_LEFT,3,&idx); CHKERRQ(ierr); xxs[j][i][idx] = tauII[0];
        
        ierr = DMStagGetLocationSlot(usr->dmeps,DOWN_RIGHT,0,&idx); CHKERRQ(ierr); xxs[j][i][idx] = txx[1]; xxy[j][i][idx] = Y[1];
        ierr = DMStagGetLocationSlot(usr->dmeps,DOWN_RIGHT,1,&idx); CHKERRQ(ierr); xxs[j][i][idx] = tzz[1];
        ierr = DMStagGetLocationSlot(usr->dmeps,DOWN_RIGHT,2,&idx); CHKERRQ(ierr); xxs[j][i][idx] = txz[1];
        ierr = DMStagGetLocationSlot(usr->dmeps,DOWN_RIGHT,3,&idx); CHKERRQ(ierr); xxs[j][i][idx] = tauII[1];
        
        ierr = DMStagGetLocationSlot(usr->dmeps,UP_LEFT,0,&idx); CHKERRQ(ierr); xxs[j][i][idx] = txx[2]; xxy[j][i][idx] = Y[2];
        ierr = DMStagGetLocationSlot(usr->dmeps,UP_LEFT,1,&idx); CHKERRQ(ierr); xxs[j][i][idx] = tzz[2];
        ierr = DMStagGetLocationSlot(usr->dmeps,UP_LEFT,2,&idx); CHKERRQ(ierr); xxs[j][i][idx] = txz[2];
        ierr = DMStagGetLocationSlot(usr->dmeps,UP_LEFT,3,&idx); CHKERRQ(ierr); xxs[j][i][idx] = tauII[2];
        
        ierr = DMStagGetLocationSlot(usr->dmeps,UP_RIGHT,0,&idx); CHKERRQ(ierr); xxs[j][i][idx] = txx[3]; xxy[j][i][idx] = Y[3];
        ierr = DMStagGetLocationSlot(usr->dmeps,UP_RIGHT,1,&idx); CHKERRQ(ierr); xxs[j][i][idx] = tzz[3];
        ierr = DMStagGetLocationSlot(usr->dmeps,UP_RIGHT,2,&idx); CHKERRQ(ierr); xxs[j][i][idx] = txz[3];
        ierr = DMStagGetLocationSlot(usr->dmeps,UP_RIGHT,3,&idx); CHKERRQ(ierr); xxs[j][i][idx] = tauII[3];
      }
      
      { // B = 0.0 (edges, c=0)
        DMStagStencil point[4];
        PetscScalar   rhs[4];
        PetscInt      ii;
        
        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 0;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = 0;
        point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = 0;
        
        rhs[0] = 0.0;
        rhs[1] = 0.0;
        rhs[2] = 0.0;
        rhs[3] = 0.0;
        
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
  ierr = DMStagVecRestoreArray(usr->dmeps,xtaulocal,&xxs); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(usr->dmeps,xtaulocal,INSERT_VALUES,usr->xtau); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (usr->dmeps,xtaulocal,INSERT_VALUES,usr->xtau); CHKERRQ(ierr);
  ierr = VecDestroy(&xtaulocal); CHKERRQ(ierr);
  
  ierr = DMStagVecRestoreArray(usr->dmeps,xyieldlocal,&xxy); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(usr->dmeps,xyieldlocal,INSERT_VALUES,usr->xyield); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (usr->dmeps,xyieldlocal,INSERT_VALUES,usr->xyield); CHKERRQ(ierr);
  ierr = VecDestroy(&xyieldlocal); CHKERRQ(ierr);
  
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
      }

      if (i==Nx-1) { // boundaries
        pointC.i = i; pointC.j = j; pointC.loc = ELEMENT; pointC.c = 0;
        ierr = DMStagGetPointStrainRates(dm,xlocal,1,&pointC,&epsIIc,&exxc,&ezzc,&exzc); CHKERRQ(ierr);
        ezzn[1] = ezzc;
      }

      if (j==0) { // boundaries
        pointC.i = i; pointC.j = j; pointC.loc = ELEMENT; pointC.c = 0;
        ierr = DMStagGetPointStrainRates(dm,xlocal,1,&pointC,&epsIIc,&exxc,&ezzc,&exzc); CHKERRQ(ierr);
        exxn[0] = exxc;
      }

      if (j==Nz-1) { // boundaries
        pointC.i = i; pointC.j = j; pointC.loc = ELEMENT; pointC.c = 0;
        ierr = DMStagGetPointStrainRates(dm,xlocal,1,&pointC,&epsIIc,&exxc,&ezzc,&exzc); CHKERRQ(ierr);
        exxn[2] = exxc;
      }

      if ((i==Nx-1) && (j==Nz-1)) { // boundaries
        pointC.i = i; pointC.j = j; pointC.loc = ELEMENT; pointC.c = 0;
        ierr = DMStagGetPointStrainRates(dm,xlocal,1,&pointC,&epsIIc,&exxc,&ezzc,&exzc); CHKERRQ(ierr);
        exxn[3] = exxc;
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
// Dimensionalize
// ---------------------------------------
PetscErrorCode ScaleSolution(DM dm, Vec x, Vec *_x, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  PetscInt       i, j, sx, sz, nx, nz,idx;
  PetscScalar    ***xxnew, ***xx;
  Vec            xnew, xnewlocal, xlocal;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Create local and global vector associated with DM
  ierr = VecDuplicate(x,&xnew);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dm,&xnewlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm,xnewlocal,&xxnew); CHKERRQ(ierr);

  ierr = DMGetLocalVector(dm,&xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm,x,INSERT_VALUES,xlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm,xlocal,&xx); CHKERRQ(ierr);

  // Get domain corners
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      ierr = DMStagGetLocationSlot(dm,ELEMENT,0,&idx); CHKERRQ(ierr); 
      xxnew[j][i][idx] = xx[j][i][idx]*usr->par->stress;

      ierr = DMStagGetLocationSlot(dm,LEFT,0,&idx); CHKERRQ(ierr); 
      xxnew[j][i][idx] = xx[j][i][idx]*usr->par->vel;

      ierr = DMStagGetLocationSlot(dm,RIGHT,0,&idx); CHKERRQ(ierr); 
      xxnew[j][i][idx] = xx[j][i][idx]*usr->par->vel;

      ierr = DMStagGetLocationSlot(dm,DOWN,0,&idx); CHKERRQ(ierr); 
      xxnew[j][i][idx] = xx[j][i][idx]*usr->par->vel;

      ierr = DMStagGetLocationSlot(dm,UP,0,&idx); CHKERRQ(ierr); 
      xxnew[j][i][idx] = xx[j][i][idx]*usr->par->vel;
    }
  }

  // Restore arrays
  ierr = DMStagVecRestoreArray(dm,xnewlocal,&xxnew); CHKERRQ(ierr);
  ierr = DMStagVecRestoreArray(dm,xlocal,&xx); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &xlocal ); CHKERRQ(ierr);

  // Map local to global
  ierr = DMLocalToGlobalBegin(dm,xnewlocal,INSERT_VALUES,xnew); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,xnewlocal,INSERT_VALUES,xnew); CHKERRQ(ierr);
  ierr = VecDestroy(&xnewlocal); CHKERRQ(ierr);

  // Assign pointers
  *_x  = xnew;
  
  PetscFunctionReturn(0);
}

PetscErrorCode ScaleCoefficient(DM dm, Vec x, Vec *_x, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  PetscInt       i, j, sx, sz, nx, nz,idx;
  PetscScalar    ***xxnew, ***xx;
  Vec            xnew, xnewlocal, xlocal;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Create local and global vector associated with DM
  ierr = VecDuplicate(x,&xnew);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dm,&xnewlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm,xnewlocal,&xxnew); CHKERRQ(ierr);

  ierr = DMGetLocalVector(dm,&xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm,x,INSERT_VALUES,xlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm,xlocal,&xx); CHKERRQ(ierr);

  // Get domain corners
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      ierr = DMStagGetLocationSlot(dm,ELEMENT,0,&idx); CHKERRQ(ierr); // C
      xxnew[j][i][idx] = xx[j][i][idx]*usr->par->vel/usr->par->length;

      ierr = DMStagGetLocationSlot(dm,ELEMENT,1,&idx); CHKERRQ(ierr); // A=eta
      xxnew[j][i][idx] = xx[j][i][idx]*usr->par->visc;

      ierr = DMStagGetLocationSlot(dm,LEFT,0,&idx); CHKERRQ(ierr); // Bx
      xxnew[j][i][idx] = xx[j][i][idx]*usr->par->stress/usr->par->length;

      ierr = DMStagGetLocationSlot(dm,RIGHT,0,&idx); CHKERRQ(ierr); // Bx
      xxnew[j][i][idx] = xx[j][i][idx]*usr->par->stress/usr->par->length;

      ierr = DMStagGetLocationSlot(dm,DOWN,0,&idx); CHKERRQ(ierr); //Bz
      xxnew[j][i][idx] = xx[j][i][idx]*usr->par->stress/usr->par->length;

      ierr = DMStagGetLocationSlot(dm,UP,0,&idx); CHKERRQ(ierr); // Bz
      xxnew[j][i][idx] = xx[j][i][idx]*usr->par->stress/usr->par->length;

      ierr = DMStagGetLocationSlot(dm,DOWN_LEFT,0,&idx); CHKERRQ(ierr); // A corner
      xxnew[j][i][idx] = xx[j][i][idx]*usr->par->visc;

      ierr = DMStagGetLocationSlot(dm,DOWN_RIGHT,0,&idx); CHKERRQ(ierr);
      xxnew[j][i][idx] = xx[j][i][idx]*usr->par->visc;

      ierr = DMStagGetLocationSlot(dm,UP_LEFT,0,&idx); CHKERRQ(ierr);
      xxnew[j][i][idx] = xx[j][i][idx]*usr->par->visc;

      ierr = DMStagGetLocationSlot(dm,UP_RIGHT,0,&idx); CHKERRQ(ierr);
      xxnew[j][i][idx] = xx[j][i][idx]*usr->par->visc;
    }
  }

  // Restore arrays
  ierr = DMStagVecRestoreArray(dm,xnewlocal,&xxnew); CHKERRQ(ierr);
  ierr = DMStagVecRestoreArray(dm,xlocal,&xx); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &xlocal ); CHKERRQ(ierr);

  // Map local to global
  ierr = DMLocalToGlobalBegin(dm,xnewlocal,INSERT_VALUES,xnew); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,xnewlocal,INSERT_VALUES,xnew); CHKERRQ(ierr);
  ierr = VecDestroy(&xnewlocal); CHKERRQ(ierr);

  // Assign pointers
  *_x  = xnew;
  
  PetscFunctionReturn(0);
}

PetscErrorCode ScaleVectorUniform(DM dm, Vec x, Vec *_x, PetscScalar scal)
{
  PetscInt       i, j, sx, sz, nx, nz,idx,dof0,dof1,dof2, ii;
  PetscScalar    ***xxnew, ***xx;
  Vec            xnew, xnewlocal, xlocal;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Create local and global vector associated with DM
  ierr = VecDuplicate(x,&xnew);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dm,&xnewlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm,xnewlocal,&xxnew); CHKERRQ(ierr);

  ierr = DMGetLocalVector(dm,&xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm,x,INSERT_VALUES,xlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm,xlocal,&xx); CHKERRQ(ierr);

  // Get domain corners
  ierr = DMStagGetDOF(dm,&dof0,&dof1,&dof2,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      for (ii = 0; ii <dof2; ii++) {
        ierr = DMStagGetLocationSlot(dm,ELEMENT,ii,&idx); CHKERRQ(ierr); // element
        xxnew[j][i][idx] = xx[j][i][idx]*scal;
      }

      for (ii = 0; ii <dof1; ii++) { // faces
        ierr = DMStagGetLocationSlot(dm,LEFT,ii,&idx); CHKERRQ(ierr);
        xxnew[j][i][idx] = xx[j][i][idx]*scal;

        ierr = DMStagGetLocationSlot(dm,RIGHT,ii,&idx); CHKERRQ(ierr);
        xxnew[j][i][idx] = xx[j][i][idx]*scal;

        ierr = DMStagGetLocationSlot(dm,DOWN,ii,&idx); CHKERRQ(ierr);
        xxnew[j][i][idx] = xx[j][i][idx]*scal;

        ierr = DMStagGetLocationSlot(dm,UP,ii,&idx); CHKERRQ(ierr);
        xxnew[j][i][idx] = xx[j][i][idx]*scal;
      }

      for (ii = 0; ii <dof0; ii++) { // nodes
        ierr = DMStagGetLocationSlot(dm,DOWN_LEFT,ii,&idx); CHKERRQ(ierr);
        xxnew[j][i][idx] = xx[j][i][idx]*scal;

        ierr = DMStagGetLocationSlot(dm,DOWN_RIGHT,ii,&idx); CHKERRQ(ierr);
        xxnew[j][i][idx] = xx[j][i][idx]*scal;

        ierr = DMStagGetLocationSlot(dm,UP_LEFT,ii,&idx); CHKERRQ(ierr);
        xxnew[j][i][idx] = xx[j][i][idx]*scal;

        ierr = DMStagGetLocationSlot(dm,UP_RIGHT,ii,&idx); CHKERRQ(ierr);
        xxnew[j][i][idx] = xx[j][i][idx]*scal;
      }
    }
  }

  // Restore arrays
  ierr = DMStagVecRestoreArray(dm,xnewlocal,&xxnew); CHKERRQ(ierr);
  ierr = DMStagVecRestoreArray(dm,xlocal,&xx); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &xlocal ); CHKERRQ(ierr);

  // Map local to global
  ierr = DMLocalToGlobalBegin(dm,xnewlocal,INSERT_VALUES,xnew); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,xnewlocal,INSERT_VALUES,xnew); CHKERRQ(ierr);
  ierr = VecDestroy(&xnewlocal); CHKERRQ(ierr);

  // Assign pointers
  *_x  = xnew;
  
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
  PetscScalar    *value_bc;
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
  ierr = PetscBagRegisterScalar(bag, &par->eta_b, 1.0e23, "eta_b", "Block shear viscosity [Pa.s]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->eta_i, 1.0e17, "eta_i", "Inclusion shear viscosity [Pa.s]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->eta_w, 1.0e17, "eta_w", "Weak zone shear viscosity [Pa.s]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->vi, 5.0e-9, "vi", "Extension/compression velocity [m/s]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->C_b, 1.0e8, "C_b", "Block Cohesion [Pa]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->C_i, 1.0e7, "C_i", "Inclusion Cohesion [Pa]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->C_w, 1.0e7, "C_w", "Weak zone Cohesion [Pa]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->P, 1.0e8, "P", "Boundary pressure [Pa]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->harmonic, 0, "harmonic", "0-no 1-yes harmonic averaging"); CHKERRQ(ierr);
  
  // scales
  par->stress = 1e8;
  par->length = par->H;
  par->visc   = 1e20;
  par->vel    = par->stress*par->length/par->visc; // stokes velocity

  // par->length = par->H;
  // par->visc   = 1e20;
  // par->vel    = par->vi; 
  // par->stress = par->vel*par->visc/par->length;

  // non-dimensionalize
  par->nd_eta_b  = par->eta_b/par->visc;
  par->nd_eta_w = par->eta_w/par->visc;
  par->nd_eta_i = par->eta_i/par->visc;
  par->nd_vi    = par->vi/par->vel;
  par->nd_C_b   = par->C_b/par->stress;
  par->nd_C_w   = par->C_w/par->stress;
  par->nd_C_i   = par->C_i/par->stress;
  par->nd_P     = par->P/par->stress;
  par->L        = par->L/par->length;
  par->H        = par->H/par->length;

  par->etamax = par->nd_eta_b;
  par->etamin = par->nd_eta_w;

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
  ierr = SNESRegister(SNESNPICARDLS,SNESCreate_PicardLS);CHKERRQ(ierr);
  
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
