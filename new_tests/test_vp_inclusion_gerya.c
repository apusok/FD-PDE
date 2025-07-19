// ---------------------------------------
// Shortening of a visco-plastic (von Mises criterion) block in the absence of gravity
// Setup from T. Gerya, 2018, Ch. 13, ex. 13.2
// run: ./test_vp_inclusion_gerya -pc_type lu -pc_factor_mat_solver_type umfpack -pc_factor_mat_ordering_type external -snes_monitor -ksp_monitor -nx 20 -nz 20 -log_view
// python test: ./python/test_vp_inclusion_gerya.py
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

#include "../new_src/fdpde_stokes.h"

// ---------------------------------------
// Application Context
// ---------------------------------------
#define FNAME_LENGTH 200

// parameters (bag)
typedef struct {
  PetscInt       nx, nz;
  PetscScalar    H;
  PetscScalar    eta_b, eta_w, vi, C_b;
  PetscScalar    stress, vel, length, visc; // scales
  PetscScalar    nd_H, nd_eta_b, nd_eta_w, nd_vi, nd_C_b, nd_C_w, etamax, etamin, nd_P; // non-dimensional
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
  PetscFunctionBeginUser;

  // Element count
  nx = usr->par->nx;
  nz = usr->par->nz;

  // Domain coords
  xmin = 0;  zmin = 0;
  xmax = 1;  zmax = 1;

  // Create the FD-pde object
  PetscCall(FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_STOKES,&fd));
  PetscCall(FDPDESetUp(fd));
  PetscCall(FDPDEGetDM(fd,&dm)); 

  // Create DM/vec for strain rates
  PetscCall(DMStagCreateCompatibleDMStag(dm,4,0,4,0,&usr->dmeps)); 
  PetscCall(DMSetUp(usr->dmeps)); 
  PetscCall(DMStagSetUniformCoordinatesProduct(usr->dmeps,xmin,xmax,zmin,zmax,0.0,0.0));
  PetscCall(DMCreateGlobalVector(usr->dmeps,&usr->xeps)); 
  PetscCall(DMCreateGlobalVector(usr->dmeps,&usr->xtau)); 
  PetscCall(DMCreateGlobalVector(usr->dmeps,&usr->xyield)); 

  // Set coefficients and BC evaluation function
  PetscCall(FDPDESetFunctionBCList(fd,FormBCList,bc_description,usr)); 
  PetscCall(FDPDESetFunctionCoefficient(fd,FormCoefficient,coeff_description,usr)); 
  PetscCall(FDPDESetFunctionCoefficientSplit(fd,FormCoefficientSplit,coeff_description,usr)); 
  PetscCall(FDPDEView(fd)); 

  // Create initial guess with a linear viscous
  PetscPrintf(PETSC_COMM_WORLD,"\n# INITIAL GUESS #\n");
  PetscCall(FDPDESolve(fd,NULL));
  PetscCall(FDPDEGetSolution(fd,&x)); 

  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s_solution_initial",usr->par->fname_out));
  PetscCall(DMStagViewBinaryPython(dm,x,fout));

  PetscCall(ScaleSolution(dm,x,&xscaled,usr));
  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s_solution_initial_dim",usr->par->fname_out));
  PetscCall(DMStagViewBinaryPython(dm,xscaled,fout));
  PetscCall(VecDestroy(&xscaled));

  PetscCall(FDPDEGetCoefficient(fd,&dmcoeff,&xcoeff));
  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s_coefficient_initial",usr->par->fname_out));
  PetscCall(DMStagViewBinaryPython(dmcoeff,xcoeff,fout));

  PetscCall(ScaleCoefficient(dmcoeff,xcoeff,&xscaled,usr));
  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s_coefficient_initial_dim",usr->par->fname_out));
  PetscCall(DMStagViewBinaryPython(dmcoeff,xscaled,fout));
  PetscCall(VecDestroy(&xscaled));

  PetscCall(FDPDEGetSolutionGuess(fd,&xguess));  
  PetscCall(VecCopy(x,xguess));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&xguess));
  usr->par->plasticity = PETSC_TRUE;

  // MatView(fd->J,PETSC_VIEWER_STDOUT_WORLD);

  // PICARD SNES Solver
  PetscPrintf(PETSC_COMM_WORLD,"\n# PICARD SOLVE #\n");
  PetscCall(FDPDESolvePicard(fd,NULL));
  
  PetscCall(FDPDEGetSolution(fd,&x));
  PetscCall(FDPDEGetSolutionGuess(fd,&xguess)); 
  PetscCall(VecCopy(x,xguess));
  PetscCall(VecDestroy(&xguess));
  PetscCall(VecDestroy(&x));

  
  // FD SNES Solver
  PetscPrintf(PETSC_COMM_WORLD,"\n# SNES SOLVE #\n");
  PetscCall(FDPDESolve(fd,NULL));
  PetscCall(FDPDEGetSolution(fd,&x)); 

  // Output solution to file
  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s_solution",usr->par->fname_out));
  PetscCall(DMStagViewBinaryPython(dm,x,fout));

  PetscCall(ScaleSolution(dm,x,&xscaled,usr));
  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s_solution_dim",usr->par->fname_out));
  PetscCall(DMStagViewBinaryPython(dm,xscaled,fout));
  PetscCall(VecDestroy(&xscaled));

  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s_strain",usr->par->fname_out));
  PetscCall(DMStagViewBinaryPython(usr->dmeps,usr->xeps,fout));

  PetscCall(ScaleVectorUniform(usr->dmeps,usr->xeps,&xscaled,usr->par->stress/usr->par->visc));
  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s_strain_dim",usr->par->fname_out));
  PetscCall(DMStagViewBinaryPython(usr->dmeps,xscaled,fout));
  PetscCall(VecDestroy(&xscaled));

  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s_stress",usr->par->fname_out));
  PetscCall(DMStagViewBinaryPython(usr->dmeps,usr->xtau,fout));

  PetscCall(ScaleVectorUniform(usr->dmeps,usr->xtau,&xscaled,usr->par->stress));
  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s_stress_dim",usr->par->fname_out));
  PetscCall(DMStagViewBinaryPython(usr->dmeps,xscaled,fout));
  PetscCall(VecDestroy(&xscaled));

  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s_yield",usr->par->fname_out));
  PetscCall(DMStagViewBinaryPython(usr->dmeps,usr->xyield,fout));

  PetscCall(ScaleVectorUniform(usr->dmeps,usr->xyield,&xscaled,usr->par->stress));
  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s_yield_dim",usr->par->fname_out));
  PetscCall(DMStagViewBinaryPython(usr->dmeps,xscaled,fout));
  PetscCall(VecDestroy(&xscaled));

  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s_residual",usr->par->fname_out));
  PetscCall(DMStagViewBinaryPython(dm,fd->r,fout));

  PetscCall(FDPDEGetCoefficient(fd,&dmcoeff,&xcoeff));
  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s_coefficient",usr->par->fname_out));
  PetscCall(DMStagViewBinaryPython(dmcoeff,xcoeff,fout));

  PetscCall(ScaleCoefficient(dmcoeff,xcoeff,&xscaled,usr));
  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s_coefficient_dim",usr->par->fname_out));
  PetscCall(DMStagViewBinaryPython(dmcoeff,xscaled,fout));
  PetscCall(VecDestroy(&xscaled));

  // Destroy objects
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&usr->xeps));
  PetscCall(VecDestroy(&usr->xtau));
  PetscCall(VecDestroy(&usr->xyield));
  PetscCall(DMDestroy(&dm));
  PetscCall(DMDestroy(&usr->dmeps));
  PetscCall(FDPDEDestroy(&fd));

  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscFunctionBeginUser;

  // block and weak inclusion params
  zblock_s = 0.2*usr->par->nd_H;
  zblock_e = 0.8*usr->par->nd_H;
  xis = usr->par->nd_H/2-0.1*usr->par->nd_H/2;
  xie = usr->par->nd_H/2+0.1*usr->par->nd_H/2;
  zis = usr->par->nd_H/2-0.1*usr->par->nd_H/2;
  zie = usr->par->nd_H/2+0.1*usr->par->nd_H/2;

  // Strain rates
  PetscCall(UpdateStrainRates(dm,x,usr)); 
  PetscCall(DMGetLocalVector(usr->dmeps, &xepslocal)); 
  PetscCall(DMGlobalToLocal (usr->dmeps, usr->xeps, INSERT_VALUES, xepslocal)); 

  // Local vectors
  PetscCall(DMCreateLocalVector (usr->dmeps,&xtaulocal)); 
  PetscCall(DMStagVecGetArray(usr->dmeps,xtaulocal,&xxs)); 

  PetscCall(DMCreateLocalVector (usr->dmeps,&xyieldlocal)); 
  PetscCall(DMStagVecGetArray(usr->dmeps,xyieldlocal,&xxy)); 

  // Get domain corners
  PetscCall(DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 
  PetscCall(DMStagGetGlobalSizes(dmcoeff,&Nx,&Nz,NULL));

  // Get dm coordinates array
  PetscCall(DMStagGetProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmcoeff,LEFT,&iprev));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmcoeff,RIGHT,&inext)); 
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmcoeff,ELEMENT,&icenter));

  // Create coefficient local vector
  PetscCall(DMCreateLocalVector(dmcoeff, &coefflocal)); 
  PetscCall(DMStagVecGetArray(dmcoeff, coefflocal, &c)); 

  // Loop over local domain - set initial density and viscosity
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      PetscInt idx;

      { // A = eta (center, c=1)
        DMStagStencil point;//, pointP;
        PetscScalar   Y,eta,eta_P,epsII,exx,ezz,exz,txx,tzz,txz,tauII;

        if ((coordz[j][icenter]<zblock_s) || (coordz[j][icenter]>zblock_e)) { 
          eta = usr->par->nd_eta_w; // top/bottom layer
          Y   = usr->par->nd_C_w; // plastic yield criterion
        } else if ((coordx[i][icenter]>=xis) && (coordx[i][icenter]<=xie) && (coordz[j][icenter]>=zis) && (coordz[j][icenter]<=zie)) {
          eta = usr->par->nd_eta_w; // inclusion
          Y   = usr->par->nd_C_w; // plastic yield criterion
        } else {
          eta = usr->par->nd_eta_b; // block
          Y = usr->par->nd_C_b; // plastic yield criterion
        }

        // second invariant of strain rate
        point.i = i; point.j = j; point.loc = ELEMENT; 
        point.c = 0; PetscCall(DMStagVecGetValuesStencil(usr->dmeps,xepslocal,1,&point,&exx)); 
        point.c = 1; PetscCall(DMStagVecGetValuesStencil(usr->dmeps,xepslocal,1,&point,&ezz)); 
        point.c = 2; PetscCall(DMStagVecGetValuesStencil(usr->dmeps,xepslocal,1,&point,&exz)); 
        point.c = 3; PetscCall(DMStagVecGetValuesStencil(usr->dmeps,xepslocal,1,&point,&epsII)); 

        if (usr->par->plasticity) {
          // Effective viscosity
          eta_P = Y/(2.0*epsII);
          eta = usr->par->etamin + 1.0/(1.0/eta_P + 1.0/eta + 1.0/usr->par->etamax); 
        }

        point.i = i; point.j = j; point.loc = ELEMENT; point.c = 1;
        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
        c[j][i][idx] = eta;

        // second invariant of stress
        txx = 2.0*eta*exx;
        tzz = 2.0*eta*ezz;
        txz = 2.0*eta*exz;
        tauII = PetscPowScalar(0.5*(txx*txx + tzz*tzz + 2.0*txz*txz),0.5);

        // save stresses for output
        PetscCall(DMStagGetLocationSlot(usr->dmeps,ELEMENT,0,&idx));  xxs[j][i][idx] = txx; xxy[j][i][idx] = Y;
        PetscCall(DMStagGetLocationSlot(usr->dmeps,ELEMENT,1,&idx));  xxs[j][i][idx] = tzz;
        PetscCall(DMStagGetLocationSlot(usr->dmeps,ELEMENT,2,&idx));  xxs[j][i][idx] = txz;
        PetscCall(DMStagGetLocationSlot(usr->dmeps,ELEMENT,3,&idx));  xxs[j][i][idx] = tauII;
      }

      { // A = eta (corner, c=0)
        DMStagStencil point[4];
        PetscScalar   eta,epsII[4], Y[4], eta_P, xp[4],zp[4];
        PetscScalar   exx[4],ezz[4],exz[4],txx[4],tzz[4],txz[4],tauII[4];
        PetscInt      ii;

        // second invariant of strain rate
        point[0].i = i; point[0].j = j; point[0].loc = DOWN_LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = DOWN_RIGHT; point[1].c = 0;
        point[2].i = i; point[2].j = j; point[2].loc = UP_LEFT;    point[2].c = 0;
        point[3].i = i; point[3].j = j; point[3].loc = UP_RIGHT;   point[3].c = 0;
        PetscCall(DMStagVecGetValuesStencil(usr->dmeps,xepslocal,4,point,exx)); 
        
        for (ii = 0; ii < 4; ii++) {point[ii].c = 1;}
        PetscCall(DMStagVecGetValuesStencil(usr->dmeps,xepslocal,4,point,ezz)); 

        for (ii = 0; ii < 4; ii++) {point[ii].c = 2;}
        PetscCall(DMStagVecGetValuesStencil(usr->dmeps,xepslocal,4,point,exz)); 

        // second invariant of strain rate
        for (ii = 0; ii < 4; ii++) {point[ii].c = 3;}
        PetscCall(DMStagVecGetValuesStencil(usr->dmeps,xepslocal,4,point,epsII)); 

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
            eta = usr->par->nd_eta_w; // inclusion
            Y[ii] = usr->par->nd_C_w; // plastic yield criterion
          } else {
            eta = usr->par->nd_eta_b; //block
            Y[ii] = usr->par->nd_C_b; // plastic yield criterion
          }

          if (usr->par->plasticity) {
            // Effective viscosity
            eta_P = Y[ii]/(2.0*epsII[ii]);
            eta = usr->par->etamin + 1.0/(1.0/eta_P + 1.0/eta + 1.0/usr->par->etamax); 
          }

          // second invariant of stress
          txx[ii] = 2.0*eta*exx[ii];
          tzz[ii] = 2.0*eta*ezz[ii];
          txz[ii] = 2.0*eta*exz[ii];
          tauII[ii] = PetscPowScalar(0.5*(txx[ii]*txx[ii] + tzz[ii]*tzz[ii] + 2.0*txz[ii]*txz[ii]),0.5);

          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, 0, &idx)); 
          c[j][i][idx] = eta;
        }

        // save stresses
        PetscCall(DMStagGetLocationSlot(usr->dmeps,DOWN_LEFT,0,&idx));  xxs[j][i][idx] = txx[0]; xxy[j][i][idx] = Y[0];
        PetscCall(DMStagGetLocationSlot(usr->dmeps,DOWN_LEFT,1,&idx));  xxs[j][i][idx] = tzz[0];
        PetscCall(DMStagGetLocationSlot(usr->dmeps,DOWN_LEFT,2,&idx));  xxs[j][i][idx] = txz[0];
        PetscCall(DMStagGetLocationSlot(usr->dmeps,DOWN_LEFT,3,&idx));  xxs[j][i][idx] = tauII[0];

        PetscCall(DMStagGetLocationSlot(usr->dmeps,DOWN_RIGHT,0,&idx));  xxs[j][i][idx] = txx[1]; xxy[j][i][idx] = Y[1];
        PetscCall(DMStagGetLocationSlot(usr->dmeps,DOWN_RIGHT,1,&idx));  xxs[j][i][idx] = tzz[1];
        PetscCall(DMStagGetLocationSlot(usr->dmeps,DOWN_RIGHT,2,&idx));  xxs[j][i][idx] = txz[1];
        PetscCall(DMStagGetLocationSlot(usr->dmeps,DOWN_RIGHT,3,&idx));  xxs[j][i][idx] = tauII[1];

        PetscCall(DMStagGetLocationSlot(usr->dmeps,UP_LEFT,0,&idx));  xxs[j][i][idx] = txx[2]; xxy[j][i][idx] = Y[2];
        PetscCall(DMStagGetLocationSlot(usr->dmeps,UP_LEFT,1,&idx));  xxs[j][i][idx] = tzz[2];
        PetscCall(DMStagGetLocationSlot(usr->dmeps,UP_LEFT,2,&idx));  xxs[j][i][idx] = txz[2];
        PetscCall(DMStagGetLocationSlot(usr->dmeps,UP_LEFT,3,&idx));  xxs[j][i][idx] = tauII[2];

        PetscCall(DMStagGetLocationSlot(usr->dmeps,UP_RIGHT,0,&idx));  xxs[j][i][idx] = txx[3]; xxy[j][i][idx] = Y[3];
        PetscCall(DMStagGetLocationSlot(usr->dmeps,UP_RIGHT,1,&idx));  xxs[j][i][idx] = tzz[3];
        PetscCall(DMStagGetLocationSlot(usr->dmeps,UP_RIGHT,2,&idx));  xxs[j][i][idx] = txz[3];
        PetscCall(DMStagGetLocationSlot(usr->dmeps,UP_RIGHT,3,&idx));  xxs[j][i][idx] = tauII[3];
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
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] = rhs[ii];
        }
      }

      { // C = 0 (center, c=0)
        DMStagStencil point;
        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 0;
        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
        c[j][i][idx] = 0.0;
      }
    }
  }

  // Restore arrays, local vectors
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL));

  PetscCall(DMStagVecRestoreArray(dmcoeff,coefflocal,&c));
  PetscCall(DMLocalToGlobalBegin(dmcoeff,coefflocal,INSERT_VALUES,coeff)); 
  PetscCall(DMLocalToGlobalEnd  (dmcoeff,coefflocal,INSERT_VALUES,coeff)); 
  PetscCall(VecDestroy(&coefflocal)); 

  // Restore and map local to global
  PetscCall(DMStagVecRestoreArray(usr->dmeps,xtaulocal,&xxs)); 
  PetscCall(DMLocalToGlobalBegin(usr->dmeps,xtaulocal,INSERT_VALUES,usr->xtau)); 
  PetscCall(DMLocalToGlobalEnd  (usr->dmeps,xtaulocal,INSERT_VALUES,usr->xtau)); 
  PetscCall(VecDestroy(&xtaulocal)); 

  PetscCall(DMStagVecRestoreArray(usr->dmeps,xyieldlocal,&xxy)); 
  PetscCall(DMLocalToGlobalBegin(usr->dmeps,xyieldlocal,INSERT_VALUES,usr->xyield)); 
  PetscCall(DMLocalToGlobalEnd  (usr->dmeps,xyieldlocal,INSERT_VALUES,usr->xyield)); 
  PetscCall(VecDestroy(&xyieldlocal)); 

  PetscCall(DMRestoreLocalVector(usr->dmeps,&xepslocal)); 

  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscFunctionBeginUser;
  
  // block and weak inclusion params
  zblock_s = 0.2*usr->par->nd_H;
  zblock_e = 0.8*usr->par->nd_H;
  xis = usr->par->nd_H/2-0.1*usr->par->nd_H/2;
  xie = usr->par->nd_H/2+0.1*usr->par->nd_H/2;
  zis = usr->par->nd_H/2-0.1*usr->par->nd_H/2;
  zie = usr->par->nd_H/2+0.1*usr->par->nd_H/2;
  
  // Strain rates
  PetscCall(UpdateStrainRates(dm,x2,usr)); 
  PetscCall(DMGetLocalVector(usr->dmeps, &xepslocal)); 
  PetscCall(DMGlobalToLocal (usr->dmeps, usr->xeps, INSERT_VALUES, xepslocal)); 
  
  // Local vectors
  PetscCall(DMCreateLocalVector (usr->dmeps,&xtaulocal)); 
  PetscCall(DMStagVecGetArray(usr->dmeps,xtaulocal,&xxs)); 
  
  PetscCall(DMCreateLocalVector (usr->dmeps,&xyieldlocal)); 
  PetscCall(DMStagVecGetArray(usr->dmeps,xyieldlocal,&xxy)); 
  
  // Get domain corners
  PetscCall(DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 
  PetscCall(DMStagGetGlobalSizes(dmcoeff,&Nx,&Nz,NULL));
  
  // Get dm coordinates array
  PetscCall(DMStagGetProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmcoeff,LEFT,&iprev));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmcoeff,RIGHT,&inext));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmcoeff,ELEMENT,&icenter));
  
  // Create coefficient local vector
  PetscCall(DMCreateLocalVector(dmcoeff, &coefflocal)); 
  PetscCall(DMStagVecGetArray(dmcoeff, coefflocal, &c)); 
  
  // Loop over local domain - set initial density and viscosity
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      PetscInt idx;
      
      { // A = eta (center, c=1)
        DMStagStencil point;//, pointP;
        PetscScalar   Y,eta,eta_P,epsII,exx,ezz,exz,txx,tzz,txz,tauII;
        
        if ((coordz[j][icenter]<zblock_s) || (coordz[j][icenter]>zblock_e)) {
          eta = usr->par->nd_eta_w; // top/bottom layer
          Y   = usr->par->nd_C_w; // plastic yield criterion
        } else if ((coordx[i][icenter]>=xis) && (coordx[i][icenter]<=xie) && (coordz[j][icenter]>=zis) && (coordz[j][icenter]<=zie)) {
          eta = usr->par->nd_eta_w; // inclusion
          Y   = usr->par->nd_C_w; // plastic yield criterion
        } else {
          eta = usr->par->nd_eta_b; // block
          Y = usr->par->nd_C_b; // plastic yield criterion
        }
        
        // second invariant of strain rate
        point.i = i; point.j = j; point.loc = ELEMENT;
        point.c = 0; PetscCall(DMStagVecGetValuesStencil(usr->dmeps,xepslocal,1,&point,&exx)); 
        point.c = 1; PetscCall(DMStagVecGetValuesStencil(usr->dmeps,xepslocal,1,&point,&ezz)); 
        point.c = 2; PetscCall(DMStagVecGetValuesStencil(usr->dmeps,xepslocal,1,&point,&exz)); 
        point.c = 3; PetscCall(DMStagVecGetValuesStencil(usr->dmeps,xepslocal,1,&point,&epsII)); 

        if (usr->par->plasticity) {
          // Effective viscosity
          eta_P = Y/(2.0*epsII);
          eta = usr->par->etamin + 1.0/(1.0/eta_P + 1.0/eta + 1.0/usr->par->etamax); 
        }
        
        point.i = i; point.j = j; point.loc = ELEMENT; point.c = 1;
        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
        c[j][i][idx] = eta;

        // second invariant of stress
        txx = 2.0*eta*exx;
        tzz = 2.0*eta*ezz;
        txz = 2.0*eta*exz;
        tauII = PetscPowScalar(0.5*(txx*txx + tzz*tzz + 2.0*txz*txz),0.5);
        
        // save stresses for output
        PetscCall(DMStagGetLocationSlot(usr->dmeps,ELEMENT,0,&idx));  xxs[j][i][idx] = txx; xxy[j][i][idx] = Y;
        PetscCall(DMStagGetLocationSlot(usr->dmeps,ELEMENT,1,&idx));  xxs[j][i][idx] = tzz;
        PetscCall(DMStagGetLocationSlot(usr->dmeps,ELEMENT,2,&idx));  xxs[j][i][idx] = txz;
        PetscCall(DMStagGetLocationSlot(usr->dmeps,ELEMENT,3,&idx));  xxs[j][i][idx] = tauII;
      }
      
      { // A = eta (corner, c=0)
        DMStagStencil point[4];
        PetscScalar   eta,epsII[4], Y[4], eta_P, xp[4],zp[4];
        PetscScalar   exx[4],ezz[4],exz[4],txx[4],tzz[4],txz[4],tauII[4];
        PetscInt      ii;
        
        // second invariant of strain rate
        point[0].i = i; point[0].j = j; point[0].loc = DOWN_LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = DOWN_RIGHT; point[1].c = 0;
        point[2].i = i; point[2].j = j; point[2].loc = UP_LEFT;    point[2].c = 0;
        point[3].i = i; point[3].j = j; point[3].loc = UP_RIGHT;   point[3].c = 0;
        PetscCall(DMStagVecGetValuesStencil(usr->dmeps,xepslocal,4,point,exx)); 
        
        for (ii = 0; ii < 4; ii++) {point[ii].c = 1;}
        PetscCall(DMStagVecGetValuesStencil(usr->dmeps,xepslocal,4,point,ezz)); 
        
        for (ii = 0; ii < 4; ii++) {point[ii].c = 2;}
        PetscCall(DMStagVecGetValuesStencil(usr->dmeps,xepslocal,4,point,exz)); 
        
        // second invariant of strain rate
        for (ii = 0; ii < 4; ii++) {point[ii].c = 3;}
        PetscCall(DMStagVecGetValuesStencil(usr->dmeps,xepslocal,4,point,epsII)); 
        
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
            eta = usr->par->nd_eta_w; // inclusion
            Y[ii] = usr->par->nd_C_w; // plastic yield criterion
          } else {
            eta = usr->par->nd_eta_b; //block
            Y[ii] = usr->par->nd_C_b; // plastic yield criterion
          }
          
          if (usr->par->plasticity) {
            // Effective viscosity
            eta_P = Y[ii]/(2.0*epsII[ii]);
            eta = usr->par->etamin + 1.0/(1.0/eta_P + 1.0/eta + 1.0/usr->par->etamax); 
          }
          
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, 0, &idx)); 
          c[j][i][idx] = eta;

          // second invariant of stress
          txx[ii] = 2.0*eta*exx[ii];
          tzz[ii] = 2.0*eta*ezz[ii];
          txz[ii] = 2.0*eta*exz[ii];
          tauII[ii] = PetscPowScalar(0.5*(txx[ii]*txx[ii] + tzz[ii]*tzz[ii] + 2.0*txz[ii]*txz[ii]),0.5);
        }
        
        // save stresses
        PetscCall(DMStagGetLocationSlot(usr->dmeps,DOWN_LEFT,0,&idx));  xxs[j][i][idx] = txx[0]; xxy[j][i][idx] = Y[0];
        PetscCall(DMStagGetLocationSlot(usr->dmeps,DOWN_LEFT,1,&idx));  xxs[j][i][idx] = tzz[0];
        PetscCall(DMStagGetLocationSlot(usr->dmeps,DOWN_LEFT,2,&idx));  xxs[j][i][idx] = txz[0];
        PetscCall(DMStagGetLocationSlot(usr->dmeps,DOWN_LEFT,3,&idx));  xxs[j][i][idx] = tauII[0];
        
        PetscCall(DMStagGetLocationSlot(usr->dmeps,DOWN_RIGHT,0,&idx));  xxs[j][i][idx] = txx[1]; xxy[j][i][idx] = Y[1];
        PetscCall(DMStagGetLocationSlot(usr->dmeps,DOWN_RIGHT,1,&idx));  xxs[j][i][idx] = tzz[1];
        PetscCall(DMStagGetLocationSlot(usr->dmeps,DOWN_RIGHT,2,&idx));  xxs[j][i][idx] = txz[1];
        PetscCall(DMStagGetLocationSlot(usr->dmeps,DOWN_RIGHT,3,&idx));  xxs[j][i][idx] = tauII[1];
        
        PetscCall(DMStagGetLocationSlot(usr->dmeps,UP_LEFT,0,&idx));  xxs[j][i][idx] = txx[2]; xxy[j][i][idx] = Y[2];
        PetscCall(DMStagGetLocationSlot(usr->dmeps,UP_LEFT,1,&idx));  xxs[j][i][idx] = tzz[2];
        PetscCall(DMStagGetLocationSlot(usr->dmeps,UP_LEFT,2,&idx));  xxs[j][i][idx] = txz[2];
        PetscCall(DMStagGetLocationSlot(usr->dmeps,UP_LEFT,3,&idx));  xxs[j][i][idx] = tauII[2];
        
        PetscCall(DMStagGetLocationSlot(usr->dmeps,UP_RIGHT,0,&idx));  xxs[j][i][idx] = txx[3]; xxy[j][i][idx] = Y[3];
        PetscCall(DMStagGetLocationSlot(usr->dmeps,UP_RIGHT,1,&idx));  xxs[j][i][idx] = tzz[3];
        PetscCall(DMStagGetLocationSlot(usr->dmeps,UP_RIGHT,2,&idx));  xxs[j][i][idx] = txz[3];
        PetscCall(DMStagGetLocationSlot(usr->dmeps,UP_RIGHT,3,&idx));  xxs[j][i][idx] = tauII[3];
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
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] = rhs[ii];
        }
      }
      
      { // C = 0 (center, c=0)
        DMStagStencil point;
        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 0;
        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
        c[j][i][idx] = 0.0;
      }
    }
  }
  
  // Restore arrays, local vectors
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL));
  
  PetscCall(DMStagVecRestoreArray(dmcoeff,coefflocal,&c));
  PetscCall(DMLocalToGlobalBegin(dmcoeff,coefflocal,INSERT_VALUES,coeff)); 
  PetscCall(DMLocalToGlobalEnd  (dmcoeff,coefflocal,INSERT_VALUES,coeff)); 
  PetscCall(VecDestroy(&coefflocal)); 
  
  // Restore and map local to global
  PetscCall(DMStagVecRestoreArray(usr->dmeps,xtaulocal,&xxs)); 
  PetscCall(DMLocalToGlobalBegin(usr->dmeps,xtaulocal,INSERT_VALUES,usr->xtau)); 
  PetscCall(DMLocalToGlobalEnd  (usr->dmeps,xtaulocal,INSERT_VALUES,usr->xtau)); 
  PetscCall(VecDestroy(&xtaulocal)); 
  
  PetscCall(DMStagVecRestoreArray(usr->dmeps,xyieldlocal,&xxy)); 
  PetscCall(DMLocalToGlobalBegin(usr->dmeps,xyieldlocal,INSERT_VALUES,usr->xyield)); 
  PetscCall(DMLocalToGlobalEnd  (usr->dmeps,xyieldlocal,INSERT_VALUES,usr->xyield)); 
  PetscCall(VecDestroy(&xyieldlocal)); 
  
  PetscCall(DMRestoreLocalVector(usr->dmeps,&xepslocal)); 
  
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscFunctionBeginUser;

  dmeps = usr->dmeps;
  xeps  = usr->xeps;

  // Local vectors
  PetscCall(DMCreateLocalVector (dmeps,&xepslocal)); 
  PetscCall(DMStagVecGetArray(dmeps,xepslocal,&xx)); 

  PetscCall(DMGetLocalVector(dm,&xlocal)); 
  PetscCall(DMGlobalToLocal (dm,x,INSERT_VALUES,xlocal)); 

  // Get domain corners
  PetscCall(DMStagGetGlobalSizes(dmeps, &Nx, &Nz,NULL));
  PetscCall(DMStagGetCorners(dmeps, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 
  
  // Get dm coordinates array
  PetscCall(DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter)); 
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm,LEFT,&iprev));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm,RIGHT,&inext)); 

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      DMStagStencil  pointC, pointN[4];
      PetscScalar    epsIIc, exxc, ezzc, exzc, epsIIn[4], exxn[4], ezzn[4], exzn[4];

      // Strain rates: center
      pointC.i = i; pointC.j = j; pointC.loc = ELEMENT; pointC.c = 0;
      PetscCall(DMStagGetPointStrainRates(dm,xlocal,1,&pointC,&epsIIc,&exxc,&ezzc,&exzc)); 

      PetscCall(DMStagGetLocationSlot(dmeps,ELEMENT,0,&idx));  xx[j][i][idx] = exxc;
      PetscCall(DMStagGetLocationSlot(dmeps,ELEMENT,1,&idx));  xx[j][i][idx] = ezzc;
      PetscCall(DMStagGetLocationSlot(dmeps,ELEMENT,2,&idx));  xx[j][i][idx] = exzc;
      PetscCall(DMStagGetLocationSlot(dmeps,ELEMENT,3,&idx));  xx[j][i][idx] = epsIIc;

      // Strain rates: corner
      pointN[0].i = i; pointN[0].j = j; pointN[0].loc = DOWN_LEFT;  pointN[0].c = 0;
      pointN[1].i = i; pointN[1].j = j; pointN[1].loc = DOWN_RIGHT; pointN[1].c = 0;
      pointN[2].i = i; pointN[2].j = j; pointN[2].loc = UP_LEFT;    pointN[2].c = 0;
      pointN[3].i = i; pointN[3].j = j; pointN[3].loc = UP_RIGHT;   pointN[3].c = 0;
      PetscCall(DMStagGetPointStrainRates(dm,xlocal,4,pointN,epsIIn,exxn,ezzn,exzn)); 
      
      if (i==0) { // boundaries
        pointC.i = i; pointC.j = j; pointC.loc = ELEMENT; pointC.c = 0;
        PetscCall(DMStagGetPointStrainRates(dm,xlocal,1,&pointC,&epsIIc,&exxc,&ezzc,&exzc)); 
        ezzn[0] = ezzc;
      }

      if (i==Nx-1) { // boundaries
        pointC.i = i; pointC.j = j; pointC.loc = ELEMENT; pointC.c = 0;
        PetscCall(DMStagGetPointStrainRates(dm,xlocal,1,&pointC,&epsIIc,&exxc,&ezzc,&exzc)); 
        ezzn[1] = ezzc;
      }

      if (j==0) { // boundaries
        pointC.i = i; pointC.j = j; pointC.loc = ELEMENT; pointC.c = 0;
        PetscCall(DMStagGetPointStrainRates(dm,xlocal,1,&pointC,&epsIIc,&exxc,&ezzc,&exzc)); 
        exxn[0] = exxc;
      }

      if (j==Nz-1) { // boundaries
        pointC.i = i; pointC.j = j; pointC.loc = ELEMENT; pointC.c = 0;
        PetscCall(DMStagGetPointStrainRates(dm,xlocal,1,&pointC,&epsIIc,&exxc,&ezzc,&exzc)); 
        exxn[2] = exxc;
      }

      if ((i==Nx-1) && (j==Nz-1)) { // boundaries
        pointC.i = i; pointC.j = j; pointC.loc = ELEMENT; pointC.c = 0;
        PetscCall(DMStagGetPointStrainRates(dm,xlocal,1,&pointC,&epsIIc,&exxc,&ezzc,&exzc)); 
        exxn[3] = exxc;
      }

      if ((i==0) || (i==Nx-1) || (j==0) || (j==Nz-1)) { // boundaries
        for (ii = 0; ii < 4; ii++) {
          epsIIn[ii] = PetscPowScalar(0.5*(exxn[ii]*exxn[ii] + ezzn[ii]*ezzn[ii] + 2.0*exzn[ii]*exzn[ii]),0.5);
        }
      }

      PetscCall(DMStagGetLocationSlot(dmeps,DOWN_LEFT,0,&idx));  xx[j][i][idx] = exxn[0];
      PetscCall(DMStagGetLocationSlot(dmeps,DOWN_LEFT,1,&idx));  xx[j][i][idx] = ezzn[0];
      PetscCall(DMStagGetLocationSlot(dmeps,DOWN_LEFT,2,&idx));  xx[j][i][idx] = exzn[0];
      PetscCall(DMStagGetLocationSlot(dmeps,DOWN_LEFT,3,&idx));  xx[j][i][idx] = epsIIn[0];

      PetscCall(DMStagGetLocationSlot(dmeps,DOWN_RIGHT,0,&idx));  xx[j][i][idx] = exxn[1];
      PetscCall(DMStagGetLocationSlot(dmeps,DOWN_RIGHT,1,&idx));  xx[j][i][idx] = ezzn[1];
      PetscCall(DMStagGetLocationSlot(dmeps,DOWN_RIGHT,2,&idx));  xx[j][i][idx] = exzn[1];
      PetscCall(DMStagGetLocationSlot(dmeps,DOWN_RIGHT,3,&idx));  xx[j][i][idx] = epsIIn[1];

      PetscCall(DMStagGetLocationSlot(dmeps,UP_LEFT,0,&idx));  xx[j][i][idx] = exxn[2];
      PetscCall(DMStagGetLocationSlot(dmeps,UP_LEFT,1,&idx));  xx[j][i][idx] = ezzn[2];
      PetscCall(DMStagGetLocationSlot(dmeps,UP_LEFT,2,&idx));  xx[j][i][idx] = exzn[2];
      PetscCall(DMStagGetLocationSlot(dmeps,UP_LEFT,3,&idx));  xx[j][i][idx] = epsIIn[2];

      PetscCall(DMStagGetLocationSlot(dmeps,UP_RIGHT,0,&idx));  xx[j][i][idx] = exxn[3];
      PetscCall(DMStagGetLocationSlot(dmeps,UP_RIGHT,1,&idx));  xx[j][i][idx] = ezzn[3];
      PetscCall(DMStagGetLocationSlot(dmeps,UP_RIGHT,2,&idx));  xx[j][i][idx] = exzn[3];
      PetscCall(DMStagGetLocationSlot(dmeps,UP_RIGHT,3,&idx));  xx[j][i][idx] = epsIIn[3];
    }
  }

  // Restore arrays
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));

  // Restore and map local to global
  PetscCall(DMStagVecRestoreArray(dmeps,xepslocal,&xx)); 
  PetscCall(DMLocalToGlobalBegin(dmeps,xepslocal,INSERT_VALUES,xeps)); 
  PetscCall(DMLocalToGlobalEnd  (dmeps,xepslocal,INSERT_VALUES,xeps)); 
  PetscCall(VecDestroy(&xepslocal)); 

  PetscCall(DMRestoreLocalVector(dm, &xlocal )); 
  
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscFunctionBeginUser;

  // Create local and global vector associated with DM
  PetscCall(VecDuplicate(x,&xnew));
  PetscCall(DMCreateLocalVector(dm,&xnewlocal)); 
  PetscCall(DMStagVecGetArray(dm,xnewlocal,&xxnew)); 

  PetscCall(DMGetLocalVector(dm,&xlocal)); 
  PetscCall(DMGlobalToLocal (dm,x,INSERT_VALUES,xlocal)); 
  PetscCall(DMStagVecGetArray(dm,xlocal,&xx)); 

  // Get domain corners
  PetscCall(DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 

  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      PetscCall(DMStagGetLocationSlot(dm,ELEMENT,0,&idx));  
      xxnew[j][i][idx] = xx[j][i][idx]*usr->par->stress;

      PetscCall(DMStagGetLocationSlot(dm,LEFT,0,&idx));  
      xxnew[j][i][idx] = xx[j][i][idx]*usr->par->vel;

      PetscCall(DMStagGetLocationSlot(dm,RIGHT,0,&idx));  
      xxnew[j][i][idx] = xx[j][i][idx]*usr->par->vel;

      PetscCall(DMStagGetLocationSlot(dm,DOWN,0,&idx));  
      xxnew[j][i][idx] = xx[j][i][idx]*usr->par->vel;

      PetscCall(DMStagGetLocationSlot(dm,UP,0,&idx));  
      xxnew[j][i][idx] = xx[j][i][idx]*usr->par->vel;
    }
  }

  // Restore arrays
  PetscCall(DMStagVecRestoreArray(dm,xnewlocal,&xxnew)); 
  PetscCall(DMStagVecRestoreArray(dm,xlocal,&xx)); 
  PetscCall(DMRestoreLocalVector(dm, &xlocal )); 

  // Map local to global
  PetscCall(DMLocalToGlobalBegin(dm,xnewlocal,INSERT_VALUES,xnew)); 
  PetscCall(DMLocalToGlobalEnd  (dm,xnewlocal,INSERT_VALUES,xnew)); 
  PetscCall(VecDestroy(&xnewlocal)); 

  // Assign pointers
  *_x  = xnew;
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ScaleCoefficient(DM dm, Vec x, Vec *_x, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  PetscInt       i, j, sx, sz, nx, nz,idx;
  PetscScalar    ***xxnew, ***xx;
  Vec            xnew, xnewlocal, xlocal;
  PetscFunctionBeginUser;

  // Create local and global vector associated with DM
  PetscCall(VecDuplicate(x,&xnew));
  PetscCall(DMCreateLocalVector(dm,&xnewlocal)); 
  PetscCall(DMStagVecGetArray(dm,xnewlocal,&xxnew)); 

  PetscCall(DMGetLocalVector(dm,&xlocal)); 
  PetscCall(DMGlobalToLocal (dm,x,INSERT_VALUES,xlocal)); 
  PetscCall(DMStagVecGetArray(dm,xlocal,&xx)); 

  // Get domain corners
  PetscCall(DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 

  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      PetscCall(DMStagGetLocationSlot(dm,ELEMENT,0,&idx));  // C
      xxnew[j][i][idx] = xx[j][i][idx]*usr->par->vel/usr->par->length;

      PetscCall(DMStagGetLocationSlot(dm,ELEMENT,1,&idx));  // A=eta
      xxnew[j][i][idx] = xx[j][i][idx]*usr->par->visc;

      PetscCall(DMStagGetLocationSlot(dm,LEFT,0,&idx));  // Bx
      xxnew[j][i][idx] = xx[j][i][idx]*usr->par->stress/usr->par->length;

      PetscCall(DMStagGetLocationSlot(dm,RIGHT,0,&idx));  // Bx
      xxnew[j][i][idx] = xx[j][i][idx]*usr->par->stress/usr->par->length;

      PetscCall(DMStagGetLocationSlot(dm,DOWN,0,&idx));  //Bz
      xxnew[j][i][idx] = xx[j][i][idx]*usr->par->stress/usr->par->length;

      PetscCall(DMStagGetLocationSlot(dm,UP,0,&idx));  // Bz
      xxnew[j][i][idx] = xx[j][i][idx]*usr->par->stress/usr->par->length;

      PetscCall(DMStagGetLocationSlot(dm,DOWN_LEFT,0,&idx));  // A corner
      xxnew[j][i][idx] = xx[j][i][idx]*usr->par->visc;

      PetscCall(DMStagGetLocationSlot(dm,DOWN_RIGHT,0,&idx)); 
      xxnew[j][i][idx] = xx[j][i][idx]*usr->par->visc;

      PetscCall(DMStagGetLocationSlot(dm,UP_LEFT,0,&idx)); 
      xxnew[j][i][idx] = xx[j][i][idx]*usr->par->visc;

      PetscCall(DMStagGetLocationSlot(dm,UP_RIGHT,0,&idx)); 
      xxnew[j][i][idx] = xx[j][i][idx]*usr->par->visc;
    }
  }

  // Restore arrays
  PetscCall(DMStagVecRestoreArray(dm,xnewlocal,&xxnew)); 
  PetscCall(DMStagVecRestoreArray(dm,xlocal,&xx)); 
  PetscCall(DMRestoreLocalVector(dm, &xlocal )); 

  // Map local to global
  PetscCall(DMLocalToGlobalBegin(dm,xnewlocal,INSERT_VALUES,xnew)); 
  PetscCall(DMLocalToGlobalEnd  (dm,xnewlocal,INSERT_VALUES,xnew)); 
  PetscCall(VecDestroy(&xnewlocal)); 

  // Assign pointers
  *_x  = xnew;
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ScaleVectorUniform(DM dm, Vec x, Vec *_x, PetscScalar scal)
{
  PetscInt       i, j, sx, sz, nx, nz,idx,dof0,dof1,dof2, ii;
  PetscScalar    ***xxnew, ***xx;
  Vec            xnew, xnewlocal, xlocal;
  PetscFunctionBeginUser;

  // Create local and global vector associated with DM
  PetscCall(VecDuplicate(x,&xnew));
  PetscCall(DMCreateLocalVector(dm,&xnewlocal)); 
  PetscCall(DMStagVecGetArray(dm,xnewlocal,&xxnew)); 

  PetscCall(DMGetLocalVector(dm,&xlocal)); 
  PetscCall(DMGlobalToLocal (dm,x,INSERT_VALUES,xlocal)); 
  PetscCall(DMStagVecGetArray(dm,xlocal,&xx)); 

  // Get domain corners
  PetscCall(DMStagGetDOF(dm,&dof0,&dof1,&dof2,NULL));
  PetscCall(DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 

  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      for (ii = 0; ii <dof2; ii++) {
        PetscCall(DMStagGetLocationSlot(dm,ELEMENT,ii,&idx));  // element
        xxnew[j][i][idx] = xx[j][i][idx]*scal;
      }

      for (ii = 0; ii <dof1; ii++) { // faces
        PetscCall(DMStagGetLocationSlot(dm,LEFT,ii,&idx)); 
        xxnew[j][i][idx] = xx[j][i][idx]*scal;

        PetscCall(DMStagGetLocationSlot(dm,RIGHT,ii,&idx)); 
        xxnew[j][i][idx] = xx[j][i][idx]*scal;

        PetscCall(DMStagGetLocationSlot(dm,DOWN,ii,&idx)); 
        xxnew[j][i][idx] = xx[j][i][idx]*scal;

        PetscCall(DMStagGetLocationSlot(dm,UP,ii,&idx)); 
        xxnew[j][i][idx] = xx[j][i][idx]*scal;
      }

      for (ii = 0; ii <dof0; ii++) { // nodes
        PetscCall(DMStagGetLocationSlot(dm,DOWN_LEFT,ii,&idx)); 
        xxnew[j][i][idx] = xx[j][i][idx]*scal;

        PetscCall(DMStagGetLocationSlot(dm,DOWN_RIGHT,ii,&idx)); 
        xxnew[j][i][idx] = xx[j][i][idx]*scal;

        PetscCall(DMStagGetLocationSlot(dm,UP_LEFT,ii,&idx)); 
        xxnew[j][i][idx] = xx[j][i][idx]*scal;

        PetscCall(DMStagGetLocationSlot(dm,UP_RIGHT,ii,&idx)); 
        xxnew[j][i][idx] = xx[j][i][idx]*scal;
      }
    }
  }

  // Restore arrays
  PetscCall(DMStagVecRestoreArray(dm,xnewlocal,&xxnew)); 
  PetscCall(DMStagVecRestoreArray(dm,xlocal,&xx)); 
  PetscCall(DMRestoreLocalVector(dm, &xlocal )); 

  // Map local to global
  PetscCall(DMLocalToGlobalBegin(dm,xnewlocal,INSERT_VALUES,xnew)); 
  PetscCall(DMLocalToGlobalEnd  (dm,xnewlocal,INSERT_VALUES,xnew)); 
  PetscCall(VecDestroy(&xnewlocal)); 

  // Assign pointers
  *_x  = xnew;
  
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscFunctionBeginUser;

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
  
  // Vx=vi on left boundary (w)
  PetscCall(DMStagBCListGetValues(bclist,'w','-',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = usr->par->nd_vi;
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  
  // Vx=-vi on right boundary (e)
  PetscCall(DMStagBCListGetValues(bclist,'e','-',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = -usr->par->nd_vi;
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  
  // Vz=vi on top boundary (n)
  PetscCall(DMStagBCListGetValues(bclist,'n','|',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = usr->par->nd_vi;
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  
  // Vz=-vi on bottom boundary (s)
  PetscCall(DMStagBCListGetValues(bclist,'s','|',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = -usr->par->nd_vi;
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));

  // pin pressure dof
  PetscCall(DMStagBCListPinCornerValue(bclist,DMSTAG_DOWN_LEFT,'o',0,usr->par->nd_P)); 

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
  PetscCall(PetscBagRegisterInt(bag, &par->nx, 5, "nx", "Element count in the x-dir")); 
  PetscCall(PetscBagRegisterInt(bag, &par->nz, 5, "nz", "Element count in the z-dir")); 

  // Physical and material parameters
  PetscCall(PetscBagRegisterScalar(bag, &par->H, 1.0e5, "H", "Size of domain in both x and z directions [m]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->eta_b, 1.0e23, "eta_b", "Block shear viscosity [Pa.s]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->eta_w, 1.0e19, "eta_w", "Weak zone and inclusion shear viscosity [Pa.s]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->vi, 5.0e-9, "vi", "Extension/compression velocity [m/s]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->C_b, 1e8, "C_b", "Block Cohesion [Pa]")); 
  
  // scales
  par->length = par->H;
  par->visc   = 1e20;
  par->vel    = par->vi; 
  par->stress = par->vel*par->visc/par->length;

  // non-dimensionalize
  par->nd_eta_b = par->eta_b/par->visc;
  par->nd_eta_w = par->eta_w/par->visc;
  par->nd_vi    = par->vi/par->vel;
  par->nd_C_b   = par->C_b/par->stress;
  par->nd_C_w   = 1e40/par->stress;
  par->nd_P     = 0;
  par->nd_H     = par->H/par->length;

  par->etamax = par->nd_eta_b;
  par->etamin = 1.e-6*par->etamax; //par->nd_eta_w; 
  par->plasticity = PETSC_FALSE;

  //  PetscPrintf(PETSC_COMM_WORLD,"etamax = %g, etamin = %g \n", par->etamax, par->etamin);

  // Input/output 
  PetscCall(PetscBagRegisterString(bag,&par->fname_out,FNAME_LENGTH,"out_solution","output_file","Name for output file, set with: -output_file <filename>")); 

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
  PetscPrintf(usr->comm,"# Test_vp_inclusion_gerya: %s \n",&(date[0]));
  PetscPrintf(usr->comm,"# --------------------------------------- #\n");
  PetscPrintf(usr->comm,"# PETSc options: %s \n",opts);
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
  PetscCall(SNESRegister(SNESPICARDLS,SNESCreate_PicardLS));
  
  // Load command line or input file if required
  PetscCall(PetscOptionsSetValue(NULL,"-snes_monitor",NULL));
  PetscCall(PetscOptionsSetValue(NULL,"-p_snes_monitor",NULL));
  PetscCall(PetscOptionsInsert(PETSC_NULLPTR,&argc,&argv,NULL)); 

  // Input user parameters and print
  PetscCall(InputParameters(&usr)); 

  // Print user parameters
  PetscCall(InputPrintData(usr)); 

  // Numerical solution using the FD pde object
  PetscCall(PetscTime(&start_time)); 
  PetscCall(Numerical_solution(usr)); 
  PetscCall(PetscTime(&end_time)); 
  PetscPrintf(PETSC_COMM_WORLD,"# Runtime: %g (sec) \n", end_time - start_time);
  PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");

  // Destroy objects
  PetscCall(PetscBagDestroy(&usr->bag)); 
  PetscCall(PetscFree(usr));             

  // Finalize main
  PetscCall(PetscFinalize());
  return 0;
}
