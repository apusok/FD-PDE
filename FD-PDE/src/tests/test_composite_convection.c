// ---------------------------------------
// Mantle convection benchmark (Blankenbach et al. 1989)
// Steady-state models:
//    1A - eta0=1e23, b=0, c=0, Ra = 1e4
//    1B - eta0=1e22, b=0, c=0, Ra = 1e5
//    1C - eta0=1e21, b=0, c=0, Ra = 1e6
//    2A - eta0=1e23, b=ln(1000), c=0
//    2B - eta0=1e23, b=ln(16384), c=ln(64), L=2500
// Time-dependent models:
//    3A - eta0=1e23, b=0, c=0, L=1500, Ra = 216000
// run: ./tests/test_composite_convection.app -pc_type lu -pc_factor_mat_solver_type umfpack -nx 10 -nz 10
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
  PetscScalar    eta0, rho0, k, cp, g, Ttop, Tbot, alpha;
  PetscScalar    b, c, dt;
  PetscInt       ts_scheme,adv_scheme,tout,tstep;
  char           fname_out[FNAME_LENGTH]; 
  char           fname_in [FNAME_LENGTH];  
} Params;

typedef struct {
  PetscScalar    length, accel, time, vel, visc, density;
  PetscScalar    temp, mass, heat, power, stress;
} ScalParams;

typedef struct {
  PetscScalar    L, H;
  PetscScalar    xmin, zmin, xmax, zmax;
  PetscScalar    eta0, rho0, k, cp, g, Ttop, Tbot, alpha;
  PetscScalar    b, c, dt;
} NondimensionalParams;

// user defined and model-dependent variables
typedef struct {
  Params        *par;
  ScalParams    *scal;
  NondimensionalParams *nd;
  PetscBag       bag;
  MPI_Comm       comm;
  PetscMPIInt    rank;
} UsrData;

// ---------------------------------------
// Function definitions
// ---------------------------------------
PetscErrorCode InputParameters(UsrData**);
PetscErrorCode InputPrintData(UsrData*);
PetscErrorCode ScalingParameters(UsrData*);
PetscErrorCode Numerical_convection(void*);
PetscErrorCode FormCoefficient_Stokes(FDPDE, DM, Vec, DM, Vec, void*);
PetscErrorCode FormBCList_Stokes(DM, Vec, DMStagBCList, void*);
PetscErrorCode FormCoefficient_Temp(FDPDE, DM, Vec, DM, Vec, void*);
PetscErrorCode FormBCList_Temp(DM, Vec, DMStagBCList, void*);
PetscErrorCode ScaleSolutionOutput(DM,Vec,DM,Vec,UsrData*,DM*,Vec*);

// ---------------------------------------
// Some descriptions
// ---------------------------------------
const char coeff_description_stokes[] =
"  << Stokes Coefficients >> \n"
"  eta_n/eta_c = eta_eff\n"
"  fux = 0 \n" 
"  fuz = rho_eff*g \n"
"  fp = 0 (incompressible)\n"
"  where:\n"
"  eta_eff = eta0 * PetscExpScalar( -b*(T-Ttop)/(Tbot-Ttop) + c*z/H ) \n"
"  rho_eff = rho0*(1-alpha*(T-Ttop)) \n";

const char bc_description_stokes[] =
"  << Stokes BCs >> \n"
"  LEFT: Vx = 0, dVz/dx = 0 (free slip) \n"
"  RIGHT: Vx = 0, dVz/dx = 0 (free slip) \n" 
"  DOWN: Vz = 0, dVx/dz = 0 (free slip) \n" 
"  UP: Vz = 0, dVx/dz = 0 (free slip) \n";

const char coeff_description_temp[] =
"  << Temperature Coefficients >> \n"
"  A = rho*cp (element)\n"
"  B = k (edge)\n"
"  C = 0 (element)\n"
"  u = [ux, uz] (edge) - Stokes velocity \n"
"  where:\n"
"  rho_eff = rho0*(1-alpha*(T-Ttop)) \n";

const char bc_description_temp[] =
"  << Temperature BCs >> \n"
"  LEFT: dT/dx = 0\n"
"  RIGHT: dT/dx = 0\n" 
"  DOWN: T = Tbot\n" 
"  UP: T = Ttop \n";


static PetscScalar EffectiveViscosity(PetscScalar T, PetscScalar Ttop, PetscScalar Tbot, PetscScalar eta0, PetscScalar b, PetscScalar c, PetscScalar H, PetscScalar z)
{ return eta0 * PetscExpScalar( -b*(T-Ttop)/(Tbot-Ttop) + c*z/H ); }

static PetscScalar EffectiveDensity(PetscScalar rho0, PetscScalar alpha, PetscScalar T, PetscScalar Ttop)
{ return rho0*(1-alpha*(T-Ttop)); }

// ---------------------------------------
// Application functions
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "Numerical_convection"
PetscErrorCode Numerical_convection(void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  FDPDE          fd[2], fdmono, *pdes, fdtemp, fdstokes;
  DM             dmPV, dmT, dmOut;
  Vec            x, xPV, xT, xOut;
  PetscInt       nx, nz;
  PetscScalar    xmin, zmin, xmax, zmax;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;

  // Nondimensionalize parameters
  ierr = ScalingParameters(usr);CHKERRQ(ierr);
  
  // Element count
  nx = usr->par->nx;
  nz = usr->par->nz;

  // Domain coords
  xmin = usr->nd->xmin;
  zmin = usr->nd->zmin;
  xmax = usr->nd->xmax;
  zmax = usr->nd->zmax;

  // Create the sub FD-pde objects
  // Stokes
  ierr = FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_STOKES,&fdstokes);CHKERRQ(ierr);
  ierr = FDPDESetUp(fdstokes);CHKERRQ(ierr);
  ierr = FDPDESetFunctionBCList(fdstokes,FormBCList_Stokes,bc_description_stokes,NULL); CHKERRQ(ierr);
  ierr = FDPDESetFunctionCoefficient(fdstokes,FormCoefficient_Stokes,coeff_description_stokes,usr); CHKERRQ(ierr);
  ierr = SNESSetFromOptions(fdstokes->snes); CHKERRQ(ierr);

  // Temperature (Advection-diffusion)
  ierr = FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_ADVDIFF,&fdtemp);CHKERRQ(ierr);
  ierr = FDPDESetUp(fdtemp);CHKERRQ(ierr);

  if (usr->par->adv_scheme==0) { ierr = FDPDEAdvDiffSetAdvectSchemeType(fdtemp,ADV_UPWIND);CHKERRQ(ierr); }
  if (usr->par->adv_scheme==1) { ierr = FDPDEAdvDiffSetAdvectSchemeType(fdtemp,ADV_FROMM);CHKERRQ(ierr); }
  
  ierr = FDPDEAdvDiffSetTimeStepSchemeType(fdtemp,TS_NONE);CHKERRQ(ierr); // steady-state
  // if (usr->par->ts_scheme == 0) { ierr = FDPDEAdvDiffSetTimeStepSchemeType(fdtemp,TS_FORWARD_EULER);CHKERRQ(ierr); }
  // if (usr->par->ts_scheme == 1) { ierr = FDPDEAdvDiffSetTimeStepSchemeType(fdtemp,TS_BACKWARD_EULER);CHKERRQ(ierr); }
  // if (usr->par->ts_scheme == 2) { ierr = FDPDEAdvDiffSetTimeStepSchemeType(fdtemp,TS_CRANK_NICHOLSON );CHKERRQ(ierr);}

  ierr = FDPDESetFunctionBCList(fdtemp,FormBCList_Temp,bc_description_temp,NULL); CHKERRQ(ierr);
  ierr = FDPDESetFunctionCoefficient(fdtemp,FormCoefficient_Temp,coeff_description_temp,usr); CHKERRQ(ierr);
  ierr = SNESSetFromOptions(fdtemp->snes); CHKERRQ(ierr);

  // Save fd-pdes in array for composite form
  fd[0] = fdstokes;
  fd[1] = fdtemp;
  
  // Create the composite FD-PDE
  ierr = FDPDECreate2(usr->comm,&fdmono);CHKERRQ(ierr);
  ierr = FDPDESetType(fdmono,FDPDE_COMPOSITE);CHKERRQ(ierr);
  ierr = FDPDCompositeSetFDPDE(fdmono,2,fd);CHKERRQ(ierr);
  ierr = FDPDESetUp(fdmono);CHKERRQ(ierr);
  ierr = FDPDEView(fdmono); CHKERRQ(ierr);
  ierr = SNESSetFromOptions(fdmono->snes); CHKERRQ(ierr);
  
  // Destroy individual FD-PDE objects
  ierr = FDPDEDestroy(&fd[0]);CHKERRQ(ierr);
  ierr = FDPDEDestroy(&fd[1]);CHKERRQ(ierr);

  // FD SNES Solver
  ierr = FDPDESolve(fdmono,NULL);CHKERRQ(ierr);

  // Get solution vector
  ierr = FDPDEGetSolution(fdmono,&x);CHKERRQ(ierr);
  ierr = FDPDECompositeSynchronizeGlobalVectors(fdmono,x);CHKERRQ(ierr);
  
  ierr = FDPDCompositeGetFDPDE(fdmono,NULL,&pdes);CHKERRQ(ierr);
  
  // Prepare solution for output
  ierr = FDPDEGetDM(pdes[0],&dmPV); CHKERRQ(ierr);
  ierr = FDPDEGetSolution(pdes[0],&xPV);CHKERRQ(ierr);
  
  ierr = FDPDEGetDM(pdes[1],&dmT); CHKERRQ(ierr);
  ierr = FDPDEGetSolution(pdes[1],&xT);CHKERRQ(ierr);

  // Scale parameters for output
  ierr = ScaleSolutionOutput(dmPV,xPV,dmT,xT,usr,&dmOut,&xOut);CHKERRQ(ierr);

  // Output solution to file
  ierr = DMStagViewBinaryPython(dmOut,xOut,usr->par->fname_out);CHKERRQ(ierr);

  // Clean up
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&xPV);CHKERRQ(ierr);
  ierr = VecDestroy(&xT);CHKERRQ(ierr);
  ierr = VecDestroy(&xOut);CHKERRQ(ierr);
  
  ierr = DMDestroy(&dmPV); CHKERRQ(ierr);
  ierr = DMDestroy(&dmT); CHKERRQ(ierr);
  ierr = DMDestroy(&dmOut); CHKERRQ(ierr);
  
  ierr = FDPDEDestroy(&fdmono);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

// ---------------------------------------
// FormCoefficient_Stokes
//    Element: fp, eta_c
//    Edges: fux, fuz
//    Corner: eta_n
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormCoefficient_Stokes"
PetscErrorCode FormCoefficient_Stokes(FDPDE fd, DM dm, Vec x, DM dmcoeff, Vec coeff, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  DM             dmT = NULL;
  Vec            *aux_vecs, xT = NULL, xTlocal, xlocal;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz, naux;
  Vec            coefflocal;
  PetscScalar    **coordx,**coordz;
  PetscInt       iprev, inext, icenter;
  PetscScalar    ***cx;
  PetscScalar    g, eta, eta0, rho, rho0, Ttop, Tbot,b,c, H, alpha;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  // User parameters
  g     = -usr->nd->g;
  eta0  = usr->nd->eta0;
  rho0  = usr->nd->rho0;
  alpha = usr->nd->alpha;
  Ttop  = usr->nd->Ttop;
  Tbot  = usr->nd->Tbot;
  b     = usr->nd->b;
  c     = usr->nd->c;
  H     = usr->nd->H;
  
  // Get dm and solution vector for Temperature
  /* You will get NULL in each pdes[] endtry if fd is not of type COMPOSITE */
  /* The fd input here will be of type STOKES */
  /*ierr = FDPDCompositeGetFDPDE(fd,NULL,&pdes);CHKERRQ(ierr);*/
  /*ierr = FDPDEGetDM(pdes[1],&dmT); CHKERRQ(ierr);*/
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
  
  // Loop over local domain - set initial density and viscosity
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

      { // fuz = rho*g - need to interpolate temp to Vz points
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
          rho = EffectiveDensity(rho0,alpha,Tinterp,Ttop);
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          cx[j][i][idx] = rho*g;
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
        eta = EffectiveViscosity(T,Ttop,Tbot,eta0,b,c,H,zp);

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
          eta = EffectiveViscosity(Tinterp[ii],Ttop,Tbot,eta0,b,c,H,zp[ii]);
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
  ierr = DMDestroy(&dmT); CHKERRQ(ierr);

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
// FormCoefficient_Temp
//    Element: A = rho*cp (dof 0), C = heat production/sink (dof 1)
//    Edges: k (dof 0), velocity (dof 1)
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormCoefficient_Temp"
PetscErrorCode FormCoefficient_Temp(FDPDE fd, DM dm, Vec x, DM dmcoeff, Vec coeff, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       i, j, sx, sz, nx, nz, naux;
  DM             dmPV = NULL;
  Vec            coefflocal;
  PetscScalar    rho, rho0, k, cp, alpha, Ttop;
  PetscScalar    ***c;
  Vec            *aux_vecs, xPV = NULL, xPVlocal, xlocal;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  // User parameters
  rho0  = usr->nd->rho0;
  k     = usr->nd->k;
  cp    = usr->nd->cp;
  alpha = usr->nd->alpha;
  Ttop  = usr->nd->Ttop;
  
  // Get dm and solution vector for Stokes velocity
  /* You will get NULL in each pdes[] endtry if fd is not of type COMPOSITE */
  /* The fd input here will be of type ADVDIFF */
  /*ierr = FDPDCompositeGetFDPDE(fd,NULL,&pdes);CHKERRQ(ierr);*/
  /*ierr = FDPDEGetDM(pdes[0],&dmPV); CHKERRQ(ierr);*/
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

  // Create coefficient local vector
  ierr = DMCreateLocalVector(dmcoeff, &coefflocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayDOF(dmcoeff, coefflocal, &c); CHKERRQ(ierr);
  
  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {

      { // A = rho*cp
        PetscScalar   T;
        DMStagStencil point;
        PetscInt      idx;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 0;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        ierr = DMStagVecGetValuesStencil(dm,xlocal,1,&point,&T); CHKERRQ(ierr);
        rho = EffectiveDensity(rho0,alpha,T,Ttop);
        c[j][i][idx] = rho*cp;
      }

      { // C = 0.0 - sources of heat/sink
        DMStagStencil point;
        PetscInt      idx;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 1;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = 0.0;
      }

      { // B = k (edge)
        DMStagStencil point[4];
        PetscInt      ii, idx;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 0;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = 0;
        point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = 0;

        for (ii = 0; ii < 4; ii++) {
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = k;
        }
      }

      { // u = velocity (edge) - Stokes velocity
        PetscScalar   v[4];
        DMStagStencil point[4];
        PetscInt      ii, idx;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 1;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 1;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = 1;
        point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = 1;
        
        ierr = DMStagVecGetValuesStencil(dmPV,xPVlocal,4,point,v); CHKERRQ(ierr);

        for (ii = 0; ii < 4; ii++) {
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
  ierr = DMDestroy(&dmPV); CHKERRQ(ierr);

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
    value_bc[k] = usr->nd->Tbot;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP: T = Ttop (Tij = 2/3*Ttop+1/3*Tij-1)
  ierr = DMStagBCListGetValues(bclist,'n','o',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = usr->nd->Ttop;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
 
  PetscFunctionReturn(0);
}

// ---------------------------------------
// ScaleSolutionOutput
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "ScaleSolutionOutput"
PetscErrorCode ScaleSolutionOutput(DM dmPV, Vec xPV, DM dmT, Vec xT, UsrData *usr, DM *_dmOut, Vec *_xOut)
{
  DM             dmOut;
  Vec            xOut, xOutlocal, xPVlocal, xTlocal;
  PetscScalar    secyear = 3600.0*24.0*365.0;
  PetscInt       i, j, sx, sz, nx, nz;
  PetscScalar    eta0, rho0, Ttop, Tbot,b,c, H, alpha;
  PetscInt       iprev, inext, icenter;
  PetscScalar    **coordx,**coordz;
  ScalParams     *scal;
  PetscScalar    ***xxout;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // User parameters
  eta0  = usr->nd->eta0;
  rho0  = usr->nd->rho0;
  alpha = usr->nd->alpha;
  Ttop  = usr->nd->Ttop;
  Tbot  = usr->nd->Tbot;
  b     = usr->nd->b;
  c     = usr->nd->c;
  H     = usr->nd->H;

  scal = usr->scal;
  
  // Create new dmstag and vector for output with scaled coordinates
  // dmOut - v (edge), P, T, eta, rho (center)
  ierr = DMStagCreateCompatibleDMStag(dmPV,0,1,4,0,&dmOut); CHKERRQ(ierr);
  ierr = DMSetUp(dmOut); CHKERRQ(ierr);
  ierr = DMStagSetUniformCoordinatesProduct(dmOut,usr->par->xmin,usr->par->xmax,usr->par->zmin,usr->par->zmax,0.0,0.0);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dmOut,&xOut);CHKERRQ(ierr);
  
  // Map all vectors to local domain - xPV, xT, xOut
  ierr = DMGetLocalVector(dmPV,&xPVlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dmPV,xPV,INSERT_VALUES,xPVlocal); CHKERRQ(ierr);

  ierr = DMGetLocalVector(dmT,&xTlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dmT,xT,INSERT_VALUES,xTlocal); CHKERRQ(ierr);

  ierr = DMCreateLocalVector(dmOut, &xOutlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayDOF(dmOut, xOutlocal, &xxout); CHKERRQ(ierr);

  // Get dmOut coordinates array
  ierr = DMStagGet1dCoordinateArraysDOFRead(dmOut,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGet1dCoordinateLocationSlot(dmOut,LEFT,&iprev);CHKERRQ(ierr);
  ierr = DMStagGet1dCoordinateLocationSlot(dmOut,RIGHT,&inext);CHKERRQ(ierr);
  ierr = DMStagGet1dCoordinateLocationSlot(dmOut,ELEMENT,&icenter);CHKERRQ(ierr);

  // Get domain corners
  ierr = DMStagGetCorners(dmOut, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  // Scale parameters up + calc eta, rho
  // dmOut - v (edge), P, T, eta, rho (center)

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {

      { // Velocity
        PetscScalar   xx[4];
        DMStagStencil point[4];
        PetscInt      idx, ii;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 0;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = 0;
        point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = 0;

        ierr = DMStagVecGetValuesStencil(dmPV,xPVlocal,4,point,xx); CHKERRQ(ierr);

        for (ii = 0; ii < 4; ii++) {
          ierr = DMStagGetLocationSlot(dmOut, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          xxout[j][i][idx] = xx[ii]*scal->vel*1.0e2/secyear; // [cm/yr]
        }
      }

      { // Pressure
        PetscScalar   xx;
        DMStagStencil point;
        PetscInt      idx;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 0;
        ierr = DMStagVecGetValuesStencil(dmPV,xPVlocal,1,&point,&xx); CHKERRQ(ierr);

        ierr = DMStagGetLocationSlot(dmOut, point.loc, 0, &idx); CHKERRQ(ierr);
        xxout[j][i][idx] = xx*scal->stress/1.0e6; // [MPa]
      }

      { // Temperature
        PetscScalar   xx;
        DMStagStencil point;
        PetscInt      idx;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 0;
        ierr = DMStagVecGetValuesStencil(dmT,xTlocal,1,&point,&xx); CHKERRQ(ierr);

        ierr = DMStagGetLocationSlot(dmOut, point.loc, 1, &idx); CHKERRQ(ierr);
        xxout[j][i][idx] = xx*scal->temp; // [K]
      }

      { // Viscosity
        PetscScalar   xx, eta, zp;
        DMStagStencil point;
        PetscInt      idx;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 0;
        ierr = DMStagVecGetValuesStencil(dmT,xTlocal,1,&point,&xx); CHKERRQ(ierr);
        zp = coordz[j][icenter];
        
        eta = EffectiveViscosity(xx,Ttop,Tbot,eta0,b,c,H,zp);
        ierr = DMStagGetLocationSlot(dmOut, point.loc, 2, &idx); CHKERRQ(ierr);
        xxout[j][i][idx] = eta*scal->visc; // [Pa.s]
      }

      { // Density
        PetscScalar   xx, rho;
        DMStagStencil point;
        PetscInt      idx;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 0;
        ierr = DMStagVecGetValuesStencil(dmT,xTlocal,1,&point,&xx); CHKERRQ(ierr);

        rho = EffectiveDensity(rho0,alpha,xx,Ttop);
        ierr = DMStagGetLocationSlot(dmOut, point.loc, 3, &idx); CHKERRQ(ierr);
        xxout[j][i][idx] = rho*scal->density; // [kg/m3]
      }

    }
  }

  // Restore access
  ierr = DMRestoreLocalVector(dmPV,&xPVlocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmT,&xTlocal); CHKERRQ(ierr);

  ierr = DMStagRestore1dCoordinateArraysDOFRead(dmOut,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArrayDOF(dmOut,xOutlocal,&xxout);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dmOut,xOutlocal,INSERT_VALUES,xOut); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dmOut,xOutlocal,INSERT_VALUES,xOut); CHKERRQ(ierr);
  ierr = VecDestroy(&xOutlocal); CHKERRQ(ierr);
  
  // return pointers
  *_dmOut = dmOut;
  *_xOut  = xOut;

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

  ierr = PetscBagRegisterScalar(bag, &par->xmin, 0.0, "xmin", "Start coordinate of domain in x-dir [km]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->zmin, 0.0, "zmin", "Start coordinate of domain in z-dir [km]"); CHKERRQ(ierr);

  ierr = PetscBagRegisterScalar(bag, &par->L, 1000.0, "L", "Length of domain in x-dir [km]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->H, 1000.0, "H", "Height of domain in z-dir [km]"); CHKERRQ(ierr);

  // Time stepping and advection
  ierr = PetscBagRegisterInt(bag, &par->tstep, 1, "tstep", "Number of time steps"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->tout,1,"tout", "Output every <tout> time steps"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->ts_scheme,0, "ts_scheme", "Time stepping scheme"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->adv_scheme,0, "adv_scheme", "Advection scheme 0-upwind, 1-fromm"); CHKERRQ(ierr);

  ierr = PetscBagRegisterScalar(bag, &par->dt, 1.0e3, "dt", "Time step size [kyr]"); CHKERRQ(ierr);

  // Physical and material parameters
  ierr = PetscBagRegisterScalar(bag, &par->g, 10.0, "g", "Gravitational acceleration [m/s2]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->eta0, 1.0e23, "eta0", "Reference viscosity [Pa.s]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->rho0, 4000.0, "rho0", "Reference density [kg/m3]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->k, 5.0, "k", "Thermal conductivity [W/m/K]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->cp, 1250.0, "cp", "Heat capacity [J/kg]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->alpha, 2.5e-5, "alpha", "Coefficient of thermal expansion [1/K]"); CHKERRQ(ierr);

  ierr = PetscBagRegisterScalar(bag, &par->Ttop, 273.0, "Ttop", "Temperature top [K]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->Tbot, 1273.0, "Tbot", "Temperature bottom [K]"); CHKERRQ(ierr);
  
  ierr = PetscBagRegisterScalar(bag, &par->b, 0.0, "b", "Effective viscosity parameter b (T-dep) [-]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->c, 0.0, "c", "Effective viscosity parameter c (depth-dep) [-]"); CHKERRQ(ierr);

  // Input/output 
  ierr = PetscBagRegisterString(bag,&par->fname_out,FNAME_LENGTH,"out_convection","output_file","Name for output file, set with: -output_file <filename>"); CHKERRQ(ierr);

  // Other variables
  par->fname_in[0] = '\0';

  par->xmax = par->xmin+par->L;
  par->zmax = par->zmin+par->H;

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
  PetscPrintf(usr->comm,"# Test_convection: %s \n",&(date[0]));
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
// ScalingParameters
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "ScalingParameters"
PetscErrorCode ScalingParameters(UsrData *usr)
{
  PetscScalar           secyear = 3600.0*24.0*365.0;
  ScalParams           *scal;
  NondimensionalParams *nd;
  Params               *par;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Allocate memory
  ierr = PetscMalloc1(1, &scal); CHKERRQ(ierr);
  ierr = PetscMalloc1(1, &nd); CHKERRQ(ierr);

  par = usr->par;

  // Characteristic lengthscales
  scal->length = par->H*1.0e3; // [m]
  scal->accel  = par->g;       // [m/s2]
  scal->time   = PetscSqrtScalar(scal->length/scal->accel); // [s]
  scal->vel    = scal->length/scal->time; // [m/s]
  scal->visc   = par->eta0; // [Pa.s]
  scal->density= par->rho0; // [kg/m3]
  scal->temp   = par->Tbot; // [K]
  scal->mass   = scal->density*scal->length*scal->length*scal->length; // [kg]
  scal->heat   = scal->mass*scal->accel*scal->length; // [J]
  scal->power  = scal->heat/scal->time; // [W]
  scal->stress = scal->visc/scal->time; // [Pa]
  
  // Scaled parameters
  nd->L = par->L*1.0e3/scal->length;
  nd->H = par->H*1.0e3/scal->length;
  nd->xmin = par->xmin*1.0e3/scal->length;
  nd->xmax = par->xmax*1.0e3/scal->length;
  nd->zmin = par->zmin*1.0e3/scal->length;
  nd->zmax = par->zmax*1.0e3/scal->length;

  nd->dt = 1.0e3*par->dt*secyear/scal->time; // [s]
  nd->g  = par->g/scal->accel;
  nd->eta0 = par->eta0/scal->visc;
  nd->rho0 = par->rho0/scal->density;

  nd->alpha = par->alpha*scal->temp;
  nd->cp    = par->cp/(scal->heat/scal->mass);
  nd->k     = par->k/(scal->power/scal->length/scal->temp);

  nd->Ttop = par->Ttop/scal->temp;
  nd->Tbot = par->Tbot/scal->temp;

  nd->b = par->b;
  nd->c = par->c;

  // return pointers
  usr->scal = scal;
  usr->nd   = nd;

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
  ierr = PetscFree(usr->scal);       CHKERRQ(ierr);
  ierr = PetscFree(usr->nd);         CHKERRQ(ierr);
  ierr = PetscFree(usr);             CHKERRQ(ierr);

  // End time
  ierr = PetscTime(&end_time); CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"# Total runtime: %g (sec) \n", end_time - start_time);
  PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");
  
  // Finalize main
  ierr = PetscFinalize();
  return ierr;
}

