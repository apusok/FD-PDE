// ---------------------------------------
// Shortening a two-phase block (Stokes-Darcy flow with PIC method)
// Rheology: visco-elasto-(visco)plastic model
// run: ./tests/test_stokesdarcy2field_vep_inclusion_pic.app -nx 100 -nz 100 -pc_type lu -pc_factor_mat_solver_type umfpack
// python test: ./tests/python/test_stokesdarcy2field_vep_inclusion_pic.py
// ---------------------------------------
static char help[] = "Application for shortening of a visco-elasto-(visco)plastic two-phase block in the absence of gravity with particles\n\n";

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
#include "../src/fdpde_stokesdarcy2field.h"
#include "../src/consteq.h"
#include "../src/dmstagoutput.h"
#include "../src/material_point.h"

// ---------------------------------------
// Application Context
// ---------------------------------------
#define FNAME_LENGTH 200
#define MAX_MAT_PHASE  2

#define PV_ELEMENT_P   0
#define PV_FACE_VS     0

#define PVCOEFF_VERTEX_A    0 
#define PVCOEFF_ELEMENT_C   0
#define PVCOEFF_ELEMENT_A   1 
#define PVCOEFF_ELEMENT_D1  2 
#define PVCOEFF_FACE_B      0 
#define PVCOEFF_FACE_D2     1 
#define PVCOEFF_FACE_D3     2

// parameters (bag)
typedef struct {
  PetscInt       nx, nz;
  PetscScalar    L, H;
  PetscScalar    xmin, zmin, xmax, zmax;
  PetscScalar    vi, P0, etamin;
  PetscScalar    eb_v0, ew_v0, lam_v, C_b, C_w, zb_v0, zw_v0;
  PetscScalar    lambda,lam_p, G, Z0, q, F, R, phi_0, n, nh;
  PetscBool      plasticity;
  PetscInt       tstep, tout, ppcell, rheology;
  PetscScalar    t, dt, tmax, dtmax, tf_tol;
  char           fname_out[FNAME_LENGTH];
  char           fname_in[FNAME_LENGTH];
  char           fdir_out[FNAME_LENGTH];
} Params;

typedef struct { 
  PetscScalar   eta0, zeta0, C, G, Z0;
} MaterialProp; 

// user defined and model-dependent variables
typedef struct {
  Params        *par;
  PetscBag       bag;
  MPI_Comm       comm;
  PetscMPIInt    rank;
  MaterialProp   mat[MAX_MAT_PHASE];
  DM             dmPV, dmeps, dmP, dmMPhase, dmswarm;
  Vec            xeps, xtau, xplast, xDP, xtau_old, xDP_old, xMPhase;
  PetscInt       nph;
} UsrData;

// ---------------------------------------
// Function definitions
// ---------------------------------------
PetscErrorCode StokesDarcy_Numerical_PIC(void*);
PetscErrorCode InputParameters(UsrData**);
PetscErrorCode InputPrintData(UsrData*);
PetscErrorCode FormCoefficient(FDPDE, DM, Vec, DM, Vec, void*);
PetscErrorCode FormCoefficient_Stokes(FDPDE, DM, Vec, DM, Vec, void*);
PetscErrorCode FormBCList(DM, Vec, DMStagBCList, void*);
PetscErrorCode UpdateStrainRates(DM,Vec,void*);
PetscErrorCode SetSwarmInitialCondition(DM,void*);
PetscErrorCode UpdateMarkerPhaseFractions(DM,DM,Vec,void*);
PetscErrorCode RheologyPointwise(PetscInt,PetscInt,PetscScalar***,PetscInt*,PetscScalar,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,void*);
PetscErrorCode RheologyPointwise_DominantPhase(PetscInt,PetscInt,PetscScalar***,PetscInt*,PetscScalar,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,void*);
PetscErrorCode RheologyPointwise_AveragePhase(PetscInt,PetscInt,PetscScalar***,PetscInt*,PetscScalar,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,void*);
PetscErrorCode DecompactRheologyVars(PetscInt,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,
                                     PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*);

PetscErrorCode GetCornerAvgFromCenter(PetscScalar*,PetscScalar*);
PetscErrorCode Get9PointCenterValues(PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscScalar***,PetscScalar*);
PetscErrorCode GetTensorPointValues(PetscInt,PetscInt,PetscInt*,PetscScalar***,PetscScalar*);

static PetscScalar WeightAverageValue(PetscScalar *a, PetscScalar *wt, PetscInt n) {
  PetscInt    i;
  PetscScalar awt = 0.0;
  for (i = 0; i <n; i++) { awt += a[i]*wt[i]; }
  return awt;
}

const char coeff_description[] = 
"  << Stokes-Darcy Coefficients >> \n"
"  A = eta_vep \n"
"  B = phi*F*ez - div(chi_s * tau_old) + grad(chi_p * diff_pold) \n"
"      (F = ((rho^s-rho^f)*U*L/eta_ref)*(g*L/U^2)) (0, if g=0) \n"
"      (tau_old: stress rotation tensor; diff_pold: pressure difference  ) \n"
"  C = 0 \n"
"  D1 = zeta_vep - 2/3*eta_vep \n"
"  D2 = -Kphi*R^2 (R: ratio of the compaction length to the global length scale) \n"
"  D3 = Kphi*R^2*F*ez \n";

const char coeff_description_Stokes[] = 
"  << Stokes Coefficients >> \n"
"  A = eta_vep \n"
"  B = F*ez - div(chi_s * tau_old) \n"
"  C = 0 \n";

const char bc_description[] =
"  << Stokes-Darcy BCs >> \n"
"  LEFT/BOTTOM: Symmetric boundary conditions (zero normal velocity, free slip) \n"
"  UP: extension Vz = Vi, free slip dVx/dz=0 \n"
"  RIGHT: compression Vx=Vi, free slip dVz/dx = 0\n";

// ---------------------------------------
// Application functions
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "StokesDarcy_Numerical_PIC"
PetscErrorCode StokesDarcy_Numerical_PIC(void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  FDPDE          fd;
  DM             dm, dmcoeff, dmswarm;
  Vec            x, xcoeff, xguess;
  PetscInt       nx, nz, istep = 0;
  PetscScalar    xmin, zmin, xmax, zmax;
  char           fout[FNAME_LENGTH];
  PetscErrorCode ierr;
  PetscFunctionBegin;

  // Element count
  nx = usr->par->nx;
  nz = usr->par->nz;

  // Domain coords
  xmin = usr->par->xmin;
  xmax = usr->par->xmin + usr->par->L;
  zmin = usr->par->zmin;
  zmax = usr->par->zmin + usr->par->H;

  // Create the FD-pde object
  ierr = FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_STOKESDARCY2FIELD,&fd);CHKERRQ(ierr);
  ierr = FDPDESetUp(fd);CHKERRQ(ierr);
  ierr = FDPDESetFunctionBCList(fd,FormBCList,bc_description,usr); CHKERRQ(ierr);
  ierr = FDPDESetFunctionCoefficient(fd,FormCoefficient,coeff_description,usr); CHKERRQ(ierr);

  // ierr = FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_STOKES,&fd);CHKERRQ(ierr);
  // ierr = FDPDESetUp(fd);CHKERRQ(ierr);
  // ierr = FDPDESetFunctionBCList(fd,FormBCList,bc_description,usr); CHKERRQ(ierr);
  // ierr = FDPDESetFunctionCoefficient(fd,FormCoefficient_Stokes,coeff_description_Stokes,usr); CHKERRQ(ierr);

  ierr = FDPDEGetDM(fd,&dm); CHKERRQ(ierr);
  usr->dmPV = dm;

  // Create dmeps/vec for strain rates - center and corner
  ierr = DMStagCreateCompatibleDMStag(dm,4,0,4,0,&usr->dmeps); CHKERRQ(ierr);
  ierr = DMSetUp(usr->dmeps); CHKERRQ(ierr);
  ierr = DMStagSetUniformCoordinatesProduct(usr->dmeps,xmin,xmax,zmin,zmax,0.0,0.0);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(usr->dmeps,&usr->xeps); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(usr->dmeps,&usr->xtau); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(usr->dmeps,&usr->xtau_old); CHKERRQ(ierr);
  
  // Create dmP for pressure
  ierr = DMStagCreateCompatibleDMStag(dm,0,0,1,0,&usr->dmP); CHKERRQ(ierr);
  ierr = DMSetUp(usr->dmP); CHKERRQ(ierr);
  ierr = DMStagSetUniformCoordinatesProduct(usr->dmP,xmin,xmax,zmin,zmax,0.0,0.0);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(usr->dmP,&usr->xDP); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(usr->dmP,&usr->xDP_old); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(usr->dmP,&usr->xplast); CHKERRQ(ierr); // xyield

  ierr = VecSet(usr->xtau_old,0.0); CHKERRQ(ierr);
  ierr = VecSet(usr->xDP_old,0.0); CHKERRQ(ierr);

  // Create dmMPhase for marker phase fractions (lithology)
  PetscInt nm = usr->nph;
  ierr = DMStagCreateCompatibleDMStag(dm,nm,nm,nm,0,&usr->dmMPhase); CHKERRQ(ierr);
  ierr = DMSetUp(usr->dmMPhase); CHKERRQ(ierr);
  ierr = DMStagSetUniformCoordinatesProduct(usr->dmMPhase,xmin,xmax,zmin,zmax,0.0,0.0);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(usr->dmMPhase,&usr->xMPhase);CHKERRQ(ierr);

  // Set up a swarm object and assign several fields
  ierr = DMStagPICCreateDMSwarm(dm,&dmswarm);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(dmswarm,"id",1,PETSC_REAL);CHKERRQ(ierr); // main field (user defined)
  ierr = DMSwarmRegisterPetscDatatypeField(dmswarm,"id0",1,PETSC_REAL);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(dmswarm,"id1",1,PETSC_REAL);CHKERRQ(ierr);
  ierr = DMStagPICFinalize(dmswarm);CHKERRQ(ierr);
  usr->dmswarm = dmswarm;
  PetscInt ppcell[] = {usr->par->ppcell,usr->par->ppcell};
  ierr = MPointCoordLayout_DomainVolumeWithCellList(dmswarm,0,NULL,0.5,ppcell,COOR_INITIALIZE);CHKERRQ(ierr);

  // Initial condition for markers
  ierr = SetSwarmInitialCondition(usr->dmswarm,usr);CHKERRQ(ierr);
  ierr = UpdateMarkerPhaseFractions(usr->dmswarm,usr->dmMPhase,usr->xMPhase,usr);CHKERRQ(ierr);

  const char *fieldname[] = {"id"};
  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/%s_pic_initial.xmf",usr->par->fdir_out,usr->par->fname_out);
  ierr = DMSwarmViewFieldsXDMF(usr->dmswarm,fout,1,fieldname); CHKERRQ(ierr);

  // Create initial guess with a linear viscous
  usr->par->plasticity = PETSC_FALSE; 
  PetscPrintf(PETSC_COMM_WORLD,"\n# INITIAL GUESS #\n");
  ierr = FDPDESolve(fd,NULL);CHKERRQ(ierr);
  ierr = FDPDEGetSolution(fd,&x);CHKERRQ(ierr); 

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/%s_solution_initial",usr->par->fdir_out,usr->par->fname_out);
  ierr = DMStagViewBinaryPython(dm,x,fout);CHKERRQ(ierr);

  ierr = FDPDEGetSolutionGuess(fd,&xguess); CHKERRQ(ierr); 
  ierr = VecCopy(x,xguess);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&xguess);CHKERRQ(ierr);
  usr->par->plasticity = PETSC_TRUE; 
  
  // FD SNES Solver
  PetscPrintf(PETSC_COMM_WORLD,"\n# SNES SOLVE #\n");
  
  // Time loop  
  while ((usr->par->t < usr->par->tmax) && (istep<usr->par->tstep)) {
    PetscPrintf(PETSC_COMM_WORLD,"# TIMESTEP %d: \n",istep);

    // StokesDarcy Solver
    ierr = FDPDESolve(fd,NULL);CHKERRQ(ierr);
    ierr = FDPDEGetSolution(fd,&x);CHKERRQ(ierr);
    
    // Update time
    usr->par->t += usr->par->dt;  // computation start from t = dt
    PetscPrintf(PETSC_COMM_WORLD,"# TIME: time = %1.12e dt = %1.12e \n",usr->par->t,usr->par->dt);

    // copy xtau, xDP to old
    ierr = VecCopy(usr->xtau, usr->xtau_old); CHKERRQ(ierr);
    ierr = VecCopy(usr->xDP, usr->xDP_old); CHKERRQ(ierr);
       
    // Output solution
    if (istep % usr->par->tout == 0 ) {
     
      // Output solution to file
      ierr = PetscSNPrintf(fout,sizeof(fout),"%s/%s_solution_ts%1.3d",usr->par->fdir_out,usr->par->fname_out,istep);
      ierr = DMStagViewBinaryPython(dm,x,fout);CHKERRQ(ierr);

      ierr = PetscSNPrintf(fout,sizeof(fout),"%s/%s_strain_ts%1.3d",usr->par->fdir_out,usr->par->fname_out,istep);
      ierr = DMStagViewBinaryPython(usr->dmeps,usr->xeps,fout);CHKERRQ(ierr);

      ierr = PetscSNPrintf(fout,sizeof(fout),"%s/%s_stress_ts%1.3d",usr->par->fdir_out,usr->par->fname_out,istep);
      ierr = DMStagViewBinaryPython(usr->dmeps,usr->xtau,fout);CHKERRQ(ierr);

      ierr = PetscSNPrintf(fout,sizeof(fout),"%s/%s_stressold_ts%1.3d",usr->par->fdir_out,usr->par->fname_out,istep);
      ierr = DMStagViewBinaryPython(usr->dmeps,usr->xtau_old,fout);CHKERRQ(ierr);

      ierr = PetscSNPrintf(fout,sizeof(fout),"%s/%s_dpold_ts%1.3d",usr->par->fdir_out,usr->par->fname_out,istep);
      ierr = DMStagViewBinaryPython(usr->dmP,usr->xDP_old,fout);CHKERRQ(ierr);

      ierr = PetscSNPrintf(fout,sizeof(fout),"%s/%s_yield_ts%1.3d",usr->par->fdir_out,usr->par->fname_out,istep);
      ierr = DMStagViewBinaryPython(usr->dmP,usr->xplast,fout);CHKERRQ(ierr);

      ierr = PetscSNPrintf(fout,sizeof(fout),"%s/%s_residual_ts%1.3d",usr->par->fdir_out,usr->par->fname_out,istep);
      ierr = DMStagViewBinaryPython(dm,fd->r,fout);CHKERRQ(ierr);

      ierr = FDPDEGetCoefficient(fd,&dmcoeff,&xcoeff);CHKERRQ(ierr);
      ierr = PetscSNPrintf(fout,sizeof(fout),"%s/%s_coefficient_ts%1.3d",usr->par->fdir_out,usr->par->fname_out,istep);
      ierr = DMStagViewBinaryPython(dmcoeff,xcoeff,fout);CHKERRQ(ierr);

      ierr = PetscSNPrintf(fout,sizeof(fout),"%s/%s_pic_ts%1.3d.xmf",usr->par->fdir_out,usr->par->fname_out,istep);
      ierr = DMSwarmViewFieldsXDMF(usr->dmswarm,fout,1,fieldname); CHKERRQ(ierr);
    }

    //clean up
    ierr = VecDestroy(&x);CHKERRQ(ierr);
    istep++;
  }

  // Destroy objects
  // ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xeps);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xtau);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xDP);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xplast);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xtau_old);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xDP_old);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xMPhase);CHKERRQ(ierr);
  ierr = DMDestroy(&usr->dmeps);CHKERRQ(ierr);
  ierr = DMDestroy(&usr->dmP);CHKERRQ(ierr);
  ierr = DMDestroy(&usr->dmMPhase);CHKERRQ(ierr);
  ierr = DMDestroy(&usr->dmswarm);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
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
  Vec            coefflocal, xlocal, xepslocal, xtaulocal, xtauoldlocal, xDPlocal, xDPoldlocal, xplastlocal, xMPhaselocal;
  PetscScalar    **coordx,**coordz;
  PetscInt       iprev, inext, icenter;
  PetscScalar    ***c, ***xx, ***_eps, ***_tauold, ***_tau, ***_DPold, ***_DP, ***_plast, ***xwt;
  PetscScalar    phi, Kphi, F, R; //dt;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  // dt = usr->par->dt;
  R = usr->par->R;
  F = usr->par->F;

  // Uniform porosity and permeability
  phi = usr->par->phi_0;
  Kphi = 1.0;

  // Get coefficient
  ierr = DMStagGetGlobalSizes(dmcoeff, &Nx, &Nz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmcoeff,LEFT,&iprev);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmcoeff,RIGHT,&inext);CHKERRQ(ierr); 
  ierr = DMStagGetProductCoordinateLocationSlot(dmcoeff,ELEMENT,&icenter);CHKERRQ(ierr);

  // Create coefficient local vector
  ierr = DMCreateLocalVector(dmcoeff, &coefflocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dmcoeff, coefflocal, &c); CHKERRQ(ierr);

  // Strain rates
  ierr = UpdateStrainRates(dm,x,usr); CHKERRQ(ierr);
  ierr = DMGetLocalVector(usr->dmeps, &xepslocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmeps, usr->xeps, INSERT_VALUES, xepslocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(usr->dmeps,xepslocal,&_eps);CHKERRQ(ierr);

  ierr = DMGetLocalVector(usr->dmeps, &xtauoldlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmeps, usr->xtau_old, INSERT_VALUES, xtauoldlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(usr->dmeps,xtauoldlocal,&_tauold);CHKERRQ(ierr);

  ierr = DMGetLocalVector(usr->dmeps, &xtaulocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmeps, usr->xtau, INSERT_VALUES, xtaulocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(usr->dmeps,xtaulocal,&_tau);CHKERRQ(ierr);

  // Get dm and solution vector
  ierr = DMGetLocalVector(dm, &xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm, x, INSERT_VALUES, xlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(dm,xlocal,&xx);CHKERRQ(ierr);

  ierr = DMGetLocalVector(usr->dmP, &xDPoldlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmP, usr->xDP_old, INSERT_VALUES, xDPoldlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(usr->dmP,xDPoldlocal,&_DPold);CHKERRQ(ierr);

  ierr = DMGetLocalVector(usr->dmP, &xDPlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmP, usr->xDP, INSERT_VALUES, xDPlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(usr->dmP,xDPlocal,&_DP);CHKERRQ(ierr);

  ierr = DMGetLocalVector(usr->dmP, &xplastlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmP, usr->xplast, INSERT_VALUES, xplastlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(usr->dmP,xplastlocal,&_plast);CHKERRQ(ierr);

  // get material phase fractions
  ierr = DMCreateLocalVector(usr->dmMPhase, &xMPhaselocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(usr->dmMPhase,usr->xMPhase,INSERT_VALUES,xMPhaselocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (usr->dmMPhase,usr->xMPhase,INSERT_VALUES,xMPhaselocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(usr->dmMPhase, xMPhaselocal, &xwt); CHKERRQ(ierr);

  // get location slots
  PetscInt  iP;
  ierr = DMStagGetLocationSlot(usr->dmP,DMSTAG_ELEMENT,0,&iP);CHKERRQ(ierr);

  PetscInt iPV;
  ierr = DMStagGetLocationSlot(dm,DMSTAG_ELEMENT,PV_ELEMENT_P,&iPV);CHKERRQ(ierr);

  PetscInt  e_slot[3],av_slot[4],b_slot[4],d2_slot[4],d3_slot[4],iL,iR,iU,iD,iC,iA,iD1;
  iL = 0; iR = 1; iD = 2; iU  = 3;
  iC = 0; iA = 1; iD1= 2;
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_ELEMENT,   PVCOEFF_ELEMENT_C,   &e_slot[iC]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_ELEMENT,   PVCOEFF_ELEMENT_A,   &e_slot[iA]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_ELEMENT,   PVCOEFF_ELEMENT_D1,  &e_slot[iD1]);CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_DOWN_LEFT, PVCOEFF_VERTEX_A,&av_slot[0]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_DOWN_RIGHT,PVCOEFF_VERTEX_A,&av_slot[1]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_UP_LEFT,   PVCOEFF_VERTEX_A,&av_slot[2]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_UP_RIGHT,  PVCOEFF_VERTEX_A,&av_slot[3]);CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_LEFT,   PVCOEFF_FACE_B,   &b_slot[iL]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_RIGHT,  PVCOEFF_FACE_B,   &b_slot[iR]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_DOWN,   PVCOEFF_FACE_B,   &b_slot[iD]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_UP,     PVCOEFF_FACE_B,   &b_slot[iU]);CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_LEFT,   PVCOEFF_FACE_D2,   &d2_slot[iL]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_RIGHT,  PVCOEFF_FACE_D2,   &d2_slot[iR]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_DOWN,   PVCOEFF_FACE_D2,   &d2_slot[iD]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_UP,     PVCOEFF_FACE_D2,   &d2_slot[iU]);CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_LEFT,   PVCOEFF_FACE_D3,   &d3_slot[iL]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_RIGHT,  PVCOEFF_FACE_D3,   &d3_slot[iR]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_DOWN,   PVCOEFF_FACE_D3,   &d3_slot[iD]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_UP,     PVCOEFF_FACE_D3,   &d3_slot[iU]);CHKERRQ(ierr);

  PetscInt iwtc[3],iwtl[3],iwtr[3],iwtd[3],iwtu[3], iwtld[3],iwtrd[3],iwtlu[3],iwtru[3];
  ierr = DMStagGetLocationSlot(usr->dmMPhase, ELEMENT, 0, &iwtc[0]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmMPhase, ELEMENT, 1, &iwtc[1]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmMPhase, ELEMENT, 2, &iwtc[2]); CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(usr->dmMPhase, DMSTAG_LEFT, 0, &iwtl[0]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmMPhase, DMSTAG_LEFT, 1, &iwtl[1]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmMPhase, DMSTAG_LEFT, 2, &iwtl[2]); CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(usr->dmMPhase, DMSTAG_RIGHT, 0, &iwtr[0]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmMPhase, DMSTAG_RIGHT, 1, &iwtr[1]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmMPhase, DMSTAG_RIGHT, 2, &iwtr[2]); CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(usr->dmMPhase, DMSTAG_DOWN, 0, &iwtd[0]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmMPhase, DMSTAG_DOWN, 1, &iwtd[1]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmMPhase, DMSTAG_DOWN, 2, &iwtd[2]); CHKERRQ(ierr);
  
  ierr = DMStagGetLocationSlot(usr->dmMPhase, DMSTAG_UP, 0, &iwtu[0]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmMPhase, DMSTAG_UP, 1, &iwtu[1]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmMPhase, DMSTAG_UP, 2, &iwtu[2]); CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(usr->dmMPhase, DOWN_LEFT, 0, &iwtld[0]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmMPhase, DOWN_LEFT, 1, &iwtld[1]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmMPhase, DOWN_LEFT, 2, &iwtld[2]); CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(usr->dmMPhase, DOWN_RIGHT, 0, &iwtrd[0]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmMPhase, DOWN_RIGHT, 1, &iwtrd[1]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmMPhase, DOWN_RIGHT, 2, &iwtrd[2]); CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(usr->dmMPhase, UP_LEFT, 0, &iwtlu[0]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmMPhase, UP_LEFT, 1, &iwtlu[1]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmMPhase, UP_LEFT, 2, &iwtlu[2]); CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(usr->dmMPhase, UP_RIGHT, 0, &iwtru[0]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmMPhase, UP_RIGHT, 1, &iwtru[1]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmMPhase, UP_RIGHT, 2, &iwtru[2]); CHKERRQ(ierr);

  PetscInt ixx, izz, ixz, iII, ixxn[4], izzn[4], ixzn[4], iIIn[4];
  ierr = DMStagGetLocationSlot(usr->dmeps, ELEMENT, 0, &ixx); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, ELEMENT, 1, &izz); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, ELEMENT, 2, &ixz); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, ELEMENT, 3, &iII); CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(usr->dmeps, DOWN_LEFT, 0, &ixxn[0]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, DOWN_LEFT, 1, &izzn[0]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, DOWN_LEFT, 2, &ixzn[0]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, DOWN_LEFT, 3, &iIIn[0]); CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(usr->dmeps, DOWN_RIGHT, 0, &ixxn[1]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, DOWN_RIGHT, 1, &izzn[1]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, DOWN_RIGHT, 2, &ixzn[1]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, DOWN_RIGHT, 3, &iIIn[1]); CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(usr->dmeps, UP_LEFT, 0, &ixxn[2]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, UP_LEFT, 1, &izzn[2]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, UP_LEFT, 2, &ixzn[2]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, UP_LEFT, 3, &iIIn[2]); CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(usr->dmeps, UP_RIGHT, 0, &ixxn[3]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, UP_RIGHT, 1, &izzn[3]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, UP_RIGHT, 2, &ixzn[3]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, UP_RIGHT, 3, &iIIn[3]); CHKERRQ(ierr);

  // Loop over local domain 
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      PetscScalar   pc[9], DPoldc[9], dx, dz;
      PetscInt      ii, im, jm, ip, jp; // iph

      if (i == 0   ) im = i; else im = i-1;
      if (i == Nx-1) ip = i; else ip = i+1;
      if (j == 0   ) jm = j; else jm = j-1;
      if (j == Nz-1) jp = j; else jp = j+1;

      // should be adapted to variable spacing
      dx = coordx[i][inext]-coordx[i][iprev];
      dz = coordz[j][inext]-coordz[j][iprev];

      // get pc, dpold - center
      ierr = Get9PointCenterValues(i,j,iPV,Nx,Nz,xx,pc);CHKERRQ(ierr);
      ierr = Get9PointCenterValues(i,j,iP,Nx,Nz,_DPold,DPoldc);CHKERRQ(ierr);

      // Prepare for pointwise rheology calculation
      PetscScalar eta_eff[9], zeta_eff[9], chis[9], chip[9], txx[9], tzz[9], txz[9], tII[9], DP[9];
      PetscScalar eta_v[9],eta_e[9],zeta_v[9],zeta_e[9], Y[9];
      PetscScalar e[4], t[4], P[2], res[14];
      PetscInt ix[4];

      // center points
      ix[0] = ixx; ix[1] = izz; ix[2] = ixz; ix[3] = iII;

      P[0] = pc[0]; P[1] = DPoldc[0];
      ierr = GetTensorPointValues(i,j,ix,_eps,e);CHKERRQ(ierr);
      ierr = GetTensorPointValues(i,j,ix,_tauold,t);CHKERRQ(ierr);
      ierr = RheologyPointwise(i,j,xwt,iwtc,phi,e,t,P,res,usr);CHKERRQ(ierr);
      ierr = DecompactRheologyVars(0,res,eta_eff,eta_v,eta_e,zeta_eff,zeta_v,zeta_e,chis,chip,txx,tzz,txz,tII,DP,Y);CHKERRQ(ierr);

      P[0] = pc[1]; P[1] = DPoldc[1];
      ierr = GetTensorPointValues(im,j,ix,_eps,e);CHKERRQ(ierr);
      ierr = GetTensorPointValues(im,j,ix,_tauold,t);CHKERRQ(ierr);
      ierr = RheologyPointwise(im,j,xwt,iwtc,phi,e,t,P,res,usr);CHKERRQ(ierr);
      ierr = DecompactRheologyVars(1,res,eta_eff,eta_v,eta_e,zeta_eff,zeta_v,zeta_e,chis,chip,txx,tzz,txz,tII,DP,Y);CHKERRQ(ierr);

      P[0] = pc[2]; P[1] = DPoldc[2];
      ierr = GetTensorPointValues(ip,j,ix,_eps,e);CHKERRQ(ierr);
      ierr = GetTensorPointValues(ip,j,ix,_tauold,t);CHKERRQ(ierr);
      ierr = RheologyPointwise(ip,j,xwt,iwtc,phi,e,t,P,res,usr);CHKERRQ(ierr);
      ierr = DecompactRheologyVars(2,res,eta_eff,eta_v,eta_e,zeta_eff,zeta_v,zeta_e,chis,chip,txx,tzz,txz,tII,DP,Y);CHKERRQ(ierr);

      P[0] = pc[3]; P[1] = DPoldc[3];
      ierr = GetTensorPointValues(i,jm,ix,_eps,e);CHKERRQ(ierr);
      ierr = GetTensorPointValues(i,jm,ix,_tauold,t);CHKERRQ(ierr);
      ierr = RheologyPointwise(i,jm,xwt,iwtc,phi,e,t,P,res,usr);CHKERRQ(ierr);
      ierr = DecompactRheologyVars(3,res,eta_eff,eta_v,eta_e,zeta_eff,zeta_v,zeta_e,chis,chip,txx,tzz,txz,tII,DP,Y);CHKERRQ(ierr);

      P[0] = pc[4]; P[1] = DPoldc[4];
      ierr = GetTensorPointValues(i,jp,ix,_eps,e);CHKERRQ(ierr);
      ierr = GetTensorPointValues(i,jp,ix,_tauold,t);CHKERRQ(ierr);
      ierr = RheologyPointwise(i,jp,xwt,iwtc,phi,e,t,P,res,usr);CHKERRQ(ierr);
      ierr = DecompactRheologyVars(4,res,eta_eff,eta_v,eta_e,zeta_eff,zeta_v,zeta_e,chis,chip,txx,tzz,txz,tII,DP,Y);CHKERRQ(ierr);

      // corner points
      PetscScalar pcorner[4], DPoldcorner[4];
      ierr = GetCornerAvgFromCenter(pc,pcorner);CHKERRQ(ierr);
      ierr = GetCornerAvgFromCenter(DPoldc,DPoldcorner);CHKERRQ(ierr);
      
      ii = 0; ix[0] = ixxn[ii]; ix[1] = izzn[ii]; ix[2] = ixzn[ii]; ix[3] = iIIn[ii]; P[0] = pcorner[ii]; P[1] = DPoldcorner[ii];
      ierr = GetTensorPointValues(i,j,ix,_eps,e);CHKERRQ(ierr);
      ierr = GetTensorPointValues(i,j,ix,_tauold,t);CHKERRQ(ierr);
      ierr = RheologyPointwise(i,j,xwt,iwtld,phi,e,t,P,res,usr);CHKERRQ(ierr);
      ierr = DecompactRheologyVars(5,res,eta_eff,eta_v,eta_e,zeta_eff,zeta_v,zeta_e,chis,chip,txx,tzz,txz,tII,DP,Y);CHKERRQ(ierr);

      ii = 1; ix[0] = ixxn[ii]; ix[1] = izzn[ii]; ix[2] = ixzn[ii]; ix[3] = iIIn[ii]; P[0] = pcorner[ii]; P[1] = DPoldcorner[ii];
      ierr = GetTensorPointValues(i,j,ix,_eps,e);CHKERRQ(ierr);
      ierr = GetTensorPointValues(i,j,ix,_tauold,t);CHKERRQ(ierr);
      ierr = RheologyPointwise(i,j,xwt,iwtrd,phi,e,t,P,res,usr);CHKERRQ(ierr);
      ierr = DecompactRheologyVars(6,res,eta_eff,eta_v,eta_e,zeta_eff,zeta_v,zeta_e,chis,chip,txx,tzz,txz,tII,DP,Y);CHKERRQ(ierr);

      ii = 2; ix[0] = ixxn[ii]; ix[1] = izzn[ii]; ix[2] = ixzn[ii]; ix[3] = iIIn[ii]; P[0] = pcorner[ii]; P[1] = DPoldcorner[ii];
      ierr = GetTensorPointValues(i,j,ix,_eps,e);CHKERRQ(ierr);
      ierr = GetTensorPointValues(i,j,ix,_tauold,t);CHKERRQ(ierr);
      ierr = RheologyPointwise(i,j,xwt,iwtlu,phi,e,t,P,res,usr);CHKERRQ(ierr);
      ierr = DecompactRheologyVars(7,res,eta_eff,eta_v,eta_e,zeta_eff,zeta_v,zeta_e,chis,chip,txx,tzz,txz,tII,DP,Y);CHKERRQ(ierr);

      ii = 3; ix[0] = ixxn[ii]; ix[1] = izzn[ii]; ix[2] = ixzn[ii]; ix[3] = iIIn[ii]; P[0] = pcorner[ii]; P[1] = DPoldcorner[ii];
      ierr = GetTensorPointValues(i,j,ix,_eps,e);CHKERRQ(ierr);
      ierr = GetTensorPointValues(i,j,ix,_tauold,t);CHKERRQ(ierr);
      ierr = RheologyPointwise(i,j,xwt,iwtru,phi,e,t,P,res,usr);CHKERRQ(ierr);
      ierr = DecompactRheologyVars(8,res,eta_eff,eta_v,eta_e,zeta_eff,zeta_v,zeta_e,chis,chip,txx,tzz,txz,tII,DP,Y);CHKERRQ(ierr);

      { // element 
        // C = 0 (center, c=0)
        c[j][i][e_slot[iC]] = 0.0;

        // A = eta_eff (center, c=1)
        c[j][i][e_slot[iA]] = eta_eff[0];

        // D1 = zeta_eff-2/3eta_eff (center, c=2)
        c[j][i][e_slot[iD1]] = zeta_eff[0]-2.0/3.0*eta_eff[0];
      }

      { // corner
        // A = eta_eff (corner, c=0)
        for (ii = 0; ii < 4; ii++) {
          c[j][i][av_slot[ii]] = eta_eff[5+ii];
        }
      }
      
      { // face
        PetscScalar B[4], D2[4], D3[4], rhs[4], divchitau[4],gradchidp[4];

        gradchidp[0] = (chip[0]*_DPold[j ][i ][iP] - chip[1]*_DPold[j ][im][iP])/dx;
        gradchidp[1] = (chip[2]*_DPold[j ][ip][iP] - chip[0]*_DPold[j ][i ][iP])/dx;
        gradchidp[2] = (chip[0]*_DPold[j ][i ][iP] - chip[3]*_DPold[jm][i ][iP])/dz;
        gradchidp[3] = (chip[4]*_DPold[jp][i ][iP] - chip[0]*_DPold[j ][i ][iP])/dz;

        //  div(chis*tau_old) = div(S) = [dSxx/dx+dSxz/dz, dSzx/dx+dSzz/dz]
        divchitau[0] = (chis[0]*_tauold[j][i ][ixx] - chis[1]*_tauold[j][im][ixx])/dx + (chis[7]*_tauold[j][i][ixzn[2]]-chis[5]*_tauold[j][i][ixzn[0]])/dz;
        divchitau[1] = (chis[2]*_tauold[j][ip][ixx] - chis[0]*_tauold[j][i ][ixx])/dx + (chis[8]*_tauold[j][i][ixzn[3]]-chis[6]*_tauold[j][i][ixzn[1]])/dz;
        divchitau[2] = (chis[6]*_tauold[j][i][ixzn[1]]-chis[5]*_tauold[j][i][ixzn[0]])/dx + (chis[0]*_tauold[j ][i][izz] - chis[3]*_tauold[jm][i][izz])/dz;
        divchitau[3] = (chis[8]*_tauold[j][i][ixzn[3]]-chis[7]*_tauold[j][i][ixzn[2]])/dx + (chis[4]*_tauold[jp][i][izz] - chis[0]*_tauold[j ][i][izz])/dz;

        // body force term
        rhs[0] = 0.0;
        rhs[1] = 0.0;
        rhs[2] = phi*F;
        rhs[3] = phi*F;

        // // RHS = 0 on the boundaries
        // if (i==0   ) { rhs[0] = 0.0; gradchidp[0] = 0.0; divchitau[0] = 0.0; }
        // if (i==Nx-1) { rhs[1] = 0.0; gradchidp[1] = 0.0; divchitau[1] = 0.0; }
        // if (j==0   ) { rhs[2] = 0.0; gradchidp[2] = 0.0; divchitau[2] = 0.0; }
        // if (j==Nz-1) { rhs[3] = 0.0; gradchidp[3] = 0.0; divchitau[3] = 0.0; }

        // B[0]  = rhs[0]-divchitau[0]+gradchidp[0];
        // B[1]  = rhs[1]+divchitau[1]-gradchidp[1];
        // B[2]  = rhs[2]-divchitau[2]+gradchidp[2];
        // B[3]  = rhs[3]+divchitau[3]-gradchidp[3];

        for (ii = 0; ii < 4; ii++) {
          B[ii]  = rhs[ii]-divchitau[ii]+gradchidp[ii];
          D2[ii] = -R*R*Kphi;
          D3[ii] = -R*R*Kphi*F;

          // B = phi*F*ek - div(chi_s*tau_old) + grad(chi_p*dP_old) (edges, c=0)
          // (F = (rho^s-rho^f)*U*L/eta_ref * g*L/U^2) 
          c[j][i][b_slot[ii]] = B[ii];

          // D2 = -R^2 * Kphi (edges, c=1)
          c[j][i][d2_slot[ii]] = D2[ii];

          // D3 = R^2 * Kphi * F (edges, c=2)
          c[j][i][d3_slot[ii]] = D3[ii];
        }
      }

      // save stresses for output + dotlam
      _tau[j][i][ixx]     = txx[0]; _tau[j][i][izz]     = tzz[0]; _tau[j][i][ixz]     = txz[0]; _tau[j][i][iII]     = tII[0];
      _tau[j][i][ixxn[0]] = txx[5]; _tau[j][i][izzn[0]] = tzz[5]; _tau[j][i][ixzn[0]] = txz[5]; _tau[j][i][iIIn[0]] = tII[5];
      _tau[j][i][ixxn[1]] = txx[6]; _tau[j][i][izzn[1]] = tzz[6]; _tau[j][i][ixzn[1]] = txz[6]; _tau[j][i][iIIn[1]] = tII[6];
      _tau[j][i][ixxn[2]] = txx[7]; _tau[j][i][izzn[2]] = tzz[7]; _tau[j][i][ixzn[2]] = txz[7]; _tau[j][i][iIIn[2]] = tII[7];
      _tau[j][i][ixxn[3]] = txx[8]; _tau[j][i][izzn[3]] = tzz[8]; _tau[j][i][ixzn[3]] = txz[8]; _tau[j][i][iIIn[3]] = tII[8];
      _DP[j][i][iP]    = DP[0];
      _plast[j][i][iP] = Y[0];
    }
  }

  // Restore arrays, local vectors
  ierr = DMStagRestoreProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArray(dmcoeff,coefflocal,&c);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  ierr = VecDestroy(&coefflocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArrayRead(usr->dmeps,xepslocal,&_eps);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmeps,&xepslocal); CHKERRQ(ierr);
  ierr = DMStagVecRestoreArrayRead(usr->dmeps,xtauoldlocal,&_tauold);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmeps,&xtauoldlocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArray(usr->dmeps,xtaulocal,&_tau);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(usr->dmeps,xtaulocal,INSERT_VALUES,usr->xtau); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (usr->dmeps,xtaulocal,INSERT_VALUES,usr->xtau); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmeps,&xtaulocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArrayRead(usr->dmP,xDPoldlocal,&_DPold);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmP,&xDPoldlocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArray(usr->dmP,xDPlocal,&_DP);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(usr->dmP,xDPlocal,INSERT_VALUES,usr->xDP); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (usr->dmP,xDPlocal,INSERT_VALUES,usr->xDP); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmP,&xDPlocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArray(usr->dmP,xplastlocal,&_plast);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(usr->dmP,xplastlocal,INSERT_VALUES,usr->xplast); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (usr->dmP,xplastlocal,INSERT_VALUES,usr->xplast); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmP,&xplastlocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArray(usr->dmMPhase,xMPhaselocal,&xwt);CHKERRQ(ierr);
  ierr = VecDestroy(&xMPhaselocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArrayRead(dm,xlocal,&xx);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&xlocal); CHKERRQ(ierr);

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
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz;
  Vec            coefflocal, xlocal, xepslocal, xtaulocal, xtauoldlocal, xDPlocal, xDPoldlocal, xplastlocal, xMPhaselocal;
  PetscScalar    **coordx,**coordz;
  PetscInt       iprev, inext, icenter;
  PetscScalar    ***c, ***xx, ***_eps, ***_tauold, ***_tau, ***_DPold, ***_DP, ***_plast, ***xwt;
  PetscScalar    phi, F;//, Kphi, dt, R; 
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  // dt = usr->par->dt;
  // R = usr->par->R;
  F = usr->par->F;

  // Uniform porosity and permeability
  phi = 0.0;
  // Kphi = 1.0;

  // Get coefficient
  ierr = DMStagGetGlobalSizes(dmcoeff, &Nx, &Nz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmcoeff,LEFT,&iprev);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmcoeff,RIGHT,&inext);CHKERRQ(ierr); 
  ierr = DMStagGetProductCoordinateLocationSlot(dmcoeff,ELEMENT,&icenter);CHKERRQ(ierr);

  // Create coefficient local vector
  ierr = DMCreateLocalVector(dmcoeff, &coefflocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dmcoeff, coefflocal, &c); CHKERRQ(ierr);

  // Strain rates
  ierr = UpdateStrainRates(dm,x,usr); CHKERRQ(ierr);
  ierr = DMGetLocalVector(usr->dmeps, &xepslocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmeps, usr->xeps, INSERT_VALUES, xepslocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(usr->dmeps,xepslocal,&_eps);CHKERRQ(ierr);

  ierr = DMGetLocalVector(usr->dmeps, &xtauoldlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmeps, usr->xtau_old, INSERT_VALUES, xtauoldlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(usr->dmeps,xtauoldlocal,&_tauold);CHKERRQ(ierr);

  ierr = DMGetLocalVector(usr->dmeps, &xtaulocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmeps, usr->xtau, INSERT_VALUES, xtaulocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(usr->dmeps,xtaulocal,&_tau);CHKERRQ(ierr);

  // Get dm and solution vector
  ierr = DMGetLocalVector(dm, &xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm, x, INSERT_VALUES, xlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(dm,xlocal,&xx);CHKERRQ(ierr);

  ierr = DMGetLocalVector(usr->dmP, &xDPoldlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmP, usr->xDP_old, INSERT_VALUES, xDPoldlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(usr->dmP,xDPoldlocal,&_DPold);CHKERRQ(ierr);

  ierr = DMGetLocalVector(usr->dmP, &xDPlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmP, usr->xDP, INSERT_VALUES, xDPlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(usr->dmP,xDPlocal,&_DP);CHKERRQ(ierr);

  ierr = DMGetLocalVector(usr->dmP, &xplastlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmP, usr->xplast, INSERT_VALUES, xplastlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(usr->dmP,xplastlocal,&_plast);CHKERRQ(ierr);

  // get material phase fractions
  ierr = DMCreateLocalVector(usr->dmMPhase, &xMPhaselocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(usr->dmMPhase,usr->xMPhase,INSERT_VALUES,xMPhaselocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (usr->dmMPhase,usr->xMPhase,INSERT_VALUES,xMPhaselocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(usr->dmMPhase, xMPhaselocal, &xwt); CHKERRQ(ierr);

  // get location slots
  PetscInt  iP;
  ierr = DMStagGetLocationSlot(usr->dmP,DMSTAG_ELEMENT,0,&iP);CHKERRQ(ierr);

  PetscInt iPV;
  ierr = DMStagGetLocationSlot(dm,DMSTAG_ELEMENT,PV_ELEMENT_P,&iPV);CHKERRQ(ierr);

  PetscInt  e_slot[3],av_slot[4],b_slot[4],iL,iR,iU,iD,iC,iA;//,iD1; // ,d2_slot[4],d3_slot[4]
  iL = 0; iR = 1; iD = 2; iU  = 3;
  iC = 0; iA = 1; //iD1= 2;
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_ELEMENT,   PVCOEFF_ELEMENT_C,   &e_slot[iC]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_ELEMENT,   PVCOEFF_ELEMENT_A,   &e_slot[iA]);CHKERRQ(ierr);
  // ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_ELEMENT,   PVCOEFF_ELEMENT_D1,  &e_slot[iD1]);CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_DOWN_LEFT, PVCOEFF_VERTEX_A,&av_slot[0]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_DOWN_RIGHT,PVCOEFF_VERTEX_A,&av_slot[1]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_UP_LEFT,   PVCOEFF_VERTEX_A,&av_slot[2]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_UP_RIGHT,  PVCOEFF_VERTEX_A,&av_slot[3]);CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_LEFT,   PVCOEFF_FACE_B,   &b_slot[iL]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_RIGHT,  PVCOEFF_FACE_B,   &b_slot[iR]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_DOWN,   PVCOEFF_FACE_B,   &b_slot[iD]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_UP,     PVCOEFF_FACE_B,   &b_slot[iU]);CHKERRQ(ierr);

  // ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_LEFT,   PVCOEFF_FACE_D2,   &d2_slot[iL]);CHKERRQ(ierr);
  // ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_RIGHT,  PVCOEFF_FACE_D2,   &d2_slot[iR]);CHKERRQ(ierr);
  // ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_DOWN,   PVCOEFF_FACE_D2,   &d2_slot[iD]);CHKERRQ(ierr);
  // ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_UP,     PVCOEFF_FACE_D2,   &d2_slot[iU]);CHKERRQ(ierr);

  // ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_LEFT,   PVCOEFF_FACE_D3,   &d3_slot[iL]);CHKERRQ(ierr);
  // ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_RIGHT,  PVCOEFF_FACE_D3,   &d3_slot[iR]);CHKERRQ(ierr);
  // ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_DOWN,   PVCOEFF_FACE_D3,   &d3_slot[iD]);CHKERRQ(ierr);
  // ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_UP,     PVCOEFF_FACE_D3,   &d3_slot[iU]);CHKERRQ(ierr);

  PetscInt iwtc[3],iwtl[3],iwtr[3],iwtd[3],iwtu[3], iwtld[3],iwtrd[3],iwtlu[3],iwtru[3];
  ierr = DMStagGetLocationSlot(usr->dmMPhase, ELEMENT, 0, &iwtc[0]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmMPhase, ELEMENT, 1, &iwtc[1]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmMPhase, ELEMENT, 2, &iwtc[2]); CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(usr->dmMPhase, DMSTAG_LEFT, 0, &iwtl[0]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmMPhase, DMSTAG_LEFT, 1, &iwtl[1]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmMPhase, DMSTAG_LEFT, 2, &iwtl[2]); CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(usr->dmMPhase, DMSTAG_RIGHT, 0, &iwtr[0]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmMPhase, DMSTAG_RIGHT, 1, &iwtr[1]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmMPhase, DMSTAG_RIGHT, 2, &iwtr[2]); CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(usr->dmMPhase, DMSTAG_DOWN, 0, &iwtd[0]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmMPhase, DMSTAG_DOWN, 1, &iwtd[1]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmMPhase, DMSTAG_DOWN, 2, &iwtd[2]); CHKERRQ(ierr);
  
  ierr = DMStagGetLocationSlot(usr->dmMPhase, DMSTAG_UP, 0, &iwtu[0]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmMPhase, DMSTAG_UP, 1, &iwtu[1]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmMPhase, DMSTAG_UP, 2, &iwtu[2]); CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(usr->dmMPhase, DOWN_LEFT, 0, &iwtld[0]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmMPhase, DOWN_LEFT, 1, &iwtld[1]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmMPhase, DOWN_LEFT, 2, &iwtld[2]); CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(usr->dmMPhase, DOWN_RIGHT, 0, &iwtrd[0]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmMPhase, DOWN_RIGHT, 1, &iwtrd[1]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmMPhase, DOWN_RIGHT, 2, &iwtrd[2]); CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(usr->dmMPhase, UP_LEFT, 0, &iwtlu[0]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmMPhase, UP_LEFT, 1, &iwtlu[1]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmMPhase, UP_LEFT, 2, &iwtlu[2]); CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(usr->dmMPhase, UP_RIGHT, 0, &iwtru[0]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmMPhase, UP_RIGHT, 1, &iwtru[1]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmMPhase, UP_RIGHT, 2, &iwtru[2]); CHKERRQ(ierr);

  PetscInt ixx, izz, ixz, iII, ixxn[4], izzn[4], ixzn[4], iIIn[4];
  ierr = DMStagGetLocationSlot(usr->dmeps, ELEMENT, 0, &ixx); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, ELEMENT, 1, &izz); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, ELEMENT, 2, &ixz); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, ELEMENT, 3, &iII); CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(usr->dmeps, DOWN_LEFT, 0, &ixxn[0]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, DOWN_LEFT, 1, &izzn[0]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, DOWN_LEFT, 2, &ixzn[0]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, DOWN_LEFT, 3, &iIIn[0]); CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(usr->dmeps, DOWN_RIGHT, 0, &ixxn[1]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, DOWN_RIGHT, 1, &izzn[1]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, DOWN_RIGHT, 2, &ixzn[1]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, DOWN_RIGHT, 3, &iIIn[1]); CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(usr->dmeps, UP_LEFT, 0, &ixxn[2]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, UP_LEFT, 1, &izzn[2]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, UP_LEFT, 2, &ixzn[2]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, UP_LEFT, 3, &iIIn[2]); CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(usr->dmeps, UP_RIGHT, 0, &ixxn[3]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, UP_RIGHT, 1, &izzn[3]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, UP_RIGHT, 2, &ixzn[3]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, UP_RIGHT, 3, &iIIn[3]); CHKERRQ(ierr);

  // Loop over local domain 
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      PetscScalar   pc[9], DPoldc[9], dx, dz;
      PetscInt      ii, im, jm, ip, jp; // iph

      if (i == 0   ) im = i; else im = i-1;
      if (i == Nx-1) ip = i; else ip = i+1;
      if (j == 0   ) jm = j; else jm = j-1;
      if (j == Nz-1) jp = j; else jp = j+1;

      // should be adapted to variable spacing
      dx = coordx[i][inext]-coordx[i][iprev];
      dz = coordz[j][inext]-coordz[j][iprev];

      // get pc, dpold - center
      ierr = Get9PointCenterValues(i,j,iPV,Nx,Nz,xx,pc);CHKERRQ(ierr);
      ierr = Get9PointCenterValues(i,j,iP,Nx,Nz,_DPold,DPoldc);CHKERRQ(ierr);

      // Prepare for pointwise rheology calculation
      PetscScalar eta_eff[9], zeta_eff[9], chis[9], chip[9], txx[9], tzz[9], txz[9], tII[9], DP[9];
      PetscScalar eta_v[9],eta_e[9],zeta_v[9],zeta_e[9], Y[9];
      PetscScalar e[4], t[4], P[2], res[14];
      PetscInt ix[4];

      // center points
      ix[0] = ixx; ix[1] = izz; ix[2] = ixz; ix[3] = iII;

      P[0] = pc[0]; P[1] = DPoldc[0];
      ierr = GetTensorPointValues(i,j,ix,_eps,e);CHKERRQ(ierr);
      ierr = GetTensorPointValues(i,j,ix,_tauold,t);CHKERRQ(ierr);
      ierr = RheologyPointwise(i,j,xwt,iwtc,phi,e,t,P,res,usr);CHKERRQ(ierr);
      ierr = DecompactRheologyVars(0,res,eta_eff,eta_v,eta_e,zeta_eff,zeta_v,zeta_e,chis,chip,txx,tzz,txz,tII,DP,Y);CHKERRQ(ierr);

      P[0] = pc[1]; P[1] = DPoldc[1];
      ierr = GetTensorPointValues(im,j,ix,_eps,e);CHKERRQ(ierr);
      ierr = GetTensorPointValues(im,j,ix,_tauold,t);CHKERRQ(ierr);
      ierr = RheologyPointwise(im,j,xwt,iwtc,phi,e,t,P,res,usr);CHKERRQ(ierr);
      ierr = DecompactRheologyVars(1,res,eta_eff,eta_v,eta_e,zeta_eff,zeta_v,zeta_e,chis,chip,txx,tzz,txz,tII,DP,Y);CHKERRQ(ierr);

      P[0] = pc[2]; P[1] = DPoldc[2];
      ierr = GetTensorPointValues(ip,j,ix,_eps,e);CHKERRQ(ierr);
      ierr = GetTensorPointValues(ip,j,ix,_tauold,t);CHKERRQ(ierr);
      ierr = RheologyPointwise(ip,j,xwt,iwtc,phi,e,t,P,res,usr);CHKERRQ(ierr);
      ierr = DecompactRheologyVars(2,res,eta_eff,eta_v,eta_e,zeta_eff,zeta_v,zeta_e,chis,chip,txx,tzz,txz,tII,DP,Y);CHKERRQ(ierr);

      P[0] = pc[3]; P[1] = DPoldc[3];
      ierr = GetTensorPointValues(i,jm,ix,_eps,e);CHKERRQ(ierr);
      ierr = GetTensorPointValues(i,jm,ix,_tauold,t);CHKERRQ(ierr);
      ierr = RheologyPointwise(i,jm,xwt,iwtc,phi,e,t,P,res,usr);CHKERRQ(ierr);
      ierr = DecompactRheologyVars(3,res,eta_eff,eta_v,eta_e,zeta_eff,zeta_v,zeta_e,chis,chip,txx,tzz,txz,tII,DP,Y);CHKERRQ(ierr);

      P[0] = pc[4]; P[1] = DPoldc[4];
      ierr = GetTensorPointValues(i,jp,ix,_eps,e);CHKERRQ(ierr);
      ierr = GetTensorPointValues(i,jp,ix,_tauold,t);CHKERRQ(ierr);
      ierr = RheologyPointwise(i,jp,xwt,iwtc,phi,e,t,P,res,usr);CHKERRQ(ierr);
      ierr = DecompactRheologyVars(4,res,eta_eff,eta_v,eta_e,zeta_eff,zeta_v,zeta_e,chis,chip,txx,tzz,txz,tII,DP,Y);CHKERRQ(ierr);

      // corner points
      PetscScalar pcorner[4], DPoldcorner[4];
      ierr = GetCornerAvgFromCenter(pc,pcorner);CHKERRQ(ierr);
      ierr = GetCornerAvgFromCenter(DPoldc,DPoldcorner);CHKERRQ(ierr);
      
      ii = 0; ix[0] = ixxn[ii]; ix[1] = izzn[ii]; ix[2] = ixzn[ii]; ix[3] = iIIn[ii]; P[0] = pcorner[ii]; P[1] = DPoldcorner[ii];
      ierr = GetTensorPointValues(i,j,ix,_eps,e);CHKERRQ(ierr);
      ierr = GetTensorPointValues(i,j,ix,_tauold,t);CHKERRQ(ierr);
      ierr = RheologyPointwise(i,j,xwt,iwtld,phi,e,t,P,res,usr);CHKERRQ(ierr);
      ierr = DecompactRheologyVars(5,res,eta_eff,eta_v,eta_e,zeta_eff,zeta_v,zeta_e,chis,chip,txx,tzz,txz,tII,DP,Y);CHKERRQ(ierr);

      ii = 1; ix[0] = ixxn[ii]; ix[1] = izzn[ii]; ix[2] = ixzn[ii]; ix[3] = iIIn[ii]; P[0] = pcorner[ii]; P[1] = DPoldcorner[ii];
      ierr = GetTensorPointValues(i,j,ix,_eps,e);CHKERRQ(ierr);
      ierr = GetTensorPointValues(i,j,ix,_tauold,t);CHKERRQ(ierr);
      ierr = RheologyPointwise(i,j,xwt,iwtrd,phi,e,t,P,res,usr);CHKERRQ(ierr);
      ierr = DecompactRheologyVars(6,res,eta_eff,eta_v,eta_e,zeta_eff,zeta_v,zeta_e,chis,chip,txx,tzz,txz,tII,DP,Y);CHKERRQ(ierr);

      ii = 2; ix[0] = ixxn[ii]; ix[1] = izzn[ii]; ix[2] = ixzn[ii]; ix[3] = iIIn[ii]; P[0] = pcorner[ii]; P[1] = DPoldcorner[ii];
      ierr = GetTensorPointValues(i,j,ix,_eps,e);CHKERRQ(ierr);
      ierr = GetTensorPointValues(i,j,ix,_tauold,t);CHKERRQ(ierr);
      ierr = RheologyPointwise(i,j,xwt,iwtlu,phi,e,t,P,res,usr);CHKERRQ(ierr);
      ierr = DecompactRheologyVars(7,res,eta_eff,eta_v,eta_e,zeta_eff,zeta_v,zeta_e,chis,chip,txx,tzz,txz,tII,DP,Y);CHKERRQ(ierr);

      ii = 3; ix[0] = ixxn[ii]; ix[1] = izzn[ii]; ix[2] = ixzn[ii]; ix[3] = iIIn[ii]; P[0] = pcorner[ii]; P[1] = DPoldcorner[ii];
      ierr = GetTensorPointValues(i,j,ix,_eps,e);CHKERRQ(ierr);
      ierr = GetTensorPointValues(i,j,ix,_tauold,t);CHKERRQ(ierr);
      ierr = RheologyPointwise(i,j,xwt,iwtru,phi,e,t,P,res,usr);CHKERRQ(ierr);
      ierr = DecompactRheologyVars(8,res,eta_eff,eta_v,eta_e,zeta_eff,zeta_v,zeta_e,chis,chip,txx,tzz,txz,tII,DP,Y);CHKERRQ(ierr);

      { // element 
        // C = 0 (center, c=0)
        c[j][i][e_slot[iC]] = 0.0;

        // A = eta_eff (center, c=1)
        c[j][i][e_slot[iA]] = eta_eff[0];

        // // D1 = zeta_eff-2/3eta_eff (center, c=2)
        // c[j][i][e_slot[iD1]] = zeta_eff[0]-2.0/3.0*eta_eff[0];
      }

      { // corner
        // A = eta_eff (corner, c=0)
        for (ii = 0; ii < 4; ii++) {
          c[j][i][av_slot[ii]] = eta_eff[5+ii];
        }
      }
      
      { // face
        PetscScalar B[4], rhs[4], divchitau[4];//,gradchidp[4]; // , D2[4], D3[4]

        // gradchidp[0] = (chip[0]*_DPold[j ][i ][iP] - chip[1]*_DPold[j ][im][iP])/dx;
        // gradchidp[1] = (chip[2]*_DPold[j ][ip][iP] - chip[0]*_DPold[j ][i ][iP])/dx;
        // gradchidp[2] = (chip[0]*_DPold[j ][i ][iP] - chip[3]*_DPold[jm][i ][iP])/dz;
        // gradchidp[3] = (chip[4]*_DPold[jp][i ][iP] - chip[0]*_DPold[j ][i ][iP])/dz;

        //  div(chis*tau_old) = div(S) = [dSxx/dx+dSxz/dz, dSzx/dx+dSzz/dz]
        divchitau[0] = (chis[0]*_tauold[j][i ][ixx] - chis[1]*_tauold[j][im][ixx])/dx + (chis[7]*_tauold[j][i][ixzn[2]]-chis[5]*_tauold[j][i][ixzn[0]])/dz;
        divchitau[1] = (chis[2]*_tauold[j][ip][ixx] - chis[0]*_tauold[j][i ][ixx])/dx + (chis[8]*_tauold[j][i][ixzn[3]]-chis[6]*_tauold[j][i][ixzn[1]])/dz;
        divchitau[2] = (chis[6]*_tauold[j][i][ixzn[1]]-chis[5]*_tauold[j][i][ixzn[0]])/dx + (chis[0]*_tauold[j ][i][izz] - chis[3]*_tauold[jm][i][izz])/dz;
        divchitau[3] = (chis[8]*_tauold[j][i][ixzn[3]]-chis[7]*_tauold[j][i][ixzn[2]])/dx + (chis[4]*_tauold[jp][i][izz] - chis[0]*_tauold[j ][i][izz])/dz;

        // body force term
        rhs[0] = 0.0;
        rhs[1] = 0.0;
        rhs[2] = phi*F;
        rhs[3] = phi*F;

        // // RHS = 0 on the boundaries
        // if (i==0   ) { rhs[0] = 0.0; gradchidp[0] = 0.0; divchitau[0] = 0.0; }
        // if (i==Nx-1) { rhs[1] = 0.0; gradchidp[1] = 0.0; divchitau[1] = 0.0; }
        // if (j==0   ) { rhs[2] = 0.0; gradchidp[2] = 0.0; divchitau[2] = 0.0; }
        // if (j==Nz-1) { rhs[3] = 0.0; gradchidp[3] = 0.0; divchitau[3] = 0.0; }

        // B[0]  = rhs[0]-divchitau[0]+gradchidp[0];
        // B[1]  = rhs[1]+divchitau[1]-gradchidp[1];
        // B[2]  = rhs[2]-divchitau[2]+gradchidp[2];
        // B[3]  = rhs[3]+divchitau[3]-gradchidp[3];

        for (ii = 0; ii < 4; ii++) {
          B[ii]  = rhs[ii]-divchitau[ii];//+gradchidp[ii];
          // D2[ii] = -R*R*Kphi;
          // D3[ii] = -R*R*Kphi*F;

          // B = phi*F*ek - div(chi_s*tau_old) + grad(chi_p*dP_old) (edges, c=0)
          // (F = (rho^s-rho^f)*U*L/eta_ref * g*L/U^2) 
          c[j][i][b_slot[ii]] = B[ii];

          // // D2 = -R^2 * Kphi (edges, c=1)
          // c[j][i][d2_slot[ii]] = D2[ii];

          // // D3 = R^2 * Kphi * F (edges, c=2)
          // c[j][i][d2_slot[ii]] = D3[ii];
        }
      }

      // save stresses for output + dotlam
      _tau[j][i][ixx]     = txx[0]; _tau[j][i][izz]     = tzz[0]; _tau[j][i][ixz]     = txz[0]; _tau[j][i][iII]     = tII[0];
      _tau[j][i][ixxn[0]] = txx[5]; _tau[j][i][izzn[0]] = tzz[5]; _tau[j][i][ixzn[0]] = txz[5]; _tau[j][i][iIIn[0]] = tII[5];
      _tau[j][i][ixxn[1]] = txx[6]; _tau[j][i][izzn[1]] = tzz[6]; _tau[j][i][ixzn[1]] = txz[6]; _tau[j][i][iIIn[1]] = tII[6];
      _tau[j][i][ixxn[2]] = txx[7]; _tau[j][i][izzn[2]] = tzz[7]; _tau[j][i][ixzn[2]] = txz[7]; _tau[j][i][iIIn[2]] = tII[7];
      _tau[j][i][ixxn[3]] = txx[8]; _tau[j][i][izzn[3]] = tzz[8]; _tau[j][i][ixzn[3]] = txz[8]; _tau[j][i][iIIn[3]] = tII[8];
      _DP[j][i][iP]    = DP[0];
      _plast[j][i][iP] = Y[0];
    }
  }

  // Restore arrays, local vectors
  ierr = DMStagRestoreProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArray(dmcoeff,coefflocal,&c);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  ierr = VecDestroy(&coefflocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArrayRead(usr->dmeps,xepslocal,&_eps);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmeps,&xepslocal); CHKERRQ(ierr);
  ierr = DMStagVecRestoreArrayRead(usr->dmeps,xtauoldlocal,&_tauold);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmeps,&xtauoldlocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArray(usr->dmeps,xtaulocal,&_tau);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(usr->dmeps,xtaulocal,INSERT_VALUES,usr->xtau); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (usr->dmeps,xtaulocal,INSERT_VALUES,usr->xtau); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmeps,&xtaulocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArrayRead(usr->dmP,xDPoldlocal,&_DPold);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmP,&xDPoldlocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArray(usr->dmP,xDPlocal,&_DP);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(usr->dmP,xDPlocal,INSERT_VALUES,usr->xDP); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (usr->dmP,xDPlocal,INSERT_VALUES,usr->xDP); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmP,&xDPlocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArray(usr->dmP,xplastlocal,&_plast);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(usr->dmP,xplastlocal,INSERT_VALUES,usr->xplast); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (usr->dmP,xplastlocal,INSERT_VALUES,usr->xplast); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmP,&xplastlocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArray(usr->dmMPhase,xMPhaselocal,&xwt);CHKERRQ(ierr);
  ierr = VecDestroy(&xMPhaselocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArrayRead(dm,xlocal,&xx);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&xlocal); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
PetscErrorCode GetTensorPointValues(PetscInt i, PetscInt j, PetscInt *idx, PetscScalar ***xx, PetscScalar *x)
{
  PetscFunctionBegin;
    x[0] = xx[j][i][idx[0]]; // xx
    x[1] = xx[j][i][idx[1]]; // zz
    x[2] = xx[j][i][idx[2]]; // xz
    x[3] = xx[j][i][idx[3]]; // II
  PetscFunctionReturn(0);
}

// ---------------------------------------
PetscErrorCode GetCornerAvgFromCenter(PetscScalar *Ac, PetscScalar *Acorner)
{
  PetscFunctionBegin;
  Acorner[0] = (Ac[0]+Ac[1]+Ac[3]+Ac[5])*0.25;
  Acorner[1] = (Ac[0]+Ac[2]+Ac[3]+Ac[6])*0.25;
  Acorner[2] = (Ac[0]+Ac[1]+Ac[4]+Ac[7])*0.25;
  Acorner[3] = (Ac[0]+Ac[2]+Ac[4]+Ac[8])*0.25;
  PetscFunctionReturn(0);
}

// ---------------------------------------
PetscErrorCode GetMatPhaseFraction(PetscInt i, PetscInt j, PetscScalar ***xwt, PetscInt *iwtc, PetscInt n, PetscScalar *wt)
{ 
  PetscInt ii;
  PetscFunctionBegin;
  for (ii = 0; ii <n; ii++) {
    wt[ii] = xwt[j][i][iwtc[ii]];
  }
  PetscFunctionReturn(0);
}

// ---------------------------------------
PetscErrorCode DecompactRheologyVars(PetscInt ii,PetscScalar *res,PetscScalar *eta_eff,PetscScalar *eta_v,PetscScalar *eta_e,PetscScalar *zeta_eff,
                                     PetscScalar *zeta_v,PetscScalar *zeta_e,PetscScalar *chis,PetscScalar *chip,
                                     PetscScalar *txx,PetscScalar *tzz,PetscScalar *txz,PetscScalar *tII,PetscScalar *DP,PetscScalar *Y)
{
  PetscFunctionBegin;
    eta_eff[ii] = res[0]; 
    eta_v[ii]   = res[1]; 
    eta_e[ii]   = res[2]; 
    zeta_eff[ii]= res[3]; 
    zeta_v[ii]  = res[4]; 
    zeta_e[ii]  = res[5]; 
    chis[ii]    = res[6]; 
    chip[ii]    = res[7]; 
    txx[ii]     = res[8]; 
    tzz[ii]     = res[9]; 
    txz[ii]     = res[10]; 
    tII[ii]     = res[11]; 
    DP[ii]      = res[12]; 
    Y[ii]       = res[13];

  PetscFunctionReturn(0);
}

// ---------------------------------------
// RheologyPointwise
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "RheologyPointwise"
PetscErrorCode RheologyPointwise(PetscInt i, PetscInt j, PetscScalar ***xwt, PetscInt *iwt, PetscScalar phi, 
                                 PetscScalar *eps, PetscScalar *tauold, PetscScalar *P, PetscScalar *res, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  if (usr->par->rheology==0) { ierr = RheologyPointwise_DominantPhase(i,j,xwt,iwt,phi,eps,tauold,P,res,usr);CHKERRQ(ierr); }
  if (usr->par->rheology==1) { ierr = RheologyPointwise_AveragePhase(i,j,xwt,iwt,phi,eps,tauold,P,res,usr);CHKERRQ(ierr); }

  PetscFunctionReturn(0);
}

// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "RheologyPointwise_DominantPhase"
PetscErrorCode RheologyPointwise_DominantPhase(PetscInt i, PetscInt j, PetscScalar ***xwt, PetscInt *iwt, PetscScalar phi, 
                                 PetscScalar *eps, PetscScalar *tauold, PetscScalar *P, PetscScalar *res, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       ii, iph;
  PetscScalar    maxwt, dt;
  PetscScalar    eta_v, zeta_v, eta_e, zeta_e, Y, YC, em, nh, lambda;//, p;
  // PetscErrorCode ierr;
  PetscFunctionBeginUser;

  dt = usr->par->dt;
  em = usr->par->etamin;
  nh = usr->par->nh;
  lambda = usr->par->lambda;

  // get dominant marker phase and properties
  maxwt = 0.0;
  ii = 0;
  for (iph = 0; iph < usr->nph; iph++) {
    if (xwt[j][i][iwt[iph]]>maxwt) ii = iph;
  }
  eta_v  = usr->mat[ii].eta0*PetscExpScalar(-lambda*phi);
  zeta_v = usr->mat[ii].zeta0*PetscExpScalar(-lambda*phi);
  eta_e  = usr->mat[ii].G*dt;
  zeta_e = usr->mat[ii].Z0*dt;
  Y      = usr->mat[ii].C; // plastic yield criterion
  YC     = usr->mat[ii].C*usr->par->lam_p; // compaction failure criteria

  // get epsII, epsp, epspII - tau_old, dP
  PetscScalar exx, ezz, exz, eII, told_xx, told_zz, told_xz, DPold; //told_II,
  exx = eps[0]; told_xx = tauold[0]; 
  ezz = eps[1]; told_zz = tauold[1];
  exz = eps[2]; told_xz = tauold[2];
  eII = eps[3]; //told_II = tauold[3];
  // p = P[0]; 
  DPold = P[1];

  PetscScalar eII_dev, inv_eta_p, inv_eta_vp, inv_zeta_p, inv_zeta_vp;
  // second invariant of deviatoric strain rate
  eII_dev = PetscPowScalar((PetscPowScalar(eII,2) - 1.0/6.0*PetscPowScalar(exx+ezz,2)),0.5);

  // plastic viscosity
  if (usr->par->plasticity) { 
    inv_eta_p  = 2.0*eII_dev/Y;
    inv_zeta_p = PetscAbs(exx+ezz)/YC; //inv_zeta_p = inv_eta_p/lam_v;
  } else { 
    inv_eta_p = 0.0;
    inv_zeta_p = 0.0;
  }
  inv_eta_vp  = PetscPowScalar(PetscPowScalar(inv_eta_p, nh) + PetscPowScalar(1.0/eta_v, nh), 1.0/nh);
  inv_zeta_vp = PetscPowScalar(PetscPowScalar(inv_zeta_p, nh) + PetscPowScalar(1.0/zeta_v, nh), 1.0/nh);

  // effective viscosities
  PetscScalar eta, zeta, chis, chip;
  eta  = em + (1.0 - phi)/(inv_eta_vp  + 1.0/eta_e);
  zeta = em + (1.0 - phi)/(inv_zeta_vp + 1.0/zeta_e);

  // elastic stress evolution parameter - remove the cutoff minimum viscosity in calculating the built-up stress
  chis = (eta-em)/eta_e;
  chip = (zeta-em)/zeta_e;

  // shear and volumetric stresses
  PetscScalar phis, div, txx, tzz, txz, tII, DP;
  phis = 1 - phi;
  div = exx + ezz;
  txx = (2.0*(eta-em)*(exx-1.0/3.0*div) + chis*told_xx)/phis;
  tzz = (2.0*(eta-em)*(ezz-1.0/3.0*div) + chis*told_zz)/phis;
  txz = (2.0*(eta-em)*exz + chis*told_xz)/phis;
  tII = PetscPowScalar(0.5*(txx*txx + tzz*tzz + 2*txz*txz),0.5);
  DP  = (-(zeta-em)*div + chip*DPold)/phis;

  // return values
  res[0]  = eta;
  res[1]  = eta_v;
  res[2]  = eta_e;
  res[3]  = zeta;
  res[4]  = zeta_v;
  res[5]  = zeta_e;
  res[6]  = chis;
  res[7]  = chip;
  res[8]  = txx;
  res[9]  = tzz;
  res[10] = txz;
  res[11] = tII;
  res[12] = DP;
  res[13] = Y;

  PetscFunctionReturn(0);
}

// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "RheologyPointwise_AveragePhase"
PetscErrorCode RheologyPointwise_AveragePhase(PetscInt i, PetscInt j, PetscScalar ***xwt, PetscInt *iwt, PetscScalar phi, 
                                 PetscScalar *eps, PetscScalar *tauold, PetscScalar *P, PetscScalar *res, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       iph;
  PetscScalar    dt, em, nh, lambda;//, p;
  PetscScalar    eta_v, zeta_v, eta_e, zeta_e, eta, zeta, chip, chis, Y;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  dt = usr->par->dt;
  em = usr->par->etamin;
  nh = usr->par->nh;
  lambda = usr->par->lambda;

  // get epsII, epsp, epspII - tau_old, dP
  PetscScalar exx, ezz, exz, eII, told_xx, told_zz, told_xz, DPold; //told_II,
  exx = eps[0]; told_xx = tauold[0]; 
  ezz = eps[1]; told_zz = tauold[1];
  exz = eps[2]; told_xz = tauold[2];
  eII = eps[3]; //told_II = tauold[3];
  //p = P[0]; 
  DPold = P[1];

  PetscScalar eII_dev;
  // second invariant of deviatoric strain rate
  eII_dev = PetscPowScalar((PetscPowScalar(eII,2) - 1.0/6.0*PetscPowScalar(exx+ezz,2)),0.5);

  // get marker phase and properties
  PetscScalar    wt[2], meta_v[2], mzeta_v[2], meta_e[2], mzeta_e[2], mY[2], mYC[2];
  PetscScalar    inv_eta_p[2], inv_eta_vp[2], inv_zeta_p[2], inv_zeta_vp[2];
  PetscScalar    meta[2], mzeta[2], mchis[2], mchip[2];

  ierr = GetMatPhaseFraction(i,j,xwt,iwt,usr->nph,wt); CHKERRQ(ierr);

  for (iph = 0; iph < usr->nph; iph++) {
    meta_v[iph]  = usr->mat[iph].eta0*PetscExpScalar(-lambda*phi);
    mzeta_v[iph] = usr->mat[iph].zeta0*PetscExpScalar(-lambda*phi);
    meta_e[iph]  = usr->mat[iph].G*dt;
    mzeta_e[iph] = usr->mat[iph].Z0*dt;
    mY[iph]      = usr->mat[iph].C; // plastic yield criterion
    mYC[iph]     = usr->mat[iph].C*usr->par->lam_p; // compaction failure criteria
  
    // plastic viscosity
    if (usr->par->plasticity) { 
      inv_eta_p[iph]  = 2.0*eII_dev/mY[iph];
      inv_zeta_p[iph] = PetscAbs(exx+ezz)/mYC[iph]; //inv_zeta_p = inv_eta_p/lam_v;
    } else { 
      inv_eta_p[iph] = 0.0;
      inv_zeta_p[iph] = 0.0;
    }
    inv_eta_vp[iph]  = PetscPowScalar(PetscPowScalar(inv_eta_p[iph], nh) + PetscPowScalar(1.0/meta_v[iph], nh), 1.0/nh);
    inv_zeta_vp[iph] = PetscPowScalar(PetscPowScalar(inv_zeta_p[iph], nh) + PetscPowScalar(1.0/mzeta_v[iph], nh), 1.0/nh);

    // effective viscosities
    meta[iph]  = em + (1.0 - phi)/(inv_eta_vp[iph]  + 1.0/meta_e[iph]);
    mzeta[iph] = em + (1.0 - phi)/(inv_zeta_vp[iph] + 1.0/mzeta_e[iph]);

    // elastic stress evolution parameter - remove the cutoff minimum viscosity in calculating the built-up stress
    mchis[iph] = (meta[iph]-em)/meta_e[iph];
    mchip[iph] = (mzeta[iph]-em)/mzeta_e[iph];
  }

  eta_v  = WeightAverageValue(meta_v,wt,usr->nph); 
  zeta_v = WeightAverageValue(mzeta_v,wt,usr->nph); 
  eta_e  = WeightAverageValue(meta_e,wt,usr->nph); 
  zeta_e = WeightAverageValue(mzeta_e,wt,usr->nph); 
  eta    = WeightAverageValue(meta,wt,usr->nph); 
  zeta   = WeightAverageValue(mzeta,wt,usr->nph); 
  chis   = WeightAverageValue(mchis,wt,usr->nph); 
  chip   = WeightAverageValue(mchip,wt,usr->nph); 
  Y      = WeightAverageValue(mY,wt,usr->nph); 

  // shear and volumetric stresses
  PetscScalar phis, div, txx, tzz, txz, tII, DP;
  phis = 1 - phi;
  div = exx + ezz;
  txx = (2.0*(eta-em)*(exx-1.0/3.0*div) + chis*told_xx)/phis;
  tzz = (2.0*(eta-em)*(ezz-1.0/3.0*div) + chis*told_zz)/phis;
  txz = (2.0*(eta-em)*exz + chis*told_xz)/phis;
  tII = PetscPowScalar(0.5*(txx*txx + tzz*tzz + 2*txz*txz),0.5);
  DP  = (-(zeta-em)*div + chip*DPold)/phis;

  // return values
  res[0]  = eta;
  res[1]  = eta_v;
  res[2]  = eta_e;
  res[3]  = zeta;
  res[4]  = zeta_v;
  res[5]  = zeta_e;
  res[6]  = chis;
  res[7]  = chip;
  res[8]  = txx;
  res[9]  = tzz;
  res[10] = txz;
  res[11] = tII;
  res[12] = DP;
  res[13] = Y;

  PetscFunctionReturn(0);
}

// ---------------------------------------
// UpdateStrainRates
// ---------------------------------------
PetscErrorCode UpdateStrainRates(DM dm, Vec x, void *ctx)
{
  UsrData        *usr = (UsrData*) ctx;
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
        exxn[0] = exxc;
      }

      if (i==Nx-1) { // boundaries
        pointC.i = i; pointC.j = j; pointC.loc = ELEMENT; pointC.c = 0;
        ierr = DMStagGetPointStrainRates(dm,xlocal,1,&pointC,&epsIIc,&exxc,&ezzc,&exzc); CHKERRQ(ierr);
        ezzn[1] = ezzc;
        exxn[1] = exxc;
      }

      if (j==0) { // boundaries
        pointC.i = i; pointC.j = j; pointC.loc = ELEMENT; pointC.c = 0;
        ierr = DMStagGetPointStrainRates(dm,xlocal,1,&pointC,&epsIIc,&exxc,&ezzc,&exzc); CHKERRQ(ierr);
        exxn[0] = exxc;
        ezzn[0] = ezzc;
      }

      if (j==Nz-1) { // boundaries
        pointC.i = i; pointC.j = j; pointC.loc = ELEMENT; pointC.c = 0;
        ierr = DMStagGetPointStrainRates(dm,xlocal,1,&pointC,&epsIIc,&exxc,&ezzc,&exzc); CHKERRQ(ierr);
        exxn[2] = exxc;
        ezzn[2] = ezzc;
      }

      if ((i==Nx-1) && (j==Nz-1)) { // boundaries
        pointC.i = i; pointC.j = j; pointC.loc = ELEMENT; pointC.c = 0;
        ierr = DMStagGetPointStrainRates(dm,xlocal,1,&pointC,&epsIIc,&exxc,&ezzc,&exzc); CHKERRQ(ierr);
        exxn[3] = exxc;
        ezzn[3] = ezzc;
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
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       k,n_bc,*idx_bc;
  PetscScalar    *value_bc,*x_bc;
  PetscScalar    vi;
  BCType         *type_bc;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  vi = usr->par->vi;

  // LEFT Boundary - Vx
  ierr = DMStagBCListGetValues(bclist,'w','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // LEFT Boundary - Vz
  ierr = DMStagBCListGetValues(bclist,'w','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // RIGHT Boundary - Vx
  ierr = DMStagBCListGetValues(bclist,'e','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = -vi;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // RIGHT Boundary - Vz
  ierr = DMStagBCListGetValues(bclist,'e','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN Boundary - Vx
  ierr = DMStagBCListGetValues(bclist,'s','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN Boundary - Vz
  ierr = DMStagBCListGetValues(bclist,'s','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP Boundary - Vx
  ierr = DMStagBCListGetValues(bclist,'n','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP Boundary - Vz
  ierr = DMStagBCListGetValues(bclist,'n','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = vi;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  ierr = DMStagBCListGetValues(bclist,'n','o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  if (n_bc){
    value_bc[0] = 0.0;
    type_bc[0] = BC_DIRICHLET_STAG;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// SetSwarmInitialCondition
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "SetSwarmInitialCondition"
PetscErrorCode SetSwarmInitialCondition(DM dmswarm, void *ctx)
{
  UsrData   *usr = (UsrData*)ctx;
  PetscScalar *pcoor,*pfield,*pfield0, *pfield1, zblock_s, zblock_e, xis, xie, zis, zie;
  PetscInt  npoints,p;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  zblock_s = 0.0;
  zblock_e = 0.6*usr->par->H;
  xis = usr->par->xmin;
  xie = usr->par->xmin+0.1*usr->par->L;
  zis = usr->par->zmin;
  zie = usr->par->zmin+0.1*usr->par->H;

  ierr = DMSwarmGetLocalSize(dmswarm,&npoints);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dmswarm,DMSwarmPICField_coor,NULL,NULL,(void**)&pcoor);CHKERRQ(ierr);
    
  ierr = DMSwarmGetField(dmswarm,"id",NULL,NULL,(void**)&pfield);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dmswarm,"id0",NULL,NULL,(void**)&pfield0);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dmswarm,"id1",NULL,NULL,(void**)&pfield1);CHKERRQ(ierr);

  for (p=0; p<npoints; p++) {
    PetscScalar xcoor,zcoor;
    
    xcoor = pcoor[2*p+0];
    zcoor = pcoor[2*p+1];

    // dummy fields used for projection
    pfield0[p] = 0;
    pfield1[p] = 0;

    // default marker phase - block
    pfield[p] = 0; 

    // weak top/bottom layer
    if ((zcoor<zblock_s) || (zcoor>zblock_e)) pfield[p] = 1;
    if ((xcoor>xis) && (xcoor<xie) && (zcoor>zis) && (zcoor<zie)) pfield[p] = 1;

    // update binary representation
    if (pfield[p]==0) pfield0[p] = 1;
    if (pfield[p]==1) pfield1[p] = 1;
  }
  ierr = DMSwarmRestoreField(dmswarm,"id",NULL,NULL,(void**)&pfield);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(dmswarm,"id0",NULL,NULL,(void**)&pfield0);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(dmswarm,"id1",NULL,NULL,(void**)&pfield1);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(dmswarm,DMSwarmPICField_coor,NULL,NULL,(void**)&pcoor);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// Update marker phase fractions
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "UpdateMarkerPhaseFractions"
PetscErrorCode UpdateMarkerPhaseFractions(DM dmswarm, DM dmMPhase, Vec xMPhase, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  DM             dm;
  PetscInt       id;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  dm = usr->dmPV;
  // Project swarm into coefficient
  id = 0;
  ierr = MPoint_ProjectQ1_arith_general_AP(dmswarm,"id0",dm,dmMPhase,0,id,xMPhase);CHKERRQ(ierr);//vertex
  ierr = MPoint_ProjectQ1_arith_general_AP(dmswarm,"id0",dm,dmMPhase,1,id,xMPhase);CHKERRQ(ierr);//face
  ierr = MPoint_ProjectQ1_arith_general_AP(dmswarm,"id0",dm,dmMPhase,2,id,xMPhase);CHKERRQ(ierr);//cell

  id = 1;
  ierr = MPoint_ProjectQ1_arith_general_AP(dmswarm,"id1",dm,dmMPhase,0,id,xMPhase);CHKERRQ(ierr);//vertex
  ierr = MPoint_ProjectQ1_arith_general_AP(dmswarm,"id1",dm,dmMPhase,1,id,xMPhase);CHKERRQ(ierr);//face
  ierr = MPoint_ProjectQ1_arith_general_AP(dmswarm,"id1",dm,dmMPhase,2,id,xMPhase);CHKERRQ(ierr);//cell

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

  usr->dmPV = NULL;
  usr->dmswarm = NULL;
  usr->dmMPhase = NULL;
  usr->dmP = NULL;
  usr->dmeps = NULL;
  
  // usr->xPV  = NULL;
  // usr->xphi = NULL;
  usr->xMPhase = NULL;
  usr->xeps = NULL;
  usr->xtau = NULL;
  usr->xtau_old = NULL;
  usr->xDP = NULL;
  usr->xDP_old = NULL;
  usr->xplast = NULL;

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
  ierr = PetscBagRegisterInt(bag, &par->ppcell, 4, "ppcell", "Number of particles/cell one-dir"); CHKERRQ(ierr);

  ierr = PetscBagRegisterScalar(bag, &par->L, 0.5, "L", "Length of domain in x-dir"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->H, 0.5, "H", "Height of domain in z-dir"); CHKERRQ(ierr);

  // Physical and material parameters
  ierr = PetscBagRegisterScalar(bag, &par->F, 0.0, "F", "Non-dimensional gravity terms, positve/negative means the direction of gravity is the negative/positive direction of z;"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->R, 1.0, "R", "Ratio of the compaction length scale to the global one, R = ((K0*eta_ref/mu)^1/2)/L"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->phi_0, 0.01, "phi_0", "Reference porosity"); CHKERRQ(ierr);
  // ierr = PetscBagRegisterScalar(bag, &par->n, 2.0, "n", "Porosity exponent"); CHKERRQ(ierr);
  // ierr = PetscBagRegisterScalar(bag, &par->lambda, 0.0, "lambda", "Exponential melt weakening factor"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->vi, 1.0, "vi", "Extension/compression velocity"); CHKERRQ(ierr);
  // Viscosity
  ierr = PetscBagRegisterScalar(bag, &par->eb_v0, 1.0e3, "eb_v0", "Block shear viscosity"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->ew_v0, 1.0e-1, "ew_v0", "Weak zone shear viscosity"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->lam_v, 1.0e-1, "lam_v", "Factors for intrinsic viscosity, lam_v = eta/zeta"); CHKERRQ(ierr);
  // Plasticity
  ierr = PetscBagRegisterScalar(bag, &par->C_b, 20.0, "C_b", "Block cohesion"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->C_w, 1e40, "C_w", "Weak zone cohesion"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->lam_p, 1.0, "lam_p", "Multiplier for the compaction failure criteria, YC = lam_p*C"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->etamin, 1e-3, "etamin", "Cutoff min value of eta"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->nh, 1.0, "nh", "Power for the harmonic plasticity"); CHKERRQ(ierr);
  // Elasticity
  ierr = PetscBagRegisterScalar(bag, &par->G, 1.0, "G", "Shear elastic modulus"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->Z0, 1.0, "Z0", "Reference poro-elastic modulus"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->q, -0.5, "q", "Exponent of the porosity-dependent relation of poro-elastic modulus"); CHKERRQ(ierr);

  // Time steps
  ierr = PetscBagRegisterScalar(bag, &par->dt, 0.1, "dt", "The size of time step"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->tmax, 1.0, "tmax", "The maximum value of t"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->tstep, 1, "tstep", "The maximum time steps"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->tout,1, "tout", "Output every tout time step"); CHKERRQ(ierr);

  ierr = PetscBagRegisterScalar(bag, &par->P0,0, "P0", "Pinned value for pressure"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->tf_tol, 1e-8, "tf_tol", "Function tolerance for solving yielding stresses"); CHKERRQ(ierr);

  // Reference compaction viscosity
  par->zb_v0 = par->eb_v0/par->lam_v;
  par->zw_v0 = par->ew_v0/par->lam_v;
  
  par->plasticity = PETSC_TRUE;
  ierr = PetscBagRegisterInt(bag, &par->rheology,0, "rheology", "0-DominantPhase VEP, 1-PhaseAverage VEP"); CHKERRQ(ierr);
  
  // Input/output 
  ierr = PetscBagRegisterString(bag,&par->fname_out,FNAME_LENGTH,"out_solution","output_file","Name for output file, set with: -output_file <filename>"); CHKERRQ(ierr);
  ierr = PetscBagRegisterString(bag,&par->fdir_out,FNAME_LENGTH,"./","output_dir","Name for output directory, set with: -output_dir <dirname>"); CHKERRQ(ierr);

  // marker phase properties
  usr->nph = 2;
  usr->mat[0].eta0 = par->eb_v0;
  usr->mat[1].eta0 = par->ew_v0;

  usr->mat[0].zeta0 = par->zb_v0;
  usr->mat[1].zeta0 = par->zw_v0;

  usr->mat[0].C = par->C_b;
  usr->mat[1].C = par->C_w;

  usr->mat[0].G = par->G;
  usr->mat[1].G = par->G;

  usr->mat[0].Z0 = par->Z0;
  usr->mat[1].Z0 = par->Z0;

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
  PetscPrintf(usr->comm,"# Test_stokesdarcy2field_vep_inclusion_pic (with particle-in-cell method): %s \n",&(date[0]));
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
PetscErrorCode Get9PointCenterValues(PetscInt i, PetscInt j, PetscInt idx, PetscInt Nx, PetscInt Nz, PetscScalar ***xx, PetscScalar *x)
{
  PetscInt  im, jm, ip, jp;
  PetscFunctionBegin;
    // get porosity, p, Plith, T - center
    if (i == 0   ) im = i; else im = i-1;
    if (i == Nx-1) ip = i; else ip = i+1;
    if (j == 0   ) jm = j; else jm = j-1;
    if (j == Nz-1) jp = j; else jp = j+1;

    x[0] = xx[j ][i ][idx]; // i  ,j   - C
    x[1] = xx[j ][im][idx]; // i-1,j   - L
    x[2] = xx[j ][ip][idx]; // i+1,j   - R
    x[3] = xx[jm][i ][idx]; // i  ,j-1 - D
    x[4] = xx[jp][i ][idx]; // i  ,j+1 - U
    x[5] = xx[jm][im][idx]; // i-1,j-1 - LD
    x[6] = xx[jm][ip][idx]; // i+1,j-1 - RD
    x[7] = xx[jp][im][idx]; // i-1,j+1 - LU
    x[8] = xx[jp][ip][idx]; // i+1,j+1 - RU
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

  // Print user parameters
  ierr = InputPrintData(usr); CHKERRQ(ierr);

  // Numerical solution using the FD pde object
  ierr = StokesDarcy_Numerical_PIC(usr); CHKERRQ(ierr);

  // Destroy objects
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
