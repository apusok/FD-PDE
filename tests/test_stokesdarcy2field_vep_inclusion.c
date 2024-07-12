// ---------------------------------------
// Shortening a two-phase block which is governed by the StokesDarcy model
// Rheology: visco-elasto-plastic model
// run: ./tests/test_stokesdarcy2field_vep_inclusion.app -nx 100 -nz 100 -pc_type lu -pc_factor_mat_solver_type umfpack
// python test: ./tests/python/test_stokesdarcy2field_vep_inclusion.py
// ---------------------------------------
static char help[] = "Application for shortening of a visco-elasto-plastic two-phase block in the absence of gravity \n\n";

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

// ---------------------------------------
// Application Context
// ---------------------------------------
#define FNAME_LENGTH 200

// parameters (bag)
typedef struct {
  PetscInt       nx, nz, scaling, bulk_eff;
  PetscScalar    L, H;
  PetscScalar    xmin, zmin, xmax, zmax;
  PetscScalar    vi, P0, etamin; //eta_v0, vi, C, P0, zeta_v0, etamin;
  PetscScalar    eb_v0, ew_v0, lam_v, C_b, C_w, zb_v0, zw_v0;
  PetscScalar    lambda,lam_p,G,Z0,q,F,R,phi_0,n,nh;
  PetscBool      plasticity;
  PetscInt       tstep, tout;
  PetscScalar    t, dt, tmax, dtmax;
  char           fname_out[FNAME_LENGTH];
  char           fname_in[FNAME_LENGTH];
  char           fdir_out[FNAME_LENGTH];
} Params;

// user defined and model-dependent variables
typedef struct {
  Params        *par;
  PetscBag       bag;
  MPI_Comm       comm;
  PetscMPIInt    rank;
  DM             dmeps;
  Vec            xeps, xtau, xyield, xDP, xtau_old, xDP_old;
} UsrData;

// ---------------------------------------
// Function definitions
// ---------------------------------------
PetscErrorCode StokesDarcy_Numerical(void*);
PetscErrorCode InputParameters(UsrData**);
PetscErrorCode InputPrintData(UsrData*);
PetscErrorCode FormCoefficient(FDPDE, DM, Vec, DM, Vec, void*);
PetscErrorCode FormBCList(DM, Vec, DMStagBCList, void*);
PetscErrorCode UpdateStrainRates(DM,Vec,void*);
PetscErrorCode UpdateStressOld(DM,void*);


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

const char bc_description[] =
"  << Stokes-Darcy BCs >> \n"
"  LEFT/BOTTOM: Symmetric boundary conditions (zero normal velocity, free slip) \n"
"  UP: extension Vz = Vi, free slip dVx/dz=0 \n"
"  RIGHT: compression Vx=Vi, free slip dVz/dx = 0\n";

// ---------------------------------------
// Application functions
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "StokesDarcy_Numerical"
PetscErrorCode StokesDarcy_Numerical(void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  FDPDE          fd;
  DM             dm, dmcoeff;
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
  ierr = FDPDEGetDM(fd,&dm); CHKERRQ(ierr);

  // Create DM/vec for strain rates, yield, deviatoric/volumetric stress, and stress_old
  ierr = DMStagCreateCompatibleDMStag(dm,4,0,4,0,&usr->dmeps); CHKERRQ(ierr);
  ierr = DMSetUp(usr->dmeps); CHKERRQ(ierr);
  ierr = DMStagSetUniformCoordinatesProduct(usr->dmeps,xmin,xmax,zmin,zmax,0.0,0.0);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(usr->dmeps,&usr->xeps); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(usr->dmeps,&usr->xyield); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(usr->dmeps,&usr->xtau); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(usr->dmeps,&usr->xDP); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(usr->dmeps,&usr->xtau_old); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(usr->dmeps,&usr->xDP_old); CHKERRQ(ierr);

  // Create DM/vec for tau_old and DP_old, Initialise the two Vecs as zeros.
  ierr = VecZeroEntries(usr->xtau_old); CHKERRQ(ierr);
  ierr = VecZeroEntries(usr->xDP_old); CHKERRQ(ierr);

  // Set coefficients and BC evaluation function
  ierr = FDPDESetFunctionBCList(fd,FormBCList,bc_description,usr); CHKERRQ(ierr);
  ierr = FDPDESetFunctionCoefficient(fd,FormCoefficient,coeff_description,usr); CHKERRQ(ierr);
  ierr = FDPDEView(fd); CHKERRQ(ierr);

  {  
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
  }

  usr->par->plasticity = PETSC_TRUE; 
  
  // FD SNES Solver
  PetscPrintf(PETSC_COMM_WORLD,"\n# SNES SOLVE #\n");
  
  // Time loop  
  while ((usr->par->t < usr->par->tmax) && (istep<usr->par->tstep)) {
    PetscPrintf(PETSC_COMM_WORLD,"# TIMESTEP %d: \n",istep);

    //StokesDarcy Solver
    ierr = FDPDESolve(fd,NULL);CHKERRQ(ierr);
    ierr = FDPDEGetSolution(fd,&x);CHKERRQ(ierr);
    
    // Update time
    usr->par->t += usr->par->dt;  // computation start from t = dt
    PetscPrintf(PETSC_COMM_WORLD,"# TIME: time = %1.12e dt = %1.12e \n",usr->par->t,usr->par->dt);

    // Update xtau_old and xDP_old
    ierr = UpdateStressOld(usr->dmeps,usr);CHKERRQ(ierr); 
       
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
      ierr = DMStagViewBinaryPython(usr->dmeps,usr->xDP_old,fout);CHKERRQ(ierr);

      ierr = PetscSNPrintf(fout,sizeof(fout),"%s/%s_yield_ts%1.3d",usr->par->fdir_out,usr->par->fname_out,istep);
      ierr = DMStagViewBinaryPython(usr->dmeps,usr->xyield,fout);CHKERRQ(ierr);

      ierr = PetscSNPrintf(fout,sizeof(fout),"%s/%s_residual_ts%1.3d",usr->par->fdir_out,usr->par->fname_out,istep);
      ierr = DMStagViewBinaryPython(dm,fd->r,fout);CHKERRQ(ierr);

      ierr = FDPDEGetCoefficient(fd,&dmcoeff,&xcoeff);CHKERRQ(ierr);
      ierr = PetscSNPrintf(fout,sizeof(fout),"%s/%s_coefficient_ts%1.3d",usr->par->fdir_out,usr->par->fname_out,istep);
      ierr = DMStagViewBinaryPython(dmcoeff,xcoeff,fout);CHKERRQ(ierr);

    }

    //clean up
    ierr = VecDestroy(&x);CHKERRQ(ierr);

    // increment timestep
    istep++;

  }

  // Destroy objects
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xeps);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xtau);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xDP);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xyield);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xtau_old);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xDP_old);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = DMDestroy(&usr->dmeps);CHKERRQ(ierr);
  ierr = FDPDEDestroy(&fd);CHKERRQ(ierr);

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

  ierr = PetscBagRegisterScalar(bag, &par->L, 0.5, "L", "Length of domain in x-dir"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->H, 0.5, "H", "Height of domain in z-dir"); CHKERRQ(ierr);

  // Physical and material parameters
  ierr = PetscBagRegisterScalar(bag, &par->F, 0.0, "F", "Non-dimensional gravity terms, positve/negative means the direction of gravity is the negative/positive direction of z;"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->R, 1.0, "R", "Ratio of the compaction length scale to the global one, R = ((K0*eta_ref/mu)^1/2)/L"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->phi_0, 0.01, "phi_0", "Reference porosity"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->n, 2.0, "n", "Porosity exponent"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->lambda, 0.0, "lambda", "Exponential melt weakening factor"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->vi, 1.0, "vi", "Extension/compression velocity"); CHKERRQ(ierr);
  // Viscosity
  ierr = PetscBagRegisterScalar(bag, &par->eb_v0, 1.0e3, "eb_v0", "Block shear viscosity"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->ew_v0, 1.0e-1, "ew_v0", "Weak zone shear viscosity"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->lam_v, 1.0e-1, "lam_v", "Factors for intrinsic visocisty, lam_v = eta/zeta"); CHKERRQ(ierr);
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
  ierr = PetscBagRegisterInt(bag, &par->tstep, 10, "tstep", "The maximum time steps"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->tout,1, "tout", "Output every tout time step"); CHKERRQ(ierr);

  ierr = PetscBagRegisterScalar(bag, &par->P0,0, "P0", "Pinned value for pressure"); CHKERRQ(ierr);


  // Reference compaction viscosity
  par->zb_v0 = par->eb_v0/par->lam_v;
  par->zw_v0 = par->ew_v0/par->lam_v;
  
  par->plasticity = PETSC_TRUE;

  // Input/output 
  ierr = PetscBagRegisterString(bag,&par->fname_out,FNAME_LENGTH,"out_solution","output_file","Name for output file, set with: -output_file <filename>"); CHKERRQ(ierr);
  ierr = PetscBagRegisterString(bag,&par->fdir_out,FNAME_LENGTH,"./","output_dir","Name for output directory, set with: -output_dir <dirname>"); CHKERRQ(ierr);

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
  PetscPrintf(usr->comm,"# Test_stokesdarcy2field_vep_0d: %s \n",&(date[0]));
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
// FormCoefficient
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormCoefficient"
PetscErrorCode FormCoefficient(FDPDE fd, DM dm, Vec x, DM dmcoeff, Vec coeff, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz;
  Vec            coefflocal, xepslocal, xtaulocal, xDPlocal, xyieldlocal, toldlocal, poldlocal;
  PetscScalar    **coordx,**coordz;
  PetscInt       iprev, inext, icenter;
  PetscScalar    lambda,R;//,q,n;
  PetscScalar    ***c, ***xxs, ***xxy, ***xxp;
  PetscScalar    dt,F,G,Z0,Z,lam_p,phi,Kphi,eta_v,zeta_v,eta_e,zeta_e,nh;
  PetscScalar    eb_v0, ew_v0,lam_v,eb_v, ew_v, zb_v, zw_v, em;
  PetscScalar    zblock_s, zblock_e, xis, xie, zis, zie;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  dt = usr->par->dt;

  lambda = usr->par->lambda;
  R = usr->par->R;
  // n = usr->par->n;
  F = usr->par->F;
  G = usr->par->G;
  Z0 = usr->par->Z0;
  // q = usr->par->q;
  lam_p = usr->par->lam_p;

  eb_v0 = usr->par->eb_v0;
  ew_v0 = usr->par->ew_v0;
  lam_v = usr->par->lam_v;
  em    = usr->par->etamin;
  nh    = usr->par->nh;

  // Uniform porosity and permeability
  phi = usr->par->phi_0;
  Kphi = 1.0;

  // Shear and compaction viscosity - uniform porosity
  eb_v = eb_v0*PetscExpScalar(-lambda*phi);
  zb_v = eb_v/lam_v; //eb_v/phi;
  ew_v = ew_v0*PetscExpScalar(-lambda*phi);
  zw_v = ew_v/lam_v; //ew_v/phi;

  // Effective shear and Compaction viscosity due to elasticity
  eta_e = G*dt;
  Z = Z0; 
  zeta_e = Z*dt;

  // block and weak inclusion params
  zblock_s = 0.0;
  zblock_e = 0.6*usr->par->H;
  xis = usr->par->xmin;
  xie = usr->par->xmin+0.1*usr->par->L;
  zis = usr->par->zmin;
  zie = usr->par->zmin+0.1*usr->par->H;

  // Strain rates
  ierr = UpdateStrainRates(dm,x,usr); CHKERRQ(ierr);
  ierr = DMGetLocalVector(usr->dmeps, &xepslocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmeps, usr->xeps, INSERT_VALUES, xepslocal); CHKERRQ(ierr);

  
  // Stress_old
  ierr = DMGetLocalVector(usr->dmeps, &toldlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmeps, usr->xtau_old, INSERT_VALUES, toldlocal); CHKERRQ(ierr);
  ierr = DMGetLocalVector(usr->dmeps, &poldlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmeps, usr->xDP_old, INSERT_VALUES, poldlocal); CHKERRQ(ierr);

  // Local vectors
  ierr = DMCreateLocalVector (usr->dmeps,&xtaulocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(usr->dmeps,xtaulocal,&xxs); CHKERRQ(ierr);

  ierr = DMCreateLocalVector (usr->dmeps,&xDPlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(usr->dmeps,xDPlocal,&xxp); CHKERRQ(ierr);

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

  // Get the cell sizes
  PetscScalar *dx, *dz;
  ierr = DMStagCellSizeLocal_2d(dm, &nx, &nz, &dx, &dz); CHKERRQ(ierr);
  
  // Loop over local domain - set initial density and viscosity
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      PetscInt idx;
      PetscScalar told_xx_e,told_zz_e,told_xz_e,told_II_e,pold,chis,chip,zeta;
      PetscScalar cs[4],told_xx[4],told_zz[4],told_xz[4],told_II[4];

      { // A = eta (center, c=1), and also compute chi_s and chi_p (center, c = 4,5)
        DMStagStencil point;
        PetscScalar   epsII,exx,ezz,exz,txx,tzz,txz,tauII,epsII_dev,Y,YC;
        PetscScalar   div;//,div2;
        PetscScalar   etaP_inv, eta, zetaP_inv, DeltaP;

        if ((coordz[j][icenter]<zblock_s) || (coordz[j][icenter]>zblock_e)) { 
          eta_v = ew_v; // top/bottom layer
          Y   = usr->par->C_w; // plastic yield criterion
          zeta_v = zw_v; // compaction viscosity
        } else if ((coordx[i][icenter]>xis) && (coordx[i][icenter]<xie) && (coordz[j][icenter]>zis) && (coordz[j][icenter]<zie)) {
          eta_v = ew_v; // inclusion
          Y   = usr->par->C_w; // plastic yield criterion
          zeta_v = zw_v; // compaction viscosity
        } else {
          eta_v = eb_v; // block
          Y = usr->par->C_b; // plastic yield criterion
          zeta_v = zb_v; // compaction viscosity
        }

        // compaction failure criteria
        YC = Y*lam_p;

        // second invariant of strain rate
        point.i = i; point.j = j; point.loc = ELEMENT; 
        point.c = 0;
        ierr = DMStagVecGetValuesStencil(usr->dmeps,xepslocal,1,&point,&exx); CHKERRQ(ierr);
        ierr = DMStagVecGetValuesStencil(usr->dmeps,toldlocal,1,&point,&told_xx_e); CHKERRQ(ierr);
        ierr = DMStagVecGetValuesStencil(usr->dmeps,poldlocal,1,&point,&pold); CHKERRQ(ierr);
        point.c = 1;
        ierr = DMStagVecGetValuesStencil(usr->dmeps,xepslocal,1,&point,&ezz); CHKERRQ(ierr);
        ierr = DMStagVecGetValuesStencil(usr->dmeps,toldlocal,1,&point,&told_zz_e); CHKERRQ(ierr);
        point.c = 2;
        ierr = DMStagVecGetValuesStencil(usr->dmeps,xepslocal,1,&point,&exz); CHKERRQ(ierr);
        ierr = DMStagVecGetValuesStencil(usr->dmeps,toldlocal,1,&point,&told_xz_e); CHKERRQ(ierr);
        point.c = 3;
        ierr = DMStagVecGetValuesStencil(usr->dmeps,xepslocal,1,&point,&epsII); CHKERRQ(ierr);
        ierr = DMStagVecGetValuesStencil(usr->dmeps,toldlocal,1,&point,&told_II_e); CHKERRQ(ierr);

        // second invariant of deviatoric strain rate
        epsII_dev = PetscPowScalar((PetscPowScalar(epsII,2) - 1.0/6.0*PetscPowScalar(exx+ezz,2)),0.5);

        PetscScalar vp_inv;
        // effective shear viscosity
        if (usr->par->plasticity) { etaP_inv = 2.0*epsII_dev/Y;}
        else { etaP_inv = 0.0;}

        vp_inv = PetscPowScalar(etaP_inv, nh) + PetscPowScalar(1.0/eta_v, nh);
        vp_inv = PetscPowScalar(vp_inv, 1.0/nh);
        eta = em + (1.0 - phi)/(vp_inv + 1.0/eta_e);
        
        // effective bulk viscosity
        //zetaP_inv = etaP_inv/lam_v;
        zetaP_inv = PetscAbs(exx+ezz)/YC;
        vp_inv = PetscPowScalar(zetaP_inv, nh) + PetscPowScalar(1.0/zeta_v, nh);
        vp_inv = PetscPowScalar(vp_inv, 1.0/nh);
        zeta = em + (1.0 - phi)/(vp_inv + 1.0/zeta_e);
        
        // elastic stress evolution parameter
        // remove the cutoff minimum viscosity in calculating the built-up stress
        chis = (eta-em)/(G*dt);
        chip = (zeta-em)/(Z*dt);
        
        point.i = i; point.j = j; point.loc = ELEMENT; point.c = 1;

        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = eta;
        
        // deviatoric stress and its second invariant
        // 1. remove the minimum etamin while calculating the built-up stress
        // 2. devide 1 - phi to fix the reduced effective stress due to porosity
        PetscScalar phis;
      
        phis = 1 - phi;
        div = exx + ezz;
        txx = (2.0*(eta-em)*(exx-1.0/3.0*div) + chis*told_xx_e)/phis;
        tzz = (2.0*(eta-em)*(ezz-1.0/3.0*div) + chis*told_zz_e)/phis;
        txz = (2.0*(eta-em)*exz + chis*told_xz_e)/phis;
        tauII = PetscPowScalar(0.5*(txx*txx + tzz*tzz + 2*txz*txz),0.5);
              
        
        // volumetric stress
        DeltaP = (-(zeta-em)*div+chip*pold)/phis;

        // save stresses for output
        ierr = DMStagGetLocationSlot(usr->dmeps,ELEMENT,0,&idx); CHKERRQ(ierr); xxs[j][i][idx] = txx; xxy[j][i][idx] = Y; xxp[j][i][idx] = DeltaP;
        ierr = DMStagGetLocationSlot(usr->dmeps,ELEMENT,1,&idx); CHKERRQ(ierr); xxs[j][i][idx] = tzz;
        ierr = DMStagGetLocationSlot(usr->dmeps,ELEMENT,2,&idx); CHKERRQ(ierr); xxs[j][i][idx] = txz;
        ierr = DMStagGetLocationSlot(usr->dmeps,ELEMENT,3,&idx); CHKERRQ(ierr); xxs[j][i][idx] = tauII;
      }

      { // A = eta (corner, c=0)
        // also compute cs = eta/(G*dt) (four corners, c =1)
        DMStagStencil point[4];
        PetscScalar   etaP_inv, eta, xp[4],zp[4],Y[4];
        PetscScalar   epsII[4],exx[4],ezz[4],exz[4],txx[4],tzz[4],txz[4],tauII[4],epsII_dev[4];
        PetscScalar   div;//,div2;
        PetscInt      ii;

        // second invariant of strain rate
        point[0].i = i; point[0].j = j; point[0].loc = DOWN_LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = DOWN_RIGHT; point[1].c = 0;
        point[2].i = i; point[2].j = j; point[2].loc = UP_LEFT;    point[2].c = 0;
        point[3].i = i; point[3].j = j; point[3].loc = UP_RIGHT;   point[3].c = 0;
        
        ierr = DMStagVecGetValuesStencil(usr->dmeps,xepslocal,4,point,exx); CHKERRQ(ierr);
        ierr = DMStagVecGetValuesStencil(usr->dmeps,toldlocal,4,point,told_xx); CHKERRQ(ierr);
                
        for (ii = 0; ii < 4; ii++) {point[ii].c = 1;}
        ierr = DMStagVecGetValuesStencil(usr->dmeps,xepslocal,4,point,ezz); CHKERRQ(ierr);
        ierr = DMStagVecGetValuesStencil(usr->dmeps,toldlocal,4,point,told_zz); CHKERRQ(ierr);

        for (ii = 0; ii < 4; ii++) {point[ii].c = 2;}
        ierr = DMStagVecGetValuesStencil(usr->dmeps,xepslocal,4,point,exz); CHKERRQ(ierr);
        ierr = DMStagVecGetValuesStencil(usr->dmeps,toldlocal,4,point,told_xz); CHKERRQ(ierr);

        // second invariant of strain rate
        for (ii = 0; ii < 4; ii++) {point[ii].c = 3;}
        ierr = DMStagVecGetValuesStencil(usr->dmeps,xepslocal,4,point,epsII); CHKERRQ(ierr);
        ierr = DMStagVecGetValuesStencil(usr->dmeps,toldlocal,4,point,told_II); CHKERRQ(ierr);

        // second invariant of deviatoric strain rate
        for (ii = 0; ii < 4; ii++) {
          epsII_dev[ii] = PetscPowScalar((PetscPowScalar(epsII[ii],2) - 1.0/6.0*PetscPowScalar(exx[ii]+ezz[ii],2)),0.5);
        }

        // coordinates
        xp[0] = coordx[i][iprev]; zp[0] = coordz[j][iprev]; 
        xp[1] = coordx[i][inext]; zp[1] = coordz[j][iprev]; 
        xp[2] = coordx[i][iprev]; zp[2] = coordz[j][inext]; 
        xp[3] = coordx[i][inext]; zp[3] = coordz[j][inext];

        for (ii = 0; ii < 4; ii++) {

          if ((zp[ii]<zblock_s) || (zp[ii]>zblock_e)) { 
            eta_v = ew_v; // top/bottom layer
            Y[ii] = usr->par->C_w; // plastic yield criterion
            zeta_v = zw_v; // compaction viscosity
          } else if ((xp[ii]>xis) && (xp[ii]<xie) && (zp[ii]>zis) && (zp[ii]<zie)) {
            eta_v = ew_v; // inclusion
            Y[ii] = usr->par->C_w; // plastic yield criterion
            zeta_v = zw_v; // compaction viscosity
          } else {
            eta_v = eb_v; //block
            Y[ii] = usr->par->C_b; // plastic yield criterion
            zeta_v = zb_v; // compaction viscosity
          }

          PetscScalar vp_inv;
          // effective shear viscosity
          if (usr->par->plasticity) { etaP_inv = 2.0*epsII_dev[ii]/Y[ii];}
          else { etaP_inv = 0.0;}

          vp_inv = PetscPowScalar(etaP_inv, nh) + PetscPowScalar(1.0/eta_v, nh);
          vp_inv = PetscPowScalar(vp_inv, 1.0/nh);
          eta = em + (1.0 - phi)/(vp_inv + 1.0/eta_e);

          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, 0, &idx); CHKERRQ(ierr);
          c[j][i][idx] = eta;

          // elastic stress evolution parameter
          cs[ii] = (eta-em)/(G*dt);
          
          // deviatoric stress and its second invariant
          PetscScalar phis;

          phis = 1 - phi;
          div = exx[ii] + ezz[ii];

          txx[ii] = (2.0*(eta-em)*(exx[ii]-1.0/3.0*div) + cs[ii]*told_xx[ii])/phis;
          tzz[ii] = (2.0*(eta-em)*(ezz[ii]-1.0/3.0*div) + cs[ii]*told_zz[ii])/phis;
          txz[ii] = (2.0*(eta-em)*exz[ii] + cs[ii]*told_xz[ii])/phis;
          tauII[ii] = PetscPowScalar(0.5*(txx[ii]*txx[ii] + tzz[ii]*tzz[ii] + 2.0*txz[ii]*txz[ii]),0.5);
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

      { // B = phi*F*ek - grad(chi_s*tau_old) + div(chi_p*dP_old) (edges, c=0)
        // (F = (rho^s-rho^f)*U*L/eta_ref * g*L/U^2) 
        DMStagStencil point[4];
        PetscScalar   rhs[4];
        PetscInt      ii,jj;

        ii = i - sx;
        jj = j - sz;

        // d(chi_s*tau_xy_old)/dz compute on the left and right
        // d(chi_s*tau_xy_old)/dx and gravity compute on the down boundary only
        // RHS = 0 on the true boundaries
        if (i > 0) {
          rhs[0] = -chis*told_xx_e/dx[ii];
          rhs[0]+= -(cs[2]*told_xz[2]-cs[0]*told_xz[0])/dz[jj];
          rhs[0]+= chip * pold/dx[ii];
        } else {
          rhs[0] = 0.0;
        }
        if (i < Nx-1) {
          rhs[1] = chis*told_xx_e/dx[ii];
          rhs[1]+= -chip*pold/dx[ii];
        } else {
          rhs[1] = 0.0;
        }
        if (j > 0) {
          rhs[2] = phi*F;
          rhs[2]+= -chis*told_zz_e/dz[jj];
          rhs[2]+= -(cs[1]*told_xz[1]-cs[0]*told_xz[0])/dx[ii];
          rhs[2]+= chip*pold/dz[jj];
        } else {
          rhs[2] = 0.0;
        }
        if (j < Nz-1) {
          rhs[3] = phi*F;
          rhs[3]+= chis*told_zz_e/dz[jj];
          rhs[3]+= -chip*pold/dz[jj];
        } else {
          rhs[3] = 0.0;
        }

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 0;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = 0;
        point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = 0;
        
        for (ii = 0; ii < 4; ii++) {
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] += rhs[ii];
        }
      }

      { // C = 0 (center, c=0)
        DMStagStencil point;
        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 0;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = 0.0;
      }

      { // D1 = zeta - 2/3*A (center, c=2)
        DMStagStencil point;
        PetscInt      idxA;
        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 2;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 1;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idxA); CHKERRQ(ierr);
        c[j][i][idx] = zeta - 2.0/3.0*c[j][i][idxA] ;
      }

      { // D2 = -R^2 * Kphi (edges, c=1)
        DMStagStencil point[4];
        // PetscScalar   xp[4],zp[4];
        PetscInt      ii;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 1;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 1;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = 1;
        point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = 1;

        // xp[0] = coordx[i][iprev  ]; zp[0] = coordz[j][icenter];
        // xp[1] = coordx[i][inext  ]; zp[1] = coordz[j][icenter];
        // xp[2] = coordx[i][icenter]; zp[2] = coordz[j][iprev  ];
        // xp[3] = coordx[i][icenter]; zp[3] = coordz[j][inext  ];

        for (ii = 0; ii < 4; ii++) {
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = -pow(R,2) * Kphi;
        }
      }

      { // D3 = R^2 * Kphi * F (edges, c=2)
        DMStagStencil point[4];
        // PetscScalar   xp[4],zp[4];
        PetscInt      ii;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 2;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 2;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = 2;
        point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = 2;

        // xp[0] = coordx[i][iprev  ]; zp[0] = coordz[j][icenter];
        // xp[1] = coordx[i][inext  ]; zp[1] = coordz[j][icenter];
        // xp[2] = coordx[i][icenter]; zp[2] = coordz[j][iprev  ];
        // xp[3] = coordx[i][icenter]; zp[3] = coordz[j][inext  ];

        for (ii = 0; ii < 2; ii++) {
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = 0.0;
        }
        //  nonzero if including the gravity
        for (ii = 2; ii < 4; ii++) {
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = -pow(R,2) * Kphi * F;
        }
      }
    }
  }

  // release dx dz
  ierr = PetscFree(dx);CHKERRQ(ierr);
  ierr = PetscFree(dz);CHKERRQ(ierr);  

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

  ierr = DMStagVecRestoreArray(usr->dmeps,xDPlocal,&xxp); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(usr->dmeps,xDPlocal,INSERT_VALUES,usr->xDP); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (usr->dmeps,xDPlocal,INSERT_VALUES,usr->xDP); CHKERRQ(ierr);
  ierr = VecDestroy(&xDPlocal); CHKERRQ(ierr);
  
  ierr = DMStagVecRestoreArray(usr->dmeps,xyieldlocal,&xxy); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(usr->dmeps,xyieldlocal,INSERT_VALUES,usr->xyield); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (usr->dmeps,xyieldlocal,INSERT_VALUES,usr->xyield); CHKERRQ(ierr);
  ierr = VecDestroy(&xyieldlocal); CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(usr->dmeps,&xepslocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmeps,&toldlocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmeps,&poldlocal); CHKERRQ(ierr);

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

  // LEFT Boundary - P
  /*
  ierr = DMStagBCListGetValues(bclist,'w','o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET_STAG;//BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  */
  
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

  // RIGHT Boundary - P
  /*
  ierr = DMStagBCListGetValues(bclist,'e','o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET_STAG;//BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  */
    
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

  // DOWN Boundary - P
  /*
  ierr = DMStagBCListGetValues(bclist,'s','o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET_STAG;//BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  */
  
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

  // UP Boundary - P
  //ierr = DMStagBCListGetValues(bclist,'n','o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  //for (k=0; k<n_bc; k++) {
  //value_bc[k] = 0.0;
  //  type_bc[k] = BC_DIRICHLET;//BC_DIRICHLET_STAG;//BC_NEUMANN;
  //}
  //ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  ierr = DMStagBCListGetValues(bclist,'n','o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<1; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET_STAG;//BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
 
  
  PetscFunctionReturn(0);
}



// ---------------------------------------
// UpdateStressOld
// ---------------------------------------
PetscErrorCode UpdateStressOld(DM dmeps, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  Vec            xtaulocal, xDPlocal, tauold_local,DPold_local;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  // Get local vectors from dmeps
  ierr = DMGetLocalVector(usr->dmeps, &xtaulocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmeps, usr->xtau, INSERT_VALUES, xtaulocal); CHKERRQ(ierr);

  ierr = DMGetLocalVector(usr->dmeps, &xDPlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmeps, usr->xDP, INSERT_VALUES, xDPlocal); CHKERRQ(ierr);

  // Create local vectors for the stress_old terms
  ierr = DMCreateLocalVector (usr->dmeps,&tauold_local); CHKERRQ(ierr);
  ierr = DMCreateLocalVector (usr->dmeps,&DPold_local); CHKERRQ(ierr);
  ierr = VecCopy(xtaulocal, tauold_local); CHKERRQ(ierr);
  ierr = VecCopy(xDPlocal, DPold_local); CHKERRQ(ierr);

  // create array from tauold_local and add the rotation terms
  // No rotation terms for now

  // Restore and map local to global
  ierr = DMLocalToGlobalBegin(usr->dmeps,tauold_local,INSERT_VALUES,usr->xtau_old); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (usr->dmeps,tauold_local,INSERT_VALUES,usr->xtau_old); CHKERRQ(ierr);
  ierr = VecDestroy(&tauold_local); CHKERRQ(ierr);

  ierr = DMLocalToGlobalBegin(usr->dmeps,DPold_local,INSERT_VALUES,usr->xDP_old); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (usr->dmeps,DPold_local,INSERT_VALUES,usr->xDP_old); CHKERRQ(ierr);
  ierr = VecDestroy(&DPold_local); CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(usr->dmeps,&xtaulocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmeps,&xDPlocal); CHKERRQ(ierr);

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
  ierr = StokesDarcy_Numerical(usr); CHKERRQ(ierr);

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
