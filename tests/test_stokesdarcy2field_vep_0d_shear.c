// ---------------------------------------
// Uniform shear stress of a visco-elasto-plastic block subject to the Stokes model
// Rheology: visco-elasto-plastic model
// run: ./test_stokesdarcy2field_vep_0d_shear.sh -pc_factor_mat_solver_type umfpack -pc_type lu -nx 5 -nz 5 -tstep 10 -tmax 1 -log_view
// python test: ./python/test_stokesdarcy2field_vep_0d_shear.py
// ---------------------------------------
static char help[] = "Application for validating the visco-elasto-plastic model with zero-dimensional problems about uniform deformation (simple shear) in the absence of gravity \n\n";

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

#include "../src/fdpde_stokesdarcy2field.h"

// ---------------------------------------
// Application Context
// ---------------------------------------
#define FNAME_LENGTH 200

// parameters (bag)
typedef struct {
  PetscInt       nx, nz, scaling, bulk_eff;
  PetscScalar    L, H;
  PetscScalar    xmin, zmin, xmax, zmax;
  PetscScalar    eta_v0, vi, C, P0, zeta_v0, etamin;
  PetscScalar    lambda,lam_p,G,Z0,q,F,R,phi_0,n;
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
"  CASE 1 -- \n"
"  LEFT/UP: no penetration, free slip, \n"
"  RIGHT: influx, Vx=-Vi   free slip dVz/dx=0 \n"
"  BOTTOM: outflux, Vz=-Vi, free slip dVx/dz = 0\n"
"  Zero normal graident for liquid pressure at all four boundaries. \n";

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
  PetscFunctionBeginUser;

  // Element count
  nx = usr->par->nx;
  nz = usr->par->nz;

  // Domain coords
  xmin = usr->par->xmin;
  xmax = usr->par->xmin + usr->par->L;
  zmin = usr->par->zmin;
  zmax = usr->par->zmin + usr->par->H;


  // Create the FD-pde object
  PetscCall(FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_STOKESDARCY2FIELD,&fd));
  PetscCall(FDPDESetUp(fd));
  PetscCall(FDPDEGetDM(fd,&dm)); 

  // Create DM/vec for strain rates, yield, deviatoric/volumetric stress, and stress_old
  PetscCall(DMStagCreateCompatibleDMStag(dm,4,0,4,0,&usr->dmeps)); 
  PetscCall(DMSetUp(usr->dmeps)); 
  PetscCall(DMStagSetUniformCoordinatesProduct(usr->dmeps,xmin,xmax,zmin,zmax,0.0,0.0));
  PetscCall(DMCreateGlobalVector(usr->dmeps,&usr->xeps)); 
  PetscCall(DMCreateGlobalVector(usr->dmeps,&usr->xyield)); 
  PetscCall(DMCreateGlobalVector(usr->dmeps,&usr->xtau)); 
  PetscCall(DMCreateGlobalVector(usr->dmeps,&usr->xDP)); 
  PetscCall(DMCreateGlobalVector(usr->dmeps,&usr->xtau_old)); 
  PetscCall(DMCreateGlobalVector(usr->dmeps,&usr->xDP_old)); 

  // Create DM/vec for tau_old and DP_old, Initialise the two Vecs as zeros.
  PetscCall(VecZeroEntries(usr->xtau_old)); 
  PetscCall(VecZeroEntries(usr->xDP_old)); 

  // Set coefficients and BC evaluation function
  PetscCall(FDPDESetFunctionBCList(fd,FormBCList,bc_description,usr)); 
  PetscCall(FDPDESetFunctionCoefficient(fd,FormCoefficient,coeff_description,usr)); 
  PetscCall(FDPDEView(fd)); 

  {  
  // Create initial guess with a linear viscous
  usr->par->plasticity = PETSC_FALSE; 
  PetscPrintf(PETSC_COMM_WORLD,"\n# INITIAL GUESS #\n");
  PetscCall(FDPDESolve(fd,NULL));
  PetscCall(FDPDEGetSolution(fd,&x)); 

  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s_solution_initial",usr->par->fdir_out,usr->par->fname_out));
  PetscCall(DMStagViewBinaryPython(dm,x,fout));

  PetscCall(FDPDEGetSolutionGuess(fd,&xguess));  
  PetscCall(VecCopy(x,xguess));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&xguess));
  }

  usr->par->plasticity = PETSC_TRUE; 
  
  // FD SNES Solver
  PetscPrintf(PETSC_COMM_WORLD,"\n# SNES SOLVE #\n");
  
  // Time loop  
  while ((usr->par->t < usr->par->tmax) && (istep<usr->par->tstep)) {
    PetscPrintf(PETSC_COMM_WORLD,"# TIMESTEP %d: \n",istep);

    //StokesDarcy Solver
    PetscCall(FDPDESolve(fd,NULL));
    PetscCall(FDPDEGetSolution(fd,&x));
    
    // Update time
    usr->par->t += usr->par->dt;  // computation start from t = dt
    PetscPrintf(PETSC_COMM_WORLD,"# TIME: time = %1.12e dt = %1.12e \n",usr->par->t,usr->par->dt);

    // Update xtau_old and xDP_old
    PetscCall(UpdateStressOld(usr->dmeps,usr)); 
       
    // Output solution
    if (istep % usr->par->tout == 0 ) {
     
      // Output solution to file
      PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s_solution_ts%1.3d",usr->par->fdir_out,usr->par->fname_out,istep));
      PetscCall(DMStagViewBinaryPython(dm,x,fout));

      PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s_strain_ts%1.3d",usr->par->fdir_out,usr->par->fname_out,istep));
      PetscCall(DMStagViewBinaryPython(usr->dmeps,usr->xeps,fout));

      PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s_stress_ts%1.3d",usr->par->fdir_out,usr->par->fname_out,istep));
      PetscCall(DMStagViewBinaryPython(usr->dmeps,usr->xtau,fout));

      PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s_stressold_ts%1.3d",usr->par->fdir_out,usr->par->fname_out,istep));
      PetscCall(DMStagViewBinaryPython(usr->dmeps,usr->xtau_old,fout));

      PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s_dpold_ts%1.3d",usr->par->fdir_out,usr->par->fname_out,istep));
      PetscCall(DMStagViewBinaryPython(usr->dmeps,usr->xDP_old,fout));

      PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s_yield_ts%1.3d",usr->par->fdir_out,usr->par->fname_out,istep));
      PetscCall(DMStagViewBinaryPython(usr->dmeps,usr->xyield,fout));

      PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s_residual_ts%1.3d",usr->par->fdir_out,usr->par->fname_out,istep));
      PetscCall(DMStagViewBinaryPython(dm,fd->r,fout));

      PetscCall(FDPDEGetCoefficient(fd,&dmcoeff,&xcoeff));
      PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s_coefficient_ts%1.3d",usr->par->fdir_out,usr->par->fname_out,istep));
      PetscCall(DMStagViewBinaryPython(dmcoeff,xcoeff,fout));
    }

    //clean up
    PetscCall(VecDestroy(&x));

    // increment timestep
    istep++;
  }

  // Destroy objects
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&usr->xeps));
  PetscCall(VecDestroy(&usr->xtau));
  PetscCall(VecDestroy(&usr->xDP));
  PetscCall(VecDestroy(&usr->xyield));
  PetscCall(VecDestroy(&usr->xtau_old));
  PetscCall(VecDestroy(&usr->xDP_old));
  PetscCall(DMDestroy(&dm));
  PetscCall(DMDestroy(&usr->dmeps));
  PetscCall(FDPDEDestroy(&fd));

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

  PetscCall(PetscBagRegisterScalar(bag, &par->L, 1.0, "L", "Length of domain in x-dir")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->H, 1.0, "H", "Height of domain in z-dir")); 

  // Physical and material parameters
  PetscCall(PetscBagRegisterScalar(bag, &par->F, 0.0, "F", "Non-dimensional gravity terms, positve/negative means the direction of gravity is the negative/positive direction of z;")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->R, 0.0, "R", "Ratio of the compaction length scale to the global one, R = ((K0*eta_ref/mu)^1/2)/L")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->phi_0, 1e-4, "phi_0", "Reference porosity")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->n, 2.0, "n", "Porosity exponent")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->lambda, 0.0, "lambda", "Exponential melt weakening factor")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->vi, 1.0, "vi", "Extension/compression velocity")); 
  // Viscosity
  PetscCall(PetscBagRegisterScalar(bag, &par->eta_v0, 1.0, "eta_v0", "Reference shear viscosity")); 
  // Plasticity
  PetscCall(PetscBagRegisterScalar(bag, &par->C, 1e40, "C", "Yield Creteria")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->lam_p, 1.0, "lam_p", "Multiplier to control the compaction plasticity, zeta_p = eta_p/(lam_p*phi)")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->etamin, 0.0, "etamin", "Cutoff min value of eta")); 
  // Elasticity
  PetscCall(PetscBagRegisterScalar(bag, &par->G, 1.0, "G", "Shear elastic modulus")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->Z0, 1.0, "Z0", "Reference poro-elastic modulus")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->q, -0.5, "q", "Exponent of the porosity-dependent relation of poro-elastic modulus")); 

  // Time steps
  PetscCall(PetscBagRegisterScalar(bag, &par->dt, 0.1, "dt", "The size of time step")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->tmax, 1.0, "tmax", "The maximum value of t")); 
  PetscCall(PetscBagRegisterInt(bag, &par->tstep, 10, "tstep", "The maximum time steps")); 
  PetscCall(PetscBagRegisterInt(bag, &par->tout,1, "tout", "Output every tout time step")); 

  PetscCall(PetscBagRegisterScalar(bag, &par->P0,0, "P0", "Pinned value for pressure")); 


  // Reference compaction viscosity
  par->zeta_v0 = par->eta_v0/par->phi_0;
  
  par->plasticity = PETSC_TRUE;

  // Input/output 
  PetscCall(PetscBagRegisterString(bag,&par->fname_out,FNAME_LENGTH,"out_solution","output_file","Name for output file, set with: -output_file <filename>")); 
  PetscCall(PetscBagRegisterString(bag,&par->fdir_out,FNAME_LENGTH,"./","output_dir","Name for output directory, set with: -output_dir <dirname>")); 

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
  PetscPrintf(usr->comm,"# Test_stokesdarcy2field_vep_0d: %s \n",&(date[0]));
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
  PetscScalar    lambda,R;//,n;
  PetscScalar    ***c, ***xxs, ***xxy, ***xxp;
  PetscScalar    dt,F,G,Z0,Z,q,eta_v0,Y,lam_p,phi,Kphi,eta_v,zeta_v,eta_e,zeta_e;
  PetscFunctionBeginUser;

  dt = usr->par->dt;

  lambda = usr->par->lambda;
  R = usr->par->R;
  // n = usr->par->n;
  F = usr->par->F;
  G = usr->par->G;
  Z0 = usr->par->Z0;
  q = usr->par->q;
  eta_v0 = usr->par->eta_v0;
  Y = usr->par->C;
  lam_p = usr->par->lam_p;

  // Uniform porosity and permeability
  phi = usr->par->phi_0;
  Kphi = 1.0;

  // Shear and compaction viscosity - uniform porosity
  eta_v = eta_v0*PetscExpScalar(-lambda*phi);
  zeta_v = eta_v/phi;

  // Effective shear and Compaction viscosity due to elasticity
  eta_e = G*dt;
  Z = Z0 * PetscPowScalar(phi, -q); // for uniform porosity only
  zeta_e = Z*dt;

  // Strain rates
  PetscCall(UpdateStrainRates(dm,x,usr)); 
  PetscCall(DMGetLocalVector(usr->dmeps, &xepslocal)); 
  PetscCall(DMGlobalToLocal (usr->dmeps, usr->xeps, INSERT_VALUES, xepslocal)); 

  // Stress_old
  PetscCall(DMGetLocalVector(usr->dmeps, &toldlocal)); 
  PetscCall(DMGlobalToLocal (usr->dmeps, usr->xtau_old, INSERT_VALUES, toldlocal)); 
  PetscCall(DMGetLocalVector(usr->dmeps, &poldlocal)); 
  PetscCall(DMGlobalToLocal (usr->dmeps, usr->xDP_old, INSERT_VALUES, poldlocal)); 

  // Local vectors
  PetscCall(DMCreateLocalVector (usr->dmeps,&xtaulocal)); 
  PetscCall(DMStagVecGetArray(usr->dmeps,xtaulocal,&xxs)); 

  PetscCall(DMCreateLocalVector (usr->dmeps,&xDPlocal)); 
  PetscCall(DMStagVecGetArray(usr->dmeps,xDPlocal,&xxp)); 

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

  // Get the cell sizes
  PetscScalar *dx, *dz;
  PetscCall(DMStagCellSizeLocal_2d(dm, &nx, &nz, &dx, &dz)); 
  
  // Loop over local domain - set initial density and viscosity
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      PetscInt idx;
      PetscScalar told_xx_e,told_zz_e,told_xz_e,told_II_e,pold,chis,chip,zeta;
      PetscScalar cs[4],told_xx[4],told_zz[4],told_xz[4],told_II[4];

      { // A = eta (center, c=1), and also compute chi_s and chi_p (center, c = 4,5)
        DMStagStencil point;
        PetscScalar   epsII,exx,ezz,exz,txx,tzz,txz,tauII,epsII_dev;
        PetscScalar   div,div2;
        PetscScalar   etaP_inv, eta, zetaP_inv, DeltaP;

        // second invariant of strain rate
        point.i = i; point.j = j; point.loc = ELEMENT; 
        point.c = 0;
        PetscCall(DMStagVecGetValuesStencil(usr->dmeps,xepslocal,1,&point,&exx)); 
        PetscCall(DMStagVecGetValuesStencil(usr->dmeps,toldlocal,1,&point,&told_xx_e)); 
        PetscCall(DMStagVecGetValuesStencil(usr->dmeps,poldlocal,1,&point,&pold)); 
        point.c = 1;
        PetscCall(DMStagVecGetValuesStencil(usr->dmeps,xepslocal,1,&point,&ezz)); 
        PetscCall(DMStagVecGetValuesStencil(usr->dmeps,toldlocal,1,&point,&told_zz_e)); 
        point.c = 2;
        PetscCall(DMStagVecGetValuesStencil(usr->dmeps,xepslocal,1,&point,&exz)); 
        PetscCall(DMStagVecGetValuesStencil(usr->dmeps,toldlocal,1,&point,&told_xz_e)); 
        point.c = 3;
        PetscCall(DMStagVecGetValuesStencil(usr->dmeps,xepslocal,1,&point,&epsII)); 
        PetscCall(DMStagVecGetValuesStencil(usr->dmeps,toldlocal,1,&point,&told_II_e)); 

        // second invariant of deviatoric strain rate
        epsII_dev = PetscPowScalar((PetscPowScalar(epsII,2) - 1.0/6.0*PetscPowScalar(exx+ezz,2)),0.5);

        // effective shear viscosity
        if (usr->par->plasticity) { etaP_inv = 2.0*epsII_dev/Y;}
        else { etaP_inv = 0.0;}
        eta = usr->par->etamin + (1.0 - phi)/(etaP_inv + 1.0/eta_v + 1.0/eta_e);
        // effective bulk viscosity
        zetaP_inv = etaP_inv * phi * lam_p;
        zeta = usr->par->etamin + (1.0-phi)/(zetaP_inv + 1.0/zeta_v + 1.0/zeta_e);

        // elastic stress evolution parameter
        chis = eta/(G*dt);
        chip = zeta/(Z*dt);
        
        point.i = i; point.j = j; point.loc = ELEMENT; point.c = 1;

        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
        c[j][i][idx] = eta;

        // deviatoric stress and its second invariant
        div = exx + ezz;
        div2= PetscPowScalar(div,2);
        txx = 2.0*eta*(exx-1.0/3.0*div) + chis*told_xx_e;
        tzz = 2.0*eta*(ezz-1.0/3.0*div) + chis*told_zz_e;
        txz = 2.0*eta*exz + chis*told_xz_e;
        tauII = PetscPowScalar(0.5*(txx*txx + tzz*tzz + 2*txz*txz + div2),0.5);
        
        // volumetric stress
        DeltaP = -(zeta - 2.0/3.0*eta)*div+chip*pold;

        // save stresses for output
        PetscCall(DMStagGetLocationSlot(usr->dmeps,ELEMENT,0,&idx));  xxs[j][i][idx] = txx; xxy[j][i][idx] = Y; xxp[j][i][idx] = DeltaP;
        PetscCall(DMStagGetLocationSlot(usr->dmeps,ELEMENT,1,&idx));  xxs[j][i][idx] = tzz;
        PetscCall(DMStagGetLocationSlot(usr->dmeps,ELEMENT,2,&idx));  xxs[j][i][idx] = txz;
        PetscCall(DMStagGetLocationSlot(usr->dmeps,ELEMENT,3,&idx));  xxs[j][i][idx] = tauII;
      }

      { // A = eta (corner, c=0)
        // also compute cs = eta/(G*dt) (four corners, c =1)
        DMStagStencil point[4];
        PetscScalar   etaP_inv, zetaP_inv, eta;//, xp[4],zp[4];
        PetscScalar   epsII[4],exx[4],ezz[4],exz[4],txx[4],tzz[4],txz[4],tauII[4],epsII_dev[4];
        PetscScalar   div,div2;
        PetscInt      ii;

        // second invariant of strain rate
        point[0].i = i; point[0].j = j; point[0].loc = DOWN_LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = DOWN_RIGHT; point[1].c = 0;
        point[2].i = i; point[2].j = j; point[2].loc = UP_LEFT;    point[2].c = 0;
        point[3].i = i; point[3].j = j; point[3].loc = UP_RIGHT;   point[3].c = 0;
        
        PetscCall(DMStagVecGetValuesStencil(usr->dmeps,xepslocal,4,point,exx)); 
        PetscCall(DMStagVecGetValuesStencil(usr->dmeps,toldlocal,4,point,told_xx)); 
                
        for (ii = 0; ii < 4; ii++) {point[ii].c = 1;}
        PetscCall(DMStagVecGetValuesStencil(usr->dmeps,xepslocal,4,point,ezz)); 
        PetscCall(DMStagVecGetValuesStencil(usr->dmeps,toldlocal,4,point,told_zz)); 

        for (ii = 0; ii < 4; ii++) {point[ii].c = 2;}
        PetscCall(DMStagVecGetValuesStencil(usr->dmeps,xepslocal,4,point,exz)); 
        PetscCall(DMStagVecGetValuesStencil(usr->dmeps,toldlocal,4,point,told_xz)); 

        // second invariant of strain rate
        for (ii = 0; ii < 4; ii++) {point[ii].c = 3;}
        PetscCall(DMStagVecGetValuesStencil(usr->dmeps,xepslocal,4,point,epsII)); 
        PetscCall(DMStagVecGetValuesStencil(usr->dmeps,toldlocal,4,point,told_II)); 

        // second invariant of deviatoric strain rate
        for (ii = 0; ii < 4; ii++) {
          epsII_dev[ii] = PetscPowScalar((PetscPowScalar(epsII[ii],2) - 1.0/6.0*PetscPowScalar(exx[ii]+ezz[ii],2)),0.5);
        }

        // // coordinates
        // xp[0] = coordx[i][iprev]; zp[0] = coordz[j][iprev]; 
        // xp[1] = coordx[i][inext]; zp[1] = coordz[j][iprev]; 
        // xp[2] = coordx[i][iprev]; zp[2] = coordz[j][inext]; 
        // xp[3] = coordx[i][inext]; zp[3] = coordz[j][inext];

        for (ii = 0; ii < 4; ii++) {

          // effective shear viscosity
          if (usr->par->plasticity) { etaP_inv = 2.0*epsII_dev[ii]/Y;}
          else { etaP_inv = 0.0;}
          eta = usr->par->etamin + (1.0-phi)/(etaP_inv + 1.0/eta_v + 1.0/eta_e); 
          // effective bulk viscosity
          zetaP_inv = etaP_inv * phi * lam_p;
          zeta = usr->par->etamin + (1.0-phi)/(zetaP_inv + 1.0/zeta_v + 1.0/zeta_e);

          // elastic stress evolution parameter
          cs[ii] = eta/(G*dt);

          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, 0, &idx)); 
          c[j][i][idx] = eta;

          // deviatoric stress and its second invariant
          
          div = exx[ii] + ezz[ii];
          div2= PetscPowScalar(div,2);
          txx[ii] = 2.0*eta*(exx[ii]-1.0/3.0*div) + cs[ii]*told_xx[ii];
          tzz[ii] = 2.0*eta*(ezz[ii]-1.0/3.0*div) + cs[ii]*told_zz[ii];
          txz[ii] = 2.0*eta*exz[ii] + cs[ii]*told_xz[ii];
          tauII[ii] = PetscPowScalar(0.5*(txx[ii]*txx[ii] + tzz[ii]*tzz[ii] + 2.0*txz[ii]*txz[ii] + div2),0.5);
          
        }

        // save stresses
        PetscCall(DMStagGetLocationSlot(usr->dmeps,DOWN_LEFT,0,&idx));  xxs[j][i][idx] = txx[0]; xxy[j][i][idx] = Y;
        PetscCall(DMStagGetLocationSlot(usr->dmeps,DOWN_LEFT,1,&idx));  xxs[j][i][idx] = tzz[0];
        PetscCall(DMStagGetLocationSlot(usr->dmeps,DOWN_LEFT,2,&idx));  xxs[j][i][idx] = txz[0];
        PetscCall(DMStagGetLocationSlot(usr->dmeps,DOWN_LEFT,3,&idx));  xxs[j][i][idx] = tauII[0];

        PetscCall(DMStagGetLocationSlot(usr->dmeps,DOWN_RIGHT,0,&idx));  xxs[j][i][idx] = txx[1]; xxy[j][i][idx] = Y;
        PetscCall(DMStagGetLocationSlot(usr->dmeps,DOWN_RIGHT,1,&idx));  xxs[j][i][idx] = tzz[1];
        PetscCall(DMStagGetLocationSlot(usr->dmeps,DOWN_RIGHT,2,&idx));  xxs[j][i][idx] = txz[1];
        PetscCall(DMStagGetLocationSlot(usr->dmeps,DOWN_RIGHT,3,&idx));  xxs[j][i][idx] = tauII[1];

        PetscCall(DMStagGetLocationSlot(usr->dmeps,UP_LEFT,0,&idx));  xxs[j][i][idx] = txx[2]; xxy[j][i][idx] = Y;
        PetscCall(DMStagGetLocationSlot(usr->dmeps,UP_LEFT,1,&idx));  xxs[j][i][idx] = tzz[2];
        PetscCall(DMStagGetLocationSlot(usr->dmeps,UP_LEFT,2,&idx));  xxs[j][i][idx] = txz[2];
        PetscCall(DMStagGetLocationSlot(usr->dmeps,UP_LEFT,3,&idx));  xxs[j][i][idx] = tauII[2];

        PetscCall(DMStagGetLocationSlot(usr->dmeps,UP_RIGHT,0,&idx));  xxs[j][i][idx] = txx[3]; xxy[j][i][idx] = Y;
        PetscCall(DMStagGetLocationSlot(usr->dmeps,UP_RIGHT,1,&idx));  xxs[j][i][idx] = tzz[3];
        PetscCall(DMStagGetLocationSlot(usr->dmeps,UP_RIGHT,2,&idx));  xxs[j][i][idx] = txz[3];
        PetscCall(DMStagGetLocationSlot(usr->dmeps,UP_RIGHT,3,&idx));  xxs[j][i][idx] = tauII[3];
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
          rhs[0]+= 2.0*chip * pold/dx[ii];
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
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] += rhs[ii];
        }
      }

      { // C = 0 (center, c=0)
        DMStagStencil point;
        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 0;
        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
        c[j][i][idx] = 0.0;
      }

      { // D1 = zeta - 2/3*A (center, c=2)
        DMStagStencil point;
        PetscInt      idxA;
        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 2;
        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 1;
        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idxA)); 
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
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
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
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] = 0.0;
        }
        //  nonzero if including the gravity
        for (ii = 2; ii < 4; ii++) {
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] = -pow(R,2) * Kphi * F;
        }
      }
    }
  }

  // release dx dz
  PetscCall(PetscFree(dx));
  PetscCall(PetscFree(dz));  

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

  PetscCall(DMStagVecRestoreArray(usr->dmeps,xDPlocal,&xxp)); 
  PetscCall(DMLocalToGlobalBegin(usr->dmeps,xDPlocal,INSERT_VALUES,usr->xDP)); 
  PetscCall(DMLocalToGlobalEnd  (usr->dmeps,xDPlocal,INSERT_VALUES,usr->xDP)); 
  PetscCall(VecDestroy(&xDPlocal)); 
  
  PetscCall(DMStagVecRestoreArray(usr->dmeps,xyieldlocal,&xxy)); 
  PetscCall(DMLocalToGlobalBegin(usr->dmeps,xyieldlocal,INSERT_VALUES,usr->xyield)); 
  PetscCall(DMLocalToGlobalEnd  (usr->dmeps,xyieldlocal,INSERT_VALUES,usr->xyield)); 
  PetscCall(VecDestroy(&xyieldlocal)); 

  PetscCall(DMRestoreLocalVector(usr->dmeps,&xepslocal)); 
  PetscCall(DMRestoreLocalVector(usr->dmeps,&toldlocal)); 
  PetscCall(DMRestoreLocalVector(usr->dmeps,&poldlocal)); 

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
        exxn[0] = exxc;
      }

      if (i==Nx-1) { // boundaries
        pointC.i = i; pointC.j = j; pointC.loc = ELEMENT; pointC.c = 0;
        PetscCall(DMStagGetPointStrainRates(dm,xlocal,1,&pointC,&epsIIc,&exxc,&ezzc,&exzc)); 
        ezzn[1] = ezzc;
        exxn[1] = exxc;
      }

      if (j==0) { // boundaries
        pointC.i = i; pointC.j = j; pointC.loc = ELEMENT; pointC.c = 0;
        PetscCall(DMStagGetPointStrainRates(dm,xlocal,1,&pointC,&epsIIc,&exxc,&ezzc,&exzc)); 
        exxn[0] = exxc;
        ezzn[0] = ezzc;
      }

      if (j==Nz-1) { // boundaries
        pointC.i = i; pointC.j = j; pointC.loc = ELEMENT; pointC.c = 0;
        PetscCall(DMStagGetPointStrainRates(dm,xlocal,1,&pointC,&epsIIc,&exxc,&ezzc,&exzc)); 
        exxn[2] = exxc;
        ezzn[2] = ezzc;
      }

      if ((i==Nx-1) && (j==Nz-1)) { // boundaries
        pointC.i = i; pointC.j = j; pointC.loc = ELEMENT; pointC.c = 0;
        PetscCall(DMStagGetPointStrainRates(dm,xlocal,1,&pointC,&epsIIc,&exxc,&ezzc,&exzc)); 
        exxn[3] = exxc;
        ezzn[3] = ezzc;
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
  PetscFunctionBeginUser;

  vi = usr->par->vi;

  // LEFT Boundary - Vx
  PetscCall(DMStagBCListGetValues(bclist,'w','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // LEFT Boundary - Vz
  PetscCall(DMStagBCListGetValues(bclist,'w','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // LEFT Boundary - P
  /*
  PetscCall(DMStagBCListGetValues(bclist,'w','o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET_STAG;//BC_NEUMANN;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  */
  
  // RIGHT Boundary - Vx
  PetscCall(DMStagBCListGetValues(bclist,'e','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = -vi;
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // RIGHT Boundary - Vz
  PetscCall(DMStagBCListGetValues(bclist,'e','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // RIGHT Boundary - P
  /*
  PetscCall(DMStagBCListGetValues(bclist,'e','o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET_STAG;//BC_NEUMANN;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  */
    
  // DOWN Boundary - Vx
  PetscCall(DMStagBCListGetValues(bclist,'s','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // DOWN Boundary - Vz
  PetscCall(DMStagBCListGetValues(bclist,'s','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = -vi;
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // DOWN Boundary - P
  /*
  PetscCall(DMStagBCListGetValues(bclist,'s','o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET_STAG;//BC_NEUMANN;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  */
  
  // UP Boundary - Vx
  PetscCall(DMStagBCListGetValues(bclist,'n','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // UP Boundary - Vz
  PetscCall(DMStagBCListGetValues(bclist,'n','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // UP Boundary - P
  /*
  PetscCall(DMStagBCListGetValues(bclist,'n','o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<1; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;//BC_NEUMANN;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  */
  // pin pressure dof
  PetscCall(DMStagBCListPinCornerValue(bclist,DMSTAG_DOWN_LEFT,'o',0,usr->par->P0)); 
  //PetscCall(DMStagBCListPinCornerValue(bclist,DMSTAG_UP_LEFT,'o',0,usr->par->P0)); 

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// UpdateStressOld
// ---------------------------------------
PetscErrorCode UpdateStressOld(DM dmeps, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  Vec            xtaulocal, xDPlocal, tauold_local,DPold_local;
  PetscFunctionBeginUser;

  // Get local vectors from dmeps
  PetscCall(DMGetLocalVector(usr->dmeps, &xtaulocal)); 
  PetscCall(DMGlobalToLocal (usr->dmeps, usr->xtau, INSERT_VALUES, xtaulocal)); 

  PetscCall(DMGetLocalVector(usr->dmeps, &xDPlocal)); 
  PetscCall(DMGlobalToLocal (usr->dmeps, usr->xDP, INSERT_VALUES, xDPlocal)); 

  // Create local vectors for the stress_old terms
  PetscCall(DMCreateLocalVector (usr->dmeps,&tauold_local)); 
  PetscCall(DMCreateLocalVector (usr->dmeps,&DPold_local)); 
  PetscCall(VecCopy(xtaulocal, tauold_local)); 
  PetscCall(VecCopy(xDPlocal, DPold_local)); 

  // create array from tauold_local and add the rotation terms
  // No rotation terms for now

  // Restore and map local to global
  PetscCall(DMLocalToGlobalBegin(usr->dmeps,tauold_local,INSERT_VALUES,usr->xtau_old)); 
  PetscCall(DMLocalToGlobalEnd  (usr->dmeps,tauold_local,INSERT_VALUES,usr->xtau_old)); 
  PetscCall(VecDestroy(&tauold_local)); 

  PetscCall(DMLocalToGlobalBegin(usr->dmeps,DPold_local,INSERT_VALUES,usr->xDP_old)); 
  PetscCall(DMLocalToGlobalEnd  (usr->dmeps,DPold_local,INSERT_VALUES,usr->xDP_old)); 
  PetscCall(VecDestroy(&DPold_local)); 

  PetscCall(DMRestoreLocalVector(usr->dmeps,&xtaulocal)); 
  PetscCall(DMRestoreLocalVector(usr->dmeps,&xDPlocal)); 

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

  // Start time
  PetscCall(PetscTime(&start_time)); 
 
  // Load command line or input file if required
  PetscCall(PetscOptionsInsert(PETSC_NULLPTR,&argc,&argv,NULL)); 

  // Input user parameters and print
  PetscCall(InputParameters(&usr)); 

  // Print user parameters
  PetscCall(InputPrintData(usr)); 

  // Numerical solution using the FD pde object
  PetscCall(StokesDarcy_Numerical(usr)); 

  // Destroy objects
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
