#include "MORbuoyancy.h"

// ---------------------------------------
// UserParamsCreate
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "UserParamsCreate"
PetscErrorCode UserParamsCreate(UsrData **_usr,int argc,char **argv)
{
  UsrData       *usr;
  PetscInt      i;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  // default parameter values and command line input
  ierr = InputParameters(&usr); CHKERRQ(ierr);

  // file input
  for (int i = 1; i < argc; i++) {
    PetscBool flg;
    
    ierr = PetscStrcmp(argv[i],"-options_file",&flg); CHKERRQ(ierr);
    if (flg) { ierr = PetscStrcpy(usr->par->fname_in, argv[i+1]); CHKERRQ(ierr); }
  }

  // print user parameters
  ierr = InputPrintData(usr); CHKERRQ(ierr);

  // scaling parameters
  ierr = DefineScalingParameters(usr); CHKERRQ(ierr);

  // non-dimensionalize parameters
  ierr = NondimensionalizeParameters(usr); CHKERRQ(ierr);

  // return pointer
  *_usr = usr;

  PetscFunctionReturn(0);
}

// ---------------------------------------
// UserParamsDestroy
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "UserParamsDestroy"
PetscErrorCode UserParamsDestroy(UsrData *usr)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = PetscFree(usr->nd); CHKERRQ(ierr);
  ierr = PetscFree(usr->scal); CHKERRQ(ierr);
  ierr = PetscBagDestroy(&usr->bag); CHKERRQ(ierr);
  ierr = PetscFree(usr); CHKERRQ(ierr);
  
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
  PetscScalar    dsol, Teta0;
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

  // domain parameters
  ierr = PetscBagRegisterInt(bag, &par->nx, 20, "nx", "Element count in the x-dir [-]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->nz, 10, "nz", "Element count in the z-dir [-]"); CHKERRQ(ierr);

  ierr = PetscBagRegisterScalar(bag, &par->xmin, 0.0, "xmin", "Start coordinate of domain in x-dir [m]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->zmin, -100.0e3, "zmin", "Start coordinate of domain in z-dir [m]"); CHKERRQ(ierr);

  ierr = PetscBagRegisterScalar(bag, &par->L, 200.0e3, "L", "Length of domain in x-dir [m]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->H, 100.0e3, "H", "Height of domain in z-dir [m]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->xMOR, 6.0e3, "xMOR", "Distance from mid-ocean ridge axis for melt extraction [m]"); CHKERRQ(ierr);

  // physical and material parameters
  ierr = PetscBagRegisterScalar(bag, &par->k_hat, -1.0, "k_hat", "Direction of unit vertical vector [-]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->g, 9.8, "g", "Gravitational acceleration [m^2/s]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->U0, 4.0, "U0", "Half-spreading rate [cm/yr]"); CHKERRQ(ierr);

  ierr = PetscBagRegisterScalar(bag, &par->Tp, 1648, "Tp", "Potential temperature of mantle [K]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->cp, 1200, "cp", "Specific heat of matrix and magma [J/kg/K]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->La, 4.0e5, "La", "Latent fusion of heat [J/kg]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->kappa, 1.0e-6, "kappa", "Thermal diffusivity [m^2/s]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->D, 1.0e-8, "D", "Chemical diffusivity [m^2/s]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->rho0, 3000, "rho0", "Reference density of matrix and magma [kg/m^3]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->drho, 500, "drho", "Reference density difference [kg/m^3]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->alpha, 3.0e-5, "alpha", "Coefficient of thermal expansion [1/K]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->beta, 0.0, "beta", "Coefficient of compositional expansion [1/wt. frac.]"); CHKERRQ(ierr);
  // ierr = PetscBagRegisterInt(bag, &par->buoyancy, 0, "buoyancy", "Level of matrix buoyancy incorporation: 0=off, 1=phi, 2=phi-C, 3=phi-C-T"); CHKERRQ(ierr);

  // phase diagram
  ierr = PetscBagRegisterScalar(bag, &par->C0, 0.85, "C0", "Reference composition [wt. frac.]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->DC, 0.1, "DC", "Compositional diff between solidus and liquidus at T0 [wt.  frac.]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->T0, 1565, "T0", "Solidus temperature at P=0, C=C0 [K]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->Ms, 400, "Ms", "Slope of solidus dTdC [K]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->Mf, 400, "Ms", "Slope of liquidus dTdC [K]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->gamma_inv, 60, "gamma_inv", "Inverse Clapeyron slope dT/dP [K/GPa]"); CHKERRQ(ierr);
  par->DT = par->Ms*par->DC;

  // two-phase flow parameters
  ierr = PetscBagRegisterScalar(bag, &par->phi0, 0.01, "phi0", "Reference porosityat which permeability equals reference permeability [-]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->n, 3.0, "n", "Exponent in porosity-permeability relationship [-]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->K0, 1.0e-13, "K0", "Reference permeability [m^2]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->phi_max, 0.1, "phi_max", "Porosity at which permeability reaches maximum [-]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->eta0, 1.0e19, "eta0", "Reference shear viscosity [Pa.s]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->zeta0, 5.0e19, "zeta0", "Reference bulk viscosity [Pa.s]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mu, 1.0, "mu", "Reference magma viscosity [Pa.s]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->eta_min, 1.0e15, "eta_min", "Cutoff minimum shear viscosity [Pa.s]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->eta_max, 1.0e25, "eta_max", "Cutoff maximum shear viscosity [Pa.s]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->lambda, 27, "lambda", "Porosity weakening of shear viscosity [-]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->EoR, 3.6e4, "EoR", "Activation energy divided by gas constant [K]"); CHKERRQ(ierr);
  
  dsol = par->cp*(par->Tp-par->T0)/par->gamma_inv*1e9/par->g/(par->rho0*par->cp - par->Tp*par->alpha/par->gamma_inv*1e9);
  Teta0 = par->Tp*exp(dsol*par->alpha*par->g/par->cp);
  ierr = PetscBagRegisterScalar(bag, &par->Teta0, Teta0, "Teta0", "Temperature at which viscosity is equal to eta0 [K]"); CHKERRQ(ierr);

  // time stepping and advection parameters
  ierr = PetscBagRegisterInt(bag, &par->ts_scheme,2, "ts_scheme", "Time stepping scheme 0-forward euler, 1-backward euler, 2-crank-nicholson"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->adv_scheme,2, "adv_scheme", "Advection scheme 0-upwind (FOU), 1-upwind2 (SOU) 2-fromm"); CHKERRQ(ierr);

  ierr = PetscBagRegisterInt(bag, &par->tout,1, "tout", "Output every tout time step"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->tstep,1, "tstep", "Maximum no of time steps"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->tmax, 1.0e6, "tmax", "Maximum time [yr]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->dtmax, 1.0e3, "dtmax", "Maximum time step size [yr]"); CHKERRQ(ierr);

  // input/output 
  par->fname_in[0] = '\0';
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
  ierr = PetscOptionsGetAll(NULL, &opts); CHKERRQ(ierr);

  // Print header and petsc options
  PetscPrintf(usr->comm,"# --------------------------------------- #\n");
  PetscPrintf(usr->comm,"# MID-OCEAN RIDGE BUOYANCY: %s \n",&(date[0]));
  PetscPrintf(usr->comm,"# --------------------------------------- #\n");
  PetscPrintf(usr->comm,"# PETSc options: %s \n",opts);
  PetscPrintf(usr->comm,"# --------------------------------------- #\n");

  // Input file info
  if (usr->par->fname_in[0] == '\0') { // string is empty
    PetscPrintf(usr->comm,"# Input options file: NONE (using default options)\n");
  }
  else {
    PetscPrintf(usr->comm,"# Input options file: %s \n",usr->par->fname_in);
  }
  PetscPrintf(usr->comm,"# --------------------------------------- #\n");

  // Print usr bag
  ierr = PetscBagView(usr->bag,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

  // Free memory
  ierr = PetscFree(opts); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// DefineScalingParameters
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "DefineScalingParameters"
PetscErrorCode DefineScalingParameters(UsrData *usr)
{
  ScalParams     *scal;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Allocate memory
  ierr = PetscMalloc1(1, &scal); CHKERRQ(ierr);

  scal->x = usr->par->H;
  scal->v = usr->par->K0*usr->par->drho*usr->par->g/usr->par->mu;
  scal->t = scal->x/scal->v;
  scal->K = usr->par->K0;
  scal->P = usr->par->drho*usr->par->g*scal->x;
  scal->eta = usr->par->eta0;
  scal->rho = usr->par->rho0;
  scal->Gamma = scal->v*scal->rho/scal->x;
  scal->H = scal->rho*usr->par->cp*usr->par->DT;

  PetscPrintf(usr->comm,"# --------------------------------------- #\n"); 
  PetscPrintf(usr->comm,"# Characteristic scales:\n");
  PetscPrintf(usr->comm,"#     [x]   = %1.12e (m    ) [v]     = %1.12e (m/s    ) [t]   = %1.12e (s   )\n",scal->x,scal->v,scal->t);
  PetscPrintf(usr->comm,"#     [K]   = %1.12e (m2   ) [P]     = %1.12e (Pa     ) [eta] = %1.12e (Pa.s)\n",scal->K,scal->P,scal->eta);
  PetscPrintf(usr->comm,"#     [rho] = %1.12e (kg/m3) [Gamma] = %1.12e (kg/m3/s) [H]   = %1.12e (J/m3)\n",scal->rho,scal->Gamma,scal->H);
 
  usr->scal = scal;

  PetscFunctionReturn(0);
}

// ---------------------------------------
// NondimensionalizeParameters
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "NondimensionalizeParameters"
PetscErrorCode NondimensionalizeParameters(UsrData *usr)
{
  NdParams       *nd;
  ScalParams     *scal;
  Params         *par;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // allocate memory
  ierr = PetscMalloc1(1, &nd); CHKERRQ(ierr);

  scal = usr->scal;
  par  = usr->par;

  // transform to SI units necessary params
  par->U0    = par->U0*1.0e-2/SEC_YEAR; //[cm/yr] to [m/s]
  par->tmax  = par->tmax*SEC_YEAR;      //[yr] to [s]
  par->dtmax = par->dtmax*SEC_YEAR;     //[yr] to [s]

  // non-dimensionalize
  nd->xmin  = nd_param(par->xmin,scal->x);
  nd->zmin  = nd_param(par->zmin,scal->x);
  nd->H     = nd_param(par->H,scal->x);
  nd->L     = nd_param(par->L,scal->x);
  nd->xMOR  = nd_param(par->xMOR,scal->x);
  nd->U0    = nd_param(par->U0,scal->v);
  nd->eta0  = nd_param(par->eta0,scal->eta);
  nd->zeta0 = nd_param(par->zeta0,scal->eta);

  // non-dimensional parameters
  nd->delta   = PetscSqrtScalar(scal->eta*scal->K/usr->par->mu)/scal->x;
  nd->alpha_s = par->alpha*par->rho0*par->DT/par->drho;
  nd->beta_s  = par->beta*par->rho0*par->DC/par->drho;
  nd->A       = par->alpha*par->g*scal->x/par->cp;
  nd->S       = par->La/par->cp/par->DT; 
  nd->PeT     = scal->x*scal->v/par->kappa;
  nd->PeC     = scal->x*scal->v/par->D;
  nd->theta_s = par->T0/par->DT;
  nd->G       = scal->x*par->drho*par->g*par->gamma_inv/par->DT;
  nd->RM      = par->Ms/par->Mf;

  PetscPrintf(usr->comm,"# --------------------------------------- #\n");
  PetscPrintf(usr->comm,"# Nondimensional parameters:\n");
  PetscPrintf(usr->comm,"#     delta   = %1.12e \n",nd->delta);
  PetscPrintf(usr->comm,"#     alpha_s = %1.12e \n",nd->alpha_s);
  PetscPrintf(usr->comm,"#     beta_s  = %1.12e \n",nd->beta_s);
  PetscPrintf(usr->comm,"#     A       = %1.12e \n",nd->A);
  PetscPrintf(usr->comm,"#     S       = %1.12e \n",nd->S);
  PetscPrintf(usr->comm,"#     PeT     = %1.12e \n",nd->PeT);
  PetscPrintf(usr->comm,"#     PeC     = %1.12e \n",nd->PeC);
  PetscPrintf(usr->comm,"#     G       = %1.12e \n",nd->G);
  PetscPrintf(usr->comm,"#     RM      = %1.12e \n",nd->RM);

  usr->nd = nd;

  PetscFunctionReturn(0);
}