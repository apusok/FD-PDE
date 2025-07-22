#include "mbuoy3.h"

// ---------------------------------------
// UserParamsCreate
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "UserParamsCreate"
PetscErrorCode UserParamsCreate(UsrData **_usr,int argc,char **argv)
{
  UsrData       *usr;
  PetscInt      i;
  PetscFunctionBeginUser;

  // default parameter values and command line input
  PetscCall(InputParameters(&usr)); 

  // file input
  for (i = 1; i < argc; i++) {
    PetscBool flg;
    
    PetscCall(PetscStrcmp(argv[i],"-options_file",&flg)); 
    if (flg) { PetscCall(PetscStrcpy(usr->par->fname_in, argv[i+1]));  }
  }

  // scaling parameters
  PetscCall(DefineScalingParameters(usr)); 

  // non-dimensionalize parameters
  PetscCall(NondimensionalizeParameters(usr)); 

  // print user parameters
  PetscCall(InputPrintData(usr)); 
  usr->par->start_run = PETSC_FALSE;

  // return pointer
  *_usr = usr;

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// UserParamsDestroy
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "UserParamsDestroy"
PetscErrorCode UserParamsDestroy(UsrData *usr)
{
  PetscFunctionBeginUser;

  PetscCall(PetscFree(usr->nd)); 
  PetscCall(PetscFree(usr->scal)); 
  PetscCall(PetscBagDestroy(&usr->bag)); 
  PetscCall(PetscFree(usr)); 
  
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
  PetscScalar    dsol, Teta0, DT;
  PetscFunctionBeginUser;

  // Allocate memory to application context
  PetscCall(PetscMalloc1(1, &usr)); 

  // initialize
  usr->par  = NULL;
  usr->nd   = NULL;
  usr->scal = NULL; 

  usr->dmPV = NULL;
  usr->dmHC = NULL;
  usr->dmVel= NULL;
  usr->dmEnth=NULL;

  usr->xPV  = NULL;
  usr->xHC  = NULL;
  usr->xVel = NULL;
  usr->xEnth= NULL;
  usr->xEnthold= NULL;

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

  // domain parameters
  PetscCall(PetscBagRegisterInt(bag, &par->nx, 20, "nx", "Element count in the x-dir [-]")); 
  PetscCall(PetscBagRegisterInt(bag, &par->nz, 10, "nz", "Element count in the z-dir [-]")); 

  PetscCall(PetscBagRegisterScalar(bag, &par->xmin, 0.0, "xmin", "Start coordinate of domain in x-dir [m]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->zmin, -100.0e3, "zmin", "Start coordinate of domain in z-dir [m]")); 

  PetscCall(PetscBagRegisterScalar(bag, &par->L, 200.0e3, "L", "Length of domain in x-dir [m]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->H, 100.0e3, "H", "Height of domain in z-dir [m]")); 

  // physical and material parameters
  PetscCall(PetscBagRegisterScalar(bag, &par->k_hat, -1.0, "k_hat", "Direction of unit vertical vector [-]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->g, 9.8, "g", "Gravitational acceleration [m^2/s]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->U0, 4.0, "U0", "Half-spreading rate [cm/yr]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->hs_factor, 2.0, "hs_factor", "Half-space cooling factor [-]")); 

  PetscCall(PetscBagRegisterScalar(bag, &par->Tp, 1648, "Tp", "Potential temperature of mantle [K]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->Ts, T_KELVIN, "Ts", "Surface temperature [K]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->cp, 1200, "cp", "Specific heat of matrix and magma [J/kg/K]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->La, 4.0e5, "La", "Latent fusion of heat [J/kg]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->kappa, 1.0e-6, "kappa", "Thermal diffusivity [m^2/s]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->D, 1.0e-8, "D", "Chemical diffusivity [m^2/s]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->rho0, 3000, "rho0", "Reference density of matrix and magma [kg/m^3]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->drho, 500, "drho", "Reference density difference [kg/m^3]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->alpha, 3.0e-5, "alpha", "Coefficient of thermal expansion [1/K]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->beta, 0.0, "beta", "Coefficient of compositional expansion [1/wt. frac.]")); 

  // buoyancy
  PetscCall(PetscBagRegisterInt(bag, &par->buoyancy, 0, "buoyancy", "Level of matrix buoyancy incorporation: 0=off, 1=phi, 2=C, 3=phi-C, 4=T, 5=phi-T, 6=C-T, 7=phi-C-T")); 
  PetscCall(PetscBagRegisterInt(bag, &par->buoy_phi, 0, "buoy_phi", "Level of buoyancy (porosity): 0=off, 1=on")); 
  PetscCall(PetscBagRegisterInt(bag, &par->buoy_C, 0, "buoy_C", "Level of buoyancy (composition): 0=off, 1=on, 2-extended")); 
  PetscCall(PetscBagRegisterInt(bag, &par->buoy_T, 0, "buoy_T", "Level of buoyancy (temperature): 0=off, 1=on, 2-extended")); 
  if ((par->buoy_phi==0) && (par->buoy_C==0) && (par->buoy_T==0)) par->buoyancy = 0;
  if ((par->buoy_phi==1) && (par->buoy_C==0) && (par->buoy_T==0)) par->buoyancy = 1;
  if ((par->buoy_phi==0) && (par->buoy_C>=1) && (par->buoy_T==0)) par->buoyancy = 2;
  if ((par->buoy_phi==1) && (par->buoy_C>=1) && (par->buoy_T==0)) par->buoyancy = 3;
  if ((par->buoy_phi==0) && (par->buoy_C==0) && (par->buoy_T>=1)) par->buoyancy = 4;
  if ((par->buoy_phi==1) && (par->buoy_C==0) && (par->buoy_T>=1)) par->buoyancy = 5;
  if ((par->buoy_phi==0) && (par->buoy_C>=1) && (par->buoy_T>=1)) par->buoyancy = 6;
  if ((par->buoy_phi==1) && (par->buoy_C>=1) && (par->buoy_T>=1)) par->buoyancy = 7;

  PetscCall(PetscBagRegisterInt(bag, &par->initial_bulk_comp, 0, "initial_bulk_comp", "0-off, 1-on Initialize bulk composition everywhere as the column beneath axis (depleted)")); 
  PetscCall(PetscBagRegisterInt(bag, &par->hc_cycles, 1, "hc_cycles", "Number of timesteps and HC solves until update PV solve")); 

  // bottom boundary forcing - only in full ridge models
  PetscCall(PetscBagRegisterInt(bag, &par->forcing, 0, "forcing", "Bottom forcing: 0=off, 1=Temp forcing dT/dx, 2=Chemical forcing dC/dx")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->dTdx_bottom, 0.01, "dTdx_bottom", "Lateral temperature gradient imposed on bottom boundary [K/km]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->dCdx_bottom, 1.0e-6, "dCdx_bottom", "Lateral compositional gradient imposed on bottom boundary [wt. frac./km]")); 

  // phase diagram
  PetscCall(PetscBagRegisterScalar(bag, &par->C0, 0.85, "C0", "Reference composition [wt. frac.]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->DC, 0.1, "DC", "Compositional diff between solidus and liquidus at T0 [wt.  frac.]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->T0, 1565, "T0", "Solidus temperature at P=0, C=C0 [K]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->Ms, 400, "Ms", "Slope of solidus dTdC [K]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->Mf, 400, "Mf", "Slope of liquidus dTdC [K]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->gamma_inv, 60, "gamma_inv", "Inverse Clapeyron slope dT/dP [K/GPa]")); 
  DT = par->Ms*par->DC;
  PetscCall(PetscBagRegisterScalar(bag, &par->DT, DT, "DT", "Reference temperature difference [K]")); 

  // two-phase flow parameters
  PetscCall(PetscBagRegisterScalar(bag, &par->n, 3.0, "n", "Exponent in porosity-permeability relationship [-]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->K0, 1.0e-7, "K0", "Reference permeability [m^2]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->eta0, 1.0e19, "eta0", "Reference shear viscosity [Pa.s]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->zeta0, 4.0e19, "zeta0", "Reference bulk viscosity [Pa.s]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->mu, 1.0, "mu", "Reference magma viscosity [Pa.s]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->eta_min, 1.0e15, "eta_min", "Cutoff minimum shear viscosity [Pa.s]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->eta_max, 1.0e25, "eta_max", "Cutoff maximum shear viscosity [Pa.s]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->lambda, 27, "lambda", "Porosity weakening of shear viscosity [-]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->EoR, 3.6e4, "EoR", "Activation energy divided by gas constant [K]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->zetaExp, -1.0, "zetaExp", "Porosity exponent in bulk viscosity [-]")); 

  PetscCall(PetscBagRegisterInt(bag, &par->visc_shear,2, "visc_shear", "0-constant, 1-porosity dependent, 2-Temp,porosity dependent")); 
  PetscCall(PetscBagRegisterInt(bag, &par->visc_bulk,2, "visc_bulk", "0-constant, 1-porosity dependent, 2-Temp,porosity dependent")); 

  PetscCall(PetscBagRegisterScalar(bag, &par->phi_init, 1.0e-4, "phi_init", "Extract initial porosity phi*phi_init")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->phi_min, PHI_CUTOFF, "phi_min", "Cutoff minimum porosity")); 

  dsol = par->cp*(par->Tp-par->T0)/par->gamma_inv*1e9/par->g/(par->rho0*par->cp - par->Tp*par->alpha/par->gamma_inv*1e9);
  Teta0 = par->Tp*exp(dsol*par->alpha*par->g/par->cp);
  PetscCall(PetscBagRegisterScalar(bag, &par->Teta0, Teta0, "Teta0", "Temperature at which viscosity is equal to eta0 [K]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->dsol, dsol, "zm", "Depth of melting [m]")); 

  // bc and melt extraction
  PetscCall(PetscBagRegisterInt(bag, &par->extract_mech,1, "extract_mech", "LEGACY: 0-no extract (T_up), 1-outflow xMOR (dH/dz=0), both depend on xmor")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->xmor, 4.0e3, "xmor", "Distance from mid-ocean ridge axis for melt extraction ~6km [m]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->fextract, 0.2, "fextract", "Percentage of melt extraction")); 

  PetscCall(PetscBagRegisterInt(bag, &par->vf_nonlinear,0, "vf_nonlinear", "0-update vf outside HC solve, 1-update vf inside HC solve")); 

  // time stepping and advection parameters
  PetscCall(PetscBagRegisterInt(bag, &par->ts_scheme,2, "ts_scheme", "Time stepping scheme 0-forward euler, 1-backward euler, 2-crank-nicholson")); 
  PetscCall(PetscBagRegisterInt(bag, &par->adv_scheme,2, "adv_scheme", "Advection scheme 0-upwind (FOU), 1-upwind2 (SOU) 2-fromm")); 
  PetscCall(PetscBagRegisterInt(bag, &par->tout,1, "tout", "Output every tout time step")); 
  PetscCall(PetscBagRegisterInt(bag, &par->tstep,1, "tstep", "Maximum no of time steps")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->dt_out, 1.0e3, "dt_out", "Output every dt_out time [yr]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->tmax, 1.0e6, "tmax", "Maximum time [yr]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->dtmax, 1.0e3, "dtmax", "Maximum time step size [yr]")); 
  PetscCall(PetscBagRegisterInt(bag, &par->restart,0, "restart", "Restart from #istep, 0-means start from beginning")); 

  PetscCall(PetscBagRegisterInt(bag, &par->full_ridge,0, "full_ridge", "0-half ridge, 1-full ridge")); 
  par->start_run = PETSC_TRUE;
  PetscCall(PetscBagRegisterBool(bag, &par->log_info,PETSC_FALSE, "model_log_info", "Output profiling data (T/F)")); 

  if (par->full_ridge) {
    par->nx   = 2*par->nx;
    par->xmin = par->xmin-par->L;
    par->L    = 2.0*par->L;
  }

  // input/output 
  par->fname_in[0] = '\0';
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
  NdParams       *nd;
  ScalParams     *scal;
  PetscFunctionBeginUser;

  scal = usr->scal;
  nd   = usr->nd;

  // print new simulation info
  if (usr->par->start_run) {
    // Get date
    PetscCall(PetscGetDate(date,30)); 
    PetscCall(PetscOptionsGetAll(NULL, &opts)); 

    // Print header and petsc options
    PetscPrintf(usr->comm,"# --------------------------------------- #\n");
    PetscPrintf(usr->comm,"# MID-OCEAN RIDGE BUOYANCY - 3-Field: %s \n",&(date[0]));
    PetscPrintf(usr->comm,"# --------------------------------------- #\n");
    PetscPrintf(usr->comm,"# PETSc options: %s \n",opts);
    PetscPrintf(usr->comm,"# --------------------------------------- #\n");

    // Free memory
    PetscCall(PetscFree(opts)); 

    // Input file info
    if (usr->par->fname_in[0] == '\0') { // string is empty
      PetscPrintf(usr->comm,"# Input options file: NONE (using default options)\n");
    }
    else {
      PetscPrintf(usr->comm,"# Input options file: %s \n",usr->par->fname_in);
    }
  }

  // Print usr bag
  PetscPrintf(usr->comm,"# --------------------------------------- #\n");
  PetscCall(PetscBagView(usr->bag,PETSC_VIEWER_STDOUT_WORLD)); 

  // Print scal and nd params
  PetscPrintf(usr->comm,"# --------------------------------------- #\n"); 
  PetscPrintf(usr->comm,"# Characteristic scales:\n");
  PetscPrintf(usr->comm,"#     [x]   = %1.12e (m    ) [v]     = %1.12e (m/s    ) [t]   = %1.12e (s   )\n",scal->x,scal->v,scal->t);
  PetscPrintf(usr->comm,"#     [K]   = %1.12e (m2   ) [P]     = %1.12e (Pa     ) [eta] = %1.12e (Pa.s)\n",scal->K,scal->P,scal->eta);
  PetscPrintf(usr->comm,"#     [rho] = %1.12e (kg/m3) [Gamma] = %1.12e (kg/m3/s) [H]   = %1.12e (J/m3)\n",scal->rho,scal->Gamma,scal->H);
 
  PetscPrintf(usr->comm,"# --------------------------------------- #\n");
  PetscPrintf(usr->comm,"# Nondimensional parameters:\n");
  PetscPrintf(usr->comm,"#     delta   = %1.12e \n",nd->delta);
  PetscPrintf(usr->comm,"#     alpha_s = %1.12e \n",nd->alpha_s);
  PetscPrintf(usr->comm,"#     beta_s  = %1.12e \n",nd->beta_s);
  PetscPrintf(usr->comm,"#     alpha_ls= %1.12e \n",nd->alpha_ls);
  PetscPrintf(usr->comm,"#     beta_ls = %1.12e \n",nd->beta_ls);
  PetscPrintf(usr->comm,"#     A       = %1.12e \n",nd->A);
  PetscPrintf(usr->comm,"#     S       = %1.12e \n",nd->S);
  PetscPrintf(usr->comm,"#     PeT     = %1.12e \n",nd->PeT);
  PetscPrintf(usr->comm,"#     PeC     = %1.12e \n",nd->PeC);
  PetscPrintf(usr->comm,"#     G       = %1.12e \n",nd->G);
  PetscPrintf(usr->comm,"#     RM      = %1.12e \n",nd->RM);

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// DefineScalingParameters
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "DefineScalingParameters"
PetscErrorCode DefineScalingParameters(UsrData *usr)
{
  ScalParams     *scal;
  Params         *par;
  PetscFunctionBeginUser;

  par  = usr->par;

  // allocate memory
  PetscCall(PetscMalloc1(1, &scal)); 

  scal->x = par->H;
  scal->v = par->K0*par->drho*par->g/par->mu;
  scal->t = scal->x/scal->v;
  scal->K = par->K0;
  scal->P = par->drho*par->g*scal->x;
  scal->eta = par->eta0;
  scal->rho = par->drho;
  scal->Gamma = scal->v*par->rho0/scal->x;
  scal->H = par->rho0*par->cp*par->DT;

  usr->scal = scal;

  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscFunctionBeginUser;

  scal = usr->scal;
  par  = usr->par;

  // allocate memory
  PetscCall(PetscMalloc1(1, &nd)); 
  nd->istep = 0;

  // transform to SI units necessary params
  nd->U0    = par->U0*1.0e-2/SEC_YEAR; //[cm/yr] to [m/s]
  nd->tmax  = par->tmax*SEC_YEAR;      //[yr] to [s]
  nd->dtmax = par->dtmax*SEC_YEAR;     //[yr] to [s]
  nd->dt_out= par->dt_out*SEC_YEAR;     //[yr] to [s]

  // non-dimensionalize
  nd->xmin  = nd_param(par->xmin,scal->x);
  nd->zmin  = nd_param(par->zmin,scal->x);
  nd->H     = nd_param(par->H,scal->x);
  nd->L     = nd_param(par->L,scal->x);
  nd->xmor = nd_param(par->xmor,scal->x);
  nd->U0    = nd_param(nd->U0,scal->v);
  nd->visc_ratio = nd_param(par->zeta0,scal->eta);
  nd->eta_min = nd_param(par->eta_min,scal->eta);
  nd->eta_max = nd_param(par->eta_max,scal->eta);
  nd->tmax  = nd_param(nd->tmax,scal->t);
  nd->dtmax = nd_param(nd->dtmax,scal->t);
  nd->dt_out = nd_param(nd->dt_out,scal->t);

  nd->dt    = 0.0;
  nd->t     = 0.0;

  // non-dimensional parameters
  nd->delta   = PetscSqrtScalar(scal->eta*scal->K/usr->par->mu)/scal->x;
  nd->alpha_s = par->alpha*par->rho0*par->DT/par->drho;
  nd->beta_s  = par->beta*par->rho0*par->DC/par->drho;
  nd->alpha_ls= par->alpha*par->DT;
  nd->beta_ls = par->beta*par->DC;
  nd->A       = par->alpha*par->g*scal->x/par->cp;
  nd->S       = par->La/par->cp/par->DT; 
  nd->PeT     = scal->x*scal->v/par->kappa;
  nd->PeC     = scal->x*scal->v/par->D;
  nd->thetaS  = par->T0/par->DT;
  nd->G       = scal->x*par->drho*par->g*par->gamma_inv/par->DT*1e-9; // GPa from gamma_inv
  nd->RM      = par->Ms/par->Mf;

  usr->nd = nd;

  PetscFunctionReturn(PETSC_SUCCESS);
}
