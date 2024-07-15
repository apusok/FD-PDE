#include "morfault.h"

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
  for (i = 1; i < argc; i++) {
    PetscBool flg;
    
    ierr = PetscStrcmp(argv[i],"-options_file",&flg); CHKERRQ(ierr);
    if (flg) { ierr = PetscStrcpy(usr->par->fname_in, argv[i+1]); CHKERRQ(ierr); }
  }

  // scaling parameters
  ierr = DefineScalingParameters(usr); CHKERRQ(ierr);

  // non-dimensionalize parameters
  ierr = NondimensionalizeParameters(usr); CHKERRQ(ierr);

  // print user parameters
  ierr = InputPrintData(usr); CHKERRQ(ierr);
  usr->par->start_run = PETSC_FALSE;

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
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Allocate memory to application context
  ierr = PetscMalloc1(1, &usr); CHKERRQ(ierr);

  // initialize
  usr->par  = NULL;
  usr->nd   = NULL;
  usr->scal = NULL; 

  usr->dmPV = NULL;
  usr->dmT  = NULL;
  usr->dmswarm = NULL;
  usr->dmVel = NULL;
  usr->dmMPhase = NULL;
  usr->dmPlith = NULL;
  usr->dmeps = NULL;
  usr->dmmatProp = NULL;
  
  usr->xPV  = NULL;
  usr->xT   = NULL;
  usr->xphi = NULL;
  usr->xVel = NULL;
  usr->xMPhase = NULL;
  usr->xPlith = NULL;
  usr->xeps = NULL;
  usr->xtau = NULL;
  usr->xtau_old = NULL;
  usr->xDP = NULL;
  usr->xDP_old = NULL;
  usr->xplast = NULL;
  usr->xmatProp = NULL;
  usr->xstrain = NULL;

  usr->plasticity = PETSC_FALSE;

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
  ierr = PetscBagRegisterInt(bag, &par->nx, 100, "nx", "Element count in the x-dir [-]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->nz, 50, "nz", "Element count in the z-dir [-]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->ppcell, 4, "ppcell", "Number of particles/cell one-dir"); CHKERRQ(ierr);

  ierr = PetscBagRegisterScalar(bag, &par->xmin, -100.0e3, "xmin", "Start coordinate of domain in x-dir [m]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->zmin, -80.0e3, "zmin", "Start coordinate of domain in z-dir [m]"); CHKERRQ(ierr);

  ierr = PetscBagRegisterScalar(bag, &par->L, 200.0e3, "L", "Length of domain in x-dir [m]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->H, 100.0e3, "H", "Height of domain in z-dir [m]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->Hs,20.0e3, "Hs", "Free-surface height [m]"); CHKERRQ(ierr);

  // physical and material parameters
  ierr = PetscBagRegisterScalar(bag, &par->k_hat, -1.0, "k_hat", "Direction of unit vertical vector [-]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->g, 9.8, "g", "Gravitational acceleration [m^2/s]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->Ttop, T_KELVIN, "Ttop", "Temperature on top boundary [K]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->Tbot, 1523.15, "Tbot", "Temperature on bottom boundary - also potential T [K]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->R, 8.314, "R", "Gas constant [J/mol/K]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->Vext, 1.0, "Vext", "Extension velocity [cm/yr]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->hs_factor, 2.0, "hs_factor", "Half-space cooling factor [-]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->drho, 500, "drho", "Reference density difference matrix-magma [kg/m3]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->rhof, 2500, "rhof", "Liquid density [kg/m3]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->age, 40.0, "age", "Initial lithospheric age [Myr]"); CHKERRQ(ierr);

  // initial perturbation
  ierr = PetscBagRegisterScalar(bag, &par->incl_x, 0e3, "incl_x", "Inclusion X-start point [m]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->incl_z, -120e3, "incl_z", "Inclusion Z-start point [m]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->incl_r, 20e3, "incl_r", "Inclusion radius [m]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->incl_dT, 10, "incl_dT", "Inclusion T perturbation [K]"); CHKERRQ(ierr);

  // two-phase flow parameters
  ierr = PetscBagRegisterScalar(bag, &par->n, 3.0, "n", "Exponent in porosity-permeability relationship [-]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->kphi0, 1.0e-7, "kphi0", "Permeability prefactor [m^2]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mu, 1.0, "mu", "Reference magma viscosity [Pa.s]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->lambda, 27, "lambda", "Porosity weakening of shear viscosity [-]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->EoR, 3.6e4, "EoR", "Activation energy divided by gas constant [K]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->q, -0.5, "q", "Exponent of the porosity-dependent relation of poro-elastic modulus"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->zetaExp, -1.0, "zetaExp", "Porosity exponent in bulk viscosity [-]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->Teta0, 1672.82, "Teta0", "Temperature at which viscosity is equal to eta0 [K]"); CHKERRQ(ierr);

  // regularization
  ierr = PetscBagRegisterScalar(bag, &par->eta_min, 1.0e15, "eta_min", "Cutoff minimum shear viscosity [Pa.s]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->eta_max, 1.0e25, "eta_max", "Cutoff maximum shear viscosity [Pa.s]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->phi_min, PHI_CUTOFF, "phi_min", "Cutoff minimum porosity"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->phi0, 1e-4, "phi0", "Reference background porosity"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->eta_K, 1e22, "eta_K", "Shear viscosity of the Kelvin VP dashpot"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->tf_tol, 1e-8, "tf_tol", "Function tolerance for solving yielding stresses"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->Nmax, 25, "Nmax", "Max Newton iteration for plasticity"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->strain_max, 0.1, "strain_max", "Total plastic strain for softening"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->hcc, 0.5, "hcc", "Relative reduction for cohesion and friction angle during strain softening"); CHKERRQ(ierr);

  // material phases for markers (markers carry only phase id)
  ierr = PetscBagRegisterInt(bag, &par->marker_phases, 6, "marker_phases", "Number of marker phases [-]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->matid_default, 5, "matid_default", "Default material phase for scaling [-]"); CHKERRQ(ierr);

  ierr = PetscBagRegisterInt(bag, &par->mat0_id, 0, "mat0_id", "Material phase 0 [-]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterString(bag,&par->mat0_name,FNAME_LENGTH,"mat0_name","stick-water","Name for material phase 0"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat0_rho0, 1000, "mat0_rho0", "Reference density for mat_phase 0 [kg/m3]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat0_alpha, 3.0e-5, "mat0_alpha", "Coefficient of thermal expansion for mat_phase 0 [1/K]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat0_cp, 4000, "mat0_cp", "Specific heat for mat_phase 0 [J/kg/K]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat0_kT, 0.6, "mat0_kT", "Thermal conductivity for mat_phase 0 [W/m/K]"); CHKERRQ(ierr);
  par->mat0_kappa = par->mat0_kT/par->mat0_rho0/par->mat0_cp;
  ierr = PetscBagRegisterScalar(bag, &par->mat0_kappa, par->mat0_kappa, "mat0_kappa", "Thermal diffusivity for mat_phase 0 [m^2/s]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->mat0_rho_function, 0, "mat0_rho_function", "Material phase 0 density function: 0-constant [-]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->mat0_eta_function, 0, "mat0_eta_function", "Material phase 0 eta function: 0-constant, 1-phi, 2-phi,T, 3-phi,T,eps [-]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->mat0_zeta_function, 0, "mat0_zeta_function", "Material phase 0 zeta function: 0-constant [-]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat0_eta0, 1.0e18, "mat0_eta0", "Material phase 0 Reference shear viscosity [Pa.s]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat0_zeta0, par->eta_max, "mat0_zeta0", "Material phase 0 Reference compaction viscosity [Pa.s]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat0_G, 1e20, "mat0_G", "Shear elastic modulus 0 [Pa]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat0_Z0, 1e40, "mat0_Z0", "Reference poro-elastic modulus 0 [Pa]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat0_C, 1e40, "mat0_C", "Material phase 0 Cohesion (Pa)"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat0_sigmat, 1e40, "mat0_sigmat", "Material phase 0 Yield stress (Pa)"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat0_theta, 30.0, "mat0_theta", "Material phase 0 Friction angle (-)"); CHKERRQ(ierr);

  ierr = PetscBagRegisterInt(bag, &par->mat1_id, 1, "mat1_id", "Material phase 1 [-]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterString(bag,&par->mat1_name,FNAME_LENGTH,"mat1_name","mantle 1","Name for material phase 1"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat1_rho0, 3000, "mat1_rho0", "Reference density for mat_phase 1 [kg/m3]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat1_alpha, 3.0e-5, "mat1_alpha", "Coefficient of thermal expansion for mat_phase 1 [1/K]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat1_cp, 1200, "mat1_cp", "Specific heat for mat_phase 1 [J/kg/K]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat1_kappa, 1.0e-6, "mat1_kappa", "Thermal diffusivity for mat_phase 1 [m^2/s]"); CHKERRQ(ierr);
  par->mat1_kT = par->mat1_kappa*par->mat1_rho0*par->mat1_cp;
  ierr = PetscBagRegisterScalar(bag, &par->mat1_kT, par->mat1_kT, "mat1_kT", "Thermal conductivity for mat_phase 1 [W/m/K]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->mat1_rho_function, 0, "mat1_rho_function", "Material phase 1 density function: 0-constant [-]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->mat1_eta_function, 0, "mat1_eta_function", "Material phase 1 eta function: 0-constant, 1-phi, 2-phi,T, 3-phi,T,eps [-]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->mat1_zeta_function, 0, "mat1_zeta_function", "Material phase 1 zeta function: 0-constant, 1-phi, 2-phi,T, 3-phi,T,eps [-]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat1_eta0, 1.0e19, "mat1_eta0", "Material phase 1 Reference shear viscosity [Pa.s]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat1_zeta0, 4.0e19, "mat1_zeta0", "Material phase 1 Reference compaction viscosity [Pa.s]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat1_G, 6e10, "mat1_G", "Shear elastic modulus 1 [Pa]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat1_Z0, 1e40, "mat1_Z0", "Reference poro-elastic modulus 1 [Pa]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat1_C, 1e40, "mat1_C", "Material phase 1 Cohesion (Pa)"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat1_sigmat, 1e40, "mat1_sigmat", "Material phase 1 Yield stress (Pa)"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat1_theta, 0.0, "mat1_theta", "Material phase 1 Friction angle (-)"); CHKERRQ(ierr);

  ierr = PetscBagRegisterInt(bag, &par->mat2_id, 2, "mat2_id", "Material phase 2 [-]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterString(bag,&par->mat2_name,FNAME_LENGTH,"mat2_name","mantle 2","Name for material phase 2"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat2_rho0, 3000, "mat2_rho0", "Reference density for mat_phase 2 [kg/m3]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat2_alpha, 3.0e-5, "mat2_alpha", "Coefficient of thermal expansion for mat_phase 2 [1/K]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat2_cp, 1200, "mat2_cp", "Specific heat for mat_phase 2 [J/kg/K]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat2_kappa, 1.0e-6, "mat2_kappa", "Thermal diffusivity for mat_phase 2 [m^2/s]"); CHKERRQ(ierr);
  par->mat2_kT = par->mat2_kappa*par->mat2_rho0*par->mat2_cp;
  ierr = PetscBagRegisterScalar(bag, &par->mat2_kT, par->mat2_kT, "mat2_kT", "Thermal conductivity for mat_phase 2 [W/m/K]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->mat2_rho_function, 0, "mat2_rho_function", "Material phase 2 density function: 0-constant [-]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->mat2_eta_function, 0, "mat2_eta_function", "Material phase 2 eta function: 0-constant, 1-phi, 2-phi,T, 3-phi,T,eps [-]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->mat2_zeta_function, 0, "mat2_zeta_function", "Material phase 2 zeta function: 0-constant, 1-phi, 2-phi,T, 3-phi,T,eps [-]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat2_eta0, 1.0e19, "mat2_eta0", "Material phase 2 Reference shear viscosity [Pa.s]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat2_zeta0, 4.0e19, "mat2_zeta0", "Material phase 2 Reference compaction viscosity [Pa.s]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat2_G, 6e10, "mat2_G", "Shear elastic modulus 2 [Pa]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat2_Z0, 1e40, "mat2_Z0", "Reference poro-elastic modulus 2 [Pa]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat2_C, 1e40, "mat2_C", "Material phase 2 Cohesion (Pa)"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat2_sigmat, 1e40, "mat2_sigmat", "Material phase 2 Yield stress (Pa)"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat2_theta, 0.0, "mat2_theta", "Material phase 2 Friction angle (-)"); CHKERRQ(ierr);

  ierr = PetscBagRegisterInt(bag, &par->mat3_id, 3, "mat3_id", "Material phase 3 [-]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterString(bag,&par->mat3_name,FNAME_LENGTH,"mat3_name","mantle 3","Name for material phase 3"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat3_rho0, 3000, "mat3_rho0", "Reference density for mat_phase 3 [kg/m3]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat3_alpha, 3.0e-5, "mat3_alpha", "Coefficient of thermal expansion for mat_phase 3 [1/K]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat3_cp, 1200, "mat3_cp", "Specific heat for mat_phase 3 [J/kg/K]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat3_kappa, 1.0e-6, "mat3_kappa", "Thermal diffusivity for mat_phase 3 [m^2/s]"); CHKERRQ(ierr);
  par->mat3_kT = par->mat3_kappa*par->mat3_rho0*par->mat3_cp;
  ierr = PetscBagRegisterScalar(bag, &par->mat3_kT, par->mat3_kT, "mat3_kT", "Thermal conductivity for mat_phase 3 [W/m/K]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->mat3_rho_function, 0, "mat3_rho_function", "Material phase 3 density function: 0-constant [-]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->mat3_eta_function, 0, "mat3_eta_function", "Material phase 3 eta function: 0-constant, 1-phi, 2-phi,T, 3-phi,T,eps [-]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->mat3_zeta_function, 0, "mat3_zeta_function", "Material phase 3 zeta function: 0-constant, 1-phi, 2-phi,T, 3-phi,T,eps [-]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat3_eta0, 1.0e19, "mat3_eta0", "Material phase 3 Reference shear viscosity [Pa.s]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat3_zeta0, 4.0e19, "mat3_zeta0", "Material phase 3 Reference compaction viscosity [Pa.s]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat3_G, 6e10, "mat3_G", "Shear elastic modulus 3 [Pa]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat3_Z0, 1e40, "mat3_Z0", "Reference poro-elastic modulus 3 [Pa]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat3_C, 1e40, "mat3_C", "Material phase 3 Cohesion (Pa)"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat3_sigmat, 1e40, "mat3_sigmat", "Material phase 3 Yield stress (Pa)"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat3_theta, 0.0, "mat3_theta", "Material phase 3 Friction angle (-)"); CHKERRQ(ierr);

  ierr = PetscBagRegisterInt(bag, &par->mat4_id, 4, "mat4_id", "Material phase 4 [-]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterString(bag,&par->mat4_name,FNAME_LENGTH,"mat4_name","mantle 4","Name for material phase 4"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat4_rho0, 3000, "mat4_rho0", "Reference density for mat_phase 4 [kg/m3]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat4_alpha, 3.0e-5, "mat4_alpha", "Coefficient of thermal expansion for mat_phase 4 [1/K]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat4_cp, 1200, "mat4_cp", "Specific heat for mat_phase 4 [J/kg/K]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat4_kappa, 1.0e-6, "mat4_kappa", "Thermal diffusivity for mat_phase 4 [m^2/s]"); CHKERRQ(ierr);
  par->mat4_kT = par->mat4_kappa*par->mat4_rho0*par->mat4_cp;
  ierr = PetscBagRegisterScalar(bag, &par->mat4_kT, par->mat4_kT, "mat4_kT", "Thermal conductivity for mat_phase 4 [W/m/K]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->mat4_rho_function, 0, "mat4_rho_function", "Material phase 4 density function: 0-constant [-]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->mat4_eta_function, 0, "mat4_eta_function", "Material phase 4 eta function: 0-constant, 1-phi, 2-phi,T, 3-phi,T,eps [-]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->mat4_zeta_function, 0, "mat4_zeta_function", "Material phase 4 zeta function: 0-constant, 1-phi, 2-phi,T, 3-phi,T,eps [-]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat4_eta0, 1.0e19, "mat4_eta0", "Material phase 4 Reference shear viscosity [Pa.s]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat4_zeta0, 4.0e19, "mat4_zeta0", "Material phase 4 Reference compaction viscosity [Pa.s]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat4_G, 6e10, "mat4_G", "Shear elastic modulus 4 [Pa]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat4_Z0, 1e40, "mat4_Z0", "Reference poro-elastic modulus 4 [Pa]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat4_C, 1e40, "mat4_C", "Material phase 4 Cohesion (Pa)"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat4_sigmat, 1e40, "mat4_sigmat", "Material phase 4 Yield stress (Pa)"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat4_theta, 0.0, "mat4_theta", "Material phase 4 Friction angle (-)"); CHKERRQ(ierr);

  ierr = PetscBagRegisterInt(bag, &par->mat5_id, 5, "mat5_id", "Material phase 5 [-]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterString(bag,&par->mat5_name,FNAME_LENGTH,"mat5_name","mantle 5","Name for material phase 5"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat5_rho0, 3000, "mat5_rho0", "Reference density for mat_phase 5 [kg/m3]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat5_alpha, 3.0e-5, "mat5_alpha", "Coefficient of thermal expansion for mat_phase 5 [1/K]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat5_cp, 1200, "mat5_cp", "Specific heat for mat_phase 5 [J/kg/K]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat5_kappa, 1.0e-6, "mat5_kappa", "Thermal diffusivity for mat_phase 5 [m^2/s]"); CHKERRQ(ierr);
  par->mat5_kT = par->mat5_kappa*par->mat5_rho0*par->mat5_cp;
  ierr = PetscBagRegisterScalar(bag, &par->mat5_kT, par->mat5_kT, "mat5_kT", "Thermal conductivity for mat_phase 5 [W/m/K]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->mat5_rho_function, 0, "mat5_rho_function", "Material phase 5 density function: 0-constant [-]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->mat5_eta_function, 0, "mat5_eta_function", "Material phase 5 eta function: 0-constant, 1-phi, 2-phi,T, 3-phi,T,eps [-]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->mat5_zeta_function, 0, "mat5_zeta_function", "Material phase 5 zeta function: 0-constant, 1-phi, 2-phi,T, 3-phi,T,eps [-]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat5_eta0, 1.0e19, "mat5_eta0", "Material phase 5 Reference shear viscosity [Pa.s]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat5_zeta0, 4.0e19, "mat5_zeta0", "Material phase 5 Reference compaction viscosity [Pa.s]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat5_G, 6e10, "mat5_G", "Shear elastic modulus 5 [Pa]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat5_Z0, 1e40, "mat5_Z0", "Reference poro-elastic modulus 5 [Pa]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat5_C, 1e40, "mat5_C", "Material phase 5 Cohesion (Pa)"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat5_sigmat, 1e40, "mat5_sigmat", "Material phase 5 Yield stress (Pa)"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->mat5_theta, 0.0, "mat5_theta", "Material phase 5 Friction angle (-)"); CHKERRQ(ierr);

  // time stepping and advection parameters
  ierr = PetscBagRegisterInt(bag, &par->ts_scheme,2, "ts_scheme", "Time stepping scheme 0-forward euler, 1-backward euler, 2-crank-nicholson"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->adv_scheme,2, "adv_scheme", "Advection scheme 0-upwind (FOU), 1-upwind2 (SOU) 2-fromm"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->tout,1, "tout", "Output every tout time step"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->tstep,0, "tstep", "Maximum no of time steps"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->dt_out, 1.0e3, "dt_out", "Output every dt_out time [yr]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->tmax, 1.0e6, "tmax", "Maximum time [yr]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->dtmax, 1.0e3, "dtmax", "Maximum time step size [yr]"); CHKERRQ(ierr);

  ierr = PetscBagRegisterInt(bag, &par->rheology,0, "rheology", "0-VEP 1-VEVP (AveragePhase) 2-VEVP (DominantPhase)"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->two_phase,0, "two_phase", "0-single (Stokes) 1-two_phase (StokesDarcy)"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->model_setup,0, "model_setup", "0-weak inclusion 1-temp perturbation 2-age-dep temp profile"); CHKERRQ(ierr);
  
  // boolean options
  ierr = PetscBagRegisterBool(bag, &par->log_info,PETSC_FALSE, "model_log_info", "Output profiling data (T/F)"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->restart,0, "restart", "Restart from #istep, 0-means start from beginning"); CHKERRQ(ierr);
  par->start_run = PETSC_TRUE;

  // input/output 
  par->fname_in[0] = '\0';
  ierr = PetscBagRegisterString(bag,&par->fname_out,FNAME_LENGTH,"out_solution","output_file","Name for output file, set with: -output_file <filename>"); CHKERRQ(ierr);

  // Material properties on markers
  usr->nph = par->marker_phases;
  usr->mat[0].rho0 = par->mat0_rho0;
  usr->mat[1].rho0 = par->mat1_rho0;
  usr->mat[2].rho0 = par->mat2_rho0;
  usr->mat[3].rho0 = par->mat3_rho0;
  usr->mat[4].rho0 = par->mat4_rho0;
  usr->mat[5].rho0 = par->mat5_rho0;

  usr->mat[0].rho_func = par->mat0_rho_function;
  usr->mat[1].rho_func = par->mat1_rho_function;
  usr->mat[2].rho_func = par->mat2_rho_function;
  usr->mat[3].rho_func = par->mat3_rho_function;
  usr->mat[4].rho_func = par->mat4_rho_function;
  usr->mat[5].rho_func = par->mat5_rho_function;

  usr->mat[0].alpha = par->mat0_alpha;
  usr->mat[1].alpha = par->mat1_alpha;
  usr->mat[2].alpha = par->mat2_alpha;
  usr->mat[3].alpha = par->mat3_alpha;
  usr->mat[4].alpha = par->mat4_alpha;
  usr->mat[5].alpha = par->mat5_alpha;

  usr->mat[0].cp = par->mat0_cp;
  usr->mat[1].cp = par->mat1_cp;
  usr->mat[2].cp = par->mat2_cp;
  usr->mat[3].cp = par->mat3_cp;
  usr->mat[4].cp = par->mat4_cp;
  usr->mat[5].cp = par->mat5_cp;

  usr->mat[0].kT = par->mat0_kT;
  usr->mat[1].kT = par->mat1_kT;
  usr->mat[2].kT = par->mat2_kT;
  usr->mat[3].kT = par->mat3_kT;
  usr->mat[4].kT = par->mat4_kT;
  usr->mat[5].kT = par->mat5_kT;

  usr->mat[0].kappa = par->mat0_kappa;
  usr->mat[1].kappa = par->mat1_kappa;
  usr->mat[2].kappa = par->mat2_kappa;
  usr->mat[3].kappa = par->mat3_kappa;
  usr->mat[4].kappa = par->mat4_kappa;
  usr->mat[5].kappa = par->mat5_kappa;

  usr->mat[0].eta0 = par->mat0_eta0;
  usr->mat[1].eta0 = par->mat1_eta0;
  usr->mat[2].eta0 = par->mat2_eta0;
  usr->mat[3].eta0 = par->mat3_eta0;
  usr->mat[4].eta0 = par->mat4_eta0;
  usr->mat[5].eta0 = par->mat5_eta0;

  usr->mat[0].zeta0 = par->mat0_zeta0;
  usr->mat[1].zeta0 = par->mat1_zeta0;
  usr->mat[2].zeta0 = par->mat2_zeta0;
  usr->mat[3].zeta0 = par->mat3_zeta0;
  usr->mat[4].zeta0 = par->mat4_zeta0;
  usr->mat[5].zeta0 = par->mat5_zeta0;

  usr->mat[0].eta_func = par->mat0_eta_function;
  usr->mat[1].eta_func = par->mat1_eta_function;
  usr->mat[2].eta_func = par->mat2_eta_function;
  usr->mat[3].eta_func = par->mat3_eta_function;
  usr->mat[4].eta_func = par->mat4_eta_function;
  usr->mat[5].eta_func = par->mat5_eta_function;

  usr->mat[0].zeta_func = par->mat0_zeta_function;
  usr->mat[1].zeta_func = par->mat1_zeta_function;
  usr->mat[2].zeta_func = par->mat2_zeta_function;
  usr->mat[3].zeta_func = par->mat3_zeta_function;
  usr->mat[4].zeta_func = par->mat4_zeta_function;
  usr->mat[5].zeta_func = par->mat5_zeta_function;

  usr->mat[0].G = par->mat0_G;
  usr->mat[1].G = par->mat1_G;
  usr->mat[2].G = par->mat2_G;
  usr->mat[3].G = par->mat3_G;
  usr->mat[4].G = par->mat4_G;
  usr->mat[5].G = par->mat5_G;

  usr->mat[0].Z0 = par->mat0_Z0;
  usr->mat[1].Z0 = par->mat1_Z0;
  usr->mat[2].Z0 = par->mat2_Z0;
  usr->mat[3].Z0 = par->mat3_Z0;
  usr->mat[4].Z0 = par->mat4_Z0;
  usr->mat[5].Z0 = par->mat5_Z0;

  usr->mat[0].C = par->mat0_C;
  usr->mat[1].C = par->mat1_C;
  usr->mat[2].C = par->mat2_C;
  usr->mat[3].C = par->mat3_C;
  usr->mat[4].C = par->mat4_C;
  usr->mat[5].C = par->mat5_C;

  usr->mat[0].sigmat = par->mat0_sigmat;
  usr->mat[1].sigmat = par->mat1_sigmat;
  usr->mat[2].sigmat = par->mat2_sigmat;
  usr->mat[3].sigmat = par->mat3_sigmat;
  usr->mat[4].sigmat = par->mat4_sigmat;
  usr->mat[5].sigmat = par->mat5_sigmat;

  usr->mat[0].theta = par->mat0_theta;
  usr->mat[1].theta = par->mat1_theta;
  usr->mat[2].theta = par->mat2_theta;
  usr->mat[3].theta = par->mat3_theta;
  usr->mat[4].theta = par->mat4_theta;
  usr->mat[5].theta = par->mat5_theta;

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
  NdParams       *nd;
  ScalParams     *scal;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  scal = usr->scal;
  nd   = usr->nd;

  if (usr->par->start_run) {
    // Get date
    ierr = PetscGetDate(date,30); CHKERRQ(ierr);
    ierr = PetscOptionsGetAll(NULL, &opts); CHKERRQ(ierr);

    // Print header and petsc options
    PetscPrintf(usr->comm,"# --------------------------------------- #\n");
    PetscPrintf(usr->comm,"# MID-OCEAN RIDGE - FAULT: %s \n",&(date[0]));
    PetscPrintf(usr->comm,"# --------------------------------------- #\n");
    PetscPrintf(usr->comm,"# PETSc options: %s \n",opts);
    PetscPrintf(usr->comm,"# --------------------------------------- #\n");

    // Free memory
    ierr = PetscFree(opts); CHKERRQ(ierr);

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
  ierr = PetscBagView(usr->bag,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

  // Print scal and nd params
  PetscPrintf(usr->comm,"# --------------------------------------- #\n"); 
  PetscPrintf(usr->comm,"# Characteristic scales:\n");
  PetscPrintf(usr->comm,"#     [x]   = %1.12e (m    ) [v]     = %1.12e (m/s    ) [t]    = %1.12e (s   )\n",scal->x,scal->v,scal->t);
  PetscPrintf(usr->comm,"#     [kphi]= %1.12e (m2   ) [tau]   = %1.12e (Pa     ) [eta]  = %1.12e (Pa.s)\n",scal->kphi,scal->tau,scal->eta);
  PetscPrintf(usr->comm,"#     [rho] = %1.12e (kg/m3) [DT]    = %1.12e (K      ) [kappa]= %1.12e (m2/s)\n",scal->rho,scal->DT,scal->kappa);
 
  PetscPrintf(usr->comm,"# --------------------------------------- #\n");
  PetscPrintf(usr->comm,"# Nondimensional parameters:\n");
  PetscPrintf(usr->comm,"#     delta   = %1.12e \n",nd->delta);
  PetscPrintf(usr->comm,"#     R       = %1.12e \n",nd->R);
  PetscPrintf(usr->comm,"#     Ra      = %1.12e \n",nd->Ra);

  PetscPrintf(usr->comm,"# --------------------------------------- #\n");
  PetscPrintf(usr->comm,"# Material phase parameters (solid):\n");
  PetscInt iph;
  for (iph = 0; iph < usr->nph; iph++) {
    PetscPrintf(usr->comm,"#     MAT PHASE   = %d \n",iph);
    PetscPrintf(usr->comm,"#     [rho0    ] = %1.12e (kg/m3) [alpha    ] = %1.12e (1/K ) [cp      ] = %1.12e (J/kg/K)\n",usr->mat[iph].rho0,usr->mat[iph].alpha,usr->mat[iph].cp);
    PetscPrintf(usr->comm,"#     [kT      ] = %1.12e (W/m/K) [kappa    ] = %1.12e (m2/s) \n",usr->mat[iph].kT,usr->mat[iph].kappa);
    PetscPrintf(usr->comm,"#     [eta0    ] = %1.12e (Pa.s ) [zeta0    ] = %1.12e (Pa.s) \n",usr->mat[iph].eta0,usr->mat[iph].zeta0);
    PetscPrintf(usr->comm,"#     [G0      ] = %1.12e (Pa   ) [Z0       ] = %1.12e (Pa  ) \n",usr->mat[iph].G,usr->mat[iph].Z0);
    PetscPrintf(usr->comm,"#     [C       ] = %1.12e (Pa   ) [sigmat   ] = %1.12e (Pa  ) [theta   ] = %1.12e (-     )\n",usr->mat[iph].C,usr->mat[iph].sigmat,usr->mat[iph].theta);
    PetscPrintf(usr->comm,"#     [rho_func] = %d [eta_func] = %d [zeta_func] = %d \n",usr->mat[iph].rho_func,usr->mat[iph].eta_func,usr->mat[iph].zeta_func);
    PetscPrintf(usr->comm,"#\n");
  }

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
  Params         *par;
  PetscInt       id;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  par  = usr->par;
  id   = usr->par->matid_default;

  // allocate memory
  ierr = PetscMalloc1(1, &scal); CHKERRQ(ierr);

  scal->x     = par->H;
  scal->eta   = usr->mat[id].eta0;
  scal->rho   = par->drho;
  scal->v     = scal->rho*usr->par->g*scal->x*scal->x/scal->eta;
  scal->t     = scal->x/scal->v;
  scal->DT    = usr->par->Tbot-usr->par->Ttop;
  scal->tau   = scal->eta*scal->v/scal->x; //scal->rho*usr->par->g*scal->x;
  scal->kappa = usr->mat[id].kappa;
  scal->kT    = usr->mat[id].kT;
  scal->kphi  = par->kphi0;

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
  PetscInt       iph;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  scal = usr->scal;
  par  = usr->par;

  // allocate memory
  ierr = PetscMalloc1(1, &nd); CHKERRQ(ierr);
  nd->istep = 0;

  // transform to SI units necessary params
  nd->Vext  = par->Vext*1.0e-2/SEC_YEAR; //[cm/yr] to [m/s]
  nd->tmax  = par->tmax*SEC_YEAR;      //[yr] to [s]
  nd->dtmax = par->dtmax*SEC_YEAR;     //[yr] to [s]
  nd->dt_out= par->dt_out*SEC_YEAR;    //[yr] to [s]

  // non-dimensionalize
  nd->xmin  = nd_param(par->xmin,scal->x);
  nd->zmin  = nd_param(par->zmin,scal->x);
  nd->H     = nd_param(par->H,scal->x);
  nd->L     = nd_param(par->L,scal->x);
  nd->Hs    = nd_param(par->Hs,scal->x);
  nd->Vext  = nd_param(nd->Vext,scal->v);
  nd->Vin   = 2.0*nd->Vext*nd->H/nd->L;
  nd->Tbot  = nd_paramT(par->Tbot,par->Ttop,scal->DT);
  nd->Ttop  = nd_paramT(par->Ttop,par->Ttop,scal->DT);

  nd->eta_min = nd_param(par->eta_min,scal->eta);
  nd->eta_max = nd_param(par->eta_max,scal->eta);
  nd->eta_K   = nd_param(par->eta_K ,scal->eta);

  nd->tmax  = nd_param(nd->tmax,scal->t);
  nd->dtmax = nd_param(nd->dtmax,scal->t);
  nd->dt_out= nd_param(nd->dt_out,scal->t);

  nd->dt    = 0.0;
  nd->t     = 0.0;
  nd->dzin  = 0.0;

  // non-dimensional parameters
  nd->delta = PetscSqrtScalar(scal->eta*scal->kphi/usr->par->mu);
  nd->R     = nd->delta/scal->x;
  nd->Ra    = scal->v*scal->x/scal->kappa; // depending on scal->v, Ra = drho*g*L^3/(kappa*eta0)

  usr->nd = nd;

  // scale material parameters
  for (iph = 0; iph < usr->nph; iph++) {
    usr->mat_nd[iph].rho0     = nd_param(usr->mat[iph].rho0,scal->rho);
    usr->mat_nd[iph].cp       = nd_param(usr->mat[iph].cp,scal->kT/scal->kappa/scal->rho);
    usr->mat_nd[iph].kT       = nd_param(usr->mat[iph].kT,scal->kT);
    usr->mat_nd[iph].kappa    = nd_param(usr->mat[iph].kappa,scal->kappa);
    usr->mat_nd[iph].eta0     = nd_param(usr->mat[iph].eta0,scal->eta);
    usr->mat_nd[iph].zeta0    = nd_param(usr->mat[iph].zeta0,scal->eta);
    usr->mat_nd[iph].G        = nd_param(usr->mat[iph].G,scal->tau);
    usr->mat_nd[iph].Z0       = nd_param(usr->mat[iph].Z0,scal->tau);
    usr->mat_nd[iph].C        = nd_param(usr->mat[iph].C,scal->tau);
    usr->mat_nd[iph].sigmat   = nd_param(usr->mat[iph].sigmat,scal->tau);
    usr->mat_nd[iph].theta    = usr->mat[iph].theta;
    usr->mat_nd[iph].alpha    = usr->mat[iph].alpha;
    usr->mat_nd[iph].rho_func = usr->mat[iph].rho_func;
    usr->mat_nd[iph].eta_func = usr->mat[iph].eta_func;
    usr->mat_nd[iph].zeta_func= usr->mat[iph].zeta_func;
  }

  PetscFunctionReturn(0);
}
