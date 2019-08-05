#include "stagridge.h"

// ---------------------------------------
// InputParameters
// ---------------------------------------
PetscErrorCode InputParameters(SolverCtx **psol)
{
  SolverCtx     *sol;
  PetscBag       bag;
  ScalData      *scal;
  UsrData       *usr;
  GridData      *grd;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // allocate memory to application context
  ierr = PetscMalloc1(1, &sol ); CHKERRQ(ierr);
  ierr = PetscMalloc1(1, &grd ); CHKERRQ(ierr);
  ierr = PetscMalloc1(1, &scal); CHKERRQ(ierr);

  // Get time, comm and rank
  sol->comm = PETSC_COMM_WORLD;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &sol->rank); CHKERRQ(ierr);

  // Create bag
  ierr = PetscBagCreate (sol->comm,sizeof(UsrData),&sol->bag); CHKERRQ(ierr);
  ierr = PetscBagGetData(sol->bag,(void **)&sol->usr);         CHKERRQ(ierr);   
  ierr = PetscBagSetName(sol->bag,"ParameterBag","- User defined Parameters for StagRidge -"); CHKERRQ(ierr);

  // Define some pointers for easy access
  bag = sol->bag;
  usr = sol->usr;

  // ---------------------------------------
  // Initialize default (user) variables
  // ---------------------------------------
  // Domain
  ierr = PetscBagRegisterInt(bag, &usr->nx, 20, "nx", "Element count in the x-dir"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &usr->nz, 20, "nz", "Element count in the z-dir"); CHKERRQ(ierr);

  ierr = PetscBagRegisterScalar(bag, &usr->xmin, 0.0, "xmin", "Start coordinate of domain in x-dir [km]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &usr->zmin, 0.0, "zmin", "Start coordinate of domain in z-dir [km]"); CHKERRQ(ierr);

  ierr = PetscBagRegisterScalar(bag, &usr->L, 1.0, "L", "Length of domain in x-dir [km]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &usr->H, 1.0, "H", "Height of domain in z-dir [km]"); CHKERRQ(ierr);

  // Physical parameters
  ierr = PetscBagRegisterScalar(bag, &usr->g, 1.0, "g", "Gravitational acceleration [m/s2]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &usr->u0, 1.0, "u0", "Half-spreading rate [cm/yr]");     CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &usr->rangle, 0.0, "rangle", "Ridge angle [deg]");       CHKERRQ(ierr);

  // Material Parameters and Rheology
  ierr = PetscBagRegisterScalar(bag, &usr->rho0, 1.0, "rho0", "Reference density [kg/m3]");      CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &usr->eta0, 1.0, "eta0", "Reference viscosity [Pa.s]");     CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &usr->ndisl, 0.0, "ndisl", "Power exponent for dislocation creep"); CHKERRQ(ierr);

  // Dimensional/dimensionless
  ierr = PetscBagRegisterInt(bag, &usr->dim, 0, "dim", "Dimensions: 0-dimensionless 1-dimensional"); CHKERRQ(ierr);

  // Model type
  ierr = PetscBagRegisterInt(bag, &usr->mtype, 0, "mtype", "Model type: 0-SOLCX, 1-SOLCX_EFF, 2-MOR_ANALYTIC"); CHKERRQ(ierr);

  // benchmarks
  ierr = PetscBagRegisterInt(bag, &usr->tests, 0, "tests", "Test benchmarks: 0 - NO, 1 - YES"); CHKERRQ(ierr);

  // SolCx parameters
  ierr = PetscBagRegisterScalar(bag, &usr->solcx_eta0, 1.0, "solcx_eta0", "SolCx benchmark: eta0"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &usr->solcx_eta1, 1.0, "solcx_eta1", "SolCx benchmark: eta1"); CHKERRQ(ierr);

  // Boundary conditions
  ierr = PetscBagRegisterInt(bag, &usr->bcleft, 0, "bcleft", "LEFT Boundary condition type: 0-FREE_SLIP, 1-NO_SLIP" ); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &usr->bcright,0, "bcright","RIGHT Boundary condition type: 0-FREE_SLIP, 1-NO_SLIP"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &usr->bcup,   0, "bcup",   "UP Boundary condition type: 0-FREE_SLIP, 1-NO_SLIP"   ); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &usr->bcdown, 0, "bcdown", "DOWN Boundary condition type: 0-FREE_SLIP, 1-NO_SLIP" ); CHKERRQ(ierr);

  // Input/output
  ierr = PetscBagRegisterString(bag,&usr->fname_out,FNAME_LENGTH,"output","output_file","Name for output file, set with: -output_file <filename>"); CHKERRQ(ierr);
  
  // Other variables
  usr->fname_in[0] = '\0';

  // MOR Analytic
  if (usr->mtype == 2) {
    usr->mor_radalpha = usr->rangle*PETSC_PI/180;
    usr->mor_sina = PetscSinScalar(usr->mor_radalpha);
    usr->mor_C1 = 2*usr->mor_sina*usr->mor_sina/(PETSC_PI-2*usr->mor_radalpha-PetscSinScalar(2*usr->mor_radalpha));
    usr->mor_C4 = -2/(PETSC_PI-2*usr->mor_radalpha-PetscSinScalar(2*usr->mor_radalpha));
  }

  // ---------------------------------------
  // Scaling variables
  // ---------------------------------------
  // Characteristic values
  if (usr->dim == 1) {
    // length [km]
    scal->charL = PetscMax(usr->L, usr->H);

    // gravity [m/s2] and derived
    scal->charg = 10.0;
    scal->chart = PetscPowScalar(scal->charL*1e3/scal->charg,0.5); //[s]
    scal->charv = scal->charL*1e5/scal->chart/SEC_YEAR; //[cm/yr]

    // viscosity [Pa.s = N.s/m2 = kg/m/s2] and derived
    scal->chareta = usr->eta0; 
    scal->charrho = scal->chareta/scal->charg/scal->charL/1e3; //[kg/m3]

  } else {
    scal->charL   = 1.0;
    scal->charg   = 1.0;
    scal->chareta = 1.0;
    scal->chart   = 1.0;
    scal->charv   = 1.0;
    scal->charrho = 1.0;
  }

  // Scale parameters
  scal->eta0 = usr->eta0/scal->chareta;
  scal->g    = usr->g/scal->charg;
  scal->u0   = usr->u0/scal->charv;
  scal->rho0 = usr->rho0/scal->charrho;

  PetscPrintf(PETSC_COMM_SELF,"# Characteristic scales:\n");
  PetscPrintf(PETSC_COMM_SELF,"#     charL=%1.12e charg=%1.12e chareta=%1.12e\n",scal->charL,scal->charg,scal->chareta);
  PetscPrintf(PETSC_COMM_SELF,"#     chart=%1.12e charv=%1.12e charrho=%1.12e\n",scal->chart,scal->charv,scal->charrho);

  // ---------------------------------------
  // Initialize grid variables
  // ---------------------------------------
  
  // Grid context - contains scaled grid defined by user options
  grd->nx = usr->nx;
  grd->nz = usr->nz;
  grd->xmin = usr->xmin/scal->charL;
  grd->zmin = usr->zmin/scal->charL;
  grd->xmax = (usr->xmin + usr->L)/scal->charL;
  grd->zmax = (usr->zmin + usr->H)/scal->charL;
  grd->dx   = (grd->xmax - grd->xmin)/(grd->nx);
  grd->dz   = (grd->zmax - grd->zmin)/(grd->nz);

  // stencil 
  grd->dofPV0 = 0; grd->dofPV1 = 1; grd->dofPV2 = 1; // Vx, Vz, P
  grd->dofCf0 = 0; grd->dofCf1 = 1; grd->dofCf2 = 1; // rho_element, edges
  grd->stencilWidth = 1;

  // dofs
  grd->dofV   = (grd->nx+1)*grd->nz + grd->nx*(grd->nz+1);
  grd->dofP   = grd->nx*grd->nz;

  // boundary conditions
  if (usr->bcleft  == 0) grd->bcleft = FREE_SLIP;
  if (usr->bcright == 0) grd->bcright= FREE_SLIP;
  if (usr->bcup    == 0) grd->bcup   = FREE_SLIP;
  if (usr->bcdown  == 0) grd->bcdown = FREE_SLIP;

  // model type
  if (usr->mtype == 0) grd->mtype = SOLCX;
  if (usr->mtype == 1) grd->mtype = SOLCX_EFF;
  if (usr->mtype == 2) grd->mtype = MOR;
  
  // return pointers
  sol->scal = scal;
  sol->grd  = grd;
  *psol     = sol;

  PetscFunctionReturn(0);
}

// ---------------------------------------
// InputPrintData
// ---------------------------------------
PetscErrorCode InputPrintData(SolverCtx *sol)
{
  char           date[30], *opts;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  
  // Get date
  ierr = PetscGetDate(date,30); CHKERRQ(ierr);

  // Get petsc command options
  ierr = PetscOptionsGetAll(NULL, &opts); CHKERRQ(ierr);

  // Print header and petsc options
  PetscPrintf(sol->comm,"# --------------------------------------- #\n");
  PetscPrintf(sol->comm,"# StagRidge: %s \n",&(date[0]));
  PetscPrintf(sol->comm,"# --------------------------------------- #\n");
  PetscPrintf(sol->comm,"# PETSc options: %s \n",opts);
  PetscPrintf(sol->comm,"# --------------------------------------- #\n");

  // Input file info
  if (sol->usr->fname_in[0] == '\0') { // string is empty
    PetscPrintf(sol->comm,"# Input options file: NONE (using default options)\n");
  }
  else {
    PetscPrintf(sol->comm,"# Input options file: %s \n",sol->usr->fname_in);
  }
  PetscPrintf(sol->comm,"# --------------------------------------- #\n");

  // Print usr bag
  ierr = PetscBagView(sol->bag,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
  PetscPrintf(sol->comm,"# --------------------------------------- #\n");

  // Free memory
  ierr = PetscFree(opts); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}