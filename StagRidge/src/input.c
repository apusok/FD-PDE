#include "stagridge.h"

// ---------------------------------------
// InputParameters
// ---------------------------------------
PetscErrorCode InputParameters(SolverCtx **psol)
{
  SolverCtx     *sol;
  PetscBag       bag;
  UsrData       *usr;
  GridData      *grd;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // allocate memory to application context
  ierr = PetscMalloc1(1, &sol); CHKERRQ(ierr);
  ierr = PetscMalloc1(1, &grd); CHKERRQ(ierr);

  // Get comm and rank
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
  ierr = PetscBagRegisterInt(bag, &usr->nx, 20, "nx", "Element count in the x-dir"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &usr->nz, 20, "nz", "Element count in the z-dir"); CHKERRQ(ierr);

  ierr = PetscBagRegisterScalar(bag, &usr->xmin, 0.0, "xmin", "Start coordinate of domain in x-dir"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &usr->zmin, 0.0, "zmin", "Start coordinate of domain in z-dir"); CHKERRQ(ierr);

  ierr = PetscBagRegisterScalar(bag, &usr->L, 1.0, "L", "Length of domain in x-dir"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &usr->H, 1.0, "H", "Height of domain in z-dir"); CHKERRQ(ierr);

  // this should be adapted 
  ierr = PetscBagRegisterScalar(bag, &usr->g   , 1.0, "g   ", "Gravitational acceleration"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &usr->eta0, 1.0, "eta0", "Reference viscosity");        CHKERRQ(ierr);

  ierr = PetscBagRegisterInt(bag, &usr->ndisl, 0, "ndisl", "Power exponent for dislocation creep"); CHKERRQ(ierr);

  // Model type
  ierr = PetscBagRegisterInt(bag, &usr->mtype, 0, "mtype", "Model type: 0 - SOLCX, 1 - MOR"); CHKERRQ(ierr);

  // Boundary conditions
  ierr = PetscBagRegisterInt(bag, &usr->bcleft, 0, "bcleft", "LEFT Boundary condition type: 0 - FREE_SLIP, 1 - NO_SLIP" ); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &usr->bcright,0, "bcright","RIGHT Boundary condition type: 0 - FREE_SLIP, 1 - NO_SLIP"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &usr->bcup,   0, "bcup",   "UP Boundary condition type: 0 - FREE_SLIP, 1 - NO_SLIP"   ); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &usr->bcdown, 0, "bcdown", "DOWN Boundary condition type: 0 - FREE_SLIP, 1 - NO_SLIP" ); CHKERRQ(ierr);

  // Input/output
  ierr = PetscBagRegisterString(bag,&usr->fname_in ,FNAME_LENGTH,"null","input_file", "Name for input file, set with: -input_file <filename>"  ); CHKERRQ(ierr);
  ierr = PetscBagRegisterString(bag,&usr->fname_out,FNAME_LENGTH,"null","output_file","Name for output file, set with: -output_file <filename>"); CHKERRQ(ierr);

  // ---------------------------------------
  // Initialize grid variables
  // ---------------------------------------
  
  // Grid context - contains scaled grid defined by user options
  grd->nx = usr->nx;
  grd->nz = usr->nz;
  grd->xmin = usr->xmin;
  grd->zmin = usr->zmin;
  grd->xmax = usr->xmin + usr->L;
  grd->zmax = usr->zmin + usr->H;
  grd->dx   = (grd->xmax - grd->xmin)/(grd->nx);
  grd->dz   = (grd->zmax - grd->zmin)/(grd->nz);

  // stencil 
  grd->dofPV0 = 0; grd->dofPV1 = 1; grd->dofPV2 = 1; // Vx, Vz, P
  grd->dofCf0 = 0; grd->dofCf1 = 0; grd->dofCf2 = 1; // rho_element
  grd->stencilWidth = 1;

  // dofs
  grd->dofV   = (grd->nx+1)*grd->nz + grd->nx*(grd->nz+1);
  grd->dofP   = grd->nx*grd->nz;

  // boundary conditions
  if (usr->bcleft  == 0) grd->bcleft = FREE_SLIP;
  if (usr->bcright == 0) grd->bcright= FREE_SLIP;
  if (usr->bcup    == 0) grd->bcup   = FREE_SLIP;
  if (usr->bcdown  == 0) grd->bcdown = FREE_SLIP;

  grd->Vleft  = 0.0;
  grd->Vright = 0.0;
  grd->Vup    = 0.0;
  grd->Vdown  = 0.0;

  // return pointers
  sol->grd = grd;
  *psol    = sol;

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

  // Print usr bag
  ierr = PetscBagView(sol->bag,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
  PetscPrintf(sol->comm,"# --------------------------------------- #\n");

  // Free memory
  ierr = PetscFree(opts); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}