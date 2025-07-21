// ---------------------------------------
// ADVDIFF benchmarks from Elman (2005): 
// Example 3.1.1 - Solves the convection-diffusion equation with zero source term, constant vertical velocity and an exponential boundary layer.
// Example 3.1.2 - Zero source term, variable vertical wind, characteristic boundary layers (Neumann BC).
// Example 3.1.3 - Zero source term, constant wind at a 30◦ angle to the left of vertical, downstream boundary layer and interior layer.
// Example 3.1.4 - Zero source term, recirculating wind, characteristic boundary layers.
// run: ./test_advdiff_elman.sh -pc_type lu -pc_factor_mat_solver_type umfpack -pc_factor_mat_ordering_type external -nx 10 -nz 10 -log_view
// python test: ./python/test_advdiff_elman.py
// ---------------------------------------
static char help[] = "Application (examples from Elman 2005) to solve the convection diffusion equation (ADVDIFF) with FD-PDE \n\n";

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

#include "../src/fdpde_advdiff.h"

// ---------------------------------------
// Application Context
// ---------------------------------------
#define FNAME_LENGTH 200

// parameters (bag)
typedef struct {
  PetscInt       nx, nz, advtype;
  PetscScalar    L, H;
  PetscScalar    xmin, zmin;
  PetscScalar    k, rho, cp, ux, uz;
  char           fname_out[FNAME_LENGTH]; 
  char           fname_in [FNAME_LENGTH];  
  char           fdir_out[FNAME_LENGTH]; 
} Params;

// user defined and model-dependent variables
typedef struct {
  Params        *par;
  PetscBag       bag;
  MPI_Comm       comm;
  PetscMPIInt    rank;
} UsrData;

// ---------------------------------------
// Function definitions
// ---------------------------------------
PetscErrorCode InputParameters(UsrData**);
PetscErrorCode InputPrintData(UsrData*);
PetscErrorCode Analytic_Elman311(DM,Vec*,void*);
PetscErrorCode Numerical_Elman311(DM*,Vec*,void*);
PetscErrorCode Numerical_Elman312(DM*,Vec*,void*);
PetscErrorCode Numerical_Elman313(DM*,Vec*,void*);
PetscErrorCode Numerical_Elman314(DM*,Vec*,void*);
PetscErrorCode FormCoefficient_Elman311(FDPDE,DM,Vec,DM,Vec,void*);
PetscErrorCode FormCoefficient_Elman312(FDPDE,DM,Vec,DM,Vec,void*);
PetscErrorCode FormCoefficient_Elman313(FDPDE,DM,Vec,DM,Vec,void*);
PetscErrorCode FormCoefficient_Elman314(FDPDE,DM,Vec,DM,Vec,void*);
PetscErrorCode FormBCList_Elman311(DM,Vec,DMStagBCList,void*);
PetscErrorCode FormBCList_Elman312(DM,Vec,DMStagBCList,void*);
PetscErrorCode FormBCList_Elman313(DM,Vec,DMStagBCList,void*);
PetscErrorCode FormBCList_Elman314(DM,Vec,DMStagBCList,void*);

// ---------------------------------------
// Some descriptions
// ---------------------------------------
const char coeff_description311[] =
"  << ADVDIFF - Coefficients for Ex 3.1.1 Elman (2005) >> \n"
"  A = rho*cp (element), rho = 1, cp = 1\n"
"  B = k (edge), k = 0.005\n"
"  C = 0 (element)\n"
"  u = [ux, uz] (edge), u = [0,1]\n";

const char bc_description311[] =
"  << BCs for Ex 3.1.1 Elman (2005) >> \n"
"  LEFT: T(-1,z) = -1\n"
"  RIGHT: T(1,z) = 1\n" 
"  DOWN: T(x,-1) = x\n" 
"  UP: T(x,1) = 0 \n";

const char coeff_description312[] =
"  << ADVDIFF - Coefficients for Ex 3.1.2 Elman (2005) >> \n"
"  A = rho*cp (element), rho = 1, cp = 1\n"
"  B = k (edge), k = 0.005\n"
"  C = 0 (element)\n"
"  u = [ux, uz] (edge), u = [0,1+(x+1)^2/4]\n";

const char bc_description312[] =
"  << BCs for Ex 3.1.2 Elman (2005) >> \n"
"  LEFT: T = (1-(1+z)/2)^3 \n"
"  RIGHT: T = (1-(1+z)/2)^2 \n" 
"  DOWN: T = 1 \n" 
"  UP: dT/dz = 0 \n";

const char coeff_description313[] =
"  << ADVDIFF - Coefficients for Ex 3.1.3 Elman (2005) >> \n"
"  A = rho*cp (element), rho = 1, cp = 1\n"
"  B = k (edge), k = 0.005\n"
"  C = 0 (element)\n"
"  u = [ux, uz] (edge), u = [-sin(pi/6),cos(pi/6)]\n";

const char bc_description313[] =
"  << BCs for Ex 3.1.3 Elman (2005) >> \n"
"  LEFT: T = 0\n"
"  RIGHT: T = 1\n" 
"  DOWN: T = 0 if x<0, 1 if x>=0\n" 
"  UP: T = 0 \n";

const char coeff_description314[] =
"  << ADVDIFF - Coefficients for Ex 3.1.4 Elman (2005) >> \n"
"  A = rho*cp (element), rho = 1, cp = 1\n"
"  B = k (edge), k = 0.005\n"
"  C = 0 (element)\n"
"  u = [ux, uz] (edge), u = [2z(1-x^2),-2x(1-z^2)]\n";

const char bc_description314[] =
"  << BCs for Ex 3.1.4 Elman (2005) >> \n"
"  LEFT: T = 0\n"
"  RIGHT: T = 1\n" 
"  DOWN: T = 0\n" 
"  UP: T = 0 \n";

// ---------------------------------------
// Application functions
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "Numerical_Elman311"
PetscErrorCode Numerical_Elman311(DM *_dm, Vec *_x, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  FDPDE          fd;
  DM             dm;
  Vec            x;
  PetscInt       nx, nz;
  PetscScalar    dx, dz,xmin, zmin, xmax, zmax;
  char           fout[FNAME_LENGTH];
  PetscFunctionBeginUser;

  // Element count
  nx = usr->par->nx;
  nz = usr->par->nz;

  // Domain coords
  dx = usr->par->L/(2*nx-2);
  dz = usr->par->H/(2*nz-2);

  // modify coord of dm such that unknowns are located on the boundaries limits (-1,1)
  xmin = usr->par->xmin-dx;
  zmin = usr->par->zmin-dz;
  xmax = usr->par->xmin+usr->par->L+dx;
  zmax = usr->par->zmin+usr->par->H+dz;

  // Create the FD-pde object
  PetscCall(FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_ADVDIFF,&fd));
  PetscCall(FDPDESetUp(fd));

  // Set advection type
  if (usr->par->advtype==0) { PetscCall(FDPDEAdvDiffSetAdvectSchemeType(fd,ADV_UPWIND)); }
  if (usr->par->advtype==1) { PetscCall(FDPDEAdvDiffSetAdvectSchemeType(fd,ADV_FROMM)); }
  
  PetscCall(FDPDEAdvDiffSetTimeStepSchemeType(fd,TS_NONE));

  // Set BC evaluation function
  PetscCall(FDPDESetFunctionBCList(fd,FormBCList_Elman311,bc_description311,NULL)); 

  // Set coefficients evaluation function
  PetscCall(FDPDESetFunctionCoefficient(fd,FormCoefficient_Elman311,coeff_description311,usr)); 

  PetscCall(FDPDEView(fd)); 

  // FD SNES Solver
  PetscCall(FDPDESolve(fd,NULL));

  // Get solution vector
  PetscCall(FDPDEGetSolution(fd,&x)); 
  PetscCall(FDPDEGetDM(fd, &dm)); 

  // Output solution to file
  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s",usr->par->fdir_out,usr->par->fname_out));
  PetscCall(DMStagViewBinaryPython(dm,x,fout));

  // Destroy FD-PDE object
  PetscCall(FDPDEDestroy(&fd));

  *_x  = x;
  *_dm = dm;

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// Numerical_Elman312
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "Numerical_Elman312"
PetscErrorCode Numerical_Elman312(DM *_dm, Vec *_x, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  FDPDE          fd;
  DM             dm;
  Vec            x;
  PetscInt       nx, nz;
  PetscScalar    dx, dz,xmin, zmin, xmax, zmax;
  char           fout[FNAME_LENGTH];
  PetscFunctionBeginUser;

  // Element count
  nx = usr->par->nx;
  nz = usr->par->nz;

  // Domain coords
  dx = usr->par->L/(2*nx-2);
  dz = usr->par->H/(2*nz-2);

  // modify coord of dm such that unknowns are located on the boundaries limits (-1,1)
  xmin = usr->par->xmin-dx;
  zmin = usr->par->zmin-dz;
  xmax = usr->par->xmin+usr->par->L+dx;
  zmax = usr->par->zmin+usr->par->H+dz;

  // Create the FD-pde object
  PetscCall(FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_ADVDIFF,&fd));
  PetscCall(FDPDESetUp(fd));
  PetscCall(FDPDEAdvDiffSetAdvectSchemeType(fd,ADV_UPWIND));
  PetscCall(FDPDEAdvDiffSetTimeStepSchemeType(fd,TS_NONE));

  // Set BC evaluation function
  PetscCall(FDPDESetFunctionBCList(fd,FormBCList_Elman312,bc_description312,NULL)); 

  // Set coefficients evaluation function
  PetscCall(FDPDESetFunctionCoefficient(fd,FormCoefficient_Elman312,coeff_description312,usr)); 

  PetscCall(FDPDEView(fd)); 

  // FD SNES Solver
  PetscCall(FDPDESolve(fd,NULL));

  // Get solution vector
  PetscCall(FDPDEGetSolution(fd,&x)); 
  PetscCall(FDPDEGetDM(fd, &dm)); 

  // Output solution to file
  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s",usr->par->fdir_out,"out_num_solution_elman312"));
  PetscCall(DMStagViewBinaryPython(dm,x,fout));

  // Destroy FD-PDE object
  PetscCall(FDPDEDestroy(&fd));

  *_x  = x;
  *_dm = dm;

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// Numerical_Elman313
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "Numerical_Elman313"
PetscErrorCode Numerical_Elman313(DM *_dm, Vec *_x, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  FDPDE          fd;
  DM             dm;
  Vec            x;
  PetscInt       nx, nz;
  PetscScalar    dx, dz,xmin, zmin, xmax, zmax;
  char           fout[FNAME_LENGTH];
  PetscFunctionBeginUser;

  // Element count
  nx = usr->par->nx;
  nz = usr->par->nz;

  // Domain coords
  dx = usr->par->L/(2*nx-2);
  dz = usr->par->H/(2*nz-2);

  // modify coord of dm such that unknowns are located on the boundaries limits (-1,1)
  xmin = usr->par->xmin-dx;
  zmin = usr->par->zmin-dz;
  xmax = usr->par->xmin+usr->par->L+dx;
  zmax = usr->par->zmin+usr->par->H+dz;

  // Create the FD-pde object
  PetscCall(FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_ADVDIFF,&fd));
  PetscCall(FDPDESetUp(fd));
  PetscCall(FDPDEAdvDiffSetAdvectSchemeType(fd,ADV_UPWIND));
  PetscCall(FDPDEAdvDiffSetTimeStepSchemeType(fd,TS_NONE));

  // Set BC evaluation function
  PetscCall(FDPDESetFunctionBCList(fd,FormBCList_Elman313,bc_description313,NULL)); 

  // Set coefficients evaluation function
  PetscCall(FDPDESetFunctionCoefficient(fd,FormCoefficient_Elman313,coeff_description313,usr)); 

  PetscCall(FDPDEView(fd)); 

  // FD SNES Solver
  PetscCall(FDPDESolve(fd,NULL));

  // Get solution vector
  PetscCall(FDPDEGetSolution(fd,&x)); 
  PetscCall(FDPDEGetDM(fd, &dm)); 

  // Output solution to file
  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s",usr->par->fdir_out,"out_num_solution_elman313"));
  PetscCall(DMStagViewBinaryPython(dm,x,fout));

  // Destroy FD-PDE object
  PetscCall(FDPDEDestroy(&fd));

  *_x  = x;
  *_dm = dm;

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// Numerical_Elman314
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "Numerical_Elman314"
PetscErrorCode Numerical_Elman314(DM *_dm, Vec *_x, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  FDPDE          fd;
  DM             dm;
  Vec            x;
  PetscInt       nx, nz;
  PetscScalar    dx, dz,xmin, zmin, xmax, zmax;
  char           fout[FNAME_LENGTH];
  PetscFunctionBeginUser;

  // Element count
  nx = usr->par->nx;
  nz = usr->par->nz;

  // Domain coords
  dx = usr->par->L/(2*nx-2);
  dz = usr->par->H/(2*nz-2);

  // modify coord of dm such that unknowns are located on the boundaries limits (-1,1)
  xmin = usr->par->xmin-dx;
  zmin = usr->par->zmin-dz;
  xmax = usr->par->xmin+usr->par->L+dx;
  zmax = usr->par->zmin+usr->par->H+dz;

  // Create the FD-pde object
  PetscCall(FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_ADVDIFF,&fd));
  PetscCall(FDPDESetUp(fd));
  PetscCall(FDPDEAdvDiffSetAdvectSchemeType(fd,ADV_UPWIND));
  PetscCall(FDPDEAdvDiffSetTimeStepSchemeType(fd,TS_NONE));

  // Set BC evaluation function
  PetscCall(FDPDESetFunctionBCList(fd,FormBCList_Elman314,bc_description314,NULL)); 

  // Set coefficients evaluation function
  PetscCall(FDPDESetFunctionCoefficient(fd,FormCoefficient_Elman314,coeff_description314,usr)); 
  PetscCall(FDPDEView(fd)); 

  // FD SNES Solver
  PetscCall(FDPDESolve(fd,NULL));

  // Get solution vector
  PetscCall(FDPDEGetSolution(fd,&x)); 
  PetscCall(FDPDEGetDM(fd, &dm)); 

  // Output solution to file
  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s",usr->par->fdir_out,"out_num_solution_elman314"));
  PetscCall(DMStagViewBinaryPython(dm,x,fout));

  // Destroy FD-PDE object
  PetscCall(FDPDEDestroy(&fd));

  *_x  = x;
  *_dm = dm;

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// FormBCList_Elman311
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormBCList_Elman311"
PetscErrorCode FormBCList_Elman311(DM dm, Vec x, DMStagBCList bclist, void *ctx)
{
  PetscInt    k,n_bc,*idx_bc;
  PetscScalar *value_bc,*x_bc;
  BCType      *type_bc;
  PetscFunctionBeginUser;
  
  // Left: T = -1.0
  PetscCall(DMStagBCListGetValues(bclist,'w','o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = -1.0;
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  
  // RIGHT: T = 1.0
  PetscCall(DMStagBCListGetValues(bclist,'e','o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 1.0;
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));

  // DOWN: T = x
  PetscCall(DMStagBCListGetValues(bclist,'s','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = x_bc[2*k];
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));

  // UP: T = 0
  PetscCall(DMStagBCListGetValues(bclist,'n','o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
 
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// FormBCList_Elman312
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormBCList_Elman312"
PetscErrorCode FormBCList_Elman312(DM dm, Vec x, DMStagBCList bclist, void *ctx)
{
  PetscInt    k,n_bc,*idx_bc;
  PetscScalar *value_bc,*x_bc;
  BCType      *type_bc;
  PetscFunctionBeginUser;

  // Left: T = 1-z^3
  PetscCall(DMStagBCListGetValues(bclist,'w','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = PetscPowReal(1.0-(1.0+x_bc[2*k+1])*0.5,3.0);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
  
  // RIGHT: T = 1.0-z^2
  PetscCall(DMStagBCListGetValues(bclist,'e','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = PetscPowReal(1.0-(1.0+x_bc[2*k+1])*0.5,3.0);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));

  // DOWN: T = 1.0
  PetscCall(DMStagBCListGetValues(bclist,'s','o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 1.0;
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));

  // UP: dT/dz = 0
  PetscCall(DMStagBCListGetValues(bclist,'n','o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
 
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// FormBCList_Elman313
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormBCList_Elman313"
PetscErrorCode FormBCList_Elman313(DM dm, Vec x, DMStagBCList bclist, void *ctx)
{
  PetscInt    k,n_bc,*idx_bc;
  PetscScalar *value_bc,*x_bc;
  BCType      *type_bc;
  PetscFunctionBeginUser;
  
  // Left: T = 0.0
  PetscCall(DMStagBCListGetValues(bclist,'w','o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  
  // RIGHT: T = 1.0
  PetscCall(DMStagBCListGetValues(bclist,'e','o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 1.0;
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));

  // DOWN: T = 0 if x<0, 1 if x>=0
  PetscCall(DMStagBCListGetValues(bclist,'s','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    if (x_bc[2*k]<0) value_bc[k] = 0.0;
    else             value_bc[k] = 1.0;
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));

  // UP: T = 0
  PetscCall(DMStagBCListGetValues(bclist,'n','o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
 
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// FormBCList_Elman314
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormBCList_Elman314"
PetscErrorCode FormBCList_Elman314(DM dm, Vec x, DMStagBCList bclist, void *ctx)
{
  PetscInt    k,n_bc,*idx_bc;
  PetscScalar *value_bc;
  BCType      *type_bc;
  PetscFunctionBeginUser;
  
  // Left: T = 0.0
  PetscCall(DMStagBCListGetValues(bclist,'w','o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  
  // RIGHT: T = 1.0
  PetscCall(DMStagBCListGetValues(bclist,'e','o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 1.0;
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));

  // DOWN: T = 0.0
  PetscCall(DMStagBCListGetValues(bclist,'s','o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));

  // UP: T = 0
  PetscCall(DMStagBCListGetValues(bclist,'n','o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
 
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// FormCoefficient_Elman311
//    Element: A = rho*cp (dof 0), C = heat production/sink (dof 1)
//    Edges: k (dof 0), velocity (dof 1)
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormCoefficient_Elman311"
PetscErrorCode FormCoefficient_Elman311(FDPDE fd, DM dm, Vec x, DM dmcoeff, Vec coeff, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       i, j, sx, sz, nx, nz;
  Vec            coefflocal;
  PetscScalar    rho, k, cp, v[2];
  PetscScalar    ***c;
  PetscFunctionBeginUser;

  // User parameters
  rho = usr->par->rho;
  k   = usr->par->k;
  cp  = usr->par->cp;
  v[0]= usr->par->ux;
  v[1]= usr->par->uz;

  // Get domain corners
  PetscCall(DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 

  // Create coefficient local vector
  PetscCall(DMCreateLocalVector(dmcoeff, &coefflocal)); 
  PetscCall(DMStagVecGetArray(dmcoeff, coefflocal, &c)); 
  
  // Loop over local domain - set initial density and viscosity
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {

      { // A = rho*cp
        DMStagStencil point;
        PetscInt      idx;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 0;
        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
        c[j][i][idx] = rho*cp;
      }

      { // C = 0.0
        DMStagStencil point;
        PetscInt      idx;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 1;
        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
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
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] = k;
        }
      }

      { // u = velocity (edge)
        DMStagStencil point[4];
        PetscInt      ii, idx;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 1;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 1;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = 1;
        point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = 1;

        for (ii = 0; ii < 2; ii++) {
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] = v[0];
        }
        for (ii = 2; ii < 4; ii++) {
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] = v[1];
        }
      }
    }
  }

  // Restore arrays, local vectors
  PetscCall(DMStagVecRestoreArray(dmcoeff,coefflocal,&c));
  PetscCall(DMLocalToGlobalBegin(dmcoeff,coefflocal,INSERT_VALUES,coeff)); 
  PetscCall(DMLocalToGlobalEnd  (dmcoeff,coefflocal,INSERT_VALUES,coeff)); 
  
  PetscCall(VecDestroy(&coefflocal)); 

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// FormCoefficient_Elman312
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormCoefficient_Elman312"
PetscErrorCode FormCoefficient_Elman312(FDPDE fd, DM dm, Vec x, DM dmcoeff, Vec coeff, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       i, j, sx, sz, nx, nz;
  Vec            coefflocal;
  PetscScalar    rho, k, cp;
  PetscScalar    **coordx,**coordz;
  PetscInt       iprev, inext, icenter;
  PetscScalar    ***c;
  PetscFunctionBeginUser;

  // User parameters
  rho = usr->par->rho;
  k   = usr->par->k;
  cp  = usr->par->cp;

  // Get domain corners
  PetscCall(DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 

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

      { // A = rho*cp
        DMStagStencil point;
        PetscInt      idx;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 0;
        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
        c[j][i][idx] = rho*cp;
      }

      { // C = 0.0
        DMStagStencil point;
        PetscInt      idx;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 1;
        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
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
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] = k;
        }
      }

      { // u = velocity (edge)
        DMStagStencil point[4];
        PetscScalar   xp[4];
        PetscInt      ii, idx;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 1;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 1;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = 1;
        point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = 1;

        xp[0] = coordx[i][iprev  ];
        xp[1] = coordx[i][inext  ];
        xp[2] = coordx[i][icenter];
        xp[3] = coordx[i][icenter];

        for (ii = 0; ii < 2; ii++) {
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] = 0.0;
        }
        for (ii = 2; ii < 4; ii++) {
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] = 1.0+(xp[ii]+1.0)*(xp[ii]+1.0)*0.25;
        }
      }
    }
  }

  // Restore arrays, local vectors
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL));
  PetscCall(DMStagVecRestoreArray(dmcoeff,coefflocal,&c));
  PetscCall(DMLocalToGlobalBegin(dmcoeff,coefflocal,INSERT_VALUES,coeff)); 
  PetscCall(DMLocalToGlobalEnd  (dmcoeff,coefflocal,INSERT_VALUES,coeff)); 
  
  PetscCall(VecDestroy(&coefflocal)); 

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// FormCoefficient_Elman313
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormCoefficient_Elman313"
PetscErrorCode FormCoefficient_Elman313(FDPDE fd, DM dm, Vec x, DM dmcoeff, Vec coeff, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       i, j, sx, sz, nx, nz;
  Vec            coefflocal;
  PetscScalar    rho, k, cp, v[2];
  PetscScalar    ***c;
  PetscFunctionBeginUser;

  // User parameters
  rho = usr->par->rho;
  k   = usr->par->k;
  cp  = usr->par->cp;
  v[0]= -PetscSinScalar(PETSC_PI/6);
  v[1]= PetscCosScalar(PETSC_PI/6);

  // Get domain corners
  PetscCall(DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 

  // Create coefficient local vector
  PetscCall(DMCreateLocalVector(dmcoeff, &coefflocal)); 
  PetscCall(DMStagVecGetArray(dmcoeff, coefflocal, &c)); 
  
  // Loop over local domain - set initial density and viscosity
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {

      { // A = rho*cp
        DMStagStencil point;
        PetscInt      idx;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 0;
        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
        c[j][i][idx] = rho*cp;
      }

      { // C = 0.0
        DMStagStencil point;
        PetscInt      idx;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 1;
        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
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
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] = k;
        }
      }

      { // u = velocity (edge)
        DMStagStencil point[4];
        PetscInt      ii, idx;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 1;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 1;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = 1;
        point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = 1;

        for (ii = 0; ii < 2; ii++) {
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] = v[0];
        }
        for (ii = 2; ii < 4; ii++) {
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] = v[1];
        }
      }
    }
  }

  // Restore arrays, local vectors
  PetscCall(DMStagVecRestoreArray(dmcoeff,coefflocal,&c));
  PetscCall(DMLocalToGlobalBegin(dmcoeff,coefflocal,INSERT_VALUES,coeff)); 
  PetscCall(DMLocalToGlobalEnd  (dmcoeff,coefflocal,INSERT_VALUES,coeff)); 
  
  PetscCall(VecDestroy(&coefflocal)); 

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// FormCoefficient_Elman314
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormCoefficient_Elman314"
PetscErrorCode FormCoefficient_Elman314(FDPDE fd, DM dm, Vec x, DM dmcoeff, Vec coeff, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       i, j, sx, sz, nx, nz;
  Vec            coefflocal;
  PetscScalar    rho, k, cp;
  PetscScalar    **coordx,**coordz;
  PetscInt       iprev, inext, icenter;
  PetscScalar    ***c;
  PetscFunctionBeginUser;

  // User parameters
  rho = usr->par->rho;
  k   = usr->par->k;
  cp  = usr->par->cp;

  // Get domain corners
  PetscCall(DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 

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

      { // A = rho*cp
        DMStagStencil point;
        PetscInt      idx;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 0;
        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
        c[j][i][idx] = rho*cp;
      }

      { // C = 0.0
        DMStagStencil point;
        PetscInt      idx;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 1;
        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
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
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] = k;
        }
      }

      { // u = velocity (edge)
        DMStagStencil point[4];
        PetscScalar   xp[4],zp[4];
        PetscInt      ii, idx;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 1;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 1;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = 1;
        point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = 1;

        xp[0] = coordx[i][iprev  ]; zp[0] = coordz[j][icenter];
        xp[1] = coordx[i][inext  ]; zp[1] = coordz[j][icenter];
        xp[2] = coordx[i][icenter]; zp[2] = coordz[j][iprev  ];
        xp[3] = coordx[i][icenter]; zp[3] = coordz[j][inext  ];

        for (ii = 0; ii < 2; ii++) {
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] = 2*zp[ii]*(1-xp[ii]*xp[ii]);
        }
        for (ii = 2; ii < 4; ii++) {
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] = -2*xp[ii]*(1-zp[ii]*zp[ii]);
        }
      }
    }
  }

  // Restore arrays, local vectors
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL));
  PetscCall(DMStagVecRestoreArray(dmcoeff,coefflocal,&c));
  PetscCall(DMLocalToGlobalBegin(dmcoeff,coefflocal,INSERT_VALUES,coeff)); 
  PetscCall(DMLocalToGlobalEnd  (dmcoeff,coefflocal,INSERT_VALUES,coeff)); 
  
  PetscCall(VecDestroy(&coefflocal)); 

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// Create analytical solution
// ---------------------------------------
PetscErrorCode Analytic_Elman311(DM dm,Vec *_x, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  PetscInt       i, j, sx, sz, nx, nz, idx, icenter;
  PetscScalar    ***xx, k;
  PetscScalar    **coordx,**coordz;
  Vec            x, xlocal;
  char           fout[FNAME_LENGTH];
  PetscFunctionBeginUser;

  k = usr->par->k;

  // Create local and global vector associated with DM
  PetscCall(DMCreateGlobalVector(dm, &x     )); 
  PetscCall(DMCreateLocalVector (dm, &xlocal)); 

  // Get array associated with vector
  PetscCall(DMStagVecGetArray(dm,xlocal,&xx)); 

  // Get domain corners
  PetscCall(DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 
  
// Get dm coordinates array
  PetscCall(DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter)); 

  // Loop over local domain to calculate the SolCx analytical solution
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      DMStagStencil  point;
      PetscScalar    xp, zp;

      point.i = i; point.j = j; point.loc = ELEMENT; point.c = 0; 
      xp = coordx[i][icenter];
      zp = coordz[j][icenter];

      PetscCall(DMStagGetLocationSlot(dm, point.loc, point.c, &idx));
      xx[j][i][idx] = xp*(1-PetscExpReal((zp-1)/k))/(1-PetscExpReal(-2/k));
    }
  }

  // Restore arrays
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));
  PetscCall(DMStagVecRestoreArray(dm,xlocal,&xx)); 

  // Map local to global
  PetscCall(DMLocalToGlobalBegin(dm,xlocal,INSERT_VALUES,x)); 
  PetscCall(DMLocalToGlobalEnd  (dm,xlocal,INSERT_VALUES,x)); 

  PetscCall(VecDestroy(&xlocal)); 

  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s",usr->par->fdir_out,"out_analytic_solution_elman311"));
  PetscCall(DMStagViewBinaryPython(dm,x,fout));

  // Assign pointers
  *_x  = x;
  
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
  PetscCall(PetscBagRegisterInt(bag, &par->nx, 4, "nx", "Element count in the x-dir")); 
  PetscCall(PetscBagRegisterInt(bag, &par->nz, 5, "nz", "Element count in the z-dir")); 

  PetscCall(PetscBagRegisterScalar(bag, &par->xmin, -1.0, "xmin", "Start coordinate of domain in x-dir")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->zmin, -1.0, "zmin", "Start coordinate of domain in z-dir")); 

  PetscCall(PetscBagRegisterScalar(bag, &par->L, 2.0, "L", "Length of domain in x-dir")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->H, 2.0, "H", "Height of domain in z-dir")); 

  // Physical and material parameters
  PetscCall(PetscBagRegisterScalar(bag, &par->k, 5.0e-3, "k", "Thermal conductivity")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->rho, 1.0, "rho", "Density")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->cp, 1.0, "cp", "Heat capacity")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->ux, 0.0, "ux", "Horizontal velocity")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->uz, 1.0, "uz", "Vertical velocity")); 
  PetscCall(PetscBagRegisterInt(bag, &par->advtype, 0, "advtype", "Advection scheme type 0-upwind, 1-fromm")); 

  // Input/output 
  PetscCall(PetscBagRegisterString(bag,&par->fname_out,FNAME_LENGTH,"out_num_solution_elman311","output_file","Name for output file, set with: -output_file <filename>")); 
  PetscCall(PetscBagRegisterString(bag,&par->fdir_out,FNAME_LENGTH,"./","output_dir","Name for output directory, set with: -output_dir <dirname>")); 

  // Other variables
  par->fname_in[0] = '\0';

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
  PetscPrintf(usr->comm,"# Test_advdiff_elman: %s \n",&(date[0]));
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
  DM              dm;
  Vec             x,xAnalytic;
  PetscLogDouble  start_time, end_time;
    
  // Initialize application
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help)); 

  // Start time
  PetscCall(PetscTime(&start_time)); 
 
  // Load command line or input file if required
  PetscCall(PetscOptionsInsert(PETSC_NULLPTR,&argc,&argv,NULL)); 

  // Input user parameters and print
  PetscCall(InputParameters(&usr)); 

  // Save input options filename
  for (int i = 1; i < argc; i++) {
    PetscBool flg;
    
    PetscCall(PetscStrcmp(argv[i],"-options_file",&flg)); 
    if (flg) { PetscCall(PetscStrcpy(usr->par->fname_in, argv[i+1]));  }
  }

  // Print user parameters
  PetscCall(InputPrintData(usr)); 

  // Numerical solution using the FD pde object
  PetscCall(Numerical_Elman311(&dm, &x, usr)); 

  // Analytical solution
  PetscCall(Analytic_Elman311(dm, &xAnalytic, usr)); 

  // Destroy objects
  PetscCall(DMDestroy(&dm)); 
  PetscCall(VecDestroy(&x)); 
  PetscCall(VecDestroy(&xAnalytic)); 

  // Numerical solution - Elman 312
  PetscCall(Numerical_Elman312(&dm, &x, usr)); 
  PetscCall(DMDestroy(&dm)); 
  PetscCall(VecDestroy(&x)); 

  // Numerical solution - Elman 313
  PetscCall(Numerical_Elman313(&dm, &x, usr)); 
  PetscCall(DMDestroy(&dm)); 
  PetscCall(VecDestroy(&x)); 

  // Numerical solution - Elman 314
  PetscCall(Numerical_Elman314(&dm, &x, usr)); 
  PetscCall(DMDestroy(&dm)); 
  PetscCall(VecDestroy(&x)); 

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
