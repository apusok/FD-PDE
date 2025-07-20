// ---------------------------------------
// Composite solver test (monolithic)
// run: ./test_composite_laplace -pc_type lu -pc_factor_mat_solver_type umfpack -pc_factor_mat_ordering_type external -nx 10 -nz 10 -log_view
// ---------------------------------------
static char help[] = "Application to solve two de-coupled Laplace problems with a monolithoc FD-PDE \n\n";

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

#include "../new_src/fdpde_advdiff.h"
#include "../new_src/fdpde_composite.h"

// ---------------------------------------
// Application Context
// ---------------------------------------
#define FNAME_LENGTH 200

// parameters (bag)
typedef struct {
  PetscInt       nx, nz;
  PetscScalar    L, H;
  PetscScalar    xmin, zmin;
  PetscScalar    k, rho, cp, ux, uz;
  char           fname_out[FNAME_LENGTH]; 
  char           fname_in [FNAME_LENGTH];  
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
PetscErrorCode Analytic_Laplace(DM,Vec*,void*);
PetscErrorCode Numerical_Laplace(DM*,Vec*,void*);
PetscErrorCode FormCoefficient_Laplace(FDPDE,DM,Vec,DM,Vec,void*);
PetscErrorCode FormCoefficient_Laplace_NL(FDPDE,DM,Vec,DM,Vec,void*);
PetscErrorCode FormCoefficient_Laplace_NL1(FDPDE,DM,Vec,DM,Vec,void*);
PetscErrorCode FormCoefficient_Laplace_NL2(FDPDE,DM,Vec,DM,Vec,void*);
PetscErrorCode FormBCList_Laplace(DM,Vec,DMStagBCList,void*);

// ---------------------------------------
// Some descriptions
// ---------------------------------------
const char coeff_description[] =
"  << ADVDIFF - Laplace Coefficients >> \n"
"  A = rho*cp (element)\n"
"  B = k (edge)\n"
"  C = 0 (element)\n"
"  u = [ux, uz] (edge)\n";

const char bc_description[] =
"  << Laplace BCs >> \n"
"  LEFT: T = 0\n"
"  RIGHT: T = 0\n" 
"  DOWN: T = 0\n" 
"  UP: T = T0*sin(pi*x)/a, T0=1, a=1 \n";

// ---------------------------------------
// Application functions
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "Numerical_Laplace"
PetscErrorCode Numerical_Laplace(DM *_dm, Vec *_x, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  FDPDE          fd;
  DM             dm;
  Vec            x;
  PetscInt       nx, nz;
  PetscScalar    dx, dz,xmin, zmin, xmax, zmax;
  PetscFunctionBeginUser;

  // Element count
  nx = usr->par->nx;
  nz = usr->par->nz;

  // Domain coords
  dx = usr->par->L/(2*nx-2);
  dz = usr->par->H/(2*nz-2);

  // modify coord of dm such that unknowns are located on the boundaries limits (0,1)
  xmin = usr->par->xmin-dx;
  zmin = usr->par->zmin-dz;
  xmax = usr->par->xmin+usr->par->L+dx;
  zmax = usr->par->zmin+usr->par->H+dz;

  // Create the FD-pde object
  PetscCall(FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_ADVDIFF,&fd));
  PetscCall(FDPDESetUp(fd));
  PetscCall(FDPDEAdvDiffSetAdvectSchemeType(fd,ADV_NONE));
  PetscCall(FDPDEAdvDiffSetTimeStepSchemeType(fd,TS_NONE));
  // User can modify the dm coordinates anywhere between FDPDESetUp() and FDPDESolve()

  // Set BC evaluation function
  PetscCall(FDPDESetFunctionBCList(fd,FormBCList_Laplace,bc_description,NULL)); 

  // Set coefficients evaluation function
  PetscCall(FDPDESetFunctionCoefficient(fd,FormCoefficient_Laplace,coeff_description,usr)); 

  PetscCall(FDPDEView(fd)); 

  // FD SNES Solver
  // PetscCall(FDPDEGetSNES(fd,&snes));
  // PetscCall(SNESSetOptionsPrefix(snes,"adv_"));
  PetscCall(FDPDESolve(fd,NULL));

  // Get solution vector
  PetscCall(FDPDEGetSolution(fd,&x)); 
  PetscCall(FDPDEGetDM(fd, &dm)); 
  VecView(x,PETSC_VIEWER_STDOUT_WORLD);

  // Output solution to file
  PetscCall(DMStagViewBinaryPython(dm,x,usr->par->fname_out));

  // Destroy FD-PDE object
  PetscCall(FDPDEDestroy(&fd));

  *_x  = x;
  *_dm = dm;

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Numerical_Laplace_Decoupled(DM _dm[], Vec _x[], void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  FDPDE          fdlaplace[2],fdmono,*pdes;
  Vec            x;
  PetscInt       nx, nz, i, n=2;
  PetscScalar    dx, dz,xmin, zmin, xmax, zmax;
  PetscFunctionBeginUser;
  
  // Element count
  nx = usr->par->nx;
  nz = usr->par->nz;
  
  // Domain coords
  dx = usr->par->L/(2*nx-2);
  dz = usr->par->H/(2*nz-2);
  
  // modify coord of dm such that unknowns are located on the boundaries limits (0,1)
  xmin = usr->par->xmin-dx;
  zmin = usr->par->zmin-dz;
  xmax = usr->par->xmin+usr->par->L+dx;
  zmax = usr->par->zmin+usr->par->H+dz;
  
  // Create the sub FD-pde objects
  for (i=0; i<n; i++) {
    FDPDE fd;
    
    PetscCall(FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_ADVDIFF,&fd));
    PetscCall(FDPDESetUp(fd));
    PetscCall(FDPDEAdvDiffSetAdvectSchemeType(fd,ADV_NONE));
    PetscCall(FDPDEAdvDiffSetTimeStepSchemeType(fd,TS_NONE));
    // User can modify the dm coordinates anywhere between FDPDESetUp() and FDPDESolve()
    
    // Set BC evaluation function
    PetscCall(FDPDESetFunctionBCList(fd,FormBCList_Laplace,bc_description,NULL)); 
    
    // Set coefficients evaluation function
    if (i == 0) {
      PetscCall(FDPDESetFunctionCoefficient(fd,FormCoefficient_Laplace_NL,coeff_description,usr)); 
    } else {
      PetscCall(FDPDESetFunctionCoefficient(fd,FormCoefficient_Laplace,coeff_description,usr)); 
    }

    if (i == 0) {
      PetscCall(FDPDESetFunctionCoefficient(fd,FormCoefficient_Laplace_NL1,coeff_description,usr)); 
    } else {
      PetscCall(FDPDESetFunctionCoefficient(fd,FormCoefficient_Laplace_NL2,coeff_description,usr)); 
    }

    fdlaplace[i] = fd;
  }
  
  // Create the composite FD-PDE
  PetscCall(FDPDECreate2(usr->comm,&fdmono));
  PetscCall(FDPDESetType(fdmono,FDPDE_COMPOSITE));
  PetscCall(FDPDECompositeSetFDPDE(fdmono,2,fdlaplace));
  PetscCall(FDPDESetUp(fdmono));
  PetscCall(FDPDEView(fdmono)); 
  
  for (i=0; i<n; i++) {
    PetscCall(FDPDEDestroy(&fdlaplace[i]));
  }

  // FD SNES Solver
  PetscCall(FDPDESolve(fdmono,NULL));
  // testing
  PetscCall(FDPDEGetSolution(fdmono,&x));
  //PetscCall(FDPDESNESComposite_Jacobi(fdmono,x));
  //PetscCall(FDPDESNESComposite_GaussSeidel(fdmono,x));
  PetscCall(VecDestroy(&x));
  
  // Get solution vector
  PetscCall(FDPDEGetSolution(fdmono,&x));
  PetscCall(FDPDECompositeSynchronizeGlobalVectors(fdmono,x));
  
  PetscCall(FDPDECompositeGetFDPDE(fdmono,NULL,&pdes));
  for (i=0; i<n; i++) {
    PetscCall(FDPDEGetDM(pdes[i],&_dm[i])); 
    PetscCall(FDPDEGetSolution(pdes[i],&_x[i]));
    printf("solution %d\n",i);
    //VecView(_x[i],PETSC_VIEWER_STDOUT_WORLD);
  }
  
  // Output solution to file
  //PetscCall(DMStagViewBinaryPython(dm,x,usr->par->fname_out));
  
  PetscCall(VecDestroy(&x));
  PetscCall(FDPDEDestroy(&fdmono));
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// FormBCList_Laplace
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormBCList_Laplace"
PetscErrorCode FormBCList_Laplace(DM dm, Vec x, DMStagBCList bclist, void *ctx)
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
  
  // RIGHT: T = 0.0
  PetscCall(DMStagBCListGetValues(bclist,'e','o',0,&n_bc,&idx_bc,NULL,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
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

  // UP: T = T0*sin(pi*x)/a, T0=1, a=1
  PetscCall(DMStagBCListGetValues(bclist,'n','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = PetscSinScalar(PETSC_PI*x_bc[2*k]);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
 
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// FormCoefficient_Laplace
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormCoefficient_Laplace"
PetscErrorCode FormCoefficient_Laplace(FDPDE fd, DM dm, Vec x, DM dmcoeff, Vec coeff, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       i, j, sx, sz, nx, nz;
  Vec            coefflocal;
  PetscScalar    rho, k, cp, v[2];
  PetscScalar    ***c;
  PetscInt       naux;
  Vec            *aux_vecs;
  PetscFunctionBeginUser;

  PetscCall(FDPDEGetAuxGlobalVectors(fd,&naux,&aux_vecs));
  //PetscPrintf(PETSC_COMM_WORLD,"Found %D auxillary vectors\n",naux);
  
  // Element: A = rho*cp (dof 0), C = heat production/sink (dof 1)
  // Edges: k (dof 0), velocity (dof 1)

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

#undef __FUNCT__
#define __FUNCT__ "FormCoefficient_Laplace_NL"
PetscErrorCode FormCoefficient_Laplace_NL(FDPDE fd, DM dm, Vec x, DM dmcoeff, Vec coeff, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       i, j, sx, sz, nx, nz;
  Vec            coefflocal;
  PetscScalar    rho, k, cp, v[2];
  PetscScalar    ***c,***LA_x_u,***LA_x_v;
  PetscInt       naux;
  Vec            *aux_vecs, x_u, x_u_local, x_v, x_v_local;
  PetscFunctionBeginUser;
  
  PetscCall(FDPDEGetAuxGlobalVectors(fd,&naux,&aux_vecs));
  //PetscPrintf(PETSC_COMM_WORLD,"Found %D auxillary vectors\n",naux);

  x_u = aux_vecs[0];
  x_v = aux_vecs[1];
  
  PetscCall(DMCreateLocalVector(dm,&x_u_local));
  PetscCall(DMGlobalToLocalBegin(dm,x_u,INSERT_VALUES,x_u_local));
  PetscCall(DMGlobalToLocalEnd(dm,x_u,INSERT_VALUES,x_u_local));
  PetscCall(DMStagVecGetArray(dm,x_u_local,&LA_x_u)); 

  PetscCall(DMCreateLocalVector(dm,&x_v_local));
  PetscCall(DMGlobalToLocalBegin(dm,x_v,INSERT_VALUES,x_v_local));
  PetscCall(DMGlobalToLocalEnd(dm,x_v,INSERT_VALUES,x_v_local));
  PetscCall(DMStagVecGetArray(dm,x_v_local,&LA_x_v)); 
  
  // Element: A = rho*cp (dof 0), C = heat production/sink (dof 1)
  // Edges: k (dof 0), velocity (dof 1)
  
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
        c[j][i][idx] = 1.0 + 10.1 * pow(LA_x_v[j][i][0] , 33.3);
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
          c[j][i][idx] = 1.0 + pow(LA_x_v[j][i][0] , 2.3) + pow(LA_x_u[j][i][0] , 1.0);
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

  PetscCall(DMStagVecRestoreArray(dm,x_u_local,&LA_x_u));
  PetscCall(VecDestroy(&x_u_local));
  PetscCall(DMStagVecRestoreArray(dm,x_v_local,&LA_x_v));
  PetscCall(VecDestroy(&x_v_local));

  PetscFunctionReturn(PETSC_SUCCESS);
}


#undef __FUNCT__
#define __FUNCT__ "FormCoefficient_Laplace_NL1"
PetscErrorCode FormCoefficient_Laplace_NL1(FDPDE fd, DM dm, Vec x, DM dmcoeff, Vec coeff, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       i, j, sx, sz, nx, nz;
  Vec            coefflocal;
  PetscScalar    rho, k, cp, v[2];
  PetscScalar    ***c,***LA_x_u,***LA_x_v;
  PetscInt       naux;
  Vec            *aux_vecs, x_u, x_u_local, x_v, x_v_local;
  PetscFunctionBeginUser;
  
  PetscCall(FDPDEGetAuxGlobalVectors(fd,&naux,&aux_vecs));
  
  x_u = aux_vecs[0];
  x_v = aux_vecs[1];
  
  PetscCall(DMCreateLocalVector(dm,&x_u_local));
  PetscCall(DMGlobalToLocalBegin(dm,x_u,INSERT_VALUES,x_u_local));
  PetscCall(DMGlobalToLocalEnd(dm,x_u,INSERT_VALUES,x_u_local));
  PetscCall(DMStagVecGetArray(dm,x_u_local,&LA_x_u)); 
  
  PetscCall(DMCreateLocalVector(dm,&x_v_local));
  PetscCall(DMGlobalToLocalBegin(dm,x_v,INSERT_VALUES,x_v_local));
  PetscCall(DMGlobalToLocalEnd(dm,x_v,INSERT_VALUES,x_v_local));
  PetscCall(DMStagVecGetArray(dm,x_v_local,&LA_x_v)); 
  
  // Element: A = rho*cp (dof 0), C = heat production/sink (dof 1)
  // Edges: k (dof 0), velocity (dof 1)
  
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
        c[j][i][idx] = 1.0 + 10.1 * LA_x_v[j][i][0];
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
          c[j][i][idx] = k * (1.0 + pow(LA_x_u[j][i][0] , 2) * exp(LA_x_v[j][i][0]) );
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
  
  PetscCall(DMStagVecRestoreArray(dm,x_u_local,&LA_x_u));
  PetscCall(VecDestroy(&x_u_local));
  PetscCall(DMStagVecRestoreArray(dm,x_v_local,&LA_x_v));
  PetscCall(VecDestroy(&x_v_local));
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "FormCoefficient_Laplace_NL2"
PetscErrorCode FormCoefficient_Laplace_NL2(FDPDE fd, DM dm, Vec x, DM dmcoeff, Vec coeff, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       i, j, sx, sz, nx, nz;
  Vec            coefflocal;
  PetscScalar    rho, k, cp, v[2];
  PetscScalar    ***c,***LA_x_u,***LA_x_v;
  PetscInt       naux;
  Vec            *aux_vecs, x_u, x_u_local, x_v, x_v_local;
  PetscFunctionBeginUser;
  
  PetscCall(FDPDEGetAuxGlobalVectors(fd,&naux,&aux_vecs));
  
  x_u = aux_vecs[0];
  x_v = aux_vecs[1];
  
  PetscCall(DMCreateLocalVector(dm,&x_u_local));
  PetscCall(DMGlobalToLocalBegin(dm,x_u,INSERT_VALUES,x_u_local));
  PetscCall(DMGlobalToLocalEnd(dm,x_u,INSERT_VALUES,x_u_local));
  PetscCall(DMStagVecGetArray(dm,x_u_local,&LA_x_u)); 
  
  PetscCall(DMCreateLocalVector(dm,&x_v_local));
  PetscCall(DMGlobalToLocalBegin(dm,x_v,INSERT_VALUES,x_v_local));
  PetscCall(DMGlobalToLocalEnd(dm,x_v,INSERT_VALUES,x_v_local));
  PetscCall(DMStagVecGetArray(dm,x_v_local,&LA_x_v)); 
  
  // Element: A = rho*cp (dof 0), C = heat production/sink (dof 1)
  // Edges: k (dof 0), velocity (dof 1)
  
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
        c[j][i][idx] = 1.0;
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
          c[j][i][idx] = k * (1.0 + pow(LA_x_v[j][i][0] , 1.0) );
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
  
  PetscCall(DMStagVecRestoreArray(dm,x_u_local,&LA_x_u));
  PetscCall(VecDestroy(&x_u_local));
  PetscCall(DMStagVecRestoreArray(dm,x_v_local,&LA_x_v));
  PetscCall(VecDestroy(&x_v_local));
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// Create analytical solution for Laplace equation
// ---------------------------------------
PetscErrorCode Analytic_Laplace(DM dm,Vec *_x, void *ctx)
{
  //UsrData       *usr = (UsrData*) ctx;
  PetscInt       i, j, sx, sz, nx, nz, idx, icenter;
  PetscScalar    ***xx, A;
  PetscScalar    **coordx,**coordz;
  Vec            x, xlocal;
  PetscFunctionBeginUser;

  A = 1.0/PetscSinhReal(PETSC_PI);

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
      xx[j][i][idx] = A*PetscSinScalar(PETSC_PI*xp)*PetscSinhReal(PETSC_PI*zp);
    }
  }

  // Restore arrays
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));
  PetscCall(DMStagVecRestoreArray(dm,xlocal,&xx)); 

  // Map local to global
  PetscCall(DMLocalToGlobalBegin(dm,xlocal,INSERT_VALUES,x)); 
  PetscCall(DMLocalToGlobalEnd  (dm,xlocal,INSERT_VALUES,x)); 

  PetscCall(VecDestroy(&xlocal)); 

  // PetscCall(DMStagViewBinaryPython(dm,x,"out_analytic_solution_laplace"));

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

  PetscCall(PetscBagRegisterScalar(bag, &par->xmin, 0.0, "xmin", "Start coordinate of domain in x-dir")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->zmin, 0.0, "zmin", "Start coordinate of domain in z-dir")); 

  PetscCall(PetscBagRegisterScalar(bag, &par->L, 1.0, "L", "Length of domain in x-dir")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->H, 1.0, "H", "Height of domain in z-dir")); 

  // Physical and material parameters
  PetscCall(PetscBagRegisterScalar(bag, &par->k, 1.0, "k", "Thermal conductivity")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->rho, 0.0, "rho", "Density")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->cp, 0.0, "cp", "Heat capacity")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->ux, 0.0, "ux", "Horizontal velocity")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->uz, 0.0, "uz", "Vertical velocity")); 

  // Input/output 
  PetscCall(PetscBagRegisterString(bag,&par->fname_out,FNAME_LENGTH,"out_num_solution_laplace","output_file","Name for output file, set with: -output_file <filename>")); 

  // Other variables
  par->fname_in[0] = '\0';

  // return pointer
  *_usr = usr;

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
  DM              dmLaplace[2];
  Vec             xLaplace[2],xAnalytic;
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

  // Numerical solution using the FD pde object
  /*
  {
    DM dm0;
    Vec x0;
    PetscCall(Numerical_Laplace(&dm0, &x0, usr)); 
    
    PetscCall(DMDestroy(&dm0)); 
    PetscCall(VecDestroy(&x0)); 
  }
  */
   
  PetscCall(Numerical_Laplace_Decoupled(dmLaplace, xLaplace, usr)); 
  
  // Analytical solution
  PetscCall(Analytic_Laplace(dmLaplace[0], &xAnalytic, usr)); 

  // Destroy objects
  PetscCall(DMDestroy(&dmLaplace[0])); 
  PetscCall(DMDestroy(&dmLaplace[1])); 
  PetscCall(VecDestroy(&xLaplace[0])); 
  PetscCall(VecDestroy(&xLaplace[1])); 
  PetscCall(VecDestroy(&xAnalytic)); 

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
