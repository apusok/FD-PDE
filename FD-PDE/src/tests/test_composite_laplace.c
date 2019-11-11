// ---------------------------------------
// Composite solver test (monolithic)
// run: ./tests/test_composite_laplace.app -pc_type lu -pc_factor_mat_solver_type umfpack -nx 10 -nz 10
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

#include "petsc.h"
#include "../fdpde_advdiff.h"
#include "../fdpde_composite.h"
#include "../dmstagoutput.h"

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
  PetscErrorCode ierr;

  PetscFunctionBegin;

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
  ierr = FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_ADVDIFF,&fd);CHKERRQ(ierr);
  ierr = FDPDESetUp(fd);CHKERRQ(ierr);
  ierr = FDPDEAdvDiffSetAdvectSchemeType(fd,ADV_NONE);CHKERRQ(ierr);
  ierr = FDPDEAdvDiffSetTimeStepSchemeType(fd,TS_NONE);CHKERRQ(ierr);
  // User can modify the dm coordinates anywhere between FDPDESetUp() and FDPDESolve()

  // Set BC evaluation function
  ierr = FDPDESetFunctionBCList(fd,FormBCList_Laplace,bc_description,NULL); CHKERRQ(ierr);

  // Set coefficients evaluation function
  ierr = FDPDESetFunctionCoefficient(fd,FormCoefficient_Laplace,coeff_description,usr); CHKERRQ(ierr);

  ierr = FDPDEView(fd); CHKERRQ(ierr);

  // FD SNES Solver
  // ierr = FDPDEGetSNES(fd,&snes);CHKERRQ(ierr);
  // ierr = SNESSetOptionsPrefix(snes,"adv_");CHKERRQ(ierr);
  ierr = FDPDESolve(fd,NULL);CHKERRQ(ierr);

  // Get solution vector
  ierr = FDPDEGetSolution(fd,&x);CHKERRQ(ierr); 
  ierr = FDPDEGetDM(fd, &dm); CHKERRQ(ierr);
  VecView(x,PETSC_VIEWER_STDOUT_WORLD);

  // Output solution to file
  ierr = DMStagViewBinaryPython(dm,x,usr->par->fname_out);CHKERRQ(ierr);

  // Destroy FD-PDE object
  ierr = FDPDEDestroy(&fd);CHKERRQ(ierr);

  *_x  = x;
  *_dm = dm;

  PetscFunctionReturn(0);
}

PetscErrorCode Numerical_Laplace_Decoupled(DM _dm[], Vec _x[], void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  FDPDE          fdlaplace[2],fdmono,*pdes;
  Vec            x;
  PetscInt       nx, nz, i, n=2;
  PetscScalar    dx, dz,xmin, zmin, xmax, zmax;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  
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
    
    ierr = FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_ADVDIFF,&fd);CHKERRQ(ierr);
    ierr = FDPDESetUp(fd);CHKERRQ(ierr);
    ierr = FDPDEAdvDiffSetAdvectSchemeType(fd,ADV_NONE);CHKERRQ(ierr);
    ierr = FDPDEAdvDiffSetTimeStepSchemeType(fd,TS_NONE);CHKERRQ(ierr);
    // User can modify the dm coordinates anywhere between FDPDESetUp() and FDPDESolve()
    
    // Set BC evaluation function
    ierr = FDPDESetFunctionBCList(fd,FormBCList_Laplace,bc_description,NULL); CHKERRQ(ierr);
    
    // Set coefficients evaluation function
    if (i == 0) {
      ierr = FDPDESetFunctionCoefficient(fd,FormCoefficient_Laplace_NL,coeff_description,usr); CHKERRQ(ierr);
    } else {
      ierr = FDPDESetFunctionCoefficient(fd,FormCoefficient_Laplace,coeff_description,usr); CHKERRQ(ierr);
    }

    if (i == 0) {
      ierr = FDPDESetFunctionCoefficient(fd,FormCoefficient_Laplace_NL1,coeff_description,usr); CHKERRQ(ierr);
    } else {
      ierr = FDPDESetFunctionCoefficient(fd,FormCoefficient_Laplace_NL2,coeff_description,usr); CHKERRQ(ierr);
    }

    fdlaplace[i] = fd;
  }
  
  // Create the composite FD-PDE
  ierr = FDPDECreate2(usr->comm,&fdmono);CHKERRQ(ierr);
  ierr = FDPDESetType(fdmono,FDPDE_COMPOSITE);CHKERRQ(ierr);
  ierr = FDPDCompositeSetFDPDE(fdmono,2,fdlaplace);CHKERRQ(ierr);
  ierr = FDPDESetUp(fdmono);CHKERRQ(ierr);
  ierr = FDPDEView(fdmono); CHKERRQ(ierr);
  
  for (i=0; i<n; i++) {
    ierr = FDPDEDestroy(&fdlaplace[i]);CHKERRQ(ierr);
  }

  
  // FD SNES Solver
  ierr = FDPDESolve(fdmono,NULL);CHKERRQ(ierr);
  // testing
  ierr = FDPDEGetSolution(fdmono,&x);CHKERRQ(ierr);
  //ierr = FDPDESNESComposite_Jacobi(fdmono,x);CHKERRQ(ierr);
  //ierr = FDPDESNESComposite_GaussSeidel(fdmono,x);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  
  // Get solution vector
  ierr = FDPDEGetSolution(fdmono,&x);CHKERRQ(ierr);
  ierr = FDPDECompositeSynchronizeGlobalVectors(fdmono,x);CHKERRQ(ierr);
  
  ierr = FDPDCompositeGetFDPDE(fdmono,NULL,&pdes);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    ierr = FDPDEGetDM(pdes[i],&_dm[i]); CHKERRQ(ierr);
    ierr = FDPDEGetSolution(pdes[i],&_x[i]);CHKERRQ(ierr);
    printf("solution %d\n",i);
    //VecView(_x[i],PETSC_VIEWER_STDOUT_WORLD);
  }
  
  // Output solution to file
  //ierr = DMStagViewBinaryPython(dm,x,usr->par->fname_out);CHKERRQ(ierr);
  
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = FDPDEDestroy(&fdmono);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
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
  PetscErrorCode ierr;
  PetscFunctionBegin;
  
  // Left: T = 0.0
  ierr = DMStagBCListGetValues(bclist,'w','o',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  
  // RIGHT: T = 0.0
  ierr = DMStagBCListGetValues(bclist,'e','o',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN: T = 0.0
  ierr = DMStagBCListGetValues(bclist,'s','o',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP: T = T0*sin(pi*x)/a, T0=1, a=1
  ierr = DMStagBCListGetValues(bclist,'n','o',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = PetscSinScalar(PETSC_PI*x_bc[2*k]);
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
 
  PetscFunctionReturn(0);
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
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  ierr = FDPDEGetAuxGlobalVectors(fd,&naux,&aux_vecs);CHKERRQ(ierr);
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
  ierr = DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  // Create coefficient local vector
  ierr = DMCreateLocalVector(dmcoeff, &coefflocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayDOF(dmcoeff, coefflocal, &c); CHKERRQ(ierr);
  
  // Loop over local domain - set initial density and viscosity
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {

      { // A = rho*cp
        DMStagStencil point;
        PetscInt      idx;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 0;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = rho*cp;
      }

      { // C = 0.0
        DMStagStencil point;
        PetscInt      idx;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 1;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
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
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
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
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = v[0];
        }
        for (ii = 2; ii < 4; ii++) {
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = v[1];
        }
      }
    }
  }

  // Restore arrays, local vectors
  ierr = DMStagVecRestoreArrayDOF(dmcoeff,coefflocal,&c);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  
  ierr = VecDestroy(&coefflocal); CHKERRQ(ierr);

  PetscFunctionReturn(0);
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
  PetscErrorCode ierr;
  
  PetscFunctionBeginUser;
  
  ierr = FDPDEGetAuxGlobalVectors(fd,&naux,&aux_vecs);CHKERRQ(ierr);
  //PetscPrintf(PETSC_COMM_WORLD,"Found %D auxillary vectors\n",naux);

  x_u = aux_vecs[0];
  x_v = aux_vecs[1];
  
  ierr = DMCreateLocalVector(dm,&x_u_local);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,x_u,INSERT_VALUES,x_u_local);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm,x_u,INSERT_VALUES,x_u_local);CHKERRQ(ierr);
  ierr = DMStagVecGetArrayDOF(dm,x_u_local,&LA_x_u); CHKERRQ(ierr);

  ierr = DMCreateLocalVector(dm,&x_v_local);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,x_v,INSERT_VALUES,x_v_local);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm,x_v,INSERT_VALUES,x_v_local);CHKERRQ(ierr);
  ierr = DMStagVecGetArrayDOF(dm,x_v_local,&LA_x_v); CHKERRQ(ierr);
  
  // Element: A = rho*cp (dof 0), C = heat production/sink (dof 1)
  // Edges: k (dof 0), velocity (dof 1)
  
  // User parameters
  rho = usr->par->rho;
  k   = usr->par->k;
  cp  = usr->par->cp;
  v[0]= usr->par->ux;
  v[1]= usr->par->uz;
  
  // Get domain corners
  ierr = DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  
  // Create coefficient local vector
  ierr = DMCreateLocalVector(dmcoeff, &coefflocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayDOF(dmcoeff, coefflocal, &c); CHKERRQ(ierr);
  
  // Loop over local domain - set initial density and viscosity
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      
      { // A = rho*cp
        DMStagStencil point;
        PetscInt      idx;
        
        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 0;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = rho*cp;
      }
      
      { // C = 0.0
        DMStagStencil point;
        PetscInt      idx;
        
        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 1;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
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
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
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
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = v[0];
        }
        for (ii = 2; ii < 4; ii++) {
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = v[1];
        }
      }
    }
  }
  
  // Restore arrays, local vectors
  ierr = DMStagVecRestoreArrayDOF(dmcoeff,coefflocal,&c);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  ierr = VecDestroy(&coefflocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArrayDOF(dm,x_u_local,&LA_x_u);CHKERRQ(ierr);
  ierr = VecDestroy(&x_u_local);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArrayDOF(dm,x_v_local,&LA_x_v);CHKERRQ(ierr);
  ierr = VecDestroy(&x_v_local);CHKERRQ(ierr);

  PetscFunctionReturn(0);
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
  PetscErrorCode ierr;
  
  PetscFunctionBeginUser;
  
  ierr = FDPDEGetAuxGlobalVectors(fd,&naux,&aux_vecs);CHKERRQ(ierr);
  
  x_u = aux_vecs[0];
  x_v = aux_vecs[1];
  
  ierr = DMCreateLocalVector(dm,&x_u_local);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,x_u,INSERT_VALUES,x_u_local);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm,x_u,INSERT_VALUES,x_u_local);CHKERRQ(ierr);
  ierr = DMStagVecGetArrayDOF(dm,x_u_local,&LA_x_u); CHKERRQ(ierr);
  
  ierr = DMCreateLocalVector(dm,&x_v_local);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,x_v,INSERT_VALUES,x_v_local);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm,x_v,INSERT_VALUES,x_v_local);CHKERRQ(ierr);
  ierr = DMStagVecGetArrayDOF(dm,x_v_local,&LA_x_v); CHKERRQ(ierr);
  
  // Element: A = rho*cp (dof 0), C = heat production/sink (dof 1)
  // Edges: k (dof 0), velocity (dof 1)
  
  // User parameters
  rho = usr->par->rho;
  k   = usr->par->k;
  cp  = usr->par->cp;
  v[0]= usr->par->ux;
  v[1]= usr->par->uz;
  
  // Get domain corners
  ierr = DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  
  // Create coefficient local vector
  ierr = DMCreateLocalVector(dmcoeff, &coefflocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayDOF(dmcoeff, coefflocal, &c); CHKERRQ(ierr);
  
  // Loop over local domain - set initial density and viscosity
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      
      { // A = rho*cp
        DMStagStencil point;
        PetscInt      idx;
        
        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 0;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = rho*cp;
      }
      
      { // C = 0.0
        DMStagStencil point;
        PetscInt      idx;
        
        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 1;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
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
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
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
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = v[0];
        }
        for (ii = 2; ii < 4; ii++) {
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = v[1];
        }
      }
    }
  }
  
  // Restore arrays, local vectors
  ierr = DMStagVecRestoreArrayDOF(dmcoeff,coefflocal,&c);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  ierr = VecDestroy(&coefflocal); CHKERRQ(ierr);
  
  ierr = DMStagVecRestoreArrayDOF(dm,x_u_local,&LA_x_u);CHKERRQ(ierr);
  ierr = VecDestroy(&x_u_local);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArrayDOF(dm,x_v_local,&LA_x_v);CHKERRQ(ierr);
  ierr = VecDestroy(&x_v_local);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
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
  PetscErrorCode ierr;
  
  PetscFunctionBeginUser;
  
  ierr = FDPDEGetAuxGlobalVectors(fd,&naux,&aux_vecs);CHKERRQ(ierr);
  
  x_u = aux_vecs[0];
  x_v = aux_vecs[1];
  
  ierr = DMCreateLocalVector(dm,&x_u_local);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,x_u,INSERT_VALUES,x_u_local);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm,x_u,INSERT_VALUES,x_u_local);CHKERRQ(ierr);
  ierr = DMStagVecGetArrayDOF(dm,x_u_local,&LA_x_u); CHKERRQ(ierr);
  
  ierr = DMCreateLocalVector(dm,&x_v_local);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,x_v,INSERT_VALUES,x_v_local);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm,x_v,INSERT_VALUES,x_v_local);CHKERRQ(ierr);
  ierr = DMStagVecGetArrayDOF(dm,x_v_local,&LA_x_v); CHKERRQ(ierr);
  
  // Element: A = rho*cp (dof 0), C = heat production/sink (dof 1)
  // Edges: k (dof 0), velocity (dof 1)
  
  // User parameters
  rho = usr->par->rho;
  k   = usr->par->k;
  cp  = usr->par->cp;
  v[0]= usr->par->ux;
  v[1]= usr->par->uz;
  
  // Get domain corners
  ierr = DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  
  // Create coefficient local vector
  ierr = DMCreateLocalVector(dmcoeff, &coefflocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayDOF(dmcoeff, coefflocal, &c); CHKERRQ(ierr);
  
  // Loop over local domain - set initial density and viscosity
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      
      { // A = rho*cp
        DMStagStencil point;
        PetscInt      idx;
        
        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 0;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = rho*cp;
      }
      
      { // C = 0.0
        DMStagStencil point;
        PetscInt      idx;
        
        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 1;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
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
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
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
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = v[0];
        }
        for (ii = 2; ii < 4; ii++) {
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = v[1];
        }
      }
    }
  }
  
  // Restore arrays, local vectors
  ierr = DMStagVecRestoreArrayDOF(dmcoeff,coefflocal,&c);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  ierr = VecDestroy(&coefflocal); CHKERRQ(ierr);
  
  ierr = DMStagVecRestoreArrayDOF(dm,x_u_local,&LA_x_u);CHKERRQ(ierr);
  ierr = VecDestroy(&x_u_local);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArrayDOF(dm,x_v_local,&LA_x_v);CHKERRQ(ierr);
  ierr = VecDestroy(&x_v_local);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;

  A = 1.0/PetscSinhReal(PETSC_PI);

  // Create local and global vector associated with DM
  ierr = DMCreateGlobalVector(dm, &x     ); CHKERRQ(ierr);
  ierr = DMCreateLocalVector (dm, &xlocal); CHKERRQ(ierr);

  // Get array associated with vector
  ierr = DMStagVecGetArrayDOF(dm,xlocal,&xx); CHKERRQ(ierr);

  // Get domain corners
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  
// Get dm coordinates array
  ierr = DMStagGet1dCoordinateArraysDOFRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGet1dCoordinateLocationSlot(dm,ELEMENT,&icenter);CHKERRQ(ierr); 

  // Loop over local domain to calculate the SolCx analytical solution
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      DMStagStencil  point;
      PetscScalar    xp, zp;

      point.i = i; point.j = j; point.loc = ELEMENT; point.c = 0; 
      xp = coordx[i][icenter];
      zp = coordz[j][icenter];

      ierr = DMStagGetLocationSlot(dm, point.loc, point.c, &idx);CHKERRQ(ierr);
      xx[j][i][idx] = A*PetscSinScalar(PETSC_PI*xp)*PetscSinhReal(PETSC_PI*zp);
    }
  }

  // Restore arrays
  ierr = DMStagRestore1dCoordinateArraysDOFRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArrayDOF(dm,xlocal,&xx); CHKERRQ(ierr);

  // Map local to global
  ierr = DMLocalToGlobalBegin(dm,xlocal,INSERT_VALUES,x); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,xlocal,INSERT_VALUES,x); CHKERRQ(ierr);

  ierr = VecDestroy(&xlocal); CHKERRQ(ierr);

  ierr = DMStagViewBinaryPython(dm,x,"out_analytic_solution_laplace");CHKERRQ(ierr);

  // Assign pointers
  *_x  = x;
  
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

  ierr = PetscBagRegisterScalar(bag, &par->xmin, 0.0, "xmin", "Start coordinate of domain in x-dir"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->zmin, 0.0, "zmin", "Start coordinate of domain in z-dir"); CHKERRQ(ierr);

  ierr = PetscBagRegisterScalar(bag, &par->L, 1.0, "L", "Length of domain in x-dir"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->H, 1.0, "H", "Height of domain in z-dir"); CHKERRQ(ierr);

  // Physical and material parameters
  ierr = PetscBagRegisterScalar(bag, &par->k, 1.0, "k", "Thermal conductivity"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->rho, 0.0, "rho", "Density"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->cp, 0.0, "cp", "Heat capacity"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->ux, 0.0, "ux", "Horizontal velocity"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->uz, 0.0, "uz", "Vertical velocity"); CHKERRQ(ierr);

  // Input/output 
  ierr = PetscBagRegisterString(bag,&par->fname_out,FNAME_LENGTH,"out_num_solution_laplace","output_file","Name for output file, set with: -output_file <filename>"); CHKERRQ(ierr);

  // Other variables
  par->fname_in[0] = '\0';

  // return pointer
  *_usr = usr;

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
  DM              dmLaplace[2];
  Vec             xLaplace[2],xAnalytic;
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

  // Save input options filename
  for (int i = 1; i < argc; i++) {
    PetscBool flg;
    
    ierr = PetscStrcmp(argv[i],"-options_file",&flg); CHKERRQ(ierr);
    if (flg) { ierr = PetscStrcpy(usr->par->fname_in, argv[i+1]); CHKERRQ(ierr); }
  }

  // Numerical solution using the FD pde object
  /*
  {
    DM dm0;
    Vec x0;
    ierr = Numerical_Laplace(&dm0, &x0, usr); CHKERRQ(ierr);
    
    ierr = DMDestroy(&dm0); CHKERRQ(ierr);
    ierr = VecDestroy(&x0); CHKERRQ(ierr);
  }
  */
   
  ierr = Numerical_Laplace_Decoupled(dmLaplace, xLaplace, usr); CHKERRQ(ierr);
  
  // Analytical solution
  ierr = Analytic_Laplace(dmLaplace[0], &xAnalytic, usr); CHKERRQ(ierr);

  // Destroy objects
  ierr = DMDestroy(&dmLaplace[0]); CHKERRQ(ierr);
  ierr = DMDestroy(&dmLaplace[1]); CHKERRQ(ierr);
  ierr = VecDestroy(&xLaplace[0]); CHKERRQ(ierr);
  ierr = VecDestroy(&xLaplace[1]); CHKERRQ(ierr);
  ierr = VecDestroy(&xAnalytic); CHKERRQ(ierr);

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
