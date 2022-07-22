// ---------------------------------------
// MMS 2D diffusion test div(k*grad(T)) = f, with k > 0
// run: ./tests/test_advdiff_mms_2d_diffusion.app -pc_type lu -pc_factor_mat_solver_type umfpack -pc_factor_mat_ordering_type external -nx 10 -nz 10
// ---------------------------------------
static char help[] = "Application to solve an MMS 2D diffusion equation (ADVDIFF) with FD-PDE \n\n";

#include "petsc.h"
#include "../src/fdpde_advdiff.h"
#include "../src/dmstagoutput.h"

// ---------------------------------------
// Function definitions
// ---------------------------------------
PetscErrorCode ComputeManufacturedSolution(DM,Vec*);
PetscErrorCode ComputeErrorNorms(DM,Vec,Vec);
PetscErrorCode FormCoefficient(FDPDE,DM,Vec,DM,Vec,void*);
PetscErrorCode FormBCList(DM,Vec,DMStagBCList,void*);

static PetscScalar get_k(PetscScalar x, PetscScalar z)
{ PetscScalar result;
  result = sin(2.0*M_PI*x)*cos(2.0*M_PI*z) + 1.5;
  return(result);
}
static PetscScalar get_T(PetscScalar x, PetscScalar z)
{ PetscScalar result;
  result = sin(2.0*M_PI*z)*cos(2.0*M_PI*x);
  return(result);
}
static PetscScalar get_f(PetscScalar x, PetscScalar z)
{ PetscScalar result;
  result = -8.0*pow(M_PI, 2)*(sin(2.0*M_PI*x)*cos(2.0*M_PI*z) + 1.5)*sin(2.0*M_PI*z)*cos(2.0*M_PI*x) - 8.0*pow(M_PI, 2)*sin(2.0*M_PI*x)*sin(2.0*M_PI*z)*cos(2.0*M_PI*x)*cos(2.0*M_PI*z);
  return(result);
}

// static PetscScalar get_k(PetscScalar x, PetscScalar z)
// { PetscScalar result;
//   result = 1.0;
//   return(result);
// }
// static PetscScalar get_T(PetscScalar x, PetscScalar z)
// { PetscScalar result;
//   result = sin(2.0*M_PI*z)*cos(2.0*M_PI*x);
//   return(result);
// }
// static PetscScalar get_f(PetscScalar x, PetscScalar z)
// { PetscScalar result;
//   result = -8.0*pow(M_PI, 2)*sin(2.0*M_PI*z)*cos(2.0*M_PI*x);
//   return(result);
// }

// ---------------------------------------
// MAIN
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "main"
int main (int argc,char **argv)
{
  FDPDE           fd;
  PetscInt        nx, nz;
  PetscScalar     L, H;
  DM              dm;
  Vec             x, xMMS;
  MPI_Comm        comm;
  PetscLogDouble  start_time, end_time;
  char            fname[PETSC_MAX_PATH_LEN] = "out_solution";
  char            fdir[PETSC_MAX_PATH_LEN] = "./";
  char            fout[PETSC_MAX_PATH_LEN];
  PetscErrorCode  ierr;
    
  // Initialize application
  ierr = PetscInitialize(&argc,&argv,(char*)0,help); if (ierr) return ierr;
  ierr = PetscTime(&start_time); CHKERRQ(ierr);

  comm = PETSC_COMM_WORLD;
  nx   = 10;
  nz   = 10;
  L    = 1.0;
  H    = 1.0;

  ierr = PetscOptionsGetInt(NULL, NULL, "-nx", &nx, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, NULL, "-nz", &nz, NULL); CHKERRQ(ierr);

  ierr = PetscOptionsGetScalar(NULL, NULL, "-L", &L, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(NULL, NULL, "-H", &H, NULL); CHKERRQ(ierr);

  ierr = PetscOptionsGetString(NULL,NULL,"-fname",fname,sizeof(fname),NULL); CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-fdir",fdir,sizeof(fdir),NULL); CHKERRQ(ierr);

  ierr = PetscOptionsInsert(PETSC_NULL,&argc,&argv,NULL); CHKERRQ(ierr);

  // Create the FD-PDE object
  ierr = FDPDECreate(comm,nx,nz,0.0,L,0.0,H,FDPDE_ADVDIFF,&fd);CHKERRQ(ierr);
  ierr = FDPDESetUp(fd);CHKERRQ(ierr);

  ierr = FDPDEAdvDiffSetAdvectSchemeType(fd,ADV_NONE);CHKERRQ(ierr);
  ierr = FDPDEAdvDiffSetTimeStepSchemeType(fd,TS_NONE);CHKERRQ(ierr);

  // Set coefficient and BC evaluation function
  ierr = FDPDESetFunctionBCList(fd,FormBCList,NULL,NULL); CHKERRQ(ierr);
  ierr = FDPDESetFunctionCoefficient(fd,FormCoefficient,NULL,NULL); CHKERRQ(ierr);

  // Compute manufactured solution
  ierr = FDPDEGetDM(fd, &dm); CHKERRQ(ierr);
  ierr = ComputeManufacturedSolution(dm,&xMMS);CHKERRQ(ierr);

  // Solve
  ierr = FDPDESolve(fd,NULL);CHKERRQ(ierr);
  ierr = FDPDEGetSolution(fd,&x);CHKERRQ(ierr); 

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/%s",fdir,fname);
  ierr = DMStagViewBinaryPython(dm,x,fout);CHKERRQ(ierr);

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/%s",fdir,"out_mms_solution");
  ierr = DMStagViewBinaryPython(dm,xMMS,fout);CHKERRQ(ierr);

  // Compute error
  ierr = ComputeErrorNorms(dm,x,xMMS);CHKERRQ(ierr);

  // Destroy objects
  ierr = DMDestroy(&dm); CHKERRQ(ierr);
  ierr = VecDestroy(&x); CHKERRQ(ierr);
  ierr = VecDestroy(&xMMS); CHKERRQ(ierr);
  ierr = FDPDEDestroy(&fd);CHKERRQ(ierr);

  ierr = PetscTime(&end_time); CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"# Total runtime: %g (sec) \n", end_time - start_time);
  PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");
  
  // Finalize main
  ierr = PetscFinalize();
  return ierr;
}

// ---------------------------------------
// FormBCList
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormBCList"
PetscErrorCode FormBCList(DM dm, Vec x, DMStagBCList bclist, void *ctx)
{
  PetscInt    k,n_bc,*idx_bc;
  PetscScalar *value_bc,*x_bc;
  BCType      *type_bc;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  
  // Left
  ierr = DMStagBCListGetValues(bclist,'w','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_T(x_bc[2*k],x_bc[2*k+1]);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  
  // RIGHT
  ierr = DMStagBCListGetValues(bclist,'e','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_T(x_bc[2*k],x_bc[2*k+1]);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN
  ierr = DMStagBCListGetValues(bclist,'s','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_T(x_bc[2*k],x_bc[2*k+1]);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP
  ierr = DMStagBCListGetValues(bclist,'n','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_T(x_bc[2*k],x_bc[2*k+1]);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
 
  PetscFunctionReturn(0);
}

// ---------------------------------------
// FormCoefficient
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormCoefficient"
PetscErrorCode FormCoefficient(FDPDE fd, DM dm, Vec x, DM dmcoeff, Vec coeff, void *ctx)
{
  PetscInt       i, j, sx, sz, nx, nz;
  Vec            coefflocal;
  PetscScalar    ***c,**coordx,**coordz;
  PetscInt       iprev, inext, icenter;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  // Element: A = 0 (dof 0), C = -f (dof 1)
  // Edges: B = -k (dof 0), u = 0 (dof 1)

  // Get domain corners
  ierr = DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  // Create coefficient local vector
  ierr = DMCreateLocalVector(dmcoeff, &coefflocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dmcoeff, coefflocal, &c); CHKERRQ(ierr);
  
  ierr = DMStagGetProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmcoeff,DMSTAG_LEFT,&iprev);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmcoeff,DMSTAG_RIGHT,&inext);CHKERRQ(ierr); 
  ierr = DMStagGetProductCoordinateLocationSlot(dmcoeff,DMSTAG_ELEMENT,&icenter);CHKERRQ(ierr);
  
  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      PetscInt      ii, idx;

      { // A = 0.0
        DMStagStencil point;
        point.i = i; point.j = j; point.loc = DMSTAG_ELEMENT;  point.c = 0;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = 0.0;
      }

      { // C = -f (mms)
        DMStagStencil point;
        point.i = i; point.j = j; point.loc = DMSTAG_ELEMENT;  point.c = 1;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = -get_f(coordx[i][icenter],coordz[j][icenter]);
      }

      { // B = -k (mms)
        DMStagStencil point[4];
        PetscScalar x[4],z[4];

        point[0].i = i; point[0].j = j; point[0].loc = DMSTAG_LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = DMSTAG_RIGHT; point[1].c = 0;
        point[2].i = i; point[2].j = j; point[2].loc = DMSTAG_DOWN;  point[2].c = 0;
        point[3].i = i; point[3].j = j; point[3].loc = DMSTAG_UP;    point[3].c = 0;

        x[0] = coordx[i][iprev  ]; z[0] = coordz[j][icenter];
        x[1] = coordx[i][inext  ]; z[1] = coordz[j][icenter];
        x[2] = coordx[i][icenter]; z[2] = coordz[j][iprev  ];
        x[3] = coordx[i][icenter]; z[3] = coordz[j][inext  ];

        for (ii = 0; ii < 4; ii++) {
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = -get_k(x[ii],z[ii]);
        }
      }

      { // u = 0 (edge)
        DMStagStencil point[4];

        point[0].i = i; point[0].j = j; point[0].loc = DMSTAG_LEFT;  point[0].c = 1;
        point[1].i = i; point[1].j = j; point[1].loc = DMSTAG_RIGHT; point[1].c = 1;
        point[2].i = i; point[2].j = j; point[2].loc = DMSTAG_DOWN;  point[2].c = 1;
        point[3].i = i; point[3].j = j; point[3].loc = DMSTAG_UP;    point[3].c = 1;

        for (ii = 0; ii < 4; ii++) {
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = 0.0;
        }
      }
    }
  }

  // Restore arrays, local vectors
  ierr = DMStagRestoreProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArray(dmcoeff,coefflocal,&c);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  
  ierr = VecDestroy(&coefflocal); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// Compute manufactured solution
// ---------------------------------------
PetscErrorCode ComputeManufacturedSolution(DM dm,Vec *_xMMS)
{
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz, idx, icenter;
  PetscScalar    ***xxMMS;
  PetscScalar    **coordx,**coordz;
  Vec            xMMS, xMMSlocal;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Create local and global vectors for mms 
  ierr = DMCreateGlobalVector(dm,&xMMS); CHKERRQ(ierr);
  ierr = DMCreateLocalVector (dm,&xMMSlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm,xMMSlocal,&xxMMS); CHKERRQ(ierr);

  // Get domain corners
  ierr = DMStagGetGlobalSizes(dm, &Nx, &Nz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  
// Get dm coordinates array
  ierr = DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,DMSTAG_ELEMENT,&icenter);CHKERRQ(ierr); 

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      ierr = DMStagGetLocationSlot(dm,DMSTAG_ELEMENT,0,&idx); CHKERRQ(ierr);
      xxMMS[j][i][idx] = get_T(coordx[i][icenter],coordz[j][icenter]);
    }
  }

  // Restore arrays
  ierr = DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);

  // Restore and map local to global
  ierr = DMStagVecRestoreArray(dm,xMMSlocal,&xxMMS); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm,xMMSlocal,INSERT_VALUES,xMMS); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,xMMSlocal,INSERT_VALUES,xMMS); CHKERRQ(ierr);
  ierr = VecDestroy(&xMMSlocal); CHKERRQ(ierr);

  // Assign pointers
  *_xMMS  = xMMS;
  
  PetscFunctionReturn(0);
}

// ---------------------------------------
// ComputeErrorNorms
// ---------------------------------------
PetscErrorCode ComputeErrorNorms(DM dm,Vec x,Vec xMMS)
{
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz;
  PetscScalar    xx, xa, dx, dz, dv;
  PetscScalar    nrm, gnrm;
  Vec            xlocal, xalocal;
  MPI_Comm       comm;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  comm = PETSC_COMM_WORLD;

  // Get domain corners
  ierr = DMStagGetGlobalSizes(dm, &Nx, &Nz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  // Map global vectors to local domain
  ierr = DMGetLocalVector(dm, &xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm, x, INSERT_VALUES, xlocal); CHKERRQ(ierr);

  ierr = DMGetLocalVector(dm, &xalocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm, xMMS, INSERT_VALUES, xalocal); CHKERRQ(ierr);

  // Initialize norms
  nrm = 0.0;
  dx = 1.0/Nx;
  dz = 1.0/Nz;
  dv = dx*dz;

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      DMStagStencil  point;

      point.i = i; point.j = j; point.loc = DMSTAG_ELEMENT; point.c = 0;
      ierr = DMStagVecGetValuesStencil(dm,xlocal, 1,&point,&xx); CHKERRQ(ierr);
      ierr = DMStagVecGetValuesStencil(dm,xalocal,1,&point,&xa); CHKERRQ(ierr);

      // Calculate error norm
      nrm += PetscAbsScalar(xx-xa)*dv;
    }
  }

  // Collect data 
  ierr = MPI_Allreduce(&nrm, &gnrm, 1, MPI_DOUBLE, MPI_SUM, comm); CHKERRQ(ierr);

  // Restore arrays and vectors
  ierr = DMRestoreLocalVector(dm,&xlocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&xalocal); CHKERRQ(ierr);

  // Print information
  PetscPrintf(comm,"# NORMS: \n");
  PetscPrintf(comm,"# Solution: norm1 = %1.12e\n",gnrm);
  PetscPrintf(comm,"# Grid info: hx = %1.12e hz = %1.12e \n",dx,dz);

  PetscFunctionReturn(0);
}
