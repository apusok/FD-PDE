// ---------------------------------------
// MMS 2D diffusion test div(k*grad(T)) = f, with k > 0
// run: ./test_advdiff_mms_2d_diffusion -pc_type lu -pc_factor_mat_solver_type umfpack -pc_factor_mat_ordering_type external -nx 10 -nz 10 -log_view
// python test: ./python/test_advdiff_mms_2d_diffusion.py
// python sympy: ./mms/mms_2d_diffusion.py
// ---------------------------------------
static char help[] = "Application to solve an MMS 2D diffusion equation (ADVDIFF) with FD-PDE \n\n";

#include "../new_src/fdpde_advdiff.h"

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
    
  // Initialize application
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscTime(&start_time)); 

  comm = PETSC_COMM_WORLD;
  nx   = 10;
  nz   = 10;
  L    = 1.0;
  H    = 1.0;

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nx", &nx, NULL)); 
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nz", &nz, NULL)); 

  PetscCall(PetscOptionsGetScalar(NULL, NULL, "-L", &L, NULL)); 
  PetscCall(PetscOptionsGetScalar(NULL, NULL, "-H", &H, NULL)); 

  PetscCall(PetscOptionsGetString(NULL,NULL,"-fname",fname,sizeof(fname),NULL)); 
  PetscCall(PetscOptionsGetString(NULL,NULL,"-fdir",fdir,sizeof(fdir),NULL)); 

  PetscCall(PetscOptionsInsert(PETSC_NULLPTR,&argc,&argv,NULL)); 

  // Create the FD-PDE object
  PetscCall(FDPDECreate(comm,nx,nz,0.0,L,0.0,H,FDPDE_ADVDIFF,&fd));
  PetscCall(FDPDESetUp(fd));

  PetscCall(FDPDEAdvDiffSetAdvectSchemeType(fd,ADV_NONE));
  PetscCall(FDPDEAdvDiffSetTimeStepSchemeType(fd,TS_NONE));

  // Set coefficient and BC evaluation function
  PetscCall(FDPDESetFunctionBCList(fd,FormBCList,NULL,NULL)); 
  PetscCall(FDPDESetFunctionCoefficient(fd,FormCoefficient,NULL,NULL)); 

  // Compute manufactured solution
  PetscCall(FDPDEGetDM(fd, &dm)); 
  PetscCall(ComputeManufacturedSolution(dm,&xMMS));

  // Solve
  PetscCall(FDPDESolve(fd,NULL));
  PetscCall(FDPDEGetSolution(fd,&x)); 

  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s",fdir,fname));
  PetscCall(DMStagViewBinaryPython(dm,x,fout));

  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s",fdir,"out_mms_solution"));
  PetscCall(DMStagViewBinaryPython(dm,xMMS,fout));

  // Compute error
  PetscCall(ComputeErrorNorms(dm,x,xMMS));

  // Destroy objects
  PetscCall(DMDestroy(&dm)); 
  PetscCall(VecDestroy(&x)); 
  PetscCall(VecDestroy(&xMMS)); 
  PetscCall(FDPDEDestroy(&fd));

  PetscCall(PetscTime(&end_time)); 
  PetscPrintf(PETSC_COMM_WORLD,"# Total runtime: %g (sec) \n", end_time - start_time);
  PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");
  
  // Finalize main
  PetscCall(PetscFinalize());
  return 0;
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
  PetscFunctionBeginUser;
  
  // Left
  PetscCall(DMStagBCListGetValues(bclist,'w','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_T(x_bc[2*k],x_bc[2*k+1]);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
  
  // RIGHT
  PetscCall(DMStagBCListGetValues(bclist,'e','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_T(x_bc[2*k],x_bc[2*k+1]);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));

  // DOWN
  PetscCall(DMStagBCListGetValues(bclist,'s','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_T(x_bc[2*k],x_bc[2*k+1]);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));

  // UP
  PetscCall(DMStagBCListGetValues(bclist,'n','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_T(x_bc[2*k],x_bc[2*k+1]);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
 
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscFunctionBeginUser;

  // Element: A = 0 (dof 0), C = -f (dof 1)
  // Edges: B = -k (dof 0), u = 0 (dof 1)

  // Get domain corners
  PetscCall(DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 

  // Create coefficient local vector
  PetscCall(DMCreateLocalVector(dmcoeff, &coefflocal)); 
  PetscCall(DMStagVecGetArray(dmcoeff, coefflocal, &c)); 
  
  PetscCall(DMStagGetProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmcoeff,DMSTAG_LEFT,&iprev));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmcoeff,DMSTAG_RIGHT,&inext)); 
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmcoeff,DMSTAG_ELEMENT,&icenter));
  
  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      PetscInt      ii, idx;

      { // A = 0.0
        DMStagStencil point;
        point.i = i; point.j = j; point.loc = DMSTAG_ELEMENT;  point.c = 0;
        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
        c[j][i][idx] = 0.0;
      }

      { // C = -f (mms)
        DMStagStencil point;
        point.i = i; point.j = j; point.loc = DMSTAG_ELEMENT;  point.c = 1;
        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
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
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
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
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] = 0.0;
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
// Compute manufactured solution
// ---------------------------------------
PetscErrorCode ComputeManufacturedSolution(DM dm,Vec *_xMMS)
{
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz, idx, icenter;
  PetscScalar    ***xxMMS;
  PetscScalar    **coordx,**coordz;
  Vec            xMMS, xMMSlocal;
  PetscFunctionBeginUser;

  // Create local and global vectors for mms 
  PetscCall(DMCreateGlobalVector(dm,&xMMS)); 
  PetscCall(DMCreateLocalVector (dm,&xMMSlocal)); 
  PetscCall(DMStagVecGetArray(dm,xMMSlocal,&xxMMS)); 

  // Get domain corners
  PetscCall(DMStagGetGlobalSizes(dm, &Nx, &Nz,NULL));
  PetscCall(DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 
  
// Get dm coordinates array
  PetscCall(DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm,DMSTAG_ELEMENT,&icenter)); 

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      PetscCall(DMStagGetLocationSlot(dm,DMSTAG_ELEMENT,0,&idx)); 
      xxMMS[j][i][idx] = get_T(coordx[i][icenter],coordz[j][icenter]);
    }
  }

  // Restore arrays
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));

  // Restore and map local to global
  PetscCall(DMStagVecRestoreArray(dm,xMMSlocal,&xxMMS)); 
  PetscCall(DMLocalToGlobalBegin(dm,xMMSlocal,INSERT_VALUES,xMMS)); 
  PetscCall(DMLocalToGlobalEnd  (dm,xMMSlocal,INSERT_VALUES,xMMS)); 
  PetscCall(VecDestroy(&xMMSlocal)); 

  // Assign pointers
  *_xMMS  = xMMS;
  
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscFunctionBeginUser;

  comm = PETSC_COMM_WORLD;

  // Get domain corners
  PetscCall(DMStagGetGlobalSizes(dm, &Nx, &Nz,NULL));
  PetscCall(DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 

  // Map global vectors to local domain
  PetscCall(DMGetLocalVector(dm, &xlocal)); 
  PetscCall(DMGlobalToLocal (dm, x, INSERT_VALUES, xlocal)); 

  PetscCall(DMGetLocalVector(dm, &xalocal)); 
  PetscCall(DMGlobalToLocal (dm, xMMS, INSERT_VALUES, xalocal)); 

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
      PetscCall(DMStagVecGetValuesStencil(dm,xlocal, 1,&point,&xx)); 
      PetscCall(DMStagVecGetValuesStencil(dm,xalocal,1,&point,&xa)); 

      // Calculate error norm
      nrm += PetscAbsScalar(xx-xa)*dv;
    }
  }

  // Collect data 
  PetscCall(MPI_Allreduce(&nrm, &gnrm, 1, MPI_DOUBLE, MPI_SUM, comm)); 

  // Restore arrays and vectors
  PetscCall(DMRestoreLocalVector(dm,&xlocal)); 
  PetscCall(DMRestoreLocalVector(dm,&xalocal)); 

  // Print information
  PetscPrintf(comm,"# NORMS: \n");
  PetscPrintf(comm,"# Solution: norm1 = %1.12e\n",gnrm);
  PetscPrintf(comm,"# Grid info: hx = %1.12e hz = %1.12e \n",dx,dz);

  PetscFunctionReturn(PETSC_SUCCESS);
}
