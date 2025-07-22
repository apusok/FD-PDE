// ---------------------------------------
// STANDALONE Benchmark for the phase-equation method
// run: ./test_dmstag_phasemethod_ -nx 40 -nz 40 -icase (0/1/2/3/4/5) -dt 0.001 -tstep 10 -gamma 1 -eps 0.025 -log_view
// python test 1: ./python/test_dmstag_phasemethod_stationary.py (for icase = 0, 1)
// python test 2: ./python/test_dmstag_phasemethod_flow.py (for icase = 2, 3, 4, 5)
// ---------------------------------------
static char help[] = "Application to benchmark the phase-equation method to capture the interface between two fluids (Using DMStag) \n\n";

// define convenient names for DMStagStencilLocation
#define ELEMENT    DMSTAG_ELEMENT

#include "../src/fdpde_dmstag.h"

// ---------------------------------------
// Application Context
// ---------------------------------------
#define FNAME_LENGTH 200

// parameters (bag)
typedef struct {
  PetscInt       nx, nz, tstep;
  PetscInt       adv_scheme,ts_scheme,tout;
  PetscInt       icase;
  PetscScalar    L, H;
  PetscScalar    xmin, zmin, zw;
  PetscScalar    gamma, eps, ux, uz;
  PetscScalar    t, dt;
  char           fname_out[FNAME_LENGTH]; 
  char           fname_in [FNAME_LENGTH];  
  char           fdir_out[FNAME_LENGTH]; 
} Params;

// user defined and model-dependent variables
typedef struct {
  Params        *par;
  PetscBag       bag;
  PetscMPIInt    rank;
  DM             dmf;
  Vec            f, fprev, dfx, dfz;
  MPI_Comm       comm;
} UsrData;

// ---------------------------------------
// Function definitions
// ---------------------------------------
PetscErrorCode Numerical_solution(void*,PetscInt);
PetscErrorCode InputParameters(UsrData**);
PetscErrorCode InputPrintData(UsrData*);
PetscErrorCode SetInitialField(DM,Vec,void*);
PetscErrorCode UpdateGrad(DM,Vec,void*);
PetscErrorCode ExplicitStep(DM,Vec,Vec,PetscScalar,void*);

// ---------------------------------------
// Application functions
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "Numerical_solution"
PetscErrorCode Numerical_solution(void *ctx,PetscInt ts_scheme)
{
  UsrData       *usr = (UsrData*) ctx;
  DM             dmf;
  Vec            f, fprev, dfx, dfz;
  PetscInt       nx, nz, istep, tstep;
  PetscInt       dof, cs;
  PetscScalar    xmin, zmin, xmax, zmax, dt, ux, uz;
  char           fout[FNAME_LENGTH];
  char           date[30];
  PetscFunctionBeginUser;

  // Element count
  nx = usr->par->nx;
  nz = usr->par->nz;

  // velocity
  ux = usr->par->ux;
  uz = usr->par->uz;

  istep = 0;
  tstep = usr->par->tstep;
  dt = usr->par->dt;

  // Get the domain ready
  xmin = usr->par->xmin;
  zmin = usr->par->zmin;
  xmax = usr->par->xmin+usr->par->L;
  zmax = usr->par->zmin+usr->par->H;

  // Create the dmf object and coordinates
  /* set up solution and residual vectors */
  PetscCall(DMStagCreate2d(usr->comm,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,nx,nz,PETSC_DECIDE,PETSC_DECIDE,0,0,1,
		        DMSTAG_STENCIL_BOX,1,NULL,NULL,&usr->dmf));  
  PetscCall(DMSetFromOptions(usr->dmf)); 
  PetscCall(DMSetUp(usr->dmf)); 
  PetscCall(DMStagSetUniformCoordinatesProduct(usr->dmf,xmin,xmax,zmin,zmax,0.0,0.0));

  // Create global vectors for f, dfx and dfz
  PetscCall(DMCreateGlobalVector(usr->dmf,&usr->f));
  PetscCall(VecDuplicate(usr->f,&usr->fprev));
  PetscCall(VecDuplicate(usr->f,&usr->dfx));
  PetscCall(VecDuplicate(usr->f,&usr->dfz));

  // short names
  dmf   = usr->dmf;
  f     = usr->f;
  fprev = usr->fprev;
  dfx   = usr->dfx;
  dfz   = usr->dfz;
  
  // Create local vectors
  //PetscCall(DMGetLocalVector(usr->dmf,&flocal));

  /* report contents of parameter structure */
  dof = nx*nz;
  PetscCall(MPI_Comm_size(usr->comm,&cs));
  PetscCall(PetscGetDate(date,30));
  PetscCall(PetscPrintf(usr->comm,"---------------Phasefield: %s-----------------\n",&(date[0])));
  PetscCall(PetscPrintf(usr->comm,"Processes: %d, DOFs/Process: %d \n",cs,dof/cs));
  PetscCall(PetscPrintf(usr->comm,"--------------------------------------\n"));
  PetscCall(PetscBagView(usr->bag,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscPrintf(usr->comm,"--------------------------------------\n"));

  // get initial distribution
  PetscCall(SetInitialField(dmf,f,usr));

  // copy f to fprev
  PetscCall(VecCopy(f, fprev));
  
  // output - initial state
  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s_initial",usr->par->fdir_out,usr->par->fname_out));
  PetscCall(DMStagViewBinaryPython(dmf,f,fout));
  

  // Time loop
  while (istep < tstep) {
    PetscPrintf(PETSC_COMM_WORLD,"# Timestep %d out of %d: time %1.4f\n\n",istep,tstep,usr->par->t);

    // update vector dfx and dfz
    PetscCall(UpdateGrad(dmf, fprev, usr)); 

    // solve the equation explicitly
    PetscCall(ExplicitStep(dmf, fprev, f, dt, usr));

    // increment time
    usr->par->t += dt;

    // 2nd order runge-kutta
    {
      Vec hk1, hk2, f_bk, fprev_bk; 
      
      // allocate storage for hk1, hk2, f_bk, fprev_bk
      PetscCall(VecDuplicate(f, &hk1)); 
      PetscCall(VecDuplicate(f, &hk2)); 
      PetscCall(VecDuplicate(f, &f_bk)); 
      PetscCall(VecDuplicate(f, &fprev_bk)); 

      // backup x and xprev
      PetscCall(VecCopy(f, f_bk)); 
      PetscCall(VecCopy(fprev, fprev_bk)); 
      
      // 1st stage - get h*k1 = f- fprev
      PetscCall(VecCopy(f, hk1)); 
      PetscCall(VecAXPY(hk1, -1.0, fprev));
      
      // 2nd stage - (t = t+0.5*dt, fprev = fprev + 0.5*hk1)
      PetscCall(VecCopy(fprev_bk, fprev)); 
      PetscCall(VecAXPY(fprev, 0.5, hk1)); 

      // correct time by half step
      usr->par->t -= 0.5*dt;
      
      // update dfx and dfz and solve for the second stage
      PetscCall(UpdateGrad(dmf, fprev, usr)); 
      PetscCall(ExplicitStep(dmf, fprev, f, dt, usr));

      // get hk2 and update the full step
      PetscCall(VecCopy(f, hk2)); 
      PetscCall(VecAXPY(hk2, -1.0, fprev));
      PetscCall(VecCopy(fprev_bk, fprev)); 
      PetscCall(VecCopy(fprev, f)); 
      PetscCall(VecAXPY(f, 1.0, hk2));

      // reset time
      usr->par->t += 0.5*dt;
      
      // check if hk1 and hk2 are zeros or NANs
      PetscScalar hk1norm, hk2norm;
      PetscCall(VecNorm(hk1, NORM_1, &hk1norm));
      PetscCall(VecNorm(hk2, NORM_1, &hk2norm));
      PetscPrintf(PETSC_COMM_WORLD, "hk1norm=%g, hk2norm=%g \n", hk1norm, hk2norm);
      
      // destroy vectors after use
      PetscCall(VecDestroy(&f_bk));
      PetscCall(VecDestroy(&fprev_bk));
      PetscCall(VecDestroy(&hk1));
      PetscCall(VecDestroy(&hk2));
    }

    // store fprev and f in dm
    usr->f     = f;
    usr->fprev = fprev;
    
    // reverse the flow after t = 0.4 , for case 2.
    if ((usr->par->icase == 2) && (usr->par->t >= 0.4)) {
      usr->par->ux = -ux;
      usr->par->uz = -uz;
    }

    // Copy f to fprev
    PetscCall(VecCopy(f,fprev));
    
    // Output solution
    if (istep % usr->par->tout == 0 ) {
      PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s_m%d_ts%1.3d",usr->par->fdir_out,usr->par->fname_out,ts_scheme,istep));
      PetscCall(DMStagViewBinaryPython(dmf,f,fout));

    }

    // Destroy objects
    //    PetscCall(VecDestroy(&f));

    // increment timestep
    istep++;
  }

  // Destroy vec and dm
  PetscCall(VecDestroy(&fprev));
  PetscCall(VecDestroy(&dfx));
  PetscCall(VecDestroy(&dfz));
  PetscCall(VecDestroy(&f));
  PetscCall(DMDestroy(&dmf)); 

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

  // Numerical solution
  PetscCall(Numerical_solution(usr,usr->par->ts_scheme));  // 0-1st, 1-rk2, 2-rk4

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
  PetscCall(PetscBagRegisterInt(bag, &par->tstep, 1, "tstep", "Number of time steps")); 

  PetscCall(PetscBagRegisterInt(bag, &par->ts_scheme,0, "ts_scheme", "Time stepping scheme")); 
  PetscCall(PetscBagRegisterInt(bag, &par->adv_scheme,0, "adv_scheme", "Advection scheme 0-center, 1-upwind1, 2-upwind2")); 
  PetscCall(PetscBagRegisterInt(bag, &par->tout,5,"tout", "Output every <tout> time steps")); 

  PetscCall(PetscBagRegisterScalar(bag, &par->xmin, 0.0, "xmin", "Start coordinate of domain in x-dir")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->zmin, 0.0, "zmin", "Start coordinate of domain in z-dir")); 

  PetscCall(PetscBagRegisterScalar(bag, &par->L, 1.0, "L", "Length of domain in x-dir")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->H, 1.0, "H", "Height of domain in z-dir")); 

  // Parameters relevant to the interface and the phase field method
  PetscCall(PetscBagRegisterScalar(bag, &par->zw, 0.6, "zw", "Location of a horizontal interface")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->eps, 0.2, "eps", "epsilon in the kernel function")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->gamma, 1.0, "gamma", "gamma in the phase field method")); 
  
  // Physical and material parameters
  PetscCall(PetscBagRegisterScalar(bag, &par->ux, 0.0, "ux", "Horizontal velocity")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->uz, 0.0, "uz", "Vertical velocity")); 

  PetscCall(PetscBagRegisterScalar(bag, &par->dt, 1.0e-2, "dt", "Time step size")); 

  // Input/output 
  PetscCall(PetscBagRegisterString(bag,&par->fname_out,FNAME_LENGTH,"out_num_advtime","output_file","Name for output file, set with: -output_file <filename>")); 
  PetscCall(PetscBagRegisterString(bag,&par->fdir_out,FNAME_LENGTH,"./","output_dir","Name for output directory, set with: -output_dir <dirname>")); 

  // case number: 0: flat interface, 1: circular interface
  PetscCall(PetscBagRegisterInt(bag, &par->icase, 0, "icase", "Case number: 0 - flat, 1 - circular")); 
  
  // Other variables
  par->fname_in[0] = '\0';
  par->t = 0.0;

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
  PetscPrintf(usr->comm,"# Test_advdiff_advtime: %s \n",&(date[0]));
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
// SetInitialField
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "SetInitialField"
PetscErrorCode SetInitialField(DM dm, Vec x, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  Vec            xlocal;
  PetscInt       i,j, sx, sz, nx, nz, icenter, icase;
  PetscScalar    zw,eps;
  PetscScalar    ***xx, **coordx, **coordz;
  PetscFunctionBeginUser;

  // some useful parameters
  zw = usr->par->zw;
  eps = usr->par->eps;
  icase = usr->par->icase;
  

  // Get domain corners
  PetscCall(DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 

  // Get dm coordinates array
  PetscCall(DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter));

  // Create coefficient local vector
  PetscCall(DMCreateLocalVector(dm, &xlocal)); 
  PetscCall(DMStagVecGetArray(dm, xlocal, &xx)); 

  // Loop over local domain - set initial density and viscosity
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      DMStagStencil point;
      PetscScalar   xp,zp, fval = 0.0;
      PetscInt      idx;

      point.i = i; point.j = j; point.loc = ELEMENT; point.c = 0;
      xp = coordx[i][icenter];
      zp = coordz[j][icenter];

      if (icase == 0) {
        // flat interface
        if (zp<=zw) {fval = 0.0;}
        else {fval = 1.0;}

      }
      if (icase == 1) {
        // circular interface
        PetscScalar  xc=0.5,zc=0.5,rc = 0.2;
        PetscScalar  rp;

        rp = PetscPowScalar((xp-xc)*(xp-xc) + (zp-zc)*(zp-zc),0.5);
        if (rp<=rc) fval = 0.0;
        else        fval = 1.0;
      }

      if (icase == 2) {
        // circular interface - kernel function initialisation
        PetscScalar  xc=0.3,zc=0.3,rc = 0.15;
        PetscScalar  rp;

        rp = PetscPowScalar((xp-xc)*(xp-xc) + (zp-zc)*(zp-zc),0.5);
        fval = 0.5*(1 + PetscTanhScalar((rp-rc)/2.0/eps));
      }

      if (icase == 3 || icase ==4) {
        // circular interface - kernel function initialisation
        PetscScalar  xc=0.5,zc=0.5,rc = 0.2;
        PetscScalar  rp;

        rp = PetscPowScalar((xp-xc)*(xp-xc) + (zp-zc)*(zp-zc),0.5);
        fval = 0.5*(1 + PetscTanhScalar((rp-rc)/2.0/eps));
      }

      if (icase == 5) {
        // circular interface - kernel function initialisation
        PetscScalar  xc=0.5,zc=0.75,rc = 0.15;
        PetscScalar  rp;

        rp = PetscPowScalar((xp-xc)*(xp-xc) + (zp-zc)*(zp-zc),0.5);
        fval = 0.5*(1 + PetscTanhScalar((rp-rc)/2.0/eps));
      }
            
      PetscCall(DMStagGetLocationSlot(dm, point.loc, point.c, &idx)); 
      xx[j][i][idx] = fval;
    }
  }

  // Restore arrays, local vectors
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));

  PetscCall(DMStagVecRestoreArray(dm,xlocal,&xx));
  PetscCall(DMLocalToGlobalBegin(dm,xlocal,INSERT_VALUES,x)); 
  PetscCall(DMLocalToGlobalEnd  (dm,xlocal,INSERT_VALUES,x)); 
  
  PetscCall(VecDestroy(&xlocal)); 

  PetscFunctionReturn(PETSC_SUCCESS);
}



// ---------------------------------------
// Update dfdx and dfdz
// ---------------------------------------
PetscErrorCode UpdateGrad(DM dm, Vec x, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz;
  PetscInt       icenter, idx;
  PetscScalar    ***df1, ***df2;
  PetscScalar    **coordx,**coordz;
  Vec            dfxlocal, dfzlocal, xlocal;
  PetscFunctionBeginUser;

  // Local vectors
  PetscCall(DMGetLocalVector(dm,&xlocal)); 
  PetscCall(DMGlobalToLocal (dm,x,INSERT_VALUES,xlocal)); 

  PetscCall(DMCreateLocalVector(dm, &dfxlocal)); 
  PetscCall(DMStagVecGetArray(dm, dfxlocal, &df1)); 

  PetscCall(DMCreateLocalVector(dm, &dfzlocal)); 
  PetscCall(DMStagVecGetArray(dm, dfzlocal, &df2)); 
  
  // Get domain corners
  PetscCall(DMStagGetGlobalSizes(dm, &Nx, &Nz,NULL));
  PetscCall(DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL));  
  
  // Get dm coordinates array
  PetscCall(DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter));

  // Get location slot
  PetscCall(DMStagGetLocationSlot(dm, ELEMENT, 0, &idx)); 

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      DMStagStencil  point[4];
      PetscScalar    dx, dz, fval[4];

      // df/dx, df/dz: center
      point[0].i = i-1; point[0].j = j  ; point[0].loc = ELEMENT; point[0].c = 0;
      point[1].i = i+1; point[1].j = j  ; point[1].loc = ELEMENT; point[1].c = 0;
      point[2].i = i  ; point[2].j = j-1; point[2].loc = ELEMENT; point[2].c = 0;
      point[3].i = i  ; point[3].j = j+1; point[3].loc = ELEMENT; point[3].c = 0;

      dx = coordx[i+1][icenter]-coordx[i][icenter];
      dz = coordz[j+1][icenter]-coordz[j][icenter];

      if ((i!=0) && (i!=Nx-1)) {
        dx = coordx[i+1][icenter] -  coordx[i-1][icenter];
      }
      if ((j!=0) && (j!=Nz-1)) {
        dz = coordz[j+1][icenter] -  coordz[j-1][icenter];
      }

      // fix the boundary cell
      if (i == 0) {
        point[0].i = i; dx = coordx[i+1][icenter] - coordx[i][icenter];
      }
      if (i == Nx-1) {
        point[1].i = i; dx = coordx[i][icenter] - coordx[i-1][icenter];
      }
      if (j == 0) {
        point[2].j = j; dz = coordz[j+1][icenter] - coordz[j][icenter];
      }
      if (j == Nz-1) {
        point[3].j = j; dz = coordz[j][icenter] - coordz[j-1][icenter];
      }

      PetscCall(DMStagVecGetValuesStencil(dm, xlocal, 4, point, fval)); 

      df1[j][i][idx] = (fval[1] - fval[0])/dx;
      df2[j][i][idx] = (fval[3] - fval[2])/dz;

    }
  }

  // Restore arrays
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));

  // Restore and map local to global
  PetscCall(DMStagVecRestoreArray(dm,dfxlocal,&df1)); 
  PetscCall(DMLocalToGlobalBegin(dm,dfxlocal,INSERT_VALUES,usr->dfx)); 
  PetscCall(DMLocalToGlobalEnd  (dm,dfxlocal,INSERT_VALUES,usr->dfx)); 
  PetscCall(VecDestroy(&dfxlocal)); 

  PetscCall(DMStagVecRestoreArray(dm,dfzlocal,&df2)); 
  PetscCall(DMLocalToGlobalBegin(dm,dfzlocal,INSERT_VALUES,usr->dfz)); 
  PetscCall(DMLocalToGlobalEnd  (dm,dfzlocal,INSERT_VALUES,usr->dfz)); 
  PetscCall(VecDestroy(&dfzlocal)); 

  PetscCall(DMRestoreLocalVector(dm, &xlocal )); 
  
  PetscFunctionReturn(PETSC_SUCCESS);
}


/* ------------------------------------------------------------------- */
PetscErrorCode ExplicitStep(DM dm, Vec xprev, Vec x, PetscScalar dt, void *ctx)
/* ------------------------------------------------------------------- */
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz, idx, icenter;
  PetscScalar    v[2], gamma, eps, t, period, cost;
  PetscScalar    **coordx,**coordz;
  PetscScalar    ***xx,***xxp;
  Vec            dfx, dfz, dfxlocal, dfzlocal, xlocal, xplocal;
  PetscFunctionBeginUser;

  // User Parameter
  gamma = usr->par->gamma;
  eps = usr->par->eps;
  v[0]= usr->par->ux;
  v[1]= usr->par->uz;

  t = usr->par->t;
  period = 4;
  cost = PetscCosScalar(M_PI * t/period);

  dfx = usr->dfx;
  dfz = usr->dfz;

  // Get domain corners
  PetscCall(DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 
  // Get global size
  PetscCall(DMStagGetGlobalSizes(dm, &Nx, &Nz,NULL));

  // Create local vector
  PetscCall(DMGetLocalVector(dm,&xplocal)); 
  PetscCall(DMGlobalToLocalBegin (dm,xprev,INSERT_VALUES,xplocal)); 
  PetscCall(DMGlobalToLocalEnd (dm,xprev,INSERT_VALUES,xplocal));

  PetscCall(DMGetLocalVector(dm,&xlocal)); 
  PetscCall(DMGlobalToLocalBegin (dm,x,INSERT_VALUES,xlocal)); 
  PetscCall(DMGlobalToLocalEnd (dm,x,INSERT_VALUES,xlocal)); 

  PetscCall(DMGetLocalVector(dm,&dfxlocal)); 
  PetscCall(DMGlobalToLocalBegin (dm,dfx,INSERT_VALUES,dfxlocal)); 
  PetscCall(DMGlobalToLocalEnd (dm,dfx,INSERT_VALUES,dfxlocal)); 

  PetscCall(DMGetLocalVector(dm,&dfzlocal)); 
  PetscCall(DMGlobalToLocalBegin (dm,dfz,INSERT_VALUES,dfzlocal)); 
  PetscCall(DMGlobalToLocalEnd (dm,dfz,INSERT_VALUES,dfzlocal)); 

  // get array from xlocal
  PetscCall(DMStagVecGetArray(dm, xlocal, &xx)); 
  PetscCall(DMStagVecGetArray(dm, xplocal, &xxp)); 

  // Get dm coordinates array
  PetscCall(DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter));

  // get the location slot
  PetscCall(DMStagGetLocationSlot(dm, ELEMENT, 0, &idx)); 
  
  // excluding the boundary points  
  if (sz==0) {sz++; nz--;}
  if (sz+nz==Nz) {nz--;}
  if (sx==0) {sx++; nx--;}
  if (sx+nx==Nx) {nx--;}

  // loop over local domain and get the RHS value
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i < sx+nx; i++) {

      DMStagStencil point[5];
      PetscInt      ii;
      PetscScalar   fe[5], dfxe[5], dfze[5], gfe[5], c[5], fval = 0.0;
      PetscScalar    dx2, dz2;
      PetscScalar   dx, dz;

      point[0].i = i;   point[0].j = j;   point[0].loc = ELEMENT; point[0].c = 0;
      point[1].i = i-1; point[1].j = j;   point[1].loc = ELEMENT; point[1].c = 0;
      point[2].i = i+1; point[2].j = j;   point[2].loc = ELEMENT; point[2].c = 0;
      point[3].i = i;   point[3].j = j-1; point[3].loc = ELEMENT; point[3].c = 0;
      point[4].i = i;   point[4].j = j+1; point[4].loc = ELEMENT; point[4].c = 0;

      // for uniform grids
      dx2 = coordx[i+1][icenter] -  coordx[i-1][icenter];
      dz2 = coordz[j+1][icenter] -  coordz[j-1][icenter];
      dx = 0.5*dx2;
      dz = 0.5*dz2;
      
      // default zero flux on boundary
      if (i==0)    {point[1] = point[0]; dx2 = coordx[i+1][icenter] -  coordx[i  ][icenter];}
      if (i==Nx-1) {point[2] = point[0]; dx2 = coordx[i  ][icenter] -  coordx[i-1][icenter];}
      if (j==0)    {point[3] = point[0]; dz2 = coordz[j+1][icenter] -  coordz[j  ][icenter];}
      if (j==Nz-1) {point[4] = point[0]; dz2 = coordz[j  ][icenter] -  coordz[j-1][icenter];}

      

      PetscCall(DMStagVecGetValuesStencil(dm,dfxlocal,5,point,dfxe)); 
      PetscCall(DMStagVecGetValuesStencil(dm,dfzlocal,5,point,dfze)); 
      PetscCall(DMStagVecGetValuesStencil(dm,xplocal ,5,point,fe)); 

      for (ii=1; ii<5; ii++) {

        PetscScalar epsAlt;  //coefficients of anti-diffusion, center

        gfe[ii] = sqrt(dfxe[ii]*dfxe[ii]+dfze[ii]*dfze[ii]);

        if (gfe[ii] > 1e-10) {epsAlt = fe[ii]*(1-fe[ii])/gfe[ii];}
        else {epsAlt = eps;}

        c[ii] = epsAlt; //coefficients at the center

      }

      //diffusion terms
      fval = gamma*(eps * ((fe[2]+fe[1]-2*fe[0])/dx/dx + (fe[4]+fe[3]-2*fe[0])/dz/dz));
      //sharpen terms
      fval -= gamma* ( (c[2]*dfxe[2] - c[1]*dfxe[1])/dx2 + (c[4]*dfze[4]-c[3]*dfze[3])/dz2);

      { // velocity on the edge and advection terms
        PetscScalar xe[3],ze[3],xp[9], zp[9], ux[9], uz[9];
        PetscInt    ii;

        xe[0] = coordx[i  ][icenter];
        xe[1] = coordx[i-1][icenter];
        xe[2] = coordx[i+1][icenter];

        ze[0] = coordz[j  ][icenter];
        ze[1] = coordz[j-1][icenter];
        ze[2] = coordz[j+1][icenter];

        xp[0] = xe[0];
        xp[1] = 0.5*(xe[0]+xe[1]);
        xp[2] = 0.5*(xe[0]+xe[2]);
        xp[3] = xe[0];
        xp[4] = xe[0];
        xp[5] = xe[1] - dx/2.0;
        xp[6] = xe[2] + dx/2.0;
        xp[7] = xe[0];
        xp[8] = xe[0];

        zp[0] = ze[0];
        zp[1] = ze[0];
        zp[2] = ze[0];
        zp[3] = 0.5*(ze[0]+ze[1]);
        zp[4] = 0.5*(ze[0]+ze[2]);
        zp[5] = ze[0];
        zp[6] = ze[0];
        zp[7] = ze[1] - dz/2.0;
        zp[8] = ze[2] + dz/2.0;
        
        // get u,v on faces
        for (ii=0; ii<9; ii++) {

          if (usr->par->icase <= 2) {ux[ii] = v[0]; uz[ii] = v[1];}
          if (usr->par->icase == 3) {ux[ii] = -1.0+2.0*xp[ii]; uz[ii] = 1.0-2.0*zp[ii];}
          if (usr->par->icase == 4) {ux[ii] = -1.0+2.0*zp[ii]; uz[ii] = 0.0;}
          if (usr->par->icase == 5) {
            ux[ii] = PetscPowScalar(PetscSinScalar(M_PI*xp[ii]),2) * PetscSinScalar(2*M_PI*zp[ii]) * cost; 
            uz[ii] = - PetscPowScalar(PetscSinScalar(M_PI*zp[ii]),2) * PetscSinScalar(2*M_PI*xp[ii]) * cost;
          }
        }

        // central difference method
        fval -= 0.5*(ux[2]*(fe[2]+fe[0]) - ux[1]*(fe[1]+fe[0]))/dx + 0.5*(uz[4]*(fe[4]+fe[0]) - uz[3]*(fe[3]+fe[0]))/dz; 

      }

      xx[j][i][idx] = xxp[j][i][idx] + dt*fval;
    }
  }
                                
  // reset sx, sz, nx, nz
  PetscCall(DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 
  // apply boundary conditions : zero flux
  if (sx==0) {
    for (j = sz; j<sz+nz; j++) {
      xx[j][0][idx] = xx[j][1][idx]; 
    }
  }

  if (sx+nx==Nx) {
    for (j = sz; j<sz+nz; j++) {
      xx[j][Nx-1][idx] = xx[j][Nx-2][idx];
    }
  }

  if (sz==0) {
    for (i = sx; i<sx+nx; i++) {
      xx[0][i][idx] = xx[1][i][idx];
    }
  }

  if (sz+nz==Nz) {
    for (i = sx; i<sx+nx; i++) {
      xx[Nz-1][i][idx] = xx[Nz-2][i][idx];
    }
  }

  // Restore arrays, local vectors
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));
  PetscCall(DMStagVecRestoreArray(dm,xlocal,&xx));
  PetscCall(DMLocalToGlobalBegin(dm,xlocal,INSERT_VALUES,x)); 
  PetscCall(DMLocalToGlobalEnd  (dm,xlocal,INSERT_VALUES,x)); 
  PetscCall(DMRestoreLocalVector(dm, &xlocal)); 

  PetscCall(DMStagVecRestoreArray(dm,xplocal,&xxp));
  PetscCall(DMLocalToGlobalBegin(dm,xplocal,INSERT_VALUES,xprev)); 
  PetscCall(DMLocalToGlobalEnd  (dm,xplocal,INSERT_VALUES,xprev)); 
  PetscCall(DMRestoreLocalVector(dm, &xplocal)); 

  PetscCall(DMRestoreLocalVector(dm, &dfxlocal)); 
  PetscCall(DMRestoreLocalVector(dm, &dfzlocal)); 

  PetscFunctionReturn(PETSC_SUCCESS);
}
