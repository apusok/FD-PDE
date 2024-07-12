// ---------------------------------------
// STANDALONE Benchmark for the phase-equation method
// run: ./tests/test_dmstag_phasemethod.app -nx 40 -nz 40 -icase (0/1/2/3/4/5) -dt 0.001 -tstep 10 -gamma 1 -eps 0.025
// python test 1: ./tests/python/test_dmstag_phasemethod_stationary.py (for icase = 0, 1)
// python test 2: ./tests/python/test_dmstag_phasemethod_flow.py (for icase = 2, 3, 4, 5)
// ---------------------------------------
static char help[] = "Application to benchmark the phase-equation method to capture the interface between two fluids (Using DMStag) \n\n";

// define convenient names for DMStagStencilLocation
#define ELEMENT    DMSTAG_ELEMENT

#include "petsc.h"
#include "../src/dmstagoutput.h"

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
  Vec            f, fprev;
  PetscInt       nx, nz, istep, tstep;
  PetscInt       dof, cs;
  PetscScalar    xmin, zmin, xmax, zmax, dt, ux, uz;
  char           fout[FNAME_LENGTH];
  char           date[30];
  PetscErrorCode ierr;

  PetscFunctionBegin;

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
  ierr = DMStagCreate2d(usr->comm,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,nx,nz,
                        PETSC_DECIDE,PETSC_DECIDE,0,0,1,
		        DMSTAG_STENCIL_BOX,1,PETSC_NULL,PETSC_NULL,&usr->dmf);  CHKERRQ(ierr);
  ierr = DMSetFromOptions(usr->dmf); CHKERRQ(ierr);
  ierr = DMSetUp(usr->dmf); CHKERRQ(ierr);
  ierr = DMStagSetUniformCoordinatesProduct(usr->dmf,xmin,xmax,zmin,zmax,0.0,0.0);CHKERRQ(ierr);

  // Create global vectors for f, dfx and dfz
  ierr = DMCreateGlobalVector(usr->dmf,&usr->f);CHKERRQ(ierr);
  ierr = VecDuplicate(usr->f,&usr->fprev);CHKERRQ(ierr);
  ierr = VecDuplicate(usr->f,&usr->dfx);CHKERRQ(ierr);
  ierr = VecDuplicate(usr->f,&usr->dfz);CHKERRQ(ierr);

  // short names
  dmf   = usr->dmf;
  f     = usr->f;
  fprev = usr->fprev;
  
  // Create local vectors
  //ierr = DMGetLocalVector(usr->dmf,&flocal);CHKERRQ(ierr);

  /* report contents of parameter structure */
  dof = nx*nz;
  ierr = MPI_Comm_size(usr->comm,&cs);
  ierr = PetscGetDate(date,30);CHKERRQ(ierr);
  PetscPrintf(usr->comm,"---------------Phasefield: %s-----------------\n",&(date[0]));
  PetscPrintf(usr->comm,"Processes: %d, DOFs/Process: %d \n",cs,dof/cs);
  ierr = PetscPrintf(usr->comm,"--------------------------------------\n");CHKERRQ(ierr);
  ierr = PetscBagView(usr->bag,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscPrintf(usr->comm,"--------------------------------------\n");CHKERRQ(ierr);

  // get initial distribution
  ierr = SetInitialField(dmf,f,usr);CHKERRQ(ierr);

  // copy f to fprev
  ierr = VecCopy(f, fprev);CHKERRQ(ierr);
  
  // output - initial state
  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/%s_initial",usr->par->fdir_out,usr->par->fname_out);
  ierr = DMStagViewBinaryPython(dmf,f,fout);CHKERRQ(ierr);
  

  // Time loop
  while (istep < tstep) {
    PetscPrintf(PETSC_COMM_WORLD,"# Timestep %d out of %d: time %1.4f\n\n",istep,tstep,usr->par->t);

    // update vector dfx and dfz
    ierr = UpdateGrad(dmf, fprev, usr); CHKERRQ(ierr);

    // solve the equation explicitly
    ierr = ExplicitStep(dmf, fprev, f, dt, usr);CHKERRQ(ierr);

    // increment time
    usr->par->t += dt;

    // 2nd order runge-kutta
    {
      Vec hk1, hk2, f_bk, fprev_bk; 
      
      // allocate storage for hk1, hk2, f_bk, fprev_bk
      ierr = VecDuplicate(f, &hk1); CHKERRQ(ierr);
      ierr = VecDuplicate(f, &hk2); CHKERRQ(ierr);
      ierr = VecDuplicate(f, &f_bk); CHKERRQ(ierr);
      ierr = VecDuplicate(f, &fprev_bk); CHKERRQ(ierr);

      // backup x and xprev
      ierr = VecCopy(f, f_bk); CHKERRQ(ierr);
      ierr = VecCopy(fprev, fprev_bk); CHKERRQ(ierr);
      
      // 1st stage - get h*k1 = f- fprev
      ierr = VecCopy(f, hk1); CHKERRQ(ierr);
      ierr = VecAXPY(hk1, -1.0, fprev);CHKERRQ(ierr);
      
      // 2nd stage - (t = t+0.5*dt, fprev = fprev + 0.5*hk1)
      ierr = VecCopy(fprev_bk, fprev); CHKERRQ(ierr);
      ierr = VecAXPY(fprev, 0.5, hk1); CHKERRQ(ierr);

      // correct time by half step
      usr->par->t -= 0.5*dt;
      
      // update dfx and dfz and solve for the second stage
      ierr = UpdateGrad(dmf, fprev, usr); CHKERRQ(ierr);
      ierr = ExplicitStep(dmf, fprev, f, dt, usr);CHKERRQ(ierr);

      // get hk2 and update the full step
      ierr = VecCopy(f, hk2); CHKERRQ(ierr);
      ierr = VecAXPY(hk2, -1.0, fprev);CHKERRQ(ierr);
      ierr = VecCopy(fprev_bk, fprev); CHKERRQ(ierr);
      ierr = VecCopy(fprev, f); CHKERRQ(ierr);
      ierr = VecAXPY(f, 1.0, hk2);CHKERRQ(ierr);

      // reset time
      usr->par->t += 0.5*dt;
      
      // check if hk1 and hk2 are zeros or NANs
      PetscScalar hk1norm, hk2norm;
      ierr = VecNorm(hk1, NORM_1, &hk1norm);
      ierr = VecNorm(hk2, NORM_1, &hk2norm);
      PetscPrintf(PETSC_COMM_WORLD, "hk1norm=%g, hk2norm=%g \n", hk1norm, hk2norm);
      
      // destroy vectors after use
      ierr = VecDestroy(&f_bk);CHKERRQ(ierr);
      ierr = VecDestroy(&fprev_bk);CHKERRQ(ierr);
      ierr = VecDestroy(&hk1);CHKERRQ(ierr);
      ierr = VecDestroy(&hk2);CHKERRQ(ierr);
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
    ierr = VecCopy(f,fprev);CHKERRQ(ierr);
    
    // Output solution
    if (istep % usr->par->tout == 0 ) {
      ierr = PetscSNPrintf(fout,sizeof(fout),"%s/%s_m%d_ts%1.3d",usr->par->fdir_out,usr->par->fname_out,ts_scheme,istep);
      ierr = DMStagViewBinaryPython(dmf,f,fout);CHKERRQ(ierr);

    }

    // Destroy objects
    //    ierr = VecDestroy(&f);CHKERRQ(ierr);

    // increment timestep
    istep++;
  }

  // Destroy vec and dm
  ierr = VecDestroy(&fprev);CHKERRQ(ierr);
  ierr = VecDestroy(&f);CHKERRQ(ierr);
  ierr = DMDestroy(&dmf); CHKERRQ(ierr);

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

  // Print user parameters
  ierr = InputPrintData(usr); CHKERRQ(ierr);

  // Numerical solution
  ierr = Numerical_solution(usr,usr->par->ts_scheme); CHKERRQ(ierr); // 0-1st, 1-rk2, 2-rk4

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
  ierr = PetscBagRegisterInt(bag, &par->tstep, 1, "tstep", "Number of time steps"); CHKERRQ(ierr);

  ierr = PetscBagRegisterInt(bag, &par->ts_scheme,0, "ts_scheme", "Time stepping scheme"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->adv_scheme,0, "adv_scheme", "Advection scheme 0-center, 1-upwind1, 2-upwind2"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->tout,5,"tout", "Output every <tout> time steps"); CHKERRQ(ierr);

  ierr = PetscBagRegisterScalar(bag, &par->xmin, 0.0, "xmin", "Start coordinate of domain in x-dir"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->zmin, 0.0, "zmin", "Start coordinate of domain in z-dir"); CHKERRQ(ierr);

  ierr = PetscBagRegisterScalar(bag, &par->L, 1.0, "L", "Length of domain in x-dir"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->H, 1.0, "H", "Height of domain in z-dir"); CHKERRQ(ierr);

  // Parameters relevant to the interface and the phase field method
  ierr = PetscBagRegisterScalar(bag, &par->zw, 0.6, "zw", "Location of a horizontal interface"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->eps, 0.2, "eps", "epsilon in the kernel function"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->gamma, 1.0, "gamma", "gamma in the phase field method"); CHKERRQ(ierr);
  

  // Physical and material parameters
  ierr = PetscBagRegisterScalar(bag, &par->ux, 0.0, "ux", "Horizontal velocity"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->uz, 0.0, "uz", "Vertical velocity"); CHKERRQ(ierr);

  ierr = PetscBagRegisterScalar(bag, &par->dt, 1.0e-2, "dt", "Time step size"); CHKERRQ(ierr);

  // Input/output 
  ierr = PetscBagRegisterString(bag,&par->fname_out,FNAME_LENGTH,"out_num_advtime","output_file","Name for output file, set with: -output_file <filename>"); CHKERRQ(ierr);
  ierr = PetscBagRegisterString(bag,&par->fdir_out,FNAME_LENGTH,"./","output_dir","Name for output directory, set with: -output_dir <dirname>"); CHKERRQ(ierr);

  // case number: 0: flat interface, 1: circular interface
  ierr = PetscBagRegisterInt(bag, &par->icase, 0, "icase", "Case number: 0 - flat, 1 - circular"); CHKERRQ(ierr);
  
  // Other variables
  par->fname_in[0] = '\0';
  par->t = 0.0;

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
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Get date
  ierr = PetscGetDate(date,30); CHKERRQ(ierr);

  // Get petsc command options
  ierr = PetscOptionsGetAll(NULL, &opts); CHKERRQ(ierr);

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
  ierr = PetscBagView(usr->bag,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
  PetscPrintf(usr->comm,"# --------------------------------------- #\n");

  // Free memory
  ierr = PetscFree(opts); CHKERRQ(ierr);

  PetscFunctionReturn(0);
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

  PetscErrorCode ierr;
  PetscFunctionBegin;

  // some useful parameters
  zw = usr->par->zw;
  eps = usr->par->eps;
  icase = usr->par->icase;
  

  // Get domain corners
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  // Get dm coordinates array
  ierr = DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter);CHKERRQ(ierr);

  // Create coefficient local vector
  ierr = DMCreateLocalVector(dm, &xlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm, xlocal, &xx); CHKERRQ(ierr);

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
            
      ierr = DMStagGetLocationSlot(dm, point.loc, point.c, &idx); CHKERRQ(ierr);
      xx[j][i][idx] = fval;
    }
  }

  // Restore arrays, local vectors
  ierr = DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);

  ierr = DMStagVecRestoreArray(dm,xlocal,&xx);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm,xlocal,INSERT_VALUES,x); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,xlocal,INSERT_VALUES,x); CHKERRQ(ierr);
  
  ierr = VecDestroy(&xlocal); CHKERRQ(ierr);

  PetscFunctionReturn(0);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Local vectors
  ierr = DMGetLocalVector(dm,&xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm,x,INSERT_VALUES,xlocal); CHKERRQ(ierr);

  ierr = DMCreateLocalVector(dm, &dfxlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm, dfxlocal, &df1); CHKERRQ(ierr);

  ierr = DMCreateLocalVector(dm, &dfzlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm, dfzlocal, &df2); CHKERRQ(ierr);
  
  // Get domain corners
  ierr = DMStagGetGlobalSizes(dm, &Nx, &Nz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr); 
  
  // Get dm coordinates array
  ierr = DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter);CHKERRQ(ierr);

  // Get location slot
  ierr = DMStagGetLocationSlot(dm, ELEMENT, 0, &idx); CHKERRQ(ierr);

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

      ierr = DMStagVecGetValuesStencil(dm, xlocal, 4, point, fval); CHKERRQ(ierr);

      df1[j][i][idx] = (fval[1] - fval[0])/dx;
      df2[j][i][idx] = (fval[3] - fval[2])/dz;

    }
  }

  // Restore arrays
  ierr = DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);

  // Restore and map local to global
  ierr = DMStagVecRestoreArray(dm,dfxlocal,&df1); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm,dfxlocal,INSERT_VALUES,usr->dfx); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,dfxlocal,INSERT_VALUES,usr->dfx); CHKERRQ(ierr);
  ierr = VecDestroy(&dfxlocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArray(dm,dfzlocal,&df2); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm,dfzlocal,INSERT_VALUES,usr->dfz); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,dfzlocal,INSERT_VALUES,usr->dfz); CHKERRQ(ierr);
  ierr = VecDestroy(&dfzlocal); CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(dm, &xlocal ); CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
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
  PetscErrorCode ierr;
  
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
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  // Get global size
  ierr = DMStagGetGlobalSizes(dm, &Nx, &Nz,NULL);CHKERRQ(ierr);

  // Create local vector
  ierr = DMGetLocalVector(dm,&xplocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin (dm,xprev,INSERT_VALUES,xplocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd (dm,xprev,INSERT_VALUES,xplocal); CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm,&xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin (dm,x,INSERT_VALUES,xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd (dm,x,INSERT_VALUES,xlocal); CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm,&dfxlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin (dm,dfx,INSERT_VALUES,dfxlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd (dm,dfx,INSERT_VALUES,dfxlocal); CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm,&dfzlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin (dm,dfz,INSERT_VALUES,dfzlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd (dm,dfz,INSERT_VALUES,dfzlocal); CHKERRQ(ierr);

  // get array from xlocal
  ierr = DMStagVecGetArray(dm, xlocal, &xx); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm, xplocal, &xxp); CHKERRQ(ierr);

  // Get dm coordinates array
  ierr = DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter);CHKERRQ(ierr);

  // get the location slot
  ierr = DMStagGetLocationSlot(dm, ELEMENT, 0, &idx); CHKERRQ(ierr);
  
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

      

      ierr = DMStagVecGetValuesStencil(dm,dfxlocal,5,point,dfxe); CHKERRQ(ierr);
      ierr = DMStagVecGetValuesStencil(dm,dfzlocal,5,point,dfze); CHKERRQ(ierr);
      ierr = DMStagVecGetValuesStencil(dm,xplocal ,5,point,fe); CHKERRQ(ierr);

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
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
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
  ierr = DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArray(dm,xlocal,&xx);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm,xlocal,INSERT_VALUES,x); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,xlocal,INSERT_VALUES,x); CHKERRQ(ierr);
  ierr = VecDestroy(&xlocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArray(dm,xplocal,&xxp);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm,xplocal,INSERT_VALUES,xprev); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,xplocal,INSERT_VALUES,xprev); CHKERRQ(ierr);
  ierr = VecDestroy(&xplocal); CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(dm, &dfxlocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &dfzlocal); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
