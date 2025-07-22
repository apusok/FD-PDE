// ---------------------------------------
// Two-phase StokesDarcy model with an interface which is captured using the Phasefield method
// R = 0, infinitely small compaction length, so that it is Stokes flow.
// Rheology: visco-elasto-plastic model, inifinitely large C, G and Z.
// run: ./test_stokesdarcy2field_rt2_ -nx 100 -nz 100 -pc_type lu -pc_factor_mat_solver_type umfpack -log_view
// python test: ./python/test_stokesdarcy2field_rt.py
// ---------------------------------------
static char help[] = "Application for two phase StokesDarcy model with a planar interface \n\n";

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

#include "../src/fdpde_stokesdarcy2field.h"

// ---------------------------------------
// Application Context
// ---------------------------------------
#define FNAME_LENGTH 200

// parameters (bag)
typedef struct {
  PetscInt       nx, nz;
  PetscInt       tstep, tout, maxckpt;
  PetscInt       wn, vfopt;
  PetscScalar    Delta;
  PetscScalar    gamma, eps;
  PetscScalar    L, H;
  PetscScalar    xmin, zmin, xmax, zmax;
  PetscScalar    P0, etamin;
  PetscScalar    lam_v, eta_d, eta_u, zeta_d, zeta_u, C_u, C_d, Fu, Fd, z_in;
  PetscScalar    lambda,lam_p,G,Z,q,R,phi_0,n,nh;
  PetscBool      plasticity;
  PetscScalar    t, dt, dtck;
  char           fname_out[FNAME_LENGTH];
  char           fname_in[FNAME_LENGTH];
} Params;

// user defined and model-dependent variables
typedef struct {
  Params        *par;
  PetscBag       bag;
  MPI_Comm       comm;
  PetscMPIInt    rank;
  DM             dmf, dmPV; //dmf - phasefield DM, dmPV - velocity
  Vec            xVel, f, fprev, dfx, dfz; //phasefield
  Vec            volf; //volume fraction
  Vec            xeps, xtau, xyield, xDP, xtau_old, xDP_old;
} UsrData;

// ---------------------------------------
// Function definitions
// ---------------------------------------
PetscErrorCode StokesDarcy_Numerical(void*);
PetscErrorCode InputParameters(UsrData**);
PetscErrorCode InputPrintData(UsrData*);

PetscErrorCode FormCoefficient_Stokes(FDPDE, DM, Vec, DM, Vec, void*);
PetscErrorCode FormBCList_Stokes(DM, Vec, DMStagBCList, void*);

PetscErrorCode SetInitialField(DM,Vec,void*);
PetscErrorCode UpdateDF(DM,Vec,void*);
PetscErrorCode ExplicitStep(DM,Vec,Vec,PetscScalar,void*);
PetscErrorCode UpdateCornerF(DM, Vec, void*);
PetscErrorCode UpdateVolFrac(DM, Vec, void*);

const char coeff_description[] = 
"  << Stokes-Darcy Coefficients >> \n"
"  A , B, C \n";

const char bc_description[] =
"  << Stokes-Darcy BCs >> \n";


// static functions
static PetscScalar volf_2d(PetscScalar f1, PetscScalar f2, PetscScalar cc, PetscScalar ar)

{ PetscScalar tol, r10, r20, r1, r2, d1, d2, fchk, aa, result;

  tol = 1e-2;
  r10 = 2*f1 - 1.0;
  r20 = 2*f2 - 1.0;
  r1 = r10;
  r2 = r20;

  // first order fix for the distance, using the Newton raphson method.
  /*
    f1 = (0.5*(1.0 + r10 - 1.0/3.0*r10*r10*r10) - ff[0]);
    f2 = (0.5*(1.0 + r20 - 1.0/3.0*r20*r20*r20) - ff[1]);
    df1 = 0.5-0.5*r10*r10;
    df2 = 0.5-0.5*r20*r20;
    
    if ((PetscAbs(f1)>tol) && (PetscAbs(df1)>tol)) {
    r1 = r10 - f1/df1;
    }

    if ((PetscAbs(f2)>tol) && (PetscAbs(df2)>tol)) {
    r2 = r20 - f2/df2;   
    }
  */
  d1 = 2.0*cc*r1;
  d2 = 2.0*cc*r2;

  //1st check: is the interface too far away
  fchk = PetscAbs(d1+d2);
  aa   = sqrt(ar*ar + 1.0);
  if (fchk < aa ) {
    PetscScalar tx, tz, tol2;
    tx = (d1-d2);

    tol2 = 10*tol;

    if (PetscAbs(tx)>1.0 && PetscAbs(tx)-1.0 <= tol2) {tx = tx/PetscAbs(tx);}
        
    if (PetscAbs(tx)-1.0 > tol) {
      PetscPrintf(PETSC_COMM_WORLD, "tx = %1.4f/n", tx);}
    if (PetscAbs(tx)-1.0 > tol2) {
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL, "line direction error: |tx| > 1.0 !");}
        
    tz = sqrt(1.0 - tx*tx);

    if (PetscAbs(tz)<tol) {
      //2.1 check: a horizontal line?
      if (d1+d2>=0) {result = PetscMin(0.5+(d1+d2), 1.0);}
      else          {result = 1.0 - PetscMin(0.5-(d1+d2), 1.0);}
    } 
    else if (PetscAbs(tx)<tol) {
      //2.2 check: a vertical line?
      if (d1>=0) {result = PetscMin(0.5+d1/ar, 1.0);}
      else       {result = 1.0 - PetscMin(0.5-d1/ar, 1.0);}
    }
    else {
      //3 check: intersection with z = 0 and z = 1.0
      PetscScalar xb, xu, k0, k1, x0, x1;

      k0 = tz/tx;

      xb = 0.5*ar + d2/tz;
      xu = 0.5*ar + d1/tz;
      k1 = -k0*xb;
          
      if      (xu<=0.0) {x1 = 0.0;}
      else if (xu>=ar ) {x1 = ar; }
      else              {x1 = xu; }
          
      if      (xb<=0.0) {x0 = 0.0;}
      else if (xb>=ar ) {x0 = ar ;}
      else              {x0 = xb ;}
        
      result = (x1 - (0.5*k0*(x1*x1 - x0*x0) + k1*(x1-x0)))/ar;

      if (result <0 || result > 1.0) {
        PetscPrintf(PETSC_COMM_WORLD, "WRONG vvf, greater than 1 or smaller than zero, volf = %1.4f", result);}

    }
  }
  else if (f1>=0.5) {result = 1.0;} // line too far, only fluid 1
  else {result =0.0;}                 // line is too far away, only fluid 2

  return(result);
}

// ---------------------------------------
// Application functions
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "StokesDarcy_Numerical"
PetscErrorCode StokesDarcy_Numerical(void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  FDPDE          fd;
  DM             dm; // dmcoeff;
  DM             dmf;
  Vec            f, fprev, dfx, dfz, volf;
  Vec            x; // xcoeff;
  PetscInt       nx, nz, istep = 0, tstep, ickpt = 0, maxckpt;
  PetscScalar    xmin, zmin, xmax, zmax, dtck, tckpt;
  PetscBool      iwrt; //if write into files or not
  char           fout[FNAME_LENGTH];
  PetscFunctionBeginUser;

  // Element count
  nx = usr->par->nx;
  nz = usr->par->nz;

  // Domain coords
  xmin = usr->par->xmin;
  xmax = usr->par->xmin + usr->par->L;
  zmin = usr->par->zmin;
  zmax = usr->par->zmin + usr->par->H;

  tstep = usr->par->tstep;
  dtck  = usr->par->dtck;
  maxckpt = usr->par->maxckpt;
  tckpt = dtck; //first check point in time
  iwrt = PETSC_TRUE;

  // Create the FD-pde object
  PetscCall(FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_STOKES,&fd));
  PetscCall(FDPDESetUp(fd));
  PetscCall(FDPDEGetDM(fd,&dm)); 

  // Create DM/vec for the velocity field
  usr->dmPV = dm;

  // Create DM/vec for the phase field
  PetscCall(DMStagCreateCompatibleDMStag(dm,1,1,1,0,&usr->dmf)); 
  PetscCall(DMSetUp(usr->dmf)); 
  PetscCall(DMStagSetUniformCoordinatesProduct(usr->dmf,xmin,xmax,zmin,zmax,0.0,0.0));
  PetscCall(DMCreateGlobalVector(usr->dmf, &usr->f)); 
  PetscCall(VecDuplicate(usr->f, &usr->fprev)); 
  PetscCall(VecDuplicate(usr->f,&usr->dfx));
  PetscCall(VecDuplicate(usr->f,&usr->dfz));
  PetscCall(VecDuplicate(usr->f,&usr->volf));

  // short names for DM and Vecs of the phase field
  dmf   = usr->dmf;
  f     = usr->f;
  fprev = usr->fprev;
  dfx   = usr->dfx;
  dfz   = usr->dfz;
  volf  = usr->volf;

  // Create a vector to store u, p in userdata
  PetscCall(FDPDEGetSolution(fd,&x));
  PetscCall(VecDuplicate(x, &usr->xVel));
  PetscCall(VecDestroy(&x));

  // Set coefficients and BC evaluation function
  PetscCall(FDPDESetFunctionBCList(fd,FormBCList_Stokes,bc_description,usr)); 
  PetscCall(FDPDESetFunctionCoefficient(fd,FormCoefficient_Stokes,coeff_description,usr)); 
  PetscCall(FDPDEView(fd)); 

  // Initialise the phase field
  PetscCall(SetInitialField(dmf,f,usr));
  PetscCall(VecCopy(f, fprev));
  //interpolate phase values on the face and edges before FDPDE solver
    PetscCall(UpdateCornerF(dmf, f, usr)); 
    PetscCall(UpdateVolFrac(dmf, f, usr)); 

  // output - initial state of the phase field
  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s_phase_initial",usr->par->fname_out));
  PetscCall(DMStagViewBinaryPython(dmf,f,fout));
  
  // FD SNES Solver
  PetscPrintf(PETSC_COMM_WORLD,"\n# SNES SOLVE #\n");
  
  // Time loop  
  //while (istep<tstep) {
  //while (ickpt < maxckpt) {
  while ((usr->par->t <= dtck*maxckpt) && (istep<tstep))  {
    //PetscPrintf(PETSC_COMM_WORLD,"# TIMESTEP %d out of %d: time %1.4f\n\n",istep, tstep, usr->par->t);
    PetscPrintf(PETSC_COMM_WORLD,"# TIME CHECK POINT %d out of %d after %d steps: time %1.4f\n\n",ickpt, maxckpt, istep, usr->par->t);
    PetscPrintf(PETSC_COMM_WORLD,"# next check piont: %1.4f; distance between check points: %1.4f\n\n", tckpt, dtck);

    if (istep>0) {
      //one step forward to get f at the next step (extract velocity data within it)
      PetscCall(UpdateDF(dmf, fprev, usr)); 
      PetscCall(ExplicitStep(dmf, fprev, f, usr->par->dt, usr)); 
    }

    //interpolate phase values on the face and edges before FDPDE solver
    PetscCall(UpdateCornerF(dmf, f, usr)); 
    PetscCall(UpdateVolFrac(dmf, f, usr)); 
    
    //StokesDarcy Solver
    PetscCall(FDPDESolve(fd,NULL));
    PetscCall(FDPDEGetSolution(fd,&x));

    //update x into the usrdata
    PetscCall(VecCopy(x, usr->xVel));

    // Update time
    usr->par->t += usr->par->dt;  // computation start from t = dt

    // 2nd order runge-kutta
#if 1
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
      usr->par->t -= 0.5*usr->par->dt;
      
      // update dfx and dfz and solve for the second stage
      PetscCall(UpdateDF(dmf, fprev, usr)); 
      PetscCall(ExplicitStep(dmf, fprev, f, usr->par->dt, usr));

      // get hk2 and update the full step
      PetscCall(VecCopy(f, hk2)); 
      PetscCall(VecAXPY(hk2, -1.0, fprev));
      PetscCall(VecCopy(fprev_bk, fprev)); 
      PetscCall(VecCopy(fprev, f)); 
      PetscCall(VecAXPY(f, 1.0, hk2));

      // reset time
      usr->par->t += 0.5*usr->par->dt;
      
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
#endif    
    
    PetscPrintf(PETSC_COMM_WORLD,"# TIME: time = %1.12e dt = %1.12e \n",usr->par->t,usr->par->dt);

    // Phasefield: save f and fprev into global vectors
    usr->f     = f;
    usr->fprev = fprev;
    // Phasefield: copy f to prev
    PetscCall(VecCopy(f,fprev)); 

    // write before changing time steps
    if (iwrt || istep == 0) {

      iwrt = PETSC_FALSE;
      
      // Output solution to file
      PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s_solution_ts%1.3d",usr->par->fname_out,ickpt));
      PetscCall(DMStagViewBinaryPython(dm,x,fout));

      PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s_phase_ts%1.3d",usr->par->fname_out,ickpt));
      PetscCall(DMStagViewBinaryPython(usr->dmf,usr->f,fout));

    }


    //check max(f) and min(f),
    PetscScalar fmax, fmin;
    PetscCall(VecMax(f,NULL,&fmax)); 
    PetscCall(VecMin(f,NULL,&fmin)); 
    PetscPrintf(PETSC_COMM_WORLD, "Phase field: Maximum of f = %1.8f, Minimum of f = %1.8f\n", fmax, fmin); 


    //check max(x) and min(x) - both face and center values are compared though, but pressure might be small if gravity <= 1 and eta << 1.
    PetscScalar xxmax, xxmin, dtt, dtgap;
    PetscCall(VecMax(x,NULL,&xxmax)); 
    PetscCall(VecMin(x,NULL,&xxmin)); 
    usr->par->gamma = PetscMax(PetscAbs(xxmax), PetscAbs(xxmin));
    if (usr->par->gamma < 1e-5) {usr->par->gamma = 1.0;}
    //change dt accordingly
    dtt = 1.0/nx/usr->par->gamma/6.0; //maximum time step allowed for boundedness
    dtgap = tckpt - usr->par->t;   // gap between the current time and the next checkpoint

    if (dtgap <= 0) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL, "dtgap is smaller or equal to zero, the next check points, tckpt, has not been updated properly");
    
    usr->par->dt = PetscMin(dtt, dtgap);

    PetscPrintf(PETSC_COMM_WORLD, "Phase field: Maximum of U = %1.8f, Minimum of U = %1.8f\n", xxmax, xxmin); 
    PetscPrintf(PETSC_COMM_WORLD, "Phase field: gamma = %1.8f\n", usr->par->gamma); 

    //check if reaching the check point
    if (dtgap <= dtt) {ickpt++; tckpt += dtck; iwrt = PETSC_TRUE;}

    //clean up
    PetscCall(VecDestroy(&x));

    // increment timestep
    istep++;

  }

  // Destroy objects
  PetscCall(VecDestroy(&fprev));
  PetscCall(VecDestroy(&f));
  PetscCall(VecDestroy(&dfx));
  PetscCall(VecDestroy(&dfz));
  PetscCall(VecDestroy(&volf)); 
  PetscCall(VecDestroy(&usr->xVel));

  PetscCall(DMDestroy(&usr->dmPV));
  PetscCall(DMDestroy(&dmf)); 
  PetscCall(FDPDEDestroy(&fd));

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
  PetscCall(PetscBagRegisterInt(bag, &par->nx, 5, "nx", "Element count in the x-dir")); 
  PetscCall(PetscBagRegisterInt(bag, &par->nz, 5, "nz", "Element count in the z-dir")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->xmin, 0.0, "xmin", "Start coordinate of domain in x-dir")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->zmin, 0.0, "zmin", "Start coordinate of domain in z-dir")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->L, 1.0, "L", "Length of domain in x-dir")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->H, 1.0, "H", "Height of domain in z-dir")); 

  // Physical and material parameters
  PetscCall(PetscBagRegisterScalar(bag, &par->Fd, 0.0, "Fd", "Non-dimensional gravity of the bottom layer")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->Fu, 0.0, "Fu", "Non-dimensional gravity of the up layer")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->R, 0.0, "R", "Ratio of the compaction length scale to the global one, R = ((K0*eta_ref/mu)^1/2)/L")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->phi_0, 0.01, "phi_0", "Reference porosity")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->n, 2.0, "n", "Porosity exponent")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->lambda, 0.0, "lambda", "Exponential melt weakening factor")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->z_in, 0.5, "z_in", "Position of the sharp interface")); 
  // Viscosity
  PetscCall(PetscBagRegisterScalar(bag, &par->eta_u, 1.0, "eta_u", "Viscosity of the upper layer")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->eta_d, 1.0, "eta_d", "Viscosity of the bottom layer")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->lam_v, 1.0e-1, "lam_v", "Factors for intrinsic visocisty, lam_v = eta/zeta")); 
  // Plasticity
  PetscCall(PetscBagRegisterScalar(bag, &par->C_u, 1e40, "C_u", "Cohesion (up)")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->C_d, 1e40, "C_d", "Cohesion (bottom)")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->lam_p, 1.0, "lam_p", "Multiplier for the compaction failure criteria, YC = lam_p*C")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->etamin, 0.0, "etamin", "Cutoff min value of eta")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->nh, 1.0, "nh", "Power for the harmonic plasticity")); 
  // Elasticity
  PetscCall(PetscBagRegisterScalar(bag, &par->G, 1e40, "G", "Shear elastic modulus")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->Z, 1e40, "Z", "Reference poro-elastic modulus")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->q, -0.5, "q", "Exponent of the porosity-dependent relation of poro-elastic modulus")); 

  // Time steps
  PetscCall(PetscBagRegisterScalar(bag, &par->dt, 0.01, "dt", "The size of time step")); 
  PetscCall(PetscBagRegisterInt(bag, &par->tstep, 11, "tstep", "The maximum time steps")); 
  PetscCall(PetscBagRegisterInt(bag, &par->tout,5, "tout", "Output every tout time step")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->dtck, 0.1, "dtck", "The size between two check points in time")); 
  PetscCall(PetscBagRegisterInt(bag, &par->maxckpt, 1, "maxckpt", "Maximum number of check points")); 

  PetscCall(PetscBagRegisterScalar(bag, &par->P0,0, "P0", "Pinned value for pressure")); 

  // Parameters for the phase field method
  PetscCall(PetscBagRegisterScalar(bag, &par->eps, 0.2, "eps", "epsilon in the kernel function")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->gamma, 1.0, "gamma", "gamma in the phase field method")); 
  //PetscCall(PetscBagRegisterBool(bag, &par->diffuse, PETSC_FALSE, "diffuse", "parameters varies smoothly in the diffusing interface")); 
  PetscCall(PetscBagRegisterInt(bag, &par->vfopt, 0, "vfopt", "vfopt = 0,1,2,3")); 

  // Parameters for the purbation at the interface
  PetscCall(PetscBagRegisterScalar(bag, &par->Delta, 0.1, "Delta", "amplitude of the perturbation")); 
  PetscCall(PetscBagRegisterInt(bag, &par->wn, 2, "wn", "wavenumber of the perturbation")); 

  // Reference compaction viscosity
  par->zeta_u = par->eta_u/par->lam_v;
  par->zeta_d = par->eta_d/par->lam_v;
  
  par->plasticity = PETSC_FALSE;//PETSC_TRUE;

  // other variables
  par->t = 0.0;

  // Input/output 
  PetscCall(PetscBagRegisterString(bag,&par->fname_out,FNAME_LENGTH,"out_solution","output_file","Name for output file, set with: -output_file <filename>")); 

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
  PetscPrintf(usr->comm,"# Test_stokesdarcy2field_phasefield: %s \n",&(date[0]));
  PetscPrintf(usr->comm,"# --------------------------------------- #\n");
  PetscPrintf(usr->comm,"# PETSc options: %s \n",opts);
  PetscPrintf(usr->comm,"# --------------------------------------- #\n");

  // Print usr bag
  PetscCall(PetscBagView(usr->bag,PETSC_VIEWER_STDOUT_WORLD)); 
  PetscPrintf(usr->comm,"# --------------------------------------- #\n");

  // Free memory
  PetscCall(PetscFree(opts)); 

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// FormCoefficient_Stokes
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormCoefficient_Stokes"
PetscErrorCode FormCoefficient_Stokes(FDPDE fd, DM dm, Vec x, DM dmcoeff, Vec coeff, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz;
  Vec            coefflocal;
  Vec            flocal, volflocal;
  PetscScalar    **coordx,**coordz;
  PetscInt       iprev, inext, icenter;
  // PetscScalar    lambda,R,n;
  PetscScalar    ***c;
  PetscScalar    phi; //dt;
  PetscScalar    eta_u, eta_d, F_u, F_d;//, z_in;
  // PetscScalar    eps, ccc;
  PetscFunctionBeginUser;

  // dt = usr->par->dt;

  // lambda = usr->par->lambda;
  // R = usr->par->R;
  // n = usr->par->n;
  // eps = usr->par->eps;

  // constant shear and compaction viscosity of up and down, force(density) up and down
  eta_u  = usr->par->eta_u;
  eta_d  = usr->par->eta_d;
  F_u     = usr->par->Fu;
  F_d     = usr->par->Fd;

  // (sharp) interface position
  // z_in = usr->par->z_in;

  // Uniform porosity and permeability
  phi = usr->par->phi_0;

  // phase field
  PetscCall(DMGetLocalVector(usr->dmf, &flocal)); 
  PetscCall(DMGlobalToLocal (usr->dmf, usr->f, INSERT_VALUES, flocal)); 

  // volume fraction
  PetscCall(DMGetLocalVector(usr->dmf, &volflocal)); 
  PetscCall(DMGlobalToLocal (usr->dmf, usr->volf, INSERT_VALUES, volflocal)); 

  // Get domain corners
  PetscCall(DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 
  PetscCall(DMStagGetGlobalSizes(dmcoeff,&Nx,&Nz,NULL));

  // Get dm coordinates array
  PetscCall(DMStagGetProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmcoeff,LEFT,&iprev));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmcoeff,RIGHT,&inext)); 
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmcoeff,ELEMENT,&icenter));

  // Create coefficient local vector
  PetscCall(DMCreateLocalVector(dmcoeff, &coefflocal)); 
  PetscCall(DMStagVecGetArray(dmcoeff, coefflocal, &c)); 

  // Get the cell sizes
  PetscScalar *dx, *dz;
  PetscCall(DMStagCellSizeLocal_2d(dm, &nx, &nz, &dx, &dz)); 

  // compute ccc = eps/Delta z
  // ccc = eps*Nz;
  
  // Loop over local domain - set initial density and viscosity
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      PetscInt idx;

      { // A = eta (center, c=1), and also compute chi_s and chi_p (center, c = 4,5)
        DMStagStencil point;
        PetscScalar   eta, volf, ff;

        // get the phase values in the element
        point.i = i; point.j = j; point.loc = ELEMENT; point.c = 0;
        PetscCall(DMStagVecGetValuesStencil(usr->dmf,flocal,1,&point,&ff)); 
        PetscCall(DMStagVecGetValuesStencil(usr->dmf,volflocal,1,&point,&volf)); 

        eta  = eta_u  * volf + eta_d  * (1.0 - volf);
        point.i = i; point.j = j; point.loc = ELEMENT; point.c = 1;

        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
        c[j][i][idx] = eta;
      }

      { // A = eta (corner, c=0)
        DMStagStencil point[4];
        PetscScalar   ff[4], volf[4];
        PetscScalar   eta;
        PetscInt      ii;

        // second invariant of strain rate
        point[0].i = i; point[0].j = j; point[0].loc = DOWN_LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = DOWN_RIGHT; point[1].c = 0;
        point[2].i = i; point[2].j = j; point[2].loc = UP_LEFT;    point[2].c = 0;
        point[3].i = i; point[3].j = j; point[3].loc = UP_RIGHT;   point[3].c = 0;

        // collect phase values for the four corners
        PetscCall(DMStagVecGetValuesStencil(usr->dmf,flocal,4,point,ff)); 
        PetscCall(DMStagVecGetValuesStencil(usr->dmf,volflocal,4,point,volf)); 
        
        for (ii = 0; ii < 4; ii++) {
          eta  = eta_u  * volf[ii] + eta_d  * (1.0 - volf[ii]);
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, 0, &idx)); 
          c[j][i][idx] = eta;
        }
      }

      { // B = phi*F*ek - grad(chi_s*tau_old) + div(chi_p*dP_old) (edges, c=0)
        // (F = (rho^s-rho^f)*U*L/eta_ref * g*L/U^2) 
        DMStagStencil point[4];
        PetscScalar   rhs[4], ff[4], volf[4], F2, F3;
        PetscInt      ii;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 0;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = 0;
        point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = 0;

        // collect phase values for the four edges
        PetscCall(DMStagVecGetValuesStencil(usr->dmf,flocal,4,point,ff)); 
        PetscCall(DMStagVecGetValuesStencil(usr->dmf,volflocal,4,point,volf)); 

        F2 = F_u*volf[2] + F_d*(1.0-volf[2]);
        F3 = F_u*volf[3] + F_d*(1.0-volf[3]);

        rhs[0] = 0.0;
        rhs[1] = 0.0;
        rhs[2] = phi*F2;
        rhs[3] = phi*F3;

        for (ii = 0; ii < 4; ii++) {
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] += rhs[ii];
        }
      }

      { // C = 0 (center, c=0)
        DMStagStencil point;
        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 0;
        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
        c[j][i][idx] = 0.0;
      }
    }
  }

  // release dx dz
  PetscCall(PetscFree(dx));
  PetscCall(PetscFree(dz));  

  // Restore arrays, local vectors
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL));

  PetscCall(DMStagVecRestoreArray(dmcoeff,coefflocal,&c));
  PetscCall(DMLocalToGlobalBegin(dmcoeff,coefflocal,INSERT_VALUES,coeff)); 
  PetscCall(DMLocalToGlobalEnd  (dmcoeff,coefflocal,INSERT_VALUES,coeff)); 
  PetscCall(VecDestroy(&coefflocal)); 

  PetscCall(DMRestoreLocalVector(usr->dmf,&flocal));   
  PetscCall(DMRestoreLocalVector(usr->dmf,&volflocal));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// FormBCList_Stokes
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormBCList_Stokes"
PetscErrorCode FormBCList_Stokes(DM dm, Vec x, DMStagBCList bclist, void *ctx)
{
  // UsrData        *usr = (UsrData*)ctx;
  PetscInt       k,n_bc,*idx_bc;
  PetscScalar    *value_bc,*x_bc;
  BCType         *type_bc;
  PetscFunctionBeginUser;

  // LEFT Boundary - Vx
  PetscCall(DMStagBCListGetValues(bclist,'w','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET; //BC_NEUMANN;//BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // LEFT Boundary - Vz
  PetscCall(DMStagBCListGetValues(bclist,'w','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  
  // RIGHT Boundary - Vx
  PetscCall(DMStagBCListGetValues(bclist,'e','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;//BC_NEUMANN; //BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // RIGHT Boundary - Vz
  PetscCall(DMStagBCListGetValues(bclist,'e','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
    
  // DOWN Boundary - Vx
  PetscCall(DMStagBCListGetValues(bclist,'s','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;//BC_NEUMANN;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // DOWN Boundary - Vz
  PetscCall(DMStagBCListGetValues(bclist,'s','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;//BC_NEUMANN;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  
  // UP Boundary - Vx
  PetscCall(DMStagBCListGetValues(bclist,'n','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;//BC_NEUMANN;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // UP Boundary - Vz
  PetscCall(DMStagBCListGetValues(bclist,'n','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;//BC_NEUMANN;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  PetscCall(DMStagBCListGetValues(bclist,'n','o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<1; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET_STAG;//BC_NEUMANN;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
 
  
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
  PetscInt       i,j, sx, sz, nx, nz, icenter;//, wn;
  PetscScalar    eps, L, Delta;
  PetscScalar    ***xx, **coordx, **coordz;
  PetscFunctionBeginUser;

  // some useful parameters
  eps = usr->par->eps;
  L   = usr->par->L;
  Delta = usr->par->Delta;
  // wn  = usr->par->wn;

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
      PetscScalar   xp,zp, fval = 0.0, xn, zw, zpb;
      PetscInt      idx;

      zw = usr->par->z_in;

      point.i = i; point.j = j; point.loc = ELEMENT; point.c = 0;
      xp = coordx[i][icenter];
      zp = coordz[j][icenter];
      
      //xn = zp - zw;
      //fval = 0.5*(1 + PetscTanhScalar(xn/2.0/eps));

      //zpb = zw + 0.1*PetscSinScalar(-0.5*M_PI + xp*(2.0*M_PI*2));
      //zpb = zw + 0.05*PetscCosScalar((xp/L)*M_PI);
      //zpb = zw + Delta*PetscCosScalar(2.0*M_PI*wn*xp);
      //zpb = zw + Delta*PetscCosScalar(4.0*M_PI*xp/L);
      zpb = zw + Delta*PetscCosScalar(M_PI*xp/L);
      
      //      if (zp <= zpb) {fval = 0.0;}
      //else           {fval = 1.0;}

      xn = zp - zpb;

      //xn = xn * sqrt(1- PetscPowScalar(Delta*4.0*M_PI/L * PetscSinScalar(4.0*M_PI*xp/L),2));
      
      fval = 0.5*(1 + PetscTanhScalar(xn/2.0/eps));
      
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
PetscErrorCode UpdateDF(DM dm, Vec x, void *ctx)
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
  PetscScalar    gamma, eps;
  PetscScalar    **coordx,**coordz;
  PetscScalar    ***xx,***xxp;
  Vec            dfx, dfz, dfxlocal, dfzlocal, xlocal, xplocal;
  Vec            xVellocal;
  PetscFunctionBeginUser;

  // User Parameter
  gamma = usr->par->gamma;
  eps = usr->par->eps;

  dfx = usr->dfx;
  dfz = usr->dfz;

  // create a dmPV and xPV in usrdata, copy data in and extract them here
  PetscCall(DMGetLocalVector(usr->dmPV, &xVellocal)); 
  PetscCall(DMGlobalToLocal (usr->dmPV, usr->xVel, INSERT_VALUES, xVellocal)); 

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
  //if (sz==0) {sz++; nz--;}
  //if (sz+nz==Nz) {nz--;}
  //if (sx==0) {sx++; nx--;}
  //if (sx+nx==Nx) {nx--;}

  // Get the cell sizes
  PetscScalar *dx, *dz;
  PetscCall(DMStagCellSizeLocal_2d(dm, &nx, &nz, &dx, &dz)); 

  // loop over local domain and get the RHS value
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i < sx+nx; i++) {

      DMStagStencil point[5];
      PetscInt      ii,ix,iz;
      PetscScalar   fe[5], dfxe[5], dfze[5], gfe[5], c[5], fval = 0.0;

      ix = i - sx;
      iz = j - sz;

      point[0].i = i;   point[0].j = j;   point[0].loc = ELEMENT; point[0].c = 0;
      point[1].i = i-1; point[1].j = j;   point[1].loc = ELEMENT; point[1].c = 0;
      point[2].i = i+1; point[2].j = j;   point[2].loc = ELEMENT; point[2].c = 0;
      point[3].i = i;   point[3].j = j-1; point[3].loc = ELEMENT; point[3].c = 0;
      point[4].i = i;   point[4].j = j+1; point[4].loc = ELEMENT; point[4].c = 0;

      // default zero flux on boundary
      if (i==0)    {point[1] = point[0];}
      if (i==Nx-1) {point[2] = point[0];}
      if (j==0)    {point[3] = point[0];}
      if (j==Nz-1) {point[4] = point[0];}
      
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
      fval = gamma*(eps * ((fe[2]+fe[1]-2*fe[0])/dx[ix]/dx[ix] + (fe[4]+fe[3]-2*fe[0])/dz[iz]/dz[iz]));

      //sharpen terms
      fval -= gamma* ( (c[2]*dfxe[2] - c[1]*dfxe[1])/(2.0*dx[ix]) + (c[4]*dfze[4]-c[3]*dfze[3])/(2.0*dz[iz]));

      //      if (i!=0 && i!=Nx-1 && j!=0 && j!=Nz-1) {
        //sharpen terms
      //  fval -= gamma* ( (c[2]*dfxe[2] - c[1]*dfxe[1])/(2.0*dx[ix]) + (c[4]*dfze[4]-c[3]*dfze[3])/(2.0*dz[iz]));
      //}


      { // velocity on the face and advection terms
        DMStagStencil pf[4];
        PetscScalar vf[4];

        pf[0].i = i; pf[0].j = j; pf[0].loc = LEFT;  pf[0].c = 0; 
        pf[1].i = i; pf[1].j = j; pf[1].loc = RIGHT; pf[1].c = 0;
        pf[2].i = i; pf[2].j = j; pf[2].loc = DOWN;  pf[2].c = 0;
        pf[3].i = i; pf[3].j = j; pf[3].loc = UP;    pf[3].c = 0;

        PetscCall(DMStagVecGetValuesStencil(usr->dmPV,xVellocal,4,pf,vf)); 

        //        PetscPrintf(PETSC_COMM_WORLD, "vf check: %g, %g, %g, %g\n", vf[0], vf[1], vf[2], vf[3]);
        
        // central difference method
        fval -= 0.5*(vf[1]*(fe[2]+fe[0]) - vf[0]*(fe[1]+fe[0]))/dx[ix] + 0.5*(vf[3]*(fe[4]+fe[0]) - vf[2]*(fe[3]+fe[0]))/dz[iz];
        
      }

      xx[j][i][idx] = xxp[j][i][idx] + dt*fval;
    }
  }
                                
  // reset sx, sz, nx, nz
  PetscCall(DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 
  // apply boundary conditions : zero flux
  if (sx==0) {
    for (j = sz; j<sz+nz; j++) {
      //   xx[j][0][idx] = xx[j][1][idx]; 
    }
  }

  if (sx+nx==Nx) {
    for (j = sz; j<sz+nz; j++) {
      // xx[j][Nx-1][idx] = xx[j][Nx-2][idx];
    }
  }

  if (sz==0) {
    for (i = sx; i<sx+nx; i++) {
      //  xx[0][i][idx] = xx[1][i][idx];
    }
  }

  if (sz+nz==Nz) {
    for (i = sx; i<sx+nx; i++) {
      //  xx[Nz-1][i][idx] = xx[Nz-2][i][idx];
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
  PetscCall(DMRestoreLocalVector(usr->dmPV, &xVellocal)); 

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// Interpolate corner and face values of f, for uniform grids only
// ---------------------------------------
PetscErrorCode UpdateCornerF(DM dm, Vec x, void *ctx)
{
  // UsrData       *usr = (UsrData*) ctx;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz;
  PetscInt       ic, il, ir, iu, id, idl, idr, iul, iur;
  PetscScalar    ***xx;
  Vec            xlocal;
  PetscFunctionBeginUser;

  // Local vectors
  PetscCall(DMGetLocalVector(dm,&xlocal)); 
  PetscCall(DMGlobalToLocal (dm,x,INSERT_VALUES,xlocal)); 
  PetscCall(DMStagVecGetArray(dm,xlocal,&xx)); 
  
  // Get domain corners
  PetscCall(DMStagGetGlobalSizes(dm, &Nx, &Nz,NULL));
  PetscCall(DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL));  
  
  // Get location slot
  PetscCall(DMStagGetLocationSlot(dm, ELEMENT,    0, &ic )); 
  PetscCall(DMStagGetLocationSlot(dm, DOWN_LEFT,  0, &idl)); 
  PetscCall(DMStagGetLocationSlot(dm, DOWN_RIGHT, 0, &idr)); 
  PetscCall(DMStagGetLocationSlot(dm, LEFT,       0, &il )); 
  PetscCall(DMStagGetLocationSlot(dm, RIGHT,      0, &ir )); 
  PetscCall(DMStagGetLocationSlot(dm, DOWN,       0, &id )); 
  PetscCall(DMStagGetLocationSlot(dm, UP,         0, &iu )); 
  PetscCall(DMStagGetLocationSlot(dm, UP_LEFT,    0, &iul)); 
  PetscCall(DMStagGetLocationSlot(dm, UP_RIGHT,   0, &iur)); 
  

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      DMStagStencil  point[4];
      PetscScalar    fval[4];

      // collect the elements points around the down left corner
      point[0].i = i-1; point[0].j = j  ; point[0].loc = ELEMENT; point[0].c = 0;
      point[1].i = i  ; point[1].j = j-1; point[1].loc = ELEMENT; point[1].c = 0;
      point[2].i = i-1; point[2].j = j-1; point[2].loc = ELEMENT; point[2].c = 0;
      point[3].i = i  ; point[3].j = j;   point[3].loc = ELEMENT; point[3].c = 0;

      // fix the boundary cell
      if (i == 0)    {point[0].i = i; point[2].i = i;}
      if (j == 0)    {point[2].j = j; point[1].j = j;}

      PetscCall(DMStagVecGetValuesStencil(dm, xlocal, 4, point, fval)); 

      xx[j][i][il]  = 0.5*(fval[3] + fval[0]); // left
      xx[j][i][id]  = 0.5*(fval[3] + fval[1]); // down
      xx[j][i][idl] = 0.25*(fval[0]+fval[1]+fval[2]+fval[3]); // downleft

      if (j==Nz-1) {
        xx[j][i][iu]  = xx[j][i][ic];
        xx[j][i][iul] = xx[j][i][il];
      }
      if (i==Nx-1) {
        xx[j][i][ir]  = xx[j][i][ic];
        xx[j][i][idr]  = xx[j][i][id];
      }
      if (i==Nx-1 && j==Nz-1) {
        xx[j][i][iur] = xx[j][i][ic];
      }
    }
  }

  // Restore and map local to global
  PetscCall(DMStagVecRestoreArray(dm,xlocal,&xx)); 
  PetscCall(DMLocalToGlobalBegin(dm,xlocal,INSERT_VALUES,x)); 
  PetscCall(DMLocalToGlobalEnd  (dm,xlocal,INSERT_VALUES,x)); 
  PetscCall(DMRestoreLocalVector(dm,&xlocal)); 
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// Update volumefraction for fluid 1 within each cube between two vertically adjacent cell center
// ---------------------------------------
PetscErrorCode UpdateVolFrac(DM dm, Vec x, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz, vfopt;
  PetscInt       ic, il, ir, iu, id, idl, iul, idr, iur, icenter, iprev, inext;
  PetscScalar    ***vvf, **coordx, **coordz;
  PetscScalar    eps;
  Vec            xlocal, vflocal;
  PetscFunctionBeginUser;

  eps = usr->par->eps;
  vfopt = usr->par->vfopt;

  // Local vectors
  PetscCall(DMGetLocalVector(dm,&xlocal)); 
  PetscCall(DMGlobalToLocal (dm,x,INSERT_VALUES,xlocal)); 

  PetscCall(DMCreateLocalVector(dm, &vflocal)); 
  PetscCall(DMStagVecGetArray(dm, vflocal, &vvf)); 

  // Get domain corners
  PetscCall(DMStagGetGlobalSizes(dm, &Nx, &Nz,NULL));
  PetscCall(DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL));  
  
  // Get location slot
  PetscCall(DMStagGetLocationSlot(dm, ELEMENT,    0, &ic )); 
  PetscCall(DMStagGetLocationSlot(dm, LEFT,       0, &il )); 
  PetscCall(DMStagGetLocationSlot(dm, RIGHT,      0, &ir )); 
  PetscCall(DMStagGetLocationSlot(dm, DOWN,       0, &id )); 
  PetscCall(DMStagGetLocationSlot(dm, UP,         0, &iu )); 
  PetscCall(DMStagGetLocationSlot(dm, DOWN_LEFT,  0, &idl )); 
  PetscCall(DMStagGetLocationSlot(dm, UP_LEFT,    0, &iul )); 
  PetscCall(DMStagGetLocationSlot(dm, DOWN_RIGHT, 0, &idr )); 
  PetscCall(DMStagGetLocationSlot(dm, UP_RIGHT,   0, &iur )); 

  // Get dm coordinates array
  PetscCall(DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm,LEFT   ,&iprev  ));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm,RIGHT  ,&inext  ));
  

  //if (sz==0) {sz++;nz--;}
  
  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      DMStagStencil  point[8];
      PetscScalar    ff[8], dx, dz, cc, ar;

      // collect the elements points around the down left corner
      point[0].i = i; point[0].j = j  ; point[0].loc = ELEMENT; point[0].c = 0;
      point[1].i = i; point[1].j = j-1; point[1].loc = ELEMENT; point[1].c = 0;
      point[2].i = i; point[2].j = j  ; point[2].loc = LEFT   ; point[2].c = 0;
      point[3].i = i; point[3].j = j-1; point[3].loc = LEFT   ; point[3].c = 0;

      point[4] = point[0]; point[4].loc = DOWN;
      point[5] = point[0]; point[5].loc = DOWN_LEFT;
      point[6] = point[0]; point[6].loc = UP;
      point[7] = point[0]; point[7].loc = UP_LEFT;

      if (j==0) {point[1].j = point[0].j; point[3].j = point[0].j;}

      PetscCall(DMStagVecGetValuesStencil(dm, xlocal, 8, point, ff)); 

      dz = coordz[j][inext] -  coordz[j  ][iprev];
      dx = coordx[i][inext  ] -  coordx[i  ][iprev  ];

      cc = eps/dz;
      ar = dx/dz;

      //      d1 = 4.0*cc*(ff[0]-0.5);
      //d2 = 4.0*cc*(ff[1]-0.5);

      //diffuse
      if (vfopt == 0) {
        vvf[j][i][id] = ff[4];
        vvf[j][i][ic] = ff[0];
        vvf[j][i][il] = ff[2];
        vvf[j][i][idl]= ff[5];
      }
      
      //sharp: staggered
      if (vfopt ==1) {
        if (ff[4]>= 0.5) {vvf[j][i][id] = 1.0;}
        else             {vvf[j][i][id] = 0.0;}
        if (ff[0]>= 0.5) {vvf[j][i][ic] = 1.0;}
        else             {vvf[j][i][ic] = 0.0;}
        if (ff[2]>= 0.5) {vvf[j][i][il] = 1.0;}
        else             {vvf[j][i][il] = 0.0;}
        if (ff[5]>= 0.5) {vvf[j][i][idl]= 1.0;}
        else             {vvf[j][i][idl]= 0.0;}
      }
      

      //1d simplification
      if (vfopt ==2 ) {
        PetscScalar fftmp;
        fftmp = ff[4];
        if      (fftmp >= 0.5+0.125/cc) {vvf[j][i][id] = 1.0;}
        else if (fftmp <= 0.5-0.125/cc) {vvf[j][i][id] = 0.0;}
        else    {vvf[j][i][id] = 0.5 - 4.0*cc*(fftmp-0.5);}

        fftmp = ff[0];
        if      (fftmp >= 0.5+0.125/cc) {vvf[j][i][ic] = 1.0;}
        else if (fftmp <= 0.5-0.125/cc) {vvf[j][i][ic] = 0.0;}
        else    {vvf[j][i][ic] = 0.5 - 4.0*cc*(fftmp-0.5);}

        fftmp = ff[2];
        if      (fftmp >= 0.5+0.125/cc) {vvf[j][i][il] = 1.0;}
        else if (fftmp <= 0.5-0.125/cc) {vvf[j][i][il] = 0.0;}
        else    {vvf[j][i][il] = 0.5 - 4.0*cc*(fftmp-0.5);}

        fftmp = ff[5];
        if      (fftmp >= 0.5+0.125/cc) {vvf[j][i][idl] = 1.0;}
        else if (fftmp <= 0.5-0.125/cc) {vvf[j][i][idl] = 0.0;}
        else    {vvf[j][i][idl] = 0.5 - 4.0*cc*(fftmp-0.5);}
      }

      //2d
      if (vfopt ==3) {
        vvf[j][i][ic] = volf_2d(ff[6], ff[4], cc, ar);
        vvf[j][i][il] = volf_2d(ff[7], ff[5], cc, ar);
        if (j>0) {
          vvf[j][i][id] = volf_2d(ff[0], ff[1], cc, ar);
          vvf[j][i][idl]= volf_2d(ff[2], ff[3], cc, ar);
        } else {
          vvf[j][i][id] = vvf[j][i][ic];
          vvf[j][i][id] = vvf[j][i][il];
        }
      }
    }
  }

  // for nodes on the up and right boundaries
  if (sz+nz == Nz) {
    j = Nz-1;
    for (i = sx; i<sx+nx; i++) {
      vvf[j][i][iul] = vvf[j][i][il];
      vvf[j][i][iu]  = vvf[j][i][ic];
    }
  }
  if (sx+nx == Nx) {
    i = Nx-1;
    for (j = sz; j<sz+nz; j++) {
      vvf[j][i][idr] = vvf[j][i][id];
      vvf[j][i][ir]  = vvf[j][i][ic];
    }
  }
  if (sx+nx==Nx && sz+nz ==Nz) {
    i = Nx-1;
    j = Nz-1;
    vvf[j][i][iur] = vvf[j][i][ir];
  }

  // Restore arrays
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));

  // Restore and map local to global
  PetscCall(DMStagVecRestoreArray(dm,vflocal,&vvf)); 
  PetscCall(DMLocalToGlobalBegin(dm,vflocal,INSERT_VALUES,usr->volf)); 
  PetscCall(DMLocalToGlobalEnd  (dm,vflocal,INSERT_VALUES,usr->volf)); 
  PetscCall(VecDestroy(&vflocal)); 

  PetscCall(DMRestoreLocalVector(dm, &xlocal )); 
  
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

  // Print user parameters
  PetscCall(InputPrintData(usr)); 

  // Numerical solution using the FD pde object
  PetscCall(StokesDarcy_Numerical(usr)); 

  // Destroy objects
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
