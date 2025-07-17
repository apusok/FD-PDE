// ---------------------------------------
// Two-phase StokesDarcy model with an interface which is captured using the Phasefield method
// R = 0, infinitely small compaction length, so that it is Stokes flow.
// Rheology: visco-elasto-plastic model, inifinitely large C, G and Z.
// run: ./tests/test_stokesdarcy2field_rt.app -nx 100 -nz 100 -pc_type lu -pc_factor_mat_solver_type umfpack
// python test: ./tests/python/test_stokesdarcy2field_rt.py
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

#include "petsc.h"
#include "../src/fdpde_stokesdarcy2field.h"
#include "../src/consteq.h"
#include "../src/dmstagoutput.h"

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
  PetscScalar    gamma, eps;  //phasefield
  PetscScalar    L, H;
  PetscScalar    xmin, zmin, xmax, zmax;
  PetscScalar    P0, etamin; //eta_v0, vi, C, P0, zeta_v0, etamin;
  PetscScalar    lam_v, eta_d, eta_u, zeta_d, zeta_u, C_u, C_d, Fu, Fd, z_in;
  PetscScalar    lambda,lam_p,G,Z,q,R,phi_0,n,nh;
  PetscBool      plasticity;
  PetscScalar    t, dt, dtck;
  char           fname_out[FNAME_LENGTH];
  char           fname_in[FNAME_LENGTH];
  char           fdir_out[FNAME_LENGTH]; 
} Params;

// user defined and model-dependent variables
typedef struct {
  Params        *par;
  PetscBag       bag;
  MPI_Comm       comm;
  PetscMPIInt    rank;
  DM             dmeps, dmf, dmPV; //dmf - phasefield DM, dmPV - velocity
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
PetscErrorCode FormCoefficient(FDPDE, DM, Vec, DM, Vec, void*);
PetscErrorCode FormBCList(DM, Vec, DMStagBCList, void*);
PetscErrorCode UpdateStrainRates(DM,Vec,void*);
PetscErrorCode UpdateStressOld(DM,void*);

//PetscErrorCode Phase_Numerical(void*);
PetscErrorCode SetInitialField(DM,Vec,void*);
PetscErrorCode UpdateDF(DM,Vec,void*);
PetscErrorCode ExplicitStep(DM,Vec,Vec,PetscScalar,void*);
PetscErrorCode UpdateCornerF(DM, Vec, void*);
PetscErrorCode UpdateVolFrac(DM, Vec, void*);

const char coeff_description[] = 
"  << Stokes-Darcy Coefficients >> \n"
"  A = eta_vep \n"
"  B = phi*F*ez - div(chi_s * tau_old) + grad(chi_p * diff_pold) \n"
"      (F = ((rho^s-rho^f)*U*L/eta_ref)*(g*L/U^2)) (0, if g=0) \n"
"      (tau_old: stress rotation tensor; diff_pold: pressure difference  ) \n"
"  C = 0 \n"
"  D1 = zeta_vep - 2/3*eta_vep \n"
"  D2 = -Kphi*R^2 (R: ratio of the compaction length to the global length scale) \n"
"  D3 = Kphi*R^2*F*ez \n";

const char bc_description[] =
"  << Stokes-Darcy BCs >> \n"
"  LEFT/BOTTOM: Symmetric boundary conditions (zero normal velocity, free slip) \n"
"  UP: extension Vz = Vi, free slip dVx/dz=0 \n"
"  RIGHT: compression Vx=Vi, free slip dVz/dx = 0\n";


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
  Vec            f, fprev;
  Vec            x, xguess; // xcoeff;
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
  PetscCall(FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_STOKESDARCY2FIELD,&fd));
  PetscCall(FDPDESetUp(fd));
  PetscCall(FDPDEGetDM(fd,&dm)); 

  // Create DM/vec for strain rates, yield, deviatoric/volumetric stress, and stress_old
  PetscCall(DMStagCreateCompatibleDMStag(dm,4,0,4,0,&usr->dmeps)); 
  PetscCall(DMSetUp(usr->dmeps)); 
  PetscCall(DMStagSetUniformCoordinatesProduct(usr->dmeps,xmin,xmax,zmin,zmax,0.0,0.0));
  PetscCall(DMCreateGlobalVector(usr->dmeps,&usr->xeps)); 
  PetscCall(DMCreateGlobalVector(usr->dmeps,&usr->xyield)); 
  PetscCall(DMCreateGlobalVector(usr->dmeps,&usr->xtau)); 
  PetscCall(DMCreateGlobalVector(usr->dmeps,&usr->xDP)); 
  PetscCall(DMCreateGlobalVector(usr->dmeps,&usr->xtau_old)); 
  PetscCall(DMCreateGlobalVector(usr->dmeps,&usr->xDP_old)); 

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

  // Create a vector to store u, p in userdata
  PetscCall(FDPDEGetSolution(fd,&x));
  PetscCall(VecDuplicate(x, &usr->xVel));

  // Create DM/vec for tau_old and DP_old, Initialise the two Vecs as zeros.
  PetscCall(VecZeroEntries(usr->xtau_old)); 
  PetscCall(VecZeroEntries(usr->xDP_old)); 

  // Set coefficients and BC evaluation function
  PetscCall(FDPDESetFunctionBCList(fd,FormBCList,bc_description,usr)); 
  PetscCall(FDPDESetFunctionCoefficient(fd,FormCoefficient,coeff_description,usr)); 
  PetscCall(FDPDEView(fd)); 


  // Initialise the phase field
  PetscCall(SetInitialField(dmf,f,usr));
  PetscCall(VecCopy(f, fprev));
  //interpolate phase values on the face and edges before FDPDE solver
    PetscCall(UpdateCornerF(dmf, f, usr)); 
    PetscCall(UpdateVolFrac(dmf, f, usr)); 
    

#if 0
  { // initialise the phase field

    while (istep<50) {
    
      //one step forward to get f at the next step (extract velocity data within it)
            PetscCall(UpdateDF(dmf, fprev, usr); 
      PetscCall(ExplicitStep(dmf, fprev, f, usr->par->dt, usr); 

      // Phasefield: copy f to prev
      PetscCall(VecCopy(f,fprev); 

      istep++;
    }

    istep = 0;

    //interpolate phase values on the face and edges before FDPDE solver
    PetscCall(UpdateCornerF(dmf, f, usr); 
    PetscCall(UpdateVolFrac(dmf, f, usr); 


  }
#endif

#if 1
  {  
  // Create initial guess with a linear viscous
  usr->par->plasticity = PETSC_FALSE; 
  PetscPrintf(PETSC_COMM_WORLD,"\n# INITIAL GUESS #\n");
  PetscCall(FDPDESolve(fd,NULL));
  PetscCall(FDPDEGetSolution(fd,&x));

  PetscCall(VecCopy(x, usr->xVel)); 

  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s_solution_initial",usr->par->fdir_out,usr->par->fname_out));
  PetscCall(DMStagViewBinaryPython(dm,x,fout));

  PetscCall(FDPDEGetSolutionGuess(fd,&xguess));  
  PetscCall(VecCopy(x,xguess));
  PetscCall(VecDestroy(&x))
  PetscCall(VecDestroy(&xguess));
  }
#endif
  usr->par->plasticity = PETSC_FALSE; //PETSC_TRUE; //switch off/on plasticity



  //-- initialise the phase field by computing 50 steps
  //PetscCall(UpdateDF(dmf, fprev, usr)); 
  //PetscCall(ExplicitStep(dmf, fprev, f, dt, usr)); 

  // output - initial state of the phase field
  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s_phase_initial",usr->par->fdir_out,usr->par->fname_out));
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
      
#if 0  //update or notupdate flow field for the second stage
      {
      // update flow field according to the phase field after 1st stage
      PetscCall(UpdateCornerF(dmf, f, usr)); 
      PetscCall(UpdateVolFrac(dmf, f, usr)); 
      PetscCall(FDPDESolve(fd,NULL));
      PetscCall(FDPDEGetSolution(fd,&x));
      PetscCall(VecCopy(x, usr->xVel));
      }
#endif
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

    // Update xtau_old and xDP_old
    PetscCall(UpdateStressOld(usr->dmeps,usr));

    // Phasefield: save f and fprev into global vectors
    usr->f     = f;
    usr->fprev = fprev;
    // Phasefield: copy f to prev
    PetscCall(VecCopy(f,fprev)); 

    // write before changing time steps
    if (iwrt || istep == 0) {

      iwrt = PETSC_FALSE;
      
      // Output solution to file
      PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s_solution_ts%1.3d",usr->par->fdir_out,usr->par->fname_out,ickpt));
      PetscCall(DMStagViewBinaryPython(dm,x,fout));
      /*
      PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s_strain_ts%1.3d",usr->par->fdir_out,usr->par->fname_out,ickpt));
      PetscCall(DMStagViewBinaryPython(usr->dmeps,usr->xeps,fout));

      PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s_stress_ts%1.3d",usr->par->fdir_out,usr->par->fname_out,ickpt));
      PetscCall(DMStagViewBinaryPython(usr->dmeps,usr->xtau,fout));

      PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s_stressold_ts%1.3d",usr->par->fdir_out,usr->par->fname_out,ickpt));
      PetscCall(DMStagViewBinaryPython(usr->dmeps,usr->xtau_old,fout));

      PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s_dpold_ts%1.3d",usr->par->fdir_out,usr->par->fname_out,ickpt));
      PetscCall(DMStagViewBinaryPython(usr->dmeps,usr->xDP_old,fout));

      PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s_yield_ts%1.3d",usr->par->fdir_out,usr->par->fname_out,ickpt));
      PetscCall(DMStagViewBinaryPython(usr->dmeps,usr->xyield,fout));

      PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s_residual_ts%1.3d",usr->par->fdir_out,usr->par->fname_out,ickpt));
      PetscCall(DMStagViewBinaryPython(dm,fd->r,fout));

      PetscCall(FDPDEGetCoefficient(fd,&dmcoeff,&xcoeff));
      PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s_coefficient_ts%1.3d",usr->par->fdir_out,usr->par->fname_out,ickpt));
      PetscCall(DMStagViewBinaryPython(dmcoeff,xcoeff,fout));
      */
      PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s_phase_ts%1.3d",usr->par->fdir_out,usr->par->fname_out,ickpt));
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
  PetscCall(DMDestroy(&dmf)); 

  PetscCall(VecDestroy(&usr->xVel));
  PetscCall(DMDestroy(&usr->dmPV));
  
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&usr->xeps));
  PetscCall(VecDestroy(&usr->xtau));
  PetscCall(VecDestroy(&usr->xDP));
  PetscCall(VecDestroy(&usr->xyield));
  PetscCall(VecDestroy(&usr->xtau_old));
  PetscCall(VecDestroy(&usr->xDP_old));
  PetscCall(DMDestroy(&dm));
  PetscCall(DMDestroy(&usr->dmeps));
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
  PetscCall(PetscBagRegisterString(bag,&par->fdir_out,FNAME_LENGTH,"./","output_dir","Name for output directory, set with: -output_dir <dirname>")); 

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
// FormCoefficient
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormCoefficient"
PetscErrorCode FormCoefficient(FDPDE fd, DM dm, Vec x, DM dmcoeff, Vec coeff, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz;
  Vec            coefflocal, xepslocal, xtaulocal, xDPlocal, xyieldlocal, toldlocal, poldlocal;
  Vec            flocal, volflocal;
  PetscScalar    **coordx,**coordz;
  PetscInt       iprev, inext, icenter;
  PetscScalar    R;//lambda,q,n;
  PetscScalar    ***c, ***xxs, ***xxy, ***xxp;
  PetscScalar    dt,F,G,Z,lam_p,phi,Kphi,eta_v,zeta_v,eta_e,zeta_e;
  PetscScalar    eta_u, eta_d, zeta_u, zeta_d, F_u, F_d, C_d, C_u; //z_in
  PetscScalar    lam_v, em, nh;
  // PetscScalar    ccc, eps;
  PetscFunctionBeginUser;

  dt = usr->par->dt;

  // lambda = usr->par->lambda;
  R = usr->par->R;
  // n = usr->par->n;
 
  G = usr->par->G;
  Z = usr->par->Z;
  // q = usr->par->q;
  lam_p = usr->par->lam_p;
  lam_v = usr->par->lam_v;
  em    = usr->par->etamin;
  nh    = usr->par->nh;

  // eps = usr->par->eps;

  // constant shear and compaction viscosity of up and down, force(density) up and down
  eta_u  = usr->par->eta_u;
  eta_d  = usr->par->eta_d;
  zeta_u = eta_u/lam_v;
  zeta_d = eta_d/lam_v;
  F_u     = usr->par->Fu;
  F_d     = usr->par->Fd;

  //  PetscPrintf(PETSC_COMM_WORLD, "fu, fd = %g, %g", F_u, F_d);
  
  // cohesion of up and down -- both are infinity in this case
  C_u   = usr->par->C_u;
  C_d = usr->par->C_d;

  // (sharp) interface position
  // z_in = usr->par->z_in;

  // Uniform porosity and permeability
  phi = usr->par->phi_0;
  Kphi = 1.0;

  // Effective shear and Compaction viscosity due to elasticity
  eta_e = G*dt;
  zeta_e = Z*dt;


  // phase field
  PetscCall(DMGetLocalVector(usr->dmf, &flocal)); 
  PetscCall(DMGlobalToLocal (usr->dmf, usr->f, INSERT_VALUES, flocal)); 

  // volume fraction
  PetscCall(DMGetLocalVector(usr->dmf, &volflocal)); 
  PetscCall(DMGlobalToLocal (usr->dmf, usr->volf, INSERT_VALUES, volflocal)); 

  // Strain rates
  PetscCall(UpdateStrainRates(dm,x,usr)); 
  PetscCall(DMGetLocalVector(usr->dmeps, &xepslocal)); 
  PetscCall(DMGlobalToLocal (usr->dmeps, usr->xeps, INSERT_VALUES, xepslocal)); 

  
  // Stress_old
  PetscCall(DMGetLocalVector(usr->dmeps, &toldlocal)); 
  PetscCall(DMGlobalToLocal (usr->dmeps, usr->xtau_old, INSERT_VALUES, toldlocal)); 
  PetscCall(DMGetLocalVector(usr->dmeps, &poldlocal)); 
  PetscCall(DMGlobalToLocal (usr->dmeps, usr->xDP_old, INSERT_VALUES, poldlocal)); 

  // Local vectors
  PetscCall(DMCreateLocalVector (usr->dmeps,&xtaulocal)); 
  PetscCall(DMStagVecGetArray(usr->dmeps,xtaulocal,&xxs)); 

  PetscCall(DMCreateLocalVector (usr->dmeps,&xDPlocal)); 
  PetscCall(DMStagVecGetArray(usr->dmeps,xDPlocal,&xxp)); 

  PetscCall(DMCreateLocalVector (usr->dmeps,&xyieldlocal)); 
  PetscCall(DMStagVecGetArray(usr->dmeps,xyieldlocal,&xxy)); 

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
      PetscScalar told_xx_e,told_zz_e,told_xz_e,told_II_e,pold,chis,chip,zeta;
      PetscScalar cs[4],told_xx[4],told_zz[4],told_xz[4],told_II[4];

      { // A = eta (center, c=1), and also compute chi_s and chi_p (center, c = 4,5)
        DMStagStencil point;
        PetscScalar   epsII,exx,ezz,exz,txx,tzz,txz,tauII,epsII_dev,Y,YC;
        PetscScalar   div;//,div2;
        PetscScalar   etaP_inv, eta, zetaP_inv, DeltaP;
        PetscScalar   ff, volf;

        // get the phase values in the element
        point.i = i; point.j = j; point.loc = ELEMENT; point.c = 0;
        PetscCall(DMStagVecGetValuesStencil(usr->dmf,flocal,1,&point,&ff)); 
        PetscCall(DMStagVecGetValuesStencil(usr->dmf,volflocal,1,&point,&volf)); 

        //        if (usr->par->diffuse) {
          // diffuse interface
        //  eta_v  = (1.0 - ff)*eta_d + ff*eta_u;
        //  zeta_v = (1.0 - ff)*zeta_d + ff*zeta_u;
        //  Y      = (1.0 - ff)*C_d + ff*C_u;
        //} else {
          // sharp interface
          //if (ff <= 0.5) {eta_v = eta_d; zeta_v = zeta_d; Y = C_d;}
          //else           {eta_v = eta_u; zeta_v = zeta_u; Y = C_u;}
          //}

        eta_v  = eta_u  * volf + eta_d  * (1.0 - volf);
        zeta_v = zeta_u * volf + zeta_d * (1.0 - volf);
        Y      = C_u    * volf + C_d    * (1.0 - volf);
        
        // compaction failure criteria
        YC = Y*lam_p;

        // second invariant of strain rate
        point.i = i; point.j = j; point.loc = ELEMENT; 
        point.c = 0;
        PetscCall(DMStagVecGetValuesStencil(usr->dmeps,xepslocal,1,&point,&exx)); 
        PetscCall(DMStagVecGetValuesStencil(usr->dmeps,toldlocal,1,&point,&told_xx_e)); 
        PetscCall(DMStagVecGetValuesStencil(usr->dmeps,poldlocal,1,&point,&pold)); 
        point.c = 1;
        PetscCall(DMStagVecGetValuesStencil(usr->dmeps,xepslocal,1,&point,&ezz)); 
        PetscCall(DMStagVecGetValuesStencil(usr->dmeps,toldlocal,1,&point,&told_zz_e)); 
        point.c = 2;
        PetscCall(DMStagVecGetValuesStencil(usr->dmeps,xepslocal,1,&point,&exz)); 
        PetscCall(DMStagVecGetValuesStencil(usr->dmeps,toldlocal,1,&point,&told_xz_e)); 
        point.c = 3;
        PetscCall(DMStagVecGetValuesStencil(usr->dmeps,xepslocal,1,&point,&epsII)); 
        PetscCall(DMStagVecGetValuesStencil(usr->dmeps,toldlocal,1,&point,&told_II_e)); 

        // second invariant of deviatoric strain rate
        epsII_dev = PetscPowScalar((PetscPowScalar(epsII,2) - 1.0/6.0*PetscPowScalar(exx+ezz,2)),0.5);

        PetscScalar vp_inv;
        // effective shear viscosity
        if (usr->par->plasticity) { etaP_inv = 2.0*epsII_dev/Y;}
        else { etaP_inv = 0.0;}

        vp_inv = PetscPowScalar(etaP_inv, nh) + PetscPowScalar(1.0/eta_v, nh);
        vp_inv = PetscPowScalar(vp_inv, 1.0/nh);
        eta = em + (1.0 - phi)/(vp_inv + 1.0/eta_e);
        
        // effective bulk viscosity
        //zetaP_inv = etaP_inv/lam_v;
        zetaP_inv = PetscAbs(exx+ezz)/YC;
        vp_inv = PetscPowScalar(zetaP_inv, nh) + PetscPowScalar(1.0/zeta_v, nh);
        vp_inv = PetscPowScalar(vp_inv, 1.0/nh);
        zeta = em + (1.0 - phi)/(vp_inv + 1.0/zeta_e);
        
        // elastic stress evolution parameter
        // remove the cutoff minimum viscosity in calculating the built-up stress
        chis = (eta-em)/(G*dt);
        chip = (zeta-em)/(Z*dt);
        
        point.i = i; point.j = j; point.loc = ELEMENT; point.c = 1;

        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
        c[j][i][idx] = eta;

        //PetscPrintf(PETSC_COMM_WORLD, "A (center) = %g \n", c[j][i][idx]); 
        
        // deviatoric stress and its second invariant
        // 1. remove the minimum etamin while calculating the built-up stress
        // 2. devide 1 - phi to fix the reduced effective stress due to porosity
        PetscScalar phis;
      
        phis = 1 - phi;
        div = exx + ezz;
        txx = (2.0*(eta-em)*(exx-1.0/3.0*div) + chis*told_xx_e)/phis;
        tzz = (2.0*(eta-em)*(ezz-1.0/3.0*div) + chis*told_zz_e)/phis;
        txz = (2.0*(eta-em)*exz + chis*told_xz_e)/phis;
        tauII = PetscPowScalar(0.5*(txx*txx + tzz*tzz + 2*txz*txz),0.5);
              
        
        // volumetric stress
        DeltaP = (-(zeta-em)*div+chip*pold)/phis;

        // save stresses for output
        PetscCall(DMStagGetLocationSlot(usr->dmeps,ELEMENT,0,&idx));  xxs[j][i][idx] = txx; xxy[j][i][idx] = Y; xxp[j][i][idx] = DeltaP;
        PetscCall(DMStagGetLocationSlot(usr->dmeps,ELEMENT,1,&idx));  xxs[j][i][idx] = tzz;
        PetscCall(DMStagGetLocationSlot(usr->dmeps,ELEMENT,2,&idx));  xxs[j][i][idx] = txz;
        PetscCall(DMStagGetLocationSlot(usr->dmeps,ELEMENT,3,&idx));  xxs[j][i][idx] = tauII;
      }

      { // A = eta (corner, c=0)
        // also compute cs = eta/(G*dt) (four corners, c =1)
        DMStagStencil point[4];
        PetscScalar   ff[4], volf[4];
        PetscScalar   etaP_inv, eta, Y[4];
        PetscScalar   epsII[4],exx[4],ezz[4],exz[4],txx[4],tzz[4],txz[4],tauII[4],epsII_dev[4];
        PetscScalar   div;//,div2;
        PetscInt      ii;

        // second invariant of strain rate
        point[0].i = i; point[0].j = j; point[0].loc = DOWN_LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = DOWN_RIGHT; point[1].c = 0;
        point[2].i = i; point[2].j = j; point[2].loc = UP_LEFT;    point[2].c = 0;
        point[3].i = i; point[3].j = j; point[3].loc = UP_RIGHT;   point[3].c = 0;

        // collect phase values for the four corners
        PetscCall(DMStagVecGetValuesStencil(usr->dmf,flocal,4,point,ff)); 
        PetscCall(DMStagVecGetValuesStencil(usr->dmf,volflocal,4,point,volf)); 
        
        
        PetscCall(DMStagVecGetValuesStencil(usr->dmeps,xepslocal,4,point,exx)); 
        PetscCall(DMStagVecGetValuesStencil(usr->dmeps,toldlocal,4,point,told_xx)); 
                
        for (ii = 0; ii < 4; ii++) {point[ii].c = 1;}
        PetscCall(DMStagVecGetValuesStencil(usr->dmeps,xepslocal,4,point,ezz)); 
        PetscCall(DMStagVecGetValuesStencil(usr->dmeps,toldlocal,4,point,told_zz)); 

        for (ii = 0; ii < 4; ii++) {point[ii].c = 2;}
        PetscCall(DMStagVecGetValuesStencil(usr->dmeps,xepslocal,4,point,exz)); 
        PetscCall(DMStagVecGetValuesStencil(usr->dmeps,toldlocal,4,point,told_xz)); 

        // second invariant of strain rate
        for (ii = 0; ii < 4; ii++) {point[ii].c = 3;}
        PetscCall(DMStagVecGetValuesStencil(usr->dmeps,xepslocal,4,point,epsII)); 
        PetscCall(DMStagVecGetValuesStencil(usr->dmeps,toldlocal,4,point,told_II)); 

        // second invariant of deviatoric strain rate
        for (ii = 0; ii < 4; ii++) {
          epsII_dev[ii] = PetscPowScalar((PetscPowScalar(epsII[ii],2) - 1.0/6.0*PetscPowScalar(exx[ii]+ezz[ii],2)),0.5);
        }

        for (ii = 0; ii < 4; ii++) {

          //          if (usr->par->diffuse) {
            // diffuse interface
          // eta_v  = (1.0 - ff[ii])*eta_d + ff[ii]*eta_u;
          //  zeta_v = (1.0 - ff[ii])*zeta_d + ff[ii]*zeta_u;
          //  Y[ii]  = (1.0 - ff[ii])*C_d + ff[ii]*C_u;
          //} else {
            // sharp interface
            //if (ff[ii] <= 0.5) {eta_v = eta_d; zeta_v = zeta_d; Y[ii] = C_d;}
            //else           {eta_v = eta_u; zeta_v = zeta_u; Y[ii] = C_u;}
            //}

            eta_v  = eta_u  * volf[ii] + eta_d  * (1.0 - volf[ii]);
            zeta_v = zeta_u * volf[ii] + zeta_d * (1.0 - volf[ii]);
            Y[ii]  = C_u    * volf[ii] + C_d    * (1.0 - volf[ii]);

          PetscScalar vp_inv;
          // effective shear viscosity
          if (usr->par->plasticity) { etaP_inv = 2.0*epsII_dev[ii]/Y[ii];}
          else { etaP_inv = 0.0;}

          vp_inv = PetscPowScalar(etaP_inv, nh) + PetscPowScalar(1.0/eta_v, nh);
          vp_inv = PetscPowScalar(vp_inv, 1.0/nh);
          eta = em + (1.0 - phi)/(vp_inv + 1.0/eta_e);

          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, 0, &idx)); 
          c[j][i][idx] = eta;


          //PetscPrintf(PETSC_COMM_WORLD, "A (corner) = %g \n", c[j][i][idx]); 

          
          // elastic stress evolution parameter
          cs[ii] = (eta-em)/(G*dt);
          
          // deviatoric stress and its second invariant
          PetscScalar phis;

          phis = 1 - phi;
          div = exx[ii] + ezz[ii];

          txx[ii] = (2.0*(eta-em)*(exx[ii]-1.0/3.0*div) + cs[ii]*told_xx[ii])/phis;
          tzz[ii] = (2.0*(eta-em)*(ezz[ii]-1.0/3.0*div) + cs[ii]*told_zz[ii])/phis;
          txz[ii] = (2.0*(eta-em)*exz[ii] + cs[ii]*told_xz[ii])/phis;
          tauII[ii] = PetscPowScalar(0.5*(txx[ii]*txx[ii] + tzz[ii]*tzz[ii] + 2.0*txz[ii]*txz[ii]),0.5);
        }

        // save stresses
        PetscCall(DMStagGetLocationSlot(usr->dmeps,DOWN_LEFT,0,&idx));  xxs[j][i][idx] = txx[0]; xxy[j][i][idx] = Y[0];
        PetscCall(DMStagGetLocationSlot(usr->dmeps,DOWN_LEFT,1,&idx));  xxs[j][i][idx] = tzz[0];
        PetscCall(DMStagGetLocationSlot(usr->dmeps,DOWN_LEFT,2,&idx));  xxs[j][i][idx] = txz[0];
        PetscCall(DMStagGetLocationSlot(usr->dmeps,DOWN_LEFT,3,&idx));  xxs[j][i][idx] = tauII[0];

        PetscCall(DMStagGetLocationSlot(usr->dmeps,DOWN_RIGHT,0,&idx));  xxs[j][i][idx] = txx[1]; xxy[j][i][idx] = Y[1];
        PetscCall(DMStagGetLocationSlot(usr->dmeps,DOWN_RIGHT,1,&idx));  xxs[j][i][idx] = tzz[1];
        PetscCall(DMStagGetLocationSlot(usr->dmeps,DOWN_RIGHT,2,&idx));  xxs[j][i][idx] = txz[1];
        PetscCall(DMStagGetLocationSlot(usr->dmeps,DOWN_RIGHT,3,&idx));  xxs[j][i][idx] = tauII[1];

        PetscCall(DMStagGetLocationSlot(usr->dmeps,UP_LEFT,0,&idx));  xxs[j][i][idx] = txx[2]; xxy[j][i][idx] = Y[2];
        PetscCall(DMStagGetLocationSlot(usr->dmeps,UP_LEFT,1,&idx));  xxs[j][i][idx] = tzz[2];
        PetscCall(DMStagGetLocationSlot(usr->dmeps,UP_LEFT,2,&idx));  xxs[j][i][idx] = txz[2];
        PetscCall(DMStagGetLocationSlot(usr->dmeps,UP_LEFT,3,&idx));  xxs[j][i][idx] = tauII[2];

        PetscCall(DMStagGetLocationSlot(usr->dmeps,UP_RIGHT,0,&idx));  xxs[j][i][idx] = txx[3]; xxy[j][i][idx] = Y[3];
        PetscCall(DMStagGetLocationSlot(usr->dmeps,UP_RIGHT,1,&idx));  xxs[j][i][idx] = tzz[3];
        PetscCall(DMStagGetLocationSlot(usr->dmeps,UP_RIGHT,2,&idx));  xxs[j][i][idx] = txz[3];
        PetscCall(DMStagGetLocationSlot(usr->dmeps,UP_RIGHT,3,&idx));  xxs[j][i][idx] = tauII[3];
      }

      { // B = phi*F*ek - grad(chi_s*tau_old) + div(chi_p*dP_old) (edges, c=0)
        // (F = (rho^s-rho^f)*U*L/eta_ref * g*L/U^2) 
        DMStagStencil point[4];
        PetscScalar   rhs[4], ff[4], volf[4], F2, F3;
        PetscInt      ii,jj;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 0;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = 0;
        point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = 0;

        // collect phase values for the four edges
        PetscCall(DMStagVecGetValuesStencil(usr->dmf,flocal,4,point,ff)); 
        PetscCall(DMStagVecGetValuesStencil(usr->dmf,volflocal,4,point,volf)); 

        //        PetscPrintf(PETSC_COMM_WORLD, "i,j = %d, %d; ff = %g, %g, %g, %g \n", i,j,ff[0], ff[1], ff[2], ff[3]); 

        //        if (usr->par->diffuse) {
          // diffuse interface
        //  F2 = (1.0 - ff[2])*F_d + ff[2]*F_u;
        //  F3 = (1.0 - ff[3])*F_d + ff[3]*F_u;
          //F2 = 1.0/((1.0-ff[2])/F_d + ff[2]/F_u);
          //F3 = 1.0/((1.0-ff[3])/F_d + ff[3]/F_u);
        //} else {
          // sharp interface
          //if (ff[2] <= 0.5) {F2 = F_d;} else {F2 = F_u;}
          //if (ff[3] <= 0.5) {F3 = F_d;} else {F3 = F_u;}

          /*
          if (ff[2] >= 0.5+0.125/ccc)      {F2 = F_u;}
          else if (ff[2] <= 0.5-0.125) {F2 = F_d;}
          else {F2 = 0.5*(F_u+F_d) + 4.0*ccc*(ff[2]-0.5)*(F_d - F_u);}

          if (ff[3] >= 0.5+0.125/ccc)      {F3 = F_u;}
          else if (ff[3] <= 0.5-0.125) {F3 = F_d;}
          else {F3 = 0.5*(F_u+F_d) + 4.0*ccc*(ff[3]-0.5)*(F_d - F_u);}
          */
          
          F2 = F_u*volf[2] + F_d*(1.0-volf[2]);
          F3 = F_u*volf[3] + F_d*(1.0-volf[3]);
          
          
          // }  
        
        ii = i - sx;
        jj = j - sz;

        // d(chi_s*tau_xy_old)/dz compute on the left and right
        // d(chi_s*tau_xy_old)/dx and gravity compute on the down boundary only
        // RHS = 0 on the true boundaries
        if (i > 0) {
          rhs[0] = -chis*told_xx_e/dx[ii];
          rhs[0]+= -(cs[2]*told_xz[2]-cs[0]*told_xz[0])/dz[jj];
          rhs[0]+= chip * pold/dx[ii];
        } else {
          rhs[0] = 0.0;
        }
        if (i < Nx-1) {
          rhs[1] = chis*told_xx_e/dx[ii];
          rhs[1]+= -chip*pold/dx[ii];
        } else {
          rhs[1] = 0.0;
        }
        if (j > 0) {
          rhs[2] = 0.5*phi*F2;
          rhs[2]+= -chis*told_zz_e/dz[jj];
          rhs[2]+= -(cs[1]*told_xz[1]-cs[0]*told_xz[0])/dx[ii];
          rhs[2]+= chip*pold/dz[jj];
        } else {
          rhs[2] = phi*F2;
        }
        if (j < Nz-1) {
          rhs[3] = 0.5*phi*F3;
          rhs[3]+= chis*told_zz_e/dz[jj];
          rhs[3]+= -chip*pold/dz[jj];
        } else {
          rhs[3] = phi*F3;
        }
        
        for (ii = 0; ii < 4; ii++) {
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] += rhs[ii];

          //                    PetscPrintf(PETSC_COMM_WORLD, "B = %g \n", c[j][i][idx]); 
          
        }
      }

      { // C = 0 (center, c=0)
        DMStagStencil point;
        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 0;
        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
        c[j][i][idx] = 0.0;
      }

      { // D1 = zeta - 2/3*A (center, c=2)
        DMStagStencil point;
        PetscInt      idxA;
        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 2;
        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 1;
        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idxA)); 
        c[j][i][idx] = zeta - 2.0/3.0*c[j][i][idxA] ;

        //        PetscPrintf(PETSC_COMM_WORLD, "D1 = %g \n", c[j][i][idx]); 
        
      }

      { // D2 = -R^2 * Kphi (edges, c=1)
        DMStagStencil point[4];
        // PetscScalar   xp[4],zp[4];
        PetscInt      ii;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 1;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 1;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = 1;
        point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = 1;

        // xp[0] = coordx[i][iprev  ]; zp[0] = coordz[j][icenter];
        // xp[1] = coordx[i][inext  ]; zp[1] = coordz[j][icenter];
        // xp[2] = coordx[i][icenter]; zp[2] = coordz[j][iprev  ];
        // xp[3] = coordx[i][icenter]; zp[3] = coordz[j][inext  ];

        for (ii = 0; ii < 4; ii++) {
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] = -pow(R,2) * Kphi;

          //          PetscPrintf(PETSC_COMM_WORLD, "D2 = %g \n", c[j][i][idx]); 
          
        }
      }

      { // D3 = R^2 * Kphi * F (edges, c=2)
        DMStagStencil point[4];
        PetscScalar   ff, volf;
        PetscInt      ii;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 2;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 2;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = 2;
        point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = 2;

        for (ii = 0; ii < 2; ii++) {
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] = 0.0;
        }
        //  nonzero if including the gravity
        for (ii = 2; ii < 4; ii++) {
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 

          // collect phase values for the four corners
          point[ii].c = 0;
          PetscCall(DMStagVecGetValuesStencil(usr->dmf,flocal,1,&point[ii],&ff)); 
          PetscCall(DMStagVecGetValuesStencil(usr->dmf,volflocal,1,&point[ii],&volf)); 

          //          if (usr->par->diffuse) {
            // diffuse interface
            //F = (1.0 - ff)*F_d + ff*F_u;
          // } else {
            // sharp interface
            //if (ff <= 0.5) {F = F_d;} else {F=F_u;}
            //}

            F = F_u*volf + F_d*(1.0-volf);
          
          
          c[j][i][idx] = -pow(R,2) * Kphi * F;

          //          PetscPrintf(PETSC_COMM_WORLD, "D3 = %g \n", c[j][i][idx]); 
          
        }
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

  // Restore and map local to global
  PetscCall(DMStagVecRestoreArray(usr->dmeps,xtaulocal,&xxs)); 
  PetscCall(DMLocalToGlobalBegin(usr->dmeps,xtaulocal,INSERT_VALUES,usr->xtau)); 
  PetscCall(DMLocalToGlobalEnd  (usr->dmeps,xtaulocal,INSERT_VALUES,usr->xtau)); 
  PetscCall(VecDestroy(&xtaulocal)); 

  PetscCall(DMStagVecRestoreArray(usr->dmeps,xDPlocal,&xxp)); 
  PetscCall(DMLocalToGlobalBegin(usr->dmeps,xDPlocal,INSERT_VALUES,usr->xDP)); 
  PetscCall(DMLocalToGlobalEnd  (usr->dmeps,xDPlocal,INSERT_VALUES,usr->xDP)); 
  PetscCall(VecDestroy(&xDPlocal)); 
  
  PetscCall(DMStagVecRestoreArray(usr->dmeps,xyieldlocal,&xxy)); 
  PetscCall(DMLocalToGlobalBegin(usr->dmeps,xyieldlocal,INSERT_VALUES,usr->xyield)); 
  PetscCall(DMLocalToGlobalEnd  (usr->dmeps,xyieldlocal,INSERT_VALUES,usr->xyield)); 
  PetscCall(VecDestroy(&xyieldlocal)); 

  PetscCall(DMRestoreLocalVector(usr->dmeps,&xepslocal)); 
  PetscCall(DMRestoreLocalVector(usr->dmeps,&toldlocal)); 
  PetscCall(DMRestoreLocalVector(usr->dmeps,&poldlocal)); 
  PetscCall(DMRestoreLocalVector(usr->dmf,  &flocal));    

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// UpdateStrainRates
// ---------------------------------------
PetscErrorCode UpdateStrainRates(DM dm, Vec x, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  DM             dmeps;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz, idx, ii;
  PetscInt       iprev, inext, icenter;
  PetscScalar    ***xx;
  PetscScalar    **coordx,**coordz;
  Vec            xeps, xepslocal,xlocal;
  PetscFunctionBeginUser;

  dmeps = usr->dmeps;
  xeps  = usr->xeps;

  // Local vectors
  PetscCall(DMCreateLocalVector (dmeps,&xepslocal)); 
  PetscCall(DMStagVecGetArray(dmeps,xepslocal,&xx)); 

  PetscCall(DMGetLocalVector(dm,&xlocal)); 
  PetscCall(DMGlobalToLocal (dm,x,INSERT_VALUES,xlocal)); 

  // Get domain corners
  PetscCall(DMStagGetGlobalSizes(dmeps, &Nx, &Nz,NULL));
  PetscCall(DMStagGetCorners(dmeps, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 
  
  // Get dm coordinates array
  PetscCall(DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter)); 
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm,LEFT,&iprev));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm,RIGHT,&inext)); 

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      DMStagStencil  pointC, pointN[4];
      PetscScalar    epsIIc, exxc, ezzc, exzc, epsIIn[4], exxn[4], ezzn[4], exzn[4];

      // Strain rates: center
      pointC.i = i; pointC.j = j; pointC.loc = ELEMENT; pointC.c = 0;
      PetscCall(DMStagGetPointStrainRates(dm,xlocal,1,&pointC,&epsIIc,&exxc,&ezzc,&exzc)); 

      PetscCall(DMStagGetLocationSlot(dmeps,ELEMENT,0,&idx));  xx[j][i][idx] = exxc;
      PetscCall(DMStagGetLocationSlot(dmeps,ELEMENT,1,&idx));  xx[j][i][idx] = ezzc;
      PetscCall(DMStagGetLocationSlot(dmeps,ELEMENT,2,&idx));  xx[j][i][idx] = exzc;
      PetscCall(DMStagGetLocationSlot(dmeps,ELEMENT,3,&idx));  xx[j][i][idx] = epsIIc;

      // Strain rates: corner
      pointN[0].i = i; pointN[0].j = j; pointN[0].loc = DOWN_LEFT;  pointN[0].c = 0;
      pointN[1].i = i; pointN[1].j = j; pointN[1].loc = DOWN_RIGHT; pointN[1].c = 0;
      pointN[2].i = i; pointN[2].j = j; pointN[2].loc = UP_LEFT;    pointN[2].c = 0;
      pointN[3].i = i; pointN[3].j = j; pointN[3].loc = UP_RIGHT;   pointN[3].c = 0;
      PetscCall(DMStagGetPointStrainRates(dm,xlocal,4,pointN,epsIIn,exxn,ezzn,exzn)); 
      
      if (i==0) { // boundaries
        pointC.i = i; pointC.j = j; pointC.loc = ELEMENT; pointC.c = 0;
        PetscCall(DMStagGetPointStrainRates(dm,xlocal,1,&pointC,&epsIIc,&exxc,&ezzc,&exzc)); 
        ezzn[0] = ezzc;
        exxn[0] = exxc;
      }

      if (i==Nx-1) { // boundaries
        pointC.i = i; pointC.j = j; pointC.loc = ELEMENT; pointC.c = 0;
        PetscCall(DMStagGetPointStrainRates(dm,xlocal,1,&pointC,&epsIIc,&exxc,&ezzc,&exzc)); 
        ezzn[1] = ezzc;
        exxn[1] = exxc;
      }

      if (j==0) { // boundaries
        pointC.i = i; pointC.j = j; pointC.loc = ELEMENT; pointC.c = 0;
        PetscCall(DMStagGetPointStrainRates(dm,xlocal,1,&pointC,&epsIIc,&exxc,&ezzc,&exzc)); 
        exxn[0] = exxc;
        ezzn[0] = ezzc;
      }

      if (j==Nz-1) { // boundaries
        pointC.i = i; pointC.j = j; pointC.loc = ELEMENT; pointC.c = 0;
        PetscCall(DMStagGetPointStrainRates(dm,xlocal,1,&pointC,&epsIIc,&exxc,&ezzc,&exzc)); 
        exxn[2] = exxc;
        ezzn[2] = ezzc;
      }

      if ((i==Nx-1) && (j==Nz-1)) { // boundaries
        pointC.i = i; pointC.j = j; pointC.loc = ELEMENT; pointC.c = 0;
        PetscCall(DMStagGetPointStrainRates(dm,xlocal,1,&pointC,&epsIIc,&exxc,&ezzc,&exzc)); 
        exxn[3] = exxc;
        ezzn[3] = ezzc;
      }
      
      if ((i==0) || (i==Nx-1) || (j==0) || (j==Nz-1)) { // boundaries
        for (ii = 0; ii < 4; ii++) {
          epsIIn[ii] = PetscPowScalar(0.5*(exxn[ii]*exxn[ii] + ezzn[ii]*ezzn[ii] + 2.0*exzn[ii]*exzn[ii]),0.5);
        }
      }

      PetscCall(DMStagGetLocationSlot(dmeps,DOWN_LEFT,0,&idx));  xx[j][i][idx] = exxn[0];
      PetscCall(DMStagGetLocationSlot(dmeps,DOWN_LEFT,1,&idx));  xx[j][i][idx] = ezzn[0];
      PetscCall(DMStagGetLocationSlot(dmeps,DOWN_LEFT,2,&idx));  xx[j][i][idx] = exzn[0];
      PetscCall(DMStagGetLocationSlot(dmeps,DOWN_LEFT,3,&idx));  xx[j][i][idx] = epsIIn[0];

      PetscCall(DMStagGetLocationSlot(dmeps,DOWN_RIGHT,0,&idx));  xx[j][i][idx] = exxn[1];
      PetscCall(DMStagGetLocationSlot(dmeps,DOWN_RIGHT,1,&idx));  xx[j][i][idx] = ezzn[1];
      PetscCall(DMStagGetLocationSlot(dmeps,DOWN_RIGHT,2,&idx));  xx[j][i][idx] = exzn[1];
      PetscCall(DMStagGetLocationSlot(dmeps,DOWN_RIGHT,3,&idx));  xx[j][i][idx] = epsIIn[1];

      PetscCall(DMStagGetLocationSlot(dmeps,UP_LEFT,0,&idx));  xx[j][i][idx] = exxn[2];
      PetscCall(DMStagGetLocationSlot(dmeps,UP_LEFT,1,&idx));  xx[j][i][idx] = ezzn[2];
      PetscCall(DMStagGetLocationSlot(dmeps,UP_LEFT,2,&idx));  xx[j][i][idx] = exzn[2];
      PetscCall(DMStagGetLocationSlot(dmeps,UP_LEFT,3,&idx));  xx[j][i][idx] = epsIIn[2];

      PetscCall(DMStagGetLocationSlot(dmeps,UP_RIGHT,0,&idx));  xx[j][i][idx] = exxn[3];
      PetscCall(DMStagGetLocationSlot(dmeps,UP_RIGHT,1,&idx));  xx[j][i][idx] = ezzn[3];
      PetscCall(DMStagGetLocationSlot(dmeps,UP_RIGHT,2,&idx));  xx[j][i][idx] = exzn[3];
      PetscCall(DMStagGetLocationSlot(dmeps,UP_RIGHT,3,&idx));  xx[j][i][idx] = epsIIn[3];
    }
  }

  // Restore arrays
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));

  // Restore and map local to global
  PetscCall(DMStagVecRestoreArray(dmeps,xepslocal,&xx)); 
  PetscCall(DMLocalToGlobalBegin(dmeps,xepslocal,INSERT_VALUES,xeps)); 
  PetscCall(DMLocalToGlobalEnd  (dmeps,xepslocal,INSERT_VALUES,xeps)); 
  PetscCall(VecDestroy(&xepslocal)); 

  PetscCall(DMRestoreLocalVector(dm, &xlocal )); 
  
  PetscFunctionReturn(PETSC_SUCCESS);
}


// ---------------------------------------
// FormBCList
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormBCList"
PetscErrorCode FormBCList(DM dm, Vec x, DMStagBCList bclist, void *ctx)
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

  // LEFT Boundary - P
  /*
  PetscCall(DMStagBCListGetValues(bclist,'w','o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET_STAG;//BC_NEUMANN;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  */
  
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

  // RIGHT Boundary - P
  /*
  PetscCall(DMStagBCListGetValues(bclist,'e','o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET_STAG;//BC_NEUMANN;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  */
    
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

  // DOWN Boundary - P
  /*
  PetscCall(DMStagBCListGetValues(bclist,'s','o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET_STAG;//BC_NEUMANN;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  */
  
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

  // UP Boundary - P
  //PetscCall(DMStagBCListGetValues(bclist,'n','o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  //for (k=0; k<n_bc; k++) {
  //  value_bc[k] = 0.0;
    //  type_bc[k] = BC_DIRICHLET;//BC_DIRICHLET_STAG;//BC_NEUMANN;
    //}
  //PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  PetscCall(DMStagBCListGetValues(bclist,'n','o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<1; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET_STAG;//BC_NEUMANN;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// UpdateStressOld
// ---------------------------------------
PetscErrorCode UpdateStressOld(DM dmeps, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  Vec            xtaulocal, xDPlocal, tauold_local,DPold_local;
  PetscFunctionBeginUser;

  // Get local vectors from dmeps
  PetscCall(DMGetLocalVector(usr->dmeps, &xtaulocal)); 
  PetscCall(DMGlobalToLocal (usr->dmeps, usr->xtau, INSERT_VALUES, xtaulocal);) 

  PetscCall(DMGetLocalVector(usr->dmeps, &xDPlocal)); 
  PetscCall(DMGlobalToLocal (usr->dmeps, usr->xDP, INSERT_VALUES, xDPlocal)); 

  // Create local vectors for the stress_old terms
  PetscCall(DMCreateLocalVector (usr->dmeps,&tauold_local)); 
  PetscCall(DMCreateLocalVector (usr->dmeps,&DPold_local)); 
  PetscCall(VecCopy(xtaulocal, tauold_local)); 
  PetscCall(VecCopy(xDPlocal, DPold_local)); 

  // create array from tauold_local and add the rotation terms
  // No rotation terms for now

  // Restore and map local to global
  PetscCall(DMLocalToGlobalBegin(usr->dmeps,tauold_local,INSERT_VALUES,usr->xtau_old)); 
  PetscCall(DMLocalToGlobalEnd  (usr->dmeps,tauold_local,INSERT_VALUES,usr->xtau_old)); 
  PetscCall(VecDestroy(&tauold_local)); 

  PetscCall(DMLocalToGlobalBegin(usr->dmeps,DPold_local,INSERT_VALUES,usr->xDP_old)); 
  PetscCall(DMLocalToGlobalEnd  (usr->dmeps,DPold_local,INSERT_VALUES,usr->xDP_old)); 
  PetscCall(VecDestroy(&DPold_local)); 

  PetscCall(DMRestoreLocalVector(usr->dmeps,&xtaulocal)); 
  PetscCall(DMRestoreLocalVector(usr->dmeps,&xDPlocal)); 

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
  PetscCall(VecDestroy(&xlocal)); 

  PetscCall(DMStagVecRestoreArray(dm,xplocal,&xxp));
  PetscCall(DMLocalToGlobalBegin(dm,xplocal,INSERT_VALUES,xprev)); 
  PetscCall(DMLocalToGlobalEnd  (dm,xplocal,INSERT_VALUES,xprev)); 
  PetscCall(VecDestroy(&xplocal)); 

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
  PetscCall(VecDestroy(&xlocal)); 
  
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
  PetscCall(PetscOptionsInsert(PETSC_NULL,&argc,&argv,NULL)); 

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
