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
  PetscErrorCode ierr;

  PetscFunctionBegin;

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
  ierr = FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_STOKESDARCY2FIELD,&fd);CHKERRQ(ierr);
  ierr = FDPDESetUp(fd);CHKERRQ(ierr);
  ierr = FDPDEGetDM(fd,&dm); CHKERRQ(ierr);

  // Create DM/vec for strain rates, yield, deviatoric/volumetric stress, and stress_old
  ierr = DMStagCreateCompatibleDMStag(dm,4,0,4,0,&usr->dmeps); CHKERRQ(ierr);
  ierr = DMSetUp(usr->dmeps); CHKERRQ(ierr);
  ierr = DMStagSetUniformCoordinatesProduct(usr->dmeps,xmin,xmax,zmin,zmax,0.0,0.0);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(usr->dmeps,&usr->xeps); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(usr->dmeps,&usr->xyield); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(usr->dmeps,&usr->xtau); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(usr->dmeps,&usr->xDP); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(usr->dmeps,&usr->xtau_old); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(usr->dmeps,&usr->xDP_old); CHKERRQ(ierr);

  // Create DM/vec for the velocity field
  usr->dmPV = dm;

  // Create DM/vec for the phase field
  ierr = DMStagCreateCompatibleDMStag(dm,1,1,1,0,&usr->dmf); CHKERRQ(ierr);
  ierr = DMSetUp(usr->dmf); CHKERRQ(ierr);
  ierr = DMStagSetUniformCoordinatesProduct(usr->dmf,xmin,xmax,zmin,zmax,0.0,0.0);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(usr->dmf, &usr->f); CHKERRQ(ierr);
  ierr = VecDuplicate(usr->f, &usr->fprev); CHKERRQ(ierr);
  ierr = VecDuplicate(usr->f,&usr->dfx);CHKERRQ(ierr);
  ierr = VecDuplicate(usr->f,&usr->dfz);CHKERRQ(ierr);
  ierr = VecDuplicate(usr->f,&usr->volf);CHKERRQ(ierr);

  // short names for DM and Vecs of the phase field
  dmf   = usr->dmf;
  f     = usr->f;
  fprev = usr->fprev;

  // Create a vector to store u, p in userdata
  ierr = FDPDEGetSolution(fd,&x);CHKERRQ(ierr);
  ierr = VecDuplicate(x, &usr->xVel);CHKERRQ(ierr);

  // Create DM/vec for tau_old and DP_old, Initialise the two Vecs as zeros.
  ierr = VecZeroEntries(usr->xtau_old); CHKERRQ(ierr);
  ierr = VecZeroEntries(usr->xDP_old); CHKERRQ(ierr);

  // Set coefficients and BC evaluation function
  ierr = FDPDESetFunctionBCList(fd,FormBCList,bc_description,usr); CHKERRQ(ierr);
  ierr = FDPDESetFunctionCoefficient(fd,FormCoefficient,coeff_description,usr); CHKERRQ(ierr);
  ierr = FDPDEView(fd); CHKERRQ(ierr);


  // Initialise the phase field
  ierr = SetInitialField(dmf,f,usr);CHKERRQ(ierr);
  ierr = VecCopy(f, fprev);CHKERRQ(ierr);
  //interpolate phase values on the face and edges before FDPDE solver
    ierr = UpdateCornerF(dmf, f, usr); CHKERRQ(ierr);
    ierr = UpdateVolFrac(dmf, f, usr); CHKERRQ(ierr);
    

#if 0
  { // initialise the phase field

    while (istep<50) {
    
      //one step forward to get f at the next step (extract velocity data within it)
            ierr = UpdateDF(dmf, fprev, usr); CHKERRQ(ierr);
      ierr = ExplicitStep(dmf, fprev, f, usr->par->dt, usr); CHKERRQ(ierr);

      // Phasefield: copy f to prev
      ierr = VecCopy(f,fprev); CHKERRQ(ierr);

      istep++;
    }

    istep = 0;

    //interpolate phase values on the face and edges before FDPDE solver
    ierr = UpdateCornerF(dmf, f, usr); CHKERRQ(ierr);
    ierr = UpdateVolFrac(dmf, f, usr); CHKERRQ(ierr);


  }
#endif

#if 1
  {  
  // Create initial guess with a linear viscous
  usr->par->plasticity = PETSC_FALSE; 
  PetscPrintf(PETSC_COMM_WORLD,"\n# INITIAL GUESS #\n");
  ierr = FDPDESolve(fd,NULL);CHKERRQ(ierr);
  ierr = FDPDEGetSolution(fd,&x);CHKERRQ(ierr);

  ierr = VecCopy(x, usr->xVel); CHKERRQ(ierr);

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/%s_solution_initial",usr->par->fdir_out,usr->par->fname_out);
  ierr = DMStagViewBinaryPython(dm,x,fout);CHKERRQ(ierr);

  ierr = FDPDEGetSolutionGuess(fd,&xguess); CHKERRQ(ierr); 
  ierr = VecCopy(x,xguess);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&xguess);CHKERRQ(ierr);
  }
#endif
  usr->par->plasticity = PETSC_FALSE; //PETSC_TRUE; //switch off/on plasticity



  //-- initialise the phase field by computing 50 steps
  //ierr = UpdateDF(dmf, fprev, usr); CHKERRQ(ierr);
  //ierr = ExplicitStep(dmf, fprev, f, dt, usr); CHKERRQ(ierr);

  // output - initial state of the phase field
  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/%s_phase_initial",usr->par->fdir_out,usr->par->fname_out);
  ierr = DMStagViewBinaryPython(dmf,f,fout);CHKERRQ(ierr);
  
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
      ierr = UpdateDF(dmf, fprev, usr); CHKERRQ(ierr);
      ierr = ExplicitStep(dmf, fprev, f, usr->par->dt, usr); CHKERRQ(ierr);
    }

    //interpolate phase values on the face and edges before FDPDE solver
    ierr = UpdateCornerF(dmf, f, usr); CHKERRQ(ierr);
    ierr = UpdateVolFrac(dmf, f, usr); CHKERRQ(ierr);
    
    //StokesDarcy Solver
    ierr = FDPDESolve(fd,NULL);CHKERRQ(ierr);
    ierr = FDPDEGetSolution(fd,&x);CHKERRQ(ierr);

    //update x into the usrdata
    ierr = VecCopy(x, usr->xVel);

    

    // Update time
    usr->par->t += usr->par->dt;  // computation start from t = dt

    // 2nd order runge-kutta
#if 1
    {
      
#if 0  //update or notupdate flow field for the second stage
      {
      // update flow field according to the phase field after 1st stage
      ierr = UpdateCornerF(dmf, f, usr); CHKERRQ(ierr);
      ierr = UpdateVolFrac(dmf, f, usr); CHKERRQ(ierr);
      ierr = FDPDESolve(fd,NULL);CHKERRQ(ierr);
      ierr = FDPDEGetSolution(fd,&x);CHKERRQ(ierr);
      ierr = VecCopy(x, usr->xVel);
      }
#endif
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
      usr->par->t -= 0.5*usr->par->dt;
      
      // update dfx and dfz and solve for the second stage
      ierr = UpdateDF(dmf, fprev, usr); CHKERRQ(ierr);
      ierr = ExplicitStep(dmf, fprev, f, usr->par->dt, usr);CHKERRQ(ierr);

      // get hk2 and update the full step
      ierr = VecCopy(f, hk2); CHKERRQ(ierr);
      ierr = VecAXPY(hk2, -1.0, fprev);CHKERRQ(ierr);
      ierr = VecCopy(fprev_bk, fprev); CHKERRQ(ierr);
      ierr = VecCopy(fprev, f); CHKERRQ(ierr);
      ierr = VecAXPY(f, 1.0, hk2);CHKERRQ(ierr);

      // reset time
      usr->par->t += 0.5*usr->par->dt;
      
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
#endif    
    
    PetscPrintf(PETSC_COMM_WORLD,"# TIME: time = %1.12e dt = %1.12e \n",usr->par->t,usr->par->dt);

    // Update xtau_old and xDP_old
    ierr = UpdateStressOld(usr->dmeps,usr);CHKERRQ(ierr);

    // Phasefield: save f and fprev into global vectors
    usr->f     = f;
    usr->fprev = fprev;
    // Phasefield: copy f to prev
    ierr = VecCopy(f,fprev); CHKERRQ(ierr);

    // write before changing time steps
    if (iwrt || istep == 0) {

      iwrt = PETSC_FALSE;
      
      // Output solution to file
      ierr = PetscSNPrintf(fout,sizeof(fout),"%s/%s_solution_ts%1.3d",usr->par->fdir_out,usr->par->fname_out,ickpt);
      ierr = DMStagViewBinaryPython(dm,x,fout);CHKERRQ(ierr);
      /*
      ierr = PetscSNPrintf(fout,sizeof(fout),"%s/%s_strain_ts%1.3d",usr->par->fdir_out,usr->par->fname_out,ickpt);
      ierr = DMStagViewBinaryPython(usr->dmeps,usr->xeps,fout);CHKERRQ(ierr);

      ierr = PetscSNPrintf(fout,sizeof(fout),"%s/%s_stress_ts%1.3d",usr->par->fdir_out,usr->par->fname_out,ickpt);
      ierr = DMStagViewBinaryPython(usr->dmeps,usr->xtau,fout);CHKERRQ(ierr);

      ierr = PetscSNPrintf(fout,sizeof(fout),"%s/%s_stressold_ts%1.3d",usr->par->fdir_out,usr->par->fname_out,ickpt);
      ierr = DMStagViewBinaryPython(usr->dmeps,usr->xtau_old,fout);CHKERRQ(ierr);

      ierr = PetscSNPrintf(fout,sizeof(fout),"%s/%s_dpold_ts%1.3d",usr->par->fdir_out,usr->par->fname_out,ickpt);
      ierr = DMStagViewBinaryPython(usr->dmeps,usr->xDP_old,fout);CHKERRQ(ierr);

      ierr = PetscSNPrintf(fout,sizeof(fout),"%s/%s_yield_ts%1.3d",usr->par->fdir_out,usr->par->fname_out,ickpt);
      ierr = DMStagViewBinaryPython(usr->dmeps,usr->xyield,fout);CHKERRQ(ierr);

      ierr = PetscSNPrintf(fout,sizeof(fout),"%s/%s_residual_ts%1.3d",usr->par->fdir_out,usr->par->fname_out,ickpt);
      ierr = DMStagViewBinaryPython(dm,fd->r,fout);CHKERRQ(ierr);

      ierr = FDPDEGetCoefficient(fd,&dmcoeff,&xcoeff);CHKERRQ(ierr);
      ierr = PetscSNPrintf(fout,sizeof(fout),"%s/%s_coefficient_ts%1.3d",usr->par->fdir_out,usr->par->fname_out,ickpt);
      ierr = DMStagViewBinaryPython(dmcoeff,xcoeff,fout);CHKERRQ(ierr);
      */
      ierr = PetscSNPrintf(fout,sizeof(fout),"%s/%s_phase_ts%1.3d",usr->par->fdir_out,usr->par->fname_out,ickpt);
      ierr = DMStagViewBinaryPython(usr->dmf,usr->f,fout);CHKERRQ(ierr);

    }


    //check max(f) and min(f),
    PetscScalar fmax, fmin;
    ierr = VecMax(f,NULL,&fmax); CHKERRQ(ierr);
    ierr = VecMin(f,NULL,&fmin); CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD, "Phase field: Maximum of f = %1.8f, Minimum of f = %1.8f\n", fmax, fmin); CHKERRQ(ierr);


    //check max(x) and min(x) - both face and center values are compared though, but pressure might be small if gravity <= 1 and eta << 1.
    PetscScalar xxmax, xxmin, dtt, dtgap;
    ierr = VecMax(x,NULL,&xxmax); CHKERRQ(ierr);
    ierr = VecMin(x,NULL,&xxmin); CHKERRQ(ierr);
    usr->par->gamma = PetscMax(PetscAbs(xxmax), PetscAbs(xxmin));
    if (usr->par->gamma < 1e-5) {usr->par->gamma = 1.0;}
    //change dt accordingly
    dtt = 1.0/nx/usr->par->gamma/6.0; //maximum time step allowed for boundedness
    dtgap = tckpt - usr->par->t;   // gap between the current time and the next checkpoint

    if (dtgap <= 0) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL, "dtgap is smaller or equal to zero, the next check points, tckpt, has not been updated properly");
    
    usr->par->dt = PetscMin(dtt, dtgap);

    PetscPrintf(PETSC_COMM_WORLD, "Phase field: Maximum of U = %1.8f, Minimum of U = %1.8f\n", xxmax, xxmin); CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD, "Phase field: gamma = %1.8f\n", usr->par->gamma); CHKERRQ(ierr);

    //check if reaching the check point
    if (dtgap <= dtt) {ickpt++; tckpt += dtck; iwrt = PETSC_TRUE;}

    //clean up
    ierr = VecDestroy(&x);CHKERRQ(ierr);

    // increment timestep
    istep++;

  }

  // Destroy objects
  ierr = VecDestroy(&fprev);CHKERRQ(ierr);
  ierr = VecDestroy(&f);CHKERRQ(ierr);
  ierr = DMDestroy(&dmf); CHKERRQ(ierr);

  ierr = VecDestroy(&usr->xVel);CHKERRQ(ierr);
  ierr = DMDestroy(&usr->dmPV);CHKERRQ(ierr);
  
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xeps);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xtau);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xDP);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xyield);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xtau_old);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xDP_old);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = DMDestroy(&usr->dmeps);CHKERRQ(ierr);
  ierr = FDPDEDestroy(&fd);CHKERRQ(ierr);

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
  ierr = PetscBagRegisterInt(bag, &par->nx, 5, "nx", "Element count in the x-dir"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->nz, 5, "nz", "Element count in the z-dir"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->xmin, 0.0, "xmin", "Start coordinate of domain in x-dir"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->zmin, 0.0, "zmin", "Start coordinate of domain in z-dir"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->L, 1.0, "L", "Length of domain in x-dir"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->H, 1.0, "H", "Height of domain in z-dir"); CHKERRQ(ierr);

  // Physical and material parameters
  ierr = PetscBagRegisterScalar(bag, &par->Fd, 0.0, "Fd", "Non-dimensional gravity of the bottom layer"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->Fu, 0.0, "Fu", "Non-dimensional gravity of the up layer"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->R, 0.0, "R", "Ratio of the compaction length scale to the global one, R = ((K0*eta_ref/mu)^1/2)/L"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->phi_0, 0.01, "phi_0", "Reference porosity"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->n, 2.0, "n", "Porosity exponent"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->lambda, 0.0, "lambda", "Exponential melt weakening factor"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->z_in, 0.5, "z_in", "Position of the sharp interface"); CHKERRQ(ierr);
  // Viscosity
  ierr = PetscBagRegisterScalar(bag, &par->eta_u, 1.0, "eta_u", "Viscosity of the upper layer"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->eta_d, 1.0, "eta_d", "Viscosity of the bottom layer"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->lam_v, 1.0e-1, "lam_v", "Factors for intrinsic visocisty, lam_v = eta/zeta"); CHKERRQ(ierr);
  // Plasticity
  ierr = PetscBagRegisterScalar(bag, &par->C_u, 1e40, "C_u", "Cohesion (up)"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->C_d, 1e40, "C_d", "Cohesion (bottom)"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->lam_p, 1.0, "lam_p", "Multiplier for the compaction failure criteria, YC = lam_p*C"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->etamin, 0.0, "etamin", "Cutoff min value of eta"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->nh, 1.0, "nh", "Power for the harmonic plasticity"); CHKERRQ(ierr);
  // Elasticity
  ierr = PetscBagRegisterScalar(bag, &par->G, 1e40, "G", "Shear elastic modulus"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->Z, 1e40, "Z", "Reference poro-elastic modulus"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->q, -0.5, "q", "Exponent of the porosity-dependent relation of poro-elastic modulus"); CHKERRQ(ierr);

  // Time steps
  ierr = PetscBagRegisterScalar(bag, &par->dt, 0.01, "dt", "The size of time step"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->tstep, 11, "tstep", "The maximum time steps"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->tout,5, "tout", "Output every tout time step"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->dtck, 0.1, "dtck", "The size between two check points in time"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->maxckpt, 1, "maxckpt", "Maximum number of check points"); CHKERRQ(ierr);

  ierr = PetscBagRegisterScalar(bag, &par->P0,0, "P0", "Pinned value for pressure"); CHKERRQ(ierr);

  // Parameters for the phase field method
  ierr = PetscBagRegisterScalar(bag, &par->eps, 0.2, "eps", "epsilon in the kernel function"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->gamma, 1.0, "gamma", "gamma in the phase field method"); CHKERRQ(ierr);
  //ierr = PetscBagRegisterBool(bag, &par->diffuse, PETSC_FALSE, "diffuse", "parameters varies smoothly in the diffusing interface"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->vfopt, 0, "vfopt", "vfopt = 0,1,2,3"); CHKERRQ(ierr);

  // Parameters for the purbation at the interface
  ierr = PetscBagRegisterScalar(bag, &par->Delta, 0.1, "Delta", "amplitude of the perturbation"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->wn, 2, "wn", "wavenumber of the perturbation"); CHKERRQ(ierr);

  // Reference compaction viscosity
  par->zeta_u = par->eta_u/par->lam_v;
  par->zeta_d = par->eta_d/par->lam_v;
  
  par->plasticity = PETSC_FALSE;//PETSC_TRUE;

  // other variables
  par->t = 0.0;

  // Input/output 
  ierr = PetscBagRegisterString(bag,&par->fname_out,FNAME_LENGTH,"out_solution","output_file","Name for output file, set with: -output_file <filename>"); CHKERRQ(ierr);
  ierr = PetscBagRegisterString(bag,&par->fdir_out,FNAME_LENGTH,"./","output_dir","Name for output directory, set with: -output_dir <dirname>"); CHKERRQ(ierr);

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
  PetscPrintf(usr->comm,"# Test_stokesdarcy2field_phasefield: %s \n",&(date[0]));
  PetscPrintf(usr->comm,"# --------------------------------------- #\n");
  PetscPrintf(usr->comm,"# PETSc options: %s \n",opts);
  PetscPrintf(usr->comm,"# --------------------------------------- #\n");

  // Print usr bag
  ierr = PetscBagView(usr->bag,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
  PetscPrintf(usr->comm,"# --------------------------------------- #\n");

  // Free memory
  ierr = PetscFree(opts); CHKERRQ(ierr);

  PetscFunctionReturn(0);
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
  PetscErrorCode ierr;

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
  ierr = DMGetLocalVector(usr->dmf, &flocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmf, usr->f, INSERT_VALUES, flocal); CHKERRQ(ierr);

  // volume fraction
  ierr = DMGetLocalVector(usr->dmf, &volflocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmf, usr->volf, INSERT_VALUES, volflocal); CHKERRQ(ierr);

  // Strain rates
  ierr = UpdateStrainRates(dm,x,usr); CHKERRQ(ierr);
  ierr = DMGetLocalVector(usr->dmeps, &xepslocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmeps, usr->xeps, INSERT_VALUES, xepslocal); CHKERRQ(ierr);

  
  // Stress_old
  ierr = DMGetLocalVector(usr->dmeps, &toldlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmeps, usr->xtau_old, INSERT_VALUES, toldlocal); CHKERRQ(ierr);
  ierr = DMGetLocalVector(usr->dmeps, &poldlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmeps, usr->xDP_old, INSERT_VALUES, poldlocal); CHKERRQ(ierr);

  // Local vectors
  ierr = DMCreateLocalVector (usr->dmeps,&xtaulocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(usr->dmeps,xtaulocal,&xxs); CHKERRQ(ierr);

  ierr = DMCreateLocalVector (usr->dmeps,&xDPlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(usr->dmeps,xDPlocal,&xxp); CHKERRQ(ierr);

  ierr = DMCreateLocalVector (usr->dmeps,&xyieldlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(usr->dmeps,xyieldlocal,&xxy); CHKERRQ(ierr);

  // Get domain corners
  ierr = DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = DMStagGetGlobalSizes(dmcoeff,&Nx,&Nz,NULL);CHKERRQ(ierr);

  // Get dm coordinates array
  ierr = DMStagGetProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmcoeff,LEFT,&iprev);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmcoeff,RIGHT,&inext);CHKERRQ(ierr); 
  ierr = DMStagGetProductCoordinateLocationSlot(dmcoeff,ELEMENT,&icenter);CHKERRQ(ierr);

  // Create coefficient local vector
  ierr = DMCreateLocalVector(dmcoeff, &coefflocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dmcoeff, coefflocal, &c); CHKERRQ(ierr);

  // Get the cell sizes
  PetscScalar *dx, *dz;
  ierr = DMStagCellSizeLocal_2d(dm, &nx, &nz, &dx, &dz); CHKERRQ(ierr);

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
        ierr = DMStagVecGetValuesStencil(usr->dmf,flocal,1,&point,&ff); CHKERRQ(ierr);
        ierr = DMStagVecGetValuesStencil(usr->dmf,volflocal,1,&point,&volf); CHKERRQ(ierr);

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
        ierr = DMStagVecGetValuesStencil(usr->dmeps,xepslocal,1,&point,&exx); CHKERRQ(ierr);
        ierr = DMStagVecGetValuesStencil(usr->dmeps,toldlocal,1,&point,&told_xx_e); CHKERRQ(ierr);
        ierr = DMStagVecGetValuesStencil(usr->dmeps,poldlocal,1,&point,&pold); CHKERRQ(ierr);
        point.c = 1;
        ierr = DMStagVecGetValuesStencil(usr->dmeps,xepslocal,1,&point,&ezz); CHKERRQ(ierr);
        ierr = DMStagVecGetValuesStencil(usr->dmeps,toldlocal,1,&point,&told_zz_e); CHKERRQ(ierr);
        point.c = 2;
        ierr = DMStagVecGetValuesStencil(usr->dmeps,xepslocal,1,&point,&exz); CHKERRQ(ierr);
        ierr = DMStagVecGetValuesStencil(usr->dmeps,toldlocal,1,&point,&told_xz_e); CHKERRQ(ierr);
        point.c = 3;
        ierr = DMStagVecGetValuesStencil(usr->dmeps,xepslocal,1,&point,&epsII); CHKERRQ(ierr);
        ierr = DMStagVecGetValuesStencil(usr->dmeps,toldlocal,1,&point,&told_II_e); CHKERRQ(ierr);

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

        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
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
        ierr = DMStagGetLocationSlot(usr->dmeps,ELEMENT,0,&idx); CHKERRQ(ierr); xxs[j][i][idx] = txx; xxy[j][i][idx] = Y; xxp[j][i][idx] = DeltaP;
        ierr = DMStagGetLocationSlot(usr->dmeps,ELEMENT,1,&idx); CHKERRQ(ierr); xxs[j][i][idx] = tzz;
        ierr = DMStagGetLocationSlot(usr->dmeps,ELEMENT,2,&idx); CHKERRQ(ierr); xxs[j][i][idx] = txz;
        ierr = DMStagGetLocationSlot(usr->dmeps,ELEMENT,3,&idx); CHKERRQ(ierr); xxs[j][i][idx] = tauII;
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
        ierr = DMStagVecGetValuesStencil(usr->dmf,flocal,4,point,ff); CHKERRQ(ierr);
        ierr = DMStagVecGetValuesStencil(usr->dmf,volflocal,4,point,volf); CHKERRQ(ierr);
        
        
        ierr = DMStagVecGetValuesStencil(usr->dmeps,xepslocal,4,point,exx); CHKERRQ(ierr);
        ierr = DMStagVecGetValuesStencil(usr->dmeps,toldlocal,4,point,told_xx); CHKERRQ(ierr);
                
        for (ii = 0; ii < 4; ii++) {point[ii].c = 1;}
        ierr = DMStagVecGetValuesStencil(usr->dmeps,xepslocal,4,point,ezz); CHKERRQ(ierr);
        ierr = DMStagVecGetValuesStencil(usr->dmeps,toldlocal,4,point,told_zz); CHKERRQ(ierr);

        for (ii = 0; ii < 4; ii++) {point[ii].c = 2;}
        ierr = DMStagVecGetValuesStencil(usr->dmeps,xepslocal,4,point,exz); CHKERRQ(ierr);
        ierr = DMStagVecGetValuesStencil(usr->dmeps,toldlocal,4,point,told_xz); CHKERRQ(ierr);

        // second invariant of strain rate
        for (ii = 0; ii < 4; ii++) {point[ii].c = 3;}
        ierr = DMStagVecGetValuesStencil(usr->dmeps,xepslocal,4,point,epsII); CHKERRQ(ierr);
        ierr = DMStagVecGetValuesStencil(usr->dmeps,toldlocal,4,point,told_II); CHKERRQ(ierr);

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

          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, 0, &idx); CHKERRQ(ierr);
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
        ierr = DMStagGetLocationSlot(usr->dmeps,DOWN_LEFT,0,&idx); CHKERRQ(ierr); xxs[j][i][idx] = txx[0]; xxy[j][i][idx] = Y[0];
        ierr = DMStagGetLocationSlot(usr->dmeps,DOWN_LEFT,1,&idx); CHKERRQ(ierr); xxs[j][i][idx] = tzz[0];
        ierr = DMStagGetLocationSlot(usr->dmeps,DOWN_LEFT,2,&idx); CHKERRQ(ierr); xxs[j][i][idx] = txz[0];
        ierr = DMStagGetLocationSlot(usr->dmeps,DOWN_LEFT,3,&idx); CHKERRQ(ierr); xxs[j][i][idx] = tauII[0];

        ierr = DMStagGetLocationSlot(usr->dmeps,DOWN_RIGHT,0,&idx); CHKERRQ(ierr); xxs[j][i][idx] = txx[1]; xxy[j][i][idx] = Y[1];
        ierr = DMStagGetLocationSlot(usr->dmeps,DOWN_RIGHT,1,&idx); CHKERRQ(ierr); xxs[j][i][idx] = tzz[1];
        ierr = DMStagGetLocationSlot(usr->dmeps,DOWN_RIGHT,2,&idx); CHKERRQ(ierr); xxs[j][i][idx] = txz[1];
        ierr = DMStagGetLocationSlot(usr->dmeps,DOWN_RIGHT,3,&idx); CHKERRQ(ierr); xxs[j][i][idx] = tauII[1];

        ierr = DMStagGetLocationSlot(usr->dmeps,UP_LEFT,0,&idx); CHKERRQ(ierr); xxs[j][i][idx] = txx[2]; xxy[j][i][idx] = Y[2];
        ierr = DMStagGetLocationSlot(usr->dmeps,UP_LEFT,1,&idx); CHKERRQ(ierr); xxs[j][i][idx] = tzz[2];
        ierr = DMStagGetLocationSlot(usr->dmeps,UP_LEFT,2,&idx); CHKERRQ(ierr); xxs[j][i][idx] = txz[2];
        ierr = DMStagGetLocationSlot(usr->dmeps,UP_LEFT,3,&idx); CHKERRQ(ierr); xxs[j][i][idx] = tauII[2];

        ierr = DMStagGetLocationSlot(usr->dmeps,UP_RIGHT,0,&idx); CHKERRQ(ierr); xxs[j][i][idx] = txx[3]; xxy[j][i][idx] = Y[3];
        ierr = DMStagGetLocationSlot(usr->dmeps,UP_RIGHT,1,&idx); CHKERRQ(ierr); xxs[j][i][idx] = tzz[3];
        ierr = DMStagGetLocationSlot(usr->dmeps,UP_RIGHT,2,&idx); CHKERRQ(ierr); xxs[j][i][idx] = txz[3];
        ierr = DMStagGetLocationSlot(usr->dmeps,UP_RIGHT,3,&idx); CHKERRQ(ierr); xxs[j][i][idx] = tauII[3];
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
        ierr = DMStagVecGetValuesStencil(usr->dmf,flocal,4,point,ff); CHKERRQ(ierr);
        ierr = DMStagVecGetValuesStencil(usr->dmf,volflocal,4,point,volf); CHKERRQ(ierr);

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
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] += rhs[ii];

          //                    PetscPrintf(PETSC_COMM_WORLD, "B = %g \n", c[j][i][idx]); 
          
        }
      }

      { // C = 0 (center, c=0)
        DMStagStencil point;
        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 0;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = 0.0;
      }

      { // D1 = zeta - 2/3*A (center, c=2)
        DMStagStencil point;
        PetscInt      idxA;
        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 2;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 1;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idxA); CHKERRQ(ierr);
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
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
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
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = 0.0;
        }
        //  nonzero if including the gravity
        for (ii = 2; ii < 4; ii++) {
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);

          // collect phase values for the four corners
          point[ii].c = 0;
          ierr = DMStagVecGetValuesStencil(usr->dmf,flocal,1,&point[ii],&ff); CHKERRQ(ierr);
          ierr = DMStagVecGetValuesStencil(usr->dmf,volflocal,1,&point[ii],&volf); CHKERRQ(ierr);

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
  ierr = PetscFree(dx);CHKERRQ(ierr);
  ierr = PetscFree(dz);CHKERRQ(ierr);  

  // Restore arrays, local vectors
  ierr = DMStagRestoreProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL);CHKERRQ(ierr);

  ierr = DMStagVecRestoreArray(dmcoeff,coefflocal,&c);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  ierr = VecDestroy(&coefflocal); CHKERRQ(ierr);

  // Restore and map local to global
  ierr = DMStagVecRestoreArray(usr->dmeps,xtaulocal,&xxs); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(usr->dmeps,xtaulocal,INSERT_VALUES,usr->xtau); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (usr->dmeps,xtaulocal,INSERT_VALUES,usr->xtau); CHKERRQ(ierr);
  ierr = VecDestroy(&xtaulocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArray(usr->dmeps,xDPlocal,&xxp); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(usr->dmeps,xDPlocal,INSERT_VALUES,usr->xDP); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (usr->dmeps,xDPlocal,INSERT_VALUES,usr->xDP); CHKERRQ(ierr);
  ierr = VecDestroy(&xDPlocal); CHKERRQ(ierr);
  
  ierr = DMStagVecRestoreArray(usr->dmeps,xyieldlocal,&xxy); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(usr->dmeps,xyieldlocal,INSERT_VALUES,usr->xyield); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (usr->dmeps,xyieldlocal,INSERT_VALUES,usr->xyield); CHKERRQ(ierr);
  ierr = VecDestroy(&xyieldlocal); CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(usr->dmeps,&xepslocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmeps,&toldlocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmeps,&poldlocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmf,  &flocal);    CHKERRQ(ierr);

  PetscFunctionReturn(0);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;

  dmeps = usr->dmeps;
  xeps  = usr->xeps;

  // Local vectors
  ierr = DMCreateLocalVector (dmeps,&xepslocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dmeps,xepslocal,&xx); CHKERRQ(ierr);

  ierr = DMGetLocalVector(dm,&xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm,x,INSERT_VALUES,xlocal); CHKERRQ(ierr);

  // Get domain corners
  ierr = DMStagGetGlobalSizes(dmeps, &Nx, &Nz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dmeps, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  
  // Get dm coordinates array
  ierr = DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter);CHKERRQ(ierr); 
  ierr = DMStagGetProductCoordinateLocationSlot(dm,LEFT,&iprev);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,RIGHT,&inext);CHKERRQ(ierr); 

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      DMStagStencil  pointC, pointN[4];
      PetscScalar    epsIIc, exxc, ezzc, exzc, epsIIn[4], exxn[4], ezzn[4], exzn[4];

      // Strain rates: center
      pointC.i = i; pointC.j = j; pointC.loc = ELEMENT; pointC.c = 0;
      ierr = DMStagGetPointStrainRates(dm,xlocal,1,&pointC,&epsIIc,&exxc,&ezzc,&exzc); CHKERRQ(ierr);

      ierr = DMStagGetLocationSlot(dmeps,ELEMENT,0,&idx); CHKERRQ(ierr); xx[j][i][idx] = exxc;
      ierr = DMStagGetLocationSlot(dmeps,ELEMENT,1,&idx); CHKERRQ(ierr); xx[j][i][idx] = ezzc;
      ierr = DMStagGetLocationSlot(dmeps,ELEMENT,2,&idx); CHKERRQ(ierr); xx[j][i][idx] = exzc;
      ierr = DMStagGetLocationSlot(dmeps,ELEMENT,3,&idx); CHKERRQ(ierr); xx[j][i][idx] = epsIIc;

      // Strain rates: corner
      pointN[0].i = i; pointN[0].j = j; pointN[0].loc = DOWN_LEFT;  pointN[0].c = 0;
      pointN[1].i = i; pointN[1].j = j; pointN[1].loc = DOWN_RIGHT; pointN[1].c = 0;
      pointN[2].i = i; pointN[2].j = j; pointN[2].loc = UP_LEFT;    pointN[2].c = 0;
      pointN[3].i = i; pointN[3].j = j; pointN[3].loc = UP_RIGHT;   pointN[3].c = 0;
      ierr = DMStagGetPointStrainRates(dm,xlocal,4,pointN,epsIIn,exxn,ezzn,exzn); CHKERRQ(ierr);
      
      if (i==0) { // boundaries
        pointC.i = i; pointC.j = j; pointC.loc = ELEMENT; pointC.c = 0;
        ierr = DMStagGetPointStrainRates(dm,xlocal,1,&pointC,&epsIIc,&exxc,&ezzc,&exzc); CHKERRQ(ierr);
        ezzn[0] = ezzc;
        exxn[0] = exxc;
      }

      if (i==Nx-1) { // boundaries
        pointC.i = i; pointC.j = j; pointC.loc = ELEMENT; pointC.c = 0;
        ierr = DMStagGetPointStrainRates(dm,xlocal,1,&pointC,&epsIIc,&exxc,&ezzc,&exzc); CHKERRQ(ierr);
        ezzn[1] = ezzc;
        exxn[1] = exxc;
      }

      if (j==0) { // boundaries
        pointC.i = i; pointC.j = j; pointC.loc = ELEMENT; pointC.c = 0;
        ierr = DMStagGetPointStrainRates(dm,xlocal,1,&pointC,&epsIIc,&exxc,&ezzc,&exzc); CHKERRQ(ierr);
        exxn[0] = exxc;
        ezzn[0] = ezzc;
      }

      if (j==Nz-1) { // boundaries
        pointC.i = i; pointC.j = j; pointC.loc = ELEMENT; pointC.c = 0;
        ierr = DMStagGetPointStrainRates(dm,xlocal,1,&pointC,&epsIIc,&exxc,&ezzc,&exzc); CHKERRQ(ierr);
        exxn[2] = exxc;
        ezzn[2] = ezzc;
      }

      if ((i==Nx-1) && (j==Nz-1)) { // boundaries
        pointC.i = i; pointC.j = j; pointC.loc = ELEMENT; pointC.c = 0;
        ierr = DMStagGetPointStrainRates(dm,xlocal,1,&pointC,&epsIIc,&exxc,&ezzc,&exzc); CHKERRQ(ierr);
        exxn[3] = exxc;
        ezzn[3] = ezzc;
      }
      
      if ((i==0) || (i==Nx-1) || (j==0) || (j==Nz-1)) { // boundaries
        for (ii = 0; ii < 4; ii++) {
          epsIIn[ii] = PetscPowScalar(0.5*(exxn[ii]*exxn[ii] + ezzn[ii]*ezzn[ii] + 2.0*exzn[ii]*exzn[ii]),0.5);
        }
      }

      ierr = DMStagGetLocationSlot(dmeps,DOWN_LEFT,0,&idx); CHKERRQ(ierr); xx[j][i][idx] = exxn[0];
      ierr = DMStagGetLocationSlot(dmeps,DOWN_LEFT,1,&idx); CHKERRQ(ierr); xx[j][i][idx] = ezzn[0];
      ierr = DMStagGetLocationSlot(dmeps,DOWN_LEFT,2,&idx); CHKERRQ(ierr); xx[j][i][idx] = exzn[0];
      ierr = DMStagGetLocationSlot(dmeps,DOWN_LEFT,3,&idx); CHKERRQ(ierr); xx[j][i][idx] = epsIIn[0];

      ierr = DMStagGetLocationSlot(dmeps,DOWN_RIGHT,0,&idx); CHKERRQ(ierr); xx[j][i][idx] = exxn[1];
      ierr = DMStagGetLocationSlot(dmeps,DOWN_RIGHT,1,&idx); CHKERRQ(ierr); xx[j][i][idx] = ezzn[1];
      ierr = DMStagGetLocationSlot(dmeps,DOWN_RIGHT,2,&idx); CHKERRQ(ierr); xx[j][i][idx] = exzn[1];
      ierr = DMStagGetLocationSlot(dmeps,DOWN_RIGHT,3,&idx); CHKERRQ(ierr); xx[j][i][idx] = epsIIn[1];

      ierr = DMStagGetLocationSlot(dmeps,UP_LEFT,0,&idx); CHKERRQ(ierr); xx[j][i][idx] = exxn[2];
      ierr = DMStagGetLocationSlot(dmeps,UP_LEFT,1,&idx); CHKERRQ(ierr); xx[j][i][idx] = ezzn[2];
      ierr = DMStagGetLocationSlot(dmeps,UP_LEFT,2,&idx); CHKERRQ(ierr); xx[j][i][idx] = exzn[2];
      ierr = DMStagGetLocationSlot(dmeps,UP_LEFT,3,&idx); CHKERRQ(ierr); xx[j][i][idx] = epsIIn[2];

      ierr = DMStagGetLocationSlot(dmeps,UP_RIGHT,0,&idx); CHKERRQ(ierr); xx[j][i][idx] = exxn[3];
      ierr = DMStagGetLocationSlot(dmeps,UP_RIGHT,1,&idx); CHKERRQ(ierr); xx[j][i][idx] = ezzn[3];
      ierr = DMStagGetLocationSlot(dmeps,UP_RIGHT,2,&idx); CHKERRQ(ierr); xx[j][i][idx] = exzn[3];
      ierr = DMStagGetLocationSlot(dmeps,UP_RIGHT,3,&idx); CHKERRQ(ierr); xx[j][i][idx] = epsIIn[3];
    }
  }

  // Restore arrays
  ierr = DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);

  // Restore and map local to global
  ierr = DMStagVecRestoreArray(dmeps,xepslocal,&xx); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dmeps,xepslocal,INSERT_VALUES,xeps); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dmeps,xepslocal,INSERT_VALUES,xeps); CHKERRQ(ierr);
  ierr = VecDestroy(&xepslocal); CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(dm, &xlocal ); CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
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
  PetscErrorCode ierr;
  
  PetscFunctionBegin;

  // LEFT Boundary - Vx
  ierr = DMStagBCListGetValues(bclist,'w','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET; //BC_NEUMANN;//BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // LEFT Boundary - Vz
  ierr = DMStagBCListGetValues(bclist,'w','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // LEFT Boundary - P
  /*
  ierr = DMStagBCListGetValues(bclist,'w','o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET_STAG;//BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  */
  
  // RIGHT Boundary - Vx
  ierr = DMStagBCListGetValues(bclist,'e','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;//BC_NEUMANN; //BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // RIGHT Boundary - Vz
  ierr = DMStagBCListGetValues(bclist,'e','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // RIGHT Boundary - P
  /*
  ierr = DMStagBCListGetValues(bclist,'e','o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET_STAG;//BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  */
    
  // DOWN Boundary - Vx
  ierr = DMStagBCListGetValues(bclist,'s','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;//BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN Boundary - Vz
  ierr = DMStagBCListGetValues(bclist,'s','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;//BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN Boundary - P
  /*
  ierr = DMStagBCListGetValues(bclist,'s','o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET_STAG;//BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  */
  
  // UP Boundary - Vx
  ierr = DMStagBCListGetValues(bclist,'n','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;//BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP Boundary - Vz
  ierr = DMStagBCListGetValues(bclist,'n','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;//BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP Boundary - P
  //ierr = DMStagBCListGetValues(bclist,'n','o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  //for (k=0; k<n_bc; k++) {
  //  value_bc[k] = 0.0;
    //  type_bc[k] = BC_DIRICHLET;//BC_DIRICHLET_STAG;//BC_NEUMANN;
    //}
  //ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  ierr = DMStagBCListGetValues(bclist,'n','o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<1; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET_STAG;//BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
 
  
  PetscFunctionReturn(0);
}



// ---------------------------------------
// UpdateStressOld
// ---------------------------------------
PetscErrorCode UpdateStressOld(DM dmeps, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  Vec            xtaulocal, xDPlocal, tauold_local,DPold_local;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  // Get local vectors from dmeps
  ierr = DMGetLocalVector(usr->dmeps, &xtaulocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmeps, usr->xtau, INSERT_VALUES, xtaulocal); CHKERRQ(ierr);

  ierr = DMGetLocalVector(usr->dmeps, &xDPlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmeps, usr->xDP, INSERT_VALUES, xDPlocal); CHKERRQ(ierr);

  // Create local vectors for the stress_old terms
  ierr = DMCreateLocalVector (usr->dmeps,&tauold_local); CHKERRQ(ierr);
  ierr = DMCreateLocalVector (usr->dmeps,&DPold_local); CHKERRQ(ierr);
  ierr = VecCopy(xtaulocal, tauold_local); CHKERRQ(ierr);
  ierr = VecCopy(xDPlocal, DPold_local); CHKERRQ(ierr);

  // create array from tauold_local and add the rotation terms
  // No rotation terms for now

  // Restore and map local to global
  ierr = DMLocalToGlobalBegin(usr->dmeps,tauold_local,INSERT_VALUES,usr->xtau_old); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (usr->dmeps,tauold_local,INSERT_VALUES,usr->xtau_old); CHKERRQ(ierr);
  ierr = VecDestroy(&tauold_local); CHKERRQ(ierr);

  ierr = DMLocalToGlobalBegin(usr->dmeps,DPold_local,INSERT_VALUES,usr->xDP_old); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (usr->dmeps,DPold_local,INSERT_VALUES,usr->xDP_old); CHKERRQ(ierr);
  ierr = VecDestroy(&DPold_local); CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(usr->dmeps,&xtaulocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmeps,&xDPlocal); CHKERRQ(ierr);

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
  PetscInt       i,j, sx, sz, nx, nz, icenter;//, wn;
  PetscScalar    eps, L, Delta;
  PetscScalar    ***xx, **coordx, **coordz;

  PetscErrorCode ierr;
  PetscFunctionBegin;

  // some useful parameters
  eps = usr->par->eps;
  L   = usr->par->L;
  Delta = usr->par->Delta;
  // wn  = usr->par->wn;

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
PetscErrorCode UpdateDF(DM dm, Vec x, void *ctx)
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
  PetscScalar    gamma, eps;
  PetscScalar    **coordx,**coordz;
  PetscScalar    ***xx,***xxp;
  Vec            dfx, dfz, dfxlocal, dfzlocal, xlocal, xplocal;
  Vec            xVellocal;
  PetscErrorCode ierr;
  
  PetscFunctionBeginUser;

  // User Parameter
  gamma = usr->par->gamma;
  eps = usr->par->eps;

  dfx = usr->dfx;
  dfz = usr->dfz;

  // create a dmPV and xPV in usrdata, copy data in and extract them here
  ierr = DMGetLocalVector(usr->dmPV, &xVellocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmPV, usr->xVel, INSERT_VALUES, xVellocal); CHKERRQ(ierr);

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
  //if (sz==0) {sz++; nz--;}
  //if (sz+nz==Nz) {nz--;}
  //if (sx==0) {sx++; nx--;}
  //if (sx+nx==Nx) {nx--;}

  // Get the cell sizes
  PetscScalar *dx, *dz;
  ierr = DMStagCellSizeLocal_2d(dm, &nx, &nz, &dx, &dz); CHKERRQ(ierr);

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

        ierr = DMStagVecGetValuesStencil(usr->dmPV,xVellocal,4,pf,vf); CHKERRQ(ierr);

        //        PetscPrintf(PETSC_COMM_WORLD, "vf check: %g, %g, %g, %g\n", vf[0], vf[1], vf[2], vf[3]);
        
        // central difference method
        fval -= 0.5*(vf[1]*(fe[2]+fe[0]) - vf[0]*(fe[1]+fe[0]))/dx[ix] + 0.5*(vf[3]*(fe[4]+fe[0]) - vf[2]*(fe[3]+fe[0]))/dz[iz];
        
      }

      xx[j][i][idx] = xxp[j][i][idx] + dt*fval;
    }
  }
                                
  // reset sx, sz, nx, nz
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
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
  ierr = DMRestoreLocalVector(usr->dmPV, &xVellocal); CHKERRQ(ierr);

  PetscFunctionReturn(0);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Local vectors
  ierr = DMGetLocalVector(dm,&xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm,x,INSERT_VALUES,xlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm,xlocal,&xx); CHKERRQ(ierr);
  
  // Get domain corners
  ierr = DMStagGetGlobalSizes(dm, &Nx, &Nz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr); 
  
  // Get location slot
  ierr = DMStagGetLocationSlot(dm, ELEMENT,    0, &ic ); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm, DOWN_LEFT,  0, &idl); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm, DOWN_RIGHT, 0, &idr); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm, LEFT,       0, &il ); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm, RIGHT,      0, &ir ); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm, DOWN,       0, &id ); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm, UP,         0, &iu ); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm, UP_LEFT,    0, &iul); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm, UP_RIGHT,   0, &iur); CHKERRQ(ierr);
  

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

      ierr = DMStagVecGetValuesStencil(dm, xlocal, 4, point, fval); CHKERRQ(ierr);

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
  ierr = DMStagVecRestoreArray(dm,xlocal,&xx); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm,xlocal,INSERT_VALUES,x); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,xlocal,INSERT_VALUES,x); CHKERRQ(ierr);
  ierr = VecDestroy(&xlocal); CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;

  eps = usr->par->eps;
  vfopt = usr->par->vfopt;

  // Local vectors
  ierr = DMGetLocalVector(dm,&xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm,x,INSERT_VALUES,xlocal); CHKERRQ(ierr);

  ierr = DMCreateLocalVector(dm, &vflocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm, vflocal, &vvf); CHKERRQ(ierr);

  // Get domain corners
  ierr = DMStagGetGlobalSizes(dm, &Nx, &Nz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr); 
  
  // Get location slot
  ierr = DMStagGetLocationSlot(dm, ELEMENT,    0, &ic ); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm, LEFT,       0, &il ); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm, RIGHT,      0, &ir ); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm, DOWN,       0, &id ); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm, UP,         0, &iu ); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm, DOWN_LEFT,  0, &idl ); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm, UP_LEFT,    0, &iul ); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm, DOWN_RIGHT, 0, &idr ); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm, UP_RIGHT,   0, &iur ); CHKERRQ(ierr);

  // Get dm coordinates array
  ierr = DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,LEFT   ,&iprev  );CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,RIGHT  ,&inext  );CHKERRQ(ierr);
  

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

      ierr = DMStagVecGetValuesStencil(dm, xlocal, 8, point, ff); CHKERRQ(ierr);

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
  ierr = DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);

  // Restore and map local to global
  ierr = DMStagVecRestoreArray(dm,vflocal,&vvf); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm,vflocal,INSERT_VALUES,usr->volf); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,vflocal,INSERT_VALUES,usr->volf); CHKERRQ(ierr);
  ierr = VecDestroy(&vflocal); CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(dm, &xlocal ); CHKERRQ(ierr);
  
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

  // Print user parameters
  ierr = InputPrintData(usr); CHKERRQ(ierr);

  // Numerical solution using the FD pde object
  ierr = StokesDarcy_Numerical(usr); CHKERRQ(ierr);

  // Destroy objects
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
