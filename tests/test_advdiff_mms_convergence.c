// ---------------------------------------
// (ADVDIFF) Advection-diffusion convergence test using MMS
// run: ./test_advdiff_mms_convergence_ -pc_type lu -pc_factor_mat_solver_type umfpack -pc_factor_mat_ordering_type external -nx 10 -nz 10 -log_view
// python test: ./python/test_advdiff_mms_convergence.py
// sympy: ./mms/mms_advdiff_convergence.py
// ---------------------------------------
static char help[] = "Application to verify the convergence accuracy of ADVDIFF FD-PDE using MMS \n\n";

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
  PetscInt       nx, nz, test;
  PetscInt       ts_scheme,adv_scheme,tout,tstep;
  PetscScalar    L, H, xmin, zmin;
  PetscScalar    t, dt, tmax, dtmax, tinit;
  PetscScalar    Q0, x0, z0, taux, tauz;
  PetscInt       bcleft,bcright,bcdown,bcup;
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
PetscErrorCode Numerical_solution(void*);
PetscErrorCode InputParameters(UsrData**);
PetscErrorCode InputPrintData(UsrData*);
PetscErrorCode FormCoefficient(FDPDE,DM,Vec,DM,Vec,void*);
PetscErrorCode FormBCList(DM,Vec,DMStagBCList,void*);

PetscErrorCode SetInitialQProfile(DM,Vec,void*);
PetscErrorCode SetInitialQCoefficient(DM,Vec,void*);

PetscErrorCode ComputeManufacturedSolution(DM,Vec*,void*);
PetscErrorCode ComputeErrorNorms(DM,Vec,Vec);

// ---------------------------------------
// Some descriptions
// ---------------------------------------
const char coeff_description[] =
"  << ADVDIFF - Coefficients >> \n"
"  A = (manufactured) \n"
"  B = (manufactured) \n"
"  C = -frhs (manufactured) \n"
"  u = [ux, uz] (manufactured) \n";

const char bc_description[] =
"  << ADVDIFF - BCs >> \n"
"  Left/right/down/up: Dirichlet (manufactured) \n";
// ---------------------------------------
// Manufactured solutions
// ---------------------------------------
static PetscScalar get_Q1(PetscScalar x, PetscScalar z, PetscScalar t)
{ PetscScalar result;
  result = sin(2.0*M_PI*z)*cos(2.0*M_PI*x);
  return(result);
}
static PetscScalar get_A1(PetscScalar x, PetscScalar z, PetscScalar t)
{ PetscScalar result;
  result = 0.0;
  return(result);
}
static PetscScalar get_B1(PetscScalar x, PetscScalar z, PetscScalar t)
{ PetscScalar result;
  result = sin(2.0*M_PI*x)*cos(2.0*M_PI*z) + 1.5;
  return(result);
}
static PetscScalar get_ux1(PetscScalar x, PetscScalar z, PetscScalar t)
{ PetscScalar result;
  result = 0.0;
  return(result);
}
static PetscScalar get_uz1(PetscScalar x, PetscScalar z, PetscScalar t)
{ PetscScalar result;
  result = 0.0;
  return(result);
}
static PetscScalar get_frhs1(PetscScalar x, PetscScalar z, PetscScalar t)
{ PetscScalar result;
  result = 8.0*pow(M_PI, 2)*(sin(2.0*M_PI*x)*cos(2.0*M_PI*z) + 1.5)*sin(2.0*M_PI*z)*cos(2.0*M_PI*x) + 8.0*pow(M_PI, 2)*sin(2.0*M_PI*x)*sin(2.0*M_PI*z)*cos(2.0*M_PI*x)*cos(2.0*M_PI*z);
  return(result);
}
static PetscScalar get_Q2(PetscScalar x, PetscScalar z, PetscScalar t)
{ PetscScalar result;
  result = sin(2.0*M_PI*z)*cos(2.0*M_PI*x);
  return(result);
}
static PetscScalar get_A2(PetscScalar x, PetscScalar z, PetscScalar t)
{ PetscScalar result;
  result = sin(2.0*M_PI*x)*cos(2.0*M_PI*z) + 1.5;
  return(result);
}
static PetscScalar get_B2(PetscScalar x, PetscScalar z, PetscScalar t)
{ PetscScalar result;
  result = pow(x, 2) + pow(z, 2) + 1.0;
  return(result);
}
static PetscScalar get_ux2(PetscScalar x, PetscScalar z, PetscScalar t)
{ PetscScalar result;
  result = x + 1.0;
  return(result);
}
static PetscScalar get_uz2(PetscScalar x, PetscScalar z, PetscScalar t)
{ PetscScalar result;
  result = pow(x, 2)*sin(2*M_PI*z);
  return(result);
}
static PetscScalar get_frhs2(PetscScalar x, PetscScalar z, PetscScalar t)
{ PetscScalar result;
  result = 4.0*M_PI*x*sin(2.0*M_PI*x)*sin(2.0*M_PI*z) - 4.0*M_PI*z*cos(2.0*M_PI*x)*cos(2.0*M_PI*z) + (sin(2.0*M_PI*x)*cos(2.0*M_PI*z) + 1.5)*(2.0*M_PI*pow(x, 2)*sin(2*M_PI*z)*cos(2.0*M_PI*x)*cos(2.0*M_PI*z) + 2*M_PI*pow(x, 2)*sin(2.0*M_PI*z)*cos(2.0*M_PI*x)*cos(2*M_PI*z) - 2.0*M_PI*(x + 1.0)*sin(2.0*M_PI*x)*sin(2.0*M_PI*z) + sin(2.0*M_PI*z)*cos(2.0*M_PI*x)) + 8.0*pow(M_PI, 2)*(pow(x, 2) + pow(z, 2) + 1.0)*sin(2.0*M_PI*z)*cos(2.0*M_PI*x);
  return(result);
}
static PetscScalar get_Q3(PetscScalar x, PetscScalar z, PetscScalar t)
{ PetscScalar result;
  result = pow(t, 3)*(pow(x, 2) + pow(z, 2));
  return(result);
}
static PetscScalar get_A3(PetscScalar x, PetscScalar z, PetscScalar t)
{ PetscScalar result;
  result = 1.0;
  return(result);
}
static PetscScalar get_B3(PetscScalar x, PetscScalar z, PetscScalar t)
{ PetscScalar result;
  result = 1.0;
  return(result);
}
static PetscScalar get_ux3(PetscScalar x, PetscScalar z, PetscScalar t)
{ PetscScalar result;
  result = 0.0;
  return(result);
}
static PetscScalar get_uz3(PetscScalar x, PetscScalar z, PetscScalar t)
{ PetscScalar result;
  result = 0.0;
  return(result);
}
static PetscScalar get_frhs3(PetscScalar x, PetscScalar z, PetscScalar t)
{ PetscScalar result;
  result = -4.0*pow(t, 3) + 3.0*pow(t, 2)*(pow(x, 2) + pow(z, 2));
  return(result);
}
static PetscScalar get_Q4(PetscScalar x, PetscScalar z, PetscScalar t)
{ PetscScalar result;
  result = pow(t, 3)*(pow(x, 2) + pow(z, 2));
  return(result);
}
static PetscScalar get_A4(PetscScalar x, PetscScalar z, PetscScalar t)
{ PetscScalar result;
  result = sin(2.0*M_PI*x)*cos(2.0*M_PI*z) + 1.0;
  return(result);
}
static PetscScalar get_B4(PetscScalar x, PetscScalar z, PetscScalar t)
{ PetscScalar result;
  result = 0.0;
  return(result);
}
static PetscScalar get_ux4(PetscScalar x, PetscScalar z, PetscScalar t)
{ PetscScalar result;
  result = x + 1.0;
  return(result);
}
static PetscScalar get_uz4(PetscScalar x, PetscScalar z, PetscScalar t)
{ PetscScalar result;
  result = pow(x, 2)*sin(2*M_PI*z);
  return(result);
}
static PetscScalar get_frhs4(PetscScalar x, PetscScalar z, PetscScalar t)
{ PetscScalar result;
  result = (sin(2.0*M_PI*x)*cos(2.0*M_PI*z) + 1.0)*(2*pow(t, 3)*pow(x, 2)*z*sin(2*M_PI*z) + 2*M_PI*pow(t, 3)*pow(x, 2)*(pow(x, 2) + pow(z, 2))*cos(2*M_PI*z) + 2*pow(t, 3)*x*(x + 1.0) + pow(t, 3)*(pow(x, 2) + pow(z, 2)) + 3*pow(t, 2)*(pow(x, 2) + pow(z, 2)));
  return(result);
}
static PetscScalar get_Q5(PetscScalar x, PetscScalar z, PetscScalar t)
{ PetscScalar result;
  result = sin(2.0*M_PI*z)*cos(2.0*M_PI*x);
  return(result);
}
static PetscScalar get_A5(PetscScalar x, PetscScalar z, PetscScalar t)
{ PetscScalar result;
  result = sin(2.0*M_PI*x)*cos(2.0*M_PI*z) + 1.5;
  return(result);
}
static PetscScalar get_B5(PetscScalar x, PetscScalar z, PetscScalar t)
{ PetscScalar result;
  result = pow(x, 2) + pow(z, 2) + 1.0;
  return(result);
}
static PetscScalar get_ux5(PetscScalar x, PetscScalar z, PetscScalar t)
{ PetscScalar result;
  result = x;
  return(result);
}
static PetscScalar get_uz5(PetscScalar x, PetscScalar z, PetscScalar t)
{ PetscScalar result;
  result = z;
  return(result);
}
static PetscScalar get_frhs5(PetscScalar x, PetscScalar z, PetscScalar t)
{ PetscScalar result;
  result = 4.0*M_PI*x*sin(2.0*M_PI*x)*sin(2.0*M_PI*z) - 4.0*M_PI*z*cos(2.0*M_PI*x)*cos(2.0*M_PI*z) + (sin(2.0*M_PI*x)*cos(2.0*M_PI*z) + 1.5)*(-2.0*M_PI*x*sin(2.0*M_PI*x)*sin(2.0*M_PI*z) + 2.0*M_PI*z*cos(2.0*M_PI*x)*cos(2.0*M_PI*z) + 2*sin(2.0*M_PI*z)*cos(2.0*M_PI*x)) + 8.0*pow(M_PI, 2)*(pow(x, 2) + pow(z, 2) + 1.0)*sin(2.0*M_PI*z)*cos(2.0*M_PI*x);
  return(result);
}
static PetscScalar get_bc_neumann_x(PetscScalar x, PetscScalar z, PetscScalar t)
{ PetscScalar result;
  result = -2.0*M_PI*sin(2.0*M_PI*x)*sin(2.0*M_PI*z);
  return(result);
}
static PetscScalar get_bc_neumann_z(PetscScalar x, PetscScalar z, PetscScalar t)
{ PetscScalar result;
  result = 2.0*M_PI*cos(2.0*M_PI*x)*cos(2.0*M_PI*z);
  return(result);
}

// ---------------------------------------
static PetscScalar get_Q(PetscInt test, PetscScalar x, PetscScalar z, PetscScalar t)
{ PetscScalar result = 0.0;
  if (test==1) result = get_Q1(x,z,t);
  if (test==2) result = get_Q2(x,z,t);
  if (test==3) result = get_Q3(x,z,t);
  if (test==4) result = get_Q4(x,z,t);
  if (test==5) result = get_Q5(x,z,t);
  return(result);
}
static PetscScalar get_A(PetscInt test, PetscScalar x, PetscScalar z, PetscScalar t)
{ PetscScalar result = 0.0;
  if (test==1) result = get_A1(x,z,t);
  if (test==2) result = get_A2(x,z,t);
  if (test==3) result = get_A3(x,z,t);
  if (test==4) result = get_A4(x,z,t);
  if (test==5) result = get_A5(x,z,t);
  return(result);
}
static PetscScalar get_B(PetscInt test, PetscScalar x, PetscScalar z, PetscScalar t)
{ PetscScalar result = 0.0;
  if (test==1) result = get_B1(x,z,t);
  if (test==2) result = get_B2(x,z,t);
  if (test==3) result = get_B3(x,z,t);
  if (test==4) result = get_B4(x,z,t);
  if (test==5) result = get_B5(x,z,t);
  return(result);
}
static PetscScalar get_ux(PetscInt test, PetscScalar x, PetscScalar z, PetscScalar t)
{ PetscScalar result = 0.0;
  if (test==1) result = get_ux1(x,z,t);
  if (test==2) result = get_ux2(x,z,t);
  if (test==3) result = get_ux3(x,z,t);
  if (test==4) result = get_ux4(x,z,t);
  if (test==5) result = get_ux5(x,z,t);
  return(result);
}
static PetscScalar get_uz(PetscInt test, PetscScalar x, PetscScalar z, PetscScalar t)
{ PetscScalar result = 0.0;
  if (test==1) result = get_uz1(x,z,t);
  if (test==2) result = get_uz2(x,z,t);
  if (test==3) result = get_uz3(x,z,t);
  if (test==4) result = get_uz4(x,z,t);
  if (test==5) result = get_uz5(x,z,t);
  return(result);
}
static PetscScalar get_frhs(PetscInt test, PetscScalar x, PetscScalar z, PetscScalar t)
{ PetscScalar result = 0.0;
  if (test==1) result = get_frhs1(x,z,t);
  if (test==2) result = get_frhs2(x,z,t);
  if (test==3) result = get_frhs3(x,z,t);
  if (test==4) result = get_frhs4(x,z,t);
  if (test==5) result = get_frhs5(x,z,t);
  return(result);
}

// ---------------------------------------
// Application functions
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "Numerical_solution"
PetscErrorCode Numerical_solution(void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  FDPDE          fd;
  DM             dm, dmcoeff;
  Vec            x, xprev, xmms, coeff, coeffprev;
  PetscInt       nx, nz, istep=0;
  PetscScalar    dx, dz,xmin, zmin, xmax, zmax; //, dt, dt_damp = 1.0e-2;
  char           fout[FNAME_LENGTH];
  PetscFunctionBeginUser;

  // Element count
  nx = usr->par->nx;
  nz = usr->par->nz;

  // Domain coords - modify coord of dm such that unknowns are located on the boundary limits (0,1)
  dx = usr->par->L/(2*nx-2);
  dz = usr->par->H/(2*nz-2);
  xmin = usr->par->xmin-dx;
  zmin = usr->par->zmin-dz;
  xmax = usr->par->xmin+usr->par->L+dx;
  zmax = usr->par->zmin+usr->par->H+dz;

  // Create the FD-pde object
  PetscCall(FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_ADVDIFF,&fd));
  PetscCall(FDPDESetUp(fd));

  // Set coefficients and BC evaluation functions
  PetscCall(FDPDESetFunctionBCList(fd,FormBCList,bc_description,usr)); 
  PetscCall(FDPDESetFunctionCoefficient(fd,FormCoefficient,coeff_description,usr)); 

  
  if ((usr->par->test<=2) || (usr->par->test==5)) { // steady - state
    if      (usr->par->test==1) { PetscCall(FDPDEAdvDiffSetAdvectSchemeType(fd,ADV_NONE)); }
    else if ((usr->par->test==2) || (usr->par->test==5)) { 
      if (usr->par->adv_scheme == 0) { PetscCall(FDPDEAdvDiffSetAdvectSchemeType(fd,ADV_UPWIND)); }
      if (usr->par->adv_scheme == 1) { PetscCall(FDPDEAdvDiffSetAdvectSchemeType(fd,ADV_UPWIND2)); }
      if (usr->par->adv_scheme == 2) { PetscCall(FDPDEAdvDiffSetAdvectSchemeType(fd,ADV_FROMM)); }
    }
    
    PetscCall(FDPDEAdvDiffSetTimeStepSchemeType(fd,TS_NONE));

    // Solve
    PetscCall(FDPDESolve(fd,NULL));
    PetscCall(FDPDEGetSolution(fd,&x)); 

    PetscCall(FDPDEGetDM(fd, &dm)); 
    PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s",usr->par->fdir_out,usr->par->fname_out));
    PetscCall(DMStagViewBinaryPython(dm,x,fout));

    // Compute manufactured solution and errors
    PetscCall(ComputeManufacturedSolution(dm,&xmms,usr)); 
    PetscCall(ComputeErrorNorms(dm,x,xmms));

    PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s_mms",usr->par->fdir_out,usr->par->fname_out));
    PetscCall(DMStagViewBinaryPython(dm,xmms,fout));

    // Destroy objects
    PetscCall(VecDestroy(&x)); 
    PetscCall(VecDestroy(&xmms)); 

  } else { // time dependent
    if (usr->par->test == 3) {
      PetscCall(FDPDEAdvDiffSetAdvectSchemeType(fd,ADV_NONE)); 
    } else {
      if (usr->par->adv_scheme == 0) { PetscCall(FDPDEAdvDiffSetAdvectSchemeType(fd,ADV_UPWIND)); }
      if (usr->par->adv_scheme == 1) { PetscCall(FDPDEAdvDiffSetAdvectSchemeType(fd,ADV_UPWIND2)); }
      if (usr->par->adv_scheme == 2) { PetscCall(FDPDEAdvDiffSetAdvectSchemeType(fd,ADV_FROMM)); }
    }

    if (usr->par->ts_scheme == 0) { PetscCall(FDPDEAdvDiffSetTimeStepSchemeType(fd,TS_FORWARD_EULER)); }
    if (usr->par->ts_scheme == 1) { PetscCall(FDPDEAdvDiffSetTimeStepSchemeType(fd,TS_BACKWARD_EULER)); }
    if (usr->par->ts_scheme == 2) { PetscCall(FDPDEAdvDiffSetTimeStepSchemeType(fd,TS_CRANK_NICHOLSON )); }
  
    // Set initial Q profile and coefficient
    PetscCall(FDPDEGetDM(fd, &dm)); 
    PetscCall(FDPDEAdvDiffGetPrevSolution(fd,&xprev));
    PetscCall(SetInitialQProfile(dm,xprev,usr));
    PetscCall(VecDestroy(&xprev));

    PetscCall(FDPDEGetCoefficient(fd,&dmcoeff,NULL));
    PetscCall(FDPDEAdvDiffGetPrevCoefficient(fd,&coeffprev));
    PetscCall(SetInitialQCoefficient(dmcoeff,coeffprev,usr));
    PetscCall(VecDestroy(&coeffprev));

    // Time loop
    while ((usr->par->t <= usr->par->tmax) && (istep<=usr->par->tstep)) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"# TIMESTEP %d: \n",istep));

      // Set dt
      // PetscCall(FDPDEAdvDiffComputeExplicitTimestep(fd,&dt));
      // usr->par->dt = PetscMin(dt,usr->par->dtmax);
      usr->par->dt = usr->par->dtmax;
      PetscCall(FDPDEAdvDiffSetTimestep(fd,usr->par->dt));

      // update approximate time
      usr->par->tinit = usr->par->t+usr->par->dt;

      // Solve
      PetscCall(FDPDESolve(fd,NULL));
      // converged = PETSC_FALSE;
      // while (!converged) {
      //   PetscCall(FDPDESolve(fd,&converged));
      //   if (!converged) { // Reduce dt if not converged
      //     usr->par->dt *= dt_damp;
      //     PetscCall(FDPDEAdvDiffSetTimestep(fd,usr->par->dt));
      //     usr->par->tinit = usr->par->t+usr->par->dt; // update approximate time
      //   }
      // }
      PetscCall(FDPDEGetSolution(fd,&x));

      // Update time
      usr->par->t += usr->par->dt;

      // Compute manufactured solution and errors per time step
      PetscCall(ComputeManufacturedSolution(dm,&xmms,usr)); 
      PetscCall(ComputeErrorNorms(dm,x,xmms));

      // Copy new solution and coefficient to old
      PetscCall(FDPDEAdvDiffGetPrevSolution(fd,&xprev));
      PetscCall(VecCopy(x,xprev));
      PetscCall(VecDestroy(&xprev));

      PetscCall(FDPDEGetCoefficient(fd,&dmcoeff,&coeff));
      PetscCall(FDPDEAdvDiffGetPrevCoefficient(fd,&coeffprev));
      PetscCall(VecCopy(coeff,coeffprev));
      PetscCall(VecDestroy(&coeffprev));

      // Output solution
      if (istep % usr->par->tout == 0 ) {
        PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s_ts%1.3d",usr->par->fdir_out,usr->par->fname_out,istep));
        PetscCall(DMStagViewBinaryPython(dm,x,fout));

        PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s_mms_ts%1.3d",usr->par->fdir_out,usr->par->fname_out,istep));
        PetscCall(DMStagViewBinaryPython(dm,xmms,fout));
      }

      // Clean up
      PetscCall(VecDestroy(&x));
      PetscCall(VecDestroy(&xmms));

      // increment timestep
      istep++;

      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"# TIME: time = %1.12e dt = %1.12e \n",usr->par->t,usr->par->dt));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n"));
    }
  }

  // Destroy objects
  PetscCall(DMDestroy(&dm));
  PetscCall(FDPDEDestroy(&fd));

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

  // Load command line or input file if required
  PetscCall(PetscOptionsInsert(PETSC_NULLPTR,&argc,&argv,NULL)); 

  // Input user parameters and print
  PetscCall(InputParameters(&usr)); 

  // Print user parameters
  PetscCall(InputPrintData(usr)); 

  // Numerical solution
  PetscCall(PetscTime(&start_time)); 
  PetscCall(Numerical_solution(usr)); 
  PetscCall(PetscTime(&end_time)); 
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"# Total runtime: %g (sec) \n", end_time - start_time));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n"));

  PetscCall(PetscBagDestroy(&usr->bag)); 
  PetscCall(PetscFree(usr));

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
  PetscCall(PetscBagRegisterInt(bag, &par->test, 1, "test", "Test: 1-steady state diffusion")); 

  PetscCall(PetscBagRegisterScalar(bag, &par->xmin, 0.0, "xmin", "Start coordinate of domain in x-dir")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->zmin, 0.0, "zmin", "Start coordinate of domain in z-dir")); 

  PetscCall(PetscBagRegisterScalar(bag, &par->L, 1.0, "L", "Length of domain in x-dir")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->H, 1.0, "H", "Height of domain in z-dir")); 

  // Physical and material parameters
  PetscCall(PetscBagRegisterScalar(bag, &par->Q0, 1.0, "Q0", "Amplitude of gaussian")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->taux, 3.0e-1, "taux", "Gaussian shape coeff x")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->tauz, 3.0e-1, "tauz", "Gaussian shape coeff z")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->x0, 0.5, "x0", "Gaussian shape coord x")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->z0, 0.5, "z0", "Gaussian shape coord z")); 

  // Time stepping and advection
  PetscCall(PetscBagRegisterInt(bag, &par->ts_scheme,2, "ts_scheme", "Time stepping scheme 0-forward euler, 1-backward euler, 2-crank-nicholson")); 
  PetscCall(PetscBagRegisterInt(bag, &par->adv_scheme,1, "adv_scheme", "Advection scheme 0-upwind, 1-fromm")); 

  PetscCall(PetscBagRegisterInt(bag, &par->tout,1, "tout", "Output every tout time step")); 
  PetscCall(PetscBagRegisterInt(bag, &par->tstep,1, "tstep", "Maximum no of time steps")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->tmax, 1.0e2, "tmax", "Maximum time [-]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->dtmax, 1.0e-4, "dtmax", "Maximum time step size [-]")); 

  // Boundary conditions
  PetscCall(PetscBagRegisterInt(bag, &par->bcleft, 0, "bcleft", "0-Dirichlet, 1-Neumann")); 
  PetscCall(PetscBagRegisterInt(bag, &par->bcright, 0, "bcright", "0-Dirichlet, 1-Neumann")); 
  PetscCall(PetscBagRegisterInt(bag, &par->bcdown, 0, "bcdown", "0-Dirichlet, 1-Neumann")); 
  PetscCall(PetscBagRegisterInt(bag, &par->bcup, 0, "bcup", "0-Dirichlet, 1-Neumann")); 

  par->t  = 0.0;
  par->dt = 0.0;
  par->tinit  = 0.0;

  // Input/output 
  PetscCall(PetscBagRegisterString(bag,&par->fname_out,FNAME_LENGTH,"out_advdiff_mms","output_file","Name for output file, set with: -output_file <filename>")); 
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
  PetscCall(PetscPrintf(usr->comm,"# --------------------------------------- #\n"));
  PetscCall(PetscPrintf(usr->comm,"# Test_advdiff_mms_convergence: %s \n",&(date[0])));
  PetscCall(PetscPrintf(usr->comm,"# --------------------------------------- #\n"));
  PetscCall(PetscPrintf(usr->comm,"# PETSc options: %s \n",opts));
  PetscCall(PetscPrintf(usr->comm,"# --------------------------------------- #\n"));

  // Print usr bag
  PetscCall(PetscBagView(usr->bag,PETSC_VIEWER_STDOUT_WORLD)); 
  PetscCall(PetscPrintf(usr->comm,"# --------------------------------------- #\n"));

  // Free memory
  PetscCall(PetscFree(opts)); 

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// SetInitialQProfile
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "SetInitialQProfile"
PetscErrorCode SetInitialQProfile(DM dm, Vec x, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  Vec            xlocal;
  PetscInt       i,j, sx, sz, nx, nz, icenter;
  PetscScalar    ***xx, **coordx, **coordz;
  PetscFunctionBeginUser;

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
      PetscScalar   xp, zp;
      PetscInt      idx;

      point.i = i; point.j = j; point.loc = ELEMENT; point.c = 0;
      xp = coordx[i][icenter]; 
      zp = coordz[j][icenter];

      PetscCall(DMStagGetLocationSlot(dm, point.loc, point.c, &idx)); 
      xx[j][i][idx] = get_Q(usr->par->test,xp,zp,usr->par->t);
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
// SetInitialQCoefficient
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "SetInitialQCoefficient"
PetscErrorCode SetInitialQCoefficient(DM dmcoeff, Vec coeff, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       i, j, sx, sz, nx, nz,iprev,inext,icenter;
  Vec            coefflocal;
  PetscScalar    ***c, **coordx, **coordz;
  PetscFunctionBeginUser;

  // Get domain corners
  PetscCall(DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 

  PetscCall(DMStagGetProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmcoeff,LEFT,&iprev));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmcoeff,RIGHT,&inext)); 
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmcoeff,ELEMENT,&icenter));

  // Create coefficient local vector
  PetscCall(DMCreateLocalVector(dmcoeff, &coefflocal)); 
  PetscCall(DMStagVecGetArray(dmcoeff, coefflocal, &c)); 
  
  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {

      { // A = Amms (center)
        DMStagStencil point;
        PetscScalar   xp,zp;
        PetscInt      idx;
        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 0;
        xp = coordx[i][icenter]; zp = coordz[j][icenter];
        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
        c[j][i][idx] = get_A(usr->par->test,xp,zp,usr->par->t);
      }

      { // C = -frhs 
        DMStagStencil point;
        PetscInt      idx;
        PetscScalar   xp,zp;
        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 1;
        xp = coordx[i][icenter]; zp = coordz[j][icenter];
        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
        c[j][i][idx] = -get_frhs(usr->par->test,xp,zp,usr->par->t);
      }

      { // B = Bmms (edge)
        DMStagStencil point[4];
        PetscScalar   xp[4],zp[4];
        PetscInt      ii, idx;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 0;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = 0;
        point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = 0;

        xp[0] = coordx[i][iprev  ]; zp[0] = coordz[j][icenter];
        xp[1] = coordx[i][inext  ]; zp[1] = coordz[j][icenter];
        xp[2] = coordx[i][icenter]; zp[2] = coordz[j][iprev  ];
        xp[3] = coordx[i][icenter]; zp[3] = coordz[j][inext  ];

        for (ii = 0; ii < 4; ii++) {
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, 0, &idx)); 
          c[j][i][idx] = get_B(usr->par->test,xp[ii],zp[ii],usr->par->t);
        }
      }

      { // u = velocity (edge) = manufactured solution
        DMStagStencil point[4];
        PetscScalar   xp[4],zp[4],val[4];
        PetscInt      ii, idx;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 1;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 1;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = 1;
        point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = 1;

        xp[0] = coordx[i][iprev  ]; zp[0] = coordz[j][icenter];
        xp[1] = coordx[i][inext  ]; zp[1] = coordz[j][icenter];
        xp[2] = coordx[i][icenter]; zp[2] = coordz[j][iprev  ];
        xp[3] = coordx[i][icenter]; zp[3] = coordz[j][inext  ];

        val[0] = get_ux(usr->par->test,xp[0],zp[0],usr->par->t);
        val[1] = get_ux(usr->par->test,xp[1],zp[1],usr->par->t);
        val[2] = get_uz(usr->par->test,xp[2],zp[2],usr->par->t);
        val[3] = get_uz(usr->par->test,xp[3],zp[3],usr->par->t);

        for (ii = 0; ii < 4; ii++) {
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, 1, &idx)); 
          c[j][i][idx] = val[ii];
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
// FormCoefficient
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormCoefficient"
PetscErrorCode FormCoefficient(FDPDE fd, DM dm, Vec x, DM dmcoeff, Vec coeff, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       i, j, sx, sz, nx, nz,iprev,inext,icenter;
  Vec            coefflocal;
  PetscScalar    ***c, **coordx, **coordz;
  PetscFunctionBeginUser;

  // Get domain corners
  PetscCall(DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 

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

      { // A = Amms (center)
        DMStagStencil point;
        PetscScalar   xp,zp;
        PetscInt      idx;
        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 0;
        xp = coordx[i][icenter]; zp = coordz[j][icenter];
        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
        c[j][i][idx] = get_A(usr->par->test,xp,zp,usr->par->tinit);
      }

      { // C = -frhs 
        DMStagStencil point;
        PetscInt      idx;
        PetscScalar   xp,zp;
        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 1;
        xp = coordx[i][icenter]; zp = coordz[j][icenter];
        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
        c[j][i][idx] = -get_frhs(usr->par->test,xp,zp,usr->par->tinit);
      }

      { // B = Bmms (edge)
        DMStagStencil point[4];
        PetscScalar   xp[4],zp[4];
        PetscInt      ii, idx;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 0;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = 0;
        point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = 0;

        xp[0] = coordx[i][iprev  ]; zp[0] = coordz[j][icenter];
        xp[1] = coordx[i][inext  ]; zp[1] = coordz[j][icenter];
        xp[2] = coordx[i][icenter]; zp[2] = coordz[j][iprev  ];
        xp[3] = coordx[i][icenter]; zp[3] = coordz[j][inext  ];

        for (ii = 0; ii < 4; ii++) {
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, 0, &idx)); 
          c[j][i][idx] = get_B(usr->par->test,xp[ii],zp[ii],usr->par->tinit);
        }
      }

      { // u = velocity (edge) = manufactured solution
        DMStagStencil point[4];
        PetscScalar   xp[4],zp[4],val[4];
        PetscInt      ii, idx;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 1;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 1;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = 1;
        point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = 1;

        xp[0] = coordx[i][iprev  ]; zp[0] = coordz[j][icenter];
        xp[1] = coordx[i][inext  ]; zp[1] = coordz[j][icenter];
        xp[2] = coordx[i][icenter]; zp[2] = coordz[j][iprev  ];
        xp[3] = coordx[i][icenter]; zp[3] = coordz[j][inext  ];

        val[0] = get_ux(usr->par->test,xp[0],zp[0],usr->par->tinit);
        val[1] = get_ux(usr->par->test,xp[1],zp[1],usr->par->tinit);
        val[2] = get_uz(usr->par->test,xp[2],zp[2],usr->par->tinit);
        val[3] = get_uz(usr->par->test,xp[3],zp[3],usr->par->tinit);

        for (ii = 0; ii < 4; ii++) {
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, 1, &idx)); 
          c[j][i][idx] = val[ii];
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
// FormBCList
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormBCList"
PetscErrorCode FormBCList(DM dm, Vec x, DMStagBCList bclist, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       k,n_bc,*idx_bc;
  PetscScalar   *value_bc,*x_bc;
  BCType        *type_bc;
  PetscFunctionBeginUser;
  
  // Left:
  PetscCall(DMStagBCListGetValues(bclist,'w','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    if (usr->par->bcleft == 0) {
      value_bc[k] = get_Q(usr->par->test,x_bc[2*k],x_bc[2*k+1],usr->par->tinit);
      type_bc[k] = BC_DIRICHLET_STAG;
    } else {
      value_bc[k] = get_bc_neumann_x(usr->par->xmin,x_bc[2*k+1],usr->par->tinit); // use real boundary coordinates
      type_bc[k] = BC_NEUMANN;
    }
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));

  // RIGHT:
  PetscCall(DMStagBCListGetValues(bclist,'e','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    if (usr->par->bcright == 0) {
      value_bc[k] = get_Q(usr->par->test,x_bc[2*k],x_bc[2*k+1],usr->par->tinit);
      type_bc[k] = BC_DIRICHLET_STAG;
    } else {
      value_bc[k] = get_bc_neumann_x(usr->par->xmin+usr->par->L,x_bc[2*k+1],usr->par->tinit);
      type_bc[k] = BC_NEUMANN;
    }
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));

  // DOWN:
  PetscCall(DMStagBCListGetValues(bclist,'s','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    if (usr->par->bcdown == 0) {
      value_bc[k] = get_Q(usr->par->test,x_bc[2*k],x_bc[2*k+1],usr->par->tinit);
      type_bc[k] = BC_DIRICHLET_STAG;
    } else {
      value_bc[k] = get_bc_neumann_z(x_bc[2*k],usr->par->zmin,usr->par->tinit);
      type_bc[k] = BC_NEUMANN;
    }
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));

  // UP:
  PetscCall(DMStagBCListGetValues(bclist,'n','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    if (usr->par->bcup == 0) {
      value_bc[k] = get_Q(usr->par->test,x_bc[2*k],x_bc[2*k+1],usr->par->tinit);
      type_bc[k] = BC_DIRICHLET_STAG;
    } else {
      value_bc[k] = get_bc_neumann_z(x_bc[2*k],usr->par->zmin+usr->par->H,usr->par->tinit);
      type_bc[k] = BC_NEUMANN;
    }
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// Compute manufactured
// ---------------------------------------
PetscErrorCode ComputeManufacturedSolution(DM dm, Vec *_xmms, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz, idx, icenter;
  PetscScalar    ***xx;
  PetscScalar    **coordx,**coordz;
  Vec            xmms,xmmslocal;
  PetscFunctionBeginUser;

  // Create local and global vectors for MMS solutions
  PetscCall(DMCreateGlobalVector(dm,&xmms     )); 
  PetscCall(DMCreateLocalVector (dm,&xmmslocal)); 
  PetscCall(DMStagVecGetArray(dm,xmmslocal,&xx)); 

  // Get domain corners and coordinates
  PetscCall(DMStagGetGlobalSizes(dm, &Nx, &Nz,NULL));
  PetscCall(DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 
  PetscCall(DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter)); 

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      PetscCall(DMStagGetLocationSlot(dm,ELEMENT,0,&idx)); 
      xx[j][i][idx] = get_Q(usr->par->test,coordx[i][icenter],coordz[j][icenter],usr->par->t);
    }
  }

  // Restore arrays
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));

  // Restore and map local to global
  PetscCall(DMStagVecRestoreArray(dm,xmmslocal,&xx)); 
  PetscCall(DMLocalToGlobalBegin(dm,xmmslocal,INSERT_VALUES,xmms)); 
  PetscCall(DMLocalToGlobalEnd  (dm,xmmslocal,INSERT_VALUES,xmms)); 
  PetscCall(VecDestroy(&xmmslocal)); 

  // Assign pointers
  *_xmms = xmms;
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// ComputeErrorNorms
// ---------------------------------------
PetscErrorCode ComputeErrorNorms(DM dm,Vec x,Vec xmms)
{
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz;
  PetscScalar    dx, dz, dv;
  PetscScalar    sum_err, gsum_err, sum_mms, gsum_mms;
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
  PetscCall(DMGlobalToLocal (dm, xmms, INSERT_VALUES, xalocal)); 

  // Initialize norms
  sum_err = 0.0; sum_mms = 0.0;

  dx = 1.0/Nx;
  dz = 1.0/Nz;
  dv = dx*dz;

  // Loop over local domain to calculate ELEMENT errors
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      PetscScalar    Qx, Qa;
      DMStagStencil  point;
      
      // Get stencil values
      point.i = i; point.j = j; point.loc = ELEMENT; point.c = 0;
      PetscCall(DMStagVecGetValuesStencil(dm, xlocal, 1,&point,&Qx)); 
      PetscCall(DMStagVecGetValuesStencil(dm, xalocal,1,&point,&Qa));  // this is porosity

      // Calculate sums for L2 norms - and normalize by magnitude of MMS solution
      sum_err += (Qx-Qa)*(Qx-Qa)*dv;
      sum_mms += Qa*Qa*dv;
    }
  }

  // Collect data 
  PetscCall(MPI_Allreduce(&sum_err, &gsum_err, 1, MPI_DOUBLE, MPI_SUM, comm)); 
  PetscCall(MPI_Allreduce(&sum_mms, &gsum_mms, 1, MPI_DOUBLE, MPI_SUM, comm)); 

  // Restore arrays and vectors
  PetscCall(DMRestoreLocalVector(dm, &xlocal )); 
  PetscCall(DMRestoreLocalVector(dm, &xalocal )); 

  // Print information
  PetscCall(PetscPrintf(comm,"# NORMS: \n"));
  PetscCall(PetscPrintf(comm,"# L2 square: num = %1.12e mms = %1.12e \n",gsum_err,gsum_mms));
  PetscCall(PetscPrintf(comm,"# Grid info: hx = %1.12e hz = %1.12e\n",dx,dz));

  PetscFunctionReturn(PETSC_SUCCESS);
}
