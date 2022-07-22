// ---------------------------------------
// (ADVDIFF) Advection-diffusion convergence test using MMS
// run: ./tests/test_advdiff_mms_convergence.app -pc_type lu -pc_factor_mat_solver_type umfpack -pc_factor_mat_ordering_type external -nx 10 -nz 10
// python test: ./tests/python/test_advdiff_mms_convergence.py
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

#include "petsc.h"
#include "../src/fdpde_advdiff.h"
#include "../src/dmstagoutput.h"

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
  // PetscBool      converged;
  PetscErrorCode ierr;

  PetscFunctionBegin;

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
  ierr = FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_ADVDIFF,&fd);CHKERRQ(ierr);
  ierr = FDPDESetUp(fd);CHKERRQ(ierr);

  // Set coefficients and BC evaluation functions
  ierr = FDPDESetFunctionBCList(fd,FormBCList,bc_description,usr); CHKERRQ(ierr);
  ierr = FDPDESetFunctionCoefficient(fd,FormCoefficient,coeff_description,usr); CHKERRQ(ierr);

  
  if ((usr->par->test<=2) || (usr->par->test==5)) { // steady - state
    if      (usr->par->test==1) { ierr = FDPDEAdvDiffSetAdvectSchemeType(fd,ADV_NONE);CHKERRQ(ierr); }
    else if ((usr->par->test==2) || (usr->par->test==5)) { 
      if (usr->par->adv_scheme == 0) { ierr = FDPDEAdvDiffSetAdvectSchemeType(fd,ADV_UPWIND);CHKERRQ(ierr); }
      if (usr->par->adv_scheme == 1) { ierr = FDPDEAdvDiffSetAdvectSchemeType(fd,ADV_UPWIND2);CHKERRQ(ierr); }
      if (usr->par->adv_scheme == 2) { ierr = FDPDEAdvDiffSetAdvectSchemeType(fd,ADV_FROMM);CHKERRQ(ierr); }
    }
    
    ierr = FDPDEAdvDiffSetTimeStepSchemeType(fd,TS_NONE);CHKERRQ(ierr);

    // Solve
    ierr = FDPDESolve(fd,NULL);CHKERRQ(ierr);
    ierr = FDPDEGetSolution(fd,&x);CHKERRQ(ierr); 

    ierr = FDPDEGetDM(fd, &dm); CHKERRQ(ierr);
    ierr = PetscSNPrintf(fout,sizeof(fout),"%s/%s",usr->par->fdir_out,usr->par->fname_out);
    ierr = DMStagViewBinaryPython(dm,x,fout);CHKERRQ(ierr);

    // Compute manufactured solution and errors
    ierr = ComputeManufacturedSolution(dm,&xmms,usr); CHKERRQ(ierr);
    ierr = ComputeErrorNorms(dm,x,xmms);CHKERRQ(ierr);

    ierr = PetscSNPrintf(fout,sizeof(fout),"%s/%s_mms",usr->par->fdir_out,usr->par->fname_out);
    ierr = DMStagViewBinaryPython(dm,xmms,fout);CHKERRQ(ierr);

    // Destroy objects
    ierr = VecDestroy(&x); CHKERRQ(ierr);
    ierr = VecDestroy(&xmms); CHKERRQ(ierr);

  } else { // time dependent
    if (usr->par->test == 3) {
      ierr = FDPDEAdvDiffSetAdvectSchemeType(fd,ADV_NONE);CHKERRQ(ierr); 
    } else {
      if (usr->par->adv_scheme == 0) { ierr = FDPDEAdvDiffSetAdvectSchemeType(fd,ADV_UPWIND);CHKERRQ(ierr); }
      if (usr->par->adv_scheme == 1) { ierr = FDPDEAdvDiffSetAdvectSchemeType(fd,ADV_UPWIND2);CHKERRQ(ierr); }
      if (usr->par->adv_scheme == 2) { ierr = FDPDEAdvDiffSetAdvectSchemeType(fd,ADV_FROMM);CHKERRQ(ierr); }
    }

    if (usr->par->ts_scheme == 0) { ierr = FDPDEAdvDiffSetTimeStepSchemeType(fd,TS_FORWARD_EULER);CHKERRQ(ierr); }
    if (usr->par->ts_scheme == 1) { ierr = FDPDEAdvDiffSetTimeStepSchemeType(fd,TS_BACKWARD_EULER);CHKERRQ(ierr); }
    if (usr->par->ts_scheme == 2) { ierr = FDPDEAdvDiffSetTimeStepSchemeType(fd,TS_CRANK_NICHOLSON );CHKERRQ(ierr); }
  
    // Set initial Q profile and coefficient
    ierr = FDPDEGetDM(fd, &dm); CHKERRQ(ierr);
    ierr = FDPDEAdvDiffGetPrevSolution(fd,&xprev);CHKERRQ(ierr);
    ierr = SetInitialQProfile(dm,xprev,usr);CHKERRQ(ierr);
    ierr = VecDestroy(&xprev);CHKERRQ(ierr);

    ierr = FDPDEGetCoefficient(fd,&dmcoeff,NULL);CHKERRQ(ierr);
    ierr = FDPDEAdvDiffGetPrevCoefficient(fd,&coeffprev);CHKERRQ(ierr);
    ierr = SetInitialQCoefficient(dmcoeff,coeffprev,usr);CHKERRQ(ierr);
    ierr = VecDestroy(&coeffprev);CHKERRQ(ierr);

    // Time loop
    while ((usr->par->t <= usr->par->tmax) && (istep<=usr->par->tstep)) {
      PetscPrintf(PETSC_COMM_WORLD,"# TIMESTEP %d: \n",istep);

      // Set dt
      // ierr = FDPDEAdvDiffComputeExplicitTimestep(fd,&dt);CHKERRQ(ierr);
      // usr->par->dt = PetscMin(dt,usr->par->dtmax);
      usr->par->dt = usr->par->dtmax;
      ierr = FDPDEAdvDiffSetTimestep(fd,usr->par->dt);CHKERRQ(ierr);

      // update approximate time
      usr->par->tinit = usr->par->t+usr->par->dt;

      // Solve
      ierr = FDPDESolve(fd,NULL);CHKERRQ(ierr);
      // converged = PETSC_FALSE;
      // while (!converged) {
      //   ierr = FDPDESolve(fd,&converged);CHKERRQ(ierr);
      //   if (!converged) { // Reduce dt if not converged
      //     usr->par->dt *= dt_damp;
      //     ierr = FDPDEAdvDiffSetTimestep(fd,usr->par->dt);CHKERRQ(ierr);
      //     usr->par->tinit = usr->par->t+usr->par->dt; // update approximate time
      //   }
      // }
      ierr = FDPDEGetSolution(fd,&x);CHKERRQ(ierr);

      // Update time
      usr->par->t += usr->par->dt;

      // Compute manufactured solution and errors per time step
      ierr = ComputeManufacturedSolution(dm,&xmms,usr); CHKERRQ(ierr);
      ierr = ComputeErrorNorms(dm,x,xmms);CHKERRQ(ierr);

      // Copy new solution and coefficient to old
      ierr = FDPDEAdvDiffGetPrevSolution(fd,&xprev);CHKERRQ(ierr);
      ierr = VecCopy(x,xprev);CHKERRQ(ierr);
      ierr = VecDestroy(&xprev);CHKERRQ(ierr);

      ierr = FDPDEGetCoefficient(fd,&dmcoeff,&coeff);CHKERRQ(ierr);
      ierr = FDPDEAdvDiffGetPrevCoefficient(fd,&coeffprev);CHKERRQ(ierr);
      ierr = VecCopy(coeff,coeffprev);CHKERRQ(ierr);
      ierr = VecDestroy(&coeffprev);CHKERRQ(ierr);

      // Output solution
      if (istep % usr->par->tout == 0 ) {
        ierr = PetscSNPrintf(fout,sizeof(fout),"%s/%s_ts%1.3d",usr->par->fdir_out,usr->par->fname_out,istep);
        ierr = DMStagViewBinaryPython(dm,x,fout);CHKERRQ(ierr);

        ierr = PetscSNPrintf(fout,sizeof(fout),"%s/%s_mms_ts%1.3d",usr->par->fdir_out,usr->par->fname_out,istep);
        ierr = DMStagViewBinaryPython(dm,xmms,fout);CHKERRQ(ierr);
      }

      // Clean up
      ierr = VecDestroy(&x);CHKERRQ(ierr);
      ierr = VecDestroy(&xmms);CHKERRQ(ierr);

      // increment timestep
      istep++;

      PetscPrintf(PETSC_COMM_WORLD,"# TIME: time = %1.12e dt = %1.12e \n",usr->par->t,usr->par->dt);
      PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");
    }
  }

  // Destroy objects
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = FDPDEDestroy(&fd);CHKERRQ(ierr);

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

  // Load command line or input file if required
  ierr = PetscOptionsInsert(PETSC_NULL,&argc,&argv,NULL); CHKERRQ(ierr);

  // Input user parameters and print
  ierr = InputParameters(&usr); CHKERRQ(ierr);

  // Print user parameters
  ierr = InputPrintData(usr); CHKERRQ(ierr);

  // Numerical solution
  ierr = PetscTime(&start_time); CHKERRQ(ierr);
  ierr = Numerical_solution(usr); CHKERRQ(ierr);
  ierr = PetscTime(&end_time); CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"# Total runtime: %g (sec) \n", end_time - start_time);
  PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");

  ierr = PetscBagDestroy(&usr->bag); CHKERRQ(ierr);
  ierr = PetscFree(usr);             CHKERRQ(ierr);

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
  ierr = PetscBagRegisterInt(bag, &par->test, 1, "test", "Test: 1-steady state diffusion"); CHKERRQ(ierr);

  ierr = PetscBagRegisterScalar(bag, &par->xmin, 0.0, "xmin", "Start coordinate of domain in x-dir"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->zmin, 0.0, "zmin", "Start coordinate of domain in z-dir"); CHKERRQ(ierr);

  ierr = PetscBagRegisterScalar(bag, &par->L, 1.0, "L", "Length of domain in x-dir"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->H, 1.0, "H", "Height of domain in z-dir"); CHKERRQ(ierr);

  // Physical and material parameters
  ierr = PetscBagRegisterScalar(bag, &par->Q0, 1.0, "Q0", "Amplitude of gaussian"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->taux, 3.0e-1, "taux", "Gaussian shape coeff x"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->tauz, 3.0e-1, "tauz", "Gaussian shape coeff z"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->x0, 0.5, "x0", "Gaussian shape coord x"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->z0, 0.5, "z0", "Gaussian shape coord z"); CHKERRQ(ierr);

  // Time stepping and advection
  ierr = PetscBagRegisterInt(bag, &par->ts_scheme,2, "ts_scheme", "Time stepping scheme 0-forward euler, 1-backward euler, 2-crank-nicholson"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->adv_scheme,1, "adv_scheme", "Advection scheme 0-upwind, 1-fromm"); CHKERRQ(ierr);

  ierr = PetscBagRegisterInt(bag, &par->tout,1, "tout", "Output every tout time step"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->tstep,1, "tstep", "Maximum no of time steps"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->tmax, 1.0e2, "tmax", "Maximum time [-]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->dtmax, 1.0e-4, "dtmax", "Maximum time step size [-]"); CHKERRQ(ierr);

  // Boundary conditions
  ierr = PetscBagRegisterInt(bag, &par->bcleft, 0, "bcleft", "0-Dirichlet, 1-Neumann"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->bcright, 0, "bcright", "0-Dirichlet, 1-Neumann"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->bcdown, 0, "bcdown", "0-Dirichlet, 1-Neumann"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->bcup, 0, "bcup", "0-Dirichlet, 1-Neumann"); CHKERRQ(ierr);

  par->t  = 0.0;
  par->dt = 0.0;
  par->tinit  = 0.0;

  // Input/output 
  ierr = PetscBagRegisterString(bag,&par->fname_out,FNAME_LENGTH,"out_advdiff_mms","output_file","Name for output file, set with: -output_file <filename>"); CHKERRQ(ierr);
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
  PetscPrintf(usr->comm,"# Test_advdiff_mms_convergence: %s \n",&(date[0]));
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

  PetscErrorCode ierr;
  PetscFunctionBegin;

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
      PetscScalar   xp, zp;
      PetscInt      idx;

      point.i = i; point.j = j; point.loc = ELEMENT; point.c = 0;
      xp = coordx[i][icenter]; 
      zp = coordz[j][icenter];

      ierr = DMStagGetLocationSlot(dm, point.loc, point.c, &idx); CHKERRQ(ierr);
      xx[j][i][idx] = get_Q(usr->par->test,xp,zp,usr->par->t);
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
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  // Get domain corners
  ierr = DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  ierr = DMStagGetProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmcoeff,LEFT,&iprev);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmcoeff,RIGHT,&inext);CHKERRQ(ierr); 
  ierr = DMStagGetProductCoordinateLocationSlot(dmcoeff,ELEMENT,&icenter);CHKERRQ(ierr);

  // Create coefficient local vector
  ierr = DMCreateLocalVector(dmcoeff, &coefflocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dmcoeff, coefflocal, &c); CHKERRQ(ierr);
  
  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {

      { // A = Amms (center)
        DMStagStencil point;
        PetscScalar   xp,zp;
        PetscInt      idx;
        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 0;
        xp = coordx[i][icenter]; zp = coordz[j][icenter];
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = get_A(usr->par->test,xp,zp,usr->par->t);
      }

      { // C = -frhs 
        DMStagStencil point;
        PetscInt      idx;
        PetscScalar   xp,zp;
        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 1;
        xp = coordx[i][icenter]; zp = coordz[j][icenter];
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
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
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, 0, &idx); CHKERRQ(ierr);
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
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, 1, &idx); CHKERRQ(ierr);
          c[j][i][idx] = val[ii];
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
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  // Get domain corners
  ierr = DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  ierr = DMStagGetProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmcoeff,LEFT,&iprev);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmcoeff,RIGHT,&inext);CHKERRQ(ierr); 
  ierr = DMStagGetProductCoordinateLocationSlot(dmcoeff,ELEMENT,&icenter);CHKERRQ(ierr);

  // Create coefficient local vector
  ierr = DMCreateLocalVector(dmcoeff, &coefflocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dmcoeff, coefflocal, &c); CHKERRQ(ierr);
  
  // Loop over local domain - set initial density and viscosity
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {

      { // A = Amms (center)
        DMStagStencil point;
        PetscScalar   xp,zp;
        PetscInt      idx;
        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 0;
        xp = coordx[i][icenter]; zp = coordz[j][icenter];
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = get_A(usr->par->test,xp,zp,usr->par->tinit);
      }

      { // C = -frhs 
        DMStagStencil point;
        PetscInt      idx;
        PetscScalar   xp,zp;
        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 1;
        xp = coordx[i][icenter]; zp = coordz[j][icenter];
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
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
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, 0, &idx); CHKERRQ(ierr);
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
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, 1, &idx); CHKERRQ(ierr);
          c[j][i][idx] = val[ii];
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  
  // Left:
  ierr = DMStagBCListGetValues(bclist,'w','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    if (usr->par->bcleft == 0) {
      value_bc[k] = get_Q(usr->par->test,x_bc[2*k],x_bc[2*k+1],usr->par->tinit);
      type_bc[k] = BC_DIRICHLET_STAG;
    } else {
      value_bc[k] = get_bc_neumann_x(usr->par->xmin,x_bc[2*k+1],usr->par->tinit); // use real boundary coordinates
      type_bc[k] = BC_NEUMANN;
    }
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // RIGHT:
  ierr = DMStagBCListGetValues(bclist,'e','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    if (usr->par->bcright == 0) {
      value_bc[k] = get_Q(usr->par->test,x_bc[2*k],x_bc[2*k+1],usr->par->tinit);
      type_bc[k] = BC_DIRICHLET_STAG;
    } else {
      value_bc[k] = get_bc_neumann_x(usr->par->xmin+usr->par->L,x_bc[2*k+1],usr->par->tinit);
      type_bc[k] = BC_NEUMANN;
    }
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN:
  ierr = DMStagBCListGetValues(bclist,'s','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    if (usr->par->bcdown == 0) {
      value_bc[k] = get_Q(usr->par->test,x_bc[2*k],x_bc[2*k+1],usr->par->tinit);
      type_bc[k] = BC_DIRICHLET_STAG;
    } else {
      value_bc[k] = get_bc_neumann_z(x_bc[2*k],usr->par->zmin,usr->par->tinit);
      type_bc[k] = BC_NEUMANN;
    }
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP:
  ierr = DMStagBCListGetValues(bclist,'n','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    if (usr->par->bcup == 0) {
      value_bc[k] = get_Q(usr->par->test,x_bc[2*k],x_bc[2*k+1],usr->par->tinit);
      type_bc[k] = BC_DIRICHLET_STAG;
    } else {
      value_bc[k] = get_bc_neumann_z(x_bc[2*k],usr->par->zmin+usr->par->H,usr->par->tinit);
      type_bc[k] = BC_NEUMANN;
    }
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  PetscFunctionReturn(0);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Create local and global vectors for MMS solutions
  ierr = DMCreateGlobalVector(dm,&xmms     ); CHKERRQ(ierr);
  ierr = DMCreateLocalVector (dm,&xmmslocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm,xmmslocal,&xx); CHKERRQ(ierr);

  // Get domain corners and coordinates
  ierr = DMStagGetGlobalSizes(dm, &Nx, &Nz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter);CHKERRQ(ierr); 

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      ierr = DMStagGetLocationSlot(dm,ELEMENT,0,&idx); CHKERRQ(ierr);
      xx[j][i][idx] = get_Q(usr->par->test,coordx[i][icenter],coordz[j][icenter],usr->par->t);
    }
  }

  // Restore arrays
  ierr = DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);

  // Restore and map local to global
  ierr = DMStagVecRestoreArray(dm,xmmslocal,&xx); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm,xmmslocal,INSERT_VALUES,xmms); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,xmmslocal,INSERT_VALUES,xmms); CHKERRQ(ierr);
  ierr = VecDestroy(&xmmslocal); CHKERRQ(ierr);

  // Assign pointers
  *_xmms = xmms;
  
  PetscFunctionReturn(0);
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
  ierr = DMGlobalToLocal (dm, xmms, INSERT_VALUES, xalocal); CHKERRQ(ierr);

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
      ierr = DMStagVecGetValuesStencil(dm, xlocal, 1,&point,&Qx); CHKERRQ(ierr);
      ierr = DMStagVecGetValuesStencil(dm, xalocal,1,&point,&Qa); CHKERRQ(ierr); // this is porosity

      // Calculate sums for L2 norms - and normalize by magnitude of MMS solution
      sum_err += (Qx-Qa)*(Qx-Qa)*dv;
      sum_mms += Qa*Qa*dv;
    }
  }

  // Collect data 
  ierr = MPI_Allreduce(&sum_err, &gsum_err, 1, MPI_DOUBLE, MPI_SUM, comm); CHKERRQ(ierr);
  ierr = MPI_Allreduce(&sum_mms, &gsum_mms, 1, MPI_DOUBLE, MPI_SUM, comm); CHKERRQ(ierr);

  // Restore arrays and vectors
  ierr = DMRestoreLocalVector(dm, &xlocal ); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &xalocal ); CHKERRQ(ierr);

  // Print information
  PetscPrintf(comm,"# NORMS: \n");
  PetscPrintf(comm,"# L2 square: num = %1.12e mms = %1.12e \n",gsum_err,gsum_mms);
  PetscPrintf(comm,"# Grid info: hx = %1.12e hz = %1.12e\n",dx,dz);

  PetscFunctionReturn(0);
}
