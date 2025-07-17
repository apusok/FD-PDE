// ---------------------------------------
// 2D diffusion to test the enthalpy implementation
// run: ./test_enthalpy_2d_diffusion.app -pc_type lu -pc_factor_mat_solver_type umfpack -pc_factor_mat_ordering_type external -snes_monitor -log_view
// python output: test_enthalpy_2d_diffusion.py
// ---------------------------------------
static char help[] = "2D Diffusion problem using the Enthalpy Method\n\n";

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
#include "../src/fdpde_enthalpy.h"
#include "../src/dmstagoutput.h"

// ---------------------------------------
// Application Context
// ---------------------------------------
#define FNAME_LENGTH 200

// parameters (bag)
typedef struct {
  PetscInt       nx, nz;
  PetscScalar    L, H;
  PetscScalar    xmin, zmin;
  PetscScalar    k, dt, t, dtmax, tmax;
  PetscInt       ts_scheme, adv_scheme, tout, tstep, ncomp;
  char           fname_out[FNAME_LENGTH]; 
  char           fdir_out[FNAME_LENGTH]; 
} Params;

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
PetscErrorCode InputPrintData(UsrData*);
PetscErrorCode Numerical_solution(void*);
PetscErrorCode FormCoefficient(FDPDE, DM, Vec, DM, Vec, void*);
PetscErrorCode FormBCList(DM, Vec, DMStagBCList, void*);
PetscErrorCode Form_PotentialTemperature(PetscScalar,PetscScalar,PetscScalar*,void*); 
PetscErrorCode Analytical_solution(DM,Vec*,void*,PetscScalar);
EnthEvalErrorCode Form_Enthalpy(PetscScalar,PetscScalar[],PetscScalar,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscInt,void*); 

const char coeff_description[] =
"  << ENTHALPY Coefficients >> \n"
"  A1 = 0, B1 = 0, C1 = -k, D1 = 0  \n"
"  A2 = 0, B2 = 0, C2 = -k, D2 = 0  \n"
"  v = [0,0], vs = [0,0], vf = [0,0] \n";

const char bc_description[] =
"  << ENTHALPY BCs >> \n"
"  TEMP: LEFT, RIGHT: T = 0.0, DOWN, UP \n"
"  COMP: LEFT, RIGHT: T = 0.0, DOWN, UP \n";

const char enthalpy_method_description[] =
"  << ENTHALPY METHOD >> \n"
"  Input: H, C, P \n"
"  Output: T  = H, \n"
"          Cf = Cs = C, \n"
"          phi = 1.0 \n";

static PetscScalar analytical_solution(PetscScalar x, PetscScalar z, PetscScalar t, PetscScalar k) { 
  return 1.0/(4*PETSC_PI*k*t)*PetscExpScalar(-(x*x+z*z)/(4*k*t)); 
}

// ---------------------------------------
// Application functions
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "Numerical_solution"
PetscErrorCode Numerical_solution(void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  Params        *par;
  FDPDE          fd;
  DM             dm, dmcoeff, dmnew, dmP;
  Vec            x, xprev, xcoeff, xcoeffprev, xAnalytic, xnew, xP, xPprev;
  PetscInt       nx, nz, istep = 0;
  PetscScalar    xmin, zmin, xmax, zmax, dx, dz;
  char           fout[FNAME_LENGTH];
  PetscFunctionBeginUser;

  par = usr->par;
  // Element count
  nx = par->nx;
  nz = par->nz;

  // Domain coords
  dx = par->L/(2*nx-2);
  dz = par->H/(2*nz-2);
  xmin = par->xmin-dx;
  zmin = par->zmin-dz;
  xmax = par->xmin+par->L+dx;
  zmax = par->zmin+par->H+dz;

  // Set up Enthalpy system
  PetscCall(FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_ENTHALPY,&fd));
  PetscCall(FDPDEEnthalpySetNumberComponentsPhaseDiagram(fd,par->ncomp));
  PetscCall(FDPDESetUp(fd));

  PetscCall(FDPDESetFunctionBCList(fd,FormBCList,bc_description,usr)); 
  PetscCall(FDPDESetFunctionCoefficient(fd,FormCoefficient,coeff_description,usr)); 

  if (par->adv_scheme==0) { PetscCall(FDPDEEnthalpySetAdvectSchemeType(fd,ADV_UPWIND)); }
  if (par->adv_scheme==1) { PetscCall(FDPDEEnthalpySetAdvectSchemeType(fd,ADV_UPWIND2)); }
  if (par->adv_scheme==2) { PetscCall(FDPDEEnthalpySetAdvectSchemeType(fd,ADV_FROMM)); }

  // PetscCall(FDPDEEnthalpySetTimeStepSchemeType(fd,TS_NONE));
  if (par->ts_scheme ==0) { PetscCall(FDPDEEnthalpySetTimeStepSchemeType(fd,TS_FORWARD_EULER)); }
  if (par->ts_scheme ==1) { PetscCall(FDPDEEnthalpySetTimeStepSchemeType(fd,TS_BACKWARD_EULER)); }
  if (par->ts_scheme ==2) { PetscCall(FDPDEEnthalpySetTimeStepSchemeType(fd,TS_CRANK_NICHOLSON ));}

  PetscCall(FDPDEEnthalpySetTimestep(fd,par->dt)); 
  PetscCall(FDPDEEnthalpySetEnthalpyMethod(fd,Form_Enthalpy,enthalpy_method_description,usr));
  PetscCall(FDPDEEnthalpySetPotentialTemp(fd,Form_PotentialTemperature,usr));
  PetscCall(FDPDEView(fd));

  // Set initial conditions at t=0.05
  PetscCall(FDPDEGetDM(fd,&dm));
  PetscCall(FDPDEEnthalpyGetPrevSolution(fd,&xprev));
  PetscCall(Analytical_solution(dm,&xAnalytic,usr,par->t)); 
  PetscCall(VecCopy(xAnalytic,xprev)); 
  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_xprev_initial",par->fdir_out));
  PetscCall(DMStagViewBinaryPython(dm,xprev,fout));
  PetscCall(VecDestroy(&xAnalytic)); 

  // Set initial coefficient structure
  PetscCall(FDPDEGetCoefficient(fd,&dmcoeff,NULL));
  PetscCall(FDPDEEnthalpyGetPrevCoefficient(fd,&xcoeffprev));
  PetscCall(FormCoefficient(fd,dm,xprev,dmcoeff,xcoeffprev,usr));
  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_xcoeffprev_initial",par->fdir_out));
  PetscCall(DMStagViewBinaryPython(dmcoeff,xcoeffprev,fout));
  PetscCall(VecDestroy(&xcoeffprev));
  PetscCall(VecDestroy(&xprev));

  // set initial pressure
  PetscCall(FDPDEEnthalpyGetPressure(fd,&dmP,NULL));
  PetscCall(FDPDEEnthalpyGetPrevPressure(fd,&xPprev));
  PetscCall(VecSet(xPprev,-1.0));
  PetscCall(VecDestroy(&xPprev));

  // Time loop
  while ((par->t <= par->tmax) && (istep<par->tstep)) {
    PetscPrintf(PETSC_COMM_WORLD,"# TIMESTEP %d: \n",istep);

    // Update time
    par->t += par->dt;

    // update pressure
    PetscCall(FDPDEEnthalpyGetPressure(fd,NULL,&xP));
    PetscCall(VecSet(xP,istep));

    // Enthalpy Solver
    PetscCall(FDPDESolve(fd,NULL));
    PetscCall(FDPDEGetSolution(fd,&x));
    // PetscCall(MatView(fd->J,PETSC_VIEWER_STDOUT_WORLD));

    // Copy solution and coefficient to old
    PetscCall(FDPDEEnthalpyGetPrevSolution(fd,&xprev));
    PetscCall(VecCopy(x,xprev));
    PetscCall(VecDestroy(&xprev));

    PetscCall(FDPDEGetCoefficient(fd,&dmcoeff,&xcoeff));
    PetscCall(FDPDEEnthalpyGetPrevCoefficient(fd,&xcoeffprev));
    PetscCall(VecCopy(xcoeff,xcoeffprev));
    PetscCall(VecDestroy(&xcoeffprev));

    PetscCall(FDPDEEnthalpyGetPrevPressure(fd,&xPprev));
    PetscCall(VecCopy(xP,xPprev));
    PetscCall(VecDestroy(&xP));
    PetscCall(VecDestroy(&xPprev));

    // Output solution
    if (istep % par->tout == 0 ) {
      PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s_ts%1.3d",par->fdir_out,par->fname_out,istep));
      PetscCall(DMStagViewBinaryPython(dm,x,fout));

      PetscCall(FDPDEEnthalpyUpdateDiagnostics(fd,dm,x,&dmnew,&xnew)); 
      PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s_enthalpy_ts%1.3d",par->fdir_out,par->fname_out,istep));
      PetscCall(DMStagViewBinaryPython(dmnew,xnew,fout));
      PetscCall(DMDestroy(&dmnew));
      PetscCall(VecDestroy(&xnew)); 

      PetscCall(Analytical_solution(dm,&xAnalytic,usr,par->t)); 
      PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s_analytical_ts%1.3d",par->fdir_out,par->fname_out,istep));
      PetscCall(DMStagViewBinaryPython(dm,xAnalytic,fout));
      PetscCall(VecDestroy(&xAnalytic)); 
    }
    PetscCall(VecDestroy(&x));

    // increment timestep
    istep++;
  }

  // Destroy objects
  PetscCall(DMDestroy(&dm));
  PetscCall(DMDestroy(&dmP));
  PetscCall(FDPDEDestroy(&fd));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// Phase Diagram
// ---------------------------------------
EnthEvalErrorCode Form_Enthalpy(PetscScalar H,PetscScalar C[],PetscScalar P,PetscScalar *_T,PetscScalar *_phi,PetscScalar *CF,PetscScalar *CS,PetscInt ncomp, void *ctx) 
{
  // UsrData      *usr = (UsrData*) ctx;
  PetscInt     ii;
  PetscScalar  T, phi;

  T = H;
  phi = 1.0;

  for (ii = 0; ii<ncomp; ii++) { 
    CS[ii] = C[ii];
    CF[ii] = C[ii];
  }

  // assign pointers
  *_T = T;
  *_phi = phi;

  // error check
  ENTH_CHECK_PHI(phi);
  return(STATE_VALID);
}

PetscErrorCode Form_PotentialTemperature(PetscScalar T,PetscScalar P,PetscScalar *_TP, void *ctx) 
{
  PetscScalar  TP;
  PetscFunctionBeginUser;

  TP = T*1.0;
  *_TP = TP;

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// FormBCList
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormBCList"
PetscErrorCode FormBCList(DM dm, Vec x, DMStagBCList bclist, void *ctx)
{
  UsrData     *usr = (UsrData*)ctx;
  PetscInt    k,n_bc,*idx_bc, ii;
  PetscScalar *value_bc,*x_bc;
  BCType      *type_bc;
  PetscFunctionBeginUser;
  
  for (ii=0; ii<usr->par->ncomp; ii++) {
    // Down:
    PetscCall(DMStagBCListGetValues(bclist,'s','o',ii,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
    for (k=0; k<n_bc; k++) {
      value_bc[k] = 0.0;
      type_bc[k] = BC_DIRICHLET;
    }
    PetscCall(DMStagBCListInsertValues(bclist,'o',ii,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

    // Top:
    PetscCall(DMStagBCListGetValues(bclist,'n','o',ii,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
    for (k=0; k<n_bc; k++) {
      value_bc[k] = 0.0;
      type_bc[k] = BC_DIRICHLET;
    }
    PetscCall(DMStagBCListInsertValues(bclist,'o',ii,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

    // Left:
    PetscCall(DMStagBCListGetValues(bclist,'w','o',ii,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
    for (k=0; k<n_bc; k++) {
      value_bc[k] = 0.0;
      type_bc[k] = BC_DIRICHLET;
    }
    PetscCall(DMStagBCListInsertValues(bclist,'o',ii,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

    // Right:
    PetscCall(DMStagBCListGetValues(bclist,'e','o',ii,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
    for (k=0; k<n_bc; k++) {
      value_bc[k] = 0.0;
      type_bc[k] = BC_DIRICHLET;
    }
    PetscCall(DMStagBCListInsertValues(bclist,'o',ii,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  }

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
  PetscInt       i, j, sx, sz, nx, nz;
  Vec            coefflocal;
  PetscScalar    ***c;
  PetscFunctionBeginUser;

  // Get domain corners
  PetscCall(DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 
  PetscCall(DMCreateLocalVector(dmcoeff, &coefflocal)); 
  PetscCall(DMStagVecGetArray(dmcoeff, coefflocal, &c)); 
  
  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {

      { // ELEMENT
        DMStagStencil point;
        PetscInt      idx;

        point.i = i; point.j = j; point.loc = ELEMENT;  
        point.c = COEFF_A1; PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
        c[j][i][idx] = 0.0;

        point.c = COEFF_B1; PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
        c[j][i][idx] = 0.0;

        point.c = COEFF_D1; PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
        c[j][i][idx] = 0.0;

        point.c = COEFF_A2; PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
        c[j][i][idx] = 0.0;

        point.c = COEFF_B2; PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
        c[j][i][idx] = 0.0;

        point.c = COEFF_D2; PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
        c[j][i][idx] = 0.0;
      }

      { // FACES
        DMStagStencil point[4];
        PetscInt      ii, idx;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT; 
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN; 
        point[3].i = i; point[3].j = j; point[3].loc = UP; 

        for (ii = 0; ii < 4; ii++) {
          point[ii].c = COEFF_C1; PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] = -usr->par->k;

          point[ii].c = COEFF_C2; PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] = -usr->par->k; 

          point[ii].c = COEFF_v; PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] = 0.0;

          point[ii].c = COEFF_vf; PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] = 0.0;

          point[ii].c = COEFF_vs; PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] = 0.0;
        }
      }
    }
  }

  // Restore arrays, local vectors
  PetscCall(DMStagVecRestoreArray(dmcoeff,coefflocal,&c));
  PetscCall(DMLocalToGlobalBegin(dmcoeff,coefflocal,INSERT_VALUES,coeff)); 
  PetscCall(DMLocalToGlobalEnd  (dmcoeff,coefflocal,INSERT_VALUES,coeff)); 
  
  PetscCall(VecDestroy(&coefflocal)); 

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// Create analytical solution
// ---------------------------------------
PetscErrorCode Analytical_solution(DM dm,Vec *_x, void *ctx, PetscScalar t)
{
  UsrData       *usr = (UsrData*) ctx;
  PetscInt       ii, i, j, sx, sz, nx, nz, idx, icenter;
  PetscScalar    ***xx;
  PetscScalar    **coordx,**coordz;
  Vec            x, xlocal;
  PetscFunctionBeginUser;

  // Create local and global vector associated with DM
  PetscCall(DMCreateGlobalVector(dm, &x     )); 
  PetscCall(DMCreateLocalVector (dm, &xlocal)); 
  PetscCall(DMStagVecGetArray(dm,xlocal,&xx)); 

  // Get data for dm
  PetscCall(DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 
  PetscCall(DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter)); 

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      PetscScalar  xp, zp, Q;
      xp = coordx[i][icenter];
      zp = coordz[j][icenter];
      Q = analytical_solution(xp,zp,t,usr->par->k);

      // enthalpy
      PetscCall(DMStagGetLocationSlot(dm, ELEMENT, 0, &idx));
      xx[j][i][idx] = Q;

      // composition
      for (ii = 0; ii <usr->par->ncomp-1; ii++) {
        PetscCall(DMStagGetLocationSlot(dm, ELEMENT, ii+1, &idx));
        xx[j][i][idx] = Q;
      }
    }
  }

  // Restore arrays
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));
  PetscCall(DMStagVecRestoreArray(dm,xlocal,&xx)); 

  // Map local to global
  PetscCall(DMLocalToGlobalBegin(dm,xlocal,INSERT_VALUES,x)); 
  PetscCall(DMLocalToGlobalEnd  (dm,xlocal,INSERT_VALUES,x)); 

  PetscCall(VecDestroy(&xlocal)); 

  // Assign pointers
  *_x  = x;
  
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
  PetscCall(PetscBagRegisterInt(bag, &par->nx, 20, "nx", "Element count in the x-dir [-]")); 
  PetscCall(PetscBagRegisterInt(bag, &par->nz, 20, "nz", "Element count in the z-dir [-]")); 

  PetscCall(PetscBagRegisterScalar(bag, &par->xmin, -0.5, "xmin", "Start coordinate of domain in x-dir [-]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->zmin, -0.5, "zmin", "Start coordinate of domain in z-dir [-]")); 

  PetscCall(PetscBagRegisterScalar(bag, &par->L, 1.0, "L", "Length of domain in x-dir [-]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->H, 1.0, "H", "Height of domain in z-dir [-]")); 

  // Physical and material parameters
  PetscCall(PetscBagRegisterScalar(bag, &par->k, 0.1, "k", "Diffusivity")); 

  // Time stepping and advection parameters
  PetscCall(PetscBagRegisterInt(bag, &par->ts_scheme,2, "ts_scheme", "Time stepping scheme 0-forward euler, 1-backward euler, 2-crank-nicholson")); 
  PetscCall(PetscBagRegisterInt(bag, &par->adv_scheme,2, "adv_scheme", "Advection scheme 0-upwind, 1-upwind2, 2-fromm")); 
  PetscCall(PetscBagRegisterInt(bag, &par->ncomp,2, "ncomp", "Number of components of phase diagram")); 

  PetscCall(PetscBagRegisterInt(bag, &par->tout,1, "tout", "Output every tout time step")); 
  PetscCall(PetscBagRegisterInt(bag, &par->tstep,10, "tstep", "Maximum no of time steps")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->tmax,0.1, "tmax", "Maximum time [-]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->dtmax,0.01, "dtmax", "Maximum time step [-]")); 
  par->t = 0.05;
  par->dt = par->dtmax;

  // Input/output 
  PetscCall(PetscBagRegisterString(bag,&par->fname_out,FNAME_LENGTH,"out_2d_diff_enth","output_file","Name for output file, set with: -output_file <filename>")); 
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
  PetscCall(PetscOptionsGetAll(NULL, &opts)); 

  // Print header and petsc options
  PetscPrintf(usr->comm,"# --------------------------------------- #\n");
  PetscPrintf(usr->comm,"# 2-D Diffusion (Enthalpy): %s \n",&(date[0]));
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
  PetscCall(PetscOptionsInsert(PETSC_NULL,&argc,&argv,NULL)); 

  // Input user parameters and print
  PetscCall(InputParameters(&usr)); 
  PetscCall(InputPrintData(usr)); 

  // Numerical solution using the FD pde object
  PetscCall(PetscTime(&start_time)); 
  PetscCall(Numerical_solution(usr)); 
  PetscCall(PetscTime(&end_time)); 
  PetscPrintf(PETSC_COMM_WORLD,"# Runtime: %g (sec) \n", end_time - start_time);
  PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");

  // Destroy objects
  PetscCall(PetscBagDestroy(&usr->bag)); 
  PetscCall(PetscFree(usr)); 

  // Finalize main
  PetscCall(PetscFinalize());
  return 0;
}