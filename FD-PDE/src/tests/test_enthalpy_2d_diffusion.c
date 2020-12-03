// ---------------------------------------
// 2D diffusion to test the enthalpy implementation
// run: ./test_enthalpy_2d_diffusion.app -pc_type lu -pc_factor_mat_solver_type umfpack -snes_monitor -log_view
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
#include "../../src/fdpde_enthalpy.h"
#include "../../src/dmstagoutput.h"

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
  PetscInt       ts_scheme, adv_scheme, tout, tstep, energy, ncomp;
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
PetscErrorCode ApplyBC_Enthalpy(DM,Vec,PetscScalar***,void*);
PetscErrorCode Form_Enthalpy(PetscScalar,PetscScalar[],PetscScalar,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscInt,void*); 
PetscErrorCode Analytical_solution(DM,Vec*,void*,PetscScalar);

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
"  Input: H, C \n"
"  Output: TP = T  = H, \n"
"          Cf = Cs = C, \n"
"          phi = 1.0, P = 0 \n";

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
  PetscErrorCode ierr;
  PetscFunctionBegin;

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
  ierr = FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_ENTHALPY,&fd);CHKERRQ(ierr);
  ierr = FDPDEEnthalpySetNumberComponentsPhaseDiagram(fd,par->ncomp);CHKERRQ(ierr);
  ierr = FDPDESetUp(fd);CHKERRQ(ierr);

  if(par->energy==0) { ierr = FDPDEEnthalpySetEnergyPrimaryVariable(fd,'T');CHKERRQ(ierr);}
  if(par->energy==1) { ierr = FDPDEEnthalpySetEnergyPrimaryVariable(fd,'H');CHKERRQ(ierr);}

  ierr = FDPDESetFunctionBCList(fd,FormBCList,bc_description,usr); CHKERRQ(ierr);
  ierr = FDPDEEnthalpySetUserBC(fd,ApplyBC_Enthalpy,usr);CHKERRQ(ierr);
  ierr = FDPDESetFunctionCoefficient(fd,FormCoefficient,coeff_description,usr); CHKERRQ(ierr);

  if (par->adv_scheme==0) { ierr = FDPDEEnthalpySetAdvectSchemeType(fd,ADV_UPWIND);CHKERRQ(ierr); }
  if (par->adv_scheme==1) { ierr = FDPDEEnthalpySetAdvectSchemeType(fd,ADV_UPWIND2);CHKERRQ(ierr); }
  if (par->adv_scheme==2) { ierr = FDPDEEnthalpySetAdvectSchemeType(fd,ADV_FROMM);CHKERRQ(ierr); }

  // ierr = FDPDEEnthalpySetTimeStepSchemeType(fd,TS_NONE);CHKERRQ(ierr);
  if (par->ts_scheme ==0) { ierr = FDPDEEnthalpySetTimeStepSchemeType(fd,TS_FORWARD_EULER);CHKERRQ(ierr); }
  if (par->ts_scheme ==1) { ierr = FDPDEEnthalpySetTimeStepSchemeType(fd,TS_BACKWARD_EULER);CHKERRQ(ierr); }
  if (par->ts_scheme ==2) { ierr = FDPDEEnthalpySetTimeStepSchemeType(fd,TS_CRANK_NICHOLSON );CHKERRQ(ierr);}

  ierr = FDPDEEnthalpySetTimestep(fd,par->dt); CHKERRQ(ierr);
  ierr = FDPDEEnthalpySetEnthalpyMethod(fd,Form_Enthalpy,enthalpy_method_description,usr);CHKERRQ(ierr);
  ierr = FDPDEView(fd);CHKERRQ(ierr);

  // Set initial conditions at t=0.05
  ierr = FDPDEGetDM(fd,&dm);CHKERRQ(ierr);
  ierr = FDPDEEnthalpyGetPrevSolution(fd,&xprev);CHKERRQ(ierr);
  ierr = Analytical_solution(dm,&xAnalytic,usr,par->t); CHKERRQ(ierr);
  ierr = VecCopy(xAnalytic,xprev); CHKERRQ(ierr);
  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_xprev_initial",par->fdir_out);
  ierr = DMStagViewBinaryPython(dm,xprev,fout);CHKERRQ(ierr);
  ierr = VecDestroy(&xAnalytic); CHKERRQ(ierr);

  // Set initial coefficient structure
  ierr = FDPDEGetCoefficient(fd,&dmcoeff,NULL);CHKERRQ(ierr);
  ierr = FDPDEEnthalpyGetPrevCoefficient(fd,&xcoeffprev);CHKERRQ(ierr);
  ierr = FormCoefficient(fd,dm,xprev,dmcoeff,xcoeffprev,usr);CHKERRQ(ierr);
  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/out_xcoeffprev_initial",par->fdir_out);
  ierr = DMStagViewBinaryPython(dmcoeff,xcoeffprev,fout);CHKERRQ(ierr);
  ierr = VecDestroy(&xcoeffprev);CHKERRQ(ierr);
  ierr = VecDestroy(&xprev);CHKERRQ(ierr);

  // set initial pressure
  ierr = FDPDEEnthalpyGetPressure(fd,&dmP,NULL);CHKERRQ(ierr);
  ierr = FDPDEEnthalpyGetPrevPressure(fd,&xPprev);CHKERRQ(ierr);
  ierr = VecSet(xPprev,-1.0);CHKERRQ(ierr);
  ierr = VecDestroy(&xPprev);CHKERRQ(ierr);

  // Time loop
  while ((par->t <= par->tmax) && (istep<par->tstep)) {
    PetscPrintf(PETSC_COMM_WORLD,"# TIMESTEP %d: \n",istep);

    // Update time
    par->t += par->dt;

    // update pressure
    ierr = FDPDEEnthalpyGetPressure(fd,NULL,&xP);CHKERRQ(ierr);
    ierr = VecSet(xP,istep);CHKERRQ(ierr);

    // Enthalpy Solver
    ierr = FDPDESolve(fd,NULL);CHKERRQ(ierr);
    ierr = FDPDEGetSolution(fd,&x);CHKERRQ(ierr);
    // ierr = MatView(fd->J,PETSC_VIEWER_STDOUT_WORLD);

    // Copy solution and coefficient to old
    ierr = FDPDEEnthalpyGetPrevSolution(fd,&xprev);CHKERRQ(ierr);
    ierr = VecCopy(x,xprev);CHKERRQ(ierr);
    ierr = VecDestroy(&xprev);CHKERRQ(ierr);

    ierr = FDPDEGetCoefficient(fd,&dmcoeff,&xcoeff);CHKERRQ(ierr);
    ierr = FDPDEEnthalpyGetPrevCoefficient(fd,&xcoeffprev);CHKERRQ(ierr);
    ierr = VecCopy(xcoeff,xcoeffprev);CHKERRQ(ierr);
    ierr = VecDestroy(&xcoeffprev);CHKERRQ(ierr);

    ierr = FDPDEEnthalpyGetPrevPressure(fd,&xPprev);CHKERRQ(ierr);
    ierr = VecCopy(xP,xPprev);CHKERRQ(ierr);
    ierr = VecDestroy(&xP);CHKERRQ(ierr);
    ierr = VecDestroy(&xPprev);CHKERRQ(ierr);

    // Output solution
    if (istep % par->tout == 0 ) {
      ierr = PetscSNPrintf(fout,sizeof(fout),"%s/%s_ts%1.3d",par->fdir_out,par->fname_out,istep);
      ierr = DMStagViewBinaryPython(dm,x,fout);CHKERRQ(ierr);

      ierr = FDPDEEnthalpyUpdateDiagnostics(fd,dm,x,&dmnew,&xnew); CHKERRQ(ierr);
      ierr = PetscSNPrintf(fout,sizeof(fout),"%s/%s_enthalpy_ts%1.3d",par->fdir_out,par->fname_out,istep);
      ierr = DMStagViewBinaryPython(dmnew,xnew,fout);CHKERRQ(ierr);
      ierr = DMDestroy(&dmnew);CHKERRQ(ierr);
      ierr = VecDestroy(&xnew); CHKERRQ(ierr);

      ierr = Analytical_solution(dm,&xAnalytic,usr,par->t); CHKERRQ(ierr);
      ierr = PetscSNPrintf(fout,sizeof(fout),"%s/%s_analytical_ts%1.3d",par->fdir_out,par->fname_out,istep);
      ierr = DMStagViewBinaryPython(dm,xAnalytic,fout);CHKERRQ(ierr);
      ierr = VecDestroy(&xAnalytic); CHKERRQ(ierr);
    }
    ierr = VecDestroy(&x);CHKERRQ(ierr);

    // increment timestep
    istep++;
  }

  // Destroy objects
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = DMDestroy(&dmP);CHKERRQ(ierr);
  ierr = FDPDEDestroy(&fd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// ---------------------------------------
// Phase Diagram
// ---------------------------------------
PetscErrorCode Form_Enthalpy(PetscScalar H,PetscScalar C[],PetscScalar P,PetscScalar *_TP,PetscScalar *_T,PetscScalar *_phi,PetscScalar *CF,PetscScalar *CS,PetscInt ncomp, void *ctx) 
{
  // UsrData      *usr = (UsrData*) ctx;
  PetscInt     ii;
  PetscScalar  T, phi, TP;
  PetscFunctionBegin;

  TP = H;
  T  = TP;
  phi = 1.0;

  for (ii = 0; ii<ncomp-1; ii++) { 
    CS[ii] = C[ii];
    CF[ii] = C[ii];
  }

  // assign pointers
  *_TP = TP;
  *_T = T;
  *_phi = phi;

  PetscFunctionReturn(0);
}

// ---------------------------------------
// ApplyBC_Enthalpy
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "ApplyBC_Enthalpy"
PetscErrorCode ApplyBC_Enthalpy(DM dm, Vec x, PetscScalar ***ff, void *ctx)
{
  UsrData     *usr = (UsrData*)ctx;
  PetscInt    ii,i,j,sx,sz,nx,nz,Nx,Nz,iT,iC;
  Vec          xlocal;
  PetscScalar ***xx;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = DMStagGetGlobalSizes(dm,&Nx,&Nz,NULL);CHKERRQ(ierr); 
  ierr = DMStagGetLocationSlot(dm,DMSTAG_ELEMENT,0,&iT); CHKERRQ(ierr);

  ierr = DMGetLocalVector(dm, &xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm, x, INSERT_VALUES, xlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm, xlocal, &xx); CHKERRQ(ierr);

  // Vertical boundaries
  for (j = sz; j<sz+nz; j++) {
    i = 0;
    ff[j][i][iT] = xx[j][i][iT] - 0.0;
    i = Nx-1;
    ff[j][i][iT] = xx[j][i][iT] - 0.0;

    for (ii=0; ii<usr->par->ncomp-1; ii++) {
      ierr = DMStagGetLocationSlot(dm,DMSTAG_ELEMENT,ii+1,&iC); CHKERRQ(ierr);
      i = 0;
      ff[j][i][iC] = xx[j][i][iC] - 0.0;
      i = Nx-1;
      ff[j][i][iC] = xx[j][i][iC] - 0.0;
    }
  }

  // Horizontal boundaries
  for (i = sx; i<sx+nx; i++) {
    j = 0;
    ff[j][i][iT] = xx[j][i][iT] - 0.0;
    j = Nz-1;
    ff[j][i][iT] = xx[j][i][iT] - 0.0;

    for (ii=0; ii<usr->par->ncomp-1; ii++) {
      ierr = DMStagGetLocationSlot(dm,DMSTAG_ELEMENT,ii+1,&iC); CHKERRQ(ierr);
      j = 0;
      ff[j][i][iC] = xx[j][i][iC] - 0.0;
      j = Nz-1;
      ff[j][i][iC] = xx[j][i][iC] - 0.0;
    }
  }

  ierr = DMStagVecRestoreArray(dm, xlocal, &xx); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &xlocal); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// FormBCList
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormBCList"
PetscErrorCode FormBCList(DM dm, Vec x, DMStagBCList bclist, void *ctx)
{
  // UsrData     *usr = (UsrData*)ctx;
  PetscInt    k,n_bc,*idx_bc;
  PetscScalar *value_bc,*x_bc;
  BCType      *type_bc;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  
  // Left: T = 0.0
  ierr = DMStagBCListGetValues(bclist,'w','o',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // RIGHT: T = 0.0
  ierr = DMStagBCListGetValues(bclist,'e','o',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

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
  Params         *par;
  PetscInt       i, j, sx, sz, nx, nz;
  Vec            coefflocal;
  PetscScalar    ***c;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  par = usr->par;
  // Get domain corners
  ierr = DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dmcoeff, &coefflocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dmcoeff, coefflocal, &c); CHKERRQ(ierr);
  
  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {

      { // ELEMENT
        DMStagStencil point;
        PetscInt      idx;

        point.i = i; point.j = j; point.loc = ELEMENT;  
        point.c = COEFF_A1; ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = 0.0;

        point.c = COEFF_B1; ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = 0.0;

        point.c = COEFF_D1; ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = 0.0;

        point.c = COEFF_A2; ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = 0.0;

        point.c = COEFF_B2; ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = 0.0;

        point.c = COEFF_D2; ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
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
          point[ii].c = COEFF_C1; ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = -usr->par->k;

          point[ii].c = COEFF_C2; ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = -usr->par->k; 

          point[ii].c = COEFF_v; ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = 0.0;

          point[ii].c = COEFF_vf; ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = 0.0;

          point[ii].c = COEFF_vs; ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = 0.0;
        }
      }
    }
  }

  // Restore arrays, local vectors
  ierr = DMStagVecRestoreArray(dmcoeff,coefflocal,&c);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  
  ierr = VecDestroy(&coefflocal); CHKERRQ(ierr);

  PetscFunctionReturn(0);
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
  PetscErrorCode ierr;
  PetscFunctionBegin;

  // Create local and global vector associated with DM
  ierr = DMCreateGlobalVector(dm, &x     ); CHKERRQ(ierr);
  ierr = DMCreateLocalVector (dm, &xlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm,xlocal,&xx); CHKERRQ(ierr);

  // Get data for dm
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter);CHKERRQ(ierr); 

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      PetscScalar  xp, zp, Q;
      xp = coordx[i][icenter];
      zp = coordz[j][icenter];
      Q = analytical_solution(xp,zp,t,usr->par->k);

      // enthalpy
      ierr = DMStagGetLocationSlot(dm, ELEMENT, 0, &idx);CHKERRQ(ierr);
      xx[j][i][idx] = Q;

      // composition
      for (ii = 0; ii <usr->par->ncomp-1; ii++) {
        ierr = DMStagGetLocationSlot(dm, ELEMENT, ii+1, &idx);CHKERRQ(ierr);
        xx[j][i][idx] = Q;
      }
    }
  }

  // Restore arrays
  ierr = DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArray(dm,xlocal,&xx); CHKERRQ(ierr);

  // Map local to global
  ierr = DMLocalToGlobalBegin(dm,xlocal,INSERT_VALUES,x); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,xlocal,INSERT_VALUES,x); CHKERRQ(ierr);

  ierr = VecDestroy(&xlocal); CHKERRQ(ierr);

  // Assign pointers
  *_x  = x;
  
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
  ierr = PetscBagRegisterInt(bag, &par->nx, 20, "nx", "Element count in the x-dir [-]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->nz, 20, "nz", "Element count in the z-dir [-]"); CHKERRQ(ierr);

  ierr = PetscBagRegisterScalar(bag, &par->xmin, -0.5, "xmin", "Start coordinate of domain in x-dir [-]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->zmin, -0.5, "zmin", "Start coordinate of domain in z-dir [-]"); CHKERRQ(ierr);

  ierr = PetscBagRegisterScalar(bag, &par->L, 1.0, "L", "Length of domain in x-dir [-]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->H, 1.0, "H", "Height of domain in z-dir [-]"); CHKERRQ(ierr);

  // Physical and material parameters
  ierr = PetscBagRegisterScalar(bag, &par->k, 0.1, "k", "Diffusivity"); CHKERRQ(ierr);

  // Time stepping and advection parameters
  ierr = PetscBagRegisterInt(bag, &par->ts_scheme,2, "ts_scheme", "Time stepping scheme 0-forward euler, 1-backward euler, 2-crank-nicholson"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->adv_scheme,2, "adv_scheme", "Advection scheme 0-upwind, 1-upwind2, 2-fromm"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->energy,0, "energy", "Energy variable: 0-T, 1-H"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->ncomp,2, "ncomp", "Number of components of phase diagram"); CHKERRQ(ierr);

  ierr = PetscBagRegisterInt(bag, &par->tout,1, "tout", "Output every tout time step"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->tstep,10, "tstep", "Maximum no of time steps"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->tmax,0.1, "tmax", "Maximum time [-]"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->dtmax,0.01, "dtmax", "Maximum time step [-]"); CHKERRQ(ierr);
  par->t = 0.05;
  par->dt = par->dtmax;

  // Input/output 
  ierr = PetscBagRegisterString(bag,&par->fname_out,FNAME_LENGTH,"out_2d_diff_enth","output_file","Name for output file, set with: -output_file <filename>"); CHKERRQ(ierr);
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
  ierr = PetscOptionsGetAll(NULL, &opts); CHKERRQ(ierr);

  // Print header and petsc options
  PetscPrintf(usr->comm,"# --------------------------------------- #\n");
  PetscPrintf(usr->comm,"# 2-D Diffusion (Enthalpy): %s \n",&(date[0]));
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
  ierr = InputPrintData(usr); CHKERRQ(ierr);

  // Numerical solution using the FD pde object
  ierr = PetscTime(&start_time); CHKERRQ(ierr);
  ierr = Numerical_solution(usr); CHKERRQ(ierr);
  ierr = PetscTime(&end_time); CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"# Runtime: %g (sec) \n", end_time - start_time);
  PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");

  // Destroy objects
  ierr = PetscBagDestroy(&usr->bag); CHKERRQ(ierr);
  ierr = PetscFree(usr); CHKERRQ(ierr);

  // Finalize main
  ierr = PetscFinalize();
  return ierr;
}