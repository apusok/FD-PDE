// ---------------------------------------
// 1D solidification problem of an  initially  liquid  semi-infinite  slab
// Equations (non-dimensional): dH/dt-div^2T=0, H=T+phi/St 
// This problem is formulated with T as primary energy variable. Composition is kept constant.
// We use H as primary variable here though.
// run: ./test_enthalpy_1d_solidification_TC_ -pc_type lu -pc_factor_mat_solver_type umfpack -pc_factor_mat_ordering_type external -snes_monitor -log_view
// python output: /python/test_enthalpy_1d_solidification_TC.py
// ---------------------------------------
static char help[] = "1D Solidification problem using the Enthalpy Method\n\n";

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

#include "../src/fdpde_enthalpy.h"

// ---------------------------------------
// Application Context
// ---------------------------------------
#define FNAME_LENGTH 200

// parameters (bag)
typedef struct {
  PetscInt       nx, nz;
  PetscScalar    L, H;
  PetscScalar    xmin, zmin;
  PetscScalar    rho, cp, k, La, T0, Tb, Tm, Ts, dtmax, tmax, tstart, DT, C0, beta;
  PetscInt       ts_scheme, adv_scheme, tout, tstep;
  PetscScalar    scal_h, scal_t;
  PetscScalar    nd_t, nd_dt, nd_dtmax, nd_tmax, St, kappa, nd_T0, nd_Tb, nd_Tm, nd_Ts;
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
PetscErrorCode Initial_solution(DM,Vec,void*);
PetscErrorCode FormCoefficient(FDPDE, DM, Vec, DM, Vec, void*);
PetscErrorCode FormBCList(DM, Vec, DMStagBCList, void*);
PetscErrorCode ApplyBC_Enthalpy(DM,Vec,PetscScalar***,void*);
EnthEvalErrorCode Form_Enthalpy(PetscScalar,PetscScalar[],PetscScalar,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscInt,void*); 

const char coeff_description[] =
"  << ENTHALPY Coefficients >> \n"
"  A1 = 0, B1 = 0, C1 = -1, D1 = 0  \n"
"  A2 = 0, B2 = 0, C2 =  0, D2 = 0  \n"
"  v = [0,0], vs = [0,0], vf = [0,0] \n";

const char bc_description[] =
"  << ENTHALPY BCs >> \n"
"  TEMP: LEFT: H = Hb, RIGHT: H = H0, DOWN, UP \n"
"  COMP: LEFT, RIGHT, DOWN, UP: C = C0 \n";

const char enthalpy_method_description[] =
"  << ENTHALPY METHOD >> \n"
"  Input: H, C, P \n"
"  Output: H = T + 1/St*phi, \n"
"          Cf = Cs = C (dummy), \n"
"          phi = 10^4(T*DT+Tm)+1001, if Ts<T<=0\n";

// static PetscScalar analytical_temp(PetscScalar x, PetscScalar t, PetscScalar beta) { 
//   PetscScalar xfront, T;
//   xfront = 2.0*beta*PetscSqrtScalar(t);
//   if (x>xfront) T = 0.0;
//   else T = erf(x/2.0/sqrt(t))/erf(beta)-1.0; 
//   return T; 
// }

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
  DM             dm, dmcoeff, dmnew;
  Vec            x, xprev, xcoeff, xcoeffprev, xnew;
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
  PetscCall(FDPDESetUp(fd));

  PetscCall(FDPDESetFunctionBCList(fd,FormBCList,bc_description,usr)); 
  PetscCall(FDPDEEnthalpySetUserBC(fd,ApplyBC_Enthalpy,usr));
  PetscCall(FDPDESetFunctionCoefficient(fd,FormCoefficient,coeff_description,usr)); 

  if (par->adv_scheme==0) { PetscCall(FDPDEEnthalpySetAdvectSchemeType(fd,ADV_UPWIND)); }
  if (par->adv_scheme==1) { PetscCall(FDPDEEnthalpySetAdvectSchemeType(fd,ADV_UPWIND2)); }
  if (par->adv_scheme==2) { PetscCall(FDPDEEnthalpySetAdvectSchemeType(fd,ADV_FROMM)); }

  if (par->ts_scheme ==0) { PetscCall(FDPDEEnthalpySetTimeStepSchemeType(fd,TS_FORWARD_EULER)); }
  if (par->ts_scheme ==1) { PetscCall(FDPDEEnthalpySetTimeStepSchemeType(fd,TS_BACKWARD_EULER)); }
  if (par->ts_scheme ==2) { PetscCall(FDPDEEnthalpySetTimeStepSchemeType(fd,TS_CRANK_NICHOLSON ));}

  PetscCall(FDPDEEnthalpySetEnthalpyMethod(fd,Form_Enthalpy,enthalpy_method_description,usr));

  // Set initial temperature profile H(T) = H(T0) (t=0)
  PetscCall(FDPDEGetDM(fd,&dm));
  PetscCall(FDPDEEnthalpyGetPrevSolution(fd,&xprev));
  PetscCall(Initial_solution(dm,xprev,usr)); 
  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_xprev_initial",par->fdir_out));
  PetscCall(DMStagViewBinaryPython(dm,xprev,fout));

  // Set initial coefficient structure
  PetscCall(FDPDEGetCoefficient(fd,&dmcoeff,NULL));
  PetscCall(FDPDEEnthalpyGetPrevCoefficient(fd,&xcoeffprev));
  PetscCall(FormCoefficient(fd,dm,xprev,dmcoeff,xcoeffprev,usr));
  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_xcoeffprev_initial",par->fdir_out));
  PetscCall(DMStagViewBinaryPython(dmcoeff,xcoeffprev,fout));
  PetscCall(VecDestroy(&xcoeffprev));
  PetscCall(VecDestroy(&xprev));

  // Time loop
  while ((par->nd_t <= par->nd_tmax) && (istep<par->tstep)) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"# TIMESTEP %d: \n",istep));

    // Update time
    par->nd_t += par->nd_dt;
    PetscCall(FDPDEEnthalpySetTimestep(fd,par->nd_dt)); 

    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"# >>> [nd,dim] time = [%1.6e,%1.6e] dt = [%1.6e,%1.6e] \n",par->nd_t,par->nd_t*par->scal_t,par->nd_dt,par->nd_dt*par->scal_t));

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

    // Output solution
    if (istep % par->tout == 0 ) {
      PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s_HC_ts%1.3d",par->fdir_out,par->fname_out,istep));
      PetscCall(DMStagViewBinaryPython(dm,x,fout));

      PetscCall(FDPDEEnthalpyUpdateDiagnostics(fd,dm,x,&dmnew,&xnew)); 
      PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s_enthalpy_ts%1.3d",par->fdir_out,par->fname_out,istep));
      PetscCall(DMStagViewBinaryPython(dmnew,xnew,fout));
      PetscCall(DMDestroy(&dmnew));
      PetscCall(VecDestroy(&xnew)); 
    }
    PetscCall(VecDestroy(&x));

    // increment timestep
    istep++;
  }

  // Destroy objects
  PetscCall(DMDestroy(&dm));
  PetscCall(FDPDEDestroy(&fd));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// Phase Diagram - transformed from T,C
// enthalpy_method(H,C,P,&T,&phi,CF,CS,ncomp,user);
// ---------------------------------------
EnthEvalErrorCode Form_Enthalpy(PetscScalar H,PetscScalar C[],PetscScalar P,PetscScalar *_T,PetscScalar *_phi,PetscScalar *CF,PetscScalar *CS,PetscInt ncomp, void *ctx) 
{
  UsrData      *usr = (UsrData*) ctx;
  PetscInt     ii;
  PetscScalar  Tsol, Tliq, Hsol, Hliq, T, phi=0.0, DT;

  // Solidus and liquidus
  Tsol = usr->par->nd_Ts;
  Tliq = usr->par->nd_Tm;

  Hsol = Tsol;
  Hliq = Tliq+1.0/usr->par->St;

  if (H <= Hsol) {
    phi = 0.0;
    for (ii = 0; ii<ncomp; ii++) { 
      CS[ii] = C[ii];
      CF[ii] = 0.0;
    }
  } else if (H >= Hliq) {
    phi = 1.0;
    for (ii = 0; ii<ncomp; ii++) { 
      CS[ii] = 0.0;
      CF[ii] = C[ii];
    }
  } else {
    for (ii = 0; ii<ncomp; ii++) { 
      CS[ii] = usr->par->T0;
      CF[ii] = usr->par->T0;
    }
    DT = usr->par->Tm-usr->par->Tb;
    phi = (1.0e4*(H*DT+usr->par->Tm)+1001)/(1.0+1.0e4*DT/usr->par->St);
  }

  // other enthalpy variables
  T = H-phi/usr->par->St;

  // assign pointers
  *_T = T;
  *_phi = phi;

  ENTH_CHECK_PHI(phi);
  return(STATE_VALID);
}

// ---------------------------------------
// ApplyBC_Enthalpy
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "ApplyBC_Enthalpy"
PetscErrorCode ApplyBC_Enthalpy(DM dm, Vec x, PetscScalar ***ff, void *ctx)
{
  UsrData     *usr = (UsrData*)ctx;
  PetscInt    i,j,sx,sz,nx,nz,Nx,Nz,iC;
  Vec          xlocal;
  PetscScalar ***xx;
  PetscFunctionBeginUser;
  
  PetscCall(DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 
  PetscCall(DMStagGetGlobalSizes(dm,&Nx,&Nz,NULL)); 
  PetscCall(DMStagGetLocationSlot(dm,DMSTAG_ELEMENT,1,&iC)); 

  PetscCall(DMGetLocalVector(dm, &xlocal)); 
  PetscCall(DMGlobalToLocal (dm, x, INSERT_VALUES, xlocal)); 
  PetscCall(DMStagVecGetArray(dm, xlocal, &xx)); 

  // Entire domain C = C0 (dummy variable)
  for (j = sz; j<sz+nz; j++) {
    for (i = sx; i<sx+nx; i++) {
      ff[j][i][iC] = xx[j][i][iC] - usr->par->C0;
    }
  }

  PetscCall(DMStagVecRestoreArray(dm, xlocal, &xx)); 
  PetscCall(DMRestoreLocalVector(dm, &xlocal)); 

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
  PetscInt    k,n_bc,*idx_bc;
  PetscScalar *value_bc,*x_bc, H_left, H_right;
  BCType      *type_bc;
  PetscFunctionBeginUser;
  
  // Left: H(T,phi) = H(Tb,0.0)
  H_left = usr->par->nd_Tb;

  // RIGHT: H(T,phi) = H(T0,1.0)
  H_right = usr->par->nd_Tb + 1.0/usr->par->St;

  // Left: T = Tb
  PetscCall(DMStagBCListGetValues(bclist,'w','o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = H_left;
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // RIGHT: T = T0
  PetscCall(DMStagBCListGetValues(bclist,'e','o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = H_right;
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// FormCoefficient
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormCoefficient"
PetscErrorCode FormCoefficient(FDPDE fd, DM dm, Vec x, DM dmcoeff, Vec coeff, void *ctx)
{
  // UsrData        *usr = (UsrData*)ctx;
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
          c[j][i][idx] = -1.0;

          point[ii].c = COEFF_C2; PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] = 0.0;

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
// Create initial solution - assume molten initial state
// ---------------------------------------
PetscErrorCode Initial_solution(DM dm,Vec x, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  PetscInt       i, j, sx, sz, nx, nz, idx, icenter;
  PetscScalar    ***xx;
  PetscScalar    **coordx,**coordz;
  Vec            xlocal;
  PetscFunctionBeginUser;

  // Create local and global vector associated with DM
  PetscCall(DMCreateLocalVector (dm, &xlocal)); 
  PetscCall(DMStagVecGetArray(dm,xlocal,&xx)); 

  // Get data for dm
  PetscCall(DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 
  PetscCall(DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter)); 

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      DMStagStencil  point;
      PetscScalar    th, phi;

      point.i = i; point.j = j; point.loc = ELEMENT; point.c = 0; 
      th = usr->par->nd_T0;
      phi = 1.0;

      PetscCall(DMStagGetLocationSlot(dm, point.loc, point.c, &idx));
      xx[j][i][idx] = th+1.0/usr->par->St*phi;

      // composition = Ci
      point.c = 1; PetscCall(DMStagGetLocationSlot(dm, point.loc, point.c, &idx));
      xx[j][i][idx] = usr->par->C0;
    }
  }

  // Restore arrays
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));
  PetscCall(DMStagVecRestoreArray(dm,xlocal,&xx)); 

  // Map local to global
  PetscCall(DMLocalToGlobalBegin(dm,xlocal,INSERT_VALUES,x)); 
  PetscCall(DMLocalToGlobalEnd  (dm,xlocal,INSERT_VALUES,x)); 

  PetscCall(VecDestroy(&xlocal));
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
  PetscCall(PetscBagRegisterInt(bag, &par->nx, 32, "nx", "Element count in the x-dir [-]")); 
  PetscCall(PetscBagRegisterInt(bag, &par->nz, 4, "nz", "Element count in the z-dir [-]")); 

  PetscCall(PetscBagRegisterScalar(bag, &par->xmin, 0.0, "xmin", "Start coordinate of domain in x-dir [m]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->zmin, 0.0, "zmin", "Start coordinate of domain in z-dir [m]")); 

  PetscCall(PetscBagRegisterScalar(bag, &par->L, 4, "L", "Length of domain in x-dir [m]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->H, 0.5, "H", "Height of domain in z-dir [m]")); 

  // Physical and material parameters
  PetscCall(PetscBagRegisterScalar(bag, &par->rho, 1.0, "rho", "Reference density [kg/m^3]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->cp, 1.0, "cp", "Specific heat capacity [J/kg/K]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->k, 1.08, "k", "Thermal conductivity [W/m/K]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->La, 70.26, "La", "Latent heat [J/kg]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->T0, 0.0, "T0", "Initial temperature [deg C]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->Tb, -45.0, "Tb", "Surface temperature [deg C]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->Tm, -0.1, "Tm", "Melting/crystallization temperature [deg C]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->Ts, -0.1001, "Ts", "Mush melting/crystallization temperature [deg C]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->C0, 1.0, "C0", "Reference composition [-]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->beta,0.516385, "beta", "Analytical factor!! Valid only for this set of parameter values [-]")); 

  // Time stepping and advection parameters
  PetscCall(PetscBagRegisterInt(bag, &par->ts_scheme,2, "ts_scheme", "Time stepping scheme 0-forward euler, 1-backward euler, 2-crank-nicholson")); 
  PetscCall(PetscBagRegisterInt(bag, &par->adv_scheme,2, "adv_scheme", "Advection scheme 0-upwind, 1-upwind2, 2-fromm")); 

  PetscCall(PetscBagRegisterInt(bag, &par->tout,1, "tout", "Output every tout time step")); 
  PetscCall(PetscBagRegisterInt(bag, &par->tstep,20, "tstep", "Maximum no of time steps")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->dtmax, 0.2, "dtmax", "Maximum time step size [-]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->tmax, 4, "tmax", "Maximum time [s]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->tstart, 0.001, "tstart", "Starting time of simulation [s]")); 

  // scale parameters
  par->scal_h = par->L;
  par->kappa = par->k/(par->rho*par->cp);
  par->scal_t = par->scal_h*par->scal_h/par->kappa;
  par->St = par->cp*(par->Tm-par->Tb)/par->La;

  par->L = par->L/par->scal_h;
  par->H = par->H/par->scal_h;

  par->nd_dtmax = par->dtmax/par->scal_t;
  par->nd_tmax = par->tmax/par->scal_t;
  par->nd_t = par->tstart/par->scal_t;
  par->nd_dt = par->nd_dtmax;

  par->DT = par->Tm-par->Tb;
  par->nd_T0 = (par->T0-par->Tm)/par->DT;
  par->nd_Tb = (par->Tb-par->Tm)/par->DT;
  par->nd_Tm = (par->Tm-par->Tm)/par->DT;
  par->nd_Ts = (par->Ts-par->Tm)/par->DT;

  // Input/output 
  PetscCall(PetscBagRegisterString(bag,&par->fname_out,FNAME_LENGTH,"out_1d_sol","output_file","Name for output file, set with: -output_file <filename>")); 
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
  PetscCall(PetscPrintf(usr->comm,"# --------------------------------------- #\n"));
  PetscCall(PetscPrintf(usr->comm,"# 1-D Solidification (Enthalpy TC/HC): %s \n",&(date[0])));
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
  PetscCall(InputPrintData(usr)); 

  // Numerical solution using the FD pde object
  PetscCall(PetscTime(&start_time)); 
  PetscCall(Numerical_solution(usr)); 
  PetscCall(PetscTime(&end_time)); 
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"# Runtime: %g (sec) \n", end_time - start_time));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n"));

  // Destroy objects
  PetscCall(PetscBagDestroy(&usr->bag)); 
  PetscCall(PetscFree(usr)); 

  // Finalize main
  PetscCall(PetscFinalize());
  return 0;
}