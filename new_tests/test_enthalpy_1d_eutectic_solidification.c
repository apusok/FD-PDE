// ---------------------------------------
// 1D solidification problem of an initially liquid semi-infinite slab with a eutectic phase diagram (Parkinson et al, 2020)
// Use the Enthalpy method with H as primary energy variable.
// run: ./test_enthalpy_1d_eutectic_solidification -pc_type lu -pc_factor_mat_solver_type umfpack -pc_factor_mat_ordering_type external -snes_monitor -log_view
// python output: python/test_enthalpy_1d_eutectic_solidification.py
// ---------------------------------------
static char help[] = "1D Solidification problem using the Enthalpy Method and a eutectic phase diagram \n\n";

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

#include "../new_src/fdpde_enthalpy.h"

// ---------------------------------------
// Application Context
// ---------------------------------------
#define FNAME_LENGTH 200

// parameters (bag)
typedef struct {
  PetscInt       nx, nz;
  PetscScalar    L, H;
  PetscScalar    xmin, zmin;
  PetscScalar    v, S, Cc, k, cp, Le, e, C0, th_inf, ps, dt, tmax, t, tprev;
  PetscScalar    Cw, thw_inf, th_i, a, b, alpha, beta, h0, CFL;
  PetscInt       ts_scheme, adv_scheme, tout, tstep, steady_state;
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
PetscErrorCode Analytical_solution(DM,Vec*,void*);
PetscErrorCode Initial_solution(DM,Vec,void*);
PetscErrorCode Initial_solution2(DM,Vec,void*);
PetscErrorCode VerifySteadyState(DM,Vec,Vec,void*);
EnthEvalErrorCode Form_Enthalpy(PetscScalar,PetscScalar[],PetscScalar,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscInt,void*); 

const char coeff_description[] =
"  << ENTHALPY Coefficients >> \n"
"  A1 = 0, B1 = 0, C1 = -1, D1 = v*dH/dz  \n"
"  A2 = 0, B2 = 0, C2 = -1/Le, D2 = v*dC/dz  \n"
"  v = [0,0], vs = [0,0], vf = [0,0]\n";

const char bc_description[] =
"  << ENTHALPY BCs >> \n"
"  TEMP: DOWN: T = Tinf, TOP: T = Teut, LEFT, RIGHT - symmetry \n"
"  COMP: DOWN: C = -1, UP: C = -1, LEFT, RIGHT - symmetry \n";

const char enthalpy_method_description[] =
"  << ENTHALPY METHOD >> \n"
"  Input: H, C, P \n"
"  Output: T, phi, CF, CS \n"
"   > see Parkinson et al (2020) for full equations \n";

PetscScalar FrommAdvection1D(PetscScalar v, PetscScalar x[], PetscScalar dz)
{
  PetscScalar vN, vS, fS, fN;
  PetscScalar xC, xN, xNN, xS, xSS;

  vS = v; vN = v;
  // vS = v[0]; vN = v[1];
  xSS = x[0]; xS = x[1]; xC = x[2]; xN = x[3]; xNN = x[4]; 
  fN = vN *(-xNN + 5*(xN+xC)-xS )/8 - fabs(vN)*(-xNN + 3*(xN-xC)+xS )/8;
  fS = vS *(-xN  + 5*(xC+xS)-xSS)/8 - fabs(vS)*(-xN  + 3*(xC-xS)+xSS)/8;
  return (fN-fS)/dz;
}

static PetscScalar eval_H(PetscScalar theta, PetscScalar phi, PetscScalar S, PetscScalar cp) { return phi*S+(phi+(1.0-phi)*cp)*theta; }
static PetscScalar eval_T(PetscScalar H, PetscScalar phi, PetscScalar S, PetscScalar cp) { return (H-phi*S)/(phi+(1.0-phi)*cp); }
static PetscScalar eval_phiEutectic(PetscScalar C, PetscScalar Cc) { return 1.0+C/Cc; }
static PetscScalar eval_Tsolidus(PetscScalar C, PetscScalar Cc, PetscScalar cp, PetscScalar ps) { return cp*PetscMax(0,-1.0/ps*(C+Cc)); }

static PetscScalar analytical_theta_liquid(PetscScalar th_inf, PetscScalar th_i, PetscScalar z, PetscScalar h0) { 
  return th_inf+(th_i-th_inf)*PetscExpScalar(z-h0); }

static PetscScalar residual_theta_mush(PetscScalar z, PetscScalar theta, PetscScalar alpha, PetscScalar beta, PetscScalar C) { 
return -z - (alpha-C)/(alpha-beta)*PetscLogScalar((alpha+1.0)/(alpha-theta))-(C-beta)/(alpha-beta)*PetscLogScalar((beta+1.0)/(beta-theta)); }

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
  Vec            x, xprev, xcoeff, xcoeffprev, xAnalytic, xnew, xguess;
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
  PetscCall(FDPDESetFunctionCoefficient(fd,FormCoefficient,coeff_description,usr)); 

  if (par->adv_scheme==0) { PetscCall(FDPDEEnthalpySetAdvectSchemeType(fd,ADV_UPWIND)); }
  if (par->adv_scheme==1) { PetscCall(FDPDEEnthalpySetAdvectSchemeType(fd,ADV_UPWIND2)); }
  if (par->adv_scheme==2) { PetscCall(FDPDEEnthalpySetAdvectSchemeType(fd,ADV_FROMM)); }

  if (par->ts_scheme ==0) { PetscCall(FDPDEEnthalpySetTimeStepSchemeType(fd,TS_FORWARD_EULER)); }
  if (par->ts_scheme ==1) { PetscCall(FDPDEEnthalpySetTimeStepSchemeType(fd,TS_BACKWARD_EULER)); }
  if (par->ts_scheme ==2) { PetscCall(FDPDEEnthalpySetTimeStepSchemeType(fd,TS_CRANK_NICHOLSON));}
  PetscCall(FDPDEEnthalpySetTimestep(fd,par->dt)); 

  PetscCall(FDPDEEnthalpySetEnthalpyMethod(fd,Form_Enthalpy,enthalpy_method_description,usr));
  
  // Calculate analytical solution
  PetscCall(FDPDEGetDM(fd,&dm));
  PetscCall(Analytical_solution(dm,&xAnalytic,usr)); 
  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_analytic_solution_TC",par->fdir_out));
  PetscCall(DMStagViewBinaryPython(dm,xAnalytic,fout));

  PetscCall(Initial_solution2(dm,xAnalytic,usr));
  PetscCall(FDPDEEnthalpyUpdateDiagnostics(fd,dm,xAnalytic,&dmnew,&xnew)); 
  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_xanalytic_enthalpy_initial",par->fdir_out));
  PetscCall(DMStagViewBinaryPython(dmnew,xnew,fout));
  PetscCall(DMDestroy(&dmnew));
  PetscCall(VecDestroy(&xnew)); 

  // Set initial profile
  PetscCall(FDPDEEnthalpyGetPrevSolution(fd,&xprev));
  PetscCall(Initial_solution(dm,xprev,usr));
  // PetscCall(Initial_solution2(dm,xprev,usr));
  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_xprev_initial",par->fdir_out));
  PetscCall(DMStagViewBinaryPython(dm,xprev,fout));

  // Initialize guess with previous solution 
  PetscCall(FDPDEGetSolutionGuess(fd,&xguess));
  PetscCall(VecCopy(xprev,xguess));

  // Set initial coefficient structure
  PetscCall(FDPDEGetCoefficient(fd,&dmcoeff,NULL));
  PetscCall(FDPDEEnthalpyGetPrevCoefficient(fd,&xcoeffprev));
  PetscCall(FormCoefficient(fd,dm,xprev,dmcoeff,xcoeffprev,usr));
  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_xcoeffprev_initial",par->fdir_out));
  PetscCall(DMStagViewBinaryPython(dmcoeff,xcoeffprev,fout));
  PetscCall(VecDestroy(&xcoeffprev));
  PetscCall(VecDestroy(&xprev));
  PetscCall(VecDestroy(&xguess));

  // Time loop
  while ((par->t <= par->tmax) && (istep<par->tstep) && (usr->par->steady_state==0)) {
    PetscPrintf(PETSC_COMM_WORLD,"# TIMESTEP %d: \n",istep);

    // Update time
    par->tprev = par->t;
    par->t    += par->dt;

    // Enthalpy Solver
    PetscCall(FDPDESolve(fd,NULL));
    PetscCall(FDPDEGetSolution(fd,&x));
    // PetscCall(MatView(fd->J,PETSC_VIEWER_STDOUT_WORLD));

    // Copy solution and coefficient to old
    PetscCall(FDPDEEnthalpyGetPrevSolution(fd,&xprev));

    // check steady-state
    PetscCall(VerifySteadyState(dm,x,xprev,usr));
    PetscCall(VecCopy(x,xprev));
    PetscCall(VecDestroy(&xprev));

    PetscCall(FDPDEGetCoefficient(fd,&dmcoeff,&xcoeff));
    PetscCall(FDPDEEnthalpyGetPrevCoefficient(fd,&xcoeffprev));
    PetscCall(VecCopy(xcoeff,xcoeffprev));
    PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s_coeff_ts%1.3d",par->fdir_out,par->fname_out,istep));
    PetscCall(DMStagViewBinaryPython(dmcoeff,xcoeff,fout));
    PetscCall(VecDestroy(&xcoeffprev));

    // Output solution and calculate fluid velocity
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
  PetscCall(VecDestroy(&xAnalytic)); 
  PetscCall(FDPDEDestroy(&fd));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// Phase Diagram  * relationships from Katz and Worster (2008)
// enthalpy_method(H,C,P,&T,&phi,CF,CS,ncomp,user);
// ---------------------------------------
EnthEvalErrorCode Form_Enthalpy(PetscScalar H,PetscScalar C[],PetscScalar P,PetscScalar *_T,PetscScalar *_phi,PetscScalar *CF,PetscScalar *CS,PetscInt ncomp, void *ctx) 
{
  UsrData      *usr = (UsrData*) ctx;
  PetscInt     ii;
  PetscScalar  Tsol, Tliq, Teut, Hsol, Hliq, Heut, phiE, T, phi=0.0, A, B, D;
  PetscScalar  S, Cc, ps, cp;

  S  = usr->par->S;
  Cc = usr->par->Cc;
  ps = usr->par->ps;
  cp = usr->par->cp;

  // Solidus and liquidus
  Tliq = -C[0];
  Hliq = eval_H(Tliq,1.0,S,cp); 

  Tsol = eval_Tsolidus(C[0],Cc,cp,ps);
  Hsol = eval_H(Tsol,0.0,S,cp);

  Teut = 0.0;
  phiE = eval_phiEutectic(C[0],Cc);
  Heut = eval_H(Teut,phiE,S,cp);

  if (H <= Hsol) { // below solidus
    phi = 0.0;
    T   = eval_T(H,phi,S,cp);
    for (ii = 0; ii<ncomp; ii++) {
      CS[ii] = C[ii];
      CF[ii] = 0.0;
    }
  } else if ((H > Hsol) && (H <= Heut)) { // eutectic-solid
    phi = H/S;
    T   = 0.0;
    for (ii = 0; ii<ncomp; ii++) {
      CS[ii] = C[ii]/(1.0-phi);
      CF[ii] = 0.0;
    }
  } else if ((H > Heut) && (H <= Hliq)) { // mush
    A = Cc*(cp-1.0)+S*(ps-1.0);
    B = Cc*(1.0-2.0*cp)+H*(1.0-ps)-C[0]*(cp-1.0)-S*ps;
    D = (Cc+C[0])*cp+ps*H;
    phi = (-B-PetscSqrtScalar(B*B-4.0*A*D))/(2.0*A);
    for (ii = 0; ii<ncomp; ii++) {
      CS[ii] = (ps*C[ii]-Cc*phi)/(phi+ps*(1.0-phi));
      CF[ii] = (C[ii]+Cc*(1.0-phi))/(phi+ps*(1.0-phi));
    }
    T = -CF[0];
    // T = eval_T(H,phi,S,cp); 
  } else { // above liquidus
    phi = 1.0;
    T = eval_T(H,phi,S,cp); //H-S;
    for (ii = 0; ii<ncomp; ii++) {
      CS[ii] = 0.0;
      CF[ii] = C[ii];
    }
  }

  // assign pointers
  *_T = T;
  *_phi = phi;

  // error checking
  ENTH_CHECK_PHI(phi);
  return(STATE_VALID);
}

// ---------------------------------------
// FormBCList
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormBCList"
PetscErrorCode FormBCList(DM dm, Vec x, DMStagBCList bclist, void *ctx)
{
  UsrData     *usr = (UsrData*)ctx;
  PetscScalar H_down, H_top, C_down, C_top,thE,thdown, phiE;
  PetscInt    k,n_bc,*idx_bc;
  PetscScalar *value_bc,*x_bc;
  BCType      *type_bc;
  PetscFunctionBeginUser;

  // eutectic - top
  thE  = 0.0;
  C_top = usr->par->C0; 
  phiE = eval_phiEutectic(C_top,usr->par->Cc);
  H_top = phiE*usr->par->S+thE;

  // bottom
  thdown = 1.0+analytical_theta_liquid(usr->par->thw_inf,usr->par->th_i,usr->par->zmin,usr->par->h0);
  H_down = usr->par->S+thdown;
  C_down = usr->par->C0;

  // H Down:
  PetscCall(DMStagBCListGetValues(bclist,'s','o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = H_down;
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // H Top:
  PetscCall(DMStagBCListGetValues(bclist,'n','o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = H_top;
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // C Down:
  PetscCall(DMStagBCListGetValues(bclist,'s','o',1,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = C_down;
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',1,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // C Top:
  PetscCall(DMStagBCListGetValues(bclist,'n','o',1,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = C_top;
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',1,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

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
  Params         *par;
  PetscInt       i, j, sx, sz, nx, nz, Nx,Nz,icenter;
  Vec            xlocal, coefflocal;
  PetscScalar    ***c;
  PetscScalar    **coordx,**coordz;
  PetscFunctionBeginUser;

  par = usr->par;
  // Get domain corners
  PetscCall(DMStagGetGlobalSizes(dmcoeff,&Nx,&Nz,NULL));
  PetscCall(DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 
  PetscCall(DMCreateLocalVector(dmcoeff, &coefflocal)); 
  PetscCall(DMStagVecGetArray(dmcoeff, coefflocal, &c)); 
  
  PetscCall(DMGetLocalVector(dm,&xlocal)); 
  PetscCall(DMGlobalToLocal (dm,x,INSERT_VALUES,xlocal)); 

  PetscCall(DMStagGetProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmcoeff,ELEMENT,&icenter)); 

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {

      { // ELEMENT
        DMStagStencil point, pointE[5];
        PetscScalar   x[5],dz;
        PetscInt      idx;

        point.i = i; point.j = j; point.loc = ELEMENT;  
        point.c = COEFF_A1; PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
        c[j][i][idx] = 0.0;

        point.c = COEFF_B1; PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
        c[j][i][idx] = 0.0;

        // D1 = v*dH/dz
        dz = par->H/par->nz;
        pointE[0].i = i; pointE[0].j = j-2; pointE[0].loc = DMSTAG_ELEMENT; pointE[0].c = 0;
        pointE[1].i = i; pointE[1].j = j-1; pointE[1].loc = DMSTAG_ELEMENT; pointE[1].c = 0;
        pointE[2].i = i; pointE[2].j = j  ; pointE[2].loc = DMSTAG_ELEMENT; pointE[2].c = 0;
        pointE[3].i = i; pointE[3].j = j+1; pointE[3].loc = DMSTAG_ELEMENT; pointE[3].c = 0;
        pointE[4].i = i; pointE[4].j = j+2; pointE[4].loc = DMSTAG_ELEMENT; pointE[4].c = 0;
        // take care of boundaries
        if (j == 0   ) { pointE[0]=pointE[2]; pointE[1]=pointE[2]; }
        if (j == 1   ) { pointE[0]=pointE[2]; }
        if (j == Nz-1) { pointE[3]=pointE[2]; pointE[4]=pointE[2]; }
        if (j == Nz-2) { pointE[4]=pointE[2]; }
        PetscCall(DMStagVecGetValuesStencil(dm,xlocal,5,pointE,x)); 
        
        point.c = COEFF_D1; PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
        c[j][i][idx] = FrommAdvection1D(par->v,x,dz);

        point.c = COEFF_A2; PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
        c[j][i][idx] = 0.0;

        point.c = COEFF_B2; PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
        c[j][i][idx] = 0.0;

        // D2 = v*dC/dz
        pointE[0].c = 1;
        pointE[1].c = 1;
        pointE[2].c = 1;
        pointE[3].c = 1;
        pointE[4].c = 1;
        PetscCall(DMStagVecGetValuesStencil(dm,xlocal,5,pointE,x)); 
        point.c = COEFF_D2; PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
        c[j][i][idx] = FrommAdvection1D(par->v,x,dz);
      }

      { // FACES
        DMStagStencil point[4];
        PetscInt      ii, idx;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT; 
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN; 
        point[3].i = i; point[3].j = j; point[3].loc = UP; 

        for (ii = 0; ii < 4; ii++) {
          // C1 = -1
          point[ii].c = COEFF_C1; PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] = -1.0;

          // C2 = -1/Le = -e
          point[ii].c = COEFF_C2; PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] = -par->e; //-1.0/par->Le;

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
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL));
  PetscCall(DMStagVecRestoreArray(dmcoeff,coefflocal,&c));
  PetscCall(DMLocalToGlobalBegin(dmcoeff,coefflocal,INSERT_VALUES,coeff)); 
  PetscCall(DMLocalToGlobalEnd  (dmcoeff,coefflocal,INSERT_VALUES,coeff)); 
  
  PetscCall(VecDestroy(&coefflocal)); 
  PetscCall(DMRestoreLocalVector(dm,&xlocal)); 

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// Create analytical solution
// ---------------------------------------
PetscErrorCode Analytical_solution(DM dm,Vec *_x, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  PetscInt       i, j, sx, sz, nx, nz, idx, icenter, it, nmax;
  PetscScalar    ***xx, tol, alpha, beta, Cw;
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

  alpha = usr->par->alpha;
  beta  = usr->par->beta;
  Cw    = usr->par->Cw;

  tol = 1e-10;
  nmax = 50;

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      DMStagStencil  point;
      PetscScalar    zp, th=0.0, phi, th_a, th_b, th_c, res_a, res_b, res_c, res;

      point.i = i; point.j = j; point.loc = ELEMENT; point.c = 0; 
      zp = coordz[j][icenter];

      if (zp<usr->par->h0) { // liquid region z<h0
        th  = analytical_theta_liquid(usr->par->thw_inf, usr->par->th_i, zp, usr->par->h0);
        phi = 1.0;
      } else { // mushy region 0>z>h0, temperature/composition needs to be calculated implicitly
        // use a bisection algorithm
        it = 0;
        th_a = -1.0; // eutectic temp
        th_b = 0.0; // liquidus temp
        res_a = residual_theta_mush(zp,th_a,alpha,beta,Cw);
        res_b = residual_theta_mush(zp,th_b,alpha,beta,Cw);
        if      (PetscAbsScalar(res_a)<=tol) th = th_a;
        else if (PetscAbsScalar(res_b)<=tol) th = th_b;
        else {
          res = PetscMin(PetscAbsScalar(res_a),PetscAbsScalar(res_b));
          while (it<=nmax && res>=tol) {
            it += 1;
            th_c = (th_a+th_b)*0.5;
            res_a = residual_theta_mush(zp,th_a,alpha,beta,Cw);
            res_b = residual_theta_mush(zp,th_b,alpha,beta,Cw);
            res_c = residual_theta_mush(zp,th_c,alpha,beta,Cw);
            res = PetscAbsScalar(res_c);
            if (res<tol) th = th_c;
            else {
              if (res_a*res_c<0) th_b = th_c;
              else               th_a = th_c;
            }
          }
        }
        phi = (Cw-usr->par->th_i)/(Cw-th);
      }

      // save data
      PetscCall(DMStagGetLocationSlot(dm, point.loc, 0, &idx));
      xx[j][i][idx] = th+1.0;

      // porosity
      PetscCall(DMStagGetLocationSlot(dm, point.loc, 1, &idx));
      xx[j][i][idx] = phi;
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
      PetscScalar    zp, th;

      point.i = i; point.j = j; point.loc = ELEMENT; point.c = 0; 
      zp = coordz[j][icenter];
      th = 1.0+analytical_theta_liquid(usr->par->thw_inf,usr->par->th_i,zp, usr->par->h0);

      PetscCall(DMStagGetLocationSlot(dm, point.loc, point.c, &idx));
      xx[j][i][idx] = usr->par->S+th;

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
// Verify enthalpy method algorithm with the analytical solution
// ---------------------------------------
PetscErrorCode Initial_solution2(DM dm,Vec x, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  PetscInt       i, j, sx, sz, nx, nz, idx, icenter, it, nmax;
  PetscScalar    ***xx, tol, alpha, beta, Cw;
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

  alpha = usr->par->alpha;
  beta  = usr->par->beta;
  Cw     = usr->par->Cw;

  tol = 1e-10;
  nmax = 50;

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      DMStagStencil  point;
      PetscScalar    zp, th=0.0, phi, th_a, th_b, th_c, res_a, res_b, res_c, res;

      point.i = i; point.j = j; point.loc = ELEMENT; point.c = 0; 
      zp = coordz[j][icenter];

      if (zp<usr->par->h0) { // liquid region z<h0
        th  = analytical_theta_liquid(usr->par->thw_inf, usr->par->th_i, zp, usr->par->h0);
        phi = 1.0;
      } else { // mushy region 0>z>h0, temperature/composition needs to be calculated implicitly
        // use a bisection algorithm
        it = 0;
        th_a = -1.0; // eutectic temp
        th_b = 0.0; // liquidus temp
        res_a = residual_theta_mush(zp,th_a,alpha,beta,Cw);
        res_b = residual_theta_mush(zp,th_b,alpha,beta,Cw);
        if      (PetscAbsScalar(res_a)<=tol) th = th_a;
        else if (PetscAbsScalar(res_b)<=tol) th = th_b;
        else {
          res = PetscMin(PetscAbsScalar(res_a),PetscAbsScalar(res_b));
          while (it<=nmax && res>=tol) {
            it += 1;
            th_c = (th_a+th_b)*0.5;
            res_a = residual_theta_mush(zp,th_a,alpha,beta,Cw);
            res_b = residual_theta_mush(zp,th_b,alpha,beta,Cw);
            res_c = residual_theta_mush(zp,th_c,alpha,beta,Cw);
            res = PetscAbsScalar(res_c);
            if (res<tol) th = th_c;
            else {
              if (res_a*res_c<0) th_b = th_c;
              else               th_a = th_c;
            }
          }
        }
        phi = (Cw-usr->par->th_i)/(Cw-th);
      }

      // save data
      // th  = 1.0+th;
      // phi = 2.0-1.0/phi;
      PetscCall(DMStagGetLocationSlot(dm, point.loc, 0, &idx));
      xx[j][i][idx] = phi*usr->par->S+th+1.0;

      // composition
      PetscCall(DMStagGetLocationSlot(dm, point.loc, 1, &idx));
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
// Verify steady-state
// ---------------------------------------
PetscErrorCode VerifySteadyState(DM dm,Vec x,Vec xprev, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  PetscInt       i, j, sx, sz, nx, nz;
  Vec            xlocal,xprevlocal;
  PetscScalar    tol, xx, xxprev,max_val,gmax_val = 0.0;
  PetscFunctionBeginUser;

  // Get local and global vector associated with DM
  PetscCall(DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 
  PetscCall(DMGetLocalVector(dm,&xlocal)); 
  PetscCall(DMGlobalToLocal (dm,x,INSERT_VALUES,xlocal)); 
  PetscCall(DMGetLocalVector(dm,&xprevlocal)); 
  PetscCall(DMGlobalToLocal (dm,xprev,INSERT_VALUES,xprevlocal)); 

  tol     = 1.0e-4*usr->par->dt;
  max_val = 0.0; 
  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      DMStagStencil  point;
      point.i = i; point.j = j; point.loc = ELEMENT; point.c = 0; 
      PetscCall(DMStagVecGetValuesStencil(dm, xlocal, 1,&point,&xx)); 
      PetscCall(DMStagVecGetValuesStencil(dm, xprevlocal,1,&point,&xxprev));  
      max_val = PetscMax(max_val,PetscAbsScalar(xx-xxprev));
    }
  }

  PetscCall(MPI_Allreduce(&max_val,&gmax_val,1,MPI_DOUBLE,MPI_MAX,usr->comm));
  if (gmax_val<tol) usr->par->steady_state = 1;

  PetscPrintf(PETSC_COMM_WORLD,"# >> Steady-state check: dt = %1.6e tol = %1.6e max(|x-xprev|) = %1.6e\n",usr->par->dt,tol,gmax_val);

  // Restore vectors
  PetscCall(DMRestoreLocalVector(dm,&xlocal)); 
  PetscCall(DMRestoreLocalVector(dm,&xprevlocal)); 
  
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
  PetscScalar    phiE, th = 0.0;
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
  PetscCall(PetscBagRegisterInt(bag, &par->nx, 10, "nx", "Element count in the x-dir [-]")); 
  PetscCall(PetscBagRegisterInt(bag, &par->nz, 40, "nz", "Element count in the z-dir [-]")); 

  PetscCall(PetscBagRegisterScalar(bag, &par->xmin, 0.0, "xmin", "Start coordinate of domain in x-dir [m]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->zmin, -4.0, "zmin", "Start coordinate of domain in z-dir [m]")); 

  PetscCall(PetscBagRegisterScalar(bag, &par->L, 1.0, "L", "Length of domain in x-dir [m]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->H, 4.0, "H", "Height of domain in z-dir [m]")); 

  // Physical and material parameters
  PetscCall(PetscBagRegisterScalar(bag, &par->v, 1.0, "v", "Frame velocity [-]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->S, 5.0, "S", "Stefan number [-]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->Cc, 5.0, "Cc", "Eutectic compositional number [-]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->k, 1.0, "k", "Ratio of conductivities [-]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->cp, 1.0, "cp", "Ratio of specific heat capacities [-]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->Le, 1e10, "Le", "Lewis number [-]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->th_inf, 1.1, "th_inf", "Dimensionless temperature in the far-field [-]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->ps, 1.0e-5, "ps", "Partition coefficient [-]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->C0, -1.0, "C0", "Initial bulk composition [-]")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->CFL, 0.2, "CFL", "CFL criterion for dt [-]")); 

  // Time stepping and advection parameters
  PetscCall(PetscBagRegisterInt(bag, &par->ts_scheme,2, "ts_scheme", "Time stepping scheme 0-forward euler, 1-backward euler, 2-crank-nicholson")); 
  PetscCall(PetscBagRegisterInt(bag, &par->adv_scheme,2, "adv_scheme", "Advection scheme 0-upwind, 1-upwind2, 2-fromm")); 

  PetscCall(PetscBagRegisterInt(bag, &par->tout,1, "tout", "Output every tout time step")); 
  PetscCall(PetscBagRegisterInt(bag, &par->tstep,10, "tstep", "Maximum no of time steps")); 
  par->steady_state = 0;

  // scale parameters
  PetscScalar scal_v;
  scal_v = PetscAbsScalar(par->v);
  par->tmax  = par->H/scal_v;
  par->dt    = par->CFL*par->H/par->nz/scal_v;
  par->t     = 0.0;
  par->tprev = 0.0;

  if (par->Le >=1e5) par->e = 0.0;
  else par->e = 1.0/par->Le;

  // enthalpy analytical solution
  phiE = 1.0+par->C0/par->Cc;
  par->Cw = phiE/(1.0-phiE);
  par->thw_inf = par->th_inf-1.0;
  par->th_i = -par->e/(1.0-par->e)*par->thw_inf;
  par->a = 0.5*(par->Cw+par->thw_inf+par->S);
  par->b = PetscSqrtScalar(par->a*par->a - par->Cw*par->thw_inf-par->S*par->th_i);
  par->alpha = par->a+par->b;
  par->beta = par->a-par->b;

  // depth of mushy layer th=0.0
  par->h0 = -((par->alpha-par->Cw)/(par->alpha-par->beta)*PetscLogScalar((par->alpha+1.0)/(par->alpha-th))+(par->Cw-par->beta)/(par->alpha-par->beta)*PetscLogScalar((par->beta+1.0)/(par->beta-th)));

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
  PetscPrintf(usr->comm,"# --------------------------------------- #\n");
  PetscPrintf(usr->comm,"# 1-D Eutectic Solidification (Enthalpy): %s \n",&(date[0]));
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
  PetscCall(PetscOptionsInsert(PETSC_NULLPTR,&argc,&argv,NULL)); 

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