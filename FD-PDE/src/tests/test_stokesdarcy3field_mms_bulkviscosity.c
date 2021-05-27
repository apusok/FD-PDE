// ---------------------------------------
// MMS test for 3-Field and 2-Field
// run: ./tests/test_stokesdarcy3field_mms_bulkviscosity.app -pc_type lu -pc_factor_mat_solver_type umfpack -pc_factor_mat_ordering_type external -nx 10 -nz 10
// python test: ./tests/python/test_stokesdarcy3field_mms_bulkviscosity.py
// python sympy: ./mms/mms_stokesdarcy3field_bulkviscosity.py
// ---------------------------------------
static char help[] = "Two-phase flow application to verify 2-Field and 3-Field formulations using MMS \n\n";

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
#include "../fdpde_stokesdarcy2field.h"
#include "../fdpde_stokesdarcy3field.h"
#include "../dmstagoutput.h"

// ---------------------------------------
// Application Context
// ---------------------------------------
#define FNAME_LENGTH 200

// parameters (bag)
typedef struct {
  PetscInt       nx, nz;
  PetscScalar    L, H;
  PetscScalar    xmin, zmin;
  PetscScalar    delta,phi0,phia,phi_min,vzeta,eta,p_s,psi_s,U_s,m,n,k_hat;
  char           fname_out[FNAME_LENGTH]; 
  char           fdir_out[FNAME_LENGTH]; 
} Params;

// user defined and model-dependent variables
typedef struct {
  Params        *par;
  PetscBag       bag;
  MPI_Comm       comm;
  PetscMPIInt    rank;
  DM             dm;
} UsrData;

// ---------------------------------------
// Function definitions
// ---------------------------------------
PetscErrorCode StokesDarcy2Field_Numerical(void*);
PetscErrorCode FormCoefficient2Field(FDPDE, DM, Vec, DM, Vec, void*);
PetscErrorCode FormBCList2Field(DM, Vec, DMStagBCList, void*);
PetscErrorCode ComputeManufacturedSolution2Field(DM,Vec*,void*);
PetscErrorCode ComputeErrorNorms2Field(DM,Vec,Vec);

PetscErrorCode StokesDarcy3Field_Numerical(void*);
PetscErrorCode FormCoefficient3Field(FDPDE, DM, Vec, DM, Vec, void*);
PetscErrorCode FormBCList3Field(DM, Vec, DMStagBCList, void*);
PetscErrorCode ComputeManufacturedSolution3Field(DM,Vec*,void*);
PetscErrorCode ComputeErrorNorms3Field(DM,Vec,Vec);

PetscErrorCode ComputeExtraParameters(DM,void*);
PetscErrorCode InputParameters(UsrData**);
PetscErrorCode InputPrintData(UsrData*);

// Manufactured solutions
static PetscScalar get_p(PetscScalar x, PetscScalar z, PetscScalar delta, PetscScalar phi0, PetscScalar phia, PetscScalar phi_min, PetscScalar vzeta, PetscScalar p_s,PetscScalar psi_s, PetscScalar U_s, PetscScalar m, PetscScalar n, PetscScalar k_hat)
{ PetscScalar result;
  result = p_s*cos(M_PI*m*x)*cos(M_PI*m*z);
  return(result);
}
static PetscScalar get_pc(PetscScalar x, PetscScalar z, PetscScalar delta, PetscScalar phi0, PetscScalar phia, PetscScalar phi_min, PetscScalar vzeta, PetscScalar p_s,PetscScalar psi_s, PetscScalar U_s, PetscScalar m, PetscScalar n, PetscScalar k_hat)
{ PetscScalar result;
  result = p_s*sin(M_PI*m*x)*sin(M_PI*m*z);
  return(result);
}
static PetscScalar get_ux(PetscScalar x, PetscScalar z, PetscScalar delta, PetscScalar phi0, PetscScalar phia, PetscScalar phi_min, PetscScalar vzeta, PetscScalar p_s,PetscScalar psi_s, PetscScalar U_s, PetscScalar m, PetscScalar n, PetscScalar k_hat)
{ PetscScalar result;
  result = M_PI*U_s*m*sin(M_PI*m*x)*cos(M_PI*m*z) + M_PI*m*psi_s*(-cos(M_PI*m*x) + 1.0)*sin(M_PI*m*z);
  return(result);
}
static PetscScalar get_uz(PetscScalar x, PetscScalar z, PetscScalar delta, PetscScalar phi0, PetscScalar phia, PetscScalar phi_min, PetscScalar vzeta, PetscScalar p_s,PetscScalar psi_s, PetscScalar U_s, PetscScalar m, PetscScalar n, PetscScalar k_hat)
{ PetscScalar result;
  result = M_PI*U_s*m*sin(M_PI*m*z)*cos(M_PI*m*x) - M_PI*m*psi_s*(-cos(M_PI*m*z) + 1.0)*sin(M_PI*m*x);
  return(result);
}
static PetscScalar get_f2ux(PetscScalar x, PetscScalar z, PetscScalar delta, PetscScalar phi0, PetscScalar phia, PetscScalar phi_min, PetscScalar vzeta, PetscScalar p_s,PetscScalar psi_s, PetscScalar U_s, PetscScalar m, PetscScalar n, PetscScalar k_hat)
{ PetscScalar result;
  result = pow(delta, 2)*(2.0*pow(M_PI, 3)*U_s*pow(m, 3)*phi0*phia*vzeta*sin(M_PI*m*x)*cos(M_PI*m*x)*pow(cos(M_PI*m*z), 2)/pow(phi0*(phia*cos(M_PI*m*x)*cos(M_PI*m*z) + 1.0) + phi_min, 2) - 2*pow(M_PI, 3)*U_s*pow(m, 3)*(1.0*vzeta/(phi0*(phia*cos(M_PI*m*x)*cos(M_PI*m*z) + 1.0) + phi_min) - 0.66666666666666663)*sin(M_PI*m*x)*cos(M_PI*m*z)) + pow(delta, 2)*(-4.0*pow(M_PI, 3)*U_s*pow(m, 3)*sin(M_PI*m*x)*cos(M_PI*m*z) - 1.0*pow(M_PI, 3)*pow(m, 3)*psi_s*(-cos(M_PI*m*x) + 1.0)*sin(M_PI*m*z) + 1.0*pow(M_PI, 3)*pow(m, 3)*psi_s*sin(M_PI*m*z)*cos(M_PI*m*x)) + M_PI*m*p_s*sin(M_PI*m*x)*cos(M_PI*m*z);
  return(result);
}
static PetscScalar get_f2uz(PetscScalar x, PetscScalar z, PetscScalar delta, PetscScalar phi0, PetscScalar phia, PetscScalar phi_min, PetscScalar vzeta, PetscScalar p_s,PetscScalar psi_s, PetscScalar U_s, PetscScalar m, PetscScalar n, PetscScalar k_hat)
{ PetscScalar result;
  result = pow(delta, 2)*(2.0*pow(M_PI, 3)*U_s*pow(m, 3)*phi0*phia*vzeta*sin(M_PI*m*z)*pow(cos(M_PI*m*x), 2)*cos(M_PI*m*z)/pow(phi0*(phia*cos(M_PI*m*x)*cos(M_PI*m*z) + 1.0) + phi_min, 2) - 2*pow(M_PI, 3)*U_s*pow(m, 3)*(1.0*vzeta/(phi0*(phia*cos(M_PI*m*x)*cos(M_PI*m*z) + 1.0) + phi_min) - 0.66666666666666663)*sin(M_PI*m*z)*cos(M_PI*m*x)) + pow(delta, 2)*(-4.0*pow(M_PI, 3)*U_s*pow(m, 3)*sin(M_PI*m*z)*cos(M_PI*m*x) + 1.0*pow(M_PI, 3)*pow(m, 3)*psi_s*(-cos(M_PI*m*z) + 1.0)*sin(M_PI*m*x) - 1.0*pow(M_PI, 3)*pow(m, 3)*psi_s*sin(M_PI*m*x)*cos(M_PI*m*z)) - k_hat*phi0*(phia*cos(M_PI*m*x)*cos(M_PI*m*z) + 1.0) + M_PI*m*p_s*sin(M_PI*m*z)*cos(M_PI*m*x);
  return(result);
}
static PetscScalar get_f2p(PetscScalar x, PetscScalar z, PetscScalar delta, PetscScalar phi0, PetscScalar phia, PetscScalar phi_min, PetscScalar vzeta, PetscScalar p_s,PetscScalar psi_s, PetscScalar U_s, PetscScalar m, PetscScalar n, PetscScalar k_hat)
{ PetscScalar result;
  result = 2*pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) - pow(M_PI, 2)*pow(m, 2)*n*p_s*phia*pow(phia*cos(M_PI*m*x)*cos(M_PI*m*z) + 1.0, n)*pow(sin(M_PI*m*x), 2)*pow(cos(M_PI*m*z), 2)/(phia*cos(M_PI*m*x)*cos(M_PI*m*z) + 1.0) + 2*pow(M_PI, 2)*pow(m, 2)*p_s*pow(phia*cos(M_PI*m*x)*cos(M_PI*m*z) + 1.0, n)*cos(M_PI*m*x)*cos(M_PI*m*z) + M_PI*m*n*phia*(k_hat - M_PI*m*p_s*sin(M_PI*m*z)*cos(M_PI*m*x))*pow(phia*cos(M_PI*m*x)*cos(M_PI*m*z) + 1.0, n)*sin(M_PI*m*z)*cos(M_PI*m*x)/(phia*cos(M_PI*m*x)*cos(M_PI*m*z) + 1.0);
  return(result);
}
static PetscScalar get_f3ux(PetscScalar x, PetscScalar z, PetscScalar delta, PetscScalar phi0, PetscScalar phia, PetscScalar phi_min, PetscScalar vzeta, PetscScalar p_s,PetscScalar psi_s, PetscScalar U_s, PetscScalar m, PetscScalar n, PetscScalar k_hat)
{ PetscScalar result;
  result = pow(delta, 2)*(-4.0*pow(M_PI, 3)*U_s*pow(m, 3)*sin(M_PI*m*x)*cos(M_PI*m*z) - 1.0*pow(M_PI, 3)*pow(m, 3)*psi_s*(-cos(M_PI*m*x) + 1.0)*sin(M_PI*m*z) + 1.0*pow(M_PI, 3)*pow(m, 3)*psi_s*sin(M_PI*m*z)*cos(M_PI*m*x)) + M_PI*m*p_s*sin(M_PI*m*x)*cos(M_PI*m*z);
  return(result);
}
static PetscScalar get_f3uz(PetscScalar x, PetscScalar z, PetscScalar delta, PetscScalar phi0, PetscScalar phia, PetscScalar phi_min, PetscScalar vzeta, PetscScalar p_s,PetscScalar psi_s, PetscScalar U_s, PetscScalar m, PetscScalar n, PetscScalar k_hat)
{ PetscScalar result;
  result = pow(delta, 2)*(-4.0*pow(M_PI, 3)*U_s*pow(m, 3)*sin(M_PI*m*z)*cos(M_PI*m*x) + 1.0*pow(M_PI, 3)*pow(m, 3)*psi_s*(-cos(M_PI*m*z) + 1.0)*sin(M_PI*m*x) - 1.0*pow(M_PI, 3)*pow(m, 3)*psi_s*sin(M_PI*m*x)*cos(M_PI*m*z)) - k_hat*phi0*(phia*cos(M_PI*m*x)*cos(M_PI*m*z) + 1.0) + M_PI*m*p_s*sin(M_PI*m*z)*cos(M_PI*m*x);
  return(result);
}
static PetscScalar get_f3p(PetscScalar x, PetscScalar z, PetscScalar delta, PetscScalar phi0, PetscScalar phia, PetscScalar phi_min, PetscScalar vzeta, PetscScalar p_s,PetscScalar psi_s, PetscScalar U_s, PetscScalar m, PetscScalar n, PetscScalar k_hat)
{ PetscScalar result;
  result = 2*pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) + M_PI*m*n*phia*pow(phia*cos(M_PI*m*x)*cos(M_PI*m*z) + 1.0, n)*(-M_PI*m*p_s*sin(M_PI*m*x)*cos(M_PI*m*z) + M_PI*m*p_s*sin(M_PI*m*z)*cos(M_PI*m*x))*sin(M_PI*m*x)*cos(M_PI*m*z)/(phia*cos(M_PI*m*x)*cos(M_PI*m*z) + 1.0) + M_PI*m*n*phia*pow(phia*cos(M_PI*m*x)*cos(M_PI*m*z) + 1.0, n)*(k_hat + M_PI*m*p_s*sin(M_PI*m*x)*cos(M_PI*m*z) - M_PI*m*p_s*sin(M_PI*m*z)*cos(M_PI*m*x))*sin(M_PI*m*z)*cos(M_PI*m*x)/(phia*cos(M_PI*m*x)*cos(M_PI*m*z) + 1.0) - 2*pow(phia*cos(M_PI*m*x)*cos(M_PI*m*z) + 1.0, n)*(-pow(M_PI, 2)*pow(m, 2)*p_s*sin(M_PI*m*x)*sin(M_PI*m*z) - pow(M_PI, 2)*pow(m, 2)*p_s*cos(M_PI*m*x)*cos(M_PI*m*z));
  return(result);
}
static PetscScalar get_f3pc(PetscScalar x, PetscScalar z, PetscScalar delta, PetscScalar phi0, PetscScalar phia, PetscScalar phi_min, PetscScalar vzeta, PetscScalar p_s,PetscScalar psi_s, PetscScalar U_s, PetscScalar m, PetscScalar n, PetscScalar k_hat)
{ PetscScalar result;
  result = 2*pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) - p_s*sin(M_PI*m*x)*sin(M_PI*m*z)/(pow(delta, 2)*(1.0*vzeta/(phi0*(phia*cos(M_PI*m*x)*cos(M_PI*m*z) + 1.0) + phi_min) - 0.66666666666666663));
  return(result);
}
static PetscScalar get_K(PetscScalar x, PetscScalar z, PetscScalar delta, PetscScalar phi0, PetscScalar phia, PetscScalar phi_min, PetscScalar vzeta, PetscScalar p_s,PetscScalar psi_s, PetscScalar U_s, PetscScalar m, PetscScalar n, PetscScalar k_hat)
{ PetscScalar result;
  result = pow(phia*cos(M_PI*m*x)*cos(M_PI*m*z) + 1.0, n);
  return(result);
}
static PetscScalar get_phi(PetscScalar x, PetscScalar z, PetscScalar delta, PetscScalar phi0, PetscScalar phia, PetscScalar phi_min, PetscScalar vzeta, PetscScalar p_s,PetscScalar psi_s, PetscScalar U_s, PetscScalar m, PetscScalar n, PetscScalar k_hat)
{ PetscScalar result;
  result = phi0*(phia*cos(M_PI*m*x)*cos(M_PI*m*z) + 1.0);
  return(result);
}
static PetscScalar get_zeta(PetscScalar x, PetscScalar z, PetscScalar delta, PetscScalar phi0, PetscScalar phia, PetscScalar phi_min, PetscScalar vzeta, PetscScalar p_s,PetscScalar psi_s, PetscScalar U_s, PetscScalar m, PetscScalar n, PetscScalar k_hat)
{ PetscScalar result;
  result = 1.0*vzeta/(phi0*(phia*cos(M_PI*m*x)*cos(M_PI*m*z) + 1.0) + phi_min);
  return(result);
}
static PetscScalar get_eta(PetscScalar x, PetscScalar z, PetscScalar delta, PetscScalar phi0, PetscScalar phia, PetscScalar phi_min, PetscScalar vzeta, PetscScalar p_s,PetscScalar psi_s, PetscScalar U_s, PetscScalar m, PetscScalar n, PetscScalar k_hat)
{ PetscScalar result;
  result = 1.0;
  return(result);
}
static PetscScalar get_xi(PetscScalar x, PetscScalar z, PetscScalar delta, PetscScalar phi0, PetscScalar phia, PetscScalar phi_min, PetscScalar vzeta, PetscScalar p_s,PetscScalar psi_s, PetscScalar U_s, PetscScalar m, PetscScalar n, PetscScalar k_hat)
{ PetscScalar result;
  result = 1.0*vzeta/(phi0*(phia*cos(M_PI*m*x)*cos(M_PI*m*z) + 1.0) + phi_min) - 0.66666666666666663;
  return(result);
}

const char coeff_description2[] =
"  << Stokes-Darcy2Field Coefficients >> \n"
"  A = delta^2*eta \n"
"  B = phi*k_hat + f2u (fux,fuz - manufactured)\n" 
"  C = f2p (manufactured) \n"
"  D1 = delta^2*xi, xi = zeta-2/3eta \n"
"  D2 = -K \n"
"  D3 = -K*k_hat \n";

const char coeff_description3[] =
"  << Stokes-Darcy3Field Coefficients >> \n"
"  A = delta^2*eta \n"
"  B = phi*k_hat + f3u (fux,fuz - manufactured)\n" 
"  C = f3p (manufactured) \n"
"  D1 = -1/(delta^2*xi), xi = zeta-2/3eta \n"
"  D2 = -K \n"
"  D3 = -K*k_hat \n"
"  D4 = -K \n"
"  DC = f3pc (manufactured) \n";

const char bc_description[] =
"  << Stokes-Darcy BCs >> \n"
"  LEFT, RIGHT, DOWN, UP: DIRICHLET (manufactured) \n";

// ---------------------------------------
// Application functions
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "StokesDarcy2Field_Numerical"
PetscErrorCode StokesDarcy2Field_Numerical(void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  FDPDE          fd;
  DM             dm;
  Vec            x, xMMS;
  PetscInt       nx, nz;
  PetscScalar    xmin, zmin, xmax, zmax;
  char           fout[FNAME_LENGTH];
  PetscErrorCode ierr;
  PetscFunctionBegin;

  // Element count
  nx = usr->par->nx;
  nz = usr->par->nz;
  xmin = usr->par->xmin;
  zmin = usr->par->zmin;
  xmax = usr->par->xmin+usr->par->L;
  zmax = usr->par->zmin+usr->par->H;

  // Create the FD-pde object
  ierr = FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_STOKESDARCY2FIELD,&fd);CHKERRQ(ierr);
  ierr = FDPDESetUp(fd);CHKERRQ(ierr);
  ierr = FDPDEGetDM(fd,&dm); CHKERRQ(ierr);

  // Set BC and coefficient evaluation functions
  ierr = FDPDESetFunctionBCList(fd,FormBCList2Field,bc_description,usr); CHKERRQ(ierr);
  ierr = FDPDESetFunctionCoefficient(fd,FormCoefficient2Field,coeff_description2,usr); CHKERRQ(ierr);
  ierr = FDPDEView(fd); CHKERRQ(ierr);

  // FD SNES Solver
  ierr = FDPDESolve(fd,NULL);CHKERRQ(ierr);
  ierr = FDPDEGetSolution(fd,&x);CHKERRQ(ierr); 

  // Create manufactured solution and compute norms
  ierr = ComputeManufacturedSolution2Field(dm,&xMMS,usr); CHKERRQ(ierr);
  ierr = ComputeErrorNorms2Field(dm,x,xMMS);CHKERRQ(ierr);

  // Output extra parameters - phi, K, zeta
  ierr = ComputeExtraParameters(dm,usr); CHKERRQ(ierr);

  // Output solution to file
  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/%s_sd2field_sol",usr->par->fdir_out,usr->par->fname_out);
  ierr = DMStagViewBinaryPython(dm,x,fout);CHKERRQ(ierr);

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/%s_sd2field_mms",usr->par->fdir_out,usr->par->fname_out);
  ierr = DMStagViewBinaryPython(dm,xMMS,fout);CHKERRQ(ierr);

  // Destroy objects
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&xMMS);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = FDPDEDestroy(&fd);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormCoefficient2Field"
PetscErrorCode FormCoefficient2Field(FDPDE fd, DM dm, Vec x, DM dmcoeff, Vec coeff, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       i, j, sx, sz, nx, nz;
  Vec            coefflocal;
  PetscScalar    **coordx,**coordz;
  PetscInt       iprev, inext, icenter;
  PetscScalar    delta,phi0,phia,phi_min,vzeta,eta,p_s,psi_s,U_s,m,n,k_hat;
  PetscScalar    ***c;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  delta = usr->par->delta;
  phi0 = usr->par->phi0;
  phia = usr->par->phia;
  phi_min = usr->par->phi_min;
  vzeta = usr->par->vzeta;
  eta  = usr->par->eta;
  p_s = usr->par->p_s;
  psi_s = usr->par->psi_s;
  U_s = usr->par->U_s;
  m = usr->par->m;
  n = usr->par->n;
  k_hat = usr->par->k_hat;

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
      PetscInt idx;

      { // A = delta^2*eta (center, c=1)
        PetscScalar   eta;
        DMStagStencil point;
        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 1;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        eta = get_eta(coordx[i][icenter],coordz[j][icenter],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
        c[j][i][idx] = delta*delta*eta;
      }

      { // A = delta^2*eta (corner, c=0)
        DMStagStencil point[4];
        PetscInt      ii;
        PetscScalar   eta, xp[4], zp[4];

        point[0].i = i; point[0].j = j; point[0].loc = DOWN_LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = DOWN_RIGHT; point[1].c = 0;
        point[2].i = i; point[2].j = j; point[2].loc = UP_LEFT;    point[2].c = 0;
        point[3].i = i; point[3].j = j; point[3].loc = UP_RIGHT;   point[3].c = 0;

        xp[0] = coordx[i][iprev  ]; zp[0] = coordz[j][iprev  ];
        xp[1] = coordx[i][inext  ]; zp[1] = coordz[j][iprev  ];
        xp[2] = coordx[i][iprev  ]; zp[2] = coordz[j][inext  ];
        xp[3] = coordx[i][inext  ]; zp[3] = coordz[j][inext  ];

        for (ii = 0; ii < 4; ii++) {
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          eta = get_eta(xp[ii],zp[ii],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
          c[j][i][idx] = delta*delta*eta;
        }
      }

      { // B = phi*k_hat + f2u (fux,fuz - manufactured) (edges, c=0)
        DMStagStencil point[4];
        PetscScalar   xp[4],zp[4],rhs[4];
        PetscInt      ii;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 0;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = 0;
        point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = 0;

        xp[0] = coordx[i][iprev  ]; zp[0] = coordz[j][icenter];
        xp[1] = coordx[i][inext  ]; zp[1] = coordz[j][icenter];
        xp[2] = coordx[i][icenter]; zp[2] = coordz[j][iprev  ];
        xp[3] = coordx[i][icenter]; zp[3] = coordz[j][inext  ];

        // Bx = f2ux
        rhs[0] = get_f2ux(xp[0],zp[0],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
        rhs[1] = get_f2ux(xp[1],zp[1],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);

        // Bz = phi*k_hat + f2uz
        rhs[2] = get_f2uz(xp[2],zp[2],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
        rhs[3] = get_f2uz(xp[3],zp[3],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);

        rhs[2] += k_hat*get_phi(xp[2],zp[2],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
        rhs[3] += k_hat*get_phi(xp[3],zp[3],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);

        for (ii = 0; ii < 4; ii++) {
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = rhs[ii];
        }
      }

      { // C =  f2p (manufactured) (center, c=0)
        DMStagStencil point;
        PetscScalar   xp, zp, rhs;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 0;
        xp = coordx[i][icenter]; zp = coordz[j][icenter];
        rhs = get_f2p(xp,zp,delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);

        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = rhs;
      }

      { // D1 = delta^2*xi (center, c=2)
        PetscScalar   xp, zp, xi;
        DMStagStencil point;
        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 2;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        xp = coordx[i][icenter]; zp = coordz[j][icenter];
        xi = get_xi(xp,zp,delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
        c[j][i][idx] = delta*delta*xi;
      }

      { // D2 = -K (edges, c=1)
        DMStagStencil point[4];
        PetscScalar   xp[4],zp[4];
        PetscInt      ii;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 1;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 1;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = 1;
        point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = 1;

        xp[0] = coordx[i][iprev  ]; zp[0] = coordz[j][icenter];
        xp[1] = coordx[i][inext  ]; zp[1] = coordz[j][icenter];
        xp[2] = coordx[i][icenter]; zp[2] = coordz[j][iprev  ];
        xp[3] = coordx[i][icenter]; zp[3] = coordz[j][inext  ];

        for (ii = 0; ii < 4; ii++) {
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = -get_K(xp[ii],zp[ii],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
        }
      }

      { // D3 = -K*k_hat (edges, c=2)
        DMStagStencil point[4];
        PetscScalar   xp[4],zp[4];
        PetscInt      ii;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 2;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 2;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = 2;
        point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = 2;

        xp[0] = coordx[i][iprev  ]; zp[0] = coordz[j][icenter];
        xp[1] = coordx[i][inext  ]; zp[1] = coordz[j][icenter];
        xp[2] = coordx[i][icenter]; zp[2] = coordz[j][iprev  ];
        xp[3] = coordx[i][icenter]; zp[3] = coordz[j][inext  ];

        for (ii = 0; ii < 2; ii++) {
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = 0.0;
        }

        for (ii = 2; ii < 4; ii++) {
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = -k_hat*get_K(xp[ii],zp[ii],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
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
#undef __FUNCT__
#define __FUNCT__ "FormBCList2Field"
PetscErrorCode FormBCList2Field(DM dm, Vec x, DMStagBCList bclist, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       k,n_bc,*idx_bc;
  PetscScalar    *value_bc,*x_bc;
  PetscScalar    delta,phi0,phia,phi_min,vzeta,eta,p_s,psi_s,U_s,m,n,k_hat;
  BCType         *type_bc;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  delta = usr->par->delta;
  phi0 = usr->par->phi0;
  phia = usr->par->phia;
  phi_min = usr->par->phi_min;
  vzeta = usr->par->vzeta;
  eta  = usr->par->eta;
  p_s = usr->par->p_s;
  psi_s = usr->par->psi_s;
  U_s = usr->par->U_s;
  m = usr->par->m;
  n = usr->par->n;
  k_hat = usr->par->k_hat;

  // LEFT Boundary - Vx
  ierr = DMStagBCListGetValues(bclist,'w','-',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_ux(x_bc[2*k],x_bc[2*k+1],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // LEFT Boundary - Vz
  ierr = DMStagBCListGetValues(bclist,'w','|',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=1; k<n_bc-1; k++) {
    value_bc[k] = get_uz(x_bc[2*k],x_bc[2*k+1],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // LEFT Boundary - P
  ierr = DMStagBCListGetValues(bclist,'w','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_p(x_bc[2*k],x_bc[2*k+1],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  
  // RIGHT Boundary - Vx
  ierr = DMStagBCListGetValues(bclist,'e','-',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_ux(x_bc[2*k],x_bc[2*k+1],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // RIGHT Boundary - Vz
  ierr = DMStagBCListGetValues(bclist,'e','|',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=1; k<n_bc-1; k++) {
    value_bc[k] = get_uz(x_bc[2*k],x_bc[2*k+1],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // RIGHT Boundary - P
  ierr = DMStagBCListGetValues(bclist,'e','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_p(x_bc[2*k],x_bc[2*k+1],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN Boundary - Vx
  ierr = DMStagBCListGetValues(bclist,'s','-',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=1; k<n_bc-1; k++) {
    value_bc[k] = get_ux(x_bc[2*k],x_bc[2*k+1],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN Boundary - Vz
  ierr = DMStagBCListGetValues(bclist,'s','|',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_uz(x_bc[2*k],x_bc[2*k+1],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN Boundary - P
  ierr = DMStagBCListGetValues(bclist,'s','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_p(x_bc[2*k],x_bc[2*k+1],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP Boundary - Vx
  ierr = DMStagBCListGetValues(bclist,'n','-',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=1; k<n_bc-1; k++) {
    value_bc[k] = get_ux(x_bc[2*k],x_bc[2*k+1],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP Boundary - Vz
  ierr = DMStagBCListGetValues(bclist,'n','|',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_uz(x_bc[2*k],x_bc[2*k+1],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP Boundary - P
  ierr = DMStagBCListGetValues(bclist,'n','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_p(x_bc[2*k],x_bc[2*k+1],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
PetscErrorCode ComputeManufacturedSolution2Field(DM dm,Vec *_xMMS, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz, idx;
  PetscInt       iprev, inext, icenter;
  PetscScalar    ***xxMMS;
  PetscScalar    **coordx,**coordz;
  Vec            xMMS, xMMSlocal;
  PetscScalar    delta,phi0,phia,phi_min,vzeta,eta,p_s,psi_s,U_s,m,n,k_hat;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  delta = usr->par->delta;
  phi0 = usr->par->phi0;
  phia = usr->par->phia;
  phi_min = usr->par->phi_min;
  vzeta = usr->par->vzeta;
  eta  = usr->par->eta;
  p_s = usr->par->p_s;
  psi_s = usr->par->psi_s;
  U_s = usr->par->U_s;
  m = usr->par->m;
  n = usr->par->n;
  k_hat = usr->par->k_hat;

  // Create local and global vectors for MMS solution
  ierr = DMCreateGlobalVector(dm,&xMMS     ); CHKERRQ(ierr);
  ierr = DMCreateLocalVector (dm,&xMMSlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm,xMMSlocal,&xxMMS); CHKERRQ(ierr);

  // Get domain corners
  ierr = DMStagGetGlobalSizes(dm, &Nx, &Nz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  
// Get dm coordinates array
  ierr = DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter);CHKERRQ(ierr); 
  ierr = DMStagGetProductCoordinateLocationSlot(dm,LEFT,&iprev);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,RIGHT,&inext);CHKERRQ(ierr); 

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {

      // pressure
      ierr = DMStagGetLocationSlot(dm,ELEMENT,0,&idx); CHKERRQ(ierr);
      xxMMS[j][i][idx] = get_p(coordx[i][icenter],coordz[j][icenter],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);

      // ux
      ierr = DMStagGetLocationSlot(dm,LEFT,0,&idx); CHKERRQ(ierr);
      xxMMS[j][i][idx] = get_ux(coordx[i][iprev],coordz[j][icenter],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);

      if (i == Nx-1) {
        ierr = DMStagGetLocationSlot(dm,RIGHT,0,&idx); CHKERRQ(ierr);
        xxMMS[j][i][idx] = get_ux(coordx[i][inext],coordz[j][icenter],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
      }
      
      // uz
      ierr = DMStagGetLocationSlot(dm,DOWN,0,&idx); CHKERRQ(ierr);
      xxMMS[j][i][idx] = get_uz(coordx[i][icenter],coordz[j][iprev],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);

      if (j == Nz-1) {
        ierr = DMStagGetLocationSlot(dm,UP,0,&idx); CHKERRQ(ierr);
        xxMMS[j][i][idx] = get_uz(coordx[i][icenter],coordz[j][inext],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
      }
    }
  }

  // Restore arrays
  ierr = DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);

  // Restore and map local to global
  ierr = DMStagVecRestoreArray(dm,xMMSlocal,&xxMMS); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm,xMMSlocal,INSERT_VALUES,xMMS); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,xMMSlocal,INSERT_VALUES,xMMS); CHKERRQ(ierr);
  ierr = VecDestroy(&xMMSlocal); CHKERRQ(ierr);

  // Assign pointers
  *_xMMS  = xMMS;
  
  PetscFunctionReturn(0);
}

// ---------------------------------------
// ComputeErrorNorms
// ---------------------------------------
PetscErrorCode ComputeErrorNorms2Field(DM dm,Vec x,Vec xMMS)
{
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz;
  PetscScalar    xx[5], xa[5], dx, dz, dv;
  PetscScalar    nrm2p, nrm2v, nrm2vx, nrm2vz, sum_err[3], gsum_err[3], sum_mms[3], gsum_mms[3];
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
  ierr = DMGlobalToLocal (dm, xMMS, INSERT_VALUES, xalocal); CHKERRQ(ierr);

  // Initialize norms
  sum_err[0] = 0.0; sum_err[1] = 0.0; sum_err[2] = 0.0;
  sum_mms[0] = 0.0; sum_mms[1] = 0.0; sum_mms[2] = 0.0;

  dx = 1.0/Nx;
  dz = 1.0/Nz;
  dv = dx*dz;

  // Loop over local domain to calculate ELEMENT errors
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      
      PetscScalar    ve[4], pe, v[4], p;
      DMStagStencil  point[5];
      
      // Get stencil values
      point[0].i = i; point[0].j = j; point[0].loc = LEFT;    point[0].c = 0; // Vx
      point[1].i = i; point[1].j = j; point[1].loc = RIGHT;   point[1].c = 0; // Vx
      point[2].i = i; point[2].j = j; point[2].loc = DOWN;    point[2].c = 0; // Vz
      point[3].i = i; point[3].j = j; point[3].loc = UP;      point[3].c = 0; // Vz
      point[4].i = i; point[4].j = j; point[4].loc = ELEMENT; point[4].c = 0; // P

      // Get numerical solution
      ierr = DMStagVecGetValuesStencil(dm, xlocal, 5, point, xx); CHKERRQ(ierr);

      // Get analytical solution
      ierr = DMStagVecGetValuesStencil(dm, xalocal, 5, point, xa); CHKERRQ(ierr);

      // Error vectors - squared
      ve[0] = (xx[0]-xa[0])*(xx[0]-xa[0]); // Left
      ve[1] = (xx[1]-xa[1])*(xx[1]-xa[1]); // Right
      ve[2] = (xx[2]-xa[2])*(xx[2]-xa[2]); // Down
      ve[3] = (xx[3]-xa[3])*(xx[3]-xa[3]); // Up
      pe    = (xx[4]-xa[4])*(xx[4]-xa[4]); // elem

      // MMS values - squared
      v[0] = xa[0]*xa[0]; // Left
      v[1] = xa[1]*xa[1]; // Right
      v[2] = xa[2]*xa[2]; // Down
      v[3] = xa[3]*xa[3]; // Up
      p    = xa[4]*xa[4]; // elem

      // Calculate sums for L2 norms as in Katz, ch 13 - but taking into account the staggered grid
      if      (i == 0   ) { sum_err[0] += ve[0]*dv*0.5; sum_err[0] += ve[1]*dv; }
      else if (i == Nx-1) sum_err[0] += ve[1]*dv*0.5;
      else                sum_err[0] += ve[1]*dv;

      if      (j == 0   ) { sum_err[1] += ve[2]*dv*0.5; sum_err[1] += ve[3]*dv; }
      else if (j == Nz-1) sum_err[1] += ve[3]*dv*0.5;
      else                sum_err[1] += ve[3]*dv;
      sum_err[2] += pe*dv;

      if      (i == 0   ) { sum_mms[0] += v[0]*dv*0.5; sum_mms[0] += v[1]*dv; }
      else if (i == Nx-1) sum_mms[0] += v[1]*dv*0.5;
      else                sum_mms[0] += v[1]*dv;

      if      (j == 0   ) { sum_mms[1] += v[2]*dv*0.5; sum_mms[1] += v[3]*dv; }
      else if (j == Nz-1) sum_mms[1] += v[3]*dv*0.5;
      else                sum_mms[1] += v[3]*dv;
      sum_mms[2] += p*dv;
    }
  }

  // Collect data 
  ierr = MPI_Allreduce(&sum_err, &gsum_err, 3, MPI_DOUBLE, MPI_SUM, comm); CHKERRQ(ierr);
  ierr = MPI_Allreduce(&sum_mms, &gsum_mms, 3, MPI_DOUBLE, MPI_SUM, comm); CHKERRQ(ierr);

  // L2 error norm = sqrt(gsum_err/gsum_mms)
  nrm2vx = PetscSqrtScalar(gsum_err[0]/gsum_mms[0]);
  nrm2vz = PetscSqrtScalar(gsum_err[1]/gsum_mms[1]);
  nrm2p  = PetscSqrtScalar(gsum_err[2]/gsum_mms[2]);
  nrm2v  = PetscSqrtScalar((gsum_err[0]+gsum_err[1])/(gsum_mms[0]+gsum_mms[1]));

  // Restore arrays and vectors
  ierr = DMRestoreLocalVector(dm, &xlocal ); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &xalocal ); CHKERRQ(ierr);

  // Print information
  PetscPrintf(comm,"# --------------------------------------- #\n");
  PetscPrintf(comm,"# NORMS 2-FIELD: \n");
  PetscPrintf(comm,"# Velocity: norm2 = %1.12e norm2x = %1.12e norm2z = %1.12e \n",nrm2v,nrm2vx,nrm2vz);
  PetscPrintf(comm,"# Pressure: norm2 = %1.12e\n",nrm2p);
  PetscPrintf(comm,"# Grid info: hx = %1.12e hz = %1.12e\n",dx,dz);

  PetscFunctionReturn(0);
}

// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "StokesDarcy3Field_Numerical"
PetscErrorCode StokesDarcy3Field_Numerical(void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  FDPDE          fd;
  DM             dm;
  Vec            x, xMMS;
  PetscInt       nx, nz;
  PetscScalar    xmin, zmin, xmax, zmax;
  char           fout[FNAME_LENGTH];
  PetscErrorCode ierr;
  PetscFunctionBegin;

  // Element count
  nx = usr->par->nx;
  nz = usr->par->nz;
  xmin = usr->par->xmin;
  zmin = usr->par->zmin;
  xmax = usr->par->xmin+usr->par->L;
  zmax = usr->par->zmin+usr->par->H;

  // Create the FD-pde object
  ierr = FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_STOKESDARCY3FIELD,&fd);CHKERRQ(ierr);
  ierr = FDPDESetUp(fd);CHKERRQ(ierr);
  ierr = FDPDEGetDM(fd,&dm); CHKERRQ(ierr);

  // Set BC and coefficient evaluation functions
  ierr = FDPDESetFunctionBCList(fd,FormBCList3Field,bc_description,usr); CHKERRQ(ierr);
  ierr = FDPDESetFunctionCoefficient(fd,FormCoefficient3Field,coeff_description3,usr); CHKERRQ(ierr);
  ierr = FDPDEView(fd); CHKERRQ(ierr);

  // FD SNES Solver
  ierr = FDPDESolve(fd,NULL);CHKERRQ(ierr);
  ierr = FDPDEGetSolution(fd,&x);CHKERRQ(ierr); 

  // Create manufactured solution and compute norms
  ierr = ComputeManufacturedSolution3Field(dm,&xMMS,usr); CHKERRQ(ierr);
  ierr = ComputeErrorNorms3Field(dm,x,xMMS);CHKERRQ(ierr);

  // Output solution to file
  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/%s_sd3field_sol",usr->par->fdir_out,usr->par->fname_out);
  ierr = DMStagViewBinaryPython(dm,x,fout);CHKERRQ(ierr);

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/%s_sd3field_mms",usr->par->fdir_out,usr->par->fname_out);
  ierr = DMStagViewBinaryPython(dm,xMMS,fout);CHKERRQ(ierr);

  // Destroy objects
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&xMMS);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = FDPDEDestroy(&fd);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormCoefficient3Field"
PetscErrorCode FormCoefficient3Field(FDPDE fd, DM dm, Vec x, DM dmcoeff, Vec coeff, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       i, j, sx, sz, nx, nz;
  Vec            coefflocal;
  PetscScalar    **coordx,**coordz;
  PetscInt       iprev, inext, icenter;
  PetscScalar    delta,phi0,phia,phi_min,vzeta,eta,p_s,psi_s,U_s,m,n,k_hat;
  PetscScalar    ***c;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  delta = usr->par->delta;
  phi0 = usr->par->phi0;
  phia = usr->par->phia;
  phi_min = usr->par->phi_min;
  vzeta = usr->par->vzeta;
  eta  = usr->par->eta;
  p_s = usr->par->p_s;
  psi_s = usr->par->psi_s;
  U_s = usr->par->U_s;
  m = usr->par->m;
  n = usr->par->n;
  k_hat = usr->par->k_hat;

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
      PetscInt idx;

      { // A = delta^2*eta (center, c=1)
        PetscScalar   eta;
        DMStagStencil point;
        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = SD3_COEFF_ELEMENT_A;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        eta = get_eta(coordx[i][icenter],coordz[j][icenter],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
        c[j][i][idx] = delta*delta*eta;
      }

      { // A = delta^2*eta (corner, c=0)
        DMStagStencil point[4];
        PetscInt      ii;
        PetscScalar   eta, xp[4], zp[4];

        point[0].i = i; point[0].j = j; point[0].loc = DOWN_LEFT;  point[0].c = SD3_COEFF_VERTEX_A;
        point[1].i = i; point[1].j = j; point[1].loc = DOWN_RIGHT; point[1].c = SD3_COEFF_VERTEX_A;
        point[2].i = i; point[2].j = j; point[2].loc = UP_LEFT;    point[2].c = SD3_COEFF_VERTEX_A;
        point[3].i = i; point[3].j = j; point[3].loc = UP_RIGHT;   point[3].c = SD3_COEFF_VERTEX_A;

        xp[0] = coordx[i][iprev  ]; zp[0] = coordz[j][iprev  ];
        xp[1] = coordx[i][inext  ]; zp[1] = coordz[j][iprev  ];
        xp[2] = coordx[i][iprev  ]; zp[2] = coordz[j][inext  ];
        xp[3] = coordx[i][inext  ]; zp[3] = coordz[j][inext  ];

        for (ii = 0; ii < 4; ii++) {
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          eta = get_eta(xp[ii],zp[ii],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
          c[j][i][idx] = delta*delta*eta;
        }
      }

      { // B = phi*k_hat + f3u (fux,fuz - manufactured) (edges, c=0)
        DMStagStencil point[4];
        PetscScalar   xp[4],zp[4],rhs[4];
        PetscInt      ii;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = SD3_COEFF_FACE_B;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = SD3_COEFF_FACE_B;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = SD3_COEFF_FACE_B;
        point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = SD3_COEFF_FACE_B;

        xp[0] = coordx[i][iprev  ]; zp[0] = coordz[j][icenter];
        xp[1] = coordx[i][inext  ]; zp[1] = coordz[j][icenter];
        xp[2] = coordx[i][icenter]; zp[2] = coordz[j][iprev  ];
        xp[3] = coordx[i][icenter]; zp[3] = coordz[j][inext  ];

        // Bx = f3ux
        rhs[0] = get_f3ux(xp[0],zp[0],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
        rhs[1] = get_f3ux(xp[1],zp[1],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);

        // Bz = phi*k_hat + f3uz
        rhs[2] = get_f3uz(xp[2],zp[2],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
        rhs[3] = get_f3uz(xp[3],zp[3],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);

        rhs[2] += k_hat*get_phi(xp[2],zp[2],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
        rhs[3] += k_hat*get_phi(xp[3],zp[3],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);

        for (ii = 0; ii < 4; ii++) {
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = rhs[ii];
        }
      }

      { // C =  f3p (manufactured) (center, c=0)
        DMStagStencil point;
        PetscScalar   xp, zp, rhs;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = SD3_COEFF_ELEMENT_C;
        xp = coordx[i][icenter]; zp = coordz[j][icenter];
        rhs = get_f3p(xp,zp,delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);

        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = rhs;
      }

      { // D1 = -1/(delta^2*xi) (center, c=2)
        PetscScalar   xp, zp, xi;
        DMStagStencil point;
        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = SD3_COEFF_ELEMENT_D1;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        xp = coordx[i][icenter]; zp = coordz[j][icenter];
        xi = get_xi(xp,zp,delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
        c[j][i][idx] = -1.0/(delta*delta*xi);
      }

      { // D2 = -K (edges, c=1)
        DMStagStencil point[4];
        PetscScalar   xp[4],zp[4];
        PetscInt      ii;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = SD3_COEFF_FACE_D2;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = SD3_COEFF_FACE_D2;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = SD3_COEFF_FACE_D2;
        point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = SD3_COEFF_FACE_D2;

        xp[0] = coordx[i][iprev  ]; zp[0] = coordz[j][icenter];
        xp[1] = coordx[i][inext  ]; zp[1] = coordz[j][icenter];
        xp[2] = coordx[i][icenter]; zp[2] = coordz[j][iprev  ];
        xp[3] = coordx[i][icenter]; zp[3] = coordz[j][inext  ];

        for (ii = 0; ii < 4; ii++) {
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = -get_K(xp[ii],zp[ii],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
        }
      }

      { // D3 = -K*k_hat (edges, c=2)
        DMStagStencil point[4];
        PetscScalar   xp[4],zp[4];
        PetscInt      ii;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = SD3_COEFF_FACE_D3;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = SD3_COEFF_FACE_D3;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = SD3_COEFF_FACE_D3;
        point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = SD3_COEFF_FACE_D3;

        xp[0] = coordx[i][iprev  ]; zp[0] = coordz[j][icenter];
        xp[1] = coordx[i][inext  ]; zp[1] = coordz[j][icenter];
        xp[2] = coordx[i][icenter]; zp[2] = coordz[j][iprev  ];
        xp[3] = coordx[i][icenter]; zp[3] = coordz[j][inext  ];

        for (ii = 0; ii < 2; ii++) {
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = 0.0;
        }

        for (ii = 2; ii < 4; ii++) {
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = -k_hat*get_K(xp[ii],zp[ii],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
        }
      }

      { // D4 = -K (edges, c=3)
        DMStagStencil point[4];
        PetscScalar   xp[4],zp[4];
        PetscInt      ii;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = SD3_COEFF_FACE_D4;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = SD3_COEFF_FACE_D4;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = SD3_COEFF_FACE_D4;
        point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = SD3_COEFF_FACE_D4;

        xp[0] = coordx[i][iprev  ]; zp[0] = coordz[j][icenter];
        xp[1] = coordx[i][inext  ]; zp[1] = coordz[j][icenter];
        xp[2] = coordx[i][icenter]; zp[2] = coordz[j][iprev  ];
        xp[3] = coordx[i][icenter]; zp[3] = coordz[j][inext  ];

        for (ii = 0; ii < 4; ii++) {
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = -get_K(xp[ii],zp[ii],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
        }
      }

      { // DC =  f3pc (manufactured) (center, c=3)
        DMStagStencil point;
        PetscScalar   xp, zp, rhs;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = SD3_COEFF_ELEMENT_DC;
        xp = coordx[i][icenter]; zp = coordz[j][icenter];
        rhs = get_f3pc(xp,zp,delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);

        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = rhs;
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
#undef __FUNCT__
#define __FUNCT__ "FormBCList3Field"
PetscErrorCode FormBCList3Field(DM dm, Vec x, DMStagBCList bclist, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       k,n_bc,*idx_bc;
  PetscScalar    *value_bc,*x_bc;
  PetscScalar    delta,phi0,phia,phi_min,vzeta,eta,p_s,psi_s,U_s,m,n,k_hat;
  BCType         *type_bc;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  delta = usr->par->delta;
  phi0 = usr->par->phi0;
  phia = usr->par->phia;
  phi_min = usr->par->phi_min;
  vzeta = usr->par->vzeta;
  eta  = usr->par->eta;
  p_s = usr->par->p_s;
  psi_s = usr->par->psi_s;
  U_s = usr->par->U_s;
  m = usr->par->m;
  n = usr->par->n;
  k_hat = usr->par->k_hat;

  // LEFT Boundary - Vx
  ierr = DMStagBCListGetValues(bclist,'w','-',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_ux(x_bc[2*k],x_bc[2*k+1],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // LEFT Boundary - Vz
  ierr = DMStagBCListGetValues(bclist,'w','|',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=1; k<n_bc-1; k++) {
    value_bc[k] = get_uz(x_bc[2*k],x_bc[2*k+1],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // LEFT Boundary - P
  ierr = DMStagBCListGetValues(bclist,'w','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_p(x_bc[2*k],x_bc[2*k+1],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  
  // RIGHT Boundary - Vx
  ierr = DMStagBCListGetValues(bclist,'e','-',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_ux(x_bc[2*k],x_bc[2*k+1],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // RIGHT Boundary - Vz
  ierr = DMStagBCListGetValues(bclist,'e','|',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=1; k<n_bc-1; k++) {
    value_bc[k] = get_uz(x_bc[2*k],x_bc[2*k+1],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // RIGHT Boundary - P
  ierr = DMStagBCListGetValues(bclist,'e','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_p(x_bc[2*k],x_bc[2*k+1],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN Boundary - Vx
  ierr = DMStagBCListGetValues(bclist,'s','-',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=1; k<n_bc-1; k++) {
    value_bc[k] = get_ux(x_bc[2*k],x_bc[2*k+1],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN Boundary - Vz
  ierr = DMStagBCListGetValues(bclist,'s','|',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_uz(x_bc[2*k],x_bc[2*k+1],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN Boundary - P
  ierr = DMStagBCListGetValues(bclist,'s','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_p(x_bc[2*k],x_bc[2*k+1],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP Boundary - Vx
  ierr = DMStagBCListGetValues(bclist,'n','-',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=1; k<n_bc-1; k++) {
    value_bc[k] = get_ux(x_bc[2*k],x_bc[2*k+1],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP Boundary - Vz
  ierr = DMStagBCListGetValues(bclist,'n','|',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_uz(x_bc[2*k],x_bc[2*k+1],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP Boundary - P
  ierr = DMStagBCListGetValues(bclist,'n','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_p(x_bc[2*k],x_bc[2*k+1],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // Compaction pressure
  // Left
  ierr = DMStagBCListGetValues(bclist,'w','o',1,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_pc(x_bc[2*k],x_bc[2*k+1],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',1,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  
  // RIGHT
  ierr = DMStagBCListGetValues(bclist,'e','o',1,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_pc(x_bc[2*k],x_bc[2*k+1],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',1,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN
  ierr = DMStagBCListGetValues(bclist,'s','o',1,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_pc(x_bc[2*k],x_bc[2*k+1],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',1,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP
  ierr = DMStagBCListGetValues(bclist,'n','o',1,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_pc(x_bc[2*k],x_bc[2*k+1],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  ierr = DMStagBCListInsertValues(bclist,'o',1,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
PetscErrorCode ComputeManufacturedSolution3Field(DM dm,Vec *_xMMS, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz, idx;
  PetscInt       iprev, inext, icenter;
  PetscScalar    ***xxMMS;
  PetscScalar    **coordx,**coordz;
  Vec            xMMS, xMMSlocal;
  PetscScalar    delta,phi0,phia,phi_min,vzeta,eta,p_s,psi_s,U_s,m,n,k_hat;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  delta = usr->par->delta;
  phi0 = usr->par->phi0;
  phia = usr->par->phia;
  phi_min = usr->par->phi_min;
  vzeta = usr->par->vzeta;
  eta  = usr->par->eta;
  p_s = usr->par->p_s;
  psi_s = usr->par->psi_s;
  U_s = usr->par->U_s;
  m = usr->par->m;
  n = usr->par->n;
  k_hat = usr->par->k_hat;

  // Create local and global vectors for MMS solution
  ierr = DMCreateGlobalVector(dm,&xMMS     ); CHKERRQ(ierr);
  ierr = DMCreateLocalVector (dm,&xMMSlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm,xMMSlocal,&xxMMS); CHKERRQ(ierr);

  // Get domain corners
  ierr = DMStagGetGlobalSizes(dm, &Nx, &Nz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  
// Get dm coordinates array
  ierr = DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter);CHKERRQ(ierr); 
  ierr = DMStagGetProductCoordinateLocationSlot(dm,LEFT,&iprev);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,RIGHT,&inext);CHKERRQ(ierr); 

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {

      // pressure
      ierr = DMStagGetLocationSlot(dm,ELEMENT,0,&idx); CHKERRQ(ierr);
      xxMMS[j][i][idx] = get_p(coordx[i][icenter],coordz[j][icenter],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);

      // compaction pressure
      ierr = DMStagGetLocationSlot(dm,ELEMENT,1,&idx); CHKERRQ(ierr);
      xxMMS[j][i][idx] = get_pc(coordx[i][icenter],coordz[j][icenter],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);

      // ux
      ierr = DMStagGetLocationSlot(dm,LEFT,0,&idx); CHKERRQ(ierr);
      xxMMS[j][i][idx] = get_ux(coordx[i][iprev],coordz[j][icenter],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);

      if (i == Nx-1) {
        ierr = DMStagGetLocationSlot(dm,RIGHT,0,&idx); CHKERRQ(ierr);
        xxMMS[j][i][idx] = get_ux(coordx[i][inext],coordz[j][icenter],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
      }
      
      // uz
      ierr = DMStagGetLocationSlot(dm,DOWN,0,&idx); CHKERRQ(ierr);
      xxMMS[j][i][idx] = get_uz(coordx[i][icenter],coordz[j][iprev],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);

      if (j == Nz-1) {
        ierr = DMStagGetLocationSlot(dm,UP,0,&idx); CHKERRQ(ierr);
        xxMMS[j][i][idx] = get_uz(coordx[i][icenter],coordz[j][inext],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
      }
    }
  }

  // Restore arrays
  ierr = DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);

  // Restore and map local to global
  ierr = DMStagVecRestoreArray(dm,xMMSlocal,&xxMMS); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm,xMMSlocal,INSERT_VALUES,xMMS); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,xMMSlocal,INSERT_VALUES,xMMS); CHKERRQ(ierr);
  ierr = VecDestroy(&xMMSlocal); CHKERRQ(ierr);

  // Assign pointers
  *_xMMS  = xMMS;
  
  PetscFunctionReturn(0);
}

// ---------------------------------------
PetscErrorCode ComputeErrorNorms3Field(DM dm,Vec x,Vec xMMS)
{
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz;
  PetscScalar    xx[6], xa[6], dx, dz, dv;
  PetscScalar    nrm2p, nrm2pc, nrm2v, nrm2vx, nrm2vz, sum_err[4], gsum_err[4], sum_mms[4], gsum_mms[4];
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
  ierr = DMGlobalToLocal (dm, xMMS, INSERT_VALUES, xalocal); CHKERRQ(ierr);

  // Initialize norms
  sum_err[0] = 0.0; sum_err[1] = 0.0; sum_err[2] = 0.0; sum_err[3] = 0.0;
  sum_mms[0] = 0.0; sum_mms[1] = 0.0; sum_mms[2] = 0.0; sum_mms[3] = 0.0;

  dx = 1.0/Nx;
  dz = 1.0/Nz;
  dv = dx*dz;

  // Loop over local domain to calculate ELEMENT errors
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      
      PetscScalar    ve[4], pe, pce, pc, v[4], p;
      DMStagStencil  point[6];
      
      // Get stencil values
      point[0].i = i; point[0].j = j; point[0].loc = LEFT;    point[0].c = 0; // Vx
      point[1].i = i; point[1].j = j; point[1].loc = RIGHT;   point[1].c = 0; // Vx
      point[2].i = i; point[2].j = j; point[2].loc = DOWN;    point[2].c = 0; // Vz
      point[3].i = i; point[3].j = j; point[3].loc = UP;      point[3].c = 0; // Vz
      point[4].i = i; point[4].j = j; point[4].loc = ELEMENT; point[4].c = 0; // P
      point[5].i = i; point[5].j = j; point[5].loc = ELEMENT; point[5].c = 1; // PC

      // Get numerical solution
      ierr = DMStagVecGetValuesStencil(dm, xlocal, 6, point, xx); CHKERRQ(ierr);

      // Get analytical solution
      ierr = DMStagVecGetValuesStencil(dm, xalocal, 6, point, xa); CHKERRQ(ierr);

      // Error vectors - squared
      ve[0] = (xx[0]-xa[0])*(xx[0]-xa[0]); // Left
      ve[1] = (xx[1]-xa[1])*(xx[1]-xa[1]); // Right
      ve[2] = (xx[2]-xa[2])*(xx[2]-xa[2]); // Down
      ve[3] = (xx[3]-xa[3])*(xx[3]-xa[3]); // Up
      pe    = (xx[4]-xa[4])*(xx[4]-xa[4]); // elem
      pce   = (xx[5]-xa[5])*(xx[5]-xa[5]); // Pc elem

      // MMS values - squared
      v[0] = xa[0]*xa[0]; // Left
      v[1] = xa[1]*xa[1]; // Right
      v[2] = xa[2]*xa[2]; // Down
      v[3] = xa[3]*xa[3]; // Up
      p    = xa[4]*xa[4]; // elem
      pc   = xa[5]*xa[5]; // pc elem

      // Calculate sums for L2 norms as in Katz, ch 13 - but taking into account the staggered grid
      if      (i == 0   ) { sum_err[0] += ve[0]*dv*0.5; sum_err[0] += ve[1]*dv; }
      else if (i == Nx-1) sum_err[0] += ve[1]*dv*0.5;
      else                sum_err[0] += ve[1]*dv;

      if      (j == 0   ) { sum_err[1] += ve[2]*dv*0.5; sum_err[1] += ve[3]*dv; }
      else if (j == Nz-1) sum_err[1] += ve[3]*dv*0.5;
      else                sum_err[1] += ve[3]*dv;
      sum_err[2] += pe*dv;
      sum_err[3] += pce*dv;

      if      (i == 0   ) { sum_mms[0] += v[0]*dv*0.5; sum_mms[0] += v[1]*dv; }
      else if (i == Nx-1) sum_mms[0] += v[1]*dv*0.5;
      else                sum_mms[0] += v[1]*dv;

      if      (j == 0   ) { sum_mms[1] += v[2]*dv*0.5; sum_mms[1] += v[3]*dv; }
      else if (j == Nz-1) sum_mms[1] += v[3]*dv*0.5;
      else                sum_mms[1] += v[3]*dv;
      sum_mms[2] += p*dv;
      sum_mms[3] += pc*dv;
    }
  }

  // Collect data 
  ierr = MPI_Allreduce(&sum_err, &gsum_err, 4, MPI_DOUBLE, MPI_SUM, comm); CHKERRQ(ierr);
  ierr = MPI_Allreduce(&sum_mms, &gsum_mms, 4, MPI_DOUBLE, MPI_SUM, comm); CHKERRQ(ierr);

  // L2 error norm = sqrt(gsum_err/gsum_mms)
  nrm2vx = PetscSqrtScalar(gsum_err[0]/gsum_mms[0]);
  nrm2vz = PetscSqrtScalar(gsum_err[1]/gsum_mms[1]);
  nrm2p  = PetscSqrtScalar(gsum_err[2]/gsum_mms[2]);
  nrm2pc = PetscSqrtScalar(gsum_err[3]/gsum_mms[3]);
  nrm2v  = PetscSqrtScalar((gsum_err[0]+gsum_err[1])/(gsum_mms[0]+gsum_mms[1]));

  // Restore arrays and vectors
  ierr = DMRestoreLocalVector(dm, &xlocal ); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &xalocal ); CHKERRQ(ierr);

  // Print information
  PetscPrintf(comm,"# --------------------------------------- #\n");
  PetscPrintf(comm,"# NORMS 3-FIELD: \n");
  PetscPrintf(comm,"# Velocity: norm2 = %1.12e norm2x = %1.12e norm2z = %1.12e \n",nrm2v,nrm2vx,nrm2vz);
  PetscPrintf(comm,"# Pressure: norm2 = %1.12e\n",nrm2p);
  PetscPrintf(comm,"# Compaction Pressure: norm2 = %1.12e\n",nrm2pc);
  PetscPrintf(comm,"# Grid info: hx = %1.12e hz = %1.12e\n",dx,dz);

  PetscFunctionReturn(0);
}

// ---------------------------------------
PetscErrorCode ComputeExtraParameters(DM dm, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz, idx;
  PetscInt       iprev, inext, icenter;
  PetscScalar    ***xx;
  PetscScalar    **coordx,**coordz;
  DM             dmextra;
  Vec            x, xlocal;
  PetscScalar    delta,phi0,phia,phi_min,vzeta,eta,p_s,psi_s,U_s,m,n,k_hat;
  char           fout[FNAME_LENGTH];
  PetscErrorCode ierr;
  PetscFunctionBegin;

  delta = usr->par->delta;
  phi0 = usr->par->phi0;
  phia = usr->par->phia;
  phi_min = usr->par->phi_min;
  vzeta = usr->par->vzeta;
  eta  = usr->par->eta;
  p_s = usr->par->p_s;
  psi_s = usr->par->psi_s;
  U_s = usr->par->U_s;
  m = usr->par->m;
  n = usr->par->n;
  k_hat = usr->par->k_hat;

  // Create new dmextra 
  ierr = DMStagCreateCompatibleDMStag(dm,0,0,5,0,&dmextra); CHKERRQ(ierr);
  ierr = DMSetUp(dmextra); CHKERRQ(ierr);
  ierr = DMStagSetUniformCoordinatesProduct(dmextra,usr->par->xmin,usr->par->xmin+usr->par->L,usr->par->zmin,usr->par->zmin+usr->par->H,0.0,0.0);CHKERRQ(ierr);

  // Create local and global vectors
  ierr = DMCreateGlobalVector(dmextra,&x); CHKERRQ(ierr);
  ierr = DMCreateLocalVector (dmextra,&xlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dmextra,xlocal,&xx); CHKERRQ(ierr);

  // Get domain corners
  ierr = DMStagGetGlobalSizes(dmextra, &Nx, &Nz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dmextra, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  
// Get dm coordinates array
  ierr = DMStagGetProductCoordinateArraysRead(dmextra,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmextra,ELEMENT,&icenter);CHKERRQ(ierr); 
  ierr = DMStagGetProductCoordinateLocationSlot(dmextra,LEFT,&iprev);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmextra,RIGHT,&inext);CHKERRQ(ierr); 

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {

      // phi - element 0
      ierr = DMStagGetLocationSlot(dmextra,ELEMENT,0,&idx); CHKERRQ(ierr);
      xx[j][i][idx] = get_phi(coordx[i][icenter],coordz[j][icenter],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);

      // K - element 1
      ierr = DMStagGetLocationSlot(dmextra,ELEMENT,1,&idx); CHKERRQ(ierr);
      xx[j][i][idx] = get_K(coordx[i][icenter],coordz[j][icenter],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);

      // zeta - element 2
      ierr = DMStagGetLocationSlot(dmextra,ELEMENT,2,&idx); CHKERRQ(ierr);
      xx[j][i][idx] = get_zeta(coordx[i][icenter],coordz[j][icenter],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);

      // eta - element 3
      ierr = DMStagGetLocationSlot(dmextra,ELEMENT,3,&idx); CHKERRQ(ierr);
      xx[j][i][idx] = get_eta(coordx[i][icenter],coordz[j][icenter],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);

      // xi - element 4
      ierr = DMStagGetLocationSlot(dmextra,ELEMENT,4,&idx); CHKERRQ(ierr);
      xx[j][i][idx] = get_xi(coordx[i][icenter],coordz[j][icenter],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
    }
  }

  // Restore arrays
  ierr = DMStagRestoreProductCoordinateArraysRead(dmextra,&coordx,&coordz,NULL);CHKERRQ(ierr);

  // Restore and map local to global
  ierr = DMStagVecRestoreArray(dmextra,xlocal,&xx); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dmextra,xlocal,INSERT_VALUES,x); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dmextra,xlocal,INSERT_VALUES,x); CHKERRQ(ierr);

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s/%s_extra_parameters",usr->par->fdir_out,usr->par->fname_out);
  ierr = DMStagViewBinaryPython(dmextra,x,fout);CHKERRQ(ierr);

  // clean 
  ierr = VecDestroy(&xlocal); CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = DMDestroy(&dmextra);CHKERRQ(ierr);
  
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
  ierr = PetscBagRegisterInt(bag, &par->nx, 4, "nx", "Element count in the x-dir"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->nz, 5, "nz", "Element count in the z-dir"); CHKERRQ(ierr);

  ierr = PetscBagRegisterScalar(bag, &par->xmin, 0.0, "xmin", "Start coordinate of domain in x-dir"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->zmin, 0.0, "zmin", "Start coordinate of domain in z-dir"); CHKERRQ(ierr);

  ierr = PetscBagRegisterScalar(bag, &par->L, 1.0, "L", "Length of domain in x-dir"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->H, 1.0, "H", "Height of domain in z-dir"); CHKERRQ(ierr);

  // Physical and material parameters
  ierr = PetscBagRegisterScalar(bag, &par->k_hat, -1.0, "k_hat", "Direction of unit vertical vector"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->delta, 1.0, "delta", "Dimensionless compaction length ~1-10"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->phi0, 0.1, "phi0", "Reference porosity"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->phia, 0.1, "phia", "Amplitude porosity perturbation"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->phi_min, 1.0e-6, "phi_min", "Cutoff porosity"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->vzeta, 10.0, "vzeta", "Reference bulk to shear viscosity ratio ~1e0-1e4"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->eta, 1.0, "eta", "Non-dimensional shear viscosity"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->p_s, 1.0, "p_s", "Pressure amplitude"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->psi_s, 1.0, "psi_s", "Vector potential function amplitude"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->U_s, 1.0, "U_s", "Scalar potential function amplitude"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->m, 2.0, "m", "Trigonometric coefficient"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->n, 3.0, "n", "Porosity exponent"); CHKERRQ(ierr);

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
  PetscPrintf(usr->comm,"# Test_stokesdarcy3field_mms_bulkviscosity: %s \n",&(date[0]));
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
  ierr = PetscTime(&start_time); CHKERRQ(ierr);
  ierr = PetscOptionsInsert(PETSC_NULL,&argc,&argv,NULL); CHKERRQ(ierr);
  ierr = InputParameters(&usr); CHKERRQ(ierr);
  ierr = InputPrintData(usr); CHKERRQ(ierr);

  // Numerical solution using the FD pde object
  ierr = StokesDarcy2Field_Numerical(usr); CHKERRQ(ierr);
  ierr = StokesDarcy3Field_Numerical(usr); CHKERRQ(ierr);

  // Destroy objects
  ierr = PetscBagDestroy(&usr->bag); CHKERRQ(ierr);
  ierr = PetscFree(usr);             CHKERRQ(ierr);

  // End time
  ierr = PetscTime(&end_time); CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"# Total runtime: %g (sec) \n", end_time - start_time);
  PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");
  ierr = PetscFinalize();
  return ierr;
}