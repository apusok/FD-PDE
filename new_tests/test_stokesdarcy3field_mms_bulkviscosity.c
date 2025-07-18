// ---------------------------------------
// MMS test for 3-Field and 2-Field
// run: ./test_stokesdarcy3field_mms_bulkviscosity -pc_type lu -pc_factor_mat_solver_type umfpack -pc_factor_mat_ordering_type external -nx 10 -nz 10
// python test: ./python/test_stokesdarcy3field_mms_bulkviscosity.py
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

#include "../new_src/fdpde_stokesdarcy2field.h"
#include "../new_src/fdpde_stokesdarcy3field.h"

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
  PetscFunctionBeginUser;

  // Element count
  nx = usr->par->nx;
  nz = usr->par->nz;
  xmin = usr->par->xmin;
  zmin = usr->par->zmin;
  xmax = usr->par->xmin+usr->par->L;
  zmax = usr->par->zmin+usr->par->H;

  // Create the FD-pde object
  PetscCall(FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_STOKESDARCY2FIELD,&fd));
  PetscCall(FDPDESetUp(fd));
  PetscCall(FDPDEGetDM(fd,&dm)); 

  // Set BC and coefficient evaluation functions
  PetscCall(FDPDESetFunctionBCList(fd,FormBCList2Field,bc_description,usr)); 
  PetscCall(FDPDESetFunctionCoefficient(fd,FormCoefficient2Field,coeff_description2,usr)); 
  PetscCall(FDPDEView(fd)); 

  // FD SNES Solver
  PetscCall(FDPDESolve(fd,NULL));
  PetscCall(FDPDEGetSolution(fd,&x)); 

  // Create manufactured solution and compute norms
  PetscCall(ComputeManufacturedSolution2Field(dm,&xMMS,usr)); 
  PetscCall(ComputeErrorNorms2Field(dm,x,xMMS));

  // Output extra parameters - phi, K, zeta
  PetscCall(ComputeExtraParameters(dm,usr)); 

  // Output solution to file
  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s_sd2field_sol",usr->par->fdir_out,usr->par->fname_out));
  PetscCall(DMStagViewBinaryPython(dm,x,fout));

  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s_sd2field_mms",usr->par->fdir_out,usr->par->fname_out));
  PetscCall(DMStagViewBinaryPython(dm,xMMS,fout));

  // Destroy objects
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&xMMS));
  PetscCall(DMDestroy(&dm));
  PetscCall(FDPDEDestroy(&fd));

  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscScalar    delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat;
  PetscScalar    ***c;
  PetscFunctionBeginUser;

  delta = usr->par->delta;
  phi0 = usr->par->phi0;
  phia = usr->par->phia;
  phi_min = usr->par->phi_min;
  vzeta = usr->par->vzeta;
  p_s = usr->par->p_s;
  psi_s = usr->par->psi_s;
  U_s = usr->par->U_s;
  m = usr->par->m;
  n = usr->par->n;
  k_hat = usr->par->k_hat;

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
      PetscInt idx;

      { // A = delta^2*eta (center, c=1)
        PetscScalar   eta;
        DMStagStencil point;
        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 1;
        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
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
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
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
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] = rhs[ii];
        }
      }

      { // C =  f2p (manufactured) (center, c=0)
        DMStagStencil point;
        PetscScalar   xp, zp, rhs;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 0;
        xp = coordx[i][icenter]; zp = coordz[j][icenter];
        rhs = get_f2p(xp,zp,delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);

        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
        c[j][i][idx] = rhs;
      }

      { // D1 = delta^2*xi (center, c=2)
        PetscScalar   xp, zp, xi;
        DMStagStencil point;
        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 2;
        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
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
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
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
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] = 0.0;
        }

        for (ii = 2; ii < 4; ii++) {
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] = -k_hat*get_K(xp[ii],zp[ii],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
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
#undef __FUNCT__
#define __FUNCT__ "FormBCList2Field"
PetscErrorCode FormBCList2Field(DM dm, Vec x, DMStagBCList bclist, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       k,n_bc,*idx_bc;
  PetscScalar    *value_bc,*x_bc;
  PetscScalar    delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat;
  BCType         *type_bc;
  PetscFunctionBeginUser;

  delta = usr->par->delta;
  phi0 = usr->par->phi0;
  phia = usr->par->phia;
  phi_min = usr->par->phi_min;
  vzeta = usr->par->vzeta;
  p_s = usr->par->p_s;
  psi_s = usr->par->psi_s;
  U_s = usr->par->U_s;
  m = usr->par->m;
  n = usr->par->n;
  k_hat = usr->par->k_hat;

  // LEFT Boundary - Vx
  PetscCall(DMStagBCListGetValues(bclist,'w','-',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_ux(x_bc[2*k],x_bc[2*k+1],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));

  // LEFT Boundary - Vz
  PetscCall(DMStagBCListGetValues(bclist,'w','|',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
  for (k=1; k<n_bc-1; k++) {
    value_bc[k] = get_uz(x_bc[2*k],x_bc[2*k+1],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));

  // LEFT Boundary - P
  PetscCall(DMStagBCListGetValues(bclist,'w','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_p(x_bc[2*k],x_bc[2*k+1],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
  
  // RIGHT Boundary - Vx
  PetscCall(DMStagBCListGetValues(bclist,'e','-',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_ux(x_bc[2*k],x_bc[2*k+1],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));

  // RIGHT Boundary - Vz
  PetscCall(DMStagBCListGetValues(bclist,'e','|',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
  for (k=1; k<n_bc-1; k++) {
    value_bc[k] = get_uz(x_bc[2*k],x_bc[2*k+1],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));

  // RIGHT Boundary - P
  PetscCall(DMStagBCListGetValues(bclist,'e','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_p(x_bc[2*k],x_bc[2*k+1],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));

  // DOWN Boundary - Vx
  PetscCall(DMStagBCListGetValues(bclist,'s','-',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
  for (k=1; k<n_bc-1; k++) {
    value_bc[k] = get_ux(x_bc[2*k],x_bc[2*k+1],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));

  // DOWN Boundary - Vz
  PetscCall(DMStagBCListGetValues(bclist,'s','|',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_uz(x_bc[2*k],x_bc[2*k+1],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));

  // DOWN Boundary - P
  PetscCall(DMStagBCListGetValues(bclist,'s','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_p(x_bc[2*k],x_bc[2*k+1],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));

  // UP Boundary - Vx
  PetscCall(DMStagBCListGetValues(bclist,'n','-',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
  for (k=1; k<n_bc-1; k++) {
    value_bc[k] = get_ux(x_bc[2*k],x_bc[2*k+1],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));

  // UP Boundary - Vz
  PetscCall(DMStagBCListGetValues(bclist,'n','|',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_uz(x_bc[2*k],x_bc[2*k+1],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));

  // UP Boundary - P
  PetscCall(DMStagBCListGetValues(bclist,'n','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_p(x_bc[2*k],x_bc[2*k+1],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));

  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscScalar    delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat;
  PetscFunctionBeginUser;

  delta = usr->par->delta;
  phi0 = usr->par->phi0;
  phia = usr->par->phia;
  phi_min = usr->par->phi_min;
  vzeta = usr->par->vzeta;
  p_s = usr->par->p_s;
  psi_s = usr->par->psi_s;
  U_s = usr->par->U_s;
  m = usr->par->m;
  n = usr->par->n;
  k_hat = usr->par->k_hat;

  // Create local and global vectors for MMS solution
  PetscCall(DMCreateGlobalVector(dm,&xMMS     )); 
  PetscCall(DMCreateLocalVector (dm,&xMMSlocal)); 
  PetscCall(DMStagVecGetArray(dm,xMMSlocal,&xxMMS)); 

  // Get domain corners
  PetscCall(DMStagGetGlobalSizes(dm, &Nx, &Nz,NULL));
  PetscCall(DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 
  
// Get dm coordinates array
  PetscCall(DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter)); 
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm,LEFT,&iprev));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm,RIGHT,&inext)); 

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {

      // pressure
      PetscCall(DMStagGetLocationSlot(dm,ELEMENT,0,&idx)); 
      xxMMS[j][i][idx] = get_p(coordx[i][icenter],coordz[j][icenter],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);

      // ux
      PetscCall(DMStagGetLocationSlot(dm,LEFT,0,&idx)); 
      xxMMS[j][i][idx] = get_ux(coordx[i][iprev],coordz[j][icenter],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);

      if (i == Nx-1) {
        PetscCall(DMStagGetLocationSlot(dm,RIGHT,0,&idx)); 
        xxMMS[j][i][idx] = get_ux(coordx[i][inext],coordz[j][icenter],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
      }
      
      // uz
      PetscCall(DMStagGetLocationSlot(dm,DOWN,0,&idx)); 
      xxMMS[j][i][idx] = get_uz(coordx[i][icenter],coordz[j][iprev],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);

      if (j == Nz-1) {
        PetscCall(DMStagGetLocationSlot(dm,UP,0,&idx)); 
        xxMMS[j][i][idx] = get_uz(coordx[i][icenter],coordz[j][inext],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
      }
    }
  }

  // Restore arrays
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));

  // Restore and map local to global
  PetscCall(DMStagVecRestoreArray(dm,xMMSlocal,&xxMMS)); 
  PetscCall(DMLocalToGlobalBegin(dm,xMMSlocal,INSERT_VALUES,xMMS)); 
  PetscCall(DMLocalToGlobalEnd  (dm,xMMSlocal,INSERT_VALUES,xMMS)); 
  PetscCall(VecDestroy(&xMMSlocal)); 

  // Assign pointers
  *_xMMS  = xMMS;
  
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscFunctionBeginUser;

  comm = PETSC_COMM_WORLD;

  // Get domain corners
  PetscCall(DMStagGetGlobalSizes(dm, &Nx, &Nz,NULL));
  PetscCall(DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 

  // Map global vectors to local domain
  PetscCall(DMGetLocalVector(dm, &xlocal)); 
  PetscCall(DMGlobalToLocal (dm, x, INSERT_VALUES, xlocal)); 

  PetscCall(DMGetLocalVector(dm, &xalocal)); 
  PetscCall(DMGlobalToLocal (dm, xMMS, INSERT_VALUES, xalocal)); 

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
      PetscCall(DMStagVecGetValuesStencil(dm, xlocal, 5, point, xx)); 

      // Get analytical solution
      PetscCall(DMStagVecGetValuesStencil(dm, xalocal, 5, point, xa)); 

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
  PetscCall(MPI_Allreduce(&sum_err, &gsum_err, 3, MPI_DOUBLE, MPI_SUM, comm)); 
  PetscCall(MPI_Allreduce(&sum_mms, &gsum_mms, 3, MPI_DOUBLE, MPI_SUM, comm)); 

  // L2 error norm = sqrt(gsum_err/gsum_mms)
  nrm2vx = PetscSqrtScalar(gsum_err[0]/gsum_mms[0]);
  nrm2vz = PetscSqrtScalar(gsum_err[1]/gsum_mms[1]);
  nrm2p  = PetscSqrtScalar(gsum_err[2]/gsum_mms[2]);
  nrm2v  = PetscSqrtScalar((gsum_err[0]+gsum_err[1])/(gsum_mms[0]+gsum_mms[1]));

  // Restore arrays and vectors
  PetscCall(DMRestoreLocalVector(dm, &xlocal )); 
  PetscCall(DMRestoreLocalVector(dm, &xalocal )); 

  // Print information
  PetscPrintf(comm,"# --------------------------------------- #\n");
  PetscPrintf(comm,"# NORMS 2-FIELD: \n");
  PetscPrintf(comm,"# Velocity: norm2 = %1.12e norm2x = %1.12e norm2z = %1.12e \n",nrm2v,nrm2vx,nrm2vz);
  PetscPrintf(comm,"# Pressure: norm2 = %1.12e\n",nrm2p);
  PetscPrintf(comm,"# Grid info: hx = %1.12e hz = %1.12e\n",dx,dz);

  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscFunctionBeginUser;

  // Element count
  nx = usr->par->nx;
  nz = usr->par->nz;
  xmin = usr->par->xmin;
  zmin = usr->par->zmin;
  xmax = usr->par->xmin+usr->par->L;
  zmax = usr->par->zmin+usr->par->H;

  // Create the FD-pde object
  PetscCall(FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_STOKESDARCY3FIELD,&fd));
  PetscCall(FDPDESetUp(fd));
  PetscCall(FDPDEGetDM(fd,&dm)); 

  // Set BC and coefficient evaluation functions
  PetscCall(FDPDESetFunctionBCList(fd,FormBCList3Field,bc_description,usr)); 
  PetscCall(FDPDESetFunctionCoefficient(fd,FormCoefficient3Field,coeff_description3,usr)); 
  PetscCall(FDPDEView(fd)); 

  // FD SNES Solver
  PetscCall(FDPDESolve(fd,NULL));
  PetscCall(FDPDEGetSolution(fd,&x)); 

  // Create manufactured solution and compute norms
  PetscCall(ComputeManufacturedSolution3Field(dm,&xMMS,usr)); 
  PetscCall(ComputeErrorNorms3Field(dm,x,xMMS));

  // Output solution to file
  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s_sd3field_sol",usr->par->fdir_out,usr->par->fname_out));
  PetscCall(DMStagViewBinaryPython(dm,x,fout));

  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s_sd3field_mms",usr->par->fdir_out,usr->par->fname_out));
  PetscCall(DMStagViewBinaryPython(dm,xMMS,fout));

  // Destroy objects
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&xMMS));
  PetscCall(DMDestroy(&dm));
  PetscCall(FDPDEDestroy(&fd));

  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscScalar    delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat;
  PetscScalar    ***c;
  PetscFunctionBeginUser;

  delta = usr->par->delta;
  phi0 = usr->par->phi0;
  phia = usr->par->phia;
  phi_min = usr->par->phi_min;
  vzeta = usr->par->vzeta;
  p_s = usr->par->p_s;
  psi_s = usr->par->psi_s;
  U_s = usr->par->U_s;
  m = usr->par->m;
  n = usr->par->n;
  k_hat = usr->par->k_hat;

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
      PetscInt idx;

      { // A = delta^2*eta (center, c=1)
        PetscScalar   eta;
        DMStagStencil point;
        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = SD3_COEFF_ELEMENT_A;
        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
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
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
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
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] = rhs[ii];
        }
      }

      { // C =  f3p (manufactured) (center, c=0)
        DMStagStencil point;
        PetscScalar   xp, zp, rhs;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = SD3_COEFF_ELEMENT_C;
        xp = coordx[i][icenter]; zp = coordz[j][icenter];
        rhs = get_f3p(xp,zp,delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);

        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
        c[j][i][idx] = rhs;
      }

      { // D1 = -1/(delta^2*xi) (center, c=2)
        PetscScalar   xp, zp, xi;
        DMStagStencil point;
        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = SD3_COEFF_ELEMENT_D1;
        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
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
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
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
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] = 0.0;
        }

        for (ii = 2; ii < 4; ii++) {
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
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
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] = -get_K(xp[ii],zp[ii],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
        }
      }

      { // DC =  f3pc (manufactured) (center, c=3)
        DMStagStencil point;
        PetscScalar   xp, zp, rhs;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = SD3_COEFF_ELEMENT_DC;
        xp = coordx[i][icenter]; zp = coordz[j][icenter];
        rhs = get_f3pc(xp,zp,delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);

        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
        c[j][i][idx] = rhs;
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
#undef __FUNCT__
#define __FUNCT__ "FormBCList3Field"
PetscErrorCode FormBCList3Field(DM dm, Vec x, DMStagBCList bclist, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       k,n_bc,*idx_bc;
  PetscScalar    *value_bc,*x_bc;
  PetscScalar    delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat;
  BCType         *type_bc;
  PetscFunctionBeginUser;

  delta = usr->par->delta;
  phi0 = usr->par->phi0;
  phia = usr->par->phia;
  phi_min = usr->par->phi_min;
  vzeta = usr->par->vzeta;
  p_s = usr->par->p_s;
  psi_s = usr->par->psi_s;
  U_s = usr->par->U_s;
  m = usr->par->m;
  n = usr->par->n;
  k_hat = usr->par->k_hat;

  // LEFT Boundary - Vx
  PetscCall(DMStagBCListGetValues(bclist,'w','-',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_ux(x_bc[2*k],x_bc[2*k+1],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));

  // LEFT Boundary - Vz
  PetscCall(DMStagBCListGetValues(bclist,'w','|',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
  for (k=1; k<n_bc-1; k++) {
    value_bc[k] = get_uz(x_bc[2*k],x_bc[2*k+1],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));

  // LEFT Boundary - P
  PetscCall(DMStagBCListGetValues(bclist,'w','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_p(x_bc[2*k],x_bc[2*k+1],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
  
  // RIGHT Boundary - Vx
  PetscCall(DMStagBCListGetValues(bclist,'e','-',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_ux(x_bc[2*k],x_bc[2*k+1],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));

  // RIGHT Boundary - Vz
  PetscCall(DMStagBCListGetValues(bclist,'e','|',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
  for (k=1; k<n_bc-1; k++) {
    value_bc[k] = get_uz(x_bc[2*k],x_bc[2*k+1],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));

  // RIGHT Boundary - P
  PetscCall(DMStagBCListGetValues(bclist,'e','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_p(x_bc[2*k],x_bc[2*k+1],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));

  // DOWN Boundary - Vx
  PetscCall(DMStagBCListGetValues(bclist,'s','-',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
  for (k=1; k<n_bc-1; k++) {
    value_bc[k] = get_ux(x_bc[2*k],x_bc[2*k+1],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));

  // DOWN Boundary - Vz
  PetscCall(DMStagBCListGetValues(bclist,'s','|',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_uz(x_bc[2*k],x_bc[2*k+1],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));

  // DOWN Boundary - P
  PetscCall(DMStagBCListGetValues(bclist,'s','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_p(x_bc[2*k],x_bc[2*k+1],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));

  // UP Boundary - Vx
  PetscCall(DMStagBCListGetValues(bclist,'n','-',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
  for (k=1; k<n_bc-1; k++) {
    value_bc[k] = get_ux(x_bc[2*k],x_bc[2*k+1],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));

  // UP Boundary - Vz
  PetscCall(DMStagBCListGetValues(bclist,'n','|',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_uz(x_bc[2*k],x_bc[2*k+1],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));

  // UP Boundary - P
  PetscCall(DMStagBCListGetValues(bclist,'n','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_p(x_bc[2*k],x_bc[2*k+1],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));

  // Compaction pressure
  // Left
  PetscCall(DMStagBCListGetValues(bclist,'w','o',1,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_pc(x_bc[2*k],x_bc[2*k+1],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',1,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
  
  // RIGHT
  PetscCall(DMStagBCListGetValues(bclist,'e','o',1,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_pc(x_bc[2*k],x_bc[2*k+1],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',1,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));

  // DOWN
  PetscCall(DMStagBCListGetValues(bclist,'s','o',1,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_pc(x_bc[2*k],x_bc[2*k+1],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',1,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));

  // UP
  PetscCall(DMStagBCListGetValues(bclist,'n','o',1,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_pc(x_bc[2*k],x_bc[2*k+1],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
    type_bc[k] = BC_DIRICHLET_STAG;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'o',1,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));

  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscScalar    delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat;
  PetscFunctionBeginUser;

  delta = usr->par->delta;
  phi0 = usr->par->phi0;
  phia = usr->par->phia;
  phi_min = usr->par->phi_min;
  vzeta = usr->par->vzeta;
  p_s = usr->par->p_s;
  psi_s = usr->par->psi_s;
  U_s = usr->par->U_s;
  m = usr->par->m;
  n = usr->par->n;
  k_hat = usr->par->k_hat;

  // Create local and global vectors for MMS solution
  PetscCall(DMCreateGlobalVector(dm,&xMMS     )); 
  PetscCall(DMCreateLocalVector (dm,&xMMSlocal)); 
  PetscCall(DMStagVecGetArray(dm,xMMSlocal,&xxMMS)); 

  // Get domain corners
  PetscCall(DMStagGetGlobalSizes(dm, &Nx, &Nz,NULL));
  PetscCall(DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 
  
// Get dm coordinates array
  PetscCall(DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter)); 
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm,LEFT,&iprev));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm,RIGHT,&inext)); 

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {

      // pressure
      PetscCall(DMStagGetLocationSlot(dm,ELEMENT,0,&idx)); 
      xxMMS[j][i][idx] = get_p(coordx[i][icenter],coordz[j][icenter],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);

      // compaction pressure
      PetscCall(DMStagGetLocationSlot(dm,ELEMENT,1,&idx)); 
      xxMMS[j][i][idx] = get_pc(coordx[i][icenter],coordz[j][icenter],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);

      // ux
      PetscCall(DMStagGetLocationSlot(dm,LEFT,0,&idx)); 
      xxMMS[j][i][idx] = get_ux(coordx[i][iprev],coordz[j][icenter],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);

      if (i == Nx-1) {
        PetscCall(DMStagGetLocationSlot(dm,RIGHT,0,&idx)); 
        xxMMS[j][i][idx] = get_ux(coordx[i][inext],coordz[j][icenter],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
      }
      
      // uz
      PetscCall(DMStagGetLocationSlot(dm,DOWN,0,&idx)); 
      xxMMS[j][i][idx] = get_uz(coordx[i][icenter],coordz[j][iprev],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);

      if (j == Nz-1) {
        PetscCall(DMStagGetLocationSlot(dm,UP,0,&idx)); 
        xxMMS[j][i][idx] = get_uz(coordx[i][icenter],coordz[j][inext],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
      }
    }
  }

  // Restore arrays
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));

  // Restore and map local to global
  PetscCall(DMStagVecRestoreArray(dm,xMMSlocal,&xxMMS)); 
  PetscCall(DMLocalToGlobalBegin(dm,xMMSlocal,INSERT_VALUES,xMMS)); 
  PetscCall(DMLocalToGlobalEnd  (dm,xMMSlocal,INSERT_VALUES,xMMS)); 
  PetscCall(VecDestroy(&xMMSlocal)); 

  // Assign pointers
  *_xMMS  = xMMS;
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
PetscErrorCode ComputeErrorNorms3Field(DM dm,Vec x,Vec xMMS)
{
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz;
  PetscScalar    xx[6], xa[6], dx, dz, dv;
  PetscScalar    nrm2p, nrm2pc, nrm2v, nrm2vx, nrm2vz, sum_err[4], gsum_err[4], sum_mms[4], gsum_mms[4];
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
  PetscCall(DMGlobalToLocal (dm, xMMS, INSERT_VALUES, xalocal)); 

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
      PetscCall(DMStagVecGetValuesStencil(dm, xlocal, 6, point, xx)); 

      // Get analytical solution
      PetscCall(DMStagVecGetValuesStencil(dm, xalocal, 6, point, xa)); 

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
  PetscCall(MPI_Allreduce(&sum_err, &gsum_err, 4, MPI_DOUBLE, MPI_SUM, comm)); 
  PetscCall(MPI_Allreduce(&sum_mms, &gsum_mms, 4, MPI_DOUBLE, MPI_SUM, comm)); 

  // L2 error norm = sqrt(gsum_err/gsum_mms)
  nrm2vx = PetscSqrtScalar(gsum_err[0]/gsum_mms[0]);
  nrm2vz = PetscSqrtScalar(gsum_err[1]/gsum_mms[1]);
  nrm2p  = PetscSqrtScalar(gsum_err[2]/gsum_mms[2]);
  nrm2pc = PetscSqrtScalar(gsum_err[3]/gsum_mms[3]);
  nrm2v  = PetscSqrtScalar((gsum_err[0]+gsum_err[1])/(gsum_mms[0]+gsum_mms[1]));

  // Restore arrays and vectors
  PetscCall(DMRestoreLocalVector(dm, &xlocal )); 
  PetscCall(DMRestoreLocalVector(dm, &xalocal )); 

  // Print information
  PetscPrintf(comm,"# --------------------------------------- #\n");
  PetscPrintf(comm,"# NORMS 3-FIELD: \n");
  PetscPrintf(comm,"# Velocity: norm2 = %1.12e norm2x = %1.12e norm2z = %1.12e \n",nrm2v,nrm2vx,nrm2vz);
  PetscPrintf(comm,"# Pressure: norm2 = %1.12e\n",nrm2p);
  PetscPrintf(comm,"# Compaction Pressure: norm2 = %1.12e\n",nrm2pc);
  PetscPrintf(comm,"# Grid info: hx = %1.12e hz = %1.12e\n",dx,dz);

  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscScalar    delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat;
  char           fout[FNAME_LENGTH];
  PetscFunctionBeginUser;

  delta = usr->par->delta;
  phi0 = usr->par->phi0;
  phia = usr->par->phia;
  phi_min = usr->par->phi_min;
  vzeta = usr->par->vzeta;
  p_s = usr->par->p_s;
  psi_s = usr->par->psi_s;
  U_s = usr->par->U_s;
  m = usr->par->m;
  n = usr->par->n;
  k_hat = usr->par->k_hat;

  // Create new dmextra 
  PetscCall(DMStagCreateCompatibleDMStag(dm,0,0,5,0,&dmextra)); 
  PetscCall(DMSetUp(dmextra)); 
  PetscCall(DMStagSetUniformCoordinatesProduct(dmextra,usr->par->xmin,usr->par->xmin+usr->par->L,usr->par->zmin,usr->par->zmin+usr->par->H,0.0,0.0));

  // Create local and global vectors
  PetscCall(DMCreateGlobalVector(dmextra,&x)); 
  PetscCall(DMCreateLocalVector (dmextra,&xlocal)); 
  PetscCall(DMStagVecGetArray(dmextra,xlocal,&xx)); 

  // Get domain corners
  PetscCall(DMStagGetGlobalSizes(dmextra, &Nx, &Nz,NULL));
  PetscCall(DMStagGetCorners(dmextra, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 
  
// Get dm coordinates array
  PetscCall(DMStagGetProductCoordinateArraysRead(dmextra,&coordx,&coordz,NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmextra,ELEMENT,&icenter)); 
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmextra,LEFT,&iprev));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmextra,RIGHT,&inext)); 

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {

      // phi - element 0
      PetscCall(DMStagGetLocationSlot(dmextra,ELEMENT,0,&idx)); 
      xx[j][i][idx] = get_phi(coordx[i][icenter],coordz[j][icenter],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);

      // K - element 1
      PetscCall(DMStagGetLocationSlot(dmextra,ELEMENT,1,&idx)); 
      xx[j][i][idx] = get_K(coordx[i][icenter],coordz[j][icenter],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);

      // zeta - element 2
      PetscCall(DMStagGetLocationSlot(dmextra,ELEMENT,2,&idx)); 
      xx[j][i][idx] = get_zeta(coordx[i][icenter],coordz[j][icenter],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);

      // eta - element 3
      PetscCall(DMStagGetLocationSlot(dmextra,ELEMENT,3,&idx)); 
      xx[j][i][idx] = get_eta(coordx[i][icenter],coordz[j][icenter],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);

      // xi - element 4
      PetscCall(DMStagGetLocationSlot(dmextra,ELEMENT,4,&idx)); 
      xx[j][i][idx] = get_xi(coordx[i][icenter],coordz[j][icenter],delta,phi0,phia,phi_min,vzeta,p_s,psi_s,U_s,m,n,k_hat);
    }
  }

  // Restore arrays
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dmextra,&coordx,&coordz,NULL));

  // Restore and map local to global
  PetscCall(DMStagVecRestoreArray(dmextra,xlocal,&xx)); 
  PetscCall(DMLocalToGlobalBegin(dmextra,xlocal,INSERT_VALUES,x)); 
  PetscCall(DMLocalToGlobalEnd  (dmextra,xlocal,INSERT_VALUES,x)); 

  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s_extra_parameters",usr->par->fdir_out,usr->par->fname_out));
  PetscCall(DMStagViewBinaryPython(dmextra,x,fout));

  // clean 
  PetscCall(VecDestroy(&xlocal)); 
  PetscCall(VecDestroy(&x));
  PetscCall(DMDestroy(&dmextra));
  
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
  PetscCall(PetscBagRegisterInt(bag, &par->nx, 4, "nx", "Element count in the x-dir")); 
  PetscCall(PetscBagRegisterInt(bag, &par->nz, 5, "nz", "Element count in the z-dir")); 

  PetscCall(PetscBagRegisterScalar(bag, &par->xmin, 0.0, "xmin", "Start coordinate of domain in x-dir")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->zmin, 0.0, "zmin", "Start coordinate of domain in z-dir")); 

  PetscCall(PetscBagRegisterScalar(bag, &par->L, 1.0, "L", "Length of domain in x-dir")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->H, 1.0, "H", "Height of domain in z-dir")); 

  // Physical and material parameters
  PetscCall(PetscBagRegisterScalar(bag, &par->k_hat, -1.0, "k_hat", "Direction of unit vertical vector")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->delta, 1.0, "delta", "Dimensionless compaction length ~1-10")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->phi0, 0.1, "phi0", "Reference porosity")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->phia, 0.1, "phia", "Amplitude porosity perturbation")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->phi_min, 1.0e-6, "phi_min", "Cutoff porosity")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->vzeta, 10.0, "vzeta", "Reference bulk to shear viscosity ratio ~1e0-1e4")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->eta, 1.0, "eta", "Non-dimensional shear viscosity")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->p_s, 1.0, "p_s", "Pressure amplitude")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->psi_s, 1.0, "psi_s", "Vector potential function amplitude")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->U_s, 1.0, "U_s", "Scalar potential function amplitude")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->m, 2.0, "m", "Trigonometric coefficient")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->n, 3.0, "n", "Porosity exponent")); 

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
  PetscPrintf(usr->comm,"# Test_stokesdarcy3field_mms_bulkviscosity: %s \n",&(date[0]));
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
  PetscCall(PetscTime(&start_time)); 
  PetscCall(PetscOptionsInsert(PETSC_NULLPTR,&argc,&argv,NULL)); 
  PetscCall(InputParameters(&usr)); 
  PetscCall(InputPrintData(usr)); 

  // Numerical solution using the FD pde object
  PetscCall(StokesDarcy2Field_Numerical(usr));
  PetscCall(StokesDarcy3Field_Numerical(usr)); 

  // Destroy objects
  PetscCall(PetscBagDestroy(&usr->bag)); 
  PetscCall(PetscFree(usr));

  // End time
  PetscCall(PetscTime(&end_time)); 
  PetscPrintf(PETSC_COMM_WORLD,"# Total runtime: %g (sec) \n", end_time - start_time);
  PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");
  PetscCall(PetscFinalize());
  return 0;
}