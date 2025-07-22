// ---------------------------------------
// MMS test to verify a power-law effective viscosity approach, where eta=eta0*(epsII/eps0)^(1/np-1), zeta = eta/phi
// run: ./test_effvisc_mms_ -pc_type lu -pc_factor_mat_solver_type umfpack -pc_factor_mat_ordering_type external -snes_monitor -ksp_monitor -nx 20 -nz 20 -log_view
// python test: ./python/test_effvisc_mms.py
// python sympy: ./mms/mms_effvisc_powerlaw.py
// ---------------------------------------
static char help[] = "Application to verify a power-law effective viscosity for Stokes and StokesDarcy using MMS \n\n";

// System of equations (nondimensional with u0 = drho*g*h^2/eta0) - also ignore body forces (k_hat=0)
// Stokes:
// -grad(P)+div(eta symgrad(v)) + rho*k_hat = 0
// div(v) = 0 
// StokesDarcy:
// -grad(P)+div(eta symgrad(v)) + grad((zeta-2/3eta)div(v))+phi*k_hat = 0
// div(v) - R^2 div(Kphi(grad(P)-k_hat)) = 0, where R=delta/h, delta = sqrt(K0*eta0/mu) 

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

#include "../src/fdpde_stokesdarcy2field.h"

// ---------------------------------------
// Application Context
// ---------------------------------------
#define FNAME_LENGTH 200

// parameters (bag)
typedef struct {
  PetscInt       test;
  PetscInt       nx, nz;
  PetscScalar    L, H;
  PetscScalar    xmin, zmin;
  PetscScalar    eta0, eps0, np, R, etamax, etamin;
  PetscScalar    phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat;
  char           fname_out[FNAME_LENGTH]; 
  char           fdir_out[FNAME_LENGTH]; 
} Params;

// user defined and model-dependent variables
typedef struct {
  Params        *par;
  PetscBag       bag;
  MPI_Comm       comm;
  PetscMPIInt    rank;
  DM             dm, dmeps;
  Vec            xeps;
} UsrData;

// ---------------------------------------
// Function definitions
// ---------------------------------------
PetscErrorCode Numerical_solution(void*);
PetscErrorCode ComputeManufacturedSolution(DM,Vec*,Vec*,void*,PetscInt);
PetscErrorCode ComputeErrorNorms(DM,Vec,Vec,Vec,PetscInt test,void*);
PetscErrorCode InputParameters(UsrData**);
PetscErrorCode InputPrintData(UsrData*);
PetscErrorCode FormCoefficient_Stokes(FDPDE, DM, Vec, DM, Vec, void*);
PetscErrorCode FormCoefficient_StokesDarcy(FDPDE, DM, Vec, DM, Vec, void*);
PetscErrorCode FormBCList(DM, Vec, DMStagBCList, void*);
PetscErrorCode UpdateStrainRates(DM,Vec,void*);

// Manufactured solutions
static PetscScalar get_p(PetscScalar x, PetscScalar z, PetscScalar eta0, PetscScalar eps0, PetscScalar np, PetscScalar phi_0, PetscScalar phi_s, PetscScalar p_s, PetscScalar psi_s, PetscScalar U_s, PetscScalar m, PetscScalar n, PetscScalar k_hat, PetscScalar R)
{ PetscScalar result;
  result = p_s*cos(M_PI*m*x)*cos(M_PI*m*z);
  return(result);
}
static PetscScalar get_ux(PetscScalar x, PetscScalar z, PetscScalar eta0, PetscScalar eps0, PetscScalar np, PetscScalar phi_0, PetscScalar phi_s, PetscScalar p_s, PetscScalar psi_s, PetscScalar U_s, PetscScalar m, PetscScalar n, PetscScalar k_hat, PetscScalar R)
{ PetscScalar result;
  result = M_PI*U_s*m*sin(M_PI*m*x)*cos(M_PI*m*z) + M_PI*m*psi_s*(1.0 - cos(M_PI*m*x))*sin(M_PI*m*z);
  return(result);
}
static PetscScalar get_uz(PetscScalar x, PetscScalar z, PetscScalar eta0, PetscScalar eps0, PetscScalar np, PetscScalar phi_0, PetscScalar phi_s, PetscScalar p_s, PetscScalar psi_s, PetscScalar U_s, PetscScalar m, PetscScalar n, PetscScalar k_hat, PetscScalar R)
{ PetscScalar result;
  result = M_PI*U_s*m*sin(M_PI*m*z)*cos(M_PI*m*x) - M_PI*m*psi_s*(1.0 - cos(M_PI*m*z))*sin(M_PI*m*x);
  return(result);
}
static PetscScalar get_phi(PetscScalar x, PetscScalar z, PetscScalar eta0, PetscScalar eps0, PetscScalar np, PetscScalar phi_0, PetscScalar phi_s, PetscScalar p_s, PetscScalar psi_s, PetscScalar U_s, PetscScalar m, PetscScalar n, PetscScalar k_hat, PetscScalar R)
{ PetscScalar result;
  result = phi_0*(phi_s*cos(M_PI*m*x)*cos(M_PI*m*z) + 1.0);
  return(result);
}
static PetscScalar get_Kphi(PetscScalar x, PetscScalar z, PetscScalar eta0, PetscScalar eps0, PetscScalar np, PetscScalar phi_0, PetscScalar phi_s, PetscScalar p_s, PetscScalar psi_s, PetscScalar U_s, PetscScalar m, PetscScalar n, PetscScalar k_hat, PetscScalar R)
{ PetscScalar result;
  result = pow(phi_s*cos(M_PI*m*x)*cos(M_PI*m*z) + 1.0, n);
  return(result);
}
static PetscScalar get_exx(PetscScalar x, PetscScalar z, PetscScalar eta0, PetscScalar eps0, PetscScalar np, PetscScalar phi_0, PetscScalar phi_s, PetscScalar p_s, PetscScalar psi_s, PetscScalar U_s, PetscScalar m, PetscScalar n, PetscScalar k_hat, PetscScalar R)
{ PetscScalar result;
  result = pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) + pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z);
  return(result);
}
static PetscScalar get_ezz(PetscScalar x, PetscScalar z, PetscScalar eta0, PetscScalar eps0, PetscScalar np, PetscScalar phi_0, PetscScalar phi_s, PetscScalar p_s, PetscScalar psi_s, PetscScalar U_s, PetscScalar m, PetscScalar n, PetscScalar k_hat, PetscScalar R)
{ PetscScalar result;
  result = pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) - pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z);
  return(result);
}
static PetscScalar get_exz(PetscScalar x, PetscScalar z, PetscScalar eta0, PetscScalar eps0, PetscScalar np, PetscScalar phi_0, PetscScalar phi_s, PetscScalar p_s, PetscScalar psi_s, PetscScalar U_s, PetscScalar m, PetscScalar n, PetscScalar k_hat, PetscScalar R)
{ PetscScalar result;
  result = -1.0*pow(M_PI, 2)*U_s*pow(m, 2)*sin(M_PI*m*x)*sin(M_PI*m*z) + 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*x))*cos(M_PI*m*z) - 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*z))*cos(M_PI*m*x);
  return(result);
}
static PetscScalar get_fux_stokes(PetscScalar x, PetscScalar z, PetscScalar eta0, PetscScalar eps0, PetscScalar np, PetscScalar phi_0, PetscScalar phi_s, PetscScalar p_s, PetscScalar psi_s, PetscScalar U_s, PetscScalar m, PetscScalar n, PetscScalar k_hat, PetscScalar R)
{ PetscScalar result;
  result = 2.0*eta0*pow(sqrt(0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) - pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) + pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 1.0*pow(-1.0*pow(M_PI, 2)*U_s*pow(m, 2)*sin(M_PI*m*x)*sin(M_PI*m*z) + 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*x))*cos(M_PI*m*z) - 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*z))*cos(M_PI*m*x), 2))/eps0, -1 + 1.0/np)*(-1 + 1.0/np)*(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) + pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z))*(0.25*(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) - pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z))*(-2*pow(M_PI, 3)*U_s*pow(m, 3)*sin(M_PI*m*x)*cos(M_PI*m*z) - 2*pow(M_PI, 3)*pow(m, 3)*psi_s*sin(M_PI*m*z)*cos(M_PI*m*x)) + 0.25*(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) + pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z))*(-2*pow(M_PI, 3)*U_s*pow(m, 3)*sin(M_PI*m*x)*cos(M_PI*m*z) + 2*pow(M_PI, 3)*pow(m, 3)*psi_s*sin(M_PI*m*z)*cos(M_PI*m*x)) + 0.5*(-1.0*pow(M_PI, 2)*U_s*pow(m, 2)*sin(M_PI*m*x)*sin(M_PI*m*z) + 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*x))*cos(M_PI*m*z) - 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*z))*cos(M_PI*m*x))*(-2.0*pow(M_PI, 3)*U_s*pow(m, 3)*sin(M_PI*m*z)*cos(M_PI*m*x) + pow(M_PI, 3)*pow(m, 3)*psi_s*(1.0 - cos(M_PI*m*z))*sin(M_PI*m*x) + pow(M_PI, 3)*pow(m, 3)*psi_s*sin(M_PI*m*x)*cos(M_PI*m*z)))*1.0/(0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) - pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) + pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 1.0*pow(-1.0*pow(M_PI, 2)*U_s*pow(m, 2)*sin(M_PI*m*x)*sin(M_PI*m*z) + 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*x))*cos(M_PI*m*z) - 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*z))*cos(M_PI*m*x), 2)) + 2.0*eta0*pow(sqrt(0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) - pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) + pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 1.0*pow(-1.0*pow(M_PI, 2)*U_s*pow(m, 2)*sin(M_PI*m*x)*sin(M_PI*m*z) + 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*x))*cos(M_PI*m*z) - 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*z))*cos(M_PI*m*x), 2))/eps0, -1 + 1.0/np)*(-1 + 1.0/np)*(0.25*(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) - pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z))*(-2*pow(M_PI, 3)*U_s*pow(m, 3)*sin(M_PI*m*z)*cos(M_PI*m*x) - 2*pow(M_PI, 3)*pow(m, 3)*psi_s*sin(M_PI*m*x)*cos(M_PI*m*z)) + 0.25*(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) + pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z))*(-2*pow(M_PI, 3)*U_s*pow(m, 3)*sin(M_PI*m*z)*cos(M_PI*m*x) + 2*pow(M_PI, 3)*pow(m, 3)*psi_s*sin(M_PI*m*x)*cos(M_PI*m*z)) + 0.5*(-1.0*pow(M_PI, 2)*U_s*pow(m, 2)*sin(M_PI*m*x)*sin(M_PI*m*z) + 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*x))*cos(M_PI*m*z) - 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*z))*cos(M_PI*m*x))*(-2.0*pow(M_PI, 3)*U_s*pow(m, 3)*sin(M_PI*m*x)*cos(M_PI*m*z) - 1.0*pow(M_PI, 3)*pow(m, 3)*psi_s*(1.0 - cos(M_PI*m*x))*sin(M_PI*m*z) - 1.0*pow(M_PI, 3)*pow(m, 3)*psi_s*sin(M_PI*m*z)*cos(M_PI*m*x)))*(-1.0*pow(M_PI, 2)*U_s*pow(m, 2)*sin(M_PI*m*x)*sin(M_PI*m*z) + 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*x))*cos(M_PI*m*z) - 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*z))*cos(M_PI*m*x))*1.0/(0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) - pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) + pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 1.0*pow(-1.0*pow(M_PI, 2)*U_s*pow(m, 2)*sin(M_PI*m*x)*sin(M_PI*m*z) + 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*x))*cos(M_PI*m*z) - 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*z))*cos(M_PI*m*x), 2)) + 2.0*eta0*pow(sqrt(0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) - pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) + pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 1.0*pow(-1.0*pow(M_PI, 2)*U_s*pow(m, 2)*sin(M_PI*m*x)*sin(M_PI*m*z) + 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*x))*cos(M_PI*m*z) - 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*z))*cos(M_PI*m*x), 2))/eps0, -1 + 1.0/np)*(-pow(M_PI, 3)*U_s*pow(m, 3)*sin(M_PI*m*x)*cos(M_PI*m*z) + pow(M_PI, 3)*pow(m, 3)*psi_s*sin(M_PI*m*z)*cos(M_PI*m*x)) + 2.0*eta0*pow(sqrt(0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) - pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) + pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 1.0*pow(-1.0*pow(M_PI, 2)*U_s*pow(m, 2)*sin(M_PI*m*x)*sin(M_PI*m*z) + 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*x))*cos(M_PI*m*z) - 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*z))*cos(M_PI*m*x), 2))/eps0, -1 + 1.0/np)*(-1.0*pow(M_PI, 3)*U_s*pow(m, 3)*sin(M_PI*m*x)*cos(M_PI*m*z) - 0.5*pow(M_PI, 3)*pow(m, 3)*psi_s*(1.0 - cos(M_PI*m*x))*sin(M_PI*m*z) - 0.5*pow(M_PI, 3)*pow(m, 3)*psi_s*sin(M_PI*m*z)*cos(M_PI*m*x)) + M_PI*m*p_s*sin(M_PI*m*x)*cos(M_PI*m*z);
  return(result);
}
static PetscScalar get_fuz_stokes(PetscScalar x, PetscScalar z, PetscScalar eta0, PetscScalar eps0, PetscScalar np, PetscScalar phi_0, PetscScalar phi_s, PetscScalar p_s, PetscScalar psi_s, PetscScalar U_s, PetscScalar m, PetscScalar n, PetscScalar k_hat, PetscScalar R)
{ PetscScalar result;
  result = 2.0*eta0*pow(sqrt(0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) - pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) + pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 1.0*pow(-1.0*pow(M_PI, 2)*U_s*pow(m, 2)*sin(M_PI*m*x)*sin(M_PI*m*z) + 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*x))*cos(M_PI*m*z) - 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*z))*cos(M_PI*m*x), 2))/eps0, -1 + 1.0/np)*(-1 + 1.0/np)*(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) - pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z))*(0.25*(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) - pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z))*(-2*pow(M_PI, 3)*U_s*pow(m, 3)*sin(M_PI*m*z)*cos(M_PI*m*x) - 2*pow(M_PI, 3)*pow(m, 3)*psi_s*sin(M_PI*m*x)*cos(M_PI*m*z)) + 0.25*(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) + pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z))*(-2*pow(M_PI, 3)*U_s*pow(m, 3)*sin(M_PI*m*z)*cos(M_PI*m*x) + 2*pow(M_PI, 3)*pow(m, 3)*psi_s*sin(M_PI*m*x)*cos(M_PI*m*z)) + 0.5*(-1.0*pow(M_PI, 2)*U_s*pow(m, 2)*sin(M_PI*m*x)*sin(M_PI*m*z) + 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*x))*cos(M_PI*m*z) - 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*z))*cos(M_PI*m*x))*(-2.0*pow(M_PI, 3)*U_s*pow(m, 3)*sin(M_PI*m*x)*cos(M_PI*m*z) - 1.0*pow(M_PI, 3)*pow(m, 3)*psi_s*(1.0 - cos(M_PI*m*x))*sin(M_PI*m*z) - 1.0*pow(M_PI, 3)*pow(m, 3)*psi_s*sin(M_PI*m*z)*cos(M_PI*m*x)))*1.0/(0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) - pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) + pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 1.0*pow(-1.0*pow(M_PI, 2)*U_s*pow(m, 2)*sin(M_PI*m*x)*sin(M_PI*m*z) + 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*x))*cos(M_PI*m*z) - 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*z))*cos(M_PI*m*x), 2)) + 2.0*eta0*pow(sqrt(0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) - pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) + pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 1.0*pow(-1.0*pow(M_PI, 2)*U_s*pow(m, 2)*sin(M_PI*m*x)*sin(M_PI*m*z) + 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*x))*cos(M_PI*m*z) - 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*z))*cos(M_PI*m*x), 2))/eps0, -1 + 1.0/np)*(-1 + 1.0/np)*(0.25*(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) - pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z))*(-2*pow(M_PI, 3)*U_s*pow(m, 3)*sin(M_PI*m*x)*cos(M_PI*m*z) - 2*pow(M_PI, 3)*pow(m, 3)*psi_s*sin(M_PI*m*z)*cos(M_PI*m*x)) + 0.25*(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) + pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z))*(-2*pow(M_PI, 3)*U_s*pow(m, 3)*sin(M_PI*m*x)*cos(M_PI*m*z) + 2*pow(M_PI, 3)*pow(m, 3)*psi_s*sin(M_PI*m*z)*cos(M_PI*m*x)) + 0.5*(-1.0*pow(M_PI, 2)*U_s*pow(m, 2)*sin(M_PI*m*x)*sin(M_PI*m*z) + 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*x))*cos(M_PI*m*z) - 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*z))*cos(M_PI*m*x))*(-2.0*pow(M_PI, 3)*U_s*pow(m, 3)*sin(M_PI*m*z)*cos(M_PI*m*x) + pow(M_PI, 3)*pow(m, 3)*psi_s*(1.0 - cos(M_PI*m*z))*sin(M_PI*m*x) + pow(M_PI, 3)*pow(m, 3)*psi_s*sin(M_PI*m*x)*cos(M_PI*m*z)))*(-1.0*pow(M_PI, 2)*U_s*pow(m, 2)*sin(M_PI*m*x)*sin(M_PI*m*z) + 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*x))*cos(M_PI*m*z) - 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*z))*cos(M_PI*m*x))*1.0/(0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) - pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) + pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 1.0*pow(-1.0*pow(M_PI, 2)*U_s*pow(m, 2)*sin(M_PI*m*x)*sin(M_PI*m*z) + 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*x))*cos(M_PI*m*z) - 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*z))*cos(M_PI*m*x), 2)) + 2.0*eta0*pow(sqrt(0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) - pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) + pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 1.0*pow(-1.0*pow(M_PI, 2)*U_s*pow(m, 2)*sin(M_PI*m*x)*sin(M_PI*m*z) + 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*x))*cos(M_PI*m*z) - 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*z))*cos(M_PI*m*x), 2))/eps0, -1 + 1.0/np)*(-pow(M_PI, 3)*U_s*pow(m, 3)*sin(M_PI*m*z)*cos(M_PI*m*x) - pow(M_PI, 3)*pow(m, 3)*psi_s*sin(M_PI*m*x)*cos(M_PI*m*z)) + 2.0*eta0*pow(sqrt(0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) - pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) + pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 1.0*pow(-1.0*pow(M_PI, 2)*U_s*pow(m, 2)*sin(M_PI*m*x)*sin(M_PI*m*z) + 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*x))*cos(M_PI*m*z) - 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*z))*cos(M_PI*m*x), 2))/eps0, -1 + 1.0/np)*(-1.0*pow(M_PI, 3)*U_s*pow(m, 3)*sin(M_PI*m*z)*cos(M_PI*m*x) + 0.5*pow(M_PI, 3)*pow(m, 3)*psi_s*(1.0 - cos(M_PI*m*z))*sin(M_PI*m*x) + 0.5*pow(M_PI, 3)*pow(m, 3)*psi_s*sin(M_PI*m*x)*cos(M_PI*m*z)) + M_PI*m*p_s*sin(M_PI*m*z)*cos(M_PI*m*x);
  return(result);
}
static PetscScalar get_fp_stokes(PetscScalar x, PetscScalar z, PetscScalar eta0, PetscScalar eps0, PetscScalar np, PetscScalar phi_0, PetscScalar phi_s, PetscScalar p_s, PetscScalar psi_s, PetscScalar U_s, PetscScalar m, PetscScalar n, PetscScalar k_hat, PetscScalar R)
{ PetscScalar result;
  result = 2*pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z);
  return(result);
}
static PetscScalar get_fux_stokesdarcy(PetscScalar x, PetscScalar z, PetscScalar eta0, PetscScalar eps0, PetscScalar np, PetscScalar phi_0, PetscScalar phi_s, PetscScalar p_s, PetscScalar psi_s, PetscScalar U_s, PetscScalar m, PetscScalar n, PetscScalar k_hat, PetscScalar R)
{ PetscScalar result;
  result = -2*pow(M_PI, 3)*U_s*pow(m, 3)*(-0.66666666666666663*eta0*pow(sqrt(0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) - pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) + pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 1.0*pow(-1.0*pow(M_PI, 2)*U_s*pow(m, 2)*sin(M_PI*m*x)*sin(M_PI*m*z) + 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*x))*cos(M_PI*m*z) - 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*z))*cos(M_PI*m*x), 2))/eps0, -1 + 1.0/np) + eta0*pow(sqrt(0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) - pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) + pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 1.0*pow(-1.0*pow(M_PI, 2)*U_s*pow(m, 2)*sin(M_PI*m*x)*sin(M_PI*m*z) + 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*x))*cos(M_PI*m*z) - 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*z))*cos(M_PI*m*x), 2))/eps0, -1 + 1.0/np)/(phi_0*(phi_s*cos(M_PI*m*x)*cos(M_PI*m*z) + 1.0)))*sin(M_PI*m*x)*cos(M_PI*m*z) + 2*pow(M_PI, 2)*U_s*pow(m, 2)*(M_PI*eta0*m*phi_s*pow(sqrt(0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) - pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) + pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 1.0*pow(-1.0*pow(M_PI, 2)*U_s*pow(m, 2)*sin(M_PI*m*x)*sin(M_PI*m*z) + 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*x))*cos(M_PI*m*z) - 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*z))*cos(M_PI*m*x), 2))/eps0, -1 + 1.0/np)*sin(M_PI*m*x)*cos(M_PI*m*z)/(phi_0*pow(phi_s*cos(M_PI*m*x)*cos(M_PI*m*z) + 1.0, 2)) - 0.66666666666666663*eta0*pow(sqrt(0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) - pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) + pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 1.0*pow(-1.0*pow(M_PI, 2)*U_s*pow(m, 2)*sin(M_PI*m*x)*sin(M_PI*m*z) + 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*x))*cos(M_PI*m*z) - 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*z))*cos(M_PI*m*x), 2))/eps0, -1 + 1.0/np)*(-1 + 1.0/np)*(0.25*(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) - pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z))*(-2*pow(M_PI, 3)*U_s*pow(m, 3)*sin(M_PI*m*x)*cos(M_PI*m*z) - 2*pow(M_PI, 3)*pow(m, 3)*psi_s*sin(M_PI*m*z)*cos(M_PI*m*x)) + 0.25*(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) + pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z))*(-2*pow(M_PI, 3)*U_s*pow(m, 3)*sin(M_PI*m*x)*cos(M_PI*m*z) + 2*pow(M_PI, 3)*pow(m, 3)*psi_s*sin(M_PI*m*z)*cos(M_PI*m*x)) + 0.5*(-1.0*pow(M_PI, 2)*U_s*pow(m, 2)*sin(M_PI*m*x)*sin(M_PI*m*z) + 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*x))*cos(M_PI*m*z) - 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*z))*cos(M_PI*m*x))*(-2.0*pow(M_PI, 3)*U_s*pow(m, 3)*sin(M_PI*m*z)*cos(M_PI*m*x) + pow(M_PI, 3)*pow(m, 3)*psi_s*(1.0 - cos(M_PI*m*z))*sin(M_PI*m*x) + pow(M_PI, 3)*pow(m, 3)*psi_s*sin(M_PI*m*x)*cos(M_PI*m*z)))*1.0/(0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) - pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) + pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 1.0*pow(-1.0*pow(M_PI, 2)*U_s*pow(m, 2)*sin(M_PI*m*x)*sin(M_PI*m*z) + 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*x))*cos(M_PI*m*z) - 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*z))*cos(M_PI*m*x), 2)) + eta0*pow(sqrt(0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) - pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) + pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 1.0*pow(-1.0*pow(M_PI, 2)*U_s*pow(m, 2)*sin(M_PI*m*x)*sin(M_PI*m*z) + 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*x))*cos(M_PI*m*z) - 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*z))*cos(M_PI*m*x), 2))/eps0, -1 + 1.0/np)*(-1 + 1.0/np)*(0.25*(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) - pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z))*(-2*pow(M_PI, 3)*U_s*pow(m, 3)*sin(M_PI*m*x)*cos(M_PI*m*z) - 2*pow(M_PI, 3)*pow(m, 3)*psi_s*sin(M_PI*m*z)*cos(M_PI*m*x)) + 0.25*(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) + pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z))*(-2*pow(M_PI, 3)*U_s*pow(m, 3)*sin(M_PI*m*x)*cos(M_PI*m*z) + 2*pow(M_PI, 3)*pow(m, 3)*psi_s*sin(M_PI*m*z)*cos(M_PI*m*x)) + 0.5*(-1.0*pow(M_PI, 2)*U_s*pow(m, 2)*sin(M_PI*m*x)*sin(M_PI*m*z) + 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*x))*cos(M_PI*m*z) - 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*z))*cos(M_PI*m*x))*(-2.0*pow(M_PI, 3)*U_s*pow(m, 3)*sin(M_PI*m*z)*cos(M_PI*m*x) + pow(M_PI, 3)*pow(m, 3)*psi_s*(1.0 - cos(M_PI*m*z))*sin(M_PI*m*x) + pow(M_PI, 3)*pow(m, 3)*psi_s*sin(M_PI*m*x)*cos(M_PI*m*z)))*1.0/(0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) - pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) + pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 1.0*pow(-1.0*pow(M_PI, 2)*U_s*pow(m, 2)*sin(M_PI*m*x)*sin(M_PI*m*z) + 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*x))*cos(M_PI*m*z) - 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*z))*cos(M_PI*m*x), 2))/(phi_0*(phi_s*cos(M_PI*m*x)*cos(M_PI*m*z) + 1.0)))*cos(M_PI*m*x)*cos(M_PI*m*z) + 2.0*eta0*pow(sqrt(0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) - pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) + pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 1.0*pow(-1.0*pow(M_PI, 2)*U_s*pow(m, 2)*sin(M_PI*m*x)*sin(M_PI*m*z) + 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*x))*cos(M_PI*m*z) - 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*z))*cos(M_PI*m*x), 2))/eps0, -1 + 1.0/np)*(-1 + 1.0/np)*(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) + pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z))*(0.25*(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) - pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z))*(-2*pow(M_PI, 3)*U_s*pow(m, 3)*sin(M_PI*m*x)*cos(M_PI*m*z) - 2*pow(M_PI, 3)*pow(m, 3)*psi_s*sin(M_PI*m*z)*cos(M_PI*m*x)) + 0.25*(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) + pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z))*(-2*pow(M_PI, 3)*U_s*pow(m, 3)*sin(M_PI*m*x)*cos(M_PI*m*z) + 2*pow(M_PI, 3)*pow(m, 3)*psi_s*sin(M_PI*m*z)*cos(M_PI*m*x)) + 0.5*(-1.0*pow(M_PI, 2)*U_s*pow(m, 2)*sin(M_PI*m*x)*sin(M_PI*m*z) + 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*x))*cos(M_PI*m*z) - 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*z))*cos(M_PI*m*x))*(-2.0*pow(M_PI, 3)*U_s*pow(m, 3)*sin(M_PI*m*z)*cos(M_PI*m*x) + pow(M_PI, 3)*pow(m, 3)*psi_s*(1.0 - cos(M_PI*m*z))*sin(M_PI*m*x) + pow(M_PI, 3)*pow(m, 3)*psi_s*sin(M_PI*m*x)*cos(M_PI*m*z)))*1.0/(0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) - pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) + pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 1.0*pow(-1.0*pow(M_PI, 2)*U_s*pow(m, 2)*sin(M_PI*m*x)*sin(M_PI*m*z) + 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*x))*cos(M_PI*m*z) - 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*z))*cos(M_PI*m*x), 2)) + 2.0*eta0*pow(sqrt(0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) - pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) + pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 1.0*pow(-1.0*pow(M_PI, 2)*U_s*pow(m, 2)*sin(M_PI*m*x)*sin(M_PI*m*z) + 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*x))*cos(M_PI*m*z) - 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*z))*cos(M_PI*m*x), 2))/eps0, -1 + 1.0/np)*(-1 + 1.0/np)*(0.25*(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) - pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z))*(-2*pow(M_PI, 3)*U_s*pow(m, 3)*sin(M_PI*m*z)*cos(M_PI*m*x) - 2*pow(M_PI, 3)*pow(m, 3)*psi_s*sin(M_PI*m*x)*cos(M_PI*m*z)) + 0.25*(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) + pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z))*(-2*pow(M_PI, 3)*U_s*pow(m, 3)*sin(M_PI*m*z)*cos(M_PI*m*x) + 2*pow(M_PI, 3)*pow(m, 3)*psi_s*sin(M_PI*m*x)*cos(M_PI*m*z)) + 0.5*(-1.0*pow(M_PI, 2)*U_s*pow(m, 2)*sin(M_PI*m*x)*sin(M_PI*m*z) + 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*x))*cos(M_PI*m*z) - 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*z))*cos(M_PI*m*x))*(-2.0*pow(M_PI, 3)*U_s*pow(m, 3)*sin(M_PI*m*x)*cos(M_PI*m*z) - 1.0*pow(M_PI, 3)*pow(m, 3)*psi_s*(1.0 - cos(M_PI*m*x))*sin(M_PI*m*z) - 1.0*pow(M_PI, 3)*pow(m, 3)*psi_s*sin(M_PI*m*z)*cos(M_PI*m*x)))*(-1.0*pow(M_PI, 2)*U_s*pow(m, 2)*sin(M_PI*m*x)*sin(M_PI*m*z) + 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*x))*cos(M_PI*m*z) - 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*z))*cos(M_PI*m*x))*1.0/(0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) - pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) + pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 1.0*pow(-1.0*pow(M_PI, 2)*U_s*pow(m, 2)*sin(M_PI*m*x)*sin(M_PI*m*z) + 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*x))*cos(M_PI*m*z) - 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*z))*cos(M_PI*m*x), 2)) + 2.0*eta0*pow(sqrt(0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) - pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) + pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 1.0*pow(-1.0*pow(M_PI, 2)*U_s*pow(m, 2)*sin(M_PI*m*x)*sin(M_PI*m*z) + 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*x))*cos(M_PI*m*z) - 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*z))*cos(M_PI*m*x), 2))/eps0, -1 + 1.0/np)*(-pow(M_PI, 3)*U_s*pow(m, 3)*sin(M_PI*m*x)*cos(M_PI*m*z) + pow(M_PI, 3)*pow(m, 3)*psi_s*sin(M_PI*m*z)*cos(M_PI*m*x)) + 2.0*eta0*pow(sqrt(0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) - pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) + pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 1.0*pow(-1.0*pow(M_PI, 2)*U_s*pow(m, 2)*sin(M_PI*m*x)*sin(M_PI*m*z) + 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*x))*cos(M_PI*m*z) - 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*z))*cos(M_PI*m*x), 2))/eps0, -1 + 1.0/np)*(-1.0*pow(M_PI, 3)*U_s*pow(m, 3)*sin(M_PI*m*x)*cos(M_PI*m*z) - 0.5*pow(M_PI, 3)*pow(m, 3)*psi_s*(1.0 - cos(M_PI*m*x))*sin(M_PI*m*z) - 0.5*pow(M_PI, 3)*pow(m, 3)*psi_s*sin(M_PI*m*z)*cos(M_PI*m*x)) + M_PI*m*p_s*sin(M_PI*m*x)*cos(M_PI*m*z);
  return(result);
}
static PetscScalar get_fuz_stokesdarcy(PetscScalar x, PetscScalar z, PetscScalar eta0, PetscScalar eps0, PetscScalar np, PetscScalar phi_0, PetscScalar phi_s, PetscScalar p_s, PetscScalar psi_s, PetscScalar U_s, PetscScalar m, PetscScalar n, PetscScalar k_hat, PetscScalar R)
{ PetscScalar result;
  result = -2*pow(M_PI, 3)*U_s*pow(m, 3)*(-0.66666666666666663*eta0*pow(sqrt(0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) - pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) + pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 1.0*pow(-1.0*pow(M_PI, 2)*U_s*pow(m, 2)*sin(M_PI*m*x)*sin(M_PI*m*z) + 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*x))*cos(M_PI*m*z) - 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*z))*cos(M_PI*m*x), 2))/eps0, -1 + 1.0/np) + eta0*pow(sqrt(0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) - pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) + pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 1.0*pow(-1.0*pow(M_PI, 2)*U_s*pow(m, 2)*sin(M_PI*m*x)*sin(M_PI*m*z) + 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*x))*cos(M_PI*m*z) - 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*z))*cos(M_PI*m*x), 2))/eps0, -1 + 1.0/np)/(phi_0*(phi_s*cos(M_PI*m*x)*cos(M_PI*m*z) + 1.0)))*sin(M_PI*m*z)*cos(M_PI*m*x) + 2*pow(M_PI, 2)*U_s*pow(m, 2)*(M_PI*eta0*m*phi_s*pow(sqrt(0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) - pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) + pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 1.0*pow(-1.0*pow(M_PI, 2)*U_s*pow(m, 2)*sin(M_PI*m*x)*sin(M_PI*m*z) + 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*x))*cos(M_PI*m*z) - 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*z))*cos(M_PI*m*x), 2))/eps0, -1 + 1.0/np)*sin(M_PI*m*z)*cos(M_PI*m*x)/(phi_0*pow(phi_s*cos(M_PI*m*x)*cos(M_PI*m*z) + 1.0, 2)) - 0.66666666666666663*eta0*pow(sqrt(0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) - pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) + pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 1.0*pow(-1.0*pow(M_PI, 2)*U_s*pow(m, 2)*sin(M_PI*m*x)*sin(M_PI*m*z) + 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*x))*cos(M_PI*m*z) - 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*z))*cos(M_PI*m*x), 2))/eps0, -1 + 1.0/np)*(-1 + 1.0/np)*(0.25*(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) - pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z))*(-2*pow(M_PI, 3)*U_s*pow(m, 3)*sin(M_PI*m*z)*cos(M_PI*m*x) - 2*pow(M_PI, 3)*pow(m, 3)*psi_s*sin(M_PI*m*x)*cos(M_PI*m*z)) + 0.25*(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) + pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z))*(-2*pow(M_PI, 3)*U_s*pow(m, 3)*sin(M_PI*m*z)*cos(M_PI*m*x) + 2*pow(M_PI, 3)*pow(m, 3)*psi_s*sin(M_PI*m*x)*cos(M_PI*m*z)) + 0.5*(-1.0*pow(M_PI, 2)*U_s*pow(m, 2)*sin(M_PI*m*x)*sin(M_PI*m*z) + 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*x))*cos(M_PI*m*z) - 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*z))*cos(M_PI*m*x))*(-2.0*pow(M_PI, 3)*U_s*pow(m, 3)*sin(M_PI*m*x)*cos(M_PI*m*z) - 1.0*pow(M_PI, 3)*pow(m, 3)*psi_s*(1.0 - cos(M_PI*m*x))*sin(M_PI*m*z) - 1.0*pow(M_PI, 3)*pow(m, 3)*psi_s*sin(M_PI*m*z)*cos(M_PI*m*x)))*1.0/(0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) - pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) + pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 1.0*pow(-1.0*pow(M_PI, 2)*U_s*pow(m, 2)*sin(M_PI*m*x)*sin(M_PI*m*z) + 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*x))*cos(M_PI*m*z) - 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*z))*cos(M_PI*m*x), 2)) + eta0*pow(sqrt(0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) - pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) + pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 1.0*pow(-1.0*pow(M_PI, 2)*U_s*pow(m, 2)*sin(M_PI*m*x)*sin(M_PI*m*z) + 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*x))*cos(M_PI*m*z) - 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*z))*cos(M_PI*m*x), 2))/eps0, -1 + 1.0/np)*(-1 + 1.0/np)*(0.25*(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) - pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z))*(-2*pow(M_PI, 3)*U_s*pow(m, 3)*sin(M_PI*m*z)*cos(M_PI*m*x) - 2*pow(M_PI, 3)*pow(m, 3)*psi_s*sin(M_PI*m*x)*cos(M_PI*m*z)) + 0.25*(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) + pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z))*(-2*pow(M_PI, 3)*U_s*pow(m, 3)*sin(M_PI*m*z)*cos(M_PI*m*x) + 2*pow(M_PI, 3)*pow(m, 3)*psi_s*sin(M_PI*m*x)*cos(M_PI*m*z)) + 0.5*(-1.0*pow(M_PI, 2)*U_s*pow(m, 2)*sin(M_PI*m*x)*sin(M_PI*m*z) + 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*x))*cos(M_PI*m*z) - 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*z))*cos(M_PI*m*x))*(-2.0*pow(M_PI, 3)*U_s*pow(m, 3)*sin(M_PI*m*x)*cos(M_PI*m*z) - 1.0*pow(M_PI, 3)*pow(m, 3)*psi_s*(1.0 - cos(M_PI*m*x))*sin(M_PI*m*z) - 1.0*pow(M_PI, 3)*pow(m, 3)*psi_s*sin(M_PI*m*z)*cos(M_PI*m*x)))*1.0/(0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) - pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) + pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 1.0*pow(-1.0*pow(M_PI, 2)*U_s*pow(m, 2)*sin(M_PI*m*x)*sin(M_PI*m*z) + 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*x))*cos(M_PI*m*z) - 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*z))*cos(M_PI*m*x), 2))/(phi_0*(phi_s*cos(M_PI*m*x)*cos(M_PI*m*z) + 1.0)))*cos(M_PI*m*x)*cos(M_PI*m*z) + 2.0*eta0*pow(sqrt(0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) - pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) + pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 1.0*pow(-1.0*pow(M_PI, 2)*U_s*pow(m, 2)*sin(M_PI*m*x)*sin(M_PI*m*z) + 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*x))*cos(M_PI*m*z) - 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*z))*cos(M_PI*m*x), 2))/eps0, -1 + 1.0/np)*(-1 + 1.0/np)*(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) - pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z))*(0.25*(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) - pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z))*(-2*pow(M_PI, 3)*U_s*pow(m, 3)*sin(M_PI*m*z)*cos(M_PI*m*x) - 2*pow(M_PI, 3)*pow(m, 3)*psi_s*sin(M_PI*m*x)*cos(M_PI*m*z)) + 0.25*(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) + pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z))*(-2*pow(M_PI, 3)*U_s*pow(m, 3)*sin(M_PI*m*z)*cos(M_PI*m*x) + 2*pow(M_PI, 3)*pow(m, 3)*psi_s*sin(M_PI*m*x)*cos(M_PI*m*z)) + 0.5*(-1.0*pow(M_PI, 2)*U_s*pow(m, 2)*sin(M_PI*m*x)*sin(M_PI*m*z) + 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*x))*cos(M_PI*m*z) - 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*z))*cos(M_PI*m*x))*(-2.0*pow(M_PI, 3)*U_s*pow(m, 3)*sin(M_PI*m*x)*cos(M_PI*m*z) - 1.0*pow(M_PI, 3)*pow(m, 3)*psi_s*(1.0 - cos(M_PI*m*x))*sin(M_PI*m*z) - 1.0*pow(M_PI, 3)*pow(m, 3)*psi_s*sin(M_PI*m*z)*cos(M_PI*m*x)))*1.0/(0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) - pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) + pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 1.0*pow(-1.0*pow(M_PI, 2)*U_s*pow(m, 2)*sin(M_PI*m*x)*sin(M_PI*m*z) + 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*x))*cos(M_PI*m*z) - 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*z))*cos(M_PI*m*x), 2)) + 2.0*eta0*pow(sqrt(0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) - pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) + pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 1.0*pow(-1.0*pow(M_PI, 2)*U_s*pow(m, 2)*sin(M_PI*m*x)*sin(M_PI*m*z) + 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*x))*cos(M_PI*m*z) - 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*z))*cos(M_PI*m*x), 2))/eps0, -1 + 1.0/np)*(-1 + 1.0/np)*(0.25*(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) - pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z))*(-2*pow(M_PI, 3)*U_s*pow(m, 3)*sin(M_PI*m*x)*cos(M_PI*m*z) - 2*pow(M_PI, 3)*pow(m, 3)*psi_s*sin(M_PI*m*z)*cos(M_PI*m*x)) + 0.25*(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) + pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z))*(-2*pow(M_PI, 3)*U_s*pow(m, 3)*sin(M_PI*m*x)*cos(M_PI*m*z) + 2*pow(M_PI, 3)*pow(m, 3)*psi_s*sin(M_PI*m*z)*cos(M_PI*m*x)) + 0.5*(-1.0*pow(M_PI, 2)*U_s*pow(m, 2)*sin(M_PI*m*x)*sin(M_PI*m*z) + 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*x))*cos(M_PI*m*z) - 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*z))*cos(M_PI*m*x))*(-2.0*pow(M_PI, 3)*U_s*pow(m, 3)*sin(M_PI*m*z)*cos(M_PI*m*x) + pow(M_PI, 3)*pow(m, 3)*psi_s*(1.0 - cos(M_PI*m*z))*sin(M_PI*m*x) + pow(M_PI, 3)*pow(m, 3)*psi_s*sin(M_PI*m*x)*cos(M_PI*m*z)))*(-1.0*pow(M_PI, 2)*U_s*pow(m, 2)*sin(M_PI*m*x)*sin(M_PI*m*z) + 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*x))*cos(M_PI*m*z) - 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*z))*cos(M_PI*m*x))*1.0/(0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) - pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) + pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 1.0*pow(-1.0*pow(M_PI, 2)*U_s*pow(m, 2)*sin(M_PI*m*x)*sin(M_PI*m*z) + 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*x))*cos(M_PI*m*z) - 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*z))*cos(M_PI*m*x), 2)) + 2.0*eta0*pow(sqrt(0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) - pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) + pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 1.0*pow(-1.0*pow(M_PI, 2)*U_s*pow(m, 2)*sin(M_PI*m*x)*sin(M_PI*m*z) + 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*x))*cos(M_PI*m*z) - 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*z))*cos(M_PI*m*x), 2))/eps0, -1 + 1.0/np)*(-pow(M_PI, 3)*U_s*pow(m, 3)*sin(M_PI*m*z)*cos(M_PI*m*x) - pow(M_PI, 3)*pow(m, 3)*psi_s*sin(M_PI*m*x)*cos(M_PI*m*z)) + 2.0*eta0*pow(sqrt(0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) - pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 0.5*pow(pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z) + pow(M_PI, 2)*pow(m, 2)*psi_s*sin(M_PI*m*x)*sin(M_PI*m*z), 2) + 1.0*pow(-1.0*pow(M_PI, 2)*U_s*pow(m, 2)*sin(M_PI*m*x)*sin(M_PI*m*z) + 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*x))*cos(M_PI*m*z) - 0.5*pow(M_PI, 2)*pow(m, 2)*psi_s*(1.0 - cos(M_PI*m*z))*cos(M_PI*m*x), 2))/eps0, -1 + 1.0/np)*(-1.0*pow(M_PI, 3)*U_s*pow(m, 3)*sin(M_PI*m*z)*cos(M_PI*m*x) + 0.5*pow(M_PI, 3)*pow(m, 3)*psi_s*(1.0 - cos(M_PI*m*z))*sin(M_PI*m*x) + 0.5*pow(M_PI, 3)*pow(m, 3)*psi_s*sin(M_PI*m*x)*cos(M_PI*m*z)) + k_hat*phi_0*(phi_s*cos(M_PI*m*x)*cos(M_PI*m*z) + 1.0) + M_PI*m*p_s*sin(M_PI*m*z)*cos(M_PI*m*x);
  return(result);
}
static PetscScalar get_fp_stokesdarcy(PetscScalar x, PetscScalar z, PetscScalar eta0, PetscScalar eps0, PetscScalar np, PetscScalar phi_0, PetscScalar phi_s, PetscScalar p_s, PetscScalar psi_s, PetscScalar U_s, PetscScalar m, PetscScalar n, PetscScalar k_hat, PetscScalar R)
{ PetscScalar result;
  result = pow(R, 2)*(-pow(M_PI, 2)*pow(m, 2)*n*p_s*phi_s*pow(phi_s*cos(M_PI*m*x)*cos(M_PI*m*z) + 1.0, n)*pow(sin(M_PI*m*x), 2)*pow(cos(M_PI*m*z), 2)/(phi_s*cos(M_PI*m*x)*cos(M_PI*m*z) + 1.0) + 2*pow(M_PI, 2)*pow(m, 2)*p_s*pow(phi_s*cos(M_PI*m*x)*cos(M_PI*m*z) + 1.0, n)*cos(M_PI*m*x)*cos(M_PI*m*z) + M_PI*m*n*phi_s*(-k_hat - M_PI*m*p_s*sin(M_PI*m*z)*cos(M_PI*m*x))*pow(phi_s*cos(M_PI*m*x)*cos(M_PI*m*z) + 1.0, n)*sin(M_PI*m*z)*cos(M_PI*m*x)/(phi_s*cos(M_PI*m*x)*cos(M_PI*m*z) + 1.0)) + 2*pow(M_PI, 2)*U_s*pow(m, 2)*cos(M_PI*m*x)*cos(M_PI*m*z);
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
  DM             dm;
  Vec            x, xMMS, xepsMMS;
  PetscInt       nx, nz;
  PetscScalar    xmin, zmin, xmax, zmax;
  char           fout[FNAME_LENGTH];
  PetscInt       test;
  PetscFunctionBeginUser;

  // Element count
  nx = usr->par->nx;
  nz = usr->par->nz;

  // Domain coords
  xmin = usr->par->xmin;
  zmin = usr->par->zmin;
  xmax = usr->par->xmin+usr->par->L;
  zmax = usr->par->zmin+usr->par->H;

  // passing solver index to test
  test = usr->par->test;
  
  // Create the FD-pde object
  if (test==1) { PetscCall(FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_STOKES,&fd)); }
  if (test==2) { PetscCall(FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_STOKESDARCY2FIELD,&fd)); }

  PetscCall(FDPDESetUp(fd));
  PetscCall(FDPDEGetDM(fd,&dm)); 

  // Create DM/vec for strain rates
  PetscCall(DMStagCreateCompatibleDMStag(dm,4,0,4,0,&usr->dmeps)); 
  PetscCall(DMSetUp(usr->dmeps)); 
  PetscCall(DMStagSetUniformCoordinatesProduct(usr->dmeps,xmin,xmax,zmin,zmax,0.0,0.0));
  PetscCall(DMCreateGlobalVector(usr->dmeps,&usr->xeps)); 

  // Create manufactured solution
  PetscCall(ComputeManufacturedSolution(dm,&xMMS,&xepsMMS,usr,test)); 

  // Set BC evaluation function
  PetscCall(FDPDESetFunctionBCList(fd,FormBCList,"n/a",usr)); 

  // Set coefficients evaluation function
  if (test==1) { PetscCall(FDPDESetFunctionCoefficient(fd,FormCoefficient_Stokes,"n/a",usr));  }
  if (test==2) { PetscCall(FDPDESetFunctionCoefficient(fd,FormCoefficient_StokesDarcy,"n/a",usr));  }
  PetscCall(FDPDEView(fd)); 
  
  // Create initial guess with a linear viscous (np=1.0)
  PetscScalar np;
  Vec         xguess;
  np = usr->par->np;
  usr->par->np = 1.0;
  PetscCall(FDPDESolve(fd,NULL));
  PetscCall(FDPDEGetSolution(fd,&x)); 
  PetscCall(FDPDEGetSolutionGuess(fd,&xguess));  
  PetscCall(VecCopy(x,xguess));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&xguess));
  usr->par->np = np;
  
  // FD SNES Solver
  PetscCall(FDPDESolve(fd,NULL));
  PetscCall(FDPDEGetSolution(fd,&x)); 

  // Output solution to file
  if (test==1) { PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s_stokes",usr->par->fdir_out,usr->par->fname_out)); }
  if (test==2) { PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s_stokesdarcy",usr->par->fdir_out,usr->par->fname_out)); }
  PetscCall(DMStagViewBinaryPython(dm,x,fout));

  if (test==1) { PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s_strain_stokes",usr->par->fdir_out,usr->par->fname_out)); }
  if (test==2) { PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s_strain_stokesdarcy",usr->par->fdir_out,usr->par->fname_out)); }
  PetscCall(DMStagViewBinaryPython(usr->dmeps,usr->xeps,fout));

  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s_residual",usr->par->fdir_out,usr->par->fname_out));
  PetscCall(DMStagViewBinaryPython(dm,fd->r,fout));

  // Compute norms
  PetscCall(ComputeErrorNorms(dm,x,xMMS,xepsMMS,test,usr));

  // Destroy objects
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&xMMS));
  PetscCall(VecDestroy(&xepsMMS));
  PetscCall(VecDestroy(&usr->xeps));
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
  PetscCall(PetscBagRegisterInt(bag, &par->nx, 4, "nx", "Element count in the x-dir")); 
  PetscCall(PetscBagRegisterInt(bag, &par->nz, 5, "nz", "Element count in the z-dir")); 

  PetscCall(PetscBagRegisterScalar(bag, &par->xmin, 0.0, "xmin", "Start coordinate of domain in x-dir")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->zmin, 0.0, "zmin", "Start coordinate of domain in z-dir")); 

  PetscCall(PetscBagRegisterScalar(bag, &par->L, 1.0, "L", "Length of domain in x-dir")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->H, 1.0, "H", "Height of domain in z-dir")); 

  // Physical and material parameters
  PetscCall(PetscBagRegisterScalar(bag, &par->eta0, 1.0, "eta0", "Reference shear viscosity")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->eps0, 0.1, "eps0", "Reference background strainrate")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->np, 1.0, "nexp", "Power-law exponent")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->phi_0, 0.1, "phi_0", "Reference porosity")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->phi_s, 0.1, "phi_s", "Porosity amplitude")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->p_s, 1.0, "p_s", "Pressure amplitude")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->psi_s, 1.0, "psi_s", "Vector potential function amplitude")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->U_s, 1.0, "U_s", "Scalar potential function amplitude")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->m, 2.0, "m", "Trigonometric coefficient")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->n, 2.0, "n", "Porosity exponent")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->k_hat, 0.0, "k_hat", "Direction of unit vertical vector +/-1.0")); 
  PetscCall(PetscBagRegisterScalar(bag, &par->R, 1.0, "R", "Compaction factor R = delta/h")); 

  par->etamax = 1.0e+5*par->eta0;
  par->etamin = 1.0e-5*par->eta0;

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
  PetscPrintf(usr->comm,"# Test_effective_viscosity: %s \n",&(date[0]));
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
// FormCoefficient_Stokes
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormCoefficient_Stokes"
PetscErrorCode FormCoefficient_Stokes(FDPDE fd, DM dm, Vec x, DM dmcoeff, Vec coeff, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz;
  Vec            coefflocal, xlocal, xepslocal;
  PetscScalar    **coordx,**coordz;
  PetscInt       iprev, inext, icenter;
  PetscScalar    eta0,eps0,np,R,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat;
  PetscScalar    ***c;
  PetscFunctionBeginUser;

  eta0 = usr->par->eta0;
  eps0 = usr->par->eps0;
  np   = usr->par->np;
  R    = usr->par->R;
  phi_0= usr->par->phi_0;
  phi_s= usr->par->phi_s;
  p_s  = usr->par->p_s;
  psi_s= usr->par->psi_s;
  U_s  = usr->par->U_s;
  m    = usr->par->m;
  n    = usr->par->n;
  k_hat= usr->par->k_hat;

  // Strain rates
  PetscCall(UpdateStrainRates(dm,x,usr)); 
  PetscCall(DMGetLocalVector(usr->dmeps, &xepslocal)); 
  PetscCall(DMGlobalToLocal (usr->dmeps, usr->xeps, INSERT_VALUES, xepslocal)); 

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

  PetscCall(DMGetLocalVector(dm, &xlocal)); 
  PetscCall(DMGlobalToLocal (dm, x, INSERT_VALUES, xlocal)); 

  // Loop over local domain - set initial density and viscosity
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      PetscInt idx;

      { // A = eta (center, c=1)
        DMStagStencil point;
        PetscScalar   eta,epsII;

        point.i = i; point.j = j; point.loc = ELEMENT; point.c = 3;
        PetscCall(DMStagVecGetValuesStencil(usr->dmeps,xepslocal,1,&point,&epsII)); 

        eta = eta0*PetscPowScalar(epsII/eps0,1.0/np-1.0);
        if      (eta > usr->par->etamax) eta = usr->par->etamax;
        else if (eta < usr->par->etamin) eta = usr->par->etamin;
        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, 1, &idx)); 
        c[j][i][idx] = eta;
      }

      { // A = eta (corner, c=0)
        DMStagStencil point[4];
        PetscScalar   eta,epsII[4];
        PetscInt      ii;

        point[0].i = i; point[0].j = j; point[0].loc = DOWN_LEFT;  point[0].c = 3;
        point[1].i = i; point[1].j = j; point[1].loc = DOWN_RIGHT; point[1].c = 3;
        point[2].i = i; point[2].j = j; point[2].loc = UP_LEFT;    point[2].c = 3;
        point[3].i = i; point[3].j = j; point[3].loc = UP_RIGHT;   point[3].c = 3;
        PetscCall(DMStagVecGetValuesStencil(usr->dmeps,xepslocal,4,point,epsII)); 

        for (ii = 0; ii < 4; ii++) {
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, 0, &idx)); 
          eta = eta0*PetscPowScalar(epsII[ii]/eps0,1.0/np-1.0);
          if      (eta > usr->par->etamax) eta = usr->par->etamax;
          else if (eta < usr->par->etamin) eta = usr->par->etamin;
          c[j][i][idx] = eta;
        }
      }

      { // B = fu (fux,fuz - manufactured) (edges, c=0)
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

        rhs[0] = get_fux_stokes(xp[0],zp[0],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
        rhs[1] = get_fux_stokes(xp[1],zp[1],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
        rhs[2] = get_fuz_stokes(xp[2],zp[2],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
        rhs[3] = get_fuz_stokes(xp[3],zp[3],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);

        for (ii = 0; ii < 4; ii++) {
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] = rhs[ii];
        }
      }

      { // C = fp (manufactured) (center, c=0)
        DMStagStencil point;
        PetscScalar   xp, zp;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 0;
        xp = coordx[i][icenter]; zp = coordz[j][icenter];
        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
        c[j][i][idx] = get_fp_stokes(xp,zp,eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
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
  PetscCall(DMRestoreLocalVector(usr->dmeps,&xepslocal)); 

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// FormCoefficient_StokesDarcy
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormCoefficient_StokesDarcy"
PetscErrorCode FormCoefficient_StokesDarcy(FDPDE fd, DM dm, Vec x, DM dmcoeff, Vec coeff, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz;
  Vec            coefflocal, xlocal, xepslocal;
  PetscScalar    **coordx,**coordz;
  PetscInt       iprev, inext, icenter;
  PetscScalar    eta0,eps0,np,R,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat;
  PetscScalar    ***c;
  PetscFunctionBeginUser;

  eta0 = usr->par->eta0;
  eps0 = usr->par->eps0;
  np   = usr->par->np;
  R    = usr->par->R;
  phi_0= usr->par->phi_0;
  phi_s= usr->par->phi_s;
  p_s  = usr->par->p_s;
  psi_s= usr->par->psi_s;
  U_s  = usr->par->U_s;
  m    = usr->par->m;
  n    = usr->par->n;
  k_hat= usr->par->k_hat;

  // Strain rates
  PetscCall(UpdateStrainRates(dm,x,usr)); 
  PetscCall(DMGetLocalVector(usr->dmeps, &xepslocal)); 
  PetscCall(DMGlobalToLocal (usr->dmeps, usr->xeps, INSERT_VALUES, xepslocal)); 

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

  PetscCall(DMGetLocalVector(dm, &xlocal)); 
  PetscCall(DMGlobalToLocal (dm, x, INSERT_VALUES, xlocal)); 
  
  // Loop over local domain - set initial density and viscosity
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      PetscInt idx;

      { // A = eta (center, c=1)
        DMStagStencil point;
        PetscScalar   eta,epsII;

        point.i = i; point.j = j; point.loc = ELEMENT; point.c = 3;
        PetscCall(DMStagVecGetValuesStencil(usr->dmeps,xepslocal,1,&point,&epsII)); 

        eta = eta0*PetscPowScalar(epsII/eps0,1.0/np-1.0);
        if      (eta > usr->par->etamax) eta = usr->par->etamax;
        else if (eta < usr->par->etamin) eta = usr->par->etamin;
        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, 1, &idx)); 
        c[j][i][idx] = eta;
      }

      { // A = eta (corner, c=0)
        DMStagStencil point[4];
        PetscScalar   eta,epsII[4];
        PetscInt      ii;

        point[0].i = i; point[0].j = j; point[0].loc = DOWN_LEFT;  point[0].c = 3;
        point[1].i = i; point[1].j = j; point[1].loc = DOWN_RIGHT; point[1].c = 3;
        point[2].i = i; point[2].j = j; point[2].loc = UP_LEFT;    point[2].c = 3;
        point[3].i = i; point[3].j = j; point[3].loc = UP_RIGHT;   point[3].c = 3;
        PetscCall(DMStagVecGetValuesStencil(usr->dmeps,xepslocal,4,point,epsII)); 

        for (ii = 0; ii < 4; ii++) {
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, 0, &idx)); 
          eta = eta0*PetscPowScalar(epsII[ii]/eps0,1.0/np-1.0);
          if      (eta > usr->par->etamax) eta = usr->par->etamax;
          else if (eta < usr->par->etamin) eta = usr->par->etamin;
          c[j][i][idx] = eta;
        }
      }

      { // B = -phi*k_hat + fu (fux,fuz - manufactured) (edges, c=0)
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

        // Bx = fux
        rhs[0] = get_fux_stokesdarcy(xp[0],zp[0],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
        rhs[1] = get_fux_stokesdarcy(xp[1],zp[1],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);

        // Bz = -phi*k_hat + fuz
        rhs[2] = get_fuz_stokesdarcy(xp[2],zp[2],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
        rhs[3] = get_fuz_stokesdarcy(xp[3],zp[3],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);

        rhs[2] -= k_hat*get_phi(xp[2],zp[2],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
        rhs[3] -= k_hat*get_phi(xp[3],zp[3],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);

        for (ii = 0; ii < 4; ii++) {
          PetscCall(DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx)); 
          c[j][i][idx] = rhs[ii];
        }
      }

      { // C = (manufactured) (center, c=0)
        DMStagStencil point;
        PetscScalar   xp, zp;

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 0;
        xp = coordx[i][icenter]; zp = coordz[j][icenter];
        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx)); 
        c[j][i][idx] = get_fp_stokesdarcy(xp,zp,eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
      }

      { // D1 = xi=zeta-2/3eta (center, c=2)
        DMStagStencil point;
        PetscScalar   phi,xi,zeta,eta,epsII;

        point.i = i; point.j = j; point.loc = ELEMENT; point.c = 3;
        PetscCall(DMStagVecGetValuesStencil(usr->dmeps,xepslocal,1,&point,&epsII)); 

        eta = eta0*PetscPowScalar(epsII/eps0,1.0/np-1.0);
        if      (eta > usr->par->etamax) eta = usr->par->etamax;
        else if (eta < usr->par->etamin) eta = usr->par->etamin;

        phi  = get_phi(coordx[i][icenter],coordz[j][icenter],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
        zeta = eta/phi;
        xi   = zeta-2.0/3.0*eta;

        PetscCall(DMStagGetLocationSlot(dmcoeff, point.loc, 2, &idx)); 
        c[j][i][idx] = xi;
      }

      { // D2 = -R^2*Kphi (edges, c=1)
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
          c[j][i][idx] = -R*R*get_Kphi(xp[ii],zp[ii],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
        }
      }

      { // D3 = R^2*Kphi*k_hat (edges, c=2)
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
          c[j][i][idx] = R*R*k_hat*get_Kphi(xp[ii],zp[ii],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
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
  PetscCall(DMRestoreLocalVector(usr->dmeps,&xepslocal)); 

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
  PetscInt       k,n_bc,*idx_bc, test;
  PetscScalar    *value_bc,*x_bc;
  PetscScalar    eta0,eps0,np,R,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat;
  BCType         *type_bc;
  PetscFunctionBeginUser;

  eta0 = usr->par->eta0;
  eps0 = usr->par->eps0;
  np   = usr->par->np;
  R    = usr->par->R;
  phi_0= usr->par->phi_0;
  phi_s= usr->par->phi_s;
  p_s  = usr->par->p_s;
  psi_s= usr->par->psi_s;
  U_s  = usr->par->U_s;
  m    = usr->par->m;
  n    = usr->par->n;
  k_hat= usr->par->k_hat;

  test = usr->par->test;

  // LEFT Boundary - Vx
  PetscCall(DMStagBCListGetValues(bclist,'w','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_ux(x_bc[2*k],x_bc[2*k+1],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // LEFT Boundary - Vz
  PetscCall(DMStagBCListGetValues(bclist,'w','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_uz(x_bc[2*k],x_bc[2*k+1],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // RIGHT Boundary - Vx
  PetscCall(DMStagBCListGetValues(bclist,'e','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_ux(x_bc[2*k],x_bc[2*k+1],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // RIGHT Boundary - Vz
  PetscCall(DMStagBCListGetValues(bclist,'e','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_uz(x_bc[2*k],x_bc[2*k+1],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // DOWN Boundary - Vx
  PetscCall(DMStagBCListGetValues(bclist,'s','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_ux(x_bc[2*k],x_bc[2*k+1],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // DOWN Boundary - Vz
  PetscCall(DMStagBCListGetValues(bclist,'s','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_uz(x_bc[2*k],x_bc[2*k+1],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // UP Boundary - Vx
  PetscCall(DMStagBCListGetValues(bclist,'n','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_ux(x_bc[2*k],x_bc[2*k+1],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  // UP Boundary - Vz
  PetscCall(DMStagBCListGetValues(bclist,'n','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  for (k=0; k<n_bc; k++) {
    value_bc[k] = get_uz(x_bc[2*k],x_bc[2*k+1],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
    type_bc[k] = BC_DIRICHLET;
  }
  PetscCall(DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

  if (test==1) {
    // Pin pressure at the entire bottom boundary
    PetscCall(DMStagBCListGetValues(bclist,'s','o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
    for (k=0; k<n_bc; k++) {
      value_bc[k] = get_p(x_bc[2*k],x_bc[2*k+1],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
      type_bc[k] = BC_DIRICHLET_STAG;
    }
    PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,NULL,&x_bc,&value_bc,&type_bc));
  }
  
  if (test==2) {
    // LEFT Boundary - P
    PetscCall(DMStagBCListGetValues(bclist,'w','o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
    for (k=0; k<n_bc; k++) {
      value_bc[k] = get_p(x_bc[2*k],x_bc[2*k+1],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
      type_bc[k] = BC_DIRICHLET;
    }
    PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

    // RIGHT Boundary - P
    PetscCall(DMStagBCListGetValues(bclist,'e','o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
    for (k=0; k<n_bc; k++) {
      value_bc[k] = get_p(x_bc[2*k],x_bc[2*k+1],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
      type_bc[k] = BC_DIRICHLET;
    }
    PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

    // DOWN Boundary - P
    PetscCall(DMStagBCListGetValues(bclist,'s','o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
    for (k=0; k<n_bc; k++) {
      value_bc[k] = get_p(x_bc[2*k],x_bc[2*k+1],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
      type_bc[k] = BC_DIRICHLET;
    }
    PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));

    // UP Boundary - P
    PetscCall(DMStagBCListGetValues(bclist,'n','o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
    for (k=0; k<n_bc; k++) {
      value_bc[k] = get_p(x_bc[2*k],x_bc[2*k+1],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
      type_bc[k] = BC_DIRICHLET;
    }
    PetscCall(DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// Compute manufactured solution
// ---------------------------------------
PetscErrorCode ComputeManufacturedSolution(DM dm,Vec *_xMMS, Vec *_xepsMMS, void *ctx, PetscInt test)
{
  UsrData       *usr = (UsrData*) ctx;
  DM             dmeps;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz, idx;
  PetscInt       iprev, inext, icenter;
  PetscScalar    ***xxMMS, ***xxeps, ***xxrhs;
  PetscScalar    **coordx,**coordz;
  Vec            xMMS, xMMSlocal, xeps, xepslocal, xrhs, xrhslocal;
  PetscScalar    eta0,eps0,np,R,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat;
  char           fout[FNAME_LENGTH];
  PetscFunctionBeginUser;

  eta0 = usr->par->eta0;
  eps0 = usr->par->eps0;
  np   = usr->par->np;
  R    = usr->par->R;
  phi_0= usr->par->phi_0;
  phi_s= usr->par->phi_s;
  p_s  = usr->par->p_s;
  psi_s= usr->par->psi_s;
  U_s  = usr->par->U_s;
  m    = usr->par->m;
  n    = usr->par->n;
  k_hat= usr->par->k_hat;

  dmeps = usr->dmeps;

  // Create local and global vectors for MMS solution
  PetscCall(DMCreateGlobalVector(dm,&xMMS     )); 
  PetscCall(DMCreateLocalVector (dm,&xMMSlocal)); 
  PetscCall(DMStagVecGetArray(dm,xMMSlocal,&xxMMS)); 

  PetscCall(DMCreateGlobalVector(dmeps,&xeps     )); 
  PetscCall(DMCreateLocalVector (dmeps,&xepslocal)); 
  PetscCall(DMStagVecGetArray(dmeps,xepslocal,&xxeps)); 

  PetscCall(DMCreateGlobalVector(dm,&xrhs     )); 
  PetscCall(DMCreateLocalVector (dm,&xrhslocal)); 
  PetscCall(DMStagVecGetArray(dm,xrhslocal,&xxrhs)); 

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
      PetscScalar    exxc, ezzc, exzc, exxn[4], ezzn[4], exzn[4];

      // pressure
      PetscCall(DMStagGetLocationSlot(dm,ELEMENT,0,&idx)); 
      xxMMS[j][i][idx] = get_p(coordx[i][icenter],coordz[j][icenter],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);

      // ux
      PetscCall(DMStagGetLocationSlot(dm,LEFT,0,&idx)); 
      xxMMS[j][i][idx] = get_ux(coordx[i][iprev],coordz[j][icenter],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);

      if (i == Nx-1) {
        PetscCall(DMStagGetLocationSlot(dm,RIGHT,0,&idx); 
        xxMMS[j][i][idx] = get_ux(coordx[i][inext],coordz[j][icenter],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R));
      }
      
      // uz
      PetscCall(DMStagGetLocationSlot(dm,DOWN,0,&idx)); 
      xxMMS[j][i][idx] = get_uz(coordx[i][icenter],coordz[j][iprev],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);

      if (j == Nz-1) {
        PetscCall(DMStagGetLocationSlot(dm,UP,0,&idx)); 
        xxMMS[j][i][idx] = get_uz(coordx[i][icenter],coordz[j][inext],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
      }

      // strain rates - center
      exxc    = get_exx(coordx[i][icenter],coordz[j][icenter],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
      ezzc    = get_ezz(coordx[i][icenter],coordz[j][icenter],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
      exzc    = get_exz(coordx[i][icenter],coordz[j][icenter],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);

      PetscCall(DMStagGetLocationSlot(dmeps,ELEMENT,0,&idx));  xxeps[j][i][idx] = exxc; 
      PetscCall(DMStagGetLocationSlot(dmeps,ELEMENT,1,&idx));  xxeps[j][i][idx] = ezzc;
      PetscCall(DMStagGetLocationSlot(dmeps,ELEMENT,2,&idx));  xxeps[j][i][idx] = exzc;
      PetscCall(DMStagGetLocationSlot(dmeps,ELEMENT,3,&idx));  xxeps[j][i][idx] = 0.0;

      // strain rates - corner
      exxn[0] = get_exx(coordx[i][iprev  ],coordz[j][iprev  ],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
      exxn[1] = get_exx(coordx[i][inext  ],coordz[j][iprev  ],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
      exxn[2] = get_exx(coordx[i][iprev  ],coordz[j][inext  ],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
      exxn[3] = get_exx(coordx[i][inext  ],coordz[j][inext  ],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);

      ezzn[0] = get_ezz(coordx[i][iprev  ],coordz[j][iprev  ],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
      ezzn[1] = get_ezz(coordx[i][inext  ],coordz[j][iprev  ],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
      ezzn[2] = get_ezz(coordx[i][iprev  ],coordz[j][inext  ],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
      ezzn[3] = get_ezz(coordx[i][inext  ],coordz[j][inext  ],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);

      exzn[0] = get_exz(coordx[i][iprev  ],coordz[j][iprev  ],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
      exzn[1] = get_exz(coordx[i][inext  ],coordz[j][iprev  ],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
      exzn[2] = get_exz(coordx[i][iprev  ],coordz[j][inext  ],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
      exzn[3] = get_exz(coordx[i][inext  ],coordz[j][inext  ],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);

      PetscCall(DMStagGetLocationSlot(dmeps,DOWN_LEFT,0,&idx));  xxeps[j][i][idx] = exxn[0];
      PetscCall(DMStagGetLocationSlot(dmeps,DOWN_LEFT,1,&idx));  xxeps[j][i][idx] = ezzn[0];
      PetscCall(DMStagGetLocationSlot(dmeps,DOWN_LEFT,2,&idx));  xxeps[j][i][idx] = exzn[0];
      PetscCall(DMStagGetLocationSlot(dmeps,DOWN_LEFT,3,&idx));  xxeps[j][i][idx] = 0.0;

      PetscCall(DMStagGetLocationSlot(dmeps,DOWN_RIGHT,0,&idx));  xxeps[j][i][idx] = exxn[1];
      PetscCall(DMStagGetLocationSlot(dmeps,DOWN_RIGHT,1,&idx));  xxeps[j][i][idx] = ezzn[1];
      PetscCall(DMStagGetLocationSlot(dmeps,DOWN_RIGHT,2,&idx));  xxeps[j][i][idx] = exzn[1];
      PetscCall(DMStagGetLocationSlot(dmeps,DOWN_RIGHT,3,&idx));  xxeps[j][i][idx] = 0.0;

      PetscCall(DMStagGetLocationSlot(dmeps,UP_LEFT,0,&idx));  xxeps[j][i][idx] = exxn[2];
      PetscCall(DMStagGetLocationSlot(dmeps,UP_LEFT,1,&idx));  xxeps[j][i][idx] = ezzn[2];
      PetscCall(DMStagGetLocationSlot(dmeps,UP_LEFT,2,&idx));  xxeps[j][i][idx] = exzn[2];
      PetscCall(DMStagGetLocationSlot(dmeps,UP_LEFT,3,&idx));  xxeps[j][i][idx] = 0.0;

      PetscCall(DMStagGetLocationSlot(dmeps,UP_RIGHT,0,&idx));  xxeps[j][i][idx] = exxn[3]; 
      PetscCall(DMStagGetLocationSlot(dmeps,UP_RIGHT,1,&idx));  xxeps[j][i][idx] = ezzn[3]; 
      PetscCall(DMStagGetLocationSlot(dmeps,UP_RIGHT,2,&idx));  xxeps[j][i][idx] = exzn[3];
      PetscCall(DMStagGetLocationSlot(dmeps,UP_RIGHT,3,&idx));  xxeps[j][i][idx] = 0.0;

      // right-hand side
      // pressure
      PetscCall(DMStagGetLocationSlot(dm,ELEMENT,0,&idx)); 
      if (test==1) xxrhs[j][i][idx] = get_fp_stokes(coordx[i][icenter],coordz[j][icenter],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
      if (test==2) xxrhs[j][i][idx] = get_fp_stokesdarcy(coordx[i][icenter],coordz[j][icenter],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);

      // ux
      PetscCall(DMStagGetLocationSlot(dm,LEFT,0,&idx)); 
      if (test==1) xxrhs[j][i][idx] = get_fux_stokes(coordx[i][iprev],coordz[j][icenter],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
      if (test==2) xxrhs[j][i][idx] = get_fux_stokesdarcy(coordx[i][iprev],coordz[j][icenter],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);

      if (i == Nx-1) {
        PetscCall(DMStagGetLocationSlot(dm,RIGHT,0,&idx)); 
        if (test==1) xxrhs[j][i][idx] = get_fux_stokes(coordx[i][inext],coordz[j][icenter],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
        if (test==2) xxrhs[j][i][idx] = get_fux_stokesdarcy(coordx[i][inext],coordz[j][icenter],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
      }
      
      // uz
      PetscCall(DMStagGetLocationSlot(dm,DOWN,0,&idx)); 
      if (test==1) xxrhs[j][i][idx] = get_fuz_stokes(coordx[i][icenter],coordz[j][iprev],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
      if (test==2) xxrhs[j][i][idx] = get_fuz_stokesdarcy(coordx[i][icenter],coordz[j][iprev],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);

      if (j == Nz-1) {
        PetscCall(DMStagGetLocationSlot(dm,UP,0,&idx)); 
        if (test==1) xxrhs[j][i][idx] = get_fuz_stokes(coordx[i][icenter],coordz[j][inext],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
        if (test==2) xxrhs[j][i][idx] = get_fuz_stokesdarcy(coordx[i][icenter],coordz[j][inext],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
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

  PetscCall(DMStagVecRestoreArray(dmeps,xepslocal,&xxeps)); 
  PetscCall(DMLocalToGlobalBegin(dmeps,xepslocal,INSERT_VALUES,xeps)); 
  PetscCall(DMLocalToGlobalEnd  (dmeps,xepslocal,INSERT_VALUES,xeps)); 
  PetscCall(VecDestroy(&xepslocal)); 

  PetscCall(DMStagVecRestoreArray(dm,xrhslocal,&xxrhs)); 
  PetscCall(DMLocalToGlobalBegin(dm,xrhslocal,INSERT_VALUES,xrhs)); 
  PetscCall(DMLocalToGlobalEnd  (dm,xrhslocal,INSERT_VALUES,xrhs)); 
  PetscCall(VecDestroy(&xrhslocal)); 

  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_mms_solution",usr->par->fdir_out));
  PetscCall(DMStagViewBinaryPython(dm,xMMS,fout));
  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/out_mms_strain",usr->par->fdir_out));
  PetscCall(DMStagViewBinaryPython(dmeps,xeps,fout));

  if (test==1) { PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s_rhs_stokes",usr->par->fdir_out,usr->par->fname_out)); }
  if (test==2) { PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s/%s_rhs_stokesdarcy",usr->par->fdir_out,usr->par->fname_out)); }
  PetscCall(DMStagViewBinaryPython(dm,xrhs,fout));

  PetscCall(VecDestroy(&xrhs)); 

  // Assign pointers
  *_xMMS    = xMMS;
  *_xepsMMS = xeps;
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// UpdateStrainRates
// ---------------------------------------
PetscErrorCode UpdateStrainRates(DM dm, Vec x, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  DM             dmeps;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz, idx;
  PetscInt       iprev, inext, icenter;
  PetscScalar    ***xx;
  PetscScalar    **coordx,**coordz;
  Vec            xeps, xepslocal,xlocal;
  PetscScalar    eta0,eps0,np,R,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat;
  PetscFunctionBeginUser;

  eta0 = usr->par->eta0;
  eps0 = usr->par->eps0;
  np   = usr->par->np;
  R    = usr->par->R;
  phi_0= usr->par->phi_0;
  phi_s= usr->par->phi_s;
  p_s  = usr->par->p_s;
  psi_s= usr->par->psi_s;
  U_s  = usr->par->U_s;
  m    = usr->par->m;
  n    = usr->par->n;
  k_hat= usr->par->k_hat;

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
      PetscScalar    epsIIc, exxc, ezzc, exzc, epsIIn[4], exxn[4], ezzn[4], exzn[4], xp[4], zp[4];
      PetscInt       ii;

      // Strain rates: center
      pointC.i = i; pointC.j = j; pointC.loc = ELEMENT; pointC.c = 0;
      PetscCall(DMStagGetPointStrainRates(dm,xlocal,1,&pointC,&epsIIc,&exxc,&ezzc,&exzc)); 

      if ((i==0) || (i==Nx-1) || (j==0) || (j==Nz-1)) {
        exzc = get_exz(coordx[i][icenter],coordz[j][icenter],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
        exxc = get_exx(coordx[i][icenter],coordz[j][icenter],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
        ezzc = get_ezz(coordx[i][icenter],coordz[j][icenter],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
        epsIIc = PetscPowScalar(0.5*(exxc*exxc + ezzc*ezzc + 2.0*exzc*exzc),0.5);
      }

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

      if ((i==0) || (i==Nx-1) || (j==0) || (j==Nz-1)) {
        xp[0] = coordx[i][iprev]; zp[0] = coordz[j][iprev];
        xp[1] = coordx[i][inext]; zp[1] = coordz[j][iprev];
        xp[2] = coordx[i][iprev]; zp[2] = coordz[j][inext];
        xp[3] = coordx[i][inext]; zp[3] = coordz[j][inext];

        for (ii = 0; ii < 4; ii++) {
          exxn[ii] = get_exx(xp[ii],zp[ii],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
          ezzn[ii] = get_ezz(xp[ii],zp[ii],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
          exzn[ii] = get_exz(xp[ii],zp[ii],eta0,eps0,np,phi_0,phi_s,p_s,psi_s,U_s,m,n,k_hat,R);
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
// ComputeErrorNorms
// ---------------------------------------
PetscErrorCode ComputeErrorNorms(DM dm,Vec x,Vec xMMS,Vec xepsMMS,PetscInt test, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz;
  PetscScalar    xx[5], xa[5], dx, dz, dv;
  PetscScalar    nrm2p, nrm2v, nrm2vx, nrm2vz, nrm2ec[3], nrm2en[3], sum_err[9], gsum_err[9], sum_mms[9], gsum_mms[9];
  Vec            xlocal, xalocal,xepsMMSlocal, xepslocal;
  PetscInt       iprev, inext, icenter, ii;
  PetscScalar    **coordx,**coordz;
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

  PetscCall(DMGetLocalVector(usr->dmeps, &xepslocal)); 
  PetscCall(DMGlobalToLocal (usr->dmeps, usr->xeps, INSERT_VALUES, xepslocal)); 

  PetscCall(DMGetLocalVector(usr->dmeps, &xepsMMSlocal)); 
  PetscCall(DMGlobalToLocal (usr->dmeps, xepsMMS, INSERT_VALUES, xepsMMSlocal)); 

  // Get dm coordinates array
  PetscCall(DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter)); 
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm,LEFT,&iprev));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm,RIGHT,&inext)); 

  // Initialize norms
  sum_err[0] = 0.0; sum_err[1] = 0.0; sum_err[2] = 0.0; 
  sum_err[3] = 0.0; sum_err[4] = 0.0; sum_err[5] = 0.0;
  sum_err[6] = 0.0; sum_err[7] = 0.0; sum_err[8] = 0.0;

  sum_mms[0] = 0.0; sum_mms[1] = 0.0; sum_mms[2] = 0.0; 
  sum_mms[3] = 0.0; sum_mms[4] = 0.0; sum_mms[5] = 0.0;
  sum_mms[6] = 0.0; sum_mms[7] = 0.0; sum_mms[8] = 0.0;

  dx = 1.0/Nx;
  dz = 1.0/Nz;
  dv = dx*dz;

  // Loop over local domain to calculate ELEMENT errors
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      
      PetscScalar    ve[4], pe, v[4], p, eps[4],epsa[4];
      PetscScalar    exxc, ezzc, exzc, exxn[4], ezzn[4], exzn[4];
      PetscScalar    exxac, ezzac, exzac, exxan[4], ezzan[4], exzan[4];
      PetscScalar    exxc_err, ezzc_err, exzc_err, exxn_err[4], ezzn_err[4], exzn_err[4];
      PetscScalar    exxc_mms, ezzc_mms, exzc_mms, exxn_mms[4], ezzn_mms[4], exzn_mms[4];
      DMStagStencil  point[5];
      
      // Get stencil values
      point[0].i = i; point[0].j = j; point[0].loc = LEFT;    point[0].c = 0; // Vx
      point[1].i = i; point[1].j = j; point[1].loc = RIGHT;   point[1].c = 0; // Vx
      point[2].i = i; point[2].j = j; point[2].loc = DOWN;    point[2].c = 0; // Vz
      point[3].i = i; point[3].j = j; point[3].loc = UP;      point[3].c = 0; // Vz
      point[4].i = i; point[4].j = j; point[4].loc = ELEMENT; point[4].c = 0; // P
      
      // Get numerical and MMS solutions
      PetscCall(DMStagVecGetValuesStencil(dm, xlocal, 5, point, xx)); 
      PetscCall(DMStagVecGetValuesStencil(dm, xalocal,5, point, xa)); 

      // Strain rates: center
      point[0].i = i; point[0].j = j; point[0].loc = ELEMENT; point[0].c = 0;
      point[1].i = i; point[1].j = j; point[1].loc = ELEMENT; point[1].c = 1;
      point[2].i = i; point[2].j = j; point[2].loc = ELEMENT; point[2].c = 2;

      PetscCall(DMStagVecGetValuesStencil(usr->dmeps, xepslocal,   3, point,eps )); 
      PetscCall(DMStagVecGetValuesStencil(usr->dmeps, xepsMMSlocal,3, point,epsa)); 
      exxc  = eps[0];  ezzc  = eps[1];  exzc  = eps[2];
      exxac = epsa[0]; ezzac = epsa[1]; exzac = epsa[2];

      // Strain rates: corner
      point[0].i = i; point[0].j = j; point[0].loc = DOWN_LEFT;  point[0].c = 0;
      point[1].i = i; point[1].j = j; point[1].loc = DOWN_RIGHT; point[1].c = 0;
      point[2].i = i; point[2].j = j; point[2].loc = UP_LEFT;    point[2].c = 0;
      point[3].i = i; point[3].j = j; point[3].loc = UP_RIGHT;   point[3].c = 0;

      PetscCall(DMStagVecGetValuesStencil(usr->dmeps, xepslocal,   4, point,eps )); 
      PetscCall(DMStagVecGetValuesStencil(usr->dmeps, xepsMMSlocal,4, point,epsa)); 
      for (ii = 0; ii < 4; ii++) { exxn[ii] = eps[ii]; exxan[ii] = epsa[ii]; }

      for (ii = 0; ii < 4; ii++) point[ii].c = 1;
      PetscCall(DMStagVecGetValuesStencil(usr->dmeps, xepslocal,   4, point,eps )); 
      PetscCall(DMStagVecGetValuesStencil(usr->dmeps, xepsMMSlocal,4, point,epsa)); 
      for (ii = 0; ii < 4; ii++) { ezzn[ii] = eps[ii]; ezzan[ii] = epsa[ii]; }

      for (ii = 0; ii < 4; ii++) point[ii].c = 2;
      PetscCall(DMStagVecGetValuesStencil(usr->dmeps, xepslocal,   4, point,eps )); 
      PetscCall(DMStagVecGetValuesStencil(usr->dmeps, xepsMMSlocal,4, point,epsa)); 
      for (ii = 0; ii < 4; ii++) { exzn[ii] = eps[ii]; exzan[ii] = epsa[ii]; }

      // Error vectors - squared
      ve[0] = (xx[0]-xa[0])*(xx[0]-xa[0]); // Left
      ve[1] = (xx[1]-xa[1])*(xx[1]-xa[1]); // Right
      ve[2] = (xx[2]-xa[2])*(xx[2]-xa[2]); // Down
      ve[3] = (xx[3]-xa[3])*(xx[3]-xa[3]); // Up
      pe    = (xx[4]-xa[4])*(xx[4]-xa[4]); // elem

      exxc_err    = (exxc-exxac)*(exxc-exxac);
      ezzc_err    = (ezzc-ezzac)*(ezzc-ezzac);
      exzc_err    = (exzc-exzac)*(exzc-exzac);

      exxn_err[0] = (exxn[0]-exxan[0])*(exxn[0]-exxan[0]);
      exxn_err[1] = (exxn[1]-exxan[1])*(exxn[1]-exxan[1]);
      exxn_err[2] = (exxn[2]-exxan[2])*(exxn[2]-exxan[2]);
      exxn_err[3] = (exxn[3]-exxan[3])*(exxn[3]-exxan[3]);

      ezzn_err[0] = (ezzn[0]-ezzan[0])*(ezzn[0]-ezzan[0]);
      ezzn_err[1] = (ezzn[1]-ezzan[1])*(ezzn[1]-ezzan[1]);
      ezzn_err[2] = (ezzn[2]-ezzan[2])*(ezzn[2]-ezzan[2]);
      ezzn_err[3] = (ezzn[3]-ezzan[3])*(ezzn[3]-ezzan[3]);

      exzn_err[0] = (exzn[0]-exzan[0])*(exzn[0]-exzan[0]);
      exzn_err[1] = (exzn[1]-exzan[1])*(exzn[1]-exzan[1]);
      exzn_err[2] = (exzn[2]-exzan[2])*(exzn[2]-exzan[2]);
      exzn_err[3] = (exzn[3]-exzan[3])*(exzn[3]-exzan[3]);

      // MMS values - squared
      v[0] = xa[0]*xa[0]; // Left
      v[1] = xa[1]*xa[1]; // Right
      v[2] = xa[2]*xa[2]; // Down
      v[3] = xa[3]*xa[3]; // Up
      p    = xa[4]*xa[4]; // elem

      exxc_mms    = exxac*exxac;
      ezzc_mms    = ezzac*ezzac;
      exzc_mms    = exzac*exzac;

      exxn_mms[0] = exxan[0]*exxan[0];
      exxn_mms[1] = exxan[1]*exxan[1];
      exxn_mms[2] = exxan[2]*exxan[2];
      exxn_mms[3] = exxan[3]*exxan[3];

      ezzn_mms[0] = ezzan[0]*ezzan[0];
      ezzn_mms[1] = ezzan[1]*ezzan[1];
      ezzn_mms[2] = ezzan[2]*ezzan[2];
      ezzn_mms[3] = ezzan[3]*ezzan[3];

      exzn_mms[0] = exzan[0]*exzan[0];
      exzn_mms[1] = exzan[1]*exzan[1];
      exzn_mms[2] = exzan[2]*exzan[2];
      exzn_mms[3] = exzan[3]*exzan[3];

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

      // strain rates - center
      sum_err[3] += exxc_err*dv;
      sum_mms[3] += exxc_mms*dv;

      sum_err[4] += ezzc_err*dv;
      sum_mms[4] += ezzc_mms*dv;

      sum_err[5] += exzc_err*dv;
      sum_mms[5] += exzc_mms*dv;

      // strain rates - corner
      if (j == 0) { 
        if (i == 0) { 
          sum_err[6] += exxn_err[0]*dv*0.25; sum_err[6] += exxn_err[1]*dv*0.5; sum_err[6] += exxn_err[2]*dv*0.5; sum_err[6] += exxn_err[3]*dv; 
          sum_err[7] += ezzn_err[0]*dv*0.25; sum_err[7] += ezzn_err[1]*dv*0.5; sum_err[7] += ezzn_err[2]*dv*0.5; sum_err[7] += ezzn_err[3]*dv; 
          sum_err[8] += exzn_err[0]*dv*0.25; sum_err[8] += exzn_err[1]*dv*0.5; sum_err[8] += exzn_err[2]*dv*0.5; sum_err[8] += exzn_err[3]*dv; 

          sum_mms[6] += exxn_mms[0]*dv*0.25; sum_mms[6] += exxn_mms[1]*dv*0.5; sum_mms[6] += exxn_mms[2]*dv*0.5; sum_mms[6] += exxn_mms[3]*dv; 
          sum_mms[7] += ezzn_mms[0]*dv*0.25; sum_mms[7] += ezzn_mms[1]*dv*0.5; sum_mms[7] += ezzn_mms[2]*dv*0.5; sum_mms[7] += ezzn_mms[3]*dv; 
          sum_mms[8] += exzn_mms[0]*dv*0.25; sum_mms[8] += exzn_mms[1]*dv*0.5; sum_mms[8] += exzn_mms[2]*dv*0.5; sum_mms[8] += exzn_mms[3]*dv; 
        } else if (i == Nx-1) { 
          sum_err[6] += exxn_err[1]*dv*0.25; sum_err[6] += exxn_err[3]*dv*0.5; 
          sum_err[7] += ezzn_err[1]*dv*0.25; sum_err[7] += ezzn_err[3]*dv*0.5; 
          sum_err[8] += exzn_err[1]*dv*0.25; sum_err[8] += exzn_err[3]*dv*0.5; 

          sum_mms[6] += exxn_mms[1]*dv*0.25; sum_mms[6] += exxn_mms[3]*dv*0.5; 
          sum_mms[7] += ezzn_mms[1]*dv*0.25; sum_mms[7] += ezzn_mms[3]*dv*0.5; 
          sum_mms[8] += exzn_mms[1]*dv*0.25; sum_mms[8] += exzn_mms[3]*dv*0.5; 
        } else { 
          sum_err[6] += exxn_err[1]*dv*0.5;  sum_err[6] += exxn_err[3]*dv; 
          sum_err[7] += ezzn_err[1]*dv*0.5;  sum_err[7] += ezzn_err[3]*dv; 
          sum_err[8] += exzn_err[1]*dv*0.5;  sum_err[8] += exzn_err[3]*dv;

          sum_mms[6] += exxn_mms[1]*dv*0.5;  sum_mms[6] += exxn_mms[3]*dv; 
          sum_mms[7] += ezzn_mms[1]*dv*0.5;  sum_mms[7] += ezzn_mms[3]*dv; 
          sum_mms[8] += exzn_mms[1]*dv*0.5;  sum_mms[8] += exzn_mms[3]*dv;
        }
      } else if (j == Nz-1) { 
        if (i == 0) { 
          sum_err[6] += exxn_err[2]*dv*0.25; sum_err[6] += exxn_err[3]*dv*0.5; 
          sum_err[7] += ezzn_err[2]*dv*0.25; sum_err[7] += ezzn_err[3]*dv*0.5; 
          sum_err[8] += exzn_err[2]*dv*0.25; sum_err[8] += exzn_err[3]*dv*0.5; 

          sum_mms[6] += exxn_mms[2]*dv*0.25; sum_mms[6] += exxn_mms[3]*dv*0.5; 
          sum_mms[7] += ezzn_mms[2]*dv*0.25; sum_mms[7] += ezzn_mms[3]*dv*0.5; 
          sum_mms[8] += exzn_mms[2]*dv*0.25; sum_mms[8] += exzn_mms[3]*dv*0.5; 
        } else if (i == Nx-1) { 
          sum_err[6] += exxn_err[3]*dv*0.25; 
          sum_err[7] += ezzn_err[3]*dv*0.25; 
          sum_err[8] += exzn_err[3]*dv*0.25; 

          sum_mms[6] += exxn_mms[3]*dv*0.25; 
          sum_mms[7] += ezzn_mms[3]*dv*0.25; 
          sum_mms[8] += exzn_mms[3]*dv*0.25; 
        } else {
          sum_err[6] += exxn_err[3]*dv*0.5; 
          sum_err[7] += ezzn_err[3]*dv*0.5; 
          sum_err[8] += exzn_err[3]*dv*0.5; 

          sum_mms[6] += exxn_mms[3]*dv*0.5; 
          sum_mms[7] += ezzn_mms[3]*dv*0.5; 
          sum_mms[8] += exzn_mms[3]*dv*0.5; 
        }
      } else {
        if (i == 0) { 
          sum_err[6] += exxn_err[2]*dv*0.5; sum_err[6] += exxn_err[3]*dv; 
          sum_err[7] += ezzn_err[2]*dv*0.5; sum_err[7] += ezzn_err[3]*dv; 
          sum_err[8] += exzn_err[2]*dv*0.5; sum_err[8] += exzn_err[3]*dv; 

          sum_mms[6] += exxn_mms[2]*dv*0.5; sum_mms[6] += exxn_mms[3]*dv; 
          sum_mms[7] += ezzn_mms[2]*dv*0.5; sum_mms[7] += ezzn_mms[3]*dv; 
          sum_mms[8] += exzn_mms[2]*dv*0.5; sum_mms[8] += exzn_mms[3]*dv; 
        } else if (i == Nx-1) { 
          sum_err[6] += exxn_err[3]*dv*0.5; 
          sum_err[7] += ezzn_err[3]*dv*0.5; 
          sum_err[8] += exzn_err[3]*dv*0.5; 

          sum_mms[6] += exxn_mms[3]*dv*0.5; 
          sum_mms[7] += ezzn_mms[3]*dv*0.5; 
          sum_mms[8] += exzn_mms[3]*dv*0.5; 
        } else {
          sum_err[6] += exxn_err[3]*dv;
          sum_err[7] += ezzn_err[3]*dv;
          sum_err[8] += exzn_err[3]*dv;

          sum_mms[6] += exxn_mms[3]*dv;
          sum_mms[7] += ezzn_mms[3]*dv;
          sum_mms[8] += exzn_mms[3]*dv;
        }
      }
    }
  }

  // Collect data 
  PetscCall(MPI_Allreduce(&sum_err, &gsum_err, 9, MPI_DOUBLE, MPI_SUM, comm)); 
  PetscCall(MPI_Allreduce(&sum_mms, &gsum_mms, 9, MPI_DOUBLE, MPI_SUM, comm)); 

  // L2 error norm = sqrt(gsum_err/gsum_mms)
  nrm2vx = PetscSqrtScalar(gsum_err[0]/gsum_mms[0]);
  nrm2vz = PetscSqrtScalar(gsum_err[1]/gsum_mms[1]);
  nrm2p  = PetscSqrtScalar(gsum_err[2]/gsum_mms[2]);
  nrm2v  = PetscSqrtScalar((gsum_err[0]+gsum_err[1])/(gsum_mms[0]+gsum_mms[1]));

  nrm2ec[0] = PetscSqrtScalar(gsum_err[3]/gsum_mms[3]);
  nrm2ec[1] = PetscSqrtScalar(gsum_err[4]/gsum_mms[4]);
  nrm2ec[2] = PetscSqrtScalar(gsum_err[5]/gsum_mms[5]);

  nrm2en[0] = PetscSqrtScalar(gsum_err[6]/gsum_mms[6]);
  nrm2en[1] = PetscSqrtScalar(gsum_err[7]/gsum_mms[7]);
  nrm2en[2] = PetscSqrtScalar(gsum_err[8]/gsum_mms[8]);

  // Restore arrays and vectors
  PetscCall(DMRestoreLocalVector(dm, &xlocal )); 
  PetscCall(DMRestoreLocalVector(dm, &xalocal )); 
  PetscCall(DMRestoreLocalVector(usr->dmeps, &xepslocal )); 
  PetscCall(DMRestoreLocalVector(usr->dmeps, &xepsMMSlocal )); 
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL));

  // Print information
  PetscPrintf(comm,"# --------------------------------------- #\n");
  PetscPrintf(comm,"# NORMS: \n");
  PetscPrintf(comm,"# Velocity test%d: norm2 = %1.12e norm2x = %1.12e norm2z = %1.12e \n",test,nrm2v,nrm2vx,nrm2vz);
  PetscPrintf(comm,"# Pressure test%d: norm2 = %1.12e\n",test,nrm2p);
  PetscPrintf(comm,"# Strain rates CENTER test%d: norm2_exx = %1.12e norm2_ezz = %1.12e norm2_exz = %1.12e \n",test,nrm2ec[0],nrm2ec[1],nrm2ec[2]);
  PetscPrintf(comm,"# Strain rates CORNER test%d: norm2_exx = %1.12e norm2_ezz = %1.12e norm2_exz = %1.12e \n",test,nrm2en[0],nrm2en[1],nrm2en[2]);
  PetscPrintf(comm,"# Grid info test%d: hx = %1.12e hz = %1.12e\n",test,dx,dz);

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

  // Numerical solution using the FD pde object - Stokes
  usr->par->test = 1;
  PetscCall(PetscTime(&start_time)); 
  PetscCall(Numerical_solution(usr)); 
  PetscCall(PetscTime(&end_time)); 
  PetscPrintf(PETSC_COMM_WORLD,"# Runtime Stokes: %g (sec) \n", end_time - start_time);
  PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");

  // Numerical solution using the FD pde object - StokesDarcy
  usr->par->test = 2;
  PetscCall(PetscTime(&start_time)); 
  PetscCall(Numerical_solution(usr)); 
  PetscCall(PetscTime(&end_time)); 
  PetscPrintf(PETSC_COMM_WORLD,"# Runtime StokesDarcy: %g (sec) \n", end_time - start_time);
  PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");

  // Destroy objects
  PetscCall(PetscBagDestroy(&usr->bag)); 
  PetscCall(PetscFree(usr));

  // Finalize main
  PetscCall(PetscFinalize());
  return 0;
}
