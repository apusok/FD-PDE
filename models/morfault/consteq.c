#include "morfault.h"

// ---------------------------------------
// HalfSpaceCoolingTemp
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "HalfSpaceCoolingTemp"
PetscScalar HalfSpaceCoolingTemp(PetscScalar Tm, PetscScalar T0, PetscScalar z, PetscScalar kappa, PetscScalar t, PetscScalar factor) 
{ // factor = 2.0
  return T0 + (Tm-T0)*erf(z/(factor*sqrt(kappa*t)));
}

// ---------------------------------------
// LithostaticPressure
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "LithostaticPressure"
PetscScalar LithostaticPressure(PetscScalar rho, PetscScalar scal_rho, PetscScalar z)
{
  return -rho*z/scal_rho;
}

// ---------------------------------------
// Material density
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "Density"
PetscScalar Density(PetscScalar rho0, PetscInt func)
{
  if (func==0) return rho0;
  return 0.0;
}

// ---------------------------------------
// Permeability
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "Permeability"
PetscScalar Permeability(PetscScalar phi, PetscScalar n) 
{ 
  return pow(phi,n);
}

// ---------------------------------------
// Mixture
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "Mixture"
PetscScalar Mixture(PetscScalar as, PetscScalar af, PetscScalar phi) 
{ 
  return af*phi + as*(1.0-phi);
}


// ---------------------------------------
// FluidVelocity
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FluidVelocity"
PetscScalar FluidVelocity(PetscScalar A, PetscScalar Kphi, PetscScalar vs, PetscScalar phi, PetscScalar gradP, PetscScalar gradPlith, PetscScalar Bf) 
{ 
  if (Kphi == 0.0) return 0.0;
  else             return vs-A*Kphi/phi*(gradP+gradPlith-Bf);
}

// ---------------------------------------
// ShearViscosity
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "ShearViscosity"
PetscScalar ShearViscosity(PetscScalar eta0, PetscScalar T, PetscScalar phi, PetscScalar EoR, PetscScalar Teta0, PetscScalar lambda, PetscInt func) 
{ 
  PetscScalar eta, eta_arr;
  // constant 
  if (func == 0) { eta = eta0; } 

  // phi-dependent, with harmonic averaging
  if (func == 1) { eta = eta0*exp(-lambda*phi); } 

  // T,phi-dep, with harmonic averaging
  if (func == 2) { 
    eta_arr = eta0*ArrheniusTerm_Viscosity(T,EoR,Teta0)*exp(-lambda*phi);
    eta = eta_arr;
    // eta = 1.0/(1.0/eta_arr + 1.0/eta_max) + eta_min;
  }
  return eta;
}

// ---------------------------------------
PetscScalar ArrheniusTerm_Viscosity(PetscScalar T, PetscScalar EoR, PetscScalar Teta0) 
{ return exp(EoR*(1.0/T-1.0/Teta0)); }

// ---------------------------------------
// CompactionViscosity
// ---------------------------------------
PetscScalar CompactionViscosity(PetscScalar zeta0, PetscScalar T, PetscScalar phi, PetscScalar EoR, PetscScalar Teta0, PetscScalar phi_min, PetscScalar zetaExp, PetscInt func) 
{ 
  PetscScalar zeta, zeta_arr;
  // constant
  if (func == 0) { zeta = zeta0; }  

  // phi-dependent, with 1/(phi+phi_min)
  if (func == 1) { zeta = zeta0*pow(phi+phi_min,zetaExp); }

  // T,phi-dep, with harmonic averaging and with 1/(phi+phi_min)
  if (func == 2) { 
    zeta_arr = zeta0*pow(phi+phi_min,zetaExp)*ArrheniusTerm_Viscosity(T,EoR,Teta0);
    zeta = zeta_arr;
    // zeta = 1.0/(1.0/zeta_arr + 1.0/eta_max) + eta_min;
  }
  return zeta;
}

// ---------------------------------------
// Poro-elastic modulus
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "PoroElasticModulus"
PetscScalar PoroElasticModulus(PetscScalar Z0, PetscScalar phi) 
{ 
  if (phi == 0.0) return Z0;
  else            return Z0*PetscPowScalar(phi,-0.5);
}

// ---------------------------------------
// TensorSecondInvariant
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "TensorSecondInvariant"
PetscScalar TensorSecondInvariant(PetscScalar axx, PetscScalar azz, PetscScalar axz) 
{ 
  return PetscPowScalar(0.5*(axx*axx + azz*azz) + axz*axz,0.5);
}

// ---------------------------------------
// Harmonic averaging
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "ViscosityHarmonicAvg"
PetscScalar ViscosityHarmonicAvg(PetscScalar eta, PetscScalar eta_min, PetscScalar eta_max) 
{ 
  return 1.0/(1.0/eta + 1.0/eta_max) + eta_min;
}

// ---------------------------------------
static PetscErrorCode tau_a(PetscScalar tauIIp, PetscScalar DPt, PetscScalar eta_ve, PetscScalar zeta_ve, PetscScalar C, PetscScalar sigmat, PetscScalar aP, PetscScalar theta, PetscScalar cdl, PetscScalar eta_vp, PetscScalar a[5])
{
  PetscFunctionBegin;
  a[0] = C*PetscCosScalar(theta) - sigmat*PetscSinScalar(theta);
  a[1] = C*PetscCosScalar(theta) + aP*PetscSinScalar(theta);
  a[2] = (cdl*zeta_ve*PetscPowScalar(PetscSinScalar(theta),2) + eta_vp)/eta_ve;
  a[3] = a[1] + DPt * PetscSinScalar(theta);
  a[4] = a[0]*(a[2] + 1.0) - a[3] - a[2]*tauIIp;
  PetscFunctionReturn(0);
}

static PetscScalar ftau(PetscScalar a[5], PetscScalar tauIIt, PetscScalar x)
{ PetscScalar y1, y2, aa;
  aa = PetscPowScalar(x*x + a[0]*a[0], 0.5);
  y1 = a[2]*tauIIt*aa/x;
  y2 = (a[2]+1.0)*aa - a[3];
  return y1 - y2;
}

static PetscScalar dftau(PetscScalar a[5], PetscScalar tauIIt, PetscScalar x)
{ PetscScalar aa, result;
  aa = PetscPowScalar(x*x + a[0]*a[0], 0.5);
  result = a[2]*tauIIt*(1/aa - aa/(x*x)) -(a[2]+1.0)*x/aa;
  return result;
}

static PetscScalar lamtau(PetscScalar a[5], PetscScalar x, PetscScalar DPt, PetscScalar sint, PetscScalar cdl, PetscScalar zeta_ve, PetscScalar eta_vp)
{ PetscScalar aa, result;
  aa = PetscPowScalar(x*x + a[0]*a[0], 0.5);
  result = (aa - a[1] - DPt*sint)/(cdl*zeta_ve*sint*sint + eta_vp);
  return result;
}

// ---------------------------------------
static PetscScalar ResF(PetscScalar xsol[3], PetscScalar *a,PetscScalar theta,PetscScalar eta_vp)
{ 
  PetscScalar tauII, DP, dotlam, F;
  tauII = xsol[0];
  DP    = xsol[1];
  dotlam= xsol[2];

  F = PetscPowScalar(tauII*tauII + a[0]*a[0], 0.5) - (a[1] + DP*PetscSinScalar(theta)) - eta_vp*dotlam;
  return F;
}

// ---------------------------------------
PetscScalar AlphaP(PetscScalar phi, PetscScalar phia) 
{ return PetscExpScalar(-phi/phia); }

// ---------------------------------------
//VEVP_hyper_tau: find the solution of tauII
// - a[5], five constants in the local Newton iterative, the output from tau_a
// - tauIIt - trial tauII (VE) 
// - tf_tol, function tolerance of the local Newton iterative
// - Nmax,   the maximum Newton iteratives
//
// Output:
// - tauII, the solution
// ---------------------------------------
static PetscScalar VEVP_hyper_tau(PetscScalar a[5],PetscScalar tauIIt, PetscScalar tf_tol, PetscInt Nmax)
{
  PetscInt          ii=0;
  PetscScalar       xn,xn1, f, df, dx = 1e40;

  // initial guess
  xn = (-a[4]+PetscPowScalar(a[4]*a[4] + 4.0*a[0]*a[2]*(a[2]+1.0)*tauIIt, 0.5 ))*0.5/(a[2]+1.0);
  xn1 = xn;
  f = ftau(a, tauIIt, xn);

  while (PetscAbs(f)> tf_tol && dx/xn > tf_tol && ii < Nmax) {
    df = dftau(a, tauIIt, xn);
    dx = -f/df;
    while (PetscAbs(dx) > PetscAbs(xn)) {dx = 0.5*dx;} // if dx is too large, take half step
    xn1 = xn + dx;

    f = ftau(a, tauIIt, xn1);
    xn = xn1;
    ii += 1;
  }

  if (ii>=Nmax) {
    PetscPrintf(PETSC_COMM_WORLD,"DIVERGENCE: VEVP_hyper_tau, max iteration reaches. ii = %d, Nmax = %d, F = %1.6e, dx = %1.6f, xn1=%1.6f tau_p = %1.6f \n", ii, Nmax, f, dx, xn1, tauIIt);
    PetscPrintf(PETSC_COMM_WORLD,"----------A1 = %1.6f, A2 = %1.6f, A3 = %1.6f, A4 = %1.6f, A5 =%1.6f \n ", a[0], a[1], a[2], a[3], a[4]);
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL, "FORCE STOPPED! VEVP_hyper_tau is divergent");
  }

  return xn;
}

// ---------------------------------------
// Plastic yielding - return tauII, DP and dotlam with/out plastic yielding
// ---------------------------------------
// input: xve[0] = tau_ve, xve[1] = DP_ve, xve[2] = eta_ve, xve[3] = zeta_ve, C, sigmat, theta (rad), aP, phi, usr
// output: xsol[0] = tauII, xsol[1] = DP, xsol[2] = dotlam
// // case 1: F < 0, tauII = tau_ve, deltap = DP_ve, dotlam = 0.0
// // case 2: F > 0 and xve[0]  = 0, analytical solution with tauII =0
// // case 3: F > 0 and xve[0] != 0, numerical solution
#undef __FUNCT__
#define __FUNCT__ "Plastic_LocalSolver"
PetscErrorCode Plastic_LocalSolver(PetscScalar *xve,PetscScalar C, PetscScalar sigmat, PetscScalar theta, PetscScalar aP, PetscScalar phi, void *ctx, PetscScalar xsol[3])
{ 
  UsrData        *usr = (UsrData*)ctx;
  PetscScalar     F,a[5],tauIIt,DPt,zeta_ve,eta_ve,eta_vp, cdl, sint;
  PetscErrorCode  ierr;
  PetscFunctionBegin;

  tauIIt  = xve[0];
  DPt     = xve[1];
  eta_ve  = xve[2];
  zeta_ve = xve[3];
  eta_vp  = usr->nd->eta_K;
  cdl     = 0.0; // should be function of phi
  sint    = PetscSinScalar(theta);

  // trial solution
  xsol[0] = tauIIt; // tauII
  xsol[1] = DPt;    // DP
  xsol[2] = 0.0;    // dotlam
  
  // get coefficients A
  ierr = tau_a(tauIIt,DPt,eta_ve,zeta_ve,C,sigmat,aP,theta,cdl,eta_vp,a); CHKERRQ(ierr);

  // check if it yields
  F = ResF(xsol,a,theta,eta_vp); CHKERRQ(ierr);
  
  if (F>0) { // yield
    if (xsol[0] <= usr->par->tf_tol) { // case 2
      xsol[0] = 0.0; 
      xsol[2] = (a[0]-a[1]-DPt*sint)/(cdl*zeta_ve*sint*sint + eta_vp);
      xsol[1] = DPt + cdl * zeta_ve * sint * xsol[2]; 
    } else { // case 3
      xsol[0] = VEVP_hyper_tau(a, tauIIt, usr->par->tf_tol, usr->par->Nmax); CHKERRQ(ierr);
      xsol[2] = lamtau(a,xsol[0],DPt,sint,cdl,zeta_ve,eta_vp);
      xsol[1] = DPt + cdl * zeta_ve * sint * xsol[2]; 
    }
    //PetscPrintf(PETSC_COMM_WORLD, "TEST, tau = %1.6f, dp = %1.6f, lam = %1.6f\n", xsol[0], xsol[1], xsol[2]);
  }
  PetscFunctionReturn(0);
}

// ---------------------------------------
// Y funstions
// ---------------------------------------
static PetscScalar tau_ini_Y(PetscScalar A1, PetscScalar A2, PetscScalar A3, PetscScalar A4, PetscScalar A5, PetscScalar xp)
{ PetscScalar A6, result;
  A6 = A3 + 1.0;
  result = -A5 + PetscPowScalar(A5*A5 + 4.0*A1*A3*A6*xp , 0.5 );
  result = 0.5*result/A6;
  return(result);
}

static PetscScalar ftau_Y(PetscScalar A1, PetscScalar A2, PetscScalar A3, PetscScalar A4, PetscScalar xp, PetscScalar x)
{ PetscScalar aa, result;
  //aa = PetscPowScalar(x*x+A1*A1, 0.5);
  aa = PetscPowScalar(x*x+A1*A1, 0.5);
  //result = A3*xp*aa/x - (1+A3)*aa + A4;
  result = aa/x - (1/A3+1)*aa + A4/xp;
  return(result);
}

static PetscScalar dftau_Y(PetscScalar A1, PetscScalar A2, PetscScalar A3, PetscScalar A4, PetscScalar xp, PetscScalar x)
{ PetscScalar aa, result;
  aa = PetscPowScalar(x*x+A1*A1, 0.5);
  //result = A3*xp*(1/aa - aa/(x*x)) - (1+A3)*x/aa;
  result = (1/aa - aa/(x*x)) - (1/A3+1)*x/aa;
  return(result);
}

// ---------------------------------------
//VEVP_hyper_tau: find the solution of tauII
// - a[5], five constants in the local Newton iterative, the output from tau_a
// - xp[2], prediction value of tauII = xp[0] and Dp = xp[1]
// - tf_tol, function tolerance of the local Newton iterative
// - Nmax,   the maximum newton iteratives
//
// Output:
// - xsol, the solution
// ---------------------------------------
static PetscScalar VEVP_hyper_tau_Y(PetscScalar a[5], PetscScalar xp[2], PetscScalar tf_tol, PetscInt Nmax)
{
  PetscInt          ii=0;
  PetscScalar       xn,xn1, f, df, dx = 1e40;

  // prepare the initial guess
  xn = tau_ini_Y(a[0], a[1], a[2], a[3], a[4], xp[0]);
  //xn1 = xn;

  // rescale xn and A1 by xp[0], A4 by A3
  xn = xn/xp[0];
  a[0] = a[0]/xp[0];
  a[3] = a[3]/a[2];

  xn1 = xn;

  f = ftau_Y(a[0], a[1], a[2],a[3], xp[0], xn);

//  PetscPrintf(PETSC_COMM_WORLD, "CHECK xn = %1.6f, xn1 = %1.6f, f = %1.6f \n", xn, xn1, f);

  while (PetscAbs(f)> tf_tol && dx/xn > tf_tol && ii < Nmax) {

    df = dftau_Y(a[0], a[1], a[2], a[3], xp[0], xn);
    dx = -f/df;

    while (PetscAbs(dx) > PetscAbs(xn)) {dx = 0.5*dx;} // if dx is too large, take half step

    xn1 = xn + dx;

//    PetscPrintf(PETSC_COMM_WORLD,"VEVP_hyper_tau: Iteration ii = %d, F = %1.6e, dF = %1.6e, xn = %1.6f, dx = %1.6f\n", ii, f, df, xn, dx);

    f = ftau_Y(a[0], a[1], a[2],a[3], xp[0], xn1);
    xn = xn1;
    ii += 1;
  }

  if (ii>=Nmax) {
    PetscPrintf(PETSC_COMM_WORLD,"DIVERGENCE: VEVP_hyper_tau, max iteration reaches. ii = %d, Nmax = %d, F = %1.6e, dx = %1.6f, xn1=%1.6f \n", ii, Nmax, f, dx, xn1);
    PetscPrintf(PETSC_COMM_WORLD,"----------A1 = %1.6f, A2 = %1.6f, A3 = %1.6f, A4 = %1.6f, A5 =%1.6f \n ", a[0], a[1], a[2], a[3], a[4]);
    PetscPrintf(PETSC_COMM_WORLD,"----------tau_p = %1.6f, p_p = %1.6f", xp[0], xp[1]);
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL, "FORCE STOPPED! VEVP_hyper_tau is divergent");
  }
  else {
//    PetscPrintf(PETSC_COMM_WORLD,"Convergence: VEVP_hyper_tau. Total iteration. ii = %d, F = %1.6e, xn1 = %1.6f \n", ii, f, xn1);
  }

  // scale back
  xn = xn*xp[0];
  a[0] = a[0]*xp[0];
  a[3] = a[3]*a[2];

  return xn;
}

static PetscScalar lamtau_Y(PetscScalar xp, PetscScalar eve, PetscScalar A1, PetscScalar x)
{ PetscScalar aa, result;
  aa = PetscPowScalar(x*x + A1*A1, 0.5);
  result = (aa*xp/x - aa)/eve;
  return(result);
}

static PetscScalar dptau_Y(PetscScalar xp[2], PetscScalar eve, PetscScalar zve, PetscScalar phi, PetscScalar A1, PetscScalar x)
{ PetscScalar aa, result;
  aa = PetscPowScalar(x*x + A1*A1, 0.5);
  result = xp[1] + zve/eve * (aa*xp[0]/x - aa) * PetscSinScalar(phi);
  return(result);
}

PetscScalar ResF_Y(PetscScalar A1, PetscScalar A2, PetscScalar evp, PetscScalar phi, PetscScalar x[3])
{ PetscScalar aa, result;
  aa = PetscPowScalar(x[0]*x[0] + A1*A1, 0.5);
  result = aa - (A2 + x[1]*PetscSinScalar(phi)) - evp*x[2];
  return(result);
}

static PetscErrorCode tau_a_Y(PetscScalar c, PetscScalar ct, PetscScalar ap, PetscScalar phi, PetscScalar eve, PetscScalar zve, PetscScalar evp, PetscScalar xp[2], PetscScalar a[5])
{
  PetscFunctionBegin;
  a[0] = c*PetscCosScalar(phi) - ct*PetscSinScalar(phi);
  a[1] = c*PetscCosScalar(phi) + ap*PetscSinScalar(phi);
  a[2] = (zve*PetscPowScalar(PetscSinScalar(phi),2) + evp)/eve;
  a[3] = a[1] + xp[1] * PetscSinScalar(phi);
  a[4] = a[0]*(a[2] +1.0) - a[3] - a[2]*xp[0];
  PetscFunctionReturn(0);
}
// ---------------------------------------
// drive code to return tau_II, dp and dotlam
// input: c, ct, ap, phi, eve, zve, evp, xp[0] = taup, x[1] = delta pp
// output: xsol[0] = tauii, xsol[1] = delta p, xsol[2] = dotlam
// case 1: f < 0, tauii = tp, deltap = pp, dotlam = 0.0
// case 2: f > 0 and xp[0] = 0, analytical solution with tauII =0
// case 3: f > 0 and xp[0] != 0, numerical solution, call VEVP_hyper_tau, dptau and lamtau
PetscErrorCode VEVP_hyper_sol_Y(PetscInt Nmax, PetscScalar tf_tol,
                              PetscScalar c, PetscScalar ct, PetscScalar ap, PetscScalar phi, PetscScalar eve, PetscScalar zve, PetscScalar evp, PetscScalar xp[2],
                              PetscScalar a[5], PetscScalar xsol[3])
{ PetscScalar     ff;
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  // prepare the constant parameter in the equation solving tauII
  ierr =  tau_a_Y(c, ct, ap, phi, eve, zve, evp, xp, a); CHKERRQ(ierr);

  //PetscPrintf(PETSC_COMM_WORLD, "A1= %1.6f, A2=%1.6f, A3=%1.6f, A4=%1.6f, A5=%1.6f \n", a[0], a[1], a[2], a[3], a[4]);

  // check if it yields
  xsol[0] = xp[0];
  xsol[1] = xp[1];
  xsol[2] = 0.0;
  ff = ResF_Y(a[0], a[1], evp, phi, xsol);

  // if yield, compute tauII, then dp and dotlam
  if (ff>0) {

    if (xsol[0] < tf_tol) {
      xsol[0] = 0.0;
      xsol[2] = ((a[0]-a[1]) - xp[1]*PetscSinScalar(phi))/(zve*PetscPowScalar(PetscSinScalar(phi),2) + evp);
      xsol[1] = xp[1] + zve * xsol[2] * PetscSinScalar(phi);
    } else {
      xsol[0] = VEVP_hyper_tau_Y(a, xp, tf_tol, Nmax); CHKERRQ(ierr);
      xsol[1] = dptau_Y(xp, eve, zve, phi, a[0], xsol[0]);
      xsol[2] = lamtau_Y(xp[0], eve, a[0], xsol[0]);
    }

    //PetscPrintf(PETSC_COMM_WORLD, "TEST, tau = %1.6f, dp = %1.6f, lam = %1.6f\n", xsol[0], xsol[1], xsol[2]);
  }
  PetscFunctionReturn(0);
}