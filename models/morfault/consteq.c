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
PetscScalar LithostaticPressure(PetscScalar rho, PetscScalar drho, PetscScalar z)
{
  return -rho*z/drho;
}

// ---------------------------------------
// Material Density - dimensional
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
// ShearViscosity
// eta0, scal_eta, T, EoR, Teta0 - dimensional
// eta_min, eta_max - non-dimensional
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "ShearViscosity"
PetscScalar ShearViscosity(PetscScalar eta0, PetscScalar scal_eta, PetscScalar T, PetscScalar phi, PetscScalar EoR, PetscScalar Teta0, PetscScalar lambda, PetscScalar eta_min, PetscScalar eta_max, PetscInt func) 
{ 
  PetscScalar eta, peta0, eta_arr;
  peta0 = eta0/scal_eta;
  // constant 
  if (func == 0) { eta = peta0; } 

  // phi-dependent, with harmonic averaging
  if (func == 1) { eta = peta0*exp(-lambda*phi); } 

  // T,phi-dep, with harmonic averaging
  if (func == 2) { 
    eta_arr = peta0*ArrheniusTerm_Viscosity(T,EoR,Teta0)*exp(-lambda*phi);
    eta = eta_arr;
    // eta = 1.0/(1.0/eta_arr + 1.0/eta_max) + eta_min;
  }
  return eta;
}

// ---------------------------------------
// Power-law ShearViscosity
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "ShearViscosity_PowerLaw"
PetscScalar ShearViscosity_PowerLaw(PetscScalar eta0, PetscScalar scal_eta, PetscScalar eII, PetscScalar T, PetscScalar P, PetscScalar Ad, PetscScalar n, PetscScalar Ea, PetscScalar Va, PetscScalar R, PetscInt func) 
{ 
  PetscScalar eta, peta0;
  peta0 = eta0/scal_eta;
  // constant 
  if (func == 0) { eta = peta0; } 

  // power-law
  if (func == 3) { 
    // if (eII == 0.0) { eta = peta0; }
    // n = 1.0;
    // else {
      PetscScalar A, ERT, arr;
      A = 1.0e6*PetscPowScalar(Ad,-1.0/n);
      // ERT = (1.0e-3*Ea + P*1.0e6*Va)/(R*T);
      ERT = (1.0e3*Ea + P*1.0e-6*Va)/(n*R*T);
      // arr = pow(exp(-ERT),-1.0/n);
      arr = PetscExpScalar(ERT);
      // eta = 0.5*PetscPowScalar(eII,1.0/n-1.0)*A*arr/scal_eta; 
      if (n==1.0) eta = 0.5*A*arr/scal_eta; 
      // PetscPrintf(PETSC_COMM_WORLD,"# BREAK A %1.12e %1.12e %1.12e %1.12e %1.12e %1.12e #\n",A,ERT,arr,eta,eta*scal_eta,eII);
    // }
  } 

  return eta;
}

// ---------------------------------------
PetscScalar ArrheniusTerm_Viscosity(PetscScalar T, PetscScalar EoR, PetscScalar Teta0) 
{ return exp(EoR*(1.0/T-1.0/Teta0)); }

// ---------------------------------------
// CompactionViscosity
// ---------------------------------------
PetscScalar CompactionViscosity(PetscScalar zeta0, PetscScalar scal_eta, PetscScalar T, PetscScalar phi, PetscScalar EoR, PetscScalar Teta0, PetscScalar phi_min, PetscScalar zetaExp, PetscScalar eta_min, PetscScalar eta_max, PetscInt func) 
{ 
  PetscScalar zeta, pzeta0, zeta_arr;
  pzeta0 = zeta0/scal_eta;

  // constant
  if (func == 0) { zeta = pzeta0; }  

  // phi-dependent, with 1/(phi+phi_min)
  if (func == 1) { zeta = pzeta0*pow(phi+phi_min,zetaExp); }

  // T,phi-dep, with harmonic averaging and with 1/(phi+phi_min)
  if (func == 2) { 
    zeta_arr = pzeta0*pow(phi+phi_min,zetaExp)*ArrheniusTerm_Viscosity(T,EoR,Teta0);
    zeta = zeta_arr;
    // zeta = 1.0/(1.0/zeta_arr + 1.0/eta_max) + eta_min;
  }
  return zeta;
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
// Poro-elastic modulus
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "PoroElasticModulus"
PetscScalar PoroElasticModulus(PetscScalar Z0, PetscScalar phi) 
{ 
  if (phi == 0.0) return 0.0;
  else            return Z0*PetscPowScalar(phi,-0.5);
}

// ---------------------------------------
// Inverse of Poro-elastic modulus (dim) - avoid singularity in 1/sqrt(phi)
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "InvPoroElasticModulus"
PetscScalar InvPoroElasticModulus(PetscScalar Z0, PetscScalar phi) 
{ 
  return PetscPowScalar(phi,0.5)/Z0;
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

// static PetscScalar tau_ini(PetscScalar A1, PetscScalar A2, PetscScalar A3, PetscScalar A4, PetscScalar A5, PetscScalar xp)
// { PetscScalar A6, result;
//   A6 = A3 + 1.0;
//   result = -A5 + PetscPowScalar(A5*A5 + 4.0*A1*A3*A6*xp , 0.5 );
//   result = 0.5*result/A6;
//   return(result);
// }

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

// static PetscScalar dptau(PetscScalar xp[2], PetscScalar eve, PetscScalar zve, PetscScalar phi, PetscScalar A1, PetscScalar x)
// { PetscScalar aa, result;
//   aa = PetscPowScalar(x*x + A1*A1, 0.5);
//   result = xp[1] + zve/eve * (aa*xp[0]/x - aa) * PetscSinScalar(phi);
//   return(result);
// }

// static PetscScalar lamtau(PetscScalar xp, PetscScalar eve, PetscScalar A1, PetscScalar x)
// { PetscScalar aa, result;
//   aa = PetscPowScalar(x*x + A1*A1, 0.5);
//   result = (aa*xp/x - aa)/eve;
//   return(result);
// }

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

  // // rescale xn and A1 by tauIIt, A4 by A3
  // xn = xn/tauIIt;
  // a[0] = a[0]/tauIIt;
  // a[3] = a[3]/a[2];

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

  // // scale back
  // xn = xn*tauIIt;
  // a[0] = a[0]*tauIIt;
  // a[3] = a[3]*a[2];

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
  eta_vp  = usr->nd->eta_vp;
  cdl     = 0.0; // should be function of phi
  sint    = PetscSinScalar(theta);

  // trial solution
  xsol[0] = tauIIt; // tauII
  xsol[1] = DPt;    // DP
  xsol[2] = 0.0;    // dotlam
  
  // get coefficients A
  ierr = tau_a(tauIIt,DPt,eta_ve,zeta_ve,C,sigmat,aP,theta,cdl,eta_vp,a); CHKERRQ(ierr);
  //PetscPrintf(PETSC_COMM_WORLD, "A1= %1.6f, A2=%1.6f, A3=%1.6f, A4=%1.6f, A5=%1.6f \n", a[0], a[1], a[2], a[3], a[4]);

  // check if it yields
  F = ResF(xsol,a,theta,eta_vp); CHKERRQ(ierr);
  
  if (F>0) { // yield
    // PetscPrintf(PETSC_COMM_WORLD, "PLASTIC - yield\n");
    if (xsol[0] <= usr->par->tf_tol) { // case 2
      xsol[0] = 0.0; 
      xsol[2] = (a[0]-a[1]-DPt*sint)/(cdl*zeta_ve*sint*sint + eta_vp);
      xsol[1] = DPt + cdl * zeta_ve * sint * xsol[2]; 
    } else { // case 3
      xsol[0] = VEVP_hyper_tau(a, tauIIt, usr->par->tf_tol, usr->par->Nmax); CHKERRQ(ierr);
      xsol[2] = lamtau(a,xsol[0],DPt,sint,cdl,zeta_ve,eta_vp);
      // xsol[1] = dptau(xve, eta_ve, zeta_ve, phi, a[0], xsol[0]);
      // xsol[2] = lamtau(xve[0], eta_ve, a[0], xsol[0]);
      xsol[1] = DPt + cdl * zeta_ve * sint * xsol[2]; 
    }
    //PetscPrintf(PETSC_COMM_WORLD, "TEST, tau = %1.6f, dp = %1.6f, lam = %1.6f\n", xsol[0], xsol[1], xsol[2]);
  }
  PetscFunctionReturn(0);
}