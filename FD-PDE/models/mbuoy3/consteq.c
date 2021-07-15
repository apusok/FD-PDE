#include "mbuoy3.h"

// ---------------------------------------
// Solidus - calculates either the solid composition (bool=1) or temp (bool=0)
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "Solidus"
PetscScalar Solidus(PetscScalar CT, PetscScalar Plith, PetscScalar G, PetscBool calc_C)
{
  PetscScalar PG = Plith*G;
  if (calc_C) { return CT - PG; }
  else        { return CT + PG; }
}

// ---------------------------------------
// Liquidus - calculates either the fluid composition (bool=1) or temp (bool=0)
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "Liquidus"
PetscScalar Liquidus(PetscScalar CT, PetscScalar Plith, PetscScalar G, PetscScalar RM, PetscBool calc_C)
{
  PetscScalar PG = Plith*G;
  if (calc_C) { return RM*CT - RM*PG - 1.0; }
  else        { return (CT+1.0+RM*PG)/RM; }
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
// TotalEnthalpy
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "TotalEnthalpy"
PetscScalar TotalEnthalpy(PetscScalar theta, PetscScalar phi, PetscScalar S)
{
  return S*phi + theta;
}

// ---------------------------------------
// Porosity -  Ridder's algorithm, from Numerical Recipes in C
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "Porosity"
#define MAXIT 60
#define XACC  1e-8
#define SIGN(a) ((a)>=0 ? 1.0:-1.0)
PetscErrorCode Porosity(PetscScalar H, PetscScalar C, PetscScalar P, PetscScalar *phi, PetscScalar S, PetscScalar G, PetscScalar RM)
{
  PetscInt    j;
  PetscScalar ans,fh,fl,fm,fnew,s,xm,xnew,xl,xh;

  PetscFunctionBegin;
  xl=0.0; xh=1.0;
  fl = PhiRes(xl,H,C,P,S,G,RM);
  fh = PhiRes(xh,H,C,P,S,G,RM);
  if (((fl > 0.0) && (fh < 0.0)) || ((fl < 0.0) && (fh > 0.0))) {
    ans = 0.0;
    for(j = 0; j < MAXIT; j++) {
      xm = 0.5*(xl+xh);
      fm = PhiRes(xm,H,C,P,S,G,RM);
      s  = sqrt(fm*fm-fl*fh);
      if (s == 0.0) {
        if (j == 0) {SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP, "Could not calculate porosity");}
        *phi = ans;
        break;
      }
      xnew = xm+(xm-xl)*(SIGN(fl-fh)*fm/s);
      if (fabs(xnew-ans) <= XACC) {
        *phi = ans;
        break;
      }
      ans  = xnew;
      fnew = PhiRes(ans,H,C,P,S,G,RM);
      if (fnew == 0.0) {
        *phi = ans;
        break;
      }
      if      ((fm<0.0)==(fnew>0.0)) { xh = ans; fh = fnew; xl = xm; fl = fm; } 
      else if ((fl<0.0)==(fnew>0.0)) { xh = ans; fh = fnew; }
      else if ((fh<0.0)==(fnew>0.0)) { xl = ans; fl = fnew; }
      else                           { SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_LIB, "Could not calculate porosity"); }
      if (fabs(xh-xl) <= XACC) {
        *phi = ans;
        break;
      }
    }
    if (j == MAXIT) {
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_LIB, "ERROR: zriddr---exceeded max its\n");
    }
  } else if (fabs(fl)< 1.0e-15) {
    *phi = xl;
  } else if (fabs(fh)< 1.0e-15) {
    *phi = xh;
  } else {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_LIB, "ERROR: zriddr\n");
  }
  PetscFunctionReturn(0);
}

// ---------------------------------------
// PhiRes
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "PhiRes"
PetscScalar PhiRes(PetscScalar phi, PetscScalar H, PetscScalar C, PetscScalar P, PetscScalar S, PetscScalar G, PetscScalar RM)
{
  return Solidus (H-S*phi,P,G,PETSC_TRUE)*(1.0-phi)
      +  Liquidus(H-S*phi,P,G,RM,PETSC_TRUE)*phi - C;
}

// ---------------------------------------
// FluidVelocity
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FluidVelocity"
PetscScalar FluidVelocity(PetscScalar vs, PetscScalar phi, PetscScalar gradP, PetscScalar gradPc, PetscScalar Bf, PetscScalar K, PetscScalar k_hat) 
{ 
  if (K == 0.0) return 0.0;
  else          return vs-K/phi*(gradP+gradPc+(1.0+Bf)*k_hat);
}

// ---------------------------------------
// BulkVelocity
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "BulkVelocity"
PetscScalar BulkVelocity(PetscScalar vs, PetscScalar vf, PetscScalar phi) 
{ 
  return vf*phi + vs*(1.0-phi);
}

// ---------------------------------------
// Permeability
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "Permeability"
PetscScalar Permeability(PetscScalar phi, PetscScalar phi_max, PetscScalar n) 
{ 
  // return pow(pow(phi,-n)+pow(phi_max,-n),-1); // harmonic averaging
  return 1.0/(pow(phi,-n)+pow(phi_max,-n));
}

// ---------------------------------------
// FluidBuoyancy
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FluidBuoyancy"
PetscScalar FluidBuoyancy(PetscScalar T, PetscScalar CF, PetscScalar alpha_s, PetscScalar beta_s) 
{ 
  return 0.0;
  // return alpha_s*T + beta_s*CF
}

// ---------------------------------------
// Buoyancy
// ---------------------------------------
PetscScalar Buoyancy_phi(PetscScalar phi, PetscInt buoy) 
{ 
  if (buoy == 1) return phi;
  return 0.0;
}

PetscScalar Buoyancy_Composition(PetscScalar C, PetscScalar CF, PetscScalar phi, PetscScalar beta_s, PetscScalar beta_ls, PetscInt buoy) 
{ 
  if (buoy == 1) return beta_s*C;
  if (buoy == 2) return beta_s*C - phi*beta_ls*CF; 
  return 0.0;
}

PetscScalar Buoyancy_Temperature(PetscScalar T, PetscScalar phi, PetscScalar alpha_s, PetscScalar alpha_ls, PetscInt buoy) 
{ 
  if (buoy == 1) return alpha_s*T;
  if (buoy == 2) return alpha_s*T - phi*alpha_ls*T; 
  return 0.0;
}

// ---------------------------------------
// HalfSpaceCoolingTemp
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "HalfSpaceCoolingTemp"
PetscScalar HalfSpaceCoolingTemp(PetscScalar Tm, PetscScalar T0, PetscScalar z, PetscScalar kappa, PetscScalar t) 
{ 
  return T0 + (Tm-T0)*erf(z/(2.0*sqrt(kappa*t)));
}

// ---------------------------------------
// ShearViscosity
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "ShearViscosity"
PetscScalar ShearViscosity(PetscScalar T, PetscScalar phi, PetscScalar EoR, PetscScalar Teta0, PetscScalar lambda, PetscScalar eta_min, PetscScalar eta_max, PetscInt visc) 
{ 
  PetscScalar eta;
  if (visc == 0) { // constant 
    eta = 1.0; 
    return eta;
  } 

  if (visc == 1) { // phi-dependent
    eta = exp(-lambda*phi);
  return 1.0/(1.0/eta + 1.0/eta_max) + eta_min; // harmonic averaging
  } 

  // T,phi-dep eta = exp(EoR*(1.0/T-1.0/Teta0)-lambda*phi);
  eta = ArrheniusTerm_Viscosity(T,EoR,Teta0)*exp(-lambda*phi);
  return 1.0/(1.0/eta + 1.0/eta_max) + eta_min; // harmonic averaging
}

PetscScalar ArrheniusTerm_Viscosity(PetscScalar T, PetscScalar EoR, PetscScalar Teta0) 
{ return exp(EoR*(1.0/T-1.0/Teta0)); }

// ---------------------------------------
// BulkViscosity
// ---------------------------------------
PetscScalar BulkViscosity(PetscScalar visc_ratio, PetscScalar T, PetscScalar phi, PetscScalar EoR, PetscScalar Teta0, PetscScalar phi_min, PetscScalar zetaExp, PetscScalar eta_min, PetscScalar eta_max, PetscInt visc) 
{ 
  PetscScalar zeta, eta_arr;
  
  if (visc == 0) { zeta = visc_ratio; }  // constant
  if (visc == 1) { zeta = visc_ratio*pow(phi+phi_min,zetaExp); } //1/(phi+phi_min)

  // T,phi-dep = zeta(phi)*exp(EoR*(1.0/T-1.0/Teta0);
  if (visc == 2) { 
    zeta = visc_ratio*pow(phi+phi_min,zetaExp);
    eta_arr = ArrheniusTerm_Viscosity(T,EoR,Teta0);
    eta_arr = 1.0/(1.0/eta_arr + 1.0/eta_max) + eta_min;
    zeta *= eta_arr;
  }
  return zeta;
}

// ---------------------------------------
// SolidDensity (scaled by drho)
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "SolidDensity"
PetscScalar SolidDensity(PetscScalar rho0, PetscScalar drho, PetscScalar T, PetscScalar C, PetscScalar alpha_s, PetscScalar beta_s, PetscInt buoy) 
{ 
  PetscScalar prho0;
  prho0 = rho0/drho;
  if ((buoy >= 2) && (buoy <= 3)) return prho0 - beta_s*C; 
  if ((buoy >= 4) && (buoy <= 5)) return prho0 - alpha_s*T; 
  if ((buoy >= 6) && (buoy <= 7)) return prho0 - beta_s*C - alpha_s*T; 
  return prho0; // constant (buoy <= 1)
}

// ---------------------------------------
// FluidDensity (scaled by drho)
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FluidDensity"
PetscScalar FluidDensity(PetscScalar rho0, PetscScalar drho, PetscScalar T, PetscScalar C, PetscScalar alpha_s, PetscScalar beta_s, PetscInt buoy) 
{ 
  PetscScalar prho0, prho1;
  prho0 = 1.0 - drho/rho0;
  prho1 = rho0/drho;
  if ((buoy >= 2) && (buoy <= 3)) return prho0*(prho1 - beta_s*C); 
  if ((buoy >= 4) && (buoy <= 5)) return prho0*(prho1 - alpha_s*T); 
  if ((buoy >= 6) && (buoy <= 7)) return prho0*(prho1 - beta_s*C - alpha_s*T); 
  return prho0*(prho1); // constant (buoy <= 1)
}

// ---------------------------------------
// BulkDensity
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "BulkDensity"
PetscScalar BulkDensity(PetscScalar rhos, PetscScalar rhof, PetscScalar phi) 
{ 
  return rhof*phi + rhos*(1.0-phi);
}