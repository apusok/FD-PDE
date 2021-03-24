#include "MORbuoyancy.h"

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
PetscScalar FluidVelocity(PetscScalar vs, PetscScalar phi, PetscScalar gradP, PetscScalar Bf, PetscScalar K, PetscScalar k_hat, PetscScalar phi_cutoff) 
{ 
  if (phi < phi_cutoff) return 0.0;
  else                  return vs-K/phi*(gradP+(1+Bf)*k_hat);
}

// ---------------------------------------
// BulkVelocity
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "BulkVelocity"
PetscScalar BulkVelocity(PetscScalar vs, PetscScalar vf, PetscScalar phi) 
{ 
  return vf*phi + vs*(1-phi);
}

// ---------------------------------------
// Permeability
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "Permeability"
PetscScalar Permeability(PetscScalar phi, PetscScalar phi0, PetscScalar phi_max, PetscScalar n, PetscScalar phi_cutoff) 
{ 
  if (phi < phi_cutoff) return 0.0;
  // return pow(phi/phi0,n);
  return pow(pow(phi/phi0,-n)+pow(phi_max/phi0,-n),-1); // harmonic averaging
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
#undef __FUNCT__
#define __FUNCT__ "Buoyancy"
PetscScalar Buoyancy(PetscScalar phi, PetscScalar T, PetscScalar C, PetscScalar alpha_s, PetscScalar beta_s, PetscInt buoy) 
{ 
  if (buoy == 0) return 0.0;
  if (buoy == 1) return phi;
  if (buoy == 2) return phi + beta_s*C; 
  if (buoy == 3) return phi + beta_s*C + alpha_s*T;
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
PetscScalar ShearViscosity(PetscScalar T, PetscScalar phi, PetscScalar EoR, PetscScalar Teta0, PetscScalar lambda, PetscScalar eta0, PetscScalar eta_min, PetscScalar eta_max, PetscInt visc) 
{ 
  PetscScalar eta;
  if (visc == 0) { // constant 
    eta = 1.0; 
    return eta;
  } 

  eta = exp(EoR*(1.0/T-1.0/Teta0)-lambda*phi); // T,phi-dep
  return 1.0/(1.0/eta + eta0/eta_max) + eta_min/eta0; // harmonic averaging
}

// ---------------------------------------
// BulkViscosity
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "BulkViscosity"
PetscScalar BulkViscosity(PetscScalar T, PetscScalar phi, PetscScalar EoR, PetscScalar Teta0, PetscScalar eta0, PetscScalar visc_ratio, PetscScalar zetaExp, PetscScalar eta_min, PetscScalar eta_max, PetscScalar phi_cutoff, PetscInt visc) 
{ 
  PetscScalar zeta;

  // below this value, it is assumed div(vs) = curly(P)/(zeta-2/3eta) = 0
  if (phi < phi_cutoff) phi = phi_cutoff; 

  if (visc == 0) { // constant 
    zeta = visc_ratio;
    return zeta;
  } 

  if (visc == 1) { // porosity dependent
    zeta = visc_ratio*pow(phi,zetaExp);;
    return zeta;
  } 

  zeta = visc_ratio*exp(EoR*(1.0/T-1.0/Teta0))*pow(phi,zetaExp); // T,phi-dep
  return 1.0/(1.0/zeta + eta0/eta_max) + eta_min/eta0; // harmonic averaging
}

// ---------------------------------------
// CompactionViscosity
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "CompactionViscosity"
PetscScalar CompactionViscosity(PetscScalar T, PetscScalar phi, PetscScalar EoR, PetscScalar Teta0, PetscScalar lambda, PetscScalar eta0, PetscScalar visc_ratio, PetscScalar zetaExp, PetscScalar eta_min, PetscScalar eta_max, PetscScalar phi_cutoff, PetscInt visc_shear, PetscInt visc_bulk) 
{ 
  PetscScalar eta, zeta, xi;
  if (phi < phi_cutoff) xi = 0.0;
  else {
    eta  = ShearViscosity(T,phi,EoR,Teta0,lambda,eta0,eta_min,eta_max,visc_shear);
    zeta = BulkViscosity(T,phi,EoR,Teta0,eta0,visc_ratio,zetaExp,eta_min,eta_max,phi_cutoff,visc_bulk);
    xi = zeta-2.0/3.0*eta;
  }
  return xi;
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
  if (buoy <= 1) return prho0;
  if (buoy == 2) return prho0 - beta_s*C; 
  if (buoy == 3) return prho0 - beta_s*C - alpha_s*T;
  return 0.0;
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
  if (buoy <= 1) return prho0*(prho1);
  if (buoy == 2) return prho0*(prho1 - beta_s*C); 
  if (buoy == 3) return prho0*(prho1 - beta_s*C - alpha_s*T);
  return 0.0;
}

// ---------------------------------------
// BulkDensity
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "BulkDensity"
PetscScalar BulkDensity(PetscScalar rhos, PetscScalar rhof, PetscScalar phi, PetscInt buoy) 
{ 
  if (buoy == 0) return rhos;
  else           return rhof*phi + rhos*(1-phi);
}