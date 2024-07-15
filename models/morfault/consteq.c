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