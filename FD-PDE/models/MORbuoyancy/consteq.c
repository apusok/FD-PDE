#include "MORbuoyancy.h"

// ---------------------------------------
// Temp2Theta
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "Temp2Theta"
PetscScalar Temp2Theta(PetscScalar x, PetscScalar Az) 
{ 
  return x*exp(-Az);
}

// ---------------------------------------
// Theta2Temp
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "Theta2Temp"
PetscScalar Theta2Temp(PetscScalar x, PetscScalar Az) 
{ 
  return x*exp( Az);
}

// ---------------------------------------
// BulkComposition
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "BulkComposition"
PetscScalar BulkComposition(PetscScalar Cf, PetscScalar Cs, PetscScalar phi) 
{ 
  return phi*Cf+(1.0-phi)*Cs;
}

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
PetscScalar TotalEnthalpy(PetscScalar phi, PetscScalar theta, PetscScalar Az, PetscScalar S, PetscScalar thetaS)
{
  return S*phi + exp(Az)*(theta + thetaS) - thetaS;
}
