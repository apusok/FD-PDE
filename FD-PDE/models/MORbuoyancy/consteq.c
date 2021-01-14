#include "MORbuoyancy.h"

// // ---------------------------------------
// // Temp2Theta
// // ---------------------------------------
// #undef __FUNCT__
// #define __FUNCT__ "Temp2Theta"
// PetscScalar Temp2Theta(PetscScalar x, PetscScalar Az) 
// { 
//   return x*exp(-Az);
// }

// // ---------------------------------------
// // Theta2Temp
// // ---------------------------------------
// #undef __FUNCT__
// #define __FUNCT__ "Theta2Temp"
// PetscScalar Theta2Temp(PetscScalar x, PetscScalar Az) 
// { 
//   return x*exp( Az);
// }

// // ---------------------------------------
// // BulkComposition
// // ---------------------------------------
// #undef __FUNCT__
// #define __FUNCT__ "BulkComposition"
// PetscScalar BulkComposition(PetscScalar Cf, PetscScalar Cs, PetscScalar phi) 
// { 
//   return phi*Cf+(1.0-phi)*Cs;
// }

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
      else                           { SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_LIB, "COuld not calculate porosity"); }
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