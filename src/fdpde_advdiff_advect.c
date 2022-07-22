#include "fdpde_advdiff.h"

#define pC  0
#define pW  1
#define pE  2
#define pS  3
#define pN  4
#define pWW 5
#define pEE 6
#define pSS 7
#define pNN 8

static PetscScalar get_upwind_flux(PetscScalar v, PetscScalar f_down, PetscScalar f_up)
{ PetscScalar result;
  result = 0.5*v*(f_up+f_down)-0.5*PetscAbsScalar(v)*(f_up-f_down);
  return(result);
}
// ---------------------------------------
/*@
AdvectionResidual - returns the residual value for the advection term for FDPDEType = ADVDIFF

Use: internal
@*/
// ---------------------------------------
PetscErrorCode AdvectionResidual(PetscScalar v[], PetscScalar x[], PetscScalar dx[], PetscScalar dz[], AdvectSchemeType advtype, PetscScalar *val)
{
  PetscScalar    fval = 0.0;
  PetscFunctionBegin;
  
  // Choose between different advection schemes
  switch (advtype) {
    case ADV_NONE:
      break;
    case ADV_UPWIND:
      fval = UpwindAdvection(v,x,dx,dz);
      break;
    case ADV_UPWIND2:
      fval = UpwindAdvection2(v,x,dx,dz);
      break;
    case ADV_FROMM:
      fval = FrommAdvection(v,x,dx,dz);
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Unknown advection scheme type! Set with FDPDEAdvDiffSetAdvectSchemeType()");
  }

  *val = fval;

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
UpwindAdvection - returns the first order upwind (FOU) residual value for the advection term for FDPDEType = ADVDIFF

Use: internal
@*/
// ---------------------------------------
PetscScalar UpwindAdvection(PetscScalar v[], PetscScalar x[], PetscScalar dx[], PetscScalar dz[])
{
  PetscScalar fW,fE,fS,fN;

  fW = get_upwind_flux(v[pW],x[pW],x[pC]);
  fE = get_upwind_flux(v[pE],x[pC],x[pE]);
  fS = get_upwind_flux(v[pS],x[pS],x[pC]);
  fN = get_upwind_flux(v[pN],x[pC],x[pN]);

  return (fE-fW)/dx[2] + (fN-fS)/dz[2];
}

// ---------------------------------------
/*@
UpwindAdvection2 - returns the second order upwind (SOU) residual value for the advection term for FDPDEType = ADVDIFF

Use: internal
@*/
// ---------------------------------------
PetscScalar UpwindAdvection2(PetscScalar v[], PetscScalar x[], PetscScalar dx[], PetscScalar dz[])
{
  PetscScalar fW,fE,fS,fN,f_down,f_up;

  // Fluxes across boundaries - including second order taylor series expansion terms
  f_down = 0.5*(3.0*x[pW]-x[pWW]); f_up = 0.5*(3.0*x[pC]-x[pE] ); fW = get_upwind_flux(v[pW],f_down,f_up);
  f_down = 0.5*(3.0*x[pC]-x[pW] ); f_up = 0.5*(3.0*x[pE]-x[pEE]); fE = get_upwind_flux(v[pE],f_down,f_up);
  f_down = 0.5*(3.0*x[pS]-x[pSS]); f_up = 0.5*(3.0*x[pC]-x[pN] ); fS = get_upwind_flux(v[pS],f_down,f_up);
  f_down = 0.5*(3.0*x[pC]-x[pS] ); f_up = 0.5*(3.0*x[pN]-x[pNN]); fN = get_upwind_flux(v[pN],f_down,f_up);

  return (fE-fW)/dx[2] + (fN-fS)/dz[2];
}

// ---------------------------------------
/*@
FrommAdvection - returns the [FROMM] residual value for the advection term for FDPDEType = ADVDIFF

Use: internal
@*/
// ---------------------------------------
PetscScalar FrommAdvection(PetscScalar v[], PetscScalar x[], PetscScalar dx[], PetscScalar dz[])
{
  PetscScalar vN, vS, vE, vW;
  PetscScalar fE, fW, fS, fN;
  PetscScalar xC, xN, xNN, xS, xSS, xE, xEE, xW, xWW;

  vN = v[pN]; vE = v[pE]; vS = v[pS]; vW = v[pW]; 

  xC = x[pC]; 
  xN = x[pN]; xNN = x[pNN]; xS = x[pS]; xSS = x[pSS]; 
  xE = x[pE]; xEE = x[pEE]; xW = x[pW]; xWW = x[pWW];

  fE = vE *(-xEE + 5*(xE+xC)-xW )/8 - fabs(vE)*(-xEE + 3*(xE-xC)+xW )/8;
  fW = vW *(-xE  + 5*(xC+xW)-xWW)/8 - fabs(vW)*(-xE  + 3*(xC-xW)+xWW)/8;
  fN = vN *(-xNN + 5*(xN+xC)-xS )/8 - fabs(vN)*(-xNN + 3*(xN-xC)+xS )/8;
  fS = vS *(-xN  + 5*(xC+xS)-xSS)/8 - fabs(vS)*(-xN  + 3*(xC-xS)+xSS)/8;

  return (fE-fW)/dx[2] + (fN-fS)/dz[2];
}