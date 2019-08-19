#include "stagridge.h"

#define pC  0
#define pW  1
#define pE  2
#define pS  3
#define pN  4
#define pWW 5
#define pEE 6
#define pSS 7
#define pNN 8

// ---------------------------------------
// AdvectionResidual
// ---------------------------------------
PetscErrorCode AdvectionResidual(SolverCtx *sol, PetscScalar v[], PetscScalar x[], PetscScalar *val)
{
  PetscScalar    fval = 0.0;
  PetscFunctionBegin;
  
  // Choose between different advection schemes
  switch (sol->grd->advtype) {
    case UPWIND:
      fval = UpwindAdvection(v,x,sol->grd->dx,sol->grd->dz);
      break;
    case FROMM:
      fval = FrommAdvection(v,x,sol->grd->dx,sol->grd->dz);
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Unknown advection type set with option: -advtype");
  }

  // Return value
  *val = fval;

  PetscFunctionReturn(0);
}

// ---------------------------------------
// UpwindAdvection
// ---------------------------------------
PetscScalar UpwindAdvection(PetscScalar v[], PetscScalar x[], PetscScalar dx, PetscScalar dz)
{
  PetscScalar vx, vz, vxmin, vxmax, vzmin, vzmax;
  PetscScalar dadx1, dadx2, dadz1, dadz2;

  vx = (v[pE]+v[pW])*0.5;
  vz = (v[pS]+v[pN])*0.5;

  vxmin  =  PetscMin(0,vx); vxmax  =  PetscMax(0,vx);
  vzmin  =  PetscMin(0,vz); vzmax  =  PetscMax(0,vz);

  dadx1 = (x[pE]-x[pC])/dx;
  dadx2 = (x[pC]-x[pW])/dx;

  dadz1 = (x[pN]-x[pC])/dz;
  dadz2 = (x[pC]-x[pS])/dz;

  return  vxmin*dadx1 + vxmax*dadx2 
        + vzmin*dadz1 + vzmax*dadz2;
}

// ---------------------------------------
// FrommAdvection
// ---------------------------------------
PetscScalar FrommAdvection(PetscScalar v[], PetscScalar x[], PetscScalar dx, PetscScalar dz)
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

  return (fE-fW)/dx + (fN-fS)/dz;
}
