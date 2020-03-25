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
UpwindAdvection - returns the [UPWIND] residual value for the advection term for FDPDEType = ADVDIFF

Use: internal
@*/
// ---------------------------------------
PetscScalar UpwindAdvection(PetscScalar v[], PetscScalar x[], PetscScalar dx[], PetscScalar dz[])
{
  PetscScalar vx, vz, vxmin, vxmax, vzmin, vzmax;
  PetscScalar dadx1, dadx2, dadz1, dadz2;

  vx = (v[pE]+v[pW])*0.5;
  vz = (v[pS]+v[pN])*0.5;

  vxmin  =  PetscMin(0,vx); vxmax  =  PetscMax(0,vx);
  vzmin  =  PetscMin(0,vz); vzmax  =  PetscMax(0,vz);

  // can also do this choice for staggered grids
  // vxmin  =  PetscMin(0,v[pE]); vxmax  =  PetscMax(0,v[pW]);
  // vzmin  =  PetscMin(0,v[pN]); vzmax  =  PetscMax(0,v[pS]);

  dadx1 = (x[pE]-x[pC])/dx[0];
  dadx2 = (x[pC]-x[pW])/dx[1];

  dadz1 = (x[pN]-x[pC])/dz[0];
  dadz2 = (x[pC]-x[pS])/dz[1];

  return  vxmin*dadx1 + vxmax*dadx2 
        + vzmin*dadz1 + vzmax*dadz2;
}

// ---------------------------------------
/*@
UpwindAdvection2 - returns the [UPWIND2] order residual value for the advection term for FDPDEType = ADVDIFF

Use: internal
@*/
// ---------------------------------------
PetscScalar UpwindAdvection2(PetscScalar v[], PetscScalar x[], PetscScalar dx[], PetscScalar dz[])
{
  PetscScalar vx, vz, vxmin, vxmax, vzmin, vzmax;
  PetscScalar dadx1, dadx2, dadz1, dadz2;

  vx = (v[pE]+v[pW])*0.5;
  vz = (v[pS]+v[pN])*0.5;

  vxmin  =  PetscMin(0,vx); vxmax  =  PetscMax(0,vx);
  vzmin  =  PetscMin(0,vz); vzmax  =  PetscMax(0,vz);

  // vxmin  =  PetscMin(0,v[pE]); vxmax  =  PetscMax(0,v[pW]);
  // vzmin  =  PetscMin(0,v[pN]); vzmax  =  PetscMax(0,v[pS]);

  dadx1 = 0.5*(-3.0*x[pC]+4.0*x[pE]-x[pEE])/dx[0];
  dadx2 = 0.5*( 3.0*x[pC]-4.0*x[pW]+x[pWW])/dx[1];

  dadz1 = 0.5*(-3.0*x[pC]+4.0*x[pN]-x[pNN])/dz[0];
  dadz2 = 0.5*( 3.0*x[pC]-4.0*x[pS]+x[pSS])/dz[1];

  return  vxmin*dadx1 + vxmax*dadx2 
        + vzmin*dadz1 + vzmax*dadz2;
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