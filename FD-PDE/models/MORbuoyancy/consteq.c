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

// // ---------------------------------------
// // Update Theta From Temperature
// // ---------------------------------------
// #undef __FUNCT__
// #define __FUNCT__ "UpdateThetaFromTemp"
// PetscErrorCode UpdateThetaFromTemp(void *ctx)
// {
//   UsrData       *usr = (UsrData*) ctx;
//   PetscInt       i, j, sx, sz, nx, nz, Nx, Nz, idx, icenter;
//   PetscScalar    ***xx, ***xxtheta;
//   PetscScalar    **coordx,**coordz;
//   Vec            x, xlocal, xtheta, xthetalocal;
//   DM             dm;
//   PetscErrorCode ierr;

//   PetscFunctionBegin;

//   dm = usr->dmHC;
//   x  = usr->xT;
//   xtheta  = usr->xTheta;

//   ierr = DMStagGetGlobalSizes(dm, &Nx, &Nz,NULL);CHKERRQ(ierr);
//   ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

//   ierr = DMGetLocalVector(dm, &xlocal); CHKERRQ(ierr);
//   ierr = DMGlobalToLocal (dm, x, INSERT_VALUES, xlocal); CHKERRQ(ierr);
//   ierr = DMStagVecGetArray(dm, xlocal, &xx); CHKERRQ(ierr);

//   ierr = DMCreateLocalVector(dm, &xthetalocal); CHKERRQ(ierr);
//   ierr = DMStagVecGetArray(dm, xthetalocal, &xxtheta); CHKERRQ(ierr);

// // Get dm coordinates array
//   ierr = DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
//   ierr = DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter);CHKERRQ(ierr); 

//   // Loop over local domain
//   for (j = sz; j < sz+nz; j++) {
//     for (i = sx; i <sx+nx; i++) {
//       PetscScalar Az;
//       Az = -usr->nd->A*coordz[j][icenter]; // check sign of non-dimensional depth!
//       ierr = DMStagGetLocationSlot(dm, ELEMENT, 0, &idx); CHKERRQ(ierr);
//       xxtheta[j][i][idx] = Temp2Theta(xx[j][i][idx],Az);
//     }
//   }

//   // Restore arrays
//   ierr = DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
//   ierr = DMStagVecRestoreArray(dm,xlocal,&xx); CHKERRQ(ierr);
//   ierr = DMStagVecRestoreArray(dm,xthetalocal,&xxtheta); CHKERRQ(ierr);

//   ierr = DMLocalToGlobalBegin(dm,xthetalocal,INSERT_VALUES,xtheta); CHKERRQ(ierr);
//   ierr = DMLocalToGlobalEnd  (dm,xthetalocal,INSERT_VALUES,xtheta); CHKERRQ(ierr);
//   ierr = VecDestroy(&xthetalocal); CHKERRQ(ierr);
//   ierr = DMRestoreLocalVector(dm,&xlocal); CHKERRQ(ierr);

//   PetscFunctionReturn(0);
// }

// // ---------------------------------------
// // UpdateComposition
// // ---------------------------------------
// #undef __FUNCT__
// #define __FUNCT__ "UpdateComposition"
// PetscErrorCode UpdateComposition(void *ctx)
// {
//   UsrData       *usr = (UsrData*) ctx;
//   PetscInt       i, j, sx, sz, nx, nz, Nx, Nz, idx, icenter;
//   PetscScalar    ***xx, ***xxCs, ***xxCf, ***xxphi;
//   PetscScalar    **coordx,**coordz;
//   Vec            xlocal, xCflocal, xCslocal, xphilocal;
//   DM             dm;
//   PetscErrorCode ierr;

//   PetscFunctionBegin;

//   dm = usr->dmHC;

//   ierr = DMStagGetGlobalSizes(dm, &Nx, &Nz,NULL);CHKERRQ(ierr);
//   ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

//   ierr = DMCreateLocalVector(dm, &xlocal); CHKERRQ(ierr);
//   ierr = DMStagVecGetArray(dm, xlocal, &xx); CHKERRQ(ierr);

//   ierr = DMGetLocalVector(dm, &xCflocal); CHKERRQ(ierr);
//   ierr = DMGlobalToLocal (dm, usr->xCf, INSERT_VALUES, xCflocal); CHKERRQ(ierr);
//   ierr = DMStagVecGetArray(dm, xCflocal, &xxCf); CHKERRQ(ierr);

//   ierr = DMGetLocalVector(dm, &xCslocal); CHKERRQ(ierr);
//   ierr = DMGlobalToLocal (dm, usr->xCs, INSERT_VALUES, xCslocal); CHKERRQ(ierr);
//   ierr = DMStagVecGetArray(dm, xCslocal, &xxCs); CHKERRQ(ierr);

//   ierr = DMGetLocalVector(dm, &xphilocal); CHKERRQ(ierr);
//   ierr = DMGlobalToLocal (dm, usr->xphi, INSERT_VALUES, xphilocal); CHKERRQ(ierr);
//   ierr = DMStagVecGetArray(dm, xphilocal, &xxphi); CHKERRQ(ierr);

// // Get dm coordinates array
//   ierr = DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
//   ierr = DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter);CHKERRQ(ierr); 

//   // Loop over local domain
//   for (j = sz; j < sz+nz; j++) {
//     for (i = sx; i <sx+nx; i++) {
//       // C = phi*Cf+(1-phi)*Cs
//       ierr = DMStagGetLocationSlot(dm, ELEMENT, 0, &idx); CHKERRQ(ierr);
//       xx[j][i][idx] = BulkComposition(xxCf[j][i][idx],xxCs[j][i][idx],xxphi[j][i][idx]);
//     }
//   }

//   // Restore arrays
//   ierr = DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);

//   ierr = DMStagVecRestoreArray(dm,xlocal,&xx); CHKERRQ(ierr);
//   ierr = DMStagVecRestoreArray(dm,xCflocal,&xxCf); CHKERRQ(ierr);
//   ierr = DMStagVecRestoreArray(dm,xCslocal,&xxCs); CHKERRQ(ierr);
//   ierr = DMStagVecRestoreArray(dm,xphilocal,&xxphi); CHKERRQ(ierr);

//   ierr = DMRestoreLocalVector(dm,&xCflocal); CHKERRQ(ierr);
//   ierr = DMRestoreLocalVector(dm,&xCslocal); CHKERRQ(ierr);
//   ierr = DMRestoreLocalVector(dm,&xphilocal); CHKERRQ(ierr);

//   ierr = DMLocalToGlobalBegin(dm,xlocal,INSERT_VALUES,usr->xC); CHKERRQ(ierr);
//   ierr = DMLocalToGlobalEnd  (dm,xlocal,INSERT_VALUES,usr->xC); CHKERRQ(ierr);
//   ierr = VecDestroy(&xlocal); CHKERRQ(ierr);

//   PetscFunctionReturn(0);
// }

// // ---------------------------------------
// // UpdateEnthalpy
// // ---------------------------------------
// #undef __FUNCT__
// #define __FUNCT__ "UpdateEnthalpy"
// PetscErrorCode UpdateEnthalpy(void *ctx)
// {
//   UsrData       *usr = (UsrData*) ctx;
//   PetscInt       i, j, sx, sz, nx, nz, Nx, Nz, idx, icenter;
//   PetscScalar    ***xx, ***xxTh, ***xxphi;
//   PetscScalar    **coordx,**coordz;
//   Vec            xlocal, xThlocal, xphilocal;
//   DM             dm;
//   PetscErrorCode ierr;

//   PetscFunctionBegin;

//   dm = usr->dmHC;

//   ierr = DMStagGetGlobalSizes(dm, &Nx, &Nz,NULL);CHKERRQ(ierr);
//   ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

//   ierr = DMCreateLocalVector(dm, &xlocal); CHKERRQ(ierr);
//   ierr = DMStagVecGetArray(dm, xlocal, &xx); CHKERRQ(ierr);

//   ierr = DMGetLocalVector(dm, &xThlocal); CHKERRQ(ierr);
//   ierr = DMGlobalToLocal (dm, usr->xTheta, INSERT_VALUES, xThlocal); CHKERRQ(ierr);
//   ierr = DMStagVecGetArray(dm, xThlocal, &xxTh); CHKERRQ(ierr);

//   ierr = DMGetLocalVector(dm, &xphilocal); CHKERRQ(ierr);
//   ierr = DMGlobalToLocal (dm, usr->xphi, INSERT_VALUES, xphilocal); CHKERRQ(ierr);
//   ierr = DMStagVecGetArray(dm, xphilocal, &xxphi); CHKERRQ(ierr);

// // Get dm coordinates array
//   ierr = DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
//   ierr = DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter);CHKERRQ(ierr); 

//   // Loop over local domain
//   for (j = sz; j < sz+nz; j++) {
//     for (i = sx; i <sx+nx; i++) {
//       PetscScalar Az;
//       Az = -usr->nd->A*coordz[j][icenter]; // check sign of non-dimensional depth!
//       ierr = DMStagGetLocationSlot(dm, ELEMENT, 0, &idx); CHKERRQ(ierr);
//       xx[j][i][idx] = TotalEnthalpy(xxphi[j][i][idx],xxTh[j][i][idx],Az,usr->nd->S,usr->nd->thetaS);
//     }
//   }

//   // Restore arrays
//   ierr = DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);

//   ierr = DMStagVecRestoreArray(dm,xlocal,&xx); CHKERRQ(ierr);
//   ierr = DMStagVecRestoreArray(dm,xThlocal,&xxTh); CHKERRQ(ierr);
//   ierr = DMStagVecRestoreArray(dm,xphilocal,&xxphi); CHKERRQ(ierr);

//   ierr = DMRestoreLocalVector(dm,&xThlocal); CHKERRQ(ierr);
//   ierr = DMRestoreLocalVector(dm,&xphilocal); CHKERRQ(ierr);

//   ierr = DMLocalToGlobalBegin(dm,xlocal,INSERT_VALUES,usr->xH); CHKERRQ(ierr);
//   ierr = DMLocalToGlobalEnd  (dm,xlocal,INSERT_VALUES,usr->xH); CHKERRQ(ierr);
//   ierr = VecDestroy(&xlocal); CHKERRQ(ierr);

//   PetscFunctionReturn(0);
// }