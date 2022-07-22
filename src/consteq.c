#include "consteq.h"

static PetscScalar interp_bilinear(PetscScalar xp, PetscScalar zp, PetscScalar x[2], PetscScalar z[2], PetscScalar qSW, PetscScalar qNW, PetscScalar qSE, PetscScalar qNE)
{ PetscScalar qS, qN,result,dx,dz;
  dx = x[1]-x[0];
  dz = z[1]-z[0];
  qS = (x[1]-xp)/dx*qSW + (xp-x[0])/dx*qSE;
  qN = (x[1]-xp)/dx*qNW + (xp-x[0])/dx*qNE;
  result = (z[1]-zp)/dz*qS + (zp-z[0])/dz*qN;
  return(result);
}

// ---------------------------------------
/*@
DMStagGetPointStrainRates - returns the second invariant of strain rate (epsII) and the strain rate components [exx,ezz,exz]

Use: user
@*/
// ---------------------------------------
PetscErrorCode DMStagGetPointStrainRates(DM dm, Vec x, PetscInt n, DMStagStencil *point, PetscScalar *epsII, PetscScalar *exx, PetscScalar *ezz, PetscScalar *exz)
{
  PetscInt          ix,info[5],i,j,dof0,dof1,dof2,dof3;
  PetscScalar       epsIIs2, eps_xx, eps_zz, eps_xz;
  PetscScalar       **coordx,**coordz;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  // error checking
  ierr = DMStagGetDOF(dm,&dof0,&dof1,&dof2,&dof3);CHKERRQ(ierr);
  if (dof1==0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only calculation of strain rates from face velocities is supported! No face field was detected.");
  if (dof1>1 ) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"More than 1 face field was detected!");

  ierr = DMStagGetGlobalSizes(dm,&info[0],&info[1],NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,DMSTAG_LEFT,&info[2]);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,DMSTAG_RIGHT,&info[3]);CHKERRQ(ierr); 
  ierr = DMStagGetProductCoordinateLocationSlot(dm,DMSTAG_ELEMENT,&info[4]);CHKERRQ(ierr);

  for (ix = 0; ix < n; ix++) {
    i = point[ix].i;
    j = point[ix].j;

    switch (point[ix].loc) {
      case DMSTAG_ELEMENT:
      {
        ierr = get_exx_center(dm,x,coordx,coordz,i,j,info,&eps_xx);CHKERRQ(ierr);
        ierr = get_ezz_center(dm,x,coordx,coordz,i,j,info,&eps_zz);CHKERRQ(ierr);
        ierr = get_exz_center(dm,x,coordx,coordz,i,j,info,&eps_xz);CHKERRQ(ierr);
        break;
      }

      case DMSTAG_DOWN_LEFT:
      {
        ierr = get_exx_corner(dm,x,coordx,coordz,i  ,j  ,info,&eps_xx);CHKERRQ(ierr);
        ierr = get_ezz_corner(dm,x,coordx,coordz,i  ,j  ,info,&eps_zz);CHKERRQ(ierr);
        ierr = get_exz_corner(dm,x,coordx,coordz,i  ,j  ,info,&eps_xz);CHKERRQ(ierr);
        break;
      }

      case DMSTAG_DOWN_RIGHT:
      {
        ierr = get_exx_corner(dm,x,coordx,coordz,i+1,j  ,info,&eps_xx);CHKERRQ(ierr);
        ierr = get_ezz_corner(dm,x,coordx,coordz,i+1,j  ,info,&eps_zz);CHKERRQ(ierr);
        ierr = get_exz_corner(dm,x,coordx,coordz,i+1,j  ,info,&eps_xz);CHKERRQ(ierr);
        break;
      }

      case DMSTAG_UP_LEFT:
      {
        ierr = get_exx_corner(dm,x,coordx,coordz,i  ,j+1,info,&eps_xx);CHKERRQ(ierr);
        ierr = get_ezz_corner(dm,x,coordx,coordz,i  ,j+1,info,&eps_zz);CHKERRQ(ierr);
        ierr = get_exz_corner(dm,x,coordx,coordz,i  ,j+1,info,&eps_xz);CHKERRQ(ierr);
        break;
      }

      case DMSTAG_UP_RIGHT:
      {
        ierr = get_exx_corner(dm,x,coordx,coordz,i+1,j+1,info,&eps_xx);CHKERRQ(ierr);
        ierr = get_ezz_corner(dm,x,coordx,coordz,i+1,j+1,info,&eps_zz);CHKERRQ(ierr);
        ierr = get_exz_corner(dm,x,coordx,coordz,i+1,j+1,info,&eps_xz);CHKERRQ(ierr);
        break;
      }

      default:
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"DMStagStencil location %d is not supported. Point location must be centers or corners.",point[ix].loc);
      break;
    }

    exx[ix] = eps_xx;
    ezz[ix] = eps_zz;
    exz[ix] = eps_xz;

    // Second invariant of strain rate
    epsIIs2 = 0.5*(eps_xx*eps_xx + eps_zz*eps_zz + 2.0*eps_xz*eps_xz);
    epsII[ix] = PetscPowScalar(epsIIs2,0.5);
  }
  ierr = DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode get_exx_center(DM dm, Vec x, PetscScalar **coordx,PetscScalar **coordz, PetscInt i, PetscInt j, PetscInt info[], PetscScalar *_eps)
{
  DMStagStencil  point[2];
  PetscInt       inext, iprev;
  PetscScalar    xx[2], eps = 0.0, dh;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  if ((i<0) || (i==info[0]) || (j<0) || (j==info[1])) {
    *_eps = eps;
    PetscFunctionReturn(0);
  }

  iprev = info[2]; inext = info[3];  
  point[0].i = i; point[0].j = j; point[0].loc = DMSTAG_LEFT;  point[0].c = 0;
  point[1].i = i; point[1].j = j; point[1].loc = DMSTAG_RIGHT; point[1].c = 0;
  ierr = DMStagVecGetValuesStencil(dm,x,2,point,xx); CHKERRQ(ierr);

  dh = coordx[i][inext]-coordx[i][iprev];
  eps = (xx[1]-xx[0])/dh;
  *_eps = eps;
  PetscFunctionReturn(0);
}

PetscErrorCode get_ezz_center(DM dm, Vec x, PetscScalar **coordx,PetscScalar **coordz, PetscInt i, PetscInt j, PetscInt info[], PetscScalar *_eps)
{
  DMStagStencil  point[2];
  PetscInt       inext, iprev;
  PetscScalar    xx[2], eps = 0.0, dh;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  if ((i<0) || (i==info[0]) || (j<0) || (j==info[1])) {
    *_eps = eps;
    PetscFunctionReturn(0);
  }

  iprev = info[2]; inext = info[3];
  point[0].i = i; point[0].j = j; point[0].loc = DMSTAG_DOWN; point[0].c = 0;
  point[1].i = i; point[1].j = j; point[1].loc = DMSTAG_UP;   point[1].c = 0;
  ierr = DMStagVecGetValuesStencil(dm,x,2,point,xx); CHKERRQ(ierr);

  dh = coordz[j][inext]-coordz[j][iprev];
  eps = (xx[1]-xx[0])/dh;
  *_eps = eps;
  PetscFunctionReturn(0);
}

PetscErrorCode get_exz_center(DM dm, Vec x, PetscScalar **coordx,PetscScalar **coordz, PetscInt i, PetscInt j, PetscInt info[], PetscScalar *_eps)
{
  PetscScalar    eps_xz_sw, eps_xz_nw, eps_xz_se, eps_xz_ne;
  PetscScalar    xi[2], zi[2], xp, zp; 
  PetscScalar    eps = 0.0;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = get_exz_corner(dm,x,coordx,coordz,i  ,j  ,info,&eps_xz_sw);CHKERRQ(ierr);
  ierr = get_exz_corner(dm,x,coordx,coordz,i  ,j+1,info,&eps_xz_nw);CHKERRQ(ierr);
  ierr = get_exz_corner(dm,x,coordx,coordz,i+1,j  ,info,&eps_xz_se);CHKERRQ(ierr);
  ierr = get_exz_corner(dm,x,coordx,coordz,i+1,j+1,info,&eps_xz_ne);CHKERRQ(ierr);

  xp   = coordx[i][info[4]];
  zp   = coordz[j][info[4]];
  xi[0] = coordx[i][info[2]]; xi[1] = coordx[i][info[3]];
  zi[0] = coordz[j][info[2]]; zi[1] = coordz[j][info[3]];
  eps = interp_bilinear(xp,zp,xi,zi,eps_xz_sw,eps_xz_nw,eps_xz_se,eps_xz_ne);
  *_eps = eps;
  PetscFunctionReturn(0);
}

PetscErrorCode get_exx_corner(DM dm, Vec x, PetscScalar **coordx,PetscScalar **coordz, PetscInt i, PetscInt j, PetscInt info[], PetscScalar *_eps)
{
  PetscScalar  eps = 0.0, eps_xx_sw, eps_xx_nw, eps_xx_se, eps_xx_ne;
  PetscScalar  xi[2], zi[2], xp, zp;
  PetscInt     Nx, Nz, iprev, inext, icenter;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = get_exx_center(dm,x,coordx,coordz,i-1,j-1,info,&eps_xx_sw);CHKERRQ(ierr);
  ierr = get_exx_center(dm,x,coordx,coordz,i  ,j-1,info,&eps_xx_se);CHKERRQ(ierr);
  ierr = get_exx_center(dm,x,coordx,coordz,i-1,j  ,info,&eps_xx_nw);CHKERRQ(ierr);
  ierr = get_exx_center(dm,x,coordx,coordz,i  ,j  ,info,&eps_xx_ne);CHKERRQ(ierr);
  Nx = info[0]; Nz = info[1]; iprev = info[2]; inext = info[3]; icenter = info[4];

  if (i == Nx) xp = coordx[i-1][inext];
  else         xp = coordx[i  ][iprev];
  if (j == Nz) zp = coordz[j-1][inext];
  else         zp = coordz[j  ][iprev];

  if (i == 0 ) xi[0] = 2.0*coordx[i][iprev]-coordx[i][icenter]; 
  else         xi[0] = coordx[i-1][icenter]; 
  if (i == Nx) xi[1] = 2.0*coordx[i-1][inext]-coordx[i-1][icenter]; 
  else         xi[1] = coordx[i][icenter]; 

  if (j == 0 ) zi[0] = 2.0*coordz[j][iprev]-coordz[j][icenter]; 
  else         zi[0] = coordz[j-1][icenter]; 
  if (j == Nz) zi[1] = 2.0*coordz[j-1][inext]-coordz[j-1][icenter]; 
  else         zi[1] = coordz[j][icenter]; 

  eps = interp_bilinear(xp,zp,xi,zi,eps_xx_sw,eps_xx_nw,eps_xx_se,eps_xx_ne);
  *_eps = eps;
  PetscFunctionReturn(0);
}

PetscErrorCode get_ezz_corner(DM dm, Vec x, PetscScalar **coordx,PetscScalar **coordz, PetscInt i, PetscInt j, PetscInt info[], PetscScalar *_eps)
{
  PetscScalar  eps = 0.0, eps_zz_sw, eps_zz_nw, eps_zz_se, eps_zz_ne;
  PetscScalar  xi[2], zi[2], xp, zp;
  PetscInt     Nx, Nz, iprev, inext, icenter;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = get_ezz_center(dm,x,coordx,coordz,i-1,j-1,info,&eps_zz_sw);CHKERRQ(ierr);
  ierr = get_ezz_center(dm,x,coordx,coordz,i  ,j-1,info,&eps_zz_se);CHKERRQ(ierr);
  ierr = get_ezz_center(dm,x,coordx,coordz,i-1,j  ,info,&eps_zz_nw);CHKERRQ(ierr);
  ierr = get_ezz_center(dm,x,coordx,coordz,i  ,j  ,info,&eps_zz_ne);CHKERRQ(ierr);
  Nx = info[0]; Nz = info[1]; iprev = info[2]; inext = info[3]; icenter = info[4];

  if (i == Nx) xp = coordx[i-1][inext];
  else         xp = coordx[i  ][iprev];
  if (j == Nz) zp = coordz[j-1][inext];
  else         zp = coordz[j  ][iprev];

  if (i == 0 ) xi[0] = 2.0*coordx[i][iprev]-coordx[i][icenter]; 
  else         xi[0] = coordx[i-1][icenter]; 
  if (i == Nx) xi[1] = 2.0*coordx[i-1][inext]-coordx[i-1][icenter]; 
  else         xi[1] = coordx[i][icenter]; 

  if (j == 0 ) zi[0] = 2.0*coordz[j][iprev]-coordz[j][icenter]; 
  else         zi[0] = coordz[j-1][icenter]; 
  if (j == Nz) zi[1] = 2.0*coordz[j-1][inext]-coordz[j-1][icenter]; 
  else         zi[1] = coordz[j][icenter]; 

  eps = interp_bilinear(xp,zp,xi,zi,eps_zz_sw,eps_zz_nw,eps_zz_se,eps_zz_ne);
  *_eps = eps;
  PetscFunctionReturn(0);
}

PetscErrorCode get_exz_corner(DM dm, Vec x, PetscScalar **coordx,PetscScalar **coordz, PetscInt i, PetscInt j, PetscInt info[], PetscScalar *_eps)
{
  DMStagStencil  point[4];
  PetscInt       icenter,iprev, inext, Nx, Nz;
  PetscScalar    xx[4], eps = 0.0, dhx, dhz;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  Nx = info[0]; Nz = info[1]; iprev = info[2]; inext = info[3]; icenter = info[4];
  
  point[0].i = i  ; point[0].j = j-1; point[0].loc = DMSTAG_LEFT; point[0].c = 0;
  point[1].i = i  ; point[1].j = j  ; point[1].loc = DMSTAG_LEFT; point[1].c = 0;
  point[2].i = i-1; point[2].j = j  ; point[2].loc = DMSTAG_DOWN; point[2].c = 0;
  point[3].i = i  ; point[3].j = j  ; point[3].loc = DMSTAG_DOWN; point[3].c = 0;
  
  if (i == 0)  point[2] = point[3];
  if (i == Nx) point[3] = point[2];

  if (j == 0)  point[0] = point[1];
  if (j == Nz) point[1] = point[0];
  ierr = DMStagVecGetValuesStencil(dm,x,4,point,xx); CHKERRQ(ierr);

  if      (i == 0)  dhx = 2.0*(coordx[i][icenter]-coordx[i][iprev]);
  else if (i == Nx) dhx = 2.0*(coordx[i-1][inext]-coordx[i-1][icenter]);
  else              dhx = coordx[i][icenter]-coordx[i-1][icenter];

  if      (j == 0)  dhz = 2.0*(coordz[j][icenter]-coordz[j][iprev]);
  else if (j == Nz) dhz = 2.0*(coordz[j-1][inext]-coordz[j-1][icenter]);
  else              dhz = coordz[j][icenter]-coordz[j-1][icenter];

  eps = 0.5*((xx[1]-xx[0])/dhz + (xx[3]-xx[2])/dhx);
  *_eps = eps;
  PetscFunctionReturn(0);
}