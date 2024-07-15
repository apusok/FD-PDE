#include "morfault.h"

// ---------------------------------------
// FormCoefficient_PV (Stokes-Darcy)
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormCoefficient_PV"
PetscErrorCode FormCoefficient_PV(FDPDE fd, DM dm, Vec x, DM dmcoeff, Vec coeff, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       i, j, ii, sx, sz, nx, nz, iprev, inext, icenter, Nx, Nz;
  PetscScalar    **coordx, **coordz, ***c, ***xx, ***xwt; 
  PetscScalar    ***_Tc, ***_phic, ***_Plith, ***_eps, ***_tauold, ***_DPold, ***_tau, ***_DP;
  PetscScalar     ***_plast, ***_matProp, ***_strain;
  Vec            coefflocal, xlocal, xTlocal, xMPhaselocal, xtaulocal, xDPlocal, xstrainlocal;
  Vec            xphilocal, xPlithlocal, xepslocal, xtauoldlocal, xDPoldlocal, xplastlocal, xmatProplocal;
  PetscScalar    k_hat[4], i_hat[4];
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  // parameters
  k_hat[0] = 0.0;
  k_hat[1] = 0.0;
  k_hat[2] = usr->par->k_hat;
  k_hat[3] = usr->par->k_hat;

  i_hat[0] = 1.0;
  i_hat[1] = 1.0;
  i_hat[2] = 0.0;
  i_hat[3] = 0.0;

  // Get coefficient
  ierr = DMStagGetGlobalSizes(dmcoeff, &Nx, &Nz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmcoeff,LEFT,&iprev);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmcoeff,RIGHT,&inext);CHKERRQ(ierr); 
  ierr = DMStagGetProductCoordinateLocationSlot(dmcoeff,ELEMENT,&icenter);CHKERRQ(ierr);

  // Create coefficient local vector
  ierr = DMCreateLocalVector(dmcoeff, &coefflocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dmcoeff, coefflocal, &c); CHKERRQ(ierr);

  // Strain rates
  // ierr = UpdateStrainRates(dm,x,usr); CHKERRQ(ierr);
  ierr = UpdateStrainRates_Array(dm,x,usr); CHKERRQ(ierr);
  ierr = DMGetLocalVector(usr->dmeps, &xepslocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmeps, usr->xeps, INSERT_VALUES, xepslocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(usr->dmeps,xepslocal,&_eps);CHKERRQ(ierr);

  ierr = DMGetLocalVector(usr->dmeps, &xtauoldlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmeps, usr->xtau_old, INSERT_VALUES, xtauoldlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(usr->dmeps,xtauoldlocal,&_tauold);CHKERRQ(ierr);

  ierr = DMGetLocalVector(usr->dmeps, &xtaulocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmeps, usr->xtau, INSERT_VALUES, xtaulocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(usr->dmeps,xtaulocal,&_tau);CHKERRQ(ierr);

  // Get dm and solution vector
  ierr = DMGetLocalVector(dm, &xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm, x, INSERT_VALUES, xlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(dm,xlocal,&xx);CHKERRQ(ierr);
  
  // Get solution vector for temperature
  ierr = DMGetLocalVector(usr->dmT,&xTlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmT,usr->xT,INSERT_VALUES,xTlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(usr->dmT,xTlocal,&_Tc);CHKERRQ(ierr);

  // Get porosity
  ierr = DMGetLocalVector(usr->dmphi,&xphilocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmphi,usr->xphi,INSERT_VALUES,xphilocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(usr->dmphi,xphilocal,&_phic);CHKERRQ(ierr);

  // Get dm and vector Plith
  ierr = DMGetLocalVector(usr->dmPlith, &xPlithlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmPlith, usr->xPlith, INSERT_VALUES, xPlithlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(usr->dmPlith,xPlithlocal,&_Plith);CHKERRQ(ierr);

  ierr = DMGetLocalVector(usr->dmPlith, &xDPoldlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmPlith, usr->xDP_old, INSERT_VALUES, xDPoldlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(usr->dmPlith,xDPoldlocal,&_DPold);CHKERRQ(ierr);

  ierr = DMGetLocalVector(usr->dmPlith, &xDPlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmPlith, usr->xDP, INSERT_VALUES, xDPlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(usr->dmPlith,xDPlocal,&_DP);CHKERRQ(ierr);

  ierr = DMGetLocalVector(usr->dmPlith, &xplastlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmPlith, usr->xplast, INSERT_VALUES, xplastlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(usr->dmPlith,xplastlocal,&_plast);CHKERRQ(ierr);

  ierr = DMGetLocalVector(usr->dmPlith, &xstrainlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmPlith, usr->xstrain, INSERT_VALUES, xstrainlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(usr->dmPlith,xstrainlocal,&_strain);CHKERRQ(ierr);

  // get material phase fractions
  ierr = DMCreateLocalVector(usr->dmMPhase, &xMPhaselocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(usr->dmMPhase,usr->xMPhase,INSERT_VALUES,xMPhaselocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (usr->dmMPhase,usr->xMPhase,INSERT_VALUES,xMPhaselocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(usr->dmMPhase, xMPhaselocal, &xwt); CHKERRQ(ierr);

  // Get material properties - for output
  ierr = DMGetLocalVector(usr->dmmatProp, &xmatProplocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmmatProp, usr->xmatProp, INSERT_VALUES, xmatProplocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(usr->dmmatProp,xmatProplocal,&_matProp);CHKERRQ(ierr);

  // get location slots
  PetscInt  iE,iP;
  ierr = DMStagGetLocationSlot(usr->dmT,ELEMENT,T_ELEMENT,&iE);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmPlith,ELEMENT,0,&iP);CHKERRQ(ierr);

  PetscInt iPV;
  ierr = DMStagGetLocationSlot(dm,ELEMENT,PV_ELEMENT_P,&iPV);CHKERRQ(ierr);

  PetscInt  e_slot[3],av_slot[4],b_slot[4],d2_slot[4],d3_slot[4],iL,iR,iU,iD,iC,iA,iD1;
  iL = 0; iR = 1; iD = 2; iU  = 3;
  iC = 0; iA = 1; iD1= 2;
  ierr = DMStagGetLocationSlot(dmcoeff,ELEMENT,   PVCOEFF_ELEMENT_C,   &e_slot[iC]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,ELEMENT,   PVCOEFF_ELEMENT_A,   &e_slot[iA]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,ELEMENT,   PVCOEFF_ELEMENT_D1,  &e_slot[iD1]);CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(dmcoeff,DOWN_LEFT, PVCOEFF_VERTEX_A,&av_slot[0]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DOWN_RIGHT,PVCOEFF_VERTEX_A,&av_slot[1]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,UP_LEFT,   PVCOEFF_VERTEX_A,&av_slot[2]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,UP_RIGHT,  PVCOEFF_VERTEX_A,&av_slot[3]);CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(dmcoeff,LEFT,   PVCOEFF_FACE_B,   &b_slot[iL]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,RIGHT,  PVCOEFF_FACE_B,   &b_slot[iR]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DOWN,   PVCOEFF_FACE_B,   &b_slot[iD]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,UP,     PVCOEFF_FACE_B,   &b_slot[iU]);CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(dmcoeff,LEFT,   PVCOEFF_FACE_D2,   &d2_slot[iL]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,RIGHT,  PVCOEFF_FACE_D2,   &d2_slot[iR]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DOWN,   PVCOEFF_FACE_D2,   &d2_slot[iD]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,UP,     PVCOEFF_FACE_D2,   &d2_slot[iU]);CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(dmcoeff,LEFT,   PVCOEFF_FACE_D3,   &d3_slot[iL]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,RIGHT,  PVCOEFF_FACE_D3,   &d3_slot[iR]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DOWN,   PVCOEFF_FACE_D3,   &d3_slot[iD]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,UP,     PVCOEFF_FACE_D3,   &d3_slot[iU]);CHKERRQ(ierr);

  PetscInt iwtc[MAX_MAT_PHASE],iwtl[MAX_MAT_PHASE],iwtr[MAX_MAT_PHASE],iwtd[MAX_MAT_PHASE],iwtu[MAX_MAT_PHASE];
  PetscInt iwtld[MAX_MAT_PHASE],iwtrd[MAX_MAT_PHASE],iwtlu[MAX_MAT_PHASE],iwtru[MAX_MAT_PHASE];
  for (ii = 0; ii < usr->nph; ii++) { 
    ierr = DMStagGetLocationSlot(usr->dmMPhase, ELEMENT,    ii, &iwtc[ii]); CHKERRQ(ierr); 
    ierr = DMStagGetLocationSlot(usr->dmMPhase, LEFT,       ii, &iwtl[ii]); CHKERRQ(ierr);
    ierr = DMStagGetLocationSlot(usr->dmMPhase, RIGHT,      ii, &iwtr[ii]); CHKERRQ(ierr);
    ierr = DMStagGetLocationSlot(usr->dmMPhase, DOWN,       ii, &iwtd[ii]); CHKERRQ(ierr); 
    ierr = DMStagGetLocationSlot(usr->dmMPhase, UP,         ii, &iwtu[ii]); CHKERRQ(ierr);
    ierr = DMStagGetLocationSlot(usr->dmMPhase, DOWN_LEFT,  ii, &iwtld[ii]); CHKERRQ(ierr);
    ierr = DMStagGetLocationSlot(usr->dmMPhase, DOWN_RIGHT, ii, &iwtrd[ii]); CHKERRQ(ierr);
    ierr = DMStagGetLocationSlot(usr->dmMPhase, UP_LEFT,    ii, &iwtlu[ii]); CHKERRQ(ierr);
    ierr = DMStagGetLocationSlot(usr->dmMPhase, UP_RIGHT,   ii, &iwtru[ii]); CHKERRQ(ierr);
  }

  PetscInt ixx, izz, ixz, iII, ixxn[4], izzn[4], ixzn[4], iIIn[4];
  ierr = DMStagGetLocationSlot(usr->dmeps, ELEMENT, 0, &ixx); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, ELEMENT, 1, &izz); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, ELEMENT, 2, &ixz); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, ELEMENT, 3, &iII); CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(usr->dmeps, DOWN_LEFT, 0, &ixxn[0]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, DOWN_LEFT, 1, &izzn[0]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, DOWN_LEFT, 2, &ixzn[0]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, DOWN_LEFT, 3, &iIIn[0]); CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(usr->dmeps, DOWN_RIGHT, 0, &ixxn[1]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, DOWN_RIGHT, 1, &izzn[1]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, DOWN_RIGHT, 2, &ixzn[1]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, DOWN_RIGHT, 3, &iIIn[1]); CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(usr->dmeps, UP_LEFT, 0, &ixxn[2]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, UP_LEFT, 1, &izzn[2]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, UP_LEFT, 2, &ixzn[2]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, UP_LEFT, 3, &iIIn[2]); CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(usr->dmeps, UP_RIGHT, 0, &ixxn[3]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, UP_RIGHT, 1, &izzn[3]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, UP_RIGHT, 2, &ixzn[3]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, UP_RIGHT, 3, &iIIn[3]); CHKERRQ(ierr);

  PetscInt imat[MATPROP_NPROP];
  for (ii = 0; ii < MATPROP_NPROP; ii++) { ierr = DMStagGetLocationSlot(usr->dmmatProp, ELEMENT, ii, &imat[ii]); CHKERRQ(ierr); }
  
  // Loop over local domain - set initial density and viscosity
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      PetscScalar   phic[9], pc[9], Plc[9], Tc[9], DPoldc[9], strain[9], dx, dz;
      PetscInt      iph, ii, im, jm, ip, jp;

      if (i == 0   ) im = i; else im = i-1;
      if (i == Nx-1) ip = i; else ip = i+1;
      if (j == 0   ) jm = j; else jm = j-1;
      if (j == Nz-1) jp = j; else jp = j+1;

      // should be adapted to variable spacing
      dx = coordx[i][inext]-coordx[i][iprev];
      dz = coordz[j][inext]-coordz[j][iprev];

      // get porosity, p, Plith, T - center
      ierr = Get9PointCenterValues(i,j,iE,Nx,Nz,_phic,phic);CHKERRQ(ierr);
      ierr = Get9PointCenterValues(i,j,iP,Nx,Nz,_Plith,Plc);CHKERRQ(ierr);
      ierr = Get9PointCenterValues(i,j,iPV,Nx,Nz,xx,pc);CHKERRQ(ierr);
      ierr = Get9PointCenterValues(i,j,iE,Nx,Nz,_Tc,Tc);CHKERRQ(ierr);
      ierr = Get9PointCenterValues(i,j,iP,Nx,Nz,_DPold,DPoldc);CHKERRQ(ierr);
      ierr = Get9PointCenterValues(i,j,iP,Nx,Nz,_strain,strain);CHKERRQ(ierr);

      // correct for liquid porosity
      for (ii = 0; ii < 9; ii++) { 
        // phic[ii] = usr->par->phi0;
        phic[ii] = 1.0 - phic[ii]; 
        if (phic[ii]<0.0) phic[ii] = 0.0;
      }

      // Prepare for pointwise rheology calculation
      PetscScalar eta_eff[9], zeta_eff[9], chis[9], chip[9], txx[9], tzz[9], txz[9], tII[9], DP[9], dotlam[9];
      PetscScalar eta_v[9],eta_e[9],eta_p[9],zeta_v[9],zeta_e[9],zeta_p[9];
      PetscScalar e[4], t[4], P[4], res[21], Z[9], G[9], C[9], sigmat[9], theta[9];
      PetscInt ix[4];

      // center points
      ix[0] = ixx; ix[1] = izz; ix[2] = ixz; ix[3] = iII;

      P[0] = pc[0]; P[1] = Plc[0]; P[2] = DPoldc[0]; P[3] = strain[0];
      ierr = GetTensorPointValues(i,j,ix,_eps,e);CHKERRQ(ierr);
      ierr = GetTensorPointValues(i,j,ix,_tauold,t);CHKERRQ(ierr);
      ierr = RheologyPointwise(i,j,xwt,iwtc,Tc[0],phic[0],P,e,t,ix,res,usr);CHKERRQ(ierr);
      ierr = DecompactRheologyVars(0,res,eta_eff,eta_v,eta_e,eta_p,zeta_eff,zeta_v,zeta_e,zeta_p,chis,chip,txx,tzz,txz,tII,DP,dotlam,Z,G,C,sigmat,theta);CHKERRQ(ierr);

      P[0] = pc[1]; P[1] = Plc[1]; P[2] = DPoldc[1]; P[3] = strain[1];
      ierr = GetTensorPointValues(im,j,ix,_eps,e);CHKERRQ(ierr);
      ierr = GetTensorPointValues(im,j,ix,_tauold,t);CHKERRQ(ierr);
      ierr = RheologyPointwise(im,j,xwt,iwtc,Tc[1],phic[1],P,e,t,ix,res,usr);CHKERRQ(ierr);
      ierr = DecompactRheologyVars(1,res,eta_eff,eta_v,eta_e,eta_p,zeta_eff,zeta_v,zeta_e,zeta_p,chis,chip,txx,tzz,txz,tII,DP,dotlam,Z,G,C,sigmat,theta);CHKERRQ(ierr);

      P[0] = pc[2]; P[1] = Plc[2]; P[2] = DPoldc[2]; P[3] = strain[2];
      ierr = GetTensorPointValues(ip,j,ix,_eps,e);CHKERRQ(ierr);
      ierr = GetTensorPointValues(ip,j,ix,_tauold,t);CHKERRQ(ierr);
      ierr = RheologyPointwise(ip,j,xwt,iwtc,Tc[2],phic[2],P,e,t,ix,res,usr);CHKERRQ(ierr);
      ierr = DecompactRheologyVars(2,res,eta_eff,eta_v,eta_e,eta_p,zeta_eff,zeta_v,zeta_e,zeta_p,chis,chip,txx,tzz,txz,tII,DP,dotlam,Z,G,C,sigmat,theta);CHKERRQ(ierr);

      P[0] = pc[3]; P[1] = Plc[3]; P[2] = DPoldc[3]; P[3] = strain[3];
      ierr = GetTensorPointValues(i,jm,ix,_eps,e);CHKERRQ(ierr);
      ierr = GetTensorPointValues(i,jm,ix,_tauold,t);CHKERRQ(ierr);
      ierr = RheologyPointwise(i,jm,xwt,iwtc,Tc[3],phic[3],P,e,t,ix,res,usr);CHKERRQ(ierr);
      ierr = DecompactRheologyVars(3,res,eta_eff,eta_v,eta_e,eta_p,zeta_eff,zeta_v,zeta_e,zeta_p,chis,chip,txx,tzz,txz,tII,DP,dotlam,Z,G,C,sigmat,theta);CHKERRQ(ierr);

      P[0] = pc[4]; P[1] = Plc[4]; P[2] = DPoldc[4]; P[3] = strain[4];
      ierr = GetTensorPointValues(i,jp,ix,_eps,e);CHKERRQ(ierr);
      ierr = GetTensorPointValues(i,jp,ix,_tauold,t);CHKERRQ(ierr);
      ierr = RheologyPointwise(i,jp,xwt,iwtc,Tc[4],phic[4],P,e,t,ix,res,usr);CHKERRQ(ierr);
      ierr = DecompactRheologyVars(4,res,eta_eff,eta_v,eta_e,eta_p,zeta_eff,zeta_v,zeta_e,zeta_p,chis,chip,txx,tzz,txz,tII,DP,dotlam,Z,G,C,sigmat,theta);CHKERRQ(ierr);

      // corner points
      PetscScalar Tcorner[4], phicorner[4], pcorner[4], Plcorner[4], DPoldcorner[4], straincorner[4];
      ierr = GetCornerAvgFromCenter(Tc,Tcorner);CHKERRQ(ierr);
      ierr = GetCornerAvgFromCenter(phic,phicorner);CHKERRQ(ierr);
      ierr = GetCornerAvgFromCenter(pc,pcorner);CHKERRQ(ierr);
      ierr = GetCornerAvgFromCenter(Plc,Plcorner);CHKERRQ(ierr);
      ierr = GetCornerAvgFromCenter(DPoldc,DPoldcorner);CHKERRQ(ierr);
      ierr = GetCornerAvgFromCenter(strain,straincorner);CHKERRQ(ierr);
      
      ii = 0; ix[0] = ixxn[ii]; ix[1] = izzn[ii]; ix[2] = ixzn[ii]; ix[3] = iIIn[ii]; P[0] = pcorner[ii]; P[1] = Plcorner[ii]; P[2] = DPoldcorner[ii]; P[3] = straincorner[ii];
      ierr = GetTensorPointValues(i,j,ix,_eps,e);CHKERRQ(ierr);
      ierr = GetTensorPointValues(i,j,ix,_tauold,t);CHKERRQ(ierr);
      ierr = RheologyPointwise(i,j,xwt,iwtld,Tcorner[0],phicorner[0],P,e,t,ix,res,usr);CHKERRQ(ierr);
      ierr = DecompactRheologyVars(5,res,eta_eff,eta_v,eta_e,eta_p,zeta_eff,zeta_v,zeta_e,zeta_p,chis,chip,txx,tzz,txz,tII,DP,dotlam,Z,G,C,sigmat,theta);CHKERRQ(ierr);

      ii = 1; ix[0] = ixxn[ii]; ix[1] = izzn[ii]; ix[2] = ixzn[ii]; ix[3] = iIIn[ii]; P[0] = pcorner[ii]; P[1] = Plcorner[ii]; P[2] = DPoldcorner[ii]; P[3] = straincorner[ii];
      ierr = GetTensorPointValues(i,j,ix,_eps,e);CHKERRQ(ierr);
      ierr = GetTensorPointValues(i,j,ix,_tauold,t);CHKERRQ(ierr);
      ierr = RheologyPointwise(i,j,xwt,iwtrd,Tcorner[1],phicorner[1],P,e,t,ix,res,usr);CHKERRQ(ierr);
      ierr = DecompactRheologyVars(6,res,eta_eff,eta_v,eta_e,eta_p,zeta_eff,zeta_v,zeta_e,zeta_p,chis,chip,txx,tzz,txz,tII,DP,dotlam,Z,G,C,sigmat,theta);CHKERRQ(ierr);

      ii = 2; ix[0] = ixxn[ii]; ix[1] = izzn[ii]; ix[2] = ixzn[ii]; ix[3] = iIIn[ii]; P[0] = pcorner[ii]; P[1] = Plcorner[ii]; P[2] = DPoldcorner[ii]; P[3] = straincorner[ii];
      ierr = GetTensorPointValues(i,j,ix,_eps,e);CHKERRQ(ierr);
      ierr = GetTensorPointValues(i,j,ix,_tauold,t);CHKERRQ(ierr);
      ierr = RheologyPointwise(i,j,xwt,iwtlu,Tcorner[2],phicorner[2],P,e,t,ix,res,usr);CHKERRQ(ierr);
      ierr = DecompactRheologyVars(7,res,eta_eff,eta_v,eta_e,eta_p,zeta_eff,zeta_v,zeta_e,zeta_p,chis,chip,txx,tzz,txz,tII,DP,dotlam,Z,G,C,sigmat,theta);CHKERRQ(ierr);

      ii = 3; ix[0] = ixxn[ii]; ix[1] = izzn[ii]; ix[2] = ixzn[ii]; ix[3] = iIIn[ii]; P[0] = pcorner[ii]; P[1] = Plcorner[ii]; P[2] = DPoldcorner[ii]; P[3] = straincorner[ii];
      ierr = GetTensorPointValues(i,j,ix,_eps,e);CHKERRQ(ierr);
      ierr = GetTensorPointValues(i,j,ix,_tauold,t);CHKERRQ(ierr);
      ierr = RheologyPointwise(i,j,xwt,iwtru,Tcorner[3],phicorner[3],P,e,t,ix,res,usr);CHKERRQ(ierr);
      ierr = DecompactRheologyVars(8,res,eta_eff,eta_v,eta_e,eta_p,zeta_eff,zeta_v,zeta_e,zeta_p,chis,chip,txx,tzz,txz,tII,DP,dotlam,Z,G,C,sigmat,theta);CHKERRQ(ierr);

      { // element 
        // C = 0 (center, c=0)
        c[j][i][e_slot[iC]] = 0.0;

        // A = eta_eff,phi (center, c=1)
        c[j][i][e_slot[iA]] = eta_eff[0];

        // D1 = zeta_eff-2/3eta_eff (center, c=2)
        c[j][i][e_slot[iD1]] = zeta_eff[0]-2.0/3.0*eta_eff[0];
      }

      { // corner
        // A = eta_eff,phi (corner, c=0)
        for (ii = 0; ii < 4; ii++) {
          c[j][i][av_slot[ii]] = eta_eff[5+ii];
        }
      }

      { // face
        PetscScalar   phi[4], K[4], B[4], D2[4], D3[4];
        PetscScalar   divchitau[4],gradchidp[4], gradPlith[4];
        PetscScalar   rhof[4], rhos[4], rho[4], wt[MAX_MAT_PHASE], rho0[MAX_MAT_PHASE];
        PetscInt      idx[MAX_MAT_PHASE];


        // porosity on edges - not for variable grid spacing
        phi[0] = (phic[0]+phic[1])*0.5;
        phi[1] = (phic[0]+phic[2])*0.5;
        phi[2] = (phic[0]+phic[3])*0.5;
        phi[3] = (phic[0]+phic[4])*0.5;

        gradPlith[0] = (Plc[0]-Plc[1])/dx;
        gradPlith[1] = (Plc[2]-Plc[0])/dx;
        gradPlith[2] = (Plc[0]-Plc[3])/dz;
        gradPlith[3] = (Plc[4]-Plc[0])/dz;

        // // on boundaries assume same gradient as in previous cell
        // if (i==Nx-1) gradPlith[1] = gradPlith[0]; 
        // if (i==0   ) gradPlith[0] = gradPlith[1]; 
        // if (j==Nz-1) gradPlith[3] = gradPlith[2]; 
        // if (j==0   ) gradPlith[2] = gradPlith[3]; 

        gradchidp[0] = (chip[0]*_DPold[j ][i ][iP] - chip[1]*_DPold[j ][im][iP])/dx;
        gradchidp[1] = (chip[2]*_DPold[j ][ip][iP] - chip[0]*_DPold[j ][i ][iP])/dx;
        gradchidp[2] = (chip[0]*_DPold[j ][i ][iP] - chip[3]*_DPold[jm][i ][iP])/dz;
        gradchidp[3] = (chip[4]*_DPold[jp][i ][iP] - chip[0]*_DPold[j ][i ][iP])/dz;

        //  div(chis*tau_old) = div(S) = [dSxx/dx+dSxz/dz, dSzx/dx+dSzz/dz]
        divchitau[0] = (chis[0]*_tauold[j][i ][ixx] - chis[1]*_tauold[j][im][ixx])/dx + (chis[7]*_tauold[j][i][ixzn[2]]-chis[5]*_tauold[j][i][ixzn[0]])/dz;
        divchitau[1] = (chis[2]*_tauold[j][ip][ixx] - chis[0]*_tauold[j][i ][ixx])/dx + (chis[8]*_tauold[j][i][ixzn[3]]-chis[6]*_tauold[j][i][ixzn[1]])/dz;
        divchitau[2] = (chis[6]*_tauold[j][i][ixzn[1]]-chis[5]*_tauold[j][i][ixzn[0]])/dx + (chis[0]*_tauold[j ][i][izz] - chis[3]*_tauold[jm][i][izz])/dz;
        divchitau[3] = (chis[8]*_tauold[j][i][ixzn[3]]-chis[7]*_tauold[j][i][ixzn[2]])/dx + (chis[4]*_tauold[jp][i][izz] - chis[0]*_tauold[j ][i][izz])/dz;

        for (ii = 0; ii < 4; ii++) {
          K[ii]  = Permeability(phi[ii],usr->par->n); // assumed uniform for all materials

          // get material phase fraction
          if (ii == 0 ) { idx[0] = iwtl[0]; idx[1] = iwtl[1]; idx[2] = iwtl[2]; idx[3] = iwtl[3]; idx[4] = iwtl[4]; idx[5] = iwtl[5]; }
          if (ii == 1 ) { idx[0] = iwtr[0]; idx[1] = iwtr[1]; idx[2] = iwtr[2]; idx[3] = iwtr[3]; idx[4] = iwtr[4]; idx[5] = iwtr[5];}
          if (ii == 2 ) { idx[0] = iwtd[0]; idx[1] = iwtd[1]; idx[2] = iwtd[2]; idx[3] = iwtd[3]; idx[4] = iwtd[4]; idx[5] = iwtd[5];}
          if (ii == 3 ) { idx[0] = iwtu[0]; idx[1] = iwtu[1]; idx[2] = iwtu[2]; idx[3] = iwtu[3]; idx[4] = iwtu[4]; idx[5] = iwtu[5];}

          // get bulk density
          for (iph = 0; iph < usr->nph; iph++) { rho0[iph] = Density(usr->mat_nd[iph].rho0,usr->mat[iph].rho_func); }
          ierr = GetMatPhaseFraction(i,j,xwt,idx,usr->nph,wt); CHKERRQ(ierr);
          rhos[ii] = WeightAverageValue(rho0,wt,usr->nph); 
          rhof[ii] = usr->nd->rhof;
          rho[ii]  = Mixture(rhos[ii],rhof[ii],phi[ii]);

          // add buoyancy terms here
          B[ii] = -divchitau[ii]+gradchidp[ii] + gradPlith[ii] + rho[ii]*k_hat[ii];
          // B[ii] = -divchitau[ii]+gradchidp[ii] + gradPlith[ii]*i_hat[ii] - phi[ii]*k_hat[ii]; 
          D2[ii] = -K[ii]*usr->nd->R*usr->nd->R;
          D3[ii] = -K[ii]*usr->nd->R*usr->nd->R*(gradPlith[ii]*i_hat[ii] - k_hat[ii]);

          // B = body force+elasticity (edges, c=0)
          c[j][i][b_slot[ii]] = B[ii];

          // D2 = -R^2*Kphi (edges, c=1)
          c[j][i][d2_slot[ii]] = D2[ii];

          // D3 = -R^2*Kphi*(grad(Plith)-rho_ell/drho*k_hat) (edges, c=2)
          c[j][i][d3_slot[ii]] = D3[ii];
        }
      }

      // save stresses for output + dotlam
      _tau[j][i][ixx]     = txx[0]; _tau[j][i][izz]     = tzz[0]; _tau[j][i][ixz]     = txz[0]; _tau[j][i][iII]     = tII[0];
      _tau[j][i][ixxn[0]] = txx[5]; _tau[j][i][izzn[0]] = tzz[5]; _tau[j][i][ixzn[0]] = txz[5]; _tau[j][i][iIIn[0]] = tII[5];
      _tau[j][i][ixxn[1]] = txx[6]; _tau[j][i][izzn[1]] = tzz[6]; _tau[j][i][ixzn[1]] = txz[6]; _tau[j][i][iIIn[1]] = tII[6];
      _tau[j][i][ixxn[2]] = txx[7]; _tau[j][i][izzn[2]] = tzz[7]; _tau[j][i][ixzn[2]] = txz[7]; _tau[j][i][iIIn[2]] = tII[7];
      _tau[j][i][ixxn[3]] = txx[8]; _tau[j][i][izzn[3]] = tzz[8]; _tau[j][i][ixzn[3]] = txz[8]; _tau[j][i][iIIn[3]] = tII[8];
      _DP[j][i][iP]    = DP[0];
      _plast[j][i][iP] = dotlam[0];

      // get density (center)
      PetscScalar rho0[MAX_MAT_PHASE],rho,wt[MAX_MAT_PHASE];
      for (iph = 0; iph < usr->nph; iph++) { rho0[iph] = Density(usr->mat_nd[iph].rho0,usr->mat[iph].rho_func); }
      ierr = GetMatPhaseFraction(i,j,xwt,iwtc,usr->nph,wt); CHKERRQ(ierr);
      rho  = WeightAverageValue(rho0,wt,usr->nph);

      // get permeability (center)
      PetscScalar Kphi;
      Kphi  = Permeability(phic[0],usr->par->n);

      // save material properties for output
      _matProp[j][i][imat[MATPROP_ELEMENT_ETA]]    = eta_eff[0];
      _matProp[j][i][imat[MATPROP_ELEMENT_ETA_V]]  = eta_v[0];
      _matProp[j][i][imat[MATPROP_ELEMENT_ETA_E]]  = eta_e[0];
      _matProp[j][i][imat[MATPROP_ELEMENT_ETA_P]]  = eta_p[0];
      _matProp[j][i][imat[MATPROP_ELEMENT_ZETA]]   = zeta_eff[0];
      _matProp[j][i][imat[MATPROP_ELEMENT_ZETA_V]] = zeta_v[0];
      _matProp[j][i][imat[MATPROP_ELEMENT_ZETA_E]] = zeta_e[0];
      _matProp[j][i][imat[MATPROP_ELEMENT_ZETA_P]] = zeta_p[0];
      _matProp[j][i][imat[MATPROP_ELEMENT_Z]]      = Z[0];
      _matProp[j][i][imat[MATPROP_ELEMENT_G]]      = G[0];
      _matProp[j][i][imat[MATPROP_ELEMENT_C]]      = C[0];
      _matProp[j][i][imat[MATPROP_ELEMENT_SIGMAT]] = sigmat[0];
      _matProp[j][i][imat[MATPROP_ELEMENT_THETA]]  = theta[0];
      _matProp[j][i][imat[MATPROP_ELEMENT_RHO]]    = rho;
      _matProp[j][i][imat[MATPROP_ELEMENT_KPHI]]   = Kphi;
      _matProp[j][i][imat[MATPROP_ELEMENT_CHIP]]   = chip[0];
      _matProp[j][i][imat[MATPROP_ELEMENT_CHIS]]   = chis[0];
    }
  }

  // Restore arrays, local vectors
  ierr = DMStagRestoreProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArray(dmcoeff,coefflocal,&c);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  ierr = VecDestroy(&coefflocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArrayRead(dm,xlocal,&xx);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&xlocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArrayRead(usr->dmeps,xepslocal,&_eps);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmeps,&xepslocal); CHKERRQ(ierr);
  ierr = DMStagVecRestoreArrayRead(usr->dmeps,xtauoldlocal,&_tauold);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmeps,&xtauoldlocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArray(usr->dmeps,xtaulocal,&_tau);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(usr->dmeps,xtaulocal,INSERT_VALUES,usr->xtau); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (usr->dmeps,xtaulocal,INSERT_VALUES,usr->xtau); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmeps,&xtaulocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArrayRead(usr->dmT,xTlocal,&_Tc);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmT,&xTlocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArrayRead(usr->dmphi,xphilocal,&_phic);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmphi,&xphilocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArrayRead(usr->dmPlith,xPlithlocal,&_Plith);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmPlith,&xPlithlocal); CHKERRQ(ierr);
  ierr = DMStagVecRestoreArrayRead(usr->dmPlith,xDPoldlocal,&_DPold);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmPlith,&xDPoldlocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArray(usr->dmPlith,xDPlocal,&_DP);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(usr->dmPlith,xDPlocal,INSERT_VALUES,usr->xDP); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (usr->dmPlith,xDPlocal,INSERT_VALUES,usr->xDP); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmPlith,&xDPlocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArray(usr->dmPlith,xplastlocal,&_plast);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(usr->dmPlith,xplastlocal,INSERT_VALUES,usr->xplast); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (usr->dmPlith,xplastlocal,INSERT_VALUES,usr->xplast); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmPlith,&xplastlocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArrayRead(usr->dmPlith,xstrainlocal,&_strain);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmPlith,&xstrainlocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArray(usr->dmmatProp,xmatProplocal,&_matProp);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(usr->dmmatProp,xmatProplocal,INSERT_VALUES,usr->xmatProp); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (usr->dmmatProp,xmatProplocal,INSERT_VALUES,usr->xmatProp); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmmatProp,&xmatProplocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArray(usr->dmMPhase,xMPhaselocal,&xwt);CHKERRQ(ierr);
  ierr = VecDestroy(&xMPhaselocal); CHKERRQ(ierr);
 
  PetscFunctionReturn(0);
}

// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormCoefficient_PV_DPL"
PetscErrorCode FormCoefficient_PV_DPL(FDPDE fd, DM dm, Vec x, DM dmcoeff, Vec coeff, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       i, j, ii, sx, sz, nx, nz, iprev, inext, icenter, Nx, Nz;
  PetscScalar    **coordx, **coordz, ***c, ***xx, ***xwt; 
  PetscScalar    ***_Tc, ***_phic, ***_Plith, ***_eps, ***_tauold, ***_DPold, ***_tau, ***_DP;
  PetscScalar     ***_plast, ***_matProp, ***_strain;
  Vec            coefflocal, xlocal, xTlocal, xMPhaselocal, xtaulocal, xDPlocal, xstrainlocal;
  Vec            xphilocal, xPlithlocal, xepslocal, xtauoldlocal, xDPoldlocal, xplastlocal, xmatProplocal;
  PetscScalar    k_hat[4], i_hat[4];
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  // parameters
  k_hat[0] = 0.0;
  k_hat[1] = 0.0;
  k_hat[2] = usr->par->k_hat;
  k_hat[3] = usr->par->k_hat;

  i_hat[0] = 1.0;
  i_hat[1] = 1.0;
  i_hat[2] = 0.0;
  i_hat[3] = 0.0;

  // Get coefficient
  ierr = DMStagGetGlobalSizes(dmcoeff, &Nx, &Nz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmcoeff,LEFT,&iprev);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmcoeff,RIGHT,&inext);CHKERRQ(ierr); 
  ierr = DMStagGetProductCoordinateLocationSlot(dmcoeff,ELEMENT,&icenter);CHKERRQ(ierr);

  // Create coefficient local vector
  ierr = DMCreateLocalVector(dmcoeff, &coefflocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dmcoeff, coefflocal, &c); CHKERRQ(ierr);

  // Strain rates
  // ierr = UpdateStrainRates(dm,x,usr); CHKERRQ(ierr);
  ierr = UpdateStrainRates_Array(dm,x,usr); CHKERRQ(ierr);
  ierr = DMGetLocalVector(usr->dmeps, &xepslocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmeps, usr->xeps, INSERT_VALUES, xepslocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(usr->dmeps,xepslocal,&_eps);CHKERRQ(ierr);

  ierr = DMGetLocalVector(usr->dmeps, &xtauoldlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmeps, usr->xtau_old, INSERT_VALUES, xtauoldlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(usr->dmeps,xtauoldlocal,&_tauold);CHKERRQ(ierr);

  ierr = DMGetLocalVector(usr->dmeps, &xtaulocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmeps, usr->xtau, INSERT_VALUES, xtaulocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(usr->dmeps,xtaulocal,&_tau);CHKERRQ(ierr);

  // Get dm and solution vector
  ierr = DMGetLocalVector(dm, &xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm, x, INSERT_VALUES, xlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(dm,xlocal,&xx);CHKERRQ(ierr);
  
  // Get solution vector for temperature
  ierr = DMGetLocalVector(usr->dmT,&xTlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmT,usr->xT,INSERT_VALUES,xTlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(usr->dmT,xTlocal,&_Tc);CHKERRQ(ierr);

  // Get porosity
  ierr = DMGetLocalVector(usr->dmphi,&xphilocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmphi,usr->xphi,INSERT_VALUES,xphilocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(usr->dmphi,xphilocal,&_phic);CHKERRQ(ierr);

  // Get dm and vector Plith
  ierr = DMGetLocalVector(usr->dmPlith, &xPlithlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmPlith, usr->xPlith, INSERT_VALUES, xPlithlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(usr->dmPlith,xPlithlocal,&_Plith);CHKERRQ(ierr);

  ierr = DMGetLocalVector(usr->dmPlith, &xDPoldlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmPlith, usr->xDP_old, INSERT_VALUES, xDPoldlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(usr->dmPlith,xDPoldlocal,&_DPold);CHKERRQ(ierr);

  ierr = DMGetLocalVector(usr->dmPlith, &xDPlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmPlith, usr->xDP, INSERT_VALUES, xDPlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(usr->dmPlith,xDPlocal,&_DP);CHKERRQ(ierr);

  ierr = DMGetLocalVector(usr->dmPlith, &xplastlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmPlith, usr->xplast, INSERT_VALUES, xplastlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(usr->dmPlith,xplastlocal,&_plast);CHKERRQ(ierr);

  ierr = DMGetLocalVector(usr->dmPlith, &xstrainlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmPlith, usr->xstrain, INSERT_VALUES, xstrainlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(usr->dmPlith,xstrainlocal,&_strain);CHKERRQ(ierr);

  // get material phase fractions
  ierr = DMCreateLocalVector(usr->dmMPhase, &xMPhaselocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(usr->dmMPhase,usr->xMPhase,INSERT_VALUES,xMPhaselocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (usr->dmMPhase,usr->xMPhase,INSERT_VALUES,xMPhaselocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(usr->dmMPhase, xMPhaselocal, &xwt); CHKERRQ(ierr);

  // Get material properties - for output
  ierr = DMGetLocalVector(usr->dmmatProp, &xmatProplocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmmatProp, usr->xmatProp, INSERT_VALUES, xmatProplocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(usr->dmmatProp,xmatProplocal,&_matProp);CHKERRQ(ierr);

  // get location slots
  PetscInt  iE,iP;
  ierr = DMStagGetLocationSlot(usr->dmT,ELEMENT,T_ELEMENT,&iE);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmPlith,ELEMENT,0,&iP);CHKERRQ(ierr);

  PetscInt iPV;
  ierr = DMStagGetLocationSlot(dm,ELEMENT,PV_ELEMENT_P,&iPV);CHKERRQ(ierr);

  PetscInt  e_slot[3],av_slot[4],b_slot[4],d2_slot[4],d3_slot[4],iL,iR,iU,iD,iC,iA,iD1;
  iL = 0; iR = 1; iD = 2; iU  = 3;
  iC = 0; iA = 1; iD1= 2;
  ierr = DMStagGetLocationSlot(dmcoeff,ELEMENT,   PVCOEFF_ELEMENT_C,   &e_slot[iC]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,ELEMENT,   PVCOEFF_ELEMENT_A,   &e_slot[iA]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,ELEMENT,   PVCOEFF_ELEMENT_D1,  &e_slot[iD1]);CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(dmcoeff,DOWN_LEFT, PVCOEFF_VERTEX_A,&av_slot[0]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DOWN_RIGHT,PVCOEFF_VERTEX_A,&av_slot[1]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,UP_LEFT,   PVCOEFF_VERTEX_A,&av_slot[2]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,UP_RIGHT,  PVCOEFF_VERTEX_A,&av_slot[3]);CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(dmcoeff,LEFT,   PVCOEFF_FACE_B,   &b_slot[iL]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,RIGHT,  PVCOEFF_FACE_B,   &b_slot[iR]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DOWN,   PVCOEFF_FACE_B,   &b_slot[iD]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,UP,     PVCOEFF_FACE_B,   &b_slot[iU]);CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(dmcoeff,LEFT,   PVCOEFF_FACE_D2,   &d2_slot[iL]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,RIGHT,  PVCOEFF_FACE_D2,   &d2_slot[iR]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DOWN,   PVCOEFF_FACE_D2,   &d2_slot[iD]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,UP,     PVCOEFF_FACE_D2,   &d2_slot[iU]);CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(dmcoeff,LEFT,   PVCOEFF_FACE_D3,   &d3_slot[iL]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,RIGHT,  PVCOEFF_FACE_D3,   &d3_slot[iR]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DOWN,   PVCOEFF_FACE_D3,   &d3_slot[iD]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,UP,     PVCOEFF_FACE_D3,   &d3_slot[iU]);CHKERRQ(ierr);

  PetscInt iwtc[MAX_MAT_PHASE],iwtl[MAX_MAT_PHASE],iwtr[MAX_MAT_PHASE],iwtd[MAX_MAT_PHASE],iwtu[MAX_MAT_PHASE];
  PetscInt iwtld[MAX_MAT_PHASE],iwtrd[MAX_MAT_PHASE],iwtlu[MAX_MAT_PHASE],iwtru[MAX_MAT_PHASE];
  for (ii = 0; ii < usr->nph; ii++) { 
    ierr = DMStagGetLocationSlot(usr->dmMPhase, ELEMENT,    ii, &iwtc[ii]); CHKERRQ(ierr); 
    ierr = DMStagGetLocationSlot(usr->dmMPhase, LEFT,       ii, &iwtl[ii]); CHKERRQ(ierr);
    ierr = DMStagGetLocationSlot(usr->dmMPhase, RIGHT,      ii, &iwtr[ii]); CHKERRQ(ierr);
    ierr = DMStagGetLocationSlot(usr->dmMPhase, DOWN,       ii, &iwtd[ii]); CHKERRQ(ierr); 
    ierr = DMStagGetLocationSlot(usr->dmMPhase, UP,         ii, &iwtu[ii]); CHKERRQ(ierr);
    ierr = DMStagGetLocationSlot(usr->dmMPhase, DOWN_LEFT,  ii, &iwtld[ii]); CHKERRQ(ierr);
    ierr = DMStagGetLocationSlot(usr->dmMPhase, DOWN_RIGHT, ii, &iwtrd[ii]); CHKERRQ(ierr);
    ierr = DMStagGetLocationSlot(usr->dmMPhase, UP_LEFT,    ii, &iwtlu[ii]); CHKERRQ(ierr);
    ierr = DMStagGetLocationSlot(usr->dmMPhase, UP_RIGHT,   ii, &iwtru[ii]); CHKERRQ(ierr);
  }

  PetscInt ixx, izz, ixz, iII, ixxn[4], izzn[4], ixzn[4], iIIn[4];
  ierr = DMStagGetLocationSlot(usr->dmeps, ELEMENT, 0, &ixx); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, ELEMENT, 1, &izz); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, ELEMENT, 2, &ixz); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, ELEMENT, 3, &iII); CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(usr->dmeps, DOWN_LEFT, 0, &ixxn[0]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, DOWN_LEFT, 1, &izzn[0]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, DOWN_LEFT, 2, &ixzn[0]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, DOWN_LEFT, 3, &iIIn[0]); CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(usr->dmeps, DOWN_RIGHT, 0, &ixxn[1]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, DOWN_RIGHT, 1, &izzn[1]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, DOWN_RIGHT, 2, &ixzn[1]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, DOWN_RIGHT, 3, &iIIn[1]); CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(usr->dmeps, UP_LEFT, 0, &ixxn[2]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, UP_LEFT, 1, &izzn[2]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, UP_LEFT, 2, &ixzn[2]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, UP_LEFT, 3, &iIIn[2]); CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(usr->dmeps, UP_RIGHT, 0, &ixxn[3]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, UP_RIGHT, 1, &izzn[3]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, UP_RIGHT, 2, &ixzn[3]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, UP_RIGHT, 3, &iIIn[3]); CHKERRQ(ierr);

  PetscInt imat[MATPROP_NPROP];
  for (ii = 0; ii < MATPROP_NPROP; ii++) { ierr = DMStagGetLocationSlot(usr->dmmatProp, ELEMENT, ii, &imat[ii]); CHKERRQ(ierr); }
  
  // Loop over local domain - set initial density and viscosity
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      PetscScalar   phic[9], pc[9], Plc[9], Tc[9], DPoldc[9], strain[9], dx, dz;
      PetscInt      iph, ii, im, jm, ip, jp;

      if (i == 0   ) im = i; else im = i-1;
      if (i == Nx-1) ip = i; else ip = i+1;
      if (j == 0   ) jm = j; else jm = j-1;
      if (j == Nz-1) jp = j; else jp = j+1;

      // should be adapted to variable spacing
      dx = coordx[i][inext]-coordx[i][iprev];
      dz = coordz[j][inext]-coordz[j][iprev];

      // get porosity, p, Plith, T - center
      ierr = Get9PointCenterValues(i,j,iE,Nx,Nz,_phic,phic);CHKERRQ(ierr);
      ierr = Get9PointCenterValues(i,j,iP,Nx,Nz,_Plith,Plc);CHKERRQ(ierr);
      ierr = Get9PointCenterValues(i,j,iPV,Nx,Nz,xx,pc);CHKERRQ(ierr);
      ierr = Get9PointCenterValues(i,j,iE,Nx,Nz,_Tc,Tc);CHKERRQ(ierr);
      ierr = Get9PointCenterValues(i,j,iP,Nx,Nz,_DPold,DPoldc);CHKERRQ(ierr);
      ierr = Get9PointCenterValues(i,j,iP,Nx,Nz,_strain,strain);CHKERRQ(ierr);

      // correct for liquid porosity
      for (ii = 0; ii < 9; ii++) { 
        phic[ii] = 1.0 - phic[ii]; 
        if (phic[ii]<0.0) phic[ii] = 0.0;
      }

      // Prepare for pointwise rheology calculation
      PetscScalar eta_eff[9], zeta_eff[9], chis[9], chip[9], txx[9], tzz[9], txz[9], tII[9], DP[9], DPdl[9], dotlam[9];
      PetscScalar eta_v[9],eta_e[9],eta_p[9],zeta_v[9],zeta_e[9];
      PetscScalar e[4], t[4], P[4], res[21], Z[9], G[9], C[9], sigmat[9], theta[9];
      PetscInt ix[4];

      // center points
      ix[0] = ixx; ix[1] = izz; ix[2] = ixz; ix[3] = iII;

      P[0] = pc[0]; P[1] = Plc[0]; P[2] = DPoldc[0]; P[3] = strain[0];
      ierr = GetTensorPointValues(i,j,ix,_eps,e);CHKERRQ(ierr);
      ierr = GetTensorPointValues(i,j,ix,_tauold,t);CHKERRQ(ierr);
      ierr = RheologyPointwise_DPL(i,j,xwt,iwtc,Tc[0],phic[0],P,e,t,ix,res,usr);CHKERRQ(ierr);
      ierr = DecompactRheologyVars_DPL(0,res,eta_eff,eta_v,eta_e,eta_p,zeta_eff,zeta_v,zeta_e,DPdl,chis,chip,txx,tzz,txz,tII,DP,dotlam,Z,G,C,sigmat,theta);CHKERRQ(ierr);

      P[0] = pc[1]; P[1] = Plc[1]; P[2] = DPoldc[1]; P[3] = strain[1];
      ierr = GetTensorPointValues(im,j,ix,_eps,e);CHKERRQ(ierr);
      ierr = GetTensorPointValues(im,j,ix,_tauold,t);CHKERRQ(ierr);
      ierr = RheologyPointwise_DPL(im,j,xwt,iwtc,Tc[1],phic[1],P,e,t,ix,res,usr);CHKERRQ(ierr);
      ierr = DecompactRheologyVars_DPL(1,res,eta_eff,eta_v,eta_e,eta_p,zeta_eff,zeta_v,zeta_e,DPdl,chis,chip,txx,tzz,txz,tII,DP,dotlam,Z,G,C,sigmat,theta);CHKERRQ(ierr);

      P[0] = pc[2]; P[1] = Plc[2]; P[2] = DPoldc[2]; P[3] = strain[2];
      ierr = GetTensorPointValues(ip,j,ix,_eps,e);CHKERRQ(ierr);
      ierr = GetTensorPointValues(ip,j,ix,_tauold,t);CHKERRQ(ierr);
      ierr = RheologyPointwise_DPL(ip,j,xwt,iwtc,Tc[2],phic[2],P,e,t,ix,res,usr);CHKERRQ(ierr);
      ierr = DecompactRheologyVars_DPL(2,res,eta_eff,eta_v,eta_e,eta_p,zeta_eff,zeta_v,zeta_e,DPdl,chis,chip,txx,tzz,txz,tII,DP,dotlam,Z,G,C,sigmat,theta);CHKERRQ(ierr);

      P[0] = pc[3]; P[1] = Plc[3]; P[2] = DPoldc[3]; P[3] = strain[3];
      ierr = GetTensorPointValues(i,jm,ix,_eps,e);CHKERRQ(ierr);
      ierr = GetTensorPointValues(i,jm,ix,_tauold,t);CHKERRQ(ierr);
      ierr = RheologyPointwise_DPL(i,jm,xwt,iwtc,Tc[3],phic[3],P,e,t,ix,res,usr);CHKERRQ(ierr);
      ierr = DecompactRheologyVars_DPL(3,res,eta_eff,eta_v,eta_e,eta_p,zeta_eff,zeta_v,zeta_e,DPdl,chis,chip,txx,tzz,txz,tII,DP,dotlam,Z,G,C,sigmat,theta);CHKERRQ(ierr);

      P[0] = pc[4]; P[1] = Plc[4]; P[2] = DPoldc[4]; P[3] = strain[4];
      ierr = GetTensorPointValues(i,jp,ix,_eps,e);CHKERRQ(ierr);
      ierr = GetTensorPointValues(i,jp,ix,_tauold,t);CHKERRQ(ierr);
      ierr = RheologyPointwise_DPL(i,jp,xwt,iwtc,Tc[4],phic[4],P,e,t,ix,res,usr);CHKERRQ(ierr);
      ierr = DecompactRheologyVars_DPL(4,res,eta_eff,eta_v,eta_e,eta_p,zeta_eff,zeta_v,zeta_e,DPdl,chis,chip,txx,tzz,txz,tII,DP,dotlam,Z,G,C,sigmat,theta);CHKERRQ(ierr);

      // corner points
      PetscScalar Tcorner[4], phicorner[4], pcorner[4], Plcorner[4], DPoldcorner[4], straincorner[4];
      ierr = GetCornerAvgFromCenter(Tc,Tcorner);CHKERRQ(ierr);
      ierr = GetCornerAvgFromCenter(phic,phicorner);CHKERRQ(ierr);
      ierr = GetCornerAvgFromCenter(pc,pcorner);CHKERRQ(ierr);
      ierr = GetCornerAvgFromCenter(Plc,Plcorner);CHKERRQ(ierr);
      ierr = GetCornerAvgFromCenter(DPoldc,DPoldcorner);CHKERRQ(ierr);
      ierr = GetCornerAvgFromCenter(strain,straincorner);CHKERRQ(ierr);
      
      ii = 0; ix[0] = ixxn[ii]; ix[1] = izzn[ii]; ix[2] = ixzn[ii]; ix[3] = iIIn[ii]; P[0] = pcorner[ii]; P[1] = Plcorner[ii]; P[2] = DPoldcorner[ii]; P[3] = straincorner[ii];
      ierr = GetTensorPointValues(i,j,ix,_eps,e);CHKERRQ(ierr);
      ierr = GetTensorPointValues(i,j,ix,_tauold,t);CHKERRQ(ierr);
      ierr = RheologyPointwise_DPL(i,j,xwt,iwtld,Tcorner[0],phicorner[0],P,e,t,ix,res,usr);CHKERRQ(ierr);
      ierr = DecompactRheologyVars_DPL(5,res,eta_eff,eta_v,eta_e,eta_p,zeta_eff,zeta_v,zeta_e,DPdl,chis,chip,txx,tzz,txz,tII,DP,dotlam,Z,G,C,sigmat,theta);CHKERRQ(ierr);

      ii = 1; ix[0] = ixxn[ii]; ix[1] = izzn[ii]; ix[2] = ixzn[ii]; ix[3] = iIIn[ii]; P[0] = pcorner[ii]; P[1] = Plcorner[ii]; P[2] = DPoldcorner[ii]; P[3] = straincorner[ii];
      ierr = GetTensorPointValues(i,j,ix,_eps,e);CHKERRQ(ierr);
      ierr = GetTensorPointValues(i,j,ix,_tauold,t);CHKERRQ(ierr);
      ierr = RheologyPointwise_DPL(i,j,xwt,iwtrd,Tcorner[1],phicorner[1],P,e,t,ix,res,usr);CHKERRQ(ierr);
      ierr = DecompactRheologyVars_DPL(6,res,eta_eff,eta_v,eta_e,eta_p,zeta_eff,zeta_v,zeta_e,DPdl,chis,chip,txx,tzz,txz,tII,DP,dotlam,Z,G,C,sigmat,theta);CHKERRQ(ierr);

      ii = 2; ix[0] = ixxn[ii]; ix[1] = izzn[ii]; ix[2] = ixzn[ii]; ix[3] = iIIn[ii]; P[0] = pcorner[ii]; P[1] = Plcorner[ii]; P[2] = DPoldcorner[ii]; P[3] = straincorner[ii];
      ierr = GetTensorPointValues(i,j,ix,_eps,e);CHKERRQ(ierr);
      ierr = GetTensorPointValues(i,j,ix,_tauold,t);CHKERRQ(ierr);
      ierr = RheologyPointwise_DPL(i,j,xwt,iwtlu,Tcorner[2],phicorner[2],P,e,t,ix,res,usr);CHKERRQ(ierr);
      ierr = DecompactRheologyVars_DPL(7,res,eta_eff,eta_v,eta_e,eta_p,zeta_eff,zeta_v,zeta_e,DPdl,chis,chip,txx,tzz,txz,tII,DP,dotlam,Z,G,C,sigmat,theta);CHKERRQ(ierr);

      ii = 3; ix[0] = ixxn[ii]; ix[1] = izzn[ii]; ix[2] = ixzn[ii]; ix[3] = iIIn[ii]; P[0] = pcorner[ii]; P[1] = Plcorner[ii]; P[2] = DPoldcorner[ii]; P[3] = straincorner[ii];
      ierr = GetTensorPointValues(i,j,ix,_eps,e);CHKERRQ(ierr);
      ierr = GetTensorPointValues(i,j,ix,_tauold,t);CHKERRQ(ierr);
      ierr = RheologyPointwise_DPL(i,j,xwt,iwtru,Tcorner[3],phicorner[3],P,e,t,ix,res,usr);CHKERRQ(ierr);
      ierr = DecompactRheologyVars_DPL(8,res,eta_eff,eta_v,eta_e,eta_p,zeta_eff,zeta_v,zeta_e,DPdl,chis,chip,txx,tzz,txz,tII,DP,dotlam,Z,G,C,sigmat,theta);CHKERRQ(ierr);

      { // element 
        // C = 0 (center, c=0)
        c[j][i][e_slot[iC]] = 0.0;

        // A = eta_eff,phi (center, c=1)
        c[j][i][e_slot[iA]] = eta_eff[0];

        // D1 = zeta_eff-2/3eta_eff (center, c=2)
        c[j][i][e_slot[iD1]] = zeta_eff[0]-2.0/3.0*eta_eff[0];
      }

      { // corner
        // A = eta_eff,phi (corner, c=0)
        for (ii = 0; ii < 4; ii++) {
          c[j][i][av_slot[ii]] = eta_eff[5+ii];
        }
      }

      { // face
        PetscScalar   phi[4], K[4], B[4], D2[4], D3[4];
        PetscScalar   divchitau[4],gradchidp[4], gradPlith[4], gradDPdl[4];
        PetscScalar   rhof[4], rhos[4], rho[4], wt[MAX_MAT_PHASE], rho0[MAX_MAT_PHASE];
        PetscInt      idx[MAX_MAT_PHASE];


        // porosity on edges - not for variable grid spacing
        phi[0] = (phic[0]+phic[1])*0.5;
        phi[1] = (phic[0]+phic[2])*0.5;
        phi[2] = (phic[0]+phic[3])*0.5;
        phi[3] = (phic[0]+phic[4])*0.5;

        gradPlith[0] = (Plc[0]-Plc[1])/dx;
        gradPlith[1] = (Plc[2]-Plc[0])/dx;
        gradPlith[2] = (Plc[0]-Plc[3])/dz;
        gradPlith[3] = (Plc[4]-Plc[0])/dz;

        gradDPdl[0] = (DPdl[0]-DPdl[1])/dx;
        gradDPdl[1] = (DPdl[2]-DPdl[0])/dx;
        gradDPdl[2] = (DPdl[0]-DPdl[3])/dz;
        gradDPdl[3] = (DPdl[4]-DPdl[0])/dz;

        gradchidp[0] = (chip[0]*_DPold[j ][i ][iP] - chip[1]*_DPold[j ][im][iP])/dx;
        gradchidp[1] = (chip[2]*_DPold[j ][ip][iP] - chip[0]*_DPold[j ][i ][iP])/dx;
        gradchidp[2] = (chip[0]*_DPold[j ][i ][iP] - chip[3]*_DPold[jm][i ][iP])/dz;
        gradchidp[3] = (chip[4]*_DPold[jp][i ][iP] - chip[0]*_DPold[j ][i ][iP])/dz;

        //  div(chis*tau_old) = div(S) = [dSxx/dx+dSxz/dz, dSzx/dx+dSzz/dz]
        divchitau[0] = (chis[0]*_tauold[j][i ][ixx] - chis[1]*_tauold[j][im][ixx])/dx + (chis[7]*_tauold[j][i][ixzn[2]]-chis[5]*_tauold[j][i][ixzn[0]])/dz;
        divchitau[1] = (chis[2]*_tauold[j][ip][ixx] - chis[0]*_tauold[j][i ][ixx])/dx + (chis[8]*_tauold[j][i][ixzn[3]]-chis[6]*_tauold[j][i][ixzn[1]])/dz;
        divchitau[2] = (chis[6]*_tauold[j][i][ixzn[1]]-chis[5]*_tauold[j][i][ixzn[0]])/dx + (chis[0]*_tauold[j ][i][izz] - chis[3]*_tauold[jm][i][izz])/dz;
        divchitau[3] = (chis[8]*_tauold[j][i][ixzn[3]]-chis[7]*_tauold[j][i][ixzn[2]])/dx + (chis[4]*_tauold[jp][i][izz] - chis[0]*_tauold[j ][i][izz])/dz;

        for (ii = 0; ii < 4; ii++) {
          K[ii]  = Permeability(phi[ii],usr->par->n); // assumed uniform for all materials

          // get material phase fraction
          if (ii == 0 ) { idx[0] = iwtl[0]; idx[1] = iwtl[1]; idx[2] = iwtl[2]; idx[3] = iwtl[3]; idx[4] = iwtl[4]; idx[5] = iwtl[5]; }
          if (ii == 1 ) { idx[0] = iwtr[0]; idx[1] = iwtr[1]; idx[2] = iwtr[2]; idx[3] = iwtr[3]; idx[4] = iwtr[4]; idx[5] = iwtr[5];}
          if (ii == 2 ) { idx[0] = iwtd[0]; idx[1] = iwtd[1]; idx[2] = iwtd[2]; idx[3] = iwtd[3]; idx[4] = iwtd[4]; idx[5] = iwtd[5];}
          if (ii == 3 ) { idx[0] = iwtu[0]; idx[1] = iwtu[1]; idx[2] = iwtu[2]; idx[3] = iwtu[3]; idx[4] = iwtu[4]; idx[5] = iwtu[5];}

          // get bulk density
          for (iph = 0; iph < usr->nph; iph++) { rho0[iph] = Density(usr->mat_nd[iph].rho0,usr->mat[iph].rho_func); }
          ierr = GetMatPhaseFraction(i,j,xwt,idx,usr->nph,wt); CHKERRQ(ierr);
          rhos[ii] = WeightAverageValue(rho0,wt,usr->nph); 
          rhof[ii] = usr->nd->rhof;
          rho[ii]  = Mixture(rhos[ii],rhof[ii],phi[ii]);

          // add buoyancy terms here
          B[ii] = -divchitau[ii]+gradchidp[ii] +gradDPdl[ii] + gradPlith[ii] + rho[ii]*k_hat[ii];
          // B[ii] = -divchitau[ii]+gradchidp[ii] +gradDPdl[ii] + gradPlith[ii]*i_hat[ii] - phi[ii]*k_hat[ii];
          D2[ii] = -K[ii]*usr->nd->R*usr->nd->R;
          D3[ii] = -K[ii]*usr->nd->R*usr->nd->R*(gradPlith[ii]*i_hat[ii] - k_hat[ii]);

          // B = body force+elasticity (edges, c=0)
          c[j][i][b_slot[ii]] = B[ii];

          // D2 = -R^2*Kphi (edges, c=1)
          c[j][i][d2_slot[ii]] = D2[ii];

          // D3 = -R^2*Kphi*(grad(Plith)-rho_ell/drho*k_hat) (edges, c=2)
          c[j][i][d3_slot[ii]] = D3[ii];
        }
      }

      // save stresses for output + dotlam
      _tau[j][i][ixx]     = txx[0]; _tau[j][i][izz]     = tzz[0]; _tau[j][i][ixz]     = txz[0]; _tau[j][i][iII]     = tII[0];
      _tau[j][i][ixxn[0]] = txx[5]; _tau[j][i][izzn[0]] = tzz[5]; _tau[j][i][ixzn[0]] = txz[5]; _tau[j][i][iIIn[0]] = tII[5];
      _tau[j][i][ixxn[1]] = txx[6]; _tau[j][i][izzn[1]] = tzz[6]; _tau[j][i][ixzn[1]] = txz[6]; _tau[j][i][iIIn[1]] = tII[6];
      _tau[j][i][ixxn[2]] = txx[7]; _tau[j][i][izzn[2]] = tzz[7]; _tau[j][i][ixzn[2]] = txz[7]; _tau[j][i][iIIn[2]] = tII[7];
      _tau[j][i][ixxn[3]] = txx[8]; _tau[j][i][izzn[3]] = tzz[8]; _tau[j][i][ixzn[3]] = txz[8]; _tau[j][i][iIIn[3]] = tII[8];
      _DP[j][i][iP]    = DP[0];
      _plast[j][i][iP] = dotlam[0];

      // get density (center)
      PetscScalar rho0[MAX_MAT_PHASE],rho,wt[MAX_MAT_PHASE];
      for (iph = 0; iph < usr->nph; iph++) { rho0[iph] = Density(usr->mat_nd[iph].rho0,usr->mat[iph].rho_func); }
      ierr = GetMatPhaseFraction(i,j,xwt,iwtc,usr->nph,wt); CHKERRQ(ierr);
      rho  = WeightAverageValue(rho0,wt,usr->nph);

      // get permeability (center)
      PetscScalar Kphi;
      Kphi  = Permeability(phic[0],usr->par->n);

      // save material properties for output
      _matProp[j][i][imat[MATPROP_ELEMENT_ETA]]    = eta_eff[0];
      _matProp[j][i][imat[MATPROP_ELEMENT_ETA_V]]  = eta_v[0];
      _matProp[j][i][imat[MATPROP_ELEMENT_ETA_E]]  = eta_e[0];
      _matProp[j][i][imat[MATPROP_ELEMENT_ETA_P]]  = eta_p[0];
      _matProp[j][i][imat[MATPROP_ELEMENT_ZETA]]   = zeta_eff[0];
      _matProp[j][i][imat[MATPROP_ELEMENT_ZETA_V]] = zeta_v[0];
      _matProp[j][i][imat[MATPROP_ELEMENT_ZETA_E]] = zeta_e[0];
      _matProp[j][i][imat[MATPROP_ELEMENT_ZETA_P]] = DPdl[0];
      _matProp[j][i][imat[MATPROP_ELEMENT_Z]]      = Z[0];
      _matProp[j][i][imat[MATPROP_ELEMENT_G]]      = G[0];
      _matProp[j][i][imat[MATPROP_ELEMENT_C]]      = C[0];
      _matProp[j][i][imat[MATPROP_ELEMENT_SIGMAT]] = sigmat[0];
      _matProp[j][i][imat[MATPROP_ELEMENT_THETA]]  = theta[0];
      _matProp[j][i][imat[MATPROP_ELEMENT_RHO]]    = rho;
      _matProp[j][i][imat[MATPROP_ELEMENT_KPHI]]   = Kphi;
      _matProp[j][i][imat[MATPROP_ELEMENT_CHIP]]   = chip[0];
      _matProp[j][i][imat[MATPROP_ELEMENT_CHIS]]   = chis[0];
    }
  }

  // Restore arrays, local vectors
  ierr = DMStagRestoreProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArray(dmcoeff,coefflocal,&c);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  ierr = VecDestroy(&coefflocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArrayRead(dm,xlocal,&xx);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&xlocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArrayRead(usr->dmeps,xepslocal,&_eps);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmeps,&xepslocal); CHKERRQ(ierr);
  ierr = DMStagVecRestoreArrayRead(usr->dmeps,xtauoldlocal,&_tauold);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmeps,&xtauoldlocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArray(usr->dmeps,xtaulocal,&_tau);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(usr->dmeps,xtaulocal,INSERT_VALUES,usr->xtau); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (usr->dmeps,xtaulocal,INSERT_VALUES,usr->xtau); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmeps,&xtaulocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArrayRead(usr->dmT,xTlocal,&_Tc);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmT,&xTlocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArrayRead(usr->dmphi,xphilocal,&_phic);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmphi,&xphilocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArrayRead(usr->dmPlith,xPlithlocal,&_Plith);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmPlith,&xPlithlocal); CHKERRQ(ierr);
  ierr = DMStagVecRestoreArrayRead(usr->dmPlith,xDPoldlocal,&_DPold);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmPlith,&xDPoldlocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArray(usr->dmPlith,xDPlocal,&_DP);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(usr->dmPlith,xDPlocal,INSERT_VALUES,usr->xDP); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (usr->dmPlith,xDPlocal,INSERT_VALUES,usr->xDP); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmPlith,&xDPlocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArray(usr->dmPlith,xplastlocal,&_plast);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(usr->dmPlith,xplastlocal,INSERT_VALUES,usr->xplast); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (usr->dmPlith,xplastlocal,INSERT_VALUES,usr->xplast); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmPlith,&xplastlocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArrayRead(usr->dmPlith,xstrainlocal,&_strain);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmPlith,&xstrainlocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArray(usr->dmmatProp,xmatProplocal,&_matProp);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(usr->dmmatProp,xmatProplocal,INSERT_VALUES,usr->xmatProp); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (usr->dmmatProp,xmatProplocal,INSERT_VALUES,usr->xmatProp); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmmatProp,&xmatProplocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArray(usr->dmMPhase,xMPhaselocal,&xwt);CHKERRQ(ierr);
  ierr = VecDestroy(&xMPhaselocal); CHKERRQ(ierr);
 
  PetscFunctionReturn(0);
}

// ---------------------------------------
// FormCoefficient_PV (Stokes)
// ---------------------------------------
PetscErrorCode FormCoefficient_PV_Stokes(FDPDE fd, DM dm, Vec x, DM dmcoeff, Vec coeff, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       i, j, ii, sx, sz, nx, nz, iprev, inext, icenter, Nx, Nz;
  PetscScalar    **coordx, **coordz, ***c, ***xx, ***xwt; 
  PetscScalar    ***_Tc, ***_phic, ***_Plith, ***_eps, ***_tauold, ***_DPold, ***_tau, ***_DP;
  PetscScalar     ***_plast, ***_matProp, ***_strain;
  Vec            coefflocal, xlocal, xTlocal, xMPhaselocal, xtaulocal, xDPlocal, xstrainlocal;
  Vec            xphilocal, xPlithlocal, xepslocal, xtauoldlocal, xDPoldlocal, xplastlocal, xmatProplocal;
  PetscScalar    k_hat[4], i_hat[4];
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  // parameters
  k_hat[0] = 0.0;
  k_hat[1] = 0.0;
  k_hat[2] = usr->par->k_hat;
  k_hat[3] = usr->par->k_hat;

  i_hat[0] = 1.0;
  i_hat[1] = 1.0;
  i_hat[2] = 0.0;
  i_hat[3] = 0.0;

  // Get coefficient
  ierr = DMStagGetGlobalSizes(dmcoeff, &Nx, &Nz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmcoeff,LEFT,&iprev);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmcoeff,RIGHT,&inext);CHKERRQ(ierr); 
  ierr = DMStagGetProductCoordinateLocationSlot(dmcoeff,ELEMENT,&icenter);CHKERRQ(ierr);

  // Create coefficient local vector
  ierr = DMCreateLocalVector(dmcoeff, &coefflocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dmcoeff, coefflocal, &c); CHKERRQ(ierr);

  // Strain rates
  // ierr = UpdateStrainRates(dm,x,usr); CHKERRQ(ierr);
  ierr = UpdateStrainRates_Array(dm,x,usr); CHKERRQ(ierr);
  ierr = DMGetLocalVector(usr->dmeps, &xepslocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmeps, usr->xeps, INSERT_VALUES, xepslocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(usr->dmeps,xepslocal,&_eps);CHKERRQ(ierr);

  ierr = DMGetLocalVector(usr->dmeps, &xtauoldlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmeps, usr->xtau_old, INSERT_VALUES, xtauoldlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(usr->dmeps,xtauoldlocal,&_tauold);CHKERRQ(ierr);

  ierr = DMGetLocalVector(usr->dmeps, &xtaulocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmeps, usr->xtau, INSERT_VALUES, xtaulocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(usr->dmeps,xtaulocal,&_tau);CHKERRQ(ierr);

  // Get dm and solution vector
  ierr = DMGetLocalVector(dm, &xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm, x, INSERT_VALUES, xlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(dm,xlocal,&xx);CHKERRQ(ierr);
  
  // Get solution vector for temperature
  ierr = DMGetLocalVector(usr->dmT,&xTlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmT,usr->xT,INSERT_VALUES,xTlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(usr->dmT,xTlocal,&_Tc);CHKERRQ(ierr);

  // Get porosity
  ierr = DMGetLocalVector(usr->dmphi,&xphilocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmphi,usr->xphi,INSERT_VALUES,xphilocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(usr->dmphi,xphilocal,&_phic);CHKERRQ(ierr);

  // Get dm and vector Plith
  ierr = DMGetLocalVector(usr->dmPlith, &xPlithlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmPlith, usr->xPlith, INSERT_VALUES, xPlithlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(usr->dmPlith,xPlithlocal,&_Plith);CHKERRQ(ierr);

  ierr = DMGetLocalVector(usr->dmPlith, &xDPoldlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmPlith, usr->xDP_old, INSERT_VALUES, xDPoldlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(usr->dmPlith,xDPoldlocal,&_DPold);CHKERRQ(ierr);

  ierr = DMGetLocalVector(usr->dmPlith, &xDPlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmPlith, usr->xDP, INSERT_VALUES, xDPlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(usr->dmPlith,xDPlocal,&_DP);CHKERRQ(ierr);

  ierr = DMGetLocalVector(usr->dmPlith, &xplastlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmPlith, usr->xplast, INSERT_VALUES, xplastlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(usr->dmPlith,xplastlocal,&_plast);CHKERRQ(ierr);

  ierr = DMGetLocalVector(usr->dmPlith, &xstrainlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmPlith, usr->xstrain, INSERT_VALUES, xstrainlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(usr->dmPlith,xstrainlocal,&_strain);CHKERRQ(ierr);

  // get material phase fractions
  ierr = DMCreateLocalVector(usr->dmMPhase, &xMPhaselocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(usr->dmMPhase,usr->xMPhase,INSERT_VALUES,xMPhaselocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (usr->dmMPhase,usr->xMPhase,INSERT_VALUES,xMPhaselocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(usr->dmMPhase, xMPhaselocal, &xwt); CHKERRQ(ierr);

  // Get material properties - for output
  ierr = DMGetLocalVector(usr->dmmatProp, &xmatProplocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmmatProp, usr->xmatProp, INSERT_VALUES, xmatProplocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(usr->dmmatProp,xmatProplocal,&_matProp);CHKERRQ(ierr);

  // get location slots
  PetscInt  iE,iP;
  ierr = DMStagGetLocationSlot(usr->dmT,ELEMENT,T_ELEMENT,&iE);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmPlith,ELEMENT,0,&iP);CHKERRQ(ierr);

  PetscInt iPV;
  ierr = DMStagGetLocationSlot(dm,ELEMENT,PV_ELEMENT_P,&iPV);CHKERRQ(ierr);

  PetscInt  e_slot[3],av_slot[4],b_slot[4],iL,iR,iU,iD,iC,iA;
  iL = 0; iR = 1; iD = 2; iU  = 3;
  iC = 0; iA = 1; 
  ierr = DMStagGetLocationSlot(dmcoeff,ELEMENT,   PVCOEFF_ELEMENT_C,   &e_slot[iC]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,ELEMENT,   PVCOEFF_ELEMENT_A,   &e_slot[iA]);CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(dmcoeff,DOWN_LEFT, PVCOEFF_VERTEX_A,&av_slot[0]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DOWN_RIGHT,PVCOEFF_VERTEX_A,&av_slot[1]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,UP_LEFT,   PVCOEFF_VERTEX_A,&av_slot[2]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,UP_RIGHT,  PVCOEFF_VERTEX_A,&av_slot[3]);CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(dmcoeff,LEFT,   PVCOEFF_FACE_B,   &b_slot[iL]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,RIGHT,  PVCOEFF_FACE_B,   &b_slot[iR]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DOWN,   PVCOEFF_FACE_B,   &b_slot[iD]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,UP,     PVCOEFF_FACE_B,   &b_slot[iU]);CHKERRQ(ierr);

  PetscInt iwtc[MAX_MAT_PHASE],iwtl[MAX_MAT_PHASE],iwtr[MAX_MAT_PHASE],iwtd[MAX_MAT_PHASE],iwtu[MAX_MAT_PHASE];
  PetscInt iwtld[MAX_MAT_PHASE],iwtrd[MAX_MAT_PHASE],iwtlu[MAX_MAT_PHASE],iwtru[MAX_MAT_PHASE];
  for (ii = 0; ii < usr->nph; ii++) { 
    ierr = DMStagGetLocationSlot(usr->dmMPhase, ELEMENT,    ii, &iwtc[ii]); CHKERRQ(ierr); 
    ierr = DMStagGetLocationSlot(usr->dmMPhase, LEFT,       ii, &iwtl[ii]); CHKERRQ(ierr);
    ierr = DMStagGetLocationSlot(usr->dmMPhase, RIGHT,      ii, &iwtr[ii]); CHKERRQ(ierr);
    ierr = DMStagGetLocationSlot(usr->dmMPhase, DOWN,       ii, &iwtd[ii]); CHKERRQ(ierr); 
    ierr = DMStagGetLocationSlot(usr->dmMPhase, UP,         ii, &iwtu[ii]); CHKERRQ(ierr);
    ierr = DMStagGetLocationSlot(usr->dmMPhase, DOWN_LEFT,  ii, &iwtld[ii]); CHKERRQ(ierr);
    ierr = DMStagGetLocationSlot(usr->dmMPhase, DOWN_RIGHT, ii, &iwtrd[ii]); CHKERRQ(ierr);
    ierr = DMStagGetLocationSlot(usr->dmMPhase, UP_LEFT,    ii, &iwtlu[ii]); CHKERRQ(ierr);
    ierr = DMStagGetLocationSlot(usr->dmMPhase, UP_RIGHT,   ii, &iwtru[ii]); CHKERRQ(ierr);
  }

  PetscInt ixx, izz, ixz, iII, ixxn[4], izzn[4], ixzn[4], iIIn[4];
  ierr = DMStagGetLocationSlot(usr->dmeps, ELEMENT, 0, &ixx); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, ELEMENT, 1, &izz); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, ELEMENT, 2, &ixz); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, ELEMENT, 3, &iII); CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(usr->dmeps, DOWN_LEFT, 0, &ixxn[0]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, DOWN_LEFT, 1, &izzn[0]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, DOWN_LEFT, 2, &ixzn[0]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, DOWN_LEFT, 3, &iIIn[0]); CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(usr->dmeps, DOWN_RIGHT, 0, &ixxn[1]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, DOWN_RIGHT, 1, &izzn[1]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, DOWN_RIGHT, 2, &ixzn[1]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, DOWN_RIGHT, 3, &iIIn[1]); CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(usr->dmeps, UP_LEFT, 0, &ixxn[2]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, UP_LEFT, 1, &izzn[2]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, UP_LEFT, 2, &ixzn[2]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, UP_LEFT, 3, &iIIn[2]); CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(usr->dmeps, UP_RIGHT, 0, &ixxn[3]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, UP_RIGHT, 1, &izzn[3]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, UP_RIGHT, 2, &ixzn[3]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, UP_RIGHT, 3, &iIIn[3]); CHKERRQ(ierr);

  PetscInt imat[MATPROP_NPROP];
  for (ii = 0; ii < MATPROP_NPROP; ii++) { ierr = DMStagGetLocationSlot(usr->dmmatProp, ELEMENT, ii, &imat[ii]); CHKERRQ(ierr); }

  // Loop over local domain - set initial density and viscosity
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      PetscScalar   phic[9], pc[9], Plc[9], Tc[9], DPoldc[9], strain[9], dx, dz;
      PetscInt      iph, ii, im, jm, ip, jp;

      if (i == 0   ) im = i; else im = i-1;
      if (i == Nx-1) ip = i; else ip = i+1;
      if (j == 0   ) jm = j; else jm = j-1;
      if (j == Nz-1) jp = j; else jp = j+1;

      // should be adapted to variable spacing
      dx = coordx[i][inext]-coordx[i][iprev];
      dz = coordz[j][inext]-coordz[j][iprev];

      // get porosity, p, Plith, T - center
      ierr = Get9PointCenterValues(i,j,iE,Nx,Nz,_phic,phic);CHKERRQ(ierr);
      ierr = Get9PointCenterValues(i,j,iP,Nx,Nz,_Plith,Plc);CHKERRQ(ierr);
      ierr = Get9PointCenterValues(i,j,iPV,Nx,Nz,xx,pc);CHKERRQ(ierr);
      ierr = Get9PointCenterValues(i,j,iE,Nx,Nz,_Tc,Tc);CHKERRQ(ierr);
      ierr = Get9PointCenterValues(i,j,iP,Nx,Nz,_DPold,DPoldc);CHKERRQ(ierr);
      ierr = Get9PointCenterValues(i,j,iP,Nx,Nz,_strain,strain);CHKERRQ(ierr);

      // set porosity phi = 0
      for (ii = 0; ii < 9; ii++) { phic[ii] = 0.0; }

      // Prepare for pointwise rheology calculation
      PetscScalar eta_eff[9], zeta_eff[9], chis[9], chip[9], txx[9], tzz[9], txz[9], tII[9], DP[9], dotlam[9];
      PetscScalar eta_v[9],eta_e[9],eta_p[9],zeta_v[9],zeta_e[9],zeta_p[9];
      PetscScalar e[4], t[4], P[4], res[21], Z[9], G[9], C[9], sigmat[9], theta[9];
      PetscInt ix[4];

      // center points
      ix[0] = ixx; ix[1] = izz; ix[2] = ixz; ix[3] = iII;

      P[0] = pc[0]; P[1] = Plc[0]; P[2] = DPoldc[0]; P[3] = strain[0];
      ierr = GetTensorPointValues(i,j,ix,_eps,e);CHKERRQ(ierr);
      ierr = GetTensorPointValues(i,j,ix,_tauold,t);CHKERRQ(ierr);
      ierr = RheologyPointwise(i,j,xwt,iwtc,Tc[0],phic[0],P,e,t,ix,res,usr);CHKERRQ(ierr);
      ierr = DecompactRheologyVars(0,res,eta_eff,eta_v,eta_e,eta_p,zeta_eff,zeta_v,zeta_e,zeta_p,chis,chip,txx,tzz,txz,tII,DP,dotlam,Z,G,C,sigmat,theta);CHKERRQ(ierr);

      P[0] = pc[1]; P[1] = Plc[1]; P[2] = DPoldc[1]; P[3] = strain[1];
      ierr = GetTensorPointValues(im,j,ix,_eps,e);CHKERRQ(ierr);
      ierr = GetTensorPointValues(im,j,ix,_tauold,t);CHKERRQ(ierr);
      ierr = RheologyPointwise(im,j,xwt,iwtc,Tc[1],phic[1],P,e,t,ix,res,usr);CHKERRQ(ierr);
      ierr = DecompactRheologyVars(1,res,eta_eff,eta_v,eta_e,eta_p,zeta_eff,zeta_v,zeta_e,zeta_p,chis,chip,txx,tzz,txz,tII,DP,dotlam,Z,G,C,sigmat,theta);CHKERRQ(ierr);

      P[0] = pc[2]; P[1] = Plc[2]; P[2] = DPoldc[2]; P[3] = strain[2];
      ierr = GetTensorPointValues(ip,j,ix,_eps,e);CHKERRQ(ierr);
      ierr = GetTensorPointValues(ip,j,ix,_tauold,t);CHKERRQ(ierr);
      ierr = RheologyPointwise(ip,j,xwt,iwtc,Tc[2],phic[2],P,e,t,ix,res,usr);CHKERRQ(ierr);
      ierr = DecompactRheologyVars(2,res,eta_eff,eta_v,eta_e,eta_p,zeta_eff,zeta_v,zeta_e,zeta_p,chis,chip,txx,tzz,txz,tII,DP,dotlam,Z,G,C,sigmat,theta);CHKERRQ(ierr);

      P[0] = pc[3]; P[1] = Plc[3]; P[2] = DPoldc[3]; P[3] = strain[3];
      ierr = GetTensorPointValues(i,jm,ix,_eps,e);CHKERRQ(ierr);
      ierr = GetTensorPointValues(i,jm,ix,_tauold,t);CHKERRQ(ierr);
      ierr = RheologyPointwise(i,jm,xwt,iwtc,Tc[3],phic[3],P,e,t,ix,res,usr);CHKERRQ(ierr);
      ierr = DecompactRheologyVars(3,res,eta_eff,eta_v,eta_e,eta_p,zeta_eff,zeta_v,zeta_e,zeta_p,chis,chip,txx,tzz,txz,tII,DP,dotlam,Z,G,C,sigmat,theta);CHKERRQ(ierr);

      P[0] = pc[4]; P[1] = Plc[4]; P[2] = DPoldc[4]; P[3] = strain[4];
      ierr = GetTensorPointValues(i,jp,ix,_eps,e);CHKERRQ(ierr);
      ierr = GetTensorPointValues(i,jp,ix,_tauold,t);CHKERRQ(ierr);
      ierr = RheologyPointwise(i,jp,xwt,iwtc,Tc[4],phic[4],P,e,t,ix,res,usr);CHKERRQ(ierr);
      ierr = DecompactRheologyVars(4,res,eta_eff,eta_v,eta_e,eta_p,zeta_eff,zeta_v,zeta_e,zeta_p,chis,chip,txx,tzz,txz,tII,DP,dotlam,Z,G,C,sigmat,theta);CHKERRQ(ierr);

      // corner points
      PetscScalar Tcorner[4], phicorner[4], pcorner[4], Plcorner[4], DPoldcorner[4], straincorner[4];
      ierr = GetCornerAvgFromCenter(Tc,Tcorner);CHKERRQ(ierr);
      ierr = GetCornerAvgFromCenter(phic,phicorner);CHKERRQ(ierr);
      ierr = GetCornerAvgFromCenter(pc,pcorner);CHKERRQ(ierr);
      ierr = GetCornerAvgFromCenter(Plc,Plcorner);CHKERRQ(ierr);
      ierr = GetCornerAvgFromCenter(DPoldc,DPoldcorner);CHKERRQ(ierr);
      ierr = GetCornerAvgFromCenter(strain,straincorner);CHKERRQ(ierr);
      
      ii = 0; ix[0] = ixxn[ii]; ix[1] = izzn[ii]; ix[2] = ixzn[ii]; ix[3] = iIIn[ii]; P[0] = pcorner[ii]; P[1] = Plcorner[ii]; P[2] = DPoldcorner[ii]; P[3] = straincorner[ii];
      ierr = GetTensorPointValues(i,j,ix,_eps,e);CHKERRQ(ierr);
      ierr = GetTensorPointValues(i,j,ix,_tauold,t);CHKERRQ(ierr);
      ierr = RheologyPointwise(i,j,xwt,iwtld,Tcorner[0],phicorner[0],P,e,t,ix,res,usr);CHKERRQ(ierr);
      ierr = DecompactRheologyVars(5,res,eta_eff,eta_v,eta_e,eta_p,zeta_eff,zeta_v,zeta_e,zeta_p,chis,chip,txx,tzz,txz,tII,DP,dotlam,Z,G,C,sigmat,theta);CHKERRQ(ierr);

      ii = 1; ix[0] = ixxn[ii]; ix[1] = izzn[ii]; ix[2] = ixzn[ii]; ix[3] = iIIn[ii]; P[0] = pcorner[ii]; P[1] = Plcorner[ii]; P[2] = DPoldcorner[ii]; P[3] = straincorner[ii];
      ierr = GetTensorPointValues(i,j,ix,_eps,e);CHKERRQ(ierr);
      ierr = GetTensorPointValues(i,j,ix,_tauold,t);CHKERRQ(ierr);
      ierr = RheologyPointwise(i,j,xwt,iwtrd,Tcorner[1],phicorner[1],P,e,t,ix,res,usr);CHKERRQ(ierr);
      ierr = DecompactRheologyVars(6,res,eta_eff,eta_v,eta_e,eta_p,zeta_eff,zeta_v,zeta_e,zeta_p,chis,chip,txx,tzz,txz,tII,DP,dotlam,Z,G,C,sigmat,theta);CHKERRQ(ierr);

      ii = 2; ix[0] = ixxn[ii]; ix[1] = izzn[ii]; ix[2] = ixzn[ii]; ix[3] = iIIn[ii]; P[0] = pcorner[ii]; P[1] = Plcorner[ii]; P[2] = DPoldcorner[ii]; P[3] = straincorner[ii];
      ierr = GetTensorPointValues(i,j,ix,_eps,e);CHKERRQ(ierr);
      ierr = GetTensorPointValues(i,j,ix,_tauold,t);CHKERRQ(ierr);
      ierr = RheologyPointwise(i,j,xwt,iwtlu,Tcorner[2],phicorner[2],P,e,t,ix,res,usr);CHKERRQ(ierr);
      ierr = DecompactRheologyVars(7,res,eta_eff,eta_v,eta_e,eta_p,zeta_eff,zeta_v,zeta_e,zeta_p,chis,chip,txx,tzz,txz,tII,DP,dotlam,Z,G,C,sigmat,theta);CHKERRQ(ierr);

      ii = 3; ix[0] = ixxn[ii]; ix[1] = izzn[ii]; ix[2] = ixzn[ii]; ix[3] = iIIn[ii]; P[0] = pcorner[ii]; P[1] = Plcorner[ii]; P[2] = DPoldcorner[ii]; P[3] = straincorner[ii];
      ierr = GetTensorPointValues(i,j,ix,_eps,e);CHKERRQ(ierr);
      ierr = GetTensorPointValues(i,j,ix,_tauold,t);CHKERRQ(ierr);
      ierr = RheologyPointwise(i,j,xwt,iwtru,Tcorner[3],phicorner[3],P,e,t,ix,res,usr);CHKERRQ(ierr);
      ierr = DecompactRheologyVars(8,res,eta_eff,eta_v,eta_e,eta_p,zeta_eff,zeta_v,zeta_e,zeta_p,chis,chip,txx,tzz,txz,tII,DP,dotlam,Z,G,C,sigmat,theta);CHKERRQ(ierr);

      { // element 
        // C = 0 (center, c=0)
        c[j][i][e_slot[iC]] = 0.0;

        // A = eta_eff,phi (center, c=1)
        c[j][i][e_slot[iA]] = eta_eff[0];
      }

      { // corner
        // A = eta_eff,phi (corner, c=0)
        for (ii = 0; ii < 4; ii++) {
          c[j][i][av_slot[ii]] = eta_eff[5+ii];
        }
      }
      
      { // face
        PetscScalar   phi[4], B[4], divchitau[4],gradchidp[4],gradPlith[4];
        PetscScalar   rhof[4], rhos[4], rho[4], wt[MAX_MAT_PHASE], rho0[MAX_MAT_PHASE];
        PetscInt      idx[MAX_MAT_PHASE];

        // porosity on edges - not for variable grid spacing
        phi[0] = (phic[0]+phic[1])*0.5;
        phi[1] = (phic[0]+phic[2])*0.5;
        phi[2] = (phic[0]+phic[3])*0.5;
        phi[3] = (phic[0]+phic[4])*0.5;

        gradPlith[0] = (Plc[0]-Plc[1])/dx;
        gradPlith[1] = (Plc[2]-Plc[0])/dx;
        gradPlith[2] = (Plc[0]-Plc[3])/dz;
        gradPlith[3] = (Plc[4]-Plc[0])/dz;

        gradchidp[0] = (chip[0]*_DPold[j ][i ][iP] - chip[1]*_DPold[j ][im][iP])/dx;
        gradchidp[1] = (chip[2]*_DPold[j ][ip][iP] - chip[0]*_DPold[j ][i ][iP])/dx;
        gradchidp[2] = (chip[0]*_DPold[j ][i ][iP] - chip[3]*_DPold[jm][i ][iP])/dz;
        gradchidp[3] = (chip[4]*_DPold[jp][i ][iP] - chip[0]*_DPold[j ][i ][iP])/dz;

        // div(chis*tau_old) = div(S) = [dSxx/dx+dSxz/dz, dSzx/dx+dSzz/dz]
        divchitau[0] = (chis[0]*_tauold[j][i ][ixx] - chis[1]*_tauold[j][im][ixx])/dx + (chis[7]*_tauold[j][i][ixzn[2]]-chis[5]*_tauold[j][i][ixzn[0]])/dz;
        divchitau[1] = (chis[2]*_tauold[j][ip][ixx] - chis[0]*_tauold[j][i ][ixx])/dx + (chis[8]*_tauold[j][i][ixzn[3]]-chis[6]*_tauold[j][i][ixzn[1]])/dz;
        divchitau[2] = (chis[6]*_tauold[j][i][ixzn[1]]-chis[5]*_tauold[j][i][ixzn[0]])/dx + (chis[0]*_tauold[j ][i][izz] - chis[3]*_tauold[jm][i][izz])/dz;
        divchitau[3] = (chis[8]*_tauold[j][i][ixzn[3]]-chis[7]*_tauold[j][i][ixzn[2]])/dx + (chis[4]*_tauold[jp][i][izz] - chis[0]*_tauold[j ][i][izz])/dz;

        for (ii = 0; ii < 4; ii++) {
          // get material phase fraction
          if (ii == 0 ) { idx[0] = iwtl[0]; idx[1] = iwtl[1]; idx[2] = iwtl[2]; idx[3] = iwtl[3]; idx[4] = iwtl[4]; idx[5] = iwtl[5]; }
          if (ii == 1 ) { idx[0] = iwtr[0]; idx[1] = iwtr[1]; idx[2] = iwtr[2]; idx[3] = iwtr[3]; idx[4] = iwtr[4]; idx[5] = iwtr[5];}
          if (ii == 2 ) { idx[0] = iwtd[0]; idx[1] = iwtd[1]; idx[2] = iwtd[2]; idx[3] = iwtd[3]; idx[4] = iwtd[4]; idx[5] = iwtd[5];}
          if (ii == 3 ) { idx[0] = iwtu[0]; idx[1] = iwtu[1]; idx[2] = iwtu[2]; idx[3] = iwtu[3]; idx[4] = iwtu[4]; idx[5] = iwtu[5];}

          // get bulk density
          for (iph = 0; iph < usr->nph; iph++) { rho0[iph] = Density(usr->mat_nd[iph].rho0,usr->mat[iph].rho_func); }
          ierr = GetMatPhaseFraction(i,j,xwt,idx,usr->nph,wt); CHKERRQ(ierr);
          rhos[ii] = WeightAverageValue(rho0,wt,usr->nph); 
          rhof[ii] = usr->nd->rhof;
          rho[ii]  = Mixture(rhos[ii],rhof[ii],phi[ii]);
          
          B[ii] = -divchitau[ii]+gradchidp[ii] + gradPlith[ii] + rho[ii]*k_hat[ii];
          //B[ii] = -divchitau[ii]+gradchidp[ii] + gradPlith[ii]*i_hat[ii] - phi[ii]*k_hat[ii];

          // B = body force+elasticity (edges, c=0)
          c[j][i][b_slot[ii]] = B[ii];
        }
      }

      // save stresses for output + dotlam
      _tau[j][i][ixx]     = txx[0]; _tau[j][i][izz]     = tzz[0]; _tau[j][i][ixz]     = txz[0]; _tau[j][i][iII]     = tII[0];
      _tau[j][i][ixxn[0]] = txx[5]; _tau[j][i][izzn[0]] = tzz[5]; _tau[j][i][ixzn[0]] = txz[5]; _tau[j][i][iIIn[0]] = tII[5];
      _tau[j][i][ixxn[1]] = txx[6]; _tau[j][i][izzn[1]] = tzz[6]; _tau[j][i][ixzn[1]] = txz[6]; _tau[j][i][iIIn[1]] = tII[6];
      _tau[j][i][ixxn[2]] = txx[7]; _tau[j][i][izzn[2]] = tzz[7]; _tau[j][i][ixzn[2]] = txz[7]; _tau[j][i][iIIn[2]] = tII[7];
      _tau[j][i][ixxn[3]] = txx[8]; _tau[j][i][izzn[3]] = tzz[8]; _tau[j][i][ixzn[3]] = txz[8]; _tau[j][i][iIIn[3]] = tII[8];
      _DP[j][i][iP]    = DP[0];
      _plast[j][i][iP] = dotlam[0];

      // get density element
      PetscScalar rho0[MAX_MAT_PHASE],rho,wt[MAX_MAT_PHASE];
      for (iph = 0; iph < usr->nph; iph++) { rho0[iph] = Density(usr->mat_nd[iph].rho0,usr->mat[iph].rho_func); }
      ierr = GetMatPhaseFraction(i,j,xwt,iwtc,usr->nph,wt); CHKERRQ(ierr);
      rho  = WeightAverageValue(rho0,wt,usr->nph);

      // get permeability element - zero
      PetscScalar Kphi;
      Kphi  = Permeability(phic[0],usr->par->n);

      // save material properties for output
      _matProp[j][i][imat[MATPROP_ELEMENT_ETA]]    = eta_eff[0];
      _matProp[j][i][imat[MATPROP_ELEMENT_ETA_V]]  = eta_v[0];
      _matProp[j][i][imat[MATPROP_ELEMENT_ETA_E]]  = eta_e[0];
      _matProp[j][i][imat[MATPROP_ELEMENT_ETA_P]]  = eta_p[0];
      _matProp[j][i][imat[MATPROP_ELEMENT_ZETA]]   = zeta_eff[0];
      _matProp[j][i][imat[MATPROP_ELEMENT_ZETA_V]] = zeta_v[0];
      _matProp[j][i][imat[MATPROP_ELEMENT_ZETA_E]] = zeta_e[0];
      _matProp[j][i][imat[MATPROP_ELEMENT_ZETA_P]] = zeta_p[0];
      _matProp[j][i][imat[MATPROP_ELEMENT_Z]]      = Z[0];
      _matProp[j][i][imat[MATPROP_ELEMENT_G]]      = G[0];
      _matProp[j][i][imat[MATPROP_ELEMENT_C]]      = C[0];
      _matProp[j][i][imat[MATPROP_ELEMENT_SIGMAT]] = sigmat[0];
      _matProp[j][i][imat[MATPROP_ELEMENT_THETA]]  = theta[0];
      _matProp[j][i][imat[MATPROP_ELEMENT_RHO]]    = rho;
      _matProp[j][i][imat[MATPROP_ELEMENT_KPHI]]   = Kphi;
      _matProp[j][i][imat[MATPROP_ELEMENT_CHIP]]   = chip[0];
      _matProp[j][i][imat[MATPROP_ELEMENT_CHIS]]   = chis[0];
    }
  }

  // Restore arrays, local vectors
  ierr = DMStagRestoreProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArray(dmcoeff,coefflocal,&c);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  ierr = VecDestroy(&coefflocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArrayRead(dm,xlocal,&xx);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&xlocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArrayRead(usr->dmeps,xepslocal,&_eps);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmeps,&xepslocal); CHKERRQ(ierr);
  ierr = DMStagVecRestoreArrayRead(usr->dmeps,xtauoldlocal,&_tauold);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmeps,&xtauoldlocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArray(usr->dmeps,xtaulocal,&_tau);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(usr->dmeps,xtaulocal,INSERT_VALUES,usr->xtau); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (usr->dmeps,xtaulocal,INSERT_VALUES,usr->xtau); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmeps,&xtaulocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArrayRead(usr->dmT,xTlocal,&_Tc);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmT,&xTlocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArrayRead(usr->dmphi,xphilocal,&_phic);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmphi,&xphilocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArrayRead(usr->dmPlith,xPlithlocal,&_Plith);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmPlith,&xPlithlocal); CHKERRQ(ierr);
  ierr = DMStagVecRestoreArrayRead(usr->dmPlith,xDPoldlocal,&_DPold);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmPlith,&xDPoldlocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArray(usr->dmPlith,xDPlocal,&_DP);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(usr->dmPlith,xDPlocal,INSERT_VALUES,usr->xDP); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (usr->dmPlith,xDPlocal,INSERT_VALUES,usr->xDP); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmPlith,&xDPlocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArray(usr->dmPlith,xplastlocal,&_plast);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(usr->dmPlith,xplastlocal,INSERT_VALUES,usr->xplast); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (usr->dmPlith,xplastlocal,INSERT_VALUES,usr->xplast); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmPlith,&xplastlocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArrayRead(usr->dmPlith,xstrainlocal,&_strain);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmPlith,&xstrainlocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArray(usr->dmmatProp,xmatProplocal,&_matProp);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(usr->dmmatProp,xmatProplocal,INSERT_VALUES,usr->xmatProp); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (usr->dmmatProp,xmatProplocal,INSERT_VALUES,usr->xmatProp); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmmatProp,&xmatProplocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArray(usr->dmMPhase,xMPhaselocal,&xwt);CHKERRQ(ierr);
  ierr = VecDestroy(&xMPhaselocal); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormCoefficient_PV_Stokes_DPL"
PetscErrorCode FormCoefficient_PV_Stokes_DPL(FDPDE fd, DM dm, Vec x, DM dmcoeff, Vec coeff, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       i, j, ii, sx, sz, nx, nz, iprev, inext, icenter, Nx, Nz;
  PetscScalar    **coordx, **coordz, ***c, ***xx, ***xwt; 
  PetscScalar    ***_Tc, ***_phic, ***_Plith, ***_eps, ***_tauold, ***_DPold, ***_tau, ***_DP;
  PetscScalar     ***_plast, ***_matProp, ***_strain;
  Vec            coefflocal, xlocal, xTlocal, xMPhaselocal, xtaulocal, xDPlocal, xstrainlocal;
  Vec            xphilocal, xPlithlocal, xepslocal, xtauoldlocal, xDPoldlocal, xplastlocal, xmatProplocal;
  PetscScalar    k_hat[4], i_hat[4];
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  // parameters
  k_hat[0] = 0.0;
  k_hat[1] = 0.0;
  k_hat[2] = usr->par->k_hat;
  k_hat[3] = usr->par->k_hat;

  i_hat[0] = 1.0;
  i_hat[1] = 1.0;
  i_hat[2] = 0.0;
  i_hat[3] = 0.0;

  // Get coefficient
  ierr = DMStagGetGlobalSizes(dmcoeff, &Nx, &Nz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmcoeff,LEFT,&iprev);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmcoeff,RIGHT,&inext);CHKERRQ(ierr); 
  ierr = DMStagGetProductCoordinateLocationSlot(dmcoeff,ELEMENT,&icenter);CHKERRQ(ierr);

  // Create coefficient local vector
  ierr = DMCreateLocalVector(dmcoeff, &coefflocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dmcoeff, coefflocal, &c); CHKERRQ(ierr);

  // Strain rates
  // ierr = UpdateStrainRates(dm,x,usr); CHKERRQ(ierr);
  ierr = UpdateStrainRates_Array(dm,x,usr); CHKERRQ(ierr);
  ierr = DMGetLocalVector(usr->dmeps, &xepslocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmeps, usr->xeps, INSERT_VALUES, xepslocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(usr->dmeps,xepslocal,&_eps);CHKERRQ(ierr);

  ierr = DMGetLocalVector(usr->dmeps, &xtauoldlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmeps, usr->xtau_old, INSERT_VALUES, xtauoldlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(usr->dmeps,xtauoldlocal,&_tauold);CHKERRQ(ierr);

  ierr = DMGetLocalVector(usr->dmeps, &xtaulocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmeps, usr->xtau, INSERT_VALUES, xtaulocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(usr->dmeps,xtaulocal,&_tau);CHKERRQ(ierr);

  // Get dm and solution vector
  ierr = DMGetLocalVector(dm, &xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm, x, INSERT_VALUES, xlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(dm,xlocal,&xx);CHKERRQ(ierr);
  
  // Get solution vector for temperature
  ierr = DMGetLocalVector(usr->dmT,&xTlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmT,usr->xT,INSERT_VALUES,xTlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(usr->dmT,xTlocal,&_Tc);CHKERRQ(ierr);

  // Get porosity
  ierr = DMGetLocalVector(usr->dmphi,&xphilocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmphi,usr->xphi,INSERT_VALUES,xphilocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(usr->dmphi,xphilocal,&_phic);CHKERRQ(ierr);

  // Get dm and vector Plith
  ierr = DMGetLocalVector(usr->dmPlith, &xPlithlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmPlith, usr->xPlith, INSERT_VALUES, xPlithlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(usr->dmPlith,xPlithlocal,&_Plith);CHKERRQ(ierr);

  ierr = DMGetLocalVector(usr->dmPlith, &xDPoldlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmPlith, usr->xDP_old, INSERT_VALUES, xDPoldlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(usr->dmPlith,xDPoldlocal,&_DPold);CHKERRQ(ierr);

  ierr = DMGetLocalVector(usr->dmPlith, &xDPlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmPlith, usr->xDP, INSERT_VALUES, xDPlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(usr->dmPlith,xDPlocal,&_DP);CHKERRQ(ierr);

  ierr = DMGetLocalVector(usr->dmPlith, &xplastlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmPlith, usr->xplast, INSERT_VALUES, xplastlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(usr->dmPlith,xplastlocal,&_plast);CHKERRQ(ierr);

  ierr = DMGetLocalVector(usr->dmPlith, &xstrainlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmPlith, usr->xstrain, INSERT_VALUES, xstrainlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(usr->dmPlith,xstrainlocal,&_strain);CHKERRQ(ierr);

  // get material phase fractions
  ierr = DMCreateLocalVector(usr->dmMPhase, &xMPhaselocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(usr->dmMPhase,usr->xMPhase,INSERT_VALUES,xMPhaselocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (usr->dmMPhase,usr->xMPhase,INSERT_VALUES,xMPhaselocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(usr->dmMPhase, xMPhaselocal, &xwt); CHKERRQ(ierr);

  // Get material properties - for output
  ierr = DMGetLocalVector(usr->dmmatProp, &xmatProplocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmmatProp, usr->xmatProp, INSERT_VALUES, xmatProplocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(usr->dmmatProp,xmatProplocal,&_matProp);CHKERRQ(ierr);

  // get location slots
  PetscInt  iE,iP;
  ierr = DMStagGetLocationSlot(usr->dmT,ELEMENT,T_ELEMENT,&iE);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmPlith,ELEMENT,0,&iP);CHKERRQ(ierr);

  PetscInt iPV;
  ierr = DMStagGetLocationSlot(dm,ELEMENT,PV_ELEMENT_P,&iPV);CHKERRQ(ierr);

  PetscInt  e_slot[3],av_slot[4],b_slot[4],iL,iR,iU,iD,iC,iA;
  iL = 0; iR = 1; iD = 2; iU  = 3;
  iC = 0; iA = 1; 
  ierr = DMStagGetLocationSlot(dmcoeff,ELEMENT,   PVCOEFF_ELEMENT_C,   &e_slot[iC]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,ELEMENT,   PVCOEFF_ELEMENT_A,   &e_slot[iA]);CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(dmcoeff,DOWN_LEFT, PVCOEFF_VERTEX_A,&av_slot[0]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DOWN_RIGHT,PVCOEFF_VERTEX_A,&av_slot[1]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,UP_LEFT,   PVCOEFF_VERTEX_A,&av_slot[2]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,UP_RIGHT,  PVCOEFF_VERTEX_A,&av_slot[3]);CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(dmcoeff,LEFT,   PVCOEFF_FACE_B,   &b_slot[iL]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,RIGHT,  PVCOEFF_FACE_B,   &b_slot[iR]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DOWN,   PVCOEFF_FACE_B,   &b_slot[iD]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,UP,     PVCOEFF_FACE_B,   &b_slot[iU]);CHKERRQ(ierr);

  PetscInt iwtc[MAX_MAT_PHASE],iwtl[MAX_MAT_PHASE],iwtr[MAX_MAT_PHASE],iwtd[MAX_MAT_PHASE],iwtu[MAX_MAT_PHASE];
  PetscInt iwtld[MAX_MAT_PHASE],iwtrd[MAX_MAT_PHASE],iwtlu[MAX_MAT_PHASE],iwtru[MAX_MAT_PHASE];
  for (ii = 0; ii < usr->nph; ii++) { 
    ierr = DMStagGetLocationSlot(usr->dmMPhase, ELEMENT,    ii, &iwtc[ii]); CHKERRQ(ierr); 
    ierr = DMStagGetLocationSlot(usr->dmMPhase, LEFT,       ii, &iwtl[ii]); CHKERRQ(ierr);
    ierr = DMStagGetLocationSlot(usr->dmMPhase, RIGHT,      ii, &iwtr[ii]); CHKERRQ(ierr);
    ierr = DMStagGetLocationSlot(usr->dmMPhase, DOWN,       ii, &iwtd[ii]); CHKERRQ(ierr); 
    ierr = DMStagGetLocationSlot(usr->dmMPhase, UP,         ii, &iwtu[ii]); CHKERRQ(ierr);
    ierr = DMStagGetLocationSlot(usr->dmMPhase, DOWN_LEFT,  ii, &iwtld[ii]); CHKERRQ(ierr);
    ierr = DMStagGetLocationSlot(usr->dmMPhase, DOWN_RIGHT, ii, &iwtrd[ii]); CHKERRQ(ierr);
    ierr = DMStagGetLocationSlot(usr->dmMPhase, UP_LEFT,    ii, &iwtlu[ii]); CHKERRQ(ierr);
    ierr = DMStagGetLocationSlot(usr->dmMPhase, UP_RIGHT,   ii, &iwtru[ii]); CHKERRQ(ierr);
  }

  PetscInt ixx, izz, ixz, iII, ixxn[4], izzn[4], ixzn[4], iIIn[4];
  ierr = DMStagGetLocationSlot(usr->dmeps, ELEMENT, 0, &ixx); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, ELEMENT, 1, &izz); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, ELEMENT, 2, &ixz); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, ELEMENT, 3, &iII); CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(usr->dmeps, DOWN_LEFT, 0, &ixxn[0]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, DOWN_LEFT, 1, &izzn[0]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, DOWN_LEFT, 2, &ixzn[0]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, DOWN_LEFT, 3, &iIIn[0]); CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(usr->dmeps, DOWN_RIGHT, 0, &ixxn[1]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, DOWN_RIGHT, 1, &izzn[1]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, DOWN_RIGHT, 2, &ixzn[1]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, DOWN_RIGHT, 3, &iIIn[1]); CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(usr->dmeps, UP_LEFT, 0, &ixxn[2]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, UP_LEFT, 1, &izzn[2]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, UP_LEFT, 2, &ixzn[2]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, UP_LEFT, 3, &iIIn[2]); CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(usr->dmeps, UP_RIGHT, 0, &ixxn[3]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, UP_RIGHT, 1, &izzn[3]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, UP_RIGHT, 2, &ixzn[3]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmeps, UP_RIGHT, 3, &iIIn[3]); CHKERRQ(ierr);

  PetscInt imat[MATPROP_NPROP];
  for (ii = 0; ii < MATPROP_NPROP; ii++) { ierr = DMStagGetLocationSlot(usr->dmmatProp, ELEMENT, ii, &imat[ii]); CHKERRQ(ierr); }
  
  // Loop over local domain - set initial density and viscosity
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      PetscScalar   phic[9], pc[9], Plc[9], Tc[9], DPoldc[9], strain[9], dx, dz;
      PetscInt      iph, ii, im, jm, ip, jp;

      if (i == 0   ) im = i; else im = i-1;
      if (i == Nx-1) ip = i; else ip = i+1;
      if (j == 0   ) jm = j; else jm = j-1;
      if (j == Nz-1) jp = j; else jp = j+1;

      // should be adapted to variable spacing
      dx = coordx[i][inext]-coordx[i][iprev];
      dz = coordz[j][inext]-coordz[j][iprev];

      // get porosity, p, Plith, T - center
      ierr = Get9PointCenterValues(i,j,iE,Nx,Nz,_phic,phic);CHKERRQ(ierr);
      ierr = Get9PointCenterValues(i,j,iP,Nx,Nz,_Plith,Plc);CHKERRQ(ierr);
      ierr = Get9PointCenterValues(i,j,iPV,Nx,Nz,xx,pc);CHKERRQ(ierr);
      ierr = Get9PointCenterValues(i,j,iE,Nx,Nz,_Tc,Tc);CHKERRQ(ierr);
      ierr = Get9PointCenterValues(i,j,iP,Nx,Nz,_DPold,DPoldc);CHKERRQ(ierr);
      ierr = Get9PointCenterValues(i,j,iP,Nx,Nz,_strain,strain);CHKERRQ(ierr);

      // set porosity phi = 0
      for (ii = 0; ii < 9; ii++) { phic[ii] = 0.0; }

      // Prepare for pointwise rheology calculation
      PetscScalar eta_eff[9], zeta_eff[9], chis[9], chip[9], txx[9], tzz[9], txz[9], tII[9], DP[9], DPdl[9], dotlam[9];
      PetscScalar eta_v[9],eta_e[9],eta_p[9],zeta_v[9],zeta_e[9];
      PetscScalar e[4], t[4], P[4], res[21], Z[9], G[9], C[9], sigmat[9], theta[9];
      PetscInt ix[4];

      // center points
      ix[0] = ixx; ix[1] = izz; ix[2] = ixz; ix[3] = iII;

      P[0] = pc[0]; P[1] = Plc[0]; P[2] = DPoldc[0]; P[3] = strain[0];
      ierr = GetTensorPointValues(i,j,ix,_eps,e);CHKERRQ(ierr);
      ierr = GetTensorPointValues(i,j,ix,_tauold,t);CHKERRQ(ierr);
      ierr = RheologyPointwise_DPL(i,j,xwt,iwtc,Tc[0],phic[0],P,e,t,ix,res,usr);CHKERRQ(ierr);
      ierr = DecompactRheologyVars_DPL(0,res,eta_eff,eta_v,eta_e,eta_p,zeta_eff,zeta_v,zeta_e,DPdl,chis,chip,txx,tzz,txz,tII,DP,dotlam,Z,G,C,sigmat,theta);CHKERRQ(ierr);

      P[0] = pc[1]; P[1] = Plc[1]; P[2] = DPoldc[1]; P[3] = strain[1];
      ierr = GetTensorPointValues(im,j,ix,_eps,e);CHKERRQ(ierr);
      ierr = GetTensorPointValues(im,j,ix,_tauold,t);CHKERRQ(ierr);
      ierr = RheologyPointwise_DPL(im,j,xwt,iwtc,Tc[1],phic[1],P,e,t,ix,res,usr);CHKERRQ(ierr);
      ierr = DecompactRheologyVars_DPL(1,res,eta_eff,eta_v,eta_e,eta_p,zeta_eff,zeta_v,zeta_e,DPdl,chis,chip,txx,tzz,txz,tII,DP,dotlam,Z,G,C,sigmat,theta);CHKERRQ(ierr);

      P[0] = pc[2]; P[1] = Plc[2]; P[2] = DPoldc[2]; P[3] = strain[2];
      ierr = GetTensorPointValues(ip,j,ix,_eps,e);CHKERRQ(ierr);
      ierr = GetTensorPointValues(ip,j,ix,_tauold,t);CHKERRQ(ierr);
      ierr = RheologyPointwise_DPL(ip,j,xwt,iwtc,Tc[2],phic[2],P,e,t,ix,res,usr);CHKERRQ(ierr);
      ierr = DecompactRheologyVars_DPL(2,res,eta_eff,eta_v,eta_e,eta_p,zeta_eff,zeta_v,zeta_e,DPdl,chis,chip,txx,tzz,txz,tII,DP,dotlam,Z,G,C,sigmat,theta);CHKERRQ(ierr);

      P[0] = pc[3]; P[1] = Plc[3]; P[2] = DPoldc[3]; P[3] = strain[3];
      ierr = GetTensorPointValues(i,jm,ix,_eps,e);CHKERRQ(ierr);
      ierr = GetTensorPointValues(i,jm,ix,_tauold,t);CHKERRQ(ierr);
      ierr = RheologyPointwise_DPL(i,jm,xwt,iwtc,Tc[3],phic[3],P,e,t,ix,res,usr);CHKERRQ(ierr);
      ierr = DecompactRheologyVars_DPL(3,res,eta_eff,eta_v,eta_e,eta_p,zeta_eff,zeta_v,zeta_e,DPdl,chis,chip,txx,tzz,txz,tII,DP,dotlam,Z,G,C,sigmat,theta);CHKERRQ(ierr);

      P[0] = pc[4]; P[1] = Plc[4]; P[2] = DPoldc[4]; P[3] = strain[4];
      ierr = GetTensorPointValues(i,jp,ix,_eps,e);CHKERRQ(ierr);
      ierr = GetTensorPointValues(i,jp,ix,_tauold,t);CHKERRQ(ierr);
      ierr = RheologyPointwise_DPL(i,jp,xwt,iwtc,Tc[4],phic[4],P,e,t,ix,res,usr);CHKERRQ(ierr);
      ierr = DecompactRheologyVars_DPL(4,res,eta_eff,eta_v,eta_e,eta_p,zeta_eff,zeta_v,zeta_e,DPdl,chis,chip,txx,tzz,txz,tII,DP,dotlam,Z,G,C,sigmat,theta);CHKERRQ(ierr);

      // corner points
      PetscScalar Tcorner[4], phicorner[4], pcorner[4], Plcorner[4], DPoldcorner[4], straincorner[4];
      ierr = GetCornerAvgFromCenter(Tc,Tcorner);CHKERRQ(ierr);
      ierr = GetCornerAvgFromCenter(phic,phicorner);CHKERRQ(ierr);
      ierr = GetCornerAvgFromCenter(pc,pcorner);CHKERRQ(ierr);
      ierr = GetCornerAvgFromCenter(Plc,Plcorner);CHKERRQ(ierr);
      ierr = GetCornerAvgFromCenter(DPoldc,DPoldcorner);CHKERRQ(ierr);
      ierr = GetCornerAvgFromCenter(strain,straincorner);CHKERRQ(ierr);
      
      ii = 0; ix[0] = ixxn[ii]; ix[1] = izzn[ii]; ix[2] = ixzn[ii]; ix[3] = iIIn[ii]; P[0] = pcorner[ii]; P[1] = Plcorner[ii]; P[2] = DPoldcorner[ii]; P[3] = straincorner[ii];
      ierr = GetTensorPointValues(i,j,ix,_eps,e);CHKERRQ(ierr);
      ierr = GetTensorPointValues(i,j,ix,_tauold,t);CHKERRQ(ierr);
      ierr = RheologyPointwise_DPL(i,j,xwt,iwtld,Tcorner[0],phicorner[0],P,e,t,ix,res,usr);CHKERRQ(ierr);
      ierr = DecompactRheologyVars_DPL(5,res,eta_eff,eta_v,eta_e,eta_p,zeta_eff,zeta_v,zeta_e,DPdl,chis,chip,txx,tzz,txz,tII,DP,dotlam,Z,G,C,sigmat,theta);CHKERRQ(ierr);

      ii = 1; ix[0] = ixxn[ii]; ix[1] = izzn[ii]; ix[2] = ixzn[ii]; ix[3] = iIIn[ii]; P[0] = pcorner[ii]; P[1] = Plcorner[ii]; P[2] = DPoldcorner[ii]; P[3] = straincorner[ii];
      ierr = GetTensorPointValues(i,j,ix,_eps,e);CHKERRQ(ierr);
      ierr = GetTensorPointValues(i,j,ix,_tauold,t);CHKERRQ(ierr);
      ierr = RheologyPointwise_DPL(i,j,xwt,iwtrd,Tcorner[1],phicorner[1],P,e,t,ix,res,usr);CHKERRQ(ierr);
      ierr = DecompactRheologyVars_DPL(6,res,eta_eff,eta_v,eta_e,eta_p,zeta_eff,zeta_v,zeta_e,DPdl,chis,chip,txx,tzz,txz,tII,DP,dotlam,Z,G,C,sigmat,theta);CHKERRQ(ierr);

      ii = 2; ix[0] = ixxn[ii]; ix[1] = izzn[ii]; ix[2] = ixzn[ii]; ix[3] = iIIn[ii]; P[0] = pcorner[ii]; P[1] = Plcorner[ii]; P[2] = DPoldcorner[ii]; P[3] = straincorner[ii];
      ierr = GetTensorPointValues(i,j,ix,_eps,e);CHKERRQ(ierr);
      ierr = GetTensorPointValues(i,j,ix,_tauold,t);CHKERRQ(ierr);
      ierr = RheologyPointwise_DPL(i,j,xwt,iwtlu,Tcorner[2],phicorner[2],P,e,t,ix,res,usr);CHKERRQ(ierr);
      ierr = DecompactRheologyVars_DPL(7,res,eta_eff,eta_v,eta_e,eta_p,zeta_eff,zeta_v,zeta_e,DPdl,chis,chip,txx,tzz,txz,tII,DP,dotlam,Z,G,C,sigmat,theta);CHKERRQ(ierr);

      ii = 3; ix[0] = ixxn[ii]; ix[1] = izzn[ii]; ix[2] = ixzn[ii]; ix[3] = iIIn[ii]; P[0] = pcorner[ii]; P[1] = Plcorner[ii]; P[2] = DPoldcorner[ii]; P[3] = straincorner[ii];
      ierr = GetTensorPointValues(i,j,ix,_eps,e);CHKERRQ(ierr);
      ierr = GetTensorPointValues(i,j,ix,_tauold,t);CHKERRQ(ierr);
      ierr = RheologyPointwise_DPL(i,j,xwt,iwtru,Tcorner[3],phicorner[3],P,e,t,ix,res,usr);CHKERRQ(ierr);
      ierr = DecompactRheologyVars_DPL(8,res,eta_eff,eta_v,eta_e,eta_p,zeta_eff,zeta_v,zeta_e,DPdl,chis,chip,txx,tzz,txz,tII,DP,dotlam,Z,G,C,sigmat,theta);CHKERRQ(ierr);

      { // element 
        // C = 0 (center, c=0)
        c[j][i][e_slot[iC]] = 0.0;

        // A = eta_eff,phi (center, c=1)
        c[j][i][e_slot[iA]] = eta_eff[0];
      }

      { // corner
        // A = eta_eff,phi (corner, c=0)
        for (ii = 0; ii < 4; ii++) {
          c[j][i][av_slot[ii]] = eta_eff[5+ii];
        }
      }

      { // face
        PetscScalar   phi[4], B[4], divchitau[4],gradchidp[4], gradPlith[4], gradDPdl[4];
        PetscScalar   rhof[4], rhos[4], rho[4], wt[MAX_MAT_PHASE], rho0[MAX_MAT_PHASE];
        PetscInt      idx[MAX_MAT_PHASE];

        // porosity on edges - not for variable grid spacing
        phi[0] = (phic[0]+phic[1])*0.5;
        phi[1] = (phic[0]+phic[2])*0.5;
        phi[2] = (phic[0]+phic[3])*0.5;
        phi[3] = (phic[0]+phic[4])*0.5;

        gradPlith[0] = (Plc[0]-Plc[1])/dx;
        gradPlith[1] = (Plc[2]-Plc[0])/dx;
        gradPlith[2] = (Plc[0]-Plc[3])/dz;
        gradPlith[3] = (Plc[4]-Plc[0])/dz;

        gradDPdl[0] = (DPdl[0]-DPdl[1])/dx;
        gradDPdl[1] = (DPdl[2]-DPdl[0])/dx;
        gradDPdl[2] = (DPdl[0]-DPdl[3])/dz;
        gradDPdl[3] = (DPdl[4]-DPdl[0])/dz;

        gradchidp[0] = (chip[0]*_DPold[j ][i ][iP] - chip[1]*_DPold[j ][im][iP])/dx;
        gradchidp[1] = (chip[2]*_DPold[j ][ip][iP] - chip[0]*_DPold[j ][i ][iP])/dx;
        gradchidp[2] = (chip[0]*_DPold[j ][i ][iP] - chip[3]*_DPold[jm][i ][iP])/dz;
        gradchidp[3] = (chip[4]*_DPold[jp][i ][iP] - chip[0]*_DPold[j ][i ][iP])/dz;

        //  div(chis*tau_old) = div(S) = [dSxx/dx+dSxz/dz, dSzx/dx+dSzz/dz]
        divchitau[0] = (chis[0]*_tauold[j][i ][ixx] - chis[1]*_tauold[j][im][ixx])/dx + (chis[7]*_tauold[j][i][ixzn[2]]-chis[5]*_tauold[j][i][ixzn[0]])/dz;
        divchitau[1] = (chis[2]*_tauold[j][ip][ixx] - chis[0]*_tauold[j][i ][ixx])/dx + (chis[8]*_tauold[j][i][ixzn[3]]-chis[6]*_tauold[j][i][ixzn[1]])/dz;
        divchitau[2] = (chis[6]*_tauold[j][i][ixzn[1]]-chis[5]*_tauold[j][i][ixzn[0]])/dx + (chis[0]*_tauold[j ][i][izz] - chis[3]*_tauold[jm][i][izz])/dz;
        divchitau[3] = (chis[8]*_tauold[j][i][ixzn[3]]-chis[7]*_tauold[j][i][ixzn[2]])/dx + (chis[4]*_tauold[jp][i][izz] - chis[0]*_tauold[j ][i][izz])/dz;

        for (ii = 0; ii < 4; ii++) {
          // get material phase fraction
          if (ii == 0 ) { idx[0] = iwtl[0]; idx[1] = iwtl[1]; idx[2] = iwtl[2]; idx[3] = iwtl[3]; idx[4] = iwtl[4]; idx[5] = iwtl[5]; }
          if (ii == 1 ) { idx[0] = iwtr[0]; idx[1] = iwtr[1]; idx[2] = iwtr[2]; idx[3] = iwtr[3]; idx[4] = iwtr[4]; idx[5] = iwtr[5];}
          if (ii == 2 ) { idx[0] = iwtd[0]; idx[1] = iwtd[1]; idx[2] = iwtd[2]; idx[3] = iwtd[3]; idx[4] = iwtd[4]; idx[5] = iwtd[5];}
          if (ii == 3 ) { idx[0] = iwtu[0]; idx[1] = iwtu[1]; idx[2] = iwtu[2]; idx[3] = iwtu[3]; idx[4] = iwtu[4]; idx[5] = iwtu[5];}

          // get bulk density
          for (iph = 0; iph < usr->nph; iph++) { rho0[iph] = Density(usr->mat_nd[iph].rho0,usr->mat[iph].rho_func); }
          ierr = GetMatPhaseFraction(i,j,xwt,idx,usr->nph,wt); CHKERRQ(ierr);
          rhos[ii] = WeightAverageValue(rho0,wt,usr->nph); 
          rhof[ii] = usr->nd->rhof;
          rho[ii]  = Mixture(rhos[ii],rhof[ii],phi[ii]);

          // add buoyancy terms here
          B[ii] = -divchitau[ii]+gradchidp[ii] +gradDPdl[ii] + gradPlith[ii] + rho[ii]*k_hat[ii];
          // B[ii] = -divchitau[ii]+gradchidp[ii] +gradDPdl[ii] + gradPlith[ii]*i_hat[ii] - phi[ii]*k_hat[ii];

          // B = body force+elasticity (edges, c=0)
          c[j][i][b_slot[ii]] = B[ii];
        }
      }

      // save stresses for output + dotlam
      _tau[j][i][ixx]     = txx[0]; _tau[j][i][izz]     = tzz[0]; _tau[j][i][ixz]     = txz[0]; _tau[j][i][iII]     = tII[0];
      _tau[j][i][ixxn[0]] = txx[5]; _tau[j][i][izzn[0]] = tzz[5]; _tau[j][i][ixzn[0]] = txz[5]; _tau[j][i][iIIn[0]] = tII[5];
      _tau[j][i][ixxn[1]] = txx[6]; _tau[j][i][izzn[1]] = tzz[6]; _tau[j][i][ixzn[1]] = txz[6]; _tau[j][i][iIIn[1]] = tII[6];
      _tau[j][i][ixxn[2]] = txx[7]; _tau[j][i][izzn[2]] = tzz[7]; _tau[j][i][ixzn[2]] = txz[7]; _tau[j][i][iIIn[2]] = tII[7];
      _tau[j][i][ixxn[3]] = txx[8]; _tau[j][i][izzn[3]] = tzz[8]; _tau[j][i][ixzn[3]] = txz[8]; _tau[j][i][iIIn[3]] = tII[8];
      _DP[j][i][iP]    = DP[0];
      _plast[j][i][iP] = dotlam[0];

      // get density (center)
      PetscScalar rho0[MAX_MAT_PHASE],rho,wt[MAX_MAT_PHASE];
      for (iph = 0; iph < usr->nph; iph++) { rho0[iph] = Density(usr->mat_nd[iph].rho0,usr->mat[iph].rho_func); }
      ierr = GetMatPhaseFraction(i,j,xwt,iwtc,usr->nph,wt); CHKERRQ(ierr);
      rho  = WeightAverageValue(rho0,wt,usr->nph);

      // get permeability (center)
      PetscScalar Kphi;
      Kphi  = Permeability(phic[0],usr->par->n);

      // save material properties for output
      _matProp[j][i][imat[MATPROP_ELEMENT_ETA]]    = eta_eff[0];
      _matProp[j][i][imat[MATPROP_ELEMENT_ETA_V]]  = eta_v[0];
      _matProp[j][i][imat[MATPROP_ELEMENT_ETA_E]]  = eta_e[0];
      _matProp[j][i][imat[MATPROP_ELEMENT_ETA_P]]  = eta_p[0];
      _matProp[j][i][imat[MATPROP_ELEMENT_ZETA]]   = zeta_eff[0];
      _matProp[j][i][imat[MATPROP_ELEMENT_ZETA_V]] = zeta_v[0];
      _matProp[j][i][imat[MATPROP_ELEMENT_ZETA_E]] = zeta_e[0];
      _matProp[j][i][imat[MATPROP_ELEMENT_ZETA_P]] = DPdl[0];
      _matProp[j][i][imat[MATPROP_ELEMENT_Z]]      = Z[0];
      _matProp[j][i][imat[MATPROP_ELEMENT_G]]      = G[0];
      _matProp[j][i][imat[MATPROP_ELEMENT_C]]      = C[0];
      _matProp[j][i][imat[MATPROP_ELEMENT_SIGMAT]] = sigmat[0];
      _matProp[j][i][imat[MATPROP_ELEMENT_THETA]]  = theta[0];
      _matProp[j][i][imat[MATPROP_ELEMENT_RHO]]    = rho;
      _matProp[j][i][imat[MATPROP_ELEMENT_KPHI]]   = Kphi;
      _matProp[j][i][imat[MATPROP_ELEMENT_CHIP]]   = chip[0];
      _matProp[j][i][imat[MATPROP_ELEMENT_CHIS]]   = chis[0];
    }
  }

  // Restore arrays, local vectors
  ierr = DMStagRestoreProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArray(dmcoeff,coefflocal,&c);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  ierr = VecDestroy(&coefflocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArrayRead(dm,xlocal,&xx);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&xlocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArrayRead(usr->dmeps,xepslocal,&_eps);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmeps,&xepslocal); CHKERRQ(ierr);
  ierr = DMStagVecRestoreArrayRead(usr->dmeps,xtauoldlocal,&_tauold);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmeps,&xtauoldlocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArray(usr->dmeps,xtaulocal,&_tau);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(usr->dmeps,xtaulocal,INSERT_VALUES,usr->xtau); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (usr->dmeps,xtaulocal,INSERT_VALUES,usr->xtau); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmeps,&xtaulocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArrayRead(usr->dmT,xTlocal,&_Tc);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmT,&xTlocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArrayRead(usr->dmphi,xphilocal,&_phic);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmphi,&xphilocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArrayRead(usr->dmPlith,xPlithlocal,&_Plith);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmPlith,&xPlithlocal); CHKERRQ(ierr);
  ierr = DMStagVecRestoreArrayRead(usr->dmPlith,xDPoldlocal,&_DPold);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmPlith,&xDPoldlocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArray(usr->dmPlith,xDPlocal,&_DP);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(usr->dmPlith,xDPlocal,INSERT_VALUES,usr->xDP); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (usr->dmPlith,xDPlocal,INSERT_VALUES,usr->xDP); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmPlith,&xDPlocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArray(usr->dmPlith,xplastlocal,&_plast);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(usr->dmPlith,xplastlocal,INSERT_VALUES,usr->xplast); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (usr->dmPlith,xplastlocal,INSERT_VALUES,usr->xplast); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmPlith,&xplastlocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArrayRead(usr->dmPlith,xstrainlocal,&_strain);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmPlith,&xstrainlocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArray(usr->dmmatProp,xmatProplocal,&_matProp);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(usr->dmmatProp,xmatProplocal,INSERT_VALUES,usr->xmatProp); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (usr->dmmatProp,xmatProplocal,INSERT_VALUES,usr->xmatProp); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmmatProp,&xmatProplocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArray(usr->dmMPhase,xMPhaselocal,&xwt);CHKERRQ(ierr);
  ierr = VecDestroy(&xMPhaselocal); CHKERRQ(ierr);
 
  PetscFunctionReturn(0);
}

// ---------------------------------------
PetscErrorCode DecompactRheologyVars(PetscInt ii,PetscScalar *res,PetscScalar *eta_eff,PetscScalar *eta_v,PetscScalar *eta_e,PetscScalar *eta_p,PetscScalar *zeta_eff,
                                     PetscScalar *zeta_v,PetscScalar *zeta_e,PetscScalar *zeta_p,PetscScalar *chis,PetscScalar *chip,
                                     PetscScalar *txx,PetscScalar *tzz,PetscScalar *txz,PetscScalar *tII,PetscScalar *DP,PetscScalar *dotlam,
                                     PetscScalar *Z,PetscScalar *G,PetscScalar *C,PetscScalar *sigmat,PetscScalar *theta)
{
  PetscFunctionBegin;
    eta_eff[ii] = res[0]; 
    eta_v[ii]   = res[1]; 
    eta_e[ii]   = res[2]; 
    eta_p[ii]   = res[3]; 
    zeta_eff[ii]= res[4]; 
    zeta_v[ii]  = res[5]; 
    zeta_e[ii]  = res[6]; 
    zeta_p[ii]  = res[7];
    chis[ii]    = res[8]; 
    chip[ii]    = res[9]; 
    txx[ii]     = res[10]; 
    tzz[ii]     = res[11]; 
    txz[ii]     = res[12]; 
    tII[ii]     = res[13]; 
    DP[ii]      = res[14]; 
    dotlam[ii]  = res[15];
    Z[ii]       = res[16];
    G[ii]       = res[17];
    C[ii]       = res[18];
    sigmat[ii]  = res[19];
    theta[ii]   = res[20];

  PetscFunctionReturn(0);
}

// ---------------------------------------
PetscErrorCode DecompactRheologyVars_DPL(PetscInt ii,PetscScalar *res,PetscScalar *eta_eff,PetscScalar *eta_v,PetscScalar *eta_e,PetscScalar *eta_p,PetscScalar *zeta_eff,
                                     PetscScalar *zeta_v,PetscScalar *zeta_e,PetscScalar *DPdl,PetscScalar *chis,PetscScalar *chip,
                                     PetscScalar *txx,PetscScalar *tzz,PetscScalar *txz,PetscScalar *tII,PetscScalar *DP,PetscScalar *dotlam,
                                     PetscScalar *Z,PetscScalar *G,PetscScalar *C,PetscScalar *sigmat,PetscScalar *theta)
{
  PetscFunctionBegin;
    eta_eff[ii] = res[0]; 
    eta_v[ii]   = res[1]; 
    eta_e[ii]   = res[2]; 
    eta_p[ii]   = res[3]; 
    zeta_eff[ii]= res[4]; 
    zeta_v[ii]  = res[5]; 
    zeta_e[ii]  = res[6]; 
    DPdl[ii]    = res[7];
    chis[ii]    = res[8]; 
    chip[ii]    = res[9]; 
    txx[ii]     = res[10]; 
    tzz[ii]     = res[11]; 
    txz[ii]     = res[12]; 
    tII[ii]     = res[13]; 
    DP[ii]      = res[14]; 
    dotlam[ii]  = res[15];
    Z[ii]       = res[16];
    G[ii]       = res[17];
    C[ii]       = res[18];
    sigmat[ii]  = res[19];
    theta[ii]   = res[20];

  PetscFunctionReturn(0);
}

// ---------------------------------------
// RheologyPointwise
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "RheologyPointwise"
PetscErrorCode RheologyPointwise(PetscInt i, PetscInt j, PetscScalar ***xwt, PetscInt *iwt, PetscScalar T, PetscScalar phi, PetscScalar *P, 
                                 PetscScalar *eps, PetscScalar *tauold, PetscInt *ix, PetscScalar *res, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  if (usr->par->rheology==0) { ierr = RheologyPointwise_VEP(i,j,xwt,iwt,T,phi,P,eps,tauold,ix,res,usr);CHKERRQ(ierr); }
  if (usr->par->rheology==1) { ierr = RheologyPointwise_VEVP(i,j,xwt,iwt,T,phi,P,eps,tauold,ix,res,usr);CHKERRQ(ierr); }

  PetscFunctionReturn(0);
}

// ---------------------------------------
// RheologyPointwise_VEP
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "RheologyPointwise_VEP"
PetscErrorCode RheologyPointwise_VEP(PetscInt i, PetscInt j, PetscScalar ***xwt, PetscInt *iwt, PetscScalar T, PetscScalar phi, PetscScalar *P, 
                                 PetscScalar *eps, PetscScalar *tauold, PetscInt *ix, PetscScalar *res, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       iph;
  PetscScalar    dt, p, Plith, DPold, Tdim, phis, strain;
  PetscScalar    eta_v, zeta_v, eta_e, zeta_e, eta_p, zeta_p, eta_ve, zeta_ve, eta, zeta, chip, chis;

  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  dt = usr->nd->dt;
  p = P[0]; Plith = P[1]; DPold = P[2]; strain = P[3];
  Tdim = dim_paramT(T,usr->par->Ttop,usr->scal->DT);
  phis = 1.0 - phi;

  // get epsII, epsp, epspII - tau_old, dP
  PetscScalar exx, ezz, exz, eII, told_xx, told_zz, told_xz, told_II, Pf;
  PetscScalar exxp, ezzp, exzp, div13, divp, eIIp;
  exx = eps[0]; told_xx = tauold[0]; 
  ezz = eps[1]; told_zz = tauold[1];
  exz = eps[2]; told_xz = tauold[2];
  eII = eps[3]; told_II = tauold[3];
  div13 = (exx + ezz)/3.0;
  Pf    = p + Plith;

  // get marker phase and properties
  PetscScalar  wt[MAX_MAT_PHASE], meta_v[MAX_MAT_PHASE], mzeta_v[MAX_MAT_PHASE], meta_e[MAX_MAT_PHASE], mzeta_e[MAX_MAT_PHASE];
  PetscScalar  meta_ve[MAX_MAT_PHASE], mzeta_ve[MAX_MAT_PHASE], mC[MAX_MAT_PHASE], mZ[MAX_MAT_PHASE], mG[MAX_MAT_PHASE];
  PetscScalar  msigmat[MAX_MAT_PHASE],mtheta[MAX_MAT_PHASE];
  PetscScalar  inv_meta_v[MAX_MAT_PHASE], inv_meta_e[MAX_MAT_PHASE], inv_mzeta_v[MAX_MAT_PHASE], inv_mzeta_e[MAX_MAT_PHASE];
  PetscScalar  meta[MAX_MAT_PHASE], mzeta[MAX_MAT_PHASE], mchis[MAX_MAT_PHASE], mchip[MAX_MAT_PHASE], meta_p[MAX_MAT_PHASE];
  PetscScalar  meta_VEP[MAX_MAT_PHASE], mzeta_p[MAX_MAT_PHASE], mzeta_VEP[MAX_MAT_PHASE];

  ierr = GetMatPhaseFraction(i,j,xwt,iwt,usr->nph,wt); CHKERRQ(ierr);

  for (iph = 0; iph < usr->nph; iph++) {
    meta_v[iph]  = ShearViscosity(usr->mat_nd[iph].eta0,Tdim,phi,usr->par->EoR,usr->par->Teta0,usr->par->beta,usr->mat_nd[iph].eta_func);
    mzeta_v[iph] = CompactionViscosity(usr->mat_nd[iph].zeta0,Tdim,phi,usr->par->EoR,usr->par->Teta0,usr->par->phi_min,usr->par->zetaExp,usr->mat_nd[iph].zeta_func); 
    
    inv_meta_v[iph]  = 1.0/meta_v[iph];
    inv_mzeta_v[iph] = 1.0/mzeta_v[iph];

    meta_e[iph]  = usr->mat_nd[iph].G*dt;
    mzeta_e[iph] = usr->mat_nd[iph].Z0*dt; // PoroElasticModulus(usr->mat_nd[iph].Z0,usr->nd->Zmax,phi)*dt;

    inv_meta_e[iph]  = 1.0/meta_e[iph];
    inv_mzeta_e[iph] = 1.0/mzeta_e[iph];

    // visco-elastic 
    meta_ve[iph]  = PetscPowScalar(inv_meta_v[iph]+inv_meta_e[iph],-1.0);
    mzeta_ve[iph] = PetscPowScalar(inv_mzeta_v[iph]+inv_mzeta_e[iph],-1.0);

    // elastic and plastic parameters
    mZ[iph] = usr->mat_nd[iph].Z0;
    mG[iph] = usr->mat_nd[iph].G;
    mC[iph] = usr->mat_nd[iph].C;
    msigmat[iph] = usr->mat_nd[iph].sigmat;
    mtheta[iph]  = usr->mat_nd[iph].theta;

    // add softening to cohesion and friction angle - need to add dotlam calculation here

    // effective deviatoric and volumetric strain rates
    exxp = ((exx-div13) + 0.5*told_xx*inv_meta_e[iph]);
    ezzp = ((ezz-div13) + 0.5*told_zz*inv_meta_e[iph]);
    exzp = (exz + 0.5*told_xz*inv_meta_e[iph]);
    eIIp = TensorSecondInvariant(exxp,ezzp,exzp);
    divp = ((exx+ezz) - DPold*inv_mzeta_e[iph]);

    // plastic viscosity
    if (usr->plasticity) { 
      PetscScalar Y;
      // von Mises
      // Y = mC[iph]; 

      // Drucker-Prager Pf 1
      // Y = mC[iph]*PetscCosScalar(PETSC_PI*mtheta[iph]/180) + Pf*PetscSinScalar(PETSC_PI*mtheta[iph]/180); 

      // Drucker-Prager Plith 2
      // Y = mC[iph]*PetscCosScalar(PETSC_PI*mtheta[iph]/180) + Plith*PetscSinScalar(PETSC_PI*mtheta[iph]/180); 

      // Drucker-Prager 3
      // Y = mC[iph] + Plith*PetscSinScalar(PETSC_PI*mtheta[iph]/180); 

      // Drucker-Prager 4
      // Y = mC[iph] + Pf*PetscSinScalar(PETSC_PI*mtheta[iph]/180); 

      // hyperbolic surface
      PetscScalar aa, bb;
      // aa = mC[iph]*PetscCosScalar(PETSC_PI*mtheta[iph]/180) + Pf*PetscSinScalar(PETSC_PI*mtheta[iph]/180);
      aa = mC[iph]*PetscCosScalar(PETSC_PI*mtheta[iph]/180) + Plith*PetscSinScalar(PETSC_PI*mtheta[iph]/180);
      bb = mC[iph]*PetscCosScalar(PETSC_PI*mtheta[iph]/180) - msigmat[iph]*PetscSinScalar(PETSC_PI*mtheta[iph]/180);
      Y = PetscPowScalar(aa*aa-bb*bb,0.5);

      meta_p[iph]   = Y/(2.0*eIIp); 
      mzeta_p[iph]  = PetscMin(usr->nd->eta_max,Y/PetscAbs(exx+ezz)); 

      // calculate dotlam

    } else { 
      meta_p[iph]    = usr->nd->eta_max;
      mzeta_p[iph]   = usr->nd->eta_max;
    }

    meta_VEP[iph]  = PetscMin(meta_p[iph],meta_ve[iph]);
    mzeta_VEP[iph] = PetscMin(mzeta_p[iph],meta_ve[iph]);

    // effective viscosities
    meta[iph] = ViscosityHarmonicAvg(meta_VEP[iph],usr->nd->eta_min,usr->nd->eta_max)*phis;
    mzeta[iph]= ViscosityHarmonicAvg(mzeta_VEP[iph],usr->nd->eta_min,usr->nd->eta_max)*phis;

    // elastic stress evolution parameter
    mchis[iph] = meta[iph]*inv_meta_e[iph];
    mchip[iph] = mzeta[iph]*inv_meta_e[iph];
  }

  eta_v  = WeightAverageValue(meta_v,wt,usr->nph); 
  zeta_v = WeightAverageValue(mzeta_v,wt,usr->nph); 
  eta_e  = WeightAverageValue(meta_e,wt,usr->nph); 
  zeta_e = WeightAverageValue(mzeta_e,wt,usr->nph); 
  eta_ve = WeightAverageValue(meta_ve,wt,usr->nph); 
  zeta_ve= WeightAverageValue(mzeta_ve,wt,usr->nph); 
  eta_p  = WeightAverageValue(meta_p,wt,usr->nph); 
  zeta_p = WeightAverageValue(mzeta_p,wt,usr->nph); 
  eta    = WeightAverageValue(meta,wt,usr->nph); 
  zeta   = WeightAverageValue(mzeta,wt,usr->nph); 
  chis   = WeightAverageValue(mchis,wt,usr->nph); 
  chip   = WeightAverageValue(mchip,wt,usr->nph); 

  PetscScalar C, G, Z, sigmat, theta;
  G      = WeightAverageValue(mG,wt,usr->nph); 
  Z      = WeightAverageValue(mZ,wt,usr->nph); 
  C      = WeightAverageValue(mC,wt,usr->nph); 
  sigmat = WeightAverageValue(msigmat,wt,usr->nph); 
  theta  = WeightAverageValue(mtheta,wt,usr->nph); 

  // update effective deviatoric and volumetric strain rates with Marker PhaseAverage
  exxp = ((exx-div13) + 0.5*told_xx/eta_e);
  ezzp = ((ezz-div13) + 0.5*told_zz/eta_e);
  exzp = (exz + 0.5*told_xz/eta_e);
  divp = ((exx+ezz) - DPold/zeta_e);

  // shear and volumetric stresses
  PetscScalar txx, tzz, txz, tII, DP;
  txx = 2*eta*exxp/phis;
  tzz = 2*eta*ezzp/phis;
  txz = 2*eta*exzp/phis;
  tII = TensorSecondInvariant(txx,tzz,txz);
  DP = -zeta*divp/phis;

  // return values
  res[0]  = eta;
  res[1]  = eta_v;
  res[2]  = eta_e;
  res[3]  = eta_p;
  res[4]  = zeta;
  res[5]  = zeta_v;
  res[6]  = zeta_e;
  res[7]  = zeta_p;
  res[8]  = chis;
  res[9]  = chip;
  res[10] = txx;
  res[11] = tzz;
  res[12] = txz;
  res[13] = tII;
  res[14] = DP;
  res[15] = 0.0;
  res[16] = Z;
  res[17] = G;
  res[18] = C;
  res[19] = sigmat;
  res[20] = theta;

  PetscFunctionReturn(0);
}

// ---------------------------------------
// RheologyPointwise_VEVP
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "RheologyPointwise_VEVP"
PetscErrorCode RheologyPointwise_VEVP(PetscInt i, PetscInt j, PetscScalar ***xwt, PetscInt *iwt, PetscScalar T, PetscScalar phi, PetscScalar *P, 
                                 PetscScalar *eps, PetscScalar *tauold, PetscInt *ix, PetscScalar *res, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       iph;
  PetscScalar    dt, tf_tol, p, Plith, DPold, Tdim, phis, dotlam, strain;
  PetscScalar    eta_v, zeta_v, eta_e, zeta_e, eta_p, zeta_p, eta, zeta, chip, chis;

  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  dt = usr->nd->dt;
  tf_tol = usr->par->tf_tol;

  p = P[0]; Plith = P[1]; DPold = P[2]; strain = P[3];
  Tdim = dim_paramT(T,usr->par->Ttop,usr->scal->DT);
  phis = 1.0 - phi;

  // get epsII, epsp, epspII - tau_old, dP
  PetscScalar exx, ezz, exz, eII, told_xx, told_zz, told_xz, told_II, Pf;
  PetscScalar exxp, ezzp, exzp, div13, divp, eIIp;
  PetscScalar txxt, tzzt, txzt, tIIt, dpt;
  PetscScalar aP,theta_rad;

  exx = eps[0]; told_xx = tauold[0]; 
  ezz = eps[1]; told_zz = tauold[1];
  exz = eps[2]; told_xz = tauold[2];
  eII = eps[3]; told_II = tauold[3];
  div13 = (exx + ezz)/3.0;
  Pf    = p + Plith;

  // get marker phase and properties
  PetscScalar  wt[MAX_MAT_PHASE], meta_v[MAX_MAT_PHASE], mzeta_v[MAX_MAT_PHASE], meta_e[MAX_MAT_PHASE], mzeta_e[MAX_MAT_PHASE];
  PetscScalar  meta_p[MAX_MAT_PHASE], mzeta_p[MAX_MAT_PHASE], meta_ve[MAX_MAT_PHASE], mzeta_ve[MAX_MAT_PHASE];
  PetscScalar  mC[MAX_MAT_PHASE], mZ[MAX_MAT_PHASE], mG[MAX_MAT_PHASE], msigmat[MAX_MAT_PHASE], mtheta[MAX_MAT_PHASE];
  PetscScalar  inv_meta_v[MAX_MAT_PHASE], inv_meta_e[MAX_MAT_PHASE], inv_mzeta_v[MAX_MAT_PHASE], inv_mzeta_e[MAX_MAT_PHASE];
  PetscScalar  meta[MAX_MAT_PHASE], mzeta[MAX_MAT_PHASE], mchis[MAX_MAT_PHASE], mchip[MAX_MAT_PHASE];
  PetscScalar  meta_VEP[MAX_MAT_PHASE], mzeta_VEP[MAX_MAT_PHASE], mdotlam[MAX_MAT_PHASE];

  ierr = GetMatPhaseFraction(i,j,xwt,iwt,usr->nph,wt); CHKERRQ(ierr);

  for (iph = 0; iph < usr->nph; iph++) {
    if (wt[iph]==0.0) { // do not solve rheology if material phase not present
      meta_v[iph]  = 0.0;
      mzeta_v[iph] = 0.0;
      meta_e[iph]  = 0.0;
      mzeta_e[iph] = 0.0;
      meta_p[iph]  = 0.0;
      mzeta_p[iph] = 0.0;
      meta[iph]    = 0.0;
      mzeta[iph]   = 0.0;
      mchis[iph]   = 0.0;
      mchip[iph]   = 0.0;
      mdotlam[iph] = 0.0;
      mG[iph]      = 0.0;
      mZ[iph]      = 0.0;
      mC[iph]      = 0.0;
      msigmat[iph] = 0.0;
      mtheta[iph]  = 0.0;
    } else {
      // viscous rheology
      meta_v[iph]  = ShearViscosity(usr->mat_nd[iph].eta0,Tdim,phi,usr->par->EoR,usr->par->Teta0,usr->par->beta,usr->mat_nd[iph].eta_func);
      mzeta_v[iph] = CompactionViscosity(usr->mat_nd[iph].zeta0,Tdim,phi,usr->par->EoR,usr->par->Teta0,usr->par->phi_min,usr->par->zetaExp,usr->mat_nd[iph].zeta_func); 
      
      meta_v[iph] = ViscosityHarmonicAvg(meta_v[iph],usr->nd->eta_min,usr->nd->eta_max);
      mzeta_v[iph]= ViscosityHarmonicAvg(mzeta_v[iph],usr->nd->eta_min,usr->nd->eta_max);

      inv_meta_v[iph]  = 1.0/meta_v[iph];
      inv_mzeta_v[iph] = 1.0/mzeta_v[iph];

      // elastic and plastic parameters
      mZ[iph] = PoroElasticModulus(usr->mat_nd[iph].Z0,usr->nd->Zmax,phi);
      mG[iph] = ElasticShearModulus(usr->mat_nd[iph].G,phi);
      mC[iph] = usr->mat_nd[iph].C;
      mtheta[iph]  = usr->mat_nd[iph].theta;
    
      // elastic rheology
      meta_e[iph]  = mG[iph]*dt;
      mzeta_e[iph] = mZ[iph]*dt;

      inv_meta_e[iph]  = 1.0/meta_e[iph];
      inv_mzeta_e[iph] = 1.0/mzeta_e[iph];

      // visco-elastic 
      meta_ve[iph]  = PetscPowScalar(inv_meta_v[iph]+inv_meta_e[iph],-1.0);
      mzeta_ve[iph] = PetscPowScalar(inv_mzeta_v[iph]+inv_mzeta_e[iph],-1.0);

      // add softening to cohesion and friction angle
      PetscScalar xsoft, noise;
      xsoft = strain/usr->par->strain_max;
      noise = 0.0;
      mC[iph]     = mC[iph]     * (1.0+noise) * PetscMax((1.0-usr->par->hcc*xsoft),usr->par->hcc);
      mtheta[iph] = mtheta[iph] * (1.0+noise) * PetscMax((1.0-usr->par->hcc*xsoft),usr->par->hcc);

      // tensile strength
      msigmat[iph] = TensileStrength(mC[iph],4.0,usr->mat_nd[iph].sigmat,0); 
      // msigmat[iph] = TensileStrength(mC[iph],4.0,usr->mat_nd[iph].sigmat,1); // user defined sigmat

      // modified deviatoric and volumetric strain rates
      exxp = ((exx-div13) + 0.5*phis*told_xx*inv_meta_e[iph]);
      ezzp = ((ezz-div13) + 0.5*phis*told_zz*inv_meta_e[iph]);
      exzp = (exz + 0.5*phis*told_xz*inv_meta_e[iph]);
      eIIp = TensorSecondInvariant(exxp,ezzp,exzp);
      divp = ((exx+ezz) - phis*DPold*inv_mzeta_e[iph]);

      // trial stress
      txxt = 2*meta_ve[iph]*exxp/phis;
      tzzt = 2*meta_ve[iph]*ezzp/phis;
      txzt = 2*meta_ve[iph]*exzp/phis;
      tIIt = TensorSecondInvariant(txxt,tzzt,txzt);
      dpt = -mzeta_ve[iph] * divp/phis;

      // visco-plastic
      if (usr->plasticity) { 
        theta_rad = PETSC_PI*mtheta[iph]/180;

        PetscScalar xve[4], stressSol[3];
        xve[0] = tIIt;
        xve[1] = dpt;
        xve[2] = meta_ve[iph];
        xve[3] = mzeta_ve[iph];

        ierr = Plastic_LocalSolver(xve,mC[iph],msigmat[iph],theta_rad,Pf,phi,usr,stressSol); CHKERRQ(ierr);
        mdotlam[iph] = stressSol[2];

        // effective viscosities
        if (PetscAbs(eIIp) > tf_tol) { meta_VEP[iph] = 0.5 * phis * stressSol[0]/eIIp; } 
        else               { meta_VEP[iph] = meta_ve[iph]; }

        if (PetscAbs(divp) > tf_tol) { mzeta_VEP[iph] = - phis * stressSol[1]/divp; } 
        else               { mzeta_VEP[iph] = mzeta_ve[iph]; }
        
        // visco-plastic viscosity - for output
        PetscScalar aa, cdl;
        cdl = PetscExpScalar(-usr->par->phi_min/phi);
        aa  = mC[iph]*PetscCosScalar(theta_rad) - msigmat[iph]*PetscSinScalar(theta_rad);
        if (mdotlam[iph]>0.0) { meta_p[iph] = phis * PetscPowScalar(stressSol[0]*stressSol[0]+aa*aa, 0.5)/mdotlam[iph]; } 
        else                  { meta_p[iph] = usr->nd->eta_max; }
          
        if ((mdotlam[iph]>0.0) & (stressSol[1]<0.0) & (cdl>0.0)) { 
          mzeta_p[iph] = -phis * stressSol[1]/(mdotlam[iph]*cdl*PetscSinScalar(theta_rad)); 
        } else { mzeta_p[iph] = usr->nd->eta_max; }

      } else { 
        meta_VEP[iph] = meta_ve[iph]; 
        mzeta_VEP[iph]= mzeta_ve[iph];
        mdotlam[iph]  = 0.0;

        meta_p[iph]   = usr->nd->eta_max;
        mzeta_p[iph]  = usr->nd->eta_max;
      }

      // effective viscosities - check cutoff
      meta[iph] = ViscosityHarmonicAvg(meta_VEP[iph],usr->nd->eta_min,usr->nd->eta_max);
      mzeta[iph]= ViscosityHarmonicAvg(mzeta_VEP[iph],usr->nd->eta_min,usr->nd->eta_max);

      // elastic stress evolution parameter
      mchis[iph] = phis * meta[iph]*inv_meta_e[iph];
      mchip[iph] = phis * mzeta[iph]*inv_meta_e[iph];
    }
  }

  eta_v  = WeightAverageValue(meta_v,wt,usr->nph); 
  zeta_v = WeightAverageValue(mzeta_v,wt,usr->nph); 
  eta_e  = WeightAverageValue(meta_e,wt,usr->nph); 
  zeta_e = WeightAverageValue(mzeta_e,wt,usr->nph); 
  eta_p  = WeightAverageValue(meta_p,wt,usr->nph); 
  zeta_p = WeightAverageValue(mzeta_p,wt,usr->nph); 
  eta    = WeightAverageValue(meta,wt,usr->nph); 
  zeta   = WeightAverageValue(mzeta,wt,usr->nph); 
  chis   = WeightAverageValue(mchis,wt,usr->nph); 
  chip   = WeightAverageValue(mchip,wt,usr->nph); 
  dotlam = WeightAverageValue(mdotlam,wt,usr->nph); 

  PetscScalar C, G, Z, sigmat, theta;
  G      = WeightAverageValue(mG,wt,usr->nph); 
  Z      = WeightAverageValue(mZ,wt,usr->nph); 
  C      = WeightAverageValue(mC,wt,usr->nph); 
  sigmat = WeightAverageValue(msigmat,wt,usr->nph); 
  theta  = WeightAverageValue(mtheta,wt,usr->nph); 

  // update effective deviatoric and volumetric strain rates with Marker PhaseAverage
  exxp = ((exx-div13) + 0.5*phis*told_xx/eta_e);
  ezzp = ((ezz-div13) + 0.5*phis*told_zz/eta_e);
  exzp = (exz + 0.5*phis*told_xz/eta_e);
  divp = ((exx+ezz) - phis*DPold/zeta_e);

  // shear and volumetric stresses
  PetscScalar txx, tzz, txz, tII, DP;
  txx = 2*eta*exxp/phis;
  tzz = 2*eta*ezzp/phis;
  txz = 2*eta*exzp/phis;
  tII = TensorSecondInvariant(txx,tzz,txz);
  DP = -zeta*divp/phis;

  // return values
  res[0]  = eta;
  res[1]  = eta_v;
  res[2]  = eta_e;
  res[3]  = eta_p;
  res[4]  = zeta;
  res[5]  = zeta_v;
  res[6]  = zeta_e;
  res[7]  = zeta_p;
  res[8]  = chis;
  res[9]  = chip;
  res[10] = txx;
  res[11] = tzz;
  res[12] = txz;
  res[13] = tII;
  res[14] = DP;
  res[15] = dotlam;
  res[16] = Z;
  res[17] = G;
  res[18] = C;
  res[19] = sigmat;
  res[20] = theta;

  PetscFunctionReturn(0);
}

// ---------------------------------------
// RheologyPointwise_DPL
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "RheologyPointwise_DPL"
PetscErrorCode RheologyPointwise_DPL(PetscInt i, PetscInt j, PetscScalar ***xwt, PetscInt *iwt, PetscScalar T, PetscScalar phi, PetscScalar *P, 
                                 PetscScalar *eps, PetscScalar *tauold, PetscInt *ix, PetscScalar *res, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       iph;
  PetscScalar    dt, tf_tol, p, Plith, DPold, Tdim, phis, dotlam, strain;
  PetscScalar    eta_v, zeta_v, eta_e, zeta_e, eta_p, zeta_p, eta, zeta, chip, chis;

  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  dt = usr->nd->dt;
  tf_tol = usr->par->tf_tol;

  p = P[0]; Plith = P[1]; DPold = P[2]; strain = P[3];
  Tdim = dim_paramT(T,usr->par->Ttop,usr->scal->DT);
  phis = 1.0 - phi;

  // get epsII, epsp, epspII - tau_old, dP
  PetscScalar exx, ezz, exz, eII, told_xx, told_zz, told_xz, told_II, Pf;
  PetscScalar exxp, ezzp, exzp, div13, divp, eIIp;
  PetscScalar txxt, tzzt, txzt, tIIt, dpt, DPdl;
  PetscScalar aP,theta_rad;

  exx = eps[0]; told_xx = tauold[0]; 
  ezz = eps[1]; told_zz = tauold[1];
  exz = eps[2]; told_xz = tauold[2];
  eII = eps[3]; told_II = tauold[3];
  div13 = (exx + ezz)/3.0;
  Pf    = p + Plith;

  // get marker phase and properties
  PetscScalar  wt[MAX_MAT_PHASE], meta_v[MAX_MAT_PHASE], mzeta_v[MAX_MAT_PHASE], meta_e[MAX_MAT_PHASE], mzeta_e[MAX_MAT_PHASE];
  PetscScalar  mC[MAX_MAT_PHASE], mZ[MAX_MAT_PHASE], mG[MAX_MAT_PHASE], msigmat[MAX_MAT_PHASE], mtheta[MAX_MAT_PHASE];
  PetscScalar  meta[MAX_MAT_PHASE], mzeta[MAX_MAT_PHASE], mchis[MAX_MAT_PHASE], mchip[MAX_MAT_PHASE], mdotlam[MAX_MAT_PHASE];
  PetscScalar  meta_p[MAX_MAT_PHASE], mDPdl[MAX_MAT_PHASE];

  ierr = GetMatPhaseFraction(i,j,xwt,iwt,usr->nph,wt); CHKERRQ(ierr);

  for (iph = 0; iph < usr->nph; iph++) {
    if (wt[iph]==0.0) { // do not solve rheology if material phase not present
      meta_v[iph]  = 0.0;
      mzeta_v[iph] = 0.0;
      meta_e[iph]  = 0.0;
      mzeta_e[iph] = 0.0;
      meta_p[iph]  = 0.0;
      mDPdl[iph]   = 0.0;
      meta[iph]    = 0.0;
      mzeta[iph]   = 0.0;
      mchis[iph]   = 0.0;
      mchip[iph]   = 0.0;
      mdotlam[iph] = 0.0;
      mG[iph]      = 0.0;
      mZ[iph]      = 0.0;
      mC[iph]      = 0.0;
      msigmat[iph] = 0.0;
      mtheta[iph]  = 0.0;
    } else {
      PetscScalar meta_ve, mzeta_ve, inv_meta_v, inv_meta_e, inv_mzeta_v, inv_mzeta_e;

      // viscous rheology
      meta_v[iph]  = ShearViscosity(usr->mat_nd[iph].eta0,Tdim,phi,usr->par->EoR,usr->par->Teta0,usr->par->beta,usr->mat_nd[iph].eta_func);
      mzeta_v[iph] = CompactionViscosity(usr->mat_nd[iph].zeta0,Tdim,phi,usr->par->EoR,usr->par->Teta0,usr->par->phi_min,usr->par->zetaExp,usr->mat_nd[iph].zeta_func); 

      // meta_v[iph] = ViscosityHarmonicAvg(meta_v[iph],usr->nd->eta_min,usr->nd->eta_max);
      // mzeta_v[iph]= ViscosityHarmonicAvg(mzeta_v[iph],usr->nd->eta_min,usr->nd->eta_max);

      inv_meta_v  = 1.0/meta_v[iph];
      inv_mzeta_v = 1.0/mzeta_v[iph];

      // elastic and plastic parameters
      mZ[iph] = PoroElasticModulus(usr->mat_nd[iph].Z0,usr->nd->Zmax,phi);
      mG[iph] = ElasticShearModulus(usr->mat_nd[iph].G,phi);
      mC[iph] = usr->mat_nd[iph].C;
      mtheta[iph]  = usr->mat_nd[iph].theta;
    
      // elastic rheology
      meta_e[iph]  = mG[iph]*dt;
      mzeta_e[iph] = mZ[iph]*dt;

      inv_meta_e  = 1.0/meta_e[iph];
      inv_mzeta_e = 1.0/mzeta_e[iph];

      // visco-elastic 
      meta_ve  = PetscPowScalar(inv_meta_v+inv_meta_e,-1.0);
      mzeta_ve = PetscPowScalar(inv_mzeta_v+inv_mzeta_e,-1.0);

      // add softening to cohesion and friction angle
      PetscScalar xsoft, noise;
      xsoft = strain/usr->par->strain_max;
      noise = 0.0;
      mC[iph]     = mC[iph]     * (1.0+noise) * PetscMax((1.0-usr->par->hcc*xsoft),usr->par->hcc);
      mtheta[iph] = mtheta[iph] * (1.0+noise) * PetscMax((1.0-usr->par->hcc*xsoft),usr->par->hcc);

      // tensile strength
      msigmat[iph] = TensileStrength(mC[iph],4.0,usr->mat_nd[iph].sigmat,0); 
      // msigmat[iph] = TensileStrength(mC[iph],4.0,usr->mat_nd[iph].sigmat,1); // user defined sigmat

      // initialize plastic viscosity
      meta_p[iph] = usr->nd->eta_max;

      // modified deviatoric and volumetric strain rates
      exxp = (exx-div13) + 0.5*phis*told_xx*inv_meta_e;
      ezzp = (ezz-div13) + 0.5*phis*told_zz*inv_meta_e;
      exzp = exz + 0.5*phis*told_xz*inv_meta_e;
      eIIp = TensorSecondInvariant(exxp,ezzp,exzp);
      divp = (exx+ezz) - phis*DPold*inv_mzeta_e;

      // trial VE stress
      txxt = 2*meta_ve*exxp/phis;
      tzzt = 2*meta_ve*ezzp/phis;
      txzt = 2*meta_ve*exzp/phis;
      tIIt = TensorSecondInvariant(txxt,tzzt,txzt);
      dpt = -mzeta_ve * divp/phis;

      // visco-plastic
      if (usr->plasticity) { 
        theta_rad = PETSC_PI*mtheta[iph]/180;

        PetscScalar xve[4], stressSol[3];
        xve[0] = tIIt;
        xve[1] = dpt;
        xve[2] = meta_ve;
        xve[3] = mzeta_ve;

        ierr = Plastic_LocalSolver(xve,mC[iph],msigmat[iph],theta_rad,Pf,phi,usr,stressSol); CHKERRQ(ierr);
        mdotlam[iph] = stressSol[2];

        PetscScalar cdl, A1, inv_meta_p;
        cdl = PetscExpScalar(-usr->par->phi_min/phi);
        A1  = mC[iph]*PetscCosScalar(theta_rad) - msigmat[iph]*PetscSinScalar(theta_rad);
        mDPdl[iph] = mdotlam[iph]*mzeta_ve*cdl*PetscSinScalar(theta_rad);

        // effective viscosity
        inv_meta_p = mdotlam[iph]/phis/PetscPowScalar(stressSol[0]*stressSol[0]+A1*A1, 0.5);
        if (inv_meta_p) meta_p[iph] = 1.0/inv_meta_p;
        meta[iph] = 1.0/(1.0/meta_ve+inv_meta_p);
        mzeta[iph]= mzeta_ve;

      } else { 
        meta[iph] = meta_ve; 
        mzeta[iph]= mzeta_ve;
        mdotlam[iph]  = 0.0;
        mDPdl[iph] = 0.0;
      }

      // elastic stress evolution parameter
      mchis[iph] = phis * meta[iph]*inv_meta_e;
      mchip[iph] = phis * mzeta[iph]*inv_meta_e;
    }
  }

  eta_v  = WeightAverageValue(meta_v,wt,usr->nph); 
  zeta_v = WeightAverageValue(mzeta_v,wt,usr->nph); 
  eta_e  = WeightAverageValue(meta_e,wt,usr->nph); 
  zeta_e = WeightAverageValue(mzeta_e,wt,usr->nph); 
  eta_p  = WeightAverageValue(meta_p,wt,usr->nph); 
  eta    = WeightAverageValue(meta,wt,usr->nph); 
  zeta   = WeightAverageValue(mzeta,wt,usr->nph); 
  chis   = WeightAverageValue(mchis,wt,usr->nph); 
  chip   = WeightAverageValue(mchip,wt,usr->nph); 
  dotlam = WeightAverageValue(mdotlam,wt,usr->nph); 
  DPdl   = WeightAverageValue(mDPdl,wt,usr->nph); 

  PetscScalar C, G, Z, sigmat, theta;
  G      = WeightAverageValue(mG,wt,usr->nph); 
  Z      = WeightAverageValue(mZ,wt,usr->nph); 
  C      = WeightAverageValue(mC,wt,usr->nph); 
  sigmat = WeightAverageValue(msigmat,wt,usr->nph); 
  theta  = WeightAverageValue(mtheta,wt,usr->nph); 

  // update effective deviatoric and volumetric strain rates with Marker PhaseAverage
  exxp = (exx-div13) + 0.5*phis*told_xx/eta_e;
  ezzp = (ezz-div13) + 0.5*phis*told_zz/eta_e;
  exzp = exz + 0.5*phis*told_xz/eta_e;
  divp = (exx+ezz) - phis*DPold/zeta_e;

  // shear and volumetric stresses
  PetscScalar txx, tzz, txz, tII, DP;
  txx = 2*eta*exxp/phis;
  tzz = 2*eta*ezzp/phis;
  txz = 2*eta*exzp/phis;
  tII = TensorSecondInvariant(txx,tzz,txz);
  DP = -zeta*divp/phis + DPdl/phis;

  // return values
  res[0]  = eta;
  res[1]  = eta_v;
  res[2]  = eta_e;
  res[3]  = eta_p;
  res[4]  = zeta;
  res[5]  = zeta_v;
  res[6]  = zeta_e;
  res[7]  = DPdl;
  res[8]  = chis;
  res[9]  = chip;
  res[10] = txx;
  res[11] = tzz;
  res[12] = txz;
  res[13] = tII;
  res[14] = DP;
  res[15] = dotlam;
  res[16] = Z;
  res[17] = G;
  res[18] = C;
  res[19] = sigmat;
  res[20] = theta;

  PetscFunctionReturn(0);
}

// ---------------------------------------
// FormCoefficient_T
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormCoefficient_T"
PetscErrorCode FormCoefficient_T(FDPDE fd, DM dm, Vec x, DM dmcoeff, Vec coeff, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       i, j, ii, sx, sz, nx, nz, Nx, Nz;
  DM             dmPV = NULL;
  Vec            coefflocal;
  PetscScalar    ***c, ***_xPVlocal, ***xwt;
  Vec            xPV = NULL, xPVlocal, xlocal, xMPhaselocal;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  // Get dm and solution vector for Stokes velocity
  dmPV = usr->dmPV;
  xPV  = usr->xPV;
  
  ierr = DMGetLocalVector(dmPV, &xPVlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dmPV, xPV, INSERT_VALUES, xPVlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(dmPV,xPVlocal,&_xPVlocal);CHKERRQ(ierr);
  
  // Get solution vector for temperature
  ierr = DMGetLocalVector(dm,&xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm,x,INSERT_VALUES,xlocal); CHKERRQ(ierr);

  // get material phase fractions
  ierr = DMCreateLocalVector(usr->dmMPhase, &xMPhaselocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(usr->dmMPhase,usr->xMPhase,INSERT_VALUES,xMPhaselocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (usr->dmMPhase,usr->xMPhase,INSERT_VALUES,xMPhaselocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(usr->dmMPhase, xMPhaselocal, &xwt); CHKERRQ(ierr);

  // Get coefficient local vector
  ierr = DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = DMStagGetGlobalSizes(dmcoeff,&Nx,&Nz,NULL);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dmcoeff, &coefflocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dmcoeff, coefflocal, &c); CHKERRQ(ierr);

  // get location slots
  PetscInt  pv_slot[4],iL,iR,iU,iD;
  iL = 0; iR  = 1;
  iD = 2; iU  = 3;
  ierr = DMStagGetLocationSlot(usr->dmPV,LEFT,   PV_FACE_VS,   &pv_slot[iL]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmPV,RIGHT,  PV_FACE_VS,   &pv_slot[iR]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmPV,DOWN,   PV_FACE_VS,   &pv_slot[iD]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmPV,UP,     PV_FACE_VS,   &pv_slot[iU]);CHKERRQ(ierr);

  PetscInt  e_slot[2],iA,iC;
  iA = 0; iC = 1;
  ierr = DMStagGetLocationSlot(dmcoeff,ELEMENT,TCOEFF_ELEMENT_A, &e_slot[iA]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,ELEMENT,TCOEFF_ELEMENT_C, &e_slot[iC]); CHKERRQ(ierr);

  PetscInt  B_slot[4],u_slot[4];
  ierr = DMStagGetLocationSlot(dmcoeff,LEFT, TCOEFF_FACE_B,&B_slot[0]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,RIGHT,TCOEFF_FACE_B,&B_slot[1]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DOWN, TCOEFF_FACE_B,&B_slot[2]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,UP,   TCOEFF_FACE_B,&B_slot[3]);CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(dmcoeff,LEFT, TCOEFF_FACE_u,&u_slot[0]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,RIGHT,TCOEFF_FACE_u,&u_slot[1]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DOWN, TCOEFF_FACE_u,&u_slot[2]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,UP,   TCOEFF_FACE_u,&u_slot[3]);CHKERRQ(ierr);

  PetscInt iwtl[MAX_MAT_PHASE],iwtr[MAX_MAT_PHASE],iwtd[MAX_MAT_PHASE],iwtu[MAX_MAT_PHASE]; 
  for (ii = 0; ii < usr->nph; ii++) { 
    ierr = DMStagGetLocationSlot(usr->dmMPhase, LEFT, ii, &iwtl[ii]); CHKERRQ(ierr);
    ierr = DMStagGetLocationSlot(usr->dmMPhase, RIGHT,ii, &iwtr[ii]); CHKERRQ(ierr);
    ierr = DMStagGetLocationSlot(usr->dmMPhase, DOWN, ii, &iwtd[ii]); CHKERRQ(ierr); 
    ierr = DMStagGetLocationSlot(usr->dmMPhase, UP,   ii, &iwtu[ii]); CHKERRQ(ierr);
  }

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {

      { // element
        // A = 1.0
        c[j][i][e_slot[iA]] = 1.0;
        
        // C = 0.0 - sources of heat/sink
        c[j][i][e_slot[iC]] = 0.0;
      }

      { // B = 1/Ra*kappa (edge)
        for (ii = 0; ii < 4; ii++) {
          PetscScalar rho0[MAX_MAT_PHASE], cp0[MAX_MAT_PHASE], kT0[MAX_MAT_PHASE], kappa0[MAX_MAT_PHASE], kappa, wt[MAX_MAT_PHASE];
          PetscInt    iph, idx[MAX_MAT_PHASE];

          // solid material density
          for (iph = 0; iph < usr->nph; iph++) { 
            rho0[iph] = Density(usr->mat_nd[iph].rho0,usr->mat[iph].rho_func); 
            cp0[iph] = usr->mat_nd[iph].cp;
            kT0[iph] = usr->mat_nd[iph].kT;
            kappa0[iph] = kT0[iph]/(rho0[iph]*cp0[iph]);
          }
         // get material phase fractions
          if (ii == 0 ) { idx[0] = iwtl[0]; idx[1] = iwtl[1]; idx[2] = iwtl[2]; idx[3] = iwtl[3]; idx[4] = iwtl[4]; idx[5] = iwtl[5];}
          if (ii == 1 ) { idx[0] = iwtr[0]; idx[1] = iwtr[1]; idx[2] = iwtr[2]; idx[3] = iwtr[3]; idx[4] = iwtr[4]; idx[5] = iwtr[5];}
          if (ii == 2 ) { idx[0] = iwtd[0]; idx[1] = iwtd[1]; idx[2] = iwtd[2]; idx[3] = iwtd[3]; idx[4] = iwtd[4]; idx[5] = iwtd[5];}
          if (ii == 3 ) { idx[0] = iwtu[0]; idx[1] = iwtu[1]; idx[2] = iwtu[2]; idx[3] = iwtu[3]; idx[4] = iwtu[4]; idx[5] = iwtu[5];}
          ierr = GetMatPhaseFraction(i,j,xwt,idx,usr->nph,wt); CHKERRQ(ierr);
          kappa= WeightAverageValue(kappa0,wt,usr->nph); 
          c[j][i][B_slot[ii]] = 1.0/usr->nd->Ra*kappa;
        }
      }

      { // u = velocity (edge) - StokesDarcy vs velocity
        PetscScalar   vs[4];
        vs[0] = _xPVlocal[j][i][pv_slot[iL]]; 
        vs[1] = _xPVlocal[j][i][pv_slot[iR]];
        vs[2] = _xPVlocal[j][i][pv_slot[iD]];
        vs[3] = _xPVlocal[j][i][pv_slot[iU]];

        for (ii = 0; ii < 4; ii++) {
          c[j][i][u_slot[ii] ] = vs[ii];
        }
      }
    }
  }

  // Restore arrays, local vectors
  ierr = DMStagVecRestoreArray(dmcoeff,coefflocal,&c);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  ierr = VecDestroy(&coefflocal); CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(dm,&xlocal); CHKERRQ(ierr);
  
  ierr = DMStagVecRestoreArrayRead(dmPV,xPVlocal,&_xPVlocal);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmPV, &xPVlocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArray(usr->dmMPhase,xMPhaselocal,&xwt);CHKERRQ(ierr);
  ierr = VecDestroy(&xMPhaselocal); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// FormCoefficient_phi
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormCoefficient_phi"
PetscErrorCode FormCoefficient_phi(FDPDE fd, DM dm, Vec x, DM dmcoeff, Vec coeff, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       i, j, ii, sx, sz, nx, nz, Nx, Nz;
  DM             dmPV = NULL;
  Vec            coefflocal;
  PetscScalar    ***c, ***_xPVlocal, ***xwt;
  Vec            xPV = NULL, xPVlocal,  xlocal, xMPhaselocal;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  // Get dm and solution vector for Stokes velocity
  dmPV = usr->dmPV;
  xPV  = usr->xPV;
  
  ierr = DMGetLocalVector(dmPV, &xPVlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dmPV, xPV, INSERT_VALUES, xPVlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(dmPV,xPVlocal,&_xPVlocal);CHKERRQ(ierr);
  
  // Get solution vector for temperature
  ierr = DMGetLocalVector(dm,&xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm,x,INSERT_VALUES,xlocal); CHKERRQ(ierr);

  // get material phase fractions
  ierr = DMCreateLocalVector(usr->dmMPhase, &xMPhaselocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(usr->dmMPhase,usr->xMPhase,INSERT_VALUES,xMPhaselocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (usr->dmMPhase,usr->xMPhase,INSERT_VALUES,xMPhaselocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(usr->dmMPhase, xMPhaselocal, &xwt); CHKERRQ(ierr);

  // Get coefficient local vector
  ierr = DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = DMStagGetGlobalSizes(dmcoeff,&Nx,&Nz,NULL);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dmcoeff, &coefflocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dmcoeff, coefflocal, &c); CHKERRQ(ierr);

   // get location slots
  PetscInt  pv_slot[4],iL,iR,iU,iD;
  iL = 0; iR  = 1; iD = 2; iU  = 3;
  ierr = DMStagGetLocationSlot(usr->dmPV,LEFT,   PV_FACE_VS,   &pv_slot[iL]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmPV,RIGHT,  PV_FACE_VS,   &pv_slot[iR]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmPV,DOWN,   PV_FACE_VS,   &pv_slot[iD]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmPV,UP,     PV_FACE_VS,   &pv_slot[iU]);CHKERRQ(ierr);
  
    PetscInt  e_slot[2],iA,iC;
  iA = 0; iC = 1;
  ierr = DMStagGetLocationSlot(dmcoeff,ELEMENT,TCOEFF_ELEMENT_A, &e_slot[iA]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,ELEMENT,TCOEFF_ELEMENT_C, &e_slot[iC]); CHKERRQ(ierr);

  PetscInt  B_slot[4],u_slot[4];
  ierr = DMStagGetLocationSlot(dmcoeff,LEFT, TCOEFF_FACE_B,&B_slot[0]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,RIGHT,TCOEFF_FACE_B,&B_slot[1]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DOWN, TCOEFF_FACE_B,&B_slot[2]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,UP,   TCOEFF_FACE_B,&B_slot[3]);CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(dmcoeff,LEFT, TCOEFF_FACE_u,&u_slot[0]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,RIGHT,TCOEFF_FACE_u,&u_slot[1]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DOWN, TCOEFF_FACE_u,&u_slot[2]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,UP,   TCOEFF_FACE_u,&u_slot[3]);CHKERRQ(ierr);

  PetscInt iwtc[MAX_MAT_PHASE]; 
  for (ii = 0; ii < usr->nph; ii++) { 
    ierr = DMStagGetLocationSlot(usr->dmMPhase, ELEMENT, ii, &iwtc[ii]); CHKERRQ(ierr); 
  }

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      PetscScalar rho0[MAX_MAT_PHASE], wt[MAX_MAT_PHASE], rhos;
      PetscInt iph;

      { // element
        // A = 1.0
        c[j][i][e_slot[iA]] = 1.0;
        
        // C = Gamma/rhos
        for (iph = 0; iph < usr->nph; iph++) { 
          rho0[iph] = Density(usr->mat_nd[iph].rho0,usr->mat[iph].rho_func); 
        }
        ierr = GetMatPhaseFraction(i,j,xwt,iwtc,usr->nph,wt); CHKERRQ(ierr);
        rhos = WeightAverageValue(rho0,wt,usr->nph); 
        c[j][i][e_slot[iC]] = usr->nd->Gamma/rhos;
      }

      { // B = 0 (edge)
        for (ii = 0; ii < 4; ii++) {
          c[j][i][B_slot[ii]] = 0.0;
        }
      }

      { // u = velocity (edge) - StokesDarcy vs velocity
        PetscScalar   vs[4];
        vs[0] = _xPVlocal[j][i][pv_slot[iL]]; 
        vs[1] = _xPVlocal[j][i][pv_slot[iR]];
        vs[2] = _xPVlocal[j][i][pv_slot[iD]];
        vs[3] = _xPVlocal[j][i][pv_slot[iU]];

        for (ii = 0; ii < 4; ii++) {
          c[j][i][u_slot[ii] ] = vs[ii];
        }
      }
    }
  }

  // Restore arrays, local vectors
  ierr = DMStagVecRestoreArray(dmcoeff,coefflocal,&c);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  ierr = VecDestroy(&coefflocal); CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(dm,&xlocal); CHKERRQ(ierr);
  
  ierr = DMStagVecRestoreArrayRead(dmPV,xPVlocal,&_xPVlocal);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmPV, &xPVlocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArray(usr->dmMPhase,xMPhaselocal,&xwt);CHKERRQ(ierr);
  ierr = VecDestroy(&xMPhaselocal); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}