#include "morfault.h"

// ---------------------------------------
// FormCoefficient_PV (Stokes-Darcy)
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
  PetscCall(DMStagGetGlobalSizes(dmcoeff, &Nx, &Nz,NULL));
  PetscCall(DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 
  PetscCall(DMStagGetProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmcoeff,LEFT,&iprev));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmcoeff,RIGHT,&inext)); 
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmcoeff,ELEMENT,&icenter));

  // Create coefficient local vector
  PetscCall(DMCreateLocalVector(dmcoeff, &coefflocal)); 
  PetscCall(DMStagVecGetArray(dmcoeff, coefflocal, &c)); 

  // Strain rates
  PetscCall(UpdateStrainRates_Array(dm,x,usr)); 
  PetscCall(DMGetLocalVector(usr->dmeps, &xepslocal)); 
  PetscCall(DMGlobalToLocal (usr->dmeps, usr->xeps, INSERT_VALUES, xepslocal)); 
  PetscCall(DMStagVecGetArrayRead(usr->dmeps,xepslocal,&_eps));

  PetscCall(DMGetLocalVector(usr->dmeps, &xtauoldlocal)); 
  PetscCall(DMGlobalToLocal (usr->dmeps, usr->xtau_old, INSERT_VALUES, xtauoldlocal)); 
  PetscCall(DMStagVecGetArrayRead(usr->dmeps,xtauoldlocal,&_tauold));

  PetscCall(DMGetLocalVector(usr->dmeps, &xtaulocal)); 
  PetscCall(DMGlobalToLocal (usr->dmeps, usr->xtau, INSERT_VALUES, xtaulocal)); 
  PetscCall(DMStagVecGetArray(usr->dmeps,xtaulocal,&_tau));

  // Get dm and solution vector
  PetscCall(DMGetLocalVector(dm, &xlocal)); 
  PetscCall(DMGlobalToLocal (dm, x, INSERT_VALUES, xlocal)); 
  PetscCall(DMStagVecGetArrayRead(dm,xlocal,&xx));
  
  // Get solution vector for temperature
  PetscCall(DMGetLocalVector(usr->dmT,&xTlocal)); 
  PetscCall(DMGlobalToLocal (usr->dmT,usr->xT,INSERT_VALUES,xTlocal)); 
  PetscCall(DMStagVecGetArrayRead(usr->dmT,xTlocal,&_Tc));

  // Get porosity
  PetscCall(DMGetLocalVector(usr->dmphi,&xphilocal)); 
  PetscCall(DMGlobalToLocal (usr->dmphi,usr->xphi,INSERT_VALUES,xphilocal)); 
  PetscCall(DMStagVecGetArrayRead(usr->dmphi,xphilocal,&_phic));

  // Get dm and vector Plith
  PetscCall(DMGetLocalVector(usr->dmPlith, &xPlithlocal)); 
  PetscCall(DMGlobalToLocal (usr->dmPlith, usr->xPlith, INSERT_VALUES, xPlithlocal)); 
  PetscCall(DMStagVecGetArrayRead(usr->dmPlith,xPlithlocal,&_Plith));

  PetscCall(DMGetLocalVector(usr->dmPlith, &xDPoldlocal)); 
  PetscCall(DMGlobalToLocal (usr->dmPlith, usr->xDP_old, INSERT_VALUES, xDPoldlocal)); 
  PetscCall(DMStagVecGetArrayRead(usr->dmPlith,xDPoldlocal,&_DPold));

  PetscCall(DMGetLocalVector(usr->dmPlith, &xDPlocal)); 
  PetscCall(DMGlobalToLocal (usr->dmPlith, usr->xDP, INSERT_VALUES, xDPlocal)); 
  PetscCall(DMStagVecGetArrayRead(usr->dmPlith,xDPlocal,&_DP));

  PetscCall(DMGetLocalVector(usr->dmPlith, &xplastlocal)); 
  PetscCall(DMGlobalToLocal (usr->dmPlith, usr->xplast, INSERT_VALUES, xplastlocal)); 
  PetscCall(DMStagVecGetArrayRead(usr->dmPlith,xplastlocal,&_plast));

  PetscCall(DMGetLocalVector(usr->dmPlith, &xstrainlocal)); 
  PetscCall(DMGlobalToLocal (usr->dmPlith, usr->xstrain, INSERT_VALUES, xstrainlocal)); 
  PetscCall(DMStagVecGetArrayRead(usr->dmPlith,xstrainlocal,&_strain));

  // get material phase fractions
  PetscCall(DMCreateLocalVector(usr->dmMPhase, &xMPhaselocal)); 
  PetscCall(DMGlobalToLocalBegin(usr->dmMPhase,usr->xMPhase,INSERT_VALUES,xMPhaselocal)); 
  PetscCall(DMGlobalToLocalEnd  (usr->dmMPhase,usr->xMPhase,INSERT_VALUES,xMPhaselocal)); 
  PetscCall(DMStagVecGetArray(usr->dmMPhase, xMPhaselocal, &xwt)); 

  // Get material properties - for output
  PetscCall(DMGetLocalVector(usr->dmmatProp, &xmatProplocal)); 
  PetscCall(DMGlobalToLocal (usr->dmmatProp, usr->xmatProp, INSERT_VALUES, xmatProplocal)); 
  PetscCall(DMStagVecGetArrayRead(usr->dmmatProp,xmatProplocal,&_matProp));

  // get location slots
  PetscInt  iE,iP;
  PetscCall(DMStagGetLocationSlot(usr->dmT,ELEMENT,T_ELEMENT,&iE));
  PetscCall(DMStagGetLocationSlot(usr->dmPlith,ELEMENT,0,&iP));

  PetscInt iPV;
  PetscCall(DMStagGetLocationSlot(dm,ELEMENT,PV_ELEMENT_P,&iPV));

  PetscInt  e_slot[3],av_slot[4],b_slot[4],d2_slot[4],d3_slot[4],iL,iR,iU,iD,iC,iA,iD1;
  iL = 0; iR = 1; iD = 2; iU  = 3;
  iC = 0; iA = 1; iD1= 2;
  PetscCall(DMStagGetLocationSlot(dmcoeff,ELEMENT,   PVCOEFF_ELEMENT_C,   &e_slot[iC]));
  PetscCall(DMStagGetLocationSlot(dmcoeff,ELEMENT,   PVCOEFF_ELEMENT_A,   &e_slot[iA]));
  PetscCall(DMStagGetLocationSlot(dmcoeff,ELEMENT,   PVCOEFF_ELEMENT_D1,  &e_slot[iD1]));

  PetscCall(DMStagGetLocationSlot(dmcoeff,DOWN_LEFT, PVCOEFF_VERTEX_A,&av_slot[0]));
  PetscCall(DMStagGetLocationSlot(dmcoeff,DOWN_RIGHT,PVCOEFF_VERTEX_A,&av_slot[1]));
  PetscCall(DMStagGetLocationSlot(dmcoeff,UP_LEFT,   PVCOEFF_VERTEX_A,&av_slot[2]));
  PetscCall(DMStagGetLocationSlot(dmcoeff,UP_RIGHT,  PVCOEFF_VERTEX_A,&av_slot[3]));

  PetscCall(DMStagGetLocationSlot(dmcoeff,LEFT,   PVCOEFF_FACE_B,   &b_slot[iL]));
  PetscCall(DMStagGetLocationSlot(dmcoeff,RIGHT,  PVCOEFF_FACE_B,   &b_slot[iR]));
  PetscCall(DMStagGetLocationSlot(dmcoeff,DOWN,   PVCOEFF_FACE_B,   &b_slot[iD]));
  PetscCall(DMStagGetLocationSlot(dmcoeff,UP,     PVCOEFF_FACE_B,   &b_slot[iU]));

  PetscCall(DMStagGetLocationSlot(dmcoeff,LEFT,   PVCOEFF_FACE_D2,   &d2_slot[iL]));
  PetscCall(DMStagGetLocationSlot(dmcoeff,RIGHT,  PVCOEFF_FACE_D2,   &d2_slot[iR]));
  PetscCall(DMStagGetLocationSlot(dmcoeff,DOWN,   PVCOEFF_FACE_D2,   &d2_slot[iD]));
  PetscCall(DMStagGetLocationSlot(dmcoeff,UP,     PVCOEFF_FACE_D2,   &d2_slot[iU]));

  PetscCall(DMStagGetLocationSlot(dmcoeff,LEFT,   PVCOEFF_FACE_D3,   &d3_slot[iL]));
  PetscCall(DMStagGetLocationSlot(dmcoeff,RIGHT,  PVCOEFF_FACE_D3,   &d3_slot[iR]));
  PetscCall(DMStagGetLocationSlot(dmcoeff,DOWN,   PVCOEFF_FACE_D3,   &d3_slot[iD]));
  PetscCall(DMStagGetLocationSlot(dmcoeff,UP,     PVCOEFF_FACE_D3,   &d3_slot[iU]));

  PetscInt iwtc[MAX_MAT_PHASE],iwtl[MAX_MAT_PHASE],iwtr[MAX_MAT_PHASE],iwtd[MAX_MAT_PHASE],iwtu[MAX_MAT_PHASE];
  PetscInt iwtld[MAX_MAT_PHASE],iwtrd[MAX_MAT_PHASE],iwtlu[MAX_MAT_PHASE],iwtru[MAX_MAT_PHASE];
  for (ii = 0; ii < usr->nph; ii++) { 
    PetscCall(DMStagGetLocationSlot(usr->dmMPhase, ELEMENT,    ii, &iwtc[ii]));  
    PetscCall(DMStagGetLocationSlot(usr->dmMPhase, LEFT,       ii, &iwtl[ii])); 
    PetscCall(DMStagGetLocationSlot(usr->dmMPhase, RIGHT,      ii, &iwtr[ii])); 
    PetscCall(DMStagGetLocationSlot(usr->dmMPhase, DOWN,       ii, &iwtd[ii]));  
    PetscCall(DMStagGetLocationSlot(usr->dmMPhase, UP,         ii, &iwtu[ii])); 
    PetscCall(DMStagGetLocationSlot(usr->dmMPhase, DOWN_LEFT,  ii, &iwtld[ii])); 
    PetscCall(DMStagGetLocationSlot(usr->dmMPhase, DOWN_RIGHT, ii, &iwtrd[ii])); 
    PetscCall(DMStagGetLocationSlot(usr->dmMPhase, UP_LEFT,    ii, &iwtlu[ii])); 
    PetscCall(DMStagGetLocationSlot(usr->dmMPhase, UP_RIGHT,   ii, &iwtru[ii])); 
  }

  PetscInt ixx, izz, ixz, iII, ixxn[4], izzn[4], ixzn[4], iIIn[4];
  PetscCall(DMStagGetLocationSlot(usr->dmeps, ELEMENT, 0, &ixx)); 
  PetscCall(DMStagGetLocationSlot(usr->dmeps, ELEMENT, 1, &izz)); 
  PetscCall(DMStagGetLocationSlot(usr->dmeps, ELEMENT, 2, &ixz)); 
  PetscCall(DMStagGetLocationSlot(usr->dmeps, ELEMENT, 3, &iII)); 

  PetscCall(DMStagGetLocationSlot(usr->dmeps, DOWN_LEFT, 0, &ixxn[0])); 
  PetscCall(DMStagGetLocationSlot(usr->dmeps, DOWN_LEFT, 1, &izzn[0])); 
  PetscCall(DMStagGetLocationSlot(usr->dmeps, DOWN_LEFT, 2, &ixzn[0])); 
  PetscCall(DMStagGetLocationSlot(usr->dmeps, DOWN_LEFT, 3, &iIIn[0])); 

  PetscCall(DMStagGetLocationSlot(usr->dmeps, DOWN_RIGHT, 0, &ixxn[1])); 
  PetscCall(DMStagGetLocationSlot(usr->dmeps, DOWN_RIGHT, 1, &izzn[1])); 
  PetscCall(DMStagGetLocationSlot(usr->dmeps, DOWN_RIGHT, 2, &ixzn[1])); 
  PetscCall(DMStagGetLocationSlot(usr->dmeps, DOWN_RIGHT, 3, &iIIn[1])); 

  PetscCall(DMStagGetLocationSlot(usr->dmeps, UP_LEFT, 0, &ixxn[2])); 
  PetscCall(DMStagGetLocationSlot(usr->dmeps, UP_LEFT, 1, &izzn[2])); 
  PetscCall(DMStagGetLocationSlot(usr->dmeps, UP_LEFT, 2, &ixzn[2])); 
  PetscCall(DMStagGetLocationSlot(usr->dmeps, UP_LEFT, 3, &iIIn[2])); 

  PetscCall(DMStagGetLocationSlot(usr->dmeps, UP_RIGHT, 0, &ixxn[3])); 
  PetscCall(DMStagGetLocationSlot(usr->dmeps, UP_RIGHT, 1, &izzn[3])); 
  PetscCall(DMStagGetLocationSlot(usr->dmeps, UP_RIGHT, 2, &ixzn[3])); 
  PetscCall(DMStagGetLocationSlot(usr->dmeps, UP_RIGHT, 3, &iIIn[3])); 

  PetscInt imat[MATPROP_NPROP];
  for (ii = 0; ii < MATPROP_NPROP; ii++) { PetscCall(DMStagGetLocationSlot(usr->dmmatProp, ELEMENT, ii, &imat[ii]));  }
  
  // Loop over local domain
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
      PetscCall(Get9PointCenterValues(i,j,iE,Nx,Nz,_phic,phic));
      PetscCall(Get9PointCenterValues(i,j,iP,Nx,Nz,_Plith,Plc));
      PetscCall(Get9PointCenterValues(i,j,iPV,Nx,Nz,xx,pc));
      PetscCall(Get9PointCenterValues(i,j,iE,Nx,Nz,_Tc,Tc));
      PetscCall(Get9PointCenterValues(i,j,iP,Nx,Nz,_DPold,DPoldc));
      PetscCall(Get9PointCenterValues(i,j,iP,Nx,Nz,_strain,strain));

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
      PetscCall(GetTensorPointValues(i,j,ix,_eps,e));
      PetscCall(GetTensorPointValues(i,j,ix,_tauold,t));
      PetscCall(RheologyPointwise(i,j,xwt,iwtc,Tc[0],phic[0],P,e,t,ix,res,usr));
      PetscCall(DecompactRheologyVars_DPL(0,res,eta_eff,eta_v,eta_e,eta_p,zeta_eff,zeta_v,zeta_e,DPdl,chis,chip,txx,tzz,txz,tII,DP,dotlam,Z,G,C,sigmat,theta));

      P[0] = pc[1]; P[1] = Plc[1]; P[2] = DPoldc[1]; P[3] = strain[1];
      PetscCall(GetTensorPointValues(im,j,ix,_eps,e));
      PetscCall(GetTensorPointValues(im,j,ix,_tauold,t));
      PetscCall(RheologyPointwise(im,j,xwt,iwtc,Tc[1],phic[1],P,e,t,ix,res,usr));
      PetscCall(DecompactRheologyVars_DPL(1,res,eta_eff,eta_v,eta_e,eta_p,zeta_eff,zeta_v,zeta_e,DPdl,chis,chip,txx,tzz,txz,tII,DP,dotlam,Z,G,C,sigmat,theta));

      P[0] = pc[2]; P[1] = Plc[2]; P[2] = DPoldc[2]; P[3] = strain[2];
      PetscCall(GetTensorPointValues(ip,j,ix,_eps,e));
      PetscCall(GetTensorPointValues(ip,j,ix,_tauold,t));
      PetscCall(RheologyPointwise(ip,j,xwt,iwtc,Tc[2],phic[2],P,e,t,ix,res,usr));
      PetscCall(DecompactRheologyVars_DPL(2,res,eta_eff,eta_v,eta_e,eta_p,zeta_eff,zeta_v,zeta_e,DPdl,chis,chip,txx,tzz,txz,tII,DP,dotlam,Z,G,C,sigmat,theta));

      P[0] = pc[3]; P[1] = Plc[3]; P[2] = DPoldc[3]; P[3] = strain[3];
      PetscCall(GetTensorPointValues(i,jm,ix,_eps,e));
      PetscCall(GetTensorPointValues(i,jm,ix,_tauold,t));
      PetscCall(RheologyPointwise(i,jm,xwt,iwtc,Tc[3],phic[3],P,e,t,ix,res,usr));
      PetscCall(DecompactRheologyVars_DPL(3,res,eta_eff,eta_v,eta_e,eta_p,zeta_eff,zeta_v,zeta_e,DPdl,chis,chip,txx,tzz,txz,tII,DP,dotlam,Z,G,C,sigmat,theta));

      P[0] = pc[4]; P[1] = Plc[4]; P[2] = DPoldc[4]; P[3] = strain[4];
      PetscCall(GetTensorPointValues(i,jp,ix,_eps,e));
      PetscCall(GetTensorPointValues(i,jp,ix,_tauold,t));
      PetscCall(RheologyPointwise(i,jp,xwt,iwtc,Tc[4],phic[4],P,e,t,ix,res,usr));
      PetscCall(DecompactRheologyVars_DPL(4,res,eta_eff,eta_v,eta_e,eta_p,zeta_eff,zeta_v,zeta_e,DPdl,chis,chip,txx,tzz,txz,tII,DP,dotlam,Z,G,C,sigmat,theta));

      // corner points
      PetscScalar Tcorner[4], phicorner[4], pcorner[4], Plcorner[4], DPoldcorner[4], straincorner[4];
      PetscCall(GetCornerAvgFromCenter(Tc,Tcorner));
      PetscCall(GetCornerAvgFromCenter(phic,phicorner));
      PetscCall(GetCornerAvgFromCenter(pc,pcorner));
      PetscCall(GetCornerAvgFromCenter(Plc,Plcorner));
      PetscCall(GetCornerAvgFromCenter(DPoldc,DPoldcorner));
      PetscCall(GetCornerAvgFromCenter(strain,straincorner));
      
      ii = 0; ix[0] = ixxn[ii]; ix[1] = izzn[ii]; ix[2] = ixzn[ii]; ix[3] = iIIn[ii]; P[0] = pcorner[ii]; P[1] = Plcorner[ii]; P[2] = DPoldcorner[ii]; P[3] = straincorner[ii];
      PetscCall(GetTensorPointValues(i,j,ix,_eps,e));
      PetscCall(GetTensorPointValues(i,j,ix,_tauold,t));
      PetscCall(RheologyPointwise(i,j,xwt,iwtld,Tcorner[0],phicorner[0],P,e,t,ix,res,usr));
      PetscCall(DecompactRheologyVars_DPL(5,res,eta_eff,eta_v,eta_e,eta_p,zeta_eff,zeta_v,zeta_e,DPdl,chis,chip,txx,tzz,txz,tII,DP,dotlam,Z,G,C,sigmat,theta));

      ii = 1; ix[0] = ixxn[ii]; ix[1] = izzn[ii]; ix[2] = ixzn[ii]; ix[3] = iIIn[ii]; P[0] = pcorner[ii]; P[1] = Plcorner[ii]; P[2] = DPoldcorner[ii]; P[3] = straincorner[ii];
      PetscCall(GetTensorPointValues(i,j,ix,_eps,e));
      PetscCall(GetTensorPointValues(i,j,ix,_tauold,t));
      PetscCall(RheologyPointwise(i,j,xwt,iwtrd,Tcorner[1],phicorner[1],P,e,t,ix,res,usr));
      PetscCall(DecompactRheologyVars_DPL(6,res,eta_eff,eta_v,eta_e,eta_p,zeta_eff,zeta_v,zeta_e,DPdl,chis,chip,txx,tzz,txz,tII,DP,dotlam,Z,G,C,sigmat,theta));

      ii = 2; ix[0] = ixxn[ii]; ix[1] = izzn[ii]; ix[2] = ixzn[ii]; ix[3] = iIIn[ii]; P[0] = pcorner[ii]; P[1] = Plcorner[ii]; P[2] = DPoldcorner[ii]; P[3] = straincorner[ii];
      PetscCall(GetTensorPointValues(i,j,ix,_eps,e));
      PetscCall(GetTensorPointValues(i,j,ix,_tauold,t));
      PetscCall(RheologyPointwise(i,j,xwt,iwtlu,Tcorner[2],phicorner[2],P,e,t,ix,res,usr));
      PetscCall(DecompactRheologyVars_DPL(7,res,eta_eff,eta_v,eta_e,eta_p,zeta_eff,zeta_v,zeta_e,DPdl,chis,chip,txx,tzz,txz,tII,DP,dotlam,Z,G,C,sigmat,theta));

      ii = 3; ix[0] = ixxn[ii]; ix[1] = izzn[ii]; ix[2] = ixzn[ii]; ix[3] = iIIn[ii]; P[0] = pcorner[ii]; P[1] = Plcorner[ii]; P[2] = DPoldcorner[ii]; P[3] = straincorner[ii];
      PetscCall(GetTensorPointValues(i,j,ix,_eps,e));
      PetscCall(GetTensorPointValues(i,j,ix,_tauold,t));
      PetscCall(RheologyPointwise(i,j,xwt,iwtru,Tcorner[3],phicorner[3],P,e,t,ix,res,usr));
      PetscCall(DecompactRheologyVars_DPL(8,res,eta_eff,eta_v,eta_e,eta_p,zeta_eff,zeta_v,zeta_e,DPdl,chis,chip,txx,tzz,txz,tII,DP,dotlam,Z,G,C,sigmat,theta));

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
          PetscCall(GetMatPhaseFraction(i,j,xwt,idx,usr->nph,wt)); 
          rhos[ii] = WeightAverageValue(rho0,wt,usr->nph); 
          rhof[ii] = usr->nd->rhof;
          rho[ii]  = Mixture(rhos[ii],rhof[ii],phi[ii]);

          // add buoyancy terms here
          B[ii] = -divchitau[ii]+gradchidp[ii] +gradDPdl[ii] + gradPlith[ii] + rho[ii]*k_hat[ii];
          // B[ii] = -divchitau[ii]+gradchidp[ii] +gradDPdl[ii] + gradPlith[ii]*i_hat[ii] - phi[ii]*k_hat[ii];
          D2[ii] = -K[ii]*usr->nd->R*usr->nd->R;
          D3[ii] = -K[ii]*usr->nd->R*usr->nd->R*(gradPlith[ii]+ rhof[ii]*k_hat[ii]);
          // D3[ii] = -K[ii]*usr->nd->R*usr->nd->R*(gradPlith[ii]*i_hat[ii] - k_hat[ii]);

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
      PetscCall(GetMatPhaseFraction(i,j,xwt,iwtc,usr->nph,wt)); 
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
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL));
  PetscCall(DMStagVecRestoreArray(dmcoeff,coefflocal,&c));
  PetscCall(DMLocalToGlobalBegin(dmcoeff,coefflocal,INSERT_VALUES,coeff)); 
  PetscCall(DMLocalToGlobalEnd  (dmcoeff,coefflocal,INSERT_VALUES,coeff)); 
  PetscCall(VecDestroy(&coefflocal)); 

  PetscCall(DMStagVecRestoreArrayRead(dm,xlocal,&xx));
  PetscCall(DMRestoreLocalVector(dm,&xlocal)); 

  PetscCall(DMStagVecRestoreArrayRead(usr->dmeps,xepslocal,&_eps));
  PetscCall(DMRestoreLocalVector(usr->dmeps,&xepslocal)); 
  PetscCall(DMStagVecRestoreArrayRead(usr->dmeps,xtauoldlocal,&_tauold));
  PetscCall(DMRestoreLocalVector(usr->dmeps,&xtauoldlocal)); 

  PetscCall(DMStagVecRestoreArray(usr->dmeps,xtaulocal,&_tau));
  PetscCall(DMLocalToGlobalBegin(usr->dmeps,xtaulocal,INSERT_VALUES,usr->xtau)); 
  PetscCall(DMLocalToGlobalEnd  (usr->dmeps,xtaulocal,INSERT_VALUES,usr->xtau)); 
  PetscCall(DMRestoreLocalVector(usr->dmeps,&xtaulocal)); 

  PetscCall(DMStagVecRestoreArrayRead(usr->dmT,xTlocal,&_Tc));
  PetscCall(DMRestoreLocalVector(usr->dmT,&xTlocal)); 

  PetscCall(DMStagVecRestoreArrayRead(usr->dmphi,xphilocal,&_phic));
  PetscCall(DMRestoreLocalVector(usr->dmphi,&xphilocal)); 

  PetscCall(DMStagVecRestoreArrayRead(usr->dmPlith,xPlithlocal,&_Plith));
  PetscCall(DMRestoreLocalVector(usr->dmPlith,&xPlithlocal)); 
  PetscCall(DMStagVecRestoreArrayRead(usr->dmPlith,xDPoldlocal,&_DPold));
  PetscCall(DMRestoreLocalVector(usr->dmPlith,&xDPoldlocal)); 

  PetscCall(DMStagVecRestoreArray(usr->dmPlith,xDPlocal,&_DP));
  PetscCall(DMLocalToGlobalBegin(usr->dmPlith,xDPlocal,INSERT_VALUES,usr->xDP)); 
  PetscCall(DMLocalToGlobalEnd  (usr->dmPlith,xDPlocal,INSERT_VALUES,usr->xDP)); 
  PetscCall(DMRestoreLocalVector(usr->dmPlith,&xDPlocal)); 

  PetscCall(DMStagVecRestoreArray(usr->dmPlith,xplastlocal,&_plast));
  PetscCall(DMLocalToGlobalBegin(usr->dmPlith,xplastlocal,INSERT_VALUES,usr->xplast)); 
  PetscCall(DMLocalToGlobalEnd  (usr->dmPlith,xplastlocal,INSERT_VALUES,usr->xplast)); 
  PetscCall(DMRestoreLocalVector(usr->dmPlith,&xplastlocal)); 

  PetscCall(DMStagVecRestoreArrayRead(usr->dmPlith,xstrainlocal,&_strain));
  PetscCall(DMRestoreLocalVector(usr->dmPlith,&xstrainlocal)); 

  PetscCall(DMStagVecRestoreArray(usr->dmmatProp,xmatProplocal,&_matProp));
  PetscCall(DMLocalToGlobalBegin(usr->dmmatProp,xmatProplocal,INSERT_VALUES,usr->xmatProp)); 
  PetscCall(DMLocalToGlobalEnd  (usr->dmmatProp,xmatProplocal,INSERT_VALUES,usr->xmatProp)); 
  PetscCall(DMRestoreLocalVector(usr->dmmatProp,&xmatProplocal)); 

  PetscCall(DMStagVecRestoreArray(usr->dmMPhase,xMPhaselocal,&xwt));
  PetscCall(VecDestroy(&xMPhaselocal)); 
 
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// FormCoefficient_PV (Stokes)
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
  PetscCall(DMStagGetGlobalSizes(dmcoeff, &Nx, &Nz,NULL));
  PetscCall(DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 
  PetscCall(DMStagGetProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmcoeff,LEFT,&iprev));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmcoeff,RIGHT,&inext)); 
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmcoeff,ELEMENT,&icenter));

  // Create coefficient local vector
  PetscCall(DMCreateLocalVector(dmcoeff, &coefflocal)); 
  PetscCall(DMStagVecGetArray(dmcoeff, coefflocal, &c)); 

  // Strain rates
  PetscCall(UpdateStrainRates_Array(dm,x,usr)); 
  PetscCall(DMGetLocalVector(usr->dmeps, &xepslocal)); 
  PetscCall(DMGlobalToLocal (usr->dmeps, usr->xeps, INSERT_VALUES, xepslocal)); 
  PetscCall(DMStagVecGetArrayRead(usr->dmeps,xepslocal,&_eps));

  PetscCall(DMGetLocalVector(usr->dmeps, &xtauoldlocal)); 
  PetscCall(DMGlobalToLocal (usr->dmeps, usr->xtau_old, INSERT_VALUES, xtauoldlocal)); 
  PetscCall(DMStagVecGetArrayRead(usr->dmeps,xtauoldlocal,&_tauold));

  PetscCall(DMGetLocalVector(usr->dmeps, &xtaulocal)); 
  PetscCall(DMGlobalToLocal (usr->dmeps, usr->xtau, INSERT_VALUES, xtaulocal)); 
  PetscCall(DMStagVecGetArray(usr->dmeps,xtaulocal,&_tau));

  // Get dm and solution vector
  PetscCall(DMGetLocalVector(dm, &xlocal)); 
  PetscCall(DMGlobalToLocal (dm, x, INSERT_VALUES, xlocal)); 
  PetscCall(DMStagVecGetArrayRead(dm,xlocal,&xx));
  
  // Get solution vector for temperature
  PetscCall(DMGetLocalVector(usr->dmT,&xTlocal)); 
  PetscCall(DMGlobalToLocal (usr->dmT,usr->xT,INSERT_VALUES,xTlocal)); 
  PetscCall(DMStagVecGetArrayRead(usr->dmT,xTlocal,&_Tc));

  // Get porosity
  PetscCall(DMGetLocalVector(usr->dmphi,&xphilocal)); 
  PetscCall(DMGlobalToLocal (usr->dmphi,usr->xphi,INSERT_VALUES,xphilocal)); 
  PetscCall(DMStagVecGetArrayRead(usr->dmphi,xphilocal,&_phic));

  // Get dm and vector Plith
  PetscCall(DMGetLocalVector(usr->dmPlith, &xPlithlocal)); 
  PetscCall(DMGlobalToLocal (usr->dmPlith, usr->xPlith, INSERT_VALUES, xPlithlocal)); 
  PetscCall(DMStagVecGetArrayRead(usr->dmPlith,xPlithlocal,&_Plith));

  PetscCall(DMGetLocalVector(usr->dmPlith, &xDPoldlocal)); 
  PetscCall(DMGlobalToLocal (usr->dmPlith, usr->xDP_old, INSERT_VALUES, xDPoldlocal)); 
  PetscCall(DMStagVecGetArrayRead(usr->dmPlith,xDPoldlocal,&_DPold));

  PetscCall(DMGetLocalVector(usr->dmPlith, &xDPlocal)); 
  PetscCall(DMGlobalToLocal (usr->dmPlith, usr->xDP, INSERT_VALUES, xDPlocal)); 
  PetscCall(DMStagVecGetArrayRead(usr->dmPlith,xDPlocal,&_DP));

  PetscCall(DMGetLocalVector(usr->dmPlith, &xplastlocal)); 
  PetscCall(DMGlobalToLocal (usr->dmPlith, usr->xplast, INSERT_VALUES, xplastlocal)); 
  PetscCall(DMStagVecGetArrayRead(usr->dmPlith,xplastlocal,&_plast));

  PetscCall(DMGetLocalVector(usr->dmPlith, &xstrainlocal)); 
  PetscCall(DMGlobalToLocal (usr->dmPlith, usr->xstrain, INSERT_VALUES, xstrainlocal)); 
  PetscCall(DMStagVecGetArrayRead(usr->dmPlith,xstrainlocal,&_strain));

  // get material phase fractions
  PetscCall(DMCreateLocalVector(usr->dmMPhase, &xMPhaselocal)); 
  PetscCall(DMGlobalToLocalBegin(usr->dmMPhase,usr->xMPhase,INSERT_VALUES,xMPhaselocal)); 
  PetscCall(DMGlobalToLocalEnd  (usr->dmMPhase,usr->xMPhase,INSERT_VALUES,xMPhaselocal)); 
  PetscCall(DMStagVecGetArray(usr->dmMPhase, xMPhaselocal, &xwt)); 

  // Get material properties for output
  PetscCall(DMGetLocalVector(usr->dmmatProp, &xmatProplocal)); 
  PetscCall(DMGlobalToLocal (usr->dmmatProp, usr->xmatProp, INSERT_VALUES, xmatProplocal)); 
  PetscCall(DMStagVecGetArrayRead(usr->dmmatProp,xmatProplocal,&_matProp));

  // get location slots
  PetscInt  iE,iP;
  PetscCall(DMStagGetLocationSlot(usr->dmT,ELEMENT,T_ELEMENT,&iE));
  PetscCall(DMStagGetLocationSlot(usr->dmPlith,ELEMENT,0,&iP));

  PetscInt iPV;
  PetscCall(DMStagGetLocationSlot(dm,ELEMENT,PV_ELEMENT_P,&iPV));

  PetscInt  e_slot[3],av_slot[4],b_slot[4],iL,iR,iU,iD,iC,iA;
  iL = 0; iR = 1; iD = 2; iU  = 3;
  iC = 0; iA = 1; 
  PetscCall(DMStagGetLocationSlot(dmcoeff,ELEMENT,   PVCOEFF_ELEMENT_C,   &e_slot[iC]));
  PetscCall(DMStagGetLocationSlot(dmcoeff,ELEMENT,   PVCOEFF_ELEMENT_A,   &e_slot[iA]));

  PetscCall(DMStagGetLocationSlot(dmcoeff,DOWN_LEFT, PVCOEFF_VERTEX_A,&av_slot[0]));
  PetscCall(DMStagGetLocationSlot(dmcoeff,DOWN_RIGHT,PVCOEFF_VERTEX_A,&av_slot[1]));
  PetscCall(DMStagGetLocationSlot(dmcoeff,UP_LEFT,   PVCOEFF_VERTEX_A,&av_slot[2]));
  PetscCall(DMStagGetLocationSlot(dmcoeff,UP_RIGHT,  PVCOEFF_VERTEX_A,&av_slot[3]));

  PetscCall(DMStagGetLocationSlot(dmcoeff,LEFT,   PVCOEFF_FACE_B,   &b_slot[iL]));
  PetscCall(DMStagGetLocationSlot(dmcoeff,RIGHT,  PVCOEFF_FACE_B,   &b_slot[iR]));
  PetscCall(DMStagGetLocationSlot(dmcoeff,DOWN,   PVCOEFF_FACE_B,   &b_slot[iD]));
  PetscCall(DMStagGetLocationSlot(dmcoeff,UP,     PVCOEFF_FACE_B,   &b_slot[iU]));

  PetscInt iwtc[MAX_MAT_PHASE],iwtl[MAX_MAT_PHASE],iwtr[MAX_MAT_PHASE],iwtd[MAX_MAT_PHASE],iwtu[MAX_MAT_PHASE];
  PetscInt iwtld[MAX_MAT_PHASE],iwtrd[MAX_MAT_PHASE],iwtlu[MAX_MAT_PHASE],iwtru[MAX_MAT_PHASE];
  for (ii = 0; ii < usr->nph; ii++) { 
    PetscCall(DMStagGetLocationSlot(usr->dmMPhase, ELEMENT,    ii, &iwtc[ii]));  
    PetscCall(DMStagGetLocationSlot(usr->dmMPhase, LEFT,       ii, &iwtl[ii])); 
    PetscCall(DMStagGetLocationSlot(usr->dmMPhase, RIGHT,      ii, &iwtr[ii])); 
    PetscCall(DMStagGetLocationSlot(usr->dmMPhase, DOWN,       ii, &iwtd[ii]));  
    PetscCall(DMStagGetLocationSlot(usr->dmMPhase, UP,         ii, &iwtu[ii])); 
    PetscCall(DMStagGetLocationSlot(usr->dmMPhase, DOWN_LEFT,  ii, &iwtld[ii])); 
    PetscCall(DMStagGetLocationSlot(usr->dmMPhase, DOWN_RIGHT, ii, &iwtrd[ii])); 
    PetscCall(DMStagGetLocationSlot(usr->dmMPhase, UP_LEFT,    ii, &iwtlu[ii])); 
    PetscCall(DMStagGetLocationSlot(usr->dmMPhase, UP_RIGHT,   ii, &iwtru[ii])); 
  }

  PetscInt ixx, izz, ixz, iII, ixxn[4], izzn[4], ixzn[4], iIIn[4];
  PetscCall(DMStagGetLocationSlot(usr->dmeps, ELEMENT, 0, &ixx)); 
  PetscCall(DMStagGetLocationSlot(usr->dmeps, ELEMENT, 1, &izz)); 
  PetscCall(DMStagGetLocationSlot(usr->dmeps, ELEMENT, 2, &ixz)); 
  PetscCall(DMStagGetLocationSlot(usr->dmeps, ELEMENT, 3, &iII)); 

  PetscCall(DMStagGetLocationSlot(usr->dmeps, DOWN_LEFT, 0, &ixxn[0])); 
  PetscCall(DMStagGetLocationSlot(usr->dmeps, DOWN_LEFT, 1, &izzn[0])); 
  PetscCall(DMStagGetLocationSlot(usr->dmeps, DOWN_LEFT, 2, &ixzn[0])); 
  PetscCall(DMStagGetLocationSlot(usr->dmeps, DOWN_LEFT, 3, &iIIn[0])); 

  PetscCall(DMStagGetLocationSlot(usr->dmeps, DOWN_RIGHT, 0, &ixxn[1])); 
  PetscCall(DMStagGetLocationSlot(usr->dmeps, DOWN_RIGHT, 1, &izzn[1])); 
  PetscCall(DMStagGetLocationSlot(usr->dmeps, DOWN_RIGHT, 2, &ixzn[1])); 
  PetscCall(DMStagGetLocationSlot(usr->dmeps, DOWN_RIGHT, 3, &iIIn[1])); 

  PetscCall(DMStagGetLocationSlot(usr->dmeps, UP_LEFT, 0, &ixxn[2])); 
  PetscCall(DMStagGetLocationSlot(usr->dmeps, UP_LEFT, 1, &izzn[2])); 
  PetscCall(DMStagGetLocationSlot(usr->dmeps, UP_LEFT, 2, &ixzn[2])); 
  PetscCall(DMStagGetLocationSlot(usr->dmeps, UP_LEFT, 3, &iIIn[2])); 

  PetscCall(DMStagGetLocationSlot(usr->dmeps, UP_RIGHT, 0, &ixxn[3])); 
  PetscCall(DMStagGetLocationSlot(usr->dmeps, UP_RIGHT, 1, &izzn[3])); 
  PetscCall(DMStagGetLocationSlot(usr->dmeps, UP_RIGHT, 2, &ixzn[3])); 
  PetscCall(DMStagGetLocationSlot(usr->dmeps, UP_RIGHT, 3, &iIIn[3])); 

  PetscInt imat[MATPROP_NPROP];
  for (ii = 0; ii < MATPROP_NPROP; ii++) { PetscCall(DMStagGetLocationSlot(usr->dmmatProp, ELEMENT, ii, &imat[ii]));  }
  
  // Loop over local domain
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
      PetscCall(Get9PointCenterValues(i,j,iE,Nx,Nz,_phic,phic));
      PetscCall(Get9PointCenterValues(i,j,iP,Nx,Nz,_Plith,Plc));
      PetscCall(Get9PointCenterValues(i,j,iPV,Nx,Nz,xx,pc));
      PetscCall(Get9PointCenterValues(i,j,iE,Nx,Nz,_Tc,Tc));
      PetscCall(Get9PointCenterValues(i,j,iP,Nx,Nz,_DPold,DPoldc));
      PetscCall(Get9PointCenterValues(i,j,iP,Nx,Nz,_strain,strain));

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
      PetscCall(GetTensorPointValues(i,j,ix,_eps,e));
      PetscCall(GetTensorPointValues(i,j,ix,_tauold,t));
      PetscCall(RheologyPointwise(i,j,xwt,iwtc,Tc[0],phic[0],P,e,t,ix,res,usr));
      PetscCall(DecompactRheologyVars_DPL(0,res,eta_eff,eta_v,eta_e,eta_p,zeta_eff,zeta_v,zeta_e,DPdl,chis,chip,txx,tzz,txz,tII,DP,dotlam,Z,G,C,sigmat,theta));

      P[0] = pc[1]; P[1] = Plc[1]; P[2] = DPoldc[1]; P[3] = strain[1];
      PetscCall(GetTensorPointValues(im,j,ix,_eps,e));
      PetscCall(GetTensorPointValues(im,j,ix,_tauold,t));
      PetscCall(RheologyPointwise(im,j,xwt,iwtc,Tc[1],phic[1],P,e,t,ix,res,usr));
      PetscCall(DecompactRheologyVars_DPL(1,res,eta_eff,eta_v,eta_e,eta_p,zeta_eff,zeta_v,zeta_e,DPdl,chis,chip,txx,tzz,txz,tII,DP,dotlam,Z,G,C,sigmat,theta));

      P[0] = pc[2]; P[1] = Plc[2]; P[2] = DPoldc[2]; P[3] = strain[2];
      PetscCall(GetTensorPointValues(ip,j,ix,_eps,e));
      PetscCall(GetTensorPointValues(ip,j,ix,_tauold,t));
      PetscCall(RheologyPointwise(ip,j,xwt,iwtc,Tc[2],phic[2],P,e,t,ix,res,usr));
      PetscCall(DecompactRheologyVars_DPL(2,res,eta_eff,eta_v,eta_e,eta_p,zeta_eff,zeta_v,zeta_e,DPdl,chis,chip,txx,tzz,txz,tII,DP,dotlam,Z,G,C,sigmat,theta));

      P[0] = pc[3]; P[1] = Plc[3]; P[2] = DPoldc[3]; P[3] = strain[3];
      PetscCall(GetTensorPointValues(i,jm,ix,_eps,e));
      PetscCall(GetTensorPointValues(i,jm,ix,_tauold,t));
      PetscCall(RheologyPointwise(i,jm,xwt,iwtc,Tc[3],phic[3],P,e,t,ix,res,usr));
      PetscCall(DecompactRheologyVars_DPL(3,res,eta_eff,eta_v,eta_e,eta_p,zeta_eff,zeta_v,zeta_e,DPdl,chis,chip,txx,tzz,txz,tII,DP,dotlam,Z,G,C,sigmat,theta));

      P[0] = pc[4]; P[1] = Plc[4]; P[2] = DPoldc[4]; P[3] = strain[4];
      PetscCall(GetTensorPointValues(i,jp,ix,_eps,e));
      PetscCall(GetTensorPointValues(i,jp,ix,_tauold,t));
      PetscCall(RheologyPointwise(i,jp,xwt,iwtc,Tc[4],phic[4],P,e,t,ix,res,usr));
      PetscCall(DecompactRheologyVars_DPL(4,res,eta_eff,eta_v,eta_e,eta_p,zeta_eff,zeta_v,zeta_e,DPdl,chis,chip,txx,tzz,txz,tII,DP,dotlam,Z,G,C,sigmat,theta));

      // corner points
      PetscScalar Tcorner[4], phicorner[4], pcorner[4], Plcorner[4], DPoldcorner[4], straincorner[4];
      PetscCall(GetCornerAvgFromCenter(Tc,Tcorner));
      PetscCall(GetCornerAvgFromCenter(phic,phicorner));
      PetscCall(GetCornerAvgFromCenter(pc,pcorner));
      PetscCall(GetCornerAvgFromCenter(Plc,Plcorner));
      PetscCall(GetCornerAvgFromCenter(DPoldc,DPoldcorner));
      PetscCall(GetCornerAvgFromCenter(strain,straincorner));
      
      ii = 0; ix[0] = ixxn[ii]; ix[1] = izzn[ii]; ix[2] = ixzn[ii]; ix[3] = iIIn[ii]; P[0] = pcorner[ii]; P[1] = Plcorner[ii]; P[2] = DPoldcorner[ii]; P[3] = straincorner[ii];
      PetscCall(GetTensorPointValues(i,j,ix,_eps,e));
      PetscCall(GetTensorPointValues(i,j,ix,_tauold,t));
      PetscCall(RheologyPointwise(i,j,xwt,iwtld,Tcorner[0],phicorner[0],P,e,t,ix,res,usr));
      PetscCall(DecompactRheologyVars_DPL(5,res,eta_eff,eta_v,eta_e,eta_p,zeta_eff,zeta_v,zeta_e,DPdl,chis,chip,txx,tzz,txz,tII,DP,dotlam,Z,G,C,sigmat,theta));

      ii = 1; ix[0] = ixxn[ii]; ix[1] = izzn[ii]; ix[2] = ixzn[ii]; ix[3] = iIIn[ii]; P[0] = pcorner[ii]; P[1] = Plcorner[ii]; P[2] = DPoldcorner[ii]; P[3] = straincorner[ii];
      PetscCall(GetTensorPointValues(i,j,ix,_eps,e));
      PetscCall(GetTensorPointValues(i,j,ix,_tauold,t));
      PetscCall(RheologyPointwise(i,j,xwt,iwtrd,Tcorner[1],phicorner[1],P,e,t,ix,res,usr));
      PetscCall(DecompactRheologyVars_DPL(6,res,eta_eff,eta_v,eta_e,eta_p,zeta_eff,zeta_v,zeta_e,DPdl,chis,chip,txx,tzz,txz,tII,DP,dotlam,Z,G,C,sigmat,theta));

      ii = 2; ix[0] = ixxn[ii]; ix[1] = izzn[ii]; ix[2] = ixzn[ii]; ix[3] = iIIn[ii]; P[0] = pcorner[ii]; P[1] = Plcorner[ii]; P[2] = DPoldcorner[ii]; P[3] = straincorner[ii];
      PetscCall(GetTensorPointValues(i,j,ix,_eps,e));
      PetscCall(GetTensorPointValues(i,j,ix,_tauold,t));
      PetscCall(RheologyPointwise(i,j,xwt,iwtlu,Tcorner[2],phicorner[2],P,e,t,ix,res,usr));
      PetscCall(DecompactRheologyVars_DPL(7,res,eta_eff,eta_v,eta_e,eta_p,zeta_eff,zeta_v,zeta_e,DPdl,chis,chip,txx,tzz,txz,tII,DP,dotlam,Z,G,C,sigmat,theta));

      ii = 3; ix[0] = ixxn[ii]; ix[1] = izzn[ii]; ix[2] = ixzn[ii]; ix[3] = iIIn[ii]; P[0] = pcorner[ii]; P[1] = Plcorner[ii]; P[2] = DPoldcorner[ii]; P[3] = straincorner[ii];
      PetscCall(GetTensorPointValues(i,j,ix,_eps,e));
      PetscCall(GetTensorPointValues(i,j,ix,_tauold,t));
      PetscCall(RheologyPointwise(i,j,xwt,iwtru,Tcorner[3],phicorner[3],P,e,t,ix,res,usr));
      PetscCall(DecompactRheologyVars_DPL(8,res,eta_eff,eta_v,eta_e,eta_p,zeta_eff,zeta_v,zeta_e,DPdl,chis,chip,txx,tzz,txz,tII,DP,dotlam,Z,G,C,sigmat,theta));

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
          PetscCall(GetMatPhaseFraction(i,j,xwt,idx,usr->nph,wt)); 
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
      PetscCall(GetMatPhaseFraction(i,j,xwt,iwtc,usr->nph,wt)); 
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
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL));
  PetscCall(DMStagVecRestoreArray(dmcoeff,coefflocal,&c));
  PetscCall(DMLocalToGlobalBegin(dmcoeff,coefflocal,INSERT_VALUES,coeff)); 
  PetscCall(DMLocalToGlobalEnd  (dmcoeff,coefflocal,INSERT_VALUES,coeff)); 
  PetscCall(VecDestroy(&coefflocal)); 

  PetscCall(DMStagVecRestoreArrayRead(dm,xlocal,&xx));
  PetscCall(DMRestoreLocalVector(dm,&xlocal)); 

  PetscCall(DMStagVecRestoreArrayRead(usr->dmeps,xepslocal,&_eps));
  PetscCall(DMRestoreLocalVector(usr->dmeps,&xepslocal)); 
  PetscCall(DMStagVecRestoreArrayRead(usr->dmeps,xtauoldlocal,&_tauold));
  PetscCall(DMRestoreLocalVector(usr->dmeps,&xtauoldlocal)); 

  PetscCall(DMStagVecRestoreArray(usr->dmeps,xtaulocal,&_tau));
  PetscCall(DMLocalToGlobalBegin(usr->dmeps,xtaulocal,INSERT_VALUES,usr->xtau)); 
  PetscCall(DMLocalToGlobalEnd  (usr->dmeps,xtaulocal,INSERT_VALUES,usr->xtau)); 
  PetscCall(DMRestoreLocalVector(usr->dmeps,&xtaulocal)); 

  PetscCall(DMStagVecRestoreArrayRead(usr->dmT,xTlocal,&_Tc));
  PetscCall(DMRestoreLocalVector(usr->dmT,&xTlocal)); 

  PetscCall(DMStagVecRestoreArrayRead(usr->dmphi,xphilocal,&_phic));
  PetscCall(DMRestoreLocalVector(usr->dmphi,&xphilocal)); 

  PetscCall(DMStagVecRestoreArrayRead(usr->dmPlith,xPlithlocal,&_Plith));
  PetscCall(DMRestoreLocalVector(usr->dmPlith,&xPlithlocal)); 
  PetscCall(DMStagVecRestoreArrayRead(usr->dmPlith,xDPoldlocal,&_DPold));
  PetscCall(DMRestoreLocalVector(usr->dmPlith,&xDPoldlocal)); 

  PetscCall(DMStagVecRestoreArray(usr->dmPlith,xDPlocal,&_DP));
  PetscCall(DMLocalToGlobalBegin(usr->dmPlith,xDPlocal,INSERT_VALUES,usr->xDP)); 
  PetscCall(DMLocalToGlobalEnd  (usr->dmPlith,xDPlocal,INSERT_VALUES,usr->xDP)); 
  PetscCall(DMRestoreLocalVector(usr->dmPlith,&xDPlocal)); 

  PetscCall(DMStagVecRestoreArray(usr->dmPlith,xplastlocal,&_plast));
  PetscCall(DMLocalToGlobalBegin(usr->dmPlith,xplastlocal,INSERT_VALUES,usr->xplast)); 
  PetscCall(DMLocalToGlobalEnd  (usr->dmPlith,xplastlocal,INSERT_VALUES,usr->xplast)); 
  PetscCall(DMRestoreLocalVector(usr->dmPlith,&xplastlocal)); 

  PetscCall(DMStagVecRestoreArrayRead(usr->dmPlith,xstrainlocal,&_strain));
  PetscCall(DMRestoreLocalVector(usr->dmPlith,&xstrainlocal)); 

  PetscCall(DMStagVecRestoreArray(usr->dmmatProp,xmatProplocal,&_matProp));
  PetscCall(DMLocalToGlobalBegin(usr->dmmatProp,xmatProplocal,INSERT_VALUES,usr->xmatProp)); 
  PetscCall(DMLocalToGlobalEnd  (usr->dmmatProp,xmatProplocal,INSERT_VALUES,usr->xmatProp)); 
  PetscCall(DMRestoreLocalVector(usr->dmmatProp,&xmatProplocal)); 

  PetscCall(DMStagVecRestoreArray(usr->dmMPhase,xMPhaselocal,&xwt));
  PetscCall(VecDestroy(&xMPhaselocal)); 
 
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
PetscErrorCode DecompactRheologyVars_DPL(PetscInt ii,PetscScalar *res,PetscScalar *eta_eff,PetscScalar *eta_v,PetscScalar *eta_e,PetscScalar *eta_p,PetscScalar *zeta_eff,
                                     PetscScalar *zeta_v,PetscScalar *zeta_e,PetscScalar *DPdl,PetscScalar *chis,PetscScalar *chip,
                                     PetscScalar *txx,PetscScalar *tzz,PetscScalar *txz,PetscScalar *tII,PetscScalar *DP,PetscScalar *dotlam,
                                     PetscScalar *Z,PetscScalar *G,PetscScalar *C,PetscScalar *sigmat,PetscScalar *theta)
{
  PetscFunctionBeginUser;

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

  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscFunctionBeginUser;

  if (usr->par->rheology==-1){ PetscCall(RheologyPointwise_VEP(i,j,xwt,iwt,T,phi,P,eps,tauold,ix,res,usr)); }
  if (usr->par->rheology==0) { PetscCall(RheologyPointwise_V(i,j,xwt,iwt,T,phi,P,eps,tauold,ix,res,usr)); }
  if (usr->par->rheology==1) { PetscCall(RheologyPointwise_VE(i,j,xwt,iwt,T,phi,P,eps,tauold,ix,res,usr)); }
  if (usr->par->rheology==2) { PetscCall(RheologyPointwise_DPL(i,j,xwt,iwt,T,phi,P,eps,tauold,ix,res,usr)); }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// RheologyPointwise_VEP: Legacy function as in previous VEP formulations
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

  PetscCall(GetMatPhaseFraction(i,j,xwt,iwt,usr->nph,wt)); 

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

  PetscFunctionReturn(PETSC_SUCCESS);
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

  PetscCall(GetMatPhaseFraction(i,j,xwt,iwt,usr->nph,wt)); 

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

      if (usr->mat_nd[iph].eta_func==2)  meta_v[iph] = ViscosityHarmonicAvg(meta_v[iph],usr->nd->eta_min,usr->nd->eta_max);
      if (usr->mat_nd[iph].zeta_func==2) mzeta_v[iph]= ViscosityHarmonicAvg(mzeta_v[iph],usr->nd->eta_min,usr->nd->eta_max);

      // meta_v[iph] = ViscosityHarmonicAvg(meta_v[iph],usr->nd->eta_min,usr->nd->eta_max);
      // mzeta_v[iph]= ViscosityHarmonicAvg(mzeta_v[iph],usr->nd->eta_min,usr->nd->eta_max);

      inv_meta_v  = 1.0/meta_v[iph];
      inv_mzeta_v = 1.0/mzeta_v[iph];

      // elastic and plastic parameters
      mZ[iph] = (1.0-phi)*usr->mat_nd[iph].Z0;
      // mZ[iph] = PoroElasticModulus(usr->mat_nd[iph].Z0,usr->nd->Zmax,phi);
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

        PetscCall(Plastic_LocalSolver(xve,mC[iph],msigmat[iph],theta_rad,Pf,phi,usr,stressSol)); 
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
      mchip[iph] = phis * mzeta[iph]*inv_mzeta_e;
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

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
PetscErrorCode RheologyPointwise_VE(PetscInt i, PetscInt j, PetscScalar ***xwt, PetscInt *iwt, PetscScalar T, PetscScalar phi, PetscScalar *P, 
                                 PetscScalar *eps, PetscScalar *tauold, PetscInt *ix, PetscScalar *res, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       iph;
  PetscScalar    dt, tf_tol, p, Plith, DPold, Tdim, phis, dotlam, strain;
  PetscScalar    eta_v, zeta_v, eta_e, zeta_e, eta_p, zeta_p, eta, zeta, chip, chis;
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

  PetscCall(GetMatPhaseFraction(i,j,xwt,iwt,usr->nph,wt)); 

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

      if (usr->mat_nd[iph].eta_func==2)  meta_v[iph] = ViscosityHarmonicAvg(meta_v[iph],usr->nd->eta_min,usr->nd->eta_max);
      if (usr->mat_nd[iph].zeta_func==2) mzeta_v[iph]= ViscosityHarmonicAvg(mzeta_v[iph],usr->nd->eta_min,usr->nd->eta_max);

      // meta_v[iph] = ViscosityHarmonicAvg(meta_v[iph],usr->nd->eta_min,usr->nd->eta_max);
      // mzeta_v[iph]= ViscosityHarmonicAvg(mzeta_v[iph],usr->nd->eta_min,usr->nd->eta_max);

      inv_meta_v  = 1.0/meta_v[iph];
      inv_mzeta_v = 1.0/mzeta_v[iph];

      // elastic and plastic parameters
      mZ[iph] = (1.0-phi)*usr->mat_nd[iph].Z0;
      // mZ[iph] = PoroElasticModulus(usr->mat_nd[iph].Z0,usr->nd->Zmax,phi);
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
      // msigmat[iph] = TensileStrength(mC[iph],4.0,usr->mat_nd[iph].sigmat,0); 
      msigmat[iph] = TensileStrength(mC[iph],4.0,usr->mat_nd[iph].sigmat,1); // user defined sigmat

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

      // VISCO-ELASTIC
      meta[iph] = meta_ve; 
      mzeta[iph]= mzeta_ve;
      mdotlam[iph] = 0.0;
      mDPdl[iph] = 0.0;

      // elastic stress evolution parameter
      mchis[iph] = phis * meta[iph]*inv_meta_e;
      mchip[iph] = phis * mzeta[iph]*inv_mzeta_e;
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

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
PetscErrorCode RheologyPointwise_V(PetscInt i, PetscInt j, PetscScalar ***xwt, PetscInt *iwt, PetscScalar T, PetscScalar phi, PetscScalar *P, 
                                 PetscScalar *eps, PetscScalar *tauold, PetscInt *ix, PetscScalar *res, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       iph;
  PetscScalar    dt, tf_tol, p, Plith, DPold, Tdim, phis, dotlam, strain;
  PetscScalar    eta_v, zeta_v, eta_e, zeta_e, eta_p, zeta_p, eta, zeta, chip, chis;
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
  PetscScalar  wt[MAX_MAT_PHASE], meta_v[MAX_MAT_PHASE], mzeta_v[MAX_MAT_PHASE];//, meta_e[MAX_MAT_PHASE], mzeta_e[MAX_MAT_PHASE];
  // PetscScalar  mC[MAX_MAT_PHASE], mZ[MAX_MAT_PHASE], mG[MAX_MAT_PHASE], msigmat[MAX_MAT_PHASE], mtheta[MAX_MAT_PHASE];
  // PetscScalar  meta[MAX_MAT_PHASE], mzeta[MAX_MAT_PHASE], mchis[MAX_MAT_PHASE], mchip[MAX_MAT_PHASE], mdotlam[MAX_MAT_PHASE];
  // PetscScalar  meta_p[MAX_MAT_PHASE], mDPdl[MAX_MAT_PHASE];

  PetscCall(GetMatPhaseFraction(i,j,xwt,iwt,usr->nph,wt)); 

  for (iph = 0; iph < usr->nph; iph++) {
    if (wt[iph]==0.0) { // do not solve rheology if material phase not present
      meta_v[iph]  = 0.0;
      mzeta_v[iph] = 0.0;
      // meta_e[iph]  = 0.0;
      // mzeta_e[iph] = 0.0;
      // meta_p[iph]  = 0.0;
      // mDPdl[iph]   = 0.0;
      // meta[iph]    = 0.0;
      // mzeta[iph]   = 0.0;
      // mchis[iph]   = 0.0;
      // mchip[iph]   = 0.0;
      // mdotlam[iph] = 0.0;
      // mG[iph]      = 0.0;
      // mZ[iph]      = 0.0;
      // mC[iph]      = 0.0;
      // msigmat[iph] = 0.0;
      // mtheta[iph]  = 0.0;
    } else {
      // PetscScalar meta_ve, mzeta_ve, inv_meta_v, inv_meta_e, inv_mzeta_v, inv_mzeta_e;

      // viscous rheology
      // meta_v[iph]  = usr->mat_nd[iph].eta0;
      // mzeta_v[iph] = usr->mat_nd[iph].zeta0;
      meta_v[iph]  = ShearViscosity(usr->mat_nd[iph].eta0,Tdim,phi,usr->par->EoR,usr->par->Teta0,usr->par->beta,usr->mat_nd[iph].eta_func);
      mzeta_v[iph] = CompactionViscosity(usr->mat_nd[iph].zeta0,Tdim,phi,usr->par->EoR,usr->par->Teta0,usr->par->phi_min,usr->par->zetaExp,usr->mat_nd[iph].zeta_func); 

      if (usr->mat_nd[iph].eta_func==2)  meta_v[iph] = ViscosityHarmonicAvg(meta_v[iph],usr->nd->eta_min,usr->nd->eta_max);
      if (usr->mat_nd[iph].zeta_func==2) mzeta_v[iph]= ViscosityHarmonicAvg(mzeta_v[iph],usr->nd->eta_min,usr->nd->eta_max);

      // inv_meta_v  = 1.0/meta_v[iph];
      // inv_mzeta_v = 1.0/mzeta_v[iph];

      // // elastic and plastic parameters
      // mZ[iph] = PoroElasticModulus(usr->mat_nd[iph].Z0,usr->nd->Zmax,phi);
      // mG[iph] = ElasticShearModulus(usr->mat_nd[iph].G,phi);
      // mC[iph] = usr->mat_nd[iph].C;
      // mtheta[iph]  = usr->mat_nd[iph].theta;
      // msigmat[iph] = usr->mat_nd[iph].sigmat;
    
      // // elastic rheology
      // meta_e[iph]  = mG[iph]*dt;
      // mzeta_e[iph] = mZ[iph]*dt;

      // inv_meta_e  = 0.0;
      // inv_mzeta_e = 0.0;

      // // visco-elastic 
      // meta_ve  = PetscPowScalar(inv_meta_v+inv_meta_e,-1.0);
      // mzeta_ve = PetscPowScalar(inv_mzeta_v+inv_mzeta_e,-1.0);

      // // initialize plastic viscosity
      // meta_p[iph] = usr->nd->eta_max;

      // // modified deviatoric and volumetric strain rates
      // exxp = (exx-div13) + 0.5*phis*told_xx*inv_meta_e;
      // ezzp = (ezz-div13) + 0.5*phis*told_zz*inv_meta_e;
      // exzp = exz + 0.5*phis*told_xz*inv_meta_e;
      // eIIp = TensorSecondInvariant(exxp,ezzp,exzp);
      // divp = (exx+ezz) - phis*DPold*inv_mzeta_e;

      // // trial VE stress
      // txxt = 2*meta_ve*exxp/phis;
      // tzzt = 2*meta_ve*ezzp/phis;
      // txzt = 2*meta_ve*exzp/phis;
      // tIIt = TensorSecondInvariant(txxt,tzzt,txzt);
      // dpt = -mzeta_ve * divp/phis;

      // // VISCOUS
      // meta[iph] = meta_v[iph]; 
      // mzeta[iph]= mzeta_v[iph];
      // mdotlam[iph] = 0.0;
      // mDPdl[iph] = 0.0;

      // // elastic stress evolution parameter
      // mchis[iph] = phis * meta[iph]*inv_meta_e;
      // mchip[iph] = phis * mzeta[iph]*inv_mzeta_e;
    }
  }

  eta_v  = WeightAverageValue(meta_v,wt,usr->nph); 
  zeta_v = WeightAverageValue(mzeta_v,wt,usr->nph); 
  // eta_e  = WeightAverageValue(meta_e,wt,usr->nph); 
  // zeta_e = WeightAverageValue(mzeta_e,wt,usr->nph); 
  // eta_p  = WeightAverageValue(meta_p,wt,usr->nph); 
  // eta    = WeightAverageValue(meta,wt,usr->nph); 
  // zeta   = WeightAverageValue(mzeta,wt,usr->nph); 
  // chis   = WeightAverageValue(mchis,wt,usr->nph); 
  // chip   = WeightAverageValue(mchip,wt,usr->nph); 
  // dotlam = WeightAverageValue(mdotlam,wt,usr->nph); 
  // DPdl   = WeightAverageValue(mDPdl,wt,usr->nph); 

  eta_e  = usr->nd->eta_max; 
  zeta_e = usr->nd->eta_max; 
  eta_p  = usr->nd->eta_max; 
  eta    = eta_v; 
  zeta   = zeta_v; 
  chis   = 0.0; 
  chip   = 0.0; 
  dotlam = 0.0; 
  DPdl   = 0.0; 

  PetscScalar C, G, Z, sigmat, theta;
  // G      = WeightAverageValue(mG,wt,usr->nph); 
  // Z      = WeightAverageValue(mZ,wt,usr->nph); 
  // C      = WeightAverageValue(mC,wt,usr->nph); 
  // sigmat = WeightAverageValue(msigmat,wt,usr->nph); 
  // theta  = WeightAverageValue(mtheta,wt,usr->nph); 

  G      = 0.0; 
  Z      = 0.0; 
  C      = 0.0; 
  sigmat = 0.0; 
  theta  = 0.0; 

  PetscScalar inv_eta_e, inv_zeta_e;
  inv_eta_e  = 0.0;
  inv_zeta_e = 0.0;

  // update effective deviatoric and volumetric strain rates with Marker PhaseAverage
  exxp = (exx-div13) + 0.5*phis*told_xx*inv_eta_e;
  ezzp = (ezz-div13) + 0.5*phis*told_zz*inv_eta_e;
  exzp = exz + 0.5*phis*told_xz*inv_eta_e;
  divp = (exx+ezz) - phis*DPold*inv_zeta_e;

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

  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscFunctionBeginUser;

  // Get dm and solution vector for Stokes velocity
  dmPV = usr->dmPV;
  xPV  = usr->xPV;
  
  PetscCall(DMGetLocalVector(dmPV, &xPVlocal)); 
  PetscCall(DMGlobalToLocal (dmPV, xPV, INSERT_VALUES, xPVlocal)); 
  PetscCall(DMStagVecGetArrayRead(dmPV,xPVlocal,&_xPVlocal));
  
  // Get solution vector for temperature
  PetscCall(DMGetLocalVector(dm,&xlocal)); 
  PetscCall(DMGlobalToLocal (dm,x,INSERT_VALUES,xlocal)); 

  // get material phase fractions
  PetscCall(DMCreateLocalVector(usr->dmMPhase, &xMPhaselocal)); 
  PetscCall(DMGlobalToLocalBegin(usr->dmMPhase,usr->xMPhase,INSERT_VALUES,xMPhaselocal)); 
  PetscCall(DMGlobalToLocalEnd  (usr->dmMPhase,usr->xMPhase,INSERT_VALUES,xMPhaselocal)); 
  PetscCall(DMStagVecGetArray(usr->dmMPhase, xMPhaselocal, &xwt)); 

  // Get coefficient local vector
  PetscCall(DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 
  PetscCall(DMStagGetGlobalSizes(dmcoeff,&Nx,&Nz,NULL));
  PetscCall(DMCreateLocalVector(dmcoeff, &coefflocal)); 
  PetscCall(DMStagVecGetArray(dmcoeff, coefflocal, &c)); 

  // get location slots
  PetscInt  pv_slot[4],iL,iR,iU,iD;
  iL = 0; iR  = 1;
  iD = 2; iU  = 3;
  PetscCall(DMStagGetLocationSlot(usr->dmPV,LEFT,   PV_FACE_VS,   &pv_slot[iL]));
  PetscCall(DMStagGetLocationSlot(usr->dmPV,RIGHT,  PV_FACE_VS,   &pv_slot[iR]));
  PetscCall(DMStagGetLocationSlot(usr->dmPV,DOWN,   PV_FACE_VS,   &pv_slot[iD]));
  PetscCall(DMStagGetLocationSlot(usr->dmPV,UP,     PV_FACE_VS,   &pv_slot[iU]));

  PetscInt  e_slot[2],iA,iC;
  iA = 0; iC = 1;
  PetscCall(DMStagGetLocationSlot(dmcoeff,ELEMENT,TCOEFF_ELEMENT_A, &e_slot[iA])); 
  PetscCall(DMStagGetLocationSlot(dmcoeff,ELEMENT,TCOEFF_ELEMENT_C, &e_slot[iC])); 

  PetscInt  B_slot[4],u_slot[4];
  PetscCall(DMStagGetLocationSlot(dmcoeff,LEFT, TCOEFF_FACE_B,&B_slot[0]));
  PetscCall(DMStagGetLocationSlot(dmcoeff,RIGHT,TCOEFF_FACE_B,&B_slot[1]));
  PetscCall(DMStagGetLocationSlot(dmcoeff,DOWN, TCOEFF_FACE_B,&B_slot[2]));
  PetscCall(DMStagGetLocationSlot(dmcoeff,UP,   TCOEFF_FACE_B,&B_slot[3]));

  PetscCall(DMStagGetLocationSlot(dmcoeff,LEFT, TCOEFF_FACE_u,&u_slot[0]));
  PetscCall(DMStagGetLocationSlot(dmcoeff,RIGHT,TCOEFF_FACE_u,&u_slot[1]));
  PetscCall(DMStagGetLocationSlot(dmcoeff,DOWN, TCOEFF_FACE_u,&u_slot[2]));
  PetscCall(DMStagGetLocationSlot(dmcoeff,UP,   TCOEFF_FACE_u,&u_slot[3]));

  PetscInt iwtl[MAX_MAT_PHASE],iwtr[MAX_MAT_PHASE],iwtd[MAX_MAT_PHASE],iwtu[MAX_MAT_PHASE]; 
  for (ii = 0; ii < usr->nph; ii++) { 
    PetscCall(DMStagGetLocationSlot(usr->dmMPhase, LEFT, ii, &iwtl[ii])); 
    PetscCall(DMStagGetLocationSlot(usr->dmMPhase, RIGHT,ii, &iwtr[ii])); 
    PetscCall(DMStagGetLocationSlot(usr->dmMPhase, DOWN, ii, &iwtd[ii]));  
    PetscCall(DMStagGetLocationSlot(usr->dmMPhase, UP,   ii, &iwtu[ii])); 
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
          PetscCall(GetMatPhaseFraction(i,j,xwt,idx,usr->nph,wt)); 
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
  PetscCall(DMStagVecRestoreArray(dmcoeff,coefflocal,&c));
  PetscCall(DMLocalToGlobalBegin(dmcoeff,coefflocal,INSERT_VALUES,coeff)); 
  PetscCall(DMLocalToGlobalEnd  (dmcoeff,coefflocal,INSERT_VALUES,coeff)); 
  PetscCall(VecDestroy(&coefflocal)); 

  PetscCall(DMRestoreLocalVector(dm,&xlocal)); 
  
  PetscCall(DMStagVecRestoreArrayRead(dmPV,xPVlocal,&_xPVlocal));
  PetscCall(DMRestoreLocalVector(dmPV, &xPVlocal)); 

  PetscCall(DMStagVecRestoreArray(usr->dmMPhase,xMPhaselocal,&xwt));
  PetscCall(VecDestroy(&xMPhaselocal)); 

  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscFunctionBeginUser;

  // Get dm and solution vector for Stokes velocity
  dmPV = usr->dmPV;
  xPV  = usr->xPV;
  
  PetscCall(DMGetLocalVector(dmPV, &xPVlocal)); 
  PetscCall(DMGlobalToLocal (dmPV, xPV, INSERT_VALUES, xPVlocal)); 
  PetscCall(DMStagVecGetArrayRead(dmPV,xPVlocal,&_xPVlocal));
  
  // Get solution vector for temperature
  PetscCall(DMGetLocalVector(dm,&xlocal)); 
  PetscCall(DMGlobalToLocal (dm,x,INSERT_VALUES,xlocal)); 

  // get material phase fractions
  PetscCall(DMCreateLocalVector(usr->dmMPhase, &xMPhaselocal)); 
  PetscCall(DMGlobalToLocalBegin(usr->dmMPhase,usr->xMPhase,INSERT_VALUES,xMPhaselocal)); 
  PetscCall(DMGlobalToLocalEnd  (usr->dmMPhase,usr->xMPhase,INSERT_VALUES,xMPhaselocal)); 
  PetscCall(DMStagVecGetArray(usr->dmMPhase, xMPhaselocal, &xwt)); 

  // Get coefficient local vector
  PetscCall(DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 
  PetscCall(DMStagGetGlobalSizes(dmcoeff,&Nx,&Nz,NULL));
  PetscCall(DMCreateLocalVector(dmcoeff, &coefflocal)); 
  PetscCall(DMStagVecGetArray(dmcoeff, coefflocal, &c)); 

   // get location slots
  PetscInt  pv_slot[4],iL,iR,iU,iD;
  iL = 0; iR  = 1; iD = 2; iU  = 3;
  PetscCall(DMStagGetLocationSlot(usr->dmPV,LEFT,   PV_FACE_VS,   &pv_slot[iL]));
  PetscCall(DMStagGetLocationSlot(usr->dmPV,RIGHT,  PV_FACE_VS,   &pv_slot[iR]));
  PetscCall(DMStagGetLocationSlot(usr->dmPV,DOWN,   PV_FACE_VS,   &pv_slot[iD]));
  PetscCall(DMStagGetLocationSlot(usr->dmPV,UP,     PV_FACE_VS,   &pv_slot[iU]));
  
    PetscInt  e_slot[2],iA,iC;
  iA = 0; iC = 1;
  PetscCall(DMStagGetLocationSlot(dmcoeff,ELEMENT,TCOEFF_ELEMENT_A, &e_slot[iA])); 
  PetscCall(DMStagGetLocationSlot(dmcoeff,ELEMENT,TCOEFF_ELEMENT_C, &e_slot[iC])); 

  PetscInt  B_slot[4],u_slot[4];
  PetscCall(DMStagGetLocationSlot(dmcoeff,LEFT, TCOEFF_FACE_B,&B_slot[0]));
  PetscCall(DMStagGetLocationSlot(dmcoeff,RIGHT,TCOEFF_FACE_B,&B_slot[1]));
  PetscCall(DMStagGetLocationSlot(dmcoeff,DOWN, TCOEFF_FACE_B,&B_slot[2]));
  PetscCall(DMStagGetLocationSlot(dmcoeff,UP,   TCOEFF_FACE_B,&B_slot[3]));

  PetscCall(DMStagGetLocationSlot(dmcoeff,LEFT, TCOEFF_FACE_u,&u_slot[0]));
  PetscCall(DMStagGetLocationSlot(dmcoeff,RIGHT,TCOEFF_FACE_u,&u_slot[1]));
  PetscCall(DMStagGetLocationSlot(dmcoeff,DOWN, TCOEFF_FACE_u,&u_slot[2]));
  PetscCall(DMStagGetLocationSlot(dmcoeff,UP,   TCOEFF_FACE_u,&u_slot[3]));

  PetscInt iwtc[MAX_MAT_PHASE]; 
  for (ii = 0; ii < usr->nph; ii++) { 
    PetscCall(DMStagGetLocationSlot(usr->dmMPhase, ELEMENT, ii, &iwtc[ii]));  
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
        PetscCall(GetMatPhaseFraction(i,j,xwt,iwtc,usr->nph,wt)); 
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
  PetscCall(DMStagVecRestoreArray(dmcoeff,coefflocal,&c));
  PetscCall(DMLocalToGlobalBegin(dmcoeff,coefflocal,INSERT_VALUES,coeff)); 
  PetscCall(DMLocalToGlobalEnd  (dmcoeff,coefflocal,INSERT_VALUES,coeff)); 
  PetscCall(VecDestroy(&coefflocal)); 

  PetscCall(DMRestoreLocalVector(dm,&xlocal)); 
  
  PetscCall(DMStagVecRestoreArrayRead(dmPV,xPVlocal,&_xPVlocal));
  PetscCall(DMRestoreLocalVector(dmPV, &xPVlocal)); 

  PetscCall(DMStagVecRestoreArray(usr->dmMPhase,xMPhaselocal,&xwt));
  PetscCall(VecDestroy(&xMPhaselocal)); 

  PetscFunctionReturn(PETSC_SUCCESS);
}