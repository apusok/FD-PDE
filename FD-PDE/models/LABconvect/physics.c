#include "LABconvect.h"

// ---------------------------------------
// FormCoefficient_PV
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormCoefficient_PV"
PetscErrorCode FormCoefficient_PV(FDPDE fd, DM dm, Vec x, DM dmcoeff, Vec coeff, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  NdParams       *nd;
  Params         *par;
  ScalParams     *scal;
  PetscInt       i, j, ii, sx, sz, nx, nz, Nx, Nz;
  DM             dmEnth;
  Vec            coefflocal, xEnthlocal;
  PetscScalar    **coordx,**coordz, k_hat[4];
  PetscInt       iprev, inext, icenter;
  PetscScalar    ***c, ***_xEnthlocal;
  PetscLogDouble tlog[2];
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  PetscTime(&tlog[0]);
  nd  = usr->nd;
  par = usr->par;
  scal= usr->scal;
  dmEnth = usr->dmEnth;

  k_hat[0] = 0.0;
  k_hat[1] = 0.0;
  k_hat[2] = usr->par->k_hat;
  k_hat[3] = usr->par->k_hat;

  // Get dm and solution vector for H, C, phi, T (Enth)
  ierr = DMGetLocalVector(dmEnth, &xEnthlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dmEnth, usr->xEnth, INSERT_VALUES, xEnthlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(dmEnth,xEnthlocal,&_xEnthlocal);CHKERRQ(ierr);

  // Get dmcoeff
  ierr = DMStagGetGlobalSizes(dmcoeff,&Nx,&Nz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmcoeff,LEFT,&iprev);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmcoeff,RIGHT,&inext);CHKERRQ(ierr); 
  ierr = DMStagGetProductCoordinateLocationSlot(dmcoeff,ELEMENT,&icenter);CHKERRQ(ierr);

  // Create coefficient local vector
  ierr = DMCreateLocalVector(dmcoeff, &coefflocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dmcoeff, coefflocal, &c); CHKERRQ(ierr);

  // get location slots
  PetscInt  iphi,iT,iC,iCF;
  ierr = DMStagGetLocationSlot(dmEnth,DMSTAG_ELEMENT,ENTH_ELEMENT_PHI,&iphi);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmEnth,DMSTAG_ELEMENT,ENTH_ELEMENT_T  ,&iT  );CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmEnth,DMSTAG_ELEMENT,ENTH_ELEMENT_C  ,&iC  );CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmEnth,DMSTAG_ELEMENT,ENTH_ELEMENT_CF ,&iCF );CHKERRQ(ierr);

  PetscInt  icoeffA,icoeffC,icoeffD1,icoeffDC,iL,iR,iU,iD;
  PetscInt  A_corner[4],B_face[4],D2_face[4],D3_face[4],D4_face[4];
  iL = 0; iR  = 1; iD = 2; iU  = 3;
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_ELEMENT,PVCOEFF_ELEMENT_A, &icoeffA);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_ELEMENT,PVCOEFF_ELEMENT_C, &icoeffC);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_ELEMENT,PVCOEFF_ELEMENT_D1,&icoeffD1);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_ELEMENT,PVCOEFF_ELEMENT_DC,&icoeffDC);CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_DOWN_LEFT, PVCOEFF_VERTEX_A,&A_corner[0]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_DOWN_RIGHT,PVCOEFF_VERTEX_A,&A_corner[1]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_UP_LEFT,   PVCOEFF_VERTEX_A,&A_corner[2]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_UP_RIGHT,  PVCOEFF_VERTEX_A,&A_corner[3]);CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_LEFT,  PVCOEFF_FACE_B, &B_face[iL]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_RIGHT, PVCOEFF_FACE_B, &B_face[iR]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_DOWN,  PVCOEFF_FACE_B, &B_face[iD]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_UP,    PVCOEFF_FACE_B, &B_face[iU]);CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_LEFT,  PVCOEFF_FACE_D2,&D2_face[iL]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_RIGHT, PVCOEFF_FACE_D2,&D2_face[iR]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_DOWN,  PVCOEFF_FACE_D2,&D2_face[iD]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_UP,    PVCOEFF_FACE_D2,&D2_face[iU]);CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_LEFT,  PVCOEFF_FACE_D3,&D3_face[iL]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_RIGHT, PVCOEFF_FACE_D3,&D3_face[iR]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_DOWN,  PVCOEFF_FACE_D3,&D3_face[iD]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_UP,    PVCOEFF_FACE_D3,&D3_face[iU]);CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_LEFT,  PVCOEFF_FACE_D4,&D4_face[iL]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_RIGHT, PVCOEFF_FACE_D4,&D4_face[iR]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_DOWN,  PVCOEFF_FACE_D4,&D4_face[iD]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_UP,    PVCOEFF_FACE_D4,&D4_face[iU]);CHKERRQ(ierr);
  
  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      PetscScalar   phic[9], Tc[9], C[3], CF[3];
      PetscInt      im, jm, ip, jp;

      // get porosity, temperature, C, CF - center
      if (usr->dtype0!=DM_BOUNDARY_PERIODIC) {
        if (i == 0   ) im = i; else im = i-1;
        if (i == Nx-1) ip = i; else ip = i+1;
      } else {
        im = i-1;
        ip = i+1;
      }
      
      if (j == 0   ) jm = j; else jm = j-1;
      if (j == Nz-1) jp = j; else jp = j+1;

      phic[0] = _xEnthlocal[jm][im][iphi]; // i-1,j-1
      phic[1] = _xEnthlocal[jm][i ][iphi]; // i  ,j-1
      phic[2] = _xEnthlocal[jm][ip][iphi]; // i+1,j-1
      phic[3] = _xEnthlocal[j ][im][iphi]; // i-1,j
      phic[4] = _xEnthlocal[j ][i ][iphi]; // i  ,j
      phic[5] = _xEnthlocal[j ][ip][iphi]; // i+1,j
      phic[6] = _xEnthlocal[jp][im][iphi]; // i-1,j+1
      phic[7] = _xEnthlocal[jp][i ][iphi]; // i  ,j+1
      phic[8] = _xEnthlocal[jp][ip][iphi]; // i+1,j+1

      Tc[0] = _xEnthlocal[jm][im][iT];
      Tc[1] = _xEnthlocal[jm][i ][iT];
      Tc[2] = _xEnthlocal[jm][ip][iT];
      Tc[3] = _xEnthlocal[j ][im][iT];
      Tc[4] = _xEnthlocal[j ][i ][iT];
      Tc[5] = _xEnthlocal[j ][ip][iT];
      Tc[6] = _xEnthlocal[jp][im][iT];
      Tc[7] = _xEnthlocal[jp][i ][iT];
      Tc[8] = _xEnthlocal[jp][ip][iT];

      C[0] = _xEnthlocal[jm][i ][iC]; 
      C[1] = _xEnthlocal[j ][i ][iC]; 
      C[2] = _xEnthlocal[jp][i ][iC]; 

      CF[0] = _xEnthlocal[jm][i ][iCF]; 
      CF[1] = _xEnthlocal[j ][i ][iCF]; 
      CF[2] = _xEnthlocal[jp][i ][iCF]; 

      { // A = delta^2*eta (center, c=1)
        PetscScalar   eta;
        // if (i==0) PetscPrintf(PETSC_COMM_WORLD,"# j=%d phic[4]=%f Tc[4]=%f \n",j,phic[4],Tc[4]);
        eta = ShearViscosity(Tc[4]*par->DT+par->T0,phic[4],par->EoR,par->Teta0,par->lambda,nd->eta_min,nd->eta_max,par->visc_shear);
        c[j][i][icoeffA] = nd->delta*nd->delta*eta;
      }

      { // A = delta^2*eta (corner, c=0)
        PetscScalar   eta, T[4], phi[4];

        // porosity and T on corners
        phi[0] = (phic[0]+phic[1]+phic[3]+phic[4])*0.25; 
        phi[1] = (phic[1]+phic[2]+phic[4]+phic[5])*0.25; 
        phi[2] = (phic[3]+phic[4]+phic[6]+phic[7])*0.25; 
        phi[3] = (phic[4]+phic[5]+phic[7]+phic[8])*0.25;

        T[0] = (Tc[0]+Tc[1]+Tc[3]+Tc[4])*0.25; 
        T[1] = (Tc[1]+Tc[2]+Tc[4]+Tc[5])*0.25; 
        T[2] = (Tc[3]+Tc[4]+Tc[6]+Tc[7])*0.25; 
        T[3] = (Tc[4]+Tc[5]+Tc[7]+Tc[8])*0.25;

        for (ii = 0; ii < 4; ii++) {
          eta = ShearViscosity(T[ii]*par->DT+par->T0,phi[ii],par->EoR,par->Teta0,par->lambda,nd->eta_min,nd->eta_max,par->visc_shear);
          c[j][i][A_corner[ii]] = nd->delta*nd->delta*eta;
        }
      }

      { // B = (phi+B)*k_hat (edges, c=0)
        PetscScalar   rhs[4], Bphi, BC, BT, Buoy = 0.0;
        PetscScalar   phi_in, T_in, C_in, CF_in;

        // Bx = 0
        rhs[0] = 0.0;
        rhs[1] = 0.0;

        // Bz DOWN
        phi_in = (phic[1]+phic[4])*0.5;
        T_in   = (Tc[1]  +Tc[4]  )*0.5;
        C_in   = (C[0]   +C[1]   )*0.5;
        CF_in  = (CF[0]  +CF[1]  )*0.5;

        Bphi = Buoyancy_phi(phi_in,par->buoy_phi);
        BC   = Buoyancy_Composition(C_in,CF_in,phi_in,nd->beta_s,nd->beta_ls,par->buoy_C);
        BT   = Buoyancy_Temperature(T_in,phi_in,nd->alpha_s,nd->alpha_ls,par->buoy_T);
        Buoy = Bphi+BC+BT;
        rhs[2] = par->k_hat*Buoy;

        // Bz UP
        phi_in = (phic[4]+phic[7])*0.5;
        T_in   = (Tc[4]  +Tc[7]  )*0.5;
        C_in   = (C[2]   +C[1]   )*0.5;
        CF_in  = (CF[2]  +CF[1]  )*0.5;

        Bphi = Buoyancy_phi(phi_in,par->buoy_phi);
        BC   = Buoyancy_Composition(C_in,CF_in,phi_in,nd->beta_s,nd->beta_ls,par->buoy_C);
        BT   = Buoyancy_Temperature(T_in,phi_in,nd->alpha_s,nd->alpha_ls,par->buoy_T);
        Buoy = Bphi+BC+BT;
        rhs[3] = par->k_hat*Buoy;

        for (ii = 0; ii < 4; ii++) c[j][i][B_face[ii]] = rhs[ii];
      }

      { // C = 0.0 (center, c=0)
        c[j][i][icoeffC] = 0.0;
      }

      { // D1 = -1/(delta^2*xi), xi=zeta-2/3eta (center, c=2)
        PetscScalar   xi, eta, zeta;
        eta  = ShearViscosity(Tc[4]*par->DT+par->T0,phic[4],par->EoR,par->Teta0,par->lambda,nd->eta_min,nd->eta_max,par->visc_shear);
        zeta = BulkViscosity(nd->visc_ratio,Tc[4]*par->DT+par->T0,phic[4],par->EoR,par->Teta0,par->phi_min,par->zetaExp,nd->eta_min,nd->eta_max,par->visc_bulk); 
        xi = zeta-2.0/3.0*eta;
        c[j][i][icoeffD1] = -1.0/(nd->delta*nd->delta*xi);
      }

      { // D2, D3, D4
        PetscScalar   K[4], D2[4], D3[4], D4[4], Bf[4], phi[4];

        // porosity on edges
        phi[0] = (phic[3]+phic[4])*0.5;
        phi[1] = (phic[5]+phic[4])*0.5;
        phi[2] = (phic[1]+phic[4])*0.5;
        phi[3] = (phic[7]+phic[4])*0.5;

        for (ii = 0; ii < 4; ii++) { 
          K[ii]  = Permeability(phi[ii],usr->par->phi_max,usr->par->n);
          Bf[ii] = FluidBuoyancy(0.0,0.0,usr->nd->alpha_s,usr->nd->beta_s);
          D2[ii] = -K[ii];
          D3[ii] = -K[ii]*(1+Bf[ii])*k_hat[ii];
          D4[ii] = -K[ii];

          // D2 = -K, K = (phi/phi0)^n (edges, c=1)
          c[j][i][D2_face[ii]] = D2[ii];

          // D3 = -K*(1+Bf)*k_hat, K = (phi/phi0)^n, Bf = 0 (edges, c=2)
          c[j][i][D3_face[ii]] = D3[ii];

          // D4 = -K, (edges, c=3)
          c[j][i][D4_face[ii]] = D4[ii];
        }
      }

      { // DC =  0 (center, c=3)
        c[j][i][icoeffDC] = 0.0;
      }
    }
  }

  // Restore arrays, local vectors
  ierr = DMStagRestoreProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL);CHKERRQ(ierr);

  ierr = DMStagVecRestoreArray(dmcoeff,coefflocal,&c);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  ierr = VecDestroy(&coefflocal); CHKERRQ(ierr);
  ierr = DMStagVecRestoreArrayRead(dmEnth,xEnthlocal,&_xEnthlocal);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmEnth, &xEnthlocal); CHKERRQ(ierr);

  PetscTime(&tlog[1]);
  if (usr->par->log_info) {
    printf("  FormCoefficient_PV: total                        %1.2e\n",tlog[1]-tlog[0]);
  }

  PetscFunctionReturn(0);
}

// ---------------------------------------
// FormCoefficient_HC
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormCoefficient_HC"
PetscErrorCode FormCoefficient_HC(FDPDE fd, DM dm, Vec x, DM dmcoeff, Vec coeff, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  Params         *par;
  PetscInt       i, j, sx, sz, nx, nz, Nx,Nz,icenter,iprev,inext;
  Vec            xEnth, xEnthlocal, xPVlocal, xlocal, coefflocal;
  DM             dmEnth;
  PetscScalar    ***c;
  PetscScalar    **coordx,**coordz;
  PetscScalar    ***_xPVlocal,***_xEnthlocal;
  PetscLogDouble tlog[2];
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  
  PetscTime(&tlog[0]);
  // get Stokes-Darcy velocities
  ierr = DMGetLocalVector(usr->dmPV, &xPVlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmPV, usr->xPV, INSERT_VALUES, xPVlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(usr->dmPV,xPVlocal,&_xPVlocal);CHKERRQ(ierr);
  
  // Update enthalpy variables - slow for each iteration
  ierr = FDPDEEnthalpyUpdateDiagnostics(fd,dm,x,&dmEnth,&xEnth); CHKERRQ(ierr);
  ierr = VecCopy(xEnth,usr->xEnth);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dmEnth,&xEnthlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dmEnth,xEnth,INSERT_VALUES,xEnthlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayRead(dmEnth,xEnthlocal,&_xEnthlocal);CHKERRQ(ierr);
  
  par = usr->par;
  // Get domain corners
  ierr = DMStagGetGlobalSizes(dmcoeff,&Nx,&Nz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dmcoeff, &coefflocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dmcoeff, coefflocal, &c); CHKERRQ(ierr);
  
  ierr = DMGetLocalVector(dm,&xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm,x,INSERT_VALUES,xlocal); CHKERRQ(ierr);
  
  ierr = DMStagGetProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmcoeff,ELEMENT,&icenter);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmcoeff,LEFT,&iprev);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmcoeff,RIGHT,&inext);CHKERRQ(ierr);

  // get location slots
  PetscInt  pv_slot[6],phi_slot,iL,iR,iU,iD,iP,iPc;
  iP = 0; iPc = 1;
  iL = 2; iR  = 3;
  iD = 4; iU  = 5;
  ierr = DMStagGetLocationSlot(usr->dmPV,DMSTAG_ELEMENT,PV_ELEMENT_P, &pv_slot[iP]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmPV,DMSTAG_ELEMENT,PV_ELEMENT_PC,&pv_slot[iPc]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmPV,DMSTAG_LEFT,   PV_FACE_VS,   &pv_slot[iL]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmPV,DMSTAG_RIGHT,  PV_FACE_VS,   &pv_slot[iR]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmPV,DMSTAG_DOWN,   PV_FACE_VS,   &pv_slot[iD]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(usr->dmPV,DMSTAG_UP,     PV_FACE_VS,   &pv_slot[iU]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmEnth,DMSTAG_ELEMENT,ENTH_ELEMENT_PHI,&phi_slot);CHKERRQ(ierr);

  PetscInt  e_slot[6],iA1,iB1,iD1,iA2,iB2,iD2;
  iA1 = 0; iB1 = 1; iD1 = 2;
  iA2 = 3; iB2 = 4; iD2 = 5;
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_ELEMENT,HCCOEFF_ELEMENT_A1, &e_slot[iA1]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_ELEMENT,HCCOEFF_ELEMENT_B1, &e_slot[iB1]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_ELEMENT,HCCOEFF_ELEMENT_D1, &e_slot[iD1]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_ELEMENT,HCCOEFF_ELEMENT_A2, &e_slot[iA2]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_ELEMENT,HCCOEFF_ELEMENT_B2, &e_slot[iB2]); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_ELEMENT,HCCOEFF_ELEMENT_D2, &e_slot[iD2]); CHKERRQ(ierr);
  
  PetscInt  c1_slot[4],c2_slot[4],v_slot[4],vf_slot[4],vs_slot[4];
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_LEFT, HCCOEFF_FACE_C1,&c1_slot[0]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_RIGHT,HCCOEFF_FACE_C1,&c1_slot[1]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_DOWN, HCCOEFF_FACE_C1,&c1_slot[2]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_UP,   HCCOEFF_FACE_C1,&c1_slot[3]);CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_LEFT, HCCOEFF_FACE_C2,&c2_slot[0]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_RIGHT,HCCOEFF_FACE_C2,&c2_slot[1]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_DOWN, HCCOEFF_FACE_C2,&c2_slot[2]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_UP,   HCCOEFF_FACE_C2,&c2_slot[3]);CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_LEFT, HCCOEFF_FACE_V,&v_slot[0]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_RIGHT,HCCOEFF_FACE_V,&v_slot[1]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_DOWN, HCCOEFF_FACE_V,&v_slot[2]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_UP,   HCCOEFF_FACE_V,&v_slot[3]);CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_LEFT, HCCOEFF_FACE_VF,&vf_slot[0]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_RIGHT,HCCOEFF_FACE_VF,&vf_slot[1]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_DOWN, HCCOEFF_FACE_VF,&vf_slot[2]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_UP,   HCCOEFF_FACE_VF,&vf_slot[3]);CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_LEFT, HCCOEFF_FACE_VS,&vs_slot[0]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_RIGHT,HCCOEFF_FACE_VS,&vs_slot[1]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_DOWN, HCCOEFF_FACE_VS,&vs_slot[2]);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmcoeff,DMSTAG_UP,   HCCOEFF_FACE_VS,&vs_slot[3]);CHKERRQ(ierr);
  
  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      
      { // ELEMENT
        PetscScalar   Az;
        Az = usr->nd->A*coordz[j][icenter];

        // A1 = exp(-Az)
        c[j][i][e_slot[iA1]] = exp(-Az);
        
        // B1 = -S
        c[j][i][e_slot[iB1]] = -usr->nd->S;
        
        // D1 = 0
        c[j][i][e_slot[iD1]] = 0.0;
        
        // A2 = 1
        c[j][i][e_slot[iA2]] = 1.0;
        
        // B2 = 1
        c[j][i][e_slot[iB2]] = 1.0;
        
        // D2 = 0.0
        c[j][i][e_slot[iD2]] = 0.0;
      }
      
      { // FACES
        PetscInt      im, jm, ii;
        PetscScalar   p[5],Az[4],v[4],vf[4],vs[4],phi[4],Bf,K,gradP[4],gradPc[4],k_hat[4],Q[5],dx,dz;

        Az[0] = usr->nd->A*coordz[j][icenter];
        Az[1] = usr->nd->A*coordz[j][icenter];
        Az[2] = usr->nd->A*coordz[j][iprev];
        Az[3] = usr->nd->A*coordz[j][inext];
        
        k_hat[0] = 0.0;
        k_hat[1] = 0.0;
        k_hat[2] = usr->par->k_hat;
        k_hat[3] = usr->par->k_hat;
        
        // grid spacing - assume constant
        dx = coordx[i][inext]-coordx[i][iprev];
        dz = coordz[j][inext]-coordz[j][iprev];

        // get PV data
        vs[0] = _xPVlocal[j][i][pv_slot[iL]]; 
        vs[1] = _xPVlocal[j][i][pv_slot[iR]];
        vs[2] = _xPVlocal[j][i][pv_slot[iD]];
        vs[3] = _xPVlocal[j][i][pv_slot[iU]];

        p[0] = _xPVlocal[j][i][pv_slot[iP]];

        if (usr->dtype0!=DM_BOUNDARY_PERIODIC) {
          if (i == 0   ) im = i; else im = i-1; p[1] = _xPVlocal[j][im][pv_slot[iP]];
          if (i == Nx-1) im = i; else im = i+1; p[2] = _xPVlocal[j][im][pv_slot[iP]];
        } else {
          im = i-1; p[1] = _xPVlocal[j][im][pv_slot[iP]];
          im = i+1; p[2] = _xPVlocal[j][im][pv_slot[iP]];
        }
        if (j == 0   ) jm = j; else jm = j-1; p[3] = _xPVlocal[jm][i][pv_slot[iP]];
        if (j == Nz-1) jm = j; else jm = j+1; p[4] = _xPVlocal[jm][i][pv_slot[iP]];

        gradP[0] = (p[0]-p[1])/dx;
        gradP[1] = (p[2]-p[0])/dx;
        gradP[2] = (p[0]-p[3])/dz;
        gradP[3] = (p[4]-p[0])/dz;

        p[0] = _xPVlocal[j][i][pv_slot[iPc]];

        if (usr->dtype0!=DM_BOUNDARY_PERIODIC) {
          if (i == 0   ) im = i; else im = i-1; p[1] = _xPVlocal[j][im][pv_slot[iPc]];
          if (i == Nx-1) im = i; else im = i+1; p[2] = _xPVlocal[j][im][pv_slot[iPc]];
        } else {
          im = i-1; p[1] = _xPVlocal[j][im][pv_slot[iPc]];
          im = i+1; p[2] = _xPVlocal[j][im][pv_slot[iPc]];
        }
        if (j == 0   ) jm = j; else jm = j-1; p[3] = _xPVlocal[jm][i][pv_slot[iPc]];
        if (j == Nz-1) jm = j; else jm = j+1; p[4] = _xPVlocal[jm][i][pv_slot[iPc]];

        gradPc[0] = (p[0]-p[1])/dx;
        gradPc[1] = (p[2]-p[0])/dx;
        gradPc[2] = (p[0]-p[3])/dz;
        gradPc[3] = (p[4]-p[0])/dz;

        // porosity
        Q[0] = _xEnthlocal[j][i][phi_slot];

        if (usr->dtype0!=DM_BOUNDARY_PERIODIC) {
          if (i == 0   ) im = i; else im = i-1; Q[1] = _xEnthlocal[j][im][phi_slot];
          if (i == Nx-1) im = i; else im = i+1; Q[2] = _xEnthlocal[j][im][phi_slot];
        } else {
          im = i-1; Q[1] = _xEnthlocal[j][im][phi_slot];
          im = i+1; Q[2] = _xEnthlocal[j][im][phi_slot];
        }
        if (j == 0   ) jm = j; else jm = j-1; Q[3] = _xEnthlocal[jm][i][phi_slot];
        if (j == Nz-1) jm = j; else jm = j+1; Q[4] = _xEnthlocal[jm][i][phi_slot];
        
        // porosity on edges
        phi[0] = (Q[1]+Q[0])*0.5;
        phi[1] = (Q[2]+Q[0])*0.5;
        phi[2] = (Q[3]+Q[0])*0.5;
        phi[3] = (Q[4]+Q[0])*0.5;
        
        // fluid and bulk velocities
        for (ii = 0; ii < 4; ii++) {
          K      = Permeability(phi[ii],usr->par->phi_max,usr->par->n);
          Bf     = FluidBuoyancy(0.0,0.0,usr->nd->alpha_s,usr->nd->beta_s);
          vf[ii] = FluidVelocity(vs[ii],phi[ii],gradP[ii],gradPc[ii],Bf,K,k_hat[ii]);
          v [ii] = BulkVelocity(vs[ii],vf[ii],phi[ii]);
        }
        
        for (ii = 0; ii < 4; ii++) {
          // C1 = -1/PeT*exp(-Az)
          c[j][i][c1_slot[ii]] = -1.0/usr->nd->PeT*exp(-Az[ii]);
          
          // C2 = -1/PeC
          c[j][i][c2_slot[ii]] = -1.0/usr->nd->PeC;
          
          // v, vf, vs
          c[j][i][v_slot[ii] ] = v[ii];
          c[j][i][vf_slot[ii]] = vf[ii];
          c[j][i][vs_slot[ii]] = vs[ii];
        }
      }
    }
  }
  
  // Restore arrays, local vectors
  ierr = DMStagRestoreProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArray(dmcoeff,coefflocal,&c);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  
  ierr = VecDestroy(&coefflocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&xlocal); CHKERRQ(ierr);
  ierr = DMStagVecRestoreArrayRead(usr->dmPV,xPVlocal,&_xPVlocal);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmPV, &xPVlocal); CHKERRQ(ierr);
  ierr = DMStagVecRestoreArrayRead(dmEnth,xEnthlocal,&_xEnthlocal);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmEnth, &xEnthlocal); CHKERRQ(ierr);
  ierr = VecDestroy(&xEnth);CHKERRQ(ierr);
  ierr = DMDestroy(&dmEnth);CHKERRQ(ierr);

  PetscTime(&tlog[1]);
  if (usr->par->log_info) {
    printf("  FormCoefficient_HC: total             %1.2e\n",tlog[1]-tlog[0]);
  }
  
  PetscFunctionReturn(0);
}

// ---------------------------------------
// Phase Diagram  * relationships from Katz (2008,2010)
// enthalpy_method(H,C,P,&T,&phi,CF,CS,ncomp,user);
// ---------------------------------------
EnthEvalErrorCode Form_Enthalpy(PetscScalar H,PetscScalar C[],PetscScalar P,PetscScalar *_T,PetscScalar *_phi,PetscScalar *CF,PetscScalar *CS,PetscInt ncomp, void *ctx) 
{
  UsrData      *usr = (UsrData*) ctx;
  PetscInt     ii;
  PetscScalar  Tsol, Tliq, Hsol, Hliq, phi, T, Cs, Cf, Ci;
  PetscScalar  G, RM, S;
  PetscErrorCode ierr;

  G  = usr->nd->G;
  RM = usr->nd->RM;
  S  = usr->nd->S;
  Ci = C[0];

  Tsol = Solidus (Ci,P,G,PETSC_FALSE);
  Tliq = Liquidus(Ci,P,G,RM,PETSC_FALSE);
  Hsol = TotalEnthalpy(Tsol,0.0,S);
  Hliq = TotalEnthalpy(Tliq,1.0,S);

  if (H<Hsol) {
    phi = 0.0;
    T   = H;
    Cs  = Ci;
    Cf  = Liquidus(Solidus(Ci,P,G,PETSC_FALSE),P,G,RM,PETSC_TRUE);
  } else if ((H>=Hsol) && (H<Hliq)) {
    ierr   = Porosity(H,Ci,P,&phi,S,G,RM);CHKERRQ(ierr);
    T  = H - phi*S;
    Cs = Solidus (T,P,G,PETSC_TRUE);
    Cf = Liquidus(T,P,G,RM,PETSC_TRUE);
  } else {
    phi = 1.0;
    T  = H - S;
    Cs = Solidus(Liquidus(Ci,P,G,RM,PETSC_FALSE),P,G,PETSC_TRUE);
    Cf = Ci;
  }

  // assign pointers
  *_T = T;
  *_phi = phi;
  CS[0] = Cs;
  CS[1] = 1.0 - Cs;
  CF[0] = Cf;
  CF[1] = 1.0 - Cf;

  // error checking
  ENTH_CHECK_PHI(phi);
  return(STATE_VALID);
}

// ---------------------------------------
// Potential Temperature
// ---------------------------------------
PetscErrorCode Form_PotentialTemperature(PetscScalar T,PetscScalar P,PetscScalar *_TP, void *ctx) 
{
  UsrData      *usr = (UsrData*) ctx;
  PetscScalar  TP, rho, Az;
  PetscFunctionBegin;

  rho  = usr->par->rho0; // bulk density
  Az   = -usr->nd->A*P*usr->par->drho/rho;
  TP = (T+usr->nd->thetaS)*exp(Az) - usr->nd->thetaS;
  *_TP = TP;

  PetscFunctionReturn(0);
}