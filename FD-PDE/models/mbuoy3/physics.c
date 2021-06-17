#include "mbuoy3.h"

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
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz;
  DM             dmEnth;
  Vec            coefflocal, xEnthlocal;
  PetscScalar    **coordx,**coordz, k_hat[4];
  PetscInt       iprev, inext, icenter;
  PetscScalar    ***c;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

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
  
  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      DMStagStencil pointC[9];
      PetscScalar   phic[9], Tc[9], C[3], CF[3];
      PetscInt idx, ii;

      // get porosity, temperature, C, CF
      pointC[0].i = i-1; pointC[0].j = j-1; pointC[0].loc = ELEMENT; pointC[0].c = ENTH_ELEMENT_PHI;
      pointC[1].i = i  ; pointC[1].j = j-1; pointC[1].loc = ELEMENT; pointC[1].c = ENTH_ELEMENT_PHI;
      pointC[2].i = i+1; pointC[2].j = j-1; pointC[2].loc = ELEMENT; pointC[2].c = ENTH_ELEMENT_PHI;
      pointC[3].i = i-1; pointC[3].j = j  ; pointC[3].loc = ELEMENT; pointC[3].c = ENTH_ELEMENT_PHI;
      pointC[4].i = i  ; pointC[4].j = j  ; pointC[4].loc = ELEMENT; pointC[4].c = ENTH_ELEMENT_PHI;
      pointC[5].i = i+1; pointC[5].j = j  ; pointC[5].loc = ELEMENT; pointC[5].c = ENTH_ELEMENT_PHI;
      pointC[6].i = i-1; pointC[6].j = j+1; pointC[6].loc = ELEMENT; pointC[6].c = ENTH_ELEMENT_PHI;
      pointC[7].i = i  ; pointC[7].j = j+1; pointC[7].loc = ELEMENT; pointC[7].c = ENTH_ELEMENT_PHI;
      pointC[8].i = i+1; pointC[8].j = j+1; pointC[8].loc = ELEMENT; pointC[8].c = ENTH_ELEMENT_PHI;

      if (i == 0   ) { pointC[0] = pointC[4]; pointC[3] = pointC[4]; pointC[6] = pointC[4]; }
      if (i == Nx-1) { pointC[2] = pointC[4]; pointC[5] = pointC[4]; pointC[8] = pointC[4]; }
      if (j == 0   ) { pointC[0] = pointC[4]; pointC[1] = pointC[4]; pointC[2] = pointC[4]; }
      if (j == Nz-1) { pointC[6] = pointC[4]; pointC[7] = pointC[4]; pointC[8] = pointC[4]; }

      ierr = DMStagVecGetValuesStencil(dmEnth,xEnthlocal,9,pointC,phic); CHKERRQ(ierr);
      for (ii = 0; ii < 9; ii++) pointC[ii].c = ENTH_ELEMENT_T;
      ierr = DMStagVecGetValuesStencil(dmEnth,xEnthlocal,9,pointC,Tc); CHKERRQ(ierr);

      pointC[0].i = i; pointC[0].j = j-1; pointC[0].loc = ELEMENT; pointC[0].c = ENTH_ELEMENT_C;
      pointC[1].i = i; pointC[1].j = j  ; pointC[1].loc = ELEMENT; pointC[1].c = ENTH_ELEMENT_C;
      pointC[2].i = i; pointC[2].j = j+1; pointC[2].loc = ELEMENT; pointC[2].c = ENTH_ELEMENT_C;

      if (j == 0   ) pointC[0] = pointC[1];
      if (j == Nz-1) pointC[2] = pointC[1]; 

      for (ii = 0; ii < 3; ii++) pointC[ii].c = ENTH_ELEMENT_C;
      ierr = DMStagVecGetValuesStencil(dmEnth,xEnthlocal,3,pointC,C); CHKERRQ(ierr);
      for (ii = 0; ii < 3; ii++) pointC[ii].c = ENTH_ELEMENT_CF;
      ierr = DMStagVecGetValuesStencil(dmEnth,xEnthlocal,3,pointC,CF); CHKERRQ(ierr);

      { // A = delta^2*eta (center, c=1)
        DMStagStencil point;
        PetscScalar   eta;
      
        eta = ShearViscosity(Tc[4]*par->DT+par->T0,phic[4],par->EoR,par->Teta0,par->lambda,nd->eta_min,nd->eta_max,par->visc_shear);
        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = PVCOEFF_ELEMENT_A;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = nd->delta*nd->delta*eta;
      }

      { // A = delta^2*eta (corner, c=0)
        DMStagStencil point[4];
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

        point[0].i = i; point[0].j = j; point[0].loc = DOWN_LEFT;  point[0].c = PVCOEFF_VERTEX_A;
        point[1].i = i; point[1].j = j; point[1].loc = DOWN_RIGHT; point[1].c = PVCOEFF_VERTEX_A;
        point[2].i = i; point[2].j = j; point[2].loc = UP_LEFT;    point[2].c = PVCOEFF_VERTEX_A;
        point[3].i = i; point[3].j = j; point[3].loc = UP_RIGHT;   point[3].c = PVCOEFF_VERTEX_A;

        for (ii = 0; ii < 4; ii++) {
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          eta = ShearViscosity(T[ii]*par->DT+par->T0,phi[ii],par->EoR,par->Teta0,par->lambda,nd->eta_min,nd->eta_max,par->visc_shear);
          c[j][i][idx] = nd->delta*nd->delta*eta;
        }
      }

      { // B = (phi+B)*k_hat (edges, c=0)
        DMStagStencil point[4];
        PetscScalar   rhs[4], Bphi, BC, BT, Buoy = 0.0;
        PetscScalar   phi_in, T_in, C_in, CF_in;
        PetscInt      ii;

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

        // UP
        phi_in = (phic[4]+phic[7])*0.5;
        T_in   = (Tc[4]  +Tc[7]  )*0.5;
        C_in   = (C[2]   +C[1]   )*0.5;
        CF_in  = (CF[2]  +CF[1]  )*0.5;

        Bphi = Buoyancy_phi(phi_in,par->buoy_phi);
        BC   = Buoyancy_Composition(C_in,CF_in,phi_in,nd->beta_s,nd->beta_ls,par->buoy_C);
        BT   = Buoyancy_Temperature(T_in,phi_in,nd->alpha_s,nd->alpha_ls,par->buoy_T);
        Buoy = Bphi+BC+BT;
        rhs[3] = par->k_hat*Buoy;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = PVCOEFF_FACE_B;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = PVCOEFF_FACE_B;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = PVCOEFF_FACE_B;
        point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = PVCOEFF_FACE_B;

        for (ii = 0; ii < 4; ii++) {
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = rhs[ii];
        }
      }

      { // C = 0.0 (center, c=0)
        DMStagStencil point;
        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = PVCOEFF_ELEMENT_C;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = 0.0;
      }

      { // D1 = -1/(delta^2*xi), xi=zeta-2/3eta (center, c=2)
        DMStagStencil point;
        PetscScalar   xi, eta, zeta;

        point.i = i; point.j = j; point.loc = ELEMENT; point.c = PVCOEFF_ELEMENT_D1;
        eta  = ShearViscosity(Tc[4]*par->DT+par->T0,phic[4],par->EoR,par->Teta0,par->lambda,nd->eta_min,nd->eta_max,par->visc_shear);
        zeta = BulkViscosity(nd->visc_ratio,Tc[4]*par->DT+par->T0,phic[4],par->EoR,par->Teta0,par->phi_min,par->zetaExp,nd->eta_min,nd->eta_max,par->visc_bulk); 
        xi = zeta-2.0/3.0*eta;

        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = -1.0/(nd->delta*nd->delta*xi);
      }

      { // D2 = -K, K = (phi/phi0)^n (edges, c=1)
        // D3 = -K*(1+Bf)*k_hat, K = (phi/phi0)^n, Bf = 0 (edges, c=2)
        // D4 = -K, (edges, c=3)
        DMStagStencil point[4];
        PetscScalar   K[4], D2[4], D3[4], D4[4], Bf[4], phi[4];

        // porosity on edges
        phi[0] = (phic[3]+phic[4])*0.5;
        phi[1] = (phic[5]+phic[4])*0.5;
        phi[2] = (phic[1]+phic[4])*0.5;
        phi[3] = (phic[7]+phic[4])*0.5;

        if ((fabs(coordx[i][icenter])<=usr->nd->xmor) && (j==nz+sz-1)) { // dphi/dz=0 just beneath the axis
          phi[3] = usr->par->fextract*phi[2];
        }

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 1;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 1;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = 1;
        point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = 1;

        for (ii = 0; ii < 4; ii++) { 
          K[ii]  = Permeability(phi[ii],usr->par->phi0,usr->par->phi_max,usr->par->n);
          Bf[ii] = FluidBuoyancy(0.0,0.0,usr->nd->alpha_s,usr->nd->beta_s);
          D2[ii] = -K[ii];
          D3[ii] = -K[ii]*(1+Bf[ii])*k_hat[ii];
          D4[ii] = -K[ii];

          // D2 = -K, K = (phi/phi0)^n (edges, c=1)
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, PVCOEFF_FACE_D2, &idx); CHKERRQ(ierr);
          c[j][i][idx] = D2[ii];

          // D3 = -K*(1+Bf)*k_hat, K = (phi/phi0)^n, Bf = 0 (edges, c=2)
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, PVCOEFF_FACE_D3, &idx); CHKERRQ(ierr);
          c[j][i][idx] = D3[ii];

          // D4 = -K, (edges, c=3)
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, PVCOEFF_FACE_D4, &idx); CHKERRQ(ierr);
          c[j][i][idx] = D4[ii];
        }
      }

      { // DC =  0 (center, c=3)
        DMStagStencil point;
        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = SD3_COEFF_ELEMENT_DC;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = 0.0;
      }
    }
  }

  // Restore arrays, local vectors
  ierr = DMStagRestoreProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL);CHKERRQ(ierr);

  ierr = DMStagVecRestoreArray(dmcoeff,coefflocal,&c);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  ierr = VecDestroy(&coefflocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmEnth, &xEnthlocal); CHKERRQ(ierr);

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
  Vec            xPVlocal, xVellocal, xlocal, coefflocal;
  PetscScalar    ***c;
  PetscScalar    **coordx,**coordz;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  // PetscPrintf(PETSC_COMM_SELF,"# BREAK HC-COEFF START #\n");

  // get Stokes-Darcy velocities
  ierr = DMGetLocalVector(usr->dmPV, &xPVlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmPV, usr->xPV, INSERT_VALUES, xPVlocal); CHKERRQ(ierr);
  ierr = DMGetLocalVector(usr->dmVel, &xVellocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmVel, usr->xVel, INSERT_VALUES, xVellocal); CHKERRQ(ierr);

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

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {

      { // ELEMENT
        DMStagStencil point, pointE[5];
        PetscScalar   x[5],dz,Az;
        PetscInt      idx;

        Az = usr->nd->A*coordz[j][icenter]; 
        // A1 = exp(-Az)
        point.i = i; point.j = j; point.loc = ELEMENT;  
        point.c = HCCOEFF_ELEMENT_A1; ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = exp(-Az); 

        // B1 = -S
        point.c = HCCOEFF_ELEMENT_B1; ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = -usr->nd->S;

        // D1 = 0
        point.c = HCCOEFF_ELEMENT_D1; ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = 0.0;

        // A2 = 1
        point.c = HCCOEFF_ELEMENT_A2; ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = 1.0;

        // B2 = 1
        point.c = HCCOEFF_ELEMENT_B2; ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = 1.0;

        // D2 = 0.0
        point.c = HCCOEFF_ELEMENT_D2; ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = 0.0;
      }

      { // FACES
        DMStagStencil point[4];
        PetscInt      ii, idx;
        PetscScalar   Az[4], v[4],vf[4],vs[4];

        point[0].i = i; point[0].j = j; point[0].loc = LEFT; 
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN; 
        point[3].i = i; point[3].j = j; point[3].loc = UP; 
        
        Az[0] = usr->nd->A*coordz[j][icenter];
        Az[1] = usr->nd->A*coordz[j][icenter];
        Az[2] = usr->nd->A*coordz[j][iprev];
        Az[3] = usr->nd->A*coordz[j][inext]; 

        for (ii = 0; ii < 4; ii++) point[ii].c = 0;
        ierr = DMStagVecGetValuesStencil(usr->dmPV,xPVlocal,4,point,vs); CHKERRQ(ierr); // solid velocity
        ierr = DMStagVecGetValuesStencil(usr->dmVel,xVellocal,4,point,vf); CHKERRQ(ierr); // fluid velocity

        for (ii = 0; ii < 4; ii++) point[ii].c = 1;
        ierr = DMStagVecGetValuesStencil(usr->dmVel,xVellocal,4,point,v); CHKERRQ(ierr); // bulk velocity

        for (ii = 0; ii < 4; ii++) {
          // C1 = -1/PeT*exp(-Az)
          point[ii].c = HCCOEFF_FACE_C1; ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = -1.0/usr->nd->PeT*exp(-Az[ii]);

          // C2 = -1/PeC
          point[ii].c = HCCOEFF_FACE_C2; ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = -1.0/usr->nd->PeC;

          point[ii].c = HCCOEFF_FACE_V; ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = v[ii];

          point[ii].c = HCCOEFF_FACE_VF; ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = vf[ii];

          point[ii].c = HCCOEFF_FACE_VS; ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = vs[ii];
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
  ierr = DMRestoreLocalVector(usr->dmPV, &xPVlocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmVel, &xVellocal); CHKERRQ(ierr);

  // PetscPrintf(PETSC_COMM_SELF,"# BREAK HC-COEFF END #\n");

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormCoefficient_HC_VF_nonlinear"
PetscErrorCode AP_FormCoefficient_HC_VF_nonlinear(FDPDE fd, DM dm, Vec x, DM dmcoeff, Vec coeff, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  Params         *par;
  PetscInt       i, j, sx, sz, nx, nz, Nx,Nz,icenter,iprev,inext;
  Vec            xEnth, xEnthlocal, xPVlocal, xlocal, coefflocal;
  DM             dmEnth;
  PetscScalar    ***c;
  PetscScalar    **coordx,**coordz;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  
  // PetscPrintf(PETSC_COMM_SELF,"# BREAK HC-COEFF-VF-NL START #\n");
  
  // get Stokes-Darcy velocities
  ierr = DMGetLocalVector(usr->dmPV, &xPVlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmPV, usr->xPV, INSERT_VALUES, xPVlocal); CHKERRQ(ierr);
  
  // Update enthalpy variables - slow for each iteration
  ierr = FDPDEEnthalpyUpdateDiagnostics(fd,dm,x,&dmEnth,&xEnth); CHKERRQ(ierr);
  ierr = VecCopy(xEnth,usr->xEnth);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dmEnth,&xEnthlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dmEnth,xEnth,INSERT_VALUES,xEnthlocal); CHKERRQ(ierr);
  
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
  
  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      
      { // ELEMENT
        DMStagStencil point, pointE[5];
        PetscScalar   x[5],dz,Az;
        PetscInt      idx;
        
        Az = usr->nd->A*coordz[j][icenter];
        // A1 = exp(-Az)
        point.i = i; point.j = j; point.loc = ELEMENT;
        point.c = HCCOEFF_ELEMENT_A1; ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = exp(-Az);
        
        // B1 = -S
        point.c = HCCOEFF_ELEMENT_B1; ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = -usr->nd->S;
        
        // D1 = 0
        point.c = HCCOEFF_ELEMENT_D1; ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = 0.0;
        
        // A2 = 1
        point.c = HCCOEFF_ELEMENT_A2; ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = 1.0;
        
        // B2 = 1
        point.c = HCCOEFF_ELEMENT_B2; ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = 1.0;
        
        // D2 = 0.0
        point.c = HCCOEFF_ELEMENT_D2; ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = 0.0;
      }
      
      { // FACES
        DMStagStencil point[14],pointQ[5];
        PetscInt      ii, idx;
        PetscScalar   pv[14], Az[4],v[4],vf[4],vs[4],phi[4],Bf,K,gradP[4],gradPc[4],k_hat[4],Q[5],dx,dz;
        
        point[0].i = i; point[0].j = j; point[0].loc = LEFT;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;
        point[3].i = i; point[3].j = j; point[3].loc = UP;
        
        Az[0] = usr->nd->A*coordz[j][icenter];
        Az[1] = usr->nd->A*coordz[j][icenter];
        Az[2] = usr->nd->A*coordz[j][iprev];
        Az[3] = usr->nd->A*coordz[j][inext];
        
        k_hat[0] = 0.0;
        k_hat[1] = 0.0;
        k_hat[2] = usr->par->k_hat;
        k_hat[3] = usr->par->k_hat;
        
        // get PV data
        point[0].i = i; point[0].j = j; point[0].loc = LEFT;    point[0].c = PV_FACE_VS;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT;   point[1].c = PV_FACE_VS;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;    point[2].c = PV_FACE_VS;
        point[3].i = i; point[3].j = j; point[3].loc = UP;      point[3].c = PV_FACE_VS;
        
        point[4].i = i  ; point[4].j = j  ; point[4].loc = ELEMENT; point[4].c = PV_ELEMENT_P;
        point[5].i = i-1; point[5].j = j  ; point[5].loc = ELEMENT; point[5].c = PV_ELEMENT_P;
        point[6].i = i+1; point[6].j = j  ; point[6].loc = ELEMENT; point[6].c = PV_ELEMENT_P;
        point[7].i = i  ; point[7].j = j-1; point[7].loc = ELEMENT; point[7].c = PV_ELEMENT_P;
        point[8].i = i  ; point[8].j = j+1; point[8].loc = ELEMENT; point[8].c = PV_ELEMENT_P;
        
        point[9].i  = i  ; point[9].j  = j  ; point[9].loc  = ELEMENT; point[9].c  = PV_ELEMENT_PC;
        point[10].i = i-1; point[10].j = j  ; point[10].loc = ELEMENT; point[10].c = PV_ELEMENT_PC;
        point[11].i = i+1; point[11].j = j  ; point[11].loc = ELEMENT; point[11].c = PV_ELEMENT_PC;
        point[12].i = i  ; point[12].j = j-1; point[12].loc = ELEMENT; point[12].c = PV_ELEMENT_PC;
        point[13].i = i  ; point[13].j = j+1; point[13].loc = ELEMENT; point[13].c = PV_ELEMENT_PC;
        
        // correct for domain edges
        if (i == 0   ) { point[5] = point[4]; point[10] = point[9]; }
        if (i == Nx-1) { point[6] = point[4]; point[11] = point[9]; }
        if (j == 0   ) { point[7] = point[4]; point[12] = point[9]; }
        if (j == Nz-1) { point[8] = point[4]; point[13] = point[9]; }
        
        ierr = DMStagVecGetValuesStencil(usr->dmPV,xPVlocal,14,point,pv); CHKERRQ(ierr);
        
        // grid spacing - assume constant
        dx = coordx[i][inext]-coordx[i][iprev];
        dz = coordz[j][inext]-coordz[j][iprev];
        
        vs[0] = pv[0];
        vs[1] = pv[1];
        vs[2] = pv[2];
        vs[3] = pv[3];
        
        gradP[0] = (pv[4]-pv[5])/dx;
        gradP[1] = (pv[6]-pv[4])/dx;
        gradP[2] = (pv[4]-pv[7])/dz;
        gradP[3] = (pv[8]-pv[4])/dz;
        
        gradPc[0] = (pv[9]-pv[10])/dx;
        gradPc[1] = (pv[11]-pv[9])/dx;
        gradPc[2] = (pv[9]-pv[12])/dz;
        gradPc[3] = (pv[13]-pv[9])/dz;
        
        // porosity
        pointQ[0].i = i  ; pointQ[0].j = j  ; pointQ[0].loc = ELEMENT; pointQ[0].c = ENTH_ELEMENT_PHI;
        pointQ[1].i = i-1; pointQ[1].j = j  ; pointQ[1].loc = ELEMENT; pointQ[1].c = ENTH_ELEMENT_PHI;
        pointQ[2].i = i+1; pointQ[2].j = j  ; pointQ[2].loc = ELEMENT; pointQ[2].c = ENTH_ELEMENT_PHI;
        pointQ[3].i = i  ; pointQ[3].j = j-1; pointQ[3].loc = ELEMENT; pointQ[3].c = ENTH_ELEMENT_PHI;
        pointQ[4].i = i  ; pointQ[4].j = j+1; pointQ[4].loc = ELEMENT; pointQ[4].c = ENTH_ELEMENT_PHI;
        
        if (i == 0   ) pointQ[1] = pointQ[0];
        if (i == Nx-1) pointQ[2] = pointQ[0];
        if (j == 0   ) pointQ[3] = pointQ[0];
        if (j == Nz-1) pointQ[4] = pointQ[0];
        ierr = DMStagVecGetValuesStencil(dmEnth,xEnthlocal,5,pointQ,Q); CHKERRQ(ierr);
        
        // porosity on edges
        phi[0] = (Q[1]+Q[0])*0.5;
        phi[1] = (Q[2]+Q[0])*0.5;
        phi[2] = (Q[3]+Q[0])*0.5;
        phi[3] = (Q[4]+Q[0])*0.5;
        
        if ((fabs(coordx[i][icenter])<=usr->nd->xmor) && (j==nz+sz-1)) { // dphi/dz=0 just beneath the axis
          phi[3] = usr->par->fextract*phi[2]; // phi[3];
        }
        
        // fluid and bulk velocities
        for (ii = 0; ii < 4; ii++) {
          K      = Permeability(phi[ii],usr->par->phi0,usr->par->phi_max,usr->par->n);
          Bf     = FluidBuoyancy(0.0,0.0,usr->nd->alpha_s,usr->nd->beta_s);
          vf[ii] = FluidVelocity(vs[ii],phi[ii],gradP[ii],gradPc[ii],Bf,K,k_hat[ii]);
          v [ii] = BulkVelocity(vs[ii],vf[ii],phi[ii]);
        }
        
        if ((fabs(coordx[i][icenter])<=usr->nd->xmor) && (j==nz+sz-1)) {
          vf[3] = usr->par->fextract*vf[3];
          v [3] = BulkVelocity(vs[3],vf[3],phi[3]);
        }
        
        for (ii = 0; ii < 4; ii++) {
          // C1 = -1/PeT*exp(-Az)
          point[ii].c = HCCOEFF_FACE_C1; ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = -1.0/usr->nd->PeT*exp(-Az[ii]);
          
          // C2 = -1/PeC
          point[ii].c = HCCOEFF_FACE_C2; ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = -1.0/usr->nd->PeC;
          
          point[ii].c = HCCOEFF_FACE_V; ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = v[ii];
          
          point[ii].c = HCCOEFF_FACE_VF; ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = vf[ii];
          
          point[ii].c = HCCOEFF_FACE_VS; ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = vs[ii];
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
  ierr = DMRestoreLocalVector(usr->dmPV, &xPVlocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmEnth, &xEnthlocal); CHKERRQ(ierr);
  ierr = VecDestroy(&xEnth);CHKERRQ(ierr);
  ierr = DMDestroy(&dmEnth);CHKERRQ(ierr);
  
  // PetscPrintf(PETSC_COMM_SELF,"# BREAK HC-COEFF-VF-NL END #\n");
  
  PetscFunctionReturn(0);
}

// ---------------------------------------
// FormCoefficient_HC_VF_nonlinear - we update vf, v locally and depends on porosity (introduces extra-nonlinearity)
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormCoefficient_HC_VF_nonlinear"
PetscErrorCode FormCoefficient_HC_VF_nonlinear(FDPDE fd, DM dm, Vec x, DM dmcoeff, Vec coeff, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  Params         *par;
  PetscInt       i, j, sx, sz, nx, nz, Nx,Nz,icenter,iprev,inext;
  Vec            xEnth, xEnthlocal, xPVlocal, xlocal, coefflocal;
  DM             dmEnth;
  PetscScalar    ***c;
  PetscScalar    **coordx,**coordz;
  PetscErrorCode ierr;
  
  const PetscReal ***_xPVlocal;
  const PetscReal ***_xEnthlocal;
  PetscInt pv_slot[14];
  PetscInt coeff_element_slot[6];
  PetscInt coeff_face_slot_C1[4],coeff_face_slot_C2[4],coeff_face_slot_V[4],coeff_face_slot_VF[4],coeff_face_slot_VS[4];
  PetscInt enth_cell_slot_phi;
  
  PetscFunctionBeginUser;

  // PetscPrintf(PETSC_COMM_SELF,"# BREAK HC-COEFF-VF-NL START #\n");

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

  {
    PetscInt ii;
    DMStagStencil point[14];
    
    i = j = 0;

    point[0].i = i; point[0].j = j; point[0].loc = LEFT;    point[0].c = PV_FACE_VS;
    point[1].i = i; point[1].j = j; point[1].loc = RIGHT;   point[1].c = PV_FACE_VS;
    point[2].i = i; point[2].j = j; point[2].loc = DOWN;    point[2].c = PV_FACE_VS;
    point[3].i = i; point[3].j = j; point[3].loc = UP;      point[3].c = PV_FACE_VS;
    
    point[4].i = i  ; point[4].j = j  ; point[4].loc = ELEMENT; point[4].c = PV_ELEMENT_P;
    point[5].i = i-1; point[5].j = j  ; point[5].loc = ELEMENT; point[5].c = PV_ELEMENT_P;
    point[6].i = i+1; point[6].j = j  ; point[6].loc = ELEMENT; point[6].c = PV_ELEMENT_P;
    point[7].i = i  ; point[7].j = j-1; point[7].loc = ELEMENT; point[7].c = PV_ELEMENT_P;
    point[8].i = i  ; point[8].j = j+1; point[8].loc = ELEMENT; point[8].c = PV_ELEMENT_P;
    
    point[9].i  = i  ; point[9].j  = j  ; point[9].loc  = ELEMENT; point[9].c  = PV_ELEMENT_PC;
    point[10].i = i-1; point[10].j = j  ; point[10].loc = ELEMENT; point[10].c = PV_ELEMENT_PC;
    point[11].i = i+1; point[11].j = j  ; point[11].loc = ELEMENT; point[11].c = PV_ELEMENT_PC;
    point[12].i = i  ; point[12].j = j-1; point[12].loc = ELEMENT; point[12].c = PV_ELEMENT_PC;
    point[13].i = i  ; point[13].j = j+1; point[13].loc = ELEMENT; point[13].c = PV_ELEMENT_PC;
    
    for (ii=0; ii<14; ii++) {
      ierr = DMStagGetLocationSlot(usr->dmPV, point[ii].loc, point[ii].c, &pv_slot[ii]); CHKERRQ(ierr);
    }
  }

  {
    DMStagStencil point;
    PetscInt      idx;
    PetscInt ii;
    
    i = j = 0;

    point.i = i; point.j = j; point.loc = ELEMENT;
    
    // A1 = exp(-Az)
    point.c = HCCOEFF_ELEMENT_A1;
    ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
    coeff_element_slot[0] = idx;
    
    // B1 = -S
    point.c = HCCOEFF_ELEMENT_B1;
    ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
    coeff_element_slot[1] = idx;
    
    // D1 = 0
    point.c = HCCOEFF_ELEMENT_D1;
    ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
    coeff_element_slot[2] = idx;
    
    // A2 = 1
    point.c = HCCOEFF_ELEMENT_A2;
    ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
    coeff_element_slot[3] = idx;
    
    // B2 = 1
    point.c = HCCOEFF_ELEMENT_B2;
    ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
    coeff_element_slot[4] = idx;
    
    // D2 = 0.0
    point.c = HCCOEFF_ELEMENT_D2;
    ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
    coeff_element_slot[5] = idx;
  }
  
  {
    DMStagStencil point[4];
    PetscInt      ii, idx;

    i = j = 0;
    
    point[0].i = i; point[0].j = j; point[0].loc = LEFT;
    point[1].i = i; point[1].j = j; point[1].loc = RIGHT;
    point[2].i = i; point[2].j = j; point[2].loc = DOWN;
    point[3].i = i; point[3].j = j; point[3].loc = UP;

    
    for (ii=0; ii<4; ii++) {
      // C1 = -1/PeT*exp(-Az)
      point[ii].c = HCCOEFF_FACE_C1;
      ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
      coeff_face_slot_C1[ii] = idx;
      
      // C2 = -1/PeC
      point[ii].c = HCCOEFF_FACE_C2;
      ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
      coeff_face_slot_C2[ii] = idx;
      
      point[ii].c = HCCOEFF_FACE_V;
      ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
      coeff_face_slot_V[ii] = idx;
      
      point[ii].c = HCCOEFF_FACE_VF;
      ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
      coeff_face_slot_VF[ii] = idx;
      
      point[ii].c = HCCOEFF_FACE_VS;
      ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
      coeff_face_slot_VS[ii] = idx;
    }
  }
  
  {
    DMStagStencil pointQ[1];
    PetscInt      ii, idx;
    
    i = j = 0;
    
    pointQ[0].i = i  ; pointQ[0].j = j  ; pointQ[0].loc = ELEMENT; pointQ[0].c = ENTH_ELEMENT_PHI;
    
    ierr = DMStagGetLocationSlot(dmEnth, pointQ[0].loc, pointQ[0].c, &idx); CHKERRQ(ierr);
    enth_cell_slot_phi = idx;
  }
  
  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {

      { // ELEMENT
        DMStagStencil point, pointE[5];
        PetscScalar   x[5],dz,Az;
        PetscInt      idx;

        Az = usr->nd->A*coordz[j][icenter]; 
        // A1 = exp(-Az)
        point.i = i; point.j = j; point.loc = ELEMENT;  
        point.c = HCCOEFF_ELEMENT_A1;
        //ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        idx = coeff_element_slot[0];
        c[j][i][idx] = exp(-Az); 

        // B1 = -S
        point.c = HCCOEFF_ELEMENT_B1;
        //ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        idx = coeff_element_slot[1];
        c[j][i][idx] = -usr->nd->S;

        // D1 = 0
        point.c = HCCOEFF_ELEMENT_D1;
        //ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        idx = coeff_element_slot[2];
        c[j][i][idx] = 0.0;

        // A2 = 1
        point.c = HCCOEFF_ELEMENT_A2;
        //ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        idx = coeff_element_slot[3];
        c[j][i][idx] = 1.0;

        // B2 = 1
        point.c = HCCOEFF_ELEMENT_B2;
        //ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        idx = coeff_element_slot[4];
        c[j][i][idx] = 1.0;

        // D2 = 0.0
        point.c = HCCOEFF_ELEMENT_D2;
        //ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        idx = coeff_element_slot[5];
        c[j][i][idx] = 0.0;
      }

      { // FACES
        DMStagStencil point[14],pointQ[5];
        PetscInt      ii, idx;
        PetscScalar   pv[14], Az[4],v[4],vf[4],vs[4],phi[4],Bf,K,gradP[4],gradPc[4],k_hat[4],Q[5],dx,dz;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT; 
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN; 
        point[3].i = i; point[3].j = j; point[3].loc = UP; 
        
        Az[0] = usr->nd->A*coordz[j][icenter];
        Az[1] = usr->nd->A*coordz[j][icenter];
        Az[2] = usr->nd->A*coordz[j][iprev];
        Az[3] = usr->nd->A*coordz[j][inext]; 

        k_hat[0] = 0.0;
        k_hat[1] = 0.0;
        k_hat[2] = usr->par->k_hat;
        k_hat[3] = usr->par->k_hat;

        // get PV data
        point[0].i = i; point[0].j = j; point[0].loc = LEFT;    point[0].c = PV_FACE_VS;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT;   point[1].c = PV_FACE_VS;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;    point[2].c = PV_FACE_VS;
        point[3].i = i; point[3].j = j; point[3].loc = UP;      point[3].c = PV_FACE_VS;

        point[4].i = i  ; point[4].j = j  ; point[4].loc = ELEMENT; point[4].c = PV_ELEMENT_P;
        point[5].i = i-1; point[5].j = j  ; point[5].loc = ELEMENT; point[5].c = PV_ELEMENT_P;
        point[6].i = i+1; point[6].j = j  ; point[6].loc = ELEMENT; point[6].c = PV_ELEMENT_P;
        point[7].i = i  ; point[7].j = j-1; point[7].loc = ELEMENT; point[7].c = PV_ELEMENT_P;
        point[8].i = i  ; point[8].j = j+1; point[8].loc = ELEMENT; point[8].c = PV_ELEMENT_P;

        point[9].i  = i  ; point[9].j  = j  ; point[9].loc  = ELEMENT; point[9].c  = PV_ELEMENT_PC;
        point[10].i = i-1; point[10].j = j  ; point[10].loc = ELEMENT; point[10].c = PV_ELEMENT_PC;
        point[11].i = i+1; point[11].j = j  ; point[11].loc = ELEMENT; point[11].c = PV_ELEMENT_PC;
        point[12].i = i  ; point[12].j = j-1; point[12].loc = ELEMENT; point[12].c = PV_ELEMENT_PC;
        point[13].i = i  ; point[13].j = j+1; point[13].loc = ELEMENT; point[13].c = PV_ELEMENT_PC;

        // correct for domain edges 
        if (i == 0   ) { point[5] = point[4]; point[10] = point[9]; }
        if (i == Nx-1) { point[6] = point[4]; point[11] = point[9]; }
        if (j == 0   ) { point[7] = point[4]; point[12] = point[9]; }
        if (j == Nz-1) { point[8] = point[4]; point[13] = point[9]; }

        //ierr = DMStagVecGetValuesStencil(usr->dmPV,xPVlocal,14,point,pv); CHKERRQ(ierr);
        for (ii=0; ii<14; ii++) {
          pv[ii] = _xPVlocal[ point[ii].j ][ point[ii].i ][ pv_slot[ii] ];
        }
        
        
        // grid spacing - assume constant
        dx = coordx[i][inext]-coordx[i][iprev];
        dz = coordz[j][inext]-coordz[j][iprev];

        vs[0] = pv[0]; 
        vs[1] = pv[1]; 
        vs[2] = pv[2]; 
        vs[3] = pv[3]; 

        gradP[0] = (pv[4]-pv[5])/dx;
        gradP[1] = (pv[6]-pv[4])/dx;
        gradP[2] = (pv[4]-pv[7])/dz;
        gradP[3] = (pv[8]-pv[4])/dz;

        gradPc[0] = (pv[9]-pv[10])/dx;
        gradPc[1] = (pv[11]-pv[9])/dx;
        gradPc[2] = (pv[9]-pv[12])/dz;
        gradPc[3] = (pv[13]-pv[9])/dz;

        // porosity
        pointQ[0].i = i  ; pointQ[0].j = j  ; pointQ[0].loc = ELEMENT; pointQ[0].c = ENTH_ELEMENT_PHI;
        pointQ[1].i = i-1; pointQ[1].j = j  ; pointQ[1].loc = ELEMENT; pointQ[1].c = ENTH_ELEMENT_PHI;
        pointQ[2].i = i+1; pointQ[2].j = j  ; pointQ[2].loc = ELEMENT; pointQ[2].c = ENTH_ELEMENT_PHI;
        pointQ[3].i = i  ; pointQ[3].j = j-1; pointQ[3].loc = ELEMENT; pointQ[3].c = ENTH_ELEMENT_PHI;
        pointQ[4].i = i  ; pointQ[4].j = j+1; pointQ[4].loc = ELEMENT; pointQ[4].c = ENTH_ELEMENT_PHI;

        if (i == 0   ) pointQ[1] = pointQ[0];
        if (i == Nx-1) pointQ[2] = pointQ[0];
        if (j == 0   ) pointQ[3] = pointQ[0];
        if (j == Nz-1) pointQ[4] = pointQ[0];
        
        //ierr = DMStagVecGetValuesStencil(dmEnth,xEnthlocal,5,pointQ,Q); CHKERRQ(ierr);
        for (ii=0; ii<5; ii++) {
          Q[ii] = _xEnthlocal[ pointQ[ii].j ][ pointQ[ii].i ][ enth_cell_slot_phi ];
        }
        
        // porosity on edges
        phi[0] = (Q[1]+Q[0])*0.5; 
        phi[1] = (Q[2]+Q[0])*0.5; 
        phi[2] = (Q[3]+Q[0])*0.5; 
        phi[3] = (Q[4]+Q[0])*0.5; 

        if ((fabs(coordx[i][icenter])<=usr->nd->xmor) && (j==nz+sz-1)) { // dphi/dz=0 just beneath the axis
          phi[3] = usr->par->fextract*phi[2]; // phi[3];
        }

        // fluid and bulk velocities
        for (ii = 0; ii < 4; ii++) {
          K      = Permeability(phi[ii],usr->par->phi0,usr->par->phi_max,usr->par->n);
          Bf     = FluidBuoyancy(0.0,0.0,usr->nd->alpha_s,usr->nd->beta_s);
          vf[ii] = FluidVelocity(vs[ii],phi[ii],gradP[ii],gradPc[ii],Bf,K,k_hat[ii]); 
          v [ii] = BulkVelocity(vs[ii],vf[ii],phi[ii]);
        }

        if ((fabs(coordx[i][icenter])<=usr->nd->xmor) && (j==nz+sz-1)) {
          vf[3] = usr->par->fextract*vf[3];
          v [3] = BulkVelocity(vs[3],vf[3],phi[3]);
        }

        for (ii = 0; ii < 4; ii++) {
          // C1 = -1/PeT*exp(-Az)
          point[ii].c = HCCOEFF_FACE_C1;
          //ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          idx = coeff_face_slot_C1[ii];
          c[j][i][idx] = -1.0/usr->nd->PeT*exp(-Az[ii]);

          // C2 = -1/PeC
          point[ii].c = HCCOEFF_FACE_C2;
          //ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          idx = coeff_face_slot_C2[ii];
          c[j][i][idx] = -1.0/usr->nd->PeC;

          point[ii].c = HCCOEFF_FACE_V;
          //ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          idx = coeff_face_slot_V[ii];
          c[j][i][idx] = v[ii];

          point[ii].c = HCCOEFF_FACE_VF;
          //ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          idx = coeff_face_slot_VF[ii];
          c[j][i][idx] = vf[ii];

          point[ii].c = HCCOEFF_FACE_VS;
          //ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          idx = coeff_face_slot_VS[ii];
          c[j][i][idx] = vs[ii];
        }
      }
    }
  }

  ierr = DMStagVecRestoreArrayRead(dmEnth,xEnthlocal,&_xEnthlocal);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArrayRead(usr->dmPV,xPVlocal,&_xPVlocal);CHKERRQ(ierr);

  
  // Restore arrays, local vectors
  ierr = DMStagRestoreProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArray(dmcoeff,coefflocal,&c);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  
  ierr = VecDestroy(&coefflocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&xlocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmPV, &xPVlocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmEnth, &xEnthlocal); CHKERRQ(ierr);
  ierr = VecDestroy(&xEnth);CHKERRQ(ierr);
  ierr = DMDestroy(&dmEnth);CHKERRQ(ierr);

  // PetscPrintf(PETSC_COMM_SELF,"# BREAK HC-COEFF-VF-NL END #\n");

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
  // TP = (T+usr->nd->thetaS-T_KELVIN/usr->par->DT)*exp(Az) - usr->nd->thetaS+T_KELVIN/usr->par->DT;
  TP = (T+usr->nd->thetaS)*exp(Az) - usr->nd->thetaS;
  *_TP = TP;

  PetscFunctionReturn(0);
}
