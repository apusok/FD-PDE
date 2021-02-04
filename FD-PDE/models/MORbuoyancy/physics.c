#include "MORbuoyancy.h"

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
  DM             dmHC;
  Vec            coefflocal, xHClocal, xphiTlocal;
  PetscScalar    **coordx,**coordz, k_hat[4];
  PetscInt       iprev, inext, icenter;
  PetscScalar    ***c;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  nd  = usr->nd;
  par = usr->par;
  scal= usr->scal;

  k_hat[0] = 0.0;
  k_hat[1] = 0.0;
  k_hat[2] = usr->par->k_hat;
  k_hat[3] = usr->par->k_hat;

  // Get dm and solution vector for H, C and enthalpy variables (phi, T)
  ierr = DMGetLocalVector(usr->dmHC, &xHClocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmHC, usr->xHC, INSERT_VALUES, xHClocal); CHKERRQ(ierr);
  ierr = DMGetLocalVector(usr->dmHC, &xphiTlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmHC, usr->xphiT, INSERT_VALUES, xphiTlocal); CHKERRQ(ierr);

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
      PetscInt idx;

      { // A = delta^2*eta (center, c=1)
        DMStagStencil point;
        PetscScalar   eta, T, phi;
        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 0;
        
        point.c = 0; ierr = DMStagVecGetValuesStencil(usr->dmHC,xphiTlocal,1,&point,&phi); CHKERRQ(ierr);
        point.c = 1; ierr = DMStagVecGetValuesStencil(usr->dmHC,xphiTlocal,1,&point,&T); CHKERRQ(ierr);
        eta = ShearViscosity(T*par->DT+par->T0,phi,par->EoR,par->Teta0,par->lambda,scal->eta,par->eta_min,par->eta_max);
        
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = nd->delta*nd->delta*eta;
      }

      { // A = delta^2*eta (corner, c=0)
        DMStagStencil point[4], pointQ[9];
        PetscInt      ii;
        PetscScalar   eta, T[4], phi[4], phic[9], Tc[9];

        point[0].i = i; point[0].j = j; point[0].loc = DOWN_LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = DOWN_RIGHT; point[1].c = 0;
        point[2].i = i; point[2].j = j; point[2].loc = UP_LEFT;    point[2].c = 0;
        point[3].i = i; point[3].j = j; point[3].loc = UP_RIGHT;   point[3].c = 0;

        // porosity and temperature
        pointQ[0].i = i-1; pointQ[0].j = j-1; pointQ[0].loc = ELEMENT; pointQ[0].c = 0;
        pointQ[1].i = i  ; pointQ[1].j = j-1; pointQ[1].loc = ELEMENT; pointQ[1].c = 0;
        pointQ[2].i = i+1; pointQ[2].j = j-1; pointQ[2].loc = ELEMENT; pointQ[2].c = 0;
        pointQ[3].i = i-1; pointQ[3].j = j  ; pointQ[3].loc = ELEMENT; pointQ[3].c = 0;
        pointQ[4].i = i  ; pointQ[4].j = j  ; pointQ[4].loc = ELEMENT; pointQ[4].c = 0;
        pointQ[5].i = i+1; pointQ[5].j = j  ; pointQ[5].loc = ELEMENT; pointQ[5].c = 0;
        pointQ[6].i = i-1; pointQ[6].j = j+1; pointQ[6].loc = ELEMENT; pointQ[6].c = 0;
        pointQ[7].i = i  ; pointQ[7].j = j+1; pointQ[7].loc = ELEMENT; pointQ[7].c = 0;
        pointQ[8].i = i+1; pointQ[8].j = j+1; pointQ[8].loc = ELEMENT; pointQ[8].c = 0;

        if (i == 0   ) { pointQ[0] = pointQ[4]; pointQ[3] = pointQ[4]; pointQ[6] = pointQ[4]; }
        if (i == Nx-1) { pointQ[2] = pointQ[4]; pointQ[5] = pointQ[4]; pointQ[8] = pointQ[4]; }
        if (j == 0   ) { pointQ[0] = pointQ[4]; pointQ[1] = pointQ[4]; pointQ[2] = pointQ[4]; }
        if (j == Nz-1) { pointQ[6] = pointQ[4]; pointQ[7] = pointQ[4]; pointQ[8] = pointQ[4]; }

        ierr = DMStagVecGetValuesStencil(usr->dmHC,xphiTlocal,9,pointQ,phic); CHKERRQ(ierr);
        for (ii = 0; ii < 9; ii++) pointQ[ii].c = 1;
        ierr = DMStagVecGetValuesStencil(usr->dmHC,xphiTlocal,9,pointQ,Tc); CHKERRQ(ierr);

        // porosity and T on edges - should do 2d bilinear interp
        phi[0] = (phic[0]+phic[1]+phic[3]+phic[4])*0.25; 
        phi[1] = (phic[1]+phic[2]+phic[4]+phic[5])*0.25; 
        phi[2] = (phic[3]+phic[4]+phic[6]+phic[7])*0.25; 
        phi[3] = (phic[4]+phic[5]+phic[7]+phic[8])*0.25;

        T[0] = (Tc[0]+Tc[1]+Tc[3]+Tc[4])*0.25; 
        T[1] = (Tc[1]+Tc[2]+Tc[4]+Tc[5])*0.25; 
        T[2] = (Tc[3]+Tc[4]+Tc[6]+Tc[7])*0.25; 
        T[3] = (Tc[4]+Tc[5]+Tc[7]+Tc[8])*0.25;

        for (ii = 0; ii < 4; ii++) {
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          eta = ShearViscosity(T[ii]*par->DT+par->T0,phi[ii],par->EoR,par->Teta0,par->lambda,scal->eta,par->eta_min,par->eta_max);
          c[j][i][idx] = nd->delta*nd->delta*eta;
        }
      }

      { // B = (phi+B)*k_hat (edges, c=0)
        DMStagStencil point[4],pointQ[3];
        PetscScalar   rhs[4], phi[3], T[3], C[3], Bphi = 0.0;
        PetscScalar   phi_in, T_in, C_in;
        PetscInt      ii;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 0;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = 0;
        point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = 0;

        // Bx = 0
        rhs[0] = 0.0;
        rhs[1] = 0.0;

        // Bz = Bphi*k_hat, Bphi = phi+B (depending on buoyancy type)
        pointQ[0].i = i; pointQ[0].j = j-1; pointQ[0].loc = ELEMENT; pointQ[0].c = 0;
        pointQ[1].i = i; pointQ[1].j = j  ; pointQ[1].loc = ELEMENT; pointQ[1].c = 0;
        pointQ[2].i = i; pointQ[2].j = j+1; pointQ[2].loc = ELEMENT; pointQ[2].c = 0;

        if (j == 0   ) pointQ[0] = pointQ[1];
        if (j == Nz-1) pointQ[2] = pointQ[1]; 

        ierr = DMStagVecGetValuesStencil(usr->dmHC,xphiTlocal,3,pointQ,phi); CHKERRQ(ierr);
        for (ii = 0; ii < 3; ii++) pointQ[ii].c = 1;
        ierr = DMStagVecGetValuesStencil(usr->dmHC,xphiTlocal,3,pointQ,T); CHKERRQ(ierr);
        ierr = DMStagVecGetValuesStencil(usr->dmHC,xHClocal,3,pointQ,C); CHKERRQ(ierr);

        phi_in = (phi[0]+phi[1])*0.5;
        T_in   = (T[0]  +T[1]  )*0.5;
        C_in   = (C[0]  +C[1]  )*0.5;
        Bphi   = Buoyancy(phi_in,T_in,C_in,nd->alpha_s,nd->beta_s,par->buoyancy);
        rhs[2] = par->k_hat*Bphi;

        phi_in = (phi[2]+phi[1])*0.5;
        T_in   = (T[2]  +T[1]  )*0.5;
        C_in   = (C[2]  +C[1]  )*0.5;
        Bphi   = Buoyancy(phi_in,T_in,C_in,nd->alpha_s,nd->beta_s,par->buoyancy);
        rhs[3] = par->k_hat*Bphi;

        for (ii = 0; ii < 4; ii++) {
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = rhs[ii];
        }
      }

      { // C = 0.0 (center, c=0)
        DMStagStencil point;
        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 0;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = 0.0;
      }

      { // D1 = delta^2*xi, xi=zeta-2/3eta (center, c=2)
        DMStagStencil point;
        PetscScalar   eta, zeta, T, phi, xi;

        point.i = i; point.j = j; point.loc = ELEMENT;
        point.c = 0; ierr = DMStagVecGetValuesStencil(usr->dmHC,xphiTlocal,1,&point,&phi); CHKERRQ(ierr);
        point.c = 1; ierr = DMStagVecGetValuesStencil(usr->dmHC,xphiTlocal,1,&point,&T); CHKERRQ(ierr);
        eta  = ShearViscosity(T*par->DT+par->T0,phi,par->EoR,par->Teta0,par->lambda,scal->eta,par->eta_min,par->eta_max);
        zeta = BulkViscosity(T*par->DT+par->T0,phi,par->EoR,par->Teta0,nd->visc_ratio,par->zetaExp,scal->eta,par->eta_min,par->eta_max);
        // PetscPrintf(PETSC_COMM_WORLD,"[%d %d] Bulk viscosity zeta = %f \n",i,j,zeta);
        // if (phi < 1e-12) xi = 0.0;
        // else             xi = zeta-2.0/3.0*eta;
        xi = zeta-2.0/3.0*eta;

        point.c = 2; ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = nd->delta*nd->delta*xi;
      }

      { // D2 = -K, K = (phi/phi0)^n (edges, c=1)
        // D3 = -K*(1+Bf)*k_hat, K = (phi/phi0)^n, Bf = 0 (edges, c=2)
        DMStagStencil point[4], pointQ[5];
        PetscScalar   xp[4],zp[4], Q[5], K[4], D2[4], D3[4], Bf[4], phi[4];
        PetscInt      ii;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 1;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 1;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = 1;
        point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = 1;

        // porosity
        pointQ[0].i = i  ; pointQ[0].j = j  ; pointQ[0].loc = ELEMENT; pointQ[0].c = 0;
        pointQ[1].i = i-1; pointQ[1].j = j  ; pointQ[1].loc = ELEMENT; pointQ[1].c = 0;
        pointQ[2].i = i+1; pointQ[2].j = j  ; pointQ[2].loc = ELEMENT; pointQ[2].c = 0;
        pointQ[3].i = i  ; pointQ[3].j = j-1; pointQ[3].loc = ELEMENT; pointQ[3].c = 0;
        pointQ[4].i = i  ; pointQ[4].j = j+1; pointQ[4].loc = ELEMENT; pointQ[4].c = 0;

        if (i == 0   ) pointQ[1] = point[0];
        if (i == Nx-1) pointQ[2] = point[0];
        if (j == 0   ) pointQ[3] = point[0];
        if (j == Nz-1) pointQ[4] = point[0];
        ierr = DMStagVecGetValuesStencil(usr->dmHC,xphiTlocal,5,pointQ,Q); CHKERRQ(ierr);

        if (i == 0   ) { xp[0] = coordx[i][icenter];} else { xp[0] = coordx[i-1][icenter];}
        if (i == Nx-1) { xp[2] = coordx[i][icenter];} else { xp[2] = coordx[i+1][icenter];}
        if (j == 0   ) { zp[0] = coordz[j][icenter];} else { zp[0] = coordz[j-1][icenter];}
        if (j == Nz-1) { zp[2] = coordz[j][icenter];} else { zp[2] = coordz[j+1][icenter];}
        xp[1] = coordx[i][icenter];
        zp[1] = coordz[j][icenter];

        // porosity on edges
        phi[0] = interp1DLin_3Points(coordx[i][iprev],xp[0],Q[1],xp[1],Q[0],xp[2],Q[2]); 
        phi[1] = interp1DLin_3Points(coordx[i][inext],xp[0],Q[1],xp[1],Q[0],xp[2],Q[2]); 
        phi[2] = interp1DLin_3Points(coordz[j][iprev],zp[0],Q[3],zp[1],Q[0],zp[2],Q[4]); 
        phi[3] = interp1DLin_3Points(coordz[j][inext],zp[0],Q[3],zp[1],Q[0],zp[2],Q[4]); 

        for (ii = 0; ii < 4; ii++) { 
          K[ii]  = Permeability(phi[ii],usr->par->phi0,usr->par->phi_max,usr->par->n);
          Bf[ii] = 0.0; // FluidBuoyancy(T,CF,usr->nd->alpha_s,usr->nd->beta_s);
          D2[ii] = -K[ii];
          D3[ii] = -K[ii]*(1+Bf[ii])*k_hat[ii];

          // D2 = -K, K = (phi/phi0)^n (edges, c=1)
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, 1, &idx); CHKERRQ(ierr);
          c[j][i][idx] = D2[ii];

          // D3 = -K*(1+Bf)*k_hat, K = (phi/phi0)^n, Bf = 0 (edges, c=2)
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, 2, &idx); CHKERRQ(ierr);
          c[j][i][idx] = D3[ii];
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

  ierr = DMRestoreLocalVector(usr->dmHC, &xHClocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmHC, &xphiTlocal); CHKERRQ(ierr);

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

        Az = -usr->nd->A*coordz[j][icenter]; 
        // A1 = exp(Az)
        point.i = i; point.j = j; point.loc = ELEMENT;  
        point.c = COEFF_A1; ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = exp(Az); 

        // B1 = -S
        point.c = COEFF_B1; ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = -usr->nd->S;

        // D1 = 0
        point.c = COEFF_D1; ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = 0.0;

        // A2 = 1
        point.c = COEFF_A2; ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = 1.0;

        // B2 = 1
        point.c = COEFF_B2; ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = 1.0;

        // D2 = 0.0
        point.c = COEFF_D2; ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
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
        
        Az[0] = -usr->nd->A*coordz[j][icenter];
        Az[1] = -usr->nd->A*coordz[j][icenter];
        Az[2] = -usr->nd->A*coordz[j][iprev];
        Az[3] = -usr->nd->A*coordz[j][inext]; 

        for (ii = 0; ii < 4; ii++) point[ii].c = 0;
        ierr = DMStagVecGetValuesStencil(usr->dmPV,xPVlocal,4,point,vs); CHKERRQ(ierr); // solid velocity
        ierr = DMStagVecGetValuesStencil(usr->dmVel,xVellocal,4,point,vf); CHKERRQ(ierr); // fluid velocity

        for (ii = 0; ii < 4; ii++) point[ii].c = 1;
        ierr = DMStagVecGetValuesStencil(usr->dmVel,xVellocal,4,point,v); CHKERRQ(ierr); // bulk velocity

        for (ii = 0; ii < 4; ii++) {
          // C1 = -1/PeT*exp(Az)
          point[ii].c = COEFF_C1; ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = -1.0/usr->nd->PeT*exp(Az[ii]);

          // C2 = -1/PeC
          point[ii].c = COEFF_C2; ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = -1.0/usr->nd->PeC;

          point[ii].c = COEFF_v; ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = v[ii];

          point[ii].c = COEFF_vf; ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = vf[ii];

          point[ii].c = COEFF_vs; ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
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

  // PetscPrintf(PETSC_COMM_WORLD,"# [Tsol = %f Tliq = %f] [H = %f Hsol = %f Hliq = %f] \n",Tsol,Tliq,H,Hsol,Hliq);

  if (H<Hsol) {
    // PetscPrintf(PETSC_COMM_WORLD,"# BELOW SOLIDUS \n");
    phi = 0.0;
    T   = H;
    Cs  = Ci;
    Cf  = Liquidus(Solidus(Ci,P,G,PETSC_FALSE),P,G,RM,PETSC_TRUE);
  } else if ((H>=Hsol) && (H<Hliq)) {
    // PetscPrintf(PETSC_COMM_WORLD,"# MUSH \n");
    ierr   = Porosity(H,Ci,P,&phi,S,G,RM);CHKERRQ(ierr);
    T  = H - phi*S;
    Cs = Solidus (T,P,G,PETSC_TRUE);
    Cf = Liquidus(T,P,G,RM,PETSC_TRUE);
  } else {
    // PetscPrintf(PETSC_COMM_WORLD,"# ABOVE LIQUIDUS \n");
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
  Az   = usr->nd->A*P*usr->par->drho/rho;
  TP = (T+usr->nd->thetaS)*exp(-Az) - usr->nd->thetaS;;
  *_TP = TP;

  PetscFunctionReturn(0);
}
