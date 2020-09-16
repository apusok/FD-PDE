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
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz;
  DM             dmHC;
  Vec            coefflocal;
  // Vec            xphi, xphilocal;
  PetscScalar    **coordx,**coordz;
  PetscInt       iprev, inext, icenter;
  PetscScalar    ***c;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  nd  = usr->nd;
  par = usr->par;

  // Get dm and solution vector for phi, theta (T), C - is it better to use fields?
  dmHC = usr->dmHC;
  // xphi  = usr->xphiprev;

  // ierr = DMCreateLocalVector(dmHC,&xphilocal);CHKERRQ(ierr);
  // ierr = DMGlobalToLocalBegin(dmHC,xphi,INSERT_VALUES,xphilocal);CHKERRQ(ierr);
  // ierr = DMGlobalToLocalEnd(dmHC,xphi,INSERT_VALUES,xphilocal);CHKERRQ(ierr);

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
        PetscScalar   eta;
        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 1;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        // eta = ShearViscosity();
        c[j][i][idx] = nd->delta*nd->delta*eta;
      }

      { // A = delta^2*eta (corner, c=0)
        DMStagStencil point[4];
        PetscInt      ii;
        PetscScalar   eta;

        point[0].i = i; point[0].j = j; point[0].loc = DOWN_LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = DOWN_RIGHT; point[1].c = 0;
        point[2].i = i; point[2].j = j; point[2].loc = UP_LEFT;    point[2].c = 0;
        point[3].i = i; point[3].j = j; point[3].loc = UP_RIGHT;   point[3].c = 0;

        for (ii = 0; ii < 4; ii++) {
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          // eta = ShearViscosity();
          c[j][i][idx] = nd->delta*nd->delta*eta;
        }
      }

      { // B = (phi+B)*k_hat (edges, c=0)
        DMStagStencil point[4], pointQ[3];
        PetscScalar   Q[3], Qinterp, B = 0.0;
        PetscScalar   zp[4],rhs[4],zQ[3];
        PetscInt      ii;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 0;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = 0;
        point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = 0;

        zp[0] = coordz[j][icenter];
        zp[1] = coordz[j][icenter];
        zp[2] = coordz[j][iprev  ];
        zp[3] = coordz[j][inext  ];

        // Bx = 0
        rhs[0] = 0.0;
        rhs[1] = 0.0;

        // Bz = (phi+B)*k_hat
        pointQ[0].i = i; pointQ[0].j = j-1; pointQ[0].loc = ELEMENT; pointQ[0].c = 0;
        pointQ[1].i = i; pointQ[1].j = j  ; pointQ[1].loc = ELEMENT; pointQ[1].c = 0;
        pointQ[2].i = i; pointQ[2].j = j+1; pointQ[2].loc = ELEMENT; pointQ[2].c = 0;

        zQ[1] = coordz[j][icenter];
        if (j == 0   ) { pointQ[0] = pointQ[1]; zQ[0] = zQ[1];}
        else           {zQ[0] = coordz[j-1][icenter];}
        if (j == Nz-1) { pointQ[2] = pointQ[1]; zQ[2] = zQ[1];}
        else           {zQ[2] = coordz[j+1][icenter];}

        // ierr = DMStagVecGetValuesStencil(dmHC,xphilocal,3,pointQ,Q); CHKERRQ(ierr);
        Qinterp = interp1DLin_3Points(zp[2],zQ[0],Q[0],zQ[1],Q[1],zQ[2],Q[2]);
        // B = UpdateSolidBuoyancy();
        rhs[2]  = par->k_hat*(Qinterp+B);

        Qinterp = interp1DLin_3Points(zp[3],zQ[0],Q[0],zQ[1],Q[1],zQ[2],Q[2]);
        // B = UpdateSolidBuoyancy();
        rhs[3]  = par->k_hat*(Qinterp+B);

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
        PetscScalar   eta, zeta;
        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 2;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        // eta  = ShearViscosity();
        // zeta = BulkViscosity();
        c[j][i][idx] = nd->delta*nd->delta*(zeta-2.0/3.0*eta);
      }

      { // D2 = -K, K = (phi/phi0)^n (edges, c=1)
        DMStagStencil point[4], pointQ[5];
        PetscScalar   xp[4],zp[4], xQ[3], zQ[3], Q[5], Qinterp, rhs[4];
        PetscInt      ii;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 1;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 1;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = 1;
        point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = 1;

        xp[0] = coordx[i][iprev  ]; zp[0] = coordz[j][icenter];
        xp[1] = coordx[i][inext  ]; zp[1] = coordz[j][icenter];
        xp[2] = coordx[i][icenter]; zp[2] = coordz[j][iprev  ];
        xp[3] = coordx[i][icenter]; zp[3] = coordz[j][inext  ];

        // get porosity - take into account domain borders
        pointQ[0].i = i-1; pointQ[0].j = j  ; pointQ[0].loc = ELEMENT; pointQ[0].c = 0;
        pointQ[1].i = i  ; pointQ[1].j = j  ; pointQ[1].loc = ELEMENT; pointQ[1].c = 0;
        pointQ[2].i = i+1; pointQ[2].j = j  ; pointQ[2].loc = ELEMENT; pointQ[2].c = 0;
        pointQ[3].i = i  ; pointQ[3].j = j-1; pointQ[3].loc = ELEMENT; pointQ[3].c = 0;
        pointQ[4].i = i  ; pointQ[4].j = j+1; pointQ[4].loc = ELEMENT; pointQ[4].c = 0;

        xQ[1] = coordx[i][icenter];
        zQ[1] = coordz[j][icenter];
        if (i == 0   ) { pointQ[0] = pointQ[1]; xQ[0] = xQ[1];}
        else           { xQ[0] = coordx[i-1][icenter];}
        if (i == Nx-1) { pointQ[2] = pointQ[1]; xQ[2] = xQ[1];}
        else           { xQ[2] = coordx[i+1][icenter];}
        if (j == 0   ) { pointQ[3] = pointQ[1]; zQ[0] = zQ[1];}
        else           { zQ[0] = coordz[j-1][icenter];}
        if (j == Nz-1) { pointQ[4] = pointQ[1]; zQ[2] = zQ[1];}
        else           { zQ[2] = coordz[j+1][icenter];}

        // ierr = DMStagVecGetValuesStencil(dmHC,xphilocal,5,pointQ,Q); CHKERRQ(ierr);

        Qinterp = interp1DLin_3Points(xp[0],xQ[0],Q[0],xQ[1],Q[1],xQ[2],Q[2]);
        rhs[0] = -PetscPowScalar(Qinterp/par->phi0,par->n); //left 

        Qinterp = interp1DLin_3Points(xp[1],xQ[0],Q[0],xQ[1],Q[1],xQ[2],Q[2]);
        rhs[1] = -PetscPowScalar(Qinterp/par->phi0,par->n); //right

        Qinterp = interp1DLin_3Points(zp[2],zQ[0],Q[3],zQ[1],Q[1],zQ[2],Q[4]);
        rhs[2] = -PetscPowScalar(Qinterp/par->phi0,par->n); // down

        Qinterp = interp1DLin_3Points(zp[3],zQ[0],Q[3],zQ[1],Q[1],zQ[2],Q[4]);
        rhs[3] = -PetscPowScalar(Qinterp/par->phi0,par->n); // up

        for (ii = 0; ii < 4; ii++) {
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = rhs[ii];
        }
      }

      { // D3 = -K*(1+Bf)*k_hat, K = (phi/phi0)^n, Bf = 0 (edges, c=2)
        DMStagStencil point[4], pointQ[3];
        PetscScalar   xp[4],zp[4],Qinterp, Q[3], rhs[4],zQ[3], K, Bf = 0.0;
        PetscInt      ii;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 2;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 2;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = 2;
        point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = 2;

        xp[0] = coordx[i][iprev  ]; zp[0] = coordz[j][icenter];
        xp[1] = coordx[i][inext  ]; zp[1] = coordz[j][icenter];
        xp[2] = coordx[i][icenter]; zp[2] = coordz[j][iprev  ];
        xp[3] = coordx[i][icenter]; zp[3] = coordz[j][inext  ];

        rhs[0] = 0.0; // dir of gravity only
        rhs[1] = 0.0; 

        // get porosity - take into account domain borders
        pointQ[0].i = i; pointQ[0].j = j-1; pointQ[0].loc = ELEMENT; pointQ[0].c = 0;
        pointQ[1].i = i; pointQ[1].j = j  ; pointQ[1].loc = ELEMENT; pointQ[1].c = 0;
        pointQ[2].i = i; pointQ[2].j = j+1; pointQ[2].loc = ELEMENT; pointQ[2].c = 0;

        zQ[1] = coordz[j][icenter];
        if (j == 0   ) { pointQ[0] = pointQ[1]; zQ[0] = zQ[1];}
        else           {zQ[0] = coordz[j-1][icenter];}
        if (j == Nz-1) { pointQ[2] = pointQ[1]; zQ[2] = zQ[1];}
        else           {zQ[2] = coordz[j+1][icenter];}

        // ierr = DMStagVecGetValuesStencil(dmHC,xphilocal,3,pointQ,Q); CHKERRQ(ierr);
        Qinterp = interp1DLin_3Points(zp[2],zQ[0],Q[0],zQ[1],Q[1],zQ[2],Q[2]);
        K       = PetscPowScalar(Qinterp/par->phi0,par->n);
        rhs[2]  = -par->k_hat*K*(1+Bf);

        Qinterp = interp1DLin_3Points(zp[3],zQ[0],Q[0],zQ[1],Q[1],zQ[2],Q[2]);
        K       = PetscPowScalar(Qinterp/par->phi0,par->n);
        rhs[3]  = -par->k_hat*K*(1+Bf); 

        for (ii = 0; ii < 4; ii++) {
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] = rhs[ii];
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
  // ierr = VecDestroy(&xphilocal);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// FormCoefficient_H
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormCoefficient_H"
PetscErrorCode FormCoefficient_H(FDPDE fd, DM dm, Vec x, DM dmcoeff, Vec coeff, void *ctx)
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  PetscFunctionReturn(0);
}

// ---------------------------------------
// FormCoefficient_C
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormCoefficient_C"
PetscErrorCode FormCoefficient_C(FDPDE fd, DM dm, Vec x, DM dmcoeff, Vec coeff, void *ctx)
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  PetscFunctionReturn(0);
}