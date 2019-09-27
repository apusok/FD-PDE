// #include "bc.h"

// // ---------------------------------------
// // DMStagExtract1DComponent
// // ---------------------------------------
// #undef __FUNCT__
// #define __FUNCT__ "DMStagExtract1DComponent"
// PetscErrorCode DMStagExtract1DComponent(DM dm, Vec x, DMStagStencilLocation loc, PetscInt c, PetscScalar a, PetscScalar *val)
// {
//   PetscInt i, j, nx, nz, sx, sz;
//   Vec  xlocal;
//   PetscErrorCode ierr;
//   PetscFunctionBegin;

//   // a is a prefactor

//   // Get domain corners
//   ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

//   // Map coefficient data to local domain
//   ierr = DMGetLocalVector(dm,&xlocal); CHKERRQ(ierr);
//   ierr = DMGlobalToLocal (dm,x,INSERT_VALUES,xlocal); CHKERRQ(ierr);

//   // Loop over local domain
//   for (j = sz; j < sz+nz; j++) {
//     for (i = sx; i <sx+nx; i++) {
//       if (loc == DMSTAG_ELEMENT) {
//         DMStagStencil point;
//         PetscScalar   xx;
//         point.i = i; point.j = j; point.loc = DMSTAG_ELEMENT; point.c = c;
//         ierr = DMStagVecGetValuesStencil(dm, xlocal, 1, &point, &xx); CHKERRQ(ierr); 
//         val[i-sx+(j-sz)*nz] = a*xx;
//       }

//       if ((loc == DMSTAG_LEFT) || (loc == DMSTAG_RIGHT)) {
//         DMStagStencil point[2];
//         PetscScalar   xx[2];
//         point[0].i = i; point[0].j = j; point[0].loc = DMSTAG_LEFT;  point[0].c = c;
//         point[1].i = i; point[1].j = j; point[1].loc = DMSTAG_RIGHT; point[1].c = c;
//         ierr = DMStagVecGetValuesStencil(dm, xlocal, 2, point, xx); CHKERRQ(ierr); 
//         val[i  -sx+(j-sz)*nz] = a*xx[0];
//         val[i+1-sx+(j-sz)*nz] = a*xx[1];
//       }

//       if ((loc == DMSTAG_UP) || (loc == DMSTAG_DOWN)) {
//         DMStagStencil point[2];
//         PetscScalar   xx[2];
//         point[0].i = i; point[0].j = j; point[0].loc = DMSTAG_DOWN;  point[0].c = c;
//         point[1].i = i; point[1].j = j; point[1].loc = DMSTAG_UP;    point[1].c = c;
//         ierr = DMStagVecGetValuesStencil(dm, xlocal, 2, point, xx); CHKERRQ(ierr); 
//         val[i  -sx+(j  -sz)*nz] = a*xx[0];
//         val[i  -sx+(j+1-sz)*nz] = a*xx[1];
//       }

//       if ((loc == DMSTAG_DOWN_LEFT) || (loc == DMSTAG_DOWN_RIGHT) || (loc == DMSTAG_UP_LEFT) || (loc == DMSTAG_UP_RIGHT)) {
//         DMStagStencil point[4];
//         PetscScalar   xx[4];
//         point[0].i = i; point[0].j = j; point[0].loc = DMSTAG_DOWN_LEFT;  point[0].c = c;
//         point[1].i = i; point[1].j = j; point[1].loc = DMSTAG_UP_LEFT;    point[1].c = c;
//         point[2].i = i; point[2].j = j; point[2].loc = DMSTAG_DOWN_RIGHT; point[2].c = c;
//         point[3].i = i; point[3].j = j; point[3].loc = DMSTAG_UP_RIGHT;   point[3].c = c;
//         ierr = DMStagVecGetValuesStencil(dm, xlocal, 4, point, xx); CHKERRQ(ierr); 
//         val[i  -sx+(j  -sz)*nz] = a*xx[0];
//         val[i  -sx+(j+1-sz)*nz] = a*xx[1];
//         val[i+1-sx+(j  -sz)*nz] = a*xx[2];
//         val[i+1-sx+(j+1-sz)*nz] = a*xx[3];
//       }
//     }
//   }

//   // Restore arrays, local vectors
//   ierr = DMRestoreLocalVector(dm,&xlocal); CHKERRQ(ierr);

//   PetscFunctionReturn(0);
// }

// // ---------------------------------------
// // Get coordinate of stencil point [works only with DMStagSetUniformCoordinatesExplicit() not Product coordinates!]
// // <should be deleted later>
// // ---------------------------------------
// PetscErrorCode GetCoordinatesStencil(DM dm, Vec vec, PetscInt n, DMStagStencil point[], PetscScalar x[], PetscScalar z[])
// {
//   PetscScalar    xx[2];
//   PetscInt       i;
//   DMStagStencil  pointx[2];
//   PetscErrorCode ierr;
//   PetscFunctionBeginUser;

//   for (i = 0; i<n; i++) {
//     // Get coordinates x,z of point
//     pointx[0] = point[i]; pointx[0].c = 0;
//     pointx[1] = point[i]; pointx[1].c = 1;

//     ierr = DMStagVecGetValuesStencil(dm,vec,2,pointx,xx); CHKERRQ(ierr);
//     x[i] = xx[0];
//     z[i] = xx[1];
//   }

//   PetscFunctionReturn(0);
// }
