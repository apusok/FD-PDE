#include "MORbuoyancy.h"

// ---------------------------------------
// Dimensionalize
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "ScaleSolution_PV"
PetscErrorCode ScaleSolution_PV(DM dm, Vec x, Vec *_x, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  PetscInt       i, j, sx, sz, nx, nz,idx;
  PetscScalar    ***xxnew, ***xx;
  Vec            xnew, xnewlocal, xlocal;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Create local and global vector associated with DM
  ierr = VecDuplicate(x,&xnew);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dm,&xnewlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm,xnewlocal,&xxnew); CHKERRQ(ierr);

  ierr = DMGetLocalVector(dm,&xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm,x,INSERT_VALUES,xlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm,xlocal,&xx); CHKERRQ(ierr);

  // Get domain corners
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      ierr = DMStagGetLocationSlot(dm,ELEMENT,0,&idx); CHKERRQ(ierr); 
      xxnew[j][i][idx] = xx[j][i][idx]*usr->scal->P; // [Pa]

      ierr = DMStagGetLocationSlot(dm,LEFT,0,&idx); CHKERRQ(ierr); 
      xxnew[j][i][idx] = xx[j][i][idx]*usr->scal->v; // [m/s]

      ierr = DMStagGetLocationSlot(dm,RIGHT,0,&idx); CHKERRQ(ierr); 
      xxnew[j][i][idx] = xx[j][i][idx]*usr->scal->v;

      ierr = DMStagGetLocationSlot(dm,DOWN,0,&idx); CHKERRQ(ierr); 
      xxnew[j][i][idx] = xx[j][i][idx]*usr->scal->v;

      ierr = DMStagGetLocationSlot(dm,UP,0,&idx); CHKERRQ(ierr); 
      xxnew[j][i][idx] = xx[j][i][idx]*usr->scal->v;
    }
  }

  // Restore arrays
  ierr = DMStagVecRestoreArray(dm,xnewlocal,&xxnew); CHKERRQ(ierr);
  ierr = DMStagVecRestoreArray(dm,xlocal,&xx); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &xlocal ); CHKERRQ(ierr);

  // Map local to global
  ierr = DMLocalToGlobalBegin(dm,xnewlocal,INSERT_VALUES,xnew); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,xnewlocal,INSERT_VALUES,xnew); CHKERRQ(ierr);
  ierr = VecDestroy(&xnewlocal); CHKERRQ(ierr);

  // Assign pointers
  *_x  = xnew;
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ScaleVectorUniform"
PetscErrorCode ScaleVectorUniform(DM dm, Vec x, Vec *_x, PetscScalar scal)
{
  PetscInt       i, j, sx, sz, nx, nz,idx,dof0,dof1,dof2, ii;
  PetscScalar    ***xxnew, ***xx;
  Vec            xnew, xnewlocal, xlocal;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Create local and global vector associated with DM
  ierr = VecDuplicate(x,&xnew);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dm,&xnewlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm,xnewlocal,&xxnew); CHKERRQ(ierr);

  ierr = DMGetLocalVector(dm,&xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm,x,INSERT_VALUES,xlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm,xlocal,&xx); CHKERRQ(ierr);

  // Get domain corners
  ierr = DMStagGetDOF(dm,&dof0,&dof1,&dof2,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      for (ii = 0; ii <dof2; ii++) {
        ierr = DMStagGetLocationSlot(dm,ELEMENT,ii,&idx); CHKERRQ(ierr); // element
        xxnew[j][i][idx] = xx[j][i][idx]*scal;
      }

      for (ii = 0; ii <dof1; ii++) { // faces
        ierr = DMStagGetLocationSlot(dm,LEFT,ii,&idx); CHKERRQ(ierr);
        xxnew[j][i][idx] = xx[j][i][idx]*scal;

        ierr = DMStagGetLocationSlot(dm,RIGHT,ii,&idx); CHKERRQ(ierr);
        xxnew[j][i][idx] = xx[j][i][idx]*scal;

        ierr = DMStagGetLocationSlot(dm,DOWN,ii,&idx); CHKERRQ(ierr);
        xxnew[j][i][idx] = xx[j][i][idx]*scal;

        ierr = DMStagGetLocationSlot(dm,UP,ii,&idx); CHKERRQ(ierr);
        xxnew[j][i][idx] = xx[j][i][idx]*scal;
      }

      for (ii = 0; ii <dof0; ii++) { // nodes
        ierr = DMStagGetLocationSlot(dm,DOWN_LEFT,ii,&idx); CHKERRQ(ierr);
        xxnew[j][i][idx] = xx[j][i][idx]*scal;

        ierr = DMStagGetLocationSlot(dm,DOWN_RIGHT,ii,&idx); CHKERRQ(ierr);
        xxnew[j][i][idx] = xx[j][i][idx]*scal;

        ierr = DMStagGetLocationSlot(dm,UP_LEFT,ii,&idx); CHKERRQ(ierr);
        xxnew[j][i][idx] = xx[j][i][idx]*scal;

        ierr = DMStagGetLocationSlot(dm,UP_RIGHT,ii,&idx); CHKERRQ(ierr);
        xxnew[j][i][idx] = xx[j][i][idx]*scal;
      }
    }
  }

  // Restore arrays
  ierr = DMStagVecRestoreArray(dm,xnewlocal,&xxnew); CHKERRQ(ierr);
  ierr = DMStagVecRestoreArray(dm,xlocal,&xx); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &xlocal ); CHKERRQ(ierr);

  // Map local to global
  ierr = DMLocalToGlobalBegin(dm,xnewlocal,INSERT_VALUES,xnew); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,xnewlocal,INSERT_VALUES,xnew); CHKERRQ(ierr);
  ierr = VecDestroy(&xnewlocal); CHKERRQ(ierr);

  // Assign pointers
  *_x  = xnew;
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ScaleTemperature"
PetscErrorCode ScaleTemperature(DM dm, Vec x, Vec *_x, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  PetscInt       i, j, sx, sz, nx, nz,idx;
  PetscScalar    ***xxnew, ***xx;
  Vec            xnew, xnewlocal, xlocal;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Create local and global vector associated with DM
  ierr = VecDuplicate(x,&xnew);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dm,&xnewlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm,xnewlocal,&xxnew); CHKERRQ(ierr);

  ierr = DMGetLocalVector(dm,&xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm,x,INSERT_VALUES,xlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm,xlocal,&xx); CHKERRQ(ierr);

  // Get domain corners
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      ierr = DMStagGetLocationSlot(dm,ELEMENT,0,&idx); CHKERRQ(ierr); 
      xxnew[j][i][idx] = xx[j][i][idx]*usr->par->DT + usr->par->T0; // [K]
    }
  }

  // Restore arrays
  ierr = DMStagVecRestoreArray(dm,xnewlocal,&xxnew); CHKERRQ(ierr);
  ierr = DMStagVecRestoreArray(dm,xlocal,&xx); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &xlocal ); CHKERRQ(ierr);

  // Map local to global
  ierr = DMLocalToGlobalBegin(dm,xnewlocal,INSERT_VALUES,xnew); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,xnewlocal,INSERT_VALUES,xnew); CHKERRQ(ierr);
  ierr = VecDestroy(&xnewlocal); CHKERRQ(ierr);

  // Assign pointers
  *_x  = xnew;
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ScaleComposition"
PetscErrorCode ScaleComposition(DM dm, Vec x, Vec *_x, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  PetscInt       i, j, sx, sz, nx, nz,idx;
  PetscScalar    ***xxnew, ***xx;
  Vec            xnew, xnewlocal, xlocal;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Create local and global vector associated with DM
  ierr = VecDuplicate(x,&xnew);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dm,&xnewlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm,xnewlocal,&xxnew); CHKERRQ(ierr);

  ierr = DMGetLocalVector(dm,&xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm,x,INSERT_VALUES,xlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm,xlocal,&xx); CHKERRQ(ierr);

  // Get domain corners
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      ierr = DMStagGetLocationSlot(dm,ELEMENT,0,&idx); CHKERRQ(ierr); 
      xxnew[j][i][idx] = xx[j][i][idx]*usr->par->DC + usr->par->C0;
    }
  }

  // Restore arrays
  ierr = DMStagVecRestoreArray(dm,xnewlocal,&xxnew); CHKERRQ(ierr);
  ierr = DMStagVecRestoreArray(dm,xlocal,&xx); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &xlocal ); CHKERRQ(ierr);

  // Map local to global
  ierr = DMLocalToGlobalBegin(dm,xnewlocal,INSERT_VALUES,xnew); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,xnewlocal,INSERT_VALUES,xnew); CHKERRQ(ierr);
  ierr = VecDestroy(&xnewlocal); CHKERRQ(ierr);

  // Assign pointers
  *_x  = xnew;
  
  PetscFunctionReturn(0);
}

// PetscErrorCode ScaleCoefficient(DM dm, Vec x, Vec *_x, void *ctx)
// {
//   UsrData       *usr = (UsrData*) ctx;
//   PetscInt       i, j, sx, sz, nx, nz,idx;
//   PetscScalar    ***xxnew, ***xx;
//   Vec            xnew, xnewlocal, xlocal;
//   PetscErrorCode ierr;

//   PetscFunctionBegin;

//   // Create local and global vector associated with DM
//   ierr = VecDuplicate(x,&xnew);CHKERRQ(ierr);
//   ierr = DMCreateLocalVector(dm,&xnewlocal); CHKERRQ(ierr);
//   ierr = DMStagVecGetArray(dm,xnewlocal,&xxnew); CHKERRQ(ierr);

//   ierr = DMGetLocalVector(dm,&xlocal); CHKERRQ(ierr);
//   ierr = DMGlobalToLocal (dm,x,INSERT_VALUES,xlocal); CHKERRQ(ierr);
//   ierr = DMStagVecGetArray(dm,xlocal,&xx); CHKERRQ(ierr);

//   // Get domain corners
//   ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

//   for (j = sz; j < sz+nz; j++) {
//     for (i = sx; i <sx+nx; i++) {
//       ierr = DMStagGetLocationSlot(dm,ELEMENT,0,&idx); CHKERRQ(ierr); // C
//       xxnew[j][i][idx] = xx[j][i][idx]*usr->par->vel/usr->par->length;

//       ierr = DMStagGetLocationSlot(dm,ELEMENT,1,&idx); CHKERRQ(ierr); // A=eta
//       xxnew[j][i][idx] = xx[j][i][idx]*usr->par->visc;

//       ierr = DMStagGetLocationSlot(dm,LEFT,0,&idx); CHKERRQ(ierr); // Bx
//       xxnew[j][i][idx] = xx[j][i][idx]*usr->par->stress/usr->par->length;

//       ierr = DMStagGetLocationSlot(dm,RIGHT,0,&idx); CHKERRQ(ierr); // Bx
//       xxnew[j][i][idx] = xx[j][i][idx]*usr->par->stress/usr->par->length;

//       ierr = DMStagGetLocationSlot(dm,DOWN,0,&idx); CHKERRQ(ierr); //Bz
//       xxnew[j][i][idx] = xx[j][i][idx]*usr->par->stress/usr->par->length;

//       ierr = DMStagGetLocationSlot(dm,UP,0,&idx); CHKERRQ(ierr); // Bz
//       xxnew[j][i][idx] = xx[j][i][idx]*usr->par->stress/usr->par->length;

//       ierr = DMStagGetLocationSlot(dm,DOWN_LEFT,0,&idx); CHKERRQ(ierr); // A corner
//       xxnew[j][i][idx] = xx[j][i][idx]*usr->par->visc;

//       ierr = DMStagGetLocationSlot(dm,DOWN_RIGHT,0,&idx); CHKERRQ(ierr);
//       xxnew[j][i][idx] = xx[j][i][idx]*usr->par->visc;

//       ierr = DMStagGetLocationSlot(dm,UP_LEFT,0,&idx); CHKERRQ(ierr);
//       xxnew[j][i][idx] = xx[j][i][idx]*usr->par->visc;

//       ierr = DMStagGetLocationSlot(dm,UP_RIGHT,0,&idx); CHKERRQ(ierr);
//       xxnew[j][i][idx] = xx[j][i][idx]*usr->par->visc;
//     }
//   }

//   // Restore arrays
//   ierr = DMStagVecRestoreArray(dm,xnewlocal,&xxnew); CHKERRQ(ierr);
//   ierr = DMStagVecRestoreArray(dm,xlocal,&xx); CHKERRQ(ierr);
//   ierr = DMRestoreLocalVector(dm, &xlocal ); CHKERRQ(ierr);

//   // Map local to global
//   ierr = DMLocalToGlobalBegin(dm,xnewlocal,INSERT_VALUES,xnew); CHKERRQ(ierr);
//   ierr = DMLocalToGlobalEnd  (dm,xnewlocal,INSERT_VALUES,xnew); CHKERRQ(ierr);
//   ierr = VecDestroy(&xnewlocal); CHKERRQ(ierr);

//   // Assign pointers
//   *_x  = xnew;
  
//   PetscFunctionReturn(0);
// }