#include "MORbuoyancy.h"

// ---------------------------------------
// Dimensionalize
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "ScaleSolutionPV"
PetscErrorCode ScaleSolutionPV(DM dm, Vec x, Vec *_x, void *ctx)
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
#define __FUNCT__ "ScaleTemperatureComposition"
PetscErrorCode ScaleTemperatureComposition(DM dm, Vec x, Vec *_x, void *ctx, PetscInt do_Comp)
{
  UsrData       *usr = (UsrData*) ctx;
  PetscInt       i, j, sx, sz, nx, nz,idx;
  PetscScalar    ***xxnew, ***xx;
  Vec            xnew, xnewlocal, xlocal;
  PetscScalar    DX, X0;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  if (do_Comp) { // composition
    DX = usr->par->DC;
    X0 = usr->par->C0;
  }
  else { // temperature
    DX = usr->par->DT;
    X0 = usr->par->T0;
  }

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
      xxnew[j][i][idx] = xx[j][i][idx]*DX + X0;
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

// ---------------------------------------
// DoOutput
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "DoOutput"
PetscErrorCode DoOutput(void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  char           fout[FNAME_LENGTH];
  Vec            xscal;
  PetscErrorCode ierr;
  
  PetscFunctionBeginUser;

  // Need to output the usr->par, usr->scal, usr->nd parameters

  // Dimensionless output
  ierr = PetscSNPrintf(fout,sizeof(fout),"%s_PV_%d",usr->par->fname_out,usr->par->out_count);
  ierr = DMStagViewBinaryPython(usr->dmPV,usr->xPV,fout);CHKERRQ(ierr);

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s_T_%d",usr->par->fname_out,usr->par->out_count);
  ierr = DMStagViewBinaryPython(usr->dmHC,usr->xT,fout);CHKERRQ(ierr);

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s_C_%d",usr->par->fname_out,usr->par->out_count);
  ierr = DMStagViewBinaryPython(usr->dmHC,usr->xC,fout);CHKERRQ(ierr);

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s_Cf_%d",usr->par->fname_out,usr->par->out_count);
  ierr = DMStagViewBinaryPython(usr->dmHC,usr->xCf,fout);CHKERRQ(ierr);

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s_Cs_%d",usr->par->fname_out,usr->par->out_count);
  ierr = DMStagViewBinaryPython(usr->dmHC,usr->xCs,fout);CHKERRQ(ierr);

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s_phi_%d",usr->par->fname_out,usr->par->out_count);
  ierr = DMStagViewBinaryPython(usr->dmHC,usr->xphi,fout);CHKERRQ(ierr);

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s_T_solidus_%d",usr->par->fname_out,usr->par->out_count);
  ierr = DMStagViewBinaryPython(usr->dmHC,usr->xT,fout);CHKERRQ(ierr);

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s_Theta_%d",usr->par->fname_out,usr->par->out_count);
  ierr = DMStagViewBinaryPython(usr->dmHC,usr->xTheta,fout);CHKERRQ(ierr);

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s_H_%d",usr->par->fname_out,usr->par->out_count);
  ierr = DMStagViewBinaryPython(usr->dmHC,usr->xH,fout);CHKERRQ(ierr);

  // Dimensional output; xscal needs to be destroyed immediately
  if (usr->par->dim_out) {
    ierr = ScaleSolutionPV(usr->dmPV,usr->xPV,&xscal,usr);CHKERRQ(ierr);
    ierr = PetscSNPrintf(fout,sizeof(fout),"%s_PV_dim_%d",usr->par->fname_out,usr->par->out_count);
    ierr = DMStagViewBinaryPython(usr->dmPV,xscal,fout);CHKERRQ(ierr);
    ierr = VecDestroy(&xscal);CHKERRQ(ierr);

    ierr = ScaleTemperatureComposition(usr->dmHC,usr->xT,&xscal,usr,0);CHKERRQ(ierr);
    ierr = PetscSNPrintf(fout,sizeof(fout),"%s_T_dim_%d",usr->par->fname_out,usr->par->out_count);
    ierr = DMStagViewBinaryPython(usr->dmHC,xscal,fout);CHKERRQ(ierr);
    ierr = VecDestroy(&xscal);CHKERRQ(ierr); 

    ierr = ScaleTemperatureComposition(usr->dmHC,usr->xC,&xscal,usr,1);CHKERRQ(ierr);
    ierr = PetscSNPrintf(fout,sizeof(fout),"%s_C_dim_%d",usr->par->fname_out,usr->par->out_count);
    ierr = DMStagViewBinaryPython(usr->dmHC,xscal,fout);CHKERRQ(ierr);
    ierr = VecDestroy(&xscal);CHKERRQ(ierr);

    ierr = ScaleTemperatureComposition(usr->dmHC,usr->xCf,&xscal,usr,1);CHKERRQ(ierr);
    ierr = PetscSNPrintf(fout,sizeof(fout),"%s_Cf_dim_%d",usr->par->fname_out,usr->par->out_count);
    ierr = DMStagViewBinaryPython(usr->dmHC,xscal,fout);CHKERRQ(ierr);
    ierr = VecDestroy(&xscal);CHKERRQ(ierr);

    ierr = ScaleTemperatureComposition(usr->dmHC,usr->xCs,&xscal,usr,1);CHKERRQ(ierr);
    ierr = PetscSNPrintf(fout,sizeof(fout),"%s_Cs_dim_%d",usr->par->fname_out,usr->par->out_count);
    ierr = DMStagViewBinaryPython(usr->dmHC,xscal,fout);CHKERRQ(ierr);
    ierr = VecDestroy(&xscal);CHKERRQ(ierr);

    ierr = ScaleTemperatureComposition(usr->dmHC,usr->xTsol,&xscal,usr,0);CHKERRQ(ierr);
    ierr = PetscSNPrintf(fout,sizeof(fout),"%s_T_solidus_dim_%d",usr->par->fname_out,usr->par->out_count);
    ierr = DMStagViewBinaryPython(usr->dmHC,xscal,fout);CHKERRQ(ierr);
    ierr = VecDestroy(&xscal);CHKERRQ(ierr);

    ierr = ScaleTemperatureComposition(usr->dmHC,usr->xTheta,&xscal,usr,0);CHKERRQ(ierr);
    ierr = PetscSNPrintf(fout,sizeof(fout),"%s_Theta_dim_%d",usr->par->fname_out,usr->par->out_count);
    ierr = DMStagViewBinaryPython(usr->dmHC,xscal,fout);CHKERRQ(ierr);
    ierr = VecDestroy(&xscal);CHKERRQ(ierr);

    ierr = ScaleVectorUniform(usr->dmHC,usr->xH,&xscal,usr->scal->H);CHKERRQ(ierr);
    ierr = PetscSNPrintf(fout,sizeof(fout),"%s_H_dim_%d",usr->par->fname_out,usr->par->out_count);
    ierr = DMStagViewBinaryPython(usr->dmHC,xscal,fout);CHKERRQ(ierr);
    ierr = VecDestroy(&xscal);CHKERRQ(ierr);
  }

  // increment out_count
  usr->par->out_count += 1;

  PetscFunctionReturn(0);
}