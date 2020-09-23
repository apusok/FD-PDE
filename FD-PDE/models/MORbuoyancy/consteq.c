#include "MORbuoyancy.h"

static PetscScalar Temp2Theta(PetscScalar x, PetscScalar Az) { return(x*exp(-Az));}
static PetscScalar Theta2Temp(PetscScalar x, PetscScalar Az) { return(x*exp( Az));}

// ---------------------------------------
// Update Theta From Temperature
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "UpdateThetaFromTemp"
PetscErrorCode UpdateThetaFromTemp(void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz, idx, icenter;
  PetscScalar    ***xx, ***xxtheta;
  PetscScalar    **coordx,**coordz;
  Vec            x, xlocal, xtheta, xthetalocal;
  DM             dm;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  dm = usr->dmHC;
  x  = usr->xT;
  xtheta  = usr->xTheta;

  ierr = DMStagGetGlobalSizes(dm, &Nx, &Nz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  ierr = DMGetLocalVector(dm, &xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm, x, INSERT_VALUES, xlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm, xlocal, &xx); CHKERRQ(ierr);

  ierr = DMCreateLocalVector(dm, &xthetalocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm, xthetalocal, &xxtheta); CHKERRQ(ierr);

// Get dm coordinates array
  ierr = DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter);CHKERRQ(ierr); 

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      PetscScalar Az;
      Az = -usr->nd->A*coordz[j][icenter]; // check sign of non-dimensional depth!
      ierr = DMStagGetLocationSlot(dm, ELEMENT, 0, &idx); CHKERRQ(ierr);
      xxtheta[j][i][idx] = Temp2Theta(xx[j][i][idx],Az);
    }
  }

  // Restore arrays
  ierr = DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArray(dm,xlocal,&xx); CHKERRQ(ierr);
  ierr = DMStagVecRestoreArray(dm,xthetalocal,&xxtheta); CHKERRQ(ierr);

  ierr = DMLocalToGlobalBegin(dm,xthetalocal,INSERT_VALUES,xtheta); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,xthetalocal,INSERT_VALUES,xtheta); CHKERRQ(ierr);
  ierr = VecDestroy(&xthetalocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&xlocal); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}