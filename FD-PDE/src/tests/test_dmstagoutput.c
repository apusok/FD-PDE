static char help[] = "DMStag output test - outputs dummy data for all combinations of dofs \n\n";
// run: ./tests/test_dmstagoutput.app

#include "petsc.h"
#include "../dmstagoutput.h"

// test0
PetscErrorCode test0(PetscInt Nx,PetscInt Nz,PetscInt dof0,PetscInt dof1,PetscInt dof2,const char fname[])
{
  DM              dm;
  Vec             x, xlocal;
  PetscInt        i,j,ii, sx, sz, nx, nz, idx;
  PetscScalar    **cx,**cz,***xx;
  PetscErrorCode  ierr;
  
  ierr = DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, Nx, Nz, PETSC_DECIDE, PETSC_DECIDE, 
                        dof0, dof1, dof2, DMSTAG_STENCIL_BOX,1, NULL,NULL, &dm);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMSetUp(dm);CHKERRQ(ierr);
  ierr = DMStagSetUniformCoordinatesProduct(dm,0.0,1.0,0.0,1.0,0.0,0.0);CHKERRQ(ierr);

  // Create data
  ierr = DMCreateGlobalVector(dm,&x);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dm, &xlocal); CHKERRQ(ierr);

  // Get local domain
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  // Map global vectors to local domain
  // ierr = DMGetLocalVector(dm, &xlocal); CHKERRQ(ierr);
  // ierr = DMGlobalToLocal (dm, x, INSERT_VALUES, xlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArrayDOF(dm, xlocal, &xx); CHKERRQ(ierr);

  // Get dm coordinates array
  ierr = DMStagGet1dCoordinateArraysDOFRead(dm,&cx,&cz,NULL);CHKERRQ(ierr);
  
  // vertex data - create dummy data (ct values)
  if (dof0) {
    for (ii = 0; ii<dof0; ii++) {
      for (j = sz; j<sz+nz; j++) {
        for (i = sx; i<sx+nx; i++) {
          ierr = DMStagGetLocationSlot(dm, DMSTAG_DOWN_LEFT,ii, &idx); CHKERRQ(ierr);
          xx[j][i][idx] = ii+1;
          ierr = DMStagGetLocationSlot(dm, DMSTAG_DOWN_RIGHT,ii, &idx); CHKERRQ(ierr);
          xx[j][i][idx] = ii+1;
          ierr = DMStagGetLocationSlot(dm, DMSTAG_UP_LEFT,ii, &idx); CHKERRQ(ierr);
          xx[j][i][idx] = ii+1;
          ierr = DMStagGetLocationSlot(dm, DMSTAG_UP_RIGHT,ii, &idx); CHKERRQ(ierr);
          xx[j][i][idx] = ii+1;
        }
      }
    }
  }

  // face data - create dummy data (coordinates)
  if (dof1) {
    PetscInt iprev=-1, inext=-1;
    ierr = DMStagGet1dCoordinateLocationSlot(dm,DMSTAG_LEFT,&iprev);CHKERRQ(ierr);
    ierr = DMStagGet1dCoordinateLocationSlot(dm,DMSTAG_RIGHT,&inext);CHKERRQ(ierr);

    for (ii = 0; ii<dof1; ii++) {
      for (j = sz; j<sz+nz; j++) {
        for (i = sx; i<sx+nx; i++) {
          ierr = DMStagGetLocationSlot(dm, DMSTAG_LEFT,ii, &idx); CHKERRQ(ierr);
          xx[j][i][idx] = (ii+1)*cx[i][iprev];
          ierr = DMStagGetLocationSlot(dm, DMSTAG_RIGHT,ii, &idx); CHKERRQ(ierr);
          xx[j][i][idx] = (ii+1)*cx[i][inext];
          ierr = DMStagGetLocationSlot(dm, DMSTAG_DOWN,ii, &idx); CHKERRQ(ierr);
          xx[j][i][idx] = (ii+1)*cz[j][iprev];
          ierr = DMStagGetLocationSlot(dm, DMSTAG_UP,ii, &idx); CHKERRQ(ierr);
          xx[j][i][idx] = (ii+1)*cz[j][inext];
        }
      }
    }
  }

  // element data - create dummy data (coordinates)
  if (dof2) {
    PetscInt icenter=-1;
    ierr = DMStagGet1dCoordinateLocationSlot(dm,DMSTAG_ELEMENT,&icenter);CHKERRQ(ierr);

    for (ii = 0; ii<dof2; ii++) {
      for (j = sz; j<sz+nz; j++) {
        for (i = sx; i<sx+nx; i++) {
          ierr = DMStagGetLocationSlot(dm, DMSTAG_ELEMENT,ii, &idx); CHKERRQ(ierr);
          xx[j][i][idx] = (ii+1)*cx[i][icenter]*cz[j][icenter];
        }
      }
    }
  }

  // Restore arrays, local vectors
  ierr = DMStagRestore1dCoordinateArraysDOFRead(dm,&cx,&cz,NULL);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArrayDOF(dm,xlocal,&xx); CHKERRQ(ierr);
  // ierr = DMRestoreLocalVector(dm,&xlocal); CHKERRQ(ierr);

  // Local to global
  ierr = DMLocalToGlobalBegin(dm,xlocal,INSERT_VALUES,x); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,xlocal,INSERT_VALUES,x); CHKERRQ(ierr);
  
  ierr = VecDestroy(&xlocal); CHKERRQ(ierr);

  // Output data
  ierr = DMStagViewBinaryPython(dm,x,fname);CHKERRQ(ierr);

  // ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

 // Destroy
  ierr = DMDestroy(&dm); CHKERRQ(ierr);
  ierr = VecDestroy(&x); CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main (int argc,char **argv)
{
  PetscInt        n;
  PetscErrorCode  ierr;
  
  n = 10;
  ierr = PetscInitialize(&argc,&argv,(char*)0,help); if (ierr) return(ierr);

  ierr = test0(n-1,n,2,2,2,"test0");CHKERRQ(ierr);
  ierr = test0(n,n,2,1,0,"test1");CHKERRQ(ierr); // can use imshow()
  ierr = test0(n+1,n,1,0,0,"test2");CHKERRQ(ierr);
  ierr = test0(n,n+1,0,1,0,"test3");CHKERRQ(ierr);
  ierr = test0(n+2,n,0,0,1,"test4");CHKERRQ(ierr);

  ierr = PetscFinalize();
  return(ierr);
}