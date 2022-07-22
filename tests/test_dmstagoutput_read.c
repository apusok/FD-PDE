static char help[] = "DMStag output and read test \n\n";
// run: ./tests/test_dmstagoutput_read.app

#include "petsc.h"
#include "../dmstagoutput.h"

PetscErrorCode test_write(PetscInt Nx,PetscInt Nz,PetscInt dof0,PetscInt dof1,PetscInt dof2,const char fname[])
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
  ierr = DMStagSetUniformCoordinatesProduct(dm,0.0,5.0,0.0,2.0,0.0,0.0);CHKERRQ(ierr);

  // Create data
  ierr = DMCreateGlobalVector(dm,&x);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dm, &xlocal); CHKERRQ(ierr);

  // Get local domain
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  // Map global vectors to local domain
  ierr = DMStagVecGetArray(dm, xlocal, &xx); CHKERRQ(ierr);

  // Get dm coordinates array
  ierr = DMStagGetProductCoordinateArraysRead(dm,&cx,&cz,NULL);CHKERRQ(ierr);
  
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
    ierr = DMStagGetProductCoordinateLocationSlot(dm,DMSTAG_LEFT,&iprev);CHKERRQ(ierr);
    ierr = DMStagGetProductCoordinateLocationSlot(dm,DMSTAG_RIGHT,&inext);CHKERRQ(ierr);

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
    ierr = DMStagGetProductCoordinateLocationSlot(dm,DMSTAG_ELEMENT,&icenter);CHKERRQ(ierr);

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
  ierr = DMStagRestoreProductCoordinateArraysRead(dm,&cx,&cz,NULL);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArray(dm,xlocal,&xx); CHKERRQ(ierr);

  // Local to global
  ierr = DMLocalToGlobalBegin(dm,xlocal,INSERT_VALUES,x); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,xlocal,INSERT_VALUES,x); CHKERRQ(ierr);
  
  ierr = VecDestroy(&xlocal); CHKERRQ(ierr);

  // Output data
  ierr = DMStagViewBinaryPython(dm,x,fname);CHKERRQ(ierr);

 // Destroy
  ierr = DMDestroy(&dm); CHKERRQ(ierr);
  ierr = VecDestroy(&x); CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode test_read(const char fname[])
{
  DM             dm;
  Vec            x;
  char           fout[200];
  PetscErrorCode ierr;

  // Read data from file
  ierr = DMStagReadBinaryPython(&dm,&x,fname);CHKERRQ(ierr);
  ierr = PetscSNPrintf(fout,sizeof(fout),"%s_new",fname);
  ierr = DMStagViewBinaryPython(dm,x,fout);CHKERRQ(ierr);
  ierr = DMDestroy(&dm); CHKERRQ(ierr);
  ierr = VecDestroy(&x); CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main (int argc,char **argv)
{
  DM             dm;
  Vec            x;
  PetscInt       nx, ny, dof0, dof1, dof2, stencil_width = 1;
  char           fname[200];
  PetscMPIInt    size;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help); if (ierr) return(ierr);

  ierr = PetscOptionsGetInt(NULL, NULL, "-nx", &nx, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, NULL, "-ny", &ny, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, NULL, "-dof0", &dof0, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, NULL, "-dof1", &dof1, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, NULL, "-dof2", &dof2, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, NULL, "-stencil_width", &stencil_width, NULL); CHKERRQ(ierr);

  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size); CHKERRQ(ierr);

  // create dummy data and output
  ierr = PetscSNPrintf(fname,sizeof(fname),"out_test_dmstagoutput_read_%d",size);
  ierr = test_write(nx,ny,dof0,dof1,dof2,fname);CHKERRQ(ierr);

  // internally there is a dm/x created like this (used in this test for comparison)
  ierr = DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,nx,ny,PETSC_DECIDE,PETSC_DECIDE, 
                        dof0, dof1, dof2, DMSTAG_STENCIL_BOX,stencil_width, NULL,NULL, &dm);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMSetUp(dm);CHKERRQ(ierr);
  ierr = DMStagSetUniformCoordinatesProduct(dm,0.0,1.0,0.0,1.0,0.0,0.0);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm,&x);CHKERRQ(ierr);

  ierr = PetscSNPrintf(fname,sizeof(fname),"out_test_dmstagoutput_read_create_%d",size);
  ierr = DMStagViewBinaryPython(dm,x,fname);CHKERRQ(ierr);
  ierr = DMDestroy(&dm); CHKERRQ(ierr);
  ierr = VecDestroy(&x); CHKERRQ(ierr);

  // read and output again
  ierr = PetscSNPrintf(fname,sizeof(fname),"out_test_dmstagoutput_read_%d",size);
  ierr = test_read(fname);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return(ierr);
}