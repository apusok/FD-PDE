static char help[] = "DMStag output and read test \n\n";
// run: ./tests/test_dmstagoutput_read.app

#include "petsc.h"
#include "../src/dmstagoutput.h"

PetscErrorCode test_write(PetscInt Nx,PetscInt Nz,PetscInt dof0,PetscInt dof1,PetscInt dof2,const char fname[])
{
  DM              dm;
  Vec             x, xlocal;
  PetscInt        i,j,ii, sx, sz, nx, nz, idx;
  PetscScalar    **cx,**cz,***xx;
  
  PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, Nx, Nz, PETSC_DECIDE, PETSC_DECIDE, 
                        dof0, dof1, dof2, DMSTAG_STENCIL_BOX,1, NULL,NULL, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm,0.0,5.0,0.0,2.0,0.0,0.0));

  // Create data
  PetscCall(DMCreateGlobalVector(dm,&x));
  PetscCall(DMCreateLocalVector(dm, &xlocal)); 

  // Get local domain
  PetscCall(DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 

  // Map global vectors to local domain
  PetscCall(DMStagVecGetArray(dm, xlocal, &xx)); 

  // Get dm coordinates array
  PetscCall(DMStagGetProductCoordinateArraysRead(dm,&cx,&cz,NULL));
  
  // vertex data - create dummy data (ct values)
  if (dof0) {
    for (ii = 0; ii<dof0; ii++) {
      for (j = sz; j<sz+nz; j++) {
        for (i = sx; i<sx+nx; i++) {
          PetscCall(DMStagGetLocationSlot(dm, DMSTAG_DOWN_LEFT,ii, &idx)); 
          xx[j][i][idx] = ii+1;
          PetscCall(DMStagGetLocationSlot(dm, DMSTAG_DOWN_RIGHT,ii, &idx)); 
          xx[j][i][idx] = ii+1;
          PetscCall(DMStagGetLocationSlot(dm, DMSTAG_UP_LEFT,ii, &idx)); 
          xx[j][i][idx] = ii+1;
          PetscCall(DMStagGetLocationSlot(dm, DMSTAG_UP_RIGHT,ii, &idx)); 
          xx[j][i][idx] = ii+1;
        }
      }
    }
  }

  // face data - create dummy data (coordinates)
  if (dof1) {
    PetscInt iprev=-1, inext=-1;
    PetscCall(DMStagGetProductCoordinateLocationSlot(dm,DMSTAG_LEFT,&iprev));
    PetscCall(DMStagGetProductCoordinateLocationSlot(dm,DMSTAG_RIGHT,&inext));

    for (ii = 0; ii<dof1; ii++) {
      for (j = sz; j<sz+nz; j++) {
        for (i = sx; i<sx+nx; i++) {
          PetscCall(DMStagGetLocationSlot(dm, DMSTAG_LEFT,ii, &idx)); 
          xx[j][i][idx] = (ii+1)*cx[i][iprev];
          PetscCall(DMStagGetLocationSlot(dm, DMSTAG_RIGHT,ii, &idx)); 
          xx[j][i][idx] = (ii+1)*cx[i][inext];
          PetscCall(DMStagGetLocationSlot(dm, DMSTAG_DOWN,ii, &idx)); 
          xx[j][i][idx] = (ii+1)*cz[j][iprev];
          PetscCall(DMStagGetLocationSlot(dm, DMSTAG_UP,ii, &idx)); 
          xx[j][i][idx] = (ii+1)*cz[j][inext];
        }
      }
    }
  }

  // element data - create dummy data (coordinates)
  if (dof2) {
    PetscInt icenter=-1;
    PetscCall(DMStagGetProductCoordinateLocationSlot(dm,DMSTAG_ELEMENT,&icenter));

    for (ii = 0; ii<dof2; ii++) {
      for (j = sz; j<sz+nz; j++) {
        for (i = sx; i<sx+nx; i++) {
          PetscCall(DMStagGetLocationSlot(dm, DMSTAG_ELEMENT,ii, &idx)); 
          xx[j][i][idx] = (ii+1)*cx[i][icenter]*cz[j][icenter];
        }
      }
    }
  }

  // Restore arrays, local vectors
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm,&cx,&cz,NULL));
  PetscCall(DMStagVecRestoreArray(dm,xlocal,&xx)); 

  // Local to global
  PetscCall(DMLocalToGlobalBegin(dm,xlocal,INSERT_VALUES,x)); 
  PetscCall(DMLocalToGlobalEnd  (dm,xlocal,INSERT_VALUES,x)); 
  
  PetscCall(VecDestroy(&xlocal)); 

  // Output data
  PetscCall(DMStagViewBinaryPython(dm,x,fname));

 // Destroy
  PetscCall(DMDestroy(&dm)); 
  PetscCall(VecDestroy(&x)); 
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode test_read(const char fname[])
{
  DM             dm;
  Vec            x;
  char           fout[200];

  // Read data from file
  PetscCall(DMStagReadBinaryPython(&dm,&x,fname));
  PetscCall(PetscSNPrintf(fout,sizeof(fout),"%s_new",fname));
  PetscCall(DMStagViewBinaryPython(dm,x,fout));
  PetscCall(DMDestroy(&dm)); 
  PetscCall(VecDestroy(&x)); 
  
  PetscFunctionReturn(PETSC_SUCCESS);
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

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nx", &nx, NULL)); 
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-ny", &ny, NULL)); 
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-dof0", &dof0, NULL)); 
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-dof1", &dof1, NULL)); 
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-dof2", &dof2, NULL)); 
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-stencil_width", &stencil_width, NULL)); 

  PetscCall(MPI_Comm_size(PETSC_COMM_WORLD,&size)); 

  // create dummy data and output
  PetscCall(PetscSNPrintf(fname,sizeof(fname),"out_test_dmstagoutput_read_%d",size));
  PetscCall(test_write(nx,ny,dof0,dof1,dof2,fname));

  // internally there is a dm/x created like this (used in this test for comparison)
  PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,nx,ny,PETSC_DECIDE,PETSC_DECIDE, 
                        dof0, dof1, dof2, DMSTAG_STENCIL_BOX,stencil_width, NULL,NULL, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm,0.0,1.0,0.0,1.0,0.0,0.0));
  PetscCall(DMCreateGlobalVector(dm,&x));

  PetscCall(PetscSNPrintf(fname,sizeof(fname),"out_test_dmstagoutput_read_create_%d",size));
  PetscCall(DMStagViewBinaryPython(dm,x,fname));
  PetscCall(DMDestroy(&dm)); 
  PetscCall(VecDestroy(&x)); 

  // read and output again
  PetscCall(PetscSNPrintf(fname,sizeof(fname),"out_test_dmstagoutput_read_%d",size));
  PetscCall(test_read(fname));

  PetscCall(PetscFinalize());
  return 0;
}