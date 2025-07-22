static char help[] = "DMStag output test - outputs dummy data for all combinations of dofs \n\n";
// run: ./test_dmstagoutput_

#include "../src/fdpde_dmstag.h"

// test0
PetscErrorCode test0(PetscInt Nx,PetscInt Nz,PetscInt dof0,PetscInt dof1,PetscInt dof2,const char fname[])
{
  DM              dm;
  Vec             x, xlocal;
  PetscInt        i,j,ii, sx, sz, nx, nz, idx;
  PetscScalar    **cx,**cz,***xx;
  PetscFunctionBeginUser;
  
  PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, Nx, Nz, PETSC_DECIDE, PETSC_DECIDE, 
                        dof0, dof1, dof2, DMSTAG_STENCIL_BOX,1, NULL,NULL, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm,0.0,1.0,0.0,1.0,0.0,0.0));

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

  // PetscCall(VecView(x,PETSC_VIEWER_STDOUT_WORLD));

 // Destroy
  PetscCall(DMDestroy(&dm)); 
  PetscCall(VecDestroy(&x)); 

  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main (int argc,char **argv)
{
  PetscInt        n;
  
  n = 10;
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));

  PetscCall(test0(n-1,n,2,2,2,"out_test_dmstagoutput0"));
  PetscCall(test0(n,n,2,1,0,"out_test_dmstagoutput1")); // can use imshow()
  PetscCall(test0(n+1,n,1,0,0,"out_test_dmstagoutput2"));
  PetscCall(test0(n,n+1,0,1,0,"out_test_dmstagoutput3"));
  PetscCall(test0(n+2,n,0,0,1,"out_test_dmstagoutput4"));

  PetscCall(PetscFinalize());
  return 0;
}