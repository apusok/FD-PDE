static char help[] = "Test for projection of material properties from dmswarm to dmstag\n\n";
// Run: mpiexec -n 2 ./test_dmswarm_project -log_view
// Visualize: 1) dmswarm - use ParaView to open xmf files
// 2) xproj - using python
// >>> import dmstagoutput as dmout
// >>> dmout.general_output_imshow('out_xproj',None,None)

#include "../new_src/fdpde_stokes.h"
#include "../new_src/fdpde_dmswarm.h"

// ---------------------------------------
PetscErrorCode test_dmswarm_project(PetscInt nx, PetscInt nz, PetscInt ppcell)
{
  FDPDE          fd;
  DM             dm, dmproj, dmswarm;
  Vec            xproj;
  PetscFunctionBeginUser;

  // set up fdpde object
  PetscCall(FDPDECreate(PETSC_COMM_WORLD,nx,nz,0.0,1.0,0.0,1.0,FDPDE_STOKES,&fd));
  PetscCall(FDPDESetUp(fd));
  PetscCall(FDPDEGetDM(fd,&dm)); 

  // create dm for projection
  PetscCall(DMStagCreateCompatibleDMStag(dm,3,3,3,0,&dmproj)); 
  PetscCall(DMSetUp(dmproj)); 
  PetscCall(DMStagSetUniformCoordinatesProduct(dmproj,0.0,1.0,0.0,1.0,0.0,0.0));
  PetscCall(DMCreateGlobalVector(dmproj,&xproj));

  // set up a swarm object
  PetscCall(DMStagPICCreateDMSwarm(dm,&dmswarm));
  PetscCall(DMSwarmRegisterPetscDatatypeField(dmswarm,"id",1,PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(dmswarm,"id0",1,PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(dmswarm,"id1",1,PETSC_REAL));
  PetscCall(DMStagPICFinalize(dmswarm));

  PetscInt ppcell2[] = {ppcell,ppcell};
  PetscCall(MPointCoordLayout_DomainVolumeWithCellList(dmswarm,0,NULL,0.5,ppcell2,COOR_INITIALIZE));

  // initial condition
  PetscScalar *pcoor, *pfield, *pfield0, *pfield1;
  PetscInt    p, npoints;
  PetscCall(DMSwarmGetLocalSize(dmswarm,&npoints));
  PetscCall(DMSwarmGetField(dmswarm,DMSwarmPICField_coor,NULL,NULL,(void**)&pcoor));
  PetscCall(DMSwarmGetField(dmswarm,"id",NULL,NULL,(void**)&pfield));
  PetscCall(DMSwarmGetField(dmswarm,"id0",NULL,NULL,(void**)&pfield0));
  PetscCall(DMSwarmGetField(dmswarm,"id1",NULL,NULL,(void**)&pfield1));
  
  for (p=0; p<npoints; p++) {
    PetscScalar xc,zc;
    
    xc = pcoor[2*p+0]-0.5;
    zc = pcoor[2*p+1]-0.5;

    if (xc*xc + zc*zc <= 0.25*0.25) {
      pfield[p]  = 1.0; 
      pfield0[p] = 0.0; 
      pfield1[p] = 1.0; 
    } else {
      pfield[p]  = 0.0; 
      pfield0[p] = 1.0; 
      pfield1[p] = 0.0; 
    }
  }
  PetscCall(DMSwarmRestoreField(dmswarm,"id",NULL,NULL,(void**)&pfield));
  PetscCall(DMSwarmRestoreField(dmswarm,"id0",NULL,NULL,(void**)&pfield0));
  PetscCall(DMSwarmRestoreField(dmswarm,"id1",NULL,NULL,(void**)&pfield1));
  PetscCall(DMSwarmRestoreField(dmswarm,DMSwarmPICField_coor,NULL,NULL,(void**)&pcoor));
  // PetscCall(DMSwarmViewXDMF(dmswarm,"out_dmswarm_init.xmf"));

  // project properties into dmproj
  PetscInt id;

  // these routines are buggy in parallel
  // id = 0; 
  // PetscCall(MPoint_ProjectQ1_arith_general(dmswarm,"id",dm,dmproj,0,id,xproj));//vertex
  // PetscCall(MPoint_ProjectQ1_arith_general(dmswarm,"id",dm,dmproj,1,id,xproj));//face
  // PetscCall(MPoint_ProjectQ1_arith_general(dmswarm,"id",dm,dmproj,2,id,xproj));//cell

  // id = 1; 
  // PetscCall(MPoint_ProjectQ1_arith_general(dmswarm,"id0",dm,dmproj,0,id,xproj));//vertex
  // PetscCall(MPoint_ProjectQ1_arith_general(dmswarm,"id0",dm,dmproj,1,id,xproj));//face
  // PetscCall(MPoint_ProjectQ1_arith_general(dmswarm,"id0",dm,dmproj,2,id,xproj));//cell

  id = 0; 
  PetscCall(MPoint_ProjectQ1_arith_general_AP(dmswarm,"id",dm,dmproj,0,id,xproj));//vertex
  PetscCall(MPoint_ProjectQ1_arith_general_AP(dmswarm,"id",dm,dmproj,1,id,xproj));//face
  PetscCall(MPoint_ProjectQ1_arith_general_AP(dmswarm,"id",dm,dmproj,2,id,xproj));//cell

  id = 1; 
  PetscCall(MPoint_ProjectQ1_arith_general_AP(dmswarm,"id0",dm,dmproj,0,id,xproj));//vertex
  PetscCall(MPoint_ProjectQ1_arith_general_AP(dmswarm,"id0",dm,dmproj,1,id,xproj));//face
  PetscCall(MPoint_ProjectQ1_arith_general_AP(dmswarm,"id0",dm,dmproj,2,id,xproj));//cell

  id = 2; 
  PetscCall(MPoint_ProjectQ1_arith_general_AP(dmswarm,"id1",dm,dmproj,0,id,xproj));//vertex
  PetscCall(MPoint_ProjectQ1_arith_general_AP(dmswarm,"id1",dm,dmproj,1,id,xproj));//face
  PetscCall(MPoint_ProjectQ1_arith_general_AP(dmswarm,"id1",dm,dmproj,2,id,xproj));//cell

  // output
  const char  *fieldname[] = {"id"};
  PetscCall(DMSwarmViewFieldsXDMF(dmswarm,"out_id_field.xmf",1,fieldname)); 
  PetscCall(DMSwarmViewXDMF(dmswarm,"out_dmswarm.xmf"));

  PetscCall(DMStagViewBinaryPython(dmproj,xproj,"out_xproj"));

  // clean
  PetscCall(VecDestroy(&xproj));
  PetscCall(DMDestroy(&dmproj));
  PetscCall(DMDestroy(&dmswarm));
  PetscCall(DMDestroy(&dm));
  PetscCall(FDPDEDestroy(&fd));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "main"
int main (int argc,char **argv)
{
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(test_dmswarm_project(11,10,3));  
  PetscCall(PetscFinalize());
  return 0;
}