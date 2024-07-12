static char help[] = "Test for projection of material properties from dmswarm to dmstag\n\n";
// Run: mpiexec -n 2 ./test_dmswarm_project.app
// Visualize: 1) dmswarm - use ParaView to open xmf files
// 2) xproj - using python
// >>> import dmstagoutput as dmout
// >>> dmout.general_output_imshow('out_xproj',None,None)

#include "petsc.h"
#include "../src/fdpde_stokes.h"
#include "../src/dmstagoutput.h"
#include "../src/material_point.h"

// ---------------------------------------
PetscErrorCode test_dmswarm_project(PetscInt nx, PetscInt nz, PetscInt ppcell)
{
  FDPDE          fd;
  DM             dm, dmproj, dmswarm;
  Vec            xproj;
  PetscErrorCode  ierr;

  // set up fdpde object
  ierr = FDPDECreate(PETSC_COMM_WORLD,nx,nz,0.0,1.0,0.0,1.0,FDPDE_STOKES,&fd);CHKERRQ(ierr);
  ierr = FDPDESetUp(fd);CHKERRQ(ierr);
  ierr = FDPDEGetDM(fd,&dm); CHKERRQ(ierr);

  // create dm for projection
  ierr = DMStagCreateCompatibleDMStag(dm,3,3,3,0,&dmproj); CHKERRQ(ierr);
  ierr = DMSetUp(dmproj); CHKERRQ(ierr);
  ierr = DMStagSetUniformCoordinatesProduct(dmproj,0.0,1.0,0.0,1.0,0.0,0.0);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dmproj,&xproj);CHKERRQ(ierr);

  // set up a swarm object
  ierr = DMStagPICCreateDMSwarm(dm,&dmswarm);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(dmswarm,"id",1,PETSC_REAL);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(dmswarm,"id0",1,PETSC_REAL);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(dmswarm,"id1",1,PETSC_REAL);CHKERRQ(ierr);
  ierr = DMStagPICFinalize(dmswarm);CHKERRQ(ierr);

  PetscInt ppcell2[] = {ppcell,ppcell};
  ierr = MPointCoordLayout_DomainVolumeWithCellList(dmswarm,0,NULL,0.5,ppcell2,COOR_INITIALIZE);CHKERRQ(ierr);

  // initial condition
  PetscScalar *pcoor, *pfield, *pfield0, *pfield1;
  PetscInt    p, npoints;
  ierr = DMSwarmGetLocalSize(dmswarm,&npoints);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dmswarm,DMSwarmPICField_coor,NULL,NULL,(void**)&pcoor);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dmswarm,"id",NULL,NULL,(void**)&pfield);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dmswarm,"id0",NULL,NULL,(void**)&pfield0);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dmswarm,"id1",NULL,NULL,(void**)&pfield1);CHKERRQ(ierr);
  
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
  ierr = DMSwarmRestoreField(dmswarm,"id",NULL,NULL,(void**)&pfield);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(dmswarm,"id0",NULL,NULL,(void**)&pfield0);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(dmswarm,"id1",NULL,NULL,(void**)&pfield1);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(dmswarm,DMSwarmPICField_coor,NULL,NULL,(void**)&pcoor);CHKERRQ(ierr);
  // ierr = DMSwarmViewXDMF(dmswarm,"out_dmswarm_init.xmf");CHKERRQ(ierr);

  // project properties into dmproj
  PetscInt id;

  // these routines are buggy in parallel
  // id = 0; 
  // ierr = MPoint_ProjectQ1_arith_general(dmswarm,"id",dm,dmproj,0,id,xproj);CHKERRQ(ierr);//vertex
  // ierr = MPoint_ProjectQ1_arith_general(dmswarm,"id",dm,dmproj,1,id,xproj);CHKERRQ(ierr);//face
  // ierr = MPoint_ProjectQ1_arith_general(dmswarm,"id",dm,dmproj,2,id,xproj);CHKERRQ(ierr);//cell

  // id = 1; 
  // ierr = MPoint_ProjectQ1_arith_general(dmswarm,"id0",dm,dmproj,0,id,xproj);CHKERRQ(ierr);//vertex
  // ierr = MPoint_ProjectQ1_arith_general(dmswarm,"id0",dm,dmproj,1,id,xproj);CHKERRQ(ierr);//face
  // ierr = MPoint_ProjectQ1_arith_general(dmswarm,"id0",dm,dmproj,2,id,xproj);CHKERRQ(ierr);//cell

  id = 0; 
  ierr = MPoint_ProjectQ1_arith_general_AP(dmswarm,"id",dm,dmproj,0,id,xproj);CHKERRQ(ierr);//vertex
  ierr = MPoint_ProjectQ1_arith_general_AP(dmswarm,"id",dm,dmproj,1,id,xproj);CHKERRQ(ierr);//face
  ierr = MPoint_ProjectQ1_arith_general_AP(dmswarm,"id",dm,dmproj,2,id,xproj);CHKERRQ(ierr);//cell

  id = 1; 
  ierr = MPoint_ProjectQ1_arith_general_AP(dmswarm,"id0",dm,dmproj,0,id,xproj);CHKERRQ(ierr);//vertex
  ierr = MPoint_ProjectQ1_arith_general_AP(dmswarm,"id0",dm,dmproj,1,id,xproj);CHKERRQ(ierr);//face
  ierr = MPoint_ProjectQ1_arith_general_AP(dmswarm,"id0",dm,dmproj,2,id,xproj);CHKERRQ(ierr);//cell

  id = 2; 
  ierr = MPoint_ProjectQ1_arith_general_AP(dmswarm,"id1",dm,dmproj,0,id,xproj);CHKERRQ(ierr);//vertex
  ierr = MPoint_ProjectQ1_arith_general_AP(dmswarm,"id1",dm,dmproj,1,id,xproj);CHKERRQ(ierr);//face
  ierr = MPoint_ProjectQ1_arith_general_AP(dmswarm,"id1",dm,dmproj,2,id,xproj);CHKERRQ(ierr);//cell

  // output
  const char  *fieldname[] = {"id"};
  ierr = DMSwarmViewFieldsXDMF(dmswarm,"out_id_field.xmf",1,fieldname); CHKERRQ(ierr);
  ierr = DMSwarmViewXDMF(dmswarm,"out_dmswarm.xmf");CHKERRQ(ierr);

  ierr = DMStagViewBinaryPython(dmproj,xproj,"out_xproj");CHKERRQ(ierr);

  // clean
  ierr = VecDestroy(&xproj);CHKERRQ(ierr);
  ierr = DMDestroy(&dmproj);CHKERRQ(ierr);
  ierr = DMDestroy(&dmswarm);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = FDPDEDestroy(&fd);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "main"
int main (int argc,char **argv)
{
  PetscErrorCode  ierr;
  ierr = PetscInitialize(&argc,&argv,(char*)0,help); if (ierr) return(ierr);
  ierr = test_dmswarm_project(11,10,3);CHKERRQ(ierr);  
  ierr = PetscFinalize();
  return(ierr);
}