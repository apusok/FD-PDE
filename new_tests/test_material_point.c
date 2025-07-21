// run: ./test_material_point -log_view

static char help[] = "Material point layout test \n\n";

#include "../new_src/fdpde_dmswarm.h"

// ---------------------------------------
PetscErrorCode test_layout(PetscInt nx,PetscInt ny)
{
  DM              dm,dmswarm;
  PetscInt        dof0,dof1,dof2,stencilWidth;
  PetscFunctionBeginUser;
  
  dof0 = 0; dof1 = 1; dof2 = 1; /* (vertex) (face) (element) */
  stencilWidth = 1;
  
  PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, nx, ny,PETSC_DECIDE, PETSC_DECIDE, dof0, dof1, dof2,
                        DMSTAG_STENCIL_BOX, stencilWidth, NULL,NULL, &dm));
  PetscCall(DMStagSetCoordinateDMType(dm,DMPRODUCT));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  
  PetscCall(DMStagSetUniformCoordinatesProduct(dm,0.0,1.0,0.0,1.0,0.0,0.0));

  PetscCall(DMStagPICCreateDMSwarm(dm,&dmswarm));
  PetscCall(DMSwarmRegisterPetscDatatypeField(dmswarm,"eta",1,PETSC_REAL));
  PetscCall(DMStagPICFinalize(dmswarm));

  /*PetscCall(MPointCoordLayout_DomainVolume(dmswarm,0.0,1,COOR_INITIALIZE));*/
  {
    PetscInt ppcell[] = {1,1};
    PetscCall(MPointCoordLayout_DomainVolumeWithCellList(dmswarm,0,NULL,0.0,ppcell,COOR_INITIALIZE));
    
    ppcell[0] = ppcell[1] = 2;
    PetscCall(MPointCoordLayout_DomainVolumeWithCellList(dmswarm,0,NULL,0.0,ppcell,COOR_APPEND));
    
    PetscCall(MPointCoordLayout_DomainFace(dmswarm,'n',0.0,6,COOR_APPEND));
    PetscCall(MPointCoordLayout_DomainFace(dmswarm,'s',0.0,10,COOR_APPEND));
    
    PetscCall(MPointCoordLayout_DomainFace(dmswarm,'w',0.0,1,COOR_APPEND));
    PetscCall(MPointCoordLayout_DomainFace(dmswarm,'e',0.0,1,COOR_APPEND));
  }

  {
    PetscReal *pcoor;
    PetscInt npoints,p;
    PetscCall(DMSwarmGetLocalSize(dmswarm,&npoints));
    PetscCall(DMSwarmGetField(dmswarm,DMSwarmPICField_coor,NULL,NULL,(void**)&pcoor));
    for (p=0; p<npoints; p++) {
      printf("%d : %+1.4e %+1.4e\n",p,pcoor[2*p],pcoor[2*p+1]);
    }
    PetscCall(DMSwarmRestoreField(dmswarm,DMSwarmPICField_coor,NULL,NULL,(void**)&pcoor));
  }
  
  PetscCall(DMSwarmViewXDMF(dmswarm,"dms.xmf"));

  PetscCall(DMDestroy(&dmswarm));
  PetscCall(DMDestroy(&dm));
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
PetscErrorCode test_advection_rk1(PetscInt nx,PetscInt ny)
{
  DM              dm,dmswarm;
  PetscInt        dof0,dof1,dof2,stencilWidth,k;
  Vec             X,Xl;
  PetscReal       ***vel;
  PetscFunctionBeginUser;
  
  dof0 = 0; dof1 = 1; dof2 = 1; /* (vertex) (face) (element) */
  stencilWidth = 1;
  
  PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, nx, ny,PETSC_DECIDE, PETSC_DECIDE, dof0, dof1, dof2,
                        DMSTAG_STENCIL_BOX, stencilWidth, NULL,NULL, &dm));
  PetscCall(DMStagSetCoordinateDMType(dm,DMPRODUCT));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  
  PetscCall(DMStagSetUniformCoordinatesProduct(dm,0.0,1.0,0.0,1.0,0.0,0.0));
  
  PetscCall(DMStagPICCreateDMSwarm(dm,&dmswarm));
  PetscCall(DMSwarmRegisterPetscDatatypeField(dmswarm,"eta",1,PETSC_REAL));
  PetscCall(DMStagPICFinalize(dmswarm));
  
  /*PetscCall(MPointCoordLayout_DomainVolume(dmswarm,0.0,1,COOR_INITIALIZE));*/
  {
    PetscInt ppcell[] = {1,1};
    PetscCall(MPointCoordLayout_DomainVolumeWithCellList(dmswarm,0,NULL,0.3,ppcell,COOR_INITIALIZE));
    
    ppcell[0] = ppcell[1] = 2;
    PetscCall(MPointCoordLayout_DomainVolumeWithCellList(dmswarm,0,NULL,0.3,ppcell,COOR_APPEND));
    
    PetscCall(MPointCoordLayout_DomainFace(dmswarm,'n',0.0,6,COOR_APPEND));
    PetscCall(MPointCoordLayout_DomainFace(dmswarm,'s',0.0,10,COOR_APPEND));
    
    PetscCall(MPointCoordLayout_DomainFace(dmswarm,'w',0.0,1,COOR_APPEND));
    PetscCall(MPointCoordLayout_DomainFace(dmswarm,'e',0.0,1,COOR_APPEND));
  }

  PetscCall(DMSwarmViewXDMF(dmswarm,"dmsA.xmf"));
  
  
  {
    PetscInt slot_vx[2],slot_vy[2],ci,cj,es[2],nele[2];
    
    PetscCall(DMCreateGlobalVector(dm,&X));
    PetscCall(DMCreateLocalVector(dm,&Xl));
    PetscCall(DMStagGetCorners(dm,&es[0],&es[1],NULL,&nele[0],&nele[1],NULL,NULL,NULL,NULL));
    
    DMStagGetLocationSlot(dm,DMSTAG_LEFT,0,&slot_vx[0]);
    DMStagGetLocationSlot(dm,DMSTAG_RIGHT,0,&slot_vx[1]);
    DMStagGetLocationSlot(dm,DMSTAG_DOWN,0,&slot_vy[0]);
    DMStagGetLocationSlot(dm,DMSTAG_UP,0,&slot_vy[1]);
    
    DMStagVecGetArray(dm,Xl,(void*)&vel);
    for (cj=es[1]; cj<es[1]+nele[1]; cj++) {
      for (ci=es[0]; ci<es[0]+nele[0]; ci++) {
        vel[cj][ci][slot_vx[0]] = 0.3;
        vel[cj][ci][slot_vx[1]] = 0.3;
        
        vel[cj][ci][slot_vy[0]] = 0.5;
        vel[cj][ci][slot_vy[1]] = 0.5;
      }
    }
    DMStagVecRestoreArray(dm,Xl,(void*)&vel);
    
    DMLocalToGlobalBegin(dm,Xl,INSERT_VALUES,X);
    DMLocalToGlobalEnd(dm,Xl,INSERT_VALUES,X);
  }
  
  
  for (k=1; k<21; k++) {
    PetscCall(MPoint_AdvectRK1(dmswarm,dm,X,0.05));
    if (k%4==0) {
      PetscCall(MPointCoordLayout_DomainFace(dmswarm,'s',0.0,10,COOR_APPEND));
    }
  }
  
  {
    PetscReal *pcoor;
    PetscInt npoints,p;
    PetscCall(DMSwarmGetLocalSize(dmswarm,&npoints));
    PetscCall(DMSwarmGetField(dmswarm,DMSwarmPICField_coor,NULL,NULL,(void**)&pcoor));
    for (p=0; p<npoints; p++) {
      printf("%d : %+1.4e %+1.4e\n",p,pcoor[2*p],pcoor[2*p+1]);
    }
    PetscCall(DMSwarmRestoreField(dmswarm,DMSwarmPICField_coor,NULL,NULL,(void**)&pcoor));
  }
  
  PetscCall(DMSwarmViewXDMF(dmswarm,"dmsB.xmf"));
  
  PetscCall(VecDestroy(&Xl));
  PetscCall(VecDestroy(&X));
  PetscCall(DMDestroy(&dmswarm));
  PetscCall(DMDestroy(&dm));
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
PetscErrorCode test_custom_tools_set(PetscInt nx,PetscInt ny)
{
  DM              dm,dmswarmA;
  PetscInt        dof0,dof1,dof2,stencilWidth;
  PetscFunctionBeginUser;
  
  dof0 = 0; dof1 = 1; dof2 = 1; /* (vertex) (face) (element) */
  stencilWidth = 1;
  
  PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, nx, ny,PETSC_DECIDE, PETSC_DECIDE, dof0, dof1, dof2,
                        DMSTAG_STENCIL_BOX, stencilWidth, NULL,NULL, &dm));
  PetscCall(DMStagSetCoordinateDMType(dm,DMPRODUCT));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  
  PetscCall(DMStagSetUniformCoordinatesProduct(dm,0.0,1.0,0.0,1.0,0.0,0.0));
  
  PetscCall(DMStagPICCreateDMSwarm(dm,&dmswarmA));
  PetscCall(DMSwarmRegisterPetscDatatypeField(dmswarmA,"eta",1,PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(dmswarmA,"xi",2,PETSC_REAL));
  PetscCall(DMStagPICFinalize(dmswarmA));
  
  {
    PetscInt ppcell[] = {1,1};
    PetscCall(MPointCoordLayout_DomainVolumeWithCellList(dmswarmA,0,NULL,0.3,ppcell,COOR_INITIALIZE));
  }
  
  PetscCall(DMSwarmFieldSet(dmswarmA,"eta",1.1));
  PetscCall(DMSwarmFieldSet(dmswarmA,"xi",2.2));
  
  {
    PetscReal *field;
    PetscInt npoints,p;
    PetscCall(DMSwarmGetLocalSize(dmswarmA,&npoints));
    PetscCall(DMSwarmGetField(dmswarmA,"eta",NULL,NULL,(void**)&field));
    for (p=0; p<npoints; p++) {
      printf("%d (eta) %+1.4e \n",p,field[p]);
    }
    PetscCall(DMSwarmRestoreField(dmswarmA,"eta",NULL,NULL,(void**)&field));
    PetscCall(DMSwarmGetField(dmswarmA,"xi",NULL,NULL,(void**)&field));
    for (p=0; p<npoints; p++) {
      printf("%d (xi) %+1.4e %+1.4e \n",p,field[2*p],field[2*p+1]);
    }
    PetscCall(DMSwarmRestoreField(dmswarmA,"xi",NULL,NULL,(void**)&field));
  }

  {
    PetscCall(DMSwarmFieldSetWithRange(dmswarmA,"eta",0,3,11.11));
    PetscCall(DMSwarmFieldSetWithRange(dmswarmA,"xi",10,16,22.22));
  }
  
  {
    PetscReal *field;
    PetscInt npoints,p;
    PetscCall(DMSwarmGetLocalSize(dmswarmA,&npoints));
    PetscCall(DMSwarmGetField(dmswarmA,"eta",NULL,NULL,(void**)&field));
    for (p=0; p<npoints; p++) {
      printf("%d (eta) %+1.4e \n",p,field[p]);
    }
    PetscCall(DMSwarmRestoreField(dmswarmA,"eta",NULL,NULL,(void**)&field));
    PetscCall(DMSwarmGetField(dmswarmA,"xi",NULL,NULL,(void**)&field));
    for (p=0; p<npoints; p++) {
      printf("%d (xi) %+1.4e %+1.4e \n",p,field[2*p],field[2*p+1]);
    }
    PetscCall(DMSwarmRestoreField(dmswarmA,"xi",NULL,NULL,(void**)&field));
  }

  {
    PetscInt list[] = {4,10,12,8};
    PetscCall(DMSwarmFieldSetWithList(dmswarmA,"eta",4,list,1001.11));
    PetscCall(DMSwarmFieldSetWithList(dmswarmA,"xi",4,list,2002.22));
  }
  
  {
    PetscReal *field;
    PetscInt npoints,p;
    PetscCall(DMSwarmGetLocalSize(dmswarmA,&npoints));
    PetscCall(DMSwarmGetField(dmswarmA,"eta",NULL,NULL,(void**)&field));
    for (p=0; p<npoints; p++) {
      printf("%d (eta) %+1.4e \n",p,field[p]);
    }
    PetscCall(DMSwarmRestoreField(dmswarmA,"eta",NULL,NULL,(void**)&field));
    PetscCall(DMSwarmGetField(dmswarmA,"xi",NULL,NULL,(void**)&field));
    for (p=0; p<npoints; p++) {
      printf("%d (xi) %+1.4e %+1.4e \n",p,field[2*p],field[2*p+1]);
    }
    PetscCall(DMSwarmRestoreField(dmswarmA,"xi",NULL,NULL,(void**)&field));
  }
  
  PetscCall(DMDestroy(&dmswarmA));
  PetscCall(DMDestroy(&dm));
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
PetscErrorCode test_custom_tools_dup_copy(PetscInt nx,PetscInt ny)
{
  DM              dm,dmswarmA,dmswarmB;
  PetscInt        dof0,dof1,dof2,stencilWidth;
  const char     *cellid;
  DMSwarmCellDM   celldm;
  PetscFunctionBeginUser;
  
  dof0 = 0; dof1 = 1; dof2 = 1; /* (vertex) (face) (element) */
  stencilWidth = 1;
  
  PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, nx, ny,PETSC_DECIDE, PETSC_DECIDE, dof0, dof1, dof2,
                        DMSTAG_STENCIL_BOX, stencilWidth, NULL,NULL, &dm));
  PetscCall(DMStagSetCoordinateDMType(dm,DMPRODUCT));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  
  PetscCall(DMStagSetUniformCoordinatesProduct(dm,0.0,1.0,0.0,1.0,0.0,0.0));
  
  PetscCall(DMStagPICCreateDMSwarm(dm,&dmswarmA));
  PetscCall(DMSwarmRegisterPetscDatatypeField(dmswarmA,"eta",1,PETSC_REAL));
  PetscCall(DMStagPICFinalize(dmswarmA));

  PetscCall(DMSwarmGetCellDMActive(dmswarmA, &celldm));
  PetscCall(DMSwarmCellDMGetCellID(celldm, &cellid));
  
  {
    PetscInt ppcell[] = {1,1};
    PetscCall(MPointCoordLayout_DomainVolumeWithCellList(dmswarmA,0,NULL,0.3,ppcell,COOR_INITIALIZE));
  }
  
  PetscCall(DMSwarmFieldSet(dmswarmA,"eta",1.1));
  
  {
    PetscReal *field;
    PetscInt npoints,p;
    PetscCall(DMSwarmGetLocalSize(dmswarmA,&npoints));
    PetscCall(DMSwarmGetField(dmswarmA,"eta",NULL,NULL,(void**)&field));
    for (p=0; p<npoints; p++) {
      field[p] = 1.0 + (PetscReal)p;
    }
    PetscCall(DMSwarmRestoreField(dmswarmA,"eta",NULL,NULL,(void**)&field));
  }

  {
    PetscInt *field;
    PetscInt npoints,p;
    PetscCall(DMSwarmGetLocalSize(dmswarmA,&npoints));
    PetscCall(DMSwarmGetField(dmswarmA,cellid,NULL,NULL,(void**)&field));
    for (p=0; p<npoints; p++) {
      field[p] = p;
    }
    PetscCall(DMSwarmRestoreField(dmswarmA,cellid,NULL,NULL,(void**)&field));
  }

  printf("<< A init >>\n");
  {
    const DM swarm = dmswarmA;
    PetscReal *field;
    PetscInt npoints,p;
    PetscInt *fieldpid;
    long *pid;
    
    PetscCall(DMSwarmGetLocalSize(swarm,&npoints));
    PetscCall(DMSwarmGetField(swarm,cellid,NULL,NULL,(void**)&fieldpid));
    PetscCall(DMSwarmGetField(swarm,"eta",NULL,NULL,(void**)&field));
    PetscCall(DMSwarmGetField(swarm,DMSwarmField_pid,NULL,NULL,(void**)&pid));
    for (p=0; p<npoints; p++) {
      printf("[swarmA] [%d] (pid) %ld (wil) %d (eta) %+1.4e \n",p,pid[p],fieldpid[p],field[p]);
    }
    PetscCall(DMSwarmRestoreField(swarm,DMSwarmField_pid,NULL,NULL,(void**)&pid));
    PetscCall(DMSwarmRestoreField(swarm,"eta",NULL,NULL,(void**)&field));
    PetscCall(DMSwarmRestoreField(swarm,cellid,NULL,NULL,(void**)&fieldpid));
  }

  /*
  {
    const DM swarm = dmswarmA;
    PetscReal *field;
    PetscInt npoints,p;
    PetscCall(DMSwarmGetLocalSize(swarm,&npoints));
    PetscCall(DMSwarmGetField(swarm,"eta",NULL,NULL,(void**)&field));
    for (p=0; p<npoints; p++) {
      printf("%d (eta) %+1.4e \n",p,field[p]);
    }
    PetscCall(DMSwarmRestoreField(swarm,"eta",NULL,NULL,(void**)&field));
    PetscCall(DMSwarmGetField(swarm,"xi",NULL,NULL,(void**)&field));
    for (p=0; p<npoints; p++) {
      printf("%d (xi) %+1.4e %+1.4e \n",p,field[2*p],field[2*p+1]);
    }
    PetscCall(DMSwarmRestoreField(swarm,"xi",NULL,NULL,(void**)&field));
  }
  */
  
  PetscCall(DMView(dmswarmA,PETSC_VIEWER_STDOUT_WORLD));
  
  PetscCall(DMStagPICCreateDMSwarm(dm,&dmswarmB));
  PetscCall(DMSwarmDuplicateRegisteredFields(dmswarmA,dmswarmB));
  PetscCall(DMSwarmRegisterPetscDatatypeField(dmswarmB,"xi",2,PETSC_REAL));
  PetscCall(DMStagPICFinalize(dmswarmB));
  PetscCall(DMSwarmSetLocalSizes(dmswarmB,0,-1));

  PetscCall(DMView(dmswarmB,PETSC_VIEWER_STDOUT_WORLD));

  {
    PetscBool copy_occurred;
    PetscInt list[] = { 4, 15, 12, 9};
    PetscCall(DMSwarmCopySubsetFieldValues(dmswarmA,4,list,dmswarmB,&copy_occurred));
  }
  
  PetscCall(DMSwarmFieldSet(dmswarmB,"xi",2.2));
  {
    const DM swarm = dmswarmB;
    PetscReal *field;
    PetscInt npoints,p;
    PetscCall(DMSwarmGetLocalSize(swarm,&npoints));
    PetscCall(DMSwarmGetField(swarm,"xi",NULL,NULL,(void**)&field));
    for (p=0; p<npoints; p++) {
      field[2*p]   = 200.0 + (PetscReal)(2*p+0);
      field[2*p+1] = 200.0 + (PetscReal)(2*p+1);
    }
    PetscCall(DMSwarmRestoreField(swarm,"xi",NULL,NULL,(void**)&field));
  }

  printf("<< B with copied eta : set xi >>\n");
  {
    const DM swarm = dmswarmB;
    PetscReal *field;
    PetscInt npoints,p;
    PetscInt *fieldpid;
    long *pid;
    
    PetscCall(DMSwarmGetLocalSize(swarm,&npoints));
    PetscCall(DMSwarmGetField(swarm,cellid,NULL,NULL,(void**)&fieldpid));
    PetscCall(DMSwarmGetField(swarm,"eta",NULL,NULL,(void**)&field));
    PetscCall(DMSwarmGetField(swarm,DMSwarmField_pid,NULL,NULL,(void**)&pid));
    for (p=0; p<npoints; p++) {
      printf("[swarmB] [%d] (pid) %ld (wil) %d (eta) %+1.4e \n",p,pid[p],fieldpid[p],field[p]);
    }
    PetscCall(DMSwarmRestoreField(swarm,"eta",NULL,NULL,(void**)&field));
    PetscCall(DMSwarmGetField(swarm,"xi",NULL,NULL,(void**)&field));
    for (p=0; p<npoints; p++) {
      printf("[swarmB] [%d] (pid) %ld (wil) %d (xi) %+1.4e %+1.4e \n",p,pid[p],fieldpid[p],field[2*p],field[2*p+1]);
    }
    PetscCall(DMSwarmRestoreField(swarm,DMSwarmField_pid,NULL,NULL,(void**)&pid));
    PetscCall(DMSwarmRestoreField(swarm,"xi",NULL,NULL,(void**)&field));
    PetscCall(DMSwarmRestoreField(swarm,cellid,NULL,NULL,(void**)&fieldpid));
  }

  /* change eta in B, delete list from A, insert B into A */
  {
    const DM swarm = dmswarmB;
    PetscReal *field;
    PetscInt npoints,p;
    PetscCall(DMSwarmGetLocalSize(swarm,&npoints));
    PetscCall(DMSwarmGetField(swarm,"eta",NULL,NULL,(void**)&field));
    for (p=0; p<npoints; p++) {
      field[p] = 1.0e3 + (PetscReal)p;
    }
    PetscCall(DMSwarmRestoreField(swarm,"eta",NULL,NULL,(void**)&field));
  }

  printf("<< A prior to deleting >>\n");
  {
    const DM swarm = dmswarmA;
    PetscReal *field;
    PetscInt npoints,p;
    PetscInt *fieldpid;
    long *pid;
    
    PetscCall(DMSwarmGetLocalSize(swarm,&npoints));
    PetscCall(DMSwarmGetField(swarm,cellid,NULL,NULL,(void**)&fieldpid));
    PetscCall(DMSwarmGetField(swarm,"eta",NULL,NULL,(void**)&field));
    PetscCall(DMSwarmGetField(swarm,DMSwarmField_pid,NULL,NULL,(void**)&pid));
    for (p=0; p<npoints; p++) {
      printf("[swarmA] [%d] (pid) %ld (wil) %d (eta) %+1.4e \n",p,pid[p],fieldpid[p],field[p]);
    }
    PetscCall(DMSwarmRestoreField(swarm,DMSwarmField_pid,NULL,NULL,(void**)&pid));
    PetscCall(DMSwarmRestoreField(swarm,"eta",NULL,NULL,(void**)&field));
    PetscCall(DMSwarmRestoreField(swarm,cellid,NULL,NULL,(void**)&fieldpid));
  }

  {
    PetscInt list[] = { 4, 15, 12, 9};
    PetscCall(DMSwarmRemovePoints(dmswarmA,4,list));
  }

  printf("<< A after deletion >>\n");
  {
    const DM swarm = dmswarmA;
    PetscReal *field;
    PetscInt npoints,p;
    PetscInt *fieldpid;
    long *pid;
    
    PetscCall(DMSwarmGetLocalSize(swarm,&npoints));
    PetscCall(DMSwarmGetField(swarm,cellid,NULL,NULL,(void**)&fieldpid));
    PetscCall(DMSwarmGetField(swarm,"eta",NULL,NULL,(void**)&field));
    PetscCall(DMSwarmGetField(swarm,DMSwarmField_pid,NULL,NULL,(void**)&pid));
    for (p=0; p<npoints; p++) {
      printf("[swarmA] [%d] (pid) %ld (wil) %d (eta) %+1.4e \n",p,pid[p],fieldpid[p],field[p]);
    }
    PetscCall(DMSwarmRestoreField(swarm,DMSwarmField_pid,NULL,NULL,(void**)&pid));
    PetscCall(DMSwarmRestoreField(swarm,"eta",NULL,NULL,(void**)&field));
    PetscCall(DMSwarmRestoreField(swarm,cellid,NULL,NULL,(void**)&fieldpid));
  }
  
  {
    PetscBool copy_occurred;
    PetscCall(DMSwarmCopyFieldValues(dmswarmB,dmswarmA,&copy_occurred));
  }

  printf("<< A after insertition >>\n");
  {
    const DM swarm = dmswarmA;
    PetscReal *field;
    PetscInt npoints,p;
    PetscInt *fieldpid;
    long *pid;
    
    PetscCall(DMSwarmGetLocalSize(swarm,&npoints));
    PetscCall(DMSwarmGetField(swarm,cellid,NULL,NULL,(void**)&fieldpid));
    PetscCall(DMSwarmGetField(swarm,"eta",NULL,NULL,(void**)&field));
    PetscCall(DMSwarmGetField(swarm,DMSwarmField_pid,NULL,NULL,(void**)&pid));
    for (p=0; p<npoints; p++) {
      printf("[swarmA] [%d] (pid) %ld (wil) %d (eta) %+1.4e \n",p,pid[p],fieldpid[p],field[p]);
    }
    PetscCall(DMSwarmRestoreField(swarm,DMSwarmField_pid,NULL,NULL,(void**)&pid));
    PetscCall(DMSwarmRestoreField(swarm,"eta",NULL,NULL,(void**)&field));
    PetscCall(DMSwarmRestoreField(swarm,cellid,NULL,NULL,(void**)&fieldpid));
  }

  PetscCall(DMDestroy(&dmswarmB));
  PetscCall(DMDestroy(&dmswarmA));
  PetscCall(DMDestroy(&dm));
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
PetscErrorCode test_custom_tools_project(PetscInt nx,PetscInt ny)
{
  DM              dmcell,dm,dmswarmA;
  PetscInt        dof0,dof1,dof2,stencilWidth;
  Vec             cellcoeff;
  //MPPropertyMap   property_labels[] = { {"eta",0} , {"rho",1} };
  PetscFunctionBeginUser;
  
  dof0 = 0; dof1 = 0; dof2 = 2; /* (vertex) (face) (element) */
  stencilWidth = 1;
  PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, nx, ny,PETSC_DECIDE, PETSC_DECIDE, dof0, dof1, dof2,
                        DMSTAG_STENCIL_BOX, stencilWidth, NULL,NULL, &dmcell));
  PetscCall(DMStagSetCoordinateDMType(dmcell,DMPRODUCT));
  PetscCall(DMSetFromOptions(dmcell));
  PetscCall(DMSetUp(dmcell));
  
  PetscCall(DMCreateGlobalVector(dmcell,&cellcoeff));
  
  dof0 = 0; dof1 = 1; dof2 = 1; /* (vertex) (face) (element) */
  stencilWidth = 1;
  PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, nx, ny,PETSC_DECIDE, PETSC_DECIDE, dof0, dof1, dof2,
                        DMSTAG_STENCIL_BOX, stencilWidth, NULL,NULL, &dm));
  PetscCall(DMStagSetCoordinateDMType(dm,DMPRODUCT));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  
  PetscCall(DMStagSetUniformCoordinatesProduct(dm,0.0,1.0,0.0,1.0,0.0,0.0));
  
  PetscCall(DMStagPICCreateDMSwarm(dm,&dmswarmA));
  PetscCall(DMSwarmRegisterPetscDatatypeField(dmswarmA,"eta",1,PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(dmswarmA,"rho",1,PETSC_REAL));
  PetscCall(DMStagPICFinalize(dmswarmA));
  
  {
    PetscInt ppcell[] = {1,1};
    PetscCall(MPointCoordLayout_DomainVolumeWithCellList(dmswarmA,0,NULL,0.3,ppcell,COOR_INITIALIZE));
  }
  
  {
    PetscReal *pcoor,*pfield;
    PetscInt npoints,p;
    
    PetscCall(DMSwarmGetLocalSize(dmswarmA,&npoints));
    PetscCall(DMSwarmGetField(dmswarmA,DMSwarmPICField_coor,NULL,NULL,(void**)&pcoor));
    
    PetscCall(DMSwarmGetField(dmswarmA,"eta",NULL,NULL,(void**)&pfield));
    for (p=0; p<npoints; p++) {
      pfield[p] = 1.0;
      if (pcoor[2*p+1] < 0.3) {
        pfield[p] = 3.0;
      }
    }
    PetscCall(DMSwarmRestoreField(dmswarmA,"eta",NULL,NULL,(void**)&pfield));
    
    PetscCall(DMSwarmGetField(dmswarmA,"rho",NULL,NULL,(void**)&pfield));
    for (p=0; p<npoints; p++) {
      pfield[p] = 120.0;
      if (pcoor[2*p+0] < 0.5) {
        pfield[p] = 33.0;
      }
    }
    PetscCall(DMSwarmRestoreField(dmswarmA,"rho",NULL,NULL,(void**)&pfield));
    PetscCall(DMSwarmRestoreField(dmswarmA,DMSwarmPICField_coor,NULL,NULL,(void**)&pcoor));
  }
  
  PetscCall(MPoint_ProjectP0_arith(dmswarmA,"eta",dm,dmcell,0,cellcoeff));
  VecView(cellcoeff,PETSC_VIEWER_STDOUT_WORLD);
  PetscCall(MPoint_ProjectP0_arith(dmswarmA,"rho",dm,dmcell,1,cellcoeff));
  VecView(cellcoeff,PETSC_VIEWER_STDOUT_WORLD);
  
  PetscCall(VecDestroy(&cellcoeff));
  PetscCall(DMDestroy(&dmswarmA));
  PetscCall(DMDestroy(&dm));
  PetscCall(DMDestroy(&dmcell));
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
PetscErrorCode test_custom_tools_project_2(PetscInt nx,PetscInt ny)
{
  DM              dmcell,dm,dmswarmA;
  PetscInt        dof0,dof1,dof2,stencilWidth;
  Vec             cellcoeff;
  PetscFunctionBeginUser;
  
  dof0 = 1; dof1 = 1; dof2 = 1; /* (vertex) (face) (element) */
  stencilWidth = 1;
  PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, nx, ny,PETSC_DECIDE, PETSC_DECIDE, dof0, dof1, dof2,
                        DMSTAG_STENCIL_BOX, stencilWidth, NULL,NULL, &dmcell));
  PetscCall(DMStagSetCoordinateDMType(dmcell,DMPRODUCT));
  PetscCall(DMSetFromOptions(dmcell));
  PetscCall(DMSetUp(dmcell));
  
  PetscCall(DMCreateGlobalVector(dmcell,&cellcoeff));
  
  dof0 = 0; dof1 = 1; dof2 = 1; /* (vertex) (face) (element) */
  stencilWidth = 1;
  PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, nx, ny,PETSC_DECIDE, PETSC_DECIDE, dof0, dof1, dof2,
                        DMSTAG_STENCIL_BOX, stencilWidth, NULL,NULL, &dm));
  PetscCall(DMStagSetCoordinateDMType(dm,DMPRODUCT));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  
  PetscCall(DMStagSetUniformCoordinatesProduct(dm,0.0,1.0,0.0,1.0,0.0,0.0));
  
  PetscCall(DMStagPICCreateDMSwarm(dm,&dmswarmA));
  PetscCall(DMSwarmRegisterPetscDatatypeField(dmswarmA,"eta",1,PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(dmswarmA,"rho",1,PETSC_REAL));
  PetscCall(DMStagPICFinalize(dmswarmA));
  
  {
    PetscInt ppcell[] = {1,1};
    PetscCall(MPointCoordLayout_DomainVolumeWithCellList(dmswarmA,0,NULL,0.3,ppcell,COOR_INITIALIZE));
  }
  
  {
    PetscReal *pcoor,*pfield;
    PetscInt npoints,p;
    
    PetscCall(DMSwarmGetLocalSize(dmswarmA,&npoints));
    PetscCall(DMSwarmGetField(dmswarmA,DMSwarmPICField_coor,NULL,NULL,(void**)&pcoor));
    
    PetscCall(DMSwarmGetField(dmswarmA,"eta",NULL,NULL,(void**)&pfield));
    for (p=0; p<npoints; p++) {
      pfield[p] = 1.0;
      if (pcoor[2*p+1] < 0.3) {
        pfield[p] = 3.0;
      }
    }
    PetscCall(DMSwarmRestoreField(dmswarmA,"eta",NULL,NULL,(void**)&pfield));
    
    
    PetscCall(DMSwarmGetField(dmswarmA,"rho",NULL,NULL,(void**)&pfield));
    for (p=0; p<npoints; p++) {
      pfield[p] = 120.0;
      if (pcoor[2*p+0] < 0.5) {
        pfield[p] = 33.0;
      }
    }
    PetscCall(DMSwarmRestoreField(dmswarmA,"rho",NULL,NULL,(void**)&pfield));
    PetscCall(DMSwarmRestoreField(dmswarmA,DMSwarmPICField_coor,NULL,NULL,(void**)&pcoor));
  }
  
  PetscCall(MPoint_ProjectQ1_arith_general(dmswarmA,"eta",dm,dmcell,0,0,cellcoeff));
  VecView(cellcoeff,PETSC_VIEWER_STDOUT_WORLD);
  
  PetscCall(MPoint_ProjectQ1_arith_general(dmswarmA,"eta",dm,dmcell,1,0,cellcoeff));
  VecView(cellcoeff,PETSC_VIEWER_STDOUT_WORLD);
  
  PetscCall(MPoint_ProjectQ1_arith_general(dmswarmA,"eta",dm,dmcell,2,0,cellcoeff));
  VecView(cellcoeff,PETSC_VIEWER_STDOUT_WORLD);
  
  PetscCall(VecDestroy(&cellcoeff));
  PetscCall(DMDestroy(&dmswarmA));
  PetscCall(DMDestroy(&dm));
  PetscCall(DMDestroy(&dmcell));
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "main"
int main (int argc,char **argv)
{
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  // PetscCall(test_layout(6,4));
  // PetscCall(test_advection_rk1(10,10));
  // PetscCall(test_custom_tools_set(4,4));
  // PetscCall(test_custom_tools_dup_copy(4,4));
  // PetscCall(test_custom_tools_project(4,4));
  PetscCall(test_custom_tools_project_2(4,4));
  PetscCall(PetscFinalize());
  return 0;
}
