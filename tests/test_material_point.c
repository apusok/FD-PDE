static char help[] = "Material point layout test \n\n";


#include "petsc.h"
#include "../material_point.h"


PetscErrorCode test_layout(PetscInt nx,PetscInt ny)
{
  DM              dm,dmswarm;
  PetscInt        dof0,dof1,dof2,stencilWidth;
  PetscErrorCode  ierr;
  
  dof0 = 0; dof1 = 1; dof2 = 1; /* (vertex) (face) (element) */
  stencilWidth = 1;
  
  ierr = DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, nx, ny,
                        PETSC_DECIDE, PETSC_DECIDE, dof0, dof1, dof2,
                        DMSTAG_STENCIL_BOX, stencilWidth, NULL,NULL, &dm);CHKERRQ(ierr);
  ierr = DMStagSetCoordinateDMType(dm,DMPRODUCT);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMSetUp(dm);CHKERRQ(ierr);
  
  ierr = DMStagSetUniformCoordinatesProduct(dm,0.0,1.0,0.0,1.0,0.0,0.0);CHKERRQ(ierr);

  ierr = DMStagPICCreateDMSwarm(dm,&dmswarm);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(dmswarm,"eta",1,PETSC_REAL);CHKERRQ(ierr);
  ierr = DMStagPICFinalize(dmswarm);CHKERRQ(ierr);

  /*ierr = MPointCoordLayout_DomainVolume(dmswarm,0.0,1,COOR_INITIALIZE);CHKERRQ(ierr);*/
  {
    PetscInt ppcell[] = {1,1};
    ierr = MPointCoordLayout_DomainVolumeWithCellList(dmswarm,0,NULL,0.0,ppcell,COOR_INITIALIZE);CHKERRQ(ierr);
    
    ppcell[0] = ppcell[1] = 2;
    ierr = MPointCoordLayout_DomainVolumeWithCellList(dmswarm,0,NULL,0.0,ppcell,COOR_APPEND);CHKERRQ(ierr);
    
    ierr = MPointCoordLayout_DomainFace(dmswarm,'n',0.0,6,COOR_APPEND);CHKERRQ(ierr);
    ierr = MPointCoordLayout_DomainFace(dmswarm,'s',0.0,10,COOR_APPEND);CHKERRQ(ierr);
    
    ierr = MPointCoordLayout_DomainFace(dmswarm,'w',0.0,1,COOR_APPEND);CHKERRQ(ierr);
    ierr = MPointCoordLayout_DomainFace(dmswarm,'e',0.0,1,COOR_APPEND);CHKERRQ(ierr);
  }

  {
    PetscReal *pcoor;
    PetscInt npoints,p;
    ierr = DMSwarmGetLocalSize(dmswarm,&npoints);CHKERRQ(ierr);
    ierr = DMSwarmGetField(dmswarm,DMSwarmPICField_coor,NULL,NULL,(void**)&pcoor);CHKERRQ(ierr);
    for (p=0; p<npoints; p++) {
      printf("%d : %+1.4e %+1.4e\n",p,pcoor[2*p],pcoor[2*p+1]);
    }
    ierr = DMSwarmRestoreField(dmswarm,DMSwarmPICField_coor,NULL,NULL,(void**)&pcoor);CHKERRQ(ierr);
  }
  
  ierr = DMSwarmViewXDMF(dmswarm,"dms.xmf");CHKERRQ(ierr);

  ierr = DMDestroy(&dmswarm);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}


PetscErrorCode test_advection_rk1(PetscInt nx,PetscInt ny)
{
  DM              dm,dmswarm;
  PetscInt        dof0,dof1,dof2,stencilWidth,k;
  Vec             X,Xl;
  PetscReal       ***vel;
  PetscErrorCode  ierr;
  
  dof0 = 0; dof1 = 1; dof2 = 1; /* (vertex) (face) (element) */
  stencilWidth = 1;
  
  ierr = DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, nx, ny,
                        PETSC_DECIDE, PETSC_DECIDE, dof0, dof1, dof2,
                        DMSTAG_STENCIL_BOX, stencilWidth, NULL,NULL, &dm);CHKERRQ(ierr);
  ierr = DMStagSetCoordinateDMType(dm,DMPRODUCT);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMSetUp(dm);CHKERRQ(ierr);
  
  ierr = DMStagSetUniformCoordinatesProduct(dm,0.0,1.0,0.0,1.0,0.0,0.0);CHKERRQ(ierr);
  
  ierr = DMStagPICCreateDMSwarm(dm,&dmswarm);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(dmswarm,"eta",1,PETSC_REAL);CHKERRQ(ierr);
  ierr = DMStagPICFinalize(dmswarm);CHKERRQ(ierr);
  
  /*ierr = MPointCoordLayout_DomainVolume(dmswarm,0.0,1,COOR_INITIALIZE);CHKERRQ(ierr);*/
  {
    PetscInt ppcell[] = {1,1};
    ierr = MPointCoordLayout_DomainVolumeWithCellList(dmswarm,0,NULL,0.3,ppcell,COOR_INITIALIZE);CHKERRQ(ierr);
    
    ppcell[0] = ppcell[1] = 2;
    ierr = MPointCoordLayout_DomainVolumeWithCellList(dmswarm,0,NULL,0.3,ppcell,COOR_APPEND);CHKERRQ(ierr);
    
    ierr = MPointCoordLayout_DomainFace(dmswarm,'n',0.0,6,COOR_APPEND);CHKERRQ(ierr);
    ierr = MPointCoordLayout_DomainFace(dmswarm,'s',0.0,10,COOR_APPEND);CHKERRQ(ierr);
    
    ierr = MPointCoordLayout_DomainFace(dmswarm,'w',0.0,1,COOR_APPEND);CHKERRQ(ierr);
    ierr = MPointCoordLayout_DomainFace(dmswarm,'e',0.0,1,COOR_APPEND);CHKERRQ(ierr);
  }

  ierr = DMSwarmViewXDMF(dmswarm,"dmsA.xmf");CHKERRQ(ierr);
  
  
  {
    PetscInt slot_vx[2],slot_vy[2],ci,cj,es[2],nele[2];
    
    ierr = DMCreateGlobalVector(dm,&X);CHKERRQ(ierr);
    ierr = DMCreateLocalVector(dm,&Xl);CHKERRQ(ierr);
    ierr = DMStagGetCorners(dm,&es[0],&es[1],NULL,&nele[0],&nele[1],NULL,NULL,NULL,NULL);
    
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
    ierr = MPoint_AdvectRK1(dmswarm,dm,X,0.05);CHKERRQ(ierr);
    if (k%4==0) {
      ierr = MPointCoordLayout_DomainFace(dmswarm,'s',0.0,10,COOR_APPEND);CHKERRQ(ierr);
    }
  }
  
  {
    PetscReal *pcoor;
    PetscInt npoints,p;
    ierr = DMSwarmGetLocalSize(dmswarm,&npoints);CHKERRQ(ierr);
    ierr = DMSwarmGetField(dmswarm,DMSwarmPICField_coor,NULL,NULL,(void**)&pcoor);CHKERRQ(ierr);
    for (p=0; p<npoints; p++) {
      printf("%d : %+1.4e %+1.4e\n",p,pcoor[2*p],pcoor[2*p+1]);
    }
    ierr = DMSwarmRestoreField(dmswarm,DMSwarmPICField_coor,NULL,NULL,(void**)&pcoor);CHKERRQ(ierr);
  }
  
  ierr = DMSwarmViewXDMF(dmswarm,"dmsB.xmf");CHKERRQ(ierr);
  
  ierr = VecDestroy(&Xl);CHKERRQ(ierr);
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = DMDestroy(&dmswarm);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode test_custom_tools_set(PetscInt nx,PetscInt ny)
{
  DM              dm,dmswarmA;
  PetscInt        dof0,dof1,dof2,stencilWidth;
  PetscErrorCode  ierr;
  
  dof0 = 0; dof1 = 1; dof2 = 1; /* (vertex) (face) (element) */
  stencilWidth = 1;
  
  ierr = DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, nx, ny,
                        PETSC_DECIDE, PETSC_DECIDE, dof0, dof1, dof2,
                        DMSTAG_STENCIL_BOX, stencilWidth, NULL,NULL, &dm);CHKERRQ(ierr);
  ierr = DMStagSetCoordinateDMType(dm,DMPRODUCT);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMSetUp(dm);CHKERRQ(ierr);
  
  ierr = DMStagSetUniformCoordinatesProduct(dm,0.0,1.0,0.0,1.0,0.0,0.0);CHKERRQ(ierr);
  
  ierr = DMStagPICCreateDMSwarm(dm,&dmswarmA);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(dmswarmA,"eta",1,PETSC_REAL);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(dmswarmA,"xi",2,PETSC_REAL);CHKERRQ(ierr);
  ierr = DMStagPICFinalize(dmswarmA);CHKERRQ(ierr);
  
  {
    PetscInt ppcell[] = {1,1};
    ierr = MPointCoordLayout_DomainVolumeWithCellList(dmswarmA,0,NULL,0.3,ppcell,COOR_INITIALIZE);CHKERRQ(ierr);
  }
  
  ierr = DMSwarmFieldSet(dmswarmA,"eta",1.1);
  ierr = DMSwarmFieldSet(dmswarmA,"xi",2.2);
  
  {
    PetscReal *field;
    PetscInt npoints,p;
    ierr = DMSwarmGetLocalSize(dmswarmA,&npoints);CHKERRQ(ierr);
    ierr = DMSwarmGetField(dmswarmA,"eta",NULL,NULL,(void**)&field);CHKERRQ(ierr);
    for (p=0; p<npoints; p++) {
      printf("%d (eta) %+1.4e \n",p,field[p]);
    }
    ierr = DMSwarmRestoreField(dmswarmA,"eta",NULL,NULL,(void**)&field);CHKERRQ(ierr);
    ierr = DMSwarmGetField(dmswarmA,"xi",NULL,NULL,(void**)&field);CHKERRQ(ierr);
    for (p=0; p<npoints; p++) {
      printf("%d (xi) %+1.4e %+1.4e \n",p,field[2*p],field[2*p+1]);
    }
    ierr = DMSwarmRestoreField(dmswarmA,"xi",NULL,NULL,(void**)&field);CHKERRQ(ierr);
  }

  {
    ierr = DMSwarmFieldSetWithRange(dmswarmA,"eta",0,3,11.11);
    ierr = DMSwarmFieldSetWithRange(dmswarmA,"xi",10,16,22.22);
  }
  
  {
    PetscReal *field;
    PetscInt npoints,p;
    ierr = DMSwarmGetLocalSize(dmswarmA,&npoints);CHKERRQ(ierr);
    ierr = DMSwarmGetField(dmswarmA,"eta",NULL,NULL,(void**)&field);CHKERRQ(ierr);
    for (p=0; p<npoints; p++) {
      printf("%d (eta) %+1.4e \n",p,field[p]);
    }
    ierr = DMSwarmRestoreField(dmswarmA,"eta",NULL,NULL,(void**)&field);CHKERRQ(ierr);
    ierr = DMSwarmGetField(dmswarmA,"xi",NULL,NULL,(void**)&field);CHKERRQ(ierr);
    for (p=0; p<npoints; p++) {
      printf("%d (xi) %+1.4e %+1.4e \n",p,field[2*p],field[2*p+1]);
    }
    ierr = DMSwarmRestoreField(dmswarmA,"xi",NULL,NULL,(void**)&field);CHKERRQ(ierr);
  }


  
  {
    PetscInt list[] = {4,10,12,8};
    ierr = DMSwarmFieldSetWithList(dmswarmA,"eta",4,list,1001.11);
    ierr = DMSwarmFieldSetWithList(dmswarmA,"xi",4,list,2002.22);
  }
  
  {
    PetscReal *field;
    PetscInt npoints,p;
    ierr = DMSwarmGetLocalSize(dmswarmA,&npoints);CHKERRQ(ierr);
    ierr = DMSwarmGetField(dmswarmA,"eta",NULL,NULL,(void**)&field);CHKERRQ(ierr);
    for (p=0; p<npoints; p++) {
      printf("%d (eta) %+1.4e \n",p,field[p]);
    }
    ierr = DMSwarmRestoreField(dmswarmA,"eta",NULL,NULL,(void**)&field);CHKERRQ(ierr);
    ierr = DMSwarmGetField(dmswarmA,"xi",NULL,NULL,(void**)&field);CHKERRQ(ierr);
    for (p=0; p<npoints; p++) {
      printf("%d (xi) %+1.4e %+1.4e \n",p,field[2*p],field[2*p+1]);
    }
    ierr = DMSwarmRestoreField(dmswarmA,"xi",NULL,NULL,(void**)&field);CHKERRQ(ierr);
  }
  
  ierr = DMDestroy(&dmswarmA);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode test_custom_tools_dup_copy(PetscInt nx,PetscInt ny)
{
  DM              dm,dmswarmA,dmswarmB;
  PetscInt        dof0,dof1,dof2,stencilWidth;
  PetscErrorCode  ierr;
  
  dof0 = 0; dof1 = 1; dof2 = 1; /* (vertex) (face) (element) */
  stencilWidth = 1;
  
  ierr = DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, nx, ny,
                        PETSC_DECIDE, PETSC_DECIDE, dof0, dof1, dof2,
                        DMSTAG_STENCIL_BOX, stencilWidth, NULL,NULL, &dm);CHKERRQ(ierr);
  ierr = DMStagSetCoordinateDMType(dm,DMPRODUCT);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMSetUp(dm);CHKERRQ(ierr);
  
  ierr = DMStagSetUniformCoordinatesProduct(dm,0.0,1.0,0.0,1.0,0.0,0.0);CHKERRQ(ierr);
  
  ierr = DMStagPICCreateDMSwarm(dm,&dmswarmA);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(dmswarmA,"eta",1,PETSC_REAL);CHKERRQ(ierr);
  ierr = DMStagPICFinalize(dmswarmA);CHKERRQ(ierr);
  
  {
    PetscInt ppcell[] = {1,1};
    ierr = MPointCoordLayout_DomainVolumeWithCellList(dmswarmA,0,NULL,0.3,ppcell,COOR_INITIALIZE);CHKERRQ(ierr);
  }
  
  ierr = DMSwarmFieldSet(dmswarmA,"eta",1.1);
  
  {
    PetscReal *field;
    PetscInt npoints,p;
    ierr = DMSwarmGetLocalSize(dmswarmA,&npoints);CHKERRQ(ierr);
    ierr = DMSwarmGetField(dmswarmA,"eta",NULL,NULL,(void**)&field);CHKERRQ(ierr);
    for (p=0; p<npoints; p++) {
      field[p] = 1.0 + (PetscReal)p;
    }
    ierr = DMSwarmRestoreField(dmswarmA,"eta",NULL,NULL,(void**)&field);CHKERRQ(ierr);
  }

  {
    PetscInt *field;
    PetscInt npoints,p;
    ierr = DMSwarmGetLocalSize(dmswarmA,&npoints);CHKERRQ(ierr);
    ierr = DMSwarmGetField(dmswarmA,DMSwarmPICField_cellid,NULL,NULL,(void**)&field);CHKERRQ(ierr);
    for (p=0; p<npoints; p++) {
      field[p] = p;
    }
    ierr = DMSwarmRestoreField(dmswarmA,DMSwarmPICField_cellid,NULL,NULL,(void**)&field);CHKERRQ(ierr);
  }

  printf("<< A init >>\n");
  {
    const DM swarm = dmswarmA;
    PetscReal *field;
    PetscInt npoints,p;
    PetscInt *fieldpid;
    long *pid;
    
    ierr = DMSwarmGetLocalSize(swarm,&npoints);CHKERRQ(ierr);
    ierr = DMSwarmGetField(swarm,DMSwarmPICField_cellid,NULL,NULL,(void**)&fieldpid);CHKERRQ(ierr);
    ierr = DMSwarmGetField(swarm,"eta",NULL,NULL,(void**)&field);CHKERRQ(ierr);
    ierr = DMSwarmGetField(swarm,DMSwarmField_pid,NULL,NULL,(void**)&pid);CHKERRQ(ierr);
    for (p=0; p<npoints; p++) {
      printf("[swarmA] [%d] (pid) %ld (wil) %d (eta) %+1.4e \n",p,pid[p],fieldpid[p],field[p]);
    }
    ierr = DMSwarmRestoreField(swarm,DMSwarmField_pid,NULL,NULL,(void**)&pid);CHKERRQ(ierr);
    ierr = DMSwarmRestoreField(swarm,"eta",NULL,NULL,(void**)&field);CHKERRQ(ierr);
    ierr = DMSwarmRestoreField(swarm,DMSwarmPICField_cellid,NULL,NULL,(void**)&fieldpid);CHKERRQ(ierr);
  }

  
  /*
  {
    const DM swarm = dmswarmA;
    PetscReal *field;
    PetscInt npoints,p;
    ierr = DMSwarmGetLocalSize(swarm,&npoints);CHKERRQ(ierr);
    ierr = DMSwarmGetField(swarm,"eta",NULL,NULL,(void**)&field);CHKERRQ(ierr);
    for (p=0; p<npoints; p++) {
      printf("%d (eta) %+1.4e \n",p,field[p]);
    }
    ierr = DMSwarmRestoreField(swarm,"eta",NULL,NULL,(void**)&field);CHKERRQ(ierr);
    ierr = DMSwarmGetField(swarm,"xi",NULL,NULL,(void**)&field);CHKERRQ(ierr);
    for (p=0; p<npoints; p++) {
      printf("%d (xi) %+1.4e %+1.4e \n",p,field[2*p],field[2*p+1]);
    }
    ierr = DMSwarmRestoreField(swarm,"xi",NULL,NULL,(void**)&field);CHKERRQ(ierr);
  }
  */
  
  ierr = DMView(dmswarmA,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  
  ierr = DMStagPICCreateDMSwarm(dm,&dmswarmB);CHKERRQ(ierr);
  ierr = DMSwarmDuplicateRegisteredFields(dmswarmA,dmswarmB);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(dmswarmB,"xi",2,PETSC_REAL);CHKERRQ(ierr);
  ierr = DMStagPICFinalize(dmswarmB);CHKERRQ(ierr);
  ierr = DMSwarmSetLocalSizes(dmswarmB,0,-1);CHKERRQ(ierr);

  ierr = DMView(dmswarmB,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  {
    PetscBool copy_occurred;
    PetscInt list[] = { 4, 15, 12, 9};
    ierr = DMSwarmCopySubsetFieldValues(dmswarmA,4,list,dmswarmB,&copy_occurred);CHKERRQ(ierr);
  }
  
  ierr = DMSwarmFieldSet(dmswarmB,"xi",2.2);
  {
    const DM swarm = dmswarmB;
    PetscReal *field;
    PetscInt npoints,p;
    ierr = DMSwarmGetLocalSize(swarm,&npoints);CHKERRQ(ierr);
    ierr = DMSwarmGetField(swarm,"xi",NULL,NULL,(void**)&field);CHKERRQ(ierr);
    for (p=0; p<npoints; p++) {
      field[2*p]   = 200.0 + (PetscReal)(2*p+0);
      field[2*p+1] = 200.0 + (PetscReal)(2*p+1);
    }
    ierr = DMSwarmRestoreField(swarm,"xi",NULL,NULL,(void**)&field);CHKERRQ(ierr);
  }

  printf("<< B with copied eta : set xi >>\n");
  {
    const DM swarm = dmswarmB;
    PetscReal *field;
    PetscInt npoints,p;
    PetscInt *fieldpid;
    long *pid;
    
    ierr = DMSwarmGetLocalSize(swarm,&npoints);CHKERRQ(ierr);
    ierr = DMSwarmGetField(swarm,DMSwarmPICField_cellid,NULL,NULL,(void**)&fieldpid);CHKERRQ(ierr);
    ierr = DMSwarmGetField(swarm,"eta",NULL,NULL,(void**)&field);CHKERRQ(ierr);
    ierr = DMSwarmGetField(swarm,DMSwarmField_pid,NULL,NULL,(void**)&pid);CHKERRQ(ierr);
    for (p=0; p<npoints; p++) {
      printf("[swarmB] [%d] (pid) %ld (wil) %d (eta) %+1.4e \n",p,pid[p],fieldpid[p],field[p]);
    }
    ierr = DMSwarmRestoreField(swarm,"eta",NULL,NULL,(void**)&field);CHKERRQ(ierr);
    ierr = DMSwarmGetField(swarm,"xi",NULL,NULL,(void**)&field);CHKERRQ(ierr);
    for (p=0; p<npoints; p++) {
      printf("[swarmB] [%d] (pid) %ld (wil) %d (xi) %+1.4e %+1.4e \n",p,pid[p],fieldpid[p],field[2*p],field[2*p+1]);
    }
    ierr = DMSwarmRestoreField(swarm,DMSwarmField_pid,NULL,NULL,(void**)&pid);CHKERRQ(ierr);
    ierr = DMSwarmRestoreField(swarm,"xi",NULL,NULL,(void**)&field);CHKERRQ(ierr);
    ierr = DMSwarmRestoreField(swarm,DMSwarmPICField_cellid,NULL,NULL,(void**)&fieldpid);CHKERRQ(ierr);
  }
  

  /* change eta in B, delete list from A, insert B into A */
  {
    const DM swarm = dmswarmB;
    PetscReal *field;
    PetscInt npoints,p;
    ierr = DMSwarmGetLocalSize(swarm,&npoints);CHKERRQ(ierr);
    ierr = DMSwarmGetField(swarm,"eta",NULL,NULL,(void**)&field);CHKERRQ(ierr);
    for (p=0; p<npoints; p++) {
      field[p] = 1.0e3 + (PetscReal)p;
    }
    ierr = DMSwarmRestoreField(swarm,"eta",NULL,NULL,(void**)&field);CHKERRQ(ierr);
  }

  
  printf("<< A prior to deleting >>\n");
  {
    const DM swarm = dmswarmA;
    PetscReal *field;
    PetscInt npoints,p;
    PetscInt *fieldpid;
    long *pid;
    
    ierr = DMSwarmGetLocalSize(swarm,&npoints);CHKERRQ(ierr);
    ierr = DMSwarmGetField(swarm,DMSwarmPICField_cellid,NULL,NULL,(void**)&fieldpid);CHKERRQ(ierr);
    ierr = DMSwarmGetField(swarm,"eta",NULL,NULL,(void**)&field);CHKERRQ(ierr);
    ierr = DMSwarmGetField(swarm,DMSwarmField_pid,NULL,NULL,(void**)&pid);CHKERRQ(ierr);
    for (p=0; p<npoints; p++) {
      printf("[swarmA] [%d] (pid) %ld (wil) %d (eta) %+1.4e \n",p,pid[p],fieldpid[p],field[p]);
    }
    ierr = DMSwarmRestoreField(swarm,DMSwarmField_pid,NULL,NULL,(void**)&pid);CHKERRQ(ierr);
    ierr = DMSwarmRestoreField(swarm,"eta",NULL,NULL,(void**)&field);CHKERRQ(ierr);
    ierr = DMSwarmRestoreField(swarm,DMSwarmPICField_cellid,NULL,NULL,(void**)&fieldpid);CHKERRQ(ierr);
  }

  {
    PetscInt list[] = { 4, 15, 12, 9};
    ierr = DMSwarmRemovePoints(dmswarmA,4,list);CHKERRQ(ierr);
  }

  printf("<< A after deletion >>\n");
  {
    const DM swarm = dmswarmA;
    PetscReal *field;
    PetscInt npoints,p;
    PetscInt *fieldpid;
    long *pid;
    
    ierr = DMSwarmGetLocalSize(swarm,&npoints);CHKERRQ(ierr);
    ierr = DMSwarmGetField(swarm,DMSwarmPICField_cellid,NULL,NULL,(void**)&fieldpid);CHKERRQ(ierr);
    ierr = DMSwarmGetField(swarm,"eta",NULL,NULL,(void**)&field);CHKERRQ(ierr);
    ierr = DMSwarmGetField(swarm,DMSwarmField_pid,NULL,NULL,(void**)&pid);CHKERRQ(ierr);
    for (p=0; p<npoints; p++) {
      printf("[swarmA] [%d] (pid) %ld (wil) %d (eta) %+1.4e \n",p,pid[p],fieldpid[p],field[p]);
    }
    ierr = DMSwarmRestoreField(swarm,DMSwarmField_pid,NULL,NULL,(void**)&pid);CHKERRQ(ierr);
    ierr = DMSwarmRestoreField(swarm,"eta",NULL,NULL,(void**)&field);CHKERRQ(ierr);
    ierr = DMSwarmRestoreField(swarm,DMSwarmPICField_cellid,NULL,NULL,(void**)&fieldpid);CHKERRQ(ierr);
  }
  
  
  {
    PetscBool copy_occurred;
    ierr = DMSwarmCopyFieldValues(dmswarmB,dmswarmA,&copy_occurred);CHKERRQ(ierr);
  }

  printf("<< A after insertition >>\n");
  {
    const DM swarm = dmswarmA;
    PetscReal *field;
    PetscInt npoints,p;
    PetscInt *fieldpid;
    long *pid;
    
    ierr = DMSwarmGetLocalSize(swarm,&npoints);CHKERRQ(ierr);
    ierr = DMSwarmGetField(swarm,DMSwarmPICField_cellid,NULL,NULL,(void**)&fieldpid);CHKERRQ(ierr);
    ierr = DMSwarmGetField(swarm,"eta",NULL,NULL,(void**)&field);CHKERRQ(ierr);
    ierr = DMSwarmGetField(swarm,DMSwarmField_pid,NULL,NULL,(void**)&pid);CHKERRQ(ierr);
    for (p=0; p<npoints; p++) {
      printf("[swarmA] [%d] (pid) %ld (wil) %d (eta) %+1.4e \n",p,pid[p],fieldpid[p],field[p]);
    }
    ierr = DMSwarmRestoreField(swarm,DMSwarmField_pid,NULL,NULL,(void**)&pid);CHKERRQ(ierr);
    ierr = DMSwarmRestoreField(swarm,"eta",NULL,NULL,(void**)&field);CHKERRQ(ierr);
    ierr = DMSwarmRestoreField(swarm,DMSwarmPICField_cellid,NULL,NULL,(void**)&fieldpid);CHKERRQ(ierr);
  }

  
  ierr = DMDestroy(&dmswarmB);CHKERRQ(ierr);
  ierr = DMDestroy(&dmswarmA);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode test_custom_tools_project(PetscInt nx,PetscInt ny)
{
  DM              dmcell,dm,dmswarmA;
  PetscInt        dof0,dof1,dof2,stencilWidth;
  Vec             cellcoeff;
  PetscErrorCode  ierr;
  //MPPropertyMap   property_labels[] = { {"eta",0} , {"rho",1} };
  
  dof0 = 0; dof1 = 0; dof2 = 2; /* (vertex) (face) (element) */
  stencilWidth = 1;
  ierr = DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, nx, ny,
                        PETSC_DECIDE, PETSC_DECIDE, dof0, dof1, dof2,
                        DMSTAG_STENCIL_BOX, stencilWidth, NULL,NULL, &dmcell);CHKERRQ(ierr);
  ierr = DMStagSetCoordinateDMType(dmcell,DMPRODUCT);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dmcell);CHKERRQ(ierr);
  ierr = DMSetUp(dmcell);CHKERRQ(ierr);
  
  ierr = DMCreateGlobalVector(dmcell,&cellcoeff);CHKERRQ(ierr);
  
  dof0 = 0; dof1 = 1; dof2 = 1; /* (vertex) (face) (element) */
  stencilWidth = 1;
  ierr = DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, nx, ny,
                        PETSC_DECIDE, PETSC_DECIDE, dof0, dof1, dof2,
                        DMSTAG_STENCIL_BOX, stencilWidth, NULL,NULL, &dm);CHKERRQ(ierr);
  ierr = DMStagSetCoordinateDMType(dm,DMPRODUCT);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMSetUp(dm);CHKERRQ(ierr);
  
  ierr = DMStagSetUniformCoordinatesProduct(dm,0.0,1.0,0.0,1.0,0.0,0.0);CHKERRQ(ierr);
  
  ierr = DMStagPICCreateDMSwarm(dm,&dmswarmA);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(dmswarmA,"eta",1,PETSC_REAL);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(dmswarmA,"rho",1,PETSC_REAL);CHKERRQ(ierr);
  ierr = DMStagPICFinalize(dmswarmA);CHKERRQ(ierr);
  
  {
    PetscInt ppcell[] = {1,1};
    ierr = MPointCoordLayout_DomainVolumeWithCellList(dmswarmA,0,NULL,0.3,ppcell,COOR_INITIALIZE);CHKERRQ(ierr);
  }
  
  {
    PetscReal *pcoor,*pfield;
    PetscInt npoints,p;
    
    ierr = DMSwarmGetLocalSize(dmswarmA,&npoints);CHKERRQ(ierr);
    ierr = DMSwarmGetField(dmswarmA,DMSwarmPICField_coor,NULL,NULL,(void**)&pcoor);CHKERRQ(ierr);
    
    ierr = DMSwarmGetField(dmswarmA,"eta",NULL,NULL,(void**)&pfield);CHKERRQ(ierr);
    for (p=0; p<npoints; p++) {
      pfield[p] = 1.0;
      if (pcoor[2*p+1] < 0.3) {
        pfield[p] = 3.0;
      }
    }
    ierr = DMSwarmRestoreField(dmswarmA,"eta",NULL,NULL,(void**)&pfield);CHKERRQ(ierr);
    
    
    ierr = DMSwarmGetField(dmswarmA,"rho",NULL,NULL,(void**)&pfield);CHKERRQ(ierr);
    for (p=0; p<npoints; p++) {
      pfield[p] = 120.0;
      if (pcoor[2*p+0] < 0.5) {
        pfield[p] = 33.0;
      }
    }
    ierr = DMSwarmRestoreField(dmswarmA,"rho",NULL,NULL,(void**)&pfield);CHKERRQ(ierr);
    ierr = DMSwarmRestoreField(dmswarmA,DMSwarmPICField_coor,NULL,NULL,(void**)&pcoor);CHKERRQ(ierr);
  }
  
  
  ierr = MPoint_ProjectP0_arith(dmswarmA,"eta",dm,dmcell,0,cellcoeff);CHKERRQ(ierr);
  VecView(cellcoeff,PETSC_VIEWER_STDOUT_WORLD);
  ierr = MPoint_ProjectP0_arith(dmswarmA,"rho",dm,dmcell,1,cellcoeff);CHKERRQ(ierr);
  VecView(cellcoeff,PETSC_VIEWER_STDOUT_WORLD);
  
  ierr = VecDestroy(&cellcoeff);CHKERRQ(ierr);
  ierr = DMDestroy(&dmswarmA);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = DMDestroy(&dmcell);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode test_custom_tools_project_2(PetscInt nx,PetscInt ny)
{
  DM              dmcell,dm,dmswarmA;
  PetscInt        dof0,dof1,dof2,stencilWidth;
  Vec             cellcoeff;
  PetscErrorCode  ierr;
  
  dof0 = 1; dof1 = 1; dof2 = 1; /* (vertex) (face) (element) */
  stencilWidth = 1;
  ierr = DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, nx, ny,
                        PETSC_DECIDE, PETSC_DECIDE, dof0, dof1, dof2,
                        DMSTAG_STENCIL_BOX, stencilWidth, NULL,NULL, &dmcell);CHKERRQ(ierr);
  ierr = DMStagSetCoordinateDMType(dmcell,DMPRODUCT);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dmcell);CHKERRQ(ierr);
  ierr = DMSetUp(dmcell);CHKERRQ(ierr);
  
  ierr = DMCreateGlobalVector(dmcell,&cellcoeff);CHKERRQ(ierr);
  
  dof0 = 0; dof1 = 1; dof2 = 1; /* (vertex) (face) (element) */
  stencilWidth = 1;
  ierr = DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, nx, ny,
                        PETSC_DECIDE, PETSC_DECIDE, dof0, dof1, dof2,
                        DMSTAG_STENCIL_BOX, stencilWidth, NULL,NULL, &dm);CHKERRQ(ierr);
  ierr = DMStagSetCoordinateDMType(dm,DMPRODUCT);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMSetUp(dm);CHKERRQ(ierr);
  
  ierr = DMStagSetUniformCoordinatesProduct(dm,0.0,1.0,0.0,1.0,0.0,0.0);CHKERRQ(ierr);
  
  ierr = DMStagPICCreateDMSwarm(dm,&dmswarmA);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(dmswarmA,"eta",1,PETSC_REAL);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(dmswarmA,"rho",1,PETSC_REAL);CHKERRQ(ierr);
  ierr = DMStagPICFinalize(dmswarmA);CHKERRQ(ierr);
  
  {
    PetscInt ppcell[] = {1,1};
    ierr = MPointCoordLayout_DomainVolumeWithCellList(dmswarmA,0,NULL,0.3,ppcell,COOR_INITIALIZE);CHKERRQ(ierr);
  }
  
  {
    PetscReal *pcoor,*pfield;
    PetscInt npoints,p;
    
    ierr = DMSwarmGetLocalSize(dmswarmA,&npoints);CHKERRQ(ierr);
    ierr = DMSwarmGetField(dmswarmA,DMSwarmPICField_coor,NULL,NULL,(void**)&pcoor);CHKERRQ(ierr);
    
    ierr = DMSwarmGetField(dmswarmA,"eta",NULL,NULL,(void**)&pfield);CHKERRQ(ierr);
    for (p=0; p<npoints; p++) {
      pfield[p] = 1.0;
      if (pcoor[2*p+1] < 0.3) {
        pfield[p] = 3.0;
      }
    }
    ierr = DMSwarmRestoreField(dmswarmA,"eta",NULL,NULL,(void**)&pfield);CHKERRQ(ierr);
    
    
    ierr = DMSwarmGetField(dmswarmA,"rho",NULL,NULL,(void**)&pfield);CHKERRQ(ierr);
    for (p=0; p<npoints; p++) {
      pfield[p] = 120.0;
      if (pcoor[2*p+0] < 0.5) {
        pfield[p] = 33.0;
      }
    }
    ierr = DMSwarmRestoreField(dmswarmA,"rho",NULL,NULL,(void**)&pfield);CHKERRQ(ierr);
    ierr = DMSwarmRestoreField(dmswarmA,DMSwarmPICField_coor,NULL,NULL,(void**)&pcoor);CHKERRQ(ierr);
  }
  
  
  ierr = MPoint_ProjectQ1_arith_general(dmswarmA,"eta",dm,dmcell,0,0,cellcoeff);CHKERRQ(ierr);
  VecView(cellcoeff,PETSC_VIEWER_STDOUT_WORLD);
  
  ierr = MPoint_ProjectQ1_arith_general(dmswarmA,"eta",dm,dmcell,1,0,cellcoeff);CHKERRQ(ierr);
  VecView(cellcoeff,PETSC_VIEWER_STDOUT_WORLD);
  
  ierr = MPoint_ProjectQ1_arith_general(dmswarmA,"eta",dm,dmcell,2,0,cellcoeff);CHKERRQ(ierr);
  VecView(cellcoeff,PETSC_VIEWER_STDOUT_WORLD);
  
  ierr = VecDestroy(&cellcoeff);CHKERRQ(ierr);
  ierr = DMDestroy(&dmswarmA);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = DMDestroy(&dmcell);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main (int argc,char **argv)
{
  PetscErrorCode  ierr;
    
  ierr = PetscInitialize(&argc,&argv,(char*)0,help); if (ierr) return(ierr);

  //ierr = test_layout(6,4);CHKERRQ(ierr);
  //ierr = test_advection_rk1(10,10);CHKERRQ(ierr);
  //ierr = test_custom_tools_set(4,4);CHKERRQ(ierr);
  //ierr = test_custom_tools_dup_copy(4,4);CHKERRQ(ierr);
  //ierr = test_custom_tools_project(4,4);CHKERRQ(ierr);
  ierr = test_custom_tools_project_2(4,4);CHKERRQ(ierr);
  
  ierr = PetscFinalize();
  return(ierr);
}
