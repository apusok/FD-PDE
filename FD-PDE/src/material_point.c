
#include <petsc.h>
#include <petscvec.h>
#include <petscdm.h>
#include <petscdmswarm.h>
#include <petscdmstag.h>
#include "dmstag_utils.h"
#include "material_point.h"


PetscErrorCode DMStagPICCreateDMSwarm(DM dmstag,DM *s)
{
  DM             dmswarm;
  PetscErrorCode ierr;
  
  PetscFunctionBeginUser;
  ierr = DMSetPointLocation(dmstag,DMLocatePoints_Stag);CHKERRQ(ierr);
  
  ierr = DMCreate(PETSC_COMM_WORLD,&dmswarm);CHKERRQ(ierr);
  ierr = DMSetType(dmswarm,DMSWARM);CHKERRQ(ierr);
  ierr = DMSetDimension(dmswarm,2);CHKERRQ(ierr);
  ierr = DMSwarmSetType(dmswarm,DMSWARM_PIC);CHKERRQ(ierr);
  ierr = DMSwarmSetCellDM(dmswarm,dmstag);CHKERRQ(ierr);
  *s = dmswarm;
  PetscFunctionReturn(0);
}

PetscErrorCode DMStagPICFinalize(DM dmswarm)
{
  DM             dmstag;
  PetscInt       n[]={0,0},nel;
  PetscErrorCode ierr;
  
  PetscFunctionBeginUser;
  ierr = DMSwarmFinalizeFieldRegister(dmswarm);CHKERRQ(ierr);
  ierr = DMSwarmGetCellDM(dmswarm,&dmstag);CHKERRQ(ierr);
  ierr = DMStagGetLocalSizes(dmstag,&n[0],&n[1],NULL);CHKERRQ(ierr);
  nel = n[0]*n[1];
  ierr = DMSwarmSetLocalSizes(dmswarm,nel*16,100);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
 Performs piece wise constant interpolation
 
 Input:
 + dmswarm - The material point object
 + propname - Textual name of material point property to project
 + dmstag - FD grid object
 + dmcell - FDstag object representing the coefficients
 + element_dof - Ondex where projected element values should be stored

 Output:
 + cellcoeff - Vector where projected properties are stored

 Notes:
 * Only properties declared as PetscReal can be projected
 * Only properties with block size = 1 can be projected
 
 Collective
*/
PetscErrorCode MPoint_ProjectP0_arith(DM dmswarm,const char propname[],
                                              DM dmstag,DM dmcell,PetscInt element_dof,Vec cellcoeff)
{
  PetscInt p,npoints,bs,c,ncell,bs_cell;
  PetscDataType type;
  PetscReal *pfield;
  PetscInt *pcellid,slot;
  Vec cellcoeff_local;
  PetscReal *coeff,*cnt;
  PetscErrorCode ierr;
  
  PetscFunctionBeginUser;
  {
    PetscInt dof0,dof1,dof2;
    ierr = DMStagGetDOF(dmcell,&dof0,&dof1,&dof2,NULL);CHKERRQ(ierr); /* (vertex) (face) (element) */
    if (dof2 == 0) SETERRQ(PetscObjectComm((PetscObject)dmcell),PETSC_ERR_SUP,"Require dmcell has element wise defined fields");
    if (dof0 != 0) SETERRQ(PetscObjectComm((PetscObject)dmcell),PETSC_ERR_SUP,"Only valid for dmcell with element wise defined fields. Detected vertex fields");
    if (dof1 != 0) SETERRQ(PetscObjectComm((PetscObject)dmcell),PETSC_ERR_SUP,"Only valid for dmcell with element wise defined fields. Detected face fields");
  }
  
  ierr = DMSwarmGetLocalSize(dmswarm,&npoints);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dmswarm,propname,&bs,&type,(void**)&pfield);CHKERRQ(ierr);
  if (type != PETSC_REAL) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Type must be PETSC_REAL for field %s",propname);
  if (bs != 1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"Block size must be 1. Found %d for field %s",bs,propname);
  ierr = DMSwarmGetField(dmswarm,DMSwarmPICField_cellid,NULL,NULL,(void**)&pcellid);CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(dmcell,DMSTAG_ELEMENT,element_dof,&slot);CHKERRQ(ierr);

  ierr = DMCreateLocalVector(dmcell,&cellcoeff_local);CHKERRQ(ierr);
  
  /* we scatter here to avoid zeroing out all fields with dof != element_dof */
  ierr = DMGlobalToLocalBegin(dmcell,cellcoeff,INSERT_VALUES,cellcoeff_local);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dmcell,cellcoeff,INSERT_VALUES,cellcoeff_local);CHKERRQ(ierr);
  
  ierr = VecGetSize(cellcoeff_local,&ncell);CHKERRQ(ierr);
  ierr = VecGetBlockSize(cellcoeff_local,&bs_cell);CHKERRQ(ierr);
  ncell = ncell / bs_cell;
  ierr = PetscCalloc1(ncell,&cnt);CHKERRQ(ierr);

  ierr = VecGetArray(cellcoeff_local,&coeff);CHKERRQ(ierr);
  for (p=0; p<npoints; p++) {
    PetscInt cellid = -1;
    
    cellid = pcellid[p];
    
    coeff[ cellid * bs_cell + slot ] += pfield[p];
    cnt[ cellid ] += 1.0;
  }
  for (c=0; c<ncell; c++) {
    PetscBool in_global_space;
    ierr = DMStagLocalElementIndexInGlobalSpace_2d(dmcell,c,&in_global_space);CHKERRQ(ierr);
    if (in_global_space) printf("cell %d : cnt %+1.4e\n",c,cnt[c]);
  }
  
  for (c=0; c<ncell; c++) {
    PetscBool in_global_space;
    ierr = DMStagLocalElementIndexInGlobalSpace_2d(dmcell,c,&in_global_space);CHKERRQ(ierr);
    if (in_global_space) {
      if (cnt[c] < 1.0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cell %D is empty. Cannot perform P0 projection",c);
      coeff[ c * bs_cell + slot ] /= cnt[c];
    }
  }
  ierr = VecRestoreArray(cellcoeff_local,&coeff);CHKERRQ(ierr);
  
  ierr = DMSwarmRestoreField(dmswarm,propname,NULL,NULL,(void**)&pfield);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(dmswarm,DMSwarmPICField_cellid,NULL,NULL,(void**)&pcellid);CHKERRQ(ierr);
  
  ierr = DMLocalToGlobalBegin(dmcell,cellcoeff_local,INSERT_VALUES,cellcoeff);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dmcell,cellcoeff_local,INSERT_VALUES,cellcoeff);CHKERRQ(ierr);

  ierr = VecDestroy(&cellcoeff_local);CHKERRQ(ierr);
  ierr = PetscFree(cnt);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

/*
 Performs piece wise constant interpolation
 
 Input:
 + dmswarm - The material point object
 + propname - Textual name of material point property to project
 + dmstag - FD grid object
 + dmcell - FDstag object representing the coefficients
 + stratrum_index - Index where projected element values should be stored
 
 Output:
 + cellcoeff - Vector where projected properties are stored
 
 Notes:
 * Only properties declared as PetscReal can be projected
 * Only properties with block size = 1 can be projected
 
 Collective
*/
PetscErrorCode v0_MPoint_ProjectQ1_arith_general(DM dmswarm,const char propname[],
                                      DM dmstag,
                                      DM dmcell,
                                      PetscInt stratrum_index, /* 0:(vertex) 1:(face) 2:(element) */
                                      PetscInt dof,Vec cellcoeff)
{
  PetscInt p,npoints,bs,c,ncell,bs_cell;
  PetscDataType type;
  PetscReal *pfield;
  PetscInt *pcellid,nslot,ns,slot[4];
  Vec cellcoeff_local,cnt_global,cnt_local;
  PetscReal ***coeff,***cnt;
  PetscErrorCode ierr;
  
  PetscFunctionBeginUser;
  ierr = DMCreateLocalVector(dmcell,&cellcoeff_local);CHKERRQ(ierr);
  ierr = VecGetSize(cellcoeff_local,&ncell);CHKERRQ(ierr);
  
  /* we scatter here to avoid zeroing out all fields with dof != element_dof */
  ierr = DMGlobalToLocalBegin(dmcell,cellcoeff,INSERT_VALUES,cellcoeff_local);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dmcell,cellcoeff,INSERT_VALUES,cellcoeff_local);CHKERRQ(ierr);

  slot[0] = slot[1] = slot[2] = slot[3] = -1;
  switch (stratrum_index) {
    case 0: // vertex
      nslot = 4;
      ierr = DMStagGetLocationSlot(dmcell,DMSTAG_UP_LEFT   ,dof,&slot[0]);CHKERRQ(ierr);
      ierr = DMStagGetLocationSlot(dmcell,DMSTAG_UP_RIGHT  ,dof,&slot[1]);CHKERRQ(ierr);
      ierr = DMStagGetLocationSlot(dmcell,DMSTAG_DOWN_LEFT ,dof,&slot[2]);CHKERRQ(ierr);
      ierr = DMStagGetLocationSlot(dmcell,DMSTAG_DOWN_RIGHT,dof,&slot[3]);CHKERRQ(ierr);
      break;
    case 1: // face
      nslot = 4;
      ierr = DMStagGetLocationSlot(dmcell,DMSTAG_UP   ,dof,&slot[0]);CHKERRQ(ierr);
      ierr = DMStagGetLocationSlot(dmcell,DMSTAG_DOWN ,dof,&slot[1]);CHKERRQ(ierr);
      ierr = DMStagGetLocationSlot(dmcell,DMSTAG_LEFT ,dof,&slot[2]);CHKERRQ(ierr);
      ierr = DMStagGetLocationSlot(dmcell,DMSTAG_RIGHT,dof,&slot[3]);CHKERRQ(ierr);
      break;
    case 2: // cell
      nslot = 1;
      ierr = DMStagGetLocationSlot(dmcell,DMSTAG_ELEMENT,dof,&slot[0]);CHKERRQ(ierr);
      break;
      
    default:
      break;
  }
  
  ierr = DMSwarmGetLocalSize(dmswarm,&npoints);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dmswarm,propname,&bs,&type,(void**)&pfield);CHKERRQ(ierr);
  if (type != PETSC_REAL) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Type must be PETSC_REAL for field %s",propname);
  if (bs != 1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"Block size must be 1. Found %d for field %s",bs,propname);
  ierr = DMSwarmGetField(dmswarm,DMSwarmPICField_cellid,NULL,NULL,(void**)&pcellid);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(dmcell,&cnt_global);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dmcell,&cnt_local);CHKERRQ(ierr);
  
  ierr = DMStagVecGetArrayDOF(dmcell,cellcoeff_local,&coeff);CHKERRQ(ierr);
  ierr = DMStagVecGetArrayDOF(dmcell,cnt_local,&cnt);CHKERRQ(ierr);

  for (p=0; p<npoints; p++) {
    PetscInt cellid = -1;
    PetscInt geid[]={0,0,0};
    
    cellid = pcellid[p];

    ierr = DMStagGetLocalElementGlobalIndices(dmcell,cellid,geid);CHKERRQ(ierr);
    for (ns=0; ns<nslot; ns++) {
      coeff[ geid[1] ][ geid[0] ][ slot[ns] ] += pfield[p];
      cnt  [ geid[1] ][ geid[0] ][ slot[ns] ] += 1.0;
    }
  }
  
  ierr = DMStagVecRestoreArrayDOF(dmcell,cnt_local,&cnt);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArrayDOF(dmcell,cellcoeff_local,&coeff);CHKERRQ(ierr);

  ierr = DMSwarmRestoreField(dmswarm,propname,NULL,NULL,(void**)&pfield);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(dmswarm,DMSwarmPICField_cellid,NULL,NULL,(void**)&pcellid);CHKERRQ(ierr);
  
  ierr = DMLocalToGlobalBegin(dmcell,cellcoeff_local,ADD_VALUES,cellcoeff);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dmcell,cellcoeff_local,ADD_VALUES,cellcoeff);CHKERRQ(ierr);

  ierr = DMLocalToGlobalBegin(dmcell,cnt_local,ADD_VALUES,cnt_global);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dmcell,cnt_local,ADD_VALUES,cnt_global);CHKERRQ(ierr);

  {
    PetscInt i,len;
    PetscReal *LA_coeff,*LA_cnt;
    
    ierr = VecGetLocalSize(cellcoeff,&len);CHKERRQ(ierr);
    ierr = VecGetArray(cellcoeff,&LA_coeff);CHKERRQ(ierr);
    ierr = VecGetArray(cnt_global,&LA_cnt);CHKERRQ(ierr);
    for (i=0; i<len; i++) {
      if (LA_cnt[i] > 0.0) {
        LA_coeff[i] = LA_coeff[i] / LA_cnt[i];
      }
    }
    ierr = VecRestoreArray(cnt_global,&LA_cnt);CHKERRQ(ierr);
    ierr = VecRestoreArray(cellcoeff,&LA_coeff);CHKERRQ(ierr);
  }
  
  ierr = VecDestroy(&cnt_global);CHKERRQ(ierr);
  ierr = VecDestroy(&cnt_local);CHKERRQ(ierr);
  ierr = VecDestroy(&cellcoeff_local);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*
 Performs piece wise constant interpolation
 
 Input:
 + dmswarm - The material point object
 + propname - Textual name of material point property to project
 + dmstag - FD grid object
 + dmcell - FDstag object representing the coefficients
 + stratrum_index - Index where projected element values should be stored
 
 Output:
 + cellcoeff - Vector where projected properties are stored
 
 Notes:
 * Only properties declared as PetscReal can be projected
 * Only properties with block size = 1 can be projected

 Collective
*/
PetscErrorCode MPoint_ProjectQ1_arith_general(DM dmswarm,const char propname[],
                                              DM dmstag,
                                              DM dmcell,
                                              PetscInt stratrum_index, /* 0:(vertex) 1:(face) 2:(element) */
                                              PetscInt stagdof,Vec cellcoeff)
{
  DM compat;
  PetscInt p,npoints,bs,c,dof[4];
  PetscDataType type;
  PetscReal *pfield;
  PetscInt *pcellid,nslot,ns,slot[4];
  Vec sum_global,sum_local,cnt_global,cnt_local;
  PetscReal ***coeff_s,***coeff,***cnt;
  PetscErrorCode ierr;
  
  PetscFunctionBeginUser;
  slot[0] = slot[1] = slot[2] = slot[3] = -1;
  switch (stratrum_index) {
    case 0: // vertex
      dof[0] = 1; dof[1] = 0; dof[2] = 0; dof[3] = 0; /* (vertex) (face) (element) */
      ierr = DMStagCreateCompatibleDMStag(dmcell,dof[0],dof[1],dof[2],dof[3],&compat);CHKERRQ(ierr);
      nslot = 4;
      ierr = DMStagGetLocationSlot(compat,DMSTAG_UP_LEFT   ,0,&slot[0]);CHKERRQ(ierr);
      ierr = DMStagGetLocationSlot(compat,DMSTAG_UP_RIGHT  ,0,&slot[1]);CHKERRQ(ierr);
      ierr = DMStagGetLocationSlot(compat,DMSTAG_DOWN_LEFT ,0,&slot[2]);CHKERRQ(ierr);
      ierr = DMStagGetLocationSlot(compat,DMSTAG_DOWN_RIGHT,0,&slot[3]);CHKERRQ(ierr);
      break;
    case 1: // face
      dof[0] = 0; dof[1] = 1; dof[2] = 0; dof[3] = 0; /* (vertex) (face) (element) */
      ierr = DMStagCreateCompatibleDMStag(dmcell,dof[0],dof[1],dof[2],dof[3],&compat);CHKERRQ(ierr);
      nslot = 4;
      ierr = DMStagGetLocationSlot(compat,DMSTAG_UP   ,0,&slot[0]);CHKERRQ(ierr);
      ierr = DMStagGetLocationSlot(compat,DMSTAG_DOWN ,0,&slot[1]);CHKERRQ(ierr);
      ierr = DMStagGetLocationSlot(compat,DMSTAG_LEFT ,0,&slot[2]);CHKERRQ(ierr);
      ierr = DMStagGetLocationSlot(compat,DMSTAG_RIGHT,0,&slot[3]);CHKERRQ(ierr);
      break;
    case 2: // cell
      dof[0] = 0; dof[1] = 0; dof[2] = 1; dof[3] = 0; /* (vertex) (face) (element) */
      ierr = DMStagCreateCompatibleDMStag(dmcell,dof[0],dof[1],dof[2],dof[3],&compat);CHKERRQ(ierr);
      nslot = 1;
      ierr = DMStagGetLocationSlot(compat,DMSTAG_ELEMENT,0,&slot[0]);CHKERRQ(ierr);
      break;
      
    default:
      break;
  }
  
  ierr = DMCreateGlobalVector(compat,&sum_global);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(compat,&sum_local);CHKERRQ(ierr);
  
  ierr = DMSwarmGetLocalSize(dmswarm,&npoints);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dmswarm,propname,&bs,&type,(void**)&pfield);CHKERRQ(ierr);
  if (type != PETSC_REAL) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Type must be PETSC_REAL for field %s",propname);
  if (bs != 1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"Block size must be 1. Found %d for field %s",bs,propname);
  ierr = DMSwarmGetField(dmswarm,DMSwarmPICField_cellid,NULL,NULL,(void**)&pcellid);CHKERRQ(ierr);
  
  ierr = DMCreateGlobalVector(compat,&cnt_global);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(compat,&cnt_local);CHKERRQ(ierr);
  
  ierr = DMStagVecGetArrayDOF(compat,sum_local,&coeff);CHKERRQ(ierr);
  ierr = DMStagVecGetArrayDOF(compat,cnt_local,&cnt);CHKERRQ(ierr);
  
  for (p=0; p<npoints; p++) {
    PetscInt cellid = -1;
    PetscInt geid[]={0,0,0};
    
    cellid = pcellid[p];
    
    ierr = DMStagGetLocalElementGlobalIndices(compat,cellid,geid);CHKERRQ(ierr);
    for (ns=0; ns<nslot; ns++) {
      coeff[ geid[1] ][ geid[0] ][ slot[ns] ] += pfield[p];
      cnt  [ geid[1] ][ geid[0] ][ slot[ns] ] += 1.0;
    }
  }
  
  ierr = DMStagVecRestoreArrayDOF(compat,cnt_local,&cnt);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArrayDOF(compat,sum_local,&coeff);CHKERRQ(ierr);
  
  ierr = DMSwarmRestoreField(dmswarm,propname,NULL,NULL,(void**)&pfield);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(dmswarm,DMSwarmPICField_cellid,NULL,NULL,(void**)&pcellid);CHKERRQ(ierr);
  
  ierr = DMLocalToGlobalBegin(compat,sum_local,ADD_VALUES,sum_global);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(compat,sum_local,ADD_VALUES,sum_global);CHKERRQ(ierr);
  
  ierr = DMLocalToGlobalBegin(compat,cnt_local,ADD_VALUES,cnt_global);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(compat,cnt_local,ADD_VALUES,cnt_global);CHKERRQ(ierr);
  
  {
    PetscInt i,len;
    PetscReal *LA_coeff,*LA_cnt;
    
    ierr = VecGetLocalSize(cnt_global,&len);CHKERRQ(ierr);
    ierr = VecGetArray(sum_global,&LA_coeff);CHKERRQ(ierr);
    ierr = VecGetArray(cnt_global,&LA_cnt);CHKERRQ(ierr);
    for (i=0; i<len; i++) {
      if (LA_cnt[i] > 0.0) {
        LA_coeff[i] = LA_coeff[i] / LA_cnt[i];
      }
    }
    ierr = VecRestoreArray(cnt_global,&LA_cnt);CHKERRQ(ierr);
    ierr = VecRestoreArray(sum_global,&LA_coeff);CHKERRQ(ierr);
  }

  {
    PetscInt n0A=0,n1A=0,n2A=0;
    PetscInt dof_list=-1;
    PetscInt *dof0A=NULL,*dof1A=NULL,*dof2A=NULL;
    PetscInt *dof0B=NULL,*dof1B=NULL,*dof2B=NULL;
    IS is_compat,is_cell;
    const PetscInt *_compat;
    const PetscInt *_cell;
    PetscInt i,len;
    PetscReal *LA_cellcoeff,*LA_coeff;
    
    switch (stratrum_index) {
      case 0: // vertex
        n0A = 1;
        dof_list = 0;
        dof0A = &dof_list;
        dof0B = &stagdof;
        break;
      case 1: // face
        n1A = 1;
        dof_list = 0;
        dof1A = &dof_list;
        dof1B = &stagdof;
        break;
      case 2: // cell
        n2A = 1;
        dof_list = 0;
        dof2A = &dof_list;
        dof2B = &stagdof;
        break;
      default:
        break;
    }
    
    ierr = DMStagISCreateL2L_2d(compat,n0A,dof0A,n1A,dof1A,n2A,dof2A,&is_compat,
                                dmcell,dof0B,dof1B,dof2B,&is_cell);CHKERRQ(ierr);
    
    ierr = ISGetLocalSize(is_compat,&len);CHKERRQ(ierr);
    ierr = ISGetIndices(is_compat,&_compat);CHKERRQ(ierr);
    ierr = ISGetIndices(is_cell,&_cell);CHKERRQ(ierr);

    ierr = VecGetArray(sum_global,&LA_coeff);CHKERRQ(ierr);
    ierr = VecGetArray(cellcoeff,&LA_cellcoeff);CHKERRQ(ierr);
    for (i=0; i<len; i++) {
      LA_cellcoeff[ _cell[i] ] = LA_coeff[ _compat[i] ];
    }
    ierr = VecRestoreArray(cellcoeff,&LA_cellcoeff);CHKERRQ(ierr);
    ierr = VecRestoreArray(sum_global,&LA_coeff);CHKERRQ(ierr);

    ierr = ISRestoreIndices(is_cell,&_cell);CHKERRQ(ierr);
    ierr = ISRestoreIndices(is_compat,&_compat);CHKERRQ(ierr);

    ierr = ISDestroy(&is_compat);CHKERRQ(ierr);
    ierr = ISDestroy(&is_cell);CHKERRQ(ierr);
  }
  
  ierr = VecDestroy(&cnt_global);CHKERRQ(ierr);
  ierr = VecDestroy(&cnt_local);CHKERRQ(ierr);
  ierr = VecDestroy(&sum_global);CHKERRQ(ierr);
  ierr = VecDestroy(&sum_local);CHKERRQ(ierr);
  ierr = DMDestroy(&compat);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*
 Assign particles coordinates within all cells in the domain. Particle coordinates are assigned cell-wise
 
 Input:
 + dmswarm - The particle object
 + factor - Controls the magnitude of the random perturbation applied to each particle position. Must be in range [0,1]
 + points_per_dim - Number of points in x/y to define within each cell
 + mode - Indicates whether new coordinates should be appending to the existing points in dmswarm, or whether
          the existing points (and size) of dmswarm should be set to zero prior to defining new coordinates
 
 Output:
 + dmswarm - On exit dmswarm will contain newly defined coordinates
 
 Notes:
 * Assumes constant cell spacing

 Collective
*/
PetscErrorCode MPointCoordLayout_DomainVolume(DM dmswarm,PetscReal factor,PetscInt points_per_dim,MPointCoordinateInsertMode mode)
{
  DM             dmstag;
  PetscInt       p,npoints,Ng[]={0,0};
  PetscReal      *LA_coor;
  PetscReal      dx[2],dxp[2],gmin[]={0,0,0},gmax[]={0,0,0};
  PetscRandom    r=NULL;
  PetscMPIInt    rank;
  PetscErrorCode ierr;
  
  PetscFunctionBeginUser;
  if (mode == COOR_APPEND) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Only mode COOR_INITIALIZE supported");
  
  ierr = DMSwarmInsertPointsUsingCellDM(dmswarm,DMSWARMPIC_LAYOUT_REGULAR,points_per_dim);CHKERRQ(ierr);
  
  ierr = DMSwarmGetCellDM(dmswarm,&dmstag);CHKERRQ(ierr);
  ierr = DMStagGetGlobalSizes(dmstag,&Ng[0],&Ng[1],NULL);CHKERRQ(ierr);
  ierr = DMStagGetBoundingBox(dmstag,gmin,gmax);CHKERRQ(ierr);
  dx[0] = (gmax[0]-gmin[0])/Ng[0];
  dx[1] = (gmax[1]-gmin[1])/Ng[1];
  dxp[0] = dx[0] / (PetscReal)points_per_dim;
  dxp[1] = dx[1] / (PetscReal)points_per_dim;
  
  if (factor > 0.0) {
    ierr = PetscRandomCreate(PETSC_COMM_SELF,&r);CHKERRQ(ierr);
    ierr = PetscRandomSetInterval(r,-factor,factor);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dmswarm),&rank);CHKERRQ(ierr);
    ierr = PetscRandomSetSeed(r,(unsigned long)rank);CHKERRQ(ierr);
    ierr = PetscRandomSeed(r);CHKERRQ(ierr);
  }
  
  ierr = DMSwarmGetField(dmswarm,DMSwarmPICField_coor,NULL,NULL,(void**)&LA_coor);CHKERRQ(ierr);
  ierr = DMSwarmGetLocalSize(dmswarm,&npoints);CHKERRQ(ierr);
  if (factor > 0.0) {

    for (p=0; p<npoints; p++) {
      PetscReal rr[2];
      
      ierr = PetscRandomGetValueReal(r,&rr[0]);CHKERRQ(ierr);
      ierr = PetscRandomGetValueReal(r,&rr[1]);CHKERRQ(ierr);
      LA_coor[2*p+0] += rr[0] * 0.5 * dxp[0];
      LA_coor[2*p+1] += rr[1] * 0.5 * dxp[1];
    }
  }
  ierr = DMSwarmRestoreField(dmswarm,DMSwarmPICField_coor,NULL,NULL,(void**)&LA_coor);CHKERRQ(ierr);
  
  /* Migrate - since since perturbing particles have have caused point to be located in a different cell, or located on another sub-domain */
  ierr = DMSwarmMigrate(dmswarm,PETSC_TRUE);CHKERRQ(ierr);

  ierr = PetscRandomDestroy(&r);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode _CellFillCoor(PetscReal gmin[],PetscReal gmax[],PetscInt np[],PetscReal coor[])
{
  PetscReal dx,dy;
  PetscInt i,j;
  PetscReal x0=gmin[0],x1=gmax[0],y0=gmin[1],y1=gmax[1];
  PetscInt npx=np[0],npy=np[1];
  
  dx = (x1 - x0) / (PetscReal)npx;
  dy = (y1 - y0) / (PetscReal)npy;
  for (j=0; j<npy; j++) {
    for (i=0; i<npx; i++) {
      PetscInt idx = i + j * npx;
      coor[2*idx+0] = x0 + 0.5 * dx + i * dx;
      coor[2*idx+1] = y0 + 0.5 * dy + j * dy;
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode _CellFaceFillCoor_n_s(PetscReal x0,PetscReal x1,PetscReal y,PetscInt npx,PetscReal coor[])
{
  PetscReal dx;
  PetscInt i;
  
  dx = (x1 - x0) / (PetscReal)npx;
  for (i=0; i<npx; i++) {
    coor[2*i+0] = x0 + 0.5 * dx + i * dx;
    coor[2*i+1] = y;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode _CellFaceFillCoor_e_w(PetscReal y0,PetscReal y1,PetscReal x,PetscInt npy,PetscReal coor[])
{
  PetscReal dy;
  PetscInt j;
  
  dy = (y1 - y0) / (PetscReal)npy;
  for (j=0; j<npy; j++) {
    coor[2*j+0] = x;
    coor[2*j+1] = y0 + 0.5 * dy + j * dy;
  }
  
  PetscFunctionReturn(0);
}

/*
 Assign particles coordinates within a subset of cells in the domain. Particle coordinates are assigned cell-wise
 
 Input:
 + dmswarm - The particle object
 + _ncells - Number of cells to fill
 + celllist[] - Array of element indices to fill. Indices are defined using the local sub-domin numbering
 + factor - Controls the magnitude of the random perturbation applied to each particle position. Must be in range [0,1]
 + points_per_dim[] - Number of points in x and y to define within each cell
 + mode - Indicates whether new coordinates should be appending to the existing points in dmswarm, or whether
 the existing points (and size) of dmswarm should be set to zero prior to defining new coordinates
 
 Output:
 + dmswarm - On exit dmswarm will contain newly defined coordinates
 
 Notes:
 * Assumes constant cell spacing

 Collective
*/
PetscErrorCode MPointCoordLayout_DomainVolumeWithCellList(DM dmswarm,
                                                 PetscInt _ncells,PetscInt celllist[],
                                                 PetscReal factor,PetscInt points_per_dim[],
                                                 MPointCoordinateInsertMode mode)
{
  DM dmstag;
  PetscInt c,ncells,nppcell,nnew,npoints_init,npoints,pcnt,p,cellid_local;
  PetscInt Ng[]={0,0,0},N[]={0,0,0},es[]={0,0};
  PetscReal dx[]={0,0,0},dxp[]={0,0,0};
  PetscReal gmin[]={0,0,0},gmax[]={0,0,0};
  PetscReal *pcoor,*cellcoor;
  PetscInt *pcellid;
  PetscReal cgmin[]={0,0};
  PetscReal cgmax[]={0,0};
  PetscInt egidx[]={0,0,0};
  PetscRandom    r=NULL;
  PetscMPIInt    rank;
  PetscErrorCode ierr;
  
  PetscFunctionBeginUser;
  if (mode == COOR_INITIALIZE) {
    ierr = DMSwarmSetLocalSizes(dmswarm,0,-1);CHKERRQ(ierr);
  }
  
  ierr = DMSwarmGetCellDM(dmswarm,&dmstag);CHKERRQ(ierr);
  ierr = DMStagGetGlobalSizes(dmstag,&Ng[0],&Ng[1],NULL);CHKERRQ(ierr);
  ierr = DMStagGetLocalSizes(dmstag,&N[0],&N[1],NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dmstag,&es[0],&es[1],NULL,NULL,NULL,NULL,NULL,NULL,NULL);
  ierr = DMStagGetBoundingBox(dmstag,gmin,gmax);CHKERRQ(ierr);
  
  if (!celllist) {
    ncells = N[0] * N[1];
  } else {
    ncells = _ncells;
  }
  if (ncells == 0) PetscFunctionReturn(0);
  
  ierr = DMSwarmGetLocalSize(dmswarm,&npoints_init);CHKERRQ(ierr);
  nppcell = points_per_dim[0] * points_per_dim[1];
  nnew = ncells * nppcell;
  ierr = DMSwarmSetLocalSizes(dmswarm,npoints_init+nnew,-1);CHKERRQ(ierr);
  ierr = DMSwarmGetLocalSize(dmswarm,&npoints);CHKERRQ(ierr);
  
  ierr = PetscMalloc1(2*nppcell,&cellcoor);CHKERRQ(ierr);
  
  dx[0] = (gmax[0]-gmin[0])/Ng[0];
  dx[1] = (gmax[1]-gmin[1])/Ng[1];
  
  if (factor > 0.0) {
    ierr = PetscRandomCreate(PETSC_COMM_SELF,&r);CHKERRQ(ierr);
    ierr = PetscRandomSetInterval(r,-factor,factor);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dmswarm),&rank);CHKERRQ(ierr);
    ierr = PetscRandomSetSeed(r,(unsigned long)rank);CHKERRQ(ierr);
    ierr = PetscRandomSeed(r);CHKERRQ(ierr);
  }

  ierr = DMSwarmGetField(dmswarm,DMSwarmPICField_coor,NULL,NULL,(void**)&pcoor);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dmswarm,DMSwarmPICField_cellid,NULL,NULL,(void**)&pcellid);CHKERRQ(ierr);
  pcnt = 0;
  if (celllist) {
    for (c=0; c<ncells; c++) {
      PetscInt cellid = celllist[c];

      /* get cell ei,ej, index */
      egidx[1] = cellid / N[0];
      egidx[0] = cellid - egidx[1] * N[0];
      egidx[0] += es[0];
      egidx[1] += es[1];
      
      cgmin[0] = gmin[0] + egidx[0]*dx[0];
      cgmax[0] = cgmin[0] + dx[0];
      
      cgmin[1] = gmin[1] + egidx[1]*dx[1];
      cgmax[1] = cgmin[1] + dx[1];
      ierr = _CellFillCoor(cgmin,cgmax,points_per_dim,cellcoor);CHKERRQ(ierr);
      
      if (factor > 0.0) {
        dxp[0] = (cgmax[0] - cgmin[0]) / (PetscReal)points_per_dim[0];
        dxp[1] = (cgmax[1] - cgmin[1]) / (PetscReal)points_per_dim[1];

        for (p=0; p<nppcell; p++) {
          PetscReal rr[2];
          
          ierr = PetscRandomGetValueReal(r,&rr[0]);CHKERRQ(ierr);
          ierr = PetscRandomGetValueReal(r,&rr[1]);CHKERRQ(ierr);
          cellcoor[2*p+0] += rr[0] * 0.5 * dxp[0];
          cellcoor[2*p+1] += rr[1] * 0.5 * dxp[1];
        }
      }
      
      for (p=0; p<nppcell; p++) {
        pcoor[2*(npoints_init + pcnt)+0] = cellcoor[2*p+0];
        pcoor[2*(npoints_init + pcnt)+1] = cellcoor[2*p+1];
        /*
         This assignment is wrong as it doesn't correspond to the stag cell layout whoch includes an extra cell
         in the local space on the left/up faces - does this matter?
         For projecting points->cell yes
        */
        //pcellid[npoints_init + pcnt] = cellid;
        ierr = DMStagGetLocalElementIndex(dmstag,egidx,&cellid_local);CHKERRQ(ierr);
        pcellid[npoints_init + pcnt] = cellid_local;

        pcnt++;
      }
    }
    
  } else {
    for (c=0; c<ncells; c++) {
      PetscInt cellid = c;

      /* get cell i,j, index */
      egidx[1] = cellid / N[0];
      egidx[0] = cellid - egidx[1] * N[0];
      egidx[0] += es[0];
      egidx[1] += es[1];
      
      cgmin[0] = gmin[0] + egidx[0]*dx[0];
      cgmax[0] = cgmin[0] + dx[0];
      
      cgmin[1] = gmin[1] + egidx[1]*dx[1];
      cgmax[1] = cgmin[1] + dx[1];
      ierr = _CellFillCoor(cgmin,cgmax,points_per_dim,cellcoor);CHKERRQ(ierr);
      
      if (factor > 0.0) {
        dxp[0] = (cgmax[0] - cgmin[0]) / (PetscReal)points_per_dim[0];
        dxp[1] = (cgmax[1] - cgmin[1]) / (PetscReal)points_per_dim[1];
        
        for (p=0; p<nppcell; p++) {
          PetscReal rr[2];
          
          ierr = PetscRandomGetValueReal(r,&rr[0]);CHKERRQ(ierr);
          ierr = PetscRandomGetValueReal(r,&rr[1]);CHKERRQ(ierr);
          cellcoor[2*p+0] += rr[0] * 0.5 * dxp[0];
          cellcoor[2*p+1] += rr[1] * 0.5 * dxp[1];
        }
      }

      for (p=0; p<nppcell; p++) {
        pcoor[2*(npoints_init + pcnt)+0] = cellcoor[2*p+0];
        pcoor[2*(npoints_init + pcnt)+1] = cellcoor[2*p+1];
        /*
         This assignment is wrong as it doesn't correspond to the stag cell layout whoch includes an extra cell
         in the local space on the left/up faces - does this matter?
         For projecting points->cell yes
         */
        //pcellid[npoints_init + pcnt] = cellid;
        ierr = DMStagGetLocalElementIndex(dmstag,egidx,&cellid_local);CHKERRQ(ierr);
        pcellid[npoints_init + pcnt] = cellid_local;

        pcnt++;
      }
    }
  }
  ierr = DMSwarmRestoreField(dmswarm,DMSwarmPICField_coor,NULL,NULL,(void**)&pcoor);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(dmswarm,DMSwarmPICField_cellid,NULL,NULL,(void**)&pcellid);CHKERRQ(ierr);

  ierr = DMSwarmMigrate(dmswarm,PETSC_TRUE);CHKERRQ(ierr);

  ierr = PetscRandomDestroy(&r);CHKERRQ(ierr);
  ierr = PetscFree(cellcoor);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode _MPointCoordLayout_FillDomainFace_NS(DM dmswarm,
                                          PetscInt JJ,PetscReal factor,PetscInt points_per_dim,
                                          MPointCoordinateInsertMode mode)
{
  DM dmstag;
  PetscInt c,ncells,nppcell,nnew,npoints_init,npoints,pcnt,p,cellid_local;
  PetscInt Ng[]={0,0,0},N[]={0,0,0};
  PetscReal dx[]={0,0,0},dxp=0;
  PetscReal gmin[]={0,0,0},gmax[]={0,0,0};
  PetscReal *pcoor,*cellcoor;
  PetscInt *pcellid;
  PetscReal cgmin[]={0,0};
  PetscReal cgmax[]={0,0};
  PetscInt elidx[]={0,0,0},egidx[]={0,0,0};
  PetscInt es[]={0,0},nele[]={0,0};
  PetscRandom    r=NULL;
  PetscMPIInt    rank;
  PetscReal      yref;
  PetscErrorCode ierr;
  
  PetscFunctionBeginUser;
  //if (mode == COOR_INITIALIZE) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Only mode COOR_APPEND supported. Initialize does not make sense when filling a sub-set of cells");
  if (mode == COOR_INITIALIZE) {
    ierr = DMSwarmSetLocalSizes(dmswarm,0,-1);CHKERRQ(ierr);
  }
  
  ierr = DMSwarmGetCellDM(dmswarm,&dmstag);CHKERRQ(ierr);
  ierr = DMStagGetGlobalSizes(dmstag,&Ng[0],&Ng[1],NULL);CHKERRQ(ierr);
  ierr = DMStagGetLocalSizes(dmstag,&N[0],&N[1],NULL);CHKERRQ(ierr);
  ierr = DMStagGetBoundingBox(dmstag,gmin,gmax);CHKERRQ(ierr);

  ierr = DMStagGetCorners(dmstag,&es[0],&es[1],NULL,&nele[0],&nele[1],NULL,NULL,NULL,NULL);
  ncells = 0;
  if (es[1] == JJ) {
    ncells = nele[0];
    elidx[1] = 0;
    yref = gmin[1];
  }
  if (es[1]+nele[1] == JJ) {
    ncells = nele[0];
    elidx[1] = N[1]-1;
    yref = gmax[1];
  }
  
  ierr = DMSwarmGetLocalSize(dmswarm,&npoints_init);CHKERRQ(ierr);
  nppcell = points_per_dim;
  nnew = ncells * nppcell;
  ierr = DMSwarmSetLocalSizes(dmswarm,npoints_init+nnew,-1);CHKERRQ(ierr);
  ierr = DMSwarmGetLocalSize(dmswarm,&npoints);CHKERRQ(ierr);
  
  ierr = PetscMalloc1(2*nppcell,&cellcoor);CHKERRQ(ierr);
  
  dx[0] = (gmax[0]-gmin[0])/Ng[0];
  dx[1] = (gmax[1]-gmin[1])/Ng[1];
  
  if (factor > 0.0) {
    ierr = PetscRandomCreate(PETSC_COMM_SELF,&r);CHKERRQ(ierr);
    ierr = PetscRandomSetInterval(r,-factor,factor);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dmswarm),&rank);CHKERRQ(ierr);
    ierr = PetscRandomSetSeed(r,(unsigned long)rank);CHKERRQ(ierr);
    ierr = PetscRandomSeed(r);CHKERRQ(ierr);
  }
  
  ierr = DMSwarmGetField(dmswarm,DMSwarmPICField_coor,NULL,NULL,(void**)&pcoor);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dmswarm,DMSwarmPICField_cellid,NULL,NULL,(void**)&pcellid);CHKERRQ(ierr);
  pcnt = 0;
  for (c=0; c<ncells; c++) {
    PetscInt cellid = c + elidx[1] * nele[0];
    
    /* get cell i,j, index */
    egidx[1] = cellid / N[0];
    egidx[0] = cellid - egidx[1] * N[0];
    egidx[0] += es[0];
    egidx[1] += es[1];
    
    cgmin[0] = gmin[0] + egidx[0]*dx[0];
    cgmax[0] = cgmin[0] + dx[0];
    
    cgmin[1] = gmin[1] + egidx[1]*dx[1];
    cgmax[1] = cgmin[1] + dx[1];
    ierr = _CellFaceFillCoor_n_s(cgmin[0],cgmax[0],yref,points_per_dim,cellcoor);CHKERRQ(ierr);
    
    if (factor > 0.0) {
      dxp = (cgmax[0] - cgmin[0]) / (PetscReal)points_per_dim;
      
      for (p=0; p<nppcell; p++) {
        PetscReal rr;
        
        ierr = PetscRandomGetValueReal(r,&rr);CHKERRQ(ierr);
        cellcoor[2*p+0] += rr * 0.5 * dxp;
      }
    }
    
    for (p=0; p<nppcell; p++) {
      pcoor[2*(npoints_init + pcnt)+0] = cellcoor[2*p+0];
      pcoor[2*(npoints_init + pcnt)+1] = cellcoor[2*p+1];
      /*
       This assignment is wrong as it doesn't correspond to the stag cell layout whoch includes an extra cell
       in the local space on the left/up faces - does this matter?
       For projecting points->cell yes
       */
      //pcellid[npoints_init + pcnt] = cellid;
      ierr = DMStagGetLocalElementIndex(dmstag,egidx,&cellid_local);CHKERRQ(ierr);
      pcellid[npoints_init + pcnt] = cellid_local;

      pcnt++;
    }
  }
  ierr = DMSwarmRestoreField(dmswarm,DMSwarmPICField_coor,NULL,NULL,(void**)&pcoor);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(dmswarm,DMSwarmPICField_cellid,NULL,NULL,(void**)&pcellid);CHKERRQ(ierr);
  
  ierr = PetscRandomDestroy(&r);CHKERRQ(ierr);
  ierr = PetscFree(cellcoor);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode _MPointCoordLayout_FillDomainFace_EW(DM dmswarm,
                                                    PetscInt II,PetscReal factor,PetscInt points_per_dim,
                                                    MPointCoordinateInsertMode mode)
{
  DM dmstag;
  PetscInt c,ncells,nppcell,nnew,npoints_init,npoints,pcnt,p,cellid_local;
  PetscInt Ng[]={0,0,0},N[]={0,0,0};
  PetscReal dx[]={0,0,0},dyp=0;
  PetscReal gmin[]={0,0,0},gmax[]={0,0,0};
  PetscReal *pcoor,*cellcoor;
  PetscInt *pcellid;
  PetscReal cgmin[]={0,0};
  PetscReal cgmax[]={0,0};
  PetscInt elidx[]={0,0,0},egidx[]={0,0,0};
  PetscInt es[]={0,0},nele[]={0,0};
  PetscRandom    r=NULL;
  PetscMPIInt    rank;
  PetscReal      xref;
  PetscErrorCode ierr;
  
  PetscFunctionBeginUser;
  if (mode == COOR_INITIALIZE) {
    ierr = DMSwarmSetLocalSizes(dmswarm,0,-1);CHKERRQ(ierr);
  }
  
  ierr = DMSwarmGetCellDM(dmswarm,&dmstag);CHKERRQ(ierr);
  ierr = DMStagGetGlobalSizes(dmstag,&Ng[0],&Ng[1],NULL);CHKERRQ(ierr);
  ierr = DMStagGetLocalSizes(dmstag,&N[0],&N[1],NULL);CHKERRQ(ierr);
  ierr = DMStagGetBoundingBox(dmstag,gmin,gmax);CHKERRQ(ierr);
  
  ierr = DMStagGetCorners(dmstag,&es[0],&es[1],NULL,&nele[0],&nele[1],NULL,NULL,NULL,NULL);
  ncells = 0;
  if (es[0] == II) {
    ncells = nele[1];
    elidx[0] = 0;
    xref = gmin[0];
  }
  if (es[0]+nele[0] == II) {
    ncells = nele[1];
    elidx[0] = N[0]-1;
    xref = gmax[0];
  }
  
  ierr = DMSwarmGetLocalSize(dmswarm,&npoints_init);CHKERRQ(ierr);
  nppcell = points_per_dim;
  nnew = ncells * nppcell;
  ierr = DMSwarmSetLocalSizes(dmswarm,npoints_init+nnew,-1);CHKERRQ(ierr);
  ierr = DMSwarmGetLocalSize(dmswarm,&npoints);CHKERRQ(ierr);
  
  ierr = PetscMalloc1(2*nppcell,&cellcoor);CHKERRQ(ierr);
  
  dx[0] = (gmax[0]-gmin[0])/Ng[0];
  dx[1] = (gmax[1]-gmin[1])/Ng[1];
  
  if (factor > 0.0) {
    ierr = PetscRandomCreate(PETSC_COMM_SELF,&r);CHKERRQ(ierr);
    ierr = PetscRandomSetInterval(r,-factor,factor);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dmswarm),&rank);CHKERRQ(ierr);
    ierr = PetscRandomSetSeed(r,(unsigned long)rank);CHKERRQ(ierr);
    ierr = PetscRandomSeed(r);CHKERRQ(ierr);
  }
  
  ierr = DMSwarmGetField(dmswarm,DMSwarmPICField_coor,NULL,NULL,(void**)&pcoor);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dmswarm,DMSwarmPICField_cellid,NULL,NULL,(void**)&pcellid);CHKERRQ(ierr);
  pcnt = 0;
  for (c=0; c<ncells; c++) {
    PetscInt cellid = elidx[0] + c * nele[0];
    
    /* get cell i,j, index */
    egidx[1] = cellid / N[0];
    egidx[0] = cellid - egidx[1] * N[0];
    egidx[0] += es[0];
    egidx[1] += es[1];
    
    cgmin[0] = gmin[0] + egidx[0]*dx[0];
    cgmax[0] = cgmin[0] + dx[0];
    
    cgmin[1] = gmin[1] + egidx[1]*dx[1];
    cgmax[1] = cgmin[1] + dx[1];
    ierr = _CellFaceFillCoor_e_w(cgmin[1],cgmax[1],xref,points_per_dim,cellcoor);CHKERRQ(ierr);
    
    if (factor > 0.0) {
      dyp = (cgmax[1] - cgmin[1]) / (PetscReal)points_per_dim;
      
      for (p=0; p<nppcell; p++) {
        PetscReal rr;
        
        ierr = PetscRandomGetValueReal(r,&rr);CHKERRQ(ierr);
        cellcoor[2*p+1] += rr * 0.5 * dyp;
      }
    }
    
    for (p=0; p<nppcell; p++) {
      pcoor[2*(npoints_init + pcnt)+0] = cellcoor[2*p+0];
      pcoor[2*(npoints_init + pcnt)+1] = cellcoor[2*p+1];
      /*
       This assignment is wrong as it doesn't correspond to the stag cell layout whoch includes an extra cell
       in the local space on the left/up faces - does this matter?
       For projecting points->cell yes
       */
      //pcellid[npoints_init + pcnt] = cellid;
      ierr = DMStagGetLocalElementIndex(dmstag,egidx,&cellid_local);CHKERRQ(ierr);
      pcellid[npoints_init + pcnt] = cellid_local;

      pcnt++;
    }
  }
  ierr = DMSwarmRestoreField(dmswarm,DMSwarmPICField_coor,NULL,NULL,(void**)&pcoor);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(dmswarm,DMSwarmPICField_cellid,NULL,NULL,(void**)&pcellid);CHKERRQ(ierr);
  
  ierr = PetscRandomDestroy(&r);CHKERRQ(ierr);
  ierr = PetscFree(cellcoor);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

/*
 Assign particles coordinates along cell boundaries of the domain. Particle coordinates are assigned face-wise
 
 Input:
 + dmswarm - The particle object
 + face - Identifies which domain face to insert points along
 + factor - Controls the magnitude of the random perturbation applied to each particle position. Must be in range [0,1]
 + points_per_dim - Number of points in x or y to define on each cell face
 + mode - Indicates whether new coordinates should be appending to the existing points in dmswarm, or whether
 the existing points (and size) of dmswarm should be set to zero prior to defining new coordinates

 Output:
 + dmswarm - On exit dmswarm will contain newly defined coordinates
 
 Notes:
 * Assumes constant cell spacing
 
 Collective
*/
PetscErrorCode MPointCoordLayout_DomainFace(DM dmswarm,char face,PetscReal factor,PetscInt points_per_dim,MPointCoordinateInsertMode mode)
{
  DM             dmstag;
  PetscInt       Ng[]={0,0,0};
  PetscErrorCode ierr;
  
  PetscFunctionBeginUser;
  ierr = DMSwarmGetCellDM(dmswarm,&dmstag);CHKERRQ(ierr);
  ierr = DMStagGetGlobalSizes(dmstag,&Ng[0],&Ng[1],NULL);CHKERRQ(ierr);
  switch (face) {
    case 'n':
      ierr = _MPointCoordLayout_FillDomainFace_NS(dmswarm,Ng[1],factor,points_per_dim,mode);CHKERRQ(ierr);
      ierr = DMSwarmMigrate(dmswarm,PETSC_TRUE);CHKERRQ(ierr);
      break;
    case 's':
      ierr = _MPointCoordLayout_FillDomainFace_NS(dmswarm,0,factor,points_per_dim,mode);CHKERRQ(ierr);
      ierr = DMSwarmMigrate(dmswarm,PETSC_TRUE);CHKERRQ(ierr);
      break;
    case 'e':
      ierr = _MPointCoordLayout_FillDomainFace_EW(dmswarm,Ng[0],factor,points_per_dim,mode);CHKERRQ(ierr);
      ierr = DMSwarmMigrate(dmswarm,PETSC_TRUE);CHKERRQ(ierr);
      break;
    case 'w':
      ierr = _MPointCoordLayout_FillDomainFace_EW(dmswarm,0,factor,points_per_dim,mode);CHKERRQ(ierr);
      ierr = DMSwarmMigrate(dmswarm,PETSC_TRUE);CHKERRQ(ierr);
      break;
    default:
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Provided face value of %c is not supported. Must use one of 'n' (north), 's' (south), 'e' (east), 'w' (west)",face);
      break;
  }
  PetscFunctionReturn(0);
}

/*
 Advect paticles using RK1 (forward Euler)
 
 Notes:
 * Performs P0 interpolation over each finite cell for both vx or vy.
 
 Not collective
*/
PetscErrorCode MPoint_AdvectRK1_Private_P0(DM dmswarm,
                                        PetscReal mpfield_coor_k[],
                                        PetscReal mpfield_coor_star[],
                                        PetscReal mpfield_coor_kp1[],DM dmstag,Vec Xlocal,PetscReal dt)
{
  PetscErrorCode    ierr;
  Vec               vp_l;
  const PetscScalar ***LA_vp;
  PetscInt          p,e,npoints;
  PetscInt          *mpfield_cell;
  PetscScalar       **cArrX,**cArrY;
  PetscInt          iPrev,iNext,iCenter,iVxLeft,iVxRight,iVyDown,iVyUp,n[2];
  DM                dm_vp,dm_mpoint;
  
  PetscFunctionBeginUser;
  dm_vp = dmstag;
  dm_mpoint = dmswarm;
  vp_l = Xlocal;
  
  ierr = DMStagGetProductCoordinateArraysRead(dm_vp,&cArrX,&cArrY,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm_vp,DMSTAG_ELEMENT,&iCenter);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm_vp,DMSTAG_LEFT,&iPrev);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm_vp,DMSTAG_RIGHT,&iNext);CHKERRQ(ierr);
  
  ierr = DMStagGetLocationSlot(dm_vp,DMSTAG_LEFT,0,&iVxLeft);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm_vp,DMSTAG_RIGHT,0,&iVxRight);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm_vp,DMSTAG_DOWN,0,&iVyDown);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm_vp,DMSTAG_UP,0,&iVyUp);CHKERRQ(ierr);
  
  ierr = DMStagVecGetArrayRead(dm_vp,vp_l,&LA_vp);CHKERRQ(ierr);
  
  ierr = DMStagGetLocalSizes(dm_vp,&n[0],&n[1],NULL);CHKERRQ(ierr);
  ierr = DMSwarmGetLocalSize(dm_mpoint,&npoints);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dm_mpoint,DMSwarmPICField_cellid,NULL,NULL,(void**)&mpfield_cell);CHKERRQ(ierr);
  for (p=0; p<npoints; p++) {
    PetscReal   coor_p_k[2],coor_p_star[2];
    PetscScalar vel_p_star[2],vLeft,vRight,vUp,vDown;
    PetscScalar x0[2],dx[2],xloc_p[2],xi_p[2];
    PetscInt    ind[2];
    
    e       = mpfield_cell[p];
    coor_p_k[0] = mpfield_coor_k[2*p+0];
    coor_p_k[1] = mpfield_coor_k[2*p+1];
    coor_p_star[0] = mpfield_coor_star[2*p+0];
    coor_p_star[1] = mpfield_coor_star[2*p+1];
    ierr = DMStagGetLocalElementGlobalIndices(dm_vp,e,ind);CHKERRQ(ierr);
    
    /* compute local coordinates: (xp-x0)/dx = (xip+1)/2 */
    x0[0] = cArrX[ind[0]][iPrev];
    x0[1] = cArrY[ind[1]][iPrev];
    
    dx[0] = cArrX[ind[0]][iNext] - x0[0];
    dx[1] = cArrY[ind[1]][iNext] - x0[1];
    
    xloc_p[0] = (coor_p_star[0] - x0[0])/dx[0];
    xloc_p[1] = (coor_p_star[1] - x0[1])/dx[1];
    
    /* Checks (xi_p is only used for this, here) */
    xi_p[0] = 2.0 * xloc_p[0] -1.0;
    xi_p[1] = 2.0 * xloc_p[1] -1.0;
    if (xi_p[0] < -1.0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"value (xi) too small %1.4e [e=%D]",(double)xi_p[0],e);
    if (xi_p[0] >  1.0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"value (xi) too large %1.4e [e=%D]",(double)xi_p[0],e);
    if (xi_p[1] < -1.0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"value (eta) too small %1.4e [e=%D]",(double)xi_p[1],e);
    if (xi_p[1] >  1.0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"value (eta) too large %1.4e [e=%D]",(double)xi_p[1],e);
    
    /* interpolate velocity */
    vLeft  = LA_vp[ind[1]][ind[0]][iVxLeft];
    vRight = LA_vp[ind[1]][ind[0]][iVxRight];
    vUp    = LA_vp[ind[1]][ind[0]][iVyUp];
    vDown  = LA_vp[ind[1]][ind[0]][iVyDown];
    vel_p_star[0] = xloc_p[0]*vRight + (1.0-xloc_p[0])*vLeft;
    vel_p_star[1] = xloc_p[1]*vUp    + (1.0-xloc_p[1])*vDown;
    
    /* Update Coordinates */
    mpfield_coor_kp1[2*p+0] = coor_p_k[0] + dt * vel_p_star[0];
    mpfield_coor_kp1[2*p+1] = coor_p_k[1] + dt * vel_p_star[1];
  }
  
  ierr = DMSwarmRestoreField(dm_mpoint,DMSwarmPICField_cellid,NULL,NULL,(void**)&mpfield_cell);CHKERRQ(ierr);
  ierr = DMStagRestoreProductCoordinateArraysRead(dm_vp,&cArrX,&cArrY,NULL);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArrayRead(dm_vp,vp_l,&LA_vp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
 Advect particles with RK1 (forward Euler)
 Notes:
 * Performs bilinear interpolation between vx or vy points
 
 Not collective
*/
PetscErrorCode MPoint_AdvectRK1_Private(DM dmswarm,
                                        PetscReal mpfield_coor_k[],
                                        PetscReal mpfield_coor_star[],
                                        PetscReal mpfield_coor_kp1[],DM dmstag,Vec Xlocal,PetscReal dt)
{
  PetscErrorCode    ierr;
  Vec               vp_l;
  const PetscScalar ***LA_vp;
  PetscInt          p,e,npoints;
  PetscInt          *mpfield_cell;
  PetscScalar       **cArrX,**cArrY;
  PetscInt          iPrev,iNext,iCenter,iVxLeft,iVxRight,iVyDown,iVyUp,n[2],N[2];
  DM                dm_vp,dm_mpoint;
  
  PetscFunctionBeginUser;
  dm_vp = dmstag;
  dm_mpoint = dmswarm;
  vp_l = Xlocal;
  
  ierr = DMStagGetProductCoordinateArraysRead(dm_vp,&cArrX,&cArrY,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm_vp,DMSTAG_ELEMENT,&iCenter);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm_vp,DMSTAG_LEFT,&iPrev);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm_vp,DMSTAG_RIGHT,&iNext);CHKERRQ(ierr);
  
  ierr = DMStagGetLocationSlot(dm_vp,DMSTAG_LEFT,0,&iVxLeft);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm_vp,DMSTAG_RIGHT,0,&iVxRight);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm_vp,DMSTAG_DOWN,0,&iVyDown);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm_vp,DMSTAG_UP,0,&iVyUp);CHKERRQ(ierr);
  
  ierr = DMStagVecGetArrayRead(dm_vp,vp_l,&LA_vp);CHKERRQ(ierr);
  
  ierr = DMStagGetGlobalSizes(dm_vp,&N[0],&N[1],NULL);CHKERRQ(ierr);
  ierr = DMStagGetLocalSizes(dm_vp,&n[0],&n[1],NULL);CHKERRQ(ierr);
  ierr = DMSwarmGetLocalSize(dm_mpoint,&npoints);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dm_mpoint,DMSwarmPICField_cellid,NULL,NULL,(void**)&mpfield_cell);CHKERRQ(ierr);
  for (p=0; p<npoints; p++) {
    PetscReal   coor_p_k[2],coor_p_star[2];
    PetscScalar vel_p_star[2];
    PetscScalar x0[2],dx[2],xi_p[2];
    PetscInt    ind[2];
    PetscReal   v_vert[4];

    e = mpfield_cell[p];
    coor_p_k[0] = mpfield_coor_k[2*p+0];
    coor_p_k[1] = mpfield_coor_k[2*p+1];
    coor_p_star[0] = mpfield_coor_star[2*p+0];
    coor_p_star[1] = mpfield_coor_star[2*p+1];
    ierr = DMStagGetLocalElementGlobalIndices(dm_vp,e,ind);CHKERRQ(ierr);
    
    /* compute local coordinates: (xp-x0)/dx = (xip+1)/2 */
    x0[0] = cArrX[ind[0]][iPrev];
    x0[1] = cArrY[ind[1]][iPrev];
    
    dx[0] = cArrX[ind[0]][iNext] - x0[0];
    dx[1] = cArrY[ind[1]][iNext] - x0[1];
    
    /* Checks (xi_p is only used for this, here) */
    xi_p[0] = 2.0 * (coor_p_star[0] - x0[0])/dx[0] - 1.0;
    xi_p[1] = 2.0 * (coor_p_star[1] - x0[1])/dx[1] - 1.0;
    
    /* interpolate velocity */
    vel_p_star[0] = vel_p_star[1] = 0.0;
    /* vx interpolation (bilinear) */
    {
      PetscInt II,JJ;
      PetscReal cv_x0[2];
      
      cv_x0[0] = x0[0];
      cv_x0[1] = x0[1];
      II = ind[0];
      JJ = ind[1];
      
      if (xi_p[1] < 0.0) {
        v_vert[2] = LA_vp[JJ][II][iVxLeft];               v_vert[3] = LA_vp[JJ][II][iVxRight];
        v_vert[0] = LA_vp[PetscMax(JJ-1,0)][II][iVxLeft]; v_vert[1] = LA_vp[PetscMax(JJ-1,0)][II][iVxRight];
        cv_x0[1] -= 0.5 * dx[1]; /* shift coordinate down by 1/2 cell depth */
      } else {
        v_vert[2] = LA_vp[PetscMin(JJ+1,N[1]-1)][II][iVxLeft]; v_vert[3] = LA_vp[PetscMin(JJ+1,N[1]-1)][II][iVxRight];
        v_vert[0] = LA_vp[JJ][II][iVxLeft];                    v_vert[1] = LA_vp[JJ][II][iVxRight];
        cv_x0[1] += 0.5 * dx[1];
      }
      /* compute local coordinates wrt to the box connecting vx points */
      xi_p[0] = 2.0 * (coor_p_star[0] - cv_x0[0])/dx[0] - 1.0;
      xi_p[1] = 2.0 * (coor_p_star[1] - cv_x0[1])/dx[1] - 1.0;
      
      vel_p_star[0] = 0.0;
      vel_p_star[0] += 0.25 * (1.0 - xi_p[0]) * (1.0 - xi_p[1]) * v_vert[0];
      vel_p_star[0] += 0.25 * (1.0 + xi_p[0]) * (1.0 - xi_p[1]) * v_vert[1];
      vel_p_star[0] += 0.25 * (1.0 - xi_p[0]) * (1.0 + xi_p[1]) * v_vert[2];
      vel_p_star[0] += 0.25 * (1.0 + xi_p[0]) * (1.0 + xi_p[1]) * v_vert[3];
    }

    /* vy interpolation (bilinear) */
    {
      PetscInt II,JJ;
      PetscReal cv_x0[2];
      
      cv_x0[0] = x0[0];
      cv_x0[1] = x0[1];
      II = ind[0];
      JJ = ind[1];
      
      if (xi_p[0] < 0.0) {
        v_vert[2] = LA_vp[JJ][PetscMax(II-1,0)][iVyUp];    v_vert[3] = LA_vp[JJ][II][iVyUp];
        v_vert[0] = LA_vp[JJ][PetscMax(II-1,0)][iVyDown];  v_vert[1] = LA_vp[JJ][II][iVyDown];
        cv_x0[0] -= 0.5 * dx[0];
      } else {
        v_vert[2] = LA_vp[JJ][II][iVyUp];    v_vert[3] = LA_vp[JJ][PetscMin(II+1,N[0]-1)][iVyUp];
        v_vert[0] = LA_vp[JJ][II][iVyDown];  v_vert[1] = LA_vp[JJ][PetscMin(II+1,N[0]-1)][iVyDown];
        cv_x0[0] += 0.5 * dx[0];
      }
      /* compute local coordinates wrt to the box connecting vy points */
      xi_p[0] = 2.0 * (coor_p_star[0] - cv_x0[0])/dx[0] - 1.0;
      xi_p[1] = 2.0 * (coor_p_star[1] - cv_x0[1])/dx[1] - 1.0;
      
      vel_p_star[1] = 0.0;
      vel_p_star[1] += 0.25 * (1.0 - xi_p[0]) * (1.0 - xi_p[1]) * v_vert[0];
      vel_p_star[1] += 0.25 * (1.0 + xi_p[0]) * (1.0 - xi_p[1]) * v_vert[1];
      vel_p_star[1] += 0.25 * (1.0 - xi_p[0]) * (1.0 + xi_p[1]) * v_vert[2];
      vel_p_star[1] += 0.25 * (1.0 + xi_p[0]) * (1.0 + xi_p[1]) * v_vert[3];
    }
    
    /* Update Coordinates */
    mpfield_coor_kp1[2*p+0] = coor_p_k[0] + dt * vel_p_star[0];
    mpfield_coor_kp1[2*p+1] = coor_p_k[1] + dt * vel_p_star[1];
  }
  
  ierr = DMSwarmRestoreField(dm_mpoint,DMSwarmPICField_cellid,NULL,NULL,(void**)&mpfield_cell);CHKERRQ(ierr);
  ierr = DMStagRestoreProductCoordinateArraysRead(dm_vp,&cArrX,&cArrY,NULL);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArrayRead(dm_vp,vp_l,&LA_vp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
 Advect particles using RK1 (forward Euler)
 
 Input:
 + dmswarm - The particle object
 + X - The vector of velocity/pressure values
 + dt - The timestep
 
 Collective
*/
PetscErrorCode MPoint_AdvectRK1(DM dmswarm,DM dmstag,Vec X,PetscReal dt)
{
  Vec            Xlocal;
  PetscReal      *mpfield_coor;
  PetscErrorCode ierr;
  
  PetscFunctionBeginUser;
  ierr = DMGetLocalVector(dmstag,&Xlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dmstag,X,INSERT_VALUES,Xlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dmstag,X,INSERT_VALUES,Xlocal);CHKERRQ(ierr);

  ierr = DMSwarmGetField(dmswarm,DMSwarmPICField_coor,NULL,NULL,(void**)&mpfield_coor);CHKERRQ(ierr);
  
  ierr = MPoint_AdvectRK1_Private(dmswarm,mpfield_coor,mpfield_coor,mpfield_coor,dmstag,Xlocal,dt);CHKERRQ(ierr);
  
  ierr = DMSwarmRestoreField(dmswarm,DMSwarmPICField_coor,NULL,NULL,(void**)&mpfield_coor);CHKERRQ(ierr);
    
  ierr = DMRestoreLocalVector(dmstag,&Xlocal);CHKERRQ(ierr);
  
  /* scatter */
  ierr = DMSwarmMigrate(dmswarm,PETSC_TRUE);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

#if 0
/* needs updating - still uses P0 interpolation for velocity */
/*
 Not collective
*/
PetscErrorCode MPoint_AdvectRK2_SEQ_Private(DM dmswarm,
                                        PetscReal mpfield_coor_k[],
                                        PetscReal mpfield_coor_kp1[],DM dmstag,Vec Xlocal,PetscReal dt)
{
  PetscErrorCode    ierr;
  Vec               vp_l;
  const PetscScalar ***LA_vp;
  PetscInt          p,e,npoints;
  PetscInt          *mpfield_cell;
  PetscScalar       **cArrX,**cArrY;
  PetscInt          iPrev,iNext,iCenter,iVxLeft,iVxRight,iVyDown,iVyUp,n[2];
  DM                dm_vp,dm_mpoint;
  
  PetscFunctionBeginUser;
  dm_vp = dmstag;
  dm_mpoint = dmswarm;
  vp_l = Xlocal;
  
  ierr = DMStagGetProductCoordinateArraysRead(dm_vp,&cArrX,&cArrY,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm_vp,DMSTAG_ELEMENT,&iCenter);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm_vp,DMSTAG_LEFT,&iPrev);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm_vp,DMSTAG_RIGHT,&iNext);CHKERRQ(ierr);
  
  ierr = DMStagGetLocationSlot(dm_vp,DMSTAG_LEFT,0,&iVxLeft);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm_vp,DMSTAG_RIGHT,0,&iVxRight);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm_vp,DMSTAG_DOWN,0,&iVyDown);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm_vp,DMSTAG_UP,0,&iVyUp);CHKERRQ(ierr);
  
  ierr = DMStagVecGetArrayRead(dm_vp,vp_l,&LA_vp);CHKERRQ(ierr);
  
  ierr = DMStagGetLocalSizes(dm_vp,&n[0],&n[1],NULL);CHKERRQ(ierr);
  ierr = DMSwarmGetLocalSize(dm_mpoint,&npoints);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dm_mpoint,DMSwarmPICField_cellid,NULL,NULL,(void**)&mpfield_cell);CHKERRQ(ierr);
  for (p=0; p<npoints; p++) {
    PetscReal   coor_p_k[2],coor_p_star[2];
    PetscScalar vel_p[2],vel_p_star[2],vLeft,vRight,vUp,vDown;
    PetscScalar x0[2],dx[2],xloc_p[2],xi_p[2];
    PetscInt    ind[2];
    
    e       = mpfield_cell[p];
    coor_p_k[0] = mpfield_coor_k[2*p+0];
    coor_p_k[1] = mpfield_coor_k[2*p+1];
    ierr = DMStagGetLocalElementGlobalIndices(dm_vp,e,ind);CHKERRQ(ierr);
    
    /* compute local coordinates: (xp-x0)/dx = (xip+1)/2 */
    x0[0] = cArrX[ind[0]][iPrev];
    x0[1] = cArrY[ind[1]][iPrev];
    
    dx[0] = cArrX[ind[0]][iNext] - x0[0];
    dx[1] = cArrY[ind[1]][iNext] - x0[1];
    
    
    /* STAGE 1 */
    { // UPDATE TO BILINEAR
      xloc_p[0] = (coor_p_k[0] - x0[0])/dx[0];
      xloc_p[1] = (coor_p_k[1] - x0[1])/dx[1];
      
      /* Checks (xi_p is only used for this, here) */
      xi_p[0] = 2.0 * xloc_p[0] -1.0;
      xi_p[1] = 2.0 * xloc_p[1] -1.0;
      if (xi_p[0] < -1.0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"value (xi) too small %1.4e [e=%D]",(double)xi_p[0],e);
      if (xi_p[0] >  1.0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"value (xi) too large %1.4e [e=%D]",(double)xi_p[0],e);
      if (xi_p[1] < -1.0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"value (eta) too small %1.4e [e=%D]",(double)xi_p[1],e);
      if (xi_p[1] >  1.0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"value (eta) too large %1.4e [e=%D]",(double)xi_p[1],e);
      
      /* interpolate velocity */
      vLeft  = LA_vp[ind[1]][ind[0]][iVxLeft];
      vRight = LA_vp[ind[1]][ind[0]][iVxRight];
      vUp    = LA_vp[ind[1]][ind[0]][iVyUp];
      vDown  = LA_vp[ind[1]][ind[0]][iVyDown];
      vel_p[0] = xloc_p[0]*vRight + (1.0-xloc_p[0])*vLeft;
      vel_p[1] = xloc_p[1]*vUp    + (1.0-xloc_p[1])*vDown;
    }
    
    /* Update Coordinates */
    coor_p_star[0] = coor_p_k[0] + 0.5 * dt * vel_p[0];
    coor_p_star[1] = coor_p_k[1] + 0.5 * dt * vel_p[1];
    

    /* STAGE 2 */
    { // UPDATE TO BILINEAR
      xloc_p[0] = (coor_p_star[0] - x0[0])/dx[0];
      xloc_p[1] = (coor_p_star[1] - x0[1])/dx[1];
      
      /* Checks (xi_p is only used for this, here) */
      xi_p[0] = 2.0 * xloc_p[0] -1.0;
      xi_p[1] = 2.0 * xloc_p[1] -1.0;
      if (xi_p[0] < -1.0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"value (xi) too small %1.4e [e=%D]",(double)xi_p[0],e);
      if (xi_p[0] >  1.0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"value (xi) too large %1.4e [e=%D]",(double)xi_p[0],e);
      if (xi_p[1] < -1.0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"value (eta) too small %1.4e [e=%D]",(double)xi_p[1],e);
      if (xi_p[1] >  1.0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"value (eta) too large %1.4e [e=%D]",(double)xi_p[1],e);
      
      /* interpolate velocity */
      vLeft  = LA_vp[ind[1]][ind[0]][iVxLeft];
      vRight = LA_vp[ind[1]][ind[0]][iVxRight];
      vUp    = LA_vp[ind[1]][ind[0]][iVyUp];
      vDown  = LA_vp[ind[1]][ind[0]][iVyDown];
      vel_p_star[0] = xloc_p[0]*vRight + (1.0-xloc_p[0])*vLeft;
      vel_p_star[1] = xloc_p[1]*vUp    + (1.0-xloc_p[1])*vDown;
    }
    
    /* Update Coordinates */
    mpfield_coor_kp1[2*p+0] = coor_p_k[0] + dt * vel_p_star[0];
    mpfield_coor_kp1[2*p+1] = coor_p_k[1] + dt * vel_p_star[1];
  }
  
  ierr = DMSwarmRestoreField(dm_mpoint,DMSwarmPICField_cellid,NULL,NULL,(void**)&mpfield_cell);CHKERRQ(ierr);
  ierr = DMStagRestoreProductCoordinateArraysRead(dm_vp,&cArrX,&cArrY,NULL);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArrayRead(dm_vp,vp_l,&LA_vp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
 Collective
*/
PetscErrorCode MPoint_AdvectRK2_MPI(DM dmswarmI,DM dmstag,Vec Xlocal,PetscReal dt)
{
  PetscInt es[] = {0,0},nele[] = {0,0};
  PetscInt *mask;
  DM dmswarmB;
  
  PetscFunctionBeginUser;
  ierr = DMStagGetCorners(dmstag,&es[0],&es[1],NULL,&nele[0],&nele[1],NULL,NULL,NULL,NULL);
  es[0] = es[1] = 0;
  ierr = PetscCalloc1(nele[0]*nele[1],&mask);CHKERRQ(ierr);
  for (ej=1; ej<nele[1]-1; ej++) {
    for (ei=1; ei<nele[0]-1; ei++) {
      mask[ei + ej * nele[0]] = 1;
    }
  }
  
  
  nestimate = 16 * ( 2 * nele[0] + 2*(nele[1] - 2) );
  
  // create swarm for points living in the boundary
  ierr = DMStagPICCreateDMSwarm(dmstag,&dmswarmB);CHKERRQ(ierr);
  ierr = DMSwarmDuplicateRegisteredFields(dmswarmB);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(dmswarmB,"coor_prime",2,PETSC_REAL);CHKERRQ(ierr);
  ierr = DMStagPICFinalize(dmswarmB);CHKERRQ(ierr);
  ierr = DMSwarmSetLocalSizes(dmswarmB,0,-1);CHKERRQ(ierr);
  
  // count and create mask
  for (p=0; p<npoints; p++) {
    
    
    
  }
  
  // copy
  ierr = DMSwarmCopySubsetFieldValues(dmswarmI,PetscInt np,PetscInt list[],dmswarmB,&copy_occurred);CHKERRQ(ierr);
  
  // delete
  ierr = DMSwarmRemovePoints(dmswarmI,PetscInt np,PetscInt list[]);CHKERRQ(ierr);

  
  PetscReal *mpfield_coor,*mpfield_prime;
  
  
  
  
  // advect b
  ierr = DMSwarmGetField(dmswarmB,DMSwarmPICField_coor,NULL,NULL,(void**)&mpfield_coor);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dmswarmB,"coor_prime",NULL,NULL,(void**)&mpfield_coor_prime);CHKERRQ(ierr);
  
  ierr = MPoint_AdvectRK1_Private(dmswarmB,mpfield_coor,mpfield_coor,mpfield_coor_prime,dmstag,Xlocal,0.5 * dt);CHKERRQ(ierr);
  
  ierr = DMSwarmRestoreField(dmswarmB,"coor_prime",NULL,NULL,(void**)&mpfield_coor_prime);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(dmswarmB,DMSwarmPICField_coor,NULL,NULL,(void**)&mpfield_coor);CHKERRQ(ierr);
  // migrate b
  ierr = DMSwarmMigrate(dmswarmB,PETSC_TRUE);CHKERRQ(ierr);
  
  // advect b
  ierr = DMSwarmGetField(dmswarmB,DMSwarmPICField_coor,NULL,NULL,(void**)&mpfield_coor);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dmswarmB,"coor_prime",NULL,NULL,(void**)&mpfield_coor_prime);CHKERRQ(ierr);
  
  ierr = MPoint_AdvectRK1_Private(dmswarmB,mpfield_coor,mpfield_coor_prime,mpfield_coor,dmstag,Xlocal,0.5 * dt);CHKERRQ(ierr);
  
  ierr = DMSwarmRestoreField(dmswarmB,"coor_prime",NULL,NULL,(void**)&mpfield_coor_prime);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(dmswarmB,DMSwarmPICField_coor,NULL,NULL,(void**)&mpfield_coor);CHKERRQ(ierr);
  // migrate b
  ierr = DMSwarmMigrate(dmswarmB,PETSC_TRUE);CHKERRQ(ierr);

  ierr = MPoint_AdvectRK2_SEQ_Private(dmswarmI,dmstag,Xlocal,dt);

  
  // insert B into I
  ierr = DMSwarmCopyFieldValues(dmswarmB,dmswarmI,&copy_occurred);CHKERRQ(ierr);

  
  ierr = DMDestroy(&dmswarmB);CHKERRQ(ierr);
  ierr = PetscFree(list);CHKERRQ(ierr);
  ierr = PetscFree(mask);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

/*
 Not collective
*/
PetscErrorCode MPoint_AdvectRK2_SEQ(DM dmswarm,DM dmstag,Vec Xlocal,PetscReal dt)
{
  PetscReal      *mpfield_coor;
  
  PetscFunctionBeginUser;
  ierr = DMSwarmGetField(dmswarm,DMSwarmPICField_coor,NULL,NULL,(void**)&mpfield_coor);CHKERRQ(ierr);
  ierr = MPoint_AdvectRK2_SEQ_Private(dmswarm,mpfield_coor,mpfield_coor,dmstag,Xlocal,dt);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(dmswarm,DMSwarmPICField_coor,NULL,NULL,(void**)&mpfield_coor);CHKERRQ(ierr);
  /* point location */
  
  PetscFunctionReturn(0);
}

/*
 Collective
*/
PetscErrorCode MPoint_AdvectRK2(DM dmswarm,DM dmstag,Vec X,PetscReal dt)
{
  Vec            Xlocal;
  PetscErrorCode ierr;
  
  PetscFunctionBeginUser;
  ierr = DMGetLocalVector(dmstag,&Xlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dmstag,X,INSERT_VALUES,Xlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dmstag,X,INSERT_VALUES,Xlocal);CHKERRQ(ierr);
  

  MPoint_AdvectRK2_SEQ(dmswarm,dmstag,Xlocal,dt);
  MPoint_AdvectRK2_MPI(dmswarm,dmstag,Xlocal,dt);
  
  
  ierr = DMRestoreLocalVector(dmstag,&Xlocal);CHKERRQ(ierr);
  
  
  PetscFunctionReturn(0);
}

#endif





/* DataBucket utilities */
#include <petsc/private/dmswarmimpl.h>    /*I   "petscdmswarm.h"   I*/
#include "../src/dm/impls/swarm/data_bucket.h"

/*
 Query if a DMSwarm defines a particular property
 
 Not collective
*/
PetscErrorCode DMSwarmQueryField(DM dm,const char fieldname[],PetscBool *found)
{
  DM_Swarm  *s = (DM_Swarm*)dm->data;
  PetscInt index = -1;
  PetscErrorCode ierr;
  
  PetscFunctionBeginUser;
  *found = PETSC_FALSE;
  DMSwarmDataFieldStringFindInList(fieldname,s->db->nfields,s->db->field,&index);
  if (index >= 0) *found = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*
 Registered all fields in dmA within in dmB
 
 Notes:
 * Data is not copied. Only field registration is performed
 * A check for duplicate fields is performed. Any duplicates are ignored (without error)
 
 Not collective
*/
PetscErrorCode DMSwarmDuplicateRegisteredFields(DM dmA,DM dmB)
{
  DM_Swarm  *swarmA = (DM_Swarm*)dmA->data;
  DMSwarmDataBucket dbA;
  PetscInt f;
  PetscBool found;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  dbA = swarmA->db;
  for (f=0; f<dbA->nfields; f++) {
    ierr = DMSwarmQueryField(dmB,dbA->field[f]->name,&found);CHKERRQ(ierr);
    if (!found) {
      ierr = DMSwarmRegisterPetscDatatypeField(dmB,dbA->field[f]->name,dbA->field[f]->bs,dbA->field[f]->petsc_type);CHKERRQ(ierr);
    }
  }
  
  PetscFunctionReturn(0);
}

/*
 Copy all fields values in dmA into dmB.
 
 Notes:
 * Only fields commong to A and B are copied.
 * Copied fields will be appended into dmB
 
 fieldsA[] = {"0","1","2"}
 fieldsB[] = {"0","1","2"} ==> A and B match, SWARM_FIELDS_SAME

 fieldsA[] = {"0","1","2"}
 fieldsB[] = {"0","1"} ==> B is SWARM_FIELDS_SUBSET

 fieldsA[] = {"0","1"}
 fieldsB[] = {"0","1","2"} ==> B is SWARM_FIELDS_SUPERSET

 fieldsA[] = {"0","1"}
 fieldsB[] = {""2"} ==> A B is SWARM_FIELDS_DISJOINT

 fieldsA[] = {"0","1","3"}
 fieldsB[] = {"0","1","2"} ==> B is SWARM_FIELDS_SUBSET

 Not collective
*/
PetscErrorCode DMSwarmCopyFieldValues(DM dmA,DM dmB,PetscBool *copy_occurred)
{
  PetscErrorCode ierr;
  PetscInt lenA,lenB,lenB_new,nfA,nfB;
  DM_Swarm  *swarmA = (DM_Swarm*)dmA->data;
  DM_Swarm  *swarmB = (DM_Swarm*)dmB->data;
  DMSwarmDataBucket dbA,dbB;
  PetscInt *fieldfrom,*fieldto;
  PetscInt f;
  PetscBool detected_common_members=PETSC_FALSE;
  
  PetscFunctionBeginUser;
  *copy_occurred = PETSC_FALSE;
  ierr = DMSwarmGetLocalSize(dmA,&lenA);CHKERRQ(ierr);
  ierr = DMSwarmGetLocalSize(dmB,&lenB);CHKERRQ(ierr);
  dbA = swarmA->db;
  dbB = swarmB->db;
  nfA = dbA->nfields;
  nfB = dbB->nfields;
  
  ierr = PetscMalloc1(nfA,&fieldfrom);CHKERRQ(ierr);
  ierr = PetscMalloc1(nfA,&fieldto);CHKERRQ(ierr);
  for (f=0; f<nfA; f++) {
    fieldfrom[f] = -1;
    fieldto[f] = -1;
  }

  for (f=0; f<nfA; f++) {
    PetscInt index = -1;
    DMSwarmDataFieldStringFindInList(dbA->field[f]->name, dbB->nfields,dbB->field,&index);
    if (index >= 0) {
      fieldfrom[f] = f;
      fieldto[f] = index;
      detected_common_members = PETSC_TRUE;
    }
  }
  
  if (detected_common_members) {
    PetscInt p;
    
    ierr = DMSwarmSetLocalSizes(dmB,lenB+lenA,-1);CHKERRQ(ierr);
    ierr = DMSwarmGetLocalSize(dmB,&lenB_new);CHKERRQ(ierr);
    
    for (f=0; f<nfA; f++) {
      if (fieldfrom[f] != -1) {

        for (p=0; p<lenA; p++) {
          ierr = DMSwarmDataFieldCopyPoint(p,       dbA->field[ fieldfrom[f] ],
                                           lenB + p,dbB->field[ fieldto[f] ]);CHKERRQ(ierr);
        }
      }
    }
    *copy_occurred = PETSC_TRUE;
  }
  
  ierr = PetscFree(fieldfrom);CHKERRQ(ierr);
  ierr = PetscFree(fieldto);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

/*
 Copy a sub-set of all field values from dmA into dmB
 
 Input:
 + dmA - DMSwarm object
 + np - Number of points to copy
 + list - Array of point indices to copy
 
 Output:
 + dmB - DMSwarm object
 + copy_occurred - flag indicating if dmB was modified (e.g. if any copy was performed)
 
 Not collective
*/
PetscErrorCode DMSwarmCopySubsetFieldValues(DM dmA,PetscInt np,PetscInt list[],DM dmB,PetscBool *copy_occurred)
{
  PetscErrorCode ierr;
  PetscInt lenA,lenB,lenB_new,nfA,nfB;
  DM_Swarm  *swarmA = (DM_Swarm*)dmA->data;
  DM_Swarm  *swarmB = (DM_Swarm*)dmB->data;
  DMSwarmDataBucket dbA,dbB;
  PetscInt *fieldfrom,*fieldto;
  PetscInt f;
  PetscBool detected_common_members=PETSC_FALSE;
  
  PetscFunctionBeginUser;
  *copy_occurred = PETSC_FALSE;
  /*ierr = DMSwarmGetLocalSize(dmA,&lenA);CHKERRQ(ierr);*/
  lenA = np;
  ierr = DMSwarmGetLocalSize(dmB,&lenB);CHKERRQ(ierr);
  dbA = swarmA->db;
  dbB = swarmB->db;
  nfA = dbA->nfields;
  nfB = dbB->nfields;
  
  ierr = PetscMalloc1(nfA,&fieldfrom);CHKERRQ(ierr);
  ierr = PetscMalloc1(nfA,&fieldto);CHKERRQ(ierr);
  for (f=0; f<nfA; f++) {
    fieldfrom[f] = -1;
    fieldto[f] = -1;
  }
  
  for (f=0; f<nfA; f++) {
    PetscInt index = -1;
    DMSwarmDataFieldStringFindInList(dbA->field[f]->name, dbB->nfields,dbB->field,&index);
    if (index >= 0) {
      fieldfrom[f] = f;
      fieldto[f] = index;
      detected_common_members = PETSC_TRUE;
    }
  }
  
  if (detected_common_members) {
    PetscInt p,index;
    
    ierr = DMSwarmSetLocalSizes(dmB,lenB+lenA,-1);CHKERRQ(ierr);
    ierr = DMSwarmGetLocalSize(dmB,&lenB_new);CHKERRQ(ierr);
    
    for (f=0; f<nfA; f++) {
      if (fieldfrom[f] != -1) {
        
        for (p=0; p<np; p++) {
          index = list[p];
          
          ierr = DMSwarmDataFieldCopyPoint(index,   dbA->field[ fieldfrom[f] ],
                                           lenB + p,dbB->field[ fieldto[f] ]);CHKERRQ(ierr);
        }
      }
    }
    *copy_occurred = PETSC_TRUE;
  }
  
  ierr = PetscFree(fieldfrom);CHKERRQ(ierr);
  ierr = PetscFree(fieldto);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode v0_DMSwarmRemovePoints(DM dm,PetscInt np,PetscInt list[])
{
  PetscInt p;
  PetscErrorCode ierr;
  
  PetscFunctionBeginUser;
  for (p=0; p<np; p++) {
    PetscInt index = list[p];
    ierr = DMSwarmRemovePointAtIndex(dm,index);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
 Remove a sub-set of points from a DMSwarm object
 
 Input:
 + dm - DMSwarm object
 + np - Number of points to remove
 + list - Array of point indices which should be removed
 
 Output:
 + dm - The modified DMSwarm
 
 Not collective
*/
PetscErrorCode DMSwarmRemovePoints(DM dm,PetscInt np,PetscInt list[])
{
  const long kill_point = -9;
  PetscInt p,npoints,endmost_valid_pid;
  long *pid,*pid_ref=NULL;
  PetscErrorCode ierr;
  
  PetscFunctionBeginUser;
  ierr = PetscSortInt(np,list);CHKERRQ(ierr);
  
  ierr = DMSwarmGetLocalSize(dm,&npoints);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dm,DMSwarmField_pid,NULL,NULL,(void**)&pid_ref);CHKERRQ(ierr);
  pid = pid_ref;
  ierr = DMSwarmRestoreField(dm,DMSwarmField_pid,NULL,NULL,(void**)&pid_ref);CHKERRQ(ierr);
  
  for (p=0; p<np; p++) {
    PetscInt index = list[p];
    pid[index] = kill_point;
  }
  
  /* set initial value */
  endmost_valid_pid = npoints - 1;
  
  /* check and recompute if required */
  if (pid[endmost_valid_pid] == kill_point) {
    PetscInt pr;
    for (pr=endmost_valid_pid; pr >= 0; pr--) {
      if (pid[pr] != kill_point) {
        endmost_valid_pid = pr;
        break;
      }
    }
  }
  
  for (p=0; p<np; p++) {
    PetscInt from,to;
    
    from = endmost_valid_pid;
    to = list[p];
    ierr = DMSwarmCopyPoint(dm,from,to);CHKERRQ(ierr);
    
    endmost_valid_pid--;
    if (pid[endmost_valid_pid] == kill_point) {
      PetscInt pr;
      for (pr=endmost_valid_pid; pr >= 0; pr--) {
        if (pid[pr] != kill_point) {
          endmost_valid_pid = pr;
          break;
        }
      }
    }
    
  }

  ierr = DMSwarmSetLocalSizes(dm,npoints-np,-1);CHKERRQ(ierr);

  /* sanity check */
  ierr = DMSwarmGetLocalSize(dm,&npoints);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dm,DMSwarmField_pid,NULL,NULL,(void**)&pid_ref);CHKERRQ(ierr);
  for (p=0; p<npoints; p++) {
    if (pid_ref[p] == kill_point) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Point at index %D should have been deleted",p);
  }
  ierr = DMSwarmRestoreField(dm,DMSwarmField_pid,NULL,NULL,(void**)&pid_ref);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*
 Assign a constant value to all entries associated with a specific DMSwarm field

 Input:
 + dm - DMSwarm
 + fieldname - Textual name of registered DMSwarm field
 + alpha - Constant value to set into field
 
 Output:
 + dm - DMSwarm with modified field values
 
 Not collective
*/
PetscErrorCode DMSwarmFieldSet(DM dm,const char fieldname[],PetscReal alpha)
{
  PetscInt p,len,bs;
  PetscDataType type;
  PetscReal *pfield;
  PetscErrorCode ierr;
  
  PetscFunctionBeginUser;
  ierr = DMSwarmGetLocalSize(dm,&len);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dm,fieldname,&bs,&type,(void**)&pfield);CHKERRQ(ierr);
  if (type != PETSC_REAL) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Type must be PETSC_REAL for field %s",fieldname);
  len = len * bs;
  for (p=0; p<len; p++) {
    pfield[p] = alpha;
  }
  ierr = DMSwarmRestoreField(dm,fieldname,NULL,NULL,(void**)&pfield);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
 Assign a constant value to entries associated with a specific DMSwarm field with indices specified by the range [ps,pe)
 
 Input:
 + dm - DMSwarm
 + fieldname - Textual name of registered DMSwarm field
 + ps - Start index
 + pe - End index
 + alpha - Constant value to set into field
 
 Output:
 + dm - DMSwarm with modified field values
 
 Not collective
*/
PetscErrorCode DMSwarmFieldSetWithRange(DM dm,const char fieldname[],PetscInt ps,PetscInt pe,PetscReal alpha)
{
  PetscInt p,len,bs;
  PetscDataType type;
  PetscReal *pfield;
  PetscErrorCode ierr;
  
  PetscFunctionBeginUser;
  ierr = DMSwarmGetLocalSize(dm,&len);CHKERRQ(ierr);
  if (ps < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Range too small start(%D) < 0",ps);
  if (pe > len) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"Range too larage end(%D) > len(%D)",pe,len);
  ierr = DMSwarmGetField(dm,fieldname,&bs,&type,(void**)&pfield);CHKERRQ(ierr);
  if (type != PETSC_REAL) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Type must be PETSC_REAL for field %s",fieldname);
  for (p=bs*ps; p<bs*pe; p++) {
    pfield[p] = alpha;
  }
  ierr = DMSwarmRestoreField(dm,fieldname,NULL,NULL,(void**)&pfield);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
 Assign a constant value to entries associated with a specific DMSwarm field with indices specified by a list
 
 Input:
 + dm - DMSwarm
 + fieldname - Textual name of registered DMSwarm field
 + np - Number of point indices to set
 + list - Array of point indices
 + alpha - Constant value to set into field
 
 Output:
 + dm - DMSwarm with modified field values
 
 Not collective
*/
PetscErrorCode DMSwarmFieldSetWithList(DM dm,const char fieldname[],PetscInt np,PetscInt list[],PetscReal alpha)
{
  PetscInt p,len,bs,b;
  PetscDataType type;
  PetscReal *pfield;
  PetscErrorCode ierr;
  
  PetscFunctionBeginUser;
  ierr = DMSwarmGetLocalSize(dm,&len);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dm,fieldname,&bs,&type,(void**)&pfield);CHKERRQ(ierr);
  if (type != PETSC_REAL) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Type must be PETSC_REAL for field %s",fieldname);
  for (p=0; p<np; p++) {
    PetscInt index = list[p];
    if (index < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Index too small start(%D) < 0",index);
    if (index > len) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"Index too larage end(%D) > len(%D)",index,len);
    for (b=0; b<bs; b++) {
      pfield[bs*index+b] = alpha;
    }
  }
  ierr = DMSwarmRestoreField(dm,fieldname,NULL,NULL,(void**)&pfield);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

