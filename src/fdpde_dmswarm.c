#include "fdpde_dmswarm.h"

// ---------------------------------------
PetscErrorCode DMStagPICCreateDMSwarm(DM dmstag,DM *s)
{
  DM             dmswarm;
  PetscFunctionBegin;

  PetscCall(DMSetPointLocation(dmstag,DMLocatePoints_Stag));
  
  PetscCall(DMCreate(PetscObjectComm((PetscObject)dmstag),&dmswarm));
  PetscCall(DMSetType(dmswarm,DMSWARM));
  PetscCall(DMSetDimension(dmswarm,2));
  PetscCall(DMSwarmSetType(dmswarm,DMSWARM_PIC));
  PetscCall(DMSwarmSetCellDM(dmswarm,dmstag));
  *s = dmswarm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
PetscErrorCode DMStagPICFinalize(DM dmswarm)
{
  DM             dmstag;
  PetscInt       n[]={0,0},nel;
  PetscFunctionBegin;

  PetscCall(DMSwarmFinalizeFieldRegister(dmswarm));
  PetscCall(DMSwarmGetCellDM(dmswarm,&dmstag));
  PetscCall(DMStagGetLocalSizes(dmstag,&n[0],&n[1],NULL));
  nel = n[0]*n[1];
  PetscCall(DMSwarmSetLocalSizes(dmswarm,nel*16,100));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
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
  const char *cellid;
  DMSwarmCellDM celldm;

  PetscFunctionBegin;
  {
    PetscInt dof0,dof1,dof2;
    PetscCall(DMStagGetDOF(dmcell,&dof0,&dof1,&dof2,NULL)); /* (vertex) (face) (element) */
    if (dof2 == 0) SETERRQ(PetscObjectComm((PetscObject)dmcell),PETSC_ERR_SUP,"Require dmcell has element wise defined fields");
    if (dof0 != 0) SETERRQ(PetscObjectComm((PetscObject)dmcell),PETSC_ERR_SUP,"Only valid for dmcell with element wise defined fields. Detected vertex fields");
    if (dof1 != 0) SETERRQ(PetscObjectComm((PetscObject)dmcell),PETSC_ERR_SUP,"Only valid for dmcell with element wise defined fields. Detected face fields");
  }
  
  PetscCall(DMSwarmGetLocalSize(dmswarm,&npoints));
  PetscCall(DMSwarmGetField(dmswarm,propname,&bs,&type,(void**)&pfield));
  if (type != PETSC_REAL) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Type must be PETSC_REAL for field %s",propname);
  if (bs != 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Block size must be 1. Found %d for field %s",bs,propname);

  PetscCall(DMSwarmGetCellDMActive(dmswarm, &celldm));
  PetscCall(DMSwarmCellDMGetCellID(celldm, &cellid));
  PetscCall(DMSwarmGetField(dmswarm,cellid,NULL,NULL,(void**)&pcellid));

  PetscCall(DMStagGetLocationSlot(dmcell,DMSTAG_ELEMENT,element_dof,&slot));

  PetscCall(DMCreateLocalVector(dmcell,&cellcoeff_local));
  
  /* we scatter here to avoid zeroing out all fields with dof != element_dof */
  PetscCall(DMGlobalToLocalBegin(dmcell,cellcoeff,INSERT_VALUES,cellcoeff_local));
  PetscCall(DMGlobalToLocalEnd(dmcell,cellcoeff,INSERT_VALUES,cellcoeff_local));
  
  PetscCall(VecGetSize(cellcoeff_local,&ncell));
  PetscCall(VecGetBlockSize(cellcoeff_local,&bs_cell));
  ncell = ncell / bs_cell;
  PetscCall(PetscCalloc1(ncell,&cnt));

  PetscCall(VecGetArray(cellcoeff_local,&coeff));
  for (p=0; p<npoints; p++) {
    PetscInt cellid = -1;
    
    cellid = pcellid[p];
    
    coeff[ cellid * bs_cell + slot ] += pfield[p];
    cnt[ cellid ] += 1.0;
  }
  for (c=0; c<ncell; c++) {
    PetscBool in_global_space;
    PetscCall(DMStagLocalElementIndexInGlobalSpace_2d(dmcell,c,&in_global_space));
    if (in_global_space) PetscCall(PetscPrintf(PETSC_COMM_WORLD,"cell %d : cnt %+1.4e\n",c,cnt[c]));
  }
  
  for (c=0; c<ncell; c++) {
    PetscBool in_global_space;
    PetscCall(DMStagLocalElementIndexInGlobalSpace_2d(dmcell,c,&in_global_space));
    if (in_global_space) {
      if (cnt[c] < 1.0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cell %D is empty. Cannot perform P0 projection",c);
      coeff[ c * bs_cell + slot ] /= cnt[c];
    }
  }
  PetscCall(VecRestoreArray(cellcoeff_local,&coeff));
  
  PetscCall(DMSwarmRestoreField(dmswarm,propname,NULL,NULL,(void**)&pfield));
  PetscCall(DMSwarmRestoreField(dmswarm,cellid,NULL,NULL,(void**)&pcellid));
  
  PetscCall(DMLocalToGlobalBegin(dmcell,cellcoeff_local,INSERT_VALUES,cellcoeff));
  PetscCall(DMLocalToGlobalEnd(dmcell,cellcoeff_local,INSERT_VALUES,cellcoeff));

  PetscCall(VecDestroy(&cellcoeff_local));
  PetscCall(PetscFree(cnt));
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// /*
//  Performs piece wise constant interpolation
 
//  Input:
//  + dmswarm - The material point object
//  + propname - Textual name of material point property to project
//  + dmstag - FD grid object
//  + dmcell - FDstag object representing the coefficients
//  + stratrum_index - Index where projected element values should be stored
 
//  Output:
//  + cellcoeff - Vector where projected properties are stored
 
//  Notes:
//  * Only properties declared as PetscReal can be projected
//  * Only properties with block size = 1 can be projected
 
//  Collective
// */
// PetscErrorCode v0_MPoint_ProjectQ1_arith_general(DM dmswarm,const char propname[],
//                                       DM dmstag,
//                                       DM dmcell,
//                                       PetscInt stratrum_index, /* 0:(vertex) 1:(face) 2:(element) */
//                                       PetscInt dof,Vec cellcoeff)
// {
//   PetscInt p,npoints,bs,c,ncell,bs_cell;
//   PetscDataType type;
//   PetscReal *pfield;
//   PetscInt *pcellid,nslot,ns,slot[4];
//   Vec cellcoeff_local,cnt_global,cnt_local;
//   PetscReal ***coeff,***cnt;
//   const char *cellid;
//   DMSwarmCellDM      celldm;
  
//   PetscFunctionBegin;
//   PetscCall(DMCreateLocalVector(dmcell,&cellcoeff_local));
//   PetscCall(VecGetSize(cellcoeff_local,&ncell));
  
//   /* we scatter here to avoid zeroing out all fields with dof != element_dof */
//   PetscCall(DMGlobalToLocalBegin(dmcell,cellcoeff,INSERT_VALUES,cellcoeff_local));
//   PetscCall(DMGlobalToLocalEnd(dmcell,cellcoeff,INSERT_VALUES,cellcoeff_local));

//   slot[0] = slot[1] = slot[2] = slot[3] = -1;
//   switch (stratrum_index) {
//     case 0: // vertex
//       nslot = 4;
//       PetscCall(DMStagGetLocationSlot(dmcell,DMSTAG_UP_LEFT   ,dof,&slot[0]));
//       PetscCall(DMStagGetLocationSlot(dmcell,DMSTAG_UP_RIGHT  ,dof,&slot[1]));
//       PetscCall(DMStagGetLocationSlot(dmcell,DMSTAG_DOWN_LEFT ,dof,&slot[2]));
//       PetscCall(DMStagGetLocationSlot(dmcell,DMSTAG_DOWN_RIGHT,dof,&slot[3]));
//       break;
//     case 1: // face
//       nslot = 4;
//       PetscCall(DMStagGetLocationSlot(dmcell,DMSTAG_UP   ,dof,&slot[0]));
//       PetscCall(DMStagGetLocationSlot(dmcell,DMSTAG_DOWN ,dof,&slot[1]));
//       PetscCall(DMStagGetLocationSlot(dmcell,DMSTAG_LEFT ,dof,&slot[2]));
//       PetscCall(DMStagGetLocationSlot(dmcell,DMSTAG_RIGHT,dof,&slot[3]));
//       break;
//     case 2: // cell
//       nslot = 1;
//       PetscCall(DMStagGetLocationSlot(dmcell,DMSTAG_ELEMENT,dof,&slot[0]));
//       break;
      
//     default:
//       break;
//   }
  
//   PetscCall(DMSwarmGetLocalSize(dmswarm,&npoints));
//   PetscCall(DMSwarmGetField(dmswarm,propname,&bs,&type,(void**)&pfield));
//   if (type != PETSC_REAL) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Type must be PETSC_REAL for field %s",propname);
//   if (bs != 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Block size must be 1. Found %d for field %s",bs,propname);

//   PetscCall(DMSwarmGetCellDMActive(dmswarm, &celldm));
//   PetscCall(DMSwarmCellDMGetCellID(celldm, &cellid));
//   PetscCall(DMSwarmGetField(dmswarm,cellid,NULL,NULL,(void**)&pcellid);

//   PetscCall(DMCreateGlobalVector(dmcell,&cnt_global));
//   PetscCall(DMCreateLocalVector(dmcell,&cnt_local));
  
//   PetscCall(DMStagVecGetArray(dmcell,cellcoeff_local,&coeff));
//   PetscCall(DMStagVecGetArray(dmcell,cnt_local,&cnt));

//   for (p=0; p<npoints; p++) {
//     PetscInt cellid = -1;
//     PetscInt geid[]={0,0,0};
    
//     cellid = pcellid[p];

//     PetscCall(DMStagGetLocalElementGlobalIndices(dmcell,cellid,geid));
//     for (ns=0; ns<nslot; ns++) {
//       coeff[ geid[1] ][ geid[0] ][ slot[ns] ] += pfield[p];
//       cnt  [ geid[1] ][ geid[0] ][ slot[ns] ] += 1.0;
//     }
//   }
  
//   PetscCall(DMStagVecRestoreArray(dmcell,cnt_local,&cnt));
//   PetscCall(DMStagVecRestoreArray(dmcell,cellcoeff_local,&coeff));

//   PetscCall(DMSwarmRestoreField(dmswarm,propname,NULL,NULL,(void**)&pfield));
//   PetscCall(DMSwarmRestoreField(dmswarm,cellid,NULL,NULL,(void**)&pcellid));
  
//   PetscCall(DMLocalToGlobalBegin(dmcell,cellcoeff_local,ADD_VALUES,cellcoeff));
//   PetscCall(DMLocalToGlobalEnd(dmcell,cellcoeff_local,ADD_VALUES,cellcoeff));

//   PetscCall(DMLocalToGlobalBegin(dmcell,cnt_local,ADD_VALUES,cnt_global));
//   PetscCall(DMLocalToGlobalEnd(dmcell,cnt_local,ADD_VALUES,cnt_global));

//   {
//     PetscInt i,len;
//     PetscReal *LA_coeff,*LA_cnt;
    
//     PetscCall(VecGetLocalSize(cellcoeff,&len));
//     PetscCall(VecGetArray(cellcoeff,&LA_coeff));
//     PetscCall(VecGetArray(cnt_global,&LA_cnt));
//     for (i=0; i<len; i++) {
//       if (LA_cnt[i] > 0.0) {
//         LA_coeff[i] = LA_coeff[i] / LA_cnt[i];
//       }
//     }
//     PetscCall(VecRestoreArray(cnt_global,&LA_cnt));
//     PetscCall(VecRestoreArray(cellcoeff,&LA_coeff));
//   }
  
//   PetscCall(VecDestroy(&cnt_global));
//   PetscCall(VecDestroy(&cnt_local));
//   PetscCall(VecDestroy(&cellcoeff_local));

//   PetscFunctionReturn(PETSC_SUCCESS);
// }

// ---------------------------------------
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
  const char *cellid;
  DMSwarmCellDM celldm;
  
  PetscFunctionBegin;
  slot[0] = slot[1] = slot[2] = slot[3] = -1;
  switch (stratrum_index) {
    case 0: // vertex
      dof[0] = 1; dof[1] = 0; dof[2] = 0; dof[3] = 0; /* (vertex) (face) (element) */
      PetscCall(DMStagCreateCompatibleDMStag(dmcell,dof[0],dof[1],dof[2],dof[3],&compat));
      nslot = 4;
      PetscCall(DMStagGetLocationSlot(compat,DMSTAG_UP_LEFT   ,0,&slot[0]));
      PetscCall(DMStagGetLocationSlot(compat,DMSTAG_UP_RIGHT  ,0,&slot[1]));
      PetscCall(DMStagGetLocationSlot(compat,DMSTAG_DOWN_LEFT ,0,&slot[2]));
      PetscCall(DMStagGetLocationSlot(compat,DMSTAG_DOWN_RIGHT,0,&slot[3]));
      break;
    case 1: // face
      dof[0] = 0; dof[1] = 1; dof[2] = 0; dof[3] = 0; /* (vertex) (face) (element) */
      PetscCall(DMStagCreateCompatibleDMStag(dmcell,dof[0],dof[1],dof[2],dof[3],&compat));
      nslot = 4;
      PetscCall(DMStagGetLocationSlot(compat,DMSTAG_UP   ,0,&slot[0]));
      PetscCall(DMStagGetLocationSlot(compat,DMSTAG_DOWN ,0,&slot[1]));
      PetscCall(DMStagGetLocationSlot(compat,DMSTAG_LEFT ,0,&slot[2]));
      PetscCall(DMStagGetLocationSlot(compat,DMSTAG_RIGHT,0,&slot[3]));
      break;
    case 2: // cell
      dof[0] = 0; dof[1] = 0; dof[2] = 1; dof[3] = 0; /* (vertex) (face) (element) */
      PetscCall(DMStagCreateCompatibleDMStag(dmcell,dof[0],dof[1],dof[2],dof[3],&compat));
      nslot = 1;
      PetscCall(DMStagGetLocationSlot(compat,DMSTAG_ELEMENT,0,&slot[0]));
      break;
      
    default:
      break;
  }
  
  PetscCall(DMCreateGlobalVector(compat,&sum_global));
  PetscCall(DMCreateLocalVector(compat,&sum_local));
  
  PetscCall(DMSwarmGetLocalSize(dmswarm,&npoints));
  PetscCall(DMSwarmGetField(dmswarm,propname,&bs,&type,(void**)&pfield));
  if (type != PETSC_REAL) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Type must be PETSC_REAL for field %s",propname);
  if (bs != 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Block size must be 1. Found %d for field %s",bs,propname);

  PetscCall(DMSwarmGetCellDMActive(dmswarm, &celldm));
  PetscCall(DMSwarmCellDMGetCellID(celldm, &cellid));
  PetscCall(DMSwarmGetField(dmswarm,cellid,NULL,NULL,(void**)&pcellid));
  
  PetscCall(DMCreateGlobalVector(compat,&cnt_global));
  PetscCall(DMCreateLocalVector(compat,&cnt_local));
  
  PetscCall(DMStagVecGetArray(compat,sum_local,&coeff));
  PetscCall(DMStagVecGetArray(compat,cnt_local,&cnt));
  
  for (p=0; p<npoints; p++) {
    PetscInt cellid = -1;
    PetscInt geid[]={0,0,0};
    
    cellid = pcellid[p];
    
    PetscCall(DMStagGetLocalElementGlobalIndices(compat,cellid,geid));
    for (ns=0; ns<nslot; ns++) {
      coeff[ geid[1] ][ geid[0] ][ slot[ns] ] += pfield[p];
      cnt  [ geid[1] ][ geid[0] ][ slot[ns] ] += 1.0;
    }
  }
  
  PetscCall(DMStagVecRestoreArray(compat,cnt_local,&cnt));
  PetscCall(DMStagVecRestoreArray(compat,sum_local,&coeff));
  
  PetscCall(DMSwarmRestoreField(dmswarm,propname,NULL,NULL,(void**)&pfield));
  PetscCall(DMSwarmRestoreField(dmswarm,cellid,NULL,NULL,(void**)&pcellid));
  
  PetscCall(DMLocalToGlobalBegin(compat,sum_local,ADD_VALUES,sum_global));
  PetscCall(DMLocalToGlobalEnd(compat,sum_local,ADD_VALUES,sum_global));
  
  PetscCall(DMLocalToGlobalBegin(compat,cnt_local,ADD_VALUES,cnt_global));
  PetscCall(DMLocalToGlobalEnd(compat,cnt_local,ADD_VALUES,cnt_global));
  
  {
    PetscInt i,len;
    PetscReal *LA_coeff,*LA_cnt;
    
    PetscCall(VecGetLocalSize(cnt_global,&len));
    PetscCall(VecGetArray(sum_global,&LA_coeff));
    PetscCall(VecGetArray(cnt_global,&LA_cnt));
    for (i=0; i<len; i++) {
      if (LA_cnt[i] > 0.0) {
        LA_coeff[i] = LA_coeff[i] / LA_cnt[i];
      }
    }
    PetscCall(VecRestoreArray(cnt_global,&LA_cnt));
    PetscCall(VecRestoreArray(sum_global,&LA_coeff));
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
    
    PetscCall(DMStagISCreateL2L_2d(compat,n0A,dof0A,n1A,dof1A,n2A,dof2A,&is_compat,
                                dmcell,dof0B,dof1B,dof2B,&is_cell));
    
    PetscCall(ISGetLocalSize(is_compat,&len));
    PetscCall(ISGetIndices(is_compat,&_compat));
    PetscCall(ISGetIndices(is_cell,&_cell));

    PetscCall(VecGetArray(sum_global,&LA_coeff));
    PetscCall(VecGetArray(cellcoeff,&LA_cellcoeff));
    for (i=0; i<len; i++) {
      LA_cellcoeff[ _cell[i] ] = LA_coeff[ _compat[i] ];
    }
    PetscCall(VecRestoreArray(cellcoeff,&LA_cellcoeff));
    PetscCall(VecRestoreArray(sum_global,&LA_coeff));

    PetscCall(ISRestoreIndices(is_cell,&_cell));
    PetscCall(ISRestoreIndices(is_compat,&_compat));

    PetscCall(ISDestroy(&is_compat));
    PetscCall(ISDestroy(&is_cell));
  }
  
  PetscCall(VecDestroy(&cnt_global));
  PetscCall(VecDestroy(&cnt_local));
  PetscCall(VecDestroy(&sum_global));
  PetscCall(VecDestroy(&sum_local));
  PetscCall(DMDestroy(&compat));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
PetscErrorCode MPoint_ProjectQ1_arith_general_AP(DM dmswarm,const char propname[],
                                              DM dmstag,
                                              DM dmcell,
                                              PetscInt stratrum_index, /* 0:(vertex) 1:(face) 2:(element) */
                                              PetscInt stagdof,Vec cellcoeff)
{
  DM compat;
  PetscDataType type;
  PetscReal *pfield;
  PetscInt  i,j,p,ns,sx,sz,nx,nz, slot[4], cslot[4], dof[4], nslot, bs, c, npoints, *pcellid;
  Vec       sum_global, sum_local, cnt_global, cnt_local, cellcoeff_local;
  PetscReal ***coeff_s,***coeff,***cnt;
  const char *cellid;
  DMSwarmCellDM celldm;
  PetscFunctionBegin;

  PetscCall(DMStagGetCorners(dmcell, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 

  // set indices and create dummy dmstag with one layer dof
  slot[0] = slot[1] = slot[2] = slot[3] = -1;
  switch (stratrum_index) {
    case 0: // vertex
      dof[0] = 1; dof[1] = 0; dof[2] = 0; dof[3] = 0; /* (vertex) (face) (element) */
      PetscCall(DMStagCreateCompatibleDMStag(dmcell,dof[0],dof[1],dof[2],dof[3],&compat));
      nslot = 4;
      PetscCall(DMStagGetLocationSlot(compat,DMSTAG_UP_LEFT   ,0,&slot[0]));
      PetscCall(DMStagGetLocationSlot(compat,DMSTAG_UP_RIGHT  ,0,&slot[1]));
      PetscCall(DMStagGetLocationSlot(compat,DMSTAG_DOWN_LEFT ,0,&slot[2]));
      PetscCall(DMStagGetLocationSlot(compat,DMSTAG_DOWN_RIGHT,0,&slot[3]));

      PetscCall(DMStagGetLocationSlot(dmcell,DMSTAG_UP_LEFT   ,stagdof,&cslot[0]));
      PetscCall(DMStagGetLocationSlot(dmcell,DMSTAG_UP_RIGHT  ,stagdof,&cslot[1]));
      PetscCall(DMStagGetLocationSlot(dmcell,DMSTAG_DOWN_LEFT ,stagdof,&cslot[2]));
      PetscCall(DMStagGetLocationSlot(dmcell,DMSTAG_DOWN_RIGHT,stagdof,&cslot[3]));
      break;
    case 1: // face
      dof[0] = 0; dof[1] = 1; dof[2] = 0; dof[3] = 0; /* (vertex) (face) (element) */
      PetscCall(DMStagCreateCompatibleDMStag(dmcell,dof[0],dof[1],dof[2],dof[3],&compat));
      nslot = 4;
      PetscCall(DMStagGetLocationSlot(compat,DMSTAG_UP   ,0,&slot[0]));
      PetscCall(DMStagGetLocationSlot(compat,DMSTAG_DOWN ,0,&slot[1]));
      PetscCall(DMStagGetLocationSlot(compat,DMSTAG_LEFT ,0,&slot[2]));
      PetscCall(DMStagGetLocationSlot(compat,DMSTAG_RIGHT,0,&slot[3]));

      PetscCall(DMStagGetLocationSlot(dmcell,DMSTAG_UP   ,stagdof,&cslot[0]));
      PetscCall(DMStagGetLocationSlot(dmcell,DMSTAG_DOWN ,stagdof,&cslot[1]));
      PetscCall(DMStagGetLocationSlot(dmcell,DMSTAG_LEFT ,stagdof,&cslot[2]));
      PetscCall(DMStagGetLocationSlot(dmcell,DMSTAG_RIGHT,stagdof,&cslot[3]));
      break;
    case 2: // cell
      dof[0] = 0; dof[1] = 0; dof[2] = 1; dof[3] = 0; /* (vertex) (face) (element) */
      PetscCall(DMStagCreateCompatibleDMStag(dmcell,dof[0],dof[1],dof[2],dof[3],&compat));
      nslot = 1;
      PetscCall(DMStagGetLocationSlot(compat,DMSTAG_ELEMENT,0,&slot[0]));
      PetscCall(DMStagGetLocationSlot(dmcell,DMSTAG_ELEMENT,stagdof,&cslot[0]));
      break;
      
    default:
      break;
  }

  // check dmswarm
  PetscCall(DMSwarmGetLocalSize(dmswarm,&npoints));
  PetscCall(DMSwarmGetField(dmswarm,propname,&bs,&type,(void**)&pfield));
  if (type != PETSC_REAL) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Type must be PETSC_REAL for field %s",propname);
  if (bs != 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Block size must be 1. Found %d for field %s",bs,propname);

  PetscCall(DMSwarmGetCellDMActive(dmswarm, &celldm));
  PetscCall(DMSwarmCellDMGetCellID(celldm, &cellid));
  PetscCall(DMSwarmGetField(dmswarm,cellid,NULL,NULL,(void**)&pcellid));
  
  // create vectors
  PetscCall(DMCreateGlobalVector(compat,&sum_global));
  PetscCall(DMCreateLocalVector(compat,&sum_local));

  PetscCall(DMCreateGlobalVector(compat,&cnt_global));
  PetscCall(DMCreateLocalVector(compat,&cnt_local));
  
  PetscCall(DMStagVecGetArray(compat,sum_local,&coeff));
  PetscCall(DMStagVecGetArray(compat,cnt_local,&cnt));
  
  for (p=0; p<npoints; p++) {
    PetscInt cellid = -1;
    PetscInt geid[]={0,0,0};
    
    cellid = pcellid[p];
    
    PetscCall(DMStagGetLocalElementGlobalIndices(compat,cellid,geid));
    for (ns=0; ns<nslot; ns++) {
      coeff[ geid[1] ][ geid[0] ][ slot[ns] ] += pfield[p];
      cnt  [ geid[1] ][ geid[0] ][ slot[ns] ] += 1.0;
    }
  }

  PetscCall(DMStagVecRestoreArray(compat,cnt_local,&cnt));
  PetscCall(DMStagVecRestoreArray(compat,sum_local,&coeff));
  
  PetscCall(DMSwarmRestoreField(dmswarm,propname,NULL,NULL,(void**)&pfield));
  PetscCall(DMSwarmRestoreField(dmswarm,cellid,NULL,NULL,(void**)&pcellid));
  
  PetscCall(DMLocalToGlobal(compat,sum_local,ADD_VALUES,sum_global));
  PetscCall(DMLocalToGlobal(compat,cnt_local,ADD_VALUES,cnt_global));

  // save in dmcell
  PetscCall(DMCreateLocalVector(dmcell,&cellcoeff_local));
  
  PetscCall(DMGlobalToLocal(compat,sum_global,INSERT_VALUES,sum_local));
  PetscCall(DMGlobalToLocal(compat,cnt_global,INSERT_VALUES,cnt_local));
  PetscCall(DMGlobalToLocal(dmcell,cellcoeff,INSERT_VALUES,cellcoeff_local));

  PetscCall(DMStagVecGetArray(compat,sum_local,&coeff));
  PetscCall(DMStagVecGetArray(compat,cnt_local,&cnt));
  PetscCall(DMStagVecGetArray(dmcell,cellcoeff_local,&coeff_s));
  
  // loop
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      for (ns=0; ns<nslot; ns++) {
        if (cnt[j][i][slot[ns]] > 0.0) {
          coeff_s[j][i][cslot[ns]] = coeff[j][i][slot[ns]]/cnt[j][i][slot[ns]];
        }
      }
    }
  }

  PetscCall(DMStagVecRestoreArray(compat,cnt_local,&cnt));
  PetscCall(DMStagVecRestoreArray(compat,sum_local,&coeff));
  PetscCall(DMStagVecRestoreArray(dmcell,cellcoeff_local,&coeff_s));

  PetscCall(DMLocalToGlobal(dmcell,cellcoeff_local,INSERT_VALUES,cellcoeff));

  // clean-up
  PetscCall(VecDestroy(&cnt_global));
  PetscCall(VecDestroy(&cnt_local));
  PetscCall(VecDestroy(&sum_global));
  PetscCall(VecDestroy(&sum_local));
  PetscCall(VecDestroy(&cellcoeff_local));
  PetscCall(DMDestroy(&compat));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// /*
//  Assign particles coordinates within all cells in the domain. Particle coordinates are assigned cell-wise
 
//  Input:
//  + dmswarm - The particle object
//  + factor - Controls the magnitude of the random perturbation applied to each particle position. Must be in range [0,1]
//  + points_per_dim - Number of points in x/y to define within each cell
//  + mode - Indicates whether new coordinates should be appending to the existing points in dmswarm, or whether
//           the existing points (and size) of dmswarm should be set to zero prior to defining new coordinates
 
//  Output:
//  + dmswarm - On exit dmswarm will contain newly defined coordinates
 
//  Notes:
//  * Assumes constant cell spacing

//  Collective
// */
// PetscErrorCode MPointCoordLayout_DomainVolume(DM dmswarm,PetscReal factor,PetscInt points_per_dim,MPointCoordinateInsertMode mode)
// {
//   DM             dmstag;
//   PetscInt       p,npoints,Ng[]={0,0};
//   PetscReal      *LA_coor;
//   PetscReal      dx[2],dxp[2],gmin[]={0,0,0},gmax[]={0,0,0};
//   PetscRandom    r=NULL;
//   PetscMPIInt    rank;
  
//   PetscFunctionBegin;
//   if (mode == COOR_APPEND) SETERRQ(PetscObjectComm((PetscObject)dmswarm),PETSC_ERR_SUP,"Only mode COOR_INITIALIZE supported");
  
//   PetscCall(DMSwarmInsertPointsUsingCellDM(dmswarm,DMSWARMPIC_LAYOUT_REGULAR,points_per_dim));
  
//   PetscCall(DMSwarmGetCellDM(dmswarm,&dmstag));
//   PetscCall(DMStagGetGlobalSizes(dmstag,&Ng[0],&Ng[1],NULL));
//   PetscCall(DMStagGetBoundingBox(dmstag,gmin,gmax));
//   dx[0] = (gmax[0]-gmin[0])/Ng[0];
//   dx[1] = (gmax[1]-gmin[1])/Ng[1];
//   dxp[0] = dx[0] / (PetscReal)points_per_dim;
//   dxp[1] = dx[1] / (PetscReal)points_per_dim;
  
//   if (factor > 0.0) {
//     PetscCall(PetscRandomCreate(PETSC_COMM_SELF,&r));
//     PetscCall(PetscRandomSetInterval(r,-factor,factor));
//     PetscCall(MPI_Comm_rank(PetscObjectComm((PetscObject)dmswarm),&rank));
//     PetscCall(PetscRandomSetSeed(r,(unsigned long)rank));
//     PetscCall(PetscRandomSeed(r));
//   }
  
//   PetscCall(DMSwarmGetField(dmswarm,DMSwarmPICField_coor,NULL,NULL,(void**)&LA_coor));
//   PetscCall(DMSwarmGetLocalSize(dmswarm,&npoints));
//   if (factor > 0.0) {

//     for (p=0; p<npoints; p++) {
//       PetscReal rr[2];
      
//       PetscCall(PetscRandomGetValueReal(r,&rr[0]));
//       PetscCall(PetscRandomGetValueReal(r,&rr[1]));
//       LA_coor[2*p+0] += rr[0] * 0.5 * dxp[0];
//       LA_coor[2*p+1] += rr[1] * 0.5 * dxp[1];
//     }
//   }
//   PetscCall(DMSwarmRestoreField(dmswarm,DMSwarmPICField_coor,NULL,NULL,(void**)&LA_coor));
  
//   /* Migrate - since since perturbing particles have have caused point to be located in a different cell, or located on another sub-domain */
//   PetscCall(DMSwarmMigrate(dmswarm,PETSC_TRUE));

//   PetscCall(PetscRandomDestroy(&r));
  
//   PetscFunctionReturn(PETSC_SUCCESS);
// }

// ---------------------------------------
static PetscErrorCode _CellFillCoor(PetscReal gmin[],PetscReal gmax[],PetscInt np[],PetscReal coor[])
{
  PetscReal dx,dy;
  PetscInt i,j;
  PetscReal x0=gmin[0],x1=gmax[0],y0=gmin[1],y1=gmax[1];
  PetscInt npx=np[0],npy=np[1];
  PetscFunctionBegin;
  
  dx = (x1 - x0) / (PetscReal)npx;
  dy = (y1 - y0) / (PetscReal)npy;
  for (j=0; j<npy; j++) {
    for (i=0; i<npx; i++) {
      PetscInt idx = i + j * npx;
      coor[2*idx+0] = x0 + 0.5 * dx + i * dx;
      coor[2*idx+1] = y0 + 0.5 * dy + j * dy;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
static PetscErrorCode _CellFaceFillCoor_n_s(PetscReal x0,PetscReal x1,PetscReal y,PetscInt npx,PetscReal coor[])
{
  PetscReal dx;
  PetscInt i;
  PetscFunctionBegin;
  
  dx = (x1 - x0) / (PetscReal)npx;
  for (i=0; i<npx; i++) {
    coor[2*i+0] = x0 + 0.5 * dx + i * dx;
    coor[2*i+1] = y;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
static PetscErrorCode _CellFaceFillCoor_e_w(PetscReal y0,PetscReal y1,PetscReal x,PetscInt npy,PetscReal coor[])
{
  PetscReal dy;
  PetscInt j;
  PetscFunctionBegin;
  
  dy = (y1 - y0) / (PetscReal)npy;
  for (j=0; j<npy; j++) {
    coor[2*j+0] = x;
    coor[2*j+1] = y0 + 0.5 * dy + j * dy;
  }
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
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
 * The input celllist[] may be NULL. In this case, all cells in the domain will be filled with points
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
  const char     *cellid;
  DMSwarmCellDM  celldm;
  
  PetscFunctionBegin;
  if (mode == COOR_INITIALIZE) {
    PetscCall(DMSwarmSetLocalSizes(dmswarm,0,-1));
  }
  
  PetscCall(DMSwarmGetCellDM(dmswarm,&dmstag));
  PetscCall(DMStagGetGlobalSizes(dmstag,&Ng[0],&Ng[1],NULL));
  PetscCall(DMStagGetLocalSizes(dmstag,&N[0],&N[1],NULL));
  PetscCall(DMStagGetCorners(dmstag,&es[0],&es[1],NULL,NULL,NULL,NULL,NULL,NULL,NULL));
  PetscCall(DMStagGetBoundingBox(dmstag,gmin,gmax));
  
  if (!celllist) {
    ncells = N[0] * N[1];
  } else {
    ncells = _ncells;
  }
  if (ncells == 0) PetscFunctionReturn(PETSC_SUCCESS);
  
  PetscCall(DMSwarmGetLocalSize(dmswarm,&npoints_init));
  nppcell = points_per_dim[0] * points_per_dim[1];
  nnew = ncells * nppcell;
  PetscCall(DMSwarmSetLocalSizes(dmswarm,npoints_init+nnew,-1));
  PetscCall(DMSwarmGetLocalSize(dmswarm,&npoints));
  
  PetscCall(PetscMalloc1(2*nppcell,&cellcoor));
  
  dx[0] = (gmax[0]-gmin[0])/Ng[0];
  dx[1] = (gmax[1]-gmin[1])/Ng[1];
  
  if (factor > 0.0) {
    PetscCall(PetscRandomCreate(PETSC_COMM_SELF,&r));
    PetscCall(PetscRandomSetInterval(r,-factor,factor));
    PetscCall(MPI_Comm_rank(PetscObjectComm((PetscObject)dmswarm),&rank));
    PetscCall(PetscRandomSetSeed(r,(unsigned long)rank));
    PetscCall(PetscRandomSeed(r));
  }

  PetscCall(DMSwarmGetField(dmswarm,DMSwarmPICField_coor,NULL,NULL,(void**)&pcoor));

  PetscCall(DMSwarmGetCellDMActive(dmswarm, &celldm));
  PetscCall(DMSwarmCellDMGetCellID(celldm, &cellid));
  PetscCall(DMSwarmGetField(dmswarm,cellid,NULL,NULL,(void**)&pcellid));

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
      PetscCall(_CellFillCoor(cgmin,cgmax,points_per_dim,cellcoor));
      
      if (factor > 0.0) {
        dxp[0] = (cgmax[0] - cgmin[0]) / (PetscReal)points_per_dim[0];
        dxp[1] = (cgmax[1] - cgmin[1]) / (PetscReal)points_per_dim[1];

        for (p=0; p<nppcell; p++) {
          PetscReal rr[2];
          
          PetscCall(PetscRandomGetValueReal(r,&rr[0]));
          PetscCall(PetscRandomGetValueReal(r,&rr[1]));
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
        PetscCall(DMStagGetLocalElementIndex(dmstag,egidx,&cellid_local));
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
      PetscCall(_CellFillCoor(cgmin,cgmax,points_per_dim,cellcoor));
      
      if (factor > 0.0) {
        dxp[0] = (cgmax[0] - cgmin[0]) / (PetscReal)points_per_dim[0];
        dxp[1] = (cgmax[1] - cgmin[1]) / (PetscReal)points_per_dim[1];
        
        for (p=0; p<nppcell; p++) {
          PetscReal rr[2];
          
          PetscCall(PetscRandomGetValueReal(r,&rr[0]));
          PetscCall(PetscRandomGetValueReal(r,&rr[1]));
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
        PetscCall(DMStagGetLocalElementIndex(dmstag,egidx,&cellid_local));
        pcellid[npoints_init + pcnt] = cellid_local;

        pcnt++;
      }
    }
  }
  PetscCall(DMSwarmRestoreField(dmswarm,DMSwarmPICField_coor,NULL,NULL,(void**)&pcoor));
  PetscCall(DMSwarmRestoreField(dmswarm,cellid,NULL,NULL,(void**)&pcellid));

  PetscCall(DMSwarmMigrate(dmswarm,PETSC_TRUE));

  PetscCall(PetscRandomDestroy(&r));
  PetscCall(PetscFree(cellcoor));
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
static PetscErrorCode _MPointCoordLayout_FillDomainFace_NS(DM dmswarm,
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
  const char     *cellid;
  DMSwarmCellDM  celldm;
  
  PetscFunctionBegin;
  if (mode == COOR_INITIALIZE) {
    PetscCall(DMSwarmSetLocalSizes(dmswarm,0,-1));
  }
  
  PetscCall(DMSwarmGetCellDM(dmswarm,&dmstag));
  PetscCall(DMStagGetGlobalSizes(dmstag,&Ng[0],&Ng[1],NULL));
  PetscCall(DMStagGetLocalSizes(dmstag,&N[0],&N[1],NULL));
  PetscCall(DMStagGetBoundingBox(dmstag,gmin,gmax));

  PetscCall(DMStagGetCorners(dmstag,&es[0],&es[1],NULL,&nele[0],&nele[1],NULL,NULL,NULL,NULL));
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
  
  PetscCall(DMSwarmGetLocalSize(dmswarm,&npoints_init));
  nppcell = points_per_dim;
  nnew = ncells * nppcell;
  PetscCall(DMSwarmSetLocalSizes(dmswarm,npoints_init+nnew,-1));
  PetscCall(DMSwarmGetLocalSize(dmswarm,&npoints));
  
  PetscCall(PetscMalloc1(2*nppcell,&cellcoor));
  
  dx[0] = (gmax[0]-gmin[0])/Ng[0];
  dx[1] = (gmax[1]-gmin[1])/Ng[1];
  
  if (factor > 0.0) {
    PetscCall(PetscRandomCreate(PETSC_COMM_SELF,&r));
    PetscCall(PetscRandomSetInterval(r,-factor,factor));
    PetscCall(MPI_Comm_rank(PetscObjectComm((PetscObject)dmswarm),&rank));
    PetscCall(PetscRandomSetSeed(r,(unsigned long)rank));
    PetscCall(PetscRandomSeed(r));
  }
  
  PetscCall(DMSwarmGetCellDMActive(dmswarm, &celldm));
  PetscCall(DMSwarmCellDMGetCellID(celldm, &cellid));
  PetscCall(DMSwarmGetField(dmswarm,cellid,NULL,NULL,(void**)&pcellid));
  PetscCall(DMSwarmGetField(dmswarm,DMSwarmPICField_coor,NULL,NULL,(void**)&pcoor));
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
    PetscCall(_CellFaceFillCoor_n_s(cgmin[0],cgmax[0],yref,points_per_dim,cellcoor));
    
    if (factor > 0.0) {
      dxp = (cgmax[0] - cgmin[0]) / (PetscReal)points_per_dim;
      
      for (p=0; p<nppcell; p++) {
        PetscReal rr;
        
        PetscCall(PetscRandomGetValueReal(r,&rr));
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
      PetscCall(DMStagGetLocalElementIndex(dmstag,egidx,&cellid_local));
      pcellid[npoints_init + pcnt] = cellid_local;

      pcnt++;
    }
  }
  PetscCall(DMSwarmRestoreField(dmswarm,DMSwarmPICField_coor,NULL,NULL,(void**)&pcoor));
  PetscCall(DMSwarmRestoreField(dmswarm,cellid,NULL,NULL,(void**)&pcellid));
  
  PetscCall(PetscRandomDestroy(&r));
  PetscCall(PetscFree(cellcoor));
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
static PetscErrorCode _MPointCoordLayout_FillDomainFace_EW(DM dmswarm,
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
  const char     *cellid;
  DMSwarmCellDM  celldm;
  
  PetscFunctionBegin;
  if (mode == COOR_INITIALIZE) {
    PetscCall(DMSwarmSetLocalSizes(dmswarm,0,-1));
  }
  
  PetscCall(DMSwarmGetCellDM(dmswarm,&dmstag));
  PetscCall(DMStagGetGlobalSizes(dmstag,&Ng[0],&Ng[1],NULL));
  PetscCall(DMStagGetLocalSizes(dmstag,&N[0],&N[1],NULL));
  PetscCall(DMStagGetBoundingBox(dmstag,gmin,gmax));
  
  PetscCall(DMStagGetCorners(dmstag,&es[0],&es[1],NULL,&nele[0],&nele[1],NULL,NULL,NULL,NULL));
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
  
  PetscCall(DMSwarmGetLocalSize(dmswarm,&npoints_init));
  nppcell = points_per_dim;
  nnew = ncells * nppcell;
  PetscCall(DMSwarmSetLocalSizes(dmswarm,npoints_init+nnew,-1));
  PetscCall(DMSwarmGetLocalSize(dmswarm,&npoints));
  
  PetscCall(PetscMalloc1(2*nppcell,&cellcoor));
  
  dx[0] = (gmax[0]-gmin[0])/Ng[0];
  dx[1] = (gmax[1]-gmin[1])/Ng[1];
  
  if (factor > 0.0) {
    PetscCall(PetscRandomCreate(PETSC_COMM_SELF,&r));
    PetscCall(PetscRandomSetInterval(r,-factor,factor));
    PetscCall(MPI_Comm_rank(PetscObjectComm((PetscObject)dmswarm),&rank));
    PetscCall(PetscRandomSetSeed(r,(unsigned long)rank));
    PetscCall(PetscRandomSeed(r));
  }
  
  PetscCall(DMSwarmGetCellDMActive(dmswarm, &celldm));
  PetscCall(DMSwarmCellDMGetCellID(celldm, &cellid));
  PetscCall(DMSwarmGetField(dmswarm,cellid,NULL,NULL,(void**)&pcellid));
  PetscCall(DMSwarmGetField(dmswarm,DMSwarmPICField_coor,NULL,NULL,(void**)&pcoor));

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
    PetscCall(_CellFaceFillCoor_e_w(cgmin[1],cgmax[1],xref,points_per_dim,cellcoor));
    
    if (factor > 0.0) {
      dyp = (cgmax[1] - cgmin[1]) / (PetscReal)points_per_dim;
      
      for (p=0; p<nppcell; p++) {
        PetscReal rr;
        
        PetscCall(PetscRandomGetValueReal(r,&rr));
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
      PetscCall(DMStagGetLocalElementIndex(dmstag,egidx,&cellid_local));
      pcellid[npoints_init + pcnt] = cellid_local;

      pcnt++;
    }
  }
  PetscCall(DMSwarmRestoreField(dmswarm,DMSwarmPICField_coor,NULL,NULL,(void**)&pcoor));
  PetscCall(DMSwarmRestoreField(dmswarm,cellid,NULL,NULL,(void**)&pcellid));
  
  PetscCall(PetscRandomDestroy(&r));
  PetscCall(PetscFree(cellcoor));
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
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
  
  PetscFunctionBegin;
  PetscCall(DMSwarmGetCellDM(dmswarm,&dmstag));
  PetscCall(DMStagGetGlobalSizes(dmstag,&Ng[0],&Ng[1],NULL));
  switch (face) {
    case 'n':
      PetscCall(_MPointCoordLayout_FillDomainFace_NS(dmswarm,Ng[1],factor,points_per_dim,mode));
      PetscCall(DMSwarmMigrate(dmswarm,PETSC_TRUE));
      break;
    case 's':
      PetscCall(_MPointCoordLayout_FillDomainFace_NS(dmswarm,0,factor,points_per_dim,mode));
      PetscCall(DMSwarmMigrate(dmswarm,PETSC_TRUE));
      break;
    case 'e':
      PetscCall(_MPointCoordLayout_FillDomainFace_EW(dmswarm,Ng[0],factor,points_per_dim,mode));
      PetscCall(DMSwarmMigrate(dmswarm,PETSC_TRUE));
      break;
    case 'w':
      PetscCall(_MPointCoordLayout_FillDomainFace_EW(dmswarm,0,factor,points_per_dim,mode));
      PetscCall(DMSwarmMigrate(dmswarm,PETSC_TRUE));
      break;
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Provided face value of %c is not supported. Must use one of 'n' (north), 's' (south), 'e' (east), 'w' (west)",face);
      break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// /*
//  Advect paticles using RK1 (forward Euler)
 
//  Notes:
//  * Performs P0 interpolation over each finite cell for both vx or vy.
 
//  Not collective
// */
// PetscErrorCode MPoint_AdvectRK1_Private_P0(DM dmswarm,
//                                         PetscReal mpfield_coor_k[],
//                                         PetscReal mpfield_coor_star[],
//                                         PetscReal mpfield_coor_kp1[],DM dmstag,Vec Xlocal,PetscReal dt)
// {
//   Vec               vp_l;
//   const PetscScalar ***LA_vp;
//   PetscInt          p,e,npoints;
//   PetscInt          *mpfield_cell;
//   PetscScalar       **cArrX,**cArrY;
//   PetscInt          iPrev,iNext,iCenter,iVxLeft,iVxRight,iVyDown,iVyUp,n[2];
//   DM                dm_vp,dm_mpoint;
//   const char        *cellid;
//   DMSwarmCellDM      celldm;
  
//   PetscFunctionBegin;
//   dm_vp = dmstag;
//   dm_mpoint = dmswarm;
//   vp_l = Xlocal;
  
//   PetscCall(DMStagGetProductCoordinateArraysRead(dm_vp,&cArrX,&cArrY,NULL));
//   PetscCall(DMStagGetProductCoordinateLocationSlot(dm_vp,DMSTAG_ELEMENT,&iCenter));
//   PetscCall(DMStagGetProductCoordinateLocationSlot(dm_vp,DMSTAG_LEFT,&iPrev));
//   PetscCall(DMStagGetProductCoordinateLocationSlot(dm_vp,DMSTAG_RIGHT,&iNext));
  
//   PetscCall(DMStagGetLocationSlot(dm_vp,DMSTAG_LEFT,0,&iVxLeft));
//   PetscCall(DMStagGetLocationSlot(dm_vp,DMSTAG_RIGHT,0,&iVxRight));
//   PetscCall(DMStagGetLocationSlot(dm_vp,DMSTAG_DOWN,0,&iVyDown));
//   PetscCall(DMStagGetLocationSlot(dm_vp,DMSTAG_UP,0,&iVyUp));
  
//   PetscCall(DMStagVecGetArrayRead(dm_vp,vp_l,&LA_vp));
  
//   PetscCall(DMStagGetLocalSizes(dm_vp,&n[0],&n[1],NULL));
//   PetscCall(DMSwarmGetLocalSize(dm_mpoint,&npoints));

//   PetscCall(DMSwarmGetCellDMActive(dmswarm, &celldm));
//   PetscCall(DMSwarmCellDMGetCellID(celldm, &cellid));
//   PetscCall(DMSwarmGetField(dm_mpoint,cellid,NULL,NULL,(void**)&mpfield_cell));

//   for (p=0; p<npoints; p++) {
//     PetscReal   coor_p_k[2],coor_p_star[2];
//     PetscScalar vel_p_star[2],vLeft,vRight,vUp,vDown;
//     PetscScalar x0[2],dx[2],xloc_p[2],xi_p[2];
//     PetscInt    ind[2];
    
//     e       = mpfield_cell[p];
//     coor_p_k[0] = mpfield_coor_k[2*p+0];
//     coor_p_k[1] = mpfield_coor_k[2*p+1];
//     coor_p_star[0] = mpfield_coor_star[2*p+0];
//     coor_p_star[1] = mpfield_coor_star[2*p+1];
//     PetscCall(DMStagGetLocalElementGlobalIndices(dm_vp,e,ind));
    
//     /* compute local coordinates: (xp-x0)/dx = (xip+1)/2 */
//     x0[0] = cArrX[ind[0]][iPrev];
//     x0[1] = cArrY[ind[1]][iPrev];
    
//     dx[0] = cArrX[ind[0]][iNext] - x0[0];
//     dx[1] = cArrY[ind[1]][iNext] - x0[1];
    
//     xloc_p[0] = (coor_p_star[0] - x0[0])/dx[0];
//     xloc_p[1] = (coor_p_star[1] - x0[1])/dx[1];
    
//     /* Checks (xi_p is only used for this, here) */
//     xi_p[0] = 2.0 * xloc_p[0] -1.0;
//     xi_p[1] = 2.0 * xloc_p[1] -1.0;
//     if (xi_p[0] < -1.0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"value (xi) too small %1.4e [e=%" PetscInt_FMT "]",(double)xi_p[0],e);
//     if (xi_p[0] >  1.0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"value (xi) too large %1.4e [e=%" PetscInt_FMT "]",(double)xi_p[0],e);
//     if (xi_p[1] < -1.0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"value (eta) too small %1.4e [e=%" PetscInt_FMT "]",(double)xi_p[1],e);
//     if (xi_p[1] >  1.0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"value (eta) too large %1.4e [e=%" PetscInt_FMT "]",(double)xi_p[1],e);
    
//     /* interpolate velocity */
//     vLeft  = LA_vp[ind[1]][ind[0]][iVxLeft];
//     vRight = LA_vp[ind[1]][ind[0]][iVxRight];
//     vUp    = LA_vp[ind[1]][ind[0]][iVyUp];
//     vDown  = LA_vp[ind[1]][ind[0]][iVyDown];
//     vel_p_star[0] = xloc_p[0]*vRight + (1.0-xloc_p[0])*vLeft;
//     vel_p_star[1] = xloc_p[1]*vUp    + (1.0-xloc_p[1])*vDown;
    
//     /* Update Coordinates */
//     mpfield_coor_kp1[2*p+0] = coor_p_k[0] + dt * vel_p_star[0];
//     mpfield_coor_kp1[2*p+1] = coor_p_k[1] + dt * vel_p_star[1];
//   }
  
//   PetscCall(DMSwarmRestoreField(dm_mpoint,cellid,NULL,NULL,(void**)&mpfield_cell));
//   PetscCall(DMStagRestoreProductCoordinateArraysRead(dm_vp,&cArrX,&cArrY,NULL));
//   PetscCall(DMStagVecRestoreArrayRead(dm_vp,vp_l,&LA_vp));
//   PetscFunctionReturn(PETSC_SUCCESS);
// }

// ---------------------------------------
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
  Vec               vp_l;
  const PetscScalar ***LA_vp;
  PetscInt          p,e,npoints;
  PetscInt          *mpfield_cell;
  PetscScalar       **cArrX,**cArrY;
  PetscInt          iPrev,iNext,iCenter,iVxLeft,iVxRight,iVyDown,iVyUp,n[2],N[2];
  DM                dm_vp,dm_mpoint;
  const char        *cellid;
  DMSwarmCellDM      celldm;
  
  PetscFunctionBegin;
  dm_vp = dmstag;
  dm_mpoint = dmswarm;
  vp_l = Xlocal;
  
  PetscCall(DMStagGetProductCoordinateArraysRead(dm_vp,&cArrX,&cArrY,NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm_vp,DMSTAG_ELEMENT,&iCenter));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm_vp,DMSTAG_LEFT,&iPrev));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm_vp,DMSTAG_RIGHT,&iNext));
  
  PetscCall(DMStagGetLocationSlot(dm_vp,DMSTAG_LEFT,0,&iVxLeft));
  PetscCall(DMStagGetLocationSlot(dm_vp,DMSTAG_RIGHT,0,&iVxRight));
  PetscCall(DMStagGetLocationSlot(dm_vp,DMSTAG_DOWN,0,&iVyDown));
  PetscCall(DMStagGetLocationSlot(dm_vp,DMSTAG_UP,0,&iVyUp));
  
  PetscCall(DMStagVecGetArrayRead(dm_vp,vp_l,&LA_vp));
  
  PetscCall(DMStagGetGlobalSizes(dm_vp,&N[0],&N[1],NULL));
  PetscCall(DMStagGetLocalSizes(dm_vp,&n[0],&n[1],NULL));
  PetscCall(DMSwarmGetLocalSize(dm_mpoint,&npoints));

  PetscCall(DMSwarmGetCellDMActive(dm_mpoint, &celldm));
  PetscCall(DMSwarmCellDMGetCellID(celldm, &cellid));
  PetscCall(DMSwarmGetField(dm_mpoint,cellid,NULL,NULL,(void**)&mpfield_cell));

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
    PetscCall(DMStagGetLocalElementGlobalIndices(dm_vp,e,ind));
    
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
  
  PetscCall(DMSwarmRestoreField(dm_mpoint,cellid,NULL,NULL,(void**)&mpfield_cell));
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm_vp,&cArrX,&cArrY,NULL));
  PetscCall(DMStagVecRestoreArrayRead(dm_vp,vp_l,&LA_vp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
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
  
  PetscFunctionBegin;
  PetscCall(DMGetLocalVector(dmstag,&Xlocal));
  PetscCall(DMGlobalToLocalBegin(dmstag,X,INSERT_VALUES,Xlocal));
  PetscCall(DMGlobalToLocalEnd(dmstag,X,INSERT_VALUES,Xlocal));

  PetscCall(DMSwarmGetField(dmswarm,DMSwarmPICField_coor,NULL,NULL,(void**)&mpfield_coor));
  PetscCall(MPoint_AdvectRK1_Private(dmswarm,mpfield_coor,mpfield_coor,mpfield_coor,dmstag,Xlocal,dt));
  PetscCall(DMSwarmRestoreField(dmswarm,DMSwarmPICField_coor,NULL,NULL,(void**)&mpfield_coor));
    
  PetscCall(DMRestoreLocalVector(dmstag,&Xlocal));
  
  /* scatter */
  PetscCall(DMSwarmMigrate(dmswarm,PETSC_TRUE));
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// #if 0
// /* needs updating - still uses P0 interpolation for velocity */
// /*
//  Not collective
// */
// PetscErrorCode MPoint_AdvectRK2_SEQ_Private(DM dmswarm,
//                                         PetscReal mpfield_coor_k[],
//                                         PetscReal mpfield_coor_kp1[],DM dmstag,Vec Xlocal,PetscReal dt)
// {
//   Vec               vp_l;
//   const PetscScalar ***LA_vp;
//   PetscInt          p,e,npoints;
//   PetscInt          *mpfield_cell;
//   PetscScalar       **cArrX,**cArrY;
//   PetscInt          iPrev,iNext,iCenter,iVxLeft,iVxRight,iVyDown,iVyUp,n[2];
//   DM                dm_vp,dm_mpoint;
//   const char     *cellid;
//   DMSwarmCellDM      celldm;
  
//   PetscFunctionBegin;
//   dm_vp = dmstag;
//   dm_mpoint = dmswarm;
//   vp_l = Xlocal;
  
//   PetscCall(DMStagGetProductCoordinateArraysRead(dm_vp,&cArrX,&cArrY,NULL));
//   PetscCall(DMStagGetProductCoordinateLocationSlot(dm_vp,DMSTAG_ELEMENT,&iCenter));
//   PetscCall(DMStagGetProductCoordinateLocationSlot(dm_vp,DMSTAG_LEFT,&iPrev));
//   PetscCall(DMStagGetProductCoordinateLocationSlot(dm_vp,DMSTAG_RIGHT,&iNext));
  
//   PetscCall(DMStagGetLocationSlot(dm_vp,DMSTAG_LEFT,0,&iVxLeft));
//   PetscCall(DMStagGetLocationSlot(dm_vp,DMSTAG_RIGHT,0,&iVxRight));
//   PetscCall(DMStagGetLocationSlot(dm_vp,DMSTAG_DOWN,0,&iVyDown));
//   PetscCall(DMStagGetLocationSlot(dm_vp,DMSTAG_UP,0,&iVyUp));
  
//   PetscCall(DMStagVecGetArrayRead(dm_vp,vp_l,&LA_vp));
  
//   PetscCall(DMStagGetLocalSizes(dm_vp,&n[0],&n[1],NULL));
//   PetscCall(DMSwarmGetLocalSize(dm_mpoint,&npoints));

//   PetscCall(DMSwarmGetCellDMActive(dmswarm, &celldm));
//   PetscCall(DMSwarmCellDMGetCellID(celldm, &cellid));
//   PetscCall(DMSwarmGetField(dm_mpoint,cellid,NULL,NULL,(void**)&mpfield_cell));
//   for (p=0; p<npoints; p++) {
//     PetscReal   coor_p_k[2],coor_p_star[2];
//     PetscScalar vel_p[2],vel_p_star[2],vLeft,vRight,vUp,vDown;
//     PetscScalar x0[2],dx[2],xloc_p[2],xi_p[2];
//     PetscInt    ind[2];
    
//     e       = mpfield_cell[p];
//     coor_p_k[0] = mpfield_coor_k[2*p+0];
//     coor_p_k[1] = mpfield_coor_k[2*p+1];
//     PetscCall(DMStagGetLocalElementGlobalIndices(dm_vp,e,ind));
    
//     /* compute local coordinates: (xp-x0)/dx = (xip+1)/2 */
//     x0[0] = cArrX[ind[0]][iPrev];
//     x0[1] = cArrY[ind[1]][iPrev];
    
//     dx[0] = cArrX[ind[0]][iNext] - x0[0];
//     dx[1] = cArrY[ind[1]][iNext] - x0[1];
    
//     /* STAGE 1 */
//     { // UPDATE TO BILINEAR
//       xloc_p[0] = (coor_p_k[0] - x0[0])/dx[0];
//       xloc_p[1] = (coor_p_k[1] - x0[1])/dx[1];
      
//       /* Checks (xi_p is only used for this, here) */
//       xi_p[0] = 2.0 * xloc_p[0] -1.0;
//       xi_p[1] = 2.0 * xloc_p[1] -1.0;
//       if (xi_p[0] < -1.0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"value (xi) too small %1.4e [e=%D]",(double)xi_p[0],e);
//       if (xi_p[0] >  1.0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"value (xi) too large %1.4e [e=%D]",(double)xi_p[0],e);
//       if (xi_p[1] < -1.0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"value (eta) too small %1.4e [e=%D]",(double)xi_p[1],e);
//       if (xi_p[1] >  1.0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"value (eta) too large %1.4e [e=%D]",(double)xi_p[1],e);
      
//       /* interpolate velocity */
//       vLeft  = LA_vp[ind[1]][ind[0]][iVxLeft];
//       vRight = LA_vp[ind[1]][ind[0]][iVxRight];
//       vUp    = LA_vp[ind[1]][ind[0]][iVyUp];
//       vDown  = LA_vp[ind[1]][ind[0]][iVyDown];
//       vel_p[0] = xloc_p[0]*vRight + (1.0-xloc_p[0])*vLeft;
//       vel_p[1] = xloc_p[1]*vUp    + (1.0-xloc_p[1])*vDown;
//     }
    
//     /* Update Coordinates */
//     coor_p_star[0] = coor_p_k[0] + 0.5 * dt * vel_p[0];
//     coor_p_star[1] = coor_p_k[1] + 0.5 * dt * vel_p[1];
    
//     /* STAGE 2 */
//     { // UPDATE TO BILINEAR
//       xloc_p[0] = (coor_p_star[0] - x0[0])/dx[0];
//       xloc_p[1] = (coor_p_star[1] - x0[1])/dx[1];
      
//       /* Checks (xi_p is only used for this, here) */
//       xi_p[0] = 2.0 * xloc_p[0] -1.0;
//       xi_p[1] = 2.0 * xloc_p[1] -1.0;
//       if (xi_p[0] < -1.0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"value (xi) too small %1.4e [e=%D]",(double)xi_p[0],e);
//       if (xi_p[0] >  1.0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"value (xi) too large %1.4e [e=%D]",(double)xi_p[0],e);
//       if (xi_p[1] < -1.0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"value (eta) too small %1.4e [e=%D]",(double)xi_p[1],e);
//       if (xi_p[1] >  1.0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"value (eta) too large %1.4e [e=%D]",(double)xi_p[1],e);
      
//       /* interpolate velocity */
//       vLeft  = LA_vp[ind[1]][ind[0]][iVxLeft];
//       vRight = LA_vp[ind[1]][ind[0]][iVxRight];
//       vUp    = LA_vp[ind[1]][ind[0]][iVyUp];
//       vDown  = LA_vp[ind[1]][ind[0]][iVyDown];
//       vel_p_star[0] = xloc_p[0]*vRight + (1.0-xloc_p[0])*vLeft;
//       vel_p_star[1] = xloc_p[1]*vUp    + (1.0-xloc_p[1])*vDown;
//     }
    
//     /* Update Coordinates */
//     mpfield_coor_kp1[2*p+0] = coor_p_k[0] + dt * vel_p_star[0];
//     mpfield_coor_kp1[2*p+1] = coor_p_k[1] + dt * vel_p_star[1];
//   }
  
//   PetscCall(DMSwarmRestoreField(dm_mpoint,cellid,NULL,NULL,(void**)&mpfield_cell));
//   PetscCall(DMStagRestoreProductCoordinateArraysRead(dm_vp,&cArrX,&cArrY,NULL));
//   PetscCall(DMStagVecRestoreArrayRead(dm_vp,vp_l,&LA_vp));
//   PetscFunctionReturn(PETSC_SUCCESS);
// }

// ---------------------------------------
// /*
//  Collective
// */
// PetscErrorCode MPoint_AdvectRK2_MPI(DM dmswarmI,DM dmstag,Vec Xlocal,PetscReal dt)
// {
//   PetscInt es[] = {0,0},nele[] = {0,0};
//   PetscInt *mask;
//   DM dmswarmB;
  
//   PetscFunctionBegin;
//   PetscCall(DMStagGetCorners(dmstag,&es[0],&es[1],NULL,&nele[0],&nele[1],NULL,NULL,NULL,NULL));
//   es[0] = es[1] = 0;
//   PetscCall(PetscCalloc1(nele[0]*nele[1],&mask));
//   for (ej=1; ej<nele[1]-1; ej++) {
//     for (ei=1; ei<nele[0]-1; ei++) {
//       mask[ei + ej * nele[0]] = 1;
//     }
//   }
  
//   nestimate = 16 * ( 2 * nele[0] + 2*(nele[1] - 2) );
  
//   // create swarm for points living in the boundary
//   PetscCall(DMStagPICCreateDMSwarm(dmstag,&dmswarmB));
//   PetscCall(DMSwarmDuplicateRegisteredFields(dmswarmB));
//   PetscCall(DMSwarmRegisterPetscDatatypeField(dmswarmB,"coor_prime",2,PETSC_REAL));
//   PetscCall(DMStagPICFinalize(dmswarmB));
//   PetscCall(DMSwarmSetLocalSizes(dmswarmB,0,-1));
  
//   // count and create mask
//   for (p=0; p<npoints; p++) {
//   }
  
//   // copy
//   PetscCall(DMSwarmCopySubsetFieldValues(dmswarmI,PetscInt np,PetscInt list[],dmswarmB,&copy_occurred));
  
//   // delete
//   PetscCall(DMSwarmRemovePoints(dmswarmI,PetscInt np,PetscInt list[]));

//   PetscReal *mpfield_coor,*mpfield_prime;
  
//   // advect b
//   PetscCall(DMSwarmGetField(dmswarmB,DMSwarmPICField_coor,NULL,NULL,(void**)&mpfield_coor));
//   PetscCall(DMSwarmGetField(dmswarmB,"coor_prime",NULL,NULL,(void**)&mpfield_coor_prime));
  
//   PetscCall(MPoint_AdvectRK1_Private(dmswarmB,mpfield_coor,mpfield_coor,mpfield_coor_prime,dmstag,Xlocal,0.5 * dt));
  
//   PetscCall(DMSwarmRestoreField(dmswarmB,"coor_prime",NULL,NULL,(void**)&mpfield_coor_prime));
//   PetscCall(DMSwarmRestoreField(dmswarmB,DMSwarmPICField_coor,NULL,NULL,(void**)&mpfield_coor));
//   // migrate b
//   PetscCall(DMSwarmMigrate(dmswarmB,PETSC_TRUE));
  
//   // advect b
//   PetscCall(DMSwarmGetField(dmswarmB,DMSwarmPICField_coor,NULL,NULL,(void**)&mpfield_coor));
//   PetscCall(DMSwarmGetField(dmswarmB,"coor_prime",NULL,NULL,(void**)&mpfield_coor_prime));
  
//   PetscCall(MPoint_AdvectRK1_Private(dmswarmB,mpfield_coor,mpfield_coor_prime,mpfield_coor,dmstag,Xlocal,0.5 * dt));
  
//   PetscCall(DMSwarmRestoreField(dmswarmB,"coor_prime",NULL,NULL,(void**)&mpfield_coor_prime));
//   PetscCall(DMSwarmRestoreField(dmswarmB,DMSwarmPICField_coor,NULL,NULL,(void**)&mpfield_coor));
//   // migrate b
//   PetscCall(DMSwarmMigrate(dmswarmB,PETSC_TRUE));

//   PetscCall(MPoint_AdvectRK2_SEQ_Private(dmswarmI,dmstag,Xlocal,dt));

//   // insert B into I
//   PetscCall(DMSwarmCopyFieldValues(dmswarmB,dmswarmI,&copy_occurred));

//   PetscCall(DMDestroy(&dmswarmB));
//   PetscCall(PetscFree(list));
//   PetscCall(PetscFree(mask));
  
//   PetscFunctionReturn(PETSC_SUCCESS);
// }

// ---------------------------------------
// /*
//  Not collective
// */
// PetscErrorCode MPoint_AdvectRK2_SEQ(DM dmswarm,DM dmstag,Vec Xlocal,PetscReal dt)
// {
//   PetscReal      *mpfield_coor;
  
//   PetscFunctionBegin;
//   PetscCall(DMSwarmGetField(dmswarm,DMSwarmPICField_coor,NULL,NULL,(void**)&mpfield_coor));
//   PetscCall(MPoint_AdvectRK2_SEQ_Private(dmswarm,mpfield_coor,mpfield_coor,dmstag,Xlocal,dt));
//   PetscCall(DMSwarmRestoreField(dmswarm,DMSwarmPICField_coor,NULL,NULL,(void**)&mpfield_coor));
//   /* point location */
  
//   PetscFunctionReturn(PETSC_SUCCESS);
// }

// ---------------------------------------
// /*
//  Collective
// */
// PetscErrorCode MPoint_AdvectRK2(DM dmswarm,DM dmstag,Vec X,PetscReal dt)
// {
//   Vec            Xlocal;
  
//   PetscFunctionBegin;
//   PetscCall(DMGetLocalVector(dmstag,&Xlocal));
//   PetscCall(DMGlobalToLocalBegin(dmstag,X,INSERT_VALUES,Xlocal));
//   PetscCall(DMGlobalToLocalEnd(dmstag,X,INSERT_VALUES,Xlocal));
  
//   PetscCall(MPoint_AdvectRK2_SEQ(dmswarm,dmstag,Xlocal,dt));
//   PetscCall(MPoint_AdvectRK2_MPI(dmswarm,dmstag,Xlocal,dt));

//   PetscCall(DMRestoreLocalVector(dmstag,&Xlocal));
//   PetscFunctionReturn(PETSC_SUCCESS);
// }

// #endif

// ---------------------------------------
/* copy x into y */
static PetscErrorCode DMSwarmDataFieldCopyPoint(const PetscInt pid_x,const DMSwarmDataField field_x,
                        const PetscInt pid_y,const DMSwarmDataField field_y )
{
  PetscFunctionBegin;
#if defined(DMSWARM_DATAFIELD_POINT_ACCESS_GUARD)
  /* check point is valid */
  if (pid_x < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"(IN) index must be >= 0");
  if (pid_x >= field_x->L) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"(IN) index must be < %D",field_x->L);
  if (pid_y < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"(OUT) index must be >= 0");
  if (pid_y >= field_y->L) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"(OUT) index must be < %D",field_y->L);
  if( field_y->atomic_size != field_x->atomic_size ) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"atomic size must match");
#endif
  PetscCall(PetscMemcpy(DMSWARM_DATAFIELD_point_access(field_y->data,pid_y,field_y->atomic_size),DMSWARM_DATAFIELD_point_access(field_x->data,pid_x,field_x->atomic_size),field_y->atomic_size));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
static PetscErrorCode _DMSwarmDataFieldStringFindInList(const char name[],const PetscInt N,const DMSwarmDataField gfield[],PetscInt *index)
{
  PetscInt       i;

  PetscFunctionBegin;
  *index = -1;
  for (i = 0; i < N; ++i) {
    PetscBool flg;
    PetscCall(PetscStrcmp(name, gfield[i]->name, &flg));
    if (flg) {
      *index = i;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
/*
 Query if a DMSwarm defines a particular property
 Not collective
*/
PetscErrorCode DMSwarmQueryField(DM dm,const char fieldname[],PetscBool *found)
{
  DM_Swarm  *s = (DM_Swarm*)dm->data;
  PetscInt index = -1;
  
  PetscFunctionBegin;
  *found = PETSC_FALSE;
  _DMSwarmDataFieldStringFindInList(fieldname,s->db->nfields,s->db->field,&index);
  if (index >= 0) *found = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
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

  PetscFunctionBegin;
  dbA = swarmA->db;
  for (f=0; f<dbA->nfields; f++) {
    PetscCall(DMSwarmQueryField(dmB,dbA->field[f]->name,&found));
    if (!found) {
      PetscCall(DMSwarmRegisterPetscDatatypeField(dmB,dbA->field[f]->name,dbA->field[f]->bs,dbA->field[f]->petsc_type));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
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
  PetscInt lenA,lenB,lenB_new,nfA,nfB;
  DM_Swarm  *swarmA = (DM_Swarm*)dmA->data;
  DM_Swarm  *swarmB = (DM_Swarm*)dmB->data;
  DMSwarmDataBucket dbA,dbB;
  PetscInt *fieldfrom,*fieldto;
  PetscInt f;
  PetscBool detected_common_members=PETSC_FALSE;
  
  PetscFunctionBegin;
  *copy_occurred = PETSC_FALSE;
  PetscCall(DMSwarmGetLocalSize(dmA,&lenA));
  PetscCall(DMSwarmGetLocalSize(dmB,&lenB));
  dbA = swarmA->db;
  dbB = swarmB->db;
  nfA = dbA->nfields;
  nfB = dbB->nfields;
  
  PetscCall(PetscMalloc1(nfA,&fieldfrom));
  PetscCall(PetscMalloc1(nfA,&fieldto));
  for (f=0; f<nfA; f++) {
    fieldfrom[f] = -1;
    fieldto[f] = -1;
  }

  for (f=0; f<nfA; f++) {
    PetscInt index = -1;
    _DMSwarmDataFieldStringFindInList(dbA->field[f]->name, dbB->nfields,dbB->field,&index);
    if (index >= 0) {
      fieldfrom[f] = f;
      fieldto[f] = index;
      detected_common_members = PETSC_TRUE;
    }
  }
  
  if (detected_common_members) {
    PetscInt p;
    
    PetscCall(DMSwarmSetLocalSizes(dmB,lenB+lenA,-1));
    PetscCall(DMSwarmGetLocalSize(dmB,&lenB_new));
    
    for (f=0; f<nfA; f++) {
      if (fieldfrom[f] != -1) {

        for (p=0; p<lenA; p++) {
          PetscCall(DMSwarmDataFieldCopyPoint(p,       dbA->field[ fieldfrom[f] ],
                                           lenB + p,dbB->field[ fieldto[f] ]));
        }
      }
    }
    *copy_occurred = PETSC_TRUE;
  }
  
  PetscCall(PetscFree(fieldfrom));
  PetscCall(PetscFree(fieldto));
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
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
  PetscInt lenA,lenB,lenB_new,nfA,nfB;
  DM_Swarm  *swarmA = (DM_Swarm*)dmA->data;
  DM_Swarm  *swarmB = (DM_Swarm*)dmB->data;
  DMSwarmDataBucket dbA,dbB;
  PetscInt *fieldfrom,*fieldto;
  PetscInt f;
  PetscBool detected_common_members=PETSC_FALSE;
  
  PetscFunctionBegin;
  *copy_occurred = PETSC_FALSE;
  /*PetscCall(DMSwarmGetLocalSize(dmA,&lenA));*/
  lenA = np;
  PetscCall(DMSwarmGetLocalSize(dmB,&lenB));
  dbA = swarmA->db;
  dbB = swarmB->db;
  nfA = dbA->nfields;
  nfB = dbB->nfields;
  
  PetscCall(PetscMalloc1(nfA,&fieldfrom));
  PetscCall(PetscMalloc1(nfA,&fieldto));
  for (f=0; f<nfA; f++) {
    fieldfrom[f] = -1;
    fieldto[f] = -1;
  }
  
  for (f=0; f<nfA; f++) {
    PetscInt index = -1;
    _DMSwarmDataFieldStringFindInList(dbA->field[f]->name, dbB->nfields,dbB->field,&index);
    if (index >= 0) {
      fieldfrom[f] = f;
      fieldto[f] = index;
      detected_common_members = PETSC_TRUE;
    }
  }
  
  if (detected_common_members) {
    PetscInt p,index;
    
    PetscCall(DMSwarmSetLocalSizes(dmB,lenB+lenA,-1));
    PetscCall(DMSwarmGetLocalSize(dmB,&lenB_new));
    
    for (f=0; f<nfA; f++) {
      if (fieldfrom[f] != -1) {
        
        for (p=0; p<np; p++) {
          index = list[p];
          
          PetscCall(DMSwarmDataFieldCopyPoint(index,   dbA->field[ fieldfrom[f] ],
                                           lenB + p,dbB->field[ fieldto[f] ]));
        }
      }
    }
    *copy_occurred = PETSC_TRUE;
  }
  
  PetscCall(PetscFree(fieldfrom));
  PetscCall(PetscFree(fieldto));
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// PetscErrorCode v0_DMSwarmRemovePoints(DM dm,PetscInt np,PetscInt list[])
// {
//   PetscInt p;
  
//   PetscFunctionBegin;
//   for (p=0; p<np; p++) {
//     PetscInt index = list[p];
//     PetscCall(DMSwarmRemovePointAtIndex(dm,index));
//   }
//   PetscFunctionReturn(PETSC_SUCCESS);
// }

// ---------------------------------------
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
  
  PetscFunctionBegin;
  PetscCall(PetscSortInt(np,list));
  
  PetscCall(DMSwarmGetLocalSize(dm,&npoints));
  PetscCall(DMSwarmGetField(dm,DMSwarmField_pid,NULL,NULL,(void**)&pid_ref));
  pid = pid_ref;
  PetscCall(DMSwarmRestoreField(dm,DMSwarmField_pid,NULL,NULL,(void**)&pid_ref));
  
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
    PetscCall(DMSwarmCopyPoint(dm,from,to));
    
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

  PetscCall(DMSwarmSetLocalSizes(dm,npoints-np,-1));

  /* sanity check */
  PetscCall(DMSwarmGetLocalSize(dm,&npoints));
  PetscCall(DMSwarmGetField(dm,DMSwarmField_pid,NULL,NULL,(void**)&pid_ref));
  for (p=0; p<npoints; p++) {
    if (pid_ref[p] == kill_point) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Point at index %D should have been deleted",p);
  }
  PetscCall(DMSwarmRestoreField(dm,DMSwarmField_pid,NULL,NULL,(void**)&pid_ref));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
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
  
  PetscFunctionBegin;
  PetscCall(DMSwarmGetLocalSize(dm,&len));
  PetscCall(DMSwarmGetField(dm,fieldname,&bs,&type,(void**)&pfield));
  if (type != PETSC_REAL) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Type must be PETSC_REAL for field %s",fieldname);
  len = len * bs;
  for (p=0; p<len; p++) {
    pfield[p] = alpha;
  }
  PetscCall(DMSwarmRestoreField(dm,fieldname,NULL,NULL,(void**)&pfield));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
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
  
  PetscFunctionBegin;
  PetscCall(DMSwarmGetLocalSize(dm,&len));
  if (ps < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Range too small start(%" PetscInt_FMT ") < 0",ps);
  if (pe > len) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Range too larage end(%" PetscInt_FMT ") > len(%" PetscInt_FMT ")",pe,len);
  PetscCall(DMSwarmGetField(dm,fieldname,&bs,&type,(void**)&pfield));
  if (type != PETSC_REAL) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Type must be PETSC_REAL for field %s",fieldname);
  for (p=bs*ps; p<bs*pe; p++) {
    pfield[p] = alpha;
  }
  PetscCall(DMSwarmRestoreField(dm,fieldname,NULL,NULL,(void**)&pfield));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
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
  
  PetscFunctionBegin;
  PetscCall(DMSwarmGetLocalSize(dm,&len));
  PetscCall(DMSwarmGetField(dm,fieldname,&bs,&type,(void**)&pfield));
  if (type != PETSC_REAL) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Type must be PETSC_REAL for field %s",fieldname);
  for (p=0; p<np; p++) {
    PetscInt index = list[p];
    if (index < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Index too small start(%" PetscInt_FMT ") < 0",index);
    if (index > len) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Index too larage end(%" PetscInt_FMT ") > len(%" PetscInt_FMT ")",index,len);
    for (b=0; b<bs; b++) {
      pfield[bs*index+b] = alpha;
    }
  }
  PetscCall(DMSwarmRestoreField(dm,fieldname,NULL,NULL,(void**)&pfield));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
PetscErrorCode DMSetPointLocation(DM dm,PetscErrorCode (*f)(DM,Vec,DMPointLocationType,PetscSF))
{
  PetscFunctionBegin;
  dm->ops->locatepoints = f;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
static PetscErrorCode DMStagGetLocalBoundingBox_2d(DM dm,PetscReal lmin[],PetscReal lmax[])
{
  PetscInt          i,start[2],n[2],ind;
  const PetscScalar *_coor;
  PetscScalar       **cArrX,**cArrY;
  PetscInt          iNext,iPrev;
  PetscReal         min[3] = {PETSC_MAX_REAL, PETSC_MAX_REAL, PETSC_MAX_REAL};
  PetscReal         max[3] = {PETSC_MIN_REAL, PETSC_MIN_REAL, PETSC_MIN_REAL};

  PetscFunctionBegin;
  PetscCall(DMStagGetCorners(dm,&start[0],&start[1],NULL,&n[0],&n[1],NULL,NULL,NULL,NULL));
  PetscCall(DMStagGetProductCoordinateArraysRead(dm,&cArrX,&cArrY,NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm,DMSTAG_LEFT,&iPrev));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm,DMSTAG_RIGHT,&iNext));

  for (ind=start[0]; ind<start[0]+n[0]; ++ind) {
    min[0] = PetscMin(min[0],cArrX[ind][iPrev]);
    min[0] = PetscMin(min[0],cArrX[ind][iNext]);
    max[0] = PetscMax(max[0],cArrX[ind][iPrev]);
    max[0] = PetscMax(max[0],cArrX[ind][iNext]);
  }
  for (ind=start[1]; ind<start[1]+n[1]; ++ind) {
    min[1] = PetscMin(min[1],cArrY[ind][iPrev]);
    min[1] = PetscMin(min[1],cArrY[ind][iNext]);
    max[1] = PetscMax(max[1],cArrY[ind][iPrev]);
    max[1] = PetscMax(max[1],cArrY[ind][iNext]);
  }
  
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm,&cArrX,&cArrY,NULL));

  if (lmin) {PetscArraycpy(lmin, min, 2);}
  if (lmax) {PetscArraycpy(lmax, max, 2);}
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
PetscErrorCode DMStagGetBoundingBox(DM dm,PetscReal gmin[],PetscReal gmax[])
{
  PetscReal      lmin[]={0,0,0},lmax[]={0,0,0};
  PetscInt       cdim=2;
  PetscMPIInt    count;
  
  PetscFunctionBegin;
  PetscCall(PetscMPIIntCast(cdim,&count));
  PetscCall(DMStagGetLocalBoundingBox_2d(dm,lmin,lmax));
  if (gmin) {PetscCallMPI(MPIU_Allreduce(lmin,gmin,count,MPIU_REAL,MPIU_MIN,PetscObjectComm((PetscObject)dm)));}
  if (gmax) {PetscCallMPI(MPIU_Allreduce(lmax,gmax,count,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)dm)));}
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
PetscErrorCode DMStagLocalElementIndexInGlobalSpace_2d(DM dm,PetscInt eidx,PetscBool *value)
{
  const DM_Stag * const stag = (DM_Stag*)dm->data;
  PetscInt iLocal,jLocal,geid[2],si,sj,ei,ej;
  
  PetscFunctionBegin;
  iLocal = eidx % stag->nGhost[0];
  jLocal = (eidx - iLocal) / stag->nGhost[0];
  geid[0] = iLocal + stag->startGhost[0];
  geid[1] = jLocal + stag->startGhost[1];

  *value = PETSC_TRUE;
  si = stag->start[0]; if (geid[0] < si) { *value = PETSC_FALSE; PetscFunctionReturn(PETSC_SUCCESS); }
  sj = stag->start[1]; if (geid[1] < sj) { *value = PETSC_FALSE; PetscFunctionReturn(PETSC_SUCCESS); }
  
  ei = si + stag->n[0]; if (geid[0] >= ei) { *value = PETSC_FALSE; PetscFunctionReturn(PETSC_SUCCESS); }
  ej = sj + stag->n[1]; if (geid[1] >= ej) { *value = PETSC_FALSE; PetscFunctionReturn(PETSC_SUCCESS); }
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
PetscErrorCode DMStagFieldISCreate_2d(DM dm,
                                    PetscInt ndof0A,PetscInt dof0A[],
                                    PetscInt ndof1A,PetscInt dof1A[],
                                    PetscInt ndof2A,PetscInt dof2A[],IS *is)
{
  PetscInt f0,f1,f2,dim,dof0,dof1,dof2,sumDof,d,scnt;
  DMStagStencil *stencil_list;
  PetscFunctionBegin;
  
  PetscCall(DMGetDimension(dm,&dim));
  PetscCall(DMStagGetDOF(dm,&dof0,&dof1,&dof2,NULL));

  if (ndof0A > dof0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Number of dof on stratum 0 requested exceeds that defined on DMStag");
  for (d=0; d<ndof0A; d++) {
    if (dof0A[d] >= dof0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"dof0[] requested exceeds that defined on DMStag");
  }

  if (ndof1A > dof1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Number of dof on stratum 1 requested exceeds that defined on DMStag");
  for (d=0; d<ndof1A; d++) {
    if (dof1A[d] >= dof1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"dof1[] requested exceeds that defined on DMStag");
  }

  if (ndof2A > dof2) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Number of dof on stratum 2 requested exceeds that defined on DMStag");
  for (d=0; d<ndof2A; d++) {
    if (dof2A[d] >= dof2) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"dof2[] requested exceeds that defined on DMStag");
  }

  f0 = f1 = f2 = 1;
  f1 = 2; // dim == 2

  sumDof = f0*ndof0A + f1*ndof1A + f2*ndof2A;
  PetscCall(PetscCalloc1(sumDof,&stencil_list));
  
  /* vertex, edge(down,left), element */
  scnt = 0;
  for (d=0; d<ndof0A; d++) {
    stencil_list[scnt].loc = DMSTAG_DOWN_LEFT;
    stencil_list[scnt].c = dof0A[d];
    scnt++;
  }
  /* edge */
  for (d=0; d<ndof1A; d++) {
    stencil_list[scnt].loc = DMSTAG_DOWN;
    stencil_list[scnt].c = dof1A[d];
    scnt++;
  }
  for (d=0; d<ndof1A; d++) {
    stencil_list[scnt].loc = DMSTAG_LEFT;
    stencil_list[scnt].c = dof1A[d];
    scnt++;
  }
  /* element */
  for (d=0; d<ndof2A; d++) {
    stencil_list[scnt].loc = DMSTAG_ELEMENT;
    stencil_list[scnt].c = dof2A[d];
    scnt++;
  }

  PetscCall(DMStagCreateISFromStencils(dm,sumDof,stencil_list,is));
  PetscCall(PetscFree(stencil_list));
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
PetscErrorCode DMStagISCreateL2L_2d(DM dmA,
                                    PetscInt n0A,PetscInt dof0A[],
                                    PetscInt n1A,PetscInt dof1A[],
                                    PetscInt n2A,PetscInt dof2A[],IS *isA,
                                    DM dmB,
                                    PetscInt dof0B[],
                                    PetscInt dof1B[],
                                    PetscInt dof2B[],IS *isB)
{
  PetscFunctionBegin;
  PetscCall(DMStagFieldISCreate_2d(dmA,n0A,dof0A,n1A,dof1A,n2A,dof2A,isA));
  PetscCall(DMStagFieldISCreate_2d(dmB,n0A,dof0B,n1A,dof1B,n2A,dof2B,isB));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
/*@C
 DMStagGetLocalElementIndex - get global i,j,k element numbers for a specified index in local element numbering
 
 Not Collective
 
 Input Parameters:
 + dm - a DMStag object
 - ind - global element numbers for a local element
 
 Output Parameter:
 . eidx - local element number
 
 Level: advanced
 
 Notes:
 The local element numbering includes ghost elements, and proceeds as standard
 PETSc ordering (x fastest, z slowest from back lower left to front upper right)
 
 ind must be at least as large as the dimension of the DMStag object, in order
 to accept the appropriate number of indices.
 
 .seealso: DMSTAG, DMStagGetLocalElementGlobalIndices()
 @*/
PetscErrorCode DMStagGetLocalElementIndex(DM dm,PetscInt *ind,PetscInt *eidx)
{
  const DM_Stag * const stag  = (DM_Stag*)dm->data;
  PetscInt              dim,iLocal,jLocal;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMSTAG);
  PetscCall(DMGetDimension(dm,&dim));
#ifdef PETSC_USE_DEBUG
  {
    PetscInt d,startGhost[DMSTAG_MAX_DIM],nGhost[DMSTAG_MAX_DIM];
    PetscCall(DMStagGetGhostCorners(dm,&startGhost[0],&startGhost[1],&startGhost[2],&nGhost[0],&nGhost[1],&nGhost[2]));
    for (d=0; d<dim; ++d) {
      if (ind[d] < startGhost[d] || ind[d] >= startGhost[d]+nGhost[d]) {
        SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"element index ind[%" PetscInt_FMT "] is out of range. It should be in [%" PetscInt_FMT ",%" PetscInt_FMT ") but it is %" PetscInt_FMT,d,startGhost[d],startGhost[d]+nGhost[d],ind[d]);
      }
    }
  }
#endif
  switch (dim) {
    case 2:
      iLocal = ind[0] - stag->startGhost[0];
      jLocal = ind[1] - stag->startGhost[1];
      *eidx =  jLocal * stag->nGhost[0] + iLocal;
      break;
    default: SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Unsupported dimension %" PetscInt_FMT,dim);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// // /* Convert an array of DMStagStencil objects to an array of indices into a local vector.
// //  The .c fields in pos must always be set (even if to 0).  */
// // PetscErrorCode DMStagStencilToIndexLocal(DM dm,PetscInt n,const DMStagStencil *pos,PetscInt *ix)
// // {
// //   const DM_Stag * const stag = (DM_Stag*)dm->data;
// //   PetscInt              idx,dim,startGhost[DMSTAG_MAX_DIM];
// //   const PetscInt        epe = stag->entriesPerElement;
  
// //   PetscFunctionBegin;
// //   PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMSTAG);
// //   PetscCall(DMGetDimension(dm,&dim));
// // #if defined(PETSC_USE_DEBUG)
// //   {
// //     PetscInt i,nGhost[DMSTAG_MAX_DIM],endGhost[DMSTAG_MAX_DIM];
// //     PetscCall(DMStagGetGhostCorners(dm,&startGhost[0],&startGhost[1],&startGhost[2],&nGhost[0],&nGhost[1],&nGhost[2]));
// //     for (i=0; i<DMSTAG_MAX_DIM; ++i) endGhost[i] = startGhost[i] + nGhost[i];
// //     for (i=0; i<n; ++i) {
// //       PetscInt dof;
// //       PetscCall(DMStagGetLocationDOF(dm,pos[i].loc,&dof));
// //       if (dof < 1) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Location %s has no dof attached",DMStagStencilLocations[pos[i].loc]);
// //       if (pos[i].c < 0) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Negative component number (%d) supplied in loc[%" PetscInt_FMT "]",pos[i].c,i);
// //       if (pos[i].c > dof-1) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Supplied component number (%" PetscInt_FMT ") for location %s is too big (maximum %" PetscInt_FMT ")",pos[i].c,DMStagStencilLocations[pos[i].loc],dof-1);
// //       if (            pos[i].i >= endGhost[0] || pos[i].i < startGhost[0] ) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Supplied x element index %D out of range. Should be in [%D,%D]",pos[i].i,startGhost[0],endGhost[0]-1);
// //       if (dim > 1 && (pos[i].j >= endGhost[1] || pos[i].j < startGhost[1])) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Supplied y element index %D out of range. Should be in [%D,%D]",pos[i].j,startGhost[1],endGhost[1]-1);
// //       if (dim > 2 && (pos[i].k >= endGhost[2] || pos[i].k < startGhost[2])) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Supplied z element index %D out of range. Should be in [%D,%D]",pos[i].k,startGhost[2],endGhost[2]-1);
// //     }
// //   }
// // #else
// //   PetscCall(DMStagGetGhostCorners(dm,&startGhost[0],&startGhost[1],&startGhost[2],NULL,NULL,NULL));
// // #endif
// //   if (dim == 1) {
// //     for (idx=0; idx<n; ++idx) {
// //       const PetscInt eLocal = pos[idx].i - startGhost[0]; /* Local element number */
// //       ix[idx] = eLocal * epe + stag->locationOffsets[pos[idx].loc] + pos[idx].c;
// //     }
// //   } else if (dim == 2) {
// //     const PetscInt epr = stag->nGhost[0];
// //     PetscCall(DMStagGetGhostCorners(dm,&startGhost[0],&startGhost[1],NULL,NULL,NULL,NULL));
// //     for (idx=0; idx<n; ++idx) {
// //       const PetscInt eLocalx = pos[idx].i - startGhost[0];
// //       const PetscInt eLocaly = pos[idx].j - startGhost[1];
// //       const PetscInt eLocal = eLocalx + epr*eLocaly;
// //       ix[idx] = eLocal * epe + stag->locationOffsets[pos[idx].loc] + pos[idx].c;
// //     }
// //   } else if (dim == 3) {
// //     const PetscInt epr = stag->nGhost[0];
// //     const PetscInt epl = stag->nGhost[0]*stag->nGhost[1];
// //     PetscCall(DMStagGetGhostCorners(dm,&startGhost[0],&startGhost[1],&startGhost[2],NULL,NULL,NULL));
// //     for (idx=0; idx<n; ++idx) {
// //       const PetscInt eLocalx = pos[idx].i - startGhost[0];
// //       const PetscInt eLocaly = pos[idx].j - startGhost[1];
// //       const PetscInt eLocalz = pos[idx].k - startGhost[2];
// //       const PetscInt eLocal  = epl*eLocalz + epr*eLocaly + eLocalx;
// //       ix[idx] = eLocal * epe + stag->locationOffsets[pos[idx].loc] + pos[idx].c;
// //     }
// //   } else SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Unsupported dimension %d",dim);
// //   PetscFunctionReturn(PETSC_SUCCESS);
// // }

// ---------------------------------------
/*@C
 DMStagGetLocalElementGlobalIndices - get global i,j,k element numbers for a specified index in local element numbering
 
 Not Collective
 
 Input Parameters:
 + dm - a DMStag object
 - eidx - local element number
 
 Output Parameter:
 . ind - array of global element numbers for local elements
 
 Level: advanced
 
 Notes:
 The local element numbering includes ghost elements, and proceeds as standard
 PETSc ordering (x fastest, z slowest from back lower left to front upper right)
 
 ind must be at least as large as the dimension of the DMStag object, in order
 to accept the appropriate number of indices.
 
 .seealso: DMSTAG, DMStagGetLocalElementIndex()
 @*/
PetscErrorCode DMStagGetLocalElementGlobalIndices(DM dm,PetscInt eidx,PetscInt *ind)
{
  const DM_Stag * const stag  = (DM_Stag*)dm->data;
  PetscInt              dim,iLocal,jLocal;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMSTAG);
  PetscCall(DMGetDimension(dm,&dim));
#ifdef PETSC_USE_DEBUG
  {
    PetscInt nelLocal;
    switch (dim) {
      case 1:
        nelLocal = stag->nGhost[0];
        break;
      case 2:
        nelLocal = stag->nGhost[0]*stag->nGhost[1];
        break;
      case 3:
        nelLocal = stag->nGhost[0]*stag->nGhost[1]*stag->nGhost[2];
        break;
      default:SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Unsupported dimension %" PetscInt_FMT,dim);
    }
    if (eidx >= nelLocal) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Element index %" PetscInt_FMT " is too large. There are only %" PetscInt_FMT " local elements",eidx,nelLocal);
  }
#endif
  switch (dim) {
    case 2:
      iLocal = eidx % stag->nGhost[0];
      jLocal = (eidx - iLocal) / stag->nGhost[0];
      ind[0] = iLocal + stag->startGhost[0];
      ind[1] = jLocal + stag->startGhost[1];
      break;
    default: SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Unsupported dimension %" PetscInt_FMT,dim);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
static PetscErrorCode DMStagLocatePointsIS_2D_Product_ConstantSpacing_Private(DM dm,Vec pos,IS *iscell)
{
  PetscInt          localSize,bs,p,npoints,start[2],n[2];
  PetscInt          *cellidx;
  const PetscScalar *_coor;
  PetscScalar       **cArrX,**cArrY;
  PetscInt          iNext,iPrev,Ng[]={0,0,0};
  PetscReal         dx[2],gmin[]={0,0,0},gmax[]={0,0,0};
  
  PetscFunctionBegin;
  
  PetscCall(VecGetLocalSize(pos,&localSize));
  PetscCall(VecGetBlockSize(pos,&bs));
  npoints = localSize/bs;
  
  PetscCall(PetscMalloc1(npoints,&cellidx));
  PetscCall(VecGetArrayRead(pos,&_coor));
  
  PetscCall(DMStagGetCorners(dm,&start[0],&start[1],NULL,&n[0],&n[1],NULL,NULL,NULL,NULL));
  PetscCall(DMStagGetProductCoordinateArraysRead(dm,&cArrX,&cArrY,NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm,DMSTAG_LEFT,&iPrev));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm,DMSTAG_RIGHT,&iNext));
  
  PetscCall(DMStagGetGlobalSizes(dm,&Ng[0],&Ng[1],NULL));
  PetscCall(DMStagGetBoundingBox(dm,gmin,gmax));
  dx[0] = (gmax[0]-gmin[0])/Ng[0];
  dx[1] = (gmax[1]-gmin[1])/Ng[1];

  for (p=0; p<npoints; p++) {
    PetscReal coor_p[2];
    PetscInt gei[]={0,0},elocal;
    
    cellidx[p] = DMLOCATEPOINT_POINT_NOT_FOUND;

    coor_p[0] = PetscRealPart(_coor[2*p]);
    coor_p[1] = PetscRealPart(_coor[2*p+1]);

    if (coor_p[0] < gmin[0]) continue;
    if (coor_p[0] > gmax[0]) continue;
    if (coor_p[1] < gmin[1]) continue;
    if (coor_p[1] > gmax[1]) continue;
    
    gei[0] = (PetscInt)( (coor_p[0] - gmin[0])/dx[0] );
    gei[1] = (PetscInt)( (coor_p[1] - gmin[1])/dx[1] );
    if (gei[0] == Ng[0]) gei[0]--;
    if (gei[1] == Ng[1]) gei[1]--;
    
    if (gei[0] < start[0]) continue;
    if (gei[1] < start[1]) continue;
    if (gei[0] >= (start[0]+n[0])) continue;
    if (gei[1] >= (start[1]+n[1])) continue;
    
    PetscCall(DMStagGetLocalElementIndex(dm,gei,&elocal));
    cellidx[p] = elocal;
  }
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm,&cArrX,&cArrY,NULL));
  PetscCall(VecRestoreArrayRead(pos,&_coor));
  
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF,npoints,cellidx,PETSC_OWN_POINTER,iscell));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
PetscErrorCode DMLocatePoints_Stag(DM dm,Vec pos,DMPointLocationType ltype,PetscSF cellSF)
{
  IS             iscell;
  PetscSFNode    *cells;
  PetscInt       p,bs,dim,npoints,nfound;
  const PetscInt *boxCells;
  
  PetscFunctionBegin;
  if (ltype != DM_POINTLOCATION_NONE) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Only DMPOINTLOCATION_NONE is supported for ltype");
  {
    DM        cdm;
    PetscBool isProduct;
    
    PetscCall(DMGetCoordinateDM(dm,&cdm));
    if (!cdm) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Cannot locate points without coordinates");
    PetscCall(PetscObjectTypeCompare((PetscObject)cdm,DMPRODUCT,&isProduct));
    if (!isProduct) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Point location only supported for product coordinates");
  }
  PetscCall(VecGetBlockSize(pos,&dim));
  switch (dim) {
    case 2:
      //PetscCall(DMStagLocatePointsIS_2D_Product_Private(dm,pos,&iscell));
      PetscCall(DMStagLocatePointsIS_2D_Product_ConstantSpacing_Private(dm,pos,&iscell));
      break;
    default: SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Unsupported spatial dimension %D",dim);
  }
  
  PetscCall(VecGetLocalSize(pos,&npoints));
  PetscCall(VecGetBlockSize(pos,&bs));
  npoints = npoints / bs;
  
  PetscCall(PetscMalloc1(npoints, &cells));
  PetscCall(ISGetIndices(iscell, &boxCells));
  
  for (p=0; p<npoints; p++) {
    cells[p].rank  = 0;
    cells[p].index = boxCells[p];
  }
  PetscCall(ISRestoreIndices(iscell, &boxCells));
  
  nfound = npoints;
  PetscCall(PetscSFSetGraph(cellSF, npoints, nfound, NULL, PETSC_OWN_POINTER, cells, PETSC_OWN_POINTER));
  PetscCall(ISDestroy(&iscell));
  
  PetscFunctionReturn(PETSC_SUCCESS);
}