/*
 Many of these functions are all taken from
 [url] https://bitbucket.org/psanan/petsc
 [branch] psanan/stagbl-working-base
 They will all be added to the next PETSc release (v3.13).
 When switching to v3.13, this file should be purged.
*/

#include <petsc.h>
#include <petscvec.h>
#include <petscdm.h>
#include <petscdmstag.h>
#include <petsc/private/dmstagimpl.h>
#include <petscdmproduct.h>


#include "dmstag_utils.h"

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
  PetscErrorCode        ierr;
  const DM_Stag * const stag  = (DM_Stag*)dm->data;
  PetscInt              dim,iLocal,jLocal;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMSTAG);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
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
      default:SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Unsupported dimension %D",dim);
    }
    if (eidx >= nelLocal) SETERRQ2(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Element index %D is too large. There are only %D local elements",eidx,nelLocal);
  }
#endif
  switch (dim) {
    case 2:
      iLocal = eidx % stag->nGhost[0];
      jLocal = (eidx - iLocal) / stag->nGhost[0];
      ind[0] = iLocal + stag->startGhost[0];
      ind[1] = jLocal + stag->startGhost[1];
      break;
    default: SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Unsupported dimension %D",dim);
  }
  PetscFunctionReturn(0);
}

/*@C
 DMStagVecGetArray - get access to local array
 
 Logically Collective
 
 This function returns a (dim+1)-dimensional array for a dim-dimensional
 DMStag.
 
 The first 1-3 dimensions indicate an element in the global
 numbering, using the standard C ordering.
 
 The final dimension in this array corresponds to a degree
 of freedom with respect to this element, for example corresponding to
 the element or one of its neighboring faces, edges, or vertices.
 
 For example, for a 3D DMStag, indexing is array[k][j][i][idx], where k is the
 index in the z-direction, j is the index in the y-direction, and i is the
 index in the x-direction.
 
 "idx" is obtained with DMStagGetLocationSlot(), since the correct offset
 into the (dim+1)-dimensional C array depends on the grid size and the number
 of dof stored at each location.
 
 Input Parameters:
 + dm - the DMStag object
 - vec - the Vec object
 
 Output Parameters:
 . array - the array
 
 Notes:
 DMStagVecRestoreArray() must be called, once finished with the array
 
 Level: beginner
 
 .seealso: DMSTAG, DMStagVecGetArrayRead(), DMStagGetLocationSlot(), DMGetLocalVector(), DMCreateLocalVector(), DMGetGlobalVector(), DMCreateGlobalVector(), DMDAVecGetArray(), DMDAVecGetArrayDOF()
 @*/
PetscErrorCode DMStagVecGetArray(DM dm,Vec vec,void *array)
{
  PetscErrorCode  ierr;
  DM_Stag * const stag = (DM_Stag*)dm->data;
  PetscInt        dim;
  PetscInt        nLocal;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMSTAG);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,2);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  ierr = VecGetLocalSize(vec,&nLocal);CHKERRQ(ierr);
  if (nLocal != stag->entriesGhost) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Vector local size %D is not compatible with DMStag local size %D\n",nLocal,stag->entriesGhost);
  switch (dim) {
    case 1:
      ierr = VecGetArray2d(vec,stag->nGhost[0],stag->entriesPerElement,stag->startGhost[0],0,(PetscScalar***)array);CHKERRQ(ierr);
      break;
    case 2:
      ierr = VecGetArray3d(vec,stag->nGhost[1],stag->nGhost[0],stag->entriesPerElement,stag->startGhost[1],stag->startGhost[0],0,(PetscScalar****)array);CHKERRQ(ierr);
      break;
    case 3:
      ierr = VecGetArray4d(vec,stag->nGhost[2],stag->nGhost[1],stag->nGhost[0],stag->entriesPerElement,stag->startGhost[2],stag->startGhost[1],stag->startGhost[0],0,(PetscScalar*****)array);CHKERRQ(ierr);
      break;
    default: SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Unsupported dimension %D",dim);
  }
  PetscFunctionReturn(0);
}

/*@C
 DMStagVecRestoreArray - restore access to a raw array
 
 Logically Collective
 
 Input Parameters:
 + dm - the DMStag object
 - vec - the Vec object
 
 Output Parameters:
 . array - the array
 
 Level: beginner
 
 .seealso: DMSTAG, DMStagVecGetArray(), DMDAVecRestoreArray(), DMDAVecRestoreArrayDOF()
 @*/
PetscErrorCode DMStagVecRestoreArray(DM dm,Vec vec,void *array)
{
  PetscErrorCode  ierr;
  DM_Stag * const stag = (DM_Stag*)dm->data;
  PetscInt        dim;
  PetscInt        nLocal;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMSTAG);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,2);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  ierr = VecGetLocalSize(vec,&nLocal);CHKERRQ(ierr);
  if (nLocal != stag->entriesGhost) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Vector local size %D is not compatible with DMStag local size %D\n",nLocal,stag->entriesGhost);
  switch (dim) {
    case 1:
      ierr = VecRestoreArray2d(vec,stag->nGhost[0],stag->entriesPerElement,stag->startGhost[0],0,(PetscScalar***)array);CHKERRQ(ierr);
      break;
    case 2:
      ierr = VecRestoreArray3d(vec,stag->nGhost[1],stag->nGhost[0],stag->entriesPerElement,stag->startGhost[1],stag->startGhost[0],0,(PetscScalar****)array);CHKERRQ(ierr);
      break;
    case 3:
      ierr = VecRestoreArray4d(vec,stag->nGhost[2],stag->nGhost[1],stag->nGhost[0],stag->entriesPerElement,stag->startGhost[2],stag->startGhost[1],stag->startGhost[0],0,(PetscScalar*****)array);CHKERRQ(ierr);
      break;
    default: SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Unsupported dimension %D",dim);
  }
  PetscFunctionReturn(0);
}

/*@C
 DMStagVecGetArrayRead - get read-only access to a local array
 
 Logically Collective
 
 See the man page for DMStagVecGetArray() for more information.
 
 Input Parameters:
 + dm - the DMStag object
 - vec - the Vec object
 
 Output Parameters:
 . array - the read-only array
 
 Notes:
 DMStagVecRestoreArrayRead() must be called, once finished with the array
 
 Level: beginner
 
 .seealso: DMSTAG, DMStagVecGetArrayRead(), DMStagGetLocationSlot(), DMGetLocalVector(), DMCreateLocalVector(), DMGetGlobalVector(), DMCreateGlobalVector(), DMDAVecGetArrayRead(), DMDAVecGetArrayDOFRead()
 @*/
PetscErrorCode DMStagVecGetArrayRead(DM dm,Vec vec,void *array)
{
  PetscErrorCode  ierr;
  DM_Stag * const stag = (DM_Stag*)dm->data;
  PetscInt        dim;
  PetscInt        nLocal;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMSTAG);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,2);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  ierr = VecGetLocalSize(vec,&nLocal);CHKERRQ(ierr);
  if (nLocal != stag->entriesGhost) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Vector local size %D is not compatible with DMStag local size %D\n",nLocal,stag->entriesGhost);
  switch (dim) {
    case 1:
      ierr = VecGetArray2dRead(vec,stag->nGhost[0],stag->entriesPerElement,stag->startGhost[0],0,(PetscScalar***)array);CHKERRQ(ierr);
      break;
    case 2:
      ierr = VecGetArray3dRead(vec,stag->nGhost[1],stag->nGhost[0],stag->entriesPerElement,stag->startGhost[1],stag->startGhost[0],0,(PetscScalar****)array);CHKERRQ(ierr);
      break;
    case 3:
      ierr = VecGetArray4dRead(vec,stag->nGhost[2],stag->nGhost[1],stag->nGhost[0],stag->entriesPerElement,stag->startGhost[2],stag->startGhost[1],stag->startGhost[0],0,(PetscScalar*****)array);CHKERRQ(ierr);
      break;
    default: SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Unsupported dimension %D",dim);
  }
  PetscFunctionReturn(0);
}

/*@C
 DMStagVecRestoreArrayRead - restore read-only access to a raw array
 
 Logically Collective
 
 Input Parameters:
 + dm - the DMStag object
 - vec - the Vec object
 
 Output Parameters:
 . array - the read-only array
 
 Level: beginner
 
 .seealso: DMSTAG, DMStagVecGetArrayRead(), DMDAVecRestoreArrayRead(), DMDAVecRestoreArrayDOFRead()
 @*/
PetscErrorCode DMStagVecRestoreArrayRead(DM dm,Vec vec,void *array)
{
  PetscErrorCode  ierr;
  DM_Stag * const stag = (DM_Stag*)dm->data;
  PetscInt        dim;
  PetscInt        nLocal;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMSTAG);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,2);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  ierr = VecGetLocalSize(vec,&nLocal);CHKERRQ(ierr);
  if (nLocal != stag->entriesGhost) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Vector local size %D is not compatible with DMStag local size %D\n",nLocal,stag->entriesGhost);
  switch (dim) {
    case 1:
      ierr = VecRestoreArray2dRead(vec,stag->nGhost[0],stag->entriesPerElement,stag->startGhost[0],0,(PetscScalar***)array);CHKERRQ(ierr);
      break;
    case 2:
      ierr = VecRestoreArray3dRead(vec,stag->nGhost[1],stag->nGhost[0],stag->entriesPerElement,stag->startGhost[1],stag->startGhost[0],0,(PetscScalar****)array);CHKERRQ(ierr);
      break;
    case 3:
      ierr = VecRestoreArray4dRead(vec,stag->nGhost[2],stag->nGhost[1],stag->nGhost[0],stag->entriesPerElement,stag->startGhost[2],stag->startGhost[1],stag->startGhost[0],0,(PetscScalar*****)array);CHKERRQ(ierr);
      break;
    default: SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Unsupported dimension %D",dim);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMStagGetProductCoordinateArrays_Private(DM dm,void* arrX,void* arrY,void* arrZ,PetscBool read)
{
  PetscErrorCode ierr;
  PetscInt       dim,d,dofCheck[DMSTAG_MAX_STRATA],s;
  DM             dmCoord;
  void*          arr[DMSTAG_MAX_DIM];
  PetscBool      checkDof;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  if (dim > DMSTAG_MAX_DIM) SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Not implemented for %D dimensions",dim);
  arr[0] = arrX; arr[1] = arrY; arr[2] = arrZ;
  ierr = DMGetCoordinateDM(dm,&dmCoord);CHKERRQ(ierr);
  if (!dmCoord) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"DM does not have a coordinate DM");
  {
    PetscBool isProduct;
    DMType    dmType;
    ierr = DMGetType(dmCoord,&dmType);CHKERRQ(ierr);
    ierr = PetscStrcmp(DMPRODUCT,dmType,&isProduct);CHKERRQ(ierr);
    if (!isProduct) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Coordinate DM is not of type DMPRODUCT");
  }
  for (s=0; s<DMSTAG_MAX_STRATA; ++s) dofCheck[s] = 0;
  checkDof = PETSC_FALSE;
  for (d=0; d<dim; ++d) {
    DM        subDM;
    DMType    dmType;
    PetscBool isStag;
    PetscInt  dof[DMSTAG_MAX_STRATA],subDim;
    Vec       coord1d_local;
    
    /* Ignore unrequested arrays */
    if (!arr[d]) continue;
    
    ierr = DMProductGetDM(dmCoord,d,&subDM);CHKERRQ(ierr);
    if (!subDM) SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Coordinate DM is missing sub DM %D",d);
    ierr = DMGetDimension(subDM,&subDim);CHKERRQ(ierr);
    if (subDim != 1) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Coordinate sub-DM is not of dimension 1");
    ierr = DMGetType(subDM,&dmType);CHKERRQ(ierr);
    ierr = PetscStrcmp(DMSTAG,dmType,&isStag);CHKERRQ(ierr);
    if (!isStag) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Coordinate sub-DM is not of type DMSTAG");
    ierr = DMStagGetDOF(subDM,&dof[0],&dof[1],&dof[2],&dof[3]);CHKERRQ(ierr);
    if (!checkDof) {
      for (s=0; s<DMSTAG_MAX_STRATA; ++s) dofCheck[s] = dof[s];
      checkDof = PETSC_TRUE;
    } else {
      for (s=0; s<DMSTAG_MAX_STRATA; ++s) {
        if (dofCheck[s] != dof[s]) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Coordinate sub-DMs have different dofs");
      }
    }
    ierr = DMGetCoordinatesLocal(subDM,&coord1d_local);CHKERRQ(ierr);
    if (read) {
      ierr = DMStagVecGetArrayRead(subDM,coord1d_local,arr[d]);CHKERRQ(ierr);
    } else {
      ierr = DMStagVecGetArray(subDM,coord1d_local,arr[d]);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*@C
 DMStagGetProductCoordinateArraysRead - extract product coordinate arrays, read-only
 
 Logically Collective
 
 See the man page for DMStagGetProductCoordinateArrays() for more information.
 
 Input Parameter:
 . dm - the DMStag object
 
 Output Parameters:
 . arrX,arrY,arrZ - local 1D coordinate arrays
 
 Level: intermediate
 
 .seealso: DMSTAG, DMPRODUCT, DMStagGetProductCoordinateArrays(), DMStagSetUniformCoordinates(), DMStagSetUniformCoordinatesProduct(), DMStagGetProductCoordinateLocationSlot()
 @*/
PetscErrorCode DMStagGetProductCoordinateArraysRead(DM dm,void* arrX,void* arrY,void* arrZ)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = DMStagGetProductCoordinateArrays_Private(dm,arrX,arrY,arrZ,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMStagRestoreProductCoordinateArrays_Private(DM dm,void *arrX,void *arrY,void *arrZ,PetscBool read)
{
  PetscErrorCode  ierr;
  PetscInt        dim,d;
  void*           arr[DMSTAG_MAX_DIM];
  DM              dmCoord;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  if (dim > DMSTAG_MAX_DIM) SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Not implemented for %D dimensions",dim);
  arr[0] = arrX; arr[1] = arrY; arr[2] = arrZ;
  ierr = DMGetCoordinateDM(dm,&dmCoord);CHKERRQ(ierr);
  for (d=0; d<dim; ++d) {
    DM  subDM;
    Vec coord1d_local;
    
    /* Ignore unrequested arrays */
    if (!arr[d]) continue;
    
    ierr = DMProductGetDM(dmCoord,d,&subDM);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(subDM,&coord1d_local);CHKERRQ(ierr);
    if (read) {
      ierr = DMStagVecRestoreArrayRead(subDM,coord1d_local,arr[d]);CHKERRQ(ierr);
    } else {
      ierr = DMStagVecRestoreArray(subDM,coord1d_local,arr[d]);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*@C
 DMStagRestoreProductCoordinateArraysRead - restore local product array access, read-only
 
 Logically Collective
 
 Input Parameter:
 . dm - the DMStag object
 
 Output Parameters:
 . arrX,arrY,arrZ - local 1D coordinate arrays
 
 Level: intermediate
 
 .seealso: DMSTAG, DMStagGetProductCoordinateArrays(), DMStagGetProductCoordinateArraysRead()
 @*/
PetscErrorCode DMStagRestoreProductCoordinateArraysRead(DM dm,void *arrX,void *arrY,void *arrZ)
{
  PetscErrorCode  ierr;
  
  PetscFunctionBegin;
  ierr = DMStagRestoreProductCoordinateArrays_Private(dm,arrX,arrY,arrZ,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
 DMStagGetProductCoordinateLocationSlot - get slot for use with local product coordinate arrays
 
 Not Collective
 
 High-level helper function to get slot indices for 1D coordinate DMs,
 for use with DMStagGetProductCoordinateArrays() and related functions.
 
 Input Parameters:
 + dm - the DMStag object
 - loc - the grid location
 
 Output Parameter:
 . slot - the index to use in local arrays
 
 Notes:
 Checks that the coordinates are actually set up so that using the
 slots from the first 1d coordinate sub-DM is valid for all the 1D coordinate sub-DMs.
 
 Level: intermediate
 
 .seealso: DMSTAG, DMPRODUCT, DMStagGetProductCoordinateArrays(), DMStagGetProductCoordinateArraysRead(), DMStagSetUniformCoordinates()
 @*/
PetscErrorCode DMStagGetProductCoordinateLocationSlot(DM dm,DMStagStencilLocation loc,PetscInt *slot)
{
  PetscErrorCode ierr;
  DM             dmCoord;
  PetscInt       dim,dofCheck[DMSTAG_MAX_STRATA],s,d;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(dm,&dmCoord);CHKERRQ(ierr);
  if (!dmCoord) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"DM does not have a coordinate DM");
  {
    PetscBool isProduct;
    DMType    dmType;
    ierr = DMGetType(dmCoord,&dmType);CHKERRQ(ierr);
    ierr = PetscStrcmp(DMPRODUCT,dmType,&isProduct);CHKERRQ(ierr);
    if (!isProduct) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Coordinate DM is not of type DMPRODUCT");
  }
  for (s=0; s<DMSTAG_MAX_STRATA; ++s) dofCheck[s] = 0;
  for (d=0; d<dim; ++d) {
    DM        subDM;
    DMType    dmType;
    PetscBool isStag;
    PetscInt  dof[DMSTAG_MAX_STRATA],subDim;
    ierr = DMProductGetDM(dmCoord,d,&subDM);CHKERRQ(ierr);
    if (!subDM) SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Coordinate DM is missing sub DM %D",d);
    ierr = DMGetDimension(subDM,&subDim);CHKERRQ(ierr);
    if (subDim != 1) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Coordinate sub-DM is not of dimension 1");
    ierr = DMGetType(subDM,&dmType);CHKERRQ(ierr);
    ierr = PetscStrcmp(DMSTAG,dmType,&isStag);CHKERRQ(ierr);
    if (!isStag) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Coordinate sub-DM is not of type DMSTAG");
    ierr = DMStagGetDOF(subDM,&dof[0],&dof[1],&dof[2],&dof[3]);CHKERRQ(ierr);
    if (d == 0) {
      const PetscInt component = 0;
      for (s=0; s<DMSTAG_MAX_STRATA; ++s) dofCheck[s] = dof[s];
      ierr = DMStagGetLocationSlot(subDM,loc,component,slot);CHKERRQ(ierr);
    } else {
      for (s=0; s<DMSTAG_MAX_STRATA; ++s) {
        if (dofCheck[s] != dof[s]) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Coordinate sub-DMs have different dofs");
      }
    }
  }
  PetscFunctionReturn(0);
}

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
  PetscErrorCode        ierr;
  const DM_Stag * const stag  = (DM_Stag*)dm->data;
  PetscInt              dim,iLocal,jLocal;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMSTAG);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
#ifdef PETSC_USE_DEBUG
  {
    PetscInt d,startGhost[DMSTAG_MAX_DIM],nGhost[DMSTAG_MAX_DIM];
    ierr = DMStagGetGhostCorners(dm,&startGhost[0],&startGhost[1],&startGhost[2],&nGhost[0],&nGhost[1],&nGhost[2]);CHKERRQ(ierr);
    for (d=0; d<dim; ++d) {
      if (ind[d] < startGhost[d] || ind[d] >= startGhost[d]+nGhost[d]) {
        SETERRQ4(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"element index ind[%D] is out of range. It should be in [%D,%D) but it is %D",d,startGhost[d],startGhost[d]+nGhost[d],ind[d]);
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
    default: SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Unsupported dimension %D",dim);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMStagStencilLocationCanonicalize(DMStagStencilLocation loc,DMStagStencilLocation *locCanonical)
{
  PetscFunctionBegin;
  switch (loc) {
    case DMSTAG_ELEMENT:
      *locCanonical = DMSTAG_ELEMENT;
      break;
    case DMSTAG_LEFT:
    case DMSTAG_RIGHT:
      *locCanonical = DMSTAG_LEFT;
      break;
    case DMSTAG_DOWN:
    case DMSTAG_UP:
      *locCanonical = DMSTAG_DOWN;
      break;
    case DMSTAG_BACK:
    case DMSTAG_FRONT:
      *locCanonical = DMSTAG_BACK;
      break;
    case DMSTAG_DOWN_LEFT :
    case DMSTAG_DOWN_RIGHT :
    case DMSTAG_UP_LEFT :
    case DMSTAG_UP_RIGHT :
      *locCanonical = DMSTAG_DOWN_LEFT;
      break;
    case DMSTAG_BACK_LEFT:
    case DMSTAG_BACK_RIGHT:
    case DMSTAG_FRONT_LEFT:
    case DMSTAG_FRONT_RIGHT:
      *locCanonical = DMSTAG_BACK_LEFT;
      break;
    case DMSTAG_BACK_DOWN:
    case DMSTAG_BACK_UP:
    case DMSTAG_FRONT_DOWN:
    case DMSTAG_FRONT_UP:
      *locCanonical = DMSTAG_BACK_DOWN;
      break;
    case DMSTAG_BACK_DOWN_LEFT:
    case DMSTAG_BACK_DOWN_RIGHT:
    case DMSTAG_BACK_UP_LEFT:
    case DMSTAG_BACK_UP_RIGHT:
    case DMSTAG_FRONT_DOWN_LEFT:
    case DMSTAG_FRONT_DOWN_RIGHT:
    case DMSTAG_FRONT_UP_LEFT:
    case DMSTAG_FRONT_UP_RIGHT:
      *locCanonical = DMSTAG_BACK_DOWN_LEFT;
      break;
    default :
      *locCanonical = DMSTAG_NULL_LOCATION;
      break;
  }
  PetscFunctionReturn(0);
}

/* Convert an array of DMStagStencil objects to an array of indices into a local vector.
 The .c fields in pos must always be set (even if to 0).  */
PetscErrorCode DMStagStencilToIndexLocal(DM dm,PetscInt n,const DMStagStencil *pos,PetscInt *ix)
{
  PetscErrorCode        ierr;
  const DM_Stag * const stag = (DM_Stag*)dm->data;
  PetscInt              idx,dim,startGhost[DMSTAG_MAX_DIM];
  const PetscInt        epe = stag->entriesPerElement;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMSTAG);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
  {
    PetscInt i,nGhost[DMSTAG_MAX_DIM],endGhost[DMSTAG_MAX_DIM];
    ierr = DMStagGetGhostCorners(dm,&startGhost[0],&startGhost[1],&startGhost[2],&nGhost[0],&nGhost[1],&nGhost[2]);CHKERRQ(ierr);
    for (i=0; i<DMSTAG_MAX_DIM; ++i) endGhost[i] = startGhost[i] + nGhost[i];
    for (i=0; i<n; ++i) {
      PetscInt dof;
      ierr = DMStagGetLocationDOF(dm,pos[i].loc,&dof);CHKERRQ(ierr);
      if (dof < 1) SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Location %s has no dof attached",DMStagStencilLocations[pos[i].loc]);
      if (pos[i].c < 0) SETERRQ2(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Negative component number (%d) supplied in loc[%D]",pos[i].c,i);
      if (pos[i].c > dof-1) SETERRQ3(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Supplied component number (%D) for location %s is too big (maximum %D)",pos[i].c,DMStagStencilLocations[pos[i].loc],dof-1);
      if (            pos[i].i >= endGhost[0] || pos[i].i < startGhost[0] ) SETERRQ3(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Supplied x element index %D out of range. Should be in [%D,%D]",pos[i].i,startGhost[0],endGhost[0]-1);
      if (dim > 1 && (pos[i].j >= endGhost[1] || pos[i].j < startGhost[1])) SETERRQ3(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Supplied y element index %D out of range. Should be in [%D,%D]",pos[i].j,startGhost[1],endGhost[1]-1);
      if (dim > 2 && (pos[i].k >= endGhost[2] || pos[i].k < startGhost[2])) SETERRQ3(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Supplied z element index %D out of range. Should be in [%D,%D]",pos[i].k,startGhost[2],endGhost[2]-1);
    }
  }
#else
  ierr = DMStagGetGhostCorners(dm,&startGhost[0],&startGhost[1],&startGhost[2],NULL,NULL,NULL);CHKERRQ(ierr);
#endif
  if (dim == 1) {
    for (idx=0; idx<n; ++idx) {
      const PetscInt eLocal = pos[idx].i - startGhost[0]; /* Local element number */
      ix[idx] = eLocal * epe + stag->locationOffsets[pos[idx].loc] + pos[idx].c;
    }
  } else if (dim == 2) {
    const PetscInt epr = stag->nGhost[0];
    ierr = DMStagGetGhostCorners(dm,&startGhost[0],&startGhost[1],NULL,NULL,NULL,NULL);CHKERRQ(ierr);
    for (idx=0; idx<n; ++idx) {
      const PetscInt eLocalx = pos[idx].i - startGhost[0];
      const PetscInt eLocaly = pos[idx].j - startGhost[1];
      const PetscInt eLocal = eLocalx + epr*eLocaly;
      ix[idx] = eLocal * epe + stag->locationOffsets[pos[idx].loc] + pos[idx].c;
    }
  } else if (dim == 3) {
    const PetscInt epr = stag->nGhost[0];
    const PetscInt epl = stag->nGhost[0]*stag->nGhost[1];
    ierr = DMStagGetGhostCorners(dm,&startGhost[0],&startGhost[1],&startGhost[2],NULL,NULL,NULL);CHKERRQ(ierr);
    for (idx=0; idx<n; ++idx) {
      const PetscInt eLocalx = pos[idx].i - startGhost[0];
      const PetscInt eLocaly = pos[idx].j - startGhost[1];
      const PetscInt eLocalz = pos[idx].k - startGhost[2];
      const PetscInt eLocal  = epl*eLocalz + epr*eLocaly + eLocalx;
      ix[idx] = eLocal * epe + stag->locationOffsets[pos[idx].loc] + pos[idx].c;
    }
  } else SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Unsupported dimension %d",dim);
  PetscFunctionReturn(0);
}

/*@C
 DMStagCreateISFromStencils - Create an IS, using global numberings, for a subset of DOF in a DMStag object
 
 Collective
 
 Input Parameters:
 + dm - the DMStag object
 . nStencil - the number of stencils provided
 - stencils - an array of DMStagStencil objects (i,j, and k are ignored)
 
 Output Parameter:
 . is - the global IS
 
 Note:
 Redundant entries in s are ignored
 
 Level: advanced
 
 .seealso: DMSTAG, IS, DMStagStencil, DMCreateGlobalVector
 @*/
PetscErrorCode DMStagCreateISFromStencils(DM dm,PetscInt nStencil,DMStagStencil* stencils,IS *is)
{
  PetscErrorCode         ierr;
  DMStagStencil          *ss;
  PetscInt               *idx,*idxLocal;
  const PetscInt         *ltogidx;
  PetscInt               p,p2,pmax,i,j,k,d,dim,count,nidx;
  ISLocalToGlobalMapping ltog;
  PetscInt               start[DMSTAG_MAX_DIM],n[DMSTAG_MAX_DIM],extraPoint[DMSTAG_MAX_DIM];
  
  PetscFunctionBegin;
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  if (dim<1 || dim>3) SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Unsupported dimension %D",dim);
  
  /* Only use non-redundant stencils */
  ierr = PetscMalloc1(nStencil,&ss);CHKERRQ(ierr);
  pmax = 0;
  for (p=0; p<nStencil; ++p) {
    PetscBool skip = PETSC_FALSE;
    DMStagStencil stencilPotential = stencils[p];
    ierr = DMStagStencilLocationCanonicalize(stencils[p].loc,&stencilPotential.loc);CHKERRQ(ierr);
    for (p2=0; p2<pmax; ++p2) { /* Quadratic complexity algorithm in nStencil */
      if (stencilPotential.loc == ss[p2].loc && stencilPotential.c == ss[p2].c) {
        skip = PETSC_TRUE;
        break;
      }
    }
    if (!skip) {
      ss[pmax] = stencilPotential;
      ++pmax;
    }
  }
  
  ierr = PetscMalloc1(pmax,&idxLocal);CHKERRQ(ierr);
  ierr = DMGetLocalToGlobalMapping(dm,&ltog);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetIndices(ltog,&ltogidx);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dm,&start[0],&start[1],&start[2],&n[0],&n[1],&n[2],&extraPoint[0],&extraPoint[1],&extraPoint[2]);CHKERRQ(ierr);
  for (d=dim; d<DMSTAG_MAX_DIM; ++d) {
    start[d]      = 0;
    n[d]          = 1; /* To allow for a single loop nest below */
    extraPoint[d] = 0;
  }
  nidx = pmax; for (d=0; d<dim; ++d) nidx *= (n[d]+1); /* Overestimate (always assumes extraPoint) */
  ierr = PetscMalloc1(nidx,&idx);CHKERRQ(ierr);
  count = 0;
  /* Note that unused loop variables are not accessed, for lower dimensions */
  for (k=start[2]; k<start[2]+n[2]+extraPoint[2]; ++k) {
    for (j=start[1]; j<start[1]+n[1]+extraPoint[1]; ++j) {
      for (i=start[0]; i<start[0]+n[0]+extraPoint[0]; ++i) {
        for(p=0; p<pmax; ++p) {
          ss[p].i = i; ss[p].j = j; ss[p].k = k;
        }
        ierr = DMStagStencilToIndexLocal(dm,pmax,ss,idxLocal);CHKERRQ(ierr);
        for(p=0; p<pmax; ++p) {
          const PetscInt gidx = ltogidx[idxLocal[p]];
          if (gidx >= 0) {
            idx[count] = gidx;
            ++count;
          }
        }
      }
    }
  }
  ierr = ISLocalToGlobalMappingRestoreIndices(ltog,&ltogidx);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)dm),count,idx,PETSC_OWN_POINTER,is);CHKERRQ(ierr);
  
  ierr = PetscFree(ss);CHKERRQ(ierr);
  ierr = PetscFree(idxLocal);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/* Note: this does a linear search, which should be bisection, but an even
 better strategy would be to use any information about which cell a particle
 is already in, as in most applications it would be found there or in a
 neighboring cell */
static PetscErrorCode DMStagLocatePointsIS_2D_Product_Private(DM dm,Vec pos,IS *iscell)
{
  PetscErrorCode    ierr;
  PetscInt          localSize,bs,p,npoints,start[2],n[2];
  PetscInt          *cellidx;
  const PetscScalar *_coor;
  PetscScalar       **cArrX,**cArrY;
  PetscInt          iNext,iPrev;
  
  PetscFunctionBegin;
  
  ierr = VecGetLocalSize(pos,&localSize);CHKERRQ(ierr);
  ierr = VecGetBlockSize(pos,&bs);CHKERRQ(ierr);
  npoints = localSize/bs;
  
  ierr = PetscMalloc1(npoints,&cellidx);CHKERRQ(ierr);
  ierr = VecGetArrayRead(pos,&_coor);CHKERRQ(ierr);
  
  ierr = DMStagGetCorners(dm,&start[0],&start[1],NULL,&n[0],&n[1],NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateArraysRead(dm,&cArrX,&cArrY,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,DMSTAG_LEFT,&iPrev);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,DMSTAG_RIGHT,&iNext);CHKERRQ(ierr);
  
  for (p=0; p<npoints; p++) {
    PetscReal coor_p[2];
    
    coor_p[0] = PetscRealPart(_coor[2*p]);
    coor_p[1] = PetscRealPart(_coor[2*p+1]);
    
    if ((coor_p[0] >= cArrX[start[0]][iPrev]) && (coor_p[0] <= cArrX[start[0]+n[0]][iPrev]) && (coor_p[1] >= cArrY[start[1]][iPrev]) && (coor_p[1] <= cArrY[start[1]+n[1]][iPrev]))
    {
      PetscInt e,ind[2];
      for (ind[0]=start[0]; ind[0]<start[0]+n[0]; ++ind[0]) {
        if (coor_p[0] <= cArrX[ind[0]][iNext]) break;
      }
      for (ind[1]=start[1]; ind[1]<start[1]+n[1]; ++ind[1]) {
        if (coor_p[1] <= cArrY[ind[1]][iNext]) break;
      }
      ierr = DMStagGetLocalElementIndex(dm,ind,&e);CHKERRQ(ierr);
      cellidx[p] = e;
    } else {
      cellidx[p] = DMLOCATEPOINT_POINT_NOT_FOUND;
    }
  }
  ierr = DMStagRestoreProductCoordinateArraysRead(dm,&cArrX,&cArrY,NULL);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(pos,&_coor);CHKERRQ(ierr);
  
  ierr = ISCreateGeneral(PETSC_COMM_SELF,npoints,cellidx,PETSC_OWN_POINTER,iscell);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMStagLocatePointsIS_2D_Product_ConstantSpacing_Private(DM dm,Vec pos,IS *iscell)
{
  PetscErrorCode    ierr;
  PetscInt          localSize,bs,p,npoints,start[2],n[2];
  PetscInt          *cellidx;
  const PetscScalar *_coor;
  PetscScalar       **cArrX,**cArrY;
  PetscInt          iNext,iPrev,Ng[]={0,0,0};
  PetscReal         dx[2],gmin[]={0,0,0},gmax[]={0,0,0};
  
  PetscFunctionBegin;
  
  ierr = VecGetLocalSize(pos,&localSize);CHKERRQ(ierr);
  ierr = VecGetBlockSize(pos,&bs);CHKERRQ(ierr);
  npoints = localSize/bs;
  
  ierr = PetscMalloc1(npoints,&cellidx);CHKERRQ(ierr);
  ierr = VecGetArrayRead(pos,&_coor);CHKERRQ(ierr);
  
  ierr = DMStagGetCorners(dm,&start[0],&start[1],NULL,&n[0],&n[1],NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateArraysRead(dm,&cArrX,&cArrY,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,DMSTAG_LEFT,&iPrev);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,DMSTAG_RIGHT,&iNext);CHKERRQ(ierr);
  
  ierr = DMStagGetGlobalSizes(dm,&Ng[0],&Ng[1],NULL);CHKERRQ(ierr);
  ierr = DMStagGetBoundingBox(dm,gmin,gmax);CHKERRQ(ierr);
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
    
    ierr = DMStagGetLocalElementIndex(dm,gei,&elocal);CHKERRQ(ierr);
    cellidx[p] = elocal;
  }
  ierr = DMStagRestoreProductCoordinateArraysRead(dm,&cArrX,&cArrY,NULL);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(pos,&_coor);CHKERRQ(ierr);
  
  ierr = ISCreateGeneral(PETSC_COMM_SELF,npoints,cellidx,PETSC_OWN_POINTER,iscell);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMLocatePoints_Stag(DM dm,Vec pos,DMPointLocationType ltype,PetscSF cellSF)
{
  PetscErrorCode ierr;
  IS             iscell;
  PetscSFNode    *cells;
  PetscInt       p,bs,dim,npoints,nfound;
  const PetscInt *boxCells;
  
  PetscFunctionBegin;
  if (ltype != DM_POINTLOCATION_NONE) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Only DMPOINTLOCATION_NONE is supported for ltype");
  {
    DM        cdm;
    PetscBool isProduct;
    
    ierr = DMGetCoordinateDM(dm,&cdm);CHKERRQ(ierr);
    if (!cdm) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Cannot locate points without coordinates");
    ierr = PetscObjectTypeCompare((PetscObject)cdm,DMPRODUCT,&isProduct);CHKERRQ(ierr);
    if (!isProduct) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Point location only supported for product coordinates");
  }
  ierr = VecGetBlockSize(pos,&dim);CHKERRQ(ierr);
  switch (dim) {
    case 2:
      //ierr = DMStagLocatePointsIS_2D_Product_Private(dm,pos,&iscell);CHKERRQ(ierr);
      ierr = DMStagLocatePointsIS_2D_Product_ConstantSpacing_Private(dm,pos,&iscell);CHKERRQ(ierr);
      break;
    default: SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Unsupported spatial dimension %D",dim);
  }
  
  ierr = VecGetLocalSize(pos,&npoints);CHKERRQ(ierr);
  ierr = VecGetBlockSize(pos,&bs);CHKERRQ(ierr);
  npoints = npoints / bs;
  
  ierr = PetscMalloc1(npoints, &cells);CHKERRQ(ierr);
  ierr = ISGetIndices(iscell, &boxCells);CHKERRQ(ierr);
  
  for (p=0; p<npoints; p++) {
    cells[p].rank  = 0;
    cells[p].index = boxCells[p];
  }
  ierr = ISRestoreIndices(iscell, &boxCells);CHKERRQ(ierr);
  
  nfound = npoints;
  ierr = PetscSFSetGraph(cellSF, npoints, nfound, NULL, PETSC_OWN_POINTER, cells, PETSC_OWN_POINTER);CHKERRQ(ierr);
  ierr = ISDestroy(&iscell);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode DMSetPointLocation(DM dm,PetscErrorCode (*f)(DM,Vec,DMPointLocationType,PetscSF))
{
  PetscFunctionBegin;
  dm->ops->locatepoints = f;
  PetscFunctionReturn(0);
}

PetscErrorCode DMStagGetLocalBoundingBox_2d(DM dm,PetscReal lmin[],PetscReal lmax[])
{
  PetscErrorCode    ierr;
  PetscInt          i,start[2],n[2],ind;
  const PetscScalar *_coor;
  PetscScalar       **cArrX,**cArrY;
  PetscInt          iNext,iPrev;
  PetscReal         min[3] = {PETSC_MAX_REAL, PETSC_MAX_REAL, PETSC_MAX_REAL};
  PetscReal         max[3] = {PETSC_MIN_REAL, PETSC_MIN_REAL, PETSC_MIN_REAL};

  PetscFunctionBegin;
  ierr = DMStagGetCorners(dm,&start[0],&start[1],NULL,&n[0],&n[1],NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateArraysRead(dm,&cArrX,&cArrY,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,DMSTAG_LEFT,&iPrev);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,DMSTAG_RIGHT,&iNext);CHKERRQ(ierr);

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
  
  ierr = DMStagRestoreProductCoordinateArraysRead(dm,&cArrX,&cArrY,NULL);CHKERRQ(ierr);

  if (lmin) {PetscArraycpy(lmin, min, 2);}
  if (lmax) {PetscArraycpy(lmax, max, 2);}
  PetscFunctionReturn(0);
}

PetscErrorCode DMStagGetBoundingBox(DM dm,PetscReal gmin[],PetscReal gmax[])
{
  PetscReal      lmin[]={0,0,0},lmax[]={0,0,0};
  PetscInt       cdim=2;
  PetscMPIInt    count;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = PetscMPIIntCast(cdim,&count);CHKERRQ(ierr);
  ierr = DMStagGetLocalBoundingBox_2d(dm,lmin,lmax);CHKERRQ(ierr);
  if (gmin) {MPIU_Allreduce(lmin,gmin,count,MPIU_REAL,MPIU_MIN,PetscObjectComm((PetscObject)dm));}
  if (gmax) {MPIU_Allreduce(lmax,gmax,count,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)dm));}
  PetscFunctionReturn(0);
}

/*
 The less efficient alternative would be to call
 
 ierr = DMStagGetLocalElementGlobalIndices(elocal,gidx[]);CHKERRQ(ierr);
 ierr = DMStagGetCorners(dm,&start[0],&start[1],NULL,&n[0],&n[1],NULL,NULL,NULL,NULL);CHKERRQ(ierr);
 and then do
 if (gidx[0] >= start[0] && gidx[0] < start[0]+n[0])
   if (gidx[1] >= start[1] && gidx[1] < start[1]+n[1]) *value = PETSC_TRUE;

*/
/*
PetscErrorCode DMStagLocalElementIndexInGlobalSpace_2d(DM dm,PetscInt elocal,PetscBool *value)
{
  const DM_Stag * const stag = (DM_Stag*)dm->data;
  PetscInt m,n,ei,ej;
  
  PetscFunctionBegin;
  m = stag->nGhost[0];
  n = stag->nGhost[1];
  ej = elocal / m;
  ei = elocal - ej *m;
  *value = PETSC_FALSE;
  if ( (ei < stag->n[0]) && (ej < stag->n[1]) ) { *value = PETSC_TRUE; }
  PetscFunctionReturn(0);
}
*/


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
  si = stag->start[0]; if (geid[0] < si) { *value = PETSC_FALSE; PetscFunctionReturn(0); }
  sj = stag->start[1]; if (geid[1] < sj) { *value = PETSC_FALSE; PetscFunctionReturn(0); }
  
  ei = si + stag->n[0]; if (geid[0] >= ei) { *value = PETSC_FALSE; PetscFunctionReturn(0); }
  ej = sj + stag->n[1]; if (geid[1] >= ej) { *value = PETSC_FALSE; PetscFunctionReturn(0); }
  
  PetscFunctionReturn(0);
}

PetscErrorCode DMStagFieldISCreate_2d(DM dm,
                                    PetscInt ndof0A,PetscInt dof0A[],
                                    PetscInt ndof1A,PetscInt dof1A[],
                                    PetscInt ndof2A,PetscInt dof2A[],IS *is)
{
  PetscInt f0,f1,f2,dim,dof0,dof1,dof2,sumDof,d,scnt;
  DMStagStencil *stencil_list;
  PetscErrorCode ierr;
  
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  ierr = DMStagGetDOF(dm,&dof0,&dof1,&dof2,NULL);CHKERRQ(ierr);

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
  ierr = PetscCalloc1(sumDof,&stencil_list);CHKERRQ(ierr);
  
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

  ierr = DMStagCreateISFromStencils(dm,sumDof,stencil_list,is);CHKERRQ(ierr);CHKERRQ(ierr);

  ierr = PetscFree(stencil_list);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode DMStagISCreateL2L_2d(DM dmA,
                                    PetscInt n0A,PetscInt dof0A[],
                                    PetscInt n1A,PetscInt dof1A[],
                                    PetscInt n2A,PetscInt dof2A[],IS *isA,
                                    DM dmB,
                                    PetscInt dof0B[],
                                    PetscInt dof1B[],
                                    PetscInt dof2B[],IS *isB)
{
  PetscErrorCode ierr;
  ierr = DMStagFieldISCreate_2d(dmA,n0A,dof0A,n1A,dof1A,n2A,dof2A,isA);CHKERRQ(ierr);
  ierr = DMStagFieldISCreate_2d(dmB,n0A,dof0B,n1A,dof1B,n2A,dof2B,isB);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
