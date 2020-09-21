/*
 Many of these functions are all taken from
 [url] https://bitbucket.org/psanan/petsc
 [branch] psanan/stagbl-working-base
 They will all be added to the next PETSc release (v3.13).
 When switching to v3.13, this file should be purged.
*/


#ifndef __dmstag_utils__
#define __dmstag_utils__

#include <petsc.h>
#include <petscvec.h>
#include <petscdm.h>

/* stolen from PETSc */
PetscErrorCode DMStagGetLocalElementGlobalIndices(DM dm,PetscInt eidx,PetscInt *ind);
PetscErrorCode DMStagVecGetArray(DM dm,Vec vec,void *array);
PetscErrorCode DMStagVecRestoreArray(DM dm,Vec vec,void *array);
PetscErrorCode DMStagVecGetArrayRead(DM dm,Vec vec,void *array);
PetscErrorCode DMStagVecRestoreArrayRead(DM dm,Vec vec,void *array);
PetscErrorCode DMStagGetProductCoordinateArraysRead(DM dm,void* arrX,void* arrY,void* arrZ);
PetscErrorCode DMStagRestoreProductCoordinateArraysRead(DM dm,void *arrX,void *arrY,void *arrZ);
PetscErrorCode DMStagGetProductCoordinateLocationSlot(DM dm,DMStagStencilLocation loc,PetscInt *slot);
PetscErrorCode DMStagGetLocalElementIndex(DM dm,PetscInt *ind,PetscInt *eidx);
PetscErrorCode DMLocatePoints_Stag(DM dm,Vec pos,DMPointLocationType ltype,PetscSF cellSF);
PetscErrorCode DMStagCreateISFromStencils(DM dm,PetscInt nStencil,DMStagStencil* stencils,IS *is);

/* new */
PetscErrorCode DMSetPointLocation(DM dm,PetscErrorCode (*f)(DM,Vec,DMPointLocationType,PetscSF));
PetscErrorCode DMStagGetBoundingBox(DM dm,PetscReal gmin[],PetscReal gmax[]);
PetscErrorCode DMStagLocalElementIndexInGlobalSpace_2d(DM dm,PetscInt elocal,PetscBool *value);

PetscErrorCode DMStagFieldISCreate_2d(DM dm,
                                      PetscInt ndof0A,PetscInt dof0A[],
                                      PetscInt ndof1A,PetscInt dof1A[],
                                      PetscInt ndof2A,PetscInt dof2A[],IS *is);

PetscErrorCode DMStagISCreateL2L_2d(DM dmA,
                                    PetscInt n0A,PetscInt dof0A[],
                                    PetscInt n1A,PetscInt dof1A[],
                                    PetscInt n2A,PetscInt dof2A[],IS *isA,
                                    DM dmB,
                                    PetscInt dof0B[],
                                    PetscInt dof1B[],
                                    PetscInt dof2B[],IS *isB);

PetscErrorCode DMStagCellSize_2d(DM dm,PetscInt nx, PetscInt ny, PetscScalar *_dx[],PetscScalar *_dy[]);
#endif
