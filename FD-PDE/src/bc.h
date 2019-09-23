/* <BC> contains generalized boundary conditions routines for FD-PDE */

#ifndef BC_H
#define BC_H

#include "petsc.h"
#include "petscvec.h"
#include "petscdm.h"

typedef struct _p_BCList BCList;

// BC type
typedef enum { BC_UNINIT, BC_DIRICHLET, BC_NEUMANN, BC_ROBIN } BCType;

// ---------------------------------------
// Struct definitions
// ---------------------------------------
// BC struct 
struct _p_BCList {
  DMStagStencil point;
  PetscInt      idx;
  BCType        type;
  PetscScalar   val;
  PetscScalar   coord[2];
  //PetscInt  i, j, c, idx;
  //DMStagStencilLocation loc;
  //PetscInt  face_id;
};

// ---------------------------------------
// Function definitions
// ---------------------------------------
PetscErrorCode FDBCListCreate(DM, BCList**, PetscInt*);
PetscErrorCode FDBCListDestroy(BCList**);
PetscErrorCode FDBCGetEntry(DM,PetscScalar**,PetscScalar**,DMStagStencilLocation, PetscInt, PetscInt, PetscInt, BCList*);

// UTILS
PetscErrorCode DMStagExtract1DComponent(DM, Vec, DMStagStencilLocation, PetscInt, PetscScalar, PetscScalar*);
PetscErrorCode GetCoordinatesStencil(DM, Vec, PetscInt, DMStagStencil[], PetscScalar[], PetscScalar[]);

#endif
