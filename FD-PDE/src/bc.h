/* <BC> contains generalized boundary conditions routines for FD-PDE */

#ifndef BC_H
#define BC_H

#include "petsc.h"

typedef struct _p_BCList BCList;

// BC type
enum BCType { BC_UNINIT, DIRICHLET, NEUMANN, ROBIN };

// ---------------------------------------
// Struct definitions
// ---------------------------------------
// BC struct 
struct _p_BCList {
  DMStagStencil point;
  PetscInt      idx;
  enum BCType   type;
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
//PetscErrorCode FDBCListDestroy(BCList*);
PetscErrorCode FDBCGetEntry(DM,PetscScalar**,PetscScalar**,DMStagStencilLocation, PetscInt, PetscInt, PetscInt, BCList*);
// PetscErrorCode FDBCApplyStokes(DM,Vec,BCList*,PetscInt, PetscScalar**,PetscScalar**,PetscScalar*, PetscScalar*,PetscInt[], PetscScalar***);

// UTILS
PetscErrorCode DMStagExtract1DComponent(DM, Vec, DMStagStencilLocation, PetscInt, PetscScalar, PetscScalar*);
PetscErrorCode GetCoordinatesStencil(DM, Vec, PetscInt, DMStagStencil[], PetscScalar[], PetscScalar[]);

#endif