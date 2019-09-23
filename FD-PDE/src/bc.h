/* <BC> contains generalized boundary conditions routines for FD-PDE */

#ifndef BC_H
#define BC_H

#include "petsc.h"
#include "petscvec.h"
#include "petscdm.h"
#include "petscdmstag.h"
#include "utils.h"

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
};

// ---------------------------------------
// Function definitions
// ---------------------------------------
PetscErrorCode FDBCListCreate(DM, BCList**, PetscInt*);
PetscErrorCode FDBCListDestroy(BCList**);
PetscErrorCode FDBCGetEntry(DM,PetscScalar**,PetscScalar**,DMStagStencilLocation, PetscInt, PetscInt, PetscInt, BCList*);

#endif
