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
};

// ---------------------------------------
// Function definitions
// ---------------------------------------
PetscErrorCode FDBCListCreate(DM, BCList**, PetscInt*);
PetscErrorCode FDBCListDestroy(BCList**);
PetscErrorCode FDBCGetEntry(DM,PetscScalar**,PetscScalar**,DMStagStencilLocation, PetscInt, PetscInt, PetscInt, BCList*);

#endif