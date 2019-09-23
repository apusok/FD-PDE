/* <BC> contains generalized boundary conditions routines for FD-PDE */

#ifndef BC_H
#define BC_H

#include "petsc.h"
#include "petscvec.h"
#include "petscdm.h"
#include "petscdmstag.h"
#include "utils.h"

// BC type
typedef enum { BC_UNINIT, BC_DIRICHLET, BC_NEUMANN, BC_ROBIN } BCType;

// ---------------------------------------
// Struct definitions
// ---------------------------------------
// BC struct 
typedef struct {
  DMStagStencil point;
  PetscInt      idx;
  BCType        type;
  PetscScalar   val;
  PetscScalar   coord[2];
} DMStagBC;

// ---------------------------------------
// Function definitions
// ---------------------------------------
PetscErrorCode DMStagBCCreateDefault(DM, DMStagBC**, PetscInt*);
PetscErrorCode DMStagBCDestroy(DMStagBC**);
PetscErrorCode FDBCGetEntry(DM,PetscScalar**,PetscScalar**,DMStagStencilLocation, PetscInt, PetscInt, PetscInt, DMStagBC*);

#endif
