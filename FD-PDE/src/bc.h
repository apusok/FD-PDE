/* <BC> contains generalized boundary conditions routines for FD-PDE */

#ifndef BC_H
#define BC_H

#include "petsc.h"
#include "petscvec.h"
#include "petscdm.h"
#include "petscdmstag.h"
#include "utils.h"

// BC type
typedef enum { BC_NULL = 0, BC_DIRICHLET, BC_NEUMANN, BC_ROBIN } BCType;


// ---------------------------------------
// Struct definitions
// ---------------------------------------
// BC struct 
typedef struct {
  DMStagStencil    point;
  PetscInt         idx;
  BCType           type;
  PetscScalar      val;
  PetscScalar      coord[2];
} DMStagBC;

typedef struct _p_DMStagBCList *DMStagBCList;

struct _p_DMStagBCList {
  PetscInt  nbc_vertex,nbc_face,nbc_element;
  PetscInt  nbc;
  DMStagBC  *bc_v,*bc_f,*bc_e;
  DM        dm;
  PetscErrorCode (*evaluate)(DMStagBCList,Vec,void*);
  void           *context;
};


// ---------------------------------------
// Function definitions
// ---------------------------------------
PetscErrorCode DMStagBCCreateDefault(DM, DMStagBC**, PetscInt*);
PetscErrorCode DMStagBCDestroy(DMStagBC**);

PetscErrorCode DMStagBCListCreate(DM,DMStagBCList*);
PetscErrorCode DMStagBCListDestroy(DMStagBCList*);
PetscErrorCode DMStagBCListView(DMStagBCList);
PetscErrorCode DMStagBCListSetupCoordinates(DMStagBCList);
PetscErrorCode DMStagBCListGetVertexBCs(DMStagBCList,PetscInt*,DMStagBC**);
PetscErrorCode DMStagBCListGetFaceBCs(DMStagBCList,PetscInt*,DMStagBC**);
PetscErrorCode DMStagBCListGetElementBCs(DMStagBCList,PetscInt*,DMStagBC**);

PetscErrorCode DMStagBCListGetValues(DMStagBCList,const char,const char,PetscInt,PetscInt*,PetscInt**,PetscScalar**,PetscScalar**,BCType**);
PetscErrorCode DMStagBCListInsertValues(DMStagBCList,const char,PetscInt,PetscInt*,PetscInt**,PetscScalar**,PetscScalar**,BCType**);

/*
PetscErrorCode DMStagBCListTraverse(
                  DMStagBCList,PetscInt,DMStagBCLocation,
                  PetscErrorCode (*f)(const DMStagStencil*,
                                      const PetscScalar*,
                                      PetscBool*,
                                      PetscScalar*,void*),
                  void*);
*/

#endif
