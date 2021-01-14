/* <DMSTAGBCLIST> contains generalized boundary conditions routines for FD-PDE and DMStag*/

#ifndef DMSTAGBCLIST_H
#define DMSTAGBCLIST_H

#include "petsc.h"
#include "dmstag_utils.h"

// BC type
typedef enum { BC_NULL = 0, BC_DIRICHLET, BC_DIRICHLET_STAG, BC_NEUMANN, BC_NEUMANN_T } BCType;

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
  PetscInt         nbc_vertex,nbc_face,nbc_element;
  PetscInt         nbc;
  DMStagBC        *bc_v,*bc_f,*bc_e;
  DM               dm;
  PetscErrorCode (*evaluate)(DM,Vec,DMStagBCList,void*);
  void            *data;
};

// ---------------------------------------
// Function definitions
// ---------------------------------------
PetscErrorCode DMStagBCListCreate(DM,DMStagBCList*);
PetscErrorCode DMStagBCListDestroy(DMStagBCList*);

PetscErrorCode DMStagBCListView(DMStagBCList);
PetscErrorCode DMStagBCListSetupCoordinates(DMStagBCList);
PetscErrorCode DMStagBCListGetVertexBCs(DMStagBCList,PetscInt*,DMStagBC**);
PetscErrorCode DMStagBCListGetFaceBCs(DMStagBCList,PetscInt*,DMStagBC**);
PetscErrorCode DMStagBCListGetElementBCs(DMStagBCList,PetscInt*,DMStagBC**);


PetscErrorCode DMStagBCListGetValues(DMStagBCList,const char,const char,PetscInt,PetscInt*,PetscInt**,PetscScalar**,PetscScalar**,PetscScalar**,BCType**);
PetscErrorCode DMStagBCListInsertValues(DMStagBCList,const char,PetscInt,PetscInt*,PetscInt**,PetscScalar**,PetscScalar**,PetscScalar**,BCType**);
PetscErrorCode DMStagBCListPinValue(DMStagBCList,const char,PetscInt,PetscScalar);
PetscErrorCode DMStagBCListPinCornerValue(DMStagBCList,DMStagStencilLocation,const char,PetscInt,PetscScalar);

#endif
