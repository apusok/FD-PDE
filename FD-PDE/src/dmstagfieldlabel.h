
#ifndef __dmstagfieldlabel_h__
#define __dmstagfieldlabel_h__

#include "petsc.h"
#include "petscdmstag.h"

typedef struct _p_DMStagFieldLabel DMStagFieldLabel;

struct _p_DMStagFieldLabel {
  char                  name[PETSC_MAX_PATH_LEN];
  PetscInt              dof_index,nloc;
  DMStagStencilLocation location[30];
  DMStagFieldLabel      *next;
};

PetscErrorCode DMStagFieldLabelAdd(DMStagFieldLabel**,const char*,PetscInt,PetscInt,DMStagStencilLocation*);
PetscErrorCode DMStagFieldLabelFind(DMStagFieldLabel*,const char*,PetscBool*,PetscInt*,PetscInt*,const DMStagStencilLocation**);
PetscErrorCode PetscObjectAttachDMStagFieldLabel(PetscObject,DMStagFieldLabel*);
PetscErrorCode PetscObjectQueryDMStagFieldLabel(PetscObject,DMStagFieldLabel**);

#endif
