
#include "petsc.h"
#include "petscdm.h"
#include "dmstagfieldlabel.h"

/*
 Inserts a label into the list.
 The first argument must be initialized to NULL before the first call to FieldLabelAdd.
 e.g.

 DMStagFieldLabel *label = NULL;
 DMStagStencilLocation loc = DMSTAG_LEFT;
 ierr = FieldLabelAdd(&label,"vx",0,1,&loc);CHKERRQ(ierr);
*/
PetscErrorCode DMStagFieldLabelAdd(DMStagFieldLabel **_f,const char name[],PetscInt dof,PetscInt nl,DMStagStencilLocation loc[])
{
  PetscErrorCode ierr;
  DMStagFieldLabel *f;

  if (!_f) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Arg 1 cannot be NULL");

  ierr = PetscCalloc1(1,&f);CHKERRQ(ierr);
  ierr = PetscSNPrintf(f->name,PETSC_MAX_PATH_LEN-1,"%s",name);CHKERRQ(ierr);
  f->dof_index = dof;
  f->nloc = nl;
  ierr = PetscMemcpy(f->location,loc,sizeof(DMStagStencilLocation)*nl);CHKERRQ(ierr);
  f->next = NULL;
  
  if (!*_f) {
    *_f = f;
  } else {
    DMStagFieldLabel *curr;
    
    curr = *_f;
    while (curr->next != NULL) {
      curr = curr->next;
    }
    curr->next = f;
  }
  
  PetscFunctionReturn(0);
}

/*
 Search for a label called "name" in the list.
*/
PetscErrorCode DMStagFieldLabelFind(DMStagFieldLabel *f,const char name[],PetscBool *found,PetscInt *_dof,PetscInt *_nl,const DMStagStencilLocation *_loc[])
{
  PetscErrorCode ierr;
  DMStagFieldLabel *curr;
  PetscInt dof = 0,nl = 0;
  DMStagStencilLocation *loc = NULL;
  PetscBool same;
  
  if (!f) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Arg 1 (DMStagFieldLabel*) cannot be NULL");
  if (!found) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Arg 3 (PetscBool*) cannot be NULL");
  *found = PETSC_FALSE;
  curr = f;
  do {
    ierr = PetscStrcmp(name,curr->name,&same);CHKERRQ(ierr);
    if (same) {
      dof = curr->dof_index;
      nl = curr->nloc;
      loc = curr->location;
      break;
    }
    curr = curr->next;
  } while (curr != NULL);
  if (nl != 0 && loc) {
    *found = PETSC_TRUE;
  }
  if (_dof)  { *_dof = dof; }
  if (_nl)  { *_nl = nl; }
  if (_loc) { *_loc = (const DMStagStencilLocation*)loc; }
  PetscFunctionReturn(0);
}

/*
 Attach the DMStagFieldLabel to any PetscObject.

 [Example]
 
 FieldLabel *label;
 Mat A;
 
 ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
 ierr = PetscObjectAttachFieldLabel((PetscObject)A,label);CHKERRQ(ierr);
*/
PetscErrorCode PetscObjectAttachDMStagFieldLabel(PetscObject pobj,DMStagFieldLabel *f)
{
  PetscErrorCode ierr;
  PetscContainer c;
  
  ierr = PetscContainerCreate(PETSC_COMM_SELF,&c);CHKERRQ(ierr);
  ierr = PetscContainerSetPointer(c,(void*)f);CHKERRQ(ierr);
  ierr = PetscObjectCompose(pobj,"__PContainer_DMStagFieldLabel__",(PetscObject)c);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
 Get the DMStagFieldLabel from a PetscObject.
 If PetscObjectAttachFieldLabel() with the provided PetscObject (arg 1)
 then the returned DMStagFieldLabel (arg 2) will be NULL.
*/
PetscErrorCode PetscObjectQueryDMStagFieldLabel(PetscObject pobj,DMStagFieldLabel **f)
{
  PetscErrorCode ierr;
  PetscContainer c = NULL;

  ierr = PetscObjectQuery(pobj,"__PContainer_DMStagFieldLabel__",(PetscObject*)&c);CHKERRQ(ierr);
  if (c) {
    ierr = PetscContainerGetPointer(c,(void**)f);CHKERRQ(ierr);
  } else {
    *f = NULL;
  }
  PetscFunctionReturn(0);
}
