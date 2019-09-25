
#include "petsc.h"
#include "petscdm.h"
#include "../dmstagfieldlabel.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main (int argc,char **argv)
{
  PetscErrorCode  ierr;
  DMStagFieldLabel *label = NULL;
  DMStagStencilLocation loc;
  const DMStagStencilLocation *loc_f;
  PetscInt nl_f,dof_f;
  PetscBool found;
  Mat A;
  
  ierr = PetscInitialize(&argc,&argv,(char*)0,NULL); if (ierr) return(ierr);
  
  loc = DMSTAG_LEFT;
  ierr = DMStagFieldLabelAdd(&label,"vx",0,1,&loc);CHKERRQ(ierr);
  
  loc = DMSTAG_UP;
  ierr = DMStagFieldLabelAdd(&label,"vy",10,1,&loc);CHKERRQ(ierr);

  loc = DMSTAG_ELEMENT;
  ierr = DMStagFieldLabelAdd(&label,"p",30,1,&loc);CHKERRQ(ierr);
  
  
  ierr = DMStagFieldLabelFind(label,"vx",&found,&dof_f,&nl_f,&loc_f);CHKERRQ(ierr);
  printf("vx: found %d : dof %d : nl %d : loc %d\n",(int)found,(int)dof_f,(int)nl_f,(int)loc_f[0]);
  
  ierr = DMStagFieldLabelFind(label,"vy",&found,&dof_f,&nl_f,&loc_f);CHKERRQ(ierr);
  printf("vy: found %d : dof %d : nl %d : loc %d\n",(int)found,(int)dof_f,(int)nl_f,(int)loc_f[0]);

  ierr = DMStagFieldLabelFind(label,"p",&found,&dof_f,&nl_f,&loc_f);CHKERRQ(ierr);
  printf("p: found %d : dof %d : nl %d : loc %d\n",(int)found,(int)dof_f,(int)nl_f,(int)loc_f[0]);

  ierr = DMStagFieldLabelFind(label,"T",&found,&dof_f,&nl_f,&loc_f);CHKERRQ(ierr);
  if (!found) {
    printf("T: found %d : dof %d : nl %d : loc %p\n",(int)found,(int)dof_f,(int)nl_f,loc_f);
  }

  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = PetscObjectAttachDMStagFieldLabel((PetscObject)A,label);CHKERRQ(ierr);
  label = NULL;
  ierr = PetscObjectQueryDMStagFieldLabel((PetscObject)A,&label);CHKERRQ(ierr);

  ierr = DMStagFieldLabelFind(label,"vy",&found,&dof_f,&nl_f,&loc_f);CHKERRQ(ierr);
  printf("vy: found %d : dof %d : nl %d : loc %d\n",(int)found,(int)dof_f,(int)nl_f,(int)loc_f[0]);

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  
  ierr = PetscFinalize();
  return(ierr);
}
