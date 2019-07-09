
#include <petscis.h>
#include <petscmat.h>
#include <petscdm.h>

/*
 
 Usage:
 
 (1) MatCreate(&A)
 (2) MatPreallocatorBegin(A,&p);
 (3) insert into p
 (4) MatPreallocatorEnd(A);
 
 */


static PetscErrorCode MatCreatePreallocator_private(Mat A,Mat *p)
{
  Mat                    preallocator;
  PetscInt               M,N,m,n,bs;
  DM                     dm;
  ISLocalToGlobalMapping l2g[] = { NULL, NULL };
  PetscErrorCode         ierr;
  
  ierr = MatGetSize(A,&M,&N);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
  ierr = MatGetBlockSize(A,&bs);CHKERRQ(ierr);
  ierr = MatGetDM(A,&dm);CHKERRQ(ierr);
  ierr = MatGetLocalToGlobalMapping(A,&l2g[0],&l2g[1]);CHKERRQ(ierr);
  
  ierr = MatCreate(PetscObjectComm((PetscObject)A),&preallocator);CHKERRQ(ierr);
  ierr = MatSetType(preallocator,MATPREALLOCATOR);CHKERRQ(ierr);
  ierr = MatSetSizes(preallocator,m,n,M,N);CHKERRQ(ierr);
  ierr = MatSetBlockSize(preallocator,bs);CHKERRQ(ierr);
  ierr = MatSetDM(preallocator,dm);CHKERRQ(ierr);
  if (l2g[0] && l2g[1]) { ierr = MatSetLocalToGlobalMapping(preallocator,l2g[0],l2g[1]);CHKERRQ(ierr); }
  ierr = MatSetUp(preallocator);CHKERRQ(ierr);
  
  ierr = PetscObjectCompose((PetscObject)A,"__mat_preallocator__",(PetscObject)preallocator);CHKERRQ(ierr);
  if (p) {
    *p = preallocator;
  }
  PetscFunctionReturn(0);
}

/* may return a NULL pointer */
PetscErrorCode MatGetPreallocator(Mat A,Mat *preallocator)
{
  PetscErrorCode ierr;
  Mat            p = NULL;
  
  ierr = PetscObjectQuery((PetscObject)A,"__mat_preallocator__",(PetscObject*)&p);CHKERRQ(ierr);
  *preallocator = p;
  PetscFunctionReturn(0);
}

/*
 Returns preallocator, a matrix of type "preallocator".
 The user should not call MatDestroy() on preallocator;
*/
PetscErrorCode MatPreallocatePhaseBegin(Mat A,Mat *preallocator)
{
  PetscErrorCode ierr;
  Mat            p = NULL;
  PetscInt       bs;
  
  ierr = MatGetPreallocator(A,&p);CHKERRQ(ierr);
  if (p) {
    ierr= MatDestroy(&p);CHKERRQ(ierr);
    p = NULL;
    ierr = PetscObjectCompose((PetscObject)A,"__mat_preallocator__",(PetscObject)p);CHKERRQ(ierr);
  }
  ierr = MatCreatePreallocator_private(A,&p);CHKERRQ(ierr);
  
  /* zap existing non-zero structure in A */
  /*
   It is a good idea to remove any exisiting non-zero structure in A to
   (i) reduce memory immediately
   (ii) to facilitate raising an error if someone trys to insert values into A after
   MatPreallocatorBegin() has been called - which signals they are doing something wrong/inconsistent
   */
  ierr = MatGetBlockSize(A,&bs);CHKERRQ(ierr);
  ierr = MatXAIJSetPreallocation(A,bs,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  
  *preallocator = p;
  PetscFunctionReturn(0);
}

PetscErrorCode MatPreallocatePhaseEnd(Mat A)
{
  PetscErrorCode ierr;
  Mat            p = NULL;
  
  ierr = MatGetPreallocator(A,&p);CHKERRQ(ierr);
  if (!p) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_USER,"Must call MatPreallocatorBegin() first");
  ierr = MatAssemblyBegin(p,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(p,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  
  /* create new non-zero structure */
  ierr = MatPreallocatorPreallocate(p,PETSC_TRUE,A);CHKERRQ(ierr);
  
  /* clean up and remove the preallocator object from A */
  ierr= MatDestroy(&p);CHKERRQ(ierr);
  p = NULL;
  ierr = PetscObjectCompose((PetscObject)A,"__mat_preallocator__",(PetscObject)p);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
