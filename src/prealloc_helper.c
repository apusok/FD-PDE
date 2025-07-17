
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


// ---------------------------------------
static PetscErrorCode MatCreatePreallocator_private(Mat A,Mat *p)
{
  Mat                    preallocator;
  PetscInt               M,N,m,n,bs;
  DM                     dm;
  ISLocalToGlobalMapping l2g[] = { NULL, NULL };
  PetscFunctionBegin;
  
  PetscCall(MatGetSize(A,&M,&N));
  PetscCall(MatGetLocalSize(A,&m,&n));
  PetscCall(MatGetBlockSize(A,&bs));
  PetscCall(MatGetDM(A,&dm));
  PetscCall(MatGetLocalToGlobalMapping(A,&l2g[0],&l2g[1]));
  
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A),&preallocator));
  PetscCall(MatSetType(preallocator,MATPREALLOCATOR));
  PetscCall(MatSetSizes(preallocator,m,n,M,N));
  PetscCall(MatSetBlockSize(preallocator,bs));
  PetscCall(MatSetDM(preallocator,dm));
  if (l2g[0] && l2g[1]) { PetscCall(MatSetLocalToGlobalMapping(preallocator,l2g[0],l2g[1])); }
  PetscCall(MatSetUp(preallocator));
  
  PetscCall(PetscObjectCompose((PetscObject)A,"__mat_preallocator__",(PetscObject)preallocator));
  if (p) {
    *p = preallocator;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
/* may return a NULL pointer */
PetscErrorCode MatGetPreallocator(Mat A,Mat *preallocator)
{
  Mat            p = NULL;
  PetscFunctionBegin;
  
  PetscCall(PetscObjectQuery((PetscObject)A,"__mat_preallocator__",(PetscObject*)&p));
  *preallocator = p;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
/*
 Returns preallocator, a matrix of type "preallocator".
 The user should not call MatDestroy() on preallocator;
*/
PetscErrorCode MatPreallocatePhaseBegin(Mat A,Mat *preallocator)
{
  Mat            p = NULL;
  PetscInt       bs;
  PetscFunctionBegin;
  
  PetscCall(MatGetPreallocator(A,&p));
  if (p) {
    PetscCall(MatDestroy(&p));
    p = NULL;
    PetscCall(PetscObjectCompose((PetscObject)A,"__mat_preallocator__",(PetscObject)p));
  }
  PetscCall(MatCreatePreallocator_private(A,&p));
  
  /* zap existing non-zero structure in A */
  /*
   It is a good idea to remove any exisiting non-zero structure in A to
   (i) reduce memory immediately
   (ii) to facilitate raising an error if someone trys to insert values into A after
   MatPreallocatorBegin() has been called - which signals they are doing something wrong/inconsistent
   */
  PetscCall(MatGetBlockSize(A,&bs));
  PetscCall(MatXAIJSetPreallocation(A,bs,NULL,NULL,NULL,NULL));
  PetscCall(MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE));
  
  *preallocator = p;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
PetscErrorCode MatPreallocatePhaseEnd(Mat A)
{
  Mat            p = NULL;
  PetscFunctionBegin;
  
  PetscCall(MatGetPreallocator(A,&p));
  if (!p) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_USER,"Must call MatPreallocatorBegin() first");
  PetscCall(MatAssemblyBegin(p,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(p,MAT_FINAL_ASSEMBLY));
  
  /* create new non-zero structure */
  PetscCall(MatPreallocatorPreallocate(p,PETSC_TRUE,A));
  
  /* clean up and remove the preallocator object from A */
  PetscCall(MatDestroy(&p));
  p = NULL;
  PetscCall(PetscObjectCompose((PetscObject)A,"__mat_preallocator__",(PetscObject)p));
  PetscFunctionReturn(PETSC_SUCCESS);
}
