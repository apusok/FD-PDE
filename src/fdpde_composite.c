#include "fdpde_composite.h"

const char pde_description[] = "Composite PDE object for coupled non-linear problems";

typedef struct {
  FDPDE     *pdelist;
  DM        *dmlist;
  Vec       *subX,*subF;
  PetscInt  n;
  PetscBool setup;
} PDEComposite;

// ---------------------------------------
PetscErrorCode FDPDECreate_Composite(FDPDE fd)
{
  PDEComposite *composite;
  PetscFunctionBegin;

  // Initialize data
  PetscCall(PetscStrallocpy(pde_description,&fd->description)); 

  fd->dof0  = 0; fd->dof1  = 0; fd->dof2  = 0;
  fd->dofc0 = 0; fd->dofc1 = 0; fd->dofc2 = 0;
  
  PetscCall(PetscCalloc1(1,&composite));
  composite->setup = PETSC_FALSE;
  fd->data = composite;
  fd->ops->form_function      = FormFunction_Composite;
  fd->ops->form_jacobian      = NULL;
  fd->ops->create_jacobian    = JacobianCreate_Composite;
  fd->ops->view               = FDPDEView_Composite;
  fd->ops->destroy            = FDPDEDestroy_Composite;

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
PetscErrorCode JacobianCreate_Composite(FDPDE fd,Mat *J)
{
  PDEComposite   *composite;

  PetscFunctionBegin;
  composite = (PDEComposite*)fd->data;
  PetscCall(FDPDECoupledCreateMatrix(composite->n,composite->dmlist,MATAIJ,J));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
PetscErrorCode FormFunction_Composite(SNES snes,Vec X,Vec F,void *ctx)
{
  FDPDE          fd = (FDPDE)ctx;
  PDEComposite   *composite;
  DM             dm;
  PetscInt       i;
  Vec            *subX,*subF;

  PetscFunctionBegin;
  composite = (PDEComposite*)fd->data;
  dm = fd->dmstag; /* dmcomposite */
  subX = composite->subX;
  subF = composite->subF;
  PetscCall(DMCompositeGetAccessArray(dm,X,composite->n,NULL,subX));
  PetscCall(DMCompositeGetAccessArray(dm,F,composite->n,NULL,subF));
  /* set auxillary vectors */
  for (i=0; i<composite->n; i++) {
    composite->pdelist[i]->naux_global_vectors = composite->n;
    composite->pdelist[i]->aux_global_vectors = subX;
  }
  for (i=0; i<composite->n; i++) {
    /* Copy state (X) and residual (F) into sub-FDPDE objects */
    /* The state vector is most likely required as this is passed to form_coefficient() */
    /* The residual vector is likely NOT required to be copied */
    PetscCall(VecCopy(subX[i],composite->pdelist[i]->x));
    PetscCall(VecCopy(subF[i],composite->pdelist[i]->r));
    
    /* evaluate residual of sub-FDPDE */
    PetscCall(SNESComputeFunction(composite->pdelist[i]->snes,subX[i],subF[i]));
  }
  PetscCall(DMCompositeRestoreAccessArray(dm,F,composite->n,NULL,subF));
  PetscCall(DMCompositeRestoreAccessArray(dm,X,composite->n,NULL,subX));
  /* probably unecesary, but NULL-ify vectors just for safety */
  for (i=0; i<composite->n; i++) {
    composite->subX[i] = NULL;
    composite->subF[i] = NULL;
  }
  /* reset auxillary vectors */
  for (i=0; i<composite->n; i++) {
    composite->pdelist[i]->naux_global_vectors = 0;
    composite->pdelist[i]->aux_global_vectors = NULL;
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
PetscErrorCode FDPDESetUp_Composite(FDPDE fd)
{
  PDEComposite *composite;
  PetscInt i;
  
  PetscFunctionBegin;
  if (fd->setupcalled) PetscFunctionReturn(PETSC_SUCCESS);
  
  composite = (PDEComposite*)fd->data;
  if (!composite->setup) SETERRQ(fd->comm,PETSC_ERR_ORDER,"Must call FDPDECompositeSetFDPDE() before FDPDESetUp()");

  for (i=0; i<composite->n; i++) {
    PetscCall(FDPDESetUp(composite->pdelist[i]));
  }
  
  /*PetscCall(fd->ops->create(fd); */
  PetscCall(DMCompositeCreate(fd->comm,&fd->dmstag));
  for (i=0; i<composite->n; i++) {
    PetscCall(DMCompositeAddDM(fd->dmstag,composite->dmlist[i]));
  }
  PetscCall(DMSetUp(fd->dmstag)); 
  
  PetscCall(DMCreateGlobalVector(fd->dmstag,&fd->x));
  PetscCall(VecDuplicate(fd->x,&fd->r));
  PetscCall(VecDuplicate(fd->x,&fd->xguess));
  
  if (!fd->ops->create_jacobian) SETERRQ(fd->comm,PETSC_ERR_ARG_NULL,"No method to create the Jacobian was provided. The FD-PDE implementation constructor is expected to set this function pointer");
  PetscCall(fd->ops->create_jacobian(fd,&fd->J)); 
  if (!fd->J) SETERRQ(fd->comm,PETSC_ERR_ARG_NULL,"Jacobian matrix for FD-PDE is NULL. The FD-PDE implementation for create_jacobian is required to create a valid Mat");
  
  PetscCall(SNESCreate(fd->comm,&fd->snes)); 
  //PetscCall(SNESSetDM(fd->snes,fd->dmstag); 
  
  if (!fd->x) SETERRQ(fd->comm,PETSC_ERR_ARG_NULL,"Solution vector for FD-PDE is NULL.");
  PetscCall(SNESSetSolution(fd->snes,fd->x));  // for FD colouring to function correctly
  
  if (!fd->ops->form_function) SETERRQ(fd->comm,PETSC_ERR_ARG_NULL,"No residual evaluation routine has been set.");
  PetscCall(SNESSetFunction(fd->snes, fd->r, fd->ops->form_function, (void*)fd)); 
  
  if (fd->ops->form_jacobian) {
    PetscCall(SNESSetJacobian(fd->snes, fd->J, fd->J, fd->ops->form_jacobian, (void*)fd)); 
  } else {
    PetscCall(SNESSetJacobian(fd->snes, fd->J, fd->J, SNESComputeJacobianDefaultColor, NULL)); 
  }
  fd->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
PetscErrorCode FDPDEView_Composite(FDPDE fd)
{
  PDEComposite *composite;
  PetscInt i;
  
  PetscFunctionBegin;
  composite = (PDEComposite*)fd->data;
  for (i=0; i<composite->n; i++) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Composite FDPDE[%D] \n",i));
    PetscCall(FDPDEView(composite->pdelist[i]));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
PetscErrorCode FDPDEDestroy_Composite(FDPDE fd)
{
  PDEComposite *composite;
  PetscInt i;
  
  PetscFunctionBegin;
  composite = (PDEComposite*)fd->data;
  for (i=0; i<composite->n; i++) {
    PetscCall(FDPDEDestroy(&composite->pdelist[i]));
    PetscCall(DMDestroy(&composite->dmlist[i]));
  }
  PetscCall(PetscFree(composite->pdelist));
  PetscCall(PetscFree(composite->dmlist));
  PetscCall(PetscFree(composite->subX));
  PetscCall(PetscFree(composite->subF));
  PetscCall(PetscFree(composite));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
PetscErrorCode FDPDECompositeSetFDPDE(FDPDE fd,PetscInt n,FDPDE pdelist[])
{
  PDEComposite *composite;
  PetscInt i;
  
  PetscFunctionBegin;
  if (fd->type != FDPDE_COMPOSITE) PetscFunctionReturn(PETSC_SUCCESS);
  composite = (PDEComposite*)fd->data;
  if (composite->setup) PetscFunctionReturn(PETSC_SUCCESS);
  composite->n = n;
  PetscCall(PetscCalloc1(n,&composite->pdelist));
  PetscCall(PetscCalloc1(n,&composite->dmlist));
  for (i=0; i<n; i++) {
    composite->pdelist[i] = pdelist[i];
    pdelist[i]->refcount++;
    PetscCall(FDPDEGetDM(pdelist[i],&composite->dmlist[i]));
  }
  PetscCall(PetscCalloc1(composite->n,&composite->subX));
  PetscCall(PetscCalloc1(composite->n,&composite->subF));
  composite->setup = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
PetscErrorCode FDPDECompositeGetFDPDE(FDPDE fd,PetscInt *n,FDPDE *pdelist[])
{
  PDEComposite *composite;
  
  PetscFunctionBegin;
  if (fd->type != FDPDE_COMPOSITE) PetscFunctionReturn(PETSC_SUCCESS);
  composite = (PDEComposite*)fd->data;
  if (n) *n = composite->n;
  if (pdelist) *pdelist = composite->pdelist;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
PetscErrorCode FDPDECompositeSynchronizeGlobalVectors(FDPDE fd,Vec X)
{
  PDEComposite   *composite;
  PetscInt       i;
  Vec            *subX;
  DM             dm;
  
  PetscFunctionBegin;
  if (fd->type != FDPDE_COMPOSITE) PetscFunctionReturn(PETSC_SUCCESS);
  composite = (PDEComposite*)fd->data;
  dm = fd->dmstag; /* dmcomposite */
  subX = composite->subX;
  PetscCall(DMCompositeGetAccessArray(dm,X,composite->n,NULL,subX));
  for (i=0; i<composite->n; i++) {
    /* Copy state (X) and residual (F) into sub-FDPDE objects */
    PetscCall(VecCopy(subX[i],composite->pdelist[i]->x));
  }
  PetscCall(DMCompositeRestoreAccessArray(dm,X,composite->n,NULL,subX));
  /* probably unecesary, but NULL-ify vectors just for safety */
  for (i=0; i<composite->n; i++) {
    composite->subX[i] = NULL;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
// PetscErrorCode FDPDECreateComposite(MPI_Comm comm,PetscInt n,FDPDE pdelist[],FDPDE *fd)
// {
//   PetscFunctionBegin;
//   PetscCall(FDPDECreate2(comm,fd));
//   PetscCall(FDPDESetType(*fd,FDPDE_COMPOSITE));
//   PetscCall(FDPDECompositeSetFDPDE(*fd,n,pdelist));
//   PetscCall(FDPDESetUp(*fd));
//   PetscFunctionReturn(PETSC_SUCCESS);
// }

// ---------------------------------------
// PetscErrorCode FDPDESNESComposite_GaussSeidel(FDPDE fd,Vec X)
// {
//   PDEComposite   *composite;
//   DM             dm;
//   PetscInt       i,its,maxit;
//   Vec            *subX,*subF;
//   PetscReal      normF,normF0,rtol,atol;

//   PetscFunctionBegin;
//   composite = (PDEComposite*)fd->data;
//   dm = fd->dmstag; /* dmcomposite */
//   subX = composite->subX;
//   subF = composite->subF;
  
//   PetscCall(SNESGetTolerances(fd->snes,&atol,&rtol,NULL,&maxit,NULL));
  
//   PetscCall(SNESComputeFunction(fd->snes,X,fd->r));
//   PetscCall(VecNorm(fd->r,NORM_2,&normF0));
  
//   for (its=0; its<maxit; its++) {

//     PetscCall(DMCompositeGetAccessArray(dm,X,composite->n,NULL,subX));
//     /* set auxillary vectors */
//     for (i=0; i<composite->n; i++) {
//       composite->pdelist[i]->naux_global_vectors = composite->n;
//       composite->pdelist[i]->aux_global_vectors = subX;
//     }
    
//     //PetscCall(SNESComputeFunction(fd->snes,X,fd->r);
//     PetscCall(DMCompositeGetAccessArray(dm,fd->r,composite->n,NULL,subF));

//     for (i=0; i<composite->n; i++) {
//       /* Copy state (X) and residual (F) into sub-FDPDE objects */
//       /* The state vector is most likely required as this is passed to form_coefficient() */
//       /* The residual vector is likely NOT required to be copied */
//       PetscCall(VecCopy(subX[i],composite->pdelist[i]->x));
//       PetscCall(VecCopy(subF[i],composite->pdelist[i]->r));

//       PetscCall(SNESSolve(composite->pdelist[i]->snes,NULL,subX[i]));
//       PetscCall(SNESComputeFunction(composite->pdelist[i]->snes,subX[i],subF[i]));
//       VecNorm(subF[i],NORM_2,&normF);
//       PetscCall(PetscPrintf(PETSC_COMM_WORLD,"[iteraton %d][pde %d] |F| %+1.6e\n",its,i,normF));
//     }

//     PetscCall(DMCompositeRestoreAccessArray(dm,fd->r,composite->n,NULL,subF));
//     PetscCall(DMCompositeRestoreAccessArray(dm,X,composite->n,NULL,subX));

//     PetscCall(SNESComputeFunction(fd->snes,X,fd->r));
//     PetscCall(VecNorm(fd->r,NORM_2,&normF));
//     PetscCall(PetscPrintf(PETSC_COMM_WORLD,"[iteraton %d] |F| %+1.6e |F|/|F0| %+1.6e\n",its,normF,normF/normF0));
//     if (normF < atol) break;
//     if (normF/normF0 < rtol) break;
//   }

//   /* copy solution */
//   PetscCall(DMCompositeGetAccessArray(dm,X,composite->n,NULL,subX));
//   for (i=0; i<composite->n; i++) {
//     PetscCall(VecCopy(subX[i],composite->pdelist[i]->x));
//   }
//   PetscCall(DMCompositeRestoreAccessArray(dm,X,composite->n,NULL,subX));
  
//   /* probably unecesary, but NULL-ify vectors just for safety */
//   for (i=0; i<composite->n; i++) {
//     composite->subX[i] = NULL;
//     composite->subF[i] = NULL;
//   }
//   /* reset auxillary vectors */
//   for (i=0; i<composite->n; i++) {
//     composite->pdelist[i]->naux_global_vectors = 0;
//     composite->pdelist[i]->aux_global_vectors = NULL;
//   }
  
//   PetscFunctionReturn(PETSC_SUCCESS);
// }

// ---------------------------------------
// PetscErrorCode FDPDESNESComposite_Jacobi(FDPDE fd,Vec X)
// {
//   PDEComposite   *composite;
//   DM             dm;
//   PetscInt       i,its,maxit;
//   Vec            *subX,*subF,*jstate;
//   PetscReal      normF,normF0,rtol,atol;
  
//   PetscFunctionBegin;
//   composite = (PDEComposite*)fd->data;
//   dm = fd->dmstag; /* dmcomposite */
//   subX = composite->subX;
//   subF = composite->subF;
  
//   PetscCall(PetscCalloc1(composite->n,&jstate));
//   for (i=0; i<composite->n; i++) {
//     jstate[i] = composite->pdelist[i]->x;
//   }

//   PetscCall(SNESGetTolerances(fd->snes,&atol,&rtol,NULL,&maxit,NULL));
  
//   PetscCall(SNESComputeFunction(fd->snes,X,fd->r));
//   PetscCall(VecNorm(fd->r,NORM_2,&normF0));
  
//   for (its=0; its<maxit; its++) {
    
//     PetscCall(DMCompositeGetAccessArray(dm,X,composite->n,NULL,subX));
//     PetscCall(DMCompositeGetAccessArray(dm,fd->r,composite->n,NULL,subF));
    
//     for (i=0; i<composite->n; i++) {

//       PetscCall(SNESSetSolution(composite->pdelist[i]->snes,subX[i]));
      
//       /* reset aux to point to state from previous iterate */
//       composite->pdelist[i]->naux_global_vectors = composite->n;
//       composite->pdelist[i]->aux_global_vectors = jstate;
      
//       PetscCall(SNESSolve(composite->pdelist[i]->snes,NULL,subX[i]));
//       PetscCall(SNESComputeFunction(composite->pdelist[i]->snes,subX[i],subF[i]));
//       PetscCall(VecNorm(subF[i],NORM_2,&normF));
//       PetscCall(PetscPrintf(PETSC_COMM_WORLD,"[iteraton %d][pde %d] |F| %+1.6e\n",its,i,normF));
//     }
    
//     /* update state */
//     for (i=0; i<composite->n; i++) {
//       PetscCall(VecCopy(subX[i],composite->pdelist[i]->x));
//     }
    
//     PetscCall(DMCompositeRestoreAccessArray(dm,fd->r,composite->n,NULL,subF));
//     PetscCall(DMCompositeRestoreAccessArray(dm,X,composite->n,NULL,subX));
    
//     PetscCall(SNESComputeFunction(fd->snes,X,fd->r));
    
//     PetscCall(VecNorm(fd->r,NORM_2,&normF));
//     PetscCall(PetscPrintf(PETSC_COMM_WORLD,"[iteraton %d] |F| %+1.6e |F|/|F0| %+1.6e\n",its,normF,normF/normF0));
//     if (normF < atol) break;
//     if (normF/normF0 < rtol) break;
//   }

//   PetscCall(DMCompositeGetAccessArray(dm,X,composite->n,NULL,subX));
//   for (i=0; i<composite->n; i++) {
//     PetscCall(VecCopy(subX[i],composite->pdelist[i]->x));
//   }
//   PetscCall(DMCompositeRestoreAccessArray(dm,X,composite->n,NULL,subX));
  
//   /* probably unecesary, but NULL-ify vectors just for safety */
//   for (i=0; i<composite->n; i++) {
//     composite->subX[i] = NULL;
//     composite->subF[i] = NULL;
//   }
//   /* reset auxillary vectors */
//   for (i=0; i<composite->n; i++) {
//     composite->pdelist[i]->naux_global_vectors = 0;
//     composite->pdelist[i]->aux_global_vectors = NULL;
//   }
//   PetscCall(PetscFree(jstate));
  
//   PetscFunctionReturn(PETSC_SUCCESS);
// }

