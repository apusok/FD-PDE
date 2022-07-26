
#include "fdpde.h"
#include "composite_prealloc_utils.h"

const char pde_description[] = "Composite PDE object for coupled non-linear problems";

static PetscErrorCode JacobianCreate_Composite(FDPDE fd,Mat *J);
       PetscErrorCode FDPDESetUp_Composite(FDPDE fd);
static PetscErrorCode FDPDEView_Composite(FDPDE fd);
static PetscErrorCode FDPDEDestroy_Composite(FDPDE fd);
static PetscErrorCode FormFunction_Composite(SNES snes,Vec X,Vec F,void *ctx);

typedef struct {
  FDPDE     *pdelist;
  DM        *dmlist;
  Vec       *subX,*subF;
  PetscInt  n;
  PetscBool setup;
} PDEComposite;

PetscErrorCode FDPDECreate_Composite(FDPDE fd)
{
  PDEComposite *composite;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  // Initialize data
  ierr = PetscStrallocpy(pde_description,&fd->description); CHKERRQ(ierr);

  fd->dof0  = 0; fd->dof1  = 0; fd->dof2  = 0;
  fd->dofc0 = 0; fd->dofc1 = 0; fd->dofc2 = 0;
  
  ierr = PetscCalloc1(1,&composite);CHKERRQ(ierr);
  composite->setup = PETSC_FALSE;
  fd->data = composite;
  fd->ops->form_function      = FormFunction_Composite;
  fd->ops->form_jacobian      = NULL;
  fd->ops->create_jacobian    = JacobianCreate_Composite;
  fd->ops->view               = FDPDEView_Composite;
  fd->ops->destroy            = FDPDEDestroy_Composite;

  PetscFunctionReturn(0);
}

static PetscErrorCode JacobianCreate_Composite(FDPDE fd,Mat *J)
{
  PDEComposite   *composite;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  composite = (PDEComposite*)fd->data;
  ierr = FDPDECoupledCreateMatrix(composite->n,composite->dmlist,MATAIJ,J);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode FormFunction_Composite(SNES snes,Vec X,Vec F,void *ctx)
{
  FDPDE          fd = (FDPDE)ctx;
  PDEComposite   *composite;
  DM             dm;
  PetscInt       i;
  Vec            *subX,*subF;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  composite = (PDEComposite*)fd->data;
  dm = fd->dmstag; /* dmcomposite */
  subX = composite->subX;
  subF = composite->subF;
  ierr = DMCompositeGetAccessArray(dm,X,composite->n,NULL,subX);CHKERRQ(ierr);
  ierr = DMCompositeGetAccessArray(dm,F,composite->n,NULL,subF);CHKERRQ(ierr);
  /* set auxillary vectors */
  for (i=0; i<composite->n; i++) {
    composite->pdelist[i]->naux_global_vectors = composite->n;
    composite->pdelist[i]->aux_global_vectors = subX;
  }
  for (i=0; i<composite->n; i++) {
    /* Copy state (X) and residual (F) into sub-FDPDE objects */
    /* The state vector is most likely required as this is passed to form_coefficient() */
    /* The residual vector is likely NOT required to be copied */
    ierr = VecCopy(subX[i],composite->pdelist[i]->x);CHKERRQ(ierr);
    ierr = VecCopy(subF[i],composite->pdelist[i]->r);CHKERRQ(ierr);
    
    /* evaluate residual of sub-FDPDE */
    ierr = SNESComputeFunction(composite->pdelist[i]->snes,subX[i],subF[i]);CHKERRQ(ierr);
  }
  ierr = DMCompositeRestoreAccessArray(dm,F,composite->n,NULL,subF);CHKERRQ(ierr);
  ierr = DMCompositeRestoreAccessArray(dm,X,composite->n,NULL,subX);CHKERRQ(ierr);
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

  PetscFunctionReturn(0);
}

PetscErrorCode FDPDESetUp_Composite(FDPDE fd)
{
  PDEComposite *composite;
  PetscInt i;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  if (fd->setupcalled) PetscFunctionReturn(0);
  
  composite = (PDEComposite*)fd->data;
  if (!composite->setup) SETERRQ(fd->comm,PETSC_ERR_ORDER,"Must call FDPDECompositeSetFDPDE() before FDPDESetUp()");

  for (i=0; i<composite->n; i++) {
    ierr = FDPDESetUp(composite->pdelist[i]);CHKERRQ(ierr);
  }
  
  /*ierr = fd->ops->create(fd); CHKERRQ(ierr);*/
  ierr = DMCompositeCreate(fd->comm,&fd->dmstag);CHKERRQ(ierr);
  for (i=0; i<composite->n; i++) {
    ierr = DMCompositeAddDM(fd->dmstag,composite->dmlist[i]);CHKERRQ(ierr);
  }
  ierr = DMSetUp(fd->dmstag); CHKERRQ(ierr);
  
  ierr = DMCreateGlobalVector(fd->dmstag,&fd->x);CHKERRQ(ierr);
  ierr = VecDuplicate(fd->x,&fd->r);CHKERRQ(ierr);
  ierr = VecDuplicate(fd->x,&fd->xguess);CHKERRQ(ierr);
  
  if (!fd->ops->create_jacobian) SETERRQ(fd->comm,PETSC_ERR_ARG_NULL,"No method to create the Jacobian was provided. The FD-PDE implementation constructor is expected to set this function pointer");
  ierr = fd->ops->create_jacobian(fd,&fd->J); CHKERRQ(ierr);
  if (!fd->J) SETERRQ(fd->comm,PETSC_ERR_ARG_NULL,"Jacobian matrix for FD-PDE is NULL. The FD-PDE implementation for create_jacobian is required to create a valid Mat");
  
  ierr = SNESCreate(fd->comm,&fd->snes); CHKERRQ(ierr);
  //ierr = SNESSetDM(fd->snes,fd->dmstag); CHKERRQ(ierr);
  
  if (!fd->x) SETERRQ(fd->comm,PETSC_ERR_ARG_NULL,"Solution vector for FD-PDE is NULL.");
  ierr = SNESSetSolution(fd->snes,fd->x); CHKERRQ(ierr); // for FD colouring to function correctly
  
  if (!fd->ops->form_function) SETERRQ(fd->comm,PETSC_ERR_ARG_NULL,"No residual evaluation routine has been set.");
  ierr = SNESSetFunction(fd->snes, fd->r, fd->ops->form_function, (void*)fd); CHKERRQ(ierr);
  
  if (fd->ops->form_jacobian) {
    ierr = SNESSetJacobian(fd->snes, fd->J, fd->J, fd->ops->form_jacobian, (void*)fd); CHKERRQ(ierr);
  } else {
    ierr = SNESSetJacobian(fd->snes, fd->J, fd->J, SNESComputeJacobianDefaultColor, NULL); CHKERRQ(ierr);
  }
  fd->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode FDPDEView_Composite(FDPDE fd)
{
  PDEComposite *composite;
  PetscInt i;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  composite = (PDEComposite*)fd->data;
  for (i=0; i<composite->n; i++) {
    PetscPrintf(PETSC_COMM_WORLD,"Composite FDPDE[%D] \n",i);
    ierr = FDPDEView(composite->pdelist[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode FDPDEDestroy_Composite(FDPDE fd)
{
  PDEComposite *composite;
  PetscInt i;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  composite = (PDEComposite*)fd->data;
  for (i=0; i<composite->n; i++) {
    ierr = FDPDEDestroy(&composite->pdelist[i]);CHKERRQ(ierr);
    ierr = DMDestroy(&composite->dmlist[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(composite->pdelist);CHKERRQ(ierr);
  ierr = PetscFree(composite->dmlist);CHKERRQ(ierr);
  ierr = PetscFree(composite->subX);CHKERRQ(ierr);
  ierr = PetscFree(composite->subF);CHKERRQ(ierr);
  ierr = PetscFree(composite);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FDPDECompositeSetFDPDE(FDPDE fd,PetscInt n,FDPDE pdelist[])
{
  PDEComposite *composite;
  PetscInt i;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  if (fd->type != FDPDE_COMPOSITE) PetscFunctionReturn(0);
  composite = (PDEComposite*)fd->data;
  if (composite->setup) PetscFunctionReturn(0);
  composite->n = n;
  ierr = PetscCalloc1(n,&composite->pdelist);CHKERRQ(ierr);
  ierr = PetscCalloc1(n,&composite->dmlist);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    composite->pdelist[i] = pdelist[i];
    pdelist[i]->refcount++;
    ierr = FDPDEGetDM(pdelist[i],&composite->dmlist[i]);CHKERRQ(ierr);
  }
  ierr = PetscCalloc1(composite->n,&composite->subX);CHKERRQ(ierr);
  ierr = PetscCalloc1(composite->n,&composite->subF);CHKERRQ(ierr);
  composite->setup = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode FDPDECompositeGetFDPDE(FDPDE fd,PetscInt *n,FDPDE *pdelist[])
{
  PDEComposite *composite;
  
  PetscFunctionBegin;
  if (fd->type != FDPDE_COMPOSITE) PetscFunctionReturn(0);
  composite = (PDEComposite*)fd->data;
  if (n) *n = composite->n;
  if (pdelist) *pdelist = composite->pdelist;
  PetscFunctionReturn(0);
}

PetscErrorCode FDPDECompositeSynchronizeGlobalVectors(FDPDE fd,Vec X)
{
  PDEComposite   *composite;
  PetscInt       i;
  Vec            *subX;
  DM             dm;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  if (fd->type != FDPDE_COMPOSITE) PetscFunctionReturn(0);
  composite = (PDEComposite*)fd->data;
  dm = fd->dmstag; /* dmcomposite */
  subX = composite->subX;
  ierr = DMCompositeGetAccessArray(dm,X,composite->n,NULL,subX);CHKERRQ(ierr);
  for (i=0; i<composite->n; i++) {
    /* Copy state (X) and residual (F) into sub-FDPDE objects */
    ierr = VecCopy(subX[i],composite->pdelist[i]->x);CHKERRQ(ierr);
  }
  ierr = DMCompositeRestoreAccessArray(dm,X,composite->n,NULL,subX);CHKERRQ(ierr);
  /* probably unecesary, but NULL-ify vectors just for safety */
  for (i=0; i<composite->n; i++) {
    composite->subX[i] = NULL;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode FDPDECreateComposite(MPI_Comm comm,PetscInt n,FDPDE pdelist[],FDPDE *fd)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = FDPDECreate2(comm,fd);CHKERRQ(ierr);
  ierr = FDPDESetType(*fd,FDPDE_COMPOSITE);CHKERRQ(ierr);
  ierr = FDPDECompositeSetFDPDE(*fd,n,pdelist);CHKERRQ(ierr);
  ierr = FDPDESetUp(*fd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FDPDESNESComposite_GaussSeidel(FDPDE fd,Vec X)
{
  PDEComposite   *composite;
  DM             dm;
  PetscInt       i,its,maxit;
  Vec            *subX,*subF;
  PetscReal      normF,normF0,rtol,atol;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  composite = (PDEComposite*)fd->data;
  dm = fd->dmstag; /* dmcomposite */
  subX = composite->subX;
  subF = composite->subF;
  
  
  ierr = SNESGetTolerances(fd->snes,&atol,&rtol,NULL,&maxit,NULL);CHKERRQ(ierr);
  
  ierr = SNESComputeFunction(fd->snes,X,fd->r);CHKERRQ(ierr);
  ierr = VecNorm(fd->r,NORM_2,&normF0);CHKERRQ(ierr);
  
  for (its=0; its<maxit; its++) {

    ierr = DMCompositeGetAccessArray(dm,X,composite->n,NULL,subX);CHKERRQ(ierr);
    /* set auxillary vectors */
    for (i=0; i<composite->n; i++) {
      composite->pdelist[i]->naux_global_vectors = composite->n;
      composite->pdelist[i]->aux_global_vectors = subX;
    }
    
    //ierr = SNESComputeFunction(fd->snes,X,fd->r);CHKERRQ(ierr);
    ierr = DMCompositeGetAccessArray(dm,fd->r,composite->n,NULL,subF);CHKERRQ(ierr);

    for (i=0; i<composite->n; i++) {
      /* Copy state (X) and residual (F) into sub-FDPDE objects */
      /* The state vector is most likely required as this is passed to form_coefficient() */
      /* The residual vector is likely NOT required to be copied */
      ierr = VecCopy(subX[i],composite->pdelist[i]->x);CHKERRQ(ierr);
      ierr = VecCopy(subF[i],composite->pdelist[i]->r);CHKERRQ(ierr);

      ierr = SNESSolve(composite->pdelist[i]->snes,NULL,subX[i]);CHKERRQ(ierr);
      ierr = SNESComputeFunction(composite->pdelist[i]->snes,subX[i],subF[i]);CHKERRQ(ierr);
      VecNorm(subF[i],NORM_2,&normF);
      printf("[iteraton %d][pde %d] |F| %+1.6e\n",its,i,normF);
    }

    ierr = DMCompositeRestoreAccessArray(dm,fd->r,composite->n,NULL,subF);CHKERRQ(ierr);
    ierr = DMCompositeRestoreAccessArray(dm,X,composite->n,NULL,subX);CHKERRQ(ierr);

    ierr = SNESComputeFunction(fd->snes,X,fd->r);CHKERRQ(ierr);
    ierr = VecNorm(fd->r,NORM_2,&normF);CHKERRQ(ierr);
    printf("[iteraton %d] |F| %+1.6e |F|/|F0| %+1.6e\n",its,normF,normF/normF0);
    if (normF < atol) break;
    if (normF/normF0 < rtol) break;
  }

  /* copy solution */
  ierr = DMCompositeGetAccessArray(dm,X,composite->n,NULL,subX);CHKERRQ(ierr);
  for (i=0; i<composite->n; i++) {
    ierr = VecCopy(subX[i],composite->pdelist[i]->x);CHKERRQ(ierr);
  }
  ierr = DMCompositeRestoreAccessArray(dm,X,composite->n,NULL,subX);CHKERRQ(ierr);
  
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
  
  PetscFunctionReturn(0);
}


PetscErrorCode FDPDESNESComposite_Jacobi(FDPDE fd,Vec X)
{
  PDEComposite   *composite;
  DM             dm;
  PetscInt       i,its,maxit;
  Vec            *subX,*subF,*jstate;
  PetscReal      normF,normF0,rtol,atol;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  composite = (PDEComposite*)fd->data;
  dm = fd->dmstag; /* dmcomposite */
  subX = composite->subX;
  subF = composite->subF;
  
  ierr = PetscCalloc1(composite->n,&jstate);CHKERRQ(ierr);
  for (i=0; i<composite->n; i++) {
    jstate[i] = composite->pdelist[i]->x;
  }

  ierr = SNESGetTolerances(fd->snes,&atol,&rtol,NULL,&maxit,NULL);CHKERRQ(ierr);
  
  ierr = SNESComputeFunction(fd->snes,X,fd->r);CHKERRQ(ierr);
  ierr = VecNorm(fd->r,NORM_2,&normF0);CHKERRQ(ierr);
  
  for (its=0; its<maxit; its++) {
    
    ierr = DMCompositeGetAccessArray(dm,X,composite->n,NULL,subX);CHKERRQ(ierr);
    ierr = DMCompositeGetAccessArray(dm,fd->r,composite->n,NULL,subF);CHKERRQ(ierr);
    
    for (i=0; i<composite->n; i++) {

      ierr = SNESSetSolution(composite->pdelist[i]->snes,subX[i]);CHKERRQ(ierr);
      
      /* reset aux to point to state from previous iterate */
      composite->pdelist[i]->naux_global_vectors = composite->n;
      composite->pdelist[i]->aux_global_vectors = jstate;
      
      ierr = SNESSolve(composite->pdelist[i]->snes,NULL,subX[i]);CHKERRQ(ierr);
      ierr = SNESComputeFunction(composite->pdelist[i]->snes,subX[i],subF[i]);CHKERRQ(ierr);
      ierr = VecNorm(subF[i],NORM_2,&normF);CHKERRQ(ierr);
      printf("[iteraton %d][pde %d] |F| %+1.6e\n",its,i,normF);
    }
    
    /* update state */
    for (i=0; i<composite->n; i++) {
      ierr = VecCopy(subX[i],composite->pdelist[i]->x);CHKERRQ(ierr);
    }
    
    ierr = DMCompositeRestoreAccessArray(dm,fd->r,composite->n,NULL,subF);CHKERRQ(ierr);
    ierr = DMCompositeRestoreAccessArray(dm,X,composite->n,NULL,subX);CHKERRQ(ierr);
    
    ierr = SNESComputeFunction(fd->snes,X,fd->r);CHKERRQ(ierr);
    
    ierr = VecNorm(fd->r,NORM_2,&normF);CHKERRQ(ierr);
    printf("[iteraton %d] |F| %+1.6e |F|/|F0| %+1.6e\n",its,normF,normF/normF0);
    if (normF < atol) break;
    if (normF/normF0 < rtol) break;
  }

  ierr = DMCompositeGetAccessArray(dm,X,composite->n,NULL,subX);CHKERRQ(ierr);
  for (i=0; i<composite->n; i++) {
    ierr = VecCopy(subX[i],composite->pdelist[i]->x);CHKERRQ(ierr);
  }
  ierr = DMCompositeRestoreAccessArray(dm,X,composite->n,NULL,subX);CHKERRQ(ierr);
  
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
  ierr = PetscFree(jstate);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

