#include <petsc.h>
#include <petscsnes.h>
#include <petsc/private/snesimpl.h>
#include <petscis.h>
#include <petscmat.h>
#include <petscdm.h>

#include "fdpde_snes.h"

typedef struct {
  Vec             X2;
  PetscErrorCode (*split_f)(SNES,Vec,Vec,Vec,void*);
  PetscReal      fnorm_adapt;
  PetscBool      consistent;
} SNES_PICARDLS;

// ---------------------------------------
PetscErrorCode SNESPicardComputeFunctionDefault(SNES snes, Vec x, Vec f, void *ctx)
{
  SNES_PICARDLS  *picard = (SNES_PICARDLS*)snes->data;
  PetscFunctionBegin;

#if defined(PETSC_USE_DEBUG)
  if (!picard->split_f) SETERRQ(PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_NULL,"Must call SNESPicardLSSetSplitFunction() before a residual can be computed");
#endif
  if (!picard->consistent) {
    PetscCall(picard->split_f(snes, x, picard->X2, f, ctx));
  } else {
    PetscCall(picard->split_f(snes, x, x, f, ctx));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
static PetscErrorCode SNESPicardComputeFunction_Consistent(SNES snes,Vec x,Vec f)
{
  SNES_PICARDLS  *picard = (SNES_PICARDLS*)snes->data;
  void           *ctx;
  PetscFunctionBegin;

#if defined(PETSC_USE_DEBUG)
  if (!picard->split_f) SETERRQ(PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_NULL,"Must call SNESPicardLSSetSplitFunction() before a residual can be computed");
#endif
  PetscCall(SNESGetFunction(snes,NULL,NULL,&ctx));
  PetscCall(VecZeroEntries(f));
  PetscCall(picard->split_f(snes, x, x, f, ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
/*
 Reference count on x is not incremeneted.
 Do not call VecDestroy() on the object returned.
*/
PetscErrorCode SNESPicardLSGetAuxillarySolution(SNES snes,Vec *x)
{
  SNES_PICARDLS *picard = (SNES_PICARDLS*)snes->data;
  PetscFunctionBegin;

  if (x) { *x = picard->X2; }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
PetscErrorCode SNESPicardLSSetSplitFunction(SNES snes,Vec F,
                                          PetscErrorCode (*f)(SNES,Vec,Vec,Vec,void*))
{
  SNES_PICARDLS  *picard = (SNES_PICARDLS*)snes->data;
  PetscFunctionBegin;

  picard->split_f = f;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
static PetscErrorCode SNESSolve_PicardLS(SNES snes)
{
  SNES_PICARDLS        *picard = (SNES_PICARDLS*)snes->data;
  PetscInt             maxits,i,lits;
  SNESLineSearchReason lssucceed;
  PetscReal            fnorm,gnorm,xnorm,ynorm;
  Vec                  Y,X,F;
  SNESLineSearch       linesearch;
  SNESConvergedReason  reason;
  PetscFunctionBegin;

  if (snes->xl || snes->xu || snes->ops->computevariablebounds) SETERRQ(PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_WRONGSTATE, "SNES solver %s does not support bounds", ((PetscObject)snes)->type_name);
  
  snes->numFailures            = 0;
  snes->numLinearSolveFailures = 0;
  snes->reason                 = SNES_CONVERGED_ITERATING;
  
  maxits = snes->max_its;               /* maximum number of iterations */
  X      = snes->vec_sol;               /* solution vector */
  F      = snes->vec_func;              /* residual vector */
  Y      = snes->vec_sol_update;        /* newton step */

  snes->iter = 0;
  snes->norm = 0.0;

  PetscCall(SNESGetLineSearch(snes, &linesearch));

  picard->consistent = PETSC_FALSE;
  
  if (!snes->vec_func_init_set) {
    PetscCall(SNESPicardComputeFunction_Consistent(snes,X,F));
  } else snes->vec_func_init_set = PETSC_FALSE;

  PetscCall(VecNorm(F,NORM_2,&fnorm));        /* fnorm <- ||F||  */
  if (fnorm < picard->fnorm_adapt) {
    PetscCall(PetscInfo(snes,"Switching to Newton (consistent) residual based on initial ||F||_2\n"));
    picard->consistent = PETSC_TRUE;
  }

  SNESCheckFunctionNorm(snes,fnorm);
  snes->norm = fnorm;
  PetscCall(SNESLogConvergenceHistory(snes,fnorm,0));
  PetscCall(SNESMonitor(snes,0,fnorm));

  /* test convergence */
  PetscCall((*snes->ops->converged)(snes,0,0.0,0.0,fnorm,&snes->reason,snes->cnvP));
  if (snes->reason) PetscFunctionReturn(PETSC_SUCCESS);

  for (i=0; i<maxits; i++) {
    
    /* Call general purpose update function */
    
    /* apply the nonlinear preconditioner */
    
    /* Solve J Y = F, where J is Jacobian matrix */
    PetscCall(SNESComputeFunction(snes,X,F));
    PetscCall(SNESComputeJacobian(snes,X,snes->jacobian,snes->jacobian_pre));
    PetscCall(KSPSetOperators(snes->ksp,snes->jacobian,snes->jacobian_pre));
    PetscCall(KSPSolve(snes->ksp,F,Y));
    PetscCall(KSPGetIterationNumber(snes->ksp,&lits));
    PetscCall(PetscInfo(snes,"iter=%" PetscInt_FMT ", linear solve iterations=%" PetscInt_FMT "\n",snes->iter,lits));
    
    if (PetscLogPrintInfo) {
    }
    
    /* Compute a (scaled) negative update in the line search routine:
     X <- X - lambda*Y
     and evaluate F = function(X) (depends on the line search).
     */
#if 1 /* line search */
    gnorm = fnorm;
    PetscCall(SNESLineSearchApply(linesearch, X, F, &fnorm, Y));
    PetscCall(SNESLineSearchGetReason(linesearch, &lssucceed));
    PetscCall(SNESLineSearchGetNorms(linesearch, &xnorm, &fnorm, &ynorm));
    PetscCall(PetscInfo(snes,"fnorm=%18.16e, gnorm=%18.16e, ynorm=%18.16e, lssucceed=%d\n",(double)gnorm,(double)fnorm,(double)ynorm,(int)lssucceed));
    if (snes->reason) break;
    SNESCheckFunctionNorm(snes,fnorm);
    if (lssucceed) {
      if (snes->stol*xnorm > ynorm) {
        PetscCall(VecCopy(X,picard->X2)); /* update cached state */

        snes->reason = SNES_CONVERGED_SNORM_RELATIVE;
        PetscFunctionReturn(PETSC_SUCCESS);
      }
      if (++snes->numFailures >= snes->maxFailures) {
        //PetscBool ismin;
        snes->reason = SNES_DIVERGED_LINE_SEARCH;
        //PetscCall(SNESNEWTONLSCheckLocalMin_Private(snes,snes->jacobian,F,fnorm,&ismin));
        //if (ismin) snes->reason = SNES_DIVERGED_LOCAL_MIN;
        break;
      }
    }
#endif
#if 0 /* By-pass linesearch and take full step */
    PetscCall(VecScale(Y,-1.0));
    PetscCall(VecAXPY(X,1.0,Y));
#endif
    
    PetscCall(VecCopy(X,picard->X2)); /* update cached state */
    
    PetscCall(SNESPicardComputeFunction_Consistent(snes,X,F)); /* compute true residual - used for stopping condition */
    PetscCall(VecNorm(F,NORM_2,&fnorm));        /* fnorm <- ||F||  */
    SNESCheckFunctionNorm(snes,fnorm);

    /* Monitor convergence */
    snes->iter = i+1;
    snes->norm = fnorm;
    snes->ynorm = ynorm;
    snes->xnorm = xnorm;
    PetscCall(SNESLogConvergenceHistory(snes,snes->norm,lits));
    PetscCall(SNESMonitor(snes,snes->iter,snes->norm));
    /* Test for convergence */
    PetscCall((*snes->ops->converged)(snes,snes->iter,xnorm,ynorm,fnorm,&snes->reason,snes->cnvP));
    if (snes->reason) break;

    if (!picard->consistent) {
      if (fnorm < picard->fnorm_adapt) {
        PetscCall(PetscInfo(snes,"Switching to Newton (consistent) residual at iteration %" PetscInt_FMT "\n",snes->iter));
        picard->consistent = PETSC_TRUE;
      }
    }
    
  }
  if (i == maxits) {
    PetscCall(PetscInfo(snes,"Maximum number of iterations has been reached: %" PetscInt_FMT "\n",maxits));
    if (!snes->reason) snes->reason = SNES_DIVERGED_MAX_IT;
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
static PetscErrorCode SNESSetUp_PicardLS(SNES snes)
{
  SNES_PICARDLS  *picard = (SNES_PICARDLS*)snes->data;
  PetscFunctionBegin;
  
  if (snes->vec_sol) {
    PetscCall(VecDuplicate(snes->vec_sol,&picard->X2));
  } else if (snes->vec_func) {
    PetscCall(VecDuplicate(snes->vec_func,&picard->X2));
  } else SETERRQ(PetscObjectComm((PetscObject)snes),PETSC_ERR_SUP,"Cannot allocate X2");

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
static PetscErrorCode SNESDestroy_PicardLS(SNES snes)
{
  SNES_PICARDLS  *picard = (SNES_PICARDLS*)snes->data;
  PetscFunctionBegin;

  PetscCall(VecDestroy(&picard->X2));
  picard->split_f = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
static PetscErrorCode SNESSetFromOptions_PicardLS(SNES snes, PetscOptionItems PetscOptionsObject)
{
  SNES_PICARDLS  *picard = (SNES_PICARDLS*)snes->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject,"SNES Picard (linesearch) options");
  PetscCall(PetscOptionsReal("-snes_picardls_fnorm_adapt","f-norm value to switch to NewtonLS","None",picard->fnorm_adapt,&picard->fnorm_adapt,NULL));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
PetscErrorCode SNESCreate_PicardLS(SNES snes)
{
  SNES_PICARDLS  *neP;
  SNESLineSearch linesearch;
  
  PetscFunctionBegin;
  snes->ops->setup          = SNESSetUp_PicardLS;
  snes->ops->solve          = SNESSolve_PicardLS;
  snes->ops->destroy        = SNESDestroy_PicardLS;
  snes->ops->setfromoptions = SNESSetFromOptions_PicardLS;
  //snes->ops->view           = SNESView_PicardLS;
  //snes->ops->reset          = SNESReset_PicardLS;
  
  snes->npcside = PC_RIGHT;
  snes->usesksp = PETSC_TRUE;
  snes->usesnpc = PETSC_TRUE;
  
  PetscCall(SNESGetLineSearch(snes, &linesearch));
  if (!((PetscObject)linesearch)->type_name) {
    PetscCall(SNESLineSearchSetType(linesearch, SNESLINESEARCHBASIC));
  }
  
  snes->alwayscomputesfinalresidual = PETSC_TRUE;
  
  PetscCall(PetscNew(&neP));
  neP->consistent = PETSC_FALSE;
  neP->fnorm_adapt = 1.0e-3;
  snes->data = (void*)neP;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
/*
 Usage: 
 (1) MatCreate(&A)
 (2) MatPreallocatorBegin(A,&p);
 (3) insert into p
 (4) MatPreallocatorEnd(A);
*/
// ---------------------------------------

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