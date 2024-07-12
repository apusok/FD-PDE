
#include <petsc.h>
#include <petscsnes.h>
#include <petsc/private/snesimpl.h>
#include "snes_picard.h"

typedef struct {
  Vec             X2;
  PetscErrorCode (*split_f)(SNES,Vec,Vec,Vec,void*);
  PetscReal      fnorm_adapt;
  PetscBool      consistent;
} SNES_PICARDLS;


PetscErrorCode SNESPicardComputeFunctionDefault(SNES snes, Vec x, Vec f, void *ctx)
{
  SNES_PICARDLS  *picard = (SNES_PICARDLS*)snes->data;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
#if defined(PETSC_USE_DEBUG)
  if (!picard->split_f) SETERRQ(PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_NULL,"Must call SNESPicardLSSetSplitFunction() before a residual can be computed");
#endif
  if (!picard->consistent) {
    ierr = picard->split_f(snes, x, picard->X2, f, ctx);CHKERRQ(ierr);
  } else {
    ierr = picard->split_f(snes, x, x, f, ctx);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESPicardComputeFunction_Consistent(SNES snes,Vec x,Vec f)
{
  SNES_PICARDLS  *picard = (SNES_PICARDLS*)snes->data;
  void           *ctx;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
#if defined(PETSC_USE_DEBUG)
  if (!picard->split_f) SETERRQ(PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_NULL,"Must call SNESPicardLSSetSplitFunction() before a residual can be computed");
#endif
  ierr = SNESGetFunction(snes,NULL,NULL,&ctx);CHKERRQ(ierr);
  ierr = VecZeroEntries(f);CHKERRQ(ierr);
  ierr = picard->split_f(snes, x, x, f, ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
 Reference count on x is not incremeneted.
 Do not call VecDestroy() on the object returned.
*/
PetscErrorCode SNESPicardLSGetAuxillarySolution(SNES snes,Vec *x)
{
  SNES_PICARDLS *picard = (SNES_PICARDLS*)snes->data;
  PetscFunctionBegin;
  if (x) { *x = picard->X2; }
  PetscFunctionReturn(0);
}


PetscErrorCode SNESPicardLSSetSplitFunction(SNES snes,Vec F,
                                          PetscErrorCode (*f)(SNES,Vec,Vec,Vec,void*))
{
  SNES_PICARDLS  *picard = (SNES_PICARDLS*)snes->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  picard->split_f = f;
  PetscFunctionReturn(0);
}



static PetscErrorCode SNESSolve_PicardLS(SNES snes)
{
  SNES_PICARDLS        *picard = (SNES_PICARDLS*)snes->data;
  PetscErrorCode       ierr;
  PetscInt             maxits,i,lits;
  SNESLineSearchReason lssucceed;
  PetscReal            fnorm,gnorm,xnorm,ynorm;
  Vec                  Y,X,F;
  SNESLineSearch       linesearch;
  SNESConvergedReason  reason;

  
  PetscFunctionBegin;
  if (snes->xl || snes->xu || snes->ops->computevariablebounds) SETERRQ1(PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_WRONGSTATE, "SNES solver %s does not support bounds", ((PetscObject)snes)->type_name);
  
  snes->numFailures            = 0;
  snes->numLinearSolveFailures = 0;
  snes->reason                 = SNES_CONVERGED_ITERATING;
  
  maxits = snes->max_its;               /* maximum number of iterations */
  X      = snes->vec_sol;               /* solution vector */
  F      = snes->vec_func;              /* residual vector */
  Y      = snes->vec_sol_update;        /* newton step */

  snes->iter = 0;
  snes->norm = 0.0;

  ierr = SNESGetLineSearch(snes, &linesearch);CHKERRQ(ierr);

  picard->consistent = PETSC_FALSE;
  
  if (!snes->vec_func_init_set) {
    ierr = SNESPicardComputeFunction_Consistent(snes,X,F);CHKERRQ(ierr);
  } else snes->vec_func_init_set = PETSC_FALSE;

  ierr = VecNorm(F,NORM_2,&fnorm);CHKERRQ(ierr);        /* fnorm <- ||F||  */
  if (fnorm < picard->fnorm_adapt) {
    ierr = PetscInfo(snes,"Switching to Newton (consistent) residual based on initial ||F||_2\n");CHKERRQ(ierr);
    picard->consistent = PETSC_TRUE;
  }

  SNESCheckFunctionNorm(snes,fnorm);
  snes->norm = fnorm;
  ierr       = SNESLogConvergenceHistory(snes,fnorm,0);CHKERRQ(ierr);
  ierr       = SNESMonitor(snes,0,fnorm);CHKERRQ(ierr);

  /* test convergence */
  ierr = (*snes->ops->converged)(snes,0,0.0,0.0,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
  if (snes->reason) PetscFunctionReturn(0);

  for (i=0; i<maxits; i++) {
    
    /* Call general purpose update function */
    
    /* apply the nonlinear preconditioner */
    
    /* Solve J Y = F, where J is Jacobian matrix */
    ierr = SNESComputeFunction(snes,X,F);CHKERRQ(ierr);
    ierr = SNESComputeJacobian(snes,X,snes->jacobian,snes->jacobian_pre);CHKERRQ(ierr);
    ierr = KSPSetOperators(snes->ksp,snes->jacobian,snes->jacobian_pre);CHKERRQ(ierr);
    ierr = KSPSolve(snes->ksp,F,Y);CHKERRQ(ierr);
    ierr = KSPGetIterationNumber(snes->ksp,&lits);CHKERRQ(ierr);
    ierr = PetscInfo2(snes,"iter=%D, linear solve iterations=%D\n",snes->iter,lits);CHKERRQ(ierr);
    
    if (PetscLogPrintInfo) {
    }
    
    /* Compute a (scaled) negative update in the line search routine:
     X <- X - lambda*Y
     and evaluate F = function(X) (depends on the line search).
     */
#if 1 /* line search */
    gnorm = fnorm;
    ierr  = SNESLineSearchApply(linesearch, X, F, &fnorm, Y);CHKERRQ(ierr);
    ierr  = SNESLineSearchGetReason(linesearch, &lssucceed);CHKERRQ(ierr);
    ierr  = SNESLineSearchGetNorms(linesearch, &xnorm, &fnorm, &ynorm);CHKERRQ(ierr);
    ierr  = PetscInfo4(snes,"fnorm=%18.16e, gnorm=%18.16e, ynorm=%18.16e, lssucceed=%d\n",(double)gnorm,(double)fnorm,(double)ynorm,(int)lssucceed);CHKERRQ(ierr);
    if (snes->reason) break;
    SNESCheckFunctionNorm(snes,fnorm);
    if (lssucceed) {
      if (snes->stol*xnorm > ynorm) {
        ierr = VecCopy(X,picard->X2);CHKERRQ(ierr); /* update cached state */

        snes->reason = SNES_CONVERGED_SNORM_RELATIVE;
        PetscFunctionReturn(0);
      }
      if (++snes->numFailures >= snes->maxFailures) {
        //PetscBool ismin;
        snes->reason = SNES_DIVERGED_LINE_SEARCH;
        //ierr         = SNESNEWTONLSCheckLocalMin_Private(snes,snes->jacobian,F,fnorm,&ismin);CHKERRQ(ierr);
        //if (ismin) snes->reason = SNES_DIVERGED_LOCAL_MIN;
        break;
      }
    }
#endif
#if 0 /* By-pass linesearch and take full step */
    ierr = VecScale(Y,-1.0);CHKERRQ(ierr);
    ierr = VecAXPY(X,1.0,Y);CHKERRQ(ierr);
#endif
    
    ierr = VecCopy(X,picard->X2);CHKERRQ(ierr); /* update cached state */
    
    ierr = SNESPicardComputeFunction_Consistent(snes,X,F);CHKERRQ(ierr); /* compute true residual - used for stopping condition */
    ierr = VecNorm(F,NORM_2,&fnorm);CHKERRQ(ierr);        /* fnorm <- ||F||  */
    SNESCheckFunctionNorm(snes,fnorm);

    /* Monitor convergence */
    snes->iter = i+1;
    snes->norm = fnorm;
    snes->ynorm = ynorm;
    snes->xnorm = xnorm;
    ierr       = SNESLogConvergenceHistory(snes,snes->norm,lits);CHKERRQ(ierr);
    ierr       = SNESMonitor(snes,snes->iter,snes->norm);CHKERRQ(ierr);
    /* Test for convergence */
    ierr = (*snes->ops->converged)(snes,snes->iter,xnorm,ynorm,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
    if (snes->reason) break;

    if (!picard->consistent) {
      if (fnorm < picard->fnorm_adapt) {
        ierr = PetscInfo1(snes,"Switching to Newton (consistent) residual at iteration %D\n",snes->iter);CHKERRQ(ierr);
        picard->consistent = PETSC_TRUE;
      }
    }
    
  }
  if (i == maxits) {
    ierr = PetscInfo1(snes,"Maximum number of iterations has been reached: %D\n",maxits);CHKERRQ(ierr);
    if (!snes->reason) snes->reason = SNES_DIVERGED_MAX_IT;
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode SNESSetUp_PicardLS(SNES snes)
{
  SNES_PICARDLS  *picard = (SNES_PICARDLS*)snes->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  
  if (snes->vec_sol) {
    ierr = VecDuplicate(snes->vec_sol,&picard->X2);CHKERRQ(ierr);
  } else if (snes->vec_func) {
    ierr = VecDuplicate(snes->vec_func,&picard->X2);CHKERRQ(ierr);
  } else SETERRQ(PetscObjectComm((PetscObject)snes),PETSC_ERR_SUP,"Cannot allocate X2");

  PetscFunctionReturn(0);
}

static PetscErrorCode SNESDestroy_PicardLS(SNES snes)
{
  SNES_PICARDLS  *picard = (SNES_PICARDLS*)snes->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = VecDestroy(&picard->X2);CHKERRQ(ierr);
  picard->split_f = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESSetFromOptions_PicardLS(PetscOptionItems *PetscOptionsObject,SNES snes)
{
  SNES_PICARDLS  *picard = (SNES_PICARDLS*)snes->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"SNES Picard (linesearch) options");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-snes_picardls_fnorm_adapt","f-norm value to switch to NewtonLS","None",picard->fnorm_adapt,&picard->fnorm_adapt,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SNESCreate_PicardLS(SNES snes)
{
  PetscErrorCode ierr;
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
  
  ierr = SNESGetLineSearch(snes, &linesearch);CHKERRQ(ierr);
  if (!((PetscObject)linesearch)->type_name) {
    ierr = SNESLineSearchSetType(linesearch, SNESLINESEARCHBASIC);CHKERRQ(ierr);
  }
  
  snes->alwayscomputesfinalresidual = PETSC_TRUE;
  
  ierr = PetscNewLog(snes,&neP);CHKERRQ(ierr);
  neP->consistent = PETSC_FALSE;
  neP->fnorm_adapt = 1.0e-3;
  snes->data = (void*)neP;
  PetscFunctionReturn(0);
}
