#include <petsc.h>
#include <petscsnes.h>
#include <petsc/private/snesimpl.h>
#include <petscis.h>
#include <petscmat.h>
#include <petscdm.h>
#include <petsc/private/dmstagimpl.h>

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

// ---------------------------------------
PetscErrorCode private_DMStagGetStencilType(DM dm,DMStagStencilType *stencilType)
{
  DM_Stag * const stag = (DM_Stag*)dm->data;
  *stencilType = stag->stencilType;
  return(0);
}

// ---------------------------------------
/* Convert an array of DMStagStencil objects to an array of indices into a local vector.
 The .c fields in pos must always be set (even if to 0).  */
static PetscErrorCode private_DMStagStencilToIndexLocal(DM dm,PetscInt n,const DMStagStencil *pos,PetscInt *ix)
{
  const DM_Stag * const stag = (DM_Stag*)dm->data;
  PetscInt              idx,dim,startGhost[DMSTAG_MAX_DIM];
  const PetscInt        epe = stag->entriesPerElement;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMSTAG);
  PetscCall(DMGetDimension(dm,&dim));
  
#if defined(PETSC_USE_DEBUG)
  PetscCall(DMStagGetGhostCorners(dm,&startGhost[0],&startGhost[1],&startGhost[2],NULL,NULL,NULL));
#else
  PetscCall(DMStagGetGhostCorners(dm,&startGhost[0],&startGhost[1],&startGhost[2],NULL,NULL,NULL));
#endif
  
  if (dim == 1) {
    for (idx=0; idx<n; ++idx) {
      const PetscInt eLocal = pos[idx].i - startGhost[0]; /* Local element number */
      ix[idx] = eLocal * epe + stag->locationOffsets[pos[idx].loc] + pos[idx].c;
    }
  } else if (dim == 2) {
    const PetscInt epr = stag->nGhost[0];
    PetscCall(DMStagGetGhostCorners(dm,&startGhost[0],&startGhost[1],NULL,NULL,NULL,NULL));
    for (idx=0; idx<n; ++idx) {
      const PetscInt eLocalx = pos[idx].i - startGhost[0];
      const PetscInt eLocaly = pos[idx].j - startGhost[1];
      const PetscInt eLocal = eLocalx + epr*eLocaly;
      ix[idx] = eLocal * epe + stag->locationOffsets[pos[idx].loc] + pos[idx].c;
    }
  } else if (dim == 3) {
    const PetscInt epr = stag->nGhost[0];
    const PetscInt epl = stag->nGhost[0]*stag->nGhost[1];
    PetscCall(DMStagGetGhostCorners(dm,&startGhost[0],&startGhost[1],&startGhost[2],NULL,NULL,NULL));
    for (idx=0; idx<n; ++idx) {
      const PetscInt eLocalx = pos[idx].i - startGhost[0];
      const PetscInt eLocaly = pos[idx].j - startGhost[1];
      const PetscInt eLocalz = pos[idx].k - startGhost[2];
      const PetscInt eLocal  = epl*eLocalz + epr*eLocaly + eLocalx;
      ix[idx] = eLocal * epe + stag->locationOffsets[pos[idx].loc] + pos[idx].c;
    }
  } else SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Unsupported dimension %d",dim);
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
static PetscErrorCode convert_in_place(DM dm,PetscInt n,const DMStagStencil *pos, PetscInt **ix)
{
  PetscInt *_ix;  
  PetscFunctionBegin;
  PetscCall(PetscMalloc1(n,&_ix));
  PetscCall(private_DMStagStencilToIndexLocal(dm,n,pos,_ix));
  //for (i=0; i<n; i++) printf("ix[%d] = %d\n",i,ix[i]);
  *ix = _ix;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
static PetscErrorCode FillStencilCentral_2D(DM dm,PetscInt i,PetscInt j,PetscInt Ni,PetscInt Nj,
                                 PetscInt *count,DMStagStencil point[])
{
  PetscInt d,v;
  PetscBool vertices=PETSC_FALSE,edges=PETSC_FALSE,cells=PETSC_FALSE;
  PetscInt dof[3];
  
  PetscFunctionBegin;
  PetscCall(DMStagGetDOF(dm,&dof[0],&dof[1],&dof[2],NULL));
  if (dof[0] > 0) vertices = PETSC_TRUE;
  if (dof[1] > 0) edges = PETSC_TRUE;
  if (dof[2] > 0) cells = PETSC_TRUE;
  if (vertices) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for vertices");
  
  v = 0;
  if (vertices) {
  }
  
  if (edges) {
    for (d=0; d<dof[1]; d++) {
      point[v].i = i;
      point[v].j = j;
      point[v].loc = DMSTAG_LEFT;
      point[v].c = d;
      v++;
    }
    
    for (d=0; d<dof[1]; d++) {
      point[v].i = i;
      point[v].j = j;
      point[v].loc = DMSTAG_RIGHT;
      point[v].c = d;
      v++;
    }
    
    for (d=0; d<dof[1]; d++) {
      point[v].i = i;
      point[v].j = j;
      point[v].loc = DMSTAG_UP;
      point[v].c = d;
      v++;
    }
    
    for (d=0; d<dof[1]; d++) {
      point[v].i = i;
      point[v].j = j;
      point[v].loc = DMSTAG_DOWN;
      point[v].c = d;
      v++;
    }
  }
  if (cells) {
    for (d=0; d<dof[2]; d++) {
      point[v].i = i;
      point[v].j = j;
      point[v].loc = DMSTAG_ELEMENT;
      point[v].c = d;
      v++;
    }
  }
  *count = v;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
static PetscErrorCode FillStencilBox_2D(DM dm,PetscInt i,PetscInt j,PetscInt Ni,PetscInt Nj,
                                 PetscInt *count,DMStagStencil point[])
{
  PetscInt d,ii,jj,si,sj,ei,ej,v,sw;
  PetscBool vertices=PETSC_FALSE,edges=PETSC_FALSE,cells=PETSC_FALSE;
  PetscInt dof[3];
  
  PetscFunctionBegin;
  PetscCall(DMStagGetStencilWidth(dm,&sw));
  PetscCall(DMStagGetDOF(dm,&dof[0],&dof[1],&dof[2],NULL));
  if (dof[0] > 0) vertices = PETSC_TRUE;
  if (dof[1] > 0) edges = PETSC_TRUE;
  if (dof[2] > 0) cells = PETSC_TRUE;
  if (vertices) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for vertices");
  
  si = PetscMax(0,i-sw);    sj = PetscMax(0,j-sw);
  ei = PetscMin(Ni-1,i+sw); ej = PetscMin(Nj-1,j+sw);
  
  v = 0;
  for (jj=sj; jj<=ej; jj++) {
    for (ii=si; ii<=ei; ii++) {
      if (vertices) {
      }
      
      if (edges) {
        for (d=0; d<dof[1]; d++) {
          point[v].i = ii;
          point[v].j = jj;
          point[v].loc = DMSTAG_LEFT;
          point[v].c = d;
          v++;
        }
        
        for (d=0; d<dof[1]; d++) {
          point[v].i = ii;
          point[v].j = jj;
          point[v].loc = DMSTAG_RIGHT;
          point[v].c = d;
          v++;
        }
        
        for (d=0; d<dof[1]; d++) {
          point[v].i = ii;
          point[v].j = jj;
          point[v].loc = DMSTAG_UP;
          point[v].c = d;
          v++;
        }
        
        for (d=0; d<dof[1]; d++) {
          point[v].i = ii;
          point[v].j = jj;
          point[v].loc = DMSTAG_DOWN;
          point[v].c = d;
          v++;
        }
      }
      if (cells) {
        for (d=0; d<dof[2]; d++) {
          point[v].i = ii;
          point[v].j = jj;
          point[v].loc = DMSTAG_ELEMENT;
          point[v].c = d;
          v++;
        }
      }
    }
  }
  *count = v;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
static PetscErrorCode FillStencilStar_2D(DM dm,PetscInt i,PetscInt j,PetscInt Ni,PetscInt Nj,
                                 PetscInt *count,DMStagStencil point[])
{
  PetscInt d,ii,v,sw,star_i[13],star_j[13],nvmax;
  PetscBool vertices=PETSC_FALSE,edges=PETSC_FALSE,cells=PETSC_FALSE;
  PetscInt dof[3];
  
  PetscFunctionBegin;
  PetscCall(DMStagGetStencilWidth(dm,&sw));
  if (sw > 2) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for anything other than star stencil width of 0, 1 or 2");
  PetscCall(DMStagGetDOF(dm,&dof[0],&dof[1],&dof[2],NULL));
  if (dof[0] > 0) vertices = PETSC_TRUE;
  if (dof[1] > 0) edges = PETSC_TRUE;
  if (dof[2] > 0) cells = PETSC_TRUE;
  if (vertices) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for vertices");
  
  /* sw = 0 */
  nvmax = 0;
  if (sw == 0) {
    star_i[nvmax] = i;   star_j[nvmax] = j;   nvmax++;
  }
  
  /* sw = 1 */
  /*
           [i,j+1]
   [i-1,j] [i,j]   [i+1,j]
           [i,j-1]
  */
  if (sw == 1) {
    star_i[nvmax] = i;   star_j[nvmax] = j+1; nvmax++;
    //
    star_i[nvmax] = i-1; star_j[nvmax] = j;   nvmax++;
    star_i[nvmax] = i;   star_j[nvmax] = j;   nvmax++;
    star_i[nvmax] = i+1; star_j[nvmax] = j;   nvmax++;
    //
    star_i[nvmax] = i;   star_j[nvmax] = j-1; nvmax++;
  }
  
  /* sw = 2 */
  /*
                       [i,j+2]
             [i-1,j+1] [i,j+1] [i+1,j+1]
     [i-2,j] [i-1,j]   [i,j]   [i+1,j]   [i+2,j]
             [i-1,j-1] [i,j-1] [i+1,j-1]
                       [i,j-2]
  */
  if (sw == 2) {
    star_i[nvmax] = i;   star_j[nvmax] = j+2; nvmax++;
    //
    star_i[nvmax] = i-1; star_j[nvmax] = j+1; nvmax++;
    star_i[nvmax] = i;   star_j[nvmax] = j+1; nvmax++;
    star_i[nvmax] = i+1; star_j[nvmax] = j+1; nvmax++;
    //
    star_i[nvmax] = i-2; star_j[nvmax] = j;   nvmax++;
    star_i[nvmax] = i-1; star_j[nvmax] = j;   nvmax++;
    star_i[nvmax] = i;   star_j[nvmax] = j;   nvmax++;
    star_i[nvmax] = i+1; star_j[nvmax] = j;   nvmax++;
    star_i[nvmax] = i+2; star_j[nvmax] = j;   nvmax++;
    //
    star_i[nvmax] = i-1; star_j[nvmax] = j-1; nvmax++;
    star_i[nvmax] = i;   star_j[nvmax] = j-1; nvmax++;
    star_i[nvmax] = i+1; star_j[nvmax] = j-1; nvmax++;
    //
    star_i[nvmax] = i;   star_j[nvmax] = j-2; nvmax++;
  }
  if (nvmax == 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for anything other than star stencil width of 0, 1 or 2");
  
  /*
  for (ii=0; ii<nvmax; ii++) {
    star_i[ii] = PetscMax(0,star_i[ii]);
    star_i[ii] = PetscMin(Ni-1,star_i[ii]);
    
    star_j[ii] = PetscMax(0,star_j[ii]);
    star_j[ii] = PetscMin(Nj-1,star_j[ii]);
  }
  */
  for (ii=0; ii<nvmax; ii++) {
    if (star_i[ii] < 0) { star_i[ii] = 0; }
    if (star_j[ii] < 0) { star_j[ii] = 0; }
    if (star_i[ii] > Ni-1) { star_i[ii] = Ni-1; }
    if (star_j[ii] > Nj-1) { star_j[ii] = Nj-1; }
  }
  
  v = 0;
  for (ii=0; ii<nvmax; ii++) {
    if (vertices) {
    }
    
    if (edges) {
      for (d=0; d<dof[1]; d++) {
        point[v].i = star_i[ii];
        point[v].j = star_j[ii];
        point[v].loc = DMSTAG_LEFT;
        point[v].c = d;
        v++;
      }
      
      for (d=0; d<dof[1]; d++) {
        point[v].i = star_i[ii];
        point[v].j = star_j[ii];
        point[v].loc = DMSTAG_RIGHT;
        point[v].c = d;
        v++;
      }
      
      for (d=0; d<dof[1]; d++) {
        point[v].i = star_i[ii];
        point[v].j = star_j[ii];
        point[v].loc = DMSTAG_UP;
        point[v].c = d;
        v++;
      }
      
      for (d=0; d<dof[1]; d++) {
        point[v].i = star_i[ii];
        point[v].j = star_j[ii];
        point[v].loc = DMSTAG_DOWN;
        point[v].c = d;
        v++;
      }
    }
    if (cells) {
      for (d=0; d<dof[2]; d++) {
        point[v].i = star_i[ii];
        point[v].j = star_j[ii];
        point[v].loc = DMSTAG_ELEMENT;
        point[v].c = d;
        v++;
      }
    }
  }
  *count = v;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
static PetscErrorCode CreateStencilBuffer_2D(DM dm,PetscInt *_max_size,DMStagStencil *point[])
{
  PetscInt sw,size=0,cellsize,dof[3];
  PetscBool vertices=PETSC_FALSE,edges=PETSC_FALSE,cells=PETSC_FALSE;
  DMStagStencilType stencilType;
  
  PetscFunctionBegin;
  PetscCall(DMStagGetStencilWidth(dm,&sw));
  PetscCall(DMStagGetDOF(dm,&dof[0],&dof[1],&dof[2],NULL));
  if (dof[0] > 0) vertices = PETSC_TRUE;
  if (dof[1] > 0) edges = PETSC_TRUE;
  if (dof[2] > 0) cells = PETSC_TRUE;
  cellsize = 0;
  if (vertices) cellsize += 4 * dof[0]; /* count all vertices */
  if (edges)    cellsize += 4 * dof[1]; /* counter left/right/up/down edges */
  if (cells)    cellsize += dof[2];
  PetscCall(private_DMStagGetStencilType(dm,&stencilType));
  switch (stencilType) {
    case  DMSTAG_STENCIL_STAR:
    if (sw > 2) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for anything other than star stencil width of 1 or 2");
    if (sw == 0) {
      size = cellsize;
    } else if (sw == 1) {
      size = 5 * cellsize;
    } else if (sw == 2) {
      size = 13 * cellsize;
    } else {
      size = 0;
    }
    break;

    case  DMSTAG_STENCIL_BOX:
    size = (2*sw+1) * (2*sw+1) * cellsize;
    break;

    default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only stencil type DMSTAG_STENCIL_STAR and DMSTAG_STENCIL_BOX supported");
    break;
  }
  
  PetscCall(PetscCalloc1(size,point));
  *_max_size = size;
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
static PetscErrorCode _preallocate_coupled(Mat p,PetscInt i,DM row_dm,PetscInt ncols,DM cols_dm[],PetscInt offset[],PetscBool col_mask[])
{
  PetscInt j;
  PetscInt *rowidx,*colidx;
  PetscInt ci,cj,sx,sz,nx,nz,Ni,Nj,ii,jj,d;
  PetscInt **indices;
  DMStagStencil *r_point_buffer,**c_point_buffer;
  PetscInt remap,r_max_size,*c_max_size;
  PetscErrorCode (*fill_stencil[50])(DM,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt*,DMStagStencil*);
  DMStagStencilType stencilType;
  PetscInt r_used,c_used;
  PetscMPIInt rank;
  
  PetscFunctionBegin;
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  for (j=0; j<ncols; j++) {
    PetscCall(private_DMStagGetStencilType(cols_dm[j],&stencilType));
    switch (stencilType) {
      case DMSTAG_STENCIL_STAR:
      fill_stencil[j] = FillStencilStar_2D;
      break;
      case DMSTAG_STENCIL_BOX:
      fill_stencil[j] = FillStencilBox_2D;
      break;
      default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only stencil type DMSTAG_STENCIL_STAR and DMSTAG_STENCIL_BOX supported");
      break;
    }
  }
  
  PetscCall(DMStagGetGlobalSizes(row_dm,&Ni,&Nj,NULL));
  PetscCall(DMStagGetCorners(row_dm,&sx,&sz,NULL,&nx,&nz,NULL,NULL,NULL,NULL));
  
  PetscCall(PetscCalloc1(ncols,&indices));
  for (d=0; d<ncols; d++) {
    ISLocalToGlobalMapping ltog;
    PetscInt ltog_size;
    
    PetscCall(DMGetLocalToGlobalMapping(cols_dm[d],&ltog));
    PetscCall(ISLocalToGlobalMappingGetIndices(ltog,(const PetscInt**)&indices[d]));
    PetscCall(ISLocalToGlobalMappingGetSize(ltog,&ltog_size));
    printf("[%d] ltog size %d\n",d,ltog_size);
  }
  
  /* insert */
  PetscCall(CreateStencilBuffer_2D(row_dm,&r_max_size,&r_point_buffer));
  printf("r_max_size[%d] = %d\n",i,r_max_size);
  
  PetscCall(PetscCalloc1(ncols,&c_point_buffer));
  PetscCall(PetscCalloc1(ncols,&c_max_size));
  for (j=0; j<ncols; j++) {
    PetscCall(CreateStencilBuffer_2D(cols_dm[j],&c_max_size[j],&c_point_buffer[j]));
    printf("[%d] c_max_size[%d] = %d\n",i,j,c_max_size[j]);
  }
  
  for (cj=sz; cj<sz+nz; cj++) {
    for (ci=sx; ci<sx+nx; ci++) {
      
      PetscCall(FillStencilCentral_2D(row_dm,ci,cj,Ni,Nj,&r_used,r_point_buffer));
      PetscCall(convert_in_place(row_dm,r_used,r_point_buffer,&rowidx));
      for (ii=0; ii<r_used; ii++) {
        //printf("ii %d : rowidx[jj] %d\n",ii,rowidx[ii]);
        if (rowidx[ii] < 0) { continue; }
        remap = indices[i][ rowidx[ii] ];
        //printf("-->ii %d : remap %d\n",ii,remap);
        rowidx[ii] = remap + offset[i];
        //printf("-->-->ii %d : rowidx[jj] %d\n",ii,rowidx[ii]);
      }
      
      for (j=0; j<ncols; j++) {
        if (col_mask[j]) continue;
        
        PetscCall(fill_stencil[j](cols_dm[j],ci,cj,Ni,Nj,&c_used,c_point_buffer[j]));
        PetscCall(convert_in_place(cols_dm[j],c_used,c_point_buffer[j],&colidx));
        
        for (jj=0; jj<c_used; jj++) {
          //printf("jj %d : colidx[jj] %d\n",jj,colidx[jj]);
          if (colidx[jj] < 0) { continue; }
          remap = indices[j][ colidx[jj] ];
          colidx[jj] = remap + offset[j];
        }

        /*
        printf("i %d j %d (ci %d cj %d) ->\n",i,j,ci,cj);
        for (ii=0; ii<r_used; ii++)  printf(" %d ",rowidx[ii]); printf("\n");
        for (jj=0; jj<c_used; jj++)  printf(" %d ",colidx[jj]); printf("\n");
        */
        
        for (ii=0; ii<r_used; ii++) {
          for (jj=0; jj<c_used; jj++) {
            {PetscCall(MatSetValue(p,rowidx[ii],colidx[jj],1.0,INSERT_VALUES));}
          }
        }
        PetscCall(PetscFree(colidx));
      }
      PetscCall(PetscFree(rowidx));
    }
  }
  
  PetscCall(MatAssemblyBegin(p,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(p,MAT_FINAL_ASSEMBLY));
  
  for (j=0; j<ncols; j++) {
    PetscCall(PetscFree(c_point_buffer[j]));
  }
  PetscCall(PetscFree(c_point_buffer));
  PetscCall(PetscFree(c_max_size));
  PetscCall(PetscFree(r_point_buffer));
  
  PetscCall(PetscFree(indices));
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
PetscErrorCode FDPDECoupledCreateMatrix(PetscInt ndm,DM dm[],MatType mtype,Mat *A)
{
  PetscInt i,d;
  PetscInt *offset,*m,*n,*M,*N,sizes[] = {0,0,0,0},Mo;
  PetscBool *col_mask;
  Mat preallocator;
  
  PetscFunctionBegin;
  /* check all DMs are DMSTAG */
  for (d=0; d<ndm; d++) {
    PetscBool isstag;
    PetscCall(PetscObjectTypeCompare((PetscObject)dm[d],DMSTAG,&isstag));
    if (!isstag) SETERRQ(PetscObjectComm((PetscObject)dm[d]),PETSC_ERR_ARG_WRONG,"DM[%" PetscInt_FMT "] is not of type DMSTAG",d);
  }
  
  /* check sizes are consistent */
  {
    PetscInt Ni,Nj,jNi,jNj;
    
    PetscCall(DMStagGetGlobalSizes(dm[0],&Ni,&Nj,NULL));
    for (d=1; d<ndm; d++) {
      PetscCall(DMStagGetGlobalSizes(dm[d],&jNi,&jNj,NULL));
      if (Ni != jNi) SETERRQ(PetscObjectComm((PetscObject)dm[0]),PETSC_ERR_ARG_WRONG,"DM (Ni=%" PetscInt_FMT ") does not match size of DM[0] (Ni=%" PetscInt_FMT ")",Ni,jNi);
      if (Nj != jNj) SETERRQ(PetscObjectComm((PetscObject)dm[0]),PETSC_ERR_ARG_WRONG,"DM (Nj=%" PetscInt_FMT ") does not match size of DM[0] (Nj=%" PetscInt_FMT ")",Nj,jNj);
    }
  }
  
  /* determine global and local sizes */
  /* compute offsets for insertions */
  PetscCall(PetscCalloc1(ndm,&offset));
  PetscCall(PetscCalloc1(ndm,&col_mask)); 
  PetscCall(PetscCalloc1(ndm,&m));
  PetscCall(PetscCalloc1(ndm,&n)); 
  PetscCall(PetscCalloc1(ndm,&M)); 
  PetscCall(PetscCalloc1(ndm,&N)); 
  for (d=0; d<ndm; d++) {
    col_mask[d] = PETSC_FALSE;
  }
  for (d=0; d<ndm; d++) {
    Vec x;
    PetscInt size;
    PetscCall(DMCreateGlobalVector(dm[d],&x)); 
    PetscCall(VecGetSize(x,&size));
    M[d] = size;
    N[d] = size;
    PetscCall(VecGetLocalSize(x,&size)); 
    m[d] = size;
    n[d] = size;
    PetscCall(VecDestroy(&x)); 
  }
  for (d=1; d<ndm; d++) {
    offset[d] = offset[d-1] + m[d-1];
  }
  for (d=0; d<ndm; d++) {
    sizes[0] += m[d];
    sizes[1] += n[d];
    sizes[2] += M[d];
    sizes[3] += N[d];
  }
  Mo = 0;
  /*PetscCall(MPI_Scan(&sizes[0], &Mo, 1, MPI_INT, MPI_SUM, PetscObjectComm((PetscObject)dm[0]))); */
  /*Mo -= sizes[0];*/
  for (d=0; d<ndm; d++) {
    offset[d] += Mo;
  }
  
  for (d=0; d<ndm; d++) {
    printf("[%d] MxN %d %d mxn %d %d\n",d,M[d],N[d],m[d],n[d]);
    printf("offset[%d] %d\n",d,offset[d]);
  }
  printf("sizes MxN %d %d <global>\n",sizes[2],sizes[3]);
  printf("sizes mxn %d %d <local>\n",sizes[0],sizes[1]);
  
  PetscCall(MatCreate(PetscObjectComm((PetscObject)dm[0]),&preallocator)); 
  PetscCall(MatSetSizes(preallocator,sizes[0],sizes[1],sizes[2],sizes[3])); 
  PetscCall(MatSetType(preallocator,MATPREALLOCATOR)); 
  PetscCall(MatSetUp(preallocator)); 
  
  /* preallocate */
  for (i=0; i<ndm; i++) {
    DM row_dm,*cols_dm;
    
    row_dm  = dm[i];
    cols_dm = dm;
    
    PetscCall(_preallocate_coupled(preallocator,i,row_dm,ndm,cols_dm,offset,col_mask)); 
  }
  
  PetscCall(MatCreate(PetscObjectComm((PetscObject)preallocator),A)); 
  PetscCall(MatSetSizes(*A,sizes[0],sizes[1],sizes[2],sizes[3])); 
  PetscCall(MatSetType(*A,mtype)); 
  PetscCall(MatSetFromOptions(*A)); 
  PetscCall(MatSetUp(*A)); 
  PetscCall(MatPreallocatorPreallocate(preallocator,PETSC_TRUE,*A)); 
  
  PetscCall(MatDestroy(&preallocator)); 
  PetscCall(PetscFree(col_mask)); 
  PetscCall(PetscFree(M)); 
  PetscCall(PetscFree(N)); 
  PetscCall(PetscFree(m)); 
  PetscCall(PetscFree(n)); 
  PetscCall(PetscFree(offset)); 
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
PetscErrorCode FDPDECoupledCreateMatrix2(PetscInt ndm,DM dm[],PetscBool mask[],MatType mtype,Mat *A)
{
  PetscInt i,j,d;
  PetscInt *offset,*m,*n,*M,*N,sizes[] = {0,0,0,0};
  PetscBool *col_mask;
  Mat preallocator;
  
  PetscFunctionBegin;
  /* check all DMs are DMSTAG */
  for (d=0; d<ndm; d++) {
    PetscBool isstag;
    PetscCall(PetscObjectTypeCompare((PetscObject)dm[d],DMSTAG,&isstag)); 
    if (!isstag) SETERRQ(PetscObjectComm((PetscObject)dm[d]),PETSC_ERR_ARG_WRONG,"DM[%" PetscInt_FMT "] is not of type DMSTAG",d);
  }
  
  /* check sizes are consistent */
  {
    PetscInt Ni,Nj,jNi,jNj;
    
    PetscCall(DMStagGetGlobalSizes(dm[0],&Ni,&Nj,NULL)); 
    for (d=1; d<ndm; d++) {
      PetscCall(DMStagGetGlobalSizes(dm[d],&jNi,&jNj,NULL)); 
      if (Ni != jNi) SETERRQ(PetscObjectComm((PetscObject)dm[0]),PETSC_ERR_ARG_WRONG,"DM (Ni=%" PetscInt_FMT ") does not match size of DM[0] (Ni=%" PetscInt_FMT ")",Ni,jNi);
      if (Nj != jNj) SETERRQ(PetscObjectComm((PetscObject)dm[0]),PETSC_ERR_ARG_WRONG,"DM (Nj=%" PetscInt_FMT ") does not match size of DM[0] (Nj=%" PetscInt_FMT ")",Nj,jNj);
    }
  }
  
  /* determine global and local sizes */
  /* compute offsets for insertions */
  PetscCall(PetscCalloc1(ndm,&offset)); 
  PetscCall(PetscCalloc1(ndm,&col_mask)); 
  PetscCall(PetscCalloc1(ndm,&m)); 
  PetscCall(PetscCalloc1(ndm,&n)); 
  PetscCall(PetscCalloc1(ndm,&M)); 
  PetscCall(PetscCalloc1(ndm,&N)); 
  for (d=0; d<ndm; d++) {
    col_mask[d] = PETSC_FALSE;
  }
  for (d=0; d<ndm; d++) {
    Vec x;
    PetscInt size;
    DMCreateGlobalVector(dm[d],&x); 
    VecGetSize(x,&size);
    M[d] = size;
    N[d] = size;
    VecGetLocalSize(x,&size);
    m[d] = size;
    n[d] = size;
    VecDestroy(&x);
  }
  for (d=1; d<ndm; d++) {
    offset[d] = offset[d-1] + m[d-1];
  }
  for (d=0; d<ndm; d++) {
    sizes[0] += m[d];
    sizes[1] += n[d];
    sizes[2] += M[d];
    sizes[3] += N[d];
  }
  for (d=0; d<ndm; d++) {
    printf("[%d] MxN %d %d mxn %d %d\n",d,M[d],N[d],m[d],n[d]);
    printf("offset[%d] %d\n",d,offset[d]);
  }
  printf("sizes MxN %d %d <global>\n",sizes[2],sizes[3]);
  printf("sizes mxn %d %d <local>\n",sizes[0],sizes[1]);
  
  PetscCall(MatCreate(PetscObjectComm((PetscObject)dm[0]),&preallocator)); 
  PetscCall(MatSetSizes(preallocator,sizes[0],sizes[1],sizes[2],sizes[3])); 
  PetscCall(MatSetType(preallocator,MATPREALLOCATOR)); 
  PetscCall(MatSetUp(preallocator)); 
  
  /* preallocate */
  for (i=0; i<ndm; i++) {
    DM row_dm,*cols_dm;
    
    row_dm  = dm[i];
    cols_dm = dm;
    
    for (j=0; j<ndm; j++) {
      col_mask[j] = mask[i*ndm + j];
    }
    
    PetscCall(_preallocate_coupled(preallocator,i,row_dm,ndm,cols_dm,offset,col_mask)); 
  }
  
  PetscCall(MatCreate(PetscObjectComm((PetscObject)preallocator),A)); 
  PetscCall(MatSetSizes(*A,sizes[0],sizes[1],sizes[2],sizes[3])); 
  PetscCall(MatSetType(*A,mtype)); 
  PetscCall(MatSetFromOptions(*A)); 
  PetscCall(MatSetUp(*A)); 
  PetscCall(MatPreallocatorPreallocate(preallocator,PETSC_TRUE,*A)); 
  
  PetscCall(MatDestroy(&preallocator)); 
  PetscCall(PetscFree(col_mask)); 
  PetscCall(PetscFree(M)); 
  PetscCall(PetscFree(N)); 
  PetscCall(PetscFree(m)); 
  PetscCall(PetscFree(n)); 
  PetscCall(PetscFree(offset)); 
  
  PetscFunctionReturn(PETSC_SUCCESS);
}