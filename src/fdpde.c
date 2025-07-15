/* Finite Differences PDE (FD-PDE) object */

#include "fdpde.h"
#include "fdpde_composite.h"
#include "dmstagoutput.h"
#include "snes_picard.h"

const char *FDPDETypeNames[] = {
  "uninit",
  "stokes",
  "advdiff",
  "stokesdarcy2field",
  "stokesdarcy3field",
  "composite",
  "enthalpy"
};

// ---------------------------------------
// FDPDECreatePDEType declarations
// ---------------------------------------
PetscErrorCode FDPDECreate_Stokes(FDPDE fd);
PetscErrorCode FDPDECreate_StokesDarcy2Field(FDPDE fd);
PetscErrorCode FDPDECreate_StokesDarcy3Field(FDPDE fd);
PetscErrorCode FDPDECreate_AdvDiff(FDPDE fd);
PetscErrorCode FDPDECreate_Composite(FDPDE fd);
PetscErrorCode FDPDECreate_Enthalpy(FDPDE fd);
PetscErrorCode FDPDESetUp_Composite(FDPDE fd);

// ---------------------------------------
/*@ FDPDECreate - creates an object that will manage the discretization of a PDE using 
finite differences on a 2D DMStag object

Input Parameters:
comm - MPI communicator
nx,nz - global number of elements in x,z directions
xs,xe - start and end coordinate of domain (limits in x direction)
zs,ze - start and end coordinate of domain (limits in z direction)
type - FDPDEType type

Output Parameters:
_fd - the new FD-PDE object

Notes:
You must call FDPDESetUp() after this call, before using the FD-PDE object. Also, you must destroy the object with FDPDEDestroy().

Use: user
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDPDECreate"
PetscErrorCode FDPDECreate(MPI_Comm comm, PetscInt nx, PetscInt nz, 
                        PetscScalar xs, PetscScalar xe, PetscScalar zs, PetscScalar ze, 
                        FDPDEType type, FDPDE *_fd)
{
  FDPDE          fd;
  FDPDEOps       ops;
  
  PetscFunctionBegin;
  // Error checking
  if (!_fd) SETERRQ(comm,PETSC_ERR_ARG_NULL,"Must provide a valid (non-NULL) pointer for fd (arg 9)");
  
  if (nx <= 0) SETERRQ(comm,PETSC_ERR_ARG_OUTOFRANGE,"Dimension 1 (arg 2) provided for FD-PDE dmstag must be > 0. Found %D",nx);
  if (nz <= 0) SETERRQ(comm,PETSC_ERR_ARG_OUTOFRANGE,"Dimension 2 (arg 3) provided for FD-PDE dmstag must be > 0. Found %D",nz);
  
  if (xs >= xe) SETERRQ(comm,PETSC_ERR_ARG_OUTOFRANGE,"Invalid x-maximum (arg 5) provided. xe > xs.");
  if (zs >= ze) SETERRQ(comm,PETSC_ERR_ARG_OUTOFRANGE,"Invalid y-maximum (arg 7) provided. ze > zs");
  
  if (type == FDPDE_COMPOSITE) SETERRQ(comm,PETSC_ERR_ARG_WRONG,"Must use FDPDECreate2() with FDPDE_COMPOSITE");
  
  // Allocate memory
  PetscCall(PetscCalloc1(1,&fd));
  fd->comm = comm;

  // Save input variables
  fd->Nx = nx;
  fd->Nz = nz;
  fd->x0 = xs;
  fd->x1 = xe;
  fd->z0 = zs;
  fd->z1 = ze;

  fd->type = type;

  fd->dm_btype0 = DM_BOUNDARY_NONE;
  fd->dm_btype1 = DM_BOUNDARY_NONE;

  PetscCall(PetscMalloc1(1,&ops));
  PetscCall(PetscMemzero(ops, sizeof(struct _FDPDEOps))); 
  fd->ops = ops;

  // Initialize struct
  fd->dmstag  = NULL;
  fd->dmcoeff = NULL;
  fd->bclist = NULL;
  fd->x     = NULL;
  fd->xguess= NULL;
  fd->r     = NULL;
  fd->coeff = NULL;
  fd->J     = NULL;
  fd->snes  = NULL;
  fd->data  = NULL;
  fd->user_context = NULL;
  fd->setupcalled = PETSC_FALSE;
  fd->linearsolve = PETSC_FALSE;
  fd->output_solver_failure_report = PETSC_TRUE;
  fd->log_info = PETSC_FALSE;

  fd->description_bc = NULL;
  fd->description_coeff = NULL;
  fd->refcount++;
  *_fd = fd;
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
/*@
FDPDESetUp - sets up the data structures inside a FD-PDE object

Input Parameter:
fd - the FD-PDE object to setup

To change the snes prefix, one should call:
  FDPDESetUp(fd);
  FDPDEGetSNES(fd,&snes);
  SNESSetOptionsPrefix(snes);
  SNESSetFromOptions(snes);
  FDPDESolve(fd);

Use: user
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDPDESetUp"
PetscErrorCode FDPDESetUp(FDPDE fd)
{
  PetscFunctionBegin;
  if (fd->setupcalled) PetscFunctionReturn(PETSC_SUCCESS);

  // Set up structures needed for FD-PDE type
  switch (fd->type) {
    case FDPDE_UNINIT:
      SETERRQ(fd->comm,PETSC_ERR_ARG_TYPENOTSET,"Un-initialized type for FD-PDE");
      break;
    case FDPDE_STOKES:
      fd->ops->create = FDPDECreate_Stokes;
      break;
    case FDPDE_STOKESDARCY2FIELD:
      fd->ops->create = FDPDECreate_StokesDarcy2Field;
      break;
    case FDPDE_STOKESDARCY3FIELD:
      fd->ops->create = FDPDECreate_StokesDarcy3Field;
      break;
    case FDPDE_ADVDIFF:
      fd->ops->create = FDPDECreate_AdvDiff;
      break;
    case FDPDE_ENTHALPY:
      fd->ops->create = FDPDECreate_Enthalpy;
      break;
    case FDPDE_COMPOSITE:
    SETERRQ(fd->comm,PETSC_ERR_ARG_WRONGSTATE,"FDPDE_COMPOSITE should never enter here");
    break;
    default:
      SETERRQ(fd->comm,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown type of FD-PDE specified");
  }

  // Set individual FD-PDE type options
  PetscCall(fd->ops->create(fd)); 

  if ((!fd->dof0) && (!fd->dof1) && (!fd->dof2)) {
    SETERRQ(fd->comm,PETSC_ERR_ARG_NULL,"All FD-PDE dmstag DOFs are zero! Cannot create DMStag object required for FD-PDE.");
  }

  if ((!fd->dofc0) && (!fd->dofc1) && (!fd->dofc2)) {
    SETERRQ(fd->comm,PETSC_ERR_ARG_NULL,"All FD-PDE dmcoeff DOFs are zero! Cannot create DMStag coefficient object required for FD-PDE.");
  }

  // Create dms and vector - specific to FD-PDE
  PetscInt stencil_width = 2;
  PetscCall(DMStagCreate2d(fd->comm, fd->dm_btype0, fd->dm_btype1, fd->Nx, fd->Nz, 
            PETSC_DECIDE, PETSC_DECIDE, fd->dof0, fd->dof1, fd->dof2, 
            DMSTAG_STENCIL_BOX, stencil_width, NULL,NULL, &fd->dmstag)); 
  PetscCall(DMSetFromOptions(fd->dmstag)); 
  PetscCall(DMSetUp         (fd->dmstag)); 
  PetscCall(DMStagSetUniformCoordinatesProduct(fd->dmstag,fd->x0,fd->x1,fd->z0,fd->z1,0.0,0.0));

  // Create DMStag object for coefficients
  PetscCall(DMStagCreateCompatibleDMStag(fd->dmstag, fd->dofc0, fd->dofc1, fd->dofc2, 0, &fd->dmcoeff)); 
  PetscCall(DMSetUp(fd->dmcoeff)); 
  PetscCall(DMStagSetUniformCoordinatesProduct(fd->dmcoeff,fd->x0,fd->x1,fd->z0,fd->z1,0.0,0.0));

  // Create global vectors
  PetscCall(DMCreateGlobalVector(fd->dmstag,&fd->x));
  PetscCall(VecDuplicate(fd->x,&fd->r));
  PetscCall(VecDuplicate(fd->x,&fd->xguess));
  PetscCall(DMCreateGlobalVector(fd->dmcoeff,&fd->coeff)); 

  // Create BClist object
  PetscCall(DMStagBCListCreate(fd->dmstag,&fd->bclist));

  // Create Jacobian
  if (!fd->ops->create_jacobian) SETERRQ(fd->comm,PETSC_ERR_ARG_NULL,"No method to create the Jacobian was provided. The FD-PDE implementation constructor is expected to set this function pointer");
  PetscCall(fd->ops->create_jacobian(fd,&fd->J)); 
  if (!fd->J) SETERRQ(fd->comm,PETSC_ERR_ARG_NULL,"Jacobian matrix for FD-PDE is NULL. The FD-PDE implementation for create_jacobian is required to create a valid Mat");

  // Create SNES - nonlinear solver context
  PetscCall(SNESCreate(fd->comm,&fd->snes)); 
  PetscCall(SNESSetDM(fd->snes,fd->dmstag)); 

  if (!fd->x) SETERRQ(fd->comm,PETSC_ERR_ARG_NULL,"Solution vector for FD-PDE is NULL.");
  PetscCall(SNESSetSolution(fd->snes,fd->x));  // for FD colouring to function correctly

  // Set function evaluation routine
  if (!fd->ops->form_function) SETERRQ(fd->comm,PETSC_ERR_ARG_NULL,"No residual evaluation routine has been set.");
  PetscCall(SNESSetFunction(fd->snes, fd->r, fd->ops->form_function, (void*)fd)); 

  // Set Jacobian
  if (fd->ops->form_jacobian) {
    PetscCall(SNESSetJacobian(fd->snes, fd->J, fd->J, fd->ops->form_jacobian, (void*)fd)); 
  } else {
    PetscCall(SNESSetJacobian(fd->snes, fd->J, fd->J, SNESComputeJacobianDefaultColor, NULL)); 
  }

  /* call setup for additional setup and return if defined */
  if (fd->ops->setup) { PetscCall(fd->ops->setup(fd));  }

  fd->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
/*@
FDPDEDestroy - destroy an FD-PDE object

Input Parameter:
fd - the FD-PDE object to destroy

Use: user
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDPDEDestroy"
PetscErrorCode FDPDEDestroy(FDPDE *_fd)
{
  FDPDE fd;
  PetscFunctionBegin;

  // Return if no object
  if (!_fd) PetscFunctionReturn(PETSC_SUCCESS);
  if (!*_fd) PetscFunctionReturn(PETSC_SUCCESS);
  fd = *_fd;
  if (fd->refcount-1 > 0) {
    fd->refcount--;
    *_fd = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  
  // Destroy FD-PDE specific objects
  if (fd->ops->destroy) { PetscCall(fd->ops->destroy(fd));  }
  fd->data = NULL;

  // Destroy FD-PDE objects
  PetscCall(DMStagBCListDestroy(&fd->bclist));
  PetscCall(PetscFree(fd->ops));
  PetscCall(VecDestroy(&fd->x));
  PetscCall(VecDestroy(&fd->r));
  PetscCall(VecDestroy(&fd->coeff));
  PetscCall(VecDestroy(&fd->xguess));
  PetscCall(MatDestroy(&fd->J));
  PetscCall(SNESDestroy(&fd->snes));

  PetscCall(DMDestroy(&fd->dmcoeff)); 
  PetscCall(DMDestroy(&fd->dmstag)); 

  fd->user_context = NULL;

  PetscCall(PetscFree(fd->description));
  PetscCall(PetscFree(fd->description_bc));
  PetscCall(PetscFree(fd->description_coeff));
  PetscCall(PetscFree(fd));
  *_fd = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
/*@
FDPDEView - ASCII print of FD-PDE info structure on PETSC_COMM_WORLD 

Input Parameter:
fd - the FD-PDE object to view

Use: user
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDPDEView"
PetscErrorCode FDPDEView(FDPDE fd)
{
  PetscFunctionBegin;

  PetscPrintf(fd->comm,"FDPDEView:\n");
  PetscPrintf(fd->comm,"  # FD-PDE type: %s\n",FDPDETypeNames[(int)fd->type]);
  
  PetscPrintf(fd->comm,"  # FD-PDE description:\n");
  PetscPrintf(fd->comm,"    %s\n",fd->description);

  PetscPrintf(fd->comm,"  # Coefficient description:\n");
  PetscPrintf(fd->comm,"    %s\n",fd->description_coeff);

  PetscPrintf(fd->comm,"  # BC description:\n");
  PetscPrintf(fd->comm,"    %s\n",fd->description_bc);

  PetscPrintf(fd->comm,"  # global size elements: %D (x-dir) %D (z-dir)\n",fd->Nx,fd->Nz);

  if (fd->dmstag) {
    PetscPrintf(fd->comm,"  # dmstag: %D (vertices) %D (faces) %D (elements)\n",fd->dof0,fd->dof1,fd->dof2);
  } else {
    PetscPrintf(fd->comm,"  # dmstag: not available\n");
  }
  if (fd->dmcoeff) {
    PetscPrintf(fd->comm,"  # dmcoeff: %D (vertices) %D (faces) %D (elements)\n",fd->dofc0,fd->dofc1,fd->dofc2);
  } else {
    PetscPrintf(fd->comm,"  # dmcoeff: not available\n");
  }
  if (fd->setupcalled) PetscPrintf(fd->comm,"  # FDPDESetUp: TRUE \n\n");
  else PetscPrintf(fd->comm,"  # FDPDESetUp: FALSE \n\n");

  // view FD-PDE specific info
  if (fd->ops->view) { PetscCall(fd->ops->view(fd));  }

  // view BC list
  //PetscCall(DMStagBCListView(fd->bclist));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
/*@
FDPDEGetDM - retrieves the main DMStag (associated with the solution) from the FD-PDE object. 

Input Parameter:
fd - the FD-PDE object

Output Parameter:
dm - the DM object

Notes:
Reference count on dm is incremented. User must call DMDestroy() on dm to free the space.

Use: user
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDPDEGetDM"
PetscErrorCode FDPDEGetDM(FDPDE fd, DM *dm)
{
  PetscFunctionBegin;

  if (!fd->dmstag) SETERRQ(fd->comm,PETSC_ERR_ORDER,"DMStag for FD-PDE not provided - Call FDPDESetUp()");
  *dm = fd->dmstag;
  PetscCall(PetscObjectReference((PetscObject)fd->dmstag));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
/*@
FDPDEGetCoefficient - retrieves the coefficient DMStag and Vector from the FD-PDE object. 

Input Parameter:
fd - the FD-PDE object

Output Parameters (optional):
dmcoeff - the DM object
coeff - the vector

Use: user
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDPDEGetCoefficient"
PetscErrorCode FDPDEGetCoefficient(FDPDE fd, DM *dmcoeff, Vec *coeff)
{
  PetscFunctionBegin;
  if (dmcoeff) *dmcoeff = fd->dmcoeff;
  if (coeff) *coeff = fd->coeff;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
/*@
FDPDESetFunctionBCList - set an evaluation function for boundary conditions 

Input Parameter:
fd - the FD-PDE object
evaluate - name of the evaluation function for boundary conditions
description - user can provide a description for BC
data - user context to be passed for evaluation (can be NULL)

Use: user
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDPDESetFunctionBCList"
PetscErrorCode FDPDESetFunctionBCList(FDPDE fd, PetscErrorCode (*evaluate)(DM,Vec,DMStagBCList,void*), const char description[], void *data)
{
  PetscFunctionBegin;

  fd->bclist->evaluate = evaluate;
  fd->bclist->data = data;

  /* free any existing name set by previous call */
  if (fd->description_bc) { PetscCall(PetscFree(fd->description_bc));  }
  if (description) { PetscCall(PetscStrallocpy(description,&fd->description_bc));  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
/*@
FDPDESetFunctionCoefficient - set an evaluation function for FD-PDE coefficients

Input Parameter:
fd - the FD-PDE object
form_coefficient - name of the evaluation function for coefficients
description - user can provide a description for coefficients
data - user context to be passed for evaluation (can be NULL)

Use: user
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDPDESetFunctionCoefficient"
PetscErrorCode FDPDESetFunctionCoefficient(FDPDE fd, PetscErrorCode (*form_coefficient)(FDPDE fd,DM,Vec,DM,Vec,void*), const char description[], void *data)
{
  PetscFunctionBegin;

  fd->ops->form_coefficient = form_coefficient;
  fd->user_context = data;

  /* free any existing name set by previous call */
  if (fd->description_coeff) { PetscCall(PetscFree(fd->description_coeff));  }
  if (description) { PetscCall(PetscStrallocpy(description,&fd->description_coeff));  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FDPDESetFunctionCoefficientSplit(FDPDE fd, PetscErrorCode (*form_coefficient)(FDPDE fd,DM,Vec,Vec,DM,Vec,void*), const char description[], void *data)
{
  PetscFunctionBegin;
  
  fd->ops->form_coefficient_split = form_coefficient;
  fd->user_context = data;
  
  /* free any existing name set by previous call */
  if (fd->description_coeff) { PetscCall(PetscFree(fd->description_coeff));  }
  if (description) { PetscCall(PetscStrallocpy(description,&fd->description_coeff));  }
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
/*@
FDPDEGetSolution - retrieves the solution vector from the FD-PDE object. 

Input Parameter:
fd - the FD-PDE object

Output Parameter:
x - the solution vector

Notes:
Reference count on x is incremented. User must call VecDestroy() on x.

Use: user
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDPDEGetSolution"
PetscErrorCode FDPDEGetSolution(FDPDE fd, Vec *x)
{
  PetscFunctionBegin;
  if (x) {
    *x = fd->x;
    PetscCall(PetscObjectReference((PetscObject)fd->x));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
/*@
FDPDEGetSNES - retrieves the SNES object from the FD-PDE object. 

Input Parameter:
fd - the FD-PDE object

Output Parameter:
snes - the snes object

Use: user
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDPDEGetSNES"
PetscErrorCode FDPDEGetSNES(FDPDE fd, SNES *snes)
{
  PetscFunctionBegin;
  if (snes) *snes = fd->snes;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
/*@
FDPDEGetDMStagBCList() - retrieves the DMStagBCList object from the FD-PDE object. 

Input Parameter:
fd - the FD-PDE object

Output Parameter:
list - the DMStagBCList object

Use: user
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDPDEGetDMStagBCList"
PetscErrorCode FDPDEGetDMStagBCList(FDPDE fd, DMStagBCList *list)
{
  PetscFunctionBegin;
  if (list) *list = fd->bclist;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
/*@
FDPDEGetSolutionGuess() - retrieves the xguess vector from the FD-PDE object. 

Input Parameter:
fd - the FD-PDE object

Output Parameter:
xguess - the solution guess vector

Notes:
Reference count on xguess is incremented. User must call VecDestroy() on xguess.

Use: user
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDPDEGetSolutionGuess"
PetscErrorCode FDPDEGetSolutionGuess(FDPDE fd, Vec *xguess)
{
  PetscFunctionBegin;
  if (xguess) {
    *xguess = fd->xguess;
    PetscCall(PetscObjectReference((PetscObject)fd->xguess));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FDPDESolveReport_Failure(FDPDE fd,PetscViewer viewer)
{
  char                filename[PETSC_MAX_PATH_LEN],filename_bin[PETSC_MAX_PATH_LEN];
  const char          *prefix;
  Vec                 F,X,dX;
  SNESConvergedReason reason;
  PetscViewer         fview;
  PetscBool           out_python = PETSC_FALSE;
  
  PetscFunctionBegin;
  PetscCall(SNESGetOptionsPrefix(fd->snes,&prefix));
  PetscCall(SNESGetConvergedReason(fd->snes,&reason)); 
  PetscPrintf(fd->comm,"=====================================================================\n");
  if (prefix) PetscPrintf(fd->comm,"====  SNES (prefix = %s) has failed to converge\n",prefix);
  else PetscPrintf(fd->comm,"====  SNES has failed to converge\n");
  
  if (viewer != PETSC_VIEWER_STDOUT_WORLD) {
    const char *vname;
    PetscCall(PetscViewerFileGetName(viewer,&vname));
    PetscPrintf(fd->comm,"====  Please inspect the following file to diagnose the problem\n");
    PetscPrintf(fd->comm,"====  %s\n",vname);
  }
  PetscPrintf(fd->comm,"=====================================================================\n");

  PetscCall(PetscOptionsGetBool(NULL,NULL,"-python_snes_failed_report",&out_python,NULL));
  
  PetscCall(SNESGetSolution(fd->snes,&X));
  PetscCall(SNESGetSolutionUpdate(fd->snes,&dX));
  PetscCall(SNESGetFunction(fd->snes,&F,NULL,NULL));
  PetscCall(SNESComputeFunction(fd->snes,X,F));
  
  PetscViewerASCIIPrintf(viewer,"[SNES failure summary]\n");
  PetscViewerASCIIPushTab(viewer);
  PetscViewerASCIIPrintf(viewer,"reason: %D (error code) ->\n",(PetscInt)reason);
  // PetscCall(SNESReasonView(fd->snes,viewer));
  PetscCall(SNESConvergedReasonView(fd->snes,viewer));

  {
    PetscInt its;
    PetscCall(SNESGetIterationNumber(fd->snes,&its));
    PetscViewerASCIIPrintf(viewer,"iterations performed: %D\n",its);
  }
  PetscViewerASCIIPopTab(viewer);
  
  PetscViewerASCIIPrintf(viewer,"[residual summary]\n");
  {
    PetscReal val;
    PetscInt loc;
    PetscViewerASCIIPushTab(viewer);
    PetscCall(VecMax(F,&loc,&val));
    PetscViewerASCIIPrintf(viewer,"max(F) %+1.12e [location %D]\n",val,loc);
    PetscCall(VecMin(F,&loc,&val));
    PetscViewerASCIIPrintf(viewer,"min(F) %+1.12e [location %D]\n",val,loc);
    PetscViewerASCIIPopTab(viewer);
  }
  
  {
    PetscInt  i,n,*its = NULL;
    PetscReal *nrm = NULL;
    PetscCall(SNESGetConvergenceHistory(fd->snes,&nrm,&its,&n));
    PetscViewerASCIIPrintf(viewer,"[convergence history]\n");
    PetscViewerASCIIPushTab(viewer);
    if (nrm && its) {
      PetscViewerASCIIPrintf(viewer,"#SNES its. ||F||_2            #KSP its.\n");
      for (i=0; i<n; i++) {
        PetscViewerASCIIPrintf(viewer,"%.4D       %1.12e %.4D\n",i,nrm[i],its[i]);
      }
    } else {
      PetscViewerASCIIPrintf(viewer,"nonlinear residual history is unavailable - must call SNESSetConvergenceHistory() to activate logging\n");
    }
    PetscViewerASCIIPopTab(viewer);
  }
  
  PetscViewerASCIIPrintf(viewer,"[SNES view]\n");
  PetscViewerASCIIPushTab(viewer);
  PetscCall(SNESView(fd->snes,viewer));
  PetscViewerASCIIPopTab(viewer);
  
  // output residual
  if (prefix) PetscSNPrintf(filename,PETSC_MAX_PATH_LEN-1,"%ssnes_failure_F-%D",prefix,fd->solves_performed);
  else PetscSNPrintf(filename,PETSC_MAX_PATH_LEN-1,"snes_failure_F-%D",fd->solves_performed);
  PetscViewerASCIIPrintf(viewer,"[residual file]\n");
  PetscViewerASCIIPushTab(viewer);
  PetscViewerASCIIPrintf(viewer,"filename: %s\n",filename);
  PetscViewerASCIIPopTab(viewer);
  if (out_python) { PetscCall(DMStagViewBinaryPython(fd->dmstag,F,filename)); }
  else {
    PetscSNPrintf(filename_bin,PETSC_MAX_PATH_LEN-1,"%s.vec",filename);
    /*PetscCall(PetscViewerASCIIOpen(fd->comm,filename,&fview));*/
    PetscCall(PetscViewerBinaryOpen(fd->comm,filename_bin,FILE_MODE_WRITE,&fview));
    PetscCall(VecView(F,fview));
    PetscCall(PetscViewerDestroy(&fview));
  }
  
  
  // output solution
  if (prefix) PetscSNPrintf(filename,PETSC_MAX_PATH_LEN-1,"%ssnes_failure_X-%D",prefix,fd->solves_performed);
  else PetscSNPrintf(filename,PETSC_MAX_PATH_LEN-1,"snes_failure_X-%D",fd->solves_performed);
  PetscViewerASCIIPrintf(viewer,"[solution file]\n");
  PetscViewerASCIIPushTab(viewer);
  PetscViewerASCIIPrintf(viewer,"filename: %s\n",filename);
  PetscViewerASCIIPopTab(viewer);
  if (out_python) { PetscCall(DMStagViewBinaryPython(fd->dmstag,X,filename)); }
  else {
    PetscSNPrintf(filename_bin,PETSC_MAX_PATH_LEN-1,"%s.vec",filename);
    /*PetscCall(PetscViewerASCIIOpen(fd->comm,filename,&fview));*/
    PetscCall(PetscViewerBinaryOpen(fd->comm,filename_bin,FILE_MODE_WRITE,&fview));
    PetscCall(VecView(X,fview));
    PetscCall(PetscViewerDestroy(&fview));
  }

  // output solution increment
  if (prefix) PetscSNPrintf(filename,PETSC_MAX_PATH_LEN-1,"%ssnes_failure_dX-%D",prefix,fd->solves_performed);
  else PetscSNPrintf(filename,PETSC_MAX_PATH_LEN-1,"snes_failure_dX-%D",fd->solves_performed);
  PetscViewerASCIIPrintf(viewer,"[solution correction file]\n");
  PetscViewerASCIIPushTab(viewer);
  PetscViewerASCIIPrintf(viewer,"filename: %s\n",filename);
  PetscViewerASCIIPopTab(viewer);
  if (out_python) { PetscCall(DMStagViewBinaryPython(fd->dmstag,dX,filename)); }
  else {
    PetscSNPrintf(filename_bin,PETSC_MAX_PATH_LEN-1,"%s.vec",filename);
    /*PetscCall(PetscViewerASCIIOpen(fd->comm,filename,&fview));*/
    PetscCall(PetscViewerBinaryOpen(fd->comm,filename_bin,FILE_MODE_WRITE,&fview));
    PetscCall(VecView(dX,fview));
    PetscCall(PetscViewerDestroy(&fview));
  }

  // output coefficient
  if (prefix) PetscSNPrintf(filename,PETSC_MAX_PATH_LEN-1,"%ssnes_failure_fdpde_coeff-%D",prefix,fd->solves_performed);
  else PetscSNPrintf(filename,PETSC_MAX_PATH_LEN-1,"snes_failure_fdpde_coeff-%D",fd->solves_performed);
  PetscViewerASCIIPrintf(viewer,"[FDPDE coefficient file]\n");
  PetscViewerASCIIPushTab(viewer);
  PetscViewerASCIIPrintf(viewer,"filename: %s\n",filename);
  PetscViewerASCIIPopTab(viewer);
  if (out_python) { PetscCall(DMStagViewBinaryPython(fd->dmcoeff,fd->coeff,filename)); }
  else {
    PetscSNPrintf(filename_bin,PETSC_MAX_PATH_LEN-1,"%s.vec",filename);
    PetscCall(PetscViewerBinaryOpen(fd->comm,filename_bin,FILE_MODE_WRITE,&fview));
    PetscCall(VecView(fd->coeff,fview));
    PetscCall(PetscViewerDestroy(&fview));
  }

  PetscViewerASCIIPrintf(viewer,"[DMStag summary]\n");
  PetscViewerASCIIPushTab(viewer);
  PetscCall(DMView(fd->dmstag,viewer));
  PetscViewerASCIIPopTab(viewer);

  PetscViewerASCIIPrintf(viewer,"[DMCoeff summary]\n");
  PetscViewerASCIIPushTab(viewer);
  PetscCall(DMView(fd->dmcoeff,viewer));
  PetscViewerASCIIPopTab(viewer);
  
  PetscViewerASCIIPrintf(viewer,"[PDE summary]\n");
  PetscViewerASCIIPushTab(viewer);
  PetscViewerASCIIPrintf(viewer,"pde: %s\n",FDPDETypeNames[(int)fd->type]);
  PetscViewerASCIIPrintf(viewer,"description: %s\n",fd->description);
  PetscViewerASCIIPopTab(viewer);
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
/*@
FDPDESolve - solve the associated system of equations contained in an FD-PDE object.

Input Parameter:
fd - the FD-PDE object

Output Parameter:
converged - value is PETSC_TRUE if SNESSolve converged (optional)

Functionality:
- calls SNESSetFromOptions() so any PETSc SNES options should be set before
- forms initial guess (copy xguess->x). xguess can be solution from previous time step or it can be specifically set 
by user by calling FDPDEGetSolutionGuess(fd,&xguess)
- solves and returns a boolean if converged/not converged
- if converged, copy new solution to xguess

Use: user
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDPDESolve"
PetscErrorCode FDPDESolve(FDPDE fd, PetscBool *converged)
{
  SNESConvergedReason reason;
  PetscFunctionBegin;

  if (!fd->setupcalled) SETERRQ(fd->comm,PETSC_ERR_ORDER,"User must call FDPDESetUp() first!");

  // Overwrite default options from command line
  PetscCall(SNESSetFromOptions(fd->snes)); 

  /* force abort of application if convergence fails - too brutal and does not let us catch and report when an error occurs */
  /*PetscCall(SNESSetErrorIfNotConverged(fd->snes,PETSC_TRUE));*/
  
  /* Activate a logger which records norm of F and number of KSP iterations at each SNES iteration */
  {
    PetscInt maxit, *its = NULL;
    PetscReal *a = NULL;
    
    PetscCall(SNESGetTolerances(fd->snes,NULL,NULL,NULL,&maxit,NULL));
    PetscCall(SNESGetConvergenceHistory(fd->snes,&a,&its,NULL));
    PetscCall(SNESSetConvergenceHistory(fd->snes,a,its,maxit+1,PETSC_TRUE));
  }
  
  // Copy initial guess to solution
  PetscCall(VecCopy(fd->xguess,fd->x));

  // Solve the non-linear system
  PetscCall(SNESSolve(fd->snes,0,fd->x));             
  PetscCall(SNESGetConvergedReason(fd->snes,&reason)); 

  if ((reason < 0) && (fd->output_solver_failure_report)) {
    const char  *prefix;
    char        filename[PETSC_MAX_PATH_LEN];
    PetscViewer viewer;
    
    /*// Example demonstrating usage of dumping report to stdout
    viewer = PETSC_VIEWER_STDOUT_WORLD
    PetscCall(FDPDESolveReport_Failure(fd,viewer));
    */
     
    PetscCall(SNESGetOptionsPrefix(fd->snes,&prefix));
    if (prefix) PetscSNPrintf(filename,PETSC_MAX_PATH_LEN-1,"%ssnes_failure-%D.report",prefix,fd->solves_performed);
    else PetscSNPrintf(filename,PETSC_MAX_PATH_LEN-1,"snes_failure-%D.report",fd->solves_performed);
    PetscCall(PetscViewerASCIIOpen(fd->comm,filename,&viewer));
    PetscCall(FDPDESolveReport_Failure(fd,viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }
  
  // Analyze convergence
  if (reason > 0) { // reason = 0 implies SNES_CONVERGED_ITERATING (which can never be true after SNESSolve executes)
    // converged - copy initial guess for next timestep
    PetscCall(VecCopy(fd->x, fd->xguess)); 
  }

  if (converged) {
    *converged = PETSC_TRUE;
    if (reason < 0) *converged = PETSC_FALSE;
  }
  fd->solves_performed++;
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FDPDESolvePicard(FDPDE fd, PetscBool *converged)
{
  SNESConvergedReason reason;
  SNES                snes_picard;
  Mat                 J;
  DM                  dmref,dm;
  
  PetscFunctionBegin;
  if (!fd->setupcalled) SETERRQ(fd->comm,PETSC_ERR_ORDER,"Must call FDPDESetUp() first!");
  if (!fd->ops->form_function_split) SETERRQ(fd->comm,PETSC_ERR_SUP,"FDPDE does not support split residual evaluation. Require that fd->ops->form_function_split() be defined");
  if (!fd->ops->form_coefficient_split) SETERRQ(fd->comm,PETSC_ERR_SUP,"FDPDE does not support split residual evaluation. Require that fd->ops->form_coefficient_split() be defined");
  
  PetscCall(VecCopy(fd->xguess,fd->x));

  PetscCall(FDPDEGetDM(fd,&dmref));
  PetscCall(DMClone(dmref,&dm));
  
  PetscCall(fd->ops->create_jacobian(fd,&J));
  
  PetscCall(SNESCreate(fd->comm,&snes_picard));
  PetscCall(SNESSetOptionsPrefix(snes_picard,"p_"));
  PetscCall(SNESSetDM(snes_picard,dm)); /* attach a clone of the DM stag - see note on manpage for SNESSetDM() */
  PetscCall(SNESSetSolution(snes_picard,fd->x)); // for FD colouring to function correctly
  
  PetscCall(SNESSetFunction(snes_picard,fd->r,SNESPicardComputeFunctionDefault,(void*)fd));
  
  PetscCall(SNESSetJacobian(snes_picard,J,J,SNESComputeJacobianDefaultColor,NULL));
  
  PetscCall(SNESSetType(snes_picard,SNESPICARDLS));
  PetscCall(SNESSetFromOptions(snes_picard));
  PetscCall(SNESSetUp(snes_picard));

  PetscCall(SNESPicardLSSetSplitFunction(snes_picard,fd->r,fd->ops->form_function_split));

  {
    Vec x2;
    
    PetscCall(SNESPicardLSGetAuxillarySolution(snes_picard,&x2));
    PetscCall(VecCopy(fd->x,x2));
  }

  PetscCall(SNESSolve(snes_picard,0,fd->x));
  
  PetscCall(VecCopy(fd->x, fd->xguess)); 
  
  PetscCall(SNESGetConvergedReason(snes_picard,&reason)); 
  if (converged) {
    *converged = PETSC_TRUE;
    if (reason < 0) *converged = PETSC_FALSE;
  }

  PetscCall(SNESDestroy(&snes_picard));
  PetscCall(DMDestroy(&dm));
  PetscCall(MatDestroy(&J));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
/*@
FDPDEGetCoordinatesArrayDMStag - retrieve the 1D coordinate arrays of the DMStag (system of equations) inside an FD-PDE object

Input parameter:
fd - the FD-PDE object

Output parameters:
cx - 1D array containing x-coordinates
cz - 1D array containing z-coordinates

Use: user
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDPDEGetCoordinatesArrayDMStag"
PetscErrorCode FDPDEGetCoordinatesArrayDMStag(FDPDE fd,PetscScalar ***cx, PetscScalar ***cz)
{
  DM             dmCoord;
  PetscScalar    **coordx,**coordz;
  PetscFunctionBegin;

  if (!fd->setupcalled) SETERRQ(fd->comm,PETSC_ERR_ORDER,"User must call FDPDESetUp() first!");
  if (!fd->dmstag) SETERRQ(fd->comm,PETSC_ERR_ARG_NULL,"System of FD-PDE not provided. Call FDPDESetUp() first!");
  if (!cx) SETERRQ(fd->comm,PETSC_ERR_ARG_NULL,"Arg 2 (cx) cannot be NULL");
  if (!cz) SETERRQ(fd->comm,PETSC_ERR_ARG_NULL,"Arg 3 (cz) cannot be NULL");
  PetscCall(DMGetCoordinateDM(fd->dmstag,&dmCoord));
  if (!dmCoord) SETERRQ(PetscObjectComm((PetscObject)fd->dmstag),PETSC_ERR_ARG_WRONGSTATE,"DMStag does not have a coordinate DM");
  {
    PetscBool isProduct;
    DMType    dmType;
    PetscCall(DMGetType(dmCoord,&dmType));
    PetscCall(PetscStrcmp(DMPRODUCT,dmType,&isProduct));
    if (!isProduct) SETERRQ(PetscObjectComm((PetscObject)fd->dmstag),PETSC_ERR_SUP,"Implementation requires coordinate DM is of type DMPRODUCT");
  }
  PetscCall(DMStagGetProductCoordinateArraysRead(fd->dmstag,&coordx,&coordz,NULL));
  *cx = coordx;
  *cz = coordz;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
/*@
FDPDERestoreCoordinatesArrayDMStag - restore the 1D coordinate arrays of the DMStag (system of equations) inside an FD-PDE object

Input parameter:
fd - the FD-PDE object
cx - 1D array containing x-coordinates
cz - 1D array containing z-coordinates

Notes:
Must be called after FDPDEGetCoordinatesArrayDMStag() and will update the coordinates of the dmcoeff and BCs.

Use: user
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDPDERestoreCoordinatesArrayDMStag"
PetscErrorCode FDPDERestoreCoordinatesArrayDMStag(FDPDE fd,PetscScalar **cx, PetscScalar **cz)
{
  DM             dm, dmcoeff;
  PetscScalar    **coordx,**coordz;
  PetscScalar    xprev, xnext, xcenter, zprev, znext, zcenter;
  PetscInt       dof0, dof1, dof2, dofc0, dofc1, dofc2;
  PetscInt       iprev=-1,inext=-1,icenter=-1,iprevc=-1,inextc=-1,icenterc=-1;
  PetscInt       i, j, sx, sz, nx, nz;
  PetscFunctionBegin;

  if (!fd->setupcalled) SETERRQ(fd->comm,PETSC_ERR_ORDER,"User must call FDPDESetUp() first!");
  if (!fd->dmstag) SETERRQ(fd->comm,PETSC_ERR_ARG_NULL,"System of FD-PDE not provided. Call FDPDESetUp() first!");
  if (!cx) SETERRQ(fd->comm,PETSC_ERR_ARG_NULL,"Arg 2 (cx) cannot be NULL");
  if (!cz) SETERRQ(fd->comm,PETSC_ERR_ARG_NULL,"Arg 3 (cz) cannot be NULL");

  dm      = fd->dmstag;
  dmcoeff = fd->dmcoeff;

  // Update dmcoord coordinates
  PetscCall(DMStagGetDOF(dm,&dof0,&dof1,&dof2,NULL));
  PetscCall(DMStagGetDOF(dmcoeff,&dofc0,&dofc1,&dofc2,NULL));

  PetscCall(DMStagGetProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL));

  if (dof2) {PetscCall(DMStagGetProductCoordinateLocationSlot(dm,DMSTAG_ELEMENT,&icenter));} 
  if (dof0 || dof1) { 
    PetscCall(DMStagGetProductCoordinateLocationSlot(dm,DMSTAG_LEFT,&iprev));
    PetscCall(DMStagGetProductCoordinateLocationSlot(dm,DMSTAG_RIGHT,&inext));
  } 

  if (dofc2) {PetscCall(DMStagGetProductCoordinateLocationSlot(dmcoeff,DMSTAG_ELEMENT,&icenterc));} 
  if (dofc0 || dofc1) { 
    PetscCall(DMStagGetProductCoordinateLocationSlot(dmcoeff,DMSTAG_LEFT,&iprevc));
    PetscCall(DMStagGetProductCoordinateLocationSlot(dmcoeff,DMSTAG_RIGHT,&inextc));
  } 

  PetscCall(DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL)); 

  for (i = sx; i<sx+nx; i++) {
    // dmstag
    if ((dof0) && (!dof1) && (!dof2)) { // dmstag only vertex
      xprev   = cx[i][iprev]; 
      xnext   = cx[i][inext];
      xcenter = (xprev+xnext)*0.5;
    } else if ((!dof0) && (!dof1) && (dof2)) { // dmstag only element
      xprev   = (cx[i][icenter]+cx[i-1][icenter])*0.5; 
      xnext   = (cx[i][icenter]+cx[i+1][icenter])*0.5;
      xcenter = cx[i][icenter];
    } else { // dmstag all 
      xprev   = cx[i][iprev]; 
      xnext   = cx[i][inext];
      xcenter = cx[i][icenter];
    }
    // dmcoeff
    if ((dofc0) && (!dofc1) && (!dofc2)) { // only vertex
      coordx[i][iprevc] = xprev;
      coordx[i][inextc] = xnext; 
    } else if ((!dofc0) && (!dofc1) && (dofc2)) { // only element
      coordx[i][icenterc] = xcenter; 
    } else { // all 
      coordx[i][iprevc] = xprev;
      coordx[i][inextc] = xnext; 
      coordx[i][icenterc] = xcenter; 
    }
  }

  for (j = sz; j<sz+nz; j++) {
    // dmstag
    if ((dof0) && (!dof1) && (!dof2)) { // dmstag only vertex
      zprev   = cz[j][iprev]; 
      znext   = cz[j][inext];
      zcenter = (zprev+znext)*0.5;
    } else if ((!dof0) && (!dof1) && (dof2)) { // dmstag only element
      zprev   = (cz[j][icenter]+cz[j-1][icenter])*0.5; 
      znext   = (cz[j][icenter]+cz[j+1][icenter])*0.5;
      zcenter = cz[j][icenter];
    } else { // dmstag all 
      zprev   = cz[j][iprev]; 
      znext   = cz[j][inext];
      zcenter = cz[j][icenter];
    }
    // dmcoeff
    if ((dofc0) && (!dofc1) && (!dofc2)) { // only vertex
      coordz[j][iprevc] = zprev;
      coordz[j][inextc] = znext; 
    } else if ((!dofc0) && (!dofc1) && (dofc2)) { // only element
      coordz[j][icenterc] = zcenter; 
    } else { // all 
      coordz[j][iprevc] = zprev;
      coordz[j][inextc] = znext; 
      coordz[j][icenterc] = zcenter; 
    }
  }

  // Restore coordinates
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL));
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm,&cx,&cz,NULL));

  // update coords of BCs
  PetscCall(DMStagBCListSetupCoordinates(fd->bclist));

  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "FDPDECreate2"
PetscErrorCode FDPDECreate2(MPI_Comm comm,FDPDE *_fd)
{
  FDPDE          fd;
  FDPDEOps       ops;

  PetscFunctionBegin;
  if (!_fd) SETERRQ(comm,PETSC_ERR_ARG_NULL,"Must provide a valid (non-NULL) pointer for fd (arg 2)");
  PetscCall(PetscCalloc1(1,&fd));
  fd->comm = comm;
  fd->type = FDPDE_UNINIT;
  PetscCall(PetscCalloc1(1,&fd->ops));
  fd->dmstag  = NULL;
  fd->dmcoeff = NULL;
  fd->bclist  = NULL;
  fd->x       = NULL;
  fd->xguess  = NULL;
  fd->r       = NULL;
  fd->coeff   = NULL;
  fd->J       = NULL;
  fd->snes    = NULL;
  fd->data    = NULL;
  fd->user_context      = NULL;
  fd->setupcalled       = PETSC_FALSE;
  fd->description_bc    = NULL;
  fd->description_coeff = NULL;
  fd->refcount++;
  *_fd = fd;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "FDPDESetType"
PetscErrorCode FDPDESetType(FDPDE fd,FDPDEType type)
{
  PetscFunctionBegin;
  if (fd->type != FDPDE_UNINIT) {
    if (fd->ops->destroy) { PetscCall(fd->ops->destroy(fd));  }
  }
  fd->type = type;
  switch (fd->type) {
    case FDPDE_UNINIT:
    SETERRQ(fd->comm,PETSC_ERR_ARG_TYPENOTSET,"Un-initialized type for FD-PDE");
    break;
    case FDPDE_STOKES:
    fd->ops->create = FDPDECreate_Stokes;
    fd->ops->setup = NULL;
    break;
    case FDPDE_ADVDIFF:
    fd->ops->create = FDPDECreate_AdvDiff;
    fd->ops->setup = NULL;
    break;
    case FDPDE_COMPOSITE:
    fd->ops->create = FDPDECreate_Composite;
    fd->ops->setup  = FDPDESetUp_Composite;
    break;
    default:
    SETERRQ(fd->comm,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown type of FD-PDE specified");
  }
  PetscCall(fd->ops->create(fd));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "FDPDESetSizes"
PetscErrorCode FDPDESetSizes(FDPDE fd,PetscInt nx,PetscInt nz,PetscScalar xs,PetscScalar xe,PetscScalar zs,PetscScalar ze)
{
  PetscFunctionBegin;
  if (nx <= 0) SETERRQ(fd->comm,PETSC_ERR_ARG_OUTOFRANGE,"Dimension 1 (arg 2) provided for FD-PDE dmstag must be > 0. Found %D",nx);
  if (nz <= 0) SETERRQ(fd->comm,PETSC_ERR_ARG_OUTOFRANGE,"Dimension 2 (arg 3) provided for FD-PDE dmstag must be > 0. Found %D",nz);
  if (xs >= xe) SETERRQ(fd->comm,PETSC_ERR_ARG_OUTOFRANGE,"Invalid x-maximum (arg 3) provided. xe > xs.");
  if (zs >= ze) SETERRQ(fd->comm,PETSC_ERR_ARG_OUTOFRANGE,"Invalid y-maximum (arg 5) provided. ze > zs");
  fd->Nx = nx;
  fd->Nz = nz;
  fd->x0 = xs;
  fd->x1 = xe;
  fd->z0 = zs;
  fd->z1 = ze;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "FDPDEGetAuxGlobalVectors"
PetscErrorCode FDPDEGetAuxGlobalVectors(FDPDE fd,PetscInt *n,Vec **vecs)
{
  PetscFunctionBegin;
  if (n)    *n = fd->naux_global_vectors;
  if (vecs) *vecs = fd->aux_global_vectors;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "FDPDEFormCoefficient"
PetscErrorCode FDPDEFormCoefficient(FDPDE fd)
{
  PetscInt i,n;
  FDPDE *pdelist = NULL;
  Vec *subX = NULL;
  
  PetscFunctionBegin;
  switch (fd->type) {
    case FDPDE_COMPOSITE:
    
    PetscCall(FDPDECompositeGetFDPDE(fd,&n,&pdelist));
    PetscCall(DMCompositeGetAccessArray(fd->dmstag,fd->x,n,NULL,subX));
    /* set auxillary vectors */
    for (i=0; i<n; i++) {
      pdelist[i]->naux_global_vectors = n;
      pdelist[i]->aux_global_vectors = subX;
    }
    for (i=0; i<n; i++) {
      PetscCall(FDPDEFormCoefficient(pdelist[i]));
    }
    PetscCall(DMCompositeRestoreAccessArray(fd->dmstag,fd->x,n,NULL,subX));
    break;
    
    default:
      if (!fd->ops->form_coefficient) SETERRQ(fd->comm,PETSC_ERR_ARG_NULL,"Form coefficient function pointer is NULL. Must call FDPDESetFunctionCoefficient() and provide a non-NULL function pointer.");
      PetscCall(fd->ops->form_coefficient(fd,fd->dmstag,fd->x,fd->dmcoeff,fd->coeff,fd->user_context));
    break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
/*@
FDPDESetLinearPreallocatorStencil() - Allocate non-zero preallocation for a linear system

Use: user
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDPDESetLinearPreallocatorStencil"
PetscErrorCode FDPDESetLinearPreallocatorStencil(FDPDE fd, PetscBool flg)
{
  PetscFunctionBegin;
  if (fd->setupcalled) SETERRQ(fd->comm,PETSC_ERR_ORDER,"Must call FDPDESetLinearPreallocatorStencil() before FDPDESetUp()");
  if (fd->type == FDPDE_ADVDIFF) PetscPrintf(PETSC_COMM_WORLD,"WARNING: This routine has no effect for FD-PDE Type = ADVDIFF! Only linear preallocator implemented.\n");
  fd->linearsolve = flg;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------
/*@
FDPDESetDMBoundaryType() - Set DMBoundaryType other than DM_BOUNDARY_NONE
Options: 
  DM_BOUNDARY_NONE, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_MIRROR, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_TWIST
Warning: not all are implemented! 

Use: user
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDPDESetDMBoundaryType"
PetscErrorCode FDPDESetDMBoundaryType(FDPDE fd, DMBoundaryType dm_btype0, DMBoundaryType dm_btype1)
{
  PetscFunctionBegin;
  if (fd->setupcalled) SETERRQ(fd->comm,PETSC_ERR_ORDER,"Must call FDPDESetDMBoundaryType() before FDPDESetUp()");
  if (dm_btype0) fd->dm_btype0 = dm_btype0;
  if (dm_btype1) fd->dm_btype1 = dm_btype1;

  PetscFunctionReturn(PETSC_SUCCESS);
}