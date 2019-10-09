/* Finite Differences PDE (FD-PDE) object */

#include "fdpde.h"

const char *FDPDETypeNames[] = {
  "uninit",
  "stokes",
  "advdiff"
};

// ---------------------------------------
// FDPDECreatePDEType declarations
// ---------------------------------------
PetscErrorCode FDPDECreate_Stokes(FDPDE fd);
//PetscErrorCode FDPDECreate_AdvDiff(FDPDE fd);

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
  PetscErrorCode ierr;

  
  PetscFunctionBegin;
  // Error checking
  if (!_fd) SETERRQ(comm,PETSC_ERR_ARG_NULL,"Must provide a valid (non-NULL) pointer for fd (arg 9)");
  
  if (nx <= 0) SETERRQ1(comm,PETSC_ERR_ARG_OUTOFRANGE,"Dimension 1 (arg 2) provided for FD-PDE dmstag must be > 0. Found %D",nx);
  if (nz <= 0) SETERRQ1(comm,PETSC_ERR_ARG_OUTOFRANGE,"Dimension 2 (arg 3) provided for FD-PDE dmstag must be > 0. Found %D",nz);
  
  if (xs >= xe) SETERRQ(comm,PETSC_ERR_ARG_OUTOFRANGE,"Invalid x-maximum (arg 5) provided. xe > xs.");
  if (zs >= ze) SETERRQ(comm,PETSC_ERR_ARG_OUTOFRANGE,"Invalid y-maximum (arg 7) provided. ze > zs");
  
  // Allocate memory
  ierr = PetscCalloc1(1,&fd);CHKERRQ(ierr);
  fd->comm = comm;

  // Save input variables
  fd->Nx = nx;
  fd->Nz = nz;
  fd->x0 = xs;
  fd->x1 = xe;
  fd->z0 = zs;
  fd->z1 = ze;

  fd->type = type;

  ierr = PetscMalloc1(1,&ops);CHKERRQ(ierr);
  ierr = PetscMemzero(ops, sizeof(struct _FDPDEOps)); CHKERRQ(ierr);
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
  fd->user_context = NULL;
  fd->setupcalled = PETSC_FALSE;

  fd->description_bc = NULL;
  fd->description_coeff = NULL;

  *_fd = fd;
  
  PetscFunctionReturn(0);
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
  PetscErrorCode ierr; 
  PetscFunctionBegin;
  if (fd->setupcalled) PetscFunctionReturn(0);

  // Set up structures needed for FD-PDE type
  switch (fd->type) {
    case FDPDE_UNINIT:
      SETERRQ(fd->comm,PETSC_ERR_ARG_TYPENOTSET,"Un-initialized type for FD-PDE");
      break;
    case FDPDE_STOKES:
      fd->ops->create = FDPDECreate_Stokes;
      break;
    case FDPDE_ADVDIFF:
      //fd->ops->create = FDPDECreate_AdvDiff;
      SETERRQ(fd->comm,PETSC_ERR_SUP,"FD-PDE type FDPDE_ADVDIFF is not yet implemented.");
      break;
    default:
      SETERRQ(fd->comm,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown type of FD-PDE specified");
  }

  // Create individual FD-PDE type
  ierr = fd->ops->create(fd); CHKERRQ(ierr);
  if (!fd->dmstag) SETERRQ(fd->comm,PETSC_ERR_ARG_NULL,"DM object for FD-PDE is NULL. The FD-PDE implementation constructor is required to create a valid DM of type DMSTAG");
  {
    PetscBool isStag;
    ierr = PetscObjectTypeCompare((PetscObject)fd->dmstag,DMSTAG,&isStag);CHKERRQ(ierr);
    if (!isStag) SETERRQ(fd->comm,PETSC_ERR_ARG_WRONGSTATE,"FD-PDE requires that the discretisation DM is of type DMSTAG");
  }

  // Create global vectors
  ierr = DMCreateGlobalVector(fd->dmstag,&fd->x);CHKERRQ(ierr);
  ierr = VecDuplicate(fd->x,&fd->r);CHKERRQ(ierr);
  ierr = VecDuplicate(fd->x,&fd->xguess);CHKERRQ(ierr);
  
  // Create coefficient dm and vector - specific to FD-PDE
  if (fd->ops->create_coefficient) {
    ierr = fd->ops->create_coefficient(fd);CHKERRQ(ierr);
  }

  // Create BClist object
  ierr = DMStagBCListCreate(fd->dmstag,&fd->bclist);CHKERRQ(ierr);

  // Create Jacobian
  if (!fd->ops->create_jacobian) SETERRQ(fd->comm,PETSC_ERR_ARG_NULL,"No method to create the Jacobian was provided. The FD-PDE implementation constructor is expected to set this function pointer");
  ierr = fd->ops->create_jacobian(fd,&fd->J); CHKERRQ(ierr);
  if (!fd->J) SETERRQ(fd->comm,PETSC_ERR_ARG_NULL,"Jacobian matrix for FD-PDE is NULL. The FD-PDE implementation for create_jacobian is required to create a valid Mat");

  // Create SNES - nonlinear solver context
  ierr = SNESCreate(fd->comm,&fd->snes); CHKERRQ(ierr);
  ierr = SNESSetDM(fd->snes,fd->dmstag); CHKERRQ(ierr);

  if (!fd->x) SETERRQ(fd->comm,PETSC_ERR_ARG_NULL,"Solution vector for FD-PDE is NULL.");
  ierr = SNESSetSolution(fd->snes,fd->x); CHKERRQ(ierr); // for FD colouring to function correctly

  // Set function evaluation routine
  if (!fd->ops->form_function) SETERRQ(fd->comm,PETSC_ERR_ARG_NULL,"No residual evaluation routine has been set.");
  ierr = SNESSetFunction(fd->snes, fd->r, fd->ops->form_function, (void*)fd); CHKERRQ(ierr);

  // Set Jacobian
  if (fd->ops->form_jacobian) {
    ierr = SNESSetJacobian(fd->snes, fd->J, fd->J, fd->ops->form_jacobian, (void*)fd); CHKERRQ(ierr);
  } else {
    ierr = SNESSetJacobian(fd->snes, fd->J, fd->J, SNESComputeJacobianDefaultColor, NULL); CHKERRQ(ierr);
  }

  fd->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(0);
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
  PetscErrorCode ierr;
  PetscFunctionBegin;

  // Return if no object
  if (!_fd) PetscFunctionReturn(0);
  if (!*_fd) PetscFunctionReturn(0);
  fd = *_fd;

  // Destroy objects
  ierr = DMStagBCListDestroy(&fd->bclist);CHKERRQ(ierr);
  ierr = PetscFree(fd->ops);CHKERRQ(ierr);
  ierr = VecDestroy(&fd->x);CHKERRQ(ierr);
  ierr = VecDestroy(&fd->r);CHKERRQ(ierr);
  ierr = VecDestroy(&fd->coeff);CHKERRQ(ierr);
  ierr = VecDestroy(&fd->xguess);CHKERRQ(ierr);
  ierr = MatDestroy(&fd->J);CHKERRQ(ierr);
  ierr = SNESDestroy(&fd->snes);CHKERRQ(ierr);

  ierr = DMDestroy(&fd->dmcoeff); CHKERRQ(ierr);
  ierr = DMDestroy(&fd->dmstag); CHKERRQ(ierr);

  fd->user_context = NULL;

  ierr = PetscFree(fd->description);CHKERRQ(ierr);
  ierr = PetscFree(fd->description_bc);CHKERRQ(ierr);
  ierr = PetscFree(fd->description_coeff);CHKERRQ(ierr);
  ierr = PetscFree(fd);CHKERRQ(ierr);
  *_fd = NULL;
  PetscFunctionReturn(0);
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
  PetscInt       dof[3];
  PetscErrorCode ierr;
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
    ierr = DMStagGetDOF(fd->dmstag,&dof[0],&dof[1],&dof[2],NULL);CHKERRQ(ierr);
    PetscPrintf(fd->comm,"  # dmstag: %D (vertices) %D (faces) %D (elements)\n",dof[0],dof[1],dof[2]);
  } else {
    PetscPrintf(fd->comm,"  # dmstag: not available\n");
  }
  if (fd->dmcoeff) {
    ierr = DMStagGetDOF(fd->dmcoeff,&dof[0],&dof[1],&dof[2],NULL);CHKERRQ(ierr);
    PetscPrintf(fd->comm,"  # dmcoeff: %D (vertices) %D (faces) %D (elements)\n",dof[0],dof[1],dof[2]);
  } else {
    PetscPrintf(fd->comm,"  # dmcoeff: not available\n");
  }
  if (fd->setupcalled) PetscPrintf(fd->comm,"  # FDPDESetUp: TRUE \n");
  else PetscPrintf(fd->comm,"  # FDPDESetUp: FALSE \n");

  // view BC list
  //ierr = DMStagBCListView(fd->bclist);CHKERRQ(ierr);

  PetscFunctionReturn(0);
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
  PetscErrorCode ierr;
  PetscFunctionBegin;

  if (!fd->dmstag) SETERRQ(fd->comm,PETSC_ERR_ORDER,"DMStag for FD-PDE not provided - Call FDPDESetUp()");
  *dm = fd->dmstag;
  ierr = PetscObjectReference((PetscObject)fd->dmstag);CHKERRQ(ierr);

  PetscFunctionReturn(0);
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
  PetscFunctionReturn(0);
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
  PetscErrorCode ierr;
  PetscFunctionBegin;

  fd->bclist->evaluate = evaluate;
  fd->bclist->data = data;

  /* free any existing name set by previous call */
  if (fd->description_bc) { ierr = PetscFree(fd->description_bc); CHKERRQ(ierr); }
  if (description) { ierr = PetscStrallocpy(description,&fd->description_bc); CHKERRQ(ierr); }

  PetscFunctionReturn(0);
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
PetscErrorCode FDPDESetFunctionCoefficient(FDPDE fd, PetscErrorCode (*form_coefficient)(DM,Vec,DM,Vec,void*), const char description[], void *data)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  fd->ops->form_coefficient = form_coefficient;
  fd->user_context = data;

  /* free any existing name set by previous call */
  if (fd->description_coeff) { ierr = PetscFree(fd->description_coeff); CHKERRQ(ierr); }
  if (description) { ierr = PetscStrallocpy(description,&fd->description_coeff); CHKERRQ(ierr); }

  PetscFunctionReturn(0);
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
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (x) {
    *x = fd->x;
    ierr = PetscObjectReference((PetscObject)fd->x);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
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
  PetscFunctionReturn(0);
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
  PetscFunctionReturn(0);
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
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (xguess) {
    *xguess = fd->xguess;
    ierr = PetscObjectReference((PetscObject)fd->xguess);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
FDPDESolve - solve the associated system of equations contained in an FD-PDE object.

Input Parameter:
fd - the FD-PDE object

Output Parameter:
converged - if not NULL tells if solver converged or not (PetscBool)

Functionality:
- calls SNESSetFromOptions() so any Petsc SNES options should be set before
- forms initial guess (copy xguess->x). xguess can be solution from previous time step or it can be specifically set 
by user by calling FDPDEGetSolutionGuess(fd,&xguess)
- solves and returns converged reason
- if converged, copy new solution to xguess

Use: user
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDPDESolve"
PetscErrorCode FDPDESolve(FDPDE fd, PetscBool *converged)
{
  SNESConvergedReason reason;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  if (!fd->setupcalled) SETERRQ(fd->comm,PETSC_ERR_ORDER,"User must call FDPDESetUp() first!");

  // Overwrite default options from command line
  ierr = SNESSetFromOptions(fd->snes); CHKERRQ(ierr);

  // Copy initial guess to solution
  ierr = VecCopy(fd->xguess,fd->x);CHKERRQ(ierr);

  // Solve the non-linear system
  ierr = SNESSolve(fd->snes,0,fd->x);             CHKERRQ(ierr);
  ierr = SNESGetConvergedReason(fd->snes,&reason); CHKERRQ(ierr);

  // Analyze convergence
  if (reason > 0) { // reason = 0 implies SNES_CONVERGED_ITERATING (which can never be true after SNESSolve executes)
    // converged - copy initial guess for next timestep
    ierr = VecCopy(fd->x, fd->xguess); CHKERRQ(ierr);
  }

  if (converged) {
    *converged = PETSC_TRUE;
    if (reason < 0) *converged = PETSC_FALSE;
  }

  PetscFunctionReturn(0);
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
  PetscErrorCode ierr;
  PetscFunctionBegin;

  if (!fd->setupcalled) SETERRQ(fd->comm,PETSC_ERR_ORDER,"User must call FDPDESetUp() first!");
  if (!fd->dmstag) SETERRQ(fd->comm,PETSC_ERR_ARG_NULL,"System of FD-PDE not provided. Call FDPDESetUp() first!");
  if (!cx) SETERRQ(fd->comm,PETSC_ERR_ARG_NULL,"Arg 2 (cx) cannot be NULL");
  if (!cz) SETERRQ(fd->comm,PETSC_ERR_ARG_NULL,"Arg 3 (cz) cannot be NULL");
  ierr = DMGetCoordinateDM(fd->dmstag,&dmCoord);CHKERRQ(ierr);
  if (!dmCoord) SETERRQ(PetscObjectComm((PetscObject)fd->dmstag),PETSC_ERR_ARG_WRONGSTATE,"DMStag does not have a coordinate DM");
  {
    PetscBool isProduct;
    DMType    dmType;
    ierr = DMGetType(dmCoord,&dmType);CHKERRQ(ierr);
    ierr = PetscStrcmp(DMPRODUCT,dmType,&isProduct);CHKERRQ(ierr);
    if (!isProduct) SETERRQ(PetscObjectComm((PetscObject)fd->dmstag),PETSC_ERR_SUP,"Implementation requires coordinate DM is of type DMPRODUCT");
  }
  ierr = DMStagGet1dCoordinateArraysDOFRead(fd->dmstag,&coordx,&coordz,NULL);CHKERRQ(ierr);
  *cx = coordx;
  *cz = coordz;
  PetscFunctionReturn(0);
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
  PetscErrorCode ierr;
  PetscFunctionBegin;

  if (!fd->setupcalled) SETERRQ(fd->comm,PETSC_ERR_ORDER,"User must call FDPDESetUp() first!");
  if (!fd->dmstag) SETERRQ(fd->comm,PETSC_ERR_ARG_NULL,"System of FD-PDE not provided. Call FDPDESetUp() first!");
  if (!cx) SETERRQ(fd->comm,PETSC_ERR_ARG_NULL,"Arg 2 (cx) cannot be NULL");
  if (!cz) SETERRQ(fd->comm,PETSC_ERR_ARG_NULL,"Arg 3 (cz) cannot be NULL");

  dm      = fd->dmstag;
  dmcoeff = fd->dmcoeff;

  // Update dmcoord coordinates
  ierr = DMStagGetDOF(dm,&dof0,&dof1,&dof2,NULL);CHKERRQ(ierr);
  ierr = DMStagGetDOF(dmcoeff,&dofc0,&dofc1,&dofc2,NULL);CHKERRQ(ierr);

  ierr = DMStagGet1dCoordinateArraysDOFRead(dmcoeff,&coordx,&coordz,NULL);CHKERRQ(ierr);

  if (dof2) {ierr = DMStagGet1dCoordinateLocationSlot(dm,DMSTAG_ELEMENT,&icenter);CHKERRQ(ierr);} 
  if (dof0 || dof1) { 
    ierr = DMStagGet1dCoordinateLocationSlot(dm,DMSTAG_LEFT,&iprev);CHKERRQ(ierr);
    ierr = DMStagGet1dCoordinateLocationSlot(dm,DMSTAG_RIGHT,&inext);CHKERRQ(ierr);
  } 

  if (dofc2) {ierr = DMStagGet1dCoordinateLocationSlot(dmcoeff,DMSTAG_ELEMENT,&icenterc);CHKERRQ(ierr);} 
  if (dofc0 || dofc1) { 
    ierr = DMStagGet1dCoordinateLocationSlot(dmcoeff,DMSTAG_LEFT,&iprevc);CHKERRQ(ierr);
    ierr = DMStagGet1dCoordinateLocationSlot(dmcoeff,DMSTAG_RIGHT,&inextc);CHKERRQ(ierr);
  } 

  ierr = DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

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
  ierr = DMStagRestore1dCoordinateArraysDOFRead(dmcoeff,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagRestore1dCoordinateArraysDOFRead(dm,&cx,&cz,NULL);CHKERRQ(ierr);

  // update coords of BCs
  ierr = DMStagBCListSetupCoordinates(fd->bclist);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
