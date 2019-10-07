/* Finite Differences PDE (FD-PDE) object */

#include "fd.h"

const char *FDPDETypeNames[] = {
  "uninit",
  "stokes",
  "advdiff"
};

// ---------------------------------------
// FDCreatePDEType declarations
// ---------------------------------------
PetscErrorCode FDCreate_Stokes(FD fd);
//PetscErrorCode FDCreate_AdvDiff(FD fd);

// ---------------------------------------
/*@ FDCreate - creates an object that will manage the discretization of a PDE using 
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
You must call FDSetUp() after this call, before using the FD-PDE object. Also, you must destroy the object with FDDestroy().

Use: user
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDCreate"
PetscErrorCode FDCreate(MPI_Comm comm, PetscInt nx, PetscInt nz, 
                        PetscScalar xs, PetscScalar xe, PetscScalar zs, PetscScalar ze, 
                        FDPDEType type, FD *_fd)
{
  FD             fd;
  FDPDEOps       ops;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  // Allocate memory
  ierr = PetscMalloc1(1,&fd);CHKERRQ(ierr);

  // Error checking
  if (!nx) SETERRQ(PetscObjectComm((PetscObject)fd),PETSC_ERR_USER,"Dimension 1 not provided for FD-PDE dmstag");
  if (!nz) SETERRQ(PetscObjectComm((PetscObject)fd),PETSC_ERR_USER,"Dimension 2 not provided for FD-PDE dmstag");

  if (xs>=xe) SETERRQ(PetscObjectComm((PetscObject)fd),PETSC_ERR_USER,"Invalid minimum/maximum x-dir global coordinates");
  if (zs>=ze) SETERRQ(PetscObjectComm((PetscObject)fd),PETSC_ERR_USER,"Invalid minimum/maximum z-dir global coordinates");

  if (!_fd) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must provide a valid (non-NULL) pointer for fd (arg 9)");

  // Save input variables
  fd->Nx = nx;
  fd->Nz = nz;
  fd->x0 = xs;
  fd->x1 = xe;
  fd->z0 = zs;
  fd->z1 = ze;

  if (type) fd->type = type;
  else      fd->type = FD_UNINIT;

  ierr = PetscMalloc1(1,&ops);CHKERRQ(ierr);
  ierr = PetscMemzero(ops, sizeof(struct _FDPDEOps)); CHKERRQ(ierr);
  fd->ops = ops;

  // Initialize struct
  fd->dmstag  = NULL;
  fd->dmcoeff = NULL;
  fd->bclist = NULL;
  fd->x     = NULL;
  fd->xold  = NULL;
  fd->r     = NULL;
  fd->coeff = NULL;
  fd->J     = NULL;
  fd->snes  = NULL;
  fd->user_context = NULL;
  fd->comm  = comm;
  fd->setupcalled = PETSC_FALSE;

  fd->description_bc = NULL;
  fd->description_coeff = NULL;

  *_fd = fd;
  
  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
FDSetUp - sets up the data structures inside a FD object

Input Parameter:
fd - the FD object to setup

To change the snes prefix, one should call:
  FDSetUp(fd);
  FDGetSNES(fd,&snes);
  SNESSetOptionsPrefix(snes);
  SNESSetFromOptions(snes);
  FDSolve(fd);

Use: user
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDSetUp"
PetscErrorCode FDSetUp(FD fd)
{
  PetscErrorCode ierr; 
  PetscFunctionBegin;
  if (fd->setupcalled) PetscFunctionReturn(0);

  // Set up structures needed for FD-PDE type
  switch (fd->type) {
    case FD_UNINIT:
      SETERRQ(PetscObjectComm((PetscObject)fd),PETSC_ERR_USER,"Un-initialized type for FD-PDE");
      break;
    case STOKES:
      fd->ops->create = FDCreate_Stokes;
      break;
    case ADVDIFF:
      //fd->ops->create = FDCreate_AdvDiff;
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)fd),PETSC_ERR_USER,"Unknown type of FD-PDE specified");
  }

  // Create individual FD-PDE type
  ierr = fd->ops->create(fd); CHKERRQ(ierr);

  // Create coefficient dm and vector - specific to FD-PDE
  if (fd->ops->create_coefficient==NULL) SETERRQ(PetscObjectComm((PetscObject)fd),PETSC_ERR_USER,"No method to create the coefficients has been set.");
  ierr = fd->ops->create_coefficient(fd);CHKERRQ(ierr);

  // Create BClist object
  if (fd->dmstag == NULL) SETERRQ(PetscObjectComm((PetscObject)fd),PETSC_ERR_USER,"DM object for FD-PDE has not been set.");
  ierr = DMStagBCListCreate(fd->dmstag,&fd->bclist);CHKERRQ(ierr);

  // Preallocator Jacobian
  if (fd->J == NULL) SETERRQ(PetscObjectComm((PetscObject)fd),PETSC_ERR_USER,"Jacobian matrix for FD-PDE has not been set.");
  if (fd->ops->jacobian_prealloc==NULL) SETERRQ(PetscObjectComm((PetscObject)fd),PETSC_ERR_USER,"No Jacobian preallocation method has been set.");
  ierr = fd->ops->jacobian_prealloc(fd); CHKERRQ(ierr);

  // Create SNES - nonlinear solver context
  ierr = SNESCreate(fd->comm,&fd->snes); CHKERRQ(ierr);
  ierr = SNESSetDM(fd->snes,fd->dmstag); CHKERRQ(ierr);

  if (fd->x == NULL) SETERRQ(PetscObjectComm((PetscObject)fd),PETSC_ERR_USER,"Solution vector for FD-PDE has not been set.");
  ierr = SNESSetSolution(fd->snes,fd->x); CHKERRQ(ierr); // for FD colouring to function correctly

  // Set function evaluation routine
  if (fd->ops->form_function==NULL) SETERRQ(PetscObjectComm((PetscObject)fd),PETSC_ERR_USER,"No residual evaluation routine has been set.");
  ierr = SNESSetFunction(fd->snes, fd->r, fd->ops->form_function, fd); CHKERRQ(ierr);

  // Set Jacobian
  ierr = SNESSetJacobian(fd->snes, fd->J, fd->J, SNESComputeJacobianDefaultColor, NULL); CHKERRQ(ierr);

  // SNES Options - default info on convergence
  ierr = PetscOptionsSetValue(NULL, "-snes_monitor",         ""); CHKERRQ(ierr);
  ierr = PetscOptionsSetValue(NULL, "-ksp_monitor",          ""); CHKERRQ(ierr);
  ierr = PetscOptionsSetValue(NULL, "-snes_converged_reason",""); CHKERRQ(ierr);
  ierr = PetscOptionsSetValue(NULL, "-ksp_converged_reason", ""); CHKERRQ(ierr);

  // overwrite default options from command line
  ierr = SNESSetFromOptions(fd->snes); CHKERRQ(ierr);

  fd->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
FDDestroy - destroy an FD object

Input Parameter:
fd - the FD object to destroy

Use: user
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDDestroy"
PetscErrorCode FDDestroy(FD *_fd)
{
  FD fd;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  fd = *_fd;

  // Return if no object
  if (!fd) PetscFunctionReturn(0);

  // Destroy objects
  ierr = DMStagBCListDestroy(&fd->bclist);CHKERRQ(ierr);
  ierr = PetscFree(fd->ops);CHKERRQ(ierr);
  ierr = VecDestroy(&fd->x);CHKERRQ(ierr);
  ierr = VecDestroy(&fd->r);CHKERRQ(ierr);
  ierr = VecDestroy(&fd->coeff);CHKERRQ(ierr);
  ierr = VecDestroy(&fd->xold);CHKERRQ(ierr);
  ierr = MatDestroy(&fd->J);CHKERRQ(ierr);
  ierr = SNESDestroy(&fd->snes);CHKERRQ(ierr);

  ierr = DMDestroy(&fd->dmcoeff); CHKERRQ(ierr);
  ierr = DMDestroy(&fd->dmstag); CHKERRQ(ierr);

  fd->bclist = NULL;
  fd->user_context = NULL;

  ierr = PetscFree(fd->description);CHKERRQ(ierr);
  ierr = PetscFree(fd->description_bc);CHKERRQ(ierr);
  ierr = PetscFree(fd->description_coeff);CHKERRQ(ierr);
  ierr = PetscFree(fd);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
FDView - ASCII print of FD info structure on PETSC_COMM_WORLD 

Input Parameter:
fd - the FD object to view

Use: user
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDView"
PetscErrorCode FDView(FD fd)
{
  PetscInt       dof[3];
  PetscErrorCode ierr;
  PetscFunctionBegin;

  PetscPrintf(PETSC_COMM_WORLD,"FDView:\n");
  PetscPrintf(PETSC_COMM_WORLD,"  # FD-PDE type: %s\n",FDPDETypeNames[(int)fd->type]);
  
  PetscPrintf(PETSC_COMM_WORLD,"  # FD-PDE description:\n");
  PetscPrintf(PETSC_COMM_WORLD,"    %s\n",fd->description);

  PetscPrintf(PETSC_COMM_WORLD,"  # Coefficient description:\n");
  if (fd->description_coeff) PetscPrintf(PETSC_COMM_WORLD,"    %s\n",fd->description_coeff);
  else PetscPrintf(PETSC_COMM_WORLD,"    NONE\n");

  PetscPrintf(PETSC_COMM_WORLD,"  # BC description:\n");
  if (fd->description_bc) PetscPrintf(PETSC_COMM_WORLD,"    %s\n",fd->description_bc);
  else PetscPrintf(PETSC_COMM_WORLD,"    NONE\n");

  PetscPrintf(PETSC_COMM_WORLD,"  # global size elements: %D (x-dir) %D (z-dir)\n",fd->Nx,fd->Nz);

  ierr = DMStagGetDOF(fd->dmstag,&dof[0],&dof[1],&dof[2],NULL);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"  # dmstag: %D (vertices) %D (faces) %D (elements)\n",dof[0],dof[1],dof[2]);

  ierr = DMStagGetDOF(fd->dmcoeff,&dof[0],&dof[1],&dof[2],NULL);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"  # dmcoeff: %D (vertices) %D (faces) %D (elements)\n",dof[0],dof[1],dof[2]);

  if (fd->setupcalled) PetscPrintf(PETSC_COMM_WORLD,"  # FDSetUp: TRUE \n");
  else PetscPrintf(PETSC_COMM_WORLD,"  # FDSetUp: FALSE \n");

  PetscPrintf(PETSC_COMM_WORLD,"\n");

  // view BC list
  //ierr = DMStagBCListView(fd->bclist);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
FDGetDM - retrieves the main DMStag (associated with the solution) from the FD object. 

Input Parameter:
fd - the FD object

Output Parameter:
dm - the DM object

Notes:
DM object not destroyed with FDDestroy(). User has to call DMDestroy() to free the space.

Use: user
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDGetDM"
PetscErrorCode FDGetDM(FD fd, DM *dm)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  if (!fd->dmstag) SETERRQ(PetscObjectComm((PetscObject)fd),PETSC_ERR_USER,"DMStag for FD-PDE not provided - Call FDCreate()");
  *dm = fd->dmstag;
  ierr = PetscObjectReference((PetscObject)fd->dmstag);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
FDSetFunctionBCList - set an evaluation function for boundary conditions 

Input Parameter:
fd - the FD object
evaluate - name of the evaluation function for boundary conditions
description - user can provide a description for BC
data - user context to be passed for evaluation (can be NULL)

Use: user
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDSetFunctionBCList"
PetscErrorCode FDSetFunctionBCList(FD fd, PetscErrorCode (*evaluate)(DM,Vec,DMStagBCList,void*), const char description[], void *data)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  if (!evaluate) SETERRQ(PetscObjectComm((PetscObject)fd),PETSC_ERR_USER,"No function is provided to calculate the BC List!");
  fd->bclist->evaluate = evaluate;
  fd->bclist->data = data;

  /* free any existing name set by previous call */
  if (fd->description_bc) { ierr = PetscFree(fd->description_bc); CHKERRQ(ierr); }
  if (description) { ierr = PetscStrallocpy(description,&fd->description_bc); CHKERRQ(ierr); }

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
FDSetFunctionCoefficient - set an evaluation function for FD-PDE coefficients

Input Parameter:
fd - the FD object
form_coefficient - name of the evaluation function for coefficients
description - user can provide a description for coefficients
data - user context to be passed for evaluation (can be NULL)

Use: user
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDSetFunctionCoefficient"
PetscErrorCode FDSetFunctionCoefficient(FD fd, PetscErrorCode (*form_coefficient)(DM,Vec,DM,Vec,void*), const char description[], void *data)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  if (!form_coefficient) SETERRQ(PetscObjectComm((PetscObject)fd),PETSC_ERR_USER,"No function is provided to calculate the coeffients!");
  fd->ops->form_coefficient = form_coefficient;
  fd->user_context = data;

  /* free any existing name set by previous call */
  if (fd->description_coeff) { ierr = PetscFree(fd->description_coeff); CHKERRQ(ierr); }
  if (description) { ierr = PetscStrallocpy(description,&fd->description_coeff); CHKERRQ(ierr); }

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
FDGetSolution - retrieves the solution vector from the FD object. 

Input Parameter:
fd - the FD object

Output Parameter:
x - the solution vector

Notes:
Vector x not destroyed with FDDestroy(). User has to call VecDestroy() separately to free the space.

Use: user
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDGetSolution"
PetscErrorCode FDGetSolution(FD fd, Vec *x)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (fd->x == NULL) SETERRQ(PetscObjectComm((PetscObject)fd),PETSC_ERR_USER,"Solution of FD-PDE not provided - Call FDSetUp() first");
  if (x) {
    *x = fd->x;
    ierr = PetscObjectReference((PetscObject)fd->x);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
FDGetSNES - retrieves the SNES object from the FD object. 

Input Parameter:
fd - the FD object

Output Parameter:
snes - the snes object

Use: user
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDGetSNES"
PetscErrorCode FDGetSNES(FD fd, SNES *snes)
{
  PetscFunctionBegin;
  //if (fd->snes == NULL) SETERRQ(PetscObjectComm((PetscObject)fd),PETSC_ERR_USER,"The SNES object for FD-PDE not provided - Call FDSetUp() first");
  if (snes) *snes = fd->snes;

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
FDGetDMStagBCList() - retrieves the DMStagBCList object from the FD object. 

Input Parameter:
fd - the FD object

Output Parameter:
list - the DMStagBCList object

Use: user
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDGetDMStagBCList"
PetscErrorCode FDGetDMStagBCList(FD fd, DMStagBCList *list)
{
  PetscFunctionBegin;
  //if (fd->bclist == NULL) SETERRQ(PetscObjectComm((PetscObject)fd),PETSC_ERR_USER,"The DMStagBCList object for FD-PDE not provided - Call FDSetUp() first");
  if (list) *list = fd->bclist;

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
FDSolve - solve the associated system of equations contained in an FD object.

Input Parameter:
fd - the FD object

Use: user
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDSolve"
PetscErrorCode FDSolve(FD fd)
{
  SNESConvergedReason reason;
  PetscInt       maxit, maxf, its;
  PetscReal      atol, rtol, stol;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  if (!fd->setupcalled) SETERRQ(PetscObjectComm((PetscObject)fd),PETSC_ERR_USER,"User must call FDSetUp() first!");

  // Copy initial guess to solution
  ierr = VecCopy(fd->xold,fd->x);CHKERRQ(ierr);

  // Solve the non-linear system
  ierr = SNESSolve(fd->snes,0,fd->x);             CHKERRQ(ierr);
  ierr = SNESGetConvergedReason(fd->snes,&reason); CHKERRQ(ierr);
  ierr = SNESGetIterationNumber(fd->snes,&its);    CHKERRQ(ierr);
  ierr = SNESGetTolerances(fd->snes, &atol, &rtol, &stol, &maxit, &maxf); CHKERRQ(ierr);
  
  // Print some diagnostics
  ierr = PetscPrintf(fd->comm,"Number of SNES iterations = %d\n",its);
  ierr = PetscPrintf(fd->comm,"SNES: atol = %g, rtol = %g, stol = %g, maxit = %D, maxf = %D\n",(double)atol,(double)rtol,(double)stol,maxit,maxf); CHKERRQ(ierr);

  // Analyze convergence
  if (reason<0) {
    // NOT converged
    if (reason < 0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_CONV_FAILED,"Nonlinear solve failed!"); CHKERRQ(ierr);
  } else {
    // converged - copy initial guess for next timestep
    ierr = VecCopy(fd->x, fd->xold); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
FDGetCoordinatesArrayDMStag - retrieve the 1D coordinate arrays of the DMStag (system of equations) inside an FD object

Input parameter:
fd - the FD object

Output parameters:
cx - 1D array containing x-coordinates
cz - 1D array containing z-coordinates

Use: user
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDGetCoordinatesArrayDMStag"
PetscErrorCode FDGetCoordinatesArrayDMStag(FD fd,PetscScalar ***cx, PetscScalar ***cz)
{
  PetscScalar    **coordx,**coordz;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  if (fd->dmstag==NULL) SETERRQ(PetscObjectComm((PetscObject)fd),PETSC_ERR_USER,"System of FD-PDE not provided. Call FDCreate()/FDSetUp() first!");
  ierr = DMStagGet1dCoordinateArraysDOFRead(fd->dmstag,&coordx,&coordz,NULL);CHKERRQ(ierr);

  *cx = coordx;
  *cz = coordz;

  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
FDRestoreCoordinatesArrayDMStag - restore the 1D coordinate arrays of the DMStag (system of equations) inside an FD object

Input parameter:
fd - the FD object
cx - 1D array containing x-coordinates
cz - 1D array containing z-coordinates

Notes:
Must be called after FDGetCoordinatesArrayDMStag() and will update the coordinates of the dmcoeff and BCs.

Use: user
@*/
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDRestoreCoordinatesArrayDMStag"
PetscErrorCode FDRestoreCoordinatesArrayDMStag(FD fd,PetscScalar **cx, PetscScalar **cz)
{
  DM             dm, dmcoeff;
  PetscScalar    **coordx,**coordz;
  PetscScalar    xprev, xnext, xcenter, zprev, znext, zcenter;
  PetscInt       dof0, dof1, dof2, dofc0, dofc1, dofc2;
  PetscInt       iprev=-1,inext=-1,icenter=-1,iprevc=-1,inextc=-1,icenterc=-1;
  PetscInt       i, j, sx, sz, nx, nz;
  PetscErrorCode ierr;
  PetscFunctionBegin;

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