/* Finite Differences PDE (FD-PDE) object */

#include "fd.h"

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

  // PetscPrintf(PETSC_COMM_WORLD,"# Break 0 xmin %f xmax %f zmin %f zmax %f#\n",xs,fd->x1,fd->z0,fd->z1);

  // Allocate memory
  ierr = PetscMalloc1(1,&fd);CHKERRQ(ierr);

  // Error checking
  if (!nx) SETERRQ(PetscObjectComm((PetscObject)fd),PETSC_ERR_USER,"Dimension 1 not provided for FD-PDE dmstag");
  if (!nz) SETERRQ(PetscObjectComm((PetscObject)fd),PETSC_ERR_USER,"Dimension 2 not provided for FD-PDE dmstag");

  // if (!xs) SETERRQ(PetscObjectComm((PetscObject)fd),PETSC_ERR_USER,"Minimum global coord value for x-dir not provided");
  // if (!zs) SETERRQ(PetscObjectComm((PetscObject)fd),PETSC_ERR_USER,"Minimum global coord value for z-dir not provided");

  // if (!xe) SETERRQ(PetscObjectComm((PetscObject)fd),PETSC_ERR_USER,"Maximum global coord value for x-dir not provided");
  // if (!ze) SETERRQ(PetscObjectComm((PetscObject)fd),PETSC_ERR_USER,"Maximum global coord value for z-dir not provided");

  if (xs>=xe) SETERRQ(PetscObjectComm((PetscObject)fd),PETSC_ERR_USER,"Invalid minimum/maximum x-dir global coordinates");
  if (zs>=ze) SETERRQ(PetscObjectComm((PetscObject)fd),PETSC_ERR_USER,"Invalid minimum/maximum z-dir global coordinates");

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
  
  *_fd = fd;
  
  PetscFunctionReturn(0);
}

// ---------------------------------------
/*@
FDSetUp - sets up the data structures inside a FD object

Input Parameter:
fd - the FD object to setup

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
  // fd->type = type;
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
  ierr = fd->ops->create_coefficient(fd);CHKERRQ(ierr);

  // Create BClist object
  ierr = DMStagBCListCreate(fd->dmstag,&fd->bclist);CHKERRQ(ierr);

  // Preallocator Jacobian
  ierr = FDJacobianPreallocator(fd);CHKERRQ(ierr);

  // Create SNES here ?

  fd->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

// ---------------------------------------
// FDDestroy
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
  ierr = PetscFree(fd);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// FDView <INCOMPLETE>
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDView"
PetscErrorCode FDView(FD fd, PetscViewer viewer)
{
  // PetscErrorCode ierr;
  PetscFunctionBegin;

  // View FD object



  PetscFunctionReturn(0);
}

// ---------------------------------------
// FDGetDM
// ---------------------------------------
/*@ FDGetDM - Retrieves the dmstag from the fd object. User has to call DMDestroy() to free the space. @*/
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

// // ---------------------------------------
// // FDSetBC
// // ---------------------------------------
// #undef __FUNCT__
// #define __FUNCT__ "FDSetBCList"
// PetscErrorCode FDSetBCList(FD fd, DMStagBCList bclist)
// {
//   PetscFunctionBegin;

//   // Save pointers to bclist
//   if (bclist) fd->bc_list = bclist;

//   PetscFunctionReturn(0);
// }

// ---------------------------------------
// FDSetFunctionBCList
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDSetFunctionBCList"
PetscErrorCode FDSetFunctionBCList(FD fd, PetscErrorCode (*evaluate)(DM,Vec,DMStagBCList,void*), void *data)
{
  // PetscErrorCode ierr; 
  PetscFunctionBegin;

  if (!evaluate) SETERRQ(PetscObjectComm((PetscObject)fd),PETSC_ERR_USER,"No function is provided to calculate the BC List!");
  fd->bclist->evaluate = evaluate;
  fd->bclist->data = data;

  PetscFunctionReturn(0);
}

// ---------------------------------------
// FDSetFunctionCoefficient
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDSetFunctionCoefficient"
PetscErrorCode FDSetFunctionCoefficient(FD fd, PetscErrorCode (*form_coefficient)(DM,Vec,DM,Vec,void*), void *data)
{
  // PetscErrorCode ierr; 
  PetscFunctionBegin;

  if (!form_coefficient) SETERRQ(PetscObjectComm((PetscObject)fd),PETSC_ERR_USER,"No function is provided to calculate the coeffients!");
  fd->ops->form_coefficient = form_coefficient;
  fd->user_context = data;

  PetscFunctionReturn(0);
}

// ---------------------------------------
// FDGetSolution
// ---------------------------------------
/*@ FDGetSolution - Retrieves the solution vector from the fd object. User has to call VecDestroy() to free the space. @*/
#undef __FUNCT__
#define __FUNCT__ "FDGetSolution"
PetscErrorCode FDGetSolution(FD fd, Vec *_x)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (fd->x == NULL) SETERRQ(PetscObjectComm((PetscObject)fd),PETSC_ERR_USER,"Solution of FD-PDE not provided - Call FDSetSolution()");
  if (fd->type == FD_UNINIT) SETERRQ(PetscObjectComm((PetscObject)fd),PETSC_ERR_USER,"Type of FD-PDE has not been set.");
  if (_x) {
    *_x = fd->x;
    ierr = PetscObjectReference((PetscObject)fd->x);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

// ---------------------------------------
// FDJacobianPreallocator
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDJacobianPreallocator"
PetscErrorCode FDJacobianPreallocator(FD fd)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  if (fd->J == NULL) SETERRQ(PetscObjectComm((PetscObject)fd),PETSC_ERR_USER,"Jacobian matrix for FD-PDE has not been set.");
  if (fd->ops->jacobian_prealloc==NULL) SETERRQ(PetscObjectComm((PetscObject)fd),PETSC_ERR_USER,"No Jacobian preallocation method has not been set.");

  // Preallocate Jacobian including bclist
  ierr = fd->ops->jacobian_prealloc(fd); CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

// ---------------------------------------
// FDCreateSNES
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDCreateSNES"
PetscErrorCode FDCreateSNES(MPI_Comm comm, FD fd)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  // Create nonlinear solver context
  ierr = SNESCreate(comm,&fd->snes); CHKERRQ(ierr);

  // set dm to snes
  ierr = SNESSetDM(fd->snes, fd->dmstag); CHKERRQ(ierr);

  // set solution - need to do this for FD colouring to function correctly
  ierr = SNESSetSolution(fd->snes, fd->x); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// FDSetOptionsPrefix
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDSetOptionsPrefix"
PetscErrorCode FDSetOptionsPrefix(FD fd,const char prefix[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscObjectSetOptionsPrefix((PetscObject)fd,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// ---------------------------------------
// FDConfigureSNES
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDConfigureSNES"
PetscErrorCode FDConfigureSNES(FD fd)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  if (fd->x == NULL) SETERRQ(PetscObjectComm((PetscObject)fd),PETSC_ERR_USER,"Solution of FD-PDE not provided");
  if (fd->type == FD_UNINIT) SETERRQ(PetscObjectComm((PetscObject)fd),PETSC_ERR_USER,"Type of FD-PDE has not been set");
  
  // set function evaluation routine
  ierr = SNESSetFunction(fd->snes, fd->r, fd->ops->form_function, fd); CHKERRQ(ierr);

  // set Jacobian
  ierr = SNESSetJacobian(fd->snes, fd->J, fd->J, SNESComputeJacobianDefaultColor, NULL); CHKERRQ(ierr);

  // SNES Options - default info on convergence
  ierr = PetscOptionsSetValue(NULL, "-snes_monitor",          ""); CHKERRQ(ierr);
  ierr = PetscOptionsSetValue(NULL, "-ksp_monitor",           ""); CHKERRQ(ierr);
  ierr = PetscOptionsSetValue(NULL, "-snes_converged_reason", ""); CHKERRQ(ierr);
  ierr = PetscOptionsSetValue(NULL, "-ksp_converged_reason",  ""); CHKERRQ(ierr);

  // overwrite default options from command line
  ierr = SNESSetFromOptions(fd->snes); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// FDSolveSNES
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDSolveSNES"
PetscErrorCode FDSolveSNES(FD fd)
{
  SNESConvergedReason reason;
  PetscInt       maxit, maxf, its;
  PetscReal      atol, rtol, stol;
  PetscErrorCode ierr;

  PetscFunctionBegin;

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
// FDSolve
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDSolve"
PetscErrorCode FDSolve(FD fd)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = FDCreateSNES(fd->comm,fd);CHKERRQ(ierr);
  ierr = FDConfigureSNES(fd); CHKERRQ(ierr);
  ierr = FDSolveSNES(fd); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}