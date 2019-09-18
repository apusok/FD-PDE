/* Finite Differences (FD) PDE object */

#include "fd.h"

// ---------------------------------------
// FDCreate
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDCreate"
PetscErrorCode FDCreate(MPI_Comm comm, FD *_fd)
{
  FD             fd;
  FDPDEOps       ops;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Allocate memory
  ierr = PetscMalloc1(1,&fd);CHKERRQ(ierr);
  ierr = PetscMalloc1(1,&ops);CHKERRQ(ierr);
  ierr = PetscMemzero(ops, sizeof(struct _FDPDEOps)); CHKERRQ(ierr);
  fd->ops = ops;

  // Initialize struct
  fd->dmstag  = NULL;
  fd->dmcoeff = NULL;
  fd->bc_list = NULL;
  fd->x     = NULL;
  fd->xguess= NULL;
  fd->r     = NULL;
  fd->coeff = NULL;
  fd->J     = NULL;
  fd->snes  = NULL;
  fd->type  = FD_UNINIT;
  fd->comm  = comm;

  *_fd = fd;
  
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
  if (fd->ops->destroy) {
    ierr = fd->ops->destroy(fd); CHKERRQ(ierr);
    ierr = PetscFree(fd->ops);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&fd->x);CHKERRQ(ierr);
  ierr = VecDestroy(&fd->r);CHKERRQ(ierr);
  ierr = VecDestroy(&fd->coeff );CHKERRQ(ierr);
  ierr = VecDestroy(&fd->xguess);CHKERRQ(ierr);
  ierr = MatDestroy(&fd->J);CHKERRQ(ierr);
  ierr = SNESDestroy(&fd->snes);CHKERRQ(ierr);
  //ierr = DMDestroy(&fd->dmcoeff); CHKERRQ(ierr);
  fd->dmstag  = NULL;
  fd->dmcoeff = NULL;
  fd->bc_list = NULL;

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
  PetscErrorCode ierr;
  PetscFunctionBegin;

  // also write some fd object info 

  // ASCII, but also need to rewrite VTK files for center/corner DM stag
  if (fd->ops->view) {
    ierr = fd->ops->view(fd,viewer); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

// ---------------------------------------
// FDSetType declarations
// ---------------------------------------
PetscErrorCode FDCreate_Stokes(FD fd);
//PetscErrorCode FDCreate_AdvDiff(FD fd);

// ---------------------------------------
// FDSetType
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDSetType"
PetscErrorCode FDSetType(FD fd, enum FDPDEType type)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  fd->type = type;
  switch (type) {
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

  // Create individual types
  ierr = fd->ops->create(fd); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// // ---------------------------------------
// // FDSetDM
// // ---------------------------------------
// #undef __FUNCT__
// #define __FUNCT__ "FDSetDM"
// PetscErrorCode FDSetDM(FD fd, DM *dm)
// {
//   PetscErrorCode ierr;
//   PetscFunctionBegin;

//   PetscFunctionReturn(0);
// }

// // ---------------------------------------
// // FDGetDM
// // ---------------------------------------
// #undef __FUNCT__
// #define __FUNCT__ "FDGetDM"
// PetscErrorCode FDGetDM(FD fd, DM *dm)
// {
//   PetscErrorCode ierr;
//   PetscFunctionBegin;

//   PetscFunctionReturn(0);
// }

// ---------------------------------------
// FDGetSolution
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDGetSolution"
PetscErrorCode FDGetSolution(FD fd, Vec *_x, Vec *_coeff)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (fd->x == NULL) SETERRQ(PetscObjectComm((PetscObject)fd),PETSC_ERR_USER,"Solution of FD-PDE not provided - Call FDSetSolution()");
  if (fd->coeff == NULL) SETERRQ(PetscObjectComm((PetscObject)fd),PETSC_ERR_USER,"Coefficient vector has not been set.");
  if (fd->type == FD_UNINIT) SETERRQ(PetscObjectComm((PetscObject)fd),PETSC_ERR_USER,"Type of FD-PDE has not been set");
  if (_x) {
    Vec x;
    ierr = VecDuplicate(fd->x,&x);CHKERRQ(ierr);
    ierr = VecCopy(fd->x,x);CHKERRQ(ierr);
    *_x = x;
    //*x = fd->x;
  }
  if (_coeff) {
    Vec coeff;
    ierr = VecDuplicate(fd->coeff,&coeff);CHKERRQ(ierr);
    ierr = VecCopy(fd->coeff,coeff);CHKERRQ(ierr);
    *_coeff = coeff;
    //*coeff = fd->coeff;
  }
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

  // SNES Options
  // Get default info on convergence
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
  ierr = VecCopy(fd->xguess,fd->x);CHKERRQ(ierr);

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
    ierr = VecCopy(fd->x, fd->xguess); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

// ---------------------------------------
// FDSetSolveSNES
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FDSetSolveSNES"
PetscErrorCode FDSetSolveSNES(FD fd)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = FDCreateSNES(fd->comm,fd); CHKERRQ(ierr);
  ierr = FDConfigureSNES(fd); CHKERRQ(ierr);
  ierr = FDSolveSNES(fd); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}