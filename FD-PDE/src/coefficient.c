
// #include "coefficient.h"

// // ---------------------------------------
// // CoefficientCreate
// // ---------------------------------------
// #undef __FUNCT__
// #define __FUNCT__ "CoefficientCreate"
// PetscErrorCode CoefficientCreate(MPI_Comm comm, Coefficient *_c, DMStagStencilLocation loc)
// {
//   Coefficient c;
//   PetscErrorCode ierr;
//   PetscFunctionBegin;

//   // Allocate memory
//   ierr = PetscMalloc1(1,&c); CHKERRQ(ierr);
//   ierr = PetscMemzero(c,sizeof(struct _p_Coefficient)); CHKERRQ(ierr);

//   c->type  = COEFF_NONE;
//   c->loc   = loc;
//   c->coeff = NULL;
//   *_c = c;

//   PetscFunctionReturn(0);
// }

// // ---------------------------------------
// // CoefficientDestroy
// // ---------------------------------------
// #undef __FUNCT__
// #define __FUNCT__ "CoefficientDestroy"
// PetscErrorCode CoefficientDestroy(Coefficient *_c)
// {
//   Coefficient c;
//   PetscErrorCode ierr;
//   PetscFunctionBegin;

//   c = *_c;
//   // Free memory
//   ierr = PetscFree(c->coeff); CHKERRQ(ierr);
//   c->user_context = NULL;
//   c->eval_function = NULL;
//   ierr = PetscFree(c); CHKERRQ(ierr);

//   PetscFunctionReturn(0);
// }

// // ---------------------------------------
// // CoefficientSetEvaluate
// // ---------------------------------------
// #undef __FUNCT__
// #define __FUNCT__ "CoefficientSetEvaluate"
// PetscErrorCode CoefficientSetEvaluate(Coefficient c, PetscErrorCode (*eval_function)(Coefficient, void*), void *data)
// {
//   PetscFunctionBegin;
//   if (c->type != COEFF_EVAL) SETERRQ(PetscObjectComm((PetscObject)c),PETSC_ERR_USER,"Coefficient is not set to be of type EVAL");
//   c->eval_function = eval_function;
//   c->user_context  = data;
//   PetscFunctionReturn(0);
// }

// // ---------------------------------------
// // CoefficientAllocateMemory
// // ---------------------------------------
// #undef __FUNCT__
// #define __FUNCT__ "CoefficientAllocateMemory"
// PetscErrorCode CoefficientAllocateMemory(FD fd, Coefficient *_c)
// {
//   Coefficient    c;
//   PetscInt       nx,nz;
//   PetscErrorCode ierr;
//   PetscFunctionBegin;

//   c = *_c;

//   // Get domain corners
//   ierr = DMStagGetCorners(fd->dmcoeff, NULL, NULL, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

//   // Allocate memory to 1D coeff and coord arrays
//   switch (c->loc) {
//     case DMSTAG_DOWN:
//       c->nx = nx; c->nz = nz+1;
//       break;
//     case DMSTAG_UP:
//       c->nx = nx; c->nz = nz+1;
//       break;
//     case DMSTAG_RIGHT:
//       c->nx = nx+1; c->nz = nz;
//       break;
//     case DMSTAG_LEFT:
//       c->nx = nx+1; c->nz = nz;
//       break;
//     case DMSTAG_ELEMENT:
//       c->nx = nx; c->nz = nz;
//       break;
//     case DMSTAG_DOWN_RIGHT:
//       c->nx = nx+1; c->nz = nz+1;
//       break;
//     case DMSTAG_DOWN_LEFT:
//       c->nx = nx+1; c->nz = nz+1;
//       break;
//     case DMSTAG_UP_RIGHT:
//       c->nx = nx+1; c->nz = nz+1;
//       break;
//     case DMSTAG_UP_LEFT:
//       c->nx = nx+1; c->nz = nz+1;
//       break;
//     default:
//       SETERRQ(PetscObjectComm((PetscObject)fd),PETSC_ERR_USER,"Unknown type of coefficient location");  
//   }
//   c->npoints = c->nx*c->nz;

//   // Allocate memory to local arrays of coefficients and coord
//   ierr = PetscMalloc((size_t)c->npoints*sizeof(PetscScalar),&c->coeff);CHKERRQ(ierr);
//   ierr = PetscMemzero(c->coeff,(size_t)c->npoints*sizeof(PetscScalar));

//   // Save fd context
//   c->fd = fd;

//   PetscFunctionReturn(0);
// }

// // ---------------------------------------
// // CoefficientEvaluate
// // ---------------------------------------
// #undef __FUNCT__
// #define __FUNCT__ "CoefficientEvaluate"
// PetscErrorCode CoefficientEvaluate(Coefficient c)
// {
//   PetscErrorCode ierr;
//   PetscFunctionBegin;
//   switch (c->type) {
//     case COEFF_NONE:
//     break;

//     case COEFF_EVAL:
//       if (c->eval_function) {
//         ierr = c->eval_function(c,c->user_context);CHKERRQ(ierr);
//       } else {
//         SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"No method was provided by user to evaluate the coefficient");
//       }
//     break;
//   }

//   PetscFunctionReturn(0);
// }