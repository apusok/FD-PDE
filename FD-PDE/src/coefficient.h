// /* <Coefficients> contain user defined elements for coefficients in a PDE */

// #ifndef COEFFICIENT_H
// #define COEFFICIENT_H

// #include "petsc.h"
// #include "fd.h"

// // ---------------------------------------
// // Enum definitions
// // ---------------------------------------
// typedef struct _p_Coefficient *Coefficient;

// enum CoefficientType { COEFF_NONE, COEFF_EVAL };

// // ---------------------------------------
// // Struct definitions
// // ---------------------------------------
// // Coefficient
// struct _p_Coefficient {
//   enum CoefficientType type;
//   DMStagStencilLocation   loc;
//   PetscScalar     *coeff; // 1D local array [i+j*nz]
//   PetscInt        nx, nz, npoints; // local sizes
//   void            *user_context; 
//   PetscErrorCode (*eval_function)(Coefficient, void*);
//   FD              fd;
// };

// // // ---------------------------------------
// // // Function definitions
// // // ---------------------------------------
// PetscErrorCode CoefficientCreate(MPI_Comm, Coefficient*, DMStagStencilLocation);
// PetscErrorCode CoefficientDestroy(Coefficient*);
// PetscErrorCode CoefficientSetEvaluate(Coefficient, PetscErrorCode (*eval_function)(Coefficient, void*), void*);
// PetscErrorCode CoefficientAllocateMemory(FD, Coefficient*);
// PetscErrorCode CoefficientEvaluate(Coefficient);

// #endif