/* <FD> contains Finite Differences PDE (FD-PDE) object */

#ifndef FD_H
#define FD_H

#include "petsc.h"
#include "prealloc_helper.h"
#include "bc.h"

// ---------------------------------------
// Enum definitions
// ---------------------------------------
// PDE type
typedef enum { FD_UNINIT = 0, STOKES, ADVDIFF } FDPDEType;

// ---------------------------------------
// Struct definitions
// ---------------------------------------
// PDE Options - Functions
typedef struct _FDPDEOps *FDPDEOps;
typedef struct _p_FD *FD;

struct _FDPDEOps {
  PetscErrorCode (*form_function)(SNES,Vec,Vec,void*);
  PetscErrorCode (*form_coefficient)(DM,Vec,DM,Vec,void*);
  PetscErrorCode (*create_coefficient)(FD);
  PetscErrorCode (*create)(FD);
  PetscErrorCode (*jacobian_prealloc)(FD);
};

// FD struct
struct _p_FD {
  FDPDEOps        ops;
  DM              dmstag,dmcoeff;
  Mat             J;
  Vec             x,xold,r,coeff;
  DMStagBCList    bclist;
  void           *user_context;
  FDPDEType       type;
  char           *description, *description_bc, *description_coeff;
  SNES            snes;
  MPI_Comm        comm;
  PetscInt        Nx,Nz;
  PetscScalar     x0,x1,z0,z1;
  PetscBool       setupcalled, solvecalled;
};

// ---------------------------------------
// Function definitions
// ---------------------------------------
PetscErrorCode FDCreate(MPI_Comm, PetscInt, PetscInt, 
                        PetscScalar, PetscScalar, PetscScalar, PetscScalar, 
                        FDPDEType, FD*);
PetscErrorCode FDSetUp(FD);
PetscErrorCode FDDestroy(FD*);
PetscErrorCode FDView(FD);

PetscErrorCode FDSetFunctionBCList(FD, PetscErrorCode (*evaluate)(DM,Vec,DMStagBCList,void*), const char description[], void*);
PetscErrorCode FDSetFunctionCoefficient(FD, PetscErrorCode (*form_coefficient)(DM,Vec,DM,Vec,void*), const char description[], void*);

PetscErrorCode FDGetDM(FD, DM*);
PetscErrorCode FDGetSolution(FD, Vec*);

PetscErrorCode FDJacobianPreallocator(FD);
PetscErrorCode FDCreateSNES(MPI_Comm, FD);
PetscErrorCode FDSetOptionsPrefix(FD,const char[]);
PetscErrorCode FDConfigureSNES(FD);
PetscErrorCode FDSolveSNES(FD);
PetscErrorCode FDSolve(FD);

#endif