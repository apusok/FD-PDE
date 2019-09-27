/* <FD> contains Finite Differences (FD) PDE object */

#ifndef FD_H
#define FD_H

#include "petsc.h"
#include "bc.h"

// ---------------------------------------
// Enum definitions
// ---------------------------------------
// PDE type
enum FDPDEType { FD_UNINIT, STOKES, ADVDIFF };

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
  PetscErrorCode (*destroy)(FD);
  PetscErrorCode (*view)(FD,PetscViewer);
  PetscErrorCode (*jacobian_prealloc)(FD);
};

// FD struct
struct _p_FD {
  FDPDEOps ops;
  DM      dmstag,dmcoeff;
  Mat     J;
  Vec     x,xguess,r,coeff;
  BCList *bc_list;
  PetscInt nbc;
  //void   *coeff_context;
  void   *user_context;
  enum FDPDEType type;
  char   *description;
  SNES    snes;
  MPI_Comm comm;
  PetscInt Nx, Nz;
};

// // ---------------------------------------
// // Function definitions
// // ---------------------------------------
PetscErrorCode FDCreate(MPI_Comm, FD*);
PetscErrorCode FDDestroy(FD*);
PetscErrorCode FDView(FD, PetscViewer);
PetscErrorCode FDSetDimensions(FD, PetscInt, PetscInt);
PetscErrorCode FDSetType(FD, enum FDPDEType);
PetscErrorCode FDSetBCList(FD, BCList*, PetscInt);
PetscErrorCode FDSetFunctionCoefficient(FD, PetscErrorCode (*form_coefficient)(DM,Vec,DM,Vec,void*), void*);
PetscErrorCode FDGetDM(FD, DM*);
PetscErrorCode FDGetSolution(FD, Vec*, Vec*);
PetscErrorCode FDCreateSNES(MPI_Comm, FD);
PetscErrorCode FDSetOptionsPrefix(FD,const char[]);
PetscErrorCode FDConfigureSNES(FD);
PetscErrorCode FDSolveSNES(FD);
PetscErrorCode FDSetSolveSNES(FD);

#endif