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
  DMStagBC *bc_list;
  PetscInt nbc;
  void   *coeff_context;
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
PetscErrorCode FDSetType(FD, enum FDPDEType);
// PetscErrorCode FDSetDM(FD, DM*);
// PetscErrorCode FDGetDM(FD, DM*);
PetscErrorCode FDGetSolution(FD, Vec*, Vec*);
PetscErrorCode FDCreateSNES(MPI_Comm, FD);
PetscErrorCode FDSetOptionsPrefix(FD,const char[]);
PetscErrorCode FDConfigureSNES(FD);
PetscErrorCode FDSolveSNES(FD);
PetscErrorCode FDSetSolveSNES(FD);

#endif
