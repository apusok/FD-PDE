/* <FDPDE> contains Finite Differences PDE (FD-PDE) object */

#ifndef FDPDE_H
#define FDPDE_H

#include "petsc.h"
#include "prealloc_helper.h"
#include "dmstagbclist.h"
#include "snes_picard.h"

// ---------------------------------------
// Enum definitions
// ---------------------------------------
// FD-PDE type
typedef enum { FDPDE_UNINIT = 0, FDPDE_STOKES, FDPDE_ADVDIFF, FDPDE_STOKESDARCY2FIELD, FDPDE_COMPOSITE } FDPDEType;

// ---------------------------------------
// Struct definitions
// ---------------------------------------
// FD-PDE Options - Functions
typedef struct _FDPDEOps *FDPDEOps;
typedef struct _p_FDPDE *FDPDE;

struct _FDPDEOps {
  PetscErrorCode (*form_function)(SNES,Vec,Vec,void*);
  PetscErrorCode (*form_function_split)(SNES,Vec,Vec,Vec,void*);
  PetscErrorCode (*form_jacobian)(SNES,Vec,Mat,Mat,void*);
  PetscErrorCode (*form_coefficient)(FDPDE,DM,Vec,DM,Vec,void*);
  PetscErrorCode (*form_coefficient_split)(FDPDE,DM,Vec,Vec,DM,Vec,void*);
  PetscErrorCode (*create_jacobian)(FDPDE,Mat*);
  PetscErrorCode (*create)(FDPDE);
  PetscErrorCode (*view)(FDPDE);
  PetscErrorCode (*destroy)(FDPDE);
  PetscErrorCode (*setup)(FDPDE);
};

// FD-PDE struct
struct _p_FDPDE {
  FDPDEOps        ops;
  DM              dmstag,dmcoeff;
  Mat             J;
  Vec             x,xguess,r,coeff;
  DMStagBCList    bclist;
  void           *data;
  void           *user_context;
  FDPDEType       type;
  char           *description, *description_bc, *description_coeff;
  SNES            snes;
  MPI_Comm        comm;
  PetscInt        Nx,Nz;
  PetscInt        dof0,dof1,dof2;
  PetscInt        dofc0,dofc1,dofc2;
  PetscScalar     x0,x1,z0,z1;
  PetscBool       setupcalled,linearsolve;
  PetscInt        naux_global_vectors;
  Vec            *aux_global_vectors;
  PetscInt        refcount;
};

// ---------------------------------------
// Function definitions
// ---------------------------------------
PetscErrorCode FDPDECreate(MPI_Comm, PetscInt, PetscInt, 
                        PetscScalar, PetscScalar, PetscScalar, PetscScalar, 
                        FDPDEType, FDPDE*);
PetscErrorCode FDPDESetUp(FDPDE);
PetscErrorCode FDPDEDestroy(FDPDE*);
PetscErrorCode FDPDEView(FDPDE);
PetscErrorCode FDPDESolve(FDPDE,PetscBool*);
PetscErrorCode FDPDESolvePicard(FDPDE,PetscBool*);

PetscErrorCode FDPDESetFunctionBCList(FDPDE, PetscErrorCode (*evaluate)(DM,Vec,DMStagBCList,void*), const char description[], void*);
PetscErrorCode FDPDESetFunctionCoefficient(FDPDE, PetscErrorCode (*form_coefficient)(FDPDE, DM,Vec,DM,Vec,void*), const char description[], void*);
PetscErrorCode FDPDESetFunctionCoefficientSplit(FDPDE, PetscErrorCode (*form_coefficient)(FDPDE, DM,Vec,Vec,DM,Vec,void*), const char description[], void*);
PetscErrorCode FDPDESetLinearPreallocatorStencil(FDPDE, PetscBool);

PetscErrorCode FDPDEGetDM(FDPDE,DM*);
PetscErrorCode FDPDEGetSolution(FDPDE,Vec*);
PetscErrorCode FDPDEGetSNES(FDPDE,SNES*);
PetscErrorCode FDPDEGetDMStagBCList(FDPDE,DMStagBCList*);
PetscErrorCode FDPDEGetCoefficient(FDPDE,DM*,Vec*);
PetscErrorCode FDPDEGetSolutionGuess(FDPDE,Vec*);

PetscErrorCode FDPDEGetCoordinatesArrayDMStag(FDPDE,PetscScalar***,PetscScalar***);
PetscErrorCode FDPDERestoreCoordinatesArrayDMStag(FDPDE,PetscScalar**,PetscScalar**);

PetscErrorCode FDPDECreate2(MPI_Comm,FDPDE*);
PetscErrorCode FDPDESetType(FDPDE,FDPDEType);
PetscErrorCode FDPDESetSizes(FDPDE,PetscInt,PetscInt,PetscScalar,PetscScalar,PetscScalar,PetscScalar);
PetscErrorCode FDPDEGetAuxGlobalVectors(FDPDE,PetscInt*,Vec**);
PetscErrorCode FDPDEFormCoefficient(FDPDE fd);

#endif
