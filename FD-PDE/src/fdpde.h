/* <FDPDE> contains Finite Differences PDE (FD-PDE) object */

#ifndef FDPDE_H
#define FDPDE_H

#include "petsc.h"
#include "prealloc_helper.h"
#include "dmstagbclist.h"

// ---------------------------------------
// Enum definitions
// ---------------------------------------
// FD-PDE type
typedef enum { FDPDE_UNINIT = 0, FDPDE_STOKES, FDPDE_ADVDIFF } FDPDEType;

// ---------------------------------------
// Struct definitions
// ---------------------------------------
// FD-PDE Options - Functions
typedef struct _FDPDEOps *FDPDEOps;
typedef struct _p_FDPDE *FDPDE;

struct _FDPDEOps {
  PetscErrorCode (*form_function)(SNES,Vec,Vec,void*);
  PetscErrorCode (*form_coefficient)(DM,Vec,DM,Vec,void*);
  PetscErrorCode (*create_coefficient)(FDPDE);
  PetscErrorCode (*create)(FDPDE);
  PetscErrorCode (*jacobian_prealloc)(FDPDE);
};

// FD-PDE struct
struct _p_FDPDE {
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
  PetscBool       setupcalled;
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
PetscErrorCode FDPDESolve(FDPDE);

PetscErrorCode FDPDESetFunctionBCList(FDPDE, PetscErrorCode (*evaluate)(DM,Vec,DMStagBCList,void*), const char description[], void*);
PetscErrorCode FDPDESetFunctionCoefficient(FDPDE, PetscErrorCode (*form_coefficient)(DM,Vec,DM,Vec,void*), const char description[], void*);

PetscErrorCode FDPDEGetDM(FDPDE,DM*);
PetscErrorCode FDPDEGetSolution(FDPDE,Vec*);
PetscErrorCode FDPDEGetSNES(FDPDE,SNES*);
PetscErrorCode FDPDEGetDMStagBCList(FDPDE,DMStagBCList*);
PetscErrorCode FDPDEGetCoefficient(FDPDE,DM*,Vec*);

PetscErrorCode FDPDEGetCoordinatesArrayDMStag(FDPDE,PetscScalar***,PetscScalar***);
PetscErrorCode FDPDERestoreCoordinatesArrayDMStag(FDPDE,PetscScalar**,PetscScalar**);

#endif