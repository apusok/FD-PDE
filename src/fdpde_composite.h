
#ifndef __fdpde_composite_h__
#define __fdpde_composite_h__

#include "fdpde.h"

// ---------------------------------------
// Function definitions
// ---------------------------------------
PetscErrorCode FDPDECompositeSetFDPDE(FDPDE,PetscInt,FDPDE*);
PetscErrorCode FDPDECompositeGetFDPDE(FDPDE,PetscInt*,FDPDE**);
PetscErrorCode FDPDECompositeSynchronizeGlobalVectors(FDPDE,Vec);
// PetscErrorCode FDPDECreateComposite(MPI_Comm comm,PetscInt n,FDPDE pdelist[],FDPDE *fd);

PetscErrorCode FDPDESetUp_Composite(FDPDE);
PetscErrorCode FDPDECreate_Composite(FDPDE);
PetscErrorCode FormFunction_Composite(SNES,Vec,Vec,void*);
PetscErrorCode JacobianCreate_Composite(FDPDE,Mat*);
PetscErrorCode FDPDEView_Composite(FDPDE);
PetscErrorCode FDPDEDestroy_Composite(FDPDE);

#endif
