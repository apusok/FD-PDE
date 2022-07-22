
#ifndef __fdpde_composite_h__
#define __fdpde_composite_h__

PetscErrorCode FDPDECompositeSetFDPDE(FDPDE,PetscInt,FDPDE*);
PetscErrorCode FDPDECompositeGetFDPDE(FDPDE,PetscInt*,FDPDE**);
PetscErrorCode FDPDECompositeSynchronizeGlobalVectors(FDPDE fd,Vec X);
PetscErrorCode FDPDECreateComposite(MPI_Comm comm,PetscInt n,FDPDE pdelist[],FDPDE *fd);

#endif
