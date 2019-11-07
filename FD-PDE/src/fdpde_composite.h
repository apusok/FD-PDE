
#ifndef __fdpde_composite_h__
#define __fdpde_composite_h__

PetscErrorCode FDPDCompositeSetFDPDE(FDPDE,PetscInt,FDPDE*);
PetscErrorCode FDPDCompositeGetFDPDE(FDPDE,PetscInt*,FDPDE**);
PetscErrorCode FDPDECompositeUpdateState(FDPDE fd,Vec X);

#endif
