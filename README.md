# RIFT-O-MAT

**RIFT-O-MAT**
Magma-Assisted Tectonics: two-phase dynamics of oceanic and continental rifts

Goal: create analytical and numerical tools to understand how magmatism promotes and shapes rifts in continental and oceanic lithosphere.

## Installation

### PETSc

The current Petsc version should be obtained from [petsc-bitbucket](https://bitbucket.org/petsc/petsc/src/master/):

`git clone <petsc bitbucket>`

`git checkout dmay/fix-MatPreallocator/maint-squash`

Configure options:

`./configure --prefix=<PATH> --download-fblaslapack=1 --download-pastix=1 --download-mpi4py=1 --download-hdf5=1 --download-scalapack --download-parmetis --download-metis --download-cmake --with-debugging=1 --download-mpich=1 --enable-shared=1 --download-petsc4py=1 --download-ptscotch=1 --with-cxx-dialect=C++11 --download-superlu_dist=1 --download-suitesparse=1 --download-ml=1 --download-hypre=1`

### Code executables
In `StagRidge/src/`:
* Clean executables: `make clean_all`
* Make executables: `make all`
* Run: `mpiexec -n 1 ./stagridge -options_file <fname> <other_options>`

### Documentation
Compile LateX documentation in `/docs/` with: `make docs`