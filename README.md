# RIFT-O-MAT

**RIFT-O-MAT**
Magma-Assisted Tectonics: two-phase dynamics of oceanic and continental rifts

Goal: create analytical and numerical tools to understand how magmatism promotes and shapes rifts in continental and oceanic lithosphere.

Documents:
* [Project Roadmap](https://www.overleaf.com/read/hsrcbxxxspsh). 
* [PDE-Benchmarks](https://www.overleaf.com/read/xzymvxdcwrrm). 

### FD-PDE Framework

This **[Wiki](https://adina@bitbucket.org/adina/rift-o-mat.git/wiki)** contains more information on the **FD-PDE framework**. 

#### Executables
In `FD-PDE/src/`:
- Clean executables: `make clean_all`
- Make tests: `make tests`
- Make models: `make models`
- Run tests (example): `./tests/test_fdpde.app`
- Visualization: using python modules and examples in `/utils/`. More info in [Wiki](https://adina@bitbucket.org/adina/rift-o-mat.git/wiki).

#### Documentation
Compile LateX documentation in `/docs/` with: `make docs`. Currently, it requires Inkscape to compile illustrations (.svg) into pdf and include them in LaTeX documents. Install Inkscape through macports: `sudo port install inkscape`.

#### Install PETSc

The current Petsc version (3.13) should be obtained from [petsc](https://gitlab.com/petsc/petsc.git):

`git clone -b maint https://gitlab.com/petsc/petsc.git petsc`

Do `git pull` in the petsc directory anytime to obtain new patches that have been added.

Configure options:

DEBUG:
`./configure --prefix=<PATH> --download-fblaslapack=1 --download-pastix=1 --download-hdf5=1 --download-scalapack --download-parmetis --download-metis --download-cmake --with-debugging=1 --download-mpich=1 --enable-shared=1 --download-ptscotch=1 --with-cxx-dialect=C++11 --download-superlu_dist=1 --download-suitesparse=1 --download-ml=1 --download-hypre=1 --download-hwloc --download-mpi4py=1 --download-petsc4py=1 --download-make`

#### Python

We use python for testing and post-processing. A default 3.x python installation should be enough for tests. However, it's best practice to use virtual environments, such that you can work on multiple projects with various packages/versions and not have conflicts (see **[Wiki](https://adina@bitbucket.org/adina/rift-o-mat.git/wiki)**).

My preferred way to install python is through [anaconda3](https://www.anaconda.com) (multi-platform), which will install all the right executables (especially conda, which is similar to brew/port). The executables should be installed in `/Users/user/anaconda3/bin/`. Check: `which python`
and `which conda`.

Update anaconda (occasionally) with: `conda update --all`

### StagRidge code executables (legacy)
In `StagRidge/src/`:

- Clean executables: `make clean_all`
- Make executables: `make all`
- Run: `mpiexec -n 1 ./stagridge -options_file <fname> <other_options>`
- Tests (requires python): in /tests/ execute `./run_tests.sh`

