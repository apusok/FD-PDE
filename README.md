# RIFT-O-MAT

**RIFT-O-MAT**
Magma-Assisted Tectonics: two-phase dynamics of oceanic and continental rifts

Goal: create analytical and numerical tools to understand how magmatism promotes and shapes rifts in continental and oceanic lithosphere.

### FD-PDE Framework
In `FD-PDE/src/`:
- Clean executables: `make clean_all`
- Make tests: `make tests`
- Run tests: 
- Visualization: using python modules and examples in /utils/

### StagRidge code executables
In `StagRidge/src/`:

- Clean executables: `make clean_all`
- Make executables: `make all`
- Run: `mpiexec -n 1 ./stagridge -options_file <fname> <other_options>`
- Tests (requires python): in /tests/ execute `./run_tests.sh`

### Documentation
Compile LateX documentation in `/docs/` with: `make docs`

## Installation
### PETSc

The current Petsc version should be obtained from [petsc-bitbucket](https://bitbucket.org/petsc/petsc/src/master/):

`git clone <petsc bitbucket>`

`git checkout dmay/fix-MatPreallocator/maint-squash`

Configure options:

DEBUG:
`./configure --prefix=<PATH> --download-fblaslapack=1 --download-pastix=1 --download-mpi4py=1 --download-hdf5=1 --download-scalapack --download-parmetis --download-metis --download-cmake --with-debugging=1 --download-mpich=1 --enable-shared=1 --download-petsc4py=1 --download-ptscotch=1 --with-cxx-dialect=C++11 --download-superlu_dist=1 --download-suitesparse=1 --download-ml=1 --download-hypre=1`

### Python
We use python for testing and post-processing. A default 3.x python installation should be enough for tests. However, it's best practice to use virtual environments, such that you can work on multiple projects with various packages/versions and not have conflicts.

My preferred way to install python is through [anaconda3](https://www.anaconda.com) (multi-platform), which will install all the right executables (especially conda, which is similar to brew/port). The executables should be installed in `/Users/user/anaconda3/bin/`. Check: `which python`
and `which conda`.

Update anaconda (occasionally) with: `conda update --all`

#### Create python virtual environment 
 We can create an environment for this project called `riftomat` in the following way (also install some libraries):

`conda create --name riftomat python=3.7 matplotlib pip numpy pandas xarray basemap`

Activate the environment with: `conda activate riftomat`. Active environment will show this at prompt: `(riftomat)$`.

More libraries can be installed inside the environment with:
`conda install jupyter Pillow netcdf4`

Exit the environment: `conda deactivate`

If you need to check install options for a specific environment, the default virtual environment location is: `/Users/user/anaconda3/envs/`

### Inkscape 
Create illustrations (.svg) and then compile images during `make docs` to include in LaTeX documents. Install through macports: `sudo port install inkscape`

