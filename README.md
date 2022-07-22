# FD-PDE (Release version)
Release version of the FD-PDE framework developed for magma dynamics and other applications in Geodynamics.

The FD-PDE Framework was created within the **RIFT-O-MAT** (Magma-Assisted Tectonics) project, funded by the European Research Council under Horizon 2020 research and innovation program, and awarded to PI Richard Katz.

The goal of the RIFT-O-MAT project is to create analytical and numerical tools to understand how magmatism promotes and shapes rifts in continental and oceanic lithosphere.

## License: MIT License

## Contributors
- Adina E. Pusok
- Dave May
- Yuan Li
- Richard Katz

## Repository contents
- `src/`: source code for FD-PDE framework
- `tests/`: tests for FD-PDE framework
- `models`: model applications using the FD-PDE Framework
- `models/mbuoy3`: mid-ocean ridge code used in Pusok et al. (GJI, 2022)
- `utils`: python routines for I/O of PETSc objects

## Description
The **FD-PDE framework** uses finite difference staggered grids for solving partial differential equations (PDEs) for single-/two-phase flow magma dynamics. 

The FD-PDE framework is based on [PETSc](https://petsc.org/release/) and make use of the new features for staggered grids, such as DMStag. Governing equations are discretized as default PDEs (i.e., FDPDE Type), and the user only specifies coefficients, constitutive equations, and boundary conditions. 

Documentation for the FD-PDE Framework and tests can be found: [FD-PDE Benchmarks](https://drive.google.com/file/d/137Dtu2ykuf7zL_C8NnR_8sGFxV8qRWOw/view?usp=sharing).

Source code is located in `/FD-PDE/src/`.
Tests are located in `/FD-PDE/tests/`.

## Tests executables
In `FD-PDE/src/`:
- Clean executables: `make clean_all`
- Make tests: `make tests`
- Run tests (example 1): `./tests/test_fdpde.app`
- Run tests (example 2): `python runApplicationTests.py` in `/FD-PDE/tests/python/`
- Visualization: using python examples in `/FD-PDE/tests/python/`.

## Dependencies
### PETSc
The current Petsc version (3.14) should be obtained from [petsc](https://gitlab.com/petsc/petsc.git):

`git clone -b maint https://gitlab.com/petsc/petsc.git petsc`

Do `git pull` in the petsc directory anytime to obtain new patches that have been added.

Configure options (change `<PATH>` accordingly):

DEBUG:

`./configure --prefix=<PATH_DEBUG> --download-fblaslapack --download-hdf5 --download-mumps --download-scalapack --download-parmetis --download-metis --download-cmake --with-debugging --download-mpich --enable-shared --download-pastix --download-ptscotch --with-cxx-dialect=C++11 --download-superlu_dist --download-spooles --download-suitesparse --download-ml --download-hypre --download-hwloc --download-mpi4py --download-petsc4py --download-make`

OPTIMIZED:

`./configure --prefix=<PATH_OPT> --FOPTFLAGS=-O2 --CXXOPTFLAGS=-O2 --COPTFLAGS=-O2 --download-fblaslapack --download-hdf5 --download-mumps --download-scalapack --download-parmetis --download-metis --download-cmake --with-debugging=0 --download-mpich --enable-shared --download-pastix --download-ptscotch --with-cxx-dialect=C++11 --download-superlu_dist --download-spooles --download-suitesparse --download-ml --download-hypre --download-hwloc --download-mpi4py --download-petsc4py --download-make`

Specify PETSc environment variable for bash (can be specified in `~/.bashrc` or `~/.bash_profile`):

`export PETSC_DIR=<PATH>`

### Python

We use python for testing and post-processing. A default 3.x python installation should be enough for tests.

Preferred way to install python is through [anaconda3](https://www.anaconda.com) (multi-platform), which will install all the right executables (especially conda, which is similar to brew/port). The executables should be installed in `/Users/user/anaconda3/bin/`. Check: `which python`
and `which conda`.

Update anaconda (occasionally) with: `conda update --all`

To use the visualization tools developed within the FD-PDE Framework, update the environmental variable (change `<PATH_FDPDE>` accordingly):

`export PYTHONPATH=<PATH_FDPDE>/utils:${PETSC_DIR}/lib/petsc/bin`

# Models
## mbuoy3
**mbuoy3** is a 2-D mid-ocean ridge, two-phase flow model with buoyancy forces (porous, compositional, thermal). 

Publication: *Buoyancy-driven flow beneath mid-ocean ridges: the role of chemical heterogeneity*

Authors: Adina E. Pusok<sup>1</sup>, Richard F. Katz<sup>1</sup>, Dave A. May<sup>2</sup>, Yuan Li<sup>1</sup>

Affiliation: 
(1) Department of Earth Sciences, University of Oxford, Oxford, United Kingdom
(2) Scripps Institution of Oceanography, UC San Diego, La Jolla, CA, USA

To compile the code, in `FD-PDE/models/mbuoy3/`:
- Clean executable: `make clean_all`
- Make executable: `make all`

Running the model: `./mbuoy3.app -options_file model_half_ridge.opts > log_out.out`

### Test run
### Visualization
### Input files
### pMELTs Jupyter notebook

