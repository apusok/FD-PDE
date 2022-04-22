# Release mbuoy3

**RIFT-O-MAT** (Magma-Assisted Tectonics): two-phase dynamics of oceanic and continental rifts

The goal of the project is to create analytical and numerical tools to understand how magmatism promotes and shapes rifts in continental and oceanic lithosphere.

Funding: This research received funding from the European Research Council under Horizon 2020 research and innovation program.

## Repository contents
- `FD-PDE/src/`: source code and tests for FD-PDE framework
- `FD-PDE/models/mbuoy3`: source code for mid-ocean ridge model
- `utils`: python routines for I/O of PETSc objects

## mbuoy3
**mbuoy3** is a 2-D mid-ocean ridge, two-phase flow model with buoyancy forces (porous, compositional, thermal). 

Publication: *Buoyancy-driven flow beneath mid-ocean ridges: the role of chemical heterogeneity*
Authors: Adina E. Pusok<sup>1</sup>, Richard F. Katz<sup>1</sup>, Dave A. May<sup>2</sup>, Yuan Li<sup>1</sup>

Affiliation: 
(1) Department of Earth Sciences, University of Oxford, Oxford, United Kingdom
(2) Scripps Institution of Oceanography, UC San Diego, La Jolla, CA, USA

To compile the code:
In `FD-PDE/models/mbuoy3/`:
- Clean executable: `make clean_all`
- Make executable: `make all`

Running the model: `./mbuoy3.app -options_file model_half_ridge.opts > log_out.out`

## FD-PDE Framework

The FD-PDE framework uses finite difference staggered grids for solving partial differential equations (FD-PDE) for single-/two-phase flow magma dynamics. 

The FD-PDE framework using PETSc [PETSc](https://petsc.org/release/) and make use of the new features for staggered grids, such as DMStag. Governing equations are discretized as default PDEs (i.e., FDPDE Type), and the user only specifies coefficients, constitutive equations, and boundary conditions.

Source code is located in `/FD-PDE/src/`

### FD-PDE Test executables
In `FD-PDE/src/`:
- Clean executables: `make clean_all`
- Make tests: `make tests`
- Run tests (example 1): `./tests/test_fdpde.app`
- Run tests (example 2): `python runApplicationTests.py` in `/FD-PDE/src/tests/python/`
- Visualization: using python examples in `/FD-PDE/src/tests/python/`.

### Install PETSc

The current Petsc version (3.14) should be obtained from [petsc](https://gitlab.com/petsc/petsc.git):

`git clone -b maint https://gitlab.com/petsc/petsc.git petsc`

Do `git pull` in the petsc directory anytime to obtain new patches that have been added.

Configure options:

DEBUG:
`./configure --prefix=<PATH> --download-fblaslapack=1 --download-pastix=1 --download-hdf5=1 --download-scalapack --download-parmetis --download-metis --download-cmake --with-debugging=1 --download-mpich=1 --enable-shared=1 --download-ptscotch=1 --with-cxx-dialect=C++11 --download-superlu_dist=1 --download-suitesparse=1 --download-ml=1 --download-hypre=1 --download-hwloc --download-mpi4py=1 --download-petsc4py=1 --download-make`

### Python

We use python for testing and post-processing. A default 3.x python installation should be enough for tests.

Preferred way to install python is through [anaconda3](https://www.anaconda.com) (multi-platform), which will install all the right executables (especially conda, which is similar to brew/port). The executables should be installed in `/Users/user/anaconda3/bin/`. Check: `which python`
and `which conda`.

Update anaconda (occasionally) with: `conda update --all`
