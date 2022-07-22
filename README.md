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
In `mbuoy3/test/` run:

1. Half-ridge model with:

`../mbuoy3.app -options_file model_half_ridge.opts > log_out.out`

2. Full-ridge model with:

`../mbuoy3.app -options_file model_full_ridge.opts > log_out.out`

The input files are actually identical, except for the option to switch on full-ridge mode: `-full_ridge 1`. This option will automatically extend the domain and adapt boundary conditions. 

### Visualization

First, make sure the `PYTHONPATH` is updated as above. Then, install some cool [Scientific Colormaps](https://www.fabiocrameri.ch/colourmaps/) from Fabio Crameri with:
`pip install cmcrameri`

Example visualization for either the half-ridge/full-ridge test. In `mbuoy3/python` run:
* `python plot_debug_output.py`
* `python plot_time_series.py`
* `plot_HR_sims_porosity.py` - for half-ridge
* `plot_FR_sims_porosity.py` - for full-ridge

More output routines can be found in `vizMORBuoyancy.py`, which can be loaded as a module in any new script.

### Input files
Input files to reproduce the simulations in Pusok et al. (GJI, 2022) are found in `mbuoy3/publication/input_files/`. All parameter variations are indicated in the manuscript. Some nomenclature for half-ridge models:
* b000 - no buoyancy (passive flow)
* b100 - porous buoyancy
* b120 - porous and compositional buoyancy

Some nomenclature for full-ridge models:
* F1 - temperature forcing, can be modified with `-forcing 1 # 0-off, 1-Temp, 2-Comp`
* F2 - compositional forcing
* dTdx, dCdx - indicates magnitude of forcing
 
The `submit_job.run` are SLURM submission files, included to help with cpu and time usage, and how to restart a simulation from a specified timestep.

### pMELTs Jupyter notebook

To install `ThermoEngine` (Ghiorso et al., 2002, Ghiorso and Wolf, 2019), follow instructions from the [ENKI website](https://enki-portal.gitlab.io/ThermoEngine/index.html). In addition, you'll need to install and start [Docker](https://docs.docker.com) before running the ENKI server.

Copy the Jupyter notebooks:
* `MOR_beta_revised.ipynb`
* `MOR_beta_revised_min.ipynb`

in `ThermoEngine/Notebooks/my_notebooks/` to reproduce the pMELTS simulations in this study.

In the `ThermoEngine` directory, run the script `./run_docker_locally.sh` to start the ENKI server locally (JupyterLab session). Run the Jupyter notebooks within the JupyterLab session.





