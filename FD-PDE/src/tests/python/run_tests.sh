#!/bin/bash

# change to python
# check if executables exist and throw error (try/error)
# flag to run fast / all tests with '--all'; need to differentiate between them
# compare log files with diff or another parser, create an output file and if the file is empty->pass, otherwise fail
# check valgrind, log files

# Tests
echo -e "TESTS:"
solveropts="-pc_type lu -pc_factor_mat_solver_type umfpack -snes_monitor -snes_converged_reason -ksp_monitor -ksp_converged_reason"

test ="PYTHON"
SECONDS=0
fname="advdiff_advtime"
echo -e "  test_$fname.py > log_$fname.out [0h 8m 46s]"
python test_$fname.py > log_$fname.out
time=$SECONDS
echo "    >> took $(($time/3600))h $(($time%3600/60))m $(($time%60))s"

test ="PYTHON"
SECONDS=0
fname="advdiff_elman"
echo -e "  test_$fname.py > log_$fname.out [0h 0m 8s]"
python test_$fname.py > log_$fname.out
time=$SECONDS
echo "    >> took $(($time/3600))h $(($time%3600/60))m $(($time%60))s"

test ="PYTHON"
SECONDS=0
fname="advdiff_laplace"
echo -e "  test_$fname.py > log_$fname.out [0h 0m 1s]"
python test_$fname.py > log_$fname.out
time=$SECONDS
echo "    >> took $(($time/3600))h $(($time%3600/60))m $(($time%60))s"

test ="PYTHON"
SECONDS=0
fname="advdiff_mms_2d_diffusion"
echo -e "  test_$fname.py > log_$fname.out [0h 0m 7s]"
python test_$fname.py > log_$fname.out
time=$SECONDS
echo "    >> took $(($time/3600))h $(($time%3600/60))m $(($time%60))s"

test ="PYTHON"
SECONDS=0
fname="advdiff_mms_convergence"
echo -e "  test_$fname.py > log_$fname.out [0h 15m 59s]"
python test_$fname.py > log_$fname.out
time=$SECONDS
echo "    >> took $(($time/3600))h $(($time%3600/60))m $(($time%60))s"

test ="FAST"
SECONDS=0
fname="composite_laplace"
echo -e "  test_$fname.c > log_$fname.out [0h 0m 1s]"
../test_$fname.app $solveropts -nx 21 -nz 11 > log_$fname.out
time=$SECONDS
echo "    >> took $(($time/3600))h $(($time%3600/60))m $(($time%60))s"

test ="FAST"
SECONDS=0
fname="coupling"
echo -e "  test_$fname.c > log_$fname.out [0h 0m 1s]"
../test_$fname.app -nx 31 -nz 11 > log_$fname.out
time=$SECONDS
echo "    >> took $(($time/3600))h $(($time%3600/60))m $(($time%60))s"

# test ="PYTHON"
SECONDS=0
fname="decoupled_convection"
echo -e "  test_$fname.py > log_$fname.out [0h 10m 10s]"
python test_$fname.py > log_$fname.out
time=$SECONDS
echo "    >> took $(($time/3600))h $(($time%3600/60))m $(($time%60))s"

test ="FAST"
SECONDS=0
fname="dmstagbclist"
echo -e "  test_$fname.c > log_$fname.out [1s]"
../test_$fname.app -nx 21 -nz 11 > log_$fname.out
time=$SECONDS
echo "    >> took $(($time/3600))h $(($time%3600/60))m $(($time%60))s"

test ="PYTHON"
SECONDS=0
fname="dmstagoutput"
echo -e "  test_$fname.py > log_$fname.out [5s]"
python test_$fname.py > log_$fname.out
mkdir -p out_$fname
mv -f out_test* out_$fname/
rm -r __pycache__
time=$SECONDS
echo "    >> took $(($time/3600))h $(($time%3600/60))m $(($time%60))s"

test ="PYTHON"
SECONDS=0
fname="effvisc_mms"
echo -e "  test_$fname.py > log_$fname.out [0h 56m 31s]"
python test_$fname.py > log_$fname.out
time=$SECONDS
echo "    >> took $(($time/3600))h $(($time%3600/60))m $(($time%60))s"

test ="FAST"
SECONDS=0
fname="fdpde"
echo -e "  test_$fname.c > log_$fname.out [0h 0m 1s]"
../test_$fname.app $solveropts -nx 21 -nz 21 > log_$fname.out
time=$SECONDS
echo "    >> took $(($time/3600))h $(($time%3600/60))m $(($time%60))s"

test ="FAST"
SECONDS=0
fname="material_point"
echo -e "  test_$fname.c > log_$fname.out [0h 0m 1s]"
../test_$fname.app $solveropts -nx 21 -nz 21 > log_$fname.out
time=$SECONDS
echo "    >> took $(($time/3600))h $(($time%3600/60))m $(($time%60))s"

test ="PYTHON"
SECONDS=0
fname="stokes_lid_driven"
echo -e "  test_$fname.py > log_$fname.out [0h 1m 13s]"
python test_$fname.py > log_$fname.out
time=$SECONDS
echo "    >> took $(($time/3600))h $(($time%3600/60))m $(($time%60))s"

test ="PYTHON"
SECONDS=0
fname="stokes_mor"
echo -e "  test_$fname.py > log_$fname.out [27s]"
python test_$fname.py > log_$fname.out
time=$SECONDS
echo "    >> took $(($time/3600))h $(($time%3600/60))m $(($time%60))s"

test ="FAST"
SECONDS=0
fname="stokes_solcx_vargrid"
echo -e "  test_$fname.c > log_$fname.out [1s]"
../test_$fname.app $solveropts -nx 21 -nz 21 > log_$fname.out
time=$SECONDS
echo "    >> took $(($time/3600))h $(($time%3600/60))m $(($time%60))s"

test ="PYTHON"
SECONDS=0
fname="stokes_rt"
echo -e "  test_$fname.c > log_$fname.out [10s]"
mkdir -p out_$fname
../test_$fname.app $solveropts -snes_type ksponly -snes_fd_color -output_dir out_stokes_rt -nt 101 -nx 21 -nz 21 > log_$fname.out
# DMSwarmViewXDMF() doesn't work with directory prefix
mv -f *.pbin *.xmf out_$fname/
time=$SECONDS
echo "    >> took $(($time/3600))h $(($time%3600/60))m $(($time%60))s"

test ="PYTHON"
SECONDS=0
fname="stokes_solcx"
echo -e "  test_$fname.py > log_$fname.out [120s]"
python test_$fname.py > log_$fname.out
time=$SECONDS
echo "    >> took $(($time/3600))h $(($time%3600/60))m $(($time%60))s"

test ="PYTHON"
SECONDS=0
fname="stokesdarcy2field_mms_compare_nd"
echo -e "  test_$fname.py > log_$fname.out [350s]"
python test_$fname.py > log_$fname.out
time=$SECONDS
echo "    >> took $(($time/3600))h $(($time%3600/60))m $(($time%60))s"

test ="PYTHON"
SECONDS=0
fname="stokesdarcy2field_mms_katz_ch13"
echo -e "  test_$fname.py > log_$fname.out [180s]"
python test_$fname.py > log_$fname.out
time=$SECONDS
echo "    >> took $(($time/3600))h $(($time%3600/60))m $(($time%60))s"

test ="PYTHON"
SECONDS=0
fname="stokesdarcy2field_mms_porosity"
echo -e "  test_$fname.py > log_$fname.out [Xs]"
python test_$fname.py > log_$fname.out
time=$SECONDS
echo "    >> took $(($time/3600))h $(($time%3600/60))m $(($time%60))s"

test ="PYTHON"
SECONDS=0
fname="stokesdarcy2field_mms_rhebergen_siam_2014"
echo -e "  test_$fname.py > log_$fname.out [1900s]"
python test_$fname.py > log_$fname.out
time=$SECONDS
echo "    >> took $(($time/3600))h $(($time%3600/60))m $(($time%60))s"

test ="FAST"
SECONDS=0
fname="trueboundary"
echo -e "  test_$fname.c > log_$fname.out [1s]"
../test_$fname.app -nx 21 -nz 11 > log_$fname.out
time=$SECONDS
echo "    >> took $(($time/3600))h $(($time%3600/60))m $(($time%60))s"

# Others - not included
# test_plastic_indenter.c
# test_vp_inclusion_gerya.c