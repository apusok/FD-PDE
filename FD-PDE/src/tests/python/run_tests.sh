#!/bin/bash

# FD-PDE
# test_composite_laplace.c
# test_coupling.c
# test_dmstagbclist.c 
# test_fdpde.c
# test_material_point.c
# test_stokes_rt.c
# test_stokes_solcx_vargrid.c

# Python
python test_dmstagoutput.py
python test_stokes_mor.py
python test_stokes_solcx.py
python test_advdiff_laplace.py
python test_advdiff_elman.py
python test_advdiff_mms_2d_diffusion.py
python test_advdiff_advtime.py
python test_stokesdarcy2field_mms_rhebergen_siam_2014.py
python test_stokesdarcy2field_mms_katz_ch13.py
python test_stokesdarcy2field_mms_compare_nd.py
python test_advdiff_mms_convergence.py
python test_stokesdarcy2field_mms_porosity.py
python test_decoupled_convection.py
python test_effvisc_mms.py

# Models
# python run_mor_mechanics.py