# Test restart feature in the MORBuoyancy model

import os

# Model options
model_opts = ' -L 200.0e3 -H 100.0e3 -zmin -100e3 -output_file out_model'+ \
             ' -nx 20 -nz 10 -U0 4 -K0 1e-7 -eta0 1e18 -zeta0 1e19 -Tp 1648 -potentialtemp 1'+ \
             ' -tstep 10 -tout 1 -tmax 1e6 -dtmax 1e2 -ts_scheme 2 '+ \
             ' -xsill 6e3 -extract_mech 0 -buoyancy 0 -visc_shear 0 -visc_bulk 1 '+ \
             ' -D1_guard 1 -visc_bulk1 1 -phi_min 1e-6 -phi_init 0.0 '

solver_opts = ' -hc_pc_type lu -hc_pc_factor_mat_solver_type umfpack -hc_pc_factor_mat_ordering_type external'+ \
              ' -hc_snes_atol 1e-10 -hc_snes_rtol 1e-20' + \
              ' -pv_pc_type lu -pv_pc_factor_mat_solver_type umfpack -pv_pc_factor_mat_ordering_type external'+ \
              ' -pv_snes_atol 1e-10 -pv_snes_rtol 1e-20'+ \
              ' -pv_snes_monitor -pv_snes_converged_reason -pv_ksp_converged_reason -pv_ksp_monitor -log_view'+ \
              ' -hc_snes_monitor -hc_snes_converged_reason -hc_ksp_converged_reason -hc_ksp_monitor'

# Non-dimensional
str1 = '../MORbuoyancy.app '+model_opts+solver_opts+' > log_out_nd_orig.out'
print('nd_orig')
os.system(str1)
str1 = '../MORbuoyancy.app '+model_opts+solver_opts+' -restart 1 > log_out_nd_r1.out'
print('nd_r1')
os.system(str1)
str1 = '../MORbuoyancy.app '+model_opts+solver_opts+' -restart 5 > log_out_nd_r5.out'
print('nd_r5')
os.system(str1)

os.system('rm -r Timestep*')

