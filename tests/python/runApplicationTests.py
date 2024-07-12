# ---------------------------------------
# Run all application tests in this directory (sequential and parallel)
# Run command:
#   python runApplicationTests.py       -> for sequential run
#   python runApplicationTests.py -n 2  -> for parallel run with -n #ncpu
#
# Warning: 
#   The following is implied: use umfpack for sequential and mumps for parallel.
#   Installed packages and existence of packages are handled by petsc, not here.
# ---------------------------------------

# Import libraries
import os, sys, getopt, time

tests_seq = ['advdiff_advtime',
          'advdiff_elman',
          'advdiff_laplace',
          'advdiff_mms_2d_diffusion',
          'advdiff_mms_convergence',
          'decoupled_convection',
          'dmstagoutput',
          'effvisc_mms',
          'stokes_lid_driven',
          'stokes_mor',
          'stokes_rt',
          'stokes_solcx',
          'stokesdarcy2field_mms_compare_nd',
          'stokesdarcy2field_mms_katz_ch13',
          'stokesdarcy2field_mms_porosity',
          'stokesdarcy2field_mms_rhebergen_siam_2014',
          'stokesdarcy3field_mms_bulkviscosity',
          'enthalpy_2d_diffusion',
          'enthalpy_1d_eutectic_solidification',
          'enthalpy_1d_solidification_TC',
          'advdiff_periodic',
          'enthalpy_periodic',
          'convection_stokes_periodic',
          'stokes_rt_compare_pic_phasefield',
          'dmstag_phasemethod_flow',
          'dmstag_phasemethod_stationary',
          'stokesdarcy2field_beam',
          'stokesdarcy2field_plume',
          'stokesdarcy2field_rt',
          'stokesdarcy2field_rt_analytical',
          'stokesdarcy2field_vep_0d_expansion',
          'stokesdarcy2field_vep_0d_shear',
          'stokesdarcy2field_vep_inclusion',
          'stokesdarcy2field_vep_inclusion_pic',
          'stokesdarcy2field_vp_inclusion']

tests_mpi = ['advdiff_advtime',
          'advdiff_elman',
          'advdiff_laplace',
          'advdiff_mms_2d_diffusion',
          'advdiff_mms_convergence',
          'dmstagoutput',
          'effvisc_mms',
          'stokes_lid_driven',
          'stokes_mor',
          'stokes_solcx',
          'stokes_rt',
          'stokes_rt_compare_pic_phasefield',
          'stokesdarcy2field_mms_compare_nd',
          'stokesdarcy2field_mms_katz_ch13',
          'stokesdarcy2field_mms_porosity',
          'stokesdarcy2field_mms_rhebergen_siam_2014',
          'enthalpy_2d_diffusion',
          'enthalpy_1d_eutectic_solidification',
          'enthalpy_1d_solidification_TC',
          'advdiff_periodic',
          'enthalpy_periodic',
          'stokesdarcy2field_vep_inclusion_pic']

# tests_mpi_fail = ['decoupled_convection',
#           'stokesdarcy3field_mms_bulkviscosity',
#           'convection_stokes_periodic']

# Default run with -1cpu which uses umfpack # if ncpu>0 uses mumps (slower, but parallel)
ncpu = -1

# Command line options
options, remainder = getopt.getopt(sys.argv[1:],'n:')

for opt, arg in options:
  if opt in ('-n'):
    ncpu = int(arg)

output_dir = 'results_'+str(ncpu)+'cpu/'

try:
  os.mkdir(output_dir)
except OSError:
  pass

if (ncpu==-1): tests = tests_seq
else:          tests = tests_mpi

print('TESTS:')
# Run tests
for i in range(0,len(tests)):
  itest = tests[i]
  run_str  = ('python test_%s.py -n %d > log_%s_%dcpu.out' % (itest,ncpu,itest,ncpu))
  print('  %s' % run_str)

  start = time.time()
  os.system(run_str)
  end = time.time()
  hours, rem = divmod(end-start, 3600)
  minutes, seconds = divmod(rem, 60)
  print('    >> took %dh %dm %.2fs' % (hours,minutes,seconds))

# rm __pycache__
os.system('find . -name __pycache__ -type d -exec rm -r {} \;')

# mv everything
print('Reorganize output in: '+output_dir)
os.system('mv log* out* '+output_dir)
