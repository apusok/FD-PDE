# ---------------------------------------
# Run Rayleigh-Taylor instability test with particles
# ---------------------------------------

# Import modules
import os
import sys, getopt

print('# --------------------------------------- #')
print('# Rayleigh-Taylor Instability+Particles (STOKES) ')
print('# --------------------------------------- #')

# Input file
fname = 'out_stokes_rt'
try:
  os.mkdir(fname)
except OSError:
  pass

# Get cpu number
ncpu = 1
options, remainder = getopt.getopt(sys.argv[1:],'n:')
for opt, arg in options:
  if opt in ('-n'):
    ncpu = int(arg)

solver_default = ' -snes_monitor -snes_converged_reason -ksp_monitor -ksp_converged_reason -log_view'
if (ncpu == -1):
  solver = ' -pc_type lu -pc_factor_mat_solver_type umfpack -pc_factor_mat_ordering_type external'
  ncpu = 1
else:
  solver = ' -pc_type lu -pc_factor_mat_solver_type mumps'

str1 = 'mpiexec -n '+str(ncpu)+' ../test_stokes_rt.app'+solver+solver_default+' -snes_type ksponly -snes_fd_color -output_dir '+fname+ \
    ' -nt 101 -nx 21 -nz 21 > log_'+fname+'.out'
print(str1)
os.system(str1)

# DMSwarmViewXDMF() doesn't work with directory prefix
os.system('mv -f *.pbin *.xmf '+fname)