# ------------------------------------------------ #
# MMS sympy for 2D diffusion: div(k*grad(T)) = f
# ------------------------------------------------ #

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os
import shutil 
import glob
import sys, getopt

# Input file
f1 = 'out_mms_2d_diff_num_solution'
fname_out = 'out_advdiff_mms_2d_diffusion'
fname_data = fname_out+'/data'
try:
  os.mkdir(fname_out)
except OSError:
  pass

try:
  os.mkdir(fname_data)
except OSError:
  pass

# Get cpu number
ncpu = 1
options, remainder = getopt.getopt(sys.argv[1:],'n:')
for opt, arg in options:
  if opt in ('-n'):
    ncpu = int(arg)

# Parameters
n = [25, 40, 50, 80, 100, 125, 150, 200, 300, 400]
solver_default = ' -snes_monitor -snes_converged_reason -ksp_monitor -ksp_converged_reason '

# Use umfpack for sequential and mumps for sequential/parallel
if (ncpu == -1):
  solver = ' -pc_type lu -pc_factor_mat_solver_type umfpack -pc_factor_mat_ordering_type external'
  ncpu = 1
else:
  solver = ' -pc_type lu -pc_factor_mat_solver_type mumps'

# Run simulations
for nx in n:
  # Create output filename
  fout1 = f1+'_'+str(nx)+'.out'

  # Run with different resolutions
  str1 = 'mpiexec -n '+str(ncpu)+' ../test_advdiff_mms_2d_diffusion '+solver+' -fname '+f1+\
    ' -fdir '+fname_data+' -nx '+str(nx)+' -nz '+str(nx)+solver_default+' > '+fname_data+'/'+fout1
  print(str1)
  os.system(str1)

# Norm variables
nrm1 = np.zeros(len(n))
hx = np.zeros(len(n))

# Parse output and save norm info
for i in range(0,len(n)):
    nx = n[i]

    fout1 = fname_data+'/'+f1+'_'+str(nx)+'.out'

    # Open file 1 and read
    f = open(fout1, 'r')
    for line in f:
        if 'Solution:' in line:
            nrm1[i] = float(line[20:39])
        if 'Grid info:' in line:
            hx[i] = float(line[18:37])
    
    f.close()

x1 = [-2.6, -1.6]
y1 = [-6, -5]
y2 = [-6, -4]

# Print convergence orders:
hx_log    = np.log10(hx)
nrm1_log = np.log10(nrm1)

# Perform linear regression
sl1, intercept, r_value, p_value, std_err = linregress(hx_log, nrm1_log)

# Plot convergence data
plt.figure(1,figsize=(6,6))

plt.grid(color='lightgray', linestyle=':')
plt.plot(np.log10(hx),np.log10(nrm1),'ko--',label='Num slope = '+str(round(sl1,5)))

plt.plot(x1,y1,'r-',label='slope=1')
plt.plot(x1,y2,'b-',label='slope=2')

plt.xlabel('log10(h)',fontweight='bold',fontsize=12)
plt.ylabel('log10||e||',fontweight='bold',fontsize=12)
plt.legend()

plt.savefig(fname_out+'/'+f1+'.pdf')

print('# --------------------------------------- #')
print('# MMS 2D diff convergence order:')
print('     T_slope = '+str(sl1))

# # os.system('rm -r __pycache__')