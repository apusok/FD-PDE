# ---------------------------------------
# Laplace (ADVDIFF) benchmark \nabla^2 T = 0
# Plot analytical and numerical solution
# ---------------------------------------

# Import modules
import numpy as np
import matplotlib.pyplot as plt
import importlib
import os
import shutil 
import glob
import sys, getopt

# Input file
fname = 'out_num_advdiff_laplace'
fname_out = 'out_advdiff_laplace'
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


print('# --------------------------------------- #')
print('# Laplace (ADVDIFF) benchmark ')
print('# --------------------------------------- #')

n = 100

# Run test
solver_default = ' -snes_monitor -snes_converged_reason -ksp_monitor -ksp_converged_reason '

# Use umfpack for sequential and mumps for parallel
if (ncpu == 1):
  solver = ' -pc_type lu -pc_factor_mat_solver_type umfpack -pc_factor_mat_ordering_type external'
else:
  solver = ' -pc_type lu -pc_factor_mat_solver_type mumps'

str1 = 'mpiexec -n '+str(ncpu)+' ../test_advdiff_laplace.app'+solver+ \
  ' -output_dir '+fname_data+' -output_file '+fname+' -nx '+str(n)+ \
  ' -nz '+str(n)+solver_default+' > '+fname_data+'/'+fname+'.out'
print(str1)
os.system(str1)

# Plot solution (numerical and analytical)
# imod = importlib.import_module(fname) # works only in current directory

spec = importlib.util.spec_from_file_location(fname,fname_data+'/'+fname+'.py')
imod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(imod)

# Load data
data = imod._PETScBinaryLoad()
imod._PETScBinaryLoadReportNames(data)

# Get data
m = data['Nx'][0]
n = data['Ny'][0]
xc = data['x1d_cell']
yc = data['y1d_cell']
T = data['X_cell']
celldof   = int(len(T)/(m*n))

# Open a figure
fig, axs = plt.subplots(1, 2,figsize=(12,6))

ax1 = axs[0]
contours = ax1.contour(xc,yc, T.reshape(n,m), 5 , colors='white',linestyles='solid',linewidths=0.5)
ax1.clabel(contours, contours.levels, inline=True, fontsize=8)
im = ax1.imshow(T.reshape(n,m), extent=[min(xc), max(xc), min(yc), max(yc)],
                origin='lower', interpolation='nearest' )
ax1.axis('image')
ax1.set_xlabel('x-dir')
ax1.set_ylabel('z-dir')
ax1.set_title('Laplace solution (numerical)')

# Plot analytical solution
fname1 = 'out_analytic_solution_laplace'

# Load python module describing data
# imod = importlib.import_module(fname1)

spec = importlib.util.spec_from_file_location(fname1,fname_data+'/'+fname1+'.py')
imod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(imod)

# Load data
data = imod._PETScBinaryLoad()
imod._PETScBinaryLoadReportNames(data)

# Get data
m = data['Nx'][0]
n = data['Ny'][0]
xc = data['x1d_cell']
yc = data['y1d_cell']
T = data['X_cell']
celldof   = int(len(T)/(m*n))

ax2 = axs[1]
contours = ax2.contour(xc,yc, T.reshape(n,m), 5 , colors='white',linestyles='solid',linewidths=0.5)
ax2.clabel(contours, contours.levels, inline=True, fontsize=8)
im = ax2.imshow(T.reshape(n,m), extent=[min(xc), max(xc), min(yc), max(yc)],
                origin='lower', interpolation='nearest' )
ax2.axis('image')
ax2.set_xlabel('x-dir')
ax2.set_title('Laplace solution (analytical)')

cbar = fig.colorbar(im,ax=axs, shrink=0.75)
cbar.ax.set_ylabel('T')

plt.savefig(fname_out+'/'+fname+'.pdf')

os.system('rm -r '+fname_data+'/__pycache__')