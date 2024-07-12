# ---------------------------------------
# ADVDIFF benchmarks from Elman (2005): 
# Example 3.1.1 
# Example 3.1.3 
# Example 3.1.4
# Plot analytical and numerical solution
# ---------------------------------------

# Import modules
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import importlib
import os
import shutil 
import glob
import sys, getopt

print('# --------------------------------------- #')
print('# Elman 2005 (ADVDIFF) benchmark ')
print('# --------------------------------------- #')

n = 100

fname_out = 'out_advdiff_elman'
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

# Input file
fname = 'out_num_solution_elman311'
fname11 = fname+'_upwind'
fname12 = fname+'_fromm'
fname_analytical = 'out_analytic_solution_elman311'
fname2 = 'out_num_solution_elman312'
fname3 = 'out_num_solution_elman313'
fname4 = 'out_num_solution_elman314'
solver_default = ' -snes_monitor -snes_converged_reason -ksp_monitor -ksp_converged_reason '

# Use umfpack for sequential and mumps for sequential/parallel
solver_default = ' -snes_monitor -snes_converged_reason -ksp_monitor -ksp_converged_reason '
if (ncpu == -1):
  solver = ' -pc_type lu -pc_factor_mat_solver_type umfpack -pc_factor_mat_ordering_type external'
  ncpu = 1
else:
  solver = ' -pc_type lu -pc_factor_mat_solver_type mumps'

# Run test
str1 = 'mpiexec -n '+str(ncpu)+' ../test_advdiff_elman'+solver+' -output_file '+fname11+' -output_dir '+fname_data+ \
  ' -nx '+str(n)+' -nz '+str(n)+solver_default+' -advtype 0 > '+fname_data+'/'+'out_num_elman1.out'
print(str1)
os.system(str1)

str2 = 'mpiexec -n '+str(ncpu)+' ../test_advdiff_elman'+solver+' -output_file '+fname12+' -output_dir '+fname_data+ \
  ' -nx '+str(n)+' -nz '+str(n)+solver_default+' -advtype 1  > '+fname_data+'/'+'out_num_elman2.out'
print(str2)
os.system(str2)

# ---------------------------------------
# Load python module describing data
# ---------------------------------------

# 0 Analytical 
# imod = importlib.import_module(fname_analytical)
spec = importlib.util.spec_from_file_location(fname_analytical,fname_data+'/'+fname_analytical+'.py')
imod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(imod)

data = imod._PETScBinaryLoad()
imod._PETScBinaryLoadReportNames(data)

# Get data
m0 = data['Nx'][0]
n0 = data['Ny'][0]
xc0 = data['x1d_cell']
yc0 = data['y1d_cell']
T0 = data['X_cell']

# 1 Numerical (upwind)
# imod = importlib.import_module(fname11)
spec = importlib.util.spec_from_file_location(fname11,fname_data+'/'+fname11+'.py')
imod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(imod)

data = imod._PETScBinaryLoad()
imod._PETScBinaryLoadReportNames(data)

# Get data
m11 = data['Nx'][0]
n11 = data['Ny'][0]
xc11 = data['x1d_cell']
yc11 = data['y1d_cell']
T11 = data['X_cell']

# 2 Numerical (fromm)
# imod = importlib.import_module(fname12)
spec = importlib.util.spec_from_file_location(fname12,fname_data+'/'+fname12+'.py')
imod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(imod)

data = imod._PETScBinaryLoad()
imod._PETScBinaryLoadReportNames(data)

# Get data
m12 = data['Nx'][0]
n12 = data['Ny'][0]
xc12 = data['x1d_cell']
yc12 = data['y1d_cell']
T12 = data['X_cell']

# 2 Numerical 312
# imod = importlib.import_module(fname2)
spec = importlib.util.spec_from_file_location(fname2,fname_data+'/'+fname2+'.py')
imod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(imod)

data = imod._PETScBinaryLoad()
imod._PETScBinaryLoadReportNames(data)

# Get data
m2 = data['Nx'][0]
n2 = data['Ny'][0]
xc2 = data['x1d_cell']
yc2 = data['y1d_cell']
T2 = data['X_cell']

# 3 Numerical 313
# imod = importlib.import_module(fname3)
spec = importlib.util.spec_from_file_location(fname3,fname_data+'/'+fname3+'.py')
imod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(imod)

data = imod._PETScBinaryLoad()
imod._PETScBinaryLoadReportNames(data)

# Get data
m3 = data['Nx'][0]
n3 = data['Ny'][0]
xc3 = data['x1d_cell']
yc3 = data['y1d_cell']
T3 = data['X_cell']

# 4 Numerical 314
# imod = importlib.import_module(fname4)
spec = importlib.util.spec_from_file_location(fname4,fname_data+'/'+fname4+'.py')
imod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(imod)

data = imod._PETScBinaryLoad()
imod._PETScBinaryLoadReportNames(data)

# Get data
m4 = data['Nx'][0]
n4 = data['Ny'][0]
xc4 = data['x1d_cell']
yc4 = data['y1d_cell']
T4 = data['X_cell']

# ---------------------------------------
# Plot data - ex 311
# ---------------------------------------

# Open a figure
fig, axs = plt.subplots(1, 3,figsize=(18,6))

# Subplot 0
ax0 = axs[0]
contours0 = ax0.contour(xc0,yc0, T0.reshape(n0,m0), 10 , colors='white',linestyles='solid',linewidths=0.5)
ax0.clabel(contours0, contours0.levels, inline=True, fontsize=8)
im0 = ax0.imshow(T0.reshape(n0,m0), extent=[min(xc0), max(xc0), min(yc0), max(yc0)],
                origin='lower', interpolation='nearest' )
ax0.axis('image')
ax0.set_xlabel('x-dir')
ax0.set_ylabel('z-dir')
ax0.set_title('Elman (2005) ex 3.1.1 solution (analytical)')

# Subplot 1
ax1 = axs[1]
contours11 = ax1.contour(xc11,yc11, T11.reshape(n11,m11), 10 , colors='white',linestyles='solid',linewidths=0.5)
ax1.clabel(contours11, contours11.levels, inline=True, fontsize=8)
im1 = ax1.imshow(T11.reshape(n11,m11), extent=[min(xc11), max(xc11), min(yc11), max(yc11)],
                origin='lower', interpolation='nearest' )
ax1.axis('image')
ax1.set_xlabel('x-dir')
ax1.set_title('Upwind (numerical)')

# Subplot 2
ax2 = axs[2]
contours12 = ax2.contour(xc12,yc12, T12.reshape(n12,m12), 10 , colors='white',linestyles='solid',linewidths=0.5)
ax2.clabel(contours12, contours12.levels, inline=True, fontsize=8)
im2 = ax2.imshow(T12.reshape(n12,m12), extent=[min(xc12), max(xc12), min(yc12), max(yc12)],
                origin='lower', interpolation='nearest' )
ax2.axis('image')
ax2.set_xlabel('x-dir')
ax2.set_title('Fromm (numerical)')

cbar = fig.colorbar(im2,ax=axs, shrink=0.75)
cbar.ax.set_ylabel('T')

plt.savefig(fname_out+'/'+fname+'.pdf')

# ---------------------------------------
# Plot data - ex 311, 312, 313, 314
# ---------------------------------------

# Open a figure
fig, axs = plt.subplots(2, 2,figsize=(12,12))

# Subplot 0
ax1 = plt.subplot(221)
contours1 = ax1.contour(xc12,yc12, T12.reshape(n12,m12), 10 , colors='white',linestyles='solid',linewidths=0.5)
ax1.clabel(contours1, contours1.levels, inline=True, fontsize=8)
im1 = ax1.imshow(T12.reshape(n12,m12), extent=[min(xc12), max(xc12), min(yc12), max(yc12)],
                origin='lower', interpolation='nearest' )
ax1.axis('image')
ax1.set_xlabel('x-dir')
ax1.set_ylabel('z-dir')
ax1.set_title('Elman (2005) Ex 3.1.1')
cbar = fig.colorbar(im1,ax=ax1, shrink=0.75)
# cbar.ax.set_ylabel('T')

# Subplot 1
ax2 = plt.subplot(222)
contours2 = ax2.contour(xc2,yc2, T2.reshape(n2,m2), 10 , colors='white',linestyles='solid',linewidths=0.5)
ax2.clabel(contours2, contours2.levels, inline=True, fontsize=8)
im2 = ax2.imshow(T2.reshape(n2,m2), extent=[min(xc2), max(xc2), min(yc2), max(yc2)],
                origin='lower', interpolation='nearest' )
ax2.axis('image')
ax2.set_xlabel('x-dir')
ax2.set_ylabel('z-dir')
ax2.set_title('Elman (2005) Ex 3.1.2')
cbar = fig.colorbar(im2,ax=ax2, shrink=0.75)

# Subplot 2
ax3 = plt.subplot(223)
contours3 = ax3.contour(xc3,yc3, T3.reshape(n3,m3), 10 , colors='white',linestyles='solid',linewidths=0.5)
ax3.clabel(contours3, contours3.levels, inline=True, fontsize=8)
im3 = ax3.imshow(T3.reshape(n3,m3), extent=[min(xc3), max(xc3), min(yc3), max(yc3)],
                origin='lower', interpolation='nearest' )
ax3.axis('image')
ax3.set_xlabel('x-dir')
ax3.set_ylabel('z-dir')
ax3.set_title('Elman (2005) Ex 3.1.3')
cbar = fig.colorbar(im3,ax=ax3, shrink=0.75)

# Subplot 3
ax4 = plt.subplot(224)
contours4 = ax4.contour(xc4,yc4, T4.reshape(n4,m4), 10 , colors='white',linestyles='solid',linewidths=0.5)
ax4.clabel(contours4, contours4.levels, inline=True, fontsize=8)
im4 = ax4.imshow(T4.reshape(n4,m4), extent=[min(xc4), max(xc4), min(yc4), max(yc4)],
                origin='lower', interpolation='nearest' )
ax4.axis('image')
ax4.set_xlabel('x-dir')
ax4.set_ylabel('z-dir')
ax4.set_title('Elman (2005) Ex 3.1.4')
cbar = fig.colorbar(im4,ax=ax4, shrink=0.75)

plt.savefig(fname_out+'/'+fname[:-3]+'.pdf')

# ---------------------------------------
# Plot data - ex 311, 312, 313, 314 - SURFACE PLOTS
# ---------------------------------------

# Open a figure
fig, axs = plt.subplots(2, 2,figsize=(12,12))
cmaps='viridis'

# Subplot 0
ax1 = plt.subplot(221, projection='3d')
X, Y = np.meshgrid(xc0, yc0)
surf1 = ax1.plot_surface(X, Y, T12.reshape(n12,m12), linewidth=0, cmap=cmaps, antialiased=False)
ax1.view_init(30, -75)
ax1.set_xlabel('x-dir')
ax1.set_ylabel('z-dir')
ax1.set_title('Elman (2005) Ex 3.1.1')
# cbar = fig.colorbar(im1,ax=ax1, shrink=0.75)

# Subplot 1
ax2 = plt.subplot(222, projection='3d')
surf2 = ax2.plot_surface(X, Y, T2.reshape(n2,m2), linewidth=0, cmap=cmaps,antialiased=False)
ax2.view_init(30, -75)
ax2.set_xlabel('x-dir')
ax2.set_ylabel('z-dir')
ax2.set_title('Elman (2005) Ex 3.1.2')
# cbar = fig.colorbar(im2,ax=ax2, shrink=0.75)

# Subplot 2
ax3 = plt.subplot(223, projection='3d')
surf3 = ax3.plot_surface(X, Y, T3.reshape(n3,m3), linewidth=0, cmap=cmaps,antialiased=False)
ax3.view_init(30, -75)
ax3.set_xlabel('x-dir')
ax3.set_ylabel('z-dir')
ax3.set_title('Elman (2005) Ex 3.1.3')
# cbar = fig.colorbar(im3,ax=ax3, shrink=0.75)

# Subplot 3
ax4 = plt.subplot(224, projection='3d')
surf4 = ax4.plot_surface(X, Y, T4.reshape(n4,m4), linewidth=0, cmap=cmaps,antialiased=False)
ax4.view_init(30, -135)
ax4.set_xlabel('x-dir')
ax4.set_ylabel('z-dir')
ax4.set_title('Elman (2005) Ex 3.1.4')
# cbar = fig.colorbar(im4,ax=ax4, shrink=0.75)

plt.savefig(fname_out+'/'+fname[:-3]+'_3D.pdf')

os.system('rm -r '+fname_data+'/__pycache__')