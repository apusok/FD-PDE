# ---------------------------------------
# ADVDIFF benchmarks from Elman (2005): 
# Example 3.1.1 
# Example 3.1.3 
# Example 3.1.4
# Plot analytical and numerical solution
# ---------------------------------------

# Import modules
import numpy as np
import matplotlib.pyplot as plt
import importlib
import os

print('# --------------------------------------- #')
print('# Elman 311 (ADVDIFF) benchmark ')
print('# --------------------------------------- #')

n = 100

# Input file
fname = 'out_num_solution_elman311'
fname0 = fname+'_upwind'
fname1 = fname+'_fromm'
fname_analytical = 'out_analytic_solution_elman311'
fname2 = 'out_num_solution_elman313'
fname3 = 'out_num_solution_elman314'

# Run test
str1 = '../test_advdiff_elman.app -pc_type lu -pc_factor_mat_solver_type umfpack -output_file '+fname0+' -nx '+str(n)+' -nz '+str(n)+' -advtype 0'
print(str1)
os.system(str1)

str2 = '../test_advdiff_elman.app -pc_type lu -pc_factor_mat_solver_type umfpack -output_file '+fname1+' -nx '+str(n)+' -nz '+str(n)+' -advtype 1'
print(str2)
os.system(str2)

# ---------------------------------------
# Load python module describing data
# ---------------------------------------

# 0 Analytical 
imod = importlib.import_module(fname_analytical)
data = imod._PETScBinaryLoad()
imod._PETScBinaryLoadReportNames(data)

# Get data
m0 = data['Nx'][0]
n0 = data['Ny'][0]
xc0 = data['x1d_cell']
yc0 = data['y1d_cell']
T0 = data['X_cell']

# 1 Numerical (upwind)
imod = importlib.import_module(fname0)
data = imod._PETScBinaryLoad()
imod._PETScBinaryLoadReportNames(data)

# Get data
m1 = data['Nx'][0]
n1 = data['Ny'][0]
xc1 = data['x1d_cell']
yc1 = data['y1d_cell']
T1 = data['X_cell']

# 2 Numerical (fromm)
imod = importlib.import_module(fname1)
data = imod._PETScBinaryLoad()
imod._PETScBinaryLoadReportNames(data)

# Get data
m2 = data['Nx'][0]
n2 = data['Ny'][0]
xc2 = data['x1d_cell']
yc2 = data['y1d_cell']
T2 = data['X_cell']

# 3 Numerical 313
imod = importlib.import_module(fname2)
data = imod._PETScBinaryLoad()
imod._PETScBinaryLoadReportNames(data)

# Get data
m3 = data['Nx'][0]
n3 = data['Ny'][0]
xc3 = data['x1d_cell']
yc3 = data['y1d_cell']
T3 = data['X_cell']

# 4 Numerical 314
imod = importlib.import_module(fname3)
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
ax0.axis(aspect='image')
ax0.set_xlabel('x-dir')
ax0.set_ylabel('z-dir')
ax0.set_title('Elman (2005) ex 3.1.1 solution (analytical)')

# Subplot 1
ax1 = axs[1]
contours1 = ax1.contour(xc1,yc1, T1.reshape(n1,m1), 10 , colors='white',linestyles='solid',linewidths=0.5)
ax1.clabel(contours1, contours1.levels, inline=True, fontsize=8)
im1 = ax1.imshow(T1.reshape(n1,m1), extent=[min(xc1), max(xc1), min(yc1), max(yc1)],
                origin='lower', interpolation='nearest' )
ax1.axis(aspect='image')
ax1.set_xlabel('x-dir')
ax1.set_title('Upwind (numerical)')

# Subplot 2
ax2 = axs[2]
contours2 = ax2.contour(xc2,yc2, T2.reshape(n2,m2), 10 , colors='white',linestyles='solid',linewidths=0.5)
ax2.clabel(contours2, contours2.levels, inline=True, fontsize=8)
im2 = ax2.imshow(T2.reshape(n2,m2), extent=[min(xc2), max(xc2), min(yc2), max(yc2)],
                origin='lower', interpolation='nearest' )
ax2.axis(aspect='image')
ax2.set_xlabel('x-dir')
ax2.set_title('Fromm (numerical)')

cbar = fig.colorbar(im2,ax=axs, shrink=0.75)
cbar.ax.set_ylabel('T')

plt.savefig(fname+'.pdf')

# ---------------------------------------
# Plot data - ex 311, 313, 314
# ---------------------------------------

# Open a figure
fig, axs = plt.subplots(1, 3,figsize=(18,6))

# Subplot 0
ax1 = axs[0]
contours1 = ax1.contour(xc1,yc1, T1.reshape(n1,m1), 10 , colors='white',linestyles='solid',linewidths=0.5)
ax1.clabel(contours1, contours1.levels, inline=True, fontsize=8)
im1 = ax1.imshow(T1.reshape(n1,m1), extent=[min(xc1), max(xc1), min(yc1), max(yc1)],
                origin='lower', interpolation='nearest' )
ax1.axis(aspect='image')
ax1.set_xlabel('x-dir')
ax1.set_ylabel('z-dir')
ax1.set_title('Elman (2005) Ex 3.1.1')
cbar = fig.colorbar(im1,ax=ax1, shrink=0.75)
# cbar.ax.set_ylabel('T')

# Subplot 1
ax3 = axs[1]
contours3 = ax3.contour(xc3,yc3, T3.reshape(n3,m3), 10 , colors='white',linestyles='solid',linewidths=0.5)
ax3.clabel(contours3, contours3.levels, inline=True, fontsize=8)
im3 = ax3.imshow(T3.reshape(n3,m3), extent=[min(xc3), max(xc3), min(yc3), max(yc3)],
                origin='lower', interpolation='nearest' )
ax3.axis(aspect='image')
ax3.set_xlabel('x-dir')
ax3.set_ylabel('z-dir')
ax3.set_title('Elman (2005) Ex 3.1.3')
cbar = fig.colorbar(im3,ax=ax3, shrink=0.75)

# Subplot 2
ax4 = axs[2]
contours4 = ax4.contour(xc4,yc4, T4.reshape(n4,m4), 10 , colors='white',linestyles='solid',linewidths=0.5)
ax4.clabel(contours4, contours4.levels, inline=True, fontsize=8)
im4 = ax4.imshow(T4.reshape(n4,m4), extent=[min(xc4), max(xc4), min(yc4), max(yc4)],
                origin='lower', interpolation='nearest' )
ax4.axis(aspect='image')
ax4.set_xlabel('x-dir')
ax4.set_ylabel('z-dir')
ax4.set_title('Elman (2005) Ex 3.1.4')
cbar = fig.colorbar(im4,ax=ax4, shrink=0.75)

plt.savefig(fname[:-3]+'.pdf')