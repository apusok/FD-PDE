# ---------------------------------------
# Laplace (ADVDIFF) benchmark \nabla^2 T = 0
# Plot analytical and numerical solution
# ---------------------------------------

# Import modules
import numpy as np
import matplotlib.pyplot as plt
import importlib
import os

# Input file
fname = 'out_advdiff_laplace'

print('# --------------------------------------- #')
print('# Laplace (ADVDIFF) benchmark ')
print('# --------------------------------------- #')

n = 100

# Run test
str1 = '../test_advdiff_laplace.app -pc_type lu -pc_factor_mat_solver_type umfpack -output_file '+fname+' -nx '+str(n)+' -nz '+str(n)
print(str1)
os.system(str1)

# Plot solution (numerical and analytical)

# Load python module describing data
imod = importlib.import_module(fname)

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
ax1.axis(aspect='image')
ax1.set_xlabel('x-dir')
ax1.set_ylabel('z-dir')
ax1.set_title('Laplace solution (numerical)')

# Plot analytical solution
fname1 = 'out_analytic_solution_laplace'

# Load python module describing data
imod = importlib.import_module(fname1)

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
ax2.axis(aspect='image')
ax2.set_xlabel('x-dir')
ax2.set_title('Laplace solution (analytical)')

cbar = fig.colorbar(im,ax=axs, shrink=0.75)
cbar.ax.set_ylabel('T')

plt.savefig(fname+'.pdf')