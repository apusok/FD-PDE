# ---------------------------------------
# Corner flow (mid-ocean ridges) benchmark
# Plot analytical and numerical solution
# ---------------------------------------

# Import modules
import numpy as np
import matplotlib.pyplot as plt
import importlib
import os
from matplotlib import rc

# Some new font
rc('font',**{'family':'serif','serif':['Times new roman']})
rc('text', usetex=True)

# Input file
fname = 'out_corner_flow_mor'

print('# --------------------------------------- #')
print('# Corner flow (mid-ocean ridges) benchmark ')
print('# --------------------------------------- #')

nz = 50
nx = 2*nz
L  = 2
H  = 1

# Run test
str1 = '../../../src/tests/test_stokes_mor.app -pc_type lu -pc_factor_mat_solver_type umfpack -output_file '+fname+ \
  ' -nx '+str(nx)+' -nz '+str(nz)+' -L '+str(L)+' -H '+str(H)+' > '+fname+'.out'
print(str1)
# os.system(str1)

# Plot solution

# Load python module describing data
imod = importlib.import_module(fname)

# Load data
data = imod._PETScBinaryLoad()
imod._PETScBinaryLoadReportNames(data)

# Get data
m = data['Nx'][0]
n = data['Ny'][0]
xv = data['x1d_vertex']
yv = data['y1d_vertex']
xc = data['x1d_cell']
yc = data['y1d_cell']
vxface = data['X_face_x']
vyface = data['X_face_y']
p = data['X_cell']

# Compute the DOF count on each stratum (face,element)
facedof_x = int(len(vxface)/((m+1)*n))
facedof_y = int(len(vyface)/(m*(n+1)))
celldof   = int(len(p)/(m*n))

# Prepare cell center velocities
vxface = vxface.reshape(n  ,m+1)
vyface = vyface.reshape(n+1,m  )

# Compute the cell center values from the face data by averaging neighbouring faces
vxc = np.zeros( [n , m] )
vyc = np.zeros( [n , m] )
for i in range(0,m):
  for j in range(0,n):
    vxc[j][i] = 0.5 * (vxface[j][i+1] + vxface[j][i])
    vyc[j][i] = 0.5 * (vyface[j+1][i] + vyface[j][i])

nind = 5

# Open a figure
# fig = plt.figure(1,figsize=(12,6))
fig, axs = plt.subplots(1, 2,figsize=(12,6))

ax1 = axs[0]
contours = ax1.contour( xc , yc , p.reshape(n,m), levels=[-20,-10, -7.5, -5, -2.5,-1] , colors='white',linestyles='solid',linewidths=0.5)
ax1.clabel(contours, contours.levels, inline=True, fontsize=8)
im = ax1.imshow( p.reshape(n,m), extent=[min(xv), max(xv), min(yv), max(yv)],
                origin='lower', cmap='inferno', interpolation='nearest' )
im.set_clim(-10,0)
Q = ax1.quiver( xc[::nind], yc[::nind], vxc[::nind,::nind], vyc[::nind,::nind], units='width', pivot='mid' )
ax1.axis(aspect='image')
ax1.set_xlabel('x-dir')
ax1.set_ylabel('z-dir')
ax1.set_title('MOR Corner Flow (numerical)')
# cbar = fig.colorbar( im, ax=ax1 )
# cbar.ax.set_ylabel('Pressure')

# Plot analytical solution
fname1 = 'out_analytic_solution_mor'

# Load python module describing data
imod = importlib.import_module(fname1)

# Load data
data = imod._PETScBinaryLoad()
imod._PETScBinaryLoadReportNames(data)

# Get data
m = data['Nx'][0]
n = data['Ny'][0]
xv = data['x1d_vertex']
yv = data['y1d_vertex']
xc = data['x1d_cell']
yc = data['y1d_cell']
vxface = data['X_face_x']
vyface = data['X_face_y']
p = data['X_cell']

# Compute the DOF count on each stratum (face,element)
facedof_x = int(len(vxface)/((m+1)*n))
facedof_y = int(len(vyface)/(m*(n+1)))
celldof   = int(len(p)/(m*n))

# Prepare cell center velocities
vxface = vxface.reshape(n  ,m+1)
vyface = vyface.reshape(n+1,m  )

# Compute the cell center values from the face data by averaging neighbouring faces
vxc = np.zeros( [n , m] )
vyc = np.zeros( [n , m] )
for i in range(0,m):
  for j in range(0,n):
    vxc[j][i] = 0.5 * (vxface[j][i+1] + vxface[j][i])
    vyc[j][i] = 0.5 * (vyface[j+1][i] + vyface[j][i])

nind = 5

# Open a figure
ax2 = axs[1]
contours = ax2.contour( xc , yc , p.reshape(n,m), levels=[-20,-10, -7.5, -5, -2.5,-1] , colors='white',linestyles='solid',linewidths=0.5)
ax2.clabel(contours, contours.levels, inline=True, fontsize=8)
im = plt.imshow( p.reshape(n,m), extent=[min(xv), max(xv), min(yv), max(yv)],
                origin='lower', cmap='inferno', interpolation='nearest' )
im.set_clim(-10,0)
Q = ax2.quiver( xc[::nind], yc[::nind], vxc[::nind,::nind], vyc[::nind,::nind], units='width', pivot='mid' )
ax2.axis(aspect='image')
ax2.set_xlabel('x-dir')
#plt.ylabel('z-dir')
ax2.set_title('MOR Corner Flow (analytical)')

cbar = fig.colorbar(im,ax=axs, shrink=0.75)
cbar.ax.set_ylabel('Pressure')

plt.savefig(fname+'.pdf')
plt.close()

# Plot one figure
fig, ax1 = plt.subplots(1,figsize=(9,6))
nind = 4
iind = 2

phi = np.ones(n*m)

contours = ax1.contour(xc,yc,p.reshape(n,m), levels=[-20,-10, -7.5, -5, -2.5,-1], colors='k',linewidths=0.5)
# ax1.clabel(contours, contours.levels, inline=True, fontsize=8)
im = ax1.imshow( phi.reshape(n,m), extent=[min(xc), max(xc), min(yc), max(yc)],vmin=1.0, vmax=1.05,
                origin='lower', cmap='ocean_r', interpolation='nearest')
cbar = fig.colorbar(im,ax=ax1, shrink=0.60,label=r'$\phi/\phi_0$' )
Q  = ax1.quiver( xc[::nind], yc[::nind], vxc[::nind,::nind], vyc[::nind,::nind], color='grey', units='width', pivot='mid')
ax1.axis(aspect='image')
ax1.set_xlabel('x/h')
ax1.set_ylabel('z/h')
ax1.set_title('MOR - corner flow solution')

plt.savefig(fname+'_v2.pdf')
plt.close()

os.system('rm -r __pycache__')