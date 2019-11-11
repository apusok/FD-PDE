# ---------------------------------------
# Mantle convection benchmark (Blankenbach et al. 1989)
# Steady-state models:
#    1A - eta0=1e23, b=0, c=0, Ra = 1e4
#    1B - eta0=1e22, b=0, c=0, Ra = 1e5
#    1C - eta0=1e21, b=0, c=0, Ra = 1e6
#    2A - eta0=1e23, b=ln(1000), c=0
#    2B - eta0=1e23, b=ln(16384), c=ln(64), L=2500
# Time-dependent models:
#    3A - eta0=1e23, b=0, c=0, L=1500, Ra = 216000
# ---------------------------------------

# Import modules
import numpy as np
import matplotlib.pyplot as plt
import importlib
import os

# Input file
fname = 'out_convection'

print('# --------------------------------------- #')
print('# Mantle convection benchmark (Blankenbach et al. 1989)')
print('# --------------------------------------- #')

n = 100

# Run test
str1 = '../test_composite_convection.app -pc_type lu -pc_factor_mat_solver_type umfpack -output_file '+fname+' -nx '+str(n)+' -nz '+str(n)
print(str1)
os.system(str1)

# Plot solution
# Load python module describing data
imod = importlib.import_module(fname)

# Load data
data = imod._PETScBinaryLoad()
imod._PETScBinaryLoadReportNames(data)

# Get data
m = data['Nx'][0]
n = data['Ny'][0]
xv = data['x1d_vertex']/1e3
yv = data['y1d_vertex']/1e3
xc = data['x1d_cell']/1e3
yc = data['y1d_cell']/1e3
vxface = data['X_face_x']
vyface = data['X_face_y']
xcenter = data['X_cell']

# Compute the DOF count on each stratum (face,element)
facedof_x = int(len(vxface)/((m+1)*n))
facedof_y = int(len(vyface)/(m*(n+1)))
celldof   = int(len(xcenter)/(m*n))

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

# Prepare center values
p0 = xcenter[0::celldof]
pavg = np.average(p0)
print(pavg)
p = p0/pavg

T = xcenter[1::celldof]
eta = xcenter[2::celldof]
rho = xcenter[3::celldof]

# Open a figure
fig, axs = plt.subplots(1, 2,figsize=(12,6))

nind = 5

ax1 = axs[0]
contours = ax1.contour(xc,yc,p.reshape(n,m), colors='white',linestyles='solid',linewidths=0.5)
ax1.clabel(contours, contours.levels, inline=True, fontsize=8)
im1 = ax1.imshow( p.reshape(n,m), extent=[min(xv), max(xv), min(yv), max(yv)],
                origin='lower', interpolation='nearest' )
Q = ax1.quiver( xc[::nind], yc[::nind], vxc[::nind,::nind], vyc[::nind,::nind], units='width', pivot='mid')
ax1.quiverkey(Q, X=0.25, Y=-0.2, U=10, label='Quiver length = 10 [cm/yr]', labelpos='E')
cbar = fig.colorbar(im1,ax=ax1, shrink=0.75, label='P (GPa)')
ax1.axis(aspect='image')
ax1.set_xlabel('x-dir [km]')
ax1.set_ylabel('z-dir [km]')
ax1.set_title('Pressure and velocity (Stokes)')

ax2 = axs[1]
contours = ax2.contour(xc,yc,T.reshape(n,m), colors='white',linestyles='solid',linewidths=0.5)
ax2.clabel(contours, contours.levels, inline=True, fontsize=8)
im2 = ax2.imshow( rho.reshape(n,m), extent=[min(xv), max(xv), min(yv), max(yv)],
                origin='lower', interpolation='nearest' )
ax2.axis(aspect='image')
cbar = fig.colorbar(im2,ax=ax2, shrink=0.75, label='T (K)')
ax2.set_xlabel('x-dir [km]')
# ax2.set_ylabel('z-dir [km]')
ax2.set_title('Temperature contours and density (AdvDiff)')

plt.savefig(fname+'.pdf')