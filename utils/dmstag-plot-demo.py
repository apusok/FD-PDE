
#
# To use this script, make sure you set python path enviroment variable to include the following paths
#   ${RIFTOMAT_ROOT}/utils:${PETSC_DIR}/lib/petsc/bin
# e.g.
#   export PYTHONPATH=${RIFTOMAT_ROOT}/utils:${PETSC_DIR}/lib/petsc/bin
#
import numpy as np
import matplotlib.pyplot as plt

# Import the auto-generated python script
import numerical_solution as stokes

# Load the data
data = stokes._PETScBinaryLoad()
stokes._PETScBinaryLoadReportNames(data)

# Get number of cells in i,j
m = data['Nx'][0]
n = data['Ny'][0]
print('# cells (i,j):', [m,n])

# Get 1D vertex coordinate arrays
xv = data['x1d_vertex']
yv = data['y1d_vertex']

# Get 1D cell center coordinate arrays
xc = data['x1d_cell']
yc = data['y1d_cell']

# Get all values defined on the cell face
facedata_x = data['X_face_x']
facedata_y = data['X_face_y']

# Get all values defined on the cell center
celldata = data['X_cell']

# Compute the DOF count on each stratum (face,element)
facedof_x = int(len(facedata_x)/((m+1)*n))
facedof_y = int(len(facedata_y)/(m*(n+1)))
celldof   = int(len(celldata)/(m*n))
print('# face DOFs-x:', facedof_x)
print('# face DOFs-y:', facedof_y)
print('# cells DOFs :', celldof)

# Get DOF-0 values defined on the cell face
vxface = facedata_x[0 : len(facedata_x) : facedof_x]
vyface = facedata_y[0 : len(facedata_y) : facedof_y]

# Get DOF-0 values defined on the cell center
p      = celldata  [0 : len(celldata)   : celldof]
# Get DOF-1 values defined on the cell center
#T      = celldata  [1 : len(celldata)   : celldof]

# Prepare cell center velocities
# For convienence, we re-shape the data, indexed j first, then i
# This ordering is chosen for consistency with what matplotlib expects when plotting
vxface = vxface.reshape( n     , m + 1 )
vyface = vyface.reshape( n + 1 , m     )

# Compute the cell center values from the face data by averaging neighbouring faces
vxc = np.zeros( [n , m] )
vyc = np.zeros( [n , m] )
for i in range(0,m):
  for j in range(0,n):
    vxc[j][i] = 0.5 * (vxface[j][i+1] + vxface[j][i])
    vyc[j][i] = 0.5 * (vyface[j+1][i] + vyface[j][i])

#
# Open a figure
#
fig, ax1 = plt.subplots(1)

#
# Plot 5 pressure contours.
# Note that we reshape & transpose the pressure array.
#
contours = plt.contour( xc , yc , p.reshape(n,m), 5, colors='white' )
plt.clabel( contours, inline=True, fontsize=8 )

#
# Plot uninterpolated pressure values (piece-wise constants),
# imshow() is not appropriate for non-constant cell spacing.
# interpolation='none' does not appear to always be supported, so I use 'nearest' as the next best thing.
#
im = plt.imshow( p.reshape(n,m), extent=[min(xv), max(xv), min(yv), max(yv)],
                origin='lower', cmap='RdYlBu', interpolation='nearest' )

#
# Overlay cell center velocity vector field
#
Q = plt.quiver( xc, yc, vxc, vyc, units='width', pivot='mid' )

plt.axis('image')

#
# Add a colour bar
#
cbar = fig.colorbar( im, ax=ax1, orientation='horizontal', fraction=.1 )
cbar.ax.set_xlabel('pressure')

plt.savefig("stokesdemo-vp.pdf")
