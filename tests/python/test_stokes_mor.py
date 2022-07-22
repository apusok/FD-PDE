# ---------------------------------------
# Corner flow (mid-ocean ridges) benchmark
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
fname = 'out_stokes_mor'
fname_data = fname+'/data'
try:
  os.mkdir(fname)
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
print('# Corner flow (mid-ocean ridges) benchmark ')
print('# --------------------------------------- #')

n = 100

# Use umfpack for sequential and mumps for sequential/parallel
solver_default = ' -snes_monitor -snes_converged_reason -ksp_monitor -ksp_converged_reason '
if (ncpu == -1):
  solver = ' -pc_type lu -pc_factor_mat_solver_type umfpack -pc_factor_mat_ordering_type external'
  ncpu = 1
else:
  solver = ' -pc_type lu -pc_factor_mat_solver_type mumps'

# Run test
fout1 = fname_data+'/log_'+fname+'_'+str(n)+'.out'
str1 = 'mpiexec -n '+str(ncpu)+' ../test_stokes_mor.app '+solver+solver_default+ \
  ' -output_file '+fname+' -output_dir '+fname_data+' -nx '+str(n)+' -nz '+str(n)+' > '+fout1
print(str1)
os.system(str1)

# Plot solution
# Load python module describing data
# imod = importlib.import_module(fname)
spec = importlib.util.spec_from_file_location(fname,fname_data+'/'+fname+'.py')
imod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(imod)

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
ax1.axis('image')
ax1.set_xlabel('x-dir')
ax1.set_ylabel('z-dir')
ax1.set_title('MOR Corner Flow (numerical)')
# cbar = fig.colorbar( im, ax=ax1 )
# cbar.ax.set_ylabel('Pressure')

# Plot analytical solution
fname1 = 'out_analytic_solution_mor'

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
ax2.axis('image')
ax2.set_xlabel('x-dir')
#plt.ylabel('z-dir')
ax2.set_title('MOR Corner Flow (analytical)')

cbar = fig.colorbar(im,ax=axs, shrink=0.75)
cbar.ax.set_ylabel('Pressure')

plt.savefig(fname+'/'+fname+'.pdf')
os.system('rm -r '+fname_data+'/__pycache__')