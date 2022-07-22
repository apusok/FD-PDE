# ---------------------------------------
# Convection with periodic BC (after the mantle convection benchmark Blankenbach et al. 1989)
# Steady-state models:
#    1A - eta0=1e23, b=0, c=0, Ra = 1e4
# ---------------------------------------

# Import modules
import numpy as np
import matplotlib.pyplot as plt
import importlib
import os
import sys, getopt
import warnings
warnings.filterwarnings('ignore')

print('# --------------------------------------- #')
print('# Convection with periodic BC')
print('# --------------------------------------- #')

# Parameters
n  = 50
Ra = 1e6
tmax = 1e-4
dtmax = 1e-7
tout = 20
tstep_max = 1000 #timesteps

# Get cpu number
ncpu = 1
options, remainder = getopt.getopt(sys.argv[1:],'n:')
for opt, arg in options:
  if opt in ('-n'):
    ncpu = int(arg)

# Use umfpack for sequential and mumps for sequential/parallel
solver_default = ' -snes_monitor -snes_converged_reason -ksp_monitor -ksp_converged_reason '
if (ncpu == -1):
  solver = ' -pc_type lu -pc_factor_mat_solver_type umfpack -pc_factor_mat_ordering_type external'
  ncpu = 1
else:
  solver = ' -pc_type lu -pc_factor_mat_solver_type mumps'

# Input file
fname = 'out_convection_periodic_n'+str(n)
fname_out = 'out_convection_periodic_n'+str(n)
fname_data = fname_out+'/data'
try:
  os.mkdir(fname_out)
except OSError:
  pass

try:
  os.mkdir(fname_data)
except OSError:
  pass

# Run test
str1 = 'mpiexec -n '+str(ncpu)+' ../test_convection_stokes_periodic.app '+solver+solver_default+ \
      ' -Ra '+str(Ra)+ \
      ' -tmax '+str(tmax)+ \
      ' -dtmax '+str(dtmax)+ \
      ' -tstep '+str(tstep_max)+ \
      ' -output_file '+fname+ \
      ' -output_dir '+fname_data+\
      ' -tout '+str(tout)+ \
      ' -nx '+str(n)+' -nz '+str(n)+' > '+fname_data+'/'+fname+'.out'
print(str1)
os.system(str1)

# Parse log file
fout1 = fname_data+'/'+fname+'.out'

f = open(fout1, 'r')
i0=0
for line in f:
  if '# TIMESTEP' in line:
      i0+=1
f.close()
tstep = i0

# Plot timesteps
for istep in range(0,tstep,tout):
  # Load python module describing data
  if (istep < 10): ft = '_ts00'+str(istep)
  if (istep >= 10) & (istep < 99): ft = '_ts0'+str(istep)
  if (istep >= 100): ft = '_ts'+str(istep)

  # Plot solution
  # Load python module describing data
  fout = fname+'_PV'+ft
  # imod = importlib.import_module(fout)
  spec = importlib.util.spec_from_file_location(fout,fname_data+'/'+fout+'.py')
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
  p = xcenter[0::celldof]

  # Load temperature
  fout = fname+'_T'+ft
  # imod = importlib.import_module(fout)
  spec = importlib.util.spec_from_file_location(fout,fname_data+'/'+fout+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod)

  # Load data
  data = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(data)
  xcenter = data['X_cell']

  T = xcenter[0::celldof]

  # Open a figure
  fig, axs = plt.subplots(1, 2,figsize=(12,6))

  nind = 2

  ax1 = axs[0]
  contours = ax1.contour(xc,yc,p.reshape(n,m), colors='white',linestyles='solid',linewidths=0.5)
  ax1.clabel(contours, contours.levels, inline=True, fontsize=8)
  im1 = ax1.imshow( p.reshape(n,m), extent=[min(xv), max(xv), min(yv), max(yv)],
                  origin='lower', interpolation='nearest' )
  Q = ax1.quiver( xc[::nind], yc[::nind], vxc[::nind,::nind], vyc[::nind,::nind], units='width', pivot='mid')
  # ax1.quiverkey(Q, X=0.25, Y=-0.2, U=10, label='Quiver length = 10 [cm/yr]', labelpos='E')
  cbar = fig.colorbar(im1,ax=ax1, shrink=0.75, label='P')
  ax1.axis('image')
  ax1.set_xlabel('x-dir')
  ax1.set_ylabel('z-dir')
  ax1.set_title('Pressure and velocity (Stokes)')

  ax2 = axs[1]
  contours = ax2.contour(xc,yc,T.reshape(n,m), colors='white',linestyles='solid',linewidths=0.5)
  ax2.clabel(contours, contours.levels, inline=True, fontsize=8)
  im2 = ax2.imshow( T.reshape(n,m), extent=[min(xv), max(xv), min(yv), max(yv)],
                  origin='lower', interpolation='nearest' )
  ax2.axis('image')
  cbar = fig.colorbar(im2,ax=ax2, shrink=0.75, label='T')
  ax2.set_xlabel('x-dir')
  # ax2.set_ylabel('z-dir')
  ax2.set_title('Temperature contours (AdvDiff)')

  plt.savefig(fname_out+'/'+fout+'.pdf')
  plt.close()

os.system('rm -r '+fname_data+'/__pycache__')
