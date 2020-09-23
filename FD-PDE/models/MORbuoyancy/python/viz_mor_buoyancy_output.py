print('# --------------------------------------- #')
print('# Mid-ocean ridge model - Buoyancy ')
print('# --------------------------------------- #')

# Import modules
import numpy as np
import matplotlib.pyplot as plt
import importlib
import os
from matplotlib import rc

# Some new font
rc('font',**{'family':'serif','serif':['Times new roman']})
rc('text', usetex=True)

# ---------------------------------------
# Function definitions
# ---------------------------------------
def plot_initial_solution(fname,dim):

  # Load data
  if (dim == 1):
    fout = fname+'_PV_dim_initial'
  else:
    fout = fname+'_PV_initial'
  imod = importlib.import_module(fout) # P,v
  data_PV = imod._PETScBinaryLoad()
  # imod._PETScBinaryLoadReportNames(data_PV)

  if (dim == 1):
    fout = fname+'_T_dim_initial'
  else:
    fout = fname+'_T_initial'
  imod = importlib.import_module(fout) # T
  data_T = imod._PETScBinaryLoad()
  # imod._PETScBinaryLoadReportNames(data_T)

  if (dim == 1):
    fout = fname+'_Theta_dim_initial'
  else:
    fout = fname+'_Theta_initial'
  imod = importlib.import_module(fout) # Theta
  data_Th = imod._PETScBinaryLoad()

  if (dim == 1):
    fout = fname+'_C_dim_initial'
  else:
    fout = fname+'_C_initial'
  imod = importlib.import_module(fout)
  data_C = imod._PETScBinaryLoad()

  if (dim == 1):
    fout = fname+'_Cf_dim_initial'
  else:
    fout = fname+'_Cf_initial'
  imod = importlib.import_module(fout)
  data_Cf = imod._PETScBinaryLoad()

  if (dim == 1):
    fout = fname+'_Cs_dim_initial'
  else:
    fout = fname+'_Cs_initial'
  imod = importlib.import_module(fout)
  data_Cs = imod._PETScBinaryLoad()

  fout = fname+'_phi_initial'
  imod = importlib.import_module(fout)
  data_phi = imod._PETScBinaryLoad()

  # Split data
  mx = data_PV['Nx'][0]
  mz = data_PV['Ny'][0]
  xc = data_PV['x1d_cell']
  zc = data_PV['y1d_cell']
  xv = data_PV['x1d_vertex']
  zv = data_PV['y1d_vertex']
  vx = data_PV['X_face_x']
  vz = data_PV['X_face_y']
  p = data_PV['X_cell']
  T = data_T['X_cell']
  Th= data_Th['X_cell']
  C = data_C['X_cell']
  Cf= data_Cf['X_cell']
  Cs= data_Cs['X_cell']
  phi= data_phi['X_cell']

  # Compute cell center velocities
  vxr  = vx.reshape(mz  ,mx+1)
  vzr  = vz.reshape(mz+1,mx  )
  
  vxc  = np.zeros([mz,mx])
  vzc  = np.zeros([mz,mx])
  vfxc = np.zeros([mz,mx])
  vfzc = np.zeros([mz,mx])

  for i in range(0,mx):
    for j in range(0,mz):
      vxc[j][i]  = 0.5 * (vxr[j][i+1] + vxr[j][i])
      vzc[j][i]  = 0.5 * (vzr[j+1][i] + vzr[j][i])

  # Plot one figure
  fig = plt.figure(1,figsize=(14,16))
  nind = 4

  labelP = r'$P [-]$'
  labelT = r'$T [-]$'
  labelC = r'$\Theta$'
  labelCf = r'$\Theta_f$'
  labelCs = r'$\Theta_s$'
  labelphi= r'$\phi$'
  labelH  = r'$H$'
  labelx = 'x/h'
  labelz = 'z/h'
  scalx  = 1
  scalP  = 1
  scalT  = 0
  scalv  = 1 

  SEC_YEAR = 31536000

  # Transform to geounits
  if (dim == 1):
    scalx  = 1e2 # km
    scalP  = 1e-9 # GPa
    scalT  = 273.15 # deg C
    scalv  = 1.0e2*SEC_YEAR # cm/yr
    labelP = r'$P [GPa]$'
    labelT = r'$T [^oC]$'
    labelx = 'x [km]'
    labelz = 'z [km]'
    labelC = r'$C$'
    labelCf = r'$C_f$'
    labelCs = r'$C_s$'

  ax = plt.subplot(4,2,1)
  im = ax.imshow( p.reshape(mz,mx)*scalP, extent=[min(xc)*scalx, max(xc)*scalx, min(zc)*scalx, max(zc)*scalx],
                  origin='lower', cmap='ocean', interpolation='nearest')
  im.set_clim(-10,0)
  cbar = fig.colorbar(im,ax=ax, shrink=0.60,label=labelP )
  Q  = ax.quiver( xc[::nind]*scalx, zc[::nind]*scalx, vxc[::nind,::nind]*scalv, vzc[::nind,::nind]*scalv, color='grey', units='width', pivot='mid')
  ax.axis(aspect='image')
  ax.set_xlabel(labelx)
  ax.set_ylabel(labelz)
  ax.set_title('a) Initial PV ')

  ax = plt.subplot(4,2,3)
  im = ax.imshow(T.reshape(mz,mx)-scalT,extent=[min(xc)*scalx, max(xc)*scalx, min(zc)*scalx, max(zc)*scalx],cmap='seismic',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.60,label=labelT)
  ax.axis(aspect='image')
  ax.set_xlabel(labelx)
  ax.set_ylabel(labelz)
  ax.set_title('b) Initial T ')

  ax = plt.subplot(4,2,5)
  im = ax.imshow(Th.reshape(mz,mx)-scalT,extent=[min(xc)*scalx, max(xc)*scalx, min(zc)*scalx, max(zc)*scalx],cmap='seismic',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.60,label=labelT)
  ax.axis(aspect='image')
  ax.set_xlabel(labelx)
  ax.set_ylabel(labelz)
  ax.set_title('c) Initial T potential ')

  ax = plt.subplot(4,2,7)
  im = ax.imshow(phi.reshape(mz,mx),extent=[min(xc)*scalx, max(xc)*scalx, min(zc)*scalx, max(zc)*scalx],cmap='seismic',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.60,label=labelH)
  ax.axis(aspect='image')
  ax.set_xlabel(labelx)
  ax.set_ylabel(labelz)
  ax.set_title('d) Initial H ')

  ax = plt.subplot(4,2,2)
  im = ax.imshow(C.reshape(mz,mx),extent=[min(xc)*scalx, max(xc)*scalx, min(zc)*scalx, max(zc)*scalx],cmap='magma',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.60,label=labelC)
  ax.axis(aspect='image')
  ax.set_xlabel(labelx)
  ax.set_ylabel(labelz)
  ax.set_title('e) Initial C ')

  ax = plt.subplot(4,2,4)
  im = ax.imshow(Cf.reshape(mz,mx),extent=[min(xc)*scalx, max(xc)*scalx, min(zc)*scalx, max(zc)*scalx],cmap='magma',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.60,label=labelCf)
  ax.axis(aspect='image')
  ax.set_xlabel(labelx)
  ax.set_ylabel(labelz)
  ax.set_title('f) Initial Cf ')

  ax = plt.subplot(4,2,6)
  im = ax.imshow(Cs.reshape(mz,mx),extent=[min(xc)*scalx, max(xc)*scalx, min(zc)*scalx, max(zc)*scalx],cmap='magma',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.60,label=labelCs)
  ax.axis(aspect='image')
  ax.set_xlabel(labelx)
  ax.set_ylabel(labelz)
  ax.set_title('g) Initial Cs ')

  ax = plt.subplot(4,2,8)
  im = ax.imshow(phi.reshape(mz,mx),extent=[min(xc)*scalx, max(xc)*scalx, min(zc)*scalx, max(zc)*scalx],cmap='magma',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.60,label=labelphi)
  ax.axis(aspect='image')
  ax.set_xlabel(labelx)
  ax.set_ylabel(labelz)
  ax.set_title('h) Initial phi ')

  if (dim == 1):
    fout = fname+'_dim_initial'
  else:
    fout = fname+'_initial'
  plt.savefig(fout+'.pdf')
  plt.close()

# ---------------------------------------
# Main
# ---------------------------------------

# Parameters
fname = 'out_model'

# Run test
str1 = '../MORbuoyancy.app'+ \
  ' -options_file ../model_test.opts -nx 200 -nz 100 -log_view '#+' > '+fname+'.out'
print(str1)
os.system(str1)

# Plot initial conditions
plot_initial_solution(fname,0)
plot_initial_solution(fname,1)

os.system('rm -r __pycache__')