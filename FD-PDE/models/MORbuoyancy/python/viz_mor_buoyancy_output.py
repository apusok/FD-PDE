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
def plot_solution_PV(fname,istep,dim):

  # Load data
  # if (dim == 1):
  #   fout = fname+'_PV_dim_'+str(istep)
  # else:
  #   fout = fname+'_PV_'+str(istep)
  
  fout = 'out_xPV_ts'+str(istep)
  imod = importlib.import_module(fout) # P,v
  data_PV = imod._PETScBinaryLoad()

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
  fig = plt.figure(1,figsize=(14,7))
  nind = 4

  labelP = r'$P [-]$'
  labelv = r'$V [-]$'
  labelx = 'x/h'
  labelz = 'z/h'
  scalx  = 1
  scalP  = 1
  scalv  = 1 

  SEC_YEAR = 31536000

  # Transform to geounits
  if (dim == 1):
    scalx  = 1e2 # km
    scalP  = 1e-9 # GPa
    scalv  = 1.0e2*SEC_YEAR # cm/yr
    labelP = r'$P [GPa]$'
    labelv = r'$V [cm/yr]$'
    labelx = 'x [km]'
    labelz = 'z [km]'

  ax = plt.subplot(2,2,1)
  im = ax.imshow( p.reshape(mz,mx)*scalP, extent=[min(xc)*scalx, max(xc)*scalx, min(zc)*scalx, max(zc)*scalx],
                  origin='lower', cmap='ocean', interpolation='nearest')
  # im.set_clim(-10,0)
  cbar = fig.colorbar(im,ax=ax, shrink=0.60,label=labelP )
  Q  = ax.quiver( xc[::nind]*scalx, zc[::nind]*scalx, vxc[::nind,::nind]*scalv, vzc[::nind,::nind]*scalv, color='grey', units='width', pivot='mid')
  ax.axis(aspect='image')
  ax.set_xlabel(labelx)
  ax.set_ylabel(labelz)
  ax.set_title('a) Stokes-Darcy tstep = '+str(istep))

  ax = plt.subplot(2,2,2)
  im = ax.imshow(p.reshape(mz,mx)*scalP,extent=[min(xc)*scalx, max(xc)*scalx, min(zc)*scalx, max(zc)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.60,label=labelP)
  ax.axis(aspect='image')
  ax.set_xlabel(labelx)
  ax.set_ylabel(labelz)
  ax.set_title('b) P')

  ax = plt.subplot(2,2,3)
  im = ax.imshow(vxr*scalv,extent=[min(xv)*scalx, max(xv)*scalx, min(zc)*scalx, max(zc)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.60,label=labelv)
  ax.axis(aspect='image')
  ax.set_xlabel(labelx)
  ax.set_ylabel(labelz)
  ax.set_title(r'$c) V_s^x$')

  ax = plt.subplot(2,2,4)
  im = ax.imshow(vzr*scalv,extent=[min(xc)*scalx, max(xc)*scalx, min(zv)*scalx, max(zv)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.60,label=labelv)
  ax.axis(aspect='image')
  ax.set_xlabel(labelx)
  ax.set_ylabel(labelz)
  ax.set_title(r'$d) V_s^z$')

  if (dim == 1):
    fout = fname+'_PV_dim_'+str(istep)
  else:
    fout = fname+'_PV_'+str(istep)
  plt.savefig(fout+'.pdf')
  plt.close()

# ---------------------------------------
def plot_solution_HC(fname,istep,dim):

  fout = 'out_xHC_ts'+str(istep)
  imod = importlib.import_module(fout)
  data_HC = imod._PETScBinaryLoad()

  # Split data
  mx = data_HC['Nx'][0]
  mz = data_HC['Ny'][0]
  xc = data_HC['x1d_cell']
  zc = data_HC['y1d_cell']
  HC_data = data_HC['X_cell']
  dof = 2
  H = HC_data[0::dof]
  C = HC_data[1::dof]

  if (istep==0):
    fout = 'out_Plith_ts'+str(istep)
    imod = importlib.import_module(fout)
    data_P = imod._PETScBinaryLoad()
    P = data_P['X_cell']

    fout = 'out_xHC_halfspace_ts'+str(istep)
    imod = importlib.import_module(fout)
    data_HC_hs = imod._PETScBinaryLoad()

    HC_data_hs = data_HC_hs['X_cell']
    H_hs = HC_data_hs[0::dof]

  # Plot one figure
  fig = plt.figure(1,figsize=(14,7))
  nind = 4

  scalx  = 1
  scalH  = 1
  labelC = r'$\Theta$'
  labelH  = r'$H [-]$'
  labelx = 'x/h'
  labelz = 'z/h'

  # Transform to geounits
  if (dim == 1):
    scalx  = 1e2 # km
    scalH  = 1
    labelx = 'x [km]'
    labelz = 'z [km]'
    labelC = r'$C$'
    labelH  = r'$H []$'

  ax = plt.subplot(2,2,1)
  im = ax.imshow(H.reshape(mz,mx)*scalH,extent=[min(xc)*scalx, max(xc)*scalx, min(zc)*scalx, max(zc)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.60,label=labelH)
  ax.axis(aspect='image')
  ax.set_xlabel(labelx)
  ax.set_ylabel(labelz)
  ax.set_title('a) H tstep = '+str(istep))

  ax = plt.subplot(2,2,2)
  im = ax.imshow(C.reshape(mz,mx),extent=[min(xc)*scalx, max(xc)*scalx, min(zc)*scalx, max(zc)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.60,label=labelC)
  ax.axis(aspect='image')
  ax.set_xlabel(labelx)
  ax.set_ylabel(labelz)
  ax.set_title('b) C ')

  if (istep==0):
    ax = plt.subplot(2,2,3)
    im = ax.imshow((H.reshape(mz,mx)-H_hs.reshape(mz,mx))*scalH,extent=[min(xc)*scalx, max(xc)*scalx, min(zc)*scalx, max(zc)*scalx],cmap='viridis',origin='lower')
    cbar = fig.colorbar(im,ax=ax, shrink=0.60,label=labelH)
    ax.axis(aspect='image')
    ax.set_xlabel(labelx)
    ax.set_ylabel(labelz)
    ax.set_title('c) H residual (corrected-halfspace)')

    ax = plt.subplot(2,2,4)
    im = ax.imshow(P.reshape(mz,mx),extent=[min(xc)*scalx, max(xc)*scalx, min(zc)*scalx, max(zc)*scalx],cmap='viridis',origin='lower')
    cbar = fig.colorbar(im,ax=ax, shrink=0.60,label=labelC)
    ax.axis(aspect='image')
    ax.set_xlabel(labelx)
    ax.set_ylabel(labelz)
    ax.set_title('d) Plith ')

  if (dim == 1):
    fout = fname+'_HC_dim_'+str(istep)
  else:
    fout = fname+'_HC_'+str(istep)
  plt.savefig(fout+'.pdf')
  plt.close()

# ---------------------------------------
def plot_solution_phiT(fname,istep,dim):

  fout = 'out_xphiT_ts'+str(istep)
  imod = importlib.import_module(fout)
  data_phiT = imod._PETScBinaryLoad()

  # Split data
  mx = data_phiT['Nx'][0]
  mz = data_phiT['Ny'][0]
  xc = data_phiT['x1d_cell']
  zc = data_phiT['y1d_cell']
  phiT_data = data_phiT['X_cell']
  dof = 2
  phi = phiT_data[0::dof]
  T = phiT_data[1::dof]

  # Plot one figure
  fig = plt.figure(1,figsize=(14,5))
  nind = 4

  scalx  = 1
  scalT  = 0
  labelphi = r'$\phi$'
  labelT  = r'$T [-]$'
  labelx = 'x/h'
  labelz = 'z/h'

  # Transform to geounits
  if (dim == 1):
    scalx  = 1e2 # km
    scalT  = 0
    labelx = 'x [km]'
    labelz = 'z [km]'
    labelphi = r'$\phi$'
    labelT  = r'$T [-]$'

  ax = plt.subplot(1,2,1)
  im = ax.imshow(phi.reshape(mz,mx),extent=[min(xc)*scalx, max(xc)*scalx, min(zc)*scalx, max(zc)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.60,label=labelphi)
  ax.axis(aspect='image')
  ax.set_xlabel(labelx)
  ax.set_ylabel(labelz)
  ax.set_title('a) phi tstep = '+str(istep))

  ax = plt.subplot(1,2,2)
  im = ax.imshow(T.reshape(mz,mx)-scalT,extent=[min(xc)*scalx, max(xc)*scalx, min(zc)*scalx, max(zc)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.60,label=labelT)
  ax.axis(aspect='image')
  ax.set_xlabel(labelx)
  ax.set_ylabel(labelz)
  ax.set_title('b) T')

  if (dim == 1):
    fout = fname+'_phiT_dim_'+str(istep)
  else:
    fout = fname+'_phiT_'+str(istep)
  plt.savefig(fout+'.pdf')
  plt.close()

# ---------------------------------------
def plot_solution_Vel(fname,istep,dim):

  fout = 'out_xVel_ts'+str(istep)
  imod = importlib.import_module(fout)
  data_Vel = imod._PETScBinaryLoad()

  # Split data
  mx = data_Vel['Nx'][0]
  mz = data_Vel['Ny'][0]
  xc = data_Vel['x1d_cell']
  zc = data_Vel['y1d_cell']
  xv = data_Vel['x1d_vertex']
  zv = data_Vel['y1d_vertex']
  vx = data_Vel['X_face_x']
  vz = data_Vel['X_face_y']

  dof = 2
  vfx = vx[0::dof]
  vbx = vx[1::dof]
  vfz = vz[0::dof]
  vbz = vz[1::dof]

  # Plot one figure
  fig = plt.figure(1,figsize=(14,7))
  nind = 4

  labelv = r'$V [-]$'
  labelx = 'x/h'
  labelz = 'z/h'
  scalx  = 1
  scalv  = 1 

  SEC_YEAR = 31536000

  # Transform to geounits
  if (dim == 1):
    scalx  = 1e2 # km
    scalv  = 1.0e2*SEC_YEAR # cm/yr
    labelv = r'$V [cm/yr]$'
    labelx = 'x [km]'
    labelz = 'z [km]'

  ax = plt.subplot(2,2,1)
  im = ax.imshow(vfx.reshape(mz,mx+1)*scalv,extent=[min(xv)*scalx, max(xv)*scalx, min(zc)*scalx, max(zc)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.60,label=labelv)
  ax.axis(aspect='image')
  ax.set_xlabel(labelx)
  ax.set_ylabel(labelz)
  ax.set_title(r'$a) V_f^x$ tstep = '+str(istep))

  ax = plt.subplot(2,2,2)
  im = ax.imshow(vfz.reshape(mz+1,mx)*scalv,extent=[min(xc)*scalx, max(xc)*scalx, min(zv)*scalx, max(zv)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.60,label=labelv)
  ax.axis(aspect='image')
  ax.set_xlabel(labelx)
  ax.set_ylabel(labelz)
  ax.set_title(r'$b) V_f^z$')

  ax = plt.subplot(2,2,3)
  im = ax.imshow(vbx.reshape(mz,mx+1)*scalv,extent=[min(xv)*scalx, max(xv)*scalx, min(zc)*scalx, max(zc)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.60,label=labelv)
  ax.axis(aspect='image')
  ax.set_xlabel(labelx)
  ax.set_ylabel(labelz)
  ax.set_title(r'$c) V^x$')

  ax = plt.subplot(2,2,4)
  im = ax.imshow(vbz.reshape(mz+1,mx)*scalv,extent=[min(xc)*scalx, max(xc)*scalx, min(zv)*scalx, max(zv)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.60,label=labelv)
  ax.axis(aspect='image')
  ax.set_xlabel(labelx)
  ax.set_ylabel(labelz)
  ax.set_title(r'$d) V^z$')

  if (dim == 1):
    fout = fname+'_Vel_dim_'+str(istep)
  else:
    fout = fname+'_Vel_'+str(istep)
  plt.savefig(fout+'.pdf')
  plt.close()

# ---------------------------------------
def plot_solution_Enthalpy(fname,istep,dim):

  # if (dim == 1):
  #   fout = fname+'_T_dim_'+str(istep)
  # else:
  #   fout = fname+'_T_'+str(istep)
  fout = 'out_Enthalpy_ts'+str(istep)
  imod = importlib.import_module(fout)
  data_Enth = imod._PETScBinaryLoad()

  # Split data
  mx = data_Enth['Nx'][0]
  mz = data_Enth['Ny'][0]
  xc = data_Enth['x1d_cell']
  zc = data_Enth['y1d_cell']
  En_data = data_Enth['X_cell']
  dof_en = 5+3*2

  H  = En_data[0::dof_en]
  T  = En_data[1::dof_en]
  TP = En_data[2::dof_en]
  phi= En_data[3::dof_en]
  P  = En_data[4::dof_en]
  C  = En_data[5::dof_en]
  Cs = En_data[7::dof_en]
  Cf = En_data[9::dof_en]

  # Plot one figure
  fig = plt.figure(1,figsize=(14,14))
  nind = 4

  labelT = r'$T [-]$'
  labelC = r'$\Theta$'
  labelCf = r'$\Theta_f$'
  labelCs = r'$\Theta_s$'
  labelphi= r'$\phi$'
  labelH  = r'$H [-]$'
  labelx = 'x/h'
  labelz = 'z/h'
  scalx  = 1
  scalT  = 0
  scalH  = 1

  # Transform to geounits
  if (dim == 1):
    scalx  = 1e2 # km
    scalT  = 273.15 # deg C
    scalH  = 1
    labelT = r'$T [^oC]$'
    labelx = 'x [km]'
    labelz = 'z [km]'
    labelC = r'$C$'
    labelCf = r'$C_f$'
    labelCs = r'$C_s$'
    labelH  = r'$H []$'

  ax = plt.subplot(4,2,1)
  im = ax.imshow(H.reshape(mz,mx)*scalH,extent=[min(xc)*scalx, max(xc)*scalx, min(zc)*scalx, max(zc)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.60,label=labelH)
  ax.axis(aspect='image')
  ax.set_xlabel(labelx)
  ax.set_ylabel(labelz)
  ax.set_title('a) H tstep = '+str(istep))

  ax = plt.subplot(4,2,2)
  im = ax.imshow(C.reshape(mz,mx),extent=[min(xc)*scalx, max(xc)*scalx, min(zc)*scalx, max(zc)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.60,label=labelC)
  ax.axis(aspect='image')
  ax.set_xlabel(labelx)
  ax.set_ylabel(labelz)
  ax.set_title('e) C ')

  ax = plt.subplot(4,2,3)
  im = ax.imshow(T.reshape(mz,mx)-scalT,extent=[min(xc)*scalx, max(xc)*scalx, min(zc)*scalx, max(zc)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.60,label=labelT)
  ax.axis(aspect='image')
  ax.set_xlabel(labelx)
  ax.set_ylabel(labelz)
  ax.set_title('b) T')

  ax = plt.subplot(4,2,4)
  im = ax.imshow(Cf.reshape(mz,mx),extent=[min(xc)*scalx, max(xc)*scalx, min(zc)*scalx, max(zc)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.60,label=labelCf)
  ax.axis(aspect='image')
  ax.set_xlabel(labelx)
  ax.set_ylabel(labelz)
  ax.set_title('f) Cf ')

  ax = plt.subplot(4,2,5)
  im = ax.imshow(TP.reshape(mz,mx)-scalT,extent=[min(xc)*scalx, max(xc)*scalx, min(zc)*scalx, max(zc)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.60,label=labelT)
  ax.axis(aspect='image')
  ax.set_xlabel(labelx)
  ax.set_ylabel(labelz)
  ax.set_title('c) T potential')

  ax = plt.subplot(4,2,6)
  im = ax.imshow(Cs.reshape(mz,mx),extent=[min(xc)*scalx, max(xc)*scalx, min(zc)*scalx, max(zc)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.60,label=labelCs)
  ax.axis(aspect='image')
  ax.set_xlabel(labelx)
  ax.set_ylabel(labelz)
  ax.set_title('g) Cs ')

  ax = plt.subplot(4,2,7)
  im = ax.imshow(P.reshape(mz,mx),extent=[min(xc)*scalx, max(xc)*scalx, min(zc)*scalx, max(zc)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.60,label='P')
  ax.axis(aspect='image')
  ax.set_xlabel(labelx)
  ax.set_ylabel(labelz)
  ax.set_title('d) P lith')

  ax = plt.subplot(4,2,8)
  im = ax.imshow(phi.reshape(mz,mx),extent=[min(xc)*scalx, max(xc)*scalx, min(zc)*scalx, max(zc)*scalx],cmap='viridis',origin='lower')
  cbar = fig.colorbar(im,ax=ax, shrink=0.60,label=labelphi)
  ax.axis(aspect='image')
  ax.set_xlabel(labelx)
  ax.set_ylabel(labelz)
  ax.set_title('h) phi ')

  if (dim == 1):
    fout = fname+'_Enthalpy_dim_'+str(istep)
  else:
    fout = fname+'_Enthalpy_'+str(istep)
  plt.savefig(fout+'.pdf')
  plt.close()

# ---------------------------------------
# Main
# ---------------------------------------

# Parameters
fname = 'out_model'

# Run test
str1 = '../MORbuoyancy.app'+ \
    ' -options_file ../model_test.opts -nx 200 -nz 100 -log_view -dim_output '#+' > '+fname+'.out'
  # ' -options_file ../model_test.opts -nx 30 -nz 15 -log_view -dim_output '#+' > '+fname+'.out'
# str1 = '../MORbuoyancy.app -options_file ../model_test.opts -log_view '
print(str1)
os.system(str1)

istep = 0
ndim  = 0

# Plot initial conditions
plot_solution_PV(fname,istep,ndim)
plot_solution_HC(fname,istep,ndim)
plot_solution_phiT(fname,istep,ndim)
plot_solution_Vel(fname,istep,ndim)
plot_solution_Enthalpy(fname,istep,ndim)

istep = 1
plot_solution_HC(fname,istep,ndim)
plot_solution_Enthalpy(fname,istep,ndim)
plot_solution_phiT(fname,istep,ndim)
plot_solution_PV(fname,istep,ndim)

# print coeff - command line
# import dmstagoutput as dmout
# dmout.general_output_imshow('out_xcoeff_ts0',None,None)

os.system('rm -r __pycache__')