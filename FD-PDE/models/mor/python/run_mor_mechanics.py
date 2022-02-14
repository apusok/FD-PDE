print('# --------------------------------------- #')
print('# Mid-ocean ridge model (mechanics) ')
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
def load_initial_solution(fname):
  # Load data
  fout = fname+'_PV_initial'
  imod = importlib.import_module(fout) # P,v
  data_PV = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(data_PV)

  fout = fname+'_phi_initial'
  imod = importlib.import_module(fout)  # porosity
  data_phi = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(data_phi)

  fout = fname+'_PV_initial_residual'
  imod = importlib.import_module(fout)  # pv residual
  res_PV = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(res_PV)

  return data_PV, data_phi, res_PV

# ---------------------------------------
def load_solution_data(fname,istep):

  if (istep < 10): ft = '_ts00'+str(istep)
  if (istep >= 10) & (istep < 99): ft = '_ts0'+str(istep)
  if (istep >= 100): ft = '_ts'+str(istep)

  # Load data
  fout = fname+'_PV_nd'+ft
  imod = importlib.import_module(fout) # P,v
  data_PV = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(data_PV)

  fout = fname+'_xF_nd'+ft
  imod = importlib.import_module(fout) # fluid velocity
  data_xF = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(data_xF)

  fout = fname+'_phi_nd'+ft
  imod = importlib.import_module(fout)  # porosity
  data_phi = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(data_phi)

  fout = fname+'_PV_residual'+ft
  imod = importlib.import_module(fout) # P,v residual
  res_PV = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(res_PV)

  fout = fname+'_phi_residual'+ft
  imod = importlib.import_module(fout)  # porosity residual
  res_phi = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(res_phi)

  return data_PV, data_phi, data_xF, res_PV, res_phi

# ---------------------------------------
def plot_all_fields(fname,istep,data_PV,data_phi,data_xF,phi0):

  if (istep < 10): ft = '_ts00'+str(istep)
  if (istep >= 10) & (istep < 99): ft = '_ts0'+str(istep)
  if (istep >= 100): ft = '_ts'+str(istep)

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

  if (istep>-1):
    vfx = data_xF['X_face_x']
    vfz = data_xF['X_face_y']

  phi = data_phi['X_cell']
  phi = 1.0 - phi # was solved for Q = 1-phi
  phi = phi/phi0
  
  # Plot all fields
  fig = plt.figure(1,figsize=(12,9))
  cmaps='RdBu_r' 

  pmax = max(p)
  pmin = min(p)
  vxmax = max(vx)
  vxmin = min(vx)
  vzmax = max(vz)
  vzmin = min(vz)
  phimax = max(phi)
  phimin = min(phi)
  vfxmax = 10*max(vx)
  vfxmin = 10*min(vx)
  vfzmax = 10*max(vz)
  vfzmin = 10*min(vz)

  ax = plt.subplot(3,2,1)
  im = ax.imshow(np.flipud(phi.reshape(mz,mx)),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=phimin,vmax=phimax,cmap=cmaps)
  ax.set_title(r'$\phi/\phi_0$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.825)

  ax = plt.subplot(3,2,2)
  im = ax.imshow(np.flipud(p.reshape(mz,mx)),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=pmin,vmax=pmax,cmap=cmaps)
  ax.set_title(r'$P/P_0$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.825)

  ax = plt.subplot(3,2,3)
  im = ax.imshow(np.flipud(vx.reshape(mz,mx+1)),extent=[min(xv), max(xv), min(zc), max(zc)],vmin=vxmin,vmax=vxmax,cmap=cmaps)
  ax.set_title(r'$v_s^x/v_0$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.825)

  ax = plt.subplot(3,2,4)
  im = ax.imshow(np.flipud(vz.reshape(mz+1,mx)),extent=[min(xc), max(xc), min(zv), max(zv)],vmin=vzmin,vmax=vzmax,cmap=cmaps)
  ax.set_title(r'$v_s^z/v_0$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.825)

  if (istep>-1):
    ax = plt.subplot(3,2,5)
    im = ax.imshow(np.flipud(vfx.reshape(mz,mx+1)),extent=[min(xv), max(xv), min(zc), max(zc)],vmin=vfxmin,vmax=vfxmax,cmap=cmaps)
    ax.set_title(r'$v_f^x/v_0$')
    cbar = fig.colorbar(im,ax=ax, shrink=0.825)

    ax = plt.subplot(3,2,6)
    im = ax.imshow(np.flipud(vfz.reshape(mz+1,mx)),extent=[min(xc), max(xc), min(zv), max(zv)],vmin=vfzmin,vmax=vfzmax,cmap=cmaps)
    ax.set_title(r'$v_f^z/v_0$')
    cbar = fig.colorbar(im,ax=ax, shrink=0.825)

  if (istep==-1):
    fout = fname+'_all_fields_initial'
  else:
    fout = fname+'_all_fields'+ft
  plt.savefig(fout+'.pdf')
  plt.close()

# ---------------------------------------
def plot_residuals(fname,istep,res_PV,*args):

  if (istep < 10): ft = '_ts00'+str(istep)
  if (istep >= 10) & (istep < 99): ft = '_ts0'+str(istep)
  if (istep >= 100): ft = '_ts'+str(istep)

  # Split data
  mx = res_PV['Nx'][0]
  mz = res_PV['Ny'][0]
  xc = res_PV['x1d_cell']
  zc = res_PV['y1d_cell']
  xv = res_PV['x1d_vertex']
  zv = res_PV['y1d_vertex']
  vx = res_PV['X_face_x']
  vz = res_PV['X_face_y']
  p = res_PV['X_cell']

  if (len(args)==4): 
    res_phi = args[3]
    phi = res_phi['X_cell']
    phimax = max(phi)
    phimin = min(phi)
    # phi = 1.0 - phi # was solved for Q = 1-phi
    # phi = phi/phi0
  
  # Plot all residuals
  fig = plt.figure(1,figsize=(9,9))
  cmaps='RdBu_r' 

  pmax = max(p)
  pmin = min(p)
  vxmax = max(vx)
  vxmin = min(vx)
  vzmax = max(vz)
  vzmin = min(vz)

  ax = plt.subplot(2,2,1)
  if (tstep>-1) and (len(args)==4):
    im = ax.imshow(np.flipud(phi.reshape(mz,mx)),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=phimin,vmax=phimax,cmap=cmaps)
    cbar = fig.colorbar(im,ax=ax, shrink=0.825)
  ax.set_title(r'$\phi/\phi_0$')

  ax = plt.subplot(2,2,2)
  im = ax.imshow(np.flipud(p.reshape(mz,mx)),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=pmin,vmax=pmax,cmap=cmaps)
  ax.set_title(r'$P/P_0$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.825)

  ax = plt.subplot(2,2,3)
  im = ax.imshow(np.flipud(vx.reshape(mz,mx+1)),extent=[min(xv), max(xv), min(zc), max(zc)],vmin=vxmin,vmax=vxmax,cmap=cmaps)
  ax.set_title(r'$v_s^x/v_0$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.825)

  ax = plt.subplot(2,2,4)
  im = ax.imshow(np.flipud(vz.reshape(mz+1,mx)),extent=[min(xc), max(xc), min(zv), max(zv)],vmin=vzmin,vmax=vzmax,cmap=cmaps)
  ax.set_title(r'$v_s^z/v_0$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.825)

  if (istep==-1):
    fout = fname+'_residual_initial'
  else:
    fout = fname+'_residual'+ft
  plt.savefig(fout+'.pdf')
  plt.close()

# ---------------------------------------
def plot_solution(fname,istep,data_PV,data_phi,data_xF,phi0):

  if (istep < 10): ft = '_ts00'+str(istep)
  if (istep >= 10) & (istep < 99): ft = '_ts0'+str(istep)
  if (istep >= 100): ft = '_ts'+str(istep)

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

  if (istep>-1):
    vfx = data_xF['X_face_x']
    vfz = data_xF['X_face_y']

    vfxr = vfx.reshape(mz  ,mx+1)
    vfzr = vfz.reshape(mz+1,mx  )

  phi = data_phi['X_cell']
  phi = 1.0 - phi # was solved for Q = 1-phi
  phi = phi/phi0
  
  # Compute cell center velocities
  vxr  = vx.reshape(mz  ,mx+1)
  vzr  = vz.reshape(mz+1,mx  )
  
  vxc  = np.zeros([mz,mx])
  vzc  = np.zeros([mz,mx])
  vfxc = np.zeros([mz,mx])
  vfzc = np.zeros([mz,mx])
  vc   = np.zeros([mz,mx])

  for i in range(0,mx):
    for j in range(0,mz):
      vxc[j][i]  = 0.5 * (vxr[j][i+1] + vxr[j][i])
      vzc[j][i]  = 0.5 * (vzr[j+1][i] + vzr[j][i])
      vc[j][i] = (vxc[j][i]**2+vzc[j][i]**2)**0.5
      if (istep>-1):
        vfxc[j][i] = 0.5 * (vfxr[j][i+1] + vfxr[j][i])
        vfzc[j][i] = 0.5 * (vfzr[j+1][i] + vfzr[j][i])

  # Plot one figure
  fig, ax1 = plt.subplots(1,figsize=(9,6))
  nind = 4
  iind = 2

  contours = ax1.contour(xc,zc,p.reshape(mz,mx), colors='k',linewidths=0.5)
  # ax1.clabel(contours, contours.levels, inline=True, fontsize=8)
  im = ax1.imshow( phi.reshape(mz,mx), extent=[min(xc), max(xc), min(zc), max(zc)],vmin=1.0, vmax=1.05,
                  origin='lower', cmap='ocean_r', interpolation='nearest')
  cbar = fig.colorbar(im,ax=ax1, shrink=0.60,label=r'$\phi/\phi_0$' )
  Q  = ax1.quiver( xc[::nind], zc[::nind], vxc[::nind,::nind], vzc[::nind,::nind], color='grey', units='width', pivot='mid')
  lw = 2*vc/vc.max()
  # ax1.streamplot(xc[1::],zc[1::],vxc[1::,1::],vzc[1::,1::], density=0.6, color='grey', linewidth=lw[1::,1::])
  if (istep>-1):
    Qf = ax1.quiver( xc[iind::nind], zc[iind::nind], vfxc[iind::nind,iind::nind], vfzc[iind::nind,iind::nind], 
    color='w', units='width', pivot='mid', width=0.002, headaxislength=3)
  ax1.axis('image')
  ax1.set_xlabel('x/h')
  ax1.set_ylabel('z/h')

  if (istep==-1):
    ax1.set_title('MOR - initial solution')
    fout = fname+'_sol_initial'
  else:
    ax1.set_title('MOR tstep = '+str(istep))
    fout = fname+'_sol'+ft

  plt.savefig(fout+'.pdf')
  plt.close()

# ---------------------------------------
# Main script
# ---------------------------------------

# Parameters
fname = 'out_mor'
factor = 1
nz = 50
nx = factor*nz
H  = 100e3 # [m]
L  = factor*H # [m]
k_hat = 0.0 # unit vertical vector [0.0, 1.0]
xMOR = 200e3 # [m]

tout  = 1       # output every X steps
tstep = 10       # max no of timesteps
tmax  = 1.0e6   # max time [yr]
dtmax = 1.0e3  # max dt [yr]

# Physical Parameters
K0   = 1e-7 # permeability [m^2]
u0   = 2.0 # half-spreading rate [cm/yr]
eta  = 1e19 # shear viscosity [Pa.s]
zeta = 1e20 # bulk viscosity [Pa.s]
phi0 = 0.01

# Solver options
pv_solver  = ' -pv_pc_type lu -pv_pc_factor_mat_solver_type umfpack -pv_pc_factor_mat_ordering_type external -pv_snes_monitor_true_residual -pv_ksp_monitor_true_residual -pv_snes_converged_reason -pv_ksp_converged_reason'
phi_solver = ' -phi_pc_type lu -phi_pc_factor_mat_solver_type umfpack -phi_pc_factor_mat_ordering_type external -phi_snes_monitor_true_residual -phi_ksp_monitor_true_residual -phi_snes_converged_reason -phi_ksp_converged_reason'

# Run test
str1 = '../mor_mechanics.app'+pv_solver+phi_solver+ \
  ' -L '+str(L)+ \
  ' -H '+str(H)+ \
  ' -xMOR '+str(xMOR)+ \
  ' -K0 '+str(K0)+ \
  ' -u0 '+str(u0)+ \
  ' -eta '+str(eta)+ \
  ' -zeta '+str(zeta)+ \
  ' -phi0 '+str(phi0)+ \
  ' -dtmax '+str(dtmax)+ \
  ' -tmax '+str(tmax)+ \
  ' -k_hat '+str(k_hat)+ \
  ' -tstep '+str(tstep)+ \
  ' -output_file '+fname+ \
  ' -tout '+str(tout)+ \
  ' -nx '+str(nx)+' -nz '+str(nz)+' > '+fname+'.out'
print(str1)
os.system(str1)

# Plot initial data
data_xF_init = []
data_PV_init, data_phi_init, res_PV_init = load_initial_solution(fname)
plot_all_fields(fname,-1,data_PV_init,data_phi_init,data_xF_init,phi0)
plot_solution(fname,-1,data_PV_init,data_phi_init,data_xF_init,phi0)
plot_residuals(fname,-1,res_PV_init)

# Plot solution every timestep
for istep in range(0,tstep,tout):
  data_PV, data_phi, data_xF, res_PV, res_phi = load_solution_data(fname,istep)
  plot_all_fields(fname,istep,data_PV,data_phi,data_xF,phi0)
  plot_solution(fname,istep,data_PV,data_phi,data_xF,phi0)
  plot_residuals(fname,istep,res_PV,res_phi)

os.system('rm -r __pycache__')