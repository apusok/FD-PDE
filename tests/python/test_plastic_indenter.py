# ---------------------------------------
# Rigid indenter test
# ---------------------------------------

# Import modules
import numpy as np
import matplotlib.pyplot as plt
import importlib
import os

# ---------------------------------------
# Function definitions
# ---------------------------------------
def plot_solution(fname,nx,initial):

  # Load data
  if (initial == 0):
    fout = fname+'_solution_initial'
  else:
    fout = fname+'_solution' 
  imod = importlib.import_module(fout) 
  data = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(data)

  mx = data['Nx'][0]
  mz = data['Ny'][0]
  xc = data['x1d_cell']
  zc = data['y1d_cell']
  xv = data['x1d_vertex']
  zv = data['y1d_vertex']
  vx = data['X_face_x']
  vz = data['X_face_y']
  p  = data['X_cell']

  if (initial == 0):
    fout = fname+'_coefficient_initial'
  else:
    fout = fname+'_coefficient'
  imod = importlib.import_module(fout) 
  data = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(data)

  etac_data = data['X_cell']
  etan = data['X_vertex']

  # split into dofs
  dof = 2
  etac = etac_data[1::dof]

  # Prepare cell center velocities
  vxface = vx.reshape(mz  ,mx+1)
  vzface = vz.reshape(mz+1,mx  )

  # Compute the cell center values from the face data by averaging neighbouring faces
  vxc = np.zeros( [mz , mx] )
  vzc = np.zeros( [mz , mx] )
  vc  = np.zeros( [mz , mx] )
  for i in range(0,mx):
    for j in range(0,mz):
      vxc[j][i] = 0.5 * (vxface[j][i+1] + vxface[j][i])
      vzc[j][i] = 0.5 * (vzface[j+1][i] + vzface[j][i])
      vc [j][i] = (vxc[j][i]**2+vzc[j][i]**2)**0.5

  velmax = max(max(vx),max(vz))
  velmin = min(min(vx),min(vz))

  # Plot all fields - P, vx, vz, v, etac, etan
  fig = plt.figure(1,figsize=(12,8))
  cmaps='RdBu_r' 

  ax = plt.subplot(2,3,1)
  im = ax.imshow(p.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps,origin='lower')
  ax.set_title(r'$P$')
  ax.set_ylabel('z')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(2,3,2)
  im = ax.imshow(vx.reshape(mz,mx+1),extent=[min(xv), max(xv), min(zc), max(zc)],vmin=velmin,vmax=velmax,cmap=cmaps,origin='lower')
  ax.set_title(r'$V_x$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(2,3,3)
  im = ax.imshow(vz.reshape(mz+1,mx),extent=[min(xc), max(xc), min(zv), max(zv)],vmin=velmin,vmax=velmax,cmap=cmaps,origin='lower')
  ax.set_title(r'$V_z$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(2,3,4)
  im = ax.imshow(vc.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps,origin='lower')
  ax.set_title(r'$V$ magnitude')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(2,3,5)
  im = ax.imshow(np.log10(etac.reshape(mz,mx)),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=-3,vmax=3,cmap=cmaps,origin='lower')
  ax.set_title(r'$\eta_{center}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(2,3,6)
  im = ax.imshow(np.log10(etan.reshape(mz+1,mx+1)),extent=[min(xv), max(xv), min(zv), max(zv)],vmin=-3,vmax=3,cmap=cmaps,origin='lower')
  ax.set_title(r'$\eta_{corner}$')
  ax.set_ylabel('z')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  plt.tight_layout() 
  if (initial == 0):
    fout = fname+'_solution_initial'+'_nx_'+str(nx)
  else:
    fout = fname+'_solution'+'_nx_'+str(nx)
  plt.savefig(fout+'.pdf')
  plt.close()

# ---------------------------------------
def plot_half_solution(fname,nx):

  # Load data
  fout = fname+'_solution' 
  imod = importlib.import_module(fout) 
  data = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(data)

  mx = data['Nx'][0]
  mz = data['Ny'][0]
  xc = data['x1d_cell']
  zc = data['y1d_cell']
  xv = data['x1d_vertex']
  zv = data['y1d_vertex']
  vx = data['X_face_x']
  vz = data['X_face_y']
  p  = data['X_cell']

  fout = fname+'_coefficient'
  imod = importlib.import_module(fout) 
  data = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(data)

  etac_data = data['X_cell']
  etan = data['X_vertex']

  # split into dofs
  dof = 2
  etac = etac_data[1::dof]

  fout = fname+'_strain' # 1. Numerical solution stokes - strain rates
  imod = importlib.import_module(fout) 
  data = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(data)

  mx = data['Nx'][0]
  mz = data['Ny'][0]
  xc = data['x1d_cell']
  zc = data['y1d_cell']
  xv = data['x1d_vertex']
  zv = data['y1d_vertex']
  eps1n = data['X_vertex']
  eps1c = data['X_cell']

  # split into dofs
  dof = 4
  exx1c = eps1c[0::dof]
  ezz1c = eps1c[1::dof]
  exz1c = eps1c[2::dof]
  eII1c = eps1c[3::dof]

  exx1n = eps1n[0::dof]
  ezz1n = eps1n[1::dof]
  exz1n = eps1n[2::dof]
  eII1n = eps1n[3::dof]

  # Prepare cell center velocities
  vxface = vx.reshape(mz  ,mx+1)
  vzface = vz.reshape(mz+1,mx  )

  # Compute the cell center values from the face data by averaging neighbouring faces
  vxc = np.zeros( [mz , mx] )
  vzc = np.zeros( [mz , mx] )
  vc  = np.zeros( [mz , mx] )
  for i in range(0,mx):
    for j in range(0,mz):
      vxc[j][i] = 0.5 * (vxface[j][i+1] + vxface[j][i])
      vzc[j][i] = 0.5 * (vzface[j+1][i] + vzface[j][i])
      vc [j][i] = (vxc[j][i]**2+vzc[j][i]**2)**0.5

  # velmax = max(max(vx),max(vz))
  # velmin = min(min(vx),min(vz))

  # Plot all fields - P, vx, vz, v, etac, etan
  fig = plt.figure(1,figsize=(6,9))
  cmaps='RdBu_r' 

  zlevel = 0.2

  # viscosity
  ax = plt.subplot(4,1,1)
  im = ax.imshow(np.log10(etac.reshape(mz,mx)),extent=[min(xc), max(xc), zlevel, max(zc)],vmin=-3,vmax=3,cmap=cmaps,origin='lower')
  ax.set_title(r'Viscosity $\eta_{center}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  # strain rate
  ax = plt.subplot(4,1,2)
  im = ax.imshow(np.log10(eII1c.reshape(mz,mx)),extent=[min(xc), max(xc), zlevel, max(zc)],cmap=cmaps,origin='lower')
  ax.set_title(r'Strain rate invariant $\epsilon_{II}^{CENTER}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  # velocity
  ax = plt.subplot(4,1,3)
  im = ax.imshow(vc.reshape(mz,mx),extent=[min(xc), max(xc), zlevel, max(zc)],cmap=cmaps,origin='lower')
  ax.set_title(r'Velocity $V$ (magnitude)')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  # pressure
  ax = plt.subplot(4,1,4)
  im = ax.imshow(p.reshape(mz,mx),extent=[min(xc), max(xc), zlevel, max(zc)],cmap=cmaps,origin='lower')
  ax.set_title(r'Pressure $P$')
  ax.set_ylabel('z')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  plt.tight_layout() 

  fout = fname+'_half_solution'+'_nx_'+str(nx)
  plt.savefig(fout+'.pdf')
  plt.close()

# ---------------------------------------
def plot_strain_rates(fname,nx):

  # Load data
  fout = fname+'_strain' # 1. Numerical solution stokes - strain rates
  imod = importlib.import_module(fout) 
  data = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(data)

  mx = data['Nx'][0]
  mz = data['Ny'][0]
  xc = data['x1d_cell']
  zc = data['y1d_cell']
  xv = data['x1d_vertex']
  zv = data['y1d_vertex']
  eps1n = data['X_vertex']
  eps1c = data['X_cell']

  # split into dofs
  dof = 4
  exx1c = eps1c[0::dof]
  ezz1c = eps1c[1::dof]
  exz1c = eps1c[2::dof]
  eII1c = eps1c[3::dof]

  exx1n = eps1n[0::dof]
  ezz1n = eps1n[1::dof]
  exz1n = eps1n[2::dof]
  eII1n = eps1n[3::dof]

  exxmax = max(max(exx1c),max(exx1n))
  exxmin = min(min(exx1c),min(exx1n))
  ezzmax = max(max(ezz1c),max(ezz1n))
  ezzmin = min(min(ezz1c),min(ezz1n))
  exzmax = max(max(exz1c),max(exz1n))
  exzmin = min(min(exz1c),min(exz1n))
  eIImax = max(max(eII1c),max(eII1n))
  eIImin = min(min(eII1c),min(eII1n))

  # Plot all fields - epsII, epsxx, epszz, epsxz 
  fig = plt.figure(1,figsize=(16,8))
  cmaps='RdBu_r' 

  ax = plt.subplot(2,4,1)
  im = ax.imshow(exx1c.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=exxmin,vmax=exxmax,cmap=cmaps,origin='lower')
  ax.set_title(r'$\epsilon_{xx}^{CENTER}$')
  ax.set_ylabel('z')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(2,4,2)
  im = ax.imshow(ezz1c.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=ezzmin,vmax=ezzmax,cmap=cmaps,origin='lower')
  ax.set_title(r'$\epsilon_{zz}^{CENTER}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(2,4,3)
  im = ax.imshow(exz1c.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=exzmin,vmax=exzmax,cmap=cmaps,origin='lower')
  ax.set_title(r'$\epsilon_{xz}^{CENTER}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(2,4,4)
  im = ax.imshow(eII1c.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=eIImin,vmax=eIImax,cmap=cmaps,origin='lower')
  ax.set_title(r'$\epsilon_{II}^{CENTER}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(2,4,5)
  im = ax.imshow(exx1n.reshape(mz+1,mx+1),extent=[min(xv), max(xv), min(zv), max(zv)],vmin=exxmin,vmax=exxmax,cmap=cmaps,origin='lower')
  ax.set_title(r'$\epsilon_{xx}^{CORNER}$')
  ax.set_ylabel('z')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(2,4,6)
  im = ax.imshow(ezz1n.reshape(mz+1,mx+1),extent=[min(xv), max(xv), min(zv), max(zv)],vmin=ezzmin,vmax=ezzmax,cmap=cmaps,origin='lower')
  ax.set_title(r'$\epsilon_{zz}^{CORNER}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(2,4,7)
  im = ax.imshow(exz1n.reshape(mz+1,mx+1),extent=[min(xv), max(xv), min(zv), max(zv)],vmin=exzmin,vmax=exzmax,cmap=cmaps,origin='lower')
  ax.set_title(r'$\epsilon_{xz}^{CORNER}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(2,4,8)
  im = ax.imshow(eII1n.reshape(mz+1,mx+1),extent=[min(xv), max(xv), min(zv), max(zv)],vmin=eIImin,vmax=eIImax,cmap=cmaps,origin='lower')
  ax.set_title(r'$\epsilon_{II}^{CORNER}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  plt.tight_layout() 
  fout = fname+'_strain'+'_nx_'+str(nx)
  plt.savefig(fout+'.pdf')
  plt.close()

# ---------------------------------------
def plot_residuals(fname,nx):

  # Load data
  fout = fname+'_residual' 
  imod = importlib.import_module(fout) 
  data = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(data)

  mx = data['Nx'][0]
  mz = data['Ny'][0]
  xc = data['x1d_cell']
  zc = data['y1d_cell']
  xv = data['x1d_vertex']
  zv = data['y1d_vertex']
  vx = data['X_face_x']
  vz = data['X_face_y']
  p  = data['X_cell']

  # Plot all fields - P, vx, vz, v, etac, etan
  fig = plt.figure(1,figsize=(12,4))
  cmaps='RdBu_r' 

  ax = plt.subplot(1,3,1)
  im = ax.imshow(p.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps,origin='lower')
  ax.set_title(r'Residual $P$')
  ax.set_ylabel('z')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(1,3,2)
  im = ax.imshow(vx.reshape(mz,mx+1),extent=[min(xv), max(xv), min(zc), max(zc)],cmap=cmaps,origin='lower')
  ax.set_title(r'Residual $V_x$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(1,3,3)
  im = ax.imshow(vz.reshape(mz+1,mx),extent=[min(xc), max(xc), min(zv), max(zv)],cmap=cmaps,origin='lower')
  ax.set_title(r'Residual $V_z$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  plt.tight_layout() 
  fout = fname+'_residual'+'_nx_'+str(nx)
  plt.savefig(fout+'.pdf')
  plt.close()

# ---------------------------------------
# Main script
# ---------------------------------------
print('# --------------------------------------- #')
print('# Plastic indenter test ')
print('# --------------------------------------- #')

# Set main parameters and run test
fname = 'out_indenter'
nx    = 100 # resolution
g     = 0.0
eta0  = 1000
fout = fname+'_'+str(nx)+'.out'
harmonic = 1
C        = 1 # cohesion, default C=1

# Run simulation
str1 = '../test_plastic_indenter -pc_type lu -pc_factor_mat_solver_type umfpack -pc_factor_mat_ordering_type external -snes_monitor_true_residual -ksp_monitor_true_residual -snes_converged_reason -ksp_converged_reason'+ \
    ' -output_file '+fname+ \
    ' -g '+str(g)+ \
    ' -eta0 '+str(eta0)+ \
    ' -C '+str(C)+ \
    ' -harmonic '+str(harmonic)+ \
    ' -nx '+str(nx)+' -nz '+str(nx)+' > '+fout
print(str1)
os.system(str1)

# Plot solution and error
plot_solution(fname,nx,0)
plot_solution(fname,nx,1)
plot_strain_rates(fname,nx)
plot_residuals(fname,nx)
plot_half_solution(fname,nx)

os.system('rm -r __pycache__')