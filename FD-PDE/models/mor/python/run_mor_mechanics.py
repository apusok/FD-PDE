print('# --------------------------------------- #')
print('# Mid-ocean ridge model (mechanics) ')
print('# --------------------------------------- #')

# Import modules
import numpy as np
import matplotlib.pyplot as plt
import importlib
import os

# Parameters
fname = 'out_mor'
nx = 50      
nz = 50
L  = 200e3
H  = 100e3
k_hat = 0.0 # unit vertical vector

tout  = 1       # output every X steps
tstep = 0      # max no of timesteps
tmax  = 1.0e3   # max time
dtmax = 1.0e-1 # max dt

# Run test
str1 = '../mor_mechanics.app -pc_type lu -pc_factor_mat_solver_type umfpack'+ \
  ' -L '+str(L)+ \
  ' -H '+str(H)+ \
  ' -dtmax '+str(dtmax)+ \
  ' -tmax '+str(tmax)+ \
  ' -k_hat '+str(k_hat)+ \
  ' -tstep '+str(tstep)+ \
  ' -output_file '+fname+ \
  ' -tout '+str(tout)+ \
  ' -nx '+str(nx)+' -nz '+str(nz)+' > '+fname+'.out'
print(str1)
os.system(str1)

# Plot solution every timestep
for istep in range(tstep+1):
  if (istep < 10): ft = '_ts00'+str(istep)
  if (istep >= 10) & (istep < 99): ft = '_ts0'+str(istep)
  if (istep >= 100): ft = '_ts'+str(istep)

  # Load data
  fout = fname+'_PV_nd'+ft
  imod = importlib.import_module(fout) # P,v
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
  p = data['X_cell']

  fout = fname+'_xF_nd'+ft
  imod = importlib.import_module(fout) # fluid velocity
  data = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(data)

  vfx = data['X_face_x']
  vfz = data['X_face_y']
  # pf = data['X_cell']

  fout = fname+'_phi_nd'+ft
  imod = importlib.import_module(fout)  # porosity
  data = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(data)
  phi = data['X_cell']
  phi = 1.0 - phi # was solved for Q = 1-phi
  
  # Open a figure
  fig = plt.figure(1,figsize=(12,9))
  cmaps='RdBu_r' 

  ax = plt.subplot(3,2,1)
  im = ax.imshow(phi.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps)
  ax.set_title(r'$\phi$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,2,2)
  im = ax.imshow(p.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps)
  ax.set_title(r'$P$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,2,3)
  im = ax.imshow(vx.reshape(mz,mx+1),extent=[min(xv), max(xv), min(zc), max(zc)],cmap=cmaps)
  ax.set_title(r'$v_s^x$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,2,4)
  im = ax.imshow(vz.reshape(mz+1,mx),extent=[min(xc), max(xc), min(zv), max(zv)],cmap=cmaps)
  ax.set_title(r'$v_s^z$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,2,5)
  im = ax.imshow(vfx.reshape(mz,mx+1),extent=[min(xv), max(xv), min(zc), max(zc)],cmap=cmaps)
  ax.set_title(r'$v_f^x$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,2,6)
  im = ax.imshow(vfz.reshape(mz+1,mx),extent=[min(xc), max(xc), min(zv), max(zv)],cmap=cmaps)
  ax.set_title(r'$v_f^z$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  fout = fname+ft
  plt.savefig(fout+'.pdf')
  plt.close()