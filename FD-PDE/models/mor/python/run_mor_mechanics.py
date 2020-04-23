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
nz = 50
nx = 2*nz
L  = 200e3 # [m]
H  = 100e3 # [m]
k_hat = 0.0 # unit vertical vector

tout  = 1       # output every X steps
tstep = 2       # max no of timesteps
tmax  = 1.0e6   # max time [yr]
dtmax = 1.0e3  # max dt [yr]

# Solver options
pv_solver  = ' -pv_pc_type lu -pv_pc_factor_mat_solver_type umfpack -pv_snes_monitor -pv_ksp_monitor -pv_snes_converged_reason -pv_ksp_converged_reason'
phi_solver = ' -phi_pc_type lu -phi_pc_factor_mat_solver_type umfpack -phi_snes_monitor -phi_ksp_monitor -phi_snes_converged_reason -phi_ksp_converged_reason'

# Run test
str1 = '../mor_mechanics.app'+pv_solver+phi_solver+ \
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
for istep in range(0,tstep,tout):
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
  print(np.count_nonzero(vfx))
  # pf = data['X_cell']

  fout = fname+'_phi_nd'+ft
  imod = importlib.import_module(fout)  # porosity
  data = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(data)
  phi = data['X_cell']
  phi = 1.0 - phi # was solved for Q = 1-phi
  
  # Compute cell center velocities
  vxr  = vx.reshape(mz  ,mx+1)
  vzr  = vz.reshape(mz+1,mx  )
  vfxr = vfx.reshape(mz  ,mx+1)
  vfzr = vfz.reshape(mz+1,mx  )

  vxc  = np.zeros([mz,mx])
  vzc  = np.zeros([mz,mx])
  vfxc = np.zeros([mz,mx])
  vfzc = np.zeros([mz,mx])
  for i in range(0,mx):
    for j in range(0,mz):
      vxc[j][i]  = 0.5 * (vxr[j][i+1] + vxr[j][i])
      vzc[j][i]  = 0.5 * (vzr[j+1][i] + vzr[j][i])
      vfxc[j][i] = 0.5 * (vfxr[j][i+1] + vfxr[j][i])
      vfzc[j][i] = 0.5 * (vfzr[j+1][i] + vfzr[j][i])

  # 1. Plot one figure
  fig, ax1 = plt.subplots(1,figsize=(9,6))
  nind = 4
  iind = 2

  contours = ax1.contour(xc,zc,p.reshape(mz,mx), colors='white',linewidths=0.5)
  # ax1.clabel(contours, contours.levels, inline=True, fontsize=8)
  im = ax1.imshow( phi.reshape(mz,mx), extent=[min(xc), max(xc), min(zc), max(zc)],
                  origin='lower', cmap='magma', interpolation='nearest')
  cbar = fig.colorbar(im,ax=ax1, shrink=0.75,label=r'$\phi$' )
  Q  = ax1.quiver( xc[::nind], zc[::nind], vxc[::nind,::nind], vzc[::nind,::nind], color='grey', units='width', pivot='mid')
  Qf = ax1.quiver( xc[iind::nind], zc[iind::nind], vfxc[iind::nind,iind::nind], vfzc[iind::nind,iind::nind], color='w', units='width', pivot='mid', width=0.002)
  ax1.axis(aspect='image')
  ax1.set_xlabel('x-dir')
  ax1.set_ylabel('z-dir')
  ax1.set_title('MOR tstep = '+str(istep))
  fout = fname+'_sol'+ft
  plt.savefig(fout+'.pdf')
  plt.close()

  # 2. Plot all fields
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
  ax.set_title(r'$\phi$')
  # ax.invert_yaxis()
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,2,2)
  im = ax.imshow(np.flipud(p.reshape(mz,mx)),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=pmin,vmax=pmax,cmap=cmaps)
  ax.set_title(r'$P/P_0$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,2,3)
  im = ax.imshow(np.flipud(vx.reshape(mz,mx+1)),extent=[min(xv), max(xv), min(zc), max(zc)],vmin=vxmin,vmax=vxmax,cmap=cmaps)
  ax.set_title(r'$v_s^x/v_0$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,2,4)
  im = ax.imshow(np.flipud(vz.reshape(mz+1,mx)),extent=[min(xc), max(xc), min(zv), max(zv)],vmin=vzmin,vmax=vzmax,cmap=cmaps)
  ax.set_title(r'$v_s^z/v_0$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,2,5)
  im = ax.imshow(np.flipud(vfx.reshape(mz,mx+1)),extent=[min(xv), max(xv), min(zc), max(zc)],vmin=vfxmin,vmax=vfxmax,cmap=cmaps)
  ax.set_title(r'$v_f^x/v_0$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,2,6)
  im = ax.imshow(np.flipud(vfz.reshape(mz+1,mx)),extent=[min(xc), max(xc), min(zv), max(zv)],vmin=vfzmin,vmax=vfzmax,cmap=cmaps)
  ax.set_title(r'$v_f^z/v_0$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  fout = fname+'_all_fields'+ft
  plt.savefig(fout+'.pdf')
  plt.close()

os.system('rm -r __pycache__')