# ---------------------------------------
# Uniform shear of a visco-elastic-plastic block subject to the Stokes model
# Setup from Keller et al. (2013)
# ---------------------------------------

# Import modules
import numpy as np
import matplotlib.pyplot as plt
import importlib
import os
import sys, getopt

# ---------------------------------------
# Function definitions
# ---------------------------------------

def plot_allinone(fname,ft,nx):

  # First row P, Vx, Vz, |V|, eta
  # Load data
  fout = fname+'_solution'+ft 
  spec = importlib.util.spec_from_file_location(fout,fname+'/'+fout+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod)
  data = imod._PETScBinaryLoad()
  # imod = importlib.import_module(fout) 
  # data = imod._PETScBinaryLoad()
  #imod._PETScBinaryLoadReportNames(data)

  mx = data['Nx'][0]
  mz = data['Ny'][0]
  xc = data['x1d_cell']
  zc = data['y1d_cell']
  xv = data['x1d_vertex']
  zv = data['y1d_vertex']
  vx = data['X_face_x']
  vz = data['X_face_y']
  p  = data['X_cell']

  fout = fname+'_coefficient'+ft
  spec = importlib.util.spec_from_file_location(fout,fname+'/'+fout+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod)
  data = imod._PETScBinaryLoad()
  # imod = importlib.import_module(fout) 
  # data = imod._PETScBinaryLoad()
  #imod._PETScBinaryLoadReportNames(data)

  etac_data = data['X_cell']
  etan = data['X_vertex']

  rhsx_data = data['X_face_x']
  rhsy_data = data['X_face_y']

  # split into dofs
  dof = 3
  etac = etac_data[1::dof]

  rhsx = rhsx_data[0::dof]
  rhsy = rhsy_data[0::dof]

  # reshaped rhs
  rx = rhsx.reshape(mz, mx+1)
  ry = rhsy.reshape(mz+1, mx)

  # Prepare cell center velocities and pressure
  vxface = vx.reshape(mz  ,mx+1)
  vzface = vz.reshape(mz+1,mx  )
  pp = p.reshape(mz,mx)

  # Compute the cell center values from the face data by averaging neighbouring faces
  vxc = np.zeros( [mz , mx] )
  vzc = np.zeros( [mz , mx] )
  vc  = np.zeros( [mz , mx] )
  divv= np.zeros( [mz , mx] )
  dx  = np.zeros( [mx] )
  dz  = np.zeros( [mz] )
  dpdx = np.zeros( [mz , mx+1] )
  dpdz = np.zeros( [mz+1 , mx] )
  
  for i in range(0,mx):
    dx[i] = xv[i+1] - xv[i]
  for j in range(0,mz):
    dz[j] = zv[j+1] - zv[j]
  
  for i in range(0,mx):
    for j in range(0,mz):
      vxc[j][i] = 0.5 * (vxface[j][i+1] + vxface[j][i])
      vzc[j][i] = 0.5 * (vzface[j+1][i] + vzface[j][i])
      vc [j][i] = (vxc[j][i]**2+vzc[j][i]**2)**0.5
      divv[j][i] = (vxface[j][i+1] - vxface[j][i])/dx[i] + (vzface[j+1][i] - vzface[j][i])/dz[j]

      if i != 0:
        dpdx[j][i+1] = pp[j][i] - pp[j][i-1]
      if j != 0:
        dpdz[j+1][i] = pp[j][i] - pp[j-1][i]

  dpdxc = np.zeros([mz,mx])
  dpdzc = np.zeros([mz,mx])

  for i in range(0,mx):
    for j in range(0,mz):
      dpdxc[j][i] = 0.5* (dpdx[j][i+1] + dpdx[j][i])
      dpdzc[j][i] = 0.5* (dpdz[j+1][i] + dpdz[j][i])

  velmax = max(max(vx),max(vz))
  velmin = min(min(vx),min(vz))

  etamax = 3
  etamin =-3

  nind = 10

  # Plot all fields - P, vx, vz, v, etac, etan
  fig = plt.figure(1,figsize=(15,9))
  cmaps='RdBu_r' 

  ax = plt.subplot(3,5,1)
  im = ax.imshow(p.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps,origin='lower')
  ax.set_title(r'$P_f$')
  ax.set_ylabel('z')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,5,2)
  im = ax.imshow(vx.reshape(mz,mx+1),extent=[min(xv), max(xv), min(zc), max(zc)],vmin=velmin,vmax=velmax,cmap=cmaps,origin='lower')
  ax.set_title(r'$V_x$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,5,3)
  im = ax.imshow(vz.reshape(mz+1,mx),extent=[min(xc), max(xc), min(zv), max(zv)],vmin=velmin,vmax=velmax,cmap=cmaps,origin='lower')
  ax.set_title(r'$V_z$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,5,4)
  im = ax.imshow(vc,extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps,origin='lower')
  ax.set_title(r'$V$ magnitude')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,5,5)
  im = ax.imshow(etac.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps,origin='lower')
  ax.set_title(r'$\eta_{center}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)


  #Second row: plot epsII, epsxx, epszz, epsxz 
    # Load data
  fout = fname+'_strain'+ft # 1. Numerical solution stokes - strain rates
  spec = importlib.util.spec_from_file_location(fout,fname+'/'+fout+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod)
  data = imod._PETScBinaryLoad()
  # imod = importlib.import_module(fout) 
  # data = imod._PETScBinaryLoad()
  #imod._PETScBinaryLoadReportNames(data)

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

  i1 = int(0.15/0.5 * mz)
  i2 = int(0.2/0.5 * mz)
  i3 = int(0.25/0.5 * mz)
  
  divmin = np.amin(divv[i1,:])
  divmax = np.amax(divv)
#  if divmin > -0.1:
#    divmin = -3


  ax = plt.subplot(3,5,6)
  im = ax.imshow(exx1c.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=exxmin,vmax=exxmax,cmap=cmaps,origin='lower')
  ax.set_title(r'$\epsilon_{xx}^{CENTER}$')
  ax.set_ylabel('z')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,5,7)
  im = ax.imshow(ezz1c.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=ezzmin,vmax=ezzmax,cmap=cmaps,origin='lower')
  ax.set_title(r'$\epsilon_{zz}^{CENTER}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,5,8)
  im = ax.imshow(exz1c.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=exzmin,vmax=exzmax,cmap=cmaps,origin='lower')
  ax.set_title(r'$\epsilon_{xz}^{CENTER}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,5,9)
  im = ax.imshow(eII1c.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=eIImin,vmax=eIImax,cmap=cmaps,origin='lower')
  ax.set_title(r'$\epsilon_{II}^{CENTER}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,5,10)
  im = ax.imshow(divv,extent=[min(xc), max(xc), min(zc), max(zc)],vmin = divmin, vmax = divmax, cmap=cmaps,origin='lower')
  ax.set_title(r'$\nabla \cdot \mathbf{V}_s$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  #Third row: Updated old value of tauII, tauxx, tauzz, tauxz
    # Load data
  fout = fname+'_stressold'+ft # 1. Numerical solution stokes - stress components
  spec = importlib.util.spec_from_file_location(fout,fname+'/'+fout+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod)
  data = imod._PETScBinaryLoad()
  # imod = importlib.import_module(fout) 
  # data = imod._PETScBinaryLoad()
  #imod._PETScBinaryLoadReportNames(data)

  mx = data['Nx'][0]
  mz = data['Ny'][0]
  xc = data['x1d_cell']
  zc = data['y1d_cell']
  xv = data['x1d_vertex']
  zv = data['y1d_vertex']
  s1n = data['X_vertex']
  s1c = data['X_cell']

  fout = fname+'_dpold'+ft # yield stress
  spec = importlib.util.spec_from_file_location(fout,fname+'/'+fout+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod)
  data = imod._PETScBinaryLoad()
  # imod = importlib.import_module(fout) 
  # data = imod._PETScBinaryLoad()
  #imod._PETScBinaryLoadReportNames(data)

  dpold_data = data['X_cell']
  

  # split into dofs
  dof = 4
  sxx1c = s1c[0::dof]
  szz1c = s1c[1::dof]
  sxz1c = s1c[2::dof]
  sII1c = s1c[3::dof]

  sxx1n = s1n[0::dof]
  szz1n = s1n[1::dof]
  sxz1n = s1n[2::dof]
  sII1n = s1n[3::dof]

  dpold = dpold_data[0::dof]

  sxxmax = max(max(sxx1c),max(sxx1n))
  sxxmin = min(min(sxx1c),min(sxx1n))
  szzmax = max(max(szz1c),max(szz1n))
  szzmin = min(min(szz1c),min(szz1n))
  sxzmax = max(max(sxz1c),max(sxz1n))
  sxzmin = min(min(sxz1c),min(sxz1n))
  sIImax = max(max(sII1c),max(sII1n))
  sIImin = min(min(sII1c),min(sII1n))

  dpmax = max(dpold)
  dpmin = min(dpold)

  stress = sxxmax

  ax = plt.subplot(3,5,11)
  im = ax.imshow(sxx1c.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=sxxmin,vmax=sxxmax,cmap=cmaps,origin='lower')
  ax.set_title(r'$\tilde{\tau}_{xx}^{o}$(Center)')
  ax.set_ylabel('z')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75, format='%.2f')

  ax = plt.subplot(3,5,12)
  im = ax.imshow(szz1c.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=szzmin,vmax=szzmax,cmap=cmaps,origin='lower')
  ax.set_title(r'$\tilde{\tau}_{zz}^{o}(Center)$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75, format='%.2f')

  ax = plt.subplot(3,5,13)
  im = ax.imshow(sxz1c.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=sxzmin,vmax=sxzmax,cmap=cmaps,origin='lower')
  ax.set_title(r'$\tilde{\tau}_{xz}^{o}$(Center)')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75, format='%.2f')

  ax = plt.subplot(3,5,14)
  im = ax.imshow(sII1c.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=sIImin,vmax=sIImax,cmap=cmaps,origin='lower')
  ax.set_title(r'$\tilde{\tau}_{II}^{o}$(Center)')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75, format='%.2f')

  ax = plt.subplot(3,5,15)
  im = ax.imshow(dpold.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=dpmin,vmax=dpmax,cmap=cmaps,origin='lower')
  ax.set_title(r'$\Delta P^{o}$(CENTER)')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  plt.tight_layout() 
  
  fout = fname+'/'+fname+'_allinone'+ft+'_nx_'+str(nx)

  plt.savefig(fout+'.pdf')
  plt.close()

  return stress

# ---------------------------------------
# Main script
# ---------------------------------------
print('# --------------------------------------- #')
print('# StokesDarcy-VEP, 0D test ')
print('# --------------------------------------- #')

# Set main parameters and run test
nx    = 5 # resolution

etamin = 0.0
C = 1e40
phi0 = 1e-4
R = 0.0 #0.01
lam = 0.0

G= 1.0
tstep = 200
tmax = 10.0
dt = 0.02

tout = 1


# filenames 
fname = 'out_stokesdarcy2field_vep_0d_shear'
fout = fname+'/'+fname+'_'+str(nx)+'.out'

try:
  os.mkdir(fname)
except OSError:
  pass

# Get cpu number
ncpu = 1
options, remainder = getopt.getopt(sys.argv[1:],'n:')
for opt, arg in options:
  if opt in ('-n'):
    ncpu = int(arg)


newton = ' -pc_factor_mat_solver_type umfpack'+ \
  ' -pc_type lu'+ \
  ' -snes_linesearch_type bt'+ \
  ' -snes_linesearch_monitor'+ \
  ' -snes_atol 1e-10'+ \
  ' -snes_rtol 1e-50'+ \
  ' -snes_stol 1e-10'+ \
  ' -snes_max_it 200'+ \
  ' -snes_monitor'+ \
  ' -snes_view'+ \
  ' -snes_monitor_true_residual'+ \
  ' -ksp_monitor_true_residual'+ \
  ' -snes_converged_reason'+ \
  ' -ksp_converged_reason'+ \
  ' -python_snes_failed_report'

solver= ' -snes_mf_operator'

sdpar = ' -R ' + str(R)+ \
  ' -phi_0 ' + str(phi0)+ \
  ' -lambda ' + str(lam)+ \
  ' -G ' + str(G) +\
  ' -tstep ' + str(tstep)+\
  ' -tmax ' + str(tmax)+\
  ' -dt ' + str(dt)

# Run simulation
str1 = 'mpiexec -n '+str(ncpu)+' ../test_stokesdarcy2field_vep_0d_shear.app' + newton + solver + sdpar + \
  ' -output_file '+fname+ \
  ' -C '+str(C)+ \
  ' -etamin '+str(etamin)+ \
  ' -output_dir '+fname+' -nx '+str(nx)+' -nz '+str(nx)+' > '+fout
print(str1)
os.system(str1)

# Parse log file
f = open(fout, 'r')
i0=0
for line in f:
  if '# TIMESTEP' in line:
      i0+=1
f.close()
tstep =i0

stress = np.zeros([tstep+1])
stress_ana = np.zeros([tstep+1])
tt = np.zeros([tstep+1])

for istep in range(0,tstep,tout):
  # Load python module describing data
  if (istep < 10): ft = '_ts00'+str(istep)
  if (istep >= 10) & (istep < 99): ft = '_ts0'+str(istep)
  if (istep >= 100): ft = '_ts'+str(istep)

  fout1 = fname
  stress[istep+1] = plot_allinone(fout1,ft,nx)
  tt[istep+1] = tt[istep] + dt
  stress_ana[istep+1] = -2*(1.0 - np.exp(-tt[istep+1]))
  print(istep, stress[istep+1])

# Run simulation #2
C = 0.2#1.0
etamin = 0.0
str2 = 'mpiexec -n '+str(ncpu)+' ../test_stokesdarcy2field_vep_0d_shear.app' + newton + solver + sdpar + \
  ' -output_file '+fname+ \
  ' -C '+str(C)+ \
  ' -etamin '+str(etamin)+ \
  ' -output_dir '+fname+' -nx '+str(nx)+' -nz '+str(nx)+' > '+fout
print(str2)
os.system(str2)

stress2 = np.zeros([tstep+1])
stress2_ana = np.zeros([tstep+1])
stress2_ana[0] = C

for istep in range(0,tstep,tout):
  # Load python module describing data
  if (istep < 10): ft = '_ts00'+str(istep)
  if (istep >= 10) & (istep < 99): ft = '_ts0'+str(istep)
  if (istep >= 100): ft = '_ts'+str(istep)

  fout1 = fname
  stress2[istep+1] = plot_allinone(fout1,ft,nx)
  stress2_ana[istep+1] = C
  print(istep, stress2[istep+1])


fig = plt.figure(1,figsize=(6,4))

ax = plt.subplot(1,1,1)
ax.plot( tt[::5], -stress[::5], 'ko-')
ax.plot( tt[::5], -stress2[::5], 'ks-')
ax.plot( tt, -stress_ana, 'r-')
ax.axhline(y=C, color = 'k', linestyle='--')
ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$\tau$')
ax.set_xlim([0,4])
ax.set_ylim([0,2.1])
plt.savefig(fname+'/'+'stress.pdf')
plt.close()

    
os.system('rm -r '+fname+'/__pycache__')
