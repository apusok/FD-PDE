# ------------------------------------------------ #
# MMS sympy for Katz - Magma dynamics - Ch 13
# ------------------------------------------------ #

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os
import importlib
import dmstagoutput as dmout
from matplotlib import rc
import sys, getopt

# Some new font
rc('font',**{'family':'serif','serif':['Times new roman']})
rc('text', usetex=True)

# Input file
f1 = 'out_mms_katz_ch13'
fname = 'out_stokesdarcy_mms_katz_ch13'
fname_data = fname+'/data'
try:
  os.mkdir(fname)
except OSError:
  pass

try:
  os.mkdir(fname_data)
except OSError:
  pass

# Get cpu number, set with option -n 2
ncpu = 1
options, remainder = getopt.getopt(sys.argv[1:],'n:')
for opt, arg in options:
  if opt in ('-n'):
    ncpu = int(arg)

# Use umfpack for sequential and mumps for parallel
solver_default = ' -snes_monitor -snes_converged_reason -ksp_monitor -ksp_converged_reason '
if (ncpu == 1):
  solver = ' -pc_type lu -pc_factor_mat_solver_type umfpack -pc_factor_mat_ordering_type external'
else:
  solver = ' -pc_type lu -pc_factor_mat_solver_type mumps'

# Parameters
n = [20, 40, 80, 100, 200, 300, 400]

delta = 1.0 # delta^2 = eta0*K0/H/H
phi_0 = 0.1 # phi_0 = phi_s
phi_s = 0.1 
p_s   = 1.0
psi_s = 1.0
U_s   = 1.0
m     = 2.0
nexp  = 3.0
e3    = 0.0 # unit vector in the z-dir (remove if ignore buoyancy)

cmaps='RdBu_r' 

# Run simulations
for nx in n:
  # Create output filename
  fout1 = fname_data+'/'+f1+'_'+str(nx)+'.out'

  # Run with different resolutions
  str1 = 'mpiexec -n '+str(ncpu)+' ../test_stokesdarcy2field_mms_katz_ch13.app '+solver+solver_default+' -delta '+str(delta)+ \
    ' -phi_0 '+str(phi_0)+' -phi_s '+str(phi_s)+' -p_s '+str(p_s)+' -psi_s '+str(psi_s)+ \
    ' -U_s '+str(U_s)+' -m '+str(m)+' -n '+str(nexp)+' -e3 '+str(e3)+ \
    ' -output_dir '+fname_data+' -nx '+str(nx)+' -nz '+str(nx)+' > '+fout1
  print(str1)
  os.system(str1)

  # Get mms solution data
  f1out = 'out_mms_solution'
  # imod = importlib.import_module(f1out)
  spec = importlib.util.spec_from_file_location(f1out,fname_data+'/'+f1out+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod)

  data = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(data)

  mx = data['Nx'][0]
  mz = data['Ny'][0]
  uxmms = data['X_face_x']
  uzmms= data['X_face_y']
  pmms = data['X_cell']

  # Get solution data
  f1out = 'out_solution'
  # imod = importlib.import_module(f1out)
  spec = importlib.util.spec_from_file_location(f1out,fname_data+'/'+f1out+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod)

  data = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(data)

  x = data['x1d_vertex']
  z = data['y1d_vertex']
  xc = data['x1d_cell']
  zc = data['y1d_cell']
  ux = data['X_face_x']
  uz= data['X_face_y']
  p = data['X_cell']

  # Get extra parameters
  f1out = 'out_extra_parameters'
  # imod = importlib.import_module(f1out)
  spec = importlib.util.spec_from_file_location(f1out,fname_data+'/'+f1out+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod)

  data = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(data)

  # xface = data['X_face_x']
  # yface = data['X_face_y']
  elem  = data['X_cell']
  dof0 = 7 # element

  psi = elem[0::dof0]
  U   = elem[1::dof0]
  phi = elem[2::dof0]
  curl_psix = elem[3::dof0]
  gradUx    = elem[4::dof0]
  curl_psiz = elem[5::dof0]
  gradUz    = elem[6::dof0]

  # Compute center velocities
  ux_sq = ux.reshape(mz  ,mx+1)
  uz_sq = uz.reshape(mz+1,mx  )

  curl_psix = curl_psix.reshape(mz,mx)
  curl_psiz = curl_psiz.reshape(mz,mx)
  gradUx    = gradUx.reshape(mz,mx)
  gradUz    = gradUz.reshape(mz,mx)

  # Compute the cell center values from the face data by averaging neighbouring faces
  uxc = np.zeros([mz,mx])
  uzc = np.zeros([mz,mx])
  for i in range(0,mx):
    for j in range(0,mz):
      uxc[j][i] = 0.5 * (ux_sq[j][i+1] + ux_sq[j][i])
      uzc[j][i] = 0.5 * (uz_sq[j+1][i] + uz_sq[j][i])

  # Plot data - mms, solution and errors for P, ux, uz
  fig = plt.figure(1,figsize=(12,12))

  ax = plt.subplot(331)
  im = ax.imshow(pmms.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps)
  ax.set_title('a) MMS P')
  ax.set_ylabel('z')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(332)
  im = ax.imshow(p.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps)
  ax.set_title('b) Numerical P')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(333)
  im = ax.imshow(pmms.reshape(mz,mx)-p.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps)
  ax.set_title('c) Error P')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(334)
  im = ax.imshow(uxmms.reshape(mz,mx+1),extent=[min(x), max(x), min(zc), max(zc)],cmap=cmaps)
  ax.set_title('d) MMS ux')
  ax.set_ylabel('z')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(335)
  im = ax.imshow(ux.reshape(mz,mx+1),extent=[min(x), max(x), min(zc), max(zc)],cmap=cmaps)
  ax.set_title('e) Numerical ux')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(336)
  im = ax.imshow(uxmms.reshape(mz,mx+1)-ux.reshape(mz,mx+1),extent=[min(x), max(x), min(zc), max(zc)],cmap=cmaps)
  ax.set_title('f) Error ux')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(337)
  im = ax.imshow(uzmms.reshape(mz+1,mx),extent=[min(xc), max(xc), min(z), max(z)],cmap=cmaps)
  ax.set_title('g) MMS uz')
  ax.set_xlabel('x')
  ax.set_ylabel('z')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(338)
  im = ax.imshow(uz.reshape(mz+1,mx),extent=[min(xc), max(xc), min(z), max(z)],cmap=cmaps)
  ax.set_title('h) Numerical uz')
  ax.set_xlabel('x')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(339)
  im = ax.imshow(uzmms.reshape(mz+1,mx)-uz.reshape(mz+1,mx),extent=[min(xc), max(xc), min(z), max(z)],cmap=cmaps)
  ax.set_title('i) Error uz')
  ax.set_xlabel('x')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  plt.tight_layout() 
  plt.savefig(fname+'/'+f1+'_nx_'+str(nx)+'.pdf')
  plt.close()

  # Plot data - mms, solution and errors for P, ux, uz
  fig = plt.figure(1,figsize=(12,4))
  nind = int(nx/20)

  ax = plt.subplot(131)
  im = ax.imshow(psi.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps)
  Q  = ax.quiver( xc[::nind], zc[::nind], curl_psix[::nind,::nind], curl_psiz[::nind,::nind], units='width', pivot='mid' )
  ax.set_title('Scalar: '+r'$\psi$'+', Vector: '+r'$\nabla\times\phi\mathbf{\hat{k}}$')
  ax.set_xlabel('x')
  ax.set_ylabel('z')
  cbar = fig.colorbar(im,ax=ax, ticks=np.linspace(np.around(np.min(psi),decimals=1), np.around(np.max(psi),decimals=1), 5), shrink=0.75, extend='both')

  ax = plt.subplot(132)
  im = ax.imshow(U.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps)
  Q  = ax.quiver( xc[::nind], zc[::nind], gradUx[::nind,::nind], gradUz[::nind,::nind], units='width', pivot='mid' )
  ax.set_title('Scalar: '+r'$U$'+', Vector: '+r'$\nabla U$')
  ax.set_xlabel('x')
  cbar = fig.colorbar(im,ax=ax, ticks=np.linspace(np.around(np.min(U), decimals=1), np.around(np.max(U), decimals=1), 5), shrink=0.75, extend='both')

  ax = plt.subplot(133)
  im = ax.imshow(phi.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps)
  Q  = ax.quiver( xc[::nind], zc[::nind], uxc[::nind,::nind], uzc[::nind,::nind], units='width', pivot='mid' )
  ax.set_title('Scalar: '+r'$\phi$'+', Vector: '+r'$\mathbf{v_s}$')
  ax.set_xlabel('x')
  cbar = fig.colorbar(im,ax=ax, ticks=np.linspace(np.around(np.min(phi), decimals=2), np.around(np.max(phi), decimals=2), 5), shrink=0.75, extend='both')

  plt.savefig(fname+'/'+f1+'_extra_nx_'+str(nx)+'.pdf')
  plt.close()

# Norm variables
nrm_v = np.zeros(len(n))
nrm_vx = np.zeros(len(n))
nrm_vz = np.zeros(len(n))
nrm_p = np.zeros(len(n))
hx = np.zeros(len(n))

# Parse output and save norm info
for i in range(0,len(n)):
    nx = n[i]

    fout1 = fname_data+'/'+f1+'_'+str(nx)+'.out'

    # Open file 1 and read
    f = open(fout1, 'r')
    for line in f:
      if 'Velocity:' in line:
          nrm_v[i] = float(line[20:38])
          nrm_vx[i] = float(line[48:66])
          nrm_vz[i] = float(line[76:94])
      if 'Pressure:' in line:
          nrm_p[i] = float(line[20:38])
      if 'Grid info:' in line:
          hx[i] = float(line[18:36])
    f.close()

# Print convergence orders:
hx_log    = np.log10(n)
nrm2v_log = np.log10(nrm_v)
nrm2p_log = np.log10(nrm_p)

# Perform linear regression
sl2v, intercept, r_value, p_value, std_err = linregress(hx_log, nrm2v_log)
sl2p, intercept, r_value, p_value, std_err = linregress(hx_log, nrm2p_log)

print('# --------------------------------------- #')
print('# MMS StokesDarcy2Field (Katz, Magma dynamics, Ch. 13) convergence order:')
print('     v_slope = '+str(sl2v)+' p_slope = '+str(sl2p))

# Plot convergence data
plt.figure(1,figsize=(6,6))

plt.grid(color='lightgray', linestyle=':')

plt.plot(n,nrm_v,'k+-',label='v sl = '+str(round(sl2v,5)))
plt.plot(n,nrm_p,'ko--',label='P sl = '+str(round(sl2p,5)))

plt.xscale("log")
plt.yscale("log")

plt.xlabel('$\sqrt{N}, N=n^2$',fontweight='bold',fontsize=12)
plt.ylabel('$E(P), E(v_s)$',fontweight='bold',fontsize=12)
plt.legend()

plt.savefig(fname+'/'+f1+'.pdf')
plt.close()

os.system('rm -r '+fname_data+'/__pycache__')

