# ------------------------------------------------ #
# MMS test for 2-Field vs 3-Field Stokes-Darcy (2-phase flow)
# ------------------------------------------------ #

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os
import sys, getopt
import importlib
import dmstagoutput as dmout
from matplotlib import rc

# Some new font
rc('font',**{'family':'serif','serif':['Times new roman']})
rc('text', usetex=True)

# Get cpu number
ncpu = 1
options, remainder = getopt.getopt(sys.argv[1:],'n:')
for opt, arg in options:
  if opt in ('-n'):
    ncpu = int(arg)

# Input file
fname = 'out_stokesdarcy3_mms_bulkviscosity'
fname_data = fname+'/data'
try:
  os.mkdir(fname)
except OSError:
  pass

try:
  os.mkdir(fname_data)
except OSError:
  pass

# Parameters
n = [20, 40, 80, 100]#, 200, 300, 400]

delta = 1.0 # 1-10
vzeta = 10.0 # 1e1-1e3
phi0  = 0.1 
phia  = 0.1 # [0.1, 0.5, 0.75, 0.95]
phi_min= 1.0e-6
p_s   = 1.0
psi_s = 1.0
U_s   = 1.0
m     = 2.0
nexp  = 3.0
k_hat = 0.0 # unit vector in the z-dir (remove if ignore buoyancy)

cmaps='RdBu_r' 

# Use umfpack for sequential and mumps for sequential/parallel
solver_default = ' -snes_monitor -snes_converged_reason -ksp_monitor -ksp_converged_reason '
if (ncpu == -1):
  solver = ' -pc_type lu -pc_factor_mat_solver_type umfpack -pc_factor_mat_ordering_type external'
  ncpu = 1
else:
  solver = ' -pc_type lu -pc_factor_mat_solver_type mumps'

# Run simulations
for nx in n:
  # Create output filename
  fout1 = fname_data+'/'+fname+'_'+str(nx)+'.out'

  # Run with different resolutions
  str1 = 'mpiexec -n '+str(ncpu)+' ../test_stokesdarcy3field_mms_bulkviscosity.app -delta '+str(delta)+ \
    ' -phi0 '+str(phi0)+' -phia '+str(phia)+' -phi_min '+str(phi_min)+' -p_s '+str(p_s)+' -psi_s '+str(psi_s)+ solver+solver_default +\
    ' -vzeta '+str(vzeta)+' -U_s '+str(U_s)+' -m '+str(m)+' -n '+str(nexp)+' -k_hat '+str(k_hat)+ \
    ' -output_dir '+fname_data+' -output_file '+fname+'_'+str(nx)+' -nx '+str(nx)+' -nz '+str(nx)+' > '+fout1
  print(str1)
  os.system(str1)

  # Get 2-field solution
  f1out = fname+'_'+str(nx)+'_sd2field_sol'
  spec = importlib.util.spec_from_file_location(f1out,fname_data+'/'+f1out+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod)
  data2 = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(data2)

  mx = data2['Nx'][0]
  mz = data2['Ny'][0]
  xv = data2['x1d_vertex']
  zv = data2['y1d_vertex']
  xc = data2['x1d_cell']
  zc = data2['y1d_cell']

  ux2 = data2['X_face_x']
  uz2= data2['X_face_y']
  p2 = data2['X_cell']

  # Get 3-field solution
  f1out = fname+'_'+str(nx)+'_sd3field_sol'
  spec = importlib.util.spec_from_file_location(f1out,fname_data+'/'+f1out+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod)
  data3 = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(data3)

  ux3 = data3['X_face_x']
  uz3 = data3['X_face_y']
  p   = data3['X_cell']
  p3  = p[0::2]
  pc3 = p[1::2]

  # Get mms solution
  f1out = fname+'_'+str(nx)+'_sd3field_mms'
  spec = importlib.util.spec_from_file_location(f1out,fname_data+'/'+f1out+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod)
  data3mms = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(data3mms)

  ux_mms = data3mms['X_face_x']
  uz_mms = data3mms['X_face_y']
  pmms   = data3mms['X_cell']
  p_mms  = pmms[0::2]
  pc_mms = pmms[1::2]

  # Get extra parameters
  f1out = fname+'_'+str(nx)+'_extra_parameters'
  spec = importlib.util.spec_from_file_location(f1out,fname_data+'/'+f1out+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod)
  data = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(data)

  ex   = data['X_cell']
  dof = 5
  phi  = ex[0::dof]
  K    = ex[1::dof]
  zeta = ex[2::dof]
  eta  = ex[3::dof]
  xi   = ex[4::dof]

  # Plot data - mms, solution and errors for P, ux, uz
  fig = plt.figure(1,figsize=(16,16))

  extentc =[min(xc), max(xc), min(zc), max(zc)]
  extentx =[min(xv), max(xv), min(zc), max(zc)]
  extentz =[min(xc), max(xc), min(zv), max(zv)]

  # first row
  ax = plt.subplot(4,4,1)
  im = ax.imshow(phi.reshape(mz,mx),extent=extentc,cmap=cmaps)
  ax.set_title(r'MMS $\phi$')
  ax.set_ylabel('z')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(4,4,2)
  im = ax.imshow(K.reshape(mz,mx),extent=extentc,cmap=cmaps)
  ax.set_title(r'$K$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(4,4,3)
  im = ax.imshow(zeta.reshape(mz,mx),extent=extentc,cmap=cmaps)
  ax.set_title(r'$\zeta$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(4,4,4)
  im = ax.imshow(xi.reshape(mz,mx),extent=extentc,cmap=cmaps)
  ax.set_title(r'$\xi$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  # second row
  ax = plt.subplot(4,4,5)
  im = ax.imshow(ux_mms.reshape(mz,mx+1),extent=extentx,cmap=cmaps)
  ax.set_title(r'MMS $u_x$')
  ax.set_ylabel('z')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(4,4,6)
  im = ax.imshow(uz_mms.reshape(mz+1,mx),extent=extentz,cmap=cmaps)
  ax.set_title(r'$u_z$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(4,4,7)
  im = ax.imshow(p_mms.reshape(mz,mx),extent=extentc,cmap=cmaps)
  ax.set_title(r'$P$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(4,4,8)
  im = ax.imshow(pc_mms.reshape(mz,mx),extent=extentc,cmap=cmaps)
  ax.set_title(r'$P_c$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  # third row
  ax = plt.subplot(4,4,9)
  im = ax.imshow(ux2.reshape(mz,mx+1),extent=extentx,cmap=cmaps)
  ax.set_title(r'2-Field $u_x$')
  ax.set_ylabel('z')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(4,4,10)
  im = ax.imshow(uz2.reshape(mz+1,mx),extent=extentz,cmap=cmaps)
  ax.set_title(r'$u_z$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(4,4,11)
  im = ax.imshow(p2.reshape(mz,mx),extent=extentc,cmap=cmaps)
  ax.set_title(r'$P$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  # third row
  ax = plt.subplot(4,4,13)
  im = ax.imshow(ux3.reshape(mz,mx+1),extent=extentx,cmap=cmaps)
  ax.set_title(r'3-Field $u_x$')
  ax.set_ylabel('z')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(4,4,14)
  im = ax.imshow(uz3.reshape(mz+1,mx),extent=extentz,cmap=cmaps)
  ax.set_title(r'$u_z$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(4,4,15)
  im = ax.imshow(p3.reshape(mz,mx),extent=extentc,cmap=cmaps)
  ax.set_title(r'$P$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(4,4,16)
  im = ax.imshow(pc3.reshape(mz,mx),extent=extentc,cmap=cmaps)
  ax.set_title(r'$P_c$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  plt.tight_layout() 
  plt.savefig(fname+'/'+fname+'_nx_'+str(nx)+'.pdf')
  plt.close()

# Norm variables
nrm_v2  = np.zeros(len(n))
nrm_p2  = np.zeros(len(n))
nrm_v3  = np.zeros(len(n))
nrm_p3  = np.zeros(len(n))
nrm_pc3 = np.zeros(len(n))
hx = np.zeros(len(n))

# Parse output and save norm info
for i in range(0,len(n)):
    nx = n[i]

    fout1 = fname_data+'/'+fname+'_'+str(nx)+'.out'

    # Open file 1 and read
    f = open(fout1, 'r')
    for line in f:
      if 'NORMS 2-FIELD' in line:
        id = 2
      if 'NORMS 3-FIELD' in line:
        id = 3
      if '# Velocity:' in line:
        if (id==2):
          nrm_v2[i] = float(line[20:38])
        if (id==3):
          nrm_v3[i] = float(line[20:38])
      if '# Pressure:' in line:
        if (id==2):
          nrm_p2[i] = float(line[20:38])
        if (id==3):
          nrm_p3[i] = float(line[20:38])
      if '# Compaction Pressure:' in line:
        nrm_pc3[i] = float(line[31:49])
      if '# Grid info:' in line:
          hx[i] = float(line[18:36])
    f.close()

print(nrm_pc3)

# Print convergence orders:
hx_log    = np.log10(n)
nrm2v_log = np.log10(nrm_v2)
nrm2p_log = np.log10(nrm_p2)

nrm3v_log = np.log10(nrm_v3)
nrm3p_log = np.log10(nrm_p3)
nrm3pc_log= np.log10(nrm_pc3)

# Perform linear regression
sl2v, intercept, r_value, p_value, std_err = linregress(hx_log, nrm2v_log)
sl2p, intercept, r_value, p_value, std_err = linregress(hx_log, nrm2p_log)

sl3v, intercept, r_value, p_value, std_err = linregress(hx_log, nrm3v_log)
sl3p, intercept, r_value, p_value, std_err = linregress(hx_log, nrm3p_log)
sl3pc,intercept, r_value, p_value, std_err = linregress(hx_log, nrm3pc_log)

print('# --------------------------------------- #')
print('# MMS StokesDarcy 2-Field vs 3-Field convergence order:')
print('  2-Field:   v_slope = '+str(sl2v)+' p_slope = '+str(sl2p))
print('  3-Field:   v_slope = '+str(sl3v)+' p_slope = '+str(sl3p)+' pc_slope = '+str(sl3pc))

# Plot convergence data
plt.figure(1,figsize=(6,6))

plt.grid(color='lightgray', linestyle=':')

plt.plot(n,nrm_v2,'k+-',label='2F: v sl = '+str(round(sl2v,5)))
plt.plot(n,nrm_p2,'ko--',label='2F: P sl = '+str(round(sl2p,5)))

plt.plot(n,nrm_v3,'b+-',label='3F: v sl = '+str(round(sl3v,5)))
plt.plot(n,nrm_p3,'bo--',label='3F: P sl = '+str(round(sl3p,5)))
plt.plot(n,nrm_pc3,'r*--',label='3F: Pc sl = '+str(round(sl3pc,5)))

plt.xscale("log")
plt.yscale("log")

plt.xlabel('$\sqrt{N}, N=n^2$',fontweight='bold',fontsize=12)
plt.ylabel('$E(P), E(P_c), E(v_s)$',fontweight='bold',fontsize=12)
plt.legend()

plt.savefig(fname+'/'+fname+'.pdf')
plt.close()

os.system('rm -r '+fname_data+'/__pycache__')