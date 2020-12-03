# ---------------------------------------
# 2D diffusion to test the enthalpy implementation
# ---------------------------------------

# Import modules
import numpy as np
import matplotlib.pyplot as plt
import importlib
import os
from scipy.stats import linregress

# Input file
fname = 'out_2d_diff_enth'
fname_out = 'out_enthalpy_2d_diffusion'
fname_data = fname_out+'/data'
try:
  os.mkdir(fname_out)
except OSError:
  pass

try:
  os.mkdir(fname_data)
except OSError:
  pass

print('# --------------------------------------- #')
print('# 2D diffusion test (ENTHALPY) ')
print('# --------------------------------------- #')

tstep = 1
tout = 1
ni = [40, 50, 80, 100]#, 125, 150, 200]
ncomp = 3 # number of chemical components - warning: many of the plots below depend on this

# Run simulations
for n in ni:
  fout = fname+'_n'+str(n)
  solver = ' -snes_converged_reason -ksp_converged_reason -snes_monitor -ksp_monitor -snes_atol 1e-10 -snes_rtol 1e-20 -log_view'
  str1 = '../test_enthalpy_2d_diffusion.app -pc_type lu -pc_factor_mat_solver_type umfpack -snes_monitor -snes_max_it 200'+ \
      ' -output_file '+fout+' -output_dir '+fname_data+' -tstep '+str(tstep)+solver+' -ts_scheme 2'+' -nx '+str(n)+' -nz '+str(n) +\
      ' -ncomp '+str(ncomp)
  print(str1)
  os.system(str1)

# Plot solution (numerical and analytical) for highest res
fig, axs = plt.subplots(1,3,figsize=(12,4))

ax1 = plt.subplot(1,3,1)
ax2 = plt.subplot(1,3,2)
ax3 = plt.subplot(1,3,3)

# imod = importlib.import_module('out_xprev_initial')
fout = 'out_xprev_initial'
spec = importlib.util.spec_from_file_location(fout,fname_data+'/'+fout+'.py')
imod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(imod)
data_prev = imod._PETScBinaryLoad()
imod._PETScBinaryLoadReportNames(data_prev)

mx = data_prev['Nx'][0]
mz = data_prev['Ny'][0]
xc = data_prev['x1d_cell']
zc = data_prev['y1d_cell']

HC_prev_data = data_prev['X_cell']
H_prev = HC_prev_data[0::ncomp]
C1_prev = HC_prev_data[1::ncomp]
C2_prev = HC_prev_data[2::ncomp]

im = ax1.imshow( H_prev.reshape(mz,mx),origin='lower', extent=[min(xc), max(xc), min(zc), max(zc)],interpolation='nearest' )
ax1.axis('image')
ax1.set_xlabel('x')
ax1.set_ylabel('z')
ax1.set_title('Initial condition at t=0.05')

for istep in range(0,tstep,tout):
  # Load python module describing data
  if (istep < 10): ft = '_ts00'+str(istep)
  if (istep >= 10) & (istep < 99): ft = '_ts0'+str(istep)
  if (istep >= 100) & (istep < 999): ft = '_ts'+str(istep)

  fout = fname+'_n'+str(n)+'_analytical'+ft
  spec = importlib.util.spec_from_file_location(fout,fname_data+'/'+fout+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod)
  data_an = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(data_an)

  fout = fname+'_n'+str(n)+'_enthalpy'+ft
  spec = importlib.util.spec_from_file_location(fout,fname_data+'/'+fout+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod)
  data_enth = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(data_enth)

  fout = fname+'_n'+str(n)+ft
  spec = importlib.util.spec_from_file_location(fout,fname_data+'/'+fout+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod)
  data_num = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(data_num)

  HC_an_data = data_an['X_cell']
  HC_enth_data = data_enth['X_cell']
  HC_num_data = data_num['X_cell']

  H_an = HC_an_data[0::ncomp]
  C1_an = HC_an_data[1::ncomp]
  C2_an = HC_an_data[2::ncomp]

  H_num = HC_num_data[0::ncomp]
  C1_num = HC_num_data[1::ncomp]
  C2_num = HC_num_data[2::ncomp]

  dof_en = 5 + 3*(ncomp-1)
  H_enth = HC_enth_data[0::dof_en]
  C1_enth = HC_enth_data[5::dof_en]
  C2_enth = HC_enth_data[6::dof_en]

  H_an_res = H_an.reshape(mz,mx)
  H_num_res = H_num.reshape(mz,mx)
  H_enth_res = H_enth.reshape(mz,mx)

  H_an_i = H_an_res[int(mz/2),:]
  H_num_i = H_num_res[int(mz/2),:]
  H_enth_i = H_enth_res[int(mz/2),:]

  pl = ax2.plot(xc,H_an_i,'-',label='tstep='+str(istep))
  pl = ax2.plot(xc,H_num_i,'ko')
  pl = ax2.plot(xc,H_enth_i,'b*')

ax2.set_xlabel('x')
# ax2.set_ylabel('z')
ax2.legend(loc='upper left')
ax2.axis('auto')
ax2.set_title('Horizontal H profile at z = 0')
ax2.grid(True,color='gray', linestyle='--', linewidth=0.5)

# Compute error norms for last timestep
nrm_H = np.zeros(len(ni))
nrm_C1 = np.zeros(len(ni))
nrm_C2 = np.zeros(len(ni))
dx = np.zeros(len(ni))
for ii in range(0,len(ni)):
  n = ni[ii]
  dx[ii] = 1.0/n
  dv = 1.0/n/n
  ft = '_ts00'+str(0)#+str(tstep-1)
  fout = fname+'_n'+str(n)+'_analytical'+ft
  spec = importlib.util.spec_from_file_location(fout,fname_data+'/'+fout+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod)
  data_an = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(data_an)

  fout = fname+'_n'+str(n)+ft
  spec = importlib.util.spec_from_file_location(fout,fname_data+'/'+fout+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod)
  data_num = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(data_num)

  mx = data_an['Nx'][0]
  mz = data_an['Ny'][0]

  HC_an_data = data_an['X_cell']
  HC_num_data = data_num['X_cell']
  H_an = HC_an_data[0::ncomp]
  C1_an = HC_an_data[1::ncomp]
  C2_an = HC_an_data[2::ncomp]

  H_num = HC_num_data[0::ncomp]
  C1_num = HC_num_data[1::ncomp]
  C2_num = HC_num_data[2::ncomp]

  H_an_res = H_an.reshape(mz,mx)
  H_num_res = H_num.reshape(mz,mx)
  C1_an_res = C1_an.reshape(mz,mx)
  C2_an_res = C2_an.reshape(mz,mx)
  C1_num_res = C1_num.reshape(mz,mx)
  C2_num_res = C2_num.reshape(mz,mx)

  for i in range(0,mx):
    for j in range(0,mz):
      nrm_H[ii] += (H_an_res[j,i]-H_num_res[j,i])**2/H_an_res[j,i]**2*dv
      nrm_C1[ii] += (C1_an_res[j,i]-C1_num_res[j,i])**2/C1_an_res[j,i]**2*dv
      nrm_C2[ii] += (C2_an_res[j,i]-C2_num_res[j,i])**2/C2_an_res[j,i]**2*dv

nrm_H_log = np.log10(nrm_H)
nrm_C1_log = np.log10(nrm_C1)
nrm_C2_log = np.log10(nrm_C2)
dx_log = np.log10(dx)

# Perform linear regression
slH, intercept, r_value, p_value, std_err = linregress(dx_log, nrm_H_log)
slC1, intercept, r_value, p_value, std_err = linregress(dx_log, nrm_C1_log)
slC2, intercept, r_value, p_value, std_err = linregress(dx_log, nrm_C2_log)

pl = ax3.plot(dx_log,nrm_H_log,'ko-',label='H slope='+str(round(slH,5)))
pl = ax3.plot(dx_log,nrm_C1_log,'ro--',label='C1 slope='+str(round(slC1,5)))
pl = ax3.plot(dx_log,nrm_C2_log,'b*--',label='C2 slope='+str(round(slC2,5)))
ax3.set_xlabel('log10(dx)')
ax3.set_title('log10||Qan-Qnum||2')
ax3.legend(loc='upper left')
ax3.axis('auto')
ax3.grid(True,color='gray', linestyle='--', linewidth=0.5)

plt.savefig(fname_out+'/'+fname+'.pdf')

os.system('rm -r '+fname_data+'/__pycache__')