# ---------------------------------------
# 1D solidification problem of an  initially  liquid  semi-infinite  slab
# ---------------------------------------

# Import modules
import numpy as np
import matplotlib.pyplot as plt
import importlib
import os

# Input file
fname = 'out_1d_sol'
fname_out = 'out_enthalpy_1d_eutectic'
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
print('# 1-D Eutectic Solidification test (ENTHALPY) ')
print('# --------------------------------------- #')

tstep = 51
tout = 10
S = 5
C = 5
nz = 201

# Run test
solver = ' -snes_converged_reason -ksp_converged_reason -snes_monitor -ksp_monitor -snes_atol 1e-10 -snes_rtol 1e-20'
str1 = '../test_enthalpy_1d_eutectic_solidification_parkinson.app -pc_type lu -pc_factor_mat_solver_type umfpack -snes_monitor -snes_max_it 200'+ \
    ' -output_file '+fname+' -output_dir '+fname_data+' -tstep '+str(tstep)+solver+' -ts_scheme 1 -S '+str(S)+' -C '+str(C)+' -nz '+str(nz)
print(str1)
os.system(str1)

# Plot solution (numerical and analytical)
fig, axs = plt.subplots(1,3,figsize=(12,4))

# temperature distribution in the slab - last timestep
ax1 = plt.subplot(1,3,1)
ax2 = plt.subplot(1,3,2)
ax3 = plt.subplot(1,3,3)

for istep in range(0,tstep,tout):
  # Load python module describing data
  if (istep < 10): ft = '_ts00'+str(istep)
  if (istep >= 10) & (istep < 99): ft = '_ts0'+str(istep)
  if (istep >= 100) & (istep < 999): ft = '_ts'+str(istep)

  # imod = importlib.import_module(fname+'_HC'+ft)
  fout = fname+'_HC'+ft
  spec = importlib.util.spec_from_file_location(fout,fname_data+'/'+fout+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod)
  data0 = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(data0)

  # imod = importlib.import_module('out_analytic_solution_TC')
  fout = 'out_analytic_solution_TC'
  spec = importlib.util.spec_from_file_location(fout,fname_data+'/'+fout+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod)
  data_an = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(data_an)

  # imod = importlib.import_module('out_xprev_initial')
  fout = 'out_xprev_initial'
  spec = importlib.util.spec_from_file_location(fout,fname_data+'/'+fout+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod)
  data_prev = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(data_prev)

  # imod = importlib.import_module(fname+'_enthalpy'+ft)
  fout = fname+'_enthalpy'+ft
  spec = importlib.util.spec_from_file_location(fout,fname_data+'/'+fout+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod)
  data_enthalpy = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(data_enthalpy)

  # coordinates
  mx = data0['Nx'][0]
  mz = data0['Ny'][0]
  xc = data0['x1d_cell']
  zc = data0['y1d_cell']

  HC_data = data0['X_cell']
  HC_prev_data = data_prev['X_cell']
  Tan_data = data_an['X_cell']
  enth_data = data_enthalpy['X_cell']

  # extract temperature data
  Th_an  = Tan_data[0::2]
  phi_an = Tan_data[1::2]
  H_an   = Th_an+phi_an*S

  H_prev = HC_prev_data[0::2]
  H_num  = HC_data[0::2]

  dof_en = 8
  H_num_en = enth_data[0::dof_en]
  T_num_en = enth_data[1::dof_en]
  TP_num_en = enth_data[2::dof_en]
  phi_num_en = enth_data[3::dof_en]
  Th_an_res = Th_an.reshape(mz,mx)
  phi_an_res = phi_an.reshape(mz,mx)
  H_an_res = H_an.reshape(mz,mx)
  H_prev_res = H_prev.reshape(mz,mx)
  H_num_res = H_num.reshape(mz,mx)
  H_num_en_res = H_num_en.reshape(mz,mx)
  T_num_en_res = T_num_en.reshape(mz,mx)
  TP_num_en_res = TP_num_en.reshape(mz,mx)
  phi_num_en_res = phi_num_en.reshape(mz,mx)

  Th_an_i = Th_an_res[:,int(mx/2)]
  phi_an_i = phi_an_res[:,int(mx/2)]
  H_an_i = H_an_res[:,int(mx/2)]
  H_prev_i = H_prev_res[:,int(mx/2)]
  H_num_i = H_num_res[:,int(mx/2)]
  H_num_en_i = H_num_en_res[:,int(mx/2)]
  T_num_en_i = T_num_en_res[:,int(mx/2)]
  TP_num_en_i = TP_num_en_res[:,int(mx/2)]
  phi_num_en_i = phi_num_en_res[:,int(mx/2)]

  pl = ax1.plot(H_num_i,zc,'-',label='tstep='+str(istep))
  pl = ax2.plot(T_num_en_i,zc,'-',label='tstep='+str(istep))
  pl = ax3.plot(phi_num_en_i,zc,'-',label='tstep='+str(istep))

pl = ax1.plot(H_an_i,zc,'k-',label='analytical')
pl = ax1.plot(H_prev_i,zc,'b--',label='initial')
ax1.set_title('a) Enthalpy')
ax1.set_xlabel('Enthalpy [-]')
ax1.set_ylabel('z [-]')
ax1.legend(loc='lower left')
ax1.axis('auto')
ax1.grid(True,color='gray', linestyle='--', linewidth=0.5)

pl = ax2.plot(Th_an_i,zc,'k-',label='analytical')
ax2.set_title('b) Temperature')
ax2.set_xlabel('Temperature [-]')
# ax2.set_ylabel('z [-]')
ax2.legend(loc='lower left')
ax2.axis('auto')
ax2.grid(True,color='gray', linestyle='--', linewidth=0.5)

pl = ax3.plot(phi_an_i,zc,color='k',label='analytical')
ax3.set_title('c) Porosity')
ax3.set_xlabel('phi [-]')
# ax3.set_ylabel('z [-]')
ax3.legend(loc='lower left')
ax3.axis('auto')
ax3.grid(True,color='gray', linestyle='--', linewidth=0.5)

# print('Coordinates')
# print(zc)
# print('Analytical H')
# print(H_an_i)
# print('Analytical T')
# print(Th_an_i)
# print('Analytical phi')
# print(phi_an_i)
# print('Numerical H')
# print(H_num_i)
# print('Numerical T')
# print(T_num_en_i)
# print('Numerical phi')
# print(phi_num_en_i)

plt.savefig(fname_out+'/'+fname+'.pdf')

os.system('rm -r '+fname_data+'/__pycache__')