# ---------------------------------------
# 1D solidification problem of an  initially  liquid  semi-infinite  slab
# ---------------------------------------

# Import modules
import numpy as np
import matplotlib.pyplot as plt
import importlib
import os

# ---------------------------------------
# Function definitions
# ---------------------------------------
def parse_log_file(fname):
  tstep = 0
  try: # try to open directory
    # parse number of timesteps
    f = open(fname, 'r')
    i0=0
    for line in f:
      if '# TIMESTEP' in line:
        i0+=1
    f.close()
    tstep = i0

    # time
    max_diff = np.zeros(tstep)

    f = open(fname, 'r')
    i0=-1
    for line in f:
      if '# TIMESTEP' in line:
        i0+=1
      if 'Steady-state check:' in line:
        max_diff[i0] = float(line[79:92])
    f.close()

    return tstep, max_diff
  except OSError:
    print('Cannot open:', fname)
    return tstep

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

tstep = 10001 # maximum tstep
tout = 100 # 200 for CFL = 0.1
S = 5
Cc = 5
nz = 41
C0 = -1
v = 1.0
CFL = 0.2

# Run test
solver = ' -snes_converged_reason -ksp_converged_reason -snes_monitor -ksp_monitor -snes_atol 1e-10 -snes_rtol 1e-20 -fp_trap -stop_enthalpy_failed'
# str1 = '../test_enthalpy_1d_eutectic_solidification.app -pc_type lu -pc_factor_mat_solver_type umfpack -pc_factor_mat_ordering_type external -snes_monitor -snes_max_it 200'+ \
str1 = 'mpiexec -n 1 ../test_enthalpy_1d_eutectic_solidification.app -pc_type lu -pc_factor_mat_solver_type mumps -snes_monitor -snes_max_it 200'+ \
    ' -output_file '+fname+ \
    ' -output_dir '+fname_data+ \
    ' -tstep '+str(tstep)+solver+ \
    ' -ts_scheme 2 -S '+str(S)+ \
    ' -CFL '+str(CFL)+ \
    ' -Cc '+str(Cc)+' -nz '+str(nz)+' -v '+str(v)+ ' > '+fname_data+'/log_'+fname+'.out'
print(str1)
os.system(str1)

# parse log file
tstep, max_diff = parse_log_file(fname_data+'/log_'+fname+'.out')

# Plot solution (numerical and analytical)
fig, axs = plt.subplots(1,4,figsize=(16,4))

ax1 = plt.subplot(1,4,1)
ax2 = plt.subplot(1,4,2)
ax3 = plt.subplot(1,4,3)
ax4 = plt.subplot(1,4,4)

# Load analytical solutions
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

# imod = importlib.import_module('out_xprev_initial')
fout = 'out_xanalytic_enthalpy_initial'
spec = importlib.util.spec_from_file_location(fout,fname_data+'/'+fout+'.py')
imod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(imod)
data_en_prev = imod._PETScBinaryLoad()
imod._PETScBinaryLoadReportNames(data_en_prev)

# coordinates
mx = data_an['Nx'][0]
mz = data_an['Ny'][0]
xc = data_an['x1d_cell']
zc = data_an['y1d_cell']
dof_en = 5+3*2

HC_prev_data = data_prev['X_cell']
enth_prev_data = data_en_prev['X_cell']
Tan_data = data_an['X_cell']

# extract temperature data
Th_an  = Tan_data[0::2]
phi_an = Tan_data[1::2]
H_an   = Th_an+phi_an*S

H_prev = HC_prev_data[0::2]
H_num_en_prev = enth_prev_data[0::dof_en]
T_num_en_prev = enth_prev_data[1::dof_en]
TP_num_en_prev = enth_prev_data[2::dof_en]
phi_num_en_prev = enth_prev_data[3::dof_en]
C_num_en_prev = enth_prev_data[5::dof_en]
Cs_num_en_prev = enth_prev_data[7::dof_en]
Cf_num_en_prev = enth_prev_data[9::dof_en]

Th_an_res = Th_an.reshape(mz,mx)
phi_an_res = phi_an.reshape(mz,mx)
H_an_res = H_an.reshape(mz,mx)
H_prev_res = H_prev.reshape(mz,mx)

H_num_en_prev_res = H_num_en_prev.reshape(mz,mx)
T_num_en_prev_res = T_num_en_prev.reshape(mz,mx)
TP_num_en_prev_res = TP_num_en_prev.reshape(mz,mx)
phi_num_en_prev_res = phi_num_en_prev.reshape(mz,mx)
C_num_en_prev_res = C_num_en_prev.reshape(mz,mx)
Cf_num_en_prev_res = Cf_num_en_prev.reshape(mz,mx)
Cs_num_en_prev_res = Cs_num_en_prev.reshape(mz,mx)

Th_an_i = Th_an_res[:,int(mx/2)]
phi_an_i = phi_an_res[:,int(mx/2)]
H_an_i = H_an_res[:,int(mx/2)]
H_prev_i = H_prev_res[:,int(mx/2)]

H_num_en_prev_i = H_num_en_prev_res[:,int(mx/2)]
T_num_en_prev_i = T_num_en_prev_res[:,int(mx/2)]
TP_num_en_prev_i = TP_num_en_prev_res[:,int(mx/2)]
phi_num_en_prev_i = phi_num_en_prev_res[:,int(mx/2)]
C_num_en_prev_i = C_num_en_prev_res[:,int(mx/2)]
Cf_num_en_prev_i = Cf_num_en_prev_res[:,int(mx/2)]
Cs_num_en_prev_i = Cs_num_en_prev_res[:,int(mx/2)]

# plot analytical data
pl = ax1.plot(H_an_i,zc,'k-',label='analytical')
pl = ax1.plot(H_prev_i,zc,'b-',label='initial')
pl = ax2.plot(Th_an_i,zc,'k-',label='analytical')
pl = ax3.plot(phi_an_i,zc,color='k',label='analytical')

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

  # imod = importlib.import_module(fname+'_enthalpy'+ft)
  fout = fname+'_enthalpy'+ft
  spec = importlib.util.spec_from_file_location(fout,fname_data+'/'+fout+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod)
  data_enthalpy = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(data_enthalpy)

  HC_data = data0['X_cell']
  enth_data = data_enthalpy['X_cell']

  H_num  = HC_data[0::2]
  H_num_en = enth_data[0::dof_en]
  T_num_en = enth_data[1::dof_en]
  TP_num_en = enth_data[2::dof_en]
  phi_num_en = enth_data[3::dof_en]

  H_num_res = H_num.reshape(mz,mx)
  H_num_en_res = H_num_en.reshape(mz,mx)
  T_num_en_res = T_num_en.reshape(mz,mx)
  TP_num_en_res = TP_num_en.reshape(mz,mx)
  phi_num_en_res = phi_num_en.reshape(mz,mx)

  H_num_i = H_num_res[:,int(mx/2)]
  H_num_en_i = H_num_en_res[:,int(mx/2)]
  T_num_en_i = T_num_en_res[:,int(mx/2)]
  TP_num_en_i = TP_num_en_res[:,int(mx/2)]
  phi_num_en_i = phi_num_en_res[:,int(mx/2)]

  pl = ax1.plot(H_num_i,zc,'*',label='tstep='+str(istep))
  pl = ax2.plot(T_num_en_i,zc,'*',label='tstep='+str(istep))
  pl = ax3.plot(phi_num_en_i,zc,'*',label='tstep='+str(istep))

ax1.set_title('a) Enthalpy')
ax1.set_xlabel('H [-]')
ax1.set_ylabel('z [-]')
ax1.legend(loc='lower left')
ax1.axis('auto')
ax1.grid(True,color='gray', linestyle='--', linewidth=0.5)

ax2.set_title('b) Temperature')
ax2.set_xlabel('Theta [-]')
# ax2.set_ylabel('z [-]')
ax2.legend(loc='lower left')
ax2.axis('auto')
ax2.grid(True,color='gray', linestyle='--', linewidth=0.5)

ax3.set_title('c) Porosity')
ax3.set_xlabel('phi [-]')
# ax3.set_ylabel('z [-]')
ax3.legend(loc='lower left')
ax3.axis('auto')
ax3.grid(True,color='gray', linestyle='--', linewidth=0.5)

pl = ax4.plot(np.log10(max_diff),'k-')
ax4.set_title('d) Steady-state convergence')
ax4.set_xlabel('tstep [-]')
ax4.set_ylabel('log10(max|x-xprev|)')
ax4.axis('auto')
ax4.grid(True,color='gray', linestyle='--', linewidth=0.5)

plt.savefig(fname_out+'/'+fname+'.pdf')

# calculate composition analytical enthalpy
xE = 1+C0/Cc
Cw = xE/(1-xE)
C_an_i = np.ones(len(H_an_i))*C0
Cf_an_i = np.zeros(len(H_an_i))
Cs_an_i = np.zeros(len(H_an_i))
for i in range(0,len(H_an_i)):
  if (phi_an_i[i]==1):
    Cf_an_i[i] = C_an_i[i]
    Cs_an_i[i] = 0.0
  else:
    Cs_an_i[i] = -Cw-1
    Cf_an_i[i] = (C_an_i[i]-(1.0-phi_an_i[i])*Cs_an_i[i])/phi_an_i[i]

# Plot solution (numerical and analytical)
fig, axs = plt.subplots(2,3,figsize=(12,8))

ax1 = plt.subplot(2,3,1)
pl = ax1.plot(H_an_i,zc,'k-',label='analytical')
pl = ax1.plot(H_num_en_prev_i,zc,'r--',label='numerical')
# ax1.set_title('a) Enthalpy')
ax1.set_xlabel('H [-]')
ax1.set_ylabel('z [-]')
ax1.legend(loc='lower left')
ax1.axis('auto')
ax1.grid(True,color='gray', linestyle='--', linewidth=0.5)

ax2 = plt.subplot(2,3,2)
pl = ax2.plot(Th_an_i,zc,'k-',label='analytical')
pl = ax2.plot(T_num_en_prev_i,zc,'r--',label='numerical')
# ax2.set_title('b) Temperature')
ax2.set_xlabel('theta [-]')
ax2.legend(loc='lower left')
ax2.axis('auto')
ax2.grid(True,color='gray', linestyle='--', linewidth=0.5)

ax3 = plt.subplot(2,3,3)
pl = ax3.plot(phi_an_i,zc,color='k',label='analytical')
pl = ax3.plot(phi_num_en_prev_i,zc,'r--',label='numerical')
# ax3.set_title('c) Porosity')
ax3.set_xlabel('phi [-]')
ax3.legend(loc='lower left')
ax3.axis('auto')
ax3.grid(True,color='gray', linestyle='--', linewidth=0.5)

ax1 = plt.subplot(2,3,4)
pl = ax1.plot(C_an_i,zc,'k-',label='analytical')
pl = ax1.plot(C_num_en_prev_i,zc,'r--',label='numerical')
# ax1.set_title('d) Bulk Composition')
ax1.set_xlabel('C [-]')
ax1.set_ylabel('z [-]')
ax1.legend(loc='lower left')
ax1.axis('auto')
ax1.grid(True,color='gray', linestyle='--', linewidth=0.5)

ax2 = plt.subplot(2,3,5)
pl = ax2.plot(Cf_an_i,zc,'k-',label='analytical')
pl = ax2.plot(Cf_num_en_prev_i,zc,'r--',label='numerical')
# ax2.set_title('e) Liquid composition')
ax2.set_xlabel('Cf [-]')
ax2.legend(loc='lower left')
ax2.axis('auto')
ax2.grid(True,color='gray', linestyle='--', linewidth=0.5)

ax3 = plt.subplot(2,3,6)
pl = ax3.plot(Cs_an_i,zc,color='k',label='analytical')
pl = ax3.plot(Cs_num_en_prev_i,zc,'r--',label='numerical')
# ax3.set_title('f) Solid composition')
ax3.set_xlabel('Cs [-]')
ax3.legend(loc='lower left')
ax3.axis('auto')
ax3.grid(True,color='gray', linestyle='--', linewidth=0.5)

plt.savefig(fname_out+'/'+fname+'_enthalpy_test_analytical.pdf')

os.system('rm -r '+fname_data+'/__pycache__')