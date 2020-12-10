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
    t = np.zeros(tstep)
    f = open(fname, 'r')
    i0=-1
    for line in f:
      if '# TIMESTEP' in line:
        i0+=1
      if '[nd,dim] time ' in line:
        t[i0] = float(line[23:35])
    f.close()

    return tstep, t
  except OSError:
    print('Cannot open:', fname)
    return tstep

# ---------------------------------------
# Function definitions
# ---------------------------------------

# Input file
fname = 'out_1d_sol_TC'
fname_out = 'out_enthalpy_1d_solid_TC'
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
print('# 1-D Solidification test (ENTHALPY T/HC) ')
print('# --------------------------------------- #')

n = 65
dt = 0.05
tend = 4
tstep = int(tend/dt)
tout = 1
tstart = 0.0

# physical parameters
Tb = -45
T0 = 0
Tm = -0.1
DT = Tm-Tb
h = 4
dx = h/n
rho = 1
cp = 1
k = 1.08
La = 70.26
St = cp*DT/La
kappa = k/rho/cp
nd_t = h**2/kappa
beta = 0.516385

# Run test
solver = ' -snes_converged_reason -ksp_converged_reason -snes_monitor -ksp_monitor -snes_atol 1e-10 -snes_rtol 1e-20'
str1 = '../test_enthalpy_1d_solidification_TC.app -pc_type lu -pc_factor_mat_solver_type umfpack -snes_monitor -snes_max_it 200'+ \
    ' -output_file '+fname+ \
    ' -output_dir '+fname_data+ \
    ' -dtmax '+str(dt)+ \
    ' -tmax '+str(tend)+ \
    ' -tstart '+str(tstart)+ \
    ' -tstep '+str(tstep)+ \
    ' -beta '+str(beta)+ \
    ' -nx '+str(n)+solver + ' > '+fname_data+'/log'+fname+'.out'
print(str1)
os.system(str1)

# Plot solution (numerical and analytical)
fig, axs = plt.subplots(2,2,figsize=(10,10))
ax1 = plt.subplot(2,2,1)
ax2 = plt.subplot(2,2,2)
ax3 = plt.subplot(2,2,3)
ax4 = plt.subplot(2,2,4)

# parse logfile
[nout,t] = parse_log_file(fname_data+'/log'+fname+'.out')

# Prepare data for time series - nondimensional
x0 = 1.0/h
# x = np.arange(0,h,dx)/h
# t = np.arange(0,tstep,tout)*(dt/nd_t)+tstart
# t[0] = tstart/nd_t # start after a non-zero time
nout = int((tstep-1)/tout+1)
T_1m = np.zeros(nout)
s_num = np.zeros(nout)

# analytical solution
#beta*exp(beta**2)*erf(beta)-St/sqrt(pi)
# xfront_an = 2.0*beta*np.sqrt(t)
# T_an = (erf(x/2/sqrt(t))/erf(beta) - 1)

for istep in range(0,tstep,tout):
  # Load python module describing data
  if (istep < 10): ft = '_ts00'+str(istep)
  if (istep >= 10) & (istep < 99): ft = '_ts0'+str(istep)
  if (istep >= 100) & (istep < 999): ft = '_ts'+str(istep)

  # Load data - m0
  # imod = importlib.import_module(fname+'_TC'+ft)
  fout = fname+'_HC'+ft
  spec = importlib.util.spec_from_file_location(fout,fname_data+'/'+fout+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod)
  data0 = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(data0)

  fout = fname+'_enthalpy'+ft
  spec = importlib.util.spec_from_file_location(fout,fname_data+'/'+fout+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod)
  data_enth = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(data_enth)

  mx = data0['Nx'][0]
  mz = data0['Ny'][0]
  xc = data0['x1d_cell']
  zc = data0['y1d_cell']

  # extract a) middle temperature profile, b) crystallization front, and c) temperature at x=1
  T_data = data0['X_cell']
  dof = 2
  Ti = T_data[0::dof]
  Ti_res = Ti.reshape(mz,mx)
  # T = Ti_res[int(mz/2),:]

  # extract temperature from the enthalpy variables
  HC_enth_data = data_enth['X_cell']
  ncomp = 2
  dof_en = 5 + 3*ncomp
  H_enth = HC_enth_data[0::dof_en]
  T_enth = HC_enth_data[1::dof_en]
  T_enth_res = T_enth.reshape(mz,mx)
  T = T_enth_res[int(mz/2),:]

  T_an = np.zeros(len(T))

  ic = 0
  for i in range(0,mx-1):
    if (xc[i]<=x0) & (xc[i+1]>x0):
      im = i
    if (T[i]<=0) & (T[i+1]>0):
      ic = i

  ii = int(istep/tout)
  T_1m[ii] = T[im]
  s_num[ii] = xc[ic]

  # plot every timestep
  if (np.mod(istep,20)==0):
    xfront_an = 2.0*beta*np.sqrt(t[istep])
    for i in range(0,len(xc)):
      ix = xc[i]
      if (ix > xfront_an):
        T_an[i] = 0.0
      else:
        T_an[i] = (np.erf(ix/2/np.sqrt(t[istep]))/np.erf(beta) - 1)

    # print(T_an*DT+Tm)
    pl = ax1.plot(xc*h,T_an*DT+Tm,'-',label='tstep = '+str(istep))
    pl = ax1.plot(xc*h,T*DT+Tm,'k*')

# print(T)
ax1.set_title('a) Temperature profile')
ax1.set_ylabel('T [oC]')
ax1.set_xlabel('x [m]')
ax1.legend(loc='lower right')
ax1.axis('auto')
ax1.grid(True,color='gray', linestyle='--', linewidth=0.5)

# print(xc)

# temperature distribution in the slab - last timestep
T_end_an = np.zeros(len(T))
xfront_an = 2.0*beta*np.sqrt(t[-1])
for i in range(0,len(xc)):
  ix = xc[i]
  if (ix > xfront_an):
    T_end_an[i] = 0.0
  else:
    T_end_an[i] = (np.erf(ix/2/np.sqrt(t[-1]))/np.erf(beta) - 1)

# T_end_an = (np.erf(x/2/np.sqrt(t[-1]))/np.erf(beta) - 1)
# T_end_an[T_end_an>0.0] = 0.0
pl = ax2.plot(xc*h,T_end_an*DT+Tm,color='black',label='analytical')
pl = ax2.plot(xc*h,T*DT+Tm,'r*',label='numerical')
ax2.set_title('b) Temperature profile in the slab at t = 4 s')
ax2.set_ylabel('T [oC]')
ax2.set_xlabel('x [m]')
ax2.legend(loc='lower right')
ax2.axis('auto')
ax2.grid(True,color='gray', linestyle='--', linewidth=0.5)

# Temperature evolution at x = 1m
# T_end_an = (np.erf(x0/2/np.sqrt(t))/np.erf(beta) - 1)
# T_end_an[T_end_an>0.0] = 0.0
T_num_1m = np.zeros(nout)
xfront_an = 2.0*beta*np.sqrt(t)
for i in range(0,len(t)):
  if (x0 > xfront_an[i]):
    T_num_1m[i] = 0.0
  else:
    T_num_1m[i] = (np.erf(x0/2/np.sqrt(t[i]))/np.erf(beta) - 1)

pl = ax3.plot(t*nd_t,T_num_1m*DT+Tm,color='black',label='analytical')
pl = ax3.plot(t*nd_t,T_1m*DT+Tm,'r*',label='numerical')
ax3.set_title('c) Temperature profile in the slab at x = 1m')
ax3.set_ylabel('T [oC]')
ax3.set_xlabel('time [s]')
ax3.legend(loc='upper right')
ax3.axis('auto')
ax3.grid(True,color='gray', linestyle='--', linewidth=0.5)

# Crystallization front 
xfront_an = 2.0*beta*np.sqrt(t)
pl = ax4.plot(t*nd_t,xfront_an*h,color='black',label='analytical')
pl = ax4.plot(t*nd_t,s_num*h,'r*',label='numerical')
ax4.set_title('d) Solidification front')
ax4.set_ylabel('x [m]')
ax4.set_xlabel('time [s]')
ax4.legend(loc='lower right')
ax4.axis('auto')
ax4.grid(True,color='gray', linestyle='--', linewidth=0.5)

plt.savefig(fname_out+'/'+fname+'.pdf')

os.system('rm -r '+fname_data+'/__pycache__')