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

print('# --------------------------------------- #')
print('# 1-D Solidification test (ENTHALPY) ')
print('# --------------------------------------- #')

n = 32
tstep = 20
tout = 1

# physical parameters
Tb = -45
T0 = 0
Tm = -0.1
DT = Tm-Tb
h = 4
dx = 0.125
dt = 0.2
rho = 1
cp = 1
k = 1.08
La = 70.26
St = cp*DT/La
kappa = k/rho/cp
nd_t = h**2/kappa

# Run test
str1 = '../test_enthalpy_1d_solidification.app -pc_type lu -pc_factor_mat_solver_type umfpack -snes_monitor -log_view'
print(str1)
# os.system(str1)

# Prepare data for time series - nondimensional
x0 = 1/h
x = np.arange(0,h,dx)/h
t = np.arange(0,tstep,tout)*(dt/nd_t)
nout = int((tstep-1)/tout+1)
T_1m = np.zeros(nout)
s_num = np.zeros(nout)

# analytical solution
beta = 0.516385
#beta*exp(beta**2)*erf(beta)-St/sqrt(pi)
# xfront_an = 2.0*beta*np.sqrt(t)
# T_an = (erf(x/2/sqrt(t))/erf(beta) - 1)

for istep in range(0,tstep,tout):
  # Load python module describing data
  if (istep < 10): ft = '_ts00'+str(istep)
  if (istep >= 10) & (istep < 99): ft = '_ts0'+str(istep)
  if (istep >= 100) & (istep < 999): ft = '_ts'+str(istep)

  # Load data - m0
  imod = importlib.import_module(fname+'_TC'+ft)
  data0 = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(data0)

  mx = data0['Nx'][0]
  mz = data0['Ny'][0]
  xc = data0['x1d_cell']
  zc = data0['y1d_cell']

  # extract a) middle temperature profile, b) crystallization front, and c) temperature at x=1
  T_data = data0['X_cell']
  Ti = T_data[0::2]
  Ti_res = Ti.reshape(mz,mx)

  # T  = Ti_res[int(mz/2),:]*DT+Tm
  T = Ti_res[int(mz/2),:]

  for i in range(0,mx-1):
    if (xc[i]<=x0) & (xc[i+1]>x0):
      im = i
    if (T[i]<=0) & (T[i+1]>0):
      ic = i

  ii = int(istep/tout)
  T_1m[ii] = T[im]
  s_num[ii] = xc[ic]

# Plot solution (numerical and analytical)
fig, axs = plt.subplots(1,3,figsize=(15,4))

# temperature distribution in the slab - last timestep
ax = plt.subplot(1,3,1)
T_end_an = (np.erf(x/2/np.sqrt(t[-1]))/np.erf(beta) - 1)
pl = ax.plot(xc,T,color='red',label='numerical')
pl = ax.plot(xc,T_end_an,color='black',label='analytical')
ax.set_title('Temperature profile in the slab at t = 4 s')
ax.set_ylabel('Temperature [-]')
ax.set_xlabel('Distance [-]')
ax.legend(loc='lower right')
ax.axis('auto')
ax.grid(True,color='gray', linestyle='--', linewidth=0.5)

# Temperature evolution at x = 1m
ax = plt.subplot(1,3,2)
T_end_an = (np.erf(x0/2/np.sqrt(t))/np.erf(beta) - 1)
pl = ax.plot(t,T_1m,color='red',label='numerical')
pl = ax.plot(t,T_end_an,color='black',label='analytical')
ax.set_title('Temperature profile in the slab at x = 1m')
ax.set_ylabel('Temperature [-]')
ax.set_xlabel('Time [-]')
ax.legend(loc='lower right')
ax.axis('auto')
ax.grid(True,color='gray', linestyle='--', linewidth=0.5)

# Crystallization front 
ax = plt.subplot(1,3,3)
xfront_an = 2.0*beta*np.sqrt(t)
pl = ax.plot(t,s_num,color='red',label='numerical')
pl = ax.plot(t,xfront_an,color='black',label='analytical')
ax.set_title('Solidification front')
ax.set_ylabel('x [-]')
ax.set_xlabel('Time [-]')
ax.legend(loc='lower right')
ax.axis('auto')
ax.grid(True,color='gray', linestyle='--', linewidth=0.5)

plt.savefig(fname+'.pdf')