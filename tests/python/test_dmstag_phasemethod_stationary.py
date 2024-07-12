# ---------------------------------------
# Phase-field method to capture the interface : stationary interface
# ---------------------------------------

# Import modules
import numpy as np
import matplotlib.pyplot as plt
import importlib
import os
import sys, getopt

# Input file
fname = 'out_dmstag_phasemethod_stationary'
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

print('# --------------------------------------- #')
print('# Phase-field method test: stationary interface (DMSTAG) ')
print('# --------------------------------------- #')

n = 40
tout = 1
zw = 0.6
eps = 1.0/n
gamma = 1.0

dt = 1.0/8.0/n

tstep = int(0.2/dt) + 1

icase = 0

ux = [0.0, 0.0, 1.0]
uz = [0.0, 0.0, 1.0]

# solver parameters
phase = ' -eps '+str(eps)+' -gamma '+str(gamma)
model = ' -L 1.0 -H 1.0 -ux '+str(ux[icase])+' -uz '+str(uz[icase])+' -adv_scheme 0'+' -dt '+str(dt)+' -zw '+str(zw) +' -icase '+str(icase)

solver = ''

newton = ''


# Run test
# Forward euler
str1 = 'mpiexec -n '+str(ncpu)+' ../test_dmstag_phasemethod' + \
       ' -nx '+str(n)+' -nz '+str(n)+' -tstep '+str(tstep) + \
       newton + model + phase + \
       ' -output_dir '+fname+' -output_file '+fname+' -tout '+str(tout)+' -ts_scheme 0'
print(str1)
os.system(str1)


# Prepare data for time series
# Load data - initial
f1out = fname+'_initial'
spec = importlib.util.spec_from_file_location(f1out,fname+'/'+f1out+'.py')
imod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(imod)
data_ini = imod._PETScBinaryLoad()
imod._PETScBinaryLoadReportNames(data_ini)
# imod = importlib.import_module(fname+'_initial')
# data_ini = imod._PETScBinaryLoad()
# imod._PETScBinaryLoadReportNames(data_ini)

# Get general data (elements, grid)
m = data_ini['Nx'][0]
n = data_ini['Ny'][0]
xc = data_ini['x1d_cell']
zc = data_ini['y1d_cell']

f_ini = data_ini['X_cell']
fres_ini = f_ini.reshape(n,m)
fx_ini = fres_ini[:,int(m/2)]

# compute the analytical solution along a vertical slice x = 0.5
fres_ana = np.zeros((n,m))

for i in range(0, m):
  for j in range(0, n):
    if icase == 0: # flat interface
      zz = zc[j] - zw
      fres_ana[j,i] = 0.5*(1.0 + np.tanh(zz/2.0/eps))
    if icase == 1: # circular interface of center = (0.5, 0.5), radius = 0.2
      zz = ((xc[i] - 0.5)**2 + (zc[j]-0.5)**2)**0.5 - 0.2
      fres_ana[i,j] = 0.5*(1.0 + np.tanh(zz/2.0/eps))

fx_ana = fres_ana[:,int(m/2)]

# Plot solution 
fig1, axs1 = plt.subplots(1, 2,figsize=(8,4))

for istep in range(0,tstep,tout):
  # Load python module describing data
  if (istep < 10): ft = '_ts00'+str(istep)
  if (istep >= 10) & (istep < 99): ft = '_ts0'+str(istep)
  if (istep >= 100) & (istep < 999): ft = '_ts'+str(istep)

  # Load data - m0
  f1out = fname+'_m0'+ft
  spec = importlib.util.spec_from_file_location(f1out,fname+'/'+f1out+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod)
  data0 = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(data0)
  # imod = importlib.import_module(fname+'_m0'+ft)
  # data0 = imod._PETScBinaryLoad()
  #imod._PETScBinaryLoadReportNames(data0)

  # Get individual data sets
  f0 = data0['X_cell']
  fres0 = f0.reshape(n,m)
  fx0 = fres0[:,int(m/2)]


  if (istep == tstep-1):
#  if (istep >= 0):      
      
    tt = dt*istep
    
    cmaps='RdBu_r'
    #color map
    ax0 = axs1[0]
    im = ax0.imshow(fres0,extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps,origin='lower')
    ct0= ax0.contour( xc , zc , fres0, levels=[0.5] , colors='black',linestyles='solid',linewidths=1.0)
    ax0.set_title('f')
    ax0.set_xlabel('x')
    ax0.set_ylabel('z')
    cbar = fig1.colorbar(im,ax=ax0, shrink=0.75)
        
    #line plot
    ax0 = axs1[1]
    ax0.plot(zc,fx0, color='black',label='t=0.2')
    #if (istep==0) | (istep==tstep-1):
    pl01 = ax0.plot(zc,fx_ini,color='gray', linestyle=':',label='initial state')
    pl02 = ax0.plot(zc,fx_ana,color='black', linestyle='--', label='kernel function')
    ax0.set_xlabel('z-dir')
    ax0.set_ylabel('f')
    ax0.set_title('RK2')
    ax0.legend(title='f(z) at x=0.5',loc='upper left',fontsize='small')
    ax0.axis('scaled')
    ax0.grid(True,color='gray', linestyle='--', linewidth=0.5)



fig1.savefig(fname+'/'+fname+'_rk2.pdf')
os.system('rm -r '+fname+'/__pycache__')