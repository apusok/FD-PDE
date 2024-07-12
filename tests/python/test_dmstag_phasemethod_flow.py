# ---------------------------------------
# Phase-field method to capture the interface : with flows
# icase = 2, translation of a circle within a uniform flow
# icase = 3, deformation of a circle under a pure shear flow
# icase = 4, deformation of a circle under a simple shear flow
# icase = 5, deformation of a circle under a periodic swirling vortex 
# ---------------------------------------

# Import modules
import numpy as np
import matplotlib.pyplot as plt
import importlib
import os
import sys, getopt

# Input file
fname = 'out_dmstag_phasemethod'
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

n = 100
tout = 10
zw = 0.6
eps = 1.0/n #0.7/n
gamma = 1.0 #2.0

dt = 1.0/8.0/n #0.5*0.5/n**2/eps

tstep = int(0.8/dt) + 1

icase = 2

ux = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
uz = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0]

# solver parameters
phase = ' -eps '+str(eps)+' -gamma '+str(gamma)
model = ' -L 1.0 -H 1.0 -ux '+str(ux[icase])+' -uz '+str(uz[icase])+' -adv_scheme 0'+' -dt '+str(dt)+' -zw '+str(zw) +' -icase '+str(icase)

solver = ''

newton = ''


# Run test
# Forward euler
str1 = 'mpiexec -n '+str(ncpu)+' ../test_dmstag_phasemethod.app' + \
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

# analytical solution - prepare the initial state
mm = 200
theta = np.linspace(0, 2*np.pi, mm)
xini = 0.5 + 0.2 * np.cos(theta)
zini = 0.5 + 0.2 * np.sin(theta)

# Plot solution 
fig1, axs1 = plt.subplots(1, 1,figsize=(4,4))

for istep in range(0,tstep,int(tstep/2)):
  # Load python module describing data
  if (istep < 10): ft = '_ts00'+str(istep)
  if (istep >= 10) & (istep < 99): ft = '_ts0'+str(istep)
  if (istep >= 100): ft = '_ts'+str(istep)

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

#  if (istep == tstep-1):
  if (istep > 0):      
      
    tt = dt*istep
    
    # #update the analytical solution -    
    if (icase==4): #simple shear
      xana = xini + 1.0*(-1.0 + 2*zini)*tt
      zana = zini
    if (icase==3): #pure shear
      xana = 0.5 + (xini - 0.5)*np.exp(2*tt)
      zana = 0.5 + (zini - 0.5)*np.exp(-2*tt)
    
    cmaps='RdBu_r'
    # color map
    ax0 = axs1
    im = ax0.imshow(fres0,extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps,origin='lower')
    ax0.contour( xc , zc , fres0, levels=[0.5] , colors='black',linestyles='solid',linewidths=1.0)
    ct01= ax0.contour( xc , zc , fres_ini, levels=[0.5] , colors='gray',linestyles='dashed',linewidths=1.0)
    ax0.set_title('RK2')
    ax0.set_xlabel('x')
    ax0.set_ylabel('z')
  #  cbar = fig1.colorbar(im,ax=ax0, shrink=0.75)
    ax0.grid(True,color='gray', linestyle=':', linewidth=0.5)
    if (icase==3 or icase==4):
      ax0.plot(xana,zana,color='gray', linestyle='dashed',linewidth=1.0, label='Analytical')
        

fig1.savefig(fname+'/'+fname+'_rk2.pdf')
os.system('rm -r '+fname+'/__pycache__')
