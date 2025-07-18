# ---------------------------------------
# Pure-advection of a Gaussian pulse test (ADVDIFF) with PERIODIC BCs
# Plot analytical and numerical solution
# ---------------------------------------

# Import modules
import numpy as np
import matplotlib.pyplot as plt
import importlib
import os
import shutil 
import glob
import sys, getopt
import warnings
warnings.filterwarnings('ignore')

# Input file
fname = 'out_advdiff_periodic'
fname_data = fname+'/data'
try:
  os.mkdir(fname)
except OSError:
  pass

try:
  os.mkdir(fname_data)
except OSError:
  pass

# Get cpu number
ncpu = 1
options, remainder = getopt.getopt(sys.argv[1:],'n:')
for opt, arg in options:
  if opt in ('-n'):
    ncpu = int(arg)

print('# --------------------------------------- #')
print('# Pure-Advection test - PERIODIC (ADVDIFF) ')
print('# --------------------------------------- #')

n = 100
tstep = 401
tout = 100
x0 = 5.0
z0 = 5.0

# Gaussian shape flags
gs = ' -dt 1e-2' 

# Use umfpack for sequential and mumps for sequential/parallel
solver_default = ' -snes_monitor -snes_converged_reason -ksp_monitor -ksp_converged_reason '
if (ncpu == -1):
  solver = ' -pc_type lu -pc_factor_mat_solver_type umfpack -pc_factor_mat_ordering_type external'
  ncpu = 1
else:
  solver = ' -pc_type lu -pc_factor_mat_solver_type mumps'

# Run test 1 - horizontal
uz = 0.0
fname_out = fname+'_hor'
str1 = 'mpiexec -n '+str(ncpu)+' ../test_advdiff_periodic '+solver+' -output_file '+fname_out+' -output_dir '+fname_data+ \
  ' -nx '+str(n)+' -nz '+str(n)+' -x0 '+str(x0)+' -z0 '+str(z0)+' -uz '+str(uz)+' -tstep '+str(tstep)+' -tout '+str(tout)+gs+solver_default+ \
  ' > '+fname_data+'/'+fname_out+'.out'
print(str1)
os.system(str1)

ux = 0.0
fname_out = fname+'_ver'
str1 = 'mpiexec -n '+str(ncpu)+' ../test_advdiff_periodic '+solver+' -output_file '+fname_out+' -output_dir '+fname_data+ \
  ' -nx '+str(n)+' -nz '+str(n)+' -x0 '+str(x0)+' -z0 '+str(z0)+' -ux '+str(ux)+' -tstep '+str(tstep)+' -tout '+str(tout)+gs+solver_default+ \
  ' > '+fname_data+'/'+fname_out+'.out'
print(str1)
os.system(str1)

fname_out = fname
str1 = 'mpiexec -n '+str(ncpu)+' ../test_advdiff_periodic '+solver+' -output_file '+fname_out+' -output_dir '+fname_data+ \
  ' -nx '+str(n)+' -nz '+str(n)+' -x0 '+str(x0)+' -z0 '+str(z0)+' -tstep '+str(tstep)+' -tout '+str(tout)+gs+solver_default+ \
  ' > '+fname_data+'/'+fname_out+'.out'
print(str1)
os.system(str1)

# Plot solution (numerical and analytical)
fig, axs = plt.subplots(3, 3,figsize=(15,15))

# Prepare data for time series
nout = int((tstep-1)/tout+1)
dt = range(0,tstep,tout)

for istep in range(0,tstep,tout):
  # Load python module describing data
  if (istep < 10): ft = '_tstep00'+str(istep)
  if (istep >= 10) & (istep < 99): ft = '_tstep0'+str(istep)
  if (istep >= 100) & (istep < 999): ft = '_tstep'+str(istep)

  # Load data 
  fname_out = fname+'_hor'+ft
  spec = importlib.util.spec_from_file_location(fname_out,fname_data+'/'+fname_out+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod) 
  data0 = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(data0)

  fname_out = fname+'_ver'+ft
  spec = importlib.util.spec_from_file_location(fname_out,fname_data+'/'+fname_out+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod) 
  data1 = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(data1)

  fname_out = fname+ft
  spec = importlib.util.spec_from_file_location(fname_out,fname_data+'/'+fname_out+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod) 
  data2 = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(data2)

  # Load analytical data
  fname_out = 'out_analytic_solution'+ft
  spec = importlib.util.spec_from_file_location(fname_out,fname_data+'/'+fname_out+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod) 
  data_an = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(data_an)

  # Get general data (elements, grid)
  m = data0['Nx'][0]
  n = data0['Ny'][0]
  xc = data0['x1d_cell']
  zc = data0['y1d_cell']

  # Get individual data sets
  T0 = data0['X_cell']
  Tres0 = T0.reshape(n,m)
  Tx0 = Tres0[int(n/2),:]
  Tz0 = Tres0[:,int(m/2)]

  T1 = data1['X_cell']
  Tres1 = T1.reshape(n,m)
  Tx1 = Tres1[int(n/2),:]
  Tz1 = Tres1[:,int(m/2)]

  T2 = data2['X_cell']
  Tres2 = T2.reshape(n,m)
  Tx2 = Tres2[int(n/2),:]
  Tz2 = Tres2[:,int(m/2)]
  r = np.zeros(len(xc))
  Td2=np.zeros(len(xc))
  for ii in range(0,len(xc)):
    r[ii] = np.sqrt(xc[ii]**2+zc[ii]**2)
    Td2[ii] = Tres2[ii,ii]

  T_an = data_an['X_cell']
  Tres_an = T_an.reshape(n,m)
  Tx_an = Tres_an[int(n/2),:]
  Tz_an = Tres_an[:,int(m/2)]

  if (istep==0):
    ax = plt.subplot(3,3,1)
    im = ax.imshow(Tres0, extent=[min(xc), max(xc), min(zc), max(zc)],origin='lower', interpolation='nearest' )
    # ax.axis('image')
    ax.set_xlabel('x-dir')
    ax.set_title('Initial conditions (num)')
    cbar = fig.colorbar(im,ax=ax, shrink=0.75)
  #   cs = ax.contour(xc,zc,-Tres0, levels = [-5,], colors='k')
  
  # if (istep % tout == 0):
  #   ax = plt.subplot(3,3,1)
  #   cs = ax.contour(xc,zc,-Tres0, levels = [-5,], colors='w')

  if (istep==tstep-1):
    ax = plt.subplot(3,3,2)
    im = ax.imshow(Tres0, extent=[min(xc), max(xc), min(zc), max(zc)],origin='lower', interpolation='nearest' )
    ax.axis('image')
    ax.set_xlabel('x-dir')
    ax.set_title('istep = '+str(istep)+' (num)')
    cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  if (istep % tout == 0):
    ax = plt.subplot(3,3,3)
    pl = ax.plot(xc,Tx0,label='istep='+str(istep))
    # if (istep==0) | (istep==tstep-1):
    #   pl21 = ax.plot(xc,Tx_an,color='gray', linestyle='--',label='istep_an='+str(istep))
    ax.set_xlabel('x-dir')
    ax.set_ylabel('Q')
    ax.set_title('x-slice')
    ax.legend(loc='upper right')
    ax.axis('scaled')
    ax.grid(True,color='gray', linestyle='--', linewidth=0.5)

  if (istep==0):
    ax = plt.subplot(3,3,4)
    im = ax.imshow(Tres1, extent=[min(xc), max(xc), min(zc), max(zc)],origin='lower', interpolation='nearest' )
    ax.axis('image')
    ax.set_xlabel('x-dir')
    ax.set_title('Initial conditions (num)')
    cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  if (istep==tstep-1):
    ax = plt.subplot(3,3,5)
    im = ax.imshow(Tres1, extent=[min(xc), max(xc), min(zc), max(zc)],origin='lower', interpolation='nearest' )
    ax.axis('image')
    ax.set_xlabel('x-dir')
    ax.set_title('istep = '+str(istep)+' (num)')
    cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  if (istep % tout == 0):
    ax = plt.subplot(3,3,6)
    pl = ax.plot(zc,Tz1,label='istep='+str(istep))
    ax.set_xlabel('z-dir')
    ax.set_ylabel('Q')
    ax.set_title('z-slice')
    ax.legend(loc='upper right')
    ax.axis('scaled')
    ax.grid(True,color='gray', linestyle='--', linewidth=0.5)

  if (istep==0):
    ax = plt.subplot(3,3,7)
    im = ax.imshow(Tres2, extent=[min(xc), max(xc), min(zc), max(zc)],origin='lower', interpolation='nearest' )
    ax.axis('image')
    ax.set_xlabel('x-dir')
    ax.set_title('Initial conditions (num)')
    cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  if (istep==tstep-1):
    ax = plt.subplot(3,3,8)
    im = ax.imshow(Tres2, extent=[min(xc), max(xc), min(zc), max(zc)],origin='lower', interpolation='nearest' )
    ax.axis('image')
    ax.set_xlabel('x-dir')
    ax.set_title('istep = '+str(istep)+' (num)')
    cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  if (istep % tout == 0):
    ax = plt.subplot(3,3,9)
    pl = ax.plot(r,Td2,label='istep='+str(istep))
    ax.set_xlabel('r')
    ax.set_ylabel('Q')
    ax.set_title('diag-slice')
    ax.legend(loc='upper right')
    ax.axis('scaled')
    ax.grid(True,color='gray', linestyle='--', linewidth=0.5)

plt.savefig(fname+'/'+fname+'.pdf')

os.system('rm -r '+fname_data+'/__pycache__')