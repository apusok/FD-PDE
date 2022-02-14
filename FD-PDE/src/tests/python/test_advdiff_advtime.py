# ---------------------------------------
# Pure-advection of a Gaussian pulse test (ADVDIFF)
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

# Input file
fname = 'out_advdiff_advtime'
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
print('# Pure-Advection test (ADVDIFF) ')
print('# --------------------------------------- #')

n = 100
tstep = 201
tout = 10

# Gaussian shape flags
adv = [0, 1, 2]
ts  = [0, 1, 2]
gs = ' -L 20.0 -A 10.0 -x0 5.0 -ux 2.0 -dt 1e-2' 
adv_label = ['Upwind','Upwind2','Fromm']

# Use umfpack for sequential and mumps for sequential/parallel
solver_default = ' -snes_monitor -snes_converged_reason -ksp_monitor -ksp_converged_reason '
if (ncpu == -1):
  solver = ' -pc_type lu -pc_factor_mat_solver_type umfpack -pc_factor_mat_ordering_type external'
  ncpu = 1
else:
  solver = ' -pc_type lu -pc_factor_mat_solver_type mumps'

# Plot solution (numerical and analytical)
fig, axs = plt.subplots(4, 3,figsize=(18,15))

# Run tests
for iadv in adv:
  for its in ts:
    # Do every ts_scheme
    fname_out = fname+'_ts'+str(its)+'_adv'+str(iadv)
    str1 = 'mpiexec -n '+str(ncpu)+' ../test_advdiff_advtime.app '+solver+' -output_file '+fname_out+' -output_dir '+fname_data+ \
      ' -nx '+str(n)+' -nz '+str(n)+' -tstep '+str(tstep)+' -tout '+str(tout)+gs+solver_default+ \
      ' -ts_scheme '+str(its)+\
      ' -adv_scheme '+str(iadv)+' > '+fname_data+'/'+fname_out+'.out'
    print(str1)
    os.system(str1)

  # Prepare data for time series
  nout = int((tstep-1)/tout+1)
  mass00 = np.zeros(nout)
  mass01 = np.zeros(nout)
  mass02 = np.zeros(nout)
  massan = np.zeros(nout)
  dt = range(0,tstep,tout)

  for istep in range(0,tstep,tout):
    # Load python module describing data
    if (istep < 10): ft = '_tstep00'+str(istep)
    if (istep >= 10) & (istep < 99): ft = '_tstep0'+str(istep)
    if (istep >= 100) & (istep < 999): ft = '_tstep'+str(istep)

    # Load data - m0
    # imod = importlib.import_module(fname+'_ts0'+'_adv'+str(iadv)+ft)
    fname_out = fname+'_ts0'+'_adv'+str(iadv)+ft
    spec = importlib.util.spec_from_file_location(fname_out,fname_data+'/'+fname_out+'.py')
    imod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imod) 

    data0 = imod._PETScBinaryLoad()
    imod._PETScBinaryLoadReportNames(data0)

    # Load data - m1
    # imod = importlib.import_module(fname+'_ts1'+'_adv'+str(iadv)+ft)
    fname_out = fname+'_ts1'+'_adv'+str(iadv)+ft
    spec = importlib.util.spec_from_file_location(fname_out,fname_data+'/'+fname_out+'.py')
    imod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imod) 

    data1 = imod._PETScBinaryLoad()
    imod._PETScBinaryLoadReportNames(data1)

    # Load data - m2
    # imod = importlib.import_module(fname+'_ts2'+'_adv'+str(iadv)+ft)
    fname_out = fname+'_ts2'+'_adv'+str(iadv)+ft
    spec = importlib.util.spec_from_file_location(fname_out,fname_data+'/'+fname_out+'.py')
    imod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imod) 

    data2 = imod._PETScBinaryLoad()
    imod._PETScBinaryLoadReportNames(data2)

    # Load analytical data
    # imod = importlib.import_module('out_analytic_solution'+ft)
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

    T1 = data1['X_cell']
    Tres1 = T1.reshape(n,m)
    Tx1 = Tres1[int(n/2),:]

    T2 = data2['X_cell']
    Tres2 = T2.reshape(n,m)
    Tx2 = Tres2[int(n/2),:]

    T_an = data_an['X_cell']
    Tres_an = T_an.reshape(n,m)
    Tx_an = Tres_an[int(n/2),:]

    # Save mass plot - can do this because it's constant grid spacing
    ii = int(istep/tout)
    mass00[ii] = np.sum(Tx0)
    mass01[ii] = np.sum(Tx1)
    mass02[ii] = np.sum(Tx2)
    massan[ii] = np.sum(Tx_an)

    if (istep % 50 == 0):
      # forward euler
      ax0 = axs[0,iadv]
      pl0 = ax0.plot(xc,Tx0,label='istep='+str(istep))
      if (istep==0) | (istep==tstep-1):
        pl01 = ax0.plot(xc,Tx_an,color='gray', linestyle='--',label='istep_an='+str(istep))
      ax0.set_xlabel('x-dir')
      ax0.set_ylabel('Q')
      ax0.set_title('ADV: '+adv_label[iadv]+' TS: Forward Euler')
      ax0.legend(title='Qx',loc='upper right')
      ax0.axis('scaled')
      ax0.grid(True,color='gray', linestyle='--', linewidth=0.5)

      # backward euler
      ax1 = axs[1,iadv]
      pl1 = ax1.plot(xc,Tx1,label='istep='+str(istep))
      if (istep==0) | (istep==tstep-1):
        pl11 = ax1.plot(xc,Tx_an,color='gray', linestyle='--',label='istep_an='+str(istep))
      ax1.set_xlabel('x-dir')
      ax1.set_ylabel('Q')
      ax1.set_title('ADV: '+adv_label[iadv]+' TS: Backward Euler')
      ax1.legend(title='Qx',loc='upper right')
      ax1.axis('scaled')
      ax1.grid(True,color='gray', linestyle='--', linewidth=0.5)

      # crank-nicholson
      ax2 = axs[2,iadv]
      pl2 = ax2.plot(xc,Tx2,label='istep='+str(istep))
      if (istep==0) | (istep==tstep-1):
        pl21 = ax2.plot(xc,Tx_an,color='gray', linestyle='--',label='istep_an='+str(istep))
      ax2.set_xlabel('x-dir')
      ax2.set_ylabel('Q')
      ax2.set_title('ADV: '+adv_label[iadv]+' TS: Crank-Nicholson')
      ax2.legend(title='Qx',loc='upper right')
      ax2.axis('scaled')
      ax2.grid(True,color='gray', linestyle='--', linewidth=0.5)

  # mass plot
  ax3 = axs[3,iadv]
  pl30 = ax3.plot(dt,massan-massan,color='gray',label='analytical')
  pl31 = ax3.plot(dt,mass00-massan,color='r',label='forward Euler')
  pl32 = ax3.plot(dt,mass01-massan,color='b',label='backward Euler')
  pl33 = ax3.plot(dt,mass02-massan,color='k',label='Crank-Nicholson')
  ax3.set_xlabel('Time steps')
  ax3.set_ylabel('Int(Q)-Int(Q_an)')
  ax3.set_title('ADV: '+adv_label[iadv])
  ax3.legend(loc='upper right')
  # ax3.axis('scaled')
  ax3.grid(True,color='gray', linestyle='--', linewidth=0.5)

plt.savefig(fname+'/'+fname+'.pdf')

os.system('rm -r '+fname_data+'/__pycache__')