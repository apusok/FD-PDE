# ---------------------------------------
# Pure-advection of a Gaussian pulse test (ADVDIFF)
# Plot analytical and numerical solution
# ---------------------------------------

# Import modules
import numpy as np
import matplotlib.pyplot as plt
import importlib
import os

# Input file
fname = 'out_advdiff_advtime'

print('# --------------------------------------- #')
print('# Pure-Advection test (ADVDIFF) ')
print('# --------------------------------------- #')

n = 100
tstep = 201
tout = 50

# Gaussian shape flags
gs = ' -L 20.0 -A 10.0 -x0 5.0 -ux 2.0 -dt 1e-2 -adv_scheme 1'

# Run test
# Forward euler
str1 = '../test_advdiff_advtime.app -pc_type lu -pc_factor_mat_solver_type umfpack -output_file '+fname+' -nx '+str(n)+' -nz '+str(n)+' -tstep '+str(tstep)+' -tout '+str(tout)+' -ts_scheme 0'+gs
print(str1)
os.system(str1)

# Backward Euler
str2 = '../test_advdiff_advtime.app -pc_type lu -pc_factor_mat_solver_type umfpack -output_file '+fname+' -nx '+str(n)+' -nz '+str(n)+' -tstep '+str(tstep)+' -tout '+str(tout)+' -ts_scheme 1'+gs
print(str2)
os.system(str2)

# Crank-Nicholson
str3 = '../test_advdiff_advtime.app -pc_type lu -pc_factor_mat_solver_type umfpack -output_file '+fname+' -nx '+str(n)+' -nz '+str(n)+' -tstep '+str(tstep)+' -tout '+str(tout)+' -ts_scheme 2'+gs
print(str3)
os.system(str3)

# Plot solution (numerical and analytical)
fig, axs = plt.subplots(3, 2,figsize=(12,12))

for istep in range(0,tstep,tout):
  # Load python module describing data
  if (istep < 10): ft = '_ts00'+str(istep)
  if (istep >= 10) & (istep < 99): ft = '_ts0'+str(istep)
  if (istep >= 100) & (istep < 999): ft = '_ts'+str(istep)

  # Load data - m0
  imod = importlib.import_module(fname+'_m0'+ft)
  data0 = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(data0)

  # Load data - m1
  imod = importlib.import_module(fname+'_m1'+ft)
  data1 = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(data1)

  # Load data - m2
  imod = importlib.import_module(fname+'_m2'+ft)
  data2 = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(data2)

  # Load analytical data
  imod = importlib.import_module('out_analytic_solution'+ft)
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

  # subplot(1,1,0)
  ax0 = axs[0,0]
  pl0 = ax0.plot(xc,Tx0,label='istep='+str(istep))
  ax0.set_xlabel('x-dir')
  ax0.set_ylabel('Q')
  ax0.set_title('Forward Euler')
  ax0.legend(title='Qx',loc='upper right')
  ax0.axis('scaled')
  ax0.grid(True,color='gray', linestyle='--', linewidth=0.5)

  # subplot(1,1,1) - analytical
  ax1 = axs[0,1]
  if (istep==0) | (istep==tstep-1):
    pl1 = ax1.plot(xc,Tx0,label='istep='+str(istep))
    pl12 = ax1.plot(xc,Tx_an,color='r', linestyle='--',label='istep_an='+str(istep))
  ax1.set_xlabel('x-dir')
  ax1.set_ylabel('Q')
  ax1.legend(title='Qx',loc='upper right')
  ax1.axis('scaled')
  ax1.grid(True,color='gray', linestyle='--', linewidth=0.5)

  # subplot(2,1,2)
  ax2 = axs[1,0]
  pl2 = ax2.plot(xc,Tx1,label='istep='+str(istep))
  ax2.set_xlabel('x-dir')
  ax2.set_ylabel('Q')
  ax2.set_title('Backward Euler')
  ax2.legend(title='Qx',loc='upper right')
  ax2.axis('scaled')
  ax2.grid(True,color='gray', linestyle='--', linewidth=0.5)

  # subplot(2,1,3) - analytical
  ax3 = axs[1,1]
  if (istep==0) | (istep==tstep-1):
    pl3 = ax3.plot(xc,Tx1,label='istep='+str(istep))
    pl13 = ax3.plot(xc,Tx_an,color='r', linestyle='--',label='istep_an='+str(istep))
  ax3.set_xlabel('x-dir')
  ax3.set_ylabel('Q')
  ax3.legend(title='Qx',loc='upper right')
  ax3.axis('scaled')
  ax3.grid(True,color='gray', linestyle='--', linewidth=0.5)

  # subplot(3,1,4)
  ax4 = axs[2,0]
  pl4 = ax4.plot(xc,Tx2,label='istep='+str(istep))
  ax4.set_xlabel('x-dir')
  ax4.set_ylabel('Q')
  ax4.set_title('Crank-Nicholson')
  ax4.legend(title='Qx',loc='upper right')
  ax4.axis('scaled')
  ax4.grid(True,color='gray', linestyle='--', linewidth=0.5)

  # subplot(3,1,5) - analytical
  ax5 = axs[2,1]
  if (istep==0) | (istep==tstep-1):
    pl5 = ax5.plot(xc,Tx2,label='istep='+str(istep))
    pl15 = ax5.plot(xc,Tx_an,color='r', linestyle='--',label='istep_an='+str(istep))
  ax5.set_xlabel('x-dir')
  ax5.set_ylabel('Q')
  ax5.legend(title='Qx',loc='upper right')
  ax5.axis('scaled')
  ax5.grid(True,color='gray', linestyle='--', linewidth=0.5)

  # ax2 = axs[1]
  # im = ax2.plot(zc,Tz,label='istep='+str(istep))
  # ax2.set_xlabel('z-dir')
  # ax2.set_ylabel('Q')
  # ax2.set_title('Z-section')
  # ax2.legend(title='Qz',loc='upper left')
  # ax2.axis('scaled')
  # # ax2.set(xlim=(xc[0], xc[-1]), ylim=(0, 5))
  # ax2.grid(True,color='gray', linestyle='--', linewidth=0.5)


plt.savefig(fname+'.pdf')