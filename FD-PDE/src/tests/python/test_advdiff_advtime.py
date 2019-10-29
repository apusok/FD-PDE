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

n = 10
tstep = 401

# Run test
str1 = '../test_advdiff_advtime.app -pc_type lu -pc_factor_mat_solver_type umfpack -output_file '+fname+' -nx '+str(n)+' -nz '+str(n)+' -tstep '+str(tstep)
print(str1)
os.system(str1)

# Plot solution (numerical and analytical)
fig, axs = plt.subplots(1, 2,figsize=(12,6))

il = 3
iline = 0

for istep in range(0,tstep,50):
  # Load python module describing data
  if (istep < 10): ft = '_ts00'+str(istep)
  if (istep >= 10) & (istep < 99): ft = '_ts0'+str(istep)
  if (istep >= 100) & (istep < 999): ft = '_ts'+str(istep)
  imod = importlib.import_module(fname+ft)

  # Load data
  data = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(data)

  # Get data
  m = data['Nx'][0]
  n = data['Ny'][0]
  xc = data['x1d_cell']
  zc = data['y1d_cell']
  T = data['X_cell']
  Tres = T.reshape(n,m)
  Tx = Tres[int(n/2),:]
  Tz = Tres[:,int(n/2)]

  ax1 = axs[0]
  im = ax1.plot(xc,Tx,label='istep='+str(istep))
  ax1.set_xlabel('x-dir')
  ax1.set_ylabel('Q')
  ax1.set_title('X-section')
  ax1.legend(title='Qx',loc='upper left')
  ax1.axis('scaled')
  # ax1.set(xlim=(xc[0], xc[-1]), ylim=(0, 5))
  ax1.grid(True,color='gray', linestyle='--', linewidth=0.5)

  ax2 = axs[1]
  im = ax2.plot(zc,Tz,label='istep='+str(istep))
  ax2.set_xlabel('z-dir')
  ax2.set_ylabel('Q')
  ax2.set_title('Z-section')
  ax2.legend(title='Qz',loc='upper left')
  ax2.axis('scaled')
  # ax2.set(xlim=(xc[0], xc[-1]), ylim=(0, 5))
  ax2.grid(True,color='gray', linestyle='--', linewidth=0.5)

  iline=iline+1

plt.savefig(fname+'.pdf')