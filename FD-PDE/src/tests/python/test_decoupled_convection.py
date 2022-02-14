# ---------------------------------------
# Mantle convection benchmark (Blankenbach et al. 1989)
# Steady-state models:
#    1A - eta0=1e23, b=0, c=0, Ra = 1e4
#    1B - eta0=1e22, b=0, c=0, Ra = 1e5
#    1C - eta0=1e21, b=0, c=0, Ra = 1e6
#    2A - eta0=1e23, b=ln(1000), c=0
#    2B - eta0=1e23, b=ln(16384), c=ln(64), L=2500
# Time-dependent models:
#    3A - eta0=1e23, b=0, c=0, L=1500, Ra = 216000
# ---------------------------------------

# Import modules
import numpy as np
import matplotlib.pyplot as plt
import importlib
import os
import sys, getopt

print('# --------------------------------------- #')
print('# Mantle convection benchmark (Blankenbach et al. 1989)')
print('# --------------------------------------- #')

nsim = 5
n = 50

# nx 50 [9h 59m 11s] nx 100 [~3days]
lbl_model = ['1A','1B','1C','2A','2B']
itest = [1, 1, 1, 2, 2]
isubtest = [10, 11, 12, 20, 21]
iL = [1.0e6, 1.0e6, 1.0e6, 1.0e6, 2.5e6]
# iRa = [1e4, 1e5, 1e6, 1e4, 1e4]
ieta0 = [1e23, 1e22, 1e21, 1e23, 1e23]
ib = [0.0, 0.0, 0.0, 6.90776, 9.70406]
ic = [0.0, 0.0, 0.0, 0.0, 4.15888]

# other physical parameters
tmax = 5e3 #[Myr] 0.25
dtmax = 20.0 #[Myr]1e-3
tout = 100 #50
ts_scheme = 2
adv_scheme = 1
tstep_max = 100000 #timesteps
H = 1e6

# Get cpu number
ncpu = 1
options, remainder = getopt.getopt(sys.argv[1:],'n:')
for opt, arg in options:
  if opt in ('-n'):
    ncpu = int(arg)

# Use umfpack for sequential and mumps for sequential/parallel
solver_default = ' -snes_monitor -snes_converged_reason -ksp_monitor -ksp_converged_reason '
if (ncpu == -1):
  solver = ' -pc_type lu -pc_factor_mat_solver_type umfpack -pc_factor_mat_ordering_type external'
  ncpu = 1
else:
  solver = ' -pc_type lu -pc_factor_mat_solver_type mumps'

for isim in range(0,nsim):
  test = itest[isim]
  subtest = isubtest[isim]
  L = iL[isim]
  # Ra = iRa[isim]
  eta0 = ieta0[isim]
  b = ib[isim]
  c = ic[isim]
  lbl = lbl_model[isim]
  nx = int(L*n/H)
  nz = n

  # Input file
  fname = 'out_convection_model'+lbl+'_n'+str(n)
  fname_out = 'out_decoupled_convection_model'+lbl+'_n'+str(n)
  fname_data = fname_out+'/data'
  try:
    os.mkdir(fname_out)
  except OSError:
    pass

  try:
    os.mkdir(fname_data)
  except OSError:
    pass

  # if (subtest>=20):
  #   dtmax /= Ra
  #   tmax /= Ra

  # Run test
  str1 = 'mpiexec -n '+str(ncpu)+' ../test_decoupled_convection.app '+solver+solver_default+ \
        ' -test '+str(test)+ \
        ' -eta0 '+str(eta0)+ \
        ' -tmax '+str(tmax)+ \
        ' -dtmax '+str(dtmax)+ \
        ' -tstep '+str(tstep_max)+ \
        ' -adv_scheme '+str(adv_scheme)+ \
        ' -ts_scheme '+str(ts_scheme)+ \
        ' -output_file '+fname+ \
        ' -output_dir '+fname_data+\
        ' -tout '+str(tout)+ \
        ' -L '+str(L)+\
        ' -b '+str(b)+' -c '+str(c)+ \
        ' -nx '+str(nx)+' -nz '+str(nz)+' > '+fname_data+'/'+fname+'.out'
  print(str1)
  os.system(str1)

  # Parse log file
  fout1 = fname_data+'/'+fname+'.out'

  f = open(fout1, 'r')
  i0=0
  for line in f:
    if '# TIMESTEP' in line:
        i0+=1
  f.close()
  tstep = i0

  # Plot timesteps
  for istep in range(0,tstep,tout):
    # Load python module describing data
    if (istep < 10): ft = '_ts00'+str(istep)
    if (istep >= 10) & (istep < 99): ft = '_ts0'+str(istep)
    if (istep >= 100): ft = '_ts'+str(istep)

    # Plot solution
    # Load python module describing data
    fout = fname+'_PV_m'+str(ts_scheme)+ft
    # imod = importlib.import_module(fout)
    spec = importlib.util.spec_from_file_location(fout,fname_data+'/'+fout+'.py')
    imod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imod)

    # Load data
    data = imod._PETScBinaryLoad()
    imod._PETScBinaryLoadReportNames(data)

    # Get data
    m = data['Nx'][0]
    n = data['Ny'][0]
    xv = data['x1d_vertex']
    yv = data['y1d_vertex']
    xc = data['x1d_cell']
    yc = data['y1d_cell']
    vxface = data['X_face_x']
    vyface = data['X_face_y']
    xcenter = data['X_cell']

    # Compute the DOF count on each stratum (face,element)
    facedof_x = int(len(vxface)/((m+1)*n))
    facedof_y = int(len(vyface)/(m*(n+1)))
    celldof   = int(len(xcenter)/(m*n))

    # Prepare cell center velocities
    vxface = vxface.reshape(n  ,m+1)
    vyface = vyface.reshape(n+1,m  )

    # Compute the cell center values from the face data by averaging neighbouring faces
    vxc = np.zeros( [n , m] )
    vyc = np.zeros( [n , m] )
    for i in range(0,m):
      for j in range(0,n):
        vxc[j][i] = 0.5 * (vxface[j][i+1] + vxface[j][i])
        vyc[j][i] = 0.5 * (vyface[j+1][i] + vyface[j][i])

    # Prepare center values
    p = xcenter[0::celldof]

    # Load temperature
    fout = fname+'_T_m'+str(ts_scheme)+ft
    # imod = importlib.import_module(fout)
    spec = importlib.util.spec_from_file_location(fout,fname_data+'/'+fout+'.py')
    imod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imod)

    # Load data
    data = imod._PETScBinaryLoad()
    imod._PETScBinaryLoadReportNames(data)
    xcenter = data['X_cell']

    T = xcenter[0::celldof]

    # Open a figure
    fig, axs = plt.subplots(1, 2,figsize=(12,6))

    nind = 2

    ax1 = axs[0]
    contours = ax1.contour(xc,yc,p.reshape(n,m), colors='white',linestyles='solid',linewidths=0.5)
    ax1.clabel(contours, contours.levels, inline=True, fontsize=8)
    im1 = ax1.imshow( p.reshape(n,m), extent=[min(xv), max(xv), min(yv), max(yv)],
                    origin='lower', interpolation='nearest' )
    Q = ax1.quiver( xc[::nind], yc[::nind], vxc[::nind,::nind], vyc[::nind,::nind], units='width', pivot='mid')
    # ax1.quiverkey(Q, X=0.25, Y=-0.2, U=10, label='Quiver length = 10 [cm/yr]', labelpos='E')
    cbar = fig.colorbar(im1,ax=ax1, shrink=0.75, label='P')
    ax1.axis('image')
    ax1.set_xlabel('x-dir')
    ax1.set_ylabel('z-dir')
    ax1.set_title('Pressure and velocity (Stokes)')

    ax2 = axs[1]
    contours = ax2.contour(xc,yc,T.reshape(n,m), colors='white',linestyles='solid',linewidths=0.5)
    ax2.clabel(contours, contours.levels, inline=True, fontsize=8)
    im2 = ax2.imshow( T.reshape(n,m), extent=[min(xv), max(xv), min(yv), max(yv)],
                    origin='lower', interpolation='nearest' )
    ax2.axis('image')
    cbar = fig.colorbar(im2,ax=ax2, shrink=0.75, label='T')
    ax2.set_xlabel('x-dir')
    # ax2.set_ylabel('z-dir')
    ax2.set_title('Temperature contours (AdvDiff)')

    plt.savefig(fname_out+'/'+fout+'.pdf')
    plt.close()

  # Parse log file
  fout1 = fname_data+'/'+fname+'.out'

  Nu = np.zeros(tstep)
  vrms = np.zeros(tstep)
  q1 = np.zeros(tstep)
  q2 = np.zeros(tstep)
  ts = np.zeros(tstep)

  # Open file 1 and read
  f = open(fout1, 'r')
  i0=0
  i1=0
  i2=0
  i3=0
  i4=0
  for line in f:
    if 'Nusselt' in line:
        Nu[i0] = float(line[23:42])
        i0+=1
    if 'Root-mean-squared' in line:
        vrms[i1] = float(line[37:56])
        i1+=1
    if 'Corner flux (down-left)' in line:
        q1[i2] = float(line[32:51])
        i2+=1
    if 'Corner flux (up-left)' in line:
        q2[i3] = float(line[30:49])
        i3+=1
    if '# TIME:' in line:
        ts[i4] = float(line[39:58])
        i4+=1
  f.close()

  # Benchmark values
  Nu_b   = [4.884409, 10.534095, 21.972465, 10.0660, 6.9299]
  vrms_b = [42.864947, 193.21454, 833.98977, 480.4334, 171.755]
  q1_b   = [8.059384, 19.079440, 45.96425, 45.96425, 18.4842] # up-left flux
  q2_b   = [0.588810, 0.722751, 0.877110, 1.00851, 0.17742] # up-right flux
  q3_b   = [8.059384, 19.079440, 45.96425, 26.8085, 14.1682] # down-right flux
  q4_b   = [0.588810, 0.722751, 0.877110, 0.497380, 0.61770] # down-left flux
  ibench = 0

  if (subtest==10):
    ibench = 0
  if (subtest==11):
    ibench = 1
  if (subtest==12):
    ibench = 2
  if (subtest==20):
    ibench = 3
  if (subtest==21):
    ibench = 4

  iNu = np.ones(tstep)*Nu_b[ibench]
  ivrms = np.ones(tstep)*vrms_b[ibench]
  iq1 = np.ones(tstep)*q1_b[ibench]
  iq2 = np.ones(tstep)*q2_b[ibench]
  iq3 = np.ones(tstep)*q3_b[ibench]
  iq4 = np.ones(tstep)*q4_b[ibench]

  # Plot diagnostics
  plt.figure(1,figsize=(12,6))

  plt.subplot(221)
  plt.grid(color='lightgray', linestyle=':')
  plt.plot(Nu,'k+--',label='Nu')
  plt.plot(iNu,'r--',label='benchmark')
  plt.ylabel('Nu',fontweight='bold',fontsize=12)
  plt.legend()

  plt.subplot(222)
  plt.grid(color='lightgray', linestyle=':')
  plt.plot(vrms,'k+--',label='vrms')
  plt.plot(ivrms,'r--',label='benchmark')
  plt.ylabel('vrms',fontweight='bold',fontsize=12)
  plt.legend()

  plt.subplot(223)
  plt.grid(color='lightgray', linestyle=':')
  plt.plot(q1,'k+--',label='q1')
  plt.plot(q2,'b+--',label='q2')
  plt.plot(iq1,'r--',label='benchmark')
  plt.plot(iq2,'r--')
  plt.xlabel('Time step',fontweight='bold',fontsize=12)
  plt.ylabel('q (Temp gradient)',fontweight='bold',fontsize=12)
  plt.legend(loc='upper right')

  plt.subplot(224)
  plt.grid(color='lightgray', linestyle=':')
  plt.plot(ts,'k+--',label='ts size')
  plt.xlabel('Time step',fontweight='bold',fontsize=12)
  plt.ylabel('Time step size (dt)',fontweight='bold',fontsize=12)

  plt.savefig(fname_out+'/'+fname+'.pdf')
  plt.close()

  os.system('rm -r '+fname_data+'/__pycache__')
