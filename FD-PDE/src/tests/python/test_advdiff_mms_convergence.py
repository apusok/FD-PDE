# ---------------------------------------
# MMS test for ADVDIFF - time convergence 
# ---------------------------------------

# ---------------------------------------
# Import modules
# ---------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
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

    # Norm variables
    if (tstep == 0): # steady-state
      err_sum = np.zeros(1)
      err_mms = np.zeros(1)
      dt      = np.zeros(1)
      hx      = np.zeros(1)
    else:
      err_sum = np.zeros(tstep)
      err_mms = np.zeros(tstep)
      dt      = np.zeros(tstep)
      hx      = np.zeros(tstep)

    # Parse output and save norm info
    f = open(fname, 'r')
    i0=-1
    for line in f:
      if '# TIMESTEP' in line:
          i0+=1
      if 'L2 square:' in line:
          err_sum[i0] = float(line[19:37])
          err_mms[i0] = float(line[43:61])
      if 'Grid info:' in line:
          hx[i0] = float(line[18:36])
      if 'TIME: time' in line:
          dt[i0] = float(line[39:57])
    f.close()

    return tstep, err_sum, err_mms, dt, hx
  except OSError:
    print('Cannot open:', fdir)
    return tstep

# ---------------------------------------
def plot_solution_mms_error(fnum,fmms,*args):
  istep = 0
  nx    = 0
  adv_scheme = 99
  if (len(args)==1): 
    istep = args[0]
  if (len(args)==2): 
    istep = args[0]
    nx    = args[1]
  if (len(args)==3): 
    istep = args[0]
    nx    = args[1]
    adv_scheme = args[2]

  # Load data
  imod = importlib.import_module(fnum) # 1. Numerical solution
  data = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(data)

  mx = data['Nx'][0]
  mz = data['Ny'][0]
  xc = data['x1d_cell']
  zc = data['y1d_cell']
  Q = data['X_cell']

  imod = importlib.import_module(fmms) # 2. MMS solution
  data = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(data)
  Q_mms = data['X_cell']

  # Plot figure
  fig, axs = plt.subplots(1, 2,figsize=(18,6))
  cmaps='RdBu_r' 

  ax = plt.subplot(131)
  im = ax.imshow(Q_mms.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps)
  ax.set_title('a) MMS solution '+' timestep = '+str(istep))
  ax.set_xlabel('x')
  ax.set_ylabel('z')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(132)
  im = ax.imshow(Q.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps)
  ax.set_title('b) Numerical solution')
  ax.set_xlabel('x')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(133)
  im = ax.imshow(Q_mms.reshape(mz,mx)-Q.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps)
  ax.set_title('c) Error')
  ax.set_xlabel('x')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  fout = fnum+'_solution'+'_adv_'+str(adv_scheme)+'_nx_'+str(nx)
  plt.savefig(fout+'.pdf')
  plt.close()

# ---------------------------------------
def plot_convergence_error_space(fname,hx,*args):
  nrm_Q = args[0]
  hx_log    = np.log10(hx)
  nrm2Q_log = np.log10(nrm_Q)

  # Perform linear regression
  sl2Q, intercept, r_value, p_value, std_err = linregress(hx_log, nrm2Q_log)
  print('# --------------------------------------- #')
  print('# MMS ADVDIFF convergence order: '+fname)
  

  plt.figure(1,figsize=(6,6))
  plt.grid(color='lightgray', linestyle=':')

  if (len(args)==1): 
    plt.plot(hx,nrm_Q,'ko--',label='Q slope='+str(round(sl2Q,5)))
    print('     Q_slope = '+str(sl2Q))

  if (len(args)>1): 
    nrm_Q1 = args[1]
    nrm_Q2 = args[2]
    nrm2Q1_log = np.log10(nrm_Q1)
    nrm2Q2_log = np.log10(nrm_Q2)
    sl2Q1, intercept, r_value, p_value, std_err = linregress(hx_log, nrm2Q1_log)
    sl2Q2, intercept, r_value, p_value, std_err = linregress(hx_log, nrm2Q2_log)

    plt.plot(hx,nrm_Q,'ko--',label='Q upwind slope='+str(round(sl2Q,5)))
    plt.plot(hx,nrm_Q1,'ro--',label='Q upwind2 slope='+str(round(sl2Q1,5)))
    plt.plot(hx,nrm_Q2,'bo--',label='Q fromm slope='+str(round(sl2Q2,5)))
    print('     Q1_slope = '+str(sl2Q)+' Q2_slope = '+str(sl2Q1)+' Q3_slope = '+str(sl2Q2))

  plt.xscale("log")
  plt.yscale("log")

  plt.xlabel('log10(h)',fontweight='bold',fontsize=12)
  plt.ylabel(r'$E(Q)$',fontweight='bold',fontsize=12)
  plt.legend()

  plt.savefig(fname+'_error_hx_L2.pdf')
  plt.close()

# ---------------------------------------
def plot_convergence_error_time(fname,dt_nrm,*args):
  nrm_Q1 = args[0]
  nrm_Q2 = args[1]
  nrm_Q3 = args[2]

  dt_log    = np.log10(dt_nrm)
  nrm2Q1_log = np.log10(nrm_Q1)
  nrm2Q2_log = np.log10(nrm_Q2)
  nrm2Q3_log = np.log10(nrm_Q3)

  # Perform linear regression
  sl2Q1, intercept, r_value, p_value, std_err = linregress(dt_log, nrm2Q1_log)
  sl2Q2, intercept, r_value, p_value, std_err = linregress(dt_log, nrm2Q2_log)
  sl2Q3, intercept, r_value, p_value, std_err = linregress(dt_log, nrm2Q3_log)
  print('# --------------------------------------- #')
  print('# MMS ADVDIFF convergence order:'+fname)
  print('     Q_slope (forward Euler)  = '+str(sl2Q1))
  print('     Q_slope (backward Euler) = '+str(sl2Q2))
  print('     Q_slope (Crank-Nicholson)= '+str(sl2Q3))

  plt.figure(1,figsize=(6,6))
  plt.grid(color='lightgray', linestyle=':')
  plt.plot(dt_nrm,nrm_Q1,'ko--',label='Q fe slope='+str(round(sl2Q1,5)))
  plt.plot(dt_nrm,nrm_Q2,'ro--',label='Q be slope='+str(round(sl2Q2,5)))
  plt.plot(dt_nrm,nrm_Q3,'bo--',label='Q cn slope='+str(round(sl2Q3,5)))
  plt.xscale("log")
  plt.yscale("log")

  plt.xlabel('Time step size',fontweight='bold',fontsize=12)
  plt.ylabel(r'$E(Q)$',fontweight='bold',fontsize=12)
  plt.legend()

  plt.savefig(fname+'_error_dt_L2.pdf')
  plt.close()

# ---------------------------------------
def test1_diffusion_space(fname,n):
  # Prepare errors and convergence
  nrm_Q = np.zeros(len(n))
  hx    = np.zeros(len(n))

  # Run simulations
  for i in range(len(n)):
    # Create output filename
    nx = n[i]
    fout = fname+'_'+str(nx)+'.out'

    # Run with different resolutions
    str1 = '../test_advdiff_mms_convergence.app -pc_type lu -pc_factor_mat_solver_type umfpack -test 1 -output_file '+fname+' -nx '+str(nx)+' -nz '+str(nx)+' > '+fout
    print(str1)
    os.system(str1)

    # Parse variables
    tstep, err_sum, err_mms, dt_num, hx_num = parse_log_file(fout)
    nrm_Q[i] = err_sum**0.5
    hx[i]    = hx_num

    # Plot solution and error
    plot_solution_mms_error(fname,fname+'_mms',0,nx)

  # Convergence plot
  plot_convergence_error_space(fname,hx,nrm_Q)

# ---------------------------------------
def test2_advection_diffusion_space(fname,n):
  # Prepare errors and convergence
  nrm_Q1 = np.zeros(len(n)) # upwind
  nrm_Q2 = np.zeros(len(n)) # upwind2
  nrm_Q3 = np.zeros(len(n)) # fromm
  hx    = np.zeros(len(n))

  nadv_scheme = [0, 1, 2]
  # Run simulations

  for adv_scheme in nadv_scheme:
    nrm_Q = np.zeros(len(n)) # dummy
    for i in range(len(n)):
      # Create output filename
      nx = n[i]
      fout = fname+'_adv'+str(adv_scheme)+'_'+str(nx)+'.out'

      # Run with different resolutions and advection schemes
      str1 = '../test_advdiff_mms_convergence.app -pc_type lu -pc_factor_mat_solver_type umfpack -test 2 -output_file '+fname+' -adv_scheme '+str(adv_scheme)+' -nx '+str(nx)+' -nz '+str(nx)+' > '+fout
      print(str1)
      os.system(str1)

      # Parse variables
      tstep, err_sum, err_mms, dt_num, hx_num = parse_log_file(fout)
      nrm_Q[i] = err_sum**0.5
      hx[i]    = hx_num

      # Plot solution and error
      plot_solution_mms_error(fname,fname+'_mms',0,nx,adv_scheme)

    print(nrm_Q)
    # Save errors
    if   (adv_scheme==0): nrm_Q1 = nrm_Q
    elif (adv_scheme==1): nrm_Q2 = nrm_Q
    else:                 nrm_Q3 = nrm_Q

  # Convergence plot
  plot_convergence_error_space(fname,hx,nrm_Q1,nrm_Q2,nrm_Q3)

# ---------------------------------------
def test3_timediff(fname,dt,tend,n):
  tout = 10 # output every x timesteps
  nts_scheme = [1] #[0, 1, 2]
  tstep_max = 10000000# max no of timesteps
  tmax = 1e-3

  # Prepare errors and convergence
  nrm_Q = np.zeros(len(dt)) # dummy
  nrm_Q1 = np.zeros(len(dt)) # fe
  nrm_Q2 = np.zeros(len(dt)) # be
  nrm_Q3 = np.zeros(len(dt)) # cn
  dt_nrm = np.zeros(len(dt))

  # Run and plot simulations
  for ts_scheme in nts_scheme:
    for i in range(len(dt)):
      dt_string = str(dt[i])
      dt_string = dt_string.replace('.','-')
      dtmax = 10**(dt[i])

      # Create output filename
      fname1 = fname+'_ts'+str(ts_scheme)+'_dt'+dt_string

      # Run test
      str1 = '../test_advdiff_mms_convergence.app -pc_type lu -pc_factor_mat_solver_type umfpack -test 3'+ \
            ' -dtmax '+str(dtmax)+ \
            ' -tmax '+str(tmax)+ \
            ' -tstep '+str(tstep_max)+ \
            ' -ts_scheme '+str(ts_scheme)+ \
            ' -output_file '+fname1+ \
            ' -tout '+str(tout)+ \
            ' -nx '+str(n)+' -nz '+str(n)+' > '+fname1+'.out'
      print(str1)
      os.system(str1)

      # Parse log file and calculate errors
      fout = fname1+'.out'
      tstep, err_sum, err_mms, dt_num, hx = parse_log_file(fout)
      dt_nrm[i] = 10**(dt[i])

      sum_Q = 0.0 
      for istep in range(tstep):
        sum_Q += err_sum[istep]*dt_num[istep]

      nrm_Q[i] = sum_Q**0.5

      # Plot solution for every timestep
      for istep in range(0,tstep,tout):
        if (istep < 10): ft = '_ts00'+str(istep)
        if (istep >= 10) & (istep < 99): ft = '_ts0'+str(istep)
        if (istep >= 100): ft = '_ts'+str(istep)
        plot_solution_mms_error(fname1+ft,fname1+'_mms'+ft,istep,n)
    
    # Save errors
    if   (ts_scheme==0): nrm_Q1 = nrm_Q
    elif (ts_scheme==1): nrm_Q2 = nrm_Q
    else:                nrm_Q3 = nrm_Q

  # Plot convergence
  # plot_convergence_error_time(fname,dt_nrm,nrm_Q1,nrm_Q2,nrm_Q3)

# ---------------------------------------
def test5_advdiff(fname,dt,n):
  tout = 10 # output every x timesteps
  ts_scheme = 2
  adv_scheme = 1
  tstep_max = 1# max no of timesteps
  tmax = 1e-3

  # Prepare errors and convergence
  nrm_Q1 = np.zeros(len(dt))
  dt_nrm = np.zeros(len(dt))

  # Run and plot simulations
  for i in range(len(dt)):
    dt_string = str(dt[i])
    dt_string = dt_string.replace('.','-')
    dtmax = 10**(dt[i])

    # Create output filename
    fname1 = fname+'_dt'+dt_string

    # Run test
    str1 = '../test_advdiff_mms_convergence.app -pc_type lu -pc_factor_mat_solver_type umfpack'+ \
          ' -dtmax '+str(dtmax)+ \
          ' -tmax '+str(tmax)+ \
          ' -tstep '+str(tstep_max)+ \
          ' -adv_scheme '+str(adv_scheme)+ \
          ' -ts_scheme '+str(ts_scheme)+ \
          ' -output_file '+fname1+ \
          ' -tout '+str(tout)+ \
          ' -nx '+str(n)+' -nz '+str(n)+' > '+fname1+'.out'
    print(str1)
    os.system(str1)

    # Parse log file and calculate errors
    fout = fname1+'.out'
    tstep, err_sum, err_mms, dt_num, hx = parse_log_file(fout)
    dt_nrm[i] = 10**(dt[i])

    sum_Q1 = 0.0 
    for istep in range(tstep):
      sum_Q1 += err_sum[istep]*dt_num[istep]

    nrm_Q1[i] = sum_Q1**0.5

    # Plot solution for every timestep
    for istep in range(0,tstep,tout):
      if (istep < 10): ft = '_ts00'+str(istep)
      if (istep >= 10) & (istep < 99): ft = '_ts0'+str(istep)
      if (istep >= 100): ft = '_ts'+str(istep)
      plot_solution_mms_error(fname1+ft,fname1+'_mms'+ft,istep,n)

  # Plot convergence
  plot_convergence_error_time(fname,dt_nrm,nrm_Q1)

# ---------------------------------------
# Main script - tests
# ---------------------------------------
print('# --------------------------------------- #')
print('# MMS tests for ADVDIFF convergence order ')
print('# --------------------------------------- #')

# 1. Steady-state diffusion
fname = 'out_mms_advdiff_01_diff'
n = [25, 40, 50, 80, 100, 125, 150, 200, 300]
test1_diffusion_space(fname,n)

# 2. Steady-state diffusion-advection
fname = 'out_mms_advdiff_02_advdiff'
n = [25, 40, 50, 80, 100, 125, 150, 200, 300]
test2_advection_diffusion_space(fname,n)

# 3. Time-dependent diffusion
fname = 'out_mms_advdiff_03_timediff'
dt   = [-5, -4.5, -4]
n    = 100
tend = 1e-3 
test3_timediff(fname,dt,tend,n)

# dt = [-6, -5.5, -5, -4.5, -4] 
# dt = [-5, -4, -3] 
# n = 100
# test5_advdiff(fname,dt,n)

os.system('rm -r __pycache__')