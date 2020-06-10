# ---------------------------------------
# MMS test for porosity evolution - verify coupled system for two-phase flow 
# Solves for coupled (P, v) and Q=(1-phi) evolution, where P-dynamic pressure, v-solid velocity, phi-porosity.
# ---------------------------------------

# Import modules
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
      nrm_v   = np.zeros(1)
      nrm_p   = np.zeros(1)
      err_sum = np.zeros(1)
      err_mms = np.zeros(1)
      dt      = np.zeros(1)
      hx      = np.zeros(1)
    else:
      nrm_v   = np.zeros(tstep)
      nrm_p   = np.zeros(tstep)
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
      if 'Velocity:' in line:
        nrm_v[i0] = float(line[20:38])
      if 'Pressure:' in line:
        nrm_p[i0] = float(line[20:38])
      if 'Porosity err-squared:' in line:
        err_sum[i0] = float(line[30:48])
        err_mms[i0] = float(line[55:73])
      if 'Grid info:' in line:
        hx[i0] = float(line[18:36])
      if 'TIME: time' in line:
        dt[i0] = float(line[39:57])
    f.close()

    return tstep, nrm_v, nrm_p, err_sum, err_mms, dt, hx
  except OSError:
    print('Cannot open:', fdir)
    return tstep

# ---------------------------------------
def plot_solution_mms_error(fname,*args):
  istep = 0
  nx    = 0
  if (len(args)==1): 
    istep = args[0]
  if (len(args)==2): 
    istep = args[0]
    nx    = args[1]

  if (istep < 10): ft = '_ts00'+str(istep)
  if (istep >= 10) & (istep < 99): ft = '_ts0'+str(istep)
  if (istep >= 100): ft = '_ts'+str(istep)

  # Load data
  fout = fname+'_PV'+ft
  imod = importlib.import_module(fout) # 1. Numerical solution
  data = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(data)

  mx = data['Nx'][0]
  mz = data['Ny'][0]
  xc = data['x1d_cell']
  zc = data['y1d_cell']
  xv = data['x1d_vertex']
  zv = data['y1d_vertex']
  vx = data['X_face_x']
  vz = data['X_face_y']
  p = data['X_cell']

  fout = fname+'_mms_PV'+ft
  imod = importlib.import_module(fout) # 2. MMS solution
  data = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(data)
  vx_mms = data['X_face_x']
  vz_mms = data['X_face_y']
  p_mms = data['X_cell']

  fout = fname+'_phi'+ft
  imod = importlib.import_module(fout)  # 3. porosity solution
  data = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(data)
  phi = data['X_cell']
  phi = 1.0 - phi # was solved for Q = 1-phi

  fout = fname+'_mms_phi'+ft
  imod = importlib.import_module(fout) # 4. MMS porosity solution
  data = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(data)
  phi_mms = data['X_cell']

  pmax = max(p_mms)
  pmin = min(p_mms)
  vxmax = max(vx_mms)
  vxmin = min(vx_mms)
  vzmax = max(vz_mms)
  vzmin = min(vz_mms)
  phimax = max(phi_mms)
  phimin = min(phi_mms)

  # Plot all fields - mms, solution and errors for P, ux, uz, phi
  fig = plt.figure(1,figsize=(12,16))
  cmaps='RdBu_r' 

  ax = plt.subplot(4,3,1)
  im = ax.imshow(phi_mms.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=phimin,vmax=phimax,cmap=cmaps)
  ax.set_title(r'a1) MMS $\phi$'+' timestep = '+str(istep))
  ax.set_ylabel('z')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(4,3,2)
  im = ax.imshow(phi.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=phimin,vmax=phimax,cmap=cmaps)
  ax.set_title(r'a2) Numerical $\phi$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(4,3,3)
  im = ax.imshow(phi_mms.reshape(mz,mx)-phi.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps)
  ax.set_title(r'a3) Error $\phi$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(4,3,4)
  im = ax.imshow(p_mms.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=pmin,vmax=pmax,cmap=cmaps)
  ax.set_title('b1) MMS P')
  ax.set_ylabel('z')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(4,3,5)
  im = ax.imshow(p.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=pmin,vmax=pmax,cmap=cmaps)
  ax.set_title('b2) Numerical P')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(4,3,6)
  im = ax.imshow(p_mms.reshape(mz,mx)-p.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps)
  ax.set_title('b3) Error P')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(4,3,7)
  im = ax.imshow(vx_mms.reshape(mz,mx+1),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=vxmin,vmax=vxmax,cmap=cmaps)
  ax.set_title('c1) MMS ux')
  ax.set_ylabel('z')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(4,3,8)
  im = ax.imshow(vx.reshape(mz,mx+1),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=vxmin,vmax=vxmax,cmap=cmaps)
  ax.set_title('c2) Numerical ux')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(4,3,9)
  im = ax.imshow(vx_mms.reshape(mz,mx+1)-vx.reshape(mz,mx+1),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps)
  ax.set_title('c3) Error ux')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(4,3,10)
  im = ax.imshow(vz_mms.reshape(mz+1,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=vzmin,vmax=vzmax,cmap=cmaps)
  ax.set_title('d1) MMS uz')
  ax.set_ylabel('z')
  ax.set_xlabel('x')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(4,3,11)
  im = ax.imshow(vz.reshape(mz+1,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=vzmin,vmax=vzmax,cmap=cmaps)
  ax.set_title('d2) Numerical uz')
  ax.set_xlabel('x')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(4,3,12)
  im = ax.imshow(vz_mms.reshape(mz+1,mx)-vz.reshape(mz+1,mx),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps)
  ax.set_title('d3) Error uz')
  ax.set_xlabel('x')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  plt.tight_layout() 
  fout = fname+ft+'_solution'+'_nx_'+str(nx)
  plt.savefig(fout+'.pdf')
  plt.close()

# ---------------------------------------
def plot_convergence_error_space(fname,hx,nrm_p,nrm_v,nrm_phi):
  hx_log    = np.log10(hx)
  nrm2phi_log = np.log10(nrm_phi)
  nrm2p_log   = np.log10(nrm_p)
  nrm2v_log   = np.log10(nrm_v)

  # Perform linear regression
  sl2phi, intercept, r_value, p_value, std_err = linregress(hx_log, nrm2phi_log)
  sl2p, intercept, r_value, p_value, std_err = linregress(hx_log, nrm2p_log)
  sl2v, intercept, r_value, p_value, std_err = linregress(hx_log, nrm2v_log)

  print('# --------------------------------------- #')
  print('# 1. MMS StokesDarcy2Field+Porosity convergence order (SPACE):')
  print('     v_slope = '+str(sl2v)+' P_slope = '+str(sl2p)+' phi_slope = '+str(sl2phi))

  plt.figure(1,figsize=(6,6))
  plt.grid(color='lightgray', linestyle=':')

  # plt.plot(hx,nrm_phi,'ko--',label=r'$\phi$ slope='+str(round(sl2phi,5)))
  plt.plot(hx,nrm_p,'bo--',label='P slope='+str(round(sl2p,5)))
  plt.plot(hx,nrm_v,'ro--',label='v slope='+str(round(sl2v,5)))

  plt.xscale("log")
  plt.yscale("log")

  plt.xlabel('log10(h)',fontweight='bold',fontsize=12)
  plt.ylabel(r'$E(\phi), E(P), E(v)$',fontweight='bold',fontsize=12)
  plt.legend()

  plt.savefig(fname+'_error_hx_L2.pdf')
  plt.close()

# ---------------------------------------
def plot_convergence_error_time(fname,dt_nrm,*args):
  nrm_phi1 = args[0]
  nrm_phi2 = args[1]
  nrm_phi3 = args[2]
  nrm_p1 = args[3]
  nrm_p2 = args[4]
  nrm_p3 = args[5]
  nrm_v1 = args[6]
  nrm_v2 = args[7]
  nrm_v3 = args[8]

  dt_log    = np.log10(dt_nrm)
  nrm2phi1_log = np.log10(nrm_phi1)
  nrm2phi2_log = np.log10(nrm_phi2)
  nrm2phi3_log = np.log10(nrm_phi3)
  nrm2p1_log = np.log10(nrm_p1)
  nrm2p2_log = np.log10(nrm_p2)
  nrm2p3_log = np.log10(nrm_p3)
  nrm2v1_log = np.log10(nrm_v1)
  nrm2v2_log = np.log10(nrm_v2)
  nrm2v3_log = np.log10(nrm_v3)

  # Perform linear regression
  sl2phi1, intercept, r_value, p_value, std_err = linregress(dt_log, nrm2phi1_log)
  sl2phi2, intercept, r_value, p_value, std_err = linregress(dt_log, nrm2phi2_log)
  sl2phi3, intercept, r_value, p_value, std_err = linregress(dt_log, nrm2phi3_log)
  sl2p1, intercept, r_value, p_value, std_err = linregress(dt_log, nrm2p1_log)
  sl2p2, intercept, r_value, p_value, std_err = linregress(dt_log, nrm2p2_log)
  sl2p3, intercept, r_value, p_value, std_err = linregress(dt_log, nrm2p3_log)
  sl2v1, intercept, r_value, p_value, std_err = linregress(dt_log, nrm2v1_log)
  sl2v2, intercept, r_value, p_value, std_err = linregress(dt_log, nrm2v2_log)
  sl2v3, intercept, r_value, p_value, std_err = linregress(dt_log, nrm2v3_log)

  print('# --------------------------------------- #')
  print('# 2. MMS StokesDarcy2Field+Porosity convergence order (TIME):')
  print('     Forward Euler  : phi_slope = '+str(sl2phi1)+' p_slope = '+str(sl2p1)+' v_slope = '+str(sl2v1))
  print('     Backward Euler : phi_slope = '+str(sl2phi2)+' p_slope = '+str(sl2p2)+' v_slope = '+str(sl2v2))
  print('     Crank-Nicholson: phi_slope = '+str(sl2phi3)+' p_slope = '+str(sl2p3)+' v_slope = '+str(sl2v3))

  plt.figure(1,figsize=(6,6))
  plt.grid(color='lightgray', linestyle=':')
  plt.plot(dt_nrm,nrm_phi1,'ko--',label=r'$\phi$ fe slope='+str(round(sl2phi1,5)))
  plt.plot(dt_nrm,nrm_phi2,'r+--',label=r'$\phi$ be slope='+str(round(sl2phi2,5)))
  plt.plot(dt_nrm,nrm_phi3,'b*--',label=r'$\phi$ cn slope='+str(round(sl2phi3,5)))
  # plt.plot(dt_nrm,nrm_p1,'ro--',label=r'P fe slope='+str(round(sl2p1,5)))
  # plt.plot(dt_nrm,nrm_p2,'r+--',label=r'P be slope='+str(round(sl2p2,5)))
  # plt.plot(dt_nrm,nrm_p3,'r*--',label=r'P cn slope='+str(round(sl2p3,5)))
  # plt.plot(dt_nrm,nrm_v1,'bo--',label=r'v fe slope='+str(round(sl2v1,5)))
  # plt.plot(dt_nrm,nrm_v2,'b+--',label=r'v be slope='+str(round(sl2v2,5)))
  # plt.plot(dt_nrm,nrm_v3,'b*--',label=r'v cn slope='+str(round(sl2v3,5)))
  plt.xscale("log")
  plt.yscale("log")

  plt.xlabel('Time step size',fontweight='bold',fontsize=12)
  plt.ylabel(r'$E(\phi), E(P), E(v)$',fontweight='bold',fontsize=12)
  plt.legend()

  plt.savefig(fname+'_error_dt_L2.pdf')
  plt.close()

# ---------------------------------------
def test1_space(fname,n):
  # Prepare errors and convergence
  nrm_p   = np.zeros(len(n))
  nrm_v   = np.zeros(len(n))
  nrm_phi = np.zeros(len(n))
  hx      = np.zeros(len(n))

  tout  = 1
  dtmax = 1e-8
  tmax  = 1e-1
  tstep_max = 0
  adv_scheme = 1
  ts_scheme  = 2
  m = 1.0

  # Run simulations
  for i in range(len(n)):
    # Create output filename
    nx = n[i]
    fout = fname+'_'+str(nx)+'.out'

    # Run with different resolutions - 1 timestep
    str1 = '../test_stokesdarcy2field_mms_porosity.app -pc_type lu -pc_factor_mat_solver_type umfpack'+ \
        ' -dtmax '+str(dtmax)+ \
        ' -tmax '+str(tmax)+ \
        ' -tstep '+str(tstep_max)+ \
        ' -adv_scheme '+str(adv_scheme)+ \
        ' -ts_scheme '+str(ts_scheme)+ \
        ' -output_file '+fname+ \
        ' -tout '+str(tout)+ \
        ' -m '+str(m)+ \
        ' -nx '+str(nx)+' -nz '+str(nx)+' > '+fout
    print(str1)
    os.system(str1)

    # Parse variables
    tstep, nrm_v_num, nrm_p_num, err_sum, err_mms, dt_num, hx_num = parse_log_file(fout)
    nrm_p[i]   = nrm_p_num
    nrm_v[i]   = nrm_v_num
    nrm_phi[i] = err_sum**0.5
    hx[i]    = hx_num

    # Plot solution and error
    plot_solution_mms_error(fname,0,nx)

  # Convergence plot
  plot_convergence_error_space(fname,hx,nrm_p,nrm_v,nrm_phi)

# ---------------------------------------
def test2_time(fname,dt,tend,n):
  # Prepare errors and convergence
  nrm_phi1 = np.zeros(len(dt)) # phi - fe
  nrm_phi2 = np.zeros(len(dt)) # be
  nrm_phi3 = np.zeros(len(dt)) # cn
  nrm_p1 = np.zeros(len(dt)) # P - fe
  nrm_p2 = np.zeros(len(dt)) # be
  nrm_p3 = np.zeros(len(dt)) # cn
  nrm_v1 = np.zeros(len(dt)) # v - fe
  nrm_v2 = np.zeros(len(dt)) # be
  nrm_v3 = np.zeros(len(dt)) # cn
  dt_nrm = np.zeros(len(dt))

  tout  = 10
  tstep_max = 10000000# max no of timesteps
  adv_scheme = 1 # fromm
  nts_scheme = [0, 1, 2]
  m = 2.0

  # Run and plot simulations
  for ts_scheme in nts_scheme:
    nrm_phi = np.zeros(len(dt)) # dummy
    nrm_p = np.zeros(len(dt)) # dummy
    nrm_v = np.zeros(len(dt)) # dummy
    for i in range(len(dt)):
      dt_string = str(dt[i])
      dt_string = dt_string.replace('.','-')
      dtmax = 10**(dt[i])

      # Create output filename
      fname1 = fname+'_ts'+str(ts_scheme)+'_dt'+dt_string

      # Run test
      str1 = '../test_stokesdarcy2field_mms_porosity.app -pc_type lu -pc_factor_mat_solver_type umfpack'+ \
        ' -dtmax '+str(dtmax)+ \
        ' -tmax '+str(tend)+ \
        ' -e3 '+str(-1.0)+ \
        ' -tstep '+str(tstep_max)+ \
        ' -adv_scheme '+str(adv_scheme)+ \
        ' -ts_scheme '+str(ts_scheme)+ \
        ' -output_file '+fname1+ \
        ' -tout '+str(tout)+ \
        ' -m '+str(m)+ \
        ' -nx '+str(n)+' -nz '+str(n)+' > '+fname1+'.out'
      print(str1)
      os.system(str1)

      # Parse log file and calculate errors
      fout = fname1+'.out'
      tstep, nrm_v_num, nrm_p_num, err_sum, err_mms, dt_num, hx_num = parse_log_file(fout)
      dt_nrm[i] = 10**(dt[i])

      sum_phi = 0.0 
      sum_p   = 0.0 
      sum_v   = 0.0 
      for istep in range(tstep):
        sum_phi += err_sum[istep]*dt_num[istep]
        sum_p   += nrm_p_num[istep]**2*dt_num[istep]
        sum_v   += nrm_v_num[istep]**2*dt_num[istep]
        # sum_Q += err_sum[istep]/err_mms[istep]*dt_num[istep]

      nrm_phi[i] = sum_phi**0.5
      nrm_p[i]   = sum_p**0.5
      nrm_v[i]   = sum_v**0.5

      # Plot solution for every timestep
      for istep in range(0,tstep,tout):
        plot_solution_mms_error(fname1,istep,n)

    # Save errors
    if   (ts_scheme==0): 
      nrm_phi1 = nrm_phi
      nrm_p1   = nrm_p
      nrm_v1   = nrm_v
    elif (ts_scheme==1): 
      nrm_phi2 = nrm_phi
      nrm_p2   = nrm_p
      nrm_v2   = nrm_v
    else: 
      nrm_phi3 = nrm_phi
      nrm_p3   = nrm_p
      nrm_v3   = nrm_v

  # Plot convergence
  plot_convergence_error_time(fname,dt_nrm,nrm_phi1,nrm_phi2,nrm_phi3,nrm_p1,nrm_p2,nrm_p3,nrm_v1,nrm_v2,nrm_v3)


# ---------------------------------------
# Main script - tests
# ---------------------------------------
print('# --------------------------------------- #')
print('# MMS tests for coupled Stokes-Darcy porosity evolution ')
print('# --------------------------------------- #')

# 1. Spatial errors 
fname = 'out_darcyporosity_01space'
n = [200, 250, 300, 350, 400] # 300-500
test1_space(fname,n)

# # 2. Temporal errors 
fname = 'out_darcyporosity_02time'
dt   = [-6, -5.5, -5, -4.5, -4]
n    = 200
tend = 5e-4 # 1e-3
test2_time(fname,dt,tend,n)

os.system('rm -r __pycache__')