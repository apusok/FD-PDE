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

# Input file
fname = 'out_darcyporosity'

print('# --------------------------------------- #')
print('# MMS test for coupled Stokes-Darcy porosity evolution ')
print('# --------------------------------------- #')

# Parameters
dt = [-6, -5.5, -5, -4.5, -4] 
# dt = [-5, -4, -3] 

n = 100
nx = n
tout = 10 # 50
ts_scheme = 2
adv_scheme = 1
# dtmax = 1e-3 #1e-4
tstep_max = 100000000 #100 # max no of timesteps
tmax = 1e-3
m = 1

cmaps1='RdBu_r' 
cmaps='RdBu_r' 

# ---------------------------------------
# Run and plot simulations
# ---------------------------------------
for idtmax in dt:

  dt_string = str(idtmax)
  dt_string = dt_string.replace('.','-')

  dtmax = 10**(idtmax)
  # Create output filename
  fname1 = fname+'_dt'+dt_string

  # Run test
  str1 = '../test_stokesdarcy2field_mms_porosity.app -pc_type lu -pc_factor_mat_solver_type umfpack'+ \
        ' -dtmax '+str(dtmax)+ \
        ' -tmax '+str(tmax)+ \
        ' -tstep '+str(tstep_max)+ \
        ' -adv_scheme '+str(adv_scheme)+ \
        ' -ts_scheme '+str(ts_scheme)+ \
        ' -output_file '+fname1+ \
        ' -tout '+str(tout)+ \
        ' -m '+str(m)+ \
        ' -nx '+str(n)+' -nz '+str(n)+' > '+fname1+'.out'
  print(str1)
  os.system(str1)

  # Parse log file - for timesteps
  fout = fname1+'.out'

  f = open(fout, 'r')
  i0=0
  for line in f:
    if '# TIMESTEP' in line:
        i0+=1
  f.close()
  tstep = i0

  # Plot solution for every timestep
  for istep in range(0,tstep,tout):

    # Load python module describing data
    if (istep < 10): ft = '_ts00'+str(istep)
    if (istep >= 10) & (istep < 99): ft = '_ts0'+str(istep)
    if (istep >= 100): ft = '_ts'+str(istep)

    # Load data
    # 1. PV solution
    fout = fname1+'_PV'+ft 
    imod = importlib.import_module(fout)
    data = imod._PETScBinaryLoad()
    imod._PETScBinaryLoadReportNames(data)

    mx = data['Nx'][0]
    mz = data['Ny'][0]
    xv = data['x1d_vertex']
    zv = data['y1d_vertex']
    xc = data['x1d_cell']
    zc = data['y1d_cell']
    vx = data['X_face_x']
    vz = data['X_face_y']
    p = data['X_cell']

    # 2. MMS PV solution
    fout = fname1+'_mms_PV'+ft 
    imod = importlib.import_module(fout)
    data = imod._PETScBinaryLoad()
    imod._PETScBinaryLoadReportNames(data)

    vx_mms = data['X_face_x']
    vz_mms = data['X_face_y']
    p_mms = data['X_cell']

    # 3. porosity solution
    fout = fname1+'_phi'+ft 
    imod = importlib.import_module(fout)
    data = imod._PETScBinaryLoad()
    imod._PETScBinaryLoadReportNames(data)

    phi = data['X_cell']
    phi = 1.0 - phi # actually was solved for Q = 1-phi

    # 4. MMS porosity solution
    fout = fname1+'_mms_phi'+ft 
    imod = importlib.import_module(fout)
    data = imod._PETScBinaryLoad()
    imod._PETScBinaryLoadReportNames(data)

    phi_mms = data['X_cell']

    # Prepare variables
    vxface = vx.reshape(mz,mx+1)
    vzface = vz.reshape(mz+1,mx)
    vxface_err = vx_mms.reshape(mz,mx+1)-vx.reshape(mz,mx+1)
    vzface_err = vz_mms.reshape(mz+1,mx)-vz.reshape(mz+1,mx)

    # Compute the cell center values from the face data by averaging neighbouring faces
    vxc = np.zeros( [mz,mx] )
    vzc = np.zeros( [mz,mx] )
    vxc_err = np.zeros( [mz,mx] )
    vzc_err = np.zeros( [mz,mx] )

    for i in range(0,mx):
      for j in range(0,mz):
        vxc[j][i] = 0.5 * (vxface[j][i+1] + vxface[j][i])
        vzc[j][i] = 0.5 * (vzface[j+1][i] + vzface[j][i])
        vxc_err[j][i] = 0.5 * (vxface_err[j][i+1] + vxface_err[j][i])
        vzc_err[j][i] = 0.5 * (vzface_err[j+1][i] + vzface_err[j][i])

    # Figure 1
    fig, axs = plt.subplots(1, 2,figsize=(12,6))
    nind = int(nx/20)

    ax = plt.subplot(121)
    im = ax.imshow(phi.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=0, vmax=0.1,cmap=cmaps1)
    Q  = ax.quiver( xc[::nind], zc[::nind], vxc[::nind,::nind], vzc[::nind,::nind], units='width', pivot='mid' )
    ax.set_title('Numerical solution: '+r'$\phi$'+', v'+' timestep = '+str(istep))
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    cbar = fig.colorbar(im,ax=ax, shrink=0.75)
    # cbar = fig.colorbar(im,ax=ax, ticks=np.linspace(0.0,0.09, 5), shrink=0.75)

    ax = plt.subplot(122)
    im = ax.imshow(phi_mms.reshape(mz,mx)-phi.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps1)
    Q  = ax.quiver( xc[::nind], zc[::nind], vxc_err[::nind,::nind], vzc_err[::nind,::nind], units='width', pivot='mid' )
    ax.set_title('Error: '+r'$\phi$'+', v')
    ax.set_xlabel('x')
    cbar = fig.colorbar(im,ax=ax, shrink=0.75)
    # cbar = fig.colorbar(im,ax=ax, shrink=0.75)

    fout = fname1+'_solution'+ft 
    plt.savefig(fout+'.pdf')
    # plt.savefig(fout+'.png')
    plt.close()

    # Figure 2 - Plot all fields - mms, solution and errors for P, ux, uz, phi
    fig = plt.figure(1,figsize=(16,12))

    ax = plt.subplot(3,4,1)
    im = ax.imshow(phi_mms.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps)
    ax.set_title('a1) MMS phi', fontweight='bold')
    ax.set_ylabel('z')
    cbar = fig.colorbar(im,ax=ax, shrink=0.75)

    ax = plt.subplot(3,4,2)
    im = ax.imshow(p_mms.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps)
    ax.set_title('b1) MMS P', fontweight='bold')
    cbar = fig.colorbar(im,ax=ax, shrink=0.75)

    ax = plt.subplot(3,4,3)
    im = ax.imshow(vx_mms.reshape(mz,mx+1),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps)
    ax.set_title('c1) MMS ux', fontweight='bold')
    cbar = fig.colorbar(im,ax=ax, shrink=0.75)

    ax = plt.subplot(3,4,4)
    im = ax.imshow(vz_mms.reshape(mz+1,mx),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps)
    ax.set_title('d1) MMS uz', fontweight='bold')
    cbar = fig.colorbar(im,ax=ax, shrink=0.75)

    ax = plt.subplot(3,4,5)
    im = ax.imshow(phi.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps)
    ax.set_title('a2) Numerical phi', fontweight='bold')
    ax.set_ylabel('z')
    cbar = fig.colorbar(im,ax=ax, shrink=0.75)

    ax = plt.subplot(3,4,6)
    im = ax.imshow(p.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps)
    ax.set_title('b2) Numerical P', fontweight='bold')
    cbar = fig.colorbar(im,ax=ax, shrink=0.75)

    ax = plt.subplot(3,4,7)
    im = ax.imshow(vx.reshape(mz,mx+1),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps)
    ax.set_title('c2) Numerical ux', fontweight='bold')
    cbar = fig.colorbar(im,ax=ax, shrink=0.75)

    ax = plt.subplot(3,4,8)
    im = ax.imshow(vz.reshape(mz+1,mx),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps)
    ax.set_title('d2) Numerical uz', fontweight='bold')
    cbar = fig.colorbar(im,ax=ax, shrink=0.75)

    ax = plt.subplot(3,4,9)
    im = ax.imshow(phi_mms.reshape(mz,mx)-phi.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps)
    ax.set_title('a3) Error phi', fontweight='bold')
    ax.set_ylabel('z')
    cbar = fig.colorbar(im,ax=ax, shrink=0.75)

    ax = plt.subplot(3,4,10)
    im = ax.imshow(p_mms.reshape(mz,mx)-p.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps)
    ax.set_title('b3) Error P', fontweight='bold')
    cbar = fig.colorbar(im,ax=ax, shrink=0.75)

    ax = plt.subplot(3,4,11)
    im = ax.imshow(vx_mms.reshape(mz,mx+1)-vx.reshape(mz,mx+1),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps)
    ax.set_title('c3) Error ux', fontweight='bold')
    cbar = fig.colorbar(im,ax=ax, shrink=0.75)

    ax = plt.subplot(3,4,12)
    im = ax.imshow(vz_mms.reshape(mz+1,mx)-vz.reshape(mz+1,mx),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps)
    ax.set_title('d3) Error uz', fontweight='bold')
    cbar = fig.colorbar(im,ax=ax, shrink=0.75)

    plt.tight_layout() 
    fout = fname1+'_all_fields'+ft 
    plt.savefig(fout+'.pdf')
    plt.close()

# ---------------------------------------
# Plot errors and convergence
# ---------------------------------------
nrm_phi1 = np.zeros(len(dt))
nrm_phi2 = np.zeros(len(dt))
nrm_vs = np.zeros(len(dt))
nrm_ps = np.zeros(len(dt))
dt_nrm = np.zeros(len(dt))

for i in range(len(dt)):
  dt_nrm[i] = 10**(dt[i])

for i in range(len(dt)):

  dt_string = str(dt[i])
  dt_string = dt_string.replace('.','-')

  dtmax = 10**(dt[i])
  # Create output filename
  fname1 = fname+'_dt'+dt_string+'.out'

  f = open(fname1, 'r')
  i0=0
  for line in f:
    if '# TIMESTEP' in line:
        i0+=1
  f.close()
  tstep = i0

  # Norm variables
  nrm_v = np.zeros(tstep)
  nrm_p = np.zeros(tstep)
  nrm_vx = np.zeros(tstep)
  nrm_vz = np.zeros(tstep)
  nrm_phi = np.zeros(tstep)
  err_sum = np.zeros(tstep)
  err_mms = np.zeros(tstep)
  dt_num  = np.zeros(tstep)
  hx      = np.zeros(tstep)

  # Parse output and save norm info
  f = open(fname1, 'r')
  i0=-1
  for line in f:
    if '# TIMESTEP' in line:
        i0+=1
    if 'Velocity:' in line:
        nrm_v[i0] = float(line[20:38])
        # nrm_vx[i0] = float(line[48:66])
        # nrm_vz[i0] = float(line[76:94])
    if 'Pressure:' in line:
        nrm_p[i0] = float(line[20:38])
    if 'Porosity:' in line:
        nrm_phi[i0] = float(line[20:38])
    if 'Porosity err-squared:' in line:
        err_sum[i0] = float(line[30:48])
        err_mms[i0] = float(line[55:73])
    if 'Grid info:' in line:
        hx[i0] = float(line[18:36])
    if 'TIME: time' in line:
        dt_num[i0] = float(line[39:57])
  f.close()

  # Calculate errors
  sum_phi1 = 0.0 
  sum_phi2 = 0.0 # normalized
  sum_v = 0.0 
  sum_p = 0.0 

  for istep in range(tstep):
    sum_phi1 += err_sum[istep]*dt_num[istep]
    sum_phi2 += err_sum[istep]/err_mms[istep]*dt_num[istep]
    sum_v += nrm_v[istep]**2*dt_num[istep]
    sum_p += nrm_p[istep]**2*dt_num[istep]

  nrm_phi1[i] = sum_phi1**0.5
  nrm_phi2[i] = sum_phi2**0.5
  nrm_vs[i] = sum_v**0.5
  nrm_ps[i] = sum_p**0.5

# Plot 
plt.figure(1,figsize=(6,6))

plt.grid(color='lightgray', linestyle=':')

plt.plot(dt_nrm,nrm_phi1,'ko--',label='phi')
plt.plot(dt_nrm,nrm_phi2,'k+-',label='phi2')
plt.plot(dt_nrm,nrm_vs,'ro--',label='v')
plt.plot(dt_nrm,nrm_ps,'bo--',label='P')

plt.xscale("log")
plt.yscale("log")

plt.xlabel('Time step size',fontweight='bold',fontsize=12)
plt.ylabel(r'$E(\phi)$',fontweight='bold',fontsize=12)
plt.legend()

plt.savefig(fname+'_errors_L2.pdf')
plt.close()

# Print convergence orders:
dt_log    = np.log10(dt_nrm)
nrm2phi1_log = np.log10(nrm_phi1)
nrm2phi2_log = np.log10(nrm_phi2)
nrm2v_log = np.log10(nrm_vs)
nrm2p_log = np.log10(nrm_ps)

# Perform linear regression
sl2phi1, intercept, r_value, p_value, std_err = linregress(dt_log, nrm2phi1_log)
sl2phi2, intercept, r_value, p_value, std_err = linregress(dt_log, nrm2phi2_log)
sl2v, intercept, r_value, p_value, std_err = linregress(dt_log, nrm2v_log)
sl2p, intercept, r_value, p_value, std_err = linregress(dt_log, nrm2p_log)

print('# --------------------------------------- #')
print('# MMS StokesDarcy2Field+Porosity convergence order:')
print('     phi1_slope = '+str(sl2phi1)+' phi2_slope = '+str(sl2phi2))
print('     v_slope = '+str(sl2v)+' P_slope = '+str(sl2p))

os.system('rm -r __pycache__')