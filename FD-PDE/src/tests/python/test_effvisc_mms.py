# ---------------------------------------
# MMS test for power-law viscosity for Stokes and StokesDarcy
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
  try: # try to open directory
    # Parse output and save norm info
    line_ind = 6
    f = open(fname, 'r')
    for line in f:
      if 'Velocity test1:' in line:
          nrm_v1_num = float(line[20+line_ind:38+line_ind])
      if 'Pressure test1:' in line:
          nrm_p1_num = float(line[20+line_ind:38+line_ind])
      if 'Velocity test2:' in line:
          nrm_v2_num = float(line[20+line_ind:38+line_ind])
      if 'Pressure test2:' in line:
          nrm_p2_num = float(line[20+line_ind:38+line_ind])
      if 'Grid info test1:' in line:
          hx_num = float(line[18+line_ind:36+line_ind])
    f.close()

    return nrm_v1_num, nrm_p1_num, nrm_v2_num, nrm_p2_num, hx_num
  except OSError:
    print('Cannot open:', fdir)
    return tstep

# ---------------------------------------
def plot_solution_mms_error(fname,nx,j):

  # Load data
  fout = fname+'_stokes' # 1. Numerical solution stokes
  imod = importlib.import_module(fout) 
  data = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(data)

  mx = data['Nx'][0]
  mz = data['Ny'][0]
  xc = data['x1d_cell']
  zc = data['y1d_cell']
  xv = data['x1d_vertex']
  zv = data['y1d_vertex']
  vx1 = data['X_face_x']
  vz1 = data['X_face_y']
  p1 = data['X_cell']

  fout = fname+'_stokesdarcy' # 2. Numerical solution stokesdarcy
  imod = importlib.import_module(fout) 
  data = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(data)

  vx2 = data['X_face_x']
  vz2 = data['X_face_y']
  p2  = data['X_cell']

  fout = 'out_mms_solution'
  imod = importlib.import_module(fout) # 3. MMS solution
  data = imod._PETScBinaryLoad()
  imod._PETScBinaryLoadReportNames(data)
  vx_mms = data['X_face_x']
  vz_mms = data['X_face_y']
  p_mms = data['X_cell']

  pmax = max(p_mms)
  pmin = min(p_mms)
  vxmax = max(vx_mms)
  vxmin = min(vx_mms)
  vzmax = max(vz_mms)
  vzmin = min(vz_mms)

  # Plot all fields - mms, solution and errors for P, ux, uz
  fig = plt.figure(1,figsize=(12,12))
  cmaps='RdBu_r' 

  ax = plt.subplot(3,3,1)
  im = ax.imshow(p_mms.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=pmin,vmax=pmax,cmap=cmaps)
  ax.set_title(r'MMS $P$')
  ax.set_ylabel('z')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,3,2)
  im = ax.imshow(p1.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=pmin,vmax=pmax,cmap=cmaps)
  ax.set_title(r'Stokes $P$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,3,3)
  im = ax.imshow(p2.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=pmin,vmax=pmax,cmap=cmaps)
  ax.set_title(r'Stokes-Darcy $P$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,3,4)
  im = ax.imshow(vx_mms.reshape(mz,mx+1),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=vxmin,vmax=vxmax,cmap=cmaps)
  ax.set_title(r'MMS $v_x$')
  ax.set_ylabel('z')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,3,5)
  im = ax.imshow(vx1.reshape(mz,mx+1),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=vxmin,vmax=vxmax,cmap=cmaps)
  ax.set_title(r'Stokes $v_x$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,3,6)
  im = ax.imshow(vx2.reshape(mz,mx+1),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=vxmin,vmax=vxmax,cmap=cmaps)
  ax.set_title(r'Stokes-Darcy $v_x$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,3,7)
  im = ax.imshow(vz_mms.reshape(mz+1,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=vzmin,vmax=vzmax,cmap=cmaps)
  ax.set_title(r'MMS $v_z$')
  ax.set_ylabel('z')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,3,8)
  im = ax.imshow(vz1.reshape(mz+1,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=vzmin,vmax=vzmax,cmap=cmaps)
  ax.set_title(r'Stokes $v_z$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,3,9)
  im = ax.imshow(vz2.reshape(mz+1,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=vzmin,vmax=vzmax,cmap=cmaps)
  ax.set_title(r'Stokes-Darcy $v_z$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  plt.tight_layout() 
  fout = fname+'_solution'+'_npind'+str(j)+'_nx_'+str(nx)
  plt.savefig(fout+'.pdf')
  plt.close()

# ---------------------------------------
def plot_convergence_error(fname,nexp,hx,nrm_p1,nrm_v1,nrm_p2,nrm_v2):
  hx_log    = np.log10(hx)
  nrmp1_log = np.log10(nrm_p1)
  nrmv1_log = np.log10(nrm_v1)
  nrmp2_log = np.log10(nrm_p2)
  nrmv2_log = np.log10(nrm_v2)

  slp1 = np.zeros(len(nexp))
  slv1 = np.zeros(len(nexp))
  slp2 = np.zeros(len(nexp))
  slv2 = np.zeros(len(nexp))

  # Perform linear regression
  for j in range(len(nexp)):
    slp1[j], intercept, r_value, p_value, std_err = linregress(hx_log[j,:], nrmp1_log[j,:])
    slv1[j], intercept, r_value, p_value, std_err = linregress(hx_log[j,:], nrmv1_log[j,:])
    slp2[j], intercept, r_value, p_value, std_err = linregress(hx_log[j,:], nrmp2_log[j,:])
    slv2[j], intercept, r_value, p_value, std_err = linregress(hx_log[j,:], nrmv2_log[j,:])

  colors = plt.cm.viridis(np.linspace(0,1,len(nexp)))
  plt.figure(1,figsize=(12,6))

  plt.subplot(121)
  plt.grid(color='lightgray', linestyle=':')

  for j in range(len(nexp)):
    plt.plot(hx[j,:],nrm_p1[j,:],'o--',color=colors[j],label='np='+str(nexp[j])+' P sl='+str(round(slp1[j],5)))
    plt.plot(hx[j,:],nrm_v1[j,:],'+--',color=colors[j],label='np='+str(nexp[j])+' v sl='+str(round(slv1[j],5)))

  plt.xscale("log")
  plt.yscale("log")
  plt.xlabel('log10(h)',fontweight='bold',fontsize=12)
  plt.ylabel(r'$E(P), E(v)$',fontweight='bold',fontsize=12)
  plt.title('a) Stokes',fontweight='bold',fontsize=12)
  plt.legend()

  plt.subplot(122)
  plt.grid(color='lightgray', linestyle=':')

  for j in range(len(nexp)):
    plt.plot(hx[j,:],nrm_p2[j,:],'o--',color=colors[j],label='np='+str(nexp[j])+' P sl='+str(round(slp2[j],5)))
    plt.plot(hx[j,:],nrm_v2[j,:],'+--',color=colors[j],label='np='+str(nexp[j])+' v sl='+str(round(slv2[j],5)))

  plt.xscale("log")
  plt.yscale("log")
  plt.xlabel('log10(h)',fontweight='bold',fontsize=12)
  plt.ylabel(r'$E(P), E(v)$',fontweight='bold',fontsize=12)
  plt.title('b) Stokes-Darcy',fontweight='bold',fontsize=12)
  plt.legend()

  plt.savefig(fname+'_error_hx_L2.pdf')
  plt.close()

# ---------------------------------------
# Main script
# ---------------------------------------
print('# --------------------------------------- #')
print('# MMS tests for power-law effective viscosity ')
print('# --------------------------------------- #')

# Set main parameters and run test
fname = 'out_effvisc'
n  = [20, 40, 100]#, 200, 300] # resolution
nexp = [1.0, 2.0, 3.0] # power-law exponent

# Prepare errors and convergence
nrm_p1  = np.zeros((len(nexp),len(n))) # 1- stokes
nrm_p2  = np.zeros((len(nexp),len(n))) # 2- stokes-darcy
nrm_v1  = np.zeros((len(nexp),len(n)))
nrm_v2  = np.zeros((len(nexp),len(n)))
hx      = np.zeros((len(nexp),len(n)))

# Run simulations
for j in range(len(nexp)):  
  for i in range(len(n)):
    # Create output filename
    inp = nexp[j]
    nx  = n[i]
    fout = fname+'_np'+str(j)+'_'+str(nx)+'.out'

    # Run with different resolutions - 1 timestep
    str1 = '../test_effvisc_mms.app -pc_type lu -pc_factor_mat_solver_type umfpack'+ \
        ' -output_file '+fname+ \
        ' -nexp '+str(inp)+ \
        ' -nx '+str(nx)+' -nz '+str(nx)+' > '+fout
    print(str1)
    os.system(str1)

    # Parse variables
    nrm_v1_num, nrm_p1_num, nrm_v2_num, nrm_p2_num, hx_num = parse_log_file(fout)
    nrm_p1[j,i] = nrm_p1_num
    nrm_v1[j,i] = nrm_v1_num
    nrm_p2[j,i] = nrm_p2_num
    nrm_v2[j,i] = nrm_v2_num
    hx[j,i]     = hx_num

    # Plot solution and error
    plot_solution_mms_error(fname,nx,j)

print(nrm_p1)
print(nrm_v1)
# Convergence plot
plot_convergence_error(fname,nexp,hx,nrm_p1,nrm_v1,nrm_p2,nrm_v2)

os.system('rm -r __pycache__')