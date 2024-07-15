# Import libraries
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Add path to vizMORfault
pathViz = './'
sys.path.append(pathViz)
import vizMORfault as vizB

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times new roman']})
rc('text', usetex=True)

class SimStruct:
  pass

def sortTimesteps(tdir):
  return int(tdir[8:])

# ---------------------------------
def load_data(A,ix):
  try: 
    A.input_dir = A.input
    A.dimensional = 1

    # search timesteps in folder
    tdir = os.listdir(A.input)
    if '.DS_Store' in tdir:
      tdir.remove('.DS_Store')
    if 'model_input.opts' in tdir:
      tdir.remove('model_input.opts')
    if 'submit_job.run' in tdir:
      tdir.remove('submit_job.run')

    tdir_check = list.copy(tdir)
    for s in tdir_check:
      if '.out' in s:
        tdir.remove(s)

    tdir_check = list.copy(tdir)
    for s in tdir_check:
      if '_r' in s:
        tdir.remove(s)
    
    tdir_check = list.copy(tdir)
    for s in tdir_check:
      if 'slurm' in s:
        tdir.remove(s)

    nt = len(tdir)
    A.nt = nt

    # sort list in increasing tstep
    tdir.sort(key=sortTimesteps)
    time_list_v0 = np.zeros(nt)
    time_list = time_list_v0.astype(int)
    for ii in range(0,nt):
      time_list[ii] = int(tdir[ii][8:])

    # Read parameters file and get scaling params
    istep = time_list[0]
    fdir = A.input_dir+'Timestep'+str(istep)
    vizB.correct_path_load_data(fdir+'/parameters.py')
    A.scal, A.nd, A.geoscal = vizB.parse_parameters_file('parameters',fdir)

    # Create labels
    A.lbl = vizB.create_labels()

    # Read grid parameters - do this operation only once
    fdir  = A.input_dir+'Timestep'+str(istep)
    vizB.correct_path_load_data(fdir+'/out_xT_ts'+str(istep)+'.py')
    A.grid = vizB.parse_grid_info('out_xT_ts'+str(istep),fdir)

    # For easy access
    A.dx = A.grid.xc[1]-A.grid.xc[0]
    A.dz = A.grid.zc[1]-A.grid.zc[0]
    A.nx = A.grid.nx
    A.nz = A.grid.nz

    # Time arrays
    A.nt_phi    = np.zeros([A.nz,A.nt])
    A.nt_epsII  = np.zeros([A.nz,A.nt])
    A.nt_divv   = np.zeros([A.nz,A.nt])
    A.nt_DP     = np.zeros([A.nz,A.nt])
    A.nt_tauII  = np.zeros([A.nz,A.nt])
    A.nt_dotlam = np.zeros([A.nz,A.nt])
    A.nt_Vfz    = np.zeros([A.nz,A.nt])
    A.nt_topo   = np.zeros(A.nt)
    A.nt_tkyr   = np.zeros(A.nt)
    
    scalx = vizB.get_scaling(A,'x',1,1)
    scalt = vizB.get_scaling(A,'t',1,1)
    scaleps = vizB.get_scaling(A,'eps',1,0)
    scal_v_ms = vizB.get_scaling(A,'v',1,0)
    scal_x_m = vizB.get_scaling(A,'x',1,0)
    scalP = vizB.get_scaling(A,'P',1,1)
    scalv = vizB.get_scaling(A,'v',1,1)

    # Loop over timesteps
    it = 0
    for istep in time_list:
      fdir  = A.input_dir+'Timestep'+str(istep)
      print('  >> >> '+'Timestep'+str(istep))

      vizB.correct_path_load_data(fdir+'/parameters.py')
      A.nd.istep, A.nd.dt, A.nd.t = vizB.parse_time_info_parameters_file('parameters',fdir)

      # Load markers
      vizB.correct_path_marker_data(fdir+'/out_pic_ts'+str(istep)+'.xmf')
      A.mark = vizB.parse_marker_file('out_pic_ts'+str(istep)+'.xmf',fdir)

      # Correct path for data
      vizB.correct_path_load_data(fdir+'/out_xphi_ts'+str(istep)+'.py')
      vizB.correct_path_load_data(fdir+'/out_xDP_ts'+str(istep)+'.py')
      vizB.correct_path_load_data(fdir+'/out_xPV_ts'+str(istep)+'.py')
      vizB.correct_path_load_data(fdir+'/out_xeps_ts'+str(istep)+'.py')
      vizB.correct_path_load_data(fdir+'/out_xtau_ts'+str(istep)+'.py')
      vizB.correct_path_load_data(fdir+'/out_xVel_ts'+str(istep)+'.py')
      vizB.correct_path_load_data(fdir+'/out_xplast_ts'+str(istep)+'.py')
      vizB.correct_path_load_data(fdir+'/out_xstrain_ts'+str(istep)+'.py')
      
      # Get data
      A.phis = vizB.parse_Element_file('out_xphi_ts'+str(istep),fdir)
      A.DP    = vizB.parse_Element_file('out_xDP_ts'+str(istep),fdir) 
      A.P, A.Vsx, A.Vsz = vizB.parse_PV_file('out_xPV_ts'+str(istep),fdir)
      A.eps = vizB.parse_Tensor_file('out_xeps_ts'+str(istep),fdir)
      A.tau = vizB.parse_Tensor_file('out_xtau_ts'+str(istep),fdir)
      A.Vfx, A.Vfz, A.Vx, A.Vz = vizB.parse_Vel_file('out_xVel_ts'+str(istep),fdir)
      A.dotlam = vizB.parse_Element_file('out_xplast_ts'+str(istep),fdir)
      A.lam = vizB.parse_Element_file('out_xstrain_ts'+str(istep),fdir)

      A.Vscx, A.Vscz, A.divVs = vizB.calc_center_velocities_div(A.Vsx,A.Vsz,A.grid.xv,A.grid.zv,A.nx,A.nz)

      # Extract 
      X = 1.0 - A.phis
      X[X<1e-10] = 1e-10
      A.nt_phi[:,it] = X[:,ix]
      A.nt_epsII[:,it] = A.eps.II_center[:,ix]*scaleps
      A.nt_divv[:,it] = A.divVs[:,ix]*scal_v_ms/scal_x_m
      A.nt_DP[:,it] = A.DP[:,ix]*scalP
      A.nt_tauII[:,it] = A.tau.II_center[:,ix]*scalP
      A.nt_dotlam[:,it] = A.dotlam[:,ix]*scal_v_ms/scal_x_m
      A.nt_Vfz[:,it] = (A.Vfz[:-1,ix]+A.Vfz[1:,ix])*0.5*scalv
      A.nt_tkyr[it]  = A.nd.t*scalt

      # Free surf
      ax = A.grid.xc[ix]-A.dx*0.5
      bx = A.grid.xc[ix]+A.dx*0.5
      ztopo = 0.0
      for j in range(0,A.mark.n):
        if (A.mark.x[j]>=ax) & (A.mark.x[j]<bx) & (A.mark.id[j]==0):
          ztopo = min(ztopo,A.mark.z[j]*scalx)
      A.nt_topo[it] = ztopo

      it += 1

      os.system('rm -r '+A.input_dir+'Timestep'+str(istep)+'/__pycache__')

    return A
  except OSError:
    print('Cannot open: '+A.input)
    return 0.0

# ---------------------------------
# Main script
# ---------------------------------

path_in ='/Users/apusok/Documents/morfault/'
path_out='/Users/apusok/Documents/morfault/Figures/'

ix = 50
sim = 'run51_VEVP_var_age_eta1e18_Vext0_res1km/'
read_data = 0

A = SimStruct()
A.input = path_in+sim
A.output_dir= path_out+sim
vizB.make_dir(A.output_dir)

fname_pickle = A.output_dir+'time_data_2D_map.txt'

if (read_data):
  # Read raw data
  A = load_data(A,ix)

  # Save data
  with open(fname_pickle,'wb') as fh:
      pickle.dump(A, fh)

else:
  # Read data
  pickle_off = open(fname_pickle,'rb')
  A = pickle.load(pickle_off)
  pickle_off.close()

scalx = vizB.get_scaling(A,'x',1,1)
extentT=[min(A.nt_tkyr), max(A.nt_tkyr), min(A.grid.zc)*scalx, max(A.grid.zc)*scalx]

fig = plt.figure(1,figsize=(A.nt/10,25))
# fig = plt.figure()
nplots = 7

ax = plt.subplot(nplots,1,1)
im = ax.imshow(np.log10(A.nt_phi),aspect='auto',extent=extentT,origin='lower')
cbar = fig.colorbar(im,ax=ax, shrink=0.70)
im.set_clim(-6,0)
pl = ax.plot(A.nt_tkyr,A.nt_topo,'w-',linewidth=0.5)
ax.set_ylabel(r'z (km)')
ax.set_title(r'$\log_{10}\phi$')

ax = plt.subplot(nplots,1,2)
im = ax.imshow(np.log10(A.nt_epsII),aspect='auto',extent=extentT,origin='lower')
cbar = fig.colorbar(im,ax=ax, shrink=0.70)
pl = ax.plot(A.nt_tkyr,A.nt_topo,'w-',linewidth=0.5)
ax.set_ylabel(r'z (km)')
ax.set_title(r'$\log_{10}\dot{\epsilon}_{II}$ (1/s)')

ax = plt.subplot(nplots,1,3)
scal0=1e-15
im = ax.imshow(A.nt_divv*scal0,aspect='auto',extent=extentT,origin='lower')
cbar = fig.colorbar(im,ax=ax, shrink=0.70)
pl = ax.plot(A.nt_tkyr,A.nt_topo,'w-',linewidth=0.5)
ax.set_ylabel(r'z (km)')
ax.set_title(r'$\nabla\cdot\textbf{v}_s \times 10^{-15}$ (1/s)')

ax = plt.subplot(nplots,1,4)
im = ax.imshow(A.nt_DP,aspect='auto',extent=extentT,origin='lower')
cbar = fig.colorbar(im,ax=ax, shrink=0.70)
pl = ax.plot(A.nt_tkyr,A.nt_topo,'w-',linewidth=0.5)
ax.set_ylabel(r'z (km)')
ax.set_title(r'$\Delta P$ (MPa)')

ax = plt.subplot(nplots,1,5)
im = ax.imshow(A.nt_tauII,aspect='auto',extent=extentT,origin='lower')
cbar = fig.colorbar(im,ax=ax, shrink=0.70)
pl = ax.plot(A.nt_tkyr,A.nt_topo,'w-',linewidth=0.5)
ax.set_ylabel(r'z (km)')
ax.set_title(r'$\tau_{II}$ (MPa)')

ax = plt.subplot(nplots,1,6)
im = ax.imshow(A.nt_dotlam,aspect='auto',extent=extentT,origin='lower')
cbar = fig.colorbar(im,ax=ax, shrink=0.70)
pl = ax.plot(A.nt_tkyr,A.nt_topo,'w-',linewidth=0.5)
ax.set_ylabel(r'z (km)')
ax.set_title(r'$\dot{\lambda}$ (1/s)')

ax = plt.subplot(nplots,1,7)
im = ax.imshow(A.nt_Vfz,aspect='auto',extent=extentT,origin='lower')
cbar = fig.colorbar(im,ax=ax, shrink=0.70)
pl = ax.plot(A.nt_tkyr,A.nt_topo,'w-',linewidth=0.5)
ax.set_xlabel('Time (kyr)')
ax.set_ylabel(r'z (km)')
ax.set_title(r'$\textbf{v}_\ell^z$ (cm/yr)')

plt.savefig(A.output_dir+'plot_1D_profile_2Dmap.png', bbox_inches = 'tight')
plt.close()
