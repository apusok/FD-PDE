# Import modules
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

# Add path to vizMORfault
pathViz = './'
sys.path.append(pathViz)
import vizMORfault as vizB

rc('font',**{'family':'serif','serif':['Times new roman']})
rc('text', usetex=True)

class SimStruct:
  pass

# ---------------------------------------
# Main
# ---------------------------------------
A = SimStruct() # Do print(A.__dict__) to see structure of A

def sortTimesteps(tdir):
  return int(tdir[8:])

# Parameters
A.dimensional = 1 # 0-nd, 1-dim
# sim = 'run28_03_phi00_phimax1e-3_Stokes/'
sim = 'run29_00_Stokes/'
A.input = '../'+sim
A.output_path_dir = '../Figures/'+sim
A.path_dir = './'

# search timesteps in folder
tdir = os.listdir(A.path_dir+A.input)
if '.DS_Store' in tdir:
  tdir.remove('.DS_Store')
if 'model_input.opts' in tdir:
  tdir.remove('model_input.opts')

tdir_check = list.copy(tdir)
for s in tdir_check:
  if '.out' in s:
    tdir.remove(s)

tdir_check = list.copy(tdir)
for s in tdir_check:
  if '_r' in s:
    tdir.remove(s)

nt = len(tdir)

# sort list in increasing tstep
tdir.sort(key=sortTimesteps)
time_list_v0 = np.zeros(nt)
time_list = time_list_v0.astype(int)
for ii in range(0,nt):
  time_list[ii] = int(tdir[ii][8:])

# Create directories
A.input_dir = A.path_dir+A.input
A.output_dir = A.output_path_dir

try:
  os.mkdir(A.output_path_dir)
except OSError:
  pass

try:
  os.mkdir(A.output_dir)
except OSError:
  pass

##########################
# Start figure
fig1 = plt.figure(1,figsize=(12,4))
ax1 = plt.subplot(1,2,1)
ax2 = plt.subplot(1,2,2)
color = plt.cm.coolwarm(np.linspace(0,1,nt))
##########################

# Read parameters file and get scaling params
istep = time_list[0]
fdir = A.input_dir+'Timestep'+str(istep)
vizB.correct_path_load_data(fdir+'/parameters.py')
A.scal, A.nd, A.geoscal = vizB.parse_parameters_file('parameters',fdir)
# A.scal, A.nd, A.geoscal = vizB.create_scaling() # if read from params file is not done

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

# plot entire domain
istart = 0
iend   = A.nx
jstart = 0
jend   = A.nz

# Loop over timesteps
itime = 0
for istep in time_list:
  fdir  = A.input_dir+'Timestep'+str(istep)
  print('  >> >> '+'Timestep'+str(istep))

  vizB.correct_path_load_data(fdir+'/parameters.py')
  A.nd.istep, A.nd.dt, A.nd.t = vizB.parse_time_info_parameters_file('parameters',fdir)

  # Load and plot markers
  # vizB.correct_path_marker_data(fdir+'/out_pic_ts'+str(istep)+'.xmf')
  # A.mark = vizB.parse_marker_file('out_pic_ts'+str(istep)+'.xmf',fdir)
  # vizB.plot_marker_id(A,istart,iend,jstart,jend,A.output_dir+'out_pic_ts'+str(istep),istep,A.dimensional)

  # Correct path for data
  vizB.correct_path_load_data(fdir+'/out_xT_ts'+str(istep)+'.py')
  vizB.correct_path_load_data(fdir+'/out_xphi_ts'+str(istep)+'.py')
  vizB.correct_path_load_data(fdir+'/out_xMPhase_ts'+str(istep)+'.py')
  vizB.correct_path_load_data(fdir+'/out_xPlith_ts'+str(istep)+'.py')
  vizB.correct_path_load_data(fdir+'/out_xDP_ts'+str(istep)+'.py')
  vizB.correct_path_load_data(fdir+'/out_xDPold_ts'+str(istep)+'.py')
  vizB.correct_path_load_data(fdir+'/out_xPV_ts'+str(istep)+'.py')
  vizB.correct_path_load_data(fdir+'/out_xeps_ts'+str(istep)+'.py')
  vizB.correct_path_load_data(fdir+'/out_xtau_ts'+str(istep)+'.py')
  vizB.correct_path_load_data(fdir+'/out_xtauold_ts'+str(istep)+'.py')
  vizB.correct_path_load_data(fdir+'/out_xPVcoeff_ts'+str(istep)+'.py')
  vizB.correct_path_load_data(fdir+'/out_xplast_ts'+str(istep)+'.py')
  vizB.correct_path_load_data(fdir+'/out_resPV_ts'+str(istep)+'.py')
  vizB.correct_path_load_data(fdir+'/out_resT_ts'+str(istep)+'.py')
  vizB.correct_path_load_data(fdir+'/out_resphi_ts'+str(istep)+'.py')
  vizB.correct_path_load_data(fdir+'/out_matProp_ts'+str(istep)+'.py')
  vizB.correct_path_load_data(fdir+'/out_xVel_ts'+str(istep)+'.py')
  vizB.correct_path_load_data(fdir+'/out_xTcoeff_ts'+str(istep)+'.py')
  vizB.correct_path_load_data(fdir+'/out_xphicoeff_ts'+str(istep)+'.py')
  vizB.correct_path_load_data(fdir+'/out_xplast_ts'+str(istep)+'.py')
  vizB.correct_path_load_data(fdir+'/out_xstrain_ts'+str(istep)+'.py')

  # Get data
  A.T = vizB.parse_Element_file('out_xT_ts'+str(istep),fdir)
  A.phis = vizB.parse_Element_file('out_xphi_ts'+str(istep),fdir)
  A.MPhase = vizB.parse_MPhase_file('out_xMPhase_ts'+str(istep),fdir)
  A.Plith = vizB.parse_Element_file('out_xPlith_ts'+str(istep),fdir)
  A.DP    = vizB.parse_Element_file('out_xDP_ts'+str(istep),fdir)
  A.DPold = vizB.parse_Element_file('out_xDPold_ts'+str(istep),fdir)
  A.dotlam= vizB.parse_Element_file('out_xplast_ts'+str(istep),fdir)
  A.P, A.Vsx, A.Vsz = vizB.parse_PV_file('out_xPV_ts'+str(istep),fdir)
  A.P_res, A.Vsx_res, A.Vsz_res = vizB.parse_PV_file('out_resPV_ts'+str(istep),fdir)
  A.T_res = vizB.parse_Element_file('out_resT_ts'+str(istep),fdir)
  A.phis_res = vizB.parse_Element_file('out_resphi_ts'+str(istep),fdir)
  A.eps = vizB.parse_Tensor_file('out_xeps_ts'+str(istep),fdir)
  A.tau = vizB.parse_Tensor_file('out_xtau_ts'+str(istep),fdir)
  A.tauold = vizB.parse_Tensor_file('out_xtauold_ts'+str(istep),fdir)
  # A.PVcoeff = vizB.parse_PVcoeff_file('out_xPVcoeff_ts'+str(istep),fdir)
  # A.PVcoeff = vizB.parse_PVcoeff_Stokes_file('out_xPVcoeff_ts'+str(istep),fdir)
  A.matProp = vizB.parse_matProp_file('out_matProp_ts'+str(istep),fdir)
  A.Vfx, A.Vfz, A.Vx, A.Vz = vizB.parse_Vel_file('out_xVel_ts'+str(istep),fdir)
  A.Tcoeff = vizB.parse_Tcoeff_file('out_xTcoeff_ts'+str(istep),fdir)
  A.phicoeff = vizB.parse_Tcoeff_file('out_xphicoeff_ts'+str(istep),fdir)
  A.dotlam = vizB.parse_Element_file('out_xplast_ts'+str(istep),fdir)
  A.lam = vizB.parse_Element_file('out_xstrain_ts'+str(istep),fdir)

  # Center velocities and mass divergence
  A.Vscx, A.Vscz, A.divVs = vizB.calc_center_velocities_div(A.Vsx,A.Vsz,A.grid.xv,A.grid.zv,A.nx,A.nz)

  ##########################
  # Extract variables of interest 
  mid_ind = int(A.nz/2)
  scalx = vizB.get_scaling(A,'x',1,1)
  scalrho = vizB.get_scaling(A,'rho',1,0)
  lblz = vizB.get_label(A,'z',1)
  lblx = vizB.get_label(A,'x',1)
  topo = np.zeros(A.nx)

  for i in range(0,A.nx):
    ztopo = 0.0
    for j in range(0,A.nz):
      if (A.matProp.rho[j,i]*scalrho<3000):
        ztopo = min(ztopo,A.grid.zc[j]*scalx)
    topo[i] = ztopo

  ax1.plot(A.grid.xc*scalx,topo,color=color[itime])
  # print('min='+str(min(topo))+' max='+str(max(topo)))

  X = 1.0 - A.phis
  X[X<0.0] = 0.0
  ax2.plot(A.grid.xc*scalx,X[mid_ind,:],color=color[itime])
  ##########################
  itime += 1

  # os.system('rm -r '+A.input_path_dir+'/'+'Timestep'+str(istep)+'/__pycache__')

##########################
# Finish plots
ax1.grid(True)
ax1.set_xlabel('x [km]')
ax1.set_ylabel('z [km]')
ax1.set_ylim([-30,-10])
ax1.set_title(r'Surface')

ax2.grid(True)
ax2.set_xlabel('x [km]')
ax2.set_ylabel(r'$\phi$')
ax2.set_title(r'Porosity')
plt.savefig(A.output_path_dir+'surf_phi.pdf', bbox_inches = 'tight')
plt.close()
##########################

os.system('rm -r '+pathViz+'/__pycache__')