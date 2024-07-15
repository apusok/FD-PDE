# Import modules
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add path to vizMORfault
pathViz = './'
sys.path.append(pathViz)
import vizMORfault as vizB

class SimStruct:
  pass

# ---------------------------------------
# Main
# ---------------------------------------
A = SimStruct() # Do print(A.__dict__) to see structure of A
A1 = SimStruct()
A2 = SimStruct()
A3 = SimStruct()

def sortTimesteps(tdir):
  return int(tdir[8:])

# Parameters
A.dimensional = 1 # 0-nd, 1-dim
sim = 'run25_VEVP_03_SD_mat2_30Myr/'
istep = 3000
A.input = '../'+sim
A.output_path_dir = '../Figures/'+sim
A.path_dir = './'

sim1 = '03_kT0' # kappa = 0 
sim2 = '01_kTlow' # kappa = 1e-7
sim3 = '02_kTrock' # kappa = 1e-6

# Create directories
A.input_dir = A.path_dir+A.input
A.output_dir = A.output_path_dir

try:
  os.mkdir(A.output_dir)
except OSError:
  pass

# Sim1
fdir  = A.input_dir+sim1+'/Timestep'+str(istep)
A1.scal, A1.nd, A1.geoscal = vizB.create_scaling() # if read from params file is not done
A1.lbl = vizB.create_labels()

vizB.correct_path_load_data(fdir+'/out_xT_ts'+str(istep)+'.py')
A1.grid = vizB.parse_grid_info('out_xT_ts'+str(istep),fdir)

# For easy access
A1.dx = A1.grid.xc[1]-A1.grid.xc[0]
A1.dz = A1.grid.zc[1]-A1.grid.zc[0]
A1.nx = A1.grid.nx
A1.nz = A1.grid.nz

# plot entire domain
istart = 0
iend   = A1.nx
jstart = 0
jend   = A1.nz

# Load and plot markers
vizB.correct_path_marker_data(fdir+'/out_pic_ts'+str(istep)+'.xmf')
A1.mark = vizB.parse_marker_file('out_pic_ts'+str(istep)+'.xmf',fdir)

# Correct path for data
vizB.correct_path_load_data(fdir+'/out_xT_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_xPV_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_xeps_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_xtau_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_xplast_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_matProp_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_xVel_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_xplast_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_xstrain_ts'+str(istep)+'.py')

# Get data
A1.T = vizB.parse_Element_file('out_xT_ts'+str(istep),fdir)
A1.dotlam= vizB.parse_Element_file('out_xplast_ts'+str(istep),fdir)
A1.P, A1.Vsx, A1.Vsz = vizB.parse_PV_file('out_xPV_ts'+str(istep),fdir)
A1.eps = vizB.parse_Tensor_file('out_xeps_ts'+str(istep),fdir)
A1.tau = vizB.parse_Tensor_file('out_xtau_ts'+str(istep),fdir)
A1.matProp = vizB.parse_matProp_file('out_matProp_ts'+str(istep),fdir)
A1.Vfx, A1.Vfz, A1.Vx, A1.Vz = vizB.parse_Vel_file('out_xVel_ts'+str(istep),fdir)
A1.dotlam = vizB.parse_Element_file('out_xplast_ts'+str(istep),fdir)
A1.lam = vizB.parse_Element_file('out_xstrain_ts'+str(istep),fdir)

# Center velocities and mass divergence
A1.Vscx, A1.Vscz, A1.divVs = vizB.calc_center_velocities_div(A1.Vsx,A1.Vsz,A1.grid.xv,A1.grid.zv,A1.nx,A1.nz)

# # Plots
# vizB.plot_mark_eta_eps_tau(A1,istart,iend,jstart,jend,A.output_dir+'out_mark_eta_eps_tau_ts'+str(istep),istep,A.dimensional)
# vizB.plot_T(A1,istart,iend,jstart,jend,A.output_dir+'out_xT_ts'+str(istep),istep,A.dimensional)
# vizB.plot_PV(A1,istart,iend,jstart,jend,A.output_dir+'out_xPV_ts'+str(istep),istep,A.dimensional,0)
# vizB.plot_Tensor(A1,istart,iend,jstart,jend,A.output_dir+'out_xeps_ts'+str(istep),istep,A.dimensional,0)
# vizB.plot_Tensor(A1,istart,iend,jstart,jend,A.output_dir+'out_xtau_ts'+str(istep),istep,A.dimensional,1)
# vizB.plot_matProp(A1,istart,iend,jstart,jend,A.output_dir+'out_matProp_ts'+str(istep),istep,A.dimensional)
# vizB.plot_Vel(A1,istart,iend,jstart,jend,A.output_dir+'out_xVel_ts'+str(istep),istep,A.dimensional)
# vizB.plot_plastic(A1,istart,iend,jstart,jend,A.output_dir+'out_xplastic_ts'+str(istep),istep,A.dimensional)

# Sim2
fdir  = A.input_dir+sim2+'/Timestep'+str(istep)
A2.scal, A2.nd, A2.geoscal = vizB.create_scaling() # if read from params file is not done
A2.lbl = vizB.create_labels()

vizB.correct_path_load_data(fdir+'/out_xT_ts'+str(istep)+'.py')
A2.grid = vizB.parse_grid_info('out_xT_ts'+str(istep),fdir)

# For easy access
A2.dx = A2.grid.xc[1]-A2.grid.xc[0]
A2.dz = A2.grid.zc[1]-A2.grid.zc[0]
A2.nx = A2.grid.nx
A2.nz = A2.grid.nz

# Load and plot markers
vizB.correct_path_marker_data(fdir+'/out_pic_ts'+str(istep)+'.xmf')
A2.mark = vizB.parse_marker_file('out_pic_ts'+str(istep)+'.xmf',fdir)

# Correct path for data
vizB.correct_path_load_data(fdir+'/out_xT_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_xPV_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_xeps_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_xtau_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_xplast_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_matProp_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_xVel_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_xplast_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_xstrain_ts'+str(istep)+'.py')

# Get data
A2.T = vizB.parse_Element_file('out_xT_ts'+str(istep),fdir)
A2.dotlam= vizB.parse_Element_file('out_xplast_ts'+str(istep),fdir)
A2.P, A2.Vsx, A2.Vsz = vizB.parse_PV_file('out_xPV_ts'+str(istep),fdir)
A2.eps = vizB.parse_Tensor_file('out_xeps_ts'+str(istep),fdir)
A2.tau = vizB.parse_Tensor_file('out_xtau_ts'+str(istep),fdir)
A2.matProp = vizB.parse_matProp_file('out_matProp_ts'+str(istep),fdir)
A2.Vfx, A2.Vfz, A2.Vx, A2.Vz = vizB.parse_Vel_file('out_xVel_ts'+str(istep),fdir)
A2.dotlam = vizB.parse_Element_file('out_xplast_ts'+str(istep),fdir)
A2.lam = vizB.parse_Element_file('out_xstrain_ts'+str(istep),fdir)

# Center velocities and mass divergence
A2.Vscx, A2.Vscz, A2.divVs = vizB.calc_center_velocities_div(A2.Vsx,A2.Vsz,A2.grid.xv,A2.grid.zv,A2.nx,A2.nz)

# # Plots
# vizB.plot_mark_eta_eps_tau(A2,istart,iend,jstart,jend,A.output_dir+'out_mark_eta_eps_tau_ts'+str(istep),istep,A.dimensional)
# vizB.plot_T(A2,istart,iend,jstart,jend,A.output_dir+'out_xT_ts'+str(istep),istep,A.dimensional)
# vizB.plot_PV(A2,istart,iend,jstart,jend,A.output_dir+'out_xPV_ts'+str(istep),istep,A.dimensional,0)
# vizB.plot_Tensor(A2,istart,iend,jstart,jend,A.output_dir+'out_xeps_ts'+str(istep),istep,A.dimensional,0)
# vizB.plot_Tensor(A2,istart,iend,jstart,jend,A.output_dir+'out_xtau_ts'+str(istep),istep,A.dimensional,1)
# vizB.plot_matProp(A2,istart,iend,jstart,jend,A.output_dir+'out_matProp_ts'+str(istep),istep,A.dimensional)
# vizB.plot_Vel(A2,istart,iend,jstart,jend,A.output_dir+'out_xVel_ts'+str(istep),istep,A.dimensional)
# vizB.plot_plastic(A2,istart,iend,jstart,jend,A.output_dir+'out_xplastic_ts'+str(istep),istep,A.dimensional)

# Sim3
fdir  = A.input_dir+sim3+'/Timestep'+str(istep)
A3.scal, A3.nd, A3.geoscal = vizB.create_scaling() # if read from params file is not done
A3.lbl = vizB.create_labels()

vizB.correct_path_load_data(fdir+'/out_xT_ts'+str(istep)+'.py')
A3.grid = vizB.parse_grid_info('out_xT_ts'+str(istep),fdir)

# For easy access
A3.dx = A3.grid.xc[1]-A3.grid.xc[0]
A3.dz = A3.grid.zc[1]-A3.grid.zc[0]
A3.nx = A3.grid.nx
A3.nz = A3.grid.nz

# Load and plot markers
vizB.correct_path_marker_data(fdir+'/out_pic_ts'+str(istep)+'.xmf')
A3.mark = vizB.parse_marker_file('out_pic_ts'+str(istep)+'.xmf',fdir)

# Correct path for data
vizB.correct_path_load_data(fdir+'/out_xT_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_xPV_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_xeps_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_xtau_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_xplast_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_matProp_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_xVel_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_xplast_ts'+str(istep)+'.py')
vizB.correct_path_load_data(fdir+'/out_xstrain_ts'+str(istep)+'.py')

# Get data
A3.T = vizB.parse_Element_file('out_xT_ts'+str(istep),fdir)
A3.dotlam= vizB.parse_Element_file('out_xplast_ts'+str(istep),fdir)
A3.P, A3.Vsx, A3.Vsz = vizB.parse_PV_file('out_xPV_ts'+str(istep),fdir)
A3.eps = vizB.parse_Tensor_file('out_xeps_ts'+str(istep),fdir)
A3.tau = vizB.parse_Tensor_file('out_xtau_ts'+str(istep),fdir)
A3.matProp = vizB.parse_matProp_file('out_matProp_ts'+str(istep),fdir)
A3.Vfx, A3.Vfz, A3.Vx, A3.Vz = vizB.parse_Vel_file('out_xVel_ts'+str(istep),fdir)
A3.dotlam = vizB.parse_Element_file('out_xplast_ts'+str(istep),fdir)
A3.lam = vizB.parse_Element_file('out_xstrain_ts'+str(istep),fdir)

# Center velocities and mass divergence
A3.Vscx, A3.Vscz, A3.divVs = vizB.calc_center_velocities_div(A3.Vsx,A3.Vsz,A3.grid.xv,A3.grid.zv,A3.nx,A3.nz)

# Plots
# vizB.plot_mark_eta_eps_tau(A3,istart,iend,jstart,jend,A.output_dir+'out_mark_eta_eps_tau_ts'+str(istep),istep,A.dimensional)
# vizB.plot_T(A3,istart,iend,jstart,jend,A.output_dir+'out_xT_ts'+str(istep),istep,A.dimensional)
# vizB.plot_PV(A3,istart,iend,jstart,jend,A.output_dir+'out_xPV_ts'+str(istep),istep,A.dimensional,0)
# vizB.plot_Tensor(A3,istart,iend,jstart,jend,A.output_dir+'out_xeps_ts'+str(istep),istep,A.dimensional,0)
# vizB.plot_Tensor(A3,istart,iend,jstart,jend,A.output_dir+'out_xtau_ts'+str(istep),istep,A.dimensional,1)
# vizB.plot_matProp(A3,istart,iend,jstart,jend,A.output_dir+'out_matProp_ts'+str(istep),istep,A.dimensional)
# vizB.plot_Vel(A3,istart,iend,jstart,jend,A.output_dir+'out_xVel_ts'+str(istep),istep,A.dimensional)
# vizB.plot_plastic(A3,istart,iend,jstart,jend,A.output_dir+'out_xplastic_ts'+str(istep),istep,A.dimensional)

os.system('rm -r '+A.input_dir+'Timestep'+str(istep)+'/__pycache__')
os.system('rm -r '+pathViz+'/__pycache__')

# Plot comparison profiles
scalx = vizB.get_scaling(A1,'x',1,1)
scaleta = vizB.get_scaling(A1,'eta',1,0)
scalrho = vizB.get_scaling(A1,'rho',1,0)
lblz = vizB.get_label(A1,'z',1)
lblx = vizB.get_label(A1,'x',1)

xind = 45 # x=-10 km
T10 = A1.T[:,xind]*A1.scal.DT+A1.scal.T0 - A1.geoscal.T
T20 = A2.T[:,xind]*A1.scal.DT+A1.scal.T0 - A1.geoscal.T
T30 = A3.T[:,xind]*A1.scal.DT+A1.scal.T0 - A1.geoscal.T

xind = 12 # x=-76 km
T11 = A1.T[:,xind]*A1.scal.DT+A1.scal.T0 - A1.geoscal.T
T21 = A2.T[:,xind]*A1.scal.DT+A1.scal.T0 - A1.geoscal.T
T31 = A3.T[:,xind]*A1.scal.DT+A1.scal.T0 - A1.geoscal.T

eta1 = np.log10(A1.matProp.eta[:,xind]*scaleta)
eta2 = np.log10(A2.matProp.eta[:,xind]*scaleta)
eta3 = np.log10(A3.matProp.eta[:,xind]*scaleta)

topo1 = np.zeros(A1.nx)
topo2 = np.zeros(A1.nx)
topo3 = np.zeros(A1.nx)

for i in range(0,A1.nx):
  ztopo = 0.0
  for j in range(0,A1.nz):
    if (A1.matProp.rho[j,i]*scalrho<3000):
      ztopo = min(ztopo,A1.grid.zc[j]*scalx)
  topo1[i] = ztopo

for i in range(0,A1.nx):
  ztopo = 0.0
  for j in range(0,A1.nz):
    if (A2.matProp.rho[j,i]*scalrho<3000):
      ztopo = min(ztopo,A1.grid.zc[j]*scalx)
  topo2[i] = ztopo

for i in range(0,A1.nx):
  ztopo = 0.0
  for j in range(0,A1.nz):
    if (A3.matProp.rho[j,i]*scalrho<3000):
      ztopo = min(ztopo,A1.grid.zc[j]*scalx)
  topo3[i] = ztopo

fig = plt.figure(1,figsize=(10,8))

# sim1 = '03_kT0' # kappa = 0 
# sim2 = '01_kTlow' # kappa = 1e-7
# sim3 = '02_kTrock' # kappa = 1e-6

ax = plt.subplot(2,2,1)
pl = ax.plot(T11,A1.grid.zc*scalx,label='kT=0')
pl = ax.plot(T21,A1.grid.zc*scalx,label='kT=0.6')
pl = ax.plot(T31,A1.grid.zc*scalx,label='kT=3.96')
plt.grid(True)
ax.set_ylabel(lblz)
# ax.set_xlabel('T (oC)')
ax.set_title('a) T (oC) at x=-75km')
ax.legend()

ax = plt.subplot(2,2,2)
pl = ax.plot(T10,A1.grid.zc*scalx,label='kT=0')
pl = ax.plot(T20,A1.grid.zc*scalx,label='kT=0.6')
pl = ax.plot(T30,A1.grid.zc*scalx,label='kT=3.96')
plt.grid(True)
# ax.set_xlabel('T (oC)')
ax.set_title('b) T (oC) at x=-10km')
ax.set_ylabel(lblz)

ax = plt.subplot(2,2,3)
pl = ax.plot(eta1,A1.grid.zc*scalx,label='kT=0')
pl = ax.plot(eta2,A1.grid.zc*scalx,label='kT=0.6')
pl = ax.plot(eta3,A1.grid.zc*scalx,label='kT=3.96')
plt.grid(True)
# ax.set_xlabel('log10(eta)')
ax.set_ylabel(lblz)
ax.set_title('c) log10(eta) at x=-75km')

ax = plt.subplot(2,2,4)
pl = ax.plot(A1.grid.xc*scalx,topo1,label='kT=0')
pl = ax.plot(A1.grid.xc*scalx,topo2,label='kT=0.6')
pl = ax.plot(A1.grid.xc*scalx,topo3,label='kT=3.96')
plt.grid(True)
ax.set_ylabel(lblz)
ax.set_title('d) Water-rock interface level')
ax.set_xlabel(lblx)
ax.set_ylim(-50,-5)

plt.savefig(A.output_dir+'out_compare_ts'+str(istep)+'.png', bbox_inches = 'tight')
plt.close()