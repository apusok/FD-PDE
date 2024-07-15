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
sim = 'run37_flow_dike_dt1e2_etaK1e20/'
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

vizB.make_dir(A.output_path_dir)
vizB.make_dir(A.output_dir)

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

mass = np.zeros(nt)

# Loop over timesteps
itime = 0
for istep in time_list:
  fdir  = A.input_dir+'Timestep'+str(istep)
  print('  >> >> '+'Timestep'+str(istep))

  vizB.correct_path_load_data(fdir+'/parameters.py')
  A.nd.istep, A.nd.dt, A.nd.t = vizB.parse_time_info_parameters_file('parameters',fdir)

  # Get data
  vizB.correct_path_load_data(fdir+'/out_xMPhase_ts'+str(istep)+'.py')
  A.MPhase = vizB.parse_MPhase_file('out_xMPhase_ts'+str(istep),fdir)

  # Calculate rock mass/entire domain (cells) - should stay constant
  imass = 0.0
  for i in range(0,A.nx):
    indWall = np.where(A.MPhase.CornerPh0[:,i]>0.0)
    indW = indWall[0][0]
    imass += indW
  mass[itime] = imass/(A.nx*A.nz)
  itime += 1

# Plot
fig = plt.figure(1,figsize=(4,4))
ax = plt.subplot(1,1,1)
ax.plot(mass,'k-')
ax.grid(True)
ax.set_xlabel('Time step')
ax.set_ylabel('Mass fraction of rock')
plt.savefig(A.output_path_dir+'rock_mass_time.pdf', bbox_inches = 'tight')
plt.close()

os.system('rm -r '+pathViz+'/__pycache__')