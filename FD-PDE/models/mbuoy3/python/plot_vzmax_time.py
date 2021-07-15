# Import modules
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pickle

rc('font',**{'family':'serif','serif':['Times new roman']})
rc('text', usetex=True)

# Add path to vizMORBuoyancy
pathViz = './'
sys.path.append(pathViz)
import vizMORBuoyancy as vizB

class SimStruct:
  pass

# ---------------------------------------
# Main
# ---------------------------------------
A = SimStruct() # Do print(A.__dict__) to see structure of A

def sortTimesteps(tdir):
  return int(tdir[8:])

# Parameters
A.tout  = 1000
A.tstep = 200000
A.istep = 14741
A.dimensional = 1 # 0-nd, 1-dim
check_entire_dir = 1 

A.input = 'run004_u04_xmor5_fextract00025'
A.output_path_dir = '../half_ridge/Figures/'
A.path_dir = '../half_ridge/'

# search timesteps in folder
tdir = os.listdir(A.path_dir+A.input)
if '.DS_Store' in tdir:
  tdir.remove('.DS_Store')
if 'log_out_'+A.input+'.out' in tdir:
  tdir.remove('log_out_'+A.input+'.out')
if 'model_half_ridge.opts' in tdir:
  tdir.remove('model_half_ridge.opts')
if 'run_job.slurm' in tdir:
  tdir.remove('run_job.slurm')
nt = len(tdir)

# sort list in increasing tstep
tdir.sort(key=sortTimesteps)

if (check_entire_dir):
  time_list_v0 = np.zeros(nt)
  time_list = time_list_v0.astype(int)
  for ii in range(0,nt):
    time_list[ii] = int(tdir[ii][8:])
else:
  time_list = range(A.istep,A.tstep+1,A.tout)

# Create directories
A.input_dir = A.path_dir+A.input+'/'
A.output_dir = A.output_path_dir+'/'+A.input+'/'
A.output_dir_real = A.output_path_dir+'/'+A.input+'/debug_output/'

print(A.input_dir)

try:
  os.mkdir(A.output_path_dir)
except OSError:
  pass

try:
  os.mkdir(A.output_dir)
except OSError:
  pass

try:
  os.mkdir(A.output_dir_real)
except OSError:
  pass
  
# init data
wmaxi = np.zeros(nt)
ti    = np.zeros(nt)
tstepi= np.zeros(nt)
cnt = 0

# loop timesteps
for istep in time_list: #range(A.istep,A.tstep+1,A.tout):
  fdir  = A.input_dir+'Timestep'+str(istep)
  
  # get data
  vizB.correct_path_load_data(fdir+'/parameters.py')
  A.scal, A.nd, A.geoscal = vizB.parse_parameters_file('parameters',fdir)
  A.nd.istep, A.nd.dt, A.nd.t = vizB.parse_time_info_parameters_file('parameters',fdir)
  
  vizB.correct_path_load_data(fdir+'/out_xPV_ts'+str(istep)+'.py')
  A.grid = vizB.parse_grid_info('out_xPV_ts'+str(istep),fdir)
  A.P, A.Pc, A.Vsx, A.Vsz = vizB.parse_PV3_file('out_xPV_ts'+str(istep),fdir)
  
  vizB.correct_path_load_data(fdir+'/out_xEnth_ts'+str(istep)+'.py')
  A.Enth = vizB.parse_Enth_file('out_xEnth_ts'+str(istep),fdir)
  
  A.dx = A.grid.xc[1]-A.grid.xc[0]
  A.dz = A.grid.zc[1]-A.grid.zc[0]
  A.nx = A.grid.nx
  A.nz = A.grid.nz
  
  scalv = vizB.get_scaling(A,'v',A.dimensional,1)
  scalt = vizB.get_scaling(A,'t',A.dimensional,1)
  scalx = vizB.get_scaling(A,'x',A.dimensional,1)

  # - extract wmax, phimax,h0 for every timestep
  ti[cnt]     = A.nd.t
  wmaxi[cnt]  = np.max(A.Vsz[0:int(A.nz/2),0:int(A.nx/2)])
  tstepi[cnt] = istep
  cnt += 1

  os.system('rm -r '+fdir+'/__pycache__')

# - plot wmax vs time (new dir)
fig = plt.figure(1,figsize=(10,5))
ax = plt.subplot(1,1,1)
# ax.plot(ti*scalt,wmaxi*scalv, label = 's.s. value = '+str(wmaxi[-1]*scalv))
ax.plot(tstepi,wmaxi*scalv, label = 's.s. value = '+str(wmaxi[-1]*scalv))
plt.grid(True)
ax.legend()
# ax.set_xlabel('Time [yr]')
ax.set_xlabel('Timestep')
ax.set_ylabel('Wmax [cm/yr]')
plt.savefig(A.output_dir+'out_wmax_time.pdf', bbox_inches = 'tight')
plt.close()