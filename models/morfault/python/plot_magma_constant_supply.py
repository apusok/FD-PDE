# Import libraries
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

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
def load_porosity_data(A):
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

    A.Vphi    = np.zeros(A.nt)
    A.phi_max = np.zeros(A.nt)
    A.t_kyr   = np.zeros(A.nt)
    A.z_max   = np.zeros(A.nt)
    
    scalx = vizB.get_scaling(A,'x',1,1)
    scalt = vizB.get_scaling(A,'t',1,1)

    # Loop over timesteps
    it = 0
    for istep in time_list:
      fdir  = A.input_dir+'Timestep'+str(istep)
      print('  >> >> '+'Timestep'+str(istep))

      vizB.correct_path_load_data(fdir+'/parameters.py')
      A.nd.istep, A.nd.dt, A.nd.t = vizB.parse_time_info_parameters_file('parameters',fdir)

      # Correct path for data
      # vizB.correct_path_load_data(fdir+'/out_xT_ts'+str(istep)+'.py')
      vizB.correct_path_load_data(fdir+'/out_xphi_ts'+str(istep)+'.py')
      
      # Get data
      # A.T = vizB.parse_Element_file('out_xT_ts'+str(istep),fdir)
      A.phis = vizB.parse_Element_file('out_xphi_ts'+str(istep),fdir)
      A.phi = 1.0 - A.phis
      # print(A.phi[60:100  ,40:60])

      # Magma parameters
      A.t_kyr[it]   = A.nd.t*scalt/1.0e3
      A.Vphi[it]    = np.sum(A.phi)*A.dx*A.dz*scalx**2
      A.phi_max[it] = np.max(A.phi)

      zdike = np.zeros(A.nx)
      for i in range(0,A.nx):
        zmax = -50.0
        for j in range(0,A.nz):
          if (A.phi[j,i]>1.0e-10):
            zmax = max(zmax,A.grid.zc[j]*scalx)
        zdike[i] = zmax

      A.z_max[it] = np.max(zdike)

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

sims = ['run43_01a_SD_setup6_Kphi1e-7_sigmabc1e-2_dtvf/',
        'run43_01a_SD_setup6_Kphi1e-7_sigmabc1e-2_upwind/',
        'run43_01a_SD_setup6_Kphi1e-7_sigmabc1e-2_upwind_minmod/',
        'run43_01a_SD_setup6_Kphi1e-7_sigmabc1e-2_fromm_nocorr/',
        'run43_01a_SD_setup6_Kphi1e-7_sigmabc1e-2_upwind_nocorr/',
        'run43_01a_SD_setup6_Kphi1e-7_sigmabc1e-2_upwind_minmod_nocorr/'
        ]

lbl = ['0.1','0.01','0.001']

A0 = SimStruct()
A1 = SimStruct()
A2 = SimStruct()
A3 = SimStruct()
A4 = SimStruct()
A5 = SimStruct()

A0.input = path_in+sims[0]
A1.input = path_in+sims[1]
A2.input = path_in+sims[2]
A3.input = path_in+sims[3]
A4.input = path_in+sims[4]
A5.input = path_in+sims[5]

A0 = load_porosity_data(A0)
A1 = load_porosity_data(A1)
A2 = load_porosity_data(A2)
A3 = load_porosity_data(A3)
A4 = load_porosity_data(A4)
A5 = load_porosity_data(A5)

fig = plt.figure(1,figsize=(14,5))

y0 = (A0.Vphi-A0.Vphi[0])/A0.Vphi[0]
y1 = (A1.Vphi-A1.Vphi[0])/A1.Vphi[0]
y2 = (A2.Vphi-A2.Vphi[0])/A2.Vphi[0]
y3 = (A3.Vphi-A3.Vphi[0])/A3.Vphi[0]
y4 = (A4.Vphi-A4.Vphi[0])/A4.Vphi[0]
y5 = (A5.Vphi-A5.Vphi[0])/A5.Vphi[0]

# tmax = A1.t_kyr[-1]
# indx = np.where(A0.t_kyr>=tmax)

# print(A1.t_kyr)
# print(A1.Vphi)
# print(A1.phi_max)
# print(A1.z_max)

ax = plt.subplot(1,3,1)
# pl0 = ax.plot(y0[ts_start[0]:ts_start[0]+dt],'k-',label=r'$\sigma_{BC}=0.1$')
# pl1 = ax.plot(y1[ts_start[1]:ts_start[1]+dt],'r-',label=r'$\sigma_{BC}=0.01$')
#pl2 = ax.plot(y2[ts_start[2]:ts_start[2]+dt],'b-',label=r'$\sigma_{BC}=0.001$')
pl0 = ax.plot(A0.t_kyr,y0,'k-',label=r'fromm')
pl1 = ax.plot(A1.t_kyr,y1,'r-',label=r'upwind')
pl2 = ax.plot(A2.t_kyr,y2,'b-',label=r'upwind-minmod')
pl2 = ax.plot(A3.t_kyr,y3,'k--',label=r'fromm no corr')
pl1 = ax.plot(A4.t_kyr,y4,'r--',label=r'upwind no corr')
pl2 = ax.plot(A5.t_kyr,y5,'b--',label=r'upwind-minmod no corr')

# pl2 = ax.plot(A2.t_kyr,y2,'b-',label=r'iter PV-$\phi$, $\Delta t=5$ yr')
# ax.set_xlim(0,tmax)
#pl2 = ax.plot(A2.t_kyr,y2,'b-',label=r'$\sigma_{BC}=0.001$')
plt.grid(True)
ax.set_ylabel(r'$(V_{magma}-V_{magma}^0)/V_{magma}^0$')
ax.set_xlabel('Time (kyr)')
#ax.set_xlabel(str(dt)+r'$\times 100$ [yrs]')
# ax.set_xlabel(r'[kyrs]')
ax.legend()

ax = plt.subplot(1,3,2)
# pl0 = ax.plot(A0.phi_max[ts_start[0]:ts_start[0]+dt],'k-',label=r'$\sigma_{BC}=0.1$')
# pl1 = ax.plot(A1.phi_max[ts_start[1]:ts_start[1]+dt],'r-',label=r'$\sigma_{BC}=0.01$')
#pl2 = ax.plot(A2.phi_max[ts_start[2]:ts_start[2]+dt],'b-',label=r'$\sigma_{BC}=0.001$')
pl0 = ax.plot(A0.t_kyr,A0.phi_max,'k-',label=r'fromm')
pl1 = ax.plot(A1.t_kyr,A1.phi_max,'r-',label=r'upwind')
pl2 = ax.plot(A2.t_kyr,A2.phi_max,'b-',label=r'upwind-minmod')
pl2 = ax.plot(A3.t_kyr,A3.phi_max,'k--',label=r'fromm no corr')
pl1 = ax.plot(A4.t_kyr,A4.phi_max,'r--',label=r'upwind no corr')
pl2 = ax.plot(A5.t_kyr,A5.phi_max,'b--',label=r'upwind-minmod no corr')
# pl2 = ax.plot(A2.t_kyr,A2.phi_max,'b-',label=r'iter PV-$\phi$, $\Delta t=5$ yr')
# ax.set_xlim(0,tmax)
plt.grid(True)
#ax.set_ylabel(r'log$_{10}\phi$ max')
ax.set_xlabel('Time (kyr)')
ax.set_ylabel(r'$\phi$ max')
#ax.set_xlabel(str(dt)+r'$\times 100$ [yrs]')
# ax.set_xlabel(r'[kyrs]')
# ax.legend()

ax = plt.subplot(1,3,3)
# pl0 = ax.plot(A0.z_max[ts_start[0]:ts_start[0]+dt],'k-',label=r'$\sigma_{BC}=0.1$')
# pl1 = ax.plot(A1.z_max[ts_start[1]:ts_start[1]+dt],'r-',label=r'$\sigma_{BC}=0.01$')
#pl2 = ax.plot(A2.z_max[ts_start[2]:ts_start[2]+dt],'b-',label=r'$\sigma_{BC}=0.001$')
pl0 = ax.plot(A0.t_kyr,A0.z_max,'k-',label=r'fromm')
pl1 = ax.plot(A1.t_kyr,A1.z_max,'r-',label=r'upwind')
pl2 = ax.plot(A2.t_kyr,A2.z_max,'b-',label=r'upwind-minmod')
pl2 = ax.plot(A3.t_kyr,A3.z_max,'k--',label=r'fromm no corr')
pl1 = ax.plot(A4.t_kyr,A4.z_max,'r--',label=r'upwind no corr')
pl2 = ax.plot(A5.t_kyr,A5.z_max,'b--',label=r'upwind-minmod no corr')
# pl2 = ax.plot(A2.t_kyr,A2.z_max,'b-',label=r'iter PV-$\phi$, $\Delta t=5$ yr')
# ax.set_xlim(0,tmax)
plt.grid(True)
ax.set_ylabel(r'z max (km)')
ax.set_xlabel('Time (kyr)')
#ax.set_xlabel(str(dt)+r'$\times 100$ [yrs]')
# ax.set_xlabel(r'[kyrs]')
# ax.legend()

plt.savefig(path_out+'plot_run43_Kphi1e-7_magma_constant_supply_masscons_v2.png', bbox_inches = 'tight')
plt.close()
