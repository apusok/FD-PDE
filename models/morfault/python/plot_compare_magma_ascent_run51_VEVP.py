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
def load_data(A):
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
    A.nt_tkyr   = np.zeros(A.nt)
    A.nt_zmagma = np.zeros(A.nt)
    A.nt_xmagma = np.zeros(A.nt)
    A.nt_imagma = np.zeros(A.nt)
    A.nt_jmagma = np.zeros(A.nt)
    A.nt_phitip = np.zeros(A.nt)
    A.nt_vmagma = np.zeros(A.nt-1)
    A.nt_vfz = np.zeros(A.nt)
    A.nt_dotlam = np.zeros(A.nt)
    A.nt_eta = np.zeros(A.nt)
    A.nt_zeta = np.zeros(A.nt)
    A.nt_delta = np.zeros(A.nt)

    scalx = vizB.get_scaling(A,'x',1,1)
    scalt = vizB.get_scaling(A,'t',1,1)
    scaleps = vizB.get_scaling(A,'eps',1,0)
    scal_v_ms = vizB.get_scaling(A,'v',1,0)
    scal_x_m = vizB.get_scaling(A,'x',1,0)
    scalP = vizB.get_scaling(A,'P',1,1)
    scalv = vizB.get_scaling(A,'v',1,1)
    scaleta = vizB.get_scaling(A,'eta',1,0)

    # Loop over timesteps
    it = 0
    for istep in time_list:
      fdir  = A.input_dir+'Timestep'+str(istep)
      print('  >> >> '+'Timestep'+str(istep))

      vizB.correct_path_load_data(fdir+'/parameters.py')
      A.nd.istep, A.nd.dt, A.nd.t = vizB.parse_time_info_parameters_file('parameters',fdir)

      # Correct path for data
      vizB.correct_path_load_data(fdir+'/out_xphi_ts'+str(istep)+'.py')
      vizB.correct_path_load_data(fdir+'/out_xVel_ts'+str(istep)+'.py')
      vizB.correct_path_load_data(fdir+'/out_xplast_ts'+str(istep)+'.py')
      vizB.correct_path_load_data(fdir+'/out_matProp_ts'+str(istep)+'.py')
      
      # Get data
      A.phis = vizB.parse_Element_file('out_xphi_ts'+str(istep),fdir)
      A.Vfx, A.Vfz, A.Vx, A.Vz = vizB.parse_Vel_file('out_xVel_ts'+str(istep),fdir)
      A.dotlam = vizB.parse_Element_file('out_xplast_ts'+str(istep),fdir)
      A.matProp = vizB.parse_matProp_file('out_matProp_ts'+str(istep),fdir)

      # Extract 
      A.phi = 1.0 - A.phis
      A.nt_tkyr[it]  = A.nd.t*scalt/1e3

      zdike = np.zeros(A.nx)
      jdike = np.zeros(A.nx)
      for i in range(0,A.nx):
        zmax = -50.0
        jj   = 0
        for j in range(0,A.nz):
          if (A.phi[j,i]>1.0e-10):
            zmax = max(zmax,A.grid.zc[j]*scalx)
            jj   = j
        zdike[i] = zmax
        jdike[i] = jj

      A.nt_zmagma[it] = np.max(zdike)
      for i in range(0,A.nx):
        if (A.nt_zmagma[it]==zdike[i]):
          A.nt_xmagma[it] = A.grid.xc[i]*scalx
          A.nt_imagma[it] = i
          A.nt_jmagma[it] = jdike[i]

      A.nt_phitip[it] = A.phi[int(A.nt_jmagma[it]),int(A.nt_imagma[it])]
      A.nt_vfz[it] = (A.Vfz[int(A.nt_jmagma[it])+1,int(A.nt_imagma[it])]+A.Vfz[int(A.nt_jmagma[it]),int(A.nt_imagma[it])])*0.5*scalv
      A.nt_dotlam[it] = A.dotlam[int(A.nt_jmagma[it]-1),int(A.nt_imagma[it])]*scal_v_ms/scal_x_m
      A.nt_eta[it] = A.matProp.eta[int(A.nt_jmagma[it]-1),int(A.nt_imagma[it])]*scaleta
      A.nt_zeta[it] = A.matProp.zeta[int(A.nt_jmagma[it]-1),int(A.nt_imagma[it])]*scaleta
      A.nt_delta[it] = np.sqrt(1e-7*(A.nt_zeta[it]+4/3*A.nt_eta[it])/1.0)

      it += 1

      os.system('rm -r '+A.input_dir+'Timestep'+str(istep)+'/__pycache__')

      A.nt_vmagma = (A.nt_zmagma[1:]-A.nt_zmagma[:-1])/(A.nt_tkyr[1:]-A.nt_tkyr[:-1])*100 #cm/yr
      
    return A
  except OSError:
    print('Cannot open: '+A.input)
    return 0.0

# ---------------------------------
# Main script
# ---------------------------------

path_in ='/Users/apusok/Documents/morfault/'
path_out='/Users/apusok/Documents/morfault/Figures/'

sim0 = 'run51_VEVP_var_age_eta1e18_Vext0/'
sim1 = 'run51_VEVP_var_age_eta1e18_Vext0_etaK1e18/'
sim2 = 'run51_VEVP_var_age_eta1e18_Vext0_zeta1e18/'

# sim0 = 'run51_VEVP_eta1e19_Vext0/'
# sim1 = 'run51_VEVP_eta1e21_Vext0/'
# sim2 = 'run51_VEVP_var_age_eta1e18_Vext0_zeta1e18/'

read_data = 0

A0 = SimStruct()
A1 = SimStruct()
A2 = SimStruct()

A0.input = path_in+sim0
A1.input = path_in+sim1
A2.input = path_in+sim2

A0.output_dir= path_out
vizB.make_dir(A0.output_dir)

fname_pickle = A0.output_dir+'time_data_compare_magma_ascent_run51.txt'

if (read_data):
  # Read raw data
  A0 = load_data(A0)
  A1 = load_data(A1)
  A2 = load_data(A2)

  # Save data
  with open(fname_pickle,'wb') as fh:
      pickle.dump(A0, fh)
      pickle.dump(A1, fh)
      pickle.dump(A2, fh)

else:
  # Read data
  pickle_off = open(fname_pickle,'rb')
  A0 = pickle.load(pickle_off)
  A1 = pickle.load(pickle_off)
  A2 = pickle.load(pickle_off)
  pickle_off.close()

fig = plt.figure(1,figsize=(10,10))
nplots = 5

ax = plt.subplot(nplots,1,1)
# pl = ax.plot(A0.nt_tkyr,np.log10(A0.nt_phitip),'k-',linewidth=0.5,label=r'$\eta_K=10^{20}$ Pa.s, $\zeta_0=10^{19}$ Pa.s')
# pl = ax.plot(A1.nt_tkyr,np.log10(A1.nt_phitip),'r-',linewidth=0.5,label=r'$\eta_K=10^{18}$ Pa.s')
pl = ax.plot(A2.nt_tkyr,np.log10(A2.nt_phitip),'b-',linewidth=0.5,label=r'$\zeta_0=10^{18}$ Pa.s')
ax.legend()
ax.set_xlabel('Time (kyr)')
ax.set_ylabel(r'$\log_{10}\phi_{tip}$')

ax = plt.subplot(nplots,1,2)
# pl = ax.plot(A0.nt_tkyr,A0.nt_zmagma,'k-',linewidth=0.5,label=r'$\eta_K=10^{20}$ Pa.s, $\zeta_0=10^{19}$ Pa.s')
# pl = ax.plot(A1.nt_tkyr,A1.nt_zmagma,'r-',linewidth=0.5,label=r'$\eta_K=10^{18}$ Pa.s')
pl = ax.plot(A2.nt_tkyr,A2.nt_zmagma,'b-',linewidth=0.5,label=r'$\zeta_0=10^{18}$ Pa.s')
ax.set_xlabel('Time (kyr)')
ax.set_ylabel(r'Depth magma tip (km)')

ax = plt.subplot(nplots,1,3)
# pl = ax.plot(A0.nt_tkyr,A0.nt_xmagma,'k-',linewidth=0.5,label=r'$\eta_K=10^{20}$ Pa.s, $\zeta_0=10^{19}$ Pa.s')
# # pl = ax.plot(A1.nt_tkyr,A1.nt_xmagma,'r-',linewidth=0.5,label=r'$\eta_K=10^{18}$ Pa.s')
# # pl = ax.plot(A2.nt_tkyr,A2.nt_xmagma,'b-',linewidth=0.5,label=r'$\zeta_0=10^{18}$ Pa.s')
# ax.set_xlabel('Time (kyr)')
# ax.set_ylabel(r'x-location magma tip (km)')

# pl = ax.plot(A0.nt_tkyr,A0.nt_delta/1e3,'k-',linewidth=0.5,label=r'$\eta_K=10^{20}$ Pa.s, $\zeta_0=10^{19}$ Pa.s')
# pl = ax.plot(A1.nt_tkyr,A1.nt_delta/1e3,'r-',linewidth=0.5,label=r'$\eta_K=10^{18}$ Pa.s')
pl = ax.plot(A2.nt_tkyr,A2.nt_delta/1e3,'b-',linewidth=0.5,label=r'$\zeta_0=10^{18}$ Pa.s')
ax.set_xlabel('Time (kyr)')
ax.set_ylabel(r'$\delta_c$ at tip (km)')

# ax = plt.subplot(nplots,1,4)
tkyr_mid0 = (A0.nt_tkyr[1:]+A0.nt_tkyr[:-1])*0.5
tkyr_mid1 = (A1.nt_tkyr[1:]+A1.nt_tkyr[:-1])*0.5
tkyr_mid2 = (A2.nt_tkyr[1:]+A2.nt_tkyr[:-1])*0.5

# ind0 = np.where(A0.nt_tkyr>50)
# print(A0.nt_zmagma[ind0[0][0]])
# ind0 = np.where(A0.nt_tkyr>200)
# print(A0.nt_zmagma[ind0[0][0]])
# ind0 = np.where(A0.nt_tkyr>500)
# print(A0.nt_zmagma[ind0[0][0]])
# ind0 = np.where(A0.nt_tkyr>900)
# print(A0.nt_zmagma[ind0[0][0]])
# ind0 = np.where(A0.nt_tkyr>1175)
# print(A0.nt_zmagma[ind0[0][0]])

# ind0 = np.where(A1.nt_tkyr>30)
# print(A1.nt_zmagma[ind0[0][0]])
# ind0 = np.where(A1.nt_tkyr>100)
# print(A1.nt_zmagma[ind0[0][0]])
# ind0 = np.where(A1.nt_tkyr>499)
# print(A1.nt_zmagma[ind0[0][0]])
# ind0 = np.where(A1.nt_tkyr>863)
# print(A1.nt_zmagma[ind0[0][0]])

# ind0 = np.where(A2.nt_tkyr>50)
# print(A2.nt_zmagma[ind0[0][0]])
# ind0 = np.where(A2.nt_tkyr>200)
# print(A2.nt_zmagma[ind0[0][0]])
# ind0 = np.where(A2.nt_tkyr>1050)
# print(A2.nt_zmagma[ind0[0][0]])
# ind0 = np.where(A2.nt_tkyr>1315)
# print(A2.nt_zmagma[ind0[0][0]])

# pl = ax.plot(tkyr_mid0,A0.nt_vmagma,'k-',linewidth=0.5,label=r'$\eta_K=10^{20}$ Pa.s, $\zeta_0=10^{19}$ Pa.s')
# # pl = ax.plot(tkyr_mid1,A1.nt_vmagma,'r-',linewidth=0.5,label=r'$\eta_K=10^{18}$ Pa.s')
# # pl = ax.plot(tkyr_mid2,A2.nt_vmagma,'b-',linewidth=0.5,label=r'$\zeta_0=10^{18}$ Pa.s')
# ax.set_xlabel('Time (kyr)')
# ax.set_ylabel(r'Tip ascent rate (cm/yr)')

ax = plt.subplot(nplots,1,4)
# pl = ax.plot(A0.nt_tkyr,A0.nt_vfz,'k-',linewidth=0.5,label=r'$\eta_K=10^{20}$ Pa.s, $\zeta_0=10^{19}$ Pa.s')
# pl = ax.plot(A1.nt_tkyr,A1.nt_vfz,'r-',linewidth=0.5,label=r'$\eta_K=10^{18}$ Pa.s')
pl = ax.plot(A2.nt_tkyr,A2.nt_vfz,'b-',linewidth=0.5,label=r'$\zeta_0=10^{18}$ Pa.s')
ax.set_xlabel('Time (kyr)')
ax.set_ylabel(r'$V_\ell^z$ at tip (cm/yr)')

ax = plt.subplot(nplots,1,5)
# pl = ax.plot(A0.nt_tkyr,A0.nt_dotlam,'k-',linewidth=0.5,label=r'$\eta_K=10^{20}$ Pa.s, $\zeta_0=10^{19}$ Pa.s')
# pl = ax.plot(A1.nt_tkyr,A1.nt_dotlam,'r-',linewidth=0.5,label=r'$\eta_K=10^{18}$ Pa.s')
pl = ax.plot(A2.nt_tkyr,A2.nt_dotlam,'b-',linewidth=0.5,label=r'$\zeta_0=10^{18}$ Pa.s')
ax.set_xlabel('Time (kyr)')
ax.set_ylabel(r'$\dot{\lambda}$ at tip (1/s)')

plt.savefig(A0.output_dir+'time_data_compare_magma_ascent_run51.png', bbox_inches = 'tight')
plt.close()