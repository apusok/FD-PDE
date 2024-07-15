# Import modules
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import math

# Add path to vizMORfault
pathViz = './'
sys.path.append(pathViz)
import vizMORfault as vizB

rc('font',**{'family':'serif','serif':['Times new roman']})
rc('text', usetex=True)

import warnings
warnings.filterwarnings('ignore')

class SimStruct:
  pass

# ---------------------------------------
# Main
# ---------------------------------------
A = SimStruct() # Do print(A.__dict__) to see structure of A

def sortTimesteps(tdir):
  return int(tdir[8:])

def A1_coeff(C,sigmat,theta):
  return C*math.cos(math.pi*theta/180)-sigmat*math.sin(math.pi*theta/180)

def A2_coeff(C,Pl,phi,phi_min,theta):
  return C*math.cos(math.pi*theta/180)+(1-math.exp(-phi_min/phi))*Pl*math.sin(math.pi*theta/180)

def F_rate_indep(tauII,DP,A1,A2,theta):
  return (tauII**2+A1**2)**0.5-(A2+DP*math.sin(math.pi*theta/180))

def F_rate_dep(F,etaK,dotlam):
  return F - etaK*dotlam

def TensorSecondInvariant(axx,azz,axz):
  return (0.5*(axx*axx + azz*azz) + axz*axz)**0.5

# Parameters
A.dimensional = 1 # 0-nd, 1-dim
sim = 'run37_flow_dike_dt1e2_etaK1e20_abs/'
A.input = '/Users/apusok/Documents/morfault/'+sim
A.output_path_dir = '/Users/apusok/Documents/morfault/Figures/'+sim

i = 50
j = 54

istep_start = 1150
istep_end   = 1160
istep_jump  = 5
dim = 1

n = int((istep_end - istep_start)/istep_jump)+1
ts    = np.zeros(n)
time_yr  = np.zeros(n)
tauII = np.zeros(n)
DP    = np.zeros(n)
C    = np.zeros(n)
sigmat    = np.zeros(n)
theta    = np.zeros(n)
Pl    = np.zeros(n)
phi    = np.zeros(n)
phi_min    = np.zeros(n)
etaK    = np.zeros(n)
dotlam    = np.zeros(n)

tauII_VE = np.zeros(n)
DP_VE    = np.zeros(n)
eta_VE   = np.zeros(n)
zeta_VE  = np.zeros(n)
eIIp     = np.zeros(n)
divp     = np.zeros(n)

# PLOTTING
tauII0 = np.arange(0,125)
DP0 = np.arange(-200,100)

fig = plt.figure(1,figsize=(10,10))
ax = plt.subplot(1,1,1)
extentE=[DP0[0], DP0[-1], tauII0[0], tauII0[-1]]
color1 = plt.cm.Greys(np.linspace(0,1,n+1))

# search timesteps in folder
tdir = os.listdir(A.input)
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

# Create directories - default to check for files
A.input_dir = A.input
vizB.make_dir(A.output_path_dir)

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

# plot entire domain
istart = 0
iend   = A.nx
jstart = 0
jend   = A.nz

print('  >> '+A.input_dir)
ii = -1

# Loop over timesteps
# for istep in time_list:
for istep1 in range(istep_start,istep_end+1,istep_jump):
  istep = time_list[istep1]
  ii += 1

  fdir  = A.input_dir+'Timestep'+str(istep)
  print('  >> >> '+'Timestep'+str(istep))

  vizB.correct_path_load_data(fdir+'/parameters.py')
  A.nd.istep, A.nd.dt, A.nd.t = vizB.parse_time_info_parameters_file('parameters',fdir)

  # Correct path for data
  vizB.correct_path_load_data(fdir+'/out_xphi_ts'+str(istep)+'.py')
  vizB.correct_path_load_data(fdir+'/out_xPlith_ts'+str(istep)+'.py')
  vizB.correct_path_load_data(fdir+'/out_xDP_ts'+str(istep)+'.py')
  vizB.correct_path_load_data(fdir+'/out_xDPold_ts'+str(istep)+'.py')
  vizB.correct_path_load_data(fdir+'/out_xPV_ts'+str(istep)+'.py')
  vizB.correct_path_load_data(fdir+'/out_xeps_ts'+str(istep)+'.py')
  vizB.correct_path_load_data(fdir+'/out_xtau_ts'+str(istep)+'.py')
  vizB.correct_path_load_data(fdir+'/out_xtauold_ts'+str(istep)+'.py')
  vizB.correct_path_load_data(fdir+'/out_xplast_ts'+str(istep)+'.py')
  vizB.correct_path_load_data(fdir+'/out_matProp_ts'+str(istep)+'.py')
  vizB.correct_path_load_data(fdir+'/out_xstrain_ts'+str(istep)+'.py')

  # Get data
  A.phis = vizB.parse_Element_file('out_xphi_ts'+str(istep),fdir)
  A.Plith = vizB.parse_Element_file('out_xPlith_ts'+str(istep),fdir)
  A.DP    = vizB.parse_Element_file('out_xDP_ts'+str(istep),fdir)
  A.DPold = vizB.parse_Element_file('out_xDPold_ts'+str(istep),fdir)
  A.P, A.Vsx, A.Vsz = vizB.parse_PV_file('out_xPV_ts'+str(istep),fdir)
  A.eps = vizB.parse_Tensor_file('out_xeps_ts'+str(istep),fdir)
  A.tau = vizB.parse_Tensor_file('out_xtau_ts'+str(istep),fdir)
  A.tauold = vizB.parse_Tensor_file('out_xtauold_ts'+str(istep),fdir)
  A.matProp = vizB.parse_matProp_file('out_matProp_ts'+str(istep),fdir)
  A.dotlam = vizB.parse_Element_file('out_xplast_ts'+str(istep),fdir)
  A.lam = vizB.parse_Element_file('out_xstrain_ts'+str(istep),fdir)

  # Scaling
  scal_eta = vizB.get_scaling(A,'eta',dim,0)
  scal_P_Pa = vizB.get_scaling(A,'P',dim,0)
  scal_v_ms = vizB.get_scaling(A,'v',dim,0)
  scal_x_m = vizB.get_scaling(A,'x',dim,0)
  scal_t_s = vizB.get_scaling(A,'t',dim,0)
  scal_t_yr = vizB.get_scaling(A,'t',dim,1)

  # Extract variables
  ts[ii]    = istep
  time_yr[ii]  = A.nd.t*scal_t_yr
  tauII[ii] = A.tau.II_center[j,i]*scal_P_Pa
  DP[ii] = A.DP[j,i]*scal_P_Pa
  C[ii] = A.matProp.C[j,i]*scal_P_Pa
  sigmat[ii] = A.matProp.sigmat[j,i]*scal_P_Pa
  theta[ii] = A.matProp.theta[j,i]
  Pl[ii] = (A.P[j,i]+A.Plith[j,i])*scal_P_Pa
  phi[ii] = 1.0 - A.phis[j,i]
  phi_min[ii] = 1e-3
  etaK[ii] = 1e20
  dotlam[ii] = A.dotlam[j,i]*scal_v_ms/scal_x_m

  # visco-elastic stresses
  eta_VE[ii]   = 1.0/(1.0/A.matProp.etaV[j,i]+1.0/A.matProp.etaE[j,i])*scal_eta
  zeta_VE[ii]  = 1.0/(1.0/A.matProp.zetaV[j,i]+1.0/A.matProp.zetaE[j,i])*scal_eta

  eta_e  = A.matProp.etaE[j,i]*scal_eta
  zeta_e = A.matProp.zetaE[j,i]*scal_eta
  phis   = A.phis[j,i]

  exx = A.eps.xx_center[j,i]*scal_v_ms/scal_x_m 
  ezz = A.eps.zz_center[j,i]*scal_v_ms/scal_x_m 
  exz = A.eps.xz_center[j,i]*scal_v_ms/scal_x_m 
  eII = A.eps.II_center[j,i]*scal_v_ms/scal_x_m 
  div13 = (exx + ezz)/3.0

  told_xx = A.tauold.xx_center[j,i]*scal_P_Pa
  told_zz = A.tauold.zz_center[j,i]*scal_P_Pa
  told_xz = A.tauold.xz_center[j,i]*scal_P_Pa
  told_II = A.tauold.II_center[j,i]*scal_P_Pa
  DPold   = A.DPold[j,i]*scal_P_Pa
  
  exxp = ((exx-div13) + 0.5*phis*told_xx/eta_e)
  ezzp = ((ezz-div13) + 0.5*phis*told_zz/eta_e)
  exzp = (exz + 0.5*phis*told_xz/eta_e)

  eIIp[ii] = TensorSecondInvariant(exxp,ezzp,exzp)
  divp[ii] = ((exx+ezz) - phis*DPold/zeta_e)

  txx = 2*eta_VE[ii]*exxp/phis
  tzz = 2*eta_VE[ii]*ezzp/phis
  txz = 2*eta_VE[ii]*exzp/phis

  tauII_VE[ii] = TensorSecondInvariant(txx,tzz,txz)
  DP_VE[ii]    = -zeta_VE[ii]*divp[ii]/phis

  print('DP = ',DP[ii],'DP_VE = ',DP_VE[ii],'Cprime = ', divp[ii])

  DP_corr = DP[ii]-DP_VE[ii]
  cdl = phis*DP_corr/zeta_VE[ii]/dotlam[ii]/math.sin(math.pi*theta[ii]/180)
  zeta_mod = -phis*DP_VE[ii]/divp[ii]
  print('DP_corr = ',DP_corr)
  # print('cdl = ',cdl)
  # print('sin_theta = ',math.sin(math.pi*theta[ii]/180))
  # print('dQ/dPeff = ',-cdl*math.sin(math.pi*theta[ii]/180))
  print('zeta = ',A.matProp.zeta[j,i]*scal_eta)
  print('zeta_mod = ',zeta_mod)
  print('\n')

  # Failure envelope
  F0 = np.zeros([len(tauII0),len(DP0)])
  F  = np.zeros([len(tauII0),len(DP0)])

  A1 = A1_coeff(C[ii],sigmat[ii],theta[ii])
  A2 = A2_coeff(C[ii],Pl[ii],phi[ii],phi_min[ii],theta[ii])

  for ia in range(0,len(tauII0)):
    for ja in range(0,len(DP0)):
      F0[ia,ja] = F_rate_indep(tauII0[ia]*1e6,DP0[ja]*1e6,A1,A2,theta[ii])
      F [ia,ja] = F_rate_dep(F0[ia,ja],etaK[ii],dotlam[ii])

  # Energy dissipation
  etaV = A.matProp.etaV[j,i]*scal_eta
  zetaV = A.matProp.zetaV[j,i]*scal_eta
  EV_shear = phis**2/(2*etaV)*(txx**2+tzz**2+2*txz**2)
  EV_vol   = phis**2*DP[ii]**2/zetaV
  EV = EV_shear + EV_vol

  EVP_shear = phis*dotlam[ii]/2*(txx**2+tzz**2+2*txz**2)*(tauII[ii]/(tauII[ii]**2+A1**2)**0.5)
  EVP_vol   = -phis*cdl*math.sin(math.pi*theta[ii]/180)*DP[ii]*dotlam[ii]
  EVP = EVP_shear + EVP_vol
  E = EV + EVP

  print('Energy dissipation: Total = ',E)
  print('Shear: V = ',EV_shear,' VP = ',EVP_shear)
  print('Vol: V = ',EV_vol,' VP = ',EVP_vol)
  print('Total: V = ',EV,' VP = ',EVP)
  print('\n')

  # Plot DP vs tauII
  scal = 1e-6
  # im = ax.imshow(F,extent=extentE,origin='lower')
  # cbar = fig.colorbar(im,ax=ax, shrink=0.70)
  F_cont = ax.contour(DP0, tauII0, F*scal, levels=[0,],linewidths=(1.0,), extend='both',colors=(color1[ii+1],))
  if(DP[ii]>0) and (divp[ii]>0):
    ax.plot(DP_VE[ii]*scal,tauII[ii]*scal,'rs')
  if (ii==n-1):
    ax.plot(DP[ii]*scal,tauII[ii]*scal,'*',color=color1[ii+1],label=r'VEVP')
    ax.plot(DP_VE[ii]*scal,tauII_VE[ii]*scal,'o',color=color1[ii+1],label=r'VE')
  else:
    ax.plot(DP[ii]*scal,tauII[ii]*scal,'*',color=color1[ii+1])
    ax.plot(DP_VE[ii]*scal,tauII_VE[ii]*scal,'o',color=color1[ii+1])

# Wrap figure and save
ax.grid(True)
ax.set_xlabel(r'$\Delta P$ [MPa]')
ax.set_ylabel(r'$\tau_{II}$ [MPa]')
ax.set_title('TS white-black: '+str(istep_start)+':'+str(istep_jump)+':'+str(istep_end))
ax.legend(loc='lower right')

plt.savefig(A.output_path_dir+'stress_state.png', bbox_inches = 'tight')
plt.close()