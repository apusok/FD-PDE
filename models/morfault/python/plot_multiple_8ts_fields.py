# Import modules
import os
import sys
import numpy as np

# Add path to vizMORfault
pathViz = './'
sys.path.append(pathViz)
import vizMORfault as vizB

class SimStruct:
  pass

def sortTimesteps(tdir):
  return int(tdir[8:])

# ---------------------------------
def load_data_ts(A,sdir,istep):
  try:
    A.dimensional = 1
    A.input = sdir

    # Read parameters file and get scaling params
    fdir = A.input+'Timestep'+str(istep)
    
    # Correct path for data
    vizB.correct_path_load_data(fdir+'/parameters.py')
    vizB.correct_path_load_data(fdir+'/out_xT_ts'+str(istep)+'.py')
    vizB.correct_path_load_data(fdir+'/out_xphi_ts'+str(istep)+'.py')
    # vizB.correct_path_load_data(fdir+'/out_xPlith_ts'+str(istep)+'.py')
    vizB.correct_path_load_data(fdir+'/out_xDP_ts'+str(istep)+'.py')
    vizB.correct_path_load_data(fdir+'/out_xDPold_ts'+str(istep)+'.py')
    vizB.correct_path_load_data(fdir+'/out_xPV_ts'+str(istep)+'.py')
    vizB.correct_path_load_data(fdir+'/out_xeps_ts'+str(istep)+'.py')
    vizB.correct_path_load_data(fdir+'/out_xtau_ts'+str(istep)+'.py')
    vizB.correct_path_load_data(fdir+'/out_xtauold_ts'+str(istep)+'.py')
    vizB.correct_path_load_data(fdir+'/out_xplast_ts'+str(istep)+'.py')
    vizB.correct_path_load_data(fdir+'/out_matProp_ts'+str(istep)+'.py')
    vizB.correct_path_load_data(fdir+'/out_xVel_ts'+str(istep)+'.py')
    vizB.correct_path_load_data(fdir+'/out_xplast_ts'+str(istep)+'.py')
    vizB.correct_path_load_data(fdir+'/out_xstrain_ts'+str(istep)+'.py')
    
    # Get data
    A.scal, A.nd, A.geoscal = vizB.parse_parameters_file('parameters',fdir)
    A.lbl = vizB.create_labels()
    A.grid = vizB.parse_grid_info('out_xT_ts'+str(istep),fdir)
    A.nd.istep, A.nd.dt, A.nd.t = vizB.parse_time_info_parameters_file('parameters',fdir)

    # For easy access
    A.dx = A.grid.xc[1]-A.grid.xc[0]
    A.dz = A.grid.zc[1]-A.grid.zc[0]
    A.nx = A.grid.nx
    A.nz = A.grid.nz
    
    # Get data
    A.T = vizB.parse_Element_file('out_xT_ts'+str(istep),fdir)
    A.phis = vizB.parse_Element_file('out_xphi_ts'+str(istep),fdir)
    # A.Plith = vizB.parse_Element_file('out_xPlith_ts'+str(istep),fdir)
    A.DP    = vizB.parse_Element_file('out_xDP_ts'+str(istep),fdir)
    A.DPold = vizB.parse_Element_file('out_xDPold_ts'+str(istep),fdir)
    A.P, A.Vsx, A.Vsz = vizB.parse_PV_file('out_xPV_ts'+str(istep),fdir)
    A.eps = vizB.parse_Tensor_file('out_xeps_ts'+str(istep),fdir)
    A.tau = vizB.parse_Tensor_file('out_xtau_ts'+str(istep),fdir)
    A.tauold = vizB.parse_Tensor_file('out_xtauold_ts'+str(istep),fdir)
    A.matProp = vizB.parse_matProp_file('out_matProp_ts'+str(istep),fdir)
    A.Vfx, A.Vfz, A.Vx, A.Vz = vizB.parse_Vel_file('out_xVel_ts'+str(istep),fdir)
    A.dotlam = vizB.parse_Element_file('out_xplast_ts'+str(istep),fdir)
    A.lam = vizB.parse_Element_file('out_xstrain_ts'+str(istep),fdir)

    # Center velocities and mass divergence
    A.Vscx, A.Vscz, A.divVs = vizB.calc_center_velocities_div(A.Vsx,A.Vsz,A.grid.xv,A.grid.zv,A.nx,A.nz)
    A.rheol = vizB.calc_VEVP_strain_rates(A)
    
    vizB.correct_path_marker_data(fdir+'/out_pic_ts'+str(istep)+'.xmf')
    A.mark = vizB.parse_marker_file('out_pic_ts'+str(istep)+'.xmf',fdir)
    
    os.system('rm -r '+A.input+'Timestep'+str(istep)+'/__pycache__')

    return A
  except OSError:
    print('Cannot open: '+A.input)
    return 0.0

# ---------------------------------
# Main script
# ---------------------------------
#ts = [0, 2000, 2300, 2360, 2520, 2700, 2780, 3000]
#sim = 'run41_01_SD_setup6_sigmabc1e-2/'

#ts = [0, 700, 830, 860, 900, 930, 1000, 1200]
#sim = 'run41_01_SD_setup2_phibc5e-3_sigmabc1e-2_phimin1e-4/'

#ts = [0, 600, 850, 1220, 1250, 1650, 1770, 2200]
#sim = 'run43_01a_SD_setup6_Kphi1e-7_sigmabc1e-1/'

ts = [0, 280, 800, 900, 920, 1000, 1500, 3000]
sim = 'run43_01a_SD_setup6_Kphi1e-7_sigmabc1e-2_upwind/'

#ts = [0, 500, 1000, 1080, 1140, 2000, 3000, 4000]
#sim = 'run43_01a_SD_setup6_Kphi1e-7_sigmabc1e-3/'

#ts = [0, 4000, 6000, 6850, 7000, 7220, 7300, 7500]
#sim = 'run43_01a_SD_setup6_Kphi1e-8_sigmabc1e-1/'

#ts = [0, 3000, 3690, 3900, 4500, 6000, 7000, 7500]
#sim = 'run43_01a_SD_setup6_Kphi1e-8_sigmabc1e-2/'

#ts = [0, 500, 670, 690, 750, 1370, 1420, 2000]
#sim = 'run43_01b_SD_setup2_Kphi1e-7_sigmabc1e-2/'

#ts = [0, 700, 890, 910, 930, 1100, 1800, 2000]
#sim = 'run43_01b_SD_setup2_Kphi1e-7_sigmabc1e-3/'

#ts = [0, 3000, 3630, 4000, 4780, 5000, 6000, 7000]
#sim = 'run43_01b_SD_setup2_Kphi1e-8_sigmabc1e-2/'

#ts = [0, 3000, 4160, 4180, 4500, 5000, 6000, 7000]
#sim = 'run43_01b_SD_setup2_Kphi1e-8_sigmabc1e-3/'

#ts = [0, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
#sim = 'run43_00_S_age2/'

# ts = [0, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
# sim = 'run43_00_S_age5/'

#ts = [0, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
#sim = 'run43_00_S_age10/'

# sdir = '/Users/Adina/Bitbucket/riftomat-private/models/morfault/'+sim
# outdir = '../Figures/'+sim

#sdir = '/Users/apusok/Documents/morfault/'+sim
#outdir = '/Users/apusok/Documents/morfault/Figures/'+sim

sdir ='/Users/apusok/Documents/morfault/'+sim
outdir='/Users/apusok/Documents/morfault/Figures/'+sim

print(' >> '+outdir)

vizB.make_dir(outdir)

A1 = SimStruct()
A2 = SimStruct()
A3 = SimStruct()
A4 = SimStruct()
A5 = SimStruct()
A6 = SimStruct()
A7 = SimStruct()
A8 = SimStruct()

A1 = load_data_ts(A1,sdir,ts[0])
A2 = load_data_ts(A2,sdir,ts[1])
A3 = load_data_ts(A3,sdir,ts[2])
A4 = load_data_ts(A4,sdir,ts[3])
A5 = load_data_ts(A5,sdir,ts[4])
A6 = load_data_ts(A6,sdir,ts[5])
A7 = load_data_ts(A7,sdir,ts[6])
A8 = load_data_ts(A8,sdir,ts[7])

# plot part/entire domain
istart = 60
iend   = 141
jstart = 0
jend   = A1.nz

# Plot fields
print(' >> porosity')
vizB.plot_8multiple_phi(A1,A2,A3,A4,A5,A6,A7,A8,istart,iend,jstart,jend,outdir,'out_plot_8multiple_phi')
print(' >> epsII')
vizB.plot_8multiple_epsII(A1,A2,A3,A4,A5,A6,A7,A8,istart,iend,jstart,jend,outdir,'out_plot_8multiple_epsII')
print(' >> divvs')
vizB.plot_8multiple_divvs(A1,A2,A3,A4,A5,A6,A7,A8,istart,iend,jstart,jend,outdir,'out_plot_8multiple_divvs')
print(' >> tauII')
vizB.plot_8multiple_tauII(A1,A2,A3,A4,A5,A6,A7,A8,istart,iend,jstart,jend,outdir,'out_plot_8multiple_tauII')
print(' >> DP')
vizB.plot_8multiple_DP(A1,A2,A3,A4,A5,A6,A7,A8,istart,iend,jstart,jend,outdir,'out_plot_8multiple_DP')
print(' >> dotlam')
vizB.plot_8multiple_dotlam(A1,A2,A3,A4,A5,A6,A7,A8,istart,iend,jstart,jend,outdir,'out_plot_8multiple_dotlam')
print(' >> lam')
vizB.plot_8multiple_lam(A1,A2,A3,A4,A5,A6,A7,A8,istart,iend,jstart,jend,outdir,'out_plot_8multiple_lam')
print(' >> etaeff')
vizB.plot_8multiple_etaeff(A1,A2,A3,A4,A5,A6,A7,A8,istart,iend,jstart,jend,outdir,'out_plot_8multiple_etaeff')
print(' >> zetaeff')
vizB.plot_8multiple_zetaeff(A1,A2,A3,A4,A5,A6,A7,A8,istart,iend,jstart,jend,outdir,'out_plot_8multiple_zetaeff')
print(' >> PV')
vizB.plot_8multiple_PV(A1,A2,A3,A4,A5,A6,A7,A8,istart,iend,jstart,jend,outdir,'out_plot_8multiple_PV')
print(' >> Vfx')
vizB.plot_8multiple_Vfx(A1,A2,A3,A4,A5,A6,A7,A8,istart,iend,jstart,jend,outdir,'out_plot_8multiple_Vfx')
print(' >> Vfz')
vizB.plot_8multiple_Vfz(A1,A2,A3,A4,A5,A6,A7,A8,istart,iend,jstart,jend,outdir,'out_plot_8multiple_Vfz')
print(' >> epsV_II')
vizB.plot_8multiple_epsVEVP_II(A1,A2,A3,A4,A5,A6,A7,A8,istart,iend,jstart,jend,outdir,'out_plot_8multiple_epsV_II',0)
print(' >> epsE_II')
vizB.plot_8multiple_epsVEVP_II(A1,A2,A3,A4,A5,A6,A7,A8,istart,iend,jstart,jend,outdir,'out_plot_8multiple_epsE_II',1)
print(' >> epsVP_II')
vizB.plot_8multiple_epsVEVP_II(A1,A2,A3,A4,A5,A6,A7,A8,istart,iend,jstart,jend,outdir,'out_plot_8multiple_epsVP_II',2)
print(' >> C_V')
vizB.plot_8multiple_C_VEVP(A1,A2,A3,A4,A5,A6,A7,A8,istart,iend,jstart,jend,outdir,'out_plot_8multiple_C_V',0)
print(' >> C_E')
vizB.plot_8multiple_C_VEVP(A1,A2,A3,A4,A5,A6,A7,A8,istart,iend,jstart,jend,outdir,'out_plot_8multiple_C_E',1)
print(' >> C_VP')
vizB.plot_8multiple_C_VEVP(A1,A2,A3,A4,A5,A6,A7,A8,istart,iend,jstart,jend,outdir,'out_plot_8multiple_C_VP',2)
