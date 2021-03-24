# Import modules
import os
import sys
import numpy as np

# Add path to vizMORBuoyancy
pathViz = './'
sys.path.append(pathViz)
import vizMORBuoyancy as vizB

class SimStruct:
  pass

# ---------------------------------------
# Main
# ---------------------------------------
A = SimStruct()

# Parameters
A.dim_output = 1
A.H = 1.0
A.tout  = 100
A.tstep = 2000
A.istep = 0

# # Run test
# str1 = '../MORbuoyancy.app -options_file model_test.opts > log_run.out'
# print(str1)
# os.system(str1)

A.input = 'modelA_noD1guard_bulk1_1e-12'
A.output_path_dir = '../bulk_viscosity/Figures'
A.path_dir = '../bulk_viscosity/'

# Create directories
A.input_dir = A.path_dir+A.input+'/'
A.output_dir = A.output_path_dir+'/'+A.input+'/'
A.output_dir_real = A.output_path_dir+'/'+A.input+'/debug_output/'

print('# OUTPUT: Debug output for mid-ocean ridge model - Buoyancy ')
print('# INPUT: '+A.input_dir)

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

# Read parameters file and get scaling params
A.scal, A.lbl = vizB.get_scaling_labels(A,'parameters_file.out','Timestep0',A.dim_output)

# Read grid parameters - choose PV file timestep0
fdir = A.input_dir+'Timestep0'
fname = 'out_xPV_ts0'

vizB.correct_path_load_data(fdir+'/'+fname+'.py')
A.grid = vizB.parse_grid_info('out_xPV_ts0',fdir)
A.dx = A.grid.xc[1]-A.grid.xc[0]
A.dz = A.grid.zc[1]-A.grid.zc[0]
# print(A.__dict__)

A.nx = A.grid.nx
A.nz = A.grid.nz

# Get time data
A.ts, A.sill, A.sol = vizB.parse_log_file(A.input_dir+'log_out.out')

# Visualize data
for istep in range(A.istep,A.tstep+1,A.tout):
  fdir  = A.input_dir+'Timestep'+str(istep)

  # Correct data
  vizB.correct_path_load_data(fdir+'/out_xPV_ts'+str(istep)+'.py')
  vizB.correct_path_load_data(fdir+'/out_xHC_ts'+str(istep)+'.py')
  vizB.correct_path_load_data(fdir+'/out_xphiT_ts'+str(istep)+'.py')
  vizB.correct_path_load_data(fdir+'/out_xEnth_ts'+str(istep)+'.py')
  vizB.correct_path_load_data(fdir+'/out_xVel_ts'+str(istep)+'.py')
  vizB.correct_path_load_data(fdir+'/out_xHCcoeff_ts'+str(istep)+'.py')

  # Get data
  A.P, A.Vsx, A.Vsz = vizB.parse_PV_file('out_xPV_ts'+str(istep),fdir)
  A.H, A.C = vizB.parse_HC_file('out_xHC_ts'+str(istep),fdir)
  A.phi, A.T = vizB.parse_HC_file('out_xphiT_ts'+str(istep),fdir)
  A.Enth = vizB.parse_Enth_file('out_xEnth_ts'+str(istep),fdir)
  A.Vfx, A.Vfz, A.Vx, A.Vz = vizB.parse_Vel_file('out_xVel_ts'+str(istep),fdir)
  A.HC_coeff = vizB.parse_HCcoeff_file('out_xHCcoeff_ts'+str(istep),fdir)

  if (istep > 0):
    vizB.correct_path_load_data(fdir+'/out_xPVcoeff_ts'+str(istep)+'.py')
    vizB.correct_path_load_data(fdir+'/out_resPV_ts'+str(istep)+'.py')
    vizB.correct_path_load_data(fdir+'/out_resHC_ts'+str(istep)+'.py')
    vizB.correct_path_load_data(fdir+'/out_matProp_ts'+str(istep)+'.py')

    A.PV_coeff = vizB.parse_PVcoeff_file('out_xPVcoeff_ts'+str(istep),fdir)
    A.resP, A.resVsx, A.resVsz = vizB.parse_PV_file('out_resPV_ts'+str(istep),fdir)
    A.resH, A.resC = vizB.parse_HC_file('out_resHC_ts'+str(istep),fdir)
    A.matProp = vizB.parse_matProps_file('out_matProp_ts'+str(istep),fdir)

  A.Vscx, A.Vscz = vizB.calc_center_velocities(A.Vsx,A.Vsz,A.nx,A.nz)
  A.Vfcx, A.Vfcz = vizB.calc_center_velocities(A.Vfx,A.Vfz,A.nx,A.nz)
  A.Vcx, A.Vcz   = vizB.calc_center_velocities(A.Vx,A.Vz,A.nx,A.nz)
  A.divmass = vizB.calc_divergence(A.Vx,A.Vz,A.dx,A.dz,A.nx,A.nz)

  # nice output
  vizB.plot_porosity_contours(A,A.output_dir_real+'out_porosity_contours_ts'+str(istep),istep)
  # vizB.plot_temperature_slices(A,'out_temp_slices_ts'+str(istep),istep)

  # debug output
  vizB.plot_PV(0,A,A.output_dir_real+'out_xPV_ts'+str(istep),istep,7,12)
  vizB.plot_HC(0,A,A.output_dir_real+'out_xHC_ts'+str(istep),istep,7,8) # H,C
  vizB.plot_HC(1,A,A.output_dir_real+'out_xphiT_ts'+str(istep),istep,7,8) # phi,T
  vizB.plot_Enth(A,A.output_dir_real+'out_xEnth_ts'+str(istep),istep)
  vizB.plot_Vel(A,A.output_dir_real+'out_xVel_ts'+str(istep),istep)
  vizB.plot_HCcoeff(A,A.output_dir_real+'out_xHCcoeff_ts'+str(istep),istep)

  if (istep > 0):
    vizB.plot_PVcoeff(A,A.output_dir_real+'out_xPVcoeff_ts'+str(istep),istep)
    vizB.plot_PV(1,A,A.output_dir_real+'out_resPV_ts'+str(istep),istep,7,12) # res PV
    vizB.plot_HC(2,A,A.output_dir_real+'out_resHC_ts'+str(istep),istep,7,8) # res HC
    vizB.plot_matProp(A,A.output_dir_real+'out_matProp_ts'+str(istep),istep)

  os.system('rm -r '+A.input_dir+'Timestep'+str(istep)+'/__pycache__')

os.system('rm -r '+pathViz+'/__pycache__')
