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

A.input = 'modelA_04'
A.output_path_dir = '../Figures'
A.path_dir = '../'

# Parameters
A.dim_output = 1
A.H     = 1.0
A.tout  = 100
A.tstep = 5000
A.istep = 0
ext = 'pdf'

# Create directories
A.input_dir = A.path_dir+A.input+'/'
A.output_dir = A.output_path_dir+'/'+A.input+'/'
A.output_dir_real = A.output_path_dir+'/'+A.input+'/porosity_stream_'+ext

print('# OUTPUT: Porosity field with temperature contours and solid velocity streamlines ')
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

# Plot sill outflux
A.ts, A.sill, A.sol = vizB.parse_log_file(A.input_dir+'log_out.out')

# Visualize data
for istep in range(A.istep,A.tstep+1,A.tout):
  fdir  = A.input_dir+'Timestep'+str(istep)

  # Correct data
  vizB.correct_path_load_data(fdir+'/out_xPV_ts'+str(istep)+'.py')
  vizB.correct_path_load_data(fdir+'/out_xHC_ts'+str(istep)+'.py')
  vizB.correct_path_load_data(fdir+'/out_xphiT_ts'+str(istep)+'.py')
  vizB.correct_path_load_data(fdir+'/out_xVel_ts'+str(istep)+'.py')

  # Get data
  A.P, A.Vsx, A.Vsz = vizB.parse_PV_file('out_xPV_ts'+str(istep),fdir)
  A.H, A.C = vizB.parse_HC_file('out_xHC_ts'+str(istep),fdir)
  A.phi, A.T = vizB.parse_HC_file('out_xphiT_ts'+str(istep),fdir)
  A.Vfx, A.Vfz, A.Vx, A.Vz = vizB.parse_Vel_file('out_xVel_ts'+str(istep),fdir)
  A.Vscx, A.Vscz = vizB.calc_center_velocities(A.Vsx,A.Vsz,A.nx,A.nz)
  A.Vfcx, A.Vfcz = vizB.calc_center_velocities(A.Vfx,A.Vfz,A.nx,A.nz)

  # output
  vizB.plot_porosity_solid_stream(A,A.output_dir_real+'/out_porosity_solid_stream'+str(istep),istep,ext)

  os.system('rm -r '+A.input_dir+'Timestep'+str(istep)+'/__pycache__')

os.system('rm -r '+pathViz+'/__pycache__')
