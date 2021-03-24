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

A.input = 'modelA_D1guard_bulk1_1e-6'
A.output_path_dir = '../bulk_viscosity/Figures'
A.path_dir = '../bulk_viscosity/'

# Create directories
A.input_dir = A.path_dir+A.input+'/'
A.output_dir = A.output_path_dir+'/'+A.input+'/'

print('# OUTPUT: Time series '+A.input_dir)

try:
  os.mkdir(A.output_path_dir)
except OSError:
  pass

try:
  os.mkdir(A.output_dir)
except OSError:
  pass

A.ts, A.sol = vizB.parse_solver_log_file(A.input_dir+'log_out.out')
vizB.plot_solver_residuals(A,A.output_dir+'out_solver_residuals')

A.sill = vizB.parse_sillflux_log_file(A.input_dir+'log_out.out')
vizB.plot_sill_outflux(A,A.output_dir+'out_sillflux')