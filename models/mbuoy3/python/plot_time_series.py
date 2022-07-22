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

A.input = 'half_ridge_f0005_hc_cycles1'
A.output_path_dir = '../sims_full_ridge/Figures/'
A.path_dir = '../sims_full_ridge/'
log_file ='log_out_'+A.input+'.out'

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

A.ts, A.sol = vizB.parse_solver_log_file(A.input_dir+log_file)
vizB.plot_solver_residuals(A,A.output_dir+'out_solver_residuals')

A.flux = vizB.parse_outflux_log_file(A.input_dir+log_file)
vizB.plot_extract_outflux(A,A.output_dir+'out_outflux')
