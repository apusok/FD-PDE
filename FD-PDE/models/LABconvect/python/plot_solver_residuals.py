# Import modules
import os
import sys
import numpy as np

# Add path to vizLABconvect
pathViz = './'
sys.path.append(pathViz)
import vizLABconvect as vizB

class SimStruct:
  pass

# ---------------------------------------
# Main
# ---------------------------------------
A = SimStruct()

sim_name = 'test'
A.input_dir = '../'+sim_name+'/'
A.output_path_dir = '../Figures/'
log_file ='log_out_'+A.input+'.out'

# Create directories
A.output_dir = A.output_path_dir+'/'+sim_name+'/'

print('# OUTPUT: Time series '+sim_name)

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
