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

# ---------------------------------------
# Main
# ---------------------------------------
A = SimStruct()

sim = 'run37_flow_dike_dt1e2_etaK1e20_abs/'
A.input = '../'+sim
A.output_path_dir = '../Figures/'+sim
A.path_dir = './'
log_file ='log_out.out'

# Create directories
A.input_dir = A.path_dir+A.input
A.output_dir = A.output_path_dir

print('# OUTPUT: Solver residuals '+A.input_dir)

try:
  os.mkdir(A.output_dir)
except OSError:
  pass

A.sol = vizB.parse_solver_log_file(A.input_dir+log_file)
# print(A.sol.PViter)
vizB.plot_solver_residuals(A,A.output_dir+'out_solver_residuals')