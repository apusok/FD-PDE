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

# Plot sill outflux
A.input = 'modelA_00_ext2_xsill3'
A.output_path_dir = '../extraction_mechanism/Figures'
A.path_dir = '../extraction_mechanism/'

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

A.ts, A.sill, A.sol = vizB.parse_log_file(A.input_dir+'log_out.out')
vizB.plot_sill_outflux(A,A.output_dir+'out_sill_flux')
vizB.plot_solver_residuals(A,A.output_dir+'out_solver_residuals')

# print(A.sol.PVres[4200:])
# print(A.sol.dt[4200:])
# print(A.sill.C[4200:])


