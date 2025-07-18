# ----------------------------------------- #
# Run and visualize ../test_dmstagoutput
# Warning: if python modules are written with a prefix (i.e. directory), the dmstagoutput and loading the files as modules will give an error. 
# Use the following instead:
#   fname = 'out_test'
#   spec = importlib.util.spec_from_file_location(fname,fname_dir+fname+'.py')
#   imod = importlib.util.module_from_spec(spec)
#   spec.loader.exec_module(imod) 
#   data = imod._PETScBinaryLoad()
# ----------------------------------------- #

# Import modules
import os
import dmstagoutput as dmout
import sys, getopt

test0 = '../test_dmstagoutput'
print('# --------------------------------------- #')
print('# Running test: '+test0)
print('# --------------------------------------- #')

# Input file
fname = 'out_dmstagoutput'
try:
  os.mkdir(fname)
except OSError:
  pass

# Get cpu number
ncpu = 1
options, remainder = getopt.getopt(sys.argv[1:],'n:')
for opt, arg in options:
  if opt in ('-n'):
    ncpu = int(arg)

# Run test
os.system('mpiexec -n '+str(ncpu)+' ../test_dmstagoutput')

# Visualize data
dmout.general_output_pcolormesh('out_test_dmstagoutput0','RdBu')
dmout.general_output_imshow('out_test_dmstagoutput1',None,'bilinear')
dmout.general_output_pcolormesh('out_test_dmstagoutput2',None)
dmout.general_output_pcolormesh('out_test_dmstagoutput3','inferno')
dmout.general_output_pcolormesh('out_test_dmstagoutput4',None)

# ierr = test0(n-1,n,2,2,2,"out_test_dmstagoutput0");CHKERRQ(ierr);
# ierr = test0(n,n,2,1,0,"out_test_dmstagoutput1");CHKERRQ(ierr); // can use imshow()
# ierr = test0(n+1,n,1,0,0,"out_test_dmstagoutput2");CHKERRQ(ierr);
# ierr = test0(n,n+1,0,1,0,"out_test_dmstagoutput3");CHKERRQ(ierr);
# ierr = test0(n+2,n,0,0,1,"out_test_dmstagoutput4");CHKERRQ(ierr);

# Move data to directories
os.system('mv -f out_test_dmstagoutput* '+fname)
os.system('rm -r __pycache__')
