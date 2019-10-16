# ----------------------------------------- #
# Run and visualize ../test_dmstagoutput.app
# ----------------------------------------- #

# Import modules
import os
import dmstagoutput as dmout

test0 = '../test_dmstagoutput.app'
print('# --------------------------------------- #')
print('# Running test: '+test0)
print('# --------------------------------------- #')

# Run test
os.system('mpiexec -n 2 ../test_dmstagoutput.app')

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