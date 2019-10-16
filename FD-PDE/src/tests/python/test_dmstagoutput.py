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
os.system('../test_dmstagoutput.app')

# Visualize data
dmout.general_output_pcolormesh('test0','RdBu')
dmout.general_output_imshow('test1',None,'bilinear')
dmout.general_output_pcolormesh('test2',None)
dmout.general_output_pcolormesh('test3','inferno')
dmout.general_output_pcolormesh('test4',None)

# ierr = test0(n-1,n,2,2,2,"test0");CHKERRQ(ierr);
# ierr = test0(n,n,2,1,0,"test1");CHKERRQ(ierr); // can use imshow()
# ierr = test0(n+1,n,1,0,0,"test2");CHKERRQ(ierr);
# ierr = test0(n,n+1,0,1,0,"test3");CHKERRQ(ierr);
# ierr = test0(n+2,n,0,0,1,"test4");CHKERRQ(ierr);