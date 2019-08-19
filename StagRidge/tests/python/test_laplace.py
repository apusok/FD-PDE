# ----------------------------------------- #
# Run convergence tests for the Temp Equation
# A. LAPLACE (diffusion operator)
# B. Advection operator
# benchmark with StagRidge
# Adina Pusok, Aug 2019
# ----------------------------------------- #

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os

# Input file
f1 = 'diff_laplace.opts' # diffusion
#f2 = 'adv.opts' # advection

# Directories
dir_in  = './input/'
dir_log = './logfiles/'
dir_out = './output/'

print('# --------------------------------------- #')
print('# Laplace (diffusion) benchmark ')
print('# --------------------------------------- #')

# Parameters
# n = [40, 80, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
n = [40, 80, 100, 200] #, 300, 400]

# Run simulations
for nx in n:

    # Create output filename
    fout1 = dir_log+f1[:-5]+'_'+str(nx)+'.out'
    #fout2 = dir_log+f2[:-5]+'_'+str(nx)+'.out'

    # Run Stagridge with different resolutions
    str1 = '../src/stagridge_advdiff -options_file '+dir_in+f1+' -nx '+str(nx)+' -nz '+str(nx)+' > '+fout1
    print(str1)
    os.system(str1)

    #str2 = '../src/stagridgeadvdiff -options_file '+dir_in+f2+' -nx '+str(nx)+' -nz '+str(nx)+' > '+fout2
    #print(str2)
    #os.system(str2)

# Norm variables
nrm1 = np.zeros(len(n))
hx = np.zeros(len(n))
hz = np.zeros(len(n))

nrm1_adv = np.zeros(len(n))
hx_adv= np.zeros(len(n))
hz_adv = np.zeros(len(n))

# Parse output and save norm info
for i in range(0,len(n)):
    nx = n[i]

    # Create output filename
    fout1 = dir_log+f1[:-5]+'_'+str(nx)+'.out'
    #fout2 = dir_log+f2[:-5]+'_'+str(nx)+'.out'

    # Open file 1 and read
    f = open(fout1, 'r')
    for line in f:
        if 'Temperature:' in line:
            nrm1[i] = float(line[23:41])
        if 'Grid info:' in line:
            hx[i] = float(line[18:36])
            hz[i] = float(line[42:60])
    
    f.close()

    # # Open file 2 and read
    # f = open(fout2, 'r')
    # for line in f:
    #     if 'Temp:' in line:
    #        nrm1p[i] = float(line[16:34])
    #     if 'Grid info:' in line:
    #         hx_1e6[i] = float(line[18:36])
    #         hz_1e6[i] = float(line[42:60])
    
    # f.close()

x1 = [1e-3, 1e-2]
y1 = [1e-9, 1e-8]
x2 = [1e-3, 1e-2]
y2 = [1e-9, 1e-7]

# Plot convergence data
plt.figure(1,figsize=(12,6))

plt.subplot(121)
plt.grid(color='lightgray', linestyle=':')
plt.plot(np.log10(hx),np.log10(nrm1),'ko--',label='Tdiff')

plt.plot(np.log10(x1),np.log10(y1),'r-',label='slope=1')
plt.plot(np.log10(x2),np.log10(y2),'b-',label='slope=2')

plt.xlabel('log10(h)',fontweight='bold',fontsize=12)
plt.ylabel('log10||e||',fontweight='bold',fontsize=12)
plt.title('A. Diffusion',fontweight='bold',fontsize=16)
plt.legend()

# plt.subplot(122)
# plt.grid(color='lightgray', linestyle=':')
# plt.plot(np.log10(hx_adv),np.log10(nrm1p_adv),'ko--',label='Tadv')

# plt.plot(np.log10(x1),np.log10(y1),'r-',label='slope=1')
# plt.plot(np.log10(x2),np.log10(y2),'b-',label='slope=2')

# plt.xlabel('log10(h)',fontweight='bold',fontsize=12)
# plt.ylabel('log10||e||',fontweight='bold',fontsize=12)
# plt.title('B. Advection',fontweight='bold',fontsize=16)
# plt.legend()

# Print convergence orders:
hx10    = np.log10(hx)
nrm1a = np.log10(nrm1)

# hx10_adv    = np.log10(hx_adv)
# nrm1a_adv = np.log10(nrm1_adv)

# Perform linear regression
sldiff, intercept, r_value, p_value, std_err = linregress(hx10, nrm1a)
# sladv, intercept, r_value, p_value, std_err = linregress(hx10_adv, nrm1a_adv)

print('# --------------------------------------- #')
print('# Temperature equation convergence order:')
print('     (diffusion): T_slope = '+str(sldiff))
# print('     (advection): v_slope = '+str(sladv))

fname = dir_out+'test_advdiff_convergence.pdf'
plt.savefig(fname)

print('# --------------------------------------- #')
print('# Printed Temperature equation convergence results to: '+fname)
print('# --------------------------------------- #')

#plt.show()
