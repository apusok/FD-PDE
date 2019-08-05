# ----------------------------------------- #
# Run convergence tests for the SolCx 
# benchmark with StagRidge
# Adina Pusok, July 2019
# ----------------------------------------- #

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os

# Input file
f1 = 'solcx_1e0.opts' # isoviscous
f2 = 'solcx_1e6.opts' # variable viscosity

# Directories
dir_in  = './input/'
dir_log = './logfiles/'
dir_out = './output/'

print('# --------------------------------------- #')
print('# SolCx benchmark ')
print('# --------------------------------------- #')

# Parameters
# n = [40, 80, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
n = [40, 80, 100, 200] #, 300, 400]

# Run simulations
for nx in n:

    # Create output filename
    fout1 = dir_log+f1[:-5]+'_'+str(nx)+'.out'
    fout2 = dir_log+f2[:-5]+'_'+str(nx)+'.out'

    # Run Stagridge with different resolutions
    str1 = '../src/stagridge -options_file '+dir_in+f1+' -nx '+str(nx)+' -nz '+str(nx)+' > '+fout1
    print(str1)
    os.system(str1)

    str2 = '../src/stagridge -options_file '+dir_in+f2+' -nx '+str(nx)+' -nz '+str(nx)+' > '+fout2
    print(str2)
    os.system(str2)

# Norm variables
nrm1v = np.zeros(len(n))
nrm1vx = np.zeros(len(n))
nrm1vz = np.zeros(len(n))
nrm1p = np.zeros(len(n))
hx = np.zeros(len(n))
hz = np.zeros(len(n))

nrm1v_1e6 = np.zeros(len(n))
nrm1vx_1e6 = np.zeros(len(n))
nrm1vz_1e6 = np.zeros(len(n))
nrm1p_1e6 = np.zeros(len(n))
hx_1e6 = np.zeros(len(n))
hz_1e6 = np.zeros(len(n))

# Parse output and save norm info
for i in range(0,len(n)):
    nx = n[i]

    # Create output filename
    fout1 = dir_log+f1[:-5]+'_'+str(nx)+'.out'
    fout2 = dir_log+f2[:-5]+'_'+str(nx)+'.out'

    # Open file 1 and read
    f = open(fout1, 'r')
    for line in f:
        if 'Velocity:' in line:
            nrm1v[i] = float(line[20:38])
            nrm1vx[i] = float(line[48:66])
            nrm1vz[i] = float(line[76:94])
        if 'Pressure:' in line:
            nrm1p[i] = float(line[20:38])
        if 'Grid info:' in line:
            hx[i] = float(line[18:36])
            hz[i] = float(line[42:60])
    
    f.close()

    # Open file 2 and read
    f = open(fout2, 'r')
    for line in f:
        if 'Velocity:' in line:
            nrm1v_1e6[i] = float(line[20:38])
            nrm1vx_1e6[i] = float(line[48:66])
            nrm1vz_1e6[i] = float(line[76:94])
        if 'Pressure:' in line:
            nrm1p_1e6[i] = float(line[20:38])
        if 'Grid info:' in line:
            hx_1e6[i] = float(line[18:36])
            hz_1e6[i] = float(line[42:60])
    
    f.close()

x1 = [1e-3, 1e-2]
y1 = [1e-9, 1e-8]
x2 = [1e-3, 1e-2]
y2 = [1e-9, 1e-7]

# Plot convergence data
plt.figure(1,figsize=(12,6))

plt.subplot(121)
plt.grid(color='lightgray', linestyle=':')
plt.plot(np.log10(hx),np.log10(nrm1v),'k+--',label='v')
plt.plot(np.log10(hx),np.log10(nrm1p),'ko--',label='P')

plt.plot(np.log10(x1),np.log10(y1),'r-',label='slope=1')
plt.plot(np.log10(x2),np.log10(y2),'b-',label='slope=2')

plt.xlabel('log10(h)',fontweight='bold',fontsize=12)
plt.ylabel('log10||e||',fontweight='bold',fontsize=12)
plt.title('A. Isoviscous',fontweight='bold',fontsize=16)
plt.legend()

plt.subplot(122)
plt.grid(color='lightgray', linestyle=':')
plt.plot(np.log10(hx_1e6),np.log10(nrm1v_1e6),'k+--',label='v')
plt.plot(np.log10(hx_1e6),np.log10(nrm1p_1e6),'ko--',label='P')

plt.plot(np.log10(x1),np.log10(y1),'r-',label='slope=1')
plt.plot(np.log10(x2),np.log10(y2),'b-',label='slope=2')

plt.xlabel('log10(h)',fontweight='bold',fontsize=12)
plt.ylabel('log10||e||',fontweight='bold',fontsize=12)
plt.title('B. 1e6 viscosity jump',fontweight='bold',fontsize=16)
plt.legend()

# Print convergence orders:
hx10    = np.log10(hx)
nrm1v10 = np.log10(nrm1v)
nrm1p10 = np.log10(nrm1p)

hx10_1e6    = np.log10(hx_1e6)
nrm1v10_1e6 = np.log10(nrm1v_1e6)
nrm1p10_1e6 = np.log10(nrm1p_1e6)

# Perform linear regression
sl1e0v, intercept, r_value, p_value, std_err = linregress(hx10, nrm1v10)
sl1e0p, intercept, r_value, p_value, std_err = linregress(hx10, nrm1p10)
sl1e6v, intercept, r_value, p_value, std_err = linregress(hx10_1e6, nrm1v10_1e6)
sl1e6p, intercept, r_value, p_value, std_err = linregress(hx10_1e6, nrm1p10_1e6)

print('# --------------------------------------- #')
print('# SolCx convergence order:')
print('     (isoviscous 1e0): v_slope = '+str(sl1e0v)+' p_slope = '+str(sl1e0p))
print('     (visc contr 1e6): v_slope = '+str(sl1e6v)+' p_slope = '+str(sl1e6p))

fname = dir_out+'test_solcx_convergence.pdf'
plt.savefig(fname)

print('# --------------------------------------- #')
print('# Printed SolCx convergence results to: '+fname)
print('# --------------------------------------- #')

#plt.show()
