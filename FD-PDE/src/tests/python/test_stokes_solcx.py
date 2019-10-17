# ---------------------------------------
# SOLCX benchmark - constant grid spacing (convergence test)
# 1. Convergence test
# 2. Plot solution for isoviscous, variable viscosity
# ---------------------------------------

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os

# Input file
f1 = 'out_solcx_1e0' # isoviscous
f2 = 'out_solcx_1e6' # variable viscosity

print('# --------------------------------------- #')
print('# SolCx benchmark ')
print('# --------------------------------------- #')

# Parameters
n = [40, 80, 100, 200, 300, 400]

# Run simulations
for nx in n:

    # Create output filename
    fout1 = f1+'_'+str(nx)+'.out'
    fout2 = f2+'_'+str(nx)+'.out'

    # Run with different resolutions
    str1 = '../test_stokes_solcx.app -pc_type lu -pc_factor_mat_solver_type umfpack -output_file '+f1+' -nx '+str(nx)+' -nz '+str(nx)+' > '+fout1
    print(str1)
    os.system(str1)

    str2 = '../test_stokes_solcx.app -pc_type lu -pc_factor_mat_solver_type umfpack -eta1 1.0e6 -output_file '+f2+' -nx '+str(nx)+' -nz '+str(nx)+' > '+fout2
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
    fout1 = f1+'_'+str(nx)+'.out'
    fout2 = f2+'_'+str(nx)+'.out'

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

fname = 'out_test_solcx_convergence.pdf'
plt.savefig(fname)

print('# --------------------------------------- #')
print('# Printed SolCx convergence results to: '+fname)
print('# --------------------------------------- #')

nind = 15

# Plot solution - ISOVISCOUS
# Import the auto-generated python script
import out_solcx_1e0 as solcx0

# Load the data
data = solcx0._PETScBinaryLoad()
solcx0._PETScBinaryLoadReportNames(data)

# Get data
m = data['Nx'][0]
n = data['Ny'][0]
xv = data['x1d_vertex']
yv = data['y1d_vertex']
xc = data['x1d_cell']
yc = data['y1d_cell']
vxface = data['X_face_x']
vyface = data['X_face_y']
p = data['X_cell']

# Compute the DOF count on each stratum (face,element)
facedof_x = int(len(vxface)/((m+1)*n))
facedof_y = int(len(vyface)/(m*(n+1)))
celldof   = int(len(p)/(m*n))

# Prepare cell center velocities
vxface = vxface.reshape(n  ,m+1)
vyface = vyface.reshape(n+1,m  )

# Compute the cell center values from the face data by averaging neighbouring faces
vxc = np.zeros( [n , m] )
vyc = np.zeros( [n , m] )
for i in range(0,m):
  for j in range(0,n):
    vxc[j][i] = 0.5 * (vxface[j][i+1] + vxface[j][i])
    vyc[j][i] = 0.5 * (vyface[j+1][i] + vyface[j][i])

# Open a figure
fig, ax1 = plt.subplots(1)
# contours = plt.contour( xc , yc , p.reshape(n,m), 5, colors='white' )
# plt.clabel(contours, inline=True, fontsize=8 )
im = plt.imshow( p.reshape(n,m), extent=[min(xv), max(xv), min(yv), max(yv)],
                origin='lower', cmap='RdBu', interpolation='nearest' )
Q = plt.quiver( xc[::nind], yc[::nind], vxc[::nind,::nind], vyc[::nind,::nind], units='width', pivot='mid' )
plt.axis(aspect='image')
plt.xlabel('x-dir')
plt.ylabel('z-dir')
plt.title('SolCx benchmark - isoviscous')
cbar = fig.colorbar( im, ax=ax1 )
cbar.ax.set_ylabel('pressure')
plt.savefig(f1+'.pdf')

# Plot solution - VARVISC
# Import the auto-generated python script
import out_solcx_1e6 as solcx6

# Load the data
data = solcx6._PETScBinaryLoad()
solcx6._PETScBinaryLoadReportNames(data)

# Get number of cells in i,j
m = data['Nx'][0]
n = data['Ny'][0]
xv = data['x1d_vertex']
yv = data['y1d_vertex']
xc = data['x1d_cell']
yc = data['y1d_cell']
vxface = data['X_face_x']
vyface = data['X_face_y']
p = data['X_cell']

# Compute the DOF count on each stratum (face,element)
facedof_x = int(len(vxface)/((m+1)*n))
facedof_y = int(len(vyface)/(m*(n+1)))
celldof   = int(len(p)/(m*n))

# Prepare cell center velocities
vxface = vxface.reshape(n  ,m+1)
vyface = vyface.reshape(n+1,m  )

# Compute the cell center values from the face data by averaging neighbouring faces
vxc = np.zeros( [n , m] )
vyc = np.zeros( [n , m] )
for i in range(0,m):
  for j in range(0,n):
    vxc[j][i] = 0.5 * (vxface[j][i+1] + vxface[j][i])
    vyc[j][i] = 0.5 * (vyface[j+1][i] + vyface[j][i])

# Open a figure
fig, ax1 = plt.subplots(1)
# contours = plt.contour( xc , yc , p.reshape(n,m), 5, colors='white' )
# plt.clabel(contours, inline=True, fontsize=8 )
im = plt.imshow( p.reshape(n,m), extent=[min(xv), max(xv), min(yv), max(yv)],
                origin='lower', cmap='RdBu', interpolation='nearest' )
Q = plt.quiver( xc[::nind], yc[::nind], vxc[::nind,::nind], vyc[::nind,::nind], units='xy', pivot='mid')
plt.axis(aspect='image')
plt.xlabel('x-dir')
plt.ylabel('z-dir')
plt.title('SolCx benchmark - var viscosity')
cbar = fig.colorbar( im, ax=ax1)
cbar.ax.set_ylabel('pressure')
plt.savefig(f2+'.pdf')