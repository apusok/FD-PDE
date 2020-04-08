# ------------------------------------------------ #
# MMS test to verify 2 non-dimensionalization schemes (Rhebergen et al. 2014, Katz-Magma dynamics)
# ------------------------------------------------ #

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os
import importlib
import dmstagoutput as dmout
from matplotlib import rc

# # Some new font
# rc('font',**{'family':'serif','serif':['Times new roman']})
# rc('text', usetex=True)

# Input file
f1 = 'out_mms_compare_nd'

# Parameters
n = [20, 40, 80, 100, 200, 300, 400]

alpha_i = [1e6] #[1e0, 1e3, 1e6]
R_i     = [1e0] #[1e0, 1e1, 1e2, 1e3]
e3_i    = [1.0] #[0.0, 1.0] # unit vector in the z-dir (remove if ignore buoyancy)

# Other parameters
phi_0 = 0.1 # phi_0 = phi_s
phi_s = 0.1 
p_s   = 1.0
psi_s = 1.0
U_s   = 1.0
m     = 2.0
nexp  = 3.0

cmaps='RdBu_r' 

for ixx in alpha_i:
  for iyy in R_i:
    for izz in e3_i:
      alpha = ixx
      R = iyy
      e3 = izz

      # Run simulations and plot solution and error
      for nx in n:
        # Create output filename
        fout1 = f1+'_'+str(nx)+'.out'

        # Run with different resolutions
        str1 = '../test_stokesdarcy2field_mms_compare_nd.app -pc_type lu -pc_factor_mat_solver_type umfpack -snes_monitor -ksp_monitor'+ \
          ' -alpha '+str(alpha)+' -R '+str(R)+ \
          ' -phi_0 '+str(phi_0)+' -phi_s '+str(phi_s)+' -p_s '+str(p_s)+' -psi_s '+str(psi_s)+ \
          ' -U_s '+str(U_s)+' -m '+str(m)+' -n '+str(nexp)+' -e3 '+str(e3)+ \
          ' -nx '+str(nx)+' -nz '+str(nx)+' > '+fout1
        print(str1)
        os.system(str1)

        # Get mms solution data
        f1out = 'out_mms_solution'
        imod = importlib.import_module(f1out)
        data = imod._PETScBinaryLoad()
        imod._PETScBinaryLoadReportNames(data)

        mx = data['Nx'][0]
        mz = data['Ny'][0]
        uxmms = data['X_face_x']
        uzmms= data['X_face_y']
        pmms = data['X_cell']

        # Get solution data MMS1
        f1out = 'out_solution_test1'
        imod = importlib.import_module(f1out)
        data = imod._PETScBinaryLoad()
        imod._PETScBinaryLoadReportNames(data)

        x = data['x1d_vertex']
        z = data['y1d_vertex']
        xc = data['x1d_cell']
        zc = data['y1d_cell']
        ux1 = data['X_face_x']
        uz1= data['X_face_y']
        p1 = data['X_cell']

        # Get solution data MMS2
        f1out = 'out_solution_test2'
        imod = importlib.import_module(f1out)
        data = imod._PETScBinaryLoad()
        imod._PETScBinaryLoadReportNames(data)

        ux2 = data['X_face_x']
        uz2= data['X_face_y']
        p2 = data['X_cell']

        # Get extra parameters
        f1out = 'out_extra_parameters'
        imod = importlib.import_module(f1out)
        data = imod._PETScBinaryLoad()
        imod._PETScBinaryLoadReportNames(data)

        elem  = data['X_cell']
        dof0 = 7 # element

        psi = elem[0::dof0]
        U   = elem[1::dof0]
        phi = elem[2::dof0]
        curl_psix = elem[3::dof0]
        gradUx    = elem[4::dof0]
        curl_psiz = elem[5::dof0]
        gradUz    = elem[6::dof0]

        # Compute center velocities - MMS1
        ux_sq = ux1.reshape(mz  ,mx+1)
        uz_sq = uz1.reshape(mz+1,mx  )

        curl_psix = curl_psix.reshape(mz,mx)
        curl_psiz = curl_psiz.reshape(mz,mx)
        gradUx    = gradUx.reshape(mz,mx)
        gradUz    = gradUz.reshape(mz,mx)

        # Compute the cell center values from the face data by averaging neighbouring faces
        uxc = np.zeros([mz,mx])
        uzc = np.zeros([mz,mx])
        for i in range(0,mx):
          for j in range(0,mz):
            uxc[j][i] = 0.5 * (ux_sq[j][i+1] + ux_sq[j][i])
            uzc[j][i] = 0.5 * (uz_sq[j+1][i] + uz_sq[j][i])

        # Plot data - mms, solution and errors for P, ux, uz
        fig = plt.figure(1,figsize=(20,12))

        ax = plt.subplot(3,5,1)
        im = ax.imshow(pmms.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps)
        ax.set_title('a1) MMS P', fontweight='bold')
        ax.set_ylabel('z')
        cbar = fig.colorbar(im,ax=ax, shrink=0.75)

        ax = plt.subplot(3,5,2)
        im = ax.imshow(p1.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps)
        ax.set_title('a2) Numerical P (ND1)', fontweight='bold')
        cbar = fig.colorbar(im,ax=ax, shrink=0.75)

        ax = plt.subplot(3,5,3)
        im = ax.imshow(p2.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps)
        ax.set_title('a3) Numerical P (ND2)', fontweight='bold')
        cbar = fig.colorbar(im,ax=ax, shrink=0.75)

        ax = plt.subplot(3,5,4)
        im = ax.imshow(pmms.reshape(mz,mx)-p1.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps)
        ax.set_title('a4) Error P (ND1)', fontweight='bold')
        cbar = fig.colorbar(im,ax=ax, shrink=0.75)

        ax = plt.subplot(3,5,5)
        im = ax.imshow(pmms.reshape(mz,mx)-p2.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps)
        ax.set_title('a5) Error P (ND2)', fontweight='bold')
        cbar = fig.colorbar(im,ax=ax, shrink=0.75)

        ax = plt.subplot(3,5,6)
        im = ax.imshow(uxmms.reshape(mz,mx+1),extent=[min(x), max(x), min(zc), max(zc)],cmap=cmaps)
        ax.set_title('b1) MMS ux', fontweight='bold')
        ax.set_ylabel('z')
        cbar = fig.colorbar(im,ax=ax, shrink=0.75)

        ax = plt.subplot(3,5,7)
        im = ax.imshow(ux1.reshape(mz,mx+1),extent=[min(x), max(x), min(zc), max(zc)],cmap=cmaps)
        ax.set_title('b2) Numerical ux (ND1)', fontweight='bold')
        cbar = fig.colorbar(im,ax=ax, shrink=0.75)

        ax = plt.subplot(3,5,8)
        im = ax.imshow(ux2.reshape(mz,mx+1),extent=[min(x), max(x), min(zc), max(zc)],cmap=cmaps)
        ax.set_title('b3) Numerical ux (ND2)', fontweight='bold')
        cbar = fig.colorbar(im,ax=ax, shrink=0.75)

        ax = plt.subplot(3,5,9)
        im = ax.imshow(uxmms.reshape(mz,mx+1)-ux1.reshape(mz,mx+1),extent=[min(x), max(x), min(zc), max(zc)],cmap=cmaps)
        ax.set_title('b4) Error ux (ND1)', fontweight='bold')
        cbar = fig.colorbar(im,ax=ax, shrink=0.75)

        ax = plt.subplot(3,5,10)
        im = ax.imshow(uxmms.reshape(mz,mx+1)-ux2.reshape(mz,mx+1),extent=[min(x), max(x), min(zc), max(zc)],cmap=cmaps)
        ax.set_title('b5) Error ux (ND2)', fontweight='bold')
        cbar = fig.colorbar(im,ax=ax, shrink=0.75)

        ax = plt.subplot(3,5,11)
        im = ax.imshow(uzmms.reshape(mz+1,mx),extent=[min(xc), max(xc), min(z), max(z)],cmap=cmaps)
        ax.set_title('c1) MMS uz', fontweight='bold')
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        cbar = fig.colorbar(im,ax=ax, shrink=0.75)

        ax = plt.subplot(3,5,12)
        im = ax.imshow(uz1.reshape(mz+1,mx),extent=[min(xc), max(xc), min(z), max(z)],cmap=cmaps)
        ax.set_title('c2) Numerical uz (ND1)', fontweight='bold')
        ax.set_xlabel('x')
        cbar = fig.colorbar(im,ax=ax, shrink=0.75)

        ax = plt.subplot(3,5,13)
        im = ax.imshow(uz2.reshape(mz+1,mx),extent=[min(xc), max(xc), min(z), max(z)],cmap=cmaps)
        ax.set_title('c3) Numerical uz (ND2)', fontweight='bold')
        ax.set_xlabel('x')
        cbar = fig.colorbar(im,ax=ax, shrink=0.75)

        ax = plt.subplot(3,5,14)
        im = ax.imshow(uzmms.reshape(mz+1,mx)-uz1.reshape(mz+1,mx),extent=[min(xc), max(xc), min(z), max(z)],cmap=cmaps)
        ax.set_title('c4) Error uz (ND1)', fontweight='bold')
        ax.set_xlabel('x')
        cbar = fig.colorbar(im,ax=ax, shrink=0.75)

        ax = plt.subplot(3,5,15)
        im = ax.imshow(uzmms.reshape(mz+1,mx)-uz2.reshape(mz+1,mx),extent=[min(xc), max(xc), min(z), max(z)],cmap=cmaps)
        ax.set_title('c5) Error uz (ND2)', fontweight='bold')
        ax.set_xlabel('x')
        cbar = fig.colorbar(im,ax=ax, shrink=0.75)

        plt.tight_layout() 
        plt.savefig(f1+'_nx_'+str(nx)+'.pdf')
        plt.close()

        # Plot data - mms, solution and errors for P, ux, uz
        fig = plt.figure(1,figsize=(12,4))
        nind = int(nx/20)

        ax = plt.subplot(131)
        im = ax.imshow(psi.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps)
        Q  = ax.quiver( xc[::nind], zc[::nind], curl_psix[::nind,::nind], curl_psiz[::nind,::nind], units='width', pivot='mid' )
        ax.set_title('Scalar: '+r'$\psi$'+', Vector: '+r'$\nabla\times\phi\mathbf{\hat{k}}$')
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        cbar = fig.colorbar(im,ax=ax, ticks=np.linspace(np.around(np.min(psi),decimals=1), np.around(np.max(psi),decimals=1), 5), shrink=0.75, extend='both')

        ax = plt.subplot(132)
        im = ax.imshow(U.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps)
        Q  = ax.quiver( xc[::nind], zc[::nind], gradUx[::nind,::nind], gradUz[::nind,::nind], units='width', pivot='mid' )
        ax.set_title('Scalar: '+r'$U$'+', Vector: '+r'$\nabla U$')
        ax.set_xlabel('x')
        cbar = fig.colorbar(im,ax=ax, ticks=np.linspace(np.around(np.min(U), decimals=1), np.around(np.max(U), decimals=1), 5), shrink=0.75, extend='both')

        ax = plt.subplot(133)
        im = ax.imshow(phi.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps)
        Q  = ax.quiver( xc[::nind], zc[::nind], uxc[::nind,::nind], uzc[::nind,::nind], units='width', pivot='mid' )
        ax.set_title('Scalar: '+r'$\phi$'+', Vector: '+r'$\mathbf{v_s}$')
        ax.set_xlabel('x')
        cbar = fig.colorbar(im,ax=ax, ticks=np.linspace(np.around(np.min(phi), decimals=2), np.around(np.max(phi), decimals=2), 5), shrink=0.75, extend='both')

        plt.savefig(f1+'_extra_nx_'+str(nx)+'.pdf')
        plt.close()

      # Norm variables
      nrm1_v = np.zeros(len(n))
      nrm1_vx = np.zeros(len(n))
      nrm1_vz = np.zeros(len(n))
      nrm1_p = np.zeros(len(n))
      nrm2_v = np.zeros(len(n))
      nrm2_vx = np.zeros(len(n))
      nrm2_vz = np.zeros(len(n))
      nrm2_p = np.zeros(len(n))
      hx = np.zeros(len(n))
      line_ind = 5

      # Parse output and save norm info
      for i in range(0,len(n)):
          nx = n[i]

          fout1 = f1+'_'+str(nx)+'.out'

          # Open file 1 and read
          f = open(fout1, 'r')
          for line in f:
            if 'Velocity MMS1:' in line:
                nrm1_v[i] = float(line[20+line_ind:38+line_ind])
                nrm1_vx[i] = float(line[48+line_ind:66+line_ind])
                nrm1_vz[i] = float(line[76+line_ind:94+line_ind])
            if 'Pressure MMS1:' in line:
                nrm1_p[i] = float(line[20+line_ind:38+line_ind])
            if 'Velocity MMS2:' in line:
                nrm2_v[i] = float(line[20+line_ind:38+line_ind])
                nrm2_vx[i] = float(line[48+line_ind:66+line_ind])
                nrm2_vz[i] = float(line[76+line_ind:94+line_ind])
            if 'Pressure MMS2:' in line:
                nrm2_p[i] = float(line[20+line_ind:38+line_ind])
            if 'Grid info MMS1:' in line:
                hx[i] = float(line[18+line_ind:36+line_ind])
          f.close()

      # Plot convergence data
      plt.figure(1,figsize=(6,6))

      plt.grid(color='lightgray', linestyle=':')

      plt.plot(n,nrm1_v,'k+-',label='v (non-dim 1)')
      plt.plot(n,nrm1_p,'ko--',label='P (non-dim 1)')
      plt.plot(n,nrm2_v,'r+-',label='v (non-dim 2)')
      plt.plot(n,nrm2_p,'ro--',label='P (non-dim 2)')

      plt.xscale("log")
      plt.yscale("log")
      
      plt.ylim(bottom=1e-5,top=1e4)
      plt.xlabel('$\sqrt{N}, N=n^2$',fontweight='bold',fontsize=12)
      plt.ylabel('$E(P), E(v_s)$',fontweight='bold',fontsize=12)
      plt.title('alpha='+str(alpha)+' R='+str(R)+' e3='+str(e3), fontweight='bold',fontsize=12)
      plt.legend()

      plt.savefig(f1+'.pdf')
      plt.close()

      # Print convergence orders:
      hx_log    = np.log10(n)
      nrmv1_log = np.log10(nrm1_v)
      nrmp1_log = np.log10(nrm1_p)
      nrmv2_log = np.log10(nrm2_v)
      nrmp2_log = np.log10(nrm2_p)

      # Perform linear regression
      slv1, intercept, r_value, p_value, std_err = linregress(hx_log, nrmv1_log)
      slp1, intercept, r_value, p_value, std_err = linregress(hx_log, nrmp1_log)
      slv2, intercept, r_value, p_value, std_err = linregress(hx_log, nrmv2_log)
      slp2, intercept, r_value, p_value, std_err = linregress(hx_log, nrmp2_log)

      print('# --------------------------------------- #')
      print('# MMS StokesDarcy2Field convergence order for different non-dimensionalizations:')
      print('    MMS1 Rhebergen et al. 2014: v_slope = '+str(slv1)+' p_slope = '+str(slp1))
      print('    MMS2 Katz, Magma dynamics : v_slope = '+str(slv2)+' p_slope = '+str(slp2))

      os.system('rm -r __pycache__')

      # fdir = 'mms_compare_nd_alpha_'+str(alpha).replace('.','-')+'_R_'+str(R).replace('.','-')+'_e3_'+str(e3).replace('.','-')+'/'
      fdir = 'mms_compare_nd_alpha_'+str(alpha)+'_R_'+str(R)+'_e3_'+str(e3)+'/'
      os.system('mkdir '+fdir)
      os.system('mv out_* '+fdir)
