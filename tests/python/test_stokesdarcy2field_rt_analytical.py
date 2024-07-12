# ---------------------------------------

# ---------------------------------------

def plot_allinone(fname,ft,nx):

  # First row P, Vx, Vz, |V|, eta
  # Load data
  fout = fname+'_solution'+ft
  #fout = fname+'_solution_initial'
  spec = importlib.util.spec_from_file_location(fout,fname+'/'+fout+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod)
  data = imod._PETScBinaryLoad()
  # imod = importlib.import_module(fout) 
  # data = imod._PETScBinaryLoad()
  #imod._PETScBinaryLoadReportNames(data)

  mx = data['Nx'][0]
  mz = data['Ny'][0]
  xc = data['x1d_cell']
  zc = data['y1d_cell']
  xv = data['x1d_vertex']
  zv = data['y1d_vertex']
  vx = data['X_face_x']
  vz = data['X_face_y']
  p  = data['X_cell']

  fout = fname+'_coefficient'+ft
  spec = importlib.util.spec_from_file_location(fout,fname+'/'+fout+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod)
  data = imod._PETScBinaryLoad()
  # imod = importlib.import_module(fout) 
  # data = imod._PETScBinaryLoad()
  #imod._PETScBinaryLoadReportNames(data)

  etac_data = data['X_cell']
  etan = data['X_vertex']

  rhsx_data = data['X_face_x']
  rhsy_data = data['X_face_y']

  # split into dofs
  dof = 3
  etac = etac_data[1::dof]

  rhsx = rhsx_data[0::dof]
  rhsy = rhsy_data[0::dof]

  # reshaped rhs
  rx = rhsx.reshape(mz, mx+1)
  ry = rhsy.reshape(mz+1, mx)

  # Prepare cell center velocities and pressure
  vxface = vx.reshape(mz  ,mx+1)
  vzface = vz.reshape(mz+1,mx  )
  pp = p.reshape(mz,mx)

  # Compute the cell center values from the face data by averaging neighbouring faces
  vxc = np.zeros( [mz , mx] )
  vzc = np.zeros( [mz , mx] )
  vc  = np.zeros( [mz , mx] )
  divv= np.zeros( [mz , mx] )
  dx  = np.zeros( [mx] )
  dz  = np.zeros( [mz] )
  dpdx = np.zeros( [mz , mx+1] )
  dpdz = np.zeros( [mz+1 , mx] )
  
  for i in range(0,mx):
    dx[i] = xv[i+1] - xv[i]
  for j in range(0,mz):
    dz[j] = zv[j+1] - zv[j]
  
  for i in range(0,mx):
    for j in range(0,mz):
      vxc[j][i] = 0.5 * (vxface[j][i+1] + vxface[j][i])
      vzc[j][i] = 0.5 * (vzface[j+1][i] + vzface[j][i])
      vc [j][i] = (vxc[j][i]**2+vzc[j][i]**2)**0.5
      divv[j][i] = (vxface[j][i+1] - vxface[j][i])/dx[i] + (vzface[j+1][i] - vzface[j][i])/dz[j]

      if i != 0:
        dpdx[j][i+1] = pp[j][i] - pp[j][i-1]
      if j != 0:
        dpdz[j+1][i] = pp[j][i] - pp[j-1][i]

  dpdxc = np.zeros([mz,mx])
  dpdzc = np.zeros([mz,mx])

  for i in range(0,mx):
    for j in range(0,mz):
      dpdxc[j][i] = 0.5* (dpdx[j][i+1] + dpdx[j][i])
      dpdzc[j][i] = 0.5* (dpdz[j+1][i] + dpdz[j][i])

  velmax = max(max(vx),max(vz))
  velmin = min(min(vx),min(vz))

  etamax = 3
  etamin =-3

  nind = 10

  # Plot all fields - P, vx, vz, v, etac, etan
  fig = plt.figure(1,figsize=(15,9))
  cmaps='RdBu_r' 

  ax = plt.subplot(3,5,1)
  im = ax.imshow(p.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps,origin='lower')
  Q  = ax.quiver( xc[::nind], zc[::nind], -dpdxc[::nind,::nind], -dpdzc[::nind,::nind], units='width', pivot='mid' )
  ax.set_title(r'$P_f, \phi(\mathbf{V}_f-\mathbf{V}_s)$')
  ax.set_title(r'$P_f$')
  ax.set_ylabel('z')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,5,2)
  im = ax.imshow(vx.reshape(mz,mx+1),extent=[min(xv), max(xv), min(zc), max(zc)],vmin=velmin,vmax=velmax,cmap=cmaps,origin='lower')
  ax.set_title(r'$V_x$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,5,3)
  im = ax.imshow(vz.reshape(mz+1,mx),extent=[min(xc), max(xc), min(zv), max(zv)],vmin=velmin,vmax=velmax,cmap=cmaps,origin='lower')
  ax.set_title(r'$V_z$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,5,4)
  im = ax.imshow(vc,extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps,origin='lower')
  ax.set_title(r'$V$ magnitude')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,5,5)
  im = ax.imshow(etac.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps,origin='lower')
  ax.set_title(r'$\eta_{center}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)


  #Second row: plot epsII, epsxx, epszz, epsxz 
    # Load data
  fout = fname+'_strain'+ft # 1. Numerical solution stokes - strain rates
  spec = importlib.util.spec_from_file_location(fout,fname+'/'+fout+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod)
  data = imod._PETScBinaryLoad()
  # imod = importlib.import_module(fout) 
  # data = imod._PETScBinaryLoad()
  #imod._PETScBinaryLoadReportNames(data)

  mx = data['Nx'][0]
  mz = data['Ny'][0]
  xc = data['x1d_cell']
  zc = data['y1d_cell']
  xv = data['x1d_vertex']
  zv = data['y1d_vertex']
  eps1n = data['X_vertex']
  eps1c = data['X_cell']

  # split into dofs
  dof = 4
  exx1c = eps1c[0::dof]
  ezz1c = eps1c[1::dof]
  exz1c = eps1c[2::dof]
  eII1c = eps1c[3::dof]

  exx1n = eps1n[0::dof]
  ezz1n = eps1n[1::dof]
  exz1n = eps1n[2::dof]
  eII1n = eps1n[3::dof]
  
  # get the deviatoric strain rates
  divc = exx1c + ezz1c
  exx1c = exx1c - 1.0/3.0*divc
  ezz1c = ezz1c - 1.0/3.0*divc
  eII1c = np.power(0.5*(np.power(exx1c,2)+np.power(ezz1c,2)+2.0*np.power(exz1c,2)) ,0.5)
  
  divn = exx1n + ezz1n
  exx1n = exx1n - 1.0/3.0*divn
  ezz1n = ezz1n - 1.0/3.0*divn
  eII1n = np.power(0.5*(np.power(exx1n,2)+np.power(ezz1n,2)+2.0*np.power(exz1n,2)) ,0.5)

  exxmax = max(max(exx1c),max(exx1n))
  exxmin = min(min(exx1c),min(exx1n))
  ezzmax = max(max(ezz1c),max(ezz1n))
  ezzmin = min(min(ezz1c),min(ezz1n))
  exzmax = max(max(exz1c),max(exz1n))
  exzmin = min(min(exz1c),min(exz1n))
  eIImax = max(max(eII1c),max(eII1n))
  eIImin = min(min(eII1c),min(eII1n))

  i1 = int(0.15/0.5 * mz)
  i2 = int(0.2/0.5 * mz)
  i3 = int(0.25/0.5 * mz)
  
  divmin = np.amin(divv[i1,:])
  divmax = np.amax(divv)
#  if divmin > -0.1:
#    divmin = -3


  ax = plt.subplot(3,5,6)
  im = ax.imshow(exx1c.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=exxmin,vmax=exxmax,cmap=cmaps,origin='lower')
  ax.set_title(r'$\epsilon_{xx}^{CENTER}$')
  ax.set_ylabel('z')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,5,7)
  im = ax.imshow(ezz1c.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=ezzmin,vmax=ezzmax,cmap=cmaps,origin='lower')
  ax.set_title(r'$\epsilon_{zz}^{CENTER}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,5,8)
  im = ax.imshow(exz1c.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=exzmin,vmax=exzmax,cmap=cmaps,origin='lower')
  ax.set_title(r'$\epsilon_{xz}^{CENTER}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,5,9)
  im = ax.imshow(eII1c.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=eIImin,vmax=eIImax,cmap=cmaps,origin='lower')
  ax.set_title(r'$\epsilon_{II}^{CENTER}$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  ax = plt.subplot(3,5,10)
  im = ax.imshow(divv,extent=[min(xc), max(xc), min(zc), max(zc)],vmin = divmin, vmax = divmax, cmap=cmaps,origin='lower')
  ax.set_title(r'$\nabla \cdot \mathbf{V}_s$')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  #Third row: Updated old value of tauII, tauxx, tauzz, tauxz
    # Load data
  fout = fname+'_stressold'+ft # historic stresses
  spec = importlib.util.spec_from_file_location(fout,fname+'/'+fout+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod)
  data = imod._PETScBinaryLoad()
  # imod = importlib.import_module(fout) 
  # data = imod._PETScBinaryLoad()
  #imod._PETScBinaryLoadReportNames(data)

  mx = data['Nx'][0]
  mz = data['Ny'][0]
  xc = data['x1d_cell']
  zc = data['y1d_cell']
  xv = data['x1d_vertex']
  zv = data['y1d_vertex']
  s1n = data['X_vertex']
  s1c = data['X_cell']

  fout = fname+'_dpold'+ft # yield stress
  spec = importlib.util.spec_from_file_location(fout,fname+'/'+fout+'.py')
  imod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(imod)
  data = imod._PETScBinaryLoad()
  # imod = importlib.import_module(fout) 
  # data = imod._PETScBinaryLoad()
  #imod._PETScBinaryLoadReportNames(data)

  dpold_data = data['X_cell']
  
  # split into dofs
  dof = 4
  sxx1c = s1c[0::dof]
  szz1c = s1c[1::dof]
  sxz1c = s1c[2::dof]
  sII1c = s1c[3::dof]

  sxx1n = s1n[0::dof]
  szz1n = s1n[1::dof]
  sxz1n = s1n[2::dof]
  sII1n = s1n[3::dof]

  dpold = dpold_data[0::dof]

  sxxmax = max(max(sxx1c),max(sxx1n))
  sxxmin = min(min(sxx1c),min(sxx1n))
  szzmax = max(max(szz1c),max(szz1n))
  szzmin = min(min(szz1c),min(szz1n))
  sxzmax = max(max(sxz1c),max(sxz1n))
  sxzmin = min(min(sxz1c),min(sxz1n))
  sIImax = max(max(sII1c),max(sII1n))
  sIImin = min(min(sII1c),min(sII1n))

  dpmax = max(dpold)
  dpmin = min(dpold)

  print(sxxmax,sxxmin,dpmax,dpmin)
  stress = sxxmax
 
  ax = plt.subplot(3,5,11)
  im = ax.imshow(sxx1c.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=sxxmin,vmax=sxxmax,cmap=cmaps,origin='lower')
  ax.set_title(r'$\tau_{xx}$(Center)')
  ax.set_ylabel('z')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75, format='%.2f')

  ax = plt.subplot(3,5,12)
  im = ax.imshow(szz1c.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=szzmin,vmax=szzmax,cmap=cmaps,origin='lower')
  ax.set_title(r'$\tau_{zz}$(Center)')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75, format='%.2f')

  ax = plt.subplot(3,5,13)
  im = ax.imshow(sxz1c.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=sxzmin,vmax=sxzmax,cmap=cmaps,origin='lower')
  ax.set_title(r'$\tau_{xz}$(Center)')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75, format='%.2f')

  ax = plt.subplot(3,5,14)
  im = ax.imshow(sII1c.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=sIImin,vmax=sIImax,cmap=cmaps,origin='lower')
  ax.set_title(r'$\tau_{II}$(Center)')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75, format='%.2f')

  ax = plt.subplot(3,5,15)
  im = ax.imshow(dpold.reshape(mz,mx),extent=[min(xc), max(xc), min(zc), max(zc)],vmin=dpmin,vmax=dpmax,cmap=cmaps,origin='lower')
  ax.set_title(r'$\Delta P$(Center)')
  cbar = fig.colorbar(im,ax=ax, shrink=0.75)

  plt.tight_layout() 
  
  fout = fname+'/'+fname+'_allinone'+ft+'_nx_'+str(nx)

  plt.savefig(fout+'.pdf')
  plt.close()

  return stress

def getk_analytical(e1, e2, l, wn):
    
    h1 = 0.5
    h2 = 0.5
    ld = l/wn
    p1 = 2*np.pi*h1/ld
    p2 = 2*np.pi*h2/ld
    
    c11 = 2*e1*p1**2/(e2*(np.cosh(2*p1) -1 -2*p1**2)) - \
        2*p2**2/(np.cosh(2*p2) -1 -2*p2**2)
    d12 = e1*(np.sinh(2*p1) - 2*p1)/(e2*(np.cosh(2*p1) -1 -2*p1**2)) + \
        (np.sinh(2*p2) - 2*p2)/(np.cosh(2*p2) -1 -2*p2**2)
    i21 = e1*p2*(np.sinh(2*p1) + 2*p1)/(e2*(np.cosh(2*p1) -1 -2*p1**2)) + \
        p2*(np.sinh(2*p2) + 2*p2)/(np.cosh(2*p2) -1 -2*p2**2)
    j22 = 2*e1*p1**2*p2/(e2*(np.cosh(2*p1) -1 -2*p1**2)) - \
        2*p2**3/(np.cosh(2*p2) -1 -2*p2**2)
        
    k = -d12/(c11*j22-d12*i21)
    
    return k


# Import modules
import numpy as np
import matplotlib.pyplot as plt
import importlib
import os
import sys, getopt

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

# Input file
fname = 'out_stokesdarcy2field_rt_analytical'
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

print('# --------------------------------------- #')
print('# RT instability, Interface capturing using the phase field method')
print('# --------------------------------------- #')

#vfopt = np.array([0,1,2,3])
vfopt = np.array([3])
nn = np.array([51, 101, 151, 201])#, 251, 301])
#nn = np.array([51])
#nn = np.array([11, 21, 31, 41, 51])

#edlist = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3])
edlist = np.array([1e-3])

numnn = np.size(nn)
numvf = np.size(vfopt)

grnn_cmp = np.zeros(numnn)
grnn_diff = np.zeros(numnn)
grid = 1/nn

ms = ('ko--', 'ks--', 'kv--', 'kx--')

fig3, axs3 = plt.subplots(figsize=(5,4))
ax3 = axs3

for ivf in range(numvf):


    for inn in range(numnn):
    
        nx = nn[inn]
        nz = nx - 1
        n = nz
        L = 1.0
        H = 1.0
        z_in = H/2.0
        eps = 1.0/n #0.7/n
        gamma = 1.0
        diffuse = vfopt[ivf]
        
        Delta = 0.02 #2.0/n
        
        #the position of nodes to extract vertical velocity
        ix_vp= int((nx-1)/2)
        iz_vp= int(nz/2 + Delta*nz)
        
        #Llist = np.array([1.0, 1.25, 1.5, 1.75, 2.0,  2.5,  3.0])
        Llist = np.array([1.0])
        numL = np.size(Llist)
        gr_cmp = np.zeros(numL)
        gr_ana = np.zeros(numL)
        
        plist = np.zeros(numL)
        
        dt = 1.0/8.0/n #0.5*0.5/n**2/eps
        
        tstep = 1#int(0.1/dt) + 1
        
        tout = 1 #int((tstep-1)/4) #10
        tout2 = 1
        
        etamin = 0.0
        phi0 = 1e-4
        R = 0.0 #1.0
        lam = 0.0
        
        G= 1e40
        Z = 1e40
        
        eta_u = 1e-3
        eta_d = 1e+3
        C_u = 1e40
        C_d = 1e40
        Fu = 1.0/phi0
        Fd = 9e-1/phi0
        
        
        nume = np.size(edlist)
        
        lam_p = 1.0  #YC = (C_b or C_w)*lam_p
        lam_v = 0.1 #zeta_v = eta_v / lam_v
        nh = 1.0
        
        fig2, axs2 = plt.subplots(figsize=(4,4))
        plt.tight_layout()
        
        for ied in range(nume):
            
            eta_d = edlist[ied]
        
            for iwn in range(numL):
                
                L = Llist[iwn]
                    
                wn = 2
                plist[iwn] = np.pi/(L/wn)
                
                # solver parameters
                phase = ' -eps '+str(eps)+' -gamma '+str(gamma) + ' -vfopt '+str(diffuse)
                model = ' -L ' + str(L) + ' -H ' +str(H)+ ' -dt '+str(dt)+' -z_in '+str(z_in) +\
                    ' -eta_u '+str(eta_u) + ' -eta_d '+str(eta_d) + ' -C_d '+str(C_d) + ' -C_u '+str(C_u) +\
                    ' -Fu ' + str(Fu) + ' -Fd ' + str(Fd) + ' -Delta ' + str(Delta) + ' -wn ' + str(2)
                
                newton = ' -pc_factor_mat_solver_type umfpack'+ \
                  ' -pc_type lu'+ \
                  ' -snes_linesearch_type bt'+ \
                  ' -snes_linesearch_monitor'+ \
                  ' -snes_atol 1e-10'+ \
                  ' -snes_rtol 1e-50'+ \
                  ' -snes_stol 1e-10'+ \
                  ' -snes_max_it 200'+ \
                  ' -snes_monitor'+ \
                  ' -snes_view'+ \
                  ' -snes_monitor_true_residual'+ \
                  ' -ksp_monitor_true_residual'+ \
                  ' -ksp_gmres_restart 300' + \
                  ' -snes_converged_reason'+ \
                  ' -ksp_converged_reason'+ \
                  ' -python_snes_failed_report'+ \
                  ' -ksp_type fgmres'
                
                solver= ' -snes_mf_operator'
                solver = ''
                
                sdpar = ' -R ' + str(R)+ \
                  ' -phi_0 ' + str(phi0)+ \
                  ' -lambda ' + str(lam)+ \
                  ' -G ' + str(G) +\
                  ' -Z ' + str(Z) +\
                  ' -lam_p '+str(lam_p)+ \
                  ' -lam_v '+str(lam_v)+ \
                  ' -etamin '+str(etamin)+ \
                  ' -nh ' +str(nh)
                
                # Run test
                # Forward euler
                str1 = 'mpiexec -n '+str(ncpu)+' ../test_stokesdarcy2field_rt' + \
                       ' -nx '+str(nx)+' -nz '+str(nz)+' -tstep '+str(tstep) + \
                       newton + model + phase + solver + sdpar + \
                       ' -output_dir '+fname+' -output_file '+fname+' -tout '+str(tout)+' -ts_scheme 0'
                print(str1)
                os.system(str1)
                
                # Prepare data for time series
                # Load data - initial
                f1out = fname+'_phase_initial'
                spec = importlib.util.spec_from_file_location(f1out,fname+'/'+f1out+'.py')
                imod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(imod)
                data_ini = imod._PETScBinaryLoad()
                imod._PETScBinaryLoadReportNames(data_ini)
                # imod = importlib.import_module(fname+'_phase_initial')
                # data_ini = imod._PETScBinaryLoad()
                # imod._PETScBinaryLoadReportNames(data_ini)
                
                # Get general data (elements, grid)
                m = data_ini['Nx'][0]
                n = data_ini['Ny'][0]
                xc = data_ini['x1d_cell']
                zc = data_ini['y1d_cell']
                
                f_ini = data_ini['X_cell']
                fres_ini = f_ini.reshape(n,m)
                fx_ini = fres_ini[:,int(m/2)]
                
                
                # Compute the analytical solution for gr_rel = vy/Delta = -K*h2*(Fu-Fd)/(2*eta_d)
                grk = getk_analytical(eta_u, eta_d, L, wn)
                gr_rel = grk*0.5*(Fu-Fd)*phi0/(2*eta_d)
                
                # Plot solution 
                fig1, axs1 = plt.subplots(1, 1,figsize=(4,4))
                
                nind = 5
                
                
                for istep in range(0,tstep,tout2):
                  # Load python module describing data
                  if (istep < 10): ft = '_ts00'+str(istep)
                  if (istep >= 10) & (istep < 99): ft = '_ts0'+str(istep)
                  if (istep >= 100): ft = '_ts'+str(istep)
                
                  # Load data - phase
                  f1out = fname+'_phase'+ft
                  spec = importlib.util.spec_from_file_location(f1out,fname+'/'+f1out+'.py')
                  imod = importlib.util.module_from_spec(spec)
                  spec.loader.exec_module(imod)
                  data0 = imod._PETScBinaryLoad()
                  # imod = importlib.import_module(fname+'_phase'+ft)
                  # data0 = imod._PETScBinaryLoad()
                  #imod._PETScBinaryLoadReportNames(data0)
                
                  # load data - PV  
                  #fout = fname+'_solution'+ft
                  fout = fname+'_solution_initial'
                  spec = importlib.util.spec_from_file_location(fout,fname+'/'+fout+'.py')
                  imod = importlib.util.module_from_spec(spec)
                  spec.loader.exec_module(imod)
                  data1 = imod._PETScBinaryLoad()
                  # imod = importlib.import_module(fout) 
                  # data1 = imod._PETScBinaryLoad()
                  #imod._PETScBinaryLoadReportNames(data)
                
                  mx = data1['Nx'][0]
                  mz = data1['Ny'][0]
                  xc = data1['x1d_cell']
                  zc = data1['y1d_cell']
                  xv = data1['x1d_vertex']
                  zv = data1['y1d_vertex']
                  vx = data1['X_face_x']
                  vz = data1['X_face_y']
                  p  = data1['X_cell']
                  
                  # Prepare cell center velocities and pressure
                  vxface = vx.reshape(mz  ,mx+1)
                  vzface = vz.reshape(mz+1,mx  )
                  pp = p.reshape(mz,mx)
                  
                  print('Computation:',np.max(vzface)/Delta,np.min(vzface)/Delta, 'Analytial:', grk, gr_rel )
                
                  # Compute the cell center values from the face data by averaging neighbouring faces
                  vxc = np.zeros( [mz , mx] )
                  vzc = np.zeros( [mz , mx] )
                  vc  = np.zeros( [mz , mx] )
                  
                  for i in range(0,mx):
                    for j in range(0,mz):
                      vxc[j][i] = 0.5 * (vxface[j][i+1] + vxface[j][i])
                      vzc[j][i] = 0.5 * (vzface[j+1][i] + vzface[j][i])
                      vc [j][i] = (vxc[j][i]**2+vzc[j][i]**2)**0.5
                
                
                  print('Maximum velocity magnitude = ', np.max(vc))
                
                  # Get individual data sets - phase
                  f0 = data0['X_cell']
                  fres0 = f0.reshape(n,m)
                  fx0 = fres0[:,int(m/2)]
                  
                  # initial shape
                  mm = 200
                  xini = np.linspace(L*0.5/n, L*(1 - 0.5/n), mm)
                  zini = z_in + Delta*np.cos(2*np.pi*2.0*xini/L)
                
                  if (istep == tstep-1):
                #  if (istep == 0):
                #  if (istep > 0):      
                      
                    tt = dt*istep
                        
                    cmaps='RdBu_r'
                    # color map
                    ax0 = axs1
                    im = ax0.imshow(fres0,extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps,origin='lower')
                 #   im = ax0.imshow(pp,extent=[min(xc), max(xc), min(zc), max(zc)],cmap=cmaps,origin='lower')
                    Q  = ax0.quiver( xc[::nind], zc[::nind], vxc[::nind,::nind], vzc[::nind,::nind], units='width', pivot='mid' )
                    cs = ax0.contour( xc , zc , fres0, levels=[0.5] , colors='black',linestyles='solid',linewidths=1.0)
                    #ct01= ax0.contour( xc , zc , fres_ini, levels=[0.5] , colors='gray',linestyles='dashed',linewidths=1.0)
                    #ax0.plot(xini, zini, 'k--')
                    ax0.set_title('Rayleigh-Taylor instability')
                    ax0.set_xlabel('x')
                    ax0.set_ylabel('z')
                    #cbar = fig1.colorbar(im,ax=ax0, shrink=0.75)
                    ax0.grid(True,color='gray', linestyle=':', linewidth=0.5)        
                    
                    p1 = cs.collections[0].get_paths()[0]  # grab the 1st path
                    coor_p1 = p1.vertices
                    print(np.max(coor_p1[:,1]), np.min(coor_p1[:,1]))
                    dd = (np.max(coor_p1[:,1])- np.min(coor_p1[:,1]))/2
                    print('Computation:',np.max((-1)**wn*vzface)/dd, 'Analytial:', grk, gr_rel )
                    print('Use vertical velocity at the picked point', vzface[iz_vp][ix_vp]/dd, 'Computed amp',dd)
                    
                    #gr_cmp[iwn] = np.max((-1)**wn*vzface)/dd
                    gr_cmp[iwn] = vzface[iz_vp][ix_vp]/dd
                    gr_ana[iwn] = gr_rel
                    
                    grnn_cmp[inn] = gr_cmp[iwn]
                    grnn_diff[inn] = np.abs(gr_ana[iwn] - grnn_cmp[inn])/gr_ana[iwn]
                
                fig1.savefig(fname+'/'+fname+'_rk1.pdf')
            
            
        
            ax1 = axs2
            l1 = ax1.plot(plist, gr_ana, 'k--')
            l2 = ax1.plot(plist, gr_cmp, 'ko')
            
        ax1.set_title('Growth rate')
        ax1.set_xlabel(r'$\phi_1$')
        ax1.set_ylabel(r'$v_y/\Delta$')
        ax1.set_yscale('log')
        plt.tight_layout()
        fig2.savefig(fname+'/'+'rt_growthrate.pdf')
        plt.close(fig2)
        
    
    ax3.plot(grid, grnn_diff, ms[ivf])
ax3.set_title('Convergence of growth rate')
ax3.set_xlabel(r'$\Delta x$')
ax3.set_ylabel(r'$|\tau_{ana}- \tau_{cmp}|/\tau_{ana}$')
#ax1.set_yscale('log')
plt.tight_layout()
fig3.savefig(fname+'/'+'rt_convergence.pdf')

os.system('rm -r '+fname+'/__pycache__')

for istep in range(0,tstep,tout):
  # Load python module describing data
  if (istep < 10): ft = '_ts00'+str(istep)
  if (istep >= 10) & (istep < 99): ft = '_ts0'+str(istep)
  if (istep >= 100): ft = '_ts'+str(istep)
#  plot_allinone(fname,ft,n)

