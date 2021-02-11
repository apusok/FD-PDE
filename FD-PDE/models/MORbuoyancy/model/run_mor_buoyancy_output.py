print('# --------------------------------------- #')
print('# Mid-ocean ridge model - Buoyancy ')
print('# --------------------------------------- #')

# Import modules
import os
import sys

# Add path to vizMORBuoyancy
pathViz = '../python/'
sys.path.append(pathViz)
import vizMORBuoyancy as vizB

# ---------------------------------------
# Main
# ---------------------------------------

# Parameters
fname = 'out_model'
tstep = 21
SEC_YEAR = 31536000
buoyancy = 0 # 0=off, 1=phi, 2=phi-C, 3=phi-C-T
k_hat = 1.0

# Run test
str1 = '../MORbuoyancy.app'+ \
    ' -options_file model_test.opts -nx 200 -nz 100'+ \
    ' -tstep '+str(tstep)+ \
    ' -buoyancy '+str(buoyancy)+ \
    ' -k_hat '+str(k_hat)+ \
    ' -log_view '
print(str1)
os.system(str1)

# Read parameters file and get scaling params

# Units and labels
lbl_P = r'$P$ [-]'
lbl_vs= r'$V_s$ [-]'
lbl_vf= r'$V_f$ [-]'
lbl_v = r'$V$ [-]'
lbl_x = r'[-]'
lbl_C = r'$\Theta$ [-]'
lbl_H = r'$H$ [-]'
lbl_phi = r'$\phi$ [-]'
lbl_T = r'$\tilde{\theta}$ [-]'
lbl_TP = r'$\theta$ [-]'
lbl_Cf = r'$\Theta_f$ [-]'
lbl_Cs = r'$\Theta_s$ [-]'
lbl_Plith = r'$P_{lith}$ [-]'
lbl_resP = r'res $P$ [-]'
lbl_resvs= r'res $V_s$ [-]'
lbl_resC = r'res $\Theta$ [-]'
lbl_resH = r'res $H$ [-]'
lbl_eta = r'$\eta$ [-]'
lbl_zeta = r'$\zeta$ [-]'
lbl_K = r'$K$ [-]'
lbl_rho = r'$\rho$ [-]'

scalx  = 1
scalP  = 1
scalv  = 1 
scalH  = 1
scalphi= 1
scalC  = 0
scalT  = 0
scaleta= 1
scalK  = 1
scalrho= 1

# if (dim == 1):
#   scalx  = 1e2 # km
#   scalP  = 1e-9 # GPa
#   scalv  = 1.0e2*SEC_YEAR # cm/yr
#   scalT  = 273.15 # deg C
#   lbl_P = r'$P [GPa]$'
#   lbl_v = r'$V [cm/yr]$'
#   lbl_x = '[km]'
#   lbl_C = r'$\Theta$'
#   lbl_H  = r'$H [-]$'
#   lbl_T = r'$T [^oC]$'

# Visualize data
for istep in range(0,tstep+1):
  fdir  = 'Timestep'+str(istep)

  fname = 'out_xPV_ts'+str(istep)
  P,Vsx,Vsz,nx,nz,xc,zc,xv,zv = vizB.parse_PV_file(fname,fdir)
  vizB.plot_PV(P,Vsx,Vsz,nx,nz,xc,zc,xv,zv,scalP,scalv,scalx,lbl_P,lbl_vs,lbl_x,fname,istep,7,12)

  fname = 'out_xHC_ts'+str(istep)
  H,C,nx,nz,xc,zc = vizB.parse_HC_file(fname,fdir)
  vizB.plot_HC(H,C,nx,nz,xc,zc,scalH,scalC,scalx,lbl_H,lbl_C,lbl_x,fname,istep,7,8)

  fname = 'out_xphiT_ts'+str(istep)
  phi,T,nx,nz,xc,zc = vizB.parse_HC_file(fname,fdir)
  vizB.plot_HC(phi,T,nx,nz,xc,zc,scalphi,scalT,scalx,lbl_phi,lbl_T,lbl_x,fname,istep,7,8)

  fname = 'out_xEnth_ts'+str(istep)
  H_e,T_e,TP_e,phi_e,P_e,C_e,Cs_e,Cf_e,nx,nz,xc,zc = vizB.parse_Enth_file(fname,fdir)
  vizB.plot_Enth(H_e,T_e,TP_e,phi_e,P_e,C_e,Cs_e,Cf_e,nx,nz,xc,zc,scalH,scalT,scalC,scalx,lbl_H,lbl_T,lbl_TP,lbl_Plith,lbl_C,lbl_Cf,lbl_Cs,lbl_phi,lbl_x,fname,istep)

  fname = 'out_xVel_ts'+str(istep)
  Vfx,Vfz,Vbx,Vbz,nx,nz,xc,zc,xv,zv = vizB.parse_Vel_file(fname,fdir)
  vizB.plot_Vel(Vfx,Vfz,Vbx,Vbz,nx,nz,xc,zc,xv,zv,scalv,scalx,lbl_vf,lbl_v,lbl_x,fname,istep)

  fname = 'out_xHCcoeff_ts'+str(istep)
  A1,B1,D1,A2,B2,D2,C1x,C1z,C2x,C2z,vx,vz,vfx,vfz,vsx,vsz = vizB.parse_HCcoeff_file(fname,fdir)
  vizB.plot_HCcoeff(A1,B1,D1,A2,B2,D2,C1x,C1z,C2x,C2z,vx,vz,vfx,vfz,vsx,vsz,nx,nz,xc,zc,xv,zv,scalx,lbl_x,fname,istep)

  if (istep > 0):
    fname = 'out_xPVcoeff_ts'+str(istep)
    A_cor,A,C,D1,Bx,Bz,D2x,D2z,D3x,D3z = vizB.parse_PVcoeff_file(fname,fdir)
    vizB.plot_PVcoeff(A_cor,A,C,D1,Bx,Bz,D2x,D2z,D3x,D3z,nx,nz,xc,zc,xv,zv,scalx,lbl_x,fname,istep)

    fname = 'out_resPV_ts'+str(istep)
    resP,resVsx,resVsz,nx,nz,xc,zc,xv,zv = vizB.parse_PV_file(fname,fdir)
    vizB.plot_PV(resP,resVsx,resVsz,nx,nz,xc,zc,xv,zv,scalP,scalv,scalx,lbl_resP,lbl_resvs,lbl_x,fname,istep,7,12)

    fname = 'out_resHC_ts'+str(istep)
    resH,resC,nx,nz,xc,zc = vizB.parse_HC_file(fname,fdir)
    vizB.plot_HC(resH,resC,nx,nz,xc,zc,scalH,scalC,scalx,lbl_resH,lbl_resC,lbl_x,fname,istep,7,8)

    fname = 'out_matProp_ts'+str(istep)
    eta,zeta,K,rho,rhof,rhos = vizB.parse_matProps_file(fname,fdir)
    vizB.plot_matProp(eta,zeta,K,rho,rhof,rhos,nx,nz,xc,zc,scaleta,scalK,scalrho,scalx,lbl_eta,lbl_zeta,lbl_K,lbl_rho,lbl_x,fname,istep)

  fname = 'out_divmass_ts'+str(istep)
  os.system('rm -r Timestep'+str(istep)+'/__pycache__')

os.system('rm -r '+pathViz+'/__pycache__')
