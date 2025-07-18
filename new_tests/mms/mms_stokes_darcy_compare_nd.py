
def write_c_method(var,varname):
  code  = 'static PetscScalar get_'+ varname + \
    '(PetscScalar x, PetscScalar z, PetscScalar alpha, PetscScalar R, PetscScalar phi_0, PetscScalar phi_s, '+ \
    'PetscScalar p_s,PetscScalar psi_s, PetscScalar U_s, PetscScalar m, PetscScalar n, PetscScalar e3)\n'
  code += '{ PetscScalar result;\n'
  code += '  result = ' + ccode(var) + ';\n'
  code += '  return(result);\n'
  code += '}'  
  print(code)

# ------------------------------------------------ #
# MMS test to verify 2 non-dimensionalization schemes (MMS1-Rhebergen et al. 2014, MMS2-Katz-Magma dynamics)
# MMS stands for Method of Manufactured Solutions 
# ------------------------------------------------ #
from sympy import *

# Symbols
x = Symbol('x')
z = Symbol('z')

alpha = Symbol('alpha')
R     = Symbol('R')
phi_0 = Symbol('phi_0')
phi_s = Symbol('phi_s')
p_s   = Symbol('p_s')
psi_s = Symbol('psi_s')
U_s   = Symbol('U_s')
n     = Symbol('n')
m     = Symbol('m')
e3    = Symbol('e3') # unit vector of gravity

# Chosen solutions and coefficients
R_alpha = R**2/(alpha+1)

# porosity
phi = phi_0*(1.0+phi_s*cos(m*pi*x)*cos(m*pi*z))

# pressure
p = p_s*cos(m*pi*x)*cos(m*pi*z)
psi = psi_s*(1.0-cos(m*pi*x))*(1.0-cos(m*pi*z))
U   = -U_s*cos(m*pi*x)*cos(m*pi*z)

# calculate mms velocities from potentials
curl_psix = diff(psi,z)
curl_psiz = -diff(psi,x)
gradUx    = diff(U,x)
gradUz    = diff(U,z)

ux = curl_psix + gradUx
uz = curl_psiz + gradUz

# mobility
Kphi = (phi/phi_0)**n

# Compute right-hand-side
dpdx = diff(p,x)
dpdz = diff(p,z)

dvxdx = diff(ux,x)
dvxdz = diff(ux,z)
dvzdz = diff(uz,z)
dvzdx = diff(uz,x)

dvx2dx = diff(dvxdx,x)
dvx2dz = diff(dvxdz,z)
dvz2dxdz = diff(dvzdx,z)

dvz2dx = diff(dvzdx,x)
dvz2dz = diff(dvzdz,z)
dvx2dzdx = diff(dvxdz,x)

divu  = diff(ux,x) + diff(uz,z)

# MMS1
darcyx_mms1 = -R_alpha*Kphi*(diff(p,x) - 0.0)
darcyz_mms1 = -R_alpha*Kphi*(diff(p,z) - e3 )
div_darcy_mms1 = diff(darcyx_mms1,x) + diff(darcyz_mms1,z)

fux_mms1 = -diff(p,x) + 0.5*(2.0*dvx2dx + dvx2dz + dvz2dxdz) + alpha*diff(divu,x)
fuz_mms1 = -diff(p,z) + 0.5*(2.0*dvz2dz + dvz2dx + dvx2dzdx) + alpha*diff(divu,z) + phi*e3
fp_mms1  = divu + div_darcy_mms1

# MMS2
darcyx_mms2 = -Kphi*(diff(p,x) - 0.0)
darcyz_mms2 = -Kphi*(diff(p,z) - e3 )
div_darcy_mms2 = diff(darcyx_mms2,x) + diff(darcyz_mms2,z)

fux_mms2 = -diff(p,x) + 0.5*R_alpha*(2.0*dvx2dx + dvx2dz + dvz2dxdz) + alpha*R_alpha*diff(divu,x)
fuz_mms2 = -diff(p,z) + 0.5*R_alpha*(2.0*dvz2dz + dvz2dx + dvx2dzdx) + alpha*R_alpha*diff(divu,z) + phi*e3
fp_mms2  = divu + div_darcy_mms2

print('MMS solutions:')
print('\n')
write_c_method(p,'p')
write_c_method(ux,'ux')
write_c_method(uz,'uz')

write_c_method(fux_mms1,'fux_mms1')
write_c_method(fuz_mms1,'fuz_mms1')
write_c_method(fp_mms1,'fp_mms1')

write_c_method(fux_mms2,'fux_mms2')
write_c_method(fuz_mms2,'fuz_mms2')
write_c_method(fp_mms2,'fp_mms2')

write_c_method(Kphi,'Kphi')
write_c_method(phi,'phi')
write_c_method(psi,'psi')
write_c_method(U,'U')

write_c_method(gradUx,'gradUx')
write_c_method(gradUz,'gradUz')
write_c_method(curl_psix,'curl_psix')
write_c_method(curl_psiz,'curl_psiz')

print('\n')