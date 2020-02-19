
def write_c_method(var,varname):
  code  = 'static PetscScalar get_'+ varname + \
    '(PetscScalar x, PetscScalar z, PetscScalar delta, PetscScalar phi_0, PetscScalar phi_s, '+ \
    'PetscScalar p_s,PetscScalar psi_s, PetscScalar U_s, PetscScalar m, PetscScalar n, PetscScalar e3)\n'
  code += '{ PetscScalar result;\n'
  code += '  result = ' + ccode(var) + ';\n'
  code += '  return(result);\n'
  code += '}'  
  print(code)

# ------------------------------------------------ #
# MMS for 2D two-phase flow example in Ch 13, Katz 2019 - Magma dynamics
# MMS stands for Method of Manufactured Solutions 
# ------------------------------------------------ #
from sympy import *

# Symbols
x = Symbol('x')
z = Symbol('z')

delta = Symbol('delta')
phi_0 = Symbol('phi_0')
phi_s = Symbol('phi_s')
p_s   = Symbol('p_s')
psi_s = Symbol('psi_s')
U_s   = Symbol('U_s')
n     = Symbol('n')
m     = Symbol('m')
e3    = Symbol('e3') # unit vector of gravity

# Chosen solutions and coefficients
# porosity
phi = phi_0*(1.0+phi_s*cos(m*pi*x)*cos(m*pi*z))

# pressure
p = p_s*cos(m*pi*x)*cos(m*pi*z)

# potential functions (original)
# psi = -psi_s*(1.0-cos(m*pi*x))*(1.0-cos(m*pi*z))
# U   = U_s*cos(m*pi*x)*cos(m*pi*z)

psi = psi_s*(1.0-cos(m*pi*x))*(1.0-cos(m*pi*z))
U   = -U_s*cos(m*pi*x)*cos(m*pi*z)

# calculate mms velocities from potentials
curl_psix = diff(psi,z)
curl_psiz = diff(psi,x)
gradUx    = diff(U,x)
gradUz    = diff(U,z)

ux = curl_psix + gradUx
uz = curl_psiz + gradUz

# mobility
Kphi = (phi/phi_0)**n

# Compute rhs
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

darcyx = -Kphi*(diff(p,x) + 0.0)
darcyz = -Kphi*(diff(p,z) + e3 )
div_darcy = diff(darcyx,x) + diff(darcyz,z)

fux = -diff(p,x) + delta**2*(2.0*dvx2dx + dvx2dz + dvz2dxdz) + delta**2*diff(divu,x)
fuz = -diff(p,z) + delta**2*(2.0*dvz2dz + dvz2dx + dvx2dzdx) + delta**2*diff(divu,z) - phi*e3
fp  = divu + div_darcy

# RHS using the potential functions (eq. 13.37)
del2U   = diff(diff(U,x),x) + diff(diff(U,z),z)
del2psi = diff(diff(psi,x),x) + diff(diff(psi,z),z)
curl_del2psi_x = diff(del2psi,z)
curl_del2psi_z = diff(del2psi,x)

fp_potential  = del2U + div_darcy
fux_potential = -diff(p,x) + delta**2*curl_del2psi_x + 3.0*delta**2*diff(del2U,x)
fuz_potential = -diff(p,z) + delta**2*curl_del2psi_z + 3.0*delta**2*diff(del2U,z) - phi*e3

print('MMS solution:')
print('\n')
write_c_method(p,'p')
write_c_method(ux,'ux')
write_c_method(uz,'uz')

write_c_method(fux,'fux')
write_c_method(fuz,'fuz')
write_c_method(fp,'fp')

write_c_method(fux,'fux_potential')
write_c_method(fuz,'fuz_potential')
write_c_method(fp,'fp_potential')

write_c_method(Kphi,'Kphi')
write_c_method(phi,'phi')
write_c_method(psi,'psi')
write_c_method(U,'U')

write_c_method(gradUx,'gradUx')
write_c_method(gradUz,'gradUz')
write_c_method(curl_psix,'curl_psix')
write_c_method(curl_psiz,'curl_psiz')

print('\n')