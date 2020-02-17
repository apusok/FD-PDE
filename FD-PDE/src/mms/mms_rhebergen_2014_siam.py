
def write_c_method(var,varname):
  code  = 'static PetscScalar get_'+ varname + '(PetscScalar x, PetscScalar z, PetscScalar k_ls, PetscScalar k_us, PetscScalar alpha, PetscScalar A)\n'
  code += '{ PetscScalar result;\n'
  code += '  result = ' + ccode(var) + ';\n'
  code += '  return(result);\n'
  code += '}'  
  print(code)

# ------------------------------------------------ #
# MMS for Rhebergen et al. 2014, SIAM - Ex. 6.1
# MMS stands for Method of Manufactured Solutions 
# ------------------------------------------------ #
from sympy import *

x = Symbol('x')
z = Symbol('z')

# Chosen coefficients
k_ls = Symbol('k_ls')
k_us = Symbol('k_us')
alpha = Symbol('alpha')
A = Symbol('A')

k = 0.25*(k_us-k_ls)/tanh(5.0)*( 2.0 + tanh(10.0*x-5.0) + tanh(10.0*z-5.0) + (2.0*(k_us-k_ls)-2.0*tanh(5.0)*(k_ls+k_us))/(k_ls-k_us) )
# k = 1.0

# Chosen solution
p = -cos(4.0*pi*x)*cos(2.0*pi*z)

dpdx = diff(p,x)
dpdz = diff(p,z)

ux = k*dpdx + sin(pi*x)*sin(2.0*pi*z) + 2.0
uz = k*dpdz + 0.5*cos(pi*x)*cos(2.0*pi*z) + 2.0

# Compute rhs
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

darcyx = -k*diff(p,x)
darcyz = -k*diff(p,z)
div_darcy = diff(darcyx,x) + diff(darcyz,z)

fux = -diff(p,x) + A*(2.0*dvx2dx + dvx2dz + dvz2dxdz) + alpha*diff(divu,x)
fuz = -diff(p,z) + A*(2.0*dvz2dz + dvz2dx + dvx2dzdx) + alpha*diff(divu,z)
fp  = divu + div_darcy

# print('MMS chosen solution:')
# print('  [ccode] p =', ccode(p) + '; \n')
# print('  [ccode] ux =', ccode(ux) + '; \n')
# print('  [ccode] uz =', ccode(uz) + '; \n')

# print('MMS chosen coefficients:')
# print('  [ccode] k =', ccode(k) + '; \n')

# print('MMS right-hand-side:')
# print('  [ccode] fux =', ccode(fux) + '; \n')
# print('  [ccode] fuz =', ccode(fuz) + '; \n')
# print('  [ccode] fp =', ccode(fp) + '; \n')

print('MMS solution:')
print('\n')
write_c_method(k,'k')
write_c_method(p,'p')
write_c_method(ux,'ux')
write_c_method(uz,'uz')

write_c_method(fux,'fux')
write_c_method(fuz,'fuz')
write_c_method(fp,'fp')
print('\n')