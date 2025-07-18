
def write_c_method(var,varname):
  code  = 'static PetscScalar get_'+ varname + '(PetscScalar x, PetscScalar z)\n'
  code += '{ PetscScalar result;\n'
  code += '  result = ' + ccode(var) + ';\n'
  code += '  return(result);\n'
  code += '}'  
  print(code)

# ------------------------------------------------ #
# MMS sympy for 2D diffusion: div(k*grad(T)) = f
# ------------------------------------------------ #
from sympy import *

x = Symbol('x')
z = Symbol('z')
k = Symbol('k')

k0 = 1.5

# Chosen coefficient
k = k0 + sin(2.0*pi*x)*cos(2.0*pi*z)
# k = 1.0

# Chosen solution
T = cos(2.0*pi*x)*sin(2.0*pi*z)

dTdx = diff(T,x)
dTdz = diff(T,z)

# Compute rhs
f = diff(k*dTdx,x)+diff(k*dTdz,z)

print('MMS chosen solutions:\n')
write_c_method(k,'k')
write_c_method(T,'T')
write_c_method(f,'f')
print('\n')