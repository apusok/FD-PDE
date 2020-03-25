
def write_c_method(var,varname):
  code  = 'static PetscScalar get_'+ varname + \
    '(PetscScalar x, PetscScalar z, PetscScalar t)\n'
  code += '{ PetscScalar result;\n'
  code += '  result = ' + ccode(var) + ';\n'
  code += '  return(result);\n'
  code += '}'  
  print(code)

# ------------------------------------------------ #
# MMS test to verify the ADVDIFF pde
# MMS stands for Method of Manufactured Solutions 
# When choosing MMS functions:
#   A, B > 0 (positive)
#   if ux, uz - constant, then upwind is 1st order, upwind2 is 2nd order
#   fromm advection scheme is always 2nd order
# ------------------------------------------------ #
from sympy import *

# Symbols
x = Symbol('x')
z = Symbol('z')
t = Symbol('t')
# ------------------------------------------------ #
# 1-diffusion steady-state (space)
ux1 = 0.0
uz1 = 0.0
Q1 = cos(2.0*pi*x)*sin(2.0*pi*z)
A1 = 0.0
B1 = 1.5 + sin(2.0*pi*x)*cos(2.0*pi*z)
frhs1 = -diff(B1*diff(Q1,x),x)-diff(B1*diff(Q1,z),z)

# Q1 = x**2*cos(4.0*pi*z)
# B1 = x**2+z**2
# ------------------------------------------------ #
# 2-advection-diffusion steady-state (space+advection scheme)
# ux2 = 1.0 # gives 1st order upwind, 2nd order upwind2
# uz2 = 1.0
# Q2 = cos(2.0*pi*x)*sin(2.0*pi*z)
# A2 = 1.0
# B2 = 1.0
# frhs2 = A2*(diff(Q2*ux2,x) + diff(Q2*uz2,z)) - (diff(B2*diff(Q2,x),x)+diff(B2*diff(Q2,z),z))

ux2 = 1.0 + x 
uz2 = x**2*sin(2*pi*z)
Q2 = cos(2.0*pi*x)*sin(2.0*pi*z)
A2 = 1.5+sin(2.0*pi*x)*cos(2.0*pi*z)
B2 = 1.0+x**2+z**2
frhs2 = A2*(diff(Q2*ux2,x) + diff(Q2*uz2,z)) - (diff(B2*diff(Q2,x),x)+diff(B2*diff(Q2,z),z))
# ------------------------------------------------ #
# 3-time-dependent diffusion (time-stepping scheme)
ux3 = 0.0
uz3 = 0.0
Q3 = t**3*(x**2+z**2)
A3 = 1.0
B3 = 1.0
frhs3 = A3*(diff(Q3,t)) - (diff(B3*diff(Q3,x),x)+diff(B3*diff(Q3,z),z))

# Q3 = exp(-t)*sin(pi*x)*sin(pi*z)
# ------------------------------------------------ #
# 4-time-dependent advection (pure-advection)
ux4 = 1.0 + x 
uz4 = x**2*sin(2*pi*z)
Q4 = t**3*(x**2+z**2)
A4 = 1.0+sin(2.0*pi*x)*cos(2.0*pi*z)
B4 = 0.0
frhs4 = A4*(diff(Q4,t) + diff(Q4*ux4,x) + diff(Q4*uz4,z))

# Q4 = t**3*sin(pi*x)*sin(pi*z)
# ------------------------------------------------ #

print('MMS solutions:')
print('\n')
write_c_method(Q1,'Q1')
write_c_method(A1,'A1')
write_c_method(B1,'B1')
write_c_method(ux1,'ux1')
write_c_method(uz1,'uz1')
write_c_method(frhs1,'frhs1')

write_c_method(Q2,'Q2')
write_c_method(A2,'A2')
write_c_method(B2,'B2')
write_c_method(ux2,'ux2')
write_c_method(uz2,'uz2')
write_c_method(frhs2,'frhs2')

write_c_method(Q3,'Q3')
write_c_method(A3,'A3')
write_c_method(B3,'B3')
write_c_method(ux3,'ux3')
write_c_method(uz3,'uz3')
write_c_method(frhs3,'frhs3')

write_c_method(Q4,'Q4')
write_c_method(A4,'A4')
write_c_method(B4,'B4')
write_c_method(ux4,'ux4')
write_c_method(uz4,'uz4')
write_c_method(frhs4,'frhs4')

print('\n')