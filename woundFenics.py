# -*- coding: utf-8 -*-
"""
Fenics Wound Code

Created on Mon Sep  7 11:58:39 2020

@author: David
"""

from __future__ import print_function
from fenics import *
import time
import math
from mshr import *

#####################
# DEFINE VARIABLES
#####################

T = 336.0              # final time
num_steps = 336      # number of time steps
dt = T / num_steps   # time step size
D_rho = 0.01         # diffusion coefficient
d_rho = 10.0         # reaction rate
D_c = 0.01
d_c = 1
height = 4
inner_radius = 7.5
outer_radius = 37.5
rho_healthy = 1000
rho_wound = 0
c_healthy = 0
c_wound = 1

#####################
# SET UP GEOMETRY
#####################

# Making a cylindrical geometry
cylinder = Cylinder(Point(0, 0, height), Point(0, 0, 0), outer_radius, outer_radius)
geometry = cylinder

# Generate the mesh
mesh = generate_mesh(geometry, 40)
File("wound_cylinder_mesh.pvd") << mesh

#####################
# SET UP ELEMENTS
#####################

# Define elements
U1 = VectorElement('P', tetrahedron, 3)
P1 = FiniteElement('P', tetrahedron, 1)
element = MixedElement([U1, P1, P1])
# Define function space
V = FunctionSpace(mesh, element)

# Define variational problem
Xi = Function(V) # This is the solution function (nonlinear)
u, rho, c = split(Xi)
N = TestFunction(V) # This is the test function
N_1, N_2, N_3 = split(N)

#####################
# BOUNDARY AND INITIAL CONDITIONS
#####################

# Define boundary condition
def outer_boundary(x, on_boundary):
    r = math.sqrt(x[0]*x[0] + x[1]*x[1])
    tol=0.2
    return on_boundary and r>outer_radius-tol

bc1 = DirichletBC(V.sub(0).sub(0), Constant(0.), outer_boundary)
bc2 = DirichletBC(V.sub(0).sub(1), Constant(0.), outer_boundary)
bc3 = DirichletBC(V.sub(0).sub(2), Constant(0.), outer_boundary)
bc4 = DirichletBC(V.sub(1), Constant(rho_healthy), outer_boundary)
bc5 = DirichletBC(V.sub(2), Constant(c_healthy), outer_boundary)

# Define initial values for displacement, cells, and cytokine
ic = Expression(((0,0,0),'(pow(x[0],2) + pow(x[1],2) > pow(inner_radius,2)) ? rho_healthy : rho_wound','(pow(x[0],2) + pow(x[1],2) > pow(inner_radius,2)) ? c_healthy : c_wound'),
                degree=1,inner_radius=inner_radius, rho_healthy=rho_healthy, rho_wound=rho_wound, c_healthy=c_healthy, c_wound=c_wound)
all_n = interpolate(ic, V)
u_n, rho_n, c_n = split(all_n)

#####################
# KINEMATICS
#####################

# Kinematics
d = u.geometric_dimension()
I = Identity(d)             # Identity tensor
FF = I + grad(u)             # Deformation gradient
C = FF.T*FF                   # Right Cauchy-Green tensor

# Invariants of deformation tensors
Ic = tr(CC)
J  = det(FF)

# Elasticity parameters
E, nu = 10.0, 0.3
mu, lmbda = Constant(E/(2*(1 + nu))), Constant(E*nu/((1 + nu)*(1 - 2*nu)))

# Stored strain energy density (compressible neo-Hookean model)
psi = (mu/2)*(Ic - 3) - mu*ln(J) + (lmbda/2)*(ln(J))**2

# Compute stress (first variation of Pi (directional derivative about u in the direction of v))
#F = derivative(Pi, u, v)
# Define the material
def SSigma(u, gamma):
    I = Identity(u.cell().d)      # Identity tensor
    F = I + grad(u)               # Deformation gradient

    # F_a = I - gamma*outer(f0, f0) # Active strain
    F_e = F*(I + (gamma/(1 - gamma))*outer(f0, f0))

    C_e = F_e.T*F_e               # Right Cauchy-Green tensor
    B_e = F_e*F_e.T               # Left Cauchy-Green tensor

    # Principle isotropic invariants
    I1 = tr(B_e)
    I2 = 0.5*(tr(B_e)**2 - tr(B_e*B_e))
    I3 = det(B_e)

    # Anisotropic (quasi) invariants
    I4_f = inner(f0, C_e*f0)
    I4_s = inner(s0, C_e*s0)
    I8_fs = inner(f0, C_e*s0)

    # Current fibre, sheet and sheet-normal directions
    f = F*f0
    s = F*s0

    # Cauchy stress
    return(  a*exp(b*(I1 - 3))*B_e - p*I \
           + 2*a_f*(I4_f - 1)*exp(b_f*(I4_f - 1)**2)*outer(f, f) \
           + 2*a_s*(I4_s - 1)*exp(b_s*(I4_s - 1)**2)*outer(s, s) \
           + a_fs*I8_fs*exp(b_fs*I8_fs**2)*(outer(f, s) + outer(s, f)))

# Second Piola Kirchoff Stress
def SS(u, gamma):
    I = Identity(u.cell().d)
    FF = I + grad(u)
    return(det(F)*inv(F)*sigma(u, gamma)*inv(F).T)

# First Piola Kirchoff Stress
def PP(u, gamma):
    I = Identity(u.cell().d)
    FF = I + grad(u)
    return(det(F)*sigma(u, gamma)*inv(F).T)

# Active stress
def SS_act():
    
    return 


# Total stress
SS_total = SS_vol() + SS_pas() + SS_act()
PP_total = FF*SS_total

# Mechanics equations
F_u = inner(PP_total, grad(N1))*dx

#####################
# TRANSPORT
#####################

# Define expressions used in variational forms
D_rho = Constant(D_rho)
d_rho = Constant(d_rho)
D_c = Constant(D_c)
d_c = Constant(d_c)

# Define source terms
S_rho = Expression('0', degree=1)
S_c = Expression('0', degree=1)

# Diffusion equation
F_rho = rho*N_1*dx + dt*dot(grad(rho), grad(N_1))*dx - (rho_n + dt*S_rho)*N_1*dx
F_c = c*N_2*dx + dt*dot(grad(c), grad(N_2))*dx - (c_n + dt*S_c)*N_2*dx

#####################
# ASSEMBLE
#####################

# Assemble the entire problem
F = F_u + F_rho + F_c
# If the problem were linear we would do:
#a, L = lhs(F), rhs(F)

# Create VTK file for saving solution
vtkfile_u = File('woundFenics/solution_u.pvd')
vtkfile_rho = File('woundFenics/solution_rho.pvd')
vtkfile_c = File('woundFenics/solution_c.pvd')

#####################
# SOLVE
#####################

# Time-stepping
# If it were linear we wound need:
#rho = Function(V)
t = 0
for n in range(num_steps):

    # Update current time
    t += dt

    # Compute solution
    # If it were linear we would have a == L
    solve(F == 0, density, bc)

    # Save to file and plot solution
    output_rho, output_c = density.split()
    vtkfile_rho << (output_rho, t)
    vtkfile_c << (output_c, t)
    #plot(rho)

    # Update previous solution
    density_n = density
    #rho_n.assign(rho)
    #c_n.assign(c)

    # Update progress bar
    #progress.update(t / T)

# Hold plot
#interactive()
