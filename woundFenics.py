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

parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "uflacs"
ffc_options = {
    "quadrature_degree": 1,
    "eliminate_zeros": True,
    "precompute_basis_const": True,
    "precompute_ip_const": True
    # "optimize": True
}

#####################
# DEFINE VARIABLES
#####################

rho_phys = 1000
rho_healthy = 1000
rho_wound = 0
c_healthy = 0
c_wound = 1e-4
c_max = c_wound

# Mechanics
kv = 0.76; # volumetric penalty for incompressibility
k0 = 3.5e-3; # 0.0511; neo hookean for skin, used previously, in MPa
kf = 5.39; # stiffness of collagen in MPa, from previous paper
k2 = 1.95; # nonlinear exponential coefficient, non-dimensional
t_rho = 5*0.005/1000; # 0.0045/rho_phys force of fibroblasts in MPa, this is per cell. so, in an average sense this is the production by the natural density See Koppenol
t_rho_c = 5*0.05/1000; # 0.045/rho_phys force of myofibroblasts enhanced by chemical, I'm assuming normalized chemical, otherwise I'd have to add a normalizing constant
K_t = 0.5; # Saturation of mechanical force by collagen
K_t_c = c_max/10.; # saturation of chemical on force. this can be calculated from steady state
vartheta_e = 2.; # physiological state of area stretch
gamma_theta = 5.; # sensitivity of heaviside function
b = Constant((0.0, 0.0, 0.0))  # Body force per unit volume

# Transport
D_rhorho = 0.0833; # 0.0833 diffusion of cells in [mm^2/hour], not normalized
D_rhoc = -1.66e-12*10/c_max/c_max; # diffusion of chemotactic gradient, an order of magnitude greater than random walk [mm^2/hour], not normalized
D_cc = 0.01208; # 0.15 diffusion of chemical TGF, not normalized. Updated from Murphy
p_rho = 0.034/2; # in 1/hour production of fibroblasts naturally, proliferation rate, not normalized, based on data of doubling rate from commercial use
p_rho_c = p_rho/2; # production enhanced by the chem, if the chemical is normalized, then suggest two fold,
p_rho_theta = p_rho/2; # enhanced production by theta
K_rho_c = c_max/10.; # saturation of cell proliferation by chemical, this one is definitely not crucial, just has to be small enough <cmax
K_rho_rho = 10000; # saturation of cell by cell, from steady state
d_rho = p_rho*(1-rho_phys/K_rho_rho); # percent of cells die per day, 10% in the original, now much less, determined to keep cells in dermis constant
p_c_rho = 90e-11/rho_phys; # 90.0e-16/rho_phys production of c by cells in g/cells/h
p_c_thetaE = 90e-10/rho_phys; # 300.0e-16/rho_phys coupling of elastic and chemical, three fold
K_c_c = 1.; # saturation of chem by chem, from steady state
d_c = 0.01/10; # 0.01 decay of chemical in 1/hours

# Local solver parameters
p_phi = 0.002/rho_phys; # production by fibroblasts, natural rate in percent/hour, 5% per day
p_phi_c = p_phi; # production up-regulation, weighted by C and rho
p_phi_theta = p_phi; # mechanosensing upregulation. no need to normalize by Hmax since Hmax = 1
K_phi_c = 0.0001; # saturation of C effect on deposition.
d_phi = 0.000970; # rate of degradation, in the order of the wound process, 100 percent in one year for wound, means 0.000116 effective per hour means degradation = 0.002 - 0.000116
d_phi_rho_c = 0.5*0.000970/rho_phys/c_max; #0.000194; // degradation coupled to chemical and cell density to maintain phi equilibrium
K_phi_rho = rho_phys*p_phi/d_phi - 1; # saturation of collagen fraction itself, from steady state
tau_omega = 10./(K_phi_rho+1); # time constant for angular reorientation, think 100 percent in one year
tau_kappa = 1./(K_phi_rho+1); # time constant, on the order of a year
gamma_kappa = 5.; # exponent of the principal stretch ratio
tau_lamdaP_a = 0.001/(K_phi_rho+1); # 1.0 time constant for direction a, on the order of a year
tau_lamdaP_s = 0.001/(K_phi_rho+1); # 1.0 time constant for direction s, on the order of a year
tau_lamdaP_n = 1.0/(K_phi_rho+1); # 1.0 time constant for direction s, on the order of a year
tol_local = 1e-8; # local tolerance (also try 1e-5)
time_step_ratio = 100; # time step ratio between local and global (explicit)
max_iter = 100; # max local iter (implicit)

# Run settings
T = 336.0              # final time
num_steps = 3360      # number of time steps
dt = T / num_steps   # time step size

# Geometry, boundary, initial condiitions
height = 4
inner_radius = 7.5
outer_radius = 37.5

kappa_healthy = Constant(0.23)
kappa_wound = Constant(0.33)
phif_healthy = Constant(1.0)
phif_wound = Constant(0.01)
a0_healthy = Constant(("1.0","0.0","0.0"))
a0_wound = Constant(("1.0","0.0","0.0"))

#####################
# SET UP GEOMETRY
#####################

# Making a cylindrical geometry
cylinder = Cylinder(Point(0, 0, height), Point(0, 0, 0), outer_radius, outer_radius)
geometry = cylinder

# Generate the mesh
mesh = generate_mesh(geometry, 20)
File("cylinder_mesh.pvd") << mesh

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

bcs = [bc1, bc2, bc3, bc4, bc5]

# Define initial values for displacement, cells, and cytokine
ic = Expression(("0.0","0.0","0.0",'(pow(x[0],2) + pow(x[1],2) > pow(inner_radius,2)) ? rho_healthy : rho_wound','(pow(x[0],2) + pow(x[1],2) > pow(inner_radius,2)) ? c_healthy : c_wound'),
                degree=1,inner_radius=inner_radius, rho_healthy=rho_healthy, rho_wound=rho_wound, c_healthy=c_healthy, c_wound=c_wound)
Xi_n = interpolate(ic, V)
u_n, rho_n, c_n = split(Xi_n)

#####################
# KINEMATICS
#####################

# Kinematics
d = u.geometric_dimension()
I = Identity(d)             # Identity tensor
FF = I + grad(u)             # Deformation gradient
CC = FF.T*FF                   # Right Cauchy-Green tensor

# Invariants of deformation tensors
I1 = tr(CC)
I2 = 0.5*(tr(CC)*tr(CC) - tr(CC*CC))
I3 = det(CC)

J  = variable(det(FF))

# Plastic
# For not just define lamdaP as 1
#a0a0 = outer(a0,a0)
#s0s0 = outer(s0,s0)
#n0n0 = outer(n0,n0)
#FFg = lamdaP_a*(a0a0) + lamdaP_s*(s0s0) + lamdaP_N*(n0n0)
#FFginv = (1./lamdaP_a)*(a0a0) + (1./lamdaP_s)*(s0s0) + (1./lamdaP_N)*(n0n0)
phif = phif_healthy
kappa = kappa_healthy
FFg = Identity(d)
FFginv = Identity(d)
Jp = variable(det(FFg))

# Elastic
FFe = FF*FFginv
CCe = FFe.T*FFe
CCeinv = inv(CCe)
Je = variable(det(FFe))

# Collagen fibers
a0 = a0_healthy
A0 = kappa*I + (1-3.*kappa)*outer(a0,a0)
a = FF*a0
A = kappa*FF*FF.T + (1.-3.0*kappa)*outer(a,a)
hat_A = A/tr(A)

# Anisotropic (quasi) invariants
I1e = variable(tr(CCe))
I4e = variable(inner(a0,CCe*a0))
#print(type(I1e))
#print(type(I4e))

# Volumetric stress
#Psivol = 0.5*phif*kv*(Je-1)*(Je-1) - 2*phif*k0*math.log(Je)
dPsivoldJe = phif*kv*(Je-1.) - 2*phif*k0/Je
SSe_vol = dPsivoldJe*Je*CCeinv/2
SS_vol = Jp*FFginv*SSe_vol*FFginv

# Stored strain energy density (compressible neo-Hookean model)
#Psif = Expression('(kf/(2.*k2))*(exp(k2*pow((kappa*I1e + (1-3*kappa)*I4e - 1),2)))', degree=2, kf = kf, k2 = k2, I1e = I1e, I4e = I4e)
Psif = (kf/(2.*k2))*(exp(k2*pow((kappa*I1e + (1-3*kappa)*I4e - 1),2)))
#Psif1 = 2*k2*kappa*(kappa*I1e + (1-3*kappa)*I4e -1)*Psif
#Psif4 = 2*k2*(1-3*kappa)*(kappa*I1e + (1-3*kappa)*I4e -1)*Psif
#SSe_pas = phif*(k0*I) # + Psif1*I + Psif4*outer(a0,a0))
#SS_pas = Jp*FFginv*SSe_pas*FFginv

# Active stress
#traction_act = (t_rho + t_rho_c*c/(K_t_c + c))*rho
#SS_act = (Jp*traction_act*phif/(tr(A)*(K_t*K_t+phif*phif)))*A0

# Total stress
#SS_total = SS_vol + SS_pas + SS_act
#PP_total = FF*SS_total

# Mechanics equations
F_u = inner(PP_total, grad(N_1))*dx - dot(b,N_1)*dx

#####################
# TRANSPORT
#####################

# Define expressions used in variational forms
D_rhorho = Constant(D_rhorho)
d_rho = Constant(d_rho)
D_cc = Constant(D_cc)
d_c = Constant(d_c)

# Define modified diffusion coefficients
#D_rhorho_bar = -3.0*(D_rhorho-phif*(D_rhorho-D_rhorho/10))*A0*Grad_rho/tr(A) - 3.0*(D_rhoc-phif*(D_rhoc-D_rhoc/10))*rho*A0*Grad_c/tr(A)
#D_cc_bar = -3*(D_cc-phif*(D_cc-D_cc/10))*A0/tr(A)

# Define source terms
He = 1./(1.+exp(-gamma_theta*(Je - vartheta_e)))
S_rho = (p_rho + p_rho_c*c/(K_rho_c+c) + p_rho_theta*He)*(1-rho/K_rho_rho)*rho - d_rho*rho
S_c = (p_c_rho*c+ p_c_thetaE*He)*(rho/(K_c_c+c)) - d_c*c

# Diffusion equation
F_rho = ((rho - rho_n)/dt)*N_2*dx + D_rhorho*dot(grad(rho),grad(N_2))*dx - (S_rho*N_2)*dx
F_c = ((c - c_n)/dt)*N_3*dx + D_cc*dot(grad(c),grad(N_3))*dx - (S_c*N_3)*dx

#####################
# ASSEMBLE
#####################

# Assemble the entire problem
F = F_u + F_rho + F_c
# If the problem were linear we would do:
#a, L = lhs(F), rhs(F)

# Create VTK file for saving solution
vtkfile_u = File('resultsWoundFenics/solution_u.pvd')
vtkfile_rho = File('resultsWoundFenics/solution_rho.pvd')
vtkfile_c = File('resultsWoundFenics/solution_c.pvd')

#####################
# SOLVE
#####################

# Time-stepping
# If it were linear we wound need:
# Xi = Function(V)
# But as it is that was already defined earlier
t = 0
for n in range(num_steps):

    # Update current time
    t += dt

    # Compute solution
    # If it were linear we would have a == L
    solve(F == 0, Xi, bcs, form_compiler_parameters=ffc_options)

    # Save to file and plot solution
    output_u, output_rho, output_c = Xi.split()
    vtkfile_u << (output_u, t)
    vtkfile_rho << (output_rho, t)
    vtkfile_c << (output_c, t)
    #plot(rho)

    # Update previous solution
    Xi_n.assign(Xi)
    #rho_n.assign(rho)
    #c_n.assign(c)

    # Update progress bar
    #progress.update(t / T)

# Hold plot
#interactive()
