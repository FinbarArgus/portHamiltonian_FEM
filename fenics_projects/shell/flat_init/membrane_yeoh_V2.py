#!/usr/bin/env python3
# This file simulates the yeoh nonlinear-elastic membrane equations

import os
import numpy as np
import matplotlib.pyplot as plt

from dolfin import *
from ufl import RestrictedElement
from fenics_shells import *
from mshr import *

# We set the default quadrature degree. UFL's built in quadrature degree
# detection often overestimates the required degree for these complicated
# forms::
parameters["form_compiler"]["quadrature_degree"] = 2
# parameters["form_compiler"]["optimize"] = True
# parameters["form_compiler"]["cpp_optimize"] = True

# Define constants
# TODO(Finbar) first test with large radius for checking if it works, [I don't know if this matters]
radius = 10 # 12.5e-3 [m]
t = Constant(1.0e-1) # 1.0e-4 [m]k
nu = Constant(0.3) # 0.5 (OR maybe should be set to 0.499)

# Yeoh parameters # Units weren't given TODO(Finbar) double check units
C_1, C_2, C_3 = Constant(0.1811e6), Constant(-0.01598e6), Constant(0.00629e6)

# Very approximate youngs modulus for lame params
E = Constant(0.8e6)

# Lame params
mu = E/(2.0*(1.0 + nu))
lmbda = 2.0*mu*nu/(1.0 - 2.0*nu)

# model = 'Yeoh'
# model = 'general'
model = 'Venant_Kirchoff'
# model = 'Neo_Hookean'

# Create circular domain and mesh
domain = Circle(Point(0., 0.), radius)
mesh = generate_mesh(domain, 64)

# Define the types of finite elements to use. This should be tested to ensure the combination I use is accurate
element = MixedElement([VectorElement("Lagrange", triangle, 1),
                        VectorElement("Lagrange", triangle, 2),
                        FiniteElement("Lagrange", triangle, 1)])

# Create a Function space for this non-linear problem
U = FunctionSpace(mesh, element)

# set up the variational problem with the full function space
u, u_t, u_ = TrialFunction(U), TestFunction(U), Function(U)
v_, beta_, w_ = split(u_)

# create a single vector field with in plane and out of plane deformation
z_ = as_vector([v_[0], v_[1], w_])

# parameterise the director vector in terms of the spherical polar angles
# I think this assumes the normal fibres don't rotate TODO(Finbar) double check this
d = as_vector([sin(beta_[1]) * cos(beta_[0]), -sin(beta_[0]), cos(beta_[1]) * cos(beta_[0])])

# The deformation gradient :math:`F` can be defined as::
F = grad(z_) + as_tensor([[1.0, 0.0],
                          [0.0, 1.0],
                          [Constant(0.0), Constant(0.0)]])

# Right Cauchy Green Tensor
C = F.T*F

# Invariants of the right Cauchy Green Tensor
I_1 = tr(C)
# These two aren't used
I_2 = 0.5 * ((tr(C))**2 - tr(C*C))
I_3 = det(C)

# From The Cauchy Green tensor we can define the stretching (membrane) strain (Green-lagrange strain)
e = 0.5 * (C - Identity(2))

if model == 'Yeoh':
    # Constitutive laws Yeohs law for the cauchy stress (Not needed unless we want to output the stress)
    # sigma = C_1 + 2*C_2*(I_1 - 3) + 3*C_3*(I_1 - 3)**2
    print('{} model is not working as of yet'.format(model))

    # Yeohs law for the strain energy density
    psi = C_1*(I_1 - 3) + C_2*(I_1 - 3)**2 + C_3*(I_1 - 3)**3

elif model == 'Neo_Hookean':
    print('{} model is not working as of yet'.format(model))

    psi = C_1*(I_1 - 3)

elif model == 'Venant_Kirchoff':
    print('{} model is not working as of yet'.format(model))

    # Venant_Kirchoff law for the strain energy density
    psi = lmbda/2.0 *((tr(e))**2 + mu*tr(e*e))

elif model == 'general':
    print('{} model is not working as of yet'.format(model))
    S = lambda X: 2.0*mu*X + ((2.0*mu*lmbda)/(2.0*mu +lmbda))*tr(X)*Identity(2)

    psi = 0.5*inner(S(e), e)
else:
    print('model, {} has not been implemented yet'.format(model))
    quit()

# Uniform load TODO(Finbar) make this perpendicular to the normal
P_uniform = Expression(('P'), P=0.0, degree=0)

# Calculate the external work as a function of the normal force
# W_ext = P_uniform*w_*dx # This calculates external work from vertical load
W_ext = P_uniform*inner(d, z_)*dx #This calculates external work from normal (pressure) load

# Now we can define the lagrangian for the complete system
# TODO(Finbar) double check that the t for thickness should be there
L = psi*t*dx - W_ext
dL = derivative(L, u_, u_t)
J = derivative(dL, u_, u)

# Boundary conditions
outer_boundary = lambda x, on_boundary: on_boundary
bc_v = DirichletBC(U.sub(0), project(Constant((0.0, 0.0)), U.sub(0).collapse()), outer_boundary) # in plane displacements
bc_a = DirichletBC(U.sub(1), project(Constant((0.0, 0.0)), U.sub(1).collapse()), outer_boundary) # alpha, beta angles
bc_w = DirichletBC(U.sub(2), project(Constant(0.0), U.sub(2).collapse()), outer_boundary) #out of plane displacement
bcs = [bc_v, bc_a, bc_w]
# bcs = DirichletBC(U, Constant((0., 0., 0., 0., 0.)), outer_boundary)

# set up the nonlinear problem
problem = NonlinearVariationalProblem(dL, u_, bcs, J)
solver = NonlinearVariationalSolver(problem)

prm = solver.parameters
info(prm, True)
prm['nonlinear_solver'] = 'newton' # default could be snes
prm['newton_solver']['maximum_iterations'] = 60
prm['newton_solver']['relative_tolerance'] = 1e-8
prm['newton_solver']['absolute_tolerance'] = 1e-10
prm['newton_solver']['relaxation_parameter'] = 1.0
prm['newton_solver']['linear_solver'] = 'gmres'
prm['newton_solver']['preconditioner'] = 'jacobi'
# prm['newton_solver']['linear_solver'] = "mumps"
# prm['newton_solver']['linear_solver'] = "lu"
prm['newton_solver']['krylov_solver']['absolute_tolerance'] = 1e-9
prm['newton_solver']['krylov_solver']['relative_tolerance'] = 1e-7
prm['newton_solver']['krylov_solver']['maximum_iterations'] = 1000
prm['newton_solver']['krylov_solver']['monitor_convergence'] = True
prm['newton_solver']['krylov_solver']['nonzero_initial_guess'] = False

# Create output directory
output_dir = "output/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# and solving::

# We apply the load with 20 continuation steps::

# P_max = 1000.0
P_max = 0.001
Ps = np.linspace(0.0, P_max, 20)
u_.assign(project(Constant((0, 0, 0, 0, 0)), U))

for i, P in enumerate(Ps):
    P_uniform.P = P
    print(P)
    (niter, cond) = solver.solve()

    v_h, theta_h, w_h = u_.split(deepcopy=True)
    z_h = project(z_, VectorFunctionSpace(mesh, "CG", 1, dim=3))
    z_h.rename('z', 'z')

    print("Increment %d of %s. Converged in %2d iterations. F:  %.2f" %(i, Ps.size, niter, F,))

    XDMFFile(output_dir + "z_{}.xdmf".format(str(i).zfill(3))).write(z_h)







