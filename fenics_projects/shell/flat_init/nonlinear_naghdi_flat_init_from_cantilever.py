#!/usr/bin/env python3
# vim: set fileencoding=utf8 :

# .. _RollUpCantilever:
# 
# ======================================
# A non-linear Naghdi roll-up cantilever
# ======================================
# 
# This demo is implemented in the single Python file :download:`demo_nonlinear-naghdi-cantilever.py`.
# 
# This demo program solves the non-linear Naghdi shell equations on a rectangular
# plate with a constant bending moment applied. The plate rolls up completely on
# itself. The numerical locking issue is cured using a Durán-Liberman approach. 
# 
# To follow this demo you should know how to:
# 
# - Define a :py:class:`MixedElement` and a :py:class:`FunctionSpace` from it.
# - Define the Durán-Liberman (MITC) reduction operator using UFL for a linear
#   problem, e.g. Reissner-Mindlin. This procedure extends simply to the non-linear
#   problem we consider here.
# - Write variational forms using the Unified Form Language.
# - Automatically derive Jabobian and residuals using :py:func:`derivative`.
# - Apply Dirichlet boundary conditions using :py:class:`DirichletBC` and :py:func:`apply`.
# - Apply Neumann boundary conditions by marking a :py:class:`FacetFunction` and
#   create a new :py:class:`Measure` object.
# - Solve non-linear problems using :py:class:`ProjectedNonlinearProblem`.
# - Output data to XDMF files with :py:class:`XDMFFile`.
# 
# This demo then illustrates how to:
# 
# - Define and solve a non-linear Naghdi shell problem with a *flat* reference
#   configuration.
#
# We begin by setting up our Python environment with the required modules::

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

# Our reference middle surface is a rectangle :math:`\omega = [0, 1] \times [-0.5,
# 0.5]`::

# Testing with units of mm, need to check whether the governing equations hold
# radius = 12.5e-3
# testing with bigger values
radius = 12.5

domain = Circle(Point(0, 0), radius)
mesh = generate_mesh(domain, 128)

# We then define our :py:class:`MixedElement` which will discretise the in-plane
# displacements :math:`v \in [\mathrm{CG}_1]^2`, rotations :math:`\beta \in
# [\mathrm{CG}_2]^2`, out-of-plane displacements :math:`w \in \mathrm{CG}_1`, the
# shear strains. Two further auxilliary fields are also considered, the reduced
# shear strain :math:`\gamma_R`, and a Lagrange multiplier field :math:`p` which
# ties together the Naghdi shear strain calculated from the primal variables and the
# reduced shear strain :math:`\gamma_R`. Both :math:`p` and :math:`\gamma_R` are
# are discretised in the space :math:`\mathrm{NED}_1`, the vector-valued Nédélec
# elements of the first kind. The final element definition is then::

element = MixedElement([VectorElement("Lagrange", triangle, 1),
                        VectorElement("Lagrange", triangle, 2),
                        FiniteElement("Lagrange", triangle, 1),
                        FiniteElement("N1curl", triangle, 1),
                        RestrictedElement(FiniteElement("N1curl", triangle, 1), "edge")])

# We then pass our ``element`` through to the :py:class:`ProjectedFunctionSpace`
# constructor.  As in the other documented demos, we can project out :math:`p`
# and :math:`\gamma_R` fields at assembly time. We specify this by passing the
# argument ``num_projected_subspaces=2``::

U = ProjectedFunctionSpace(mesh, element, num_projected_subspaces=2)
U_F = U.full_space
U_P = U.projected_space

# We assume constant material parameters; Young's modulus :math:`E`, Poisson's
# ratio :math:`\nu`, and thickness :math:`t`::

# E, nu = Constant(0.8E6), Constant(0.5)
E, nu = Constant(0.8E6), Constant(0.0)
print(E)
print(nu)
mu = E/(2.0*(1.0 + nu))
lmbda = 2.0*mu*nu/(1.0 - 2.0*nu) 
# t = Constant(1E-4)
# Testing higher values
t = Constant(1E-1)

# Using only the `full` function space object ``U_F`` we setup our variational
# problem by defining the Lagrangian of our problem. We begin by creating a
# :py:class:`Function` and splitting it into each individual component function::
    
u, u_t, u_ = TrialFunction(U_F), TestFunction(U_F), Function(U_F)
v_, beta_, w_, Rgamma_, p_ = split(u_)

# For the Naghdi problem it is convienient to recombine the in-plane
# displacements :math:`v` and out-of-plane displacements :math:`w` into a single
# vector field :math:`z`:: 

z_ = as_vector([v_[0], v_[1], w_])

# We can now define our non-linear Naghdi strain measures. Assuming the normal
# fibres of the shell are unstrechable, we can parameterise the director vector
# field :math:`d: \omega \to \mathbb{R}^3` using the two independent rotations
# :math:`\beta`:: 

d = as_vector([sin(beta_[1])*cos(beta_[0]), -sin(beta_[0]), cos(beta_[1])*cos(beta_[0])])
   
# The deformation gradient :math:`F` can be defined as::
 
F = grad(z_) + as_tensor([[1.0, 0.0],
                         [0.0, 1.0],
                         [Constant(0.0), Constant(0.0)]])

# From which we can define the stretching (membrane) strain :math:`e`::

e = 0.5*(F.T*F - Identity(2))

# The curvature (bending) strain :math:`k`::

k = 0.5*(F.T*grad(d) + grad(d).T*F)

# and the shear strain :math:`\gamma`::

gamma = F.T*d

# We then define the constitutive law in terms of a general dual strain
# measure tensor :math:`X`:: 

S = lambda X: 2.0*mu*X + ((2.0*mu*lmbda)/(2.0*mu + lmbda))*tr(X)*Identity(2) 

# From which we can define the membrane energy density::

psi_N = 0.5*t*inner(S(e), e)

# the bending energy density::

psi_K = 0.5*(t**3/12.0)*inner(S(k), k)

# and the shear energy density::

psi_T = 0.5*t*mu*inner(Rgamma_, Rgamma_)

# and the total energy density from all three contributions::

psi = psi_N + psi_K + psi_T

# We define the Durán-Liberman reduction operator by tying the shear strain
# calculated with the displacement variables :math:`\gamma = F^T d` to the
# reduced shear strain :math:`\gamma_R` using the Lagrange multiplier field
# :math:`p`::

L_R = inner_e(gamma - Rgamma_, p_)

# We then turn to defining the boundary conditions and external loading.  On the
# outer domain we apply clamped boundary conditions which corresponds
# to constraining all generalised displacement fields to zero::

outer_boundary = lambda x, on_boundary: on_boundary
bc_v = DirichletBC(U.sub(0), Constant((0.0, 0.0)), outer_boundary) # in plane displacements
bc_a = DirichletBC(U.sub(1), Constant((0.0, 0.0)), outer_boundary) # alpha, beta angles
bc_w = DirichletBC(U.sub(2), Constant(0.0), outer_boundary) #out of plane displacement
bcs = [bc_v, bc_a, bc_w]

# M_right = Expression(('M'), M=0.0, degree=0)
F_uniform = Expression(('F'), F=0.0, degree=0)

# calculate external work as a function of the load, the vertical displacement and the face length?? check this
# Finbar todo, make this normal to surface and calculate correct work with correct displacement.
W_ext = F_uniform*z_[2]*dx

# We can now define our Lagrangian for the complete system::

L = psi*dx + L_R - W_ext
F = derivative(L, u_, u_t) 
J = derivative(F, u_, u)

# Before setting up the non-linear problem with the special `ProjectedFunctionSpace`
# functionality::

u_p_ = Function(U_P)
problem = ProjectedNonlinearProblem(U_P, F, u_, u_p_, bcs=bcs, J=J)
solver = NewtonSolver()

# and solving::

solver.parameters['error_on_nonconvergence'] = False
solver.parameters['maximum_iterations'] = 20
solver.parameters['linear_solver'] = "petsc"
solver.parameters['absolute_tolerance'] = 1E-20
solver.parameters['relative_tolerance'] = 1E-6

output_dir = "output/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# We apply the load with 20 continuation steps::

F_max = 1000.0
Fs = np.linspace(0.0, F_max, 30)

w_hs = []
v_hs = []

for i, F in enumerate(Fs):
    F_uniform.F = F
    solver.solve(problem, u_p_.vector())
    
    v_h, theta_h, w_h, Rgamma_h, p_h = u_.split(deepcopy=True)
    z_h = project(z_, VectorFunctionSpace(mesh, "CG", 1, dim=3))
    z_h.rename('z', 'z')

    XDMFFile(output_dir + "z_{}.xdmf".format(str(i).zfill(3))).write(z_h)

    # w_hs.append(w_h(0.0, 0.0))
    # v_hs.append(v_h(0.0, 0.0)[0])

# This problem has a simple closed-form analytical solution which we plot against
# for comparison::


# fig = plt.figure(figsize=(5.0, 5.0/1.648))
# plt.plot(v_hs/length, "x", label="$v_h/L$")
# plt.plot(w_hs/length, "o", label="$w_h/L$")
# plt.xlabel("$M/M_{\mathrm{max}}$")
# plt.ylabel("normalised displacement")
# plt.legend()
# plt.tight_layout()
# plt.savefig("output/cantilever-displacement-plot.pdf")
# plt.savefig("output/cantilever-displacement-plot.png")

