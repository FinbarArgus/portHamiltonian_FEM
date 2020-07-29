from fenics import *
import numpy as np

if __name__ == '__main__':

    timeIntScheme = 'SE'

    nx = 1
    ny = 1
    xLength = 1.0
    yLength = 0.25

    # solve the wave equation
    tFinal = 20
    numSteps = 2000

    # ------------------------------# Create Mesh #-------------------------------#
    mesh = RectangleMesh(Point(xLength, 0.0), Point(0.0, yLength), nx, ny)

    # time and time step
    dt_value = tFinal/numSteps
    dt = Constant(dt_value)

    # long as the boundary between the two models are the same
    Bl_em = 1
    R1_em = 0.0
    R2_em = 0.0
    K_em = 1
    m_em = 2
    L_em = 0.1

    J_em = np.array([[0, Bl_em, 0], [-Bl_em, 0, -1], [0, 1, 0]])

    R_em = np.array([[R1_em, 0, 0], [0, R2_em, 0], [0, 0, 0]])

    A_conv_em = np.array([[1/L_em, 0, 0], [0, 1/m_em, 0], [0, 0, K_em]])

    A_em = Constant((J_em - R_em).dot(A_conv_em))

    # ------------------------------# Create Function Spaces and functions#-------------------------------#

    # create function spaces for mesh and for Electromechanical ODE variables
    odeSpace = FiniteElement('R', triangle, 0) #order 0, 1 variable
    element = MixedElement([odeSpace, odeSpace, odeSpace])
    U = FunctionSpace(mesh, element)

    # Define trial and test functions
    xode_0, xode_1, xode_2  = TrialFunctions(U)
    xode = as_vector([xode_0, xode_1, xode_2])
    v_xode_0, v_xode_1, v_xode_2 = TestFunctions(U)
    v_xode = as_vector([v_xode_0, v_xode_1, v_xode_2])

    # Define functions for solutions at previous and current time steps
    u_n = Function(U)
    xode_0_n, xode_1_n, xode_2_n = split(u_n)
    xode_n = as_vector([xode_0_n, xode_1_n, xode_2_n])

    u_ = Function(U)
    xode_0_, xode_1_, xode_2_ = split(u_)

    # -------------------------------# Boundary Conditions #---------------------------------#

    # Left edge boundary condition for marking
    class LeftMarker(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], 0.0)

    # Initialise mesh function for neumann boundary. Facets are dim()-1, initialise subdomains to index 0
    boundaryDomains = MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
    boundaryDomains.set_all(0)

    leftMark= LeftMarker()
    leftMark.mark(boundaryDomains, 0)

    # redefine ds so that ds[0] is the dirichlet boundaries and ds[1] are the neumann boundaries
    ds = Measure('ds', domain=mesh, subdomain_data=boundaryDomains)

    # Forced input
    uInput = Expression('sin(3.14159*t)', degree=2, t=0)

    # -------------------------------# Problem Definition #---------------------------------#

    if timeIntScheme == 'SE':
        # THIS DOESNT WORK
        #implement Symplectic Euler integration
        xode_int = as_vector([xode[0], xode[1], xode_n[2]])
    elif timeIntScheme == 'EE':
        # THIS WORKS
        #implement Explicit Euler integration
        xode_int = as_vector([xode_n[0], xode_n[1], xode_n[2]])
    elif timeIntScheme == 'IE':
        # THIS WORKS
        #implement Implicit Euler integration
        xode_int = as_vector([xode[0], xode[1], xode[2]])

    A_xode_int= dot(A_em, xode_int)

    # Create residual function
    F = (-dot(v_xode, xode - xode_n)/dt + \
            v_xode[0]*(A_em[0, 0]*xode_int[0] + A_em[0, 1]*xode_int[1] + A_em[0, 2]*xode_int[2]) + \
            v_xode[1]*(A_em[1, 0]*xode_int[0] + A_em[1, 1]*xode_int[1] + A_em[1, 2]*xode_int[2]) + \
            v_xode[2]*(A_em[2, 0]*xode_int[0] + A_em[2, 1]*xode_int[1] + A_em[2, 2]*xode_int[2]) + \
            v_xode[0]*uInput)*ds(0)
    # F = (v_xode[0]*(A_em[0,0]*xode_int[0]))*ds(0)

    a = lhs(F)
    L = rhs(F)

    # Assemble matrices
    A = assemble(a)
    B = assemble(L)

    # Create vector for output displacement
    disp_vec = [0]

    # -------------------------------# Solve Loop #---------------------------------#

    t = 0
    for n in range(numSteps):
        set_log_level(LogLevel.ERROR)
        t += dt_value
        uInput.t = t

        B = assemble(L)

        #solve for this time step
        solve(A, u_.vector(), B)

        #output displacement
        disp = assemble(xode_2_*ds(0))/yLength
        disp_vec.append(disp)

        # Update previous solution
        u_n.assign(u_)

