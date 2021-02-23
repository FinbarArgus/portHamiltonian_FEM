from fenics import *
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import mshr
from mpi4py import MPI
import control.matlab as ctrl
import sys
import petsc4py.PETSc as pet
from control_input import *

def wave_2D_solve(tFinal, numSteps, outputDir,
                  nx, xLength, yLength,
                  domainShape='R', timeIntScheme='SV', dirichletImp='weak',
                  K_wave=1, rho=1, BOUNDARYPLOT=False, interConnection=None,
                  analytical=False, saveP=True, controlType=None,
                  input_stop_t=None, control_start_t=None, basis_order=(1, 1)):

    """ function for solving the 2D wave equation in fenics

    The leftmost boundary is a forced dirichlet condition
    The rightmost boundary is a fixed zero dirichlet boundary
    The other boundaries are zero neumann boundaries

    :param tFinal: [float] final time of solve
    :param nx: [int] approximate number of nodes in x direction
    :param xL: [float] x length of domain
    :param yL: [float] y length of domain
    :param domainShape: [string] 'R' for rectangle or 'S_1C' for square with 1 input channel
    :param dirichletImplementation: [string] how the dirichlet BCs are implemented either 'strong' or 'weak'
    :param timeIntScheme: [string] what time integration scheme to use. List of possible below
                        'EH' = Explicit Heuns, 'SE' = Symplectic Euler, 'SV' = Stormer Verlet
    :param K_wave: [float] material stiffness
    :param rho: [float] material density
    :param BOUNDARYPLOT: [bool] whether the boundary conditions should be plotted for debugging
    :param interConnection: [string] type of model for the electromechanical system at the boundary
                            either IC for 3 dof or None for non interconnection (just wave equation).
                            For IC, the electromechanical system doesn't have have a capacitor.
    :param saveP: [bool] whether to save p throughout the domain as an xdf file for paraview postprocessing
    :param controlType: If control is implemented this is the type of control, either 'casimir' or 'passivity'
                        TODO Currently only Passivity based control is working
    :param input_stop_t: For control, when the input voltage stops
    :param control_start_t: Time that control starts
    :poram basis_order: order of the lagrange and Raviart-Thomas elements respectively
    :return:
            H_array: array of output vectors of length numSteps
            numCells: number of cells in mesh
    """

    # set up parallel processing if it is used
    comm = MPI.comm_world
    numProcs = comm.Get_size()
    if numProcs > 1:
        parallel = True
    else:
        parallel = False
        
    rank = comm.Get_rank()

    # check case combinations
    if controlType is 'lqr':
        print('lqr no longer implemented')
        quit()
    if controlType is 'passivity':
        if interConnection is not 'IC':
            print('passivity control only implemented for IC interconnection')
            quit()
    if controlType is 'casimir':
        if interConnection is not None:
            print('casimir control not implemented for IC interconnection')
            quit()
    if analytical:
        if not domainShape == 'R':
            print('analytical solution for rectangle domain only')
            quit()
    if interConnection == 'IC':
        # ensure weak boundary conditions for interconnection
        if not dirichletImp == 'weak':
            print('can only do interconnection model with weak boundary conditions')
            quit()

    # This is a boolean for outputting bc types for visualisation of the boundary conditions in paraview
    BOUNDARYPLOT = False
    # Find initial time
    tic = time.time()
    if rank == 0:
        print('Started solving for case: {}, {}, {} '.format(domainShape, timeIntScheme, dirichletImp))

    # ------------------------------# Create Mesh #-------------------------------#
    if domainShape == 'R':
        xInputStart = 0.0  # at what x location is the input boundary
        mainRectangle = mshr.Rectangle(Point(0.0, 0.0), Point(xLength, yLength))
        mesh = mshr.generate_mesh(mainRectangle, nx)
    elif domainShape == 'S_1C':

        xInputStart = -0.10
        # See paper for diagram of the square domain with small rectangle input channel
        inputRectanglePoint1 = Point(xInputStart, xLength/2 - yLength/2)
        inputRectanglePoint2 = Point(0.0, xLength/2 + yLength/2)
        mainRectangle = mshr.Rectangle(Point(0.0, 0.0), Point(xLength, xLength))
        inputRectangle = mshr.Rectangle(inputRectanglePoint1, inputRectanglePoint2)
        domain = mainRectangle + inputRectangle
        mesh = mshr.generate_mesh(domain, nx)

        # Mark cells for refinement. (Refine near the input rectangle)
        inputPoint = Point(0, xLength/2)
        for i in range(1):
            cell_markers = MeshFunction("bool", mesh, mesh.topology().dim())
            for c in cells(mesh):
                if c.midpoint().distance(inputPoint) <= yLength:
                    cell_markers[c] = True
                else:
                    cell_markers[c] = False

            # Refine mesh
            mesh = refine(mesh, cell_markers)

    else:
        print('domainShape \"{}\" hasn\'t been created'.format(domainShape))
        quit()

    #get number of cells in mesh
    numCells = mesh.num_cells()
    if rank == 0:
        print('number of cells = {}'.format(numCells))

    # time and time step
    dt_value = tFinal/numSteps
    dt = Constant(dt_value)

    # get number of intermediate steps required in time integration scheme
    numIntSubSteps = 1
    if timeIntScheme in ['EH', 'SV']:
        numIntSubSteps = 2
    # Constant wave speed
    c_squared = Constant(K_wave/rho)
    c_squared_float = K_wave/rho
    c = np.sqrt(K_wave/rho)
    K_wave = Constant(K_wave)
    rho_val = rho
    # rho = Constant(rho)

    if interConnection == 'IC':
        if timeIntScheme not in ['SV', 'SE', 'SM']:
            print('only SE, SV and SM time int schemes implemented for interconnection model')
            quit()

        Bl_em = 5
        R1_em = 0.0
        R2_em = 0.0
        K_em = 5
        m_em = 0.15
        L_em = 0.1

        J_em = np.array([[0, Bl_em, 0], [-Bl_em, 0, -1], [0, 1, 0]])

        R_em = np.array([[R1_em, 0, 0], [0, R2_em, 0], [0, 0, 0]])

        A_conv_em = np.array([[1/L_em, 0, 0], [0, 1/m_em, 0], [0, 0, K_em]])

        A_em = Constant((J_em - R_em).dot(A_conv_em))

    # ------------------------------# Create Function Spaces and functions#-------------------------------#

    # Create function spaces
    if interConnection == 'IC':

        # create function spaces for mesh and for Electromechanical ODE variables
        P1 = FiniteElement('P', triangle, basis_order[0])
        RT = FiniteElement('RT', triangle, basis_order[1])
        odeSpace = FiniteElement('R', triangle, 0)  # order 0, 1 variable

        element = MixedElement([P1, RT, odeSpace, odeSpace, odeSpace])
        U = FunctionSpace(mesh, element)

        # Define trial and test functions
        p, q, xode_0, xode_1, xode_2 = TrialFunctions(U)
        xode = as_vector([xode_0, xode_1, xode_2])
        v_p, v_q, v_xode_0, v_xode_1, v_xode_2 = TestFunctions(U)
        v_xode = as_vector([v_xode_0, v_xode_1, v_xode_2])

        # Define functions for solutions at previous and current time steps
        u_m = Function(U)
        p_m, q_m, xode_0_m, xode_1_m, xode_2_m = split(u_m)
        xode_m = as_vector([xode_0_m, xode_1_m, xode_2_m])

        u_ = Function(U)
        p_, q_, xode_0_, xode_1_, xode_2_ = split(u_)
        xode_ = as_vector([xode_0_, xode_1_, xode_2_])

        u_temp = Function(U)
        p_temp, q_temp, xode_0_temp, xode_1_temp, xode_2_temp = split(u_temp)
        xode_temp = as_vector([xode_0_temp, xode_1_temp, xode_2_temp])

    else:
        # interconnection with a lumped param model is not used, this is the basic wave equation
        P1 = FiniteElement('P', triangle, basis_order[0])
        RT = FiniteElement('RT', triangle, basis_order[1])

        element = MixedElement([P1, RT])
        U = FunctionSpace(mesh, element)

        # Define trial and test functions
        p, q = TrialFunctions(U)
        v_p, v_q = TestFunctions(U)

        # Define functions for solutions at previous and current time steps
        u_m = Function(U)
        p_m, q_m = split(u_m)

        u_ = Function(U)
        p_, q_ = split(u_)

        u_temp = Function(U)
        p_temp, q_temp = split(u_temp)


    # -------------------------------# Boundary Conditions #---------------------------------#

    # Left edge boundary condition for defining DirchletBC
    def forced_boundary(x, on_boundary):
        return on_boundary and near(x[0], xInputStart)

    # Right edge boundary condition for defining DirichletBC
    def fixed_boundary(x, on_boundary):
        return on_boundary and near(x[0], xLength)

    # Left edge boundary condition for marking
    class LeftMarker(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], xInputStart)

    # Right edge boundary condition for marking
    class RightMarker(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], xLength)

    # This is the Top boundary for rectangle domain, empty boundary for square domain
    class TopMarker(SubDomain):
        if domainShape == 'R':
            def inside(self, x, on_boundary):
                return on_boundary and near(x[1], yLength)
        elif domainShape == 'S_1C':
            def inside(self, x, on_boundary):
                return False


    # general/bottom boundary for marking
    class GeneralMarker(SubDomain):
        if domainShape == 'R':
            def inside(self, x, on_boundary):
                return on_boundary and near(x[1], 0.0)

        elif domainShape == 'S_1C':
            def inside(self, x, on_boundary):
                return on_boundary and (near(x[1], 0) or near(x[1], xLength) or
                                        near(x[1], xLength/2 - yLength/2) or
                                        near(x[1], xLength/2 + yLength/2) or
                                        near(x[0], 0))

    # Initialise mesh function for neumann boundary. Facets are dim()-1, initialise subdomains to index 4
    boundaryDomains = MeshFunction('size_t', mesh, mesh.topology().dim()-1)
    boundaryDomains.set_all(4)

    leftMark = LeftMarker()
    leftMark.mark(boundaryDomains, 0)

    TopMark = TopMarker()
    TopMark.mark(boundaryDomains, 1)

    rightMark = RightMarker()
    rightMark.mark(boundaryDomains, 2)

    generalMark = GeneralMarker()
    generalMark.mark(boundaryDomains, 3)

    # redefine ds so that ds(0) is left boundary, ds(1) is top, ds(2) is right, and ds(3) is bottom/general boundary
    ds = Measure('ds', domain=mesh, subdomain_data=boundaryDomains)
    # Get normal for boundaries
    n = FacetNormal(mesh)

    if BOUNDARYPLOT:
        # output mesh to file if BOUNDARYPLOT is True
        File(os.path.join(outputDir,'markers.pvd')) << boundaryDomains
        quit()

    # Define what time step q and p variables are used in each substep
    # The underscore variables are used to calculate the energy flow out of
    # the boundaries
    if timeIntScheme == 'SV':
        q0 = q_m
        q1 = q_temp
        p0 = p_m
        p1 = p
    elif timeIntScheme == 'EE':
        q0 = None
        q1 = q_m
        p0 = None
        p1 = p_m
    elif timeIntScheme == 'IE':
        q0 = None
        q1 = q
        p0 = None
        p1 = p
    elif timeIntScheme == 'SE':
        q0 = None
        q1 = q_m
        p0 = None
        p1 = p
    elif timeIntScheme == 'SM':
        q0 = None
        q1 = 0.5*(q_m + q)
        p0 = None
        p1 = 0.5*(p_m + p)
    elif timeIntScheme == 'EH':
        q0 = q_m
        q1 = 0.5 * (q_m + q_temp)
        p0 = p_m
        p1 = 0.5 * (p_m + p_temp)

    # ___Top Boundary__
    if controlType is 'casimir':
        # TODO this top boundary will be the control
        # Neumann Impedance condition
        qTopBoundary1 = 0.5*(p_m + p)
        qTopBoundary1_ = 0.5*(p_m + p_)
        pTopBoundary1 = p1
        # Dirichlet Impedance condition should have the same result as above
        # pTopBoundary1 = dot(0.5*q_m + q, n)
        # qTopBoundary1 = dot(0.5*(q_m + q), n)
        # qTopBoundary1_ = dot(0.5*(q_m + q_), n)
        # THe above currently only works for 1 substep methods
    else:
        # All other cases have a zero neumann condition
        qTopBoundary0 = Constant(0.0)
        qTopBoundary1 = Constant(0.0)
        qTopBoundary1_ = qTopBoundary1
        pTopBoundary0 = p0
        pTopBoundary1 = p1

    # ___Right Boundary__
    if controlType is 'casimir':
        # set right boundary to be an impedance boundary
        # This boundary dissipates energy and should model the waves leaving the domain
        pRightBoundary1 = dot(0.5*(q_m + q), n)
        pRightBoundary1_ = dot(0.5*(q_m + q_), n)
    else:
        # all other cases right boundary is a zero dirichlet condition
        pRightBoundary0 = Constant(0.0)
        pRightBoundary1 = Constant(0.0)
        pRightBoundary1_ = Constant(0.0)

    # all cases the qRightBoundary is the same as q0, q1
    if numIntSubSteps >1:
        qRightBoundary0 = dot(q0, n)
    qRightBoundary1 = dot(q1, n)

    # ___Bottom/general Boundary__
    # All cases have a zero neumann condition
    qGenBoundary0 = Constant(0.0)
    qGenBoundary1 = Constant(0.0)

    # all cases the pTopBoundary is the same as p0, p1
    pGenBoundary0 = p0
    pGenBoundary1 = p1

    # __Left boundary__
    if not interConnection:
        if analytical:
            cSqrtConst = c*np.sqrt(pi**2/(4*xLength**2) + 4*pi**2/yLength**2)
            pLeftBoundary1 = Expression('cSqrtConst*sin(2*pi*x[1]/(yLength) + pi/2)*cos(t*cSqrtConst)',
                       degree=8, t=0, yLength=yLength, cSqrtConst=cSqrtConst)
        else:
            if controlType is 'casimir':
                pLeftBoundary1 = Expression('3*sin(8*pi*t)', degree=2, t=0)
            else:
                pLeftBoundary1 = Expression('(t<0.25) ? 5*sin(8*pi*t) : 0.0', degree=2, t=0)


        pLeftBoundary0 = pLeftBoundary1
        pLeftBoundary0_ = pLeftBoundary1
        pLeftBoundary1_ = pLeftBoundary1
    else:
        if timeIntScheme == 'SV':
            pLeftBoundary0 = xode_m[1]/(m_em)
            pLeftBoundary1 = xode[1]/(m_em)
            pLeftBoundary0_ = xode_m[1]/(m_em)
            pLeftBoundary1_ = xode_[1]/(m_em)
        elif timeIntScheme == 'SE':
            pLeftBoundary1 = xode[1]/(m_em)
            pLeftBoundary1_ = xode_[1]/(m_em)
        elif timeIntScheme == 'SM':
            pLeftBoundary1 = 0.5*(xode_m[1] + xode[1])/(m_em)
            pLeftBoundary1_ = 0.5*(xode_m[1] + xode_[1])/(m_em)

        else:
            print('only SV and SE time int schemes have been implemented for interconnection')
            quit()
    # all cases the qLeftBoundary is the same as q0, q1
    if numIntSubSteps > 1:
        qLeftBoundary0 = dot(q0, n)
    qLeftBoundary1 = dot(q1, n)

    # If we have dirichlet conditions applied in the strong way set them here, otherwise
    # they are set in the integration by parts term in the problem definition
    if dirichletImp == 'strong':
        bc_forced = DirichletBC(U.sub(0), pLeftBoundary1, forced_boundary)

        # fixed boundary at right boundary
        bc_fixed = DirichletBC(U.sub(0), Constant(0.0), fixed_boundary)

        # List of boundary condition objects
        bcs = [bc_forced, bc_fixed]
    else:
        bcs = []


    # -------------------------------# define interconnection voltage input #---------------------------------#

    if interConnection:
        if domainShape == 'R':
            if controlType == None:
                uInput = Expression('(t<0.25) ? 10*sin(8*pi*t) : 0.0', degree=2, t=0)
            elif controlType == 'passivity':
                # cpp_code = '''
                #     (t<input_stop_t) ? 10*sin(8*pi*t) : (t<control_start_t) ? 0.0 : (h_1>1) ? -10.0 : 10.0
                # '''
                uInput = Passive_control_input(t=0.0, h_1=0.0,
                                               input_stop_t=input_stop_t, control_start_t=control_start_t)

        elif domainShape == 'S_1C':
            uInput = Expression('(t<0.25) ? 10*sin(8*pi*t) : 0.0', degree=2, t=0)

    # -------------------------------# Problem Definition #---------------------------------#

    # Define variational problem
    if timeIntScheme is 'SV':
        if dirichletImp == 'strong':
            if numIntSubSteps > 1:
                F_temp = (p - p_m)*v_p*dx + 0.5*dt*c_squared*(-dot(q0, grad(v_p))*dx +
                                                                 qTopBoundary0*v_p*ds(1) +
                                                                 qGenBoundary0*v_p*ds(3)) + \
                         dot(q - q_m, v_q)*dx + 0.5*dt*(dot(grad(p0), v_q))*dx

            F = (p - p_m)*v_p*dx + dt*c_squared*(-dot(q1, grad(v_p))*dx +
                                                 qTopBoundary1*v_p*ds(1) +
                                                 qGenBoundary1*v_p*ds(3)) + \
                dot(q - q_temp, v_q)*dx + 0.5*dt*(dot(grad(p1), v_q))*dx
        elif dirichletImp == 'weak':
            if numIntSubSteps > 1:
                F_temp = (p - p_m)*v_p*dx + 0.5*dt*c_squared*(-dot(q0, grad(v_p))*dx +
                                                              qLeftBoundary0*v_p*ds(0) +
                                                              qTopBoundary0*v_p*ds(1) +
                                                              qRightBoundary0*v_p*ds(2) +
                                                              qGenBoundary0*v_p*ds(3)) + \
                         dot(q - q_m, v_q)*dx + 0.5*dt*(-div(v_q)*p0*dx +
                                                              dot(v_q, n)*pLeftBoundary0*ds(0) +
                                                              dot(v_q, n)*pTopBoundary0*ds(1) +
                                                              dot(v_q, n)*pRightBoundary0*ds(2) +
                                                              dot(v_q, n)*pGenBoundary0*ds(3))

            F = (p - p_m)*v_p*dx + dt*c_squared*(-dot(q1, grad(v_p))*dx +
                                                 qLeftBoundary1*v_p*ds(0) +
                                                 qTopBoundary1*v_p*ds(1) +
                                                 qRightBoundary1*v_p*ds(2) +
                                                 qGenBoundary1*v_p*ds(3)) + \
                dot(q - q_temp, v_q)*dx + 0.5*dt*(-div(v_q)*p1*dx +
                                                 dot(v_q, n)*pLeftBoundary1*ds(0) +
                                                 dot(v_q, n)*pTopBoundary1*ds(1) +
                                                 dot(v_q, n)*pRightBoundary1*ds(2) +
                                                 dot(v_q, n)*pGenBoundary1*ds(3))
    elif timeIntScheme in ['IE', 'EE', 'SE', 'SM', 'EH']:
        if dirichletImp == 'strong':
            if numIntSubSteps >1:
                F_temp = (p - p_m)*v_p*dx + dt*c_squared*(-dot(q0, grad(v_p))*dx +
                                                     qTopBoundary0*v_p*ds(1) +
                                                     qGenBoundary0*v_p*ds(3)) + \
                    dot(q - q_m, v_q)*dx + dt*(dot(grad(p0), v_q))*dx

            F = (p - p_m)*v_p*dx + dt*c_squared*(-dot(q1, grad(v_p))*dx +
                                                 qTopBoundary1*v_p*ds(1) +
                                                 qGenBoundary1*v_p*ds(3)) + \
                dot(q - q_m, v_q)*dx + dt*(dot(grad(p1), v_q))*dx
        elif dirichletImp == 'weak':
            if numIntSubSteps >1:
                F_temp = (p - p_m)*v_p*dx + dt*c_squared*(-dot(q0, grad(v_p))*dx +
                                                     qLeftBoundary0*v_p*ds(0) +
                                                     qTopBoundary0*v_p*ds(1) +
                                                     qRightBoundary0*v_p*ds(2) +
                                                     qGenBoundary0*v_p*ds(3)) + \
                    dot(q - q_m, v_q)*dx + dt*(-div(v_q)*p0*dx +
                                                     dot(v_q, n)*pLeftBoundary0*ds(0) +
                                                     dot(v_q, n)*pTopBoundary0*ds(1) +
                                                     dot(v_q, n)*pRightBoundary0*ds(2) +
                                                     dot(v_q, n)*pGenBoundary0*ds(3))

            F = (p - p_m)*v_p*dx + dt*c_squared*(-dot(q1, grad(v_p))*dx +
                                                 qLeftBoundary1*v_p*ds(0) +
                                                 qTopBoundary1*v_p*ds(1) +
                                                 qRightBoundary1*v_p*ds(2) +
                                                 qGenBoundary1*v_p*ds(3)) + \
                dot(q - q_m, v_q)*dx + dt*(-div(v_q)*p1*dx +
                                                 dot(v_q, n)*pLeftBoundary1*ds(0) +
                                                 dot(v_q, n)*pTopBoundary1*ds(1) +
                                                 dot(v_q, n)*pRightBoundary1*ds(2) +
                                                 dot(v_q, n)*pGenBoundary1*ds(3))

        else:
            print('dirichlet implementation {} not implemented'.format(dirichletImp))
            quit()

    else:
        print('time integration scheme {} not implemented'.format(timeIntScheme))
        quit()

    if interConnection == 'IC':
        if timeIntScheme is 'SV':
            F_temp_em = (-dot(v_xode, xode - xode_m)/(0.5*dt) +
                         v_xode[0]*(A_em[0, 0]*xode_m[0] + A_em[0, 1]*xode_m[1] + A_em[0, 2]*xode[2]) +
                         v_xode[1]*(A_em[1, 0]*xode_m[0] + A_em[1, 1]*xode_m[1] + A_em[1, 2]*xode[2]) +
                         v_xode[2]*(A_em[2, 0]*xode_m[0] + A_em[2, 1]*xode_m[1] + A_em[2, 2]*xode[2]) +
                         v_xode[0]*uInput + K_wave*yLength*v_xode[1]*qLeftBoundary0)*ds(0)

            F_temp += F_temp_em

            F_em = (-(v_xode[0]*(xode[0] - xode_m[0]) +
                      v_xode[1]*(xode[1] - xode_m[1]) +
                      v_xode[2]*(xode[2] - xode_temp[2]))/dt +
                    0.5*v_xode[0]*(A_em[0, 0]*xode_m[0] + A_em[0, 1]*xode_m[1] + A_em[0, 2]*xode_temp[2] +
                                   A_em[0, 0]*xode[0] + A_em[0, 1]*xode[1] + A_em[0, 2]*xode_temp[2]) +
                    0.5*v_xode[1]*(A_em[1, 0]*xode_m[0] + A_em[1, 1]*xode_m[1] + A_em[1, 2]*xode_temp[2] +
                                   A_em[1, 0]*xode[0] + A_em[1, 1]*xode[1] + A_em[1, 2]*xode_temp[2]) +
                    0.5*v_xode[2]*(A_em[2, 0]*xode[0] + A_em[2, 1]*xode[1] + A_em[2, 2]*xode_temp[2]) +
                    v_xode[0]*uInput + K_wave*yLength*v_xode[1]*qLeftBoundary1)*ds(0)
            # don't multiply the input by a half because it is for a full step

            F += F_em
        elif timeIntScheme is 'SE':

            F_em = (-dot(v_xode, xode - xode_m)/dt +
                        v_xode[0]*(A_em[0, 0]*xode[0] + A_em[0, 1]*xode[1] + A_em[0, 2]*xode_m[2]) +
                        v_xode[1]*(A_em[1, 0]*xode[0] + A_em[1, 1]*xode[1] + A_em[1, 2]*xode_m[2]) +
                        v_xode[2]*(A_em[2, 0]*xode[0] + A_em[2, 1]*xode[1] + A_em[2, 2]*xode_m[2]) +
                        v_xode[0]*uInput + K_wave*yLength*v_xode[1]*qLeftBoundary1)*ds(0)

            F += F_em

        elif timeIntScheme is 'SM':

            F_em = (-dot(v_xode, xode - xode_m)/dt +
                    0.5*v_xode[0]*(A_em[0, 0]*xode[0] + A_em[0, 0]*xode_m[0] +
                                   A_em[0, 1]*xode[1] + A_em[0, 1]*xode_m[1] +
                                   A_em[0, 2]*xode[2] + A_em[0, 2]*xode_m[2]) +
                    0.5*v_xode[1]*(A_em[1, 0]*xode[0] + A_em[1, 0]*xode_m[0] +
                                   A_em[1, 1]*xode[1] + A_em[1, 1]*xode_m[1] +
                                   A_em[1, 2]*xode[2] + A_em[1, 2]*xode_m[2]) +
                    0.5*v_xode[2]*(A_em[2, 0]*xode[0] + A_em[2, 0]*xode_m[0] +
                                   A_em[2, 1]*xode[1] + A_em[2, 1]*xode_m[1] +
                                   A_em[2, 2]*xode[2] + A_em[2, 2]*xode_m[2]) +
                    v_xode[0]*uInput + K_wave*yLength*v_xode[1]*qLeftBoundary1)*ds(0)

            F += F_em
        else:
            print('interconnection is not implemented for {}'.format(timeIntScheme))
            quit()

    a = lhs(F)
    b = rhs(F)
    if numIntSubSteps > 1:
        a_temp = lhs(F_temp)
        b_temp = rhs(F_temp)

    # Assemble matrices
    A = assemble(a)
    B = assemble(b)
    if numIntSubSteps > 1:
        A_temp = assemble(a_temp)
        B_temp = assemble(b_temp)

    # Apply boundary conditions to matrices
    [bc.apply(A, B) for bc in bcs]
    if numIntSubSteps > 1:
        [bc.apply(A_temp, B_temp) for bc in bcs]

    # to calculate the boundary energy flow
    # first I define what time step q and p variables are used in each substep in
    # order to calculate the energy flow through the boundaries
    if timeIntScheme == 'SV':
        q0_ = q_temp
        q1_ = q_temp
    elif timeIntScheme == 'EE':
        q0_ = None
        q1_ = q_m
    elif timeIntScheme == 'IE':
        q0_ = None
        q1_ = q_
    elif timeIntScheme == 'SE':
        q0_ = None
        q1_ = q_m
    elif timeIntScheme == 'SM':
        q0_ = None
        q1_ = 0.5*(q_m + q_)
        p1_ = 0.5*(p_m + p_)
    elif timeIntScheme == 'EH':
        q0_ = q_m
        q1_ = q_temp

    if numIntSubSteps >1:
        # These are the energy contributions from the left and top boundaries for the first half timestep
        pT_L_q_temp_left = 0.5*dt*K_wave*dot(q0_, n)*pLeftBoundary0_*ds(0)

        # These are the energy contributions from the left, general and right boundaries for the second half timestep
        pT_L_q_left = 0.5*dt*K_wave*dot(q1_, n)*pLeftBoundary1_*ds(0)

        # Total energy contribution from the boundaries for the time step
        pT_L_q = pT_L_q_temp_left + pT_L_q_left  # + pT_L_q_gen + pT_L_q_right

    else:
        pT_L_q_left = dt*K_wave*dot(q1_, n)*pLeftBoundary1_*ds(0)
        if controlType == 'casimir':
            pT_L_q_top = dt*K_wave*qTopBoundary1_*p1_*ds(1)
            pT_L_q_right = dt*K_wave*dot(q1_, n)*pRightBoundary1_*ds(2)
        else:
            pT_L_q_top = 0.0
            pT_L_q_right = 0.0
        # pT_L_q_gen = dt*K_wave*q_bNormal*p1*ds(1)
        # pT_L_q_right = dt*K_wave*dot(q1, n)*pRightBoundary*ds(2)

        pT_L_q = pT_L_q_left + pT_L_q_top + pT_L_q_right

    # -------------------------------# Set up output and plotting #---------------------------------#

    # Create xdmf Files for visualisation
    if saveP:
        xdmfFile_p = XDMFFile(os.path.join(outputDir, 'p.xdmf'))
        xdmfFile_p.parameters["flush_output"] = True
        if analytical:
            xdmfFile_p_exact = XDMFFile(os.path.join(outputDir, 'p_exact.xdmf'))
            xdmfFile_p_exact.parameters["flush_output"] = True

#   if saveQ:
#       xdmfFile_q = XDMFFile(os.path.join(outputDir, 'q.xdmf'))
#       xdmfFile_q.parameters["flush_output"] = True

    # Create progress bar
    progress = Progress('Time-stepping', numSteps)

    # Create vector for output
    E_vec = []
    H_vec = []
    # HLeft_vec= [0]
    # HGen_vec= [0]
    # HRight_vec= [0]
    t_vec = []
    disp_vec = []
    bEnergy_vec =[]
    inpEnergy_vec =[]
    H_em_vec =[]
    H_wave_vec =[]
    if analytical:
        error_vec = []
        error_vec_max = []
        p_point_vec = []
        p_point_exact_vec = []

    if rank == 0:
        # Create plot for Energy Residual or analytic error
        fig, ax = plt.subplots(1, 1)

        # create line object for plotting
        if analytical:
            line, = ax.plot([], lw=0.5, color='k', linestyle='-',
                        label='analytical error {}, {}, {}'.format(domainShape,timeIntScheme,dirichletImp))
        else:
            if controlType is None:
                line, = ax.plot([], lw=0.5, color='k', linestyle='-',
                                label='Energy Residual {}, {}, {}'.format(domainShape,timeIntScheme,dirichletImp))
            else:
                line, = ax.plot([], lw=0.5, color='k', linestyle='-',
                                label='Hamiltonian {}, {}, {}'.format(domainShape,timeIntScheme,dirichletImp))

        # text = ax.text(0.001, 0.001, "")
        ax.set_xlim(0, tFinal)
        if domainShape == 'R':
            if analytical:
                # ax.set_ylim(0, 0.1)
                ax.set_ylim(0, 2.0)
            else:
                if controlType is None:
                    ax.set_ylim(-0.03, 0.05)
                    if dirichletImp == 'weak' and timeIntScheme in ['SV', 'SE', 'EH']:
                        ax.set_ylim(-0.0005, 0.0008)
                elif controlType is 'casimir':
                    ax.set_ylim(0, 3.5)
                else:
                    ax.set_ylim(0, 0.4)

        elif domainShape == 'S_1C':
            ax.set_ylim(-0.001, 0.001)

        ax.legend()
        if analytical:
            ax.set_ylabel('analytical error')
        else:
            if controlType is None:
                ax.set_ylabel('Energy Residual [J]')
            else:
                ax.set_ylabel('Hamiltonian [J]')

        ax.set_xlabel('Time [s]')

        ax.add_artist(line)
        fig.canvas.draw()

        # cache the background
        axBackground = fig.canvas.copy_from_bbox(ax.bbox)

        plt.show(block=False)

    #set initial condition if doing analytical solution
    if analytical:
        p_init = Expression('cSqrtConst*sin(pi*x[0]/(2*xLength) + pi/2)*sin(2*pi*x[1]/yLength + pi/2)',
                   degree=8, xLength=xLength, yLength=yLength, cSqrtConst=cSqrtConst)
        p_init_interp = interpolate(p_init, U.sub(0).collapse())
        q_init_interp = interpolate(Constant([0.0, 0.0]), U.sub(1).collapse())
        assign(u_m,[p_init_interp, q_init_interp])

        #expression for exact solution
        p_exact = Expression('cSqrtConst*sin(pi*x[0]/(2*xLength) + pi/2)*sin(2*pi*x[1]/yLength + pi/2)*cos(t*cSqrtConst)',
                            degree=8, t=0, xLength=xLength, yLength=yLength, cSqrtConst=cSqrtConst)

        H_init = assemble((0.5*p_m*p_m + 0.5*c_squared*inner(q_m, q_m))*dx)*rho_val

        p_e = interpolate(p_exact, U.sub(0).collapse())
        out_p, out_q = u_m.split(True)
        err_L2 = errornorm(p_exact, out_p, 'L2')
        # err_RMS = err_L2/np.sqrt(len(p_e.vector().get_local()))
        err_integral = assemble(((p_e - out_p)*(p_e - out_p))*dx)

        if rank == 0:
            print('Initial L2 Error = {}'.format(err_L2))
            print('Initial Int Error = {}'.format(err_integral))
    else:
        if controlType is 'passivity':
            p_init = Expression('sin(4*pi*x[0]/(xLength))',
                                degree=8, xLength=xLength)
            q_init = Expression(('sin(4*pi*x[0]/(xLength))', '0.0'),
                                degree=8, xLength=xLength)
            p_init_interp = interpolate(p_init, U.sub(0).collapse())
            q_init_interp = interpolate(q_init, U.sub(1).collapse())
            xode_0_init_interp = interpolate(Constant(0.0), U.sub(2).collapse())
            xode_1_init_interp = interpolate(Constant(0.0), U.sub(3).collapse())
            xode_2_init_interp = interpolate(Constant(0.0), U.sub(4).collapse())
            assign(u_m,[p_init_interp, q_init_interp, xode_0_init_interp, xode_1_init_interp, xode_2_init_interp])
            H_init = assemble((0.5*p_m*p_m + 0.5*c_squared*inner(q_m, q_m))*dx)*rho_val
        else:
            H_init = 0

    # -------------------------------# Solve Loop #---------------------------------#

    surfPlot = False
    #em variables that are zero if not interconnection
    H_em = 0
    disp = 0

    boundaryEnergy = 0
    inputEnergy = 0
    # HLeft = 0
    # HGen = 0
    # HRight = 0
    # Time Stepping
    t = 0
    for n in range(numSteps):
        if rank == 0:
            set_log_level(LogLevel.ERROR)
        if numIntSubSteps == 1:
            t += dt_value
        else:
            # If doing a sub step increase time by partial amount according to time int scheme
            # for stormer Verlet we do the first half step before increasing the time
            if timeIntScheme == 'SV':
                pass
            else:
                # TODO(Finbar) Make sure this is correct for Explicit Heuns method
                t += dt_value

        if not interConnection:
            pLeftBoundary1.t = t
        else:
            uInput.t = t
            if controlType == 'passivity':
                if t >= control_start_t:
                    # we can get the value anywhere in the domain, as xode is a the same throughout the domain
                    uInput.h_1 = out_xode_0(0.01, 0.01)


        # set up 1st sub-step if the timeIntScheme has 2 sub-steps
        if numIntSubSteps > 1:
            # A_temp = assemble(a_temp)
            B_temp = assemble(b_temp)

            # Apply boundary conditions to matrices for first sub-step
            [bc.apply(A_temp, B_temp) for bc in bcs]

            # solve first sub-step
            solve(A_temp, u_temp.vector(), B_temp)

            # If multiple sub steps we must calculate the input power into the domain with the
            # half time step input, so we do it here before stepping the input (pLeftBoundary or uInput)
            # forward to the full time step step
            if interConnection:
                # energy contribution from EM boundary
                if timeIntScheme == 'SV':
                    inpPowerXdt = 0.5*(dt*uInput*xode_0_m/L_em)*ds(0)
            else:
                boundaryEnergy += assemble(pT_L_q_temp_left)

            # increase t to full time step
            if timeIntScheme == 'SV':
                t += dt_value
            else:
                #TODO(Finbar) make sure this is correct for EH as well
                pass

            if not interConnection:
                pLeftBoundary0.t = t
                pLeftBoundary1.t = t
            else:
                uInput.t = t
                # TODO include passivity part here as well

        # Assemble matrix for second step
        # The values from u_temp are now automatically in a and b for multi-sub-step methods
        # A = assemble(a)
        B = assemble(b)

        # Apply boundary conditions to matrices
        [bc.apply(A, B) for bc in bcs]

        solve(A, u_.vector(), B)


        # split solution
        if not interConnection:
            out_p, out_q = u_.split(True)
        else:
            if interConnection == 'IC':
                out_p, out_q, out_xode_0, out_xode_1, out_xode_2 = u_.split()

        # Save solution to file
        if saveP:
            xdmfFile_p.write(out_p, t)
#        if saveQ:
#            xdmfFile_q.write(out_q, t)

        # Update progress bar
        if rank == 0:
            set_log_level(LogLevel.PROGRESS)
            progress += 1

        # Calculate Hamiltonian and plot energy
        if not interConnection:
            if numIntSubSteps == 1:
                boundaryEnergy += assemble(pT_L_q)
            else:
                boundaryEnergy += assemble(pT_L_q_left)

            H_wave = assemble((0.5*p_*p_ + 0.5*c_squared*inner(q_, q_))*dx)*rho_val

            E = H_wave + boundaryEnergy - H_init
            H = H_wave
            if rank == 0:
                print('Hamiltonian = {}'.format(H))
                print('Energy Res = {}'.format(E))
        else:
            # energy contribution from EM boundary
            if timeIntScheme == 'SV':
                inpPowerXdt += 0.5*(dt*uInput*xode_0_/L_em)*ds(0)
            elif timeIntScheme == 'SE' or timeIntScheme == 'IE':
                inpPowerXdt = (dt*uInput*xode_0_/L_em)*ds(0)
            elif timeIntScheme == 'EE':
                inpPowerXdt = (dt*uInput*xode_0_m/L_em)*ds(0)
            elif timeIntScheme == 'SM':
                inpPowerXdt = 0.5*(dt*uInput*(xode_0_m + xode_0_)/L_em)*ds(0)
            else:
                print('timeIntScheme = {} is not implemented for interconnection'.format(timeIntScheme))
                quit()

            inputEnergy += assemble(inpPowerXdt)/yLength

            # get Hamiltonian of the wave domain
            # Multiply H_wave by rho to get units of energy
            H_wave = assemble((0.5*p_*p_ + 0.5*c_squared*inner(q_, q_))*dx)*rho_val

            # get Hamiltonian from electromechanical system
            if interConnection == 'IC':
                H_em = assemble(0.5*(xode_[0]*xode_[0]/L_em + xode_[1]*xode_[1]/m_em +
                                     K_em*xode_[2]*xode_[2])*ds(0))/yLength

            E = -inputEnergy + H_wave + H_em - H_init
            H = H_wave + H_em

            boundaryEnergy += assemble(pT_L_q)

            #output displacement
            if interConnection == 'IC':
                disp = assemble(xode_2_*ds(0))/yLength

            if rank == 0:
                print('disp = {}'.format(disp))
                print('H_em = {}'.format(H_em))
                print('H_wave = {}'.format(H_wave))
                print('Energy Residual = {}'.format(E))

        if analytical:
            # compare error
            p_exact.t = t
            p_e = interpolate(p_exact, U.sub(0).collapse())
            if t>0.099 and surfPlot and rank == 0:
                plt.close(fig)
                fig = plt.figure()
                c = plot(p_e - out_p, mode='color')
                # plot(mesh)
                plt.xlabel('x [m]')
                plt.ylabel('y [m]')
                plt.title('t = {:.4f}'.format(t))
                plt.colorbar(c, orientation='horizontal', label='error')
                plt.show()
                # surfPlot = False

            # choose random point to get p at to output and check difference
            xPoint = 0.155
            yPoint = 0.189
            if not parallel:
                p_point = out_p(xPoint, yPoint)
                p_point_exact = p_e(xPoint, yPoint)
            else:
                # This is temporary because I haven't yet enabled finding point values in parallel
                p_point = 0
                p_point_exact = 0

            err_L2 = errornorm(p_exact, out_p, 'L2')
            # err_RMS = err_L2/np.sqrt(len(p_e.vector().get_local()))
            err_max = np.abs(p_e.vector().get_local() - out_p.vector().get_local()).max()
            # err_integral = assemble(((p_e-out_p)*(p_e-out_p))*dx)

            if rank == 0:
                print('analytic L2 error = {}'.format(err_L2))
                print('analytic max error = {}'.format(err_max))
            if saveP:
                xdmfFile_p_exact.write(p_e, t)

        E_vec.append(E)
        H_vec.append(H)
        t_vec.append(t)
        disp_vec.append(disp)
        bEnergy_vec.append(boundaryEnergy)
        inpEnergy_vec.append(inputEnergy)
        H_em_vec.append(H_em)
        H_wave_vec.append(H_wave)
        if analytical:
            error_vec.append(err_L2)
            error_vec_max.append(err_max)
            p_point_vec.append(p_point)
            p_point_exact_vec.append(p_point_exact)

        # Update previous solution
        u_m.assign(u_)

        if rank == 0:
            # Plot Hamiltonian
            # set new data point
            if analytical:
                line.set_data(t_vec, error_vec)
            else:
                if controlType is None:
                    line.set_data(t_vec, E_vec)
                else:
                    line.set_data(t_vec, H_vec)
            # text.set_text('HI')
            # restore background
            fig.canvas.restore_region(axBackground)
            # Redraw the points
            ax.draw_artist(line)
            # ax.draw_artist(text)
            # Fill in the axes rectangle
            fig.canvas.blit(ax.bbox)
            # make sure all events have happened
            fig.canvas.flush_events()

    # Calculate total time
    totalTime = time.time() - tic
    if rank == 0:
        print('{}, {}, {} Simulation finished in {} seconds'.format(domainShape, timeIntScheme,
                                                                dirichletImp, totalTime))

    if analytical:
        H_array = np.zeros((numSteps, 12))
    else:
        H_array = np.zeros((numSteps, 8))

    H_array[:, 0] = np.array(t_vec)
    H_array[:, 1] = np.array(H_vec)
    H_array[:, 2] = np.array(E_vec)
    H_array[:, 3] = np.array(disp_vec)
    H_array[:, 4] = np.array(bEnergy_vec)
    H_array[:, 5] = np.array(inpEnergy_vec)
    H_array[:, 6] = np.array(H_em_vec)
    H_array[:, 7] = np.array(H_wave_vec)

    if analytical:
        #for analytic this is error against analytic solution, for nonanalytic it is equation residual
        H_array[:, 8] = np.array(error_vec)
        H_array[:, 9] = np.array(error_vec_max)
        H_array[:, 10] = np.array(p_point_vec)
        H_array[:, 11] = np.array(p_point_exact_vec)

    return H_array, numCells

