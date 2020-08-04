from fenics import *
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import mshr

def wave_2D_solve(tFinal, numSteps, outputDir,
                  nx, xLength, yLength,
                  domainShape='R', timeIntScheme='SV', dirichletImp='weak',
                  K_wave=1, rho=1, BOUNDARYPLOT=False, interConnection=None,
                  analytical=False, saveP=True):

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
                            either IC for 3 dof or IC4 for 4 dof, or None for non interconnection (just wave equation)
    :param saveP [bool] whether to save p throughout the domain as an xdf file for paraview postprocessing
    :return:
            H_array: array of output vectors of length numSteps
            numCells: number of cells in mesh
    """

    BOUNDARYPLOT = False
    # Find initial time
    tic = time.time()
    print('Started solving for case: {}, {}, {} '.format(domainShape, timeIntScheme, dirichletImp))

    # ------------------------------# Create Mesh #-------------------------------#
    if domainShape == 'R':
        xInputStart = 0.0  # at what x location is the input boundary
        mainRectangle = mshr.Rectangle(Point(0.0, 0.0), Point(xLength, yLength))
        mesh = mshr.generate_mesh(mainRectangle, nx)
    elif domainShape == 'S_1C':

        xInputStart = -0.10
        inputRectanglePoint1 = Point(xInputStart, xLength/2 - yLength/2)
        inputRectanglePoint2 = Point(0.0, xLength/2 + yLength/2)
        mainRectangle = mshr.Rectangle(Point(0.0, 0.0), Point(xLength, xLength))
        inputRectangle = mshr.Rectangle(inputRectanglePoint1, inputRectanglePoint2)
        domain = mainRectangle + inputRectangle
        mesh = mshr.generate_mesh(domain, nx)

        # Mark cells for refinement
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

        # plt.figure()
        # plot(mesh)
        # plt.show()
    else:
        print('domainShape \"{}\" hasnt been created'.format(domainShape))
        quit()

    if analytical:
        if not domainShape == 'R':
            print('analytical solution for rectangle domain only')

    #get number of cells in mesh
    numCells = mesh.num_cells()
    print('number of cells = {}'.format(numCells))

    # time and time step
    dt_value = tFinal/numSteps
    dt = Constant(dt_value)

    # get order of time integration scheme
    numIntSubSteps = 1
    if timeIntScheme in ['EH', 'SV']:
        numIntSubSteps = 2
    # Constant wave speed
    c_squared = Constant(K_wave/rho)
    c = np.sqrt(K_wave/rho)
    K_wave = Constant(K_wave)
    rho_val = rho
    # rho = Constant(rho)

    if interConnection == 'IC4':
        if timeIntScheme not in ['SV', 'SE']:
            print('only SE and SV time int schemes implemented for interconnection model')
            quit()

        # Constants for the Electromechanical model
        Bl_em = 5
        R1_em = 0.0
        R2_em = 0.0
        K_em = 5
        m_em = 0.15
        L_em = 0.1
        C_em = 1.0

        #Create A matrix for the electromechanical part of the model
        J_em = np.array([[0, Bl_em, -1, 0], [-Bl_em, 0, 0, -1], [1, 0, 0, 0], [0, 1, 0, 0]])

        R_em = np.array([[R1_em, 0, 0, 0], [0, R2_em, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

        A_conv_em = np.array([[1/L_em, 0, 0, 0], [0, 1/m_em, 0, 0], [0, 0, 1/C_em, 0], [0, 0, 0, K_em]])

        A_em = Constant((J_em - R_em).dot(A_conv_em))
    elif interConnection == 'IC':
        if timeIntScheme not in ['SV', 'SE']:
            print('only SE and SV time int schemes implemented for interconnection model')
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

    if interConnection == 'IC4':
        # ensure weak boundary conditions for interconnection
        if not dirichletImp == 'weak':
            print('can only do interconnection model with weak boundary conditions')
            quit()

        # create function spaces for mesh and for Electromechanical ODE variables
        P1 = FiniteElement('P', triangle, 1)
        RT = FiniteElement('RT', triangle, 1)
        odeSpace = FiniteElement('R', triangle, 0) #order 0, 1 variable
        element = MixedElement([P1, RT, odeSpace, odeSpace, odeSpace, odeSpace])
        U = FunctionSpace(mesh, element)

        # Define trial and test functions
        p, q, xode_0, xode_1, xode_2, xode_3 = TrialFunctions(U)
        xode = as_vector([xode_0, xode_1, xode_2, xode_3])
        v_p, v_q, v_xode_0, v_xode_1, v_xode_2, v_xode_3 = TestFunctions(U)
        v_xode = as_vector([v_xode_0, v_xode_1, v_xode_2, v_xode_3])

        # Define functions for solutions at previous and current time steps
        u_n = Function(U)
        p_n, q_n, xode_0_n, xode_1_n, xode_2_n, xode_3_n = split(u_n)
        xode_n = as_vector([xode_0_n, xode_1_n, xode_2_n, xode_3_n])

        u_ = Function(U)
        p_, q_, xode_0_, xode_1_, xode_2_, xode_3_ = split(u_)
        xode_ = as_vector([xode_0_, xode_1_, xode_2_, xode_3_])

        u_temp = Function(U)
        p_temp, q_temp, xode_0_temp, xode_1_temp, xode_2_temp, xode_3_temp = split(u_temp)
        xode_temp = as_vector([xode_0_temp, xode_1_temp, xode_2_temp, xode_3_temp])

    elif interConnection == 'IC':
        # ensure weak boundary conditions for interconnection
        if not dirichletImp == 'weak':
            print('can only do interconnection model with weak boundary conditions')
            quit()

        # create function spaces for mesh and for Electromechanical ODE variables
        P1 = FiniteElement('P', triangle, 1)
        RT = FiniteElement('RT', triangle, 1)
        odeSpace = FiniteElement('R', triangle, 0)  # order 0, 1 variable
        element = MixedElement([P1, RT, odeSpace, odeSpace, odeSpace])
        U = FunctionSpace(mesh, element)

        # Define trial and test functions
        p, q, xode_0, xode_1, xode_2 = TrialFunctions(U)
        xode = as_vector([xode_0, xode_1, xode_2])
        v_p, v_q, v_xode_0, v_xode_1, v_xode_2 = TestFunctions(U)
        v_xode = as_vector([v_xode_0, v_xode_1, v_xode_2])

        # Define functions for solutions at previous and current time steps
        u_n = Function(U)
        p_n, q_n, xode_0_n, xode_1_n, xode_2_n = split(u_n)
        xode_n = as_vector([xode_0_n, xode_1_n, xode_2_n])

        u_ = Function(U)
        p_, q_, xode_0_, xode_1_, xode_2_ = split(u_)
        xode_ = as_vector([xode_0_, xode_1_, xode_2_])

        u_temp = Function(U)
        p_temp, q_temp, xode_0_temp, xode_1_temp, xode_2_temp = split(u_temp)
        xode_temp = as_vector([xode_0_temp, xode_1_temp, xode_2_temp])

    else:
        # interconnection is not used, this is the basic wave equation
        P1 = FiniteElement('P', triangle, 1)
        RT = FiniteElement('RT', triangle, 1)
        element = MixedElement([P1, RT])
        U = FunctionSpace(mesh, element)
        #Also create function space for just P1

        # Define trial and test functions
        p, q = TrialFunctions(U)
        v_p, v_q = TestFunctions(U)

        # Define functions for solutions at previous and current time steps
        u_n = Function(U)
        p_n, q_n = split(u_n)

        u_ = Function(U)
        p_, q_ = split(u_)

        u_temp= Function(U)
        p_temp, q_temp= split(u_temp)


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

    # all other boundaries for marking
    class GeneralMarker(SubDomain):
        if domainShape == 'R':
            def inside(self, x, on_boundary):
                return on_boundary and (near(x[1], 0) or near(x[1], yLength))

        elif domainShape == 'S_1C':
            def inside(self, x, on_boundary):
                return on_boundary and (near(x[1], 0) or near(x[1], xLength) or
                                        near(x[1], xLength/2 - yLength/2) or
                                        near(x[1], xLength/2 + yLength/2) or
                                        near(x[0], 0))

    # Initialise mesh function for neumann boundary. Facets are dim()-1, initialise subdomains to index 0
    boundaryDomains = MeshFunction('size_t', mesh, mesh.topology().dim()-1)
    boundaryDomains.set_all(3)

    leftMark= LeftMarker()
    leftMark.mark(boundaryDomains, 0)

    generalMark= GeneralMarker()
    generalMark.mark(boundaryDomains, 1)

    rightMark= RightMarker()
    rightMark.mark(boundaryDomains, 2)

    # redefine ds so that ds(0) is left boundary, ds(1) is general and ds(2) is right boundary
    ds = Measure('ds', domain=mesh, subdomain_data=boundaryDomains)

    if BOUNDARYPLOT:
        # output mesh to file if BOUNDARYPLOT is True
        File(os.path.join(outputDir,'markers.pvd')) << boundaryDomains
        quit()

    # Get normal
    n = FacetNormal(mesh)

    # Make an expression for the boundary term for top and bottom edges
    q_bNormal = Constant(0.0)
    # Forced boundary
    if not interConnection:
        if analytical:
            cSqrtConst = c*np.sqrt(pi**2/(4*xLength**2) + 4*pi**2/yLength**2)
            pLeftBoundary = Expression('cSqrtConst*sin(2*pi*x[1]/(yLength) + pi/2)*cos(t*cSqrtConst)',
                       degree=8, t=0, yLength=yLength, cSqrtConst=cSqrtConst)
        else:
            pLeftBoundary = Expression('(t<0.25) ? 5*sin(8*pi*t) : 0.0', degree=2, t=0)
        pLeftBoundary_temp = pLeftBoundary
        pLeftBoundary_temp_ = pLeftBoundary
        pLeftBoundary_ = pLeftBoundary
    else:
        if timeIntScheme == 'SV':
            pLeftBoundary_temp = xode_n[1]/(m_em)
            pLeftBoundary = xode[1]/(m_em)
            pLeftBoundary_temp_ = xode_n[1]/(m_em)
            pLeftBoundary_ = xode_[1]/(m_em)
        elif timeIntScheme == 'SE':
            pLeftBoundary = xode[1]/(m_em)
            pLeftBoundary_ = xode_[1]/(m_em)
        else:
            print('only SV and SE time int schemes have been implemented for interconnection')
            quit()

        if domainShape == 'R':
            uInput = Expression('(t<0.25) ? 10*sin(8*pi*t) : 0.0', degree=2, t=0)
        elif domainShape == 'S_1C':
            uInput = Expression('(t<0.25) ? 10*sin(8*pi*t) : 0.0', degree=2, t=0)

    mirrorBoundary = Constant(0.0)

    # If we have dirichlet conditions applied in the strong way set them here, otherwise
    # they are set in the integration by parts term in the problem definition
    if dirichletImp == 'strong':
        bc_forced = DirichletBC(U.sub(0), pLeftBoundary, forced_boundary)

        # fixed boundary at right boundary
        bc_fixed = DirichletBC(U.sub(0), Constant(0.0), fixed_boundary)

        # List of boundary condition objects
        bcs = [bc_forced, bc_fixed]
    else:
        bcs = []


    # -------------------------------# Problem Definition #---------------------------------#

    # Define variational problem
    if timeIntScheme == 'SV':
    # Implement Stormer Verlet Scheme
        if dirichletImp == 'strong':
            F_temp = (p - p_n)*v_p*dx + 0.5*dt*c_squared*(-dot(q_n, grad(v_p))*dx + q_bNormal*v_p*ds(1)) + \
                     dot(q - q_n, v_q)*dx + 0.5*dt*dot(grad(p_n), v_q)*dx
            F = (p - p_n)*v_p*dx + dt*c_squared*(-dot(q_temp, grad(v_p))*dx + q_bNormal*v_p*ds(1)) + \
                dot(q - q_temp, v_q)*dx + 0.5*dt*dot(grad(p), v_q)*dx
        elif dirichletImp == 'weak':
            #TODO(only the q part of this temp equation (the first step of stormer-verlet
            # needs to be solved, for now just solve it all
            if not interConnection:
                F_temp = (p - p_n)*v_p*dx + 0.5*dt*c_squared*(-dot(q_n, grad(v_p))*dx + q_bNormal*v_p*ds(1) +
                                                              dot(q_n, n)*v_p*ds(0) + dot(q_n, n)*v_p*ds(2)) + \
                         dot(q - q_n, v_q)*dx + 0.5*dt*(-div(v_q)*p_n*dx + dot(v_q, n)*pLeftBoundary_temp*ds(0) +
                                                        dot(v_q, n)*mirrorBoundary*ds(2) + dot(v_q, n)*p_n*ds(1))
                F = (p - p_n)*v_p*dx + dt*c_squared*(-dot(q_temp, grad(v_p))*dx + q_bNormal*v_p*ds(1) +
                                                     dot(q_temp, n)*v_p*ds(0) + dot(q_temp, n)*v_p*ds(2)) + \
                    dot(q - q_temp, v_q)*dx + 0.5*dt*(-div(v_q)*p*dx + dot(v_q, n)*pLeftBoundary*ds(0) +
                                                      dot(v_q, n)*mirrorBoundary*ds(2) + dot(v_q, n)*p*ds(1))

            else:
                if interConnection == 'IC4':
                    F_temp_em = (-dot(v_xode, xode - xode_n)/(0.5*dt) + \
                              v_xode[0]*(A_em[0, 0]*xode_n[0] + A_em[0, 1]*xode_n[1] +
                                         A_em[0, 2]*xode[2] + A_em[0, 3]*xode[3]) + \
                              v_xode[1]*(A_em[1, 0]*xode_n[0] + A_em[1, 1]*xode_n[1] +
                                         A_em[1, 2]*xode[2] + A_em[1, 3]*xode[3]) + \
                              v_xode[2]*(A_em[2, 0]*xode_n[0] + A_em[2, 1]*xode_n[1] +
                                         A_em[2, 2]*xode[2] + A_em[2, 3]*xode[3]) + \
                              v_xode[3]*(A_em[3, 0]*xode_n[0] + A_em[3, 1]*xode_n[1] +
                                         A_em[3, 2]*xode[2] + A_em[3, 3]*xode[3]) + \
                              v_xode[0]*uInput + K_wave*yLength*v_xode[1]*dot(q_n,n))*ds(0) #I dont think it matters what q_ here

                    F_temp_wave = (p - p_n)*v_p*dx + 0.5*dt*c_squared*(-dot(q_n, grad(v_p))*dx + q_bNormal*v_p*ds(1) +
                                                                  dot(q_n, n)*v_p*ds(0) + dot(q_n, n)*v_p*ds(2)) + \
                             dot(q - q_n, v_q)*dx + 0.5*dt*(-div(v_q)*p_n*dx + dot(v_q, n)*pLeftBoundary_temp*ds(0) +
                                                            dot(v_q, n)*mirrorBoundary*ds(2) + dot(v_q, n)*p_n*ds(1))
                    F_temp = F_temp_em + F_temp_wave

                    F_em = (-(v_xode[0]*(xode[0] - xode_n[0]) + \
                           v_xode[1]*(xode[1] - xode_n[1]) + \
                           v_xode[2]*(xode[2] - xode_temp[2]) + \
                           v_xode[3]*(xode[3] - xode_temp[3]))/dt + \
                         0.5*v_xode[0]*(A_em[0, 0]*xode_n[0] + A_em[0, 1]*xode_n[1] +
                                        A_em[0, 2]*xode_temp[2] + A_em[0, 3]*xode_temp[3] + \
                                        A_em[0, 0]*xode[0] + A_em[0, 1]*xode[1] +
                                        A_em[0, 2]*xode_temp[2] + A_em[0, 3]*xode_temp[3]) + \
                         0.5*v_xode[1]*(A_em[1, 0]*xode_n[0] + A_em[1, 1]*xode_n[1] +
                                        A_em[1, 2]*xode_temp[2] + A_em[1, 3]*xode_temp[3] + \
                                        A_em[1, 0]*xode[0] + A_em[1, 1]*xode[1] +
                                        A_em[1, 2]*xode_temp[2] + A_em[1, 3]*xode_temp[3]) + \
                         0.5*v_xode[2]*(A_em[2, 0]*xode[0] + A_em[2, 1]*xode[1] +
                                        A_em[2, 2]*xode_temp[2] + A_em[2, 3]*xode_temp[3]) + \
                         0.5*v_xode[3]*(A_em[3, 0]*xode[0] + A_em[3, 1]*xode[1] +
                                        A_em[3, 2]*xode_temp[2] + A_em[3, 3]*xode_temp[3]) + \
                         v_xode[0]*uInput + K_wave*yLength*v_xode[1]*dot(q_temp,n))*ds(0)
                    # don't multiply the input by a half because it is for a full step

                    F_wave = (p - p_n)*v_p*dx + dt*c_squared*(-dot(q_temp, grad(v_p))*dx + q_bNormal*v_p*ds(1) +
                                                         dot(q_temp, n)*v_p*ds(0) + dot(q_temp, n)*v_p*ds(2)) + \
                        dot(q - q_temp, v_q)*dx + 0.5*dt*(-div(v_q)*p*dx + dot(v_q, n)*pLeftBoundary*ds(0) +
                                                          dot(v_q, n)*mirrorBoundary*ds(2) + dot(v_q, n)*p*ds(1))
                    F = F_em + F_wave

                else:
                    F_temp_em = (-dot(v_xode, xode - xode_n)/(0.5*dt) + \
                                 v_xode[0]*(A_em[0, 0]*xode_n[0] + A_em[0, 1]*xode_n[1] + A_em[0, 2]*xode[2]) + \
                                 v_xode[1]*(A_em[1, 0]*xode_n[0] + A_em[1, 1]*xode_n[1] + A_em[1, 2]*xode[2]) + \
                                 v_xode[2]*(A_em[2, 0]*xode_n[0] + A_em[2, 1]*xode_n[1] + A_em[2, 2]*xode[2]) + \
                                 v_xode[0]*uInput + K_wave*yLength*v_xode[1]*dot(q_n, n))*ds(0)  # I dont think it matters what q_ here

                    F_temp_wave = (p - p_n)*v_p*dx + 0.5*dt*c_squared*(-dot(q_n, grad(v_p))*dx + q_bNormal*v_p*ds(1) +
                                                                       dot(q_n, n)*v_p*ds(0) + dot(q_n, n)*v_p*ds(2)) + \
                                  dot(q - q_n, v_q)*dx + 0.5*dt*(
                                          -div(v_q)*p_n*dx + dot(v_q, n)*pLeftBoundary_temp*ds(0) +
                                          dot(v_q, n)*mirrorBoundary*ds(2) + dot(v_q, n)*p_n*ds(1))
                    F_temp = F_temp_em + F_temp_wave

                    F_em = (-(v_xode[0]*(xode[0] - xode_n[0]) + \
                              v_xode[1]*(xode[1] - xode_n[1]) + \
                              v_xode[2]*(xode[2] - xode_temp[2]))/dt + \
                            0.5*v_xode[0]*(A_em[0, 0]*xode_n[0] + A_em[0, 1]*xode_n[1] + A_em[0, 2]*xode_temp[2] + \
                                           A_em[0, 0]*xode[0] + A_em[0, 1]*xode[1] + A_em[0, 2]*xode_temp[2]) + \
                            0.5*v_xode[1]*(A_em[1, 0]*xode_n[0] + A_em[1, 1]*xode_n[1] + A_em[1, 2]*xode_temp[2] + \
                                           A_em[1, 0]*xode[0] + A_em[1, 1]*xode[1] + A_em[1, 2]*xode_temp[2]) + \
                            0.5*v_xode[2]*(A_em[2, 0]*xode[0] + A_em[2, 1]*xode[1] + A_em[2, 2]*xode_temp[2]) + \
                            v_xode[0]*uInput + K_wave*yLength*v_xode[1]*dot(q_temp, n))*ds(0)
                    # don't multiply the input by a half because it is for a full step

                    F_wave = (p - p_n)*v_p*dx + dt*c_squared*(-dot(q_temp, grad(v_p))*dx + q_bNormal*v_p*ds(1) +
                                                              dot(q_temp, n)*v_p*ds(0) + dot(q_temp, n)*v_p*ds(2)) + \
                             dot(q - q_temp, v_q)*dx + 0.5*dt*(-div(v_q)*p*dx + dot(v_q, n)*pLeftBoundary*ds(0) +
                                                               dot(v_q, n)*mirrorBoundary*ds(2) + dot(v_q, n)*p*ds(1))
                    F = F_em + F_wave

        else:
            print('dirichlet implementation {} not implemented'.format(dirichletImp))
            exit()

    elif timeIntScheme == 'EE':
    # implement Explicit Euler
        if dirichletImp == 'strong':
            F = (p - p_n)*v_p*dx + dt*c_squared*(-dot(q_n, grad(v_p))*dx + q_bNormal*v_p*ds(1)) + \
                dot(q - q_n, v_q)*dx + dt*dot(grad(p_n), v_q)*dx
        elif dirichletImp == 'weak':
            F = (p - p_n)*v_p*dx + dt*c_squared*(-dot(q_n, grad(v_p))*dx + q_bNormal*v_p*ds(1) + \
                                                 dot(q_n, n)*v_p*ds(0) + dot(q_n, n)*v_p*ds(2)) + \
                dot(q - q_n, v_q)*dx - dt*div(v_q)*p_n*dx + dt*dot(v_q, n)*pLeftBoundary*ds(0) + \
                dt*dot(v_q, n)*mirrorBoundary*ds(2) + dt*dot(v_q, n)*p_n*ds(1)
        else:
            print('dirichlet implementation {} not implemented'.format(dirichletImp))
            exit()
    elif timeIntScheme == 'IE':
    # implement Implicit Euler
        if dirichletImp == 'strong':
            F = (p - p_n)*v_p*dx + dt*c_squared*(-dot(q, grad(v_p))*dx + q_bNormal*v_p*ds(1)) + \
                dot(q - q_n, v_q)*dx + dt*dot(grad(p), v_q)*dx
        elif dirichletImp == 'weak':
            F = (p - p_n)*v_p*dx + dt*c_squared*(-dot(q, grad(v_p))*dx + q_bNormal*v_p*ds(1) + \
                                                 dot(q, n)*v_p*ds(0) + dot(q, n)*v_p*ds(2)) + \
                dot(q - q_n, v_q)*dx - dt*div(v_q)*p*dx + dt*dot(v_q, n)*pLeftBoundary*ds(0) + \
                dt*dot(v_q, n)*mirrorBoundary*ds(2) + dt*dot(v_q, n)*p*ds(1)
        else:
            print('dirichlet implementation {} not implemented'.format(dirichletImp))
            exit()
    elif timeIntScheme == 'SE':
    #implement Symplectic Euler
        if dirichletImp == 'strong':
            F = (p - p_n)*v_p*dx + dt*c_squared*(-dot(q_n, grad(v_p))*dx + q_bNormal*v_p*ds(1)) + \
            dot(q - q_n, v_q)*dx + dt*dot(grad(p), v_q)*dx
        elif dirichletImp == 'weak':
            if not interConnection:
                F = (p - p_n)*v_p*dx + dt*c_squared*(-dot(q_n, grad(v_p))*dx + q_bNormal*v_p*ds(1) + \
                    dot(q_n, n)*v_p*ds(0) + dot(q_n, n)*v_p*ds(2)) + \
                    dot(q - q_n, v_q)*dx - dt*div(v_q)*p*dx + dt*dot(v_q, n)*pLeftBoundary*ds(0) + \
                    dt*dot(v_q, n)*mirrorBoundary*ds(2) + dt*dot(v_q, n)*p*ds(1)
            else:
                if interConnection == 'IC4':
                    F_em = (-dot(v_xode, xode - xode_n)/dt + \
                                 v_xode[0]*(A_em[0, 0]*xode[0] + A_em[0, 1]*xode[1] +
                                            A_em[0, 2]*xode_n[2] + A_em[0, 3]*xode_n[3]) + \
                                 v_xode[1]*(A_em[1, 0]*xode[0] + A_em[1, 1]*xode[1] +
                                            A_em[1, 2]*xode_n[2] + A_em[1, 3]*xode_n[3]) + \
                                 v_xode[2]*(A_em[2, 0]*xode[0] + A_em[2, 1]*xode[1] + \
                                            A_em[2, 2]*xode_n[2] + A_em[2, 3]*xode_n[3]) + \
                                 v_xode[3]*(A_em[3, 0]*xode[0] + A_em[3, 1]*xode[1] +
                                            A_em[3, 2]*xode_n[2] + A_em[3, 3]*xode_n[3]) + \
                            v_xode[0]*uInput + K_wave*yLength*v_xode[1]*dot(q_n,n))*ds(0)

                    F_wave = (p - p_n)*v_p*dx + dt*c_squared*(-dot(q_n, grad(v_p))*dx + q_bNormal*v_p*ds(1) +
                                                              dot(q_n, n)*v_p*ds(0) + dot(q_n, n)*v_p*ds(2)) + \
                              dot(q - q_n, v_q)*dx + dt*(-div(v_q)*p*dx + dot(v_q, n)*pLeftBoundary*ds(0) +
                                                             dot(v_q, n)*mirrorBoundary*ds(2) + dot(v_q, n)*p*ds(1))
                    F = F_em + F_wave
                else:
                    F_em = (-dot(v_xode, xode - xode_n)/dt + \
                            v_xode[0]*(A_em[0, 0]*xode[0] + A_em[0, 1]*xode[1] + A_em[0, 2]*xode_n[2]) + \
                            v_xode[1]*(A_em[1, 0]*xode[0] + A_em[1, 1]*xode[1] + A_em[1, 2]*xode_n[2]) + \
                            v_xode[2]*(A_em[2, 0]*xode[0] + A_em[2, 1]*xode[1] + A_em[2, 2]*xode_n[2]) + \
                            v_xode[0]*uInput + K_wave*yLength*v_xode[1]*dot(q_n, n))*ds(
                        0)  # I dont think it matters what q_ here

                    F_wave = (p - p_n)*v_p*dx + dt*c_squared*(-dot(q_n, grad(v_p))*dx + q_bNormal*v_p*ds(1) +
                                                              dot(q_n, n)*v_p*ds(0) + dot(q_n, n)*v_p*ds(2)) + \
                             dot(q - q_n, v_q)*dx + dt*(-div(v_q)*p*dx + dot(v_q, n)*pLeftBoundary*ds(0) +
                                                        dot(v_q, n)*mirrorBoundary*ds(2) + dot(v_q, n)*p*ds(1))
                    F = F_em + F_wave

        else:
            print('dirichlet implementation {} not implemented'.format(dirichletImp))
            exit()

    elif timeIntScheme == 'EH':
    # implement Explicit Heuns
        if dirichletImp == 'strong':
            F_temp = (p - p_n)*v_p*dx + dt*c_squared*(-dot(q_n, grad(v_p))*dx + q_bNormal*v_p*ds(1)) + \
                     dot(q - q_n, v_q)*dx + dt*dot(grad(p_n), v_q)*dx
            F = (p - p_n)*v_p*dx + 0.5*(dt*c_squared*(-dot(q_n, grad(v_p))*dx + q_bNormal*v_p*ds(1)) + \
                                        dt*c_squared*(-dot(q_temp, grad(v_p))*dx + dt*q_bNormal*v_p*ds(1))) + \
                 dot(q - q_n, v_q)*dx + 0.5*(dt*dot(grad(p_n), v_q)*dx + \
                                            dt*dot(grad(p_temp), v_q)*dx)

        elif dirichletImp == 'weak':
            F_temp = (p - p_n)*v_p*dx + dt*c_squared*(-dot(q_n, grad(v_p))*dx + q_bNormal*v_p*ds(1) + \
                                                      dot(q_n, n)*v_p*ds(0) + dot(q_n, n)*v_p*ds(2)) + \
                      dot(q - q_n, v_q)*dx + dt*(-div(v_q)*p_n*dx + dot(v_q, n)*pLeftBoundary_temp*ds(0) + \
                                                dot(v_q, n)*mirrorBoundary*ds(2) + dot(v_q, n)*p_n*ds(1))
            F = (p - p_n)*v_p*dx + 0.5*dt*c_squared*(-dot(q_n, grad(v_p))*dx + q_bNormal*v_p*ds(1) + \
                                                      dot(q_n, n)*v_p*ds(0) + dot(q_n, n)*v_p*ds(2) + \
                                                      -dot(q_temp, grad(v_p))*dx + q_bNormal*v_p*ds(1) + \
                                                      dot(q_temp, n)*v_p*ds(0) + dot(q_temp, n)*v_p*ds(2)) + \
                 dot(q - q_n, v_q)*dx + 0.5*dt*(-div(v_q)*p_n*dx + dot(v_q, n)*pLeftBoundary*ds(0) + \
                                                dot(v_q, n)*mirrorBoundary*ds(2) + dot(v_q, n)*p_n*ds(1) + \
                                                -div(v_q)*p_temp*dx + dot(v_q, n)*pLeftBoundary*ds(0) + \
                                                dot(v_q, n)*mirrorBoundary*ds(2) + dot(v_q, n)*p_temp*ds(1))

        else:
            print('dirichlet implementation {} not implemented'.format(dirichletImp))
            exit()
    else:
        print('time integration scheme {} not implemented'.format(timeIntScheme))
        exit()


    a = lhs(F)
    L = rhs(F)
    if numIntSubSteps > 1:
        a_temp = lhs(F_temp)
        L_temp = rhs(F_temp)

    # Assemble matrices
    A = assemble(a)
    B = assemble(L)
    if numIntSubSteps > 1:
        A_temp = assemble(a_temp)
        B_temp = assemble(L_temp)

    # Apply boundary conditions to matrices
    [bc.apply(A, B) for bc in bcs]
    if numIntSubSteps > 1:
        [bc.apply(A_temp, B_temp) for bc in bcs]

    # Create the L_p matrix from notes to calculate the boundary energy flow
    # first i define what time step q and p variables are used in each substep in
    # order to calculate the energy flow through the boundaries
    if timeIntScheme == 'SV':
        q0 = q_temp
        q1 = q_temp
        p0 = p_n
        p1 = p_
    elif timeIntScheme == 'EE':
        q0 = q_n
        q1 = q_n
        p0 = p_n
        p1 = p_n
    elif timeIntScheme == 'IE':
        q0 = q_
        q1 = q_
        p0 = p_
        p1 = p_
    elif timeIntScheme == 'SE':
        q0 = q_n
        q1 = q_n
        p0 = p_
        p1 = p_
    elif timeIntScheme == 'EH':
        q0 = q_n
        q1 = q_temp
        p0 = p_n
        p1 = p_temp

    if numIntSubSteps >1:
        # These are the energy contributions from the left, general and right boundaries for the first half timestep
        pT_L_q_temp_left = 0.5*dt*K_wave*dot(q0, n)*pLeftBoundary_temp_*ds(0)

        # These are the energy contributions from the left, general and right boundaries for the second half timestep
        pT_L_q_left = 0.5*dt*K_wave*dot(q1, n)*pLeftBoundary_*ds(0)

        # Total energy contribution from the boundaries for the time step
        pT_L_q = pT_L_q_temp_left + pT_L_q_left  # + pT_L_q_gen + pT_L_q_right

    else:
        pT_L_q_left = dt*K_wave*dot(q1, n)*pLeftBoundary_*ds(0)
        # pT_L_q_gen = dt*K_wave*q_bNormal*p1*ds(1)
        # pT_L_q_right = dt*K_wave*dot(q1, n)*mirrorBoundary*ds(2)

        pT_L_q = pT_L_q_left # + pT_L_q_gen + pT_L_q_right

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

    # Create vector for hamiltonian
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
        error_vec_int = []
        p_point_vec = []
        p_point_exact_vec = []

    # Create plot for Energy Residual or analytic error
    fig, ax = plt.subplots(1, 1)

    # create line object for plotting
    if analytical:
        line, = ax.plot([], lw=0.5, color='k', linestyle='-',
                    label='analytical error {}, {}, {}'.format(domainShape,timeIntScheme,dirichletImp))
    else:
        line, = ax.plot([], lw=0.5, color='k', linestyle='-',
                        label='Energy Residual {}, {}, {}'.format(domainShape,timeIntScheme,dirichletImp))

    # text = ax.text(0.001, 0.001, "")
    ax.set_xlim(0, tFinal)
    if domainShape == 'R':
        if analytical:
            ax.set_ylim(0, 0.01)
        else:
            ax.set_ylim(-0.03, 0.05)
            if dirichletImp == 'weak' and timeIntScheme in ['SV', 'SE', 'EH']:
                ax.set_ylim(-0.0005, 0.0008)

    elif domainShape == 'S_1C':
        ax.set_ylim(-0.001, 0.001)

    ax.legend()
    if analytical:
        ax.set_ylabel('analytical error')
    else:
        ax.set_ylabel('Energy Residual [J]')
    ax.set_xlabel('Time [s]')

    ax.add_artist(line)
    fig.canvas.draw()

    # cache the background
    axBackground = fig.canvas.copy_from_bbox(ax.bbox)

    plt.show(block=False)

    # -------------------------------# Solve Loop #---------------------------------#

    #set initial condition if doing analytical solution
    if analytical:
        p_init = Expression('cSqrtConst*sin(pi*x[0]/(2*xLength) + pi/2)*sin(2*pi*x[1]/yLength + pi/2)',
                   degree=8, xLength=xLength, yLength=yLength, cSqrtConst=cSqrtConst)
        p_init_interp = interpolate(p_init, U.sub(0).collapse())
        q_init_interp = interpolate(Constant([0.0, 0.0]), U.sub(1).collapse())
        assign(u_n,[p_init_interp, q_init_interp])

        #expression for exact solution
        p_exact = Expression('cSqrtConst*sin(pi*x[0]/(2*xLength) + pi/2)*sin(2*pi*x[1]/yLength + pi/2)*cos(t*cSqrtConst)',
                            degree=8, t=0, xLength=xLength, yLength=yLength, cSqrtConst=cSqrtConst)

        H_init = assemble((0.5*p_n*p_n + 0.5*c_squared*inner(q_n, q_n))*dx)*rho_val

        p_e = interpolate(p_exact, U.sub(0).collapse())
        out_p, out_q = u_n.split(True)
        err_L2 = errornorm(p_exact, out_p, 'L2')
        err_RMS = err_L2/np.sqrt(len(p_e.vector().get_local()))
        err_integral = assemble(((p_e - out_p)*(p_e - out_p))*dx)

        print('Initial RMS Error = {}'.format(err_RMS))
        print('Initial Int Error = {}'.format(err_integral))
    else:
        H_init = 0


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
            pLeftBoundary.t = t
        else:
            uInput.t = t


        # set up 1st sub-step if the timeIntScheme has 2 sub-steps
        if numIntSubSteps > 1:
            # A_temp = assemble(a_temp)
            B_temp = assemble(L_temp)

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
                    inpPowerXdt = 0.5*(dt*uInput*xode_0_n/L_em)*ds(0)
            else:
                boundaryEnergy += assemble(pT_L_q_temp_left)

            # increase t to full time step
            if timeIntScheme == 'SV':
                t += dt_value
            else:
                #TODO(Finbar) make sure this is correct for EH as well
                pass

            if not interConnection:
                pLeftBoundary.t = t
            else:
                uInput.t = t

        # Assemble matrix for second step
        # The values from u_temp are now automatically in a and L for multi-sub-step methods
        # A = assemble(a)
        B = assemble(L)

        # Apply boundary conditions to matrices
        [bc.apply(A, B) for bc in bcs]

        solve(A, u_.vector(), B)


        # split solution
        if not interConnection:
            out_p, out_q = u_.split(True)
        else:
            if interConnection == 'IC4':
                out_p, out_q, out_xode_0, out_xode_1, out_xode_2, out_xode_3 = u_.split()
            elif interConnection == 'IC':
                out_p, out_q, out_xode_0, out_xode_1, out_xode_2 = u_.split()

        # Save solution to file
        if saveP:
            xdmfFile_p.write(out_p, t)
#        if saveQ:
#            xdmfFile_q.write(out_q, t)

        # Update progress bar
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
            print('Hamiltonian = {}'.format(H))
            print('Energy Res = {}'.format(E))
        else:
            # energy contribution from EM boundary
            if timeIntScheme == 'SV':
                inpPowerXdt += 0.5*(dt*uInput*xode_0_/L_em)*ds(0)
            elif timeIntScheme == 'SE' or 'IE':
                inpPowerXdt = (dt*uInput*xode_0_/L_em)*ds(0)
            elif timeIntScheme == 'EE':
                inpPowerXdt = (dt*uInput*xode_0_n/L_em)*ds(0)
            else:
                print('timeIntScheme = {} is not implemented for interconnection'.format(timeIntScheme))

            inputEnergy += assemble(inpPowerXdt)/yLength

            # get Hamiltonian of the wave domain
            # Multiply H_wave by rho to get units of energy
            H_wave = assemble((0.5*p_*p_ + 0.5*c_squared*inner(q_, q_))*dx)*rho_val

            # get Hamiltonian from electromechanical system
            if interConnection == 'IC4':
                H_em = assemble(0.5*(xode_[0]*xode_[0]/L_em + xode_[1]*xode_[1]/m_em +
                                     xode_[2]*xode_[2]/C_em + K_em*xode_[3]*xode_[3])*ds(0))/yLength
            elif interConnection == 'IC':
                H_em = assemble(0.5*(xode_[0]*xode_[0]/L_em + xode_[1]*xode_[1]/m_em +
                                     K_em*xode_[2]*xode_[2])*ds(0))/yLength

            E = -inputEnergy + H_wave + H_em
            H = H_wave + H_em

            boundaryEnergy += assemble(pT_L_q)

            #output displacement
            if interConnection == 'IC4':
                disp = assemble(xode_3_*ds(0))/yLength
            elif interConnection == 'IC':
                disp = assemble(xode_2_*ds(0))/yLength

            print('disp = {}'.format(disp))
            print('H_em = {}'.format(H_em))
            print('H_wave = {}'.format(H_wave))
            print('Energy Residual = {}'.format(E))

        if analytical:
            # compare error
            p_exact.t = t
            p_e = interpolate(p_exact, U.sub(0).collapse())
            if t>0.099 and surfPlot:
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
            p_point = out_p(xPoint, yPoint)
            p_point_exact = p_e(xPoint, yPoint)

            err_L2 = errornorm(p_exact, out_p, 'L2')
            err_RMS = err_L2/np.sqrt(len(p_e.vector().get_local()))
            err_max = np.abs(p_e.vector().get_local() - out_p.vector().get_local()).max()
            err_integral = assemble(((p_e-out_p)*(p_e-out_p))*dx)

            print('analytic RMS error = {}'.format(err_RMS))
            print('analytic max error = {}'.format(err_max))
            print('analytic integral error = {}'.format(err_integral))
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
            error_vec.append(err_RMS)
            error_vec_int.append(err_integral)
            p_point_vec.append(p_point)
            p_point_exact_vec.append(p_point_exact)

        # Update previous solution
        u_n.assign(u_)

        # Plot Hamiltonian
        # set new data point
        if analytical:
            line.set_data(t_vec, error_vec_int)
        else:
            line.set_data(t_vec, E_vec)
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
        H_array[:, 9] = np.array(error_vec_int)
        H_array[:, 10] = np.array(p_point_vec)
        H_array[:, 11] = np.array(p_point_exact_vec)

    return H_array, numCells

