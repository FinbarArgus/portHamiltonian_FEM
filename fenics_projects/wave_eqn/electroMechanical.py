from fenics import *
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import mshr
import paperPlotSetup

def electromechanical_solve(tFinal, numSteps, outputDir,
                  nx, ny, xLength, yLength,
                  domainShape='R', timeIntScheme='SV', dirichletImp='weak'):


    """ function for solving the 2D wave equation in fenics

    The leftmost boundary is a forced dirichlet condition
    The rightmost boundary is a fixed zero dirichlet boundary
    The other boundaries are zero neumann boundaries

    :param tFinal: [float] final time of solve
    :param nx: [int] number of nodes in x direction
    :param ny: [int] number of nodes in y direction
    :param xL: [float] x length of domain
    :param yL: [float] y length of domain
    :param domainShape: [string] 'R' for rectangle or 'R_1C' for rectangle with 1 input channel
    :param dirichletImplementation: [string] how the dirichlet BCs are implemented either 'strong' or 'weak'
    :param timeIntScheme: [string] what time integration scheme to use. List of possible below
                        'EH' = Explicit Heuns, 'SE' = Symplectic Euler, 'SV' = Stormer Verlet
    :return:
    """

    # Find initial time
    tic = time.time()
    print('Started solving for case: {}, {}, {} '.format(domainShape, timeIntScheme, dirichletImp))

    # ------------------------------# Create Mesh #-------------------------------#
    if domainShape == 'R':
        xInputStart = 0.0  # at what x location is the input boundary
        mesh = RectangleMesh(Point(xLength, 0.0), Point(0.0, yLength), nx, ny)
    else:
        print('domainShape \"{}\" hasnt been created'.format(domainShape))
        quit()

    # time and time step
    dt_value = tFinal/numSteps
    dt = Constant(dt_value)

    # get order of time integration scheme
    numIntSubSteps = 1
    if timeIntScheme in ['EH', 'SV']:
        numIntSubSteps = 2
    # Constant wave speed

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

    # Create function spaces

    # ensure weak boundary conditions for interconnection
    if not dirichletImp == 'weak':
        print('can only do interconnection model with weak boundary conditions')
        quit()

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
    xode_ = as_vector([xode_0_, xode_1_, xode_2_])

    u_temp = Function(U)
    xode_0_temp, xode_1_temp, xode_2_temp = split(u_temp)
    xode_temp = as_vector([xode_0_temp, xode_1_temp, xode_2_temp])

    # -------------------------------# Boundary Conditions #---------------------------------#

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
        def inside(self, x, on_boundary):
            return on_boundary and (near(x[1], 0) or near(x[1], yLength))


    # Initialise mesh function for neumann boundary. Facets are dim()-1, initialise subdomains to index 0
    boundaryDomains = MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
    boundaryDomains.set_all(0)

    leftMark= LeftMarker()
    leftMark.mark(boundaryDomains, 0)

    # redefine ds so that ds[0] is the dirichlet boundaries and ds[1] are the neumann boundaries
    ds = Measure('ds', domain=mesh, subdomain_data=boundaryDomains)

    # Forced input
    uInput = Expression('sin(3.14159*t)', degree=2, t=0)

    bcs = []

    # -------------------------------# Problem Definition #---------------------------------#

    if timeIntScheme == 'SE':
        #implement Symplectic Euler
        xode_int = as_vector([xode[0], xode[1], xode_n[2]])
        # xode_int = xode
        F_em = (-dot(v_xode, xode - xode_n)/dt + dot(v_xode, A_em*xode_int) + \
             v_xode[0]*uInput )*ds(0)
        F = F_em

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


    # -------------------------------# Set up output and plotting #---------------------------------#

    # Create progress bar
    progress = Progress('Time-stepping', numSteps)

    # Create vector for hamiltonian
    E_vec = [0]
    H_vec = [0]
    # HLeft_vec= [0]
    # HGen_vec= [0]
    # HRight_vec= [0]
    t_vec = [0]
    disp_vec = [0]

    # Create plot for hamiltonian
    fig, ax = plt.subplots(1, 1)


    # create line object for plotting
    line, = ax.plot([], lw=0.5, color='k', linestyle='-',
                    label='Hamiltonian {}, {}, {}'.format(domainShape,timeIntScheme,dirichletImp))

    # text = ax.text(0.001, 0.001, "")
    ax.set_xlim(0, tFinal)
    # ax.set_ylim(0, 600)
    ax.set_ylim(0, 15)

    ax.legend()
    ax.set_ylabel('Hamiltonian [J]')
    ax.set_xlabel('Time [s]')

    ax.add_artist(line)
    fig.canvas.draw()

    # cache the background
    axBackground = fig.canvas.copy_from_bbox(ax.bbox)

    plt.show(block=False)

    # -------------------------------# Solve Loop #---------------------------------#

    boundaryEnergy = 0
    # HLeft = 0
    # HGen = 0
    # HRight = 0
    # Time Stepping
    t = 0
    for n in range(numSteps):
        set_log_level(LogLevel.ERROR)
        t += dt_value

        uInput.t = t


        # set up 1st sub-step if the timeIntScheme has 2 sub-steps
        if numIntSubSteps > 1:
            A_temp = assemble(a_temp)
            B_temp = assemble(L_temp)

            # Apply boundary conditions to matrices for first sub-step
            [bc.apply(A_temp, B_temp) for bc in bcs]

            # solve first sub-step
            solve(A_temp, u_temp.vector(), B_temp)

        # Assemble matrix for second step
        # The values from u_temp are now automatically in a and L for multi-sub-step methods
        # A = assemble(a)
        B = assemble(L)

        # Apply boundary conditions to matrices
        # [bc.apply(A, B) for bc in bcs]

        solve(A, u_.vector(), B)

        # split solution
        out_xode_0, out_xode_1, out_xode_2 = u_.split()
        # out_xode = as_vector([out_xode_0, out_xode_1, out_xode_2])
        #split previous solution
        # out_p_n, out_q_n, out_xode_0_n, out_xode_1_n, out_xode_2_n = u_n.split()

        # Update progress bar
        set_log_level(LogLevel.PROGRESS)
        progress += 1

        # Calculate Hamiltonian and plot energy
        inpPowerXdt = dt*uInput*xode_0_*ds(0)
        boundaryEnergy += assemble(inpPowerXdt)/yLength
        H_em = assemble(0.5*(xode_[0]*xode_[0]/L_em + xode_[1]*xode_[1]/m_em + K_em*xode_[2]*xode_[2])*ds(0))/yLength
        print('boundary energy = {}'.format(boundaryEnergy))
        print('Hamiltonian     = {}'.format(H_em))
        E = -boundaryEnergy + H_em # Multiply H_wave by rho to get units of energy
        H = H_em

        #output displacement
        disp = assemble(xode_2_*ds(0))/yLength

        E_vec.append(E)
        H_vec.append(H)
        # HLeft_vec.append(HLeft)
        # HGen_vec.append(HGen)
        # HRight_vec.append(HRight)
        t_vec.append(t)
        disp_vec.append(disp)

        # Update previous solution
        u_n.assign(u_)

        # Plot Hamiltonian
        # set new data point
        line.set_data(t_vec, H_vec)
        # text.set_text('HI')
        # line.set_data(0.1, 1)
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

    return H_vec, E_vec, t_vec, disp_vec

