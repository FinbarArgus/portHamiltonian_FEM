from fenics import *
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import mshr
import paperPlotSetup

def wave_2D_solve(tFinal, numSteps, outputDir,
                  nx, ny, xLength, yLength,
                  domainShape='R', timeIntScheme='SV', dirichletImp='weak',
                  hookesK=1, rho=1, BOUNDARYPLOT=False):

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
    :param hookesK: [float] material stiffness
    :param rho: [float] material density
    :param BOUNDARYPLOT: [bool] whether the boundary conditions should be plotted for debugging
    :return:
    """

    # Find initial time
    tic = time.time()
    print('Started solving for case: {}, {}, {} '.format(domainShape, timeIntScheme, dirichletImp))

    # ------------------------------# Create Mesh #-------------------------------#
    if domainShape == 'R':
        xInputStart = 0.0  # at what x location is the input boundary
        mesh = RectangleMesh(Point(xLength, 0.0), Point(0.0, yLength), nx, ny)
    elif domainShape == 'R_1C':

        xInputStart = -0.10
        inputRectanglePoint1 = Point(xInputStart, 0.45)
        inputRectanglePoint2 = Point(0.0, 0.55)
        mainRectangle = mshr.Rectangle(Point(0.0, 0.0), Point(xLength, yLength))
        inputRectangle = mshr.Rectangle(inputRectanglePoint1, inputRectanglePoint2)
        domain = mainRectangle + inputRectangle
        mesh = mshr.generate_mesh(domain, 64)
    else:
        print('domainShape \"{}\" hasnt been created'.format(domainShape))
        quit()

    #get number of cells in mesh
    numCells = mesh.num_cells()

    # time and time step
    dt_value = tFinal/numSteps
    dt = Constant(dt_value)

    # get order of time integration scheme
    numIntSubSteps = 1
    if timeIntScheme in ['EH', 'SV']:
        numIntSubSteps = 2
    # Constant wave speed
    c_squared = hookesK/rho

    # ------------------------------# Create Function Spaces and functions#-------------------------------#

    # Create function spaces
    P1 = FiniteElement('P', triangle, 1)
    RT = FiniteElement('RT', triangle, 1)
    element = MixedElement([P1, RT])
    U = FunctionSpace(mesh, element)

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
        def inside(self, x, on_boundary):
            return on_boundary and (near(x[1], 0) or near(x[1], yLength))


    # Initialise mesh function for neumann boundary. Facets are dim()-1, initialise subdomains to index 0
    if BOUNDARYPLOT:
        from vtkplotter.dolfin import plot as vtkplot
        # TODO(Finbar) vtkplot only works when meshFunction dim is zero, this is only
        # TODO used to check the boundary values
        boundaryDomains = MeshFunction('size_t', mesh, 0)
    else:
        boundaryDomains = MeshFunction('size_t', mesh, mesh.topology().dim()-1)
    boundaryDomains.set_all(0)

    leftMark= LeftMarker()
    leftMark.mark(boundaryDomains, 0)

    generalMark= GeneralMarker()
    generalMark.mark(boundaryDomains, 1)

    rightMark= RightMarker()
    rightMark.mark(boundaryDomains, 2)

    # redefine ds so that ds[0] is the dirichlet boundaries and ds[1] are the neumann boundaries
    ds = Measure('ds', domain=mesh, subdomain_data=boundaryDomains)

    # Get normal
    n = FacetNormal(mesh)

    # Make an expression for the boundary term, for now we just make it equal to zero
    q_bNormal = Constant(0.0)
    # Forced boundary
    forceExpression = Expression('(t<0.25) ? sin(8*3.14159*t) : 0.0', degree=2, t=0)
    mirrorBoundary = Constant(0.0)

    # If we have dirichlet conditions applied in the strong way set them here, otherwise
    # they are set in the integration by parts term in the problem definition
    if dirichletImp == 'strong':
        bc_forced = DirichletBC(U.sub(0), forceExpression, forced_boundary)

        # fixed boundary at right boundary
        bc_fixed = DirichletBC(U.sub(0), Constant(0.0), fixed_boundary)

        # List of boundary condition objects
        bcs = [bc_forced, bc_fixed]
    else:
        bcs = []

    if BOUNDARYPLOT:
        vtkplot(boundaryDomains)

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
            F_temp = (p - p_n)*v_p*dx + 0.5*dt*c_squared*(-dot(q_n, grad(v_p))*dx + q_bNormal*v_p*ds(1) +
                                                          dot(q_n, n)*v_p*ds(0) + dot(q_n, n)*v_p*ds(2)) + \
                     dot(q - q_n, v_q)*dx + 0.5*dt*(-div(v_q)*p_n*dx + dot(v_q, n)*forceExpression*ds(0) +
                                                    dot(v_q, n)*mirrorBoundary*ds(2) + dot(v_q, n)*p_n*ds(1))
            F = (p - p_n)*v_p*dx + dt*c_squared*(-dot(q_temp, grad(v_p))*dx + q_bNormal*v_p*ds(1) +
                                                 dot(q_temp, n)*v_p*ds(0) + dot(q_temp, n)*v_p*ds(2)) + \
                dot(q - q_temp, v_q)*dx + 0.5*dt*(-div(v_q)*p*dx + dot(v_q, n)*forceExpression*ds(0) +
                                                  dot(v_q, n)*mirrorBoundary*ds(2) + dot(v_q, n)*p*ds(1))
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
                dot(q - q_n, v_q)*dx - dt*div(v_q)*p_n*dx + dt*dot(v_q, n)*forceExpression*ds(0) + \
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
                dot(q - q_n, v_q)*dx - dt*div(v_q)*p*dx + dt*dot(v_q, n)*forceExpression*ds(0) + \
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
            F = (p - p_n)*v_p*dx + dt*c_squared*(-dot(q_n, grad(v_p))*dx + q_bNormal*v_p*ds(1) + \
                dot(q_n, n)*v_p*ds(0) + dot(q_n, n)*v_p*ds(2)) + \
                dot(q - q_n, v_q)*dx - dt*div(v_q)*p*dx + dt*dot(v_q, n)*forceExpression*ds(0) + \
                dt*dot(v_q, n)*mirrorBoundary*ds(2) + dt*dot(v_q, n)*p*ds(1)
        else:
            print('dirichlet implementation {} not implemented'.format(dirichletImp))
            exit()

    elif timeIntScheme == 'EH':
    # implement Explicit Heuns
        if dirichletImp == 'strong':
            F_temp = (p - p_n)*v_p*dx + dt*c_squared*(-dot(q_n, grad(v_p))*dx + dt*q_bNormal*v_p*ds(1)) + \
                     dot(q - q_n, v_q)*dx + dt*dot(grad(p_n), v_q)*dx
            F = (p - p_n)*v_p*dx + 0.5*(dt*c_squared*(-dot(q_n, grad(v_p))*dx + dt*q_bNormal*v_p*ds(1)) + \
                                        dt*c_squared*(-dot(q_temp, grad(v_p))*dx + dt*q_bNormal*v_p*ds(1))) + \
                 dot(q - q_n, v_q)*dx + 0.5*(dt*dot(grad(p_n), v_q)*dx + \
                                            dt*dot(grad(p_temp), v_q)*dx)

        elif dirichletImp == 'weak':
            F_temp = (p - p_n)*v_p*dx + dt*c_squared*(-dot(q_n, grad(v_p))*dx + dt*q_bNormal*v_p*ds(1) + \
                                                      dot(q_n, n)*v_p*ds(0) + dot(q_n, n)*v_p*ds(2)) + \
                      dot(q - q_n, v_q)*dx + dt*(-div(v_q)*p_n*dx + dot(v_q, n)*forceExpression*ds(0) + \
                                                dot(v_q, n)*mirrorBoundary*ds(2) + dot(v_q, n)*p_n*ds(1))
            F = (p - p_n)*v_p*dx + 0.5*dt*c_squared*(-dot(q_n, grad(v_p))*dx + dt*q_bNormal*v_p*ds(1) + \
                                                      dot(q_n, n)*v_p*ds(0) + dot(q_n, n)*v_p*ds(2) + \
                                                      -dot(q_temp, grad(v_p))*dx + dt*q_bNormal*v_p*ds(1) + \
                                                      dot(q_temp, n)*v_p*ds(0) + dot(q_temp, n)*v_p*ds(2)) + \
                 dot(q - q_n, v_q)*dx + 0.5*dt*(-div(v_q)*p_n*dx + dot(v_q, n)*forceExpression*ds(0) + \
                                                dot(v_q, n)*mirrorBoundary*ds(2) + dot(v_q, n)*p_n*ds(1) + \
                                                -div(v_q)*p_temp*dx + dot(v_q, n)*forceExpression*ds(0) + \
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
    # These are the energy contributions from the left, general and right boundaries for the first half timestep
    pT_L_q_temp_left = 0.5*dt*c_squared*dot(q0, n)*forceExpression*ds(0)
    pT_L_q_temp_gen = 0.5*dt*c_squared*q_bNormal*p0*ds(1)
    pT_L_q_temp_right = 0.5*dt*c_squared*dot(q0, n)*mirrorBoundary*ds(2)

    # These are the energy contributions from the left, general and right boundaries for the second half timestep
    pT_L_q_left = pT_L_q_temp_left + 0.5*dt*c_squared*dot(q1, n)*forceExpression*ds(0)
    pT_L_q_gen = pT_L_q_temp_gen + 0.5*dt*c_squared*q_bNormal*p1*ds(1)
    pT_L_q_right = pT_L_q_temp_right+ 0.5*dt*c_squared*dot(q1, n)*mirrorBoundary*ds(2)

    # Total energy contribution from the boundaries for the time step
    pT_L_q = pT_L_q_left + pT_L_q_gen + pT_L_q_right

    # -------------------------------# Set up output and plotting #---------------------------------#

    # Create xdmf Files for visualisation
    xdmfFile_p = XDMFFile(os.path.join(outputDir, 'p.xdmf'))
    xdmfFile_q = XDMFFile(os.path.join(outputDir, 'q.xdmf'))
    xdmfFile_p.parameters["flush_output"] = True
    xdmfFile_q.parameters["flush_output"] = True

    # Create progress bar
    progress = Progress('Time-stepping', numSteps)

    # Create vector for hamiltonian
    E_vec = [0]
    H_vec = [0]
    # HLeft_vec= [0]
    # HGen_vec= [0]
    # HRight_vec= [0]
    t_vec = [0]

    # Create plot for hamiltonian
    fig, ax = plt.subplots(1, 1)


    # create line object for plotting
    line, = ax.plot([], lw=0.5, color='k', linestyle='-',
                    label='Hamiltonian {}, {}, {}'.format(domainShape,timeIntScheme,dirichletImp))

    # text = ax.text(0.001, 0.001, "")
    ax.set_xlim(0, tFinal)
    if domainShape == 'R':
        # ax.set_ylim(0, 600)
        ax.set_ylim(0, 0.04)
    elif domainShape == 'squareWInput':
        ax.set_ylim(0, 0.03)

    ax.legend()
    ax.set_ylabel('Hamiltonian [J/kg]')
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

        forceExpression.t = t


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
        A = assemble(a)
        B = assemble(L)

        # Apply boundary conditions to matrices
        [bc.apply(A, B) for bc in bcs]

        solve(A, u_.vector(), B)

        # split solution
        out_p, out_q = u_.split()

        # Save solution to file
        xdmfFile_p.write(out_p, t)
        xdmfFile_q.write(out_q, t)

        # Update progress bar
        set_log_level(LogLevel.PROGRESS)
        progress += 1

        # Calculate Hamiltonian and plot energy
        boundaryEnergy += assemble(pT_L_q)
        internalEnergy = assemble((0.5*p_*p_ + 0.5*c_squared*inner(q_, q_))*dx)

        E = internalEnergy + boundaryEnergy
        H = internalEnergy
        # HLeft += assemble(pT_L_q_left)
        # HGen += assemble(pT_L_q_gen)
        # HRight += assemble(pT_L_q_right)

        E_vec.append(E)
        H_vec.append(H)
        # HLeft_vec.append(HLeft)
        # HGen_vec.append(HGen)
        # HRight_vec.append(HRight)
        t_vec.append(t)

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

    return H_vec, E_vec, t_vec, numCells

