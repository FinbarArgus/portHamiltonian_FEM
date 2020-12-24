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

def calculate_eigen(tFinal, numSteps, outputDir,
                  nx, xLength, yLength,
                  domainShape='R', timeIntScheme='SV', dirichletImp='weak',
                  K_wave=1, rho=1,
                  analytical=False,
                  basis_order=(1, 1),
                  eigen_type='continuous'):
    """

    :param tFinal: total time for simulation, doesn't matter for continuous eigen_type
    :param numSteps: time step for sim, doesn't matter for continuous eigen_type
    :param outputDir: directory to save data in if needed
    :param nx: approximate number of cells in x direction
    :param xLength: length of domain in x direction
    :param yLength: length of domain in y direction
    :param domainShape: shape of domain, TODO only 'R' for Rectangle implemented so far
    :param timeIntScheme: The time integration scheme used, only SE for Symplectic euler used so far, this
                        doesn't matter for continuous eigen_type
    :param dirichletImp: weak or strong dirichlet implementation, TODO only weak works in this eigen calc so far
    :param K_wave: stiffness of wave domain
    :param rho: density of wave domain
    :param analytical: True if comparing to analytical
    :param basis_order: tuple, 1st entry is order of lagrange element, 2nd entry is order of RT element
    :param eigen_type: whether we are calculating the eigenvalues of the continuous x_dot = A*x system or
                        the discrete x_(i+1) = A*x_i system. Currently only continuous method has been checked
    :return:
    """

    # TODO parallel hasn't been checked for this function
    comm = MPI.COMM_WORLD
    numProcs = comm.Get_size()
    if numProcs > 1:
        parallel = True
    else:
        parallel = False

    rank = comm.Get_rank()

    # Find initial time
    tic = time.time()
    if rank == 0:
        print('Started solving for case: {}, {}, {} '.format(domainShape, timeIntScheme, dirichletImp))

    # ------------------------------# Create Mesh #-------------------------------#
    if domainShape == 'R':
        xInputStart = 0.0  # at what x location is the input boundary
        mainRectangle = mshr.Rectangle(Point(0.0, 0.0), Point(xLength, yLength))
        mesh = mshr.generate_mesh(mainRectangle, nx)

    else:
        print('domainShape \"{}\" hasnt been created'.format(domainShape))
        quit()

    if analytical:
        if not domainShape == 'R':
            print('analytical solution for rectangle domain only')

    #get number of cells in mesh
    numCells = mesh.num_cells()
    if rank == 0:
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
    c_squared_float = K_wave/rho
    c = np.sqrt(K_wave/rho)
    K_wave = Constant(K_wave)
    rho_val = rho
    # rho = Constant(rho)

    # Define function spaces
    P1 = FiniteElement('P', triangle, basis_order[0])
    RT = FiniteElement('RT', triangle, basis_order[1])

    element = MixedElement([P1, RT])
    U = FunctionSpace(mesh, element)

    p_eigs, q_eigs = TrialFunctions(U)
    v_p_eigs, v_q_eigs = TestFunctions(U)

    p_eigs_n, q_eigs_n = TrialFunctions(U)
    v_p_eigs_n, v_q_eigs_n = TestFunctions(U)

    # Mark boundaries
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

    # Get normal
    n = FacetNormal(mesh)

    # mirrorBoundary = Constant(0.0)
    # # Make an expression for the boundary term for top and bottom edges
    # q_bNormal = Constant(0.0)

    # calculate exact eigenvalues
    #TODO make sure this is calculating all of the eigenvalues correctly, i.e am I doubling up on eigenvalues
    # maybe II=0 and JJ=0 shouldn't be used?
    comp_eig_vals_exact = []
    for II in range(0, 15):
        for JJ in range(1, 40):
            comp_eig_val = c*np.sqrt((JJ*np.pi/xLength)**2 + (II*np.pi/yLength)**2)
            # if comp_eig_val not in comp_eig_vals_exact:
            comp_eig_vals_exact.append(comp_eig_val)
    comp_eig_vals_exact.sort()

    # create petsc matrix and calculate eigenvalues of the A matrix
    if eigen_type == 'continuous':
        A_wbc_dofs = PETScMatrix()
        M_wbc_dofs = PETScMatrix()
        # This is the variational form of the wave equation with x_dot = 0 or rather the RHS of x_dot = Ax
        #            F_eigs = c_squared*(-dot(q_eigs, grad(v_p_eigs))*dx + q_bNormal*v_p_eigs*ds(1) +
        #                                dot(q_eigs, n)*v_p_eigs*ds(0) + dot(q_eigs, n)*v_p_eigs*ds(2)) - \
        #                     div(v_q_eigs)*p_eigs*dx + dot(v_q_eigs, n)*pLeftBoundary*ds(0) + \
        #                     dot(v_q_eigs, n)*mirrorBoundary*ds(2) + dot(v_q_eigs, n)*p_eigs*ds(1)
        # Without boundary terms
        F_eigs = c_squared*(-dot(q_eigs, grad(v_p_eigs))*dx +
                            dot(q_eigs, n)*v_p_eigs*ds(0) + dot(q_eigs, n)*v_p_eigs*ds(2)) - \
                 div(v_q_eigs)*p_eigs*dx + dot(v_q_eigs, n)*p_eigs*ds(1)
        F_eigs_M = p_eigs_n*v_p_eigs_n*dx + dot(q_eigs_n, v_q_eigs_n)*dx

        a_eigs = lhs(F_eigs)
        m_eigs = lhs(F_eigs_M)

        assemble(a_eigs, tensor=A_wbc_dofs)
        assemble(m_eigs, tensor=M_wbc_dofs)

        bc_eigens = DirichletBC(U.sub(0), Constant(0.0), 'on_boundary')
        bdry_dofs = bc_eigens.get_boundary_values().keys()
        boundary_dofs = list(bdry_dofs)

        # Remove dofs with boundary conditions, this isn't needed for weak implementation
        # A_numpy = np.delete(np.delete(A_wbc_dofs.array(), boundary_dofs, axis=0), boundary_dofs, axis=1)
        # M_numpy = np.delete(np.delete(M_wbc_dofs.array(), boundary_dofs, axis=0), boundary_dofs, axis=1)
        A_numpy = A_wbc_dofs.array()
        M_numpy = M_wbc_dofs.array()

        # if this is true use simple numpy inverse and np.linalg.eig to calculate eigs
        # if it is False use SLEPc
        compare_with_numpy = True
        if compare_with_numpy:
            # Use numpy inv and eig to calculate eigs, currently this works better than SLEPc
            M_numpy_inv = np.linalg.inv(M_numpy)
            Minv_A_numpy = M_numpy_inv.dot(A_numpy)
            eig_vals_numpy, eig_vecs_numpy = np.linalg.eig(Minv_A_numpy)

            eig_vals_list_numpy = []
            for eig_val in eig_vals_numpy:
                if abs(eig_val) > 1e-8:
                    eig_vals_list_numpy.append(eig_val)
            eig_vals_list_numpy.sort(key=lambda x: abs(x.imag), reverse=False)
            real_eig_vals_list_numpy = [x.real for x in eig_vals_list_numpy]
            comp_eig_vals_list_numpy = [x.imag for x in eig_vals_list_numpy]
            comp_eig_vals_pos_numpy = [x for x in comp_eig_vals_list_numpy if x>1e-8]
            # comp_eig_vals_pos_numpy.insert(0, 0.0)

            num_eigs = min(51, len(comp_eig_vals_pos_numpy))
            print('exact eigenvalues complex parts are')
            print(comp_eig_vals_exact)
            print('The first {} eigenvalues positive complex parts from numpy are'.format(num_eigs))
            print(comp_eig_vals_pos_numpy)
            # Calculate percent error on eigenvalues
            eig_error_percent = 100*(np.array(comp_eig_vals_exact)[:num_eigs] - comp_eig_vals_pos_numpy[:num_eigs])/ \
                                np.array(comp_eig_vals_pos_numpy)[:num_eigs]
            print('eigenvalue percentage error is')
            print(eig_error_percent)

            # create out_array for output
            out_array = np.zeros((num_eigs, 3))
            out_array[:, 0] = np.array(comp_eig_vals_exact)[:num_eigs]  # exact eigenvalues
            out_array[:, 1] = comp_eig_vals_pos_numpy[:num_eigs]    # model eigenvalues
            # also save the real parts of eig vals so we can check that they are all zero
            out_array[:, 2] = real_eig_vals_list_numpy[:num_eigs]


        else:
            #create petsc array
            A_petsc = pet.Mat().create()
            A_petsc.setSizes(A_numpy.shape)
            A_petsc.setType('aij')
            A_petsc.setUp()
            A_petsc.setValues(range(0, A_numpy.shape[0]), range(0, A_numpy.shape[0]), A_numpy)
            A_petsc.assemble()
            A_petsc = PETScMatrix(A_petsc)

            M_petsc = pet.Mat().create()
            M_petsc.setSizes(M_numpy.shape)
            M_petsc.setType('aij')
            M_petsc.setUp()
            M_petsc.setValues(range(0, M_numpy.shape[0]), range(0, M_numpy.shape[0]), M_numpy)
            M_petsc.assemble()
            M_petsc = PETScMatrix(M_petsc)


            eigensolver = SLEPcEigenSolver(A_petsc, M_petsc)
            eigensolver.parameters['spectrum'] = 'smallest imaginary'
            eigensolver.parameters['solver'] = 'arnoldi'
            eigensolver.solve()
            conv = eigensolver.get_number_converged()
            print('number of eigenvalues converged = {}'.format(conv))
            num_eigs_full = conv
            # extract first num_eigs eigenpair
            real_eig_vals_list_full = []
            comp_eig_vals_list_full = []
            for i in range(num_eigs_full):
                real_eig_vals, comp_eig_vals, real_eig_vecs, comp_eig_vecs = eigensolver.get_eigenpair(i)
                real_eig_vals_list_full.append(real_eig_vals)
                comp_eig_vals_list_full.append(comp_eig_vals)

            real_eig_vals_list = []
            comp_eig_vals_list = []
            comp_eig_vals_pos = [0.0]
            for real_val, comp_val in zip(real_eig_vals_list_full, comp_eig_vals_list_full):
                if abs(comp_val) > 1e-8:
                    real_eig_vals_list.append(real_val)
                    comp_eig_vals_list.append(comp_val)
                    if comp_val > 0:
                        comp_eig_vals_pos.append(comp_val)

            num_eigs = len(real_eig_vals_list)


            # Calculate percent error on eigenvalues
            eig_error_percent = 100*(np.array(comp_eig_vals_exact)[:30] - np.array(comp_eig_vals_pos)[:30])/ \
                                np.array(comp_eig_vals_pos)[:30]

            print('The first {} eigenvalues real parts are'.format(num_eigs))
            print(real_eig_vals_list)

            print('exact eigenvalues complex parts are'.format(num_eigs))
            print(comp_eig_vals_exact)
            print('The positive eigenvalue complex parts are'.format(num_eigs))
            print(comp_eig_vals_pos)
            print('eigenvalue percentage error is')
            print(eig_error_percent)

            out_array = np.zeros((num_eigs, 3))
            out_array[0, :] = np.array(comp_eig_vals_exact)[:num_eigs]  # exact eigenvalues
            out_array[1, :] = np.array(comp_eig_vals_pos)[:num_eigs]    # model eigenvalues
            # also save the real eig vals so we can check that the real parts are all zero
            out_array[2, :] = np.array(real_eig_vals_list)[:num_eigs]

    #            u_eigs = Function(U)
    #            u_eigs.vector()[:] = real_eig_vecs
    #            p_eig_vec, q_eig_vec = u_eigs.split(True)
    #
    #            xdmfFile_eig_p_vec = XDMFFile(os.path.join(outputDir, 'eig_p.xdmf'))
    #            xdmfFile_eig_p_vec.parameters["flush_output"] = True
    #            xdmfFile_eig_p_vec.write(p_eig_vec)

        return out_array, numCells

    elif eigen_type == 'discrete':
        # This is finding the eigenvalues for the time discretised system
        # TODO include this for the control to ensure eigenvalues are below 1
        if timeIntScheme == 'SE':
            F_eigs = p_eigs*v_p_eigs*dx + dot(q_eigs, v_q_eigs)*dx - dt*div(v_q_eigs)*p_eigs*dx + \
                     dt*dot(v_q_eigs, n)*p_eigs*ds(1)
            #dt*dot(v_q_eigs, n)*pLeftBoundary*ds(0) +
            F_eigs_n = -p_eigs_n*v_p_eigs_n*dx + dt*c_squared*(-dot(q_eigs_n, grad(v_p_eigs_n))*dx + \
                                                               dot(q_eigs_n, n)*v_p_eigs_n*ds(0) + dot(q_eigs_n, n)*v_p_eigs_n*ds(2)) - dot(q_eigs_n, v_q_eigs_n)*dx
        else:
            print('eigen problem not implemented for any timeIntScheme but SE')
            quit()

        A_eigs_petsc = PETScMatrix()
        A_eigs_n_petsc = PETScMatrix()

        a_eigs = lhs(F_eigs)
        a_eigs_n = lhs(F_eigs_n)
        B1 = rhs(F_eigs)
        B2 = rhs(F_eigs_n)

        assemble(a_eigs, tensor=A_eigs_petsc)
        assemble(a_eigs_n, tensor=A_eigs_n_petsc)

        A_eigs_numpy = A_eigs_petsc.array()
        A_eigs_numpy_inv = np.linalg.inv(A_eigs_numpy)
        A_eigs_n_numpy = A_eigs_n_petsc.array()
        A_numpy = -A_eigs_numpy_inv.dot(A_eigs_n_numpy)

        # calculate eigenvalues and ves with numpy
        eig_vals, eig_vecs = np.linalg.eig(A_numpy)

        #create petsc array
        A_petsc = pet.Mat().create()
        A_petsc.setSizes(A_numpy.shape)
        A_petsc.setType('aij')
        A_petsc.setUp()
        A_petsc.setValues(range(0, A_numpy.shape[0]), range(0, A_numpy.shape[0]), A_numpy)
        A_petsc.assemble()
        A_petsc = PETScMatrix(A_petsc)
        # A.setType('dense')

        # calculate eigenvalues and vecs with SLEPc
        eigensolver = SLEPcEigenSolver(A_petsc)
        eigensolver.solve()
        # extract first eigenpair
        real_eig_vals, comp_eig_vals, real_eig_vecs, comp_eig_vecs = eigensolver.get_eigenpair(0)

        print('first 10 eigenvalues are')
        print(eig_vals[:10])
        print('first SLEPc eigenvalue is')
        print(real_eig_vals)
        print('maximum numpy eigenvalue is {}'.format(max(eig_vals)))

        return

