from fenics import *
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import mshr

"""V5 applies integration by parts to both equations and applies the boundary
conditions to the boundary term in the parts.

wave_2D_V4 has a choice of boundary conditions 
"DDNN" Solves the seperated wave equation with boundary conditions applied
with DirichletBC for the p input and right boundary.
"NNNN" has all neumann with the left side a force input neumann, I think this is ill-posed

 This code implents a rectangle domain and a more complicated domain where the sinusoidal wave
 starts on an input rectangle then travels into the main rectangle.
 The circular wavefront is seen in the main square part of the domain."""

version = 'V5'
# Choose operating case
case = 'rectangle'
# case = 'squareWInput'
#Define boundary type where the 4 letters are eith D for Dirchlet or N for Neumann
# The ordering is Left, Right, Bottom, Top
boundaryType = 'DDNN'
boundaryType = 'DDNN_PH'
# boundaryType = 'NNNN'

BOUNDARYPLOT = False

# Find initial time
tic = time.time()

# -------------------------------# Define params #---------------------------------#

# constant wave speed
c = 1
c_squared = Constant(c**2)

# -------------------------------# Create mesh #---------------------------------#


if case == 'rectangle':
    # time and time step
    tFinal = 1.5
    numSteps = 3000
    dt_value = tFinal/numSteps
    dt = Constant(dt_value)

    # Create mesh
    nx = 80
    ny = 20
    xInputStart = 0.0  # at what x location is the input boundary
    xLength = 1.0
    yLength = 0.25
    mesh = RectangleMesh(Point(xLength, 0.0), Point(0.0, yLength), nx, ny)
elif case == 'squareWInput':
    # time and time step
    tFinal = 1.0
    numSteps = 3000
    dt_value = tFinal/numSteps
    dt = Constant(dt_value)

    # create mesh
    xLength = 1.0
    yLength = 1.0
    xInputStart = -0.10
    inputRectanglePoint1 = Point(xInputStart, 0.45)
    inputRectanglePoint2 = Point(0.0, 0.55)
    mainRectangle = mshr.Rectangle(Point(0.0, 0.0), Point(xLength, yLength))
    inputRectangle = mshr.Rectangle(inputRectanglePoint1, inputRectanglePoint2)
    domain = mainRectangle + inputRectangle
    mesh = mshr.generate_mesh(domain, 64)
else:
    print('case \"{}\" hasnt been created'.format(case))
    quit()

# ------------------------------# Setup Directories #-------------------------------#

# Create output dir
outputDir = 'output_' + version + '_' + case + boundaryType
if not os.path.exists(outputDir):
    os.mkdir(outputDir)

# V1 output dir for loading
V1outputDir = 'output_' + 'V1' + '_' + case
# V2 output dir for loading
V2outputDir = 'output_' + 'V2' + '_' + case
# V3 output dir for loading
V3outputDir = 'output_' + 'V3' + '_' + case
# V4 output dir for loading
V4outputDir = 'output_' + 'V4' + '_' + case + 'DDNN'

plotDir = os.path.join(outputDir, 'plots')
if not os.path.exists(plotDir):
    os.mkdir(plotDir)

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
        return on_boundary and not near(x[0], xInputStart) and not near(x[0], xLength)


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

# If we have dirichlet conditions applied in the typical way set them here
if boundaryType == 'DDNN':
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
# int by parts implicit
# F = (p - p_n)*v_p*dx - dt*c_squared*dot(q, grad(v_p))*dx + dt*q_bNormal*v_p*ds(1) + \
#     dot(q - q_n, v_q)*dx + dt*dot(grad(p), v_q)*dx
# Simplectic stormer-verlet integration
if boundaryType == 'DDNN':
    F = (p - p_n)*v_p*dx - dt*c_squared*dot(q_n, grad(v_p))*dx + dt*q_bNormal*v_p*ds(1) + \
        dot(q - q_n, v_q)*dx + dt*dot(grad(p), v_q)*dx
elif boundaryType == 'DDNN_PH':
    # Here the boundary terms are applied to the int by parts term
    # F = (p - p_n)*v_p*dx - dt*c_squared*dot(q_n, grad(v_p))*dx + dt*q_bNormal*v_p*ds(1) + \
    #     dot(q - q_n, v_q)*dx - dt*div(v_q)*p*dx + dt*dot(v_q, n)*forceExpression*ds(0) + \
    #     dt*dot(v_q, n)*mirrorBoundary*ds(2)
    # The above didn't work, maybe have to include all boundary terms, since
    # We don't have a dirichletBC applied to set the test functions to zero on the boundary
    F = (p - p_n)*v_p*dx - dt*c_squared*dot(q_n, grad(v_p))*dx + dt*q_bNormal*v_p*ds(1) + \
        dt*dot(q_n, n)*v_p*ds(0) + dt*dot(q_n, n)*v_p*ds(2) + \
        dot(q - q_n, v_q)*dx - dt*div(v_q)*p*dx + dt*dot(v_q, n)*forceExpression*ds(0) + \
        dt*dot(v_q, n)*mirrorBoundary*ds(2) + dt*dot(v_q, n)*p*ds(1)
elif boundaryType == 'NNNN':
    F = (p - p_n)*v_p*dx - dt*c_squared*dot(q_n, grad(v_p))*dx + dt*q_bNormal*v_p*ds(1) + \
        dt*q_bNormal*v_p*ds(2) + dt*forceExpression*v_p*ds(0) + \
        dot(q - q_n, v_q)*dx + dt*dot(grad(p), v_q)*dx

a = lhs(F)
L = rhs(F)

# Assemble matrices
A = assemble(a)
B = assemble(L)

# Apply boundary conditions to matrices
[bc.apply(A, B) for bc in bcs]

# -------------------------------# Set up output and plotting #---------------------------------#

# Create xdmf Files for visualisation
xdmfFile_p = XDMFFile(os.path.join(outputDir, 'p.xdmf'))
xdmfFile_q = XDMFFile(os.path.join(outputDir, 'q.xdmf'))
xdmfFile_p.parameters["flush_output"] = True
xdmfFile_q.parameters["flush_output"] = True

# Create progress bar
progress = Progress('Time-stepping', numSteps)

# Create vector for hamiltonian
H_vec = [0]
t_vec = [0]

# Create plot for hamiltonian
fig, ax = plt.subplots(1, 1)

if case=='rectangle' and boundaryType.startswith('DDNN'):
    # Load hamiltonian values from V1
    H_array_V1 = np.load(os.path.join(V1outputDir, 'H_array.npy'))
    # Load hamiltonian values from V2
    H_array_V2 = np.load(os.path.join(V2outputDir, 'H_array.npy'))
    # Load hamiltonian values from V3
    H_array_V3 = np.load(os.path.join(V3outputDir, 'H_array.npy'))
    # Load hamiltonian values from V4
    H_array_V4 = np.load(os.path.join(V4outputDir, 'H_array.npy'))


# create line objects for plotting
line, = ax.plot([], lw=2, color='k', linestyle='--', label='Port-Hamiltonian')

if case == 'rectangle' and boundaryType.startswith('DDNN'):
    line_V1, = ax.plot([], lw=2, color='b', label='Implicit 1st-order Euler')
    line_V2, = ax.plot([], lw=2, color='r', label='Explicit 1st-order Euler')
    line_V3, = ax.plot([], lw=2, color='g', label='Explicit 2nd-order Heuns')
    line_V4, = ax.plot([], lw=2, color='c', label='Stormer-Verlet Symplectic')

# text = ax.text(0.001, 0.001, "")
ax.set_xlim(0, tFinal)
if case == 'rectangle':
    # ax.set_ylim(0, 600)
    ax.set_ylim(0, 600)
elif case == 'squareWInput':
    ax.set_ylim(0, 400)


ax.legend()
ax.set_ylabel('Hamiltonian [Joules]', fontsize=14)
ax.set_xlabel('Time [s]', fontsize=14)

ax.add_artist(line)
if case == 'rectangle' and boundaryType.startswith('DDNN'):
    ax.add_artist(line_V1)
    ax.add_artist(line_V2)
    ax.add_artist(line_V3)
    ax.add_artist(line_V4)

fig.canvas.draw()

if case == 'rectangle' and boundaryType.startswith('DDNN'):
    line_V1.set_data(H_array_V1[:,1], H_array_V1[:, 0])
    line_V2.set_data(H_array_V2[:,1], H_array_V2[:, 0])
    line_V3.set_data(H_array_V3[:,1], H_array_V3[:, 0])
    line_V4.set_data(H_array_V4[:,1], H_array_V4[:, 0])

# cache the background
axBackground = fig.canvas.copy_from_bbox(ax.bbox)

plt.show(block=False)

# -------------------------------# Solve Loop #---------------------------------#

# Time Stepping
t = 0
for n in range(numSteps):
    set_log_level(LogLevel.ERROR)
    t += dt_value

    forceExpression.t = t
    A = assemble(a)
    B = assemble(L)

    # Apply boundary conditions to matrices
    [bc.apply(A, B) for bc in bcs]

    solve(A, u_.vector(), B)

    # split solution
    out_p, out_q = u_.split()

    # Plot solution
    # plot(out_p, title='p')
    # plot(out_q, title='q')

    # Save solution to file
    xdmfFile_p.write(out_p, t)
    xdmfFile_q.write(out_q, t)

    # Update previous solution
    u_n.assign(u_)

    # Update progress bar
    set_log_level(LogLevel.PROGRESS)
    progress += 1

    # Calculate Hamiltonian and plot energy
    H = out_p.vector().inner(out_p.vector()) + c**2*out_q.vector().inner(out_q.vector())
    H_vec.append(H)
    t_vec.append(t)

    # Plot Hamiltonian
    # set new data point
    line.set_data(t_vec, H_vec)
    # text.set_text('HI')
    # line.set_data(0.1, 1)
    # restore background
    fig.canvas.restore_region(axBackground)

    # Redraw the points
    if case == 'rectangle' and boundaryType.startswith('DDNN'):
        ax.draw_artist(line_V1)
        ax.draw_artist(line_V2)
        ax.draw_artist(line_V3)
        ax.draw_artist(line_V4)
    ax.draw_artist(line)
    # ax.draw_artist(text)

    # Fill in the axes rectangle
    fig.canvas.blit(ax.bbox)

    # make sure all events have happened
    fig.canvas.flush_events()

# Save hamiltonian values
H_array = np.zeros((numSteps + 1, 2))
H_array[:,0] = np.array(H_vec)
H_array[:,1] = np.array(t_vec)
np.save(os.path.join(outputDir, 'H_array.npy'), H_array)

plt.savefig(os.path.join(plotDir, 'Hamiltonian.png'), dpi=500)
# overwrite limits if we want to zoom
ax.set_xlim(0.25, tFinal)
ax.set_ylim(425, 435)
plt.savefig(os.path.join(plotDir, 'HamiltonianZoom.png'), dpi=500)

if BOUNDARYPLOT:
    vtkplot(out_p)

# Calculate total time
totalTime = time.time() - tic
print('Simulation finished in {} seconds'.format(totalTime))

# plot final solution
