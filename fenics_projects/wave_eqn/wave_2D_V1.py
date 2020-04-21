from fenics import *
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import mshr

"""wave_2D_V1 Solves the wave eqn for the seperated wave equation with boundary conditions applied 
with DirichletBC. The neumann type condition is applied as a dirichletBC in the split eqns, as q = (0,0)

 This code implents a rectangle domain and a more complicated domain where the sinusoidal wave
 starts on an input rectangle then travels into the main rectangle.
 The circular wavefront is seen in the main square part of the domain."""

version = 'V1'

# Find initial time
tic = time.time()

# constant wave speed
c = 1

# Create mesh
# Choose operating case
case = 'rectangle'
# case = 'squareWInput'

if case == 'rectangle':
    # time and time step
    tFinal = 0.5
    numSteps = 1000
    dt_value = tFinal / numSteps

    # Create mesh
    nx = 80
    ny = 20
    xInputStart = 0.0 # at what x location is the input boundary
    xLength = 1.0
    yLength = 0.25
    mesh = RectangleMesh(Point(xLength, 0.0), Point(0.0, yLength), nx, ny)
elif case == 'squareWInput':
    # time and time step
    tFinal = 1.0
    numSteps = 3000
    dt_value = tFinal / numSteps

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

# Create output dir
outputDir = 'output_' + version + '_' + case
if not os.path.exists(outputDir):
    os.mkdir(outputDir)

plotDir = os.path.join(outputDir, 'plots')
if not os.path.exists(plotDir):
    os.mkdir(plotDir)

# Create function spaces
P1 = FiniteElement('P', triangle, 1)
RT = FiniteElement('RT', triangle, 1)
element = MixedElement([P1, RT])
U = FunctionSpace(mesh, element)


# input left edge boundary condition
def forced_boundary(x, on_boundary):
    return on_boundary and near(x[0], xInputStart)


# Define right edge boundary condition
def fixed_boundary(x, on_boundary):
    return on_boundary and near(x[0], xLength)


# Define boundary on all other edges
def general_boundary(x, on_boundary):
    return on_boundary and not near(x[0], xInputStart) and not near(x[0], xLength)


# Forced boundary
force_expression = Expression('(t<0.25) ? sin(8*3.14159*t) : 0.0', degree=2, t=0)
bc_forced = DirichletBC(U.sub(0), force_expression, forced_boundary)

# fixed boundary at right boundary
bc_fixed = DirichletBC(U.sub(0), Constant(0.0), fixed_boundary)

# General q = 0 boundary at top and bottom
bc_general = DirichletBC(U.sub(1), Constant((0, 0)), general_boundary)

# List of boundary condition objects
bcs = [bc_forced, bc_fixed, bc_general]

# Define trial and test functions
p, q = TrialFunctions(U)

v_p, v_q = TestFunctions(U)

# Define functions for solutions at previous and current time steps
u_n = Function(U)
p_n, q_n = split(u_n)

u_ = Function(U)
p_, q_ = split(u_)

# Define params
c_squared = Constant(c**2)
dt = Constant(dt_value)

# Define variational problem
# Explicit
# F = (p - p_n)*v_p*dx - dt*c_squared*div(q_n)*v_p*dx + \
#      dot(q - q_n, v_q)*dx - dt*dot(grad(p_n),v_q)*dx
# Implicit
F = (p - p_n)*v_p*dx + dt*c_squared*div(q)*v_p*dx + \
     dot(q - q_n, v_q)*dx + dt*dot(grad(p),v_q)*dx
a = lhs(F)
L = rhs(F)

# Assemble matrices
A = assemble(a)
B = assemble(L)

# Apply boundary conditions to matrices
[bc.apply(A, B) for bc in bcs]

# Create xdmf Files for visualisation
xdmfFile_p = XDMFFile(os.path.join(outputDir, 'p.xdmf'))
xdmfFile_q = XDMFFile(os.path.join(outputDir, 'q.xdmf'))

# Create progress bar
progress = Progress('Time-stepping', numSteps)

# Create vector for hamiltonian
H_vec = [0]
t_vec = [0]

#Create plot for hamiltonian
fig, ax = plt.subplots(1, 1)

line, = ax.plot([], lw=1, color='k')
# text = ax.text(0.001, 0.001, "")
ax.set_xlim(0 , tFinal)
if case == 'rectangle':
    ax.set_ylim(0, 500)
elif case == 'squareWInput':
    ax.set_ylim(0, 300)

ax.set_ylabel('Hamiltonian [Joules]', fontsize=14)
ax.set_xlabel('Time [s]', fontsize=14)

ax.add_artist(line)
fig.canvas.draw()

# cache the background
axBackground = fig.canvas.copy_from_bbox(ax.bbox)

plt.show(block=False)

# Time Stepping
t = 0
for n in range(numSteps):
    set_log_level(LogLevel.ERROR)
    t += dt_value

    force_expression.t = t
    A = assemble(a)
    B = assemble(L)

    # Apply boundary conditions to matrices
    [bc.apply(A, B) for bc in bcs]

    solve(A, u_.vector(), B)

    #split solution
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
    ax.draw_artist(line)
    # ax.draw_artist(text)

    # Fill in the axes rectangle
    fig.canvas.blit(ax.bbox)

    # make sure all events have happened
    fig.canvas.flush_events()

plt.savefig(os.path.join(plotDir, 'Hamiltonian.png'), dpi=500)

H_array = np.zeros((numSteps + 1, 2))
H_array[:,0] = np.array(H_vec)
H_array[:,1] = np.array(t_vec)
np.save(os.path.join(outputDir, 'H_array.npy'), H_array)
# Calculate total time
totalTime = time.time() - tic
print('Simulation finished in {} seconds'.format(totalTime))

# plot final solution




