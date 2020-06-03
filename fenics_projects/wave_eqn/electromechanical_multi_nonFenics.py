from __future__ import division, print_function
import numpy as np
import scipy as sp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.linalg
import os


def lqr(A, B, Q, R):
    """Solve the continuous time lqr controller.

    dx/dt = A x + B u

    cost = integral x.T*Q*x + u.T*R*u
    """


    # ref Bertsekas, p.151

    # first, try to solve the ricatti equation
    X = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))

    # compute the LQR gain
    K = np.matrix(scipy.linalg.inv(R) * (B.T * X))

    eigVals, eigVecs = scipy.linalg.eig(A - B * K)

    return K, X, eigVals


def dlqr(A, B, Q, R):
    """Solve the discrete time lqr controller.

    x[k+1] = A x[k] + B u[k]

    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    """


    # ref Bertsekas, p.151

    # first, try to solve the ricatti equation
    X = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))

    # compute the LQR gain
    K = np.matrix(scipy.linalg.inv(B.T * X * B + R) * (B.T * X * A))

    eigVals, eigVecs = scipy.linalg.eig(A - B * K)

    return K, X, eigVals


class ToyProblems:
    """
    class for simulating multiple pendulums
    """

    def __init__(self, Bl, R1, R2, K, m, L, controlInputMax, r1, gamma1, gamma2, gamma3, controlType='IDA_PBC'):
        self.Bl = Bl
        self.R1 = R1
        self.R2 = R2
        self.K = K
        self. m = m
        self.L = L
        self.controlInputMax = controlInputMax
        self.r1 = r1
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.gamma3 = gamma3
        self.controlType = controlType


    def electromechanical(self, t, xRow):
        """

        :param x: state variables [1,3]
        :param u: inputs [3,1]
        :return: state derivs [3,1]
        """
        #turn x into a column vector
        x = xRow.reshape((3,1))

        # create step function y_desired
        y_desired = 0
        if t < 20.0:
            y_desired = 1
        elif t<40.0:
            y_desired = 0
        elif t<60.0:
            y_desired = 1

        y_error = x[2] - y_desired

        x_desired = np.array([[0], [0], [y_desired]])


        fOld = np.zeros([3, ])

        J = np.array([[0, Bl, 0], [-Bl, 0, -1], [0, 1, 0]])

        R = np.array([[R1, 0, 0], [0, R2, 0], [0, 0, 0]])

        A_conv = np.array([[1/L, 0, 0], [0, 1/m, 0], [0, 0, K]])

        B = np.array([[1], [0], [0]])


        if controlType == 'IDA_PBC':

            u1 = -(r1 * gamma1 + R1/L)*x[0] + (Bl*gamma2/(L*gamma1) - Bl/m - 2 * gamma2)*x[1] + gamma3 * (y_error)
            f = (J - R).dot(A_conv.dot(x)) + B*u1

            # CHECK
            J_d = np.array([[0, Bl/(L*gamma1)-2, 1], [-Bl/(L*gamma1)+2, 0, -K/gamma3], [-1, K/gamma3, 0]])

            R_d = np.array([[r1, 0, 0], [0, 2*K/gamma3 +(R2-2)/(m*gamma2), 0], [0, 0, 0]])

            A_d_conv = np.array([[gamma1, 0, 0], [0, gamma2, 0], [0, 0, gamma3]])


            f_check = (J_d - R_d).dot(A_d_conv.dot(x-x_desired))
            f = f_check

            # print("discrepency = {}".format(f_check - f))

        elif controlType == 'lqr':
            # calculate the control input with LQR
            A = (J - R).dot(A_conv)

            Q = np.eye(3)
            R = np.eye(1)
            R[0,0] = 0.01

            Q[0,0] = 0
            Q[1,1] = 0
            Q[2,2] = 100

            K_lqr, S, E = lqr(A, B, Q, R)

            u1 = -K_lqr.dot(x-x_desired)
            f = (J - R).dot(A_conv.dot(x)) + B*u1

        elif controlType == 'PD':
            # calculate the control input with PD control
            kP = 20
            kD = 12 # This acts on the momentum, not the velocity
            u1 = - kP * y_error - kD * x[1]

            f = (J - R).dot(A_conv.dot(x)) + B*u1

        elif controlType == 'sin':

            u1 = np.sin(3.14*t)
            f = (J - R).dot(A_conv.dot(x)) + B*u1


        if u1 > controlInputMax:
            # print('control input of {} was above controlInputMax, capping to {}'.format(u1, controlInputMax))
            u1 = controlInputMax
        elif u1 < -controlInputMax:
            # print('control input of {} was above controlInputMax, capping to -{}'.format(u1, controlInputMax))
            u1 = -controlInputMax



        return f.flatten()

    def angle_to_cartesian(self, y):
        cartesian_pos = np.zeros([4, len(y[0,:])])

        cartesian_pos[0, :] = l_1 * np.sin(y[0,:])
        cartesian_pos[1, :] = -l_1 * np.cos(y[0,:])
        cartesian_pos[2, :] = cartesian_pos[0, :] + l_2 * np.sin(y[1,:])
        cartesian_pos[3, :] = cartesian_pos[1, :] - l_2 * np.cos(y[1,:])

        return cartesian_pos



Bl = 1
R1 = 0.0
R2 = 0.0
K = 1
m = 2
L = 0.1
controlInputMax = 50

controlType = 'sin'
# controlType = 'IDA_PBC'
# controlType = 'lqr'

# IDA_PBC control parameters
# r1 = 1.0
# gamma1 = 10
# gamma2 = 0.1
# gamma3 = 70
colorVec = [ 'c', 'b', 'g', 'y', 'orange', 'r']

ii = 0

r1 = 1.0
gamma1 = 80
gamma2 = 15 + 20*ii
gamma3 = 40

toyP = ToyProblems(Bl, R1, R2, K, m, L, controlInputMax, r1, gamma1, gamma2, gamma3, controlType=controlType)
x0 = np.array([0, 0, 0.0])
tStop = 20.0
nSteps = 2001
tSpan = np.array([0, tStop])
sol = solve_ivp(toyP.electromechanical, tSpan, x0, t_eval=np.linspace(0, tStop, nSteps),
                method='RK45')
# print(sol.t)

# recreate y_desired for output time points
yDesiredVec = np.zeros((len(sol.t),))
for II in range(len(sol.t)):
    t = sol.t[II]
    if t < 20.0:
         yDesiredVec[II] = 1
    elif t<40.0:
        yDesiredVec[II] = 0
    elif t<60.0:
       yDesiredVec[II] = 1

motorPos = sol.y[2, :]
h_1 = sol.y[0, :]
h_2 = sol.y[1, :]

if controlType =='sin':
    Hamiltonian = h_1**2/(2*L) + h_2**2/(2*m) + K*motorPos**2/2
else:
    Hamiltonian = gamma1*h_1**2/2 +gamma2*h_2**2 + gamma3*(motorPos - yDesiredVec)**2 / 2

#fig = plt.figure()
#ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
#ax.grid()

#line, = ax.plot([], [], 'o-', lw=2)
#time_template = 'time = %.1fs'
#time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text


def animate(i):
    thisx = [0, motorPos[i]]
    thisy = [0, 0]

    line.set_data(thisx, thisy)
    time_text.set_text(time_template % (sol.t[i]))
    return line, time_text


# ani = animation.FuncAnimation(fig, animate, np.arange(1, len(sol.y[0,:])),
#                              interval=10, blit=True, init_func=init)

# ani.save('simple_pendulum.mp4', fps=15)
# plt.show()

dataArray = np.load(os.path.join('output', 'output_em_ver', 'H_array.npy'))

fig = plt.figure()
plt.xlabel('Time [s]', fontsize=14)
plt.ylabel('Output Displacement [m]', fontsize=14)
plt.ylim(-1.2, 1.2)
plt.title("Step Response")
plt.plot(sol.t, motorPos, color=colorVec[ii], linestyle=':', label='scipy')
plt.xlim(0,20)
plt.plot(dataArray[:, 0], dataArray[:, 3], color='r', linestyle='--', label='fenics')
plt.legend(loc='lower right', fontsize=10)
savePath = os.path.join('output', 'plots', 'em_disp_scipy.png')
fig.savefig(savePath, dpi=1000, bbox_inches='tight')

fig = plt.figure()

plt.xlabel('Time [s]', fontsize=14)
plt.ylabel('Hamiltonian',fontsize=14)
plt.ylim(0, 15)
plt.title("Hamiltonian")

plt.plot(sol.t, Hamiltonian, color=colorVec[ii], linestyle=':', label='scipy')
plt.plot(dataArray[:, 0], dataArray[:, 1], color='r', linestyle='--', label='fenics')
plt.legend(loc='upper right', fontsize=10)
savePath = os.path.join('output', 'plots', 'em_Hamiltonian_scipy.png')
fig.savefig(savePath, dpi=1000, bbox_inches='tight')


fig = plt.figure()
plt.xlabel('Time [s]', fontsize=14)
plt.ylabel('displacement error', fontsize=14)
plt.title("Error between fenics and scipy")
plt.plot(dataArray[:, 0], np.sqrt(np.square(dataArray[:, 3]-motorPos)), color='r', linestyle='--')

# plt.legend(loc='upper right', fontsize=10)
savePath = os.path.join('output', 'plots', 'em_displacement_error_between_fenics_scipy.png')
fig.savefig(savePath, dpi=1000, bbox_inches='tight')
# plt.show()

fig = plt.figure()
plt.xlabel('Time [s]', fontsize=14)
plt.ylabel('Energy residual', fontsize=14)
plt.title("Energy residual for fenics and scipy")
plt.plot(dataArray[:, 0], np.sqrt(np.square(dataArray[:, 2])), color='r', linestyle='--',
         label='fenics')

# plt.legend(loc='upper right', fontsize=10)
savePath = os.path.join('output', 'plots', 'em_energy_residual_fenics.png')
fig.savefig(savePath, dpi=1000, bbox_inches='tight')
