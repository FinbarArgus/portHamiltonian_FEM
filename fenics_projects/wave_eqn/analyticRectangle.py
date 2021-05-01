import numpy as np
from numpy import sqrt, sin, cos, pi
import matplotlib
import matplotlib.pyplot as plt
import time
import os

# This Script solves the analytic wave equation for a rectangle, 
# see the book Polyanin: Handbook of linear partial differential equations for engineers and scientists

L1 = 1.0 # xLength
L2 = 0.25 # yLength
c_squared = 1.5
c = np.sqrt(c_squared)
pi = np.pi
# Boundary conditions
# f(0,y,t) = g1, f(a,y,t) = 0
# f(x,0,t) = 0, f(a,b,t) = 0
# g1 = -5sin(8*pi*t)
# Initial conditions
# f(x,y,0) = 0
# f_dot(x,y,0) = 0
# calculated parameters

sumEnd = 100 # large value to sum integers to
A_m = 2 #for m not equal to 0

p_n = np.zeros((sumEnd,1))
for n in range(1, sumEnd):
    p_n[n] = n*pi/L1

q_m = np.zeros((sumEnd,1))
for m in range(1, sumEnd):
    q_m[m] = m*pi/L2


xVec = np.array([0.2])
yVec = np.array([0.1])
tVec = np.linspace(0,1,10)
w = np.zeros((len(xVec), len(yVec), len(tVec)))

#pre calculate some terms

for xCount, x in enumerate(xVec):
    for yCount, y in enumerate(yVec):
        for tCount, t in enumerate(tVec):
            print('solving for x = {}, y = {}, t = {}'.format(x, y, t))
            sum = 0
            for n in range(1, sumEnd):
                for m in range(1, sumEnd):
                    if m ==0:
                        A_m = 1
                    else:
                        A_m = 2

                    lamda_nm = sqrt(p_n[n]**2 + q_m[m]**2)
                    multiplier = A_m*p_n[n] / (lamda_nm*q_m[m])
                    sum += multiplier*sin(p_n[n]*x)*cos(q_m[m]*y)*sin(q_m[m]*L2)*(
                        sin(c*lamda_nm*t)*(sin((c*lamda_nm - 8*pi)*t) / (2*(c*lamda_nm - 8*pi)) +
                                           sin((c*lamda_nm + 8*pi)*t) / (2*(c*lamda_nm + 8*pi))) +
                        cos(c*lamda_nm*t)*(cos((c*lamda_nm + 8*pi)*t) / (2*(c*lamda_nm + 8*pi)) +
                                           sin((c*lamda_nm - 8*pi)*t) / (2*(c*lamda_nm - 8*pi)) -
                                           1.0/(2*(c*lamda_nm + 8*pi)) - 1.0/(2*(c*lamda_nm - 8*pi)))
                    )
                    print(sum)

            w[xCount, yCount, tCount] = -10*c/(8*pi*L1*L2)*sum

fig, ax = plt.subplots(1, 1)
ax.set_xlim(0, tVec[-1])
ax.set_ylim(-10.0, 10.0)
ax.set_xlabel('Time [s]')
ax.set_ylabel('w [m]')

plt.plot(tVec, w[0, 0, :])

plt.show()


print('done')





