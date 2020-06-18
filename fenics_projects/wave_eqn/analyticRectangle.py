import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
import os

# This Script solves the analytic wave equation for a rectangle

a = 1.0 # xLength
b = 0.25 # yLength
c_squared = 1.5

# Boundary conditions
# f(0,y,t) = 0, f(a,y,t) = 0
# f(x,0,t) = 0, f(a,b,t) = 0
# Initial conditions
# f(x,y,0) = 10*x*(xLength - x) + 20*y*(yLength - y)
# f_dot(x,y,0) = 0
# calculated parameters
# alpha =
# beta =

sumEnd = 100 # large value to sum integers to
for n in range(1, sumEnd+1):
    for m in range(1, sumEnd + 1):
        alpha = 1
        beta = 1

