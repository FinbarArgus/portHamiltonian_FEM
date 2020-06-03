from fenics import *
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import mshr
import paperPlotSetup
from electroMechanical import *

if __name__ == '__main__':


    print('starting script and timer')
    tic = time.time()

    domainShape = 'R'
    timeIntScheme = 'SE'
    dirichletImp = 'weak'
    # ------------------------------# Setup Directories #-------------------------------#

    # Create output dir
    outputDir = 'output'
    if not os.path.exists(outputDir):
        os.mkdir(outputDir)

    subDir = 'output_' + 'em_ver'

    outputSubDir = os.path.join(outputDir, subDir)
    if not os.path.exists(outputSubDir):
        os.mkdir(outputSubDir)

    nx = 1
    ny = 1
    xLength = 1.0
    yLength = 0.25


    # solve the wave equation
    tFinal = 20
    numSteps = 2000

    H_vec, E_vec, t_vec, disp_vec = electromechanical_solve(tFinal, numSteps, outputSubDir,
                                nx, ny, xLength, yLength,
                                domainShape=domainShape, timeIntScheme=timeIntScheme,
                                dirichletImp=dirichletImp)

    # -------------------------------# Set up output and plotting #---------------------------------#

    H_array = np.zeros((numSteps + 1, 4))
    H_array[:, 0] = np.array(t_vec)
    H_array[:, 1] = np.array(H_vec)
    H_array[:, 2] = np.array(E_vec)
    H_array[:, 3] = np.array(disp_vec)
    np.save(os.path.join(outputSubDir, 'H_array.npy'), H_array)

    totalTime = time.time() - tic
    print('All Simulations finished in {} seconds'.format(totalTime))


