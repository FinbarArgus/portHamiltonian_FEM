from fenics import *
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import mshr
import paperPlotSetup
from wave_2D import *

if __name__ == '__main__':


    print('starting script and timer')
    tic = time.time()

    # -------------------------------# Define params #---------------------------------#

    # stiffness
    hookesK = 1.0
    # density
    rho = 1.0

    # caseArray = [['R', 'IE', 'weak'],
    #             ['R', 'IE', 'strong'],
    #             ['R', 'EE', 'weak'],
    #             ['R', 'EE', 'strong'],
    #             ['R', 'SE', 'weak'],
    #             ['R', 'SE', 'strong'],
    #             ['R', 'EH', 'weak'],
    #             ['R', 'EH', 'strong'],
    #             ['R', 'SV', 'weak'],
    #             ['R', 'SV', 'strong']]
    # caseArray = [['R', 'SV', 'strong', 40],
    #            ['R', 'SV', 'strong', 160],
    #            ['R', 'SV', 'weak', 40]]
#     caseArray = [['R', 'SV', 'strong', 60],
#                  ['R', 'SV', 'strong', 100],
#                  ['R', 'SV', 'strong', 120],
#                  ['R', 'SV', 'weak', 100],
#                  ['R', 'SV', 'weak', 120],
#                  ['R', 'SV', 'weak', 160],
#                  ['R', 'SV', 'weak', 60]]
#    caseArray = [['R', 'SV', 'strong', 20],
#                ['R', 'SV', 'weak', 20]]
    caseArray = [['R', 'SE', 'weak', 20, 'IC']]

    for caseVec in caseArray:
        domainShape = caseVec[0]
        timeIntScheme = caseVec[1]
        dirichletImp = caseVec[2]

        assert len(caseVec) in [3, 4, 5], 'caseArray should be vectors of length 3, 4 or 5'
        # ------------------------------# Setup Directories #-------------------------------#

        # Create output dir
        outputDir = 'output'
        IC_BOOL = False
        if not os.path.exists(outputDir):
            os.mkdir(outputDir)
        subDir = 'output_' + domainShape + '_' + timeIntScheme + '_' + dirichletImp
        if len(caseVec) == 4:
            # we have a nx value to set
            subDir = subDir + '_nx' + str(caseVec[3])
        elif len(caseVec) == 5:
            # add 'IC' to the subDir name to denote interconnection
            subDir = subDir + '_' + caseVec[4]
            # also set bool that specifies interconnection as true
            IC_BOOL = (caseVec[4] == 'IC')

        outputSubDir = os.path.join(outputDir, subDir)
        if not os.path.exists(outputSubDir):
            os.mkdir(outputSubDir)

        # Create the mesh
        if len(caseVec) ==3:
            nx = 80
        else:
            nx = caseVec[3]

        ny = int(nx/4)
        xLength = 1.0
        yLength = 0.25


        # solve the wave equation
        tFinal = 1.5
        numSteps = 3000

        H_vec, E_vec, t_vec, numCells = wave_2D_solve(tFinal, numSteps, outputSubDir,
                                    nx, ny, xLength, yLength,
                                    domainShape=domainShape, timeIntScheme=timeIntScheme, dirichletImp=dirichletImp,
                                    hookesK=hookesK, rho=rho, interConnection=IC_BOOL)

        # -------------------------------# Set up output and plotting #---------------------------------#

        H_array = np.zeros((numSteps + 1, 3))
        H_array[:, 0] = np.array(t_vec)
        H_array[:, 1] = np.array(H_vec)
        H_array[:, 2] = np.array(E_vec)
        np.save(os.path.join(outputSubDir, 'H_array.npy'), H_array)

        numCells_save = np.zeros((1))
        numCells_save[0] = numCells
        np.save(os.path.join(outputSubDir, 'numCells.npy'), numCells_save)

    totalTime = time.time() - tic
    print('All Simulations finished in {} seconds'.format(totalTime))


