from fenics import *
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import mshr
# import paperPlotSetup
from wave_2D import *
from mpi4py import MPI

if __name__ == '__main__':


    print('starting script and timer')
    tic = time.time()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    # -------------------------------# Define params #---------------------------------#

    # stiffness
    K_wave = 3.0 # 3.0
    # density
    rho = 2.0 # 2.0

    # caseName = 'passivity_control_IC_t4_5'
    caseName = 'lqr_control_t8_0'
    quick = True
    if quick:
        # caseArray = [['R', 'SM', 'weak', 40, 'IC', 0.4, 400, 'Mem', 'passivity']]
        caseArray = [['R', 'SM', 'weak', 3, 'IC', 0.4, 400, 'Mem', 'lqr']]
        input_stop_t = 0.1
        control_start_t = 0.2
    else:
        # caseArray = [['R', 'SM', 'weak', 40, 'IC', 8.0, 8000, 'Mem', 'passivity']]
        caseArray = [['R', 'SM', 'weak', 40, 'IC', 8.0, 8000, 'Mem', 'lqr']]
        input_stop_t = 0.0
        control_start_t = 1.0
    # Create output dir
    outputBaseDir = 'control_output'
    if not os.path.exists(outputBaseDir):
        os.mkdir(outputBaseDir)
    outputDir = os.path.join(outputBaseDir, caseName)
    if not os.path.exists(outputDir):
        os.mkdir(outputDir)

    for caseVec in caseArray:
        domainShape = caseVec[0]
        timeIntScheme = caseVec[1]
        dirichletImp = caseVec[2]

        assert len(caseVec) in [7, 8, 9], 'caseArray should be vectors of length 7 or 8'
        # ------------------------------# Setup Directories #-------------------------------#

        IC_BOOL = False
        subDir = domainShape + '_' + timeIntScheme + '_' + dirichletImp
        # we have a nx value to set
        subDir = subDir + '_nx' + str(caseVec[3])
        # also set IC that specifies whether interconnection is being modelled and if so whether it is 3DOF or 4DOF
        if caseVec[4].startswith('IC'):
            IC = caseVec[4]
        else:
            IC = None

        ANALYTICAL_BOOL = (caseVec[4] == 'analytical')
        # add 'IC', 'wave', or 'analytical' to the subDir name to denote interconnection
        subDir = subDir + '_' + caseVec[4]
        # add time of sim and number of steps
        subDir = subDir + '_t' + str(caseVec[5]).replace('.','_') +\
                            '_steps' + str(caseVec[6])

        outputSubDir = os.path.join(outputDir, subDir)
        if not os.path.exists(outputSubDir):
            if rank == 0:
                os.mkdir(outputSubDir)

        # approx number of elems in x direction
        nx = caseVec[3]
        xLength = 1.0

        if caseVec[0] == 'R':
            yLength = 0.25 #0.25
        elif caseVec[0] == 'S_1C':
            yLength = 0.1 #0.25

        # set final time and number of steps
        tFinal = caseVec[5]
        numSteps = caseVec[6]

        # whether we want to save a p xdf file
        saveP = True
        if len(caseVec) > 7:
            if caseVec[7] == 'noMem':
                saveP = False

        if len(caseVec) > 8:
            controlType = caseVec[8]
        else:
            controlType = None

        # solve the wave equation
        H_array, numCells = wave_2D_solve(tFinal, numSteps, outputSubDir,
                                    nx, xLength, yLength,
                                    domainShape=domainShape, timeIntScheme=timeIntScheme,
                                    dirichletImp=dirichletImp,
                                    K_wave=K_wave, rho=rho, interConnection=IC,
                                    analytical=ANALYTICAL_BOOL, saveP=saveP, controlType=controlType,
                                    input_stop_t=input_stop_t, control_start_t=control_start_t)

        # -------------------------------# Set up output and plotting #---------------------------------#

        np.save(os.path.join(outputSubDir, 'H_array.npy'), H_array)

        numCells_save = np.zeros((1))
        numCells_save[0] = numCells
        np.save(os.path.join(outputSubDir, 'numCells.npy'), numCells_save)

    if rank == 0:
        totalTime = time.time() - tic
        print('All Simulations finished in {} seconds'.format(totalTime))


