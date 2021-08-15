import numpy as np
import time
import os
from wave_2D import *
from mpi4py import MPI
import test_case_check

if __name__ == '__main__':


    print('starting script and timer')
    tic = time.time()

    # -------------------------------# Define params #---------------------------------#

    # stiffness
    K_wave = 3.0 # 3.0
    # density
    rho = 2.0 # 2.0

    # caseName = 'test_cases'
    caseName = 'test_cases'
    if caseName == 'test_cases':
        # This is the quick test
        caseArray = [['R', 'IE', 'weak', 40, 'wave', 0.1, 200, 'noMem'],
                     ['R', 'EE', 'weak', 40, 'wave', 0.1, 200, 'noMem'],
                     ['R', 'EH', 'weak', 40, 'wave', 0.1, 200, 'noMem'],
                     ['R', 'SV', 'strong', 20, 'wave', 0.1, 200, 'noMem'],
                     ['R', 'SV', 'weak', 20, 'wave', 0.1, 200, 'noMem'],
                     ['R', 'SV', 'weak', 4, 'analytical', 0.1, 200, 'noMem', (3, 3)],
                     ['R', 'SV', 'weak', 20, 'analytical', 0.1, 200, 'noMem', (1, 2)],
                     ['R', 'SE', 'weak', 60, 'analytical', 0.1, 200, 'noMem', (1, 1)],
                     ['R', 'SV', 'weak', 60, 'IC', 0.1, 100, 'noMem'],
                     ['R', 'SM', 'weak', 60, 'IC', 0.1, 100, 'noMem'],
                     ['S_1C', 'SV', 'weak', 40, 'IC', 0.1, 100, 'noMem']]

    elif caseName == 'test_cases_long':
        # This is the old long tests for cases that are in the paper
        caseArray = [['R', 'IE', 'weak', 80, 'wave', 1.5, 3000, 'noMem'],
                      ['R', 'EE', 'weak', 80, 'wave', 1.5, 3000, 'noMem'],
                      ['R', 'EH', 'weak', 80, 'wave', 1.5, 3000, 'noMem'],
                      ['R', 'SV', 'strong', 20, 'wave', 1.5, 3000, 'noMem'],
                      ['R', 'SV', 'weak', 20, 'wave', 1.5, 3000, 'noMem'],
                      ['R', 'SV', 'weak', 4, 'analytical', 1.5, 3000, 'noMem', (3, 3)],
                      ['R', 'SV', 'weak', 20, 'analytical', 1.5, 3000, 'noMem', (1, 2)],
                      ['R', 'SE', 'weak', 80, 'analytical', 1.5, 3000, 'noMem', (3, 3)],
                      ['R', 'SV', 'weak', 120, 'IC', 4.5, 3000, 'noMem'],
                      ['R', 'SM', 'weak', 120, 'IC', 4.5, 3000, 'noMem'],
                      ['S_1C', 'SV', 'weak', 60, 'IC', 8, 16000, 'noMem']]
    else:
        print(caseName + ' not implemented')
        exit()


    for caseVec in caseArray:
        domainShape = caseVec[0]
        timeIntScheme = caseVec[1]
        dirichletImp = caseVec[2]

        assert len(caseVec) in [7, 8, 9], 'caseArray should be vectors of length 7, 8, or 9'
        # ------------------------------# Setup Directories #-------------------------------#

        # Create output dir
        outputDir = os.path.join('test_output', caseName)
        IC_BOOL = False
        if not os.path.exists(outputDir):
            os.mkdir(outputDir)
        subDir = domainShape + '_' + timeIntScheme + '_' + dirichletImp
        # we have a nx value to set
        subDir = subDir + '_nx' + str(caseVec[3])
        # also set IC that specifies whether interconnection is being modelled and if so whether it is 3DOF or 4DOF
        if caseVec[4].startswith('IC'):
            IC = caseVec[4]
        else:
            IC = None

        # create boolean that determines whether the simulation that compares with the analytical is being run
        ANALYTICAL_BOOL = (caseVec[4] == 'analytical')
        # add 'IC', 'wave', or 'analytical' to the subDir name to denote the type of model
        subDir = subDir + '_' + caseVec[4]
        # add time of sim and number of steps
        subDir = subDir + '_t' + str(caseVec[5]).replace('.','_') +\
                            '_steps' + str(caseVec[6])
        if len(caseVec) > 8:
            basis_order = caseVec[8]
            subDir = subDir + '_P{}_RT{}'.format(basis_order[0], basis_order[1])
        else:
            basis_order = (1, 1)

        outputSubDir = os.path.join(outputDir, subDir)
        if not os.path.exists(outputSubDir):
            os.mkdir(outputSubDir)

        # approx number of elems in x direction
        nx = caseVec[3]
        xLength = 1.0

        if caseVec[0] == 'R':
            yLength = 0.25
        elif caseVec[0] == 'S_1C':
            yLength = 0.1

        # set final time and number of steps
        tFinal = caseVec[5]
        numSteps = caseVec[6]

        # whether we want to save a xdf file for the p variable
        saveP = True
        if len(caseVec) > 7:
            if caseVec[7] == 'noMem':
                saveP = False

        # solve the wave equation
        H_array, numCells = wave_2D_solve(tFinal, numSteps, outputSubDir,
                                    nx, xLength, yLength,
                                    domainShape=domainShape, timeIntScheme=timeIntScheme,
                                    dirichletImp=dirichletImp,
                                    K_wave=K_wave, rho=rho, interConnection=IC,
                                    analytical=ANALYTICAL_BOOL, saveP=saveP, basis_order=basis_order)


        # -------------------------------# Set up output #---------------------------------#

        np.save(os.path.join(outputSubDir, 'H_array.npy'), H_array)

        # Save number of cells
        numCells_save = np.zeros((1))
        numCells_save[0] = numCells
        np.save(os.path.join(outputSubDir, 'numCells.npy'), numCells_save)



    totalTime = time.time() - tic
    print('All Simulations finished in {} seconds'.format(totalTime))
    print('now running test case check')

    # now compare results with ground truth
    test_case_check.check_cases(caseName)



