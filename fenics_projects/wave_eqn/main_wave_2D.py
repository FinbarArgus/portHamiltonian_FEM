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

#    caseName = 'strong_and_weak_method_variation'
#    caseArray = [['R', 'IE', 'weak', 80, 'wave', 1.5, 3000, 'Mem'],
#                ['R', 'IE', 'strong', 80, 'wave', 1.5, 3000, 'Mem'],
#                ['R', 'EE', 'weak', 80, 'wave', 1.5, 3000, 'Mem'],
#                ['R', 'EE', 'strong', 80, 'wave', 1.5, 3000, 'Mem'],
#                ['R', 'SE', 'weak', 80, 'wave', 1.5, 3000, 'Mem'],
#                ['R', 'SE', 'strong', 80, 'wave', 1.5, 3000, 'Mem'],
#                ['R', 'EH', 'weak', 80, 'wave', 1.5, 3000, 'Mem'],
#                ['R', 'EH', 'strong', 80, 'wave', 1.5, 3000, 'Mem'],
#                ['R', 'SV', 'weak', 80, 'wave', 1.5, 3000, 'Mem'],
#                ['R', 'SV', 'strong', 80, 'wave', 1.5, 3000, 'Mem']]
#    caseArray = [['R', 'SM', 'weak', 80, 'wave', 1.5, 3000, 'Mem'],
#                ['R', 'SM', 'strong', 80, 'wave', 1.5, 3000, 'Mem']]

    #    caseName = 'strong_and_weak_spatial_variation'
#    caseArray = [['R', 'SV', 'strong', 20, 'wave', 1.5, 3000, 'Mem'],
#                 ['R', 'SV', 'strong', 40, 'wave', 1.5, 3000, 'Mem'],
#                 ['R', 'SV', 'strong', 60, 'wave', 1.5, 3000, 'Mem'],
#                 ['R', 'SV', 'strong', 80, 'wave', 1.5, 3000, 'Mem'],
#                 ['R', 'SV', 'strong', 100, 'wave', 1.5, 3000, 'Mem'],
#                 ['R', 'SV', 'strong', 120, 'wave', 1.5, 3000, 'Mem'],
#                 ['R', 'SV', 'strong', 160, 'wave', 1.5, 3000, 'Mem'],
#                 ['R', 'SV', 'weak', 20, 'wave', 1.5, 3000, 'noMem'],
#                 ['R', 'SV', 'weak', 40, 'wave', 1.5, 3000, 'noMem'],
#                 ['R', 'SV', 'weak', 60, 'wave', 1.5, 3000, 'noMem'],
#                 ['R', 'SV', 'weak', 80, 'wave', 1.5, 3000, 'noMem'],
#                 ['R', 'SV', 'weak', 100, 'wave', 1.5, 3000, 'noMem'],
#                 ['R', 'SV', 'weak', 120, 'wave', 1.5, 3000, 'Mem'],
#                 ['R', 'SV', 'weak', 160, 'wave', 1.5, 3000, 'noMem']]

#    caseName = 'analytical_t1_5_spaceVariation'
#    caseArray =  [['R', 'SV', 'weak', 6, 'analytical', 1.5, 3000, 'noMem', (3, 3)],
#                  ['R', 'SV', 'weak', 10, 'analytical', 1.5, 3000, 'noMem', (3, 3)],
#                  ['R', 'SV', 'weak', 14, 'analytical', 1.5, 3000, 'noMem', (3, 3)]]
#    caseArray =  [['R', 'SV', 'weak', 4, 'analytical', 1.5, 3000, 'noMem', (3, 3)],
#                 ['R', 'SV', 'weak', 6, 'analytical', 1.5, 3000, 'noMem', (3, 3)],
    #             ['R', 'SV', 'weak', 8, 'analytical', 1.5, 3000, 'noMem', (3, 3)],
#                 ['R', 'SV', 'weak', 10, 'analytical', 1.5, 3000, 'noMem', (3, 3)],
#                 ['R', 'SV', 'weak', 12, 'analytical', 1.5, 3000, 'noMem', (3, 3)],
#                 ['R', 'SV', 'weak', 14, 'analytical', 1.5, 3000, 'noMem', (3, 3)],
#                 ['R', 'SV', 'weak', 16, 'analytical', 1.5, 3000, 'noMem', (3, 3)]]
#                 ['R', 'SV', 'weak', 8, 'analytical', 1.5, 3000, 'noMem', (2, 2)],
#                 ['R', 'SV', 'weak', 12, 'analytical', 1.5, 3000, 'noMem', (2, 2)],
#                 ['R', 'SV', 'weak', 16, 'analytical', 1.5, 3000, 'noMem', (2, 2)],
#                 ['R', 'SV', 'weak', 20, 'analytical', 1.5, 3000, 'noMem', (2, 2)],
#                 ['R', 'SV', 'weak', 24, 'analytical', 1.5, 3000, 'noMem', (2, 2)],
#                 ['R', 'SV', 'weak', 28, 'analytical', 1.5, 3000, 'noMem', (2, 2)]]
    #caseArray = [['R', 'SV', 'weak', 20, 'analytical', 1.5, 3000, 'noMem', (2, 2)],
    #            ['R', 'SV', 'weak', 30, 'analytical', 1.5, 3000, 'noMem', (2, 2)],
    #            ['R', 'SV', 'weak', 40, 'analytical', 1.5, 3000, 'noMem', (2, 2)]]
                # ['R', 'SV', 'weak', 60, 'analytical', 1.5, 3000, 'noMem', (2, 2)],
                # ['R', 'SV', 'weak', 80, 'analytical', 1.5, 3000, 'noMem', (2, 2)],
                # ['R', 'SV', 'weak', 100, 'analytical', 1.5, 3000, 'noMem', (2, 2)]]
#    caseArray = [['R', 'SV', 'weak', 20, 'analytical', 1.5, 3000, 'noMem', (2, 1)],
#                 ['R', 'SV', 'weak', 30, 'analytical', 1.5, 3000, 'noMem', (2, 1)],
#                 ['R', 'SV', 'weak', 40, 'analytical', 1.5, 3000, 'noMem', (2, 1)],
#                 ['R', 'SV', 'weak', 60, 'analytical', 1.5, 3000, 'noMem', (2, 1)],
#                 ['R', 'SV', 'weak', 80, 'analytical', 1.5, 3000, 'noMem', (2, 1)],
#                 ['R', 'SV', 'weak', 100, 'analytical', 1.5, 3000, 'noMem', (2, 1)],
#                 ['R', 'SV', 'weak', 20, 'analytical', 1.5, 3000, 'noMem', (1, 2)],
#                 ['R', 'SV', 'weak', 30, 'analytical', 1.5, 3000, 'noMem', (1, 2)],
#                 ['R', 'SV', 'weak', 40, 'analytical', 1.5, 3000, 'noMem', (1, 2)],
#                 ['R', 'SV', 'weak', 60, 'analytical', 1.5, 3000, 'noMem', (1, 2)],
#                 ['R', 'SV', 'weak', 80, 'analytical', 1.5, 3000, 'noMem', (1, 2)],
#                 ['R', 'SV', 'weak', 100, 'analytical', 1.5, 3000, 'noMem', (1, 2)]]
    # For these analytic sims leave without order and will resort to 1st order for P and RT
#    caseArray = [['R', 'SV', 'weak', 60, 'analytical', 1.5, 3000, 'noMem'],
#                ['R', 'SV', 'weak', 80, 'analytical', 1.5, 3000, 'noMem'],
#                ['R', 'SV', 'weak', 100, 'analytical', 1.5, 3000, 'noMem'],
#                ['R', 'SV', 'weak', 120, 'analytical', 1.5, 3000, 'noMem'],
#                ['R', 'SV', 'weak', 140, 'analytical', 1.5, 3000, 'noMem'],
#                ['R', 'SV', 'weak', 160, 'analytical', 1.5, 3000, 'noMem']]
#                ['R', 'SE', 'weak', 60, 'analytical', 1.5, 6000, 'noMem'],
#                ['R', 'SE', 'weak', 80, 'analytical', 1.5, 6000, 'noMem'],
#                ['R', 'SE', 'weak', 100, 'analytical', 1.5, 6000, 'noMem'],
#                ['R', 'SE', 'weak', 120, 'analytical', 1.5, 6000, 'noMem'],
#                ['R', 'SE', 'weak', 140, 'analytical', 1.5, 6000, 'noMem'],
#                ['R', 'SE', 'weak', 160, 'analytical', 1.5, 6000, 'noMem'],
#                ['R', 'SM', 'weak', 60, 'analytical', 1.5, 6000, 'noMem'],
#                ['R', 'SM', 'weak', 80, 'analytical', 1.5, 6000, 'noMem'],
#                ['R', 'SM', 'weak', 100, 'analytical', 1.5, 6000, 'noMem'],
#                ['R', 'SM', 'weak', 120, 'analytical', 1.5, 6000, 'noMem'],
#                ['R', 'SM', 'weak', 140, 'analytical', 1.5, 6000, 'noMem'],
#                ['R', 'SM', 'weak', 160, 'analytical', 1.5, 6000, 'noMem']]

    #    caseName = 'analytical_t10_0_schemeVariation_3rdOrder'
#    caseArray = [['R', 'SV', 'weak', 80, 'analytical', 10.0, 20000, 'noMem', (3, 3)],
#    caseArray = [['R', 'SE', 'weak', 80, 'analytical', 20.0, 80000, 'noMem', (3, 3)]]
#                ['R', 'IE', 'weak', 80, 'analytical', 10.0, 40000, 'noMem', (3, 3)],
#    caseArray = [['R', 'EH', 'weak', 80, 'analytical', 5.0, 40000, 'noMem', (3, 3)]]
#    caseArray = [['R', 'SM', 'weak', 80, 'analytical', 10.0, 40000, 'noMem', (3, 3)]]

#    caseName = 'IC_t4_5_timeVariation'
#    caseArray = [['R', 'SV', 'weak', 120, 'IC', 4.5, 3000, 'noMem'],
#                ['R', 'SV', 'weak', 120, 'IC', 4.5, 4500, 'noMem'],
#                ['R', 'SV', 'weak', 120, 'IC', 4.5, 6000, 'noMem'],
#                ['R', 'SV', 'weak', 120, 'IC', 4.5, 7500, 'noMem'],
#                ['R', 'SV', 'weak', 120, 'IC', 4.5, 9000, 'Mem'],
#    caseArray = [['R', 'SM', 'weak', 120, 'IC', 4.5, 3000, 'noMem'],
#                ['R', 'SM', 'weak', 120, 'IC', 4.5, 4500, 'noMem'],
#                ['R', 'SM', 'weak', 120, 'IC', 4.5, 6000, 'noMem'],
#                ['R', 'SM', 'weak', 120, 'IC', 4.5, 7500, 'noMem'],
#                ['R', 'SM', 'weak', 120, 'IC', 4.5, 9000, 'noMem']]

#    caseName = 'analytical_t1_5_timeVariation'
#    caseArray = [['R', 'SV', 'weak', 80, 'analytical', 1.5, 3000, 'Mem', (3, 3)],
#                 ['R', 'SV', 'weak', 80, 'analytical', 1.5, 4250, 'noMem', (3, 3)],
#                 ['R', 'SV', 'weak', 80, 'analytical', 1.5, 3500, 'noMem', (3, 3)],
#                 ['R', 'SV', 'weak', 80, 'analytical', 1.5, 5000, 'noMem', (3, 3)],
#                 ['R', 'SV', 'weak', 80, 'analytical', 1.5, 6000, 'noMem', (3, 3)],
#                 ['R', 'SE', 'weak', 80, 'analytical', 1.5, 3000, 'noMem', (3, 3)],
#                 ['R', 'SE', 'weak', 80, 'analytical', 1.5, 3500, 'noMem', (3, 3)],
#                 ['R', 'SE', 'weak', 80, 'analytical', 1.5, 4250, 'noMem', (3, 3)],
#                 ['R', 'SE', 'weak', 80, 'analytical', 1.5, 5000, 'noMem', (3, 3)],
#                 ['R', 'SE', 'weak', 80, 'analytical', 1.5, 6000, 'noMem', (3, 3)]]

#    caseName = 'IC_SV_t_20_0_singleRun'
#    caseArray = [['R', 'SV', 'weak', 120, 'IC', 20, 40000, 'Mem']]

#    caseName = 'IC_square_IC_SV_t8_0'
#    caseArray = [['S_1C', 'SV', 'weak', 60, 'IC', 8, 16000, 'Mem']]

#    caseName = 'IC4_SV_singleRun'
#    caseArray = [['R', 'SV', 'weak', 120, 'IC4', 1.5, 3000, 'Mem']]

# The above are confirmed cases for paper
# The below are temporary cases
#    caseName = 'analytical_t1_5_visualisation'
#    caseArray = [['R', 'SV', 'weak', 80, 'analytical', 1.5, 3000, 'Mem', (3, 3)]]

    #   timeIntScheme = 'SV'

# The below case may not be needed, only need either IC or IC4
#    caseName = 'IC4_{}_t4_5_timeVariation'.format(timeIntScheme)
#    caseArray = [['R', timeIntScheme, 'weak', 120, 'IC4', 4.5, 1500, 'Mem'],
#                ['R', timeIntScheme, 'weak', 120, 'IC4', 4.5, 3000, 'Mem'],
#                ['R', timeIntScheme, 'weak', 120, 'IC4', 4.5, 4500, 'Mem'],
#                ['R', timeIntScheme, 'weak', 120, 'IC4', 4.5, 6000, 'Mem'],
#                ['R', timeIntScheme, 'weak', 120, 'IC4', 4.5, 7500, 'Mem'],
#                ['R', timeIntScheme, 'weak', 120, 'IC4', 4.5, 9000, 'Mem']]

    caseName = 'tempSingleRun'
    caseArray = [['R', 'SE', 'weak', 20, 'analytical', 0.5, 1000, 'noMem', (1, 1)]]

    for caseVec in caseArray:
        domainShape = caseVec[0]
        timeIntScheme = caseVec[1]
        dirichletImp = caseVec[2]

        assert len(caseVec) in [7, 8, 9], 'caseArray should be vectors of length 7 or 8'
        # ------------------------------# Setup Directories #-------------------------------#

        # Create output dir
        outputDir = os.path.join('output', caseName)
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

        ANALYTICAL_BOOL = (caseVec[4] == 'analytical')
        # add 'IC', 'wave', or 'analytical' to the subDir name to denote interconnection
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


        # solve the wave equation
        H_array, numCells = wave_2D_solve(tFinal, numSteps, outputSubDir,
                                    nx, xLength, yLength,
                                    domainShape=domainShape, timeIntScheme=timeIntScheme,
                                    dirichletImp=dirichletImp,
                                    K_wave=K_wave, rho=rho, interConnection=IC,
                                    analytical=ANALYTICAL_BOOL, saveP=saveP, basis_order=basis_order)

        # -------------------------------# Set up output and plotting #---------------------------------#

        np.save(os.path.join(outputSubDir, 'H_array.npy'), H_array)

        numCells_save = np.zeros((1))
        numCells_save[0] = numCells
        np.save(os.path.join(outputSubDir, 'numCells.npy'), numCells_save)

    if rank == 0:
        totalTime = time.time() - tic
        print('All Simulations finished in {} seconds'.format(totalTime))


