import numpy as np
import time
import os
from calculate_eigen import *
from mpi4py import MPI

print('starting script and timer')
tic = time.time()

# currently parallel isn't working for eigenvalue calcs
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
# -------------------------------# Define params #---------------------------------#

# stiffness
K_wave = 3.0  # 3.0
# density
rho = 2.0  # 2.0

caseName = 'eigenValue_spaceVariation'
CALCULATE_EIGENS = True
# caseArray = [['R', 'SE', 'weak', 5, 'analytical', 0.5, 1000, 'noMem', (1, 1)],
#              ['R', 'SE', 'weak', 10, 'analytical', 0.5, 1000, 'noMem', (1, 1)],
#              ['R', 'SE', 'weak', 15, 'analytical', 0.5, 1000, 'noMem', (1, 1)],
#              ['R', 'SE', 'weak', 20, 'analytical', 0.5, 1000, 'noMem', (1, 1)],
#              ['R', 'SE', 'weak', 25, 'analytical', 0.5, 1000, 'noMem', (1, 1)],
#              ['R', 'SE', 'weak', 30, 'analytical', 0.5, 1000, 'noMem', (1, 1)]]

caseArray = [['R', 'SE', 'weak', 5, 'analytical', 0.5, 1000, 'noMem', (2, 2)],
             ['R', 'SE', 'weak', 10, 'analytical', 0.5, 1000, 'noMem', (2, 2)],
             ['R', 'SE', 'weak', 15, 'analytical', 0.5, 1000, 'noMem', (2, 2)],
             ['R', 'SE', 'weak', 20, 'analytical', 0.5, 1000, 'noMem', (2, 2)],
             ['R', 'SE', 'weak', 25, 'analytical', 0.5, 1000, 'noMem', (2, 2)]]

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
    subDir = subDir + '_t' + str(caseVec[5]).replace('.', '_') + \
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
        yLength = 0.25  # 0.25
    elif caseVec[0] == 'S_1C':
        yLength = 0.1  # 0.25

    # set final time and number of steps
    tFinal = caseVec[5]
    numSteps = caseVec[6]

    # whether we want to save a p xdf file
    saveP = True
    if len(caseVec) > 7:
        if caseVec[7] == 'noMem':
            saveP = False


    # solve the wave equation
    out_array, numCells = calculate_eigen(tFinal, numSteps, outputSubDir,
                                      nx, xLength, yLength,
                                      domainShape=domainShape, timeIntScheme=timeIntScheme,
                                      dirichletImp=dirichletImp,
                                      K_wave=K_wave, rho=rho,
                                      analytical=ANALYTICAL_BOOL, basis_order=basis_order)

    # -------------------------------# Set up output and plotting #---------------------------------#

    np.save(os.path.join(outputSubDir, 'out_array.npy'), out_array)

    numCells_save = np.zeros((1))
    numCells_save[0] = numCells
    np.save(os.path.join(outputSubDir, 'numCells.npy'), numCells_save)

if rank == 0:
    totalTime = time.time() - tic
    print('All Simulations finished in {} seconds'.format(totalTime))

