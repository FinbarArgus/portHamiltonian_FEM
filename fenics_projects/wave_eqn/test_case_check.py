import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
import os


# This script checks the output of the test_cases run in
def check_cases(caseName):
    # fast test cases
    if caseName == 'test_cases':
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
        # test cases for cases that are in the paper
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

    outputDir = os.path.join('test_output', caseName)
    if not os.path.exists(outputDir):
        print('The output dir {} doesn\'t exist'.format(outputDir))
        exit()

    for caseVec in caseArray:
        domainShape = caseVec[0]
        timeIntScheme = caseVec[1]
        dirichletImp = caseVec[2]

        assert len(caseVec) in [7, 8, 9], 'caseArray should be vectors of length 7, 8, or 9'
        # ------------------------------# Setup Directories #-------------------------------#

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
        subDir = subDir + '_t' + str(caseVec[5]).replace('.', '_') + \
                 '_steps' + str(caseVec[6])
        if len(caseVec) > 8:
            basis_order = caseVec[8]
            subDir = subDir + '_P{}_RT{}'.format(basis_order[0], basis_order[1])

        outputSubDir = os.path.join(outputDir, subDir)
        if not os.path.exists(outputSubDir):
            print('subdirectory of {} doesn\t exits'.format(outputSubDir))

        # subdir for ground truth
        groundTruthDir = os.path.join(outputDir, 'ground_truth')
        groundTruthSubDir = os.path.join(groundTruthDir, subDir)

        data_array = np.load(os.path.join(outputSubDir, 'H_array.npy'))
        gt_array = np.load(os.path.join(groundTruthSubDir, 'H_array.npy'))

        eps = 1e-10
        # Get difference in Hamiltonian percent

        hamDiff = max(abs((data_array[:, 1] - gt_array[:, 1])))
        resDiff = max(abs((data_array[:, 2] - gt_array[:, 2])))
        hamDiffPerc = 100*max(abs((data_array[:, 1] - gt_array[:, 1])/(abs(gt_array[:, 1])+eps)))
        resDiffPerc = 100*max(abs((data_array[:, 2] - gt_array[:, 2])/(abs(gt_array[:, 2])+eps)))
        if ANALYTICAL_BOOL:
            analyticDiff = max(abs(data_array[:, 8] - gt_array[:, 8]))
            analyticDiffPerc = 100*max(abs((data_array[:, 8] - gt_array[:, 8])/(abs(gt_array[:, 8]) + eps)))

        if hamDiffPerc > 1e-5 or resDiffPerc > 1e-5 :
            print('{} test FAILED'.format(subDir))
            print('Hamiltonian max diff    : {}%'.format(hamDiffPerc))
            print('Energy res max diff     : {}%'.format(resDiffPerc))
            print('Hamiltonian total diff  : {}'.format(hamDiff))
            print('Energy res total diff   : {}'.format(resDiff))
            if ANALYTICAL_BOOL:
                print('analytic diff of        : {}%'.format(analyticDiffPerc))
                print('analytic total diff of  : {}'.format(analyticDiff))

        else:
            if ANALYTICAL_BOOL:
                if analyticDiffPerc > 1e-7:
                    print('{} test FAILED'.format(subDir))
                else:
                    print('{} test passed'.format(subDir))

                print('Hamiltonian max diff    : {}%'.format(hamDiffPerc))
                print('Energy res max diff     : {}%'.format(resDiffPerc))
                print('Hamiltonian total diff  : {}'.format(hamDiff))
                print('Energy res total diff   : {}'.format(resDiff))
                print('analytic diff of        : {}%'.format(analyticDiffPerc))
                print('analytic total diff of  : {}'.format(analyticDiff))

            else:
                print('{} test passed'.format(subDir))
                print('Hamiltonian max diff    : {}%'.format(hamDiffPerc))
                print('Energy res max diff     : {}%'.format(resDiffPerc))
                print('Hamiltonian total diff  : {}'.format(hamDiff))
                print('Energy res total diff   : {}'.format(resDiff))

if __name__ == '__main__':
    caseName = 'test_cases'
    # caseName = 'test_cases_long'
    check_cases(caseName)















