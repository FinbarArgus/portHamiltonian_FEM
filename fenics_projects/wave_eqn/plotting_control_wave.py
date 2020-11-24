import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import paperPlotSetup

paperPlotSetup.Setup_Plot(3)
plotPNG = False
plotEPS = True

caseName = 'passivity_control_IC_t4_5'
outputBaseDir = 'control_output'
outputDir = os.path.join(outputBaseDir, caseName)
if not os.path.exists(outputDir):
    print('The output dir {} doesn\'t exist'.format(outputDir))
    quit()

plotDir = os.path.join(outputDir, 'plots')
if not os.path.exists(plotDir):
    os.mkdir(plotDir)

caseArray = [['R', 'SM', 'weak', 40, 'IC', 4.0, 4000]]
outputSubDirArray = []
for caseVec in caseArray:
    subDir = caseVec[0] + '_' + caseVec[1] + '_' +caseVec[2] + \
             '_nx' + str(caseVec[3]) + '_' + caseVec[4] + '_t' + \
             str(caseVec[5]).replace('.','_') + '_steps' + str(caseVec[6])

    outputSubDirArray.append(os.path.join(outputDir, subDir))

tFinal = caseArray[0][5]
# Plot the hamiltonian
fig, ax = plt.subplots(1, 1)
ax.set_xlim(0, tFinal)
# ax.set_ylim(-0.01, 0.01)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Hamiltonian [J]')
for count, dir in enumerate(outputSubDirArray):
    dataArray = np.load(os.path.join(dir, 'H_array.npy'))
    plt.plot(dataArray[:, 0], dataArray[:, 1], lw=0.5, color='r', linestyle='-',
             label='Interconnection SV')
ax.legend()
if plotPNG:
    plt.savefig(os.path.join(plotDir, 'ICHamiltonian.png'), dpi=500)
if plotEPS:
    plt.savefig(os.path.join(plotDir, 'ICHamiltonian.eps'))
plt.show()

plt.close(fig)

