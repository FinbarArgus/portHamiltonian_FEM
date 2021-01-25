import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import paperPlotSetup
from tabulate import tabulate

paperPlotSetup.Setup_Plot(3)
plotPNG = False
plotEPS = True

# -------------------------------# Access Data #---------------------------------#

caseName = 'strong_and_weak_method_variation'
outputDir = os.path.join('output', caseName)
if not os.path.exists(outputDir):
    print('The output dir {} doesn\'t exist'.format(outputDir))
    quit()

plotDir = os.path.join(outputDir, 'plots')
if not os.path.exists(plotDir):
    os.mkdir(plotDir)

caseArray = [['R', 'IE', 'weak', 80, 'wave', 1.5, 3000],
             ['R', 'IE', 'strong', 80, 'wave', 1.5, 3000],
             ['R', 'EE', 'weak', 80, 'wave', 1.5, 3000],
             ['R', 'EE', 'strong', 80, 'wave', 1.5, 3000],
             ['R', 'SE', 'weak', 80, 'wave', 1.5, 3000],
             ['R', 'SE', 'strong', 80, 'wave', 1.5, 3000],
             ['R', 'EH', 'weak', 80, 'wave', 1.5, 3000],
             ['R', 'EH', 'strong', 80, 'wave', 1.5, 3000],
             ['R', 'SV', 'weak', 80, 'wave', 1.5, 3000],
             ['R', 'SV', 'strong', 80, 'wave', 1.5, 3000],
             ['R', 'SM', 'weak', 80, 'wave', 1.5, 3000],
             ['R', 'SM', 'strong', 80, 'wave', 1.5, 3000]]

colorVec= ['b', 'b', 'r', 'r', 'c', 'c', 'g', 'g', 'k', 'k', 'orange', 'orange']
lineStyleArray = ['-', (0, (5, 10)), '-', (0, (5, 10)), '-',
                  (0, (5, 10)), '-', (0, (5, 10)), '-',
                  (0, (5, 10)), '-', (0, (5, 10))]


outputSubDirArray = []
for caseVec in caseArray:
    subDir = caseVec[0] + '_' + caseVec[1] + '_' +caseVec[2] + \
             '_nx' + str(caseVec[3]) + '_' + caseVec[4] + '_t' + \
             str(caseVec[5]).replace('.','_') + '_steps' + str(caseVec[6])
    outputSubDirArray.append(os.path.join(outputDir, subDir))

# -------------------------------# Plotting #---------------------------------#

# ------------# Plot Hamiltonian and energy residual over time#---------------#
tFinal = caseArray[0][5]
# Plot the energy residual
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(17.4/2.54, 23.4/2.54))
ax.set_xlim(0, tFinal)
ax.set_ylim(-0.02, 0.03)
ax2.set_xlabel('Time [s]')
ax.set_ylabel('Energy Residual [J]')
ax2.set_ylabel('Energy Residual [J]')
ax.text(0.18,0.025,'(a)',fontsize=20)
ax2.text(0.18,0.0004,'(b)',fontsize=20)
for count, dir in enumerate(outputSubDirArray):
    dataArray = np.load(os.path.join(dir, 'H_array.npy'))
    ax.plot(dataArray[:, 0], dataArray[:, 2], lw=0.5, color=colorVec[count], linestyle=lineStyleArray[count],
             label='{} {}'.format(caseArray[count][1], caseArray[count][2]))
    ax2.plot(dataArray[:, 0], dataArray[:, 2], lw=0.5, color=colorVec[count], linestyle=lineStyleArray[count],
            label='{} {}'.format(caseArray[count][1], caseArray[count][2]))

ax.legend(loc=1, ncol=3)
#plt.show()
# set ylimits for zoom
ax2.set_xlim(0, tFinal)
ax2.set_ylim(-0.0005, 0.0005)
if plotPNG:
    plt.savefig(os.path.join(plotDir, 'EnergyRes.png'), dpi=500)
if plotEPS:
    plt.savefig(os.path.join(plotDir, 'EnergyRes.eps'))
plt.close(fig)

# Plot the hamiltonian
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(17.4/2.54, 23.4/2.54))
ax.set_xlim(0, tFinal)
ax.set_ylim(0.0, 2.2)
ax2.set_xlabel('Time [s]')
ax.set_ylabel('Hamiltonian [J]')
ax2.set_ylabel('Hamiltonian [J]')
ax.text(0.18,2.03,'(a)',fontsize=20)
ax2.text(0.18,1.91582,'(b)',fontsize=20)
for count, dir in enumerate(outputSubDirArray):
    dataArray = np.load(os.path.join(dir, 'H_array.npy'))
    ax.plot(dataArray[:, 0], dataArray[:, 1], lw=0.5, color=colorVec[count], linestyle=lineStyleArray[count],
             label='{} {}'.format(caseArray[count][1], caseArray[count][2]))
    ax2.plot(dataArray[:, 0], dataArray[:, 1], lw=0.5, color=colorVec[count], linestyle=lineStyleArray[count],
            label='{} {}'.format(caseArray[count][1], caseArray[count][2]))
ax.legend(ncol=3)
# overwrite limits if we want to zoom
ax2.set_xlim(0.0, tFinal)
ax2.set_ylim(1.914, 1.916)
if plotPNG:
    plt.savefig(os.path.join(plotDir, 'Hamiltonian.png'), dpi=500)
if plotEPS:
    plt.savefig(os.path.join(plotDir, 'Hamiltonian.eps'))
plt.close(fig)

# ------------# Plotting boundary energy residual for multiple cell numbers #---------------#

caseName = 'strong_and_weak_spatial_variation'
outputDir = os.path.join('output', caseName)
if not os.path.exists(outputDir):
    print('The output dir {} doesn\'t exist'.format(outputDir))
    quit()

plotDir = os.path.join(outputDir, 'plots')
if not os.path.exists(plotDir):
    os.mkdir(plotDir)

caseArray = [['R', 'SV', 'strong', 20, 'wave', 1.5, 3000],
             ['R', 'SV', 'strong', 40, 'wave', 1.5, 3000],
             ['R', 'SV', 'strong', 60, 'wave', 1.5, 3000],
             ['R', 'SV', 'strong', 80, 'wave', 1.5, 3000],
             ['R', 'SV', 'strong', 100, 'wave', 1.5, 3000],
             ['R', 'SV', 'strong', 120, 'wave', 1.5, 3000],
             ['R', 'SV', 'strong', 160, 'wave', 1.5, 3000],
             ['R', 'SV', 'weak', 20, 'wave', 1.5, 3000],
             ['R', 'SV', 'weak', 40, 'wave', 1.5, 3000],
             ['R', 'SV', 'weak', 60, 'wave', 1.5, 3000],
             ['R', 'SV', 'weak', 80, 'wave', 1.5, 3000],
             ['R', 'SV', 'weak', 100, 'wave', 1.5, 3000],
             ['R', 'SV', 'weak', 120, 'wave', 1.5, 3000],
             ['R', 'SV', 'weak', 160, 'wave', 1.5, 3000]]

colorVec= ['r', 'orange', 'y', 'g', 'c', 'b', 'm', 'r', 'orange', 'y', 'g', 'c', 'b', 'm']
lineStyleArray = ['--', '--', '--', '--', '--', '--', '--', '-', '-', '-', '-', '-', '-', '-']

outputSubDirArray = []
for caseVec in caseArray:
    subDir = caseVec[0] + '_' + caseVec[1] + '_' +caseVec[2] + \
             '_nx' + str(caseVec[3]) + '_' + caseVec[4] + '_t' + \
             str(caseVec[5]).replace('.','_') + '_steps' + str(caseVec[6])
    outputSubDirArray.append(os.path.join(outputDir, subDir))

fig, ax = plt.subplots(1, 1)
ax.set_xlim(0, tFinal)
ax.set_ylim(0, 0.1)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Energy Residual [J]')
for count, dir in enumerate(outputSubDirArray):
    dataArray = np.load(os.path.join(dir, 'H_array.npy'))
    numCells = np.load(os.path.join(dir, 'numCells.npy'))[0]
    plt.plot(dataArray[:, 0], dataArray[:, 2], lw=0.5, color=colorVec[count], linestyle=lineStyleArray[count],
             label='{} {}, numCells = {}'.format(caseArray[count][1], caseArray[count][2], numCells))
    if count == 6:
        #only plot strong boundary imp
        break

ax.legend(loc='center right')
#plt.show()
if plotPNG:
    plt.savefig(os.path.join(plotDir, 'EnergyResDistForNumCells.png'), dpi=500)
if plotEPS:
    plt.savefig(os.path.join(plotDir, 'EnergyResDistForNumCells.eps'))
plt.close(fig)

# -----# Plot the final boundary energy residual against number of cells #-------#

fig, ax = plt.subplots(1, 1)
ax.set_xlim(0, 14000)
ax.set_ylim(0, 0.16)
ax.set_xlabel('number of Elements')
ax.set_ylabel('Energy Residual [J]')
numCell_Res = np.zeros((len(caseArray), 4))
for count, dir in enumerate(outputSubDirArray):
    dataArray = np.load(os.path.join(dir, 'H_array.npy'))
    numCells = np.load(os.path.join(dir, 'numCells.npy'))[0]
    avCellSize = 0.25/numCells
    avCellLength = np.sqrt(avCellSize)
    numCell_Res[count, :] = [numCells, avCellSize, avCellLength, np.sqrt(np.square(dataArray[-1,2]))]

plt.plot(numCell_Res[0:7, 0], numCell_Res[0:7, 3], lw=1.5, color='b', linestyle='', marker='o',
        label='SV strong')
plt.plot(numCell_Res[7:14, 0], numCell_Res[7:14, 3], lw=1.5, color='r', linestyle='', marker='x',
         label='SV weak')

ax.legend()
plt.grid()
#plt.show()
if plotPNG:
    plt.savefig(os.path.join(plotDir, 'EnergyResVsNumCells.png'), dpi=500)
if plotEPS:
    plt.savefig(os.path.join(plotDir, 'EnergyResVsNumCells.eps'))
ax.set_xlim(1e2, 1e5)
ax.set_ylim(1e-5, 1)
plt.xscale('log')
plt.yscale('log')
if plotPNG:
    plt.savefig(os.path.join(plotDir, 'EnergyResVsNumCellsLogScale.png'), dpi=500)
if plotEPS:
    plt.savefig(os.path.join(plotDir, 'EnergyResVsNumCellsLogScale.eps'))
plt.close(fig)

#Plot against average cell size
fig, ax = plt.subplots(1, 1)
ax.set_xlim(0, 0.0014)
ax.set_ylim(0, 0.16)
ax.set_xlabel('Average Element Size [$m^2$]')
ax.set_ylabel('Energy Residual [J]')
plt.plot(numCell_Res[0:7, 1], numCell_Res[0:7, 3], lw=1.5, color='b', linestyle='', marker='o',
         label='SV strong')
plt.plot(numCell_Res[7:14, 1], numCell_Res[7:14, 3], lw=1.5, color='r', linestyle='', marker='x',
         label='SV weak')

ax.legend()
plt.grid()
#plt.show()
if plotPNG:
    plt.savefig(os.path.join(plotDir, 'EnergyResVsAvCellSize.png'), dpi=500)
if plotEPS:
    plt.savefig(os.path.join(plotDir, 'EnergyResVsAvCellSize.eps'))
ax.set_xlim(1e-5, 1e-2)
ax.set_ylim(1e-5, 1)
plt.xscale('log')
plt.yscale('log')
if plotPNG:
    plt.savefig(os.path.join(plotDir, 'EnergyResVsAvCellSizeLogScale.png'), dpi=500)
if plotEPS:
    plt.savefig(os.path.join(plotDir, 'EnergyResVsAvCellSizeLogScale.eps'))
plt.close(fig)

#Plot against average cell Length
fig, ax = plt.subplots(1, 1)
ax.set_xlim(0, 0.04)
ax.set_ylim(0, 0.16)
ax.set_xlabel('Characteristic Element Length [$m$]')
ax.set_ylabel('Energy Residual [J]')
plt.plot(numCell_Res[0:7, 2], numCell_Res[0:7, 3], lw=1.5, color='b', linestyle='', marker='o',
         label='SV strong')
plt.plot(numCell_Res[7:14, 2], numCell_Res[7:14, 3], lw=1.5, color='r', linestyle='', marker='x',
         label='SV weak')

#create quadratic line to plot
xQuad = np.linspace(numCell_Res[0,2], numCell_Res[6,2], 1000)
scale = 100
yQuad = scale*np.square(xQuad)
plt.plot(xQuad, yQuad, lw=1.0, color='k',linestyle='--', label='Quadratic Trend ($100x^2$)')

ax.legend()
plt.grid()
#plt.show()
if plotPNG:
    plt.savefig(os.path.join(plotDir, 'EnergyResVsAvCellLength.png'), dpi=500)
if plotEPS:
    plt.savefig(os.path.join(plotDir, 'EnergyResVsAvCellLength.eps'))
ax.set_xlim(1e-3, 1e-1)
ax.set_ylim(1e-5, 1)
plt.xscale('log')
plt.yscale('log')
if plotPNG:
    plt.savefig(os.path.join(plotDir, 'EnergyResVsAvCellLengthLogScale.png'), dpi=500)
if plotEPS:
    plt.savefig(os.path.join(plotDir, 'EnergyResVsAvCellLengthLogScale.eps'))
plt.close(fig)

strongGradient = (np.log(numCell_Res[6,3]) - np.log(numCell_Res[1,3]))/ \
                 (np.log(numCell_Res[6,2]) - np.log(numCell_Res[1,2]))

# -----# Plot the analytic error against characteristic cell length for wave model#-------#

caseName = 'analytical_t1_5_spaceVariation'
outputDir = os.path.join('output', caseName)
if not os.path.exists(outputDir):
    print('The output dir {} doesn\'t exist'.format(outputDir))
    quit()

plotDir = os.path.join(outputDir, 'plots')
if not os.path.exists(plotDir):
    os.mkdir(plotDir)

caseArray = [['R', 'SV', 'weak', 20, 'analytical', 1.5, 3000, (1, 1)],
             ['R', 'SV', 'weak', 30, 'analytical', 1.5, 3000, (1, 1)],
             ['R', 'SV', 'weak', 40, 'analytical', 1.5, 3000, (1, 1)],
             ['R', 'SV', 'weak', 60, 'analytical', 1.5, 3000, (1, 1)],
             ['R', 'SV', 'weak', 80, 'analytical', 1.5, 3000, (1, 1)],
             ['R', 'SV', 'weak', 100, 'analytical', 1.5, 3000, (1, 1)],
             ['R', 'SV', 'weak', 8, 'analytical', 1.5, 3000, (2, 2)],
             ['R', 'SV', 'weak', 12, 'analytical', 1.5, 3000, (2, 2)],
             ['R', 'SV', 'weak', 16, 'analytical', 1.5, 3000, (2, 2)],
             ['R', 'SV', 'weak', 20, 'analytical', 1.5, 3000, (2, 2)],
             ['R', 'SV', 'weak', 24, 'analytical', 1.5, 3000, (2, 2)],
             ['R', 'SV', 'weak', 30, 'analytical', 1.5, 3000, (2, 2)],
             ['R', 'SV', 'weak', 20, 'analytical', 1.5, 3000, (2, 1)],
             ['R', 'SV', 'weak', 30, 'analytical', 1.5, 3000, (2, 1)],
             ['R', 'SV', 'weak', 40, 'analytical', 1.5, 3000, (2, 1)],
             ['R', 'SV', 'weak', 60, 'analytical', 1.5, 3000, (2, 1)],
             ['R', 'SV', 'weak', 80, 'analytical', 1.5, 3000, (2, 1)],
             ['R', 'SV', 'weak', 100, 'analytical', 1.5, 3000, (2, 1)],
             ['R', 'SV', 'weak', 20, 'analytical', 1.5, 3000, (1, 2)],
             ['R', 'SV', 'weak', 30, 'analytical', 1.5, 3000, (1, 2)],
             ['R', 'SV', 'weak', 40, 'analytical', 1.5, 3000, (1, 2)],
             ['R', 'SV', 'weak', 60, 'analytical', 1.5, 3000, (1, 2)],
             ['R', 'SV', 'weak', 80, 'analytical', 1.5, 3000, (1, 2)],
             ['R', 'SV', 'weak', 100, 'analytical', 1.5, 3000, (1, 2)],
             ['R', 'SV', 'weak', 6, 'analytical', 1.5, 3000, (3, 3)],
             ['R', 'SV', 'weak', 8, 'analytical', 1.5, 3000, (3, 3)],
             ['R', 'SV', 'weak', 10, 'analytical', 1.5, 3000, (3, 3)],
             ['R', 'SV', 'weak', 12, 'analytical', 1.5, 3000, (3, 3)],
             ['R', 'SV', 'weak', 14, 'analytical', 1.5, 3000, (3, 3)],
             ['R', 'SV', 'weak', 16, 'analytical', 1.5, 3000, (3, 3)]]

#             ['R', 'SE', 'weak', 60, 'analytical', 1.5, 6000],
#             ['R', 'SE', 'weak', 80, 'analytical', 1.5, 6000],
#             ['R', 'SE', 'weak', 100, 'analytical', 1.5, 6000],
#             ['R', 'SE', 'weak', 120, 'analytical', 1.5, 6000],
#             ['R', 'SE', 'weak', 140, 'analytical', 1.5, 6000],
#             ['R', 'SE', 'weak', 160, 'analytical', 1.5, 6000]]

outputSubDirArray = []
for caseVec in caseArray:
    subDir = caseVec[0] + '_' + caseVec[1] + '_' +caseVec[2] + \
             '_nx' + str(caseVec[3]) + '_' + caseVec[4] + '_t' + \
             str(caseVec[5]).replace('.','_') + '_steps' + str(caseVec[6])
    if len(caseVec) > 7:
        subDir = subDir + '_P{}_RT{}'.format(caseVec[7][0], caseVec[7][1])

    outputSubDirArray.append(os.path.join(outputDir, subDir))

numCell_Res = np.zeros((len(caseArray), 4))
for count, dir in enumerate(outputSubDirArray):
    dataArray = np.load(os.path.join(dir, 'H_array.npy'))
    numCells = np.load(os.path.join(dir, 'numCells.npy'))[0]
    avCellSize = 0.25/numCells
    avCellLength = np.sqrt(avCellSize)
    numCell_Res[count, :] = [numCells, avCellSize, avCellLength,
                             np.max(abs(dataArray[:,8]))]

#Plot L2 error against average cell Length
fig, ax = plt.subplots(1, 1)
# ax.set_xlim(0, 0.04)
# ax.set_ylim(0, 0.16)
ax.set_xlabel('Characteristic Element Length [$m$]')
ax.set_ylabel('Max $L^2$ Error Norm')
plt.plot(numCell_Res[0:6, 2], numCell_Res[0:6, 3], lw=1.5, color='r', linestyle='', marker='x',
         label='SV P1RT1'.format(caseArray[0][5]/caseArray[0][6]))
plt.plot(numCell_Res[12:18, 2], numCell_Res[12:18, 3], lw=1.5, color='g', linestyle='', marker='v',
         label='SV P2RT1'.format(caseArray[12][5]/caseArray[12][6]))
plt.plot(numCell_Res[18:24, 2], numCell_Res[18:24, 3], lw=1.5, color='m', linestyle='', marker='^',
         label='SV P1RT2'.format(caseArray[18][5]/caseArray[18][6]))
plt.plot(numCell_Res[6:12, 2], numCell_Res[6:12, 3], lw=1.5, color='b', linestyle='', marker='o',
        label='SV P2RT2'.format(caseArray[6][5]/caseArray[6][6]))
plt.plot(numCell_Res[24:30, 2], numCell_Res[24:30, 3], lw=1.5, color='c', linestyle='', marker='*',
         label='SV P3RT3'.format(caseArray[24][5]/caseArray[24][6]))

# plt.plot(numCell_Res[6:12, 2], numCell_Res[6:12, 3], lw=1.5, color='b', linestyle='', marker='o',
#         label='SE, $\Delta t$ = {:.1E}'.format(caseArray[6][5]/caseArray[6][6]))

# Create a table with tabulate to output in latex
# First create array with values for P1RT1, P2RT2 and P3RT3
tableArray = np.zeros((6, 9))
tableArray[:, 0] = range(6)
tableArray[:, 1] = numCell_Res[0:6, 0]
tableArray[:, 2] = numCell_Res[0:6, 2]
tableArray[:, 3] = numCell_Res[0:6, 3]
tableArray[1:, 4] = (np.log10(numCell_Res[0:5, 3]) - np.log10(numCell_Res[1:6, 3])) / (np.log10(numCell_Res[0:5, 2]) - np.log10(numCell_Res[1:6, 2]))
tableArray[:, 5] = numCell_Res[12:18, 3]
tableArray[1:, 6] = (np.log10(numCell_Res[12:17, 3]) - np.log10(numCell_Res[13:18, 3])) / (np.log10(numCell_Res[12:17, 2]) - np.log10(numCell_Res[13:18, 2]))
tableArray[:, 7] = numCell_Res[18:24, 3]
tableArray[1:, 8] = (np.log10(numCell_Res[18:23, 3]) - np.log10(numCell_Res[19:24, 3])) / (np.log10(numCell_Res[18:23, 2]) - np.log10(numCell_Res[19:24, 2]))

# df = pd.DataFrame(data=tableArray, index=['RL', 'NoE', 'CEL [$m$]', 'L^2', 'OoC', 'L^2', 'OoC', 'L^2', 'OOC'])
headers = ['RL', 'NoE', 'CEL [$m$]', 'L^2', 'OoC', 'L^2', 'OoC', 'L^2', 'OOC']
table1 = tabulate(tableArray, headers=headers, tablefmt='latex', floatfmt=('g', 'g', '.4f', '.2f', '.2f',
                                                                           '.2f', '.2f', '.2f', '.2f'))
print(table1)

# create table for P2RT2
tableArray2 = np.zeros((6, 5))
tableArray2[:, 0] = range(6)
tableArray2[:, 1] = numCell_Res[6:12, 0]
tableArray2[:, 2] = numCell_Res[6:12, 2]
tableArray2[:, 3] = numCell_Res[6:12, 3]
tableArray2[1:, 4] = (np.log10(numCell_Res[6:11, 3]) - np.log10(numCell_Res[7:12, 3])) / (np.log10(numCell_Res[6:11, 2]) - np.log10(numCell_Res[7:12, 2]))

headers2 = ['RL', 'NoE', 'CEL [$m$]', 'L^2', 'OoC']
table2 = tabulate(tableArray2, headers=headers2, tablefmt='latex', floatfmt=('g', 'g', '.4f', '.2f', '.2f'))
print(table2)

# create table for P3RT3
tableArray3 = np.zeros((6, 5))
tableArray3[:, 0] = range(6)
tableArray3[:, 1] = numCell_Res[24:30, 0]
tableArray3[:, 2] = numCell_Res[24:30, 2]
tableArray3[:, 3] = numCell_Res[24:30, 3]
tableArray3[1:, 4] = (np.log10(numCell_Res[24:29, 3]) - np.log10(numCell_Res[25:30, 3])) / (np.log10(numCell_Res[24:29, 2]) - np.log10(numCell_Res[25:30, 2]))

headers3 = ['RL', 'NoE', 'CEL [$m$]', 'L^2', 'OoC']
table3 = tabulate(tableArray3, headers=headers3, tablefmt='latex', floatfmt=('g', 'g', '.4f', '.3f', '.3f'))
print(table3)

#create quadratic line to plot
xQuad = np.linspace(numCell_Res[0,2], numCell_Res[5,2], 1000)
scale = 10000
yQuad = scale*np.square(xQuad)
plt.plot(xQuad, yQuad, lw=1.0, color='k', linestyle='--', label='Quadratic Trend ({}$x^2$)'.format(scale))

xCubic = np.linspace(numCell_Res[6,2], numCell_Res[11,2], 1000)
scale = 20000
yCubic = scale*np.power(xCubic, 3)
plt.plot(xCubic, yCubic , lw=1.0, color='k', linestyle=':', label='Cubic Trend ({}$x^3$)'.format(scale))

xQuartic = np.linspace(numCell_Res[24,2], numCell_Res[29,2], 1000)
scale = 7000
yQuartic = scale*np.power(xQuartic, 4)
plt.plot(xQuartic, yQuartic, lw=1.0, color='k', linestyle='-', label='Quartic Trend ({}$x^4$)'.format(scale))

ax.legend(loc='lower right')
plt.grid()
#plt.show()
if plotPNG:
    plt.savefig(os.path.join(plotDir, 'analyticL2ErrorVsAvCellLength.png'), dpi=500)
if plotEPS:
    plt.savefig(os.path.join(plotDir, 'analyticL2ErrorVsAvCellLength.eps'))
ax.set_xlim(1e-3, 10)
ax.set_ylim(1e-3, 10)
plt.xscale('log')
plt.yscale('log')
if plotPNG:
    plt.savefig(os.path.join(plotDir, 'analyticL2ErrorVsAvCellLengthLogScale.png'), dpi=500)
if plotEPS:
    plt.savefig(os.path.join(plotDir, 'analyticL2ErrorVsAvCellLengthLogScale.eps'))
plt.close(fig)

strongGradient = (np.log(numCell_Res[5,3]) - np.log(numCell_Res[1,3]))/ \
                 (np.log(numCell_Res[5,2]) - np.log(numCell_Res[1,2]))

# print(strongGradient)

# ------------# Plot analytical L2 error for 1.5 second run for all time steps#---------------#

timeIntScheme = 'SV'
tFinal = 1.5
caseName = 'analytical_t{}_timeVariation'.format(tFinal).replace('.','_')
outputDir = os.path.join('output', caseName)
if not os.path.exists(outputDir):
    print('The output dir {} doesn\'t exist'.format(outputDir))
    quit()

plotDir = os.path.join(outputDir, 'plots')
if not os.path.exists(plotDir):
    os.mkdir(plotDir)


caseArray = [['R', 'SE', 'weak', 80, 'analytical', tFinal, 3000, 'noMem'],
            ['R', 'SE', 'weak', 80, 'analytical', tFinal, 3500, 'noMem'],
            ['R', 'SE', 'weak', 80, 'analytical', tFinal, 4250, 'noMem'],
            ['R', 'SE', 'weak', 80, 'analytical', tFinal, 5000, 'noMem'],
            ['R', 'SE', 'weak', 80, 'analytical', tFinal, 6000, 'noMem'],
            ['R', 'SV', 'weak', 80, 'analytical', tFinal, 3000, 'noMem'],
            ['R', 'SV', 'weak', 80, 'analytical', tFinal, 4250, 'noMem'],
            ['R', 'SV', 'weak', 80, 'analytical', tFinal, 3500, 'noMem'],
            ['R', 'SV', 'weak', 80, 'analytical', tFinal, 5000, 'noMem'],
            ['R', 'SV', 'weak', 80, 'analytical', tFinal, 6000, 'noMem']]

colorVec = ['c', 'b', 'g', 'm', 'r', 'k', 'c', 'b', 'g', 'm', 'r', 'k']
lineStyleArray = ['--', '--', '--', '--', '--', '--', '-', '-', '-', '-', '-', '-']

outputSubDirArray = []
for caseVec in caseArray:
    subDir = caseVec[0] + '_' + caseVec[1] + '_' +caseVec[2] + \
             '_nx' + str(caseVec[3]) + '_' + caseVec[4] + '_t' + \
             str(caseVec[5]).replace('.','_') + '_steps' + str(caseVec[6])

    outputSubDirArray.append(os.path.join(outputDir, subDir))

fig, ax = plt.subplots(1, 1)
# TODO change the below back to (6,3)
numStepsRes = np.zeros((len(caseArray),3))
for count, dir in enumerate(outputSubDirArray):
    dataArray = np.load(os.path.join(dir, 'H_array.npy'))
    numStepsRes[count,0] = caseArray[count][5]/(caseArray[count][6])
    # L2 error
    numStepsRes[count,1] = np.max(dataArray[:,8])

    plt.plot(dataArray[:,0], dataArray[:, 8], lw=1.5, color=colorVec[count], linestyle=lineStyleArray[count],
            label='{} $\Delta t$ = {:.5f}'.format(caseArray[count][1], caseArray[count][5]/caseArray[count][6]))

plt.xlabel('Time [s]')
plt.ylabel('$L^2$ Error')
plt.xlim(0.0, tFinal)
#if timeIntScheme == 'SE':
#    plt.ylim(0.0, 0.004)
#else:
#    plt.ylim(0.0, 0.003)

plt.legend()
if plotPNG:
    plt.savefig(os.path.join(plotDir, 'analyticL2ErrorOverTime.png'),
                dpi=500)
if plotEPS:
    plt.savefig(os.path.join(plotDir, 'analyticL2ErrorOverTime.eps'))
plt.close(fig)

# plot p and p_exact vs time
fig, ax = plt.subplots(1, 1)
for count, dir in enumerate(outputSubDirArray):
    dataArray = np.load(os.path.join(dir, 'H_array.npy'))
    plt.plot(dataArray[:,0], dataArray[:, 10], lw=1.5, color=colorVec[count], linestyle=lineStyleArray[count],
             label='p numerical, {}, $\Delta t$ = {:.5f}'.format(caseArray[count][1],
                                                                 caseArray[count][5]/caseArray[count][6]))
plt.plot(dataArray[:, 0], dataArray[:, 11], lw=1.5, color='orange', linestyle='--',
         label='p analytic')

plt.xlabel('Time [s]')
plt.ylabel('p')
plt.xlim(0.0, tFinal)
#if timeIntScheme == 'SE':
#    plt.ylim(0.0, 0.002)
#else:
#    plt.ylim(0.0, 0.003)

plt.legend()
# plt.show()
if plotPNG:
    plt.savefig(os.path.join(plotDir, 'analyticPVsTime.png'), dpi=500)
if plotEPS:
    plt.savefig(os.path.join(plotDir, 'analyticPVsTime.eps'))
plt.close(fig)

# plot Energy resiudal vs time for analytical comparison
fig, ax = plt.subplots(1, 1)
for count, dir in enumerate(outputSubDirArray):
    dataArray = np.load(os.path.join(dir, 'H_array.npy'))
    plt.plot(dataArray[:,0], dataArray[:, 2], lw=1.5, color=colorVec[count], linestyle=lineStyleArray[count],
             label='${}, \Delta t$ = {:.5f}'.format(caseArray[count][1], caseArray[count][5]/caseArray[count][6]))

plt.xlabel('Time [s]')
plt.ylabel('Energy Residual (J)')
plt.xlim(0.0, tFinal)
#if timeIntScheme == 'SE':
#    plt.ylim(0.0, 0.01)
#else:
#    plt.ylim(0.0, 0.003)

plt.legend()
if plotPNG:
    plt.savefig(os.path.join(plotDir, 'analyticEnergyResidualOverTime.png'),
                dpi=500)
if plotEPS:
    plt.savefig(os.path.join(plotDir, 'analyticEnergyResidualOverTime.eps'))
plt.close(fig)

# ------------# Plot analytical L^2 error over step size#---------------#

fig, ax = plt.subplots(1, 1)
plt.plot(numStepsRes[:5, 0], numStepsRes[:5, 1], lw=1.5, color='b', linestyle='', marker='o',
         label='{}'.format(caseArray[0][1]))
plt.plot(numStepsRes[5:10, 0], numStepsRes[5:10, 1], lw=1.5, color='r', linestyle='', marker='x',
         label='{}'.format(caseArray[5][1]))
#create linear line to plot
xLin = np.linspace(numStepsRes[0,0], numStepsRes[len(caseArray)-1,0], 1000)

scale = 115
#scale = 40
yLin = scale*xLin
plt.plot(xLin, yLin, lw=1.0, color='k', label='Linear Trend ({}$x$)'.format(scale))
##create quadratic line to plot
xQuad = np.linspace(numStepsRes[0,0], numStepsRes[len(caseArray)-1,0], 1000)
scale = 13000
yQuad = scale*np.square(xQuad)
plt.plot(xQuad, yQuad, lw=1.0, color='k', linestyle='--', label='Quadratic Trend ({}$x^2$)'.format(scale))
##create Cubic line to plot
#xCubic = np.linspace(numStepsRes[0,0], numStepsRes[5,0], 1000)
#scale = 4000
#yCubic = scale*np.power(xQuad, 3)
#plt.plot(xCubic , yCubic, lw=1.0, color='k', linestyle='--', label='Cubic Trend ($4000x^3$)')
plt.xlabel('Time Step Size [s]')
plt.ylabel('Max $L^2$ Error Norm')
plt.legend()
plt.grid()
#plt.show()
if plotPNG:
    plt.savefig(os.path.join(plotDir, 'analyticL2ErrorVsStepSize.png'), dpi=500)
if plotEPS:
    plt.savefig(os.path.join(plotDir, 'analyticL2ErrorVsStepSize.eps'))
plt.xlim(1e-4, 1e-3)
plt.ylim(1e-4, 1e-1)
plt.xscale('log')
plt.yscale('log')
if plotPNG:
    plt.savefig(os.path.join(plotDir, 'analyticL2ErrorVsStepSizeLogScale.png'), dpi=500)
if plotEPS:
    plt.savefig(os.path.join(plotDir, 'analyticL2ErrorVsStepSizeLogScale.eps'))
plt.close(fig)

# -----# Plot the analytic error for long times for various schemes, wave model#-------#

caseName = 'analytical_t10_0_schemeVariation_3rdOrder'
outputDir = os.path.join('output', caseName)
if not os.path.exists(outputDir):
    print('The output dir {} doesn\'t exist'.format(outputDir))
    quit()

plotDir = os.path.join(outputDir, 'plots')
if not os.path.exists(plotDir):
    os.mkdir(plotDir)

tFinal = 10.0
caseArray = [['R', 'EH', 'weak', 80, 'analytical', 5.0, 40000],
            ['R', 'IE', 'weak', 80, 'analytical', 10.0, 40000],
            #['R', 'SE', 'weak', 80, 'analytical', 10.0, 40000],
            ['R', 'SM', 'weak', 80, 'analytical', 10.0, 40000],
            ['R', 'SV', 'weak', 80, 'analytical', 10.0, 20000]]

colorVec = ['g', 'b', 'k', 'r', 'cyan']
lineStyleArray = ['-', '-', '-', '-', '-']

outputSubDirArray = []
for caseVec in caseArray:
    subDir = caseVec[0] + '_' + caseVec[1] + '_' +caseVec[2] + \
             '_nx' + str(caseVec[3]) + '_' + caseVec[4] + '_t' + \
             str(caseVec[5]).replace('.','_') + '_steps' + str(caseVec[6])

    outputSubDirArray.append(os.path.join(outputDir, subDir))

# Make plot for L2 error over long times

fig, ax = plt.subplots(1, 1)
for count, dir in enumerate(outputSubDirArray):
    dataArray = np.load(os.path.join(dir, 'H_array.npy'))
    plt.plot(dataArray[:,0], dataArray[:, 8], lw=0.2, color=colorVec[count], linestyle=lineStyleArray[count],
             label='{}, $\Delta t$ = {}'.format(caseArray[count][1], caseArray[count][5]/caseArray[count][6]))

plt.xlabel('Time [s]')
plt.ylabel('$L^2$ Error Norm')
plt.xlim(0.0, tFinal)
plt.ylim(0, 0.15)

plt.legend()
if plotPNG:
    plt.savefig(os.path.join(plotDir, 'analyticL2LongTimeSchemeVariation.png'),
                dpi=500)
if plotEPS:
    plt.savefig(os.path.join(plotDir, 'analyticL2LongTimeSchemeVariation.eps'))
plt.close(fig)

# ------------# Plotting results for eigenvalues #---------------#

caseName = 'eigenValue_spaceVariation'
outputDir = os.path.join('output', caseName)
if not os.path.exists(outputDir):
    print('The output dir {} doesn\'t exist'.format(outputDir))
    quit()

plotDir = os.path.join(outputDir, 'plots')
if not os.path.exists(plotDir):
    os.mkdir(plotDir)

caseArray = [['R', 'SE', 'weak', 5, 'analytical', 0.5, 1000, (1, 1)],
             ['R', 'SE', 'weak', 10, 'analytical', 0.5, 1000, (1, 1)],
             ['R', 'SE', 'weak', 15, 'analytical', 0.5, 1000, (1, 1)],
             ['R', 'SE', 'weak', 20, 'analytical', 0.5, 1000, (1, 1)],
             ['R', 'SE', 'weak', 25, 'analytical', 0.5, 1000, (1, 1)],
             ['R', 'SE', 'weak', 30, 'analytical', 0.5, 1000, (1, 1)]]
#              ['R', 'SE', 'weak', 5, 'analytical', 0.5, 1000, (2, 2)],
#              ['R', 'SE', 'weak', 10, 'analytical', 0.5, 1000, (2, 2)],
#              ['R', 'SE', 'weak', 15, 'analytical', 0.5, 1000, (2, 2)],
#              ['R', 'SE', 'weak', 20, 'analytical', 0.5, 1000, (2, 2)],
#              ['R', 'SE', 'weak', 25, 'analytical', 0.5, 1000, (2, 2)]]

outputSubDirArray = []
for caseVec in caseArray:
    subDir = caseVec[0] + '_' + caseVec[1] + '_' +caseVec[2] + \
             '_nx' + str(caseVec[3]) + '_' + caseVec[4] + '_t' + \
             str(caseVec[5]).replace('.','_') + '_steps' + str(caseVec[6])
    if len(caseVec) > 7:
        subDir = subDir + '_P{}_RT{}'.format(caseVec[7][0], caseVec[7][1])

    outputSubDirArray.append(os.path.join(outputDir, subDir))

numCell_Res = np.zeros((len(caseArray), 3))
eig_error_array = np.zeros((len(caseArray),))
data_list = []
for count, dir in enumerate(outputSubDirArray):
    dataArray = np.load(os.path.join(dir, 'out_array.npy'))
    data_list.append(dataArray)
    numCells = np.load(os.path.join(dir, 'numCells.npy'))[0]
    avCellSize = 0.25/numCells
    avCellLength = np.sqrt(avCellSize)
    numCell_Res[count, :] = [numCells, avCellSize, avCellLength]
    # calculate the percentage error on the first eigenvalue
    eig_error_array[count] = 100*abs((dataArray[0, 0] - dataArray[0, 1])/
                    dataArray[0, 0])

# Plot the convergence of the first eigenvalue

fig, ax = plt.subplots(1, 1)
# ax.set_xlim(0, 0.04)
# ax.set_ylim(0, 0.16)
ax.set_xlabel('Characteristic Element Length [$m$]')
ax.set_ylabel('First Eigenfrequency % Error')
plt.plot(numCell_Res[0:6, 2], eig_error_array[0:6], lw=1.5, color='r', linestyle='', marker='x',
         label='P1RT1')
# plt.plot(numCell_Res[6:11, 2], eig_error_array[6:11], lw=1.5, color='b', linestyle='', marker='o',
#          label='P2RT2')

xQuad = np.linspace(numCell_Res[0,2], numCell_Res[-1,2], 1000)
scale = 50
yQuad = scale*np.square(xQuad)
plt.plot(xQuad, yQuad, lw=1.0, color='k',linestyle='--', label='Quadratic Trend ($50x^2$)')

ax.legend(loc='lower right')
plt.grid()
#plt.show()

if plotPNG:
    plt.savefig(os.path.join(plotDir, 'eigenValuePercentErrorVsAvCellLength.png'), dpi=500)
if plotEPS:
    plt.savefig(os.path.join(plotDir, 'eigenValuePercentErrorVsAvCellLength.eps'))
ax.set_xlim(1e-2, 1e-1)
ax.set_ylim(1e-3, 1)
plt.xscale('log')
plt.yscale('log')
if plotPNG:
    plt.savefig(os.path.join(plotDir, 'eigenValuePercentErrorVsAvCellLengthLogScale.png'), dpi=500)
if plotEPS:
    plt.savefig(os.path.join(plotDir, 'eigenValuePercentErrorVsAvCellLengthLogScale.eps'))
plt.close(fig)

fig, ax = plt.subplots(1, 1)
ax.set_xlabel('Eigenfrequency Number')
ax.set_ylabel('Complex Eigenfrequency')
plt.plot(range(1, 52), data_list[5][:, 0], color='k', linestyle='', marker='x', markersize=5.0, label='Analytic')
plt.plot(range(1, 52), data_list[5][:, 1], color='r', linestyle='', marker='x', markersize=5.0, label='P1RT1',
         markerfacecolor='none')
# plt.plot(range(1, 52), data_list[10][:, 1], color='b', linestyle='', marker='^', markersize=5.0, label='P2RT2',
#          markerfacecolor='none')
ax.set_xlim(0, 50)
ax.set_ylim(0, 80)
ax.legend(loc='lower right')
plt.grid()

if plotPNG:
    plt.savefig(os.path.join(plotDir, 'eigenValueVsAnalytic.png'), dpi=500)
if plotEPS:
    plt.savefig(os.path.join(plotDir, 'eigenValueVsAnalytic.eps'))

# plt.show()

plt.close(fig)

# ------------# Plotting results for interconnection wave and electromechanical #---------------#
caseName = 'IC_SV_t_20_0_singleRun'
outputDir = os.path.join('output', caseName)
if not os.path.exists(outputDir):
    print('The output dir {} doesn\'t exist'.format(outputDir))
    quit()

plotDir = os.path.join(outputDir, 'plots')
if not os.path.exists(plotDir):
    os.mkdir(plotDir)

subDir = 'R_SV_weak_nx120_IC_t20_steps40000'
outputSubDirArray = [os.path.join(outputDir, subDir)]
# ------------# Plot Hamiltonian and energy residual over time#---------------#

tFinal = 20.0 #TODO get this from data array
# Plot the energy residual
fig, ax = plt.subplots(1, 1)
ax.set_xlim(0, tFinal)
# ax.set_ylim(-0.0002, 0.0008)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Energy Residual [J]')
for count, dir in enumerate(outputSubDirArray):
    dataArray = np.load(os.path.join(dir, 'H_array.npy'))
    plt.plot(dataArray[:, 0], dataArray[:, 2], lw=0.5, color='b', linestyle='-',
             label='Total Residual')
    # Plot wave equation residual
    waveResid = np.sqrt(np.square(dataArray[:, 7] + dataArray[:,4]))
    plt.plot(dataArray[:, 0], waveResid, lw=0.5, color='r', linestyle=lineStyleArray[count],
             label='Wave Residual')
    # Plot EM residual
    emResid = np.sqrt(np.square(dataArray[:, 6] - dataArray[:,5] - dataArray[:,4]))
    plt.plot(dataArray[:, 0], emResid, lw=0.5, color='g', linestyle=lineStyleArray[count],
             label='EM Residual')
    #plot check
    totalResid2 = dataArray[:, 6] +dataArray[:, 7] - dataArray[:,5]
    # plt.plot(dataArray[:, 0], totalResid2, lw=0.5, color='c', linestyle=lineStyleArray[count],
    #         label='check TotalResidual')

ax.legend()
# plt.show()
if plotPNG:
    plt.savefig(os.path.join(plotDir, 'ICEnergyRes.png'), dpi=500)
if plotEPS:
    plt.savefig(os.path.join(plotDir, 'ICEnergyRes.eps'))
plt.close(fig)

# Plot the hamiltonian
fig, ax = plt.subplots(1, 1)
ax.set_xlim(0, tFinal)
# ax.set_ylim(-0.01, 0.01)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Hamiltonian [J]')
for count, dir in enumerate(outputSubDirArray):
    dataArray = np.load(os.path.join(dir, 'H_array.npy'))
    plt.plot(dataArray[:, 0], dataArray[:, 1], lw=0.5, color=colorVec[count], linestyle=lineStyleArray[count],
             label='Interconnection SV')
ax.legend()
# plt.show()
if plotPNG:
    plt.savefig(os.path.join(plotDir, 'ICHamiltonian.png'), dpi=500)
if plotEPS:
    plt.savefig(os.path.join(plotDir, 'ICHamiltonian.eps'))
# overwrite limits if we want to zoom
ax.set_xlim(0.25, tFinal)
ax.set_ylim(0.009, 0.011)
if plotPNG:
    plt.savefig(os.path.join(plotDir, 'ICHamiltonianZoom.png'), dpi=500)
if plotEPS:
    plt.savefig(os.path.join(plotDir, 'ICHamiltonianZoom.eps'))
# overwrite limits if we want to zoom
ax.set_xlim(0.25, tFinal)
ax.set_ylim(0.0002, 0.0003)
# plt.savefig(os.path.join(plotDir, 'IC_HamiltonianZoomZoom.png'), dpi=500)
plt.close(fig)

#plot the motor displacement
fig, ax = plt.subplots(1, 1)
ax.set_xlim(0, tFinal)
# ax.set_ylim(-0.0002, 0.0008)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Displacement [m]')
for count, dir in enumerate(outputSubDirArray):
    dataArray = np.load(os.path.join(dir, 'H_array.npy'))
    plt.plot(dataArray[:, 0], dataArray[:, 3], lw=0.5, color=colorVec[count], linestyle=lineStyleArray[count],
             label='Interconnection SV')

ax.legend()
#plt.show()
if plotPNG:
    plt.savefig(os.path.join(plotDir, 'ICDisplacement.png'), dpi=500)
if plotEPS:
    plt.savefig(os.path.join(plotDir, 'ICDisplacement.eps'))
plt.close(fig)

# ------------# Plot max energy residual over step size#---------------#

caseName = 'IC_t4_5_timeVariation'
outputDir = os.path.join('output', caseName)
if not os.path.exists(outputDir):
    print('The output dir {} doesn\'t exist'.format(outputDir))
    quit()

plotDir = os.path.join(outputDir, 'plots')
if not os.path.exists(plotDir):
    os.mkdir(plotDir)

caseArray = [['R', 'SV', 'weak', 120, 'IC', 4.5, 3000],
             ['R', 'SV', 'weak', 120, 'IC', 4.5, 4500],
             ['R', 'SV', 'weak', 120, 'IC', 4.5, 6000],
             ['R', 'SV', 'weak', 120, 'IC', 4.5, 7500],
             ['R', 'SV', 'weak', 120, 'IC', 4.5, 9000],
             ['R', 'SM', 'weak', 120, 'IC', 4.5, 3000],
             ['R', 'SM', 'weak', 120, 'IC', 4.5, 4500],
             ['R', 'SM', 'weak', 120, 'IC', 4.5, 6000],
             ['R', 'SM', 'weak', 120, 'IC', 4.5, 7500],
             ['R', 'SM', 'weak', 120, 'IC', 4.5, 9000]]
#             ['R', 'SV', 'weak', 120, 'IC4', 4.5, 1500],
#             ['R', 'SV', 'weak', 120, 'IC4', 4.5, 3000],
#             ['R', 'SV', 'weak', 120, 'IC4', 4.5, 4500],
#             ['R', 'SV', 'weak', 120, 'IC4', 4.5, 6000],
#             ['R', 'SV', 'weak', 120, 'IC4', 4.5, 7500],
#             ['R', 'SV', 'weak', 120, 'IC4', 4.5, 9000],
#             ['R', 'SE', 'weak', 120, 'IC4', 4.5, 1500],
#             ['R', 'SE', 'weak', 120, 'IC4', 4.5, 3000],
#             ['R', 'SE', 'weak', 120, 'IC4', 4.5, 4500],
#             ['R', 'SE', 'weak', 120, 'IC4', 4.5, 6000],
#             ['R', 'SE', 'weak', 120, 'IC4', 4.5, 7500],
#             ['R', 'SE', 'weak', 120, 'IC4', 4.5, 9000]]

outputSubDirArray = []
for caseVec in caseArray:
    subDir = caseVec[0] + '_' + caseVec[1] + '_' +caseVec[2] + \
             '_nx' + str(caseVec[3]) + '_' + caseVec[4] + '_t' + \
             str(caseVec[5]).replace('.','_') + '_steps' + str(caseVec[6])

    outputSubDirArray.append(os.path.join(outputDir, subDir))

numStepsRes = np.zeros((len(caseArray),2))
for count, dir in enumerate(outputSubDirArray):
    dataArray = np.load(os.path.join(dir, 'H_array.npy'))
    numStepsRes[count,0] = caseArray[count][5]/(caseArray[count][6])
    numStepsRes[count,1] = np.max(abs(dataArray[:,2]))


fig, ax = plt.subplots(1, 1)
plt.plot(numStepsRes[0:5, 0], numStepsRes[0:5, 1], lw=1.5, color='r', linestyle='', marker='x',
         label='SV wave-EM')
plt.plot(numStepsRes[5:10, 0], numStepsRes[5:10, 1], lw=1.5, color='b', linestyle='', marker='o',
         label='SM wave-EM')
#plt.plot(numStepsRes[12:18, 0], numStepsRes[12:18, 1], lw=1.5, color='g', linestyle='', marker='o',
#         label='SE wave-EM 4DOF')
#create linear line to plot
#xLin = np.linspace(numStepsRes[0,0], numStepsRes[4,0], 1000)
#scale = 800
#yLin = scale*xLin
#plt.plot(xLin, yLin, lw=1.0, color='k', label='Linear Trend ($800x$)')
#create quadratic line to plot
xQuad = np.linspace(numStepsRes[0,0], numStepsRes[len(caseArray)-1,0], 1000)
scale = 400
yQuad = scale*np.square(xQuad)
plt.plot(xQuad, yQuad, lw=1.0, color='k',linestyle='--', label='Quadratic Trend ($400x^2$)')
#create Cubic line to plot
#xCubic = np.linspace(numStepsRes[0,0], numStepsRes[5,0], 1000)
#scale = 4000
#yCubic = scale*np.power(xQuad, 3)
#plt.plot(xCubic , yCubic, lw=1.0, color='k', linestyle='--', label='Cubic Trend ($4000x^3$)')
plt.xlabel('Time Step Size [s]')
plt.ylabel('Max Energy Residual [J]')
plt.legend()
plt.grid()
#plt.show()
if plotPNG:
    plt.savefig(os.path.join(plotDir, 'ICEnergyResVsNumSteps.png'), dpi=500)
if plotEPS:
    plt.savefig(os.path.join(plotDir, 'ICEnergyResVsNumSteps.eps'))
plt.xlim(3e-4, 4e-3)
plt.ylim(1e-14, 1e-2)
plt.xscale('log')
plt.yscale('log')
if plotPNG:
    plt.savefig(os.path.join(plotDir, 'ICEnergyResVsNumStepsLogScale.png'), dpi=500)
if plotEPS:
    plt.savefig(os.path.join(plotDir, 'ICEnergyResVsNumStepsLogScale.eps'))
plt.close(fig)


# ------------# Plotting results for square domain interconnection#---------------#
caseName = 'IC_square_IC_SV_t8_0'

outputDir = os.path.join('output', caseName)
if not os.path.exists(outputDir):
    print('The output dir {} doesn\'t exist'.format(outputDir))
    quit()

plotDir = os.path.join(outputDir, 'plots')
if not os.path.exists(plotDir):
    os.mkdir(plotDir)

subDir = 'S_1C_SV_weak_nx60_IC_t8_steps16000'
outputSubDirArray = [os.path.join(outputDir, subDir)]
# ------------# Plot Hamiltonian and energy residual over time#---------------#

tFinal = 8.0 #TODO get this from data array
# Plot the energy residual
fig, ax = plt.subplots(1, 1)
ax.set_xlim(0, tFinal)
# ax.set_ylim(-0.0002, 0.0008)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Energy Residual [J]')
for count, dir in enumerate(outputSubDirArray):
    dataArray = np.load(os.path.join(dir, 'H_array.npy'))
    plt.plot(dataArray[:, 0], dataArray[:, 2], lw=0.5, color='b', linestyle='-',
             label='Total Residual')
    # Plot wave equation residual
    waveResid = np.sqrt(np.square(dataArray[:, 7] + dataArray[:,4]))
    plt.plot(dataArray[:, 0], waveResid, lw=0.5, color='r', linestyle=lineStyleArray[count],
             label='Wave Residual')
    # Plot EM residual
    emResid = np.sqrt(np.square(dataArray[:, 6] - dataArray[:,5] - dataArray[:,4]))
    plt.plot(dataArray[:, 0], emResid, lw=0.5, color='g', linestyle=lineStyleArray[count],
             label='EM Residual')
    #plot check
    totalResid2 = dataArray[:, 6] +dataArray[:, 7] - dataArray[:,5]
    # plt.plot(dataArray[:, 0], totalResid2, lw=0.5, color='c', linestyle=lineStyleArray[count],
    #         label='check TotalResidual')

ax.legend()
# plt.show()
if plotPNG:
    plt.savefig(os.path.join(plotDir, 'ICSquareEnergyRes.png'), dpi=500)
if plotEPS:
    plt.savefig(os.path.join(plotDir, 'ICSquareEnergyRes.eps'))
plt.close(fig)

# Plot the hamiltonian
fig, ax = plt.subplots(1, 1)
ax.set_xlim(0, tFinal)
# ax.set_ylim(-0.01, 0.01)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Hamiltonian [J]')
for count, dir in enumerate(outputSubDirArray):
    dataArray = np.load(os.path.join(dir, 'H_array.npy'))
    plt.plot(dataArray[:, 0], dataArray[:, 1], lw=0.5, color=colorVec[count], linestyle=lineStyleArray[count],
             label='Interconnection SV')
ax.legend()
# plt.show()
if plotPNG:
    plt.savefig(os.path.join(plotDir, 'ICSquareHamiltonian.png'), dpi=500)
if plotEPS:
    plt.savefig(os.path.join(plotDir, 'ICSquareHamiltonian.eps'))

# ------------# Plotting results for 4DOF Interconnection#---------------#
#TODO include below if i want 4dof results
quit()

subDir = 'R_SV_weak_IC_t1_5_steps3000_4DOF'
outputSubDirArray = [os.path.join(outputDir, subDir)]
# ------------# Plot Hamiltonian and energy residual over time#---------------#

tFinal = 1.5 #TODO get this from data array
# Plot the energy residual
fig, ax = plt.subplots(1, 1)
ax.set_xlim(0, tFinal)
# ax.set_ylim(-0.0002, 0.0008)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Energy Residual [J]')
for count, dir in enumerate(outputSubDirArray):
    dataArray = np.load(os.path.join(dir, 'H_array.npy'))
    plt.plot(dataArray[:, 0], dataArray[:, 2], lw=0.5, color='b', linestyle='-',
             label='Total Residual')
    # Plot wave equation residual
    waveResid = np.sqrt(np.square(dataArray[:, 7] + dataArray[:,4]))
    plt.plot(dataArray[:, 0], waveResid, lw=0.5, color='r', linestyle=lineStyleArray[count],
             label='Wave Residual')
    # Plot EM residual
    emResid = np.sqrt(np.square(dataArray[:, 6] - dataArray[:,5] - dataArray[:,4]))
    plt.plot(dataArray[:, 0], emResid, lw=0.5, color='g', linestyle=lineStyleArray[count],
             label='EM Residual')
    #plot check
    totalResid2 = dataArray[:, 6] +dataArray[:, 7] - dataArray[:,5]
    # plt.plot(dataArray[:, 0], totalResid2, lw=0.5, color='c', linestyle=lineStyleArray[count],
    #         label='check TotalResidual')

ax.legend()
# plt.show()
if plotPNG:
    plt.savefig(os.path.join(plotDir, 'IC4DOFEnergyRes.png'), dpi=500)
if plotEPS:
    plt.savefig(os.path.join(plotDir, 'IC4DOFEnergyRes.eps'))
plt.close(fig)

# Plot the hamiltonian
fig, ax = plt.subplots(1, 1)
ax.set_xlim(0, tFinal)
# ax.set_ylim(-0.01, 0.01)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Hamiltonian [J]')
for count, dir in enumerate(outputSubDirArray):
    dataArray = np.load(os.path.join(dir, 'H_array.npy'))
    plt.plot(dataArray[:, 0], dataArray[:, 1], lw=0.5, color=colorVec[count], linestyle=lineStyleArray[count],
             label='Interconnection SV')
ax.legend()
# plt.show()
if plotPNG:
    plt.savefig(os.path.join(plotDir, 'IC4DOFHamiltonian.png'), dpi=500)
if plotEPS:
    plt.savefig(os.path.join(plotDir, 'IC4DOFHamiltonian.eps'))
# overwrite limits if we want to zoom
ax.set_xlim(0.25, tFinal)
ax.set_ylim(0.009, 0.011)
if plotPNG:
    plt.savefig(os.path.join(plotDir, 'IC4DOFHamiltonianZoom.png'), dpi=500)
if plotEPS:
    plt.savefig(os.path.join(plotDir, 'IC4DOFHamiltonianZoom.eps'))
# overwrite limits if we want to zoom
ax.set_xlim(0.25, tFinal)
ax.set_ylim(0.0002, 0.0003)
# plt.savefig(os.path.join(plotDir, 'IC_HamiltonianZoomZoom.png'), dpi=500)
plt.close(fig)

#plot the motor displacement
fig, ax = plt.subplots(1, 1)
ax.set_xlim(0, tFinal)
# ax.set_ylim(-0.0002, 0.0008)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Displacement [m]')
for count, dir in enumerate(outputSubDirArray):
    dataArray = np.load(os.path.join(dir, 'H_array.npy'))
    plt.plot(dataArray[:, 0], dataArray[:, 3], lw=0.5, color=colorVec[count], linestyle=lineStyleArray[count],
             label='Interconnection SV')

ax.legend()
#plt.show()
if plotPNG:
    plt.savefig(os.path.join(plotDir, 'IC4DOFDisplacement.eps'), dpi=500)
if plotEPS:
    plt.savefig(os.path.join(plotDir, 'IC4DOFDisplacement.png'))
plt.close(fig)







