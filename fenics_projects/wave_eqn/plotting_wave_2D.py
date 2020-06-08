import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import paperPlotSetup

paperPlotSetup.Setup_Plot(3)

# -------------------------------# Access Data #---------------------------------#
outputDir = 'output'
if not os.path.exists(outputDir):
    print('The output dir {} doesn\'t exist'.format(outputDir))
plotDir = os.path.join(outputDir, 'plots')
if not os.path.exists(plotDir):
    os.mkdir(plotDir)

caseArray = [['R', 'IE', 'weak'],
             ['R', 'IE', 'strong'],
             ['R', 'EE', 'weak'],
             ['R', 'EE', 'strong'],
             ['R', 'SE', 'weak'],
             ['R', 'SE', 'strong'],
             ['R', 'EH', 'weak'],
             ['R', 'EH', 'strong'],
             ['R', 'SV', 'weak'],
             ['R', 'SV', 'strong']]

colorArray = ['b', 'b', 'r', 'r', 'c', 'c', 'g', 'g', 'k', 'k']
lineStyleArray = ['-', '--', '-', '--', '-', '--', '-', '--', '-', '--',]

outputSubDirArray = []
for caseVec in caseArray:
    subDir = 'output_' + caseVec[0] + '_' + caseVec[1] + '_' +caseVec[2]
    outputSubDirArray.append(os.path.join(outputDir, subDir))

# -------------------------------# Plotting #---------------------------------#

# ------------# Plot Hamiltonian and energy residual over time#---------------#
tFinal = 1.5 #TODO get this from data array
# Plot the energy residual
fig, ax = plt.subplots(1, 1)
ax.set_xlim(0, tFinal)
ax.set_ylim(-0.0002, 0.0008)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Energy Residual [J/kg]')
for count, dir in enumerate(outputSubDirArray):
    dataArray = np.load(os.path.join(dir, 'H_array.npy'))
    plt.plot(dataArray[:, 0], dataArray[:, 2], lw=0.5, color=colorArray[count], linestyle=lineStyleArray[count],
             label='{} {}'.format(caseArray[count][1], caseArray[count][2]))

ax.legend()
#plt.show()
plt.savefig(os.path.join(plotDir, 'EnergyRes.png'), dpi=500, bbox_inches='tight')
plt.close(fig)

# Plot the hamiltonian
fig, ax = plt.subplots(1, 1)
ax.set_xlim(0, tFinal)
ax.set_ylim(0.0, 0.04)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Hamiltonian [J/kg]')
for count, dir in enumerate(outputSubDirArray):
    dataArray = np.load(os.path.join(dir, 'H_array.npy'))
    plt.plot(dataArray[:, 0], dataArray[:, 1], lw=0.5, color=colorArray[count], linestyle=lineStyleArray[count],
             label='{} {}'.format(caseArray[count][1], caseArray[count][2]))
ax.legend()
# plt.show()
plt.savefig(os.path.join(plotDir, 'Hamiltonian.png'), dpi=500, bbox_inches='tight')
# overwrite limits if we want to zoom
ax.set_xlim(0.25, tFinal)
ax.set_ylim(0.0312, 0.0317)
plt.savefig(os.path.join(plotDir, 'HamiltonianZoom.png'), dpi=500, bbox_inches='tight')
# overwrite limits if we want to zoom
ax.set_xlim(0.25, tFinal)
ax.set_ylim(0.06268, 0.06279)
ax.set_ylim(0.03134, 0.03142)
plt.savefig(os.path.join(plotDir, 'HamiltonianZoomZoom.png'), dpi=500, bbox_inches='tight')
plt.close(fig)

# ------------# Plotting boundary energy residual for multiple cell numbers #---------------#

caseArray = [['R', 'SV', 'strong', 20],
             ['R', 'SV', 'strong', 40],
             ['R', 'SV', 'strong', 60],
             ['R', 'SV', 'strong'],
             ['R', 'SV', 'strong', 100],
             ['R', 'SV', 'strong', 120],
             ['R', 'SV', 'strong', 160],
             ['R', 'SV', 'weak', 20],
             ['R', 'SV', 'weak', 40],
             ['R', 'SV', 'weak', 60],
             ['R', 'SV', 'weak'],
             ['R', 'SV', 'weak', 100],
             ['R', 'SV', 'weak', 120],
             ['R', 'SV', 'weak', 160]]

colorArray = ['r', 'orange', 'y', 'g', 'c', 'b', 'm', 'r', 'orange', 'y', 'g', 'c', 'b', 'm']
lineStyleArray = ['--', '--', '--', '--', '--', '--', '--', '-', '-', '-', '-', '-', '-', '-']

outputSubDirArray = []
for caseVec in caseArray:
    subDir = 'output_' + caseVec[0] + '_' + caseVec[1] + '_' +caseVec[2]
    if len(caseVec) > 3:
        # we have a nx value to set
        subDir = subDir + '_nx' + str(caseVec[3])
    outputSubDirArray.append(os.path.join(outputDir, subDir))

fig, ax = plt.subplots(1, 1)
ax.set_xlim(0, tFinal)
ax.set_ylim(-0.0002, 0.0008)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Energy Residual [J/kg]')
for count, dir in enumerate(outputSubDirArray):
    dataArray = np.load(os.path.join(dir, 'H_array.npy'))
    numCells = np.load(os.path.join(dir, 'numCells.npy'))[0]
    plt.plot(dataArray[:, 0], dataArray[:, 2], lw=0.5, color=colorArray[count], linestyle=lineStyleArray[count],
             label='{} {}, numCells = {}'.format(caseArray[count][1], caseArray[count][2], numCells))

ax.legend(loc='upper right')
#plt.show()
plt.savefig(os.path.join(plotDir, 'EnergyResDistForNumCells.png'), dpi=500, bbox_inches='tight')
plt.close(fig)

# -----# Plot the final boundary energy residual against number of cells #-------#

fig, ax = plt.subplots(1, 1)
ax.set_xlim(0, 14000)
ax.set_ylim(0, 0.0035)
ax.set_xlabel('number of Cells')
ax.set_ylabel('Energy Residual [J/kg]')
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
plt.savefig(os.path.join(plotDir, 'EnergyResVsNumCells.png'), dpi=500, bbox_inches='tight')
ax.set_xlim(1e2, 1e5)
ax.set_ylim(1e-7, 1e-2)
plt.xscale('log')
plt.yscale('log')
plt.savefig(os.path.join(plotDir, 'EnergyResVsNumCellsLogScale.png'), dpi=500, bbox_inches='tight')
plt.close(fig)

#Plot against average cell size
fig, ax = plt.subplots(1, 1)
ax.set_xlim(0, 0.0014)
ax.set_ylim(0, 0.0035)
ax.set_xlabel('Average Cell Size [$m^2$]')
ax.set_ylabel('Energy Residual [J/kg]')
plt.plot(numCell_Res[0:7, 1], numCell_Res[0:7, 3], lw=1.5, color='b', linestyle='', marker='o',
         label='SV strong')
plt.plot(numCell_Res[7:14, 1], numCell_Res[7:14, 3], lw=1.5, color='r', linestyle='', marker='x',
         label='SV weak')

ax.legend()
plt.grid()
#plt.show()
plt.savefig(os.path.join(plotDir, 'EnergyResVsAvCellSize.png'), dpi=500, bbox_inches='tight')
ax.set_xlim(1e-5, 1e-2)
ax.set_ylim(1e-7, 1e-2)
plt.xscale('log')
plt.yscale('log')
plt.savefig(os.path.join(plotDir, 'EnergyResVsAvCellSizeLogScale.png'), dpi=500, bbox_inches='tight')
plt.close(fig)

#Plot against average cell Length
fig, ax = plt.subplots(1, 1)
ax.set_xlim(0, 0.04)
ax.set_ylim(0, 0.0035)
ax.set_xlabel('Average Cell Length [$m$]')
ax.set_ylabel('Energy Residual [J/kg]')
plt.plot(numCell_Res[0:7, 2], numCell_Res[0:7, 3], lw=1.5, color='b', linestyle='', marker='o',
         label='SV strong')
plt.plot(numCell_Res[7:14, 2], numCell_Res[7:14, 3], lw=1.5, color='r', linestyle='', marker='x',
         label='SV weak')

ax.legend()
plt.grid()
#plt.show()
plt.savefig(os.path.join(plotDir, 'EnergyResVsAvCellLength.png'), dpi=500, bbox_inches='tight')
ax.set_xlim(1e-3, 1e-1)
ax.set_ylim(1e-7, 1e-2)
plt.xscale('log')
plt.yscale('log')
plt.savefig(os.path.join(plotDir, 'EnergyResVsAvCellLengthLogScale.png'), dpi=500, bbox_inches='tight')
plt.close(fig)

strongGradient = (np.log(numCell_Res[6,3]) - np.log(numCell_Res[1,3]))/ \
                 (np.log(numCell_Res[6,2]) - np.log(numCell_Res[1,2]))

print(strongGradient)

# ------------# Plotting results for interconnection wave and electromechanical #---------------#


subDir = 'output_R_SV_weak_IC'
outputSubDirArray = [os.path.join(outputDir, subDir)]
# ------------# Plot Hamiltonian and energy residual over time#---------------#

tFinal = 4.5 #TODO get this from data array
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
             label='em Residual')
    #plot check
    totalResid2 = dataArray[:, 6] +dataArray[:, 7] - dataArray[:,5]
    plt.plot(dataArray[:, 0], totalResid2, lw=0.5, color='c', linestyle=lineStyleArray[count],
             label='check TotalResidual')

ax.legend()
# plt.show()
plt.savefig(os.path.join(plotDir, 'IC_EnergyRes.png'), dpi=500, bbox_inches='tight')
plt.close(fig)

# Plot the hamiltonian
fig, ax = plt.subplots(1, 1)
ax.set_xlim(0, tFinal)
# ax.set_ylim(-0.01, 0.01)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Hamiltonian [J]')
for count, dir in enumerate(outputSubDirArray):
    dataArray = np.load(os.path.join(dir, 'H_array.npy'))
    plt.plot(dataArray[:, 0], dataArray[:, 1], lw=0.5, color=colorArray[count], linestyle=lineStyleArray[count],
             label='Interconnection SV')
ax.legend()
# plt.show()
plt.savefig(os.path.join(plotDir, 'IC_Hamiltonian.png'), dpi=500, bbox_inches='tight')
# overwrite limits if we want to zoom
ax.set_xlim(0.25, tFinal)
ax.set_ylim(0.009, 0.011)
plt.savefig(os.path.join(plotDir, 'IC_HamiltonianZoom.png'), dpi=500, bbox_inches='tight')
# overwrite limits if we want to zoom
ax.set_xlim(0.25, tFinal)
ax.set_ylim(0.0002, 0.0003)
# plt.savefig(os.path.join(plotDir, 'IC_HamiltonianZoomZoom.png'), dpi=500, bbox_inches='tight')
plt.close(fig)

#plot the motor displacement
fig, ax = plt.subplots(1, 1)
ax.set_xlim(0, tFinal)
# ax.set_ylim(-0.0002, 0.0008)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Displacement [m]')
for count, dir in enumerate(outputSubDirArray):
    dataArray = np.load(os.path.join(dir, 'H_array.npy'))
    plt.plot(dataArray[:, 0], dataArray[:, 3], lw=0.5, color=colorArray[count], linestyle=lineStyleArray[count],
             label='Interconnection SV')

ax.legend()
#plt.show()
plt.savefig(os.path.join(plotDir, 'IC_Displacement.png'), dpi=500, bbox_inches='tight')
plt.close(fig)
