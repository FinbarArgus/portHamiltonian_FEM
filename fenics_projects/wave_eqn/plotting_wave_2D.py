import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import paperPlotSetup

paperPlotSetup.Setup_Plot(3)

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
             ['R', 'SV', 'strong', 80, 'wave', 1.5, 3000]]

colorVec= ['b', 'b', 'r', 'r', 'c', 'c', 'g', 'g', 'k', 'k']
lineStyleArray = ['-', '--', '-', '--', '-', '--', '-', '--', '-', '--',]

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
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 12))
ax.set_xlim(0, tFinal)
ax.set_ylim(-0.02, 0.03)
ax2.set_xlabel('Time [s]')
ax.set_ylabel('Energy Residual [J]')
ax2.set_ylabel('Energy Residual [J]')
for count, dir in enumerate(outputSubDirArray):
    dataArray = np.load(os.path.join(dir, 'H_array.npy'))
    ax.plot(dataArray[:, 0], dataArray[:, 2], lw=0.5, color=colorVec[count], linestyle=lineStyleArray[count],
             label='{} {}'.format(caseArray[count][1], caseArray[count][2]))
    ax2.plot(dataArray[:, 0], dataArray[:, 2], lw=0.5, color=colorVec[count], linestyle=lineStyleArray[count],
            label='{} {}'.format(caseArray[count][1], caseArray[count][2]))

ax.legend(loc=1)
#plt.show()
# set ylimits for zoom
ax2.set_xlim(0, tFinal)
ax2.set_ylim(-0.0005, 0.0005)
plt.savefig(os.path.join(plotDir, 'EnergyRes.png'), dpi=500, bbox_inches='tight')
plt.close(fig)

# Plot the hamiltonian
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 12))
ax.set_xlim(0, tFinal)
ax.set_ylim(0.0, 2.2)
ax2.set_xlabel('Time [s]')
ax.set_ylabel('Hamiltonian [J]')
ax2.set_ylabel('Hamiltonian [J]')
for count, dir in enumerate(outputSubDirArray):
    dataArray = np.load(os.path.join(dir, 'H_array.npy'))
    ax.plot(dataArray[:, 0], dataArray[:, 1], lw=0.5, color=colorVec[count], linestyle=lineStyleArray[count],
             label='{} {}'.format(caseArray[count][1], caseArray[count][2]))
    ax2.plot(dataArray[:, 0], dataArray[:, 1], lw=0.5, color=colorVec[count], linestyle=lineStyleArray[count],
            label='{} {}'.format(caseArray[count][1], caseArray[count][2]))
ax.legend()
# overwrite limits if we want to zoom
ax2.set_xlim(0.0, tFinal)
ax2.set_ylim(1.914, 1.916)
plt.savefig(os.path.join(plotDir, 'Hamiltonian.png'), dpi=500, bbox_inches='tight')
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
plt.savefig(os.path.join(plotDir, 'EnergyResDistForNumCells.png'), dpi=500, bbox_inches='tight')
plt.close(fig)

# -----# Plot the final boundary energy residual against number of cells #-------#

fig, ax = plt.subplots(1, 1)
ax.set_xlim(0, 14000)
ax.set_ylim(0, 0.16)
ax.set_xlabel('number of Cells')
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
plt.savefig(os.path.join(plotDir, 'EnergyResVsNumCells.png'), dpi=500, bbox_inches='tight')
ax.set_xlim(1e2, 1e5)
ax.set_ylim(1e-5, 1)
plt.xscale('log')
plt.yscale('log')
plt.savefig(os.path.join(plotDir, 'EnergyResVsNumCellsLogScale.png'), dpi=500, bbox_inches='tight')
plt.close(fig)

#Plot against average cell size
fig, ax = plt.subplots(1, 1)
ax.set_xlim(0, 0.0014)
ax.set_ylim(0, 0.16)
ax.set_xlabel('Average Cell Size [$m^2$]')
ax.set_ylabel('Energy Residual [J]')
plt.plot(numCell_Res[0:7, 1], numCell_Res[0:7, 3], lw=1.5, color='b', linestyle='', marker='o',
         label='SV strong')
plt.plot(numCell_Res[7:14, 1], numCell_Res[7:14, 3], lw=1.5, color='r', linestyle='', marker='x',
         label='SV weak')

ax.legend()
plt.grid()
#plt.show()
plt.savefig(os.path.join(plotDir, 'EnergyResVsAvCellSize.png'), dpi=500, bbox_inches='tight')
ax.set_xlim(1e-5, 1e-2)
ax.set_ylim(1e-5, 1)
plt.xscale('log')
plt.yscale('log')
plt.savefig(os.path.join(plotDir, 'EnergyResVsAvCellSizeLogScale.png'), dpi=500, bbox_inches='tight')
plt.close(fig)

#Plot against average cell Length
fig, ax = plt.subplots(1, 1)
ax.set_xlim(0, 0.04)
ax.set_ylim(0, 0.16)
ax.set_xlabel('Characteristic Cell Length [$m$]')
ax.set_ylabel('Energy Residual [J]')
plt.plot(numCell_Res[0:7, 2], numCell_Res[0:7, 3], lw=1.5, color='b', linestyle='', marker='o',
         label='SV strong')
plt.plot(numCell_Res[7:14, 2], numCell_Res[7:14, 3], lw=1.5, color='r', linestyle='', marker='x',
         label='SV weak')

#create quadratic line to plot
xQuad = np.linspace(numCell_Res[0,2], numCell_Res[6,2], 1000)
scale = 100
yQuad = scale*np.square(xQuad)
plt.plot(xQuad, yQuad, lw=1.0, color='k', label='Quadratic Trend ($100x^2$)')

ax.legend()
plt.grid()
#plt.show()
plt.savefig(os.path.join(plotDir, 'EnergyResVsAvCellLength.png'), dpi=500, bbox_inches='tight')
ax.set_xlim(1e-3, 1e-1)
ax.set_ylim(1e-5, 1)
plt.xscale('log')
plt.yscale('log')
plt.savefig(os.path.join(plotDir, 'EnergyResVsAvCellLengthLogScale.png'), dpi=500, bbox_inches='tight')
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

caseArray = [['R', 'SV', 'weak', 60, 'analytical', 1.5, 3000],
             ['R', 'SV', 'weak', 80, 'analytical', 1.5, 3000],
             ['R', 'SV', 'weak', 100, 'analytical', 1.5, 3000],
             ['R', 'SV', 'weak', 120, 'analytical', 1.5, 3000],
             ['R', 'SV', 'weak', 140, 'analytical', 1.5, 3000],
             ['R', 'SV', 'weak', 160, 'analytical', 1.5, 3000],
             ['R', 'SE', 'weak', 60, 'analytical', 1.5, 6000],
             ['R', 'SE', 'weak', 80, 'analytical', 1.5, 6000],
             ['R', 'SE', 'weak', 100, 'analytical', 1.5, 6000],
             ['R', 'SE', 'weak', 120, 'analytical', 1.5, 6000],
             ['R', 'SE', 'weak', 140, 'analytical', 1.5, 6000],
             ['R', 'SE', 'weak', 160, 'analytical', 1.5, 6000]]

outputSubDirArray = []
for caseVec in caseArray:
    subDir = caseVec[0] + '_' + caseVec[1] + '_' +caseVec[2] + \
             '_nx' + str(caseVec[3]) + '_' + caseVec[4] + '_t' + \
             str(caseVec[5]).replace('.','_') + '_steps' + str(caseVec[6])

    outputSubDirArray.append(os.path.join(outputDir, subDir))

numCell_Res = np.zeros((len(caseArray), 5))
for count, dir in enumerate(outputSubDirArray):
    dataArray = np.load(os.path.join(dir, 'H_array.npy'))
    numCells = np.load(os.path.join(dir, 'numCells.npy'))[0]
    avCellSize = 0.25/numCells
    avCellLength = np.sqrt(avCellSize)
    numCell_Res[count, :] = [numCells, avCellSize, avCellLength,
                             np.max(abs(dataArray[:,8])), np.sqrt(np.max(dataArray[:,9]))]

#Plot L2 error against average cell Length
fig, ax = plt.subplots(1, 1)
# ax.set_xlim(0, 0.04)
# ax.set_ylim(0, 0.16)
ax.set_xlabel('Characteristic Cell Length [$m$]')
ax.set_ylabel('Maximum RMS Error')
plt.plot(numCell_Res[0:6, 2], numCell_Res[0:6, 3], lw=1.5, color='r', linestyle='', marker='x',
         label='SV weak, dt = {}'.format(caseArray[0][5]/caseArray[0][6]))
plt.plot(numCell_Res[6:12, 2], numCell_Res[6:12, 3], lw=1.5, color='b', linestyle='', marker='o',
         label='SE weak, dt = {}'.format(caseArray[6][5]/caseArray[6][6]))

#create quadratic line to plot
xQuad = np.linspace(numCell_Res[0,2], numCell_Res[5,2], 1000)
scale = 70
yQuad = scale*np.square(xQuad)
plt.plot(xQuad, yQuad, lw=1.0, color='k', label='Quadratic Trend ({}$x^2$)'.format(scale))

ax.legend()
plt.grid()
#plt.show()
plt.savefig(os.path.join(plotDir, 'analyticRMSErrorVsAvCellLength.png'), dpi=500, bbox_inches='tight')
ax.set_xlim(1e-3, 1e-2)
ax.set_ylim(1e-4, 1e-2)
plt.xscale('log')
plt.yscale('log')
plt.savefig(os.path.join(plotDir, 'analyticRMSErrorVsAvCellLengthLogScale.png'), dpi=500, bbox_inches='tight')
plt.close(fig)

strongGradient = (np.log(numCell_Res[5,3]) - np.log(numCell_Res[1,3]))/ \
                 (np.log(numCell_Res[5,2]) - np.log(numCell_Res[1,2]))

# print(strongGradient)

#Plot integral error against average cell Length
fig, ax = plt.subplots(1, 1)
# ax.set_xlim(0, 0.04)
# ax.set_ylim(0, 0.16)
ax.set_xlabel('Characteristic Cell Length [$m$]')
ax.set_ylabel('Maximum integral error')
plt.plot(numCell_Res[0:6, 2], numCell_Res[0:6, 4], lw=1.5, color='r', linestyle='', marker='x',
         label='SV weak, dt = {:.1E}'.format(caseArray[0][5]/caseArray[0][6]))
plt.plot(numCell_Res[6:12, 2], numCell_Res[6:12, 4], lw=1.5, color='b', linestyle='', marker='o',
         label='SE weak, dt = {:.1E}'.format(caseArray[6][5]/caseArray[6][6]))

#create quadratic line to plot
xQuad = np.linspace(numCell_Res[0,2], numCell_Res[5,2], 1000)
scale = 10000
yQuad = scale*np.square(xQuad)
plt.plot(xQuad, yQuad, lw=1.0, color='k', label='Quadratic Trend ({}$x^2$)'.format(scale))

ax.legend()
plt.grid()
#plt.show()
plt.savefig(os.path.join(plotDir, 'analyticIntErrorVsAvCellLength.png'), dpi=500, bbox_inches='tight')
ax.set_xlim(1e-3, 1e-2)
ax.set_ylim(1e-2, 1)
plt.xscale('log')
plt.yscale('log')
plt.savefig(os.path.join(plotDir, 'analyticIntErrorVsAvCellLengthLogScale.png'), dpi=500, bbox_inches='tight')
plt.close(fig)

strongGradient = (np.log(numCell_Res[5,4]) - np.log(numCell_Res[1,4]))/ \
                 (np.log(numCell_Res[5,2]) - np.log(numCell_Res[1,2]))

# print(strongGradient)

# ------------# Plot analytical RMS error for 0.3 second run for all time steps#---------------#

timeIntScheme = 'SV'
tFinal = 0.1
caseName = 'analytical_{}_t{}_timeVariation'.format(timeIntScheme, tFinal).replace('.','_')
outputDir = os.path.join('output', caseName)
if not os.path.exists(outputDir):
    print('The output dir {} doesn\'t exist'.format(outputDir))
    quit()

plotDir = os.path.join(outputDir, 'plots')
if not os.path.exists(plotDir):
    os.mkdir(plotDir)
if timeIntScheme == 'SE':
    caseArray = [['R', timeIntScheme, 'weak', 120, 'analytical', tFinal, 50],
                 ['R', timeIntScheme, 'weak', 120, 'analytical', tFinal, 100],
                 ['R', timeIntScheme, 'weak', 120, 'analytical', tFinal, 200],
                 ['R', timeIntScheme, 'weak', 120, 'analytical', tFinal, 400],
                 ['R', timeIntScheme, 'weak', 120, 'analytical', tFinal, 800],
                 ['R', timeIntScheme, 'weak', 120, 'analytical', tFinal, 1600]]
else:
    caseArray = [['R', timeIntScheme, 'weak', 120, 'analytical', tFinal, 200],
                 ['R', timeIntScheme, 'weak', 120, 'analytical', tFinal, 400],
                 ['R', timeIntScheme, 'weak', 120, 'analytical', tFinal, 800],
                 ['R', timeIntScheme, 'weak', 120, 'analytical', tFinal, 1600]]

colorVec = ['c', 'b', 'g', 'm', 'r', 'k']

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
    # RMS error
    numStepsRes[count,1] = np.max(dataArray[:,8])
    # sqrt the integral^2 error
    numStepsRes[count,2] = np.sqrt(np.max(dataArray[:,9]))
    plt.plot(dataArray[:,0], dataArray[:, 8], lw=1.5, color=colorVec[count], linestyle='-',
            label='dt = {:.5f}'.format(caseArray[count][5]/caseArray[count][6]))

plt.xlabel('Time [s]')
plt.ylabel('RMS Error')
plt.xlim(0.0, tFinal)
#if timeIntScheme == 'SE':
#    plt.ylim(0.0, 0.004)
#else:
#    plt.ylim(0.0, 0.003)

plt.legend()
plt.savefig(os.path.join(plotDir, 'analyticRMSErrorOverTime{}.png'.format(caseArray[0][1])),
            dpi=500, bbox_inches='tight')
plt.close(fig)

# plot int error over time
fig, ax = plt.subplots(1, 1)
for count, dir in enumerate(outputSubDirArray):
    dataArray = np.load(os.path.join(dir, 'H_array.npy'))
    plt.plot(dataArray[:,0], np.sqrt(dataArray[:, 9]), lw=1.5, color=colorVec[count], linestyle='-',
             label='dt = {:.5f}'.format(caseArray[count][5]/caseArray[count][6]))

plt.xlabel('Time [s]')
plt.ylabel('Integral Error')
plt.xlim(0.0, tFinal)
#if timeIntScheme == 'SE':
#    plt.ylim(0.0, 0.01)
#else:
#    plt.ylim(0.0, 0.003)

plt.legend()
plt.savefig(os.path.join(plotDir, 'analyticIntErrorOverTime{}.png'.format(caseArray[0][1])),
            dpi=500, bbox_inches='tight')
plt.close(fig)

# plot p and p_exact vs time
fig, ax = plt.subplots(1, 1)
for count, dir in enumerate(outputSubDirArray):
    dataArray = np.load(os.path.join(dir, 'H_array.npy'))
    plt.plot(dataArray[:,0], dataArray[:, 10], lw=1.5, color=colorVec[count], linestyle='-',
             label='p numerical, dt = {:.5f}'.format(caseArray[count][5]/caseArray[count][6]))
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
plt.savefig(os.path.join(plotDir, 'analyticPVsTime{}.png'.format(timeIntScheme)), dpi=500, bbox_inches='tight')
plt.close(fig)

# plot Energy resiudal vs time for analytical comparison
fig, ax = plt.subplots(1, 1)
for count, dir in enumerate(outputSubDirArray):
    dataArray = np.load(os.path.join(dir, 'H_array.npy'))
    plt.plot(dataArray[:,0], dataArray[:, 2], lw=1.5, color=colorVec[count], linestyle='-',
             label='dt = {:.5f}'.format(caseArray[count][5]/caseArray[count][6]))

plt.xlabel('Time [s]')
plt.ylabel('Energy Residual (J)')
plt.xlim(0.0, tFinal)
#if timeIntScheme == 'SE':
#    plt.ylim(0.0, 0.01)
#else:
#    plt.ylim(0.0, 0.003)

plt.legend()
plt.savefig(os.path.join(plotDir, 'analyticEnergyResidualOverTime{}.png'.format(timeIntScheme)),
            dpi=500, bbox_inches='tight')
plt.close(fig)

# ------------# Plot analytical RMS error over step size#---------------#

fig, ax = plt.subplots(1, 1)
plt.plot(numStepsRes[:, 0], numStepsRes[:, 1], lw=1.5, color='r', linestyle='', marker='x',
         label='{} weak'.format(timeIntScheme))
#create linear line to plot
xLin = np.linspace(numStepsRes[0,0], numStepsRes[len(caseArray)-1,0], 1000)
if timeIntScheme == 'SE':
    scale = 1.5
elif timeIntScheme == 'SV':
    scale = 0.04
else:
    scale = 40
yLin = scale*xLin
plt.plot(xLin, yLin, lw=1.0, color='k', label='Linear Trend ({}$x$)'.format(scale))
##create quadratic line to plot
#xQuad = np.linspace(numStepsRes[0,0], numStepsRes[5,0], 1000)
#scale = 400
#yQuad = scale*np.square(xQuad)
#plt.plot(xQuad, yQuad, lw=1.0, color='k', label='Quadratic Trend ($400x^2$)')
##create Cubic line to plot
#xCubic = np.linspace(numStepsRes[0,0], numStepsRes[5,0], 1000)
#scale = 4000
#yCubic = scale*np.power(xQuad, 3)
#plt.plot(xCubic , yCubic, lw=1.0, color='k', linestyle='--', label='Cubic Trend ($4000x^3$)')
plt.xlabel('Time Step Size [s]')
plt.ylabel('Max RMS Error')
plt.legend()
plt.grid()
#plt.show()
plt.savefig(os.path.join(plotDir, 'analyticRMSErrorVsStepSize{}.png'.format(timeIntScheme)), dpi=500, bbox_inches='tight')
plt.xlim(1e-5, 1e-2)
plt.ylim(1e-4, 1e-2)
plt.xscale('log')
plt.yscale('log')
plt.savefig(os.path.join(plotDir, 'analyticRMSErrorVsStepSize{}LogScale.png'.format(timeIntScheme)), dpi=500, bbox_inches='tight')
plt.close(fig)

# ------------# Plot analytical integral error over step size#---------------#

fig, ax = plt.subplots(1, 1)
plt.plot(numStepsRes[:, 0], numStepsRes[:, 2], lw=1.5, color='r', linestyle='', marker='x',
         label='{} wave vs analytic'.format(timeIntScheme))
#create linear line to plot
xLin = np.linspace(numStepsRes[0,0], numStepsRes[len(caseArray)-1,0], 1000)
if timeIntScheme == 'SE':
    scale = 150
elif timeIntScheme == 'SV':
    scale = 0.04
else:
    scale = 4000

yLin = scale*xLin
plt.plot(xLin, yLin, lw=1.0, color='k', label='Linear Trend ({}$x$)'.format(scale))
##create quadratic line to plot
#xQuad = np.linspace(numStepsRes[0,0], numStepsRes[5,0], 1000)
#scale = 400
#yQuad = scale*np.square(xQuad)
#plt.plot(xQuad, yQuad, lw=1.0, color='k', label='Quadratic Trend ($400x^2$)')
##create Cubic line to plot
#xCubic = np.linspace(numStepsRes[0,0], numStepsRes[5,0], 1000)
#scale = 4000
#yCubic = scale*np.power(xQuad, 3)
#plt.plot(xCubic , yCubic, lw=1.0, color='k', linestyle='--', label='Cubic Trend ($4000x^3$)')
plt.xlabel('Time Step Size [s]')
plt.ylabel('Max Integral Error')
plt.legend()
plt.grid()
#plt.show()
plt.savefig(os.path.join(plotDir, 'analyticIntErrorVsStepSize{}.png'.format(timeIntScheme)), dpi=500, bbox_inches='tight')
plt.xlim(1e-5, 1e-2)
plt.ylim(8e-3, 1)
plt.xscale('log')
plt.yscale('log')
plt.savefig(os.path.join(plotDir, 'analyticIntErrorVsStepSize{}LogScale.png'.format(timeIntScheme)), dpi=500, bbox_inches='tight')
plt.close(fig)


# -----# Plot the analytic error for long times for various schemes, wave model#-------#

caseName = 'analytical_t20_0_schemeVariation'
outputDir = os.path.join('output', caseName)
if not os.path.exists(outputDir):
    print('The output dir {} doesn\'t exist'.format(outputDir))
    quit()

plotDir = os.path.join(outputDir, 'plots')
if not os.path.exists(plotDir):
    os.mkdir(plotDir)

tFinal = 10.0
# TODO Run these at nx = 120 or 160
caseArray = [['R', 'EH', 'weak', 80, 'analytical', 10.0, 20000],
            ['R', 'IE', 'weak', 80, 'analytical', 10.0, 40000],
            ['R', 'SE', 'weak', 80, 'analytical', 10.0, 40000],
            ['R', 'SV', 'weak', 80, 'analytical', 10.0, 20000]]

colorVec = ['g', 'b', 'r', 'k']
lineStyleArray = ['-', '-', '-', '-']

outputSubDirArray = []
for caseVec in caseArray:
    subDir = caseVec[0] + '_' + caseVec[1] + '_' +caseVec[2] + \
             '_nx' + str(caseVec[3]) + '_' + caseVec[4] + '_t' + \
             str(caseVec[5]).replace('.','_') + '_steps' + str(caseVec[6])

    outputSubDirArray.append(os.path.join(outputDir, subDir))

# Make plot for RMS error over long times

fig, ax = plt.subplots(1, 1)
for count, dir in enumerate(outputSubDirArray):
    dataArray = np.load(os.path.join(dir, 'H_array.npy'))
    plt.plot(dataArray[:,0], dataArray[:, 8], lw=0.2, color=colorVec[count], linestyle=lineStyleArray[count],
             label='{}, dt = {}'.format(caseArray[count][1], caseArray[count][5]/caseArray[count][6]))

plt.xlabel('Time [s]')
plt.ylabel('RMS Error')
plt.xlim(0.0, tFinal)
plt.ylim(0, 0.06)

plt.legend()
plt.savefig(os.path.join(plotDir, 'analyticRMSLongTimeSchemeVariation.png'),
            dpi=500, bbox_inches='tight')
plt.close(fig)

# Make plot for Integral error over long times

fig, ax = plt.subplots(1, 1)
for count, dir in enumerate(outputSubDirArray):
    dataArray = np.load(os.path.join(dir, 'H_array.npy'))
    plt.plot(dataArray[:,0], np.sqrt(dataArray[:, 9]), lw=0.2, color=colorVec[count], linestyle=lineStyleArray[count],
             label='{}, dt = {}'.format(caseArray[count][1], caseArray[count][5]/caseArray[count][6]))

plt.xlabel('Time [s]')
plt.ylabel('Integral Error')
plt.xlim(0.0, tFinal)
plt.ylim(0, 4)

plt.legend()
plt.savefig(os.path.join(plotDir, 'analyticIntErrorLongTimeSchemeVariation.png'),
            dpi=500, bbox_inches='tight')
plt.close(fig)


# ------------# Plotting results for interconnection wave and electromechanical #---------------#

# TODO remove this quit
quit()

subDir = 'R_SV_weak_IC_t20_steps40000'
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
             label='em Residual')
    #plot check
    totalResid2 = dataArray[:, 6] +dataArray[:, 7] - dataArray[:,5]
    # plt.plot(dataArray[:, 0], totalResid2, lw=0.5, color='c', linestyle=lineStyleArray[count],
    #         label='check TotalResidual')

ax.legend()
# plt.show()
plt.savefig(os.path.join(plotDir, 'ICEnergyRes.png'), dpi=500, bbox_inches='tight')
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
plt.savefig(os.path.join(plotDir, 'ICHamiltonian.png'), dpi=500, bbox_inches='tight')
# overwrite limits if we want to zoom
ax.set_xlim(0.25, tFinal)
ax.set_ylim(0.009, 0.011)
plt.savefig(os.path.join(plotDir, 'ICHamiltonianZoom.png'), dpi=500, bbox_inches='tight')
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
    plt.plot(dataArray[:, 0], dataArray[:, 3], lw=0.5, color=colorVec[count], linestyle=lineStyleArray[count],
             label='Interconnection SV')

ax.legend()
#plt.show()
plt.savefig(os.path.join(plotDir, 'ICDisplacement.png'), dpi=500, bbox_inches='tight')
plt.close(fig)

# ------------# Plot max energy residual over step size#---------------#

caseArray = [['R', 'SV', 'weak', 120, 'IC', 4.5, 1500],
             ['R', 'SV', 'weak', 120, 'IC', 4.5, 3000],
             ['R', 'SV', 'weak', 120, 'IC', 4.5, 4500],
             ['R', 'SV', 'weak', 120, 'IC', 4.5, 6000],
             ['R', 'SV', 'weak', 120, 'IC', 4.5, 7500],
             ['R', 'SV', 'weak', 120, 'IC', 4.5, 9000]]
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
    #TODO(Finbar) Include str(caseVec[3]) in the below string when everything is rerun
    subDir = caseVec[0] + '_' + caseVec[1] + '_' +caseVec[2] +\
                '_' + caseVec[4] + '_t' +\
                str(caseVec[5]).replace('.','_') + '_steps' + str(caseVec[6])

    outputSubDirArray.append(os.path.join(outputDir, subDir))

numStepsRes = np.zeros((18,2))
for count, dir in enumerate(outputSubDirArray):
    dataArray = np.load(os.path.join(dir, 'H_array.npy'))
    numStepsRes[count,0] = caseArray[count][5]/(caseArray[count][6])
    numStepsRes[count,1] = np.max(abs(dataArray[:,2]))


fig, ax = plt.subplots(1, 1)
plt.plot(numStepsRes[0:6, 0], numStepsRes[0:6, 1], lw=1.5, color='r', linestyle='', marker='x',
         label='SV wave-EM 3DOF')
plt.plot(numStepsRes[6:12, 0], numStepsRes[6:12, 1], lw=1.5, color='b', linestyle='', marker='o',
         label='SV wave-EM 4DOF')
plt.plot(numStepsRes[12:18, 0], numStepsRes[12:18, 1], lw=1.5, color='g', linestyle='', marker='o',
         label='SE wave-EM 4DOF')
#create linear line to plot
xLin = np.linspace(numStepsRes[0,0], numStepsRes[5,0], 1000)
scale = 800
yLin = scale*xLin
plt.plot(xLin, yLin, lw=1.0, color='k', label='Linear Trend ($800x$)')
#create quadratic line to plot
xQuad = np.linspace(numStepsRes[0,0], numStepsRes[5,0], 1000)
scale = 400
yQuad = scale*np.square(xQuad)
plt.plot(xQuad, yQuad, lw=1.0, color='k', label='Quadratic Trend ($400x^2$)')
#create Cubic line to plot
#xCubic = np.linspace(numStepsRes[0,0], numStepsRes[5,0], 1000)
#scale = 4000
#yCubic = scale*np.power(xQuad, 3)
#plt.plot(xCubic , yCubic, lw=1.0, color='k', linestyle='--', label='Cubic Trend ($4000x^3$)')
plt.xlabel('Time Step Size [s]')
plt.ylabel('Maximum Energy Residual [J]')
plt.legend()
plt.grid()
#plt.show()
plt.savefig(os.path.join(plotDir, 'ICEnergyResVsNumSteps.png'), dpi=500, bbox_inches='tight')
plt.xlim(3e-4, 4e-3)
# plt.ylim(9e-5, 1e-2)
plt.xscale('log')
plt.yscale('log')
plt.savefig(os.path.join(plotDir, 'ICEnergyResVsNumStepsLogScale.png'), dpi=500, bbox_inches='tight')
plt.close(fig)


# ------------# Plotting results for square domain interconnection#---------------#

subDir = 'S_1C_SV_weak_IC_t8_steps16000'
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
             label='em Residual')
    #plot check
    totalResid2 = dataArray[:, 6] +dataArray[:, 7] - dataArray[:,5]
    # plt.plot(dataArray[:, 0], totalResid2, lw=0.5, color='c', linestyle=lineStyleArray[count],
    #         label='check TotalResidual')

ax.legend()
# plt.show()
plt.savefig(os.path.join(plotDir, 'ICSquareEnergyRes.png'), dpi=500, bbox_inches='tight')
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
plt.savefig(os.path.join(plotDir, 'ICSquareHamiltonian.png'), dpi=500, bbox_inches='tight')

# ------------# Plotting results for 4DOF Interconnection#---------------#

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
             label='em Residual')
    #plot check
    totalResid2 = dataArray[:, 6] +dataArray[:, 7] - dataArray[:,5]
    # plt.plot(dataArray[:, 0], totalResid2, lw=0.5, color='c', linestyle=lineStyleArray[count],
    #         label='check TotalResidual')

ax.legend()
# plt.show()
plt.savefig(os.path.join(plotDir, 'IC4DOFEnergyRes.png'), dpi=500, bbox_inches='tight')
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
plt.savefig(os.path.join(plotDir, 'IC4DOFHamiltonian.png'), dpi=500, bbox_inches='tight')
# overwrite limits if we want to zoom
ax.set_xlim(0.25, tFinal)
ax.set_ylim(0.009, 0.011)
plt.savefig(os.path.join(plotDir, 'IC4DOFHamiltonianZoom.png'), dpi=500, bbox_inches='tight')
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
    plt.plot(dataArray[:, 0], dataArray[:, 3], lw=0.5, color=colorVec[count], linestyle=lineStyleArray[count],
             label='Interconnection SV')

ax.legend()
#plt.show()
plt.savefig(os.path.join(plotDir, 'IC4DOFDisplacement.png'), dpi=500, bbox_inches='tight')
plt.close(fig)






