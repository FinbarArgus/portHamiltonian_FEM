import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import paperPlotSetup
import h5py
from scipy import fft

paperPlotSetup.Setup_Plot(3)
plotPNG = False
plotEPS = True

caseName = 'analytical_t1_5_visualisation'
outputDir = os.path.join('output', caseName)
if not os.path.exists(outputDir):
    print('The output dir {} doesn\'t exist'.format(outputDir))
    quit()

plotDir = os.path.join(outputDir, 'plots')
if not os.path.exists(plotDir):
    os.mkdir(plotDir)

caseArray = [['R', 'SV', 'weak', 80, 'analytical', 1.5, 3000]]
caseVec = caseArray[0]

subDir = caseVec[0] + '_' + caseVec[1] + '_' +caseVec[2] + \
             '_nx' + str(caseVec[3]) + '_' + caseVec[4] + '_t' + \
             str(caseVec[5]).replace('.','_') + '_steps' + str(caseVec[6])

dir = os.path.join(outputDir, subDir)

dataArray = np.load(os.path.join(dir, 'H_array.npy'))

timeStepSize = caseVec[5]/(caseVec[6])

pRaw = dataArray[:, 10]
#find location to stop where a full cycle has been
stop = False
count = -1
while not stop:
    if abs(pRaw[0] - pRaw[count])<0.01:
        stop=True
    count = count - 1

tRaw = dataArray[:, 0]
n = len(tRaw) + count
numCopies = 10
nLong = n*numCopies
tFinal = tRaw[count]
sampleRate = n/tFinal

#tVec = tRaw[:n]
tVec = np.zeros((nLong,))
for i in range(numCopies):
    tVec[i*n:(i+1)*n] = tRaw[:n] + i*tFinal

# fig, ax = plt.subplots(1,1)

pExactRaw = dataArray[:, 11]
pVec = np.zeros((nLong,))
pExactVec = np.zeros((nLong,))
for i in range(numCopies):
    pVec[i*n:(i+1)*n] = pRaw[:n]
    pExactVec[i*n:(i+1)*n] = pExactRaw[:n]

pFreq = fft.rfft(pVec)
X = fft.rfftfreq(nLong, 1/sampleRate)
pFreq_final = np.abs(pFreq)

pExactFreq = fft.rfft(pExactVec)
pExactFreq_final = np.abs(pExactFreq)

# find maximum frequency
first_mode_idx = np.argmax(pFreq_final)
first_mode_freq = X[first_mode_idx]
conf_int = X[first_mode_idx+1] - X[first_mode_idx]
print(' first mode frequency is {} with a confidence interval of + or - {}'.format(first_mode_freq, conf_int/2))

fig, ax = plt.subplots(1, 2)

ax[0].plot(tVec, pVec, 'b-')
ax[0].plot(tVec, pExactVec, 'r--')
ax[1].plot(X[:400], pFreq_final[:400], 'bo')
ax[1].plot(X[:400], pExactFreq_final[:400], 'rx')

plt.show()












