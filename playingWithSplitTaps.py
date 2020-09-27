# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 19:14:51 2020

@author: minja
"""



from measurement import measurement
from readAndEncode import *
from time import sleep
import argparse
import matplotlib
from threading import Thread
import json
#%%

def longestTapInSequence(splitSequence):
    
    maxSequenceLen = 0
    whichTap = -1
    meanTapLenInSeq = 0
    allTapLensInSeq = []
    
    for i in range(0,splitSequence.shape[1]):

        temp = len(splitSequence[1,i])
        meanTapLenInSeq += temp
        allTapLensInSeq.append(temp)
        if temp > maxSequenceLen:
            maxSequenceLen = temp
            whichTap = i
            
    meanTapLenInSeq = meanTapLenInSeq/splitSequence.shape[1]
    return maxSequenceLen, whichTap, meanTapLenInSeq, allTapLensInSeq


def longestTapInAllData(allSplits):
    
    maxSequenceLen = 0
    sequenceIdx = -1
    tapIdx = -1
    
    allMeanLens = []
    allTapLens = []
    
    for i, series in enumerate(allSplits):
        tempMax, tempWhichTap, meanTapLenInSeq, allTapLensInSeq = longestTapInSequence(series)
        allMeanLens.append(meanTapLenInSeq)
        allTapLens.append(allTapLensInSeq)
        if tempMax > maxSequenceLen:
            maxSequenceLen = tempMax
            sequenceIdx = i
            tapIdx = tempWhichTap
            
    return maxSequenceLen, sequenceIdx, tapIdx, allMeanLens, allTapLens


#%%


dataPath = r'C:\data\icef\tapping\raw data'

txtfile = 'allSplits.txt'  

allData = readAllDataAndSplitFromTxt(dataPath, txtfile)




#%%
# split[1,34].shape # prvi kanal (osa ziroskopa), 34i tap


plt.figure()
plt.plot(split[1,34])
plt.plot(split[2,34])
plt.plot(split[3,34])
plt.show()


allSplitsTemp = [dataPoint['measurement'].splitTaps(dataPoint['peak_indices']) for dataPoint in allData]

allSplits = [split[0] for split in allSplitsTemp]


maxTapLen, whichSequence, whichTap, allMeanLens, allTapLens = longestTapInAllData(allSplits)

# check it out
maxMeasurementWinner = allData[whichSequence]['measurement']
maxTapWinner = allSplits[whichSequence][:,whichTap]
maxMeasurementWinner.plotSignals()

plt.figure()
for axis in maxTapWinner:
    plt.plot(axis)
plt.show()

plt.figure()
plt.hist(allMeanLens)
plt.title('Tap length distribution')
plt.show()

allDiagnoses = [dataPoint['measurement'].diagnosis for dataPoint in allData]

import pandas as pd
df = pd.DataFrame({'diag':allDiagnoses, 'meanLens': allMeanLens, 'allTapLensInSeq': allTapLens})
diagMeans = df.groupby('diag')['meanLens'].mean()
diagConcats = df.groupby('diag')['allTapLensInSeq'].sum()

# MEAN LENGTHS BY DIAGNOSIS
plt.figure()
diagMeans.plot.bar()
plt.show()

# BOXPLOTS BY DIAGNOSIS
plt.figure()
plt.boxplot(diagConcats)
plt.xticks([1,2,3,4],['CTRL', 'MSA', 'PD', 'PSP'])
plt.title('Tap lengths by diagnosis: boxplot')
plt.show()        

# NUMBER OF EXISTING TAPS BY DIAGNOSIS:
    
numTapsByDiagnosis =  [len(d) for d in diagConcats]
plt.figure()
plt.bar(['CTRL', 'MSA', 'PD', 'PSP'],numTapsByDiagnosis)
plt.title('Total number of taps by diagnosis')
plt.show()


