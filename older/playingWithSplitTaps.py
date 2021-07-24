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
    
    maxSequenceLen = -1
    whichTap = -1
    sumTapLenInSeq = 0
    allTapLensInSeq = []
    
    for i in range(0,splitSequence.shape[1]):

        temp = len(splitSequence[1,i])
        sumTapLenInSeq += temp
        allTapLensInSeq.append(temp)

        if temp > maxSequenceLen:
            maxSequenceLen = temp
            whichTap = i
            
    meanTapLenInSeq = sumTapLenInSeq/splitSequence.shape[1]
    return maxSequenceLen, whichTap, meanTapLenInSeq, allTapLensInSeq


def longestTapInAllData(allSplits):
    
    maxSequenceLen = -1
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
            
        if sum(np.array(allTapLensInSeq)<5):
            print('DUDE!')
            print('Sequence ', i)
            

            
    return maxSequenceLen, sequenceIdx, tapIdx, allMeanLens, allTapLens


#%%


dataPath = r'C:\data\icef\tapping\raw data'

txtfile = 'allSplits.txt'  

allData = readAllDataAndSplitFromTxt(dataPath, txtfile)




#%%
# split[1,34].shape # prvi kanal (osa ziroskopa), 34i tap


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

#how about all taps in dataset?

allTapsInDataset = diagConcats.sum()
plt.figure()
plt.subplot(1,2,1)
plt.boxplot(allTapsInDataset)
plt.title('All taps in dataset: boxplot')
plt.subplot(1,2,2)
plt.hist(allTapsInDataset)
plt.title('All taps in dataset: histogram')
plt.show()

medianTapLen = np.median(allTapsInDataset)
print('MEDIAN tap length: ', medianTapLen)
print('MAX tap length: {} for tap {} in sequence {}. Diagnosis: {} '.format(maxTapLen, whichTap, whichSequence, maxMeasurementWinner.diagnosis))
