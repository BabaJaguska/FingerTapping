# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 20:08:48 2020

@author: minja
"""

#%%
from measurement import measurement
from readAndEncode import *
from time import sleep
import argparse
import matplotlib
from threading import Thread
import json
matplotlib.use('Qt5Agg')

def addPoints(arr, points):
    points = [point for point in points if point not in arr]
    arr = np.append(arr, points)    
    arr = np.sort(arr)
    return arr

def removePoints(arr, roughPoints):
    
    def closest(arr, point):
        
        if point in arr:
            return point
        
        for pointCandidate in range(point - 3, point + 3):
            if pointCandidate in arr:
                return np.where(arr == pointCandidate)
            
        print('POINT {} NOT FOUND.'.format(point))
        
        return -1
    
    point_indices = [closest(arr, point) for point in roughPoints]     
    
    arr = np.delete(arr, point_indices)  
    
    return arr

def modifySignalSplits(splitPoints, modifyDict):
    
    if modifyDict['add']:    
        splitPoints = addPoints(splitPoints, modifyDict['add'])
        
    if modifyDict['sub']:
        splitPoints = removePoints(splitPoints, modifyDict['sub'])
    
    return splitPoints

        
def findMeasurementByID(data, ID):
    
    for idx, datum in enumerate(data):
        if datum['measurement'].id == ID:
            return datum, idx
        
    print('COULD NOT FIND MEASUREMENT')
    return

def writeAllSignalSplitsToFile(data, filename):
    
    if os.path.exists(filename):
        print('REWRITING FILE !!!')           
    
    with open(filename, 'w+') as f:
        for datum in data:
            temp = {'id': datum['measurement'].id,
                    'allSplitPoints': datum['peak_indices'].tolist()}
            print(json.dumps(temp), file = f)

            
    return
            
 

#%%

dataPath = r'C:\data\icef\tapping\raw data'
allData = readAllDataAndAutoSplit(dataPath)
    
#==== CHECK SAVED ====
modifiers = []
try:
    with open('./ModifyPoints.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            modifyFile = json.loads(line)
            modifiers.append(modifyFile)
except:
    print('Could not read the requested file.')

# print(modifiers)

# ========= MODIFY SPLIT POINTS ========
for modifierDict in modifiers:
    print('Modifying ', modifierDict['id'])
    tempDatum, idx = findMeasurementByID(allData, modifierDict['id'])
    print(idx)
    tempSplits = tempDatum['peak_indices']
    allData[idx]['peak_indices'] = modifySignalSplits(tempSplits, modifierDict)
    

# === Write all splits to file
fileToWriteAll = './allSplits.txt'
writeAllSignalSplitsToFile(allData, fileToWriteAll)
    

    
    
    