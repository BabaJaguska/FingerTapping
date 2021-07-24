# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 20:08:48 2020

@author: minja
"""

#%%


import matplotlib
import numpy as np
import os
import json



            
 

#%%

#==== CHECK SAVED ====
modifierFile = './ModifyPoints.txt'
fileToWrite = './allSplits.txt'

def modifyAutoSplits(allData, modifierFile, fileToWrite):
    modifiers = []
    try:
        with open(modifierFile, 'r') as f:
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
        
    
    writeAllSignalSplitsToFile(allData, fileToWrite)

    return 
    

    
    
    