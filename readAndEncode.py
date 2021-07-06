# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 16:07:04 2020

@author: minja
"""

from measurement import measurement
import os
from tqdm import tqdm
from pathlib import Path
from scipy import io
import numpy as np
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
import json


def readAdjust(filename):
    
    sig = io.loadmat(filename)
    
    file = os.path.basename(filename)
    
    diagnosisFolder = Path(filename).parents[1]
    
    diagnosis = os.path.basename(diagnosisFolder)

    if file[2] == '_':
        initials = file[0:2]
        date = file[3:13]
        time_of_measurement = file[14:22]
    else:
        initials = file[0:3]
        date = file[4:14]
        time_of_measurement = file[15:23]

    temp = measurement(sig['fsr'][0],
                       sig['gyro1'][0],
                       sig['gyro1'][1],
                       sig['gyro1'][2],
                       sig['gyro2'][0],
                       sig['gyro2'][1],
                       sig['gyro2'][2],
                       sig['tap_task'][0], 
                       sig['time'][0],
                       sig['time_tap'][0],
                       sig['ttapstart'][0,0], 
                       sig['ttapstop'][0,0], 
                       diagnosis,
                       initials,
                       date,
                       time_of_measurement)
                           
    return temp


def readPatientFiles(folder):
    
    data = []
    
    for r, _, files in os.walk(folder):
        for file in files:
            data.append(readAdjust(os.path.join(r,file)))
            
    return data




def encodeDiagnosis(diagnosis):
    le = LabelEncoder()
    le.fit(['CTRL','MSA','PD','PSP'])
    # This is the encoding:
    # CTRL -----> 0 -----> [1,0,0,0]
    # MSA ----->  1 -----> [0,1,0,0]
    # PD -----> 2 -----> [0,0,1,0]
    # PSP -----> 3 -----> [0,0,0,1]
    
    diagnosis = le.transform([diagnosis])
    oneHotDiagnosis = np.zeros((1,4),dtype='uint8')
    idx = diagnosis[0]
    oneHotDiagnosis[0][idx] = 1
    return oneHotDiagnosis[0]

def kodiraj(numPrediction):
    switch = {0:'CTRL',
              1:'MSA',
              2:'PD',
              3:'PSP'}
    
    return switch[numPrediction]

# Distribution of classes?
def classDistribution(sigs):
    diagnoses = [sig.diagnosis for sig in sigs]
    CTRL = np.sum([d =='CTRL' for d in diagnoses])
    PD = np.sum([d =='PD' for d in diagnoses])
    PSP = np.sum([d =='PSP' for d in diagnoses])
    MSA = np.sum([d =='MSA' for d in diagnoses])

    print('There are {} CTRL subjects, {} MSA, {} PD and {} PSP'. format(CTRL, MSA, PD, PSP))

    plt.bar(['CTRL','MSA','PD','PSP'],[CTRL,MSA,PD,PSP])
    plt.title('Number of signals recorded by diagnosis')
    plt.show()
    return {'CTRL':CTRL,'MSA':MSA,'PD':PD,'PSP':PSP}



def cropAndReshape(signals, nSeconds, test=False):

    # test argument means whether you are forming a test set or not
    # because if you are only one crop is taken randomly
    # as opposed to all crops
    
    X = []  # signals
    Y = []  # diagnoses
  
    for sig in tqdm(signals):
        crops = sig.packAndCrop(nSeconds)
        if test:
            keepIndex = np.random.randint(0,len(crops))
            crops = [crops[keepIndex]]
            
        X = X + crops
        nCrops = len(crops)
        
        for i in range(0,nCrops):
            Y.append(encodeDiagnosis(sig.diagnosis))
    
    l = len(X)
    indices = np.arange(l)
    np.random.shuffle(indices)
    
    Xshuffle = []
    Yshuffle = []
    for i in indices:
        Xshuffle.append(X[i])
        Yshuffle.append(Y[i])
    
    Y = np.reshape(Yshuffle,(l,4))
    X = np.reshape(Xshuffle, (l,Xshuffle[0].shape[0],Xshuffle[0].shape[1]))
    X = np.swapaxes(X,1,2)
    print('Shape of X: ', X.shape)

    return X,Y

def getUniquePatients(root):
    # get Unique Patients and corresponding diagnoses
    allPatientFolders = []
    allPatientDiagnoses = []
    
    for r,subdirs,_ in tqdm(os.walk(root)):
        if not subdirs: # if you are in a patient folder
            allPatientFolders.append(r)
            allPatientDiagnoses.append(os.path.basename(os.path.dirname(r)))
    
    
    allPatientFolders =  np.array(allPatientFolders)
    allPatientDiagnoses = np.array(allPatientDiagnoses) # for stratification
    
    print('INFO:')
    print('There are a total of {} unique patients'.format(len(allPatientFolders)))
    
    return allPatientFolders, allPatientDiagnoses

def readAllDataAndAutoSplit(dataPath, integrateFirst, filename):
    
    print('Reading all relevant data. Auto-splitting taps...\n')     
  
    allData = []
    allPatientFolders, allPatientDiagnoses = getUniquePatients(dataPath)
    
    for patientFolder in tqdm(allPatientFolders):
        currentPatientMeasurements = readPatientFiles(patientFolder)
        for mes in currentPatientMeasurements:
            if not mes.isRightHand():
                continue
            
            temp = {}
            intermediate_signal, peak_indices = mes.findTapSplits(integrateFirst)
            temp['intermediate_signal'] = intermediate_signal
            temp['peak_indices'] = peak_indices
            temp['measurement'] = mes
            allData.append(temp)
    return allData

def readAllDataAndSplitFromTxt(dataPath, txtfile, integrateFirst):
    print('Reading all relevant data. Reading split points from file...\n')     
  
    allData = []
    allPatientFolders, allPatientDiagnoses = getUniquePatients(dataPath)
    
    splitDicts = []
    with open(txtfile, 'r') as f:
        lines = f.readlines()
        for line in lines:
            temp = json.loads(line)
            temp['allSplitPoints'] = [int(point) for point in temp['allSplitPoints']]
            splitDicts.append(temp)
            
        
    
    k = -1
    for patientFolder in tqdm(allPatientFolders):
        currentPatientMeasurements = readPatientFiles(patientFolder)
        for mes in currentPatientMeasurements:
            if not mes.isRightHand():
                continue
            k += 1
            temp = {}
            intermediate_signal, _ = mes.findTapSplits(integrateFirst)
            temp['intermediate_signal'] = intermediate_signal
            temp['peak_indices'] = splitDicts[k]['allSplitPoints']
            temp['measurement'] = mes
            allData.append(temp)
            if splitDicts[k]['id'].lower() != mes.id.lower():
                raise('ID MISMATCH!!!!')
            
            
    return allData


            