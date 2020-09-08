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


def readAdjust(filename):
    
    sig = io.loadmat(filename)
    
    file = os.path.basename(filename)
    
    diagnosisFolder = Path(filename).parents[1]
    
    diagnosis = os.path.basename(diagnosisFolder)
    
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
                       file[0:2], 
                       file[3:13],
                       file[14:22])
                           
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