# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 13:37:11 2019

@author: minja
"""
#%%
# =============================================================================
#  load stuff
# =============================================================================
import matplotlib.pyplot as plt
import numpy as np
from scipy import io
import time

from tqdm import tqdm
import os
#import pandas as pd
from sklearn.preprocessing import LabelEncoder
#from sklearn.model_selection import train_test_split
#import keras
from keras.models import Model, load_model
from keras.layers import Input, Conv1D, Flatten, Dropout, MaxPooling1D, Dense, Reshape,UpSampling1D
from keras.layers import Activation, BatchNormalization
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.constraints import max_norm
from scipy.signal import decimate
from numpy import ma
path='C:/data/tapping/'
#%%
class measurement:
    def __init__(self, fsr,
                 gyro1x, gyro1y, gyro1z,gyro2x, gyro2y, gyro2z,
                 tap_task, time, time_tap, ttapstart, ttapstop, diagnosis,
                 initials, date, timeOfMeasurement):
        
        # acceleration
#         self.acc1x = acc1x # thumb
#         self.acc1y = acc1y # thumb
#         self.acc1z = acc1z # thumb
#         self.acc1Vec = np.sqrt(np.square(self.acc1x)+
#                                np.square(self.acc1y)+
#                                np.square(self.acc1z))
        
#         self.acc2x = acc2x # forefinger
#         self.acc2y = acc2y # forefinger
#         self.acc2z = acc2z # forefinger
#         self.acc2Vec = np.sqrt(np.square(self.acc2x)+
#                                np.square(self.acc2y)+
#                                np.square(self.acc2z))
        
        decimateRate = 1
        # force
        self.fsr = decimate(fsr,decimateRate)
                
        # angular velocity
        self.gyro1x = decimate(gyro1x,decimateRate) # thumb
        self.gyro1y = decimate(gyro1y,decimateRate) # thumb
        self.gyro1z = decimate(gyro1z,decimateRate) # thumb
        self.gyro1Vec = np.sqrt(np.square(self.gyro1x)+
                                np.square(self.gyro1y)+
                                np.square(self.gyro1z))
        
        self.gyro2x = decimate(gyro2x,decimateRate) # forefinger
        self.gyro2y = decimate(gyro2y,decimateRate) # forefinger
        self.gyro2z = decimate(gyro2z,decimateRate) # forefinger
        self.gyro2Vec = np.sqrt(np.square(self.gyro2x)+
                                np.square(self.gyro2y)+
                                np.square(self.gyro2z))        
        
               
        # other
        self.fs = int(200/decimateRate) # sampling rate [Hz] AFTER DECIMATION
        self.tap_task = tap_task # LHEO/LHEC/RHEO/RHEC (left or right hand/eyes open or closed)
        self.time = np.linspace(0,len(self.gyro1x)/self.fs,len(self.gyro1x))
        self.time_tap = time_tap
        self.ttapstart = ttapstart+0.3 #single value, when the actual signal started SECONDS
        self.ttapstop = ttapstop-0.3 #single value, when the actual signal stopped SECONDS
        self.diagnosis = diagnosis # PD, PSP, MSA, CTRL
        self.initials = initials # person name and surname initials 
        self.date = date # date of recording
        self.timeOfMeasurement = timeOfMeasurement #what time that date
        self.length = len(self.gyro1x)
        self.tappingLength = self.ttapstop - self.ttapstart
       
        self.mvcSustained = max(self.fsr[0:int(ttapstart*self.fs)])
        self.mvcTapping = max(self.fsr[int(ttapstart*self.fs):int(ttapstop*self.fs)])
        #self.mvcTotal = self.mvcTapping if self.mvcTapping>self.mvcSustained else self.mvcSustained
        
        
        #normalize gyro
        
        #optionally:
#        shiftParam1 = min(np.min(self.gyro1x[int(ttapstart*self.fs):int(ttapstop*self.fs)]),
#                           np.min(self.gyro1y[int(ttapstart*self.fs):int(ttapstop*self.fs)]),
#                           np.min(self.gyro1z[int(ttapstart*self.fs):int(ttapstop*self.fs)]))
#        
#        shiftParam2 = min(np.min(self.gyro2x[int(ttapstart*self.fs):int(ttapstop*self.fs)]),
#                           np.min(self.gyro2y[int(ttapstart*self.fs):int(ttapstop*self.fs)]),
#                           np.min(self.gyro2z[int(ttapstart*self.fs):int(ttapstop*self.fs)]))
#        
#        self.gyro1x,self.gyro1y,self.gyro1z = self.gyro1x-shiftParam1,self.gyro1y-shiftParam1,self.gyro1z-shiftParam1
#        self.gyro2x,self.gyro2y,self.gyro2z = self.gyro2x-shiftParam2,self.gyro2y-shiftParam2,self.gyro2z-shiftParam2
#        
#        # mandatory:
#        
#        denominator1 = max(np.max(self.gyro1x[int(ttapstart*self.fs):int(ttapstop*self.fs)]),
#                           np.max(self.gyro1y[int(ttapstart*self.fs):int(ttapstop*self.fs)]),
#                           np.max(self.gyro1z[int(ttapstart*self.fs):int(ttapstop*self.fs)]))
#        
#        denominator2 = max(np.max(self.gyro2x[int(ttapstart*self.fs):int(ttapstop*self.fs)]),
#                           np.max(self.gyro2y[int(ttapstart*self.fs):int(ttapstop*self.fs)]),
#                           np.max(self.gyro2z[int(ttapstart*self.fs):int(ttapstop*self.fs)]))
#        
#        self.gyro1x,self.gyro1y,self.gyro1z = self.gyro1x/denominator1,self.gyro1y/denominator1,self.gyro1z/denominator1
#        self.gyro2x,self.gyro2y,self.gyro2z = self.gyro2x/denominator2,self.gyro2y/denominator2,self.gyro2z/denominator2

        
        # normalize FSR
        self.normalizedFSR = self.fsr/self.mvcSustained
        self.id = self.diagnosis + '_' + self.initials + '_' +self.date 
        
        
        self.gyro1xT = self.gyro1x[int(self.ttapstart*self.fs):int(self.ttapstop*self.fs)]
        self.gyro1yT = self.gyro1y[int(self.ttapstart*self.fs):int(self.ttapstop*self.fs)]
        self.gyro1zT = self.gyro1z[int(self.ttapstart*self.fs):int(self.ttapstop*self.fs)]
        self.gyro2xT = self.gyro2x[int(self.ttapstart*self.fs):int(self.ttapstop*self.fs)]
        self.gyro2yT = self.gyro2y[int(self.ttapstart*self.fs):int(self.ttapstop*self.fs)]
        self.gyro2zT = self.gyro2z[int(self.ttapstart*self.fs):int(self.ttapstop*self.fs)]
        
    def packAndCrop(self, seconds):
        
        # matrix for feeding into a net
        # shape: nsignals(13 or 5 probably) x seconds*SamplingRate 
        

        #allPacked = np.concatenate(([self.gyro1x],[self.gyro1y],[self.gyro1z],
                                   # [self.gyro2x],[self.gyro2y],[self.gyro2z]), axis=0) 
        
        
        #allPacked = allPacked[:, int(self.fs*self.ttapstart):int(self.fs*self.ttapstop)]
        
        crops = []
        
       #modelPackage = np.zeros((6,seconds*self.fs),dtype='float32')
        #fin = allPacked.shape[1] if allPacked.shape[1]<seconds*self.fs else seconds*self.fs
        #modelPackage[:,:fin] = allPacked[:,:fin]
        
        #crops.append(modelPackage)
        
        

        
        nCrops = int((len(self.gyro1xT) - seconds*self.fs)/self.fs)
        
        for i in range(nCrops):
#             A = (i+1)*self.fs
#             B = (i+1)*self.fs + seconds*self.fs
            
            
            ######################
            ##transform spherical mozda?

            
             #gyro1xT, gyro1yT, gyro1zT,gyro2xT, gyro2yT, gyro2zT = self.transformSpheric()
            
             gyro1xT, gyro1yT, gyro1zT,gyro2xT, gyro2yT, gyro2zT = self.gyro1xT, self.gyro1yT, self.gyro1zT,self.gyro2xT, self.gyro2yT,self.gyro2zT 
             #integral svake ose
             #gyro1xT, gyro1yT, gyro1zT,gyro2xT, gyro2yT, gyro2zT = np.cumsum(self.gyro1xT), np.cumsum(self.gyro1yT), np.cumsum(self.gyro1zT),np.cumsum(self.gyro2xT), np.cumsum(self.gyro2yT),np.cumsum(self.gyro2zT)
             #from scipy.signal import detrend
             #gyro1xT, gyro1yT, gyro1zT,gyro2xT, gyro2yT, gyro2zT = detrend(gyro1xT), detrend(gyro1yT), detrend(gyro1zT), detrend(gyro2xT), detrend(gyro2yT), detrend(gyro2zT)
            ######################
            
             x1min = np.min(gyro1xT[(i+1)*self.fs:(i+1)*self.fs + seconds*self.fs])
             y1min = np.min(gyro1yT[(i+1)*self.fs:(i+1)*self.fs + seconds*self.fs])
             z1min = np.min(gyro1zT[(i+1)*self.fs:(i+1)*self.fs + seconds*self.fs])
             x2min = np.min(gyro2xT[(i+1)*self.fs:(i+1)*self.fs + seconds*self.fs])
             y2min = np.min(gyro2yT[(i+1)*self.fs:(i+1)*self.fs + seconds*self.fs])
             z2min = np.min(gyro2zT[(i+1)*self.fs:(i+1)*self.fs + seconds*self.fs])
             
#             shift1 = np.min([x1min,y1min,z1min])
#             shift2 = np.min([x2min,y2min,z2min])
             
             ### ako sa spherical radis onda ne mozes da normalizujes po svima
            
             
             ## ako ne onda ok
#             x1min = shift1
#             y1min = shift1
#             z1min = shift1
#             x2min = shift2
#             y2min = shift2
#             z2min = shift2
             #####
             
             ##probaj bez ikakvog skaliranja
             x1min = 0
             y1min = 0
             z1min = 0
             x2min = 0
             y2min = 0
             z2min = 0
              
             cropgx1, cropgy1, cropgz1 = gyro1xT[(i+1)*self.fs:(i+1)*self.fs + seconds*self.fs]-x1min,gyro1yT[(i+1)*self.fs:(i+1)*self.fs + seconds*self.fs]-y1min,gyro1zT[(i+1)*self.fs:(i+1)*self.fs + seconds*self.fs]-z1min
             cropgx2, cropgy2, cropgz2 = gyro2xT[(i+1)*self.fs:(i+1)*self.fs + seconds*self.fs]-x2min,gyro2yT[(i+1)*self.fs:(i+1)*self.fs + seconds*self.fs]-y2min,gyro2zT[(i+1)*self.fs:(i+1)*self.fs + seconds*self.fs]-z2min
             
             
             dx1 = np.max(cropgx1)
             dy1 = np.max(cropgy1)
             dz1 = np.max(cropgz1)
             dx2 = np.max(cropgx2)
             dy2 = np.max(cropgy2)
             dz2 = np.max(cropgz2)
             
#             import code
#             code.interact(local = locals())
             
#             denom1 = np.max([dx1,dy1, dz1])
#             denom2 = np.max([dx2, dy2, dz2])
             
             
             #### ako nije sphera onda
#             dx1 = denom1
#             dy1 = denom1
#             dz1 = denom1
#             
#             dx2 = denom2
#             dy2 = denom2
#             dz2 = denom2
             
             ###
              ##probaj bez ikakvog skaliranja
              
             dx1 = 1
             dy1 = 1
             dz1 = 1
             dx2 = 1
             dy2 = 1
             dz2 = 1
             
             cropgx1, cropgy1, cropgz1 = cropgx1/dx1, cropgy1/dy1, cropgz1/dz1
             cropgx2, cropgy2, cropgz2 = cropgx2/dx2, cropgy2/dy2, cropgz2/dz2
             
             
#            temp1 = allPacked[:3,((i+1)*self.fs):((i+1)*self.fs + seconds*self.fs)]
#            temp2 = allPacked[3:6,((i+1)*self.fs):((i+1)*self.fs + seconds*self.fs)]
#            up1 = temp1 - np.min(temp1)
#            up2 = temp2 - np.min(temp2)
#            temp11 = up1/np.max(up1)
#            temp22 = up2/np.max(up2)
#            #temp = np.transpose(temp)
#            temp = np.concatenate((temp11,temp22),axis = 0)
             temp = np.concatenate(([cropgx1], [cropgy1], [cropgz1],[cropgx2], [cropgy2], [cropgz2]))
             #temp = np.concatenate(([cropgx1], [cropgx2]))
             #ajmo samo vektorski intenzitet
             #temp = np.concatenate(([cropgx1],[cropgx2]))
             crops.append(temp)
            
        
        return crops
       
    def sumUp(self):
        temp = {'lenFsr': len(self.fsr),
               'lenGyroThumb': len(self.gyro1x),
               'lenGyroForefinger': len(self.gyro2x),
               'lenTime': len(self.time)}
        temp['MATCHING_LENGTHS'] =  len(set(temp.values()))==1
        temp['durationInSecs'] = self.length/self.fs
        return temp
    
    def plotSignals(self, xlim = []):
        # Optionally pass a tuple for zooming in on the x axis (xmin,xmax)
        if len(xlim) <2:
            xlim = (0,self.length/self.fs)
            
#         # accelerometers     
#         plt.figure(figsize=(16,12))
# #         plt.plot(self.time,self.acc1x)
# #         plt.plot(self.time,self.acc1y)
# #         plt.plot(self.time,self.acc1z)
# #         plt.plot(self.time,self.acc2x)
# #         plt.plot(self.time,self.acc2y)
# #         plt.plot(self.time,self.acc2z)
#         plt.axvline(x = self.ttapstart,color = 'b')
#         plt.axvline(x = self.ttapstop, color = 'r')
#         plt.xlim(xlim)
#         plt.legend(('AccThumbX', 'AccThumbY', 'AccThumbZ',
#                     'AccIndexX', 'AccIndexY', 'AccIndexZ'))
#         plt.title('Accelerometers for subject ' + self.initials +
#                   ' Date: ' + self.date +
#                   ' Diagnosis: ' + self.diagnosis +
#                    'Task: ' + self.tap_task)
#         plt.show()
        
#         # vector accelerometer
#         plt.figure(figsize=(16,5))
#         plt.plot(self.time, self.acc1Vec)
#         plt.plot(self.time, self.acc2Vec)
#         plt.axvline(x = self.ttapstart,color = 'b')
#         plt.axvline(x = self.ttapstop, color = 'r')
#         plt.xlim(xlim)
#         plt.legend(('AccThumbVector', 'AccIndexVector'))
#         plt.title('Accelerometer Vectors')
#         plt.show()
        
        # gyro1
        plt.figure(figsize = (16,5))
        plt.plot(self.time,self.gyro1x)
        plt.plot(self.time,self.gyro1y)
        plt.plot(self.time,self.gyro1z)
        plt.legend(['GyroThumbX', 'GyroThumbY', 'GyroThumbZ'])
        plt.axvline(x = self.ttapstart,color = 'b')
        plt.axvline(x = self.ttapstop, color = 'r')
        plt.xlim(xlim)
        plt.title('Gyro THUMB data')
        
        # gyro2
        plt.figure(figsize = (16,5))
        plt.plot(self.time,self.gyro2x)
        plt.plot(self.time,self.gyro2y)
        plt.plot(self.time,self.gyro2z)
        plt.legend(['GyroIndexX', 'GyroIndexY', 'GyroIndexZ'])
        plt.axvline(x = self.ttapstart,color = 'b')
        plt.axvline(x = self.ttapstop, color = 'r')
        plt.xlim(xlim)
        plt.title('Gyro INDEX data')
        
        # vector gyro
        plt.figure(figsize=(16,5))
        plt.plot(self.time, self.gyro1Vec)
        plt.plot(self.time, self.gyro2Vec)
        plt.axvline(x = self.ttapstart,color = 'b')
        plt.axvline(x = self.ttapstop, color = 'r')
        plt.xlim(xlim)
        plt.legend(('GyroThumbVector', 'GyroIndexVector'))
        plt.title('Gyro Vectors')
        plt.show()
        
        # force
        plt.figure(figsize=(16,5))
        plt.plot(self.time,self.normalizedFSR)
        plt.xlim(xlim)
        plt.axvline(x = self.ttapstart,color = 'b')
        plt.axvline(x = self.ttapstop, color = 'r')
        plt.title('Normalized FSR')
        plt.show()
        
        
    def transformSpheric(self):
        x1 = self.gyro1xT
        y1 = self.gyro1yT
        z1 = self.gyro1zT
            
        x2 = self.gyro2xT
        y2 = self.gyro2yT
        z2 = self.gyro2zT
            
        def transpher(x,y,z):
            
            R = np.sqrt(x**2 + y**2 + z**2)
            mask = np.logical_and(np.equal(x,0), np.equal(y,0))
                
            xMasked = ma.masked_array(x, mask)
            yMasked = ma.masked_array(y, mask)
            zMasked = ma.masked_array(z, mask)
      
            phi = np.arccos(xMasked/np.sqrt(np.square(xMasked) + np.square(yMasked)))
            theta = np.arccos(zMasked/np.sqrt(np.square(xMasked) + np.square(yMasked) + np.square(zMasked)))
                
            phi = ma.filled(phi, 0)
            theta = ma.filled(theta,0)
                
            return R,phi,theta
            
            
        R1, phi1, theta1 = transpher(x1,y1,z1)
        R2, phi2, theta2 = transpher(x2,y2,z2)
            
        return R1, phi1, theta1, R2, phi2, theta2
        
            

#%%
def readAdjust(root, directory, file):
    
    sig = io.loadmat(root + directory + '/' + file)
    
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
                       directory,
                       file[0:2], 
                       file[3:13],
                       file[14:22])
                           
    return temp
#%%
# =============================================================================
#     READ THE DATA
# =============================================================================

root = 'C:/data/tapping/raw data/'

_,dirs,_ = os.walk(root).__next__()

sigs = []
for d in tqdm(dirs):
    _,_,files = os.walk(root+d).__next__()
    sigs = sigs +[readAdjust(root, d, file) for file in files]
#%%
# =============================================================================
#     ckecck if stuff is okay
# =============================================================================
sigs[12].sumUp()
print('INFO:')
print('There are a total of {} files'.format(len(sigs)))

temp = [s.sumUp()['MATCHING_LENGTHS'] for s in sigs]
if len(set(temp))==1:
    print('All signals contain gyro and fsr data of the same length')
else:
    print('Some files contain data of unequal lengths')

    
sigs[8].plotSignals([0,10])
sigs[8].sumUp()
#%%

# Minimal length of signals?

m = 10000
i = 1
for ix,sig in enumerate(sigs):
    if sig.tappingLength < m:
        m = sig.tappingLength
        i = ix
        
# The shortest signal is sigs[86], corresponding to a CTRL, rhec. Lasts a total of 12s, and 7.25s of active tapping.
# Going to remove those with active signal <10s...maybe
print('Shortest length: {}s, found for signal number {} '.format(m,i))

#%%

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

#%%
    

d = classDistribution(sigs)


CTRLind = [i for i,sig in enumerate(sigs) if sig.diagnosis == 'CTRL']
MSAind = [i for i,sig in enumerate(sigs) if sig.diagnosis == 'MSA']
PDind = [i for i,sig in enumerate(sigs) if sig.diagnosis == 'PD']
PSPind = [i for i,sig in enumerate(sigs) if sig.diagnosis == 'PSP']

inds = [[CTRLind], [MSAind],[PDind],[PSPind]]

#%%
def splitOneDiagnosis(DIAGind, trainPercent, testPercent):
    np.random.seed(12345)
    trainInd = np.random.choice(DIAGind,round(trainPercent*len(DIAGind)),replace = False)
    Xtrain =  [sigs[i] for i in trainInd]
    leftover = [i for i in DIAGind if i not in trainInd]
    
    testInd = np.random.choice(leftover,round(testPercent*len(DIAGind)), replace = False)
    Xtest = [sigs[i] for i in testInd]
    leftover = [i for i in leftover if i not in testInd]
    
    Xval = [sigs[i] for i in leftover]
    
    return Xtrain, Xtest, Xval

# Split into train, test, val sets

trainPercent = 0.7
testPercent = 0.2
#valPercent = 0.1


dataTrain =[]
dataTest = []
dataVal = []

for DIAGind in inds:
    diagTrain, diagTest, diagVal = splitOneDiagnosis(DIAGind[0],trainPercent,testPercent) 
    dataTrain = dataTrain + diagTrain
    dataTest = dataTest + diagTest
    dataVal = dataVal + diagVal
    
print(len(dataTrain), len(dataTest), len(dataVal))
classDistribution(dataTrain)
classDistribution(dataTest)
classDistribution(dataVal)
#%%


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



#%%
def defCallbacks(weightFile):
    
    checkpoint = ModelCheckpoint(weightFile,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only = False,
                                 save_weights_only = True,
                                 mode='max')
    early = EarlyStopping(monitor='val_loss',
                          patience = 20,
                          verbose = 1,
                          mode='min')
#     def step_decay(epoch):
#         initial_lrate = 0.001
#         rate_drop = 0.25
#         nEpochs = 5
#         lrate = initial_lrate * math.pow(rate_drop, math.floor(epoch/nEpochs))
#         return lrate 
    
#     lrate = LearningRateScheduler(step_decay, verbose = 1)
    lr_plateau = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1,
                                  patience = 8, min_lr = 0.0000000001,
                                  verbose = 1)
    
    return [checkpoint, early, lr_plateau]

def examineHistory(history,modelName):
    

    # plot Accuracy over Epochs
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train Acc','Val Acc'])
    plt.title('Accuracy for {} over epochs'.format(modelName))
    plt.show()


    # plot Loss over Epochs
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train Loss','Val Loss'])
    plt.title('Loss for {} over epochs'.format(modelName))
    plt.show()
    
    print("Max val accuracy: ", max(history['val_acc']))
    print("Max train accuracy: ", max(history['acc']))
        
    return max(history['val_acc']),max(history['acc'])
#%%
#==============================================================================
# =============================================================================
#     >>>>>>>>>>>>>>>>>> CLASSIFIER <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# =============================================================================
# =============================================================================
# THE DATA TO USE
# =============================================================================

nSeconds = 8
Xtrain, Ytrain = cropAndReshape(dataTrain,nSeconds)
Xtest, Ytest = cropAndReshape(dataTest,nSeconds)
#XtestMulti, YtestMulti = cropAndReshape(dataTest,nSeconds)
Xval, Yval = cropAndReshape(dataVal,nSeconds)

#%%
# =============================================================================
# THE MODEL
# =============================================================================

def CNNModel(inputShape, nConvLayers, kernel_size, kernel_constraint,nUnits,initialFilters):
    
    input1 = Input(shape = inputShape)

    #convolutions
    x = input1
    
    
    for i in range(0,nConvLayers):
        
        nFilters = 128 if initialFilters*(2**(i))>128 else initialFilters*(2**(i))
        #inside = 3 if i>1 else 2
        inside = 2
        k = 11 if i ==0 else kernel_size
        for temp in range(0,inside):
            x = Conv1D(filters = nFilters,
                  kernel_size = k,
                  padding = 'same',
                  strides = 1,
                  kernel_initializer = 'he_normal',
                  kernel_constraint = max_norm(kernel_constraint),
                  name = 'Conv1x5{}{}'.format(i,temp))(x)
            
            x = Activation('relu',name='ReLu{}{}'.format(i,temp))(x)
            x = BatchNormalization()(x)
            
            x = MaxPooling1D((2),
                      padding = 'same')(x)
    
    x = Dropout(0.7)(x) #0.6!
    
    # Fully connected
    x = Flatten()(x)
    
    x = Dense(nUnits,
              kernel_constraint = max_norm(1),
              kernel_initializer = 'he_normal')(x)
    x = Activation('relu', name = 'reLU_dense')(x)
    x = Dropout(0.7)(x)
    
    x = Dense(4)(x)
    x = Activation('softmax',name = 'Softmax')(x)

    m = Model(input1,x)
    return m



def fitModel(model, modelName, Xtrain, Ytrain, Xval, Yval, epochs,batch_size):
    
    tic = time.time()
    
    # make file name
    tm = time.gmtime()
    weightFile = path+'shuffleBEST_WEIGHTS{}.{}.{}.{}.{}.{}.h5'.format(modelName,tm[2],tm[1],tm[0],tm[3]+1,tm[4])
    
    #define callbacks
    callbacks = defCallbacks(weightFile)
    
    # FIT THE MODEL
    history = model.fit(x = Xtrain, y = Ytrain,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data = (Xval,Yval),
                        callbacks = callbacks,
                        shuffle=True)
    toc = time.time()
    print("Finished training in {} min ({} h)".format(round((toc-tic)/60,2),round((toc-tic)/3600,2)))

    
    # Save the weights
    #model.save_weights(str(modelName)+'.h5') # ???????
    
    return history

#%% LEFT & RIGHT 
# =============================================================================
#  >>>>>>>>>>>>>>>>>>>>>>>> AUTOENCODER <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# =============================================================================
# =============================================================================
# SEPARATE APPROACH - AUTOENCODER INITIATED net
    # but you also need this for right hand only approach!!
# =============================================================================
nSeconds = 8
dataTestRight = [x  for x in dataTest if (x.tap_task=='RHEO' or x.tap_task=='RHEC')]
dataTestLeft = [x  for x in dataTest if (x.tap_task=='LHEO' or x.tap_task=='LHEC')]

dataValRight = [x  for x in dataVal if (x.tap_task=='RHEO' or x.tap_task=='RHEC')]
dataValLeft = [x  for x in dataVal if (x.tap_task=='LHEO' or x.tap_task=='LHEC')]

dataTrainRight = [x  for x in dataTrain if (x.tap_task=='RHEO' or x.tap_task=='RHEC')]
dataTrainLeft = [x  for x in dataTrain if (x.tap_task=='LHEO' or x.tap_task=='LHEC')]

dataTrainAutoencoder = np.concatenate((dataTrain,dataTestLeft),0)
dataTrainAutoencoder.shape


XtrainA, YtrainA = cropAndReshape(dataTrainAutoencoder,nSeconds)
XtestRight, YtestRight = cropAndReshape(dataTestRight,nSeconds)
#XtestMulti, YtestMulti = cropAndReshape(dataTest,nSeconds)
XvalA, YvalA = cropAndReshape(dataVal,nSeconds)

XtrainRight, YtrainRight = cropAndReshape(dataTrainRight,nSeconds)
XvalRight, YvalRight = cropAndReshape(dataValRight,nSeconds)


#%%
# =============================================================================
#     TRAAAAAIIIIIINNNNNN!! (NO AUTOENCODER. RANDOM INITIALIZATION)
# =============================================================================

#modelDepths = []

ConvLayers = [3]   
epochs = 200
batch_sizes = [16]
kernelSizes = [11] #so far for pos in best 11
res = []
kernel_constraints = [2]
nDenseUnits = [32]
nInitialFilters = [32]

for kernel_size in kernelSizes:
    for nConvLayers in ConvLayers:
        for kernel_constraint in kernel_constraints:
            for nUnits in nDenseUnits:
                for initialFilters in nInitialFilters:
                    for batch_size in batch_sizes:
                        vals = []
                        trains = []
                        testa = []
                        for i in range(5):
                        
                            model = CNNModel((XtrainRight.shape[1],XtrainRight.shape[2]),nConvLayers, kernel_size,kernel_constraint,nUnits,initialFilters)
                            model.summary()
                            #opt = optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)
                            optA = optimizers.Adam(lr=0.001)
                            model.compile(optimizer = optA,
                                     loss='categorical_crossentropy',
                                     metrics = ['accuracy'])
                            #model.summary()
                    
                            modelName = 'CNNShuffled'+str(batch_size)+'Batch'+ str(nConvLayers)+'ConvLayers'+str(kernel_size)+'KERNEL'+str(nUnits)+'DenseUnits'+str(initialFilters)+'initFilt'
                            #saveModelTopology(model,modelName)
                            model.save(path+modelName+'CEO.h5')
                    
                            print('TRAINING...')
                    
                            history = fitModel(model,modelName, XtrainRight, YtrainRight, XvalRight, YvalRight, epochs,batch_size)
                            valAcc,trainAcc = examineHistory(history.history,modelName)
                            vals.append(valAcc)
                            trains.append(trainAcc)
                            actual=model.evaluate(XtestRight,YtestRight)
                            testa.append(actual[1])
                        res.append({'history':history,
                                    'kernelSize':kernel_size,
                                    'nConvLayers':nConvLayers,
                                    'batchSize':batch_size,
                                    'constraint':kernel_constraint,
                                    'nDenseUnits':nUnits,
                                    'nInitialFilters':initialFilters,
                                    'bestValAcc':vals,
                                    'bestTrainAcc':trains,
                                    'meanBestVal': np.mean(vals),
                                    'meanBestTrain':np.mean(trains),
                                    'testAccuracy:':testa})
                
        
        
res       


#%%
def CNNModelAuto(inputShape, nConvLayers, kernel_size, kernel_constraint,initFilters,nUnits):
    
    input1 = Input(shape = inputShape)

    #convolutions
    x = input1
    
    
    for i in range(0,nConvLayers):
        
        nFilters = 128 if initFilters*2**(i)>128 else initFilters*2**(i)
        #inside = 3 if i>1 else 2
        inside = 2
        for temp in range(0,inside):
            x = Conv1D(filters = nFilters,
                  kernel_size = kernel_size,
                  padding = 'same',
                  strides = 1,
                  kernel_initializer = 'he_normal',
                  kernel_constraint = max_norm(kernel_constraint),
                  name = 'Conv1x5{}{}'.format(i,temp))(x)
            
            x = Activation('relu',name='ReLu{}{}'.format(i,temp))(x)
            x = BatchNormalization()(x)
            
            x = MaxPooling1D((2),
                      padding = 'same')(x)
    
    x = Dropout(0.6)(x)
    
    # Fully connected
    x = Flatten(name ='flatWhite')(x)
    
#    x = Dense(nUnits,
#              kernel_constraint = max_norm(kernel_constraint),
#              kernel_initializer = 'he_normal')(x)
#    x = Activation('relu', name = 'reLU_dense')(x)
#    x = Dropout(0.5,name='FCstuffEncoded')(x)
    
    
    filtTemp = 128 if nConvLayers>2 else (nConvLayers)*initFilters
    rShape = (int(np.ceil(inputShape[0]/(2**(2*nConvLayers)))),filtTemp)
#    print(rShape)
#    x = Dense(int(np.ceil(inputShape[0]/(2**(2*nConvLayers))))*filtTemp,
#              activation='relu',
#              kernel_initializer='he_normal',
#              name='denseDude')(x)
    x = Reshape(rShape)(x)
    # decoder
    
    for i in range(0,nConvLayers):
        
        nFilters = 128 if i <nConvLayers-2 else initFilters*(nConvLayers-i)
        
        for temp in range(0,2):
            x = Conv1D(filters = nFilters,
                  kernel_size = kernel_size,
                  padding = 'same',
                  strides = 1,
                  kernel_initializer = 'he_normal',
                  kernel_constraint = max_norm(kernel_constraint))(x)
            
            x = Activation('relu')(x)
            x = BatchNormalization()(x)
            
            x = UpSampling1D(2)(x)

    x = Conv1D(kernel_size = 1, strides = 1, filters = 6,
               padding = 'same',
               kernel_initializer = 'he_normal',
               kernel_constraint = max_norm(kernel_constraint))(x)
    x = Activation('sigmoid')(x)
            
    m = Model(input1,x)
    #features = m.get_layer('flatWhite').output
    features = m.get_layer('flatWhite').output
    encoder = Model(input1,features)
 
    return m,encoder


def defCallbacksA(weightFile):
    
    checkpoint = ModelCheckpoint(weightFile,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only = False,
                                 save_weights_only = True,
                                 mode='min')
    early = EarlyStopping(monitor='val_loss',
                          patience = 20,
                          verbose = 1,
                          mode='min')
#     def step_decay(epoch):
#         initial_lrate = 0.001
#         rate_drop = 0.25
#         nEpochs = 5
#         lrate = initial_lrate * math.pow(rate_drop, math.floor(epoch/nEpochs))
#         return lrate 
    
#     lrate = LearningRateScheduler(step_decay, verbose = 1)
    lr_plateau = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1,
                                  patience = 8, min_lr = 0.0000000001,
                                  verbose = 1)
    
    return [checkpoint, early, lr_plateau]
#%% AUTOENCODER INITIATED
inputShape = (XtrainRight.shape[1],XtrainRight.shape[2])
[autoencoder,encoder] = CNNModelAuto(inputShape,3,11,3,32,64)
autoencoder.summary()
autoencoder.compile(optimizer = 'adam',
                    metrics = ['accuracy'],
                    loss = 'binary_crossentropy')






callbacks = defCallbacksA(path+'autoencoderWeightsRIGHTshuffle.h5')
historyA = autoencoder.fit(XtrainRight,XtrainRight,
                          epochs = 100,
                          batch_size = 64,
                          validation_data = (XvalRight,XvalRight), #XvalA,XvalA
                          callbacks = callbacks,
                          shuffle = True)



autoencoder.load_weights(path + 'autoencoderWeightsRIGHTshuffle.h5')
for l1,l2 in zip(encoder.layers,autoencoder.layers[:27]):
    l1.set_weights(l2.get_weights())


encoder.save(path+'encoderCEOshuffleRIGHT.h5')


examineHistory(historyA.history,'a')


#%% predict some vals
reconstructed = autoencoder.predict(XvalRight[22:24,:,:])
t = np.arange(0,len(XvalRight[22])/200,1/200)
plt.figure(figsize = (10,5))
plt.plot(t,XvalRight[22])
plt.xlabel('t[s]')
plt.title('Original normalized 6-channel gyroscope signal')
plt.show()
plt.figure(figsize = (10,5))
plt.plot(t,reconstructed[0])
plt.title('Autoencoder reconstructed signal')
plt.xlabel('t[s]')
plt.show()
#%%
#utilize initialization
def modelFromAuto(nUnits,dropoutRate):
    path='C:/data/tapping/'
    encoder = load_model(path+'encoderCEOshuffleRIGHT.h5')
    x = encoder.output
    x = Dense(nUnits,kernel_constraint = max_norm(1),
              kernel_initializer = 'he_normal',
              activation = 'relu')(x)
    x = Dropout(dropoutRate)(x)
    x = Dense(4, activation = 'softmax')(x)
    
    m = Model(encoder.input,x)
    return m

#%%

nUnits = [64]
dropoutRate = [0.7]
res = []
 
for rate in  dropoutRate:
    for units in nUnits:
        vacc = []
        tacc = []
        a = []
        for i in range(5):
            
            mA = modelFromAuto(units,rate)
#            for l in mA.layers[:27]:
#                l.trainable = False
            op = optimizers.Adam(lr = 0.001)
            #op1 = optimizers.SGD(lr = 0.0001,momentum = 0.8, nesterov= True )
            mA.compile(optimizer = op,
                       metrics = ['accuracy'],
                       loss = 'categorical_crossentropy')
            tm = time.gmtime()
            callbacks = defCallbacks(path+'autoencoderBasedRightShuffled{}-{}-{}.{}.{}.{}.h5'.format(tm[0],tm[1],tm[2],tm[3],tm[4],tm[5]))
            h = mA.fit(XtrainRight,YtrainRight,
                       validation_data = (XvalRight,YvalRight),
                       epochs = 150,
                       batch_size = 16,
                       callbacks = callbacks)
            
            vA,tA = examineHistory(h.history,'bla')
            actual=mA.evaluate(XtestRight,YtestRight)
            vacc.append(vA)
            tacc.append(tA)
            a.append(actual)
        res.append({'rate':rate,
                    'nUnits':units,
                    'vallAcc':vacc,
                    'trainAcc':tacc,
                    'meanval':np.mean(vacc),
                    'testAcc':a[1]})
       
res
#%%

# =============================================================================
# TEST TEST TEST TEST RIGHT MULTICTOP RIGHT TEST TEST TEST
# =============================================================================
from sklearn.metrics import confusion_matrix
import itertools


XtestRight.shape
YtestRight.shape

path = 'C:/data/tapping/'


def predictSignal(signalX, signalY, model, verbose = 1):
    signalX = np.expand_dims(signalX,axis=0)
    prediction = model.predict(signalX)
    prediction = prediction[0]
    prediction = np.round(np.float32(prediction),2)
    
    d = {0: 'HC',
        1: 'MSA',
        2: 'PD',
        3: 'PSP'}
    
    

    actualDiagnosis = d[np.argmax(signalY)]
    predictedDiagnosis = d[np.argmax(prediction)]
    
    if verbose:
        print('predictedDiagnosis: ',predictedDiagnosis)
        print('actualDiagnosis: ', actualDiagnosis)
        print('Certainty: \nHC: {} \nMSA: {} \nPD: {} \nPSP: {}'.format(prediction[0],
                                                                      prediction[1],
                                                                      prediction[2],
                                                                      prediction[3]))
        print('#################################################')
   
    
    return {"Predicted": predictedDiagnosis, "Actual": actualDiagnosis}


def confuseMat(Xtest, Ytest,model):
    p, a = [], []
    for X,Y in zip(Xtest,Ytest):
        temp = predictSignal(X,Y,model,verbose=0)
        p.append(temp['Predicted'])
        a.append(temp['Actual'])
    confMat = confusion_matrix(a,p)
    print(confMat)
    sumaPoRedovima = confMat.astype('float').sum(axis=1)
    confMatPerc = [gore/dole for gore,dole in zip(confMat,sumaPoRedovima)]
    confMatPerc = np.matrix(confMatPerc)*100
    
    return [confMat,confMatPerc]


def plotConfMat(cm,cmPerc):
    plt.figure(figsize = (10,10))
    plt.imshow(cmPerc)
    plt.colorbar()
    for i,j in itertools.product(range(4),range(4)):
        plt.text(i,j,'{}%\n({})'.format(round(cmPerc[j,i],2),cm[j,i],'d'),
                horizontalalignment = "center",
                color = "white" if cmPerc[j,i]<60 else "black",
                size = 15)
    tick_marks = np.arange(4)
    classes = ['HC','MSA','PD','PSP']
    plt.xticks(tick_marks,classes,rotation = 45,size = 15)
    plt.yticks(tick_marks,classes,size = 15)
    plt.ylabel('True label',size = 15)
    plt.xlabel('\nPredicted label',size = 15)
    #plt.style.use(['tableau-colorblind10'])
    #plt.rcParams['image.cmap'] = 'viridis'
    plt.title('Confusion matrix\n', size = 17)
    plt.show()
    return
#%%
cm, cmPerc = confuseMat(XtestRight,YtestRight,model)
plotConfMat(cm,cmPerc)

#%%
#svaki pojedinacno
import random
accEach = []
predEach = []
predAll=[]
dijagnoze=[]
najmanjeCropova = 100
srednjePred=[]
predProb = []
for datic in dataTestRight:
    tempX,tempY = cropAndReshape([datic],nSeconds)
    print(tempX.shape)
    tempCropova = tempX.shape[0]
    if tempCropova<najmanjeCropova:
        najmanjeCropova = tempCropova
        ##
    random.seed(123)
    idx = range(4)
    _,tempAcc = mA.evaluate(tempX[idx],tempY[idx])
    # a vidi sad koje tacno brojeve vraca, ne samo max index
    predikcije = np.mean(mA.predict(tempX[idx]),axis=0)
    paMean = np.argmax(predikcije)
    srednjePred.append(paMean)
    predProb.append(predikcije)
    tempPred = [np.argmax(p) for p in mA.predict(tempX[idx])]
    pa=[np.argmax(p) for p in mA.predict(tempX[idx])]
    predAll.append(pa)
    accEach.append(tempAcc) # na onoliko koliko si izabrala odbiraka
    predEach.append(tempPred)
    dijagnoze.append(datic.diagnosis)

    ##

print('najmanji broj kropova u fajlu je: ',najmanjeCropova)
nule = 0
jediniceBas = 0
for a in accEach:
    if a<=0.5:
        nule+=1
    if a>0.99999:
        jediniceBas+=1
        
        
print('zapravo accuracy: ',100*(1-nule/len(dataTestRight)))

#%%
def kodiraj(numPrediction):
    switch = {0:'CTRL',
              1:'MSA',
              2:'PD',
              3:'PSP'}
    
    return switch[numPrediction]

kodiraniSmoothies = [kodiraj(pred) for pred in srednjePred]
cmm = confusion_matrix(dijagnoze,kodiraniSmoothies)
cmmPerc = [c/s for c,s in zip(cmm,np.sum(cmm,axis=1))]
cmmPerc=np.matrix(cmmPerc)*100
plotConfMat(cmm,cmmPerc)

#%% 
from keras.utils import plot_model

plot_model(mA,path+'juhuModelSlika.png',show_shapes=True)


#%%
#precision and recall

def metrics(confmat):
    allOfClass = np.sum(confmat,axis=1)
    allAsClass = np.sum(confmat,axis=0)
    correctly = np.diag(confmat)
    precision = np.round(np.divide(correctly, allAsClass)*100,2)
    recall = np.round(np.divide(correctly, allOfClass)*100,2)
    allall = np.sum(confmat)
    allcorrect = np.sum(correctly)
    acc = np.round(allcorrect*100/allall,2)
    return acc, precision, recall

#%%
    
def hasWeights(layername):
    ok = True
    if "activation" in layername or\
    "dropout" in layername or\
    "FLAT" in layername  or\
    "flatten" in layername or\
    "input" in layername or\
    "reshape" in layername:
        ok = False
    return ok

    #"pooling" in layername or\
    #"ReLu" in layername or\
    #"reLU" in layername or\
    #"Softmax" in layername or\

from keras import backend as K    
inp = model.input
outs = [layer.output for layer in model.layers if hasWeights(layer.name)]
functor = K.function([inp]+ [K.learning_phase()],outs)
izlazi = functor([XtrainRight,0])


#%%

temp = izlazi[3] #maxpool 1
#plt.figure(figsize = (12,8))
#plt.plot(temp[12,:,0]) #osoba red br 12, kanal 0ti
#plt.show()
#plt.plot(temp[12,:,1]) #osoba red br 12, kanal 1vi
#plt.show()
#plt.plot(temp[12,:,10]) #osoba red br 12, kanal 10i
#plt.show()

plt.plot(temp[98,:,:]) #osoba red br 12, svih 32 kanala  #MSA
plt.title("MSA, CONV1")
plt.show()

plt.plot(temp[555,:,:]) #osoba red br 122, svih 32 kanala #CTRL
plt.title("CTRL, CONV1")
plt.show()


plt.plot(temp[66,:,:]) #osoba red br 122, svih 32 kanala #PD
plt.title("PD, CONV1")
plt.show()

plt.plot(temp[1100,:,:]) #osoba red br 122, svih 32 kanala #PSP
plt.title("PSP CONV1")
plt.title("PSP, CONV1")
plt.show()
###

temp = izlazi[7] #maxpool posle 2 conv
plt.plot(temp[98,:,:]) #osoba red br 12, svih 32 kanala  #MSA
plt.title("MSA, CONV2")
plt.show()

plt.plot(temp[555,:,:]) #osoba red br 122, svih 32 kanala #CTRL
plt.title("CTRL CONV2")
plt.show()

plt.plot(temp[66,:,:]) #osoba red br 122, svih 32 kanala #PD
plt.title("PD, CONV2")
plt.show()

plt.plot(temp[1100,:,:]) #osoba red br 122, svih 32 kanala #PSP
plt.title("PSP CONV2")
plt.show()
###
temp = izlazi[11] #max posle conv3
plt.plot(temp[98,:,:]) #osoba red br 12, svih 32 kanala  #MSA
plt.title("MSA, CONV3")
plt.show()

plt.plot(temp[555,:,:]) #osoba red br 122, svih 32 kanala #CTRL
plt.title("CTRL CONV3")
plt.show()

plt.plot(temp[66,:,:]) #osoba red br 122, svih 32 kanala #PD
plt.show()

plt.plot(temp[1100,:,:]) #osoba red br 122, svih 32 kanala #PSP
plt.title("PSP CONV3")
plt.show()

##
temp = izlazi[15] #max posle conv 4
plt.plot(temp[98,:,:]) #osoba red br 12, svih 32 kanala  #MSA
plt.title("MSA, CONV4")
plt.show()

plt.plot(temp[555,:,:]) #osoba red br 122, svih 32 kanala #CTRL
plt.title("CTRL CONV4")
plt.show()

plt.plot(temp[66,:,:]) #osoba red br 122, svih 32 kanala #PD
plt.show()

plt.plot(temp[1100,:,:]) #osoba red br 122, svih 32 kanala #PSP
plt.title("PSP CONV4")
plt.show()

##
temp = izlazi[19]
plt.plot(temp[98,:,:]) #osoba red br 12, svih 32 kanala  #MSA
plt.title("MSA, CONV5")
plt.show()

plt.plot(temp[555,:,:]) #osoba red br 122, svih 32 kanala #CTRL
plt.title("CTRL CONV5")
plt.show()

plt.plot(temp[66,:,:]) #osoba red br 122, svih 32 kanala #PD
plt.show()

plt.plot(temp[1100,:,:]) #osoba red br 122, svih 32 kanala #PSP
plt.title("PSP CONV5")
plt.show()

##
temp = izlazi[23]
plt.plot(temp[98,:,:]) #osoba red br 12, svih 32 kanala  #MSA
plt.title("MSA, CONV6")
plt.show()

plt.plot(temp[555,:,:]) #osoba red br 122, svih 32 kanala #CTRL
plt.title("CTRL CONV6")
plt.show()

plt.plot(temp[66,:,:]) #osoba red br 122, svih 32 kanala #PD
plt.title("PD, CONV6")
plt.show()

plt.plot(temp[1100,:,:]) #osoba red br 122, svih 32 kanala #PSP
plt.title("PSP CONV6")
plt.show()

## 
#%%

fig = plt.figure(figsize = (20,20))

temp = izlazi[3]

plt.subplot(641)
plt.plot(temp[98,0:200,8]) #osoba red br 12, svih 32 kanala  #MSA i 98 je neki msa izgleda nije strasan
plt.plot(temp[98,0:200,12]) #osoba red br 12, svih 32 kanala  #MSA i 98 je neki msa izgleda nije strasan
plt.plot(temp[98,0:200,28]) 
plt.title("MSA")
plt.ylabel("CONV1")

plt.subplot(642)
plt.plot(temp[555,0:200,8]) #osoba red br 122, svih 32 kanala #CTRL i 555 je ctrl mozda lepsa i 1130 je ok
plt.plot(temp[555,0:200,12]) #osoba red br 122, svih 32 kanala #CTRL i 555 je ctrl mozda lepsa i 1130 je ok
plt.plot(temp[555,0:200,28]) 
plt.title("CTRL")

plt.subplot(643)
plt.plot(temp[66,0:200,8]) #osoba red br 122, svih 32 kanala #PD i 66 je pd i 77 ali cudan bas
plt.plot(temp[66,0:200,12]) #osoba red br 122, svih 32 kanala #PD i 66 je pd i 77 ali cudan bas
plt.plot(temp[66,0:200,28]) 
plt.title("PD")

plt.subplot(644)
plt.plot(temp[1100,0:200,8]) #osoba red br 122, svih 32 kanala #PSP #imas i 211  a i 1100 nije strasno i 1120 mozda ok 
plt.plot(temp[1100,0:200,12]) #
plt.plot(temp[1100,0:200,28]) 
plt.title("PSP")


#kanal 8 je zanimljiv kod conv1
# i 12 nesto fali kod msa
#mozda i 13
#i mozes 28

####
temp = izlazi[7]
plt.subplot(645)
plt.plot(temp[98,0:200,8]) #osoba red br 12, svih 32 kanala  #MSA i 98 je neki msa izgleda nije strasan
plt.plot(temp[98,0:200,12]) #osoba red br 12, svih 32 kanala  #MSA i 98 je neki msa izgleda nije strasan
plt.plot(temp[98,0:200,28]) 
plt.ylabel("CONV2")


plt.subplot(646)
plt.plot(temp[555,0:200,8]) #osoba red br 122, svih 32 kanala #CTRL i 555 je ctrl mozda lepsa i 1130 je ok
plt.plot(temp[555,0:200,12]) #osoba red br 122, svih 32 kanala #CTRL i 555 je ctrl mozda lepsa i 1130 je ok
plt.plot(temp[555,0:200,28]) 


plt.subplot(647)
plt.plot(temp[66,0:200,8]) #osoba red br 122, svih 32 kanala #PD i 66 je pd i 77 ali cudan bas
plt.plot(temp[66,0:200,12]) #osoba red br 122, svih 32 kanala #PD i 66 je pd i 77 ali cudan bas
plt.plot(temp[66,0:200,28]) 


plt.subplot(648)
plt.plot(temp[1100,0:200,8]) #osoba red br 122, svih 32 kanala #PSP #imas i 211  a i 1100 nije strasno i 1120 mozda ok 
plt.plot(temp[1100,0:200,12]) #
plt.plot(temp[1100,0:200,28]) 
#####
temp = izlazi[11]
plt.subplot(649)
plt.plot(temp[98,0:200,8]) #osoba red br 12, svih 32 kanala  #MSA i 98 je neki msa izgleda nije strasan
plt.plot(temp[98,0:200,12]) #osoba red br 12, svih 32 kanala  #MSA i 98 je neki msa izgleda nije strasan
plt.plot(temp[98,0:200,28]) 
plt.ylabel("CONV3")


plt.subplot(6,4,10)
plt.plot(temp[555,0:200,8]) #osoba red br 122, svih 32 kanala #CTRL i 555 je ctrl mozda lepsa i 1130 je ok
plt.plot(temp[555,0:200,12]) #osoba red br 122, svih 32 kanala #CTRL i 555 je ctrl mozda lepsa i 1130 je ok
plt.plot(temp[555,0:200,28]) 


plt.subplot(6,4,11)
plt.plot(temp[66,0:200,8]) #osoba red br 122, svih 32 kanala #PD i 66 je pd i 77 ali cudan bas
plt.plot(temp[66,0:200,12]) #osoba red br 122, svih 32 kanala #PD i 66 je pd i 77 ali cudan bas
plt.plot(temp[66,0:200,28]) 


plt.subplot(6,4,12)
plt.plot(temp[1100,0:200,8]) #osoba red br 122, svih 32 kanala #PSP #imas i 211  a i 1100 nije strasno i 1120 mozda ok 
plt.plot(temp[1100,0:200,12]) #
plt.plot(temp[1100,0:200,28]) 

##############
temp = izlazi[15]
plt.subplot(6,4,13)
plt.plot(temp[98,0:200,8]) #osoba red br 12, svih 32 kanala  #MSA i 98 je neki msa izgleda nije strasan
plt.plot(temp[98,0:200,12]) #osoba red br 12, svih 32 kanala  #MSA i 98 je neki msa izgleda nije strasan
plt.plot(temp[98,0:200,28]) 
plt.ylabel("CONV4")


plt.subplot(6,4,14)
plt.plot(temp[555,0:200,8]) #osoba red br 122, svih 32 kanala #CTRL i 555 je ctrl mozda lepsa i 1130 je ok
plt.plot(temp[555,0:200,12]) #osoba red br 122, svih 32 kanala #CTRL i 555 je ctrl mozda lepsa i 1130 je ok
plt.plot(temp[555,0:200,28]) 


plt.subplot(6,4,15)
plt.plot(temp[66,0:200,8]) #osoba red br 122, svih 32 kanala #PD i 66 je pd i 77 ali cudan bas
plt.plot(temp[66,0:200,12]) #osoba red br 122, svih 32 kanala #PD i 66 je pd i 77 ali cudan bas
plt.plot(temp[66,0:200,28]) 


plt.subplot(6,4,16)
plt.plot(temp[1100,0:200,8]) #osoba red br 122, svih 32 kanala #PSP #imas i 211  a i 1100 nije strasno i 1120 mozda ok 
plt.plot(temp[1100,0:200,12]) #
plt.plot(temp[1100,0:200,28]) 

########3
temp = izlazi[19]
plt.subplot(6,4,17)
plt.plot(temp[98,0:200,8]) #osoba red br 12, svih 32 kanala  #MSA i 98 je neki msa izgleda nije strasan
plt.plot(temp[98,0:200,12]) #osoba red br 12, svih 32 kanala  #MSA i 98 je neki msa izgleda nije strasan
plt.plot(temp[98,0:200,28]) 
plt.ylabel("CONV5")


plt.subplot(6,4,18)
plt.plot(temp[555,0:200,8]) #osoba red br 122, svih 32 kanala #CTRL i 555 je ctrl mozda lepsa i 1130 je ok
plt.plot(temp[555,0:200,12]) #osoba red br 122, svih 32 kanala #CTRL i 555 je ctrl mozda lepsa i 1130 je ok
plt.plot(temp[555,0:200,28]) 


plt.subplot(6,4,19)
plt.plot(temp[66,0:200,8]) #osoba red br 122, svih 32 kanala #PD i 66 je pd i 77 ali cudan bas
plt.plot(temp[66,0:200,12]) #osoba red br 122, svih 32 kanala #PD i 66 je pd i 77 ali cudan bas
plt.plot(temp[66,0:200,28]) 


plt.subplot(6,4,20)
plt.plot(temp[1100,0:200,8]) #osoba red br 122, svih 32 kanala #PSP #imas i 211  a i 1100 nije strasno i 1120 mozda ok 
plt.plot(temp[1100,0:200,12]) #
plt.plot(temp[1100,0:200,28]) 
#########3
temp = izlazi[23]
plt.subplot(6,4,21)
plt.plot(temp[98,0:200,8]) #osoba red br 12, svih 32 kanala  #MSA i 98 je neki msa izgleda nije strasan
plt.plot(temp[98,0:200,12]) #osoba red br 12, svih 32 kanala  #MSA i 98 je neki msa izgleda nije strasan
plt.plot(temp[98,0:200,28]) 
plt.ylabel("CONV6")


plt.subplot(6,4,22)
plt.plot(temp[555,0:200,8]) #osoba red br 122, svih 32 kanala #CTRL i 555 je ctrl mozda lepsa i 1130 je ok
plt.plot(temp[555,0:200,12]) #osoba red br 122, svih 32 kanala #CTRL i 555 je ctrl mozda lepsa i 1130 je ok
plt.plot(temp[555,0:200,28]) 


plt.subplot(6,4,23)
plt.plot(temp[66,0:200,8]) #osoba red br 122, svih 32 kanala #PD i 66 je pd i 77 ali cudan bas
plt.plot(temp[66,0:200,12]) #osoba red br 122, svih 32 kanala #PD i 66 je pd i 77 ali cudan bas
plt.plot(temp[66,0:200,28]) 


plt.subplot(6,4,24)
plt.plot(temp[1100,0:200,8]) #osoba red br 122, svih 32 kanala #PSP #imas i 211  a i 1100 nije strasno i 1120 mozda ok 
plt.plot(temp[1100,0:200,12]) #
plt.plot(temp[1100,0:200,28]) 

fig.tight_layout()
plt.show()