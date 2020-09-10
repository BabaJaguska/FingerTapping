# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 22:59:23 2020

@author: minja
"""


from scipy.signal import decimate
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
# %matplotlib qt <-- for drawing in a separate window; to cancel use %matplotlib inline


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
        self.id = '_'.join([self.diagnosis, self.initials, self.date, self.timeOfMeasurement])
        
        
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
        
    def mostProminentAxis(self):
        # in the Index finger gyro
        maxAxis = []
        maxAxisVal = 0
        maxAxisName = ''
        
        names = ['Index_X', 'Index_Y', 'Index_Z']
        
        for i,signal in enumerate([self.gyro2xT, self.gyro2yT, self.gyro2zT]):
            
        
            temp = np.mean(np.square(signal))
            
       
            if temp > maxAxisVal:
                maxAxisVal = temp
                maxAxis = signal
                maxAxisName = names[i]
                
        return maxAxis, maxAxisVal, maxAxisName
    
    def findTapSplits(self, method = 'PanTompkins'):
        
        
        if method == 'PanTompkins':
            
            ref,_,_ = self.mostProminentAxis()
            
            ref = ref - np.mean(ref)
            
            # bandpass filter
            lowcut = 0.4
            highcut = 5
            filter_order = 2
            
            
            nyquist_freq = 0.5 * self.fs
            low = lowcut / nyquist_freq
            high = highcut / nyquist_freq
            b, a = butter(filter_order, [low, high], btype="band")
            ref = filtfilt(b, a, ref, method = 'gust')
            
            # accentuate peaks
            ref = np.power(ref, 2) * np.sign(ref)
            
                   
            # find peaks (minima)
            peak_indices, peak_params = find_peaks(-ref, 
                                         prominence = 1) # flip signal
            
            q75peak = np.quantile(peak_params['prominences'], 0.75)
            
            
            # remove too small peaks
            remove_idx = [i for i, idx in enumerate(peak_indices) if peak_params['prominences'][i] < q75peak/5 ]
            peak_indices = np.delete(peak_indices, remove_idx)

            
        return ref, peak_indices
    
    def splitTaps(self, peak_indices):
        
        taps = []
        for rawSig in [self.gyro1xT, self.gyro1yT, self.gyro1zT, self.gyro2xT, self.gyro2yT, self.gyro2zT]:
            taps.append(np.split(rawSig, peak_indices))
            
        taps = np.array(taps)
        
        order = ['Gyro Thumb X', 'Gyro Thumb Y', 'Gyro Thumb Z', 'Gyro Index X', 'Gyro Index Y', 'Gyro Index Z']
        
        return taps, order
    
    def isRightHand(self):
        if self.tap_task in ['RHEC', 'RHEO']:
            return 1
        return 0
        
            

        
            

