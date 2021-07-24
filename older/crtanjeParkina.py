# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 14:39:40 2019

@author: minja
"""

import numpy as np
from matplotlib import pyplot as plt

#from TappingRedoneTransforms import XtrainRight
from TappingRedone import XtrainRight

t = np.linspace(0,3,600)

msa1t = XtrainRight[98,:600,:3]
msa1i= XtrainRight[98,:600,3:]



pd1t = XtrainRight[66,:600,:3]
pd1i = XtrainRight[66,:600,3:]


#pd2 = XtrainRight[88]
ctrl1t = XtrainRight[555,:600,:3]
ctrl1i = XtrainRight[555,:600,3:]
#ctrl2 = XtrainRight[1130]

psp1t = XtrainRight[1100,:600,:3]
psp1i = XtrainRight[1100,:600,3:]
#psp2 = XtrainRight[1120]

##### PLOT ###


SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 18

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE, titleweight ='bold')     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.rcParams['font.family'] = 'serif'
plt. rcParams["legend.loc"] = 'upper left'

fig = plt.figure(figsize = (20,18))

## CTRL
plt.subplot(421)
plt.plot(t,ctrl1t[:,0])
plt.plot(t,ctrl1t[:,1],'--')
plt.plot(t,ctrl1t[:,2])
plt.ylabel(r"$\omega_{THUMB}$ [rad/s]")

plt.title("HC participant - angular velocity of the thumb")
plt.xlabel("t [s]")
plt.legend(['X','Y','Z'])

plt.subplot(422)
plt.plot(t,ctrl1i[:,0])
plt.plot(t,ctrl1i[:,1],'--')
plt.plot(t,ctrl1i[:,2])
plt.title("HC participant - angular velocity of the index finger")
plt.xlabel("t [s]")
plt.ylabel(r"$\omega_{INDEX}$ [rad/s]")

#MSA
plt.subplot(423)
plt.plot(t,msa1t[:,0])
plt.plot(t,msa1t[:,1],'--')
plt.plot(t,msa1t[:,2])
plt.title("MSA patient - angular velocity of the thumb")
plt.xlabel("t [s]")
plt.ylabel(r"$\omega_{THUMB}$ [rad/s]")

plt.subplot(424)
plt.plot(t,msa1i[:,0])
plt.plot(t,msa1i[:,1],'--')
plt.plot(t,msa1i[:,2])
plt.title("MSA patient - angular velocity of the index finger")
plt.xlabel("t [s]")
plt.ylabel(r"$\omega_{INDEX}$ [rad/s]")

#PD
plt.subplot(425)
plt.plot(t,pd1t[:,0])
plt.plot(t,pd1t[:,1],'--')
plt.plot(t,pd1t[:,2])
plt.title("PD patient - angular velocity of the thumb")
plt.xlabel("t [s]")
plt.ylabel(r"$\omega_{THUMB}$ [rad/s]")

plt.subplot(426)
plt.plot(t,pd1i[:,0])
plt.plot(t,pd1i[:,1],'--')
plt.plot(t,pd1i[:,2])
plt.title("PD patient - angular velocity of the index finger")
plt.xlabel("t [s]")
plt.ylabel(r"$\omega_{INDEX}$ [rad/s]")

#PSP
plt.subplot(427)
plt.plot(t,psp1t[:,0])
plt.plot(t,psp1t[:,1],'--')
plt.plot(t,psp1t[:,2])
plt.title("PSP patient - angular velocity of the thumb")
plt.xlabel("t [s]")
plt.ylabel(r"$\omega_{THUMB}$ [rad/s]")

plt.subplot(428)
plt.plot(t,psp1i[:,0])
plt.plot(t,psp1i[:,1],'--')
plt.plot(t,psp1i[:,2])
plt.title("PSP patient - angular velocity of the index finger")
plt.xlabel("t [s]")
plt.ylabel(r"$\omega_{INDEX}$ [rad/s]")

fig.tight_layout()

plt.show()

