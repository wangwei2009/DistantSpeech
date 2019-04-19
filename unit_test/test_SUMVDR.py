"""
Test superdirective MVDR beamforming algorithms that can update weights online (as opposed to batch processing)

.. reference::

.. moduleauthor:: John McDonough, Kenichi Kumatani <k_kumatani@ieee.org>
"""
import argparse, json
import os.path
import pickle
import wave
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import os
import time
import librosa

from beamformer.MicArray import MicArray
from beamformer.fixedbeamformer import fixedbeamfomer
from beamformer.adaptivebeamformer import adaptivebeamfomer


filepath = "E:/work/matlab/Github/beamformer/sound/rec1/"
filename= os.listdir(filepath)
r = 0.032
c = 343
x = np.zeros((4,427680))

h = signal.firwin(513,0.01,pass_zero=False)
for i in range(0,4):
    x1,sr = librosa.load(filepath+filename[i],sr=None)
    x[i, :] = x1#signal.filtfilt(h,1,x1)


frameLen = 256
hop = frameLen/2
overlap = frameLen - hop
nfft = 256
c = 343
r = 0.032
fs = sr

start = time.clock()

MicArray = MicArray(arrayType='circular', r=0.032, M=4)
angle = np.array([197, 0]) / 180 * np.pi

fixedbeamformer = fixedbeamfomer(MicArray,frameLen,hop,nfft,c,r,fs)
yout,WNG,DI = fixedbeamformer.superDirectiveMVDR(x,angle,WNG=True,DI=True)
yout,WNG,DI = fixedbeamformer.delaysum(x,angle,WNG=True,DI=True)

adaptivebeamfomer = adaptivebeamfomer(MicArray,frameLen,hop,nfft,c,r,fs)
yout,WNG,DI = adaptivebeamfomer.AdaptiveMVDR2(x,angle,WNG=True,DI=True)

yout = np.squeeze(yout)
end = time.clock()
plt.plot(DI)
plt.show()
# wavfile.write('ds_fft_oop.wav',16000,yout)
wavfile.write('mvdrpy_stft_oop.wav',16000,yout)
print(end-start)
