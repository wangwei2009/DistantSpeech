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
from beamformer.MVDR import MVDR
from beamformer.MicArray import MicArray


from scipy.io import wavfile
from scipy.signal import windows
from scipy import signal
from  scipy.signal import stft
import os
import time
from matplotlib import pyplot as plt
import librosa


filepath = "E:/work/matlab/Github/beamformer/sound/rec1/" #添加路径
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

MicArray = MicArray(arrayType = 'circular',r = 0.032,M = 4)
angle = np.array([197, 0]) / 180 * np.pi

MVDRbeamformer = MVDR(MicArray,frameLen,hop,nfft,c,r,fs)
# yout = MVDRbeamformer.superDirectiveMVDR(x)
# yout = MVDRbeamformer.superDirectiveMVDR2(x,angle)
yout = MVDRbeamformer.AdaptiveMVDR(x,angle)
yout = np.squeeze(yout)
end = time.clock()
plt.plot(yout)
plt.show()
# wavfile.write('sumvdrpy_fft4_HP.wav',16000,yout)
wavfile.write('mvdrpy_stft.wav',16000,yout)
print(end-start)
