"""
Test beamformer

"""

import numpy as np

import time

from beamformer.MicArray import MicArray
from beamformer.fixedbeamformer import fixedbeamfomer
from beamformer.adaptivebeamformer import adaptivebeamfomer
from beamformer.utils import mesh,pmesh,load_wav,load_pcm

import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
from scipy.io import wavfile

filepath = "E:/work/matlab/Github/beamformer/sound/rec1/"
# filepath = "E:/work/kws/mycroft-latest/DistantSpeech/unit_test/pcm/"
x,sr = load_wav(filepath)
sr = 16000
r = 0.032
c = 343

frameLen = 256
hop = frameLen/2
overlap = frameLen - hop
nfft = 256
c = 340
r = 0.032
fs = sr

start = time.clock()

MicArray = MicArray(arrayType='circular', r=0.032, M=4)
angle = np.array([197, 0]) / 180 * np.pi

# fixedbeamformer = fixedbeamfomer(MicArray,frameLen,hop,nfft,c,r,fs)
# yout = fixedbeamformer.process(x,angle,method='MVDR')

adaptivebeamfomer = adaptivebeamfomer(MicArray,frameLen,hop,nfft,c,r,fs)
yout = adaptivebeamfomer.process(x,angle,retH=True,retWNG=True,retDI=True)

end = time.clock()
# sd.default.channels = 1
# sd.play(yout,fs)
# sd.play(yout['out'],fs)
# sd.wait()
# sd.play(yout,fs)
mesh(yout['beampattern'])
# pmesh(yout['beampattern'])
# wavfile.write('ds_fft_oop.wav',16000,yout)
wavfile.write('out/mvdrpy_oop_04.wav',16000,yout['data']/1.08)
print(end-start)
