"""
Test beamformer

"""

import numpy as np

import time

from beamformer.MicArray import MicArray
from beamformer.adaptivebeamformer import adaptivebeamfomer
from beamformer.utils import mesh,pmesh,load_wav,load_pcm

import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
from scipy.io import wavfile

filepath = "test_audio/rec1/"
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

adaptivebeamfomer = adaptivebeamfomer(MicArray,frameLen,hop,nfft,c,r,fs)
yout = adaptivebeamfomer.process(x,angle,retH=True,retWNG=True,retDI=True)

end = time.clock()
print(end-start)

# listen processed result
sd.default.channels = 1
sd.play(yout['data'],fs)
sd.wait()

# view beampattern
mesh(yout['beampattern'])
# pmesh(yout['beampattern'])

# save audio
# wavfile.write('output/output_fixedbeamformer.wav',16000,yout['data'])


