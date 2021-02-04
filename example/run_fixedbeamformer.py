"""
Test beamformer

"""

import numpy as np
import os
import sys

import time

from DistantSpeech.beamformer.MicArray import MicArray
from DistantSpeech.beamformer.fixedbeamformer import fixedbeamformer
from DistantSpeech.beamformer.adaptivebeamformer import adaptivebeamfomer
from DistantSpeech.beamformer.utils import mesh,pmesh,load_wav,load_pcm, visual

import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf




filepath = "./test_audio/rec1/"
print(os.chdir(sys.path[0]))
x,sr = load_wav(os.path.abspath(filepath))
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

fixedbeamformer = fixedbeamformer(MicArray,frameLen,hop,nfft,c,r,fs)
# """
# fixed beamformer precesing function
# method:
# 'DS':   delay-and-sum beamformer
# 'MVDR': MVDR beamformer under isotropic noise field
#
# """
yout = fixedbeamformer.process(x,angle,method=2,retH=True,retWNG=True,retDI=True)

end = time.clock()
print(end-start)

# # listen processed result
# sd.default.channels = 1
# sd.play(yout['data'],fs)
# sd.wait()

# view beampattern
mesh(yout['beampattern'])
# pmesh(yout['beampattern'])

# save audio
# wavfile.write('output/output_fixedbeamformer.wav',16000,yout['data'])

visual(x[0,:],yout['data'])

