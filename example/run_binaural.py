"""
Test coherence-based dual-mic speech enhancement

"""

import numpy as np

import time

from beamformer.MicArray import MicArray
from beamformer.utils import mesh,pmesh,load_wav,load_pcm
from coherence.BinauralEnhancement import BinauralEnhancement

import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
from scipy.io import wavfile

filepath = "test_audio/rec/"
x,sr = load_wav(filepath)
# x = x[::2,:]
x = x[[3,1],:]
sr = 16000
c = 343
M = x.shape[0]

frameLen = 256
hop = frameLen/2
overlap = frameLen - hop
nfft = 256
c = 340
r = 0.032*2
fs = sr

start = time.clock()

MicArray = MicArray(arrayType='linear', r=r, c=c, M=M)
angle = np.array([0, 0]) / 180 * np.pi

Binaural = BinauralEnhancement(MicArray,frameLen,hop,nfft,c,r,fs)

yout = Binaural.process(x, angle, method=3)

end = time.clock()
print(end-start)

# listen processed result
sd.default.channels = 1
# sd.play(np.mean(x,axis=0),fs)
# sd.wait()
sd.play(yout['data'],fs)
sd.wait()

# view beampattern
# mesh(yout['beampattern'])
# pmesh(yout['beampattern'])

# save audio
# wavfile.write('output/output_fixedbeamformer.wav',16000,yout['data'])


