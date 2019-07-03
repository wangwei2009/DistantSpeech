"""
Test McCowan postfilter

"""

import numpy as np

import time

from beamformer.MicArray import MicArray
from beamformer.fixedbeamformer import fixedbeamfomer
from beamformer.adaptivebeamformer import adaptivebeamfomer
from beamformer.utils import mesh,pmesh,load_wav,load_pcm
from postfilter.postfilter import PostFilter

import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
from scipy.io import wavfile

filepath = "test_audio/an101-mtms-arrA/"
x,sr = load_wav(filepath)
sr = 16000
r = 0.032
c = 343
M = x.shape[0]

frameLen = 256
hop = frameLen/2
overlap = frameLen - hop
nfft = 256
c = 340
r = 0.032
fs = sr

start = time.clock()

MicArray = MicArray(arrayType='linear', r=0.032, M=M)
angle = np.array([197, 0]) / 180 * np.pi

postfilter1 = PostFilter(MicArray,frameLen,hop,nfft,c,r,fs)

yout = postfilter1.process(x,np.mean(x,axis=0), angle,method='MVDR')

end = time.clock()
print(end-start)

# listen processed result
sd.default.channels = 1
sd.play(np.mean(x,axis=0),fs)
sd.wait()
sd.play(yout['data'],fs)
sd.wait()

# view beampattern
# mesh(yout['beampattern'])
# pmesh(yout['beampattern'])

# save audio
# wavfile.write('output/output_fixedbeamformer.wav',16000,yout['data'])


