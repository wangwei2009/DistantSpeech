"""
Test beamformer

"""

import numpy as np

import time

from DistantSpeech.beamformer.MicArray import MicArray
from DistantSpeech.beamformer.adaptivebeamformer import adaptivebeamfomer
from DistantSpeech.beamformer.utils import mesh, pmesh, load_wav, load_pcm

import matplotlib.pyplot as plt

# import sounddevice as sd
import soundfile as sf
from scipy.io import wavfile

# filepath = "test_audio/rec1/"
# x, sr = load_wav(filepath)

filepath = "/home/wangwei/work/speechenhancement/audio_data/3m"
x = load_pcm(filepath)  # [N, samples]
print(x.shape)

sr = 16000
r = 0.032
c = 343
M = 2

frameLen = 256
hop = frameLen / 2
overlap = frameLen - hop
nfft = 256
c = 340
r = 0.032
fs = sr


MicArray = MicArray(arrayType='linear', r=0.04, M=M)
angle = np.array([0, 0]) / 180 * np.pi

adaptivebeamfomer_obj = adaptivebeamfomer(MicArray, frameLen, hop, nfft, c, r, fs)
yout = adaptivebeamfomer_obj.process(x, angle, method=3)

# save audio
wavfile.write('output_3m.wav', 16000, yout['data'])
