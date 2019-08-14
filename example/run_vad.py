"""
Test beamformer

"""

import numpy as np

import time

from beamformer.MicArray import MicArray
from beamformer.fixedbeamformer import fixedbeamformer
from beamformer.adaptivebeamformer import adaptivebeamfomer
from beamformer.utils import mesh,pmesh,load_wav,load_pcm

import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
from scipy.io import wavfile
import webrtcvad

filepath = "test_audio/rec1/"
x,sr = load_wav(filepath)

vad = webrtcvad.Vad(3)
framLen = 160
frameNum = len(x[0,:])//framLen
for i in range(0,frameNum):
    frame = (x[0,i*framLen:(i+1)*framLen] * 32768).astype('<i2').tostring()
    result = vad.is_speech(frame,16000)
    if result == False:
        print("noise  =", i*128.0/16000.0, "\n")