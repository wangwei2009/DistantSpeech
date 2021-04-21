from scipy.signal import windows
from scipy import signal
import numpy as np
import warnings
from DistantSpeech.beamformer.ArraySim import ArraySim


class MicArray(object):

    def __init__(self, arrayType='circular', r=0.032, c=343, M=4, n_fft=256):
        self.arrayType = arrayType
        self.c = c
        self.r = r
        self.fs = 16000
        self.M = M
        self.n_fft = n_fft
        self.half_bin = round(self.n_fft / 2 + 1)
        self.angle = np.array([197, 0]) / 180 * np.pi
        self.gamma = np.arange(0, 360, int(360 / self.M)) * np.pi / 180

        self.array_sim = ArraySim(arrayType, spacing=r, M=M)
