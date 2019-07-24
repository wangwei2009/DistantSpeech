from scipy.signal import windows
from scipy import signal
import numpy as np
import warnings


class MicArray(object):

    def __init__(self, arrayType = 'circular',r = 0.032, c=343, M = 4):

        self.arrayType = arrayType
        self.c = c
        self.r = r
        self.fs = 16000
        self.M = M
        self.angle = np.array([197, 0]) / 180 * np.pi
        self.gamma = np.arange(0,360,int(360/self.M))* np.pi / 180

        self.Fvv = gen_noise_msc(self)


def gen_noise_msc(mic=MicArray, nfft=256, fs=16000):

    M = mic.M
    r = mic.r
    half_bin = round(nfft/2+1)
    Fvv = np.zeros((half_bin, M, M))
    c = 340
    f = np.linspace(0,fs/2,half_bin)
    f[0] = 1e-6
    k_optimal = 1

    if mic.arrayType == 'circular':
        for i in range(0,M):
            for j in range(0,M):
                if i == j:
                    Fvv[:,i,j] = np.ones(half_bin)
                else:
                    mic_rad = np.abs(mic.gamma[i]-mic.gamma[j])
                    dij = np.sqrt(r**2+r**2-2*r*r*np.cos(mic_rad))
                    Fvv[:,i,j] = np.sin(2 * np.pi * f * dij / c)/(2 * np.pi * f * dij/ c)
                    Fvv[0,i,j] = 0.998
    if mic.arrayType == 'linear':
        for i in range(0,M):
            for j in range(0,M):
                if i == j:
                    Fvv[:,i,j] = np.ones(half_bin)
                else:
                    dij = np.abs(i-j)*r
                    Fvv[:,i,j] = np.sin(2 * np.pi * f * dij / c)/(2 * np.pi * f * dij/ c)
                    Fvv[0,i,j] = 0.998
    if mic.arrayType == 'arbitrary':
        warnings.warn("diffuse noise field for arbitrary not implemented yet!")
    return Fvv
