import numpy as np


def gen_noise_msc(M, nfft, fs, r):

    half_bin = round(nfft/2+1)
    Fvv = np.zeros((half_bin, M, M))
    N_FFT = nfft
    c = 340
    f = np.linspace(0,fs/2,half_bin)
    f[0] = 1e-6
    k_optimal = 1
    for i in range(0,M):
        for j in range(0,M):
            if i == j:
                Fvv[:,i,j] = np.ones(half_bin)
            else:
                if np.abs(i-j) == 1:
                    dij = r*np.sqrt(2)
                elif np.abs(i-j) == 2:
                    dij = r*2
                elif np.abs(i-j) == 3:
                    dij = r*np.sqrt(2)
                Fvv[:,i,j] = np.sin(2 * np.pi * f * dij / c)/(2 * np.pi * f * dij/ c)
                Fvv[0,i,j] = 0.998
    return Fvv
