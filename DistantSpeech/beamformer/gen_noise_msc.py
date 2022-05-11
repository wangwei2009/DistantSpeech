import numpy as np
import warnings
from matplotlib import pyplot as plt
from DistantSpeech.beamformer.MicArray import MicArray


def gen_noise_msc(mic=MicArray, M=4, r=0.032, nfft=256, fs=16000, c=340):
    half_bin = round(nfft / 2 + 1)
    Fvv = np.zeros((half_bin, M, M))
    f = np.linspace(0, fs / 2, half_bin)
    f[0] = 1e-6
    k_optimal = 1
    for i in range(0, M):
        for j in range(0, M):
            dij = np.sqrt(np.sum((mic.mic_loc[i, :] - mic.mic_loc[j, :]) ** 2))
            if i == j:
                Fvv[0, i, j] = 0.998
            else:
                Fvv[:, i, j] = np.sin(2 * np.pi * f * dij / c) / (2 * np.pi * f * dij / c)

    return Fvv


if __name__ == "__main__":
    mic_array = MicArray(arrayType='linear')
    gamma = gen_noise_msc(mic=mic_array)
    plt.figure()
    i, j = 0, 2
    distance = np.linalg.norm(mic_array.mic_loc[i, :] - mic_array.mic_loc[j, :])
    plt.plot(gamma[:, i, j])
    plt.show()
    plt.title('ideal diffuse noise field between mic{} and mic{}\n(distance:{} cm)'.format(j, j, distance))
    plt.savefig('gamma.png')
