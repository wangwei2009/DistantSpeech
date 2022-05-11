from scipy.signal import windows
from scipy import signal
import numpy as np
import warnings

from DistantSpeech.beamformer.ArraySim import ArraySim


def cart2sph(x, y, z):
    azimuth = np.arctan2(y, x)
    elevation = np.arctan2(z, np.sqrt(x**2 + y**2))
    r = np.sqrt(x**2 + y**2 + z**2)
    return azimuth, elevation, r


def sph2cart(azimuth, elevation, r):
    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)
    return x, y, z


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

        self.array_type = arrayType
        self.mic_loc = np.zeros((M, 3))
        self.mic_loc = self.array_init()

        """
        # %                   ^ +z
        # %                   |
        # %                   |
        # %                   |
        # %                   |
        # %                    --------------------> 90 degree
        # %                  /                    +y
        # %                 /
        # %                /
        # %               / +x
        # %              0 degree
        # % for linear arrays, mic0 lines on -y axis
        # % 
        # % for circular arrays, mic0 lies on +x axis
        """

    def array_init(self, mic_loc=None):
        if self.array_type == 'circular':
            az = np.arange(0, 360, int(360 / self.M)) * np.pi / 180  # azimuth of each mic
            elevation = 0
            for m in range(self.M):
                self.mic_loc[m, :] = sph2cart(az[m], elevation, self.r)
        elif self.array_type == 'linear':
            self.mic_loc[:, 1] = (np.arange(self.M) - (self.M - 1) / 2) * self.r
        else:
            assert self.mic_loc.shape == mic_loc.shape, 'user defined mic location should be 2-D array with shape M X 3'
            self.mic_loc = mic_loc

        return self.mic_loc


if __name__ == "__main__":
    mic_array = MicArray(arrayType='circular', M=8)
    print(mic_array.mic_loc)
