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
        # % for linear arrays, mic0 lines on +y axis
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
            self.mic_loc[:, 1] = -(np.arange(self.M) - (self.M - 1) / 2) * self.r
        else:
            assert self.mic_loc.shape == mic_loc.shape, 'user defined mic location should be 2-D array with shape M X 3'
            self.mic_loc = mic_loc

        return self.mic_loc


def compute_tau(mic_array: MicArray, incident_angle):
    """compute delay time between mic, use (0,0,0) as reference point bu default,
    negative tau[m] indicates signal arrives at mic[m] in advance of zero point

    Parameters
    ----------
    mic_array : MicArray
        mic array object
    incident_angle : np.array, [2] or [2,1]
        source signal imping angle, must be radius

    Returns
    -------
    tau: np.array, [M, 1]
        delay time between mic

    Examples
    -------
    >>> mic_array = MicArray(arrayType='linear', M=4)
    >>> compute_tau(mic_array, np.array([30, 0]) / 180 * np.pi)
    >>> print(tau)
    [[ 0.024]
    [ 0.008]
    [-0.008]
    [-0.024]]
    """
    az = incident_angle[0]
    el = incident_angle[1] if len(incident_angle.shape) > 0 else 0
    x0, y0, z0 = sph2cart(az, el, 1)
    p0 = -1 * np.array([x0, y0, z0])  # unit-vector for impinging signal
    tau = np.zeros((mic_array.M, 1))
    for m in range(mic_array.M):
        mic_loc_m = -1 * mic_array.mic_loc[m, :]
        # cosine angle between impinging signal and mic_m
        cos_theta = np.sum(mic_loc_m * p0) / (np.linalg.norm(p0) * np.linalg.norm(mic_loc_m) + 1e-12)
        # delay between impinging signal and mic_m
        tau[m] = -1 * np.linalg.norm(mic_loc_m) * cos_theta / mic_array.c

    return tau


if __name__ == "__main__":
    mic_array = MicArray(arrayType='linear', M=4)
    print(mic_array.mic_loc)
    tau = compute_tau(mic_array, np.array([30, 0]) / 180 * np.pi)
    print(tau * mic_array.c)
