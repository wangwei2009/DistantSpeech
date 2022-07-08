import numpy as np

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
        self.freq_bin = np.linspace(0, self.half_bin - 1, self.half_bin)
        self.gamma = np.arange(0, 360, int(360 / self.M)) * np.pi / 180

        self.tau = np.zeros((self.M, 1))
        self.omega = 2 * np.pi * self.freq_bin * self.fs / self.n_fft

        self.array_type = arrayType
        self.mic_loc = np.zeros((M, 3))
        self.mic_loc = self.array_init()
        self.array_sim = ArraySim(coordinate=self.mic_loc)

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
        # % for linear arrays, mic0 lines on +x axis
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
            self.mic_loc[:, 0] = -(np.arange(self.M) - (self.M - 1) / 2) * self.r
        else:
            assert self.mic_loc.shape == mic_loc.shape, 'user defined mic location should be 2-D array with shape M X 3'
            self.mic_loc = mic_loc

        return self.mic_loc

    def steering_vector(self, look_direction=0):
        """compute steer vector given look direction

        Parameters
        ----------
        look_direction : int, optional
            look direction, interval in [0, 360),by default 0

        Returns
        -------
        np.array, [M, bin]
            steer vector
        """
        # omega = 2 * np.pi * self.freq_bin * self.fs / self.n_fft
        a = np.zeros((self.M, self.half_bin), dtype=complex)
        for k in range(self.half_bin):
            # tau = np.random.randn(self.M, 1)
            tau = self.compute_tau(incident_angle=np.array([look_direction, 0]) * np.pi / 180)
            a[:, k : k + 1] = np.exp(-1j * self.omega[k] * tau)

        return a.T  # [half_bin, M]

    def compute_tau(self, incident_angle, normalize=False):
        """compute delay time between mic, use (0,0,0) as reference point bu default,
        negative tau[m] indicates signal arrives at mic[m] in advance of zero point

        Parameters
        ----------
        mic_array : MicArray
            mic array object
        incident_angle : np.array, [2] or [2,1]
            source signal imping angle, must be radius
        normalize: bool
            if normalize is True, normalize tau by first mic

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
        p0 = p0[:, np.newaxis]
        p0_norm = np.sqrt(p0.T @ p0)

        c_inv = 1.0 / self.c
        # mic_norm = -1 * np.sqrt(np.sum(self.mic_loc * self.mic_loc, axis=-1))  # [M, ]
        for m in range(self.M):
            mic_loc_m = -1 * self.mic_loc[m : m + 1, :]
            mic_m_norm = np.sqrt(mic_loc_m @ mic_loc_m.T)
            # mic_m_norm = mic_norm[m]
            # cosine angle between impinging signal and mic_m
            cos_theta = mic_loc_m @ p0 / (p0_norm * mic_m_norm + 1e-12)
            # delay between impinging signal and mic_m
            self.tau[m] = -1 * mic_m_norm * cos_theta * c_inv

        if normalize:
            self.tau = self.tau - self.tau[0, 0]

        return self.tau


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
    M = 4
    c = 343
    r = 0.032
    mic_array = MicArray(arrayType='circular', M=4)
    print(mic_array.mic_loc)
    tau = mic_array.compute_tau(np.array([0, 0]) / 180 * np.pi, normalize=True)
    print(tau * mic_array.c)
    # print((tau[-1] - tau[0]) * c)
    # print((M - 1) * r / 2)
    # print(np.abs((tau[-1] - tau[0]) * c - (M - 1) * r / 2))
    # print(np.linalg.norm(mic_array.mic_loc[0] - mic_array.mic_loc[1]))
    # print((tau[0] - tau[1]) * c + r)
    # print((tau[0] * c))
    # print(tau[1])
