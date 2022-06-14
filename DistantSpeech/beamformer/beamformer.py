import numpy as np
from DistantSpeech.beamformer.MicArray import MicArray, compute_tau
from DistantSpeech.beamformer.gen_noise_msc import gen_noise_msc
import warnings
from DistantSpeech.transform.transform import Transform
from matplotlib import pyplot as plt
from scipy.linalg import eigh


def steering(XXs):
    """
    Compute the Steering Vector (rank 1)
    Args:
        XXs (np.ndarray):
            The spatial correlation matrix (nb_of_bins, nb_of_channels, nb_of_channels)
    Returns:
        (np.ndarray):
            The steering vector in the frequency domain (nb_of_bins, nb_of_channels)
    """

    nb_of_bins = XXs.shape[0]
    nb_of_channels = XXs.shape[1]

    # get the biggest eigenvector
    vs = np.linalg.eigh(XXs)[1][:, :, -1]

    # normalized by reference sensor
    v0s = np.tile(np.expand_dims(vs[:, 0], axis=1), (1, nb_of_channels))
    vs /= np.exp(1j * np.angle(v0s))

    return vs


def blind_analytic_normalization(vector, noise_psd_matrix, eps=0):
    """Reduces distortions in beamformed ouptput.

    :param vector: Beamforming vector
        with shape (..., sensors)
    :param noise_psd_matrix:
        with shape (..., sensors, sensors)
    :return: Scaled Deamforming vector
        with shape (..., sensors)

    >>> vector = np.random.normal(size=(5, 6)).view(complex128)
    >>> vector.shape
    (5, 3)
    >>> noise_psd_matrix = np.random.normal(size=(5, 3, 6)).view(complex128)
    >>> noise_psd_matrix = noise_psd_matrix + noise_psd_matrix.swapaxes(-2, -1)
    >>> noise_psd_matrix.shape
    (5, 3, 3)
    >>> w1 = blind_analytic_normalization_legacy(vector, noise_psd_matrix)
    >>> w2 = blind_analytic_normalization(vector, noise_psd_matrix)
    >>> np.testing.assert_allclose(w1, w2)

    """
    nominator = np.einsum('...a,...ab,...bc,...c->...', vector.conj(), noise_psd_matrix, noise_psd_matrix, vector)
    nominator = np.abs(np.sqrt(nominator))

    denominator = np.einsum('...a,...ab,...b->...', vector.conj(), noise_psd_matrix, vector)
    denominator = np.abs(denominator)

    normalization = nominator / (denominator + eps)
    return vector * normalization[..., np.newaxis]


def phase_correction(vector):
    """Phase correction to reduce distortions due to phase inconsistencies.
    Args:
        vector: Beamforming vector with shape (..., bins, sensors).
    Returns: Phase corrected beamforming vectors. Lengths remain.
    """
    w = vector.copy()
    F, D = w.shape
    for f in range(1, F):
        w[f, :] *= np.exp(-1j * np.angle(np.sum(w[f, :] * w[f - 1, :].conj(), axis=-1, keepdims=True)))
    return w


def get_gev_vector(target_psd_matrix, noise_psd_matrix):
    """
    Returns the GEV beamforming vector.
    :param target_psd_matrix: Target PSD matrix
        with shape (bins, sensors, sensors)
    :param noise_psd_matrix: Noise PSD matrix
        with shape (bins, sensors, sensors)
    :return: Set of beamforming vectors with shape (bins, sensors)
    """
    bins, sensors, _ = target_psd_matrix.shape
    beamforming_vector = np.empty((bins, sensors), dtype=complex)
    for f in range(bins):
        try:
            eigenvals, eigenvecs = eigh(target_psd_matrix[f, :, :], noise_psd_matrix[f, :, :])
            beamforming_vector[f, :] = eigenvecs[:, -1]
        except np.linalg.LinAlgError:
            print('LinAlg error for frequency {}'.format(f))
            beamforming_vector[f, :] = np.ones((sensors,)) / np.trace(noise_psd_matrix[f]) * sensors
    return beamforming_vector


def compute_pmwf_weight(xi, Rxx, Rvv_inv, Gmin=0.0631, beta=1):
    """compute parameterized multichannel non-causal Wiener filter
    refer to
        "On Optimal Frequency-Domain Multichannel Linear Filtering for Noise Reduction"

    Parameters
    ----------
    xi : np.array, [half_bin]
        prior snr
    Rxx : np.array, [M, M, bin]
        target psd matrix
    Rvv_inv : np.array, [M, M, bin]
        inversion of noise psd matrix
    Gmin : float, optional
        _description_, by default 0.0631
    beta : int, optional
        _description_, by default 1

    Returns
    -------
    w: complex np.array, [half_bin, M]
        complex pwmwf weights

    """
    half_bin = xi.shape[0]
    channels = Rxx.shape[0]
    u = np.zeros((half_bin, channels, 1))
    u[:, 0, 0] = 1
    w = (Rvv_inv @ Rxx @ u).squeeze() / (beta + xi[:, None])

    return w


def compute_mvdr_weight(steer_vector, Rvv_inv, Gmin=0.0631, beta=1):
    """compute mvdr weight given steer vector and Rvv_inv

    Parameters
    ----------
    steer_vector : complex np.array, [bins, M]
        steer vector, constructed from tdoa(free field) or pca(reverb)
    Rvv_inv : complex np.array, [bins, M, M]
        inversion of noise spatial correlation matrix(PSD matrix)
    Gmin : float, optional
        _description_, by default 0.0631
    beta : int, optional
        _description_, by default 1

    Returns
    -------
    w: complex weights, [bins, M]
        mvdr weights
    """
    num = Rvv_inv @ steer_vector[..., None]  # [bins, M, 1]
    w = num / (steer_vector[:, None, :].conj() @ num)

    return w.squeeze()


class beamformer(object):
    """
    beamformer base class
    """

    def __init__(self, mic=MicArray, frame_len=256, hop=None, nfft=None, c=343, r=0.032, fs=16000):
        self.MicArray = mic
        self.M = mic.M

        self.frameLen = frame_len
        if hop is None:
            self.hop = int(frame_len // 2)
        else:
            self.hop = int(hop)
        self.overlap = frame_len - self.hop
        if nfft is None:
            self.nfft = int(frame_len)
        else:
            self.nfft = int(nfft)
        self.c = c
        self.r = r
        self.fs = fs
        self.half_bin = round(self.nfft / 2 + 1)
        self.freq_bin = np.linspace(0, self.half_bin - 1, self.half_bin)
        self.omega = 2 * np.pi * self.freq_bin * self.fs / self.nfft

        self.Ryy = np.zeros((self.M, self.M, self.half_bin), dtype=complex)
        self.Rss = np.zeros((self.M, self.M, self.half_bin), dtype=complex)
        self.Rnn = np.zeros((self.M, self.M, self.half_bin), dtype=complex)

        for k in range(self.half_bin):
            self.Ryy[:, :, k] = np.eye(self.M, dtype=complex)
            self.Rss[:, :, k] = np.eye(self.M, dtype=complex)
            self.Rnn[:, :, k] = np.eye(self.M, dtype=complex)

        # generate diffuse model
        self.Fvv = gen_noise_msc(mic=self.MicArray, M=self.M, r=self.r, nfft=self.nfft, fs=self.fs, c=self.c)

        self.transformer = Transform(n_fft=self.nfft, hop_length=self.hop, channel=self.M)

    def get_steering_vector(self):
        pass

    def get_covariance(self):
        pass

    def get_covariance_yy(self, z, alpha=0.92):
        """
        compute noisy covariance matrix
        :param z: [half_bin, ch]
        :param alpha: smoothing factor
        :return: [ch. ch, half_bin]
        """
        for k in range(self.half_bin):
            self.Ryy[:, :, k] = alpha * self.Ryy[:, :, k] + (1 - alpha) * z[k, :].conj().T @ z[k, :]

        return self.Ryy

    def getweights(self, a, weightType="DS", Rvv=None, Rvv_inv=None, Ryy=None, Diagonal=1e-3):
        """
        compute beamformer weights

        """
        if Rvv is None:
            warnings.warn("Rvv not provided,using eye(M,M)\n")
            Rvv = np.eye(self.M)
        if Rvv_inv is None:
            Fvv_k = (Rvv) + Diagonal * np.eye(self.M)  # Diagonal loading
            Fvv_k_inv = np.linalg.inv(Fvv_k)
        else:
            Fvv_k_inv = Rvv_inv

        if weightType == "src":
            weights = a
            weights[1:] = 0  # output channel 1
        elif weightType == "DS":
            weights = a / self.M  # delay-and-sum weights
        elif weightType == "MVDR":
            weights = Fvv_k_inv @ a / (a.conj().T @ Fvv_k_inv @ a)  # MVDR weights
        elif weightType == "TFGSC":
            # refer to Jingdong Chen, "Noncausal (Frequency-Domain) Optimal
            # Filters," in Microphone Array Signal Processing,page.134-135
            u = np.zeros((self.M, 1))
            u[0] = 1
            temp = Fvv_k_inv @ Ryy
            weights = (temp - np.eye(self.M)) @ u / (np.trace(temp) - self.M)  # FD-TFGSC weights
        else:
            raise ValueError("Unknown beamformer weights: %s" % weightType)
        return weights

    def calcWNG(self, ak, Hk):
        """
        calculate White Noise Gain per frequency bin

        """
        WNG = np.squeeze(np.abs(Hk.conj().T @ ak) ** 2 / np.real((Hk.conj().T @ Hk)))

        return 10 * np.log10(WNG)

    def calcDI(self, ak, Hk, Fvvk):
        """
        calculate directive index per frequency bin

        """
        WNG = np.squeeze(np.abs(Hk.conj().T @ ak) ** 2 / np.real((Hk.conj().T @ Fvvk @ Hk)))

        return 10 * np.log10(WNG)

    def compute_beampattern(self, mic_array: MicArray, weights=None, look_angle=np.array([0, 0]) / 180 * np.pi):
        """_summary_

        Parameters
        ----------
        mic_array : MicArray
            mic array object
        weights : np.array, [M, half_bin], optional
            beamformer complex weights, if not given, use delay and sum weight, by default None
        look_angle : np.array, optional
            look angle, by default np.array([0, 0])/180*np.pi

        Returns
        -------
        beamout: np.array, [360, half_bin]
            2-d beampattern
        """
        tau0 = compute_tau(mic_array, look_angle)
        beamout = np.zeros([360, mic_array.half_bin])
        H = weights if weights is not None else np.zeros((mic_array.M, mic_array.half_bin), dtype=complex)
        for az in range(0, 360, 1):
            tau = compute_tau(mic_array, np.array([az * np.pi / 180, 0]))
            for k in range(0, mic_array.half_bin):
                a0 = np.exp(-1j * self.omega[k] * tau0)
                if weights is None:
                    H[:, k : k + 1] = self.getweights(a0, weightType="DS")
                a = np.exp(-1j * self.omega[k] * tau)
                beamout[az, k] = np.abs(np.squeeze(H[:, k : k + 1].conj().T @ a))

        return 20 * np.log10(beamout + 1e-12)

    def beampattern(self, omega, H):
        """
        calculate directive index per frequency bin

        """
        half_bin = H.shape[1]
        r = 0.032
        angle = np.arange(0, 360, 360)
        beamout = np.zeros([360, half_bin])
        for az in range(0, 360, 1):
            tao = -1 * r * np.cos(0) * np.cos(az * np.pi / 180 - self.gamma) / self.c
            tao = tao[:, np.newaxis]
            for k in range(0, half_bin):
                a = np.exp(-1j * omega[k] * tao)
                beamout[az, k] = np.abs(np.squeeze(H[:, k, np.newaxis].conj().T @ a))

        return 10 * np.log10(beamout)


if __name__ == "__main__":

    sr = 16000

    frameLen = 256
    hop = frameLen / 2
    overlap = frameLen - hop
    nfft = 256
    c = 340
    r = 0.08
    fs = sr

    MicArrayObj = MicArray(arrayType="linear", r=r, M=10)
    angle = np.array([90, 0]) / 180 * np.pi  # look angle

    beamformer_obj = beamformer(MicArrayObj, frameLen, hop, nfft, c, fs)

    beampattern = beamformer_obj.compute_beampattern(MicArrayObj, look_angle=angle)

    # create same figure as Page.45 in "Microphone Array Signal Processing"
    plt.figure()
    plt.plot(beampattern[0:180, 32])  # k*fs/N = 2KHz
    plt.ylim([-50, 0])
    plt.grid()
    plt.savefig('beam.png')

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    theta = np.arange(0, 180, 1) / 180 * np.pi
    ax.plot(theta, beampattern[0:180, 32])
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    fig.savefig('beam_polar.png')
