import numpy as np
from DistantSpeech.beamformer.MicArray import MicArray, compute_tau
from DistantSpeech.beamformer.gen_noise_msc import gen_noise_msc
import warnings
from DistantSpeech.transform.transform import Transform
from DistantSpeech.beamformer.utils import mesh
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


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
    angle = np.array([0, 0]) / 180 * np.pi  # look angle

    beamformer_obj = beamformer(MicArrayObj, frameLen, hop, nfft, c, fs)

    beampattern = beamformer_obj.compute_beampattern(MicArrayObj, look_angle=angle)

    # create same figure as Page.45 in "Microphone Array Signal Processing"
    plt.figure()
    plt.plot(beampattern[90:270, 32])  # k*fs/N = 2KHz
    plt.ylim([-50, 0])
    plt.grid()
    plt.savefig('beam.png')

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    theta = np.arange(0, 180, 1) / 180 * np.pi
    ax.plot(theta, beampattern[90:270, 32])
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    fig.savefig('beam_polar.png')
