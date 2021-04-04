import numpy as np
from .MicArray import MicArray
import warnings


class beamformer(MicArray):
    """
    beamformer base class
    """

    def __init__(self, mic=MicArray, frame_len=256, hop=None, nfft=None, c=343, r=0.032, fs=16000):
        MicArray.__init__(self, arrayType=mic.arrayType, r=mic.r, M=mic.M)
        self.M = mic.M

        if hop is None:
            self.hop = int(frame_len//2)
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

        self.Ryy = np.zeros((self.M, self.M, self.half_bin), dtype=np.complex)
        self.Rss = np.zeros((self.M, self.M, self.half_bin), dtype=np.complex)
        self.Rnn = np.zeros((self.M, self.M, self.half_bin), dtype=np.complex)

        for k in range(self.half_bin):
            self.Ryy[:, :, k] = np.eye(self.M, dtype=np.complex)
            self.Rss[:, :, k] = np.eye(self.M, dtype=np.complex)
            self.Rnn[:, :, k] = np.eye(self.M, dtype=np.complex)

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

    def getweights(self, a, weightType='DS', Rvv=None, Rvv_inv=None, Ryy=None, Diagonal=1e-3):
        """
        compute beamformer weights

        """
        if Rvv is None:
            warnings.warn("Rvv not provided,using eye(M,M)\n")
            Rvv = np.eye([self.M, self.M])
        if Rvv_inv is None:
            Fvv_k = (Rvv) + Diagonal * np.eye(self.M)  # Diagonal loading
            Fvv_k_inv = np.linalg.inv(Fvv_k)
        else:
            Fvv_k_inv = Rvv_inv

        if weightType == 'src':
            weights = a
            weights[1:] = 0  # output channel 1
        elif weightType == 'DS':
            weights = a / self.M  # delay-and-sum weights
        elif weightType == 'MVDR':
            weights = Fvv_k_inv @ a / \
                      (a.conj().T @ Fvv_k_inv @ a)  # MVDR weights
        elif weightType == 'TFGSC':
            # refer to Jingdong Chen, "Noncausal (Frequency-Domain) Optimal
            # Filters," in Microphone Array Signal Processing,page.134-135
            u = np.zeros((self.M, 1))
            u[0] = 1
            temp = Fvv_k_inv @ Ryy
            weights = (temp - np.eye(self.M)) @ u / \
                      (np.trace(temp) - self.M)  # FD-TFGSC weights
        else:
            raise ValueError('Unknown beamformer weights: %s' % weightType)
        return weights

    def calcWNG(self, ak, Hk):
        """
        calculate White Noise Gain per frequency bin

        """
        WNG = np.squeeze(np.abs(Hk.conj().T @ ak) ** 2 / \
                         np.real((Hk.conj().T @ Hk)))

        return 10 * np.log10(WNG)

    def calcDI(self, ak, Hk, Fvvk):
        """
        calculate directive index per frequency bin

        """
        WNG = np.squeeze(np.abs(Hk.conj().T @ ak) ** 2 / \
                         np.real((Hk.conj().T @ Fvvk @ Hk)))

        return 10 * np.log10(WNG)

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
