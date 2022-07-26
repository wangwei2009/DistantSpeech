import numpy as np
import math
from scipy.signal import windows
from DistantSpeech.coherence.getweights_coherent import getweghts_coherent
from DistantSpeech.postfilter.postfilter import PostFilter
from DistantSpeech.beamformer.beamformer import beamformer


class BinauralEnhancement(beamformer):
    def __init__(self, MicArray, frameLen=256, hop=None, nfft=None, c=343, d=0.032, fs=16000):

        beamformer.__init__(self, MicArray, frame_len=frameLen, hop=hop, nfft=nfft, c=c, r=d / 2, fs=fs)
        self.M = MicArray.M
        self.fs = MicArray.fs
        self.d = d
        self.G = np.ones([self.half_bin], dtype=complex)
        self.Fvv_est = np.ones([self.half_bin, self.M, self.M], dtype=complex) * 0.98

        self.NumSpec = int((self.M * self.M - self.M) / 2)
        self.Pxii = np.zeros([self.half_bin, self.M])
        self.Pxij = np.zeros([self.half_bin, self.NumSpec], dtype=complex)

    def updateMSC(self, alpha=0, UPPPER_THRESHOLD=0.98):
        t = 0
        for i in range(0, self.M - 1):
            for j in range(i + 1, self.M):
                self.Fvv_est[:, i, j] = self.Pxij[:, t] / np.sqrt(self.Pxii[:, i] * self.Pxii[:, j])
                t = t + 1
        # np.where(self.Fvv_est > UPPPER_THRESHOLD, UPPPER_THRESHOLD, self.Fvv_est)

    def update_CSD_PSD(self, Z: np.ndarray, alpha=0.8):
        """
        :param
            Z: input spectral,[fft_bin, Nele]

        :return:
            W: post-filter weights
        """
        t = 0
        M = Z.shape[1]

        # update auto-spectral
        # Pxii_curr = np.einsum('ij,il->ijl', Z, Z.conj())
        # self.Pxii = alpha * self.Pxii + (1 - alpha) * Pxii_curr

        # update auto-spectral
        for i in range(0, M):
            # auto-spectral
            Pxii_curr = np.real(Z[:, i] * Z[:, i].conj())
            self.Pxii[:, i] = alpha * self.Pxii[:, i] + (1 - alpha) * Pxii_curr

        # update cross-spectral
        for i in range(0, M - 1):
            for j in range(i + 1, M):
                # cross - spectral
                Pxij_curr = Z[:, i] * Z[:, j].conj()
                # average
                self.Pxij[:, t] = alpha * self.Pxij[:, t] + (1 - alpha) * Pxij_curr
                t = t + 1

    def getweights(self, Z: np.ndarray, method=3):
        """
        :param
            Z: input spectral,[Nele,fft_bin]

        :return:
            W: post-filter weights
        """
        alpha = 0.6
        self.update_CSD_PSD(Z, alpha=alpha)
        self.updateMSC()
        Fvv_UPPER = 0.98

        snr = np.zeros((self.half_bin,))

        for k in range(0, self.half_bin):
            self.G[k], snr[k] = getweghts_coherent(self.Fvv_est[k, 0, 1], self.Fvv[k, 0, 1], k, method=method, r=self.d)

        return self.G, snr

    def process(self, x):
        """
        process core function
        :param x: input time signal, (samples, channels)
        :return: enhanced signal, (n_samples,)
        """

        assert x.shape[1] == 2

        output = np.zeros(x.shape)

        D = self.transform.stft(x)

        frameNum = D.shape[1]

        snr = np.zeros((self.half_bin, frameNum))

        for n in range(frameNum):
            X_n = D[:, n, :]

            self.H, snr[:, n] = self.getweights(X_n, method=3)

            yf = self.H[:, None] * X_n

            x_n = self.transform.istft(yf)

            output[n * self.hop : (n + 1) * self.hop] = x_n

        return output, snr


if __name__ == "__main__":

    from DistantSpeech.beamformer.MicArray import MicArray

    sr = 16000
    r = 0.032
    c = 343

    frameLen = 256
    hop = frameLen / 2
    overlap = frameLen - hop
    nfft = 256
    c = 340
    r = 0.032
    fs = sr

    MicArrayObj = MicArray(arrayType="circular", r=0.032, M=2)
    angle = np.array([197, 0]) / 180 * np.pi

    dual_mic = BinauralEnhancement(MicArrayObj, frameLen, hop, nfft, c, fs)

    x = np.random.rand(16000 * 3, 2)

    yout = dual_mic.process(x)
