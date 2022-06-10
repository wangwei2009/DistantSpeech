from scipy.signal import windows
from scipy import signal
import numpy as np
from .MicArray import MicArray
from .beamformer import beamformer
import warnings


class FixedBeamformer(beamformer):
    def __init__(self, MicArray, frameLen=256, hop=None, nfft=None, c=343, fs=16000):

        beamformer.__init__(self, MicArray, frame_len=frameLen, hop=hop, nfft=nfft, c=c, fs=fs)
        self.angle = np.array([197, 0]) / 180 * np.pi
        self.gamma = MicArray.gamma
        self.window = windows.hann(self.frameLen, sym=False)
        self.win_scale = np.sqrt(1.0 / self.window.sum() ** 2)
        self.freq_bin = np.linspace(0, self.half_bin - 1, self.half_bin)
        self.omega = 2 * np.pi * self.freq_bin * self.fs / self.nfft

        self.H = np.ones([self.M, self.half_bin], dtype=complex)

        self.angle = np.array([0, 0]) / 180 * np.pi

        self.AlgorithmList = ["src", "DS", "MVDR"]
        self.AlgorithmIndex = 0

    def process(self, x, angle, method=1, retH=False, retWNG=False, retDI=False):
        """
        fixed beamformer precesing function
        :param x: input data, [channels, samples]
        :param angle: incident angle, [-pi, pi]
        :param method:
        :param retH:
        :param retWNG:
        :param retDI:
        :return:
        """
        """
        method:
        'DS':   delay-and-sum beamformer
        'MVDR': MVDR beamformer under isotropic noise field
        """
        X = self.transformer.stft(x.transpose())
        frameNum = X.shape[1]

        tao = -1 * self.r * np.cos(angle[1]) * np.cos(angle[0] - self.gamma) / self.c

        if retWNG:
            WNG = np.ones(self.half_bin)
        else:
            WNG = None
        if retDI:
            DI = np.ones(self.half_bin)
        else:
            DI = None
        if retH is False:
            beampattern = None

        # Fvv = gen_noise_msc(M, self.nfft, self.fs, self.r)
        # H = np.mat(np.ones([self.half_bin, self.M]), dtype=complex).T
        if (all(angle == self.angle) is False) or (method != self.AlgorithmIndex):
            if method != self.AlgorithmIndex:
                self.AlgorithmIndex = method
            for k in range(0, self.half_bin):
                a = np.mat(np.exp(-1j * self.omega[k] * tao)).T  # propagation vector
                self.H[:, k, np.newaxis] = self.getweights(
                    a, self.AlgorithmList[method], self.Fvv[k, :, :], Diagonal=1e-1
                )

                if retWNG:
                    WNG[k] = self.calcWNG(a, self.H[:, k, np.newaxis])
                if retDI:
                    DI[k] = self.calcDI(a, self.H[:, k, np.newaxis], self.Fvv[k, :, :])

        Y = np.ones([self.half_bin, frameNum], dtype=complex)
        for t in range(0, frameNum):
            Z = X[:, t, :].transpose()

            x_fft = np.array(np.conj(self.H)) * Z
            Y[:, t] = np.sum(x_fft, axis=0)

        yout = self.transformer.istft(Y)

        # update angle
        self.angle = angle

        # calculate beampattern
        if retH:
            beampattern = self.beampattern(self.omega, self.H)

        return {"data": yout, "WNG": WNG, "DI": DI, "beampattern": beampattern}


if __name__ == "__main__":
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

    MicArrayObj = MicArray(arrayType="circular", r=0.032, M=4)
    angle = np.array([197, 0]) / 180 * np.pi

    fixedbeamformerObj = FixedBeamformer(MicArrayObj, frameLen, hop, nfft, c, fs)

    x = np.random.rand(16000 * 5)

    # """
    # fixed beamformer precesing function
    # method:
    # 'DS':   delay-and-sum beamformer
    # 'MVDR': MVDR beamformer under isotropic noise field
    #
    # """
    yout = fixedbeamformerObj.process(x, angle, method=2, retH=True, retWNG=True, retDI=True)
