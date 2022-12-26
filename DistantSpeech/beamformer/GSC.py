"""
time domain GSC beamformer
====================

----------



"""
import numpy as np
from scipy.signal import windows

from DistantSpeech.noise_estimation.mcra import NoiseEstimationMCRA
from DistantSpeech.noise_estimation.mcspp_base import McSppBase
from DistantSpeech.noise_estimation.omlsa_multi import NsOmlsaMulti
from DistantSpeech.noise_estimation.mc_mcra import McMcra
from DistantSpeech.noise_estimation.mcspp import McSpp
from DistantSpeech.transform.transform import Transform
from DistantSpeech.beamformer.beamformer import beamformer
from DistantSpeech.beamformer.fixedbeamformer import TimeAlignment
from DistantSpeech.beamformer.MicArray import MicArray
from DistantSpeech.adaptivefilter.FastFreqLms import FastFreqLms
from DistantSpeech.adaptivefilter.feature import Emphasis, FilterDcNotch16


class GSC(beamformer):
    def __init__(
        self,
        mic_array: MicArray,
        frameLen=256,
        angle=[197, 0],
    ):
        beamformer.__init__(self, mic_array, frame_len=frameLen)
        self.mic_array = mic_array

        self.angle = np.array(angle) / 180 * np.pi if isinstance(angle, list) else angle
        self.gamma = mic_array.gamma
        self.window = windows.hann(self.frameLen, sym=False)
        self.win_scale = np.sqrt(1.0 / self.window.sum() ** 2)
        self.freq_bin = np.linspace(0, self.half_bin - 1, self.half_bin)
        self.omega = 2 * np.pi * self.freq_bin * self.fs / self.nfft

        self.window = np.sqrt(windows.hann(self.frameLen, sym=False))

        self.transformer = Transform(n_fft=self.nfft, hop_length=self.hop, channel=self.M)

        # self.pad_data = np.zeros([mic_array.M, round(frameLen / 2)])
        # self.last_output = np.zeros(round(frameLen / 2))

        self.H = np.ones([self.M, self.half_bin], dtype=complex) / self.M

        self.method = 'MVDR'

        self.frameCount = 0
        self.calc = 1
        self.estPos = 200

        self.Rvv = np.zeros((self.half_bin, self.M, self.M), dtype=complex)
        self.Rvv_inv = np.zeros((self.half_bin, self.M, self.M), dtype=complex)
        self.Ryy = np.zeros((self.half_bin, self.M, self.M), dtype=complex)

        self.AlgorithmList = ['src', 'DS', 'MVDR', 'TFGSC']
        self.AlgorithmIndex = 0

        # blocking matrix
        self.BM = np.zeros((self.M, self.M - 1, self.half_bin), dtype=complex)
        # noise reference
        self.U = np.zeros((self.M - 1, self.half_bin), dtype=complex)
        # fixed beamformer weights for upper path
        self.W = np.zeros((self.M, self.half_bin), dtype=complex)
        # MNC weights for lower path
        self.G = np.zeros((self.M - 1, self.half_bin), dtype=complex)
        self.Pest = np.ones(self.half_bin)

        self.Yfbf = np.zeros((self.half_bin), dtype=complex)

        self.mcra = NoiseEstimationMCRA(nfft=self.nfft)
        self.omlsa_multi = NsOmlsaMulti(nfft=self.nfft, cal_weights=True, M=self.M)
        self.mcspp = McSppBase(nfft=self.nfft, channels=self.M)
        self.mc_mcra = McMcra(nfft=self.nfft, channels=self.M)
        self.spp = self.mc_mcra

        self.time_alignment = TimeAlignment(mic_array, angle=self.angle)
        self.aic_filter = FastFreqLms(filter_len=frameLen, n_channels=self.M - 1)
        self.dc_notch_mic = []
        for _ in range(self.M):
            self.dc_notch_mic.append(FilterDcNotch16(radius=0.98))

    def fixed_beamformer(self, x):
        """fixed beamformer

        Parameters
        ----------
        x : np.array
            input multichannel data, [samples, chs]

        Returns
        -------
        np.array
            output data, [samples, 1]
        """

        return np.mean(x, axis=1, keepdims=True)

    def blocking_matrix(self, x):
        """fixed blocking matrix

        Parameters
        ----------
        x : np.array
            input multichannel data, [samples, chs]

        Returns
        -------
        bm_output : np.array
            output data, [samples, chs-1]
        """
        samples, channels = x.shape
        bm_output = np.zeros((samples, channels - 1))
        for m in range(channels - 1):
            bm_output[:, m] = x[:, m] - x[:, m + 1]

        return bm_output

    def aic(self, y_fbf, bm_output):
        """adaptive interference cancellation block

        Parameters
        ----------
        y_fbf : np.array
            output from fixed beamformer, upper path, (n_samples, 1)
        bm_output : np.array
            output from blocking matrix output, lower path, (n_samples, n_chs)

        Returns
        -------
        output_n : np.array
            _description_
        """
        # AIC block
        output_n, _ = self.aic_filter.update(bm_output, y_fbf, fir_truncate=30)

        return output_n

    def process1(self, x):

        samples, channels = x.shape
        output = np.zeros(samples)

        for m in range(channels):
            x[:, m], self.dc_notch_mic[m].mem = self.dc_notch_mic[m].filter_dc_notch16(x[:, m])

        # overlaps-save approach, no need to use hop_size
        frameNum = int((samples) / self.frameLen)

        for n in range(frameNum):
            x_n = x[n * self.frameLen : (n + 1) * self.frameLen, :]

            x_aligned = self.time_alignment.process(x_n)

            fixed_output = self.fixed_beamformer(x_aligned)

            bm_output = self.blocking_matrix(x_aligned)

            # AIC block
            output_n, _ = self.aic_filter.update(bm_output, fixed_output, fir_truncate=30)

            # output[n * self.frameLen : (n + 1) * self.frameLen] = fixed_output[:, 0]
            # output[n * self.frameLen : (n + 1) * self.frameLen] = np.squeeze(bm_output[:, 0])
            output[n * self.frameLen : (n + 1) * self.frameLen] = np.squeeze(output_n)

        return output

    def process(self, x, angle, method=2, retH=False, retWNG=False, retDI=False):
        """
        beamformer process function

        """
        X = self.transformer.stft(x.transpose())
        frameNum = X.shape[1]

        M = len(x[:, 1])

        outputlength = x.shape[1]

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

        alpha_y = 0.8
        alpha_v = 0.9998

        mu = 0.01
        rho = 0.998

        if (all(angle == self.angle) is False) or (method != self.AlgorithmIndex):
            # update look angle and algorithm
            if all(angle == self.angle) is False:
                self.angle = angle
                tao = -1 * self.r * np.cos(self.angle[1]) * np.cos(self.angle[0] - self.gamma) / self.c
            if method != self.AlgorithmIndex:
                self.AlgorithmIndex = method
            # reset flag
            self.frameCount = 0
            self.calc = 1

            # update blocking matrix
            for k in range(0, self.half_bin):
                a_k = np.mat(np.exp(-1j * self.omega[k] * tao)).T
                self.W[:, k, np.newaxis] = a_k / (a_k.conj().T @ a_k)
                for i in range(0, M - 1):
                    self.BM[0, i, k] = a_k[0]
                    self.BM[i + 1, i, k] = -1 * a_k[i + 1]
        Y = np.ones([self.half_bin, frameNum], dtype=complex)
        for t in range(0, frameNum):
            self.spp.estimation(X[:, t, :])
            # xt = x[:, t * self.hop:t * self.hop + self.frameLen] * self.window
            Z = X[:, t, :].transpose()
            # Z = np.fft.rfft(xt)  # *win_scale
            if (all(angle == self.angle) is False) or (method != self.AlgorithmIndex):
                # update look angle and algorithm
                if all(angle == self.angle) is False:
                    self.angle = angle
                    tao = -1 * self.r * np.cos(self.angle[1]) * np.cos(self.angle[0] - self.gamma) / self.c
                if method != self.AlgorithmIndex:
                    self.AlgorithmIndex = method
                # reset flag
                self.frameCount = 0
                self.calc = 1

            self.mcra.estimation(np.abs(Z[0, :] * np.conj(Z[0, :])))

            if self.AlgorithmIndex == 0:
                Y[:, t] = Z[0, :]  # output channel_1
            else:
                for k in range(0, self.half_bin):
                    a = np.mat(np.exp(-1j * self.omega[k] * tao)).T  # propagation vector

                    if self.mcra.p[k] > 0.5:
                        is_speech = 1
                    else:
                        is_speech = 0

                    # if self.frameCount == 200 and self.calc == 0:

                    if t > 312 and k == self.half_bin - 1:
                        # reset weights update flag
                        self.calc = 0
                    # print("calculating MVDR weights...\n")
                    Diagonal = 1e-6

                    # generate the reference noise signals
                    self.U[:, k] = np.squeeze(self.BM[:, :, k]).conj().T @ Z[:, k]
                    # fixed beamformer output
                    self.Yfbf[k] = self.W[:, k].conj().T @ Z[:, k]

                    # residual output
                    Y[k, t] = self.Yfbf[k] - self.G[:, k].conj().T @ self.U[:, k]

                    # use speech presence probability to control AIC update
                    self.Pest[k] = 1  # rho * self.Pest[k] + (1 - rho) * np.sum(np.power(np.abs(Z[:,k]),2))
                    # update MNC weights
                    self.G[:, k] = (
                        self.G[:, k] + mu * (1 - self.spp.p[k]) * self.U[:, k] * Y[k, t].conj() / self.Pest[k]
                    )

                    if retWNG:
                        WNG[k] = self.calcWNG(a, self.H[:, k, np.newaxis])
                    if retDI:
                        DI[k] = self.calcDI(a, self.H[:, k, np.newaxis], self.Fvv[k, :, :])

                self.omlsa_multi.estimation(
                    np.real(Y[:, t] * np.conj(Y[:, t])), np.real(self.U * np.conj(self.U)).transpose()
                )

                # post-filter
                Y[:, t] = Y[:, t] * self.spp.G

        yout = self.transformer.istft(Y)

        # calculate beampattern
        if retH:
            beampattern = self.beampattern(self.omega, self.H)

        return {'data': yout, 'WNG': WNG, 'DI': DI, 'beampattern': beampattern}
