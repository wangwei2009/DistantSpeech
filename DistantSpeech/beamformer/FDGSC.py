"""
overlap-save frequency domain GSC
==================

----------

..    [1].Efficient frequency-domain realization of robust generalized, sidelobe cancellers.

"""
import argparse
from time import time
from turtle import update

import numpy as np
from scipy.signal import windows

from DistantSpeech.adaptivefilter.FastFreqLms import FastFreqLms
from DistantSpeech.beamformer.MicArray import MicArray
from DistantSpeech.beamformer.beamformer import beamformer
from DistantSpeech.beamformer.utils import load_audio as audioread
from DistantSpeech.beamformer.utils import save_audio as audiowrite
from DistantSpeech.beamformer.utils import visual, DelaySamples
from DistantSpeech.noise_estimation.mcspp_base import McSppBase
from DistantSpeech.transform.transform import Transform
from DistantSpeech.noise_estimation import McSpp
from DistantSpeech.adaptivefilter.feature import Emphasis, FilterDcNotch16
from DistantSpeech.beamformer.fixedbeamformer import TimeAlignment
from DistantSpeech.noise_estimation.mcra import NoiseEstimationMCRA
from DistantSpeech.noise_estimation.omlsa_multi import NsOmlsaMulti
from DistantSpeech.beamformer.ccafbounds import ccafbounds


from DistantSpeech.beamformer.gsc_bm import AdaptiveBlockingMatrixFilter as bm_filter
from DistantSpeech.beamformer.gsc_aic import AdaptiveInterferenceCancellation as aic_filter


class FDGSC(beamformer):
    def __init__(
        self,
        mic_array: MicArray,
        frameLen=256,
        angle=[197, 0],
    ):

        beamformer.__init__(
            self,
            mic_array,
            frame_len=frameLen,
        )
        # overlap-save fft
        self.nfft = frameLen * 2

        self.angle = np.array(angle) / 180 * np.pi if isinstance(angle, list) else angle

        self.time_alignment = TimeAlignment(mic_array, angle=self.angle)

        self.gamma = mic_array.gamma
        self.window = windows.hann(self.frameLen, sym=False)
        self.win_scale = np.sqrt(1.0 / self.window.sum() ** 2)
        self.freq_bin = np.linspace(0, self.half_bin - 1, self.half_bin)
        self.omega = 2 * np.pi * self.freq_bin * self.fs / self.nfft

        self.H = np.ones([self.M, self.half_bin], dtype=complex)

        self.AlgorithmList = ['src', 'DS', 'MVDR']
        self.AlgorithmIndex = 0

        self.transformer = Transform(n_fft=self.nfft, hop_length=self.hop, channel=self.M)

        self.bm = []
        for m in range(self.M):
            self.bm.append(
                bm_filter(
                    filter_len=frameLen,
                    mu=0.1,
                    alpha=0.9,
                    non_causal=False,
                    constrain=True,
                    weight_norm=True,
                )
            )

        self.aic_filter = aic_filter(
            filter_len=frameLen,
            n_channels=self.M,
            mu=0.1,
            alpha=0.9,
            non_causal=False,
            constrain=True,
            weight_norm=True,
        )

        self.delay_fbf = DelaySamples(self.frameLen, frameLen)

        self.delay_x = DelaySamples(self.frameLen, int(self.frameLen / 2), channel=self.M)
        self.delay_aligned = DelaySamples(self.frameLen, int(self.frameLen / 2), channel=self.M)

        self.spp = McSppBase(nfft=frameLen * 2, channels=self.M)
        self.spp = NoiseEstimationMCRA(nfft=frameLen * 2)
        self.spp.L = 60

        self.spp_fbf = NoiseEstimationMCRA(nfft=frameLen * 2)
        self.spp_fbf.L = 60

        self.transform = Transform(n_fft=frameLen * 2, hop_length=frameLen, channel=1)
        self.transform_x = Transform(n_fft=frameLen * 2, hop_length=frameLen, channel=self.M)

        self.omlsa_multi = NsOmlsaMulti(nfft=frameLen * 2, cal_weights=True, M=self.M)
        self.transform_fbf = Transform(n_fft=frameLen * 2, hop_length=frameLen, channel=1)
        self.transform_bm = Transform(n_fft=frameLen * 2, hop_length=frameLen, channel=self.M - 1)

        self.phi, self.psi = ccafbounds(self.MicArray.mic_loc.T, p=129, order=frameLen)

        self.dc_notch_mic = []
        for _ in range(self.M):
            self.dc_notch_mic.append(FilterDcNotch16(radius=0.98))

    def adaption_control(self, fbf, time_alignment):
        pass

    def fixed_delay(self):
        pass

    def fixed_beamformer(self, x):
        """fixed beamformer

        Parameters
        ----------
        x : np.array
            input multichannel data, [samples, chs]

        Returns
        -------
        np.array
            output data, [samples, ]
        """

        return np.mean(x, axis=1, keepdims=True)
        # return output[0:1, :]

    def blocking_matrix(self, x, fixed_output=None, p=1.0, mode=1):
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

        # fixed blocking matrix
        if mode == 0:
            bm_output = np.zeros((samples, channels - 1))
            for m in range(channels - 1):
                bm_output[:, m] = x[:, m] - x[:, m + 1]

        # fixed blocking matrix, use fbf as input signal
        if mode == 1:
            bm_output = np.zeros((samples, channels))
            for m in range(channels):
                bm_output[:, m] = fixed_output - x[:, m]

        # adaptive blocking matrix, estimate RTF
        if mode == 2:
            bm_output = np.zeros((samples, channels - 1))
            for m in range(1, channels):
                bm_output[:, m : m + 1], w = self.bm[m].update(
                    x[:, m - 1],
                    x[:, m],
                    p=p,
                    filter_p=True,
                    update=True,
                    # fir_truncate=10,
                )
        # adaptive blocking matrix, use fbf as input signal
        if mode == 3:
            bm_output = np.zeros((samples, channels))
            for m in range(channels):
                bm_output[:, m : m + 1], w = self.bm[m].update(
                    fixed_output,
                    x[:, m],
                    p=p,
                    filter_p=True,
                    update=True,
                    # fir_truncate=10,
                )

        return bm_output

    def aic(self, z):
        """
        :param z:
        :return:
        """
        pass

    def process(self, x, postfilter=False, dc_notch=True):
        """
        process core function
        :param x: input time signal, (n_samples, n_chs)
        :return: enhanced signal, (n_samples,)
        """

        n_samples, channel = x.shape
        output = np.zeros(n_samples)

        if dc_notch:
            for m in range(self.M):
                x[:, m], self.dc_notch_mic[m].mem = self.dc_notch_mic[m].filter_dc_notch16(x[:, m])

        # adaptive block matrix path
        bm_output = np.zeros((n_samples, self.M))
        aligned_output = np.zeros((n_samples, self.M))
        aligned_output_delayed = np.zeros((n_samples, self.M))
        fix_output = np.zeros((n_samples,))
        fix_output_delayed = np.zeros((n_samples,))

        # overlaps-save approach, no need to use hop_size
        frameNum = int((n_samples) / self.frameLen)

        p = np.zeros((self.spp.half_bin, frameNum))
        G = np.zeros((self.spp.half_bin, frameNum))

        t = 0

        for n in range(frameNum):
            x_n = x[n * self.frameLen : (n + 1) * self.frameLen, :]

            x_aligned = self.time_alignment.process(x_n)

            # fixed beamformer path
            fixed_output = self.fixed_beamformer(x_aligned)

            # D = self.transform.stft(fixed_output[:, 0])
            D = self.transform_x.stft(x_n)

            self.spp.estimation(D[:, 0, :])
            p[:, n] = self.spp.p
            # p[:, n] = np.sqrt(p[:, n])
            # p[:, n] = p[:, n] ** 2

            p_bm = p[:, n : n + 1].copy()  # * np.mean(p[16:128, n : n + 1], axis=0, keepdims=True)
            p1 = p[:32]

            if np.mean(p_bm[32:128]) > 0.8:
                # p1[:] = 0.999
                p1[p1[:, n] < 0.8, n] = 0.8
            else:
                p_bm[:] = 1e-3
            # adaptive block matrix
            x_n = self.delay_x.delay(x_n)
            x_aligned_delayed_n = self.delay_aligned.delay(x_aligned)
            bm_output_n = self.blocking_matrix(
                x_aligned_delayed_n,
                fixed_output,
                # p=p[:, n : n + 1],
                mode=3,
            )
            bm_output[n * self.frameLen : (n + 1) * self.frameLen, :] = np.squeeze(bm_output_n)

            # # # fix delay
            # fixed_output = self.fbf[n * self.frameLen : (n + 1) * self.frameLen, np.newaxis]
            fixed_output_delayed_n = self.delay_fbf.delay(fixed_output)
            # fixed_output = fixed_output.T

            Y = self.transform_fbf.stft(fixed_output_delayed_n)
            self.spp_fbf.estimation(Y[:, 0, :])
            p_fbf = self.spp.p

            # AIC block
            output_n, _ = self.aic_filter.update(
                bm_output[n * self.frameLen : (n + 1) * self.frameLen, :],
                fixed_output_delayed_n,
                # p=1 - p_fbf[:, np.newaxis],  # p[:, n : n + 1],
                p=1 - np.mean(p[:, n : n + 1]),
                # fir_truncate=10,
            )

            if postfilter:
                Y = self.transform_fbf.stft(output_n)
                U = self.transform_bm.stft(bm_output[:, :-1])

                self.omlsa_multi.estimation(
                    np.real(Y[:, 0, 0] * np.conj(Y[:, 0, 0])), np.real(U[:, 0, :] * np.conj(U[:, 0, :]))
                )

                # post-filter
                G[:, t] = np.sqrt(self.omlsa_multi.G)
                t += 1
                Y[:, 0, 0] = Y[:, 0, 0] * np.sqrt(self.omlsa_multi.G)
                output_n = self.transform_fbf.istft(Y)

            # output[n * self.frameLen : (n + 1) * self.frameLen] = bm_d[0, :]
            fix_output[n * self.frameLen : (n + 1) * self.frameLen] = fixed_output[:, 0]
            fix_output_delayed[n * self.frameLen : (n + 1) * self.frameLen] = fixed_output_delayed_n[:, 0]
            aligned_output[n * self.frameLen : (n + 1) * self.frameLen, :] = x_aligned[:, :]
            aligned_output_delayed[n * self.frameLen : (n + 1) * self.frameLen, :] = x_aligned_delayed_n[:, :]
            output[n * self.frameLen : (n + 1) * self.frameLen] = np.squeeze(output_n)
            bm_output[n * self.frameLen : (n + 1) * self.frameLen, 0] = bm_output[
                n * self.frameLen : (n + 1) * self.frameLen, 0
            ]

        return (
            output,
            p,
            fix_output,
            fix_output_delayed,
            bm_output,
            aligned_output,
            aligned_output_delayed,
            self.bm,
            self.aic_filter,
        )


def main(args):

    frameLen = 1024
    hop = frameLen / 2
    overlap = frameLen - hop
    nfft = 256
    r = 0.032
    fs = 16000
    M = 4

    # start = tim

    MicArrayObj = MicArray(arrayType='linear', r=0.05, M=M)
    angle = np.array([197, 0]) / 180 * np.pi

    x = np.random.randn(M, 16000 * 5)
    start = time()

    fdgsc = FDGSC(MicArrayObj, frameLen, angle)

    yout, _ = fdgsc.process(x)

    print(yout.shape)

    # audiowrite('wav/out_aic.wav', yout)

    end = time()
    print(end - start)

    visual(x[0, :], yout)

    return


def test_delay():
    from matplotlib import pyplot as plt

    t = np.arange(8000) / 1000.0
    f = 1000
    fs = 1000
    x = np.sin(2 * np.pi * f / fs * t)

    delay = 128
    buffer_len = 512
    delay_fbf = DelayObj(buffer_len, delay)

    n_frame = int(len(x) / buffer_len)

    output = np.zeros(len(x))

    for n in range(n_frame):
        output[n * buffer_len : (n + 1) * buffer_len] = delay_fbf.delay(x[n * buffer_len : (n + 1) * buffer_len])

    plt.figure()
    plt.plot(x)
    plt.plot(np.squeeze(output))
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Feature Extractor")
    parser.add_argument("-l", "--listen", action='store_true', help="set to listen output")  # if set true
    parser.add_argument("-s", "--save", action='store_true', help="set to save output")  # if set true

    args = parser.parse_args()
    main(args)
    # test_delay()
