"""
Subband GSC beamformer
refer to
    Robust Adaptive Beamforming Algorithm using Instantaneous Direction of Arrival 
    with Enhanced Noise Suppression Capability. Acoustics, Speech, and Signal Processing, 1988. ICASSP-88., 1988 International Conference on. 1. I-133 . 10.1109/ICASSP.2007.366634. 
Author:
    Wang Wei
"""
import argparse
from time import time

import numpy as np
from scipy.signal import windows

from DistantSpeech.adaptivefilter.FastFreqLms import FastFreqLms
from DistantSpeech.beamformer.MicArray import MicArray
from DistantSpeech.beamformer.beamformer import beamformer
from DistantSpeech.beamformer.utils import load_audio as audioread
from DistantSpeech.beamformer.utils import save_audio as audiowrite
from DistantSpeech.beamformer.utils import visual
from DistantSpeech.transform.transform import Transform
from DistantSpeech.noise_estimation import McSpp
from DistantSpeech.beamformer.FDGSC import FDGSC, DelayObj

import librosa
import matplotlib.pyplot as plt
from DistantSpeech.transform.transform import Transform
from DistantSpeech.beamformer.utils import pmesh, mesh, load_wav, save_audio, load_pcm, pt
from DistantSpeech.beamformer.utils import load_audio as audioread
from DistantSpeech.beamformer.utils import save_audio as audiowrite
from DistantSpeech.beamformer.beamformer import beamformer
from DistantSpeech.beamformer.MicArray import MicArray, compute_tau
from DistantSpeech.noise_estimation import McSpp, McSppBase
from DistantSpeech.transform.subband import Subband
from DistantSpeech.adaptivefilter.SubbandLMS import SubbandLMS
from DistantSpeech.beamformer.utils import DelaySamples, DelayFrames
from DistantSpeech.beamformer.fixedbeamformer import TimeAlignment
from DistantSpeech.adaptivefilter.feature import Emphasis, FilterDcNotch16
from DistantSpeech.adaptivefilter.SubbandLmsMc import SubbandLmsMc
from DistantSpeech.noise_estimation.omlsa_multi import NsOmlsaMulti


class DelayObj(object):
    def __init__(self, buffer_size, delay, channel=1):
        self.buffer_szie = buffer_size
        self.n_delay = delay

        self.buffer = np.zeros((channel, buffer_size + delay))

    def delay(self, x):
        """
        delay x for self.delay point
        :param x: (n_samples,) or (n_chs, n_samples)
        :return:
        """
        if len(x.shape) == 1:
            x = x[np.newaxis, :]
        data_len = x.shape[1]

        self.buffer[:, -data_len:] = x
        output = self.buffer[:, :data_len].copy()
        self.buffer[:, : self.n_delay] = self.buffer[:, -self.n_delay :]

        return output


class SubbandGSC(beamformer):
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
            self.bm.append(SubbandLMS(filter_len=2, num_bands=frameLen * 2, mu=1e-1))

        self.aic_filter = SubbandLmsMc(
            filter_len=2,
            num_bands=frameLen * 2,
            channel=self.M,
            mu=0.01,
            alpha=0.8,
        )

        self.delay_fbf = DelaySamples(self.frameLen, int(frameLen))

        self.delay_obj_bm = DelayObj(self.frameLen, 40, channel=self.M)

        self.spp = McSpp(nfft=frameLen * 2, channels=self.M)
        # self.spp = McSppBase(nfft=frameLen * 2, channels=self.M)
        self.transform = Transform(n_fft=frameLen * 2, hop_length=frameLen, channel=self.M)
        self.spp.mcra.L = 10

        self.transform_fixed = Transform(n_fft=frameLen * 2, hop_length=frameLen, channel=1)

        self.dc_notch_mic = []
        for _ in range(self.M):
            self.dc_notch_mic.append(FilterDcNotch16(radius=0.98))

        self.bm_output_n = np.zeros((self.frameLen, self.M))
        self.W_aic = np.zeros((self.bm[0].half_band, 2, self.M), dtype=complex)

        self.omlsa_multi = NsOmlsaMulti(nfft=frameLen * 2, cal_weights=True, M=self.M)
        self.transform_fbf = Transform(n_fft=frameLen * 2, hop_length=frameLen, channel=1)
        self.transform_bm = Transform(n_fft=frameLen * 2, hop_length=frameLen, channel=self.M)

    def fixed_delay(self):
        pass

    def fixed_beamformer(self, x):
        """fixed beamformer

        Parameters
        ----------
        x : np.array
            input multichannel data, [chs, samples]

        Returns
        -------
        np.array
            output data, [1, samples]
        """

        return np.mean(x, axis=1, keepdims=True)
        # return output[0:1, :]

    def blocking_matrix(self, x):
        """

        :param x: (n_chs, frame_len)
        :return: bm output, (n_chs-1, frame_len)
        """
        output = np.zeros((self.M - 1, x.shape[1]))
        for m in range(self.M - 1):
            output[m, :] = self.bm[m].update(x[m, :], x[m + 1, :])

        return output

    def aic(self, z):
        """
        :param z:
        :return:
        """
        pass

    def process(self, x, postfilter=False):
        """
        process core function
        :param x: input time signal, (n_chs, n_samples)
        :return: enhanced signal, (n_samples,)
        """

        for m in range(self.M):
            x[m, :], self.dc_notch_mic[m].mem = self.dc_notch_mic[m].filter_dc_notch16(x[m, :])

        output = np.zeros(x.shape[1])

        # for m in range(self.M):
        #     x[m, :], self.dc_notch_mic[m].mem = self.dc_notch_mic[m].filter_dc_notch16(x[m, :])

        # adaptive block matrix path
        bm_output = np.zeros((x.shape[1], self.M))
        aligned_output = np.zeros((x.shape[1], self.M))
        fix_output = np.zeros((x.shape[1],))
        aic_output = np.zeros((x.shape[1],))

        # overlaps-save approach, no need to use hop_size
        frameNum = int((x.shape[1]) / self.frameLen)

        p = np.zeros((self.spp.half_bin, frameNum))
        G = np.zeros((self.spp.half_bin, frameNum))

        t = 0
        for n in range(frameNum):
            x_n = x[:, n * self.frameLen : (n + 1) * self.frameLen]

            x_aligned = self.time_alignment.process(x_n.T)
            aligned_output[n * self.frameLen : (n + 1) * self.frameLen, :] = x_aligned

            D = self.transform.stft(x_aligned)

            fixed_output = self.fixed_beamformer(x_aligned)

            p[:, n] = self.spp.estimation(D[:, 0, :])
            # p[:, n] = np.sqrt(p[:, n])
            # p[:, n] = p[:, n] ** 2

            # # fixed beamformer path
            # fixed_output = self.fixed_beamformer(x_n)

            # adaptive block matrix
            # x_n = self.delay_obj_bm.delay(x_n)
            for m in range(self.M):
                self.bm_output_n[:, m], _ = self.bm[m].update(
                    fixed_output[:, 0],
                    x_aligned[:, m],
                    p=p[:, n : n + 1],
                )
                bm_output[n * self.frameLen : (n + 1) * self.frameLen, m] = np.squeeze(self.bm_output_n[:, m])

            # # # # fix delay
            fixed_output = self.delay_fbf.delay(fixed_output)
            # fixed_output = fixed_output.T

            # AIC block
            output_n, _ = self.aic_filter.update(
                bm_output[n * self.frameLen : (n + 1) * self.frameLen, :],
                fixed_output,
                p=1 - p[:, n : n + 1],
            )

            if postfilter:
                Y = self.transform_fbf.stft(output_n)
                U = self.transform_bm.stft(bm_output)

                self.omlsa_multi.estimation(
                    np.real(Y[:, 0, 0] * np.conj(Y[:, 0, 0])), np.real(U[:, 0, :] * np.conj(U[:, 0, :]))
                )

                # post-filter
                # G[:, t] = np.sqrt(self.omlsa_multi.G)
                G[:, t] = np.sqrt(self.omlsa_multi.xi_hat)
                t += 1
                Y[:, 0, 0] = Y[:, 0, 0] * np.sqrt(self.omlsa_multi.G)
                # output_n = self.transform_fbf.istft(Y)

            # output[n * self.frameLen : (n + 1) * self.frameLen] = fixed_output[:, 0]
            # output[n * self.frameLen : (n + 1) * self.frameLen] = np.squeeze(bm_output[:, 0])
            output[n * self.frameLen : (n + 1) * self.frameLen] = np.squeeze(output_n)

            # output[n * self.frameLen : (n + 1) * self.frameLen] = bm_d[0, :]
            fix_output[n * self.frameLen : (n + 1) * self.frameLen] = fixed_output[:, 0]
            output[n * self.frameLen : (n + 1) * self.frameLen] = np.squeeze(output_n)
            # output[n * self.frameLen : (n + 1) * self.frameLen] = bm_output[
            #     n * self.frameLen : (n + 1) * self.frameLen, 0
            # ]

        return output, fix_output, bm_output, p, aligned_output


def main(args):

    frameLen = 512
    hop = frameLen / 2
    overlap = frameLen - hop
    nfft = 256
    r = 0.032
    fs = 16000
    M = 6

    # start = tim

    MicArrayObj = MicArray(arrayType='linear', r=0.05, M=M)
    angle = np.array([197, 0]) / 180 * np.pi

    x = np.random.randn(M, 16000 * 5)
    start = time()

    gsc = SubbandGSC(MicArrayObj, frameLen, angle)

    yout, _ = gsc.process(x)

    print(yout.shape)

    # audiowrite('wav/out_aic.wav', yout)

    end = time()
    print(end - start)

    visual(x[0, :], yout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Feature Extractor")
    parser.add_argument("-l", "--listen", action='store_true', help="set to listen output")  # if set true
    parser.add_argument("-s", "--save", action='store_true', help="set to save output")  # if set true

    args = parser.parse_args()
    main(args)
    # test_delay()
