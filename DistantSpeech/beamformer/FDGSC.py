"""
overlap-save frequency domain GSC
refer to
    Efficient frequency-domain realization of robust generalized, sidelobe cancellers
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
from DistantSpeech.beamformer.utils import visual, DelaySamples
from DistantSpeech.noise_estimation.mcspp_base import McSppBase
from DistantSpeech.transform.transform import Transform
from DistantSpeech.noise_estimation import McSpp


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


class FDGSC(beamformer):
    def __init__(self, MicArray, frameLen=256, hop=None, nfft=None, channels=4, c=343, r=0.032, fs=16000):

        beamformer.__init__(self, MicArray, frame_len=frameLen, hop=hop, nfft=nfft, c=c, fs=fs)
        self.angle = np.array([197, 0]) / 180 * np.pi
        self.gamma = MicArray.gamma
        self.window = windows.hann(self.frameLen, sym=False)
        self.win_scale = np.sqrt(1.0 / self.window.sum() ** 2)
        self.freq_bin = np.linspace(0, self.half_bin - 1, self.half_bin)
        self.omega = 2 * np.pi * self.freq_bin * self.fs / self.nfft

        self.H = np.ones([self.M, self.half_bin], dtype=complex)

        self.angle = np.array([0, 0]) / 180 * np.pi

        self.AlgorithmList = ['src', 'DS', 'MVDR']
        self.AlgorithmIndex = 0

        self.transformer = Transform(n_fft=self.nfft, hop_length=self.hop, channel=self.M)

        self.bm = []
        for m in range(self.M):
            self.bm.append(FastFreqLms(filter_len=frameLen, mu=0.1, alpha=0.9, non_causal=True))

        self.aic_filter = FastFreqLms(filter_len=frameLen, n_channels=self.M, mu=0.1, alpha=0.9, non_causal=True)

        self.delay_fbf = DelaySamples(self.frameLen, int(frameLen / 2))

        self.delay_obj_bm = DelayObj(self.frameLen, 8, channel=self.M)

        self.spp = McSpp(nfft=frameLen * 2, channels=2)
        # self.spp = McSppBase(nfft=frameLen * 2, channels=self.M)
        self.transform = Transform(n_fft=frameLen * 2, hop_length=frameLen, channel=2)
        self.spp.mcra.L = 10

    def fixed_delay(self):
        pass

    def fixed_beamformer(self, x):
        """

        :param x: input signal, (n_chs, frame_len)
        :return:
        """
        return np.mean(x, axis=0, keepdims=True)

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

    def process(self, x):
        """
        process core function
        :param x: input time signal, (n_chs, n_samples)
        :return: enhanced signal, (n_samples,)
        """

        output = np.zeros(x.shape[1])

        # adaptive block matrix path
        bm_output = np.zeros((x.shape[1], self.M))
        fix_output = np.zeros((x.shape[1],))

        # overlaps-save approach, no need to use hop_size
        frameNum = int((x.shape[1]) / self.frameLen)

        p = np.zeros((self.spp.half_bin, frameNum))

        D = self.transform.stft(np.transpose(x[[0, -1], :]))

        for n in range(frameNum):
            x_n = x[:, n * self.frameLen : (n + 1) * self.frameLen]

            p[:, n] = self.spp.estimation(D[:, n, :])
            p[:, n] = np.sqrt(p[:, n])

            # fixed beamformer path
            fixed_output = self.fixed_beamformer(x_n)

            # adaptive block matrix
            for m in range(self.M):
                bm_output_n, _ = self.bm[m].update(fixed_output.T, x_n[m, :], p=p[:, n : n + 1], fir_truncate=30)
                bm_output[n * self.frameLen : (n + 1) * self.frameLen, m] = np.squeeze(bm_output_n)

            # fix delay
            fixed_output = self.delay_fbf.delay(fixed_output.T)

            print('fixed_output:{}'.format(fixed_output.shape))
            # AIC block
            output_n, _ = self.aic_filter.update(
                bm_output[n * self.frameLen : (n + 1) * self.frameLen, :],
                fixed_output,
                p=1 - p[:, n : n + 1],
                fir_truncate=30,
            )

            output[n * self.frameLen : (n + 1) * self.frameLen] = np.squeeze(output_n)
            # output[n * self.frameLen : (n + 1) * self.frameLen] = np.squeeze(
            #     bm_output[n * self.frameLen : (n + 1) * self.frameLen, 0]
            # )

        return output, p


def main(args):
    target = audioread("wav/target.wav")
    interf = audioread("wav/interf.wav")

    frameLen = 1024
    hop = frameLen / 2
    overlap = frameLen - hop
    nfft = 256
    c = 340
    r = 0.032
    fs = 16000

    # start = tim

    MicArrayObj = MicArray(arrayType='linear', r=0.05, M=3)
    angle = np.array([197, 0]) / 180 * np.pi

    x = MicArrayObj.array_sim.generate_audio(target, interference=interf, snr=0)
    print(x.shape)
    audiowrite('wav/target_90_interf_30.wav', x.transpose())

    start = time()

    fdgsc = FDGSC(MicArrayObj, frameLen, hop, nfft, c, fs)

    yout = fdgsc.process(x)

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
