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
from DistantSpeech.transform.multirate import frac_delay, fractional_delay_filter_bank


def fir_filter(x, fir_coeffs, fir_cache):
    """_summary_

    Parameters
    ----------
    x : np.array
        input multichannel data, [chs, samples]
    filter_coeffs : np.array
        fir filter coeffs, [chs, fir_filter_len]
    fir_cache : np.array
        fir filter cache, [chs, fir_filter_len-1]

    Returns
    -------
    _type_
        _description_
    """

    M, input_len = x.shape
    fir_filter_len = fir_coeffs.shape[1]

    fir_input = np.zeros((fir_filter_len - 1 + input_len, M))
    fir_input[: fir_filter_len - 1, :] = fir_cache[:]
    fir_input[fir_filter_len - 1 :, :] = x.T

    output = np.zeros(x.shape)

    fir_coeffs = np.fliplr(fir_coeffs)

    for m in range(M):
        for n in range(input_len):
            output[m, n] = fir_coeffs[m : m + 1, :] @ fir_input[n : n + fir_filter_len, m : m + 1]

    return output, x[:, -fir_filter_len + 1 :].T


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
    def __init__(
        self,
        mic_array: MicArray,
        frameLen=256,
        hop=None,
        nfft=None,
        fs=16000,
        angle=[197, 0],
    ):

        beamformer.__init__(self, mic_array, frame_len=frameLen, hop=hop, nfft=nfft, fs=fs)
        self.angle = np.array(angle) / 180 * np.pi if isinstance(angle, list) else angle

        # construct fraction delay filter for time-alignment in fixed beamformer
        self.tau = mic_array.compute_tau(self.angle)
        print('tau:{}'.format(self.tau))
        self.tau = -(self.tau - np.max(self.tau))
        delay_samples = np.array(self.tau)[:, 0] * mic_array.fs
        print('delay_samples:{}'.format(delay_samples))
        self.delay_filter = fractional_delay_filter_bank(delay_samples)
        self.delay_filter_len = self.delay_filter.shape[1]
        print('self.delay_filter:{}'.format(self.delay_filter.shape))
        self.fir_cache = np.zeros((self.delay_filter_len - 1, self.M))

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
            self.bm.append(FastFreqLms(filter_len=frameLen, mu=0.1, alpha=0.9, non_causal=True))

        self.aic_filter = FastFreqLms(filter_len=frameLen, n_channels=self.M, mu=0.1, alpha=0.9, non_causal=True)

        self.delay_fbf = DelaySamples(self.frameLen, int(frameLen / 2))

        self.delay_obj_bm = DelayObj(self.frameLen, 8, channel=self.M)

        self.spp = McSpp(nfft=frameLen * 2, channels=self.M)
        # self.spp = McSppBase(nfft=frameLen * 2, channels=self.M)
        self.transform = Transform(n_fft=frameLen * 2, hop_length=frameLen, channel=self.M)
        self.spp.mcra.L = 10

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

        output, self.fir_cache = fir_filter(x, self.delay_filter, self.fir_cache)

        return np.mean(output, axis=0, keepdims=True)
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

        D = self.transform.stft(np.transpose(x))

        for n in range(frameNum):
            x_n = x[:, n * self.frameLen : (n + 1) * self.frameLen]

            p[:, n] = self.spp.estimation(D[:, n, :])
            # p[:, n] = np.sqrt(p[:, n])

            # fixed beamformer path
            fixed_output = self.fixed_beamformer(x_n)

            # adaptive block matrix
            for m in range(self.M):
                bm_output_n, _ = self.bm[m].update(fixed_output.T, x_n[m, :], p=p[:, n : n + 1], fir_truncate=30)
                bm_output[n * self.frameLen : (n + 1) * self.frameLen, m] = np.squeeze(bm_output_n)

            # fix delay
            # fixed_output = self.delay_fbf.delay(fixed_output.T)
            fixed_output = fixed_output.T

            # AIC block
            output_n, _ = self.aic_filter.update(
                bm_output[n * self.frameLen : (n + 1) * self.frameLen, :],
                fixed_output,
                p=1 - p[:, n : n + 1],
                fir_truncate=30,
            )

            # output[n * self.frameLen : (n + 1) * self.frameLen] = fixed_output[0, :]
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
    r = 0.032
    fs = 16000

    # start = tim

    MicArrayObj = MicArray(arrayType='linear', r=0.05, M=3)
    angle = np.array([197, 0]) / 180 * np.pi

    x = MicArrayObj.array_sim.generate_audio(target, interference=interf, snr=0)
    print(x.shape)
    audiowrite('wav/target_90_interf_30.wav', x.transpose())

    start = time()

    fdgsc = FDGSC(MicArrayObj, frameLen, hop, nfft, fs)

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
