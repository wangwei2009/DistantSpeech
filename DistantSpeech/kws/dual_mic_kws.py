"""
dual mic keyword spotting
refer to
    Hotword Cleaner: Dual-Microphone Adaptive Noise Cancellation With Deferred Filter Coefficients for Robust Keyword Spotting
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
from DistantSpeech.beamformer.utils import load_wav
from DistantSpeech.beamformer.utils import visual
from DistantSpeech.transform.transform import Transform


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


class DelayBuffer(object):
    """delay a vector for delay frame"""

    def __init__(self, data_len, delay):
        self.data_len = data_len
        self.n_delay = delay + 1

        self.buffer = np.zeros((self.n_delay, data_len))

    def delay(self, x_vec):
        """
        delay x for self.delay point
        :param x: (n_samples,)
        :return:
        """
        x_vec = np.squeeze(x_vec)

        assert len(x_vec) == self.data_len

        self.buffer[-1, :] = x_vec
        output = self.buffer[0, :].copy()
        self.buffer[:-1, :] = self.buffer[1:, :]

        return output


class DualMicKws(beamformer):
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

        self.anc = FastFreqLms(filter_len=frameLen, mu=0.1, alpha=0.1)
        self.cleaner_filter = FastFreqLms(filter_len=frameLen, mu=0.1, alpha=0.1)

        self.delay_obj = DelayBuffer(self.frameLen, 18)

        self.delay_obj_bm = DelayObj(self.frameLen, 8, channel=self.M)

    def fixed_delay(self):
        pass

    def fixed_beamformer(self, x):
        """

        :param x: input signal, (n_chs, frame_len)
        :return:
        """
        return np.mean(x, axis=0, keepdims=True)

    def bm(self, x):
        """

        :param x: (n_chs, frame_len)
        :return: bm output, (n_chs-1, frame_len)
        """
        output = np.zeros((self.M - 1, x.shape[1]))
        for m in range(self.M - 1):
            output[m, :] = self.anc[m].update(x[m, :], x[m + 1, :])

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

        cleaner_output = np.zeros(x.shape[1])

        # adaptive block matrix path
        bm_output = np.zeros((x.shape[1], self.M))
        fix_output = np.zeros((x.shape[1],))

        # overlaps-save approach, no need to use hop_size
        frameNum = int((x.shape[1]) / self.frameLen)

        for n in range(frameNum):
            x_n = x[:, n * self.frameLen : (n + 1) * self.frameLen]

            bm_update = True
            aic_update = False

            # fixed beamformer path
            fixed_output = self.fixed_beamformer(x_n)

            # lower_path_delayed = self.delay_obj_bm.delay(x_n)

            bm_output_n, weights = self.anc.update(x_n[0], x_n[1], update=bm_update)
            weights_delayed = self.delay_obj.delay(weights)
            self.cleaner_filter.set_weights(weights_delayed)
            cleaner_output_n, _ = self.cleaner_filter.update(x_n[0], x_n[1], update=False)

            bm_output[n * self.frameLen : (n + 1) * self.frameLen, 0] = np.squeeze(cleaner_output_n)
            # bm_output[n * self.frameLen : (n + 1) * self.frameLen, 0] = np.squeeze(bm_output_n)

            output[n * self.frameLen : (n + 1) * self.frameLen] = np.squeeze(
                bm_output[n * self.frameLen : (n + 1) * self.frameLen, 0]
            )

        return output


def main(args):
    target = audioread("samples/audio_samples/cleanspeech_aishell3.wav")

    frameLen = 1024
    hop = frameLen / 2
    overlap = frameLen - hop
    nfft = 256
    c = 340
    r = 0.032
    fs = 16000

    # start = tim

    MicArrayObj = MicArray(arrayType='linear', r=0.04, M=2)
    angle = np.array([197, 0]) / 180 * np.pi

    if args.real:
        x, _ = load_wav('samples/kws_samples/diy/3')
    else:
        x = MicArrayObj.array_sim.generate_audio(target, snr=60)
        print(x.shape)
        kws_data = audioread('samples/kws_samples/himia.wav')

        MicArrayKwsObj = MicArray(arrayType='linear', r=0.04, M=2)
        kws_data_reverb = MicArrayKwsObj.array_sim.generate_audio(kws_data, source_angle=0, snr=60)

        print(kws_data_reverb.shape)
        start_pos = 120000
        for m in range(x.shape[0]):
            x[m, start_pos : start_pos + kws_data_reverb.shape[1]] += kws_data_reverb[m, :]
        audiowrite('DistantSpeech/kws/wav/target_90_interf_30.wav', x.transpose())

    print(x.shape)
    start = time()

    fdgsc = DualMicKws(MicArrayObj, frameLen, hop, nfft, c, fs)

    yout = fdgsc.process(x)

    print(yout.shape)

    audiowrite('DistantSpeech/kws/wav/out_bm_diy.wav', yout)

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
    delay_obj = DelayObj(buffer_len, delay)

    n_frame = int(len(x) / buffer_len)

    output = np.zeros(len(x))

    for n in range(n_frame):
        output[n * buffer_len : (n + 1) * buffer_len] = delay_obj.delay(x[n * buffer_len : (n + 1) * buffer_len])

    plt.figure()
    plt.plot(x)
    plt.plot(np.squeeze(output))
    plt.show()


def test_delay_buffer():
    from matplotlib import pyplot as plt

    t = np.arange(8000) / 1000.0
    f = 1000
    fs = 1000
    x = np.sin(2 * np.pi * f / fs * t)

    delay = 0
    buffer_len = 3
    x_vec = np.array([1, 2, 3])
    delay_obj = DelayBuffer(buffer_len, delay)

    x_vec_delayed = delay_obj.delay(x_vec)
    print(x_vec_delayed)
    x_vec_delayed = delay_obj.delay(x_vec)
    print(x_vec_delayed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Feature Extractor")
    parser.add_argument("-l", "--listen", action='store_true', help="set to listen output")  # if set true
    parser.add_argument("-s", "--save", action='store_true', help="set to save output")  # if set true

    parser.add_argument("--real", action='store_true', help="set to use real recordings")  # if set true

    args = parser.parse_args()
    main(args)
    # test_delay()
    # test_delay_buffer()
