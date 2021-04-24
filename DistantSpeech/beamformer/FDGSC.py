"""
overlap-save frequency domain GSC
"""
import argparse
import numpy as np
from scipy.signal import windows
from time import time

from DistantSpeech.noise_estimation.mcra import NoiseEstimationMCRA
from DistantSpeech.noise_estimation.mcspp_base import McSppBase
from DistantSpeech.noise_estimation.omlsa_multi import NsOmlsaMulti
from DistantSpeech.noise_estimation.mc_mcra import McMcra
from DistantSpeech.transform.transform import Transform
from DistantSpeech.beamformer.beamformer import beamformer
from DistantSpeech.beamformer.ArraySim import generate_audio
from DistantSpeech.beamformer.utils import load_audio as audioread
from DistantSpeech.beamformer.utils import save_audio as audiowrite
from DistantSpeech.beamformer.ArraySim import ArraySim
from DistantSpeech.adaptivefilter.FastFreqLms import FastFreqLms
from DistantSpeech.beamformer.MicArray import MicArray
from DistantSpeech.beamformer.utils import visual


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

        output = self.buffer[:, :data_len].copy()
        self.buffer[:, :self.n_delay] = self.buffer[:, -self.n_delay:]
        self.buffer[:, -data_len:] = x

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
            self.bm.append(FastFreqLms(filter_len=frameLen))

        self.aic_filter = FastFreqLms(filter_len=frameLen, n_channels=self.M - 1)

        self.delay_obj = DelayObj(self.frameLen, 8)

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
        bm_output = np.zeros((x.shape[1], self.M - 1))
        fix_output = np.zeros((x.shape[1],))

        # overlaps-save approach, no need to use hop_size
        frameNum = int((x.shape[1]) / self.frameLen)

        for n in range(frameNum):
            x_n = x[:, n * self.frameLen:(n + 1) * self.frameLen]

            # fixed beamformer path
            fixed_output = self.fixed_beamformer(x_n)

            # fix delay
            fixed_output = self.delay_obj.delay(fixed_output)

            # adaptive block matrix
            for m in range(self.M - 1):
                bm_output_n, _ = self.bm[0].update(x_n[m, :], x_n[m + 1, :])
                bm_output[n * self.frameLen:(n + 1) * self.frameLen, m] = np.squeeze(bm_output_n)

            # AIC block
            output_n, _ = self.aic_filter.update(bm_output[n * self.frameLen:(n + 1) * self.frameLen, :],
                                                 fixed_output.T)

            output[n * self.frameLen:(n + 1) * self.frameLen] = np.squeeze(bm_output_n)

        return output


def main(args):
    signal = audioread("../adaptivefilter/cleanspeech.wav")
    # fs = 16000
    # mic_array = ArraySim(array_type='linear', spacing=0.05)
    # array_data = mic_array.generate_audio(signal)
    # print(array_data.shape)
    # audiowrite('wav/array_data2.wav', array_data.transpose(), fs)

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

    x = MicArrayObj.array_sim.generate_audio(signal)
    print(x.shape)
    audiowrite('wav/x_cleanspeech.wav', x.transpose())

    start = time()

    fdgsc = FDGSC(MicArrayObj, frameLen, hop, nfft, c, fs)

    yout = fdgsc.process(x)

    print(yout.shape)

    audiowrite('wav/out_bm.wav', yout)

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
        output[n * buffer_len:(n + 1) * buffer_len] = delay_obj.delay(x[n * buffer_len:(n + 1) * buffer_len])

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
