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


class DelayObj(object):
    def __init__(self, buffer_size, delay, channel=1):
        self.buffer_szie = buffer_size
        self.delay = delay

        self.buffer = np.zeros((buffer_size + delay, channel))

    def delay(self, x):
        """
        delay x for self.delay point
        :param x: (n_samples,) or (n_samples, n_chs)
        :return:
        """
        if x.dim == 1:
            x = x[:, np.newaxis]
        data_len = x.shape[0]
        self.buffer[:data_len, :] = self.buffer[-data_len:, :]
        self.buffer[-data_len:, :] = x

        return self.buffer[:data_len, :]


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

        self.delay_obj = DelayObj(frameLen, 5)

    def fixed_delay(self):
        pass

    def fixed_beamformer(self, x):
        """
        
        :param x: input signal, (n_chs, frame_len)
        :return: 
        """
        return np.mean(x, axis=0)

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

        # adaptive block matrix path
        output = np.zeros((self.M - 1, x.shape[1]))
        frameNum = int((x.shape[1] - self.overlap) / self.hop)

        for n in range(frameNum):
            x_n = x[:, n * self.hop:n * self.hop + self.frameLen]

            # fixed beamformer path
            fixed_output = self.fixed_beamformer(x)

            # fix delay
            fixed_output = self.delay_obj.delay(fixed_output)

            # adaptive block matrix
            for m in range(self.M - 1):
                output_n, _ = self.bm[0].update(x_n[m, :], x_n[m+1, :])
                output[m, n * self.hop:n * self.hop + self.frameLen] = np.squeeze(output_n)


        return output

        pass

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
                self.H[:, k, np.newaxis] = self.getweights(a, self.AlgorithmList[method], self.Fvv[k, :, :],
                                                           Diagonal=1e-1)

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

        return {'data': yout,
                'WNG': WNG,
                'DI': DI,
                'beampattern': beampattern
                }
        pass

    #     self.M = channels
    #     self.angle = np.array([197, 0]) / 180 * np.pi
    #     self.gamma = MicArray.gamma
    #     self.window = windows.hann(self.frameLen, sym=False)
    #     self.win_scale = np.sqrt(1.0 / self.window.sum() ** 2)
    #     self.freq_bin = np.linspace(0, self.half_bin - 1, self.half_bin)
    #     self.omega = 2 * np.pi * self.freq_bin * self.fs / self.nfft
    #
    #     self.window = np.sqrt(windows.hann(self.frameLen, sym=False))
    #
    #     self.transformer = Transform(n_fft=self.nfft, hop_length=self.hop, channel=self.M)
    #
    #     self.pad_data = np.zeros([MicArray.M, round(frameLen / 2)])
    #     self.last_output = np.zeros(round(frameLen / 2))
    #
    #     self.H = np.ones([self.M, self.half_bin], dtype=complex) / self.M
    #
    #     self.angle = np.array([0, 0]) / 180 * np.pi
    #
    #     self.method = 'MVDR'
    #
    #     self.frameCount = 0
    #     self.calc = 1
    #     self.estPos = 200
    #
    #     self.Rvv = np.zeros((self.half_bin, self.M, self.M), dtype=complex)
    #     self.Rvv_inv = np.zeros((self.half_bin, self.M, self.M), dtype=complex)
    #     self.Ryy = np.zeros((self.half_bin, self.M, self.M), dtype=complex)
    #
    #     self.AlgorithmList = ['src', 'DS', 'MVDR', 'TFGSC']
    #     self.AlgorithmIndex = 0
    #
    #     # blocking matrix
    #     self.BM = np.zeros((self.M, self.M - 1, self.half_bin), dtype=complex)
    #     # noise reference
    #     self.U = np.zeros((self.M - 1, self.half_bin), dtype=complex)
    #     # fixed beamformer weights for upper path
    #     self.W = np.zeros((self.M, self.half_bin), dtype=complex)
    #     # MNC weights for lower path
    #     self.G = np.zeros((self.M - 1, self.half_bin), dtype=complex)
    #     self.Pest = np.ones(self.half_bin)
    #
    #     self.Yfbf = np.zeros((self.half_bin), dtype=complex)
    #
    #     self.mcra = NoiseEstimationMCRA(nfft=self.nfft)
    #     self.omlsa_multi = NsOmlsaMulti(nfft=self.nfft, cal_weights=True, M=channels)
    #     self.mcspp = McSppBase(nfft=self.nfft, channels=channels)
    #     self.mc_mcra = McMcra(nfft=self.nfft, channels=channels)
    #     self.spp = self.mc_mcra
    #


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
    # audiowrite('wav/x_cleanspeech.wav', x.transpose())

    start = time()

    fdgsc = FDGSC(MicArrayObj, frameLen, hop, nfft, c, fs)
    # """
    # fixed beamformer precesing function
    # method:
    # 'DS':   delay-and-sum beamformer
    # 'MVDR': MVDR beamformer under isotropic noise field
    #
    # """
    yout = fdgsc.process(x)

    print(yout.shape)

    audiowrite('wav/bm.wav', yout.transpose())

    end = time()
    print(end - start)

    return

    # listen processed result
    if (args.listen):
        sd.default.channels = 1
        sd.play(yout['data'], fs)
        sd.wait()

    # view beampattern
    print(yout['beampattern'].shape)
    mesh(yout['beampattern'])
    plt.title('beampattern')
    pmesh(yout['beampattern'])

    # save audio
    if (args.save):
        wavfile.write('output/output_fixedbeamformer.wav', 16000, yout['data'])

    visual(x[0, :], yout['data'])
    plt.title('spectrum')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Feature Extractor")
    parser.add_argument("-l", "--listen", action='store_true', help="set to listen output")  # if set true
    parser.add_argument("-s", "--save", action='store_true', help="set to save output")  # if set true

    args = parser.parse_args()
    main(args)
