import os
import pickle
import librosa
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sounddevice as sd
from librosa import util
from librosa.filters import get_window
from numba import jit
from DistantSpeech.transform.prototype_filter import PrototypeFilter
from DistantSpeech.transform.design_nyquist_filter import (
    design_Nyquist_analyasis_filter_prototype,
    design_Nyquist_synthesis_filter_prototype,
)


class Subband(object):
    def __init__(self, channel=1, n_fft=256, hop_length=128, window=None, m=2):
        self.channel = channel
        self.n_fft = n_fft
        self.frame_length = self.n_fft
        self.hop_length = hop_length

        r = int(n_fft / hop_length / 2)  # Decimation factor
        M = n_fft
        # m = 2  # Prototype filter length factor
        self.h, self.g = self.design_prototype_filter(n_fft, m, r=r)
        print('self.h:{}'.format(self.h.shape))
        print('self.g:{}'.format(self.g.shape))

        self.first_frame = 1
        if window is not None:
            self.window = window
        else:
            self.window = PrototypeFilter().get_prototype_filter()

        self.half_bin = int(self.n_fft / 2 + 1)
        self.D = 2

        self.win_len = self.h.shape[0]

        self.overlap = self.win_len - self.hop_length
        self.previous_input = np.zeros((self.overlap, self.channel))
        self.previous_output = np.zeros((self.win_len, self.channel))

        self.W0 = np.sum(self.g**2)  # find W0

    def design_prototype_filter(
        self, num_bands, m, r=1, outputdir='/home/wangwei/work/DistantSpeech/DistantSpeech/transform/prototype.ny'
    ):
        M = num_bands
        D = M // (2**r)  # window shift
        if D == 0:
            D = 1

        if not os.path.exists(outputdir):
            os.makedirs(outputdir)

        analysis_filename = os.path.join(outputdir, 'h-M%d-m%d-r%d.pickle' % (M, m, r))
        synthesis_filename = os.path.join(outputdir, 'g-M%d-m%d-r%d.pickle' % (M, m, r))
        if os.path.exists(analysis_filename) and os.path.exists(synthesis_filename):
            with open(analysis_filename, 'rb') as fp:
                # print('Loading analysis prototype from \'%s\'' % analysis_filename)
                h = pickle.load(fp)
            # Read synthesis prototype 'g'
            with open(synthesis_filename, 'rb') as fp:
                # print('Loading synthesis prototype from \'%s\'' % synthesis_filename)
                g = pickle.load(fp)
        else:
            (h, beta) = design_Nyquist_analyasis_filter_prototype(M, m, D)
            print('Inband aliasing error: %f dB' % (10 * np.log(beta)))

            (g, epsir) = design_Nyquist_synthesis_filter_prototype(h, M, m, D)
            print('Residual aliasing distortion: %f dB' % (10 * np.log(epsir)))

            (h, beta) = design_Nyquist_analyasis_filter_prototype(M, m, D)
            print('Inband aliasing error: %f dB' % (10 * np.log(beta)))
            analysis_filename = os.path.join(outputdir, 'h-M%d-m%d-r%d.pickle' % (M, m, r))
            with open(analysis_filename, 'wb') as hfp:
                pickle.dump(h.flatten(), hfp, True)

            (g, epsir) = design_Nyquist_synthesis_filter_prototype(h, M, m, D)
            print('Residual aliasing distortion: %f dB' % (10 * np.log(epsir)))
            synthesis_filename = os.path.join(outputdir, 'g-M%d-m%d-r%d.pickle' % (M, m, r))
            with open(synthesis_filename, 'wb') as gfp:
                pickle.dump(g.flatten(), gfp, True)

            coeff_filename = os.path.join(outputdir, 'M=%d-m=%d-r=%d.m' % (M, m, r))
            with open(coeff_filename, 'w') as cfp:
                for coeff in h:
                    cfp.write('%e ' % (coeff))
                cfp.write('\n')
                for coeff in g:
                    cfp.write('%e ' % (coeff))

        self.h = h.squeeze()
        self.g = g.squeeze()

        return self.h, self.g

    def analysis(self, x):
        """
        streaming multi-channel Short Time Fourier Transform
        :param x: [samples, channels] or [samples,]
        :return: [half_bin, frames, channels] or [half_bin, frames]
        """
        if len(x.shape) == 1:  # single channel
            x = x[:, np.newaxis]
        x = np.vstack((self.previous_input, x))
        x = np.asfortranarray(x)
        # Compute the number of frames that will fit. The end may get truncated.
        n_frames = int((x.shape[0] - self.overlap) / self.hop_length)
        Y = np.zeros((self.half_bin, n_frames, self.channel), dtype=complex)
        for ch in range(self.channel):
            x_ch = x[:, ch]
            for n in range(n_frames):
                x_ch_n = x_ch[self.hop_length * n : self.hop_length * n + self.win_len]
                windowed = np.flip(x_ch_n) * self.h
                x_sumed = np.sum(np.reshape(windowed, (self.D, -1)), axis=0)
                Y[:, n, ch] = np.fft.rfft(x_sumed)

            self.previous_input[:, ch] = x[-self.overlap :, ch]

        return np.squeeze(Y)

    def synthesis(self, Y):
        """
        streaming single channel inverse short time fourier transform
        :param Y: [half_bin, frames]
        :return: single channel time data
        """
        if len(Y.shape) == 1:  # single frame
            Y = Y[:, np.newaxis]
        n_frames = Y.shape[1]
        expected_signal_len = self.n_fft + self.hop_length * (n_frames - 1)
        output = np.zeros(expected_signal_len)
        tdl = self.previous_output[:, 0].copy()
        for n in range(n_frames):
            y = np.fft.irfft(Y[:, n : n + 1], axis=0)
            # print(y)
            y_repmat = np.tile(y, (self.D, 1))
            y_windowed = y_repmat[:, 0] * self.g

            tdl[self.hop_length :] = tdl[: -self.hop_length]
            tdl[: self.hop_length] = 0.0
            tdl = tdl + y_windowed

            t = n * self.hop_length
            output[t : t + self.hop_length] = (self.n_fft * self.hop_length) * np.flip(tdl[-self.hop_length :])

        self.previous_output[:, 0] = tdl.copy()

        return output[: self.hop_length * n_frames] / self.hop_length

    def stft(self, x):

        return self.analysis(x)

    def istft(self, Y):

        return self.synthesis(Y)

    def magphase(self, D, power=1):
        mag = np.abs(D)
        mag **= power
        phase = np.exp(1.0j * np.angle(D))

        return mag, phase


if __name__ == "__main__":
    from DistantSpeech.beamformer.utils import mesh
    from DistantSpeech.beamformer.utils import load_audio as audioread
    from DistantSpeech.beamformer.utils import save_audio as audiowrite
    from tqdm import tqdm

    filename = "DistantSpeech/transform/speech1.wav"
    x, sr = librosa.load(filename, sr=None)
    print(x.shape)
    x_ch2 = np.vstack((x, x)).T
    print(x_ch2.shape)

    n_fft = 512
    hop_length = 128
    transform = Subband(n_fft=512, hop_length=128, channel=2)

    data_recon = np.zeros(x.shape)

    for n in tqdm(range(len(x) - hop_length)):
        if np.mod(n, hop_length) == 0:
            input_vector = x_ch2[n : n + hop_length, :]
            X = transform.analysis(input_vector)
            x_rec = transform.synthesis(X[:, 0])
            data_recon[n : n + hop_length] = x_rec

    audiowrite('/home/wangwei/work/DistantSpeech/example/wpe/speech1_rec.wav', data_recon)
    # mesh()

    # compare difference between original signal and reconstruction signal
    # plt.figure()
    # plt.plot(data[: len(data_recon[160:])] - data_recon[160:])
    # plt.title("difference between source and reconstructed signal")
    # plt.xlabel("samples")
    # plt.ylabel("amplitude")
    # plt.show()

    # if you want to listen the reconstructed signal, uncomment section below
    # sd.play(data_recon, sr)
    # sd.wait()
