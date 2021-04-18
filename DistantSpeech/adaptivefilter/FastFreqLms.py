"""
% frequency-domain fast block-lms algorithm using overlap-save method
%
%
% Created by Wang wei
"""
import numpy as np
import argparse
from DistantSpeech.beamformer.utils import load_audio
from scipy.signal import convolve as conv
from matplotlib import pyplot as plt
from tqdm import tqdm
from DistantSpeech.adaptivefilter.BaseFilter import BaseFilter
from DistantSpeech.adaptivefilter.BlockLMS import BlockLms
from numpy.fft import rfft as fft
from numpy.fft import irfft as ifft


class FastFreqLms(BaseFilter):
    def __init__(self, filter_len=128, mu=0.01, constrain=True):
        BaseFilter.__init__(self, filter_len=filter_len, mu=mu)
        self.input_buffer = np.zeros((filter_len * 2, 1))               # to store [old, new]

        self.n_fft = self.filter_len * 2

        self.w_pad = np.zeros((self.filter_len * 2, 1))

        self.W = np.fft.rfft(self.w_pad, axis=0)

        self.P = np.zeros((self.n_fft // 2 + 1, 1))
        self.alpha = 0.9

        self.constrain = constrain

    def update_input(self, xt_vec: np.array):
        if xt_vec.ndim == 1:
            xt_vec = xt_vec[:, np.newaxis]
        self.input_buffer[:self.filter_len, :] = self.input_buffer[self.filter_len:, :]   # old
        self.input_buffer[self.filter_len:, :] = xt_vec                                   # new

    def update(self, x_n_vec, d_n_vec):
        self.update_input(x_n_vec)
        X = np.fft.rfft(self.input_buffer, n=self.n_fft, axis=0)
        self.P = self.alpha * self.P + (1 - self.alpha) * np.real((X.conj() * X))

        y = np.fft.irfft(X * self.W, axis=0)[-self.filter_len:, :]

        if d_n_vec.ndim == 1:
            d_n_vec = d_n_vec[:, np.newaxis]
        e = d_n_vec - y

        e_pad = np.concatenate((np.zeros((self.filter_len, 1)), e), axis=0)

        E = fft(e_pad, n=self.n_fft, axis=0)

        eps = 1e-6
        grad = X.conj() * E / (self.P + eps)

        if self.constrain:
            grad_1 = ifft(grad, n=self.n_fft, axis=0)
            grad_1[-self.filter_len:] = 0
            grad = fft(grad_1, n=self.n_fft, axis=0)

        self.W = self.W + 2 * self.mu * grad  # update filter weights

        w_est = ifft(self.W, n=self.n_fft, axis=0)
        self.w = w_est[:self.filter_len, :]

        return e, self.w


def main(args):
    # src = load_audio('cleanspeech_aishell3.wav')
    src = load_audio('cleanspeech.wav')
    print(src.shape)
    rir = load_audio('rir.wav')
    rir = rir[200:]
    rir = rir[:512, np.newaxis]

    # src = awgn(src, 30)
    print(src.shape)

    SNR = 20
    data_clean = conv(src, rir[:, 0])
    data = data_clean[:len(src)]
    # data = awgn(data, SNR)

    filter_len = 512
    w = np.zeros((filter_len, 1))

    fdaf = FastFreqLms(filter_len=filter_len, mu=0.1)

    lms = BaseFilter(filter_len=filter_len, mu=0.01, normalization=False)
    nlms = BaseFilter(filter_len=filter_len, mu=0.1)
    #
    blms = BlockLms(block_len=2, filter_len=filter_len, mu=0.1)
    #
    est_err_lms = np.zeros(np.size(data))
    est_err_nlms = np.zeros(np.size(data))
    est_err_blms = np.zeros(np.size(data))
    est_err_fdaf = np.zeros(np.size(data))

    block_len = filter_len
    block_num = len(src) // block_len

    for n in tqdm(range((len(src)))):
        if n < filter_len:
            continue
        _, w_lms = lms.update(src[n], data[n])
        _, w_nlms = nlms.update(src[n], data[n])
        _, w_blms = blms.update(src[n], data[n])
        if np.mod(n, block_len) == 0:
            _, w_fdaf = fdaf.update(src[n-filter_len:n], data[n-filter_len:n])
        est_err_lms[n] = np.sum(np.abs(rir - w_lms[:len(rir)])**2)
        est_err_nlms[n] = np.sum(np.abs(rir - w_nlms[:len(rir)])**2)
        est_err_blms[n] = np.sum(np.abs(rir - w_blms[:len(rir)]) ** 2)
        est_err_fdaf[n] = np.sum(np.abs(rir - w_fdaf[:len(rir)]) ** 2)

    plt.plot(10 * np.log(est_err_lms / np.sum(np.abs(rir[:, 0])**2) + 1e-12))
    plt.plot(10 * np.log(est_err_nlms / np.sum(np.abs(rir[:, 0])**2)) + 1e-12)
    plt.plot(10 * np.log(est_err_blms / np.sum(np.abs(rir[:, 0]) ** 2)) + 1e-12)
    plt.plot(10 * np.log(est_err_fdaf / np.sum(np.abs(rir[:, 0]) ** 2)) + 1e-12)
    plt.legend(['lms', 'nlms', 'block-lms', 'fd-lms'], loc='upper right')
    plt.ylabel("$\||\hat{w}-w\||_2$")
    plt.title('weight estimation error vs step')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Feature Extractor")
    parser.add_argument("-l", "--listen", action='store_true', help="set to listen output")  # if set true
    parser.add_argument("-s", "--save", action='store_true', help="set to save output")  # if set true

    args = parser.parse_args()
    main(args)
