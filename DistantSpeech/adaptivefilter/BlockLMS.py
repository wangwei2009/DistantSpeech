"""
time-domain block-lms
High computation cost when long filter length

 Created by Wang wei
"""
import numpy as np
import argparse
from DistantSpeech.beamformer.utils import load_audio
from scipy.signal import convolve as conv
from matplotlib import pyplot as plt
from tqdm import tqdm
from DistantSpeech.adaptivefilter.BaseFilter import BaseFilter


class BlockLms(BaseFilter):
    def __init__(self, block_len=4, filter_len=16, mu=0.1):
        BaseFilter.__init__(self, filter_len=filter_len, mu=mu)
        self.L = block_len                        # block length
        self.d = np.zeros((block_len, 1))         # expectation vector
        self.e = np.zeros((block_len, 1))         # error vector

        self.input_matrix = np.zeros((block_len, filter_len))  # block matrix

        self.counter = 0

    def update_input_matrix(self, x_n):
        self.update_input(x_n)
        self.input_matrix[1:, :] = self.input_matrix[:-1, :]
        self.input_matrix[0, :] = self.input_buffer.T

    def update(self, x_n, d_n, alpha=1e-4):
        self.update_input_matrix(x_n)

        # current filter output
        y = self.w.T @ self.input_buffer          # linear convolution, can lead to fast frequency-domain LMS
        # current error
        en = d_n - y

        # update error vector
        self.e[1:, 0] = self.e[:-1, 0]
        self.e[0] = en                          # update err vector

        if self.counter % self.L == 0:
            norm = np.linalg.norm(self.input_matrix, axis=1, keepdims=True) ** 2 + alpha
            grad = np.sum(self.input_matrix * self.e / norm, axis=0)  # linear correlation

            self.w = self.update_coef(grad)

        return en, self.w


def main(args):
    # src = load_audio('cleanspeech_aishell3.wav')
    src = load_audio('cleanspeech.wav')
    print(src.shape)
    rir = load_audio('rir.wav')
    rir = rir[200:]
    rir = rir[:256, np.newaxis]

    # src = awgn(src, 30)
    print(src.shape)

    SNR = 20
    data_clean = conv(src, rir[:, 0])
    data = data_clean[:len(src)]
    # data = awgn(data, SNR)

    filter_len = 256
    w = np.zeros((filter_len, 1))

    lms = BaseFilter(filter_len=filter_len, mu=0.01, normalization=False)
    nlms = BaseFilter(filter_len=filter_len, mu=0.01)

    blms = BlockLms(block_len=2, filter_len=filter_len, mu=0.01)

    est_err_lms = np.zeros(np.size(data))
    est_err_nlms = np.zeros(np.size(data))
    est_err_blms = np.zeros(np.size(data))

    for n in tqdm(range((len(src)))):
        _, w_lms = lms.update(src[n], data[n])
        _, w_nlms = nlms.update(src[n], data[n])
        _, w_blms = blms.update(src[n], data[n])
        est_err_lms[n] = np.sum(np.abs(rir - w_lms[:len(rir)])**2)
        est_err_nlms[n] = np.sum(np.abs(rir - w_nlms[:len(rir)])**2)
        est_err_blms[n] = np.sum(np.abs(rir - w_blms[:len(rir)]) ** 2)

    plt.plot(10 * np.log(est_err_lms / np.sum(np.abs(rir[:, 0])**2) + 1e-12))
    plt.plot(10 * np.log(est_err_nlms / np.sum(np.abs(rir[:, 0])**2)) + 1e-12)
    plt.plot(10 * np.log(est_err_blms / np.sum(np.abs(rir[:, 0]) ** 2)) + 1e-12)
    plt.legend(['lms', 'nlms', 'block-lms'], loc='upper right')
    plt.ylabel("$\||\hat{w}-w\||_2$")
    plt.title('weight estimation error vs step')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Feature Extractor")
    parser.add_argument("-l", "--listen", action='store_true', help="set to listen output")  # if set true
    parser.add_argument("-s", "--save", action='store_true', help="set to save output")  # if set true

    args = parser.parse_args()
    main(args)
