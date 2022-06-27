import numpy as np
import argparse
from scipy.signal import convolve as conv
from matplotlib import pyplot as plt
from tqdm import tqdm
from DistantSpeech.beamformer.utils import load_audio
from DistantSpeech.adaptivefilter import BaseFilter


class SubbandLMS(BaseFilter):
    def __init__(self, filter_len=2, num_bands=257, mu=0.1, normalization=True, alpha=0.9):
        self.filter_len = filter_len
        self.num_bands = num_bands

        self.W = np.zeros((filter_len, num_bands), dtype=complex)
        self.mu = mu
        self.norm = normalization
        self.alpha = alpha

        self.input_buffer = np.zeros((filter_len, num_bands), dtype=complex)
        self.P = np.zeros((num_bands,))

    def update_input(self, xt):
        """

        :param xt:
        :return:
        """
        # update input buffer
        self.input_buffer[1:, :] = self.input_buffer[:-1, :]
        self.input_buffer[0, :] = xt

    def update(self, x_n, d_n, alpha=1e-4):
        self.update_input(x_n)

        # error signal
        filter_output = np.einsum('ij,ij->j', self.W.conj(), self.input_buffer)
        err = d_n - filter_output

        if self.norm:
            self.P = (
                self.alpha * self.P
                + (1 - self.alpha) * np.einsum('ij,ij->j', self.input_buffer.conj(), self.input_buffer).real
            )
            grad = self.input_buffer * err.conj() / (self.P + alpha)
        else:
            grad = self.input_buffer * err.conj()  # LMS

        self.update_coef(grad)

        return err, self.W

    def update_coef(self, grad):

        if grad.ndim == 1:
            grad = grad[:, np.newaxis]
        self.W = self.W + 2 * self.mu * grad

        return self.W

    def filter(self, data, data_d):
        err = np.zeros(np.size(data))
        for n in range(len(data)):
            self.update_input(data[n])
            err[n] = self.update(data[n], data_d[n])

        return err


def main(args):
    src = load_audio('cleanspeech_aishell3.wav')
    print(src.shape)
    rir = load_audio('rir.wav')
    rir = rir[200:]
    rir = rir[:512, np.newaxis]

    # src = awgn(src, 30)
    print(src.shape)

    SNR = 20
    data_clean = conv(src, rir[:, 0])
    data = data_clean[: len(src)]
    # data = awgn(data, SNR)

    filter_len = 512
    w = np.zeros((filter_len, 1))

    lms = BaseFilter(filter_len=filter_len, mu=0.1, normalization=False)
    nlms = BaseFilter(filter_len=filter_len, mu=0.1)

    est_err_lms = np.zeros(np.size(data))
    est_err_nlms = np.zeros(np.size(data))

    for n in tqdm(range((len(src)))):
        _, w_lms = lms.update(src[n], data[n])
        _, w_nlms = nlms.update(src[n], data[n])
        est_err_lms[n] = np.sum(np.abs(rir - w_lms[: len(rir)]) ** 2)
        est_err_nlms[n] = np.sum(np.abs(rir - w_nlms[: len(rir)]) ** 2)

    plt.plot(10 * np.log(est_err_lms / np.sum(np.abs(rir[:, 0]) ** 2) + 1e-12))
    plt.plot(10 * np.log(est_err_nlms / np.sum(np.abs(rir[:, 0]) ** 2)) + 1e-12)
    plt.legend(['lms', 'nlms'], loc='upper right')
    plt.ylabel("$\||\hat{w}-w\||_2$")
    plt.title('weight estimation error vs step')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Feature Extractor")
    parser.add_argument("-l", "--listen", action='store_true', help="set to listen output")  # if set true
    parser.add_argument("-s", "--save", action='store_true', help="set to save output")  # if set true

    args = parser.parse_args()
    main(args)
