"""
Multidelay Block Frequency Domain Adaptive Filte
==============

----------


.. [1] "Multidelay block frequency domain adaptive filter," in IEEE Transactions on Acoustics, Speech, and Signal Processing, vol. 38, no. 2, pp. 373-376, Feb. 1990, doi: 10.1109/29.103078.

"""
import argparse

import numpy as np
from matplotlib import pyplot as plt
from numpy.fft import irfft as ifft
from numpy.fft import rfft as fft
from scipy.signal import convolve as conv
from tqdm import tqdm

from DistantSpeech.adaptivefilter.BaseFilter import BaseFilter, awgn
from DistantSpeech.beamformer.utils import load_audio, DelaySamples


# def circshift(X, axis=0, Xm=None):
#     if axis == 0:
#         X[1:, :] = X[:-1, :]
#         self.X[0, :] = Xm


def update_matrix(X, Xm, axis=0):
    if axis == 0:
        X[:, 1:] = X[:, :-1]
        X[:, 0:1] = Xm

    return X


class Mdf(BaseFilter):
    def __init__(
        self, filter_len=1024, mu=0.01, num_block=1, constrain=True, n_channels=1, alpha=0.8, non_causal=False
    ):
        BaseFilter.__init__(self, filter_len=filter_len, mu=mu)
        self.n_channels = n_channels

        self.num_block = num_block
        self.block_len = int(self.filter_len / self.num_block)
        self.n_fft = 2 * self.block_len
        self.half_bin = self.n_fft // 2 + 1

        self.X = np.zeros((self.half_bin, self.num_block), dtype=complex)  # block matrix

        self.input_buffer = np.zeros((self.n_fft, n_channels))  # to store [old, new]

        self.w_pad = np.zeros((self.filter_len * 2, n_channels))

        self.W = np.zeros((self.half_bin, self.num_block), dtype=complex)

        self.Pm = np.zeros((self.half_bin, self.num_block))
        self.P = np.zeros((self.half_bin, 1))
        self.alpha = alpha

        self.constrain = constrain

        self.non_causal = non_causal
        if non_causal:
            self.delay_samples = DelaySamples(self.filter_len, int(self.filter_len / 2))

    def update_input_matrix(self, Xm):

        self.X[1:, :] = self.X[:-1, :]
        self.X[0, :] = Xm

    def set_weights(self, weights):
        weights = np.squeeze(weights)
        assert len(weights) == self.filter_len
        self.w[:, 0] = np.squeeze(weights)
        self.w_pad[: self.filter_len, 0] = self.w[:, 0]
        self.W = np.fft.rfft(self.w_pad, axis=0)

    def update_input(self, xt_vec: np.array):
        """
        update input data buffer
        :param xt_vec: the signal need to be filtered, (n_samples,) or (n_samples, n_chs)
        :return:
        """
        if xt_vec.ndim == 1:
            xt_vec = xt_vec[:, np.newaxis]
        assert self.n_channels == xt_vec.shape[1]
        self.input_buffer[: self.block_len, :] = self.input_buffer[self.block_len :, :]  # old
        self.input_buffer[self.block_len :, :] = xt_vec  # new

        return self.input_buffer

    def update(self, x_n_vec, d_n_vec, update=True, p=1.0, fir_truncate=None):
        """fast frequency lms update function

        Parameters
        ----------
        x_n_vec : np.array, (n_samples,) or (n_samples, n_chs)
            input signal
        d_n_vec : np.array,  (n_samples,) or (n_samples, 1)
            expected signal
        update : bool, optional
            control whether update filter coeffs, by default True
        p : float, optional
            speech present prob, by default 1.0
        fir_truncate : np.array, optional
            fir truncate length, by default None

        Returns
        -------
        e : np.array, (n_samples, 1)
            error output signal
        w : np.array,  (filter_len, 1)
            estimated filter coeffs
        """
        self.update_input(x_n_vec)

        Xm = fft(self.input_buffer, n=self.n_fft, axis=0)

        self.X = update_matrix(self.X, Xm)

        Pm = np.sum(np.real((Xm.conj() * Xm)), axis=1, keepdims=True)
        self.Pm = update_matrix(self.Pm, Pm)

        self.P = self.alpha * self.P + (1 - self.alpha) * np.sum(self.Pm, axis=1, keepdims=True)

        # save only the last half frame to avoid circular convolution effects
        y = ifft(np.sum(self.X * self.W, axis=1, keepdims=True), axis=0)[-self.block_len :, :]

        # use causal filter to estimate non-causal system will introduce a delay(filter_len/2) compare to expected signal
        if self.non_causal:
            d_n_vec = self.delay_samples.delay(d_n_vec)

        if d_n_vec.ndim == 1:
            d_n_vec = d_n_vec[:, np.newaxis]
        e = d_n_vec - y

        e_pad = np.concatenate((np.zeros((self.block_len, 1)), e), axis=0)

        E = fft(e_pad, n=self.n_fft, axis=0)

        eps = 1e-6
        grad = self.X.conj() * E / (self.P + eps)

        if self.constrain:
            grad_1 = ifft(grad, n=self.n_fft, axis=0)
            grad_1[-self.block_len :] = 0
            grad = fft(grad_1, n=self.n_fft, axis=0)

        if update:
            self.W = self.W + p * 2 * self.mu * grad  # update filter weights

        w_est = ifft(self.W, n=self.n_fft, axis=0)
        self.w = w_est[: self.block_len, :]

        self.w = np.reshape(self.w.T, (-1, 1))

        if fir_truncate is not None:
            w_shift = self.w.copy()
            w_shift[:fir_truncate] = 0.0
            w_shift[-fir_truncate:] = 0.0
            self.W = np.fft.rfft(w_shift, n=self.n_fft, axis=0)

        return e, self.w


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Feature Extractor")
    parser.add_argument("-l", "--listen", action='store_true', help="set to listen output")  # if set true
    parser.add_argument("-s", "--save", action='store_true', help="set to save output")  # if set true

    args = parser.parse_args()
    # main(args)
    mdf = Mdf(num_block=2)

    x = np.random.rand(16000)
    d = np.random.rand(16000)

    for n in tqdm(range(len(x) - mdf.block_len)):
        if np.mod(n, mdf.block_len) == 0:
            input_vector = x[n : n + mdf.block_len]
            d_vector = d[n : n + mdf.block_len]
            err, w_mdf = mdf.update(input_vector, d_vector)

    print(w_mdf.shape)
