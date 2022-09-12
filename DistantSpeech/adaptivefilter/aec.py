"""
Multidelay Block Frequency Domain Adaptive Filte
==============

----------

..    [1].J. S. Soo, K. K. Pang Multidelay block frequency adaptive filter, 
      IEEE Trans. Acoust. Speech Signal Process., Vol. ASSP-38, No. 2, 
      February 1990.

..    [2].Valin, J.-M., On Adjusting the Learning Rate in Frequency Domain Echo 
      Cancellation With Double-Talk. IEEE Transactions on Audio,
      Speech and Language Processing, Vol. 15, No. 3, pp. 1030-1034, 2007.
      http://people.xiph.org/~jm/papers/valin_taslp2006.pdf

"""
import argparse

import numpy as np
from matplotlib import pyplot as plt
from numpy.fft import irfft as ifft
from numpy.fft import rfft as fft
from scipy.signal import convolve as conv
from tqdm import tqdm

from DistantSpeech.adaptivefilter import BaseFilter, awgn, Mdf
from DistantSpeech.beamformer.utils import load_audio, DelaySamples
from DistantSpeech.adaptivefilter.mdf import mdf_adjust_prop


# def circshift(X, axis=0, Xm=None):
#     if axis == 0:
#         X[1:, :] = X[:-1, :]
#         self.X[0, :] = Xm


def update_matrix(X, Xm, axis=0):
    if axis == 0:
        X[:, 1:] = X[:, :-1]
        X[:, 0:1] = Xm

    return X


class Aec(BaseFilter):
    def __init__(
        self,
        filter_len=1024,
        mu=0.01,
        num_block=1,
        constrain=True,
        n_channels=1,
        alpha=0.8,
        prop=True,
        non_causal=False,
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
        self.W = np.zeros((self.half_bin, self.num_block), dtype=complex)
        self.foreground = np.zeros((self.half_bin, self.num_block), dtype=complex)

        self.Y_pre = np.zeros((self.half_bin, n_channels), dtype=complex)
        self.Py = np.zeros((self.half_bin, n_channels))

        self.E_pre = np.zeros((self.half_bin, n_channels), dtype=complex)
        self.Pe = np.zeros((self.half_bin, n_channels))
        self.Ryy = np.zeros((self.half_bin, n_channels))
        self.Rey = np.zeros((self.half_bin, n_channels))

        self.mu_opt = np.zeros((self.half_bin, n_channels))

        self.D = 3
        self.grad_buffer = np.zeros((self.half_bin, self.D), dtype=complex)

        self.Pm = np.zeros((self.half_bin, self.num_block))
        self.P = np.zeros((self.half_bin, 1))
        self.alpha = alpha

        self.Davg1 = 0  # （前景滤波器输出能量-背景滤波器输出能量）平滑
        self.Davg2 = 0
        self.Dvar1 = 0
        self.Dvar2 = 0  #

        self.prop = prop

        self.window = 0.5 - 0.5 * np.cos(
            2 * np.pi * (np.linspace(0, self.n_fft - 1, num=self.n_fft)) / self.n_fft
        )

        self.gamma = 0.8
        self.beta = 1.0

        self.adapted = 0

        self.fs = 16000

        self.beta0 = (2.0 * self.block_len) / self.fs
        self.beta_max = (0.5 * self.block_len) / self.fs

        self.power_1 = np.zeros((self.half_bin, n_channels))
        self.power = np.zeros((self.half_bin, n_channels))

        self.cnt = 0

        self.sum_adapt = 0

        self.leak_estimate = 0.1

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

    def update_correlation(self, Y, Y_pre, alpha1, alpha2):
        self.Py = (1 - alpha1) * self.Py + alpha1 * (
            np.abs(Y * Y.conj()) - np.abs(Y_pre * Y_pre.conj())
        )
        self.Y_pre[:] = Y
        self.Ryy = (1 - alpha2) * self.Ryy + alpha2 * self.Py

    def transfer_logic(self, e_f, e_b, y_f, y_b):
        # transfer logic
        if 10 * np.log10(np.sum(np.abs(e_f)) / (np.sum(np.abs(e_b)) + 1e-6)) > 3:
            self.foreground[:] = self.W
            y_f = y_b.copy()

        return e_f, e_b, y_f, y_b

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

        Sxx = np.sum(x_n_vec**2)  # echo energy

        Xm = fft(self.input_buffer, n=self.n_fft, axis=0)  # eq.1

        self.X = update_matrix(self.X, Xm)  # eq.2

        ss = 0.35 / self.num_block
        ss_1 = 1 - ss
        # % Smooth far end energy estimate over time */
        self.power = ss_1 * self.power + ss * np.abs(Xm) ** 2

        Pm = np.sum(np.real((Xm.conj() * Xm)), axis=1, keepdims=True)  # eq.12
        self.Pm = update_matrix(self.Pm, Pm)  # eq.13

        self.P = self.alpha * self.P + (1 - self.alpha) * np.sum(
            self.Pm, axis=1, keepdims=True
        )  # eq.11

        Y = np.sum(self.X * self.W, axis=1, keepdims=True)

        # save only the last half frame to avoid circular convolution effects
        # y = ifft(np.sum(self.X * self.W, axis=1, keepdims=True), axis=0)[
        #     -self.block_len :, :
        # ]
        y_b = ifft(np.sum(self.X * self.W, axis=1, keepdims=True), axis=0)[-self.block_len :, :]
        y = ifft(np.sum(self.X * self.foreground, axis=1, keepdims=True), axis=0)[
            -self.block_len :, :
        ]
        y_f = y

        # use causal filter to estimate non-causal system will introduce a delay(filter_len/2) compare to expected signal
        if self.non_causal:
            d_n_vec = self.delay_samples.delay(d_n_vec)

        if d_n_vec.ndim == 1:
            d_n_vec = d_n_vec[:, np.newaxis]
        e = d_n_vec - y
        e_b = d_n_vec - y_b
        e_f = d_n_vec - y_f

        e_f, e_b, y_f, y_b = self.transfer_logic(e_f, e_b, y, y_b)
        out = d_n_vec - y_f
        assert id(y_f) != id(y_b)

        e_pad = np.concatenate((np.zeros((self.block_len, 1)), e_b), axis=0)

        E = fft(e_pad, n=self.n_fft, axis=0)

        Sey = np.sum(e_b * y_b)  # % 误差与估计回声相关能量和
        Syy = np.sum(y_b)  # % 估计回声能量和
        Sdd = np.sum(d_n_vec**2)  # % 近端信号能量和

        gamma = 0.8
        gamma_1 = 1 - gamma
        self.Py = gamma_1 * self.Py + gamma * (np.abs(Y) ** 2 - np.abs(self.Y_pre) ** 2)  # eq.17
        self.Y_pre[:] = Y

        self.Pe = gamma_1 * self.Pe + gamma * (np.abs(E) ** 2 - np.abs(self.E_pre) ** 2)  # eq.18
        self.E_pre[:] = E

        Syy = np.sum(y_b**2)
        See = np.sum(e_b**2)

        alpha = Syy / See
        alpha = self.beta0 * np.minimum(alpha, 1.0)  # eq.22
        alpha_1 = 1 - alpha  # eq.22

        self.Ryy = alpha_1 * self.Ryy + alpha * self.Py * self.Py  # eq.20
        self.Rey = alpha_1 * self.Rey + alpha * self.Py * self.Pe  # eq.21

        # % leak_estimate is the linear regression result */
        self.leak_estimate = np.sum(self.Rey) / (np.sum(self.Ryy) + 1e-6)  # eq. 19

        self.mu_opt = self.leak_estimate * np.abs(Y) ** 2 / (np.abs(E) ** 2 + 1e-3)  # eq.16
        self.mu_opt = np.maximum(np.minimum(self.mu_opt, 0.1), 1e-3)

        if self.cnt < 5:
            self.cnt = self.cnt + 1
            self.mu_opt = np.ones((self.half_bin, self.n_channels)) * 0.1

        eps = 1e-6
        grad = self.X.conj() * E / (self.P + eps)

        if self.constrain:
            grad_1 = ifft(grad, n=self.n_fft, axis=0)
            grad_1[-self.block_len :] = 0
            grad = fft(grad_1, n=self.n_fft, axis=0)

        # grad = grad * p * self.mu_opt

        if update:
            if self.prop:
                prop_coeffs = mdf_adjust_prop(self.W, self.num_block)
                for n in range(self.num_block):
                    self.W[:, n] = self.W[:, n] + prop_coeffs[n, 0] * self.mu_opt[:, 0] * grad[:, n]
            else:
                self.W = self.W + self.mu_opt * grad  # update filter weights
            # self.W = self.W + prop * self.mu_opt * grad  # update filter weights

        w_est = ifft(self.W, n=self.n_fft, axis=0)
        self.w = w_est[: self.block_len, :]

        self.w = np.reshape(self.w.T, (-1, 1))

        if fir_truncate is not None:
            w_shift = self.w.copy()
            w_shift[:fir_truncate] = 0.0
            w_shift[-fir_truncate:] = 0.0
            self.W = np.fft.rfft(w_shift, n=self.n_fft, axis=0)

        return out, self.w


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Feature Extractor")
    parser.add_argument(
        "-l", "--listen", action="store_true", help="set to listen output"
    )  # if set true
    parser.add_argument(
        "-s", "--save", action="store_true", help="set to save output"
    )  # if set true

    args = parser.parse_args()
    # main(args)
    mdf = Aec(num_block=2)

    x = np.random.rand(16000)
    d = np.random.rand(16000)

    for n in tqdm(range(len(x) - mdf.block_len)):
        if np.mod(n, mdf.block_len) == 0:
            input_vector = x[n : n + mdf.block_len]
            d_vector = d[n : n + mdf.block_len]
            err, w_mdf = mdf.update(input_vector, d_vector)

    print(w_mdf.shape)
