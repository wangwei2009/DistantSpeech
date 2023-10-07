"""
% frequency-domain fast block-lms algorithm using overlap-save method
% refer to
%
%
%
% Created by Wang wei
"""
import argparse

import numpy as np
from matplotlib import pyplot as plt
from numpy.fft import irfft as ifft
from numpy.fft import rfft as fft
from scipy.signal import convolve as conv
from tqdm import tqdm

from DistantSpeech.adaptivefilter.BaseFilter import BaseFilter, awgn
from DistantSpeech.adaptivefilter.FastFreqLms import FastFreqLms


class AdaptiveBlockingMatrixFilter(FastFreqLms):
    def __init__(
        self,
        filter_len=128,
        hop_len=None,
        win_len=None,
        mu=0.01,
        constrain=True,
        n_channels=1,
        alpha=0.9,
        non_causal=False,
        two_path=False,
    ):
        FastFreqLms.__init__(
            self,
            filter_len=filter_len,
            hop_len=hop_len,
            win_len=win_len,
            mu=mu,
            constrain=constrain,
            n_channels=n_channels,
            alpha=alpha,
            non_causal=non_causal,
            two_path=two_path,
        )

        self.m_upper_bound = np.zeros((self.n_fft // 2,))
        self.m_lower_bound = np.zeros((self.n_fft // 2,))

        deltax = 0.001
        for i in range(self.n_fft // 2):
            self.m_upper_bound[i] = deltax
            self.m_lower_bound[i] = -deltax
        self.m_upper_bound[self.n_fft // 4] = 0.9
        self.m_upper_bound[self.n_fft // 4 + 1] = 0.3
        self.m_upper_bound[self.n_fft // 4 - 1] = 0.3
        self.m_upper_bound[self.n_fft // 4 + 2] = 0.05
        self.m_upper_bound[self.n_fft // 4 - 2] = 0.05

    def update(self, x_n_vec, d_n_vec, update=True, p=1.0, fir_truncate=None, filter_p=False):
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

        X, e = self.compute_freq_conv(x_n_vec, d_n_vec)

        grad = self.compute_freq_xcorr(X, e)

        if update:
            self.W = self.W + p * self.mu * grad  # update filter weights

        if self.constrain:
            w = ifft(self.W, n=self.n_fft, axis=0)
            w[-self.hop_len :] = 0

            limit_ind = self.n_fft // 4 - 1
            for i in range(limit_ind, 0, -1):
                w[i] = np.minimum(np.maximum(w[i], self.m_lower_bound[i]), self.m_upper_bound[i])
                w[self.n_fft // 2 - i] = np.minimum(
                    np.maximum(w[self.n_fft // 2 - i], self.m_lower_bound[self.n_fft // 2 - i]),
                    self.m_upper_bound[self.n_fft // 2 - i],
                )
            w[0] = np.minimum(np.maximum(w[0], self.m_lower_bound[0]), self.m_upper_bound[0])
            w[self.n_fft // 4] = np.minimum(
                np.maximum(w[self.n_fft // 4], self.m_lower_bound[self.n_fft // 4]), self.m_upper_bound[self.n_fft // 4]
            )

            # for i in range(self.n_fft // 2):
            #     w[i] = np.minimum(np.maximum(w[i], self.m_lower_bound[i]), 0.1)

            self.W = fft(w, n=self.n_fft, axis=0)

        w_est = ifft(self.W, n=self.n_fft, axis=0)
        self.w = w_est[: self.filter_len, :]

        if fir_truncate is not None:
            w_shift = self.w.copy()
            w_shift[:fir_truncate] = 0.0
            w_shift[-fir_truncate:] = 0.0
            self.W = np.fft.rfft(w_shift, n=self.n_fft, axis=0)

        return e, self.w


def main(args):
    # src = load_audio('cleanspeech_aishell3.wav')
    src = load_audio('cleanspeech.wav')
    print(src.shape)
    src = np.random.randn(len(src))  # white noise, best for adaptive filter
    rir = load_audio('rir.wav')
    rir = rir[199:]
    rir = rir[:512, np.newaxis]

    # src = awgn(src, 30)
    print(src.shape)

    SNR = 20
    data_clean = conv(src, rir[:, 0])
    data = data_clean[: len(src)]
    data = awgn(data, SNR)

    filter_len = 512
    hop_len = 128

    fdaf = FastFreqLms(
        filter_len=filter_len,
        hop_len=hop_len,
        mu=1e-1,
    )

    est_err_fdaf = np.zeros(len(data) - hop_len)

    eps = 1e-6

    w_fdaf = np.zeros((filter_len, 1))

    output = np.zeros((len(data), 1))

    for n in tqdm(range((len(src)))):
        if n == 0:
            continue
        if np.mod(n, hop_len) == 0:
            output[n - hop_len : n], w_fdaf = fdaf.update(src[n - hop_len : n], data[n - hop_len : n])
        est_err_fdaf[n - hop_len] = np.sum(np.abs(rir - w_fdaf[: len(rir)]) ** 2)

    rir_norm = np.sum(np.abs(rir[:, 0]) ** 2)
    plt.plot(10 * np.log10(est_err_fdaf / rir_norm + eps))
    plt.legend(['fd-lms'], loc='upper right')
    plt.ylabel("$\||\hat{w}-w\||_2$")
    plt.title('weight estimation error vs step')
    plt.show()
    plt.savefig('flms2.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Feature Extractor")
    parser.add_argument("-l", "--listen", action='store_true', help="set to listen output")  # if set true
    parser.add_argument("-s", "--save", action='store_true', help="set to save output")  # if set true

    args = parser.parse_args()
    main(args)
    # test_aic()
