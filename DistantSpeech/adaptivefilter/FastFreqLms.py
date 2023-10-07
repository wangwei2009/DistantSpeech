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
from DistantSpeech.adaptivefilter.BlockLMS import BlockLms
from DistantSpeech.beamformer.utils import load_audio, DelaySamples
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


class FastFreqLms(BaseFilter):
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
        BaseFilter.__init__(self, filter_len=filter_len, mu=mu)
        self.n_channels = n_channels

        # circular convolution:
        # linear convolution: win_len + filter_len - 1
        self.hop_len = filter_len if hop_len is None else hop_len
        self.win_len = filter_len * 2 if win_len is None else win_len

        self.input_buffer = np.zeros((self.win_len, n_channels))  # to store [old, new]

        # circular convolution length L, should at least have hop_len valid output
        # 2*L - N1 - N2 + 1 >= hop_len, usually L(n_fft) == N1(win_len),
        # N1 >= hop_len + N2 - 1
        min_win_len = self.hop_len + self.filter_len - 1
        self.n_fft = 2 ** ((int)(np.log2(min_win_len)) + 1)

        self.overlap = self.win_len - self.hop_len
        self.w_pad = np.zeros((self.n_fft, n_channels))

        self.W = np.fft.rfft(self.w_pad, axis=0)

        self.P = np.zeros((self.n_fft // 2 + 1, 1))
        self.alpha = alpha

        self.constrain = constrain

        self.non_causal = non_causal
        if non_causal:
            self.delay_samples = DelaySamples(self.filter_len, int(self.filter_len / 2))

        self.window = 0.5 - 0.5 * np.cos(2 * np.pi * (np.linspace(0, self.n_fft - 1, num=self.n_fft)) / self.n_fft)
        self.window = self.window[:, np.newaxis]

        self.two_path = two_path
        if self.two_path:
            self.foreground = np.fft.rfft(self.w_pad, axis=0)

        self.e_pre = np.zeros((self.filter_len, 1))

        self.tranform_y = Transform(n_fft=self.filter_len * 2, hop_length=self.filter_len, channel=1)

        # self.p = np.zeros((self.n_fft // 2 + 1, 1))
        self.p = 0.5

    def transfer_logic(self, e_f, e_b, y_f, y_b):
        # # transfer logic
        if 10 * np.log10(np.sum(np.abs(e_f)) / (np.sum(np.abs(e_b)) + 1e-6) + 1e-6) > 3:
            self.foreground[:] = self.W
            # y_f = y_b.copy()
            # % Apply a smooth transition so as to not introduce blocking artifacts */
            y_f = self.window[self.filter_len :] * y_f + self.window[: self.filter_len] * y_b

        return e_f, e_b, y_f, y_b

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
        assert self.n_channels == xt_vec.shape[1], 'n_channels:{} != xt_vec.shape[1]:{}'.format(
            self.n_channels, xt_vec.shape[1]
        )
        self.input_buffer[: self.overlap, :] = self.input_buffer[-(self.overlap) :, :]  # old
        self.input_buffer[-self.hop_len :, :] = xt_vec  # new

        return self.input_buffer

    def compute_freq_conv(self, x_n_vec, d_n_vec):
        """compute frequency-domain convolution between input x and expected d

        Parameters
        ----------
        x_n_vec : np.array, (n_samples,) or (n_samples, n_chs)
            input signal
        d_n_vec : np.array,  (n_samples,) or (n_samples, 1)
            expected signal

        Returns
        -------
        X : np.array, (n_fft, n_chs)
            filtered output signal
        e : np.array,  (n_samples, 1)
            error output signal
        """

        self.update_input(x_n_vec)
        X = np.fft.rfft(self.input_buffer, n=self.n_fft, axis=0)
        self.P = self.alpha * self.P + (1 - self.alpha) * np.sum(np.real((X.conj() * X)), axis=1, keepdims=True)

        # save only the last half frame to avoid circular convolution effects and sum of multichannel signal
        y = np.fft.irfft(np.sum(X * self.W, axis=-1))[-self.hop_len :]
        y = y[:, np.newaxis]

        if self.two_path:
            y_f = np.fft.irfft(X * self.foreground, axis=0)[-self.filter_len :, :]
            y_f = np.sum(y_f, axis=1, keepdims=True)

        # use causal filter to estimate non-causal system will introduce a delay(filter_len/2) compare to expected signal
        if self.non_causal:
            d_n_vec = self.delay_samples.delay(d_n_vec)

        if d_n_vec.ndim == 1:
            d_n_vec = d_n_vec[:, np.newaxis]
        e = d_n_vec - y

        if self.two_path:
            e_f = d_n_vec - y_f
            e_f, e_b, y, y_b = self.transfer_logic(e_f, e, y_f, y)
            e = d_n_vec - y

        return X, e

    def compute_freq_xcorr(self, X, e):

        e_pad = np.concatenate((np.zeros((self.overlap, 1)), e), axis=0)

        E = fft(e_pad, n=self.n_fft, axis=0)

        self.P[self.P < 1e-4] = 1e-4
        grad = X.conj() * E / self.P

        return grad

    def gradient_constraint(self, grad):

        grad_1 = ifft(grad, n=self.n_fft, axis=0)
        grad_1[-self.hop_len :] = 0
        grad = fft(grad_1, n=self.n_fft, axis=0)

        return grad

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

        if self.constrain:
            grad = self.gradient_constraint(grad)

        if update:
            self.W = self.W + p * 2 * self.mu * grad  # update filter weights

        w_est = ifft(self.W, n=self.n_fft, axis=0)
        self.w = w_est[: self.filter_len, :]

        if fir_truncate is not None:
            w_shift = self.w.copy()
            w_shift[:fir_truncate] = 0.0
            w_shift[-fir_truncate:] = 0.0
            self.W = np.fft.rfft(w_shift, n=self.n_fft, axis=0)

        return e, self.w


def test_aic():
    """
    estimate ATF or RTF
    :return:
    """

    data = load_audio('DistantSpeech/adaptivefilter/wav/cleanspeech_reverb_ch3_rt60_110.wav')
    x = data[:, 0]  # to estimate rtf
    # x = load_audio('cleanspeech.wav')     # to estimate atf
    x = np.mean(data, axis=1)
    d = data[:, 1]

    filter_len = 1024

    delay = 8
    delay_obj = DelayObj(len(d), delay)

    d = delay_obj.delay(d)
    d = np.squeeze(d)

    block_len = filter_len

    fdaf = FastFreqLms(filter_len=filter_len, mu=0.1, alpha=0.8)

    output = np.zeros(len(x))

    for n in tqdm(range((len(x)))):
        if n < filter_len:
            continue
        if np.mod(n, block_len) == 0:
            output_n, w_fdaf = fdaf.update(x[n - filter_len : n], d[n - filter_len : n])
            output[n - filter_len : n] = np.squeeze(output_n)

    # save_audio('wav/aic_out.wav', output)
    plt.plot(output)
    plt.show()

    return output


def main(args):
    src = np.random.randn(8 * 16000)  # white noise, best for adaptive filter
    rir = load_audio('DistantSpeech/adaptivefilter/rir.wav')
    rir = rir[199:]
    rir = rir[:512, np.newaxis]

    # src = awgn(src, 30)
    print(src.shape)

    SNR = 20
    data_clean = conv(src, rir[:, 0])
    data = data_clean[: len(src)]
    data = awgn(data, SNR)

    filter_len = 512
    # hop_len = 512

    fdaf = FastFreqLms(
        filter_len=filter_len,
        mu=1e-1,
    )

    hop = fdaf.hop_len

    est_err_fdaf = np.zeros(len(data) - hop)

    eps = 1e-6

    w_fdaf = np.zeros((filter_len, 1))

    output = np.zeros((len(data), 1))

    for n in tqdm(range((len(src)))):
        if n == 0:
            continue
        if np.mod(n, hop) == 0:
            output[n - hop : n], w_fdaf = fdaf.update(src[n - hop : n], data[n - hop : n])
        est_err_fdaf[n - hop] = np.sum(np.abs(rir - w_fdaf[: len(rir)]) ** 2)

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
