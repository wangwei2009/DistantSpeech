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
from DistantSpeech.adaptivefilter import FastFreqLms


class AdaptiveInterferenceCancellation(FastFreqLms):
    def __init__(
        self,
        filter_len=128,
        hop_len=None,
        win_len=None,
        mu=0.01,
        constrain=True,
        weight_norm=False,
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

        self.weight_norm = weight_norm

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

        if self.weight_norm:
            maxnorm = 0.003
            norm = np.sum(np.abs(self.W) ** 2) / self.n_fft / self.n_fft
            if norm > maxnorm:
                norm = np.sqrt(maxnorm / norm)
            else:
                norm = 1.0
        else:
            norm = 1.0

        if self.constrain:
            W_1 = ifft(self.W, n=self.n_fft, axis=0) * norm
            W_1[-self.hop_len :] = 0
            self.W = fft(W_1, n=self.n_fft, axis=0)

        w_est = ifft(self.W, n=self.n_fft, axis=0)
        self.w = w_est[: self.filter_len, :]

        if fir_truncate is not None:
            w_shift = self.w.copy()
            w_shift[:fir_truncate] = 0.0
            w_shift[-fir_truncate:] = 0.0
            self.W = np.fft.rfft(w_shift * norm, n=self.n_fft, axis=0)

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
