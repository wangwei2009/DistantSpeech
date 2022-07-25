"""
WPE dereverberation
==============

----------


.. [1] RLS-Based Adaptive Dereverberation Tracing Abrupt Position Change of Target Speaker.
.. [2] "Dereverberation for reverberation-robust microphone arrays," 21st European Signal Processing Conference (EUSIPCO 2013), 2013, pp. 1-5.
.. [3] Adaptive Multichannel Dereverberation for Automatic Speech Recognition

"""

from tkinter.tix import Tree
import numpy as np
import argparse
from scipy.signal import convolve as conv
from matplotlib import pyplot as plt
from tqdm import tqdm
from DistantSpeech.beamformer.utils import load_audio
from DistantSpeech.adaptivefilter.BaseFilter import BaseFilter
from DistantSpeech.adaptivefilter.SubbandAF import SubbandAF
from DistantSpeech.transform.subband import Subband
from DistantSpeech.transform.transform import Transform
from DistantSpeech.beamformer.utils import DelayFrames, DelaySamples


class Wpe(SubbandAF):
    def __init__(
        self,
        channels=2,
        filter_len=2,
        num_bands=512,
        forgetting_factor=0.998,
        delay=4,
        mu=0.5,
        normalization=True,
        alpha=0.9,
        m=2,
        hop_length=None,
        input_td=False,
    ):
        SubbandAF.__init__(
            self,
            filter_len=filter_len,
            num_bands=num_bands,
            mu=mu,
            normalization=normalization,
            alpha=alpha,
            m=m,
            hop_length=hop_length,
            input_td=input_td,
        )

        self.channels = channels

        self.input_buffer = np.zeros((self.half_band, self.channels, filter_len), dtype=complex)
        self.input_buffer_delayed = np.zeros((self.half_band, self.channels), dtype=complex)

        self.W = np.zeros((self.half_band, self.channels, self.channels * filter_len), dtype=complex)

        self.forgetting_factor = forgetting_factor
        self.forgetting_factor_inv = 1.0 / self.forgetting_factor

        self.transform_x = Subband(n_fft=num_bands, hop_length=self.hop_length, channel=self.channels)
        self.transform_d = Subband(n_fft=num_bands, hop_length=self.hop_length, channel=self.channels)

        self.P = np.zeros(
            (self.half_band, self.filter_len * self.channels, self.filter_len * self.channels), dtype=complex
        )
        for band in range(self.half_band):
            self.P[band, ...] = np.eye(self.filter_len * self.channels, dtype=complex) * 1e-3

        self.D = delay
        self.DelayObj = DelaySamples(self.hop_length, int(self.D * self.hop_length), channel=self.channels)

        self.var = np.zeros((self.half_band, 1))

    def buffer_input(self, xt):
        """_summary_

        Parameters
        ----------
        xt : complex np.array, [bins, ch]
            _description_
        input_buufer : complex np.array, [bins, ch*L]
            _description_

        Returns
        -------
        _type_
            _description_
        """
        # update input buffer
        if self.filter_len == 1:
            self.input_buffer[:, :, 0] = xt
        else:
            for ch in range(self.channels):
                self.input_buffer[:, ch, 1:] = self.input_buffer[:, ch, :-1]
                self.input_buffer[:, ch, 0] = xt[:, ch]

        return self.input_buffer

    def compute_filter_output(self, W, X):

        # assert W.shape == X.shape

        # filter_output = np.zeros((self.half_band, self.channels), dtype=complex)

        filter_output = np.einsum('kmi, ki->km', W.conj(), X)
        # print(filter_output.shape)

        return filter_output

    def delay_input(self, DelayObj, input_buffer):
        for k in range(self.half_band):
            input_buffer_k_delayed = DelayObj[k].delay(input_buffer[k, ...].T)
            self.input_buffer_delayed[k, ...] = input_buffer_k_delayed.T

        return self.input_buffer_delayed

    def delay_input_data(self, DelayObj, x_n):
        for k in range(self.half_band):
            x_n_k_delayed = DelayObj[k].delay(x_n)
            self.input_buffer_delayed[k, ...] = x_n_k_delayed

        return self.input_buffer_delayed

    def update(self, x_n, alpha=1e-4, p=None):
        """WPE update function

        Parameters
        ----------
        x_n : np.array, float64, [sapmes, ch]
            input data
        alpha : _type_, optional
            _description_, by default 1e-4
        p : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_
        """
        self.return_td = False

        x_n_td_delayed = self.DelayObj.delay(x_n)

        x_n_delayed, d_n = self.check_input_data(x_n_td_delayed, x_n)

        input_delayed_buffer = self.buffer_input(x_n_delayed)

        X = np.reshape(input_delayed_buffer, (self.half_band, -1))  # [k, C*N]

        filter_output = np.einsum('kmi, ki->km', self.W.conj(), X)

        # prior error
        err = d_n - filter_output  # [K,C]

        alpha = 0.98
        var_n = np.abs(np.einsum('ij, ij->i', d_n.conj(), d_n)) / self.channels
        self.var = alpha * self.var + (1 - alpha) * var_n[:, None]
        # self.var = alpha * self.var + (1 - alpha) * np.abs(d_n[:, 0:1] * d_n[:, 0:1].conj())

        if 0:
            # nlms
            for ch in range(self.channels):
                self.W[:, ch, :] = self.W[:, ch, :] + err[:, ch : ch + 1].conj() * X / (self.var + 1e-6) * self.mu
        else:
            # gain vector
            num = np.einsum('kij, kj->ki', self.P, X)  # [K, C*N, C*N] * [K, C*N] -> [K, C*N]
            kn = num / (
                # self.forgetting_factor * self.var
                # + np.einsum('ij, ij->i', X.conj(), num)[..., None]
                self.forgetting_factor * self.var
                + np.sum(X.conj() * num, axis=-1, keepdims=True)
            )  # [k, C*N]

            # update inversion matrix
            self.P = (
                self.P - np.einsum('ij,il,ilk->ijk', kn, X.conj(), self.P, optimize=True)
            ) * self.forgetting_factor_inv
            # self.P = (self.P - kn[..., None] @ X[:, None, :].conj() @ self.P) * self.forgetting_factor_inv

            for ch in range(self.channels):
                self.W[:, ch, :] = self.W[:, ch, :] + err[:, ch : ch + 1].conj() * kn

        if self.return_td:
            output = self.transform_d.synthesis(err[:, 0])

        return output, self.W


def main(args):

    ch = 4
    x = np.random.rand(16000 * 5, ch)

    n_fft = 256
    hop_length = 64
    wpe = Wpe(filter_len=2, delay=1, channels=ch, num_bands=n_fft, hop_length=hop_length)

    for n in tqdm(range(len(x) - hop_length)):
        if np.mod(n, hop_length) == 0:
            input_vector = x[n : n + hop_length, :]
            err, w_rls = wpe.update(input_vector)


def check_func1():
    a = np.random.rand(2, 3, 5)
    b = np.random.rand(2, 5)
    c = np.einsum('kmi,ki->km', a, b)
    print(c)

    filter_output = np.zeros((2, 3))
    for k in range(2):
        for ch in range(3):
            filter_output[k, ch] = np.sum(a[k, ch] * b[k])
    print(filter_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Feature Extractor")
    parser.add_argument("-l", "--listen", action='store_true', help="set to listen output")  # if set true
    parser.add_argument("-s", "--save", action='store_true', help="set to save output")  # if set true

    args = parser.parse_args()
    main(args)
