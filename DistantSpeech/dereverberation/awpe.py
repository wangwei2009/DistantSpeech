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
        self.input_buffer_delayed = np.zeros((self.half_band, self.channels, filter_len), dtype=complex)

        self.W = np.zeros((self.half_band, self.channels, self.channels * filter_len), dtype=complex)

        self.forgetting_factor = forgetting_factor
        self.forgetting_factor_inv = 1.0 / self.forgetting_factor

        self.transform_x = Transform(n_fft=num_bands, hop_length=self.hop_length, channel=self.channels)
        self.transform_d = Transform(n_fft=num_bands, hop_length=self.hop_length, channel=self.channels)

        self.P = np.zeros(
            (self.half_band, self.filter_len * self.channels, self.filter_len * self.channels), dtype=complex
        )
        for band in range(self.half_band):
            self.P[band, ...] = np.eye(self.filter_len * self.channels, dtype=complex) / 1e-3

        self.DelayObj = []
        self.D = 2
        for k in range(self.half_band):
            self.DelayObj.append(DelaySamples(self.filter_len, self.D, channel=self.channels, dtype=complex))

    def update_input(self, xt):
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

    def update(self, x_n, alpha=1e-4, p=None):
        self.return_td = False

        d_n = self.update_input_data(x_n, x_n, alpha=alpha, p=p)

        input_buffer_delayed = self.delay_input(self.DelayObj, self.input_buffer)

        X = np.reshape(input_buffer_delayed, (self.half_band, -1))  # [k, C*N]

        filter_output = self.compute_filter_output(self.W, X)  # [K,C]

        # prior error
        err = d_n - filter_output  # [K,C]

        # gain vector
        num = np.einsum('kij, kj->ki', self.P, X.conj())  # [K, C*N, C*N] * [K, C*N] -> [K, C*N]
        kn = num / (
            # self.forgetting_factor + np.einsum('ijk, ijk->ik', self.input_buffer[..., None], num)
            self.forgetting_factor
            + np.sum(X * num, axis=-1, keepdims=True)
        )  # [k, C*N]

        # update inversion matrix
        self.P = (self.P - kn[..., None] @ X[:, None, :].conj() @ self.P) * self.forgetting_factor_inv

        for ch in range(self.channels):
            self.W[:, ch, ch * self.filter_len : (ch + 1) * self.filter_len] = (
                self.W[:, ch, ch * self.filter_len : (ch + 1) * self.filter_len]
                + err[:, ch : ch + 1] * kn[:, ch * self.filter_len : (ch + 1) * self.filter_len]
            )

        if self.return_td:
            err = self.transform_d.synthesis(err[:, 0])

        return err, self.W


def main(args):

    x = np.random.rand(16000 * 5, 2)
    d = np.random.rand(16000 * 5, 2)

    wpe = Wpe(filter_len=3)

    for n in tqdm(range(len(x) - wpe.transform_x.hop_length)):
        if np.mod(n, wpe.transform_x.hop_length) == 0:
            input_vector = x[n : n + wpe.transform_x.hop_length, :]
            d_vector = d[n : n + wpe.transform_x.hop_length, :]
            err, w_rls = wpe.update(input_vector, d_vector)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Feature Extractor")
    parser.add_argument("-l", "--listen", action='store_true', help="set to listen output")  # if set true
    parser.add_argument("-s", "--save", action='store_true', help="set to save output")  # if set true

    args = parser.parse_args()
    main(args)
