import numpy as np
import argparse
from scipy.signal import convolve as conv
from matplotlib import pyplot as plt
from tqdm import tqdm
from DistantSpeech.beamformer.utils import load_audio
from DistantSpeech.adaptivefilter.BaseFilter import BaseFilter
from DistantSpeech.adaptivefilter.SubbandAF import SubbandAF
from DistantSpeech.transform.subband import Subband


class SubbandRLS(SubbandAF):
    def __init__(
        self,
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

        self.forgetting_factor = forgetting_factor
        self.forgetting_factor_inv = 1.0 / self.forgetting_factor

        self.P = np.zeros((self.half_band, self.filter_len, self.filter_len), dtype=complex)
        for band in range(self.half_band):
            self.P[band, ...] = np.eye(self.filter_len, dtype=complex) / 1e-3

    def update(self, x_n, d_n, alpha=1e-4, p=None):
        self.return_td = False

        d_n = self.update_input_data(x_n, d_n, alpha=alpha, p=p)

        filter_output = self.compute_filter_output(self.W, self.input_buffer)

        # prior error
        err = d_n - filter_output

        # gain vector
        num = self.P @ self.input_buffer[:, :, np.newaxis]  # [k, N, N] @ [K, N, 1] = [K, N, 1]
        kn = num[..., 0] / (
            # self.forgetting_factor + np.einsum('ijk, ijk->ik', self.input_buffer[..., None], num)
            self.forgetting_factor
            + np.sum(self.input_buffer.conj() * num[..., 0], axis=-1, keepdims=True)
        )  # [k, N]

        # update inversion matrix
        self.P = (self.P - kn[..., None] @ self.input_buffer[:, None, :].conj() @ self.P) * self.forgetting_factor_inv

        grad = err[:, None].conj() * kn
        self.update_coef(grad)

        if self.return_td:
            err = self.transform_d.synthesis(err)

        return err, self.W


def main(args):

    x = np.random.rand(16000 * 5)
    d = np.random.rand(16000 * 5)

    subbang_rls = SubbandRLS(filter_len=1)

    for n in tqdm(range(len(x) - subbang_rls.transform_x.hop_length)):
        if np.mod(n, subbang_rls.transform_x.hop_length) == 0:
            input_vector = x[n : n + subbang_rls.transform_x.hop_length]
            d_vector = d[n : n + subbang_rls.transform_x.hop_length]
            err, w_rls = subbang_rls.update(input_vector, d_vector)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Feature Extractor")
    parser.add_argument("-l", "--listen", action='store_true', help="set to listen output")  # if set true
    parser.add_argument("-s", "--save", action='store_true', help="set to save output")  # if set true

    args = parser.parse_args()
    main(args)
