import numpy as np
import argparse
from scipy.signal import convolve as conv
from matplotlib import pyplot as plt
from tqdm import tqdm
from DistantSpeech.beamformer.utils import load_audio
from DistantSpeech.adaptivefilter import BaseFilter
from DistantSpeech.adaptivefilter.SubbandAF import SubbandAF
from DistantSpeech.transform.subband import Subband


class SubbandLMS(SubbandAF):
    def __init__(
        self, filter_len=2, num_bands=512, mu=0.1, normalization=True, alpha=0.9, m=2, hop_length=None, input_td=False
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

    def update(self, x_n, d_n, alpha=1e-4, p=None):
        """subband LMS filter update function

        Parameters
        ----------
        x_n : np.array
            input signal, [samples, ]
        d_n : np.array
            expected signal, [samples, ]
        alpha : _type_, optional
            _description_, by default 1e-4
        p : float or np.array, optional
            update probability, float or [half_band, ], by default None

        Returns
        -------
        err : np.array,
            error output signal, if return time domain, [frame_len,] , else [half_band, ]
        self.W : complex np.array
            filter weights, [half_band, firter_len]
        """

        assert x_n.shape == d_n.shape, 'x_n and d_n must be same shape of [samples, ]'
        assert len(x_n.shape) == 1, 'x_n must be shape of [samples, ]'
        if p is not None:
            if not isinstance(p, float):
                if len(p.shape) == 2:
                    p = p[:, 0]
        else:
            p = np.ones((self.half_band,))

        self.return_td = False

        d_n = self.update_input_data(x_n, d_n, alpha=alpha, p=p)  # [self.half_band, ]

        # error signal
        filter_output = self.compute_filter_output(self.W, self.input_buffer)  # [self.half_band, ]
        if p is not None:
            assert p.shape[0] == self.half_band
            err = d_n - filter_output * p
        else:
            err = d_n - filter_output

        if self.norm:
            self.P = (
                self.alpha * self.P
                + (1 - self.alpha) * np.sum(self.input_buffer.conj() * self.input_buffer, axis=-1).real
            )
            grad = self.input_buffer * err[:, None].conj() / (self.P[:, None] + alpha)
        else:
            grad = self.input_buffer * err[:, None].conj()  # LMS

        self.update_coef(grad, p=p[:, None])

        if self.return_td:
            err = self.transform_d.synthesis(err)
        return err, self.W


def test(args):
    x = np.random.rand(16000 * 5)
    d = np.random.rand(16000 * 5)

    subbang_lms = SubbandLMS()

    for n in tqdm(range(len(x) - subbang_lms.transform_x.hop_length)):
        if np.mod(n, subbang_lms.transform_x.hop_length) == 0:
            input_vector = x[n : n + subbang_lms.transform_x.hop_length]
            d_vector = d[n : n + subbang_lms.transform_x.hop_length]
            err, w_rls = subbang_lms.update(input_vector, d_vector)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Feature Extractor")
    parser.add_argument("-l", "--listen", action='store_true', help="set to listen output")  # if set true
    parser.add_argument("-s", "--save", action='store_true', help="set to save output")  # if set true

    args = parser.parse_args()
    test(args)
