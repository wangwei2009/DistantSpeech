import numpy as np
import argparse
from scipy.signal import convolve as conv
from matplotlib import pyplot as plt
from tqdm import tqdm
from DistantSpeech.beamformer.utils import load_audio
from DistantSpeech.adaptivefilter import BaseFilter
from DistantSpeech.adaptivefilter.SubbandAF import SubbandAF
from DistantSpeech.transform.subband import Subband
from DistantSpeech.transform.transform import Transform


class SubbandLmsMc(SubbandAF):
    def __init__(
        self,
        filter_len=2,
        num_bands=512,
        channel=1,
        mu=0.1,
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

        self.M = channel

        self.input_buffer = np.zeros((self.half_band, filter_len, self.M), dtype=complex)

        self.W = np.zeros((self.half_band, filter_len, channel), dtype=complex)
        self.mu = np.ones((self.half_band, 1, self.M)) * mu

        self.transform_x = Transform(n_fft=num_bands, hop_length=self.hop_length, channel=channel)
        self.transform_d = Transform(n_fft=num_bands, hop_length=self.hop_length)

    def update_input(self, xt):
        """_summary_

        Parameters
        ----------
        xt : np.array
            buffer input data, [half_band, channel]

        Returns
        -------
        self.input_buffer : np.array
            buffered data, [half_band, filter_len, channel]
        """
        # update input buffer
        self.input_buffer[:, 1:, :] = self.input_buffer[:, :-1, :]
        self.input_buffer[:, 0, :] = xt

        return self.input_buffer

    def update_input_data(self, x_n, d_n, alpha=1e-4, p=None):
        """buffer input data

        Parameters
        ----------
        x_n : np.array
            input data, [samples, channel] or [half_band, channel]
        d_n : np.array
            expected signal
        alpha : _type_, optional
            _description_, by default 1e-4
        p : _type_, optional
            _description_, by default None

        Returns
        -------
        d_n : complex np.array
            expected data, [half_band, ]
        """
        if 'float' in str(x_n.dtype) and 'float' in str(d_n.dtype):
            # x_n = self.transform_x.analysis(x_n)
            # d_n = self.transform_d.analysis(d_n)
            x_n = self.transform_x.analysis(x_n)  # [half_band, 1, C]

            d_n = np.squeeze(self.transform_d.analysis(d_n))  # [half_band, ]
            self.return_td = True
        self.update_input(x_n[:, 0, :])

        return d_n

    def compute_filter_output(self, W, X):
        """compute subband filter response

        Parameters
        ----------
        W : complex np.array
            multichannel filter weights, , [self.half_band, filter_len, channel]
        X : complex np.array
            multichannel input data, [self.half_band, filter_len, channel]

        Returns
        -------
        filter_output : complex np.array
            filter response, [self.half_band,]
        """

        assert W.shape == X.shape

        filter_output = np.einsum('ijk,ijk->i', W.conj(), X)

        return filter_output

    def update_coef(self, grad, p=None):
        """update filter weights

        Parameters
        ----------
        grad : complex np.array
            filter gradient, [half_bands, filter_len, channel]
        p : np.array, optional
            update probability, [half_bands, 1], by default None

        Returns
        -------
        _type_
            _description_
        """

        if grad.ndim == 1:
            grad = grad[:, np.newaxis]
        if p is not None:
            self.W = self.W + 2 * self.mu * grad * p[..., None]
        else:
            self.W = self.W + 2 * self.mu * grad

        return self.W

    def update(self, x_n, d_n, alpha=1e-4, p=None):
        """_summary_

        Parameters
        ----------
        x_n : np.array
            multichannel input data, [samples, channel]
        d_n : np.array
            expected signal, [samples,] or [samples, channel]
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

        d_n = self.update_input_data(x_n, d_n, alpha=alpha, p=p)

        # error signal
        filter_output = self.compute_filter_output(self.W, self.input_buffer)
        if p is not None:
            assert p.shape[0] == self.half_band
            err = d_n - filter_output * np.squeeze(p)
        else:
            err = d_n - filter_output
        if self.norm:
            self.P = (
                self.alpha * self.P
                + (1 - self.alpha)
                * np.einsum('ijk, ijk -> i', self.input_buffer.conj(), self.input_buffer).real
                / self.M
            )
            grad = self.input_buffer * err[:, None, None].conj() / (self.P[:, None, None] + alpha)
        else:
            grad = self.input_buffer * err[:, None, None].conj()  # LMS

        self.update_coef(grad, p=p)

        if self.return_td:
            err = self.transform_d.synthesis(err)

        return err, self.W


def test(args):
    frame_len = 512
    M = 4
    x = np.random.rand(16000 * 5, M)
    d = np.random.rand(16000 * 5)

    subband_lms = SubbandLmsMc(filter_len=2, num_bands=frame_len * 2, hop_length=frame_len, channel=M)

    for n in tqdm(range(len(x) - subband_lms.transform_x.hop_length)):
        if np.mod(n, subband_lms.transform_x.hop_length) == 0:
            input_vector = x[n : n + subband_lms.transform_x.hop_length]
            d_vector = d[n : n + subband_lms.transform_x.hop_length]
            err, w_rls = subband_lms.update(input_vector, d_vector)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Feature Extractor")
    parser.add_argument("-l", "--listen", action='store_true', help="set to listen output")  # if set true
    parser.add_argument("-s", "--save", action='store_true', help="set to save output")  # if set true

    args = parser.parse_args()
    test(args)
