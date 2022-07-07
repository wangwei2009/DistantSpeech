import numpy as np
import argparse
from scipy.signal import convolve as conv
from matplotlib import pyplot as plt
from tqdm import tqdm
from DistantSpeech.beamformer.utils import load_audio
from DistantSpeech.adaptivefilter import BaseFilter
from DistantSpeech.transform.subband import Subband


class SubbandAF(BaseFilter):
    def __init__(
        self, filter_len=2, num_bands=512, mu=0.1, normalization=True, alpha=0.9, m=2, hop_length=None, input_td=False
    ):
        self.filter_len = filter_len
        half_band = int(num_bands / 2) + 1
        self.half_band = half_band

        self.W = np.zeros((half_band, filter_len), dtype=complex)
        self.mu = np.ones((half_band, 1)) * mu
        self.norm = normalization
        self.alpha = alpha

        self.input_buffer = np.zeros((half_band, filter_len), dtype=complex)
        self.P = np.zeros((half_band,))

        # r = int(n_fft / hop_length / 2)  # Decimation factor
        if hop_length is None:
            hop_length = int(num_bands / 2)
        self.transform_x = Subband(n_fft=num_bands, hop_length=hop_length, m=m)
        self.transform_d = Subband(n_fft=num_bands, hop_length=hop_length, m=m)

        self.return_td = False

    def update_input(self, xt):
        """

        :param xt:
        :return:
        """
        # update input buffer
        self.input_buffer[:, 1:] = self.input_buffer[:, :-1]
        self.input_buffer[:, 0] = xt

    def update_input_data(self, x_n, d_n, alpha=1e-4, p=None):
        if 'float' in str(x_n.dtype) and 'float' in str(d_n.dtype):
            x_n = self.transform_x.analysis(x_n)
            d_n = self.transform_d.analysis(d_n)
            self.return_td = True
        self.update_input(x_n)

        return d_n

    def update(self, x_n, d_n, alpha=1e-4, p=None):

        raise NotImplementedError

    def update_coef(self, grad, p=None):

        if grad.ndim == 1:
            grad = grad[:, np.newaxis]
        if p is not None:
            self.W = self.W + 2 * self.mu * grad * p
        else:
            self.W = self.W + 2 * self.mu * grad

        return self.W

    def compute_filter_output(self, W, X):

        assert W.shape == X.shape

        filter_output = np.einsum('ij,ij->i', W.conj(), X)

        return filter_output

    def filter(self, data, data_d):
        err = np.zeros(np.size(data))
        for n in range(len(data)):
            self.update_input(data[n])
            err[n] = self.update(data[n], data_d[n])

        return err


def main(args):

    subband_af = SubbandAF()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Feature Extractor")
    parser.add_argument("-l", "--listen", action='store_true', help="set to listen output")  # if set true
    parser.add_argument("-s", "--save", action='store_true', help="set to save output")  # if set true

    args = parser.parse_args()
    main(args)
