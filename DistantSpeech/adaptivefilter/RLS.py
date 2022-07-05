"""
time-domain block-lms
High computation cost when long filter length

 Created by Wang wei
"""
import numpy as np
import argparse
from DistantSpeech.beamformer.utils import load_audio
from scipy.signal import convolve as conv
from matplotlib import pyplot as plt
from tqdm import tqdm
from DistantSpeech.adaptivefilter import BaseFilter, awgn


class Rls(BaseFilter):
    def __init__(self, filter_len=1024, mu=0.5, forgetting_factor=0.9998, delta=1e-3, normalization=True):
        BaseFilter.__init__(self, filter_len=filter_len, mu=mu)

        self.norm = normalization

        self.P = np.eye(self.filter_len) * 1 / delta
        self.forgetting_factor = forgetting_factor
        self.forgetting_factor_inv = 1.0 / self.forgetting_factor

    def update(self, x_n, d_n, alpha=1e-4):
        self.update_input(x_n)

        # prior error
        err = d_n - self.w.T @ self.input_buffer

        # gain vector
        num = self.P @ self.input_buffer
        kn = num / (self.forgetting_factor + self.input_buffer.T @ num)

        # update inversion matrix
        self.P = (self.P - kn @ self.input_buffer.T @ self.P) * self.forgetting_factor_inv

        grad = err * kn
        self.update_coef(grad)

        return err, self.w


def main(args):
    x = np.random.rand(160)
    d = np.random.rand(160)

    rls = Rls()

    for n in tqdm(range(len(x))):
        err, w_rls = rls.update(x[n], d[n])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Feature Extractor")
    parser.add_argument("-l", "--listen", action='store_true', help="set to listen output")  # if set true
    parser.add_argument("-s", "--save", action='store_true', help="set to save output")  # if set true

    args = parser.parse_args()
    main(args)
