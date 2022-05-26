"""
Multi-Channel Speech Presence Probability
==============

----------


.. [1] M. Souden, J. Chen, J. Benesty, and S. Affes, An integrated solution for online multichannel noise tracking and
    reduction, IEEE Trans. Audio, Speech, Lang. Process., vol. 19, no. 7, pp. 2159–2169, Sep. 2011.
   [2] Bagheri, S., Giacobello, D. (2019) Exploiting Multi-Channel Speech Presence Probability in Parametric
    Multi-Channel Wiener Filter. Proc. Interspeech 2019, 101-105, DOI: 10.21437/Interspeech.2019-2665

"""

import os

import numpy as np
from scipy.signal import convolve
import soundfile as sf

from DistantSpeech.noise_estimation import NoiseEstimationMCRA, MCRA2, McSppBase
from DistantSpeech.beamformer.utils import load_pcm


class McSpp(McSppBase):
    def __init__(self, nfft=256, channels=4) -> None:
        super().__init__(nfft=nfft, channels=channels)

        # self.mcra = MCRA2(nfft=self.nfft)
        self.mcra.L = 10

        self.w = np.zeros((self.channels, self.half_bin), dtype=complex)
        self.xi_last = np.zeros(self.half_bin)

        self.frm_cnt = 0

    def estimate_psd(self, y, alpha):
        pass

    def compute_posterior_snr(self, y):
        pass

    def compute_prior_snr(self, y):
        pass

    def compute_q_local(self, y, Phi_vv_inv, Phi_yy, k, q_max=0.99, q_min=0.01):
        # eq.10,
        self.psi[k] = np.real(y @ Phi_vv_inv @ np.conj(y).transpose())
        self.psi_tilde[k] = np.real(np.trace(Phi_vv_inv @ Phi_yy))

        if self.psi[k] >= self.psi_0 or self.psi_tilde[k] > self.psi_tilde_0:
            self.q_local[k] = q_min
        elif self.psi_tilde[k] < self.M:
            self.q_local[k] = q_max
        else:
            self.q_local[k] = (self.psi_tilde_0 - self.psi_tilde[k]) / (self.psi_tilde_0 - self.M)

            self.q_local[k] = np.minimum(np.maximum(self.q_local[k], q_min), q_max)

        return self.q_local[k]

    def compute_weight_k(self, xi, Rxx, Rvv_inv, k, Gmin=0.0631, beta=1):
        u = np.zeros((self.channels, 1))
        u[0, 0] = 1
        self.w[:, k : k + 1] = Rvv_inv @ Rxx @ u / (beta + xi)

    def smooth_psd(self, x, previous_x, win, alpha):
        """
        smooth spectrum in frequency and time
        :param x: current x
        :param previous_x: last time x
        :param win: smooth window
        :param alpha: smooth factor
        :return: smoothed x
        """
        w = len(win)

        # smoothing in frequency
        smoothed_f = convolve(x, win)
        smoothed_f_val = smoothed_f[int((w - 1) / 2) : int(-((w - 1) / 2))]

        # smoothing in time
        smoothed_x = alpha * previous_x + (1 - alpha) * smoothed_f_val

        return smoothed_x

    def estimation_core(self, y, psd_yy=None, diag_value=1e-8):
        diag = np.eye(self.channels) * diag_value
        self.Phi_xx = self.Phi_yy - self.Phi_vv

        for k in range(self.half_bin):

            Phi_vv_inv = np.linalg.inv(self.Phi_vv[:, :, k] + diag)

            self.xi[k] = np.real(np.trace(Phi_vv_inv @ self.Phi_xx[:, :, k]))
            if self.frm_cnt > 1:
                if self.xi[k] < 1e-6:
                    self.xi[k] = self.xi_last[k]

            self.xi[k] = np.minimum(np.maximum(self.xi[k], 1e-6), 1e8)

            self.gamma[k] = np.abs(
                y[k : k + 1, :].conj() @ Phi_vv_inv @ self.Phi_xx[:, :, k] @ Phi_vv_inv @ y[k : k + 1, :].T
            )
            self.gamma[k] = np.minimum(np.maximum(self.gamma[k], 1e-6), 1e8)

            self.compute_weight_k(self.xi[k], self.Phi_xx[:, :, k], Phi_vv_inv, k, beta=1)

        self.compute_p(alpha_p=0)
        self.update_noise_psd(y, psd_yy=psd_yy, beta=1.0)

    def estimation(self, y, diag=1e-4, repeat=False):
        """mcspp estimation function

        Parameters
        ----------
        y : np.array
            input data, [half_bin, channels]
        """

        M = self.channels
        diag_value = np.eye(M) * diag

        psd_yy = np.einsum('ij,il->ijl', y, y.conj())

        self.Phi_yy = self.alpha * self.Phi_yy + (1 - self.alpha) * np.transpose(psd_yy, (1, 2, 0))

        self.compute_q(y, q_max=0.99, q_min=1e-2)
        self.q = np.sqrt(np.sqrt(self.q))
        # self.q = self.q / 2

        # self.p = np.sqrt(1 - self.q)

        if self.frm_cnt < 10:
            self.Phi_vv = self.Phi_yy
            self.q[:] = 0.99

        self.estimation_core(y, psd_yy=psd_yy, diag_value=1e-8)
        if repeat:
            self.q = np.sqrt(1 - self.p)
            self.estimation_core(y, psd_yy=psd_yy, diag_value=1e-8)

        self.xi_last[:] = self.xi  # .copy()

        self.frm_cnt = self.frm_cnt + 1

        return self.p


def main(args):

    from DistantSpeech.transform.transform import Transform
    from DistantSpeech.noise_estimation.mcspp import McSpp

    channel = 6
    sample_length = 16000 * 10
    array_data = np.random.rand(channel, sample_length)
    mcspp = McSpp(nfft=512, channels=channel)
    transform = Transform(n_fft=512, hop_length=256, channel=channel)
    D = transform.stft(np.transpose(array_data))
    for n in range(D.shape[1]):
        y = D[:, n, :]  # [half_bin, M]
        mcspp.estimation(y)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Neural Feature Extractor")
    parser.add_argument("-l", "--listen", action='store_true', help="set to listen output")  # if set true
    parser.add_argument("-s", "--save", action='store_true', help="set to save output")  # if set true
    parser.add_argument("-e", "--eval", action='store_true', help="set to save output")  # if set true

    args = parser.parse_args()
    main(args)
