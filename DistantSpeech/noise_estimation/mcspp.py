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
from DistantSpeech.noise_estimation import McMcra
from DistantSpeech.beamformer.MicArray import MicArray

EPSILON = np.finfo(np.float32).eps


def condition_covariance(x, gamma):
    """see https://stt.msu.edu/users/mauryaas/Ashwini_JPEN.pdf (2.3)"""
    scale = gamma * np.trace(x) / x.shape[-1]
    scaled_eye = np.eye(x.shape[-1]) * scale
    return (x + scaled_eye) / (1 + gamma)


def condition_covariance_bin(x, gamma):
    half_bin, M, M = x.shape
    for k in range(half_bin):
        scale = gamma * np.trace(x[k]) / x[k].shape[-1]
        scaled_eye = np.eye(x[k].shape[-1]) * scale
        x[k] = (x[k] + scaled_eye) / (1 + gamma)

    return x


class McSpp(McSppBase):
    def __init__(self, nfft=256, channels=4, mic_array=None) -> None:
        super().__init__(nfft=nfft, channels=channels)

        self.mcra = NoiseEstimationMCRA(nfft=self.nfft)
        # self.mcra = McMcra(nfft=nfft, channels=channels)
        self.mcra.L = 10
        # self.mcra.init_frame = 50

        self.xi_last = np.zeros(self.half_bin)
        self.Phi_vv_inv_bin = np.zeros((self.half_bin, self.channels, self.channels), dtype=complex)

        self.frm_cnt = 0

        self.alpha_d = 0.92
        self.alpha = 0.92

        if mic_array is not None:
            self.mic_array = mic_array
            self.steer_vector = self.mic_array.steering_vector(look_direction=30)

    def estimate_psd(self, y, alpha):
        pass

    def compute_posterior_snr(self, y):
        pass

    def compute_prior_snr(self, y):
        pass

    def compute_p(self, p_max=1.0, p_min=0.0, alpha_p=0):
        """compute posterior speech presence probability

        Parameters
        ----------
        p_max : float, optional
            max p, by default 1.0
        p_min : float, optional
            min p, by default 0.0
        alpha_p : int, optional
            average factor, by default 0 indicate no average
        """
        p = 1 / (1 + self.q / (1 - self.q) * (1 + self.xi) * np.exp(-1 * (self.gamma / (1 + self.xi))))
        self.p = alpha_p * self.p + (1 - alpha_p) * p
        self.p = np.minimum(np.maximum(self.p, p_min), p_max)

    def compute_q(self, y: np.array, q_max=0.99, q_min=0.01):
        """priori speech absence probability

        Parameters
        ----------
        y : np.array, [bin, channel]
            input signal
        q_max : float, optional
            max q, by default 0.99
        q_min : float, optional
            min q, by default 0.01

        Returns
        -------
        np.array

        """
        self.mcra.estimation(np.abs(y[:, 0] * y[:, 0].conj()))

        # self.q = np.sqrt(1 - self.mcra.p / 2)
        self.q = np.sqrt(1 - self.mcra.p)
        self.q = np.minimum(np.maximum(self.q, q_min), q_max)

        if np.mean(self.q[8:32]) > 0.8:
            self.q[:] = 0.999999

        # if self.frm_cnt > 800:
        #     self.q[:] = 0.9999999
        # else:
        #     self.q[:] = self.mcra.q
        # self.q[:] = self.mcra.q[:]

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
        w = Rvv_inv @ Rxx @ u / (beta + xi)
        self.w[k : k + 1, :] = w.T

    def compute_weight_from_steer_vector(self, xi, Rxx, Rvv_inv, k, Gmin=0.0631, beta=1):
        w = (
            Rvv_inv
            @ self.steer_vector[:, k : k + 1]
            / (self.steer_vector[:, k : k + 1].conj().T @ Rvv_inv @ self.steer_vector[:, k : k + 1])
        )
        # print(w.shape)
        self.w[k : k + 1, :] = w.T

        return w

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

    def check_data(self, Phi_xx, diag_value=1e-8):
        diag = np.eye(self.channels) * diag_value
        diag_bin = np.broadcast_to(diag, (self.half_bin, self.channels, self.channels))
        for k in range(self.half_bin):
            if (np.diag(self.Phi_xx[k]) < 0).any():
                self.Phi_vv[k] = self.Phi_yy[k] - diag_bin[k] * 2
                self.Phi_xx[k] = diag_bin[k]
                self.Phi_vv_inv[k] = np.linalg.inv(self.Phi_yy[k] + diag_bin[k])

        self.Phi_xx = self.Phi_yy - self.Phi_vv

        xx = np.trace(self.Phi_xx, axis1=-2, axis2=-1).real
        index = np.where(xx < 1e-1)
        self.Phi_vv[index] = self.Phi_yy[index] - diag_bin[index] * 2
        self.Phi_xx[index] = diag_bin[index]

        return self.Phi_xx

    def estimation_core(self, y, psd_yy=None, diag_value=1e-8):
        diag = np.eye(self.channels) * diag_value
        diag_bin = np.broadcast_to(diag, (self.half_bin, self.channels, self.channels))

        # self.Phi_vv = condition_covariance(self.Phi_vv, 1e-6)
        # self.Phi_vv = condition_covariance_bin(self.Phi_vv, 1e-6)

        # Make sure matrix is hermitian
        self.Phi_vv = 0.5 * (self.Phi_vv + np.conj(self.Phi_vv.swapaxes(-1, -2)))

        self.Phi_xx = self.Phi_yy - self.Phi_vv

        self.Phi_vv_inv = np.linalg.inv(self.Phi_vv + diag_bin)

        # self.xi = np.trace(self.Phi_vv_inv @ self.Phi_xx, axis1=-2, axis2=-1).real
        self.xi = np.trace(np.real(self.Phi_vv_inv @ self.Phi_yy), axis1=-2, axis2=-1) - self.channels
        if self.frm_cnt == 1164:
            print('xi[58,1164] = {}'.format(self.xi[58]))
        index = np.where(self.xi < 0)
        # self.Phi_vv_inv[index] = np.linalg.inv(self.Phi_yy[index] + diag_bin[index])

        if self.frm_cnt < 5:
            self.Phi_vv_inv[index] = np.linalg.inv(self.Phi_yy[index] + diag_bin[index])
        else:
            self.Phi_vv_inv[index] = np.linalg.inv(self.Phi_yy[index])
        # self.Phi_vv[index] = self.Phi_yy[index]
        self.xi = np.trace(np.real(self.Phi_vv_inv @ self.Phi_yy), axis1=-2, axis2=-1) - self.channels

        self.xi = np.minimum(np.maximum(self.xi, 1e-6), 1e8)

        self.gamma = (
            y[:, None, :].conj() @ self.Phi_vv_inv @ self.Phi_yy @ self.Phi_vv_inv @ y[:, :, None]
            - y[:, None, :].conj() @ self.Phi_vv_inv @ y[:, :, None]
        ).real.squeeze()
        self.gamma = np.minimum(np.maximum(self.gamma, 1e-6), 1e8)

        # index = np.where(self.gamma / self.xi > 20)
        # self.gamma[index] = self.xi[index] * 20

        self.compute_p(alpha_p=0)
        # self.update_noise_psd(y, psd_yy=psd_yy, beta=1.0)

    def estimation(self, y, diag_value=1e-4, repeat=False):
        """mcspp estimation function

        Parameters
        ----------
        y : np.array
            input data, [half_bin, channels]
        """

        M = self.channels
        diag_value = np.eye(M) * diag_value

        psd_yy = np.einsum('ij,il->ijl', y, y.conj())

        self.Phi_yy = self.alpha * self.Phi_yy + (1 - self.alpha) * psd_yy

        self.compute_q(y, q_max=0.99, q_min=1e-2)
        # self.q = np.sqrt(np.sqrt(self.q))
        # self.q = self.q / 2

        # self.p = np.sqrt(1 - self.q)

        if self.frm_cnt < 10:
            self.Phi_vv = self.Phi_yy
            self.q[:] = 0.99

        self.estimation_core(y, psd_yy=psd_yy, diag_value=diag_value)
        self.update_noise_psd(y, psd_yy=psd_yy, beta=1.0)
        if repeat:
            # self.q = np.sqrt(1 - self.p)
            self.estimation_core(y, psd_yy=psd_yy, diag_value=diag_value)

        self.compute_pmwf_weight(self.xi, self.Phi_xx, self.Phi_vv_inv, beta=10)
        # for k in range(self.half_bin):
        #     self.compute_weight_from_steer_vector(self.xi, self.Phi_xx, self.Phi_vv_inv[k], k)

        # diag = np.eye(self.channels) * diag_value
        # diag_bin = np.broadcast_to(diag, (self.half_bin, self.channels, self.channels))

        # noise_psd_matrix = condition_covariance(self.Phi_vv, 1e-6)
        # noise_psd_matrix /= np.trace(noise_psd_matrix, axis1=-2, axis2=-1)[..., None, None]
        # self.w = self.get_gev_vector(self.Phi_xx, noise_psd_matrix)
        # self.w = self.phase_correction(self.w)
        # self.w = self.blind_analytic_normalization(self.w, noise_psd_matrix)
        # if normalization:
        #     W_gev = blind_analytic_normalization(W_gev, noise_psd_matrix)

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
