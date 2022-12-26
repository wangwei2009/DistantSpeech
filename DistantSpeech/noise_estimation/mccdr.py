"""
Multi-Channel Speech Presence Probability
================

----------


.. [1] M. Taseska and E. A. P. Habets, "Nonstationary Noise PSD Matrix Estimation for Multichannel
    Blind Speech Extraction," in IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 25, no. 11, pp. 2223-2236, Nov. 2017, doi: 10.1109/TASLP.2017.2750239.

"""

import os

import numpy as np
from scipy.signal import convolve

from DistantSpeech.beamformer.MicArray import MicArray
from DistantSpeech.coherence.BinauralEnhancement import BinauralEnhancement
from DistantSpeech.noise_estimation.mcra import NoiseEstimationMCRA
from DistantSpeech.noise_estimation.mcspp_base import McSppBase


class McCDR(McSppBase):
    def __init__(self, nfft=256, channels=4) -> None:
        super().__init__()

        self.channels = channels

        self.nfft = nfft
        self.half_bin = int(self.nfft / 2 + 1)
        self.lambda_d = np.zeros(self.half_bin)
        self.alpha_d = 0.95

        self.alpha = 0.92

        self.alpha_s = 0.8
        self.delta_s = 5
        self.alpha_p = 0.2

        self.ell = 1
        self.b = [0.25, 0.5, 0.25]

        self.S = np.zeros(self.half_bin)
        self.Smin = np.zeros(self.half_bin)
        self.Stmp = np.zeros(self.half_bin)
        self.q = np.ones(self.half_bin) * 0.6
        self.p = np.zeros(self.half_bin)
        self.alpha_tilde = np.zeros(self.half_bin)

        self.Phi_yy = np.zeros((self.channels, self.channels, self.half_bin))
        self.Phi_vv = np.zeros((self.channels, self.channels, self.half_bin))
        self.Phi_xx = np.zeros((self.channels, self.channels, self.half_bin))

        self.xi = np.zeros(self.half_bin)
        self.gamma = np.zeros(self.half_bin)
        self.L = 125

        self.mcra = NoiseEstimationMCRA(nfft=self.nfft)

        self.MicArray = MicArray()
        self.Gamma_estimator = BinauralEnhancement(self.MicArray)
        self.Gamma = np.zeros(self.half_bin)

        self.frm_cnt = 0

    def estimate_psd(self, y, alpha):
        pass

    def compute_posterior_snr(self, y):
        pass

    def compute_prior_snr(self, y):
        pass

    def compute_q(self, y, q_max=0.99, q_min=0.01):

        l_min = 0.1
        l_max = 0.998
        c = 3
        rho = 2.5

        tmp = np.power(10, c * rho / 10)

        self.q_local = l_min + (l_max - l_min) * tmp / (tmp + np.power(self.Gamma), rho)

        self.mcra.estimation(np.abs(y[:, 0] * np.conj(y[:, 0])))

        self.q = np.sqrt(1 - self.mcra.p / 2)
        self.q = np.minimum(np.maximum(self.q, q_min), q_max)

        return self.q

    def compute_p(self, p_max=1.0, p_min=0.0):
        """
        compute posterior speech presence probability
        :param p_max:
        :param p_min:
        :return:
        """
        self.p = 1 / (1 + self.q / (1 - self.q) * (1 + self.xi) * np.exp(-1 * (self.gamma / (1 + self.xi))))
        self.p = np.minimum(np.maximum(self.p, p_min), p_max)

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

    def estimation(self, y):

        alpha = 0.6
        self.Gamma_estimator.update_CSD_PSD(y.transpose(), alpha=alpha)
        self.Gamma_estimator.updateMSC()

        for k in range(self.half_bin):

            self.Gamma[k] = np.real(
                (self.Gamma_estimator.Fvv[k, 0, 2] - self.Gamma_estimator.Fvv_est[k, 0, 2])
                / (self.Gamma_estimator.Fvv_est[k, 0, 2])
                - np.exp(-1j * np.angle(y[0, 2, k]))
            )

            self.Phi_yy[:, :, k] = self.alpha * self.Phi_yy[:, :, k] + (1 - self.alpha) * (
                np.conj(y[k : k + 1, :]).transpose() @ y[k : k + 1, :]
            )

            if self.frm_cnt < 100:
                self.Phi_vv[:, :, k] = self.Phi_yy[:, :, k]

            self.Phi_xx = self.Phi_yy - self.Phi_vv

            Phi_vv_inv = np.linalg.inv(self.Phi_vv[:, :, k] + np.eye(self.channels) * 1e-6)

            self.xi[k] = np.trace(Phi_vv_inv @ self.Phi_yy[:, :, k]) - self.channels
            self.xi[k] = np.maximum(self.xi[k], 1e-6)

            self.gamma[k] = (
                y[k : k + 1, :] @ Phi_vv_inv @ self.Phi_xx[:, :, k] @ Phi_vv_inv @ np.conj(y[k : k + 1, :]).transpose()
            )

        self.compute_q(y)
        self.compute_p(p_max=0.99, p_min=0.01)
        self.update_noise_psd(y, beta=1.0)

        self.frm_cnt = self.frm_cnt + 1

    def update_noise_psd(self, y: np.ndarray, beta=1.0):
        """
        update noise PSD using spp
        :param y: complex noisy signal vector, [half_bin, channel]
        :param beta:
        :return:
        """
        self.alpha_tilde = self.alpha_d + (1 - self.alpha_d) * self.p  # eq 5,

        # eq.17 in [1]
        for k in range(self.half_bin):
            self.Phi_vv[:, :, k] = self.alpha_tilde[k] * self.Phi_vv[:, :, k] + beta * (1 - self.alpha_tilde[k]) * (
                np.conj(y[k : k + 1, :]).transpose() @ y[k : k + 1, :]
            )


def main(args):
    from DistantSpeech.transform.transform import Transform
    from DistantSpeech.beamformer.utils import pmesh, load_wav
    from matplotlib import pyplot as plt
    import librosa
    import time
    from scipy.io import wavfile

    filepath = "../../example/test_audio/rec1/"  # [u1,u2,u3,y]
    # filepath = "./test_audio/rec1_mcra_gsc/"     # [y,u1,u2,u3]
    x, sr = load_wav(os.path.abspath(filepath))  # [channel, samples]
    sr = 16000
    r = 0.032
    c = 343

    frameLen = 256
    hop = frameLen / 2
    overlap = frameLen - hop
    nfft = 256
    c = 340
    r = 0.032
    fs = sr

    print(x.shape)
    channel = x.shape[0]

    transform = Transform(n_fft=512, hop_length=256, channel=channel)

    D = transform.stft(x.transpose())  # [F,T,Ch]
    Y, _ = transform.magphase(D, 2)
    print(Y.shape)
    pmesh(librosa.power_to_db(Y[:, :, -1]))
    plt.savefig('pmesh.png')

    mcspp = McSppBase(nfft=512, channels=4)
    noise_psd = np.zeros((Y.shape[0], Y.shape[1]))
    p = np.zeros((Y.shape[0], Y.shape[1]))
    Yout = np.zeros((Y.shape[0], Y.shape[1]), dtype=type(Y))
    y = np.zeros(x.shape[1])

    start = time.process_time()

    for n in range(Y.shape[1]):
        mcspp.estimation(Y[:, n, :])
        p[:, n] = mcspp.p
        # Yout[:, n] = D[:, n, 0] * omlsa_multi.G

    end = time.process_time()
    print(end - start)

    # y = transform.istft(Yout)

    # pmesh(librosa.power_to_db(noise_psd))
    # plt.savefig('noise_psd.png')

    pmesh(p)
    plt.savefig('p.png')

    plt.plot(y)
    plt.show()

    # save audio
    if args.save:
        wavfile.write('output_omlsa_multi4.wav', 16000, y)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Neural Feature Extractor")
    parser.add_argument("-l", "--listen", action='store_true', help="set to listen output")  # if set true
    parser.add_argument("-s", "--save", action='store_true', help="set to save output")  # if set true

    args = parser.parse_args()
    main(args)
