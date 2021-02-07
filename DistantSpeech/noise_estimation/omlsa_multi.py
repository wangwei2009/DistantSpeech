import argparse
import os

import numpy as np
from mpmath import expint
from scipy.signal import windows

from DistantSpeech.noise_estimation.NoiseEstimationBase import NoiseEstimationBase
from DistantSpeech.noise_estimation.mcra import NoiseEstimationMCRA


class NsOmlsaMulti(NoiseEstimationBase):
    def __init__(self, nfft=256, M=4) -> None:
        super(NsOmlsaMulti, self).__init__(nfft=nfft)

        self.G_H1 = np.ones(self.half_bin)
        self.G = np.ones(self.half_bin)

        self.gamma = np.ones(self.half_bin)
        self.zeta_Y = np.ones(self.half_bin)  # smoothed fixed bf
        self.zeta_U = np.zeros((M - 1, self.half_bin))  # smoothed ref
        self.MU_Y = np.ones(self.half_bin)  # estimated noise from fixed bf channel
        self.MU_U = np.zeros((M - 1, self.half_bin))  # estimated noise from ref channel
        self.lambda_hat_d = np.ones(self.half_bin)

        self.gamma = np.ones(self.half_bin)  # posteriori SNR for fixed bf output
        self.LAMBDA_Y = np.ones(self.half_bin)  # posteriori SNR
        self.LAMBDA_U = np.ones(self.half_bin)  #

        self.Omega = np.ones(self.half_bin)
        self.gamma_s = np.ones(self.half_bin)

        self.q_hat = np.ones(self.half_bin)

        self.xi_hat = np.ones(self.half_bin)

        self.first_frame = 1
        self.M = M
        self.noise_est_ref = []

        self.noise_est_fixed = NoiseEstimationMCRA(nfft=self.nfft)  # initialize noise estimator
        for ch in range(M - 1):
            self.noise_est_ref.append(NoiseEstimationMCRA(nfft=self.nfft))

    def estimation(self, y: np.ndarray, u: np.ndarray):

        assert len(y) == self.half_bin

        if self.first_frame == 1:
            self.first_frame = 0
        else:
            self.MU_Y = self.noise_est_fixed.estimation(y)
            for ch in range(self.M - 1):
                self.MU_U[ch] = self.noise_est_ref[ch].estimation(u[ch, :])

            win = windows.hanning(3)
            alpha = 0.92

            self.zeta_Y = self.smooth_psd(y, self.zeta_Y, win, alpha)  # Eq 21
            for ch in range(self.M - 1):
                self.zeta_U = self.smooth_psd(u[ch, :], self.zeta_U, win, alpha)

            self.LAMBDA_Y = self.zeta_Y / self.MU_Y
            self.LAMBDA_U = np.max(self.zeta_U / self.MU_U, axis=0)

            # Eq.6 The transient beam - to - reference ratio(TBRR)
            eps = 0.01
            self.Omega = np.max((self.zeta_Y - self.MU_Y), 0) / \
                         np.max(np.max(self.zeta_U - self.MU_U, axis=0), eps * self.MU_Y)
            self.Omega = np.maximum(self.Omega, 0.1)
            self.Omega = np.minimum(self.Omega, 100)

            Bmin = 1.66
            # Eq.27 posteriori SNR at the beamformer output
            self.gamma_s = np.minimum(y / self.MU_Y * Bmin, 100)

            gamma_high = 0.1 * np.power(10, 2)
            gamma_low = 1
            Omega_low = 0.3
            Omega_high = 3
            # Eq.29, The a priori signal absence probability
            for k in range(self.half_bin):
                if self.gamma_s[k] < gamma_low or self.Omega[k] < Omega_low:
                    self.q_hat[k] = 1
                else:
                    self.q_hat[k] = np.maximum((gamma_high - self.gamma_s[k]) / (gamma_high - gamma_low),
                                               (Omega_high - self.Omega[k]) / (Omega_high - Omega_low))
                    self.q_hat[k] = min(max(self.q_hat[k], 0), 1)

            # posteriori SNR
            self.gamma = y / np.maximum(self.lambda_d, 1e-10)

            # Eq 30, priori SNR
            self.xi_hat = alpha * np.power(self.G_H1, 2) * self.gamma + (1 - alpha) * np.maximum(self.gamma - 1, 0)

            #
            nu = self.gamma * self.xi_hat / (1 + self.xi_hat)

            # Eq 31, the spectral gain function of the LSA estimator when the % signal is surely present
            self.G_H1 = self.xi_hat / (1 + self.xi_hat) * np.exp(0.5 * expint(nu))

            # Eq 28, the signal presence probability
            self.p = 1 / (1 + self.q_hat / (1 - self.q_hat) * (1 + self.xi_hat) * np.exp(-1 * nu))

            self.update_noise_psd(y)

            # # Eq.35, OMLSA gain function
            # self.G = np.real(np.pow(self.G_H1,self.p) * Gmin ^ (1 - self.p))
            # G(k, L) = min(G(k, L), 1);

            return self.lambda_d


def main(args):
    from DistantSpeech.transform.transform import Transform
    from DistantSpeech.beamformer.utils import pmesh, load_wav
    from matplotlib import pyplot as plt
    import librosa

    filepath = "./test_audio/rec1/"
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

    transform = Transform(n_fft=320, hop_length=160, channel=channel)

    D = transform.stft(x.transpose())     # [F,T,Ch]
    Y, _ = transform.magphase(D, 2)
    print(Y.shape)
    pmesh(librosa.power_to_db(Y[:, :, -1]))
    plt.savefig('pmesh.png')

    omlsa_multi = NsOmlsaMulti(nfft=320)
    noise_psd = np.zeros(Y.shape)
    p = np.zeros(Y.shape)
    for n in range(Y.shape[1]):
        omlsa_multi.estimation(Y[:, n, :])
        noise_psd[:, n] = omlsa_multi.lambda_d
        p[:, n] = omlsa_multi.p

    pmesh(librosa.power_to_db(noise_psd))
    plt.savefig('noise_psd.png')

    pmesh(p)
    plt.savefig('p.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Feature Extractor")
    parser.add_argument("-l", "--listen", action='store_true', help="set to listen output")  # if set true
    parser.add_argument("-s", "--save", action='store_true', help="set to save output")  # if set true

    args = parser.parse_args()
    main(args)
