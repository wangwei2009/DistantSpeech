"""
coherence-based speech enhancement Algorithms
================

implement some coherence-based dual-mic noise reduction algorithms

see reference below to get more details

----------


.. [1] Cohen, I., Gannot, S. & Berdugo, B. An Integrated Real-Time Beamforming and Postfiltering System for
    Nonstationary Noise Environments.EURASIP J. Adv. Signal Process. 2003, 936861 (2003).

.. [2] Yousefian, N., Loizou, P. C., & Hansen, J. H. L. (2014). A coherence-based noise
    reduction algorithm for binaural hearing aids. Speech Communication, 58, 101–110


"""
import argparse
import os

import numpy as np

from DistantSpeech.noise_estimation.NoiseEstimationBase import NoiseEstimationBase
from DistantSpeech.noise_estimation.mcra import NoiseEstimationMCRA


class NsOmlsaMulti(NoiseEstimationBase):
    def __init__(self, nfft=256, M=4, cal_weights=False) -> None:
        super(NsOmlsaMulti, self).__init__(nfft=nfft)

        self.G_H1 = np.ones(self.half_bin)
        self.G = np.ones(self.half_bin)
        self.Gmin = -12
        self.Gmin = np.power(10, (self.Gmin / 10))

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
        self.q_min = 1e-6
        self.q_max = 0.9999998

        self.alpha_d = 0.85  # smooth factor for p

        self.xi_hat = np.ones(self.half_bin)

        self.first_frame = 1
        self.M = M
        self.noise_est_ref = []

        self.noise_est_fixed = NoiseEstimationMCRA(nfft=self.nfft)  # initialize noise estimator
        for ch in range(M - 1):
            self.noise_est_ref.append(NoiseEstimationMCRA(nfft=self.nfft))

        self.win = np.array([0.25, 0.5, 0.25])
        self.alpha_s = 0.8

        self.cal_weights = cal_weights

    def estimation(self, y: np.ndarray, u: np.ndarray):
        """
        multi-channel omlsa noise estimation
        :param y: length of half_bin, beamformer output
        :param u: [half_bin, M-1], M-1 channels of block matrix output
        :return:
        """

        assert len(y) == self.half_bin

        self.MU_Y = self.noise_est_fixed.estimation(y)
        for ch in range(self.M - 1):
            self.MU_U[ch, :] = self.noise_est_ref[ch].estimation(u[:, ch])

        if self.first_frame == 1:
            self.first_frame = 0
            self.lambda_d = y

            self.zeta_Y = y
            for ch in range(self.M - 1):
                self.zeta_U[ch, :] = u[:, ch]
        else:

            alpha = 0.921

            self.zeta_Y = self.smooth_psd(y, self.zeta_Y, self.win, self.alpha_s)  # Eq 21
            for ch in range(self.M - 1):
                self.zeta_U[ch, :] = self.smooth_psd(u[:, ch], self.zeta_U[ch, :], self.win, self.alpha_s)

            self.LAMBDA_Y = self.zeta_Y / (self.MU_Y + 1e-6)
            self.LAMBDA_U = np.max(self.zeta_U / (self.MU_U + 1e-6), axis=0)

            # Eq.6 The transient beam - to - reference ratio(TBRR)
            eps = 0.01
            self.Omega = np.maximum((self.zeta_Y - self.MU_Y), 1e-6) / (
                np.maximum(np.max(self.zeta_U - self.MU_U, axis=0), eps * self.MU_Y) + 1e-6
            )
            self.Omega = np.maximum(self.Omega, 0.1)
            self.Omega = np.minimum(self.Omega, 100)

            Bmin = 1.66
            # Eq.27 posteriori SNR at the beamformer output
            self.gamma_s = np.minimum(y / (self.MU_Y * Bmin + 1e-6), 100)

            gamma_high = 0.1 * np.power(10, 2)
            gamma_low = 1
            Omega_high = 3
            Omega_low = 0.3
            # Eq.29, The a priori signal absence probability
            for k in range(self.half_bin):
                if self.gamma_s[k] < gamma_low or self.Omega[k] < Omega_low:
                    self.q_hat[k] = 1
                else:
                    self.q_hat[k] = max(
                        (gamma_high - self.gamma_s[k]) / (gamma_high - gamma_low),
                        (Omega_high - self.Omega[k]) / (Omega_high - Omega_low),
                    )
                self.q_hat[k] = min(max(self.q_hat[k], self.q_min), self.q_max)

            gamma_pre = self.gamma.copy()
            # posteriori SNR
            self.gamma = y / np.maximum(self.lambda_d, 1e-10)

            # Eq 30, priori SNR
            self.xi_hat = alpha * np.power(self.G_H1, 2) * gamma_pre + (1 - alpha) * np.maximum(self.gamma - 1, 0)

            #
            nu = self.gamma * self.xi_hat / (1 + self.xi_hat)

            # Eq 31, the spectral gain function of the LSA estimator when the % signal is surely present
            # self.G_H1 = self.xi_hat / (1 + self.xi_hat) * np.exp(0.5 * e1(nu))
            self.G_H1 = self.xi_hat / (1 + self.xi_hat)

            # Eq 28, the signal presence probability
            self.p = 1 / (1 + self.q_hat / (1 - self.q_hat) * (1 + self.xi_hat) * np.exp(-1 * nu))

            self.update_noise_psd(y, beta=1.47)

            # Eq.35, OMLSA gain function
            if self.cal_weights:
                self.G = np.power(self.G_H1, self.p) * np.power(self.Gmin, (1 - self.p))
                self.G = np.maximum(np.minimum(self.G, 1), self.Gmin)

            return self.lambda_d


def main(args):
    from DistantSpeech.transform.transform import Transform
    from DistantSpeech.beamformer.utils import pmesh, load_wav
    from matplotlib import pyplot as plt
    import librosa
    import time
    from scipy.io import wavfile

    # filepath = "./test_audio/rec1/"  # [u1,u2,u3,y]
    # filepath = "./test_audio/rec1_mcra_gsc/"  # [y,u1,u2,u3]
    # x, sr = load_wav(os.path.abspath(filepath))  # [channel, samples]

    x = np.random.rand(4, 16000 * 5)
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

    omlsa_multi = NsOmlsaMulti(nfft=512, cal_weights=True)
    noise_psd = np.zeros((Y.shape[0], Y.shape[1]))
    p = np.zeros((Y.shape[0], Y.shape[1]))
    Yout = np.zeros((Y.shape[0], Y.shape[1]), dtype=type(Y))
    y = np.zeros(x.shape[1])

    start = time.process_time()

    for n in range(Y.shape[1]):
        omlsa_multi.estimation(Y[:, n, 0], Y[:, n, 1:])
        noise_psd[:, n] = omlsa_multi.lambda_d
        p[:, n] = omlsa_multi.p
        Yout[:, n] = D[:, n, 0] * omlsa_multi.G

    end = time.process_time()
    print(end - start)

    y = transform.istft(Yout[..., None])

    pmesh(librosa.power_to_db(noise_psd))
    plt.savefig('noise_psd.png')

    pmesh(p)
    plt.savefig('p.png')

    plt.plot(y)
    plt.show()

    # save audio
    if args.save:
        wavfile.write('output_omlsa_multi4.wav', 16000, y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Feature Extractor")
    parser.add_argument("-l", "--listen", action='store_true', help="set to listen output")  # if set true
    parser.add_argument("-s", "--save", action='store_true', help="set to save output")  # if set true

    args = parser.parse_args()
    main(args)
