"""
Multi-Channel Speech Presence Probability
================

----------


.. [1] M. Taseska and E. A. P. Habets, "Nonstationary Noise PSD Matrix Estimation for Multichannel
    Blind Speech Extraction," in IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 25, no. 11, pp. 2223-2236, Nov. 2017, doi: 10.1109/TASLP.2017.2750239.
.. [2] A. Schwarz and W. Kellermann, "Coherent-to-Diffuse Power Ratio Estimation for Dereverberation," in IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 23, no. 6, pp. 1006-1018, June 2015, doi: 10.1109/TASLP.2015.2418571.

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
        super().__init__(nfft=nfft, channels=channels)

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

        self.MicArray = MicArray(arrayType="circular", r=0.032, M=self.channels)
        self.Gamma_estimator = BinauralEnhancement(self.MicArray, frameLen=nfft, nfft=nfft)
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

    def estimate_ddr(self, y, unbias=True):
        """estimate multichannel Coherent-to-Diffusion Ratio

        Parameters
        ----------
        y : complex np.array
            input signal, [half_bin, channels]
        """
        alpha = 0.9
        self.Gamma_estimator.update_CSD_PSD(y, alpha=alpha)
        self.Gamma_estimator.updateMSC()

        if unbias:
            # eq.[25] in [2]
            Fn = self.Gamma_estimator.Fvv[:, 0, 2]
            Fn2 = Fn**2
            Fx = self.Gamma_estimator.Fvv_est[:, 0, 2]
            Fx2 = np.abs(Fx) ** 2
            Gamma = (Fn * Fx.real - Fx2 - np.sqrt(Fn2 * Fx.real**2 - Fn2 * Fx2 + Fn2 - 2 * Fn * Fx.real + Fx2)) / (
                np.minimum(Fx2 - 1, -1e-3)
            )
        else:
            # eq.[41], in [1]
            Gamma = np.real(
                (self.Gamma_estimator.Fvv[:, 0, 2] - self.Gamma_estimator.Fvv_est[:, 0, 2])
                / (
                    (self.Gamma_estimator.Fvv_est[:, 0, 2])
                    - np.exp(1j * np.angle(self.Gamma_estimator.Pxij[:, 1]) + 1e-3)
                )
            )

        Gamma = Gamma**2

        Gamma[Gamma > 1] = 1
        Gamma[Gamma < 0] = 1e-3

        return Gamma

    def estimation(self, y, theta=135):
        """estimate multichannel Coherent-to-Diffusion Ratio

        Parameters
        ----------
        y : complex np.array
            input signal, [half_bin, channels]
        """
        Gamma = self.estimate_ddr(y)

        return Gamma

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
    from DistantSpeech.beamformer.utils import load_audio as audioread
    from matplotlib import pyplot as plt
    import librosa
    import time
    from scipy.io import wavfile

    array_data = []
    for n in range(2, 6):
        filename = '/home/wangwei/work/DistantSpeech/samples/audio_samples/xmos/rec1/音轨-{}.wav'.format(n)
        # filename = '/home/wangwei/work/DistantSpeech/samples/audio_samples/xmos/meeting/1/wav/音轨-{}.wav'.format(n)
        # filename = '/home/wangwei/work/DistantSpeech/samples/audio_samples/xmos/aioffice/1/ch4/音轨-{}.wav'.format(n)
        # filename = '/home/wangwei/work/DistantSpeech/samples/audio_samples/xmos/meeting/2/ch4//音轨-{}.wav'.format(n)
        # filename = '/home/wangwei/work/DistantSpeech/samples/audio_samples/xmos/anechoic/2/ch4/音轨-{}.wav'.format(n)
        # filename = '/home/wangwei/work/DistantSpeech/samples/audio_samples/xmos/meeting/2/ch4/音轨-{}.wav'.format(n)
        # filename = '/home/wangwei/work/DistantSpeech/samples/audio_samples/xmos/anechoic/1/ch4/音轨-{}.wav'.format(n)
        # filename = '/home/wangwei/work/DistantSpeech/samples/audio_samples/xmos/office/1/ch4/音轨-{}.wav'.format(n)
        filename = '/home/wangwei/work/DistantSpeech/samples/audio_samples/xmos/aioffice/5/ch4/音轨-{}.wav'.format(n)
        data_ch = audioread(filename)
        array_data.append(data_ch)
    x = np.array(array_data).T
    print(x.shape)
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
    channel = x.shape[1]

    transform = Transform(n_fft=512, hop_length=256, channel=channel)

    D = transform.stft(x)  # [F,T,Ch]
    Y, _ = transform.magphase(D, 2)
    print(Y.shape)
    pmesh(librosa.power_to_db(Y[:, :, -1]))
    plt.savefig('pmesh.png')

    mcspp = McCDR(nfft=512, channels=4)
    noise_psd = np.zeros((Y.shape[0], Y.shape[1]))
    p = np.zeros((Y.shape[0], Y.shape[1]))
    Yout = np.zeros((Y.shape[0], Y.shape[1]), dtype=type(Y))
    y = np.zeros(x.shape[0])

    start = time.process_time()

    for n in range(Y.shape[1]):
        p[:, n] = mcspp.estimation(Y[:, n, :])
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
