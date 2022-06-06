"""
Multi-Channel Speech Presence Probability
==============

----------


.. [1] M. Souden, J. Chen, J. Benesty and S. Affes, "Gaussian Model-Based Multichannel Speech Presence Probability,"
    in IEEE Transactions on Audio, Speech, and Language Processing, vol. 18, no. 5, pp. 1072-1077, July 2010,
    doi: 10.1109/TASL.2009.2035150.
   [2] Bagheri, S., Giacobello, D. (2019) Exploiting Multi-Channel Speech Presence Probability in Parametric
    Multi-Channel Wiener Filter. Proc. Interspeech 2019, 101-105, DOI: 10.21437/Interspeech.2019-2665

"""

from math import gamma
import os

import numpy as np
from scipy.signal import convolve
import soundfile as sf
from pesq import pesq
from pystoi.stoi import stoi

from DistantSpeech.noise_estimation.mcra import NoiseEstimationMCRA


class McSppBase(object):
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
        self.G_H1 = np.zeros(self.half_bin)
        self.G = np.zeros(self.half_bin)

        self.w = np.zeros((self.half_bin, self.channels), dtype=complex)

        self.Phi_yy = np.zeros((self.half_bin, self.channels, self.channels), dtype=complex)
        self.Phi_vv = np.zeros((self.half_bin, self.channels, self.channels), dtype=complex)
        self.Phi_vv_inv = np.zeros((self.half_bin, self.channels, self.channels), dtype=complex)
        self.Phi_xx = np.zeros((self.half_bin, self.channels, self.channels), dtype=complex)

        self.psd_yy = np.zeros((self.half_bin, self.channels, self.channels), dtype=np.complex128)

        self.xi = np.zeros(self.half_bin)
        self.gamma = np.zeros(self.half_bin)
        self.L = 125

        self.win = np.array([0.25, 0.5, 0.25])
        self.alpha_s = 0.8

        self.diagonal_eps = np.eye(self.channels) * 1e-6

        self.mcra = NoiseEstimationMCRA(nfft=self.nfft)
        self.mcra.L = 15

        self.frm_cnt = 0

    def estimate_psd(self, y, alpha):
        pass

    def estimate_noisy_psd(self, y, alpha=0.92):
        # [F,C] *[F,C]->[F,C,C]
        self.psd_yy = np.einsum('ij,il->ijl', y, y.conj())
        # smooth
        self.Phi_yy = alpha * self.Phi_yy + (1 - alpha) * self.psd_yy

        return self.Phi_yy

    def compute_posterior_snr(self, y):
        pass

    def compute_prior_snr(self, y):
        pass

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

        self.mcra.estimation(np.abs(y[:, 0] * np.conj(y[:, 0])))

        self.q = np.sqrt(1 - self.mcra.p / 2)
        # self.q = np.sqrt(1 - self.mcra.p)
        self.q = np.minimum(np.maximum(self.q, q_min), q_max)

        return self.q

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

    def compute_omlsa_weight(self, xi, p, Gmin=0.0631):
        """compute om-lsa gain given estimated xi and p

        Parameters
        ----------
        xi : np.ndarray, [bin,]
            prioir snr
        p : np.array, [bin,]
            posterior speech presence probability
        Gmin : float, optional
            gain floor, by default 0.0631
        """
        self.G_H1 = xi / (1 + xi)
        self.G = np.power(self.G_H1, p) * np.power(Gmin, (1 - p))
        self.G = np.maximum(np.minimum(self.G, 1), Gmin)
        self.G[:2] = 0

    def compute_pmwf_weight(self, xi, Rxx, Rvv_inv, Gmin=0.0631, beta=1):
        """compute parameterized multichannel non-causal Wiener filter
        refer to
            "On Optimal Frequency-Domain Multichannel Linear Filtering for Noise Reduction"

        Parameters
        ----------
        xi : np.array, [half_bin]
            prior snr
        Rxx : np.array, [M, M, bin]
            target psd matrix
        Rvv_inv : np.array, [M, M, bin]
            inversion of noise psd matrix
        Gmin : float, optional
            _description_, by default 0.0631
        beta : int, optional
            _description_, by default 1
        """
        u = np.zeros((self.half_bin, self.channels, 1))
        u[:, 0, 0] = 1
        self.w = (Rvv_inv @ Rxx @ u).squeeze() / (beta + xi[:, None])

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

    def estimation(self, y: np.array):
        """estimate psd and speech speech presence probability,
           for base method, we use real type for saving computation

        Parameters
        ----------
        y : complex np.array
            input signal, [bin, channel]
        """

        self.estimate_noisy_psd(y, alpha=self.alpha)

        self.Phi_xx = self.Phi_yy - self.Phi_vv

        diag_bin = np.broadcast_to(self.diagonal_eps, (self.half_bin, self.channels, self.channels))

        self.Phi_vv_inv.real = np.linalg.inv(self.Phi_vv.real + diag_bin)

        self.xi = np.trace(self.Phi_vv_inv.real @ self.Phi_xx.real, axis1=-2, axis2=-1)

        self.gamma = (
            y[:, None, :].conj() @ self.Phi_vv_inv.real @ self.Phi_xx.real @ self.Phi_vv_inv.real @ y[:, :, None]
        ).real.squeeze()

        self.xi = np.minimum(np.maximum(self.xi, 1e-6), 1e6)
        self.gamma = np.minimum(np.maximum(self.gamma, 1e-6), 1e6)

        self.compute_q(y)
        self.compute_p(p_max=0.99, p_min=0.01)
        self.update_noise_psd(y, psd_yy=self.psd_yy, beta=1.0)

        self.compute_pmwf_weight(self.xi, self.Phi_xx, self.Phi_vv_inv)

        self.frm_cnt = self.frm_cnt + 1

        return self.p

    def update_noise_psd(self, y: np.ndarray, psd_yy=None, beta=1.0):
        """update noise PSD using spp

        Parameters
        ----------
        y : np.ndarray, [half_bin, channels]
            input signal
        psd_yy : np.ndarray, [bin, M, M]
            noisy PSD matrix, by default None
        beta : float, optional
            _description_, by default 1.0
        """

        self.alpha_tilde = self.alpha_d + (1 - self.alpha_d) * self.p  # eq 5,
        # self.alpha_tilde[...] = 0.98

        # # eq.17 in [1]
        if psd_yy is not None:
            self.Phi_vv = (
                self.alpha_tilde[:, None, None] * self.Phi_vv + beta * (1 - self.alpha_tilde[:, None, None]) * psd_yy
            )
        else:
            for k in range(self.half_bin):
                self.Phi_vv[k] = self.alpha_tilde[k] * self.Phi_vv[k] + beta * (1 - self.alpha_tilde[k]) * (
                    y[k : k + 1, :].T @ y[k : k + 1, :].conj()
                )


def main(args):
    from DistantSpeech.transform.transform import Transform
    from DistantSpeech.beamformer.utils import pmesh, load_wav
    from matplotlib import pyplot as plt
    import librosa
    import time
    from scipy.io import wavfile

    filepath = "example/test_audio/p232/"  # [u1,u2,u3,y]
    # filepath = "./test_audio/rec1_mcra_gsc/"     # [y,u1,u2,u3]
    x, sr = load_wav(os.path.abspath(filepath))  # [channel, samples]
    audio_file = 'example/test_audio/SSB19180462#noise-free-sound-0665#4.70_3.93_3.00_2.46_1.72_218.8051_42.3619_0.3466#-2#3.4485138549309093#0.5211135715489874.wav'
    wave_data, sr = sf.read(os.path.abspath(audio_file))  # (frames x channels)
    x = wave_data.transpose()
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
    # pmesh(librosa.power_to_db(Y[:, :, -1]))
    # plt.savefig('pmesh.png')

    mcspp = McSppBase(nfft=512, channels=channel)
    noise_psd = np.zeros((Y.shape[0], Y.shape[1]))
    p = np.zeros((Y.shape[0], Y.shape[1]))
    Yout = np.zeros((Y.shape[0], Y.shape[1]), dtype=type(Y))
    y = np.zeros(x.shape[1])

    start = time.process_time()

    for n in range(Y.shape[1]):
        mcspp.estimation(D[:, n, :])
        p[:, n] = mcspp.p
        Yout[:, n] = D[:, n, 0] * mcspp.G

    end = time.process_time()
    print(end - start)

    y = transform.istft(Yout)

    if args.eval:
        ref_path = os.path.abspath(os.path.join(root_path, 'ref', audio_file))
        ref, sr = sf.read(ref_path)
        assert fs == sr
        if len(ref.shape) >= 2:
            ref = ref[:, 0]

        nsy = wave_data[:, 0]
        enh = y[256:]
        nsy = nsy[: len(enh)]
        ref = ref[: len(enh)]

        summary = {
            'ref_pesq': pesq(sr, ref, nsy, 'wb'),
            'enh_pesq': pesq(sr, ref, enh, 'wb'),
            'ref_stoi': stoi(ref, nsy, sr, extended=False),
            'enh_stoi': stoi(ref, enh, sr, extended=False),
            'ref_estoi': stoi(ref, nsy, sr, extended=True),
            'enh_estoi': stoi(ref, enh, sr, extended=True),
        }
        for key in summary.keys():
            print('{}:{}'.format(key, summary[key]))

    # pmesh(librosa.power_to_db(noise_psd))
    # plt.savefig('noise_psd.png')

    # pmesh(p)
    # plt.savefig('p.png')
    #
    # plt.plot(y)
    # plt.show()

    # save audio
    if args.save:
        audio = (y * np.iinfo(np.int16).max).astype(np.int16)
        wavfile.write('./output_mcsppbase_p232_3.wav', 16000, audio)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Neural Feature Extractor")
    parser.add_argument("-l", "--listen", action='store_true', help="set to listen output")  # if set true
    parser.add_argument("-s", "--save", action='store_true', help="set to save output")  # if set true
    parser.add_argument("-e", "--eval", action='store_true', help="set to save output")  # if set true

    args = parser.parse_args()
    main(args)
