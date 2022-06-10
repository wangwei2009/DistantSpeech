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

from DistantSpeech.noise_estimation.mcra import NoiseEstimationMCRA
from DistantSpeech.noise_estimation.mcspp_base import McSppBase


class McMcra(McSppBase):
    def __init__(self, nfft=256, channels=4) -> None:
        super().__init__()

        self.channels = channels
        self.M = self.channels

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

        self.psi = np.zeros(self.half_bin)
        self.psi_tilde = np.zeros(self.half_bin)
        self.psi_global = np.zeros(self.half_bin)
        self.psi_frame = []
        self.q = np.ones(self.half_bin) * 0.6
        self.q_local = np.ones(self.half_bin) * 0.999
        self.q_global = np.ones(self.half_bin)
        self.q_frame = []
        self.q_max = 0.99
        self.q_min = 0

        self.psi_0 = 100
        self.psi_tilde_0 = 100

        self.p = np.zeros(self.half_bin)
        self.alpha_tilde = np.zeros(self.half_bin)
        self.G_H1 = np.zeros(self.half_bin)
        self.G = np.zeros(self.half_bin)

        self.Phi_yy = np.zeros((self.channels, self.channels, self.half_bin))
        self.Phi_vv = np.zeros((self.channels, self.channels, self.half_bin))
        self.Phi_vv_inv = np.zeros((self.channels, self.channels, self.half_bin))
        self.Phi_xx = np.zeros((self.channels, self.channels, self.half_bin))

        self.xi = np.zeros(self.half_bin)
        self.gamma = np.zeros(self.half_bin)
        self.L = 125

        self.mcra = NoiseEstimationMCRA(nfft=self.nfft)

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

    def compute_q(self, y):
        """

        :param y:
        :param Phi_vv_inv:
        :param Phi_yy:
        :param k:
        :param q_max:
        :param q_min:
        :return:
        """

        self.psi_global = self.smooth_psd(self.psi, self.psi_global, self.win, 0)
        frame_range = np.array([60, 2000])
        frame_range = frame_range * self.nfft / 16000
        psi_frame = np.mean(self.psi_global[int(frame_range[0]) : int(frame_range[1])])
        self.psi_frame.append(psi_frame)
        for k in range(self.half_bin):
            if self.psi_global[k] < self.psi_0:
                self.q_global[k] = self.q_max
            else:
                self.q_global[k] = self.q_min
        if psi_frame < self.psi_0 / 8:
            q_frame = self.q_max
        else:
            q_frame = self.q_min

        # self.q_frame.append(q_frame)

        # self.q = self.q_local * q_frame
        # self.q = self.q_local * self.q_global * q_frame
        self.q = self.q_local
        # self.q[:] = self.q_local * q_frame

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

    def compute_weight(self, xi, Gmin=0.0631):
        self.G_H1 = self.xi / (1 + self.xi)
        self.G = np.power(self.G_H1, self.p) * np.power(Gmin, (1 - self.p))
        self.G = np.maximum(np.minimum(self.G, 1), Gmin)
        self.G[:2] = 0

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

        for k in range(self.half_bin):
            self.Phi_yy[:, :, k] = self.alpha * self.Phi_yy[:, :, k] + (1 - self.alpha) * np.real(
                (np.conj(y[k : k + 1, :]).transpose() @ y[k : k + 1, :])
            )

            if self.frm_cnt < 5:
                self.Phi_vv[:, :, k] = self.Phi_yy[:, :, k]

            self.Phi_xx = self.Phi_yy - self.Phi_vv

            Phi_vv_inv = np.linalg.inv(self.Phi_vv[:, :, k] + np.eye(self.channels) * 1e-6)

            self.xi[k] = np.trace(Phi_vv_inv @ self.Phi_yy[:, :, k]) - self.channels
            self.xi[k] = np.minimum(np.maximum(self.xi[k], 1e-6), 1e6)

            self.gamma[k] = np.real(
                y[k : k + 1, :].conj() @ Phi_vv_inv @ self.Phi_xx[:, :, k] @ Phi_vv_inv @ y[k : k + 1, :].T
            )
            self.gamma[k] = np.minimum(np.maximum(self.gamma[k], 1e-6), 1e6)

            self.q_local[k] = self.compute_q_local(y[k : k + 1, :], Phi_vv_inv, self.Phi_yy[:, :, k], k)

        self.compute_q(y)
        self.compute_p(p_max=0.99, p_min=0.01)
        self.update_noise_psd(y, beta=1.0)
        self.compute_weight(self.xi)

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
            self.Phi_vv[:, :, k] = np.real(
                self.alpha_tilde[k] * self.Phi_vv[:, :, k]
                + beta * (1 - self.alpha_tilde[k]) * (np.conj(y[k : k + 1, :]).transpose() @ y[k : k + 1, :])
            )


def main(args):
    from DistantSpeech.transform.transform import Transform
    from DistantSpeech.beamformer.utils import pmesh, load_wav
    from matplotlib import pyplot as plt
    import librosa
    import time
    from scipy.io import wavfile

    filepath = "example/test_audio/rec1/"  # [u1,u2,u3,y]
    # filepath = "./test_audio/rec1_mcra_gsc/"     # [y,u1,u2,u3]
    x, sr = load_wav(os.path.abspath(filepath))  # [channel, samples]
    sr = 16000
    r = 0.032
    c = 343

    x, sr = librosa.load('/home/wangwei/work/DistantSpeech/example/test_audio/sim/mix3/mix.wav', sr=None, mono=False)

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

    mcspp = McMcra(nfft=512, channels=channel)
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
        from pesq import pesq
        from pystoi.stoi import stoi

        ref_path = './path_to_ref'
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

    pmesh(p)
    plt.savefig('p.png')
    np.save('q.npz', mcspp.q)

    plt.plot(y)
    plt.show()

    # save audio
    if args.save:
        audio = (y * np.iinfo(np.int16).max).astype(np.int16)
        wavfile.write('output/output_mc_mcra2.wav', 16000, audio)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Neural Feature Extractor")
    parser.add_argument("-l", "--listen", action='store_true', help="set to listen output")  # if set true
    parser.add_argument("-s", "--save", action='store_true', help="set to save output")  # if set true
    parser.add_argument("-e", "--eval", action='store_true', help="set to save output")  # if set true

    args = parser.parse_args()
    main(args)
