"""

==============

----------


.. [1] Rangachari, S. and Loizou, P. (2006). A noise estimation algorithm  for
    highly non-stationary environments. Speech Communication, 28,220-231

"""
import argparse
import os

import numpy as np
import soundfile as sf
from DistantSpeech.noise_estimation.NoiseEstimationBase import NoiseEstimationBase


class MCRA2(NoiseEstimationBase):
    def __init__(self, nfft=256) -> None:
        super(MCRA2, self).__init__(nfft=nfft)
        self.Sf = np.zeros(self.half_bin)

    def estimation(self, Y: np.ndarray):

        assert len(Y) == self.half_bin

        for k in range(self.half_bin - 1):
            if self.frm_cnt == 0:
                self.Smin[k] = Y[k]
                self.Stmp[k] = Y[k]
                self.lambda_d[k] = Y[k]
                self.p[k] = 1.0
            else:
                S_pre = self.S
                self.Sf[k] = Y[k - 1] * self.b[0] + Y[k] * self.b[1] + Y[k + 1] * self.b[2]  # eq 6,frequency smoothing
                self.S[k] = self.alpha_s * self.S[k] + (1 - self.alpha_s) * self.Sf[k]  # eq 7,time smoothing

                gamma = 0.998
                beta = 0.8
                eta = 0.7
                # eq 3 minimal - tracking
                if self.Smin[k] < self.S[k]:
                    self.Smin[k] = gamma * self.Smin[k] + (1 - gamma) / (1 - beta) * (self.S[k] - beta * S_pre[k])
                else:
                    self.Smin[k] = self.S[k]

                Sr = self.S[k] / (self.Smin[k] + 1e-6)

                if Sr > self.delta_s:
                    I = 1
                else:
                    I = 0

                self.p[k] = (
                    self.alpha_p * self.p[k] + (1 - self.alpha_p) * I
                )  # eq 14,updata speech presence probability
                self.p[k] = max(min(self.p[k], 1.0), 0.0)

        self.frm_cnt = self.frm_cnt + 1
        self.lambda_d[self.half_bin - 1] = 1e-8
        self.ell = self.ell + 1
        self.update_noise_psd(Y)

        return self.lambda_d


def main(args):
    from DistantSpeech.transform.transform import Transform
    from DistantSpeech.beamformer.utils import pmesh, load_wav
    from matplotlib import pyplot as plt
    import librosa

    filepath = "/home/wangwei/work/DistantSpeech/example/test_audio/rec1/"
    x, sr = load_wav(os.path.abspath(filepath))  # [channel,samples]

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

    transform = Transform(n_fft=320, hop_length=160)

    D = transform.stft(x[0, :])
    Y, _ = transform.magphase(D, 2)
    Y = np.squeeze(Y)
    print(Y.shape)
    pmesh(librosa.power_to_db(Y))
    plt.savefig('pmesh.png')

    mcra = MCRA2(nfft=320)
    noise_psd = np.zeros(Y.shape)
    p = np.zeros(Y.shape)
    for n in range(Y.shape[1]):
        mcra.estimation(Y[:, n])
        noise_psd[:, n] = mcra.lambda_d
        p[:, n] = mcra.p

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
