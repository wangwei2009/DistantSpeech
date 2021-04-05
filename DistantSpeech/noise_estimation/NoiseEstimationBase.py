import numpy as np
from scipy.signal import convolve


class NoiseEstimationBase(object):
    def __init__(self, nfft=256) -> None:
        super().__init__()

        self.nfft = nfft
        self.half_bin = int(self.nfft / 2 + 1)
        self.lambda_d = np.zeros(self.half_bin)
        self.alpha_d = 0.95

        self.alpha_s = 0.8
        self.delta_s = 5
        self.alpha_p = 0.2

        self.ell = 1
        self.b = [0.25, 0.5, 0.25]

        self.S = np.zeros(self.half_bin)
        self.Smin = np.zeros(self.half_bin)
        self.Stmp = np.zeros(self.half_bin)
        self.p = np.zeros(self.half_bin)
        self.alpha_tilde = np.zeros(self.half_bin)

        self.L = 12

        self.frm_cnt = 0

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
        smoothed_f_val = smoothed_f[int((w - 1) / 2):int(-((w - 1) / 2))]

        # smoothing in time
        smoothed_x = alpha * previous_x + (1 - alpha) * smoothed_f_val

        return smoothed_x

    def estimation(self, X):
        pass

    def update_noise_psd(self, Y: np.ndarray, beta=1.0):
        self.alpha_tilde = self.alpha_d + (1 - self.alpha_d) * self.p  # eq 5,

        # eq 4,update noise spectrum
        self.lambda_d = self.alpha_tilde * self.lambda_d + beta * (1 - self.alpha_tilde) * Y
