
import numpy as np

class NoiseEstimationBase(object):
    def __init__(self, nfft=320) -> None:
        super().__init__()

        self.nfft = nfft
        self.half_bin = int(self.nfft/2+1)
        self.lambda_d = np.zeros(self.half_bin)
        self.alpha_d = 0.95

        self.alpha_s = 0.8;
        self.delta_s = 5;
        self.alpha_p = 0.2;

        self.ell = 1;
        self.b = [0.25,0.5,0.25]

        self.S = np.zeros(self.half_bin)
        self.Smin = np.zeros(self.half_bin)
        self.Stmp = np.zeros(self.half_bin)
        self.p = np.zeros(self.half_bin)
        self.alpha_tilde = np.zeros(self.half_bin)

        self.L = 125

        self.frm_cnt = 0;

    def estimation(self, X):
        pass

    def update_noise_psd(self,Y: np.ndarray):

        self.alpha_tilde = self.alpha_d + (1-self.alpha_d)*self.p;            # eq 5,

        # eq 4,update noise spectrum
        self.lambda_d = self.alpha_tilde*self.lambda_d + (1-self.alpha_tilde)*Y;
