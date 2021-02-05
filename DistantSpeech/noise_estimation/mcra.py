
import numpy as np
import argparse
import os
import time
from .NoiseEstimationBase import NoiseEstimationBase

class NoiseEstimationMCRA(NoiseEstimationBase):
    def __init__(self) -> None:
        super(NoiseEstimationMCRA, self).__init__()

    def estimation(self, Y: np.ndarray):

        assert  len(Y) == self.half_bin

        for k in range(self.half_bin-1):
            if self.frm_cnt == 0:
                self.Smin[k] = Y[k]
                self.Stmp[k] = Y[k]
                self.lambda_d[k] = Y[k]
                self.p[k] = 1.0
            else:
                Sf = Y[k-1]*self.b[0]+Y[k]*self.b[1]+Y[k+1]*self.b[2];    # eq 6,frequency smoothing
                self.S[k] = self.alpha_s*self.S[k]+(1-self.alpha_s)*Sf;                     # eq 7,time smoothing         

                self.Smin[k] = np.min(self.Smin[k],self.S[k]);                            # eq 8/9 minimal-tracking
                self.Stmp[k] = np.min(self.Stmp[k],self.S[k]);

                if self.ell%self.L==0:
                    self.Smin[k] = np.min(self.Stmp[k],self.S[k]);                        # eq 10/11
                    self.Stmp[k] = self.S[k];

                    self.ell = 0;                                                  # loop count

                Sr = self.S[k]/(self.Smin[k]+1e-6);

                if Sr > self.delta_s:
                    I = 1
                else:
                    I = 0

                self.p[k] = self.alpha_p*self.p[k] + (1-self.alpha_p)*I;                    # eq 14,updata speech presence probability     
                self.p[k] = max(min(self.p[k],1.0),0.0);

        self.update_noise_psd(Y)



def main(args):

    from DistantSpeech.transform.transform import stft, istft
    from DistantSpeech.beamformer.utils import mesh,pmesh,load_wav,load_pcm, visual

    filepath = "DistantSpeech/example/test_audio/rec1/"
    x,sr = load_wav(os.path.abspath(filepath))
    sr = 16000
    r = 0.032
    c = 343

    frameLen = 256
    hop = frameLen/2
    overlap = frameLen - hop
    nfft = 256
    c = 340
    r = 0.032
    fs = sr

    Y = stft(x)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Neural Feature Extractor")
    parser.add_argument("-l","--listen", action='store_true', help="set to listen output") # if set true
    parser.add_argument("-s","--save", action='store_true', help="set to save output") # if set true

    args = parser.parse_args()
    main(args)






