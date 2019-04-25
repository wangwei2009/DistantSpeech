from scipy.signal import windows
from scipy import signal
import numpy as np
from beamformer.GenNoiseMSC import gen_noise_msc


class MicArray(object):

    def __init__(self, arrayType = 'circular',r = 0.032,M = 4):


        self.c = 343
        self.r = r
        self.fs = 16000
        self.M = M
        self.angle = np.array([197, 0]) / 180 * np.pi
        self.gamma = np.arange(0,360,int(360/self.M))* np.pi / 180


    def getMVDRweight(self,a,Rvv,Diagonal = 1e-3):
        """
        compute MVDR weights
               Rvv^-1*a
         W = ------------
              a'*Rvv^-1*a

        """
        Fvv_k = np.mat((Rvv) + Diagonal * np.eye(self.M))  # Diagonal loading
        Fvv_k_inv = np.linalg.inv(Fvv_k)
        return np.array(Fvv_k_inv * a) / \
                  np.array(a.conj().T * Fvv_k_inv * a)  # MVDR weights



