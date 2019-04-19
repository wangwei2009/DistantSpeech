from scipy.signal import windows
from scipy import signal
import numpy as np
from beamformer.GenNoiseMSC import gen_noise_msc
from beamformer.MicArray import MicArray
import warnings


class beamformer(object):
    """
    beamformer base class
    """
    def __init__(self):
        pass

    def getweights(self, a, weightType = 'DS', Rvv=None, Diagonal = 1e-3):
        """
        compute beamformer weights

        """
        if weightType == 'DS':
            weights = a/self.M                           # delay-and-sum weights
        elif weightType == 'MVDR':
            if Rvv is None:
                warnings.warn("Rvv not provided,using eye(M,M)\n")
                Rvv = np.eye([self.M,self.M])
            Fvv_k = np.mat((Rvv) + Diagonal * np.eye(self.M))  # Diagonal loading
            Fvv_k_inv = np.linalg.inv(Fvv_k)
            weights =  np.array(Fvv_k_inv * a) / \
                   np.array(a.conj().T * Fvv_k_inv * a)  # MVDR weights
        else:
            raise ValueError('Unknown beamformer weights: %s' % weightType)
        return weights

    def calcWNG(self, a, H):
            """
            calculate White Noise Gain

            """
            WNG = 0
            return 0
