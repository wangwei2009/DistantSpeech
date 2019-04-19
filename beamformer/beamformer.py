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
            Fvv_k = (Rvv) + Diagonal * np.eye(self.M)  # Diagonal loading
            Fvv_k_inv = np.linalg.inv(Fvv_k)
            weights =  Fvv_k_inv @ a / \
                    (a.conj().T @ Fvv_k_inv @ a)  # MVDR weights
        else:
            raise ValueError('Unknown beamformer weights: %s' % weightType)
        return weights

    def calcWNG(self, ak, Hk):
            """
            calculate White Noise Gain per frequency bin

            """
            WNG = np.squeeze(np.abs(Hk.conj().T@ak)**2/ \
                                    np.real((Hk.conj().T@Hk)))

            return 10*np.log10(WNG)
    def calcDI(self, ak, Hk,Fvvk):
            """
            calculate directive index per frequency bin

            """
            WNG = np.squeeze(np.abs(Hk.conj().T@ak)**2/ \
                                    np.real((Hk.conj().T@Fvvk@Hk)))

            return 10*np.log10(WNG)
