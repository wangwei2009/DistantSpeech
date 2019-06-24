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

    def getweights(self, a, weightType = 'DS', Rvv=None, Ryy=None, Diagonal = 1e-3):
        """
        compute beamformer weights

        """
        if weightType == 'src':
            weights = a
            weights[1:] = 0                             # output channel 1
        elif weightType == 'DS':
            weights = a/self.M                           # delay-and-sum weights
        elif weightType == 'MVDR':
            if Rvv is None:
                warnings.warn("Rvv not provided,using eye(M,M)\n")
                Rvv = np.eye([self.M,self.M])
            Fvv_k = (Rvv) + Diagonal * np.eye(self.M)  # Diagonal loading
            Fvv_k_inv = np.linalg.inv(Fvv_k)
            weights =  Fvv_k_inv @ a / \
                    (a.conj().T @ Fvv_k_inv @ a)  # MVDR weights
        elif weightType == 'TFGSC':
            # refer to Jingdong Chen, "Noncausal (Frequency-Domain) Optimal
            # Filters," in Microphone Array Signal Processing,page.134-135
            if Rvv is None:
                warnings.warn("Rvv not provided,using eye(M,M)\n")
                Rvv = np.eye([self.M,self.M])
            Fvv_k = (Rvv) + Diagonal * np.eye(self.M)  # Diagonal loading
            Fvv_k_inv = np.linalg.inv(Fvv_k)
            u = np.zeros((self.M,1))
            u[0] = 1
            temp = Fvv_k_inv @ Ryy
            weights =  (temp- np.eye(self.M))@u/ \
                    (np.trace(temp)-self.M)  # FD-TFGSC weights
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
    def beampattern(self, omega,H):
            """
            calculate directive index per frequency bin

            """
            half_bin = H.shape[1]
            r = 0.032
            angle = np.arange(0,360,360)
            beamout = np.zeros([360,half_bin])
            for az in range(0,360,1):
                tao = -1 * r * np.cos(0) * np.cos(az*np.pi/180 - self.gamma) / self.c
                tao = tao[:,np.newaxis]
                for k in range(0, half_bin):
                    a = np.exp(-1j * omega[k] * tao)
                    beamout[az,k] = np.abs(np.squeeze(H[:,k,np.newaxis].conj().T@a))

            return 10*np.log10(beamout)
