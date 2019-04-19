from scipy.signal import windows
from scipy import signal
import numpy as np
from beamformer.GenNoiseMSC import gen_noise_msc
from beamformer.MicArray import MicArray
from beamformer.beamformer import beamformer
class adaptivebeamfomer(beamformer):

    def __init__(self, MicArray,frameLen=256,hop=None,nfft=None,c=343,r=0.032,fs=16000):

        self.MicArray = MicArray
        self.frameLen = frameLen
        if hop is None:
            self.hop = int(frameLen//2)
        else:
            self.hop = int(hop)
        self.overlap = frameLen - hop
        if nfft is None:
            self.nfft = int(frameLen)
        else:
            self.nfft = int(nfft)
        self.c = c
        self.r = r
        self.fs = fs
        self.half_bin = round(nfft / 2 + 1)
        self.M = 4
        self.angle = np.array([197, 0]) / 180 * np.pi
        self.gamma = MicArray.gamma
        self.window = windows.hann(self.frameLen, sym=False)
        self.win_scale = np.sqrt(1.0/self.window.sum()**2)
        self.freq_bin = np.linspace(0, self.half_bin - 1, self.half_bin)
        self.omega = 2 * np.pi * self.freq_bin * self.fs / self.nfft

    def calTau(self,angle):
        pass

    def AdaptiveMVDR(self,x,angle):
        """
        MVDR beamformer

        """
        frameNum = round((len(x[1, :]) - self.overlap) // self.hop)
        M = len(x[:, 1])

        outputlength = self.frameLen + (frameNum - 1) * self.hop
        norm = np.zeros(outputlength, dtype=x.dtype)

        window = windows.hann(self.frameLen, sym=False)
        win_scale = np.sqrt(1.0/window.sum()**2)

        tao = -1 * self.r * np.cos(angle[1]) * np.cos(angle[0] - self.gamma) / self.c

        yout = np.zeros(outputlength, dtype=x.dtype)

        Rvv = np.ones((self.half_bin, self.M, self.M), dtype=complex)
        alpha = 0.9
        H = np.mat(np.ones([self.half_bin, self.M]), dtype=complex).T

        for k in range(0, self.half_bin):
            a = np.mat(np.exp(-1j * self.omega[k] * tao)).T  # propagation vector
            H[:,k] = self.getMVDRweight(a,Rvv[k, :, :])

        for t in range(0, frameNum):
            xt = x[:, t * self.hop:t * self.hop + self.frameLen] * window
            # Z = np.fft.rfft(xt)*win_scale
            Z = np.fft.rfft(xt)
            if t<200:
                for k in range(0, self.half_bin):
                    Rvv[k, :, :] = alpha * Rvv[k, :, :] + (1 - alpha) * np.dot(Z[:, k,np.newaxis],Z[:, k,np.newaxis].conj().transpose())
                    a = np.mat(np.exp(-1j * self.omega[k] * tao)).T  # propagation vector
                    H[:, k] = self.getMVDRweight(a, Rvv[k, :, :])
            # if t == 200:
            #     for k in range(0, self.half_bin):
            #         a = np.mat(np.exp(-1j * self.omega[k] * tao)).T  # propagation vector
            #         H[:, k] = self.getMVDRweight(a, Rvv[k, :, :])

            x_fft = np.array(np.conj(H)) * Z*win_scale
            yf = np.sum(x_fft, axis=0)
            Cf = np.fft.irfft(yf)*window.sum()
            yout[t * self.hop:t * self.hop + self.frameLen] += Cf*window
            norm[..., t * self.hop:t * self.hop + self.frameLen] += window ** 2

        yout /= np.where(norm > 1e-10, norm, 1)
        return yout

    def AdaptiveMVDR2(self,x,angle):
        """
        MVDR beamformer using built-in stft

        """
        f, t, Zxx = signal.stft(x, self.fs)
        Zout = np.zeros((1,Zxx.shape[1],Zxx.shape[2]),dtype=complex)

        tao = -1 * self.r * np.cos(angle[1]) * np.cos(angle[0] - self.gamma) / self.c

        Fvv = np.ones((self.half_bin, self.M, self.M),dtype=complex)
        H = np.mat(np.ones([self.half_bin, self.M]), dtype=complex).T

        alpha = 0.9

        for k in range(0, self.half_bin):
            for t in range(0,200):
                Fvv[k,:,:] = alpha*Fvv[k,:,:] + (1-alpha)*np.dot(Zxx[:,k,t,np.newaxis],Zxx[:,k,t,np.newaxis].conj().transpose())/(self.win_scale**2)
            a = np.mat(np.exp(-1j * self.omega[k] * tao)).T  # propagation vector
            H[:,k] = self.getMVDRweight(a,Fvv[k, :, :],Diagonal = 1e-6)

        for t in range(0, Zxx.shape[2]):
            x_fft = np.array(np.conj(H)) * Zxx[:,:,t]
            yf = np.sum(x_fft, axis=0)
            Zout[:,:,t] = yf

        _, xrec = signal.istft(Zout, self.fs)
        return xrec

