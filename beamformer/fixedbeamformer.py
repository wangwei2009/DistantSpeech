from scipy.signal import windows
from scipy import signal
import numpy as np
from beamformer.GenNoiseMSC import gen_noise_msc
from beamformer.MicArray import MicArray
from beamformer.beamformer import beamformer
import warnings

class fixedbeamfomer(beamformer):

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

    def superDirectiveMVDR2(self,x,angle):
        """
        MVDR beamformer under isotropic noise field

        """
        frameNum = round((len(x[1, :]) - self.overlap) // self.hop)
        M = len(x[:, 1])

        outputlength = self.frameLen + (frameNum - 1) * self.hop
        norm = np.zeros(outputlength, dtype=x.dtype)

        window = windows.hann(self.frameLen, sym=False)
        win_scale = np.sqrt(1.0/window.sum()**2)

        tao = -1 * self.r * np.cos(angle[1]) * np.cos(angle[0] - self.gamma) / self.c

        yout = np.zeros(outputlength, dtype=x.dtype)

        Fvv = gen_noise_msc(M, self.nfft, self.fs, self.r)
        H = np.mat(np.ones([self.half_bin, self.M]), dtype=complex).T

        for k in range(0, self.half_bin):
            a = np.mat(np.exp(-1j * self.omega[k] * tao)).T  # propagation vector
            H[:,k] = self.getweights(a,'MVDR',Fvv[k, :, :])

        for t in range(0, frameNum):
            xt = x[:, t * self.hop:t * self.hop + self.frameLen] * window
            Z = np.fft.rfft(xt)*win_scale

            x_fft = np.array(np.conj(H)) * Z
            yf = np.sum(x_fft, axis=0)
            Cf = np.fft.irfft(yf)*window.sum()
            yout[t * self.hop:t * self.hop + self.frameLen] += Cf*window
            norm[..., t * self.hop:t * self.hop + self.frameLen] += window ** 2

        yout /= np.where(norm > 1e-10, norm, 1.0)
        return yout

    def superDirectiveMVDR(self,x,angle,retH=True,retWNG = False, retDI = False):
        """
        superdirective MVDR beamformer using built-in STFT function

        """
        f, t, Zxx = signal.stft(x, self.fs)
        Zout = np.zeros((1,Zxx.shape[1],Zxx.shape[2]),dtype=complex)

        tao = -1 * self.r * np.cos(angle[1]) * np.cos(angle[0] - self.gamma) / self.c
        tao = tao[:,np.newaxis]

        Fvv = gen_noise_msc(self.M, self.nfft, self.fs, self.r)
        H = np.ones([self.M,self.half_bin], dtype=complex)

        if retWNG:
            WNG = np.ones(self.half_bin)
        else:
            WNG = None
        if retDI:
            DI = np.ones(self.half_bin)
        else:
            DI = None
        if retH is None:
            beampattern = None


        for k in range(0, self.half_bin):
            a = np.exp(-1j * self.omega[k] * tao)  # propagation vector
            H[:,k,np.newaxis] = self.getweights(a,'MVDR',Fvv[k, :, :])

            if retWNG:
                WNG[k] = self.calcWNG(a, H[:,k,np.newaxis])
            if retDI:
                DI[k] = self.calcDI(a, H[:, k, np.newaxis],Fvv[k,:,:])

        for t in range(0, Zxx.shape[2]):
            x_fft = np.array(np.conj(H)) * Zxx[:,:,t]
            yf = np.sum(x_fft, axis=0)
            Zout[:,:,t] = yf

        _, xrec = signal.istft(Zout, self.fs)

        if retH:
            beampattern = self.beampattern(self.omega,H)

        return {'out':xrec,
                'WNG':WNG,
                'DI':DI,
                'beampattern':beampattern
                }

    def delaysum(self,x,angle,retH=True,retWNG=True,retDI=True):
        """
        delay-and-sum beamformer using built-in stft

        """
        f, t, Zxx = signal.stft(x, self.fs)
        Zout = np.zeros((1,Zxx.shape[1],Zxx.shape[2]),dtype=complex)

        tao = -1.0 * self.r * np.cos(angle[1]) * np.cos(angle[0] - self.gamma) / self.c
        tao = tao[:, np.newaxis]

        H = np.ones([self.M,self.half_bin], dtype=complex)

        if retWNG:
            WNG = np.ones(self.half_bin)
        if retDI:
            DI = np.ones(self.half_bin)
            Fvv = gen_noise_msc(self.M, self.nfft, self.fs, self.r)

        for k in range(0, self.half_bin):
            a = np.exp(-1j * self.omega[k] * tao)   # propagation vector
            H[:,k,np.newaxis] = self.getweights(a,'DS')
            if retWNG:
                WNG[k] = self.calcWNG(a, H[:,k,np.newaxis])
            if retDI:
                DI[k] = self.calcDI(a, H[:, k, np.newaxis],Fvv[k,:,:])


        for t in range(0, Zxx.shape[2]):
            x_fft = np.array(np.conj(H)) * Zxx[:,:,t]
            yf = np.sum(x_fft, axis=0)
            Zout[:,:,t] = yf

        _, xrec = signal.istft(Zout, self.fs)

        if retH:
            beampattern = self.beampattern(self.omega,H)

        return {'out':xrec,
                'WNG':WNG,
                'DI':DI,
                'beampattern':beampattern
                }





