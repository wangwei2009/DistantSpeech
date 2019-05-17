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
        self.pad_data = np.zeros([MicArray.M,round(frameLen/2)])
        self.last_output = np.zeros(round(frameLen / 2))

        self.Fvv = gen_noise_msc(self.M, self.nfft, self.fs, self.r)
        self.H = np.ones([self.M, self.half_bin], dtype=complex)

        self.angle = np.array([0, 0]) / 180 * np.pi

        self.method = 'DS'

    def data_ext(self, x, axis=-1):
        """
        pad front of data with self.pad_data
        """
        left_ext = self.pad_data
        ext = np.concatenate((left_ext,
                              x),
                             axis=axis)
        return ext

    def process(self,x,angle,method='DS',retH=False,retWNG = False, retDI = False):
        """
        fixed beamformer precesing function
        method:
        'DS':   delay-and-sum beamformer
        'MVDR': MVDR beamformer under isotropic noise field
        """
        x = self.data_ext(x)
        self.pad_data = x[:, -1 * round(self.nfft / 2):]
        frameNum = round((len(x[1, :]) - self.overlap) // self.hop)
        M = len(x[:, 1])

        outputlength = self.frameLen + (frameNum - 1) * self.hop
        norm = np.zeros(outputlength, dtype=x.dtype)

        # window = windows.hamming(self.frameLen, sym=False)
        window = np.sqrt(windows.hann(self.frameLen, sym=False))
        norm[:round(self.nfft / 2)] +=window[round(self.nfft / 2):] ** 2
        win_scale = np.sqrt(1.0/window.sum()**2)

        tao = -1 * self.r * np.cos(angle[1]) * np.cos(angle[0] - self.gamma) / self.c

        if retWNG:
            WNG = np.ones(self.half_bin)
        else:
            WNG = None
        if retDI:
            DI = np.ones(self.half_bin)
        else:
            DI = None
        if retH is False:
            beampattern = None

        yout = np.zeros(outputlength, dtype=x.dtype)

        # Fvv = gen_noise_msc(M, self.nfft, self.fs, self.r)
        # H = np.mat(np.ones([self.half_bin, self.M]), dtype=complex).T
        if (all(angle == self.angle) is False) or (method!= self.method) :
            if method!= self.method:
                self.method = method
            for k in range(0, self.half_bin):
                a = np.mat(np.exp(-1j * self.omega[k] * tao)).T  # propagation vector
                self.H[:, k, np.newaxis] = self.getweights(a,method,self.Fvv[k, :, :],Diagonal=1e-1)

                if retWNG:
                    WNG[k] = self.calcWNG(a, self.H[:,k,np.newaxis])
                if retDI:
                    DI[k] = self.calcDI(a, self.H[:, k, np.newaxis],self.Fvv[k,:,:])

        for t in range(0, frameNum):
            xt = x[:, t * self.hop:t * self.hop + self.frameLen] * window
            Z = np.fft.rfft(xt)*win_scale

            x_fft = np.array(np.conj(self.H)) * Z
            yf = np.sum(x_fft, axis=0)
            Cf = np.fft.irfft(yf)*window.sum()
            yout[t * self.hop:t * self.hop + self.frameLen] += Cf*window
            norm[..., t * self.hop:t * self.hop + self.frameLen] += window ** 2

        norm[..., -1*round(self.nfft / 2):] +=window[:round(self.nfft / 2)] ** 2
        yout /= np.where(norm > 1e-10, norm, 1.0)

        yout[:round(self.nfft/2)] += self.last_output
        self.last_output = yout[-1*round(self.nfft/2):]
        yout = yout[:-1*round(self.nfft/2)]

        # update angle
        self.angle = angle

        # calculate beampattern
        if retH:
            beampattern = self.beampattern(self.omega,self.H)

        return {'data':yout,
                'WNG':WNG,
                'DI':DI,
                'beampattern':beampattern
                }


    def superDirectiveMVDR(self,x,angle,retH=False,retWNG = False, retDI = False):
        """
        superdirective MVDR beamformer using built-in STFT function

        """
        x = self.data_ext(x)
        window = windows.hann(self.frameLen, sym=False)
        self.pad_data = x[:,-1*round(self.nfft / 2):]
        f, t, Zxx = signal.stft(x, self.fs, boundary=None, padded=False)
        Zout = np.zeros((1,Zxx.shape[1],Zxx.shape[2]),dtype=complex)

        tao = -1 * self.r * np.cos(angle[1]) * np.cos(angle[0] - self.gamma) / self.c
        tao = tao[:,np.newaxis]

        # Fvv = gen_noise_msc(self.M, self.nfft, self.fs, self.r)
        # H = np.ones([self.M,self.half_bin], dtype=complex)

        if retWNG:
            WNG = np.ones(self.half_bin)
        else:
            WNG = None
        if retDI:
            DI = np.ones(self.half_bin)
        else:
            DI = None
        if retH is False:
            beampattern = None

        # weights updated only if current angle differ from previous.
        if all(angle==self.angle) is False:
            for k in range(0, self.half_bin):
                a = np.exp(-1j * self.omega[k] * tao)  # propagation vector
                self.H[:,k,np.newaxis] = self.getweights(a,'DS',self.Fvv[k, :, :])

                if retWNG:
                    WNG[k] = self.calcWNG(a, self.H[:,k,np.newaxis])
                if retDI:
                    DI[k] = self.calcDI(a, self.H[:, k, np.newaxis],self.Fvv[k,:,:])
        # filter data
        for t in range(0, Zxx.shape[2]):
            x_fft = np.array(np.conj(self.H)) * Zxx[:,:,t]
            yf = np.sum(x_fft, axis=0)
            Zout[:,:,t] = yf

        # reconstruct signal
        _, xrec = signal.istft(Zout, self.fs,boundary=False)

        xrec[0,:round(self.nfft/2)] += self.last_output
        self.last_output = xrec[0,-1*round(self.nfft/2):]

        xrec[0, :round(self.nfft / 2)] = xrec[0,:round(self.nfft/2)]*window[:round(self.nfft / 2)]**1
        xrec[0, -1*round(self.nfft/2):] = xrec[0, :round(self.nfft / 2)] * window[round(self.nfft / 2):]**1


        xrec = xrec[0,:-1*round(self.nfft/2)]

        # update angle
        self.angle = angle

        # calculate beampattern
        if retH:
            beampattern = self.beampattern(self.omega,self.H)

        return {'out':xrec,
                'WNG':WNG,
                'DI':DI,
                'beampattern':beampattern
                }





