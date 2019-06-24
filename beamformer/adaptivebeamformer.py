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

        self.pad_data = np.zeros([MicArray.M,round(frameLen/2)])
        self.last_output = np.zeros(round(frameLen / 2))

        self.Fvv = gen_noise_msc(self.M, self.nfft, self.fs, self.r)
        self.H = np.ones([self.M, self.half_bin], dtype=complex)/self.M

        self.angle = np.array([0, 0]) / 180 * np.pi

        self.method = 'MVDR'

        self.frameCount = 0

        self.Rvv = np.zeros((self.half_bin, self.M, self.M), dtype=complex)
        self.Ryy = np.zeros((self.half_bin, self.M, self.M), dtype=complex)

    def data_ext(self, x, axis=-1):
        """
        pad front of data with self.pad_data
        """
        left_ext = self.pad_data
        ext = np.concatenate((left_ext,
                              x),
                             axis=axis)
        return ext

    def process(self,x,angle, method='MVDR',retH=False,retWNG = False, retDI = False):
        """
        MVDR beamformer

        """
        # x = self.data_ext(x)
        # self.pad_data = x[:, -1 * round(self.nfft / 2):]
        frameNum = round((len(x[1, :]) - self.overlap) // self.hop)
        M = len(x[:, 1])

        outputlength = self.frameLen + (frameNum - 1) * self.hop
        norm = np.zeros(outputlength, dtype=x.dtype)

        # window = windows.hann(self.frameLen, sym=False)
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

        alpha_y = 0.8
        alpha_v = 0.9998

        for t in range(0, frameNum):
            xt = x[:, t * self.hop:t * self.hop + self.frameLen] * window
            Z = np.fft.rfft(xt)#*win_scale
            if (all(angle == self.angle) is False) or (method != self.method):
                if method != self.method:
                    self.method = method
                    self.frameCount = 0

            for k in range(0, self.half_bin):
                a = np.mat(np.exp(-1j * self.omega[k] * tao)).T  # propagation vector
                # recursive average Ryy
                self.Ryy[k, :, :] = alpha_y * self.Ryy[k, :, :] + (1 - alpha_y) * np.dot(Z[:, k, np.newaxis],
                                                                                 Z[:, k, np.newaxis].conj().transpose())
                if t<200:
                    # recursive average Rvv
                    self.Rvv[k, :, :] = alpha_v * self.Rvv[k, :, :] + (1 - alpha_v) * np.dot(Z[:, k,np.newaxis],Z[:, k,np.newaxis].conj().transpose())
                if method == 'MVDR'and t == 200:
                    self.H[:, k, np.newaxis] = self.getweights(a, method, Rvv=self.Rvv[k, :, :], Diagonal=1e-6)

                    if retWNG:
                        WNG[k] = self.calcWNG(a, self.H[:, k, np.newaxis])
                    if retDI:
                        DI[k] = self.calcDI(a, self.H[:, k, np.newaxis], self.Fvv[k, :, :])
                if method == 'TFGSC':
                    self.H[:, k, np.newaxis] = self.getweights(a, method, Rvv=self.Rvv[k, :, :], Ryy=self.Ryy[k, :, :], Diagonal=1e-6)
                    if retWNG:
                        WNG[k] = self.calcWNG(a, self.H[:, k, np.newaxis])
                    if retDI:
                        DI[k] = self.calcDI(a, self.H[:, k, np.newaxis], self.Fvv[k, :, :])


            x_fft = np.array(np.conj(self.H)) * Z
            yf = np.sum(x_fft, axis=0)
            Cf = np.fft.irfft(yf)#*window.sum()
            yout[t * self.hop:t * self.hop + self.frameLen] += Cf*window
            norm[..., t * self.hop:t * self.hop + self.frameLen] += window ** 2


        norm[..., -1 * round(self.nfft / 2):] += window[:round(self.nfft / 2)] ** 2
        # yout /= np.where(norm > 1e-10, norm, 1.0)

        yout[:round(self.nfft / 2)] += self.last_output
        self.last_output = yout[-1 * round(self.nfft / 2):]
        yout = yout[:-1 * round(self.nfft / 2)]

        # calculate beampattern
        if retH:
            beampattern = self.beampattern(self.omega,self.H)

        return {'data':yout,
                'WNG':WNG,
                'DI':DI,
                'beampattern':beampattern
                }

    def AdaptiveMVDR2(self,x,angle):
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

    def AdaptiveMVDR(self,x,angle,retH=True,retWNG=True,retDI=True):
        """
        MVDR beamformer using built-in stft

        """
        f, t, Zxx = signal.stft(x, self.fs)
        Zout = np.zeros((1,Zxx.shape[1],Zxx.shape[2]),dtype=complex)

        tao = -1 * self.r * np.cos(angle[1]) * np.cos(angle[0] - self.gamma) / self.c
        tao = tao[:, np.newaxis]

        Rvv = np.ones((self.half_bin, self.M, self.M),dtype=complex)
        H = np.ones([ self.M,self.half_bin], dtype=complex)

        if retWNG:
            WNG = np.ones(self.half_bin)
        else:
            WNG = None
        if retDI:
            DI = np.ones(self.half_bin)
            Fvv = gen_noise_msc(self.M, self.nfft, self.fs, self.r)
        else:
            DI = None
        if retH is None:
            beampattern = None

        alpha = 0.9

        for k in range(0, self.half_bin):
            for t in range(0,200):
                Rvv[k,:,:] = alpha*Rvv[k,:,:] + (1-alpha)*np.dot(Zxx[:,k,t,np.newaxis],Zxx[:,k,t,np.newaxis].conj().transpose())/(self.win_scale**2)
            a = np.exp(-1j * self.omega[k] * tao)      # propagation vector
            H[:,k,np.newaxis] = self.getweights(a,weightType='MVDR',Rvv=Rvv[k, :, :],Diagonal = 1e-6)
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

