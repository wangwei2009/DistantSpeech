from scipy.signal import windows
from scipy import signal
import numpy as np
from .MicArray import MicArray
from .beamformer import beamformer
from DistantSpeech.transform.transform import Transform
from DistantSpeech.noise_estimation.mcra import NoiseEstimationMCRA


class adaptivebeamfomer(beamformer):

    def __init__(self, MicArray,frameLen=256,hop=None,nfft=None,c=343,r=0.032,fs=16000):

        beamformer.__init__(self, MicArray, frame_len=frameLen, hop=hop, nfft=nfft, c=c, fs=fs)
        self.M = 4
        self.angle = np.array([197, 0]) / 180 * np.pi
        self.gamma = MicArray.gamma
        self.window = windows.hann(self.frameLen, sym=False)
        self.win_scale = np.sqrt(1.0/self.window.sum()**2)
        self.freq_bin = np.linspace(0, self.half_bin - 1, self.half_bin)
        self.omega = 2 * np.pi * self.freq_bin * self.fs / self.nfft

        self.H = np.ones([self.M, self.half_bin], dtype=complex)/self.M

        self.angle = np.array([0, 0]) / 180 * np.pi

        self.method = 'MVDR'

        self.frameCount = 0
        self.calc = 0
        self.estPos = None                   # set None to use vad

        self.Rvv = np.zeros((self.half_bin, self.M, self.M), dtype=complex)
        self.Rvv_inv = np.zeros((self.half_bin, self.M, self.M), dtype=complex)
        self.Ryy = np.zeros((self.half_bin, self.M, self.M), dtype=complex)

        self.AlgorithmList = ['src', 'DS', 'MVDR', 'TFGSC']
        self.AlgorithmIndex = 0

        self.transformer = Transform(n_fft=self.nfft, hop_length=self.hop, channel=self.M)
        self.mcra = NoiseEstimationMCRA(nfft=self.nfft)

        self.update_noise_psd_flag = 0

    def process(self,x,angle, method=2, retH=False, retWNG=False, retDI=False):
        """
        beamformer process function

        """
        X = self.transformer.stft(x.transpose())
        frameNum = X.shape[1]

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

        alpha_y = 0.8
        alpha_v = 0.9998

        Y = np.ones([self.half_bin, frameNum], dtype=complex)
        for t in range(0, frameNum):
            Z = X[:, t, :].transpose()
            if (all(angle == self.angle) is False) or (method != self.AlgorithmIndex):
                # update look angle and algorithm
                if all(angle == self.angle) is False:
                    self.angle = angle
                if method != self.AlgorithmIndex:
                    self.AlgorithmIndex = method
                # reset flag
                self.frameCount = 0
                self.calc = 0

            self.mcra.estimation(np.abs(Z[0, :] * np.conj(Z[0, :])))

            for k in range(0, self.half_bin):
                a = np.mat(np.exp(-1j * self.omega[k] * tao)).T  # propagation vector
                # recursive average Ryy
                self.Ryy[k, :, :] = alpha_y * self.Ryy[k, :, :] + (1 - alpha_y) * np.dot(Z[:, k, np.newaxis],
                                                                                 Z[:, k, np.newaxis].conj().transpose())
                Diagonal = 1e-6
                if self.estPos is not None:                      # use start frame
                    if self.frameCount<self.estPos:
                        self.update_noise_psd_flag = 1
                        self.frameCount = self.frameCount + 1
                elif self.mcra.p[k] < 0.4:                        # use mcra-based vad
                    self.update_noise_psd_flag = 1
                if self.update_noise_psd_flag:
                    self.Rvv[k, :, :] = alpha_v * self.Rvv[k, :, :] + (1 - alpha_v) * np.dot(Z[:, k,np.newaxis],Z[:, k,np.newaxis].conj().transpose())
                    self.update_noise_psd_flag = 0

                    # print("calculating MVDR weights...\n")
                    Rvv_k = (self.Rvv[k, :, :]) + Diagonal * np.eye(self.M)  # Diagonal loading
                    self.Rvv_inv[k,:,:] = np.linalg.inv(Rvv_k)
                self.H[:, k, np.newaxis] = self.getweights(a,
                                                            self.AlgorithmList[method],
                                                            Rvv=self.Rvv[k, :, :],
                                                            Rvv_inv=self.Rvv_inv[k,:,:],
                                                            Ryy=self.Ryy[k, :, :],
                                                            Diagonal=Diagonal)

                if retWNG:
                    WNG[k] = self.calcWNG(a, self.H[:, k, np.newaxis])
                if retDI:
                    DI[k] = self.calcDI(a, self.H[:, k, np.newaxis], self.Fvv[k, :, :])

            x_fft = np.array(np.conj(self.H)) * Z
            Y[:, t] = np.sum(x_fft, axis=0)

        yout = self.transformer.istft(Y)

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

