import numpy as np
from scipy.signal import windows

# from vad.vad import vad
from DistantSpeech.noise_estimation.mcra import NoiseEstimationMCRA
from .beamformer import beamformer


class GSC(beamformer):

    def __init__(self, MicArray,frameLen=256,hop=None,nfft=None,c=343,r=0.032,fs=16000):

        beamformer.__init__(self, MicArray)
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

        self.H = np.ones([self.M, self.half_bin], dtype=complex)/self.M

        self.angle = np.array([0, 0]) / 180 * np.pi

        self.method = 'MVDR'

        self.frameCount = 0
        self.calc = 1
        self.estPos = 200

        self.Rvv = np.zeros((self.half_bin, self.M, self.M), dtype=complex)
        self.Rvv_inv = np.zeros((self.half_bin, self.M, self.M), dtype=complex)
        self.Ryy = np.zeros((self.half_bin, self.M, self.M), dtype=complex)

        self.AlgorithmList = ['src', 'DS', 'MVDR', 'TFGSC']
        self.AlgorithmIndex = 0

        # blocking matrix
        self.BM = np.zeros((self.M, self.M - 1,self.half_bin),dtype=complex)
        # noise reference
        self.U = np.zeros((self.M - 1, self.half_bin),dtype=complex)
        # fixed beamformer weights for upper path
        self.W = np.zeros((self.M, self.half_bin),dtype=complex)
        # MNC weights for lower path
        self.G = np.zeros((self.M - 1, self.half_bin),dtype=complex)
        self.Pest = np.ones(self.half_bin)

        self.Yfbf = np.zeros((self.half_bin),dtype=complex)

        self.mcra = NoiseEstimationMCRA(nfft=self.nfft)



    def data_ext(self, x, axis=-1):
        """
        pad front of data with self.pad_data
        """
        left_ext = self.pad_data
        ext = np.concatenate((left_ext,
                              x),
                             axis=axis)
        return ext

    def process(self,x,angle, method=2,retH=False,retWNG = False, retDI = False):
        """
        beamformer process function

        """
        x = self.data_ext(x)
        self.pad_data = x[:, -1 * round(self.nfft / 2):]
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

        mu = 0.01
        rho = 0.998

        # if vad():
        #     is_speech = 1
        #     print("speech detected!!!\n")
        # else:
        #     is_speech = 0

        if (all(angle == self.angle) is False) or (method != self.AlgorithmIndex):
            # update look angle and algorithm
            if all(angle == self.angle) is False:
                self.angle = angle
                tao = -1 * self.r * np.cos(self.angle[1]) * np.cos(self.angle[0] - self.gamma) / self.c
            if method != self.AlgorithmIndex:
                self.AlgorithmIndex = method
            # reset flag
            self.frameCount = 0
            self.calc = 1

            # update blocking matrix
            for k in range(0, self.half_bin):
                a_k = np.mat(np.exp(-1j * self.omega[k] * tao)).T
                self.W[:,k, np.newaxis] = a_k / (a_k.conj().T @ a_k)
                for i in range(0, M-1):
                    self.BM[0, i, k] = a_k[0]
                    self.BM[i + 1, i, k] = -1 * a_k[i + 1]
        Y = np.ones([self.half_bin], dtype=complex)
        for t in range(0, frameNum):
            xt = x[:, t * self.hop:t * self.hop + self.frameLen] * window
            Z = np.fft.rfft(xt)#*win_scale
            if (all(angle == self.angle) is False) or (method != self.AlgorithmIndex):
                # update look angle and algorithm
                if all(angle == self.angle) is False:
                    self.angle = angle
                    tao = -1 * self.r * np.cos(self.angle[1]) * np.cos(self.angle[0] - self.gamma) / self.c
                if method != self.AlgorithmIndex:
                    self.AlgorithmIndex = method
                # reset flag
                self.frameCount = 0
                self.calc = 1

            self.mcra.estimation(np.abs(Z[0,:]*np.conj(Z[0,:])))

            # if t < 300:
            #     is_speech = 0
            # else:
            #     is_speech = 1

            for k in range(0, self.half_bin):
                a = np.mat(np.exp(-1j * self.omega[k] * tao)).T  # propagation vector
                # recursive average Ryy
                self.Ryy[k, :, :] = alpha_y * self.Ryy[k, :, :] + (1 - alpha_y) * np.dot(Z[:, k, np.newaxis],
                                                                                 Z[:, k, np.newaxis].conj().transpose())
                if self.mcra.p[k] > 0.5:
                    is_speech = 1
                else:
                    is_speech = 0

                if is_speech==0:
                    # recursive average Rvv
                    self.Rvv[k, :, :] = alpha_v * self.Rvv[k, :, :] + (1 - alpha_v) * np.dot(Z[:, k,np.newaxis],Z[:, k,np.newaxis].conj().transpose())
                # if self.frameCount == 200 and self.calc == 0:

                if t>312 and k == self.half_bin-1:
                    # reset weights update flag
                    self.calc = 0
                # print("calculating MVDR weights...\n")
                Diagonal = 1e-6

                if self.AlgorithmIndex == 0:
                    Y[k] = Z[0, k] # output channel_1
                else:
                    # generate the reference noise signals
                    self.U[:, k] = np.squeeze(self.BM[:, :, k]).conj().T @ Z[:, k]
                    # fixed beamformer output
                    self.Yfbf[k] = self.W[:, k].conj().T @ Z[:, k]

                    # residual output
                    Y[k] = self.Yfbf[k] - self.G[:,k].conj().T@self.U[:,k]

                    # use speech presence probability to control AIC update
                    self.Pest[k] = 1  # rho * self.Pest[k] + (1 - rho) * np.sum(np.power(np.abs(Z[:,k]),2))
                    # update MNC weights
                    self.G[:, k] = self.G[:, k] + mu * (1 - self.mcra.p[k]) * self.U[:, k] * Y[k].conj() / self.Pest[k]

                    if retWNG:
                        WNG[k] = self.calcWNG(a, self.H[:, k, np.newaxis])
                    if retDI:
                        DI[k] = self.calcDI(a, self.H[:, k, np.newaxis], self.Fvv[k, :, :])

            Cf = np.fft.irfft(Y)#*window.sum()
            yout[t * self.hop:t * self.hop + self.frameLen] += Cf*window
            norm[..., t * self.hop:t * self.hop + self.frameLen] += window ** 2

            if self.frameCount< self.estPos:
                self.frameCount = self.frameCount + 1
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