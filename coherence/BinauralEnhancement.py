import numpy as np
import math
from scipy.signal import windows
from coherence.getweights_coherent import getweghts_coherent
from postfilter.postfilter import PostFilter


class BinauralEnhancement(PostFilter):

    def __init__(self, MicArray, frameLen=256, hop=None, nfft=None, c=343, r=0.032, fs=16000):

        PostFilter.__init__(self, MicArray, frameLen=frameLen, hop=hop, nfft=nfft, c=c, r=r, fs=fs)
        self.M = MicArray.M
        self.fs = self.MicArray.fs
        self.G = np.ones([self.half_bin], dtype=complex)
        self.Fvv_est = np.ones([self.half_bin,self.M,self.M],dtype=complex)*0.98


    def updateMSC(self, alpha=0, UPPPER_THRESHOLD=0.98):
        t = 0
        for i in range(0,self.M-1):
            for j in range(i+1,self.M):
                self.Fvv_est[:,i,j] = self.Pxij[t,:]/np.sqrt(self.Pxii[i,:]*self.Pxii[j,:])
                t = t+1
        # np.where(self.Fvv_est > UPPPER_THRESHOLD, UPPPER_THRESHOLD, self.Fvv_est)

    def getweights(self, Z:np.ndarray,method=3):
        """
        :param
            Z: input spectral,[Nele,fft_bin]

        :return:
            W: post-filter weights
        """
        alpha = 0.6
        self.update_CSD_PSD(Z,alpha=alpha)
        self.updateMSC()
        Fvv_UPPER = 0.98
        for i in range(0,self.M-1):
            for j in range(i+1,self.M):
                for k in range(0, self.half_bin):
                    self.G[k] = getweghts_coherent(self.Fvv_est[k, i, j],self.Fvv[k, i, j],k,method=method)
        return self.G


    def process(self,x,angle,method=1, retH=False,retWNG = False, retDI = False):
        """
        :param
            x:time-aligned multichannel input signal
            DS:beamformed signal
        :return
            yout:postfilterd output
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

        if retH is False:
            beampattern = None

        yout = np.zeros(outputlength, dtype=x.dtype)

        if (all(angle == self.angle) is False) or (method!= self.method) :
            if method!= self.method:
                self.method = method
        for t in range(0, frameNum-1):
            xt = x[:, t * self.hop:t * self.hop + self.frameLen] * window
            Z = np.fft.rfft(xt)#*win_scale

            self.H = self.getweights(Z,method=method)

            yf = np.array(self.H) * Z[0,:]
            Cf = np.fft.irfft(yf)#*window.sum()
            Cf = np.squeeze(Cf)
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
                'beampattern':beampattern
                }
