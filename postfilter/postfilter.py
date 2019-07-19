import numpy as np
from scipy.signal import windows
from scipy import signal
from beamformer.beamformer import beamformer
from beamformer.fixedbeamformer import fixedbeamfomer


class PostFilter(fixedbeamfomer):

    def __init__(self, MicArray, frameLen=256, hop=None, nfft=None, c=343, r=0.032, fs=16000):

        fixedbeamfomer.__init__(self, MicArray, frameLen=frameLen, hop=hop, nfft=nfft, c=c, r=r, fs=fs)
        self.M = 8
        self.half_bin = 129
        self.NumSpec = int((self.M * self.M - self.M) / 2)
        self.Pxii = np.zeros([self.M,self.half_bin])
        self.Pxij = np.zeros([self.NumSpec, self.half_bin])
        self.H = np.ones([1, self.half_bin], dtype=complex)

    def update_CSD_PSD(self, Z:np.ndarray, alpha=0.8):
        """
        :param
            Z: input spectral,[Nele,fft_bin]

        :return:
            W: post-filter weights
        """
        t = 0
        M = Z.shape[0]

        # update auto-spectral
        for i in range(0,M):
            # auto-spectral
            Pxii_curr = np.real(Z[i,:]*Z[i,:].conj())
            self.Pxii[i,:] = alpha * self.Pxii[i,:]+(1 - alpha) * Pxii_curr

        # update cross-spectral
        for i in range(0,M-1):
            for j in range(i+1,M):
                # cross - spectral
                Pxij_curr = Z[i,:]* Z[j,:].conj()
                # average
                self.Pxij[t,:] = alpha * self.Pxij[t,:]+(1 - alpha) * Pxij_curr
                t = t + 1


    def getweights(self, Z:np.ndarray):
        """
        :param
            Z: input spectral,[Nele,fft_bin]

        :return:
            W: post-filter weights
        """
        alpha = 0.8
        self.update_CSD_PSD(Z,alpha=alpha)
        SPECTRAL_FLOOR = 0.4
        t = 0
        M = Z.shape[0]
        N = M

        Pss_e = np.zeros([self.NumSpec, self.half_bin])

        Pssnn = np.sum(self.Pxii,axis=0) / M

        for i in range(0,M-1):
            for j in range(i+1,M):
                # eq.22 estimate source signal's PSD
                Fvv_i_j = self.Fvv[:,i,j]
                Fvv_i_j = np.where(Fvv_i_j < 0.7, Fvv_i_j, 0.7)
                Pss_e[t,:] = (np.real(self.Pxij[t,:]) - 0.5 * np.real(Fvv_i_j) * (self.Pxii[i,:] + self.Pxii[j,:])) \
                / \
                (np.ones([1, self.half_bin]) - Fvv_i_j)

                t = t + 1
        t = 1
        # eq.23
        # take the average of multichanel signal to improve robustness
        if Pss_e.shape[0]>1:
            Pss = np.sum(Pss_e,axis=0) * 2 / (N * N - N)
        else:
            Pss = Pss_e

        W = np.real(Pss)/ Pssnn

        return W


    def process(self,x,DS,angle,method='DS',retH=False,retWNG = False, retDI = False):
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
            Z = np.fft.rfft(xt)*win_scale

            bf_t = DS[t * self.hop:t * self.hop + self.frameLen] * window
            Bf_t = np.fft.rfft(bf_t)*win_scale

            self.H = self.getweights(Z)

            yf = np.array(self.H) * Bf_t
            Cf = np.fft.irfft(yf)*window.sum()
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

