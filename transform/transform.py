import numpy as np
from scipy.signal import windows


class transform(object):
    def __init__(self, frameLength=256, hop=128, nfft=256, fs=16000, M=4):
        self.frameLen = int(frameLength)
        if hop is None:
            self.hop = int(self.frameLen//2)
        else:
            self.hop = int(hop)
        self.overlap = self.frameLen - self.hop
        self.nfft = int(nfft)
        self.half_bin = int(nfft/2+1)
        self.fs = fs
        self.M = M
        self.pad_data = np.zeros([self.M, round(self.frameLen / 2)])
        self.last_output = np.zeros(round(self.frameLen / 2))

    def data_ext(self, x, axis=-1):
        """
        pad front of data with self.pad_data
        """
        left_ext = self.pad_data
        ext = np.concatenate((left_ext,
                              x),
                             axis=axis)
        return ext

    def stft(self,x):
        """
        Multi-channel STFT
        :param
            x: input time data,[channel,sample]

        :return:
            X: time-frequency data,[frame,channel,frequency]
        """

        if x.ndim == 1:
            x = x[:,np.newaxis].transpose()
        x = self.data_ext(x)
        self.pad_data = x[:, -1 * round(self.nfft / 2):]
        frameNum = round((len(x[0, :]) - self.overlap) // self.hop)
        M = len(x[:, 1])

        outputlength = int(self.frameLen + (frameNum - 1) * self.hop)
        X = np.ones((frameNum,self.M,self.half_bin), dtype=complex)
        norm = np.zeros(outputlength, dtype=x.dtype)

        # window = windows.hann(self.frameLen, sym=False)
        window = np.sqrt(windows.hann(self.frameLen, sym=False))
        norm[:round(self.nfft / 2)] +=window[round(self.nfft / 2):] ** 2
        win_scale = np.sqrt(1.0/window.sum()**2)

        for t in range(0, frameNum):
            xt = x[:, t * self.hop:t * self.hop + self.frameLen] * window
            X[t,:,:] = np.fft.rfft(xt)#*win_scale

        return X

    def istft(self,X):
        """
        inverse stft
        :param
            X: time-frequency data,[frame,frequency]

        :return:
            x: single channel time data
        """
        X = np.squeeze(X)

        frameNum = X.shape[0]
        M = X.shape[1]

        outputlength = self.frameLen + (frameNum - 1) * self.hop
        x = np.zeros(outputlength, dtype=float)
        norm = np.zeros(outputlength, dtype=float)

        # window = windows.hann(self.frameLen, sym=False)
        window = np.sqrt(windows.hann(self.frameLen, sym=False))
        norm[:round(self.nfft / 2)] +=window[round(self.nfft / 2):] ** 2
        win_scale = np.sqrt(1.0/window.sum()**2)

        for t in range(0, frameNum):
            xt = np.fft.irfft(X[t,:])
            # overlap-add
            x[t * self.hop:t * self.hop + self.frameLen] += xt*window#*win_scale

        return x

    def spectrum(self,X, mode=1):
        """
        plot spectrum
        :param
            X: time-frequency data,[frame,frequency]

        :return:
            x: single channel time data
        """
        # for now X should be 2-dim
        X = np.squeeze(X)


        if mode !=1 & mode != 2:
            mode = 1
        if mode  == 1:
            X = np.abs(X)
        elif mode == 2:
            X = X*X.conj()

        X = 10*np.log10(X)

        from matplotlib import pyplot as plt
        t = np.linspace(0,X.shape[0],X.shape[0])
        f = np.linspace(0,self.half_bin*self.fs/self.nfft,self.half_bin)
        plt.pcolormesh(t,f,X.transpose())
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sample]')
        plt.show()
