import librosa
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sounddevice as sd
from librosa import util
from librosa.filters import get_window
from numba import jit
from DistantSpeech.transform.prototype_filter import PrototypeFilter


class Subband(object):
    def __init__(self, channel=1, n_fft=256, hop_length=128, window=None):
        self.channel = channel
        self.n_fft = n_fft
        self.frame_length = self.n_fft
        self.hop_length = hop_length

        self.first_frame = 1
        if window is not None:
            self.window = window
        else:
            self.window = PrototypeFilter().get_prototype_filter()

        self.half_bin = int(self.n_fft / 2 + 1)
        self.D = 2

        self.win_len = self.window.shape[0]

        self.overlap = self.win_len - self.hop_length
        self.previous_input = np.zeros((self.overlap, self.channel))
        self.previous_output = np.zeros((self.win_len, self.channel))

    def analysis(self, x):
        """
        streaming multi-channel Short Time Fourier Transform
        :param x: [samples, channels] or [samples,]
        :return: [half_bin, frames, channels] or [half_bin, frames]
        """
        if len(x.shape) == 1:  # single channel
            x = x[:, np.newaxis]
        x = np.vstack((self.previous_input, x))
        x = np.asfortranarray(x)
        # Compute the number of frames that will fit. The end may get truncated.
        n_frames = int((x.shape[0] - self.overlap) / self.hop_length)
        Y = np.zeros((self.half_bin, n_frames, self.channel), dtype=complex)
        for ch in range(self.channel):
            x_ch = x[:, ch]
            for n in range(n_frames):
                x_ch_n = x_ch[self.hop_length * n : self.hop_length * n + self.win_len]
                windowed = np.flip(x_ch_n) * self.window
                x_sumed = np.sum(np.reshape(windowed, (self.D, -1)), axis=0)
                Y[:, n, ch] = np.fft.rfft(x_sumed)

        self.previous_input[:, ch] = x[-self.overlap :, ch]

        return np.squeeze(Y)

    def synthesis(self, Y):
        """
        streaming single channel inverse short time fourier transform
        :param Y: [half_bin, frames]
        :return: single channel time data
        """
        if len(Y.shape) == 1:  # single frame
            Y = Y[:, np.newaxis]
        n_frames = Y.shape[1]
        expected_signal_len = self.n_fft + self.hop_length * (n_frames - 1)
        output = np.zeros(expected_signal_len)
        tdl = self.previous_output[:, 0].copy()
        for n in range(n_frames):
            y = np.fft.irfft(Y[:, n : n + 1], axis=0)
            # print(y)
            y_repmat = np.tile(y, (self.D, 1))
            y_windowed = y_repmat[:, 0] * self.window

            tdl[self.hop_length :] = tdl[: -self.hop_length]
            tdl[: self.hop_length] = 0.0
            tdl = tdl + y_windowed

            t = n * self.hop_length
            output[t : t + self.hop_length] = (self.n_fft * self.hop_length) * np.flip(tdl[-self.hop_length :])

        self.previous_output[:, 0] = tdl.copy()

        return output[: self.hop_length * n_frames]

    def magphase(self, D, power=1):
        mag = np.abs(D)
        mag **= power
        phase = np.exp(1.0j * np.angle(D))

        return mag, phase


if __name__ == "__main__":
    from DistantSpeech.beamformer.utils import mesh

    filename = "DistantSpeech/transform/speech1.wav"
    data, sr = librosa.load(filename, sr=None)

    data_recon = np.zeros(len(data))
    t = 0
    frame_length = 128
    stream = librosa.stream(
        filename,
        block_length=1,
        frame_length=frame_length,
        hop_length=frame_length,
        mono=True,
    )
    transform = Subband(n_fft=512, hop_length=128)
    n_frames = int((data.shape[0]) / 128)
    print('n_frames:{}'.format(n_frames))
    mag_data = np.zeros((257, n_frames + 1))
    for y_block in stream:
        if len(y_block) >= 128:
            D = transform.analysis(y_block)  # [half_bin, n_frame]
            mag_data[:, t] = np.abs(D)
            # d = transform.istft(D)
            # data_recon[t * frame_length : (t + 1) * frame_length] = d
        t = t + 1
    mesh()

    # compare difference between original signal and reconstruction signal
    # plt.figure()
    # plt.plot(data[: len(data_recon[160:])] - data_recon[160:])
    # plt.title("difference between source and reconstructed signal")
    # plt.xlabel("samples")
    # plt.ylabel("amplitude")
    # plt.show()

    # if you want to listen the reconstructed signal, uncomment section below
    # sd.play(data_recon, sr)
    # sd.wait()
