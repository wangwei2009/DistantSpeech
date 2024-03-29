import numpy as np
import argparse
from DistantSpeech.beamformer.utils import load_audio
from scipy.signal import convolve as conv
from matplotlib import pyplot as plt
from tqdm import tqdm


def awgn(x, snr, seed=7):
    '''
    加入高斯白噪声 Additive White Gaussian Noise
    :param x: 原始信号
    :param snr: 信噪比
    :return: 加入噪声后的信号
    '''
    np.random.seed(seed)  # 设置随机种子
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(x**2) / len(x)
    npower = xpower / snr
    noise = np.random.randn(len(x)) * np.sqrt(npower)
    return x + noise


class BaseFilter(object):
    def __init__(self, filter_len=1024, mu=0.1, normalization=True):
        self.filter_len = filter_len

        self.w = np.zeros((filter_len, 1))
        self.mu = mu
        self.norm = normalization

        self.input_buffer = np.zeros((filter_len, 1))

    def update_input(self, xt):
        """

        :param xt:
        :return:
        """
        # update input buffer
        self.input_buffer[1:] = self.input_buffer[:-1]
        self.input_buffer[0] = xt

    def update(self, x_n, d_n, eps=1e-4, p=1.0):
        """nlms update function

        Parameters
        ----------
        x_n : np.array
            input signal, one sample
        d_n : np.array
            expected signal, one sample
        alpha : float, optional
            minv value to prevent divided by zero, by default 1e-4

        Returns
        -------
        err : np.array
            err signal, one sample
        w : np.array
            estimated filter coeffs, [filter_len, ]
        """
        self.update_input(x_n)

        # error signal
        err = d_n - self.w.T @ self.input_buffer

        if self.norm:
            grad = self.input_buffer * err / (self.input_buffer.T @ self.input_buffer + eps)
        else:
            grad = self.input_buffer * err  # LMS

        if grad.ndim == 1:
            grad = grad[:, np.newaxis]
        self.w = self.w + 2 * p * self.mu * grad

        return err, self.w

    def update_coef(self, grad):

        if grad.ndim == 1:
            grad = grad[:, np.newaxis]
        self.w = self.w + 2 * self.mu * grad

        return self.w

    def filter(self, data, data_d):
        """nlms filter function, estimated filter weights stored in self.w

        Parameters
        ----------
        data : np.array
            input signal, [samples, ]
        data_d : np.array
            expected signal, [samples, ]

        Returns
        -------
        err : np.array
            err signal, [samples, ]
        w : np.array
            estimated filter weights, [filter_len, ]
        """
        data = np.squeeze(data)
        data_d = np.squeeze(data_d)
        err = np.zeros(len(data))
        for n in range(len(data)):
            err[n], w = self.update(data[n], data_d[n])

        return err


def main(args):
    src = load_audio('/home/wangwei/work/DistantSpeech/samples/audio_samples/cleanspeech_aishell3.wav')
    print(src.shape)
    rir = load_audio('/home/wangwei/work/DistantSpeech/samples/audio_samples/rir.wav')
    rir = rir[200:]
    rir = rir[:512, np.newaxis]

    # src = awgn(src, 30)
    print(src.shape)

    SNR = 20
    data_clean = conv(src, rir[:, 0])
    data = data_clean[: len(src)]
    # data = awgn(data, SNR)

    filter_len = 512
    w = np.zeros((filter_len, 1))

    lms = BaseFilter(filter_len=filter_len, mu=0.1, normalization=False)
    nlms = BaseFilter(filter_len=filter_len, mu=0.1)

    est_err_lms = np.zeros(np.size(data))
    est_err_nlms = np.zeros(np.size(data))

    for n in tqdm(range((len(src)))):
        _, w_lms = lms.update(src[n], data[n])
        _, w_nlms = nlms.update(src[n], data[n])
        est_err_lms[n] = np.sum(np.abs(rir - w_lms[: len(rir)]) ** 2)
        est_err_nlms[n] = np.sum(np.abs(rir - w_nlms[: len(rir)]) ** 2)

    plt.plot(10 * np.log(est_err_lms / np.sum(np.abs(rir[:, 0]) ** 2) + 1e-12))
    plt.plot(10 * np.log(est_err_nlms / np.sum(np.abs(rir[:, 0]) ** 2)) + 1e-12)
    plt.legend(['lms', 'nlms'], loc='upper right')
    plt.ylabel("$\||\hat{w}-w\||_2$")
    plt.title('weight estimation error vs step')
    plt.show()


def test():
    src = load_audio('/home/wangwei/work/DistantSpeech/samples/audio_samples/cleanspeech_aishell3.wav')
    print(src.shape)
    rir = load_audio('/home/wangwei/work/DistantSpeech/samples/audio_samples/rir.wav')
    rir = rir[200:]
    rir = rir[:512, np.newaxis]

    # src = awgn(src, 30)
    print(src.shape)

    SNR = 20
    data_clean = conv(src, rir[:, 0])
    data = data_clean[: len(src)]
    # data = awgn(data, SNR)

    filter_len = 512
    w = np.zeros((filter_len, 1))

    lms = BaseFilter(filter_len=filter_len, mu=0.1, normalization=False)
    nlms = BaseFilter(filter_len=filter_len, mu=0.1)

    err_lms = lms.filter(src, data)
    err_nlms = nlms.filter(src, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Feature Extractor")
    parser.add_argument("-l", "--listen", action='store_true', help="set to listen output")  # if set true
    parser.add_argument("-s", "--save", action='store_true', help="set to save output")  # if set true

    args = parser.parse_args()
    test()
