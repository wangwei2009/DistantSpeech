import numpy as np
import argparse
from DistantSpeech.beamformer.utils import load_audio
from scipy.signal import convolve as conv
from matplotlib import pyplot as plt
from tqdm import tqdm

def main(args):
    src = load_audio('cleanspeech.wav')
    print(src.shape)
    rir = load_audio('rir.wav')
    rir = rir[200:, np.newaxis]
    rir = rir[:512, :]

    src = np.concatenate((src, src, src, src), axis=0)
    print(src.shape)
    data = conv(src, rir[:, 0])

    filter_len = 1024
    input_buffer = np.zeros((filter_len, 1))
    w = np.zeros((filter_len, 1))

    err = np.zeros(np.size(data))
    output = np.zeros(np.size(data))

    est_err = np.zeros(np.size(data))

    #  time - domain NLMS
    mu = 0.1
    alpha = 1e-4

    for n in tqdm(range((len(src)))):
        # update input buffer
        input_buffer[1:] = input_buffer[:-1]
        input_buffer[0] = src[n]

        # error signal
        err[n] = data[n] - w.T @ input_buffer

        # store err as final output
        output[n] = err[n]

        # update filter coef
        # w = w + 2 * mu * input_buffer * err[n]  # LMS
        w = w + 2 * mu * input_buffer * err[n] / (input_buffer.T @ input_buffer+alpha) # NLMS

        est_err[n] = (rir - w[:len(rir), :]).T @ (rir - w[:len(rir), :])
        # est_err[n] = np.sum(np.abs(rir - w[:len(rir)]))

    plt.plot(10 * np.log(est_err / np.sum(np.abs(rir)) + 1e-12))
    # plt.plot(est_err)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Feature Extractor")
    parser.add_argument("-l", "--listen", action='store_true', help="set to listen output")  # if set true
    parser.add_argument("-s", "--save", action='store_true', help="set to save output")  # if set true

    args = parser.parse_args()
    main(args)