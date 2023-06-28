"""
overlap-save demo

Created by Wang wei
"""
import argparse

import numpy as np
from numpy.fft import irfft as ifft
from numpy.fft import rfft as fft
from scipy.signal import convolve as conv

from DistantSpeech.beamformer.utils import load_audio, save_audio


def overlap_save(signal, filter, stride=4, pad_start=True):
    filter = np.squeeze(filter)
    filter_len = len(filter)
    # N = stride, N2 = len(filter), eg.64
    # N1 point circular convolution
    # N1 - N2 + 1 = N
    # N1 >= N + N2 - 1 = 67
    minimal_win_len = stride + filter_len - 1

    # round to 2^n
    n_fft = (int)(2 ** ((int)(np.log2(minimal_win_len)) + 1))

    # pad input signal
    if pad_start:
        signal = np.concatenate((np.zeros((n_fft - stride - filter_len + stride - 1,)), signal))

    # pad filter
    w = np.concatenate((filter, np.zeros((n_fft - filter_len,))))
    W = fft(w)

    output = np.zeros(signal.shape)

    n_block = (len(signal) - n_fft) // stride
    for n in range(n_block):
        x = signal[n * stride : n * stride + n_fft]
        Y = W * fft(x)
        y = ifft(Y)

        # only the last N1 - N2 + 1 are valid, and keep the stride for streaming logic
        output[n * stride : (n + 1) * stride] = y[-(n_fft - filter_len + 1) :][:stride]

    return output


def main(args):
    # src = load_audio('cleanspeech_aishell3.wav')
    signal = load_audio('samples/audio_samples/cleanspeech_aishell3.wav')
    print(signal.shape)
    rir = load_audio('DistantSpeech/adaptivefilter/rir.wav')
    rir = rir[199:]
    rir = rir[:512]
    print(rir.shape)

    output = overlap_save(signal, rir)
    output_ref = conv(signal, rir)

    min_len = min(len(output), len(output_ref))

    print(f'diff:{np.sum(np.abs(output[:min_len]) - np.abs(output_ref[:min_len]))}')

    save_audio("overla_save.wav", output)
    save_audio("overla_save_conv.wav", output_ref)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Feature Extractor")
    parser.add_argument("-l", "--listen", action='store_true', help="set to listen output")  # if set true
    parser.add_argument("-s", "--save", action='store_true', help="set to save output")  # if set true

    args = parser.parse_args()
    main(args)
    # test_aic()
