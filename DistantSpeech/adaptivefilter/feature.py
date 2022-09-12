import numpy as np
import copy
from DistantSpeech.beamformer.utils import load_audio as audioread
from DistantSpeech.beamformer.utils import save_audio as audiowrite


class Emphasis(object):
    def __init__(self, alpha=0.98) -> None:
        self.memD = 0.0
        self.memE = 0.0
        self.alpha = alpha

    def pre_emphsis(self, input):

        output = np.zeros(input.shape)
        for n in range(input.shape[0]):
            output[n] = input[n] - self.alpha * self.memD
            self.memD = input[n]

        return output

    def de_emphsis(self, input):

        output = np.zeros(input.shape)
        for n in range(input.shape[0]):
            output[n] = input[n] + self.alpha * self.memE
            self.memE = output[n]

        return output


class FilterDcNotch16(object):
    def __init__(self, radius=0.9) -> None:
        self.radius = radius
        self.notch_mem = np.zeros((2,))

    def filter_dc_notch16(self, input):
        mem = self.notch_mem
        out = np.zeros(input.shape)
        den2 = self.radius * self.radius + 0.7 * (1 - self.radius) * (1 - self.radius)
        len = input.shape[0]
        for i in range(len):
            vin = input[i]
            vout = mem[0] + vin
            mem[0] = mem[1] + 2 * (-vin + self.radius * vout)
            mem[1] = vin - (den2 * vout)
            out[i] = self.radius * vout

        return out, mem


if __name__ == "__main__":
    x = np.random.rand(16000)
    d = np.random.rand(16000)
    src = audioread(
        "/home/wangwei/work/DistantSpeech/samples/audio_samples/cleanspeech_aishell3.wav"
    )
    emphsis = Emphasis()
    out1 = emphsis.pre_emphsis(src)
    audiowrite("out_preemphsis.wav", out1)
    out2 = emphsis.de_emphsis(out1)
    audiowrite("out_deemphsis.wav", out2)
    dc_notch = FilterDcNotch16()
    out3, mem = dc_notch.filter_dc_notch16(out2)
    audiowrite("out_notch.wav", out3)
