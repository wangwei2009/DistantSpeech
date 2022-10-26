"""
time domain GSC beamformer
==============

----------



"""
import numpy as np
from DistantSpeech.noise_estimation.mcra import NoiseEstimationMCRA
from DistantSpeech.noise_estimation.mcspp_base import McSppBase
from DistantSpeech.noise_estimation.omlsa_multi import NsOmlsaMulti
from DistantSpeech.noise_estimation.mc_mcra import McMcra
from DistantSpeech.noise_estimation.mcspp import McSpp
from DistantSpeech.transform.transform import Transform
from DistantSpeech.beamformer.beamformer import beamformer
from DistantSpeech.beamformer.fixedbeamformer import TimeAlignment
from DistantSpeech.beamformer.MicArray import MicArray
from DistantSpeech.adaptivefilter.FastFreqLms import FastFreqLms
from DistantSpeech.adaptivefilter.feature import Emphasis, FilterDcNotch16


class TDGSC(beamformer):
    def __init__(
        self,
        mic_array: MicArray,
        frameLen=256,
        angle=[197, 0],
    ):
        beamformer.__init__(self, mic_array, frame_len=frameLen)
        self.mic_array = mic_array

        self.angle = np.array(angle) / 180 * np.pi if isinstance(angle, list) else angle

        self.time_alignment = TimeAlignment(mic_array, angle=self.angle)
        self.aic_filter = FastFreqLms(filter_len=frameLen, n_channels=self.M - 1, non_causal=True)
        self.dc_notch_mic = []
        for _ in range(self.M):
            self.dc_notch_mic.append(FilterDcNotch16(radius=0.98))

        self.mcra = NoiseEstimationMCRA(nfft=frameLen * 2)
        self.mcra.L = 65
        self.transform = Transform(n_fft=frameLen * 2, hop_length=frameLen, channel=1)

        self.spp = self.mcra

        self.omlsa_multi = NsOmlsaMulti(nfft=frameLen * 2, cal_weights=True, M=self.M)
        self.transform_fbf = Transform(n_fft=frameLen * 2, hop_length=frameLen, channel=1)
        self.transform_bm = Transform(n_fft=frameLen * 2, hop_length=frameLen, channel=self.M - 1)

    def fixed_beamformer(self, x):
        """fixed beamformer

        Parameters
        ----------
        x : np.array
            input multichannel data, [samples, chs]

        Returns
        -------
        np.array
            output data, [samples, 1]
        """

        x_aligned = self.time_alignment.process(x)

        return np.mean(x_aligned, axis=1, keepdims=True), x_aligned

    def blocking_matrix(self, x):
        """fixed blocking matrix

        Parameters
        ----------
        x : np.array
            input multichannel data, [samples, chs]

        Returns
        -------
        bm_output : np.array
            output data, [samples, chs-1]
        """
        samples, channels = x.shape
        bm_output = np.zeros((samples, channels - 1))
        for m in range(channels - 1):
            bm_output[:, m] = x[:, m] - x[:, m + 1]

        return bm_output

    def aic(self, y_fbf, bm_output, p=1.0):
        """adaptive interference cancellation block

        Parameters
        ----------
        y_fbf : np.array
            output from fixed beamformer, upper path, (n_samples, 1)
        bm_output : np.array
            output from blocking matrix output, lower path, (n_samples, n_chs)

        Returns
        -------
        output_n : np.array
            _description_
        """
        # AIC block
        output_n, _ = self.aic_filter.update(bm_output, y_fbf, fir_truncate=30, p=p)

        return output_n

    def process(self, x, postfilter=False):
        """time domain GSC beamformer processing function

        Parameters
        ----------
        x : np.array
            input multichannel data, [samples, chs]

        Returns
        -------
        output : np.array
            output enhanced data, [samples, ]
        """

        samples, channels = x.shape
        output = np.zeros(samples)

        for m in range(channels):
            x[:, m], self.dc_notch_mic[m].mem = self.dc_notch_mic[m].filter_dc_notch16(x[:, m])

        D = self.transform.stft(x[:, 0])
        print(D.shape)

        # overlaps-save approach, no need to use hop_size
        frameNum = int((samples) / self.frameLen)

        p = np.zeros((self.spp.half_bin, frameNum))
        G = np.zeros((self.spp.half_bin, frameNum))

        t = 0

        for n in range(frameNum):
            x_n = x[n * self.frameLen : (n + 1) * self.frameLen, :]

            G[:, t] = self.spp.estimation(D[:, t, :])
            p[:, t] = self.spp.p

            fixed_output, x_aligned = self.fixed_beamformer(x_n)

            bm_output = self.blocking_matrix(x_aligned)

            # AIC block
            output_n = self.aic(fixed_output, bm_output, p=1 - p[:, t : t + 1])

            if postfilter:
                Y = self.transform_fbf.stft(output_n)
                U = self.transform_bm.stft(bm_output)

                self.omlsa_multi.estimation(
                    np.real(Y[:, 0, 0] * np.conj(Y[:, 0, 0])), np.real(U[:, 0, :] * np.conj(U[:, 0, :]))
                )

                # post-filter
                G[:, t] = np.sqrt(self.omlsa_multi.G)
                t += 1
                Y[:, 0, 0] = Y[:, 0, 0] * np.sqrt(self.omlsa_multi.G)
                output_n = self.transform_fbf.istft(Y)

            # output[n * self.frameLen : (n + 1) * self.frameLen] = fixed_output[:, 0]
            # output[n * self.frameLen : (n + 1) * self.frameLen] = np.squeeze(bm_output[:, 0])
            output[n * self.frameLen : (n + 1) * self.frameLen] = np.squeeze(output_n)

        return output, p


if __name__ == "__main__":

    M = 4
    frameLen = 512
    mic_array = MicArray(arrayType="circular", r=0.032, M=M)
    gsc_beamformer = TDGSC(mic_array, frameLen=frameLen)
    x = np.random.randn(16000 * 5, 4)
    output = gsc_beamformer.process(x)
