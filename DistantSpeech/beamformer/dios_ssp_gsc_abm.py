# # Copyright (C) 2017 Beijing Didi Infinity Technology and Development Co.,Ltd.
# All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# 	http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Description: The ABM consists of adaptive filters between the FBF output and
# the sensor channels: The signal of interest is adaptively subtracted from the
# sidelobe cancelling path in order to prevent target signal cancellation by the
# AIC. The time delay ensures causality of the adaptive filters.
# ==============================================================================*/

# include "dios_ssp_gsc_abm.h"

import numpy as np
from numpy.fft import irfft as ifft
from numpy.fft import rfft as fft
from DistantSpeech.beamformer.utils import load_audio, save_audio, delayline
from DistantSpeech.noise_estimation.mcra import NoiseEstimationMCRA


class objFGSCabm(object):
    def __init__(
        self,
        num_mic=4,
        fft_size=128,
        overlap_sigs=4,
        overlap_fft=2,
        dlysync=32,
        forgetfactor=0.99,
        stepsize=0.5,
        threshdiv0=0.0001,
        rate=16000,
        tconst_freezing=100.0,
    ) -> None:
        self.nmic = num_mic  # number of microphones */
        self.fftsize = fft_size  # length of FFT */
        self.fftoverlap = overlap_fft  # overlap factor of FFT */
        self.sigsoverlap = overlap_sigs  # overlap factor of input signal segments */
        self.syncdly = dlysync  # synchronization delay */
        self.lambda_bm = forgetfactor * np.power(
            1.0 - 1.0 / (3.0 * self.fftsize), self.fftsize / (2 * self.fftoverlap)
        )  # forgetting factor power estimation */
        self.delta = threshdiv0  # threshold factor for preventing division by zero */

        self.half_bin = self.fftsize // 2 + 1

        self.mu = 2 * stepsize * (1 - self.lambda_bm)
        self.delta = threshdiv0

        self.nu = complex(
            1.0 - np.exp(-self.fftsize / (2 * self.fftoverlap * tconst_freezing * rate)), 0
        )  # forgetting factor for adaptive filter coefficients */
        self.count_sigsegments = 0  # counter for input signal segments */
        self.Xdline = np.zeros((self.nmic, self.fftsize))  # delayline of beamsteering output for causality */
        self.xrefdline = np.zeros((self.fftsize // 2 + self.syncdly,))  # delayline for adaptive filter input */
        self.xfref = np.zeros((self.fftsize // 2 + 1,), dtype=complex)  # frequency-domain adaptive filter input */
        self.hf = np.zeros((self.nmic, self.fftsize // 2 + 1), dtype=complex)  # adaptive filter transfer functions */
        self.ytmp = np.zeros((self.fftsize,))  # temporary signal buffer in time domain */
        self.yf = np.zeros((self.fftsize // 2 + 1,), dtype=complex)  # adaptive filter output in frequency domain */
        self.e = np.zeros((self.fftsize,))  # time-domain error signal */
        self.E = np.zeros(
            (self.nmic, self.fftsize // (2 * self.fftoverlap))
        )  # abm output signals of internal processing in processOneDataBlock() */
        self.ef = np.zeros((self.fftsize // 2 + 1,), dtype=complex)  # frequency-domain error signal */
        self.muf = np.zeros((self.fftsize // 2 + 1,), dtype=complex)  # normalized step size in frequency domain */
        self.nuf = np.zeros(
            (self.fftsize // 2 + 1,), dtype=complex
        )  # frequency-domain forgetting factor for adaptive filter coefficients */
        self.yftmp = np.zeros((self.fftsize // 2 + 1,), dtype=complex)  # temporary signal buffer in frequency domain */
        self.pxfref = np.zeros((self.fftsize // 2 + 1,))  # instantaneous power estimate of adaptive filter input */
        self.sf = np.zeros((self.nmic, self.fftsize // 2 + 1))
        self.pftmp = np.zeros((self.fftsize // 2 + 1,))  # temporary variable for calculation of normalized stepsize */
        self.m_upper_bound = np.zeros((self.fftsize // 2,))
        self.m_lower_bound = np.zeros((self.fftsize // 2,))

        deltax = 0.001
        for i in range(self.fftsize // 2):
            self.m_upper_bound[i] = deltax
            self.m_lower_bound[i] = -deltax
        self.m_upper_bound[self.fftsize // 4] = 1.3
        if self.nmic > 2:
            self.m_upper_bound[self.fftsize // 4 + 1] = 0.6
            self.m_upper_bound[self.fftsize // 4 - 1] = 0.6
            self.m_upper_bound[self.fftsize // 4 + 2] = 0.15
            self.m_upper_bound[self.fftsize // 4 - 2] = 0.15
        elif self.nmic == 2:
            self.m_upper_bound[self.fftsize // 4] = 1.1
            self.m_upper_bound[self.fftsize // 4 + 1] = 0.7
            self.m_upper_bound[self.fftsize // 4 - 1] = 0.7
            self.m_upper_bound[self.fftsize // 4 + 2] = 0.3
            self.m_upper_bound[self.fftsize // 4 - 2] = 0.3
            self.m_upper_bound[self.fftsize // 4 + 3] = 0.1
            self.m_upper_bound[self.fftsize // 4 - 3] = 0.1
        # self.abm_FFT =dios_ssp_share_rfft_init(self.fftsize)
        self.fft_out = np.zeros((self.fftsize,), dtype=complex)
        self.fft_in = np.zeros((self.fftsize,))

        # initialize ABM with coefficients for desired signal from steering direction and acoustic free-field condition */
        dios_ssp_gsc_gscabm_initabmfreefield(self)

        self.mcra = NoiseEstimationMCRA(nfft=128)

        print("gscabm.fftsize:{}".format(self.fftsize))
        print("gscabm.fftoverlap:{}".format(self.fftoverlap))
        print("gscabm.sigsoverlap:{}".format(self.sigsoverlap))
        print("gscabm.forgetfactor:{}".format(forgetfactor))
        print("gscabm.stepsize:{}".format(stepsize))
        print("threshdiv0:{}".format(threshdiv0))
        print("lambda_bm:{}".format(self.lambda_bm))
        print("rate:{}".format(rate))
        print("tconst_freezing:{}".format(tconst_freezing))
        print("gscabm.mu:{}".format(self.mu))
        print("gscabm.syncdly:{}".format(self.syncdly))
        print("gscabm.nu.r:{}".format(self.nu))


# def dios_ssp_gsc_gscabm_reset(gscabm : objFGSCabm):
# 	# count variable for filling up abm input signal buffers */
# 	gscabm.count_sigsegments = 0

# 	for m in range(gscabm.nmic):
# 		gscabm.Xdline[m, :] = 0
# 	gscabm.xrefdline[:] = 0

# 	# init adaptive filter input in the frequency domain */
# 	gscabm.xfref[:] = 0.0
# 	# adaptive filter output in frequency domain */
# 	gscabm.yf[:] = 0.0
# 	# frequency-domain error signal */
# 	gscabm.ef[:] = 0.0
# 	# normalized stepsize in frequency domain */
# 	gscabm.muf[:] = 0.0
# 	# frequency-domain forgetting factor for adaptive filter coefficients */
# 	gscabm.nuf[:] = 0.0
# 	# temporary signal buffer in frequency domain */
# 	gscabm.yftmp[:] = 0.0
# 	# temporary signal buffer in time domain */
# 	gscabm.ytmp[:gscabm.fftsize] = 0
# 	# time-domain error signal */
# 	gscabm.e[:gscabm.fftsize] = 0
# 	# instantaneous power estimate of adaptive filter input */
# 	gscabm.pxfref[:gscabm.fftsize // 2 + 1] = 0
# 	# time-domain error signal for abm output */
# 	for m in range(gscabm.nmic):
# 		gscabm.E[m, :gscabm.fftsize / (2 * gscabm.fftoverlap)] = 0

# 	# power estimate after recursive filtering */
# 	for m in range(gscabm.nmic):
#         gscabm.sf[m, :gscabm.fftsize // 2 + 1] = 0
# 	memset(gscabm.pftmp, 0, sizeof(float) * (gscabm.fftsize // 2 + 1))
# 	# adaptive filters in frequency domain */
# 	for m in range(gscabm.nmic):
# 	{
#         for n in range(gscabm.fftsize // 2 + 1):
#         {
#             gscabm.hf[m][n].i = 0.0f
#             gscabm.hf[m][n].r = 0.0f
#         }
# 	}

# 	for (int i = 0 i < gscabm.fftsize i++)
# 	{
#         gscabm.fft_out[i] = 0.0
#         gscabm.fft_in[i] = 0.0
# 	}

# 	float deltax = 0.001f
# 	for (int i = 0 i < gscabm.fftsize // 2 i++)
# 	{
#         gscabm.m_upper_bound[i] = deltax
#         gscabm.m_lower_bound[i] = -deltax
# 	}
# 	gscabm.m_upper_bound[gscabm.fftsize / 4] = 1.3f
# 	if (gscabm.nmic > 2)
# 	{
#         gscabm.m_upper_bound[gscabm.fftsize / 4 + 1] = 0.6f
#         gscabm.m_upper_bound[gscabm.fftsize / 4 - 1] = 0.6f
#         gscabm.m_upper_bound[gscabm.fftsize / 4 + 2] = 0.15f
#        gscabm. m_upper_bound[gscabm.fftsize / 4 - 2] = 0.15f
# 	}
# 	else if (gscabm.nmic == 2)
# 	{
#         gscabm.m_upper_bound[gscabm.fftsize / 4] = 1.1f
#         gscabm.m_upper_bound[gscabm.fftsize / 4 + 1] = 0.7f
#         gscabm.m_upper_bound[gscabm.fftsize / 4 - 1] = 0.7f
#         gscabm.m_upper_bound[gscabm.fftsize / 4 + 2] = 0.3f
#         gscabm.m_upper_bound[gscabm.fftsize / 4 - 2] = 0.3f
#         gscabm.m_upper_bound[gscabm.fftsize / 4 + 3] = 0.1f
#         gscabm.m_upper_bound[gscabm.fftsize / 4 - 3] = 0.1f
# 	}

# dios_ssp_gsc_gscabm_initabmfreefield(self)

# 	return 0
# }


def dios_ssp_gsc_gscabm_initabmfreefield(gscabm: objFGSCabm):
    gscabm.ytmp[: gscabm.fftsize] = 0
    gscabm.ytmp[gscabm.syncdly] = 1
    gscabm.hf[0, :] = fft(gscabm.ytmp)

    for k in range(1, gscabm.nmic):
        gscabm.hf[k, : (gscabm.fftsize // 2 + 1)] = gscabm.hf[0, : (gscabm.fftsize // 2 + 1)]


def dios_ssp_gsc_gscabm_processonedatablock(gscabm: objFGSCabm, ctrl_abm, ctrl_aic):
    for ch in range(gscabm.nmic):
        gscabm.xfref[: gscabm.half_bin] = fft(gscabm.Xdline[ch])

        gscabm.pxfref = np.abs(gscabm.xfref) ** 2
        gscabm.sf[ch] = gscabm.lambda_bm * gscabm.sf[ch] + (1.0 - gscabm.lambda_bm) * gscabm.pxfref
        for i in range(gscabm.fftsize // 2 + 1):

            # 1.normalization term of FLMS . muf */
            if gscabm.sf[ch][i] < gscabm.delta:
                gscabm.pftmp[i] = 1.0 / gscabm.delta
            else:
                gscabm.pftmp[i] = 1.0 / gscabm.sf[ch][i]

        gscabm.pftmp *= gscabm.mu

        # 2.introduction of control signal */
        gscabm.pftmp *= ctrl_abm

        # 3.conversion from float to xcomplex format */
        gscabm.muf = gscabm.pftmp + 1j * 0.0
        # 4.prevents freezing of adaptive filters */
        gscabm.nuf = ctrl_aic + 1j * 0.0
        gscabm.nuf = gscabm.nuf * gscabm.nu
        # 5.compute adaptive filter output */
        gscabm.yf = gscabm.xfref * gscabm.hf[ch]

        # ifft of adaptive filter output: y is then constrained to be y = [0 | new] */
        # gscabm.m_pFFT.FFTInv_CToR(gscabm.yf, gscabm.ytmp)
        gscabm.ytmp = ifft(gscabm.yf)

        # /* compute error signal in time-domain with circular convolution constraint e = [0 | new] */
        gscabm.e[gscabm.fftsize // 2 :] = gscabm.xrefdline[: gscabm.fftsize // 2] - gscabm.ytmp[gscabm.fftsize // 2 :]

        # last block of error signal . abm output signal in time domain */
        gscabm.E[ch, :] = gscabm.e[gscabm.fftsize - gscabm.fftsize // (2 * gscabm.fftoverlap) :]

        gscabm.ef = fft(gscabm.e)

        # update of adaptive filter */
        # 1.conjugate of reference signal */
        gscabm.yftmp[:] = gscabm.xfref.conj()

        # 2.enovation term */
        gscabm.yftmp = gscabm.yftmp * gscabm.ef

        # 3.stepsize term */
        gscabm.yftmp = gscabm.yftmp * gscabm.muf

        # 4.update */
        gscabm.hf[ch] = gscabm.hf[ch] + gscabm.yftmp
        # against freezing of the adaptive filter coefficients */
        gscabm.hf[ch] = gscabm.hf[ch] - gscabm.hf[ch] * gscabm.nuf

        # circular correlation constraint (hf = [new | 0]) . hf */
        # gscabm.m_pFFT.FFTInv_CToR(gscabm.hf[ch], gscabm.ytmp)
        gscabm.ytmp = ifft(gscabm.hf[ch])

        gscabm.ytmp[gscabm.fftsize // 2 :] = 0

        limit_ind = gscabm.fftsize // 4 - 3
        for i in range(limit_ind, 0, -1):
            gscabm.ytmp[i] = np.minimum(np.maximum(gscabm.ytmp[i], gscabm.m_lower_bound[i]), gscabm.m_upper_bound[i])
            gscabm.ytmp[gscabm.fftsize // 2 - i] = np.minimum(
                np.maximum(gscabm.ytmp[gscabm.fftsize // 2 - i], gscabm.m_lower_bound[gscabm.fftsize // 2 - i]),
                gscabm.m_upper_bound[gscabm.fftsize // 2 - i],
            )
        gscabm.ytmp[0] = np.minimum(np.maximum(gscabm.ytmp[0], gscabm.m_lower_bound[0]), gscabm.m_upper_bound[0])

        gscabm.hf[ch] = fft(gscabm.ytmp)


def dios_ssp_gsc_gscabm_process(gscabm: objFGSCabm, X, xref, Y, ctrl_abm, ctrl_aic, index=0):

    # buffer input signal segments, input signal segments are shorter than or equal to the processing data blocks adaptive filter input signal = [old | new] */
    delayline(
        xref,
        gscabm.xrefdline,
        gscabm.fftsize // 2 + gscabm.syncdly - gscabm.fftsize // (2 * gscabm.sigsoverlap),
        gscabm.fftsize // 2 + gscabm.syncdly,
    )
    for i in range(gscabm.nmic):
        delayline(
            X[i, :], gscabm.Xdline[i], gscabm.fftsize - gscabm.fftsize // (2 * gscabm.sigsoverlap), gscabm.fftsize
        )
    # for n in range(gscabm.fftsize):
    # 	print("{:.4f},".format(gscabm.Xdline[0][n]), end="")

    if gscabm.count_sigsegments == (gscabm.sigsoverlap // gscabm.fftoverlap - 1):  # 4 / 2 - 1 */
        # process when input signal buffers are filled */
        dios_ssp_gsc_gscabm_processonedatablock(gscabm, ctrl_abm, ctrl_aic)
        gscabm.count_sigsegments = 0
    else:
        # fill input signal block until enough blocks are available */
        gscabm.count_sigsegments += 1

    # write processed data to the abm output */
    output_len = gscabm.fftsize // (2 * gscabm.sigsoverlap)
    output_start = gscabm.count_sigsegments * gscabm.fftsize // (2 * gscabm.sigsoverlap)
    return gscabm.E[:, output_start : output_start + output_len].T
    # for i in range(gscabm.nmic):
    # Y[i, :(gscabm.fftsize // (2 * gscabm.sigsoverlap))] = gscabm.E[i, gscabm.count_sigsegments * gscabm.fftsize // (2 * gscabm.sigsoverlap):]


if __name__ == "__main__":

    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    output = np.zeros((11,))
    delayline(data, output, 5, 11)
    print(data)
    print(output)

    gscabm = objFGSCabm()
    ctrl_abm = np.load('/home/wangwei/work/athena-signal/examples/ctrl_abm.npy')
    ctrl_aic = np.load('/home/wangwei/work/athena-signal/examples/ctrl_aic.npy')
    print(ctrl_abm.shape)

    m_outSteering = []
    for n in range(4):
        filename = '/home/wangwei/work/athena-signal/examples/out_steering{}.wav'.format(n)
        data_ch = load_audio(filename) * 32768
        m_outSteering.append(data_ch)
    m_outSteering = np.array(m_outSteering).T
    print(m_outSteering.shape)

    m_outFBF = load_audio('/home/wangwei/work/athena-signal/examples/out_fbf.wav') * 32768

    # fbf = load_audio('/home/wangwei/work/DistantSpeech/example/out_fbf.wav') * 32768
    bm = load_audio('/home/wangwei/work/DistantSpeech/example/out_bm.wav') * 32768
    print(bm.shape)

    output = np.zeros(m_outSteering.shape)

    for n in range(ctrl_abm.shape[0]):
        output[n * 16 : (n + 1) * 16, :] = dios_ssp_gsc_gscabm_process(
            gscabm,
            m_outSteering[n * 16 : (n + 1) * 16, :].T,
            m_outFBF[n * 16 : (n + 1) * 16],
            0,
            ctrl_abm[n, :],
            ctrl_aic[n, :],
        )

    # Hf = ifft(gscabm.Hf[:, 0, : ], axis=-1)
    # np.save('hf.npy', Hf)

    save_audio('output.wav', output / 32768)
