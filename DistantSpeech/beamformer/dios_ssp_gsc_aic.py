import numpy as np
from numpy.fft import irfft as ifft
from numpy.fft import rfft as fft
from DistantSpeech.beamformer.utils import load_audio, save_audio, delayline
from DistantSpeech.noise_estimation.mcra import NoiseEstimationMCRA


class objFGSCaic(object):
    def __init__(
        self,
        num_mic=4,
        fft_size=128,
        overlap_sigs=4,
        overlap_fft=4,
    ) -> None:
        self.nmic = num_mic  # number of microphones */
        self.fftsize = fft_size  # FFT length */
        self.fftoverlap = overlap_fft  # FFT overlap */
        self.sigsoverlap = overlap_sigs  # overlap factor of input signal segments */
        self.lambda_aic = 0.944750  # forgetting factor power estimation */
        self.mu = 0.033150  # stepsize */
        self.delta_con = 0.0001  # threshold for constant regularization */
        self.delta_dyn = 0.00001  # threshold for dynamic regularization */
        self.s0_dyn = 0.00001  # 'lobe' of dynamic regularization function */
        self.regularize_dyn = 0  # use dynamic regularization or constant regularization */
        self.ntaps = 64  # number of filter taps */
        self.bdlinesize = 1  # length of block delay line */
        self.pbdlinesize = 1  # block delayline length for partitioned block adaptive filter input */
        self.syncdly = 72  # delay for causality of adaptive filters */
        self.nu = complex(0.00001, 0)  # forgetting factor for adaptive filter coefficients */
        self.count_sigsegments = 0  # counter for input signal segments */
        self.xrefdline = np.zeros((self.fftsize // 2 + self.syncdly,))  # delay line reference signal */
        self.half_bin = self.fftsize // 2 + 1
        self.Xfdline = np.zeros(
            (self.nmic, self.bdlinesize, self.half_bin), dtype=complex
        )  # block delay line for filter inputs in frequency domain */
        self.Xfbdline = np.zeros(
            (self.nmic, self.pbdlinesize, self.half_bin), dtype=complex
        )  # block delay line for partitioned block adaptive filter input */
        self.Xdline = np.zeros((self.nmic, self.fftsize))  # block delay line for filter inputs in time domain */
        self.Xffilt = np.zeros((self.nmic, self.fftsize), dtype=complex)  # adaptive filter input signal */
        self.yftmp = np.zeros((self.half_bin,), dtype=complex)  # temporary vector in frequency domain */
        self.ytmp = np.zeros((self.fftsize,))  # temporary vector in time domain */
        self.yhf = np.zeros((self.half_bin,), dtype=complex)  # adaptive filter output in frequency domain */
        self.Hf = np.zeros(
            (self.nmic, self.pbdlinesize, self.half_bin), dtype=complex
        )  # adaptive filter coefficients frequency domain */
        self.e = np.zeros((self.fftsize,))  # error signal in time domain */
        self.z = np.zeros(
            ((int)(self.fftsize / (2 * self.fftoverlap)),)
        )  # time-domain aic output signals of internal processing in processOneDataBlock() */
        self.ef = np.zeros((self.half_bin,), dtype=complex)  # error signal in frequency domain */
        self.pXf = np.zeros((self.half_bin,))  # instantaneous power estimate of adaptive filter input */
        self.sftmp = np.zeros(
            (self.half_bin,)
        )  # temporary variable for recursive power estimation in frequency domain */
        self.sf = np.zeros((self.half_bin,))  # power estimate of adaptive filter input in frequency domain */
        self.muf = np.zeros((self.half_bin,), dtype=complex)  # normalized stepsize in frequency domain */
        self.nuf = np.zeros(
            (self.half_bin,), dtype=complex
        )  # frequency-domain forgetting factor for adaptive filter coefficients */
        self.maxnorm = 0.003  # maximally allowed filter norm, used by norm constraint */

        self.mcra = NoiseEstimationMCRA(nfft=128)


def dios_ssp_gsc_gscaic_processonedatablock(gscaic: objFGSCaic, ctrl_abm: np.array, ctrl_aic: np.array):

    # reset filter output and power estimate to zero */
    gscaic.yhf[:] = 0
    gscaic.pXf[:] = 0

    for k in range(gscaic.nmic):
        # fft of filter inputs */
        gscaic.Xffilt[k, : gscaic.half_bin] = fft(gscaic.Xdline[k])

        # block delay line for partitioned block adaptive filters
        # pbdlinesize = 1, no delay */
        for i in range(gscaic.pbdlinesize - 1):
            gscaic.Xfbdline[k, i + 1, :] = gscaic.Xfbdline[k, i, :]
        gscaic.Xfbdline[k, 0, :] = gscaic.Xffilt[k, : gscaic.half_bin]

        for i in range(1):
            # 1.power estimate of adaptive filter inputs for later recursion */
            gscaic.sftmp = np.abs(gscaic.Xfbdline[k, i]) ** 2
            # 2.summing up power estimate */
            gscaic.pXf = gscaic.pXf + gscaic.sftmp
            # 3.filter with adaptive filters */
            gscaic.yftmp = gscaic.Hf[k, i, :] * gscaic.Xfbdline[k, i, :]
            # 4.summing up filter outputs */
            gscaic.yhf = gscaic.yhf + gscaic.yftmp

    # ifft of adaptive filter output */
    # gscaic.m_pFFT.FFTInv_CToR(gscaic.yhf, gscaic.ytmp)=0
    gscaic.ytmp = ifft(gscaic.yhf)  # / gscaic.fftsize

    # compute error signal in time-domain with circular convolution constraint e = [0 | new] */
    gscaic.e[gscaic.fftsize // 2 :] = gscaic.xrefdline[: gscaic.fftsize // 2] - gscaic.ytmp[gscaic.fftsize // 2 :]

    # copy output samples */
    gscaic.z[:] = gscaic.e[gscaic.fftsize - gscaic.fftsize // (2 * gscaic.fftoverlap) :]

    # fourier transform of aic error signal */
    gscaic.ef[:] = fft(gscaic.e)

    # adaptation
    # recursive power estimate of adaptive filter input all the computations
    # are done with real signals, for later usage, they are transformed to the
    # complex xcomplex format instantaneous energy estimation */
    gscaic.sf = gscaic.lambda_aic * gscaic.sf + (1.0 - gscaic.lambda_aic) * gscaic.pXf

    # normalization term of FLMS . muf */
    # regularization: dynamical or constant */
    if gscaic.regularize_dyn == 1:
        gscaic.sftmp = gscaic.sf + gscaic.delta_dyn * np.exp(-gscaic.sf / gscaic.s0_dyn)

        for k in range(gscaic.fftsize // 2 + 1):
            if gscaic.sftmp[k] < 10e-6:
                gscaic.sftmp[k] = 1.0 / (10e-6)
            else:
                gscaic.sftmp[k] = 1.0 / gscaic.sftmp[k]
    else:
        for k in range(gscaic.fftsize // 2 + 1):
            if gscaic.sf[k] < gscaic.delta_con:
                gscaic.sftmp[k] = 1.0 / gscaic.delta_con
            else:
                gscaic.sftmp[k] = 1.0 / gscaic.sf[k]

    # 1.introduction of stepsize */
    gscaic.sftmp *= gscaic.mu
    # 2.introduction of control signal */
    gscaic.sftmp *= ctrl_aic
    # 3.conversion from float to xcomplex format */
    gscaic.muf = gscaic.sftmp + 1j * 0.0
    # 4.prevents freezing of adaptive filters */
    gscaic.nuf = ctrl_abm + 1j * 0.0
    # gscaic.nuf = 1.0 + 1j * 0.
    gscaic.nuf = gscaic.nuf * gscaic.nu

    norm = 0.0

    for k in range(gscaic.nmic):
        for i in range(1):
            # 1.conjugate of reference signal */
            gscaic.yftmp[:] = gscaic.Xfbdline[k, i, :].conj()

            # 2.enovation term */
            gscaic.yftmp = gscaic.yftmp * gscaic.ef

            # 3.stepsize term */
            gscaic.yftmp = gscaic.yftmp * gscaic.muf

            # 4.update */
            gscaic.Hf[k, i, :] = gscaic.Hf[k, i, :] + gscaic.yftmp

            # 5.norm constraint */
            norm += np.sum(np.abs(gscaic.Hf[k, i, :]) ** 2)

    # norm constraint */
    norm /= gscaic.fftsize * gscaic.fftsize
    if norm > gscaic.maxnorm:
        norm = np.sqrt(gscaic.maxnorm / norm)
    else:
        norm = 1.0

    for k in range(gscaic.nmic):
        for i in range(1):
            # against freezing of the adaptive filter coefficients */
            gscaic.Hf[k, i, :] = gscaic.Hf[k, i, :] - gscaic.Hf[k, i, :] * gscaic.nuf

            # circular correlation constraint (Hf[k][i] = [new | 0]) . Hf[k][i] */
            # gscaic.m_pFFT.FFTInv_CToR(gscaic.Hf[k][i], gscaic.ytmp)=0
            gscaic.ytmp[:] = ifft(gscaic.Hf[k, i, :])
            gscaic.ytmp[gscaic.fftsize // 2 :] = 0

            # norm constraint */
            gscaic.ytmp *= norm

            gscaic.Hf[k, i, :] = fft(gscaic.ytmp)


def dios_ssp_gsc_gscaic_process(gscaic: objFGSCaic, xref, X, y, ctrl_abm, ctrl_aic):
    # buffer input signal segments, input signal segments are shorter than or equal to
    # the processing data blocks, adaptive filter input signal = [old | new] */
    for k in range(gscaic.nmic):
        delayline(X[k], gscaic.Xdline[k], gscaic.fftsize - gscaic.fftsize // (2 * gscaic.sigsoverlap), gscaic.fftsize)

    # delay time-domain fixed beamformer output, syncdly = 72 samples */
    delayline(
        xref,
        gscaic.xrefdline,
        gscaic.fftsize // 2 + gscaic.syncdly - gscaic.fftsize // (2 * gscaic.sigsoverlap),
        gscaic.fftsize // 2 + gscaic.syncdly,
    )

    if gscaic.count_sigsegments == (gscaic.sigsoverlap / gscaic.fftoverlap - 1):
        # process when input signal buffers are filled */
        dios_ssp_gsc_gscaic_processonedatablock(gscaic, ctrl_abm, ctrl_aic)
        gscaic.count_sigsegments = 0
    else:
        # fill input signal block until enough blocks are available */
        gscaic.count_sigsegments = gscaic.count_sigsegments + 1

    return gscaic.z[
        gscaic.count_sigsegments
        * gscaic.fftsize
        // (2 * gscaic.sigsoverlap) : gscaic.count_sigsegments
        * gscaic.fftsize
        // (2 * gscaic.sigsoverlap)
        + (gscaic.fftsize // (2 * gscaic.sigsoverlap))
    ]


# write processed data to the aic output */
# y[:(gscaic.fftsize / (2 * gscaic.sigsoverlap))] = (gscaic.z[gscaic.count_sigsegments * gscaic.fftsize / (2 * gscaic.sigsoverlap)])
# memcpy(y, &(gscaic.z[gscaic.count_sigsegments * gscaic.fftsize / (2 * gscaic.sigsoverlap)]), (gscaic.fftsize / (2 * gscaic.sigsoverlap)) * sizeof(float));


if __name__ == "__main__":

    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    output = np.zeros((11,))
    delayline(data, output, 5, 11)
    print(data)
    print(output)

    gscaic = objFGSCaic()
    ctrl_abm = np.load('/home/wangwei/work/athena-signal/examples/ctrl_abm.npy')
    ctrl_aic = np.load('/home/wangwei/work/athena-signal/examples/ctrl_aic.npy')
    print(ctrl_abm.shape)

    bm = []
    for n in range(1, 5):
        filename = '/home/wangwei/work/athena-signal/examples/bm{}.wav'.format(n)
        data_ch = load_audio(filename) * 32768
        bm.append(data_ch)
    bm = np.array(bm).T
    print(bm.shape)
    fbf = load_audio('/home/wangwei/work/athena-signal/examples/fbf.wav') * 32768

    # fbf = load_audio('/home/wangwei/work/DistantSpeech/example/out_fbf.wav') * 32768
    # bm = load_audio('/home/wangwei/work/DistantSpeech/example/out_bm.wav') * 32768
    # print(bm.shape)

    output = np.zeros(fbf.shape)

    for n in range(ctrl_abm.shape[0]):
        output[n * 16 : (n + 1) * 16] = dios_ssp_gsc_gscaic_process(
            gscaic, fbf[n * 16 : (n + 1) * 16], bm[n * 16 : (n + 1) * 16, :].T, 0, ctrl_abm[n, :], ctrl_aic[n, :]
        )

    Hf = ifft(gscaic.Hf[:, 0, :], axis=-1)
    np.save('hf.npy', Hf)

    save_audio('output.wav', output / 32768)
