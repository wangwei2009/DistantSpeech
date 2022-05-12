import librosa
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from librosa import util
from librosa.filters import get_window
from numba import jit


def stft(
    y,
    n_fft=2048,
    hop_length=None,
    win_length=None,
    window="hann",
    center=True,
    dtype=np.complex64,
    pad_mode="reflect",
):
    """Short-time Fourier transform (STFT). [1]_ (chapter 2)

    The STFT represents a signal in the time-frequency domain by
    computing discrete Fourier transforms (DFT) over short overlapping
    windows.

    This function returns a complex-valued matrix D such that

    - `np.abs(D[f, t])` is the magnitude of frequency bin `f`
      at frame `t`, and

    - `np.angle(D[f, t])` is the phase of frequency bin `f`
      at frame `t`.

    The integers `t` and `f` can be converted to physical units by means
    of the utility functions `frames_to_sample` and `fft_frequencies`.

    .. [1] M. Müller. "Fundamentals of Music Processing." Springer, 2015


    Parameters
    ----------
    y : np.ndarray [shape=(n,)], real-valued
        input signal

    n_fft : int > 0 [scalar]
        length of the windowed signal after padding with zeros.
        The number of rows in the STFT matrix `D` is (1 + n_fft/2).
        The default value, n_fft=2048 samples, corresponds to a physical
        duration of 93 milliseconds at a sample rate of 22050 Hz, i.e. the
        default sample rate in librosa. This value is well adapted for music
        signals. However, in speech processing, the recommended value is 512,
        corresponding to 23 milliseconds at a sample rate of 22050 Hz.
        In any case, we recommend setting `n_fft` to a power of two for
        optimizing the speed of the fast Fourier transform (FFT) algorithm.

    hop_length : int > 0 [scalar]
        number of audio samples between adjacent STFT columns.

        Smaller values increase the number of columns in `D` without
        affecting the frequency resolution of the STFT.

        If unspecified, defaults to `win_length / 4` (see below).

    win_length : int <= n_fft [scalar]
        Each frame of audio is windowed by `window()` of length `win_length`
        and then padded with zeros to match `n_fft`.

        Smaller values improve the temporal resolution of the STFT (i.e. the
        ability to discriminate impulses that are closely spaced in time)
        at the expense of frequency resolution (i.e. the ability to discriminate
        pure tones that are closely spaced in frequency). This effect is known
        as the time-frequency localization tradeoff and needs to be adjusted
        according to the properties of the input signal `y`.

        If unspecified, defaults to ``win_length = n_fft``.

    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        Either:

        - a window specification (string, tuple, or number);
          see `scipy.signal.get_window`

        - a window function, such as `scipy.signal.hanning`

        - a vector or array of length `n_fft`


        Defaults to a raised cosine window ("hann"), which is adequate for
        most applications in audio signal processing.

        .. see also:: `filters.get_window`

    center : boolean
        If `True`, the signal `y` is padded so that frame
        `D[:, t]` is centered at `y[t * hop_length]`.

        If `False`, then `D[:, t]` begins at `y[t * hop_length]`.

        Defaults to `True`,  which simplifies the alignment of `D` onto a
        time grid by means of `librosa.core.frames_to_samples`.
        Note, however, that `center` must be set to `False` when analyzing
        signals with `librosa.stream`.

        .. see also:: `stream`

    dtype : numeric type
        Complex numeric type for `D`.  Default is single-precision
        floating-point complex (`np.complex64`).

    pad_mode : string or function
        If `center=True`, this argument is passed to `np.pad` for padding
        the edges of the signal `y`. By default (`pad_mode="reflect"`),
        `y` is padded on both sides with its own reflection, mirrored around
        its first and last sample respectively.
        If `center=False`,  this argument is ignored.

        .. see also:: `np.pad`


    Returns
    -------
    D : np.ndarray [shape=(1 + n_fft/2, n_frames), dtype=dtype]
        Complex-valued matrix of short-term Fourier transform
        coefficients.


    See Also
    --------
    istft : Inverse STFT

    reassigned_spectrogram : Time-frequency reassigned spectrogram


    Notes
    -----
    This function caches at level 20.


    Examples
    --------

    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> D = np.abs(librosa.stft(y))
    >>> D
    array([[2.58028018e-03, 4.32422794e-02, 6.61255598e-01, ...,
            6.82710262e-04, 2.51654536e-04, 7.23036574e-05],
           [2.49403086e-03, 5.15930466e-02, 6.00107312e-01, ...,
            3.48026224e-04, 2.35853557e-04, 7.54836728e-05],
           [7.82410789e-04, 1.05394892e-01, 4.37517226e-01, ...,
            6.29352580e-04, 3.38571583e-04, 8.38094638e-05],
           ...,
           [9.48568513e-08, 4.74725084e-07, 1.50052492e-05, ...,
            1.85637656e-08, 2.89708542e-08, 5.74304337e-09],
           [1.25165826e-07, 8.58259284e-07, 1.11157215e-05, ...,
            3.49099771e-08, 3.11740926e-08, 5.29926236e-09],
           [1.70630571e-07, 8.92518756e-07, 1.23656537e-05, ...,
            5.33256745e-08, 3.33264900e-08, 5.13272980e-09]], dtype=float32)

    Use left-aligned frames, instead of centered frames

    >>> D_left = np.abs(librosa.stft(y, center=False))


    Use a shorter hop length

    >>> D_short = np.abs(librosa.stft(y, hop_length=64))


    Display a spectrogram

    >>> import matplotlib.pyplot as plt
    >>> librosa.display.specshow(librosa.amplitude_to_db(D,
    ...                                                  ref=np.max),
    ...                          y_axis='log', x_axis='time')
    >>> plt.title('Power spectrogram')
    >>> plt.colorbar(format='%+2.0f dB')
    >>> plt.tight_layout()
    >>> plt.show()
    """

    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length // 4)

    if window is not None:
        fft_window = window
    else:
        fft_window = get_window(window, win_length, fftbins=True)
        fft_window = np.sqrt(fft_window)

    # Pad the window out to n_fft size
    fft_window = util.pad_center(fft_window, size=n_fft)

    # Reshape so that the window can be broadcast
    fft_window = fft_window.reshape((-1, 1))

    # Check audio is valid
    util.valid_audio(y)

    # Pad the time series so that frames are centered
    if center:
        y = np.pad(y, int(n_fft // 2), mode=pad_mode)

    # Window the time series.
    y_frames = util.frame(y, frame_length=n_fft, hop_length=hop_length)

    # Pre-allocate the STFT matrix
    stft_matrix = np.empty((int(1 + n_fft // 2), y_frames.shape[1]), dtype=dtype, order="F")

    # how many columns can we fit within MAX_MEM_BLOCK?
    n_columns = int(util.MAX_MEM_BLOCK / (stft_matrix.shape[0] * stft_matrix.itemsize))

    for bl_s in range(0, stft_matrix.shape[1], n_columns):
        bl_t = min(bl_s + n_columns, stft_matrix.shape[1])

        stft_matrix[:, bl_s:bl_t] = np.fft.rfft(fft_window * y_frames[:, bl_s:bl_t], axis=0)
    return stft_matrix


@jit(nopython=True, cache=True)
def __overlap_add(y, ytmp, hop_length):
    # numba-accelerated overlap add for inverse stft
    # y is the pre-allocated output buffer
    # ytmp is the windowed inverse-stft frames
    # hop_length is the hop-length of the STFT analysis

    n_fft = ytmp.shape[0]
    for frame in range(ytmp.shape[1]):
        sample = frame * hop_length
        y[sample : (sample + n_fft)] += ytmp[:, frame]


def istft(
    stft_matrix,
    hop_length=None,
    win_length=None,
    window="hann",
    center=True,
    dtype=np.float32,
    length=None,
):
    """
    Inverse short-time Fourier transform (ISTFT).

    Converts a complex-valued spectrogram `stft_matrix` to time-series `y`
    by minimizing the mean squared error between `stft_matrix` and STFT of
    `y` as described in [1]_ up to Section 2 (reconstruction from MSTFT).

    In general, window function, hop length and other parameters should be same
    as in stft, which mostly leads to perfect reconstruction of a signal from
    unmodified `stft_matrix`.

    .. [1] D. W. Griffin and J. S. Lim,
        "Signal estimation from modified short-time Fourier transform,"
        IEEE Trans. ASSP, vol.32, no.2, pp.236–243, Apr. 1984.

    Parameters
    ----------
    stft_matrix : np.ndarray [shape=(1 + n_fft/2, t)]
        STFT matrix from `stft`

    hop_length : int > 0 [scalar]
        Number of frames between STFT columns.
        If unspecified, defaults to `win_length / 4`.

    win_length : int <= n_fft = 2 * (stft_matrix.shape[0] - 1)
        When reconstructing the time series, each frame is windowed
        and each sample is normalized by the sum of squared window
        according to the `window` function (see below).

        If unspecified, defaults to `n_fft`.

    window : string, tuple, number, function, np.ndarray [shape=(n_fft,)]
        - a window specification (string, tuple, or number);
          see `scipy.signal.get_window`
        - a window function, such as `scipy.signal.hanning`
        - a user-specified window vector of length `n_fft`

        .. see also:: `filters.get_window`

    center : boolean
        - If `True`, `D` is assumed to have centered frames.
        - If `False`, `D` is assumed to have left-aligned frames.

    dtype : numeric type
        Real numeric type for `y`.  Default is 32-bit float.

    length : int > 0, optional
        If provided, the output `y` is zero-padded or clipped to exactly
        `length` samples.

    Returns
    -------
    y : np.ndarray [shape=(n,)]
        time domain signal reconstructed from `stft_matrix`

    See Also
    --------
    stft : Short-time Fourier Transform

    Notes
    -----
    This function caches at level 30.

    Examples
    --------
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> D = librosa.stft(y)
    >>> y_hat = librosa.istft(D)
    >>> y_hat
    array([ -4.812e-06,  -4.267e-06, ...,   6.271e-06,   2.827e-07], dtype=float32)

    Exactly preserving length of the input signal requires explicit padding.
    Otherwise, a partial frame at the end of `y` will not be represented.

    >>> n = len(y)
    >>> n_fft = 2048
    >>> y_pad = librosa.util.fix_length(y, n + n_fft // 2)
    >>> D = librosa.stft(y_pad, n_fft=n_fft)
    >>> y_out = librosa.istft(D, length=n)
    >>> np.max(np.abs(y - y_out))
    1.4901161e-07
    """

    n_fft = 2 * (stft_matrix.shape[0] - 1)

    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length // 4)

    if window is not None:
        ifft_window = window
    else:
        ifft_window = get_window(window, win_length, fftbins=True)
        ifft_window = np.sqrt(ifft_window)

    # Pad out to match n_fft, and add a broadcasting axis
    ifft_window = util.pad_center(ifft_window, size=n_fft)[:, np.newaxis]

    # For efficiency, trim STFT frames according to signal length if available
    if length:
        if center:
            padded_length = length + int(n_fft)
        else:
            padded_length = length
        n_frames = min(stft_matrix.shape[1], int(np.ceil(padded_length / hop_length)))
    else:
        n_frames = stft_matrix.shape[1]

    expected_signal_len = n_fft + hop_length * (n_frames - 1)
    y = np.zeros(expected_signal_len, dtype=dtype)

    n_columns = int(util.MAX_MEM_BLOCK // (stft_matrix.shape[0] * stft_matrix.itemsize))

    frame = 0
    for bl_s in range(0, n_frames, n_columns):
        bl_t = min(bl_s + n_columns, n_frames)

        # invert the block and apply the window function
        ytmp = ifft_window * np.fft.irfft(stft_matrix[:, bl_s:bl_t], axis=0)

        # Overlap-add the istft block starting at the i'th frame
        __overlap_add(y[frame * hop_length :], ytmp, hop_length)

        frame += bl_t - bl_s

    # Normalize by sum of squared window
    # ifft_window_sum = window_sumsquare(window,
    #                                    n_frames,
    #                                    win_length=win_length,
    #                                    n_fft=n_fft,
    #                                    hop_length=hop_length,
    #                                    dtype=dtype)
    #
    # approx_nonzero_indices = ifft_window_sum > util.tiny(ifft_window_sum)
    # y[approx_nonzero_indices] /= ifft_window_sum[approx_nonzero_indices]

    if length is None:
        # If we don't need to control length, just do the usual center trimming
        # to eliminate padded data
        if center:
            y = y[int(n_fft // 2) : -int(n_fft // 2)]
    else:
        if center:
            # If we're centering, crop off the first n_fft//2 samples
            # and then trim/pad to the target length.
            # We don't trim the end here, so that if the signal is zero-padded
            # to a longer duration, the decay is smooth by windowing
            start = int(n_fft // 2)
        else:
            # If we're not centering, start at 0 and trim/pad as necessary
            start = 0

        y = util.fix_length(y[start:], length)

    return y


class Transform(object):
    def __init__(self, channel=1, n_fft=256, hop_length=128, window=None):
        self.channel = channel
        self.n_fft = n_fft
        self.frame_length = self.n_fft
        self.hop_length = hop_length
        self.previous_input = np.zeros((self.hop_length, self.channel))
        self.previous_output = np.zeros((self.hop_length, self.channel))
        self.first_frame = 1
        if window is not None:
            self.window = window
        else:
            self.window = get_window("hann", n_fft, fftbins=True)
            self.window = np.sqrt(self.window)
        self.half_bin = int(self.n_fft / 2 + 1)

    def stft(self, x):
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
        n_frames = 1 + int((x.shape[0] - self.frame_length) / self.hop_length)
        Y = np.zeros((self.half_bin, n_frames, self.channel), dtype=np.complex128)
        for ch in range(self.channel):
            Y[:, :, ch] = stft(
                x[:, ch],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                center=False,
                window=self.window,
            )  # [1 + n_fft/2, n_frames]
            self.previous_input[:, ch] = x[-self.hop_length :, ch]

        return np.squeeze(Y)

    def istft(self, Y):
        """
        streaming single channel inverse short time fourier transform
        :param Y: [half_bin, frames]
        :return: single channel time data
        """
        if len(Y.shape) == 1:  # single frame
            Y = Y[:, np.newaxis]
        x = istft(
            Y,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            center=False,
            window=self.window,
        )
        x[: self.hop_length] += self.previous_output[:, 0]
        self.previous_output[:, 0] = x[-self.hop_length :]
        return x[: -self.hop_length]

    def magphase(self, D, power=1):
        mag = np.abs(D)
        mag **= power
        phase = np.exp(1.0j * np.angle(D))

        return mag, phase


if __name__ == "__main__":
    filename = "DistantSpeech/transform/speech1.wav"
    data, sr = librosa.load(filename, sr=None)

    data_recon = np.zeros(len(data))
    t = 0
    frame_length = 1120
    stream = librosa.stream(
        filename,
        block_length=1,
        frame_length=frame_length,
        hop_length=frame_length,
        mono=True,
    )
    transform = Transform(n_fft=320, hop_length=160)
    for y_block in stream:
        if len(y_block) >= 1024:
            D = transform.stft(y_block)  # [half_bin, n_frame]
            d = transform.istft(D)
            data_recon[t * frame_length : (t + 1) * frame_length] = d
        t = t + 1

    # compare difference between original signal and reconstruction signal
    plt.figure()
    plt.plot(data[: len(data_recon[160:])] - data_recon[160:])
    plt.title("difference between source and reconstructed signal")
    plt.xlabel("samples")
    plt.ylabel("amplitude")
    plt.show()

    # if you want to listen the reconstructed signal, uncomment section below
    # sd.play(data_recon, sr)
    # sd.wait()
