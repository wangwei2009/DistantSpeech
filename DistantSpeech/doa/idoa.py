"""
Instance DOA
refer to
    "Directional Interference Suppression Using a Spatial Relative Transfer Function Feature," ICASSP 2019 - 2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2019, pp. 661-665, doi: 10.1109/ICASSP.2019.8682442.
    "Acoustic Localization Using Spatial Probability in Noisy and Reverberant Environments," 2019 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA), 2019, pp. 353-357, doi: 10.1109/WASPAA.2019.8937163. 
Author:
    Wang Wei
"""
# from matplotlib.pyplot import angle_spectrum
from tqdm import tqdm
import numpy as np
from DistantSpeech.beamformer.MicArray import MicArray
from DistantSpeech.transform.transform import Transform
from DistantSpeech.beamformer.utils import pmesh
from DistantSpeech.noise_estimation.mcra import NoiseEstimationMCRA
from DistantSpeech.adaptivefilter.feature import Emphasis, FilterDcNotch16


class Idoa(object):
    def __init__(self, mic_array: MicArray) -> None:
        """estimate Spatial Speech Probability

        Parameters
        ----------
        mic_array : MicArray
            mic array object
        """
        self.mic_array = mic_array
        self.transform = Transform(channel=mic_array.M, n_fft=mic_array.n_fft, hop_length=int(mic_array.n_fft / 2))
        self.spp = NoiseEstimationMCRA(nfft=mic_array.n_fft)
        self.spp.L = 65

        self.half_bin = self.transform.half_bin
        half_bin = self.half_bin

        idoa_dim = self.mic_array.M - 1
        self.idoa_dim = idoa_dim

        if self.mic_array.arrayType == 'linear':
            theta = np.linspace(0, 180, num=180) / 180 * np.pi
        if self.mic_array.arrayType == 'circular':
            theta = np.linspace(0, 360, num=360) / 180 * np.pi

        n_theta = theta.shape[0]
        self.n_theta = n_theta

        # Presence of a sound source
        self.Delta = np.zeros((half_bin, n_theta))
        self.mu_Delta = np.zeros((half_bin, n_theta))
        self.B_hat = np.zeros((half_bin, idoa_dim), dtype=complex)  # estimated RTF

        self.Y_smooth = np.zeros((half_bin,))  #
        self.Y_xcorr_smooth = np.zeros((half_bin, idoa_dim), dtype=complex)  #

        self.Psi = np.zeros((half_bin, idoa_dim, n_theta), dtype=complex)
        gamma = np.zeros((half_bin, n_theta))
        lambdda = np.zeros((half_bin, n_theta))
        lambdda = np.zeros((half_bin, n_theta))

        self.Lambda = np.zeros((half_bin, n_theta))

        alpha = 0.02
        self.p = np.zeros((half_bin, n_theta))

        # predefined free-field RTF
        for theta_n in range(theta.shape[0]):
            steer_vector = mic_array.steering_vector(look_direction=theta_n)
            self.Psi[:, :, theta_n] = steer_vector[:, 1:] / steer_vector[:, 0:1]

        self.p_h0 = np.zeros((half_bin, n_theta))  # eq.11, target speech absence
        self.p_hd = np.zeros((half_bin, n_theta))  # eq.9, target speech present
        self.p_h0[:, ...] = 0.5
        self.p_hd[:, ...] = 0.5
        self.beta = 7.6
        self.beta_n = np.zeros((n_theta,))  # eq.9, target speech present

        self.p = np.zeros((half_bin, n_theta))

        self.emphsis = []
        for m in range(self.mic_array.M):
            self.emphsis.append(Emphasis())

    def estimate(self, X):
        """spatial speech presence probability estimation core function

        Parameters
        ----------
        X : complex np.array
            input multichannel frequency data, [half_bin, n_frames, channels]

        Returns
        -------
        p : np.array
            speech presence probability, [half_bin, n_frames, n_theta]
        """
        half_bin, n_frames, M = X.shape
        assert half_bin == self.half_bin

        n_theta = self.n_theta
        idoa_dim = self.idoa_dim

        # Presence of a sound source
        Delta = np.zeros((half_bin, n_theta))

        B_hat = np.zeros((half_bin, n_frames, idoa_dim), dtype=complex)  # estimated RTF

        alpha = 0.02
        p = np.zeros((half_bin, n_frames, n_theta))

        beta = 7.6
        beta_n = np.zeros((n_frames, n_theta))  # eq.9, target speech present

        Y_xcorr_curr = X[:, :, 1:] * X[:, :, 0:1].conj()
        Y_curr = X[:, :, 0] * X[:, :, 0].conj()

        for n in range(n_frames):
            # estimate RTF
            self.Y_smooth = (1 - alpha) * self.Y_smooth + alpha * np.abs(Y_curr[:, n])
            self.Y_xcorr_smooth = (1 - alpha) * self.Y_xcorr_smooth + alpha * Y_xcorr_curr[:, n, :]
            B_hat = self.Y_xcorr_smooth / self.Y_smooth[:, np.newaxis]
            # print(B_hat.shape)

            for theta_n in range(n_theta):
                den = np.linalg.norm(self.Psi[:, :, theta_n], axis=-1) * np.linalg.norm(B_hat[:, :], axis=-1)
                Delta[:, theta_n] = np.real(np.einsum('ij,ij->i', self.Psi[:, :, theta_n].conj(), B_hat[:, :])) / (
                    den + 1e-6
                )  # eq.8

            avg = (1 - self.p) * 0.8
            self.mu_Delta = avg * self.mu_Delta + (1 - avg) * Delta
            beta_n[n, :] = 1 / (1 - np.mean(self.mu_Delta[72:128, :], axis=0))
            # beta_n[n, theta_n] = beta

            # eq.11, p(Delta | H0), target speech absence
            self.p_h0 = 1 + np.cos(np.pi * Delta)

            # eq.9, p(Delta | Hd), target speech present
            self.p_hd = beta_n * np.exp(beta * (Delta - 1))

            # eq.13 log-likelihood ratio
            self.Lambda = self.p_hd / (self.p_h0 + 1e-6)

            # eq. 12, Target speech presence probability
            self.p = self.Lambda / (1 + self.Lambda)
            p[:, n, :] = self.p[:]

        return p

    def process(self, x, pre_emphsis=False):
        """spatial speech presence probability estimation

        Parameters
        ----------
        x : np.array
            multichannel input signal, [samples, channels]

        """

        if pre_emphsis:
            for m in range(self.mic_array.M):
                x[:, m] = self.emphsis[m].pre_emphsis(x[:, m])

        X = self.transform.stft(x)

        half_bin, n_frames, M = X.shape
        assert half_bin == self.half_bin

        p = self.estimate(X)

        out = np.maximum(np.mean(p[64:128, :, target_direction], axis=0), 0.01) * X[:, 0, 0]

        output = self.transform.istft(out)

        return output


if __name__ == "__main__":
    from DistantSpeech.beamformer.utils import load_audio as audioread
    from DistantSpeech.beamformer.utils import pmesh, mesh, load_wav, save_audio, load_pcm, pt

    import matplotlib.pyplot as plt
    from tqdm import tqdm

    M = 4
    c = 343
    r = 0.032
    mic_array = MicArray(arrayType='circular', M=4, n_fft=512)
    print(mic_array.mic_loc)
    idoa = Idoa(mic_array)

    array_data = []
    for n in range(2, 6):
        filename = '/home/wangwei/work/DistantSpeech/samples/audio_samples/xmos/rec1/音轨-{}.wav'.format(n)
        # filename = '/home/wangwei/work/DistantSpeech/samples/audio_samples/xmos/meeting/1/wav/音轨-{}.wav'.format(n)
        # filename = '/home/wangwei/work/DistantSpeech/samples/audio_samples/xmos/aioffice/1/ch4/音轨-{}.wav'.format(n)
        # filename = '/home/wangwei/work/DistantSpeech/samples/audio_samples/xmos/meeting/2/ch4//音轨-{}.wav'.format(n)
        # filename = '/home/wangwei/work/DistantSpeech/samples/audio_samples/xmos/anechoic/2/ch4/音轨-{}.wav'.format(n)
        # filename = '/home/wangwei/work/DistantSpeech/samples/audio_samples/xmos/meeting/2/ch4/音轨-{}.wav'.format(n)
        # filename = '/home/wangwei/work/DistantSpeech/samples/audio_samples/xmos/anechoic/1/ch4/音轨-{}.wav'.format(n)
        # filename = '/home/wangwei/work/DistantSpeech/samples/audio_samples/xmos/office/1/ch4/音轨-{}.wav'.format(n)
        filename = '/home/wangwei/work/DistantSpeech/samples/audio_samples/xmos/aioffice/5/ch4/音轨-{}.wav'.format(n)
        data_ch = audioread(filename)
        array_data.append(data_ch)
    x = np.array(array_data).T
    print(x.shape)

    hop_len = idoa.transform.hop_length
    half_bin = idoa.transform.half_bin
    n_theta = idoa.n_theta
    n_frames = (int)(x.shape[0] / hop_len)
    print('n_frames:{}'.format(n_frames))

    p = np.zeros((half_bin, n_frames, n_theta))

    Out = np.zeros((x.shape[0],))

    target_direction = 135

    for n in tqdm(range(n_frames)):
        x_n = x[n * hop_len : n * hop_len + hop_len, :]
        output = idoa.process(x_n)
        p[:, n, :] = idoa.p
        Out[n * hop_len : (n + 1) * hop_len] = output

    array2D = p[:, :, target_direction].T

    plt.figure(figsize=(14, 8))
    size = array2D.shape
    Y = np.arange(0, size[0], 1)
    X = np.arange(0, size[1], 1)
    X, Y = np.meshgrid(X, Y)

    im = plt.pcolormesh(X, Y, array2D, shading='auto')
    plt.colorbar(im)
    plt.show()

    plt.savefig('idoa_p.png')

    plt.figure(figsize=(14, 8))
    plt.plot(np.mean(p[64:128, :, target_direction], axis=0))
    plt.grid()
    plt.show()
    plt.savefig('p.png')

    save_audio('out_spp.wav', Out)
