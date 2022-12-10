# from matplotlib.pyplot import angle_spectrum
from tqdm import tqdm
import numpy as np
from DistantSpeech.beamformer.MicArray import MicArray
from DistantSpeech.transform.transform import Transform
from DistantSpeech.beamformer.utils import pmesh
from DistantSpeech.noise_estimation.mcra import NoiseEstimationMCRA


class srp(object):
    def __init__(self, mic_array: MicArray) -> None:
        self.mic_array = mic_array
        self.transform = Transform(channel=mic_array.M, n_fft=mic_array.n_fft, hop_length=int(mic_array.n_fft / 2))
        self.spp = NoiseEstimationMCRA(nfft=mic_array.n_fft)
        self.spp.L = 65

    def compute_angle_spectrum(self, x, phat=True, resolution=1):
        """steer response power

        Parameters
        ----------
        x : np.array
            multichannel input data, [samples, chs]
        phat : bool, optional
            phase-transform, by default True
        resolution : int, optional
            angle interval, by default 1

        Returns
        -------
        angle_spectrum : np.array
            angle spectrum, [360, n_frame]
        """
        y = self.transform.stft(x)  # [bins, frames, M]
        n_frame = y.shape[1]
        p = np.zeros(y[..., 0].shape)

        for n in range(n_frame):
            self.spp.estimation(y[:, n, 0])
            p[:, n] = self.spp.p

        p_frame = np.mean(p[32:64], axis=0)

        angle_spectrum = np.zeros((360, n_frame))
        for angle in tqdm(np.arange(0, 360, resolution)):
            steer_vecotr = self.mic_array.steering_vector(look_direction=angle)  # [bin, M]
            for n in range(n_frame):
                y_p = steer_vecotr.conj() * y[:, n, :]
                if phat:
                    y_p = y_p / (np.abs(y_p) + 1e-6)
                angle_spectrum[angle : angle + resolution, n] = np.sum(np.abs(np.sum(y_p, axis=-1)))

        return angle_spectrum, p

    def show(self, x):
        angle_spectrum = self.compute_angle_spectrum(x)
        pmesh(angle_spectrum)


if __name__ == "__main__":
    from tqdm import tqdm
    from matplotlib import pyplot as plt
    from DistantSpeech.beamformer.utils import load_audio as audioread
    from DistantSpeech.beamformer.utils import pmesh
    from librosa import power_to_db

    array_data = []
    for n in range(2, 6):
        filename = '/home/wangwei/work/DistantSpeech/samples/audio_samples/xmos/rec1/音轨-{}.wav'.format(n)
        data_ch = audioread(filename)
        array_data.append(data_ch)
    array_data = np.transpose(np.array(array_data))
    print(array_data.shape)

    mic_array = MicArray(arrayType='circular', r=0.032, M=4, n_fft=512)
    srp_obj = srp(mic_array)

    angle_spectrum = srp_obj.compute_angle_spectrum(array_data, resolution=1)
    pmesh(power_to_db(angle_spectrum))
    plt.savefig('angle_spectrum.png')
