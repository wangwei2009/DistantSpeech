from matplotlib.pyplot import angle_spectrum

import numpy as np
from DistantSpeech.beamformer.MicArray import MicArray
from DistantSpeech.transform.transform import Transform
from DistantSpeech.beamformer.utils import pmesh


class srp(object):
    def __init__(self, mic_array: MicArray) -> None:
        self.mic_array = mic_array
        self.transform = Transform(channel=mic_array.M, n_fft=mic_array.n_fft, hop_length=int(mic_array.n_fft / 2))

    def compute_angle_spectrum(self, x, phat=True, resolution=1):
        y = self.transform.stft(x)  # [bins, frames, M]

        n_frame = y.shape[1]
        angle_spectrum = np.zeros((360, n_frame))
        for n in tqdm(range(n_frame)):
            y_n = y[:, n, :]
            for angle in np.arange(0, 360, resolution):
                # for angle in range(360):
                steer_vecotr = self.mic_array.steering_vector(look_direction=angle)  # [M, bin]
                if phat:
                    angle_spectrum[angle : angle * resolution, n] = np.sum(
                        np.abs(np.sum(steer_vecotr.conj().T * y_n, axis=-1) ** 2)
                    )
                else:
                    angle_spectrum[angle : angle * resolution, n] = np.sum(
                        np.abs(np.sum(steer_vecotr.conj().T * y_n / np.abs(y_n.T), axis=-1) ** 2)
                    )

        # np.einsum('ij,il->ijl', target_n, target_n.conj())

        return angle_spectrum

    def show(self, x):
        angle_spectrum = self.compute_angle_spectrum(x)
        pmesh(angle_spectrum)


if __name__ == "__main__":
    from tqdm import tqdm
    from matplotlib import pyplot as plt
    from DistantSpeech.beamformer.utils import load_audio as audioread
    from DistantSpeech.beamformer.utils import pmesh
    from librosa import power_to_db

    audio_path = '/home/wangwei/work/DistantSpeech/example/mix.wav'
    audiodata = audioread(audio_path)
    print(audiodata.shape)

    mic_array = MicArray(arrayType='circular', r=0.05, M=6)
    srp_obj = srp(mic_array)

    # srp_obj.show()
    angle_spectrum = srp_obj.compute_angle_spectrum(audiodata, resolution=10)
    pmesh(power_to_db(angle_spectrum))
    plt.savefig('angle_spectrum.png')
