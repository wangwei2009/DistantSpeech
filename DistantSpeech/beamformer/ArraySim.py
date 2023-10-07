import argparse
import numpy as np
import pyroomacoustics as pra
from DistantSpeech.beamformer.utils import save_audio as audiowrite
from DistantSpeech.beamformer.utils import load_audio as audioread


def unit_vec3D(phi):
    return np.array([[np.cos(phi), np.sin(phi), 0]]).T


def linear_3d_array(center, M, phi, d):
    """
    Creates an array of uniformly spaced linear points in 3D

    Parameters
    ----------
    center: array_like
        The center of the array
    M: int
        The number of points
    phi: float
        The counterclockwise rotation of the array (from the x-axis)
    d: float
        The distance between neighboring points

    Returns
    -------
    ndarray (3, M)
        The array of points
    """
    # get array unit vector, (1, 0, 0) represent array lies in x-axis
    u = unit_vec3D(phi)

    # equal spacing
    a = d * (np.arange(M)[np.newaxis, :] - (M - 1.0) / 2.0)

    location = a * -u + np.array(center)[:, np.newaxis]

    return location  # (3, M)  mic-1 in +x-axis


def circular_3d_array(center, M, phi0, radius):
    """
    Creates an array of uniformly spaced circular points in 2D

    Parameters
    ----------
    center: array_like
        The center of the array
    M: int
        The number of points
    phi0: float
        The counterclockwise rotation of the first element in the array (from
        the x-axis)
    radius: float
        The radius of the array

    Returns
    -------
    ndarray (3, M)
        The array of points
    """
    phi = np.arange(M) * 2.0 * np.pi / M
    return np.array(center)[:, np.newaxis] + radius * np.vstack((np.cos(phi + phi0), np.sin(phi + phi0), np.zeros(M)))


def cart2sph(x, y, z):
    x_pow2 = x**2
    y_pow2 = y**2
    azimuth = np.arctan2(y, x)
    elevation = np.arctan2(z, np.sqrt(x_pow2 + y_pow2))
    r = np.sqrt(x_pow2 + y_pow2 + z**2)
    return azimuth, elevation, r


def sph2cart(azimuth, elevation, r):
    """

    :param azimuth: 0 lines in x-axis
    :param elevation: 0 lines in x-y plane
    :param r:
    :return:
    """
    cos_elevation = np.cos(elevation)  # 0 lines in x-y plane
    x = r * cos_elevation * np.cos(azimuth)  # 0 lines in x-axis
    y = r * cos_elevation * np.sin(azimuth)  # 90 lines in y-axis
    z = r * np.sin(elevation)
    return [x, y, z]


def generate_rir(
    room_sz=[3, 3, 2.5],
    pos_src=np.array([[1, 2.9, 0.5], [1, 2, 0.5]]),
    pos_rcv=np.array([[0.5, 1, 0.5], [1, 1, 0.5], [1.5, 1, 0.5]]),
    T60=1.0,
):
    """use gpuRIR to generate RIRs

    Parameters
    ----------
    room_sz : list, optional
        Size of the room [m], by default [3, 3, 2.5]
    pos_src : np.array, optional
        Positions of the sources [m], [num_src, 3], by default np.array([[1, 2.9, 0.5], [1, 2, 0.5]])
    pos_rcv : np.array, optional
        Position of the receivers [m], [num_rcv, 3], by default np.array([[0.5, 1, 0.5], [1, 1, 0.5], [1.5, 1, 0.5]])
    T60 : float, optional
        Time for the RIR to reach 60dB of attenuation [s], by default 1.0

    Returns
    -------
    RIRs : np.array
        3D ndarray The first axis is the source, the second the receiver and the third the time, [num_src, num_rcv, samples]
    """

    import gpuRIR

    nb_src = pos_src.shape[0]  # Number of sources
    nb_rcv = pos_rcv.shape[0]  # Number of receivers

    # orV_rcv = np.matlib.repmat(np.array([0,1,0]), nb_rcv, 1) # Vectors pointing in the same direction than the receivers
    mic_pattern = "omni"  # Receiver polar pattern
    abs_weights = [0.9] * 5 + [0.5]  # Absortion coefficient ratios of the walls
    # T60 = 0.8	 # Time for the RIR to reach 60dB of attenuation [s]
    att_diff = 15.0  # Attenuation when start using the diffuse reverberation model [dB]
    att_max = 60.0  # Attenuation at the end of the simulation [dB]
    fs = 16000.0  # Sampling frequency [Hz]

    beta = gpuRIR.beta_SabineEstimation(room_sz, T60, abs_weights=abs_weights)  # Reflection coefficients
    Tdiff = gpuRIR.att2t_SabineEstimator(att_diff, T60)  # Time to start the diffuse reverberation model [s]
    Tmax = gpuRIR.att2t_SabineEstimator(att_max, T60)  # Time to stop the simulation [s]
    nb_img = gpuRIR.t2n(Tdiff, room_sz)  # Number of image sources in each dimension
    RIRs = gpuRIR.simulateRIR(room_sz, beta, pos_src, pos_rcv, nb_img, Tmax, fs, Tdiff=Tdiff, mic_pattern=mic_pattern)

    t = np.arange(int(np.ceil(Tmax * fs))) / fs

    return RIRs, t


def callback_mix(premix, snr=0, sir=0, ref_mic=0, n_src=None, n_tgt=None):

    # first normalize all separate recording to have unit power at microphone one
    p_mic_ref = np.std(premix[:, ref_mic, :], axis=1)
    premix /= p_mic_ref[:, None, None]

    # now compute the power of interference signal needed to achieve desired SIR
    if n_src > n_tgt:
        sigma_i = np.sqrt(10 ** (-sir / 10) / (n_src - n_tgt))
        premix[n_tgt:n_src, :, :] *= sigma_i

    max_value = np.max(np.abs(premix))

    # compute noise variance
    sigma_n = np.sqrt(10 ** (-snr / 10))

    # Mix down the recorded signals
    mix = np.sum(premix[:n_src, :], axis=0) + sigma_n * np.random.randn(*premix.shape[1:])

    mix /= max_value
    premix /= max_value

    return mix


class ArraySim(object):
    def __init__(
        self,
        array_type="linear",
        M=3,
        spacing=0.032,
        coordinate=None,
        fs=16000,
        energy_absorption=0.7,
        room_size=[5.0, 3.0, 3.0],
        anechoic=False,
    ):
        assert array_type in ["linear", "circular", "arbitrary"]

        self.room_size = room_size

        self.corners = np.array([[0, 0], [0, 3], [5, 3], [5, 0]]).T  # [x,y]
        if room_size is not None:
            self.corners[0, 2] = room_size[0]
            self.corners[0, 3] = room_size[0]
            self.corners[1, 1] = room_size[1]
            self.corners[1, 2] = room_size[1]

        self.height = 3.0

        self.center_loc = np.zeros((3,))
        self.center_loc[0] = (self.corners[0, 0] + self.corners[0, 2]) / 2
        self.center_loc[1] = (self.corners[1, 0] + self.corners[1, 2]) / 2
        self.center_loc[2] = 0.5

        if coordinate is not None:
            self.R = coordinate.T + self.center_loc[:, None]
        else:
            if array_type == 'linear':
                self.R = linear_3d_array(self.center_loc, M, 0, spacing)
            if array_type == 'circular':
                self.R = circular_3d_array(self.center_loc, M, 0, spacing)

        if anechoic:
            self.room = pra.AnechoicRoom(fs=fs)
        else:
            # set max_order to a low value for a quick (but less accurate) RIR
            self.room = pra.Room.from_corners(
                self.corners,
                fs=fs,
                max_order=3,
                materials=pra.Material(energy_absorption, 0.15),
                ray_tracing=True,
                air_absorption=True,
            )
            self.room.extrude(self.height, materials=pra.Material(0.7, 0.15))

            # Set the ray tracing parameters
            self.room.set_ray_tracing(receiver_radius=0.1, n_rays=10000, energy_thres=1e-5)

        # add 3-microphone array
        self.room.add_microphone(self.R)

    def save_audio(self, filename, fs=16000):
        signals_data = self.room.mic_array.signals

        audiowrite(filename, signals_data.transpose(), fs)

    def get_audio(self):
        """
        get simulation data, should only be called after generate_audio()
        :return:

        """

        return self.room.mic_array.signals.transpose()  # (n_samples, n_channels)

    def generate_audio(
        self,
        source_signal,
        interference=None,
        source_angle=90,
        source_distance=1.0,
        interf_angle=30,
        interf_distance=1.5,
        snr=30,
        sir=15,
        return_premix=True,
        backend='pyroomacoustic',
        verbose=False,
    ):
        assert backend in ['pyroomacoustic', 'gpuRIR']

        if interference is not None:
            interf = interference
        else:
            interf = np.random.random(len(source_signal)) / 10

        # # SNR = 10log10(sum(s.^2)/sum(n^2))
        # noise_scale = np.sqrt(np.sum(source_signal**2) / (np.sum(interf**2) * np.power(10, snr / 10.0) + 1e-6))
        # interf = noise_scale * interf

        source_location = sph2cart(source_angle * np.pi / 180, 0, source_distance)
        source_location += self.center_loc

        interf_location = sph2cart(interf_angle * np.pi / 180, 0, interf_distance)
        interf_location += self.center_loc

        if verbose:
            print('source_location:{}'.format(source_location))
            print('interf_location:{}'.format(interf_location))
            print('center_loc:{}'.format(self.center_loc))
            print('source_location_real:{}'.format(source_location))
            print('interf_location_real:{}'.format(interf_location))

        # add source and set the signal to WAV file content
        self.room.add_source(source_location, signal=source_signal)
        if interference is not None:
            self.room.add_source(interf_location, signal=interf)

        # compute image sources
        self.room.image_source_model()  # [num_mic, num_src, samples]

        # print(self.room.rir.shape)

        self.room.plot_rir()
        t60 = pra.experimental.measure_rt60(self.room.rir[0][0], fs=self.room.fs)
        print(f"The RT60 is {t60 * 1000:.0f} ms")
        # fig = plt.gcf()
        # fig.set_size_inches(20, 10)
        # plt.show()

        if interference is not None:
            # the extra arguments are given in a dictionary
            callback_mix_kwargs = {
                'snr': snr,  # SNR target is 30 decibels
                'sir': sir,  # SIR target is 10 decibels
                'n_src': 2,
                'n_tgt': 1,
                'ref_mic': 0,
            }
        else:
            # the extra arguments are given in a dictionary
            callback_mix_kwargs = {
                'snr': snr,  # SNR target is 30 decibels
                'n_src': 1,
                'n_tgt': 1,
                'ref_mic': 0,
            }
        if backend == 'gpuRIR':
            sources_pos = []
            for source in range(len(np.array(self.room.sources))):
                sources_pos.append(self.room.sources[source].position)

            RIRs, _ = generate_rir(self.room_size, np.array(sources_pos), self.R.T, T60=t60)
            print('RIRs:{}'.format(RIRs.shape))
            min_len = np.minimum(self.room.rir[0][0].shape[0], RIRs[0][0].shape[0])
            for source in range(callback_mix_kwargs['n_src']):
                for mic in range(self.R.shape[1]):
                    min_len = np.minimum(self.room.rir[mic][source].shape[0], RIRs[source][mic].shape[0])
                    self.room.rir[mic][source][:min_len] = RIRs[source][mic][:min_len]
                    # self.room.rir[mic][source][RIRs[source][mic].shape[0] :] = 0
                    # self.room.rir[mic][source][200:] = 0

        # callback_mix_kwargs = Callback_mix_kwargs(snr=snr, sir=sir)
        premix = self.room.simulate(
            callback_mix=callback_mix, callback_mix_kwargs=callback_mix_kwargs, return_premix=return_premix
        )
        # print(room.mic_array.signals.shape)

        signals_reverb = self.room.mic_array.signals  # [n_ch, samples]

        signals_reverb = signals_reverb[:, : len(source_signal)]

        return signals_reverb, premix


def generate_audio(
    array_type="linear",
    spacing=0.032,
    coordinate=None,
    fs=16000,
    target=None,
    interf=None,
    sir=20,
    snr=20,
):
    assert array_type in ["linear", "circular", "arbitrary"]
    # corners = np.array([[0, 0], [0, 3], [5, 3], [5, 1], [3, 1], [3, 0]]).T  # [x,y]
    corners = np.array([[0, 0], [0, 3], [5, 3], [5, 0]]).T  # [x,y]

    height = 3.0

    center_loc = np.mean(corners, axis=0)

    room = pra.Room.from_corners(corners)
    room.extrude(height)

    # fig, ax = room.plot()
    # ax.set_xlim([0, 5])
    # ax.set_ylim([0, 3])
    # ax.set_zlim([0, 2])
    # plt.show()

    # specify signal source
    signal = target

    # set max_order to a low value for a quick (but less accurate) RIR
    room = pra.Room.from_corners(
        corners,
        fs=fs,
        max_order=8,
        materials=pra.Material(0.2, 0.15),
        ray_tracing=True,
        air_absorption=True,
    )
    room.extrude(height, materials=pra.Material(0.2, 0.15))

    # Set the ray tracing parameters
    room.set_ray_tracing(receiver_radius=0.5, n_rays=10000, energy_thres=1e-5)

    # add source and set the signal to WAV file content
    room.add_source([3.50, 1.0, 0.5], signal=signal)

    # add two-microphone array
    R = np.array([[3.45, 3.50, 3.55], [2.0, 2.0, 2.0], [0.5, 0.5, 0.5]])  # [[x], [y], [z]]
    room.add_microphone(R)

    # compute image sources
    room.image_source_model()

    # visualize 3D polyhedron room and image sources
    fig, ax = room.plot(img_order=3)
    fig.set_size_inches(18.5, 10.5)

    # plt.show()

    room.plot_rir()
    # fig = plt.gcf()
    # fig.set_size_inches(20, 10)

    # plt.show()

    t60 = pra.experimental.measure_rt60(room.rir[0][0], fs=room.fs)
    print(f"The RT60 is {t60 * 1000:.0f} ms")

    room.simulate()
    print(room.mic_array.signals.shape)

    signals_reverb = room.mic_array.signals  # [n_ch, samples]
    # signals_reverb = signals_reverb.astype(np.float32)
    # wavfile.write('clean_speech_reverb.wav', fs, signals_reverb.T)

    signals_reverb = signals_reverb[:, : len(signal)]

    return signals_reverb


def main(args):
    signal = audioread("samples/audio_samples/cleanspeech_aishell3.wav")
    fs = 16000
    mic_array = ArraySim(array_type="linear", spacing=0.05)
    array_data, premix = mic_array.generate_audio(signal)
    # audiowrite('mix.wav', np.transpose(array_data))
    # audiowrite('premix.wav', np.transpose(premix[1, ...]))
    print(array_data.shape)
    print(premix.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Feature Extractor")
    parser.add_argument("-l", "--listen", action="store_true", help="set to listen output")  # if set true
    parser.add_argument("-s", "--save", action="store_true", help="set to save output")  # if set true

    args = parser.parse_args()
    main(args)
