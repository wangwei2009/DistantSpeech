import numpy as np
from DistantSpeech.beamformer.MicArray import MicArray


def test_tau():
    r = 0.032
    M = 4
    c = 343

    theta = 60
    incident_ang = np.array([theta, 0]) / 180 * np.pi
    mic_array = MicArray(arrayType='linear', M=M, r=r, c=c)
    tau = mic_array.compute_tau(incident_ang)
    assert (tau[-1] - tau[0]) * c - (M - 1) * r * np.cos(incident_ang[0]) < 1e-6

    mic_array = MicArray(arrayType='circular', M=M, r=r, c=c)
    tau = mic_array.compute_tau(incident_ang)

    # check mic location
    assert (
        np.abs(
            np.linalg.norm(mic_array.mic_loc[0] - mic_array.mic_loc[1])
            - np.sqrt(r**2 + r**2 - 2 * r * r * np.cos(2 * np.pi / M))
        )
        < 1e-6
    )

    theta = 0
    incident_ang = np.array([theta, 0]) / 180 * np.pi
    mic_array = MicArray(arrayType='circular', M=M, r=r, c=c)
    tau = mic_array.compute_tau(incident_ang)
    assert (tau[0] - tau[1]) * c + r < 1e-6
