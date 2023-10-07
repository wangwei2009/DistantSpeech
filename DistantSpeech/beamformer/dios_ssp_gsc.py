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
from DistantSpeech.beamformer.utils import load_audio, save_audio
from dios_ssp_gsc_abm import objFGSCabm, dios_ssp_gsc_gscabm_process
from dios_ssp_gsc_aic import objFGSCaic, dios_ssp_gsc_gscaic_process
from DistantSpeech.beamformer.fixedbeamformer import TimeAlignment
from DistantSpeech.beamformer.MicArray import MicArray
from DistantSpeech.noise_estimation.mcra import NoiseEstimationMCRA


if __name__ == "__main__":

    mic_diameter = 0.032
    sound_speed = 343
    look_direction = [197, 0]

    fs = 16000
    M = 4
    angle = np.array(look_direction) / 180 * np.pi if isinstance(look_direction, list) else look_direction
    mic_array = MicArray(arrayType="circular", r=mic_diameter, M=M, n_fft=512)

    array_data = []
    for n in range(2, 6):
        filename = '/home/wangwei/work/DistantSpeech/samples/audio_samples/xmos/rec1/音轨-{}.wav'.format(n)
        data_ch = load_audio(filename)
        array_data.append(data_ch)
    array_data = np.array(array_data).T * 32768
    print(array_data.shape)

    time_alignment = TimeAlignment(mic_array, angle=angle)

    gscabm = objFGSCabm()
    gscaic = objFGSCaic()
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
    fbf = load_audio('/home/wangwei/work/athena-signal/examples/fbf.wav') * 32768

    bm = []
    for n in range(1, 5):
        filename = '/home/wangwei/work/athena-signal/examples/bm{}.wav'.format(n)
        data_ch = load_audio(filename) * 32768
        bm.append(data_ch)
    bm = np.array(bm).T
    print(bm.shape)

    # # fbf = load_audio('/home/wangwei/work/DistantSpeech/example/out_fbf.wav') * 32768
    # bm = load_audio('/home/wangwei/work/DistantSpeech/example/out_bm.wav') * 32768
    # print(bm.shape)

    output_bm = np.zeros(m_outSteering.shape)
    output_aic = np.zeros((m_outSteering.shape[0],))

    ctrl_abm[:] = 1
    ctrl_aic[:] = 1

    for n in range(ctrl_abm.shape[0]):
        x_n = array_data[n * 16 : (n + 1) * 16, :]
        x_aligned = time_alignment.process(x_n)
        m_outFBF[n * 16 : (n + 1) * 16] = np.mean(x_aligned, axis=1)
        output_bm[n * 16 : (n + 1) * 16, :] = dios_ssp_gsc_gscabm_process(
            gscabm,
            # m_outSteering[n * 16 : (n + 1) * 16, :].T,
            x_aligned.T,
            m_outFBF[n * 16 : (n + 1) * 16],
            0,
            ctrl_abm[n, :],
            ctrl_aic[n, :],
        )
        output_aic[n * 16 : (n + 1) * 16] = dios_ssp_gsc_gscaic_process(
            gscaic,
            m_outFBF[n * 16 : (n + 1) * 16],
            output_bm[n * 16 : (n + 1) * 16, :].T,
            0,
            ctrl_abm[n, :],
            ctrl_aic[n, :],
        )

    # Hf = ifft(gscabm.Hf[:, 0, : ], axis=-1)
    # np.save('hf.npy', Hf)

    save_audio('output.wav', output_aic / 32768)
