"""
PyAudio Example: Make a wire between input and output (i.e., record a
few samples and play them back immediately).
"""
import time
import pyaudio
import numpy as np
from beamformer.MicArray import MicArray
from beamformer.fixedbeamformer import fixedbeamfomer
from beamformer.adaptivebeamformer import adaptivebeamfomer
from beamformer.utils import mesh, pmesh, load_wav


r = 0.032
c = 343

frameLen = 256
hop = frameLen / 2
overlap = frameLen - hop
nfft = 256
c = 343
r = 0.032
fs = 16000

MicArray = MicArray(arrayType='circular', r=0.032, M=4)
angle = np.array([270, 0]) / 180 * np.pi

fixedbeamformer = fixedbeamfomer(MicArray, frameLen, hop, nfft, c, r, fs)
# yout = fixedbeamformer.superDirectiveMVDR2(x,angle)


CHUNK = 2048
WIDTH = 2
CHANNELS = 6
RATE = 16000
RECORD_SECONDS = 10

p = pyaudio.PyAudio()

streamOut = p.open(
    format=p.get_format_from_width(WIDTH),
    channels=1,
    rate=RATE,
    input=False,
    output=True,
    # output_device_index=4,
    frames_per_buffer=CHUNK,
)

stream = p.open(
    format=p.get_format_from_width(WIDTH),
    channels=CHANNELS,
    rate=RATE,
    input=True,
    output=False,
    # output_device_index=4,
    frames_per_buffer=CHUNK,
)


print("* recording")

while True:
    data = stream.read(CHUNK)  # read audio stream

    if CHANNELS == 6:
        # print(len(data))
        # samps = np.fromstring(data, dtype=np.int16)
        samps = np.fromstring(data, dtype='<i2').astype(np.float32, order='C') / 32768.0
        # samps = np.fromstring(data, dtype=np.float32)
        # print(len(samps))
        start = time.clock()
        samps = np.reshape(samps, (CHUNK, 6))
        method = 'MVDR'
        yout = fixedbeamformer.process(samps[:, 1:5].T, angle, method)

        samps = samps[:, 1]
        # samps = yout['out']
        samps = yout
        data = (samps * 32768).astype('<i2').tostring()
        # data = samps.astype(np.int16).tostring()
        end = time.clock()
        print(end - start, '\n')

    streamOut.write(data, CHUNK)  # play back audio stream

print("* done")

stream.stop_stream()
stream.close()

p.terminate()
