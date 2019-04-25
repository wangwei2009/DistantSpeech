"""
PyAudio Example: Make a wire between input and output (i.e., record a
few samples and play them back immediately).
"""

import pyaudio
import numpy as np

CHUNK = 1024
WIDTH = 2
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 10

p = pyaudio.PyAudio()

stream = p.open(format=p.get_format_from_width(WIDTH),
                channels=CHANNELS,
                rate=RATE,
                input=True,
                output=True,
                # output_device_index=4,
                frames_per_buffer=CHUNK)
streamOut = p.open(format=p.get_format_from_width(WIDTH),
                channels=1,
                rate=RATE,
                input=False,
                output=True,
                # output_device_index=4,
                frames_per_buffer=CHUNK)

print("* recording")

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)  #read audio stream

    if CHANNELS == 6:
         # print(len(data))
        samps = np.fromstring(data, dtype=np.int16)
        # print(len(samps))

        samps = np.reshape(samps, (1024, 6))
        samps = samps[:, 0]
        data = samps.astype(np.int16).tostring()

    streamOut.write(data, CHUNK)  #play back audio stream

print("* done")

stream.stop_stream()
stream.close()

p.terminate()