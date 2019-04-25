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
from beamformer.utils import mesh,pmesh,load_wav



r = 0.032
c = 343

frameLen = 256
hop = frameLen/2
overlap = frameLen - hop
nfft = 256
c = 343
r = 0.032
fs = 16000

MicArray = MicArray(arrayType='circular', r=0.032, M=4)
angle = np.array([197, 0]) / 180 * np.pi

fixedbeamformer = fixedbeamfomer(MicArray,frameLen,hop,nfft,c,r,fs)
# yout = fixedbeamformer.superDirectiveMVDR2(x,angle)


CHUNK = 16384
WIDTH = 2
CHANNELS = 6
RATE = 16000
RECORD_SECONDS = 10

p = pyaudio.PyAudio()

streamOut = p.open(format=p.get_format_from_width(WIDTH),
                channels=1,
                rate=RATE,
                input=False,
                output=True,
                # output_device_index=4,
                frames_per_buffer=CHUNK)

# define callback (2)
def callback(in_data, frame_count, time_info, status):
    # data = wf.readframes(frame_count)
    if CHANNELS == 6:
         # print(len(data))
        samps = np.fromstring(in_data, dtype=np.int16)
        # print(len(samps))
        start = time.clock()
        samps = np.reshape(samps, (CHUNK, 6))
        yout = fixedbeamformer.superDirectiveMVDR(samps[:,0:4].T, angle)

        # samps = samps[:, 0]
        samps = yout['out']
        data = samps.astype(np.int16).tostring()
        end = time.clock()
        streamOut.write(data, CHUNK)  # play back audio stream
        print(end - start,'\n')
    return (data, pyaudio.paContinue)
stream = p.open(format=p.get_format_from_width(WIDTH),
                channels=CHANNELS,
                rate=RATE,
                input=True,
                output=False,
                # output_device_index=4,
                frames_per_buffer=CHUNK,
                stream_callback=callback)


print("* recording")

stream.start_stream()

while stream.is_active():
    time.sleep(0.1)

stream.stop_stream()
stream.close()

p.terminate()

# for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
#     data = stream.read(CHUNK)  #read audio stream
#
#     if CHANNELS == 6:
#          # print(len(data))
#         samps = np.fromstring(data, dtype=np.int16)
#         # print(len(samps))
#         start = time.clock()
#         samps = np.reshape(samps, (CHUNK, 6))
#         yout = fixedbeamformer.superDirectiveMVDR(samps[:,0:4].T, angle)
#
#         # samps = samps[:, 0]
#         samps = yout['out']
#         data = samps.astype(np.int16).tostring()
#         end = time.clock()
#         print(end - start,'\n')
#
#     # streamOut.write(data, CHUNK)  #play back audio stream
#
# print("* done")
#
# stream.stop_stream()
# stream.close()
#
# p.terminate()