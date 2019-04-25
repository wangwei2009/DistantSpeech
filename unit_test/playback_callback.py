"""PyAudio Example: Play a wave file (callback version)."""

import pyaudio
import wave
import time
import sys
from beamformer.utils import mesh,pmesh,load_wav
import numpy as np
from beamformer.MicArray import MicArray
from beamformer.fixedbeamformer import fixedbeamfomer
import time


if len(sys.argv) < 2:
    print("Plays a wave file.\n\nUsage: %s filename.wav" % sys.argv[0])
    sys.exit(-1)

filepath = "E:/work/matlab/Github/beamformer/sound/rec1/"
x,sr = load_wav(filepath)
r = 0.032
c = 343

frameLen = 256
hop = int(frameLen/2)
overlap = frameLen - hop
nfft = 256
c = 343
r = 0.032
fs = sr
WIDTH = 2
start = time.clock()

MicArray = MicArray(arrayType='circular', r=0.032, M=4)
angle = np.array([197, 0]) / 180 * np.pi

frameNum = round(x.shape[1]/hop)-1
wf = wave.open(sys.argv[1], 'rb')

# instantiate PyAudio (1)
p = pyaudio.PyAudio()

t = 0
fixedbeamformer = fixedbeamfomer(MicArray,256,128,nfft,c,r,fs)
# define callback (2)
def callback(in_data, frame_count, time_info, status):
    global t
    # data = wf.readframes(frame_count)
    data = x[:,t*frameLen:t*frameLen+frameLen]
    start = time.clock()
    yout = fixedbeamformer.superDirectiveMVDR2(data, angle)
    # samps = yout['out']
    samps = yout
    end = time.clock()
    print(end - start, '\n')
    # samps = data[1, :]

    data = (samps * 32768).astype('<i2').tostring()
    t = t+1
    return (data, pyaudio.paContinue)

# open stream using callback (3)
stream = p.open(format=p.get_format_from_width(WIDTH),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True,
                stream_callback=callback,
                frames_per_buffer=frameLen)

# start the stream (4)
stream.start_stream()

# wait for stream to finish (5)
while stream.is_active():
    time.sleep(0.1)

# stop stream (t_for6)
stream.stop_stream()
stream.close()
wf.close()

# close PyAudio (7)
p.terminate()