import pyaudio
import wave
import numpy as np

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 6
RATE = 16000
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "voice.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("* recording")

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    samps = np.fromstring(data, dtype='<i2').astype(np.float32, order='C') / 32768.0
    # start = time.clock()
    samps = np.reshape(samps, (CHUNK, 6))
    samps[:,3] = samps[:,3]/10

    data = (samps * 32768).astype('<i2').tostring()
    frames.append(data)

print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()