import pyaudio
import time
import threading
import wave
import numpy as np

import time
import pyaudio
import numpy as np
from beamformer.fixedbeamformer import fixedbeamfomer
from beamformer.adaptivebeamformer import adaptivebeamfomer
from beamformer.utils import mesh,pmesh,load_wav
from beamformer.MicArray import MicArray


class realtime_processing(object):
    def __init__(self, chunk=1024, channels=6, rate=16000):
        self.CHUNK = chunk
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = channels
        self.RATE = rate
        self._running = True
        self._frames = []
        self.input_device_index = 0
        self.method = 0

    def audioDevice(self):
        pass

    def start(self):
        threading._start_new_thread(self.__recording, ())

    def __recording(self):
        r = 0.032
        c = 343

        frameLen = 256
        hop = frameLen / 2
        overlap = frameLen - hop
        nfft = 256
        c = 343
        r = 0.032
        fs = 16000

        Array = MicArray(arrayType='circular', r=0.032, M=4)
        angle = np.array([197, 0]) / 180 * np.pi

        fixedbeamformer = fixedbeamfomer(Array, frameLen, hop, nfft, c, r, fs)
        MVDR = adaptivebeamfomer(Array, frameLen, hop, nfft, c, r, fs)
        self._running = True
        self._frames = []
        p = pyaudio.PyAudio()
        streamOut = p.open(format=self.FORMAT,
                        channels=1,
                        rate=self.RATE,
                        input=False,
                        output=True,
                        # output_device_index=4,
                        frames_per_buffer=self.CHUNK)

        stream = p.open(format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        input=True,
                        output=False,
                        # output_device_index=4,
                        frames_per_buffer=self.CHUNK)
        while (self._running):
            data = stream.read(self.CHUNK)
            if self.CHANNELS == 6:
                # print(len(data))
                # samps = np.fromstring(data, dtype=np.int16)
                samps = np.fromstring(data, dtype='<i2').astype(np.float32, order='C') / 32768.0
                # samps = np.fromstring(data, dtype=np.float32)
                # print(len(samps))
                # start = time.clock()
                samps = np.reshape(samps, (self.CHUNK, 6))
                if self.method == 0:
                    method = 'src'
                elif self.method == 1:
                    method = 'DS'
                elif self.method == 2:
                    method = 'MVDR'
                yout = fixedbeamformer.process(samps[:, 1:5].T, angle,method)
                # yout = MVDR.process(samps[:, 1:5].T, angle, method)

                # samps = samps[:, 1]
                # samps = yout['out']
                samps = yout['data']
                data = (samps * 32768).astype('<i2').tostring()
                # data = samps.astype(np.int16).tostring()
                # end = time.clock()
                # print(end - start, '\n')

            streamOut.write(data, self.CHUNK)  # play back audio stream

        stream.stop_stream()
        stream.close()
        p.terminate()

    def stop(self):
        self._running = False
    def changeAlgorithm(self,index):
        self.method = index

    def save(self, filename):

        p = pyaudio.PyAudio()
        if not filename.endswith(".wav"):
            filename = filename + ".wav"
        wf = wave.open(filename, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(self._frames))
        wf.close()
        print("Saved")

