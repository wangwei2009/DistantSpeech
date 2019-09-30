import pyaudio
import time
import threading
import wave
import numpy as np

import time
import pyaudio
import numpy as np
from beamformer.fixedbeamformer import fixedbeamformer

class realtime_processing(object):
    def __init__(self, EnhancementMehtod=fixedbeamformer, angle=0,chunk=1024, channels=6, rate=16000,Recording=False):
        self.CHUNK = chunk
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = channels
        self.RATE = rate
        self._running = True
        self._frames = []
        self.input_device_index = 0
        self.method = 0
        self.EnhancementMethod = EnhancementMehtod
        self.angle = angle
        self.isRecording = Recording

    def audioDevice(self):
        pass

    def start(self):
        if self.isRecording:
            print('Recording...\n')
        threading._start_new_thread(self.__recording, ())

    def __recording(self):
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
            MultiChannelData = np.zeros((self.CHUNK, 6), dtype=float)
            if self.CHANNELS == 6:

                samps = np.fromstring(data, dtype='<i2').astype(np.float32, order='C') / 32768.0
                # start = time.clock()
                MultiChannelData = np.reshape(samps, (self.CHUNK, 6))

                yout = self.EnhancementMethod.process(MultiChannelData[:, 1:5].T, self.angle,self.method)
                MultiChannelData[:,5] = yout['data']
                data = (MultiChannelData[:,5] * 32768).astype('<i2').tostring()
                # end = time.clock()
                # print(end - start, '\n')
                if self.isRecording:
                    MultiChannelPCM = (MultiChannelData * 32768).astype('<i2').tostring()
                    self._frames.append(MultiChannelPCM)

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
        wf.setnchannels(6)
        wf.setsampwidth(p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(self._frames))
        wf.close()
        print("Saved")

