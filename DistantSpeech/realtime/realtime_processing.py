import threading
import wave
import time

import pyaudio
import numpy as np


class realtime_processing(object):
    def __init__(
        self,
        EnhancementMehtod=None,
        angle=0,
        chunk=1024,
        channels=6,
        rate=16000,
        Recording=False,
        duplex=False,
        save_rec_to_file=False,
    ):
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
        self.save_rec_to_file = save_rec_to_file
        if self.save_rec_to_file:
            self.wf = wave.open('output.wav', 'wb')
            self.wf.setnchannels(6)
            self.wf.setsampwidth(2)
            self.wf.setframerate(self.RATE)
        self.duplex = duplex

        self.p = pyaudio.PyAudio()
        input_device = self.get_audio_input_devices()
        print(input_device)

    def audioDevice(self):
        pass

    def get_audio_devices(self):
        p = pyaudio.PyAudio()
        devices = []
        for i in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(i)
            print(f"Device {i}: {device_info}")
            devices.append(p.get_device_info_by_index(i))
        return devices

    def get_audio_input_devices(self):
        devices = []
        for item in self.get_audio_devices():
            if item.get('maxInputChannels') == 6:
                devices.append(item)
        return devices

    def get_audio_output_devices(self):
        devices = []
        for item in self.get_audio_devices():
            if item.get('maxOutputChannels') > 0:
                devices.append(item)
        return devices

    def start(self):
        if self.isRecording:
            print('Recording...0\n')
            #print(self.get_audio_input_devices())
            rec_thread = threading.Thread(target=self.__recording)
            print('Recording...1\n')
            rec_thread.start()

    def process(self, data):
    
        if self.EnhancementMethod is None:
            return data[:, 1]
        else:
            output = self.EnhancementMethod.process(data)
            return output["data"]

    def __recording(self):
        self._running = True
        self._frames = []
        print('Recording...2\n')

        if self.duplex:
            streamOut = self.p.open(
                format=self.FORMAT,
                channels=1,
                rate=self.RATE,
                input=False,
                output=True,
                # output_device_index=4,
                frames_per_buffer=self.CHUNK,
            )

        stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            output=False,
            input_device_index=3,
            frames_per_buffer=self.CHUNK,
        )
        print('Recording...\n')
        try:
            while self._running:
                self._frames = []
                data = stream.read(self.CHUNK)
                
                if self.CHANNELS == 6:

                    samps = np.frombuffer(data, dtype='<i2').astype(np.float32, order='C') / 32768.0
                    start = time.perf_counter()
                    MultiChannelData = np.reshape(samps, (self.CHUNK, 6))

                    MultiChannelData[:, 5] = self.process(MultiChannelData[:, 1:5])
                    end = time.perf_counter()
                    time_cost = end - start
                    if time_cost > self.CHUNK / 16000:
                        print(f'time_cost overflow: {time_cost}')
                    if self.save_rec_to_file:
                        MultiChannelPCM = (MultiChannelData * 32768).astype('<i2').tobytes()
                        self._frames.append(MultiChannelPCM)
                        self.wf.writeframes(b''.join(self._frames))
                    else:
                        data = (MultiChannelData[:, 5] * 32768).astype('<i2').tobytes()

                if self.duplex:
                    streamOut.write(data, self.CHUNK)  # play back audio stream

        except KeyboardInterrupt:
            print('stop recording...')
            stream.stop_stream()
            stream.close()
            if self.duplex:
                streamOut.stop_stream()
                streamOut.close()

            if self.save_rec_to_file:
                self.wf.close()

            self.p.terminate()

    def stop(self):
        self._running = False

    def changeAlgorithm(self, index):
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


if __name__ == "__main__":

    ptr = realtime_processing(Recording=True, save_rec_to_file=True)
    ptr.start()
