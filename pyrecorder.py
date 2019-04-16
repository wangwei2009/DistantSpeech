import pyaudio
import time
import threading
import wave
import numpy as np
 
class Recorder():
    def __init__(self, chunk=1024, channels=1, rate=16000):
        self.CHUNK = chunk
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = channels
        self.RATE = rate
        self._running = True
        self._frames = []
        self.input_device_index = 0
    def audioDevice(self):
        p = pyaudio.PyAudio()
        # for i in range(p.get_device_count()):
        #     print(p.get_device_info_by_index(i))
        info = p.get_host_api_info_by_index(0)
        NumDevices = info.get('deviceCount')
        print("available input device:")
        for i in range(0, NumDevices):
            if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))
            if p.get_device_info_by_host_api_device_index(0, i).get('name') == 'ReSpeaker 4 Mic Array (UAC1.0) ':
                print('choose input:\n')
                print(p.get_device_info_by_index(i))
                self.input_device_index = i
                self.CHANNELS = 6
        print("default input:\n", p.get_default_input_device_info())
    def start(self):
        threading._start_new_thread(self.__recording, ())
    def __recording(self):
        self._running = True
        self._frames = []
        p = pyaudio.PyAudio()
        stream = p.open(format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        input=True,
                        frames_per_buffer=self.CHUNK,
                        input_device_index=self.input_device_index,
                        input_host_api_specific_stream_info=paWASAPI)
        while(self._running):
            data = stream.read(self.CHUNK)
            
            samps = np.fromstring(data, dtype=np.int16)
            #print(len(samps))
                        
            samps = np.reshape(samps, (1024, 1))
            samps = samps[:,0]
            data_frame = samps.astype(np.int16).tostring()
            self._frames.append(data)
 
        stream.stop_stream()
        stream.close()
        p.terminate()
 
    def stop(self):
        self._running = False
 
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
 
if __name__ == "__main__":
    
    for i in range(1,4):
        rec = Recorder()
        rec.audioDevice()
        a = int(input('请输入相应数字开始:'))
        if a == 1:
            begin = time.time()
            print("Start recording")
            rec.start()
            b = int(input('请输入相应数字停止:'))
            if b == 2:
                print("Stop recording")
                rec.stop()
                fina = time.time()
                t = fina - begin
                print('录音时间为%ds'%t)
                rec.save("2_%d.wav"%i)

