import sounddevice as sd
import soundfile as sf


sr = 16000
duration = 5
myrecording = sd.rec(int(duration * sr), samplerate=sr, channels=1)
sd.wait()
sd.play(myrecording, sr)
status = sd.wait()
# sf.write("New Record.wav", myrecording, sr)

# data, fs = sf.read("New Record.wav", dtype='float32')
# sd.play(data, fs)
# status = sd.wait()