from transform.transform import transform
import librosa
from matplotlib import pyplot as plt
filepath = "test_audio/rec1/y_gsc.wav"
x1, sr = librosa.load(filepath, sr=None)

fs = 16000
frameLen = 256
hop = frameLen/2
overlap = frameLen - hop
nfft = 256

transform1 = transform(frameLength=frameLen,hop=hop,nfft=nfft,fs=fs,M=1)

X = transform1.stft(x1)
x = transform1.istft(X)

transform1.spectrum(X)

# plt.figure()
# plt.plot(x1)
# plt.figure()
# plt.plot(x)
# plt.show()
