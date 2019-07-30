

import time
import pyaudio
import numpy as np
from beamformer.MicArray import MicArray
from beamformer.adaptivebeamformer import adaptivebeamfomer
from beamformer.utils import mesh,pmesh,load_wav

from beamformer.realtime_processing import realtime_processing

if __name__ == "__main__":

    r = 0.032
    c = 343

    frameLen = 256
    hop = frameLen / 2
    overlap = frameLen - hop
    nfft = 256
    c = 343
    r = 0.032*2
    fs = 16000

    MicArray = MicArray(arrayType='circular', r=r, c=c, M=4)
    angle = np.array([270, 0]) / 180 * np.pi

    adaptivebeamfomer = adaptivebeamfomer(MicArray, frameLen, hop, nfft, c, r, fs)

    rec = realtime_processing(EnhancementMehtod=adaptivebeamfomer,angle=angle)
    rec.audioDevice()
    print("Start processing...\n")
    rec.start()
    while True:
        a = int(input('"select algorithm: \n0.src  \n1.delaysum  \n2.MVDR\n'))
        rec.changeAlgorithm(a)
        # time.sleep(0.1)