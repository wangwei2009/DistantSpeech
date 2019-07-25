

import time
import pyaudio
import numpy as np
from beamformer.MicArray import MicArray
from beamformer.fixedbeamformer import fixedbeamformer
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
    r = 0.032
    fs = 16000

    MicArray = MicArray(arrayType='circular', r=0.032, M=4)
    angle = np.array([270, 0]) / 180 * np.pi

    fixedbeamformer = fixedbeamformer(MicArray, frameLen, hop, nfft, c, r, fs)
    # yout = fixedbeamformer.superDirectiveMVDR2(x,angle)

    rec = realtime_processing()
    rec.audioDevice()
    print("Start processing...\n")
    rec.start()
    while True:
        a = int(input('"select algorithm: \n0.src  \n1.delaysum  \n2.supperdirective\n'))
        rec.changeAlgorithm(a)
        time.sleep(0.1)