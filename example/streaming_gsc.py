"""
  stream test for adaptive beamformer
"""

import time
import pyaudio
import numpy as np
from beamformer.MicArray import MicArray
from beamformer.adaptivebeamformer import adaptivebeamfomer
from beamformer.utils import mesh,pmesh,load_wav
from beamformer.GSC import GSC

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

    MicArray = MicArray(arrayType='circular', r=r, c=c, M=4)
    angle = np.array([270, 0]) / 180 * np.pi

    GSC_1 = GSC(MicArray, frameLen, hop, nfft, c, r, fs)

    rec = realtime_processing(EnhancementMehtod=GSC_1,angle=angle,Recording=True)
    rec.audioDevice()
    print("Start processing...\n")
    rec.start()
    while True:
        a = int(input('"select algorithm: \n'
                      '0.src  \n'
                      '1.GSC  \n'))
        if a==9:
            filename = 'output/rec1.wav'
            print('\nRecording finished: ' + repr(filename))
            rec.stop()
            rec.save(filename)
            break
    rec.changeAlgorithm(a)
        # time.sleep(0.1)