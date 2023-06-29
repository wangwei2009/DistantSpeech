"""
  stream test for adaptive beamformer
"""

from DistantSpeech.beamformer.MicArray import MicArray

from DistantSpeech.beamformer.realtime_processing import realtime_processing
from DistantSpeech.beamformer.FDGSC import FDGSC

if __name__ == "__main__":

    r = 0.032
    c = 343

    frameLen = 256
    hop = frameLen / 2
    overlap = frameLen - hop
    c = 343
    r = 0.032
    fs = 16000

    mic_array = MicArray(arrayType='circular', r=r, c=c, M=4)
    look_direction = 180

    GSC_1 = FDGSC(mic_array, frameLen=frameLen, angle=[look_direction, 0])

    rec = realtime_processing(EnhancementMehtod=GSC_1, angle=look_direction, Recording=False)
    rec.audioDevice()
    print("Start processing...\n")
    rec.start()
