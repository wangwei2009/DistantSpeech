"""
Test beamformer

"""

import argparse
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from scipy.io import wavfile

from DistantSpeech.beamformer.GSC import GSC
from DistantSpeech.beamformer.MicArray import MicArray
from DistantSpeech.beamformer.realtime_processing import realtime_processing
from DistantSpeech.beamformer.utils import load_wav, visual


def main(args):
    filepath = "test_audio/rec1/"
    print(os.chdir(sys.path[0]))
    x, sr = load_wav(os.path.abspath(filepath))  # [ch,sample]
    sr = 16000
    r = 0.032
    c = 343

    frameLen = 256
    hop = frameLen / 2
    overlap = frameLen - hop
    nfft = 256
    c = 340
    r = 0.032
    fs = sr

    start = time.process_time()

    MicArrayObj = MicArray(arrayType='circular', r=0.032, M=4)
    angle = np.array([197, 0]) / 180 * np.pi

    GSC_1 = GSC(MicArrayObj, frameLen, hop, nfft, c, r, fs)

    if args.file:
        yout = GSC_1.process(x, angle, method=3)

        end = time.process_time()
        print(end - start)

        # listen processed result
        if args.play:
            sd.default.channels = 1
            sd.play(yout['data'], fs)
            sd.wait()

        # view beampattern
        # mesh(yout['beampattern'])
        # pmesh(yout['beampattern'])

        # save audio
        if args.save:
            wavfile.write('output/output_gscbeamformer7_mcra_p.wav', 16000, yout['data'])

        visual(x[0, :], yout['data'])
        plt.title('spectrum')
    elif args.live:
        rec = realtime_processing(EnhancementMehtod=GSC_1, angle=angle, Recording=False)
        rec.audioDevice()
        print("Start processing...\n")
        rec.start()
        while True:
            a = int(input('"select algorithm: \n'
                          '0.src  \n'
                          '1.GSC  \n'))
            if a == 9:
                filename = 'output/rec1.wav'
                print('\nRecording finished: ' + repr(filename))
                rec.stop()
                rec.save(filename)
                break
            rec.changeAlgorithm(a)
            # time.sleep(0.1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Feature Extractor")
    parser.add_argument("-l", "--live", action='store_true', help="listen microphone")  # if set true
    parser.add_argument("-p", "--play", action='store_true', help="play output")  # if set true
    parser.add_argument("-s", "--save", action='store_true', help="set to save output")  # if set true
    parser.add_argument("-f", "--file", action='store_true', help="read from file")  # if set true

    args = parser.parse_args()
    main(args)
