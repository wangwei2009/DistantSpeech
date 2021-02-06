"""
Test beamformer

"""

import numpy as np
import os
import sys

import time

from DistantSpeech.beamformer.MicArray import MicArray
from DistantSpeech.beamformer.GSC import GSC
from DistantSpeech.beamformer.utils import mesh,pmesh,load_wav,load_pcm, visual

import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
from scipy.io import wavfile
import argparse


def main(args):
    filepath = "test_audio/rec1/"
    print(os.chdir(sys.path[0]))
    x,sr = load_wav(os.path.abspath(filepath))
    sr = 16000
    r = 0.032
    c = 343

    frameLen = 256
    hop = frameLen/2
    overlap = frameLen - hop
    nfft = 256
    c = 340
    r = 0.032
    fs = sr

    start = time.process_time()

    MicArrayObj = MicArray(arrayType='circular', r=0.032, M=4)
    angle = np.array([197, 0]) / 180 * np.pi

    GSC_1 = GSC(MicArrayObj,frameLen,hop,nfft,c,r,fs)
    yout = GSC_1.process(x,angle,method=3)

    end = time.process_time()
    print(end-start)

    # listen processed result
    if args.listen:
        sd.default.channels = 1
        sd.play(yout['data'],fs)
        sd.wait()

    # view beampattern
    # mesh(yout['beampattern'])
    # pmesh(yout['beampattern'])

    # save audio
    if args.save:
        wavfile.write('output/output_fixedbeamformer.wav',16000,yout['data'])

    visual(x[0,:],yout['data'])
    plt.title('spectrum')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Feature Extractor")
    parser.add_argument("-l","--listen", action='store_true', help="set to listen output") # if set true
    parser.add_argument("-s","--save", action='store_true', help="set to save output") # if set true

    args = parser.parse_args()
    main(args)


