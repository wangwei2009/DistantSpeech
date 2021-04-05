"""
Test beamformer

"""

import numpy as np

import time
import os
import sys
import argparse

from DistantSpeech.beamformer.MicArray import MicArray
from DistantSpeech.beamformer.adaptivebeamformer import adaptivebeamfomer
from DistantSpeech.beamformer.utils import mesh,pmesh,load_wav,load_pcm

import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
from scipy.io import wavfile


def main(args):
    filepath = "./test_audio/rec1/"
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

    MicArray_obj = MicArray(arrayType='circular', r=0.032, M=4)
    angle = np.array([197, 0]) / 180 * np.pi

    adaptivebeamfomer_obj = adaptivebeamfomer(MicArray_obj,frameLen,hop,nfft,c,r,fs)
    yout = adaptivebeamfomer_obj.process(x,angle, method=2)

    end = time.process_time()
    print(end-start)

    # listen processed result
    sd.default.channels = 1
    sd.play(yout['data'],fs)
    sd.wait()

    # view beampattern
    # mesh(yout['beampattern'])
    # pmesh(yout['beampattern'])

    # save audio
    if(args.save):
        wavfile.write('example/output/output_fixedbeamformer.wav',16000,yout['data'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Feature Extractor")
    parser.add_argument("-l","--listen", action='store_true', help="set to listen output") # if set true
    parser.add_argument("-s","--save", action='store_true', help="set to save output") # if set true

    args = parser.parse_args()
    main(args)


