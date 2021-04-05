"""
Test beamformer

"""

import numpy as np
import os
import sys

import time
import argparse

from DistantSpeech.beamformer.MicArray import MicArray
from DistantSpeech.beamformer.fixedbeamformer import fixedbeamformer
from DistantSpeech.beamformer.adaptivebeamformer import adaptivebeamfomer
from DistantSpeech.beamformer.utils import mesh,pmesh,load_wav,load_pcm, visual

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

    MicArrayObj = MicArray(arrayType='circular', r=0.032, M=4)
    angle = np.array([197, 0]) / 180 * np.pi

    fixedbeamformerObj = fixedbeamformer(MicArrayObj,frameLen,hop,nfft,c,fs)
    # """
    # fixed beamformer precesing function
    # method:
    # 'DS':   delay-and-sum beamformer
    # 'MVDR': MVDR beamformer under isotropic noise field
    #
    # """
    yout = fixedbeamformerObj.process(x,angle,method=2,retH=True,retWNG=True,retDI=True)

    end = time.process_time()
    print(end-start)

    # listen processed result
    if(args.listen):
        sd.default.channels = 1
        sd.play(yout['data'],fs)
        sd.wait()

    # view beampattern
    mesh(yout['beampattern'])
    plt.title('beampattern')
    # pmesh(yout['beampattern'])

    # save audio
    if(args.save):
        wavfile.write('output/output_fixedbeamformer.wav',16000,yout['data'])

    visual(x[0,:],yout['data'])
    plt.title('spectrum')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Feature Extractor")
    parser.add_argument("-l","--listen", action='store_true', help="set to listen output") # if set true
    parser.add_argument("-s","--save", action='store_true', help="set to save output") # if set true

    args = parser.parse_args()
    main(args)

