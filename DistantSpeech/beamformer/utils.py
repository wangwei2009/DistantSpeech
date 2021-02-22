import os

import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def spec(x):
    D = librosa.stft(x)  # STFT of y
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    return S_db


def visual(x,y=None,sr=16000):

    if y is not None:
        S_db1 = spec(x)
        S_db2 = spec(y)

        plt.figure()
        plt.subplot(2, 1, 1)
        librosa.display.specshow(S_db1, y_axis='linear', x_axis='time',sr=sr)
        plt.colorbar()

        plt.subplot(2, 1, 2)
        librosa.display.specshow(S_db2, y_axis='linear', x_axis='time',sr=sr)
        plt.colorbar()

        plt.show()
    else:
        S_db = spec(x)
        plt.figure()
        librosa.display.specshow(S_db, y_axis='linear', x_axis='time',sr=sr)
        plt.colorbar()
        plt.show()  
        
def mesh(array2D):
    """
    plot 2D array

    """
    size = array2D.shape
    Y = np.arange(0, size[0], 1)
    X = np.arange(0, size[1], 1)
    X, Y = np.meshgrid(X, Y)
    fig1 = plt.figure()
    ax = Axes3D(fig1)
    surf = ax.plot_surface(X, Y, array2D, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig1.colorbar(surf, shrink=0.5, aspect=5)
    # plt.show()

def pmesh(array2D):
    """
    plot color 2D array

    """
    size = array2D.shape
    Y = np.arange(0, size[0], 1)
    X = np.arange(0, size[1], 1)
    X, Y = np.meshgrid(X, Y)

    im = plt.pcolormesh(X, Y, array2D, shading='auto')
    plt.colorbar(im)
    plt.show()


def find_files(filepath,fileType:str):
    """
    find all wav file in a directory

    """
    import os
    filename = os.listdir(filepath)
    wavlist = []
    for names in filename:
        if names.endswith(fileType):
            wavlist.append(names)
    return wavlist


def load_wav(filepath):
    """
    load all wav file in a directory
    :return M*L ndarray

    """
    import librosa
    filename = find_files(filepath,".wav")
    wavdata_list = []
    is_channel_1 = 1
    min_len = 0
    for names in filename:
        x1, sr = librosa.load(os.path.join(filepath, names), sr=None)
        if is_channel_1:
            min_len = len(x1)
            is_channel_1 = 0
        else:
            if len(x1) < min_len:
                min_len = len(x1)
        wavdata_list.append(x1)

    M = len(filename)
    wavdata = np.zeros([M, min_len])
    for i in range(0, M):
        wavdata[i, :] = wavdata_list[i][:min_len]
    return wavdata, sr     # return M*L ndarray


def load_pcm(filepath):
    """
    load all wav file in a directory
    :return M*L ndarray

    """
    import numpy as np
    filename = find_files(filepath,".pcm")
    wavdata_list = []
    for names in filename:
        x1 = np.memmap(filepath + names, dtype='h', mode='r')/32768.0
        wavdata_list.append(x1)
    L = len(wavdata_list[0])
    M = len(filename)
    wavdata = np.zeros([M,L])
    for i in range(0,M):
        wavdata[i,:] = wavdata_list[i]
    return wavdata     # return M*L ndarray

def filter(x):
    """
    filter M*L ndarray

    """
    from scipy import signal
    h = signal.firwin(513, 0.01, pass_zero=False)
    size = x.shape
    M = size[0]
    L = size[1]
    for i in range(0,M):
        x[i,:] = signal.filtfilt(h, 1, x[i,:])
    return x