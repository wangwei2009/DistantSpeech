import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
    ax.plot_surface(X, Y, array2D)
    plt.show()

def pmesh(array2D):
    """
    plot color 2D array

    """
    size = array2D.shape
    Y = np.arange(0, size[0], 1)
    X = np.arange(0, size[1], 1)
    X, Y = np.meshgrid(X, Y)

    plt.pcolormesh(X, Y, array2D)
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
    for names in filename:
        x1, sr = librosa.load(filepath + names, sr=None)
        wavdata_list.append(x1)
    L = len(wavdata_list[0])
    M = len(filename)
    wavdata = np.zeros([M,L])
    for i in range(0,M):
        wavdata[i,:] = wavdata_list[i]
    return wavdata,sr     # return M*L ndarray


def load_pcm(filepath):
    """
    load all wav file in a directory
    :return M*L ndarray

    """
    import librosa
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