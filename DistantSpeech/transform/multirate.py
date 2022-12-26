import numpy as np


def fractional_delay_filter_bank(delays):
    """
    Creates a fractional delay filter bank of windowed sinc filters

    Parameters
    ----------
    delays: 1d narray
        The delays corresponding to each filter in fractional samples

    Returns
    -------
    numpy array
        [filter_len, chs]
        An ndarray where the ith col contains the fractional delay filter
        corresponding to the ith delay. The number of columns of the matrix
        is proportional to the maximum delay.
    """

    delays = np.array(delays)

    # subtract the minimum delay, so that all delays are positive
    delays -= delays.min()

    # constants and lengths
    N = delays.shape[0]
    L = 81
    filter_length = L + int(np.ceil(delays).max())

    # allocate a flat array for the filter bank that we'll reshape at the end
    bank_flat = np.zeros(N * filter_length)

    # separate delays in integer and fractional parts
    di = np.floor(delays).astype(np.int64)
    df = delays - di

    # broadcasting tricks to compute at once all the locations
    # and sinc times that must be computed
    T = np.arange(L)
    indices = T[None, :] + (di[:, None] + filter_length * np.arange(N)[:, None])
    sinc_times = T - df[:, None] - (L - 1) / 2

    # we'll need to window also all the sincs at once
    windows = np.tile(np.hanning(L), N)

    # compute all sinc with one call
    bank_flat[indices.ravel()] = windows * np.sinc(sinc_times.ravel())

    return np.reshape(bank_flat, (N, -1)).T


def frac_delay(delta, N, w_max=0.9, C=4):
    """
    Compute optimal fractionnal delay filter according to

    Design of Fractional Delay Filters Using Convex Optimization
    William Putnam and Julius Smith

    Parameters
    ----------
    delta:
        delay of filter in (fractionnal) samples
    N:
        number of taps
    w_max:
        Bandwidth of the filter (in fraction of pi) (default 0.9)
    C:
        sets the number of constraints to C*N (default 4)
    """

    # constraints
    N_C = int(C * N)
    w = np.linspace(0, w_max * np.pi, N_C)[:, np.newaxis]

    n = np.arange(N)

    try:
        from cvxopt import solvers, matrix
    except:
        raise ValueError("To use the frac_delay function, the cvxopt module is necessary.")

    f = np.concatenate((np.zeros(N), np.ones(1)))

    A = []
    b = []
    for i in range(N_C):
        Anp = np.concatenate(([np.cos(w[i] * n), -np.sin(w[i] * n)], [[0], [0]]), axis=1)
        Anp = np.concatenate(([-f], Anp), axis=0)
        A.append(matrix(Anp))
        b.append(matrix(np.concatenate(([0], np.cos(w[i] * delta), -np.sin(w[i] * delta)))))

    solvers.options["show_progress"] = False
    sol = solvers.socp(matrix(f), Gq=A, hq=b)

    h = np.array(sol["x"])[:-1, 0]

    """
    import matplotlib.pyplot as plt
    w = np.linspace(0, np.pi, 2*N_C)
    F = np.exp(-1j*w[:,np.newaxis]*n)
    Hd = np.exp(-1j*delta*w)
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(np.abs(np.dot(F,h) - Hd))
    plt.subplot(3,1,2)
    plt.plot(np.diff(np.angle(np.dot(F,h))))
    plt.subplot(3,1,3)
    plt.plot(h)
    """

    return h


def low_pass(numtaps, B, epsilon=0.1):

    bands = [0, (1 - epsilon) * B, B, 0.5]
    desired = [1, 0]

    from scipy.signal import remez

    h = remez(numtaps, bands, desired, grid_density=32)

    """
    import matplotlib.pyplot as plt
    w = np.linspace(0, np.pi, 8*numtaps)
    F = np.exp(-1j*w[:,np.newaxis]*np.arange(numtaps))
    Hd = np.exp(-1j*w)
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(np.abs(np.dot(F,h)))
    plt.subplot(3,1,2)
    plt.plot(np.angle(np.dot(F,h)))
    plt.subplot(3,1,3)
    plt.plot(h)
    """

    return h


def resample(x, p, q):

    import fractions

    gcd = fractions.gcd(p, q)
    p /= gcd
    q /= gcd

    m = np.maximum(p, q)
    h = low_pass(10 * m + 1, 1.0 / (2.0 * m))

    x_up = np.kron(x, np.concatenate(([1], np.zeros(p - 1))))

    from scipy.signal import fftconvolve

    x_rs = fftconvolve(x_up, h)

    x_ds = x_rs[h.shape[0] / 2 + 1 :: q]
    x_ds = x_ds[: np.floor(x.shape[0] * p / q)]

    """
    import matplotlib.pyplot as plt
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(x)
    plt.subplot(3,1,2)
    plt.plot(x_rs)
    plt.subplot(3,1,3)
    plt.plot(x_ds)
    """

    return x_ds


if __name__ == "__main__":

    delta = 2.5
    N = 128
    w_max = 0.9
    C = 4

    # constraints
    N_C = int(C * N)
    w = np.linspace(0, w_max * np.pi, N_C)[:, np.newaxis]

    n = np.arange(N)

    try:
        from cvxopt import solvers, matrix
    except:
        raise ValueError("To use the frac_delay function, the cvxopt module is necessary.")

    f = np.concatenate((np.zeros(N), np.ones(1)))

    A = []
    b = []
    for i in range(N_C):
        Anp = np.concatenate(([np.cos(w[i] * n), -np.sin(w[i] * n)], [[0], [0]]), axis=1)
        Anp = np.concatenate(([-f], Anp), axis=0)
        A.append(matrix(Anp))
        b.append(matrix(np.concatenate(([0], np.cos(w[i] * delta), -np.sin(w[i] * delta)))))

    solvers.options["show_progress"] = False
    sol = solvers.socp(matrix(f), Gq=A, hq=b)

    h = np.array(sol["x"])[:-1, 0]

    import matplotlib.pyplot as plt

    w = np.linspace(0, np.pi, 2 * N_C)
    F = np.exp(-1j * w[:, np.newaxis] * n)
    Hd = np.exp(-1j * delta * w)
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(np.abs(np.dot(F, h) - Hd))
    plt.subplot(3, 1, 2)
    plt.plot(np.diff(np.angle(np.dot(F, h))))
    plt.subplot(3, 1, 3)
    plt.plot(h)
