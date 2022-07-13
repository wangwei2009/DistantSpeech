import numpy as np
from DistantSpeech.beamformer.utils import DelaySamples


def test_delaysamples():
    for ch in range(2):
        for data_len in [1, 10, 100]:
            for delay in [0, 1, 5, 50, 150]:
                delay_obj = DelaySamples(data_len, delay, channel=ch)

                x = np.random.rand(1000, ch)
                y = np.random.rand(1000, ch)

                for n in range(x.shape[0] // data_len):
                    xn = x[n * data_len : n * data_len + data_len, :]
                    y[n * data_len : n * data_len + data_len, :] = delay_obj.delay(xn)

                if delay == 0:
                    assert np.sum(np.abs(y - x)) < 1e-5, "delay error when data_len={}, delay={}".format(
                        data_len, delay
                    )
                else:
                    assert np.sum(np.abs(y[delay:] - x[:-delay])) < 1e-5
