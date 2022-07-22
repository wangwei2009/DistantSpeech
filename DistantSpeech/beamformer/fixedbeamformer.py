from scipy.signal import windows
from scipy import signal
import numpy as np
from DistantSpeech.beamformer.MicArray import MicArray, compute_tau
from DistantSpeech.beamformer.beamformer import beamformer
from DistantSpeech.beamformer.gen_noise_msc import gen_noise_msc
from DistantSpeech.beamformer.beamformer import compute_mvdr_weight
import warnings


class FixedBeamformer(beamformer):
    def __init__(self, MicArray, frameLen=256, hop=None, nfft=None, c=343, fs=16000, r=0.032):

        beamformer.__init__(self, MicArray, frame_len=frameLen, hop=hop, nfft=nfft, c=c, fs=fs)
        self.angle = np.array([197, 0]) / 180 * np.pi

        self.angle = (197, 0)

        self.AlgorithmList = ["src", "DS", "MVDR"]
        self.AlgorithmIndex = 0

        self.W = self.compute_weights(look_angle=self.angle)

    def compute_weights(
        self,
        look_angle=[90, 0],
        weightType="SD",
        diag_value=1e-3,
    ):
        """compute beamformer weights

        Parameters
        ----------
        look_angle : list or tuple, optional
            look angle in degree, by default [90, 0]
        weightType : str, optional
            fixedbeamformer type in ['DS', 'SD'], by default "DS"
        diag_value : float, optional
            diag value, by default 1e-6

        Returns
        -------
        w: complex weights, [bins, M]
            fixedbeamformer weights
        """

        look_angle_rad = np.array(look_angle) / 180 * np.pi

        a0 = self.compute_steering_vector_from_doa(look_angle=look_angle)

        if weightType == 'DS':
            W = a0 / self.M

        if weightType == 'SD':
            Fvv = gen_noise_msc(mic=self.MicArray)
            diag = np.eye(self.M) * diag_value
            diag_bin = np.broadcast_to(diag, (self.half_bin, self.M, self.M))
            W = compute_mvdr_weight(a0, np.linalg.inv(Fvv + diag_bin))

        return W

    def process_freframe(self, X_n):
        """processe single frame in subband
           override this function if you have different process pipline

        Parameters
        ----------
        X_n : np.array, complex array, [bins, C]
            multichannel frequency frame

        Returns
        -------
        Yf : np.array, [bins,]
            processed frame
        """

        # apply weights
        Yf = np.einsum('ij, ij->i', self.W.conj(), X_n)

        return Yf

    def process(self, x, angle=(0, 0)):
        """process core function for fixedbeamformer

        Parameters
        ----------
        x : np.array, [samples, channel]
            time-domain multichannel input chunk signal

        Returns
        -------
        output : np.array, [samples, ]
            enhanced time-domain single channel signal
        """

        assert x.shape[1] >= 2

        angle = list(angle)

        # re-calculate weights
        if angle != self.angle:
            # look_angle_rad = np.array(angle) / 180 * np.pi
            self.W = self.compute_weights(look_angle=angle)

        D = self.transform.stft(x)

        half_bin, frameNum, channel = D.shape

        # enhanced single channel spectrum
        Yf = np.zeros((half_bin, frameNum, 1), dtype=complex)

        # frame online processing block
        for n in range(frameNum):
            X_n = D[:, n, :]

            Yf[:, n, 0] = self.process_freframe(X_n)

        output = self.transform.istft(Yf)

        assert output.shape[0] == frameNum * self.hop, 'output:{}, x:{}'.format(output.shape, x.shape)

        return output.squeeze()


if __name__ == "__main__":
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
    M = 4

    MicArrayObj = MicArray(arrayType="circular", r=0.032, M=M)
    angle = [197, 0]

    fixedbeamformerObj = FixedBeamformer(MicArrayObj, frameLen, hop, nfft, c, fs)

    W = fixedbeamformerObj.compute_weights(angle, weightType='DS')
    W1 = fixedbeamformerObj.compute_weights(angle, weightType='SD')

    x = np.random.rand(16000 * 5, M)

    # """
    # fixed beamformer precesing function
    # method:
    # 'DS':   delay-and-sum beamformer
    # 'MVDR': MVDR beamformer under isotropic noise field
    #
    # """
    yout = fixedbeamformerObj.process(x, angle)

    beampattern = fixedbeamformerObj.compute_beampattern(MicArrayObj, weights=W.T)
    beampattern1 = fixedbeamformerObj.compute_beampattern(MicArrayObj, weights=W1.T)
    from DistantSpeech.beamformer.utils import pmesh, mesh

    mesh(beampattern[:, 2:-2].T)
