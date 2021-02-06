import numpy as np
import argparse
import os
import time
from DistantSpeech.noise_estimation.NoiseEstimationBase import NoiseEstimationBase
from DistantSpeech.noise_estimation.mcra import NoiseEstimationMCRA


class NsOmlsaMulti(NoiseEstimationBase):
    def __init__(self, nfft=256, M=4) -> None:
        super(NsOmlsaMulti, self).__init__(nfft=nfft)

        self.G_H1 = np.ones(self.half_bin)

        self.gamma = np.ones(self.half_bin)
        self.zeta_Y = np.ones(self.half_bin)              # smoothed fixed bf
        self.zeta_U = np.zeros((M, self.half_bin))        # smoothed ref
        self.MU_Y = np.ones(self.half_bin)                # estimated noise from fixed bf channel
        self.MU_U = np.zeros((M-1, self.half_bin))        # estimated noise from ref channel
        self.lambda_hat_d = np.ones(self.half_bin)

        self.gamma = np.ones(self.half_bin)               # posteriori SNR for fixed bf output
        self.LAMBDA_Y = np.ones(self.half_bin)            # posteriori SNR
        self.LAMBDA_U = np.ones(self.half_bin)            #

        self.first_frame = 1
        self.M = M
        self.noise_est_ref = []

        self.noise_est_fixed  = NoiseEstimationMCRA(nfft=self.nfft)   # initialize noise estimator
        for ch in range(M-1):
            self.noise_est_ref.append(NoiseEstimationMCRA(nfft=self.nfft))


    def estimation(self, y: np.ndarray, u: np.ndarray):

        assert len(y) == self.half_bin

        if self.first_frame == 1:
            self.first_frame = 0
        else:
            self.MU_Y = self.noise_est_fixed.estimation(y)
            for ch in range(self.M-1):
                self.MU_U[ch] = self.noise_est_ref[ch].estimation(u[ch,:])

            self.zeta_Y = self.smooth_psd(y, self.zeta_Y,)zeta_Y(:,L-1), Y_l, b, alpha_s); % Eq 21
            for ch = 2:M
                zeta_U(:,ch,L) = smoothPSD(zeta_U(:,ch,L-1), U_l(:,ch-1), b, alpha_s);
            end




        for L = 1:size(U, 1) - 1

        Y_l = Y(:, L);
        U_l = squeeze(U(L,:,:)).';

% U_l = squeeze(U(L,:,:));

% ----------------- estimate / update
noise
psd - -------------

% sig_Y = abs([Y_l;
conj(flipud(Y_l(2: end - 1)))]);
% sig_U = abs([U_l;
conj(flipud(U_l(2: end - 1,:)))]);
sig_Y = abs(Y_l);
sig_U = abs(U_l);
nsY_ps = sig_Y. ^ 2;
nsU_ps = sig_U. ^ 2;

if L == 1
    % Initialize
    variables
    at
    the
    first
    frame
    for all frequency bins k
    G_H1(:, 1) = 1;
    gamma(:, 1) = 1;
    zeta_Y(:, 1) = sig_Y. ^ 2;
    MU_Y(:, 1) = sig_Y. ^ 2;
    lambda_hat_d(:, 1) = sig_Y. ^ 2;

    parametersY = initialise_parameters([nsY_ps;
    flipud(nsY_ps(2: end - 1))], Srate, method);

    for ch = 2:M
    zeta_U(:, ch, L) = sig_U(:, ch - 1).^ 2;
    parametersU
    {ch} = initialise_parameters([nsU_ps(:, ch - 1);flipud(nsU_ps(2: end - 1, ch - 1))], Srate, method);
    end

else
    parametersY = noise_estimation([nsY_ps;
    flipud(nsY_ps(2: end - 1))], method, parametersY);
    MU_Y(:, L) = parametersY.noise_ps(1: half_bin);
    % MU_Y(:, L) = lambda_d_Y(:, L);

    for ch = 2:M
    parametersU
    {ch} = noise_estimation([nsU_ps(:, ch - 1);flipud(nsU_ps(2: end - 1, ch - 1))], method, parametersU
    {ch});
    MU_U(:, ch, L) = parametersU
    {ch}.noise_ps(1: half_bin);
    % MU_U(:, ch, L) = lambda_d_u(:, ch, L);
    end

    zeta_Y(:, L) = smoothPSD(zeta_Y(:, L - 1), Y_l, b, alpha_s); % Eq
    21
    for ch = 2:M
    zeta_U(:, ch, L) = smoothPSD(zeta_U(:, ch, L - 1), U_l(:, ch - 1), b, alpha_s);
    end

    % ---------------------------------------------------------

    LAMBDA_Y(:, L) = zeta_Y(:, L)./ MU_Y(:, L); % Eq
    22
    LAMBDA_U(:, L) = max(zeta_U(:, 2: end, L)./ MU_U(:, 2: M, L), [], 2); % Eq
    23

    % Eq
    6
    The
    transient
    beam - to - reference
    ratio(TBRR)
    eps = 0.01;
    Omega(:, L) = max((zeta_Y(:, L) - MU_Y(:, L)), 0)./ ...
    max(max(zeta_U(:, 2: end, L) - MU_U(:, 2: M, L), [], 2), eps * MU_Y(:, L));
    Omega(:, L) = max(Omega(:, L), 0.1);
    Omega(:, L) = min(Omega(:, L), 100);

    % Eq
    27
    posteriori
    SNR
    at
    the
    beamformer
    output
    gamma_s(:, L) = min(abs(Y_l). ^ 2. / (MU_Y(:, L) * Bmin), 100);

    [H0s, H0t, H1] = hypothesis(LAMBDA_Y(:, L), LAMBDA_U(:, L), Omega(:, L), gamma_s(:, L));


    for k = 1:half_bin

    % Eq
    29, The
    a
    priori
    signal
    absence
    probability
    pr.gamma_high = 0.1 * 10 ^ 2;
    pr.Omega_low = 0.3;
    pr.Omega_high = 3;
    if (gamma_s(k, L) < pr.gamma_low | | Omega(k, L) < pr.Omega_low)
        q_hat(k, L) = 1;
    else
        q_hat(k, L) = max(...
        max(...
            (pr.gamma_high - gamma_s(k, L)) / (pr.gamma_high - pr.gamma_low), ...
            ((pr.Omega_high - Omega(k, L)) / (pr.Omega_high - pr.Omega_low))...
            ), 0);
        end
        q_hat(k, L) = q_hat(k, L);
    % q_hat(k, L) = qhat_Y(k, L);

    % posteriori
    SNR
    gamma(k, L) = abs(Y(k, L)) ^ 2 / max(lambda_hat_d(k, L), 1e-10);

    % Eq
    30, priori
    SNR
    xi_hat(k, L) = alpha * G_H1(k, L - 1). ^ 2 * gamma(k, L - 1) + (1 - alpha) * max(gamma(k, L) - 1, 0);
    % xi_hat(k, L) = max(xi_hat(k, L), xi_min);

    %
    nu(k, L) = gamma(k, L) * xi_hat(k, L) / (1 + xi_hat(k, L));

    % Eq
    31, the
    spectral
    gain
    function
    of
    the
    LSA
    estimator
    when
    the
    % signal is surely
    present
    G_H1(k, L) = xi_hat(k, L) / (1 + xi_hat(k, L)) * exp(0.5 * expint(nu(k)));

    % Eq
    28, the
    signal
    presence
    probability
    p(k, L) = 1 / (1 + q_hat(k, L) / (1 - q_hat(k, L)) * (1 + xi_hat(k, L)) * exp(-1 * nu(k, L)));
    % p(k, L) = phat_Y(k, L);

    % Eq
    33, time - varying
    frequencydependent
    smoothing
    factor
    alpha_widetilde_d(k, L) = alpha_d + (1 - alpha_d) * p(k, L);

    % Eq
    32, An
    estimate
    for noise PSD is
        % obtained
        by
        recursively
        averaging
        past
        spectral
        power
        values
    % of
    the
    noisy
    measurement
    lambda_hat_d(k, L + 1) = alpha_widetilde_d(k, L) * lambda_hat_d(k, L) + ...
    beta * (1 - alpha_widetilde_d(k, L)) * abs(Y_l(k)) ^ 2;
    % lambda_hat_d(k, L) = lambda_d_Y(k, L);

    % Eq
    35, OM
    LSA
    gain
    function
    G(k, L) = real(G_H1(k, L) ^ p(k, L) * Gmin ^ (1 - p(k, L)));
    G(k, L) = min(G(k, L), 1);

    end
    % wiener
    filter
    % alpha_ns = 2;
    % beta_specsub = 1;
    % mu = 1;
    % Gmin_specsub = 0.15;
    % G(:, L) = spectral_subtraction(p(:, L), alpha_ns, beta_specsub, mu, Gmin_specsub);

    end
    end


        for k in range(self.half_bin - 1):
            if self.frm_cnt == 0:
                self.Smin[k] = Y[k]
                self.Stmp[k] = Y[k]
                self.lambda_d[k] = Y[k]
                self.p[k] = 1.0
            else:
                Sf = Y[k - 1] * self.b[0] + Y[k] * self.b[1] + Y[k + 1] * self.b[2]  # eq 6,frequency smoothing
                self.S[k] = self.alpha_s * self.S[k] + (1 - self.alpha_s) * Sf  # eq 7,time smoothing

                self.Smin[k] = np.minimum(self.Smin[k], self.S[k])  # eq 8/9 minimal-tracking
                self.Stmp[k] = np.minimum(self.Stmp[k], self.S[k])

                if self.ell % self.L == 0:
                    self.Smin[k] = np.minimum(self.Stmp[k], self.S[k])  # eq 10/11
                    self.Stmp[k] = self.S[k]

                    self.ell = 0  # loop count

                Sr = self.S[k] / (self.Smin[k] + 1e-6);

                if Sr > self.delta_s:
                    I = 1
                else:
                    I = 0

                self.p[k] = self.alpha_p * self.p[k] + (
                            1 - self.alpha_p) * I;  # eq 14,updata speech presence probability
                self.p[k] = max(min(self.p[k], 1.0), 0.0)

        self.frm_cnt = self.frm_cnt + 1
        self.lambda_d[self.half_bin - 1] = 1e-8
        self.ell = self.ell + 1
        self.update_noise_psd(Y)


def main(args):
    from DistantSpeech.transform.transform import Transform
    from DistantSpeech.beamformer.utils import mesh, pmesh, load_wav, load_pcm, visual
    from matplotlib import pyplot as plt
    import librosa

    filepath = "example/test_audio/rec1/"
    x, sr = load_wav(os.path.abspath(filepath))
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

    print(x.shape)

    transform = Transform(n_fft=320, hop_length=160)

    D = transform.stft(x[0, :])
    Y, _ = transform.magphase(D, 2)
    print(Y.shape)
    pmesh(librosa.power_to_db(Y))
    plt.savefig('pmesh.png')

    mcra = NsOmlsaMulti(nfft=320)
    noise_psd = np.zeros(Y.shape)
    p = np.zeros(Y.shape)
    for n in range(Y.shape[1]):
        mcra.estimation(Y[:, n])
        noise_psd[:, n] = mcra.lambda_d
        p[:, n] = mcra.p

    pmesh(librosa.power_to_db(noise_psd))
    plt.savefig('noise_psd.png')

    pmesh(p)
    plt.savefig('p.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Feature Extractor")
    parser.add_argument("-l", "--listen", action='store_true', help="set to listen output")  # if set true
    parser.add_argument("-s", "--save", action='store_true', help="set to save output")  # if set true

    args = parser.parse_args()
    main(args)
