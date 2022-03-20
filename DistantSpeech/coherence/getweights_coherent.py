"""
coherence-based speech enhancement Algorithms
==============

implement some coherence-based dual-mic noise reduction algorithms

see reference below to get more details

----------


.. [1] N. Yousefian and P. C. Loizou, "A Dual-Microphone Speech Enhancement Algorithm
    Based on the Coherence Function," in IEEE Transactions on Audio, Speech, and
    Language Processing, vol. 20, no. 2, pp. 599-609, Feb. 2012.

.. [2] Yousefian, N., Loizou, P. C., & Hansen, J. H. L. (2014). A coherence-based noise
    reduction algorithm for binaural hearing aids. Speech Communication, 58, 101–110

.. [3] Ji, Y., Byun, J., Park, Y. (2017) Coherence-Based Dual-Channel Noise Reduction
    Algorithm in a Complex Noisy Environment. Proc. Interspeech 2017

.. [4] Yousefian, N., & Loizou, P. C. (2013). A Dual-Microphone Algorithm That Can Cope
     With Competing-Talker Scenarios. IEEE Transactions on Audio, Speech, and Language Processing

"""
import numpy as np
import math


def getweghts_coherent(Fvv_est: np.ndarray, Fvv_diffuse: np.ndarray, k, method=3):

    c = 340
    fs = 16000
    nfft = 256
    r = 0.032 * 2
    GAIN_FLOOR = 0.1

    if method == 0:
        G = 1  # don't change anything

    if method == 1:
        # refer to [1]
        alpha_low = 16
        alpha_hi = 2
        beta_low = -0.1
        beta_hi = -0.3
        mu = 0.5
        gamma = 0.6

        if k <= 16:
            G1 = 1 - np.power(np.abs(np.real(Fvv_est)), alpha_low)  # eq.15
            Q = beta_low

        else:
            G1 = 1 - np.power(np.abs(np.real(Fvv_est)), alpha_hi)  # eq.15
            Q = beta_hi
        if np.imag(Fvv_est) < Q:
            G2 = mu
        else:
            G2 = 1
        G = G1 * G2
        # print("G = \n",G)
    if method == 2:
        # refer to [2]
        G = (1 - (np.real(Fvv_est)) ** 2 - np.imag(Fvv_est) ** 2) / (2 * (1 - np.real(Fvv_est)))
    if method == 3:
        # refer to [3]
        Fvv_UPPER = 0.98
        GAIN_FLOOR = 0.2
        Fy_real = np.real(Fvv_est)
        Fy_imag = np.imag(Fvv_est)
        Fn = Fvv_diffuse

        if Fy_real > Fvv_UPPER:
            Fy_real = Fvv_UPPER
        abs_Fvv_est = np.sqrt(Fy_real**2 + Fy_imag**2)
        if abs_Fvv_est > Fvv_UPPER:
            abs_Fvv_est = Fvv_UPPER
        if Fn > Fvv_UPPER:
            Fn = Fvv_UPPER

        DDR = (np.abs(Fn) ** 2 - abs_Fvv_est**2) / (abs_Fvv_est**2 - 1)  # eq.10
        K = DDR / (DDR + 1)
        theta_s = 90 * np.pi / 180  # target, endfire
        theta_i = 0 * np.pi / 180  # interference, broadside
        constant = 2 * np.pi * k * fs * r / ((nfft * c))
        sin_alpha = np.sin(constant * np.sin(theta_s))
        cos_alpha = np.cos(constant * np.sin(theta_s))

        A = sin_alpha * K - Fy_imag
        B = cos_alpha * K - Fy_real + Fn * (1 - K)  # eq.20
        C = (Fy_real - Fn * (1 - K)) * sin_alpha - Fy_imag * cos_alpha

        T = K - cos_alpha * (Fy_real - Fn * (1 - K)) - Fy_imag * sin_alpha
        sin_beta = (-1 * B * C - A * T) / (A**2 + B**2)  # eq.21
        G = (Fy_imag - sin_beta * K) / (sin_alpha - sin_beta)  # eq.12

    if method == 4:
        # refer to [3]
        Fvv_UPPER = 0.98
        GAIN_FLOOR = 0.2
        Fy_real = np.real(Fvv_est)
        Fy_imag = np.imag(Fvv_est)
        Fn = Fvv_diffuse

        if Fy_real > Fvv_UPPER:
            Fy_real = Fvv_UPPER
        abs_Fvv_est = np.sqrt(Fy_real**2 + Fy_imag**2)
        if abs_Fvv_est > Fvv_UPPER:
            abs_Fvv_est = Fvv_UPPER
        if Fn > Fvv_UPPER:
            Fn = Fvv_UPPER

        DDR = (np.abs(Fn) ** 2 - abs_Fvv_est**2) / (abs_Fvv_est**2 - 1)  # eq.10
        K = DDR / (DDR + 1)
        theta_s = 90 * np.pi / 180  # target, endfire
        theta_i = 0 * np.pi / 180  # interference, broadside
        constant = 2 * np.pi * k * fs * r / ((nfft * c))
        sin_alpha = np.sin(constant * np.sin(theta_s))
        cos_alpha = np.cos(constant * np.sin(theta_s))

        A = sin_alpha * K - Fy_imag
        B = cos_alpha * K - Fy_real + Fn * (1 - K)  # eq.20
        C = (Fy_real - Fn * (1 - K)) * sin_alpha - Fy_imag * cos_alpha

        T = K - cos_alpha * (Fy_real - Fn * (1 - K)) - Fy_imag * sin_alpha
        sin_beta = (-1 * B * C - A * T) / (A**2 + B**2)  # eq.21
        cos_beta = (A * C - B * T) / (A**2 + B**2)  # eq.22

        A_ = cos_alpha - cos_beta
        B_ = cos_beta + Fn * (1 - K)  # eq.16
        C_ = sin_alpha - sin_beta
        D_ = sin_beta * K

        if np.abs(Fy_imag - sin_alpha) < np.abs(Fy_imag - sin_beta):  # eq.18
            gamma_ = 1
        else:
            gamma_ = -1

        T_ = (abs_Fvv_est**2) * (A_**2 + C_**2) - (A_ * D_ - B_ * C_) ** 2

        G = -1 * (A_ * B_ + C_ * D_) + gamma_ * np.sqrt(T_) / (A_**2 + C_**2)  # eq.17

    if method == 5:
        # refer to [4]
        Fy_real = np.real(Fvv_est)
        Fy_imag = np.imag(Fvv_est)
        Fn = Fvv_diffuse
        d = r

        theta = 90 * np.pi / 180  # interference broadside
        ata = 0 * np.pi / 180  # target endfire
        omega = 2 * np.pi * (k - 1) / nfft
        tao = fs * d / c
        omega_ = omega * tao
        beta = omega_ * np.cos(ata)
        alpha = omega_ * np.cos(theta)
        constant = 2 * np.pi * k * fs * d / ((nfft * c))

        A = Fy_imag - np.sin(omega_)
        B = np.cos(omega_) - Fy_real
        C = Fy_real * np.sin(omega_) - Fy_imag * np.cos(omega_)
        T = 1 - Fy_real * np.cos(omega_) - Fy_imag * np.sin(omega_)

        sin_alpha = (-1 * B * C + A * T) / (A**2 + B**2)  # eq.14
        SNR = (sin_alpha - Fy_imag) / (Fy_imag - np.sin(beta))  # eq.10
        G = np.sqrt(SNR / (SNR + 1))

    if G < GAIN_FLOOR:
        G = GAIN_FLOOR  # *sign(G(k));
    if G >= 1:
        G = 1
    if math.isnan(G):
        G = GAIN_FLOOR

    return G
