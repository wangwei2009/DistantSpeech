# ## CCAF Bound Calculator
# # This function calculates the constant upper and lower matrices phi
# # and psi for the Coefficient-Constained Adaptive Filters (CCAF's) for
# # the Blocking Matrix (BM) of a Robust Generalized Sidelobe Canceller
# # (GSC) Beamformer.
# #
# # *Syntax*
# #
# # |[phi, psi] = ccafbounds(m, fs, c, p, order)|
# #
# # *Inputs*
# #
# # * |m| - 3xM matrix of microphone positions, each column a coordinate
# #     in R^3 (meters)
# # * |fs| - Audio sample rate (Hertz)
# # * |c| - Speed of sound (meters/sec)
# # * |p| - Estimated propagation time across the array in samples
# # * |order| - Order of the adaptive filters
# #
# # *Outputs*
# #
# # * |phi| - Matrix of upper bounds, where each column is a vector of
# #      bounds for a the adaptive filter of a single track in the BM.
# # * |psi| - Matrix of lower bounds with the same structure as psi.
# #
# # *Notes*
# #
# # This code is based on equations derived for a linear microphone
# # array (see references) where the effect of the coefficient bounds
# # would be to expand the main lobe of the beamformer depending on the
# # sine of a parameter delta-theta, the supposed error in steering
# # angle of the beamformer.  However, here we consider a beamformer of
# # arbitary geometry and thus the parameter "delta-theta" no longer
# # makes sense.  Our current adaptation of the algorithm is
# # 
# # # Consider the center of the array as the centroid of array,
# #   calculated as the arithmetic mean of the mic coordinates.
# # # Hard-code a value for sin(delta-theta), which we know must be
# #   bounded [-1 1].  At present we've selected .05
# # 
# # Note that these adaptations will still recreate the original
# # performance of the algorithm for a linear array, where our selection
# # of .05 for sin(delta-theta) corresponds to a steering angle error of
# # about +/- 3 degrees.
# #
# # *References*
# #
# # * Hoshuyama, Osamu, Akihiko Sugiyama, and Akihiro Hirano. "A Rboust
# # Adaptive Beamformer with a Blocking Matrix Using
# # Coefficient-Constrained Adaptive Filters." IEICE Trans. Fundamentals
# # E82-A (1999): 640-47.
# #
# # Written by Phil Townsend (jptown0@engr.uky.edu) 8-12-08

# ## Function Declaration
import numpy as np
def ccafbounds(m, fs=16000, c=343, p=1, order=1):
    """CCAF Bound Calculator
        This function calculates the constant upper and lower matrices phi
        and psi for the Coefficient-Constained Adaptive Filters (CCAF's) for
        the Blocking Matrix (BM) of a Robust Generalized Sidelobe Canceller
        (GSC) Beamformer.

        # Calculate Bounds
        In the original paper, the bound vectors for each adaptive filter are
        calculated as
        
        $$\phi_{m,n} = \frac{1}{\pi\ max(.1, \ 
        (n\ ^\_ \ P)\ ^\_ \ T_m, \ ^\_ (n\ ^\_ \ P)\ ^\_ \ T_m)} \quad
        \psi = \ ^\_ \phi\ \forall\ m, n$$
        
        $$T_m = \frac{b_mf_s}{c}\sin\Delta\theta $$

        # where bm is the distance of the mth microphone to the center of the
        # array.  Remember that we must fudge for sin(delta-theta) in R^3.

        # # *Notes*
        # #
        # # This code is based on equations derived for a linear microphone
        # # array (see references) where the effect of the coefficient bounds
        # # would be to expand the main lobe of the beamformer depending on the
        # # sine of a parameter delta-theta, the supposed error in steering
        # # angle of the beamformer.  However, here we consider a beamformer of
        # # arbitary geometry and thus the parameter "delta-theta" no longer
        # # makes sense.  Our current adaptation of the algorithm is
        # # 
        # # # Consider the center of the array as the centroid of array,
        # #   calculated as the arithmetic mean of the mic coordinates.
        # # # Hard-code a value for sin(delta-theta), which we know must be
        # #   bounded [-1 1].  At present we've selected .05
        # # 
        # # Note that these adaptations will still recreate the original
        # # performance of the algorithm for a linear array, where our selection
        # # of .05 for sin(delta-theta) corresponds to a steering angle error of
        # # about +/- 3 degrees.
        # #
        # # *References*
        # #
        # # * Hoshuyama, Osamu, Akihiko Sugiyama, and Akihiro Hirano. "A Rboust
        # # Adaptive Beamformer with a Blocking Matrix Using
        # # Coefficient-Constrained Adaptive Filters." IEICE Trans. Fundamentals
        # # E82-A (1999): 640-47.
        # #
        # # Written by Phil Townsend (jptown0@engr.uky.edu) 8-12-08

    Parameters
    ----------
    m : np.array
        matrix of microphone positions, each column a coordinate in R^3 (meters), [3, M]
    fs : _type_
        Audio sample rate (Hertz)
    c : _type_
        Speed of sound (meters/sec)
    p : _type_
        Estimated propagation time across the array in samples
    order : _type_
        Order of the adaptive filters

    Returns
    -------
    phi : np.array
        Matrix of upper bounds, where each column is a vector of 
        bounds for a the adaptive filter of a single track in the BM.(order, M)
    psi :
         Matrix of lower bounds with the same structure as psi.
    """

    # # number of microphones in the array
    M = m.shape[1]
    phi = np.zeros((order, M));  # initialze upper bound matrix for iteration

    # # iterate over all microphones
    for mIter in range(M):   
        sinDt = .34;  # 20 degree, kludge for 3-D (see Notes section above)
        arrayCentroid = np.mean(m,axis=1);  # use centroid as "center" of array
        bm = np.sqrt(np.sum((m[:,mIter]-arrayCentroid)**2));  # Get mic distance
                                                        # from centroid
        Tm = bm*fs*sinDt/c;  # Hoshuyama equation
        for nIter in range(1, order+1):  # Set bound for each tap of this adaptive filter
            phi[nIter-1, mIter] = 1/(np.pi*max([.1, (nIter-p)-Tm, -(nIter-p)-Tm]));  # directly from Hoshuyama paper
    psi = -phi;  # psi is simply the opposite of phi


    return phi, psi
