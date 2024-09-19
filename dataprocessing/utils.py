import numpy as np
import math

def arrayFactorGivenAngleForULA(numAntennaElements, theta, normalizedAntDistance=0.5, angleWithArrayNormal=0):

    indices = np.arange(numAntennaElements)
    if (angleWithArrayNormal == 1):
        arrayFactor = np.exp(-1j * 2 * np.pi * normalizedAntDistance * indices * np.sin(theta))
    else:  # default
        arrayFactor = np.exp(-1j * 2 * np.pi * normalizedAntDistance * indices * np.cos(theta))
    return arrayFactor / np.sqrt(numAntennaElements)  # normalize to have unitary norm

def getNarrowBandULAMIMOChannel(azimuths_tx, azimuths_rx, p_gainsdB, number_Tx_antennas, number_Rx_antennas,
                                normalizedAntDistance=0.5, angleWithArrayNormal=0, pathPhases=None):
    
    azimuths_tx = np.deg2rad(azimuths_tx)
    azimuths_rx = np.deg2rad(azimuths_rx)
    # nt = number_Rx_antennas * number_Tx_antennas #np.power(antenna_number, 2)
    m = np.shape(azimuths_tx)[0]  # number of rays
    H = np.matrix(np.zeros((number_Rx_antennas, number_Tx_antennas)))

    gain_dB = p_gainsdB
    path_gain = np.power(10, gain_dB / 10)
    path_gain = np.sqrt(path_gain)

    #generate uniformly distributed random phase in radians
    if pathPhases is None:
        pathPhases = 2*np.pi * np.random.rand(len(path_gain))
    else:
        #convert from degrees to radians
        pathPhases = np.deg2rad(pathPhases)

    #include phase information, converting gains in complex-values
    path_complexGains = path_gain * np.exp(1j * pathPhases)

    # recall that in the narrowband case, the time-domain H is the same as the
    # frequency-domain H
    for i in range(m):
        at = np.matrix(arrayFactorGivenAngleForULA(number_Tx_antennas, azimuths_tx[i], normalizedAntDistance,
                                                   angleWithArrayNormal))
        ar = np.matrix(arrayFactorGivenAngleForULA(number_Rx_antennas, azimuths_rx[i], normalizedAntDistance,
                                                   angleWithArrayNormal))
        H = H + path_complexGains[i] * ar.conj().T * at  # outer product of ar Hermitian and at
        #H = H + path_complexGains[i] * ar
    factor = (np.linalg.norm(path_complexGains) / np.sum(path_complexGains)) * np.sqrt(
        number_Rx_antennas * number_Tx_antennas)  # scale channel matrix
    H *= factor  # normalize for compatibility with Anum's Matlab code

    return H

def watts_to_dbm(power_watts):
    dbm = 10 * math.log10(power_watts * 1000)
    return dbm

def dbm_to_watts(dbm):
    return 10 ** (dbm / 10)

def degrees_to_radians(degrees):
    return np.radians(degrees)

def dft_codebook(dim):
    seq = np.matrix(np.arange(dim))
    mat = seq.conj().T * seq
    w = np.exp(-1j * 2 * np.pi * mat / dim)
    return w

def getDFTOperatedChannel(H, number_Tx_antennas, number_Rx_antennas):
    wt = dft_codebook(number_Tx_antennas)
    wr = dft_codebook(number_Rx_antennas)
    dictionaryOperatedChannel = wr.conj().T * H * wt
    # dictionaryOperatedChannel2 = wr.T * H * wt.conj()
    return dictionaryOperatedChannel  # return equivalent channel after precoding and combining
