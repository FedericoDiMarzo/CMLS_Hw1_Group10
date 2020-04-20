import numpy as np
import common


def feature1(**kwargs):
    import random
    """
    Example of feature using both mfcc and windowed_audio
    """
    return random.randint(0, 10)


def zcr_mean(**kwargs):
    """
    Mean of the zero crossing rate for all the frames
    """
    windowed_audio = kwargs['windowed_audio']
    fs = kwargs['fs']

    N = windowed_audio.shape[0]
    zcr = np.abs(np.diff(windowed_audio, axis=0))
    zcr = np.sum(zcr, axis=0)
    zcr *= 1 / (2 * N) * fs
    return np.mean(zcr)


def spectral_centroid_mean(**kwargs):
    """
    Mean of the spectral centroid for all the frames
    """
    stft = kwargs['stft']

    N = stft.shape[1]
    k = np.arange(1, stft.shape[0] + 1)
    k = np.transpose(np.tile(k, (N, 1)))
    ctr = np.sum(k * stft, axis=0) / np.sum(stft, axis=0)
    return np.mean(ctr)

def spectral_flux(**kwargs):
    fs = kwargs['fs']
    magnitude_spectrum = kwargs['stft']
    # convert to frequency domain
    timebins, freqbins = np.shape(magnitude_spectrum)
    # when do these blocks begin (time in seconds)?
    timestamps = (np.arange(0, timebins - 1) * (timebins / float(fs)))
    sf = np.sqrt(np.sum(np.diff(np.abs(magnitude_spectrum)) ** 2, axis=1)) / freqbins
    return np.mean(sf[1:])


def spectral_decrease_mean(**kwargs):
    """
    Mean of the spectral decrease for all the frames
    """
    stft = kwargs['stft']

    N = stft.shape[1]
    k = np.arange(1, stft.shape[0])
    k = np.transpose(np.tile(k, (N, 1)))
    sdc = np.diff(np.abs(stft), axis=0) / k
    sdc = np.sum(sdc, axis=0) / np.sum(stft, axis=0)
    return np.mean(sdc)


def mfcc_mean(cep_coef):
    """
    Closure for defining _mfcc_mean for a certain coefficient
    :param cep_coef: mfcc coefficient
    :return: closure to _mfcc_mean
    """

    def _mfcc_mean(**kwargs):
        """
        Calculates the mean for a certain mfcc coefficient (cep_coef)
        """
        mfcc = kwargs['mfcc']
        return np.mean(mfcc, axis=1)[cep_coef - common.cep_start]

    return _mfcc_mean


def mfcc_std(cep_coef):
    """
    Closure for defining _mfcc_std for a certain coefficient
    :param cep_coef: mfcc coefficient
    :return: closure to _mfcc_std
    """

    def _mfcc_std(**kwargs):
        """
        Calculates the std for a certain mfcc coefficient (cep_coef)
        """
        mfcc = kwargs['mfcc']
        return np.std(mfcc, axis=1)[cep_coef - common.cep_start]

    return _mfcc_std


# Used to store feature name and function reference
feature_functions = {
    'feature1': feature1,
    'zcr_mean': zcr_mean,
    'spectral_centroid_mean': spectral_centroid_mean,
    'spectral_decrease_mean': spectral_decrease_mean,
    'spectral_flux': spectral_flux

}

# Manually adding the features for every cep_coeffient
for c in range(common.cep_start, common.cep_end):
    feature_functions['mfcc_' + str(c) + '_mean'] = mfcc_mean(c)
    feature_functions['mfcc_' + str(c) + '_std'] = mfcc_std(c)
