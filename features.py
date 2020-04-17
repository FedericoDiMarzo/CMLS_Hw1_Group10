import numpy as np

cep_start = 3
cep_end = 14


def feature1(mfcc, windowed_audio, fs):
    """
    Example of feature using both mfcc and windowed_audio
    """
    return 0


def zcr_mean(mfcc, windowed_audio, fs):
    """
    Mean of the zero crossing rate for all the frames
    """
    N = windowed_audio.shape[0]
    zcr = np.abs(np.diff(windowed_audio, axis=0))
    zcr = np.sum(zcr, axis=0)
    zcr *= 1 / (2 * N) * fs
    return np.mean(zcr)


def spectral_centroid_mean(mfcc, windowed_audio, fs):
    """
    Mean of the spectral centroid for all the frames
    """
    N = mfcc.shape[1]
    k = np.arange(1, mfcc.shape[0] + 1)
    k = np.transpose(np.tile(k, (N, 1)))

    # print(np.any(np.isnan(np.sum(k * mfcc, axis=0))), np.any(np.isnan(np.sum(mfcc, axis=0))))
    ctr = np.sum(k * mfcc, axis=0) / np.sum(mfcc, axis=0)

    return np.mean(ctr)


def spectral_decrease_mean(mfcc, windowed_audio, fs):
    """
    Mean of the spectral decrease for all the frames
    """
    N = mfcc.shape[1]
    k = np.arange(1, mfcc.shape[0])
    k = np.transpose(np.tile(k, (N, 1)))

    sdc = np.diff(np.abs(mfcc), axis=0) / k
    sdc = np.sum(sdc, axis=0) / np.sum(mfcc, axis=0)

    return np.mean(sdc)


def mfcc_mean(cep_coef):
    """
    Closure for defining _mfcc_mean for a certain coefficient
    :param cep_coef: mfcc coefficient
    :return: closure to _mfcc_mean
    """

    def _mfcc_mean(mfcc, windowed_audio, fs):
        """
        Calculates the mean for a certain mfcc coefficient (cep_coef)
        """
        return np.mean(mfcc, axis=1)[cep_coef - cep_start]

    return _mfcc_mean


# Used to store feature name and function reference
feature_functions = {
    'feature1': feature1,
    'zcr_mean': zcr_mean,
    'spectral_centroid_mean': spectral_centroid_mean,
    'spectral_decrease_mean': spectral_decrease_mean
}

# Manually adding the features for every cep_coeffient
for c in range(cep_start, cep_end):
    feature_functions['mfcc_' + str(c) + '_mean'] = mfcc_mean(c)
