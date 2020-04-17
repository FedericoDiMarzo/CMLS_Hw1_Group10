import numpy as np

def feature1(mfcc, windowed_audio, fs):
    """
    Example of feature using both mfcc and windowed_audio
    """
    return 0


def zcr_mean(mfcc, windowed_audio, fs):
    """
    Mean of the zero crossing rate of the frames
    """
    N = windowed_audio.shape[0]
    zcr = np.abs(np.diff(windowed_audio, axis=0))
    zcr = np.sum(zcr, axis=0)
    zcr *= 1 / (2 * N) * fs
    return np.mean(zcr)


# Used to store feature name and function reference
feature_functions = {
    'feature1': feature1,
    'zcr_mean': zcr_mean,
}
