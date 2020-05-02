import numpy as np
import scipy as sp
import common
import librosa


# def feature1(**kwargs):
#     import random
#     """
#     Example of feature using both mfcc and windowed_audio
#     """
#     return random.randint(0, 10)


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
    stft = kwargs['stft']  # stft being already a power spectrum (**2 in common)
    freqbins, N = np.shape(stft)
    sf = np.sqrt(np.sum(np.diff(np.abs(stft)) ** 2, axis=1)) / freqbins  # 513 differences² calculated
    return np.mean(sf)


def spectral_rolloff(**kwargs):
    fs = kwargs['fs']
    stft = kwargs['stft']
    k = 0.85
    freqbins, N = np.shape(stft)
    spectralSum = np.sum(stft, axis=1)  # 513x1 for the whole song
    # find frequency-bin indices where the cumulative sum of all bins is higher
    # than k-percent of the sum of all bins. Lowest index = Rolloff
    sr = np.where(np.cumsum(spectralSum) >= k * sum(spectralSum))[0][0]
    # convert frequency-bin index to frequency in Hz
    sr = (sr / freqbins) * (fs / 2.0)
    return sr


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


def onset_events(filter_type):
    """
    Number of onsets event over a threshold per second

    :param filter_type, either lowpass or highpass
    """

    def _onset_events(**kwargs):
        """
        A different onset ratio is calculated for a low band
        and an high band
        """
        audio = kwargs['audio']
        fs = kwargs['fs']
        cutoff_frequency = 700  # in Hz
        ftr = sp.signal.firwin(2 ** 5 - 1, cutoff_frequency, pass_zero=filter_type, fs=fs)
        x = sp.signal.lfilter(ftr, 1, audio)
        fs = kwargs['fs']

        # duration in seconds
        duration = x.size / fs

        # root mean square energy
        rmse = librosa.feature.rms(
            y=x ** 2,
            frame_length=common.win_length,
            hop_length=common.hop_size
        ).flatten()

        # logarithmic compression
        rmse_log = np.log(1 + rmse)

        # novelty function
        nvt = np.diff(rmse_log)
        nvt[nvt < 0] = 0  # half wave rectification

        # adaptive threshold
        thr = sp.signal.medfilt(nvt, 9)
        thr += 0.01

        # hits over the threshold
        hits = np.zeros_like(nvt)
        hits[nvt > thr] = 1
        total_hits = np.sum(hits)
        hits_per_second = total_hits / duration

        return hits_per_second

    return _onset_events


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
    'zcr_mean': zcr_mean,
    'spectral_centroid_mean': spectral_centroid_mean,
    'spectral_decrease_mean': spectral_decrease_mean,
    'spectral_flux': spectral_flux,
    'spectral_rolloff': spectral_rolloff,
    'onset_rate_low': onset_events('lowpass'),
    'onset_rate_high': onset_events('highpass'),

}

# Manually adding the features for every cep_coeffient
for c in range(common.cep_start, common.cep_end):
    feature_functions['mfcc_' + str(c) + '_mean'] = mfcc_mean(c)
    feature_functions['mfcc_' + str(c) + '_std'] = mfcc_std(c)
