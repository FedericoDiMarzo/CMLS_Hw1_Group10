import numpy as np
import scipy as sp
import common
import librosa


# TODO: some features are corrupted

def zcr(**kwargs):
    """
    Calculates the zero crossing rate vector
    """
    windowed_audio = kwargs['windowed_audio']
    fs = kwargs['fs']

    N = windowed_audio.shape[0]
    zcr = np.abs(np.diff(windowed_audio, axis=0))
    zcr = np.sum(zcr, axis=0)
    zcr *= 1 / (2 * N) * fs
    return zcr


def centroid(X):
    """
    Calculates the centroid vector from the matrix X
    :param X: time-frequency input matrix
    :return: centroid vector, containing the centroid for every column
    """
    n_frames = X.shape[1]
    k = np.arange(X.shape[0])
    k = np.reshape(k, (-1, 1))
    k = np.repeat(k, n_frames, axis=1)
    return np.sum(np.abs(X) * k, axis=0) / np.sum(X, axis=0)


def spectral_flux(X):
    """
    Calculates the spectral flux vector from the matrix X
    :param X: time-frequency input matrix
    :return: spectral flux vector, containing the spectral flux for every column
    """
    freqbins = X.shape[0]
    return np.sqrt(np.sum(np.diff(np.abs(X)) ** 2, axis=1)) / freqbins


def spectral_spread(X):
    """
    Calculates the spectral spread vector from the matrix X
    :param X: time-frequency input matrix
    :return: spectral flux vector, containing the spectral spread for every column
    """
    k = np.arange(X.shape[0])
    k = np.reshape(k, (-1, 1))
    k = np.repeat(k, X.shape[1], axis=1)
    ctr = centroid(X)
    ctr = np.reshape(ctr, (-1, 1))
    ctr = np.repeat(ctr, len(k), axis=1)
    ctr = np.transpose(ctr)  # to get the right sizes

    return np.sum(((k - ctr) ** 2) * X) / np.sum(X)


def zcr_mean(**kwargs):
    """
    Mean of the zero crossing rate for all the frames
    """
    zero_crossing_rate = zcr(**kwargs)
    return np.mean(zero_crossing_rate)


def zcr_std(**kwargs):
    """
    Standard variation of the zero crossing rate for all the frames
    """
    zero_crossing_rate = zcr(**kwargs)
    return np.std(zero_crossing_rate)


def spectral_centroid_mean(**kwargs):
    """
    Mean of the spectral centroid for all the frames
    """
    stft = kwargs['stft']
    return np.mean(centroid(stft))


def spectral_flux_mean(**kwargs):
    """
    Mean of the spectral flux for all the frames
    """
    stft = kwargs['stft']  # stft being already a power spectrum (**2 in common)
    return np.mean(spectral_flux(stft))


def spectral_spread_mean(**kwargs):
    """
    Mean of the spectral spread for all the frames
    """
    stft = kwargs['stft']  # stft being already a power spectrum (**2 in common)
    return np.mean(spectral_spread(stft))


def spectral_rolloff_mean(**kwargs):
    """
    Mean of the spectral roll-off for all the frames
    """
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
        x = mfcc[cep_coef]
        return np.std(x)

    return _mfcc_std


def chroma_mean(chroma_coef):
    """
    Closure for defining _chroma_mean for a certain coefficient
    :param chroma_coef: chroma coefficient
    :return: closure to _chroma_mean
    """

    def _chroma_mean(**kwargs):
        """
        Calculates the mean for a certain chroma coefficient
        """
        chroma_spectrum = kwargs['chroma_spectrum']
        x = chroma_spectrum[chroma_coef]
        return np.mean(x)

    return _chroma_mean


def chroma_std(chroma_coef):
    """
    Closure for defining _chroma_std for a certain coefficient
    :param chroma_coef: chroma coefficient
    :return: closure to _chroma_std
    """

    def _chroma_std(**kwargs):
        """
        Calculates the standard deviation for a certain chroma coefficient
        """
        chroma_spectrum = kwargs['chroma_spectrum']
        x = chroma_spectrum[chroma_coef]
        return np.std(x)

    return _chroma_std


def chroma_centroid_mean(**kwargs):
    """
    Calculates the mean of centroid for the chroma spectrum
    """
    chroma_spectrum = kwargs['chroma_spectrum']
    return np.mean(centroid(chroma_spectrum))


def chroma_flux_mean(**kwargs):
    """
    Calculates the mean of spectral flux for the chroma spectrum
    """
    chroma_spectrum = kwargs['chroma_spectrum']
    return np.mean(spectral_flux(chroma_spectrum))


def chroma_spread_mean(**kwargs):
    """
    Calculates the mean of spectral spread for the chroma spectrum
    """
    chroma_spectrum = kwargs['chroma_spectrum']
    return np.mean(spectral_spread(chroma_spectrum))


def chroma_max(**kwargs):
    """
    Calculates the max chroma bin
    """
    chroma_spectrum = kwargs['chroma_spectrum']
    return np.max(chroma_spectrum.flatten())


def chroma_min(**kwargs):
    """
    Calculates the min chroma bin
    """
    chroma_spectrum = kwargs['chroma_spectrum']
    return np.min(chroma_spectrum.flatten())


# Used to store feature name and function reference
feature_functions = {
    'zcr_mean': zcr_mean,
    'zcr_std': zcr_std,
    'spectral_centroid_mean': spectral_centroid_mean,
    'spectral_decrease_mean': spectral_decrease_mean,
    'spectral_flux_mean': spectral_flux_mean,
    'spectral_rolloff_mean': spectral_rolloff_mean,
    'spectral_spread_mean': spectral_spread_mean,
    'onset_rate_low': onset_events('lowpass'),
    'onset_rate_high': onset_events('highpass'),
    'chroma_max': chroma_max,
    'chroma_min': chroma_min,
    'chroma_centroid_mean': chroma_centroid_mean,
    'chroma_flux_mean': chroma_flux_mean,
    'chroma_spread_mean': chroma_spread_mean,

}

# Manually adding the features for every cep_coeffient
for c in range(common.cep_end - common.cep_start):
    feature_functions['mfcc_' + str(c + common.cep_start) + '_mean'] = mfcc_mean(c)
    feature_functions['mfcc_' + str(c + common.cep_start) + '_std'] = mfcc_std(c)

# Manually adding the features for every chroma
for c in range(12):
    feature_functions['chroma_' + str(c) + '_mean'] = chroma_mean(c)
    feature_functions['chroma_' + str(c) + '_std'] = chroma_std(c)
