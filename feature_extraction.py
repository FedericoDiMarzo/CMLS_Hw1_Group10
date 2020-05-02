import librosa
import features
import os
import numpy as np
import scipy as sp
import pandas as pd
import time
import matplotlib.pyplot as plt
import common


def preprocess(audio_path):
    """
    Process an audio file from a path,
    calculating mfcc.

    :param audio_path: url of the audio file
    :return: (mfcc, windowed_audio, fs)
    """
    # TODO: fine tune these parameters
    n_fft = common.n_fft
    win_length = common.win_length
    hop_size = common.hop_size
    n_mels = common.n_mels
    cep_start = common.cep_start
    cep_end = common.cep_end
    window = common.window
    fmin = common.fmin
    fmax = common.fmax

    # loading from the path
    audio, fs = librosa.load(audio_path, sr=None)  # TODO: do we need preprocessing?

    # audio preprocessing
    audio = audio / np.max(audio)

    # time domain windowing
    window = sp.signal.get_window(window=window, Nx=win_length)
    n_window = int(np.floor((len(audio) - win_length) / hop_size))
    windowed_audio = np.zeros((n_window, win_length))

    for i in range(n_window):
        windowed_audio[i] = audio[i * hop_size:i * hop_size + win_length] * window

    # exctracting mfcc
    stft = np.abs(librosa.stft(
        y=audio,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_size,
        window=window,
    )) ** 2

    mel_filter = librosa.filters.mel(
        sr=fs,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax
    )

    mel_log_spectrogram = np.log10(np.dot(mel_filter, stft) + 1e-16)

    mfcc = sp.fft.dct(mel_log_spectrogram, norm='ortho', axis=0)[cep_start:cep_end]

    return stft, mfcc, windowed_audio, fs, audio, mel_log_spectrogram


def extract_features(audio_path):
    """
    Calculates all features defined in features module for
    a certain target file.

    :param audio_path of the audio file
    :return: DataFrame of computed features
    """

    stft, mfcc, windowed_audio, fs, audio, mel_log_spectrogram = preprocess(audio_path)

    # iterating through the feature functions
    computed_features = np.zeros(len(features.feature_functions))
    for i, func_name in enumerate(sorted(features.feature_functions)):
        func = features.feature_functions[func_name]
        computed_features[i] = func(
            stft=stft,
            mfcc=mfcc,
            windowed_audio=windowed_audio,
            fs=fs,
            audio=audio,
            mel_log_spectrogram=mel_log_spectrogram
        )

    return computed_features


def extract_all():
    """
    Calculates the features for every source song.

    :return: the whole DataFrame
    """
    # setting a timer to calculate function exectution time
    timer_start = time.time()

    train_root = 'sources'

    n_files = sum([len(files) for r, d, files in os.walk(train_root)])
    features_names = sorted(features.feature_functions)
    features_names.append('CLASS')
    number_of_features = len(features.feature_functions)
    train_set = np.zeros((n_files, len(features_names)))

    i = 0
    for cls_index, cls in enumerate(common.classes):
        folder_path = os.path.join(train_root, cls)
        audio_path_list = [os.path.join(folder_path, audio_path) for audio_path in os.listdir(folder_path)]
        for audio_path in audio_path_list:
            # calculates the features for a track
            train_set[i, :number_of_features] = extract_features(audio_path)

            # setting the label for the class
            train_set[i, -1] = cls_index

            i += 1

        # console output
        print('class ' + cls + ' extracted...')

    data_frame = pd.DataFrame(train_set, columns=features_names)

    # console output
    execution_time = time.time() - timer_start
    print('',
          '-- feature_extraction completed --',
          'tracks: ' + str(n_files),
          'features: ' + str(number_of_features),
          'execution time: ' + '{:.2f}'.format(execution_time) + 's',
          '',
          sep="\n")

    return data_frame
