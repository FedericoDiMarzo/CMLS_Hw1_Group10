import librosa
import features
import os
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt


def preprocess(audio_path):
    """
    Process an audio file from a path,
    calculating mfcc.

    :param audio_path: url of the audio file
    :return: (mfcc, windowed_audio, fs)
    """
    # TODO: fine tune these parameters
    n_fft = 1024
    win_length = 1024
    hop_size = int(win_length / 2)
    n_mels = 40
    n_cep = 12  # tipically 10-14
    window = 'hann'
    fmin = 133.33
    fmax = 6853.8

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

    mfcc = sp.fft.dct(mel_log_spectrogram, norm='ortho', axis=0)[1:n_cep + 1]

    return mfcc, windowed_audio, fs,


def extract_features(audio_path):
    """
    Calculates all features defined in features module for
    a certain target file.
    :param url of the audio file
    :return: DataFrame of computed features
    """

    mfcc, audio, fs = preprocess(audio_path)

    # iterating through the feature functions
    computed_features = np.zeros(len(features.feature_functions))
    for i, func_name in enumerate(sorted(features.feature_functions)):
        func = features.feature_functions[func_name]
        computed_features[i] = func(mfcc, audio, fs)

    return computed_features


def extract_all():
    train_root = 'sources'
    classes = ['classical', 'country', 'disco', 'jazz']

    n_files = sum([len(files) for r, d, files in os.walk(train_root)])
    features_names = sorted(features.feature_functions)
    features_names.extend(classes)
    number_of_features = len(features.feature_functions)
    train_set = np.zeros((n_files, len(features_names)))

    i = 0
    for cls in classes:
        folder_path = os.path.join(train_root, cls)
        audio_path_list = [os.path.join(folder_path, audio_path) for audio_path in os.listdir(folder_path)]
        for audio_path in audio_path_list:
            # calculates the features for a track
            train_set[i, :number_of_features] = extract_features(audio_path)

            # setting the label for the class
            class_index = number_of_features + classes.index(cls)
            train_set[i, class_index] = 1

            i += 1

    data_frame = pd.DataFrame(train_set, columns=features_names)

    return data_frame
