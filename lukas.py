import librosa
import numpy as np
import os
import IPython.display as ipd
import matplotlib.pyplot as plt
import librosa.display


audio_path = '/Users/lukasmalm/Desktop/computermusic/CMLS_Hw1_Group10/sources/jazz/jazz.00059.au'
audio, fs = librosa.load(audio_path, sr=None)
ipd.Audio(audio_path)

#display waveform

plt.figure(figsize=(14, 5))
librosa.display.waveplot(audio, sr=fs)
plt.show()


def chroma_freq():
    # Loadign the file
    x, sr = librosa.load('../simple_piano.wav')hop_length = 512
    chromagram = librosa.feature.chroma_stft(x, sr=sr, hop_length=hop_length)
    plt.figure(figsize=(15, 5))
    librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='coolwarm')