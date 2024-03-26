import os
import matplotlib.pyplot as plt
plt.style.use('dark_background')
import seaborn as sns
import numpy as np
import librosa
from IPython.display import Audio
import librosa
import streamlit as st


def audio_waveframe(file_path):
    # Load the audio file
    audio_data, sampling_rate = librosa.load(file_path)
    # Calculate the duration of the audio file
    duration = len(audio_data) / sampling_rate
    # Create a time array for plotting
    time = np.arange(0, duration, 1/sampling_rate)
    # Plot the waveform
    plt.figure(figsize=(30, 10))
    plt.plot(time, audio_data, color='blue')
    plt.title('Audio Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    # plt.savefig('audio_waveframe.png')
    plot = st.pyplot(plt)

    return plot


# def spectrogram(file_path):
#     # Compute the short-time Fourier transform (STFT)
#     n_fft = 500  # Number of FFT points 2048
#     hop_length = 1  # Hop length for STFT 512
#     audio_data, sampling_rate = librosa.load(file_path)
#     stft = librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length)
#     # Convert the magnitude spectrogram to decibels (log scale)
#     spectrogram = librosa.amplitude_to_db(np.abs(stft))
#     # Plot the spectrogram
#     plt.figure(figsize=(30, 10))
#     # librosa.display.specshow(spectrogram, sr=sampling_rate, hop_length=hop_length, x_axis='time', y_axis='linear')
#     librosa.display.specshow(spectrogram, sr=sampling_rate, hop_length=hop_length)
#     plt.colorbar(format='%+2.0f dB')
#     plt.title('Spectrogram')
#     plt.xlabel('Time (s)')
#     plt.ylabel('Frequency (Hz)')
#     plt.tight_layout()
#     # plt.savefig('spectrogram.png')
#     plot = st.pyplot(plt)

#     return plot


def spectrogram(file_path):
    
    y, sr = librosa.load(file_path)
    # Compute the spectrogram
    D = librosa.stft(y)
    # Convert magnitude spectrogram to decibels
    DB = librosa.amplitude_to_db(abs(D))
    # Plot the spectrogram
    plt.figure(figsize=(10, 5))
    librosa.display.specshow(DB, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()
    plot = st.pyplot(plt)

    return plot



def audio_signals(file_path):
    aw = audio_waveframe(file_path)
    spg = spectrogram(file_path)

    return aw, spg
