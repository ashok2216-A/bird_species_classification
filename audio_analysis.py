import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import librosa
import librosa.display
from IPython.display import Audio
import streamlit as st

class AudioVisualizer:
    def __init__(self, file_path):
        # Initialize with file path
        self.file_path = file_path
        self.audio_data, self.sampling_rate = librosa.load(file_path)

    def audio_waveform(self):
        # Calculate the duration of the audio file
        duration = len(self.audio_data) / self.sampling_rate
        # Create a time array for plotting
        time = np.arange(0, duration, 1 / self.sampling_rate)
        # Plot the waveform
        plt.figure(figsize=(30, 10))
        plt.plot(time, self.audio_data, color='blue')
        plt.title('Audio Waveform')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plot = st.pyplot(plt)
        return plot

    def spectrogram(self):
        # Compute the short-time Fourier transform (STFT)
        D = librosa.stft(self.audio_data)
        # Convert magnitude spectrogram to decibels
        DB = librosa.amplitude_to_db(abs(D))
        # Plot the spectrogram
        plt.figure(figsize=(20, 5))
        librosa.display.specshow(DB, sr=self.sampling_rate, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.tight_layout()
        plot = st.pyplot(plt)
        return plot

    def audio_signals(self):
        # Call both audio waveform and spectrogram
        aw = self.audio_waveform()
        spg = self.spectrogram()
        return aw, spg


# Example usage:
# file_path = 'your_audio_file_path_here.wav'
# audio_viz = AudioVisualizer(file_path)
# aw, spg = audio_viz.audio_signals()
