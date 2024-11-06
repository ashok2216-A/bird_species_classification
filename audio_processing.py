import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Audio
import pandas as pd


class AudioFeatureExtractor:
    def __init__(self, data_dir):
        """
        Initialize the AudioFeatureExtractor object.

        Parameters:
        data_dir (str): Path to the directory containing audio files.
        """
        self.data_dir = data_dir
        self.features = []
        self.labels = []

    def extract_features(self, file_path):
        """
        Extract features from a single audio file.

        Parameters:
        file_path (str): Path to the audio file.

        Returns:
        np.array: Flattened features (MFCCs) from the audio file.
        """
        # Load audio file
        audio, sample_rate = librosa.load(file_path)
        # Extract features using Mel-Frequency Cepstral Coefficients (MFCC)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        # Flatten the features into a 1D array (mean of MFCCs across time)
        flattened_features = np.mean(mfccs.T, axis=0)
        return flattened_features

    def load_data_and_extract_features(self):
        """
        Load dataset from the directory and extract features from each audio file.

        Returns:
        np.array: Extracted features.
        np.array: Labels for each audio file.
        """
        # Loop through each audio file in the dataset directory
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.wav'):
                file_path = os.path.join(self.data_dir, filename)
                # Extract label from filename (assumes label is in the first part of the filename)
                label = filename.split('-')[0]
                self.labels.append(label)
                # Extract features from audio file
                feature = self.extract_features(file_path)
                self.features.append(feature)

        return np.array(self.features), np.array(self.labels)

# Usage
# data_dir = '/path/to/your/audio/dataset'  # Specify the path to your dataset
# audio_extractor = AudioFeatureExtractor(data_dir)
# features, labels = audio_extractor.load_data_and_extract_features()

# # Optionally, you can check the features and labels
# print(f"Features shape: {features.shape}")
# print(f"Labels: {labels[:5]}")
