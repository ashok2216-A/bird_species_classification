import os
import matplotlib.pyplot as plt
plt.style.use('dark_background')
import seaborn as sns
import numpy as np
import librosa
from IPython.display import Audio
import pandas as pd


# Function to extract features from audio file
def extract_features(file_path):
    # Load audio file
    audio, sample_rate = librosa.load(file_path)
    # Extract features using Mel-Frequency Cepstral Coefficients (MFCC)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    # Flatten the features into a 1D array
    flattened_features = np.mean(mfccs.T, axis=0)
    
    return flattened_features

# Function to load dataset and extract features
def load_data_and_extract_features(data_dir):
    labels = []
    features = []
    # Loop through each audio file in the dataset directory
    for filename in os.listdir(data_dir):
        if filename.endswith('.wav'):
            file_path = os.path.join(data_dir, filename)
            # Extract label from filename
            label = filename.split('-')[0]
            labels.append(label)
            # Extract features from audio file
            feature = extract_features(file_path)
            features.append(feature)
            
    return np.array(features), np.array(labels)
