import os
import warnings
import librosa
# import soundfile as sf
import streamlit as st
import tempfile
from joblib import dump, load
from audio_analysis import audio_signals
from audio_processing import extract_features

st.header('Bird Species Classification')
st.markdown('Sound of 114 Species of Birds :bird:')
st.header('', divider='rainbow')

# Decorator for caching function results
@st.cache_data
def load_model(model_path):
    return load(model_path)

@st.cache_data
def predict_emotion(audio_path, _model):
    extracted_features = extract_features(audio_path).reshape(1, -1)
    return _model.predict(extracted_features)


audio_file = st.file_uploader("Upload an Audio file", type=["mp3", "wav", "ogg"], accept_multiple_files=False)

if audio_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(audio_file.read())
        st.success("Audio file successfully uploaded and stored temporally.")
    file_path = tmp_file.name
    audio_signals(file_path)
    audio_data, sampling_rate = librosa.load(file_path)
    st.audio(audio_data, sample_rate=sampling_rate)
    
    # Load the model
    model_path = 'model.joblib'
    model = load_model(model_path)
    
    class_file = open('classes.txt', 'r')
    encoded_class_file = open('encoded_classes.txt', 'r')
    labels_list = class_file.read()
    encoded_label = encoded_class_file.read()
    # Predict the emotion
    y_predict = predict_emotion(file_path, model)
    st.write(y_predict)
    labels = {}
    for label, prediction in zip(encoded_label, labels_list):
        labels[label] = prediction
    
    # Display predicted class
    if y_predict[0] in labels.keys():
        st.subheader(f'Predicted Class: :rainbow[{labels[y_predict[0]]}]')
    else:
        st.write('Class not Found')
else:
    st.markdown('File not Found!')
