import os
import warnings
import librosa
import streamlit as st
import tempfile
import json
from joblib import dump, load
# import soundfile as sf
from audio_analysis import audio_signals
from audio_processing import extract_features


st.header('Bird Species Classification')
st.markdown('Sound of 114 Species of Birds :bird: :penguin: :hatched_chick:')
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

st.header('', divider='rainbow')
st.markdown('Download the Sample Audio here :point_down:')
st.page_link("https://dibird.com/", label="DiBird.com", icon="🌎")
st.subheader('Scientific Name of 114 Species of Birds :bird:')
with st.container(height=300):
    st.markdown(list(labels_list.values()))

if audio_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(audio_file.read())
        st.success("Audio file successfully uploaded and stored temporally.")
    file_path = tmp_file.name
    audio_data, sampling_rate = librosa.load(file_path)
    st.audio(audio_data, sample_rate=sampling_rate)
    audio_signals(file_path)
    
    # Load the model
    model_path = 'model.joblib'
    model = load_model(model_path)
    
    class_file = open('classes.json', 'r').read()
    labels_list = json.loads(class_file)
    # Predict the emotion
    y_predict = predict_emotion(file_path, model)
    
    # Display predicted class
    if str(y_predict[0]) in labels_list.keys():
        st.subheader(f'Predicted Class: :rainbow[{labels_list[str(y_predict[0])][:-6]}]')
    else:
        st.write('Class not Found')
            
else:
    st.markdown('File not Found!')
    

