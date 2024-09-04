import os
import numpy as np
import warnings
import librosa
import streamlit as st
import tempfile
import json
import pandas as pd
from joblib import dump, load
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
# import soundfile as sf
from audio_analysis import audio_signals
from audio_processing import extract_features


st.header('Bird Species Classification')
st.markdown('Sound of 114 Species of Birds :bird: :penguin: :hatched_chick:')
st.header('', divider='rainbow')

# Decorator for caching function results
@st.cache_data
def loaded_model(model_path):
    return load_model(model_path)

@st.cache_data
def predict_class(audio_path, model):
    extracted_feature = extract_features(audio_path)
    extracted_feature = extracted_feature.reshape(1, 1, extracted_feature.shape[0])
    prediction = model.predict(extracted_feature)
    predicted_class_index = np.argmax(prediction)
    print('HI',predicted_class_index)
    # predicted_class_label = label_encoder.inverse_transform([predicted_class_index])[0]
    return predicted_class_index

audio_file = st.file_uploader("Upload an Audio file", type=["mp3", "wav", "ogg"], accept_multiple_files=False)
# Load the model
model_path = 'bird_audio_classification_model.h5'
model = loaded_model(model_path)

class_file = open('classes.json', 'r').read()
labels_list = json.loads(class_file)

st.markdown('Download the Sample Audio here :point_down:')
st.page_link("https://dibird.com/", label="DiBird.com", icon="🌎")
st.subheader('Scientific Name of 114 Species of Birds :bird:')
# with st.container(height=300):
#     st.markdown(list(labels_list.values()))
birds = pd.DataFrame(list(class_file.items()), columns=['ID', 'Birds'])
st.table(birds)
st.header('', divider='rainbow')

if audio_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(audio_file.read())
        st.success("Audio file successfully uploaded and stored temporally.")
    file_path = tmp_file.name
    audio_data, sampling_rate = librosa.load(file_path)
    st.audio(audio_data, sample_rate=sampling_rate)
    audio_signals(file_path)
    # Predict the class
    y_predict = predict_class(file_path, model)
    # Display predicted class
    if str(y_predict) in labels_list.keys():
        st.subheader(f'Predicted Class: :rainbow[{labels_list[str(y_predict)][:-6]}]')
    else:
        st.write('Class not Found')      
else:
    st.markdown('File not Found!')
    
