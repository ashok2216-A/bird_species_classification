import os
import warnings
import librosa
import streamlit as st
from joblib import dump, load
from audio_analysis import audio_signals
from audio_processing import extract_features

st.header('Bird Species Classification')
st.markdown('Sound of 114 Species of Birds :bird:')
st.header('', divider='rainbow')

uploaded_file = st.file_uploader("Choose a Audio file", accept_multiple_files=False)

# Check if a file is uploaded
if uploaded_file is not None:
    # Get the path of the uploaded file
    file_path = uploaded_file.name
    st.write("Uploaded file path:", file_path)
    if os.path.exists(file_path):
        st.write("Uploaded file path:", file_path)
    else:
        st.write("Error: File not found.")

audio_data, sampling_rate = librosa.load(file_path)
st.audio(audio_data, sample_rate=sampling_rate)

audio_signals(file_path)

# Decorator for caching function results
@st.cache_data
def load_model(model_path):
    return load(model_path)

@st.cache_data
def predict_emotion(audio_path, _model):
    extracted_features = extract_features(audio_path).reshape(1, -1)
    return _model.predict(extracted_features)


# Load the model
model_path = 'model.joblib'
model = load_model(model_path)

# Predict the emotion
y_predict = predict_emotion(file_path, model)

# Mapping for emotion labels
labels_list = ['Fear', 'Angry', 'Neutral', 'Sad', 'Pleasant_Suprised', 'Disgust', 'Happy']
encoded_label = [2, 0, 4, 6, 5, 1, 3]

labels = {}
for label, prediction in zip(encoded_label, labels_list):
    labels[label] = prediction

# Display predicted class
if y_predict[0] in labels.keys():
    st.subheader(f'Predicted Class: :rainbow[{labels[y_predict[0]]}]')




# import os
# import warnings
# import librosa
# import streamlit as st
# from joblib import dump, load
# from audio_analysis import audio_signals
# from audio_processing import extract_features
# # from audio_record import record_audio

# st.header('Emotional Audio Classification', divider='rainbow')

# emotion = st.selectbox(
#     'Select a Emotion',
#     ('Angry', 'Disgust', 'Fear',
#        'Happy', 'Neutral', 'Pleasant_Suprised','Sad'))

# st.write('You selected:', emotion)

# dirs = os.listdir(f'raw_audio_data/{emotion}')

# if emotion == 'Angry':

#     file_list = []
#     for dir_list in dirs:
#     	files = ('raw_audio_data/'+emotion+'/'+dir_list)
#     	file_list.append(files)
        
#     file_path = st.selectbox(
#     'Select a Audio file',
#     tuple(file_list))

# elif emotion == 'Disgust':
    
#     file_list = []
#     for dir_list in dirs:
#     	files = ('raw_audio_data/'+emotion+'/'+dir_list)
#     	file_list.append(files)
        
#     file_path = st.selectbox(
#     'Select a Audio file',
#     tuple(file_list))
    
# elif emotion == 'Fear':
    
#     file_list = []
#     for dir_list in dirs:
#     	files = ('raw_audio_data/'+emotion+'/'+dir_list)
#     	file_list.append(files)
        
#     file_path = st.selectbox(
#     'Select a Audio file',
#     tuple(file_list))
    
# elif emotion == 'Happy':
    
#     file_list = []
#     for dir_list in dirs:
#     	files = ('raw_audio_data/'+emotion+'/'+dir_list)
#     	file_list.append(files)
        
#     file_path = st.selectbox(
#     'Select a Audio file',
#     tuple(file_list))
    
# elif emotion == 'Neutral':
    
#     file_list = []
#     for dir_list in dirs:
#     	files = ('raw_audio_data/'+emotion+'/'+dir_list)
#     	file_list.append(files)
        
#     file_path = st.selectbox(
#     'Select a Audio file',
#     tuple(file_list))
    
# elif emotion == 'Pleasant_Suprised':
    
#     file_list = []
#     for dir_list in dirs:
#     	files = ('raw_audio_data/'+emotion+'/'+dir_list)
#     	file_list.append(files)
        
#     file_path = st.selectbox(
#     'Select a Audio file',
#     tuple(file_list))
    
# elif emotion == 'Sad':

#     file_list = []
#     for dir_list in dirs:
#     	files = ('raw_audio_data/'+emotion+'/'+dir_list)
#     	file_list.append(files)
        
#     file_path = st.selectbox(
#     'Select a Audio file',
#     tuple(file_list))

# else:
#     st.subheader('File not found')

# audio_signals(file_path)


# audio_data, sampling_rate = librosa.load(file_path)
# audio_data = st.audio(audio_data, sample_rate=sampling_rate)

# model_path = 'audio_classifier_model.joblib'
# model = load(model_path)

# audio = file_path
# print(audio)

# extracted_features = extract_features(audio).reshape(1, -1)
# # extracted_features = x_test[112].reshape(1, -1)
# y_predict = model.predict(extracted_features)
# labels_list = ['Fear', 'Angry', 'Neutral', 'Sad', 'Pleasant_Suprised', 'Disgust', 'Happy']
# encoded_label = [2, 0, 4, 6, 5, 1, 3]

# labels = {}
# for label, prediction in zip(encoded_label, labels_list):
#     labels[label] = prediction
# if y_predict[0] in labels.keys():
#     st.subheader(f'Predicted Class: :rainbow[{labels[y_predict[0]]}]')
