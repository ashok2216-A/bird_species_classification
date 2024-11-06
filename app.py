# import os
# import numpy as np
# import warnings
# import librosa
# import streamlit as st
# import tempfile
# import json
# from PIL import Image
# import pandas as pd
# from joblib import dump, load
# import wikipedia
# import requests
# # import wikipediaapi
# # import LLM
# from sklearn.preprocessing import LabelEncoder
# from tensorflow.keras.models import load_model
# # import soundfile as sf
# from audio_analysis import audio_signals
# from audio_processing import extract_features
# import os
# from dotenv import load_dotenv
# import json
# import streamlit as st
# from huggingface_hub import InferenceApi, login, InferenceClient

# st.set_page_config(
#     page_title="BirdSense",
#     page_icon=":bird:",
#     initial_sidebar_state="expanded",
#     menu_items={
#         'Get Help': 'https://ashok2216-myportfolio-github-io.vercel.app/#contact',
#         'Report a bug': "https://ashok2216-myportfolio-github-io.vercel.app/#contact",
#         'About': "https://ashok2216-myportfolio-github-io.vercel.app/"
#     }
# )


# # Load environment variables
# load_dotenv()
# hf_token = os.getenv("HF_TOKEN")
# if hf_token is None:
#     raise ValueError("Hugging Face token not found. Please set the HF_TOKEN environment variable.")

# # Authenticate with Hugging Face
# login(hf_token)
# hf_api_key = st.secrets["HF_TOKEN"]
# image = Image.open('logo.PNG')
# st.image(
#     image, width=250
# )
# st.subheader('Bird Species Classification')
# # st.markdown('Sound of 114 Bird Species :bird: :penguin: :hatched_chick:')
# st.header('', divider='rainbow')

# @st.cache_data
# def loaded_model(model_path):
#     return load_model(model_path)

# @st.cache_data
# def predict_class(audio_path, model):
#     extracted_feature = extract_features(audio_path)
#     extracted_feature = extracted_feature.reshape(1, 1, extracted_feature.shape[0])
#     prediction = model.predict(extracted_feature)
#     predicted_class_index = np.argmax(prediction)
#     print('HI',predicted_class_index)
#     # predicted_class_label = label_encoder.inverse_transform([predicted_class_index])[0]
#     return predicted_class_index

# @st.cache_data
# def get_bird_details(predicted_class):
#     headers = {
#         "Authorization": f"Bearer {hf_token}",
#         "Content-Type": "application/json",
#     }
#     payload = {
#         "inputs": f"Tell me about the bird species {predicted_class}",
#     }

#     response = requests.post(
#         "https://api-inference.huggingface.co/models/zephyr",
#         headers=headers,
#         json=payload
#     )

#     if response.status_code == 200:
#         return response.json()[0]['generated_text']
#     else:
#         st.error("Failed to retrieve bird details from Zephyr.")
#         return None

# audio_file = st.file_uploader("Upload an Audio file", type=["mp3", "wav", "ogg"], accept_multiple_files=False)
# # Load the model
# model_path = 'bird_audio_classification_model.h5'
# model = loaded_model(model_path)

# class_file = open('classes.json', 'r').read()
# labels_list = json.loads(class_file)

# st.markdown('Download the Sample Audio here :point_down:')
# st.page_link("https://dibird.com/", label="DiBird.com", icon="🐦")
# st.subheader('Scientific Name of 114 Birds Species :bird:')

# with st.container(height=300):
#     st.markdown(list(labels_list.values()))
# # birds = pd.DataFrame(class_file)
# # st.table(birds)
# st.header('', divider='rainbow')

# if audio_file is not None:
#     with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
#         tmp_file.write(audio_file.read())
#         st.success("Audio file successfully uploaded and stored temporally.")
#     file_path = tmp_file.name
#     audio_data, sampling_rate = librosa.load(file_path)
#     st.audio(audio_data, sample_rate=sampling_rate)
#     audio_signals(file_path)
#     # Predict the class
#     y_predict = predict_class(file_path, model)
#     # Display predicted class
#     if str(y_predict) in labels_list.keys():
#         pred = labels_list[str(y_predict)][:-6]
#         st.subheader(f'Predicted Class: :rainbow[{pred}]') 
#         st.image(wikipedia.page(pred).images[0], caption=labels_list[str(y_predict)][:-6], width=200)
#         st.markdown(wikipedia.summary(pred))
#         get_bird_details(pred)
#         st.subheader(f'Predicted Class: {pred}')
#         st.markdown(bird_details)
        
#         # if user_input := f"Explain about {pred} bird":
#         # # Generate and display assistant response   
#         #     response = LLM.respond(user_input, st.session_state.messages, max_tokens = 500, temperature = 0.70, top_p = 0.95)
#         #     st.markdown(response)
#         #     st.page_link(wikipedia.page(pred).url, label="Explore more in Wikipedia.com", icon="🌎")
#             # st.session_state.messages.append({"role": "assistant", "content": response})
#     else:
#         st.write('Class not Found')      
# else:
#     st.markdown('File not Found!')

import os
import numpy as np
import warnings
import librosa
import streamlit as st
import tempfile
import json
from PIL import Image
import pandas as pd
from joblib import dump, load
import wikipedia
from huggingface_hub import InferenceApi, login
from audio_analysis import audio_signals
from audio_processing import extract_features

# Load Hugging Face token from environment variable
hf_token = os.getenv("HF_TOKEN")
if hf_token is None:
    raise ValueError("Hugging Face token not found. Please set the HF_TOKEN environment variable.")

# Authenticate with Hugging Face API
login(hf_token)

# Initialize Hugging Face Inference API for Zephyr model
client = InferenceApi(repo_id="HuggingFaceH4/zephyr-7b-beta")

# Streamlit configuration
st.set_page_config(
    page_title="BirdSense",
    page_icon=":bird:",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://ashok2216-myportfolio-github-io.vercel.app/#contact',
        'Report a bug': "https://ashok2216-myportfolio-github-io.vercel.app/#contact",
        'About': "https://ashok2216-myportfolio-github-io.vercel.app/"
    }
)

# Load image and display
image = Image.open('logo.PNG')
st.image(image, width=250)
st.subheader('Bird Species Classification')
st.header('', divider='rainbow')

# Cache for loading the model
@st.cache_data
def loaded_model(model_path):
    return load_model(model_path)

# Predict the bird class based on the audio
@st.cache_data
def predict_class(audio_path, model):
    extracted_feature = extract_features(audio_path)
    extracted_feature = extracted_feature.reshape(1, 1, extracted_feature.shape[0])
    prediction = model.predict(extracted_feature)
    predicted_class_index = np.argmax(prediction)
    return predicted_class_index

# File uploader for audio
audio_file = st.file_uploader("Upload an Audio file", type=["mp3", "wav", "ogg"], accept_multiple_files=False)

# Load the model
model_path = 'bird_audio_classification_model.h5'
model = loaded_model(model_path)

# Load the class labels
class_file = open('classes.json', 'r').read()
labels_list = json.loads(class_file)

# Display the sample audio download link
st.markdown('Download the Sample Audio here :point_down:')
st.page_link("https://dibird.com/", label="DiBird.com", icon="🐦")
st.subheader('Scientific Name of 114 Birds Species :bird:')

with st.container(height=300):
    st.markdown(list(labels_list.values()))

# Process uploaded audio file and predict the bird species
if audio_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(audio_file.read())
        st.success("Audio file successfully uploaded and stored temporally.")
    file_path = tmp_file.name
    audio_data, sampling_rate = librosa.load(file_path)
    st.audio(audio_data, sample_rate=sampling_rate)
    audio_signals(file_path)

    # Predict the class of the bird species
    y_predict = predict_class(file_path, model)
    
    if str(y_predict) in labels_list.keys():
        predicted_class = labels_list[str(y_predict)][:-6]
        st.subheader(f'Predicted Class: :rainbow[{predicted_class}]') 
        
        # Fetch bird details from Wikipedia
        try:
            bird_summary = wikipedia.summary(predicted_class, sentences=2)
            st.markdown(bird_summary)
        except wikipedia.exceptions.DisambiguationError as e:
            st.write("Multiple results found. Please be more specific.")
        
        # Display bird image from Wikipedia
        try:
            st.image(wikipedia.page(predicted_class).images[0], caption=predicted_class, width=200)
        except Exception as e:
            st.write("Image not available.")
        
        # Call Zephyr model for more detailed information (optional)
        response = client(inputs=f"Tell me about the {predicted_class} bird.")
        st.subheader("More Details from Zephyr Model:")
        st.write(response["generated_text"])
        
        # Provide a link to Wikipedia for further exploration
        st.page_link(wikipedia.page(predicted_class).url, label="Explore more on Wikipedia", icon="🌎")
    else:
        st.write('Class not Found')
else:
    st.markdown('File not Found!')

