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

# # Get the Hugging Face token from environment variables
# load_dotenv()
# hf_token = os.getenv("HF_TOKEN")
# if hf_token is None:
#     raise ValueError("Hugging Face token not found. Please set the HF_TOKEN environment variable.")
# # Authenticate with Hugging Face
# login(hf_token)

# # Model information and links
# model_links = {
#     "Zephyr-7B": "HuggingFaceH4/zephyr-7b-beta"
# }
# model_info = {
#     "Zephyr-7B": {
#         'description': """Zephyr 7B is a Huggingface model, fine-tuned for helpful and instructive interactions.""",
#         'logo': 'https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha/resolve/main/thumbnail.png'
#     }
# }

# # Inference API Initialization
# client = InferenceClient('HuggingFaceH4/zephyr-7b-beta')

# # Reset conversation button
# def reset_conversation():
#     return [
#         {"role": "system", "content": "You are a knowledgeable and empathetic ornithologist assistant providing accurate and relevant information based on user input."}
#     ]

# # Initialize conversation and chat history
# messages = reset_conversation()

# # Display chat history
# for message in messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# def respond(message, history, max_tokens, temperature, top_p):
#     # Prepare the list of messages for the chat completion
#     messages = [{"role": "system", "content": history[0]["content"]}]

#     for val in history:
#         if val["role"] == "user":
#             messages.append({"role": "user", "content": val["content"]})
#         elif val["role"] == "assistant":
#             messages.append({"role": "assistant", "content": val["content"]})

#     messages.append({"role": "user", "content": message})

#     # Generate response
#     response = ""
#     response_container = st.empty()  # Placeholder to update the response text dynamically

#     for message in client.chat_completion(
#         messages,
#         max_tokens=max_tokens,
#         stream=True,
#         temperature=temperature,
#         top_p=top_p,
#     ):
#         token = message.choices[0].delta.content
#         response += token
#         # response_container.text(response)  # Stream the response

#     return response
# # Load environment variables
# load_dotenv()
# hf_token = os.getenv("HF_TOKEN")
# if hf_token is None:
#     raise ValueError("Hugging Face token not found. Please set the HF_TOKEN environment variable.")
# # Authenticate with Hugging Face
# login(hf_token)

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
        
#         user_input = f"Explain about {pred} bird"
#         # Generate and display assistant response   
#         if user_input:
#             response = respond(user_input, messages, max_tokens = 500, temperature = 0.70, top_p = 0.95)
#             st.markdown(response)
#             messages.append({"role": "assistant", "content": response})
#             st.page_link(wikipedia.page(pred).url, label="Explore more in Wikipedia.com", icon="🌎")
#             # st.session_state.messages.append({"role": "assistant", "content": response})
#     else:
#         st.write('Class not Found')      
# else:
#     st.markdown('File not Found!')

