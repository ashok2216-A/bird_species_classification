import os
import numpy as np
import warnings
import librosa
import streamlit as st
import tempfile
import json
from PIL import Image
from joblib import dump, load
import wikipedia
from tensorflow.keras.models import load_model
from huggingface_hub import InferenceClient, login
from dotenv import load_dotenv
from audio_processing import AudioFeatureExtractor
from audio_analysis import audio_signals


class BirdClassifier:
    def __init__(self, model_path, class_file):
        self.model = load_model(model_path)
        self.labels_list = self.load_labels(class_file)
    
    @staticmethod
    def load_labels(class_file):
        with open(class_file, 'r') as file:
            return json.load(file)
    
    def predict_class(self, audio_path):
        extracted_feature = extract_features(audio_path)
        extracted_feature = extracted_feature.reshape(1, 1, extracted_feature.shape[0])
        prediction = self.model.predict(extracted_feature)
        predicted_class_index = np.argmax(prediction)
        return predicted_class_index


class Assistant:
    def __init__(self, hf_token, model_name):
        load_dotenv()
        self.hf_token = hf_token
        self.model_name = model_name
        self.client = InferenceClient(model_name)
        self.login()
        self.messages = self.reset_conversation()

    def login(self):
        if self.hf_token is None:
            raise ValueError("Hugging Face token not found. Please set the HF_TOKEN environment variable.")
        login(self.hf_token)

    @staticmethod
    def reset_conversation():
        return [
            {"role": "system", "content": "You are a knowledgeable and empathetic ornithologist assistant providing accurate and relevant information based on user input."}
        ]
    
    def respond(self, message, max_tokens=500, temperature=0.70, top_p=0.95):
        # Prepare the list of messages for the chat completion
        messages = [{"role": "system", "content": self.messages[0]["content"]}]
        for val in self.messages:
            messages.append({"role": val["role"], "content": val["content"]})

        messages.append({"role": "user", "content": message})

        response = ""
        for msg in self.client.chat_completion(
                messages,
                max_tokens=max_tokens,
                stream=True,
                temperature=temperature,
                top_p=top_p,
        ):
            token = msg.choices[0].delta.content
            response += token

        self.messages.append({"role": "assistant", "content": response})
        return response


class App:
    def __init__(self, bird_classifier, assistant):
        self.bird_classifier = bird_classifier
        self.assistant = assistant
        self.image = Image.open('logo.PNG')
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
    
    def display_logo(self):
        st.image(self.image, width=250)
    
    def display_intro(self):
        st.subheader('Bird Species Classification')
        st.header('', divider='rainbow')

    def container_heading(self):
        st.markdown('Download the Sample Audio here :point_down:')
        st.page_link("https://dibird.com/", label="DiBird.com", icon="🐦")
        st.subheader('Scientific Name of 114 Birds Species :bird:')
        with st.container(height=300):
            st.markdown(list(labels_list.values()))
        st.header('', divider='rainbow')
    
    def display_audio_input(self):
        audio_file = st.file_uploader("Upload an Audio file", type=["mp3", "wav", "ogg"], accept_multiple_files=False)
        if audio_file is not None:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(audio_file.read())
                st.success("Audio file successfully uploaded and stored temporarily.")
            return tmp_file.name
        return None

    def process_audio(self, file_path):
        audio_data, sampling_rate = librosa.load(file_path)
        st.audio(audio_data, sample_rate=sampling_rate)
        audio_signals(file_path)
        
        y_predict = self.bird_classifier.predict_class(file_path)
        return y_predict

    def display_prediction(self, y_predict, labels_list):
        if str(y_predict) in labels_list.keys():
            pred = labels_list[str(y_predict)][:-6]
            st.subheader(f'Predicted Class: :rainbow[{pred}]') 
            st.image(wikipedia.page(pred).images[0], caption=labels_list[str(y_predict)][:-6], width=200)
            st.markdown(wikipedia.summary(pred))
            response = self.assistant.respond(f"Explain about {pred} bird")
            st.markdown(response)
            st.page_link(wikipedia.page(pred).url, label="Explore more in Wikipedia.com", icon="🌎")
        else:
            st.write('Class not Found')

    def run(self):
        self.display_logo()
        self.display_intro
        self.container_heading()

        audio_file_path = self.display_audio_input()

        if audio_file_path:
            y_predict = self.process_audio(audio_file_path)
            self.display_prediction(y_predict, self.bird_classifier.labels_list)


if __name__ == "__main__":
        
    model_path = 'bird_audio_classification_model.h5'
    class_file = 'classes.json'
    hf_token = os.getenv("HF_TOKEN")

    bird_classifier = BirdClassifier(model_path, class_file)
    assistant = Assistant(hf_token, "HuggingFaceH4/zephyr-7b-beta")
    app = App(bird_classifier, assistant)
    app.run()
    
    # Example usage
    # If you want to visualize audio signals:
    file_path = 'your_audio_file_path_here.wav'
    audio_viz = AudioVisualizer(file_path)
    aw, spg = audio_viz.audio_signals()

    # For audio feature extraction:
    data_dir = '/path/to/your/audio/dataset'  # Specify the path to your dataset
    audio_extractor = AudioFeatureExtractor(data_dir)
    features, labels = audio_extractor.load_data_and_extract_features()

    # Optionally, check the features and labels:
    print(f"Features shape: {features.shape}")
    print(f"Labels: {labels[:5]}")

    # For recording and saving audio:
    output_filename = "recorded_audio.wav"
    recorder = AudioRecorder(output_filename, duration=5)
    recorder.record_audio()
    recorder.terminate()

    print(f"Audio recorded and saved as {output_filename}")
