import os
from dotenv import load_dotenv
import json
import streamlit as st
from huggingface_hub import InferenceApi, login, InferenceClient

# Get the Hugging Face token from environment variables
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token is None:
    raise ValueError("Hugging Face token not found. Please set the HF_TOKEN environment variable.")
# Authenticate with Hugging Face
login(hf_token)

# Model information and links
model_links = {
    "Zephyr-7B": "HuggingFaceH4/zephyr-7b-beta"
}
model_info = {
    "Zephyr-7B": {
        'description': """Zephyr 7B is a Huggingface model, fine-tuned for helpful and instructive interactions.""",
        'logo': 'https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha/resolve/main/thumbnail.png'
    }
}

# Inference API Initialization
client = InferenceClient('HuggingFaceH4/zephyr-7b-beta')

# Reset conversation button
def reset_conversation():
    return [
        {"role": "system", "content": "You are a knowledgeable and empathetic ornithologist assistant providing accurate and relevant information based on user input."}
    ]

# Initialize conversation and chat history
messages = reset_conversation()

# Display chat history
for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def respond(message, history, max_tokens, temperature, top_p):
    # Prepare the list of messages for the chat completion
    messages = [{"role": "system", "content": history[0]["content"]}]

    for val in history:
        if val["role"] == "user":
            messages.append({"role": "user", "content": val["content"]})
        elif val["role"] == "assistant":
            messages.append({"role": "assistant", "content": val["content"]})

    messages.append({"role": "user", "content": message})

    # Generate response
    response = ""
    response_container = st.empty()  # Placeholder to update the response text dynamically

    for message in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        token = message.choices[0].delta.content
        response += token
        # response_container.text(response)  # Stream the response

    return response
