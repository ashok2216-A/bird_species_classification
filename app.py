import os
import warnings
import librosa
# import soundfile as sf
import streamlit as st
import tempfile
import json
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
    
    class_file = open('classes.txt', 'r').read()
    # labels_list = json.loads(class_file)
    label_list={66: 'Ornate Tinamou_sound',
 59: 'Moluccan Megapode_sound',
 76: 'Red-throated Piping Guan_sound',
 51: 'Little Chachalaca_sound',
 102: 'Undulated Tinamou_sound',
 6: 'Baudo Guan_sound',
 110: 'White-crested Guan_sound',
 33: 'Dusky Megapode_sound',
 12: 'Black-capped Tinamou_sound',
 8: 'Berlepschs Tinamou_sound',
 50: 'Lesser Rhea_sound',
 83: 'Scaled Chachalaca_sound',
 32: 'Darwins Nothura_sound',
 58: 'Micronesian Megapode_sound',
 113: 'Yellow-legged Tinamou_sound',
 84: 'Slaty-breasted Tinamou_sound',
 7: 'Bearded Guan_sound',
 96: 'Tataupa Tinamou_sound',
 45: 'Grey-legged Tinamou_sound',
 86: 'Solitary Tinamou_sound',
 13: 'Black-fronted Piping Guan_sound',
 14: 'Blue-throated Piping Guan_sound',
 97: 'Tawny-breasted Tinamou_sound',
 70: 'Plain Chachalaca_sound',
 71: 'Puna Tinamou_sound',
 82: 'Rusty-margined Guan_sound',
 23: 'Chestnut-winged Chachalaca_sound',
 88: 'Southern Brown Kiwi_sound',
 104: 'Variegated Tinamou_sound',
 28: 'Colombian Chachalaca_sound',
 111: 'White-throated Tinamou_sound',
 74: 'Red-faced Guan_sound',
 63: 'Northern Cassowary_sound',
 57: 'Melanesian Megapode_sound',
 101: 'Trinidad Piping Guan_sound',
 49: 'Lesser Nothura_sound',
 80: 'Rufous-vented Chachalaca_sound',
 112: 'White-winged Guan_sound',
 77: 'Red-winged Tinamou_sound',
 79: 'Rufous-headed Chachalaca_sound',
 20: 'Chaco Chachalaca_sound',
 37: 'East Brazilian Chachalaca_sound',
 64: 'Okarito Kiwi_sound',
 99: 'Thicket Tinamou_sound',
 69: 'Philippine Megapode_sound',
 103: 'Vanuatu Megapode_sound',
 39: 'Emu_sound',
 100: 'Tongan Megapode_sound',
 89: 'Southern Cassowary_sound',
 19: 'Cauca Guan_sound',
 94: 'Taczanowskis Tinamou_sound',
 40: 'Great Spotted Kiwi_sound',
 109: 'White-browed Guan_sound',
 56: 'Marail Guan_sound',
 60: 'New Guinea Scrubfowl_sound',
 4: 'Barred Tinamou_sound',
 30: 'Crested Guan_sound',
 106: 'West Mexican Chachalaca_sound',
 47: 'Hooded Tinamou_sound',
 48: 'Huayco Tinamou_sound',
 98: 'Tepui Tinamou_sound',
 44: 'Grey-headed Chachalaca_sound',
 78: 'Rufous-bellied Chachalaca_sound',
 15: 'Brazilian Tinamou_sound',
 61: 'Nicobar Megapode_sound',
 22: 'Chestnut-headed Chachalaca_sound',
 24: 'Chilean Tinamou_sound',
 26: 'Cinereous Tinamou_sound',
 38: 'Elegant Crested Tinamou_sound',
 27: 'Collared Brushturkey_sound',
 10: 'Black Tinamou_sound',
 65: 'Orange-footed Scrubfowl_sound',
 0: 'Andean Guan_sound',
 53: 'Little Tinamou_sound',
 3: 'Band-tailed Guan_sound',
 54: 'Maleo_sound',
 29: 'Common Ostrich_sound',
 107: 'White-bellied Chachalaca_sound',
 31: 'Curve-billed Tinamou_sound',
 43: 'Grey Tinamou_sound',
 5: 'Bartletts Tinamou_sound',
 95: 'Tanimbar Megapode_sound',
 87: 'Somali Ostrich_sound',
 21: 'Chestnut-bellied Guan_sound',
 16: 'Brown Tinamou_sound',
 35: 'Dwarf Cassowary_sound',
 11: 'Black-billed Brushturkey_sound',
 68: 'Patagonian Tinamou_sound',
 17: 'Brushland Tinamou_sound',
 18: 'Buff-browed Chachalaca_sound',
 42: 'Greater Rhea_sound',
 93: 'Sula Megapode_sound',
 91: 'Spixs Guan_sound',
 81: 'Rusty Tinamou_sound',
 75: 'Red-legged Tinamou_sound',
 67: 'Pale-browed Tinamou_sound',
 2: 'Australian Brushturkey_sound',
 1: 'Andean Tinamou_sound',
 55: 'Malleefowl_sound',
 36: 'Dwarf Tinamou_sound',
 46: 'Highland Tinamou_sound',
 85: 'Small-billed Tinamou_sound',
 25: 'Choco Tinamou_sound',
 9: 'Biak Scrubfowl_sound',
 34: 'Dusky-legged Guan_sound',
 62: 'North Island Brown Kiwi_sound',
 73: 'Red-billed Brushturkey_sound',
 105: 'Wattled Brushturkey_sound',
 108: 'White-bellied Nothura_sound',
 92: 'Spotted Nothura_sound',
 52: 'Little Spotted Kiwi_sound',
 72: 'Quebracho Crested Tinamou_sound',
 41: 'Great Tinamou_sound',
 90: 'Speckled Chachalaca_sound'
 }
    
    # Predict the emotion
    y_predict = predict_emotion(file_path, model)
    
    # Display predicted class
    if y_predict[0] in labels_list.keys():
        st.subheader(f'Predicted Class: :rainbow[{labels_list[y_predict[0]]}]')
    else:
        st.write('Class not Found')
else:
    st.markdown('File not Found!')
