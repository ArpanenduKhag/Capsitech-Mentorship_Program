import os
import wave
import json
import streamlit as st
from vosk import Model, KaldiRecognizer
import urllib.request
import zipfile


# Set model folder and download URL for a small Vosk model
MODEL_PATH = "model"
MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"


def download_and_extract_model(model_url, model_path):
    zip_path = "model.zip"
    st.write("Downloading Vosk model... (this may take a few minutes)")
    urllib.request.urlretrieve(model_url, zip_path)
    st.write("Extracting model...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(".")
    os.remove(zip_path)
    # The extracted folder is likely named "vosk-model-small-en-us-0.15"
    extracted_folder = "vosk-model-small-en-us-0.15"
    if os.path.exists(extracted_folder):
        os.rename(extracted_folder, model_path)


# Check if model folder exists and contains files
if not os.path.exists(MODEL_PATH) or not os.listdir(MODEL_PATH):
    download_and_extract_model(MODEL_URL, MODEL_PATH)

# Try to load the model
try:
    model = Model(MODEL_PATH)
    st.write("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model: {e}")

# Initialize Streamlit App
st.title("Speech Recognition from Uploaded Audio with Vosk")

# File uploader for WAV files
uploaded_file = st.file_uploader(
    "Upload a WAV file (mono, 16-bit, 16000 Hz)", type=["wav"]
)
if uploaded_file is not None:
    try:
        wf = wave.open(uploaded_file, "rb")
    except Exception as e:
        st.error(f"Error reading audio file: {e}")
    else:
        # Check that the audio file has the correct format
        if (
            wf.getnchannels() != 1
            or wf.getsampwidth() != 2
            or wf.getframerate() != 16000
        ):
            st.error(
                "Audio file must be WAV format with 1 channel, 16-bit, and 16000 Hz sample rate."
            )
        else:
            # Initialize recognizer
            recognizer = KaldiRecognizer(model, wf.getframerate())
            transcription = ""

            # Process the audio in chunks
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    transcription += result.get("text", "") + " "
            # Get any final partial result
            final_result = json.loads(recognizer.FinalResult())
            transcription += final_result.get("text", "")

            st.write("**Recognized Speech:**")
            st.write(transcription)
