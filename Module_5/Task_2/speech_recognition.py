import os
import wave
import json
import streamlit as st
from vosk import Model, KaldiRecognizer

# Initialize Streamlit App
st.title("Speech Recognition from Uploaded Audio with Vosk")

# Load the Vosk Model
MODEL_PATH = "model/vosk-model-small-en-us-0.15"
if not os.path.exists(MODEL_PATH):
    st.error("Please download the Vosk model and place it in the 'model' folder!")
    st.stop()

model = Model(MODEL_PATH)

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
