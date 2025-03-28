import os
import queue
import time
import json
import pyaudio
import streamlit as st
from vosk import Model, KaldiRecognizer

# Initialize Streamlit App
st.title("Real-time Speech Recognition with Vosk")

# Load the Vosk Model
MODEL_PATH = "model"
if not os.path.exists(MODEL_PATH):
    st.error("Please download the Vosk model and place it in the 'model' folder!")
    st.stop()

model = Model(MODEL_PATH)

# Setup Audio Stream
audio_queue = queue.Queue()


def callback(in_data, frame_count, time_info, status):
    audio_queue.put(in_data)
    return None, pyaudio.paContinue


# Initialize PyAudio
p = pyaudio.PyAudio()
stream = p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=16000,
    input=True,
    frames_per_buffer=4096,
    stream_callback=callback,
)

stream.start_stream()

# Recognizer
recognizer = KaldiRecognizer(model, 16000)

st.write("🎙 Speak something and see the transcription below:")

transcription = st.empty()

try:
    while True:
        audio_data = audio_queue.get()
        if recognizer.AcceptWaveform(audio_data):
            result = json.loads(recognizer.Result())
            text = result.get("text", "")
            transcription.write(f"**Recognized Speech:** {text}")
        time.sleep(0.1)

except KeyboardInterrupt:
    st.write("Speech recognition stopped.")
    stream.stop_stream()
    stream.close()
    p.terminate()
