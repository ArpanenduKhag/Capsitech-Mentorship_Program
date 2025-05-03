import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import av
import numpy as np
import json
import os
from vosk import Model, KaldiRecognizer

# Load Vosk model (make sure path is correct)
if not os.path.exists("model"):
    st.error("Please download a model from https://alphacephei.com/vosk/models and unpack it as 'model' in the current folder.")
    st.stop()

model = Model("model")

# Title
st.title("🎙️ Real-Time Speech Recognition")
st.write("Speak into your microphone and see the transcribed text below:")

# Text display placeholder
output_text = st.empty()

# Define AudioProcessor
class AudioProcessor(AudioProcessorBase):
    def __init__(self) -> None:
        self.rec = KaldiRecognizer(model, 16000)

    def recv(self, frame: av.AudioFrame) -> None:
        audio = frame.to_ndarray()
        audio_bytes = audio.flatten().astype(np.int16).tobytes()

        if self.rec.AcceptWaveform(audio_bytes):
            result = json.loads(self.rec.Result())
            text = result.get("text", "")
