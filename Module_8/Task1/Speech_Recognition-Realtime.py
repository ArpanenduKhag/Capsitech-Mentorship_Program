import os
import json
import streamlit as st
from vosk import Model, KaldiRecognizer
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import numpy as np

# Download model if not present
MODEL_PATH = "model"
MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"


def download_model():
    import urllib.request
    import zipfile

    st.write("Downloading model...")
    urllib.request.urlretrieve(MODEL_URL, "model.zip")
    with zipfile.ZipFile("model.zip", "r") as zip_ref:
        zip_ref.extractall(".")
    os.rename("vosk-model-small-en-us-0.15", MODEL_PATH)
    os.remove("model.zip")


if not os.path.exists(MODEL_PATH):
    download_model()

# Load Vosk model
model = Model(MODEL_PATH)

# Title
st.title("🎤 Real-time Speech Recognition with Vosk & Streamlit")

# Session state to hold transcript
if "transcript" not in st.session_state:
    st.session_state.transcript = ""


# Audio Processor
class AudioProcessor:
    def __init__(self) -> None:
        self.rec = KaldiRecognizer(model, 16000)
        self.rec.SetWords(True)

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray()
        audio = audio.mean(axis=0)  # Convert stereo to mono
        data = audio.astype(np.int16).tobytes()

        if self.rec.AcceptWaveform(data):
            result = json.loads(self.rec.Result())
            text = result.get("text", "")
            if text:
                st.session_state.transcript += " " + text

        return frame


# Start WebRTC
ctx = webrtc_streamer(
    key="example",
    mode=WebRtcMode.SENDRECV,
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
)

# Display result
st.subheader("📝 Recognized Text:")
st.write(st.session_state.transcript)
