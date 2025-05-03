import streamlit as st
import speech_recognition as sr

st.title("🎙️ Speech Recognition App")
st.write("Upload an audio file or use your microphone to transcribe speech to text.")

recognizer = sr.Recognizer()

# Upload an audio file
audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3"])

if audio_file is not None:
    st.audio(audio_file, format="audio/wav")

    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            st.success("Transcription:")
            st.write(text)
        except sr.UnknownValueError:
            st.error("Could not understand the audio.")
        except sr.RequestError as e:
            st.error(f"API error: {e}")
