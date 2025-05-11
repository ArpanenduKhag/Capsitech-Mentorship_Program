import gradio as gr
import speech_recognition as sr

recognizer = sr.Recognizer()

def transcribe_microphone():
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source, phrase_time_limit=5)

    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Sorry, I couldn't understand that."
    except sr.RequestError as e:
        return f"API error: {e}"

iface = gr.Interface(fn=transcribe_microphone,
                     inputs=[],
                     outputs="text",
                     live=True,
                     title="🎤 Real-Time Speech Recognition",
                     description="Speak into the microphone and get the transcription below.")

iface.launch(share=True)
