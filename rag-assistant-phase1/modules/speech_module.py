import speech_recognition as sr
from gtts import gTTS
import os
import tempfile

def speech_to_text(timeout=5, phrase_time_limit=10):
    """
    Capture audio from the microphone and convert it to text using Google Speech Recognition.
    Returns the recognized text, or raises an exception on failure.
    """
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("[Speech] Please speak now...")
        audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
    try:
        text = recognizer.recognize_google(audio)
        print(f"[Speech] Recognized: {text}")
        return text
    except sr.UnknownValueError:
        print("[Speech] Could not understand audio.")
        return ""
    except sr.RequestError as e:
        print(f"[Speech] Could not request results; {e}")
        return ""

def text_to_speech(text, lang='en'):
    """
    Convert text to speech and play it using the system's default player.
    """
    tts = gTTS(text=text, lang=lang)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
        tts.save(fp.name)
        audio_path = fp.name
    print(f"[Speech] Playing audio response...")
    # Windows
    if os.name == 'nt':
        os.system(f'start {audio_path}')
    # macOS
    elif os.uname().sysname == 'Darwin':
        os.system(f'afplay {audio_path}')
    # Linux
    else:
        os.system(f'mpg123 {audio_path}')
