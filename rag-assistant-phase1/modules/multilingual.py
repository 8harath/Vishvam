from langdetect import detect
from googletrans import Translator

translator = Translator()

def detect_language(text):
    """
    Detect the language of the input text. Returns ISO 639-1 code (e.g., 'en', 'hi').
    """
    try:
        return detect(text)
    except Exception:
        return 'en'  # Default to English if detection fails

def translate_to_english(text, src_lang=None):
    """
    Translate text from src_lang (or auto) to English.
    """
    try:
        if src_lang:
            result = translator.translate(text, src=src_lang, dest='en')
        else:
            result = translator.translate(text, dest='en')
        return result.text
    except Exception:
        return text

def translate_from_english(text, dest_lang):
    """
    Translate text from English to dest_lang.
    """
    try:
        result = translator.translate(text, src='en', dest=dest_lang)
        return result.text
    except Exception:
        return text
