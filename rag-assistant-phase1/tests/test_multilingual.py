from modules import multilingual

def test_detect_language():
    assert multilingual.detect_language("Hello, how are you?") == 'en'
    assert multilingual.detect_language("नमस्ते, आप कैसे हैं?") == 'hi'

def test_translate_to_english():
    hindi_text = "नमस्ते, आप कैसे हैं?"
    english = multilingual.translate_to_english(hindi_text, src_lang='hi')
    assert "how are you" in english.lower() or "hello" in english.lower()

def test_translate_from_english():
    english_text = "How are you?"
    hindi = multilingual.translate_from_english(english_text, dest_lang='hi')
    assert "कैसे" in hindi or "आप" in hindi
