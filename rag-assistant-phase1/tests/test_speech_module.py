def test_speech_to_text(monkeypatch):
    # Mock the recognizer to return a fixed string
    class DummyRecognizer:
        def listen(self, source, timeout=5, phrase_time_limit=10):
            return "dummy audio"
        def recognize_google(self, audio):
            return "Test speech input"
    class DummyMicrophone:
        def __enter__(self):
            return "dummy source"
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
    monkeypatch.setattr(speech_module.sr, 'Recognizer', lambda: DummyRecognizer())
    monkeypatch.setattr(speech_module.sr, 'Microphone', DummyMicrophone)
    result = speech_module.speech_to_text()
    assert result == "Test speech input"
