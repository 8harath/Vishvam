# 🗣️ Phase 2: Speech I/O & Multilingual Support

## ✅ Phase 2 Completion Summary

This phase extends the RAG Assistant with robust voice interaction and multilingual capabilities, building on the core text-based RAG pipeline from Phase 1.

---

## 🎯 **Objectives**

- Enable users to interact with the assistant using both text and voice queries.
- Provide spoken responses using text-to-speech.
- Automatically detect the language of user queries.
- Translate non-English queries to English for processing.
- Translate answers back to the user's language for output.
- Ensure all new features are modular and do not disrupt Phase 1 functionality.

---

## 🛠️ **Technologies & Libraries Used**

| Feature                | Library/Tool         | Purpose                                      |
|------------------------|---------------------|----------------------------------------------|
| Speech-to-Text         | `SpeechRecognition` | Convert spoken queries to text               |
| Text-to-Speech         | `gTTS`              | Synthesize spoken answers from text          |
| Language Detection     | `langdetect`        | Detect the language of user input            |
| Translation            | `googletrans`       | Translate queries and answers (multilingual) |
| Core RAG Pipeline      | (Phase 1)           | PDF parsing, chunking, embedding, retrieval  |

---

## 🧩 **Key Implementation Details**

- **Modular Design:**  
  - Added `modules/speech_module.py` for all voice input/output.
  - Added `modules/multilingual.py` for language detection and translation.
- **Main Application Integration:**  
  - After PDF processing, the CLI enters an interactive Q&A mode.
  - Users can choose between text or voice input for each query.
  - The system detects the query language, translates as needed, retrieves answers, and translates responses back.
  - Optionally, answers can be spoken aloud in the user's language.
- **Backward Compatibility:**  
  - All Phase 1 features (text-based RAG, PDF parsing, chunking, semantic search) remain fully functional.
  - Voice and multilingual features are opt-in and do not interfere with text-only workflows.

---

## 🚀 **How to Use (Phase 2 Features)**

1. **Run the Application:**
   ```sh
   python rag-assistant-phase1/main.py rag-assistant-phase1/sample_data/product_manual.pdf
   ```

2. **After PDF processing and demo search, you will see:**
   ```
   🗣️  Entering interactive Q&A mode. Type 'exit' to quit.
   Choose input mode: [1] Text [2] Voice :
   ```

3. **Interact:**
   - Type `1` for text input, or `2` for voice input (speak your query).
   - You can use any supported language for your query.
   - The assistant will print and (optionally) speak the answer in your language.
   - Type `exit` to quit.

---

## 🧪 **Testing**

- **Unit tests** for speech and multilingual modules are provided (see `tests/`).
- Manual testing covers:
  - English and non-English queries (e.g., Hindi).
  - Both text and voice input/output.
  - Fallback to text-only mode.

---

## 📦 **Dependencies**

Make sure to install all required packages:
```sh
pip install -r requirements.txt
```
Key packages for Phase 2:
- `SpeechRecognition`
- `gTTS`
- `langdetect`
- `googletrans==4.0.0rc1`
- (and all Phase 1 dependencies)

---

## 📝 **Summary**

Phase 2 successfully adds voice-driven and multilingual interaction to the RAG Assistant, making it accessible and user-friendly for a diverse, global audience. All new features are modular and backward-compatible.