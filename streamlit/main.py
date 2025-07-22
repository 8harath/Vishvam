# ——— Patch 1: Stop Streamlit watcher hitting torch._classes.__path__ ———
import torch
class _DummyPath:
    def __init__(self):
        self._path = []
    def __getattr__(self, name):
        return []
torch._classes.__path__ = _DummyPath()

# ——— Patch 2: Make SentenceTransformer.to() fall back to to_empty() on meta modules ———
import sentence_transformers as _st
_BaseST = _st.SentenceTransformer
class SentenceTransformer(_BaseST):
    def to(self, *args, **kwargs):
        try:
            return super().to(*args, **kwargs)
        except NotImplementedError:
            return super().to_empty(*args, **kwargs)

# ——— Standard imports ———
import streamlit as st
import streamlit.components.v1 as components
import PyPDF2
import numpy as np
from typing import List
from langdetect import detect, detect_langs
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from gtts import gTTS
import speech_recognition as sr
import tempfile, base64
import requests, time

# ——— Configuration ———
GENAI_API_KEY = "AIzaSyA5xtoT9HAjH-wsa7OHFXlBjRRcXwCFBMg"
DID_API_KEY = "a3ZpcGFyeWExQGdtYWlsLmNvbQ:CcPcUZ9Lylz6kWnA0QJMj"
AVATAR_IMAGE_URL = "https://raw.githubusercontent.com/de-id/live-streaming-demo/main/alex_v2_idle_image.png"

# ——— RAGSingleLanguage class ———
class RAGSingleLanguage:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model    = genai.GenerativeModel('gemini-1.5-flash')
        self.embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.chunks: List[str] = []
        self.embeddings: np.ndarray = np.array([])
        self.language: str = 'en'

    def detect_languages(self, text: str) -> List[str]:
        seg_size = 1000
        probs = {}
        for i in range(0, len(text), seg_size):
            seg = text[i:i+seg_size]
            try:
                for lang in detect_langs(seg):
                    probs[lang.lang] = max(probs.get(lang.lang, 0.0), lang.prob)
            except:
                continue
        langs = [l for l,p in probs.items() if p >= 0.2]
        return langs or ['en']

    def translate(self, text: str, tgt: str) -> str:
        try:
            src = detect(text)
        except:
            src = 'en'
        if src.lower() == tgt.lower():
            return text
        prompt = f"Translate to {tgt.upper()}:\n\n{text}"
        try:
            return self.model.generate_content(prompt).text.strip()
        except:
            return text

    def process_document(self, pdf_file, chunk_size: int = 500) -> List[str]:
        reader = PyPDF2.PdfReader(pdf_file)
        pages = [p.extract_text() or "" for p in reader.pages]
        full = " ".join(pages).split()
        self.chunks = [
            " ".join(full[i:i+chunk_size])
            for i in range(0, len(full), chunk_size)
        ]
        self.embeddings = self.embedder.encode(
            self.chunks,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return self.detect_languages(" ".join(pages))

    def set_language(self, lang: str):
        self.language = lang

    def answer_question(self, question: str, top_k: int = 5) -> str:
        q_en = self.translate(question, 'en')
        q_emb = self.embedder.encode([
            q_en
        ], convert_to_numpy=True, normalize_embeddings=True)
        doc_embeds = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        sims = cosine_similarity(q_emb, doc_embeds)[0]
        top_indices = np.argsort(sims)[::-1][:top_k]
        contexts = [f"[Score: {sims[i]:.2f}]\n{self.chunks[i]}" for i in top_indices]
        ctx = "\n\n".join(contexts)

        prompt = (
            "Answer the following question using only the provided context. "
            "Be accurate and detailed. If the answer is not present, say: "
            "'I apologize, but I cannot find this information in the documentation. "
            "Please contact customer support for accurate assistance on this matter.'\n\n"
            f"Context:\n{ctx}\n\nQuestion: {q_en}"
        )

        try:
            out = self.model.generate_content(prompt).text.strip()
        except Exception as e:
            return f"Error: {e}"
        return self.translate(out, self.language)

# ——— Voice Input ———
def recognize_voice(lang_code='en-IN') -> str:
    r = sr.Recognizer()
    with sr.Microphone() as src:
        st.info("🎤 Adjusting for ambient noise…")
        r.adjust_for_ambient_noise(src, duration=1)
        st.info("Listening…")
        try:
            audio = r.listen(src, timeout=10, phrase_time_limit=10)
        except sr.WaitTimeoutError:
            st.warning("⏰ No speech detected.")
            return ""
    try:
        return r.recognize_google(audio, language=lang_code)
    except sr.UnknownValueError:
        st.error("❗ Could not understand audio.")
    except sr.RequestError as e:
        st.error(f"🚫 Speech API error: {e}")
    return ""

# ——— D-ID Avatar Generator ———
def generate_did_avatar_video(answer_text: str, image_url: str) -> str:
    url = "https://api.d-id.com/talks"
    headers = {
        "Authorization": f"Basic {base64.b64encode(DID_API_KEY.encode()).decode()}",
        "Content-Type": "application/json"
    }
    payload = {
        "source_url": image_url,
        "script": {
            "type": "text",
            "input": answer_text,
            "provider": {
                "type": "microsoft",
                "voice_id": "en-US-GuyNeural",
                "voice_config": {"style": "Cheerful"}
            }
        },
        "config": {"stitch": True}
    }
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code not in [200, 201]:
        st.error(f"❌ Avatar video request failed: {response.text}")
        return ""
    talk_id = response.json().get("id")
    if not talk_id:
        st.error("❌ Talk ID not found in response.")
        return ""
    for _ in range(30):
        time.sleep(2)
        check = requests.get(f"https://api.d-id.com/talks/{talk_id}", headers=headers)
        if check.status_code == 200:
            data = check.json()
            if data.get("status") == "done":
                return data.get("result_url")
    st.warning("⚠️ Avatar video is still processing or timed out.")
    return ""

# ——— Main App ———
def main():
    st.set_page_config(page_title="Voice‑Viz RAG", page_icon="🔊")
    st.title("🔊 AI Helpdesk")

    if 'rag' not in st.session_state:
        st.session_state.rag       = RAGSingleLanguage(GENAI_API_KEY)
        st.session_state.file_done = False
        st.session_state.langs     = []
        st.session_state.lang      = None
        st.session_state.voice_q   = ""

    st.sidebar.header("How to use")
    st.sidebar.markdown("""
    1. Upload PDF  
    2. Select language  
    3. Type or speak your question  
    4. Read or listen to the answer  
    """)

    uploaded = st.file_uploader("Upload your PDF manual", type="pdf")
    if uploaded and not st.session_state.file_done:
        with st.spinner("Processing PDF…"):
            try:
                st.session_state.langs = st.session_state.rag.process_document(uploaded)
                st.session_state.file_done = True
                st.success("✅ Document processed!")
            except Exception as e:
                st.error(f"Failed: {e}")
                return

    if st.session_state.file_done:
        sel = st.selectbox("Select language:", [l.upper() for l in st.session_state.langs])
        if sel:
            st.session_state.lang = sel.lower()
            st.session_state.rag.set_language(st.session_state.lang)

        st.markdown("**Type your question**")
        typed = st.text_input(f"Ask in {sel}:")
        st.markdown("**Or use voice input**")
        if st.button("🎙️ Speak Your Question"):
            code = st.session_state.lang + "-IN" if st.session_state.lang == "en" else st.session_state.lang
            recd = recognize_voice(code)
            if recd:
                st.session_state.voice_q = recd
                st.success(f"🎤 You said: {recd}")
            else:
                st.warning("No speech recognized.")

        question = typed or st.session_state.voice_q
        if st.button("Get Answer") and question:
            st.markdown(f"🔍 Question: `{question}`")
            with st.spinner("Thinking…"):
                answer = st.session_state.rag.answer_question(question)
            st.markdown(f"**Answer ({sel}):** {answer}")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                gTTS(text=answer, lang=st.session_state.lang).save(fp.name)
                mp3_bytes = open(fp.name, "rb").read()
            b64 = base64.b64encode(mp3_bytes).decode()

            html = f"""
            <audio id='player' controls>
              <source src='data:audio/mp3;base64,{b64}' type='audio/mp3'/>
            </audio>
            <canvas id='canvas' width='300' height='100'></canvas>
            <script>
              const audio = document.getElementById('player');
              const canvas = document.getElementById('canvas');
              const ctx = canvas.getContext('2d');
              const audioCtx = new (window.AudioContext||window.webkitAudioContext)();
              const source = audioCtx.createMediaElementSource(audio);
              const analyser = audioCtx.createAnalyser();
              analyser.fftSize = 256;
              source.connect(analyser);
              analyser.connect(audioCtx.destination);
              const data = new Uint8Array(analyser.frequencyBinCount);

              function drawLine() {{
                requestAnimationFrame(drawLine);
                analyser.getByteTimeDomainData(data);
                let sum = 0;
                for (let i=0; i<data.length; i++) {{
                  const v = data[i] - 128;
                  sum += v*v;
                }}
                const rms = Math.sqrt(sum/data.length);
                const maxLen = canvas.width / 2 * (rms/128);
                const y = canvas.height / 2;

                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.beginPath();
                ctx.moveTo((canvas.width / 2) - maxLen, y);
                ctx.lineTo((canvas.width / 2) + maxLen, y);
                ctx.lineWidth = 4;
                ctx.strokeStyle = '#4CAF50';
                ctx.stroke();
              }}

              audio.onplay = () => {{
                audioCtx.resume().then(() => drawLine());
              }};
            </script>
            """
            components.html(html, height=150)
            st.session_state.voice_q = ""

            st.markdown("### 🧑‍💼 Speaking AI Avatar")
            with st.spinner("Generating avatar video…"):
                video_url = generate_did_avatar_video(answer, AVATAR_IMAGE_URL)
                if video_url:
                    st.video(video_url)
                else:
                    st.error("Failed to load avatar video.")

if __name__ == "__main__":
    main()
