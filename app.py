import streamlit as st
import os
import pandas as pd
import glob
import base64
import shutil
from datetime import datetime
from gtts import gTTS
from langdetect import detect
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ==========================================
# 1. KONFIGURASI & PENYIMPANAN
# ==========================================
st.set_page_config(page_title="Secure Corporate Library", page_icon="🛡️", layout="wide")

ADMIN_PASSWORD = "admin123" 
DB_DIR = "permanent_library_db"
PDF_STORAGE_DIR = "stored_pdfs"
LOG_FILE = "logs_evaluasi.csv"

for folder in [PDF_STORAGE_DIR, DB_DIR]:
    if not os.path.exists(folder): os.makedirs(folder)

if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
else:
    st.error("⚠️ GROQ_API_KEY tidak ditemukan!")
    st.stop()

# ==========================================
# 2. FUNGSI CORE
# ==========================================
def save_log(query, response, pages):
    """Mencatat aktivitas pertanyaan ke CSV."""
    df = pd.DataFrame([{
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "User_Query": query,
        "AI_Response_Snippet": response[:100] + "...",
        "Source_Pages": str(pages)
    }])
    df.to_csv(LOG_FILE, mode='a', header=not os.path.exists(LOG_FILE), index=False)

def get_audio_html(text):
    try:
        lang = detect(text) if text else 'id'
        if lang not in ['id', 'en', 'ja', 'ko']: lang = 'id'
        tts = gTTS(text=text, lang=lang)
        filename = "temp_voice.mp3"
        tts.save(filename)
        with open(filename, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        if os.path.exists(filename): os.remove(filename)
        return f'<audio controls style="width: 100%;"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
    except: return ""

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_vectorstore():
    if os.path.exists(DB_DIR) and glob.glob(f"{DB_DIR}/*"):
        return Chroma(persist_directory=DB_DIR, embedding_function=load_embeddings(), collection_name="admin_lib")
    return None

def add_to_library(uploaded_file):
    fname = uploaded_file.name
    save_path = os.path.join(PDF_STORAGE_DIR, fname)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    loader = PyPDFLoader(save_path)
    pages = loader.load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(pages)
    return Chroma.from_documents(chunks, load_embeddings(), persist_directory=DB_DIR, collection_name="admin_lib")

# ==========================================
# 3. SIDEBAR & ADMIN PANEL
# ==========================================
if "messages" not in st.session_state: st.session_state.messages = []
if "is_admin" not in st.session_state: st.session_state.is_admin = False
if "vectorstore" not in st.session_state: st.session_state.vectorstore = get_vectorstore()

with st.sidebar:
    st.title("🏢 Navigation")
    
    with st.popover("🔐 Login Admin", use_container_width=True):
        input_pass = st.text_input("Password", type="password")
        if st.button("Submit"):
            if input_pass == ADMIN_PASSWORD:
                st.session_state.is_admin = True
                st.rerun()
            else: st.error("Salah!")
        
        if st.session_state.is_admin:
            if st.button("Logout"):
                st.session_state.is_admin = False
                st.rerun()

    st.divider()

    if st.session_state.is_admin:
        tab1, tab2 = st.tabs(["📦 Library", "📊 Reports"])
        
        with tab1:
            st.subheader("Manage Files")
            uploaded_pdf = st.file_uploader("Upload PDF", type="pdf")
            if st.button("Add PDF"):
                if uploaded_pdf:
                    st.session_state.vectorstore = add_to_library(uploaded_pdf)
                    st.success("Added!")
                    st.rerun()
        
        with tab2:
            st.subheader("Activity Logs")
            if os.path.exists(LOG_FILE):
                log_df = pd.read_csv(LOG_FILE)
                st.dataframe(log_df.tail(10), use_container_width=True)
                st.download_button("Download Full Report", log_df.to_csv(index=False), "report.csv", "text/csv")
            else:
                st.info("No logs yet.")
    else:
        st.info("User Mode: Ask questions below.")

# ==========================================
# 4. CHAT INTERFACE
# ==========================================
st.title("🛡️ Secure Knowledge Hub")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if "ref" in m and st.session_state.is_admin:
            with st.expander("🔍 References (Admin Only)"): st.markdown(m["ref"])

if prompt := st.chat_input("Ask something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    if st.session_state.vectorstore:
        with st.chat_message("assistant"):
            results = st.session_state.vectorstore.similarity_search(prompt, k=3)
            context = "\n".join([d.page_content for d in results])
            ans = ChatGroq(model_name="llama-3.1-8b-instant").invoke(f"Context: {context}\nQuestion: {prompt}").content
            
            st.markdown(ans)
            
            # Log & Admin Ref
            pages = sorted(list(set([d.metadata.get('page', 0) + 1 for d in results])))
            save_log(prompt, ans, pages)
            
            ref_text = f"**Pages:** {', '.join(map(str, pages))}\n\n"
            for d in results: ref_text += f"- *Hal {d.metadata.get('page',0)+1}:* {d.page_content[:150]}...\n"
            
            if st.session_state.is_admin:
                with st.expander("🔍 References (Admin Only)"): st.markdown(ref_text)
            
            st.session_state.messages.append({"role": "assistant", "content": ans, "ref": ref_text})
            
            # Persistent Audio Player
            audio_html = get_audio_html(ans)
            if audio_html: st.markdown(audio_html, unsafe_allow_html=True)
    else:
        st.info("Library is empty.")
