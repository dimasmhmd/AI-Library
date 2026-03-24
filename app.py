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
# 1. KONFIGURASI & STORAGE
# ==========================================
st.set_page_config(page_title="Corporate AI Library", page_icon="🏢", layout="wide")

ADMIN_PASSWORD = "admin123" 
DB_DIR = "permanent_library_db"
PDF_STORAGE_DIR = "stored_pdfs"
LOG_FILE = "logs_aktivitas.csv"

# Inisialisasi Folder
for folder in [PDF_STORAGE_DIR, DB_DIR]:
    if not os.path.exists(folder): os.makedirs(folder)

if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
else:
    st.error("⚠️ GROQ_API_KEY tidak ditemukan di Secrets!")
    st.stop()

# ==========================================
# 2. FUNGSI PENDUKUNG (AUDIO & LOGS)
# ==========================================
def save_log(query, response, pages):
    """Mencatat aktivitas chat ke CSV menggunakan Pandas."""
    new_entry = pd.DataFrame([{
        "Waktu": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Pertanyaan": query,
        "Jawaban_Singkat": response[:100] + "...",
        "Halaman_Ref": str(pages)
    }])
    new_entry.to_csv(LOG_FILE, mode='a', header=not os.path.exists(LOG_FILE), index=False)

def get_audio_html(text):
    """Generate audio dengan deteksi aksen otomatis."""
    try:
        lang = detect(text) if text else 'id'
        if lang not in ['id', 'en']: lang = 'id'
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
    if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
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
# 3. SIDEBAR: NEW CHAT & ADMIN
# ==========================================
if "messages" not in st.session_state: st.session_state.messages = []
if "is_admin" not in st.session_state: st.session_state.is_admin = False
if "vectorstore" not in st.session_state: st.session_state.vectorstore = get_vectorstore()
if "current_audio" not in st.session_state: st.session_state.current_audio = None

with st.sidebar:
    st.title("🏢 Menu Utama")
    
    # --- FITUR NEW CHAT ---
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.session_state.current_audio = None
        st.rerun()

    st.divider()

    # --- ADMIN POPOVER ---
    with st.popover("🔐 Admin Access", use_container_width=True):
        input_pass = st.text_input("Password", type="password")
        if st.button("Login"):
            if input_pass == ADMIN_PASSWORD:
                st.session_state.is_admin = True
                st.rerun()
            else: st.error("Password Salah!")
        
        if st.session_state.is_admin:
            if st.button("Logout"):
                st.session_state.is_admin = False
                st.rerun()

    # --- PANEL KONTROL ADMIN ---
    if st.session_state.is_admin:
        tab_lib, tab_rep = st.tabs(["📦 Library", "📊 Reports"])
        
        with tab_lib:
            st.subheader("Update Database")
            up_pdf = st.file_uploader("Tambah PDF", type="pdf")
            if st.button("Proses Dokumen"):
                if up_pdf:
                    with st.spinner("Mengindeks..."):
                        st.session_state.vectorstore = add_to_library(up_pdf)
                        st.success("Berhasil!")
                        st.rerun()
            
            if st.button("🗑️ Reset All Library"):
                if os.path.exists(DB_DIR):
                    shutil.rmtree(DB_DIR)
                    st.session_state.vectorstore = None
                    st.rerun()
        
        with tab_rep:
            st.subheader("Log Aktivitas")
            if os.path.exists(LOG_FILE):
                df_log = pd.read_csv(LOG_FILE)
                st.dataframe(df_log.tail(10), use_container_width=True)
                st.download_button("Ekspor CSV", df_log.to_csv(index=False), "log_ai.csv")
            else:
                st.info("Belum ada data.")

# ==========================================
# 4. HALAMAN CHAT
# ==========================================
st.title("📚 AI Corporate Hub")

# Tampilkan status library
if st.session_state.vectorstore is None:
    st.warning("⚠️ Library masih kosong. Harap hubungi Admin.")

# Menampilkan Percakapan
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if "ref" in m and st.session_state.is_admin:
            with st.expander("🔍 Sumber Referensi (Admin Only)"):
                st.markdown(m["ref"])

# Input Chat
if prompt := st.chat_input("Tanyakan sesuatu pada dokumen..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    if st.session_state.vectorstore:
        with st.chat_message("assistant"):
            with st.spinner("Mencari jawaban..."):
                results = st.session_state.vectorstore.similarity_search(prompt, k=3)
                context = "\n".join([d.page_content for d in results])
                
                llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.1)
                ans = llm.invoke(f"Konteks: {context}\nPertanyaan: {prompt}").content
                
                # Tampilkan Jawaban
                st.markdown(ans)
                
                # Olah Referensi
                pages = sorted(list(set([d.metadata.get('page', 0) + 1 for d in results])))
                ref_text = f"**Halaman:** {', '.join(map(str, pages))}\n\n"
                for doc in results:
                    ref_text += f"- *Hal {doc.metadata.get('page',0)+1}:* {doc.page_content[:150]}...\n"
                
                # Tampilkan Referensi jika Admin
                if st.session_state.is_admin:
                    with st.expander("🔍 Sumber Referensi (Admin Only)"):
                        st.markdown(ref_text)
                
                # Simpan Log & History
                save_log(prompt, ans, pages)
                st.session_state.messages.append({"role": "assistant", "content": ans, "ref": ref_text})
                
                # Audio Player
                st.session_state.current_audio = get_audio_html(ans)
                st.rerun()
    else:
        st.error("Library kosong.")

# Audio Player Persisten di bawah
if st.session_state.current_audio:
    st.divider()
    st.markdown(st.session_state.current_audio, unsafe_allow_html=True)
