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
# 1. KONFIGURASI & KONSTANTA
# ==========================================
st.set_page_config(page_title="Corporate AI Library", page_icon="📚", layout="wide")

ADMIN_PASSWORD = "admin123" 
DB_DIR = "permanent_library_db"

if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
else:
    st.error("⚠️ GROQ_API_KEY tidak ditemukan di Secrets!")
    st.stop()

# ==========================================
# 2. FUNGSI CORE
# ==========================================
def get_audio_html(text):
    try:
        detected_lang = detect(text) if text else 'id'
        if detected_lang not in ['id', 'en', 'ja', 'ko']: detected_lang = 'id'
        tts = gTTS(text=text, lang=detected_lang)
        filename = "temp_voice.mp3"
        tts.save(filename)
        with open(filename, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        if os.path.exists(filename): os.remove(filename)
        return f"""
            <div style="background-color: #f0f2f6; padding: 10px; border-radius: 10px; border-left: 5px solid #ff4b4b; margin-top: 10px;">
                <p style="margin-bottom: 5px; font-weight: bold;">🔊 Audio ({detected_lang.upper()}):</p>
                <audio controls style="width: 100%;"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>
            </div>
            """
    except: return ""

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_vectorstore():
    if os.path.exists(DB_DIR):
        return Chroma(persist_directory=DB_DIR, embedding_function=load_embeddings(), collection_name="admin_lib")
    return None

def get_document_list(vectorstore):
    """Mengambil daftar nama file unik dari metadata ChromaDB."""
    if vectorstore is None: return []
    try:
        data = vectorstore.get()
        metadatas = data.get('metadatas', [])
        # Ambil 'source' dari metadata dan bersihkan path-nya
        sources = set([os.path.basename(m.get('source', 'Unknown')) for m in metadatas])
        return sorted(list(sources))
    except: return []

def add_to_library(uploaded_file):
    fname = uploaded_file.name
    with open(fname, "wb") as f:
        f.write(uploaded_file.getbuffer())
    loader = PyPDFLoader(fname)
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(pages)
    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=load_embeddings(),
        persist_directory=DB_DIR, collection_name="admin_lib"
    )
    if os.path.exists(fname): os.remove(fname) # Hapus file temp setelah di-embed
    return vectorstore

# ==========================================
# 3. SIDEBAR & ADMIN FEATURES
# ==========================================
if "messages" not in st.session_state: st.session_state.messages = []
if "current_audio" not in st.session_state: st.session_state.current_audio = None
if "vectorstore" not in st.session_state: st.session_state.vectorstore = get_vectorstore()

with st.sidebar:
    st.header("🔐 Admin Panel")
    input_pass = st.text_input("Password Admin", type="password")
    is_admin = input_pass == ADMIN_PASSWORD

    if is_admin:
        st.success("🔓 Mode Admin Aktif")
        
        # --- FITUR DAFTAR DOKUMEN ---
        st.subheader("📄 Daftar Isi Library")
        doc_list = get_document_list(st.session_state.vectorstore)
        if doc_list:
            for i, doc in enumerate(doc_list):
                st.caption(f"{i+1}. {doc}")
        else:
            st.info("Library kosong.")
        
        st.divider()
        uploaded_pdf = st.file_uploader("Upload PDF Baru", type="pdf")
        if st.button("➕ Tambahkan ke Library"):
            if uploaded_pdf:
                with st.spinner("Mengindeks dokumen..."):
                    st.session_state.vectorstore = add_to_library(uploaded_pdf)
                    st.success("Berhasil!")
                    st.rerun()
        
        if st.button("🔴 Reset Library"):
            if os.path.exists(DB_DIR):
                shutil.rmtree(DB_DIR)
                st.session_state.vectorstore = None
                st.rerun()
    else:
        st.info("Hanya Admin yang dapat mengelola dokumen.")

    st.divider()
    if st.button("🗑️ Hapus Riwayat Chat"):
        st.session_state.messages = []
        st.session_state.current_audio = None
        st.rerun()

# ==========================================
# 4. HALAMAN UTAMA (USER)
# ==========================================
st.title("📚 Corporate Knowledge Hub")
st.markdown("Akses informasi terpercaya dari basis data perusahaan.")

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if prompt := st.chat_input("Tanyakan sesuatu pada library..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    if st.session_state.vectorstore:
        with st.chat_message("assistant"):
            with st.spinner("Mencari jawaban..."):
                results = st.session_state.vectorstore.similarity_search(prompt, k=3)
                context = "\n".join([d.page_content for d in results])
                pages = sorted(list(set([d.metadata.get('page', 0) + 1 for d in results])))
                
                llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.1)
                sys_prompt = f"Anda asisten pintar. Jawab dengan ramah berdasarkan konteks.\n\nKONTEKS:\n{context}"
                ans = llm.invoke(sys_prompt + f"\n\nPERTANYAAN: {prompt}").content
                
                full_res = f"{ans}\n\n> 📍 **Referensi:** Halaman {', '.join(map(str, pages))}"
                st.markdown(full_res)
                
                st.session_state.current_audio = get_audio_html(ans)
                with st.expander("🔍 Lihat Referensi Asli"):
                    for doc in results: st.info(f"**Hal {doc.metadata.get('page',0)+1}:** {doc.page_content}")

                st.session_state.messages.append({"role": "assistant", "content": full_res})
                st.rerun()
    else:
        st.info("Library belum tersedia. Silakan hubungi Admin.")

if st.session_state.current_audio:
    st.divider()
    st.markdown(st.session_state.current_audio, unsafe_allow_html=True)
