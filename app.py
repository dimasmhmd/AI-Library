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
st.set_page_config(page_title="Pro Corporate Library", page_icon="🏢", layout="wide")

ADMIN_PASSWORD = "admin123" 
DB_DIR = "permanent_library_db"
PDF_STORAGE_DIR = "stored_pdfs" # Folder untuk file PDF asli

# Buat folder jika belum ada
if not os.path.exists(PDF_STORAGE_DIR):
    os.makedirs(PDF_STORAGE_DIR)

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
                <p style="margin-bottom: 5px; font-weight: bold; color: #31333F;">🔊 Audio Output ({detected_lang.upper()}):</p>
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
    if vectorstore is None: return []
    try:
        data = vectorstore.get()
        metadatas = data.get('metadatas', [])
        sources = set([m.get('source', 'Unknown') for m in metadatas])
        return sorted(list(sources))
    except: return []

def add_to_library(uploaded_file):
    fname = uploaded_file.name
    save_path = os.path.join(PDF_STORAGE_DIR, fname)
    
    # Simpan file asli secara permanen untuk fitur download
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    loader = PyPDFLoader(save_path)
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(pages)
    
    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=load_embeddings(),
        persist_directory=DB_DIR, collection_name="admin_lib"
    )
    return vectorstore

def delete_specific_doc(vectorstore, source_path):
    try:
        # Hapus dari Vector DB
        vectorstore.delete(where={"source": source_path})
        # Hapus file asli dari storage
        if os.path.exists(source_path):
            os.remove(source_path)
        return True
    except Exception as e:
        st.error(f"Gagal menghapus: {e}")
        return False

# ==========================================
# 3. SIDEBAR & ADMIN POP OVER
# ==========================================
if "messages" not in st.session_state: st.session_state.messages = []
if "current_audio" not in st.session_state: st.session_state.current_audio = None
if "vectorstore" not in st.session_state: st.session_state.vectorstore = get_vectorstore()

with st.sidebar:
    st.title("🏢 Navigation")
    
    with st.popover("🔐 Login Admin", use_container_width=True):
        input_pass = st.text_input("Admin Password", type="password")
        is_admin = input_pass == ADMIN_PASSWORD
        if is_admin: st.success("Authenticated")
        elif input_pass: st.error("Invalid Password")

    st.divider()

    if is_admin:
        st.subheader("🛠️ Library Manager")
        
        doc_list = get_document_list(st.session_state.vectorstore)
        if doc_list:
            st.caption("Dokumen Tersimpan:")
            for doc_path in doc_list:
                doc_name = os.path.basename(doc_path)
                with st.container(border=True):
                    st.text(f"📄 {doc_name}")
                    col1, col2 = st.columns(2)
                    
                    # Fitur Download
                    if os.path.exists(doc_path):
                        with open(doc_path, "rb") as f:
                            col1.download_button(
                                label="📥", 
                                data=f, 
                                file_name=doc_name, 
                                mime="application/pdf",
                                key=f"dl_{doc_path}"
                            )
                    
                    # Fitur Hapus
                    if col2.button("🗑️", key=f"del_{doc_path}", use_container_width=True):
                        if delete_specific_doc(st.session_state.vectorstore, doc_path):
                            st.toast(f"{doc_name} dihapus!")
                            st.rerun()
        else:
            st.info("Library kosong.")

        st.divider()
        uploaded_pdf = st.file_uploader("Upload New PDF", type="pdf", label_visibility="collapsed")
        if st.button("➕ Add to Library", use_container_width=True):
            if uploaded_pdf:
                with st.spinner("Indexing..."):
                    st.session_state.vectorstore = add_to_library(uploaded_pdf)
                    st.success("Done!")
                    st.rerun()
    else:
        st.info("Mode Pengguna: Tanya jawab dengan dokumen perusahaan.")

    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.current_audio = None
        st.rerun()

# ==========================================
# 4. HALAMAN UTAMA
# ==========================================
st.title("📚 Corporate Knowledge Base")

if st.session_state.vectorstore is None:
    st.info("👋 Library kosong. Admin dapat mengunggah dokumen via panel login.")

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if prompt := st.chat_input("Tanyakan sesuatu..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    if st.session_state.vectorstore:
        with st.chat_message("assistant"):
            with st.spinner("Berpikir..."):
                results = st.session_state.vectorstore.similarity_search(prompt, k=3)
                context = "\n".join([d.page_content for d in results])
                pages = sorted(list(set([d.metadata.get('page', 0) + 1 for d in results])))
                
                llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.1)
                ans = llm.invoke(f"Konteks: {context}\nPertanyaan: {prompt}").content
                
                full_res = f"{ans}\n\n> 📍 **Ref:** Hal {', '.join(map(str, pages))}"
                st.markdown(full_res)
                st.session_state.current_audio = get_audio_html(ans)
                st.session_state.messages.append({"role": "assistant", "content": full_res})
                st.rerun()

if st.session_state.current_audio:
    st.divider()
    st.markdown(st.session_state.current_audio, unsafe_allow_html=True)
