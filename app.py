import streamlit as st
import os
import sys
import json
import uuid
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Dict

# Configure page
st.set_page_config(
    page_title="Analisis Klaim INA-CBG Edisi 2 Berbasis AI",
    page_icon="📄",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .stButton > button[kind="primary"] {
        background-color: transparent !important;
        border: 2px solid rgba(255, 255, 255, 0.3) !important;
        color: white !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        background-color: rgba(255, 255, 255, 0.1) !important;
        border-color: rgba(255, 255, 255, 0.6) !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    </style>
""", unsafe_allow_html=True)

# Pustaka LangChain & Komponen AI
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

# --- USER SESSION MANAGEMENT ---
def get_or_create_user_id():
    """Membuat atau mengambil user ID unik dengan persistensi menggunakan file."""
    if 'user_id' not in st.session_state:
        user_id = load_user_id_from_file()
        
        if not user_id:
            user_id = str(uuid.uuid4())
            save_user_id_to_file(user_id)
        
        st.session_state.user_id = user_id
        st.session_state.session_created = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return st.session_state.user_id

def load_user_id_from_file():
    """Load user ID dari file lokal."""
    user_id_file = ".streamlit_user_id"
    if os.path.exists(user_id_file):
        try:
            with open(user_id_file, 'r', encoding='utf-8') as f:
                user_id = f.read().strip()
                if user_id:
                    return user_id
        except Exception as e:
            print(f"Error loading user ID: {e}")
    return None

def save_user_id_to_file(user_id):
    """Simpan user ID ke file lokal."""
    user_id_file = ".streamlit_user_id"
    try:
        with open(user_id_file, 'w', encoding='utf-8') as f:
            f.write(user_id)
        return True
    except Exception as e:
        print(f"Error saving user ID: {e}")
        return False

def reset_user_session():
    """Reset user session - hapus file user_id dan buat yang baru."""
    user_id_file = ".streamlit_user_id"
    if os.path.exists(user_id_file):
        try:
            os.remove(user_id_file)
        except:
            pass
    
    if 'user_id' in st.session_state:
        del st.session_state.user_id
    if 'current_conversation_id' in st.session_state:
        del st.session_state.current_conversation_id
    if 'current_messages' in st.session_state:
        del st.session_state.current_messages
    if 'conversation_title' in st.session_state:
        del st.session_state.conversation_title

def get_user_conversations(user_id):
    """Mengambil daftar semua percakapan user."""
    history_dir = "user_histories"
    user_dir = os.path.join(history_dir, user_id)
    
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)
        return []
    
    conversations = []
    for filename in os.listdir(user_dir):
        if filename.endswith('.json'):
            conv_id = filename.replace('.json', '')
            filepath = os.path.join(user_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if data.get('messages'):
                        conversations.append({
                            'id': conv_id,
                            'title': data.get('title', 'Percakapan Baru'),
                            'created': data.get('created', ''),
                            'updated': data.get('updated', ''),
                            'message_count': len(data.get('messages', []))
                        })
            except Exception as e:
                print(f"Error loading conversation {filename}: {e}")
                continue
    
    conversations.sort(key=lambda x: x.get('updated', ''), reverse=True)
    return conversations

def load_conversation(user_id, conversation_id):
    """Memuat percakapan tertentu."""
    history_dir = "user_histories"
    filepath = os.path.join(history_dir, user_id, f"{conversation_id}.json")
    
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading conversation: {e}")
            return None
    return None

def save_conversation(user_id, conversation_id, title, messages):
    """Menyimpan percakapan."""
    history_dir = "user_histories"
    user_dir = os.path.join(history_dir, user_id)
    
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)
    
    filepath = os.path.join(user_dir, f"{conversation_id}.json")
    
    existing_data = load_conversation(user_id, conversation_id)
    created_time = existing_data.get('created') if existing_data else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    data = {
        'id': conversation_id,
        'title': title,
        'created': created_time,
        'updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'messages': messages
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def delete_conversation(user_id, conversation_id):
    """Menghapus percakapan."""
    history_dir = "user_histories"
    filepath = os.path.join(history_dir, user_id, f"{conversation_id}.json")
    
    if os.path.exists(filepath):
        os.remove(filepath)
        return True
    return False

def create_new_conversation():
    """Membuat percakapan baru."""
    st.session_state.current_conversation_id = str(uuid.uuid4())
    st.session_state.current_messages = []
    st.session_state.conversation_title = "Percakapan Baru"

def generate_title_from_first_question(question):
    """Generate judul dari pertanyaan pertama."""
    title = question.strip()[:25]
    if len(question.strip()) > 50:
        title += "..."
    return title

def initialize_conversation_state(user_id):
    """Inisialisasi state percakapan saat aplikasi dimulai atau di-refresh."""
    if 'current_conversation_id' not in st.session_state:
        conversations = get_user_conversations(user_id)
        if conversations and len(conversations) > 0:
            last_conv = conversations[0]
            conv_data = load_conversation(user_id, last_conv['id'])
            if conv_data:
                st.session_state.current_conversation_id = last_conv['id']
                st.session_state.current_messages = conv_data.get('messages', [])
                st.session_state.conversation_title = conv_data.get('title', 'Percakapan Baru')
            else:
                create_new_conversation()
        else:
            create_new_conversation()

def setup_environment():
    """Memuat API key dari Streamlit secrets atau .env file."""
    api_key = None
    
    try:
        api_key = st.secrets.get("GROQ_API_KEY")
    except:
        pass
    
    if not api_key:
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        st.error("❌ **ERROR: GROQ_API_KEY tidak ditemukan!**")
        
        tab1, tab2 = st.tabs(["☁️ Streamlit Cloud", "💻 Development Lokal"])
        
        with tab1:
            st.markdown("**Untuk deployment di Streamlit Cloud:**")
            st.code('GROQ_API_KEY = "your_groq_api_key_here"', language="toml")
            st.caption("1. Buka Settings > Secrets\n2. Paste kode di atas\n3. Ganti dengan API key Anda\n4. Save")
        
        with tab2:
            st.markdown("**Untuk development lokal:**")
            st.code('GROQ_API_KEY=your_groq_api_key_here', language="text")
            st.caption("1. Buat file .env di root folder\n2. Paste kode di atas\n3. Ganti dengan API key Anda\n4. Restart app")
        
        st.link_button("🔑 Dapatkan API Key GRATIS", "https://console.groq.com/keys")
        
        st.info("""
        ℹ️ **GROQ API - GRATIS UNLIMITED!**
        
        ✅ Gratis selamanya
        ✅ 30 requests per menit
        ✅ Super cepat (1-2 detik)
        ✅ Model: Llama 3.1 70B
        ✅ Tanpa credit card
        
        Daftar di: https://console.groq.com
        """)
        st.stop()
    
    os.environ["GROQ_API_KEY"] = api_key

def load_json_database(json_file: str) -> Dict:
    """Load JSON database"""
    with st.spinner(f"📄 Memuat database JSON: {json_file}..."):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            st.error(f"❌ ERROR membaca JSON: {e}")
            st.stop()

def create_smart_chunks(json_data: Dict) -> List[Document]:
    """SMART CHUNKING: Satu case = satu chunk dengan struktur optimal"""
    with st.spinner("🧩 Membuat Smart Chunks..."):
        documents = []
        
        for case in json_data['cases']:
            # Format chunk yang OPTIMAL untuk pencarian
            chunk_text = f"""ID: {case['id']}
DIAGNOSA: {case.get('diagnosa_utama', '')} - {case['diagnosa'][:200]}
KODE: {', '.join(case['kode_diagnosa'][:5])}
KATEGORI: {case['kategori']}
PROSEDUR: {case['prosedur'][:150] if case['prosedur'] else 'Tidak ada'}
ASPEK KODING: {case['aspek_koding'][:300]}
KEYWORDS: {', '.join(case['keywords'][:15])}"""
            
            doc = Document(
                page_content=chunk_text,
                metadata={
                    'id': case['id'],
                    'diagnosa_utama': case.get('diagnosa_utama', ''),
                    'diagnosa': case['diagnosa'],
                    'kode': case['kode_diagnosa'],
                    'kategori': case['kategori'],
                    'prosedur': case['prosedur'],
                    'aspek_koding': case['aspek_koding'],
                    'perhatian_khusus': case['perhatian_khusus'],
                    'keywords': case['keywords']
                }
            )
            
            documents.append(doc)
        
        return documents

@st.cache_resource
def create_vector_store(_documents: List[Document]):
    """Buat vector store dari documents"""
    with st.spinner("🔍 Membuat Vector Store..."):
        try:
            embedding_model = HuggingFaceEmbeddings(
                model_name="paraphrase-multilingual-mpnet-base-v2"
            )
            db = FAISS.from_documents(_documents, embedding_model)
            return db
        except Exception as e:
            st.error(f"❌ ERROR membuat vector store: {e}")
            st.stop()

def smart_search(db, query: str, json_data: Dict, k: int = 3) -> List[Document]:
    """SMART SEARCH: Optimized hybrid search"""
    import re
    
    # Cek apakah ada kode ICD di query
    kode_pattern = r'\b[A-Z]\d{2}(?:\.\d+)?\b'
    kode_found = re.findall(kode_pattern, query.upper())
    
    # Jika ada kode spesifik, cari langsung
    if kode_found:
        filtered_docs = []
        for case in json_data['cases']:
            if any(kode in case['kode_diagnosa'] for kode in kode_found):
                doc = Document(
                    page_content="",
                    metadata={
                        'id': case['id'],
                        'diagnosa': case['diagnosa'],
                        'kode': case['kode_diagnosa'],
                        'prosedur': case['prosedur'],
                        'aspek_koding': case['aspek_koding'],
                        'perhatian_khusus': case['perhatian_khusus']
                    }
                )
                filtered_docs.append(doc)
        
        if filtered_docs:
            return filtered_docs[:k]
    
    # Semantic search
    results = db.similarity_search(query, k=k)
    return results

def run_groq_rag(db, json_data: Dict, query: str) -> str:
    """GROQ RAG: Gratis, Cepat, Akurat!"""
    with st.spinner("🤖 AI Groq sedang menganalisis (super cepat!)..."):
        try:
            llm = ChatGroq(
                model="moonshotai/kimi-k2-instruct",  # Model stabil Groq (gratis!)
                temperature=0.1,
                max_tokens=1024
            )
            
            # SINGLE SEARCH (hemat & cepat!)
            top_docs = smart_search(db, query, json_data, k=3)
            
            # Build context dari metadata
            context_parts = []
            for doc in top_docs:
                meta = doc.metadata
                context_part = f"""
DIAGNOSA: {meta.get('diagnosa', 'N/A')}
KODE ICD: {', '.join(meta.get('kode', []))}
PROSEDUR: {meta.get('prosedur') or 'Tidak ada prosedur khusus'}
ASPEK KODING: {meta.get('aspek_koding', 'N/A')}
PERHATIAN KHUSUS: {meta.get('perhatian_khusus') or 'Tidak ada'}
---"""
                context_parts.append(context_part)
            
            context = "\n".join(context_parts)
            
            # PROMPT YANG SUPER EFEKTIF
            prompt = f"""Kamu adalah ahli koding ICD-10 dan INA-CBG. Jawab dengan RINGKAS, AKURAT, dan TERSTRUKTUR.

KONTEKS DATABASE:
{context}

PERTANYAAN: {query}

INSTRUKSI:
1. Jawab LANGSUNG dengan struktur:
   - DIAGNOSA: (singkat)
   - KODE ICD-10/ICD-9: (list dengan penjelasan 1 kalimat)
   - PROSEDUR: (jika ada)
   - ASPEK KODING: (poin penting saja, max 3-4 poin)
   - PERHATIAN KHUSUS: (jika ada, max 2-3 poin)

2. ATURAN PENTING:
   - Jawab PADAT dan FOKUS (hindari pengulangan)
   - Jika ada kode kombinasi, jelaskan secara SINGKAT
   - Jika info tidak lengkap di konteks, katakan "Tidak ditemukan dalam database"
   - Gunakan bullet points (•) untuk list

3. FORMAT CONTOH:
DIAGNOSA: [nama diagnosa lengkap]

KODE ICD-10: 
• A01.0 - Typhoid fever
• A09 - Tidak dikoding jika sudah ada A01.0

PROSEDUR: [jika ada, jika tidak: "Tidak ada prosedur khusus"]

ASPEK KODING:
• [poin penting 1]
• [poin penting 2]

PERHATIAN KHUSUS: [jika ada, singkat saja]

Jawaban:"""
            
            response = llm.invoke(prompt)
            return response.content
            
        except Exception as e:
            st.error(f"❌ ERROR: {e}")
            return f"Maaf, terjadi kesalahan: {str(e)}"

# --- MAIN APP ---
def main():
    st.markdown("""
        <h1 style='margin-top: -3rem; padding-top: 0;'>
            📄 Analisis Klaim INA-CBG Edisi 2 Berbasis AI
        </h1>
    """, unsafe_allow_html=True)
    
    setup_environment()
    user_id = get_or_create_user_id()
    initialize_conversation_state(user_id)

    # Sidebar
    with st.sidebar:
        st.header("💬 Riwayat Chat")
        
        if st.button("➕ Percakapan Baru", use_container_width=True, type="primary"):
            create_new_conversation()
            st.rerun()
        
        st.divider()
        
        conversations = get_user_conversations(user_id)
        
        if conversations:
            st.caption(f"📋 {len(conversations)} percakapan tersimpan")
            
            for conv in conversations:
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    is_active = conv['id'] == st.session_state.current_conversation_id
                    
                    if st.button(
                        conv['title'],
                        key=f"conv_{conv['id']}",
                        use_container_width=True,
                        disabled=is_active
                    ):
                        conv_data = load_conversation(user_id, conv['id'])
                        if conv_data:
                            st.session_state.current_conversation_id = conv['id']
                            st.session_state.current_messages = conv_data.get('messages', [])
                            st.session_state.conversation_title = conv_data.get('title', 'Percakapan Baru')
                            st.rerun()
                
                with col2:
                    if st.button("🗑", key=f"del_{conv['id']}"):
                        if delete_conversation(user_id, conv['id']):
                            if conv['id'] == st.session_state.current_conversation_id:
                                remaining_convs = get_user_conversations(user_id)
                                if remaining_convs:
                                    first_conv = remaining_convs[0]
                                    conv_data = load_conversation(user_id, first_conv['id'])
                                    if conv_data:
                                        st.session_state.current_conversation_id = first_conv['id']
                                        st.session_state.current_messages = conv_data.get('messages', [])
                                        st.session_state.conversation_title = conv_data.get('title', 'Percakapan Baru')
                                else:
                                    create_new_conversation()
                            st.rerun()
        else:
            st.info("💭 Belum ada percakapan")
        
        st.divider()
        
        # Info Groq
        st.success("⚡ Powered by Groq - GRATIS & CEPAT!")
        st.caption(f"🔐 Session: {user_id[:12]}...")
        
        if st.button("🔄 Reset Session", use_container_width=True):
            reset_user_session()
            st.rerun()

    # Main content
    st.subheader(f"💬 {st.session_state.conversation_title}")
    st.caption(f"📊 {len(st.session_state.current_messages)} pesan")
    st.divider()

    json_file = "medical_database_structured.json"

    if not os.path.exists(json_file):
        st.error(f"❌ ERROR: File '{json_file}' tidak ditemukan.")
        st.info("💡 Jalankan converter: `python convert_medical_db_to_json.py`")
        st.stop()

    try:
        json_data = load_json_database(json_file)
        documents = create_smart_chunks(json_data)
        vector_db = create_vector_store(documents)

        st.success(f"✅ Database: {json_data['metadata']['total_cases']} cases | ⚡ Groq: GRATIS & SUPER CEPAT!")

        if len(st.session_state.current_messages) == 0:
            st.info("👋 Tanyakan tentang diagnosa, kode ICD, prosedur, atau aspek koding apapun!")
        
        for msg in st.session_state.current_messages:
            with st.chat_message("user"):
                st.write(msg["question"])
            with st.chat_message("assistant"):
                st.markdown(msg["answer"])

        pertanyaan_user = st.chat_input("💭 Ajukan pertanyaan Anda...")

        if pertanyaan_user:
            if pertanyaan_user.strip():
                with st.chat_message("user"):
                    st.write(pertanyaan_user)
                
                final_answer = run_groq_rag(vector_db, json_data, pertanyaan_user)
                
                with st.chat_message("assistant"):
                    st.markdown(final_answer)

                chat_entry = {
                    "question": pertanyaan_user,
                    "answer": final_answer,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                st.session_state.current_messages.append(chat_entry)

                if len(st.session_state.current_messages) == 1:
                    st.session_state.conversation_title = generate_title_from_first_question(pertanyaan_user)

                save_conversation(
                    user_id,
                    st.session_state.current_conversation_id,
                    st.session_state.conversation_title,
                    st.session_state.current_messages
                )
                
                st.rerun()
            else:
                st.warning("⚠️ Silakan masukkan pertanyaan terlebih dahulu.")

    except Exception as e:
        st.error(f"❌ TERJADI KESALAHAN: {e}")

if __name__ == "__main__":
    main()