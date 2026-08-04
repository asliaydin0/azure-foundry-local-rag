import streamlit as st
import os
from retriever import VectorRetriever

# 1. Sayfa Tasarımı 
st.set_page_config(page_title="TechLas Workspace", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

def load_css(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

try:
    load_css("style.css")
except FileNotFoundError:
    st.warning("⚠️ 'style.css' dosyası bulunamadı.")

# 2. Sidebar (Sol Menü) - Dosya Yükleme Özelliği İle
with st.sidebar:
    st.markdown("### ⚙️ Sistem Paneli")
    st.caption("Veri Bağlantısı: Aktif")
    st.divider()
    
    # --- DOSYA YÜKLEME ALANI ---
    st.markdown("#### 📂 Yeni Kaynak Yükle")
    yuklenen_dosya = st.file_uploader("PDF, TXT, MD veya DOCX", type=["pdf", "txt", "md", "docx"])
    
    if yuklenen_dosya is not None:
        save_path = os.path.join(".", yuklenen_dosya.name)
        with open(save_path, "wb") as f:
            f.write(yuklenen_dosya.getbuffer())
        st.success(f"✅ {yuklenen_dosya.name} eklendi!")
    
    st.divider()
    
    # --- ESKİ TARANAN KAYNAKLAR LİSTESİ ---
    st.markdown("#### Taranan Kaynaklar")
    dosyalar = [
        "FullStack_Gelistirme_Notlari.md",
        "Frontend_Rehberi.pdf",
        "Masaustu_Uygulama_Mimarisi.docx",
        "Mobil_Gelistirme_Kilavuzu.txt",
        "YapayZeka_Optimizasyon_Makalesi.pdf"
    ]
    
    html_content = '<div class="scrollable-sources">'
    for dosya in dosyalar:
        html_content += f'<div class="file-card">📄 {dosya}</div>'
    html_content += '</div>'
    
    st.markdown(html_content, unsafe_allow_html=True)
    
    st.divider()

    # Nefes alan animasyonlu durum bildirgesi
    st.markdown('''
        <div class="status-indicator">
            <div class="status-dot"></div>
            Sistem Aktif ve Hazır
        </div>
    ''', unsafe_allow_html=True)

# 3. Ana Ekran Başlığı 
st.markdown('<h1 class="techlas-title">TechLas Workspace</h1>', unsafe_allow_html=True)
st.markdown('<p class="techlas-subtitle">Yazılım mimarisi ve geliştirme veritabanı üzerinden çalışan kapalı devre sistem.</p>', unsafe_allow_html=True)

# 4. Modelleri ve Arama Motorunu Önbelleğe Alma
@st.cache_resource
def load_ai_system():
    retriever = VectorRetriever()
    manager = retriever.embedder.manager
    
    hedef_model_id = "Phi-3.5-mini-instruct-generic-cpu:2"
    modeller = manager.catalog.list_models()
    llm_model = next((m for m in modeller if m.id == hedef_model_id), None)
    
    llm_model.load()
    chat_client = llm_model.get_chat_client()
    
    chat_client.settings.max_tokens = 300
    chat_client.settings.temperature = 0.1
    
    return retriever, chat_client

with st.spinner("Sistem başlatılıyor..."):
    retriever, chat_client = load_ai_system()

# 5. Sohbet Geçmişi (Session State) Yönetimi
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 6. Kullanıcıdan Yeni Soru Alma ve RAG Süreci
if prompt := st.chat_input("Veritabanında aramak istediğiniz konuyu yazın..."):
    
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Veriler analiz ediliyor..."):
            
            bulunan_dokumanlar = retriever.search(prompt, top_k=2)
            baglam_metni = "\n\n".join([f"- Kaynak: {doc['source']}\n  İçerik: {doc['text']}" for doc in bulunan_dokumanlar])
            
            system_prompt = f"""Sen sadece sana verilen dokümanları okuyabilen kısıtlı bir asistansın.
Dış dünyadan veya kendi eğitim verinden HİÇBİR ŞEY BİLMİYORSUN.

KURAL 1: Sadece ama SADECE aşağıdaki BAĞLAM bölümünde yazan metinleri kullanarak cevap ver.
KURAL 2: Eğer kullanıcının sorusunun cevabı aşağıdaki BAĞLAM içinde AÇIKÇA GEÇMİYORSA, kesinlikle hiçbir şey uydurma ve tam olarak şu cümleyi söyle: "Üzgünüm, sağlanan dokümanlarda bu sorunun cevabı bulunmamaktadır."
KURAL 3: Bağlamda olmayan sahte bir dosya veya kaynak ismi ASLA üretme.

BAĞLAM:
{baglam_metni}
"""
            
            try:
                response = chat_client.complete_chat(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                )
                
                if hasattr(response, 'choices'):
                    cevap = response.choices[0].message.content
                else:
                    cevap = str(response)
                    
                st.markdown(cevap)
                
                with st.expander("Kaynak Detayları"):
                    for i, doc in enumerate(bulunan_dokumanlar):
                        st.markdown(f"**Kaynak {i+1}: {doc['source']}**\n\n{doc['text']}")
                        
            except Exception as e:
                cevap = f"❌ Sistem Hatası: {e}"
                st.error(cevap)
            
            st.session_state.messages.append({"role": "assistant", "content": cevap})