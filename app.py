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

# --- YENİ: AÇILIR EKRAN (POPUP) FONKSİYONU ---
@st.dialog("📋 Veritabanındaki Kaynaklar")
def kaynaklari_goster():
    st.markdown("Yapay zeka motorunun anlık olarak beslendiği dokümanlar:")
    
    dosyalar = [
        "FullStack_Gelistirme_Notlari.md",
        "Frontend_Rehberi.pdf",
        "Masaustu_Uygulama_Mimarisi.docx",
        "Mobil_Gelistirme_Kilavuzu.txt",
        "YapayZeka_Optimizasyon_Makalesi.pdf"
    ]
    
    html_content = '<div class="scrollable-sources" style="max-height: 50vh;">'
    for dosya in dosyalar:
        html_content += f'<div class="file-card">📄 {dosya}</div>'
    html_content += '</div>'
    
    st.markdown(html_content, unsafe_allow_html=True)

# 2. Sidebar (Sol Menü)
with st.sidebar:
    # Marka ve Ana Durum
    st.markdown("""
        <div style='text-align: center; margin-bottom: 20px;'>
            <h2 style='margin-bottom: 0px; background: linear-gradient(90deg, #D926A9, #00F0FF); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>Sistem Paneli</h2>
            <span style='color:#948AA3; font-size:0.85rem; font-weight: 300;'>TechLas Yerel Ağ Arayüzü</span>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="techlas-divider"></div>', unsafe_allow_html=True)
    
    # --- YENİ: SİSTEM BİLGİ KARTLARI (Micro-Dashboard) ---
    st.markdown("#### Sistem Metrikleri")
    st.markdown("""
        <div class="info-card">
            <div><span style='color:#948AA3'>Model:</span> <b>Phi-3.5-mini</b></div>
            <div><span style='color:#948AA3'>Vektör DB:</span> <b>Chroma/FAISS</b></div>
            <div><span style='color:#948AA3'>Ortam:</span> <b style='color:#00F0FF'>Localhost</b></div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="techlas-divider"></div>', unsafe_allow_html=True)
    
    # --- DOSYA YÜKLEME ALANI ---
    st.markdown("#### 📂 Veritabanını Besle")
    yuklenen_dosya = st.file_uploader("Dosya Yükle", type=["pdf", "txt", "md", "docx"], label_visibility="collapsed")
    
    if yuklenen_dosya is not None:
        save_path = os.path.join(".", yuklenen_dosya.name)
        with open(save_path, "wb") as f:
            f.write(yuklenen_dosya.getbuffer())
        # sağ alttan popup çıkar
        st.toast(f"{yuklenen_dosya.name} başarıyla eklendi!", icon="✅")
    
    st.markdown('<div class="techlas-divider"></div>', unsafe_allow_html=True)
    
    # --- AÇILIR EKRAN BUTONU ---
    st.markdown("#### Veri Yönetimi")
    if st.button("🔗 Aktif Kaynakları İncele", use_container_width=True):
        kaynaklari_goster()
    
    # Alt kısmı boş bırakıp en alta sabitlemek için boşluk
    st.markdown("<br><br>", unsafe_allow_html=True)

    # Canlı Durum Bildirgesi
    st.markdown('''
        <div class="status-indicator">
            <div class="status-dot"></div>
            Motor Çevrimiçi
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
            
            # --- YENİ: KATI HALÜSİNASYON ENGELLEYİCİ PROMPT ---
            system_prompt = f"""Sen TechLas firmasının resmi ve son derece katı kuralları olan yapay zeka asistanısın.

GÖREVİN: 
Kullanıcının sorusunu SADECE ama SADECE aşağıdaki BAĞLAM bölümünde verilen metinleri okuyarak cevaplamak.

KESİN KURALLAR (BUNLARI İHLAL EDEMEZSİN):
1. Eğer kullanıcının sorduğu soru BAĞLAM metninin içinde AÇIKÇA VE DOĞRUDAN geçmiyorsa, parçaları birleştirip tahmin yürütmek KESİNLİKLE YASAKTIR.
2. Bağlamda cevabı olmayan sorular için SADECE şu cümleyi kuracaksın: "Üzgünüm, mevcut veritabanımda bu konu hakkında bir bilgi bulunmuyor." Başka hiçbir kelime ekleme.
3. Asla sahte tanımlar üretme.

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