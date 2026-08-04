import streamlit as st
from retriever import VectorRetriever

# 1. Sayfa Tasarımı ve Sekme Ayarları
st.set_page_config(page_title="Yerel AI Asistanı", page_icon="🤖", layout="wide")

st.title("🤖 Yerel Yapay Zeka Asistanı")
st.markdown("Dokümanlarınızdan beslenen, tamamen internetsiz çalışan yapay zeka.")

# 2. Modelleri ve Arama Motorunu Önbelleğe (Cache) Alma
# Streamlit her tıklamada sayfayı baştan okur. Modellerin her soruda tekrar 
# yüklenip zaman kaybettirmemesi için @st.cache_resource kullanıyoruz.
@st.cache_resource
def load_ai_system():
    retriever = VectorRetriever()
    manager = retriever.embedder.manager
    
    hedef_model_id = "Phi-3.5-mini-instruct-generic-cpu:2"
    modeller = manager.catalog.list_models()
    llm_model = next((m for m in modeller if m.id == hedef_model_id), None)
    
    llm_model.load()
    chat_client = llm_model.get_chat_client()
    
    # Bellek ve Yaratıcılık Ayarları
    chat_client.settings.max_tokens = 300
    chat_client.settings.temperature = 0.1
    
    return retriever, chat_client

# Modeller yüklenirken ekranda dönen bir yükleme animasyonu (spinner) göster
with st.spinner("Yapay Zeka Motoru Yükleniyor (Bu işlem ilk açılışta biraz sürebilir)..."):
    retriever, chat_client = load_ai_system()

# 3. Sohbet Geçmişi (Session State) Yönetimi
if "messages" not in st.session_state:
    st.session_state.messages = []

# Eski mesajları ekrana bas
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 4. Kullanıcıdan Yeni Soru Alma ve RAG Süreci
if prompt := st.chat_input("Yazılım geliştirme ile ilgili bir soru sorun..."):
    
    # Kullanıcının mesajını ekrana yaz ve geçmişe kaydet
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Asistanın Cevap Bölümü
    with st.chat_message("assistant"):
        with st.spinner("Dokümanlar taranıyor ve cevap üretiliyor..."):
            
            # Arama (Retrieval)
            bulunan_dokumanlar = retriever.search(prompt, top_k=2)
            baglam_metni = "\n\n".join([f"- Kaynak: {doc['source']}\n  İçerik: {doc['text']}" for doc in bulunan_dokumanlar])
            
            # Sistem Komutu (Prompt) - Halüsinasyonu engellemek için katı kurallar!
            system_prompt = f"""Sen sadece sana verilen dokümanları okuyabilen kısıtlı bir asistansın.
Dış dünyadan veya kendi eğitim verinden HİÇBİR ŞEY BİLMİYORSUN.

KURAL 1: Sadece ama SADECE aşağıdaki BAĞLAM bölümünde yazan metinleri kullanarak cevap ver.
KURAL 2: Eğer kullanıcının sorusunun cevabı aşağıdaki BAĞLAM içinde AÇIKÇA GEÇMİYORSA, kesinlikle hiçbir şey uydurma ve tam olarak şu cümleyi söyle: "Üzgünüm, sağlanan dokümanlarda bu sorunun cevabı bulunmamaktadır."
KURAL 3: Bağlamda olmayan sahte bir dosya veya kaynak ismi ASLA üretme.


BAĞLAM:
{baglam_metni}
"""
            
            # Cevap Üretimi (Generation)
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
                    
                # Cevabı ekrana yazdır
                st.markdown(cevap)
                
                # UX Dokunuşu: Kullanılan kaynakları şık bir açılır menüde (expander) göster
                with st.expander("🔍 Kullanılan Kaynakları Gör"):
                    for i, doc in enumerate(bulunan_dokumanlar):
                        st.info(f"**Kaynak {i+1}: {doc['source']}**\n\n{doc['text']}")
                        
            except Exception as e:
                cevap = f"❌ Bir hata oluştu: {e}"
                st.error(cevap)
            
            # Asistanın cevabını geçmişe kaydet
            st.session_state.messages.append({"role": "assistant", "content": cevap})