import streamlit as st
import os
from retriever import VectorRetriever
from document_manager import (
    delete_document,
    get_library_stats,
    ingest_file,
)

# 1. Sayfa Tasarımı ve Temel Ayarlar
st.set_page_config(page_title="TechLas Workspace", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

def load_css(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

try:
    load_css("style.css")
except FileNotFoundError:
    st.warning("⚠️ 'style.css' dosyası bulunamadı.")

# --- SESSION STATE ---
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "son_yuklenen" not in st.session_state:
    st.session_state.son_yuklenen = None
if "son_silinen" not in st.session_state:
    st.session_state.son_silinen = None
if "silme_bekleyen" not in st.session_state:
    st.session_state.silme_bekleyen = None

# Bildirimler
if st.session_state.son_yuklenen:
    ad, parca = st.session_state.son_yuklenen
    if parca > 0:
        st.toast(f"✅ {ad} — {parca} bölüm hafızaya eklendi", icon="📚")
    else:
        st.toast(f"⚠️ {ad} kaydedildi ancak metin çıkarılamadı", icon="⚠️")
    st.session_state.son_yuklenen = None

if st.session_state.son_silinen:
    st.toast(f"🗑️ {st.session_state.son_silinen} kaldırıldı", icon="✅")
    st.session_state.son_silinen = None

# 2. AI Sistemi (Sidebar'dan önce — dosya işlemleri embedder gerektirir)
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


def hafizaya_ekle(dosya_adi: str) -> int:
    """Dosyayı okur, vektörleştirir ve hafızaya yazar."""
    path = os.path.join(".", dosya_adi)
    if not os.path.exists(path):
        raise FileNotFoundError(f"'{dosya_adi}' bulunamadı.")
    return ingest_file(path, retriever.embedder, dosya_adi)


def kaynaktan_sil(dosya_adi: str) -> None:
    """Dosyayı diskten ve vektör hafızasından kaldırır."""
    if not delete_document(dosya_adi):
        raise FileNotFoundError(f"'{dosya_adi}' silinemedi veya bulunamadı.")

# --- KAYNAK KÜTÜPHANESİ DİYALOGU ---
@st.dialog("📂 Kaynak Kütüphanesi", width="small")
def kaynak_kutuphanesi_goster():
    kaynaklar, toplam_parca = get_library_stats()
    hafizada_count = sum(1 for k in kaynaklar if k["indexed"])

    st.markdown(
        f'<p class="dialog-summary">{len(kaynaklar)} kaynak · {toplam_parca} bölüm · {hafizada_count} hafızada</p>',
        unsafe_allow_html=True,
    )

    if not kaynaklar:
        st.markdown("""
            <div class="empty-library">
                <span class="empty-icon">📂</span>
                <span class="empty-text">Henüz kaynak yok</span>
                <span class="empty-hint">Sol panelden belge yükleyin</span>
            </div>
        """, unsafe_allow_html=True)
        return

    for i, kaynak in enumerate(kaynaklar):
        if st.session_state.silme_bekleyen == kaynak["name"]:
            st.markdown(f"""
                <div class="delete-confirm-box">
                    <span class="delete-confirm-title">🗑️ Silme Onayı</span>
                    <span class="delete-confirm-text"><b>{kaynak["name"]}</b> ve hafızadaki tüm bölümleri kalıcı olarak silinecek.</span>
                </div>
            """, unsafe_allow_html=True)
            onay_col, iptal_col = st.columns(2)
            with onay_col:
                if st.button("Evet, Sil", key=f"confirm_del_{i}", use_container_width=True, type="primary"):
                    try:
                        kaynaktan_sil(kaynak["name"])
                        st.session_state.son_silinen = kaynak["name"]
                        st.session_state.silme_bekleyen = None
                        st.rerun()
                    except Exception as e:
                        st.error(f"Silinemedi: {e}")
            with iptal_col:
                if st.button("İptal", key=f"cancel_del_{i}", use_container_width=True):
                    st.session_state.silme_bekleyen = None
                    st.rerun()
            continue

        durum_class = "indexed" if kaynak["indexed"] else "pending"
        durum_text = f'{kaynak["chunks"]} bölüm okundu' if kaynak["indexed"] else "Henüz okunmadı"

        if kaynak["indexed"]:
            info_col, del_col = st.columns([5.5, 1.3])
            action_cols = (info_col, None, del_col)
        else:
            info_col, teach_col, del_col = st.columns([4, 2.2, 1.3])
            action_cols = (info_col, teach_col, del_col)

        with action_cols[0]:
            st.markdown(f"""
                <div class="source-row source-row-inline {durum_class}">
                    <div class="source-icon">{kaynak["icon"]}</div>
                    <div class="source-details">
                        <span class="source-name" title="{kaynak["name"]}">{kaynak["name"]}</span>
                        <span class="source-meta">{kaynak["ext"]} · {kaynak["size"]} · {durum_text}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        if action_cols[1] is not None:
            with action_cols[1]:
                if st.button("Hafızaya Ekle", key=f"dlg_teach_{i}", use_container_width=True):
                    try:
                        with st.spinner("Hafızaya ekleniyor..."):
                            parca = hafizaya_ekle(kaynak["name"])
                        st.session_state.son_yuklenen = (kaynak["name"], parca)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Hafızaya eklenemedi: {e}")

        with action_cols[2]:
            if st.button("Sil", key=f"dlg_del_{i}", use_container_width=True):
                st.session_state.silme_bekleyen = kaynak["name"]
                st.rerun()

# 3. Sidebar — Profesyonel Kontrol Paneli
kaynaklar, toplam_parca = get_library_stats()

with st.sidebar:
    st.markdown("""
        <div class="sidebar-brand">
            <div class="brand-icon">⚡</div>
            <div class="brand-text">
                <span class="brand-title">TechLas</span>
                <span class="brand-sub">Yerel RAG Workspace</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="techlas-divider"></div>', unsafe_allow_html=True)

    st.markdown(f"""
        <div class="micro-dashboard">
            <div class="metrics-grid">
                <div class="metric-card">
                    <span class="metric-value">{len(kaynaklar)}</span>
                    <span class="metric-label">Kaynak</span>
                </div>
                <div class="metric-card">
                    <span class="metric-value">{toplam_parca}</span>
                    <span class="metric-label">Bölüm</span>
                </div>
            </div>
            <div class="system-info-row">
                <span class="info-tag"><span class="tag-dot model"></span>Phi-3.5-mini</span>
                <span class="info-tag"><span class="tag-dot db"></span>Yerel Hafıza</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="techlas-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
        <div class="sidebar-section-header">
            <p class="sidebar-section-title">Belge Yükle</p>
            <p class="sidebar-section-hint">PDF · TXT · MD · DOCX</p>
        </div>
    """, unsafe_allow_html=True)

    yuklenen_dosya = st.file_uploader(
        "Dosya seçin veya sürükleyin",
        type=["pdf", "txt", "md", "docx"],
        label_visibility="collapsed",
        key=f"uploader_{st.session_state.uploader_key}",
    )

    if yuklenen_dosya is not None:
        save_path = os.path.join(".", yuklenen_dosya.name)
        with open(save_path, "wb") as f:
            f.write(yuklenen_dosya.getbuffer())

        with st.spinner("Hafızaya ekleniyor..."):
            parca_sayisi = hafizaya_ekle(yuklenen_dosya.name)

        st.session_state.son_yuklenen = (yuklenen_dosya.name, parca_sayisi)
        st.session_state.uploader_key += 1
        st.rerun()

    st.markdown('<div class="techlas-divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-header"><p class="sidebar-section-title">Veri Yönetimi</p></div>', unsafe_allow_html=True)

    if st.button(
        f"📂 Kaynak Kütüphanesi ({len(kaynaklar)})",
        use_container_width=True,
        key="open_library",
    ):
        kaynak_kutuphanesi_goster()

    if kaynaklar:
        bekleyen = sum(1 for k in kaynaklar if not k["indexed"])
        if bekleyen > 0:
            st.markdown(
                f'<p class="sidebar-section-hint sidebar-hint-after-btn">{bekleyen} dosya henüz okunmadı</p>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="sidebar-flex-spacer"></div>', unsafe_allow_html=True)

    st.markdown('<div class="techlas-divider divider-bottom"></div>', unsafe_allow_html=True)

    st.markdown("""
        <div class="status-indicator status-indicator-bottom">
            <div class="status-dot"></div>
            Motor Çevrimiçi · Kapalı Devre
        </div>
    """, unsafe_allow_html=True)

# 4. Ana Ekran Başlığı
st.markdown('<h1 class="techlas-title">TechLas Workspace</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="techlas-subtitle">Yazılım mimarisi ve geliştirme veritabanı üzerinden çalışan kapalı devre sistem.</p>',
    unsafe_allow_html=True,
)

# 5. Sohbet Geçmişi
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 6. RAG Sohbet
if prompt := st.chat_input("Veritabanında aramak istediğiniz konuyu yazın..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Veriler analiz ediliyor..."):
            bulunan_dokumanlar = retriever.search(prompt, top_k=2)
            baglam_metni = "\n\n".join(
                [f"- Kaynak: {doc['source']}\n  İçerik: {doc['text']}" for doc in bulunan_dokumanlar]
            )

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
                        {"role": "user", "content": prompt},
                    ]
                )

                if hasattr(response, "choices"):
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
