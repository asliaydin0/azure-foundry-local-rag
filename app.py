import concurrent.futures
import re
import streamlit as st
import os
import html
from retriever import VectorRetriever, build_context
from document_manager import (
    delete_document,
    get_library_stats,
    ingest_file,
)
import chat_history

CHAT_TIMEOUT_SECONDS = 120

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
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = chat_history.create_chat_id()
if "messages" not in st.session_state:
    st.session_state.messages = []

# Bildirimler
if st.session_state.son_yuklenen:
    ad, parca = st.session_state.son_yuklenen
    if parca > 0:
        st.toast(f"✅ {ad} — {parca} bölüm hafızaya eklendi", icon="📚")
    else:
        st.toast(f"⚠️ {ad} kaydedildi ancak metin çıkarılamadı", icon="⚠️")
    st.session_state.son_yuklenen = None

if st.session_state.son_silinen:
    st.toast(st.session_state.son_silinen, icon="✅")
    st.session_state.son_silinen = None

# 2. AI Sistemi (Sidebar'dan önce — dosya işlemleri embedder gerektirir)
@st.cache_resource
def load_ai_system(_retriever_version: int = 2):
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


def complete_chat_with_timeout(chat_client, messages, timeout: int = CHAT_TIMEOUT_SECONDS):
    """SDK'da doğrudan timeout olmadığı için tamamlamayı sınırlı sürede çalıştırır."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(chat_client.complete_chat, messages)
        return future.result(timeout=timeout)


def _chat_hata_mesaji(exc: Exception) -> str:
    mesaj = str(exc).lower()
    if isinstance(exc, TimeoutError) or "timeout" in mesaj:
        return "⏱️ Modelin cevap verme süresi aşıldı. Lütfen daha kısa bir soru sorun veya işlemi tekrarlayın."
    if "cancelled" in mesaj or "canceled" in mesaj:
        return "⏱️ Modelin cevap verme süresi aşıldı. Lütfen daha kısa bir soru sorun veya işlemi tekrarlayın."
    return f"❌ Sistem Hatası: {exc}"


_META_LEAK_RE = re.compile(
    r"^(?:Kaynak|Kaynaklar|İçerik|Belge|Source|Content|Document)\s*[:\-]\s*.+?(?:\n|$)",
    re.IGNORECASE | re.MULTILINE,
)
_COMBINED_META_RE = re.compile(
    r"^(?:Kaynak|Kaynaklar)\s*[:\-][^\n]*(?:\s*İçerik\s*[:\-][^\n]*)?",
    re.IGNORECASE,
)


def _sanitize_assistant_reply(text: str) -> str:
    """Modelin bağlam meta verilerini kopyalamasını temizler."""
    cleaned = (text or "").strip()
    for _ in range(4):
        updated = _COMBINED_META_RE.sub("", cleaned).strip()
        updated = _META_LEAK_RE.sub("", updated).strip()
        if updated == cleaned:
            break
        cleaned = updated
    return cleaned


def _build_rag_system_prompt(baglam_metni: str) -> str:
    return f"""Sen TechLas firmasının resmi yapay zeka asistanısın.

GÖREVİN:
Kullanıcının sorusunu YALNIZCA aşağıdaki BAĞLAM bölümündeki bilgilerle cevapla.

KESİN KURALLAR (İHLAL EDİLEMEZ):
1. Bağlamda cevap yoksa SADECE şunu yaz: "Üzgünüm, mevcut veritabanımda bu konu hakkında bir bilgi bulunmuyor."
2. Tahmin yürütme, bilgi uydurma, parçaları birleştirip çıkarım yapma.
3. YANIT FORMATI — ÇOK ÖNEMLİ:
   - Cevabına DOĞRUDAN konuya girerek başla. İlk cümle sorunun cevabı olsun.
   - "Kaynak", "İçerik", "Belge", "Context", "Source", "Content" kelimelerini ASLA yazma.
   - Dosya adı, uzantı (.pdf, .docx), parça numarası veya meta etiket KULLANMA.
   - Bağlamdaki etiketleri, başlıkları veya yapıyı cevaba KOPYALAMA.
   - Kaynaklar arayüzde ayrı gösterilir; sen sadece doğal, akıcı Türkçe cevap ver.

BAĞLAM (iç yapıyı kullanıcıya yansıtma):
{baglam_metni}
"""

with st.spinner("Sistem başlatılıyor..."):
    retriever, chat_client = load_ai_system()


def _save_current_chat() -> None:
    if st.session_state.messages:
        chat_history.save_chat(st.session_state.current_chat_id, st.session_state.messages)


def _start_new_chat() -> None:
    _save_current_chat()
    st.session_state.current_chat_id = chat_history.create_chat_id()
    st.session_state.messages = []


def _load_chat_session(chat_id: str) -> None:
    if chat_id == st.session_state.current_chat_id:
        return
    _save_current_chat()
    st.session_state.current_chat_id = chat_id
    st.session_state.messages = chat_history.load_chat(chat_id)


def hafizaya_ekle(dosya_adi: str) -> int:
    """Dosyayı okur, vektörleştirir ve hafızaya yazar."""
    from document_manager import DOCS_DIR, ensure_docs_dir
    ensure_docs_dir()
    path = os.path.join(DOCS_DIR, dosya_adi)
    if not os.path.exists(path):
        raise FileNotFoundError(f"'{dosya_adi}' bulunamadı.")
    return ingest_file(path, retriever.embedder, dosya_adi)


def kaynaktan_sil(dosya_adi: str) -> dict:
    """Dosyayı diskten ve vektör hafızasından iki aşamada kaldırır."""
    result = delete_document(dosya_adi)
    if not result["success"]:
        raise FileNotFoundError(f"'{dosya_adi}' sistemde veya hafızada bulunamadı.")
    return result


def _silme_toast_mesaji(result: dict) -> str:
    ad = result["filename"]
    if result["file_deleted"] and result["vectors_removed"] > 0:
        return f"{ad} sistemden ve hafızadan tamamen silindi!"
    if result["file_deleted"]:
        return f"{ad} sistemden silindi!"
    if result["vectors_removed"] > 0:
        return f"{ad} hafızadan silindi ({result['vectors_removed']} bölüm)!"
    return f"{ad} kaldırıldı."


def _kutuphane_kapatildi() -> None:
    st.session_state.silme_bekleyen = None


def _silme_onay_ekrani(dosya_adi: str) -> None:
    """Silme onayını liste dışında gösterir; işlem sonrası fragment yenilenir."""
    st.markdown(f"""
        <div class="delete-confirm-box">
            <span class="delete-confirm-title">🗑️ Silme Onayı</span>
            <span class="delete-confirm-text"><b>{dosya_adi}</b> ve hafızadaki tüm bölümleri kalıcı olarak silinecek.</span>
        </div>
    """, unsafe_allow_html=True)

    onay_col, iptal_col = st.columns(2)
    with onay_col:
        if st.button("Evet, Sil", key="confirm_del_active", use_container_width=True, type="primary"):
            try:
                result = kaynaktan_sil(dosya_adi)
                st.session_state.silme_bekleyen = None
                mesaj = _silme_toast_mesaji(result)
                st.session_state.son_silinen = mesaj
                st.toast(mesaj, icon="✅")
                if result.get("file_warning"):
                    st.warning(result["file_warning"])
                st.rerun(scope="fragment")
            except PermissionError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Silinemedi: {e}")
    with iptal_col:
        if st.button("İptal", key="cancel_del_active", use_container_width=True):
            st.session_state.silme_bekleyen = None
            st.rerun(scope="fragment")

# --- KAYNAK KÜTÜPHANESİ DİYALOGU ---
@st.dialog("Kaynak Kütüphanesi", width="small", on_dismiss=_kutuphane_kapatildi)
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

    bekleyen_silme = st.session_state.silme_bekleyen
    if bekleyen_silme:
        if any(k["name"] == bekleyen_silme for k in kaynaklar):
            _silme_onay_ekrani(bekleyen_silme)
            return
        st.session_state.silme_bekleyen = None

    for i, kaynak in enumerate(kaynaklar):
        durum_class = "indexed" if kaynak["indexed"] else "pending"
        durum_text = f'{kaynak["chunks"]} bölüm okundu' if kaynak["indexed"] else "Henüz okunmadı"

        if kaynak["indexed"]:
            info_col, del_col = st.columns([4.8, 1.6])
            action_cols = (info_col, None, del_col)
        else:
            info_col, teach_col, del_col = st.columns([3.4, 3.2, 1.5])
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
                if st.button("➕ Hafızaya Ekle", key=f"dlg_teach_{i}", use_container_width=True, help="Asistana öğret"):
                    try:
                        with st.spinner("Hafızaya ekleniyor..."):
                            parca = hafizaya_ekle(kaynak["name"])
                        if parca > 0:
                            st.toast(f"✅ {kaynak['name']} — {parca} bölüm hafızaya eklendi", icon="📚")
                        else:
                            st.toast(f"⚠️ {kaynak['name']} kaydedildi ancak metin çıkarılamadı", icon="⚠️")
                    except Exception as e:
                        st.error(f"Hafızaya eklenemedi: {e}")

        with action_cols[2]:
            if st.button("🗑 Sil", key=f"dlg_del_{i}", use_container_width=True, help="Kaynağı kaldır"):
                st.session_state.silme_bekleyen = kaynak["name"]
                st.rerun(scope="fragment")

# 3. Sidebar — Profesyonel Kontrol Paneli
kaynaklar, toplam_parca = get_library_stats()
gecmis_sohbetler = chat_history.list_chats()

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

    st.markdown('<div class="new-chat-marker"></div>', unsafe_allow_html=True)
    if st.button("➕ Yeni Sohbet", use_container_width=True, type="primary", key="new_chat"):
        _start_new_chat()
        st.toast("Yeni sohbet başlatıldı", icon="✨")
        st.rerun()

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
        from document_manager import DOCS_DIR, ensure_docs_dir
        ensure_docs_dir()
        save_path = os.path.join(DOCS_DIR, yuklenen_dosya.name)
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
        f"Kaynak Kütüphanesi ({len(kaynaklar)})",
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

    st.markdown('<div class="techlas-divider"></div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="sidebar-section-header chat-history-section">'
        f'<p class="sidebar-section-title">Geçmiş Sohbetler</p>'
        f'<p class="sidebar-section-hint">{len(gecmis_sohbetler)} kayıtlı oturum</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if not gecmis_sohbetler:
        st.markdown(
            '<p class="sidebar-section-hint chat-history-empty">Henüz kayıtlı sohbet yok</p>',
            unsafe_allow_html=True,
        )
    else:
        with st.container(height=300, border=False, gap=0):
            st.markdown('<span class="chat-history-scroll-inner"></span>', unsafe_allow_html=True)
            for sohbet in gecmis_sohbetler[:20]:
                aktif = sohbet["id"] == st.session_state.current_chat_id
                if aktif:
                    st.markdown('<span class="chat-history-row-active"></span>', unsafe_allow_html=True)
                if st.button(
                    sohbet["title"],
                    key=f"hist_{sohbet['id']}",
                    use_container_width=True,
                    type="secondary",
                    help=sohbet["date_label"],
                ):
                    _load_chat_session(sohbet["id"])
                    st.rerun()

    st.markdown("""
        <div class="sidebar-footer-fixed">
            <div class="techlas-divider divider-bottom"></div>
            <div class="status-indicator status-indicator-bottom">
                <div class="status-dot"></div>
                Motor Çevrimiçi · Kapalı Devre
            </div>
        </div>
    """, unsafe_allow_html=True)

# 4. Ana Ekran Başlığı
st.markdown('<h1 class="techlas-title">TechLas Workspace</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="techlas-subtitle">Yazılım mimarisi ve geliştirme veritabanı üzerinden çalışan kapalı devre sistem.</p>',
    unsafe_allow_html=True,
)

USER_AVATAR = "👤"
ASSISTANT_AVATAR = "⚡"


def render_chat_message(role: str, content: str) -> None:
    avatar = USER_AVATAR if role == "user" else ASSISTANT_AVATAR
    with st.chat_message(role, avatar=avatar):
        st.markdown(f'<span class="chat-role-marker chat-role-{role}"></span>', unsafe_allow_html=True)
        st.markdown(content)

# 5. Sohbet Geçmişi
for msg in st.session_state.messages:
    render_chat_message(msg["role"], msg["content"])

# 6. RAG Sohbet
if prompt := st.chat_input("Veritabanında aramak istediğiniz konuyu yazın..."):
    render_chat_message("user", prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        st.markdown('<span class="chat-role-marker chat-role-assistant"></span>', unsafe_allow_html=True)
        with st.spinner("Veriler analiz ediliyor..."):
            bulunan_dokumanlar = retriever.search(prompt, top_k=2)
            baglam_metni = build_context(bulunan_dokumanlar)

            system_prompt = _build_rag_system_prompt(baglam_metni)

            try:
                response = complete_chat_with_timeout(
                    chat_client,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    timeout=CHAT_TIMEOUT_SECONDS,
                )

                if hasattr(response, "choices"):
                    cevap = response.choices[0].message.content
                else:
                    cevap = str(response)

                cevap = _sanitize_assistant_reply(cevap)

                st.markdown(cevap)

                with st.expander("📎 Kaynak Detayları", expanded=False):
                    for i, doc in enumerate(bulunan_dokumanlar):
                        st.markdown(
                            f'<div class="source-terminal-block">'
                            f'<span class="source-terminal-label">Kaynak {i + 1}: {html.escape(doc["source"])}</span>'
                            f'<pre class="source-terminal-text">{html.escape(doc["text"])}</pre>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

            except Exception as e:
                cevap = _chat_hata_mesaji(e)
                st.warning(cevap)

            st.session_state.messages.append({"role": "assistant", "content": cevap})
            _save_current_chat()
