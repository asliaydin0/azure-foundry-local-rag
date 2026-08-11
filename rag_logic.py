"""RAG karar mantığı — Streamlit bağımsız, test edilebilir."""

import re

from retriever import build_context

NO_CONTEXT_MESSAGE = (
    "⚠️ Sistem hafızasında bu soruyu yanıtlayacak uygun bir bağlam/belge bulunamadı."
)

_SYSTEM_IDENTITY = (
    "Senin adın TechLas Workspace. İnternet bağlantısı olmadan, %100 yerel ve gizlilik odaklı "
    "çalışan bir yapay zeka asistanısın. Phi-3.5-mini modeli ve RAG (Retrieval-Augmented Generation) "
    "mimarisi üzerine inşa edildin. Kendi sistemin ve mimarin hakkındaki sorulara her zaman doğal, "
    "yetkin ve siberpunk temasına uygun bir dille cevap ver."
)

_IDENTITY_QUESTION_PATTERNS = (
    r"techlas",
    r"tech\s*las",
    r"\bworkspace\b",
    r"sen\s+kimsin",
    r"\bkimsin\b",
    r"ne\s+(?:sin|sun|yaparsın|yapabilirsin)",
    r"bu\s+sistem",
    r"sistem\s+nasıl",
    r"nasıl\s+çalış",
    r"\brag\b",
    r"phi[\-\s]?3",
    r"yerel.*asistan",
    r"kendini\s+tanıt",
    r"hangi\s+model",
    r"mimari",
    r"kapalı\s+devre",
    r"internetsiz",
)

_DOCUMENT_QUESTION_PATTERNS = (
    r"yüklediğim\s+belge",
    r"belgelerim",
    r"yüklediğim\s+dosya",
    r"hafızandaki\s+belge",
    r"kaynaklarım",
    r"ne\s+biliyorsun",
)


def is_document_meta_question(prompt: str) -> bool:
    """Kullanıcının yüklediği belgeler / hafıza durumu hakkındaki sorular."""
    text = prompt.lower().strip()
    return any(re.search(p, text) for p in _DOCUMENT_QUESTION_PATTERNS)


def is_identity_question(prompt: str) -> bool:
    """Asistan kimliği / sistem mimarisi sorularını tespit eder (belge soruları hariç)."""
    text = prompt.lower().strip()
    if is_document_meta_question(text):
        return False
    return any(re.search(p, text) for p in _IDENTITY_QUESTION_PATTERNS)


def should_early_exit(search_results: list, prompt: str) -> bool:
    """Arama sonucu yoksa ve kimlik/belge meta sorusu değilse erken çıkış yapılır."""
    if search_results:
        return False
    if is_identity_question(prompt) or is_document_meta_question(prompt):
        return False
    return True


def build_rag_system_prompt(
    baglam_metni: str,
    *,
    identity_mode: bool = False,
    document_mode: bool = False,
) -> str:
    if identity_mode:
        return f"""{_SYSTEM_IDENTITY}

KİMLİK VE SİSTEM SORULARI:
Kullanıcı TechLas Workspace, senin kimliğin veya sistem mimarin hakkında soru soruyor. Yukarıdaki kimliğini ve yerel RAG mimarisini kullanarak doğal, akıcı ve siberpunk temasında Türkçe cevap ver. BAĞLAM boş olsa bile yanıt ver; kendini tanıt ve sistemi açıkla.

YANIT FORMATI:
- Cevabına doğrudan konuya girerek başla.
- "Kaynak", "İçerik", "Belge" gibi meta etiketler kullanma.

BAĞLAM (varsa ek bilgi; kimlik sorularında zorunlu değil):
{baglam_metni or "(Boş — kimlik/sistem sorusu)"}
"""

    if document_mode:
        return f"""{_SYSTEM_IDENTITY}

BELGE / HAFIZA SORULARI:
Kullanıcı yüklediği belgeler veya hafızandaki bilgiler hakkında soru soruyor.
- BAĞLAM doluysa yalnızca oradaki bilgilerle özet cevap ver.
- BAĞLAM boşsa dürüstçe henüz indekslenmiş belge olmadığını söyle; sol panelden belge yükleyebileceğini kısaca belirt.
- Bilgi uydurma; dosya adı veya meta etiket kullanma.

YANIT FORMATI:
- Cevabına doğrudan konuya girerek başla.
- "Kaynak", "İçerik", "Belge" gibi meta etiketler kullanma.

BAĞLAM:
{baglam_metni or "(Boş — henüz indekslenmiş belge yok)"}
"""

    return f"""{_SYSTEM_IDENTITY}

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


def execute_rag_query(
    prompt: str,
    retriever,
    chat_client,
    complete_chat_fn,
    *,
    top_k: int = 2,
    sanitize_reply_fn=None,
) -> dict:
    """
    RAG hattının Streamlit bağımsız çekirdeği.

    Dönüş: route (early_exit | llm), answer, sources, identity_mode
    """
    kimlik_sorusu = is_identity_question(prompt)
    belge_sorusu = is_document_meta_question(prompt)
    bulunan_dokumanlar = retriever.search(prompt, top_k=top_k)

    if should_early_exit(bulunan_dokumanlar, prompt):
        return {
            "route": "early_exit",
            "answer": NO_CONTEXT_MESSAGE,
            "sources": [],
            "identity_mode": False,
            "document_mode": False,
        }

    baglam_metni = build_context(bulunan_dokumanlar) if bulunan_dokumanlar else ""
    system_prompt = build_rag_system_prompt(
        baglam_metni,
        identity_mode=kimlik_sorusu,
        document_mode=belge_sorusu,
    )

    response = complete_chat_fn(
        chat_client,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )

    if hasattr(response, "choices"):
        cevap = response.choices[0].message.content
    else:
        cevap = str(response)

    if sanitize_reply_fn:
        cevap = sanitize_reply_fn(cevap)

    return {
        "route": "llm",
        "answer": cevap,
        "sources": bulunan_dokumanlar,
        "identity_mode": kimlik_sorusu,
        "document_mode": belge_sorusu,
    }
