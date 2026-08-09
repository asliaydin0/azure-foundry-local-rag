# ⚡ TechLas Workspace

> **Kapalı devre. Yerel zeka. Sıfır bulut.**

TechLas Workspace, **Microsoft Foundry Local** üzerinde çalışan, tamamen **çevrimdışı** ve **gizlilik odaklı** bir RAG (Retrieval-Augmented Generation) asistanıdır. Sorularınız yalnızca sizin yüklediğiniz belgelerden cevaplanır; verileriniz makinenizden çıkmaz, harici bir API'ye gönderilmez.

Synthwave/Cyberpunk arayüzü, geçmiş sohbet yönetimi ve katı anti-halüsinasyon kurallarıyla **üretim kalitesinde** bir yerel bilgi asistanı deneyimi sunar.

---

## 🧠 Proje Özeti

| | |
|---|---|
| **Ne yapar?** | Yerel belgelerinizden anlamsal arama yapar, ilgili parçaları bulur ve **Phi-3.5-mini** modeliyle bağlama dayalı cevap üretir. |
| **Nerede çalışır?** | %100 yerel — inference, embedding ve vektör arama kullanıcının bilgisayarında gerçekleşir. |
| **Kimler için?** | Gizlilik gerektiren dokümantasyon, teknik notlar, eğitim materyalleri ve kapalı devre bilgi tabanları. |

> **Not:** Modeller ilk çalıştırmada Foundry Local katalogundan indirilebilir. İndirme tamamlandıktan sonra sistem internetsiz kullanılabilir.

---

## 🛠️ Tech Stack

| Katman | Teknoloji |
|---|---|
| **Dil** | Python 3 |
| **Arayüz** | Streamlit |
| **Yerel LLM** | Microsoft Foundry Local · `Phi-3.5-mini-instruct-generic-cpu` |
| **Embedding** | Foundry Local · `qwen3-embedding-0.6b-generic-cpu` |
| **Vektör Depolama** | SQLite (`knowledge_base.db`) — BLOB formatında embedding saklama |
| **Vektör Arama** | Cosine similarity (NumPy) · eşik filtreli `top-k` retrieval |
| **RAG Mimarisi** | Retrieve → Context Build → Prompt → Generate |
| **Belge İşleme** | `pypdf`, `python-docx` |
| **Sohbet Geçmişi** | Yerel JSON (`chats/`) |
| **Tasarım** | Özel Synthwave/Cyberpunk CSS (`style.css`) |

---

## ✨ Temel Özellikler

- **🔒 %100 Yerel & Gizli** — Soru, belge ve cevaplar makinenizde kalır; cloud API yok.
- **🎯 Sıfır Halüsinasyon Odaklı** — Katı system prompt, benzerlik eşiği (`≥ 0.55`) ve boş bağlamda erken çıkış; alakasız içerik LLM'e gönderilmez.
- **📚 Çoklu Dosya Desteği** — PDF, TXT, MD ve DOCX yükleme, parçalama ve vektörleştirme.
- **🗂️ Kaynak Kütüphanesi** — Belgeleri listeleme, hafızaya ekleme ve güvenli silme (disk + vektör DB).
- **💬 Geçmiş Sohbet Yönetimi** — Oturumlar `chats/` altında kalıcı JSON olarak saklanır; geçmişten devam edilebilir.
- **📎 Dinamik Kaynak Gösterimi** — Her cevabın altında kullanılan belge parçaları terminal tarzı expander ile gösterilir.
- **🌆 Synthwave UI** — Neon çerçeveler, cam efektli sohbet balonları, kompakt sidebar ve karanlık tema.
- **🛡️ Sağlamlık Katmanları** — Model yükleme kontrolü, timeout koruması, meta veri sızıntısı temizleme.

---

## ⚙️ Çalışma Mimarisi

```
┌─────────────┐    ┌──────────────────┐    ┌─────────────┐    ┌──────────────┐
│  Kullanıcı  │───▶│  Belge Yükleme   │───▶│  Parçalama  │───▶│  Embedding   │
│   Sorusu    │    │  (documents/)    │    │  (800 char) │    │  (qwen3)     │
└─────────────┘    └──────────────────┘    └─────────────┘    └──────┬───────┘
       │                                                              │
       │                                                              ▼
       │                                                    ┌─────────────────┐
       │                                                    │ SQLite Vektör   │
       │                                                    │    Deposu       │
       │                                                    └────────┬────────┘
       │                                                             │
       ▼                                                             ▼
┌─────────────┐    ┌──────────────────┐    ┌─────────────┐    ┌──────────────┐
│   Cevap     │◀───│  Phi-3.5-mini    │◀───│   Prompt    │◀───│ Vector Search│
│  (Streamlit)│    │  (Foundry Local) │    │ Engineering │    │ (cosine ≥0.55)│
└─────────────┘    └──────────────────┘    └─────────────┘    └──────────────┘
```

### Adım adım akış

1. **Belge Yükleme** — Kullanıcı PDF/TXT/MD/DOCX dosyasını sidebar'dan yükler; dosya `documents/` klasörüne kaydedilir.
2. **Parçalama (Chunking)** — Metin ~800 karakterlik, %100 overlap'li parçalara bölünür.
3. **Embedding / Vektörizasyon** — Her parça `qwen3-embedding` modeliyle sayısal vektöre dönüştürülür.
4. **SQLite Kaydı** — Parça metni, kaynak dosya adı ve embedding BLOB olarak `knowledge_base.db`'ye yazılır.
5. **Vector Search** — Kullanıcı sorusu vektörleştirilir; tüm parçalarla cosine similarity hesaplanır, eşik üstü en iyi `top-k` sonuç seçilir.
6. **LLM Prompting** — Seçilen parçalar bağlam metnine dönüştürülür; katı kurallı system prompt ile Phi-3.5-mini'ye gönderilir.
7. **Cevap & Kaynak** — Model cevabı arayüzde gösterilir; kullanılan parçalar **Kaynak Detayları** expander'ında listelenir.

---

## 📁 Proje Yapısı

```
azure-foundry-local-rag/
├── app.py                 # Ana Streamlit uygulaması (UI + RAG sohbet)
├── retriever.py           # Vektör arama ve bağlam oluşturma
├── embedding.py           # Foundry Local embedding istemcisi
├── database.py            # SQLite şema ve vektör silme işlemleri
├── document_manager.py    # Belge ingest, chunk, silme, kütüphane istatistikleri
├── chat_history.py        # Geçmiş sohbet JSON depolama
├── style.css              # Synthwave/Cyberpunk tema
├── requirements.txt       # Python bağımlılıkları
├── documents/             # Yüklenen kaynak belgeler
├── chats/                 # Kayıtlı sohbet oturumları
└── knowledge_base.db      # Vektör veritabanı (otomatik oluşur)
```

---

## 🚀 Kurulum ve Çalıştırma

### Gereksinimler

- **Python** 3.10+
- **Microsoft Foundry Local** runtime (SDK ile birlikte gelir)
- Yeterli **RAM** (Phi-3.5-mini + embedding modeli için önerilen: 8 GB+)

### 1. Depoyu klonlayın

```bash
git clone https://github.com/<kullanici>/azure-foundry-local-rag.git
cd azure-foundry-local-rag
```

### 2. Sanal ortam oluşturun

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Bağımlılıkları yükleyin

```bash
pip install -r requirements.txt
```

### 4. Uygulamayı başlatın

```bash
streamlit run app.py
```

Tarayıcınızda varsayılan adres: **http://localhost:8501**

### İlk kullanım

1. Sol panelden **Veri Yönetimi** bölümüne bir belge yükleyin.
2. Dosya otomatik olarak hafızaya (vektör DB) eklenir.
3. Ana ekrandan sorunuzu yazın — sistem yalnızca yüklediğiniz belgelerden cevap üretir.

---

## 🔧 Yapılandırma Sabitleri

| Sabit | Dosya | Açıklama |
|---|---|---|
| `SIMILARITY_THRESHOLD = 0.55` | `retriever.py` | Minimum cosine similarity eşiği |
| `CHAT_TIMEOUT_SECONDS = 120` | `app.py` | LLM cevap zaman aşımı |
| `MAX_CHUNK_CHARS = 500` | `retriever.py` | Bağlam parça limiti |
| `CHUNK_SIZE = 800` | `document_manager.py` | Ingest parça boyutu |

---

## 📜 Lisans & Marka

**TechLas Workspace** — Yerel RAG · Kapalı Devre · Synthwave Edition

Microsoft Foundry Local SDK kullanımı, ilgili Microsoft lisans koşullarına tabidir.

---

<p align="center">
  <sub>⚡ Veriniz sizde kalır. Zeka yerelde çalışır. ⚡</sub>
</p>
