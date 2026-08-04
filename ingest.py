import sqlite3
from embedding import LocalEmbedder
from database import DB_NAME, init_db

# Asistanımıza öğreteceğimiz yazılım ve kodlama odaklı bilgi bankası (Chunk'lar halinde)
ORNEK_DOKUMANLAR = [
    {
        "source": "FullStack_Gelistirme_Notlari.md",
        "text": "Backend Mimarisinde Django ve Supabase Kullanımı: Django, ORM yapısı sayesinde veritabanı işlemlerini oldukça kolaylaştırır. Supabase gibi PostgreSQL tabanlı servislerle entegre edildiğinde, çok kiracılı (multi-tenant) platformlar hızlı, güvenli ve gerçek zamanlı (real-time) olarak ölçeklendirilebilir."
    },
    {
        "source": "Frontend_Rehberi.pdf",
        "text": "React ve TypeScript ile Güvenli Arayüzler: React projelerinde TypeScript kullanmak, derleme aşamasında tip hatalarını (type errors) yakalamamızı sağlar. Özellikle UI bileşenlerine aktarılan prop'ların arayüzleri (interface) önceden tanımlandığında, çalışma zamanı (runtime) çökmelerinin önüne geçilir ve kod okunabilirliği artar."
    },
    {
        "source": "Masaustu_Uygulama_Mimarisi.docx",
        "text": "Python ve PyQt6 ile Masaüstü Araçları: Python ile görsel algoritma tasarımı veya kod üretimi yapan masaüstü ortamları geliştirirken PyQt6 kütüphanesi modern arayüzler sunar. Olay güdümlü (event-driven) mimari kullanılarak buton tıklamaları ve ekran geçişleri ana işlemci döngüsünü (main thread) tıkamadan asenkron olarak yönetilmelidir."
    },
    {
        "source": "Mobil_Gelistirme_Kilavuzu.txt",
        "text": "Flutter ve Dart ile Çapraz Platform Uygulamalar: Flutter, tek bir kod tabanıyla hem iOS hem de Android'e çıktı verebilen güçlü bir framework'tür. Dart dilinin sağladığı asenkron yapı (Future, async/await) sayesinde, API çağrıları veya AI destekli içerik üretim işlemleri sırasında ekran donmaları engellenir."
    },
    {
        "source": "YapayZeka_Optimizasyon_Makalesi.pdf",
        "text": "Ağ Yönlendirme ve Genetik Algoritmalar: Genetik algoritmalar ve Karınca Kolonisi Optimizasyonu (ACO), ağlarda en iyi rotayı (Quality of Service) bulmak için sıkça kullanılır. Q-Learning gibi pekiştirmeli öğrenme teknikleriyle birleştirildiğinde, sistem zamanla ağ gecikmelerini öğrenerek parametrelerini kendi kendine optimize eder."
    }
]

def ingest_documents():
    print("🚀 Yazılım dökümanları yükleme ve vektörleme süreci başlatılıyor...\n")
    
    # 1. Veritabanının hazır olduğundan emin olalım
    init_db()
    
    # 2. Vektör motorunu başlat
    embedder = LocalEmbedder()
    
    # 3. SQLite bağlantısını aç
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    print("\n📚 Dökümanlar okunuyor, vektörlere çevriliyor ve veritabanına yazılıyor...")
    
    eklenen_sayisi = 0
    for doc in ORNEK_DOKUMANLAR:
        metin = doc["text"]
        kaynak = doc["source"]
        
        # Metni vektöre çeviriyoruz
        vektor = embedder.embed_text(metin)
        
        if vektor is not None:
            # Sayı dizisini SQLite'ın saklayabileceği BLOB (binary) formata çeviriyoruz
            blob_vektor = embedder.vector_to_blob(vektor)
            
            # Veritabanına ekliyoruz
            cursor.execute("""
                INSERT INTO documents (source_file, chunk_text, embedding_vector)
                VALUES (?, ?, ?)
            """, (kaynak, metin, blob_vektor))
            
            eklenen_sayisi += 1
            print(f"  ✅ Yüklendi: [{kaynak}] -> '{metin[:45]}...'")
        else:
            print(f"  ❌ Atlandı (Vektör hatası): '{metin[:30]}...'")
            
    conn.commit()
    conn.close()
    
    print(f"\n🎉 İşlem tamamlandı! Toplam {eklenen_sayisi} yazılım bilgi bloğu veritabanına kaydedildi.")

if __name__ == "__main__":
    ingest_documents()