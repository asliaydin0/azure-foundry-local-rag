import sqlite3
import numpy as np
from embedding import LocalEmbedder
from database import DB_NAME

def cosine_similarity(v1, v2):
    """İki vektör (sayı dizisi) arasındaki benzerlik oranını hesaplar (0 ile 1 arası)."""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    # 0'a bölme hatasını engellemek için küçük bir kontrol
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
        
    return dot_product / (norm_v1 * norm_v2)

class VectorRetriever:
    def __init__(self):
        print("🔍 Arama motoru başlatılıyor...")
        self.embedder = LocalEmbedder()
        
    def search(self, query: str, top_k: int = 2):
        """Kullanıcının sorusuna en uygun 'top_k' sayıdaki dökümanı bulur."""
        print(f"\nSoru: '{query}'")
        print("Soru vektöre çevriliyor ve veritabanında aranıyor...")
        
        # 1. Kullanıcının sorusunu vektöre çevir
        query_vector = self.embedder.embed_text(query)
        if query_vector is None:
            return []
            
        # 2. Veritabanındaki tüm dökümanları çek
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT id, source_file, chunk_text, embedding_vector FROM documents")
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        
        # 3. Her bir döküman ile sorunun benzerliğini hesapla
        for row in rows:
            doc_id = row[0]
            source = row[1]
            text = row[2]
            blob_data = row[3]
            
            # BLOB verisini tekrar numpy dizisine çevir
            doc_vector = self.embedder.blob_to_vector(blob_data)
            
            # Benzerlik skorunu hesapla
            score = cosine_similarity(query_vector, doc_vector)
            
            results.append({
                "id": doc_id,
                "source": source,
                "text": text,
                "score": score
            })
            
        # 4. Sonuçları en yüksek skordan en düşüğe doğru sırala
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Sadece en iyi (top_k) sonuçları döndür
        return results[:top_k]

# Test aşaması
if __name__ == "__main__":
    retriever = VectorRetriever()
    
    # Asistanımıza veritabanından bulması için bir soru soralım
    test_sorusu = "Mobil uygulama geliştirirken arayüz donmalarını nasıl engelleyebilirim?"
    
    bulunan_dokumanlar = retriever.search(test_sorusu, top_k=2)
    
    print("\n✅ En Alakalı Sonuçlar Bulundu:")
    for i, doc in enumerate(bulunan_dokumanlar):
        print(f"\n--- Sonuç {i+1} (Benzerlik Skoru: {doc['score']:.4f}) ---")
        print(f"Kaynak: {doc['source']}")
        print(f"Metin: {doc['text']}")