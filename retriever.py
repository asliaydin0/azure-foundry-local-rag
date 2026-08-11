import sqlite3
import numpy as np
from database import DB_NAME

MAX_CHUNK_CHARS = 500
MAX_CONTEXT_CHARS = 1800
SIMILARITY_THRESHOLD = 0.55


def truncate_text(text: str, max_chars: int) -> str:
    """Metni token limitini aşmamak için güvenli şekilde kısaltır."""
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def cosine_similarity(v1, v2):
    """İki vektör (sayı dizisi) arasındaki benzerlik oranını hesaplar (0 ile 1 arası)."""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    # 0'a bölme hatasını engellemek için küçük bir kontrol
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
        
    return dot_product / (norm_v1 * norm_v2)

def build_context(documents: list) -> str:
    """Arama sonuçlarını model bağlam limitine uygun tek metne dönüştürür."""
    parts: list[str] = []
    total_len = 0

    for i, doc in enumerate(documents, start=1):
        snippet = truncate_text(doc.get("text", ""), MAX_CHUNK_CHARS)
        block = f"[{i}]\n{snippet}"
        block_len = len(block)

        if total_len + block_len > MAX_CONTEXT_CHARS:
            remaining = MAX_CONTEXT_CHARS - total_len
            if remaining > 80:
                parts.append(truncate_text(block, remaining))
            break

        parts.append(block)
        total_len += block_len + 2

    return "\n\n".join(parts)


class VectorRetriever:
    def __init__(self, embedder=None):
        print("🔍 Arama motoru başlatılıyor...")
        if embedder is not None:
            self.embedder = embedder
        else:
            from embedding import get_local_embedder
            self.embedder = get_local_embedder()
        
    def search(self, query: str, top_k: int = 2, min_score: float = SIMILARITY_THRESHOLD):
        """Kullanıcının sorusuna en uygun 'top_k' sayıdaki dökümanı bulur."""
        print(f"\nSoru: '{query}'")
        print("Soru vektöre çevriliyor ve veritabanında aranıyor...")

        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT id, source_file, chunk_text, embedding_vector FROM documents")
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return []

        query_vector = self.embedder.embed_text(query)
        if query_vector is None:
            return []

        results = []

        for row in rows:
            doc_id = row[0]
            source = row[1]
            text = row[2]
            blob_data = row[3]

            doc_vector = self.embedder.blob_to_vector(blob_data)
            score = cosine_similarity(query_vector, doc_vector)

            results.append({
                "id": doc_id,
                "source": source,
                "text": text,
                "score": score
            })

        results.sort(key=lambda x: x["score"], reverse=True)

        top_results = [
            item for item in results
            if item["score"] >= min_score
        ][:top_k]

        for item in top_results:
            item["text"] = truncate_text(item["text"], MAX_CHUNK_CHARS)
        return top_results

    def build_context(self, documents: list) -> str:
        return build_context(documents)

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