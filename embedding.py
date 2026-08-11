import numpy as np
from foundry_local_sdk import FoundryLocalManager, Configuration

PROJECT_NAME = "azure-foundry-local-rag"
EMBEDDING_MODEL_ID = "qwen3-embedding-0.6b-generic-cpu:1"

_embedder_instance = None


def get_foundry_manager() -> FoundryLocalManager:
    """SDK singleton kuralına uygun şekilde tek manager örneği döndürür."""
    if FoundryLocalManager.instance is not None:
        return FoundryLocalManager.instance
    config = Configuration(PROJECT_NAME)
    FoundryLocalManager.initialize(config)
    return FoundryLocalManager.instance


def get_local_embedder() -> "LocalEmbedder":
    """Process içinde tek embedder örneği kullanılır (Streamlit rerun güvenli)."""
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = LocalEmbedder()
    return _embedder_instance


class LocalEmbedder:
    def __init__(self):
        print("⚙️ Embedding (Vektörleme) motoru başlatılıyor...")
        self.manager = get_foundry_manager()

        self.model_id = EMBEDDING_MODEL_ID
        modeller = self.manager.catalog.list_models()
        self.model = next((m for m in modeller if m.id == self.model_id), None)

        if not self.model:
            raise ValueError(f"❌ '{self.model_id}' modeli katalogda bulunamadı!")

        print(f"🧠 {self.model_id} hafızaya alınıyor (İlk seferde indirebilir)...")
        self.model.download()
        self.model.load()

        self.client = self.model.get_embedding_client()
        print("✅ Embedding motoru hazır!")

    def embed_text(self, text: str):
        """Bir metni matematiksel vektör listesine çevirir."""
        try:
            # 1. Önce doğrudan parametre adı yazmadan (positional) vermeyi deniyoruz
            response = self.client.generate_embedding(text)
            
            # Gelen yanıtın yapısını kontrol ederek vektörü alıyoruz
            if hasattr(response, 'embedding'):
                return response.embedding
            elif hasattr(response, 'data') and len(response.data) > 0:
                item = response.data[0]
                return getattr(item, 'embedding', item)
            else:
                return response
                
        except TypeError as e:
            # Eğer yine hata verirse, metodun hangi parametreleri istediğini yazdırıyoruz
            import inspect
            print(f"🔍 Metodun Beklediği Parametreler: {inspect.signature(self.client.generate_embedding)}")
            print(f"❌ Hata detayı: {e}")
            return None
        except Exception as e:
            print(f"❌ Vektör oluşturma hatası: {e}")
            return None

    def vector_to_blob(self, vector_list):
        """Sayı dizisini SQLite'ın saklayabileceği BLOB (ikili) formata dönüştürür."""
        return np.array(vector_list, dtype=np.float32).tobytes()

    def blob_to_vector(self, blob_data):
        """SQLite'tan okunan BLOB verisini tekrar Python sayı listesine çevirir."""
        return np.frombuffer(blob_data, dtype=np.float32)

# Bu dosyayı tek başına çalıştırdığımızda küçük bir test yapacak
if __name__ == "__main__":
    embedder = LocalEmbedder()
    test_metin = "Bu bir test cümlesidir."
    print(f"\n🔍 '{test_metin}' cümlesi vektöre çevriliyor...")
    
    vektor = embedder.embed_text(test_metin)
    if vektor is not None:
        print(f"✅ Başarılı! Toplam {len(vektor)} boyutlu sayı dizisi elde edildi.")
        print(f"Sayıların ilk 5 tanesi: {vektor[:5]}")