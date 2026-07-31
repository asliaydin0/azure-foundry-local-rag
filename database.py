import sqlite3
import os

DB_NAME = "knowledge_base.db"

def init_db():
    """SQLite veritabanını ve gerekli tabloyu oluşturur."""
    print("📦 Veritabanı bağlantısı kuruluyor...")
    
    # DB dosyasına bağlan (yoksa otomatik oluşturur)
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Dokümanları ve vektörlerini tutacağımız tabloyu tasarlıyoruz
    # chunk_text: Dokümanın okunan metin parçası
    # embedding_vector: Yapay zekanın arama yapacağı sayısal vektör (BLOB formatında)
    # source_file: Metnin hangi kılavuzdan/dokümandan geldiği (Kaynak göstermek için)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file TEXT,
            chunk_text TEXT,
            embedding_vector BLOB
        )
    """)
    
    conn.commit()
    conn.close()
    print("✅ Veritabanı ve 'documents' tablosu başarıyla hazırlandı!")

if __name__ == "__main__":
    init_db()