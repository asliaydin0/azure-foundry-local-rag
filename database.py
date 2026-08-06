import sqlite3
import os

DB_NAME = "knowledge_base.db"

def init_db(quiet: bool = False):
    """SQLite veritabanını ve gerekli tabloyu oluşturur."""
    if not quiet:
        print("📦 Veritabanı bağlantısı kuruluyor...")

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

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
    if not quiet:
        print("✅ Veritabanı ve 'documents' tablosu başarıyla hazırlandı!")


def delete_vectors_by_source(source_file: str) -> int:
    """Yalnızca belirtilen kaynağa ait vektör parçalarını siler."""
    init_db(quiet=True)
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM documents WHERE source_file = ?", (source_file,))
    count = cursor.fetchone()[0] or 0
    cursor.execute("DELETE FROM documents WHERE source_file = ?", (source_file,))
    conn.commit()
    conn.close()
    return count


def count_vectors_by_source(source_file: str) -> int:
    init_db(quiet=True)
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM documents WHERE source_file = ?", (source_file,))
    count = cursor.fetchone()[0] or 0
    conn.close()
    return count
if __name__ == "__main__":
    init_db()