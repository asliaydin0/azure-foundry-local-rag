import os
import sqlite3
from pathlib import Path

from database import DB_NAME, init_db

SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".md", ".docx")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
DOCS_DIR = "."

EXT_ICONS = {
    ".pdf": "📕",
    ".txt": "📄",
    ".md": "📝",
    ".docx": "📘",
}


def format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def get_supported_files() -> list[str]:
    if not os.path.isdir(DOCS_DIR):
        return []
    return sorted(
        f for f in os.listdir(DOCS_DIR)
        if f.lower().endswith(SUPPORTED_EXTENSIONS)
    )


def extract_text(filepath: str) -> str:
    ext = Path(filepath).suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(filepath)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if ext == ".docx":
        from docx import Document
        doc = Document(filepath)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    if ext in (".txt", ".md"):
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    return ""


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def get_library_stats() -> tuple[list[dict], int]:
    init_db()
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT source_file, COUNT(*) FROM documents GROUP BY source_file")
    chunk_counts = dict(cursor.fetchall())
    cursor.execute("SELECT COUNT(*) FROM documents")
    total_chunks = cursor.fetchone()[0] or 0
    conn.close()

    files = []
    for name in get_supported_files():
        path = os.path.join(DOCS_DIR, name)
        ext = Path(name).suffix.lower()
        size_bytes = os.path.getsize(path)
        chunks = chunk_counts.get(name, 0)
        files.append({
            "name": name,
            "ext": ext.lstrip(".").upper(),
            "icon": EXT_ICONS.get(ext, "📄"),
            "size": format_size(size_bytes),
            "size_bytes": size_bytes,
            "chunks": chunks,
            "indexed": chunks > 0,
        })
    return files, total_chunks


def ingest_file(filepath: str, embedder, source_name=None) -> int:
    init_db()
    source = source_name or os.path.basename(filepath)
    text = extract_text(filepath)
    chunks = chunk_text(text)
    if not chunks:
        return 0

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM documents WHERE source_file = ?", (source,))

    count = 0
    for chunk in chunks:
        vector = embedder.embed_text(chunk)
        if vector is not None:
            blob = embedder.vector_to_blob(vector)
            cursor.execute(
                "INSERT INTO documents (source_file, chunk_text, embedding_vector) VALUES (?, ?, ?)",
                (source, chunk, blob),
            )
            count += 1

    conn.commit()
    conn.close()
    return count


def delete_document(filename: str) -> bool:
    filepath = os.path.join(DOCS_DIR, filename)
    init_db()
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM documents WHERE source_file = ?", (filename,))
    conn.commit()
    conn.close()

    if os.path.exists(filepath):
        os.remove(filepath)
        return True
    return False
