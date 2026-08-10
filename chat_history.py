import json
import os
import uuid
from datetime import datetime

CHATS_DIR = "chats"


def ensure_chats_dir() -> None:
    os.makedirs(CHATS_DIR, exist_ok=True)


def create_chat_id() -> str:
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def _chat_path(chat_id: str) -> str:
    return os.path.join(CHATS_DIR, f"{chat_id}.json")


def derive_title(messages: list) -> str:
    for msg in messages:
        if msg.get("role") == "user":
            text = (msg.get("content") or "").strip().replace("\n", " ")
            if text:
                return text[:48] + ("..." if len(text) > 48 else "")
    return "Yeni Sohbet"


def format_date_label(iso_str: str) -> str:
    if not iso_str:
        return ""
    try:
        return datetime.fromisoformat(iso_str).strftime("%d.%m.%Y · %H:%M")
    except ValueError:
        return iso_str[:16]


def save_chat(chat_id: str, messages: list) -> None:
    if not messages:
        return

    ensure_chats_dir()
    path = _chat_path(chat_id)
    now = datetime.now().isoformat(timespec="seconds")

    existing: dict = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            existing = json.load(f)

    data = {
        "id": chat_id,
        "title": derive_title(messages),
        "created_at": existing.get("created_at", now),
        "updated_at": now,
        "messages": messages,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_chat(chat_id: str) -> list:
    path = _chat_path(chat_id)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("messages", [])


def list_chats() -> list:
    ensure_chats_dir()
    chats: list = []

    for fname in os.listdir(CHATS_DIR):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(CHATS_DIR, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            messages = data.get("messages", [])
            if not messages:
                continue
            chat_id = data.get("id", fname[:-5])
            chats.append({
                "id": chat_id,
                "title": data.get("title") or derive_title(messages),
                "updated_at": data.get("updated_at", ""),
                "message_count": len(messages),
                "date_label": format_date_label(data.get("updated_at", "")),
            })
        except (json.JSONDecodeError, OSError):
            continue

    chats.sort(key=lambda c: c["updated_at"], reverse=True)
    return chats


def delete_chat(chat_id: str) -> bool:
    """Tek bir sohbet JSON dosyasını diskten kalıcı olarak siler."""
    path = _chat_path(chat_id)
    if not os.path.exists(path):
        return False
    try:
        os.remove(path)
        return True
    except OSError:
        return False


def delete_all_chats() -> int:
    """chats/ altındaki tüm sohbet JSON dosyalarını siler; silinen dosya sayısını döndürür."""
    ensure_chats_dir()
    deleted = 0
    for fname in os.listdir(CHATS_DIR):
        if not fname.endswith(".json"):
            continue
        try:
            os.remove(os.path.join(CHATS_DIR, fname))
            deleted += 1
        except OSError:
            continue
    return deleted
