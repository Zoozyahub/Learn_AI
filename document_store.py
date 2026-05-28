"""
Простое файловое хранилище метаданных документов.
В продакшне заменить на SQLite/PostgreSQL.
"""
import json
import os
from threading import Lock


DATA_FILE = "./document_store.json"
_lock = Lock()


class DocumentStore:
    def __init__(self, data_file: str = DATA_FILE):
        self.data_file = data_file
        self._ensure_file()

    def _ensure_file(self):
        if not os.path.exists(self.data_file):
            self._write({})

    def _read(self) -> dict:
        with open(self.data_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write(self, data: dict):
        with open(self.data_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _next_id(self, user_docs: list) -> int:
        if not user_docs:
            return 1
        return max(d["id"] for d in user_docs) + 1

    def get_documents(self, user_id: str) -> list[dict]:
        with _lock:
            data = self._read()
            return data.get(user_id, [])

    def add_document(
        self,
        user_id: str,
        original_name: str,
        file_path: str,
        file_size: int
    ) -> dict:
        with _lock:
            data = self._read()
            user_docs = data.get(user_id, [])
            doc = {
                "id": self._next_id(user_docs),
                "name": original_name,
                "file_path": file_path,
                "file_size": file_size,
                "pinned": False,
                "enabled": True,
            }
            user_docs.append(doc)
            data[user_id] = user_docs
            self._write(data)
            return doc

    def delete_document(self, user_id: str, doc_id: int) -> bool:
        with _lock:
            data = self._read()
            user_docs = data.get(user_id, [])
            new_docs = [d for d in user_docs if d["id"] != doc_id]
            if len(new_docs) == len(user_docs):
                return False
            data[user_id] = new_docs
            self._write(data)
            return True

    def update_document(self, user_id: str, doc_id: int, **kwargs) -> dict | None:
        with _lock:
            data = self._read()
            user_docs = data.get(user_id, [])
            for doc in user_docs:
                if doc["id"] == doc_id:
                    for key, value in kwargs.items():
                        if key in doc:
                            doc[key] = value
                    data[user_id] = user_docs
                    self._write(data)
                    return doc
            return None

    def toggle_pin(self, user_id: str, doc_id: int) -> dict | None:
        with _lock:
            data = self._read()
            user_docs = data.get(user_id, [])
            for doc in user_docs:
                if doc["id"] == doc_id:
                    doc["pinned"] = not doc["pinned"]
                    data[user_id] = user_docs
                    self._write(data)
                    return doc
            return None