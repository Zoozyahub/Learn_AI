"""
Векторная база (Chroma) + эмбеддер.

Ключевые изменения относительно прежней версии:
  • Модель эмбеддингов — BAAI/bge-m3 (мультиязычная, контекст до 8192 токенов).
  • Фикс бага с именем файла: при индексации каждый чанк получает
    metadata["document_id"] и metadata["filename"] = публичное имя.
    Раньше в source попадал случайный путь временного файла — и агент
    показывал "asff89...pdf" вместо "school.pdf".
  • Поиск умеет фильтровать по списку активных document_id
    (where={"document_id": {"$in": [...]}}) — RAG только по активным документам.
  • Эмбеддер кэшируется (singleton), чтобы не грузить модель на каждый запрос.
"""

import os
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Конфиг ─────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_DEVICE = "cuda"          # RTX 3060 Ti
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 250
VECTOR_DB_ROOT = "./vector_db"


# ── Эмбеддер (кэшируем — грузим модель один раз на процесс) ────
@lru_cache(maxsize=1)
def get_embedder(model_name: str = EMBEDDING_MODEL_NAME) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )


# ── Доступ к коллекции пользователя ────────────────────────────
def create_or_open_vector_database_by_user(user_id: str = "default_user") -> Chroma:
    return Chroma(
        collection_name=f"user_{user_id}_collection",
        persist_directory=f"{VECTOR_DB_ROOT}/{user_id}",
        embedding_function=get_embedder(),
    )


def delete_vector_database_by_user(user_id: str = "default_user") -> None:
    vector_store = create_or_open_vector_database_by_user(user_id)
    vector_store.delete_collection()


# ── Загрузка с диска ───────────────────────────────────────────
def load_documents_from_disk(file_paths: list[str]) -> list[Document]:
    documents: list[Document] = []
    for file_path in file_paths:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Файл не найден: {file_path}")
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            data = PyMuPDFLoader(file_path).load()
        elif ext == ".docx":
            data = Docx2txtLoader(file_path).load()
        else:
            raise ValueError(f"Неподдерживаемый формат файла: {ext}")
        documents.extend(data)
    return documents


# ── Индексация одного документа ────────────────────────────────
def index_document(
    user_id: str,
    document_id: int,
    file_path: str,
    public_name: str,
) -> list[dict]:
    """
    Парсит файл, режет на чанки, кладёт в Chroma.

    В метадату КАЖДОГО чанка пишем:
        document_id  — для фильтрации активных документов при поиске
        filename     — публичное имя (фикс бага с потерей имени)
        page         — номер страницы (как было)

    Возвращает список метаданных чанков для записи в SQL-таблицу chunks:
        [{chromadb_chunk_id, chunk_index, content, page_number, token_count}, ...]
    """
    raw_docs = load_documents_from_disk([file_path])

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(raw_docs)

    # Стабильные id чанков в Chroma: doc_{document_id}_chunk_{i}
    ids: list[str] = []
    chunk_records: list[dict] = []
    for i, chunk in enumerate(chunks):
        chunk_id = f"doc_{document_id}_chunk_{i}"
        ids.append(chunk_id)

        # Перезаписываем metadata осмысленными значениями
        page = chunk.metadata.get("page", 0)
        chunk.metadata = {
            "document_id": document_id,
            "filename": public_name,   # ← публичное имя, не путь временного файла
            "page": page,
        }

        chunk_records.append({
            "chromadb_chunk_id": chunk_id,
            "chunk_index": i,
            "content": chunk.page_content,
            "page_number": int(page) if isinstance(page, (int, float)) else 0,
            "token_count": len(chunk.page_content.split()),  # грубая оценка
        })

    vector_store = create_or_open_vector_database_by_user(user_id)
    vector_store.add_documents(chunks, ids=ids)
    return chunk_records


def delete_document_from_vector_database(user_id: str, document_id: int) -> None:
    """Удаляет все чанки документа из Chroma по метадате document_id."""
    vector_store = create_or_open_vector_database_by_user(user_id)
    vector_store.delete(where={"document_id": document_id})


# ── Поиск (с фильтром по активным документам) ──────────────────
def search_documents_in_vector_database(
    user_id: str,
    query: str,
    top_k: int = 5,
    active_document_ids: list[int] | None = None,
) -> list[Document]:
    """
    Поиск по коллекции пользователя.
    Если передан active_document_ids — ищем ТОЛЬКО по этим документам.
    Пустой список → возвращаем [] (нет активных документов — нечего искать).
    """
    vector_store = create_or_open_vector_database_by_user(user_id)

    where = None
    if active_document_ids is not None:
        if len(active_document_ids) == 0:
            return []
        # Chroma требует $in для списка; для одного значения тоже корректно
        where = {"document_id": {"$in": active_document_ids}}

    return vector_store.similarity_search(query, k=top_k, filter=where)


def search_with_scores(
    user_id: str,
    query: str,
    top_k: int = 5,
    active_document_ids: list[int] | None = None,
) -> list[tuple[Document, float]]:
    """
    То же, что search_documents_in_vector_database, но возвращает пары
    (Document, score). score — расстояние Chroma (меньше = ближе).
    Нужен для записи similarity_score в message_sources.
    """
    vector_store = create_or_open_vector_database_by_user(user_id)

    where = None
    if active_document_ids is not None:
        if len(active_document_ids) == 0:
            return []
        where = {"document_id": {"$in": active_document_ids}}

    return vector_store.similarity_search_with_score(query, k=top_k, filter=where)