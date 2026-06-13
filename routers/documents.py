"""
Эндпоинты документов.

Поток загрузки (асинхронная индексация без Celery):
  1. POST /api/documents — сохраняем файл на диск, создаём строку Document
     со status="processing", сразу возвращаем её фронту.
  2. BackgroundTasks парсит/чанкует/эмбедит файл в фоне, по завершении
     ставит status="ready" (или "error", если что-то упало).
  3. Фронт поллит GET /api/documents, пока есть документы в "processing".

Все маршруты работают с текущим (захардкоженным) user_id/conversation_id.
"""

import os
import uuid
import shutil

from fastapi import APIRouter, UploadFile, File, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from agent.database import get_db, SessionLocal, Document, Chunk
from agent.deps import get_current_user_id, get_current_conversation_id, vector_user_id
from agent.vector_database import (
    index_document,
    delete_document_from_vector_database,
    EMBEDDING_MODEL_NAME,
)

router = APIRouter(prefix="/api/documents", tags=["documents"])

STORAGE_ROOT = "./storage"
ALLOWED_EXT = {".pdf", ".docx", ".doc"}


# ── Схемы ответов ──────────────────────────────────────────────
class DocumentOut(BaseModel):
    id: int
    name: str
    pinned: bool
    enabled: bool
    status: str          # processing | ready | error
    size: float

    @classmethod
    def from_orm_doc(cls, d: Document) -> "DocumentOut":
        return cls(
            id=d.document_id,
            name=d.public_name,
            pinned=d.is_pinned,
            enabled=d.is_active,
            status=d.status,
            size=d.size,
        )


class TogglePayload(BaseModel):
    enabled: bool | None = None
    pinned: bool | None = None


# ── Фоновая индексация ─────────────────────────────────────────
def _index_in_background(document_id: int, user_id: int, file_path: str, public_name: str):
    """
    Выполняется в фоне. Открывает СВОЮ сессию (сессия запроса уже закрыта).
    Любая ошибка → status="error", документ не зависает в processing.
    """
    db: Session = SessionLocal()
    try:
        chunk_records = index_document(
            user_id=vector_user_id(user_id),
            document_id=document_id,
            file_path=file_path,
            public_name=public_name,
        )

        # Сохраняем чанки в SQL (для будущих source-ссылок в чате)
        for rec in chunk_records:
            db.add(Chunk(
                document_id=document_id,
                chunk_index=rec["chunk_index"],
                content=rec["content"],
                token_count=rec["token_count"],
                page_number=rec["page_number"],
                chromadb_chunk_id=rec["chromadb_chunk_id"],
                embedding_model=EMBEDDING_MODEL_NAME,
            ))

        doc = db.get(Document, document_id)
        if doc:
            doc.status = "ready"
        db.commit()
    except Exception as e:
        db.rollback()
        doc = db.get(Document, document_id)
        if doc:
            doc.status = "error"
            db.commit()
        print(f"[index error] document_id={document_id}: {e}")
    finally:
        db.close()


# ── Список документов ──────────────────────────────────────────
@router.get("", response_model=list[DocumentOut])
def list_documents(
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
    conversation_id: int = Depends(get_current_conversation_id),
):
    stmt = (
        select(Document)
        .where(
            Document.conversation_id == conversation_id,
            Document.is_deleted == False,  # noqa: E712
        )
        .order_by(Document.created_at.desc())
    )
    docs = db.execute(stmt).scalars().all()
    return [DocumentOut.from_orm_doc(d) for d in docs]


# ── Загрузка (один или несколько файлов) ───────────────────────
@router.post("", response_model=list[DocumentOut])
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
    conversation_id: int = Depends(get_current_conversation_id),
):
    user_dir = os.path.join(STORAGE_ROOT, str(user_id))
    os.makedirs(user_dir, exist_ok=True)

    created: list[Document] = []
    for upload in files:
        ext = os.path.splitext(upload.filename)[1].lower()
        if ext not in ALLOWED_EXT:
            raise HTTPException(400, f"Неподдерживаемый формат: {upload.filename}")

        # Сохраняем под уникальным именем, оригинал — в public_name
        stored_path = os.path.join(user_dir, f"{uuid.uuid4().hex}{ext}")
        with open(stored_path, "wb") as out:
            shutil.copyfileobj(upload.file, out)
        size_mb = os.path.getsize(stored_path) / 1024 / 1024

        doc = Document(
            user_id=user_id,
            conversation_id=conversation_id,
            public_name=upload.filename,
            minio_path=stored_path,     # пока локальный путь
            size=size_mb,
            status="processing",
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)
        created.append(doc)

        # Запускаем индексацию в фоне (.doc не парсится лоадерами — упадёт в error,
        # что корректно отразится статусом)
        background_tasks.add_task(
            _index_in_background, doc.document_id, user_id, stored_path, upload.filename
        )

    return [DocumentOut.from_orm_doc(d) for d in created]


# ── Удаление ───────────────────────────────────────────────────
@router.delete("/{document_id}", status_code=204)
def delete_document(
    document_id: int,
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    doc = db.get(Document, document_id)
    if not doc or doc.is_deleted:
        raise HTTPException(404, "Документ не найден")

    # Чистим Chroma и файл; SQL помечаем удалённым (мягкое удаление)
    try:
        delete_document_from_vector_database(vector_user_id(user_id), document_id)
    except Exception as e:
        print(f"[chroma delete warn] {e}")
    if doc.minio_path and os.path.isfile(doc.minio_path):
        try:
            os.remove(doc.minio_path)
        except OSError:
            pass

    doc.is_deleted = True
    db.commit()
    return None


# ── Повторная индексация (для документов в статусе error) ──────
@router.post("/{document_id}/reindex", response_model=DocumentOut)
def reindex_document(
    document_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    doc = db.get(Document, document_id)
    if not doc or doc.is_deleted:
        raise HTTPException(404, "Документ не найден")
    if not doc.minio_path or not os.path.isfile(doc.minio_path):
        raise HTTPException(409, "Исходный файл недоступен — переиндексация невозможна")

    # Чистим возможные остатки от предыдущей неудачной попытки
    try:
        delete_document_from_vector_database(vector_user_id(user_id), document_id)
    except Exception as e:
        print(f"[chroma reindex cleanup warn] {e}")
    db.query(Chunk).filter(Chunk.document_id == document_id).delete()

    doc.status = "processing"
    db.commit()
    db.refresh(doc)

    background_tasks.add_task(
        _index_in_background, doc.document_id, user_id, doc.minio_path, doc.public_name
    )
    return DocumentOut.from_orm_doc(doc)


# ── Pin / enabled (один документ) ──────────────────────────────
@router.patch("/{document_id}", response_model=DocumentOut)
def patch_document(
    document_id: int,
    payload: TogglePayload,
    db: Session = Depends(get_db),
):
    doc = db.get(Document, document_id)
    if not doc or doc.is_deleted:
        raise HTTPException(404, "Документ не найден")
    if payload.pinned is not None:
        doc.is_pinned = payload.pinned
    if payload.enabled is not None:
        doc.is_active = payload.enabled
    db.commit()
    db.refresh(doc)
    return DocumentOut.from_orm_doc(doc)


# ── Выбрать/снять все (только незакреплённые, как на фронте) ────
class ToggleAllPayload(BaseModel):
    enabled: bool


@router.patch("", response_model=list[DocumentOut])
def toggle_all(
    payload: ToggleAllPayload,
    db: Session = Depends(get_db),
    conversation_id: int = Depends(get_current_conversation_id),
):
    stmt = select(Document).where(
        Document.conversation_id == conversation_id,
        Document.is_deleted == False,  # noqa: E712
        Document.is_pinned == False,   # noqa: E712
    )
    docs = db.execute(stmt).scalars().all()
    for d in docs:
        d.is_active = payload.enabled
    db.commit()

    # Возвращаем полный актуальный список
    all_stmt = (
        select(Document)
        .where(Document.conversation_id == conversation_id, Document.is_deleted == False)  # noqa: E712
        .order_by(Document.created_at.desc())
    )
    all_docs = db.execute(all_stmt).scalars().all()
    return [DocumentOut.from_orm_doc(d) for d in all_docs]