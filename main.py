from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import os
import uuid
import json
import asyncio
from typing import Optional
from document_store import DocumentStore
from agent.agent import Agent

app = FastAPI(title="RAG Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Хранилище агентов по user_id (в продакшне заменить на Redis/DB)
agents: dict[str, Agent] = {}
# Хранилище документов
doc_store = DocumentStore()

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def get_agent(user_id: str) -> Agent:
    if user_id not in agents:
        agents[user_id] = Agent()
    return agents[user_id]


# ── Модели ──────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    user_id: str
    message: str
    enabled_doc_ids: list[int] = []  # Список ID активных документов


class DocumentToggle(BaseModel):
    enabled: bool


class DocumentRename(BaseModel):
    name: str


# ── Эндпоинты документов ────────────────────────────────────────────

@app.get("/api/documents")
async def get_documents(user_id: str = "default_user"):
    """Возвращает список документов пользователя"""
    return doc_store.get_documents(user_id)


@app.post("/api/documents/upload")
async def upload_documents(
    user_id: str = "default_user",
    files: list[UploadFile] = File(...)
):
    """Загружает файлы, сохраняет на диск и добавляет в векторную БД"""
    from agent.vector_database import load_documents_from_disk, add_documents_to_vector_database

    uploaded = []
    file_paths = []

    for file in files:
        # Сохраняем файл на диск
        file_id = str(uuid.uuid4())
        ext = os.path.splitext(file.filename)[1].lower()
        safe_name = f"{file_id}{ext}"
        file_path = os.path.join(UPLOAD_DIR, safe_name)

        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        # Добавляем в store
        doc_record = doc_store.add_document(
            user_id=user_id,
            original_name=file.filename,
            file_path=file_path,
            file_size=len(content)
        )
        uploaded.append(doc_record)
        file_paths.append(file_path)

    # Индексируем в векторную БД
    try:
        documents = load_documents_from_disk(file_paths)
        add_documents_to_vector_database(user_id, documents)
    except Exception as e:
        # Если индексация упала — возвращаем ошибку, но файлы уже сохранены
        raise HTTPException(status_code=500, detail=f"Ошибка индексации: {str(e)}")

    return {"uploaded": uploaded}


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: int, user_id: str = "default_user"):
    """Удаляет документ из store (из векторной БД удаление по имени файла — опционально)"""
    success = doc_store.delete_document(user_id, doc_id)
    if not success:
        raise HTTPException(status_code=404, detail="Документ не найден")
    return {"deleted": doc_id}


@app.patch("/api/documents/{doc_id}/toggle")
async def toggle_document(doc_id: int, body: DocumentToggle, user_id: str = "default_user"):
    """Включает или выключает документ из RAG"""
    doc = doc_store.update_document(user_id, doc_id, enabled=body.enabled)
    if not doc:
        raise HTTPException(status_code=404, detail="Документ не найден")
    return doc


@app.patch("/api/documents/{doc_id}/rename")
async def rename_document(doc_id: int, body: DocumentRename, user_id: str = "default_user"):
    """Переименовывает документ"""
    doc = doc_store.update_document(user_id, doc_id, name=body.name)
    if not doc:
        raise HTTPException(status_code=404, detail="Документ не найден")
    return doc


@app.patch("/api/documents/{doc_id}/pin")
async def pin_document(doc_id: int, user_id: str = "default_user"):
    """Закрепляет/открепляет документ"""
    doc = doc_store.toggle_pin(user_id, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Документ не найден")
    return doc


# ── Чат ─────────────────────────────────────────────────────────────

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Стриминг ответа от LLM через SSE.
    enabled_doc_ids — список ID документов, которые нужно учитывать в RAG.
    """
    agent = get_agent(request.user_id)

    # Получаем имена активных файлов для фильтрации в векторной БД
    enabled_filenames = []
    if request.enabled_doc_ids:
        all_docs = doc_store.get_documents(request.user_id)
        enabled_filenames = [
            os.path.basename(d["file_path"])
            for d in all_docs
            if d["id"] in request.enabled_doc_ids
        ]

    async def stream_response():
        try:
            # Запускаем агента в отдельном потоке (синхронный код)
            loop = asyncio.get_event_loop()
            answer = await loop.run_in_executor(
                None,
                lambda: agent.invoke(request.user_id, request.message, enabled_filenames)
            )

            # Стримим ответ по словам для эффекта печатания
            words = answer.split(" ")
            for i, word in enumerate(words):
                chunk = word + (" " if i < len(words) - 1 else "")
                yield f"data: {json.dumps({'text': chunk, 'done': False})}\n\n"
                await asyncio.sleep(0.02)  # ~50 слов/сек

            yield f"data: {json.dumps({'text': '', 'done': True})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


@app.delete("/api/chat/history")
async def clear_history(user_id: str = "default_user"):
    """Очищает историю диалога"""
    if user_id in agents:
        agents[user_id].clear_history()
    return {"cleared": True}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)