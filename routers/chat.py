"""
Эндпоинты чата.

Поток POST /api/chat (SSE-стриминг):
  1. Грузим историю диалога из messages (только не удалённые, по порядку).
  2. Определяем активные документы (is_active, status="ready") текущего диалога.
  3. Сохраняем сообщение пользователя в messages.
  4. Если активных документов нет — первым событием шлём предупреждение
     ("notice") и отвечаем на общих знаниях; иначе делаем RAG-поиск.
  5. Стримим токены ответа (события "token").
  6. По завершении сохраняем ответ ассистента в messages и источники
     в message_sources, шлём финальное событие "done" с источниками.

Формат SSE: каждое событие — строка `data: {json}\n\n`.
Типы событий (поле "type"):
  notice  — предупреждение об отсутствии активных документов
  token   — кусок текста ответа
  done    — финал: {message_id, sources:[{document_id, name, page, score}]}
  error   — ошибка
"""

import json
import time

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from agent.database import (
    get_db, SessionLocal, Message, Document, Chunk, MessageSource, LLM,
)
from agent.deps import get_current_user_id, get_current_conversation_id
from agent.agent import Agent
from langchain_core.messages import HumanMessage, AIMessage

router = APIRouter(prefix="/api/chat", tags=["chat"])

# Агент создаётся один раз (модель/граф переиспользуются между запросами).
# Он stateless — историю передаём в каждом вызове.
_agent = Agent()

NO_DOCS_NOTICE = (
    "Сейчас нет активных источников, поэтому ответ не строится на ваших документах. "
    "Выберите документ слева, чтобы я отвечал по нему. "
    "А пока — вот что я могу сказать по этому вопросу, опираясь на общие знания:\n\n"
)


# ── Схемы ──────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str


class MessageOut(BaseModel):
    id: int
    type: str            # "human" | "ai"
    text: str
    sources: list[dict] = []   # [{document_id, name, page, score}]


# ── История диалога ────────────────────────────────────────────
def _load_history(db: Session, conversation_id: int):
    """LangChain-сообщения из БД (только не удалённые, по времени)."""
    rows = db.execute(
        select(Message)
        .where(
            Message.conversation_id == conversation_id,
            Message.is_deleted == False,  # noqa: E712
        )
        .order_by(Message.created_at)
    ).scalars().all()

    history = []
    for m in rows:
        if m.role == "user":
            history.append(HumanMessage(content=m.content))
        elif m.role == "assistant":
            history.append(AIMessage(content=m.content))
    return history


def _active_document_ids(db: Session, conversation_id: int) -> list[int]:
    rows = db.execute(
        select(Document.document_id).where(
            Document.conversation_id == conversation_id,
            Document.is_deleted == False,   # noqa: E712
            Document.is_active == True,      # noqa: E712
            Document.status == "ready",
        )
    ).scalars().all()
    return list(rows)


def _default_llm_id(db: Session) -> int | None:
    llm = db.execute(
        select(LLM).where(LLM.technical_name == "gpt-oss:20b-cloud")
    ).scalar_one_or_none()
    return llm.llm_id if llm else None


# ── Список сообщений (для загрузки диалога на фронте) ───────────
@router.get("/messages", response_model=list[MessageOut])
def get_messages(
    db: Session = Depends(get_db),
    conversation_id: int = Depends(get_current_conversation_id),
):
    rows = db.execute(
        select(Message)
        .where(
            Message.conversation_id == conversation_id,
            Message.is_deleted == False,  # noqa: E712
        )
        .order_by(Message.created_at)
    ).scalars().all()

    out: list[MessageOut] = []
    for m in rows:
        sources = []
        if m.role == "assistant":
            src_rows = db.execute(
                select(MessageSource, Document)
                .join(Document, MessageSource.document_id == Document.document_id)
                .where(MessageSource.message_id == m.message_id)
            ).all()
            for ms, doc in src_rows:
                sources.append({
                    "document_id": doc.document_id,
                    "name": doc.public_name,
                    "score": ms.similarity_score,
                })
        out.append(MessageOut(
            id=m.message_id,
            type="human" if m.role == "user" else "ai",
            text=m.content,
            sources=sources,
        ))
    return out


# ── Удаление сообщения (мягкое) ────────────────────────────────
@router.delete("/messages/{message_id}", status_code=204)
def delete_message(
    message_id: int,
    db: Session = Depends(get_db),
    conversation_id: int = Depends(get_current_conversation_id),
):
    m = db.get(Message, message_id)
    if not m or m.is_deleted or m.conversation_id != conversation_id:
        raise HTTPException(404, "Сообщение не найдено")
    m.is_deleted = True
    db.commit()
    return None


# ── Основной стриминговый эндпоинт ─────────────────────────────
@router.post("")
def chat(
    payload: ChatRequest,
    user_id: int = Depends(get_current_user_id),
    conversation_id: int = Depends(get_current_conversation_id),
):
    text = payload.message.strip()
    if not text:
        raise HTTPException(400, "Пустое сообщение")

    def event_stream():
        # Своя сессия на время стрима (request-scoped закроется раньше)
        db: Session = SessionLocal()
        try:
            history = _load_history(db, conversation_id)
            active_ids = _active_document_ids(db, conversation_id)
            llm_id = _default_llm_id(db)

            # 1. Сохраняем сообщение пользователя
            user_msg = Message(
                conversation_id=conversation_id,
                role="user",
                content=text,
            )
            db.add(user_msg)
            db.commit()
            db.refresh(user_msg)
            yield _sse({"type": "user_saved", "message_id": user_msg.message_id})

            # 2. Поиск (только по активным). Нет активных → предупреждение.
            has_docs = len(active_ids) > 0
            retrieved = []
            if has_docs:
                retrieved = _agent.retrieve(user_id, text, active_ids)
            else:
                yield _sse({"type": "notice", "text": NO_DOCS_NOTICE})

            state = {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "user_query": text,
                "active_document_ids": active_ids,
                "retrieved": retrieved,
                "history": history,
            }

            # 3. Стримим ответ
            t0 = time.time()
            full_answer_parts: list[str] = []
            for token in _agent.stream_answer(state):
                full_answer_parts.append(token)
                yield _sse({"type": "token", "text": token})

            full_answer = "".join(full_answer_parts)
            latency_ms = (time.time() - t0) * 1000

            # 4. Сохраняем ответ ассистента
            ai_msg = Message(
                conversation_id=conversation_id,
                role="assistant",
                content=full_answer,
                llm_id=llm_id,
                latency_ms=latency_ms,
            )
            db.add(ai_msg)
            db.commit()
            db.refresh(ai_msg)

            # 5. Источники: агрегируем по document_id (лучший score = минимальное
            #    расстояние). Берём страницы для отображения.
            best_by_doc: dict[int, dict] = {}
            for r in retrieved:
                doc_id = r.get("document_id")
                if doc_id is None:
                    continue
                cur = best_by_doc.get(doc_id)
                if cur is None or r["score"] < cur["score"]:
                    best_by_doc[doc_id] = r

            sources_out = []
            for doc_id, r in best_by_doc.items():
                # message_sources: одна строка на (message, document)
                db.add(MessageSource(
                    message_id=ai_msg.message_id,
                    document_id=doc_id,
                    similarity_score=r["score"],
                ))
                doc = db.get(Document, doc_id)
                sources_out.append({
                    "document_id": doc_id,
                    "name": doc.public_name if doc else r["filename"],
                    "page": r["page"],
                    "score": r["score"],
                })
            db.commit()

            yield _sse({
                "type": "done",
                "message_id": ai_msg.message_id,
                "sources": sources_out,
            })
        except Exception as e:
            db.rollback()
            print(f"[chat error] {e}")
            yield _sse({"type": "error", "text": "Произошла ошибка при генерации ответа."})
        finally:
            db.close()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # отключаем буферизацию nginx, если есть
        },
    )


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"