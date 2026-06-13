"""
Агент генерации карточек для запоминания (flashcards, LangGraph).

Проще генератора тестов: нет типов вопросов, вариантов и квот.
Пайплайн (узлы графа):
  load_chunks  → чанки активных документов из SQL (chunks + documents)
  sample       → случайно оставляет N чанков (по умолчанию 10)
  draft        → по каждому чанку генерит ОДНУ карточку (front/back)
  finalize     → убирает дубли, придумывает заголовок набора, собирает JSON

Особенности (как в test_generator):
  • active_document_ids НЕ обязателен в state: если его нет — берём активные
    документы диалога из SQL (is_active = True AND is_deleted = False).
  • Чанки из SQL (Chunk.content / page_number / document_id) + join на
    Document.public_name.
  • Structured output: модель отдаёт JSON, мы снимаем markdown-обёртку и
    валидируем в Pydantic вручную (надёжнее with_structured_output на Ollama).
  • В финальный JSON кладём id/front/back (как в моке) + source (filename/page).
  • id набора и карточек — placeholder'ы; реальные проставляются при сохранении
    в SQL (save_flashcards).

Запуск:  python -m agent.flashcard_generator
"""

from __future__ import annotations

import json
import random
import re
import uuid
from typing import Optional, TypedDict

from pydantic import BaseModel, Field

from sqlalchemy import select

from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI

from agent.database import (
    SessionLocal, Chunk, Document, FlashcardSet, Flashcard,
)
from agent.deps import get_current_user_id, get_current_conversation_id


# ════════════════════════════════════════════════════════════════
# КОНФИГ
# ════════════════════════════════════════════════════════════════
SAMPLE_CHUNKS = 10          # сколько чанков → столько карточек
SCHEMA_VERSION = "1.0.0"
DEFAULT_SETTINGS = {"shuffle": False}


# ════════════════════════════════════════════════════════════════
# PYDANTIC-СХЕМЫ
# ════════════════════════════════════════════════════════════════
class Card(BaseModel):
    front: str = Field(description="Лицевая сторона: краткий вопрос или термин")
    back: str = Field(description="Оборот: точный и краткий ответ/определение")
    # технические поля-источники (в финальный JSON идут в блок source)
    sourceDocumentId: Optional[int] = None
    sourcePage: Optional[int] = None
    sourceFilename: Optional[str] = None

    def is_valid(self) -> bool:
        return bool(self.front.strip()) and bool(self.back.strip())


class TitleResult(BaseModel):
    title: str = Field(description="Краткий осмысленный заголовок набора карточек")


# ════════════════════════════════════════════════════════════════
# STATE
# ════════════════════════════════════════════════════════════════
class FlashcardGenState(TypedDict, total=False):
    user_id: int
    conversation_id: int
    active_document_ids: list[int]      # опционально; иначе берём из SQL
    chunks: list[dict]
    sampled: list[dict]
    cards: list[Card]
    title: str
    flashcards_json: dict


# ════════════════════════════════════════════════════════════════
# АГЕНТ
# ════════════════════════════════════════════════════════════════
class FlashcardGeneratorAgent:
    def __init__(
        self,
        model: Optional[object] = None,
        sample_chunks: int = SAMPLE_CHUNKS,
        seed: Optional[int] = None,
    ):
        self.sample_chunks = sample_chunks
        self._rng = random.Random(seed)
        if model is None:
            self.model = ChatOpenAI(
                base_url="http://localhost:11434/v1",
                model="gemma4:31b-cloud",
                api_key="1",
                temperature=0.4,
                model_kwargs={"response_format": {"type": "json_object"}},
            )
        else:
            self.model = model
        self.graph = self._build_graph()

    # ── Граф ──────────────────────────────────────────────────
    def _build_graph(self):
        g = StateGraph(FlashcardGenState)
        g.add_node("load_chunks", self._load_chunks_node)
        g.add_node("sample", self._sample_node)
        g.add_node("draft", self._draft_node)
        g.add_node("finalize", self._finalize_node)
        g.set_entry_point("load_chunks")
        g.add_edge("load_chunks", "sample")
        g.add_edge("sample", "draft")
        g.add_edge("draft", "finalize")
        return g.compile()

    # ── Structured output (как в test_generator) ──────────────
    def _structured(self, schema: type[BaseModel], messages):
        msgs = list(messages)
        msgs.append({
            "role": "system",
            "content": (
                "Ответь СТРОГО валидным JSON-объектом по описанной схеме, "
                "без markdown, без ```-блоков, без пояснений до или после."
            ),
        })
        resp = self.model.invoke(msgs)
        raw = getattr(resp, "content", resp)
        if isinstance(raw, list):
            raw = "".join(
                b.get("text", "") if isinstance(b, dict) else str(b) for b in raw
            )
        data = self._parse_json_object(str(raw))
        if schema is Card:
            data = self._normalize_card(data)
        return schema.model_validate(data)

    @staticmethod
    def _parse_json_object(text: str) -> dict:
        s = text.strip()
        fence = re.match(r"^```[a-zA-Z]*\s*(.*?)\s*```$", s, re.DOTALL)
        if fence:
            s = fence.group(1).strip()
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            start, end = s.find("{"), s.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(s[start:end + 1])
            raise

    @staticmethod
    def _normalize_card(data: dict) -> dict:
        """gemma путает имена полей — приводим к front/back."""
        if not isinstance(data, dict):
            return data
        if "front" not in data:
            for k in ("question", "term", "q", "prompt", "face", "title"):
                if k in data and isinstance(data[k], str):
                    data["front"] = data[k]
                    break
        if "back" not in data:
            for k in ("answer", "definition", "a", "response", "reverse", "text"):
                if k in data and isinstance(data[k], str):
                    data["back"] = data[k]
                    break
        return data

    # ── Узел 1-2: чанки активных документов ───────────────────
    def _load_chunks_node(self, state: FlashcardGenState) -> dict:
        chunks = self.load_active_chunks(
            conversation_id=state["conversation_id"],
            active_document_ids=state.get("active_document_ids"),
        )
        if not chunks:
            raise ValueError(
                "Нет чанков для генерации карточек: нет активных документов "
                "или они без проиндексированных чанков."
            )
        return {"chunks": chunks}

    def load_active_chunks(
        self,
        conversation_id: int,
        active_document_ids: Optional[list[int]] = None,
    ) -> list[dict]:
        db = SessionLocal()
        try:
            stmt = (
                select(
                    Chunk.content,
                    Chunk.page_number,
                    Chunk.document_id,
                    Document.public_name,
                )
                .join(Document, Document.document_id == Chunk.document_id)
                .where(
                    Document.conversation_id == conversation_id,
                    Document.is_deleted.is_(False),
                )
            )
            if active_document_ids:
                stmt = stmt.where(Chunk.document_id.in_(active_document_ids))
            else:
                stmt = stmt.where(Document.is_active.is_(True))
            rows = db.execute(stmt).all()
        finally:
            db.close()

        return [
            {
                "content": content,
                "page": page_number,
                "document_id": document_id,
                "filename": public_name,
            }
            for content, page_number, document_id, public_name in rows
        ]

    # ── Узел 3: случайная выборка ─────────────────────────────
    def _sample_node(self, state: FlashcardGenState) -> dict:
        chunks = state["chunks"]
        k = min(self.sample_chunks, len(chunks))
        return {"sampled": self._rng.sample(chunks, k)}

    # ── Узел 4: карточка на чанк ───────────────────────────────
    def _draft_node(self, state: FlashcardGenState) -> dict:
        cards: list[Card] = []
        for i, chunk in enumerate(state["sampled"]):
            try:
                card = self._draft_one(chunk)
            except Exception as e:
                print(f"[draft] чанк {i} пропущен: {e}")
                continue
            if not card.is_valid():
                print(f"[draft] чанк {i} отбракован: пустой front/back")
                continue
            cards.append(card)
        if not cards:
            raise RuntimeError("LLM не вернула ни одной валидной карточки.")
        return {"cards": cards}

    def _draft_one(self, chunk: dict) -> Card:
        system = (
            "Ты — методист, делающий карточки для запоминания (flashcards) по "
            "учебным материалам. По данному фрагменту составь ОДНУ карточку. "
            "Лицевая сторона (front) — короткий вопрос или термин, проверяющий "
            "ОДИН важный факт из фрагмента. Оборот (back) — точный, краткий и "
            "самодостаточный ответ (1-2 предложения), который можно понять без "
            "контекста. Не делай front слишком общим, а back — слишком длинным. "
            "Опирайся ТОЛЬКО на текст фрагмента, не выдумывай фактов.\n\n"
            "Верни JSON-объект СТРОГО с такими полями:\n"
            "{\n"
            '  "front": "<вопрос или термин>",\n'
            '  "back": "<краткий точный ответ>",\n'
            f'  "sourceDocumentId": {chunk["document_id"]},\n'
            f'  "sourcePage": {chunk["page"]},\n'
            f'  "sourceFilename": "{chunk["filename"]}"\n'
            "}\n"
            'Поле вопроса называется именно "front", ответа — "back".'
        )
        user = (
            f"Документ: {chunk['filename']} (страница {chunk['page']})\n\n"
            f"Фрагмент:\n{chunk['content']}"
        )
        card = self._structured(Card, [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ])
        card.sourceDocumentId = chunk["document_id"]
        card.sourcePage = chunk["page"]
        card.sourceFilename = chunk["filename"]
        return card

    # ── Узел 5: дедуп + заголовок + сборка JSON ───────────────
    def _finalize_node(self, state: FlashcardGenState) -> dict:
        cards = self._dedupe(state["cards"])

        title = self._make_title(cards)

        flashcards_json = {
            "schemaVersion": SCHEMA_VERSION,
            "id": f"fcset_{uuid.uuid4().hex[:8]}",   # placeholder до сохранения
            "title": title,
            "settings": dict(DEFAULT_SETTINGS),
            "cards": [
                {
                    "id": f"fc_{i + 1}",
                    "front": c.front.strip(),
                    "back": c.back.strip(),
                    "source": {
                        "documentId": c.sourceDocumentId,
                        "page": c.sourcePage,
                        "filename": c.sourceFilename,
                    },
                }
                for i, c in enumerate(cards)
            ],
        }
        return {"title": title, "flashcards_json": flashcards_json}

    @staticmethod
    def _dedupe(cards: list[Card]) -> list[Card]:
        """Убираем карточки с одинаковым (нормализованным) front."""
        seen: set[str] = set()
        out: list[Card] = []
        for c in cards:
            key = re.sub(r"\s+", " ", c.front.strip().lower())
            if key in seen:
                continue
            seen.add(key)
            out.append(c)
        return out

    def _make_title(self, cards: list[Card]) -> str:
        """Отдельный лёгкий запрос: заголовок набора по лицевым сторонам."""
        listing = "\n".join(f"- {c.front}" for c in cards)
        system = (
            "Придумай краткий осмысленный заголовок набора карточек, "
            "отражающий тему. Верни JSON-объект: {\"title\": \"...\"}."
        )
        user = f"Карточки набора (лицевые стороны):\n{listing}"
        try:
            res: TitleResult = self._structured(TitleResult, [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ])
            return res.title.strip() or "Карточки по материалам"
        except Exception as e:
            print(f"[title] не удалось сгенерировать заголовок ({e}); "
                  "ставлю дефолтный.")
            return "Карточки по материалам"

    # ── Публичный вызов ───────────────────────────────────────
    def generate(self, state: FlashcardGenState) -> dict:
        result = self.graph.invoke(state)
        return result["flashcards_json"]

    # ── Сохранение в SQL (id = реальные из БД) ────────────────
    def save_flashcards(
        self, user_id: int, conversation_id: int, flashcards_json: dict
    ) -> int:
        """
        Создаёт FlashcardSet + Flashcard'ы, проставляет реальные id в JSON,
        возвращает flashcard_set_id.
        """
        db = SessionLocal()
        try:
            fset = FlashcardSet(
                user_id=user_id,
                conversation_id=conversation_id,
                title=flashcards_json["title"],
            )
            db.add(fset)
            db.commit()
            db.refresh(fset)

            flashcards_json["id"] = f"fcset_{fset.flashcard_set_id}"
            for card in flashcards_json["cards"]:
                row = Flashcard(
                    flashcard_set_id=fset.flashcard_set_id,
                    front=card["front"],
                    back=card["back"],
                )
                db.add(row)
                db.flush()                      # получить flashcard_id
                card["id"] = f"fc_{row.flashcard_id}"
            db.commit()
            return fset.flashcard_set_id
        finally:
            db.close()


# ════════════════════════════════════════════════════════════════
# ТЕСТОВЫЙ ПРОГОН
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    user_id = get_current_user_id()
    conversation_id = get_current_conversation_id()

    agent = FlashcardGeneratorAgent(seed=42)

    state: FlashcardGenState = {
        "user_id": user_id,
        "conversation_id": conversation_id,
    }

    print("⏳ Генерация карточек по активным документам...")
    flashcards_json = agent.generate(state)

    print("\n📦 Итоговый набор карточек:\n")
    print(json.dumps(flashcards_json, ensure_ascii=False, indent=2))

    # Раскомментируй, чтобы сохранить в SQL и получить реальные id:
    # set_id = agent.save_flashcards(user_id, conversation_id, flashcards_json)
    # print(f"\n💾 Сохранено, flashcard_set_id = {set_id}")