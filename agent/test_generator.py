"""
Агент генерации тестов (LangGraph).

Пайплайн (узлы графа):
  load_chunks  → подтягивает чанки активных документов из SQL (chunks + documents)
  sample       → случайно оставляет N чанков (по умолчанию 10)
  draft        → по каждому чанку генерит черновой вопрос (structured output)
  rerank       → из черновиков отбирает M лучших (по умолчанию 5), убирает
                 дубликаты, придумывает заголовок теста (structured output)
  assemble     → собирает итоговый JSON под схему MOCK_TEST_DATA и валидирует

Особенности:
  • active_document_ids НЕ обязателен в state: если его нет — агент сам берёт
    активные документы из SQL по conversation_id
    (is_active = True AND is_deleted = False).
  • Чанки берём из SQL (Chunk.content / page_number / document_id) + join на
    Document.public_name, чтобы вопрос ссылался на человекочитаемое имя файла.
  • Structured output: модель отвечает JSON, мы снимаем markdown-обёртку
    (```json ... ```) и валидируем в Pydantic-схему вручную. Это надёжнее
    with_structured_output поверх Ollama, где gemma корректно генерит JSON,
    но оборачивает его в ```-блок (см. _structured / _parse_json_object).
  • testId итогового теста = test_id из SQL-таблицы tests. Мы создаём строку Test
    ПОСЛЕ сборки (нужен test_data), поэтому в JSON он проставляется при сохранении
    в БД. В самом агенте testId — placeholder; финальный id отдаёт сохранение.

Запуск:  python -m agent.test_generator
"""

from __future__ import annotations

import json
import random
import re
import uuid
from typing import Optional, TypedDict, Literal

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import select

from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI

from agent.database import (
    SessionLocal, Chunk, Document, Test,
    DEFAULT_USER_ID, DEFAULT_CONVERSATION_ID,
)
from agent.deps import get_current_user_id, get_current_conversation_id


# ════════════════════════════════════════════════════════════════
# КОНФИГ
# ════════════════════════════════════════════════════════════════
SAMPLE_CHUNKS = 10          # сколько чанков уходит в генерацию (шаг 3)
FINAL_QUESTIONS = 5         # сколько вопросов остаётся после реранка (шаг 5)
SCHEMA_VERSION = "1.0.0"

# Квоты по типам вопросов. LLM сама любит делать всё multiple_choice,
# поэтому тип задаём принудительно на этапе генерации, а реранк отбирает
# нужное количество каждого типа.
DRAFT_QUOTA = {"single_choice": 6, "multiple_choice": 4}   # на генерации
FINAL_QUOTA = {"single_choice": 3, "multiple_choice": 2}   # после реранка

DEFAULT_SETTINGS = {
    "feedbackMode": "immediate",
    "difficulty": "medium",
    "questionCount": FINAL_QUESTIONS,
    "timeLimitSec": 480,
    "showTimer": True,
}


# ════════════════════════════════════════════════════════════════
# PYDANTIC-СХЕМЫ (для structured output И для валидации итога)
# ════════════════════════════════════════════════════════════════
class Option(BaseModel):
    id: str = Field(description="Буква варианта: a, b, c, d, (e)")
    text: str = Field(description="Текст варианта ответа")
    isCorrect: bool = Field(description="Правильный ли это вариант")
    explanation: str = Field(
        description="Пояснение, ПОЧЕМУ вариант верный или неверный. "
                    "Обязательно и для верных, и для неверных."
    )


class Question(BaseModel):
    id: str = Field(description="Идентификатор вопроса, напр. q1")
    type: Literal["single_choice", "multiple_choice"]
    stem: str = Field(description="Формулировка вопроса")
    hints: list[str] = Field(
        default_factory=list,
        description="1-2 подсказки, наводящие на ответ, но не раскрывающие его",
    )
    options: list[Option] = Field(description="Варианты ответа (4-5 штук)")
    generalExplanation: str = Field(
        description="Итоговое пояснение к вопросу в целом"
    )
    # Технические поля — на что ссылается вопрос (источник). В финальный JSON
    # фронта не обязаны идти, но полезны для message_sources / отладки.
    sourceDocumentId: Optional[int] = Field(
        default=None, description="document_id чанка-источника"
    )
    sourcePage: Optional[int] = Field(
        default=None, description="Номер страницы источника"
    )
    sourceFilename: Optional[str] = Field(
        default=None, description="Имя файла источника"
    )

    # ── Валидация бизнес-правил, которые LLM иногда нарушает ──
    def validate_business_rules(self) -> list[str]:
        errs: list[str] = []
        correct = [o for o in self.options if o.isCorrect]
        if len(self.options) < 2:
            errs.append(f"{self.id}: меньше 2 вариантов")
        if not correct:
            errs.append(f"{self.id}: нет правильного варианта")
        if self.type == "single_choice" and len(correct) != 1:
            errs.append(f"{self.id}: single_choice должен иметь ровно 1 верный")
        if self.type == "multiple_choice" and len(correct) < 1:
            errs.append(f"{self.id}: multiple_choice без верных вариантов")
        ids = [o.id for o in self.options]
        if len(ids) != len(set(ids)):
            errs.append(f"{self.id}: дублирующиеся id вариантов")
        for o in self.options:
            if not o.explanation.strip():
                errs.append(f"{self.id}/{o.id}: пустое пояснение")
        return errs


class RerankResult(BaseModel):
    """Результат шага реранка: заголовок + индексы выбранных вопросов."""
    title: str = Field(description="Краткий осмысленный заголовок теста")
    selected_indices: list[int] = Field(
        description=f"Индексы (0-based) {FINAL_QUESTIONS} лучших вопросов "
                    "из переданного списка, без повторов по смыслу"
    )


# ════════════════════════════════════════════════════════════════
# STATE
# ════════════════════════════════════════════════════════════════
class TestGenState(TypedDict, total=False):
    user_id: int
    conversation_id: int
    active_document_ids: list[int]      # опционально; если нет — берём из SQL
    chunks: list[dict]                  # [{content, page, document_id, filename}]
    sampled: list[dict]
    drafts: list[Question]
    title: str
    test_json: dict                     # итоговый результат


# ════════════════════════════════════════════════════════════════
# АГЕНТ
# ════════════════════════════════════════════════════════════════
class TestGeneratorAgent:
    def __init__(
        self,
        model: Optional[object] = None,
        draft_quota: Optional[dict] = None,
        final_quota: Optional[dict] = None,
        seed: Optional[int] = None,
    ):
        self.draft_quota = draft_quota or dict(DRAFT_QUOTA)
        self.final_quota = final_quota or dict(FINAL_QUOTA)
        # сколько чанков семплим = сумма квоты на генерации
        self.sample_chunks = sum(self.draft_quota.values())
        self.final_questions = sum(self.final_quota.values())
        self._rng = random.Random(seed)
        if model is None:
            self.model = ChatOpenAI(
                base_url="http://localhost:11434/v1",
                model="gemma4:31b-cloud",
                api_key="1",
                temperature=0.4,   # чуть выше, чем у RAG — нужно разнообразие
                # просим Ollama вернуть JSON-объект (мягкая страховка;
                # обёртку ``` всё равно снимаем в _parse_json_object)
                model_kwargs={"response_format": {"type": "json_object"}},
            )
        else:
            self.model = model
        self.graph = self._build_graph()

    # ── Граф ──────────────────────────────────────────────────
    def _build_graph(self):
        g = StateGraph(TestGenState)
        g.add_node("load_chunks", self._load_chunks_node)
        g.add_node("sample", self._sample_node)
        g.add_node("draft", self._draft_node)
        g.add_node("rerank", self._rerank_node)
        g.add_node("assemble", self._assemble_node)
        g.set_entry_point("load_chunks")
        g.add_edge("load_chunks", "sample")
        g.add_edge("sample", "draft")
        g.add_edge("draft", "rerank")
        g.add_edge("rerank", "assemble")
        return g.compile()

    # ── Хелпер structured output ──────────────────────────────
    def _structured(self, schema: type[BaseModel], messages):
        """
        Просим модель отдать ТОЛЬКО JSON, берём сырой ответ, срезаем
        markdown-обёртку (```json ... ```), валидируем в Pydantic-схему.

        Так надёжнее, чем with_structured_output поверх Ollama: gemma
        корректно генерит JSON, но оборачивает его в ```-блок, на котором
        json_mode-парсер LangChain спотыкается. Здесь мы этот блок снимаем
        сами. Это и есть «починка под схему» (шаг 7) — без второго вызова LLM.
        """
        # Подсказываем модели: только JSON, по схеме.
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
        if isinstance(raw, list):  # на случай блочного content
            raw = "".join(
                b.get("text", "") if isinstance(b, dict) else str(b) for b in raw
            )
        data = self._parse_json_object(str(raw))
        if schema is Question:
            data = self._normalize_question(data)
        return schema.model_validate(data)

    @staticmethod
    def _normalize_question(data: dict) -> dict:
        """
        gemma не держит точные имена полей схемы. Приводим её вывод к схеме:
          • синонимы ключей вопроса → stem;
          • синонимы внутри вариантов → text / isCorrect / explanation;
          • автогенерация id вариантов (a, b, c, …), если их нет.
        Работаем мягко: чего не нашли — оставляем как есть, остальное
        добьёт валидация Pydantic.
        """
        if not isinstance(data, dict):
            return data

        # --- поле вопроса (stem) ---
        if "stem" not in data:
            for k in ("question", "text", "prompt", "question_text", "stem_text"):
                if k in data and isinstance(data[k], str):
                    data["stem"] = data[k]
                    break

        # --- общее пояснение ---
        if "generalExplanation" not in data:
            for k in ("general_explanation", "generalExplanationText",
                      "summary", "overall_explanation"):
                if k in data:
                    data["generalExplanation"] = data[k]
                    break

        # --- подсказки ---
        if "hints" not in data:
            for k in ("hint", "tips", "clues"):
                if k in data:
                    v = data[k]
                    data["hints"] = v if isinstance(v, list) else [v]
                    break

        # --- варианты ---
        opts = data.get("options") or data.get("answers") or data.get("choices")
        if isinstance(opts, list):
            letters = "abcdefghij"
            norm_opts = []
            for i, o in enumerate(opts):
                if not isinstance(o, dict):
                    # вариант пришёл строкой — оборачиваем
                    o = {"text": str(o)}
                # id
                if not o.get("id"):
                    o["id"] = letters[i] if i < len(letters) else f"o{i}"
                # text
                if "text" not in o:
                    for k in ("answer", "value", "label", "option"):
                        if k in o:
                            o["text"] = o[k]
                            break
                # isCorrect
                if "isCorrect" not in o:
                    for k in ("is_correct", "correct", "isRight", "right"):
                        if k in o:
                            o["isCorrect"] = bool(o[k])
                            break
                    else:
                        o.setdefault("isCorrect", False)
                # explanation
                if "explanation" not in o:
                    for k in ("explain", "reason", "rationale", "feedback"):
                        if k in o:
                            o["explanation"] = o[k]
                            break
                    else:
                        o.setdefault("explanation", "")
                norm_opts.append(o)
            data["options"] = norm_opts

        return data

    @staticmethod
    def _parse_json_object(text: str) -> dict:
        """
        Достаёт JSON-объект из сырого текста модели:
          1) снимает ```json ... ``` / ``` ... ``` обёртку,
          2) если мусор по краям — вырезает от первой { до последней }.
        """
        s = text.strip()
        # снять ограждающий ```-блок
        fence = re.match(r"^```[a-zA-Z]*\s*(.*?)\s*```$", s, re.DOTALL)
        if fence:
            s = fence.group(1).strip()
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            # последний шанс: от первой { до последней }
            start, end = s.find("{"), s.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(s[start:end + 1])
            raise

    # ── Узел 1-2: загрузка чанков активных документов ─────────
    def _load_chunks_node(self, state: TestGenState) -> dict:
        chunks = self.load_active_chunks(
            conversation_id=state["conversation_id"],
            active_document_ids=state.get("active_document_ids"),
        )
        if not chunks:
            raise ValueError(
                "Нет чанков для генерации теста: нет активных документов "
                "или они без проиндексированных чанков."
            )
        return {"chunks": chunks}

    def load_active_chunks(
        self,
        conversation_id: int,
        active_document_ids: Optional[list[int]] = None,
    ) -> list[dict]:
        """
        Чанки активных документов из SQL.
        Если active_document_ids передан — фильтруем по нему.
        Иначе берём все активные документы диалога
        (is_active = True AND is_deleted = False).
        """
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

    # ── Узел 3: случайная выборка + назначение типов ──────────
    def _sample_node(self, state: TestGenState) -> dict:
        chunks = state["chunks"]
        k = min(self.sample_chunks, len(chunks))
        sampled = self._rng.sample(chunks, k)

        # Раздаём типы по квоте (масштабируем, если чанков меньше плана).
        plan_total = sum(self.draft_quota.values())
        n_single = round(self.draft_quota["single_choice"] / plan_total * k)
        forced = (["single_choice"] * n_single
                  + ["multiple_choice"] * (k - n_single))
        self._rng.shuffle(forced)

        sampled = [
            {**chunk, "forced_type": ftype}
            for chunk, ftype in zip(sampled, forced)
        ]
        return {"sampled": sampled}

    # ── Узел 4: черновые вопросы (по одному на чанк) ──────────
    def _draft_node(self, state: TestGenState) -> dict:
        drafts: list[Question] = []
        for i, chunk in enumerate(state["sampled"]):
            try:
                q = self._draft_one(chunk, idx=i)
            except Exception as e:
                print(f"[draft] чанк {i} пропущен: {e}")
                continue
            errs = q.validate_business_rules()
            if errs:
                print(f"[draft] чанк {i} отбракован: {errs}")
                continue
            drafts.append(q)
        if not drafts:
            raise RuntimeError("LLM не вернула ни одного валидного вопроса.")
        return {"drafts": drafts}

    def _draft_one(self, chunk: dict, idx: int) -> Question:
        forced_type = chunk.get("forced_type", "single_choice")
        if forced_type == "single_choice":
            type_rule = (
                'Тип вопроса — РОВНО "single_choice": ровно ОДИН вариант '
                "верный, остальные неверные. Дай 4 варианта."
            )
        else:
            type_rule = (
                'Тип вопроса — РОВНО "multiple_choice": НЕСКОЛЬКО вариантов '
                "верны (от 2 до 4). Дай 4-5 вариантов."
            )

        system = (
            "Ты — методист, составляющий тестовые вопросы по учебным материалам. "
            "По данному фрагменту текста придумай ОДИН проверочный вопрос. "
            f"{type_rule} К КАЖДОМУ варианту — "
            "пояснение (почему верный / почему неверный). Добавь 1-2 подсказки, "
            "наводящие, но не раскрывающие ответ, и итоговое пояснение. "
            "Опирайся ТОЛЬКО на текст фрагмента; не выдумывай фактов, которых в "
            "нём нет.\n\n"
            "Верни JSON-объект СТРОГО с такими именами полей (camelCase, "
            "именно так, без переименований):\n"
            "{\n"
            f'  "id": "q{idx + 1}",\n'
            f'  "type": "{forced_type}",\n'
            '  "stem": "<текст вопроса>",\n'
            '  "hints": ["<подсказка1>", "<подсказка2>"],\n'
            '  "options": [\n'
            '    {"id": "a", "text": "...", "isCorrect": true,  "explanation": "..."},\n'
            '    {"id": "b", "text": "...", "isCorrect": false, "explanation": "..."}\n'
            '  ],\n'
            '  "generalExplanation": "<итоговое пояснение>",\n'
            f'  "sourceDocumentId": {chunk["document_id"]},\n'
            f'  "sourcePage": {chunk["page"]},\n'
            f'  "sourceFilename": "{chunk["filename"]}"\n'
            "}\n"
            f'Поле "type" должно быть РОВНО "{forced_type}". '
            "Поле с текстом вопроса называется именно \"stem\". У каждого "
            "варианта обязательно есть \"id\" (буквы a, b, c, …)."
        )
        user = (
            f"Документ: {chunk['filename']} (страница {chunk['page']})\n\n"
            f"Фрагмент:\n{chunk['content']}"
        )
        q = self._structured(Question, [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ])
        # Тип навязываем жёстко — модель его часто игнорирует.
        q.type = forced_type
        # На случай, если модель проигнорировала источники/id — проставим сами.
        q.id = f"q{idx + 1}"
        q.sourceDocumentId = chunk["document_id"]
        q.sourcePage = chunk["page"]
        q.sourceFilename = chunk["filename"]
        return q

    # ── Узел 5: реранк по квотам типов + заголовок ────────────
    def _rerank_node(self, state: TestGenState) -> dict:
        drafts = state["drafts"]
        # фактически доступная квота (не больше, чем есть вопросов каждого типа)
        by_type = {t: [i for i, q in enumerate(drafts) if q.type == t]
                   for t in self.final_quota}
        target = {t: min(self.final_quota[t], len(by_type[t]))
                  for t in self.final_quota}

        listing = "\n\n".join(
            f"[{i}] ({q.type}) {q.stem}" for i, q in enumerate(drafts)
        )
        quota_str = ", ".join(f"{n} шт. типа {t}" for t, n in target.items())
        system = (
            f"Ты — редактор теста. Из списка вопросов выбери лучшие: {quota_str}. "
            "Бери самые содержательные и проверяющие понимание, БЕЗ смысловых "
            "повторов (если два вопроса про одно — оставь один). Верни индексы "
            "выбранных вопросов (0-based) и краткий осмысленный заголовок теста, "
            "отражающий тему материала."
        )
        user = f"Вопросы:\n{listing}"

        preferred: list[int] = []
        title = "Тест по материалам"
        try:
            res: RerankResult = self._structured(RerankResult, [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ])
            preferred = [i for i in res.selected_indices if 0 <= i < len(drafts)]
            preferred = list(dict.fromkeys(preferred))  # уникальные, порядок сохр.
            title = res.title.strip() or title
        except Exception as e:
            print(f"[rerank] structured output не сработал ({e}); "
                  "беру вопросы по квоте детерминированно.")

        # Добираем по квоте: сперва берём из preference, потом — оставшиеся
        # того же типа по порядку. Так 3+2 гарантированы, даже если LLM
        # выбрала не то / выбрала мало / вернула один тип.
        chosen: list[int] = []
        for t in self.final_quota:
            need = target[t]
            picked = [i for i in preferred if drafts[i].type == t][:need]
            if len(picked) < need:
                rest = [i for i in by_type[t] if i not in picked]
                picked += rest[: need - len(picked)]
            chosen.extend(picked)

        selected = [drafts[i] for i in chosen]
        # перенумеровываем q1..qN
        for j, q in enumerate(selected):
            q.id = f"q{j + 1}"
        return {"drafts": selected, "title": title}

    # ── Узел 6-7: сборка + валидация ──────────────────────────
    def _assemble_node(self, state: TestGenState) -> dict:
        questions = state["drafts"]

        all_errs: list[str] = []
        for q in questions:
            all_errs.extend(q.validate_business_rules())
        if all_errs:
            # Бизнес-правила нарушены — это не «кривой JSON» (структуру
            # гарантирует Pydantic), а логика. Чиним детерминированно:
            # выкидываем невалидные вопросы. LLM-ретрай тут не нужен.
            bad_ids = {e.split(":")[0].split("/")[0] for e in all_errs}
            print(f"[assemble] отбраковка вопросов {bad_ids}: {all_errs}")
            questions = [q for q in questions if q.id not in bad_ids]
            for j, q in enumerate(questions):
                q.id = f"q{j + 1}"
        if not questions:
            raise RuntimeError("После валидации не осталось ни одного вопроса.")

        settings = dict(DEFAULT_SETTINGS)
        settings["questionCount"] = len(questions)

        test_json = {
            "schemaVersion": SCHEMA_VERSION,
            # placeholder; реальный testId проставится при сохранении в SQL
            "testId": f"draft-{uuid.uuid4().hex[:8]}",
            "title": state.get("title", "Тест по материалам"),
            "settings": settings,
            "questions": [self._question_to_json(q) for q in questions],
        }
        return {"test_json": test_json}

    @staticmethod
    def _question_to_json(q: Question) -> dict:
        return {
            "id": q.id,
            "type": q.type,
            "stem": q.stem,
            "hints": q.hints,
            "options": [
                {
                    "id": o.id,
                    "text": o.text,
                    "isCorrect": o.isCorrect,
                    "explanation": o.explanation,
                }
                for o in q.options
            ],
            "generalExplanation": q.generalExplanation,
            "source": {
                "documentId": q.sourceDocumentId,
                "page": q.sourcePage,
                "filename": q.sourceFilename,
            },
        }

    # ── Публичный вызов ───────────────────────────────────────
    def generate(self, state: TestGenState) -> dict:
        result = self.graph.invoke(state)
        return result["test_json"]

    # ── Сохранение в SQL (testId = test_id из БД) ─────────────
    def save_test(self, user_id: int, conversation_id: int, test_json: dict) -> int:
        """
        Создаёт строку Test, проставляет реальный testId = test_id и
        возвращает его. test_data хранит весь JSON.
        """
        db = SessionLocal()
        try:
            test = Test(
                user_id=user_id,
                conversation_id=conversation_id,
                title=test_json["title"],
                test_data=test_json,
            )
            db.add(test)
            db.commit()
            db.refresh(test)
            test_json["testId"] = test.test_id
            test.test_data = dict(test_json)  # перезапишем с финальным testId
            db.commit()
            return test.test_id
        finally:
            db.close()


# ════════════════════════════════════════════════════════════════
# ТЕСТОВЫЙ ПРОГОН
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import json

    user_id = get_current_user_id()
    conversation_id = get_current_conversation_id()

    agent = TestGeneratorAgent(seed=42)

    # active_document_ids НЕ передаём — агент сам возьмёт активные документы
    # диалога из SQL (как и будет в проде, когда активность задаёт UI).
    state: TestGenState = {
        "user_id": user_id,
        "conversation_id": conversation_id,
    }

    print("⏳ Генерация теста по активным документам...")
    test_json = agent.generate(state)

    print("\n📦 Итоговый тест:\n")
    print(json.dumps(test_json, ensure_ascii=False, indent=2))

    # Раскомментируй, чтобы сохранить тест в SQL и получить реальный testId:
    # test_id = agent.save_test(user_id, conversation_id, test_json)
    # print(f"\n💾 Сохранено в tests, test_id = {test_id}")