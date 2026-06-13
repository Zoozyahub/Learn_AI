"""
RAG-агент (stateless).

Отличия от прежней версии:
  • Никакого self.conversation_history — история передаётся снаружи
    (грузится из таблицы messages перед каждым вызовом). Удалил сообщение
    в БД → его не будет в истории.
  • RAG только по активным документам (active_document_ids).
  • Если активных документов нет — отвечаем на общих знаниях, добавив
    в начало предупреждение (формирует роутер, не агент).
  • Источники возвращаются отдельной структурой (а не внутри текста):
    [{document_id, filename, page, score}], чтобы фронт отрисовал блок,
    а роутер записал их в message_sources.
  • Поддержка стриминга: stream_answer() — генератор токенов.

Граф LangGraph оставлен для совместимости (search → respond), но для
стриминга мы вызываем узлы напрямую, т.к. .stream() модели удобнее
дёргать вне графа.
"""

from typing import TypedDict, Annotated, Iterator, Optional
import operator

from langgraph.graph import StateGraph
from langchain_core.messages import (
    AnyMessage, HumanMessage, SystemMessage, AIMessage,
)
from langchain_openai import ChatOpenAI

from agent.vector_database import search_with_scores
from agent.deps import vector_user_id


SYSTEM_PROMPT_RAG = (
    "Ты — помощник, который помогает пользователю на основе предоставленных "
    "документов. Отвечай, опираясь на предоставленную информацию. "
    "Если информации в документах недостаточно, честно сообщи об этом. "
    "Указывать источники в тексте не нужно — они показываются отдельно."
)

SYSTEM_PROMPT_NO_DOCS = (
    """Ты ассистент по обучению SunData и умеешь следующее:
    - Ответить на все вопросы по загруженным документам
    - Сгенерировать тест для проверки знаний
    - Подготовить карточки для запоминания фактов из текста
    Однако если ты видишь этот промпт значит пользователь задаёт вопрос без выбранных документов.
    Если вопрос не касается функционала описанного выше, то дай обычной ответ основываясь на своих знаниях."""
)


class AgentState(TypedDict, total=False):
    user_id: int
    conversation_id: int
    user_query: str
    active_document_ids: list[int]
    retrieved: list[dict]            # [{content, filename, page, document_id, score}]
    answer: str
    history: list[AnyMessage]


class Agent:
    def __init__(self, model: Optional[object] = None, top_k: int = 5):
        self.top_k = top_k
        if model is None:
            self.model = ChatOpenAI(
                base_url="http://localhost:11434/v1",
                model="gemma4:31b-cloud",
                api_key="1",
                temperature=0.1,
            )
        else:
            self.model = model
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("search", self._search_node)
        graph.add_node("respond", self._respond_node)
        graph.set_entry_point("search")
        graph.add_edge("search", "respond")
        return graph.compile()

    # ── Поиск (узел) ───────────────────────────────────────────
    def _search_node(self, state: AgentState) -> dict:
        retrieved = self.retrieve(
            user_id=state["user_id"],
            query=state["user_query"],
            active_document_ids=state.get("active_document_ids"),
        )
        return {"retrieved": retrieved}

    def retrieve(
        self,
        user_id: int,
        query: str,
        active_document_ids: Optional[list[int]],
    ) -> list[dict]:
        """
        Возвращает релевантные чанки. Если active_document_ids пуст/None —
        возвращаем []. Решение «искать или нет» принимает роутер заранее
        (передаёт None, если активных документов нет).
        """
        if not active_document_ids:
            return []
        pairs = search_with_scores(
            user_id=vector_user_id(user_id),
            query=query,
            top_k=self.top_k,
            active_document_ids=active_document_ids,
        )
        out: list[dict] = []
        for doc, score in pairs:
            page = doc.metadata.get("page", 0)
            out.append({
                "content": doc.page_content,
                "filename": doc.metadata.get("filename", "unknown"),
                "document_id": doc.metadata.get("document_id"),
                "page": int(page) + 1 if isinstance(page, (int, float)) else 1,
                "score": float(score),
            })
        return out

    # ── Сборка промпта ─────────────────────────────────────────
    def _build_messages(self, state: AgentState) -> list[AnyMessage]:
        retrieved = state.get("retrieved", [])
        has_docs = len(retrieved) > 0

        system = SYSTEM_PROMPT_RAG if has_docs else SYSTEM_PROMPT_NO_DOCS
        messages: list[AnyMessage] = [SystemMessage(content=system)]

        # История диалога (user/assistant из БД)
        messages.extend(state.get("history", []))

        if has_docs:
            context = "\n---\n".join(
                f"Источник: {d['filename']}, страница: {d['page']}\n{d['content']}"
                for d in retrieved
            )
            user_content = (
                f"Контекст из документов:\n{context}\n\n"
                f"Вопрос пользователя: {state['user_query']}"
            )
        else:
            user_content = state["user_query"]

        messages.append(HumanMessage(content=user_content))
        return messages

    # ── Ответ (узел, нестриминговый) ───────────────────────────
    def _respond_node(self, state: AgentState) -> dict:
        messages = self._build_messages(state)
        response = self.model.invoke(messages)
        return {"answer": response.content}

    def invoke(self, state: AgentState) -> dict:
        """Нестриминговый вызов: возвращает {answer, retrieved}."""
        result = self.graph.invoke(state)
        return {
            "answer": result.get("answer", ""),
            "retrieved": result.get("retrieved", []),
        }

    # ── Стриминг ───────────────────────────────────────────────
    def stream_answer(self, state: AgentState) -> Iterator[str]:
        """
        Генератор токенов ответа. Поиск выполняется до стрима
        (retrieved кладётся в state заранее роутером через retrieve()).
        """
        messages = self._build_messages(state)
        for chunk in self.model.stream(messages):
            token = getattr(chunk, "content", "") or ""
            if token:
                yield token