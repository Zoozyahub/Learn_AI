from typing import TypedDict, List, Annotated
from agent.vector_database import search_documents_in_vector_database
import operator
from langgraph.graph import StateGraph
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI


class AgentState(TypedDict):
    user_id: str
    retrieved_docs: list[dict]
    user_query: str
    answer: str
    enabled_filenames: list[str]   # ← фильтр по активным файлам
    messages: Annotated[list[AnyMessage], operator.add]


class Agent:
    def __init__(self, model="default"):
        if model == "default":
            self.model = ChatOpenAI(
                base_url="http://localhost:11434/v1",
                model="qwen3:4b",
                api_key='1',
                temperature=0.1
            )
        else:
            self.model = model

        self.graph = self._build_graph()
        self.conversation_history: List[AnyMessage] = []

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("search_in_vector_db", self.search_in_vector_db)
        graph.add_node("get_response", self.get_response)
        graph.set_entry_point("search_in_vector_db")
        graph.add_edge("search_in_vector_db", "get_response")
        return graph.compile()

    def search_in_vector_db(self, state: AgentState) -> dict:
        user_query = state['user_query']
        user_id = state['user_id']
        enabled_filenames = state.get('enabled_filenames', [])

        results = search_documents_in_vector_database(user_id, user_query, top_k=5)

        # Фильтруем по активным файлам, если список не пустой
        if enabled_filenames:
            results = [
                doc for doc in results
                if doc.metadata.get("filename") in enabled_filenames
            ]
            # Берём top-3 после фильтрации
            results = results[:3]

        docs_list = []
        for doc in results:
            docs_list.append({
                "Содержимое": doc.page_content,
                "Имя файла": doc.metadata.get("filename", "unknown"),
                "Страница": str(int(doc.metadata.get("page", 0)) + 1)
            })

        return {"retrieved_docs": docs_list}

    def get_response(self, state: AgentState) -> dict:
        context_str = ""
        for doc in state['retrieved_docs']:
            context_str += (
                f"Источник: {doc['Имя файла']}, "
                f"Страница: {doc['Страница']}, "
                f"Содержимое: {doc['Содержимое']}\n---\n"
            )

        messages_for_prompt = [
            SystemMessage(content=(
                "Ты - помощник, который помогает пользователю на основе предоставленных документов. "
                "Отвечай ТОЛЬКО на основе предоставленной информации. "
                "Обязательно указывай в ответе из какого файла и страницы взята информация. "
                "Если информации недостаточно, скажи пользователю, что не можешь ответить на вопрос."
            ))
        ]

        messages_for_prompt.extend(self.conversation_history)

        if context_str:
            current_query = f"Контекст из документов:\n{context_str}\n\nВопрос пользователя: {state['user_query']}"
        else:
            current_query = (
                f"Активные документы не содержат релевантной информации по вопросу.\n\n"
                f"Вопрос пользователя: {state['user_query']}"
            )

        messages_for_prompt.append(HumanMessage(content=current_query))

        response = self.model.invoke(messages_for_prompt)
        return {"answer": response.content, "messages": [response]}

    def invoke(self, user_id: str, query: str, enabled_filenames: list[str] = None) -> str:
        """Основной метод вызова агента"""
        initial_state = AgentState(
            user_id=user_id,
            user_query=query,
            retrieved_docs=[],
            answer="",
            enabled_filenames=enabled_filenames or [],
            messages=[]
        )

        result = self.graph.invoke(initial_state)

        self.conversation_history.append(HumanMessage(content=query))
        self.conversation_history.append(AIMessage(content=result["answer"]))

        return result["answer"]

    def get_history(self) -> List[AnyMessage]:
        return self.conversation_history

    def clear_history(self):
        self.conversation_history = []