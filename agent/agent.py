from typing import TypedDict, List, Annotated
from vector_database import search_documents_in_vector_database
import operator
from langgraph.graph import StateGraph
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI


class AgentState(TypedDict):
    """
    Определяет состояние графа
    
    Атрибуты:
        user_id: ID пользователя
        retrieved_docs: Найденные документы из векторной БД
        user_query: Вопрос пользователя
        answer: Ответ LLM
        messages: История сообщений для контекста
    """
    user_id: str
    retrieved_docs: list[dict]
    user_query: str
    answer: str
    messages: Annotated[list[AnyMessage], operator.add]


class Agent:
    def __init__(self, model="default"):
        if model == "default":
            self.model = ChatOpenAI(
                base_url="http://localhost:11434/v1",
                model="qwen3:8b",
                api_key='1',
                temperature=0.1
            )
        else:
            self.model = model
        
        self.graph = self._build_graph()
        # Храним историю отдельно - только пары user/assistant
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
        
        results = search_documents_in_vector_database(user_id, user_query, top_k=3)
        
        docs_list = []
        for doc in results:
            docs_list.append({
                "Содержимое": doc.page_content,
                "Имя файла": doc.metadata.get("filename"),
                "Страница": str(int(doc.metadata.get("page")) + 1)
            })
        
        return {"retrieved_docs": docs_list}
    
    def get_response(self, state: AgentState) -> dict:
        # Формируем контекст из найденных документов
        context_str = ""
        for doc in state['retrieved_docs']:
            context_str += f"Источник: {doc['Имя файла']}, Страница: {doc['Страница']}, Содержимое: {doc['Содержимое']}\n---\n"
        
        # Создаем список сообщений для промпта
        messages_for_prompt = [
            SystemMessage(content="""Ты - помощник, который помогает пользователю на основе предоставленных документов. Отвечай ТОЛЬКО на основе предоставленной информации.
                          Обязательно указывай в ответе из какого файла и страницы взята информация. Если информации недостаточно, скажи пользователю, что не можешь ответить на вопрос.""")
        ]
        
        # Добавляем историю диалога (только user/assistant сообщения)
        messages_for_prompt.extend(self.conversation_history)
        
        
        # Добавляем текущий вопрос с контекстом
        current_query = f"Контекст из документов:\n{context_str}\n\nВопрос пользователя: {state['user_query']}"
        messages_for_prompt.append(HumanMessage(content=current_query))
        
        # Получаем ответ от модели
        print('\n=== PROMPT ===\n')
        print(messages_for_prompt)
        print('\n=== END OF PROMPT ===\n')
        
        response = self.model.invoke(messages_for_prompt)
        
        # Возвращаем ответ (НЕ добавляем в messages здесь)
        return {"answer": response.content, "messages": [response]}
    
    def invoke(self, user_id: str, query: str) -> str:
        """Основной метод для вызова агента"""
        initial_state = AgentState(
            user_id=user_id,
            user_query=query,
            retrieved_docs=[],
            answer="",
            messages=[]  # Пустой список для state
        )
        
        # Запускаем граф
        result = self.graph.invoke(initial_state)
        
        # Сохраняем в историю ТОЛЬКО пару вопрос-ответ
        self.conversation_history.append(HumanMessage(content=query))
        self.conversation_history.append(AIMessage(content=result["answer"]))
        
        return result["answer"]
    
    def get_history(self) -> List[AnyMessage]:
        """Возвращает чистую историю диалога"""
        return self.conversation_history
    
    def print_history(self):
        """Печатает историю в читаемом формате"""
        print("\n=== ИСТОРИЯ ДИАЛОГА ===")
        for msg in self.conversation_history:
            role = "Пользователь" if isinstance(msg, HumanMessage) else "Ассистент"
            print(f"\n{role}: {msg.content[:200]}{'...' if len(msg.content) > 200 else ''}")
        print("\n" + "="*50 + "\n")
    
    def clear_history(self):
        """Очищает историю диалога"""
        self.conversation_history = []


if __name__ == "__main__":
    # Пример использования
    agent = Agent()
    
    while True:
        user_id = "test_user"
        query = input("\nВведите ваш вопрос (или 'exit' для выхода, 'history' для просмотра истории): ")
        
        if query.lower() == 'exit':
            break
        
        if query.lower() == 'history':
            agent.print_history()
            continue
        
        if query.lower() == 'clear':
            agent.clear_history()
            print("История очищена!")
            continue
        
        answer = agent.invoke(user_id, query)
        print(f"\nОтвет агента: {answer}")