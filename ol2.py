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
        self.messages = list()
        
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
                "Страница": doc.metadata.get("page")
            })
        return {"retrieved_docs": docs_list}
    
    def get_response(self, state: AgentState) -> dict:
        # Формируем контекст из найденных документов
        final_str = ""
        for i in state['retrieved_docs']:
            final_str += f"Источник: {i['Имя файла']}, Страница: {i['Страница']}, Содержимое: {i['Содержимое']}\n---\n"
        
        # Создаем промпт с системным сообщением и контекстом
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="Ты - помощник, который помогает пользователю на основе предоставленных документов. Отвечай ТОЛЬКО на основе предоставленной информации."),
            *state['messages'][:-1],
            HumanMessage(content=f"Контекст из документов:\n{final_str}\n\nВопрос пользователя: {state['user_query']}")
        ])
        
        chain = prompt | self.model
        response = chain.invoke({})
        print('\n=== PROMPT ===\n')
        print(prompt)
        return {"answer": response.content, "messages": [response]}
    
    def invoke(self, user_id: str, query: str, state: AgentState) -> str:
        """Основной метод для вызова агента"""
        initial_state = AgentState(
            user_id=user_id,
            user_query=query,
            retrieved_docs=[],
            answer="",
            messages=self.messages.copy()
        )
        
        # Запускаем граф
        result = self.graph.invoke(initial_state)
        self.messages.append(HumanMessage(content=query))
        self.messages.extend(result["messages"])
        return result["answer"]

    def print_history(self):
        """Возвращает историю сообщений агента"""
        return self.messages

if __name__ == "__main__":
    # Пример использования
    agent = Agent()
 
    while True:
        user_id = "test_user"
        query = input("Введите ваш вопрос (или 'exit' для выхода): ")
        if query.lower() == 'exit':
            break
        answer = agent.invoke(user_id, query)
        print("Ответ агента:", answer)
    print("Итоговая история сообщений:", agent.print_history())
    