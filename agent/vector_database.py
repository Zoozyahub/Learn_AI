import os
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_embedder(model_name: str = "Qwen/Qwen3-Embedding-0.6B") -> SentenceTransformer:
    model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cuda"},  # Используем GPU для ускорения вычислений
        encode_kwargs={"normalize_embeddings": True}  # Нормализация эмбеддингов
    )
    return model

def create_or_open_vector_database_by_user(user_id: str = "default_user") -> Chroma:
    """
    Создает или открывает и возвращает экземпляр векторной базы данных Chroma для заданного пользователя.
    Параметры:
        user_id (str): Идентификатор пользователя для создания отдельной коллекции. По умолчанию "default_user".
    Возвращает:
        Chroma: Экземпляр векторной базы данных Chroma, связанный с указанным пользователем.
    """
    embedding_model = get_embedder()
    vector_store = Chroma(
        collection_name=f"user_{user_id}_collection",
        persist_directory=f"./vector_db/{user_id}",
        embedding_function=embedding_model
    )
    
    return vector_store

def delete_vector_database_by_user(user_id: str = "default_user") -> None:
    """
    Удаляет векторную базу данных Chroma для заданного пользователя.
    Параметры:
        user_id (str): Идентификатор пользователя для удаления соответствующей коллекции. По умолчанию "default_user".
    """
    vector_store = Chroma(
        collection_name=f"user_{user_id}_collection",
        persist_directory=f"./vector_db/{user_id}",
        embedding_function=get_embedder()
    )
    vector_store.delete_collection()
    
def load_documents_from_disk(file_paths: list[str]) -> list[Document]:
    """
    Загружает документы с диска из указанных путей.
    Параметры:
        file_paths (list[str]): Список путей к файлам для загрузки.
    Возвращает:
        list[Document]: Список загруженных документов.
    """
    documents = []
    for file_path in file_paths:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Файл не найден: {file_path}")
    
        ext = os.path.splitext(file_path)[1].lower()
        file_name = os.path.basename(file_path)
        if ext == ".pdf":
            loader = PyMuPDFLoader(file_path)
            data = loader.load()
        elif ext == ".docx":
            loader = Docx2txtLoader(file_path)
            data = loader.load()
        else:
            raise ValueError(f"Неподдерживаемый формат файла: {ext}")
        documents.extend(data)
    return documents
    

def add_documents_to_vector_database(user_id: str, documents: list[Document]) -> None:
    """
    Добавляет документы в векторную базу данных Chroma для заданного пользователя.
    Метаданные каждого Document сохраняются в чанках.
    """
    # 1. Обогатим метаданные: добавим удобочитаемое имя файла
    for doc in documents:
        source = doc.metadata.get("source", "")
        if source:
            doc.metadata["filename"] = os.path.basename(source)
        else:
            doc.metadata["filename"] = "unknown"

    # 2. Разбиваем ВСЕ документы сразу — LangChain сохранит metadata
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        length_function=len
    )
    chunks = splitter.split_documents(documents)

    print(f"Создано {len(chunks)} чанков из {len(documents)} документов.")

    # 3. Добавляем в существующую Chroma-базу
    vector_store = create_or_open_vector_database_by_user(user_id)
    vector_store.add_documents(chunks)


def search_documents_in_vector_database(user_id: str, query: str, top_k: int = 5) -> list[Document]:
    """
    Выполняет поиск по векторной базе данных Chroma для заданного пользователя.
    Параметры:
        user_id (str): Идентификатор пользователя для поиска в соответствующей коллекции.
        query (str): Запрос для поиска.
        top_k (int): Количество возвращаемых результатов. По умолчанию 5.
    Возвращает:
        list[Document]: Список найденных документов.
    """
    vector_store = create_or_open_vector_database_by_user(user_id)
    results = vector_store.similarity_search(query, k=top_k)
    return results

if __name__ == "__main__":
    # # тестирование загрузчика 
    # documents = load_documents_from_disk(["C:\\Users\\kuzoy\\Downloads\\UI_ Комплексный справочник по теоретическим основам от юзабилити до анализа путей пользователей.pdf"])
    # print(f"Загружено {len(documents)} документов.")
    # for doc in documents:
    #     print(doc.metadata.get("source").split('\\')[-1])  # Печать первых 500 символов каждого документа
    #     print(doc.page_content[:200])
    #     print("-----")
    
    
    # # теситрования создания базы и добавления документов
    # user_id = "test_user"
    # documents = load_documents_from_disk([
    #     "C:\\Users\\kuzoy\\Downloads\\UI_ Комплексный справочник по теоретическим основам от юзабилити до анализа путей пользователей.pdf",
    # ])
    # add_documents_to_vector_database(user_id, documents)
    # print(f"Документы добавлены в векторную базу данных для пользователя '{user_id}'.")
    
    
    # # тестирования удаления базы
    # delete_vector_database_by_user("test_user")
    # print("Векторная база данных для пользователя 'test_user' удалена.")
    
    
    # Тестирования поиска
    # Вывод сколько всего чанков в базе
    # print(f"Всего чанков в базе для пользователя 'test_user': {vector_store.count_documents()}")
    
    user_id = "test_user"
    query = "Что такое Customer Journey Mapping?"
    results = search_documents_in_vector_database(user_id, query, top_k=3)
    print(f"Результаты поиска для пользователя '{user_id}' по запросу '{query}':")
    for i, doc in enumerate(results):
        print(f"Результат {i+1}:")
        print("Источник:", doc.metadata.get("filename"), "Страница:", doc.metadata.get("page"))
        print("Содержимое:", doc.page_content)  # Печать первых 500 символов найденного документа
        print("-----")
    