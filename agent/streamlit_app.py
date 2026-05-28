import streamlit as st
import os
import tempfile
import atexit
from agent import Agent
from vector_database import (
    load_documents_from_disk,
    add_documents_to_vector_database,
    delete_vector_database_by_user,
    create_or_open_vector_database_by_user
)

# Константы
USER_ID = "streamlit_test"

# Инициализация session_state
if "agent" not in st.session_state:
    st.session_state.agent = Agent()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

if "processing" not in st.session_state:
    st.session_state.processing = False

# Функция очистки БД при закрытии приложения
def cleanup_on_exit():
    try:
        delete_vector_database_by_user(USER_ID)
        print(f"База данных для пользователя '{USER_ID}' удалена при закрытии приложения.")
    except Exception as e:
        print(f"Ошибка при удалении базы данных: {e}")

# Регистрируем функцию очистки
atexit.register(cleanup_on_exit)

# Настройка страницы
st.set_page_config(
    page_title="RAG Agent Demo",
    page_icon="🤖",
    layout="wide"
)

# Заголовок
st.title("🤖 RAG Kuptsova - Демонстрация")
st.markdown("---")

# Боковая панель с загрузкой файлов
with st.sidebar:
    st.header("📁 Управление документами")
    
    # Загрузка файлов
    uploaded_files = st.file_uploader(
        "Загрузите документы (PDF или DOCX)",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        key="file_uploader"
    )
    
    # Обработка загруженных файлов
    if uploaded_files:
        if st.button("📥 Добавить документы в базу", use_container_width=True):
            with st.spinner("Обрабатываю документы..."):
                try:
                    # Сохраняем файлы во временную директорию
                    temp_files = []
                    for uploaded_file in uploaded_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                            tmp_file.write(uploaded_file.getbuffer())
                            temp_files.append(tmp_file.name)
                    
                    # Загружаем документы
                    documents = load_documents_from_disk(temp_files)
                    
                    # Добавляем в базу
                    add_documents_to_vector_database(USER_ID, documents)
                    
                    # Сохраняем список загруженных файлов
                    for uploaded_file in uploaded_files:
                        if uploaded_file.name not in st.session_state.uploaded_files:
                            st.session_state.uploaded_files.append(uploaded_file.name)
                    
                    # Удаляем временные файлы
                    for temp_file in temp_files:
                        os.remove(temp_file)
                    
                    st.success(f"✅ Загружено {len(documents)} страниц из {len(uploaded_files)} файлов!")
                    
                except Exception as e:
                    st.error(f"❌ Ошибка при обработке файлов: {str(e)}")
    
    st.markdown("---")
    
    # Информация о загруженных файлах
    st.subheader("📚 Загруженные документы")
    if st.session_state.uploaded_files:
        for i, filename in enumerate(st.session_state.uploaded_files, 1):
            st.text(f"{i}. {filename}")
    else:
        st.info("Документы еще не загружены")
    
    st.markdown("---")
    
    # Кнопка очистки базы данных
    if st.button("🗑️ Очистить базу данных", use_container_width=True, type="secondary"):
        try:
            delete_vector_database_by_user(USER_ID)
            st.session_state.uploaded_files = []
            st.session_state.agent.clear_history()
            st.session_state.chat_history = []
            st.success("✅ База данных успешно очищена!")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Ошибка при очистке базы: {str(e)}")
    
    # Кнопка очистки истории чата
    if st.button("🔄 Очистить историю чата", use_container_width=True, type="secondary"):
        st.session_state.agent.clear_history()
        st.session_state.chat_history = []
        st.success("✅ История чата очищена!")
        st.rerun()
    
    st.markdown("---")
    
    # Информация о системе
    st.subheader("ℹ️ Информация")
    st.info(f"**User ID:** {USER_ID}\n\n**Модель:** Qwen3:4b")

# Основная область - чат
st.header("💬 Чат с агентом")

# Контейнер для истории чата
chat_container = st.container()

with chat_container:
    # Отображение истории чата
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Поле ввода сообщения
if prompt := st.chat_input("Задайте вопрос по загруженным документам..."):
    # Проверяем, загружены ли документы
    if not st.session_state.uploaded_files:
        st.warning("⚠️ Сначала загрузите документы в боковой панели!")
    else:
        # Добавляем сообщение пользователя в историю
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Отображаем сообщение пользователя
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Отображаем индикатор обработки
        with st.chat_message("assistant"):
            with st.spinner("🔍 Анализирую вопрос..."):
                try:
                    # Получаем ответ от агента
                    answer = st.session_state.agent.invoke(USER_ID, prompt)
                    
                    # Добавляем ответ в историю
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    
                    # Отображаем ответ
                    st.markdown(answer)
                    
                except Exception as e:
                    error_message = f"❌ Ошибка при обработке запроса: {str(e)}"
                    st.error(error_message)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_message})

# Дополнительная информация внизу страницы
st.markdown("---")
with st.expander("📖 Инструкция по использованию"):
    st.markdown("""
    ### Как использовать приложение:
    
    1. **Загрузка документов:**
       - В боковой панели выберите PDF или DOCX файлы
       - Нажмите "Добавить документы в базу"
       - Дождитесь завершения обработки
    
    2. **Общение с агентом:**
       - Введите вопрос в поле ввода внизу страницы
       - Агент проанализирует загруженные документы и даст ответ
       - История чата сохраняется в течение сессии
    
    3. **Управление данными:**
       - "Очистить базу данных" - удаляет все загруженные документы
       - "Очистить историю чата" - очищает переписку, но сохраняет документы
    
    4. **Важно:**
       - При закрытии приложения база данных автоматически удаляется
       - Агент отвечает только на основе загруженных документов
    """)

st.markdown("---")
st.caption("🚀 Powered by LangChain, LangGraph & Streamlit")