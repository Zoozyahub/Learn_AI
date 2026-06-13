"""
Пока авторизации нет — текущий пользователь и диалог захардкожены.
Когда появится auth, заменить get_current_user_id на извлечение из токена,
а get_current_conversation_id — на conversation_id из запроса/пути.
Это единственное место, которое придётся трогать.
"""

from agent.database import DEFAULT_USER_ID, DEFAULT_CONVERSATION_ID


def get_current_user_id() -> int:
    return DEFAULT_USER_ID


def get_current_conversation_id() -> int:
    return DEFAULT_CONVERSATION_ID


# Chroma-коллекции именуются строкой user_id — держим один источник правды.
def vector_user_id(user_id: int) -> str:
    return str(user_id)