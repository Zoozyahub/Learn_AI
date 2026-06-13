from datetime import datetime
from typing import Optional

from sqlalchemy import (
    create_engine, String, Text, Integer, Boolean, Float,
    ForeignKey, DateTime, JSON, CheckConstraint,
    PrimaryKeyConstraint, Index, text, desc, Enum as SQLEnum, select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

# ================================================================
# НАСТРОЙКИ БД
# ================================================================
DATABASE_URL = "sqlite:///./app_database.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False}, echo=False)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


# ================================================================
# МОДЕЛИ (ТАБЛИЦЫ)
# ================================================================

# 1. SUBSCRIPTIONS & USERS
class Subscription(Base):
    __tablename__ = "subscriptions"
    subscription_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    cost: Mapped[int] = mapped_column(Integer, nullable=False)
    level: Mapped[int] = mapped_column(Integer, nullable=False)


class User(Base):
    __tablename__ = "users"
    user_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    surname: Mapped[str] = mapped_column(String, nullable=False)
    email: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    phone: Mapped[str] = mapped_column(String, nullable=False)
    password: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    role: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class UserSubscription(Base):
    __tablename__ = "user_subscriptions"
    user_subscription_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.user_id"), nullable=False)
    subscription_id: Mapped[int] = mapped_column(ForeignKey("subscriptions.subscription_id"), nullable=False)
    start_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    end_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    __table_args__ = (
        Index("user_subscriptions_user_id_end_date_index", "user_id", "end_date"),
    )


# 2. CONVERSATIONS
class Conversation(Base):
    __tablename__ = "conversations"
    conversation_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.user_id"), nullable=False)
    title: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_message_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    is_archived: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    __table_args__ = (
        Index("conversations_user_id_last_message_at_index", "user_id", desc("last_message_at")),
    )


# 3. LLMs & MESSAGES
class LLM(Base):
    __tablename__ = "llms"
    llm_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    public_name: Mapped[str] = mapped_column(String, nullable=False)
    technical_name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    min_level: Mapped[int] = mapped_column(Integer, nullable=False)


class Message(Base):
    __tablename__ = "messages"
    message_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.conversation_id"), nullable=False)
    llm_id: Mapped[Optional[int]] = mapped_column(ForeignKey("llms.llm_id"), nullable=True)
    role: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    reasoning: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    latency_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_deleted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    __table_args__ = (
        CheckConstraint(text("role IN ('user', 'assistant', 'system', 'tool')"), name="check_role_value"),
        Index("messages_conversation_id_created_at_index", "conversation_id", "created_at"),
    )


# 4. DOCUMENTS & CHUNKS (RAG)
class Document(Base):
    __tablename__ = "documents"
    document_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.user_id"), nullable=False)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.conversation_id"), nullable=False)
    public_name: Mapped[str] = mapped_column(String, nullable=False)
    minio_path: Mapped[str] = mapped_column(String, nullable=False)  # пока локальный путь
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_deleted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    size: Mapped[float] = mapped_column(Float, nullable=False)
    is_pinned: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    # status: processing → ready | error
    status: Mapped[str] = mapped_column(
        SQLEnum("processing", "ready", "error", name="document_status"),
        nullable=False,
        default="processing",
    )
    __table_args__ = (
        Index("documents_conversation_id_index", "conversation_id", sqlite_where=text("is_deleted = 0")),
    )


class Chunk(Base):
    __tablename__ = "chunks"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.document_id"), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    chromadb_chunk_id: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    embedding_model: Mapped[str] = mapped_column(String, nullable=False)
    __table_args__ = (
        Index("chunks_documents_id_chunk_index_index", "document_id", "chunk_index"),
    )


class MessageSource(Base):
    __tablename__ = "message_sources"
    message_id: Mapped[int] = mapped_column(ForeignKey("messages.message_id"), nullable=False)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.document_id"), nullable=False)
    similarity_score: Mapped[float] = mapped_column(Float, nullable=False)
    __table_args__ = (
        PrimaryKeyConstraint("message_id", "document_id"),
    )


# 5. TESTS
class Test(Base):
    __tablename__ = "tests"
    test_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.user_id"), nullable=False)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.conversation_id"), nullable=False)
    title: Mapped[str] = mapped_column(String, nullable=False)
    test_data: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    __table_args__ = (
        Index("tests_user_id_index", "user_id"),
    )


class TestAttempt(Base):
    __tablename__ = "test_attempts"
    test_attempt_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    test_id: Mapped[int] = mapped_column(ForeignKey("tests.test_id"), nullable=False)
    current_score: Mapped[int] = mapped_column(Integer, nullable=False)
    max_score: Mapped[int] = mapped_column(Integer, nullable=False)
    test_data: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    __table_args__ = (
        Index("test_attempts_test_id_index", "test_id"),
    )


# 6. FLASHCARDS
class FlashcardSet(Base):
    __tablename__ = "flashcard_sets"
    flashcard_set_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.user_id"), nullable=False)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.conversation_id"), nullable=False)
    title: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


class Flashcard(Base):
    __tablename__ = "flashcards"
    flashcard_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    flashcard_set_id: Mapped[int] = mapped_column(ForeignKey("flashcard_sets.flashcard_set_id"), nullable=False)
    front: Mapped[str] = mapped_column(Text, nullable=False)
    back: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    __table_args__ = (
        Index("flashcards_flashcard_set_id_index", "flashcard_set_id"),
    )


class FlashcardReview(Base):
    __tablename__ = "flashcard_reviews"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.user_id"), nullable=False)
    flashcard_id: Mapped[int] = mapped_column(ForeignKey("flashcards.flashcard_id"), nullable=False)
    is_success: Mapped[bool] = mapped_column(Boolean, nullable=False)
    reviewed_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    next_review_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    repetition_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    repetition_success_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    __table_args__ = (
        Index("flashcard_reviews_user_id_next_review_at_index", "user_id", "next_review_at"),
    )


# ================================================================
# СЕССИЯ (FastAPI dependency)
# ================================================================
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ================================================================
# ИНИЦИАЛИЗАЦИЯ + ДЕФОЛТНЫЕ user/conversation (пока нет авторизации)
# ================================================================
DEFAULT_USER_ID = 1
DEFAULT_CONVERSATION_ID = 1


def init_db():
    Base.metadata.create_all(bind=engine)

    # Создаём дефолтного пользователя и диалог, чтобы FK документов был валиден.
    db = SessionLocal()
    try:
        if db.get(User, DEFAULT_USER_ID) is None:
            db.add(User(
                user_id=DEFAULT_USER_ID,
                name="Default", surname="User",
                email="default@example.com", phone="—",
                password="—", role=0,
            ))
        if db.get(Conversation, DEFAULT_CONVERSATION_ID) is None:
            db.add(Conversation(
                conversation_id=DEFAULT_CONVERSATION_ID,
                user_id=DEFAULT_USER_ID,
                title="Default conversation",
            ))

        # Дефолтная LLM
        existing_llm = db.execute(
            select(LLM).where(LLM.technical_name == "gpt-oss:20b-cloud")
        ).scalar_one_or_none()
        if existing_llm is None:
            db.add(LLM(
                public_name="GPT OSS 20B",
                technical_name="gpt-oss:20b-cloud",
                description="GPT OSS 20B (cloud)",
                min_level=1,
            ))
        db.commit()
    finally:
        db.close()
    print("✅ База данных и все таблицы успешно созданы!")


if __name__ == "__main__":
    init_db()