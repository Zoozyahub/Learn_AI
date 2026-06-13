"""
Точка входа FastAPI.

Запуск:
    cd backend
    python database.py        # один раз: создать таблицы + дефолтные user/conversation
    uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agent.database import init_db
from routers import documents, chat

app = FastAPI(title="SunData API")

# Vite по умолчанию на 5173; добавь свой адрес, если другой.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router)
app.include_router(chat.router)


@app.on_event("startup")
def _startup():
    init_db()


@app.get("/api/health")
def health():
    return {"status": "ok"}