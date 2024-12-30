import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from chat.router import router as chat_router
from Piggy.router import router as piggy_router

app = FastAPI(
    docs_url="/sugar-ai/docs",
)

app.include_router(chat_router, prefix="/sugar-ai/chat")
app.include_router(piggy_router, prefix="/sugar-ai/piggy")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)