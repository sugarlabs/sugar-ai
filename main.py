import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from chat.router import router as chat_router
# from piggy.router import router as piggy_router

# Create a FastAPI application instance with custom documentation URL
app = FastAPI(
    docs_url="/sugar-ai/docs",
)

# Include the chat router with a specified prefix for endpoint paths
app.include_router(chat_router, prefix="/sugar-ai/chat")
# Include the piggy router with a specified prefix for endpoint paths (currently commented out)
# app.include_router(piggy_router, prefix="/sugar-ai/piggy")

# Add CORS middleware to allow cross-origin requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin
    allow_credentials=True,  # Allow sending of credentials (e.g., cookies)
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)