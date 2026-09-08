import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from gold_ai.api.routes import router

app = FastAPI(title="Gold AI API", version="0.1.0")

ALLOW_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
