import logging
import os
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from gold_ai.api.routes import router

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=1)


@asynccontextmanager
async def lifespan(application: FastAPI):
    from gold_ai.startup import ensure_vector_store
    import asyncio

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(_executor, ensure_vector_store)
    yield


app = FastAPI(title="Gold AI API", version="0.1.0", lifespan=lifespan)

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
