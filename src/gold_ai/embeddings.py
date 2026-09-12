import math

from google import genai
from google.genai import types
from langchain_core.embeddings import Embeddings

from gold_ai.config import GOOGLE_API_KEY


MODEL_NAME = "gemini-embedding-001"
OUTPUT_DIMENSIONALITY = 768


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))

    if norm == 0:
        return vector

    return [value / norm for value in vector]


class GeminiEmbeddings(Embeddings):
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        result = self.client.models.embed_content(
            model=MODEL_NAME,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=OUTPUT_DIMENSIONALITY,
            ),
        )

        return [
            _normalize(embedding.values)
            for embedding in result.embeddings
        ]

    def embed_query(self, text: str) -> list[float]:
        result = self.client.models.embed_content(
            model=MODEL_NAME,
            contents=text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=OUTPUT_DIMENSIONALITY,
            ),
        )

        return _normalize(result.embeddings[0].values)


def create_embeddings():
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is not set in the environment.")

    return GeminiEmbeddings(GOOGLE_API_KEY)


def embed_documents(chunks, embeddings):
    texts = [chunk.page_content for chunk in chunks]

    return embeddings.embed_documents(texts)