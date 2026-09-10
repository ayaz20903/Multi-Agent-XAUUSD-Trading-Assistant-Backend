from pathlib import Path

from langchain_chroma import Chroma


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHROMA_DIR = PROJECT_ROOT / "chroma_db"


def create_vector_store(chunks, embeddings, vectors):
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    ids = [f"chunk-{i}" for i in range(len(chunks))]

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    vector_store = Chroma(
        collection_name="gold_strategy",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    vector_store.add_texts(
        texts=texts,
        metadatas=metadatas,
        ids=ids,
        embeddings=vectors,
    )

    return vector_store


def load_vector_store(embeddings):
    return Chroma(
        collection_name="gold_strategy",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )