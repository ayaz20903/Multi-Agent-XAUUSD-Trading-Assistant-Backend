from gold_ai.loader import load_documents
from gold_ai.chunker import chunk_documents
from gold_ai.embeddings import create_embeddings, embed_documents
from gold_ai.vector_store import create_vector_store


def ingest_documents():
    documents = load_documents()

    print("Documents loaded:", len(documents))

    chunks = chunk_documents(documents)

    print("Chunks created:", len(chunks))

    embeddings = create_embeddings()

    vectors = embed_documents(
        chunks,
        embeddings
    )

    print("Embeddings created:", len(vectors))

    create_vector_store(
        chunks,
        embeddings,
        vectors
    )

    print("Vector store created successfully!")


if __name__ == "__main__":
    ingest_documents()